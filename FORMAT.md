# SLZ1 Frame Format Specification

**Version:** 2
**Byte order:** Little-endian unless otherwise noted ("BE" labels the
big-endian fields explicitly).

This document specifies the wire format the StreamLZ GPU encoder
produces and the GPU decoder accepts. The format is shared with the
pre-3.0 CPU codec at the frame and block level; this document covers
only the GPU subset.

The constants in this document are mirrored byte-for-byte in
`src/format/streamlz_constants.zig` (host side) and
`src/common/gpu_wire_format.cuh` / `src/common/gpu_huffman.cuh`
(device side). Discrepancies are bugs.

---

## Frame structure

```
┌───────────────────────────┐
│    Frame header           │  14-26 bytes
├───────────────────────────┤
│    Block 0                │  8-byte block header + block payload
├───────────────────────────┤
│    End mark               │  4 bytes (0x00000000)
├───────────────────────────┤
│    Content checksum       │  4 bytes (optional, XXH32; flag bit 1)
├───────────────────────────┤
│    Chunk-Merkle checksum  │  4 bytes (default-on, XXH32 root; flag bit 5)
└───────────────────────────┘
```

The GPU encoder always emits exactly one block. The decoder accepts
multi-block frames at the parser level but, in practice, every frame
the encoder produces today is single-block.

---

## Frame header

Minimum 14 bytes (no optional fields). Maximum 26 bytes (all optional
fields).

```
Offset  Size  Field
─────────────────────────────────────────
 0       4    Magic = 0x534C5A31 ("SLZ1" little-endian)
 4       1    Version. Must be 2; decoders reject any other value.
 5       1    Flags (see below)
 6       1    Codec ID. The GPU encoder always writes 1 (Fast).
 7       1    Internal level (1-6; see Level mapping below)
 8       1    Block-size log2 offset from 16 (so 2 → 256 KB)
 9       4    SC group size (IEEE 754 binary32 little-endian)
13       1    Reserved (must be 0)
14       8    Content size (int64 LE, present iff flag bit 0 set)
22       4    Dictionary ID (uint32 LE, present iff flag bit 3 set)
```

### Flags byte at offset 5

| Bit | Name                | Meaning |
|-----|---------------------|---------|
| 0   | ContentSizePresent  | 8-byte content size follows the fixed header. The GPU encoder always sets this bit. |
| 1   | ContentChecksum     | 4-byte XXH32 of the whole uncompressed content follows the end mark. Opt-in: the GPU encoder sets it when `content_checksum` is requested (CLI `--checksum`). |
| 2   | BlockChecksums      | Per-block XXH32 checksum follows each block payload. The GPU encoder does not set this bit. |
| 3   | DictionaryIdPresent | 4-byte dictionary ID follows the optional content size. The ID is opaque: it names a preset dictionary BOTH sides must already have (the wire never carries dictionary bytes). The decoder resolves the ID in its registry (`src/dict/dictionary.zig`) and rejects unknown IDs with `error.UnknownDictionary`. See "Preset dictionaries" below for how dictionary bytes extend the match window. |
| 4   | ParallelDecodeMeta  | Legacy CPU-codec sidecar; the GPU decoder skips it, the GPU encoder never sets it. |
| 5   | ChunkMerkleChecksum | 4-byte trailer after the end mark (after the bit-1 trailer when both are set): XXH32 over the concatenated per-chunk hashes (LE, chunk-index order) of the decompressed content, where each per-chunk hash is itself XXH32 over the concatenated XXH32s (LE) of the chunk's 1024-byte segments (last segment partial; chunk grid = the frame's effective chunk size). The two-level shape exists so GPUs can hash one thread per segment. A self-verification root, NOT a plain content hash - do not compare external XXH32(file) against it. The GPU encoder sets this by default since 2026-06-11; whether a given frame carries it is always announced by this flag bit, never assumed. |
| 6   | ChunkSizeTable      | A chunk-size table footer follows the end mark, BEFORE the bit-1/bit-5 trailers. See "Chunk-size table footer" below. Lets a decoder locate every chunk in parallel instead of walking the chunk chain. Default-on since 2026-06-12 (`--no-chunk-table` opts out); whether a given frame carries it is always announced by this flag bit. Decoders that do not know this bit decode the frame correctly by ignoring the footer. |
| 7   | Reserved            | Must be 0. |

### Codec ID at offset 6

| Value | Meaning |
|-------|---------|
| 1     | Fast codec. The only value the GPU encoder writes. |

Decoders reject any other value.

### Level mapping

The byte at offset 7 is the **internal** (engine) level, not the
user-facing level. User L1-L5 maps to internal 1, 2, 3, 5, 6
respectively (internal 4 is skipped because the parser variant it
would have selected consistently lost to internal 5 on every
workload we measured).

### Block size at offset 8

Stored as `log2(block_size) - 16`. The GPU encoder always writes 2
(256 KB block size). Decoders accept any value in `[0, 6]` (64 KB up
to 4 MB).

### sc_group_size at offset 9

A 32-bit IEEE 754 little-endian float in units of 256 KB chunks. The
GPU encoder picks adaptively from `{0.25, 0.5}`:

- **0.25** (64 KB sub-chunks) below the GPU saturation threshold -
  each decoder warp gets its own sub-chunk, more sub-chunks =
  more parallelism.
- **0.5** (128 KB sub-chunks) at or above saturation - larger
  sub-chunks compress better and the decoder is already saturated.

The saturation threshold is `sm_count × 48 warps × 128 KB`. On an
RTX 4060 Ti (34 SMs) this is ~209 MB.

Decoders must use the parsed value when computing chunk boundaries;
no hard-coded SC group size is correct.

### Content size at offset 14 (optional)

64-bit signed integer little-endian (negative values mean "unknown",
but the GPU encoder never writes a negative value). Capped at 4 GB
on the decoder side - frames claiming more return
`error.ContentSizeTooLarge`.

---

## Block header

Each block is preceded by an 8-byte header:

```
Offset  Size  Field
─────────────────────────────────────────
 0       4    Compressed payload size, packed with flag bits
 4       4    Decompressed payload size (uint32 LE)
```

The compressed-size field packs flags into the high bits:

| Bit  | Name                   | Meaning |
|------|------------------------|---------|
| 31   | Uncompressed           | Payload is raw bytes (no internal block header, no chunk headers). |
| 30   | ParallelDecodeMetadata | Legacy sidecar marker. The GPU encoder does not set this bit. Decoders skip such blocks (they contribute 0 to dst). |
| 0-29 | Size                   | Payload size in bytes. |

An entire 4-byte word of zeros at the block position is the **end
mark** - no decompressed-size field follows.

---

## End mark and content checksum

The block list ends with 4 zero bytes:

```
00 00 00 00
```

If `ChunkSizeTable` (bit 6) is set, the chunk-size table footer
follows the end mark first, then the trailers below.

### Chunk-size table footer (flag bit 6)

`num_chunks` 3-byte little-endian entries, one per chunk in frame
order. Each entry is the chunk's TOTAL wire size: its 2-byte internal
header, plus its 4-byte chunk header when present (compressed and
memset chunks have one; uncompressed chunks do not), plus its
payload. The exclusive prefix sum of the entries gives every chunk's
byte offset within the block payload, so a decoder can locate all
chunks with one contiguous read instead of walking the chunk chain.

The footer is self-locating: `num_chunks` derives from
`content_size` and the effective chunk size (both in the frame
header), and the trailer sizes are announced by flag bits, so its
position is `frame_end - trailers - 3 * num_chunks`. Two integrity
properties hold by construction and decoders should verify them: the
entries sum to the block's compressed size, and each entry agrees
with the size field in the chunk header it points to. The encoder
emits the footer only on compressed-body frames (never on
uncompressed-body fallbacks) and only when the effective chunk size
is at least 64 KB.

Trailers follow (after the table footer when present) in flag-bit
order:

1. If `ContentChecksum` (bit 1) is set, a 4-byte XXH32 of the whole
   uncompressed content. Opt-in via the encoder's `content_checksum`
   option (CLI `--checksum`).
2. If `ChunkMerkleChecksum` (bit 5) is set, the 4-byte Merkle root
   (default-on since 2026-06-11). Definition: each chunk's hash is
   XXH32 over the concatenated XXH32s (LE) of its 1024-byte segments
   (last segment partial); the root is XXH32 over the concatenated
   per-chunk hashes in chunk-index order. The chunk grid is the
   frame's effective chunk size, so the root depends on the `--sc`
   setting: it is a self-verification value, not a
   content-addressable file hash. The decoder recomputes it from its
   own output (on the GPU) and rejects the frame with
   `error.ChecksumMismatch` on disagreement.

---

## Block payload

Every compressed block payload begins with a 2-byte **internal block
header**, followed by a sequence of chunk-aligned payloads, optionally
followed by an **SC tail prefix table**.

### Internal block header (2 bytes)

Bytes 0 and 1 are not the same byte order as the rest of the frame -
this header is **inspected byte-by-byte**, not read as a u16.

```
Byte 0 (bit layout):
  [3:0]  Magic nibble. Must be 0x5.
  [4]    SelfContained flag. The GPU encoder always sets this bit.
  [5]    TwoPhase flag. The GPU encoder never sets this bit.
  [6]    RestartDecoder ("keyframe") flag. Always set on the first
         block of every frame; the encoder also sets it on every
         block when SelfContained is on.
  [7]    Uncompressed flag. Set when this block decoded as memcpy
         (no LZ tokens). For uncompressed chunks see the chunk
         header type-1 path below.

Byte 1 (bit layout):
  [6:0]  Decoder type. The GPU encoder writes 1 (Fast); decoders
         also accept 2 (Turbo) since the wire format is identical.
  [7]    UseChecksums flag. The GPU encoder never sets this bit.
```

### SelfContained mode

When SelfContained is set, the block payload is divided into chunks
that decode independently. The encoder guarantees no LZ
back-reference reaches across a chunk boundary, so each chunk can be
handed to its own decoder warp.

The first eight bytes of each chunk past the first arrive at the
decoder garbage - the decoder's initial `Copy64` only fires when the
chunk's `base_offset == 0`, and SC chunks 1..N-1 do not satisfy that.
The encoder repairs this by appending an **SC tail prefix table**
after the last chunk's payload:

```
[chunk-1 first 8 bytes][chunk-2 first 8 bytes]...[chunk-N-1 first 8 bytes]
```

Total table size: `(num_chunks - 1) × 8` bytes. The decoder
restores each chunk's first 8 bytes from this table (on the GPU via
`slzScPrefixApplyKernel` when checksum verification runs, and on the
host) after
parallel decode completes.

---

## Chunk header

Each chunk starts with a 4-byte little-endian header:

```
Bits   Field
─────────────────────────────────────────
 0-17  Compressed size minus 1 (0..262143; chunk_size_bits = 18)
18-19  Type: 0 = normal LZ, 1 = memset
20-31  Reserved (must be 0)
```

### Type 0 (normal)

Followed by `compressed_size` bytes of payload. The payload is one
or more sub-chunks back-to-back. There is no extra header between
the chunk header and the first sub-chunk.

### Type 1 (memset)

Followed by 1 byte: the fill value. The chunk decompresses to
`decompressed_size` copies of that byte. The encoder produces this
shape when every byte in the chunk is equal.

---

## Sub-chunk header

Within a chunk's payload, each sub-chunk starts with a 3-byte
**big-endian** header:

```
Bit   Field
─────────────────────────────────────────
23    LZ flag. 1 = sub-chunk is LZ-compressed (use the LZ decode
      pipeline); 0 = sub-chunk is raw bytes (just memcpy).
22-19 Decode mode (4 bits). Identifies the entropy-stream layout
      inside the LZ-compressed payload.
18-0  Compressed payload size (19 bits, up to 512 KB; the encoder
      never emits a sub-chunk larger than 128 KB).
```

The decode-mode nibble switches between sub-chunk variants. Today the
GPU encoder uses only two:

- **Mode = 0**: raw sub-chunk. The payload is uncompressed bytes.
- **Mode = 1**: LZ-compressed sub-chunk with per-stream entropy
  selection (see Sub-chunk payload below).

Other mode values are reserved.

---

## Sub-chunk payload (LZ-compressed)

An LZ-compressed sub-chunk's payload contains six byte streams,
concatenated in this order:

```
[8 raw init bytes]     present only on the first sub-chunk of the
                       frame; matches INITIAL_LITERAL_COPY_BYTES
[literals stream]      see below
[tokens stream]        see below
[off16 stream]         16-bit match offsets (split hi/lo when
                       entropy-coded)
[off32 stream]         match offsets that didn't fit in 16 bits
[length stream]        extended-length tail bytes
```

Each stream has its own per-stream header that picks raw vs Huffman.

### Literal and token streams

Both use the entropy-chunk header convention:

| First byte high nibble | Stream form |
|------------------------|-------------|
| 0                      | Raw memcpy: `[3-byte BE size][raw bytes]` |
| 4                      | Huffman-coded: `[5-byte non-compact header][Huffman body]` |
| 8 (high bit set)       | Compact memcpy (2-byte header, payload ≤ 0xFFF bytes). |

The 5-byte **non-compact header** layout used by chunk type 4 is:

```
byte 0:   [type:4 | dst_size_minus_1[17:14]:4]
bytes 1-4 (BE u32): [dst_size_minus_1[13:0]:14 | comp_size:18]
```

`dst_size_minus_1` reconstructs as `(top4 << 14) | low14`. The
compressed-payload size occupies the low 18 bits of the BE u32 at
bytes 1..4.

The GPU encoder only emits type 0 (raw) and type 4 (Huffman) for
literals and tokens. Any other chunk-type byte in a sub-chunk's
literal or token header is rejected at scan-walk time as an
unsupported stream shape.

### off16 stream

```
[2-byte LE count]   number of 16-bit offsets in this sub-chunk
                    (or 0xFFFF, see below)
[2 × count bytes]   the offsets, little-endian u16 pairs
```

When the count field equals `0xFFFF` (the **entropy marker**), the
off16 stream is split into hi/lo byte planes, each entropy-coded:

```
[2-byte 0xFFFF marker]
[hi-plane entropy chunk]   chunk_type = 4 (Huffman) when worthwhile,
                           chunk_type = 0 (raw bytes) otherwise.
[lo-plane entropy chunk]   same shape, separate decision.
```

Each plane's chunk uses the same 5-byte non-compact header as
literals + tokens (chunk_type=4) or the 3-byte raw header
(chunk_type=0).

The split decision is made per-plane: a plane uses Huffman only when
`5 + huff_body < 3 + count` (i.e., Huffman beats raw). The combined
split form is emitted only when the two-plane total beats the
unsplit `count + 2` raw form.

### off32 stream

The off32 header is a packed 3-byte field, plus optional u16 extras,
plus the offset payload itself:

```
[3-byte packed counts]
  bits 23..12  block-0 off32 count (12 bits; 4095 escapes)
  bits 11..0   block-1 off32 count (12 bits; 4095 escapes)
[optional 2-byte LE extra for block 0 count] iff packed-bits-23..12 == 4095
[optional 2-byte LE extra for block 1 count] iff packed-bits-11..0  == 4095
[off32 entries]
```

Each off32 entry is either 3 or 4 bytes. The high byte of the third
byte selects the form: values `< 0xC0` mean a 3-byte entry,
values `>= 0xC0` mean a 4-byte entry that encodes an offset wider
than 22 bits.

### length stream

Raw bytes, no header. Tokens that mark themselves as "extended
length" name a number of bytes to consume from this stream. The
extended-length marker threshold is 251: token-stream bytes with raw
value > 251 (i.e., 252-255) indicate the length spills into the
length stream.

---

## Preset dictionaries

When the frame header carries a dictionary ID (flag bit 3), the named
dictionary's bytes are logically prepended BELOW every sub-chunk's
output window: a match whose source address falls k bytes below the
sub-chunk's first output byte reads dictionary byte `dict_len - k`.
No token form changes - a dictionary reference is an ordinary match
whose distance exceeds the current position within the sub-chunk.
Rules:

- A match source may straddle the boundary (start in the dictionary,
  run into the sub-chunk's own output); per-byte sequential semantics
  apply unchanged. The GPU encoder never emits straddling sources,
  but decoders must accept them.
- Reaches below the dictionary itself (hostile frames) decode as
  0x00 bytes; they are not an error.
- The GPU encoder emits dictionary references only at off16-encodable
  distances (<= 65535), so positions in a second 64 KB LZ block never
  reference the dictionary. The off32 form's block-relative encoding
  also reaches dictionary space (`v` larger than the block's start
  offset) and decoders support it.
- The dictionary applies per sub-chunk: every sub-chunk's window
  starts with the full dictionary reachable, independent of its
  neighbors. The chunk-Merkle and content checksums are computed over
  the decompressed content only and are dictionary-independent.

---

## 32-stream BIL Huffman wire format

Chunk type 4 (Huffman) bodies use the bounded-interleaved (BIL)
layout. The body decomposes into:

```
[128 B weights]      256 4-bit code lengths, packed low-nibble first
[96 B sub-header]    32 × u24 LE per-stream byte sizes
[4 B K]              u32 LE interleaved-word count
                     K = min(words[s] for s in 0..31), with
                     words[s] = (size[s] + 3) / 4
[K × 128 B]          interleaved area: K rows × 32 streams × 4 bytes;
                     row w holds lane s's word w at byte offset
                     (w × 128 + s × 4)
[tail area]          per-stream bytes from word K onward, concatenated
                     at exclusive-prefix-sum-of-tail-sizes offsets
```

Why this layout: in the hot loop every warp lane refills the same
word index simultaneously. The interleaved area lets that refill be
**one coalesced 128-byte sector load** instead of 32 scattered
4-byte loads - measured ~12-13% kernel speedup over the prior
concatenated layout at ~0.01% header growth.

All 32 stream sizes are stored explicitly so each lane can compute
its own `words[s]` locally and warp-min-reduce to K without an
extra total-payload round trip.

Code lengths are limited to **11 bits**. The decode-side LUT is
1024 entries wide and indexed by 10 bits of the bit buffer; an
11-bit code can't be resolved by 10 bits and falls into an escape
entry that consumes one extra bit before re-indexing.

The encoder writes streams zero-padded up to a 4-byte BIL word
boundary so the decoder's u32 reads never tear across the
last-byte-of-stream / start-of-next-allocation boundary.

---

## Constants summary

| Constant                       | Value         | Source                         |
|--------------------------------|---------------|--------------------------------|
| Magic                          | `0x534C5A31`  | `SLZ_FRAME_MAGIC`              |
| Version                        | `2`           | `SLZ_FRAME_VERSION`            |
| Codec (Fast)                   | `1`           | `SLZ_CODEC_FAST_LZ`            |
| Min frame-header size          | `14`          | `SLZ_FRAME_MIN_HDR_SIZE`       |
| Chunk size                     | `262144`      | `SLZ_CHUNK_SIZE_BYTES` (256 KB)|
| Chunk size bits                | `18`          | `chunk_size_bits`              |
| Sub-chunk header bytes         | `3`           | `SUBCHUNK_HDR_BYTES`           |
| Sub-chunk LZ flag bit          | `0x800000`    | `SUBCHUNK_LZ_FLAG_BIT`         |
| Sub-chunk mode shift / mask    | `19` / `0xF`  | `SUBCHUNK_MODE_*`              |
| Sub-chunk comp-size mask       | `0x7FFFF`     | `SUBCHUNK_COMP_SIZE_MASK`      |
| LZ block size                  | `0x10000`     | `LZ_BLOCK_SIZE` (64 KB)        |
| Initial literal copy bytes     | `8`           | `INITIAL_LITERAL_COPY_BYTES`   |
| Initial recent offset          | `-8`          | `INITIAL_RECENT_OFFSET`        |
| Off16 entropy marker           | `0xFFFF`      | `OFF16_ENTROPY_MARKER`         |
| Off32 long-entry tag           | `0xC0`        | `OFF32_LONG_ENTRY_TAG`         |
| Off32 count-field bits         | `12`          | `OFF32_COUNT_FIELD_BITS`       |
| Extended-length threshold      | `251`         | `EXT_LENGTH_THRESHOLD`         |
| Huffman chunk type             | `4`           | `HUFF_CHUNK_TYPE`              |
| Huffman max code length        | `11`          | `HUFF_MAX_CODE_LEN`            |
| Huffman LUT entries            | `1024`        | `HUFF_LUT_ENTRIES`             |
| Huffman stream count           | `32`          | `HUFF_NUM_STREAMS`             |
| Huffman weights bytes          | `128`         | `HUFF_WEIGHTS_BYTES`           |
| Huffman sub-header bytes       | `96`          | `HUFF_SUBHEADER_BYTES`         |
| Huffman BIL K bytes            | `4`           | `HUFF_BIL_K_BYTES`             |
| Block-uncompressed flag        | `0x80000000`  | `block_uncompressed_flag`      |
| End mark                       | `0x00000000`  | `end_mark`                     |
| Block internal magic nibble    | `0x05`        | `SLZ_INT_BLOCK_MAGIC`          |
| Decoder ID (Fast / Turbo)      | `1` / `2`     | `SLZ_DECODER_FAST/TURBO`       |
| Safe space                     | `64`          | `safe_space` (decoder slack)   |

For the algorithmic reasoning behind the BIL layout, the parallel-
parse hot loop, and the warp-cooperative byte copies, see
[docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md).
