# SLZ1 Frame Format Specification

**Version:** 2
**Status:** Stable (backward-compatible changes only)
**Byte order:** Little-endian unless otherwise noted
**Notation:** `BSR(x)` denotes Bit Scan Reverse — the 0-based index of the most significant set bit, equivalent to `floor(log2(x))` for x > 0.

---

## Overview

SLZ1 is a framed container format for StreamLZ compressed data. It wraps one or more independently compressed blocks with a header describing the codec, level, and optional metadata. The frame format enables streaming compression/decompression of arbitrarily large data using a sliding window.

The design is similar in spirit to LZ4 frames and Zstandard frames: a magic number, a descriptor, a sequence of blocks, and an optional checksum.

---

## Frame Structure

```
┌──────────────────────────┐
│    Frame Header           │  14-26 bytes
├──────────────────────────┤
│    Block 0                │  8 bytes header + payload
├──────────────────────────┤
│    Block 1                │  8 bytes header + payload
├──────────────────────────┤
│    ...                    │
├──────────────────────────┤
│    Sidecar Block          │  (optional, legacy; not emitted by current encoder)
├──────────────────────────┤
│    End Mark               │  4 bytes (0x00000000)
├──────────────────────────┤
│    Content Checksum       │  4 bytes (optional)
└──────────────────────────┘
```

---

## Frame Header

**Minimum size:** 14 bytes (no optional fields)
**Maximum size:** 26 bytes (all optional fields present)

```
Offset  Size  Field
─────────────────────────────────────────
0       4     Magic number (0x53_4C_5A_31 = "SLZ1")
4       1     Version (must be 2; decoders must reject unknown values)
5       1     Flags (see below)
6       1     Codec ID
7       1     Compression level (1-9 codec level; user-facing levels 1-11 map to these)
8       1     Block size (log2 encoding, see below)
9       1     SC group size (chunks per self-contained group; typically 4)
10      4     Reserved (must be 0)
14      8     Content size (int64, present if flag bit 0 set)
22      4     Dictionary ID (uint32, present if flag bit 3 set)
```

### Flags (byte at offset 5)

| Bit | Name                         | Meaning                                           |
|-----|------------------------------|---------------------------------------------------|
| 0   | ContentSizePresent           | 8-byte content size follows the fixed header       |
| 1   | ContentChecksum              | 4-byte XXH32 checksum after the end mark           |
| 2   | BlockChecksums               | 4-byte XXH32 checksum after each block payload     |
| 3   | DictionaryIdPresent          | 4-byte dictionary ID follows content size (if any) |
| 4   | ParallelDecodeMetadataPresent | Frame contains a sidecar block for parallel Fast decode. **Legacy:** current encoder (v2.0+) does not set this flag; L2-L5 now use SC group-parallel decode instead. Decoders must still handle it for backward compatibility with older frames. |
| 5-7 | Reserved                     | Must be 0                                          |

### Codec ID (byte at offset 6)

| Value | Codec | Description                              |
|-------|-------|------------------------------------------|
| 0     | High  | High-ratio codec (optimal/lazy parsing)  |
| 1     | Fast  | Fast codec (greedy/lazy parsing)         |

### Block Size Encoding (byte at offset 8)

The block size is stored as `log2(blockSize) - 16`. To decode:

```
blockSize = 1 << (value + 16)
```

| Stored value | Block size |
|-------------|------------|
| 0           | 64 KB      |
| 1           | 128 KB     |
| 2           | 256 KB (default) |
| 3           | 512 KB     |
| 4           | 1 MB       |
| 5           | 2 MB       |
| 6           | 4 MB (maximum)   |

The block size must be a power of 2 in the range [64 KB, 4 MB].

### SC Group Size (byte at offset 9)

Number of 256 KB chunks per self-contained group. Must be 1-255 (0 is
invalid). Default is 4; the encoder adaptively scales this based on file
size (targeting ~16 groups for parallel decode). Decoders must use this
value, not a hardcoded constant, when walking SC-block group boundaries.

---

## Block Header

Each block is preceded by an 8-byte header:

```
Offset  Size  Field
─────────────────────────────────────────
0       4     Compressed size (uint32, with flag bits)
4       4     Decompressed size (uint32)
```

**Compressed size field flags:**

| Bit | Name                        | Meaning |
|-----|-----------------------------|---------|
| 31  | Uncompressed                | Block is stored (raw bytes, no chunk headers) |
| 30  | ParallelDecodeMetadata      | Block is a sidecar (see below), not decompressible data |
| 0-29 | Size                       | Compressed payload size in bytes |

- A value of `0x00000000` in the first 4 bytes is the **end mark** (no decompressed size follows).

**Uncompressed blocks:** When bit 31 is set, the entire payload is raw data with no internal block header and no chunk headers. The decompressor copies the payload directly to the output.

**Sidecar blocks:** When bit 30 is set, the block carries parallel-decode metadata. Its `decompressed_size` is 0; it produces no output. Serial decoders must skip the `compressed_size` bytes and advance without writing to dst. See the [Sidecar Block](#sidecar-block-parallel-decode-metadata) section.

**Decompressed size:** The number of bytes this block produces when decompressed. For the last block in a stream, this may be smaller than the frame's block size.

---

## End Mark

The end of the block sequence is signaled by 4 zero bytes:

```
00 00 00 00
```

This is distinguished from a valid block header because no valid block has a compressed size of 0.

---

## Content Checksum (optional)

If the `ContentChecksum` flag is set in the frame header, a 4-byte XXH32 checksum of the entire uncompressed content follows immediately after the end mark.

---

## Block Payload: Internal Block Header

Within each compressed block payload, a 2-byte **internal block header** describes the codec and flags for the block's chunks:

```
Byte 0 (bit layout):
  [3:0]  Magic nibble (must be 0x5)
  [4]    SelfContained flag
  [5]    TwoPhase flag
  [6]    RestartDecoder flag
  [7]    Uncompressed flag

Byte 1 (bit layout):
  [6:0]  DecoderType (0=High, 1=Fast, 2=Turbo)
  [7]    UseChecksums flag
```

### DecoderType values

| Value | Type  | Description                                    |
|-------|-------|------------------------------------------------|
| 0     | High  | 4-stream format (cmd, offs, lit, len)          |
| 1     | Fast  | 6-stream format (lit, deltalit, cmd, off16, off32, len) |
| 2     | Turbo | Same wire format as Fast                       |

### SelfContained flag

When set, chunks are grouped (default: 4 chunks per group, stored in the frame header's `sc_group_size` field) for parallel decompression. Within a group, chunks are decoded sequentially and may reference output from earlier chunks in the same group via LZ back-references. Between groups, there are no cross-references, enabling full parallelism across groups.

**Self-contained prefix table:** When SelfContained is set, the block payload is followed by a suffix table containing the first 8 bytes of each chunk except the first. This table has `(numChunks - 1) * 8` bytes. During parallel decompression, each chunk's initial 8 bytes may be decoded incorrectly (the LZ back-reference for `InitialRecentOffset = 8` cannot reach the prior chunk's output). After all chunks are decoded, the decompressor overwrites each chunk's first 8 bytes from this table to restore correctness.

---

## Sub-chunk Structure (Fast codec)

Each 256 KB chunk in the Fast codec is divided into two **sub-chunks** of up to 128 KB each. Each sub-chunk has its own independent set of six encoded streams. The first sub-chunk covers bytes [0, 131072) of the chunk output; the second covers [131072, 262144).

Sub-chunk boundaries are significant for the legacy sidecar path: the cross-chunk dependency analysis operates at chunk granularity, and the parallel decode worker assignment operates at slice boundaries that are multiples of 16 chunks (4 MB). The current encoder uses SC group-parallel mode for all Fast levels, so sidecar-based slicing is only relevant when decoding older frames.

---

## Chunk Structure

Each block is divided into one or more **chunks** of up to 256 KB (262,144 bytes). The chunk is the fundamental unit of compression.

### Chunk Header

Each chunk begins with a 4-byte little-endian header, optionally followed by a 3-byte checksum:

```
Bytes 0-3 (little-endian uint32):
  [ChunkSizeBits-1 : 0]  Compressed size minus 1
  [ChunkSizeBits+1 : ChunkSizeBits]  Type (0=normal, 1=memset)
  [31 : ChunkSizeBits+2]  Reserved (must be 0)
```

With the default `ChunkSizeBits = 18` (256 KB chunk):
- Bits [17:0] = compressed size minus 1 (0 .. 262,143)
- Bits [19:18] = type
- Bits [31:20] = reserved

**Type values:**

| Value | Meaning | Payload |
|-------|---------|---------|
| 0 | Normal | Compressed data follows |
| 1 | Memset | Byte at offset 4 is the fill value |

If the `UseChecksums` flag is set in the internal block header, 3 bytes of checksum follow the 4-byte header (bytes 4-6). This is the upper 24 bits of a CRC32 of the chunk's compressed payload (big-endian byte order).

**Chunk sizes:**
- Normal chunk: 4 bytes header (+ 3 bytes checksum if enabled) + compressed payload
- Memset chunk: 5 bytes total (4-byte header + 1-byte fill value)
- Uncompressed chunk: compressed size equals decompressed size; payload is raw bytes

---

## Sidecar Block (Parallel-Decode Metadata)

> **Legacy wire format.** The current encoder (v2.0+) uses SC
> (self-contained) group-parallel mode for L1-L5 and no longer emits
> sidecar blocks. This section documents the sidecar wire format so
> that decoders can still read older .slz frames that contain sidecars.

The sidecar block enables parallel Fast decode for frames that were
encoded without SC mode. It is an optional block marked with bit 30
(`ParallelDecodeMetadata`) in the block header's compressed_size field.
Sidecar blocks have `decompressed_size == 0` and produce no output.

The sidecar carries the pre-computed cross-chunk dependency data that
the parallel decoder needs to execute before spawning per-slice worker
threads: match copy operations and literal byte values that cross slice
boundaries.

### When present

The sidecar is present when the frame header's flag bit 4
(`ParallelDecodeMetadataPresent`) is set. In older encoders this
occurred for non-SC Fast codec (L2-L5) frames. The current encoder
(v2.0+) no longer sets this flag — all Fast levels now use SC
group-parallel decode instead. L1 has always used SC mode.
High codec (L6-L11) frames never carry a sidecar — L6-L8 use SC
group-parallel decode, and L9-L11 use two-phase parallel decode,
neither of which requires a sidecar.

### Sidecar block body wire format

```
Offset  Size      Field
─────────────────────────────────────────
0       4         Magic: 'PDSC' (0x43534450 LE)
4       1         Sidecar version (must be 2)
5       3         Reserved (must be 0)
8       varint    num_match_ops (LEB128 u32)
8+N     varint    num_literal_runs (LEB128 u32)
```

Followed by two sections:

#### Match ops section

For each match op (in monotonically-increasing target_start order):

| Field | Encoding | Description |
|-------|----------|-------------|
| delta_target | varint | `target_start - prev_target_start` (prev = 0 for first) |
| offset | varint | `target_start - src_start` (always > 0) |
| length | varint | Number of bytes to copy |

The decoder executes these as byte-wise forward copies: `dst[target_start .. target_start+length] = dst[src_start .. src_start+length]`. These propagate cross-slice match chain values.

#### Literal runs section

For each run (consecutive-position byte groups, sorted by position):

| Field | Encoding | Description |
|-------|----------|-------------|
| delta_position | varint | `run_start - prev_run_end` (prev_run_end = 0 for first) |
| run_length | varint | Number of consecutive bytes |
| bytes | raw | `run_length` literal byte values |

The decoder writes these directly to dst at the indicated positions.

### Compression characteristics

Delta encoding makes positions 1-2 varint bytes each instead of 8 raw bytes. Match offsets are stored as `target - src` (small positive integers). Literal bytes are grouped into maximal consecutive runs to amortize the per-byte position overhead.

Typical sidecar sizes (legacy frames only; the current encoder does not emit sidecars):
- L1-L4 enwik8 (100 MB): ~150 KB (~0.15% of input)
- L5 enwik8 (100 MB): ~1.2 MB (~1.2% of input)

L5 sidecars are larger because the lazy parser produces longer-distance matches with deeper transitive cross-chunk dependency chains.

### Decoder behavior (legacy sidecar path)

1. Parse the sidecar block body.
2. Execute all `literal_bytes` as scattered writes to dst (serial).
3. Execute all `match_ops` as sequential forward copies to dst (serial).
4. Dispatch parallel workers, each decoding a contiguous slice of chunks.

Decoders that recognize the block flag but don't support parallel decode
must skip the sidecar block by advancing past its `compressed_size`
bytes. The sidecar is redundant — the serial decoder produces identical
output without it.

---

## Codec Wire Formats

### Fast/Turbo Codec (6-stream format)

The Fast codec splits each chunk into up to two 128 KB sub-chunks, with six parallel byte streams per sub-chunk:

| Stream      | Type     | Content                                         |
|-------------|----------|-------------------------------------------------|
| Cmd         | byte     | Flag/command tokens encoding literal+match pairs |
| Lit         | byte     | Raw literal bytes                                |
| DeltaLit    | byte     | Delta-coded literal bytes (optional)             |
| Off16       | uint16   | Near (16-bit) match offsets                      |
| Off32       | uint32   | Far (32-bit) match offsets                       |
| Length      | byte     | Variable-length extra lengths                    |

**Flag byte encoding (Cmd stream):**

| Value      | Meaning                                                |
|------------|--------------------------------------------------------|
| >= 24      | Inline literal+match: bits[2:0]=literal count, bits[6:3]=match length, bit[7]=use recent offset |
| 0          | Long literal run: read length from Length stream (+64)  |
| 1          | Long near-offset match: read length from Length stream (+91) |
| 2          | Long far-offset match: read length from Length stream (+29) |
| 3..23      | Short far-offset match: length = flagbyte + 5          |

Each sub-chunk's streams are entropy-coded (Huffman or tANS) or stored raw, with a per-stream header indicating the encoding method.

### High Codec (4-stream format)

The High codec uses four streams per chunk:

| Stream  | Type    | Content                                        |
|---------|---------|------------------------------------------------|
| Cmd     | byte    | Interleaved literal-length and match-length tokens |
| Offs    | int32   | Match offsets (when not using a recent offset)  |
| Lit     | byte    | Raw literal bytes                               |
| Len     | int32   | Overflow lengths (when token can't hold the full length) |

The command stream encodes (literal_length, match_length, use_recent_offset) triples. Short lengths are packed inline; longer lengths spill into the Len stream. The High codec supports a recent-offset carousel of the last 3 offsets, selectable by index in the command token.

---

## Entropy Coding

Individual streams within a chunk are compressed using one of:

1. **Raw (memcopy)** — stream is stored uncompressed
2. **Huffman** — canonical Huffman coding with code lengths transmitted as a compact header
3. **tANS** — table-based asymmetric numeral systems with 4-way interleaved state

The choice is made per-stream based on a cost model that estimates the compressed size of each option. The stream header byte identifies which method was used.

### Huffman Code Length Encoding

Code lengths (0-11 bits) are transmitted using a two-level scheme: first a set of "code length code lengths" (3-bit values for a small alphabet), then the actual code lengths are entropy-coded using that meta-code.

### tANS Table Encoding

The tANS frequency table is transmitted as a sequence of symbol counts using variable-length coding. The decoder rebuilds the decoding table from these frequencies. Four interleaved tANS states are maintained for throughput.

---

## Sliding Window

In the framed streaming API, a **sliding window** provides cross-block back-references:

- The compressor maintains a buffer of `windowSize + blockSize` bytes.
- After compressing each block, decoded output slides forward: the most recent `windowSize` bytes are retained as dictionary context for the next block.
- The decompressor maintains the same window and passes the dictionary context to the block decoder so LZ back-references can reach into previously decoded blocks.
- Default window size: 128 MB. L11 uses 128 MB for best ratio.

When `SelfContained` mode is enabled, chunks are grouped (4 per group by default). Cross-group references are disabled, enabling parallel decompression. Within a group, chunks are decoded sequentially with cross-chunk context. Cross-block references via the sliding window are still active unless the frame compressor operates in fully independent mode.

**Dictionary ID:** The optional `DictionaryId` field in the frame header is an opaque 4-byte identifier that tags which pre-shared dictionary was used during compression. It does **not** carry dictionary content — both compressor and decompressor must have the dictionary available externally. If present, the dictionary is logically prepended to the sliding window before the first block (i.e., it occupies the initial window state so the first block can reference it via LZ back-references).

---

## Offset Encoding

Match offsets are encoded in two ranges:

### Low range (offset < 16,776,456)

```
bucket = BSR(offset + 760) - 9
nibble = (offset - 8) & 0xF
packed = nibble + 16 * bucket
```

This produces a single byte (0x00..0xEF) that the entropy coder compresses.

### High range (offset >= 16,776,456)

```
extraBits = BSR(offset - 16,710,912)
packed = extraBits | 0xF0
```

The packed byte (0xF0..0xFF) is followed by `extraBits` raw bits encoding the remainder.

---

## Constants Summary

| Constant           | Value       | Description                        |
|--------------------|-------------|------------------------------------|
| Magic number       | 0x534C5A31  | "SLZ1" in little-endian            |
| Version            | 2           | Current format version             |
| Chunk size         | 262,144     | 256 KB                             |
| Sub-chunk size     | 131,072     | 128 KB (Fast codec only)           |
| Min block size     | 65,536      | 64 KB                              |
| Max block size     | 4,194,304   | 4 MB                               |
| Default block size | 262,144     | 256 KB (= chunk size)              |
| Max dictionary     | 1,073,741,824 | 1 GB                             |
| Default window     | 134,217,728 | 128 MB                             |
| SafeSpace          | 64          | Extra bytes needed past output end |
| Huffman LUT bits   | 11          | 2048-entry decode table            |
| Initial copy bytes | 8           | Verbatim bytes at chunk start      |
| SC group size      | 4 (default) | Chunks per group in self-contained mode; adaptive 4-255 |
| Sidecar magic      | 0x43534450  | "PDSC" in little-endian (legacy)   |
| Sidecar version    | 2           | Sidecar format version (legacy)    |

---

## Parallel Decode Strategy Summary

### Decode

| Levels | Codec | Strategy | Mechanism |
|--------|-------|----------|-----------|
| L1     | Fast  | SC group-parallel | Encoder emits self-contained chunks (1 chunk per group). Decoded via `decompressCoreParallel`. No sidecar needed. |
| L2-L4  | Fast  | SC group-parallel | Encoder emits self-contained groups (adaptive sizing, ~16 groups per file). Each group decoded independently, no sidecar needed. Legacy frames may contain a sidecar instead; decoders must handle both paths. |
| L5     | Fast  | SC group-parallel | Same as L2-L4. Legacy frames may contain a larger sidecar (~1.2% overhead); decoders must handle both paths. |
| L6-L8  | High  | SC group-parallel | Encoder constrains chunks to self-contained groups (adaptive size, ~16 groups per file). Each group decoded independently, no sidecar needed. |
| L9-L11 | High  | Two-phase parallel | Phase 1: parallel entropy decode + token resolution. Phase 2: serial token execution. No sidecar (64 MB dictionary window makes cross-slice deps ubiquitous). |

### Compress

| Levels | Codec | Threading |
|--------|-------|-----------|
| L1     | Fast  | Parallel (per-chunk workers, SC mode) |
| L2-L5  | Fast  | Parallel (SC, per-group workers, adaptive group sizing) |
| L6-L8  | High  | Parallel (per-group workers, SC mode) |
| L9-L11 | High  | Parallel (per-block workers, sliding window) |
