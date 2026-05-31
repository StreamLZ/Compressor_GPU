# Vulkan L1 Codec — Phase 1 Design Spec

Authoritative design for the Vulkan port of StreamLZ's CUDA L1 (greedy LZ77)
codec. This document fixes the algorithm, the wire-format-equivalent
streams, the shader I/O, the warp-intrinsic mapping, and the Zig host glue
that lands in `src_vulkan/`. Phases 2–4 implement against it without
re-litigating these decisions.

Locked scope (phase 1):

- No frame / block / chunk-header wrapping. Encoder produces — and decoder
  consumes — the four raw streams (`lit`, `cmd`, `off16`, `length`) plus
  their sizes. Phase 2 adds the wrapping.
- Feature-complete L1 algorithm: all five token types implemented, both
  the warp-greedy parser (`scanBlock`) and the full `emitCmd` serializer.
- Single workgroup per chunk (matches CUDA's `blockDim=(32,1,1)`,
  `grid.x=n_chunks`). Phase 1 only tests single-chunk inputs (≤128 KB);
  the design is multi-chunk-clean for phase 2.
- Round-trip correctness (Vk-encode → Vk-decode = identity). Byte-equal
  output to the CUDA encoder is **not** a phase-1 requirement.
- Substrate: hand-written `src_vulkan/` (`vk_api.zig`, `descriptors.zig`,
  `dispatch.zig`). The `.comp` shaders are substrate-agnostic GLSL.

---

## 1. Algorithm summary — encoder (`slzLzEncodeKernel`, `use_chain=0`)

Source: `src/encode/lz_kernel.cu`, `src/encode/lz_greedy_parser.cuh`,
`src/encode/lz_token_emit.cuh`, `src/encode/lz_format.cuh`.

One **workgroup of 32 lanes per chunk**. All state below is per-warp.

### 1.1 Hash-table init (one warp, hash_bits = 17)

```
const uint32_t hash_size = 1u << hash_bits;        // 1<<17 = 131072 entries
const uint32_t hash_mask = hash_size - 1;          // 0x1FFFF
uint32_t* ht = global_hash + chunk_id * hash_size; // per-chunk slot
for (i = lane; i < hash_size; i += 32) ht[i] = HASH_EMPTY;  // 0xFFFFFFFFu
__syncwarp();
```

Per-chunk hash table = 128 K × `u32` = **512 KB** of device-local SSBO.
The encoder needs this much scratch per concurrently-active chunk. Phase 1
only dispatches one chunk at a time, so we provision exactly 512 KB.

### 1.2 Two-pass block design

LZ matches and literal runs **never cross the 64 KB block boundary**
(`LZ_BLOCK_SIZE = 0x10000`). The kernel runs `scanBlock` twice:

- **Block 1**: `start = anchor`, `end = min(src_size, LZ_BLOCK_SIZE)`.
  `block2_start = 0` (no off32 base shift yet).
- **Block 2** (only when `src_size > LZ_BLOCK_SIZE`):
  `start = max(anchor, LZ_BLOCK_SIZE)`, `end = src_size`.
  `block2_start = LZ_BLOCK_SIZE`. Records `cmd_stream2_offset` =
  `streams.token_count` at the boundary.

The shared `OutputStreams` cursor (`lit_count`, `token_count`,
`off16_count`, `off32_pos`, `off32_count`, `length_count`) accumulates
across both passes. Only `off32_count` is reset at the boundary (it is
captured into `off32_count_block1`/`block2` separately for header packing).

Phase 1 inputs are ≤ 128 KB so both passes can fire, but the **off32
stream and `cmd_stream2_offset` are not emitted into a header**; the
encoder communicates them back as scalar outputs (see §4).

### 1.3 `scanBlock` inner loop (warp-greedy parser)

For each iteration the warp covers up to **25 candidate positions** in
parallel (`active_count = WARP_SIZE - HASH_LOOKAHEAD = 32 - 7 = 25`) so
that lane K's 8-byte hash key can be assembled from lanes K..K+7 via
`__shfl_down_sync(0..7)`.

Per position `my_pos = pos + lane`:

1. **Build key4 / key8** — 8 `shfl_down` shuffles assemble two words from
   the per-lane source byte. `key4 = low 32 bits` (used for direct content
   comparison); `key8 = full 64-bit` (used for the 6-byte hash via
   `hashKey6`).
2. **Hash**: `h = hashKey6(key8, hash_bits, hash_mask)`. The 6-byte hash
   matches the CPU's `FastMatchHasher` for `hasher_k=6` — Fibonacci
   multiplier `0x79B97F4A7C150000` (the `0x9E3779B97F4A7C15ULL << 16`),
   index = `(word * mult) >> (64 - hash_bits)` then `& hash_mask`.
3. **Serial-order simulation**. The CPU writes `ht[h(P)] = P` after
   reading it. To match the CPU's reads, each lane K computes:
   - `bucket_same = __match_any_sync(FULL_WARP_MASK, h)` → bitmask of
     lanes with the same hash value as me.
   - `active_mask = __ballot_sync(FULL_WARP_MASK, is_active)`.
   - `lower_same_bucket = bucket_same & active_mask & ((1 << lane) - 1)`.
   - If `lower_same_bucket != 0`, CPU would have read the entry written by
     the **highest lower** lane with my hash. That lane index is
     `(WARP_SIZE - 1) - __clz(lower_same_bucket)`. Test their `key4` via a
     warp-scope `__shfl_sync(FULL_WARP_MASK, key4, key_src_lane)` — if
     equal and `their_pos + 8 <= my_pos` we have a hash match referencing
     that lane's position.
   - Otherwise: CPU saw the pre-warp `ht[h]`. Load it and compare
     (`readU32LE(src + ref_val) == key4` + offset-8 guard).
4. **Secondary match attempts** when `is_active && !hash_match`:
   - **XOR / recent-offset match** when `recent_offset < 0` and
     `my_pos >= |recent_offset|`. If bytes 1..3 of `key4` equal bytes 1..3
     of `*(src + my_pos + recent_offset)` (i.e. `(key4 ^ recent_word) &
     0xFFFFFF00 == 0`), match starts at `my_pos + 1`, length `≥ 3`,
     offset = recent (encoded as `use_recent`).
   - **−8 fixed-offset match** when `my_pos >= 8`. If 4 bytes at
     `my_pos − 8` equal `key4`, match starts at `my_pos`, offset = 8.
5. **Match-type ballot**: `match_ballot = __ballot_sync(has_match)`.
   `my_match_type ∈ {0=hash, 1=xor, 2=eight}`.
6. **No-match stride**: if `match_ballot == 0`,
   `dist = pos - anchor`; if `dist >= 128` then
   `step = min((dist >> 7) + 1, 16)` else `step = 1`. Advance `pos += step`
   and continue.
7. **Hash-table writes (warp-coherent)**. Every lane up to
   `write_limit = (match_ballot != 0) ? __ffs(match_ballot) - 1 :
   no_match_step - 1` writes its bucket. To match Vulkan's lack of a
   "highest-indexed lane wins" guarantee on overlapping stores, the port
   does explicit highest-lane-in-bucket election:
   - `write_mask = __ballot_sync(is_writer)`.
   - `bucket_grp = __match_any_sync(h)`.
   - `group_mask = bucket_grp & write_mask`.
   - `top_lane = group_mask ? (31 - __clz(group_mask)) : -1`.
   - Only the lane `lane == top_lane` performs `ht[h] = my_pos`.
8. **Match resolution + extension**. The winning lane =
   `first_lane = __ffs(match_ballot) - 1`. `winning_type` is broadcast
   from that lane via `__shfl_sync`. Compute `match_pos`, `match_ref`,
   `min_match_len`:
   - hash:   `match_pos = pos+first_lane`, `match_ref = shfl(my_ref, first_lane)`, `min=4`.
   - xor:    `match_pos = pos+first_lane+1`, `match_ref = match_pos - |recent_offset|`, `min=3`.
   - eight:  `match_pos = pos+first_lane`, `match_ref = match_pos - 8`, `min=4`.

   Extend in 32-lane strides: each iteration compares
   `src[match_pos + check] != src[match_ref + check]` across the warp
   (with the past-end positions force-`true`), reduce with
   `__ballot_sync`, and `match_len = ext + __ffs(mm_mask) - 1` when a
   mismatch is found.

9. **Far-offset minimum-match filter**: if
   `resolved_off > NEAR_OFFSET_MAX (0xFFFF)` and `match_len < 14`
   (`FAR_OFFSET_MIN_MATCH`), reject the match and try `pos = match_pos+1`.

10. **Backward extension** (lane 0 only). Walks `mp, mr` backward while
    `mp > anchor && mr > 0 && src[mp-1] == src[mr-1]`. The step count is
    `__shfl_sync`-broadcast and applied to `match_pos`, `match_ref`,
    `match_len`.

11. **Literal copy + token emit**. `lit_len = match_pos - anchor`. All 32
    lanes cooperatively copy literals into `lit_buf`. Then lane 0 picks:
    - **Fast inline path** (`lit_len ≤ 7 && match_len ≤ 15 &&
      resolved_off ≤ 0xFFFF`): one token byte, optional u16 off16.
    - Otherwise: `emitCmd` (the full serializer; see §3).

12. **Match-range rehash** (only when `enable_match_rehash != 0`; L1 sets
    this to 0 so it's dead-code for phase 1 but still ported for parity
    with the higher levels in phase 2). Inserts hash entries at
    exponentially-spaced offsets inside the just-emitted match, using the
    same highest-lane-wins gating as step 7.

13. `anchor = match_pos + match_len`; `pos = anchor`; loop.

### 1.4 Trailing literals

After the `while (pos + 4 <= end_pos)` loop, any
`trailing = end_pos - anchor > 0` is copied into `lit_buf` and a single
literal-only `emitCmd(lit_len=trailing, match_len=0, offset=0)` is issued
on lane 0.

---

## 2. Algorithm summary — decoder (`decodeSubChunkRawMode`, `OFF16_SPLIT=false`)

Source: `src/decode/lz_decode_raw.cuh` + `src/decode/lz_decode_core.cuh`.

One **workgroup of 32 lanes per chunk** decoding back into `dst`. State:
`cmd_pos`, `lit_pos`, `off16_pos`, `dst_pos`, `recent_offset`,
`length_offset`. `dst_pos` starts at `dst_offset + initial_copy` (phase 1
sets both to zero; the 8-byte initial copy is the encoder's
`is_first` prefix that we are choosing **not** to wire through in phase 1
since no header wrapping exists — see §4 for the corollary that the
encoder also skips the initial copy in phase 1).

Outer loop runs while `cmd_pos < cmd_size`:

### 2.1 Fast batched path (preferred)

Load 32 token bytes coalesced. Test
`my_is_long = my_cmd < TOKEN_SHORT_MIN (24)`; if `__ballot_sync` is zero
for the whole warp (no long tokens in this batch), execute the fast path:

1. **Parallel decode** standard token bit-fields per lane:
   `lit = cmd & 7`, `match = (cmd >> 3) & 0xF`, `use_recent = cmd >> 7`,
   `consumes_off16 = (!use_recent) ? 1 : 0`.
2. **Prefix-scan** `consumes_off16` (Hillis-Steele warp scan in
   `warpScanU32`) → per-lane `my_off16_local`, total.
3. **Off16 load**: each lane that consumes an off16 loads its own u16 LE
   from `off16_raw + (off16_pos + my_off16_local) * 2` and sets
   `my_match_offset = -int32_t(v)`.
4. **Use-recent fill**: lanes with `use_recent` inherit the previous
   "fresh" offset via a per-lane scan
   (`my_prefix = fresh_mask & ((2u << lane) - 1)`,
   `src_lane = lastBitSet(my_prefix)`) and a warp shuffle.
5. **Prefix-scan** `my_total = lit + match` → per-lane `my_dst_local`,
   total. Same for `lit` → `my_lit_local`.
6. **Sequential cooperative copy**: for `k = 0..batch_size-1`, broadcast
   lane `k`'s `(lit_len, match_len, match_off, dst_local, lit_local)` via
   `__shfl_sync`, then perform `warpLiteralCopy` and `warpMatchCopy` for
   that token. `__syncwarp()` between copies.
7. Advance cursors: `cmd_pos += batch_size`,
   `off16_pos += total_off16_used`, `dst_pos += total_dst`,
   `lit_pos += total_lit`. If any fresh offsets were read,
   `recent_offset = shfl(my_match_offset, lastBitSet(fresh_mask))`.

### 2.2 Slow scalar path (when batch contains a long token)

Lane 0 only decodes one token:

- `token >= TOKEN_SHORT_MIN (24)`: standard. Fields as in 2.1; consume one
  off16 entry if `!use_recent` and `off16_pos < off16_count`.
- `token == TOKEN_LONG_LITERAL (0)`: literal-only command.
  `lit_len = readLength(...) + LONG_LITERAL_BASE (64)`. `use_recent = 1`
  (no offset consumed).
- `token == TOKEN_LONG_NEAR (1)`: near-offset long match.
  `match_len = readLength(...) + LONG_NEAR_BASE (91)`. Consume one off16.
- `token == TOKEN_LONG_FAR (2)`: far-offset long match. **Not used in
  phase 1** because we have no off32 stream — the encoder will never emit
  it (we reject the corresponding `emitCmd` Step-4 far paths). See §11
  for the constraint that forces this.
- Token in `3..23`: short-far match (`match = token + SHORT_FAR_BASE = 5
  + token`). **Not used in phase 1** for the same reason.

`cmd_pos++`. Broadcast the four parsed values to all lanes. Cooperative
literal/match copy via `warpLiteralCopy` / `warpMatchCopy`. Update
`recent_offset` only if `!use_recent`.

### 2.3 Trailing literals

After the loop, `trailing = max(0, lit_size - lit_pos)` bytes are copied
verbatim into `dst[dst_pos ..]`. (Emitted by the encoder when the final
`emitCmd` for trailing literals fires.)

---

## 3. The five token types — exact bit layouts

Constants are pulled from `src/decode/slz_wire_format.cuh` and
`src/encode/lz_token_emit.cuh`. All bit-layouts are **per byte** unless
noted.

| #  | Token range       | Name            | Layout (low → high bit)                              | Side streams consumed                  | Emitted by encoder when                                                            |
|----|-------------------|-----------------|------------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------|
| 0  | `token == 0`      | long literal    | tag only                                             | `length` (1 or 3 bytes)                | `remaining_lit ≥ LITERAL_RUN_LENGTH_THRESHOLD (64)`                                |
| 1  | `token == 1`      | long near       | tag only                                             | `length` (1 or 3 bytes), `off16` (2 B) | `offset ≤ 0xFFFF` and (`match > 90` or fast path failed)                           |
| 2  | `token == 2`      | long far        | tag only                                             | `length` (1 or 3 bytes), `off32` (3/4) | `offset > 0xFFFF` and `match_len - 5 ∉ [0, 23]`                                    |
| 3  | `token ∈ [3, 23]` | short far       | `match_len = token + 5` (so 8..28 inline)            | `off32` (3/4 bytes)                    | `offset > 0xFFFF` and `match_len - 5 ∈ [0, 23]`                                    |
| 4  | `token ≥ 24`      | standard        | bits 0–2 = `lit_len`; bits 3–6 = `match_len`; bit 7 = `use_recent` | `off16` (2 B) iff `!use_recent`     | `lit ≤ 7 && match ≤ 15 && offset ≤ 0xFFFF` (fast path; also various continuations) |

Decoder bases that reconstruct the actual lengths:

```
LONG_LITERAL_BASE = 64    // lit_len = readLength() + 64
LONG_NEAR_BASE    = 91    // match_len = readLength() + 91 (= NEAR_CONT_MATCH_MAX + 1)
LONG_FAR_BASE     = 29    // match_len = readLength() + 29 (= 5 + 23 + 1)
SHORT_FAR_BASE    = 5     // match_len = token + 5
```

### 3.1 Length stream encoding

Source: `writeLengthValue` in `lz_token_emit.cuh`, `readLength` in
`slz_wire_format.cuh`.

`value ∈ [0, LENGTH_INLINE_MAX = 251]` → **1 byte** equal to `value`.

`value > 251` → 3 bytes:
- byte 0 = `(low2 − 4) & 0xFF` where `low2 = value & 3`. With
  `low2 ∈ [0,3]` this wraps to one of `{0xFC, 0xFD, 0xFE, 0xFF}` — those
  four values are reserved tags above `EXT_LENGTH_THRESHOLD = 251`.
- bytes 1..2 = u16 LE `remainder = (value - low2 - 252) >> 2`.

Decoder side: read byte; if `v > 251`, read 2 more bytes as u16 LE and
`v += extra * 4`. The low-2-bit info lives in the tag (`tag - 0xFC`); the
expansion `v += extra * EXT_LENGTH_SCALE (4)` plus the inherited tag low
nibble reconstructs the original.

### 3.2 Off16 stream

Plain interleaved u16 LE entries: byte at offset `2*i` is the low byte of
entry `i`, byte at `2*i+1` is the high byte. Each entry encodes
`offset` (the absolute backward distance), where `0` is the sentinel
"use recent offset" already encoded into the token's bit 7.

### 3.3 Off32 stream

Source: `writeOffset32` / read by `lz_decode_general.cuh` (not used by raw
mode). **Not produced by phase 1.** Format kept here for the eventual
phase 2 implementation:

- Default entry = 3 bytes LE.
- If `offset ≥ OFF32_LARGE_TAG (0xC00000)`: low 22 bits get `OFF32_LARGE_TAG`
  OR'd in (the high two bits become the marker `0b11`); the LE-24 write
  stores that; **one extra byte** follows carrying
  `(offset - truncated) >> 22`. So large-offset entries are 4 bytes.
- Decoder detects the marker via the high byte of the 24-bit entry being
  `≥ OFF32_LONG_ENTRY_TAG (0xC0)`.

---

## 4. Stream layout — phase 1 interface

Phase 1 has **no headers**. The encoder produces four streams plus a
sidecar size struct; the decoder consumes exactly those.

### 4.1 Per-stream device buffers (separate VkBuffer per stream)

| Stream        | Type             | Worst-case size (per chunk)         | SSBO purpose            |
|---------------|------------------|-------------------------------------|-------------------------|
| `lit_buf`    | `uint8_t[]`     | `src_size` (every byte a literal)   | Verbatim literals       |
| `cmd_buf`    | `uint8_t[]`     | `src_size / 4` (1 token per ≥ 4-byte match) | Token / command stream  |
| `off16_buf`  | `uint16_t[]` (LE) | `src_size / 2` (off16 entry covers ≥ 4 src bytes) | Match offsets ≤ 0xFFFF |
| `length_buf` | `uint8_t[]`     | `src_size / 4` (extended lengths only on long tokens) | Variable-length stream  |

Total worst case = `2 * src_size`. Phase 1 just over-provisions to
`2 * src_size` per output side-buffer; phase 2 tightens this when it adds
real allocation. The encoder writes them at offset 0 into each buffer.

### 4.2 Returned sizes (sidecar struct, host-visible u32 SSBO)

After the encoder dispatch completes, a `comp_sizes` SSBO contains:

```
struct L1Sizes {
    uint32_t lit_size;       // bytes written into lit_buf
    uint32_t cmd_size;       // bytes written into cmd_buf
    uint32_t off16_count;    // number of u16 entries in off16_buf
    uint32_t length_used;    // bytes written into length_buf
};
```

The host reads this back via a small map+invalidate or a transient
staging buffer, and uses the values to size the decoder's input bindings
in the subsequent round-trip dispatch.

### 4.3 Original size + initial-copy decision

`src_size` is required by the decoder to bound the loop and copy the
trailing literals correctly. Phase 1 conveys it via push constant
(see §5.2), **not** through a header.

Phase 1 *also* skips the 8-byte initial-literal copy because there is no
header layer to record an `is_first` flag. The encoder sets `anchor = 0`
unconditionally and the decoder sets `initial_copy = 0`,
`dst_offset = 0`. Round-trip remains correct (no bytes are lost — the
first 8 bytes become regular literals in the `lit_buf` stream emitted by
the eventual `emitCmd(trailing-literals)` or by the first match's
literal prefix). Phase 2 reintroduces the initial copy when it wraps the
streams in a sub-chunk header.

---

## 5. Shader I/O bindings

### 5.1 Encoder shader — `src_vulkan/shaders/lz_encode.comp`

```glsl
#version 460
#extension GL_KHR_shader_subgroup_basic              : require
#extension GL_KHR_shader_subgroup_ballot             : require
#extension GL_KHR_shader_subgroup_shuffle            : require
#extension GL_KHR_shader_subgroup_shuffle_relative   : require
#extension GL_KHR_shader_subgroup_arithmetic         : require
#extension GL_KHR_shader_subgroup_vote               : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
// Pipeline creation pins subgroupSize = 32 via
// VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT +
// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo on devices that
// support VK_EXT_subgroup_size_control. Phase 1 hard-asserts at probe
// time that the device's subgroupSize is 32.

layout(set = 0, binding = 0) readonly  buffer Src       { uint8_t b[]; } src_buf;
layout(set = 0, binding = 1) writeonly buffer Lit       { uint8_t b[]; } lit_buf;
layout(set = 0, binding = 2) writeonly buffer Cmd       { uint8_t b[]; } cmd_buf;
layout(set = 0, binding = 3) writeonly buffer Off16     { uint8_t b[]; } off16_buf; // u16 LE, byte-addressed
layout(set = 0, binding = 4) writeonly buffer Length    { uint8_t b[]; } length_buf;
layout(set = 0, binding = 5)          buffer Hash      { uint  e[]; } hash_buf;     // 128K entries (hash_bits=17)
layout(set = 0, binding = 6) writeonly buffer Sizes     { uint  s[4]; } sizes_buf;  // [lit, cmd, off16, len]

layout(push_constant) uniform PC {
    uint src_size;       // bytes in src_buf to encode (<= LZ_BLOCK_SIZE in single-block phase 1; ≤ 128 KB in two-block phase 1)
    uint hash_bits;      // 17 for L1
    // No use_chain (always greedy = 0). No enable_match_rehash (L1=0).
    // No chunk_id — single-chunk dispatch.
} pc;
```

Notes:

- `uint8_t` SSBO addressing requires
  `GL_EXT_shader_explicit_arithmetic_types_int8` and the
  `shaderInt8` / `storageBuffer8BitAccess` features. Both are mandatory in
  Vulkan 1.3 core; tier-1 NVIDIA is fine. (`vk_api.zig`'s feature struct
  will gain the flag.)
- `Hash` is `readwrite` (default with no qualifier). All accesses are
  warp-coherent so no `coherent` decoration is needed; the encoder's
  `__syncwarp`s map to `subgroupBarrier()` and that is sufficient.
- `Sizes` is `writeonly` and lane 0 stores the four totals after the
  scanBlock passes complete.

### 5.2 Decoder shader — `src_vulkan/shaders/lz_decode.comp`

```glsl
#version 460
#extension GL_KHR_shader_subgroup_basic              : require
#extension GL_KHR_shader_subgroup_ballot             : require
#extension GL_KHR_shader_subgroup_shuffle            : require
#extension GL_KHR_shader_subgroup_shuffle_relative   : require
#extension GL_KHR_shader_subgroup_arithmetic         : require
#extension GL_KHR_shader_subgroup_vote               : require
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly  buffer Cmd     { uint8_t b[]; } cmd_buf;
layout(set = 0, binding = 1) readonly  buffer Lit     { uint8_t b[]; } lit_buf;
layout(set = 0, binding = 2) readonly  buffer Off16   { uint8_t b[]; } off16_buf;
layout(set = 0, binding = 3) readonly  buffer Length  { uint8_t b[]; } length_buf;
layout(set = 0, binding = 4)           buffer Dst     { uint8_t b[]; } dst_buf;

layout(push_constant) uniform PC {
    uint cmd_size;          // bytes in cmd_buf
    uint lit_size;          // bytes in lit_buf
    uint off16_count;       // entries in off16_buf
    uint length_remaining;  // bytes in length_buf
    uint dst_size;          // original input size; bounds dst writes
} pc;
```

Notes:

- No `initial_copy` or `dst_offset` push constants — phase 1 pins both to
  zero, so the GLSL hard-codes `dst_pos = 0` / `recent_offset = -8`.
- `dst_size` is *informational* in the raw-mode decoder (it does no
  bounds checks); it is still passed so the trailing-literal copy can be
  size-clamped if necessary and so phase 2 multi-chunk extension has a
  natural slot.

### 5.3 Workgroup + sub-group size

- `local_size_x = 32, 1, 1`.
- Dispatch group count `[1, 1, 1]` (one workgroup per chunk; phase 1 =
  one chunk).
- Subgroup-size requirement: **32**. Enforced via
  `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` on pipeline
  creation when `VK_EXT_subgroup_size_control` is available; otherwise
  the probe path falls back to checking
  `VkPhysicalDeviceSubgroupProperties.subgroupSize == 32` at startup.
  Probe failure → `error.UnsupportedSubgroupSize` and the codec refuses
  to run.

### 5.4 Buffer usage flags

- Encoder bindings: `src` = `STORAGE | TRANSFER_DST`; `lit`/`cmd`/`off16`/
  `length`/`sizes` = `STORAGE | TRANSFER_SRC` (for readback in round-trip
  tests); `hash` = `STORAGE` (device-local only).
- Decoder bindings: `cmd`/`lit`/`off16`/`length` = `STORAGE |
  TRANSFER_DST`; `dst` = `STORAGE | TRANSFER_SRC`.
- All bindings phase-1: `STORAGE_BUFFER` descriptors via the existing
  `descriptors.getOrCreate(..., n_storage = 7 / 5, push_size = …)`.

---

## 6. Warp-intrinsic mapping — every call site

Listed in source order. Each row gives the CUDA call, the GLSL
replacement, and a one-line rationale specific to the site.

### 6.1 `lz_greedy_parser.cuh` (encoder, `scanBlock`)

| # | Line(s)         | CUDA                                                    | GLSL                                                          | Notes                                                   |
|---|-----------------|---------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------|
| 1 | 43–49           | `__shfl_down_sync(FULL_WARP_MASK, my_byte, k)` for k=1..7 | `subgroupShuffleDown(my_byte, k)`                             | Build 8-byte hash key from per-lane bytes               |
| 2 | 82              | `__match_any_sync(FULL_WARP_MASK, h)`                   | `match_any(h)` emulation (see §7)                             | First match_any site: bucket-coherence for hash         |
| 3 | 83              | `__ballot_sync(FULL_WARP_MASK, is_active)`              | `uint(subgroupBallot(is_active).x)`                           | Active-lane mask                                        |
| 4 | 106             | `__clz(lower_same_bucket)`                              | `(lower_same_bucket == 0u) ? 32 : 31 - findMSB(lower_same_bucket)` | `__clz(0)==32` on CUDA; explicit guard for GLSL |
| 5 | 108             | `__shfl_sync(FULL_WARP_MASK, key4, key_src_lane)`       | `subgroupShuffle(key4, key_src_lane)`                         | Broadcasts the highest-lower lane's key4 for comparison |
| 6 | 188             | `__ballot_sync(FULL_WARP_MASK, has_match)`              | `uint(subgroupBallot(has_match).x)`                           | Match ballot                                            |
| 7 | 219, 236        | `__ffs(match_ballot)`                                   | `(match_ballot == 0u) ? 0 : findLSB(match_ballot) + 1`        | Lowest-set-bit + 1; explicit zero guard                 |
| 8 | 222             | `__ballot_sync(FULL_WARP_MASK, is_writer)`              | `uint(subgroupBallot(is_writer).x)`                           | Writer-lane mask for hash-table store                   |
| 9 | 223             | `__match_any_sync(FULL_WARP_MASK, h)`                   | `match_any(h)` emulation (see §7)                             | Second match_any site: bucket-group for HT writes       |
| 10 | 225             | `__clz(group_mask)`                                     | `(group_mask == 0u) ? 32 : 31 - findMSB(group_mask)`          | Top-lane election                                       |
| 11 | 237             | `__shfl_sync(FULL_WARP_MASK, my_match_type, first_lane)` | `subgroupShuffle(my_match_type, first_lane)`                 | Broadcast winning match type                            |
| 12 | 245             | `__shfl_sync(FULL_WARP_MASK, my_ref, first_lane)`       | `subgroupShuffle(my_ref, first_lane)`                         | Broadcast winning match ref                             |
| 13 | 271             | `__ballot_sync(FULL_WARP_MASK, mm)`                     | `uint(subgroupBallot(mm).x)`                                  | Match-extension mismatch ballot                         |
| 14 | 273             | `__ffs(mm_mask)`                                        | `findLSB(mm_mask) + 1`  (mm_mask known non-zero here)         | Mismatch-position extraction                            |
| 15 | 310             | `__shfl_sync(FULL_WARP_MASK, bw_steps, 0)`              | `subgroupBroadcast(bw_steps, 0)`                              | Broadcast lane-0's backward-extension count             |
| 16 | 364             | `__ballot_sync(FULL_WARP_MASK, rehash_active)`          | `uint(subgroupBallot(rehash_active).x)`                       | Rehash writer mask (only fires when `enable_match_rehash != 0`; dead-code at L1) |
| 17 | 365             | `__match_any_sync(FULL_WARP_MASK, h_rehash)`            | `match_any(h_rehash)` (dead at L1)                            | Rehash bucket group                                     |
| 18 | 367             | `__clz(group_mask)`                                     | `(group_mask == 0u) ? 32 : 31 - findMSB(group_mask)`          | Rehash top-lane election                                |
| 19 | various `__syncwarp` | `__syncwarp()`                                      | `subgroupBarrier(); subgroupMemoryBarrierBuffer();`           | Buffer barrier covers `hash_buf` write→read ordering    |

All warp loops on the encode-side that touch `ht[]` (read-then-write
within the same iteration) need `subgroupMemoryBarrierBuffer()` between
the read phase and the write phase to enforce ordering across the
participating lanes; the source CUDA achieves this implicitly through
`__syncwarp` after each scope.

### 6.2 `lz_decode_raw.cuh` (decoder, `decodeSubChunkRawMode`)

| # | Line(s)   | CUDA                                              | GLSL                                                                 | Notes                                              |
|---|-----------|---------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------|
| 1 | 51, 79    | `__ballot_sync(FULL_WARP_MASK, pred)`             | `uint(subgroupBallot(pred).x)`                                       | `any_long`, `fresh_mask`                          |
| 2 | 88        | `lastBitSet(my_prefix)` = `31 - __clz(my_prefix)` | `31 - findMSB(my_prefix)` (guarded by `my_prefix != 0`)              | Recent-offset src-lane scan                        |
| 3 | 89        | `__shfl_sync(FULL_WARP_MASK, my_match_offset, src_lane)` | `subgroupShuffle(my_match_offset, src_lane)`                 | Recent-offset broadcast                            |
| 4 | 102–106   | `__shfl_sync(FULL_WARP_MASK, X, k)` for X ∈ {lit_len, match_len, match_offset, dst_local, lit_local} | `subgroupShuffle(X, k)` | Per-batch broadcast in the cooperative-copy loop  |
| 5 | 131       | `__shfl_sync(FULL_WARP_MASK, my_match_offset, last_fresh)` | `subgroupShuffle(my_match_offset, last_fresh)`              | Latch `recent_offset` after batch                  |
| 6 | 179–183   | `__shfl_sync(FULL_WARP_MASK, X, 0)`               | `subgroupBroadcast(X, 0)`                                            | Broadcast slow-path scalar parse                   |
| 7 | 202–203   | `__shfl_sync(FULL_WARP_MASK, X, 0)`               | `subgroupBroadcast(X, 0)`                                            | Re-broadcast `recent_offset`, `length_offset`      |
| 8 | warpScanU32 internals | `__shfl_up_sync(FULL_WARP_MASK, inclusive, d)` | `subgroupShuffleUp(inclusive, d)`                          | Hillis-Steele inclusive scan; reused in 3 places  |
| 9 | warpScanU32 last     | `__shfl_sync(FULL_WARP_MASK, inclusive, 31)` | `subgroupBroadcast(inclusive, 31)`                              | Total broadcast                                    |
| 10 | `__syncwarp` after literal/match copy | `__syncwarp()`             | `subgroupBarrier(); subgroupMemoryBarrierBuffer();`                | `dst` write→read ordering for the next iteration  |

`warpLiteralCopy` and `warpMatchCopy` are pure strided loops with no
intrinsics — port verbatim.

---

## 7. `__match_any_sync` emulation

The encoder uses `__match_any_sync` at exactly **two semantic sites**
(both inside `scanBlock`); the source code spells them out three times
because the rehash branch reuses the same primitive a third time when
`enable_match_rehash != 0` (dead at L1).

### 7.1 Sites and inputs

| Site | Source line | Key             | Result consumed as                                                                       |
|------|-------------|-----------------|------------------------------------------------------------------------------------------|
| A    | 82          | `h` (hash bucket) | `bucket_same` for serial-order simulation (find highest lower lane with same hash)        |
| B    | 223         | `h` (hash bucket) | `bucket_grp` for hash-table-write election (highest writer in each bucket wins)           |
| (C)  | 365         | `h_rehash`      | Same role as B for match-range rehash. Dead at L1; needed at L2+ in phase 2.              |

### 7.2 GLSL replacement

```glsl
// Emulates CUDA's __match_any_sync(FULL_WARP_MASK, key) for a 32-lane warp.
// Returns a bitmask of lanes whose `key` equals mine. All 32 lanes must
// be converged when calling (the loop issues subgroupShuffle 32 times).
uint match_any(uint key) {
    uint my_mask = 0u;
    for (uint k = 0u; k < 32u; k++) {
        uint k_key = subgroupShuffle(key, k);
        if (k_key == key) my_mask |= (1u << k);
    }
    return my_mask;
}
```

Cost: 32 shuffles + 32 compares per call. The encoder calls this 2× per
position (~25 positions per warp iteration), so ~50 shuffle batches per
iter — acceptable for phase 1. Phase 2 can swap in a smarter version
(per-byte ballot reduction) if profiling demands it; the function shape
keeps that local.

The dead L1 site (C) still compiles inside the
`if (enable_match_rehash)` branch — the push constant `enable_match_rehash`
is hard-zero at L1, the branch is divergent-free across the warp (all 32
lanes take the same path), and the GLSL compiler will not eliminate the
`match_any` call because of side-effect conservatism. To preserve compile
parity with the CUDA path the port keeps the branch present and trusts
the dynamic zero to skip it; phase 2 lifts it via spec constant.

---

## 8. Hash-table design

- L1 uses `hash_bits = 17`. `hash_size = 1 << 17 = 131072` entries of
  `uint32_t` = **512 KB per chunk**.
- The buffer is device-local (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`,
  no host visibility) and the descriptor is `storage` (read/write).
- The encoder zeroes (`HASH_EMPTY = 0xFFFFFFFFu`) it on entry using a
  32-lane stride. Phase 1 reuses the same buffer across encode invocations
  by **re-initializing each time** inside the shader — there is no
  cross-call persistence requirement.
- For phase-2 multi-chunk dispatch, the buffer becomes
  `hash_size × n_chunks × 4` bytes (matches the CUDA layout); chunk_id
  becomes a push constant and `ht = hash_buf + chunk_id * hash_size`.
  Phase 1 hard-codes `chunk_id = 0`.

---

## 9. Host-side interface — `src_vulkan/l1_codec.zig`

New file (does not exist yet). Exposes:

```zig
const std = @import("std");
const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");
const probe = @import("probe.zig");

/// Returned by encodeL1Sync; passed straight into decodeL1Sync.
pub const L1Sizes = extern struct {
    lit_size: u32,
    cmd_size: u32,
    off16_count: u32,
    length_used: u32,
};

/// One contiguous device-side L1 stream bundle. Buffers are
/// owned by the L1Codec instance; the slices are valid until
/// L1Codec.releaseStreams(self, &streams) or codec deinit.
pub const L1Streams = struct {
    lit_buf: vk.VkBuffer,
    cmd_buf: vk.VkBuffer,
    off16_buf: vk.VkBuffer,
    length_buf: vk.VkBuffer,
    sizes: L1Sizes,
};

pub const EncodeResult = struct {
    streams: L1Streams,
    gpu_ns: u64,        // dispatch.zig timestamp delta
};

pub const L1Codec = struct {
    ctx: *driver.Context,
    cache: *descriptors.Cache,
    // Scratch buffers (lazy-grown to the largest src_size seen so far).
    src_buf: vk.VkBuffer = null,
    src_mem: vk.VkDeviceMemory = null,
    src_cap: usize = 0,
    lit_buf: vk.VkBuffer = null, lit_mem: vk.VkDeviceMemory = null,
    cmd_buf: vk.VkBuffer = null, cmd_mem: vk.VkDeviceMemory = null,
    off16_buf: vk.VkBuffer = null, off16_mem: vk.VkDeviceMemory = null,
    length_buf: vk.VkBuffer = null, length_mem: vk.VkDeviceMemory = null,
    sizes_buf: vk.VkBuffer = null, sizes_mem: vk.VkDeviceMemory = null,
    hash_buf: vk.VkBuffer = null, hash_mem: vk.VkDeviceMemory = null,
    dst_buf: vk.VkBuffer = null, dst_mem: vk.VkDeviceMemory = null,

    pub fn init(ctx: *driver.Context, cache: *descriptors.Cache) L1Codec { ... }
    pub fn deinit(self: *L1Codec) void { ... }

    /// Upload src_bytes to a device buffer, run the encode kernel, and
    /// read back the 4 sizes. The returned L1Streams.{lit,cmd,off16,length}_buf
    /// handles refer to device-resident buffers owned by `self`.
    /// Round-trip lifetime: encode → decode → next encode invalidates the
    /// streams. (Phase 1 has no concurrency; phase 2 may pool them.)
    pub fn encodeL1Sync(self: *L1Codec, src_bytes: []const u8) !EncodeResult { ... }

    /// Run the decode kernel against `streams` and copy decoded bytes
    /// into `dst_bytes`. `dst_bytes.len` must equal `original_size`.
    pub fn decodeL1Sync(
        self: *L1Codec,
        streams: L1Streams,
        original_size: usize,
        dst_bytes: []u8,
    ) !void { ... }
};
```

Internals:

1. Encode flow:
   - Ensure all scratch buffers ≥ `src_bytes.len * 2` (lit/cmd/off16/length
     worst case) + `hash_buf` = 512 KB.
   - Upload `src_bytes` into `src_buf` via a staging buffer + transfer
     submit.
   - `getOrCreate("lz_encode", tier, lz_encode_spv, n_storage=7,
     push_size=8)`.
   - `allocSet` with `{src, lit, cmd, off16, length, hash, sizes}`.
   - Push constants `[src_size:u32, hash_bits:u32 = 17]`.
   - `submitOne` with `group_count = [1, 1, 1]`.
   - Read back `sizes_buf` via staging.

2. Decode flow:
   - `getOrCreate("lz_decode", tier, lz_decode_spv, n_storage=5,
     push_size=20)`.
   - `allocSet` with `{cmd, lit, off16, length, dst}`.
   - Push constants `[cmd_size, lit_size, off16_count, length_used,
     original_size]` (5 × u32 = 20 B).
   - `submitOne` with `group_count = [1, 1, 1]`.
   - Copy `dst_buf` → `dst_bytes` via staging.

Round-trip test target:

```zig
test "L1 round-trip on canonical inputs" {
    var codec = L1Codec.init(&driver.g_default, &dispatch_cache);
    defer codec.deinit();
    inline for (.{ short, mid, long }) |bytes| {
        const r = try codec.encodeL1Sync(bytes);
        var out = try allocator.alloc(u8, bytes.len);
        defer allocator.free(out);
        try codec.decodeL1Sync(r.streams, bytes.len, out);
        try std.testing.expectEqualSlices(u8, bytes, out);
    }
}
```

---

## 10. Workgroup dispatch

- **One workgroup per chunk**, matching CUDA's
  `blockDim = (32, 1, 1)`, `grid = (n_chunks, 1, 1)`.
- Phase 1 = single chunk → `group_count = [1, 1, 1]`.
- `gl_WorkGroupID.x` plays the role of CUDA's `blockIdx.x` (= `chunk_id`).
  Hard-coded to 0 in phase 1; phase 2 will read it.
- `gl_SubgroupInvocationID` (range [0, 31]) plays the role of CUDA's
  `threadIdx.x & LANE_MASK` (= `lane`).
- `gl_LocalInvocationID.x` equals `gl_SubgroupInvocationID` under
  `local_size_x = 32` with `subgroupSize == 32` and full-subgroups
  required; we use `gl_SubgroupInvocationID` explicitly because that is
  the value the subgroup intrinsics treat as canonical.

---

## 11. Edge cases & gotchas

1. **`off16_buf` byte vs u16 addressing.** The CUDA encoder writes via
   `storeU16LE(off16_buf + count * 2, value)`, treating `off16_buf` as a
   byte array. The GLSL port keeps the same byte addressing (the SSBO is
   `uint8_t b[]`) and explicitly composes two `uint8_t` stores per
   off16 entry — equivalent to a `memcpy(&u16, dst, 2)` per the source's
   `storeU16LE`. Avoid the temptation to declare `Off16` as `uint16_t b[]`
   on the GLSL side; alignment requirements (offset multiple of 2) hold
   here but become brittle when phase 2 stitches multiple streams into a
   single buffer.

2. **Off16 hi/lo split is decoder-only and is NOT phase 1.** The
   `OFF16_SPLIT=true` instantiation of `decodeSubChunkRawMode` services
   entropy-coded off16 chunks (high byte plane + low byte plane).
   Phase 1 hard-codes `OFF16_SPLIT=false` (interleaved u16 LE); the
   encoder never produces a split. The GLSL port omits the
   `if constexpr (OFF16_SPLIT)` branch entirely.

3. **`__match_any_sync` is the *only* CUDA-specific intrinsic without a
   direct GLSL equivalent.** Every other warp primitive maps 1:1 to
   `KHR_shader_subgroup_*`. The emulation in §7 is correct but ~32×
   more shuffles than CUDA's hardware-accelerated instruction. Phase 1
   accepts the slowdown; profiling at phase 4 decides whether to optimize.

4. **No off32 stream in phase 1 ⇒ encoder must forbid all far-offset
   tokens.** The decoder we're porting (raw mode) does not consume an
   off32 stream. To prevent the encoder from ever emitting tokens
   `{2, 3..23}`, the GLSL `emitCmd` path takes one of two implementation
   stances:
   - **(chosen)** Replace the far-offset branch of `emitCmd` with a
     **reject-and-fall-back** path: when the parser sees
     `resolved_off > NEAR_OFFSET_MAX`, it skips the match entirely
     (advance `pos = match_pos + 1` and continue). This matches the
     existing `FAR_OFFSET_MIN_MATCH` reject pattern (lz_greedy_parser.cuh
     L289) but extends to *all* match lengths, not just short ones.
   - The encoder still passes the legal CPU/CUDA stream-shape (no off32
     ever produced), so the decoder can stay raw-mode forever.

   Phase 2 will lift this restriction by porting `decodeSubChunkGeneral`
   alongside an off32 SSBO.

5. **`enable_match_rehash` is dead at L1.** The CUDA call site passes
   `l4_features = 0` when level ≤ 3. The GLSL port keeps the branch (with
   the §7C call site) inside `if (enable_match_rehash != 0u)`. Push
   constant is hard-zero; branch is statically uniform across the warp;
   no correctness concern but the compiler can't dead-strip it (could be
   lifted via spec constant in phase 4).

6. **`emitWithLiteral1` is NOT used.** `scanBlock` always calls plain
   `emitCmd`, never `emitWithLiteral1` (that helper is invoked only by
   the chain parser and the CPU CPU path, not by `scanBlock`).
   The GLSL port omits it entirely.

7. **`HASH_EMPTY = 0xFFFFFFFFu`, not 0.** Naively zeroing the buffer
   (Vulkan's default for fresh allocations) would falsely report position
   0 as a hit. The shader does the explicit fill at the top of `main()`.

8. **Backward extension can fail to terminate cleanly across lane 0.**
   The CUDA source guards with `lane == 0` and uses a serial loop; the
   GLSL port does the same. Lane 0 is the only writer; other lanes wait
   on the `subgroupBroadcast(bw_steps, 0)` for the result. No divergence
   issue at subgroup level.

9. **`writeLengthValue`'s tag byte underflows intentionally**
   (`(low2 - 4) & 0xFF` wraps in CPU/CUDA `uint32_t` arithmetic). GLSL
   `uint` arithmetic is also wrapping mod 2^32, so the port is a direct
   transliteration — no special handling needed.

10. **`subgroupShuffle(value, lane)` is undefined for `lane >= subgroupSize`
    on some backends.** The CUDA code at site §6.1.5 (line 108) guards
    against this with `key_src_lane = (lower_same_bucket != 0u) ? top :
    lane` — same trick the GLSL port uses (self-shuffle when no lower
    same-bucket lane exists). All other shuffle call sites pass values in
    `[0, 31]` by construction (`__ffs - 1`, `lastBitSet`, or a literal).

11. **`subgroupBarrier()` does *not* imply a memory barrier.** Sites
    where a `__syncwarp` ordered cross-lane reads of a shared buffer
    must additionally call `subgroupMemoryBarrierBuffer()` (or
    `memoryBarrierBuffer()`). The two hot sites are the hash-table
    write→read boundary in `scanBlock` (entered every loop iter) and the
    `dst` buffer write→read between cooperative copies in the decoder
    (every literal/match copy pair). Listed explicitly in §6.1.19 and
    §6.2.10.

12. **Subgroup size must be exactly 32.** Devices that report
    `subgroupSize` of 16 (Intel) or 64 (some AMD) cannot run this codec
    unaltered — every shuffle constant, every `__match_any` loop, every
    `__ffs/__clz` guard is hard-coded to 32. Probe rejects at startup;
    phase 4 may add a 16/64 path but not phase 1.
