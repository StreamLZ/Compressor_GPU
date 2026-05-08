# nvCOMP LZ4 GPU Kernel Architecture

Reverse-engineered from `nvcomp64_5.dll` v5.2, CUDA 13 build.
Disassembled via `cuobjdump --extract-elf` + `nvdisasm` on the sm_89 (Ada Lovelace) cubin.
Profiled with `ncu` on an RTX 4060 Ti (16 GB, CC 8.9).

Source files:
- SASS listing: `C:\tmp\nvcomp_test\lz4_decompress_sm89.sass` (extracted from `Program.236.sm_89.cubin`)
- Profile report: `C:\tmp\nvcomp_test\profile.ncu-rep`
- Resource usage dump: `C:\tmp\nvcomp_cubins\resource_usage.txt`
- Test harness: `C:\tmp\nvcomp_test\main.cu`

All SASS line references below are to `lz4_decompress_sm89.sass`.

---

## 1. Kernel Identity and Launch Configuration

There are two template instantiations of the decompress kernel:

```
lz4DecompressBatchKernel<true>   — with correctness checking
lz4DecompressBatchKernel<false>  — without correctness checking (production path)
```

This document analyzes `<false>` (the production path), beginning at line 26057 of the SASS listing.

**Profiled launch parameters** (4 MB input, 64 × 64 KB chunks):

| Parameter | Value |
|---|---|
| Grid | (32, 1, 1) |
| Block | (32, 2, 1) = 64 threads |
| Warps per block | 2 |
| Registers per thread | 48 (allocated 48) |
| Static shared memory | 1792 bytes per block (896 per warp) |
| Dynamic shared memory | 0 |
| Chunks processed | 64 (2 per block × 32 blocks) |

Key insight: **one warp (32 threads) handles one chunk**. `blockDim.y = 2` packs two independent warps into one block to share an SM and improve occupancy. There is no cooperation between the two warps — they decode entirely separate chunks.

---

## 2. Shared Memory Layout

Each warp gets 896 bytes of shared memory, addressed via a per-warp base register `R5`:

```
R5 = threadIdx.y * 0x380    (0x380 = 896 decimal)
```

The layout within each warp's 896-byte region:

| Offset | Size | Purpose |
|---|---|---|
| `[R5 + 0x000]` – `[R5 + 0x0FF]` | 256 bytes | Compressed data staging window |
| `[R5 + 0x100]` – `[R5 + 0x10B]` | 12 bytes | Per-token work items (offset, match_length, literal_length) |
| `[R5 + 0x280]` – `[R5 + 0x283]` | 4 bytes | Compressed stream read cursor (per-lane, via `R9 = lane * 4 + R5`) |
| `[R5 + 0x300]` – `[R5 + 0x37F]` | 128 bytes | Per-lane output write cursor (32 lanes × 4 bytes) |

The compressed data window at offset 0x000 holds 256 bytes — one "page" of compressed input. When the decode cursor advances past this window, a refill cycle reloads the next 256 bytes (see §3).

The per-lane output cursor at offset 0x300 tracks where each lane would next write in the output buffer. This is used for the cooperative match copy and the warp-wide binary search (§6).

---

## 3. Phase 1: Compressed Data Staging (Bulk Load)

**SASS location**: lines 26059–26173 (offsets `0x000`–`0x720`)

Before token parsing begins, and again whenever the decode cursor crosses a 256-byte boundary, the entire warp cooperates to load compressed data from global memory into shared memory.

### Fast path (8-byte loads)
When the remaining compressed data is large enough (≥ 256 bytes):

```
R26 = base_ptr + lane_id * 8 + window_offset    (global address)
LD.E.64 R26, [R26.64]                            (8-byte global load)
R17 = R5 + lane_id * 8                           (shared memory address)
STS.64 [R17], R26                                (8-byte shared store)
```

32 lanes × 8 bytes = 256 bytes loaded in a single coalesced transaction.
This is at SASS lines 26219–26224 (offsets `0x9c0`–`0xa10`).

### Slow path (byte-at-a-time loads)
When near the end of the compressed stream, it falls back to predicated single-byte loads:

```
LD.E.U8 R13, [R6.64]           ; load byte at lane's global offset
STS.U8 [R12], R13              ; store to shared memory
LD.E.U8 R15, [R6.64+0x20]     ; next byte, +32 stride
STS.U8 [R12+0x20], R15
... (8 iterations per lane, covering 256 bytes)
```

Each lane loads 8 bytes at stride 0x20 (32), so 32 lanes × 8 bytes = 256 bytes.
Bounds checking is done per-byte via `ISETP.GE` against the compressed size `R2`.
This is at SASS lines 26120–26172 (offsets `0x3d0`–`0x710`).

After staging completes, a `BSYNC` barrier ensures all lanes see the data before parsing begins.

---

## 4. Phase 2: Serial Token Parsing

**SASS location**: lines 26291–26339 (offsets `0xdf0`–`0x1050`)

LZ4 tokens are parsed **serially on a single lane** (whichever lane's turn it is — tracked via a counter `R21`). The rest of the warp waits or assists with data movement.

### Token byte decode

```
LDS.U8 R19, [R19]                    ; load token byte from shared memory
SHF.R.U32.HI R27, RZ, 0x4, R19      ; literal_length = token >> 4
PRMT R17, R27, 0x9910, RZ            ; zero-extend to check if == 0xF
ISETP.NE.AND P1, PT, R17, 0xf, PT   ; if literal_length != 15, skip extension
```

This is standard LZ4: the top nibble is literal length, bottom nibble (extracted later at offset `0x11e0`) is match length minus 4.

### Varint length extension

When literal_length == 15, the kernel enters a byte-at-a-time extension loop:

```
.L_x_545:
    LDS.U8 R26, [R26]                     ; load next byte from shared memory
    ISETP.EQ.AND P2, PT, R26, 0xff, PT    ; is it 255?
    IMAD.IADD R17, R26, 0x1, R17          ; accumulate into total length
    PRMT R11, R26, 0x7610, R11            ; save last byte
    @!P1 BRA P2, `(.L_x_545)             ; loop while byte == 255 AND within window
```

If the extension bytes cross the 256-byte shared memory window boundary, it falls through to a second loop (`.L_x_548`, offsets `0xfb0`–`0x1010`) that reads directly from global memory via `LD.E.U8`:

```
.L_x_548:
    IADD3 R26, P1, R22, R30, RZ          ; compute global address
    LD.E.U8 R26, [R26.64]                ; load from global memory
    ISETP.NE.AND P1, PT, R26, 0xff, PT   ; check for terminator
    IMAD.IADD R17, R26, 0x1, R17         ; accumulate
    @!P1 BRA `(.L_x_548)                 ; loop while byte == 255
```

The same two-tier approach (shared memory first, global memory fallback) applies to match length extension (`.L_x_557` and `.L_x_560`, offsets `0x12c0`–`0x1400`).

### Offset decode

After literal length parsing, the 16-bit match offset is loaded from the compressed stream:

```
LD.E.U8 R6, [R26.64]         ; low byte
LD.E.U8 R29, [R26.64+0x1]    ; high byte
PRMT R6, R29, 0x7604, R6     ; combine: offset = (high << 8) | low
```

The `PRMT` (byte permute) instruction assembles the 16-bit little-endian offset in a single cycle rather than using shifts and ORs.

---

## 5. Phase 3: Work Distribution via Shared Memory

**SASS location**: lines 26343–26354 (offsets `0x10a0`–`0x1110`)

After parsing a token, the active lane publishes the work item to shared memory so other lanes can participate:

```
@!P1 STS [R9+0x280], R30      ; compressed stream cursor → smem[0x280]
@!P1 STS [R28+0x108], R27     ; literal/match length → smem[0x108]
@!P1 STS [R9+0x300], R6       ; output write cursor → smem[0x300]
```

The kernel processes tokens in batches. Register `R21` counts how many tokens the current "active lane" has processed. When `R21 & 0xFF >= 0x20` (32 tokens), the active lane rotates:

```
LOP3.LUT R17, R21, 0xff, RZ, 0xc0, !PT  ; token_count & 0xFF
ISETP.GE.U32.AND P1, PT, R17, 0x20, PT  ; >= 32?
@!P1 BRA P2, `(.L_x_563)                ; continue if not
```

This rotation prevents any single lane from becoming a serialization bottleneck. After 32 tokens, the kernel yields to the cooperative literal/match copy phases before the next lane takes over parsing.

---

## 6. Phase 4: Cooperative Literal Copy

**SASS location**: lines 26436–26541 (offsets `0x1590`–`0x1bd0`)

Once a token's literal length is known (broadcast from the parsing lane via shared memory), the warp cooperates to copy literal bytes from the compressed stream to the output.

### Step 1: Broadcast work from the active lane

Each lane reads the work item from shared memory:

```
@!P3 LDS R17, [R28+0x100]    ; match offset
@!P3 LDS R19, [R28+0x104]    ; match/literal length
@!P3 LDS R34, [R28+0x108]    ; literal length for this token
```

### Step 2: Short literal copy (< 16 bytes)

For short literals, only the active lane copies byte-by-byte:

```
LD.E.U8 R43, [R28.64]        ; load from compressed stream (global)
ST.E.U8 [R36.64], R43        ; store to output (global)
```

This loop runs at offsets `0x17a0`–`0x19b0` and is unrolled by 4 with a countdown:

```
.L_x_571:
    LD.E.U8 R43, [R28.64]          ; byte 0
    ST.E.U8 [R36.64], R43
    LD.E.U8 R45, [R28.64+0x1]      ; byte 1
    ST.E.U8 [R32.64], R45
    LD.E.U8 R44, [R28.64+0x2]      ; byte 2
    ST.E.U8 [R26.64], R44
    LD.E.U8 R29, [R28.64+0x3]      ; byte 3
    ST.E.U8 [R36.64], R29
    IADD3 R41, R41, -0x4, RZ       ; countdown -= 4
    @P2 BRA `(.L_x_571)
```

Between each byte, it reloads and updates the output cursor from shared memory (`LDS/STS [R9+0x300]`), maintaining the sequential output position across all lanes.

### Step 3: Long literal copy (≥ 16 bytes)

For longer literals, the warp fans out with each lane taking bytes at a stride:

```
IMAD.IADD R45, R20, 0x1, R43     ; per-lane offset = threadIdx.x * 4 + base
LD.E R28, [R28.64]                ; 4-byte load from compressed stream
PRMT R45, R28, 0x7770, RZ        ; extract byte 0
PRMT R42, R28, 0x7771, RZ        ; extract byte 1
PRMT R44, R28, 0x7772, RZ        ; extract byte 2
ST.E.U8 [R26.64], R45            ; store byte 0
ST.E.U8 [R26.64+0x1], R42        ; store byte 1
ST.E.U8 [R26.64+0x2], R44        ; store byte 2
ST.E.U8 [R26.64+0x3], R29        ; store byte 3
```

Each lane loads 4 bytes (one `LD.E` = 32-bit load) and scatters them via `PRMT` byte-extract + individual `ST.E.U8` stores. The stride between lanes is 0x80 (128 bytes), so 32 lanes × 4 bytes = 128 bytes per iteration. This loop is at offsets `0x1d80`–`0x1f70`.

When the compressed input cursor has run past the shared memory window but there's still data to copy, it falls back to per-lane single-byte loads with a stride of 0x20 (32):

```
.L_x_586:
    LD.E.U8 R29, [R28.64]       ; load one byte per lane from global memory
    ST.E.U8 [R26.64], R29       ; store one byte per lane to output
    IADD3 R43, R43, 0x20, RZ    ; advance by warp width (32)
    @!P1 BRA `(.L_x_586)
```

This is at offsets `0x1fd0`–`0x2060`.

---

## 7. Phase 5: Cooperative Match Copy via Warp Vote/Shuffle

**SASS location**: lines 26545–26643 (offsets `0x1bf0`–`0x2130`)

This is the most architecturally interesting section. After literal copy, any lane that has a pending match copy (match_length > 0 stored in `R34`) needs to execute it. But match copies must be serialized because they can overlap (a match at offset < match_length requires byte-at-a-time repeat copy). The kernel uses warp-level voting and shuffling to coordinate.

### Step 1: Find lanes with pending work

```
ISETP.NE.AND P1, PT, R34, RZ, PT    ; does this lane have remaining match bytes?
VOTE.ANY R35, PT, P1                 ; R35 = bitmask of lanes with work
```

`R35` is now a 32-bit mask where bit N is set if lane N has pending match copy work.

### Step 2: Select highest-priority lane

```
.L_x_587:
    BREV R36, R35                    ; bit-reverse the mask
    FLO.U32.SH R36, R36             ; find leading one → lane index
```

`BREV` + `FLO.U32.SH` together find the **lowest-numbered** set bit (highest priority lane). This selects which lane's match copy to execute next.

### Step 3: Broadcast that lane's parameters to all lanes

```
SHFL.IDX PT, R38, R34, R36, 0x1f    ; R38 = selected lane's match_length
LDS R29, [R9+0x280]                  ; load compressed cursor
LDS R41, [R9+0x300]                  ; load output cursor
SHFL.IDX PT, R37, R29, R36, 0x1f    ; R37 = selected lane's src offset
SHFL.IDX PT, R41, R41, R36, 0x1f    ; R41 = selected lane's dst offset
```

`SHFL.IDX` reads register `R34` from lane `R36` and writes the result to `R38` in all lanes. This is a single-cycle broadcast across the warp.

### Step 4: Cooperative match copy

With all lanes knowing the source offset, destination, and length, they cooperate:

**Wide path** (match_length ≥ 4, offset ≥ 4 — no overlap):

Each lane handles bytes at offset `threadIdx.x * 4`, loading 4 bytes and scattering:

```
LD.E R28, [R28.64]               ; 4-byte load from match source
PRMT R45, R28, 0x7770, RZ        ; extract byte 0
PRMT R42, R28, 0x7771, RZ        ; extract byte 1  
PRMT R44, R28, 0x7772, RZ        ; extract byte 2
PRMT R29, R28, 0x7773, RZ        ; extract byte 3
ST.E.U8 [R26.64], R45
ST.E.U8 [R26.64+0x1], R42
ST.E.U8 [R26.64+0x2], R44
ST.E.U8 [R26.64+0x3], R29
```

Stride per iteration: 0x80 (128 bytes). This is at offsets `0x1e00`–`0x1f50`.

**Narrow path** (match_length or offset too small for parallel copy):

Falls back to single-lane byte copy — only lane 0 (or the selected lane) runs:

```
.L_x_586:
    LD.E.U8 R29, [R28.64]
    ST.E.U8 [R26.64], R29
    IADD3 R43, R43, 0x20, RZ
    @!P1 BRA `(.L_x_586)
```

### Step 5: Clear completed lane from mask, repeat

```
IMAD.MOV.U32 R27, RZ, RZ, 0x1
SHF.L.U32 R36, R27, R36, RZ      ; R36 = 1 << selected_lane
LOP3.LUT R35, R36, R35, RZ, 0x3c ; R35 = R35 XOR R36 (clear that bit)
ISETP.NE.AND P1, PT, R35, RZ, PT ; any lanes left?
@P1 BRA `(.L_x_587)              ; loop back to process next lane
```

`LOP3.LUT` with constant `0x3c` = XOR operation. This clears the bit for the just-processed lane, and the loop continues until all lanes' match copies are done.

This is a **warp-sequential scan** — match copies are processed one lane at a time across the warp, but each individual copy can use all 32 lanes for the actual data movement.

---

## 8. Phase 6: Multi-Lane Output Alignment and Binary Search

**SASS location**: lines 26651–26718 (offsets `0x2180`–`0x2580`)

After the literal+match copy for a batch of tokens, the kernel needs to advance the output cursor to account for all the work done. Because multiple lanes may have been producing output in parallel, it uses a binary search through the per-lane output positions stored in shared memory at offset `0x300`.

### The alignment step

```
LDS R28, [R13+0x300]               ; lane N's output position
LDS R38, [R5+0x300]                ; lane 0's output position
IMAD.IADD R37, R28, 0x1, -R17     ; adjusted position
LOP3.LUT R26, R26, 0x3, RZ, 0xc0  ; check 4-byte alignment
IADD3 R26, -R26, 0x4, RZ          ; bytes to next 4-byte boundary
IMNMX.U32 R34, R19, R26, PT       ; min(remaining, alignment_padding)
```

This computes how many bytes need to be written to reach a 4-byte-aligned output position, then selects the smaller of (remaining bytes) and (alignment padding) for the next copy.

### The binary search

The binary search finds which lane's output cursor is closest to a target position:

```
SHF.R.U32.HI R38, RZ, 0x1, R31   ; mid = (lo + hi) / 2
IMAD R29, R38, 0x4, R5            ; smem address = mid * 4 + base
LDS R27, [R29+0x300]              ; load mid lane's output position
ISETP.GT.AND P2, PT, R27, R26, PT ; compare with target
SEL R31, R38, R31, P2             ; update hi or lo
SEL R38, R38, RZ, !P2             ; update the other bound
```

This repeats 5 times (offsets `0x22f0`–`0x2580`), halving the range each time: 32 → 16 → 8 → 4 → 2 → 1. Five iterations of binary search = log₂(32) = exact coverage of all 32 lanes.

The purpose: when the warp has been processing multiple tokens where different lanes were producing output, the binary search determines which lane "owns" a given output byte position. This is needed to resolve the sequential output ordering — even though match copies were done cooperatively, the output stream must be strictly ordered.

After the binary search identifies the owning lane, `SHFL.IDX` broadcasts that lane's data to all lanes for the next cooperative operation:

```
SHFL.IDX PT, R41, R19, R40, 0x1f   ; broadcast from found lane
```

---

## 9. Phase 7: Warp Synchronization Primitives

The kernel uses four internal helper functions, called via `CALL.REL.NOINC` (a non-inlined function call within the same cubin):

| Function | Offset | Purpose |
|---|---|---|
| `shflsync_idx_p` | `0x4690` | `WARPSYNC` + `SHFL.IDX` — synchronized shuffle |
| `votesync_any` | `0x46d0` | `WARPSYNC` + `VOTE.ANY` — synchronized vote |
| `votesync_ballot` | `0x4740` | `WARPSYNC` + `VOTE.ANY` + bitmask — ballot |
| `warpsync` | `0x47a0` | `WARPSYNC` only — barrier |

These are called at warp reconvergence points (after divergent branches) to ensure all lanes are synchronized before the next cooperative operation. On sm_89, `WARPSYNC` is a hardware barrier that ensures all lanes in the warp have reached the same point.

The `votesync_any` helper also includes error propagation logic:

```
ISETP.NE.U32.AND P1, PT, R27, 0x1, PT   ; check error flag
WARPSYNC 0xffffffff
VOTE.ANY R27, PT, !P1                    ; any lane has error?
SEL R29, RZ, 0x1, !P1                    ; propagate to all lanes
```

---

## 10. Overall Decode Flow (Summary)

```
┌─────────────────────────────────────────────────────────┐
│ KERNEL LAUNCH: 1 warp per chunk, 2 warps per block      │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 1. BULK LOAD: 32 lanes load 256 bytes of compressed     │
│    data from global memory → shared memory              │
│    (coalesced 8-byte loads, or predicated byte loads     │
│    near end of stream)                                   │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 2. SERIAL TOKEN PARSE: one lane reads token byte from   │
│    shared memory, extracts literal_length (top nibble)   │
│    and match_length (bottom nibble), extends via varint  │
│    loop if == 15/255                                     │
│                                                          │
│    Publishes {offset, lit_len, match_len} to shared mem  │
│    After 32 tokens, rotates to next lane                 │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 3. COOPERATIVE LITERAL COPY                              │
│    Short (<16B): single lane, unrolled × 4               │
│    Long (≥16B): all 32 lanes, stride 128B per iter,      │
│    using LD.E (4B) + PRMT byte extract + ST.E.U8         │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 4. COOPERATIVE MATCH COPY                                │
│    a) VOTE.ANY → find lanes with pending matches         │
│    b) BREV+FLO → select lowest-numbered lane             │
│    c) SHFL.IDX → broadcast src/dst/len to all lanes      │
│    d) All lanes copy in parallel (4B per lane per iter)   │
│    e) XOR to clear done lane, repeat until mask == 0     │
│                                                          │
│    Falls back to single-lane byte copy for overlapping   │
│    matches (offset < match_length)                       │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 5. OUTPUT POSITION RECONCILIATION                        │
│    Binary search (5 iters, log₂32) through per-lane      │
│    output cursors in shared memory to determine lane     │
│    ownership of output positions                         │
│    SHFL.IDX to broadcast from owning lane                │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 6. WINDOW REFILL CHECK                                   │
│    If decode cursor crossed 256-byte boundary,           │
│    branch back to step 1 for next window                 │
│    Otherwise loop back to step 2 for next token          │
└─────────────────────────────────────────────────────────┘
```

---

## 11. Key Architectural Observations

### What the GPU buys you

The GPU's advantage is NOT within-chunk parallelism for token parsing — that remains strictly serial, exactly as on CPU. The advantages are:

1. **Chunk-level parallelism**: With 64KB chunks on a 4 MB file, there are 64 independent decode streams. The RTX 4060 Ti has 34 SMs, each capable of running multiple warps concurrently. All 64 chunks decode simultaneously.

2. **Memory bandwidth for copies**: During literal and match copies, the warp fans out across 32 lanes. A single match copy of 128 bytes completes in one iteration (32 lanes × 4 bytes). On CPU, this would be one `memcpy` call or a few SIMD iterations.

3. **Latency hiding via occupancy**: While one warp stalls on a global memory load, other warps on the same SM continue executing. The serial token parse generates many dependent loads (load token → compute offset → load match data), each with ~hundreds of cycles of latency. High occupancy hides this.

### What the GPU does NOT do

1. **No speculative parallel token parsing**: The kernel does not attempt to guess where token boundaries are or speculatively decode multiple tokens simultaneously within a chunk. Token parsing is strictly serial.

2. **No shared-memory decompression buffer**: The output is written directly to global memory, not staged through shared memory. Only the *compressed* input is cached in shared memory. This means match copies that reference recently-written output must read from global memory (potentially hitting L1/L2 cache).

3. **No hardware decompression engine usage in this path**: The `lz4DecompressBatchKernel` is the CUDA software path. The hardware decompression engine (available on Hopper/Ada) is dispatched through a separate code path using `CUmemDecompressParams_st` (visible in the same cubin as `fill_cudecomp_params` and `init_cudecomp_sort` kernels).

### Performance-critical patterns

1. **PRMT for byte manipulation**: The `PRMT` (permute) instruction is used extensively for byte extraction from 32-bit loads. `PRMT R, src, 0x7770, RZ` extracts byte 0, `0x7771` extracts byte 1, etc. This is a single-cycle operation that replaces what would otherwise be shift+mask sequences.

2. **Two-tier memory access**: The shared memory window acts as an L0 cache for compressed data. Token parsing reads from shared memory (LDS = ~30 cycles latency) rather than global memory (LDG = ~hundreds of cycles). When the window is exhausted, a cooperative refill loads the next 256 bytes.

3. **Predicated execution vs. branching**: Many operations use per-lane predicates (`@P1`, `@!P2`) instead of branches, avoiding warp divergence. Bounds checks against the compressed size are particularly careful to use predicates.

4. **Lane rotation for fairness**: The 32-token batch limit per active lane prevents one lane from monopolizing the serial decode path. This is important because while one lane parses tokens, the other 31 lanes are idle (or participating in copies).

### Resource utilization

- **48 registers** per thread is moderate for sm_89 (max 255). This allows reasonable occupancy — the register file can support `65536 / (48 * 64) = 21` blocks per SM (limited by `max_blocks_per_multiprocessor = 24`).
- **1792 bytes shared memory** per block is very small (max 101 KB on sm_89). Shared memory is not the occupancy limiter.
- **The occupancy limiter is register count**: `launch__occupancy_limit_registers = 20` blocks (from the profile), meaning registers constrain occupancy more than shared memory (23 blocks) or block size (24 blocks).

---

## 12. Appendix: Instruction Reference

Key SASS instructions used in this kernel:

| Instruction | Meaning |
|---|---|
| `LDG.E.64` | Load 8 bytes from global memory |
| `LD.E.U8` | Load 1 byte from global memory |
| `LD.E` | Load 4 bytes from global memory |
| `LDS.U8` | Load 1 byte from shared memory |
| `LDS` | Load 4 bytes from shared memory |
| `STS.64` | Store 8 bytes to shared memory |
| `STS.U8` | Store 1 byte to shared memory |
| `STS` | Store 4 bytes to shared memory |
| `ST.E.U8` | Store 1 byte to global memory |
| `PRMT` | Byte permute (extract/rearrange bytes within registers) |
| `SHFL.IDX` | Warp shuffle — read a register from a specific lane |
| `VOTE.ANY` | Warp vote — returns bitmask of lanes where predicate is true |
| `BREV` | Bit reverse |
| `FLO.U32.SH` | Find leading one (with shift) — used to find lowest set bit after BREV |
| `WARPSYNC` | Warp barrier synchronization |
| `BSSY` / `BSYNC` | Block-level synchronization (convergence barrier setup/wait) |
| `S2R` | Special register read (thread ID, block ID) |
| `SHF.R.U32.HI` | Funnel shift right (used for division by power of 2) |
| `LOP3.LUT` | 3-input logic operation with lookup table (AND, OR, XOR, etc.) |
| `ISETP` | Integer set predicate (comparison → predicate register) |
| `IMAD` | Integer multiply-add (also used for MOV, ADD via identity operands) |
| `IADD3` | 3-operand integer add (with optional carry in/out) |
| `IMNMX` | Integer min/max |
| `SEL` | Select (conditional move based on predicate) |
| `LEA` | Load effective address (scaled add for pointer arithmetic) |
| `MATCH.ANY` | Warp-wide value match — returns bitmask of lanes holding the same value |
| `REDUX.OR` | Warp-wide uniform reduction (OR across lanes → uniform register) |
| `R2UR` | Move register to uniform register (predicate conversion) |
| `STG.E.U8` | Store 1 byte to global memory |
| `STG.E.U16` | Store 2 bytes to global memory |
| `STG.E.64` | Store 8 bytes to global memory |
| `SHFL.DOWN` | Warp shuffle — read from a lane N positions below |
| `PLOP3.LUT` | Predicate logic operation (3-input, lookup table) |

---

# nvCOMP LZ4 GPU Compress Kernel Architecture

Same cubin as the decompress kernel (`Program.236.sm_89.cubin`).
The compress kernel SASS begins at line 30515 of `lz4_decompress_sm89.sass`.

There are six template instantiations, varying by hash entry type and bitshuffle:

| Variant | Template | Hash entry | Bitshuffle |
|---|---|---|---|
| `<h, 0>` | `unsigned char, false` | 8-bit | No |
| `<h, 1>` | `unsigned char, true` | 8-bit | Yes |
| `<t, 0>` | `unsigned short, false` | 16-bit | No |
| `<t, 1>` | `unsigned short, true` | 16-bit | Yes |
| `<j, 0>` | `unsigned int, false` | 32-bit | No |
| `<j, 1>` | `unsigned int, true` | 32-bit | Yes |

The profiled run used `<h, 0>` (8-bit hash entries, no bitshuffle) for 64 KB chunks.
This document analyzes that variant. SASS line references are to the same file as the decompress analysis.

---

## 13. Compress Kernel Launch Configuration

**Profiled launch parameters** (4 MB input, 64 × 64 KB chunks):

| Parameter | Value |
|---|---|
| Grid | (64, 1, 1) |
| Block | (32, 1, 1) = 32 threads |
| Warps per block | 1 |
| Registers per thread | 44 (allocated 48) |
| Shared memory | 0 bytes |
| Chunks processed | 64 (1 per block) |

Key difference from decompress: **one warp per block**, not two. And **zero shared memory** — the compress kernel operates entirely on global memory and registers. There is no shared-memory staging buffer.

---

## 14. Phase 1: Hash Table Initialization

**SASS location**: lines 30536–30622 (offsets `0x0130`–`0x0620`)

Before compression begins, each lane initializes its portion of the hash table. The hash table lives in **global memory**, not shared memory, at an address computed via:

```
R10 = chunk_id * 0x4000 + lane_offset    (0x4000 = 16384 entries per chunk)
LEA R4, P0, R10, c[0x0][0x180], 0x1     (hash_table_base + offset * 2)
```

The `* 0x4000` stride and the `LEA ... 0x1` (shift by 1 = multiply by 2) confirm the hash table uses **16-bit entries** (U16) with **16384 slots per chunk** (16K × 2 bytes = 32 KB per chunk).

Each entry is initialized to `0xFFFF` (sentinel for "no match"):

```
IMAD.MOV.U32 R3, RZ, RZ, 0xffff
STG.E.U16 [R14.64], R3          ; store 0xFFFF
STG.E.U16 [R20.64], R3          ; +32 entries
STG.E.U16 [R22.64], R3          ; +64 entries
STG.E.U16 [R24.64], R3          ; +96 entries
```

The init loop is unrolled ×4, advancing by 128 entries (0x80) per iteration across the 32 lanes:
- 32 lanes × 4 stores × stride of 0x20 = 32 lanes × 4 = 128 entries per iteration
- Loop continues until all 16384 entries are cleared

The loop at `.L_x_974` (offsets `0x0290`–`0x0440`) handles the bulk, with a tail loop at `.L_x_976` handling the remainder.

---

## 15. Phase 2: Hash Computation via SHFL.DOWN

**SASS location**: lines 30659–30678 (offsets `0x0840`–`0x0930`)

The hash is computed from 4 consecutive input bytes using a clever warp-shuffle trick:

### Step 1: Each lane loads one byte at its position

```
LDG.E.U8 R13, [R10.64]    ; lane N loads input[position + N]
```

### Step 2: Combine 4 bytes into a 32-bit word using SHFL.DOWN

```
SHFL.DOWN PT, R10, R13, 0x1, 0x1f     ; get byte from lane N+1
IMAD.SHL.U32 R10, R10, 0x100, RZ      ; shift left by 8
LOP3.LUT R13, R10, R13, RZ, 0xfc      ; OR with current byte → 2 bytes

SHFL.DOWN PT, R10, R13, 0x2, 0x1f     ; get 2 bytes from lane N+2
IMAD.U32 R10, R10, 0x10000, RZ        ; shift left by 16
LOP3.LUT R32, R13, R10, RZ, 0xfc      ; OR → 4 bytes in R32
```

After this sequence, each lane holds a 32-bit word containing `input[pos+N..pos+N+3]` — four consecutive bytes starting at that lane's position. This is a **rolling 4-byte window** constructed entirely from warp shuffles, avoiding redundant global memory loads.

The `LOP3.LUT ... 0xfc` = OR operation. `IMAD.SHL.U32 R10, R10, 0x100` = shift left by 8 bits (multiply by 256). `IMAD.U32 R10, R10, 0x10000` = shift left by 16 bits.

---

## 16. Phase 3: Warp-Wide Match Finding via MATCH.ANY

**SASS location**: lines 30687–30698 (offsets `0x09b0`–`0x0a40`)

This is the most novel part of the compress kernel. Instead of a traditional hash table probe, the kernel uses the hardware `MATCH.ANY` instruction for initial duplicate detection:

```
MATCH.ANY R12, R32     ; R12 = bitmask of lanes whose R32 equals this lane's R32
```

`MATCH.ANY` is a **single-cycle warp-wide comparison** that returns, for each lane, a bitmask of all other lanes holding the same 32-bit value. This effectively finds all positions within the current 32-byte window that share the same 4-byte prefix — i.e., potential LZ4 matches.

### Active lane masking

Before the MATCH instruction, the kernel computes which lanes are "active" (within bounds):

```
IADD3 R28, R0, -0xc, -R27          ; remaining = input_size - 12 - position
IMNMX R28, R28, 0x1d, PT           ; min(remaining, 29) — cap at 29 active lanes
ISETP.GE.U32.AND P1, PT, R19, R28  ; is this lane beyond the active range?

IADD3 R10, -R28, 0x20, RZ          ; inactive_count = 32 - active_lanes
SHF.R.U32.HI R33, RZ, R10, R11    ; create bitmask of active lanes
REDUX.OR UR6, R33                  ; uniform reduce: UR6 = active lane mask
R2UR P0, URZ, R33                  ; convert to predicate
@!P0 BRA.DIV UR6, `(.L_x_987)     ; divergent branch: only active lanes execute MATCH
MATCH.ANY R12, R32                 ; find matching lanes
```

The `REDUX.OR` + `R2UR` + `BRA.DIV` sequence ensures only lanes within the valid input range participate in the match, preventing false matches from uninitialized data in the last few lanes.

### Processing match results

After MATCH.ANY:

```
BREV R10, R12                       ; bit-reverse the match mask
FLO.U32 R10, R10                    ; find leading one → lowest matching lane
IADD3 R38, -R10, 0x1f, RZ           ; convert to lane index
ISETP.NE.AND P0, PT, R38, R19, P0   ; exclude self-matches (lane != self)
VOTE.ANY R10, PT, P0                ; any lane found a non-self match?
```

The kernel finds the **lowest-numbered lane** with a matching 4-byte value. Self-matches (where a lane matches itself) are excluded. If any non-self match exists, the kernel proceeds to validate and extend it.

---

## 17. Phase 4: Hash Table Probe (Fallback Path)

**SASS location**: lines 30716–30751 (offsets `0x0b30`–`0x0d60`)

When MATCH.ANY finds no match within the current 32-byte warp window, the kernel falls back to the traditional hash table:

### Hash index computation

```
BREV R11, R32                                ; bit-reverse the 4-byte value
LOP3.LUT R10, R32, 0xc375, RZ, 0x3c         ; XOR with magic constant 0xc375
IMAD.IADD R11, R10, 0x1, R11                ; combine: hash = bitrev(val) + (val ^ 0xc375)
LOP3.LUT R12, R11, R22, RZ, 0xc0            ; hash & table_mask (R22 = table_size - 1)
IMAD.WIDE R14, R17, 0x4000, R12             ; hash_table_base + chunk_id * 16K + hash
LEA R10, P3, R14, c[0x0][0x180], 0x1        ; address = base + offset * 2 (U16 entries)
```

The hash function uses `BREV` (bit reverse) + XOR with `0xc375` + addition. This is a cheap multiplicative-style hash that distributes well across the 16K table.

### Table lookup and validation

```
LDG.E.U16 R15, [R10.64]                     ; load existing hash entry (16-bit position)
PRMT R14, R15, 0x9910, RZ                   ; zero-extend to check for 0xFFFF sentinel
ISETP.NE.AND P2, PT, R14, -0x1, PT          ; entry != 0xFFFF? (not empty)
```

If the entry is not empty (someone previously wrote to this hash slot), the kernel validates the match by loading 4 bytes from the referenced position and comparing:

```
LDG.E.U8 R36, [R14.64]          ; load 4 bytes at the hash-referenced position
LDG.E.U8 R37, [R14.64+0x1]
LDG.E.U8 R39, [R14.64+0x2]
LDG.E.U8 R41, [R14.64+0x3]
PRMT R36, R37, 0x7604, R36      ; assemble 4-byte word
PRMT R36, R39, 0x7054, R36
PRMT R41, R41, 0x654, R36
ISETP.EQ.AND P0, PT, R41, R32, PT  ; compare with current 4-byte word
```

If the 4 bytes match, the kernel has found a valid LZ4 match via the hash table.

### Hash table update

After probing (whether a match was found or not), the kernel updates the hash entry with the current position:

```
@!P0 STG.E.U16 [R10.64], R15    ; write current position to hash table
```

This is done only by lanes that did NOT find a match through the MATCH.ANY path and did not already have a valid hash-table match — preventing unnecessary overwrites.

The hash table update also uses MATCH.ANY to avoid redundant writes when multiple lanes hash to the same slot:

```
MATCH.ANY R12, R12               ; find lanes with same hash index
FLO.U32 R14, R12                 ; find lowest-numbered such lane
ISETP.NE.AND P0, PT, R14, R19   ; only the lowest lane writes
@!P0 STG.E.U16 [R10.64], R15   ; single writer per hash slot
```

---

## 18. Phase 5: Match Extension

**SASS location**: lines 30929–30962 (offsets `0x16e0`–`0x18a0`)

Once a match is found (either via MATCH.ANY or hash table), the kernel extends it forward byte-by-byte to determine the full match length. This uses **warp-parallel comparison**:

### Parallel byte comparison

```
.L_x_1025:
    IMAD.IADD R13, R19, 0x1, R34         ; lane's offset into extension
    IMAD.IADD R15, R28, 0x1, R13         ; match source position + offset
    LDG.E.U8 R10, [R10.64]               ; load byte from match source
    LDG.E.U8 R13, [R12.64]               ; load byte from current position
    ISETP.NE.AND P0, PT, R13, R10, PT    ; do they differ?
.L_x_1021:
    VOTE.ANY R10, PT, P0                  ; any lane found a mismatch?
    ISETP.NE.AND P0, PT, R10, RZ, PT
    @P0 BRA `(.L_x_1024)                 ; if mismatch found, stop
    IADD3 R34, R34, 0x20, RZ             ; advance by 32 (warp width)
    @!P0 BRA `(.L_x_1025)               ; continue extending
```

Each lane compares one byte at offset `threadIdx.x + extension_base`. All 32 lanes compare simultaneously, checking 32 bytes per iteration. `VOTE.ANY` detects the first mismatch across the warp.

### Finding the exact mismatch position

```
.L_x_1024:
    BREV R10, R10                  ; bit-reverse the mismatch mask
    FLO.U32 R11, R10               ; find the first mismatching lane
    IADD3 R26, R34, 0x1f, -R11    ; match_length = extension_base + 31 - first_mismatch
```

`BREV` + `FLO` finds the lowest-numbered lane with a mismatch, giving the exact match length.

---

## 19. Phase 6: Token Emission

**SASS location**: lines 30800–30895 (offsets `0x0fd0`–`0x14e0`) and lines 30964–31139 (offsets `0x18b0`–`0x22a0`)

After finding a match (or reaching end of input), the kernel emits LZ4 tokens. Token emission is done by **a single lane** (lane 0, `R19 == 0`) while other lanes participate in the literal copy.

### Token byte encoding

```
IMNMX.U32 R10, R32, 0xf, PT      ; literal_length_capped = min(literal_length, 15)
IMAD R15, R10, 0x10, R11          ; token = (literal_length_capped << 4) | match_length_nibble
STG.E.U8 [R4.64], R15            ; write token byte
```

### Literal length extension (varint)

When literal_length ≥ 15, extension bytes are written. The kernel computes how many 0xFF bytes to write:

```
IADD3 R10, R32, -0xf, RZ                  ; remaining = literal_length - 15
IMAD.WIDE.U32 R10, R10, -0x7f7f7f7f, RZ   ; magic multiply for div-by-255
SHF.R.U32.HI R12, RZ, 0x7, R5             ; R12 = remaining / 255 (via multiply-shift)
```

The `IMAD.WIDE.U32 ... -0x7f7f7f7f` is a **compiler trick for division by 255**: multiply by the modular inverse of 255 and take the high word, then shift. This avoids an expensive integer division.

Extension bytes are written cooperatively by the warp:

```
SEL R5, R14, 0xff, P3       ; if lane < num_full_255s, write 0xFF; else write remainder
STG.E.U8 [R2.64], R5        ; lane 0 writes byte 0
STG.E.U8 [R2.64+0x20], R11  ; lane 1 writes byte 1 (stride 0x20 = 32)
STG.E.U8 [R2.64+0x40], R13  ; lane 2 writes byte 2
STG.E.U8 [R2.64+0x60], R15  ; lane 3 writes byte 3
```

The unrolled loop at `.L_x_1010` (offsets `0x12a0`–`0x13d0`) and `.L_x_1032` (offsets `0x1c00`–`0x1d50`) writes 4 extension bytes per iteration, with each of the 4 lanes handling one byte at stride 0x20 from the output pointer. Each byte is either `0xFF` (if more full groups remain) or the final remainder (via `SEL`).

### Literal data copy

Literal bytes are copied from source to output cooperatively:

```
.L_x_1013:
    LDG.E.U8 R3, [R2.64]          ; load literal byte from source
    STG.E.U8 [R4.64], R3          ; store to compressed output
    IADD3 R11, R11, 0x20, RZ      ; advance by warp width
    @!P0 BRA `(.L_x_1013)
```

Each lane copies one byte at stride 0x20 (32), so 32 bytes per iteration.

### Match offset encoding

The 16-bit match offset is written as two bytes (little-endian):

```
SHF.R.U32.HI R5, RZ, 0x8, R13    ; high byte = offset >> 8
STG.E.U8 [R2.64], R13            ; write low byte
STG.E.U8 [R2.64+0x1], R5         ; write high byte
```

### Match length extension

Same varint pattern as literal length extension, using the same division-by-255 trick.

---

## 20. Phase 7: Output Size Finalization

**SASS location**: lines 31149–31160 (offsets `0x2300`–`0x23b0`)

After the main loop completes, lane 0 writes the final compressed size:

```
IMAD.MOV.U32 R2, RZ, RZ, R21          ; R21 = total output bytes written
@!P0 IMAD.MOV.U32 R2, RZ, RZ, 0x1    ; if input was empty, output = 1
@!P0 STG.E.U8 [R8.64], RZ            ; write a zero byte for empty input
STG.E.64 [R4.64], R2                  ; write compressed size to output array
EXIT
```

---

## 21. Compress Kernel: Key Architectural Observations

### The MATCH.ANY trick is the headline

The `MATCH.ANY` instruction is the single most interesting aspect of this kernel. It turns the warp into a **32-entry associative lookup**: every lane holds a 4-byte value, and in a single cycle, every lane knows which other lanes hold the same value. This gives:

- **Zero-latency match finding** within a 32-byte window (no memory access needed)
- **Automatic deduplication** — if bytes 5 and 17 have the same 4-byte prefix, both lanes see each other
- **No hash collisions** — MATCH.ANY is an exact comparison, not a hash

The hash table is the **fallback** for matches beyond the 32-byte warp window. Within the window, MATCH.ANY is strictly superior.

### No shared memory at all

Unlike the decompress kernel, the compress kernel uses zero shared memory. The hash table lives in global memory (32 KB per chunk at 16K × 2-byte entries). This is surprising — shared memory would provide much lower latency for hash probes. Possible reasons:

1. The MATCH.ANY path handles nearby matches without any memory access
2. Hash table probes are infrequent enough that global memory latency is acceptable
3. Keeping the hash table in global memory allows larger tables (shared memory is limited to ~100 KB per SM)
4. With 0 shared memory usage, more blocks can be resident on each SM, improving occupancy

### Warp-parallel match extension

The match extension phase checks 32 bytes per cycle (one byte per lane). This is 32× faster than serial extension. For long matches, the warp sweeps forward in 32-byte chunks until `VOTE.ANY` detects a mismatch, then `BREV` + `FLO` pinpoints the exact byte.

### Division by 255 via multiply-shift

The varint encoding of literal/match lengths requires knowing how many 0xFF bytes to emit (length / 255). Rather than using an integer division, the kernel uses:

```
IMAD.WIDE.U32 R10, R10, -0x7f7f7f7f, RZ    ; multiply by magic constant
SHF.R.U32.HI R12, RZ, 0x7, R5              ; shift right by 7
```

`-0x7f7f7f7f` = `0x80808081` in unsigned, which is the multiplicative inverse of 255 mod 2³². The high word of the 64-bit product, shifted right by 7, gives the exact quotient. This is a standard compiler optimization for constant division, but seeing it in GPU assembly confirms the compiler is generating it correctly.

### Rolling hash via SHFL.DOWN

The 4-byte hash key construction via two `SHFL.DOWN` operations is efficient:
- `SHFL.DOWN 1` gets the next byte, shifted left 8 and ORed → 2-byte partial
- `SHFL.DOWN 2` gets the next 2 bytes, shifted left 16 and ORed → 4-byte word

This means each lane's hash key is constructed from its own byte plus 3 neighbors, using only warp shuffles — no extra global memory loads. The last 3 lanes (29, 30, 31) get garbage from the shuffle (they read from non-existent lanes), but they are masked out by the active-lane check before MATCH.ANY.

### Compress flow summary

```
┌─────────────────────────────────────────────────────────┐
│ KERNEL LAUNCH: 1 warp per chunk, 1 warp per block       │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 1. HASH TABLE INIT: 32 lanes zero 16K × U16 entries    │
│    Write 0xFFFF to all slots (sentinel for "no match")  │
│    Unrolled ×4, ~128 entries per iteration               │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 2. MAIN LOOP (per 32-byte window):                       │
│    a) Each lane loads 1 byte from input                  │
│    b) SHFL.DOWN ×2 to build 4-byte rolling hash key     │
│    c) MATCH.ANY → find lanes with identical 4-byte key   │
│       (single-cycle warp-wide associative lookup!)       │
│    d) If match found within warp → use it directly       │
│    e) If no warp match → probe global hash table         │
│    f) Update hash table with current positions           │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 3. MATCH EXTENSION: all 32 lanes compare in parallel    │
│    32 bytes checked per iteration                        │
│    VOTE.ANY detects first mismatch                       │
│    BREV + FLO pinpoints exact position                   │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 4. TOKEN EMISSION:                                       │
│    a) Encode token byte (lit_len << 4 | match_len)       │
│    b) Write varint extension bytes (div-255 trick)       │
│    c) Copy literal bytes (32 lanes, 1 byte each)         │
│    d) Write 16-bit match offset (little-endian)          │
│    e) Write match length extension if needed              │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 5. ADVANCE: move window by (literal_length + match_len) │
│    Loop back to step 2 until input exhausted             │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│ 6. FINALIZE: lane 0 writes compressed size to output    │
└─────────────────────────────────────────────────────────┘
```
