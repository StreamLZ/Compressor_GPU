# StreamLZ-GPU Optimization Ideas (kernel-level annex to /v4_ideas.md)

> **Status 2026-06-10**: still live. The baseline below (HEAD 4f49093)
> remains representative for the kernels these ideas target - the LZ
> decode and Huffman production kernels are unchanged since; only the
> decode BOOKKEEPING kernels changed (fused slzCompactAllDescsKernel +
> parallel slzMergeHuffDescsParKernel, 2026-06-10), which none of the
> six ideas touch. Ideas 1-6 are ALL still unimplemented in production.
> Related: tools/huff_test explored idea-2/5-adjacent LUT-build and
> decode upgrades on the tANS side (v4_ideas #11). This file is indexed
> from v4_ideas.md - new kernel-level ideas go HERE, new structural/
> feature ideas go in v4_ideas.md.


Forward-looking optimization candidates for the GPU compress and decode
paths. Each entry is self-contained: where to apply, what the change is,
why it should work, predicted win, and a harness-first plan.

Source material: `docs/nvcomp_lz4_architecture.md` (nvCOMP LZ4 / zstd / ANS
reverse-engineering, sections 1–60, with a 60-trick steal-list in §60),
and HEAD 4f49093 NCU profile data captured at `c:\tmp\prof_l{1,5}_*.ncu-rep`
on RTX 4060 Ti sm_89.

## Current baseline (HEAD 4f49093, post-BIL)

### Decode (enwik8 100MB, kernel best-of-30, ms)

| Kernel | L1 | L5 | Notes |
|--------|---:|---:|-------|
| `slzLzDecodeRawKernel` (L1/L2 raw) | 3.31 | — | NCU: 61% SM, 72% Mem SoL, 49% L1TEX stall, IPC 2.55 |
| `slzLzDecodeKernel` (L3+ entropy) | — | 3.70 | NCU: 63% SM, 73% Mem SoL, 47% L1TEX stall, IPC 2.60 |
| `slzHuffBuildLutKernel` | — | 0.77 | NCU: 53% SM, **compute-bound**, top stall "Fixed latency dep" 36%, IPC 2.18 |
| `slzHuffDecode4StreamKernel` | — | 0.76 | NCU: 32% SM, **98% Mem SoL (saturated)**, top stall L1TEX 71%, IPC 0.90 |

### Encode (enwik8 + silesia, LZ kernel ms)

| Level | enwik8 | silesia | Notes |
|------:|-------:|--------:|-------|
| L1 |  70 | 132 | greedy parser |
| L3 |  84 | 155 | greedy parser |
| L4 |  84 | 151 | greedy parser (was 108 / 203 pre hash_bits=17 fix) |
| L5 | 236 | 412 | serial chain parser (was 283 / 20,811 pre hash_bits=17 fix) |

The "silesia L5 PATHOLOGICAL" 20,811 ms figure was diagnosed in
2026-05-27 to be **PCIe spill of the chain hash table** (at hash_bits=20
the chain parser's per-chunk hash is ~8 MB; silesia at sc=0.25 has 3,248
chunks → 26 GB hash allocation that overflows 16 GB VRAM and pages
through PCIe at ~30 GB/s vs VRAM's ~360 GB/s). Fixed by lowering
hash_bits to 17 for L4/L5 in `levels.zig` — per-chunk hash drops to
~1 MB which fits in VRAM for any reasonable input size. Cost: +0.07 %
raw bytes at L5, +0.1 pp at L4 (rounding), no decode-side impact.

---

## Idea 1 — MATCH.ANY warp match-finder for LZ encode

**Status note (2026-05-27):** the original "HIGHEST IMPACT" framing of
this idea was driven by the 20.8 s silesia-L5 figure. That figure was
PCIe-spill, now fixed (see baseline table above — silesia L5 went
20,811 → 412 ms with a one-line hash_bits change). The chain parser is
still serial on lane 0 (NCU shows ~1.1 active threads/warp), so a
warp-cooperative redesign would still help — but the practical upside
is now closer to 2-3× on the remaining 412 ms, not the 50× that the
hash-bits fix already delivered.

**Where to apply:** `src/gpu/encode/lz_kernel.cu` — both the warp-parallel
greedy parser (L1-L4, in `lz_greedy_parser.cuh`) and the serial chain
parser (L5+, in `lz_chain_parser.cuh`).

**nvCOMP source:** doc §15-16 (LZ4 compress) and §51 (zstd compress). The
exact SASS pattern is at `lz4_decompress_sm89.sass` lines 30659-30698.

**Current code:** Both parsers do sequential per-position hash table
probes. Each lane (or lane 0 in chain mode) computes a 4-byte hash,
loads a hash bucket from global memory (~hundreds of cycles), validates
the match, and advances. For silesia binary data, L5's chain parser
hits CHAIN_MAX_STEPS-guarded deep chains; the result is ~20.8 seconds to
encode 213 MB.

**The trick (§15-16):**

1. Each lane loads ONE byte at its position (`input[pos + lane]`).
2. Two `__shfl_down_sync` ops build a rolling 4-byte hash key per lane —
   each lane ends up holding `input[pos+lane..pos+lane+3]`. Zero extra
   loads, pure register shuffles.
3. `MATCH.ANY R12, R32` — single CUDA instruction that returns, for each
   lane, a bitmask of OTHER lanes whose 4-byte key equals this lane's
   key. This is a **warp-wide associative groupby in ONE cycle.**
4. `BREV` + `FLO.U32` finds the lowest-numbered matching lane = nearest
   in-warp match.
5. Within the 32-byte window, **match-finding cost goes to zero** (no
   memory access). Only matches that escape the warp window fall through
   to the global hash table.

The CUDA intrinsic is `__match_any_sync(mask, value)` (sm_70+), and the
PTX is `match.any.sync.b32`. Fully supported on sm_89.

**Why this fixes silesia L5:** The chain-parser pathology comes from
chasing long hash chains in global memory. With MATCH.ANY, the parser
first checks the local 32-byte window — for highly-repetitive binary
data (mostly what silesia's compressed regions look like), most matches
ARE local. The fallback chain walk only fires for genuinely-distant
matches, which are rare in pathological data.

**Predicted win:**
- silesia L5: **20,811 ms → low-thousands ms** (5-10× speedup), if most
  matches land in-warp
- enwik8 L5: 283 ms → 200-250 ms (10-30%)
- L1-L4: marginal-to-moderate (greedy parser already warp-parallel; this
  removes hash table probe latency from the inner loop)

**Risks:**
- Restructuring the parser is non-trivial; the existing token emission
  + recent-offset tracking + chain validation logic must continue to work
- May change which matches are found (greedy vs chain are different
  ratio/speed tradeoffs); ratio could shift
- The 32-byte window is small — for genuinely sparse data the fallback
  hash table dominates anyway

**Harness-first plan:**
1. Build a standalone CUDA harness in `tools/lz_test/` (parallel to
   `tools/huff_test/`) that exercises just the match-finding inner loop
   against a CPU oracle.
2. Implement MATCH.ANY-only variant first (no hash table at all) to
   bound the in-warp-only win.
3. Add hash-table fallback for cross-window matches.
4. A/B against current chain parser on silesia L5 enwik8 L5 with the
   harness before any production-kernel changes.
5. Effort: 1-2 weeks.

---

## Idea 2 — Strided LUT fill with tiered unroll for HuffBuild (BEST ROI)

**Where to apply:** `src/gpu/decode/huffman_kernel.cu` lines 121-150
(`slzHuffBuildLutKernel`, Pass 1 single-symbol fill and Pass 2
dual-symbol fill).

**nvCOMP source:** doc §41, Trick #9. SASS at
`zstd_decompress_sm89.sass` lines 11860-11954.

**Current code:**

```cpp
// Pass 1: single-symbol entries
for (int s = lane; s < HUFF_ALPHABET; s += WARP_SIZE) {
    int L = code_lengths[s];
    if (L == 0 || L > MAX_CODE_LEN) continue;
    uint32_t aligned = codes[s] << (MAX_CODE_LEN - L);
    uint32_t span = 1u << (MAX_CODE_LEN - L);
    uint32_t entry = packLutEntry((uint8_t)s, 0, (uint8_t)L, LUT_NUM_SYMS_SINGLE);
    for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;  // SERIAL per-lane
}
```

The inner `for i = 0; i < span; i++` is **strictly serial on the active
lane** while 31 lanes are idle. For a length-1 code (span = 1024), ONE
lane writes 1024 LUT entries sequentially. For Pass 2 (dual-symbol) the
serial cost is much worse — outer × inner × span on a single lane.

**The trick (§41):** Invert the dimensionality. Three loop tiers, each
with a different unroll depth, parallelizing the fill across all 32
lanes within a tier:

- **Tier 1 (short codes, L_s small → span large ≥ 16):** 16 strided
  `STS.U16` writes per loop iteration, stride 32 between writes. 32
  lanes each handle a different "stride group", collectively filling 32
  consecutive entries per "store burst". Result: 512 LUT entries filled
  per warp-iteration.
- **Tier 2 (medium codes, span 4-15):** 8-store unroll.
- **Tier 3 (long codes, span 1-3):** current symbol-parallel approach
  with serial inner is fine — span is tiny anyway.

The dispatch by tier is branch-based but warp-uniform (the kernel knows
each symbol's `L` from `code_lengths[s]`, so it can route to the right
tier).

**Why this fixes the bottleneck:** HuffBuild is compute-bound at 53% SM
Busy with top stall "Fixed latency dep" — the serial fill's dependent
stores are exactly the kind of work that creates fixed-latency
dependency chains. Removing serial fills converts 32× more lane
utilization in the dominant Pass 2 phase.

**Predicted win:** 40-60% off `slzHuffBuildLutKernel` = 0.77 ms → ~0.35
ms. That's ~8% off total L5 decode aggregate (4.60 ms).

**Risks:**
- Refactor of the LUT fill logic; canonical Huffman codes are
  deterministic, so any change must produce a bit-identical LUT (easy
  to verify via CPU oracle).
- The branchy tier dispatch could itself cost something if codes are
  evenly distributed across lengths; mitigation = pre-sort symbols by
  length and process tiers in batches.

**Harness-first plan:**
1. Add a `huff_lut_fill_experiments.cu` in `tools/huff_test/`. Reuse
   existing CPU oracle from prior huff experiments.
2. Implement Tier 1 (deep unroll) first against synthetic skewed
   distributions (short codes dominant) — should show the biggest win
   on text-like inputs.
3. Add Tiers 2-3, verify bit-identical LUT vs CPU oracle on enwik8 +
   silesia weight tables.
4. Backport to `slzHuffBuildLutKernel` once harness shows win.
5. Effort: 1-2 days.

---

## Idea 3 — Multi-warp-per-chunk LZ decode (UNCERTAIN, structural)

**Where to apply:** `src/gpu/decode/lz_kernel.cu` — both
`slzLzDecodeRawKernel` (L1/L2) and `slzLzDecodeKernel` (L3+). Currently
launched with `Block (32, 2, 1) = 64 threads` (2 warps), `Grid (763)`
(one block per chunk).

**nvCOMP source:** doc §55 (ANS 2D grid dispatch:
`gridDim = (num_chunks, num_warps_per_chunk)`) + the open lever flagged
in `FAILED_EXPERIMENTS.md` line 2587-3055 "PP-v2 IS the ceiling for
L1-thrash attacks. Further gains require structural changes: different
parallelization granularity (multiple warps per sub-chunk cooperating,
cross-warp pipelining)."

**Current state:** The LZ decode kernel is at its documented structural
ceiling. 13 separate micro-optimizations tried in May 2026
(parallel parse, wide loads, cp.async, __ldcs, L1 carveout, etc.) —
only parallel-parse (PP-v2) won, contributing 21-25% off baseline.
Further per-instruction tweaks land in SASS without moving wall-clock.

**The trick:** Restructure to `gridDim = (num_chunks, K)` with K=2 or
K=4 cooperating warps per chunk. Warps can pipeline:

- Warp 0: parse + broadcast tokens for batch N
- Warp 1: execute (literal copy + match copy) for batch N-1
- (with K=4) Warps 2-3: do batch N-2 and N-3

Per-chunk critical path becomes
`(parse_time + copy_time) / K` instead of
`parse_time + copy_time`. The cross-warp coordination needs:

- Shared-memory ring buffer of parsed tokens (size = K * batch_size)
- One `__syncthreads()` per batch (cheap on a 2-4 warp block)
- The atomic-OR completion bit pattern from nvCOMP zstd §29 (without
  the NANOSLEEP — within a block, `__syncthreads()` is the right primitive)

**Why this might fix the ceiling:** The current kernel at 73% SOL with
26.3% irreducible scheduling gap. Inter-warp pipelining attacks the
"scheduling gap" by overlapping warp-level dependencies — when warp 0
stalls on a parse load, warp 1 can be doing useful copy work *within
the same block*, sharing L1.

**Predicted win:** Unknown — FAILED_EXPERIMENTS notes this hasn't been
tried. Hopeful ballpark: 10-25% off LZ decode kernel = ~0.4-0.9 ms off
L5 aggregate. Could also be 0% if the warps end up serializing on the
shared parsed-tokens ring.

**Risks:**
- Multi-day rewrite; correctness debugging is non-trivial because the
  cross-warp ring + sync sequence is harder to validate than current
  single-warp design.
- Register pressure may force occupancy down if warps need more state.
- Could regress if the cross-warp coordination overhead exceeds the
  pipelining benefit on short chunks.

**Harness-first plan:**
1. Prototype the cross-warp ring + sync pattern in
   `tools/lz_decode_test/` (new, mirror `tools/huff_test/`).
2. Test against CPU oracle on enwik8 L1 + L5 sub-chunks first.
3. Build A/B: PP-v2 single-warp vs K=2 multi-warp.
4. Only backport to production kernel if A/B shows clear win.
5. Effort: 1-2 weeks. Tackle AFTER ideas 1-2 since this is the most
   speculative.

---

## Idea 4 — Hillis-Steele scan for `buildCanonicalCodes`

**Where to apply:** `src/gpu/common/gpu_huffman.cuh` lines 102-122
(`buildCanonicalCodes`). Called by lane 0 in both
`slzHuffBuildLutKernel` (decode) and `slzHuffBuildTablesKernel` (encode).

**nvCOMP source:** doc §34, Trick #6. SASS at
`zstd_decompress_sm89.sass` lines 17043-17066.

**Current code:** Three serial loops, all on lane 0:

```cpp
// Loop 1: histogram code lengths (256 iters, lane 0 only)
for (int s = 0; s < HUFF_ALPHABET; s++) length_count[L]++;

// Loop 2: cumulative-sum + left-shift (12 iters, lane 0 only) ← CLASSIC HILLIS-STEELE TARGET
for (int L = 1; L <= HUFF_MAX_CODE_LEN; L++) {
    code = (code + length_count[L - 1]) << 1;
    next_code[L] = code;
}

// Loop 3: per-symbol code assignment (256 iters, lane 0 only)
for (int s = 0; s < HUFF_ALPHABET; s++)
    codes[s] = (L != 0) ? next_code[L]++ : 0u;
```

**The trick (§34):** Replace serial loops with warp-cooperative ops:
- **Loop 1:** 32 lanes each cover 8 symbols, `atomicAdd` to a shared-mem
  13-bucket histogram. ~30 cycles vs 256 serial.
- **Loop 2:** 5-step Hillis-Steele scan on 13 entries (1 warp). 5
  shuffles + 5 adds = ~10 cycles vs 12+12 = 24 serial.
- **Loop 3:** All 32 lanes cover 8 symbols each. MATCH.ANY (or
  `__match_any_sync`) on the length value finds lanes sharing the same
  L; `POPC(LT_MASK & match_mask)` gives intra-group offset; single
  leader updates `next_code[L]` count. ~50 cycles vs 256 serial.

**Why this might help:** The `buildCanonicalCodes` function runs inside
`slzHuffBuildLutKernel` (decode) and `slzHuffBuildTablesKernel`
(encode). On the decode side, this is part of HuffBuild's 0.77 ms
cost — but Idea 2 (strided LUT fill) likely dominates HuffBuild's
work, so this would be a secondary win after that.

**Predicted win:** Folded into Idea 2's projection — together they could
push HuffBuild from 0.77 ms to ~0.30 ms (~60% off).

**Risks:** Minimal. Code change is contained to one function. Output
must be bit-identical to current canonical codes; trivial to verify.

**Harness-first plan:**
1. Same harness as Idea 2 (`tools/huff_test/`).
2. Implement and verify bit-identical `codes[]` array vs current
   serial implementation.
3. Effort: a few hours if done alongside Idea 2.

---

## Idea 5 — `LOP3.LUT 0xfe` fused 3-input OR (verify, don't refactor)

**Where to apply:** `src/gpu/decode/huffman_kernel.cu` BIL refill sites
(lines 240, 291, 337, 390). Same `memcpy + __byte_perm + bit_buf |=`
pattern at all four sites.

**nvCOMP source:** doc §28, Trick #3.

**The trick:** `LOP3.LUT R, A, B, C, 0xfe, !PT` is a single-instruction
3-input OR (`A | B | C`). nvCOMP uses this in bitstream assembly to
combine multiple byte loads in one instruction instead of two ORs.

**Current code:** Our BIL refill does:
```cpp
uint32_t word;
memcpy(&word, w_ptr, sizeof(uint32_t));    // alignment-safe load → emits byte loads + shifts + ORs
uint32_t v = __byte_perm(word, 0, 0x0123); // LE → BE swap
bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);  // merge into bit buffer
```

When `w_ptr` is unaligned (75% of the time per BIL's wire format), nvcc
emits 4 byte loads + shifts + ORs. The pair of ORs in the middle may
already fuse to `LOP3.LUT 0xfe`, but it's worth confirming.

**Why might help:** If the compiler ISN'T emitting `LOP3.LUT 0xfe` for
the byte-merge pattern, manually using inline PTX (or restructuring the
load) could save 1-2 instructions per refill. With ~3000 refills per
warp on a 64KB sub-chunk, this adds up to ~6000 cycles saved per warp.

**Predicted win:** 1-2% off `slzHuffDecode4StreamKernel` if not already
present. The decode is memory-saturated (98% Mem SoL) so compute
savings only help to the extent they let warps catch up on memory
latency.

**Risks:** None — pure verification + minor inline PTX if needed.

**Plan:**
1. Disassemble `huffman_kernel.ptx` for the BIL refill sites:
   `nvdisasm --print-line-info huffman_kernel.cubin > huff_decode.sass`
2. Search for `LOP3.LUT` near the refill addresses. Confirm 0xfe is
   emitted for the byte-merge.
3. If not, write inline PTX variant in `tools/huff_test/` and A/B test.
4. Effort: 1-2 hours.

---

## Idea 6 — Packed multi-value butterfly reduction (BIL encoder, minor)

**Where to apply:** `src/gpu/encode/huffman_kernel.cu` lines 331-335 and
384-387 — the K-min and total_tail-sum reductions in
`slzHuffEncode4StreamKernel`.

**nvCOMP source:** doc §45, Trick #11.

**Current code:** Two SEPARATE 5-step butterfly reductions:

```cpp
uint32_t K = my_words;
for (int off = 16; off > 0; off >>= 1) {           // butterfly #1 (min)
    uint32_t v = __shfl_xor_sync(FULL_WARP_MASK, K, off);
    K = (v < K) ? v : K;
}
// ... unrelated code ...
uint32_t total_tail = my_tail_bytes;
for (int off = 16; off > 0; off >>= 1) {           // butterfly #2 (sum)
    total_tail += __shfl_xor_sync(FULL_WARP_MASK, total_tail, off);
}
```

**The trick (§45):** Interleave the two butterflies so each round does
BOTH reductions:

```cpp
uint32_t K = my_words;
uint32_t total_tail = my_tail_bytes;
for (int off = 16; off > 0; off >>= 1) {
    uint32_t vK = __shfl_xor_sync(FULL_WARP_MASK, K, off);
    uint32_t vT = __shfl_xor_sync(FULL_WARP_MASK, total_tail, off);
    K = (vK < K) ? vK : K;
    total_tail += vT;
}
```

Per nvCOMP doc: "NVIDIA's SHFL has no extra cost for multiple register
operands within the same instruction group, so packing 4-way reductions
costs essentially the same as 1-way."

**Why might help:** Halves the shuffle instruction count in the
encoder's per-body finalization. Encoder isn't a critical-path
bottleneck (HuffDecode dominates aggregate decode time), but free
instruction-count savings.

**Predicted win:** <1% on encode side. Truly polish.

**Risks:** None — semantically identical refactor.

**Plan:**
1. Direct in-source change, no harness needed (semantically identical).
2. A/B kernel time on enwik8 L3 encode bench.
3. Effort: 30 minutes including verification.

---

## Ideas explicitly RULED OUT after analysis

These appear in the nvCOMP doc's 18-trick steal-list but don't apply to
our codec (tANS-specific) or our architecture (encoder-only patterns
where the decoder doesn't have the equivalent):

| # | Trick | Why ruled out |
|---|-------|---------------|
| #4 | `MUFU.RCP` modular arithmetic | tANS/FSE state advancement only; we're pure Huffman, no modular state |
| #5 | NANOSLEEP spin-wait inter-block sync | Our sidecar approach wins for >4 blocks per nvCOMP doc §32 explicit |
| #8 | Packed-nibble decode | Already optimal in `unpackWeightByte` (2 instructions) |
| #10 | `LOP3.LUT 0xf8` branchless flag-byte parsing | Our chunk headers parsed CPU-side; scan kernels are <0.1ms (off critical path) |
| #12 | `ATOMS.ADD` intra-block coordination | We don't have intra-block work queues |
| #13 | Fixed-size tANS tables `<512, 54>` | tANS only |
| #14 | Leader-atomic + warp scan for variable output | We use the scan part; we don't need the atomic because output regions pre-reserved via `descs[i].dst_offset` — we're already at the strictly-better variant |
| #15 | `__constant__` memory for read-mostly tables | Our LUT is per-block runtime-built (depends on per-body symbol distribution); can't pre-stage |
| #17 | Privatized histograms with MATCH.ANY (decode side) | Decode doesn't need histograms; encode-side variant lives in Idea 4 |
| #18 | Power-of-2 normalized counts | tANS only |

## Things deliberately NOT attempting

From `FAILED_EXPERIMENTS.md`, these have been tried and lost (some
multiple times):

- **Word-interleaved 4-stream Huffman layout** (lines 1817-1862): BIL is
  the successor.
- **Software prefetch in Huff decode** (1863-1907): depth-1 too shallow.
- **u16 single-symbol Huffman LUT** (1908-1952): lost 18% by dropping
  dual-symbol fast path.
- **64-bit output stores in Huff decode** (1953-1998): lost 22% — wstore32 is the sweet spot.
- **Persistent atomic-work-queue Huff decoder** (1999-2045): lost 3× —
  GPU hardware block scheduler is already a better work queue.
- **Cold-branch escape Huff decode** (2046-2089): lost 2% — call-site
  barrier outweighed save.
- **All LZ decode micro-optimizations** (2090-3055): 13 attacks tried,
  only parallel-parse won. The meta-conclusion is clear: PP-v2 is the
  ceiling; further gains need STRUCTURAL changes (Idea 3 above).

Re-trying any of the above without a fundamentally different approach
is wasted effort. Read the relevant `FAILED_EXPERIMENTS` entry before
attempting.

---

## Suggested execution order

1. **Idea 5 first** (1-2 hours). Pure verification, banks a quick check.
2. **Idea 2 + Idea 4 together** (1-2 days). Best ROI; harness-first in
   `tools/huff_test/`. ~8% off L5 decode aggregate if both land.
3. **Idea 6** (30 min). Free polish on encoder side.
4. **Idea 1** (1-2 weeks). Was the silesia L5 50× win — already landed
   via the hash_bits=17 patch (see baseline table). Remaining upside on
   the chain parser is 2-3× from warp-cooperative redesign. Build new
   `tools/lz_test/` harness if pursuing. The standalone encode-L5
   harness from 2026-05-27 lives at `tools/encode_l5_silesia/`.
5. **Idea 3** (1-2 weeks). Only if encoder work is done and decode is
   still the bottleneck. Most speculative; could be 0% or 20%.
