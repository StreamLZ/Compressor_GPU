# v4 Ideas — future work

Candidate work beyond the 2026-06-09 wave (`db1e061`). State at that
commit: sc=0.25 + hb=17 defaults, A-024 fixed, A-023 on both backends;
on enwik9 1 GB StreamLZ beats nvCOMP LZ4 at L1 (52.6% vs 53.6%, 24.3
vs 33.0 ms decode kernel) and nvCOMP Zstd at L5 (35.50% vs 35.75%,
34.2 vs 50.8 ms). Ideas are ordered by leverage-per-risk, not by size.

A standing guardrail first: per `FAILED_EXPERIMENTS.md` (2026-05-24
meta-entry, re-confirmed twice on 2026-06-09), the decode kernels are
issue-bound at ~92% SM throughput and ~91% occupancy. **Per-load
latency attacks (staging, prefetch, wide loads, software pipelining)
do not move wall-clock and should not be retried.** The only lever
that has ever worked is reducing per-warp instruction count or
per-warp serial chain length. Every speed idea below is that class.

---

## 1. Flat batched literal copy (token-ownership binary search)

**What**: In the PP fast path (`lz_decode_raw.cuh`), replace the 32
serialized per-token `warpLiteralCopy` calls with ONE warp-wide pass
over the batch's concatenated literal bytes. The per-token prefix sums
(`my_lit_local`, `my_dst_local`) already exist in registers; stage all
32 to shared memory (currently 0 bytes used — no occupancy cost), then
each lane copies flat literal byte `i`, finding its owning token
`k = max{k : lit_prefix[k] <= i}` with a 5-step shared-memory binary
search, and writes to `dst_pos + dst_local[k] + (i - lit_prefix[k])`.
Two `__syncwarp()`s per batch instead of today's 64.

**Why**: NCU shows 17.9/32 average active threads — the idle lanes are
concentrated in the copy phase (parse is already fully parallel). A
typical token carries ~5-8 literal bytes, so today's per-token copy
iteration runs at ~20% lane efficiency plus ~10 instructions of
shuffle/sync overhead. Flattened, every copy iteration moves 32 bytes
at full width. This is the same instruction-count lever as the two
historical wins (PP +22%, PP-for-OFF16_SPLIT +21-25%), and nvCOMP
ships exactly this design (the §8 binary-search in
`docs/nvcomp_lz4_architecture.md`).

**Expected**: ~1.2× decode kernel alone (literals are ~55% of output
bytes at L1). Zero ratio impact. The safest half of the #1/#2 pair —
the literal stream is separate from the output, so there is no
read-after-write hazard inside the batch.

**Cost/risk**: days; rewrites the correctness core of both decode
kernels; needs full level/scale re-verification. Do this FIRST and
measure before deciding on #2.

## 2. Flat independent-match copy + ordered fallback

**What**: After #1, classify each token's match: *independent* if its
source range lies entirely before this batch's output start
(`match_src + match_len <= batch_dst_start`) and `dist >= 4`.
Flatten all independent matches in one ownership-search pass exactly
like #1; run the dependent remainder serially in token order (the
current path). Conservative classification — any doubt → serial.

**Why**: same lever; matches are the other ~45% of output bytes.

**Expected**: #1+#2 combined ≈ 1.3-1.6× decode kernel (enwik9 L1
24.3 → ~15-18 ms, would be ~2× nvCOMP LZ4). Shrinks toward 1.2-1.3×
if 30-40% of match bytes are batch-local (recent-offset-heavy text
keeps matches near). Zero ratio impact. Note e2e barely moves either
way (PCIe-bound: kernel is ~16% of 149 ms) — this pays off only for
device-resident (D2D) pipelines.

**Cost/risk**: days; the dependency analysis is the hard part and a
wrong "independent" call is silent corruption. Gate on #1 landing
cleanly and on measuring the actual batch-local match fraction first
(cheap kernel instrumentation).

## 3. Measure, then maybe parallelize, long-token parsing

**What**: `db1e061` stopped one long token from serializing the 32
short tokens ahead of it (PP-prefix truncation), but each long token
still costs a full serial iteration. Step 1 is pure measurement: count
serial iterations vs PP batches per workload (two atomics in a debug
build). Only if the serial fraction is >~10% on inputs we care about,
extend PP to parse long tokens in-lane: ballot the long lanes, prefix-
scan each lane's `length_stream` byte consumption (the tag byte
determines 1-vs-3 bytes), and fold the result into the existing scans.

**Why**: deep-redundancy inputs have materially longer matches (enwik9
L1 hits 50.3% vs enwik8's 58.6% at the same settings), so
`TOKEN_LONG_NEAR` frequency grows exactly on the inputs where decode
speed matters most. But we have NOT measured the residual cost since
the truncation fix — it may already be negligible (enwik9 sc=0.5
showed no change from the truncation fix, which argues it is small).

**Cost/risk**: measurement is hours; the parallel long-token parse is
1-2 days and a fiddly two-level variable-length scan. Don't build it
without the measurement.

## 4. Tighten `total_subchunks` from worst-case BOUND to actual

**What**: `uploadInputAndPrefixSum` (both backends) sizes entropy
scratch as `num_chunks × ceil(256 KB / sub_chunk_cap)` using the
hardcoded 256 KB chunk constant. At sc=0.25 the real chunks are 64 KB
with exactly 1 sub-chunk each — the bound is 2× too big. The host
builds `chunk_descs` anyway; summing `ceil(decomp_size / cap)` over
them gives the exact count for the same loop cost.

**Why**: (a) L3-L5 1 GB scratch drops 12 GB → 6 GB — today 12 GB
*barely* fits a 16 GB card and will fail on 12 GB cards; (b) it puts
the per-region byte offsets at 1 GB sc=0.25 back under 4 GiB, which
makes the **VK A-024 residual moot at current scales** (VK's u32 push
constants stop truncating) — i.e. this is the cheap path to VK 1 GB
L3+ decode. Keep the kernels' self-gating untouched; only the host
sizing changes.

**Cost/risk**: small, host-only, both backends in step. Highest
correctness-value-per-line item on this list.

## 5. Close the VK A-024 residual permanently (BDA entropy scratch)

**What**: Address `entropy_scratch` in the VK Huffman-decode and
LZ-general kernels through `VK_KHR_buffer_device_address`
(`buffer_reference` + uvec2 address, the proven A-008 pattern) instead
of a descriptor-bound SSBO, so region offsets are full 64-bit device
addresses.

**Why**: #4 defers the 4 GiB ceiling; this removes it (multi-GB
inputs, future sc choices, no silent edge). Also retires the A-005
`entropy_slot_stride.x` truncation in the same stroke.

**Cost/risk**: medium; mechanical copy of the A-008 BDA recipe across
2 kernels + dispatch; needs the A-024-style threshold tests on VK.

## 6. Re-differentiate L2 (product decision, not code)

**What**: Since hb=17-everywhere, L1 and L2 emit byte-identical
output. Options: (a) document L2 as an alias of L1; (b) repurpose L2
as the "max LZ ratio" profile — sc=0.5 (+~2 pp ratio at 1 GB, ~1.8×
slower large-input decode), giving the `--sc 0.5` tradeoff a stable
level name; (c) give L2 hb=18 back together with the A-023 batching
(now exists on both backends) so its 16 GB hash pages in batches
instead of thrashing.

**Why**: a level that duplicates another is API noise; (b) is the
most honest mapping of the real tradeoff surface we measured.

## 7. CUDA regression test for the A-023 batched dispatch

**What**: ptest case that sets
`encode_lz.g_force_batch_count_for_test` (e.g. batch=7 over a
64-chunk input), encodes, and asserts byte-identity against the
unbatched encode — mirroring `srcVK/tests/a023_batched_lz_dispatch.zig`.

**Why**: the CUDA batched path only triggers organically above ~8 GB
of hash demand; CI never exercises it today. Hours of work, closes a
real coverage hole introduced by the backport.

## 8. v4 wire format: self-describing tokens (remove the serial parse)

**What**: The remaining structural ceiling is that token k cannot be
located without decoding tokens 0..k-1. A v4 format could emit a small
per-batch sidecar (e.g. one u16 per 32 tokens: total cmd/lit/off16
consumption) letting every warp jump straight to its batch — or
restructure tokens at fixed stride. Encoder cost is a few bytes per
batch (~0.1% of output).

**Why**: `FAILED_EXPERIMENTS.md` names format change as one of two
remaining structural levers. Combined with #1/#2 this is the
"saturate DRAM bandwidth" endgame (~2× beyond #1/#2, speculative).

**Cost/risk**: weeks; breaks format compatibility; both backends +
both directions. Only worth it if decode throughput becomes the
product's headline number.

## 9. Tensor-core bulk copy for ≥256 B matches (parking lot)

From the FAILED_EXPERIMENTS "not tried" list. Only pays for very long
matches, which the text-workload match-length histogram says are rare.
Keep parked unless a binary-heavy workload (silesia-like or model
weights) becomes a target.

## 10. A-021: fuse the remaining VK decode bookkeeping dispatches

**What**: Apply the A-017 fusion pattern (which collapsed 4×
`compact_huff_descs` dispatches into one grid_x=4 kernel for a 2.4×
kernel-time win) to the remaining three unfused dispatches in the VK
decode front-half: `compact_raw_descs`, `gather_raw_off16`,
`merge_huff_descs`. Per `srcVK/PerfSweep.md`, these plus the A-006
explicit barriers and A-007 descriptor ABI account for the VK L3-L5
decode-kernel gap.

**Why**: VK L3-L5 1 GB decode kernels run ~48 ms vs CUDA's ~34 ms
(1.4×, re-measured 2026-06-10). e2e stays within the 10% parity bar
because PCIe amortizes it, but the kernel gap is the difference
between VK beating nvCOMP Zstd comfortably (CUDA: 34 vs 51 ms) and
barely (VK: 48 vs 51 ms). Note the 2026-06-10 A-024 revision added two
more huff-decode dispatches (3-region split), nudging the wrong way —
fusion has slightly more to reclaim now.

**Expected**: PerfSweep attributed ~1.45 ms of the 2.13 ms enwik8-scale
gap to these dispatches (attribution flagged "unverified"). Plausibly
closes half the 1.4× gap. Zero ratio impact.

**Cost/risk**: medium-low — A-017 is a proven template; ~1 day + the
standard cross-backend SHA regression.

## 11. Per-chunk adaptive entropy: Huffman vs tANS, pick the smaller

**The idea (2026-06-10)**: at encode time, decide PER CHUNK whether
the entropy body is Huffman or tANS, and emit whichever is smaller
(ties → Huffman). History makes this a both-axes win, not a tradeoff:

- tANS originally lost to Huffman on *overall* enwik8 + silesia ratio,
  so it was removed and Huffman got the optimization budget. BUT the
  `tools/huff_test` harness (tans_test.cu — production-snapshot
  testbed for `slzTansFseBuildKernel` + `slzTans32DecodeKernel`, four
  optimization variants, byte-exact verified) later pushed GPU tANS-32
  decode FASTER than the Huffman kernel of the time.
- If tANS is only ever used where it is strictly smaller, ratio is
  monotone-better by construction (the chunk_type byte already exists
  on the wire — the selector flag is free).
- If tANS decode throughput ≥ Huffman's, every flipped chunk also
  decodes faster, so the throughput-bound predecode pass speeds up
  too. Better ratio AND better decode speed simultaneously.
- Same shape as zstd's per-block entropy selection — proven practice.

**Gate 0 — MEASURED 2026-06-10 (RTX 4060 Ti), premise inverted**:
- tANS-32 harness (`tans_test.exe`, production snapshot, 3052 chunks,
  27.8 MB lit+tok, byte-exact): best variant `sl+aligned` 0.773 ms =
  **36.0 GB/s** (baseline 25.4, sharedlut 34.8, staged 30.6).
  FSE table build: 0.611 ms / 3052 tables.
- Production `slzHuffDecode4StreamKernel` in situ (nsys, enwik8 L5
  `-db`, min of 5): **0.727 ms** for the FULL entropy payload
  (lit + tok + off16 hi/lo) → ≥ 38 GB/s even crediting only the
  lit+tok bytes; higher with off16 counted. Huffman LUT build:
  0.160 ms.
- **Harness upgraded same day**: the two Huffman learnings tANS never
  received were transplanted and verified byte-exact —
  `slzTans32DecodeU32StoreKernel` (4-byte output interleave → u32
  stores): 37.8 GB/s; `slzTans32DecodeBilKernel` (+ BIL word-
  interleaved comp layout, host-transcoded snapshot): **40.8 GB/s**
  (0.681 ms). +13% over the prior best.
- **Exact Huffman denominator** (NCU `smsp__sass_data_bytes_mem_
  global_op_st` = 38.36 MB on enwik8 L5): production Huffman =
  38.36 MB / 0.727 ms = **52.8 GB/s** — the earlier "≥38" bound was
  loose (off16 planes add ~10 MB to the payload).
- Verdict: even fully upgraded, tANS decodes at ~0.77× Huffman's
  per-byte rate, and its table build is 4× dearer (0.61 vs 0.16 ms).
  The structural reason: Huffman's LUT yields up to 2 symbols per
  lookup (double-symbol entries, X2 batches); the tANS state chain
  yields exactly 1, serially dependent. The strict both-axes-win
  claim stays dead at the current design. One known lever remains
  untried: dual FSE states per lane (zstd's interleaved-states ILP
  trick — wire-level change, doubles the latency-hiding on the
  state→LUT→state chain). If pursued, it belongs in the v4 wire
  format design alongside the selector.

**Cross-check vs `docs/nvcomp_lz4_architecture.md` §22-§61 (the zstd
+ standalone-ANS steal-list), 2026-06-10**:
- **The 4× table-build gap is attackable.** Our `slzTansFseBuildKernel`
  (0.611 ms) is the naive serial spread; nvCOMP parallelizes exactly
  this with MATCH.ANY symbol bucketing (#1, §27 — "one-pass parallel
  histogram + scatter", ~20 cycles per 32 symbols), Hillis-Steele
  prefix scans (#6), and tiered strided LUT fill (#9). Applying those
  should land tANS table build near Huffman's 0.16 ms — removing one
  of the two strikes against the selector.
- **The decode-core gap stands.** Nothing in nvCOMP's GPU decode path
  breaks tANS's 1-symbol-per-lookup serial state chain — their zstd
  decompress runs ~19.7 GB/s e2e on the same card, well under our
  entropy-stage rates, and they do NOT use interleaved dual states on
  GPU. Dual-states-per-lane remains OUR untried lever, not a proven
  one.
- **Marginal decode adds**: LOP3.LUT 3-input OR (#3) — our BIL refill
  is already one-OR-per-word so upside is limited to the byte-tail
  path; 2D chunk×stream dispatch (#16) matters for production
  integration at small chunk counts, not for the harness's per-byte
  rate.
- **Encoder build-out gets cheaper than assumed**: leader-atomic +
  warp-scan variable-size output (#14/#2), MUFU.RCP division-free
  state advance (#4), MATCH.ANY privatized histograms (#17), and
  power-of-2 normalized counts (#18) together sketch most of a GPU
  tANS *encoder* — shrinking the "resurrect the host encoder"
  line-item in the build-out plan.
- The idea SURVIVES in weakened form: per-chunk min() is still
  ratio-monotone, and the decode-speed cost is epsilon — the huff
  predecode is only ~0.73 ms of the ~4 ms enwik8 L5 d2d (and a
  similar minor slice at 1 GB), so even 20% of bytes flipping to a
  0.9× codec costs ~0.02 ms. Net: a RATIO lever that is
  speed-neutral in practice, not a speed win. Gate 1 (the ratio
  dry-run) is now the decisive gate, and the bar should be a bit
  higher to pay for the added pipeline complexity.

**Gate 1 (a day, host-only)**: selection does NOT require double
compression — both body sizes are computable from the per-chunk
histogram (Huffman: Σ count×codelen from the table build that already
runs; tANS: the old tans_encoder.zig exact bit-count dry-runner,
resurrect from git). Run the dry-run selector over real L3/L5 chunk
streams on enwik8/enwik9/silesia and report chunks-flipped + end-ratio
delta. Expect flips concentrated in tokens/off16-hi (skewed, where
Huffman's integer-bit loss is largest) and few in text literals
(near-flat ~5-6 bits). If the delta is < ~0.1-0.2 pp, stop here.

**Build-out (if both gates pass)**: decode side is moderate — scan
already dispatches on chunk_type; descs partition into a Huffman
batch + a tANS batch with two predecode launches; the tANS decode +
table-build kernels exist in prototype under tools/huff_test. Encoder:
resurrect the host-side tANS encoder for the winning chunks only
(acceptable at L4/L5), GPU tANS encode later if it matters. Biggest
multiplier is the VK port (GLSL tANS decode + table build, in step,
per port-means-port). Wire format: re-introduce a tANS chunk type
(fresh tag; do NOT resurrect the retired 1/5/6/7 forms). The paired
combined-body trick (old types 5/7) is a real header-amortization
idea worth re-evaluating once the basic selector ships.

## 12. Small-items basket (carried from the retired todo.md deferrals)

- **`slzCompressAsync` input-size gap**: inputs outside [128 B, ~1 GiB]
  return SLZ_ERROR_UNSUPPORTED on the async path; the >1 GiB side is
  the caller-relevant one (TODO2's deferral-defender). Close by D2H-ing
  from `d_input` for the uncompressed-body path, or document loudly.
- **DevBuf abstraction** for the (ptr, size) pairs on both contexts —
  deferred twice already; only worth it bundled with another context
  refactor.
- **C ABI default-level mismatch** (Zig Options.level=1 vs C header 5)
  — documented as intentional; revisit only if the L2-alias decision
  (#6) changes level semantics anyway.
