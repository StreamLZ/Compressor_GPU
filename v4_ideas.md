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

## 1. Flat batched literal copy (token-ownership binary search) — ✅ DONE 2026-06-10, won less than projected

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

**DONE 2026-06-10** (both backends, `decodeSubChunkRawMode` /
`_SLZ_DECODE_RAW_BODY` — the PP path lives only in the raw-mode
decoder; the general decoder is serial per-token and untouched).
Implementation as specced: prefix sums staged to 512 B shared,
one flat warp-wide pass with the 5-step ownership search, then a
ballot-driven match loop (lit-only tokens now cost zero iterations).
**Measured: real but ~3× less than projected.** CUDA enwik9 L1
24.3 → 22.9 ms (−5.8%), L5 34.0 → 33.1 ms (−2.8%); enwik8 L1
2.85 → 2.64 ms. VK enwik9 L5 43.4 → 40.9 ms (−5.9% — VK gains more
because each eliminated copy call also drops two subgroup barriers).
vs nvCOMP margins: L1 1.36× → 1.44×, L5 1.49× → 1.54×; VK-vs-CUDA
gap 1.31× → 1.24×. Why short of ~1.2×: the literal-side overhead was
only part of the batch cost — the serial per-token MATCH loop (45%
of bytes, still one warpMatchCopy + sync per token) now dominates,
which is exactly #2's target. Verification: ptest 49/0/0, ptest_vk
150/9/0, both D2D sweeps all-verify-OK, enwik9 L3 1 GB SHA MATCH on
both backends. #2's projection should be re-derived from the
post-#1 profile before committing to it.

## 2. Flat independent-match copy + ordered fallback — ✅ DONE 2026-06-10, delivered ~2× what #1 did

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

**DONE 2026-06-10** (both backends, same wave as #1). The dependency
analysis collapsed to one provable condition: independent ⟺
`src + match_len <= dst_pos` (batch output start) — such a match
reads only pre-batch-final bytes, and since NO match's read range
extends past its own output end, dependents never read a later
token's output; hoisting all independents into a flat pass (same
ownership-search shape as #1, 3 more shared arrays) preserves
sequential semantics exactly. The entry's `dist >= 4` condition was
unnecessary at byte granularity. Dependent remainder stays in the
ballot loop. **Measured (cumulative #1+#2 in parens)**: CUDA enwik9
L1 22.9 → 20.8 ms (24.3 → 20.8, 1.17×), L5 33.1 → 29.5 (34.0 → 29.5,
1.15×); enwik8 L1 2.64 → 2.27 (2.85 → 2.27, 1.26×), L5 4.03 → 3.31
(4.06 → 3.31, 1.23×; LZ kernel alone 3.13 → 2.45); silesia L5 7.13 →
6.03. VK enwik9 L5 40.9 → 35.7 (43.4 → 35.7, 1.22×); VK D2D L5
kernels 5.62 → 4.41. nvCOMP margins: enwik8 kernel-sum L1 **2.01×**,
L5 1.73×; enwik9 kernel L1 1.59×, L5 **1.72×**. Combined #1+#2 came
in at 1.15-1.26× vs the projected 1.3-1.6× — the lower end of the
shrink the entry itself predicted for recent-offset-heavy text
(enwik9's deeper redundancy keeps more matches batch-local, hence
the smaller win at 1 GB than at 100 MB). Verification: ptest 49/0/0,
ptest_vk 150/9/0, both D2D sweeps verify-OK, 1 GB SHA MATCH both
backends, Intel iGPU decode L1+L5 SHA MATCH + cross-device encode
byte-identity. Next decode lever is structural: #15 (multi-warp) or
#8 (wire format).

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

**MEASURED 2026-06-10 — parked, don't build.** Instrumentation now
permanent: `SLZ_COUNT_PP` compile-time flag in lz_decode_raw.cuh
(device counters, default 0/zero-cost) + `-db` readback via
cuModuleGetGlobal under env `SLZ_COUNT_PP=1`. Results (raw-mode
decoder, post-#1/#2):
  enwik8 L1: serial tokens 0.16%, serial iters 4.8%, avg batch 31.1
  enwik9 L1: serial tokens 0.38%, serial iters 10.2%, avg batch 30.0
  enwik9 L5: serial tokens 0.29%, serial iters 8.2%, avg batch 30.4
The db1e061 truncation fix did the heavy lifting — long tokens cost
exactly one serial iteration each and batches stay ~30-full. The
in-lane long-token parse could only remove per-iteration overhead
(the long copies themselves stay), bounding the win well under the
iteration fractions. Not worth 1-2 days of fiddly scan code.

## 4. Tighten `total_subchunks` from worst-case BOUND to actual — ✅ DONE, verified 2026-06-10

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

**Status**: the exact-sum landed on BOTH backends during the
2026-06-10 A-024 region-offset wave (`uploadInputAndPrefixSum`,
host-path only; this entry pre-dated that commit and went stale).
Runtime verification completed 2026-06-10 evening — the first-ever
VK 1 GB L3+ decodes: enwik9 L3 AND L5 SHA-256 MATCH on RTX 4060 Ti;
L3 154.4 ms e2e / 44.9 ms kernels, L5 152.0 / 45.4 (-db, r=10).
e2e at parity with CUDA (PCIe-bound); kernel gap 1.32× vs CUDA's
34.4 = the A-021 fusion debt (#10), and VK now beats nvCOMP Zstd's
50.8 ms at 1 GB on kernels. RESIDUAL: the true-D2D path keeps the
worst-case bound (host can't read device descs) — at 1 GB L3+ D2D
that's 12 GB scratch again; noted in the #12 basket.

## 5. Close the VK A-024 residual permanently (BDA entropy scratch) — ✅ DONE 2026-06-10

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
**DONE 2026-06-10**: both kernels converted (huff_decode_4stream +
lz_decode general) via the A-008 recipe — three pre-offset region
addresses as push-constant lo/hi pairs, in-shader offsets stay
region-relative u32. Bonus: the A-024 3-dispatch huff split collapsed
back to ONE dispatch (CUDA-identical shape, region pick from
d_compact_counts), and the dead entropy_slot_stride left the ABI
(A-005 retired). KERNEL_DECLS: huff 7/4B → 5/24B, lz 8/16B → 5/32B.
Verified: ptest_vk 150/9/0, enwik9 1 GB L3+L5 SHA MATCH, D2D sweep
all-verified, Intel iGPU L5 SHA MATCH (BDA on both vendors); enwik9
L5 kernels 35.7 → 35.5 ms. The >4 GiB-offset regime (~1.2 GB+ inputs)
has no remaining known ceiling but awaits a test asset that large.


## 6. Re-differentiate L2 — ✅ DONE 2026-06-10: L2 = greedy + match-range rehash, LZ-only

**What**: Since hb=17-everywhere, L1 and L2 emit byte-identical
output. Options: (a) document L2 as an alias of L1; (b) repurpose L2
as the "max LZ ratio" profile — sc=0.5 (+~2 pp ratio at 1 GB, ~1.8×
slower large-input decode), giving the `--sc 0.5` tradeoff a stable
level name; (c) give L2 hb=18 back together with the A-023 batching
(now exists on both backends) so its 16 GB hash pages in batches
instead of thrashing.

**Why**: a level that duplicates another is API noise; (b) is the
most honest mapping of the real tradeoff surface we measured.

**RESOLVED 2026-06-10 with a fourth option found during the
discussion**: the match-range rehash (previously the L3→L4
distinction) does not require Huffman — it's a pure encoder-search
improvement, format-unchanged, decoder-unaware. L2 now = greedy +
rehash, entropy-free (one-line gate per backend: `level >= 4 or
level == 2`). Measured: enwik8 58.64% → **57.28%** (−1.36 pp),
silesia 47.83% → **47.21%**; encode kernel 70 → 91 ms (+29%, still
1052 MB/s); decode FASTER (2.27 → 2.12 ms enwik8 — fewer tokens).
Ladder now has a real step at every rung: L1 fastest-encode →
L2 +rehash → L3 +Huffman → L4 +rehash → L5 chain. Cross-backend L2
frames byte-identical; roundtrip SHA MATCH; both suites green.
Options (b)/(c) not taken: (b) remains available via `--sc 0.5` on
any level; (c) measured ≈0.1 pp, near-noise.

## 7. Backport the srcVK test suite to CUDA — ✅ first wave DONE 2026-06-10

The VK port accumulated 158 tests vs CUDA's 22; nothing flowed back
(port discipline ran one-way). First wave landed in
`src/encode/gpu_regression_tests.zig`:
- ✅ **A-023 forced-batch byte-identity** (batch=7 and batch=1 over a
  48-chunk input, L1 greedy + L5 chain, byte-identical to unbatched
  plus roundtrip) — was this entry's original ask.
- ✅ **Real-corpus roundtrips** (assets/web.txt full × L1-L5 + 4 odd
  offset/length slices) — the test class that caught VK's byte-65544
  while every synthetic shape passed.
- ✅ Two latent infra bugs found and fixed by writing them: BOTH
  module_loader `init()`s were thread-unsafe (latch set before the
  work / `.in_progress` returned false to concurrent callers — the
  exact race srcVK fixed with g_init_lock; all but one GPU test fn
  had been silently SKIPPING in every ptest run), and once init was
  locked, the GPU test fns raced each other on the shared g_default
  contexts — now serialized behind `lockGpuTests()`. ptest went
  19 pass/4 skip → **25 pass/0 skip/0 fail**.

**Remaining backlog**: NONE — the VK→CUDA parity-debt ledger
(BACKPORTS.md, 2026-06-10) was fully executed and retired the same
day (`git log --follow -- BACKPORTS.md` for the audit trail: A-017
compact fusion + parallel merge, device-name print, per-kernel `-db`
timings, encode phase profiler, sanitize.bat, huff conformance,
C ABI, cli_smoke, L5 hardening, `zig build ptx`). The SHA
byte-identity gate moved to CLAUDE.md as a standing rule. The two
deliberately-deferred tails live in the #12 basket below.

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

## 10. A-021: fuse the remaining VK decode bookkeeping dispatches — ✅ DONE 2026-06-10

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

**2026-06-10 update**: CUDA now provides direct reference kernels to
mirror — `slzCompactAllDescsKernel` (5-way fused compact incl. the
raw pair, `ee925e4`) and `slzMergeHuffDescsParKernel` (4-block
parallel merge, 0.199 → 0.067 ms CUDA-side, `ec6071d`). Port those
shapes rather than designing fresh; see the A-017/A-021 2026-06-10
updates in srcVK/PortAdaptations.md.

**DONE 2026-06-10 (same day)**: both kernels mirrored
(`compact_all_descs_kernel.comp` grid_x=5 as the only compact path,
`merge_huff_descs_par_kernel.comp` grid_x=4 with serial fallback —
both matching CUDA's host shapes). Dispatches 12 → 10 per L3+
decode. Measured: enwik9 kernels L3 44.9 → 42.9 ms, L5 45.4 →
43.4 ms — gap vs CUDA 1.32× → 1.26×; enwik8 L5 D2D 5.77 → 5.62 ms.
SHA MATCH at 1 GB; ptest_vk 150/9/0. The residual ~1.26× lives in
kernel_fn itself (the LZ workhorse) per the A-021 attribution note —
that's #1/#2/#15 territory, not more fusion.

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
- **The 4× table-build gap is attackable — root cause located
  (2026-06-10).** `initLut` in `tools/huff_test/tans_decode_kernel.cu`
  runs the whole 4-way interleaved spread ON LANE 0: a serial
  double loop assigning up to 2048 slots one at a time while 31
  lanes idle. That IS the 0.611 ms.
  **Implementation spec for the parallel spread** (next session,
  harness-ready, verify = BIL decode stays byte-exact against the
  new LUTs):
  1. The slot recurrence is a pure function of cumulative weights:
     symbol s (weight w, cum-weight ws) puts `y_j(w, ws) =
     (w + ((ws - j - 1) & 3)) >> 2` slots into quarter j, at quarter
     cursor = `q_base[j] + prefix-over-prior-symbols of y_j`.
  2. So: Hillis-Steele warp scan over weights (32 symbols per batch,
     carry across batches) gives every symbol its ws; four more
     scans over y_j give its quarter cursors; then EVERY SYMBOL
     fills its own slots independently — lane l takes symbols
     l, l+32, ... and writes its w slots with
     `running_w = w + cursor` advancing j=0..3, k=0..y_j in the
     original order (order inside a symbol matters; order between
     symbols no longer does).
  3. Weight-1 symbols fill the `slots_left..L-1` tail — embarrassingly
     parallel already.
  **IMPLEMENTED + MEASURED same day** (`slzTansFseBuildParKernel` +
  `buildPackedLut4WayParallel` in tans_upgrade_kernels.cuh): 0.611 ms
  → **0.228 ms**, verified byte-exact via BIL decode against the new
  LUTs. One subtlety found the hard way: the serial spread's
  `weights_sum` accumulates ONLY w>1 symbols (weight-1 symbols are
  tail-only), so the scan input is `w>1 ? w : 0`. The residual
  0.228 ms is dominated by the lane-0 serial FSE bit-parse
  (`decodeFseWeights` — adaptive field widths, inherently
  sequential); near the floor for this table format. The table-build
  strike against the selector is now 1.4× (0.228 vs 0.160), down
  from 3.8×. No MATCH.ANY needed — the prefix-scan form ports to VK
  with plain subgroup ops.
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

**Gate 1 — MEASURED 2026-06-10** (`[gate1]` in the harness driver:
real production tANS sizes from the snapshot vs exact canonical
Huffman bit-counts on the same decoded bytes, production framing on
both sides, enwik8-L5 lit+tok, 3052 chunks / 27.8 MB raw):
- huff-only 20.963 MB, tans-only 20.885 MB (tANS WINS overall on
  these streams by 0.37%), per-chunk min 20.850 MB.
- **Selector saves 0.54% of entropy-stage bytes vs huff-only;
  1618/3052 chunks (53%) flip to tANS.**
- Scaled to the frame: ≈ 0.29% smaller .slz ≈ **0.1 pp of end ratio**
  on this corpus slice — right at the go/no-go bar. Untested upside:
  the off16 hi/lo planes (more skewed than lit/tok → tANS should win
  bigger) and enwik9-class inputs (deeper redundancy). Known
  approximations: huff stream/word padding estimated flat (+64 B/chunk)
  and 11-bit escape pairs not modeled — both small and roughly offsetting.
- Speed economics post-upgrades: flipping 53% of chunks costs
  ~nothing (tANS decode 40.8 GB/s vs huff 52.8; build 0.228 vs
  0.160 ms; predecode is <1 ms of a 4+ ms decode).
**Next measurement before any pipeline work**: extend the dry-run to
the off16 planes + an enwik9 L5 snapshot — if those push the total
toward ~0.2 pp, the selector is worth building; if it stays ≈0.1 pp,
park it.
**Scoped 2026-06-10**: the extension needs REAL tANS sizes for data
that was never tANS-encoded (off16 planes; enwik9), and a size MODEL
would swamp the 0.1-vs-0.2 pp signal — so the prerequisite is
resurrecting the retired host tANS encoder as a measurement tool:
`git show 8982fee^:src/encode/entropy/tans_encoder.zig` (1388 lines;
needs src/io/bit_writer.zig from the same commit), plus an
encoder-side per-sub-chunk plane dump hook (encode_huff.zig has the
raw lit/tok/off16-hi/lo bytes AND the exact produced huff sizes —
better than gate1's +64 B approximations). Own-session-sized.

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
- **Host-unit mirrors** (from the retired BACKPORTS.md D table):
  port `srcVK/tests/{decoder,encoder}_unit.zig` cases (33 tests —
  descriptor walking, header building) AS-TOUCHED — only when next
  modifying that logic, not as a standalone task.
- **Runner-level serial phases** (retired BACKPORTS.md, optional):
  replace the lockGpuTests SRWLOCK with srcVK's 3-phase runner design.
  Got LESS attractive 2026-06-10: the lock now also binds the CUDA
  context per thread; a redesign would re-solve that for cosmetic
  gain. Only revisit if the lock pattern causes a real failure.
- ~~**VK D2D walk-batching mystery**~~ ✅ SOLVED 2026-06-10 (same
  day): not the walk at all — stream-path D2D copies recorded into
  the TRANSFER cmdbuf, which submits BEFORE the compute leg, so the
  chunk-descs copy read walk output before walk ran (and the
  output-side copy latently read the PREVIOUS decode's output).
  Fix: compute-cmdbuf-ordered D2D + `d2d_input_offset` transfer-leg
  variant for the input frame copy; walk now batches (workaround
  removed). Full record: PortAdaptations A-026. The hoped ~0.4-0.5 ms
  was mostly the walk kernel itself (which overlaps nothing); real
  win was ~0.1 ms at L1/L2 plus a genuine correctness fix.
- **D2D entropy-scratch sizing still worst-case** (#4 residual,
  2026-06-10): the true-D2D path can't host-sum the chunk descs, so
  1 GB L3+ D2D would allocate the old 12 GB scratch. Cheap exact
  option: D2H the 4-byte `d_total_subchunks_buf` after prefix-sum —
  but that adds a mid-pipeline sync point, undoing the A-026
  single-submit batching. Only matters if 1 GB-class D2D becomes a
  real workload; bundle with any D2D scale work.
- **Gather-overlap** (retired BACKPORTS.md B2 tail): run
  slzGatherRawOff16Kernel on a second stream under merge+LUT-build.
  Prize ~0.07 ms (enwik8) / ~0.85 ms (1 GB); cost is cross-stream
  ordering complexity in decode_dispatch. Bundle with any future
  stream-architecture work, not standalone.
- **LOP3.LUT 0xfe verification** (retired GPU_IDEAS idea 5): confirm
  nvcc already fuses the BIL-refill byte-merge ORs into LOP3; 1-2 h,
  upside <=1-2% on a 98%-memory-bound kernel - verify, don't refactor.
- **Packed butterfly reduction in the BIL encoder** (retired
  GPU_IDEAS idea 6): ~30 min encode-side polish, minor.
- **BUG: L3+ true-D2D decode fails** (found 2026-06-10 by
  toolsench_d2d.bat during the README perf refresh): slzDecompressAsync
  verify-FAILs on enwik8 L3 with zero kernel time (instant return);
  L1/L2 D2D pass. Suspect the 2026-06-09 A-024 region-offset rework -
  the D2D path had no L3+ coverage (c_abi_tests D2D case runs at L1).
  FIX FIRST, then extend c_abi_tests.zig's D2D roundtrip to L5, then
  re-measure the README L5 async-wall cell (marked with * there).

## 13. Fuzzing the decoder (and differential CUDA-vs-VK fuzz)

Not VK parity debt (neither backend has fuzzing) — new work. The
exposure is real and GPU-specific: `slzWalkFrameKernel` parses
untrusted frame bytes ON DEVICE, so a malformed frame doesn't segfault
like a CPU parser — it OOB-reads/writes silently in VRAM or spins a
kernel into the 2 s WDDM TDR watchdog (= driver-level device reset, a
DoS primitive if frames ever arrive from untrusted sources).

Classic coverage-guided fuzzing doesn't transplant: GPU launch
overhead caps exec rate at ~100-1000/s, there is no SanitizerCoverage
for PTX (fuzzer is blind inside kernels), OOB needs compute-sanitizer
(10-100x slowdown) to be visible at all, and a TDR kills the CUDA
context so the harness must treat device-reset as a finding and
re-init.

Three tiers, cheapest first:
1. **Mutation fuzz (decoder rejects gracefully)**: mutate the golden
   .slz frames — structure-aware (walk header fields, chunk sizes,
   sub-chunk counts) plus random flips/truncation — and assert clean
   rejection: correct error code, no hang, no TDR, no sanitizer
   findings (run a sampled subset under tools/sanitize.bat).
2. **Roundtrip property fuzz (encoder)**: randomized structured inputs
   × random level/sc, byte-identity assert. The D-wave L5 hardening
   cases are hand-picked instances of this; fuzzing generalizes them.
3. **Differential CUDA-vs-VK fuzz** (the standout): feed both backends
   the same mutated frame, diff accept/reject decision AND output
   bytes. Two independent implementations of one format = a free
   oracle; any divergence is a bug in one of them. This is the
   project's "runtime oracle" principle applied systematically.

Prereq: audit what the walk kernel currently validates (status field
coverage, WALK_MAX_CHUNKS, size cross-checks) so tier 1 starts from a
documented contract of what "graceful rejection" means per corruption
class. Effort: tier 1 ~a day; tier 3 ~a day on top (mostly harness
plumbing — both CLIs already exist).

## 14. MATCH.ANY warp-cooperative match finder for the L5 chain parser

(Merged from the retired docs/GPU_IDEAS.md idea 1; full analysis incl.
the nvCOMP SASS references in git history.)

**What**: the L5 chain parser runs serially on lane 0 (~1.1 active
threads/warp per NCU). nvCOMP-style warp window: each lane builds a
rolling 4-byte key via two shuffles, `__match_any_sync` returns each
lane's set of key-equal lanes in one instruction, BREV+FLO picks the
nearest in-warp match - match-finding inside the 32-byte window costs
ZERO memory traffic; only matches escaping the window fall through to
the global hash chain.

**Why NOW**: the C3 phase profiler (2026-06-10) measured the LZ chain
parse at **86% of L5 encode wall** (266 of 311 ms on enwik8) - it is
THE encode-perf target, and this is the only credible plan on file
for it. Expected 2-3x on the chain parse for repetitive data.

**Risks**: parser restructure must preserve token emission +
recent-offset + chain-validation semantics; found-match set may shift
(ratio drift) - gate with the cross-backend SHA expectation updated
deliberately, never silently. Harness-first in a new `tools/lz_test/`
(CPU oracle, MATCH.ANY-only variant first to bound the in-warp win).
Effort: 1-2 weeks. Any change must be mirrored to srcVK
(`subgroupShuffle` + the A-002 match_any emulation already exist).

## 15. Multi-warp-per-chunk LZ decode — ✅ DONE 2026-06-11: 2-warp pipeline, default ON

(Merged from the retired docs/GPU_IDEAS.md idea 3.)

**What**: regrid the LZ decode to `(num_chunks, K)` with K=2-4 warps
pipelining per chunk (warp 0 parses batch N while warp 1 executes
batch N-1) over a shared-memory token ring + one `__syncthreads()`
per batch.

**Why**: FAILED_EXPERIMENTS' meta-conclusion - PP-v2 is the ceiling
for single-warp latency attacks; remaining gains need STRUCTURAL
parallelism. This is the named open structural lever.

**Sequencing**: strictly after #1/#2 (flat batched copies) - same
kernel, and #1/#2's lane-efficiency win changes this idea's
cost/benefit. Could be 0% (ring serialization) or 10-25%;
harness-first A/B mandatory. Effort: 1-2 weeks, most speculative
item on this list.
**DONE 2026-06-11.** NCU gate first (admin counters unlocked): the
post-#1/#2 raw kernel measured 52.7% SM throughput (was ~92%), 22.9
long_scoreboard stall, 0.0 barrier stall — the issue pipe was no
longer saturated, so the structural lever had room. Implementation
(`lz_decode_raw_pipeline.cuh`, K=2): warp 0 parses batch N+1 (PP
scans → PipeBatch shared double buffer) while warp 1 executes batch
N's flat copies; __syncthreads()-paced (intra-block spin-waits
deadlock — no warp-scheduler fairness; measured instantly). Serial
long tokens drain the pipeline then run cooperatively. Two bugs cost
a debugging session: the prime-failure path skipped the serial
handler (most chunks open with a long match at the 64 KB boundary),
and `(1u << batch_size) - 1u` at batch_size==32 is UB that silently
dropped every dependent match of full batches. Measured (default ON,
SLZ_NO_PIPELINE=1 escape): enwik8 L1 2.28 → **1.93 ms**, enwik9 L1
20.9 → **17.2 ms** (58.2 GB/s, nvCOMP LZ4 margin 1.92×), silesia L1
4.16 → 3.87. D2D async wall L1 4.04 → **3.27 ms** (1.46× vs nvCOMP).
Cumulative #1+#2+#15 on enwik9 L1: 24.3 → 17.2 = **1.41×**. ptest
50/0/0 with pipeline forced; 1 GB SHA MATCH. Side-find: the default
`zig build` does NOT rebuild streamlz_gpu.dll — every D2D number
since 2026-06-10 15:29 measured a stale DLL (`zig build gpulib` now
required; README async cells were re-measured fresh). REMAINING: VK
mirror (port-means-port — divergence OPEN until then; GLSL barrier()
replaces __syncthreads, same PipeBatch shape), and K=3/4 variants
were not explored (the 1-deep __syncthreads pipeline already
captures most of the overlap; deeper needs persistent-threads).
**Post-#15 NCU profile (2026-06-11, slz_lz_pipe_post15.ncu-rep)**:
SM 63.3% (was 52.7), memory 63.3%, long_scoreboard 22.9 → **8.9**
(the pipeline did its job), but **barrier stall 9.5 is now the top
stall** — the __syncthreads pacing itself. Escalation ladder, in
order (STRATEGY: finish ALL of these on the L1/L2 raw kernel FIRST,
then port the finished design to the L3+ general kernel once —
solve complexity on the simple kernel, port second):
1. **mbarrier depth** — cuda::barrier<thread_scope_block> per-slot
   arrive/wait replaces the full-block __syncthreads rendezvous;
   directly attacks the measured 9.5 barrier stall; the supported
   form of the spin-wait that deadlocked. sm_80+, works in the
   nvcc→PTX flow.
2. **K=4 role-split copiers** — warp1 flat literals, warp2 flat
   independent matches (provably concurrent with literals: disjoint
   writes, pre-batch reads), warp3 dependent matches (waits on lit
   pass). Widens the flat passes; SM+memory both at 63% = headroom
   on both axes. Composes with mbarrier.
3. **L3+ general-kernel port LAST** (after 1+2 are proven): the
   OFF16_SPLIT=true PP path in the general kernel never got #15;
   L5 LZ slice is 2.5-3.2 ms. Port the finished design once.
Combined 1+2 honest estimate: 10-20% more on the raw kernel
(enwik9 L1 17.2 → ~14-15.5 ms).
**Escalations 1+2 RESOLVED 2026-06-11 (same session):**
mbarrier (escalation 1) measured and REJECTED — cuda::pipeline ring
S=2/S=4 = 2.18/2.64 ms vs 1.90 __syncthreads on enwik8 L1; the
barrier stall was inherent slower-side wait, not rendezvous cost
(FAILED_EXPERIMENTS.md "mbarrier pipeline ring"). K=4 copier team
(escalation 2) SHIPPED default: 1 parser warp + 3-warp copier team
(96 flat-copy lanes, two tight passes — merged pool measured 3.00 ms
and lost), team-internal order via __barrier_sync_count(1,96) named
barrier (inline-asm bar.sync = optimizer wall, +0.8 ms), one
__syncthreads per batch for the slot handoff, blockDim (32,4),
12 blocks/SM. Measured: enwik8 L1 1.90 → **1.77 ms** (kernel-sum
1.88, nvCOMP LZ4 2.54x), enwik9 L1 17.15 → **16.22 ms** (61.7 GB/s,
nvCOMP 2.03x), D2D wall 3.27 → 3.13 (1.53x); silesia L1 pays 3.87 →
4.05 (binary = parser-bound + blocks/SM halved; still under the
pre-#15 4.16). K=3 rejected (uneven SMSP packing: 19.0 ms e9).
ptest 50/0/0, 1 GB SHA MATCH, D2D verify OK. Remaining knob if
silesia matters more later: ship both K=2 and K=4 kernel entries and
pick per frame by token/lit statistics from the walk. L3+ port and
VK mirror now target THIS design (K=4 + named barrier).
**K=4 NCU verdict (2026-06-11, post-ship re-profile)**: SM 71.9%,
memory 71.9% (lockstep — balanced saturation), warps 92.9%,
long_scoreboard 6.6, barrier 8.2. The residual barrier stall is the
lockstep tax (faster side waits per batch) — the mbarrier
alternative measured worse, so this is the design's floor. L1 raw
kernel is DONE at 52.7 → 71.9% SM and 2.28 → 1.77 ms in one wave;
remaining levers are #8 (wire format, changes the work itself) or
K=8 width (would crater parser-bound corpora; silesia already pays
at K=4). Next: port THIS design to the L3+ general kernel.
**L3+ port DONE 2026-06-11 (same day):** slzLzDecodeGeneralPipelinedKernel
reuses parseSubChunkHeaders on the parser warp + shared ParsedStreams
broadcast; mode-1/off32-free sub-chunks take the K=4 pipeline
(templated on off16_split), the rest fall back to warp-level
decodeSubChunkGeneral in-kernel. enwik8 L3/L4/L5 kernels 3.49/3.42/3.31
-> 2.88/2.93/2.93 ms; enwik9 L5 29.5 -> 26.27 (nvCOMP Zstd 1.93x);
silesia L3+ all improved; D2D L5 wall 4.80 -> 4.49 (1.39x). All SHA +
suite gates green. v4 #15 is now COMPLETE on CUDA across all levels;
remaining: VK mirror (TDR fix first).
**VK TDR SOLVED 2026-06-11 (late session):** the hang was the parse
broadcast APPARATUS, not the parse — `if (is_parser)` gating +
`s_parse[8]` shared broadcast + `barrier()` in the bridge TDR'd at
multi-workgroup scale on the NVIDIA Vulkan driver even with the
decode body disabled. Fix: BOTH subgroups run the header parse
redundantly (identical register values from the same read-only
bytes — the exact structure of the proven per-warp macro); no
gating, no shared, no barrier in the parse. Full pipeline (prime +
steady overlap + serial long tokens) now SHA MATCH on enwik8 L1 at
1526 workgroups. Perf lesson while fixing: GLSL memoryBarrierBuffer
is a DEVICE-scope fence — one per batch cost +26% kernel time; the
pipeline needs it ONCE per drain (before the serial path reads
copier output), with subgroup-scope fences inside executeBatch
covering per-batch copier-lane visibility. STATUS: correct but
SLOWER than the single-warp kernel (2.89 vs 2.36 ms enwik8 L1
kernel) — the per-batch workgroup barrier() costs more on the VK
driver than the K=2 overlap saves. Stays OPT-IN (SLZ_VK_PIPELINE=1);
the path to VK perf is the K=4 redesign (3-warp copier team — needs
a named-barrier substitute; GLSL has none, so likely full barrier()
with the parser parked or a 2-subgroup team layout), not tuning the
K=2 form. ptest_vk 151/9/0 on the default path.
**VK K=4 measured 2026-06-11 (same session): pipeline REJECTED on VK
— A-028.** K=4 ported (96-lane team flats, deps on warp 1 gated by a
uniform dep_mask check so dep-free batches pay one barrier,
groupMemoryBarrier for workgroup-scope dst visibility), byte-correct
— and slower still: single-warp 2.36 / K=2 2.89 / K=4 3.18 ms
(enwik8 L1 kernel). On the NVIDIA VK driver the per-batch workgroup
barrier costs more than overlap + width recover — opposite of CUDA.
VK default STAYS single-warp; both pipeline forms remain opt-in
(SLZ_VK_PIPELINE=1). Catalogued as PortAdaptations A-028 (accepted,
measured). The VK-vs-CUDA decode gap (1.33-1.35×) is now a
documented architectural cost of the driver's barrier pricing, not
an unported optimization.








## 16. Dictionary support (zstd-style preset dictionaries)

**What**: accept an external dictionary at encode AND decode time —
the classic zstd `-D` shape: prepend D bytes of shared context the
match finder may reference (negative offsets at chunk start) without
emitting them. Wire format: a dictionary ID (e.g. xxhash of the dict
bytes) in the frame header so decode can refuse a missing/mismatched
dict. API: `slzCompressWithDict(dict, ...)` /
`slzDecompressWithDict(dict, ...)` plus CLI `-D <file>`.

**Why**: the small-input story is currently weak — a 4 KB JSON blob
barely compresses because every chunk starts cold. Dictionaries are
THE standard fix and the main feature gap vs zstd/nvCOMP for
many-small-records workloads (logs, KV stores, RPC payloads). Also a
natural fit for the GPU batch shape: one dict staged once in VRAM,
thousands of small records decoded against it in one launch.

**Sketch**: decode side is the easy half — stage dict bytes
immediately before each chunk's output window (or address them via a
second SSBO/base pointer) so back-references reaching past the chunk
start land in the dict; the LZ kernels' copy loops need a base-adjust,
not a redesign. Encode side: seed the hash/chain tables with dict
positions before parsing real input (hb tables already exist; dict
entries get negative-space indices). Levels: greedy (L1) first, chain
(L5) after. Both backends in step per port-means-port; cross-backend
SHA gate extends to dict frames.

**Cost/risk**: medium — wire-format addition (header dict-ID field),
new ABI surface, and the negative-offset window touches the decode
kernels' bounds logic (corruption risk class, needs the full test
battery + fuzz tier from #13). No ratio/speed effect on dict-less
frames. Gate on a measured win: build the host-side prototype first
and measure ratio lift on a real small-records corpus (e.g. 10k JSON
docs) before touching kernels.
