# VK → CUDA backport ledger

The srcVK port was developed CUDA→VK one-way, but the port effort
produced fixes, optimizations, tooling, and tests that CUDA never
received. This is the comprehensive audit (2026-06-10), assembled from
`srcVK/PortAdaptations.md`, `srcVK/ToDo.md` (lessons learned), the
srcVK test suite, and direct source comparison. Companion to
`v4_ideas.md` (forward-looking work) — this file tracks parity debt
specifically.

Status legend: ✅ done · 🔲 open · 🚫 evaluated, not applicable.

## A. Already backported (audit trail)

| Item | VK origin | CUDA landing |
|---|---|---|
| ✅ L1/L2 decode gate (skip scan/compact/gather/merge/huff) | VK level≥2 gate, tightened to ≥3 both sides | `db1e061` |
| ✅ A-023 batched LZ dispatch (VRAM-budgeted encode) | `srcVK/encode/encode_lz.zig` | `db1e061` (cuMemGetInfo budget variant) |
| ✅ Entropy-scratch allocation gated to level≥3 | VK strict-VMA necessity | `db1e061` / `b77067a` |
| ✅ Exact `total_subchunks` (replaces 2× worst-case bound) | landed both sides in step | `b77067a` |
| ✅ `-b`/`-ba` encoder-buffer trim before decompress | VK OOM exposed it; fixed both | `b464f89` |
| ✅ `init()` thread-safety (SRWLOCK, latch-after-work) | srcVK `g_init_lock` / `g_encode_init_lock` lesson | `eb5cf44` |
| ✅ GPU-test serialization under the parallel runner | srcVK race-class lesson #3 | `eb5cf44` (`lockGpuTests`) |
| ✅ A-023 forced-batch regression test + hook | `srcVK/tests/a023_batched_lz_dispatch.zig` | `eb5cf44` |
| ✅ Real-corpus roundtrip tests (web.txt + slices) | srcVK l3_l4 corpus suite (byte-65544 lesson) | `eb5cf44` |

## B. Performance backports — open

### B1. 🔲 Fuse the 4 `slzCompactHuffDescsKernel` launches (A-017 mirror)

VK fused its four compact_huff dispatches (lit/tok/hi/lo) into one
grid-spanning kernel for a 2.4× kernel-time win (0.59 → 0.25 ms,
`srcVK` A-017). CUDA still launches the kernel FOUR times per decode
(`src/decode/scan_gpu.zig:221-237`), measured at 4 × ~75 µs = ~0.3 ms
per enwik8-L5 decode (nsys 2026-06-10) — ~7% of the 4.3 ms d2d.
Mechanical: pass a stream-selector via blockIdx.y (or grid.x
partition), keep per-stream params in a small array. Verify: ptest +
SHA on the goldens. **Effort: hours. Expected: ~0.1-0.2 ms off every
L3+ decode.**

### B2. 🔲 Evaluate fusing compact_raw + gather_raw_off16 + merge_huff

The A-021 close-path on VK (v4_ideas #10) applies in spirit to CUDA
too: nsys shows merge 0.20 ms + compact_raw 0.08 + gather 0.08 per
decode. Smaller prize than B1 and the kernels have ordering deps —
measure after B1 lands before deciding.

## C. Observability / tooling backports — open

### C1. 🔲 Print the device name in the CUDA CLI

VK prints `Device: NVIDIA GeForce RTX 4060 Ti` at startup (and the
project memory REQUIRES device names next to perf numbers). The CUDA
CLI prints nothing — `cuDeviceGetName` at init + one line in
bench/info modes. **Effort: <1 hour.**

### C2. 🔲 Surface per-kernel timings in `-db`

The begin/endKernelTiming infrastructure and the C ABI
`slzGetLastTimings` already exist on CUDA, with labels at every launch
site — but the CLI never surfaces them; we had to use nsys to get
`slzHuffDecode4StreamKernel` numbers (2026-06-10). Add a `-kt` flag
(or env var) to `-db` printing the per-kernel table, mirroring VK's
`SLZ_VK_PROFILE_DECODE` query-pool report. **Effort: hours.**

### C3. 🔲 Encode-side phase profiler

VK's `SLZ_VK_PROFILE_PHASES` QPC accumulators (encode + decode)
directly located the 238 ms d2h_final bottleneck that became a 3.2×
encode win. CUDA has only the decode-side `SLZ_E2E_TIMER` (with two
known-stale columns — TODO2 items, fields sampled at the same
instant). Backport the per-phase accumulator pattern to
`fast_framed.zig` / `encode_lz.zig` and fix the decode columns while
in there. **Effort: ~half day. Value: prerequisite for the next
encode-perf push.**

### C4. 🔲 compute-sanitizer as the validation-layer analog

srcVK lesson: "always run validation before claiming done" caught 3
real bugs. CUDA equivalent: a `compute-sanitizer --tool memcheck`
(and racecheck) pass over ptest + a 1 GB decode, run at milestones.
Document the invocation in a tools/ script. **Effort: hours to
script; minutes per use.**

## D. Test-suite backports — open (see v4_ideas #7 for the wave-1 detail)

| Item | VK source | Effort |
|---|---|---|
| 🔲 Huffman kernel conformance (isolated encode→LUT→decode, 5 cases; catches bitbufRefill-class bugs that e2e masks) | `srcVK/tests/huff_decode_conformance.zig` + CPU oracle in `tools/huff_test/huff_ref.c` | ~1 day |
| 🔲 C ABI tests — CUDA `slzCompressAsync`/D2D/timings currently has ZERO tests (old ABI tests were src_vulkan's, deleted) | `srcVK/tests/async_d2d_api.zig` | ~1 day |
| 🔲 CLI smoke tests (subprocess -c/-d/-b against goldens; env-inherit lesson) | `srcVK/tests/cli_smoke.zig` | hours |
| 🔲 L5 chain-parser hardening cases (long-match truncation, mixed-class offsets, block 1→2 recent_offset carry) | srcVK L5 hardening (+4, `f08713d`) | hours |
| 🔲 Host-unit mirrors (descriptor walking, header building) where logic is shared | `srcVK/tests/{decoder,encoder}_unit.zig` (33 tests) | as-touched |
| 🔲 (optional) Runner-level serial phases instead of the lockGpuTests mutex | `srcVK/test_runner_parallel.zig` 3-phase design | optional |

## E. Process practices worth adopting symmetrically

- **SHA byte-identity vs the other backend on the 3 goldens after any
  encoder change** — VK treats this as MANDATORY (the H2 lesson). CUDA
  changes get this implicitly only when someone re-runs VK cross-tests.
  Make it a stated gate for CUDA encoder PRs too.
- **Runtime oracle over static diff** — already in project memory;
  applies in both directions.
- **PTX freshness QoL**: the gate only errors; a `zig build ptx` step
  that shells nvcc via vcvars (mirroring how `build_vk.bat` was
  retired by the depfile fix) would remove the manual
  touch-the-other-PTXs dance. Optional.

## F. Evaluated — NOT applicable to CUDA

- **A-008 BDA hash addressing** — CUDA pointers are already 64-bit raw.
- **A-024 per-region scratch bindings / 3-dispatch huff decode** —
  works around maxStorageBufferRange; CUDA uses u64 params (1 dispatch).
- **Persistent VkPipelineCache** — CUDA's JIT/SASS cache is driver-side.
- **d2h_offset_gather / single-submit patterns** — solve the
  Vulkan-on-WDDM per-submit floor; CUDA's sync copies are ~37 µs.
- **glslc -MD depfile graph** — CUDA analog is the PTX gate (see E).
- **Subgroup-size pinning, SPIR-V Int8 patcher, A-006/A-007/A-013/A-014
  language workarounds** — VK-only by construction.
