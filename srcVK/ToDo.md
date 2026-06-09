# srcVK Vulkan Port — L1-L5 Done, Async/Perf Roadmap

**Last updated:** 2026-06-08, HEAD `c141fdf` + uncommitted Phase 2B.
**Status:** L1 + L2 + L3 + L4 + **L5 SHIPPED on decode + encode** (all 5 levels VK→VK roundtrip MATCH on web + enwik8 + silesia, byte-identical to CUDA SHA on L1, L2, L3-web/enwik8, L4, L5). Phase 2A encoder + Phase 2A-decoder + Phase 2B + Phase 3 + Phase 2A.5 + iter 4f all done. **ptest_vk: 108/0/0 on both NVIDIA + Intel iGPU.** Only remaining encoder residual: silesia L3 is 0.019% larger than CUDA L3 (A-008, accepted — Vulkan 4 GiB SSBO cap on hash table forces `hash_bits=18` clamp on inputs > 128 MiB; VK→VK silesia L3 decodes its own output correctly). Remaining work: async API, perf parity sweep, BDA workaround. See `srcVK/PortAdaptations.md` for the catalog of CUDA-VK divergences + their resolution status.

**Build-graph note**: As of 2026-06-08 the build graph at `build.zig::addSrcVkShaderSteps` correctly tracks `.glsl` `#include` deps via glslc's `-MD` depfile output (A-012 RESOLVED). Plain `zig build streamlz_vk` after editing any srcVK `.glsl` header now invalidates the right SPIR-V blobs. `tools/build_vk.bat` is kept as a force-clean utility but is no longer required after `.glsl` edits.

---

## CRITICAL HYGIENE RULES (read first if you're a new agent on this codebase)

1. **Read `srcVK/PortAdaptations.md` BEFORE making any new VK-vs-CUDA divergence**, and ADD a new A-NNN entry if you introduce one. Each entry must declare both **static** AND **runtime** verification status. "NOT DONE" on runtime verification is a known risk surface and should block ship unless explicitly approved.

2. **Adversarial reviews must verify runtime code-path matching, not just per-line static matching.** 8 rounds of static diffs missed A-001 because they asked "do these lines match?" not "does VK reach the same line CUDA reaches for this input?" Use CUDA as runtime oracle: dump variable values from both backends + diff. ~30-60 min single agent run vs hours of static cycle.

3. **5-min static-diff probes hit diminishing returns fast.** If 3 rounds haven't found the bug, switch to runtime instrumentation (dump VK + CUDA values to JSONL, side-by-side diff). The setup is ~30-60 min one-time but produces the answer.

4. **"Legitimate VK adaptation"** is a high-risk classification. Most ARE legitimate (GLSL really lacks the CUDA feature), but each one creates a code path CUDA never exercises — meaning CUDA can't be the test oracle for that path. Add to PortAdaptations.md + verify the divergent path explicitly.

5. **GPU benchmarks must run serially.** Never parallelize VK + CUDA bench runs (or VK + VK) — they contend for GPU/WDDM resources and produce biased numbers. Run one bench, wait for completion, then the next.

(Retired 2026-06-08: the "always use tools/build_vk.bat after editing .glsl" rule — A-012 is resolved; plain `zig build` now invalidates correctly. The script is kept for force-clean scenarios but is no longer mandatory.)

This file is the source of truth for "what's left." All of L1 is locked-in
production code; the meaningful remaining work is L2-L5 to reach full CUDA
parity. The `/src_vulkan/` directory contains an earlier (now-deprecated)
implementation that ported less faithfully; its `TODO.md` informed parts of
this roadmap.

---

## Status: L1 is DONE on both decode AND encode

| Criterion | NVIDIA RTX 4060 Ti | Intel(R) Graphics iGPU |
|---|---|---|
| Structural 1:1 port audit | ✅ | ✅ |
| Byte-equal roundtrip (web + enwik8 + silesia) | ✅ | ✅ |
| Cross-backend (CUDA ↔ VK both directions) | ✅ | ✅ |
| **Decode** perf ≤ 1.10× CUDA on large workloads | ✅ **0.97-0.98×** | works |
| **Encode** perf ≤ 1.10× CUDA on large workloads | ✅ **0.82-0.87× (FASTER)** | byte-identical, works |
| **Encode SHA byte-identical to CUDA on all 3 goldens** | ✅ | ✅ |
| ptest_vk | **74/0/0** | **74/0/0** |
| VK validation (spec) | 0 VUIDs | 0 VUIDs |
| VK validation (synchronization) | 0 hazards | 0 hazards |
| VK validation (best practices, bench mode) | 0 | 0 |

Current bench numbers (NVIDIA, `SLZ_VK_DEVICE_INDEX=1`, `-db -r 5` or `-b -r 5`, post-warmup):

**L1 decode + encode:**
| File | Decode e2e | Decode kernel | Encode | vs CUDA decode | vs CUDA encode |
|---|---|---|---|---|---|
| enwik8 (95 MB) | 15.7 ms | 3.0 ms | **103 ms** | 0.97× | **0.82×** |
| silesia (203 MB) | 30.0 ms | 5.5 ms | **209 ms** | 0.97× | **0.87×** |
| web.txt (4.3 MB) | 5.3 ms | 1.3 ms | 19 ms | 2.6× (WDDM floor) | 1.73× (WDDM floor) |

**L5 decode (post-A-016 + A-017 — u32 store fix + fused compact_huff, 2026-06-08):**
| File | Decode e2e | huff_decode | compact_huff (×4 fused) | lz_decode | vs CUDA decode |
|---|---|---|---|---|---|
| enwik8 L5 (95 MB) | 16.6 ms | 0.77 ms | 0.25 ms (was 0.59) | 3.34 ms | **1.03×** (was 1.15× pre-fix) |
| silesia L5 (203 MB) | 32.7 ms | TBD | TBD | TBD | **1.03×** (was 1.17× pre-fix) |

Decoder reference: CUDA enwik8 L1 16.2 ms / silesia L1 31.1 ms / web L1 2.0 ms / enwik8 L5 16.2 ms / silesia L5 31.7 ms.
Encoder reference: CUDA enwik8 125 ms / silesia 241 ms / web 11 ms.
Huffman decode kernel parity: VK 0.77 ms ≈ CUDA 0.71 ms (1.08×).

---

## L2-L5 roadmap to full CUDA parity

The CUDA encoder supports five levels:

| Level | Parser | Hash bits | Huffman | LZ rehash | Status |
|---|---|---|---|---|---|
| **L1** | greedy | 17 | NO | NO | ✅ **DONE** — byte-identical to CUDA + FASTER on large workloads |
| **L2** | greedy | 18 | NO | NO | ✅ **DONE** (`7bbcf77`) — byte-identical to CUDA + FASTER; 9 ptest cases added |
| **L3 encode** | greedy | 19* | YES | NO | ✅ **ENCODER SHIPPED** (`f9e84e3` + `30f36d3`) — byte-identical to CUDA on web/enwik8; silesia 0.02% larger (4 GiB SSBO cap). Decoder blocked on Phase 3. |
| **L4 encode** | greedy | 17 | YES | YES (`p_l4=1`) | ✅ **ENCODER SHIPPED** (same commits — `p_l4=1` flag already plumbed in encode_lz.zig) |
| **L3 decode** | n/a | n/a | YES | n/a | ✅ **DONE** (`4270ea4` iter 4f) — VK→VK roundtrip MATCH on web/enwik8/silesia |
| **L4 decode** | n/a | n/a | YES | n/a | ✅ **DONE** (same commit) — VK→VK roundtrip MATCH on web/enwik8/silesia |
| **L5 encode** | **chain parser** | 17 | YES | YES | ✅ **DONE** (Phase 2B, uncommitted) — byte-identical to CUDA SHA on web (`AE11A3CF...`) + enwik8 (`030BD807...`) + silesia (`CF6A2BCC...`); kernel 0.93-0.98× CUDA |
| **L5 decode** | n/a | n/a | YES | n/a | ✅ **DONE** (same Huffman + LZ decode pipeline as L3/L4) — VK→VK L5 roundtrip MATCH on all 3 corpora |

*L3 hash_bits clamps from 19 → 18 only when single-frame input ≥ 128 MiB at sc=0.25 (Vulkan 4 GiB maxStorageBufferRange cap). Affects only silesia at L3 today; cost is 0.02% larger output. See "Accepted residuals" below.

**Important correction from L1 work**: The original `/src_vulkan/TODO.md` claimed L2 needed Huffman. **It didn't.** Verified via `src/encode/fast_framed.zig:247-251`: Huffman gates at `opts.level >= 3`. L2 is greedy parser + bigger hash table only. The VK encoder already passes `hash_bits = levels.hashBitsForLevel(level)` through to the kernel (`srcVK/encode/encode_lz.zig:66`), so L2 worked with zero code changes once L1 was byte-identical. The encode iter-3 + iter-4 fixes that brought L1 to byte-identity automatically applied to L2.

### Phase 2-prep — VERIFY L2 WORKS — **DONE** (`7bbcf77`)

L2 SHIPPED with zero new code. Quick verification + ptest cases added:

| File | VK L2 SHA | CUDA L2 SHA | VK encode ms | CUDA encode ms | Ratio |
|---|---|---|---|---|---|
| web.txt | match | match | 24 | 12 | 2.0× (WDDM floor) |
| enwik8 | match | match | **112** | 130 | **0.86× — FASTER** |
| silesia | match | match | **223** | 249 | **0.90× — FASTER** |

9 new ptest cases (5 in-process + 4 cross-backend SHA gate). ptest_vk 74/0/0 → **83/0/0** both backends.

### Phase 2A — Huffman codec (unlocks L3 + L4)

Goal: encoder produces chunk type 4 (Huffman-coded) sub-chunks that the CUDA
decoder reads; decoder reads CUDA-produced chunk type 4. Cross-backend
roundtrip parity across L3+L4 in both directions.

**Status: ENCODER SHIPPED.** Decoder blocked on Phase 3.

✅ **Phase 2A encoder (`f9e84e3` + `30f36d3`):**
- `srcVK/encode/huff_build_tables_kernel.comp` — full faithful port, line-for-line equivalent including Kraft-budget height-limit redistribution
- `srcVK/encode/huff_encode_4stream_kernel.comp` — full faithful port including BIL interleaved + tail areas
- `srcVK/encode/encode_huff.zig` — three distinct Impl entry points (literals/tokens/off16), not collapsed
- Wired into `srcVK/encode/fast_framed.zig:417-422` at `opts.level >= 3`
- Adversarial review verdict: FAITHFUL_PORT_NO_CONCERNS, ship_as_is
- 0 stubbed functions, 0 TODOs, 0 simplified algorithms
- Empirical proof: VK L3 encode SHA byte-identical to CUDA L3 encode on enwik8 + web.txt; silesia 0.02% larger (4 GiB SSBO cap, see residuals)
- Encode kernel perf parity with CUDA (80.6 ms VK vs 80.8 ms CUDA on enwik8 L3)
- ptest_vk 83/0/0 preserved
- L4 = L3 + `p_l4=1` flag already plumbed in encode_lz.zig

✅ **Phase 2A decoder iter 1 (`5b438bc`):** `gpuScanChunks` host driver port (~190 LOC Zig) — drives Phase 3 scan/compact kernels end-to-end. Adversarial review FAITHFUL.

✅ **Phase 2A decoder iter 2 (`6a1da40`):** `srcVK/decode/huff_build_lut_kernel.comp` — full 8-pass faithful port (~391 LOC GLSL). All 8 passes present: weights unpack, parallel histogram, canonical bucket starts, parallel LUT zero, fused code-assignment + Pass-1 fan-out (with `__match_any_sync` emulation), Pass-2 dual-symbol, Pass-3 escape entries, cache-streaming bulk dump. Adversarial review verdict: NO simplifications.

✅ **Phase 2A decoder iter 3 + fix (`d5860d1` + `ac6696f`):** `srcVK/decode/huff_decode_4stream_kernel.comp` — full faithful port (~616 LOC GLSL). All 4 phases of `decodeStreamBoundedIL` present (preamble byte-drain + INTERLEAVED X2 hot loop + TAIL X2 hot loop + single-symbol tail finisher). u64 bit-buffer split to uvec2 pair with proven equivalence (top-10-bits-live-in-hi invariant since MAX_CODE_LEN=10 < REFILL_BITS=32). `__byte_perm` emulated via manual shift+mask `bswapU32`. `__shfl_up_sync` → `subgroupShuffleUp`. Plus `runHuffBuildAndDecode` host wiring (~75 LOC). Adversarial review caught a CRITICAL bug in `bitbufRefill` (hi/lo SWAPPED — would have produced garbage decode); fixed at `ac6696f`.

✅ **Conformance test (`de9d7eb`):** `srcVK/tests/huff_decode_conformance.zig` — 5 isolated kernel-level tests (1 KiB random, 8 KiB English, 64 KiB random, 16 KiB skewed alphabet, 4097 B odd-length). Drives encoder + decoder Huffman chain in isolation (bypasses merge/scan/compact). Empirically proven to catch the `bitbufRefill` hi/lo swap bug class (re-introduced bug → all 5 new tests failed while every pre-existing test passed). ptest_vk 83 → **88/0/0** both backends.

✅ **Phase 2A decoder iter 4 (`ac013b5`):** lz_decode general workhorse port — 855 LOC of CUDA reference faithfully ported across 4 GLSL files (lz_decode_kernel.comp + lz_decode_general.glsl + lz_header_parse.glsl + lz_dispatch.glsl + slz_wire_format.glsl). All 5 token types (SHORT/LONG_LITERAL/LONG_NEAR/SHORT_FAR/LONG_FAR), MAX_BLOCKS_PER_SUBCHUNK=2 transition, CHUNK_FLAG_UNCOMPRESSED + CHUNK_FLAG_MEMSET fast paths, recent_offset tracking, per-block + final trailing literals, mode-0 delta literal copy all present. New `compute_to_compute_barrier` proc added (CUDA's per-launch implicit ordering doesn't carry to VK). Adversarial review: structurally faithful but L3+ empirically broken at byte 65544 (deferred to iter 4b).

✅ **Phase 2A decoder iter 4b (`bcaa1f1`):** `.level` field defaulted to 1 in `dispatchCompressedBlock` → L2 gate never fired → raw kernel ran on Huffman-coded streams → silent wrong output. Fix: thread `hdr.level` from frame header through `DecodeRequest.level`. State change: silent corruption → loud `error.KernelLaunchFailed` (exposed iter 4c blocker).

✅ **Phase 2A decoder iter 4c (`11eb101`):** per-binding offset support in `procLaunchKernel` ABI + 5+ dispatch site fixes (scan_parse / 4× compact_huff / compact_raw / mergeHuffDescs / gatherRawOff16 / runHuffBuildAndDecode). 14 launch_kernel call sites audited; backward-compat preserved via `binding_offsets: ?[*]const u64` null default. `minStorageBufferOffsetAlignment` captured + enforced. `d_compact_counts` layout widened from packed 6×u32 → strided 6×256 B slots. 401 LOC across 11 files. State change after iter 4c: small inputs → still `KernelLaunchFailed`; large inputs → silent corruption at byte 65544.

✅ **Phase 2A decoder iter 4d (`b3fab31`):** 20 L3/L4 ptest cases added (10 synthetic-PASS + 10 real-corpus-FAIL). The 10 corpus tests serve as deterministic reproducer + diagnostic gate. ptest now reports honest 98/0/10 instead of false-positive 88/0/0. All 10 failures converge on `first_diff=65544` with rich diagnostic output (total_diffs, trailing_zeros, hex windows).

✅ **Phase 2A decoder iter 4f (`4270ea4`):** **RESOLVED the byte-65544 bug.** Replaced `lz_dispatch.glsl`'s `lit_in_scratch || cmd_in_scratch` fork-to-general guard with explicit 8-arm dispatch enumeration over `(off16_split, lit_in_scratch, cmd_in_scratch)`. Each arm dispatches `decodeSubChunkRawMode_{true,false}` with the right SSBO names — matching CUDA's polymorphic pointer semantics via GLSL macro-textual specialization. ptest 98/0/10 → **108/0/0**. L3+L4 VK→VK roundtrip MATCH on web + enwik8 + silesia. Per PortAdaptations.md A-001 (now RESOLVED). The fix was masked for 5+ hours by stale SPIR-V (A-012 build-graph gap); recovery via `rm zig-out/srcvk_shaders/lz_decode_kernel.spv` + rebuild. `tools/build_vk.bat` (`db4406c`) added as workaround.

✅ **Phase 2A-decoder is COMPLETE.** All 4 iters (4 + 4b + 4c + 4d) + 4f shipped. L3 + L4 work end-to-end. ptest_vk: 108/0/0 both backends.

~~⏸️ **OPEN BUG — sub-chunk-N>0 decoder loop in `lz_decode_kernel.comp`:**~~ ✅ RESOLVED at `4270ea4`. The "byte-65544" symptom was the A-001 dispatch fork. The previous diagnosis of "sub-chunk-N>0 outer loop" was correct in narrowing but wrong in localization — fix was in the dispatcher, not the decoder body.
The byte-65544 failures are at the SUB-CHUNK 0 → SUB-CHUNK N (N>0) boundary in the OUTER dispatch loop, NOT at the block-1 → block-2 transition (which doesn't run for L3 web.txt; sc_group_size=0.25 → eff_chunk=64KB → one chunk = one sub-chunk = one LZ block).

In-shader diagnostic at END of sub-chunk 1's inner-while loop:
  - `dst_pos = 131072` (decoder thinks it wrote all 65 KB of sub-chunk 1)
  - `lit_pos = lit_size`, `cmd_pos = cmd_size` (all consumption complete)
  - But output bytes 65544..131071 are zeros (CUDA→VK) or garbage (VK→VK)

Three live hypotheses (iter 4d root-cause-analyst agent — 180 min didn't isolate):
  1. Cooperative literal/match copy loops only execute for lane 0 in sub-chunks N>0 (warp divergence)
  2. Memory barrier semantics between sub-chunk iterations (missing subgroupMemoryBarrierBuffer)
  3. Huffman-decoded scratch is all-zeros for sub-chunks 1+ (decoder dutifully writes zeros — would mean Huffman or scan/merge bug, not LZ)

**Suggested first investigation step:** dump entropy_scratch contents at sub-chunk 1 offset. If all zeros → hypothesis 3 (upstream bug); if non-zero → hypothesis 1 or 2 (LZ decoder bug).

These 10 failing tests will turn GREEN when fixed; ptest will report 108/0/0.

**Also documented (smaller, related):** "small-input KernelLaunchFailed" on 70KB web L3 + 1MB enwik8 L3 CUDA→VK (per iter 4c adversarial review). Likely a procLaunchKernel runtime guard or specific alignment issue; not currently caught by the 10 corpus tests; investigate alongside the byte-65544 fix.

### Phase 2B — L5 chain parser — ✅ **DONE** (uncommitted, 2026-06-08)

Goal (met): L5 .slz files byte-identical to CUDA L5 on web + enwik8 + silesia.

✅ **Port:** Filled in `srcVK/encode/lz_chain_parser.glsl` (90 → 575 LOC, +485). All 6 stubs ported faithfully from `src/encode/lz_chain_parser.cuh`:
- `isLazyMatchBetter`, `isMatchBetter`, `isBetterThanRecentMatch` helpers (the `.length` field shadowed GLSL's reserved method — A-014 catalogs the decomposed-int signature workaround)
- `findMatchChain`, `insertChainRange`, `scanBlockChain` macros (A-013 catalogs the u16-packed-as-u32 SSBO adaptation: NEXT_HASH_SIZE u16 entries → NEXT_HASH_SIZE/2 u32 words via masked read/write, safe because chain parser is lane-0-only serial)
- Host plumbing (`encode_lz.zig:87-91`) was already split for chain-mode footprint (`hash_size + hash_size + NEXT_HASH_SIZE/2` u32 words per chunk); no Zig changes needed.
- Kernel dispatch (`lz_encode_kernel.comp:151-208`) was already wired to route `pc.use_chain==1` through `scanBlockChain` with the 3-table SSBO layout; no kernel changes needed.

✅ **Verification (NVIDIA RTX 4060 Ti, `SLZ_VK_DEVICE_INDEX=1`):**
- SHA byte-identical to CUDA L5 on all 3 corpora (web/enwik8/silesia — MATCH on all)
- ptest_vk: 108/0/0 preserved on NVIDIA + Intel iGPU
- 0 VUIDs / 0 sync hazards / 0 best-practices errors on L5 validation run
- Kernel perf vs CUDA L5: web 0.98×, enwik8 0.93×, silesia 0.95× (on-par-to-faster — both backends serialize chain on lane 0)

✅ **Adversarial review (post-port):** 26 specific findings reviewed against the CUDA reference (chain walk, u16-packing, lazy-1/lazy-2 gates, recent-match XOR partial-length, far-class margins, insert ordering, trailing-literals, block 1→2 propagation, macro shadowing/re-eval); verdict **PORT_FAITHFUL**, recommendation **ship_as_is**.

Future hardening (NOT a blocker): add L5-specific ptest cases that exercise long-match chain truncation, mixed-class offset compares, large-literal-run threshold transitions, block 1 → block 2 boundary crossings with recent_offset carryover.

### Phase 3 — Multi-kernel decode pipeline (L2+ requires this)

L1 host-input decode currently uses CPU `buildChunkDescriptors` (correct
per port discipline — CUDA does this on CPU for the host-input L1 path
too). TWO things need the multi-kernel GPU graph:
  (a) **L2+ host-input decode** — the entropy-decoded sub-chunks need
      GPU scan/walk/compact to feed Huffman LUT + decode kernels
  (b) **D2D entry point at EVERY level** (`decompressFramedFromDevice`)
      — when input is device-resident, CPU can't walk it; the chain
      walk runs on GPU via walk_frame_kernel. Used by the v3 C ABI.
      Applies to L1 D2D as well as L2+ D2D, per CUDA.

**Kernels to port** (CUDA: `src/decode/scan_parse_kernel.cuh`,
`prefix_sum_chunks_kernel.cuh`, etc.):
- `srcVK/decode/scan_parse_kernel.comp` — parse compressed-block headers on GPU
- `srcVK/decode/prefix_sum_chunks_kernel.comp` — running offsets (L1 has this for prefix_sum_chunks_fn already — port for the D2D path)
- `srcVK/decode/compact_huff_descs_kernel.comp` — gather entropy-decoded chunk descs
- `srcVK/decode/compact_raw_descs_kernel.comp` — gather raw chunk descs
- `srcVK/decode/walk_frame_kernel.comp` — walk parsed frame (D2D entry — applies at ALL levels per CUDA, not L2-only)
- `srcVK/decode/merge_huff_descs_kernel.comp` — merge huff descs
- `srcVK/decode/gather_raw_off16_kernel.comp` — gather raw off16 stream
- `srcVK/decode/lz_decode_kernel.comp` — L2 huff-aware LZ workhorse (L1 uses `lz_decode_raw_kernel.comp`)

**Host glue:** generalize `decode_dispatch.zig::fullGpuLaunchImpl` to use the
GPU chain when L2+ OR when the caller supplies device-resident input via
the D2D entry point `decompressFramedFromDevice`. The D2D entry point
applies at EVERY level (L1 through L5) — used by the v3 C ABI for
device-input, device-output decode where the frame and output never
leave VRAM. The chunk walk runs on GPU via the new walk_frame_kernel.

Current state: `decompressFramedFromDevice` returns
`error.NotImplementedL2` (the sentinel name is misleading — it's "D2D
entry not implemented yet" rather than "L2-specific"). Phase 3 unblocks
L1 D2D as the first milestone; Phase 2A-decoder + Phase 3 together
unblock L2+ D2D.

LOC estimate: ~1,500 GLSL + ~600 Zig. ~3-4 single agents.

### Phase 4 — True D2D + Async API

L1 has stubs for these but not actual implementations:
- `decompressFramedFromDevice` — pure-D2D decode entry point at EVERY
  level (used by v3 C ABI). Currently returns `error.NotImplementedL2`
  (misnamed sentinel — actual meaning is "D2D entry not implemented yet").
  L1 D2D unblocked by Phase 3 (walk_frame_kernel + host glue); L2+ D2D
  also needs Phase 2A-decoder.
- Async APIs (`slzCompressAsync_vk`, `slzCompressAsyncPoll_vk`, etc.) — not wired

**The decode/encode procs surface already supports streams** (iter-11 split
queue, iter-13 early submit, iter-15 single-submit/single-wait, encode iter-3
transfer-queue gather, encode iter-4 imported D2H), so async is mostly
host-glue work.

Items:
- True buffer-device-address D2D for `slzCompress_vk` / `slzDecompress_vk`
- Async encode/decode worker pool wrapping the sync codec
- `slzGetLastTimings_vk` to drain VkQueryPool data

LOC estimate: ~600 Zig. ~1-2 single agents.

### Phase 5 — Conformance + perf parity validation

- Extend `tests/cross_backend_roundtrip.zig` for L1-L5 in both directions on both backends
- Benchmark vs CUDA on each level; target ≤1.10× on large workloads
- Document any residual gaps as accepted tradeoffs

LOC estimate: ~400 Zig. ~1 single agent.

### Total estimate to full CUDA parity

| Phase | GLSL LOC | Zig LOC | Agent runs |
|---|---|---|---|
| ~~2-prep (verify L2)~~ | ✅ DONE | ✅ DONE | ✅ |
| ~~2A encoder (Huffman, L3 + L4)~~ | ✅ DONE (~700 GLSL + 500 Zig) | | ✅ 1 + 1 review + 1 fix |
| ~~2A decoder iter 1-3 (gpuScanChunks + huff_build_lut + huff_decode_4stream + conformance test)~~ | ✅ DONE (~1,000 GLSL + 400 Zig) | | ✅ 3 + 3 reviews + 1 fix + 1 test |
| ~~2A decoder iter 4 + 4b + 4c + 4d + 4f (lz_decode general + .level fix + offset ABI + 20 L3/L4 tests + dispatch fork resolved)~~ | ✅ DONE (~700 GLSL + 1000 Zig + 868 LOC tests + 127 LOC fix) | | ✅ 5 commits + 1 catalog |
| ~~3 (GPU decode pipeline)~~ | ✅ DONE | | ✅ 1 + 1 review + 1 fix |
| ~~2B (L5 chain parser + fix L5 broken branch)~~ | ✅ DONE (+485 GLSL, 0 Zig) | | ✅ 1 port + 1 adversarial review |
| 4 (D2D + async — note: L1 D2D already opened by Phase 3) | 0 | ~600 | 1-2 |
| 5 (conformance + perf) | 0 | ~400 | 1 |
| **REMAINING TOTAL** | **0** | **~1,000** | **2-3** |

---

## Kernel inventory (current state)

### L1 kernels — REAL (working in production, byte-identical to CUDA)

| File | Status | Purpose |
|---|---|---|
| `decode/lz_decode_raw_kernel.comp` | ✅ real | L1 raw-mode LZ decode workhorse |
| `decode/prefix_sum_chunks_kernel.comp` | ✅ real | L1 prefix sum over chunk descs |
| `encode/lz_encode_kernel.comp` | ✅ real (~300 LOC) | L1 greedy LZ encode — NCU-verified parity with CUDA PTX |
| `encode/assemble_measure_kernel.comp` | ✅ real | Per-chunk sub-chunk sizing |
| `encode/assemble_write_kernel.comp` | ✅ real | Per-chunk payload write |
| `encode/frame_assemble_kernel.comp` | ✅ real | Frame splice + headers |

### L2+ kernels — STUBS (waiting for Phase 2A/3)

| File | Phase | CUDA reference | Approx CUDA LOC |
|---|---|---|---|
| `encode/huff_build_tables_kernel.comp` | 2A | `src/encode/huffman_kernel.cu` | ~350 |
| `encode/huff_encode_4stream_kernel.comp` | 2A | `src/encode/huffman_kernel.cu` | ~350 |
| `decode/huff_build_lut_kernel.comp` | 2A | `src/decode/huffman_kernel.cu` | ~350 |
| `decode/huff_decode_4stream_kernel.comp` | 2A | `src/decode/huffman_kernel.cu` | ~350 |
| `decode/scan_parse_kernel.comp` | 3 | `src/decode/scan_parse_kernel.cuh` | 234 |
| `decode/walk_frame_kernel.comp` | 3 | `src/decode/walk_frame_kernel.cuh` | 200 |
| `decode/compact_huff_descs_kernel.comp` | 3 | `src/decode/compact_descs_kernels.cuh` | ~45 |
| `decode/compact_raw_descs_kernel.comp` | 3 | `src/decode/compact_descs_kernels.cuh` | ~45 |
| `decode/merge_huff_descs_kernel.comp` | 3 | `src/decode/merge_huff_descs_kernel.cuh` | 62 |
| `decode/gather_raw_off16_kernel.comp` | 3 | `src/decode/gather_raw_off16_kernel.cuh` | 38 |
| `decode/lz_decode_kernel.comp` | 3 | `src/decode/lz_decode_kernels.cuh` (huff path) | ~400 |

**Plus needed:** `encode/lz_chain_encode_kernel.comp` (no stub yet) for Phase 2B
— CUDA reference `src/encode/lz_chain_parser.cuh` ~440 LOC.

---

## Architecture invariants — DO NOT VIOLATE

These kept us out of trouble during L1 and remain load-bearing for L2-L5.

### procs.* surface (decode/vulkan_api.zig)

The codec calls `procs.h2d(dst, src, size)`. CUDA and VK both. Same name,
args, semantics. Vulkan plumbing (vkCmdCopyBuffer, VMA, fences, queues)
lives UNDER procs. Adding new slots is OK when the slot expresses a Vulkan-
native concept CUDA doesn't have a direct analog for.

Current slots (26): `malloc_device, free_device, h2d, d2h, d2h_offset,
h2d_async, d2h_async, d2d, memset_d8, memset_d8_async, malloc_host,
free_host, stream_sync, ctx_sync, ctx_set_current, ctx_get_current,
stream_create, stream_destroy, launch_kernel, event_create, event_record,
event_synchronize, event_elapsed_time, event_destroy, stream_flush_transfer,
stream_compute_barrier, d2h_offset_gather`.

The `d2h_offset_gather` slot (iter-3 + iter-4) is the highest-leverage VK
adaptation — eliminates per-call submit floor for both per-chunk D2Hs AND
the final ~50 MB output D2H.

### Function decomposition (CUDA-verbatim)

Do NOT rename or restructure:
- `decode/decode_dispatch.zig::fullGpuLaunchImpl` + `uploadInputAndPrefixSum` + `runBackHalf` + `finalizeOutput` + `runLzPipeline`
- `decode/decode_dispatch.zig::runHuffBuildAndDecode` / `mergeHuffDescs` / `gatherRawOff16` (L2 stubs to be filled in Phase 3)
- `encode/fast_framed.zig::compressFramedOne` + `encode/encode_lz.zig::gpuCompressImpl` + `encode/encode_assemble.zig::gpuAssembleFrameImpl` / `gpuFrameAssembleImpl`

### L2 gate at `decode_dispatch.zig:724`

`if (req.level >= 2) { ... }` wraps ensureDeviceBuf for L2 buffers + L2 kernel
dispatches. Skipped entirely on L1. CUDA wastefully dispatches 7 L2 kernels
on L1 (~388 µs/decode); we don't. **Phase 3 fills the L2 branch.**

### `buildChunkDescriptors` MUST stay CPU walk on L1 host-input path

In `decode/streamlz_decoder.zig`. The prior `/src_vulkan/` session moved this
to GPU for the host-input path — **THAT WAS THE CANONICAL PORT VIOLATION.**
Keep it CPU for the L1 host-input decode path.

**Phase 3's GPU `walk_frame_kernel` is for a SEPARATE entry point**:
the D2D path (`decompressFramedFromDevice`), where the input is
device-resident and CPU walk isn't possible. CUDA has the same split
— two entry points, each uses its appropriate walk. The L1 host-input
path keeps the CPU walk; the D2D path uses the GPU walk. Don't merge
the two.

### KERNEL_DECLS (decode + encode module_loader.zig)

Pipeline binding counts + push constant sizes per kernel. Do NOT change
without auditing all call sites for matching `params[]` layout. Adding L2
kernels means adding KERNEL_DECLS entries with correct
`pin_subgroup_32: bool` (true unless `local_size_x == 1`).

### Subgroup size pinned to 32 (gpu_warp.glsl contract)

Both decode + encode pipeline creation pin `requiredSubgroupSize=32` +
`REQUIRE_FULL_SUBGROUPS_BIT`. Device-pick guard rejects devices that can't
satisfy `[minSubgroupSize<=32, maxSubgroupSize>=32]`. Kernels in
`gpu_warp.glsl` hardcode `WARP_SIZE=32u`. **NVIDIA only offers 32; Intel
supports the pin via UHD/Iris/Arc; AMD also supports via wave32 path.**

### Persistent buffers on EncodeContext

Encoder relies on three persistent host buffers, all page-aligned via
`gpu_dec_driver.allocHost` → `procMallocHost` (Vulkan VMA host alloc), so
the iter-8 LRU import cache hits on every call after the first:

- `gpu_out_buf` (iter-3): destination of the per-chunk D2H gather. ~50 MB
  on enwik8.
- `d2h_final_buf` (iter-4): destination of the final-frame D2H. ~50 MB on
  enwik8.

These eliminate the staging memcpy at the host. Without them the iter-7/8/12
import path can't fire (refuses non-page-aligned host buffers).

### VkPhysicalDeviceVulkan12Features + Vulkan13Features required

`bufferDeviceAddress`, `shaderInt8`, `storageBuffer8BitAccess`,
`uniformAndStorageBuffer8BitAccess`, `subgroupSizeControl`,
`computeFullSubgroups`. Enabled at VkDevice creation. **If any of these
isn't supported, init returns `error.BackendNotAvailable` so the codec falls
back to CPU.**

### Persistent disk-backed VkPipelineCache (commit 02560d6)

All 17 compute pipelines (11 decode + 6 encode) share a process-wide
VkPipelineCache that loads from `%LOCALAPPDATA%/streamlz_vk/pipeline_cache.bin`
at init and saves at process exit (atexit-registered). Driver validates
header per Vulkan 1.0 spec § 10.5.4 — silently discards mismatches.

Cold-start savings on NVIDIA RTX 4060 Ti: ~38 ms (5.4%). Warm bench unchanged
(bench pre-warms within a single process). Adds correctness for one-shot
CLI tools that repeatedly invoke encode/decode.

---

## Lessons learned (apply to L2-L5 work)

### Always run validation layer before claiming "done"

Standard layer catches spec violations. Synchronization layer catches missing
barriers and race conditions. Best-practices layer catches anti-patterns.
**Three real latent bugs were caught only by validation:**

- VUID-vkBindBufferMemory-memory-02985 (iter-12 import path missing buffer-create chain — `9022dde`)
- VUID-vkCmdCopyBuffer/EndCommandBuffer-recording (cmdbuf invalidated by destroyed buffer — `7960b98`)
- VUID-VkShaderModuleCreateInfo-pCode-08737 (glslc optimizer bug — `23d8ab3`)

```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH = "c:/tmp/vk_layer_settings.txt"  # contents below
./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/v.slz assets/web.txt 2>&1 | Select-String "VUID|Validation Error"
```

`vk_layer_settings.txt`:
```
khronos_validation.enables = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
khronos_validation.report_flags = error,warn,perf,info
```

### SHA byte-identical to CUDA is MANDATORY — roundtrip-passes is INSUFFICIENT

The H2 disaster taught us this the hard way. Encode iter-1 H2 broke
compression on inputs ≥ 8 MiB — produced essentially uncompressed output
that VALID-roundtripped through VK→VK. It passed all roundtrip + cross-
backend tests because all-raw output is structurally valid SLZ. **Was caught
only by comparing output SIZE to CUDA.**

**Every encode iteration MUST verify SHA256(VK_output) == SHA256(CUDA_output)**
on web/enwik8/silesia. We had to bisect + revert 2 commits (H2 + iter-2)
when this was missed. The cost of one extra SHA check >>> the cost of
reverting a perf iteration.

The same applies to any L2-L5 work touching encode.

### NEVER hand-type Vulkan constants

Commit `6ca302c` was a 4-line constant correction that fixed an entire
broken-pin commit. Two enum values had been typed from memory and were both
wrong. **Always pull constants from `C:/VulkanSDK/1.4.341.1/Include/vulkan/vulkan_core.h`** —
grep the header, copy the exact value, paste.

### Per-phase QPC profiler is the localization tool

When perf is off and you don't know why, **add QPC checkpoints around every
phase** before guessing. The encode iter-4 win (3.2× speedup) was directly
enabled by the encode phase profiler (`d83ea21`) which localized 71% of the
encode wall to ONE call (`enc.d2h_final` at 238 ms).

Pattern: `g_phase_<name>_ns: i64 = 0` accumulators + QPC checkpoints
inline + `printAndResetPhaseProfile` gated on `SLZ_VK_PROFILE_PHASES=1`.
Decode side mirrors it (`g_phase_*_ns` in `decode/decode_dispatch.zig`).

### Zig std subprocess spawn defaults to EMPTY env

`std.Io.Threaded.InitOptions.environ` defaults to `.empty`. Spawned children
get NO PATH, NO SystemRoot, NO PROGRAMDATA. Vulkan loader can't find ICDs
without them. **Pass `.{ .environ = .{ .block = .global } }` when spawning
subprocesses that need registry-discovered hardware drivers.** Closed at
commit `e8fc91f`.

### glslc -O optimizer bug — Int8 capability not auto-declared

When the optimizer constant-folds `uint8_t(SLZ1_FRAME_MAGIC & 0xFFu)` it
emits `OpConstant %uchar` but fails to declare `OpCapability Int8`. Driver
validation rejects the module. **Patch SPIR-V binaries post-compile in
`encode/module_loader.zig::loadShaderModule`** — splice `OpCapability Int8`
into the capability block when any 8-bit storage cap is present without it.
Workaround documented at commit `23d8ab3`.

### Vulkan-on-WDDM has a per-submit floor that CUDA doesn't

Each `vkQueueSubmit + vkWaitForFences` round-trip is ~50-150 µs on Windows
WDDM. CUDA's synchronous DtoH calls batch submission cost across the
kernel-mode driver, hitting ~37 µs. **You cannot reduce per-call submit cost
in Vulkan; you can only reduce the number of submits.** Pattern: batch N
copies into one `vkCmdCopyBuffer` with N regions, or apply iter-7+11+15
(transfer queue + imported dst + single submit).

This is also why web.txt small-file workloads stay ~1.7-2.6× CUDA on both
decode + encode — the per-decode/per-encode overhead floor doesn't shrink
with input size. **Documented as accepted structural limit.** NCU + nsys +
deep-research-agent investigations all confirmed this is not fixable from
the app side.

### The iter-7+11+15 transferable pattern is GENUINELY transferable

This is the canonical fix for any per-call host-bounce pattern:
- **iter-7 D2H import:** `VK_EXT_external_memory_host` wraps the user's
  destination buffer as VkDeviceMemory; kernel writes directly into host RAM
- **iter-11 transfer queue:** dedicated VK_QUEUE_TRANSFER_BIT queue (DMA
  engine) + binary semaphore split-submit
- **iter-15 single submit:** consolidate phases into one submit + one wait
  per call by recording into the same cmdbuf

Applied to encode three times:
- **encode iter-3** (`7d37231`) — per-chunk D2H gather: 1526 submits → 1, **2× speedup**
- **encode iter-4** (`d9d8bba`) — final-frame D2H: 238 ms → 8 ms, **3.2× speedup**

Both REQUIRED the destination buffer to be page-aligned (procMallocHost). When
the host buffer comes from the caller (CLI), wrap it in a persistent
EncodeContext-owned pinned buffer + memcpy at the end (~10 ms cost vs ~200 ms
savings).

**The decoder uses the same pattern via iter-7/12 import for output/input
buffers.** Any L2+ kernel that emits a per-chunk host bounce should use this
pattern out of the gate.

### NCU is CUDA-only — use Nsight Graphics for Vulkan kernel profiling

Nsight Compute (NCU) 2025.4 has no Vulkan compute shader support. For
per-kernel SPIR-V metrics (occupancy, warp stalls, memory throughput),
use **Nsight Graphics → GPU Trace Profiler**. NCU is still useful for
profiling the CUDA reference side.

For pure GPU kernel timing on Vulkan, `VkQueryPool` timestamps inside our
own bench give per-kernel durations apples-to-apples with CUDA's bench.

### ptest_vk parallelism — three race classes caught

1. Decode/encode init races (lock with `g_init_lock` / `g_encode_init_lock`)
2. Encode init latching (use `defer if (ok)` for the `initialized=true` flip)
3. Cross-queue submit races on shared compute queue (lock via `g_dispatcher_lock`)

Plus iter-14's per-(StreamEntry × pipeline) descriptor sets to avoid races
on shared descriptor state.

### The H2 bisect lesson

Encode iter-1 H2 broke compression silently. We caught it only by careful
manual comparison of VK output size vs CUDA output size after a user
pushback ("are you sure L2 uses Huffman?"). The bisect via `git worktree`
isolated the regression cleanly. **Three takeaways:**

1. Bisect with worktrees, not destructive history rewrites
2. Don't trust a "passes all tests" claim if the tests don't check the
   property you care about (compression ratio vs file roundtrip)
3. The CLAUDE.md `NEVER REVERT WITHOUT PERMISSION` rule is right — we
   preserved the H2 + iter-2 commits in history, did clean reverts (not
   `git reset`), and could re-attempt the work on cleaner foundations

---

## File map

### Core paths

- `decode/streamlz_decoder.zig` — `decompressFramed` + CPU `buildChunkDescriptors`
- `decode/decode_dispatch.zig` — `fullGpuLaunchImpl` orchestrator + L2 gate at line 724 + per-phase QPC
- `decode/decode_context.zig` — `DecodeContext`, `ensureDeviceBuf` (grow-only), profiling
- `decode/scan_gpu.zig` — `gpuPrefixSumChunksImpl` (L1) + L2 stubs
- `decode/module_loader.zig` — VkDevice init, transfer queue, KERNEL_DECLS, procs impls, LRU import cache, streamEndAndWait, persistent_desc_sets, **persistent VkPipelineCache** (iter-cache), **procD2HOffsetGather** (iter-3)
- `decode/vulkan_api.zig` — procs surface (26 slots)
- `decode/driver.zig` — facade
- `encode/fast_framed.zig` — `compressFramedOne` + persistent gpu_out_buf + persistent d2h_final_buf + **per-phase encode QPC accumulators** (profiler)
- `encode/encode_lz.zig` — `gpuCompressImpl` + per-chunk D2H gather call (iter-3)
- `encode/encode_assemble.zig` — frame assembly passes
- `encode/encode_context.zig` — `EncodeContext` + `gpu_out_buf` (iter-3) + `d2h_final_buf` (iter-4)
- `encode/module_loader.zig` — encode VkDevice init + shader module loader + Int8 SPIR-V patcher + uses shared VkPipelineCache via `pipelineCacheHandle()`
- `encode/vulkan_ffi.zig` — encode-side procs

### CLI

- `cli/bench_decompress.zig` — `-db` (post-`d5adeb8` pure-kernel metric)
- `cli/bench_compress.zig` — `-b` (encode + roundtrip) + encode phase profiler wiring
- `cli/decompress.zig` — `-d`
- `cli/compress.zig` — `-c`
- `cli.zig` — dispatcher, prints `Device: <name>` at startup
- `cli/util.zig` — Args + DeviceSelector (`SLZ_VK_DEVICE_INDEX` env)

### Tests

- `tests/decoder_unit.zig` (22 host-only)
- `tests/encoder_unit.zig` (11 host-only)
- `tests/dispatch_unit.zig` (7 including KERNEL_DECLS ABI check)
- `tests/kernel_conformance.zig` (per-test context)
- `tests/l1_decode_roundtrip.zig` (per-test context)
- `tests/l1_encode_roundtrip.zig` (per-test context)
- `tests/cross_backend_roundtrip.zig` (subprocess — needs env-inherit `makeIo`)
- `tests/cli_smoke.zig` (subprocess — needs env-inherit `makeIo`)
- `test_runner_parallel.zig` (3-phase: serial_first → parallel → serial)

### Reference docs

- `audit.md` — canonical file mapping, port rules
- `PortInstructions.md` — per-file fleshout checklist
- `gameplan.md` — comprehensive port playbook
- This file (`ToDo.md`) — status + L2-L5 roadmap

---

## How to measure

### Bench (warm)

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"   # 1 = NVIDIA, 0 = Intel iGPU
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz   # decode bench
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt              # encode + roundtrip
```

Both bench paths pre-warm via an untimed first call, then measure. Reported
times EXCLUDE pipeline compile + Vulkan setup. "gpu kernel best/mean" is
sum of per-kernel VkQueryPool timestamps — apples-to-apples with CUDA.

### Per-phase host overhead (decode + encode both supported now)

```powershell
$env:SLZ_VK_PROFILE_PHASES = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz 2>&1 | Select-String "phase:"
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt 2>&1 | Select-String "phase:|Compress"
```

Decode phases in `decode/decode_dispatch.zig` (`g_phase_*_ns` accumulators).
Encode phases in `encode/fast_framed.zig` + `encode_lz.zig` + `encode_assemble.zig`
(`g_enc_phase_*_ns`).

### Cold-start measurement

```powershell
# Delete cache first
Remove-Item -Force $env:LOCALAPPDATA/streamlz_vk/pipeline_cache.bin -ErrorAction SilentlyContinue
Measure-Command { ./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/cold.slz assets/enwik8.txt 2>$null | Out-Null }
# Cache is now populated; second run is faster
Measure-Command { ./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/cold2.slz assets/enwik8.txt 2>$null | Out-Null }
```

### ptest_vk (after any srcVK change)

```powershell
Remove-Item -Force zig-out/bin/streamlz_vk.exe
zig build streamlz_vk -Doptimize=ReleaseFast
$env:SLZ_VK_DEVICE_INDEX = "1"
zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
$env:SLZ_VK_DEVICE_INDEX = "0"
zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
```

Expect 74/0/0 on both backends.

### SHA byte-identical to CUDA (MANDATORY for any encode change)

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"
foreach ($a in @("web.txt", "enwik8.txt", "silesia_all.tar")) {
  ./zig-out/bin/streamlz_vk.exe -c -l 1 -o "c:/tmp/vk_$a.slz" "assets/$a" 2>$null | Out-Null
  ./zig-out/bin/streamlz.exe -c -l 1 -o "c:/tmp/cu_$a.slz" "assets/$a" 2>$null | Out-Null
  $vk = (Get-FileHash -Algorithm SHA256 "c:/tmp/vk_$a.slz").Hash
  $cu = (Get-FileHash -Algorithm SHA256 "c:/tmp/cu_$a.slz").Hash
  Write-Host "$a MATCH=$($vk -eq $cu)"
}
```

Expected SHAs at HEAD `d9d8bba`:
- web: `AAA7F03B...`
- enwik8: `EE9471B7...`
- silesia: `D2970BF5...`

If CUDA binary is missing: rebuild via `tools\build_gpu.bat` (requires nvcc).

### Validation layer

See "Lessons learned" section above. Expect 0 VUIDs + 0 SYNC hazards.

### nsys VK + CUDA trace

```powershell
$NSYS = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
$env:VK_ADD_LAYER_PATH = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/vulkan-layers"
$env:VK_INSTANCE_LAYERS = "VK_LAYER_NV_nsight-sys"
$env:ENABLE_VK_LAYER_NV_nsight_sys = 1
& $NSYS profile --trace=vulkan --output=c:/tmp/trace --force-overwrite=true ./zig-out/bin/streamlz_vk.exe -db -r 3 tests/goldens/enwik8.txt.L1.slz
& $NSYS stats c:/tmp/trace.nsys-rep --report vulkan_api_sum
```

**Caveat:** nsys VK trace layer inflates absolute timings. Trust call counts +
relative shape; verify magnitudes via VkQueryPool / QPC / phase profiler.

For per-kernel SPIR-V metrics (occupancy, warp stalls, etc.): use **Nsight
Graphics → GPU Trace Profiler** (separate tool from NCU). NCU 2025.4 is
CUDA-only.

---

## Known accepted residuals (NOT blocking)

1. **web.txt small-file regime** — decode 2.6×, encode 1.73× CUDA. Structural
   Vulkan-on-WDDM submit floor. Four-iteration negative-result chain
   (iter-13/14/15 on decode + investigations on encode) confirmed not
   fixable from app side.

2. **L3 silesia 0.02% larger than CUDA** — Vulkan `maxStorageBufferRange =
   4 GiB - 1` on all desktop GPUs. At L3 (`hash_bits=19`), workloads with
   `num_chunks ≥ 2048` (single-frame inputs ≥ 128 MiB at sc=0.25) would
   exceed the cap. Phase 2A.5 (`30f36d3`) clamps `hash_bits` down to 18 for
   those workloads. Cost is 0.02% larger output on silesia L3; no effect
   on any other workload. CUDA's raw device-pointer addressing has no
   equivalent per-binding limit. See "Future improvement #1" below.

3. **Pre-existing single-thread mode `TestExpectedEqual` (2 tests)** —
   `SLZ_VK_TEST_THREADS=1` surfaces test ordering sensitivity that predates
   recent encode work. Parallel mode (the spec'd config) is rock-solid at
   83/0/0. Documented but unfixed.

4. **One-shot CLI shows 1 OOM warning per direction per process** — best-
   practices layer flags first `vkAllocateMemory` failure for non-pinned
   host buffers. Sticky disable flips on first OOM (commit `0af24ff`); bench
   mode is silent (pinned buffers).

5. **L3/L4/L5 decode `gpu kernel best` time 1.20x-1.37x CUDA on large
   workloads** — Phase 5 perf sweep (2026-06-08, see `srcVK/PerfSweep.md`)
   measured this as a residual structural cost driven by A-006 (explicit
   compute-to-compute barriers), A-007 (per-binding offset descriptor
   ABI), and unfused `compact_raw_descs` / `gather_raw_off16` /
   `merge_huff_descs` dispatches (A-017 only fused `compact_huff_descs`).
   **e2e** decode timings stay inside the 10% bar at every (level, corpus)
   large-workload cell (0.96x-1.03x) because host overhead amortizes the
   kernel-time gap at 95-203 MB inputs. Catalogued as **A-021** in
   `srcVK/PortAdaptations.md`. Future close path: apply A-017's fusion
   pattern to the remaining three dispatches.

## Phase 5 — Conformance + perf parity (2026-06-08, DONE)

- Cross-backend test coverage extended in `srcVK/tests/l3_l4_cross_backend.zig`:
  added VK->CUDA reverse direction for L3/L4 (8 tests), plus L5 in both
  directions, L5 SHA byte-identity, L5 VK->VK pattern, and L5 real-corpus
  (10 tests). Total +18 tests. ptest_vk: 122/9/0 -> **140/9/0** on both
  NVIDIA RTX 4060 Ti and Intel iGPU.
- Full perf sweep across L1-L5 x web/enwik8/silesia x decode/encode x VK/CUDA
  (60 cells) — see `srcVK/PerfSweep.md` for the formatted tables. Verdict:
  **e2e parity within 10% bar on every large-workload cell** at every
  level; encoder kernel times 0.81x-1.01x CUDA (multiple FASTER cells).
  Two residuals catalogued: A-008 (already known; silesia L3 0.019% size
  delta confirmed unchanged) + **A-021** (new; L3/L4/L5 decode kernel-time
  gap on large workloads, e2e still inside bar).

## Future improvements (optional; NOT blocking)

### #1: Buffer-Device-Address (BDA) loads for >4 GiB SSBO workloads

Phase 2A.5 (`30f36d3`) accepts a 0.02% compression cost on silesia L3
because Vulkan's `maxStorageBufferRange = 4 GiB - 1` would otherwise
truncate the hash table binding at L3 on large inputs. For genuinely
larger workloads (L3 inputs ≥ ~200 MiB), the same cap will affect them
the same way (clamp to `hash_bits=18`).

**The Vulkan-native fix:** rewrite the LZ kernel's hash table accesses
to use `VK_KHR_buffer_device_address` (bufferDeviceAddressFeatures).
BDA loads/stores bypass the per-binding cap entirely — the kernel
addresses memory by raw `uint64_t` device pointer, the same way CUDA
addresses device memory. The cap only applies to descriptor-bound
buffers.

**Scope:**
- Enable `bufferDeviceAddressFeatures` at VkDevice creation (already
  enabled per finishingL1.md V12 features — confirm)
- Modify `srcVK/encode/lz_encode_kernel.comp` to take `d_hash_persist`
  as a `uint64_t` BDA address (push constant) instead of an SSBO binding
- Modify host glue to pass `vkGetBufferDeviceAddress()` result instead
  of binding the descriptor
- Verify SHA byte-identical to CUDA on silesia L3 (would close the
  0.02% gap completely)

**Risk:** medium. The LZ kernel is the most-tuned kernel in the codebase
(NCU-verified parity with CUDA PTX). BDA loads compile differently than
SSBO loads — verify perf doesn't regress.

**Effort:** ~1 single agent, ~3-4 hours. Optional — current residual is
0.02% which is well within compression-ratio noise on real workloads.

---

## Recent commit history (last 20)

```
db4406c  tools/build_vk.bat — clean-build script (workaround for A-012 stale-SPV)
4270ea4  Phase 2A-decoder iter 4f — 8-arm raw-mode dispatch resolves A-001 (108/0/0 GREEN; L3+L4 VK→VK MATCH on all 3 goldens)
b5e11c9  PortAdaptations.md — catalog of CUDA-VK divergences
8002198  ToDo.md wave update for iter 4 + 4b + 4c + 4d
b3fab31  Phase 2A-decoder iter 4d — add 20 L3/L4 ptest cases (10 currently failing) as deterministic reproducer for sub-chunk-N>0 decoder bug
11eb101  Phase 2A-decoder iter 4c — per-binding offset ABI + 5 dispatch site fixes (L3+ decode still broken)
bcaa1f1  Phase 2A-decoder iter 4b — fix .level defaulting to 1 (silent corruption → loud failure)
ac013b5  Phase 2A-decoder iter 4 — port LZ general workhorse + L2 gate (partial; 855 LOC)
de9d7eb  Phase 2A-decoder iter 3 follow-up — huff_decode conformance test (catches bitbufRefill bug class)
ac6696f  Phase 2A-decoder iter 3 fix — bitbufRefill hi/lo swap
d5860d1  Phase 2A-decoder iter 3 — port huff_decode_4stream + wire runHuffBuildAndDecode
6a1da40  Phase 2A-decoder iter 2 — port huff_build_lut faithfully (8 passes)
5b438bc  Phase 2A-decoder iter 1 — port gpuScanChunks host driver
1509bee  Phase 3 follow-up — fix latent host param-packing bug in 3 L2-gated dispatch sites
4ef283b  Phase 3 — port 6/7 decode pipeline kernels + open L1 D2D entry
b4747ff  ToDo.md — clarify decompressFramedFromDevice is D2D at ALL levels
148162a  ToDo.md — mark Phase 2A encoder + 2A.5 shipped
30f36d3  Phase 2A.5 — clamp hash_bits for Vulkan 4 GiB SSBO cap (silesia L3 1.176x → 1.0002x)
f9e84e3  Phase 2A encoder — full faithful Huffman encoder port (L3 + L4 encode shipped)
cabf64e  ToDo.md — mark L2 SHIPPED (Phase 2-prep done)
7bbcf77  ptest_vk — add 9 L2 test cases (5 in-process + 4 cross-backend SHA gate)
38cbaea  ToDo.md — update for L1 done on both decode + encode
d9d8bba  encode iter-4 — final-frame D2H import path (3.2x encode speedup)
d83ea21  encode per-phase QPC profile — locates 238 ms d2h_final as the 190 ms gap
02560d6  persistent VkPipelineCache — disk-backed across process launches
7d37231  encode iter-3 — D2H gather (sync transfer-queue): 2.0x speedup
81bde77  Revert "encode iter-1 H2 — stream-routing + D2H cache hazard fix"
7d368fa  Revert "encode iter-2 — D2H gather collapses per-chunk loop"
d4f1f41  replace finishingL1.md with ToDo.md — L1 done, L2-L5 roadmap
0af24ff  OOM noise cleanup — per-direction sticky disable
e8fc91f  tests: inherit parent env in subprocess test harnesses (74/0/0)
23d8ab3  valfix C — Int8 capability glslc workaround
7960b98  valfix B — cmdbuf state via deferred staging destroy
9022dde  valfix A — handle types on imported VkBuffer
6ca302c  fix two Vulkan constants — subgroup pin now actually pins
332a04c  pin compute pipelines to subgroupSize=32 + device-pick guard
bfe746a  encode iter-1 H3 — verify iter-12 import auto-fires on encode H2D
bb675ab  encode iter-1 H1 — per-chunk D2H with srcOffset (2.2-2.6x faster)
7bc951e  decode iter-15 — single submit + single wait per decode
```

All on `main`. Nothing pushed to remote yet.

---

## Project rules (from `CLAUDE.md`)

**NEVER REVERT WITHOUT PERMISSION.** Never run `git checkout --`, `git
restore`, `git revert`, or any command that undoes file changes without the
user explicitly saying "revert", "undo", or "restore". If a change made
things slower or didn't help, LEAVE THE CODE AS-IS and report the results.
The user decides whether to revert. No exceptions.

---

## Memory rules (from auto-memory)

Relevant entries in `~/.claude/projects/c--Users-james-JAMESWORK2025-Repos-Compressor-GPU/memory/`:

- `feedback-port-dont-reinvent` — mirror CUDA values/dispatch shapes/workgroup sizes verbatim
- `feedback-port-means-port` — if CUDA does X on CPU and VK does X on GPU, that's a port violation
- `feedback-verify-device-name` — perf measurements must verify "Device: NVIDIA GeForce RTX 4060 Ti" appears (default may pick Intel iGPU)
- `feedback-decode-over-encode` — SUPERSEDED, encode is in scope and complete on L1
- `feedback-l1-completion-bar` — SUPERSEDED, current done-bar is exceeded on both decode + encode

---

## Recommended next-session workflow

1. **Read this whole file.** Then `gameplan.md` Section 8 for additional context.
2. **Verify build still works:**
   ```powershell
   Remove-Item -Force zig-out/bin/streamlz_vk.exe
   zig build streamlz_vk -Doptimize=ReleaseFast
   ```
3. **Verify tests still 74/0/0 on both backends:**
   ```powershell
   $env:SLZ_VK_DEVICE_INDEX = "1"; zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
   $env:SLZ_VK_DEVICE_INDEX = "0"; zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
   ```
4. **Verify perf still in shape:**
   ```powershell
   $env:SLZ_VK_DEVICE_INDEX = "1"
   ./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz   # decode
   ./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt              # encode
   ```
5. **Verify SHA byte-identical to CUDA on all 3 goldens** — see "How to measure" above.
6. **Verify validation layer still clean** — see "Lessons learned" above.
7. **If working on encode changes: ALWAYS re-run the SHA check after.**
8. **Pick a phase from the L2-L5 roadmap above** and launch.
