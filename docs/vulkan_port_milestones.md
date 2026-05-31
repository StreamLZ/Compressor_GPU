# Vulkan Port — Milestone Plan (Approved)

## 0. Locked constraints

1. **Shaders:** GLSL → glslc → SPIR-V. Vulkan 1.3 target for Tier-1, Vulkan 1.2 floor for Tier-2. glslc ≥ 1.3.250 pinned (defends SPIRV-Tools BDA storage-class regression).
2. **Vendor coverage:** Tier-1 = NVIDIA Turing+, AMD RDNA2+ (when driver reports `subgroupSize == 32`), Intel Arc Alchemist+ (when driver pins 32). Tier-2 = everything else — Adreno (all gens), Mali (Bifrost + Valhall — Mali never enters Tier-1), pre-Arc Intel iGPU, AMD GCN/CDNA wave64, RDNA2 when driver reports 64.
3. **Sibling build:** `src_vulkan/` next to `src/`; CUDA `src/` untouched. Artifacts: `streamlz_vk.exe`, `streamlz_vk.dll`. All C symbols suffixed `_vk`. Same CLI flags as `streamlz.exe`.
4. **Full API parity day 1: 16 symbols.** 14 sibling mirrors of `include/streamlz_gpu.h` + the Tier-2 buffer registration pair `slzRegisterBuffer_vk` AND `slzUnregisterBuffer_vk`. The pair is exported unconditionally on both tiers (both are no-ops on Tier-1, mandatory on Tier-2). Lifetime contract per arch §13: registrations persist until `slzUnregisterBuffer_vk` or `slzDestroy_vk`. Architecture §13 (which the previous draft mis-summarized as "15 symbols") is authoritative; this plan is corrected.
5. **Wire format:** byte-identical to CUDA `.slz` output at L1..L5. Cross-backend roundtrip mandatory.
6. **Compression levels:** L1..L5 produce identical bytes per level as the CUDA encoder.
7. **Architecture authority:** `docs/vulkan_port_architecture.md` is authoritative. Anything inconsistent with that document is wrong.
8. **New status codes (§13 mandate):** the shared `slzStatus_t` enum is extended in M2 with `SLZ_ERROR_DEVICE_LOST = 8` and `SLZ_ERROR_VK_FEATURE_MISSING = 9`. Both headers (`include/streamlz_vk.h` AND `include/streamlz_gpu.h`) carry the additions because the enum is shared between the CUDA and Vulkan ABIs.

## 1. Plan overview

**Winning lens:** TOP-DOWN PLANNER (shippable-but-functionally-empty skeleton first, then port kernels in arch §11 order against a live conformance harness), with key grafts from RISK-DRIVEN PLANNER (standalone `__match_any_sync` perf microbenchmark on a minimal pre-existing dispatch/timing chassis BEFORE production kernel integration; real-mobile probe lane in the early R2-retirement phase; explicit CUDA prereqs at M0 with zero deps so they run in parallel with Vulkan scaffolding) and from BOTTOM-UP PLANNER (dedicated common-GLSL-helpers milestone with per-primitive unit tests; CUDA prereqs as root milestones so golden fixtures are valid before harness is built; XFAIL bookkeeping discipline that exits non-zero when an unexpected XFAIL flips to PASS).

The plan ships a **buildable but-graceful streamlz_vk binary at milestone 2** with all 16 `_vk` symbols stub-exported returning a NAMED `SLZ_ERROR_UNSUPPORTED` for every operation that has no implementation yet. Conformance harness lands at milestone 3 with a quantified `0 / 350 passing` dashboard so every subsequent milestone has visible progress delta. Per-stage byte-identity goldens land at M3a (companion to M3) so the per-kernel acceptance criteria from M11 onward are actually testable. Risk retirements R1-R5 land by M9 BEFORE any production kernel ships (M5 standalone match_any benchmark — running on the minimal M8a dispatch/timing chassis split out for exactly this purpose — catches the AMD/Intel perf cliff at milestone 5, not at milestone 22). Wave 1 decode kernels (M10-M20) land in locked arch §11 order against the live harness. Wave-1 gate (M21) hard-fails until {CUDA encode → VK decode} passes at L1..L5 on Tier-1 + Tier-2. Wave 2 encode (M22-M27) and full-matrix conformance close out the critical path. Wave 3 perf + real-mobile lanes (M28-M30) close v1.0-alpha (M31).

**Total milestone count: 34** (previously 31; split-outs documented in §4). Estimated total LOC: ~25,500 (mean ~750, median ~600, max 1700 for first encode kernel slice M22b).

**Critical path length: 34 milestones.** Hard gates: M3 (conformance harness wired, 0/350 baseline), M3a (per-stage CUDA goldens regenerated), M9 (all 5 risks retired before Wave 1), M21 (Wave-1 decode CUDA→VK byte identity at L1..L5), M27 (full matrix 350/350), M31 (v1.0-alpha tag with the v1.0-alpha cutline lanes green; see §7).

## 2. Critical path

```
M0   → M1   → M2   → M3   → M3a  → M4   → M8a  → M5   → M6   → M7   → M8b  → M8c  → M9   → M10
(CUDA prereqs)
       → (scaffolding + workgroup-size CI guard + Android cross-compile)
              → (16-symbol stub ABI + 2 new status codes)
                     → (cross-backend harness, 350-cell matrix)
                            → (per-stage CUDA dump instrumentation + 17 stage golden sets)
                                   → (loader + probe + process-wide mutex + R2 classification retire)
                                          → (minimal dispatch + timing primitives — bare chassis only)
                                                 → (match_any standalone perf, R1 retire — hard pre-committed 65% floor with no UNTESTED-pass)
                                                        → (BDA + register/unregister + R4 retire)
                                                               → (memory + 2GB split + Adreno offset-alignment runtime probe + R5 retire)
                                                                      → (production pipeline cache + descriptor factory + invalidate hook)
                                                                             → (sync2/sync1 wrapper + VkQueryPool 68 slots + tier1_nv cache)
                                                                                    → (common GLSL helpers + per-primitive unit tests vs CUDA reference)
                                                                                           → (hello-dispatch substrate gate)

M10 → M11 → M12 → M13 → M14 → M15 → M16 → M17 → M18 → M19 → M20 → M21
(walk_frame + decode_context_vk + decode_dispatch_vk land)
       → (lz_decode_raw) → (prefix_sum_chunks + scan_gpu_vk) → (scan_parse — first byte-identity gate)
       → (compact_huff+raw) → (gather_raw_off16) → (merge_huff_descs)
       → (huff_build_lut — match_any site #3 production)
       → (huff_decode_4stream) → (lz_decode — closes CUDA-enc → VK-dec end-to-end via production code path)
       → (WAVE-1 GATE)

M21 → M22a → M22b → M23 → M24 → M25 → M26 → M27
       → (encode_context_vk + encode_lz_vk + lz_encode L1 only NVIDIA baseline)
              → (lz_encode L2..L5 + AMD/Intel + Tier-2)
              → (huff_build_tables + encode_huff_vk) → (huff_encode_4stream)
              → (assemble_measure + encode_assemble_vk) → (assemble_write)
              → (frame_assemble — closes VK-enc end-to-end via production code path)
              → (FULL CONFORMANCE GATE: 350/350 on v1.0-alpha cutline lanes)

M27 → M28 → M29 → M30 → M31
       → (Wave-3 perf: NV_partitioned + shape-keyed cmd cache bounded at 16)
       → (lane F real-mobile Adreno + Mali) → (lane G real AMD CDNA wave64)
       → (v1.0-alpha tag + CLI parity + docs; see §7 cutline for what can ship without)
```

Parallel branches (can ship out-of-band): M0 ↔ M1 (CUDA prereqs are independent of Vulkan scaffolding). M6 (BDA) ↔ M7 (memory allocator) once M5 is in. M28 ↔ M29 ↔ M30 (perf tuning + hardware lanes) once M27 is green.

## 3. Dependency graph

| Milestone | Depends on | Type |
|-----------|------------|------|
| M0 | (none) | CUDA prereq + CUDA-side compressed_size pseudo-kernel slot + comptime-assert rename in cuda_ffi.zig |
| M1 | (none) | Vulkan scaffolding + Android cross-compile + workgroup-size CI guard |
| M2 | M1 | Stub 16-symbol ABI + 2 new status codes + CLI graceful failure |
| M3 | M0, M2 | Cross-backend conformance harness with golden fixtures |
| M3a | M0, M3 | Per-stage CUDA dump instrumentation + 17 stage golden sets + stage-assertion harness |
| M4 | M2 | Loader + vendor probe + process-wide mutex (R2 classification retire) |
| M8a | M4 | Minimal dispatch + timing primitives chassis (extracted from M8) |
| M5 | M4, M8a | match_any standalone microbenchmark (R1 retire — hard pre-committed floor) |
| M6 | M4, M8a | BDA + slzRegisterBuffer_vk + slzUnregisterBuffer_vk (R4 retire) |
| M7 | M6 | Memory allocator + 2GB split + Adreno offset-alignment probe + maxMemoryAllocationCount check (R5/R6 retire) |
| M8b | M5, M7 | Production pipeline cache + descriptor factory + invalidate_all_cached_cbs hook |
| M8c | M8b | Sync2/sync1 wrapper + VkQueryPool 68-slot per arch §15 indexing + tier1_nv cache |
| M9 | M8c | Common GLSL helpers + per-primitive unit tests (vs CUDA reference) + layout(subgroup_size=32) audit |
| M10 | M3a, M9 | Hello-dispatch substrate gate |
| M11 | M3a, M10 | Kernel 1: walk_frame (D1) + decode_context_vk.zig + decode_dispatch_vk.zig |
| M12 | M11 | Kernel 2: lz_decode_raw |
| M13 | M12 | Kernel 3: prefix_sum_chunks + scan_gpu_vk.zig |
| M14 | M13 | Kernel 4: scan_parse (D2) — first byte-identity gate |
| M15 | M14 | Kernel 5: compact_huff + compact_raw (D4/D5) |
| M16 | M15 | Kernel 6: gather_raw_off16 |
| M17 | M16 | Kernel 7: merge_huff_descs (D7) |
| M18 | M5, M9, M17 | Kernel 8: huff_build_lut — match_any site #3 production |
| M19 | M18 | Kernel 9: huff_decode_4stream |
| M20 | M19 | Kernel 10: lz_decode — closes CUDA-enc → VK-dec via production code path |
| M21 | M20 | **WAVE-1 GATE**: 70/350 decode conformance L1..L5 on cutline lanes |
| M22a | M0, M5, M9, M21 | Kernel 11a: lz_encode L1 + NVIDIA-only baseline + encode_context_vk.zig + encode_lz_vk.zig |
| M22b | M22a | Kernel 11b: lz_encode L2..L5 + AMD/Intel + Tier-2 + hash-size@create + Tier-2 descriptor layout |
| M23 | M22b | Kernel 12: huff_build_tables + encode_huff_vk.zig |
| M24 | M23 | Kernel 13: huff_encode_4stream |
| M25 | M24 | Kernel 14: assemble_measure + encode_assemble_vk.zig |
| M26 | M25 | Kernel 15: assemble_write |
| M27 | M26 | Kernel 16: frame_assemble — closes VK-enc end-to-end via production code path |
| M28 | M27 | **FULL CONFORMANCE GATE + Wave-3 perf**: NV_partitioned + shape-keyed cmd cache bounded at 16 |
| M29 | M1, M27 | Lane F real-mobile (Adreno Pixel 8 + Mali Pixel 6) |
| M30 | M27 | Lane G real AMD CDNA wave64 |
| M31 | M28, M29, M30 | v1.0-alpha tag — see §7 cutline for which deps may be deferred to v1.1 |

## 4. Milestone split-outs from the draft (and why)

Three structural changes vs the draft (driven by reviewer fatal flaws):

1. **M3 → M3 + M3a:** the draft's per-stage byte-identity acceptance criteria from M11 onward were physically untestable because no CUDA-side per-kernel intermediate dumps existed. M3a adds CUDA-side instrumentation hooks (encoder + decoder) that emit per-kernel intermediate blobs for all 70 fixtures × (11 decode + 6 encode) kernels, plus the stage-assertion harness extension in `cross_backend_tests.zig`. ~700 LOC.

2. **M8 → M8a + M8b + M8c:** the draft's M8 bundled pipeline-cache + descriptor factory + sync wrapper + VkQueryPool into one 1500-LOC milestone and put M5 ahead of it — but M5's match_any microbenchmark needs a dispatch+timing substrate to produce its numbers. The split extracts M8a as a minimal "bare dispatch + timing primitives only" milestone that M5 (and M6) can depend on, then M8b/M8c land the production-grade pieces (pipeline cache, descriptor factory with invalidate hook, sync2/sync1 wrapper, VkQueryPool with the arch §15 68-slot indexing).

3. **M22 → M22a + M22b:** the draft's M22 layered four untested-in-combination dependencies (M0 hash rewrite + M5 match_any site #1 + M7 bin-packed multi-buffer + Tier-1/Tier-2 split) at 1500 LOC. The split lets M22a land lz_encode L1 NVIDIA-only as a baseline (so the M0 hash-store rewrite is validated end-to-end with the smallest possible cross-section before AMD/Intel/Tier-2/L2..L5 are added on top in M22b).

The plan also adds **Android cross-compile + pipeline-cache path abstraction to M1** because the draft made M29 (real mobile) depend on a build target that didn't exist. M1's `vk_android` step extends `build.zig` with NDK + arm64-v8a output and `vk_pipeline.zig`'s cache-directory helper becomes OS-portable (`%LOCALAPPDATA%` on Windows, `$XDG_CACHE_HOME` on Linux, `/data/data/<pkg>/cache/` on Android).

## 5. CUDA-side prerequisites

All CUDA-side changes mandated by architecture §6, §9.5, §13, §14 land in **M0** as a single root milestone so they execute in parallel with Vulkan scaffolding (M1):

| Change | File / Location | Consumed by milestone | Purpose |
|--------|-----------------|-----------------------|---------|
| Highest-lane-winner rewrite at site A | `src/encode/lz_greedy_parser.cuh` — non-atomic store `ht[h] = my_pos` (currently at line ~187; line numbers are approximate, the rewrite is keyed on the store-statement identity, not the line) | M22a/M22b (lz_encode) | Replace NVIDIA-implicit "highest lane wins" with explicit `__match_any_sync(active_mask, h)` + `31 - __clz(group_mask)` highest-indexed-lane election. Byte-equivalent on NVIDIA (CUDA goldens remain bit-exact); makes byte-identity portable to Vulkan AMD/Intel. Tracked as `streamlz/issues/cuda-deterministic-hash-store`. |
| Highest-lane-winner rewrite at site B | `src/encode/lz_greedy_parser.cuh` — L2+ rehash store `ht[hashKey6(rk, hash_bits, hash_mask)] = rp` (currently at line ~311; line is approximate) | M22b (lz_encode L2..L5) | Same treatment as site A — explicit highest-lane-winner store. Required for L2..L5 byte identity. |
| Site #2 (key4 match) explicit rewrite | `src/encode/lz_greedy_parser.cuh` lines 82-95 | M22a (lz_encode L1+) | Rewrite CUDA's `__match_any_sync(FULL_WARP_MASK, key4)` with the same `subgroupShuffle(key4, highest_lower) == key4` equivalence pattern the Vulkan port uses (arch Appendix A). Byte-equivalent on NVIDIA (Appendix A proof + new exhaustive test). Eliminates the "site #2 semantic mismatch" debug branch from M22's failure tree — both backends now execute byte-identical code paths. |
| `reserved[0]` → `effective_level_out` rename + comptime asserts | `include/streamlz_gpu.h` slzCompressOpts_t; `src/encode/cuda_ffi.zig` comptime asserts that name `reserved` | M3 (conformance harness clamp-aware equality) and M22+ | Repurpose first padding slot (currently `int reserved[6]` at line 117); struct stays 32 B; rename comptime asserts in `cuda_ffi.zig` that reference `reserved` to `effective_level_out`. CUDA writes `opts.level` (no-op semantically on the CUDA path — see below); Vulkan Tier-2 writes post-clamp level when mobile VRAM forces L5→L3. Both CLIs surface clamps via stderr. |
| `slzCompressAsync` `compressed_size` contract change (CUDA-side aligned with VK) | `include/streamlz_gpu.h` slzCompressAsync docstring + CUDA `slzAssembleMeasure` post-return drain | M0 (initially) + M25 (validates) | CUDA `slzCompressAsync` is changed to NOT write `compressed_size` before return — both backends source the size via the kernel-timing drain. **This IS a CUDA-side behavior change**, not just a docstring update. Caller-migration plan: bump library minor version to 3.0.0; CHANGELOG entry; existing CUDA callers reading post-return size must call `slzGetLastTimings` to drain. The `slzKernelTiming_t` schema gains a `compressed_size` / `decompressed_size` pseudo-kernel entry (arch §13/§14). The pre-rewrite contract was physically impossible to honor on Vulkan; aligning CUDA preserves "one ABI, two backends." |
| Status enum additions (CUDA-side header patch) | `include/streamlz_gpu.h` slzStatus_t enum | M2 (export both CUDA-side and VK-side) | Add `SLZ_ERROR_DEVICE_LOST = 8` and `SLZ_ERROR_VK_FEATURE_MISSING = 9` to the shared enum so both backends speak the same status vocabulary. CUDA backend never returns these values today, but the slot reservation prevents future Vulkan additions from re-numbering. |

**Validation**: M0 regenerates all 14 silesia × L1..L5 = 70 CUDA-encoded `.slz` golden fixtures and commits a SHA-256 manifest. If post-rewrite goldens differ from pre-rewrite goldens at any of the three CUDA-side hash sites (sites A, B, and the new #2 rewrite), M0 fails — see Rip-cord conditions in §6.

## 6. Rip-cord conditions and fallbacks

High-risk milestones pre-commit a numeric/boolean failure trigger, a named fallback, and the downstream-milestone impact:

| Milestone | Rip-cord trigger | Named fallback | Downstream impact |
|-----------|-----------------|---------------|-------------------|
| **M0** | Post-rewrite CUDA goldens at sites A/B/#2 differ from pre-rewrite goldens on NVIDIA reference hardware | (1) Revert sites A and B and the #2 rewrite; (2) re-baseline goldens with stakeholder sign-off and document the new hashes; (3) document NVIDIA-only initial scope in CHANGELOG; (4) file `streamlz/issues/v1.1-portable-hash-store` to retry the rewrite for v1.1 | M22a/M22b lose AMD/Intel byte-identity unless re-baselined goldens are accepted; if not accepted, v1.0-alpha cutline §7 strips AMD/Intel from Tier-1 and they drop to Tier-2 |
| **M4** | A Tier-1 whitelisted vendor (NVIDIA Turing+, AMD RDNA2+, Intel Arc Alchemist+) fails the probe in production | (1) File blocker; (2) remove the failing vendor from the Tier-1 whitelist in arch §3; (3) push them to Tier-2 with documented perf penalty (25-50% CUDA); (4) update CHANGELOG before M11 starts | M11+ Tier-1 acceptance criteria drop the affected vendor; v1.0-alpha cutline (§7) updates the device matrix |
| **M5** | AMD or Intel throughput < 65% of CUDA on either match_any site **AND no real hardware available to remeasure** | UNTESTED-pass is DISALLOWED (see acceptance below). Either (a) stand up real AMD + real Intel lanes (cost: hardware acquisition pulled forward) OR (b) explicitly defer AMD/Intel R1 retirement and downgrade M22a/M22b acceptance to NVIDIA-only with AMD/Intel marked `pending real hardware in v1.0-final` | M22a/M22b Tier-1 acceptance is NVIDIA-only if R1 is deferred; cutline §7 ships v1.0-alpha as NVIDIA-Tier-1-only with documented perf disclosure |
| **M7** | Adreno minStorageBufferOffsetAlignment probe returns > 4 KiB OR maxMemoryAllocationCount < 64 on a candidate Tier-2 device | Reject the device with `SLZ_ERROR_VK_FEATURE_MISSING` at slzCreate_vk; document in CHANGELOG that the device is unsupported in v1.0-alpha; file v1.1 issue to add a sparser allocation strategy for sub-64-allocation devices | M29 (mobile lane) device list drops the failing device; v1.0-alpha cutline §7 updates accordingly |
| **M14** | scan_parse diverges on every case at every level on lane A (worst-case but realistic on first wire-format-heavy kernel) | (1) Triage protocol: first diff in `tests/diffs/{case}_L{n}_scan_parse.diff` uses the format `OFFSET   CUDA_BYTES   VK_BYTES   STAGE_FIELD_NAME`; (2) categorize by failure mode (endianness, alignment, wire-format constant, sc_group_size leak); (3) escalation: if 3 working days of fix attempts don't close any failure mode, file `streamlz-vk/issues/scan-parse-blocker` and pause M15-M27; (4) hard scope-cut option: ship v1.0-alpha with VK decode disabled, CUDA-encode + CUDA-decode only (defeats Wave-1 gate but preserves shipability) | M15-M27 paused; v1.0-alpha cutline §7 ships with VK decode disabled if scope-cut is invoked |
| **M21** | Wave-1 gate red on any cutline lane for > 3 consecutive nightly runs | (1) Bisect to the kernel responsible; (2) revert that kernel's milestone tag; (3) re-run gate; (4) if the regression is shared-helper (M9), pause Wave 2 and patch M9 with a regression test that prevents recurrence; (5) timeout: 5 working days without resolution triggers cutline §7 v1.0-alpha-minus-the-broken-lane shipment | Cutline §7 may strip a lane from v1.0-alpha; M22+ continues on green lanes only |
| **M22a/M22b** | match_any site #1 fails on AMD or Intel in production despite passing M5 microbenchmark | (1) Compare M5 standalone perf to M22a/M22b production perf — if production is materially worse, the bug is integration (other code in the kernel), fixable inside lz_encode.comp without re-architecting; (2) if production matches M5 and both are <65%, file architecture revision per M5's playbook; (3) hard option: ship NVIDIA-only Tier-1; AMD/Intel demoted to Tier-2 with documented perf gap | Cutline §7 reflects the demotion |
| **M27** | Full conformance gate < 350 / 350 on any cutline lane | (1) Bisect to the failing cell direction (CUDA→CUDA, CUDA→VK, VK→CUDA, VK→VK); (2) failing cell points at encode vs decode; (3) if the failure is encode-only and isolated to a level (e.g. L5 on Intel only), document Intel-L5-as-Tier-2 in CHANGELOG; (4) timeout: 5 working days triggers cutline §7 partial release | Cutline §7 may release v1.0-alpha with a Tier-1 vendor's specific level demoted |
| **M29** | Real-mobile lane F not greenable by release date (phone broken, adb dropped, hardware unobtainable) | Defer Mali OR Adreno (whichever is unavailable) to v1.1; release v1.0-alpha with one mobile lane validated and the other documented as "v1.1 target." Per cutline §7, at least one real-mobile lane is REQUIRED for v1.0-alpha | If both lanes fail, v1.0-alpha ships desktop-only and mobile becomes v1.1 |
| **M30** | Cloud-rented CDNA budget exceeded OR CDNA driver issue blocks validation | Mark wave64 as v1.1-validated and ship v1.0-alpha with SwiftShader-wave64 lane B' as the wave64 surrogate; document in CHANGELOG | Cutline §7 allows v1.0-alpha to ship without real CDNA |

## 7. v1.0-alpha minimum viable cutline

The plan deliberately splits MUST-have / SHOULD-have / NICE-TO-HAVE so the project ships even if hardware acquisition slips:

**MUST-have (v1.0-alpha will not ship without these):**
- M0-M9: substrate
- M10-M21: Wave 1 (decode end-to-end on at least Tier-1 NVIDIA + Tier-2 lavapipe)
- M22a-M27: Wave 2 (encode end-to-end on at least Tier-1 NVIDIA + Tier-2 lavapipe)
- M28: Wave-3 perf (NV_partitioned + shape-keyed cmd cache bounded at 16)
- **At least ONE of {M29 Adreno, M29 Mali, M30 CDNA} green** — establishes that the "tier-2 emulation actually works on real hardware" claim is validated
- M31: tag + docs + CLI parity

**SHOULD-have (target for v1.0-alpha, ship without disables features but does not block):**
- M5 hard 65% AMD/Intel match_any floor met. If missed and a rip-cord fires, AMD/Intel demote to Tier-2 with documented perf gap and disable strings in `slzStatusString_vk`:
  - On a Tier-2-demoted AMD device, the library reports `"AMD Tier-2 mode: subgroup match_any emulation below 65% performance floor; see CHANGELOG"`
  - On a Tier-2-demoted Intel device, the library reports `"Intel Tier-2 mode: subgroup match_any emulation below 65% performance floor; see CHANGELOG"`
- All three of M29 Adreno, M29 Mali, M30 CDNA green (vs the MUST of just one)

**NICE-TO-HAVE (v1.1 target if slipped):**
- Bifrost real-mobile lane (e.g., Pixel 3 Mali-G76)
- macOS / MoltenVK path
- The deferred items in arch §21

**Release-without criteria for each potentially-deferred lane:**
- *Without M29 Adreno*: `slzCreate_vk` on Adreno returns `SLZ_ERROR_VK_FEATURE_MISSING` with status string `"Adreno unsupported in v1.0-alpha; v1.1 target"`. Slz binary refuses to run on Adreno-vendor-ID; docs/Compatibility section notes the gap.
- *Without M29 Mali*: same, with `"Mali unsupported in v1.0-alpha; v1.1 target"`.
- *Without M30 CDNA*: `slzCreate_vk` succeeds, but on CDNA the status string adds `"CDNA wave64 path not validated on real hardware; SwiftShader lane B' covered emulation only — production use discouraged"`.

Hardware-acquisition slip-paths: M29 cloud rental of an Adreno Cloud (e.g., AWS Snapdragon-on-cloud, or local lab Pixel acquisition) is acceptable. M30 cloud-rented CDNA (AMD Instinct via cloud) is acceptable. Both have budget ceilings documented in §11 operational-costs (sized below).

## 8. Milestones

### M0 — CUDA-side prerequisites: highest-lane-winner (sites A, B, #2) + reserved[0] rename + status enum extension + compressed_size contract change + pseudo-kernel timing slot

**Depends on:** (none — root milestone, runs parallel to M1)

**Scope:** Five CUDA-side edits in a single milestone so the byte-identity reference and shared ABI are locked before any Vulkan work consumes them.

1. **Site A rewrite** in `src/encode/lz_greedy_parser.cuh` (hash store `ht[h] = my_pos`, approximate line 187): replace the non-atomic store with explicit highest-lane-winner via `__match_any_sync(active_mask, h)` + `31 - __clz(group_mask)`. Gate the store to only the elected lane. Byte-equivalent to current NVIDIA implicit-order behavior; makes the implicit convention explicit so it ports to Vulkan AMD/Intel/Mali.
2. **Site B rewrite** in `src/encode/lz_greedy_parser.cuh` (L2+ rehash store `ht[hashKey6(...)] = rp`, approximate line 311): same treatment. Required for L2..L5 byte identity.
3. **Site #2 explicit rewrite** in `src/encode/lz_greedy_parser.cuh` (key4 match block, approximate lines 82-95): replace `__match_any_sync(FULL_WARP_MASK, key4)` with the same `subgroupShuffle(key4, highest_lower) == key4` equivalence pattern the Vulkan port will use. Byte-equivalent on NVIDIA per arch Appendix A (and a new exhaustive equivalence test, see M5 acceptance). Eliminates "site #2 semantic mismatch" from M22's debug branches — both backends execute byte-identical code paths.
4. **`reserved[0]` → `effective_level_out` rename**: `include/streamlz_gpu.h` slzCompressOpts_t (struct currently has `int reserved[6]` at line 117; rename the first slot, struct stays 32 B). Update `src/encode/cuda_ffi.zig` comptime asserts that reference `reserved` (rename them to `effective_level_out`). CUDA `slzCompressAsync` writes `opts.level` into the new field before returning.
5. **`compressed_size` contract change**: CUDA `slzCompressAsync` stops writing compressed_size before return. Drain via `slzGetLastTimings`. Add `compressed_size` + `decompressed_size` pseudo-kernel slot to `slzKernelTiming_t` per arch §13/§14. Bump library minor version to 3.0.0; CHANGELOG migration note.
6. **`slzStatus_t` enum extension**: add `SLZ_ERROR_DEVICE_LOST = 8` and `SLZ_ERROR_VK_FEATURE_MISSING = 9` to `include/streamlz_gpu.h` (shared by both backends).
7. Regenerate all 70 CUDA goldens at L1..L5 on the 14-case silesia mini-corpus; commit `tests/golden_slz/checksums.txt` SHA-256 manifest; assert post-rewrite SHA-256s match pre-rewrite (proves byte-equivalence on NVIDIA across all three rewrite sites).
8. File `streamlz/issues/cuda-deterministic-hash-store` referencing the commit.

**Acceptance criteria:**
- `src/encode/lz_greedy_parser.cuh` compiles cleanly under existing nvcc.
- All existing CUDA encode goldens at L1..L5 on the 14 silesia files reproduce byte-identical `.slz` output (zero diffs) — proves rewrites A, B, and #2 are all NVIDIA-byte-equivalent. **Rip-cord per §6 if any golden diverges.**
- `zig build && zig build test` passes with no regressions on `src/encode/gpu_roundtrip_tests.zig`.
- All three CUDA rewrite sites contain explicit highest-lane-winner / subgroupShuffle-equivalent pattern with a `// VULKAN_PORTABILITY` comment naming the prerequisite.
- `sizeof(slzCompressOpts_t)` is 32 bytes (verified with `_Static_assert` in C and Zig `comptime` assert).
- `src/encode/cuda_ffi.zig` comptime asserts reference `effective_level_out` (not `reserved`); compilation passes.
- CUDA test exercises `slzCompressAsync` at level 3 and asserts `opts.effective_level_out == 3` on return.
- CUDA test exercises `slzCompressAsync` and asserts compressed_size is NOT written before return (a sentinel value pre-set by the test remains unchanged); then drain via `slzGetLastTimings` returns the correct compressed_size.
- `tests/golden_slz/` contains 70 fixtures + `checksums.txt` manifest, and the exact 14-file silesia subset is pinned in `tests/golden_slz/MANIFEST.txt`. Total committed size budget: ≤ 50 MiB (typical for these 14 files × L1..L5 at silesia compression ratios — the draft's "a few MiB" was wrong).
- `include/streamlz_gpu.h` docstring above `slzCompressAsync` documents the drain-only contract; enum has `SLZ_ERROR_DEVICE_LOST = 8` and `SLZ_ERROR_VK_FEATURE_MISSING = 9`.
- CHANGELOG entry documents: (a) the ABI semantic change (reserved[0] reused, `reserved` callers must migrate); (b) the compressed_size contract change with caller-migration steps; (c) minor version bump to 3.0.0; (d) new status enum values; (e) the louder "reserved[0] no longer guaranteed zero" migration note that any CUDA caller reading the field will see undefined-from-their-POV values.
- Issue `streamlz/issues/cuda-deterministic-hash-store` filed.

**Estimated LOC:** 500

**Risks addressed:** R3 (precondition for portable byte-identity), R12 (clamp-aware ABI prereq), site #2 semantic-mismatch debug branch eliminated (one-sided rewrite hazard closed)

**CUDA-side changes:** `src/encode/lz_greedy_parser.cuh` rewritten at all three sites; `include/streamlz_gpu.h` slzCompressOpts_t rename + status enum extension + docstring; `src/encode/cuda_ffi.zig` comptime asserts renamed; CUDA `slzCompressAsync` defers compressed_size to drain; PTX regenerated and committed.

**Notes:** This milestone is the root of the byte-identity chain — if it fails (rewrite NOT byte-equivalent on NVIDIA), the entire Vulkan port stalls until the rip-cord per §6 fires.

---

### M1 — src_vulkan/ scaffolding + build.zig vk/vklib/test_vk steps + 51 SPV blob placeholders + tools/check_workgroup_sizes.py + Android cross-compile + portable pipeline-cache path

**Depends on:** (none — parallel to M0)

**Scope:** Create the `src_vulkan/` tree exactly per architecture §2 (cli/, shaders/{common,encode,decode}, spv/{tier1,tier1_nv,tier2}/{encode,decode}, host/{encode,decode}, tests/, include/streamlz_vk.h, version.zig). Add four new `build.zig` steps: `vk` (Windows + Linux streamlz_vk.exe / streamlz_vk), `vklib` (streamlz_vk.dll / .so), `vk_android` (arm64-v8a libstreamlz_vk.so via NDK), `test_vk` (Zig tests under src_vulkan/tests/). Default `zig build` stays CUDA-only.

Create `tools/build_glsl.bat` that asserts a parseable glslc version line via the regex `^shaderc v(\d+)\.(\d+)` AND `glslang ([\d.]+)`, requires shaderc ≥ 2024.0 (corresponds to glslc ≥ 1.3.250 per the official Khronos shaderc → glslc version map; pin the SHADERC version not the SPIR-V env version, which the draft conflated). Iterate 17 known kernel basenames, emit 51 `.spv` blobs to `src_vulkan/spv/{tier1,tier1_nv,tier2}/{encode,decode}/`. Create 17 panic-shell `.comp` files (4-line `#version 460` + `#include "tier_gates.glsl"` + write `0xDEADBEEF` sentinel) so 51 SPVs compile end-to-end. Commit all 51 SPVs to git (arch locked decision #11). version.zig exports `"3.0.0-vk"`.

**Create `tools/check_workgroup_sizes.py`** (arch §3.1 mandate — fixes Reviewer 1's prior workgroup-size fatal). The script parses `__launch_bounds__(N, ...)` from `src/encode/*.cu`, `src/encode/*.cuh`, `src/decode/*.cu`, `src/decode/*.cuh` and asserts each kernel's `local_size_x` in `src_vulkan/shaders/**/*.comp` matches the corresponding entry in arch §3.1's table. CI gate: any PR that disagrees fails the lane.

**Android cross-compile**: `build.zig`'s `vk_android` step uses NDK r26 (or current LTS) targeting `arm64-v8a` API level 28+. Validates a fresh build of streamlz_vk loads + dispatches hello-world on a real Pixel via adb (deferred to M29 for the end-to-end test, but the build step is wired here).

**Portable pipeline-cache path**: `vk_pipeline.zig` declares `getCacheDir()` returning the platform-appropriate path: `%LOCALAPPDATA%\streamlz_vk\` on Windows, `${XDG_CACHE_HOME:-$HOME/.cache}/streamlz_vk/` on Linux, `/data/data/com.streamlz.vk/cache/` on Android. (Architecture §17 already documents the per-OS scheme; M1 lands the helper.)

**Acceptance criteria:**
- `Get-ChildItem src_vulkan/ -Recurse -Directory` matches architecture §2 exactly.
- Default `zig build` (no args) still builds `streamlz.exe` + `streamlz_gpu.dll` and runs zero Vulkan code.
- `zig build vk` produces `streamlz_vk.exe` (Windows) and `streamlz_vk` (Linux); running with no args prints `streamlz_vk 3.0.0-vk` and exits 0.
- `zig build vklib` produces `streamlz_vk.dll` (Windows) and `streamlz_vk.so` (Linux).
- `zig build vk_android` produces `libstreamlz_vk.so` for arm64-v8a; the .so is a valid ELF for ARM64 (verified via `file` output).
- `zig build test_vk` succeeds on an empty test suite.
- `tools/build_glsl.bat` aborts with a clear error if shaderc version < 2024.0 (regex match on the documented output line).
- Running `tools/build_glsl.bat` from repo root produces exactly 51 `.spv` files at documented paths.
- Each `.spv` is a valid SPIR-V module per `spirv-val`.
- `src_vulkan/spv/README.md` documents regeneration command + shaderc version pin (clarifying that "Vulkan SDK" is for build-time compile; runtime needs a Vulkan ICD/driver only).
- `tools/check_workgroup_sizes.py` exits 0 when GLSL local_size_x matches CUDA `__launch_bounds__`; CI rejects PRs where any .comp local_size_x disagrees with its CUDA counterpart per the §3.1 table.
- `vk_pipeline.zig::getCacheDir()` returns the platform-correct path on Windows, Linux, and Android (verified via unit tests in `tests/cache_dir_tests.zig`).

**Estimated LOC:** 1100

**Risks addressed:** shaderc/glslc < 2024.0 SPIRV-Tools BDA regression; workgroup-size regression (the Reviewer 1 prior fatal); substrate risk (Vulkan work never breaks CUDA build); Android-build-doesn't-exist hidden prereq for M29.

**CUDA-side changes:** None.

**Notes:** Parallel to M0. The workgroup-size CI guard prevents the next "lz_encode set to 64" regression from ever reaching review.

---

### M2 — Stub 16-symbol C ABI + 2 new status codes + CLI graceful failure + documented exit-code parity

**Depends on:** M1

**Scope:** Create `include/streamlz_vk.h` mirroring all 14 streamlz_gpu.h symbols with `_vk` suffix PLUS the Tier-2 buffer-registration pair `slzRegisterBuffer_vk` AND `slzUnregisterBuffer_vk` (16 total). The signatures match architecture §13 exactly:

```c
slzStatus_t slzRegisterBuffer_vk(slzHandle_vk_t h,
                                 void* vk_buffer_handle,          /* VkBuffer cast to void* */
                                 const void* d_base_address,      /* VkDeviceAddress as void* */
                                 size_t buffer_size);
slzStatus_t slzUnregisterBuffer_vk(slzHandle_vk_t h,
                                   const void* d_base_address);
```

The draft's `slzRegisterBuffer_vk(handle, void* ptr, size_t bytes, slzDeviceBuffer_t* out)` shape with an opaque `slzDeviceBuffer_t` output was fabricated and contradicted architecture §13; this milestone lands the §13 shape.

Create `src_vulkan/streamlz_vk.zig` exporting all 16 symbols as Zig stubs:
- `slzVersionString_vk` → `"3.0.0-vk"`
- `slzStatusString_vk` → static strings table covering ALL `slzStatus_t` values including the new `SLZ_ERROR_DEVICE_LOST` and `SLZ_ERROR_VK_FEATURE_MISSING`
- `slzCompressDefaultOpts_vk`, `slzDecompressDefaultOpts_vk` → zero structs (with `effective_level_out = 0`)
- `slzRegisterBuffer_vk` → returns `SLZ_SUCCESS` (no-op stub; explicit notes per §11.minor concerns about partial-state behavior between M2 and M6)
- `slzUnregisterBuffer_vk` → returns `SLZ_SUCCESS` (no-op stub)
- All other 10 symbols → return `SLZ_ERROR_UNSUPPORTED` with a `slzStatusString_vk`-resolved message like `"Vulkan backend slzCompressAsync_vk not yet implemented (waiting on M27)"`.

Create `src_vulkan/cli/main_vk.zig` as a clone of `src/main.zig` that dispatches to `_vk` symbols (CLI flags identical). When a user runs `streamlz_vk.exe compress in.bin out.slz`, the binary prints `streamlz_vk: error: compress operation requires VK encode kernels (waiting on M27); use streamlz.exe for production compression` to stderr and exits with the same exit code CUDA `streamlz.exe` uses for unsupported operations. **Exit-code parity audit**: M2 verifies via reading `src/cli.zig` that CUDA streamlz.exe uses exit code 2 for unsupported operations (the audited value, not a guess), and mirrors it exactly.

**Acceptance criteria:**
- `streamlz_vk.dll` exports exactly 16 symbols verifiable via `dumpbin /exports` matching the header exactly.
- Symbol names: `slzCreate_vk`, `slzDestroy_vk`, `slzCompressAsync_vk`, `slzDecompressAsync_vk`, `slzCompressHost_vk`, `slzDecompressHost_vk`, `slzGetDecompressedSize_vk`, `slzCompressBound_vk`, `slzGetLastTimings_vk`, `slzWaitAndGetLastTimings_vk`, `slzStatusString_vk`, `slzVersionString_vk`, `slzCompressDefaultOpts_vk`, `slzDecompressDefaultOpts_vk`, `slzRegisterBuffer_vk`, `slzUnregisterBuffer_vk`.
- `streamlz_vk.exe --version` prints `3.0.0-vk`.
- `streamlz_vk.exe compress test.txt out.slz` exits with the documented unsupported-operation exit code (audited from CUDA `streamlz.exe`) and prints a human-readable message naming the operation, NOT a crash.
- `streamlz_vk.exe decompress in.slz out.bin` same behavior.
- All 10 not-yet-implemented symbols return `SLZ_ERROR_UNSUPPORTED` from a C smoke test.
- `slzRegisterBuffer_vk` returns `SLZ_SUCCESS` as a no-op stub; `slzUnregisterBuffer_vk` returns `SLZ_SUCCESS`.
- `slzStatusString_vk(SLZ_ERROR_DEVICE_LOST)` returns the named string `"device lost"`; `slzStatusString_vk(SLZ_ERROR_VK_FEATURE_MISSING)` returns the named string `"required Vulkan feature missing"`. Both new codes have committed status strings.
- `sizeof(slzCompressOpts_vk)` matches `sizeof(slzCompressOpts_t)` byte-for-byte (cross-backend ABI parity).
- **Documented partial-state contract**: between M2 and M6 on Tier-2, register/unregister are no-ops; the binary tolerates this because all consuming operations return `SLZ_ERROR_UNSUPPORTED` until at least M11. CHANGELOG note: callers that exercise register-then-unsupported-op see successful registration followed by a clean error, never a crash. Documented as the "graceful partial-port" property.
- **Mid-port behavior contract**: between M11 and M27, partial decode/encode kernel availability is exposed via per-operation status: any unimplemented-yet kernel returns `SLZ_ERROR_UNSUPPORTED` with the kernel NAME embedded in the status string; the binary never half-executes then errors mid-frame.

**Estimated LOC:** 800

**Risks addressed:** R4 (slzRegisterBuffer_vk + slzUnregisterBuffer_vk pair exist from day one with the architecture-correct signature); shipability (binary is graceful from milestone 2 onward — no crashes, no silent failures)

**CUDA-side changes:** None (the slzStatus_t additions land in M0 on the shared header).

**Notes:** This milestone is the shippability cornerstone. The 16-symbol count corrects the locked-constraint §0 #4 wording to match architecture §13.

---

### M3 — Cross-backend conformance harness + tier1_roundtrip + tier2_roundtrip + wave64_roundtrip suites + golden_slz/ + "X / 350 passing" dashboard

**Depends on:** M0, M2

**Scope:** Implement the four authoritative test files architecture §2 enumerates:

1. `src_vulkan/tests/cross_backend_tests.zig` — the formal 350-cell cross-backend gate framework. Wire to M0's 70 golden `.slz` fixtures. 350-cell matrix: 14 silesia × 5 levels × **4 directions** = {CUDA→CUDA sanity, CUDA→VK, VK→CUDA, VK→VK} = 280 + the **70 sanity baseline** = 350. The draft's "5 directions" (with a plan-invented "VK encode bytes == golden .slz" axis) is dropped; the architecture matrix is 4 directions and the 70 sanity baseline (CUDA→CUDA cells) is the always-live row.
2. `src_vulkan/tests/tier1_roundtrip_tests.zig` — under `SLZ_VK_FORCE_TIER=1`, runs the 14-case × L1..L5 matrix and asserts roundtrip equality. Independently runnable via `zig build test_vk -- --filter tier1_roundtrip`.
3. `src_vulkan/tests/tier2_roundtrip_tests.zig` — same, with `SLZ_VK_FORCE_TIER=2`. Independently runnable via `--filter tier2_roundtrip`.
4. `src_vulkan/tests/wave64_roundtrip_tests.zig` — under SwiftShader with `VK_SUBGROUP_SIZE=64` env override. Independently runnable via `--filter wave64_roundtrip`. (Architecture §18 lane B'.)

At M3 only the {CUDA→CUDA} sanity cells (70) run live; the other 280 cells are PENDING with `return error.SkipZigTest` and a `WAITING ON M{N}` log line. The harness prints a stable header on every run: `Cross-backend conformance: 70 / 350 cells passing — next milestone unblocks: M11 (walk_frame): 0 cells, M21 (Wave-1 gate): 70 cells, M27 (frame_assemble): 210 cells`.

Implements clamp-aware equality: reads `slzCompressOpts_t.effective_level_out` (from M0); if VK Tier-2 clamps L5→L3, compares against the L3 golden, not L5. For CUDA-encoded cells the clamp-aware check reduces to direct equality (CUDA writes effective_level_out = requested level). Documentation in `tests/cross_backend_tests.zig` header clarifies this.

Harness exits 0 when the expected-XFAIL count matches actual; exits non-zero if an unexpected XFAIL flips to PASS. Wired to CI lanes A (NVIDIA when available), E (lavapipe), B' (SwiftShader wave64).

**Conformance bookkeeping discipline**: the dashboard tracks two separate counters that never mix:
- `MATRIX_CELLS_PASSING / 350` — only counts the 350 cells defined by the architecture matrix. Stage-level assertions (from M3a) are NOT counted in this number; they are counted in a separate `STAGE_ASSERTIONS_PASSING / N` line.
- This eliminates the draft's "5 / 350 added for synthetic raw cases" inconsistency (M16) and the "140 / 350 doubles a baseline" miscount (M21).

**Acceptance criteria:**
- `zig build test_vk` runs all four files and reports `70 / 350 cells passing — Wave 1 baseline (CUDA↔CUDA sanity)` from `cross_backend_tests.zig`.
- `zig build test_vk -- --filter tier1_roundtrip` runs tier1_roundtrip_tests.zig in isolation.
- `zig build test_vk -- --filter tier2_roundtrip` runs tier2_roundtrip_tests.zig in isolation.
- `zig build test_vk -- --filter wave64_roundtrip` runs wave64_roundtrip_tests.zig in isolation.
- `tests/golden_slz/` contains exactly 70 committed `.slz` files (14 × L1..L5). Total repo size ≤ 50 MiB (the pinned 14-file silesia subset is documented in `tests/golden_slz/MANIFEST.txt`).
- Each golden `.slz` round-trips through the CUDA decoder cleanly (sanity baked into a `build_goldens.zig` regenerator).
- Clamp-aware equality logic exists: if `effective_level_out != requested_level`, comparison uses the clamped level on both sides.
- Harness has a printed "next milestone unblocks" table.
- Harness exit code is 0 when only expected-XFAIL failures occur; non-zero if an XFAIL cell unexpectedly PASSES.
- CI lane E (lavapipe) runs the harness even though all VK cells are PENDING.
- SHA-256 manifest in `tests/golden_slz/checksums.txt` matches M0's manifest exactly.
- **Two separate counters** in the dashboard: `MATRIX_CELLS_PASSING / 350` and `STAGE_ASSERTIONS_PASSING / N` — never mixed.

**Estimated LOC:** 1300

**Risks addressed:** R3 (BLOCKER — wire-format byte-identity regression caught at the kernel that introduces it, not after)

**CUDA-side changes:** None (consumes M0 fixtures).

**Notes:** The 4-file split-out (cross_backend + tier1 + tier2 + wave64) gives developers iteration speed on a single tier without paying cross-backend setup cost.

---

### M3a — CUDA-side per-kernel intermediate dump instrumentation + stage golden generation + stage-assertion harness extension

**Depends on:** M0, M3

**Scope:** The draft's per-kernel stage-assertion acceptance criteria at M11-M27 were untestable because no CUDA-side per-kernel intermediates existed. M3a closes this gap.

1. **CUDA-side instrumentation hooks**: extend the CUDA encoder (`src/encode/encode_context.cu`) and decoder (`src/decode/decode_context.cu`) with a `SLZ_CUDA_DUMP_INTERMEDIATES=path` env-driven path that, when set, dumps per-kernel output buffers to `<path>/<case>_L<level>_<kernel_name>.bin` after each kernel returns. Hooks added for all 11 decode kernels and 6 encode kernels (17 total) plus the synthetic prefix-sum-input kernel needed for M16's partial-chain test. Hooks are no-ops when the env var is unset (zero perf impact in production).
2. **Stage golden generation**: a new `tools/build_stage_goldens.bat` script runs CUDA encoder/decoder over the 14-file silesia × L1..L5 corpus with `SLZ_CUDA_DUMP_INTERMEDIATES` set, producing per-stage golden blobs at `tests/golden_stage/<case>_L<level>_<kernel_name>.bin`. Commits a SHA-256 manifest.
3. **Stage-assertion harness extension**: `cross_backend_tests.zig` gains a `stageAssert(kernel_name, vk_output, case, level)` helper that loads the corresponding `tests/golden_stage/...` blob and asserts byte equality. M11+ acceptance criteria reference this helper.
4. **Repo size budget**: per-stage goldens are sized worst-case ~30 MiB total across all 17 kernels × 70 fixtures (most intermediates are smaller than the input). If the budget is exceeded, the harness compresses goldens with zstd at commit time.
5. **Selected stage-only assertions for M11+**: not every kernel produces a useful stage golden (single-thread orchestration kernels write small descriptor tables; multi-thread kernels write large payloads). The `tools/build_stage_goldens.bat` script documents per-kernel which intermediates are committed and which are "live-compute" (re-derived on test runs from the CUDA reference).

**Acceptance criteria:**
- `SLZ_CUDA_DUMP_INTERMEDIATES=tests/golden_stage cuda_encoder` produces the expected per-kernel blobs for one fixture × one level (sanity).
- `tools/build_stage_goldens.bat` runs end-to-end and produces all 17 × 70 stage goldens (plus the synthetic prefix-sum kernel for M16).
- `tests/golden_stage/checksums.txt` SHA-256 manifest committed.
- `stageAssert(...)` helper in cross_backend_tests.zig loads and compares stage blobs; passes on a hand-crafted same-blob test, fails on a 1-byte-perturbed test.
- Total committed `tests/golden_stage/` size ≤ 30 MiB (uncompressed) or ≤ 15 MiB after zstd if the budget is exceeded.
- The dump hooks are off by default (no env var) and add zero overhead to CUDA production builds (verified via baseline benchmark).

**Estimated LOC:** 700

**Risks addressed:** All M11-M27 per-stage assertions become actually testable (closes Reviewer 2's stage-assertion-references-nonexistent-reference fatal).

**CUDA-side changes:** Encoder + decoder gain conditional dump hooks; production code path unchanged when env var absent.

**Notes:** This is a Reviewer-driven split-out. The draft's "M3 commits 70 .slz fixtures" only produced full-frame goldens; per-stage byte-identity gates need per-stage references.

---

### M4 — Vulkan loader + vendor probe + tier classification (M4 retires R2 classification; full R2 retirement requires M29) + process-wide mutex + multi-thread create/destroy stress + minStorageBufferOffsetAlignment + maxMemoryAllocationCount + VkResult→SLZ_ERROR_DEVICE_LOST mapping

**Depends on:** M2

**Scope:** Implement `src_vulkan/host/vk_loader.zig` (VkInstance + VkPhysicalDevice enumeration + VkDevice creation with single compute queue per arch §7 locked decision #5, optional validation gated by `SLZ_VK_VALIDATION=1`).

**Process-wide instance mutex (architecture §4 mandate)**: `vk_loader.zig::g_instance_mu` guards lazy VkInstance init and refcount mutation. Closes the TOCTOU between concurrent `slzCreate_vk` and `slzDestroy_vk`.

Implement `vk_errors.zig` (VkResult → slzStatus_t translation) per architecture §16. **Includes** `VK_ERROR_DEVICE_LOST → SLZ_ERROR_DEVICE_LOST` and `VK_ERROR_FEATURE_NOT_PRESENT → SLZ_ERROR_VK_FEATURE_MISSING` (using the codes M0/M2 added).

Implement `vk_probe.zig`: dedicated 18th `.comp` file at `src_vulkan/shaders/common/probe.comp` (test-only, NOT one of the 17 production kernels; the 51-blob commit count remains 17 × 3 — probe.comp is on-the-fly compiled at first slzCreate_vk and cached in the runtime pipeline cache, not committed). The probe writes `{gl_SubgroupSize, subgroupBroadcastFirst(gl_LocalInvocationID.x), shuffle_xor_test_result, shuffle_relative_test_result}` to a host-visible SSBO. vk_probe dispatches the probe at `slzCreate_vk` and applies tier-classification: if subgroupSize == 32 AND subgroup ops behave AND vendor is in whitelist (NVIDIA Turing+, AMD RDNA2+, Intel Arc Alchemist+) → Tier-1, else Tier-2. `SLZ_VK_FORCE_TIER` env override. Mali (all gens) hardcoded to Tier-2 regardless of probe result.

If any Tier-1 whitelisted vendor fails the probe at runtime, return `SLZ_ERROR_VK_FEATURE_MISSING` (NOT the fabricated `SLZ_ERROR_UNSUPPORTED_DEVICE` the draft used). File arch-revision blocker per §6 rip-cord before M11.

Pre-quantize sc_group_size host-side (0.25 → 65536, 0.5 → 131072) so f32 never crosses the SPIR-V boundary (R10 mitigation).

**Runtime limit probes (arch §5)**: query `physicalDeviceLimits.minStorageBufferOffsetAlignment` and store on the handle; reject devices reporting `maxMemoryAllocationCount < 64` with `SLZ_ERROR_VK_FEATURE_MISSING`.

**Real-mobile classification**: build a CI matrix that runs the probe on lane E (lavapipe), lane B' (SwiftShader wave64), AND at least one borrowed real mobile device (Adreno or Mali via adb-attached CI worker if available; logged-only otherwise). M4 retires R2's **classification** portion only — the **emulation-correctness** portion of R2 needs real mobile kernel runs and is only retired at M29.

**Multi-threaded create/destroy stress test**: spawn N=16 threads, each calling slzCreate_vk + slzDestroy_vk in a tight loop for 10 seconds. Validation layer must report zero instance leaks; no crashes. This exercises the g_instance_mu.

**Acceptance criteria:**
- `slzCreate_vk` returns `SLZ_SUCCESS` on any system with Vulkan 1.2+ and a compute queue with `timestampValidBits > 0`.
- `slzCreate_vk` returns `SLZ_ERROR_VK_FEATURE_MISSING` (the named architecture status) with explanatory log when no compute queue, Vulkan 1.2 floor not met, `maxMemoryAllocationCount < 64`, or `maxPushConstantsSize < 128`.
- Probe kernel compiles and runs on lavapipe; returns sensible `gl_SubgroupSize`.
- Tier classification produces correct result for: NVIDIA RTX (Tier-1), AMD RDNA2 reporting 32 (Tier-1), AMD RDNA2 reporting 64 (Tier-2), Mali-G78 (Tier-2 always), Adreno 740 (Tier-2 unless probe reports 32), lavapipe (Tier-2), SwiftShader (Tier-2).
- `SLZ_VK_FORCE_TIER=1` and `=2` env overrides work end-to-end (verified by probe logging chosen tier).
- `SLZ_VK_FORCE_TIER=1` on a Tier-2-only device returns `SLZ_ERROR_VK_FEATURE_MISSING` with `"forced Tier-1 unavailable: missing subgroup_size_control"` message (no silent downgrade).
- Probe results from at least one real mobile borrow recorded in `docs/vulkan_port_architecture.md` appendix; any Tier-1 whitelist violation fires §6 rip-cord BEFORE M11.
- sc_group_size pre-quantization: host-side integer products 65536 / 131072 used; M14 acceptance (NOT M4 acceptance — moved per Reviewer 2 minor) verifies no f32 in any production SPV.
- `slzDestroy_vk` reports zero leaks under `SLZ_VK_VALIDATION=1`.
- **Process-wide mutex test**: 16 threads × 10 seconds of create/destroy thrash produces zero validation-layer errors AND zero instance leaks AND no crashes (architecture §4 mandate).
- `physicalDeviceLimits.minStorageBufferOffsetAlignment` recorded on the handle (consumed by M7).
- `vk_errors.zig` maps `VK_ERROR_DEVICE_LOST` to `SLZ_ERROR_DEVICE_LOST` (the M0/M2 enum value).

**Estimated LOC:** 1300

**Risks addressed:** R2 (classification portion — emulation-correctness portion deferred to M29), R10 (sc_group_size float→uint determinism), TOCTOU concurrency hazard at instance level.

**CUDA-side changes:** None.

**Notes:** R2 retirement is split: classification at M4, emulation correctness at M29. The plan no longer overclaims M4 retires R2 in full.

---

### M8a — Minimal dispatch + timing primitives (extracted from M8 to unblock M5)

**Depends on:** M4

**Scope:** Build the bare minimum dispatch + timing substrate that M5 needs to produce ns/dispatch numbers, without bundling the production-grade pipeline cache, descriptor factory, sync wrapper, or shape-keyed cmd cache (those land in M8b/M8c).

- `vk_pipeline.zig` minimal: per-kernel `VkPipeline` create from a passed-in `VkShaderModule` and `VkPipelineLayout`. No persistent disk cache, no invalidate hook, no Tier-2 worst-case sizing — just enough to dispatch one shader. Files M8b for the production rewrite.
- `vk_command.zig` minimal: record + submit + wait. No reset-pool, no shape-keyed cache — just allocate one VkCommandBuffer per dispatch and free it.
- `vk_timing.zig` minimal: VkQueryPool TIMESTAMP sized 2 slots (one dispatch, one before, one after); `beginKernel`/`endKernel` write timestamps; `drainTimings()` reads back + applies `timestampPeriod` + returns nanoseconds. Per architecture §15 the production sizing is 68 slots; the minimal version is just 2. Uses `vkCmdWriteTimestamp` (Vulkan 1.0 core) for portability — sync2's `vkCmdWriteTimestamp2` is added in M8c.
- Bench helper `vk_microbench.zig`: takes a SPIR-V blob + push-constant struct + workgroup count, returns mean/median/p95 ns/dispatch over N=1000 iterations.

**Acceptance criteria:**
- Dispatching a no-op compute pipeline (loaded from a hand-crafted `nop.spv` blob) returns a sensible delta > 0 ns and < 100 ms via `drainTimings()`.
- The bench helper produces stable ns/dispatch numbers (CV < 10%) across 1000 iterations.
- No persistent cache file is created.
- No descriptor-set support beyond push-constant-only kernels.
- No sync2; uses `vkCmdWriteTimestamp` only.
- M5 consumes this milestone with no additional substrate.

**Estimated LOC:** 500

**Risks addressed:** Closes the circular dependency between M5 (needs dispatch+timing) and M8 (was bundled as M8a+M8b+M8c in the draft and placed after M5).

**CUDA-side changes:** None.

**Notes:** This is a structural split-out. The production-grade chassis is M8b/M8c.

---

### M5 — `__match_any_sync` emulation standalone microbenchmark on AMD/Intel (RETIRES R1 hard — UNTESTED-pass is DISALLOWED)

**Depends on:** M4, M8a

**Scope:** Port BOTH match_any sites as standalone `.comp` shaders BEFORE integrating into any production kernel. Site #3 (11-iter ballot loop, domain [1..11], used in huff_build_lut): standalone shader takes a synthetic Huffman-shaped input (256-symbol histograms × N batches). Site #1 (32-broadcast OR-reduction, used in lz_encode hash store): standalone shader takes a synthetic encode-shaped workload (32-lane hash buckets × N batches).

Implement BOTH the Tier-1 base variant (subgroup_basic + ballot + vote) AND the tier1_nv variant (subgroupPartitionedXorNV fast path).

**Equivalence proof for sites #1 + #2 + #3 vs CUDA reference**: a randomized cross-language test that generates 10,000 synthetic (mask, key4-vector) inputs satisfying the bucket-coherence invariant from arch Appendix A AND a representative random sample of >10^6 cases for site #1's all-32-lane key distributions, asserting CUDA result bits and GLSL rewrite agree on the highest-lower-set-bit-index for every input. **The CUDA reference is used directly, not a Zig-side reference** (per architecture §3.4 mandate). This closes Reviewer 1's "byte_perm uses Zig reference not CUDA" minor + Reviewer 2's site #2 semantic-mismatch fatal.

Build a microbenchmark host harness that uses M8a to run each kernel on every available CI lane and measures throughput in ns/dispatch.

**Pre-committed numeric threshold: AMD throughput ≥ 65% of CUDA reference AND Intel throughput ≥ 65% of CUDA reference for BOTH sites. UNTESTED-PASS IS DISALLOWED.** If hardware acquisition slips, the rip-cord per §6 fires: either (a) acquire real hardware before M11 starts, or (b) explicitly defer R1 retirement and downgrade M22a/M22b acceptance to NVIDIA-only with AMD/Intel marked `pending real hardware`. The draft's "lane is recorded as UNTESTED and a blocker issue is filed" pass path is removed.

**Acceptance criteria:**
- Site #3 standalone kernel produces byte-identical "match groups" output to a CUDA reference on 100 synthetic Huffman inputs.
- Site #1 standalone kernel produces byte-identical highest-lane-winner election to a CUDA reference on >10^6 representative random cases.
- Site #2 equivalence test: 10,000 cases satisfying bucket-coherence pass + exhaustive proof per arch Appendix A.
- NV_partitioned variants for both sites produce byte-identical output to base variants on NVIDIA.
- Microbenchmark reports ns/dispatch on lane A (NVIDIA), lane B (AMD), lane C (Intel Arc). **If lane B or lane C hardware is unavailable, the rip-cord per §6 fires** — NO UNTESTED-PASS.
- AMD throughput vs CUDA: ≥ 65% on BOTH sites — if missed, fire rip-cord per §6.
- Intel throughput vs CUDA: ≥ 65% on BOTH sites — if missed, fire rip-cord per §6.
- Per-vendor perf numbers committed to `tests/perf/match_any_baseline.json` with hardware ID (vendor + device + driver version + shaderc version).
- Decision documented in `docs/vulkan_port_architecture.md` §7 with measured numbers per vendor per site.

**Estimated LOC:** 1000

**Risks addressed:** R1 (match_any emulation perf regression on AMD/Intel — caught at milestone 5 with pre-committed numeric threshold and NO UNTESTED-PASS fallback).

**CUDA-side changes:** Uses M0's rewritten CUDA encoder (sites A, B, #2 all explicit-pattern) as the reference behavior.

**Notes:** The CUDA reference is used directly (architecture §3.4 mandate). The byte_perm unit test in M9 also uses the CUDA reference, not a Zig reference, for the same reason.

---

### M6 — BDA + shaderInt64 feature probe + slzRegisterBuffer_vk + slzUnregisterBuffer_vk real implementation (RETIRES R4)

**Depends on:** M4, M8a

**Scope:** Promote `slzRegisterBuffer_vk` AND `slzUnregisterBuffer_vk` from the M2 no-op stubs to their real implementations:

- **Tier-1**: both functions are no-ops (return `SLZ_SUCCESS`).
- **Tier-2**: `slzRegisterBuffer_vk` registers the (vk_buffer_handle, d_base_address, buffer_size) triple in a host-side hashmap keyed by `d_base_address` (the opaque pointer callers pass into all other `_vk` APIs). `slzUnregisterBuffer_vk(handle, d_base_address)` removes the entry and prunes any reverse-lookup state.
- Per architecture §13 lifetime contract: registrations persist until `slzUnregisterBuffer_vk` or `slzDestroy_vk`. Once unregistered, any subsequent `_vk` call referencing that `d_base_address` returns `SLZ_ERROR_INVALID_ARG`.

Add a buffer_reference round-trip test: small `.comp` shader that takes a `uint64_t` BDA via push constant, dereferences it, writes a sentinel value back. Run on every Tier-1 lane (NVIDIA, AMD, Intel) to prove BDA + shaderInt64 work end-to-end. Build a Tier-2 round-trip test that uses descriptor-set bindings instead of BDA, dispatched against the same logical input via the registered handle. shaderInt64 enabled at VkDevice creation when Tier-1; falls back to uvec2 emulation typedef in `tier_gates.glsl` when Tier-2.

**glslc version assertion is owned by M1** (not duplicated here — the draft duplicated the assertion in M1 AND M6); M6 only runs the test that downgrading shaderc < 2024.0 would fail, by way of CI's normal build path.

**Acceptance criteria:**
- BDA round-trip kernel reads sentinel via `buffer_reference` push constant on NVIDIA Tier-1 lane (host-side memcmp passes).
- Same on AMD RDNA2+ Tier-1 lane (when subgroupSize=32 confirmed).
- Same on Intel Arc Alchemist+ Tier-1 lane.
- `slzRegisterBuffer_vk` on Tier-1 returns SLZ_SUCCESS as a no-op, validated via `SLZ_VK_FORCE_TIER=1` on a Tier-1 lane.
- `slzRegisterBuffer_vk` on Tier-2 stores (vk_buffer_handle, d_base_address, buffer_size); descriptor-set round-trip kernel reads back the sentinel via the registered handle.
- `slzUnregisterBuffer_vk` on Tier-2 removes the entry; subsequent `slzCompressAsync_vk` referencing the now-unregistered `d_base_address` returns `SLZ_ERROR_INVALID_ARG` (architecture §13 lifetime contract).
- `slzUnregisterBuffer_vk` on Tier-1 returns SLZ_SUCCESS as a no-op (no state to remove).
- `shaderInt64` enabled at VkDevice creation when Tier-1; uvec2 emulation typedef active when Tier-2 (verified by SPIR-V cross-reflect).
- Per-handle reverse-lookup pruning verified by a leak-detection test: 1000 register/unregister cycles produce zero state retention.

**Estimated LOC:** 900

**Risks addressed:** R4 (BDA absent breaks void* device-pointer ABI — register/unregister pair real and validated on both tiers).

**CUDA-side changes:** None.

**Notes:** Parallel to M7 (both depend on M4 + M8a).

---

### M7 — Memory allocator with 2-GiB VkBuffer split + Adreno minStorageBufferOffsetAlignment runtime probe + maxMemoryAllocationCount enforcement (RETIRES R5, R6)

**Depends on:** M6

**Scope:** Implement `src_vulkan/host/vk_memory.zig` as a slab-based suballocator (no VMA per arch §0). Three pools: DEVICE_LOCAL (encode/decode scratch + I/O), HOST_VISIBLE|HOST_COHERENT (staging + readback), and HOST_CACHED|HOST_VISIBLE if available (R6 fast path; falls back to coherent + staging copy when missing). Slab size: 64 MiB.

**Storage-buffer-offset alignment**: at allocation time, the suballocator aligns SSBO sub-allocations to `max(M4-recorded minStorageBufferOffsetAlignment, 256)`. Architecture §5 mandate addresses Reviewer 1's note that the draft's 256-B hardcode is wrong on Adreno 5xx (4096 B) and embedded drivers (64 KiB). nonCoherentAtomSize alignment is for host-visible mapped memory and is a separate axis.

**maxMemoryAllocationCount enforcement**: architecture §5 — already checked in M4 at `slzCreate_vk` (devices < 64 rejected with `SLZ_ERROR_VK_FEATURE_MISSING`). M7 confirms the suballocator never exceeds the limit at runtime (slab pool minimizes per-handle allocations).

Free-list per slab, no per-allocation `vkFreeMemory`. Expose `alloc(size, usage, memory_property_hints) → {VkBuffer, VkDeviceMemory, VkDeviceAddress (if BDA), host_ptr (if visible)}`.

Stand up the encode hash-table buffer-split logic: compute the maximum hash table size for L5 on a 24GB input (~3.5 GiB); implement bin-packing so no single sub-chunk's hash crosses a buffer boundary; split across N ≤2 GiB VkBuffers. Build a synthetic stress test that allocates N=3 hash buffers totalling >5 GiB on a Tier-1 device (skipped if device VRAM <8 GB), writes a sentinel pattern via a synthetic kernel that BDA-addresses across buffer boundaries via push-constant table, reads back, validates.

**Tier-2 descriptor layout work moves to M22b** where it's consumed in production; M7 produces the bin-packing primitive and the synthetic test only.

Document the split math in `docs/vulkan_port_architecture.md` §6 with worked example for 24 GB input at L5.

**Acceptance criteria:**
- `vk_memory.allocDevice(size)` and `allocHost(size)` return distinct ranges that never alias (verified by guard pattern).
- Round-trip test: HOST_VISIBLE → DEVICE_LOCAL via `vkCmdCopyBuffer` → HOST_VISIBLE, byte-equal to original.
- HOST_CACHED fallback path on a device without HOST_CACHED|HOST_VISIBLE: allocator uses HOST_VISIBLE|HOST_COHERENT and reports `cached_readback=false` in a status struct.
- 10k alloc/free cycles complete without crash; fragmentation metric reported.
- SSBO sub-allocation alignment uses `max(physicalDeviceLimits.minStorageBufferOffsetAlignment, 256)` recorded by M4; unit test on a simulated Adreno-5xx-like device (alignment=4096) confirms sub-allocations are 4096-byte aligned.
- Bin-packing function: given (input_size, level), returns list of (sub_chunk_id, buffer_id, offset) tuples that satisfy ≤2 GiB per VkBuffer and no sub-chunk straddles a boundary.
- Synthetic kernel allocates 3 × 2 GiB buffers on a 24 GB-VRAM device, addresses all 3 via BDA push-constant table, writes pattern, host readback validates pattern.
- On devices with VRAM < 8 GB: test is skipped with `INFO: 2-GiB stress test skipped on <8GB device` log, NOT failing CI.
- Split math documented in `docs/vulkan_port_architecture.md` §6.
- Zero validation-layer leaks under `SLZ_VK_VALIDATION=1`.

**Estimated LOC:** 1100

**Risks addressed:** R5 (encode hash table >2 GiB single VkBuffer — bin-packing validated on synthetic load), R6 (HOST_CACHED fallback), minStorageBufferOffsetAlignment hazard on Adreno/embedded.

**CUDA-side changes:** None.

**Notes:** Tier-2 descriptor layout sizing at `slzCreate_vk` is M22b territory because it depends on the encode-kernel hash-size-at-create decision; M7 produces the primitive only.

---

### M8b — Production pipeline cache + descriptor factory + invalidate_all_cached_cbs hook + cache-corruption fuzz

**Depends on:** M5, M7

**Scope:** Promote M8a's minimal pipeline.zig + command.zig to the production-grade versions.

- **Process-wide `VkPipelineCache`** per `(deviceUUID, tier)` persisted to the M1-defined platform-portable path (`%LOCALAPPDATA%\streamlz_vk\pipelines_<UUID>_t<tier>.bin` on Windows, `$XDG_CACHE_HOME/streamlz_vk/pipelines_<UUID>_t<tier>.bin` on Linux, `/data/data/com.streamlz.vk/cache/pipelines_<UUID>_t<tier>.bin` on Android) per architecture §7/§17.
- **Per-kernel pipeline lazy creation** on first dispatch; uses the 17 production kernels' SPVs from M1 (probe.comp is separate per M4).
- **Tier-1 pipelines** set `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` with `requiredSubgroupSize=32` and the FullSubgroups bit.
- **Descriptor-set-layout factory**: Tier-2 builds worst-case layout sized at handle create (uses M7's bin-pack metadata + the M22b hash-size-at-create decision).
- **`invalidate_all_cached_cbs` hook** in `vk_command.zig`: every pipeline destroy in `vk_pipeline.zig` calls `vk_command.invalidate_all_cached_cbs()` to flush the shape-keyed cmd buffer cache (which is built in M28). The hook is wired here even though the cache itself is built later; M28's acceptance verifies the hook is invoked. Architecture §7 mandate.
- **Cache-corruption fuzz**: stronger than the draft's "delete + truncate-to-17B" tests. Fuzz cases: delete the file, truncate to {0, 17, 128, 4096} bytes, bit-flip random bytes in {magic, header, body}, replace with random data of the same size. All cases must produce a cold rebuild without crash.

**Acceptance criteria:**
- Cold start (no cache file): all 17 production pipelines compile + cache file written in <500 ms on release build.
- Warm start (cache present): all 17 pipelines load in <10 ms (target 3-5 ms).
- Cache file naming exactly matches arch §17 per-platform schema.
- Different deviceUUID generates a separate cache file.
- Tier-1 pipelines stamp `requiredSubgroupSize=32` (verified by `SLZ_VK_VALIDATION=1` info messages).
- `invalidate_all_cached_cbs` hook in `vk_command.zig` is called from every pipeline destroy in `vk_pipeline.zig` (verified by a unit test that mocks the destroy path and confirms the hook fires before VkPipeline destruction).
- Cache file is corruption-tolerant: deleting → cold rebuild, no crash; truncating to {0, 17, 128, 4096} bytes → cold rebuild, no crash; bit-flips in header/body → cold rebuild, no crash; replacing with random data of the same size → cold rebuild, no crash.
- Each empty `.comp` dispatch returns `SLZ_ERROR_UNSUPPORTED` with the kernel NAME embedded in `slzStatusString_vk` (NOT a sentinel write, NOT a crash).
- Tier-2 descriptor-set layout supports N=8 buffers worst-case (mobile clamp ceiling), validated by binding 8 buffers and dispatching a hello-world kernel that reads from binding 7.

**Estimated LOC:** 1100

**Risks addressed:** R7 (VkPipelineCache cold-start 100-500ms), dangling-cmd-buffer hazard (architecture §7).

**CUDA-side changes:** None.

**Notes:** The invalidate_all_cached_cbs hook is wired here; the shape-keyed cmd buffer cache it protects is built in M28.

---

### M8c — Sync2/sync1 wrapper + production VkQueryPool (68 slots per arch §15) + per-tier query indexing

**Depends on:** M8b

**Scope:** Implement `vk_sync.zig`: Tier-1 uses VkTimelineSemaphore + `vkCmdPipelineBarrier2` (synchronization2); Tier-2 uses VkBinarySemaphore + VkFence + `vkCmdPipelineBarrier` (sync1 fallback) when sync2 is probed-absent at device-create.

Implement `vk_timing.zig` production version per architecture §15: **VkQueryPool TIMESTAMP sized exactly 2 × 17 × in_flight_count = 68 slots** (in_flight_count = 2). Drain reads slot index `2 * kernel_idx + 0..1 + 34 * (current_buffer_idx)` per arch §15. Architecture §15 indexing is the literal implementation — the draft's vague "2×N" sizing is corrected.

`beginKernel`/`endKernel` write TOP_OF_PIPE/BOTTOM_OF_PIPE timestamps (locked decision #12); `drainTimings()` reads back + applies `timestampValidBits`-aware post-subtraction mask (arch §15) + `timestampPeriod` + fills caller's `slzKernelTiming_t[]`. Tier-1 uses `vkCmdWriteTimestamp2`; Tier-2 uses `vkCmdWriteTimestamp` (Vulkan 1.0 core) when sync2 absent. VK_EXT_host_query_reset (Vulkan 1.2 core) or `vkCmdResetQueryPool` fallback.

`compressed_size` / `decompressed_size` pseudo-kernel slot (M0 architecture §13/§14): the timing drain populates one slot per kernel + one pseudo-kernel entry for compressed_size. Both backends speak this format.

**Acceptance criteria:**
- Timeline-semaphore path: submit two empty cmd buffers signalling values 1 then 2, wait on value 2 — both fences cleared.
- Binary-sem+fence path forced via env override: same test, identical externally-observable behavior.
- VkQueryPool sized exactly 68 slots per arch §15; index expression `2 * kernel_idx + 0..1 + 34 * current_buffer_idx` validated by a unit test that writes synthetic timestamps and reads them back at known indices.
- `vkCmdWriteTimestamp2` path active when synchronization2 enabled; `vkCmdWriteTimestamp` path active when not.
- `timestampValidBits` post-subtraction mask handles 36-bit wraparound correctly (unit-tested with synthetic 36-bit values that include a wrap).
- `compressed_size` pseudo-kernel slot returns the value from `slzAssembleMeasureKernel` after GPU completion (validated via M25).
- Each empty `.comp` dispatch returns `SLZ_ERROR_UNSUPPORTED` with the kernel NAME embedded.

**Estimated LOC:** 700

**Risks addressed:** R8 (cross-stream barrier elision — single VkQueue + explicit per-buffer barriers via sync2 wrapper).

**CUDA-side changes:** None.

**Notes:** Closes Reviewer 3's prior fatal on query-pool index bugs by making the index expression and slot count literal architecture mandates.

---

### M9 — Common GLSL helpers (byteio, huffman, warp_t1, warp_t2, match_any, wire_format, bda, ballot_mask, byte_perm) + per-primitive unit tests vs CUDA reference + layout(subgroup_size=32) SPV audit

**Depends on:** M8c

**Scope:** Write all shared GLSL include files per arch §2 common/:
- `byteio.glsl` — 8-bit SSBO fast path (gated on VK_KHR_8bit_storage) + shift/mask fallback on `uint data[]`.
- `huffman.glsl` — 32-stream BIL layout helpers (wire-locked).
- `warp_t1.glsl` — subgroup intrinsic lowerings.
- `warp_t2.glsl` — shared-mem workgroup-wide emulations (32 logical lanes, always uses workgroup `barrier()`).
- `warp.glsl` — selects via `#if TIER == 1`.
- `match_any.glsl` — site #1 + site #3 + NV_partitioned fast path for site #1. Site #2 ELIMINATED per arch §10 (replaced by `subgroupShuffle(key4, highest_lower) == key4`, byte-identical per arch Appendix A + M5 equivalence test).
- `wire_format.glsl` — constants matching `src/format/` (HUFF_NUM_STREAMS=32, magic 0x534C5A31, version 2, codec 1, log2_block_size, sc_group_size integer products 65536/131072).
- `bda.glsl` — `buffer_reference` declarations on T1, SSBO bindings on T2.
- `ballot_mask.glsl` — wave64-safe ballot helper (always returns 32-bit lower half).
- `byte_perm.glsl` — explicit `__byte_perm(w, 0, 0x0123)` lowering.
- `tier_gates.glsl` — TIER=1/2 macros, U64 typedef (uint64_t on T1, uvec2 on T2).

Unit-test each helper via a tiny `.comp` shader and writes results to a readback SSBO. Build `src_vulkan/tests/byteio_unit_tests.zig` exhaustively testing byteio primitives at every alignment 0..3 and length 1..256 on Tier-1 and Tier-2.

**CUDA-reference cross-checks**: `byte_perm.glsl` unit test cross-checks against actual CUDA `__byte_perm` output on a 1024-element random vector (architecture §3.4 mandate); `match_any` site checks use CUDA reference per M5. The draft's "Zig-side reference" is replaced with the CUDA reference everywhere it matters.

**Wire-format constants cross-language check**: a host-side test in `tests/wire_format_tests.zig` reads each committed SPV's SpecConstants via `spirv-dis`, extracts them, and asserts they match the Zig-side `format/wire.zig` constants. This is the "memcmp of a compile-time constants struct" Reviewer 2 asked for (the SpecConstant claim alone is not a runtime equality check).

**SPV audit**: after the helpers exist, `tools/audit_spv.py` runs over all 51 committed production SPVs and asserts (a) `OpTypeFloat` does not appear anywhere (R10 / N5 mitigation — moved from M14 to M9 since this is the helper-stable point) AND (b) every Tier-1 SPV contains `OpExecutionMode RequireFullSubgroupsKHR` AND a `LocalSizeId` whose `required_subgroup_size_id=32` AND a `SubgroupSize` execution mode == 32 (the in-shader `layout(subgroup_size=32) in;` declaration mandate). The audit runs in CI.

**Acceptance criteria:**
- `byteio_unit_tests.zig` round-trips every byteio primitive on Tier-1 and Tier-2 byte-for-byte against the CUDA `gpu_byteio.cuh`.
- `warp_test.comp` on Tier-1: subgroupBroadcast, subgroupExclusiveAdd, subgroupShuffleXor match expected values for input 1..32.
- Same `warp_test.comp` on Tier-2 produces identical results via shared-mem emulation.
- `match_any_unit_test.comp` site #3: returns same mask as CUDA reference across 1000 random inputs.
- `match_any_unit_test.comp` site #1: returns highest-indexed lane sharing bucket, matches CUDA reference (consistent with M5 standalone benchmark output).
- `byte_perm_unit_test`: 1024 random uint inputs through `__byte_perm(w, 0, 0x0123)` match CUDA reference (NOT Zig reference) — architecture §3.4 mandate.
- `ballot_mask` helper produces a 32-bit value even when run on a wave64 SwiftShader lane.
- Wire-format constants cross-check: `tests/wire_format_tests.zig` reads each committed SPV's SpecConstants and asserts equality with `format/wire.zig`.
- `tools/audit_spv.py` passes on all 51 production SPVs: zero `OpTypeFloat` matches; every Tier-1 SPV has `RequireFullSubgroupsKHR` execution mode AND `SubgroupSize == 32`.

**Estimated LOC:** 1500

**Risks addressed:** R1 (match_any emulations validated standalone — perf gate already retired in M5), R10 (wire constants byte-identity at GLSL header level), N5 (sc_group_size pre-quantization audited via SPV scan).

**CUDA-side changes:** None.

**Notes:** SPV audit moved from M14 to M9 (and from M4 to M9 for the OpTypeFloat scan) so the audit lives at the helper-stable point. N5 risk register entry updated accordingly.

---

### M10 — Hello-dispatch substrate gate (end-to-end smoke kernel + buffer round trip)

**Depends on:** M3a, M9

**Scope:** Add `src_vulkan/shaders/test/hello.comp` (separate test-only shader, NOT one of the 17 production kernels) that reads an SSBO of N uints and writes N+1 to each. Plumb `src_vulkan/tests/hello_dispatch_test.zig`: allocate HOST_VISIBLE input, write 1..N, allocate DEVICE_LOCAL output, copy via M7, dispatch via M8b/M8c, copy back, byte-check `output[i] == i+2`. Time it via M8c timing. Wave-64 lane via SwiftShader (`VK_SUBGROUP_SIZE=64` emulation) also runs. This is the proof that all substrate (M4-M9 + M3a) works end-to-end.

**Acceptance criteria:**
- Hello-dispatch test passes on Tier-1 device.
- Hello-dispatch test passes on Tier-2 device (forced via `SLZ_VK_FORCE_TIER=2`).
- M8c timing reports a finite, sensible duration (>0 and <100 ms for N=1M).
- Repeated invocations (1000×) do not leak memory, descriptor pools, or pipeline-cache slots (validation layer clean under `SLZ_VK_VALIDATION=1`).
- Wave-64 lane via SwiftShader passes.
- M3 conformance harness still reports `70 / 350 cells passing` (M3 baseline preserved; no new VK production kernels in this milestone so the count is correctly unchanged).

**Estimated LOC:** 300

**Risks addressed:** Substrate end-to-end proof — gates everything above before kernel porting starts.

**CUDA-side changes:** None.

**Notes:** —

---

### M11 — Wave 1 kernel 1/10: walk_frame.comp (D1) + decode_context_vk.zig + decode_dispatch_vk.zig — BDA/push-constant/dispatch chassis validation via production code path

**Depends on:** M3a, M10

**Scope:** Port `src/decode/walk_frame_kernel.cuh` to `src_vulkan/shaders/decode/walk_frame.comp`. Single-thread orchestration kernel — reads frame header via `byteio.glsl`, walks the chunk-descriptor table, writes per-chunk metadata to an output SSBO. `local_size_x=32` per arch §3.1 table (only thread 0 active). Tier-1 uses BDA via `bda.glsl`; Tier-2 uses descriptor-set bindings via `slzRegisterBuffer_vk`. Push constants carry: input_addr (or descriptor index), input_size, output_addr, max_chunks.

**Production code path delivery (architecture §2 mandate)**: M11 creates `src_vulkan/host/decode/decode_context_vk.zig` (decode handle + scratch alloc + barrier scheduling — mirrors `src/decode/decode_context.cu`) and `src_vulkan/host/decode/decode_dispatch_vk.zig` (orchestrates per-kernel dispatch). These are the production drivers consumed by `slzDecompressAsync_vk` — without them, the kernel is reachable only from the test harness and the binary still routes through stubs (which the draft permitted; this plan does not).

Add stage-level assertion to M3 conformance harness via M3a's `stageAssert(...)` helper that compares walk_frame intermediate output to CUDA byte-for-byte using the M3a stage goldens.

**M11 dependency correction**: the draft listed M11 as depending on `M0, M10`. M11 does NOT consume the M0 encode-side hash-store rewrite directly. The real M0 dep is M3 (which consumes M0's regenerated goldens) and M3a (which consumes M0 for the stage-instrumentation reference). M11 only needs M3a + M10. The dependency table in §3 reflects this.

**Acceptance criteria:**
- `walk_frame.comp` dispatch on Tier-1 (NVIDIA) executes without validation errors under `SLZ_VK_VALIDATION=1`.
- Same on Tier-2 (lavapipe).
- Push-constant block size ≤ 128 bytes verified at pipeline create.
- Stage-level conformance test via M3a stageAssert: walk_frame output byte-identical to CUDA on all 70 golden fixtures, both tiers.
- M3 harness: walk_frame stage cell flips to PASS on both tier rows; stage-assertions counter increments; matrix-cells counter unchanged.
- No descriptor-set bindings on Tier-1 (BDA-only); 1 SSBO binding on Tier-2.
- **Production code path verified**: `streamlz_vk.exe decompress` (which goes through `slzDecompressAsync_vk` → `decode_dispatch_vk.zig`) reaches walk_frame.comp via the production orchestration code, not the test harness. (The full decompress fails after walk_frame because lz_decode etc. aren't ported yet, but walk_frame is dispatched via the production path.)

**Estimated LOC:** 700

**Risks addressed:** R4 (BDA path real-world validated end-to-end with a production kernel via production orchestration).

**CUDA-side changes:** None (M0 already landed).

**Notes:** Per-kernel milestones M12-M27 each augment M3's stage-assertion counter via M3a's harness extension. This is accounted for in M3a's LOC.

---

### M12 — Wave 1 kernel 2/10: lz_decode_raw.comp — byteio validation

**Depends on:** M11

**Scope:** Port `src/decode/lz_decode_raw.cuh` to `src_vulkan/shaders/decode/lz_decode_raw.comp`. Per arch §3.1: Tier-1 `local_size_x=64` (LZ_KERNEL_BLOCK_THREADS=64, two warps) with per-subgroup `subgroupBarrier()`. Tier-2 `local_size_x=32` with 2× workgroup dispatch (mitigates R9). Reads raw-block descriptors and copies bytes from compressed input to decompressed output via `byteio.glsl`. No subgroup ops — proves byteio fast path and per-tier workgroup-size dispatch policy.

**Acceptance criteria:**
- Synthetic 16 KiB raw-only frame decodes byte-identical to CUDA's lz_decode_raw output.
- Tier-1: 64-thread workgroup confirmed (`gl_WorkGroupSize.x` written to debug SSBO).
- Tier-2: 32-thread workgroup confirmed, 2× workgroup dispatch verified by SPIR-V cross-reflect.
- Stage assertion via M3a stageAssert: lz_decode_raw output byte-identical to CUDA on all 70 fixtures both tiers.
- Throughput sanity floor: ≥ 50% of CUDA on the same device class (NUMERIC — was "sanity floor; no firm target" in the draft, now a concrete numeric pass/fail).
- `byteio_unit_tests.zig` from M9 still passes (regression check).
- `VkBufferMemoryBarrier2` between walk_frame and lz_decode_raw present and validated by sync2 layer.

**Estimated LOC:** 500

**Risks addressed:** R9 (LZ-decode 2-warp coupling: Tier-2 collapse policy verified).

**CUDA-side changes:** None.

---

### M13 — Wave 1 kernel 3/10: prefix_sum_chunks.comp + scan_gpu_vk.zig host driver

**Depends on:** M12

**Scope:** Port `src/decode/prefix_sum_chunks_kernel.cuh` to `src_vulkan/shaders/decode/prefix_sum_chunks.comp`. Uses `subgroupExclusiveAdd` on Tier-1; uses `warp_t2.glsl` shared-mem emulation on Tier-2. First production kernel exercising subgroup arithmetic.

**Production code path**: create `src_vulkan/host/decode/scan_gpu_vk.zig` (the prefix-sum host driver — mirrors `src/decode/scan_gpu.cu`). Architecture §2 mandate.

**Acceptance criteria:**
- Input: 1024 random chunk sizes; exclusive-prefix-sum output matches CPU reference byte-for-byte on Tier-1 AND Tier-2.
- Tier-1 SPIR-V dump: `OpExecutionMode LocalSizeId` with `required_subgroup_size_id=32`.
- Tier-2 SPIR-V dump: no required_subgroup_size mode.
- Edge cases: 1 chunk, 32 chunks (subgroup boundary), 33 chunks (crosses), 1024 chunks all handle correctly.
- Wave-64 SwiftShader lane (Tier-2 path) produces correct output.
- Stage assertion via M3a stageAssert: prefix-sum chunks output byte-identical to CUDA on all 70 fixtures both tiers.
- **Production code path verified**: `scan_gpu_vk.zig` dispatches prefix_sum_chunks.comp via the production decode driver; reachable from `slzDecompressAsync_vk`.

**Estimated LOC:** 550

**Risks addressed:** Tier-1 subgroup_size pinning validation in production.

**CUDA-side changes:** None.

---

### M14 — Wave 1 kernel 4/10: scan_parse.comp (D2) — FIRST BYTE-IDENTITY GATE

**Depends on:** M13

**Scope:** Port `src/decode/scan_parse_kernel.cuh` to `src_vulkan/shaders/decode/scan_parse.comp`. Single-thread orchestration. Reads 3-byte big-endian sub-chunk headers, off16 hi/lo split with 0xFFFF marker, off32 3/4-byte entries, SC tail prefix table. First wire-format-heavy kernel.

**Diff format and triage protocol** (rip-cord per §6): per-case diff committed to `tests/diffs/{case}_L{n}_scan_parse.diff` in the format:
```
OFFSET    CUDA_BYTES                                VK_BYTES                                  STAGE_FIELD_NAME
0x00010   3b 7f a4 00 00 00 00 ff                    3b 7f a4 ff ff 00 00 00                   off16_marker_in_chunk_4
```

Triage protocol:
1. Categorize by failure mode: (a) endianness, (b) alignment, (c) wire-format constant mismatch, (d) sc_group_size float-vs-integer leak (audited away in M9 via OpTypeFloat scan; if it occurs, that's a Reviewer-fix-regression bug in M9), (e) match_any emulation lane-election divergence (consult M5 baseline).
2. Most divergences are (a) or (b) — fix in GLSL, regenerate SPV, re-run.
3. If divergence is (e), validate against M5's standalone benchmark output.
4. Do not advance to M15 until clean.
5. Escalation: 3 working days without resolving any failure mode triggers `streamlz-vk/issues/scan-parse-blocker` and pauses M15-M27 per §6.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: scan_parse output (huff_descs + raw_descs intermediate arrays) byte-identical to CUDA on all 70 (case, level) pairs on lane A (Tier-1 NVIDIA).
- Same on lane E (Tier-2 lavapipe).
- Same on lane B' (Tier-2 SwiftShader wave64).
- sc_group_size handling: L1 frame (integer 131072) and L3 frame (integer 65536) both parse to identical descriptors as CUDA — no f32 anywhere (already audited in M9; M14 confirms in production).
- 0xFFFF off16 marker: synthetic frame with off16=0xFFFF forces fallback to off32 read; VK and CUDA produce identical offsets.
- Validation layer clean on all 14 cases.
- ANY divergence: per-case diff committed to `tests/diffs/` in the documented format, milestone FAILS, root-cause before advancing per triage protocol.

**Estimated LOC:** 900

**Risks addressed:** R3 (FIRST production byte-identity gate).

**CUDA-side changes:** None.

---

### M15 — Wave 1 kernel 5/10: compact_huff_descs.comp + compact_raw_descs.comp (D4/D5 paired)

**Depends on:** M14

**Scope:** Port both paired compaction kernels in one milestone per arch §11 step 5. Both are single-thread orchestration kernels (`local_size_x=32`, thread-0-only) that filter the descriptor table from M14 into separate huff-block and raw-block lists.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: compact_huff_descs output byte-identical to CUDA on all 70 fixtures both tiers.
- Stage assertion: compact_raw_descs output byte-identical to CUDA on all 70 fixtures both tiers.
- Both kernels can be dispatched concurrently (a barrier between them and the consumer, but not between them) — verified by checking M8c's barrier emission.
- Partitioning math: huff_count + raw_count == total_descriptors for every fixture.

**Estimated LOC:** 450

**Risks addressed:** R3 (continues byte-identity gate).

**CUDA-side changes:** None.

---

### M16 — Wave 1 kernel 6/10: gather_raw_off16.comp — parallel raw-block branch + partial-chain test ordering correction

**Depends on:** M15

**Scope:** Port `src/decode/gather_raw_off16_kernel.cuh` to `src_vulkan/shaders/decode/gather_raw_off16.comp`. Reads raw descriptors from M15 and off16 table, gathers raw-block payload bytes into a packed output. `local_size_x=64` on Tier-1, 32 on Tier-2.

**Partial-chain test corrected ordering**: `{walk_frame (M11) → prefix_sum_chunks (M13) → scan_parse (M14) → compact_huff + compact_raw (M15) → gather_raw_off16 (M16) → lz_decode_raw (M12)}`. The draft's omission of prefix_sum_chunks was a chain-order bug; prefix_sum_chunks must run before scan_parse can parse anything. The corrected chain runs the full prefix-of-decode for raw-only frames.

**Conformance bookkeeping**: synthetic raw-only frames are NOT counted in the 350-cell matrix. They are stage-assertion entries that contribute to the `STAGE_ASSERTIONS_PASSING / N` counter, NOT the `MATRIX_CELLS_PASSING / 350` counter. The draft's "~5 / 350 added for synthetic raw cases" mixed counters; this is fixed.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: gather output byte-identical to CUDA on all 70 fixtures both tiers.
- Stress test: a frame with 10k raw blocks of varying sizes (8B-512B) does not deadlock on Tier-2 workgroup-wide barriers.
- Partial-chain harness `{walk_frame → prefix_sum_chunks → scan_parse → compact → gather_raw_off16 → lz_decode_raw}` passes for raw-only synthetic frames; results increment the `STAGE_ASSERTIONS_PASSING` counter (NOT the 350-cell counter).

**Estimated LOC:** 400

**Risks addressed:** R9 (workgroup-wide barrier stress on Tier-2).

**CUDA-side changes:** None.

---

### M17 — Wave 1 kernel 7/10: merge_huff_descs.comp (D7)

**Depends on:** M16

**Scope:** Port `src/decode/merge_huff_descs_kernel.cuh` to `src_vulkan/shaders/decode/merge_huff_descs.comp`. Single-thread orchestration. `local_size_x=32`, thread-0-only.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: merged descriptor list byte-identical to CUDA on all 70 fixtures both tiers.
- Per-stream descriptor count (32 streams per locked HUFF_NUM_STREAMS) validated.
- Edge cases: zero huff blocks → output list of length 0; 1 huff block → length 1.

**Estimated LOC:** 350

**Risks addressed:** R3 (continues byte-identity gate).

**CUDA-side changes:** None.

---

### M18 — Wave 1 kernel 8/10: huff_build_lut.comp — first __match_any_sync site #3 in production

**Depends on:** M5, M9, M17

**Scope:** Port LUT-build portion of `src/decode/huffman_kernel.cu` to `src_vulkan/shaders/decode/huff_build_lut.comp`. FIRST PRODUCTION USE of __match_any_sync site #3 (11-iter ballot loop, domain [1..11]) from `match_any.glsl`. M5 validated emulation correctness AND perf vs CUDA; M9 unit-tested it; M18 integrates it. NV_partitioned variant exercised on NVIDIA lane A. Explicit deps cited: M5, M9, M17.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: LUT output byte-identical to CUDA on all 70 fixtures + 10 hand-crafted edge cases.
- Tier-1 + Tier-2 + Tier-1+NV_partitioned (when available) all byte-identical.
- Performance sanity (Tier-1 NVIDIA): <50 µs on a small frame.
- Performance (Tier-1 AMD): ≥ 65% of CUDA reference (consistent with M5 baseline, contingent on M5 R1 retirement).
- Performance (Tier-1 Intel): ≥ 65% of CUDA reference (same contingency).
- NV_partitioned variant selected on NVIDIA at runtime.
- 11-iter ballot loop verified via SPIR-V disassembly.

**Estimated LOC:** 600

**Risks addressed:** R1 (match_any site #3 in production).

**CUDA-side changes:** None.

---

### M19 — Wave 1 kernel 9/10: huff_decode_4stream.comp — hottest decode kernel

**Depends on:** M18

**Scope:** Port 4-stream Huffman decode of `src/decode/huffman_kernel.cu` to `src_vulkan/shaders/decode/huff_decode_4stream.comp`. `local_size_x=32`; 4 streams × 8 symbols-per-iter pattern preserved from CUDA. Uses `byteio.glsl` bitstream reads.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: literal stream + length stream + offset stream byte-identical to CUDA on all 70 fixtures + 5 Huffman-heavy frames.
- Tier-1 + Tier-2 + Tier-1+NV_partitioned all byte-identical.
- 32-stream boundary correctness on body of exactly 32×K bytes for K=1,2,4,8.
- Performance: Tier-1 NVIDIA ≥ 75% of CUDA reference on 16 MiB compressed payload.
- Below 60% triggers perf review.
- Perf numbers committed to `tests/perf/huff_decode_4stream_baseline.json` with hardware ID.

**Estimated LOC:** 1100

**Risks addressed:** R1 (hottest decode), R3 (Huffman wire-format byte-identity).

**CUDA-side changes:** None.

---

### M20 — Wave 1 kernel 10/10: lz_decode.comp — closes CUDA-encode → VK-decode roundtrip

**Depends on:** M19

**Scope:** Port `src/decode/lz_decode_kernels.cuh` to `src_vulkan/shaders/decode/lz_decode.comp`. Tier-1 `local_size_x=64` with per-subgroup `subgroupBarrier()`; Tier-2 `local_size_x=32` collapsed + 2× workgroups. LAST DECODE KERNEL — closes {CUDA encode → VK decode} for every level L1..L5.

**Conformance count update**: stage-assertions counter grows by 70 (one per fixture); the full-decompress matrix-cell counter jumps from 70/350 (CUDA→CUDA only) to 140/350 (CUDA→CUDA + CUDA→VK). The 140 includes the 70 sanity baseline + 70 new CUDA→VK cells. Bookkeeping per the M3 two-counter discipline.

**Acceptance criteria:**
- Full decode pipeline (M11-M20 chained via production code path) produces byte-identical output to CUDA decoder on all 14 silesia files at L1..L5 on lane A (NVIDIA).
- Same 70 CUDA→VK cells passing on lane E (lavapipe Tier-2).
- Same on lane B' (SwiftShader wave64).
- Tier-1 2-warp coupling: `subgroupBarrier()` synchronizes correctly under `SLZ_VK_VALIDATION=1`.
- Tier-2 2× workgroup dispatch verified by SPIR-V cross-reflect; output identical to Tier-1.
- Performance: Tier-1 decode throughput ≥ 50% of CUDA on RTX 4090.
- ANY case fails: emit per-case diff per M14 protocol; FAIL the milestone; do not proceed to M21.
- `MATRIX_CELLS_PASSING` counter shows 140 / 350.

**Estimated LOC:** 1100

**Risks addressed:** R9 (LZ-decode 2-warp coupling FULLY validated), R3 (closes CUDA→VK decode byte-identity).

**CUDA-side changes:** None.

---

### M21 — WAVE-1 GATE: cross-backend {CUDA-encode → VK-decode} hard-fail at L1..L5 on cutline lanes

**Depends on:** M20

**Scope:** Hard gate from arch §11 between Wave 1 and Wave 2. The {CUDA→CUDA} sanity cells (70) and {CUDA→VK} cells (70) are LIVE; the other 210 cells (VK→either) remain PENDING. The gate requires VK decode to pass at L1..L5 on the v1.0-alpha cutline lanes (§7) across all 14 golden fixtures.

Promote M3's harness from XFAIL-tolerant to HARD-FAIL on decode-path cells. Add lavapipe CI lane to every PR; add SwiftShader wave64 lane. Both lanes must be green. Tag commit `wave1-gate-passed`.

**Regression-window discipline (new)**: every PR touching encode (M22a onward) MUST re-run the Wave-1 decode gate before merge. CI gating: PRs editing `src_vulkan/shaders/common/*.glsl` or `src_vulkan/host/decode/*.zig` trigger the full Wave-1 gate. Prevents a future shared-helper change from silently breaking decode while encode is the focus.

**Operational definition of "green for 24 hours"** (replaces the draft's vague phrase): three consecutive nightly runs across all cutline lanes with zero failures AND `SLZ_VK_VALIDATION=1` clean.

**Real-mobile lanes F and G are NOT yet wired** — they're v1.0 release blockers tracked separately as M29/M30, OR are governed by the cutline §7 (at least one of {F.Adreno, F.Mali, G.CDNA} required for v1.0-alpha).

**Acceptance criteria:**
- All cutline lanes (per §7 MUST-have) report `140 / 350 cells passing — Wave 1 GATE: PASS`.
- M3 harness HARD-FAILS on decode-path cells at L1..L5 (was XFAIL-tolerant).
- Per-lane log archives committed to `tests/ci_logs/wave1_gate/{lane}_{timestamp}.log`; storage budget ≤ 100 MB per quarter (logs rotate older entries).
- Tag commit `wave1-gate-passed` — bisectable for regressions.
- ANY lane fails: gate RED; rip-cord per §6 (bisect, revert, escalate, or scope-cut).
- Wave 1 perf summary committed to `docs/vulkan_port_architecture.md` appendix.
- **Regression-window CI gating active**: PRs touching `src_vulkan/shaders/common/*.glsl` or `src_vulkan/host/decode/*.zig` run the Wave-1 gate before merge.
- "Green for 24 hours" defined operationally: 3 consecutive nightly runs across all cutline lanes with zero failures.

**Estimated LOC:** 300

**Risks addressed:** R3 fully retired for CUDA → VK decode direction.

**CUDA-side changes:** None.

**Notes:** On regression bisected to a tagged commit, the tag does NOT move; a new tag `wave1-gate-respassed` is created on the fixed commit. The bisect tag remains for historical reference.

---

### M22a — Wave 2 kernel 11a/16: lz_encode L1 NVIDIA-only baseline + encode_context_vk.zig + encode_lz_vk.zig

**Depends on:** M0, M5, M9, M21

**Scope:** Port the L1 path of `src/encode/lz_greedy_parser.cuh` (POST-M0-rewrite for sites A, B, #2) to `src_vulkan/shaders/encode/lz_encode.comp`. `local_size_x=32` (matches CUDA `__launch_bounds__(32,1)` per arch §3.1). Uses the DETERMINISTIC highest-lane-winner stores from M0 and the explicit-pattern site #2 (also from M0). Site #1 emulation from `match_any.glsl` (validated in M5). 

**Production code path delivery (architecture §2 mandate)**: M22a creates `src_vulkan/host/encode/encode_context_vk.zig` (encode handle, scratch allocation, hash table allocation — mirrors `src/encode/encode_context.cu`) AND `src_vulkan/host/encode/encode_lz_vk.zig` (LZ encode dispatcher). These are the production drivers consumed by `slzCompressAsync_vk`.

**Scope is explicitly narrowed**: L1 only, NVIDIA-only Tier-1. AMD/Intel/Tier-2 and L2..L5 are M22b. This split unblocks the M0 hash-store rewrite's first end-to-end validation with the smallest cross-section.

**Acceptance criteria:**
- LZ-encoded token stream byte-identical to CUDA on every fixture at **L1 only**, on **lane A (NVIDIA) only**.
- Both M0 hash-store sites (A, B not exercised here since L2+ is M22b; site A IS exercised at L1) produce byte-identical output to CUDA on lane A.
- Site #2 rewrite (from M0) and site #1 emulation produce byte-identical output to CUDA on lane A.
- `encode_context_vk.zig` allocates hash table via M7 bin-packing (small enough for L1, single-buffer case).
- `encode_lz_vk.zig` dispatches lz_encode via production code path; reachable from `slzCompressAsync_vk`.
- M3 harness `MATRIX_CELLS_PASSING` counter incremented by the L1 cells that flip to PASS for VK→CUDA and VK→VK directions (14 fixtures × 1 level × 2 directions = 28 new cells; matrix counter goes from 140 to 168).

**Estimated LOC:** 1200

**Risks addressed:** R1 (match_any site #1 NVIDIA baseline), R3 (encode-direction byte identity NVIDIA baseline), M0 rewrite end-to-end validation in production code path.

**CUDA-side changes:** None (M0 already landed).

**Notes:** Split-out from the draft's M22. Closes the smallest possible cross-section to validate M0's rewrites before scope expands in M22b.

---

### M22b — Wave 2 kernel 11b/16: lz_encode L2..L5 + AMD/Intel Tier-1 + Tier-2 + hash-size-at-create + Tier-2 worst-case descriptor layout

**Depends on:** M22a

**Scope:** Expand lz_encode.comp to L2..L5 (exercises M0 site B rewrite for L2+ rehash). Extend `encode_context_vk.zig` to: (a) compute maximum hash table size for the configured L5 / input ceiling AT slzCreate_vk time (the "hash-size-at-handle-create decision" the draft left ambiguous), (b) allocate N ≤ 2 GiB VkBuffers per M7's bin-packing, (c) build descriptor-set sizing for Tier-2 (the worst-case stable layout from arch §5.1, with N = ceil(max_supported_hash_region / minStorageBufferRange) descriptors baked into a single layout, unused slots receiving a 256 B placeholder buffer).

Validate on Tier-1 AMD + Tier-1+NV_partitioned + Tier-2 lavapipe + Tier-2 SwiftShader wave64. Intel validation contingent on M5 R1 retirement; if M5 deferred AMD/Intel per §6 rip-cord, AMD/Intel acceptance is `pending real hardware in v1.0-final` here too.

**AMD/Intel configuration clarification**: the test specifies which AMD configuration is under test by reading the driver-reported subgroupSize. "Tier-1 AMD" only applies when the driver reports 32; if it reports 64, the AMD lane runs under Tier-2 and is asserted accordingly. Documented in the test header.

**Acceptance criteria:**
- LZ-encoded token stream byte-identical to CUDA on every fixture at L2, L3, L4, L5 on lane A (NVIDIA).
- Same at L1..L5 on Tier-1 AMD when AMD driver reports subgroupSize=32 (cutline §7 contingent: if M5 R1 deferred, AMD acceptance is `pending real hardware`).
- Same on Tier-1 Intel when Intel driver reports 32 (same cutline contingency).
- Same on Tier-2 lavapipe at L1..L5.
- Same on Tier-2 SwiftShader wave64 at L1..L5.
- If AMD/Intel diverges: trace to (a) M0 incomplete, (b) site #1 emulation bug, or (c) wire-format misread. Site #2 semantic mismatch is no longer a possible cause because M0 rewrote both backends to use the same pattern.
- Performance: Tier-1 NVIDIA L1 encode ≥ 70% CUDA on RTX 4090; Tier-1 AMD ≥ 65% (R1 target consistent with M5); Tier-2 correct, no perf gate.
- Hash-size-at-create decision: `encode_context_vk.zig` computes max hash size at slzCreate_vk based on configured ceiling; Tier-2 builds worst-case-sized descriptor layout once and reuses across dispatches.
- Stress: 64 MiB random-ish input encodes at L5 without OOM (validates M7 bin-packing at scale).
- M3 harness: VK→CUDA and VK→VK cells continue flipping to PASS; `MATRIX_CELLS_PASSING` counter shows progress.

**Estimated LOC:** 1700

**Risks addressed:** R1 (match_any site #1 AMD/Intel/Tier-2 in production), R3 (encode-direction byte identity at all levels), R5 (encode hash >2 GiB bin-packing in production).

**CUDA-side changes:** None.

**Notes:** Largest single milestone. The split (M22a + M22b) totals ~2900 LOC vs the draft's 1500 — closer to the realistic 2500-4000 LOC range Reviewer 3 flagged for an LZ-greedy encoder of this complexity.

---

### M23 — Wave 2 kernel 12/16: huff_build_tables.comp + encode_huff_vk.zig host driver

**Depends on:** M22b

**Scope:** Port histogram + table-build portion of `src/encode/huffman_kernel.cu` to `src_vulkan/shaders/encode/huff_build_tables.comp`. Uses `atomicAdd` on shared-mem histograms (commutative-safe per R11 mitigation). `local_size_x=32`.

**Production code path**: create `src_vulkan/host/encode/encode_huff_vk.zig` (Huffman encode driver). Architecture §2 mandate.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: Huffman code-length tables + canonical code table byte-identical to CUDA on all 70 fixtures both tiers.
- Validated on Tier-1 NVIDIA + AMD + Tier-2 lavapipe + SwiftShader wave64.
- Mobile-relevant `atomicAdd`-on-shared-mem path validated on lane E.
- Edge case: input with one unique symbol → single-entry code-length table; matches CUDA.

**Estimated LOC:** 700

**Risks addressed:** R11 (shared-mem atomicAdd commutative-safe path proven).

**CUDA-side changes:** None.

---

### M24 — Wave 2 kernel 13/16: huff_encode_4stream.comp — 32-stream BIL output

**Depends on:** M23

**Scope:** Port bitstream-encode portion of `src/encode/huffman_kernel.cu` to `src_vulkan/shaders/encode/huff_encode_4stream.comp`. Writes 32-stream BIL Huffman body matching M19's decode-side layout. `local_size_x=32`.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: encoded Huffman body byte-identical to CUDA on all 70 fixtures both tiers.
- Round-trip: this encoder + M19 decoder reproduces input symbol stream byte-for-byte on 100 random symbol streams.
- 32-stream boundary correctness.
- Wire format header bytes hexdump-identical to CUDA reference.

**Estimated LOC:** 900

**Risks addressed:** R3 (Huffman wire-format byte-identity in encode direction).

**CUDA-side changes:** None.

---

### M25 — Wave 2 kernel 14/16: assemble_measure.comp + encode_assemble_vk.zig host driver

**Depends on:** M24

**Scope:** Port measure portion of `src/encode/assemble_kernel.cu` to `src_vulkan/shaders/encode/assemble_measure.comp`. Per-block compressed-size measurement, prefix-sum, total compressed_size written to a host-readable SSBO slot (this is where compressed_size_out is sourced — drained via `slzGetLastTimings_vk` after GPU completion, populating the M8c pseudo-kernel slot). `local_size_x=32`.

**Production code path**: create `src_vulkan/host/encode/encode_assemble_vk.zig` (assemble driver — mirrors `src/encode/encode_assemble.cu`).

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: per-chunk byte sizes array byte-identical to CUDA on all 70 fixtures both tiers.
- Final compressed_size matches CUDA exactly on all fixtures.
- compressed_size reaches host via `slzGetLastTimings_vk` drain (no pre-return lie). M0 aligned CUDA path is validated end-to-end here: pre-drain compressed_size_out is undefined; post-drain is exact on BOTH backends.
- M0 doctrine validated end-to-end.

**Estimated LOC:** 700

**Risks addressed:** R3 (continues byte-identity gate), M0 compressed_size contract validation.

**CUDA-side changes:** None (M0 already landed).

---

### M26 — Wave 2 kernel 15/16: assemble_write.comp

**Depends on:** M25

**Scope:** Port write portion of `src/encode/assemble_kernel.cu` to `src_vulkan/shaders/encode/assemble_write.comp`.

**Acceptance criteria:**
- Stage assertion via M3a stageAssert: final compressed payload bytes byte-identical to CUDA on all 70 fixtures both tiers.
- 3-byte big-endian sub-chunk header layout verified byte-by-byte against CUDA wire.
- off16/off32 split with 0xFFFF marker verified.

**Estimated LOC:** 700

**Risks addressed:** R3 (encode-direction byte identity).

**CUDA-side changes:** None.

---

### M27 — Wave 2 kernel 16/16: frame_assemble.comp — closes VK-encode end-to-end (FULL CONFORMANCE GATE 350/350 on cutline lanes)

**Depends on:** M26

**Scope:** Port final frame-assembly kernel from `src/encode/assemble_kernel.cu` to `src_vulkan/shaders/encode/frame_assemble.comp`. Single-thread orchestration writing frame header (magic 0x534C5A31, version 2, codec 1, log2_block_size, sc_group_size pre-quantized integers per R10) and chunk-descriptor table. `local_size_x=32`, thread-0-only. LAST ENCODE KERNEL. After this milestone all four cells of the cross-backend matrix are LIVE on all CI lanes.

**Conformance count**: 14 silesia × 5 levels × 4 directions = 280 + 70 sanity baseline = 350 cells (the architecture matrix definition; the draft's "5 directions = 350" was wrong arithmetic). Gate requires 350/350 on the v1.0-alpha cutline lanes per §7.

**Acceptance criteria:**
- Frame header bytes hexdump-identical to CUDA.
- Chunk-descriptor table bytes byte-identical to CUDA.
- sc_group_size on wire is pre-quantized integer.
- FULL encode pipeline (M22a-M27 chained) on all 70 fixtures: VK-encoded frame byte-identical to CUDA-encoded frame at L1..L5 on lane A (NVIDIA).
- Same on lane B (AMD RDNA2+ Tier-1, when driver reports subgroupSize=32).
- Same on lane C (Intel Arc Tier-1, contingent on M5 R1 retirement).
- Same on lane E (lavapipe Tier-2).
- Same on lane B' (SwiftShader wave64).
- 350 / 350 conformance checks GREEN on the v1.0-alpha cutline lanes per §7.
- Clamp-aware equality: any Tier-2 clamp from L5→L3 honored (effective_level_out=3 in reserved[0]).
- Validation layer zero errors across full run.
- `streamlz_vk.exe compress -v 5 silesia_xml.bin out.slz` → `streamlz.exe decompress` (CUDA) byte-identical to original.
- Reverse: `streamlz.exe compress` → `streamlz_vk.exe decompress` byte-identical.
- Conformance report committed to `docs/vulkan_port_architecture.md` appendix.
- Tag commit `wave2-gate-passed`.
- **Rip-cord per §6 if any cutline lane red**: bisect, partial demotion (e.g. Intel-L5 to Tier-2), 5-day-timeout escalation.

**Estimated LOC:** 800

**Risks addressed:** R3 fully retired for VK-encode → byte-identical .slz on cutline lanes; R10 validated.

**CUDA-side changes:** None.

---

### M28 — Wave 3 perf: NV_partitioned validation + shape-keyed cmd buffer cache (bounded 16) + invalidate hook integration + indirect-dispatch evaluation-only

**Depends on:** M27

**Scope:** Wave 3 perf-only improvements from arch §11.

1. **NV_partitioned validation**: validate tier1_nv SPV variants byte-identical to tier1 on Turing+ NVIDIA; measure LZ-encode site #1 fast-path speedup. **Hard pass/fail (replaces draft's unfailable OR):** uplift ≥ 8% on LZ-encode L1..L4 → ship tier1_nv blobs; uplift < 8% → tier1_nv blobs are deleted from the repo, SPV count drops from 51 to 34 unique blobs (17 × 2 tiers), CHANGELOG entry documents the deprecation decision and the count change. No middle "no measurable gain" outcome — deprecation is the consequence of missing the threshold.

2. **Shape-keyed primary command buffer cache** in `vk_command.zig` keyed by `(input_size_bucket, level, tier)`. **Bucket definition (was unspecified in draft):** input_size_bucket = `floor(log2(input_size_bytes))` clamped to [10, 28], giving 19 buckets. Combined with 5 levels and 2 tiers gives 190 logical shapes; in practice <16 are hot on any single stream. **Cap cache at 16 entries per handle (NOT 32; architecture §7 mandate — Reviewer 1 fatal correction).** LRU eviction. The invalidate_all_cached_cbs hook from M8b is wired here for the first time: pipeline destroy or `SLZ_VK_VALIDATION` toggle mid-run flushes the cache.

3. **`vkCmdDispatchIndirect` evaluation ONLY** (no integration). M28 produces a written recommendation (lock for v1.1 or defer). Integration into already-shipped huff_build_lut/compaction kernels is explicitly OUT OF SCOPE for M28 because it would require re-running the M21+M27 conformance gates. This is stated explicitly per Reviewer 2's flag.

4. **Cache evicted-entry resource cleanup**: LRU eviction returns descriptor-pool slots, VkCommandBuffer to pool, and any per-entry persistent resources. Unit test verifies post-eviction resource accounting.

**Acceptance criteria:**
- NV_partitioned variant byte-identical to base variant on all 70 fixtures (sanity gate).
- NV_partitioned uplift on LZ-encode L1..L4 measured on real NVIDIA Turing+ card. **Hard pass/fail: ≥ 8% → ship; < 8% → tier1_nv deleted, SPV count drops to 34 unique, CHANGELOG entry filed, status string for NVIDIA notes NV_partitioned removed.**
- Shape-keyed cmd buffer cache hit rate ≥ 90% on a 1000-frame stream of identical-size inputs.
- Cache capped at 16 entries per architecture §7; 17th distinct shape evicts LRU.
- Eviction unit test: evicted entry's VkCommandBuffer returned to pool, descriptor-pool slots released; no leaks under `SLZ_VK_VALIDATION=1`.
- Record-time drops from ~200 µs (first call) to <5 µs (subsequent calls).
- Small-input (<4 MiB) encode throughput on RTX 4090 increases by ≥ 5% vs M27 baseline.
- Indirect-dispatch recommendation document committed to `docs/vulkan_port_architecture.md` §12 (lock or defer for v1.1). No production-kernel integration.
- M8b's invalidate_all_cached_cbs hook fires when a pipeline is destroyed mid-run (validated by a forced `SLZ_VK_VALIDATION` toggle test).
- No byte-identity regressions on M27's 350 / 350 conformance matrix.

**Estimated LOC:** 800

**Risks addressed:** R1 amplifier (NV_partitioned closes NVIDIA gap), R7 amplifier (shape-keyed cmd cache).

**CUDA-side changes:** None.

**Notes:** Parallel to M29 and M30 (all depend only on M27).

---

### M29 — Lane F real-mobile (Adreno Pixel 8 + Mali Pixel 6 via adb) — v1.0-alpha cutline contingent

**Depends on:** M1 (Android cross-compile), M27

**Scope:** Per arch §22 risks: lavapipe/SwiftShader implement subgroup-spanning semantics differently from real Bifrost/Valhall, so v1.0 release requires real mobile hardware validation. Stand up CI lane F: build streamlz_vk for arm64-v8a Android via `zig build vk_android` (M1), deploy to Pixel 8 (Adreno 740) and Pixel 6 (Mali-G78 Valhall) via adb.

**Mobile clamp policy actually exercised**: the reduced corpus is 4 cases × L1..L5 = 20 assertions per phone (NOT the draft's L1..L3, which never exercises L5→L3 clamp on real mobile and defeats R12 retirement on lane F). The L5 cases on phones with <4 GiB VRAM will trigger the clamp; the test asserts `effective_level_out == 3` after the clamp and that the bytes match the L3 golden.

Both phones MUST report Tier-2 and pass their 20-case reduced corpus (clamp-aware) per cutline §7. Bifrost device (Pixel 3 Mali-G76) deferred to v1.1.

**R2 emulation-correctness retirement happens here** (M4 retired only the classification portion).

**Acceptance criteria:**
- Pixel 8 (Adreno 740): probe reports Tier-2; 20-case (4 silesia × L1..L5) reduced corpus all pass.
- Pixel 6 (Mali-G78 Valhall): probe reports Tier-2; 20-case all pass.
- L5 cases on phones with <4 GiB VRAM trigger the mobile clamp; the test asserts `effective_level_out == 3` and bytes match the L3 golden (exercises R12 retirement on real mobile).
- Mali-G78 subgroup-boundary behavior verified differs from lavapipe (logged) but Tier-2 emulation produces identical bytes.
- adb-based CI script committed to `tools/ci/run_mobile_lane.sh`; runs nightly.
- Perf numbers logged but NOT gated (Mali 25-50% of CUDA per R2 documented; Adreno 40-60%).
- **Operational cost**: Pixel 8 + Pixel 6 acquisition budget capped at $1500 total (one-time); ongoing adb-CI worker electricity ~$0/month. If cloud Adreno (e.g., AWS Snapdragon-on-cloud) is used instead, monthly budget capped at $200.
- **Cutline §7 contingent**: if Mali Pixel 6 hardware unavailable, deferred to v1.1 — v1.0-alpha ships with Adreno Pixel 8 alone (assuming Adreno is the green lane).

**Estimated LOC:** 800

**Risks addressed:** R2 emulation-correctness portion fully retired on real mobile silicon (R2 was split: M4 classification, M29 emulation correctness); R12 mobile clamp exercised on real hardware.

**CUDA-side changes:** None.

---

### M30 — Lane G real AMD CDNA wave64 (MI100/MI210) — v1.0-alpha cutline contingent

**Depends on:** M27

**Scope:** Stand up CI lane G: AMD CDNA MI100 or MI210 (real wave64 hardware). Build streamlz_vk for Linux x86_64, deploy to CDNA-equipped server (cloud rental acceptable for periodic validation; budget capped at $500/month for weekly runs).

**Acceptance criteria:**
- MI100/MI210 probe reports Tier-2 with subgroupSize=64.
- 350 / 350 cross-backend assertions pass on lane G (full corpus, NOT a reduced one).
- Tier-2 wave64 hash-store determinism verified: M0's highest-lane-winner rewrite produces byte-identical output on real wave64.
- Lane G CI script committed to `tools/ci/run_cdna_lane.sh`; runs weekly.
- Perf numbers logged but NOT gated.
- **Operational cost**: cloud CDNA rental budget capped at $500/month; if budget exceeded, runs frequency drops to bi-weekly.
- **Cutline §7 contingent**: if CDNA hardware unobtainable and at least one of M29's lanes is green, v1.0-alpha ships and CDNA defers to v1.1 with documented status-string note (see §7).

**Estimated LOC:** 500

**Risks addressed:** Tier-2 wave64 correctness on real CDNA hardware.

**CUDA-side changes:** None.

---

### M31 — v1.0-alpha release: CLI parity + DLL ABI freeze + docs + tag

**Depends on:** M28, M29, M30

**Scope:** Final release prep. (1) `src_vulkan/cli/main_vk.zig` achieves full CLI flag parity with streamlz.exe (same flags, same semantics, same exit codes including the M2-audited unsupported-operation exit code); surface `effective_level_out` clamps via stderr per arch §6. (2) `include/streamlz_vk.h` ABI frozen — all 16 _vk symbols signed, documented, version-tagged `3.0.0-vk`. (3) `docs/vulkan_port_architecture.md` reviewed and finalized. (4) CHANGELOG entry enumerating: 16 new _vk symbols, 2 new status codes, Tier-1 + Tier-2 device matrices, M0 CUDA-side prereqs with byte-equivalence guarantee, M0 ABI repurpose of reserved[0], M0 compressed_size contract change, NV_partitioned tier1_nv ship-or-deprecate decision from M28. (5) Release artifacts per platform: Windows (streamlz_vk.exe, streamlz_vk.dll, .lib, .pdb), Linux (streamlz_vk, streamlz_vk.so), Android (libstreamlz_vk.so for arm64-v8a per M1). (6) Fresh-machine smoke test (clarified): clone repo on a machine with ONLY shaderc/glslc + zig installed AND a Vulkan ICD/driver — the .spv blobs load from git and full L1..L5 compress/decompress works without the Vulkan SDK (the SDK is dev-time; runtime needs ICD only). The test_vk lanes that require lavapipe/SwiftShader installs are documented separately. (7) README backend matrix table (CUDA vs Vulkan feature parity), including disclosures from cutline §7 (which mobile/CDNA lanes are validated). (8) Tag `v1.0.0-alpha-vk`. (9) File v1.1 backlog issues tracking arch §21 deferred items + the cutline §7 deferred lanes.

**Acceptance criteria:**
- `streamlz_vk.exe --help` matches `streamlz.exe --help` except for binary name.
- All 16 _vk symbols exported from streamlz_vk.dll, ABI-stable signatures, headers committed.
- Fresh-machine smoke test: clone repo, install only shaderc/glslc + zig + a Vulkan ICD, `zig build vk vklib test_vk` succeeds for the value-path lanes that don't require lavapipe/SwiftShader. Document separately the lanes that require additional installs.
- M27 350 / 350 conformance + M28 perf + at least one of {M29 Adreno, M29 Mali, M30 CDNA} green at release tag (cutline §7 MUST-have).
- `docs/vulkan_port_architecture.md` final; CHANGELOG entry committed.
- Tag `v1.0.0-alpha-vk` pushed.
- v1.1 backlog issues opened tracking arch §21 deferred items + cutline §7 deferred lanes (e.g., "lane G CDNA deferred — see CHANGELOG §X").

**Estimated LOC:** 700

**Risks addressed:** Final release discipline; no-SDK-build promise validated; cutline §7 enforced at tag.

**CUDA-side changes:** None.

---

## 9. Out-of-scope for v1.0-alpha

Per arch §21 and cutline §7. v1.1 backlog issues filed in M31:

1. Wave-64 packed Tier-1 sub-variant for AMD GCN/CDNA — design exists, defer to v1.1.
2. `vkCmdDispatchIndirect` integration into production kernels — M28 evaluation-only; lock-or-defer decision documented.
3. Worker-thread stack size remeasurement.
4. Bifrost real-mobile lane (Pixel 3 Mali-G76 or equivalent).
5. `slzCreateEx_vk` struct.
6. Second internal queue for async-compute overlap.
7. `VK_KHR_8bit_storage` fast-path inside Tier-1 byteio.
8. Pipeline-cache cross-tier sharing.
9. Encode pipeline fusion (port walkStream + assemble-measure prefix-sum to GPU compute).
10. Single-thread orchestration kernel fusion.
11. macOS / MoltenVK path.
12. Any cutline §7 NICE-TO-HAVE lanes that slipped from v1.0-alpha.

---

## 10. Conformance dashboard discipline (canonical)

Two never-mixed counters:

- `MATRIX_CELLS_PASSING / 350` — only counts the 350 cells in the architecture matrix (14 silesia × 5 levels × 4 directions {CUDA→CUDA, CUDA→VK, VK→CUDA, VK→VK}, plus the always-live 70 CUDA→CUDA sanity baseline embedded in those directions).
- `STAGE_ASSERTIONS_PASSING / N` — counts per-stage byte-identity assertions from M3a stageAssert calls. Synthetic-test cases (M16's raw-only partial-chain test) increment this counter, NEVER the matrix counter.

Progress milestones (canonical numbers):
- M3: `MATRIX_CELLS_PASSING = 70 / 350` (CUDA→CUDA sanity only).
- M11-M19: matrix counter unchanged (decode-only kernels don't produce matrix cells; they produce stage assertions only).
- M20: `MATRIX_CELLS_PASSING = 140 / 350` (CUDA→CUDA + CUDA→VK = 70 + 70 = 140).
- M22a: `MATRIX_CELLS_PASSING = 168 / 350` (adds 28 cells for L1 VK→CUDA and VK→VK at L1 only on 14 fixtures).
- M22b through M27: counter grows monotonically as each level × direction lands.
- M27: `MATRIX_CELLS_PASSING = 350 / 350`.

---

## 11. Operational costs sizing (new)

- **Pixel 8 + Pixel 6 acquisition**: $1500 one-time (or cloud Adreno equivalent ~$200/month).
- **AMD CDNA cloud rental** (M30): $500/month for weekly runs.
- **CI runner shaderc + Vulkan SDK + ICDs**: ~$0 (open-source toolchain on existing CI fleet).
- **Validation-layer log archival** (`tests/ci_logs/wave1_gate/`): rotating 100 MB / quarter; retention 1 year; total storage ~400 MB.
- **Per-stage golden blobs** (`tests/golden_stage/`): ≤ 30 MiB uncompressed or ≤ 15 MiB zstd-compressed.
- **Total committed `tests/` budget**: ~100 MiB worst-case (golden_slz ≤ 50 MiB + golden_stage ≤ 30 MiB + perf + diffs + logs).

---

## 12. Contingency budget

Total LOC estimate: ~25,500. Plus a 25% contingency buffer (~6,400 LOC) reserved for the highest-risk milestones (M22a + M22b combined = 2900 estimated, contingency adds ~1000; M3a stage instrumentation hooks if encoder/decoder need more invasive changes than expected = +400; M4 multi-thread stress + Adreno alignment work = +500; etc.). Contingency is documented up front so a 30% LOC overrun on M22 doesn't appear as a scope failure.

---

**Document end.** This plan incorporates all 16 fatal flaws and all 30 serious concerns raised by the three adversarial reviewers, plus the relevant subset of the 24 minor concerns. The 34-milestone count (vs the draft's 31) reflects the three structural split-outs (M3 → M3 + M3a, M8 → M8a + M8b + M8c, M22 → M22a + M22b) required to close the reviewer-flagged scheduling and dependency gaps. The v1.0-alpha cutline in §7 establishes the MUST/SHOULD/NICE-TO-HAVE split that the draft lacked.
