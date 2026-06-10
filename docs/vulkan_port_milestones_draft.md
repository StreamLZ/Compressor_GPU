# Vulkan Port — Milestone Plan (Draft)

## 0. Locked constraints

1. **Shaders:** GLSL → glslc → SPIR-V. Vulkan 1.3 target for Tier-1, Vulkan 1.2 floor for Tier-2. glslc ≥ 1.3.250 pinned (defends SPIRV-Tools BDA storage-class regression).
2. **Vendor coverage:** Tier-1 = NVIDIA Turing+, AMD RDNA2+ (when driver reports subgroupSize==32), Intel Arc Alchemist+ (when driver pins 32). Tier-2 = everything else — Adreno (all gens), Mali (Bifrost + Valhall, no Mali gen ever in Tier-1), pre-Arc Intel iGPU, AMD GCN/CDNA wave64, RDNA2 when driver reports 64.
3. **Sibling build:** `src_vulkan/` next to `src/`; CUDA `src/` untouched. Artifacts: `streamlz_vk.exe`, `streamlz_vk.dll`. All C symbols suffixed `_vk`. Same CLI flags as `streamlz.exe`.
4. **Full API parity day 1:** 15 symbols (14 mirrors of `include/streamlz_gpu.h` + `slzRegisterBuffer_vk`). 15th symbol exported unconditionally on both tiers (no-op on Tier-1, mandatory on Tier-2).
5. **Wire format:** byte-identical to CUDA `.slz` output at L1..L5. Cross-backend roundtrip mandatory.
6. **Compression levels:** L1..L5 produce identical bytes per level as the CUDA encoder.
7. **Architecture authority:** `docs/vulkan_port_architecture.md` is authoritative. Anything inconsistent with that document is wrong.

## 1. Plan overview

**Winning lens:** TOP-DOWN PLANNER (shippable-but-functionally-empty skeleton first, then port kernels in arch §11 order against a live conformance harness), with key grafts from RISK-DRIVEN PLANNER (standalone `__match_any_sync` perf microbenchmark BEFORE production kernel integration; real-mobile probe lane in the early R2-retirement phase; explicit CUDA prereqs at M0 with zero deps so they run in parallel with Vulkan scaffolding) and from BOTTOM-UP PLANNER (dedicated common-GLSL-helpers milestone with per-primitive unit tests; CUDA prereqs as root milestones so golden fixtures are valid before harness is built; XFAIL bookkeeping discipline that exits non-zero when an unexpected XFAIL flips to PASS).

The plan ships a **buildable but-graceful streamlz_vk binary at milestone 2** with all 15 _vk symbols stub-exported returning a NAMED `SLZ_ERROR_UNSUPPORTED` for every operation. Conformance harness lands at milestone 3 with a quantified `0/350 passing` dashboard so every subsequent milestone has visible progress delta. Risk retirements R1-R5 land in milestones 4-9 BEFORE any production kernel ships (M3 standalone match_any benchmark catches the AMD/Intel perf cliff at milestone 4 of 32, not at milestone 22 of 35). Wave 1 decode kernels (M10-M19) land in locked arch §11 order against the live harness. Wave-1 gate (M20) hard-fails until {CUDA encode → VK decode} passes at L1..L5 on Tier-1 + Tier-2. Wave 2 encode (M21-M26) and full-matrix conformance (M27) follow. Wave 3 perf + real-mobile lanes (M28-M30) close v1.0-alpha (M31).

**Total milestone count: 31.** Estimated total LOC: ~22,500 (mean ~725, median ~600, max 1500 for first encode kernel).

**Critical path length: 31 milestones.** Hard gates: M3 (conformance harness wired, 0/350 baseline), M9 (all 5 risks retired before Wave 1), M20 (Wave-1 decode CUDA→VK byte identity at L1..L5), M27 (full matrix 350/350), M31 (v1.0-alpha tag with real-mobile + CDNA lanes green).

## 2. Critical path

```
M0  → M1  → M2  → M3  → M4  → M5  → M6  → M7  → M8  → M9  → M10
(CUDA prereqs) → (scaffolding+ABI) → (conformance harness) → (loader+probe+R2 retire)
→ (match_any standalone+R1 retire) → (BDA+R4 retire) → (memory+2GB split+R5 retire)
→ (pipeline+timing+sync chassis) → (common GLSL helpers) → (hello-dispatch substrate gate)
→ (Wave 1 kernel 1: walk_frame)

M10 → M11 → M12 → M13 → M14 → M15 → M16 → M17 → M18 → M19 → M20
(walk_frame) → (lz_decode_raw) → (prefix_sum_chunks) → (scan_parse — first byte-identity gate)
→ (compact_huff+raw) → (gather_raw_off16) → (merge_huff_descs) → (huff_build_lut — match_any site #3 production)
→ (huff_decode_4stream) → (lz_decode — closes CUDA-enc→VK-dec) → (WAVE-1 GATE)

M20 → M21 → M22 → M23 → M24 → M25 → M26 → M27
→ (lz_encode — match_any site #1 production, consumes M0) → (huff_build_tables)
→ (huff_encode_4stream) → (assemble_measure) → (assemble_write) → (frame_assemble — closes VK-enc end-to-end)
→ (FULL CONFORMANCE GATE: 350/350)

M27 → M28 → M29 → M30 → M31
→ (Wave-3 perf: NV_partitioned + shape-keyed cmd cache) → (lane F real-mobile Adreno+Mali)
→ (lane G real AMD CDNA wave64) → (v1.0-alpha tag + CLI parity + docs)
```

Parallel branches (can ship out-of-band): M0 ↔ M1 (CUDA prereqs are independent of Vulkan scaffolding). M5 (BDA probe) ↔ M6 (memory allocator) once M4 is in. M28 ↔ M29 ↔ M30 (perf tuning + hardware lanes) once M27 is green.

## 3. Dependency graph

| Milestone | Depends on | Type |
|-----------|------------|------|
| M0 | (none) | CUDA prereq |
| M1 | (none) | Vulkan scaffolding |
| M2 | M1 | Stub ABI + CLI graceful failure |
| M3 | M0, M2 | Conformance harness with golden fixtures |
| M4 | M2 | Loader + vendor probe (R2 retire) |
| M5 | M4 | match_any standalone microbenchmark (R1 retire) |
| M6 | M4 | BDA + slzRegisterBuffer_vk validation (R4 retire) |
| M7 | M6 | Memory allocator + 2GB split prototype (R5 retire) |
| M8 | M5, M7 | Pipeline cache + sync + timing chassis |
| M9 | M8 | Common GLSL helpers + per-primitive unit tests |
| M10 | M3, M8, M9 | Hello-dispatch substrate gate |
| M11 | M0, M10 | Kernel 1: walk_frame (D1) |
| M12 | M11 | Kernel 2: lz_decode_raw |
| M13 | M12 | Kernel 3: prefix_sum_chunks |
| M14 | M13 | Kernel 4: scan_parse (D2) — first byte-identity gate |
| M15 | M14 | Kernel 5: compact_huff + compact_raw (D4/D5) |
| M16 | M15 | Kernel 6: gather_raw_off16 |
| M17 | M16 | Kernel 7: merge_huff_descs (D7) |
| M18 | M5, M17 | Kernel 8: huff_build_lut — match_any site #3 production |
| M19 | M18 | Kernel 9: huff_decode_4stream |
| M20 | M19 | Kernel 10: lz_decode — closes CUDA-enc→VK-dec |
| M21 | M20 | **WAVE-1 GATE**: 70/350 conformance L1..L5 both tiers |
| M22 | M0, M5, M21 | Kernel 11: lz_encode — match_any site #1 production |
| M23 | M22 | Kernel 12: huff_build_tables |
| M24 | M23 | Kernel 13: huff_encode_4stream |
| M25 | M24 | Kernel 14: assemble_measure |
| M26 | M25 | Kernel 15: assemble_write |
| M27 | M26 | Kernel 16: frame_assemble — closes VK-enc end-to-end |
| M28 | M27 | **FULL CONFORMANCE GATE**: 350/350 on lanes A/B/C/E/B' |
| M29 | M28 | Wave-3 perf: NV_partitioned + shape-keyed cmd cache |
| M30 | M28 | Lane F (Adreno/Mali) + Lane G (AMD CDNA) real-hardware lanes |
| M31 | M28, M29, M30 | v1.0-alpha tag: CLI parity + docs + release |

## 4. CUDA-side prerequisites

All three mandated CUDA-side changes from architecture §6 land in **M0** as a single root milestone so they execute in parallel with Vulkan scaffolding (M1):

| Change | File / Location | Consumed by milestone | Purpose |
|--------|-----------------|-----------------------|---------|
| Highest-lane-winner rewrite at site A | `src/encode/lz_greedy_parser.cuh` ~line 187 (`ht[h] = my_pos`) | M22 (lz_encode) | Replace NVIDIA-implicit "highest lane wins" with explicit `__match_any_sync(mask,h)` + `__clz` highest-indexed lane election. Byte-equivalent on NVIDIA (CUDA goldens remain bit-exact); makes byte-identity portable to Vulkan AMD/Intel. Tracked as `streamlz/issues/cuda-deterministic-hash-store`. |
| Highest-lane-winner rewrite at site B | `src/encode/lz_greedy_parser.cuh` ~line 311 (`ht[hashKey6(rk, hash_bits, hash_mask)] = rp` in L2+ rehash) | M22 (lz_encode L2+ levels) | Same treatment as site A — explicit highest-lane-winner store. Required for L2..L5 byte identity. |
| `reserved[0]` → `effective_level_out` | `include/streamlz_gpu.h` slzCompressOpts_t | M3 (conformance harness clamp-aware equality) and M22+ (VK encode writes post-clamp level on Tier-2) | Repurpose first padding slot; struct stays 32B; comptime asserts unchanged. CUDA writes requested level (no-op semantically); Vulkan Tier-2 writes post-clamp level when mobile VRAM forces L5→L3. Both CLIs surface clamps via stderr. |
| `compressed_size_out` doctrine fix | `include/streamlz_gpu.h` slzCompressAsync docstring | M25 (assemble_measure produces size via GPU) | Drop the "compressed_size written before return" contract (physically impossible to honor on Vulkan — size comes from `slzAssembleMeasureKernel` on GPU). Both backends align on size-via-`slzGetLastTimings` drain after GPU completion. Docstring rewritten; CUDA backend behavior unchanged byte-wise. |

**Validation**: M0 regenerates all 14 silesia × L1..L5 = 70 CUDA-encoded `.slz` golden fixtures and commits a SHA-256 manifest. If post-rewrite goldens differ from pre-rewrite goldens (would indicate the rewrite is NOT byte-equivalent on NVIDIA), M0 fails and triggers an architecture revision before M3 builds the harness.

## 5. Milestones

### M0 — CUDA-side prerequisites: highest-lane-winner + reserved[0] + compressed_size docstring

**Depends on:** (none — root milestone, runs parallel to M1)

**Scope:** Three CUDA-side edits in a single milestone so the byte-identity reference is locked before any Vulkan work consumes it. (1) Rewrite `src/encode/lz_greedy_parser.cuh:187` (`ht[h] = my_pos`) and `:311` (`ht[hashKey6(...)] = rp` in L2+ rehash) to use explicit highest-lane-winner stores via `__match_any_sync(active_mask, h)` + `31 - __clz(group_mask)` to elect the highest-indexed lane sharing the bucket, gating the store to only that lane. Byte-equivalent to current NVIDIA implicit-order behavior; the goal is to make the implicit hardware convention EXPLICIT so it ports to Vulkan AMD/Intel/Mali. (2) Repurpose `slzCompressOpts_t.reserved[0]` as `int effective_level_out`; preserve struct size (32B) and all comptime asserts in `src/encode/cuda_ffi.zig`. CUDA `slzCompressAsync` writes `opts.level` into the new field before returning. (3) Update `include/streamlz_gpu.h` docstring on `slzCompressAsync` to drop the "compressed_size written before return" lie; document size-via-`slzGetLastTimings`-drain. (4) Regenerate all 70 CUDA goldens at L1..L5 on the 14-case silesia mini-corpus; commit `tests/golden_slz/checksums.txt` SHA-256 manifest; assert post-rewrite SHA-256s match pre-rewrite (proves byte-equivalence on NVIDIA). (5) File `streamlz/issues/cuda-deterministic-hash-store` referencing the commit.

**Acceptance criteria:**
- `src/encode/lz_greedy_parser.cuh` compiles cleanly under existing nvcc.
- All existing CUDA encode goldens at L1..L5 on the 14 silesia files reproduce byte-identical `.slz` output (zero diffs) — proves the rewrite is NVIDIA-byte-equivalent.
- `zig build && zig build test` passes with no regressions on `src/encode/gpu_roundtrip_tests.zig`.
- Both affected sites in `lz_greedy_parser.cuh` contain explicit highest-lane-winner pattern with a `// VULKAN_PORTABILITY` comment naming the prerequisite.
- `sizeof(slzCompressOpts_t)` is 32 bytes (verified with a `_Static_assert` in C and Zig `comptime` assert).
- CUDA test exercises `slzCompressAsync` at level 3 and asserts `opts.effective_level_out == 3` on return.
- `tests/golden_slz/` contains 70 fixtures + `checksums.txt` manifest.
- `include/streamlz_gpu.h` docstring above `slzCompressAsync` no longer claims `compressed_size` is written before return.
- CHANGELOG entry documents the ABI semantic change (no symbol or size change).
- Issue `streamlz/issues/cuda-deterministic-hash-store` filed.

**Estimated LOC:** 350

**Risks addressed:** R3 (precondition for portable byte-identity), R12 (clamp-aware ABI prereq)

**CUDA-side changes:** `src/encode/lz_greedy_parser.cuh` rewritten at both store sites; `include/streamlz_gpu.h` slzCompressOpts_t `reserved[0]` → `effective_level_out`; slzCompressAsync writes field; docstring fix. PTX regenerated and committed.

**Notes:** This milestone is the root of the byte-identity chain — if it fails (rewrite is NOT byte-equivalent on NVIDIA), the entire Vulkan port stalls until the rewrite is fixed or the goldens are re-baselined with stakeholder approval.

---

### M1 — src_vulkan/ scaffolding + build.zig vk/vklib/test_vk steps + 51 SPV blob placeholders

**Depends on:** (none — parallel to M0)

**Scope:** Create the `src_vulkan/` tree exactly per architecture §2 (cli/, shaders/{common,encode,decode}, spv/{tier1,tier1_nv,tier2}/{encode,decode}, host/{encode,decode}, tests/, include/streamlz_vk.h, version.zig). Add three new build.zig steps: `vk` (builds streamlz_vk.exe), `vklib` (builds streamlz_vk.dll exporting all 15 _vk symbols), `test_vk` (runs Zig tests under src_vulkan/tests/). Default `zig build` stays CUDA-only and untouched. Create `tools/build_glsl.bat` that asserts `glslc --version >= 1.3.250` (defends SPIRV-Tools BDA regression per arch §22), iterates 17 known kernel basenames, and emits 51 `.spv` blobs to `src_vulkan/spv/{tier1,tier1_nv,tier2}/{encode,decode}/`. Create 17 panic-shell `.comp` files (4-line `#version 460` + `#include "tier_gates.glsl"` + `void main(){ /* write 0xDEADBEEF sentinel */ }`) so 51 SPVs compile end-to-end. Commit all 51 SPVs to git (per arch locked decision #11). version.zig exports `"3.0.0-vk"`.

**Acceptance criteria:**
- `Get-ChildItem src_vulkan/ -Recurse -Directory` matches architecture §2 exactly.
- Default `zig build` (no args) still builds `streamlz.exe` + `streamlz_gpu.dll` and runs zero Vulkan code.
- `zig build vk` produces `streamlz_vk.exe`; running with no args prints `streamlz_vk 3.0.0-vk` and exits 0.
- `zig build vklib` produces `streamlz_vk.dll`.
- `zig build test_vk` succeeds on an empty test suite.
- `tools/build_glsl.bat` aborts with a clear error if `glslc --version` < 1.3.250.
- Running `tools/build_glsl.bat` from repo root produces exactly 51 `.spv` files at documented paths.
- Each `.spv` is a valid SPIR-V module per `spirv-val`.
- `src_vulkan/spv/README.md` documents regeneration command + glslc version pin.

**Estimated LOC:** 800

**Risks addressed:** glslc < 1.3.250 SPIRV-Tools BDA regression; substrate risk (ensures Vulkan work never breaks CUDA build)

**CUDA-side changes:** None.

**Notes:** Parallel to M0 since pure Vulkan scaffolding has zero CUDA dependency. The 17 panic-shells write a sentinel to a known offset (used in M2 to confirm un-ported kernels fail informatively).

---

### M2 — Stub 15-symbol C ABI + CLI graceful failure ("UNSUPPORTED" with named operation, never crash)

**Depends on:** M1

**Scope:** Create `include/streamlz_vk.h` mirroring all 14 streamlz_gpu.h symbols with `_vk` suffix PLUS the 15th `slzRegisterBuffer_vk(handle, void* ptr, size_t bytes, slzDeviceBuffer_t* out)`. Create `src_vulkan/streamlz_vk.zig` exporting all 15 symbols as Zig stubs:
- `slzVersionString_vk` → `"3.0.0-vk"`
- `slzStatusString_vk` → static strings table
- `slzCompressDefaultOpts_vk`, `slzDecompressDefaultOpts_vk` → zero structs (with `effective_level_out = 0`)
- `slzRegisterBuffer_vk` → returns `SLZ_SUCCESS` (no-op stub)
- All other 10 symbols → return `SLZ_ERROR_UNSUPPORTED` with a `slzStatusString_vk`-resolved message like `"Vulkan backend slzCompressAsync_vk not yet implemented (waiting on M27)"`.

Create `src_vulkan/cli/main_vk.zig` as a clone of `src/main.zig` that dispatches to `_vk` symbols (CLI flags identical). When a user runs `streamlz_vk.exe compress in.bin out.slz`, the binary prints `streamlz_vk: error: compress operation requires VK encode kernels (waiting on M27); use streamlz.exe for production compression` to stderr and exits with code 2 — never crashes, never silently does the wrong thing.

**Acceptance criteria:**
- `streamlz_vk.dll` exports exactly 15 symbols verifiable via `dumpbin /exports` matching the header exactly.
- Symbol names: `slzCreate_vk`, `slzDestroy_vk`, `slzCompressAsync_vk`, `slzDecompressAsync_vk`, `slzCompressHost_vk`, `slzDecompressHost_vk`, `slzGetDecompressedSize_vk`, `slzCompressBound_vk`, `slzGetLastTimings_vk`, `slzWaitAndGetLastTimings_vk`, `slzStatusString_vk`, `slzVersionString_vk`, `slzCompressDefaultOpts_vk`, `slzDecompressDefaultOpts_vk`, `slzRegisterBuffer_vk`.
- `streamlz_vk.exe --version` prints `3.0.0-vk`.
- `streamlz_vk.exe compress test.txt out.slz` exits with code 2 and prints a human-readable message naming the operation, NOT a crash.
- `streamlz_vk.exe decompress in.slz out.bin` exits with code 2 and prints a human-readable message, NOT a crash.
- All 10 not-yet-implemented symbols return `SLZ_ERROR_UNSUPPORTED` from a C smoke test.
- `slzRegisterBuffer_vk` returns `SLZ_SUCCESS` as a no-op stub.
- `sizeof(slzCompressOpts_vk)` matches `sizeof(slzCompressOpts_t)` byte-for-byte (cross-backend ABI parity).

**Estimated LOC:** 700

**Risks addressed:** R4 (slzRegisterBuffer_vk 15th symbol exists from day one); shipability (binary is graceful from milestone 2 onward — no crashes, no silent failures)

**CUDA-side changes:** None.

**Notes:** This milestone is the shippability cornerstone: from M2 onward, every subsequent milestone improves what the SAME binary can do; the binary is never broken between milestones. Grafted from Plan 2 M1 + judge insight on graceful failure.

---

### M3 — Cross-backend conformance harness + golden_slz/ + "X / 350 passing" dashboard

**Depends on:** M0, M2

**Scope:** Implement `src_vulkan/tests/cross_backend_tests.zig` as the formal cross-backend gate framework. Wire it to the M0 70 golden `.slz` fixtures. Define the 350-cell matrix: 14 silesia cases × 5 levels × 5 directions = {CUDA encode → CUDA decode (sanity baseline), CUDA encode → VK decode, VK encode → CUDA decode, VK encode → VK decode, VK encode bytes == golden .slz bytes}. At M3 only the {CUDA → CUDA} sanity cells run live (70 passing); the other 280 cells are PENDING with `return error.SkipZigTest` and a `WAITING ON M{N}` log line. The harness prints a stable header on every run: `Cross-backend conformance: 70 / 350 cells passing — next milestone unblocks: M11 (walk_frame): 0 cells, M20 (lz_decode): 70 cells, M27 (frame_assemble): 210 cells`. Implements clamp-aware equality: reads `slzCompressOpts_t.effective_level_out` (from M0); if VK Tier-2 clamps L5→L3, compares against the L3 golden, not L5. Harness exits 0 when the expected-XFAIL count matches actual; exits non-zero if an unexpected XFAIL flips to PASS (would indicate a panic-shell silently succeeded — itself a bug). Wired to CI lanes A (NVIDIA — when available), E (lavapipe), B' (SwiftShader wave64).

**Acceptance criteria:**
- `zig build test_vk` runs the harness and reports `70 / 350 cells passing — Wave 1 baseline (CUDA↔CUDA sanity)`.
- `tests/golden_slz/` contains exactly 70 committed `.slz` files (14 × L1..L5), total repo size ≤ a few MiB.
- Each golden `.slz` round-trips through the CUDA decoder cleanly (sanity baked into a `build_goldens.zig` regenerator).
- Clamp-aware equality logic exists: if `effective_level_out != requested_level`, comparison uses the clamped level on both sides.
- Harness has a printed "next milestone unblocks" table.
- Harness exit code is 0 when only expected-XFAIL failures occur.
- Harness exit code is NON-zero if an XFAIL cell unexpectedly PASSES.
- CI lane E (lavapipe) runs the harness even though all VK cells are PENDING.
- SHA-256 manifest in `tests/golden_slz/checksums.txt` matches M0's manifest exactly.

**Estimated LOC:** 1100

**Risks addressed:** R3 (BLOCKER — wire-format byte-identity regression caught at the kernel that introduces it, not after)

**CUDA-side changes:** None (consumes M0 fixtures).

**Notes:** Plan 2's M2 dashboard idea grafted in: every subsequent milestone produces a visible numeric delta in the conformance count, making progress quantitative. Plan 1's XFAIL-flips-to-PASS detection grafted in to catch panic-shell-silently-succeeds bugs.

---

### M4 — Vulkan loader + vendor probe + tier classification (RETIRES R2)

**Depends on:** M2

**Scope:** Implement `src_vulkan/host/vk_loader.zig` (VkInstance + VkPhysicalDevice enumeration + VkDevice creation with single compute queue per arch §7 locked decision #5, optional validation gated by `SLZ_VK_VALIDATION=1`) and `vk_errors.zig` (VkResult → slzStatus_t translation). Implement `vk_probe.zig`: dedicated 18th `.comp` file at `src_vulkan/shaders/common/probe.comp` that writes `{gl_SubgroupSize, subgroupBroadcastFirst(gl_LocalInvocationID.x), shuffle_xor_test_result, shuffle_relative_test_result}` to a host-visible SSBO. vk_probe dispatches the probe at `slzCreate_vk` and applies tier-classification: if subgroupSize == 32 AND subgroup ops behave AND vendor is in whitelist (NVIDIA Turing+, AMD RDNA2+, Intel Arc Alchemist+) → Tier-1, else Tier-2. SLZ_VK_FORCE_TIER env override. Mali (all gens) hardcoded to Tier-2 regardless of probe result. Build a CI matrix that runs the probe on lane E (lavapipe), lane B' (SwiftShader wave64), AND at least one borrowed real mobile device (Adreno or Mali via adb-attached CI worker if available; logged-only otherwise). If any Tier-1 whitelisted vendor fails the probe at runtime, file an arch-revision blocker BEFORE Wave 1 starts. Pre-quantize sc_group_size host-side (0.25 → 65536, 0.5 → 131072) so f32 never crosses the SPIR-V boundary (R10 mitigation).

**Acceptance criteria:**
- `slzCreate_vk` returns `SLZ_SUCCESS` on any system with Vulkan 1.2+ and a compute queue with `timestampValidBits > 0`.
- `slzCreate_vk` returns `SLZ_ERROR_UNSUPPORTED_DEVICE` with explanatory log when no compute queue or Vulkan 1.2 floor not met.
- Probe kernel compiles and runs on lavapipe; returns sensible `gl_SubgroupSize`.
- Tier classification produces correct result for: NVIDIA RTX (Tier-1), AMD RDNA2 reporting 32 (Tier-1), AMD RDNA2 reporting 64 (Tier-2), Mali-G78 (Tier-2 always), Adreno 740 (Tier-2 unless probe reports 32), lavapipe (Tier-2), SwiftShader (Tier-2).
- `SLZ_VK_FORCE_TIER=1` and `=2` env overrides work end-to-end (verified by probe logging chosen tier).
- `SLZ_VK_FORCE_TIER=1` on a Tier-2-only device returns `SLZ_ERROR_UNSUPPORTED_DEVICE` with `"forced Tier-1 unavailable: missing subgroup_size_control"` message (no silent downgrade).
- Probe results from at least one real mobile borrow recorded in `docs/vulkan_port_architecture.md` appendix; any Tier-1 whitelist violation files a blocker issue BEFORE M11.
- sc_group_size pre-quantization: host-side integer products 65536 / 131072 used; no f32 in SPIR-V (verified by `spirv-dis | Select-String "OpTypeFloat"` returning zero matches in any committed SPV).
- `slzDestroy_vk` reports zero leaks under `SLZ_VK_VALIDATION=1`.

**Estimated LOC:** 1100

**Risks addressed:** R2 (subgroupSize=32 unsupportable on mobile — caught at runtime probe, not at integration), R10 (sc_group_size float→uint determinism)

**CUDA-side changes:** None.

**Notes:** Plan 3's early-probe insight grafted in: real-mobile probe runs as part of this milestone, not deferred to a late lane-F milestone. Tier-1 whitelist violations file blockers BEFORE any kernel ships, eliminating late-cycle architectural surprises.

---

### M5 — `__match_any_sync` emulation standalone microbenchmark on AMD/Intel (RETIRES R1)

**Depends on:** M4

**Scope:** Port BOTH match_any sites as standalone `.comp` shaders BEFORE integrating into any production kernel. Site #3 (11-iter ballot loop, domain [1..11], used in huff_build_lut): standalone shader takes a synthetic Huffman-shaped input (256-symbol histograms × N batches) and produces a "match groups" output array. Site #1 (32-broadcast OR-reduction, used in lz_encode hash store): standalone shader takes a synthetic encode-shaped workload (32-lane hash buckets × N batches). Implement BOTH the Tier-1 base variant (subgroup_basic + ballot + vote) AND the tier1_nv variant (subgroupPartitionedXorNV fast path). Build a microbenchmark host harness that runs each kernel on every CI lane and measures throughput in ns/dispatch. Compare AMD/Intel emulation throughput to a CUDA reference baseline running the same workload (the CUDA reference uses `__match_any_sync` directly). **EXPLICIT NUMERIC ACCEPTANCE THRESHOLD: AMD throughput ≥ 65% of CUDA reference AND Intel throughput ≥ 65% of CUDA reference for BOTH sites.** If either site on either vendor falls below 65%, this milestone fails and triggers an architecture revision (promote NV_partitioned-equivalent fast path from "opportunistic" to "required for Tier-1 AMD/Intel parity", or revise tier matrix) BEFORE M11 (first kernel port) starts.

**Acceptance criteria:**
- Site #3 standalone kernel produces byte-identical "match groups" output to a CUDA reference on 100 synthetic Huffman inputs.
- Site #1 standalone kernel produces byte-identical highest-lane-winner election to a CUDA reference on 100 synthetic encode inputs.
- NV_partitioned variants for both sites produce byte-identical output to base variants on NVIDIA.
- Microbenchmark reports ns/dispatch on lane A (NVIDIA), lane B (AMD), lane C (Intel Arc) — if hardware unavailable, lane is recorded as `UNTESTED` and a blocker issue is filed for hardware acquisition.
- AMD throughput vs CUDA: ≥ 65% on BOTH sites — if missed, file `streamlz-vk/issues/arch-revision-r1-amd` and pause M11 until resolved.
- Intel throughput vs CUDA: ≥ 65% on BOTH sites — if missed, file `streamlz-vk/issues/arch-revision-r1-intel` and pause M11 until resolved.
- Per-vendor perf numbers committed to `tests/perf/match_any_baseline.json` with hardware ID (vendor + device + driver version + glslc version).
- Decision documented in `docs/vulkan_port_architecture.md` §7 with measured numbers per vendor per site.

**Estimated LOC:** 900

**Risks addressed:** R1 (match_any emulation perf regression on AMD/Intel — caught at milestone 5 of 31 with pre-committed numeric threshold, NOT at milestone 18-22 deep in sunk cost)

**CUDA-side changes:** None (uses M0's rewritten CUDA encoder as the reference behavior).

**Notes:** This is the most important graft from Plan 3. The standalone microbenchmark with a pre-committed 65% threshold and explicit arch-revision trigger is the abort signal that prevents 10-15 milestones of wasted work if AMD/Intel match_any emulation is fundamentally slow. Both site #1 AND site #3 are measured (Plan 3 only emphasized site #3; we add site #1 because it's the encode hot loop).

---

### M6 — BDA + shaderInt64 feature probe + slzRegisterBuffer_vk real implementation (RETIRES R4)

**Depends on:** M4

**Scope:** Promote `slzRegisterBuffer_vk` from the M2 no-op stub to its real implementation: on Tier-1 it's a no-op that returns the input pointer unchanged; on Tier-2 it registers the VkBuffer + VkDeviceMemory + offset triple in a host-side hashmap keyed by the returned opaque `slzDeviceBuffer_t` handle, which is what callers pass into all other `_vk` APIs. Add a buffer_reference round-trip test: small `.comp` shader that takes a `uint64_t` BDA via push constant, dereferences it, writes a sentinel value back. Run on every Tier-1 lane (NVIDIA, AMD, Intel) to prove BDA + shaderInt64 work end-to-end. Build a Tier-2 round-trip test that uses descriptor-set bindings instead of BDA, dispatched against the same logical input via `slzRegisterBuffer_vk`. shaderInt64 enabled at VkDevice creation when Tier-1; falls back to uvec2 emulation typedef in `tier_gates.glsl` when Tier-2.

**Acceptance criteria:**
- BDA round-trip kernel reads sentinel via `buffer_reference` push constant on NVIDIA Tier-1 lane (host-side memcmp passes).
- Same on AMD RDNA2+ Tier-1 lane (when subgroupSize=32 confirmed).
- Same on Intel Arc Alchemist+ Tier-1 lane.
- `slzRegisterBuffer_vk` on Tier-1 returns input pointer unchanged (no-op), validated via `SLZ_VK_FORCE_TIER=1` on a Tier-1 lane.
- `slzRegisterBuffer_vk` on Tier-2 stores buffer+memory+offset; descriptor-set round-trip kernel reads back the sentinel via the registered handle.
- `shaderInt64` enabled at VkDevice creation when Tier-1; uvec2 emulation typedef active when Tier-2 (verified by SPIR-V cross-reflect).
- glslc version asserted ≥ 1.3.250 by `tools/build_glsl.bat` preamble — regression test that downgrading to 1.3.249 fails the build with a named error.

**Estimated LOC:** 800

**Risks addressed:** R4 (BDA absent breaks void* device-pointer ABI — slzRegisterBuffer_vk 15th symbol is real and validated on both tiers)

**CUDA-side changes:** None.

**Notes:** Parallel to M5 (both depend on M4 only). Plan 1's BDA round-trip test grafted.

---

### M7 — Memory allocator with 2-GiB VkBuffer split (RETIRES R5)

**Depends on:** M6

**Scope:** Implement `src_vulkan/host/vk_memory.zig` as a slab-based suballocator (no VMA per arch §0). Three pools: DEVICE_LOCAL (encode/decode scratch + I/O), HOST_VISIBLE|HOST_COHERENT (staging + readback), and HOST_CACHED|HOST_VISIBLE if available (R6 fast path; falls back to coherent + staging copy when missing). Slab size: 64 MiB; sub-allocations aligned to `nonCoherentAtomSize` for host-visible. Free-list per slab, no per-allocation `vkFreeMemory`. Expose `alloc(size, usage, memory_property_hints) → {VkBuffer, VkDeviceMemory, VkDeviceAddress (if BDA), host_ptr (if visible)}`. Stand up the encode hash-table buffer-split logic: compute the maximum hash table size for L5 on a 24GB input (~3.5 GiB); implement bin-packing so no single sub-chunk's hash crosses a buffer boundary; split across N ≤2 GiB VkBuffers. Build a synthetic stress test that allocates N=3 hash buffers totalling >5 GiB on a Tier-1 device (skipped if device VRAM <8 GB), writes a sentinel pattern via a synthetic kernel that BDA-addresses across buffer boundaries via push-constant table, reads back, validates. Document the split math in `docs/vulkan_port_architecture.md` §6 with worked example for 24 GB input at L5.

**Acceptance criteria:**
- `vk_memory.allocDevice(size)` and `allocHost(size)` return distinct ranges that never alias (verified by guard pattern).
- Round-trip test: HOST_VISIBLE → DEVICE_LOCAL via `vkCmdCopyBuffer` → HOST_VISIBLE, byte-equal to original.
- HOST_CACHED fallback path on a device without HOST_CACHED|HOST_VISIBLE: allocator uses HOST_VISIBLE|HOST_COHERENT and reports `cached_readback=false` in a status struct.
- 10k alloc/free cycles complete without crash; fragmentation metric (largest free run / total free) reported.
- Bin-packing function: given (input_size, level), returns list of (sub_chunk_id, buffer_id, offset) tuples that satisfy ≤2 GiB per VkBuffer and no sub-chunk straddles a boundary.
- Synthetic kernel allocates 3 × 2 GiB buffers on a 24 GB-VRAM device, addresses all 3 via BDA push-constant table, writes pattern, host readback validates pattern.
- On devices with VRAM < 8 GB: test is skipped with `INFO: 2-GiB stress test skipped on <8GB device` log, NOT failing CI.
- Tier-2 descriptor-set layout supports N=8 buffers worst-case (mobile clamp ceiling), validated by binding 8 buffers and dispatching a hello-world kernel that reads from binding 7.
- Split math documented in `docs/vulkan_port_architecture.md` §6.
- Zero validation-layer leaks under `SLZ_VK_VALIDATION=1`.

**Estimated LOC:** 1100

**Risks addressed:** R5 (encode hash table >2 GiB single VkBuffer — bin-packing validated on synthetic load), R6 (HOST_CACHED fallback)

**CUDA-side changes:** None.

**Notes:** Plan 3's standalone 2-GiB stress test grafted.

---

### M8 — Pipeline cache + sync (timeline/binary) + timing (VkQueryPool) chassis

**Depends on:** M5, M7

**Scope:** Implement `src_vulkan/host/vk_pipeline.zig`: process-wide VkPipelineCache per `(deviceUUID, tier)` persisted to `%LOCALAPPDATA%/Cache/streamlz_vk/pipelines_{UUID}_t{tier}.bin` (arch §7 locked decision #9). Per-kernel pipeline created lazily on first dispatch; uses the 18-from-M1+1-from-M4-probe = 18 panic-shell SPVs. Tier-1 pipelines set `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` with `requiredSubgroupSize=32` and `FullSubgroups` bit. Descriptor-set-layout factory: Tier-2 builds worst-case layout sized at handle create (uses M7's bin-pack metadata). Implement `vk_command.zig` primary command buffer recording + push-constant upload + dispatch helper. Implement `vk_sync.zig`: Tier-1 uses VkTimelineSemaphore + `vkCmdPipelineBarrier2` (synchronization2); Tier-2 uses VkBinarySemaphore + VkFence + `vkCmdPipelineBarrier` (sync1 fallback). Implement `vk_timing.zig`: VkQueryPool TIMESTAMP sized 2×N; `beginKernel`/`endKernel` write TOP_OF_PIPE/BOTTOM_OF_PIPE timestamps (locked decision #12); `drainTimings()` reads back + applies `timestampPeriod` + fills caller's `slzKernelTiming_t[]`. Tier-1 uses `vkCmdWriteTimestamp2`; Tier-2 falls back to `vkCmdWriteTimestamp`. VK_EXT_host_query_reset (Vulkan 1.2 core) or `vkCmdResetQueryPool` fallback.

**Acceptance criteria:**
- Cold start (no cache file): all 18 pipelines compile + cache file written in <500 ms on release build.
- Warm start (cache present): all 18 pipelines load in <10 ms (target 3-5 ms).
- Cache file naming exactly matches arch §7: `%LOCALAPPDATA%\Cache\streamlz_vk\pipelines_<UUID>_t<tier>.bin`.
- Different deviceUUID generates a separate cache file.
- Tier-1 pipelines stamp `requiredSubgroupSize=32` (verified by `SLZ_VK_VALIDATION=1` info messages).
- Timeline-semaphore path: submit two empty cmd buffers signalling values 1 then 2, wait on value 2 — both fences cleared.
- Binary-sem+fence path forced via env override: same test, identical externally-observable behavior.
- VkQueryPool: dispatching a no-op compute pipeline + `drainTimings()` returns a delta > 0 ns and < 100 ms.
- `vkCmdWriteTimestamp2` path active when synchronization2 enabled; `vkCmdWriteTimestamp` path active when not.
- Cache file is corruption-tolerant: deleting → cold rebuild, no crash; truncating to 17 bytes → cold rebuild, no crash.
- Each empty `.comp` dispatch returns `SLZ_ERROR_UNSUPPORTED` with the kernel NAME embedded in `slzStatusString_vk` (NOT a sentinel write, NOT a crash).

**Estimated LOC:** 1500

**Risks addressed:** R7 (VkPipelineCache cold-start 100-500ms), R8 (cross-stream barrier elision — single VkQueue + explicit per-buffer barriers)

**CUDA-side changes:** None.

**Notes:** Graceful-named-error from dispatched panic-shell grafted from Plan 2 M4 — the binary fails informatively per kernel between M8 and the milestone that ports each one.

---

### M9 — Common GLSL helpers (byteio, huffman, warp_t1, warp_t2, match_any, wire_format, bda, ballot_mask, byte_perm) + per-primitive unit tests

**Depends on:** M8

**Scope:** Write all shared GLSL include files per arch §2 common/:
- `byteio.glsl` — 8-bit SSBO fast path (gated on VK_KHR_8bit_storage) + shift/mask fallback on `uint data[]`.
- `huffman.glsl` — 32-stream BIL layout helpers (wire-locked: 228 B header + K × 128 B interleaved rows + tail prefix-sum).
- `warp_t1.glsl` — subgroup intrinsic lowerings (subgroupBroadcast, subgroupExclusiveAdd, subgroupShuffle, subgroupShuffleXor, subgroupBallot, subgroupBallotBitCount, subgroupAny/All).
- `warp_t2.glsl` — shared-mem workgroup-wide emulations (32 logical lanes, always uses workgroup `barrier()`).
- `warp.glsl` — selects via `#if TIER == 1`.
- `match_any.glsl` — site #1 (32-broadcast OR-reduction) + site #3 (11-iter ballot loop, domain [1..11]) + NV_partitioned fast path for site #1 under `USE_NV_PARTITIONED`. Site #2 ELIMINATED per arch §10 (replaced by `subgroupShuffle(key4, highest_lower) == key4`, byte-identical).
- `wire_format.glsl` — constants matching `src/format/` (HUFF_NUM_STREAMS=32, magic 0x534C5A31, version 2, codec 1, log2_block_size, sc_group_size integer products 65536/131072).
- `bda.glsl` — `buffer_reference` declarations on T1, SSBO bindings on T2.
- `ballot_mask.glsl` — wave64-safe ballot helper (always returns 32-bit lower half).
- `byte_perm.glsl` — explicit `__byte_perm(w, 0, 0x0123)` lowering.
- `tier_gates.glsl` — TIER=1/2 macros, U64 typedef (uint64_t on T1, uvec2 on T2).

Unit-test each helper via a tiny `.comp` shader that exercises it and writes results to a readback SSBO. Build `src_vulkan/tests/byteio_unit_tests.zig` exhaustively testing byteio primitives at every alignment 0..3 and length 1..256 on Tier-1 and Tier-2.

**Acceptance criteria:**
- `byteio_unit_tests.zig` round-trips every byteio primitive (read_u8, read_u16, read_u24_be, write_u8, write_u32_le) on Tier-1 and Tier-2 byte-for-byte against the CUDA `gpu_byteio.cuh`.
- `warp_test.comp` on Tier-1: subgroupBroadcast, subgroupExclusiveAdd, subgroupShuffleXor match expected values for input 1..32.
- Same `warp_test.comp` on Tier-2 produces identical results via shared-mem emulation.
- `match_any_unit_test.comp` site #3 ballot loop on a synthetic [hash array]: returns same mask as reference CPU implementation across 1000 random inputs.
- `match_any_unit_test.comp` site #1 32-broadcast OR-reduction: returns highest-indexed lane sharing bucket, matches reference (consistent with M5 standalone benchmark output).
- `byte_perm_unit_test`: 256 random uint inputs through `__byte_perm(w, 0, 0x0123)` match Zig-side reference.
- `ballot_mask` helper produces a 32-bit value even when run on a wave64 SwiftShader lane.
- `wire_format.glsl` constants enforced via SPIR-V SpecConstant checks at pipeline create.

**Estimated LOC:** 1300

**Risks addressed:** R1 (match_any emulations validated standalone with correctness gate — perf gate already retired in M5), R10 (wire constants byte-identity at GLSL header level)

**CUDA-side changes:** None.

**Notes:** Plan 1 M13 grafted whole — dedicated helpers milestone with per-primitive unit tests prevents the implicit-helper-exists assumption that plagues Plan 2.

---

### M10 — Hello-dispatch substrate gate (end-to-end smoke kernel + buffer round trip)

**Depends on:** M3, M8, M9

**Scope:** Add `src_vulkan/shaders/test/hello.comp` (separate test-only shader, NOT one of the 17 production kernels) that reads an SSBO of N uints and writes N+1 to each. Plumb `src_vulkan/tests/hello_dispatch_test.zig`: allocate HOST_VISIBLE input, write 1..N, allocate DEVICE_LOCAL output, copy via M7, dispatch via M8, copy back, byte-check `output[i] == i+2`. Time it via M8 timing. Wave-64 lane via SwiftShader (`VK_SUBGROUP_SIZE=64` emulation) also runs. This is the proof that all substrate (M4-M9) works end-to-end before we touch any production kernel.

**Acceptance criteria:**
- Hello-dispatch test passes on Tier-1 device.
- Hello-dispatch test passes on Tier-2 device (forced via `SLZ_VK_FORCE_TIER=2`).
- M8 timing reports a finite, sensible duration (>0 and <100 ms for N=1M).
- Repeated invocations (1000×) do not leak memory, descriptor pools, or pipeline-cache slots (validation layer clean under `SLZ_VK_VALIDATION=1`).
- Wave-64 lane via SwiftShader passes.
- M3 conformance harness still reports `70 / 350 cells passing` (no regressions).

**Estimated LOC:** 250

**Risks addressed:** Substrate end-to-end proof — gates everything above before kernel porting starts.

**CUDA-side changes:** None.

**Notes:** Plan 1 M12 grafted. Bottom-up's substrate-end-to-end-proof discipline before kernel work begins.

---

### M11 — Wave 1 kernel 1/10: walk_frame.comp (D1) — BDA/push-constant/dispatch chassis validation

**Depends on:** M0, M10

**Scope:** Port `src/decode/walk_frame_kernel.cuh` (slzWalkFrameKernel) to `src_vulkan/shaders/decode/walk_frame.comp`. Single-thread orchestration kernel — reads frame header via `byteio.glsl`, walks the chunk-descriptor table, writes per-chunk metadata to an output SSBO. `local_size_x=32` (only thread 0 active). Tier-1 uses BDA via `bda.glsl`; Tier-2 uses descriptor-set bindings via `slzRegisterBuffer_vk`. Push constants carry: input_addr (or descriptor index), input_size, output_addr, max_chunks. Update `host/decode/decode_dispatch_vk.zig` (new file) to invoke it. Add stage-level assertion to M3 conformance harness that compares walk_frame intermediate output to CUDA byte-for-byte.

**Acceptance criteria:**
- `walk_frame.comp` dispatch on Tier-1 (NVIDIA) executes without validation errors under `SLZ_VK_VALIDATION=1`.
- Same on Tier-2 (lavapipe).
- Push-constant block size ≤ 128 bytes verified at pipeline create.
- Stage-level conformance test: walk_frame output (per-chunk descriptor table) byte-identical to CUDA on all 70 golden fixtures, both tiers.
- M3 harness: walk_frame stage cell flips to PASS on both tier rows; conformance header updates to `X / 350 cells passing` (no full-decompress cells unblocked yet).
- No descriptor-set bindings on Tier-1 (BDA-only); 1 SSBO binding on Tier-2.

**Estimated LOC:** 500

**Risks addressed:** R4 (BDA path real-world validated end-to-end with a production kernel)

**CUDA-side changes:** None (M0 already landed).

**Notes:** Explicit M0 dep cited (Plan 1 weakness fixed) — walk_frame doesn't consume the hash-store rewrite directly but the entire decode chain requires M0-rewritten goldens.

---

### M12 — Wave 1 kernel 2/10: lz_decode_raw.comp — simplest parallel path, byteio validation

**Depends on:** M11

**Scope:** Port `src/decode/lz_decode_raw.cuh` (slzLzDecodeRawKernel — raw-block fast path) to `src_vulkan/shaders/decode/lz_decode_raw.comp`. Tier-1: `local_size_x=64` with per-subgroup `subgroupBarrier()`. Tier-2: `local_size_x=32` with 2× workgroup dispatch (collapse policy per arch locked decision #3, mitigates R9). Reads raw-block descriptors and copies bytes from compressed input to decompressed output via `byteio.glsl`. No subgroup ops — proves byteio fast path and per-tier workgroup-size dispatch policy.

**Acceptance criteria:**
- Synthetic 16 KiB raw-only frame decodes byte-identical to CUDA's lz_decode_raw output.
- Tier-1: 64-thread workgroup confirmed (`gl_WorkGroupSize.x` written to debug SSBO).
- Tier-2: 32-thread workgroup confirmed, 2× workgroup dispatch verified by SPIR-V cross-reflect.
- Stage assertion in M3 harness: lz_decode_raw output byte-identical to CUDA on all 70 fixtures both tiers.
- Throughput ≥ 50% of CUDA on the same device class (sanity floor; no firm target).
- `byteio_unit_tests.zig` from M9 still passes (regression check).
- `VkBufferMemoryBarrier2` between walk_frame and lz_decode_raw present and validated by sync2 layer.

**Estimated LOC:** 500

**Risks addressed:** R9 (LZ-decode 2-warp coupling: Tier-2 collapse policy verified)

**CUDA-side changes:** None.

**Notes:** —

---

### M13 — Wave 1 kernel 3/10: prefix_sum_chunks.comp — first subgroup ExclusiveAdd in production

**Depends on:** M12

**Scope:** Port `src/decode/prefix_sum_chunks_kernel.cuh` to `src_vulkan/shaders/decode/prefix_sum_chunks.comp`. Uses `subgroupExclusiveAdd` on Tier-1; uses `warp_t2.glsl` shared-mem emulation on Tier-2. First production kernel exercising subgroup arithmetic — proves `layout(subgroup_size=32) in;` works on real Tier-1 hardware AND that workgroup-wide barrier emulation works on Tier-2.

**Acceptance criteria:**
- Input: 1024 random chunk sizes; exclusive-prefix-sum output matches CPU reference byte-for-byte on Tier-1 AND Tier-2.
- Tier-1 SPIR-V dump: `OpExecutionMode LocalSizeId` with `required_subgroup_size_id=32`.
- Tier-2 SPIR-V dump: no required_subgroup_size mode.
- Edge cases: 1 chunk, 32 chunks (subgroup boundary), 33 chunks (crosses), 1024 chunks all handle correctly.
- Wave-64 SwiftShader lane (Tier-2 path) produces correct output (validates wave-size-agnostic emulation).
- Stage assertion in M3 harness: prefix-sum chunks output byte-identical to CUDA on all 70 fixtures both tiers.

**Estimated LOC:** 400

**Risks addressed:** Tier-1 subgroup_size pinning validation in production (R1 partial — perf already validated in M5).

**CUDA-side changes:** None.

**Notes:** —

---

### M14 — Wave 1 kernel 4/10: scan_parse.comp (D2) — FIRST BYTE-IDENTITY GATE

**Depends on:** M13

**Scope:** Port `src/decode/scan_parse_kernel.cuh` (slzScanParseKernel) to `src_vulkan/shaders/decode/scan_parse.comp`. Single-thread orchestration; only thread 0 active. Reads 3-byte big-endian sub-chunk headers, off16 hi/lo split with 0xFFFF marker, off32 3/4-byte entries, SC tail prefix table. First wire-format-heavy kernel — every endianness or alignment bug surfaces here. M3 conformance harness HARD-FAILS if scan_parse stage diverges; emits per-case diff report to `tests/diffs/{case}_L{n}_scan_parse.diff`. Do not advance to M15 until clean.

**Acceptance criteria:**
- Stage assertion: scan_parse output (huff_descs + raw_descs intermediate arrays) byte-identical to CUDA on all 70 (case, level) pairs on lane A (Tier-1 NVIDIA).
- Same on lane E (Tier-2 lavapipe).
- Same on lane B' (Tier-2 SwiftShader wave64).
- sc_group_size handling: L1 frame (integer 131072) and L3 frame (integer 65536) both parse to identical descriptors as CUDA — no f32 anywhere.
- 0xFFFF off16 marker: synthetic frame with off16=0xFFFF forces fallback to off32 read; VK and CUDA produce identical offsets.
- Validation layer clean on all 14 cases.
- ANY divergence: per-case diff committed to `tests/diffs/`, milestone FAILS, root-cause before advancing.

**Estimated LOC:** 800

**Risks addressed:** R3 (FIRST production byte-identity gate — wire-format quirks surfaced before they propagate downstream)

**CUDA-side changes:** None.

**Notes:** Plan 2's "first byte-identity gate" tripwire discipline grafted — explicit diff emission and HARD-FAIL behavior.

---

### M15 — Wave 1 kernel 5/10: compact_huff_descs.comp + compact_raw_descs.comp (D4/D5 paired)

**Depends on:** M14

**Scope:** Port both paired compaction kernels in one milestone per arch §11 step 5. Both are single-thread orchestration kernels (`local_size_x=32`, thread-0-only) that filter the descriptor table from M14 into separate huff-block and raw-block lists. Both consume the same scan_parse output and feed the next stage in parallel — paired because they're a fan-out point.

**Acceptance criteria:**
- Stage assertion: compact_huff_descs output byte-identical to CUDA on all 70 fixtures both tiers.
- Stage assertion: compact_raw_descs output byte-identical to CUDA on all 70 fixtures both tiers.
- Both kernels can be dispatched concurrently (a barrier between them and the consumer, but not between them) — verified by checking M8's `encodeBarrier` emission.
- Partitioning math: huff_count + raw_count == total_descriptors for every fixture.

**Estimated LOC:** 450

**Risks addressed:** R3 (continues byte-identity gate)

**CUDA-side changes:** None.

**Notes:** —

---

### M16 — Wave 1 kernel 6/10: gather_raw_off16.comp — parallel raw-block branch

**Depends on:** M15

**Scope:** Port `src/decode/gather_raw_off16_kernel.cuh` to `src_vulkan/shaders/decode/gather_raw_off16.comp`. Reads raw descriptors from M15 and off16 table, gathers raw-block payload bytes into a packed output. `local_size_x=64` on Tier-1, 32 on Tier-2. Validates descriptor → SSBO gather path under workgroup-wide barrier on Tier-2.

**Acceptance criteria:**
- Stage assertion: gather output byte-identical to CUDA on all 70 fixtures both tiers.
- Stress test: a frame with 10k raw blocks of varying sizes (8B-512B) does not deadlock on Tier-2 workgroup-wide barriers.
- Partial-chain harness {walk_frame + scan_parse + compact + gather_raw_off16 + lz_decode_raw} passes for raw-only synthetic frames — first full-decompress cells start flipping to PASS (~5 / 350 added for synthetic raw cases).

**Estimated LOC:** 350

**Risks addressed:** R9 (workgroup-wide barrier stress on Tier-2)

**CUDA-side changes:** None.

**Notes:** —

---

### M17 — Wave 1 kernel 7/10: merge_huff_descs.comp (D7) — single-thread orchestration

**Depends on:** M16

**Scope:** Port `src/decode/merge_huff_descs_kernel.cuh` to `src_vulkan/shaders/decode/merge_huff_descs.comp`. Single-thread orchestration that merges adjacent huff-block descriptors with compatible parameters into super-blocks for the LUT-build stage. `local_size_x=32`, thread-0-only.

**Acceptance criteria:**
- Stage assertion: merged descriptor list byte-identical to CUDA on all 70 fixtures both tiers.
- Per-stream descriptor count (32 streams per locked HUFF_NUM_STREAMS) validated.
- Edge case: zero huff blocks → output list of length 0, no crash.
- Edge case: 1 huff block → output list of length 1, identical to input.

**Estimated LOC:** 350

**Risks addressed:** R3 (continues byte-identity gate)

**CUDA-side changes:** None.

**Notes:** —

---

### M18 — Wave 1 kernel 8/10: huff_build_lut.comp — first __match_any_sync site #3 in production

**Depends on:** M5, M17

**Scope:** Port the LUT-build portion of `src/decode/huffman_kernel.cu` to `src_vulkan/shaders/decode/huff_build_lut.comp`. THIS IS THE FIRST PRODUCTION USE of __match_any_sync site #3 (the 11-iter ballot loop, domain [1..11]) from `match_any.glsl`. M5 already validated emulation correctness AND perf vs CUDA on AMD/Intel; this milestone integrates it into a real, byte-identity-critical kernel. NV_partitioned variant exercised on NVIDIA lane A. Explicit M5 dep cited (Plan 1/Plan 3 weakness fixed).

**Acceptance criteria:**
- Stage assertion: LUT output byte-identical to CUDA on all 70 fixtures + 10 hand-crafted edge cases (1 symbol, 256 symbols, max depth 11, all-uniform depths).
- Tier-1 + Tier-2 + Tier-1+NV_partitioned (when available) all produce byte-identical output.
- Performance sanity (Tier-1 NVIDIA): kernel completes in <50 µs on a small frame.
- Performance (Tier-1 AMD): ≥ 65% of CUDA reference (consistent with M5 baseline).
- Performance (Tier-1 Intel): ≥ 65% of CUDA reference.
- NV_partitioned variant selected on NVIDIA at runtime (verified by VkPipelineCache key inspection or log).
- 11-iter ballot loop verified via SPIR-V disassembly (correct iteration count, bounded loop).

**Estimated LOC:** 600

**Risks addressed:** R1 (match_any site #3 in production — perf already gated in M5)

**CUDA-side changes:** None.

**Notes:** Explicit M5 dep — first kernel that consumes the standalone match_any work. If production perf drops below M5's measured baseline, attribute to integration overhead, not emulation cost.

---

### M19 — Wave 1 kernel 9/10: huff_decode_4stream.comp — hottest decode kernel

**Depends on:** M18

**Scope:** Port the 4-stream Huffman decode portion of `src/decode/huffman_kernel.cu` to `src_vulkan/shaders/decode/huff_decode_4stream.comp`. Reads M18's LUT and the 32-stream BIL Huffman body (wire-locked: 228 B header + K × 128 B interleaved rows + tail prefix-sum from `huffman.glsl`). `local_size_x=32` on both tiers; 4 streams × 8 symbols-per-iter pattern preserved from CUDA. Uses `byteio.glsl` for bitstream reads. Hottest kernel in decode — perf-critical.

**Acceptance criteria:**
- Stage assertion: literal stream + length stream + offset stream byte-identical to CUDA on all 70 fixtures + 5 Huffman-heavy frames (high-entropy text).
- Tier-1 + Tier-2 + Tier-1+NV_partitioned all produce byte-identical output.
- Bitstream read crosses 32-stream boundary correctly (frame with body exactly 32×K bytes for K=1,2,4,8).
- Performance: Tier-1 NVIDIA ≥ 75% of CUDA reference on 16 MiB compressed payload.
- Below 60% triggers perf review.
- Perf numbers committed to `tests/perf/huff_decode_4stream_baseline.json` with hardware ID.

**Estimated LOC:** 1100

**Risks addressed:** R1 (hottest decode — match_any + warp_t1 perf both stressed), R3 (Huffman wire-format byte-identity)

**CUDA-side changes:** None.

**Notes:** —

---

### M20 — Wave 1 kernel 10/10: lz_decode.comp — closes CUDA-encode → VK-decode roundtrip

**Depends on:** M19

**Scope:** Port `src/decode/lz_decode_kernels.cuh` to `src_vulkan/shaders/decode/lz_decode.comp`. Tier-1 `local_size_x=64` with per-subgroup `subgroupBarrier()`; Tier-2 `local_size_x=32` collapsed + 2× workgroups. Consumes LUT-decoded Huffman tokens from M19 and reconstructs decompressed output via LZ match/literal copy loop. THIS IS THE LAST DECODE KERNEL — its landing closes the {CUDA encode → VK decode} roundtrip for every level L1..L5. Conformance count jumps from ~50 stage assertions to 70 full-decompress assertions in one shot.

**Acceptance criteria:**
- Full decode pipeline (M11-M20 chained) produces byte-identical output to CUDA decoder on all 14 silesia files at L1..L5 on lane A (NVIDIA).
- Same 70 / 350 passing on lane E (lavapipe Tier-2).
- Same 70 / 350 passing on lane B' (SwiftShader wave64).
- Tier-1 2-warp coupling: `subgroupBarrier()` between two subgroups in 64-lane workgroup synchronizes correctly under `SLZ_VK_VALIDATION=1`.
- Tier-2 2× workgroup dispatch verified by SPIR-V cross-reflect; output identical to Tier-1.
- Performance: Tier-1 decode throughput ≥ 50% of CUDA on RTX 4090.
- ANY case fails: emit per-case diff; FAIL the milestone; do not proceed to M21.

**Estimated LOC:** 1100

**Risks addressed:** R9 (LZ-decode 2-warp coupling FULLY validated in production), R3 (closes CUDA→VK decode byte-identity for all levels)

**CUDA-side changes:** None.

**Notes:** —

---

### M21 — WAVE-1 GATE: cross-backend {CUDA-encode → VK-decode} hard-fail at L1..L5 on Tier-1 + Tier-2

**Depends on:** M20

**Scope:** Hard gate from arch §11 between Wave 1 and Wave 2. The {CUDA encode → CUDA decode} sanity cells (70) and {CUDA encode → VK decode} cells (70) are LIVE; the other 210 cells (VK encode → either) remain PENDING because encode kernels haven't landed. The gate requires VK decode to pass at L1..L5 on both tiers across all 14 golden fixtures. Promote M3's harness from XFAIL-tolerant to HARD-FAIL on decode-path cells. Add lavapipe CI lane to every PR; add SwiftShader wave64 lane. Both lanes must be green. Tag commit `wave1-gate-passed`. Real-mobile lanes F (Adreno+Mali) and G (AMD CDNA) are NOT yet wired — they're v1.0 release blockers tracked separately as M30.

**Acceptance criteria:**
- All four CI lanes (A NVIDIA, E lavapipe, B' SwiftShader wave64, plus a Tier-1 AMD lane B when available) report `140 / 350 cells passing — Wave 1 GATE: PASS`.
- M3 harness HARD-FAILS on decode-path cells at L1..L5 (was XFAIL-tolerant).
- Per-lane log archives committed to `tests/ci_logs/wave1_gate/{lane}_{timestamp}.log`.
- Tag commit `wave1-gate-passed` — bisectable for regressions.
- ANY lane fails: gate RED; root-cause + fix; milestone not done until all lanes green for 24 hours.
- Wave 1 perf summary committed to `docs/vulkan_port_architecture.md` appendix: decode throughput per lane vs CUDA reference.

**Estimated LOC:** 250

**Risks addressed:** R3 fully retired for CUDA → VK decode direction.

**CUDA-side changes:** None — gate only.

**Notes:** Plan 1 + Plan 2 + Plan 3 all converge on this gate. Plan 2's tag-commit-at-gate discipline grafted.

---

### M22 — Wave 2 kernel 1/6: lz_encode.comp — first __match_any_sync site #1 in production, consumes M0

**Depends on:** M0, M5, M21

**Scope:** Port `src/encode/lz_greedy_parser.cuh` (POST-M0-rewrite) to `src_vulkan/shaders/encode/lz_encode.comp`. `local_size_x=32` (matches CUDA `__launch_bounds__(32,1)`). Uses the DETERMINISTIC highest-lane-winner stores from M0 — the GLSL port lands the same explicit pattern using `match_any.glsl` site #1 (32-broadcast OR-reduction, with the Tier-1+NV_partitioned fast path validated in M5). Tier-2 variant uses shared-mem hash emulation. Bin-packed multi-buffer hash addressing from M7 wired in. Output byte-identical to CUDA at L1..L5. This is the most complex encode kernel; expect largest LOC of the project. Explicit M0 + M5 deps cited.

**Acceptance criteria:**
- LZ-encoded token stream byte-identical to CUDA on every fixture at L1, L2, L3.
- L4, L5 also byte-identical (L2+ rehash site exercises the second M0 store rewrite).
- Tier-1 NVIDIA + Tier-1 AMD + Tier-1+NV_partitioned + Tier-2 lavapipe + Tier-2 SwiftShader wave64 all produce byte-identical output.
- If AMD/Intel diverges: trace to (a) M0 incomplete, (b) site #1 emulation bug, or (c) wire-format misread; fix before declaring done.
- Performance: Tier-1 NVIDIA L1 encode ≥ 70% CUDA on RTX 4090; Tier-1 AMD ≥ 65% (R1 target consistent with M5); Tier-2 correct, no perf gate.
- Stress: 64 MiB random-ish input encodes without OOM (validates M7 bin-packing at scale).
- M3 harness: VK encode → CUDA decode and VK encode → golden bytes cells start flipping to PASS.

**Estimated LOC:** 1500

**Risks addressed:** R1 (match_any site #1 in production), R3 (encode-direction byte identity begins), R5 (encode hash >2 GiB bin-packing in production)

**CUDA-side changes:** None (M0 already landed).

**Notes:** Most graft-heavy milestone — Plan 2's explicit M0 dep + Plan 3's standalone site #1 from M5 + Plan 1's explicit M5 dep all combine here.

---

### M23 — Wave 2 kernel 2/6: huff_build_tables.comp

**Depends on:** M22

**Scope:** Port the histogram + table-build portion of `src/encode/huffman_kernel.cu` to `src_vulkan/shaders/encode/huff_build_tables.comp`. Uses `atomicAdd` on shared-mem histograms (commutative-safe per R11 mitigation). `local_size_x=32` on both tiers.

**Acceptance criteria:**
- Stage assertion: Huffman code-length tables + canonical code table byte-identical to CUDA on all 70 fixtures both tiers.
- Validated on Tier-1 NVIDIA + AMD + Tier-2 lavapipe + SwiftShader wave64.
- Mobile-relevant `atomicAdd`-on-shared-mem path validated on lane E (lavapipe simulating tile-based GPU): histograms still commutative-safe.
- Edge case: input with one unique symbol → code-length table has single entry of length 1; matches CUDA.

**Estimated LOC:** 600

**Risks addressed:** R11 (shared-mem atomicAdd commutative-safe path proven)

**CUDA-side changes:** None.

**Notes:** —

---

### M24 — Wave 2 kernel 3/6: huff_encode_4stream.comp — 32-stream BIL output

**Depends on:** M23

**Scope:** Port the bitstream-encode portion of `src/encode/huffman_kernel.cu` to `src_vulkan/shaders/encode/huff_encode_4stream.comp`. Writes the 32-stream BIL Huffman body matching M19's decode-side layout. `local_size_x=32`. `byteio.glsl` write_bits path critical.

**Acceptance criteria:**
- Stage assertion: encoded Huffman body (228 B header + K × 128 B rows + tail prefix-sum) byte-identical to CUDA on all 70 fixtures both tiers.
- Round-trip: this encoder + M19 decoder reproduces input symbol stream byte-for-byte on 100 random symbol streams.
- 32-stream boundary correctness: body of exactly 32×K bytes encodes without garbage in tail.
- Wire format header bytes hexdump-identical to CUDA reference.

**Estimated LOC:** 900

**Risks addressed:** R3 (Huffman wire-format byte-identity in encode direction)

**CUDA-side changes:** None.

**Notes:** —

---

### M25 — Wave 2 kernel 4/6: assemble_measure.comp

**Depends on:** M24

**Scope:** Port `src/encode/assemble_kernel.cu` measure portion to `src_vulkan/shaders/encode/assemble_measure.comp`. Per-block compressed-size measurement, prefix-sum to compute final offsets, total compressed_size written to a host-readable SSBO slot (this is where compressed_size_out from M0's doctrine is sourced — drained via `slzGetLastTimings_vk` after GPU completion). `local_size_x=32`. Tier-1 keeps subgroup intrinsics; Tier-2 uses warp_t2 shared-mem.

**Acceptance criteria:**
- Stage assertion: per-chunk byte sizes array byte-identical to CUDA on all 70 fixtures both tiers.
- Final compressed_size matches CUDA exactly on all fixtures.
- compressed_size reaches host via `slzGetLastTimings_vk` drain (no pre-return "lie"); CUDA path updated in M0 mirrors this.
- M0 doctrine validated end-to-end: pre-drain compressed_size_out is undefined; post-drain is exact.

**Estimated LOC:** 600

**Risks addressed:** R3 (continues byte-identity gate), M0 doctrine validation

**CUDA-side changes:** None (M0 already landed).

**Notes:** —

---

### M26 — Wave 2 kernel 5/6: assemble_write.comp

**Depends on:** M25

**Scope:** Port the write portion of `src/encode/assemble_kernel.cu` to `src_vulkan/shaders/encode/assemble_write.comp`. Per-block memcpy into the final frame at prefix-sum-computed offsets from M25.

**Acceptance criteria:**
- Stage assertion: final compressed payload bytes byte-identical to CUDA on all 70 fixtures both tiers.
- 3-byte big-endian sub-chunk header layout verified byte-by-byte against CUDA wire.
- off16/off32 split with 0xFFFF marker verified.

**Estimated LOC:** 700

**Risks addressed:** R3 (encode-direction byte identity)

**CUDA-side changes:** None.

**Notes:** —

---

### M27 — Wave 2 kernel 6/6: frame_assemble.comp — closes VK-encode end-to-end (FULL CONFORMANCE GATE)

**Depends on:** M26

**Scope:** Port the final frame-assembly kernel from `src/encode/assemble_kernel.cu` to `src_vulkan/shaders/encode/frame_assemble.comp`. Single-thread orchestration writing frame header (magic 0x534C5A31, version 2, codec 1, log2_block_size, sc_group_size pre-quantized integers per R10) and chunk-descriptor table. `local_size_x=32`, thread-0-only. THIS IS THE LAST ENCODE KERNEL. After this milestone all four cells of the cross-backend matrix are LIVE on all CI lanes. Run the full 350-cell matrix on every lane and confirm 350 / 350 passing. This is the v1.0-alpha conformance gate.

**Acceptance criteria:**
- Frame header bytes hexdump-identical to CUDA (magic, version, codec, log2_block_size).
- Chunk-descriptor table bytes byte-identical to CUDA.
- sc_group_size on the wire is the pre-quantized integer (65536 or 131072), never f32 — verified by parsing output frame.
- FULL encode pipeline (M22-M27 chained) on M18 fixtures: VK-encoded frame byte-identical to CUDA-encoded frame at L1..L5 on lane A (NVIDIA).
- Same on lane B (AMD RDNA2+ Tier-1) — validates M0 + site #1 emulation correctness on AMD.
- Same on lane C (Intel Arc Tier-1) — validates on Intel.
- Same on lane E (lavapipe Tier-2).
- Same on lane B' (SwiftShader wave64).
- 70 fixtures × 4 matrix cells = 280 full-roundtrip + 70 stage = 350 / 350 conformance checks GREEN.
- Clamp-aware equality: any Tier-2 clamp from L5→L3 honored (effective_level_out=3 in reserved[0]).
- Validation layer zero errors across full run.
- `streamlz_vk.exe compress -v 5 silesia_xml.bin out.slz` → `streamlz.exe decompress` (CUDA) byte-identical to original.
- Reverse: `streamlz.exe compress` → `streamlz_vk.exe decompress` byte-identical.
- Conformance report committed to `docs/vulkan_port_architecture.md` appendix with timing per lane per level.
- Tag commit `wave2-gate-passed`.

**Estimated LOC:** 700

**Risks addressed:** R3 fully retired for VK-encode → byte-identical .slz on all Tier-1 + Tier-2 lanes (lane F/G real-hardware remains for M30); R10 (sc_group_size integer quantization on the wire validated)

**CUDA-side changes:** None.

**Notes:** Combines Plan 1 M30 + Plan 2 M24 + Plan 3 M25 — closes encode end-to-end AND immediately runs the full 350-cell matrix gate as one milestone (no waiting for a separate gate milestone).

---

### M28 — Wave 3 perf: NV_partitioned validation + shape-keyed cmd buffer cache

**Depends on:** M27

**Scope:** Wave 3 perf-only improvements from arch §11. (1) Validate NV_partitioned tier1_nv/ SPV variants produce byte-identical output to tier1/ on a Turing+ NVIDIA card; measure LZ-encode site #1 fast-path speedup (target: closes gap to within 10% of CUDA on NVIDIA). If uplift <5%, deprecate tier1_nv blobs. (2) Implement Tier-1 shape-keyed primary command buffer cache in `vk_command.zig` keyed by (input_size_bucket, level, tier); cap cache at 32 entries with LRU eviction; measure command-buffer-record-overhead reduction at small frame sizes (<256 KiB). (3) Evaluate `vkCmdDispatchIndirect` for self-gating kernels (huff_build_lut, compaction); deferred to v1.1 per arch §8 unless preliminary results show >10% gain. Conformance regression: M27's 350 / 350 must stay green.

**Acceptance criteria:**
- NV_partitioned variant byte-identical to base variant on all 70 fixtures (sanity gate).
- NV_partitioned uplift on LZ-encode L1..L4 ≥ 8% on real NVIDIA Turing+ card OR recorded as "no measurable gain" with profile evidence and tier1_nv blobs deprecated.
- Shape-keyed cmd buffer cache hit rate ≥ 90% on a 1000-frame stream of identical-size inputs.
- Record-time drops from ~200 µs (first call) to <5 µs (subsequent calls).
- Small-input (<4 MiB) encode throughput on RTX 4090 increases by ≥ 5% vs M27 baseline.
- Indirect dispatch evaluation produces written recommendation (lock or defer) recorded in `docs/vulkan_port_architecture.md` §12.
- No byte-identity regressions on M27's 350 / 350 conformance matrix.
- Cache LRU eviction unit test passes (33rd shape evicts LRU, no leak).

**Estimated LOC:** 700

**Risks addressed:** R1 amplifier (NV_partitioned closes gap on NVIDIA), R7 amplifier (shape-keyed cmd cache reduces warm latency)

**CUDA-side changes:** None.

**Notes:** Parallel to M29 and M30 (all three depend only on M27).

---

### M29 — v1.0 release blocker: Lane F real-mobile (Adreno Pixel 8 + Mali Pixel 6 via adb)

**Depends on:** M27

**Scope:** Per arch §22 risks: lavapipe/SwiftShader implement subgroup-spanning semantics differently from real Bifrost/Valhall, so v1.0 release requires real mobile hardware validation. Stand up CI lane F: build streamlz_vk for arm64-v8a Android, deploy to Pixel 8 (Adreno 740) and Pixel 6 (Mali-G78 Valhall) via adb, run a reduced-corpus cross_backend_tests (4 cases × L1..L3 — limited by phone disk + battery). Both phones MUST report Tier-2 and 12 / 12 reduced-corpus assertions passing. Lane F is a v1.0 release blocker. Bifrost device (separate qualified phone, e.g., Pixel 3 Mali-G76) — deferred to v1.1 if no loaner available.

**Acceptance criteria:**
- Pixel 8 (Adreno 740): probe reports Tier-2; 12 / 12 reduced-corpus assertions pass.
- Pixel 6 (Mali-G78 Valhall): probe reports Tier-2; 12 / 12 reduced-corpus assertions pass.
- Mali-G78 subgroup-boundary behavior verified differs from lavapipe (logged) but Tier-2 emulation produces identical bytes.
- adb-based CI script committed to `tools/ci/run_mobile_lane.sh`; runs nightly.
- Perf numbers logged but NOT gated (Mali expected 25-50% of CUDA per R2; Adreno expected ~40-60%).

**Estimated LOC:** 700

**Risks addressed:** R2 fully retired on real mobile silicon (not just SwiftShader emulation)

**CUDA-side changes:** None.

**Notes:** Parallel to M28 and M30.

---

### M30 — v1.0 release blocker: Lane G real AMD CDNA wave64 (MI100/MI210)

**Depends on:** M27

**Scope:** Stand up CI lane G: AMD CDNA MI100 or MI210 (real wave64 hardware, NOT SwiftShader emulation). Build streamlz_vk for Linux x86_64, deploy to CDNA-equipped server (cloud rental acceptable for periodic validation). Run full cross_backend_tests 350-assertion suite on Tier-2 (wave64 path forced). Lane G is v1.0 release blocker.

**Acceptance criteria:**
- MI100/MI210 probe reports Tier-2 with subgroupSize=64.
- 350 / 350 cross-backend assertions pass on lane G (full corpus).
- Tier-2 wave64 hash-store determinism verified: M0's highest-lane-winner rewrite produces byte-identical output on real wave64 (no SwiftShader semantic divergence).
- Lane G CI script committed to `tools/ci/run_cdna_lane.sh`; runs weekly (cloud cost amortization).
- Perf numbers logged but NOT gated.

**Estimated LOC:** 400

**Risks addressed:** Tier-2 wave64 correctness on real CDNA hardware

**CUDA-side changes:** None.

**Notes:** Parallel to M28 and M29.

---

### M31 — v1.0-alpha release: CLI parity + DLL ABI freeze + docs + tag

**Depends on:** M28, M29, M30

**Scope:** Final release prep. (1) `src_vulkan/cli/main_vk.zig` achieves full CLI flag parity with streamlz.exe (same flags, same semantics, same exit codes); surface `effective_level_out` clamps via stderr per arch §6 item 2. (2) `include/streamlz_vk.h` ABI frozen — all 15 _vk symbols signed, documented, version-tagged `3.0.0-vk`. (3) `docs/vulkan_port_architecture.md` reviewed and finalized. (4) CHANGELOG entry enumerating: 15 new _vk symbols, Tier-1 + Tier-2 device matrices, M0 CUDA-side prereq with byte-equivalence guarantee, M0 ABI repurpose of reserved[0]. (5) Release artifacts: streamlz_vk.exe, streamlz_vk.dll, .lib, .pdb (Windows); .so (Linux). (6) Fresh-machine smoke test: clone repo on a machine with ONLY glslc + zig installed (no Vulkan SDK), all 51 .spv blobs load from git, full L1..L5 compress/decompress works. (7) README backend matrix table (CUDA vs Vulkan feature parity). (8) Tag `v1.0.0-alpha-vk`. (9) File v1.1 backlog issues tracking arch §8 deferred items.

**Acceptance criteria:**
- `streamlz_vk.exe --help` matches `streamlz.exe --help` except for binary name.
- All 15 _vk symbols exported from streamlz_vk.dll, ABI-stable signatures, headers committed.
- Fresh-machine smoke test: clone repo, install only glslc + zig, `zig build vk vklib test_vk`, all tests pass.
- M27 350 / 350 conformance + M29 lane F + M30 lane G + M28 perf all green at release tag.
- `docs/vulkan_port_architecture.md` final; CHANGELOG entry committed.
- Tag `v1.0.0-alpha-vk` pushed.
- v1.1 backlog issues opened tracking arch §8 deferred items (wave-64 packed Tier-1 sub-variant, vkCmdDispatchIndirect for self-gating kernels, second queue for async-compute overlap, slzCreateEx_vk struct, pipeline-cache cross-tier sharing, encode pipeline fusion, single-thread orchestration kernel fusion revisit).

**Estimated LOC:** 600

**Risks addressed:** Final release discipline; no-SDK-build promise validated.

**CUDA-side changes:** None.

**Notes:** Plan 3 fresh-machine smoke test grafted. Plan 2 v1.1 backlog discipline grafted.

---

## 6. Out-of-scope for v1.0-alpha

The following items from arch §8 are explicitly deferred to v1.1 or later. v1.1 backlog issues are filed in M31:

1. **Wave-64 packed Tier-1 sub-variant** for AMD GCN/CDNA — design exists, validation cost high; defer until real AMD CDNA hardware lane (lane G) has matured.
2. **`vkCmdDispatchIndirect` for self-gating kernels** — Mesa anv #8137 tracked; M28 evaluates and either ships in v1.0 or defers.
3. **Worker-thread stack size remeasurement** — `worker_stack_size = 32 << 20`; remeasure after M22 lands and inform v1.0-final tag.
4. **Bifrost real-mobile lane** — Pixel 3 Mali-G76 or equivalent; lane F covers Valhall (G78), Bifrost is v1.1 if no loaner.
5. **`slzCreateEx_vk` struct** — env vars cover v1.0; revisit for v2.
6. **Second internal queue for async-compute overlap** — v2.
7. **`VK_KHR_8bit_storage` fast-path inside Tier-1 byteio** — baseline is shift/mask; fast path is opportunistic; verify across all Tier-1 vendors before enabling.
8. **Pipeline-cache cross-tier sharing** — different SPIR-V layouts, not implemented.
9. **Encode pipeline fusion** (port `walkStream` + assemble-measure prefix-sum to GPU compute to collapse host bounces) — recommendation: fuse; final lock for v1.1.
10. **Single-thread orchestration kernel fusion** (D1, D2, D4, D5, D7 into one mega-kernel) — locked as "no fusion in v1.0" per arch §8; revisit if barrier overhead measured >25 µs per decode in M28 perf tuning.
