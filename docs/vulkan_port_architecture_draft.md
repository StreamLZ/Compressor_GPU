# Vulkan Port — Architecture (Draft)

> **Status:** Synthesis of four proposals (MINIMUM-RISK, PERFORMANCE-FIRST, MAXIMUM PORTABILITY, PRAGMATIC TIERED) judged across CORRECTNESS, PORTABILITY, and PERFORMANCE lenses. Winner: **PRAGMATIC TIERED** (median rank: 2; two judges placed it #1 or #2; the third placed it #2 in pure perf). The winner is the only proposal that simultaneously hits day-1 mobile coverage, day-1 wire-format byte-identity, and ~85% CUDA parity on NVIDIA Turing+. Grafts from the other three sharpen specific weaknesses (perf, correctness gates, mobile soak).

---

## 0. Locked constraints

These are non-negotiable inputs from the user; no decision in this document overrides them:

1. **Shaders:** GLSL, compiled via `glslc` → SPIR-V (Vulkan 1.3 target for Tier-1, Vulkan 1.2 floor for Tier-2).
2. **Vendor coverage:** NVIDIA + AMD + Intel desktop + mobile (Adreno / Mali). Solution must either be subgroup-size-agnostic OR force `subgroupSize=32` via `VK_EXT_subgroup_size_control` — we do **both** via tiers.
3. **Sibling build:** `src_vulkan/` next to `src/`; CUDA `src/` untouched. Artifacts: `streamlz_vk.exe`, `streamlz_vk.dll`. All C symbols suffixed `_vk` (e.g. `slzCompress_vk`). Same CLI flags as `streamlz.exe`.
4. **Full API parity day 1:** all 14 symbols in `include/streamlz_gpu.h` get `_vk` siblings (`slzCompress*`, `slzDecompress*`, `slzCompressBound`, `slzCreate`/`slzDestroy`, `slzKernelTiming_t`, `slzStatus_t` enum).
5. **Wire format:** byte-identical to CUDA `.slz` output. Cross-backend roundtrip mandatory at L1..L5.
6. **Compression levels:** L1..L5 produce identical bytes per level as the CUDA encoder (same hash sizes, same parser dispatch).

---

## 1. Foundation summary

Anchored in `docs/vulkan_port_map.md`. Key facts the architecture leans on:

- **Kernel count:** 17 `__global__` kernels total (6 encode under `src/encode/`, 11 decode under `src/decode/`). ~5,800 LOC of device source to port to GLSL.
- **Warp intrinsics in use:** 56 `__shfl_sync`, 7 `__shfl_down_sync`, 5 `__ballot_sync`, 45 `__syncwarp`, 3 `__match_any_sync`, plus `popc`/`clz`/`ffs`/`byte_perm` (counts verbatim from foundation §3). Unused: `__activemask`, `__any_sync`, `__all_sync`, cooperative groups, `cp.async`, `ldmatrix`, inline PTX, textures, 64-bit atomics. The only device atomic is `atomicAdd` on `__shared__`.
- **Hot risk:** `__match_any_sync` site #1 (encode bucket hash) requires emulation on non-NVIDIA. Site #2 (encode key4) is eliminable via byte-identical `subgroupShuffle == key4` rewrite. Site #3 (decode LUT length, domain 1..11) is warm, cheap.
- **All 17 push-constant blocks fit in Vulkan's 128 B minimum guarantee.** Worst case: `slzFrameAssembleKernel` at ~92 B.
- **Wire format invariants** (see foundation §7): magic `0x534C5A31`, version `2`, codec `1`, `log2_block_size - 16 = 2`, `HUFF_NUM_STREAMS = 32` (HARD-LOCKED — `static_assert` in `src/common/gpu_huffman.cuh`), `sc_group_size ∈ {0.25, 0.5}` → integer products 65536 or 131072 exactly representable in f32.
- **Float→uint determinism:** SPIR-V `OpConvertFToU` RTZ matches nvcc `__float2uint_rz` for the inputs streamlz actually uses; we will additionally **pre-quantize on the host** (graft from PERFORMANCE-FIRST) so this never becomes a portability concern.

---

## 2. Directory layout (src_vulkan/ tree)

```
src_vulkan/
├── cli/
│   └── main_vk.zig                 # streamlz_vk.exe entry — mirrors src/cli.zig contracts
├── streamlz_vk.zig                 # C ABI exports (_vk symbols)
├── version.zig                     # "3.0.0-vk"
├── shaders/
│   ├── common/                     # included by every .comp via -I
│   │   ├── byteio.glsl             # 8-bit SSBO fast path + manual shift/mask fallback
│   │   ├── huffman.glsl            # 32-stream BIL layout (wire-locked)
│   │   ├── warp.glsl               # subgroup wrappers: SHFL, BALLOT, SYNC
│   │   ├── wire_format.glsl        # magic, level→codec map, chunk/sub-chunk layout
│   │   ├── match_any.glsl          # site #1/#2/#3 emulations behind #ifdef
│   │   ├── tier_gates.glsl         # TIER=1/2 macros, U64 typedef (uint64_t vs uvec2)
│   │   ├── bda.glsl                # buffer_reference structs (tier1) / SSBO bindings (tier2)
│   │   └── byte_perm.glsl          # explicit 0x0123 swap, unit-testable
│   ├── encode/                     # 6 .comp files — 1:1 with src/encode/ kernels
│   │   ├── lz_encode.comp
│   │   ├── huff_build_tables.comp
│   │   ├── huff_encode_4stream.comp
│   │   ├── assemble_measure.comp
│   │   ├── assemble_write.comp
│   │   └── frame_assemble.comp
│   └── decode/                     # 11 .comp files
│       ├── walk_frame.comp
│       ├── prefix_sum_chunks.comp
│       ├── scan_parse.comp
│       ├── compact_huff_descs.comp
│       ├── compact_raw_descs.comp
│       ├── gather_raw_off16.comp
│       ├── merge_huff_descs.comp
│       ├── huff_build_lut.comp
│       ├── huff_decode_4stream.comp
│       ├── lz_decode.comp
│       └── lz_decode_raw.comp
├── spv/                            # committed SPIR-V (mirrors src/*.ptx policy)
│   ├── tier1/{encode,decode}/*.spv # 17 blobs, requiredSubgroupSize=32, BDA, int64
│   └── tier2/{encode,decode}/*.spv # 17 blobs, subgroup-agnostic, descriptors, uvec2
├── host/
│   ├── vk_loader.zig               # VkInstance/Device, feature probe → tier select, probe kernel
│   ├── vk_memory.zig               # roll-your-own slab suballocator (>2 GiB hash bin-packing)
│   ├── vk_pipeline.zig             # VkPipelineCache + per-kernel pipeline create + module load
│   ├── vk_command.zig              # primary command buffers, optional shape-keyed cache (Tier-1)
│   ├── vk_sync.zig                 # timeline semaphores (Tier-1) / binary-sem + fence (Tier-2)
│   ├── vk_timing.zig               # VkQueryPool TIMESTAMP → slzKernelTiming_t
│   ├── vk_errors.zig               # VkResult → slzStatus_t mapping
│   ├── vk_probe.zig                # one-warp subgroup-size verification kernel runner
│   ├── encode/                     # mirrors src/encode/ host-side Zig orchestration
│   │   ├── encode_context_vk.zig
│   │   ├── encode_lz_vk.zig
│   │   ├── encode_huff_vk.zig
│   │   └── encode_assemble_vk.zig
│   └── decode/
│       ├── decode_context_vk.zig
│       ├── decode_dispatch_vk.zig
│       └── scan_gpu_vk.zig
├── tests/
│   ├── cross_backend_tests.zig     # PR #1 gate — CUDA↔VK byte-identity, both directions
│   ├── golden_slz/                 # pre-recorded .slz from known-good CUDA build
│   ├── tier1_roundtrip_tests.zig   # SLZ_VK_FORCE_TIER=1, 14-case matrix
│   ├── tier2_roundtrip_tests.zig   # SLZ_VK_FORCE_TIER=2, 14-case matrix
│   ├── byteio_unit_tests.zig       # byte_perm, shift/mask, uvec2 carry chain
│   └── lavapipe_ci.zig             # software-Vulkan correctness gate
├── include/
│   └── streamlz_vk.h               # _vk-suffixed mirror of streamlz_gpu.h
tools/
└── build_glsl.bat                  # NEW: glslc per kernel × 2 tiers = 34 .spv blobs
```

`format/` and `share/` are reused verbatim (wire format is identical). `build.zig` grows three new steps (`vk`, `vklib`, `test_vk`); default `zig build` stays CUDA-only.

---

## 3. Shader organization

**One `.comp` source per CUDA kernel; two `.spv` artifacts per source (Tier-1 + Tier-2).**

Every `.comp` starts with the same preamble pulled in via `#include`:

```glsl
#version 460
#extension GL_KHR_shader_subgroup_basic         : require
#extension GL_KHR_shader_subgroup_ballot        : require
#extension GL_KHR_shader_subgroup_shuffle       : require
#extension GL_KHR_shader_subgroup_shuffle_relative : require
#extension GL_KHR_shader_subgroup_arithmetic    : require
#extension GL_KHR_shader_subgroup_vote          : require
#include "tier_gates.glsl"
#include "byteio.glsl"
#include "huffman.glsl"
#include "warp.glsl"
#include "match_any.glsl"
#include "wire_format.glsl"
#include "bda.glsl"
```

`tier_gates.glsl` defines the per-tier compilation profile:

```glsl
#if TIER == 1
    #extension GL_EXT_buffer_reference                   : require
    #extension GL_EXT_buffer_reference2                  : require
    #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
    #extension GL_EXT_shader_explicit_arithmetic_types_int8  : enable
    #define U64        uint64_t
    #define u64_add(a,b) ((a) + (b))
    #define u64_shr(a,n) ((a) >> (n))
    #define SLZ_HAS_BDA 1
    layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    // requiredSubgroupSize=32 set host-side via VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
#else  // TIER == 2
    #define U64 uvec2
    uvec2 u64_add(uvec2 a, uvec2 b) {
        uvec2 r;
        r.x = a.x + b.x;
        r.y = a.y + b.y + uint(r.x < a.x);  // carry
        return r;
    }
    uvec2 u64_shr(uvec2 a, uint n) { /* explicit 64-bit shift via 2x32 */ }
    #define SLZ_HAS_BDA 0
    layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
    // Tier-2 always launches 32 logical lanes; physical subgroupSize discovered at runtime,
    // mismatch handled via shared-memory shuffle emulation in warp.glsl.
#endif
```

Two-warp encode/decode kernels (`lz_encode.comp`, `lz_decode.comp`) override to `local_size_x = 64` (two logical 32-warps per workgroup, matching CUDA's two-warps-per-block packing).

`huffman.glsl` carries the **wire-locked constant** `const uint HUFF_NUM_STREAMS = 32u;` regardless of tier — Tier-1 maps it to the physical subgroup; Tier-2 always runs 32 logical streams via shared-memory shuffle emulation.

`byte_perm.glsl` exports the exact GLSL expression for `__byte_perm(w, 0, 0x0123)` (grafted from MAX-PORTABILITY; it appears in the Huffman bit-stream pointer arithmetic and a wrong formula desyncs every byte downstream):

```glsl
uint byte_perm_0x0123(uint x) {
    return ((x << 24) & 0xFF000000u)
         | ((x <<  8) & 0x00FF0000u)
         | ((x >>  8) & 0x0000FF00u)
         | ((x >> 24) & 0x000000FFu);
}
```

This helper has a dedicated unit test in `tests/byteio_unit_tests.zig`.

---

## 4. VkInstance / VkDevice / queue model

- **`VkInstance`:** one per process, lazy-init on first `slzCreate_vk`, refcounted, destroyed at last `slzDestroy_vk`. `apiVersion = VK_API_VERSION_1_3`. Validation layer (`VK_LAYER_KHRONOS_validation`) + `VK_EXT_debug_utils` enabled when `SLZ_VK_VALIDATION=1` is set (decision #12).
- **`VkPhysicalDevice`:** selected by `SLZ_VK_DEVICE=<index>` env override, else first `DISCRETE_GPU`, else first `INTEGRATED_GPU`. **Tier selection runs at this point** (see §9).
- **`VkDevice`:** one per `slzHandle_vk_t` (matches the CUDA-context "distinct handles fully independent" contract).
- **Queues:** exactly **one** `VkQueue` from the lowest-index family advertising `VK_QUEUE_COMPUTE_BIT`. No second internal queue, no async-compute queue. Rationale: (a) wire-format byte-identity is the dominant constraint and multi-queue submission orderings can perturb atomic histograms (the math is commutative so bytes are safe, but the contract is "bytes, not best-effort"); (b) per-judge perf analysis the cached-command-buffer + indirect-dispatch wins matter more than the second queue. (Decision #7 resolved to single-queue. The PERFORMANCE-FIRST two-queue idea is deferred to v2 — see §21.)
- **`VkCommandPool`:** one per `slzHandle_vk_t` with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` for in-place re-record; one TRANSIENT pool for one-shot upload helpers.

---

## 5. Memory allocation strategy

**Roll our own minimal slab suballocator over `VkDeviceMemory` pools.** No VMA dependency (SLZ has only 2-3 distinct allocation classes; full VMA would bloat the build and pull in C++).

Two pools per `slzContext_vk`:

1. **Scratch pool** — single 64 MiB `VkDeviceMemory` block on `DEVICE_LOCAL`. Holds the timestamp query pool, Huffman LUT, parse staging, persistent device scratch. Bump+free-list at 256 B alignment (covers `minStorageBufferOffsetAlignment` on every known device).
2. **Huge pool** — grown on-demand for hash tables and `d_input`/`d_output` mirrors. Allocated/returned per call (matches `cudaFreeAsync` semantics).

**Usage flags:** every `VkBuffer` requests the union `STORAGE_BUFFER_BIT | SHADER_DEVICE_ADDRESS_BIT | TRANSFER_SRC_BIT | TRANSFER_DST_BIT`. Free on modern drivers, eliminates per-buffer usage tracking.

**`>2 GiB` hash region (foundation R5):** when a kernel requests a buffer exceeding `maxStorageBufferRange` (4 GiB on Tier-1 NVIDIA/AMD; 1 GiB on some Tier-2 mobile), bin-pack into `N ≤ min(maxStorageBufferRange, 1<<31)` sub-buffers. Encode shader receives:
- Tier-1: array of `VkDeviceAddress` (u64) values in push constants, selects bin by `chunk_id >> hash_bits_per_bin`.
- Tier-2: array descriptor binding (`descriptorCount = N`, baked at pipeline create — *not* `descriptorIndexing` which is optional).

**Mobile fit clamp (graft from MAX-PORTABILITY, with correction):** when total VRAM < 4 GiB on Tier-2, clamp `hash_bits` down by 2 — *but only emit a warning + adjust the chosen compression LEVEL*, not silently break wire-format equivalence. A Tier-2 device that can't host the L5 hash region runs L3 (or L2) instead, with `slzGetLastTimings_vk` reporting the effective level. We **never** produce a non-byte-identical `.slz` from the same `(input, level)` pair — that's a hard contract.

**Per-handle isolation:** every `VkBuffer`/`VkDeviceMemory` is owned by its `slzContext_vk`, freed in `slzDestroy_vk`. No global pool.

---

## 6. Descriptor sets + push constants + buffer device address

**Tier-1 (BDA path):**
- Every kernel SSBO argument is a `VkDeviceAddress` passed in push constants.
- `bda.glsl` declares `buffer_reference` structs with explicit `buffer_reference_align`:

```glsl
layout(buffer_reference, std430, buffer_reference_align = 16)
buffer InputBuf  { uint8_t data[]; };

layout(buffer_reference, std430, buffer_reference_align = 16)
buffer HashBuf   { uint     data[]; };
```

- **One** `VkDescriptorSetLayout` per `slzContext_vk` (set=0, zero bindings — kept only so `vkCmdBindDescriptorSets` is valid where needed). 16 of 17 kernels never bind a descriptor set at all.
- Per-kernel `VkPipelineLayout` carries (a) the empty set layout and (b) one `VkPushConstantRange` covering `[0, sizeof(KernelParams_<name>)]` with stage `VK_SHADER_STAGE_COMPUTE_BIT`.

**Tier-2 (descriptor-set path):**
- Each kernel has a `VkDescriptorSetLayout` with N `STORAGE_BUFFER` bindings (typically 4-8 per kernel, always ≤16). `slzFrameAssembleKernel` is split into a 2-dispatch sub-pipeline on Tier-2 if it pushes past the per-stage descriptor limit.
- Sets allocated from one `VkDescriptorPool` per `slzContext_vk` sized for `2 * sum(kernel_bindings)` (double-buffered for overlap between record/execute).
- Push constants on Tier-2 carry only *scalars* (sizes, offsets, level, pre-quantized `sc_group_size`); buffers come from the descriptor set.

**Push-constant struct discipline (both tiers):**
- `KernelParams_<name>` is a packed Zig struct mirroring a GLSL `push_constant` block; both sides use `std430`.
- Build-time `comptime` assert: `@sizeOf(KernelParams_<name>) <= 128` for every kernel.
- Tier-1 layout (worst case): `[u64 ptr] × 8 = 64 B + scalars ≤ 28 B = 92 B` (matches foundation's measurement of `slzFrameAssembleKernel`).
- Tier-2 layout: scalars only (~32-48 B).

**`sc_group_size` quantization (graft from PERFORMANCE-FIRST):** the f32 value `{0.25, 0.5}` is converted on the *host* to its exact integer product `{65536u, 131072u}` and pushed as `uint`. No f32 ever crosses the SPIR-V boundary in any compute path — eliminates `OpConvertFToU` rounding as a portability concern on every driver.

---

## 7. Command buffer recording strategy

**Primary command buffers, no secondaries** (mobile drivers historically buggy on secondaries).

**Tier-1: hybrid record-once-per-shape cache (graft from PERFORMANCE-FIRST).**
- Small LRU on each `slzHandle_vk_t` keyed by `(level, num_chunks_quantized, num_blocks_quantized, hash_layout)`.
- Cache hit: reuse the cached `VkCommandBuffer`, re-push only the BDA pointer fields and per-call scalars via `vkCmdPushConstants` (push constants are *replaceable* without re-record; this is the whole point of BDA + push constants).
- Cache miss: record fresh in-place via `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`.
- Cache size: bounded at 16 entries per handle (typical workloads cycle through 1-3 shapes; 16 is generous).
- Eliminates the 50-200 µs record cost on hot loops where Vulkan submit overhead dominates.

**Tier-2: re-record per submission.** No cache. Tier-2 perf is a correctness contract, not a perf contract — record overhead is invisible against the 2-5× slower kernels.

**Both tiers:**
- One-shot recording uses `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` (Tier-2) or `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` (Tier-1 cached buffers).
- Pool reset is lazy via `vkResetCommandPool` after the completion fence signals.
- **No kernel fusion in v1** (decisions #4/#5 — preserve CUDA split-submission shape). The 5 single-thread orchestration kernels (D1/D2/D4/D5/D7) remain as 5 separate `vkCmdDispatch(1,1,1)` calls — ~25 µs added barrier cost per decode, trivial.
- **Indirect dispatch (`vkCmdDispatchIndirect`)** is wired but **deferred to v1.1** for `HuffBuildLut`/`Decode`/`LzDecode`/`LzDecodeRaw` (graft idea from PERFORMANCE-FIRST is recorded in §21 — over-launch + early-return matches CUDA byte-for-byte and we want that reference established before switching).

---

## 8. Synchronization (semaphores, fences, barriers)

**Tier-1 (timeline-first):**
- One `VkSemaphore` of type `VK_SEMAPHORE_TYPE_TIMELINE` per `slzHandle_vk_t`, monotonically incremented per submission. Acts as the internal `pipeline_stream`/`work_stream` pair: front-half submit signals `value=N+1`, back-half waits `value=N+1` and signals `value=N+2`. No second semaphore needed.
- Caller-facing async ABI exposes three handles: `VkSemaphore wait_semaphore` (timeline; library waits on this before submit), `VkSemaphore signal_semaphore` (timeline; library signals at submit completion), `VkFence completion_fence` (binary; for hosts wanting a hard "is it done?" check). All three may be `VK_NULL_HANDLE` (sync mode).
- `VK_KHR_synchronization2` adopted: `vkCmdPipelineBarrier2` + `VkMemoryBarrier2` with explicit per-access masks (graft from PERFORMANCE-FIRST; finer-grained than the coarse global barrier in MIN-RISK).

**Tier-2 fallback (graft from MAX-PORTABILITY):**
- Binary `VkSemaphore` + `VkFence` only. Async signature exposes `(wait, signal, fence)` but the in-flight count is capped at 1 (binary semaphores can't be waited multiple times).
- Cross-stream ordering collapses to single-submit with `srcStageMask=COMPUTE_SHADER`/`dstStageMask=COMPUTE_SHADER`, `SHADER_WRITE → SHADER_READ|WRITE` between dependent kernels.

**Intra-command-buffer hazards (both tiers):**
- Between every pair of dependent dispatches: `vkCmdPipelineBarrier2` with
  - `srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT`,
  - `dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT`,
  - `srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT`,
  - `dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT`.
- Per-buffer `VkBufferMemoryBarrier2` is used where the dependent-buffer set is small (e.g. prefix-sum → compact); falls back to `VkMemoryBarrier2` (global) for the wide-fan dispatches. Surfaces ordering bugs earlier under the validation layer.

---

## 9. Subgroup strategy + tier model

**Two SPIR-V variants per kernel, runtime feature-probe selects.**

**Tier-1 (NVIDIA Turing+, AMD RDNA2+, Intel Arc, modern Mali Valhall):**
- Hard-requires `VK_EXT_subgroup_size_control` with `subgroupSizeControl=VK_TRUE`, `computeFullSubgroups=VK_TRUE`, and the physical-device range `[minSubgroupSize, maxSubgroupSize]` must include 32.
- Pipeline creation pins `requiredSubgroupSize=32` via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` + `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`.
- **Runtime probe (graft from PRAGMATIC TIERED; reinforced by judge feedback):** at slzCreate_vk after pipeline create, a one-warp probe kernel reads back `gl_SubgroupSize` via `subgroupBroadcastFirst` and writes it to a host-visible buffer. If the value isn't 32, the device is rejected with `SLZ_ERROR_VK_FEATURE_MISSING`. Catches Mesa RADV <22.3-class bugs where the extension is advertised but the pin is silently ignored.
- Additionally requires: `VK_KHR_buffer_device_address`, `shaderInt64`, `VK_KHR_timeline_semaphore`, all six `GL_KHR_shader_subgroup_*` GLSL extensions.

**Tier-2 (Adreno <650, Mali Bifrost, pre-Arc Intel iGPU, anything failing the Tier-1 probe):**
- Vulkan 1.2 floor (1.1 was considered but post-2022 mobile is universally 1.2; gain `host_query_reset` core + `timeline_semaphore` as optional).
- Subgroup-size-agnostic. Spec constant `SLZ_SUBGROUP_SIZE` baked at pipeline create from `VkPhysicalDeviceSubgroupProperties.subgroupSize`.
- Workgroup is always 32 *logical* lanes. On 64-wide physical subgroups, upper 32 lanes execute then results are gated via `gl_LocalInvocationID.x < 32`. On 16-wide subgroups, two logical-warp iterations run in lockstep. On variable-width Bifrost (sg=4/8), shared-memory shuffles emulate cross-lane ops.
- Power-of-two subgroup check at probe time; non-power-of-two rejected with `SLZ_ERROR_VK_FEATURE_MISSING` (mitigates the failure mode MAX-PORTABILITY acknowledges).
- `HUFF_NUM_STREAMS = 32` invariant preserved — Tier-2 *always* runs 32 logical streams.

**Out of scope for v1 (deferred to v2):**
- AMD GCN/CDNA fixed-wave-64 (PERFORMANCE-FIRST's packed variant is recorded in §21 as a future Tier-1 sub-variant).
- AMD GCN/CDNA fixed-wave-64 currently falls to Tier-2.

**Per-vendor whitelist override (graft from PRAGMATIC TIERED failure modes):** `vk_loader.zig` carries a small static table `(vendor_id, device_id) → forced_tier` for known-straddle devices (e.g. Intel Battlemage with `minSubgroupSize=8, maxSubgroupSize=16`). Env var `SLZ_VK_FORCE_TIER={1,2}` overrides everything for debugging.

---

## 10. Warp intrinsic mapping table

All mappings in `shaders/common/warp.glsl`. Macros so the `.comp` bodies textually mirror the `.cu` bodies.

| CUDA intrinsic | Tier-1 lowering | Tier-2 lowering | Notes |
|----------------|-----------------|-----------------|-------|
| `__shfl_sync(mask, v, src)` | `subgroupShuffle(v, src)` | sg==32: same; else shared-mem shuffle | 56 sites |
| `__shfl_up_sync(_, v, d)` | `subgroupShuffleUp(v, d)` | shared-mem with edge mask | 3 sites |
| `__shfl_down_sync(_, v, d)` | `subgroupShuffleDown(v, d)` | shared-mem with edge mask | 7 sites |
| `__shfl_xor_sync(_, v, m)` | `subgroupShuffleXor(v, m)` | `subgroupAdd`/`subgroupMin` rewrite at butterfly sites | 2 sites |
| `__ballot_sync(_, p)` | `subgroupBallot(p).x` | sg==32: same; else shared-mem OR reduce | 5 sites; mask discarded (always `FULL_WARP_MASK`) |
| `__syncwarp()` | `subgroupBarrier(); subgroupMemoryBarrierShared();` | `barrier(); memoryBarrierShared();` (workgroup-wide) | 45 sites |
| `__popc(x)` | `bitCount(x)` | same | core, 3 sites |
| `__clz(x)` | `(x == 0u) ? 32 : 31 - findMSB(x)` | same | core, 2 sites; guarded |
| `__ffs(x)` | `(x == 0u) ? 0 : findLSB(x) + 1` | same | core, 6 sites; guarded |
| `__byte_perm(w, 0, 0x0123)` | `byte_perm_0x0123(w)` (see §3) | same | 4 sites; unit-tested |

**`__match_any_sync` resolution (the perf-critical hot site):**

| Site | Location | Hotness | Tier-1 NVIDIA | Tier-1 AMD/Intel | Tier-2 | Byte-determining? |
|------|----------|---------|----------------|-------------------|--------|-------------------|
| #1   | encode bucket hash (`lz_greedy_parser.cuh`) | HOT | `subgroupPartitionedXorNV` via `VK_NV_shader_subgroup_partitioned` (1 instr) | 32× `subgroupBroadcast` OR-reduction (~32× cycles) | Same OR-reduction over shared mem | YES |
| #2   | encode key4 match | HOT | **ELIMINATED** — replaced with byte-identical `subgroupShuffle(key4, highest_lower) == key4` | Same | Same | YES |
| #3   | decode LUT length, domain [1..11] | WARM | 11-iter `subgroupBallot(L == k)` loop | Same | Same | NO |

The NVIDIA fast path (site #1) is gated behind `#ifdef SLZ_NV_PARTITIONED` and selected at pipeline create when the extension is present. The slow OR-reduction is the universal fallback. Site #2 elimination is the foundation-certified rewrite — strictly faster and *byte-identical* on every backend; never compile the original.

---

## 11. Kernel porting order

Two waves, gated by `cross_backend_tests.zig` (which **lands in PR #1 before any kernel ports**).

**PR #1:** Cross-backend conformance harness + scaffolding (`vk_loader`, `vk_memory`, `vk_pipeline`, `vk_sync`, `vk_timing`, `vk_probe`). Empty .comp shells for all 17 kernels (they panic if dispatched). Test harness runs the four-direction matrix (CUDA→CUDA, CUDA→VK, VK→CUDA, VK→VK) against the golden-file fixtures (graft from MAX-PORTABILITY: pre-recorded `.slz` blobs from a known-good CUDA build — defends against the "both backends produce the same wrong bytes" silent-corruption mode). Until kernels exist, only CUDA→CUDA passes; the others are expected `SKIP` with a recorded reason.

**Wave 1 — DECODE PATH** (unblocks Vulkan-reads-CUDA-encoded `.slz` first; wire format is the spec):

| Order | Kernel | Rationale |
|-------|--------|-----------|
| 1 | `slzWalkFrameKernel` | Single-thread orchestration; validates BDA/push-constant/dispatch chassis |
| 2 | `slzLzDecodeRawKernel` | Simplest data path; validates byteio emulation |
| 3 | `slzPrefixSumChunksKernel` | Subgroup `ExclusiveAdd` primitive test |
| 4 | `slzScanParseKernel` | Wire-format heavy; first byte-identity gate on real bytes |
| 5 | `slzCompactHuffDescsKernel` + `slzCompactRawDescsKernel` | Pair; share scan helpers |
| 6 | `slzGatherRawOff16Kernel` | Raw-block branch |
| 7 | `slzMergeHuffDescsKernel` | Orchestration |
| 8 | `slzHuffBuildLutKernel` | First `__match_any_sync` site (site #3, warm) |
| 9 | `slzHuffDecode4StreamKernel` | Hottest decode; validates wire-locked 32-stream layout |
| 10 | `slzLzDecodeKernel` | Closes the loop on CUDA-encode → Vulkan-decode roundtrip |

**GATE:** `cross_backend_tests` direction (b) CUDA-encode → Vulkan-decode passes for silesia L1/L3/L5 on Tier-1 AND Tier-2 before starting Wave 2.

**Wave 2 — ENCODE PATH:**

| Order | Kernel | Rationale |
|-------|--------|-----------|
| 11 | `slzAssembleMeasureKernel` | Small, prefix-sum-shaped; warms up encode driver |
| 12 | `slzAssembleWriteKernel` | Paired with above |
| 13 | `slzFrameAssembleKernel` | Largest push-constant kernel (92 B); validates 128 B fit |
| 14 | `slzHuffBuildTablesKernel` | Code-length assignment; no `__match_any` |
| 15 | `slzHuffEncode4StreamKernel` | Wire-output-determining 32-stream BIL encode |
| 16 | `slzLzEncodeKernel` | **LAST** — carries both `__match_any_sync` hot sites; biggest perf risk; isolating to one PR keeps it bisectable |

**GATE:** all four `cross_backend_tests` directions pass at L1..L5 on Tier-1 AND Tier-2.

**Wave 3 — Tier-2 broad enable + perf tuning** (NV_partitioned fast path, indirect dispatch, shape-keyed command buffer cache).

---

## 12. SPIR-V build pipeline

`tools/build_glsl.bat` mirrors `tools/build_gpu.bat`. Probes `%VULKAN_SDK%\Bin\glslc.exe` (fails with "install Vulkan SDK ≥1.3.250 or fetch pre-built `.spv` blobs from git" if absent). Per `.comp` source, emits two variants:

```bat
glslc --target-env=vulkan1.3 -O ^
      -DTIER=1 ^
      -I src_vulkan\shaders\common ^
      -o src_vulkan\spv\tier1\<dir>\<name>.spv ^
      src_vulkan\shaders\<dir>\<name>.comp

glslc --target-env=vulkan1.2 -O ^
      -DTIER=2 ^
      -I src_vulkan\shaders\common ^
      -o src_vulkan\spv\tier2\<dir>\<name>.spv ^
      src_vulkan\shaders\<dir>\<name>.comp
```

17 sources × 2 tiers = **34 `.spv` blobs**, all committed to git under `src_vulkan/spv/{tier1,tier2}/` (decision #15 — alongside the existing `.ptx` policy, preserves the `git clone && zig build vk` no-SDK promise).

**`glslc` version pin (graft from PRAGMATIC TIERED failure modes):** the script asserts `glslc --version` reports ≥1.3.250. Earlier versions had SPIRV-Tools BDA storage-class regressions that produced silent garbage reads.

`build.zig` grows a `vk_spv_freshness` step modeled exactly on the existing PTX freshness step: `mtime`-compares every `.comp` and every `common/*.glsl` header against every `.spv` and fails the build with `run tools\build_glsl.bat` if anything is stale. **CI runs the actual `glslc` and diffs against committed `.spv` (graft from PRAGMATIC TIERED failure modes)** — `git status -s` after `build_glsl.bat` must be clean. The freshness step CANNOT be bypassed (no `-Dskip_spv_check` flag).

Embedding: `vk_loader.zig` carries a generated 34-entry table `(kernel_name, tier) → @embedFile("../spv/tierN/.../foo.spv")` blob, fed to `vkCreateShaderModule` at `slzCreate_vk`. SPIR-V is sized, not null-terminated (unlike PTX).

New `build.zig` steps:
- `zig build vk` — `streamlz_vk.exe`
- `zig build vklib` — `streamlz_vk.dll` (+ `streamlz_vk.lib` import lib)
- `zig build test_vk` — runs all three test suites (see §18)

Default `zig build` remains CUDA-only.

---

## 13. C API surface (_vk suffix)

`include/streamlz_vk.h` mirrors `include/streamlz_gpu.h` symbol-for-symbol with `_vk` suffix. 14 symbols + 2 new status codes. Both DLLs (`streamlz_gpu.dll`, `streamlz_vk.dll`) can coexist in one process — different symbol tables, different opaque contexts, fully independent.

```c
// --- handle lifecycle ---
slzStatus_t     slzCreate_vk(slzHandle_vk_t* out_handle);
slzStatus_t     slzDestroy_vk(slzHandle_vk_t handle);

// --- default options ---
slzCompressOpts_t   slzCompressDefaultOpts_vk(void);
slzDecompressOpts_t slzDecompressDefaultOpts_vk(void);

// --- pure host helpers ---
slzStatus_t     slzCompressBound_vk(slzHandle_vk_t h,
                                    size_t input_size,
                                    slzCompressOpts_t opts,
                                    size_t* out_bound);
slzStatus_t     slzGetDecompressedSize_vk(slzHandle_vk_t h,
                                          const void* bytes,
                                          size_t n,
                                          size_t* out_size);

// --- host-pointer entry points (spawn 32 MiB-stack worker) ---
slzStatus_t     slzCompressHost_vk(slzHandle_vk_t h,
                                   const void* in, size_t in_sz,
                                   void* out, size_t cap,
                                   size_t* out_sz,
                                   slzCompressOpts_t opts);
slzStatus_t     slzDecompressHost_vk(slzHandle_vk_t h,
                                     const void* in, size_t in_sz,
                                     void* out, size_t cap,
                                     size_t* out_sz,
                                     slzDecompressOpts_t opts);

// --- async / device-pointer entry points ---
slzStatus_t     slzCompressAsync_vk(slzHandle_vk_t h,
                                    const void* d_input, size_t input_size,
                                    void* d_output, size_t max_compressed_size,
                                    size_t* compressed_size,         /* written BEFORE return */
                                    slzCompressOpts_t opts,
                                    void* wait_semaphore,            /* VkSemaphore or NULL */
                                    void* signal_semaphore,          /* VkSemaphore or NULL */
                                    void* completion_fence);         /* VkFence or NULL */
slzStatus_t     slzDecompressAsync_vk(slzHandle_vk_t h,
                                      const void* d_input, size_t input_size,
                                      void* d_output, size_t max_decompressed_size,
                                      size_t* decompressed_size,
                                      slzDecompressOpts_t opts,
                                      void* wait_semaphore,
                                      void* signal_semaphore,
                                      void* completion_fence);

// --- timing drain ---
slzStatus_t     slzGetLastTimings_vk(slzHandle_vk_t h,
                                     slzKernelTiming_t* out, size_t cap,
                                     size_t* out_count);
slzStatus_t     slzWaitAndGetLastTimings_vk(slzHandle_vk_t h,
                                            void* signal_semaphore,
                                            slzKernelTiming_t* out, size_t cap,
                                            size_t* out_count);

// --- diagnostics ---
const char*     slzStatusString_vk(slzStatus_t s);
const char*     slzVersionString_vk(void);  // "3.0.0-vk"
```

**ABI struct reuse (verbatim from CUDA header):**
- `slzCompressOpts_t = { int level; int enable_profiling; int reserved[6]; }` (32 B)
- `slzDecompressOpts_t = { int enable_profiling; int reserved[7]; }` (32 B)
- `slzKernelTiming_t = { const char* name; float ms; }` (16 B on 64-bit). Same struct, same field order — cross-backend perf tooling continues to work.
- `slzHandle_vk_t = struct slzContext_vk*` (opaque)

**Status enum additions (append-only; backward-compatible):**
- `SLZ_ERROR_DEVICE_LOST = 8`
- `SLZ_ERROR_VK_FEATURE_MISSING = 9`

**Kernel timing name strings:** static `const char*` literals match the CUDA kernel names verbatim (`"slzLzEncodeKernel"` etc.) even though the implementation is a `.comp` shader — cross-backend profilers line up.

**CLI (`streamlz_vk.exe`):** fork of `src/cli.zig` at `src_vulkan/cli/main_vk.zig`, imports `_vk` symbols, identical command-line surface — end users see only the binary name change.

---

## 14. Async + device-pointer API design

The CUDA `void* stream` slot is replaced by the **(wait_semaphore, signal_semaphore, completion_fence) triple** at the async entry points. The three parameters surface the Vulkan sync model honestly (timeline semaphores + fences are not freely interchangeable; pretending they are at the ABI layer creates a worse abstraction than honesty).

**Behavior matrix:**

| `wait_semaphore` | `signal_semaphore` | `completion_fence` | Mode |
|------------------|--------------------|--------------------|------|
| NULL | NULL | NULL | **Synchronous** — library `vkQueueSubmit` + `vkWaitSemaphores` (internal) before returning; matches CUDA stream=0 blocking |
| valid | NULL | valid | Async, fence-only completion check |
| valid | valid | valid | Full async, both signal paths |
| any combination of NULL/valid | always permitted | each parameter is independent |

**Device pointers:**
- `d_input` / `d_output` are u64 `VkDeviceAddress` values cast through `void*` (same shape as CUDA `CUdeviceptr` cast through `void*`).
- Caller obtains the address via `vkGetBufferDeviceAddress` on their own `VkBuffer` — we never see the `VkBuffer` handle. Matches CUDA's "any `CUdeviceptr` will do" contract; lets callers use their own allocator.
- Tier-2 (no BDA): same ABI surface, but internally we maintain a small `(VkDeviceAddress → (VkBuffer, offset))` reverse lookup table populated by an additional `slzRegisterBuffer_vk(handle, VkBuffer)` helper. Callers on Tier-2 devices must register their buffers once; the Tier-1 path ignores the registration.

**`compressed_size_out`:** written *before* `vkQueueSubmit` returns — matches CUDA semantics. The value is known from host-side `compressBound` + assemble-measure metadata before the GPU work runs.

**`slzDecompressAsync_vk` fast-path/fallback:** try D2D via the device-address path; if input/output don't satisfy alignment or coherence, return `SLZ_ERROR_UNSUPPORTED` so the host fallback (`slzDecompressHost_vk`) takes over. Exact CUDA behavior.

---

## 15. Timing model

One `VkQueryPool` of type `VK_QUERY_TYPE_TIMESTAMP` per `slzHandle_vk_t`. Sized 68 slots (17 kernels × 2 timestamps × 2 in-flight buffer).

**Recording (per dispatch, when `opts.enable_profiling != 0`):**
```c
vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,    pool, 2*k+0);
vkCmdDispatch(cb, ...);  // (or vkCmdDispatchIndirect in v1.1)
vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, pool, 2*k+1);
```

Stage choices locked per decision #18.

**Drain (`slzGetLastTimings_vk`):**
- `vkGetQueryPoolResults` with `VK_QUERY_RESULT_64_BIT` (+ `VK_QUERY_RESULT_WAIT_BIT` for the `Wait...` variant).
- Multiply `(end_ticks - start_ticks)` by `VkPhysicalDeviceLimits.timestampPeriod` (ns/tick) × `1e-6` → milliseconds.
- **`timestampValidBits` handling:** queue family reports the valid bit count (typically 64 on NVIDIA/AMD desktop, 36 on some Adreno/Mali). Mask both timestamps to `(1ULL << bits) - 1` *before* subtraction so wraparound is handled. Unit-tested with synthetic 36-bit values.
- Result writes into `slzKernelTiming_t` with `name` = the static kernel name string and `ms` = the computed delta.

**Pool reset:** `vkResetQueryPool` from host via `VK_EXT_host_query_reset` (or Vulkan 1.2 core feature). Fallback to `vkCmdResetQueryPool` at the start of each recording on Tier-2 devices lacking host-reset.

**`enable_profiling=0`:** skips the `vkCmdWriteTimestamp2` ops entirely — zero per-dispatch overhead.

---

## 16. Error mapping

`vk_errors.zig::vkToSlz(VkResult) → slzStatus_t`:

| `VkResult` | `slzStatus_t` | Notes |
|------------|---------------|-------|
| `VK_SUCCESS` | `SLZ_SUCCESS` | |
| `VK_ERROR_OUT_OF_HOST_MEMORY` | `SLZ_ERROR_OUT_OF_MEMORY` | reuse code 5 |
| `VK_ERROR_OUT_OF_DEVICE_MEMORY` | `SLZ_ERROR_OUT_OF_MEMORY` | reuse |
| `VK_ERROR_OUT_OF_POOL_MEMORY` | `SLZ_ERROR_OUT_OF_MEMORY` | reuse |
| `VK_ERROR_DEVICE_LOST` | `SLZ_ERROR_DEVICE_LOST` | **NEW = 8** |
| `VK_ERROR_FEATURE_NOT_PRESENT` | `SLZ_ERROR_VK_FEATURE_MISSING` | **NEW = 9** |
| `VK_ERROR_EXTENSION_NOT_PRESENT` | `SLZ_ERROR_VK_FEATURE_MISSING` | |
| `VK_ERROR_INCOMPATIBLE_DRIVER` | `SLZ_ERROR_VK_FEATURE_MISSING` | |
| `VK_ERROR_INITIALIZATION_FAILED` | `SLZ_ERROR_INVALID_HANDLE` at create, `SLZ_ERROR_INTERNAL` elsewhere | |
| `VK_ERROR_INVALID_EXTERNAL_HANDLE` | `SLZ_ERROR_INVALID_ARG` | |
| `VK_ERROR_FORMAT_NOT_SUPPORTED` | `SLZ_ERROR_UNSUPPORTED` | reuse |
| anything else | `SLZ_ERROR_INTERNAL` | logged via debug callback when `SLZ_VK_VALIDATION=1` |

Every Vulkan call site uses a `tryVk()` Zig helper that checks the result and propagates the mapped status up the call stack — same shape as the existing `tryCu` pattern on the CUDA side.

`slzStatusString_vk` adds entries for codes 8 and 9 (`"device lost"`, `"required Vulkan feature missing"`).

**Validation layer toggle:** env var `SLZ_VK_VALIDATION=1` enables `VK_LAYER_KHRONOS_validation` + `VK_EXT_debug_utils`; debug messenger routes severity ≥ WARNING to stderr with a `[VK]` prefix. No new ABI struct (`slzCreateEx_vk` deferred — see §21).

---

## 17. Pipeline cache strategy

**Process-wide `VkPipelineCache`** (graft from PRAGMATIC TIERED, refined per judge feedback: process-wide gives better hit rate across multiple handles than per-handle).

**Backing store:**
- Windows: `%LOCALAPPDATA%\streamlz_vk\pipelines.bin`
- Linux: `$XDG_CACHE_HOME/streamlz_vk/pipelines.bin` (default `~/.cache/streamlz_vk/pipelines.bin`)

**File header** (16 bytes before the `VkPipelineCache` blob; we validate before passing to Vulkan):
```
[0..4]   magic = 0x534C5A56 ("SLZV")
[4..8]   library version (major<<16 | minor<<8 | patch)
[8..12]  vendor_id
[12..14] device_id
[14..16] tier (1 or 2)
```
Plus Vulkan's own `VkPipelineCacheHeaderVersion` immediately after, which the driver validates against vendor/device/driver UUID. Either mismatch silently discards and rebuilds.

**Lifecycle:**
- `slzCreate_vk` (process first call): read file (cap 16 MiB), validate our header + Vulkan header, pass blob to `vkCreatePipelineCache`. On any failure, move file aside to `pipelines.bin.bad` and create empty cache.
- `slzDestroy_vk` (process last handle): `vkGetPipelineCacheData` → write to `pipelines.bin.tmp` → atomic `ReplaceFileW` / `rename`. Concurrent processes: last one to write wins; both are valid.
- Sandboxed environments (`%LOCALAPPDATA%` unwritable): in-memory cache only, no persistence. Functionality preserved; cold-start cost remains.

**Cache size budget:** ~200-800 KiB per (device, tier) tuple. Warm-start budget: ~3-5 ms for all 17 pipelines (vs ~30-60 ms cold).

---

## 18. Test strategy

Three test suites under `src_vulkan/tests/`, driven by `zig build test_vk`:

**(1) `cross_backend_tests.zig` — PR #1 GATE (R3).** The single most important addition; addresses the foundation's explicit risk that no cross-backend conformance test exists today.
- Matrix of `{CUDA encode, VK encode} × {CUDA decode, VK decode}` on the 14-case silesia subset at L1..L5.
- Each direction asserts byte equality (sha256 of `.slz` for encode→encode comparisons; structural equality + sha256 of decoded output for cross-decode).
- **Golden `.slz` fixtures** (graft from MAX-PORTABILITY): pre-recorded blobs from a known-good CUDA build live in `src_vulkan/tests/golden_slz/`. Cross-backend equality alone doesn't catch "both backends produce the same wrong bytes"; comparing against frozen goldens does.
- First-differing-byte dump on mismatch with hex context.
- Runs both Tier-1 and Tier-2 (forced via `SLZ_VK_FORCE_TIER`).

**(2) `tier1_roundtrip_tests.zig` + `tier2_roundtrip_tests.zig`** — port of the existing `src/encode/gpu_roundtrip_tests.zig`, 14-case matrix per tier.

**(3) `byteio_unit_tests.zig`** — small Zig-native unit tests for the helpers most likely to silently corrupt bytes: `byte_perm_0x0123`, uvec2 carry-chain arithmetic (Tier-2), shift/mask byte reader, timestamp-mask wraparound. Run on every commit, including the lavapipe CI lane.

**CI matrix:**

| Lane | Platform | Tier | Purpose |
|------|----------|------|---------|
| A | Windows + Mesa lavapipe (software Vulkan, sg=4) | Tier-2 | Free, no-hardware bit-exact correctness gate; runs every PR |
| B | Linux + SwiftShader (sg=4, no shaderInt64) | Tier-2 | Validates uvec2 emulation path even harder |
| C | Self-hosted Windows + NVIDIA RTX 4060 Ti | Tier-1 | Primary perf reference; daily |
| D | Self-hosted Windows + AMD RDNA2 | Tier-1 | AMD validation; daily |
| E | Self-hosted + Intel Arc | Tier-1 | Intel validation; weekly |
| F | (Future) self-hosted + Adreno/Mali via adb | Tier-2 | Real mobile soak; before v1.0 release |

Lanes A and B (graft from MAX-PORTABILITY) provide bit-exact regression coverage without GPU hardware in CI — catches semantic bugs even when hardware is unavailable. Lavapipe + SwiftShader together exercise non-32 subgroup widths AND uvec2 emulation, which no hardware lane reaches.

**`SLZ_VK_VALIDATION=1` is forced on for lanes A and B** (graft from PERFORMANCE-FIRST) so BDA alignment bugs, descriptor mismatches, and barrier omissions surface as test failures rather than silent wrong bytes.

**No `-Dskip_spv_check` flag exists.** CI runs `tools\build_glsl.bat` and then `git status -s` — any unclean output fails the lane. Prevents stale `.spv` commits.

---

## 19. Grafted ideas (from runner-up proposals)

Captured here so the execution workflow has a single index of what came from where and why each graft survived the synthesis:

| Idea | Source | Why grafted |
|------|--------|-------------|
| Pre-quantize `sc_group_size` to integer 65536/131072 on the host | PERFORMANCE-FIRST | Eliminates `OpConvertFToU` RTZ as a portability risk on every driver, including lavapipe/SwiftShader. Stronger byte-identity guarantee than trusting `FPRoundingMode`. |
| Cached primary command buffers keyed by `(level, shape)` on Tier-1 | PERFORMANCE-FIRST | Kills the 50-200 µs record cost on hot encode loops. Push constants update without re-record. |
| `VkMemoryBarrier2` + `vkCmdPipelineBarrier2` with explicit per-access masks (sync2) | PERFORMANCE-FIRST | Finer-grained than the coarse `SHADER_WRITE → SHADER_READ\|WRITE` global barrier; surfaces ordering bugs earlier under validation. |
| `VK_NV_shader_subgroup_partitioned` for `__match_any_sync` site #1 on NVIDIA | PERFORMANCE-FIRST | Recovers ~30-40% of L1 encode throughput vs the 32-broadcast OR-reduction. Gated, optional, doesn't affect bytes. |
| Wave-64 packed Tier-1 sub-variant for AMD GCN/CDNA | PERFORMANCE-FIRST | Deferred to v2 (recorded in §21); recovers data-center AMD without falling to Tier-2. |
| Single SPIR-V variant with runtime `gl_SubgroupSize` branching — for Tier-2 only | MAX-PORTABILITY | Tier-2 covers Mali Bifrost (sg=4/8) and exotic Adreno via one blob. We do NOT use this for Tier-1 (per-pin requiredSubgroupSize=32 is faster). |
| uvec2 u64 emulation + uint+shift/mask byteio + descriptor-set ABI — for Tier-2 only | MAX-PORTABILITY | Recovers mobile devices without `shaderInt64` / `shaderInt8` / BDA. |
| Lavapipe + SwiftShader CI lanes | MAX-PORTABILITY | Bit-exact regression without GPU hardware; only proposal that catches non-32 subgroup width bugs. |
| Golden `.slz` fixtures from known-good CUDA build | MAX-PORTABILITY | Defends against the "both backends produce the same wrong bytes" mode that pure cross-backend equality cannot catch. |
| Mobile hash-bits clamp (gracefully degrade) | MAX-PORTABILITY (corrected) | Original was "silently clamp bits"; we corrected to "warn + drop to a lower published level" so byte-identity is preserved per-`(input, level)`. |
| Cross-backend conformance test in PR #1 BEFORE any kernel ports | MINIMUM-RISK | Operationalizes R3 immediately. The strongest correctness gate of any proposal. |
| Decode-first port order, gating encode on decode-direction conformance | MINIMUM-RISK | Wire format is the spec; CUDA→VK decode roundtrip catches reader bugs in 10 kernels of work, not 17. |
| Foundation-certified `subgroupShuffle == key4` rewrite for `__match_any_sync` site #2 | MINIMUM-RISK (universally adopted) | Free perf win on every backend; eliminates a hot site at zero variant cost. |
| Runtime probe kernel verifying `gl_SubgroupSize == 32` on Tier-1 | PRAGMATIC TIERED (own) | Catches drivers that advertise `subgroup_size_control` but silently ignore `requiredSubgroupSize=32` (pre-22.3 RADV class). |
| Per-vendor whitelist override in `vk_loader.zig` | PRAGMATIC TIERED (own) | Lets straddle devices (e.g. Intel Battlemage) be forced to the correct tier. |
| Process-wide `VkPipelineCache` (not per-handle) with versioned header | PRAGMATIC TIERED + judge synthesis | Better hit rate across handles; cleaner invalidation than per-handle. |
| Pin `glslc ≥ 1.3.250` at build time | PRAGMATIC TIERED | Defends against the SPIRV-Tools BDA storage-class regression that produces silent garbage reads. |
| Explicit `__byte_perm` formula in `byte_perm.glsl` with a unit test | MAX-PORTABILITY (best articulated) + judge call-out | Lives on the Huffman bit-stream pointer path; a wrong formula desyncs every byte. |
| CI runs `tools\build_glsl.bat` then `git status -s` to catch stale `.spv` | PERFORMANCE-FIRST | Prevents the "developer forgot to commit `.spv`" silent-drift mode. |

---

## 20. Explicitly rejected ideas (and why)

| Rejected idea | Source | Why rejected |
|---------------|--------|--------------|
| Single SPIR-V variant for ALL devices (Tier-1 too) | MINIMUM-RISK / MAX-PORTABILITY | Forces NVIDIA Turing+ to either run the slow OR-reduction (MINIMUM) or the uvec2/shared-mem path (MAX-PORT). Forfeits ~40% encode perf on the most common GPU vendor for ~zero correctness gain. |
| Three SPIR-V variants per kernel (vanilla / sg32 / wave64) | PERFORMANCE-FIRST | The wave-64 packed variant runs "twice the work per dispatch with upper-half ballot masking" — a novel code path with no CUDA reference where a single masking bug silently corrupts output ON AMD ONLY. Deferred to v2 as a Tier-1 sub-variant once cross-backend testing matures. |
| Hard-fail at create if BDA is missing (no Tier-2 fallback) | MINIMUM-RISK | Foundation R4 — drops a meaningful slice of pre-2020 mobile. The two-tier design is *the whole point* of recovering them gracefully. |
| Vulkan 1.1 floor with NO hard extension requirements | MAX-PORTABILITY | Tier-1 NVIDIA/AMD/Intel-Arc perf drops to 50-62% of CUDA on the hardware streamlz is most used on. The portability-as-default lens loses too much perf on the dominant devices. We get the portability via Tier-2 instead. |
| No `VK_NV_shader_subgroup_partitioned` fast path | MINIMUM-RISK / MAX-PORTABILITY | Foundation lists it as optional; judge analysis pegs it as the single biggest perf lever on NVIDIA L1 encode. Free win — gated, doesn't affect bytes, optional extension probe. |
| `slzCreateEx_vk` struct for validation toggle | (rejected per decision #12) | Env var `SLZ_VK_VALIDATION=1` preserves 1:1 ABI shape with CUDA. Can be promoted to a struct in v2 without breaking existing callers. |
| Async-compute / multi-queue submission for overlap | PERFORMANCE-FIRST | Risks non-determinism in histogram-atomic interleavings (the math is commutative — bytes are safe — but the contract is "bytes, not best-effort"). Single VkQueue collapses concerns. |
| VMA (AMD Vulkan Memory Allocator) dependency | (default) | Adds a C++ dependency to a pure-Zig build. SLZ has 2-3 allocation classes; our slab allocator suffices. |
| Indirect dispatch (`vkCmdDispatchIndirect`) in v1 | PERFORMANCE-FIRST | Has a known Intel Arc driver bug listed in failure modes. Over-launch + early-return matches CUDA byte-for-byte; we want that reference established first. Re-evaluate in v1.1. |
| Single-thread orchestration kernel fusion (5 → 1 mega-kernel) | (decision #5) | Adds dispatch divergence from CUDA. ~25 µs barrier overhead per decode is trivially small. Preserve the CUDA shape. |
| Silently clamp hash bits on mobile producing non-byte-identical output | MAX-PORTABILITY (as-written) | Breaks the L1..L5 byte-identity invariant per `(input, level)`. Corrected: warn + run a published lower level. |
| Pre-recording command buffers globally (all dispatches) | PERFORMANCE-FIRST extension | Tier-2 explicitly does not cache; Tier-1 only caches when the shape repeats. Full caching cross-kernel is deferred to v1.1. |

---

## 21. Open questions deferred to milestone planning

1. **Wave-64 packed Tier-1 sub-variant for AMD GCN/CDNA** — design exists (PERFORMANCE-FIRST), validation cost high. Defer to v1.1 once cross-backend testing on GCN hardware is available.
2. **`vkCmdDispatchIndirect` for self-gating kernels (HuffBuildLut/Decode/LzDecode)** — Intel Arc driver bug recorded; re-evaluate once driver fixes ship. Likely v1.1.
3. **Worker-thread stack size re-measurement** — `worker_stack_size = 32 << 20` was sized for CUDA; remeasure for Vulkan after first encode kernel lands.
4. **Real-mobile CI lane (Adreno + Mali via adb)** — slated for v1.0 release blocker but not blocking v1.0-alpha.
5. **`slzCreateEx_vk` struct for validation/tier-force/device-index** — env vars cover v1; struct API in v2 if user demand materializes.
6. **Second internal queue for async-compute overlap** — deferred to v2 pending evidence of caller workloads that need it.
7. **`VK_KHR_8bit_storage` fast-path inside Tier-1 byteio** — currently Tier-1 uses BDA + native `uint8_t data[]`; verify on all Tier-1 vendors that this lights up the 8-bit path (it should). If not, add as Tier-1.5.
8. **Pipeline-cache cross-handle sharing within a single process** — process-wide cache file is loaded once but each handle currently creates its own `VkPipelineCache` from the blob. Consider a shared `VkPipelineCache` object behind a refcount.

---

## 22. Risks accepted

These are conscious trade-offs the architecture takes; the project owner has the inputs to revisit them later.

| Risk | Severity | Mitigation in this architecture |
|------|----------|--------------------------------|
| **R1** — `__match_any_sync` emulation regresses L1-L4 encode throughput on AMD/Intel | Medium | NVIDIA gets `NV_partitioned` fast path; site #2 eliminated universally; AMD/Intel L1 lands at ~65-75% CUDA which is acceptable for v1. |
| **R2** — Tier-1 `subgroupSize=32` pin unsupportable on a non-trivial mobile class | Medium | Tier-2 path catches every device that fails the Tier-1 probe. Mali Bifrost / Adreno <650 run Tier-2 at 25-50% CUDA — slow but functional. |
| **R3** — Wire-format byte-identity regression caught only late | **High** → **Low** | Cross-backend conformance test lands in PR #1 BEFORE any kernel ports. Golden `.slz` fixtures + lavapipe CI + SwiftShader CI catch the "both backends agree on wrong bytes" mode. |
| **R4** — BDA absent breaks `void*` device-pointer ABI | Low | Tier-2 path uses descriptor sets + `slzRegisterBuffer_vk` reverse lookup; ABI surface unchanged for callers. |
| **R5** — Encode hash table > 2 GiB | Low | Bin-packing into ≤2 GiB sub-buffers wired into `vk_memory.zig`; encode shader receives address array via push constants (Tier-1) or descriptor array (Tier-2). |
| 2 SPIR-V variants × 17 kernels = 34 blobs committed (~600-1200 KiB) | Low | Accepted; preserves the no-SDK-build promise; freshness step + CI `git status` prevent drift. |
| Tier-2 perf is 2-5× slower than Tier-1 on the same hardware class | Accepted | Tier-2 is a correctness contract, not a perf contract. Documentation tells callers "use `slzCompressHost` if you need perf on mobile". |
| Indirect dispatch not in v1; over-launch matches CUDA | Low | Costs negligible at typical block sizes; deferred to v1.1 once Intel Arc driver bug clears. |
| Single VkQueue forgoes async-compute overlap with caller workloads | Low | Caller can create a second `slzHandle_vk_t` with its own device for overlap. Deferred to v2. |
| `glslc` SPIRV-Tools regressions silently miscompile BDA loads | Low | Pinned `glslc ≥ 1.3.250` at build; validation layer + cross-backend tests catch in CI. |
| Driver lies about honoring `requiredSubgroupSize=32` | Low | Runtime probe kernel verifies after pipeline create; rejects device with `SLZ_ERROR_VK_FEATURE_MISSING`. |
| Pipeline cache from a different driver version silently loaded | Low | Vulkan's own `VkPipelineCacheHeaderVersion` validates vendor/device/driver UUID; mismatch silently discards. Our 16-byte prefix adds library version + tier as belt-and-suspenders. |
| Cross-backend test corpus drift when CUDA wire format changes | Accepted | Byte-identity is a hard invariant per foundation; regen of golden fixtures requires explicit project-owner sign-off. |

---

**Document end.** Synthesis notes: PRAGMATIC TIERED won the median across the three judge lenses (CORRECTNESS: #1; PORTABILITY: #2; PERFORMANCE: #2). Key grafts that hardened it: (a) PERFORMANCE-FIRST's cached command buffers + sync2 + NV_partitioned + sc_group_size pre-quantization recovered the perf gap; (b) MAX-PORTABILITY's lavapipe/SwiftShader CI + golden `.slz` fixtures + explicit `byte_perm` unit test closed correctness blind spots; (c) MINIMUM-RISK's "PR #1 lands the conformance test" gating discipline made the entire port roadmap auditable from day one.
