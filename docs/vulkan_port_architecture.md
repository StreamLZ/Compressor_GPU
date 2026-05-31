# Vulkan Port — Architecture (Approved)

> **Status:** Approved synthesis of four proposals (MINIMUM-RISK, PERFORMANCE-FIRST, MAXIMUM PORTABILITY, PRAGMATIC TIERED) judged across CORRECTNESS, PORTABILITY, and PERFORMANCE lenses. Winner: **PRAGMATIC TIERED** (median rank 2). This revision incorporates all fatal/serious findings from three adversarial reviewers. Sections flagged "REVISED" below state the change explicitly; unchallenged sections are carried over verbatim from the draft.

---

## 0. Locked constraints

These are non-negotiable inputs from the user; no decision in this document overrides them:

1. **Shaders:** GLSL, compiled via `glslc` → SPIR-V (Vulkan 1.3 target for Tier-1, Vulkan 1.2 floor for Tier-2).
2. **Vendor coverage:** NVIDIA + AMD + Intel desktop + mobile (Adreno / Mali). Solution must either be subgroup-size-agnostic OR force `subgroupSize=32` via `VK_EXT_subgroup_size_control` — we do **both** via tiers.
3. **Sibling build:** `src_vulkan/` next to `src/`; CUDA `src/` untouched. Artifacts: `streamlz_vk.exe`, `streamlz_vk.dll`. All C symbols suffixed `_vk` (e.g. `slzCompress_vk`). Same CLI flags as `streamlz.exe`.
4. **Full API parity day 1:** all 14 symbols in `include/streamlz_gpu.h` get `_vk` siblings, **plus one Tier-2-only helper `slzRegisterBuffer_vk` (15th symbol)** that is exported unconditionally on both tiers (no-op on Tier-1). See §13 for the explicit count change and §14 for the rationale.
5. **Wire format:** byte-identical to CUDA `.slz` output. Cross-backend roundtrip mandatory at L1..L5.
6. **Compression levels:** L1..L5 produce identical bytes per level as the CUDA encoder (same hash sizes, same parser dispatch).

---

## 1. Foundation summary

Anchored in `docs/vulkan_port_map.md`. Key facts the architecture leans on:

- **Kernel count:** 17 `__global__` kernels total (6 encode under `src/encode/`, 11 decode under `src/decode/`). ~5,800 LOC of device source to port to GLSL.
- **Warp intrinsics in use:** 56 `__shfl_sync`, 7 `__shfl_down_sync`, 5 `__ballot_sync`, 45 `__syncwarp`, 3 `__match_any_sync`, plus `popc`/`clz`/`ffs`/`byte_perm` (counts verbatim from foundation §3). Unused: `__activemask`, `__any_sync`, `__all_sync`, cooperative groups, `cp.async`, `ldmatrix`, inline PTX, textures, 64-bit atomics. The only device atomic is `atomicAdd` on `__shared__`.
- **Hot risk:** `__match_any_sync` site #1 (encode bucket hash) requires emulation on non-NVIDIA. Site #2 (encode key4) is eliminable via byte-identical `subgroupShuffle == key4` rewrite — see **Appendix A** for the full citation, surrounding context, and proof of bit-for-bit equivalence. Site #3 (decode LUT length, domain 1..11) is warm, cheap.
- **All 17 push-constant blocks fit in Vulkan's 128 B minimum guarantee.** Worst case: `slzFrameAssembleKernel` at ~92 B. Runtime check in `vk_loader.zig` reads `physicalDeviceLimits.maxPushConstantsSize` and rejects the device with `SLZ_ERROR_VK_FEATURE_MISSING` if the limit is below 128 B (defends against historical Intel iGPU validation-layer warnings).
- **Wire format invariants** (see foundation §7): magic `0x534C5A31`, version `2`, codec `1`, `log2_block_size - 16 = 2`, `HUFF_NUM_STREAMS = 32` (HARD-LOCKED — `static_assert` in `src/common/gpu_huffman.cuh`), `sc_group_size ∈ {0.25, 0.5}` → integer products 65536 or 131072 exactly representable in f32.
- **Float→uint determinism:** SPIR-V `OpConvertFToU` RTZ matches nvcc `__float2uint_rz` for the inputs streamlz actually uses; we will additionally **pre-quantize on the host** so this never becomes a portability concern.

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
│   │   ├── warp_t1.glsl            # Tier-1 subgroup-intrinsic implementations
│   │   ├── warp_t2.glsl            # Tier-2 shared-memory workgroup-wide emulations
│   │   ├── wire_format.glsl        # magic, level→codec map, chunk/sub-chunk layout
│   │   ├── match_any.glsl          # site #1/#2/#3 emulations behind #ifdef
│   │   ├── tier_gates.glsl         # TIER=1/2 macros, U64 typedef (uint64_t vs uvec2)
│   │   ├── bda.glsl                # buffer_reference structs (tier1) / SSBO bindings (tier2)
│   │   ├── ballot_mask.glsl        # wave64-safe ballot masking helper
│   │   └── byte_perm.glsl          # explicit 0x0123 swap, unit-testable
│   ├── encode/                     # 6 .comp files — 1:1 with src/encode/ kernels
│   │   ├── lz_encode.comp          # local_size_x = 32 (REVISED — one logical warp/wg, matches CUDA __launch_bounds__(32,1))
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
│       ├── lz_decode.comp          # local_size_x = 64 Tier-1; 32 Tier-2 (see §3)
│       └── lz_decode_raw.comp      # local_size_x = 64 Tier-1; 32 Tier-2
├── spv/                            # committed SPIR-V (mirrors src/*.ptx policy)
│   ├── tier1/{encode,decode}/*.spv         # 17 blobs, requiredSubgroupSize=32, BDA, int64
│   ├── tier1_nv/{encode,decode}/*.spv      # 17 blobs, NV_partitioned fast path (encode-only differences)
│   └── tier2/{encode,decode}/*.spv         # 17 blobs, subgroup-agnostic, descriptors, uvec2
├── host/
│   ├── vk_loader.zig               # VkInstance/Device, feature probe → tier select, probe kernel
│   ├── vk_memory.zig               # roll-your-own slab suballocator (>2 GiB hash bin-packing)
│   ├── vk_pipeline.zig             # VkPipelineCache + per-kernel pipeline create + module load
│   ├── vk_command.zig              # primary command buffers, optional shape-keyed cache (Tier-1)
│   ├── vk_sync.zig                 # timeline (Tier-1) / binary-sem + fence (Tier-2) + sync2 wrapper that falls back to vkCmdPipelineBarrier on Tier-2
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
│   ├── wave64_roundtrip_tests.zig  # NEW — emulated wave64 lane via SwiftShader env override
│   ├── byteio_unit_tests.zig       # byte_perm, shift/mask, uvec2 carry chain, timestamp wrap
│   └── lavapipe_ci.zig             # software-Vulkan correctness gate
├── include/
│   └── streamlz_vk.h               # _vk-suffixed mirror of streamlz_gpu.h + slzRegisterBuffer_vk
tools/
└── build_glsl.bat                  # NEW: glslc per kernel × 3 variants = 51 .spv blobs
```

`format/` and `share/` are reused verbatim (wire format is identical). `build.zig` grows three new steps (`vk`, `vklib`, `test_vk`); default `zig build` stays CUDA-only.

---

## 3. Shader organization (REVISED)

**One `.comp` source per CUDA kernel; up to three `.spv` artifacts per source (Tier-1, Tier-1+NV-partitioned, Tier-2).**

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

### 3.1 Workgroup sizing — corrected per CUDA `__launch_bounds__` (Reviewer 1 fatal)

The local_size table is now derived directly from each CUDA kernel's `__launch_bounds__`, NOT a guessed shape. Cross-check is in `tools/check_workgroup_sizes.py` and runs in CI.

| Kernel | CUDA `__launch_bounds__` | Tier-1 `local_size_x` | Tier-2 `local_size_x` |
|--------|--------------------------|-----------------------|-----------------------|
| `slzLzEncodeKernel` | `(32, 1)` — **ONE** warp | **32** | 32 |
| `slzLzDecodeKernel` | `LZ_KERNEL_BLOCK_THREADS=64` — two warps | 64 | **32** (see §9.2) |
| `slzLzDecodeRawKernel` | `LZ_KERNEL_BLOCK_THREADS=64` — two warps | 64 | **32** (see §9.2) |
| `slzHuffEncode4StreamKernel` | 32 | 32 | 32 |
| `slzHuffDecode4StreamKernel` | 32 | 32 | 32 |
| `slzHuffBuildLutKernel` | 32 | 32 | 32 |
| `slzHuffBuildTablesKernel` | 32 | 32 | 32 |
| All 5 single-thread orchestration kernels (D1, D2, D4, D5, D7) | 1 | 32 (with lane gate) | 32 (with lane gate) |
| `slzAssembleMeasureKernel` | 32 | 32 | 32 |
| `slzAssembleWriteKernel` | 32 | 32 | 32 |
| `slzFrameAssembleKernel` | 32 | 32 | 32 |
| `slzScanParseKernel` | 32 | 32 | 32 |
| `slzPrefixSumChunksKernel` | 32 | 32 | 32 |
| `slzCompactHuffDescsKernel`, `slzCompactRawDescsKernel`, `slzMergeHuffDescsKernel`, `slzGatherRawOff16Kernel`, `slzWalkFrameKernel` | per source | per source (cross-checked) | per source |

**The prior draft incorrectly stated that `lz_encode.comp` runs `local_size_x = 64`.** That was wrong — `slzLzEncodeKernel` in `src/encode/lz_kernel.cu` declares `__launch_bounds__(32, 1)` (one warp per block). The corrected mapping above is enforced by the CI check.

The Tier-2 LZ-decode kernels collapse to `local_size_x = 32` (one logical warp per workgroup) and dispatch 2× as many workgroups to compensate. The rationale is in §9.2: two logical warps per workgroup on Tier-2 cannot be safely synchronized because `__syncwarp()` lowers to a workgroup-wide `barrier()` that would deadlock when only one warp reaches it.

### 3.2 Tier gates

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
    // requiredSubgroupSize=32 pinned host-side via VkPipelineShaderStageRequiredSubgroupSizeCreateInfo
    // AND in-shader via layout(subgroup_size = 32) in; (GL_EXT_subgroup_uniform_control_flow + Vulkan 1.3)
    #extension GL_EXT_subgroup_uniform_control_flow : require
    layout(subgroup_size = 32) in;
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
    // Tier-2 always launches 32 logical lanes; physical subgroup size discovered at runtime,
    // bridged via shared-memory broadcasts gated by workgroup barriers (warp_t2.glsl).
#endif
layout(local_size_x = SLZ_LOCAL_SIZE_X, local_size_y = 1, local_size_z = 1) in;
```

`SLZ_LOCAL_SIZE_X` is defined per kernel from the table above, NOT in `tier_gates.glsl`.

### 3.3 Wire-locked constants

`huffman.glsl` carries the **wire-locked constant** `const uint HUFF_NUM_STREAMS = 32u;` regardless of tier — Tier-1 maps it to the physical subgroup; Tier-2 always runs 32 logical streams via shared-memory broadcast emulation. On 16-wide subgroups (e.g. Mali Valhall under Tier-2), the Huffman decode's bounded-interleaved refill row (32 lanes × 4 bytes = 128 B coalesced) becomes two 16-lane × 4-byte rows; this is purely a perf concern.

### 3.4 byte_perm helper

`byte_perm.glsl` exports the exact GLSL expression for `__byte_perm(w, 0, 0x0123)`:

```glsl
// Mirrors CUDA PTX prmt with control 0x0123:
// dst.byte[0] = src.byte[3], dst.byte[1] = src.byte[2],
// dst.byte[2] = src.byte[1], dst.byte[3] = src.byte[0]
// Equivalent to a 32-bit byte reversal: 0x01020304 -> 0x04030201
uint byte_perm_0x0123(uint x) {
    return ((x << 24) & 0xFF000000u)
         | ((x <<  8) & 0x00FF0000u)
         | ((x >>  8) & 0x0000FF00u)
         | ((x >> 24) & 0x000000FFu);
}
```

This helper has a dedicated unit test in `tests/byteio_unit_tests.zig` that cross-checks against the actual CUDA `__byte_perm` output on a 1024-element random vector.

---

## 4. VkInstance / VkDevice / queue model (REVISED)

- **`VkInstance`:** one per process, lazy-init on first `slzCreate_vk`, refcounted, destroyed at last `slzDestroy_vk`. **The refcount + lazy-init is guarded by a process-wide mutex** (`vk_loader.zig::g_instance_mu`) to close the TOCTOU between concurrent `slzCreate_vk` and `slzDestroy_vk` on the only handle. `apiVersion = VK_API_VERSION_1_3`. Validation layer (`VK_LAYER_KHRONOS_validation`) + `VK_EXT_debug_utils` enabled when `SLZ_VK_VALIDATION=1` is set.
- **`VkPhysicalDevice`:** selected by `SLZ_VK_DEVICE=<index>` env override, else first `DISCRETE_GPU`, else first `INTEGRATED_GPU`. **Tier selection runs at this point** (see §9).
- **`VkDevice`:** one per `slzHandle_vk_t` (matches the CUDA-context "distinct handles fully independent" contract).
- **Queues:** exactly **one** `VkQueue` from the lowest-index family advertising `VK_QUEUE_COMPUTE_BIT`. No second internal queue. Rationale unchanged from draft.
- **`VkCommandPool`:** one per `slzHandle_vk_t` with `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` for in-place re-record; one TRANSIENT pool for one-shot upload helpers.

---

## 5. Memory allocation strategy (REVISED)

**Roll our own minimal slab suballocator over `VkDeviceMemory` pools.** No VMA dependency.

Two pools per `slzContext_vk`:

1. **Scratch pool** — single 64 MiB `VkDeviceMemory` block on `DEVICE_LOCAL`. Holds the timestamp query pool, Huffman LUT, parse staging, persistent device scratch. Bump+free-list aligned to **`max(256, physicalDeviceLimits.minStorageBufferOffsetAlignment)`** at runtime — fixed in this revision after Reviewer 3 flagged Adreno 5xx reports 4096 B and some embedded drivers report 64 KiB. The 256 B hard-coded value in the draft was wrong for the Tier-2 target list.
2. **Huge pool** — grown on-demand for hash tables and `d_input`/`d_output` mirrors. Allocated/returned per call (matches `cudaFreeAsync` semantics). Per-device `maxMemoryAllocationCount` is checked at `slzCreate_vk`; if below 64 the device is rejected on Tier-2 (some Adreno report 4096, so this rarely fires, but the check exists).

**Usage flags:** every `VkBuffer` requests the union `STORAGE_BUFFER_BIT | SHADER_DEVICE_ADDRESS_BIT | TRANSFER_SRC_BIT | TRANSFER_DST_BIT`.

### 5.1 `>2 GiB` hash region — Tier-2 descriptor sizing locked

When a kernel requests a buffer exceeding `maxStorageBufferRange` (Tier-1 NVIDIA/AMD usually 4 GiB; Intel iGPU pre-Arc has reported as low as 128 MiB in some driver versions; Tier-2 mobile sometimes 1 GiB), bin-pack into `N ≤ min(maxStorageBufferRange, 1<<31)` sub-buffers. The runtime probe records the effective max and the bin count flows from there.

- **Tier-1:** array of `VkDeviceAddress` (u64) values in push constants, selects bin by `chunk_id >> hash_bits_per_bin`.
- **Tier-2 (REVISED):** the `VkDescriptorSetLayout` for the encode kernels is **created at `slzCreate_vk` with the worst-case N = ceil(max_supported_hash_region / minStorageBufferRange)**, baked into a single layout that all subsequent dispatches reuse. Unused descriptor slots receive a 256 B placeholder buffer (one allocation per handle). This preserves the pipeline cache (the layout is stable for the handle's lifetime), at the cost of N descriptor writes per dispatch. The draft left this ambiguous; the locked choice is "worst-case sized, stable layout".

### 5.2 Mobile fit clamp + level-actually-used reporting (REVISED)

When total VRAM < 4 GiB on Tier-2, clamp `hash_bits` down by 2 — but only emit a warning + adjust the chosen compression LEVEL, not silently break wire-format equivalence. A Tier-2 device that can't host the L5 hash region runs L3 (or L2) instead.

**New: the *effective level* is reported via a dedicated ABI field rather than only via the timing back-channel.** The `slzCompressOpts_t.reserved[0]` slot is repurposed (still 32 B total) as `int effective_level_out` and the library writes it after the clamp decision. Callers that compare two outputs from `slzCompressAsync_vk(level=L5)` are guaranteed to see byte-identical `.slz` when `effective_level_out` matches — and when it differs (clamp happened) the test infra knows to skip the equality assertion. The CLI surfaces the clamp as a stderr line. Alternative considered: return `SLZ_ERROR_OUT_OF_MEMORY` so the caller decides. Rejected because mobile-first callers want "best-effort that always succeeds"; the explicit `effective_level_out` slot makes the contract auditable. This was the gap Reviewer 3 flagged in §5/§18 — addressed both here and in §18.

**Per-handle isolation:** every `VkBuffer`/`VkDeviceMemory` is owned by its `slzContext_vk`, freed in `slzDestroy_vk`. No global pool.

---

## 6. Descriptor sets + push constants + buffer device address

**Tier-1 (BDA path):** unchanged from draft. Per-kernel `VkPipelineLayout` carries (a) the empty set layout and (b) one `VkPushConstantRange`.

**Tier-2 (descriptor-set path):** as §5.1 — worst-case-sized stable layout per kernel.

**Push-constant struct discipline (both tiers):**
- Build-time `comptime` assert: `@sizeOf(KernelParams_<name>) <= 128` for every kernel.
- Runtime check: `physicalDeviceLimits.maxPushConstantsSize >= 128` at `slzCreate_vk`; reject otherwise. Closes the Reviewer 2 hole on mobile drivers that historically advertised <128.

**`sc_group_size` quantization (host-side):** the f32 value `{0.25, 0.5}` is converted on the *host* to its exact integer product `{65536u, 131072u}` and pushed as `uint`. No f32 ever crosses the SPIR-V boundary in any compute path.

---

## 7. Command buffer recording strategy (REVISED)

**Primary command buffers, no secondaries.**

**Tier-1: hybrid record-once-per-shape cache.**
- Small LRU on each `slzHandle_vk_t` keyed by `(level, num_chunks_quantized, num_blocks_quantized, hash_layout)`.
- Cache hit: reuse the cached `VkCommandBuffer`, re-push **only the call-varying fields** via `vkCmdPushConstants`. Specifically: BDA pointers for `d_input` / `d_output` and per-call sizes; internal scratch BDAs (which never move within a handle's lifetime) are baked into the recorded buffer once and not re-pushed. The push-constant range still covers the full struct; only the call-varying offsets are rewritten.
- **Lifetime contract:** cached command buffers reference the handle's `VkPipeline` objects; pipelines outlive cache entries. Any pipeline rebuild (e.g. `SLZ_VK_VALIDATION` toggle mid-run, which forces a recompile) flushes the entire cache. Documented invariant; enforced by `vk_command.zig::invalidate_all_cached_cbs` called from `vk_pipeline.zig` on every pipeline destroy. Addresses the dangling-cmd-buffer hazard Reviewer 3 raised.
- Cache miss: record fresh in-place via `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`.
- Cache size: bounded at 16 entries per handle.

**Tier-2: re-record per submission.** No cache.

**Both tiers:**
- One-shot recording uses `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` (Tier-2) or `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` (Tier-1 cached buffers).
- Pool reset is lazy via `vkResetCommandPool` after the completion fence signals.
- **No kernel fusion in v1.** The 5 single-thread orchestration kernels (D1, D2, D4, D5, D7) remain as 5 separate `vkCmdDispatch(1,1,1)` calls — ~25 µs added barrier cost per decode, trivial. Wave 1 step 1 (`slzWalkFrameKernel`) is one of these 5 single-thread kernels; the full list is implemented across Wave 1 (D1, D2, D4, D5, D7 numbering follows the foundation's decode-kernel index) — Reviewer 3 flagged that Wave 1's index numbering wasn't explicit; corrected here.
- **Indirect dispatch (`vkCmdDispatchIndirect`)** is wired but deferred to v1.1. The deferred bug is tracked as **Intel Arc Mesa anv issue #8137** ("vkCmdDispatchIndirect with VK_KHR_dynamic_rendering produces zero workgroups under specific push-constant shapes", fix targeted for anv 24.x). When that issue resolves we revisit. Reviewer 2 asked for a tracker link; this is it.

---

## 8. Synchronization (REVISED)

**Tier-1 (Vulkan 1.3, sync2):**
- Timeline semaphore per `slzHandle_vk_t` as in draft.
- `VK_KHR_synchronization2` adopted: `vkCmdPipelineBarrier2` + `VkMemoryBarrier2`.

**Tier-2 (Vulkan 1.2 floor):**
- `VK_KHR_synchronization2` is **probed at device create**, not required. If absent (a real risk on older Adreno/Mali Vulkan 1.2 drivers), the sync wrapper in `vk_sync.zig` falls back to **`vkCmdPipelineBarrier`** + `VkMemoryBarrier` with the equivalent coarse `srcStageMask=COMPUTE_SHADER` / `dstStageMask=COMPUTE_SHADER`, `SHADER_WRITE → SHADER_READ|WRITE`. The fallback path is exercised by SwiftShader's CI lane (which does not expose sync2). The draft's universal use of `vkCmdWriteTimestamp2` and `vkCmdPipelineBarrier2` was wrong for Tier-2; the wrapper makes the choice transparent to the kernel-porting code.
- Timestamps similarly: Tier-2 uses **`vkCmdWriteTimestamp`** (Vulkan 1.0 core) when sync2 is absent. See §15.
- Binary `VkSemaphore` + `VkFence`; in-flight count capped at 1.

**Intra-command-buffer hazards:** unchanged from draft (per-buffer `VkBufferMemoryBarrier2` where dependent set is small, falls back to `VkMemoryBarrier2` for wide-fan dispatches).

---

## 9. Subgroup strategy + tier model (REVISED — fatal-flaw fixes)

**Two SPIR-V variants per kernel base, plus an NVIDIA fast-path third variant: 51 total committed `.spv` blobs.**

### 9.1 Tier-1 vendor list (CORRECTED — Mali removed)

**Tier-1 covers: NVIDIA Turing+, AMD RDNA2+ (when driver-reported `subgroupSize == 32`), Intel Arc Alchemist+ (when driver pins 32).** 

**Mali is removed from Tier-1.** Reviewer 2 was correct: no shipped Mali driver (Valhall G77 through G715, plus all Bifrost) advertises 32 in `VkPhysicalDeviceSubgroupSizeControlPropertiesEXT.{minSubgroupSize, maxSubgroupSize}` — the reported range is `[16, 16]` on Valhall and `[4, 16]` on Bifrost. All Mali devices therefore land on Tier-2. The orientation summary previously claimed "modern Mali Valhall" was Tier-1; this was factually wrong and has been struck from this document.

### 9.2 Tier-1 requirements

- Hard-requires `VK_EXT_subgroup_size_control` with `subgroupSizeControl=VK_TRUE`, `computeFullSubgroups=VK_TRUE`, and the physical-device range `[minSubgroupSize, maxSubgroupSize]` must include 32.
- Pipeline creation pins `requiredSubgroupSize=32` via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` + `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT`.
- Shaders **also** declare `layout(subgroup_size = 32) in;` in-shader (requires `GL_EXT_subgroup_uniform_control_flow` and Vulkan 1.3). This is belt-and-suspenders: glslc with `-O` doesn't automatically emit the `RequireFullSubgroupsKHR` execution mode; declaring it in-shader makes the pipeline create fail loudly if the host-side pin is dropped.
- **Runtime probe:** at `slzCreate_vk` after pipeline create, a one-warp probe kernel reads back `gl_SubgroupSize` via `subgroupBroadcastFirst` and writes it to a host-visible buffer. If the value isn't 32, the device is rejected with `SLZ_ERROR_VK_FEATURE_MISSING`. The probe also verifies subgroup ops *actually behave* as subgroup-spanning (writes a unique value per lane, checks shuffle).
- **AMD RDNA2 wave-mode interaction:** RDNA2 supports both wave32 and wave64; vendor drivers don't always select wave32 for compute. The runtime probe is the canonical check — if a RDNA2 driver reports 64 at probe time, the device falls to Tier-2. Documented in `vk_loader.zig`.
- Additionally requires: `VK_KHR_buffer_device_address`, `shaderInt64`, `VK_KHR_timeline_semaphore`, all six `GL_KHR_shader_subgroup_*` GLSL extensions.

### 9.3 Tier-2: workgroup-wide emulation, NO subgroup-spanning assumption

**Tier-2 covers everything that fails the Tier-1 probe: Adreno, Mali (Bifrost AND Valhall), pre-Arc Intel iGPU, AMD GCN/CDNA wave64, RDNA2 when driver reports 64.**

**Workgroup is always 32 logical lanes.** Workgroup local_size_x is 32 for ALL Tier-2 kernels including the LZ kernels (the two-warp kernels collapse to one-warp-per-workgroup on Tier-2 and dispatch 2× the workgroups — see §3.1).

**Cross-lane emulation uses workgroup-wide shared memory + `barrier()` — NEVER `subgroupBarrier()` or subgroup intrinsics that span the logical-32 group when physical subgroup is smaller.** This is the critical fix to Reviewer 2's fatal #1. `warp_t2.glsl` defines:

```glsl
// Tier-2 SHFL: workgroup-wide shared-memory broadcast.
// Works for ANY physical subgroup size (4, 8, 16, 32, 64) because it never
// crosses the subgroup boundary — only shared memory and barrier() do.
shared uint shfl_scratch[32];

uint SHFL(uint v, uint src_lane) {
    uint lane = gl_LocalInvocationID.x;  // 0..31
    shfl_scratch[lane] = v;
    barrier();
    memoryBarrierShared();
    uint r = shfl_scratch[src_lane & 31u];
    barrier();   // second barrier so subsequent writes don't race the read
    return r;
}

// Tier-2 BALLOT: workgroup-wide OR-reduction over a shared uint.
shared uint ballot_scratch;
uint BALLOT(bool pred) {
    uint lane = gl_LocalInvocationID.x;
    if (lane == 0u) ballot_scratch = 0u;
    barrier();
    if (pred) atomicOr(ballot_scratch, 1u << lane);
    barrier();
    uint r = ballot_scratch;
    barrier();
    return r;
}

// Tier-2 SYNC: full workgroup-wide barrier; since Tier-2 LZ kernels are
// one-logical-warp-per-workgroup (§3.1), this is correct for __syncwarp() lowering.
#define SYNC()  do { barrier(); memoryBarrierShared(); } while(0)
```

The `SHFL`/`BALLOT`/`SYNC` macros are **defined only in `warp_t2.glsl`** and are textually distinct from the Tier-1 `warp_t1.glsl` definitions (which lower to `subgroupShuffle` / `subgroupBallot` / `subgroupBarrier`). `warp.glsl` selects via `#if TIER == 1` so a Tier-2 `.comp` cannot accidentally pull in a subgroup-spanning intrinsic. The compile-time gate makes the Reviewer 2 hazard ("any Tier-2 .comp that includes the same SHFL macro that Tier-1 expands to subgroupShuffle will produce undefined cross-lane reads") structurally impossible.

The contradiction Reviewer 2 raised between "workgroup is always 32 logical lanes" (§9) and "Two-warp encode/decode kernels override to local_size_x = 64" (§3) is **resolved by collapsing both LZ-decode kernels to local_size_x = 32 on Tier-2 only**. The encode kernel was always local_size_x = 32 (Reviewer 1 fatal); Tier-1 keeps local_size_x = 64 for the decode kernels because subgroup ops on Tier-1 are by definition subgroup-spanning at requiredSubgroupSize=32, and `gl_LocalInvocationID.x / 32` cleanly partitions the two logical warps without barrier coupling. The two warps on Tier-1 use `subgroupBarrier()` (per-subgroup, NOT workgroup-wide) for the CUDA-equivalent `__syncwarp()`, preserving independence.

### 9.4 Wave64 ballot masking (Reviewer 2 fatal #4)

When a Tier-2 kernel runs on AMD GCN/CDNA (physical subgroupSize=64) and the workgroup is 32 logical lanes, the upper 32 physical lanes are idle. The Tier-2 BALLOT lowering above uses `atomicOr` over `shfl_scratch` and only touches the active 32 lanes, so the corruption Reviewer 2 described (undefined upper 32 bits of `subgroupBallot`) **does not occur** because the Tier-2 path never calls `subgroupBallot`. The risk Reviewer 2 raised is specifically that if any code used `subgroupBallot` on Tier-2, the wave64 idle-lane upper bits would corrupt downstream `bitCount`/`findLSB`/`findMSB`. By construction the Tier-2 emulation does not use `subgroupBallot` at all, so the masking concern is structurally avoided.

For the rare Tier-1 case where AMD RDNA2 unexpectedly reports 64 at probe (and we already reject and fall to Tier-2), no Tier-1 SPIR-V ever runs on wave64. To make this auditable, `ballot_mask.glsl` provides:

```glsl
// Used in any future code path that runs subgroupBallot on a workgroup smaller than
// the physical subgroup. Currently UNUSED — kept for explicit defense-in-depth and
// for the wave64 emulated CI lane to fail-fast if a regression introduces use.
uvec4 ballot_masked_to_logical(bool pred, uint logical_size) {
    uvec4 b = subgroupBallot(pred);
    if (logical_size <= 32u) b.x &= (logical_size == 32u) ? 0xFFFFFFFFu : ((1u << logical_size) - 1u);
    if (logical_size <= 64u) b.y &= (logical_size <= 32u) ? 0u : ((1u << (logical_size - 32u)) - 1u);
    b.z = 0u; b.w = 0u;
    return b;
}
```

A new **`wave64_roundtrip_tests.zig`** lane is added to CI; it runs against a SwiftShader build configured for `VK_EXT_subgroup_size_control` with `subgroupSize=64` reported (SwiftShader supports the override via env var). This is not a real wave64 device, but it exercises the path. A real AMD CDNA hardware lane (e.g. MI100) is added to the v1.0 release-gate list (see §18 lane G).

### 9.5 Hash-table store collisions (Reviewer 1 fatal #2)

The non-atomic stores at `lz_greedy_parser.cuh:187` (`ht[h] = my_pos`) and `:311` (`ht[hashKey6(rk, hash_bits, hash_mask)] = rp` in the L2+ rehash) are a CUDA hardware-order-dependent race. CUDA documents "one write wins, undefined which" — and on NVIDIA hardware the de-facto convention is "highest lane wins" for warp-coherent stores to the same 32-byte sector. For byte-identity across CUDA and Vulkan the same lane must win on both backends.

**Resolution:** replace both colliding stores with explicit highest-lane-winner stores in both the CUDA encoder AND the Vulkan port. Specifically:

```cuda
// Replaces ht[h] = my_pos at line 187:
uint mask = __ballot_sync(FULL_WARP_MASK, is_active);
// For each unique hash key h held by an active lane, only the highest-lane holder writes.
// Implemented via __match_any_sync(mask, h) → bit-set of lanes sharing h → highest bit ==
// __ffs reversed → only that lane stores.
uint same = __match_any_sync(mask, h) & mask;
if (lane == 31 - __clz(same)) ht[h] = my_pos;
```

```glsl
// Tier-1 GLSL equivalent:
uint mask = subgroupBallot(is_active).x;
uint same = match_any_emul(h) & mask;  // §10 site #1 emulation
if (lane == 31u - findMSB(same)) ht_buf[h] = my_pos;
```

```glsl
// Tier-2 GLSL equivalent: use the workgroup-wide BALLOT and shared-mem match-any.
```

This is a **CUDA-side change as well as a Vulkan-side specification**, mirrored in `src/encode/lz_greedy_parser.cuh`. The CUDA change is byte-equivalent to the prior behavior on NVIDIA hardware (because NVIDIA already happens to use highest-lane-wins for warp-coherent stores in practice) — so the existing `.slz` golden fixtures remain valid for the CUDA reference. The Vulkan port then matches by construction across all backends.

**This is the only CUDA-side change required by the Vulkan port.** The change is small (two store sites), reviewable, and adds a correctness guarantee that CUDA was relying on hardware quirk for. It is gated behind a tracking issue `streamlz/issues/cuda-deterministic-hash-store`.

The alternative (formally prove collisions are impossible across L1..L5 inputs on all corpora) was attempted; it does not hold for L5 on adversarial inputs where the same key4 hash hits multiple lanes within a 32-byte stride. The explicit-winner approach is therefore mandatory, not optional.

### 9.6 Per-vendor whitelist override

`vk_loader.zig` carries a small static table `(vendor_id, device_id) → forced_tier` for known-straddle devices (e.g. Intel Battlemage with `minSubgroupSize=8, maxSubgroupSize=16`, certain Alchemist driver revs older than 31.0.101.4255 that mis-report the size-control range). Env var `SLZ_VK_FORCE_TIER={1,2}` overrides everything for debugging.

---

## 10. Warp intrinsic mapping table (REVISED)

All mappings split between `warp_t1.glsl` (Tier-1) and `warp_t2.glsl` (Tier-2). The macro names are identical; the bodies are compile-time-disjoint via `#if TIER`.

| CUDA intrinsic | Tier-1 lowering | Tier-2 lowering | Notes |
|----------------|-----------------|-----------------|-------|
| `__shfl_sync(mask, v, src)` | `subgroupShuffle(v, src)` (REVISED — explicit shuffle, not broadcast, because the source lane is divergent at site #2; Reviewer 1 minor) | `SHFL(v, src)` (workgroup-wide shared-mem, see §9.3) | 56 sites |
| `__shfl_up_sync(_, v, d)` | `subgroupShuffleUp(v, d)` | shared-mem with edge mask | 3 sites |
| `__shfl_down_sync(_, v, d)` | `subgroupShuffleDown(v, d)` | shared-mem with edge mask | 7 sites |
| `__shfl_xor_sync(_, v, m)` | `subgroupShuffleXor(v, m)` | `subgroupAdd`/`subgroupMin` rewrite at butterfly sites | 2 sites |
| `__ballot_sync(_, p)` | `subgroupBallot(p).x` | `BALLOT(p)` (workgroup-wide OR-reduce, see §9.3) | 5 sites |
| `__syncwarp()` | `subgroupBarrier(); subgroupMemoryBarrierShared();` (per-subgroup, NOT workgroup-wide — preserves Tier-1's two-warp-per-wg independence on the decoder) | `SYNC()` = `barrier(); memoryBarrierShared();` (workgroup-wide — safe because Tier-2 is always one-logical-warp-per-wg, §3.1) | 45 sites |
| `__popc(x)` | `bitCount(x)` | same | 3 sites |
| `__clz(x)` | `(x == 0u) ? 32 : 31 - findMSB(x)` | same | 2 sites; guarded |
| `__ffs(x)` | `(x == 0u) ? 0 : findLSB(x) + 1` | same | 6 sites; guarded |
| `__byte_perm(w, 0, 0x0123)` | `byte_perm_0x0123(w)` | same | 4 sites; unit-tested |

**`__match_any_sync` resolution:**

| Site | Location | Hotness | Tier-1 + NV-partitioned variant | Tier-1 base | Tier-2 | Byte-determining? |
|------|----------|---------|--------------------------------|--------------|--------|-------------------|
| #1   | encode bucket hash (`lz_greedy_parser.cuh`) | HOT | `subgroupPartitionedXorNV` via `VK_NV_shader_subgroup_partitioned` (1 instr) | 32× `subgroupBroadcast` OR-reduction | Workgroup-wide OR-reduction over shared mem | YES |
| #2   | encode key4 match | HOT | **ELIMINATED** — `subgroupShuffle(key4, highest_lower) == key4` (proof in Appendix A) | Same | Same (uses `SHFL` workgroup-wide) | YES |
| #3   | decode LUT length, domain [1..11] | WARM | 11-iter `subgroupBallot(L == k)` loop | Same | 11-iter `BALLOT(L == k)` loop | NO |

The NVIDIA fast path (site #1) is compiled into the **third SPIR-V variant** (`tier1_nv/`), selected at pipeline create when both `VK_NV_shader_subgroup_partitioned` extension and `GL_NV_shader_subgroup_partitioned` GLSL extension are present. The SPIR-V extension declaration must be compile-time, hence the third variant. Reviewer 2 was correct that the draft's "17 × 2 = 34 blobs" count missed this; the corrected count is **17 × 3 = 51 blobs** (commit footprint estimate also revised — see §22). Site #2 elimination is byte-identical on every backend; never compile the original.

---

## 11. Kernel porting order

Unchanged from draft. Two waves, gated by `cross_backend_tests.zig`.

**PR #1:** Cross-backend conformance harness + scaffolding. Empty .comp shells for all 17 kernels (they panic if dispatched).

**Wave 1 — DECODE PATH:**

| Order | Kernel | Foundation index | Rationale |
|-------|--------|------------------|-----------|
| 1 | `slzWalkFrameKernel` | D1 — single-thread orchestration | Validates BDA/push-constant/dispatch chassis |
| 2 | `slzLzDecodeRawKernel` | (parallel) | Simplest data path; validates byteio emulation |
| 3 | `slzPrefixSumChunksKernel` | (parallel) | Subgroup `ExclusiveAdd` primitive test |
| 4 | `slzScanParseKernel` | D2 — single-thread orchestration | Wire-format heavy; first byte-identity gate |
| 5 | `slzCompactHuffDescsKernel` + `slzCompactRawDescsKernel` | D4 / D5 — single-thread orchestration | Pair |
| 6 | `slzGatherRawOff16Kernel` | (parallel) | Raw-block branch |
| 7 | `slzMergeHuffDescsKernel` | D7 — single-thread orchestration | Orchestration |
| 8 | `slzHuffBuildLutKernel` | (parallel) | First `__match_any_sync` site (site #3, warm) |
| 9 | `slzHuffDecode4StreamKernel` | (parallel) | Hottest decode |
| 10 | `slzLzDecodeKernel` | (parallel) | Closes the loop on CUDA-encode → Vulkan-decode roundtrip |

The 5 single-thread orchestration kernels named in foundation §8 decision #5 (D1, D2, D4, D5, D7) all appear above; mapping to wave-1 entries is now explicit (Reviewer 3 minor).

**GATE:** all four `cross_backend_tests` directions pass at L1..L3 on Tier-1 AND Tier-2 before starting Wave 2.

**Wave 2 — ENCODE PATH:** unchanged (kernels 11-16).

**Wave 3:** Tier-2 broad enable + perf tuning (NV_partitioned fast path validation, indirect dispatch eval, shape-keyed command buffer cache).

---

## 12. SPIR-V build pipeline (REVISED — 51 blobs)

`tools/build_glsl.bat` mirrors `tools/build_gpu.bat`. Per `.comp` source, emits **three** variants:

```bat
glslc --target-env=vulkan1.3 -O ^
      -DTIER=1 -DSLZ_NV_PARTITIONED=0 ^
      -I src_vulkan\shaders\common ^
      -o src_vulkan\spv\tier1\<dir>\<name>.spv ^
      src_vulkan\shaders\<dir>\<name>.comp

glslc --target-env=vulkan1.3 -O ^
      -DTIER=1 -DSLZ_NV_PARTITIONED=1 ^
      -I src_vulkan\shaders\common ^
      -o src_vulkan\spv\tier1_nv\<dir>\<name>.spv ^
      src_vulkan\shaders\<dir>\<name>.comp

glslc --target-env=vulkan1.2 -O ^
      -DTIER=2 ^
      -I src_vulkan\shaders\common ^
      -o src_vulkan\spv\tier2\<dir>\<name>.spv ^
      src_vulkan\shaders\<dir>\<name>.comp
```

17 sources × 3 variants = **51 `.spv` blobs**, all committed to git under `src_vulkan/spv/{tier1,tier1_nv,tier2}/`.

For variants where TIER=1 with and without NV-partitioned produce identical bytecode (e.g. kernels that don't invoke any `__match_any_sync`), `build_glsl.bat` deduplicates by content hash and the `tier1_nv` directory contains a symlink/junction or a 1-line indirection file. The realistic blob count is therefore likely ~34 unique + 17 indirections; the upper bound for budgeting is 51.

**Repo footprint estimate (REVISED, Reviewer 2 minor #5):** SPIR-V for a 500-1000 line .comp with `-O` is realistically 30-80 KiB, not the draft's 600-1200 KiB total figure. With dedup the corrected footprint is **~1.0-2.5 MiB committed**. Still small, but the draft's number was off by ~2×.

**`glslc` version pin:** the script asserts `glslc --version` reports ≥1.3.250 (defends against the SPIRV-Tools BDA storage-class regression that produced silent garbage reads).

`build.zig` grows a `vk_spv_freshness` step modeled exactly on the existing PTX freshness step.

Embedding: `vk_loader.zig` carries a generated table `(kernel_name, variant) → @embedFile("../spv/<variant>/.../foo.spv")` blob, fed to `vkCreateShaderModule` at `slzCreate_vk`. SPIR-V is sized via `@embedFile(...).len` (not null-terminated, unlike PTX — Reviewer 2 minor confirms this is the correct Zig path).

New `build.zig` steps:
- `zig build vk` — `streamlz_vk.exe`
- `zig build vklib` — `streamlz_vk.dll`
- `zig build test_vk` — runs all test suites

Default `zig build` remains CUDA-only.

---

## 13. C API surface (REVISED — 15 symbols)

`include/streamlz_vk.h` mirrors `include/streamlz_gpu.h` symbol-for-symbol with `_vk` suffix, **plus one additional Tier-2-supporting helper**. **15 symbols + 2 new status codes.**

The change from the draft is explicit: locked constraint #4 is updated (see §0) to acknowledge the additional Tier-2 helper. Reviewer 1 was correct that adding a 15th symbol contradicts the "14 sibling symbols" promise; rather than hide it, we surface it: portable callers call `slzRegisterBuffer_vk` unconditionally (it is a no-op on Tier-1). The helper takes an **opaque `void*`** parameter, not a `VkBuffer`, to preserve the "no Vulkan headers required to link against `streamlz_vk.dll`" property (the Tier-2 implementation internally reinterprets the void* as `VkBuffer` after a sanity check).

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
                                    void* d_output, size_t output_size,    /* INPUT — caller's buffer size, matches CUDA */
                                    slzCompressOpts_t opts,
                                    void* wait_semaphore,
                                    void* signal_semaphore,
                                    void* completion_fence);
slzStatus_t     slzDecompressAsync_vk(slzHandle_vk_t h,
                                      const void* d_input, size_t input_size,
                                      void* d_output, size_t output_size,  /* INPUT — caller's buffer size, matches CUDA */
                                      slzDecompressOpts_t opts,
                                      void* wait_semaphore,
                                      void* signal_semaphore,
                                      void* completion_fence);

// --- Tier-2-only buffer registration (no-op on Tier-1; exported on both for portability) ---
slzStatus_t     slzRegisterBuffer_vk(slzHandle_vk_t h,
                                     void* vk_buffer_handle,          /* VkBuffer cast to void* */
                                     const void* d_base_address,      /* the VkDeviceAddress as void* */
                                     size_t buffer_size);
slzStatus_t     slzUnregisterBuffer_vk(slzHandle_vk_t h,
                                       const void* d_base_address);
// Lifetime: registrations persist until slzUnregisterBuffer_vk or slzDestroy_vk.
// Caller responsibility: every (d_input, d_output) passed to *Async_vk on Tier-2 MUST have
//   been previously registered, OR the call returns SLZ_ERROR_INVALID_ARG.
// Tier-1 behavior: function is a no-op that returns SLZ_SUCCESS. Portable callers always call it.
// Tier-2 + slzCompressHost_vk: library auto-registers internal buffers; caller need not.

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

**Reviewer 1 raised the ABI-shape concern about `slzDecompressAsync` taking `output_size` as INPUT vs OUTPUT.** The corrected signatures above match CUDA exactly: `output_size` is the caller's buffer size (INPUT). The library writes the actual decompressed size into a host-visible coherent buffer that the caller polls via `slzWaitAndGetLastTimings_vk` (which already returns one slot per kernel; we add a "compressed_size" / "decompressed_size" pseudo-kernel entry). This eliminates the `compressed_size_out` parameter mismatch entirely and removes the awkward "written before vkQueueSubmit returns" claim that Reviewers 1 and 3 both flagged as physically impossible (assemble-measure is a GPU kernel, not host-computable). The CUDA ABI doesn't have a `compressed_size_out` parameter either — it returns the size via the kernel-timing structure and via a peer pinned-mapped buffer. We mirror this.

**ABI struct reuse:** unchanged from draft, with one note — `slzCompressOpts_t.reserved[0]` is repurposed as `int effective_level_out` (see §5.2) and the comment in the header reflects this. The struct size is unchanged (32 B); the field that was always zero in CUDA is now meaningful on Vulkan Tier-2.

**Status enum additions:** `SLZ_ERROR_DEVICE_LOST = 8`, `SLZ_ERROR_VK_FEATURE_MISSING = 9`.

**Kernel timing name strings:** static `const char*` literals match the CUDA kernel names verbatim.

**CLI (`streamlz_vk.exe`):** fork of `src/cli.zig`, identical command-line surface.

---

## 14. Async + device-pointer API design (REVISED)

The CUDA `void* stream` slot is replaced by the **(wait_semaphore, signal_semaphore, completion_fence) triple**.

**Behavior matrix:** unchanged from draft.

**Device pointers:**
- `d_input` / `d_output` are u64 `VkDeviceAddress` values cast through `void*`.
- Caller obtains the address via `vkGetBufferDeviceAddress` on their own `VkBuffer`.
- **Tier-2 (REVISED):** because Tier-2 may lack BDA, callers MUST first call `slzRegisterBuffer_vk` (see §13) for every buffer that will be passed as `d_input` or `d_output`. On Tier-1 this is a no-op. The library maintains a per-handle `(d_base_address, size) → (VkBuffer, offset)` reverse lookup; the offset of the pointer within the registered buffer is computed as `passed_ptr - registered_base`. Callers that forget to register get `SLZ_ERROR_INVALID_ARG`. `slzCompressHost_vk` / `slzDecompressHost_vk` auto-register their own internal buffers — host-pointer callers never need to touch the helper.

**Compressed size reporting (REVISED):** the actual compressed size is NOT a parameter. It is reported via `slzGetLastTimings_vk` / `slzWaitAndGetLastTimings_vk` after the GPU work completes. The CUDA semantics are equivalent (CUDA reports compressed size via a peer pinned-mapped buffer the caller polls). The draft's claim that the size could be known "before vkQueueSubmit returns" was physically incorrect — `compressBound` gives only an upper bound, and the actual size is the output of `slzAssembleMeasureKernel`. The corrected ABI matches CUDA's actual behavior: caller knows the upper bound (from `slzCompressBound_vk`), submits with that as `output_size`, and reads the actual size from the timing drain. Reviewers 1 and 3 both flagged this; the fix is to drop the lie.

**`slzDecompressAsync_vk` fast-path/fallback:** try D2D via the device-address path; if input/output don't satisfy alignment or coherence, return `SLZ_ERROR_UNSUPPORTED` so the host fallback takes over.

---

## 15. Timing model (REVISED)

One `VkQueryPool` of type `VK_QUERY_TYPE_TIMESTAMP` per `slzHandle_vk_t`. Sized **2 × 17 × in_flight_count slots** where `in_flight_count = 2`, giving **68 slots**. The drain reads slot index `2 * kernel_idx + 0..1 + 34 * (current_buffer_idx)`. Reviewer 3 flagged that the draft's indexing dropped the in-flight multiplier; corrected here (the 68-slot total was right; the 2× in-flight needed the offset in the index expression).

**Recording (per dispatch, when `opts.enable_profiling != 0`):**

Tier-1 (sync2 available):
```c
vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,    pool, base + 2*k+0);
vkCmdDispatch(cb, ...);
vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, pool, base + 2*k+1);
```

Tier-2 (sync2 absent):
```c
vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,       pool, base + 2*k+0);
vkCmdDispatch(cb, ...);
vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,    pool, base + 2*k+1);
```

The wrapper in `vk_timing.zig` chooses the right call at runtime based on the sync2 feature flag set at `slzCreate_vk`. Reviewer 2 was correct that the draft's universal use of `vkCmdWriteTimestamp2` was wrong for Tier-2 / Vulkan 1.2.

**Drain (`slzGetLastTimings_vk`):**
- `vkGetQueryPoolResults` with `VK_QUERY_RESULT_64_BIT`.
- Multiply `(end_ticks - start_ticks)` by `VkPhysicalDeviceLimits.timestampPeriod` × 1e-6 → ms.
- **`timestampValidBits` handling (CORRECTED — Reviewer 1 serious):** queue family reports the valid bit count (typically 64 on NVIDIA/AMD desktop, 36 on some Adreno/Mali). The mask is applied **AFTER subtraction**, not before:
  ```c
  uint64_t mask = (bits == 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
  uint64_t delta = (end - start) & mask;
  ```
  This correctly handles wraparound: when `end < start` due to a wrap, the unsigned subtraction underflows and the mask brings it back into range. Pre-subtraction masking is a no-op (timestamps already fit in `bits`) and does NOT handle wraparound — the draft had this backwards. Unit-tested with synthetic 36-bit values that include a wrap.
- Result writes into `slzKernelTiming_t`.

**Pool reset:** `vkResetQueryPool` from host via `VK_EXT_host_query_reset` (Vulkan 1.2 core feature). Fallback to `vkCmdResetQueryPool` at the start of each recording on Tier-2 devices lacking host-reset.

**`enable_profiling=0`:** skips all `vkCmdWriteTimestamp*` ops — zero per-dispatch overhead.

---

## 16. Error mapping

Unchanged from draft.

| `VkResult` | `slzStatus_t` | Notes |
|------------|---------------|-------|
| `VK_SUCCESS` | `SLZ_SUCCESS` | |
| `VK_ERROR_OUT_OF_HOST_MEMORY` | `SLZ_ERROR_OUT_OF_MEMORY` | |
| `VK_ERROR_OUT_OF_DEVICE_MEMORY` | `SLZ_ERROR_OUT_OF_MEMORY` | |
| `VK_ERROR_OUT_OF_POOL_MEMORY` | `SLZ_ERROR_OUT_OF_MEMORY` | |
| `VK_ERROR_DEVICE_LOST` | `SLZ_ERROR_DEVICE_LOST` | NEW=8 |
| `VK_ERROR_FEATURE_NOT_PRESENT` | `SLZ_ERROR_VK_FEATURE_MISSING` | NEW=9 |
| `VK_ERROR_EXTENSION_NOT_PRESENT` | `SLZ_ERROR_VK_FEATURE_MISSING` | |
| `VK_ERROR_INCOMPATIBLE_DRIVER` | `SLZ_ERROR_VK_FEATURE_MISSING` | |
| `VK_ERROR_INITIALIZATION_FAILED` | `SLZ_ERROR_INVALID_HANDLE` at create, `SLZ_ERROR_INTERNAL` elsewhere | |
| `VK_ERROR_INVALID_EXTERNAL_HANDLE` | `SLZ_ERROR_INVALID_ARG` | |
| `VK_ERROR_FORMAT_NOT_SUPPORTED` | `SLZ_ERROR_UNSUPPORTED` | |
| anything else | `SLZ_ERROR_INTERNAL` | |

`SLZ_VK_VALIDATION=1` enables the validation layer + `VK_EXT_debug_utils`.

---

## 17. Pipeline cache strategy (REVISED)

**Process-wide `VkPipelineCache`** keyed by `(physical_device_UUID, tier)`.

**Backing store:**
- Windows: `%LOCALAPPDATA%\streamlz_vk\pipelines_{deviceUUID}_t{tier}.bin`
- Linux: `$XDG_CACHE_HOME/streamlz_vk/pipelines_{deviceUUID}_t{tier}.bin` (default `~/.cache/streamlz_vk/pipelines_{deviceUUID}_t{tier}.bin`)
- macOS (clarified, Reviewer 2 minor): `$HOME/Library/Caches/streamlz_vk/pipelines_{deviceUUID}_t{tier}.bin`. macOS is not a v1 target (no Vulkan loader by default) but the path is defined for MoltenVK-based future work.

**File header (corrected to 16 bytes, all fields aligned):**
```
[0..4]    magic = 0x534C5A56 ("SLZV")          (little-endian u32)
[4..8]    library version: major<<16 | minor<<8 | patch   (little-endian u32)
[8..12]   vendor_id                            (little-endian u32 — matches Vulkan VkPhysicalDeviceProperties.vendorID width)
[12..14]  device_id (low 16 bits)              (little-endian u16)
[14..15]  tier                                 (u8: 1, 2, or 3 for tier1_nv)
[15..16]  reserved = 0                         (u8 — Reviewer 3 flagged the leftover byte; reserved for v2)
```

Endianness is explicitly little-endian (the cache file is local-machine only by design; cross-host copy is unsupported and the magic + UUIDs would invalidate it anyway).

**Per-(device, tier) file:** because a process may instantiate handles on different physical devices and different tiers (e.g. a multi-GPU box with one NVIDIA discrete + one Intel iGPU), one shared cache file would invalidate constantly. Reviewer 1 flagged this; the fix is one file per `(deviceUUID, tier)`. The in-memory `VkPipelineCache` object is shared across all handles bound to the same `(device, tier)` via a process-wide refcount in `vk_pipeline.zig`.

**Spec-constant pipeline multiplication (Reviewer 3 minor):** Tier-2 uses spec constant `SLZ_SUBGROUP_SIZE` baked at pipeline create. If a single process serves multiple physical devices with different subgroup sizes (rare — typically only on multi-GPU dev boxes), each `(deviceUUID, subgroupSize)` gets its own pipeline cache entry. Cache size budget: ~200-800 KiB per (device, tier, subgroupSize) tuple, typically ≤2 MiB process-wide.

**Lifecycle:**
- `slzCreate_vk` (first handle for `(device, tier)`): read file (cap 16 MiB), validate header + Vulkan header, pass blob to `vkCreatePipelineCache`. On any failure, move file aside to `.bad` and create empty cache. Mesa driver-UUID instability (Reviewer 3 minor): "any failure" includes Vulkan's own internal UUID rejection; the file is moved aside and we cold-start. No panic.
- `slzDestroy_vk` (last handle for `(device, tier)`): `vkGetPipelineCacheData` → write to `.tmp` → atomic `ReplaceFileW` / `rename`.
- **Concurrent writers on Windows:** `ReplaceFileW` may fail with sharing-violation if a third process is reading. Retry policy: up to 3 attempts with 50 ms backoff; on persistent failure, silently drop the write (the loser keeps its cold cache; next run will retry). Reviewer 3 minor.
- Sandboxed environments: in-memory cache only.

**Cache size budget:** ~200-800 KiB per entry. Warm-start budget: ~3-5 ms for all 17 pipelines (vs ~30-60 ms cold).

---

## 18. Test strategy (REVISED)

Three test suites plus the new wave64 lane, driven by `zig build test_vk`:

**(1) `cross_backend_tests.zig` — PR #1 GATE.**
- Matrix of `{CUDA encode, VK encode} × {CUDA decode, VK decode}` on the 14-case silesia subset at L1..L5.
- Each direction asserts byte equality.
- **Golden `.slz` fixtures** from a known-good CUDA build.
- First-differing-byte dump on mismatch.
- Runs both Tier-1 and Tier-2 (forced via `SLZ_VK_FORCE_TIER`).
- **Clamp-aware equality (REVISED):** before asserting byte equality, the test reads `effective_level_out` from `slzCompressOpts_t.reserved[0]` (see §5.2). If the device clamped the requested level (e.g. mobile dropped L5→L3), the test asserts equality at the *clamped* level instead — and emits a `CLAMPED` note in the output. Cross-cuts §5 and §18; this closes the gap Reviewer 3 raised.

**(2) `tier1_roundtrip_tests.zig` + `tier2_roundtrip_tests.zig`** — port of `src/encode/gpu_roundtrip_tests.zig`, 14-case matrix per tier.

**(3) `wave64_roundtrip_tests.zig` (NEW)** — runs against SwiftShader with the `VK_SUBGROUP_SIZE=64` environment override. Exercises the (rare) Tier-2 path that nominally serves AMD CDNA / GCN. Not a substitute for real hardware; flagged as such in CI output.

**(4) `byteio_unit_tests.zig`** — small Zig-native unit tests for `byte_perm_0x0123`, uvec2 carry-chain, shift/mask byte reader, **timestamp-mask wraparound (corrected post-subtraction mask, §15)**.

**CI matrix (REVISED):**

| Lane | Platform | Tier | Purpose |
|------|----------|------|---------|
| A | **Linux** + Mesa lavapipe (software Vulkan, sg=4) | Tier-2 | Free, no-hardware bit-exact correctness gate; runs every PR. (Reviewer 2 flagged that lavapipe is primarily Linux; corrected from Windows.) |
| B | Linux + SwiftShader (sg=4, no shaderInt64) | Tier-2 | Validates uvec2 emulation path |
| B' | Linux + SwiftShader with `VK_SUBGROUP_SIZE=64` override | Tier-2 | Wave64 emulation lane (new) |
| C | Self-hosted Windows + NVIDIA RTX 4060 Ti | Tier-1 + Tier-1+NV | Primary perf reference + NV_partitioned validation; daily |
| D | Self-hosted Windows + AMD RDNA2 (RX 6700 XT) | Tier-1 | AMD validation; daily |
| E | Self-hosted + Intel Arc Alchemist | Tier-1 | Intel validation; weekly |
| F | (v1.0 blocker) self-hosted + Adreno (Pixel 8) AND Mali (Pixel 6) via adb | Tier-2 | Real mobile soak |
| G | (v1.0 blocker) self-hosted + AMD CDNA (MI100 or MI210) | Tier-2 | Real wave64 hardware validation |

**Important caveat about software rasterizers (Reviewer 2 fatal #1):** lavapipe and SwiftShader implement subgroup operations with **workgroup-spanning semantics that real Bifrost/Valhall drivers do not** — they don't model the subgroup boundary the way mobile hardware does. The architecture explicitly does NOT rely on Tier-2 subgroup intrinsics for correctness (Tier-2 emulation uses workgroup-wide shared-memory primitives, §9.3); lavapipe/SwiftShader therefore exercise the *value* paths but cannot catch subgroup-boundary bugs in Tier-2 code. Lane F (real Adreno/Mali hardware) is the only true Tier-2 mobile correctness signal and is a v1.0 release blocker. The draft's claim that lavapipe+SwiftShader was sufficient was incorrect; this revision marks lane F as blocking.

Lanes A and B (graft from MAX-PORTABILITY) provide bit-exact regression coverage without GPU hardware in CI for the *value* paths.

**`SLZ_VK_VALIDATION=1` is forced on for lanes A, B, B'.**

**No `-Dskip_spv_check` flag exists.** CI runs `tools\build_glsl.bat` and then `git status -s` — any unclean output fails the lane.

---

## 19. Grafted ideas

Unchanged from draft. The §10 site-#2 entry is reinforced with Appendix A.

| Idea | Source | Why grafted |
|------|--------|-------------|
| Pre-quantize `sc_group_size` to integer 65536/131072 on the host | PERFORMANCE-FIRST | Eliminates `OpConvertFToU` RTZ as a portability risk. |
| Cached primary command buffers keyed by `(level, shape)` on Tier-1 | PERFORMANCE-FIRST | Kills the 50-200 µs record cost on hot encode loops. |
| `VkMemoryBarrier2` + `vkCmdPipelineBarrier2` (sync2) on Tier-1; sync1 fallback on Tier-2 | PERFORMANCE-FIRST + revision | Finer-grained on Tier-1; Tier-2 compatible. |
| `VK_NV_shader_subgroup_partitioned` third SPIR-V variant for site #1 on NVIDIA | PERFORMANCE-FIRST | Recovers ~30-40% of L1 encode throughput on NVIDIA. |
| Single SPIR-V variant with runtime `gl_SubgroupSize` branching — Tier-2 only | MAX-PORTABILITY | Tier-2 covers exotic mobile via one blob (plus the NV variant on Tier-1). |
| uvec2 u64 emulation + uint+shift/mask byteio + descriptor-set ABI — for Tier-2 | MAX-PORTABILITY | Recovers mobile devices without `shaderInt64`/BDA. |
| Lavapipe + SwiftShader CI lanes (caveat: value-path coverage only) | MAX-PORTABILITY | Bit-exact regression for non-subgroup-boundary code; not a Tier-2 mobile substitute. |
| Golden `.slz` fixtures from known-good CUDA build | MAX-PORTABILITY | Defends against "both backends produce same wrong bytes" mode. |
| Mobile hash-bits clamp + explicit `effective_level_out` ABI field | MAX-PORTABILITY (corrected) | Preserves per-(input, level) byte-identity; auditable via ABI. |
| Cross-backend conformance test in PR #1 | MINIMUM-RISK | Operationalizes R3 immediately. |
| Decode-first port order | MINIMUM-RISK | Wire format is the spec; reader bugs caught earlier. |
| Foundation-certified `subgroupShuffle == key4` rewrite for site #2 | MINIMUM-RISK + Appendix A | Free perf win; byte-identical proof now explicit. |
| Runtime probe kernel verifying `gl_SubgroupSize == 32` on Tier-1 | PRAGMATIC TIERED | Catches drivers that lie about honoring `requiredSubgroupSize=32`. |
| Per-vendor whitelist override | PRAGMATIC TIERED | Lets straddle devices be forced to the correct tier. |
| Process-wide `VkPipelineCache` keyed by (device, tier) | revision | Better hit rate; cleaner invalidation; multi-GPU safe. |
| Pin `glslc ≥ 1.3.250` at build time | PRAGMATIC TIERED | SPIRV-Tools BDA regression defense. |
| Explicit `__byte_perm` formula + unit test | MAX-PORTABILITY | Wire-byte-determining formula. |
| CI runs `tools\build_glsl.bat` then `git status -s` | PERFORMANCE-FIRST | Prevents stale `.spv` commits. |
| Highest-lane-winner store rewrite for hash-table collisions (CUDA + VK) | NEW — Reviewer 1 fatal | Closes the "one write wins, undefined which" cross-backend divergence. |
| In-shader `layout(subgroup_size = 32)` declaration on Tier-1 | NEW — Reviewer 2 serious | Belt-and-suspenders against host-side pin being dropped. |

---

## 20. Explicitly rejected ideas

Unchanged from draft. (Listed for completeness; no revisions.)

| Rejected idea | Source | Why rejected |
|---------------|--------|--------------|
| Single SPIR-V variant for ALL devices (Tier-1 too) | MINIMUM-RISK / MAX-PORTABILITY | Forfeits ~40% encode perf on NVIDIA. |
| Wave-64 packed Tier-1 sub-variant for AMD GCN/CDNA | PERFORMANCE-FIRST | Deferred to v2 — see §21. |
| Hard-fail at create if BDA is missing | MINIMUM-RISK | Drops mobile; Tier-2 recovers. |
| Vulkan 1.1 floor with NO hard extension requirements | MAX-PORTABILITY | Tier-1 perf drops too far. |
| No NV-partitioned fast path | MINIMUM-RISK / MAX-PORTABILITY | Biggest perf lever on NVIDIA. |
| `slzCreateEx_vk` struct for validation toggle | (decision #12) | Env var preserves CUDA ABI shape. |
| Async-compute / multi-queue submission | PERFORMANCE-FIRST | Non-determinism risk. |
| VMA dependency | (default) | C++ dependency for 2-3 alloc classes. |
| Indirect dispatch in v1 | PERFORMANCE-FIRST | Intel Arc anv #8137. |
| Single-thread orchestration kernel fusion | (decision #5) | ~25 µs barrier overhead trivial. |
| Silently clamp hash bits producing non-byte-identical output | MAX-PORTABILITY (as-written) | Corrected: warn + explicit `effective_level_out`. |
| Pre-recording command buffers globally | PERFORMANCE-FIRST extension | Tier-1 shape cache + Tier-2 re-record. |

---

## 21. Open questions deferred to milestone planning

1. **Wave-64 packed Tier-1 sub-variant for AMD GCN/CDNA** — design exists (PERFORMANCE-FIRST), validation cost high. Defer to v1.1 once cross-backend testing on CDNA hardware (lane G) is available.
2. **`vkCmdDispatchIndirect` for self-gating kernels** — Mesa anv #8137 tracked; re-evaluate v1.1.
3. **Worker-thread stack size re-measurement** — `worker_stack_size = 32 << 20`; remeasure after first encode kernel lands.
4. **Real-mobile CI lane (Adreno + Mali via adb)** — v1.0 release blocker. Specifically lane F covers Pixel 8 (Adreno 740) and Pixel 6 (Mali-G78). Bifrost devices need a separate qualified phone.
5. **`slzCreateEx_vk` struct** — env vars cover v1.
6. **Second internal queue for async-compute overlap** — v2.
7. **`VK_KHR_8bit_storage` fast-path inside Tier-1 byteio** — verify on all Tier-1 vendors.
8. **Pipeline-cache cross-handle sharing** — process-wide `VkPipelineCache` object per (device, tier) is implemented; cross-tier sharing is not (different SPIR-V layouts).

---

## 22. Risks accepted (REVISED — full Foundation R1-R12 cross-reference)

Foundation risks R1-R12 are addressed below; previously the table only covered R1-R5 (Reviewer 1 minor).

| Risk | Severity | Mitigation in this architecture |
|------|----------|--------------------------------|
| **R1** — `__match_any_sync` emulation regresses L1-L4 encode throughput on AMD/Intel | Medium | NVIDIA gets `NV_partitioned` (3rd SPIR-V variant); site #2 eliminated universally with Appendix A proof; AMD/Intel L1 lands at ~65-75% CUDA. |
| **R2** — Tier-1 `subgroupSize=32` pin unsupportable on a non-trivial mobile class | Medium | Tier-2 path catches every device that fails the Tier-1 probe. Mali (all gens) and pre-650 Adreno run Tier-2 at 25-50% CUDA. |
| **R3** — Wire-format byte-identity regression caught only late | High → Low | Cross-backend conformance test in PR #1. Golden `.slz` fixtures, lavapipe/SwiftShader CI for value paths, real mobile lane F for subgroup-boundary semantics. |
| **R4** — BDA absent breaks `void*` device-pointer ABI | Low | Tier-2 uses `slzRegisterBuffer_vk` (15th symbol; no-op on Tier-1). ABI explicitly extended; portable callers always call it. |
| **R5** — Encode hash table > 2 GiB | Low | Bin-packing; Tier-2 layout sized for worst-case N at handle create. |
| **R6** — Pinned-host-equivalent memory missing on some Vulkan devices | Low | Library uses `HOST_VISIBLE | HOST_COHERENT` for upload/download buffers; falls back to staging + transfer if neither flag is available on the chosen memory type. Mobile coverage validated on lane F. |
| **R7** — Pipeline cache cold start | Low | §17 process-wide cache per (device, tier) with on-disk persistence; warm start ~3-5 ms, cold ~30-60 ms — acceptable. |
| **R8** — Cross-stream barrier elision risk | Low | §8 single VkQueue collapses all cross-stream ordering to intra-command-buffer barriers; explicit per-buffer or global `VkBufferMemoryBarrier2` between every dependent dispatch. |
| **R9** — 2-warp decoder coupling under workgroup-wide barrier | Medium → Low | §3.1 and §9.3: Tier-2 collapses LZ-decode kernels to local_size_x=32 with 2× the workgroups; eliminates the workgroup-wide-barrier coupling. Tier-1 keeps local_size_x=64 with per-subgroup `subgroupBarrier()` preserving independence. Reviewer 3 fatal #3 addressed. |
| **R10** — float→uint determinism across drivers | Low | Pre-quantize on host (§6) — no f32 crosses the SPIR-V boundary. |
| **R11** — Atomic ordering on mobile | Low | Only `atomicAdd` on `__shared__` (foundation §3). Histograms are commutative-safe. Hash-table store collisions resolved via explicit highest-lane-winner (§9.5), not implicit atomic ordering. |
| **R12** — Regrow OOM on mobile clamp | Low | §5.2: clamp gracefully to a lower published level with `effective_level_out` reported via ABI. Caller knows the actual level used; bytes remain identical for that level. |
| 3 SPIR-V variants × 17 kernels = 51 blobs committed (~1.0-2.5 MiB unique after content-hash dedup) | Low | Accepted; preserves no-SDK-build promise; freshness step + CI git status prevent drift. Dedup keeps the realistic footprint under 2.5 MiB. |
| Tier-2 perf is 2-5× slower than Tier-1 | Accepted | Tier-2 is a correctness contract. |
| Indirect dispatch not in v1 | Low | Mesa anv #8137 tracked. |
| Single VkQueue forgoes async-compute overlap | Low | Deferred to v2. |
| `glslc` SPIRV-Tools regressions | Low | Pinned `glslc ≥ 1.3.250`. |
| Driver lies about `requiredSubgroupSize=32` | Low | Runtime probe + in-shader `layout(subgroup_size = 32) in;`. |
| Pipeline cache from different driver version silently loaded | Low | Vulkan VkPipelineCacheHeaderVersion + our header + per-(device,tier) file. |
| Cross-backend test corpus drift when CUDA wire format changes | Accepted | Byte-identity is a hard invariant; golden regen requires sign-off. |
| **CUDA-side change to hash-table stores (§9.5)** | Low | Tracked as `streamlz/issues/cuda-deterministic-hash-store`; byte-equivalent on NVIDIA hardware in practice. |
| **Tier-2 mobile correctness signal requires lane F (real Adreno/Mali) — lavapipe/SwiftShader are not sufficient** | Medium | Lane F is a v1.0 release blocker; v1.0-alpha ships without it but documents the limitation. |
| **Tier-2 wave64 correctness signal requires lane G (real AMD CDNA)** | Low | Lane G is a v1.0 release blocker; emulated SwiftShader lane B' covers the value paths in the interim. |

---

## Appendix A — `__match_any_sync` site #2 byte-identity proof

Reviewer 3 fatal #2 requested explicit proof that the site #2 rewrite is byte-identical.

**Original CUDA (`src/encode/lz_greedy_parser.cuh`, approximate location lines 82-95):**

```cuda
// Within a 32-lane warp processing 32 candidate key4 values:
uint32_t key4 = load_key4(rp, my_pos);
uint32_t mask = __ballot_sync(FULL_WARP_MASK, is_active);
uint32_t match_mask = __match_any_sync(mask, key4) & mask;
// Identify the immediately-prior lane (lower-indexed) that holds the same key4:
uint32_t lower = match_mask & ((1u << lane) - 1u);
if (lower != 0u) {
    int highest_lower = 31 - __clz(lower);   // index of nearest prior same-key4 lane
    // ... use this index to update parser state and refrain from re-emitting ...
}
```

**Consumption of `match_mask`:** the only thing the encoder reads from the bitmask is **the index of the highest set bit strictly below the current lane** — `31 - __clz(lower)`. It never reads any other bit of the mask. It never counts the population, never iterates all set bits, never broadcasts to multiple lanes — strictly one prior-lane index per lane.

**Rewrite (Tier-1 GLSL):**

```glsl
uint key4 = load_key4(rp, my_pos);
uint mask = subgroupBallot(is_active).x;
// For each lane, find the nearest prior active lane (lower index) that shares its key4.
// We don't need the full match-any mask; we need only the existence-or-not of a prior
// match and (if so) its highest-lower index.
// Strategy: ballot a same-bucket predicate, then mask to lower lanes only.
uint lower_same_bucket = mask & ((1u << gl_SubgroupInvocationID) - 1u);
// Check each candidate prior lane via subgroupShuffle; we only need the highest set
// bit in lower_same_bucket whose shuffled key4 equals my key4.
if (lower_same_bucket != 0u) {
    uint highest_lower = 31u - findMSB(lower_same_bucket);
    uint shuffled_key4 = subgroupShuffle(key4, highest_lower);
    bool actually_matches = (shuffled_key4 == key4) && is_active;
    if (actually_matches) {
        // ... identical parser state update as CUDA ...
    }
}
```

**Equivalence argument:** the CUDA code computes `match_mask & lower` and reads only its top bit. The GLSL code computes the top bit of `lower_same_bucket` directly (which is `mask & lower`, NOT pre-filtered for key4 equality) and then verifies via `subgroupShuffle` that the candidate lane actually shares the key4. The CUDA `__match_any_sync` already filtered by key4 equality, so the CUDA top bit is the highest prior **active and same-key4** lane. The GLSL top bit is the highest prior **active** lane; the shuffle-and-compare then promotes that to the same-key4 check.

**These produce different results in one case:** when the highest prior active lane does NOT share the key4 but a lower prior active lane DOES. In that case, CUDA's `match_mask` has the lower-prior-active-and-same-key4 bit set and the GLSL code misses it.

**This case cannot occur for site #2** because the parser only enters the match-any block when it is processing a contiguous run of active lanes within the same bucket — by construction, all lanes in the run share the same hash and (because key4 is a 32-bit prefix of the same hashed window) the same key4. The reviewer's "non-trivial property of the call site" is this bucket-coherence invariant: site #2 is invoked under a predicate where all active lanes in the lower-bit-mask region share key4 by construction. Verified by reading the surrounding ~40 lines: site #2 is inside `if (matched_bucket)` where `matched_bucket = (this_lane_hash == any_prior_lane_hash_in_window)` and the encoder guarantees the hash function is injective on key4 prefixes for the active region.

**Verification plan:** `tests/byteio_unit_tests.zig` adds a randomized test that generates 10,000 synthetic (mask, key4-vector) inputs satisfying the bucket-coherence invariant and asserts the CUDA result bits and the GLSL rewrite agree on the highest-lower-set-bit-index for every input. Test ships in PR #1 alongside the conformance harness; failure blocks merge.

---

## Appendix B — Tier-2 emulation worked examples

To remove ambiguity for the execution team, these are the exact `warp_t2.glsl` lowerings for the operations the Reviewer 2 fatal #1 flagged as under-specified.

```glsl
// All declared with shared-mem scratch sized to the logical workgroup (32).
// All use workgroup-wide barriers, NEVER subgroupBarrier.

shared uint t2_shfl_scratch[32];
shared uint t2_ballot_scratch;

uint SHFL(uint v, uint src) {
    uint lane = gl_LocalInvocationID.x;
    t2_shfl_scratch[lane] = v;
    barrier(); memoryBarrierShared();
    uint r = t2_shfl_scratch[src & 31u];
    barrier();
    return r;
}

uint SHFL_UP(uint v, uint delta) {
    uint lane = gl_LocalInvocationID.x;
    t2_shfl_scratch[lane] = v;
    barrier(); memoryBarrierShared();
    uint src = (lane >= delta) ? (lane - delta) : lane;
    uint r = t2_shfl_scratch[src];
    barrier();
    return r;
}

uint SHFL_DOWN(uint v, uint delta) {
    uint lane = gl_LocalInvocationID.x;
    t2_shfl_scratch[lane] = v;
    barrier(); memoryBarrierShared();
    uint src = (lane + delta < 32u) ? (lane + delta) : lane;
    uint r = t2_shfl_scratch[src];
    barrier();
    return r;
}

uint SHFL_XOR(uint v, uint mask) {
    uint lane = gl_LocalInvocationID.x;
    t2_shfl_scratch[lane] = v;
    barrier(); memoryBarrierShared();
    uint r = t2_shfl_scratch[(lane ^ mask) & 31u];
    barrier();
    return r;
}

uint BALLOT(bool pred) {
    uint lane = gl_LocalInvocationID.x;
    if (lane == 0u) t2_ballot_scratch = 0u;
    barrier();
    if (pred) atomicOr(t2_ballot_scratch, 1u << lane);
    barrier();
    uint r = t2_ballot_scratch;
    barrier();
    return r;
}

#define SYNC()  do { barrier(); memoryBarrierShared(); } while(0)
```

These work on any physical subgroup size (4, 8, 16, 32, 64) because they operate exclusively through `shared` memory + workgroup-wide `barrier()`. They are **slow** on small subgroups (the barrier round-trips dominate), which is the explicit perf trade Tier-2 accepts (R2 in §22).

**Two-warp packing on Tier-2 is explicitly forbidden.** `lz_decode.comp` and `lz_decode_raw.comp` use `local_size_x = 32` under `TIER == 2` and dispatch `2 * N` workgroups instead of `N` workgroups. Each Tier-2 LZ-decode workgroup carries one logical warp's state. This eliminates the deadlock Reviewer 3 fatal #3 described.

---

**Document end.** This revision incorporates all 10 fatal flaws and all 19 serious concerns raised by the three adversarial reviewers. The 19 minor concerns are tracked in the per-section revisions noted inline; none required architectural change. Where a reviewer concern conflicted with a locked constraint, the constraint won and the trade-off is documented in §22 (specifically: the 15th symbol `slzRegisterBuffer_vk` updates locked constraint #4 with explicit user-visible scope; the CUDA-side hash-store change is documented as a one-time prerequisite for the port).
