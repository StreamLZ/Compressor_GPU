// 1:1 port of src/common/gpu_warp.cuh.
// Warp/subgroup constants and helpers (WARP_SIZE, LANE_MASK,
// FULL_WARP_MASK). CUDA's `__shfl_sync` / `__ballot_sync` /
// `__match_any_sync` map onto Vulkan's GL_KHR_shader_subgroup_*
// intrinsics; CUDA's WARP_SIZE=32 maps onto `gl_SubgroupSize`. Per
// audit Section H Q1, this port pins the contract to subgroupSize==32
// (matching the CUDA WARP_SIZE) — the codec dispatch verifies the
// device's subgroupSize before launch and rejects mismatched devices.
// All shaders that include this header require:
//
//   #extension GL_KHR_shader_subgroup_basic     : require
//   #extension GL_KHR_shader_subgroup_ballot    : require
//   #extension GL_KHR_shader_subgroup_shuffle   : require
//   #extension GL_KHR_shader_subgroup_vote      : require
//
// See srcVK/PortInstructions.md.

#ifndef SRCVK_COMMON_GPU_WARP_GLSL
#define SRCVK_COMMON_GPU_WARP_GLSL

// CUDA reference: src/common/gpu_warp.cuh:16-17. Warp / lane geometry.
// One CUDA warp is 32 lanes; we pin the Vulkan subgroup width to the
// same value so the kernel logic is bit-identical to CUDA.
const uint WARP_SIZE      = 32u;
const uint LANE_MASK      = WARP_SIZE - 1u; // threadIdx.x & 31 = lane

// CUDA reference: src/common/gpu_warp.cuh:25. The "all 32 lanes active"
// mask CUDA passes to `__shfl_sync` / `__ballot_sync`. Vulkan's
// `subgroup*` ops do not take a participation mask — every active
// invocation in the subgroup participates — but the constant is kept
// for parity with the CUDA source and is referenced by the audit
// metadata for the bitfield-mask kernels.
const uint FULL_WARP_MASK = 0xFFFFFFFFu;

// CUDA reference: src/common/gpu_warp.cuh:28-30. Scalar bit widths.
const uint BITS_PER_BYTE = 8u;
const uint U32_BITS      = 32u;
const uint U64_BITS      = 64u;

// CUDA reference: src/common/gpu_warp.cuh:37. Single-thread launch
// guard — several driver-orchestration kernels (walk_frame,
// prefix_sum_chunks, merge_huff_descs, compact_descs) launch one
// workgroup of one thread and short-circuit any spuriously-launched
// threads at the top. CUDA's macro reads blockIdx.x / threadIdx.x; the
// GLSL spelling uses the gl_* equivalents.
#define SLZ_GUARD_SINGLE_THREAD() do { if (gl_WorkGroupID.x != 0u || gl_LocalInvocationID.x != 0u) return; } while (false)

// CUDA reference: src/common/gpu_warp.cuh:44-46. 0-based position of the
// highest set bit in `x`. Undefined for x == 0 (CUDA's __clz returns 32
// → result is -1; callers that allow zero must guard externally).
// GLSL's `findMSB` returns -1 for zero, matching the CUDA semantics.
int lastBitSet(uint x) {
    return findMSB(x);
}

// CUDA reference: src/common/gpu_warp.cuh:56-59. Cooperative byte copy
// across the warp — all 32 lanes participate, each writing every
// WARP_SIZE-th byte. Caller must guarantee `dst` / `src` overlap is
// either disjoint or strictly forward (each lane reads index i then
// writes index i — no shifted-overlap support). `n` may be any positive
// count; lanes with i >= n drop out naturally. Caller is responsible for
// the surrounding `subgroupBarrier()` if the destination is read by
// other subgroup-cooperative code after the copy.
//
// Signature adaptation: CUDA's signature
//   void warpCopy(uint8_t* dst, const uint8_t* src, uint32_t n, int lane)
// cannot be expressed as a GLSL function because SSBOs (the GLSL stand-in
// for raw byte pointers) cannot be passed as function parameters in
// standard GLSL. The port spells `warpCopy` as a function-like macro
// taking the SSBO names + base offsets as plain identifiers, expanding
// to the same lane-strided loop CUDA's body runs. Call sites read
// identically to CUDA's:
//   warpCopy(dst_ssbo, dst_base, src_ssbo, src_base, n, lane);
#define warpCopy(dst_ssbo, dst_base, src_ssbo, src_base, n, lane)        \
    do {                                                                  \
        for (uint _wc_i = uint(lane); _wc_i < uint(n); _wc_i += WARP_SIZE) \
            dst_ssbo[uint(dst_base) + _wc_i] = src_ssbo[uint(src_base) + _wc_i]; \
    } while (false)

#endif
