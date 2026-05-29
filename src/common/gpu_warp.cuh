// ── StreamLZ GPU - warp / lane / bit-width constants ────────────
// Shared by every CUDA kernel in src/gpu/ (LZ + Huffman, decode +
// encode). Pure header: #pragma once, constants and one trivial
// helper, no kernels and no translation-unit state.
//
// #include'd via a relative path ("../common/gpu_warp.cuh") into the
// existing single .cu translation units - never compiled separately,
// so the one-.cu-per-.ptx build is unchanged.
#pragma once

#include <cstdint>

// ── Warp / lane geometry ────────────────────────────────────────
// One CUDA warp is 32 lanes. LANE_MASK extracts a lane index from a
// thread index (threadIdx.x & LANE_MASK).
static constexpr uint32_t WARP_SIZE      = 32;
static constexpr uint32_t LANE_MASK      = WARP_SIZE - 1;   // threadIdx.x & 31 = lane

// ── Warp-intrinsic participation mask ───────────────────────────
// FULL_WARP_MASK is the "all 32 lanes active" mask passed to warp-wide
// intrinsics (__shfl_sync / __ballot_sync / __match_any_sync). It is
// orthogonal to the geometry constants above - every existing kernel in
// src/gpu/ launches with a full warp and so only ever needs this one
// mask.
static constexpr uint32_t FULL_WARP_MASK = 0xFFFFFFFFu;     // all 32 lanes active

// ── Scalar bit widths ───────────────────────────────────────────
static constexpr uint32_t BITS_PER_BYTE = 8;
static constexpr uint32_t U32_BITS      = 32;
static constexpr uint32_t U64_BITS      = 64;

// ── Single-thread launch guard ──────────────────────────────────
// Several driver-orchestration kernels (walk_frame, prefix_sum_chunks,
// merge_huff_descs, compact_descs) launch one block of one thread and
// short-circuit any spuriously-launched threads at the top. The macro
// makes the intent self-documenting at the call site.
#define SLZ_GUARD_SINGLE_THREAD() do { if (blockIdx.x != 0 || threadIdx.x != 0) return; } while (0)

// ── Last-bit-set index ──────────────────────────────────────────
// Returns the 0-based position of the highest set bit in `x`. Undefined
// for x == 0 (CUDA's __clz returns 32 → result is -1; callers that
// allow zero must guard externally). Wraps the `(U32_BITS - 1) - __clz(x)`
// idiom used in several warp-reduction sites.
__device__ __forceinline__ int lastBitSet(uint32_t x) {
    return (int)(U32_BITS - 1) - __clz(x);
}

// ── Cooperative byte copy across the warp ───────────────────────
// All 32 lanes participate, each writing every WARP_SIZE-th byte.
// Caller must guarantee `dst` / `src` overlap is either disjoint or
// strictly forward (each lane reads index i then writes index i - no
// shifted-overlap support). `n` may be any positive count; lanes with
// i >= n drop out naturally. Caller is responsible for the surrounding
// `__syncwarp()` if the destination is read by other warp-cooperative
// code after the copy.
__device__ __forceinline__ void warpCopy(uint8_t* dst, const uint8_t* src,
                                          uint32_t n, int lane) {
    for (uint32_t i = (uint32_t)lane; i < n; i += WARP_SIZE) dst[i] = src[i];
}
