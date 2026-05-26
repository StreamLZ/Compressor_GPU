// ── StreamLZ GPU — warp / lane / bit-width constants ────────────
// Shared by every CUDA kernel in src/gpu/ (LZ + Huffman, decode +
// encode). Pure header: #pragma once, constants and one trivial
// helper, no kernels and no translation-unit state.
//
// #include'd via a relative path ("../common/gpu_warp.cuh") into the
// existing single .cu translation units — never compiled separately,
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
// orthogonal to the geometry constants above — every existing kernel in
// src/gpu/ launches with a full warp and so only ever needs this one
// mask.
static constexpr uint32_t FULL_WARP_MASK = 0xFFFFFFFFu;     // all 32 lanes active

// ── Scalar bit widths ───────────────────────────────────────────
static constexpr uint32_t BITS_PER_BYTE = 8;
static constexpr uint32_t U32_BITS      = 32;
static constexpr uint32_t U64_BITS      = 64;
