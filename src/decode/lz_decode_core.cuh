// ── StreamLZ LZ decode - shared warp primitives ────────────────
// Hillis-Steele warp prefix scan + warp-cooperative copy helpers used
// by both LZ decoders (lz_decode_raw.cuh + lz_decode_general.cuh).
// Each decoder #includes this header for the helpers it needs.
//
// Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"

// Hillis-Steele warp exclusive scan, returns (lane's exclusive prefix, total).
__device__ __forceinline__ void warpScanU32(uint32_t v, uint32_t& exclusive, uint32_t& total) {
    uint32_t inclusive = v;
    const int lane = threadIdx.x & LANE_MASK;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t n = __shfl_up_sync(FULL_WARP_MASK, inclusive, d);
        if (lane >= d) inclusive += n;
    }
    exclusive = inclusive - v;
    total = __shfl_sync(FULL_WARP_MASK, inclusive, 31);
}

// ── Warp-cooperative copy primitives (raw-mode, no bounds check) ──
// The raw-mode decoder writes inside a known-safe dst_size window so per-
// store bounds checks are unnecessary. The general decoder has its own
// inlined copies with dst_end_abs guards.
//
// Match copy: a non-overlapping match (match_dist >= match_len) is
// distributed across all 32 lanes; an overlapping match must run
// sequentially on lane 0 because lane k would read dst[match_src+k]
// before lane k-1 wrote it.
//
// Both are __forceinline__ - the prior inline bodies they replace were
// in the decode hot loop, and nvcc lays them out identically.
__device__ __forceinline__ void warpLiteralCopy(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    const uint8_t* __restrict__ lit, uint32_t lit_pos,
    uint32_t lit_len, int lane
) {
    for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
        dst[dst_pos + i] = lit[lit_pos + i];
}

__device__ __forceinline__ void warpMatchCopy(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    uint32_t match_src, uint32_t match_len, int32_t match_dist, int lane
) {
    if (match_dist >= (int32_t)match_len && match_len >= MIN_PARALLEL_MATCH_LEN) {
        for (uint32_t i = lane; i < match_len; i += WARP_SIZE)
            dst[dst_pos + i] = dst[match_src + i];
    } else if (lane == 0) {
        for (uint32_t i = 0; i < match_len; i++)
            dst[dst_pos + i] = dst[match_src + i];
    }
}

// ── v4 #16: dictionary back-reference read ────────────────────────
// Read one match-source byte that may lie BELOW the sub-chunk's output
// window: with a preset dictionary, the negative space immediately
// before `window_base` maps onto the dictionary's tail (dict byte
// `dict_len - k` sits k bytes below the window). The u32 wrap
// arithmetic makes the below-window test correct even when the
// arithmetic source address wrapped negative (chunk 0 windows):
// `window_base - addr` recovers the true distance for any |distance|
// < 2^31. Reads reaching below the dictionary itself (hostile frames)
// clamp to 0x00 - same byte every lane, no out-of-bounds access.
__device__ __forceinline__ uint8_t readBackRefByte(
    const uint8_t* __restrict__ dst, uint32_t addr,
    uint32_t window_base,
    const uint8_t* __restrict__ dict, uint32_t dict_len
) {
    uint32_t below = window_base - addr; // >0 and <2^31  ⟺  addr < base
    if (below != 0 && below < 0x80000000u)
        return (below <= dict_len) ? dict[dict_len - below] : (uint8_t)0;
    return dst[addr];
}

// Match copy with dictionary reach. HAS_DICT=false compiles to exactly
// warpMatchCopy (the per-byte select folds away), so call sites pay
// nothing on dictionary-less frames. A match whose source STRADDLES
// the window base (starts in the dictionary, runs into the window's
// own output) is handled per byte; self-overlap semantics are
// unchanged because the dist/len relation is independent of where the
// source bytes live.
template <bool HAS_DICT>
__device__ __forceinline__ void warpMatchCopyD(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    uint32_t match_src, uint32_t match_len, int32_t match_dist, int lane,
    uint32_t window_base, const uint8_t* __restrict__ dict, uint32_t dict_len
) {
    if constexpr (!HAS_DICT) {
        warpMatchCopy(dst, dst_pos, match_src, match_len, match_dist, lane);
        return;
    }
    if (match_dist >= (int32_t)match_len && match_len >= MIN_PARALLEL_MATCH_LEN) {
        for (uint32_t i = lane; i < match_len; i += WARP_SIZE)
            dst[dst_pos + i] = readBackRefByte(dst, match_src + i, window_base, dict, dict_len);
    } else if (lane == 0) {
        for (uint32_t i = 0; i < match_len; i++)
            dst[dst_pos + i] = readBackRefByte(dst, match_src + i, window_base, dict, dict_len);
    }
}

// ── Bounded warp-cooperative copy primitives ──────────────────────
// The general decoder writes into a window where the last store may
// land past the legal dst_end_abs (encoder may overcommit dst_size
// when a sub-chunk's effective decode size shrinks). Each store is
// guarded by `dst_pos + i < dst_end_abs`. Literal copies additionally
// guard against running past lit_size when the encoder wrote a
// shorter literal stream than the token sequence implies.
__device__ __forceinline__ void warpLiteralCopyBounded(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    const uint8_t* __restrict__ lit, uint32_t lit_pos,
    uint32_t lit_len, uint32_t dst_end_abs, uint32_t lit_size, int lane
) {
    for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
        if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
            dst[dst_pos + i] = lit[lit_pos + i];
}

__device__ __forceinline__ void warpMatchCopyBounded(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    uint32_t match_src, uint32_t match_len, int32_t match_dist,
    uint32_t dst_end_abs, int lane
) {
    if (match_dist >= (int32_t)match_len && match_len >= MIN_PARALLEL_MATCH_LEN) {
        for (uint32_t i = lane; i < match_len; i += WARP_SIZE)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = dst[match_src + i];
    } else if (lane == 0) {
        for (uint32_t i = 0; i < match_len; i++)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = dst[match_src + i];
    }
}

// v4 #16: bounded match copy with dictionary reach (the general
// decoder's counterpart of warpMatchCopyD; same zero-cost guarantee
// at HAS_DICT=false).
template <bool HAS_DICT>
__device__ __forceinline__ void warpMatchCopyBoundedD(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    uint32_t match_src, uint32_t match_len, int32_t match_dist,
    uint32_t dst_end_abs, int lane,
    uint32_t window_base, const uint8_t* __restrict__ dict, uint32_t dict_len
) {
    if constexpr (!HAS_DICT) {
        warpMatchCopyBounded(dst, dst_pos, match_src, match_len, match_dist, dst_end_abs, lane);
        return;
    }
    if (match_dist >= (int32_t)match_len && match_len >= MIN_PARALLEL_MATCH_LEN) {
        for (uint32_t i = lane; i < match_len; i += WARP_SIZE)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = readBackRefByte(dst, match_src + i, window_base, dict, dict_len);
    } else if (lane == 0) {
        for (uint32_t i = 0; i < match_len; i++)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = readBackRefByte(dst, match_src + i, window_base, dict, dict_len);
    }
}
