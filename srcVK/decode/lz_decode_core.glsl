// 1:1 port of src/decode/lz_decode_core.cuh.
// Shared warp/subgroup primitives used by raw + general decoders:
// warpScanU32, warpLiteralCopy, warpMatchCopy.
// See srcVK/PortInstructions.md.

#ifndef SRCVK_DECODE_LZ_DECODE_CORE_GLSL
#define SRCVK_DECODE_LZ_DECODE_CORE_GLSL

#include "slz_wire_format.glsl"

// CUDA reference: src/decode/lz_decode_core.cuh:12-22. Hillis-Steele warp
// exclusive scan. Returns (exclusive prefix, total) via out params.
//
// VK adaptation: CUDA's `__shfl_up_sync(FULL_WARP_MASK, x, d)` maps to
// GLSL's `subgroupShuffleUp(x, d)`. Final broadcast of lane-31's inclusive
// value uses `subgroupShuffle(x, 31)`. The control flow is bit-identical
// to the CUDA source.
void warpScanU32(uint v, int lane, out uint exclusive, out uint total) {
    uint inclusive = v;
    for (int d = 1; d < 32; d <<= 1) {
        uint n = subgroupShuffleUp(inclusive, uint(d));
        if (lane >= d) inclusive += n;
    }
    exclusive = inclusive - v;
    total = subgroupShuffle(inclusive, 31u);
}

// CUDA reference: src/decode/lz_decode_core.cuh:36-43. Warp-cooperative
// literal copy (no bounds check). Each of the 32 lanes writes every
// WARP_SIZE-th byte.
//
// VK adaptation: GLSL cannot pass SSBOs as function parameters. The CUDA
// signature
//   warpLiteralCopy(uint8_t* dst, uint dst_pos, const uint8_t* lit, uint lit_pos, uint lit_len, int lane)
// becomes a function-like macro taking the SSBO names + base offsets as
// plain identifiers, expanding to the same lane-strided loop.
#define warpLiteralCopy(dst_ssbo, dst_pos, lit_ssbo, lit_pos, lit_len, lane) \
    do {                                                                    \
        for (uint _wlc_i = uint(lane); _wlc_i < uint(lit_len); _wlc_i += WARP_SIZE) \
            (dst_ssbo)[uint(dst_pos) + _wlc_i] = (lit_ssbo)[uint(lit_pos) + _wlc_i]; \
    } while (false)

// CUDA reference: src/decode/lz_decode_core.cuh:45-56. Warp-cooperative
// match copy. A non-overlapping match (match_dist >= match_len) is
// distributed across all 32 lanes; an overlapping match runs sequentially
// on lane 0 because lane k would read dst[match_src+k] before lane k-1
// wrote it.
//
// VK adaptation: see warpLiteralCopy above for the macro pattern. The
// match source reads from `dst_ssbo` (same buffer the writes go into).
#define warpMatchCopy(dst_ssbo, dst_pos, match_src, match_len, match_dist, lane) \
    do {                                                                        \
        if (int(match_dist) >= int(match_len) && uint(match_len) >= MIN_PARALLEL_MATCH_LEN) { \
            for (uint _wmc_i = uint(lane); _wmc_i < uint(match_len); _wmc_i += WARP_SIZE) \
                (dst_ssbo)[uint(dst_pos) + _wmc_i] = (dst_ssbo)[uint(match_src) + _wmc_i]; \
        } else if (uint(lane) == 0u) {                                          \
            for (uint _wmc_j = 0u; _wmc_j < uint(match_len); _wmc_j += 1u)      \
                (dst_ssbo)[uint(dst_pos) + _wmc_j] = (dst_ssbo)[uint(match_src) + _wmc_j]; \
        }                                                                       \
    } while (false)

// CUDA reference: src/decode/lz_decode_core.cuh:65-73. Bounded literal
// copy used by the general decoder. VK adaptation mirrors warpLiteralCopy
// — macro form with SSBO + base offsets in scope.
#define warpLiteralCopyBounded(dst_ssbo, dst_pos, lit_ssbo, lit_pos, lit_len, dst_end_abs, lit_size, lane) \
    do {                                                                                                 \
        for (uint _wlcb_i = uint(lane); _wlcb_i < uint(lit_len); _wlcb_i += WARP_SIZE)                   \
            if (uint(dst_pos) + _wlcb_i < uint(dst_end_abs) && uint(lit_pos) + _wlcb_i < uint(lit_size))  \
                (dst_ssbo)[uint(dst_pos) + _wlcb_i] = (lit_ssbo)[uint(lit_pos) + _wlcb_i];               \
    } while (false)

// CUDA reference: src/decode/lz_decode_core.cuh:75-89. Bounded match
// copy used by the general decoder.
#define warpMatchCopyBounded(dst_ssbo, dst_pos, match_src, match_len, match_dist, dst_end_abs, lane) \
    do {                                                                                            \
        if (int(match_dist) >= int(match_len) && uint(match_len) >= MIN_PARALLEL_MATCH_LEN) {       \
            for (uint _wmcb_i = uint(lane); _wmcb_i < uint(match_len); _wmcb_i += WARP_SIZE)        \
                if (uint(dst_pos) + _wmcb_i < uint(dst_end_abs))                                    \
                    (dst_ssbo)[uint(dst_pos) + _wmcb_i] = (dst_ssbo)[uint(match_src) + _wmcb_i];    \
        } else if (uint(lane) == 0u) {                                                              \
            for (uint _wmcb_j = 0u; _wmcb_j < uint(match_len); _wmcb_j += 1u)                       \
                if (uint(dst_pos) + _wmcb_j < uint(dst_end_abs))                                    \
                    (dst_ssbo)[uint(dst_pos) + _wmcb_j] = (dst_ssbo)[uint(match_src) + _wmcb_j];    \
        }                                                                                           \
    } while (false)

#endif
