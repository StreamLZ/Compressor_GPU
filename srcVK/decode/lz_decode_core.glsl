// 1:1 port of src/decode/lz_decode_core.cuh.
// Shared warp/subgroup primitives used by raw + general decoders:
// warpScanU32, warpLiteralCopy, warpMatchCopy.

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

// CUDA reference: src/decode/lz_decode_core.cuh:68-77 (readBackRefByte,
// v4 #16). Read one match-source byte that may lie BELOW the sub-chunk's
// output window: with a preset dictionary, the negative space immediately
// before `window_base` maps onto the dictionary's tail (dict byte
// `dict_len - k` sits k bytes below the window). The u32 wrap arithmetic
// makes the below-window test correct even when the arithmetic source
// address wrapped negative (chunk 0 windows). Reads reaching below the
// dictionary itself (hostile frames) clamp to 0x00.
//
// VK adaptation: out-param statement macro (CUDA returns the byte; GLSL
// macros cannot return, and the two-SSBO read rules out a function) —
// the same shape as _SLZ_OFF16_READ_FALSE/_TRUE in lz_decode_raw.glsl.
#define _SLZ_READ_BACKREF(out_b, dst_ssbo, addr, window_base, dict_ssbo, dict_len) \
    do {                                                                           \
        uint _rb_below = uint(window_base) - uint(addr);                           \
        if (_rb_below != 0u && _rb_below < 0x80000000u) {                          \
            (out_b) = (_rb_below <= uint(dict_len))                                \
                ? (dict_ssbo)[uint(dict_len) - _rb_below] : uint8_t(0u);           \
        } else {                                                                   \
            (out_b) = (dst_ssbo)[uint(addr)];                                      \
        }                                                                          \
    } while (false)

// CUDA reference: src/decode/lz_decode_core.cuh:86-103 (warpMatchCopyD,
// v4 #16). Match copy with dictionary reach — the HAS_DICT=true template
// instantiation. The HAS_DICT=false instantiation IS warpMatchCopy above
// (CUDA compiles it to exactly that), so dictionary-less call sites keep
// using warpMatchCopy and pay nothing. A match whose source STRADDLES the
// window base is handled per byte; self-overlap semantics are unchanged
// because the dist/len relation is independent of where the source bytes
// live.
#define warpMatchCopyD(dst_ssbo, dst_pos, match_src, match_len, match_dist, lane, \
                       window_base, dict_ssbo, dict_len)                          \
    do {                                                                          \
        if (int(match_dist) >= int(match_len) && uint(match_len) >= MIN_PARALLEL_MATCH_LEN) { \
            for (uint _wmd_i = uint(lane); _wmd_i < uint(match_len); _wmd_i += WARP_SIZE) { \
                uint8_t _wmd_b;                                                   \
                _SLZ_READ_BACKREF(_wmd_b, dst_ssbo, uint(match_src) + _wmd_i,     \
                                  window_base, dict_ssbo, dict_len);              \
                (dst_ssbo)[uint(dst_pos) + _wmd_i] = _wmd_b;                      \
            }                                                                     \
        } else if (uint(lane) == 0u) {                                           \
            for (uint _wmd_j = 0u; _wmd_j < uint(match_len); _wmd_j += 1u) {      \
                uint8_t _wmd_b2;                                                  \
                _SLZ_READ_BACKREF(_wmd_b2, dst_ssbo, uint(match_src) + _wmd_j,    \
                                  window_base, dict_ssbo, dict_len);              \
                (dst_ssbo)[uint(dst_pos) + _wmd_j] = _wmd_b2;                     \
            }                                                                     \
        }                                                                         \
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

// CUDA reference: src/decode/lz_decode_core.cuh:141-161 (warpMatchCopyBoundedD,
// v4 #16). Bounded match copy with dictionary reach — the general decoder's
// counterpart of warpMatchCopyD; the HAS_DICT=false instantiation IS
// warpMatchCopyBounded above, so dictionary-less call sites keep using it.
#define warpMatchCopyBoundedD(dst_ssbo, dst_pos, match_src, match_len, match_dist, \
                              dst_end_abs, lane, window_base, dict_ssbo, dict_len) \
    do {                                                                           \
        if (int(match_dist) >= int(match_len) && uint(match_len) >= MIN_PARALLEL_MATCH_LEN) { \
            for (uint _wmbd_i = uint(lane); _wmbd_i < uint(match_len); _wmbd_i += WARP_SIZE) { \
                if (uint(dst_pos) + _wmbd_i < uint(dst_end_abs)) {                 \
                    uint8_t _wmbd_b;                                               \
                    _SLZ_READ_BACKREF(_wmbd_b, dst_ssbo, uint(match_src) + _wmbd_i, \
                                      window_base, dict_ssbo, dict_len);           \
                    (dst_ssbo)[uint(dst_pos) + _wmbd_i] = _wmbd_b;                 \
                }                                                                  \
            }                                                                      \
        } else if (uint(lane) == 0u) {                                            \
            for (uint _wmbd_j = 0u; _wmbd_j < uint(match_len); _wmbd_j += 1u) {    \
                if (uint(dst_pos) + _wmbd_j < uint(dst_end_abs)) {                 \
                    uint8_t _wmbd_b2;                                              \
                    _SLZ_READ_BACKREF(_wmbd_b2, dst_ssbo, uint(match_src) + _wmbd_j, \
                                      window_base, dict_ssbo, dict_len);           \
                    (dst_ssbo)[uint(dst_pos) + _wmbd_j] = _wmbd_b2;                \
                }                                                                  \
            }                                                                      \
        }                                                                          \
    } while (false)

#endif
