// 1:1 port of src/decode/lz_decode_general.cuh.
//
// General entropy-capable LZ decoder (L2+). Subgroup-cooperative variant
// of the CUDA warp-cooperative path: handles off32, delta literals, and
// multi-block sub-chunks (up to MAX_BLOCKS_PER_SUBCHUNK 64 KB LZ blocks
// each). Mode bits = (sub-chunk-header >> 19) & 0xF: only mode == 0
// (delta literals) vs mode != 0 (raw literals) is distinguished here.

#ifndef SRCVK_DECODE_LZ_DECODE_GENERAL_GLSL
#define SRCVK_DECODE_LZ_DECODE_GENERAL_GLSL

#include "lz_decode_core.glsl"

// CUDA reference: src/decode/lz_decode_general.cuh:33-39 (TokenType enum).
// Token-type tags assigned by the lane-0 parser and consumed after the
// subgroup broadcast. Only TOKEN_TYPE_LONG_LITERAL is tested by name
// (its recent_offset must be preserved); the others differ only in which
// offset stream they pull from, kept named for debuggability.
const uint TOKEN_TYPE_SHORT        = 0u;
const uint TOKEN_TYPE_LONG_LITERAL = 1u;
const uint TOKEN_TYPE_LONG_NEAR    = 2u;
const uint TOKEN_TYPE_SHORT_FAR    = 3u;
const uint TOKEN_TYPE_LONG_FAR     = 4u;

// CUDA reference: src/decode/lz_decode_general.cuh:21-25 (DecodeOutput).
// Per-call destination region for decodeSubChunkGeneral. The decoder
// only writes inside `dst[dst_offset .. dst_offset + dst_size)`.
//
// VK adaptation: the CUDA pointer field `dst` becomes an SSBO + base
// offset pair at the call site. The struct here keeps the dst_size +
// dst_offset scalars; dst_ssbo is passed as a separate macro argument.
struct DecodeOutput {
    uint dst_size;
    uint dst_offset;
};

// CUDA reference: src/decode/lz_decode_general.cuh:46-57
// (deltaLiteralCopyBounded). Delta-literal mode (sub-chunk mode == 0):
// each output byte = literal byte + the byte at dst[i + recent_offset].
// Used in the inner token loop AND the per-block / final trailing-
// literal flushes — same pattern at three sites.
//
// VK adaptation: macro form (GLSL cannot pass SSBOs as function args).
// CUDA's `__forceinline__` is N/A here — glslc inlines all macros by
// definition. The lit byte is read from either `comp_ssbo` (raw literal
// stream in compressed blob) or `lit_scratch_ssbo` (Huffman-decoded
// literals in entropy scratch), selected by lit_in_scratch — the same
// flag the CUDA pointer trick collapses transparently.
// `lit_pos` is the ABSOLUTE byte offset into the source SSBO; `lit_end_abs` is the
// ABSOLUTE end of the literal stream. See note on warpLiteralCopyBoundedSel.
#define deltaLiteralCopyBounded(dst_ssbo, dst_pos,                                    \
                                comp_ssbo, lit_scratch_ssbo, lit_in_scratch, lit_pos, \
                                count, dst_end_abs, lit_end_abs,                     \
                                recent_offset, lane)                                  \
    do {                                                                              \
        for (uint _dlc_i = uint(lane); _dlc_i < uint(count); _dlc_i += WARP_SIZE) {   \
            if (uint(dst_pos) + _dlc_i < uint(dst_end_abs) &&                         \
                uint(lit_pos) + _dlc_i < uint(lit_end_abs)) {                         \
                uint _dlc_match_src = uint(int(uint(dst_pos) + _dlc_i) +              \
                                            int(recent_offset));                       \
                uint _dlc_litb = (uint(lit_in_scratch) != 0u)                          \
                    ? uint((lit_scratch_ssbo)[uint(lit_pos) + _dlc_i])                \
                    : uint((comp_ssbo)[uint(lit_pos) + _dlc_i]);                      \
                uint _dlc_refb = uint((dst_ssbo)[_dlc_match_src]);                    \
                (dst_ssbo)[uint(dst_pos) + _dlc_i] = uint8_t((_dlc_litb + _dlc_refb) & 0xFFu); \
            }                                                                          \
        }                                                                              \
    } while (false)

// VK adaptation: literal-source-aware bounded copy. The CUDA decoder
// calls warpLiteralCopyBounded with the resolved `lit` pointer (either
// inside the compressed blob or inside entropy scratch); in GLSL the
// SSBO can't be selected by a runtime value, so this wrapper does the
// per-byte branch.
// `lit_pos` is the ABSOLUTE byte offset into the source SSBO; `lit_end_abs` is the
// ABSOLUTE end of the literal stream within that same SSBO. The CUDA original uses
// a relative offset + relative size and compares them, so VK must convert one side
// to match — we pass both as absolute here to mirror the CUDA semantics exactly.
#define warpLiteralCopyBoundedSel(dst_ssbo, dst_pos,                                  \
                                   comp_ssbo, lit_scratch_ssbo, lit_in_scratch,      \
                                   lit_pos, lit_len, dst_end_abs, lit_end_abs, lane) \
    do {                                                                              \
        for (uint _wls_i = uint(lane); _wls_i < uint(lit_len); _wls_i += WARP_SIZE) { \
            if (uint(dst_pos) + _wls_i < uint(dst_end_abs) &&                         \
                uint(lit_pos) + _wls_i < uint(lit_end_abs)) {                         \
                uint _wls_b = (uint(lit_in_scratch) != 0u)                            \
                    ? uint((lit_scratch_ssbo)[uint(lit_pos) + _wls_i])                \
                    : uint((comp_ssbo)[uint(lit_pos) + _wls_i]);                      \
                (dst_ssbo)[uint(dst_pos) + _wls_i] = uint8_t(_wls_b);                 \
            }                                                                          \
        }                                                                              \
    } while (false)

// VK adaptation: token-source-aware byte read. `cmd_in_scratch` picks
// either `comp_ssbo` (raw token stream) or `cmd_scratch_ssbo` (Huffman-
// decoded tokens). Used for the lane-0 token reads in the inner loop.
#define readCmdByteSel(out_b, comp_ssbo, cmd_scratch_ssbo, cmd_in_scratch, cmd_off)  \
    do {                                                                              \
        if (uint(cmd_in_scratch) != 0u) {                                             \
            (out_b) = uint((cmd_scratch_ssbo)[uint(cmd_off)]);                         \
        } else {                                                                       \
            (out_b) = uint((comp_ssbo)[uint(cmd_off)]);                                \
        }                                                                              \
    } while (false)

// CUDA reference: src/decode/lz_decode_general.cuh:77-309
// (decodeSubChunkGeneral<OFF16_SPLIT> template body). Two macro
// instantiations mirror the CUDA template specializations: _false
// reads off16 from the interleaved raw stream (off16_raw); _true reads
// from the split hi/lo bytes (off16_hi/off16_lo).
//
// VK adaptation: GLSL has no templates. The CUDA `if constexpr
// (OFF16_SPLIT)` branch becomes two macro forms with the off16 read
// shape hard-coded. The macro body is shared via _SLZ_DECODE_GENERAL_BODY,
// parameterized on `off16_read_macro` (one of _SLZ_OFF16_READ_GEN_FALSE
// or _SLZ_OFF16_READ_GEN_TRUE).
//
// Args (all u32 unless noted):
//   comp_ssbo                - SSBO holding lit / cmd / off16_raw /
//                              off32_raw1 / off32_raw2 / len_stream
//                              (one SSBO; per-stream byte offsets below)
//   ps_lit_ptr, ps_lit_size  - ParsedStreams literal stream
//   ps_cmd_ptr, ps_cmd_size  - ParsedStreams command stream
//   ps_off16_raw, ps_off16_count - interleaved off16 (used by _false)
//   off16_hi_ssbo, ps_off16_hi   - split-hi off16 (used by _true)
//   off16_lo_ssbo, ps_off16_lo   - split-lo off16 (used by _true)
//   ps_off32_raw1, ps_off32_count1 - off32 triples for LZ block 0
//   ps_off32_raw2, ps_off32_count2 - off32 triples for LZ block 1
//   ps_len_stream, ps_len_avail    - extended-length side stream
//   ps_off16_split                 - unused (templated-out in CUDA;
//                                    only here for ABI parity)
//   ps_cmd_stream2_offset          - byte offset in cmd stream where
//                                    LZ block 1 begins (0 if 1-block sub-chunk)
//   ps_initial_copy                - 8 if the first 8 bytes of dst were
//                                    raw-copied by the header parser
//   dst_ssbo                       - destination SSBO
//   out_dst_size, out_dst_offset   - DecodeOutput fields
//   mode                           - sub-chunk mode bits; mode == 0
//                                    triggers delta-literal copy
//   lane                           - subgroup lane id (gl_SubgroupInvocationID & 31)

// CUDA reference: src/decode/lz_decode_general.cuh:152-159 / 170-176
// (the two `if constexpr (OFF16_SPLIT)` branches inside the lane-0
// off16 reads). Both branches read ONE off16 entry into a u16; the
// caller treats it as `-(int32_t)v` for the match offset.
//
// Args: out_v is the destination u32 (will hold a u16-range value);
// off16_*_ssbo + base are the source SSBOs + base byte offsets; entry_idx
// is the off16 entry index.
#define _SLZ_OFF16_READ_GEN_FALSE(out_v, comp_ssbo, off16_raw_base, entry_idx,       \
                                   off16_hi_ssbo, off16_hi_base,                     \
                                   off16_lo_ssbo, off16_lo_base)                     \
    do {                                                                              \
        uint _o16g_lo = uint((comp_ssbo)[uint(off16_raw_base) +                       \
                                          (entry_idx) * OFF16_ENTRY_BYTES + 0u]);     \
        uint _o16g_hi = uint((comp_ssbo)[uint(off16_raw_base) +                       \
                                          (entry_idx) * OFF16_ENTRY_BYTES + 1u]);     \
        (out_v) = readU16LE(_o16g_lo, _o16g_hi);                                       \
    } while (false)

#define _SLZ_OFF16_READ_GEN_TRUE(out_v, comp_ssbo, off16_raw_base, entry_idx,        \
                                  off16_hi_ssbo, off16_hi_base,                      \
                                  off16_lo_ssbo, off16_lo_base)                      \
    do {                                                                              \
        uint _o16g_lo = uint((off16_lo_ssbo)[uint(off16_lo_base) + (entry_idx)]);    \
        uint _o16g_hi = uint((off16_hi_ssbo)[uint(off16_hi_base) + (entry_idx)]);    \
        (out_v) = _o16g_lo | (_o16g_hi << 8);                                          \
    } while (false)

// CUDA reference: src/decode/lz_decode_general.cuh:77-309. Shared body
// for the two OFF16_SPLIT specializations.
#define _SLZ_DECODE_GENERAL_BODY(comp_ssbo, ps_lit_ptr, ps_lit_size,                 \
                                  ps_cmd_ptr, ps_cmd_size,                           \
                                  ps_off16_raw, ps_off16_count,                      \
                                  off16_hi_ssbo, ps_off16_hi,                        \
                                  off16_lo_ssbo, ps_off16_lo,                        \
                                  ps_off32_raw1, ps_off32_count1,                    \
                                  ps_off32_raw2, ps_off32_count2,                    \
                                  ps_len_stream, ps_len_avail,                       \
                                  ps_off16_split,                                    \
                                  ps_cmd_stream2_offset, ps_initial_copy,            \
                                  lit_scratch_ssbo, ps_lit_in_scratch,                \
                                  cmd_scratch_ssbo, ps_cmd_in_scratch,                \
                                  dst_ssbo, out_dst_size, out_dst_offset,            \
                                  mode, lane, off16_read_macro)                       \
    do {                                                                              \
        uint _dg_cmd_pos = 0u, _dg_lit_pos = 0u;                                      \
        uint _dg_off16_pos = 0u, _dg_off32_pos = 0u;                                  \
        uint _dg_dst_pos = uint(out_dst_offset) + uint(ps_initial_copy);              \
        int  _dg_recent_offset = INITIAL_RECENT_OFFSET;                               \
        /* CUDA reference: src/decode/lz_decode_general.cuh:119-121. */               \
        /* Clamp dst_end_abs to handle the (corrupt) case where */                    \
        /* dst_offset + dst_size would overflow u32. */                               \
        uint _dg_dst_end_abs = (uint(out_dst_size) > 0xFFFFFFFFu - uint(out_dst_offset)) \
            ? 0xFFFFFFFFu                                                              \
            : (uint(out_dst_offset) + uint(out_dst_size));                            \
        uint _dg_length_offset = 0u;                                                   \
        uint _dg_off32_block_base = uint(ps_off32_raw1);                              \
        uint _dg_off32_block_count = uint(ps_off32_count1);                           \
        uint _dg_block_dst_start = uint(out_dst_offset);                              \
                                                                                       \
        uint _dg_block_cmd_end =                                                      \
            (uint(ps_cmd_stream2_offset) > 0u &&                                       \
             uint(ps_cmd_stream2_offset) < uint(ps_cmd_size))                          \
                ? uint(ps_cmd_stream2_offset)                                          \
                : uint(ps_cmd_size);                                                   \
                                                                                       \
        /* CUDA reference: src/decode/lz_decode_general.cuh:133-134. */               \
        /* Prefetch first token. Hoisted outside the for-loop (both */                \
        /* block_iter iterations need the carried value across the boundary). */      \
        uint _dg_prefetched_token = 0u;                                               \
        if (uint(lane) == 0u && _dg_cmd_pos < _dg_block_cmd_end) {                    \
            readCmdByteSel(_dg_prefetched_token, comp_ssbo,                            \
                            cmd_scratch_ssbo, ps_cmd_in_scratch,                       \
                            uint(ps_cmd_ptr) + _dg_cmd_pos);                          \
        }                                                                              \
                                                                                       \
        /* CUDA reference: src/decode/lz_decode_general.cuh:136-289. */               \
        /* Outer block loop: up to MAX_BLOCKS_PER_SUBCHUNK (=2) 64 KB blocks. */      \
        for (uint _dg_block_iter = 0u;                                                \
             _dg_block_iter < MAX_BLOCKS_PER_SUBCHUNK;                                 \
             _dg_block_iter = _dg_block_iter + 1u) {                                   \
            while (_dg_cmd_pos < _dg_block_cmd_end) {                                 \
                uint _dg_token = 0u, _dg_lit_len = 0u, _dg_match_len = 0u;            \
                int  _dg_match_offset = _dg_recent_offset;                            \
                uint _dg_use_recent = 0u;                                             \
                uint _dg_token_type = TOKEN_TYPE_SHORT;                               \
                                                                                       \
                /* CUDA reference: src/decode/lz_decode_general.cuh:142-204. */       \
                /* Token parse (lane 0 only). 5 dispatch paths: SHORT, */             \
                /* LONG_LITERAL, LONG_NEAR, LONG_FAR, and SHORT_FAR (fallthrough). */ \
                if (uint(lane) == 0u) {                                               \
                    _dg_token = _dg_prefetched_token;                                 \
                    _dg_cmd_pos = _dg_cmd_pos + 1u;                                   \
                    if (_dg_cmd_pos < _dg_block_cmd_end) {                            \
                        readCmdByteSel(_dg_prefetched_token, comp_ssbo,                \
                                        cmd_scratch_ssbo, ps_cmd_in_scratch,           \
                                        uint(ps_cmd_ptr) + _dg_cmd_pos);              \
                    }                                                                  \
                                                                                       \
                    if (_dg_token >= TOKEN_SHORT_MIN) {                               \
                        /* CUDA reference: src/decode/lz_decode_general.cuh:147-161. */ \
                        /* SHORT: 1-byte token, inline off16 or use_recent. */        \
                        _dg_token_type = TOKEN_TYPE_SHORT;                            \
                        _dg_lit_len = _dg_token & TOKEN_LIT_MASK;                     \
                        _dg_match_len = (_dg_token >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK; \
                        _dg_use_recent = (_dg_token >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK; \
                        if (_dg_use_recent == 0u && _dg_off16_pos < uint(ps_off16_count)) { \
                            uint _dg_v;                                               \
                            off16_read_macro(_dg_v, comp_ssbo, ps_off16_raw,          \
                                              _dg_off16_pos,                          \
                                              off16_hi_ssbo, ps_off16_hi,             \
                                              off16_lo_ssbo, ps_off16_lo);             \
                            _dg_match_offset = -int(_dg_v);                           \
                            _dg_off16_pos = _dg_off16_pos + 1u;                       \
                        }                                                              \
                    } else if (_dg_token == TOKEN_LONG_LITERAL) {                     \
                        /* CUDA reference: src/decode/lz_decode_general.cuh:162-165. */ \
                        _dg_token_type = TOKEN_TYPE_LONG_LITERAL;                     \
                        uint _dg_extlen;                                              \
                        readLength(comp_ssbo, uint(ps_len_stream),                     \
                                    _dg_length_offset, uint(ps_len_avail), _dg_extlen); \
                        _dg_lit_len = _dg_extlen + LONG_LITERAL_BASE;                 \
                    } else if (_dg_token == TOKEN_LONG_NEAR) {                        \
                        /* CUDA reference: src/decode/lz_decode_general.cuh:166-179. */ \
                        _dg_token_type = TOKEN_TYPE_LONG_NEAR;                        \
                        uint _dg_extlen2;                                             \
                        readLength(comp_ssbo, uint(ps_len_stream),                     \
                                    _dg_length_offset, uint(ps_len_avail), _dg_extlen2); \
                        _dg_match_len = _dg_extlen2 + LONG_NEAR_BASE;                 \
                        if (_dg_off16_pos < uint(ps_off16_count)) {                    \
                            uint _dg_v2;                                               \
                            off16_read_macro(_dg_v2, comp_ssbo, ps_off16_raw,         \
                                              _dg_off16_pos,                          \
                                              off16_hi_ssbo, ps_off16_hi,             \
                                              off16_lo_ssbo, ps_off16_lo);             \
                            _dg_match_offset = -int(_dg_v2);                          \
                            _dg_off16_pos = _dg_off16_pos + 1u;                       \
                        }                                                              \
                        _dg_use_recent = 0u;                                          \
                    } else if (_dg_token == TOKEN_LONG_FAR) {                         \
                        /* CUDA reference: src/decode/lz_decode_general.cuh:180-191. */ \
                        _dg_token_type = TOKEN_TYPE_LONG_FAR;                         \
                        uint _dg_extlen3;                                             \
                        readLength(comp_ssbo, uint(ps_len_stream),                     \
                                    _dg_length_offset, uint(ps_len_avail), _dg_extlen3); \
                        _dg_match_len = _dg_extlen3 + LONG_FAR_BASE;                  \
                        if (_dg_off32_pos < _dg_off32_block_count) {                  \
                            /* off32 entries are 3 bytes each (LE u24). */            \
                            uint _dg_pbase = _dg_off32_block_base                     \
                                + _dg_off32_pos * OFF32_ENTRY_BYTES;                  \
                            uint _dg_p0 = uint((comp_ssbo)[_dg_pbase + 0u]);          \
                            uint _dg_p1 = uint((comp_ssbo)[_dg_pbase + 1u]);          \
                            uint _dg_p2 = uint((comp_ssbo)[_dg_pbase + 2u]);          \
                            uint _dg_v3 = readLE24(_dg_p0, _dg_p1, _dg_p2);            \
                            _dg_match_offset = int(_dg_block_dst_start)               \
                                             - int(_dg_v3) - int(_dg_dst_pos);        \
                            _dg_off32_pos = _dg_off32_pos + 1u;                       \
                        }                                                              \
                        _dg_use_recent = 0u;                                          \
                    } else {                                                          \
                        /* CUDA reference: src/decode/lz_decode_general.cuh:192-203. */ \
                        /* SHORT_FAR: token < TOKEN_SHORT_MIN and not one of the */   \
                        /* named long forms — match_len encoded inline as */          \
                        /* token + SHORT_FAR_BASE, off32 entry consumed. */           \
                        _dg_token_type = TOKEN_TYPE_SHORT_FAR;                        \
                        _dg_match_len = _dg_token + SHORT_FAR_BASE;                   \
                        if (_dg_off32_pos < _dg_off32_block_count) {                  \
                            uint _dg_pbase2 = _dg_off32_block_base                    \
                                + _dg_off32_pos * OFF32_ENTRY_BYTES;                  \
                            uint _dg_p0b = uint((comp_ssbo)[_dg_pbase2 + 0u]);        \
                            uint _dg_p1b = uint((comp_ssbo)[_dg_pbase2 + 1u]);        \
                            uint _dg_p2b = uint((comp_ssbo)[_dg_pbase2 + 2u]);        \
                            uint _dg_v4 = readLE24(_dg_p0b, _dg_p1b, _dg_p2b);         \
                            _dg_match_offset = int(_dg_block_dst_start)               \
                                             - int(_dg_v4) - int(_dg_dst_pos);        \
                            _dg_off32_pos = _dg_off32_pos + 1u;                       \
                        }                                                              \
                        _dg_use_recent = 0u;                                          \
                    }                                                                  \
                }                                                                      \
                                                                                       \
                /* CUDA reference: src/decode/lz_decode_general.cuh:212-220. */       \
                /* Broadcast parsed values from lane 0 to all lanes. The */           \
                /* lit_pos / dst_pos shfls are formally redundant (every lane */      \
                /* applies the same update below) but the CUDA banner */              \
                /* documents a ~2% perf regression on L3 enwik8 from removing */      \
                /* them — kept for reorder-barrier effect. */                          \
                _dg_token_type   = subgroupShuffle(_dg_token_type, 0u);               \
                _dg_lit_len      = subgroupShuffle(_dg_lit_len, 0u);                  \
                _dg_match_len    = subgroupShuffle(_dg_match_len, 0u);                \
                _dg_match_offset = subgroupShuffle(_dg_match_offset, 0u);             \
                _dg_use_recent   = subgroupShuffle(_dg_use_recent, 0u);               \
                _dg_cmd_pos      = subgroupShuffle(_dg_cmd_pos, 0u);                  \
                _dg_lit_pos      = subgroupShuffle(_dg_lit_pos, 0u);                  \
                _dg_off16_pos    = subgroupShuffle(_dg_off16_pos, 0u);                \
                _dg_off32_pos    = subgroupShuffle(_dg_off32_pos, 0u);                \
                                                                                       \
                /* CUDA reference: src/decode/lz_decode_general.cuh:222-235. */       \
                /* Warp-cooperative literal copy. mode == 0 → delta literal; */       \
                /* mode != 0 → raw literal. */                                         \
                if (_dg_lit_len > 0u) {                                                \
                    if (uint(mode) == 0u) {                                           \
                        deltaLiteralCopyBounded(dst_ssbo, _dg_dst_pos,                 \
                                                comp_ssbo, lit_scratch_ssbo,           \
                                                ps_lit_in_scratch,                     \
                                                uint(ps_lit_ptr) + _dg_lit_pos,        \
                                                _dg_lit_len, _dg_dst_end_abs,         \
                                                uint(ps_lit_ptr) + uint(ps_lit_size), _dg_recent_offset, lane); \
                    } else {                                                           \
                        warpLiteralCopyBoundedSel(dst_ssbo, _dg_dst_pos,               \
                                                comp_ssbo, lit_scratch_ssbo,           \
                                                ps_lit_in_scratch,                     \
                                                uint(ps_lit_ptr) + _dg_lit_pos,        \
                                                _dg_lit_len, _dg_dst_end_abs,         \
                                                uint(ps_lit_ptr) + uint(ps_lit_size), lane);             \
                    }                                                                  \
                    subgroupBarrier();                                                 \
                    subgroupMemoryBarrierBuffer();                                     \
                    _dg_dst_pos = _dg_dst_pos + _dg_lit_len;                          \
                    _dg_lit_pos = _dg_lit_pos + _dg_lit_len;                          \
                }                                                                      \
                                                                                       \
                /* CUDA reference: src/decode/lz_decode_general.cuh:238-245. */       \
                /* Warp-cooperative match copy. */                                     \
                if (_dg_match_len > 0u) {                                              \
                    uint _dg_match_src = uint(int(_dg_dst_pos) + _dg_match_offset);   \
                    int  _dg_match_dist = -_dg_match_offset;                          \
                    warpMatchCopyBounded(dst_ssbo, _dg_dst_pos,                        \
                                          _dg_match_src, _dg_match_len,                \
                                          _dg_match_dist, _dg_dst_end_abs, lane);     \
                    subgroupBarrier();                                                 \
                    subgroupMemoryBarrierBuffer();                                     \
                    _dg_dst_pos = _dg_dst_pos + _dg_match_len;                        \
                }                                                                      \
                                                                                       \
                /* CUDA reference: src/decode/lz_decode_general.cuh:247-253. */       \
                /* dst_pos / lit_pos shfls (see notes on the earlier shfls). */       \
                _dg_dst_pos = subgroupShuffle(_dg_dst_pos, 0u);                       \
                _dg_lit_pos = subgroupShuffle(_dg_lit_pos, 0u);                       \
                if (_dg_use_recent == 0u &&                                            \
                    _dg_token_type != TOKEN_TYPE_LONG_LITERAL) {                       \
                    _dg_recent_offset = _dg_match_offset;                             \
                }                                                                      \
                _dg_recent_offset = subgroupShuffle(_dg_recent_offset, 0u);           \
                _dg_length_offset = subgroupShuffle(_dg_length_offset, 0u);           \
            }                                                                          \
                                                                                       \
            /* CUDA reference: src/decode/lz_decode_general.cuh:256-276. */           \
            /* Per-block trailing literals (at 64 KB boundary). */                     \
            subgroupBarrier();                                                         \
            subgroupMemoryBarrierBuffer();                                             \
            _dg_dst_pos = subgroupShuffle(_dg_dst_pos, 0u);                           \
            _dg_lit_pos = subgroupShuffle(_dg_lit_pos, 0u);                           \
            {                                                                          \
                uint _dg_block_end = _dg_block_dst_start + LZ_BLOCK_SIZE;             \
                if (_dg_block_end > _dg_dst_end_abs) _dg_block_end = _dg_dst_end_abs; \
                uint _dg_block_trailing = (_dg_block_end > _dg_dst_pos)               \
                    ? (_dg_block_end - _dg_dst_pos) : 0u;                              \
                if (uint(mode) == 0u) {                                                \
                    deltaLiteralCopyBounded(dst_ssbo, _dg_dst_pos,                     \
                                            comp_ssbo, lit_scratch_ssbo,               \
                                            ps_lit_in_scratch,                         \
                                            uint(ps_lit_ptr) + _dg_lit_pos,            \
                                            _dg_block_trailing, _dg_dst_end_abs,      \
                                            uint(ps_lit_ptr) + uint(ps_lit_size), _dg_recent_offset, lane); \
                } else {                                                               \
                    warpLiteralCopyBoundedSel(dst_ssbo, _dg_dst_pos,                   \
                                            comp_ssbo, lit_scratch_ssbo,               \
                                            ps_lit_in_scratch,                         \
                                            uint(ps_lit_ptr) + _dg_lit_pos,            \
                                            _dg_block_trailing, _dg_dst_end_abs,      \
                                            uint(ps_lit_ptr) + uint(ps_lit_size), lane);                 \
                }                                                                      \
                subgroupBarrier();                                                     \
                subgroupMemoryBarrierBuffer();                                         \
                _dg_dst_pos = _dg_dst_pos + _dg_block_trailing;                       \
                _dg_lit_pos = _dg_lit_pos + _dg_block_trailing;                       \
            }                                                                          \
                                                                                       \
            /* CUDA reference: src/decode/lz_decode_general.cuh:278-288. */           \
            /* Advance to block 2. On the last iter these writes are dead — */        \
            /* the for-loop bound exits before they are read. Letting the */          \
            /* setup run on the dead iteration trades a never-taken `break` */        \
            /* for a few unconditional writes that the compiler DCEs. */               \
            _dg_block_cmd_end = uint(ps_cmd_size);                                    \
            _dg_off32_block_base = uint(ps_off32_raw2);                               \
            _dg_off32_block_count = uint(ps_off32_count2);                            \
            _dg_off32_pos = 0u;                                                       \
            _dg_block_dst_start = _dg_dst_pos;                                        \
            if (uint(lane) == 0u && _dg_cmd_pos < _dg_block_cmd_end) {                \
                readCmdByteSel(_dg_prefetched_token, comp_ssbo,                        \
                                cmd_scratch_ssbo, ps_cmd_in_scratch,                   \
                                uint(ps_cmd_ptr) + _dg_cmd_pos);                      \
            }                                                                          \
        }                                                                              \
                                                                                       \
        /* CUDA reference: src/decode/lz_decode_general.cuh:291-308. */               \
        /* Final trailing literals. The mode == 0 branch uses the bounded */          \
        /* delta-literal helper; the mode != 0 branch open-codes the loop */          \
        /* without the lit_pos bound check (CUDA's original form — the */             \
        /* trailing count itself derives from lit_size, so the bound is moot). */     \
        subgroupBarrier();                                                             \
        subgroupMemoryBarrierBuffer();                                                 \
        _dg_dst_pos = subgroupShuffle(_dg_dst_pos, 0u);                               \
        _dg_lit_pos = subgroupShuffle(_dg_lit_pos, 0u);                               \
        {                                                                              \
            uint _dg_trailing = (uint(ps_lit_size) > _dg_lit_pos)                     \
                ? (uint(ps_lit_size) - _dg_lit_pos) : 0u;                              \
            if (uint(mode) == 0u) {                                                    \
                deltaLiteralCopyBounded(dst_ssbo, _dg_dst_pos,                         \
                                        comp_ssbo, lit_scratch_ssbo,                   \
                                        ps_lit_in_scratch,                             \
                                        uint(ps_lit_ptr) + _dg_lit_pos,                \
                                        _dg_trailing, _dg_dst_end_abs,                \
                                        uint(ps_lit_ptr) + uint(ps_lit_size), _dg_recent_offset, lane);   \
            } else {                                                                   \
                /* CUDA reference: src/decode/lz_decode_general.cuh:304-307. */        \
                /* Final-trailing lacks the lit_pos<lit_size guard CUDA's */            \
                /* bounded helper would add; mirror that here, also picking */         \
                /* the source SSBO based on ps_lit_in_scratch. */                       \
                for (uint _dg_i = uint(lane); _dg_i < _dg_trailing; _dg_i += WARP_SIZE) { \
                    if (_dg_dst_pos + _dg_i < _dg_dst_end_abs) {                       \
                        uint _dg_litb = (uint(ps_lit_in_scratch) != 0u)                 \
                            ? uint((lit_scratch_ssbo)[uint(ps_lit_ptr) + _dg_lit_pos + _dg_i]) \
                            : uint((comp_ssbo)[uint(ps_lit_ptr) + _dg_lit_pos + _dg_i]); \
                        (dst_ssbo)[_dg_dst_pos + _dg_i] = uint8_t(_dg_litb);            \
                    }                                                                   \
                }                                                                       \
            }                                                                           \
        }                                                                              \
    } while (false)

// CUDA reference: src/decode/lz_decode_general.cuh:77 (OFF16_SPLIT=false
// instantiation). Reads off16 from interleaved u16 entries in
// `comp_ssbo[ps_off16_raw + entry*2]`.
#define decodeSubChunkGeneral_false(comp_ssbo, ps_lit_ptr, ps_lit_size,              \
                                    ps_cmd_ptr, ps_cmd_size,                         \
                                    ps_off16_raw, ps_off16_count,                    \
                                    off16_hi_ssbo, ps_off16_hi,                      \
                                    off16_lo_ssbo, ps_off16_lo,                      \
                                    ps_off32_raw1, ps_off32_count1,                  \
                                    ps_off32_raw2, ps_off32_count2,                  \
                                    ps_len_stream, ps_len_avail,                     \
                                    ps_off16_split,                                  \
                                    ps_cmd_stream2_offset, ps_initial_copy,          \
                                    lit_scratch_ssbo, ps_lit_in_scratch,              \
                                    cmd_scratch_ssbo, ps_cmd_in_scratch,              \
                                    dst_ssbo, out_dst_size, out_dst_offset,          \
                                    mode, lane)                                       \
    _SLZ_DECODE_GENERAL_BODY(comp_ssbo, ps_lit_ptr, ps_lit_size,                     \
                              ps_cmd_ptr, ps_cmd_size,                               \
                              ps_off16_raw, ps_off16_count,                          \
                              off16_hi_ssbo, ps_off16_hi,                            \
                              off16_lo_ssbo, ps_off16_lo,                            \
                              ps_off32_raw1, ps_off32_count1,                        \
                              ps_off32_raw2, ps_off32_count2,                        \
                              ps_len_stream, ps_len_avail,                           \
                              ps_off16_split,                                        \
                              ps_cmd_stream2_offset, ps_initial_copy,                \
                              lit_scratch_ssbo, ps_lit_in_scratch,                    \
                              cmd_scratch_ssbo, ps_cmd_in_scratch,                    \
                              dst_ssbo, out_dst_size, out_dst_offset,                \
                              mode, lane, _SLZ_OFF16_READ_GEN_FALSE)

// CUDA reference: src/decode/lz_decode_general.cuh:77 (OFF16_SPLIT=true
// instantiation). Reads off16 from split hi/lo byte streams in
// `off16_hi_ssbo[ps_off16_hi + entry]` / `off16_lo_ssbo[ps_off16_lo + entry]`.
#define decodeSubChunkGeneral_true(comp_ssbo, ps_lit_ptr, ps_lit_size,               \
                                   ps_cmd_ptr, ps_cmd_size,                          \
                                   ps_off16_raw, ps_off16_count,                     \
                                   off16_hi_ssbo, ps_off16_hi,                       \
                                   off16_lo_ssbo, ps_off16_lo,                       \
                                   ps_off32_raw1, ps_off32_count1,                   \
                                   ps_off32_raw2, ps_off32_count2,                   \
                                   ps_len_stream, ps_len_avail,                      \
                                   ps_off16_split,                                   \
                                   ps_cmd_stream2_offset, ps_initial_copy,           \
                                   lit_scratch_ssbo, ps_lit_in_scratch,               \
                                   cmd_scratch_ssbo, ps_cmd_in_scratch,               \
                                   dst_ssbo, out_dst_size, out_dst_offset,           \
                                   mode, lane)                                        \
    _SLZ_DECODE_GENERAL_BODY(comp_ssbo, ps_lit_ptr, ps_lit_size,                     \
                              ps_cmd_ptr, ps_cmd_size,                               \
                              ps_off16_raw, ps_off16_count,                          \
                              off16_hi_ssbo, ps_off16_hi,                            \
                              off16_lo_ssbo, ps_off16_lo,                            \
                              ps_off32_raw1, ps_off32_count1,                        \
                              ps_off32_raw2, ps_off32_count2,                        \
                              ps_len_stream, ps_len_avail,                           \
                              ps_off16_split,                                        \
                              ps_cmd_stream2_offset, ps_initial_copy,                \
                              lit_scratch_ssbo, ps_lit_in_scratch,                    \
                              cmd_scratch_ssbo, ps_cmd_in_scratch,                    \
                              dst_ssbo, out_dst_size, out_dst_offset,                \
                              mode, lane, _SLZ_OFF16_READ_GEN_TRUE)

#endif
