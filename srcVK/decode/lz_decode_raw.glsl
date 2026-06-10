// 1:1 port of src/decode/lz_decode_raw.cuh.
// Raw-mode sub-chunk decoder used by lz_decode_raw_kernel.comp. L1 hot
// path. Streamlined single-LZ-block path: no off32, no delta literals,
// no block split.
//

#ifndef SRCVK_DECODE_LZ_DECODE_RAW_GLSL
#define SRCVK_DECODE_LZ_DECODE_RAW_GLSL

#include "lz_decode_core.glsl"

// CUDA reference: src/decode/lz_decode_raw.cuh:24-211.
// Templated `decodeSubChunkRawMode<OFF16_SPLIT>` in CUDA. Both
// instantiations are live: <false> for interleaved-u16 off16 (the
// common raw L1 path), <true> for an entropy-coded hi/lo split off16
// (general-path callers).
//
// VK adaptation: GLSL cannot pass SSBOs as function parameters AND
// cannot select a struct field on a compile-time template parameter.
// The CUDA `if constexpr (OFF16_SPLIT)` branches that swap between
// `off16_raw + entry*2` and `off16_hi[entry]/off16_lo[entry]` reads are
// realized as two separate macro instantiations:
//   decodeSubChunkRawMode_false(...) → mirrors decodeSubChunkRawMode<false>
//   decodeSubChunkRawMode_true(...)  → mirrors decodeSubChunkRawMode<true>
// Each macro hardcodes the off16 read shape its CUDA instantiation
// produces; nvcc dead-code-eliminates the unused branch + unused pointer
// params for each specialization, which is exactly what the two macros
// here reproduce.

// CUDA reference: src/decode/lz_decode_raw.cuh:67-76 / 152-156 / 168-173.
// Off16 read helper macros. Read one off16 entry from the appropriate
// stream(s) for the current OFF16_SPLIT instantiation.
//   _SLZ_OFF16_READ_FALSE: read u16 from off16_raw at entry_idx*2 / +1
//   _SLZ_OFF16_READ_TRUE : read lo from off16_lo[entry], hi from off16_hi[entry]
#define _SLZ_OFF16_READ_FALSE(out_v, off16_ssbo, off16_base, entry_idx, off16_hi_ssbo, off16_hi_base, off16_lo_ssbo, off16_lo_base) \
    do {                                                                                              \
        uint _o16_lo = uint((off16_ssbo)[uint(off16_base) + (entry_idx) * OFF16_ENTRY_BYTES + 0u]);   \
        uint _o16_hi = uint((off16_ssbo)[uint(off16_base) + (entry_idx) * OFF16_ENTRY_BYTES + 1u]);   \
        (out_v) = readU16LE(_o16_lo, _o16_hi);                                                        \
    } while (false)

#define _SLZ_OFF16_READ_TRUE(out_v, off16_ssbo, off16_base, entry_idx, off16_hi_ssbo, off16_hi_base, off16_lo_ssbo, off16_lo_base) \
    do {                                                                                              \
        uint _o16_lo = uint((off16_lo_ssbo)[uint(off16_lo_base) + (entry_idx)]);                      \
        uint _o16_hi = uint((off16_hi_ssbo)[uint(off16_hi_base) + (entry_idx)]);                      \
        (out_v) = _o16_lo | (_o16_hi << 8);                                                           \
    } while (false)

// Core macro body — parameterized on the off16 read shape via
// `off16_read_macro`. Both decodeSubChunkRawMode_false and
// decodeSubChunkRawMode_true delegate here.
//
// Args (all u32 unless noted):
//   cmd_ssbo, cmd_base    - command (token) stream SSBO + base byte offset
//   cmd_size              - bytes in the cmd stream
//   lit_ssbo, lit_base    - literal stream SSBO + base
//   lit_size              - bytes in the lit stream
//   off16_ssbo, off16_base- off16 stream SSBO + base (used by _FALSE branch)
//   off16_hi_ssbo, off16_hi_base - hi-byte off16 SSBO + base (used by _TRUE branch)
//   off16_lo_ssbo, off16_lo_base - lo-byte off16 SSBO + base (used by _TRUE branch)
//   off16_count           - number of off16 entries
//   length_ssbo, length_base - length stream SSBO + base
//   length_remaining      - bytes available in length stream
//   dst_ssbo              - destination SSBO (writes go here AND match reads come from here)
//   dst_size              - decompressed size of this sub-chunk
//   initial_copy          - 8 if this sub-chunk did the initial copy, else 0
//   dst_offset            - absolute dst position where this sub-chunk starts
//   lane                  - this invocation's lane id (gl_SubgroupInvocationID & 31)
//   off16_read_macro      - _SLZ_OFF16_READ_FALSE or _SLZ_OFF16_READ_TRUE
#define _SLZ_DECODE_RAW_BODY(cmd_ssbo, cmd_base, cmd_size, lit_ssbo, lit_base, lit_size, \
                              off16_ssbo, off16_base,                                    \
                              off16_hi_ssbo, off16_hi_base,                              \
                              off16_lo_ssbo, off16_lo_base,                              \
                              off16_count,                                               \
                              length_ssbo, length_base, length_remaining,                \
                              dst_ssbo, dst_size, initial_copy, dst_offset, lane,        \
                              off16_read_macro)                                          \
    do {                                                                                  \
        uint _ds_cmd_pos = 0u, _ds_lit_pos = 0u, _ds_off16_pos = 0u;                      \
        uint _ds_dst_pos = uint(dst_offset) + uint(initial_copy);                         \
        int  _ds_recent_offset = INITIAL_RECENT_OFFSET;                                   \
        uint _ds_length_offset = 0u;                                                      \
                                                                                          \
        while (_ds_cmd_pos < uint(cmd_size)) {                                            \
            /* CUDA reference: src/decode/lz_decode_raw.cuh:46-135.                      \
               Parallel-parse fast path. 32 tokens per outer iter. */                     \
            {                                                                             \
                uint _ds_remaining = uint(cmd_size) - _ds_cmd_pos;                        \
                uint _ds_batch_size = _ds_remaining < WARP_SIZE ? _ds_remaining : WARP_SIZE; \
                uint _ds_my_cmd = (uint(lane) < _ds_batch_size)                           \
                    ? uint((cmd_ssbo)[uint(cmd_base) + _ds_cmd_pos + uint(lane)])         \
                    : 0u;                                                                  \
                bool _ds_my_is_long = (uint(lane) < _ds_batch_size) &&                    \
                                      (_ds_my_cmd < TOKEN_SHORT_MIN);                     \
                uvec4 _ds_long_ballot = subgroupBallot(_ds_my_is_long);                   \
                uint _ds_any_long = _ds_long_ballot.x;                                    \
                                                                                          \
                /* PP-prefix truncation (CUDA ref: src/decode/lz_decode_raw.cuh,         \
                   db1e061): a long token at lane j used to force tokens 0..j-1          \
                   through the serial path one at a time. Truncate the batch to          \
                   the all-short prefix and PP it; the next window starts AT the          \
                   long token and goes serial exactly once. findLSB == CUDA's            \
                   __ffs(x)-1 (x != 0 here). The PP body already handles                  \
                   batch_size < 32 via the _ds_my_valid guards. */                        \
                if (_ds_any_long != 0u)                                                   \
                    _ds_batch_size = uint(findLSB(_ds_any_long));                         \
                                                                                          \
                if (_ds_batch_size > 0u) {                                                \
                    bool _ds_my_valid = uint(lane) < _ds_batch_size;                      \
                    uint _ds_my_lit_len   = _ds_my_valid ? (_ds_my_cmd & TOKEN_LIT_MASK) : 0u; \
                    uint _ds_my_match_len = _ds_my_valid                                  \
                        ? ((_ds_my_cmd >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK) : 0u;    \
                    uint _ds_my_use_recent = _ds_my_valid                                 \
                        ? ((_ds_my_cmd >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK) : 0u; \
                    uint _ds_my_consumes_off16 = (_ds_my_valid && _ds_my_use_recent == 0u) ? 1u : 0u; \
                                                                                          \
                    uint _ds_my_off16_local, _ds_total_off16_used;                        \
                    warpScanU32(_ds_my_consumes_off16, int(lane),                         \
                                _ds_my_off16_local, _ds_total_off16_used);                \
                                                                                          \
                    int _ds_my_match_offset = _ds_recent_offset;                          \
                    if (_ds_my_consumes_off16 != 0u) {                                    \
                        uint _ds_entry_idx = _ds_off16_pos + _ds_my_off16_local;          \
                        if (_ds_entry_idx < uint(off16_count)) {                          \
                            uint _ds_v;                                                   \
                            off16_read_macro(_ds_v, off16_ssbo, off16_base, _ds_entry_idx, \
                                             off16_hi_ssbo, off16_hi_base,                \
                                             off16_lo_ssbo, off16_lo_base);               \
                            _ds_my_match_offset = -int(_ds_v);                            \
                        }                                                                 \
                    }                                                                     \
                                                                                          \
                    uvec4 _ds_fresh_ballot = subgroupBallot(_ds_my_consumes_off16 != 0u); \
                    uint _ds_fresh_mask = _ds_fresh_ballot.x;                             \
                    /* CUDA reference: src/decode/lz_decode_raw.cuh:79-88.               \
                       `(2u << lane) - 1u` inclusive-prefix mask (lane in [0,31]). */     \
                    uint _ds_my_prefix = _ds_fresh_mask & ((2u << uint(lane)) - 1u);      \
                    int _ds_src_lane = (_ds_my_prefix != 0u)                              \
                        ? lastBitSet(_ds_my_prefix) : 0;                                  \
                    int _ds_shuffled_off = subgroupShuffle(_ds_my_match_offset,           \
                                                            uint(_ds_src_lane));          \
                    if (_ds_my_use_recent != 0u && _ds_my_prefix != 0u) {                 \
                        _ds_my_match_offset = _ds_shuffled_off;                           \
                    }                                                                     \
                                                                                          \
                    uint _ds_my_total = _ds_my_lit_len + _ds_my_match_len;                \
                    uint _ds_my_dst_local, _ds_total_dst;                                 \
                    warpScanU32(_ds_my_total, int(lane),                                  \
                                _ds_my_dst_local, _ds_total_dst);                         \
                    uint _ds_my_lit_local, _ds_total_lit;                                 \
                    warpScanU32(_ds_my_lit_len, int(lane),                                \
                                _ds_my_lit_local, _ds_total_lit);                         \
                                                                                          \
                    for (uint _ds_k = 0u; _ds_k < _ds_batch_size; _ds_k += 1u) {          \
                        uint _ds_k_lit_len   = subgroupShuffle(_ds_my_lit_len, _ds_k);    \
                        uint _ds_k_match_len = subgroupShuffle(_ds_my_match_len, _ds_k);  \
                        int  _ds_k_match_off = subgroupShuffle(_ds_my_match_offset, _ds_k); \
                        uint _ds_k_dst_local = subgroupShuffle(_ds_my_dst_local, _ds_k);  \
                        uint _ds_k_lit_local = subgroupShuffle(_ds_my_lit_local, _ds_k);  \
                                                                                          \
                        uint _ds_this_dst_pos = _ds_dst_pos + _ds_k_dst_local;            \
                        uint _ds_this_lit_pos = _ds_lit_pos + _ds_k_lit_local;            \
                                                                                          \
                        if (_ds_k_lit_len > 0u) {                                         \
                            warpLiteralCopy(dst_ssbo, _ds_this_dst_pos,                   \
                                            lit_ssbo, uint(lit_base) + _ds_this_lit_pos,  \
                                            _ds_k_lit_len, lane);                         \
                            subgroupBarrier();                                            \
                            subgroupMemoryBarrierBuffer();                                \
                        }                                                                 \
                        uint _ds_copy_dst = _ds_this_dst_pos + _ds_k_lit_len;             \
                                                                                          \
                        if (_ds_k_match_len > 0u) {                                       \
                            uint _ds_match_src = uint(int(_ds_copy_dst) + _ds_k_match_off); \
                            warpMatchCopy(dst_ssbo, _ds_copy_dst,                         \
                                          _ds_match_src, _ds_k_match_len,                 \
                                          -_ds_k_match_off, lane);                        \
                            subgroupBarrier();                                            \
                            subgroupMemoryBarrierBuffer();                                \
                        }                                                                 \
                    }                                                                     \
                                                                                          \
                    _ds_cmd_pos   += _ds_batch_size;                                      \
                    _ds_off16_pos += _ds_total_off16_used;                                \
                    _ds_dst_pos   += _ds_total_dst;                                       \
                    _ds_lit_pos   += _ds_total_lit;                                       \
                                                                                          \
                    if (_ds_fresh_mask != 0u) {                                           \
                        int _ds_last_fresh = lastBitSet(_ds_fresh_mask);                  \
                        _ds_recent_offset = subgroupShuffle(_ds_my_match_offset,          \
                                                             uint(_ds_last_fresh));       \
                    }                                                                     \
                    continue;                                                             \
                }                                                                         \
            }                                                                             \
                                                                                          \
            /* CUDA reference: src/decode/lz_decode_raw.cuh:137-203.                     \
               Slow path: lane-0 token parse with broadcast. */                           \
            uint _ds_lit_len = 0u, _ds_match_len = 0u;                                    \
            int  _ds_match_offset = _ds_recent_offset;                                    \
            uint _ds_use_recent = 0u;                                                     \
                                                                                          \
            if (uint(lane) == 0u) {                                                       \
                uint _ds_token = uint((cmd_ssbo)[uint(cmd_base) + _ds_cmd_pos]);          \
                if (_ds_token >= TOKEN_SHORT_MIN) {                                       \
                    _ds_lit_len = _ds_token & TOKEN_LIT_MASK;                             \
                    _ds_match_len = (_ds_token >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK;  \
                    _ds_use_recent = (_ds_token >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK; \
                    if (_ds_use_recent == 0u && _ds_off16_pos < uint(off16_count)) {      \
                        uint _ds_v2;                                                      \
                        off16_read_macro(_ds_v2, off16_ssbo, off16_base, _ds_off16_pos,   \
                                         off16_hi_ssbo, off16_hi_base,                    \
                                         off16_lo_ssbo, off16_lo_base);                   \
                        _ds_match_offset = -int(_ds_v2);                                  \
                        _ds_off16_pos += 1u;                                              \
                    }                                                                     \
                } else if (_ds_token == TOKEN_LONG_LITERAL) {                             \
                    _ds_use_recent = 1u;                                                  \
                    uint _ds_extlen;                                                      \
                    readLength(length_ssbo, uint(length_base),                            \
                               _ds_length_offset, uint(length_remaining), _ds_extlen);    \
                    _ds_lit_len = _ds_extlen + LONG_LITERAL_BASE;                         \
                } else if (_ds_token == TOKEN_LONG_NEAR) {                                \
                    uint _ds_extlen2;                                                     \
                    readLength(length_ssbo, uint(length_base),                            \
                               _ds_length_offset, uint(length_remaining), _ds_extlen2);   \
                    _ds_match_len = _ds_extlen2 + LONG_NEAR_BASE;                         \
                    if (_ds_off16_pos < uint(off16_count)) {                              \
                        uint _ds_v3;                                                      \
                        off16_read_macro(_ds_v3, off16_ssbo, off16_base, _ds_off16_pos,   \
                                         off16_hi_ssbo, off16_hi_base,                    \
                                         off16_lo_ssbo, off16_lo_base);                   \
                        _ds_match_offset = -int(_ds_v3);                                  \
                        _ds_off16_pos += 1u;                                              \
                    }                                                                     \
                }                                                                         \
            }                                                                             \
            _ds_cmd_pos += 1u;                                                            \
                                                                                          \
            /* Broadcast parsed values from lane 0 to all lanes. */                       \
            _ds_lit_len      = subgroupShuffle(_ds_lit_len, 0u);                          \
            _ds_match_len    = subgroupShuffle(_ds_match_len, 0u);                        \
            _ds_match_offset = subgroupShuffle(_ds_match_offset, 0u);                     \
            _ds_use_recent   = subgroupShuffle(_ds_use_recent, 0u);                       \
            _ds_off16_pos    = subgroupShuffle(_ds_off16_pos, 0u);                        \
                                                                                          \
            if (_ds_lit_len > 0u) {                                                       \
                warpLiteralCopy(dst_ssbo, _ds_dst_pos,                                    \
                                lit_ssbo, uint(lit_base) + _ds_lit_pos,                   \
                                _ds_lit_len, lane);                                       \
                subgroupBarrier();                                                        \
                subgroupMemoryBarrierBuffer();                                            \
            }                                                                             \
            _ds_dst_pos += _ds_lit_len;                                                   \
            _ds_lit_pos += _ds_lit_len;                                                   \
                                                                                          \
            if (_ds_match_len > 0u) {                                                     \
                uint _ds_match_src = uint(int(_ds_dst_pos) + _ds_match_offset);           \
                warpMatchCopy(dst_ssbo, _ds_dst_pos,                                      \
                              _ds_match_src, _ds_match_len,                               \
                              -_ds_match_offset, lane);                                   \
                subgroupBarrier();                                                        \
                subgroupMemoryBarrierBuffer();                                            \
            }                                                                             \
            _ds_dst_pos += _ds_match_len;                                                 \
                                                                                          \
            if (_ds_use_recent == 0u) _ds_recent_offset = _ds_match_offset;               \
            _ds_recent_offset = subgroupShuffle(_ds_recent_offset, 0u);                   \
            _ds_length_offset = subgroupShuffle(_ds_length_offset, 0u);                   \
        }                                                                                 \
                                                                                          \
        /* CUDA reference: src/decode/lz_decode_raw.cuh:206-209. Trailing literals. */    \
        {                                                                                 \
            uint _ds_trailing = (uint(lit_size) > _ds_lit_pos)                            \
                ? (uint(lit_size) - _ds_lit_pos) : 0u;                                    \
            for (uint _ds_i = uint(lane); _ds_i < _ds_trailing; _ds_i += WARP_SIZE)       \
                (dst_ssbo)[_ds_dst_pos + _ds_i] =                                         \
                    (lit_ssbo)[uint(lit_base) + _ds_lit_pos + _ds_i];                     \
        }                                                                                 \
    } while (false)

// CUDA reference: src/decode/lz_decode_raw.cuh:24 (OFF16_SPLIT=false
// instantiation). Off16 is interleaved u16 in `off16_raw`. The
// off16_hi/off16_lo SSBO/base args are unused by this macro — pass the
// raw SSBO + 0 as placeholders.
#define decodeSubChunkRawMode_false(cmd_ssbo, cmd_base, cmd_size, lit_ssbo, lit_base, lit_size, \
                                    off16_ssbo, off16_base, off16_count,                       \
                                    length_ssbo, length_base, length_remaining,                \
                                    dst_ssbo, dst_size, initial_copy, dst_offset, lane)        \
    _SLZ_DECODE_RAW_BODY(cmd_ssbo, cmd_base, cmd_size, lit_ssbo, lit_base, lit_size,           \
                          off16_ssbo, off16_base,                                              \
                          off16_ssbo, 0u, off16_ssbo, 0u,                                      \
                          off16_count,                                                         \
                          length_ssbo, length_base, length_remaining,                          \
                          dst_ssbo, dst_size, initial_copy, dst_offset, lane,                  \
                          _SLZ_OFF16_READ_FALSE)

// CUDA reference: src/decode/lz_decode_raw.cuh:24 (OFF16_SPLIT=true
// instantiation). Off16 is split into separate hi/lo byte streams. The
// off16_raw SSBO/base args are unused — pass the hi SSBO + 0 as
// placeholders.
#define decodeSubChunkRawMode_true(cmd_ssbo, cmd_base, cmd_size, lit_ssbo, lit_base, lit_size, \
                                   off16_hi_ssbo, off16_hi_base,                              \
                                   off16_lo_ssbo, off16_lo_base, off16_count,                 \
                                   length_ssbo, length_base, length_remaining,                \
                                   dst_ssbo, dst_size, initial_copy, dst_offset, lane)        \
    _SLZ_DECODE_RAW_BODY(cmd_ssbo, cmd_base, cmd_size, lit_ssbo, lit_base, lit_size,          \
                          off16_hi_ssbo, 0u,                                                  \
                          off16_hi_ssbo, off16_hi_base,                                       \
                          off16_lo_ssbo, off16_lo_base,                                       \
                          off16_count,                                                        \
                          length_ssbo, length_base, length_remaining,                         \
                          dst_ssbo, dst_size, initial_copy, dst_offset, lane,                 \
                          _SLZ_OFF16_READ_TRUE)

#endif
