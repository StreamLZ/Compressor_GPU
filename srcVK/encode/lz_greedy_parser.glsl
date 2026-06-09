// 1:1 port of src/encode/lz_greedy_parser.cuh.
// Warp-parallel greedy LZ parser. All 32 lanes probe the hash table in
// lockstep and simulate the CPU's serial hash-write order. L1 hot path.
//
// CUDA -> GLSL intrinsic mapping (verbatim, FULL_WARP_MASK assumed):
//   __shfl_sync(mask, v, src)     -> subgroupShuffle(v, src)
//   __shfl_down_sync(mask, v, d)  -> subgroupShuffleDown(v, d)
//   __ballot_sync(mask, pred)     -> subgroupBallot(pred).x  (low 32 bits)
//   __match_any_sync(mask, key)   -> openCodedMatchAny(key)  (see below)
//   __ffs(x)                      -> (x == 0u) ? 0u : (findLSB(x) + 1u)
//   __clz(x)                      -> (x == 0u) ? 32u : (31u - findMSB(x))
//   __popc(x)                     -> bitCount(x)
//   __syncwarp()                  -> subgroupBarrier(); subgroupMemoryBarrierBuffer();
//
// See srcVK/PortInstructions.md.

#ifndef SRCVK_ENCODE_LZ_GREEDY_PARSER_GLSL
#define SRCVK_ENCODE_LZ_GREEDY_PARSER_GLSL

#include "lz_format.glsl"
#include "lz_token_emit.glsl"

// CUDA reference: src/encode/lz_greedy_parser.cuh:13-18. Greedy-parser
// tuning constants.
const uint HASH_LOOKAHEAD            = 7u;
const uint XOR_PATTERN_BYTES123_MASK = 0xFFFFFF00u;
const uint XOR_MATCH_MIN_LEN         = 3u;
const uint NO_MATCH_STEP_THRESHOLD   = 128u;
const uint NO_MATCH_STEP_SHIFT       = 7u;
const uint NO_MATCH_STEP_MAX         = 16u;

// VK adaptation: CUDA's `__match_any_sync(FULL_WARP_MASK, key)` returns
// a bitmask of every lane whose `key` equals this lane's. GLSL has no
// direct equivalent — we open-code it by walking the 32 lanes via
// subgroupShuffle and comparing keys. This is the same shape the
// foundation-wave match_any_bench used and is documented at
// srcVK/PortInstructions.md (intrinsic mapping table).
uint openCodedMatchAny(uint key) {
    uint mask = 0u;
    for (uint i = 0u; i < WARP_SIZE; i += 1u) {
        uint k = subgroupShuffle(key, i);
        if (k == key) mask |= (1u << i);
    }
    return mask;
}

// VK adaptation: CUDA __ffs returns 1+index_of_lowest_set_bit, or 0 for
// x==0. GLSL findLSB returns -1 for 0, so we guard.
uint slzFfs(uint x) {
    return (x == 0u) ? 0u : (uint(findLSB(x)) + 1u);
}

// VK adaptation: CUDA __clz on a u32 returns 32 for x==0, else
// count-of-leading-zeros. GLSL findMSB returns -1 for 0; convert via
// `31 - findMSB`.
uint slzClz(uint x) {
    return (x == 0u) ? 32u : (31u - uint(findMSB(x)));
}

// CUDA reference: src/encode/lz_greedy_parser.cuh:26-393. scanBlock:
// warp-parallel greedy scan of positions [start_pos .. end_pos). All
// stream cursors live in the in-scope identifiers (no OutputStreams
// struct); the macro takes the SSBO names + cursor names matching the
// emit-side conventions in lz_token_emit.glsl.
//
// VK adaptation: macro form so the SSBOs are visible. Local-state
// variables are uniquely prefixed `_sb_` to avoid name collisions with
// caller scope.
//
// A-008 (BDA): `ht_buf` is a HashU32Ref (buffer_reference) carrying the
// raw device address of the per-chunk hash region. `ht_base` is a u32-
// word offset within that region; the macro dereferences via
// `(ht_buf).e[ht_base + h]` — bypassing the per-SSBO-binding 4 GiB cap
// that would otherwise force `hash_bits=18` on inputs >= 128 MiB at L3.
#define scanBlock(src_buf, src_base, src_size_in,                                                   \
                  ht_buf, ht_base, hash_bits_in, hash_mask_in,                                      \
                  lit_buf, lit_base, lit_count,                                                     \
                  token_buf, token_base, token_count,                                               \
                  off16_buf, off16_base, off16_count,                                               \
                  off32_buf, off32_base, off32_pos, off32_count,                                    \
                  len_buf, len_base, len_count,                                                     \
                  anchor_inout, recent_offset_inout,                                                \
                  start_pos_in, end_pos_in, block2_start_in,                                        \
                  enable_match_rehash_in, lane_in)                                                  \
    do {                                                                                            \
        uint _sb_pos = uint(start_pos_in);                                                          \
        uint _sb_end_pos = uint(end_pos_in);                                                        \
        uint _sb_src_size = uint(src_size_in);                                                      \
        uint _sb_hash_bits = uint(hash_bits_in);                                                    \
        uint _sb_hash_mask = uint(hash_mask_in);                                                    \
        int  _sb_lane = int(lane_in);                                                               \
                                                                                                    \
        while (_sb_pos + MIN_MATCH <= _sb_end_pos) {                                                \
            uint _sb_my_pos = _sb_pos + uint(_sb_lane);                                             \
            uint _sb_my_byte = (_sb_my_pos < _sb_src_size)                                          \
                ? uint((src_buf)[uint(src_base) + _sb_my_pos]) : 0u;                                \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:42-53.                              \
               8-byte lookahead via shfl_down. */                                                   \
            uint _sb_b1 = subgroupShuffleDown(_sb_my_byte, 1u);                                     \
            uint _sb_b2 = subgroupShuffleDown(_sb_my_byte, 2u);                                     \
            uint _sb_b3 = subgroupShuffleDown(_sb_my_byte, 3u);                                     \
            uint _sb_b4 = subgroupShuffleDown(_sb_my_byte, 4u);                                     \
            uint _sb_b5 = subgroupShuffleDown(_sb_my_byte, 5u);                                     \
            uint _sb_b6 = subgroupShuffleDown(_sb_my_byte, 6u);                                     \
            uint _sb_b7 = subgroupShuffleDown(_sb_my_byte, 7u);                                     \
            uint _sb_key4 = _sb_my_byte | (_sb_b1 << 8) | (_sb_b2 << 16) | (_sb_b3 << 24);          \
            uvec2 _sb_key8 = uvec2(_sb_key4,                                                        \
                                   _sb_b4 | (_sb_b5 << 8) | (_sb_b6 << 16) | (_sb_b7 << 24));       \
                                                                                                    \
            uint _sb_remaining = _sb_end_pos - _sb_pos;                                             \
            uint _sb_active_count;                                                                  \
            if (_sb_remaining >= WARP_SIZE) _sb_active_count = WARP_SIZE - HASH_LOOKAHEAD;          \
            else if (_sb_remaining >= HASH_LOOKAHEAD + 1u)                                          \
                _sb_active_count = _sb_remaining - HASH_LOOKAHEAD;                                  \
            else _sb_active_count = 0u;                                                             \
            bool _sb_is_active = (uint(_sb_lane) < _sb_active_count);                               \
                                                                                                    \
            uint _sb_h = hashKey6(_sb_key8, _sb_hash_bits, _sb_hash_mask);                          \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:82-84.                              \
               __match_any_sync over the hash bucket. */                                            \
            uint _sb_bucket_same = openCodedMatchAny(_sb_h);                                        \
            uint _sb_active_mask = subgroupBallot(_sb_is_active).x;                                 \
            uint _sb_lower_same_bucket = _sb_bucket_same & _sb_active_mask                          \
                                       & ((1u << uint(_sb_lane)) - 1u);                             \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:86-108.                             \
               VULKAN_PORTABILITY explicit shuffle-and-compare for the                             \
               same-bucket key4 test. The shuffle is issued at warp scope                          \
               (all 32 lanes converged), then consumed only inside the                             \
               divergent branch. */                                                                 \
            uint _sb_key_src_lane = (_sb_lower_same_bucket != 0u)                                   \
                ? ((WARP_SIZE - 1u) - slzClz(_sb_lower_same_bucket))                                \
                : uint(_sb_lane);                                                                   \
            uint _sb_their_key4_warp = subgroupShuffle(_sb_key4, _sb_key_src_lane);                 \
                                                                                                    \
            bool _sb_hash_match = false;                                                            \
            uint _sb_hash_ref = 0u;                                                                 \
                                                                                                    \
            if (_sb_is_active) {                                                                    \
                if (_sb_lower_same_bucket != 0u) {                                                  \
                    uint _sb_highest_lower = _sb_key_src_lane;                                      \
                    bool _sb_same_key = (_sb_their_key4_warp == _sb_key4);                          \
                    uint _sb_their_pos = _sb_pos + _sb_highest_lower;                               \
                    if (_sb_same_key && _sb_their_pos + 8u <= _sb_my_pos) {                         \
                        _sb_hash_match = true;                                                      \
                        _sb_hash_ref = _sb_their_pos;                                               \
                    }                                                                               \
                } else {                                                                            \
                    /* A-008 (BDA): ht_buf is a HashU32Ref (buffer_reference), */                   \
                    /* not an SSBO; index via the runtime-sized array field .e[] */                 \
                    /* instead of the raw [] operator. Macro shape preserved 1:1 */                 \
                    /* with CUDA — only the addressing primitive changes. */                        \
                    uint _sb_ref_val = (ht_buf).e[uint(ht_base) + _sb_h];                           \
                    if (_sb_ref_val != HASH_EMPTY && _sb_ref_val + 8u <= _sb_my_pos) {              \
                        uint _sb_p0 = uint((src_buf)[uint(src_base) + _sb_ref_val + 0u]);           \
                        uint _sb_p1 = uint((src_buf)[uint(src_base) + _sb_ref_val + 1u]);           \
                        uint _sb_p2 = uint((src_buf)[uint(src_base) + _sb_ref_val + 2u]);           \
                        uint _sb_p3 = uint((src_buf)[uint(src_base) + _sb_ref_val + 3u]);           \
                        uint _sb_rk = readU32LE(_sb_p0, _sb_p1, _sb_p2, _sb_p3);                    \
                        if (_sb_rk == _sb_key4) {                                                   \
                            _sb_hash_match = true;                                                  \
                            _sb_hash_ref = _sb_ref_val;                                             \
                        }                                                                           \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:152-174.                            \
               Secondary match attempts: (1) XOR pattern at cursor+1                               \
               with recent_offset; (2) -8 fixed offset. */                                         \
            bool _sb_xor_match = false;                                                             \
            bool _sb_eight_match = false;                                                           \
            if (_sb_is_active && !_sb_hash_match && int(recent_offset_inout) < 0) {                 \
                uint _sb_ro_dist = uint(-int(recent_offset_inout));                                 \
                if (_sb_my_pos >= _sb_ro_dist) {                                                    \
                    uint _sb_ref = _sb_my_pos - _sb_ro_dist;                                        \
                    uint _sb_p0 = uint((src_buf)[uint(src_base) + _sb_ref + 0u]);                   \
                    uint _sb_p1 = uint((src_buf)[uint(src_base) + _sb_ref + 1u]);                   \
                    uint _sb_p2 = uint((src_buf)[uint(src_base) + _sb_ref + 2u]);                   \
                    uint _sb_p3 = uint((src_buf)[uint(src_base) + _sb_ref + 3u]);                   \
                    uint _sb_recent_word = readU32LE(_sb_p0, _sb_p1, _sb_p2, _sb_p3);               \
                    uint _sb_xor_v = _sb_key4 ^ _sb_recent_word;                                    \
                    if ((_sb_xor_v & XOR_PATTERN_BYTES123_MASK) == 0u) {                            \
                        if (_sb_my_pos + 1u + XOR_MATCH_MIN_LEN <= _sb_end_pos) {                   \
                            _sb_xor_match = true;                                                   \
                        }                                                                           \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
            if (_sb_is_active && !_sb_hash_match && !_sb_xor_match &&                               \
                _sb_my_pos >= MIN_HASH_MATCH_OFFSET) {                                              \
                uint _sb_eight_ref = _sb_my_pos - MIN_HASH_MATCH_OFFSET;                            \
                uint _sb_p0 = uint((src_buf)[uint(src_base) + _sb_eight_ref + 0u]);                 \
                uint _sb_p1 = uint((src_buf)[uint(src_base) + _sb_eight_ref + 1u]);                 \
                uint _sb_p2 = uint((src_buf)[uint(src_base) + _sb_eight_ref + 2u]);                 \
                uint _sb_p3 = uint((src_buf)[uint(src_base) + _sb_eight_ref + 3u]);                 \
                uint _sb_eight_word = readU32LE(_sb_p0, _sb_p1, _sb_p2, _sb_p3);                    \
                if (_sb_eight_word == _sb_key4) _sb_eight_match = true;                             \
            }                                                                                       \
                                                                                                    \
            bool _sb_has_match = _sb_hash_match || _sb_xor_match || _sb_eight_match;                \
            uint _sb_my_match_type = _sb_hash_match ? 0u                                            \
                                   : (_sb_xor_match ? 1u                                            \
                                   : (_sb_eight_match ? 2u : 0u));                                  \
            uint _sb_my_ref = _sb_hash_ref;                                                         \
            uint _sb_match_ballot = subgroupBallot(_sb_has_match).x;                                \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:196-203.                            \
               Adaptive no-match step. */                                                           \
            uint _sb_no_match_step = 1u;                                                            \
            if (_sb_match_ballot == 0u) {                                                           \
                uint _sb_dist = _sb_pos - uint(anchor_inout);                                       \
                if (_sb_dist >= NO_MATCH_STEP_THRESHOLD) {                                          \
                    uint _sb_step = (_sb_dist >> NO_MATCH_STEP_SHIFT) + 1u;                         \
                    _sb_no_match_step = (_sb_step > NO_MATCH_STEP_MAX)                              \
                                      ? NO_MATCH_STEP_MAX : _sb_step;                               \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:215-228.                            \
               VULKAN_PORTABILITY explicit highest-lane-winner rewrite                              \
               for the hash-table writes. */                                                       \
            {                                                                                       \
                uint _sb_write_limit = (_sb_match_ballot != 0u)                                     \
                    ? (slzFfs(_sb_match_ballot) - 1u)                                               \
                    : (_sb_no_match_step - 1u);                                                     \
                bool _sb_is_writer = _sb_is_active && (uint(_sb_lane) <= _sb_write_limit);          \
                uint _sb_write_mask = subgroupBallot(_sb_is_writer).x;                              \
                uint _sb_bucket_grp = openCodedMatchAny(_sb_h);                                     \
                uint _sb_group_mask = _sb_bucket_grp & _sb_write_mask;                              \
                int _sb_top_lane = (_sb_group_mask != 0u)                                           \
                    ? int((WARP_SIZE - 1u) - slzClz(_sb_group_mask)) : -1;                          \
                if (_sb_is_writer && _sb_lane == _sb_top_lane) {                                    \
                    /* A-008 (BDA): see comment above on `.e[]` indexing. */                        \
                    (ht_buf).e[uint(ht_base) + _sb_h] = _sb_my_pos;                                 \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            if (_sb_match_ballot == 0u) {                                                           \
                _sb_pos += _sb_no_match_step;                                                       \
                continue;                                                                           \
            }                                                                                       \
                                                                                                    \
            uint _sb_first_lane = slzFfs(_sb_match_ballot) - 1u;                                    \
            uint _sb_winning_type = subgroupShuffle(_sb_my_match_type, _sb_first_lane);             \
            uint _sb_match_pos;                                                                     \
            uint _sb_match_ref;                                                                     \
            uint _sb_min_match_len;                                                                 \
                                                                                                    \
            if (_sb_winning_type == 0u) {                                                           \
                _sb_match_pos = _sb_pos + _sb_first_lane;                                           \
                _sb_match_ref = subgroupShuffle(_sb_my_ref, _sb_first_lane);                        \
                _sb_min_match_len = MIN_MATCH;                                                      \
            } else if (_sb_winning_type == 1u) {                                                    \
                _sb_match_pos = _sb_pos + _sb_first_lane + 1u;                                      \
                _sb_match_ref = _sb_match_pos - uint(-int(recent_offset_inout));                    \
                _sb_min_match_len = XOR_MATCH_MIN_LEN;                                              \
            } else {                                                                                \
                _sb_match_pos = _sb_pos + _sb_first_lane;                                           \
                _sb_match_ref = _sb_match_pos - MIN_HASH_MATCH_OFFSET;                              \
                _sb_min_match_len = MIN_MATCH;                                                      \
            }                                                                                       \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:261-278.                            \
               Match extension from min_match_len, capped at end_pos. */                            \
            uint _sb_match_len = _sb_min_match_len;                                                 \
            uint _sb_max_match = _sb_end_pos - _sb_match_pos;                                       \
            bool _sb_ext_found = false;                                                             \
            for (uint _sb_ext = _sb_min_match_len; _sb_ext < _sb_max_match;                         \
                 _sb_ext += WARP_SIZE) {                                                            \
                uint _sb_check = _sb_ext + uint(_sb_lane);                                          \
                bool _sb_mm;                                                                        \
                if (_sb_check >= _sb_max_match) {                                                   \
                    _sb_mm = true;                                                                  \
                } else {                                                                            \
                    _sb_mm = ((src_buf)[uint(src_base) + _sb_match_pos + _sb_check]                 \
                            != (src_buf)[uint(src_base) + _sb_match_ref + _sb_check]);              \
                }                                                                                   \
                uint _sb_mm_mask = subgroupBallot(_sb_mm).x;                                        \
                if (_sb_mm_mask != 0u) {                                                            \
                    _sb_match_len = _sb_ext + slzFfs(_sb_mm_mask) - 1u;                             \
                    _sb_ext_found = true;                                                           \
                    break;                                                                          \
                }                                                                                   \
            }                                                                                       \
            if (!_sb_ext_found) _sb_match_len = _sb_max_match;                                      \
                                                                                                    \
            {                                                                                       \
                int _sb_neg_off = -int(_sb_match_pos - _sb_match_ref);                              \
                uint _sb_off_param = (_sb_neg_off == int(recent_offset_inout))                      \
                    ? 0u : uint(-_sb_neg_off);                                                      \
                uint _sb_resolved_off = _sb_off_param;                                              \
                if (_sb_resolved_off == 0u) {                                                       \
                    _sb_resolved_off = uint(-int(recent_offset_inout));                             \
                }                                                                                   \
                                                                                                    \
                if (_sb_resolved_off > NEAR_OFFSET_MAX &&                                           \
                    _sb_match_len < FAR_OFFSET_MIN_MATCH) {                                         \
                    _sb_pos = _sb_match_pos + 1u;                                                   \
                    continue;                                                                       \
                }                                                                                   \
                                                                                                    \
                /* CUDA reference: src/encode/lz_greedy_parser.cuh:296-313.                        \
                   Backward extension on lane 0 then broadcast. */                                 \
                uint _sb_bw_steps = 0u;                                                             \
                if (_sb_lane == 0) {                                                                \
                    uint _sb_mp = _sb_match_pos;                                                    \
                    uint _sb_mr = _sb_match_ref;                                                    \
                    while (_sb_mp > uint(anchor_inout) && _sb_mr > 0u &&                            \
                           (src_buf)[uint(src_base) + _sb_mp - 1u] ==                               \
                           (src_buf)[uint(src_base) + _sb_mr - 1u]) {                               \
                        _sb_mp -= 1u;                                                               \
                        _sb_mr -= 1u;                                                               \
                        _sb_bw_steps += 1u;                                                         \
                    }                                                                               \
                }                                                                                   \
                _sb_bw_steps = subgroupShuffle(_sb_bw_steps, 0u);                                   \
                _sb_match_pos -= _sb_bw_steps;                                                      \
                _sb_match_ref -= _sb_bw_steps;                                                      \
                _sb_match_len += _sb_bw_steps;                                                      \
                                                                                                    \
                uint _sb_lit_len = _sb_match_pos - uint(anchor_inout);                              \
                for (uint _sb_i = uint(_sb_lane); _sb_i < _sb_lit_len; _sb_i += WARP_SIZE) {        \
                    (lit_buf)[uint(lit_base) + (lit_count) + _sb_i] =                               \
                        (src_buf)[uint(src_base) + uint(anchor_inout) + _sb_i];                    \
                }                                                                                   \
                (lit_count) += _sb_lit_len;                                                         \
                                                                                                    \
                if (_sb_lit_len <= MAX_INLINE_LITERALS                                              \
                    && _sb_match_len <= MAX_INLINE_MATCH                                            \
                    && _sb_resolved_off <= NEAR_OFFSET_MAX) {                                       \
                    if (_sb_lane == 0) {                                                            \
                        uint _sb_tok = _sb_lit_len + TOKEN_MATCH_SHIFT_MUL * _sb_match_len;         \
                        if (_sb_off_param == 0u) _sb_tok = _sb_tok | TOKEN_RECENT_FLAG;             \
                        (token_buf)[uint(token_base) + (token_count)] = uint8_t(_sb_tok);           \
                        (token_count) += 1u;                                                         \
                        if (_sb_off_param != 0u) {                                                  \
                            _slzStoreOff16(off16_buf, off16_base, off16_count, _sb_off_param);      \
                            (off16_count) += 1u;                                                     \
                        }                                                                           \
                    }                                                                               \
                } else {                                                                            \
                    if (_sb_lane == 0) {                                                            \
                        emitCmd(token_buf, token_base, token_count,                                 \
                                off16_buf, off16_base, off16_count,                                 \
                                off32_buf, off32_base, off32_pos, off32_count,                      \
                                len_buf, len_base, len_count,                                       \
                                _sb_lit_len, _sb_match_len, _sb_off_param,                          \
                                uint(anchor_inout), block2_start_in, int(recent_offset_inout));     \
                    }                                                                               \
                }                                                                                   \
                                                                                                    \
                (recent_offset_inout) = _sb_neg_off;                                                \
            }                                                                                       \
                                                                                                    \
            /* CUDA reference: src/encode/lz_greedy_parser.cuh:354-371.                            \
               L4+ match-range rehash. Insert hash entries inside the                              \
               just-emitted match at exponentially spaced offsets. */                              \
            if (uint(enable_match_rehash_in) != 0u) {                                               \
                uint _sb_ri = 1u << uint(_sb_lane);                                                 \
                uint _sb_rp = _sb_match_pos + _sb_ri;                                               \
                bool _sb_rehash_active = (_sb_ri < _sb_match_len)                                   \
                                       && (_sb_rp + 8u <= _sb_src_size);                            \
                uvec2 _sb_rk = uvec2(0u, 0u);                                                       \
                if (_sb_rehash_active) {                                                            \
                    uint _sb_p0 = uint((src_buf)[uint(src_base) + _sb_rp + 0u]);                    \
                    uint _sb_p1 = uint((src_buf)[uint(src_base) + _sb_rp + 1u]);                    \
                    uint _sb_p2 = uint((src_buf)[uint(src_base) + _sb_rp + 2u]);                    \
                    uint _sb_p3 = uint((src_buf)[uint(src_base) + _sb_rp + 3u]);                    \
                    uint _sb_p4 = uint((src_buf)[uint(src_base) + _sb_rp + 4u]);                    \
                    uint _sb_p5 = uint((src_buf)[uint(src_base) + _sb_rp + 5u]);                    \
                    uint _sb_p6 = uint((src_buf)[uint(src_base) + _sb_rp + 6u]);                    \
                    uint _sb_p7 = uint((src_buf)[uint(src_base) + _sb_rp + 7u]);                    \
                    _sb_rk = readU64LE(_sb_p0, _sb_p1, _sb_p2, _sb_p3,                              \
                                        _sb_p4, _sb_p5, _sb_p6, _sb_p7);                            \
                }                                                                                   \
                uint _sb_h_rehash = hashKey6(_sb_rk, _sb_hash_bits, _sb_hash_mask);                 \
                uint _sb_write_mask = subgroupBallot(_sb_rehash_active).x;                          \
                uint _sb_bucket_grp = openCodedMatchAny(_sb_h_rehash);                              \
                uint _sb_group_mask = _sb_bucket_grp & _sb_write_mask;                              \
                int  _sb_top_lane = (_sb_group_mask != 0u)                                          \
                    ? int((WARP_SIZE - 1u) - slzClz(_sb_group_mask)) : -1;                          \
                if (_sb_rehash_active && _sb_lane == _sb_top_lane) {                                \
                    /* A-008 (BDA): see comment above on `.e[]` indexing. */                        \
                    (ht_buf).e[uint(ht_base) + _sb_h_rehash] = _sb_rp;                              \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            (anchor_inout) = int(_sb_match_pos + _sb_match_len);                                    \
            _sb_pos = uint(anchor_inout);                                                           \
        }                                                                                           \
                                                                                                    \
        /* CUDA reference: src/encode/lz_greedy_parser.cuh:378-393.                                \
           Trailing literals up to end_pos. */                                                     \
        {                                                                                           \
            uint _sb_trailing = _sb_end_pos - uint(anchor_inout);                                   \
            if (_sb_trailing > 0u) {                                                                \
                for (uint _sb_i = uint(_sb_lane); _sb_i < _sb_trailing; _sb_i += WARP_SIZE) {       \
                    (lit_buf)[uint(lit_base) + (lit_count) + _sb_i] =                               \
                        (src_buf)[uint(src_base) + uint(anchor_inout) + _sb_i];                    \
                }                                                                                   \
                (lit_count) += _sb_trailing;                                                        \
                                                                                                    \
                if (_sb_lane == 0) {                                                                \
                    emitCmd(token_buf, token_base, token_count,                                     \
                            off16_buf, off16_base, off16_count,                                     \
                            off32_buf, off32_base, off32_pos, off32_count,                          \
                            len_buf, len_base, len_count,                                           \
                            _sb_trailing, 0u, 0u,                                                   \
                            uint(anchor_inout), block2_start_in, int(recent_offset_inout));         \
                }                                                                                   \
                                                                                                    \
                (anchor_inout) = int(_sb_end_pos);                                                  \
            }                                                                                       \
        }                                                                                           \
    } while (false)

#endif
