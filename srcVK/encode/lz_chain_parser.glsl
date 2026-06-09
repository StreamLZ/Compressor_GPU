// 1:1 port of src/encode/lz_chain_parser.cuh.
// Serial chain-hash lazy LZ parser (use_chain==1 path). L5 surface.
// See srcVK/PortInstructions.md.

#ifndef SRCVK_ENCODE_LZ_CHAIN_PARSER_GLSL
#define SRCVK_ENCODE_LZ_CHAIN_PARSER_GLSL

#include "lz_format.glsl"
#include "lz_token_emit.glsl"

// CUDA reference: src/encode/lz_chain_parser.cuh:16-32. Lazy-match cost
// weights and chain-parser tuning constants.
const int  LAZY_LEN_WEIGHT       = 5;
const int  LAZY_MATCH_PENALTY    = 5;
const int  LAZY_STEP_WEIGHT      = 4;
const int  NEAR_OFFSET_COST_BITS = 16;
const int  FAR_OFFSET_COST_BITS  = 32;
const int  OFFSET_CLASS_LEN_MARGIN = 5;
const int  RECENT_MATCH_MIN       = 2;
const int  RECENT_LEN_MARGIN      = 1;
const int  RECENT_LEN_MARGIN_FAR  = 4;
const uint LONG_LIT_RUN_THRESHOLD = 64u;
const uint CHAIN_MIN_LOOKAHEAD    = 5u;

// CUDA reference: src/encode/lz_chain_parser.cuh:38-45. isLazyMatchBetter:
// returns true if `cand` (found at pos + step) is better than `current`.
// Port of CPU isLazyMatchBetter.
//
// VK adaptation: CUDA takes two `ChainMatch` structs by value. GLSL
// treats `.length` as the built-in array/string length method even on
// a struct member — `cand.length` errors with `'length' : does not
// operate on this type`. The two helpers below are spelled with
// explicit (cand_len, cand_off, cur_len, cur_off) int parameters
// instead. The arithmetic is byte-identical to CUDA.
bool isLazyMatchBetter(int cand_length, int cand_offset,
                       int current_length, int current_offset, int step) {
    int bits_cand = (cand_offset > 0)
        ? ((cand_offset > int(NEAR_OFFSET_MAX)) ? FAR_OFFSET_COST_BITS : NEAR_OFFSET_COST_BITS) : 0;
    int bits_cur  = (current_offset > 0)
        ? ((current_offset > int(NEAR_OFFSET_MAX)) ? FAR_OFFSET_COST_BITS : NEAR_OFFSET_COST_BITS) : 0;
    return LAZY_LEN_WEIGHT * (cand_length - current_length)
         - LAZY_MATCH_PENALTY - (bits_cand - bits_cur) > step * LAZY_STEP_WEIGHT;
}

// CUDA reference: src/encode/lz_chain_parser.cuh:49-56. isMatchBetter:
// accounts for near/far offset cost when comparing two candidate matches.
bool isMatchBetter(int match_length, int match_offset,
                   int best_length, int best_offset) {
    if (match_length == best_length) return match_offset < best_offset;
    if ((match_offset <= int(NEAR_OFFSET_MAX)) == (best_offset <= int(NEAR_OFFSET_MAX)))
        return match_length > best_length;
    if (best_offset <= int(NEAR_OFFSET_MAX)) return match_length > best_length + OFFSET_CLASS_LEN_MARGIN;
    return match_length >= best_length - OFFSET_CLASS_LEN_MARGIN;
}

// CUDA reference: src/encode/lz_chain_parser.cuh:60-65. isBetterThanRecentMatch:
// returns true if the hash match is preferable to the recent-offset match.
bool isBetterThanRecentMatch(int recent_match_length, int match_length, int match_offset) {
    return recent_match_length < RECENT_MATCH_MIN ||
        (recent_match_length + RECENT_LEN_MARGIN < match_length &&
         (recent_match_length + RECENT_LEN_MARGIN_FAR < match_length ||
          match_offset <= int(NEAR_OFFSET_MAX)));
}

// VK adaptation: CUDA exposes `uint16_t* next_hash` for the chain link
// table; SPIR-V std430 has no native u16 SSBO, so the host packs two
// u16 entries per u32 word inside the same global_hash SSBO (see
// `srcVK/encode/lz_encode_kernel.comp:158` table_stride). The two
// helpers below pack/unpack a single 16-bit slot at `idx` in the lane-
// 0-only chain parser; no atomics because the parser is serial on
// lane 0 (no other invocation touches these words).
//
// A-008 (BDA): `ht_buf` is now a HashU32Ref (buffer_reference) carrying
// the raw device address of the per-chunk hash region; the helpers
// below dereference via `(ht_buf).e[idx]` instead of `(ht_buf)[idx]`.
// The 2x u16-into-u32 pack/unpack arithmetic is otherwise unchanged.
#define chainNextHashRead(ht_buf, next_hash_base, idx)                                              \
    (((ht_buf).e[uint(next_hash_base) + (uint(idx) >> 1u)] >> ((uint(idx) & 1u) * 16u)) & 0xFFFFu)

#define chainNextHashWrite(ht_buf, next_hash_base, idx, value)                                      \
    do {                                                                                            \
        uint _cnh_widx = uint(next_hash_base) + (uint(idx) >> 1u);                                  \
        uint _cnh_shift = (uint(idx) & 1u) * 16u;                                                   \
        uint _cnh_old = (ht_buf).e[_cnh_widx];                                                        \
        uint _cnh_keep = _cnh_old & ~(0xFFFFu << _cnh_shift);                                       \
        uint _cnh_new  = _cnh_keep | ((uint(value) & 0xFFFFu) << _cnh_shift);                       \
        (ht_buf).e[_cnh_widx] = _cnh_new;                                                             \
    } while (false)

// VK adaptation: helper to load 4 src bytes as a U32 LE word at a
// given absolute source offset. CUDA's readU32LE takes `const uint8_t*`
// directly; here we wrap the per-byte loads + readU32LE pack in a macro
// so the call sites stay 1:1 with CUDA.
#define _slzLoadU32(src_buf, src_base, off, out_word)                                               \
    do {                                                                                            \
        uint _slu_p0 = uint((src_buf)[uint(src_base) + (off) + 0u]);                                \
        uint _slu_p1 = uint((src_buf)[uint(src_base) + (off) + 1u]);                                \
        uint _slu_p2 = uint((src_buf)[uint(src_base) + (off) + 2u]);                                \
        uint _slu_p3 = uint((src_buf)[uint(src_base) + (off) + 3u]);                                \
        (out_word) = readU32LE(_slu_p0, _slu_p1, _slu_p2, _slu_p3);                                 \
    } while (false)

// VK adaptation: helper to call read8safe with 8 byte-loads at a given
// absolute source offset. Caller must guarantee `pos <= src_size`.
#define _slzRead8Safe(src_buf, src_base, pos_abs, src_size_in, out_word8)                           \
    do {                                                                                            \
        uint _slr_avail = (uint(pos_abs) + 8u <= uint(src_size_in))                                 \
            ? 8u : (uint(src_size_in) > uint(pos_abs) ? (uint(src_size_in) - uint(pos_abs)) : 0u);  \
        uint _slr_b0 = (0u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 0u]) : 0u; \
        uint _slr_b1 = (1u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 1u]) : 0u; \
        uint _slr_b2 = (2u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 2u]) : 0u; \
        uint _slr_b3 = (3u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 3u]) : 0u; \
        uint _slr_b4 = (4u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 4u]) : 0u; \
        uint _slr_b5 = (5u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 5u]) : 0u; \
        uint _slr_b6 = (6u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 6u]) : 0u; \
        uint _slr_b7 = (7u < _slr_avail) ? uint((src_buf)[uint(src_base) + uint(pos_abs) + 7u]) : 0u; \
        uint _slr_lo = _slr_b0 | (_slr_b1 << 8) | (_slr_b2 << 16) | (_slr_b3 << 24);                \
        uint _slr_hi = _slr_b4 | (_slr_b5 << 8) | (_slr_b6 << 16) | (_slr_b7 << 24);                \
        (out_word8) = uvec2(_slr_lo, _slr_hi);                                                      \
    } while (false)

// CUDA reference: src/encode/lz_chain_parser.cuh:83-267. findMatchChain:
// chain-hash match finder for the lazy parser. Port of CPU
// findMatchWithChainHasher. Called by lane 0 only.
//
// VK adaptation: macro form so the SSBO buffers (input + 3 hash tables)
// remain visible at the call site; GLSL cannot pass SSBOs as function
// arguments. Writes its (length, offset) result into the in-scope
// `out_match_length` + `out_match_offset` int identifiers.
//
// A-008 (BDA): `ht_buf` is a HashU32Ref (buffer_reference) carrying the
// raw device address of the per-chunk hash region. The 3 sub-table bases
// (first_hash_base, long_hash_base, next_hash_base) are byte-word
// offsets within that BDA region — pointer arithmetic preserved
// verbatim from the SSBO version; only the dereference primitive
// changed from `(ht_buf)[i]` to `(ht_buf).e[i]`.
#define findMatchChain(src_buf, src_base, src_size_in,                                              \
                       pos_in, recent_offset_in,                                                    \
                       ht_buf, first_hash_base, long_hash_base, next_hash_base,                     \
                       hash_bits_in, hash_mask_in,                                                  \
                       end_pos_in, lit_run_length_in,                                               \
                       out_match_length, out_match_offset)                                          \
    do {                                                                                            \
        uint _fmc_pos = uint(pos_in);                                                               \
        uint _fmc_src_size = uint(src_size_in);                                                     \
        uint _fmc_end_pos = uint(end_pos_in);                                                       \
        int  _fmc_recent_offset = int(recent_offset_in);                                            \
        uint _fmc_hash_bits = uint(hash_bits_in);                                                   \
        uint _fmc_hash_mask = uint(hash_mask_in);                                                   \
                                                                                                    \
        /* Read 4 bytes at current position */                                                      \
        uint _fmc_bytes_at_pos;                                                                     \
        _slzLoadU32(src_buf, src_base, _fmc_pos, _fmc_bytes_at_pos);                                \
                                                                                                    \
        uvec2 _fmc_at_src;                                                                          \
        _slzRead8Safe(src_buf, src_base, _fmc_pos, _fmc_src_size, _fmc_at_src);                     \
        uint _fmc_ha = hashTableA(_fmc_hash_bits, _fmc_hash_mask, _fmc_at_src);                     \
        uint _fmc_hb = hashTableB(_fmc_hash_bits, _fmc_hash_mask, _fmc_at_src);                     \
        uint _fmc_hb_tag = hashTagB(_fmc_at_src);                                                   \
                                                                                                    \
        /* (a) Recent-offset match: compare 4 bytes at src[pos] vs src[pos + recent_offset] */     \
        int _fmc_recent_match_length = 0;                                                           \
        bool _fmc_early_return = false;                                                             \
        int  _fmc_early_len = 0;                                                                    \
        {                                                                                           \
            uint _fmc_recent_ref = uint(int(_fmc_pos) + _fmc_recent_offset);                        \
            if (_fmc_recent_ref < _fmc_pos) {                                                       \
                uint _fmc_recent_word;                                                              \
                _slzLoadU32(src_buf, src_base, _fmc_recent_ref, _fmc_recent_word);                  \
                uint _fmc_xor_val = _fmc_bytes_at_pos ^ _fmc_recent_word;                           \
                if (_fmc_xor_val == 0u) {                                                           \
                    /* Full 4-byte match — extend forward */                                        \
                    uint _fmc_ml = 4u;                                                              \
                    uint _fmc_max_ext = _fmc_end_pos - _fmc_pos;                                    \
                    while (_fmc_ml < _fmc_max_ext &&                                                \
                           (src_buf)[uint(src_base) + _fmc_pos + _fmc_ml] ==                        \
                           (src_buf)[uint(src_base) + _fmc_recent_ref + _fmc_ml]) {                 \
                        _fmc_ml += 1u;                                                              \
                    }                                                                               \
                                                                                                    \
                    /* Insert current position into hash tables. */                                 \
                    uint _fmc_prev_head = (ht_buf).e[uint(first_hash_base) + _fmc_ha];                \
                    chainNextHashWrite(ht_buf, next_hash_base,                                      \
                                       _fmc_pos & NEXT_HASH_INDEX_MASK,                             \
                                       _fmc_prev_head & NEXT_HASH_INDEX_MASK);                      \
                    (ht_buf).e[uint(first_hash_base) + _fmc_ha] = _fmc_pos;                           \
                    (ht_buf).e[uint(long_hash_base) + _fmc_hb] =                                      \
                        (_fmc_hb_tag & LONG_HASH_TAG_MASK) |                                        \
                        (_fmc_pos << LONG_HASH_TAG_BITS);                                           \
                                                                                                    \
                    _fmc_early_return = true;                                                       \
                    _fmc_early_len = int(_fmc_ml);                                                  \
                } else {                                                                            \
                    /* Partial recent match: __ffs(x)-1 / 8 → findLSB(x)/8 */                       \
                    _fmc_recent_match_length = int(findLSB(_fmc_xor_val)) / int(BITS_PER_BYTE);     \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
                                                                                                    \
        if (_fmc_early_return) {                                                                    \
            (out_match_length) = _fmc_early_len;                                                    \
            (out_match_offset) = 0;                                                                 \
        } else {                                                                                    \
            /* Min match length bump for long literal runs */                                       \
            int _fmc_minimum_match_length = int(MIN_MATCH);                                         \
            if (uint(lit_run_length_in) >= LONG_LIT_RUN_THRESHOLD) {                                \
                if (_fmc_recent_match_length < 3) _fmc_recent_match_length = 0;                     \
                _fmc_minimum_match_length += 1;                                                     \
            }                                                                                       \
                                                                                                    \
            int _fmc_best_offset = 0;                                                               \
            int _fmc_best_match_length = 0;                                                         \
                                                                                                    \
            /* (b) Walk first_hash chain (max CHAIN_MAX_STEPS steps) */                             \
            {                                                                                       \
                uint _fmc_head = (ht_buf).e[uint(first_hash_base) + _fmc_ha];                         \
                uint _fmc_candidate_offset = _fmc_pos - _fmc_head;                                  \
                                                                                                    \
                if (_fmc_candidate_offset <= NEAR_OFFSET_MAX) {                                     \
                    if (_fmc_candidate_offset != 0u) {                                              \
                        uint _fmc_chain_steps = CHAIN_MAX_STEPS;                                    \
                        uint _fmc_hash_value = _fmc_head;                                           \
                        while (_fmc_candidate_offset < LZ_BLOCK_SIZE) {                             \
                            if (_fmc_candidate_offset > MIN_HASH_MATCH_OFFSET) {                    \
                                if (_fmc_candidate_offset <= _fmc_pos) {                            \
                                    uint _fmc_ref = _fmc_pos - _fmc_candidate_offset;               \
                                    uint _fmc_ref_word;                                             \
                                    _slzLoadU32(src_buf, src_base, _fmc_ref, _fmc_ref_word);        \
                                    if (_fmc_ref_word == _fmc_bytes_at_pos) {                       \
                                        bool _fmc_quick_ok = true;                                  \
                                        if (_fmc_best_match_length >= 4) {                          \
                                            uint _fmc_tail_pos = _fmc_pos + uint(_fmc_best_match_length); \
                                            if (_fmc_tail_pos >= _fmc_end_pos) {                    \
                                                _fmc_quick_ok = false;                              \
                                            } else {                                                \
                                                if ((src_buf)[uint(src_base) + _fmc_tail_pos] !=    \
                                                    (src_buf)[uint(src_base) + _fmc_tail_pos - _fmc_candidate_offset]) \
                                                    _fmc_quick_ok = false;                          \
                                            }                                                       \
                                        }                                                           \
                                        if (_fmc_quick_ok) {                                        \
                                            uint _fmc_ml2 = 4u;                                     \
                                            uint _fmc_max_ext2 = _fmc_end_pos - _fmc_pos;           \
                                            while (_fmc_ml2 < _fmc_max_ext2 &&                      \
                                                   (src_buf)[uint(src_base) + _fmc_pos + _fmc_ml2] == \
                                                   (src_buf)[uint(src_base) + _fmc_ref + _fmc_ml2]) { \
                                                _fmc_ml2 += 1u;                                     \
                                            }                                                       \
                                            int _fmc_cand_len = int(_fmc_ml2);                      \
                                            if (_fmc_cand_len > _fmc_best_match_length &&           \
                                                _fmc_cand_len >= _fmc_minimum_match_length) {       \
                                                _fmc_best_match_length = _fmc_cand_len;             \
                                                _fmc_best_offset = int(_fmc_candidate_offset);      \
                                            }                                                       \
                                        }                                                           \
                                    }                                                               \
                                }                                                                   \
                                _fmc_chain_steps -= 1u;                                             \
                                if (_fmc_chain_steps == 0u) break;                                  \
                            }                                                                       \
                            uint _fmc_previous_offset = _fmc_candidate_offset;                      \
                            _fmc_hash_value = chainNextHashRead(ht_buf, next_hash_base,             \
                                                                _fmc_hash_value & NEXT_HASH_INDEX_MASK); \
                            /* Chain positions are modulo 64K */                                    \
                            _fmc_candidate_offset = (_fmc_pos - _fmc_hash_value) & 0xFFFFu;         \
                            if (_fmc_candidate_offset <= _fmc_previous_offset) break;               \
                        }                                                                           \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            /* (c) Check long_hash secondary table */                                               \
            {                                                                                       \
                uint _fmc_lh_value = (ht_buf).e[uint(long_hash_base) + _fmc_hb];                      \
                if (((_fmc_hb_tag ^ _fmc_lh_value) & LONG_HASH_TAG_MASK) == 0u) {                   \
                    uint _fmc_cand_pos = _fmc_lh_value >> LONG_HASH_TAG_BITS;                       \
                    if (_fmc_cand_pos < _fmc_pos) {                                                 \
                        uint _fmc_cand_off = _fmc_pos - _fmc_cand_pos;                              \
                        if (_fmc_cand_off >= MIN_HASH_MATCH_OFFSET && _fmc_cand_off <= NEAR_OFFSET_MAX) { \
                            uint _fmc_ref3 = _fmc_pos - _fmc_cand_off;                              \
                            uint _fmc_ref_word3;                                                    \
                            _slzLoadU32(src_buf, src_base, _fmc_ref3, _fmc_ref_word3);              \
                            if (_fmc_ref_word3 == _fmc_bytes_at_pos) {                              \
                                bool _fmc_quick_ok3 = true;                                         \
                                if (_fmc_best_match_length >= 4) {                                  \
                                    uint _fmc_tail_pos3 = _fmc_pos + uint(_fmc_best_match_length);  \
                                    if (_fmc_tail_pos3 >= _fmc_end_pos)                             \
                                        _fmc_quick_ok3 = false;                                     \
                                    else if ((src_buf)[uint(src_base) + _fmc_tail_pos3] !=          \
                                             (src_buf)[uint(src_base) + _fmc_tail_pos3 - _fmc_cand_off]) \
                                        _fmc_quick_ok3 = false;                                     \
                                }                                                                   \
                                if (_fmc_quick_ok3) {                                               \
                                    uint _fmc_ml3 = 4u;                                             \
                                    uint _fmc_max_ext3 = _fmc_end_pos - _fmc_pos;                   \
                                    while (_fmc_ml3 < _fmc_max_ext3 &&                              \
                                           (src_buf)[uint(src_base) + _fmc_pos + _fmc_ml3] ==       \
                                           (src_buf)[uint(src_base) + _fmc_ref3 + _fmc_ml3]) {      \
                                        _fmc_ml3 += 1u;                                             \
                                    }                                                               \
                                    int _fmc_cand_len3 = int(_fmc_ml3);                             \
                                    if (_fmc_cand_len3 >= _fmc_minimum_match_length &&              \
                                        isMatchBetter(_fmc_cand_len3, int(_fmc_cand_off),           \
                                                      _fmc_best_match_length, _fmc_best_offset)) { \
                                        _fmc_best_match_length = _fmc_cand_len3;                    \
                                        _fmc_best_offset = int(_fmc_cand_off);                      \
                                    }                                                               \
                                }                                                                   \
                            }                                                                       \
                        }                                                                           \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            /* (d) Fixed offset-8 fallback */                                                       \
            if (_fmc_pos >= MIN_HASH_MATCH_OFFSET) {                                                \
                uint _fmc_ref4 = _fmc_pos - MIN_HASH_MATCH_OFFSET;                                  \
                uint _fmc_ref_word4;                                                                \
                _slzLoadU32(src_buf, src_base, _fmc_ref4, _fmc_ref_word4);                          \
                if (_fmc_ref_word4 == _fmc_bytes_at_pos) {                                          \
                    uint _fmc_ml4 = 4u;                                                             \
                    uint _fmc_max_ext4 = _fmc_end_pos - _fmc_pos;                                   \
                    while (_fmc_ml4 < _fmc_max_ext4 &&                                              \
                           (src_buf)[uint(src_base) + _fmc_pos + _fmc_ml4] ==                       \
                           (src_buf)[uint(src_base) + _fmc_ref4 + _fmc_ml4]) {                      \
                        _fmc_ml4 += 1u;                                                             \
                    }                                                                               \
                    int _fmc_cand_len4 = int(_fmc_ml4);                                             \
                    if (_fmc_cand_len4 >= _fmc_best_match_length &&                                 \
                        _fmc_cand_len4 >= _fmc_minimum_match_length) {                              \
                        _fmc_best_match_length = _fmc_cand_len4;                                    \
                        _fmc_best_offset = int(MIN_HASH_MATCH_OFFSET);                              \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            /* (e) Insert current position into all three hash tables */                            \
            {                                                                                       \
                uint _fmc_prev_head2 = (ht_buf).e[uint(first_hash_base) + _fmc_ha];                   \
                chainNextHashWrite(ht_buf, next_hash_base,                                          \
                                   _fmc_pos & NEXT_HASH_INDEX_MASK,                                 \
                                   _fmc_prev_head2 & NEXT_HASH_INDEX_MASK);                         \
                (ht_buf).e[uint(first_hash_base) + _fmc_ha] = _fmc_pos;                               \
                (ht_buf).e[uint(long_hash_base) + _fmc_hb] =                                          \
                    (_fmc_hb_tag & LONG_HASH_TAG_MASK) |                                            \
                    (_fmc_pos << LONG_HASH_TAG_BITS);                                               \
            }                                                                                       \
                                                                                                    \
            /* (f) Return best match vs recent match (prefer recent if close) */                    \
            if (_fmc_best_offset == 0 ||                                                            \
                !isBetterThanRecentMatch(_fmc_recent_match_length,                                  \
                                          _fmc_best_match_length, _fmc_best_offset)) {              \
                (out_match_length) = _fmc_recent_match_length;                                      \
                (out_match_offset) = 0;                                                             \
            } else {                                                                                \
                (out_match_length) = _fmc_best_match_length;                                        \
                (out_match_offset) = _fmc_best_offset;                                              \
            }                                                                                       \
        }                                                                                           \
    } while (false)

// CUDA reference: src/encode/lz_chain_parser.cuh:274-302. insertChainRange:
// insert positions [from..to) into the chain hash tables. Used after
// emitting a match to populate hash entries for the matched byte range.
//
// VK adaptation: macro form so the input + hash-table SSBOs stay visible.
#define insertChainRange(src_buf, src_base, src_size_in,                                            \
                         from_in, to_in,                                                            \
                         ht_buf, first_hash_base, long_hash_base, next_hash_base,                   \
                         hash_bits_in, hash_mask_in)                                                \
    do {                                                                                            \
        uint _icr_from = uint(from_in);                                                             \
        uint _icr_to   = uint(to_in);                                                               \
        uint _icr_src_size = uint(src_size_in);                                                     \
        uint _icr_hash_bits = uint(hash_bits_in);                                                   \
        uint _icr_hash_mask = uint(hash_mask_in);                                                   \
                                                                                                    \
        /* longHash at exponentially spaced positions: i = 0, 1, 3, 7, 15, ... */                   \
        {                                                                                           \
            uint _icr_len = _icr_to - _icr_from;                                                    \
            uint _icr_i = 0u;                                                                       \
            while (_icr_i < _icr_len) {                                                             \
                uint _icr_p = _icr_from + _icr_i;                                                   \
                uvec2 _icr_at;                                                                      \
                _slzRead8Safe(src_buf, src_base, _icr_p, _icr_src_size, _icr_at);                   \
                uint _icr_hb = hashTableB(_icr_hash_bits, _icr_hash_mask, _icr_at);                 \
                uint _icr_hb_tag = hashTagB(_icr_at);                                               \
                (ht_buf).e[uint(long_hash_base) + _icr_hb] =                                          \
                    (_icr_hb_tag & LONG_HASH_TAG_MASK) |                                            \
                    (_icr_p << LONG_HASH_TAG_BITS);                                                 \
                _icr_i = 2u * _icr_i + 1u;                                                          \
            }                                                                                       \
        }                                                                                           \
                                                                                                    \
        /* firstHash + nextHash at every position */                                                \
        for (uint _icr_p = _icr_from; _icr_p < _icr_to; _icr_p += 1u) {                             \
            uvec2 _icr_at2;                                                                         \
            _slzRead8Safe(src_buf, src_base, _icr_p, _icr_src_size, _icr_at2);                      \
            uint _icr_ha = hashTableA(_icr_hash_bits, _icr_hash_mask, _icr_at2);                    \
            uint _icr_prev_head = (ht_buf).e[uint(first_hash_base) + _icr_ha];                        \
            chainNextHashWrite(ht_buf, next_hash_base,                                              \
                               _icr_p & NEXT_HASH_INDEX_MASK,                                       \
                               _icr_prev_head & NEXT_HASH_INDEX_MASK);                              \
            (ht_buf).e[uint(first_hash_base) + _icr_ha] = _icr_p;                                     \
        }                                                                                           \
    } while (false)

// CUDA reference: src/encode/lz_chain_parser.cuh:312-440. scanBlockChain:
// serial chain-hash lazy parser for one block; lane 0 does all work.
// VK adaptation: macro form so the SSBO buffers (input + 3 hash tables +
// output substreams) remain visible at the call site; GLSL cannot pass
// SSBOs as function arguments.
#define scanBlockChain(src_buf, src_base, src_size_in,                                              \
                       ht_buf, first_hash_base, long_hash_base, next_hash_base,                     \
                       hash_bits_in, hash_mask_in,                                                  \
                       lit_buf, lit_base, lit_count,                                                \
                       token_buf, token_base, token_count,                                          \
                       off16_buf, off16_base, off16_count,                                          \
                       off32_buf, off32_base, off32_pos, off32_count,                               \
                       len_buf, len_base, len_count,                                                \
                       anchor_inout, recent_offset_inout,                                           \
                       start_pos_in, end_pos_in, block2_start_in,                                   \
                       lane_in)                                                                     \
    do {                                                                                            \
        if (uint(lane_in) == 0u) {                                                                  \
            uint _sbc_src_size = uint(src_size_in);                                                 \
            uint _sbc_end_pos = uint(end_pos_in);                                                   \
            uint _sbc_pos = uint(start_pos_in);                                                     \
                                                                                                    \
            /* Guard: need at least CHAIN_MIN_LOOKAHEAD bytes of lookahead */                       \
            if (_sbc_pos + CHAIN_MIN_LOOKAHEAD >= _sbc_end_pos) {                                   \
                /* Handle trailing literals and return */                                           \
                uint _sbc_trailing0 = _sbc_end_pos - uint(anchor_inout);                            \
                if (_sbc_trailing0 > 0u) {                                                          \
                    for (uint _sbc_i = 0u; _sbc_i < _sbc_trailing0; _sbc_i += 1u) {                 \
                        (lit_buf)[uint(lit_base) + (lit_count) + _sbc_i] =                          \
                            (src_buf)[uint(src_base) + uint(anchor_inout) + _sbc_i];                \
                    }                                                                               \
                    (lit_count) += _sbc_trailing0;                                                  \
                    emitCmd(token_buf, token_base, token_count,                                     \
                            off16_buf, off16_base, off16_count,                                     \
                            off32_buf, off32_base, off32_pos, off32_count,                          \
                            len_buf, len_base, len_count,                                           \
                            _sbc_trailing0, 0u, 0u,                                                 \
                            uint(anchor_inout), block2_start_in, int(recent_offset_inout));         \
                    (anchor_inout) = int(_sbc_end_pos);                                             \
                }                                                                                   \
            } else {                                                                                \
                while (_sbc_pos + CHAIN_MIN_LOOKAHEAD < _sbc_end_pos) {                             \
                    /* Find a match at current position */                                          \
                    uint _sbc_cur_lit_run = _sbc_pos - uint(anchor_inout);                          \
                    int _sbc_match_len;                                                             \
                    int _sbc_match_off;                                                             \
                    findMatchChain(src_buf, src_base, _sbc_src_size,                                \
                                   _sbc_pos, recent_offset_inout,                                   \
                                   ht_buf, first_hash_base, long_hash_base, next_hash_base,         \
                                   hash_bits_in, hash_mask_in,                                      \
                                   _sbc_end_pos, _sbc_cur_lit_run,                                  \
                                   _sbc_match_len, _sbc_match_off);                                 \
                    if (_sbc_match_len < 2) {                                                       \
                        _sbc_pos += 1u;                                                             \
                        continue;                                                                   \
                    }                                                                               \
                                                                                                    \
                    /* Lazy-1: check pos+1 for a better match */                                    \
                    while (_sbc_pos + CHAIN_MIN_LOOKAHEAD + 1u < _sbc_end_pos) {                    \
                        int _sbc_lazy1_len;                                                         \
                        int _sbc_lazy1_off;                                                         \
                        findMatchChain(src_buf, src_base, _sbc_src_size,                            \
                                       _sbc_pos + 1u, recent_offset_inout,                          \
                                       ht_buf, first_hash_base, long_hash_base, next_hash_base,     \
                                       hash_bits_in, hash_mask_in,                                  \
                                       _sbc_end_pos, _sbc_cur_lit_run + 1u,                         \
                                       _sbc_lazy1_len, _sbc_lazy1_off);                             \
                        if (_sbc_lazy1_len >= 2 &&                                                  \
                            isLazyMatchBetter(_sbc_lazy1_len, _sbc_lazy1_off,                       \
                                              _sbc_match_len, _sbc_match_off, 0)) {                 \
                            _sbc_pos += 1u;                                                         \
                            _sbc_match_len = _sbc_lazy1_len;                                        \
                            _sbc_match_off = _sbc_lazy1_off;                                        \
                        } else {                                                                    \
                            /* Lazy-2: check pos+2 (only if current match length > 2) */            \
                            if (_sbc_pos + CHAIN_MIN_LOOKAHEAD + 2u >= _sbc_end_pos ||              \
                                _sbc_match_len == 2) break;                                         \
                            int _sbc_lazy2_len;                                                     \
                            int _sbc_lazy2_off;                                                     \
                            findMatchChain(src_buf, src_base, _sbc_src_size,                        \
                                           _sbc_pos + 2u, recent_offset_inout,                      \
                                           ht_buf, first_hash_base, long_hash_base, next_hash_base, \
                                           hash_bits_in, hash_mask_in,                              \
                                           _sbc_end_pos, _sbc_cur_lit_run + 2u,                     \
                                           _sbc_lazy2_len, _sbc_lazy2_off);                         \
                            if (_sbc_lazy2_len >= 2 &&                                              \
                                isLazyMatchBetter(_sbc_lazy2_len, _sbc_lazy2_off,                   \
                                                  _sbc_match_len, _sbc_match_off, 1)) {             \
                                _sbc_pos += 2u;                                                     \
                                _sbc_match_len = _sbc_lazy2_len;                                    \
                                _sbc_match_off = _sbc_lazy2_off;                                    \
                            } else {                                                                \
                                break;                                                              \
                            }                                                                       \
                        }                                                                           \
                    }                                                                               \
                                                                                                    \
                    /* Resolve actual offset for backward extension */                              \
                    uint _sbc_actual_offset;                                                        \
                    if (_sbc_match_off == 0)                                                        \
                        _sbc_actual_offset = uint(-int(recent_offset_inout));                       \
                    else                                                                            \
                        _sbc_actual_offset = uint(_sbc_match_off);                                  \
                                                                                                    \
                    /* Enforce minimum match length for far offsets */                              \
                    uint _sbc_off_param = uint(_sbc_match_off);                                     \
                    uint _sbc_resolved_off = _sbc_off_param;                                        \
                    if (_sbc_resolved_off == 0u)                                                    \
                        _sbc_resolved_off = uint(-int(recent_offset_inout));                        \
                    if (_sbc_resolved_off > NEAR_OFFSET_MAX &&                                      \
                        uint(_sbc_match_len) < FAR_OFFSET_MIN_MATCH) {                              \
                        _sbc_pos += 1u;                                                             \
                        continue;                                                                   \
                    }                                                                               \
                                                                                                    \
                    /* Backward extension: extend match backward into literal run */                \
                    while (_sbc_pos > uint(anchor_inout) && _sbc_pos > _sbc_actual_offset) {        \
                        uint _sbc_prev = _sbc_pos - 1u;                                             \
                        uint _sbc_back = _sbc_prev - _sbc_actual_offset;                            \
                        if ((src_buf)[uint(src_base) + _sbc_prev] !=                                \
                            (src_buf)[uint(src_base) + _sbc_back]) break;                           \
                        _sbc_pos -= 1u;                                                             \
                        _sbc_match_len += 1;                                                        \
                    }                                                                               \
                                                                                                    \
                    /* Compute literal run and emit with literal-1 splitting */                     \
                    uint _sbc_lit_len = _sbc_pos - uint(anchor_inout);                              \
                    emitWithLiteral1(src_buf, src_base,                                             \
                                     lit_buf, lit_base, lit_count,                                  \
                                     token_buf, token_base, token_count,                            \
                                     off16_buf, off16_base, off16_count,                            \
                                     off32_buf, off32_base, off32_pos, off32_count,                 \
                                     len_buf, len_base, len_count,                                  \
                                     uint(anchor_inout), _sbc_lit_len,                              \
                                     uint(_sbc_match_len), _sbc_off_param,                          \
                                     block2_start_in, int(recent_offset_inout));                    \
                                                                                                    \
                    /* Update recent offset */                                                      \
                    (recent_offset_inout) = -int(_sbc_actual_offset);                               \
                                                                                                    \
                    /* Advance past the match */                                                    \
                    uint _sbc_match_end = _sbc_pos + uint(_sbc_match_len);                          \
                                                                                                    \
                    /* Insert matched range positions into hash tables */                           \
                    if (_sbc_match_end > _sbc_pos + 1u && _sbc_match_end + 8u <= _sbc_src_size) {   \
                        insertChainRange(src_buf, src_base, _sbc_src_size,                          \
                                         _sbc_pos + 1u, _sbc_match_end,                             \
                                         ht_buf, first_hash_base, long_hash_base, next_hash_base,   \
                                         hash_bits_in, hash_mask_in);                               \
                    }                                                                               \
                                                                                                    \
                    (anchor_inout) = int(_sbc_match_end);                                           \
                    _sbc_pos = _sbc_match_end;                                                      \
                }                                                                                   \
                                                                                                    \
                /* Trailing literals at end_pos */                                                  \
                {                                                                                   \
                    uint _sbc_trailing = _sbc_end_pos - uint(anchor_inout);                         \
                    if (_sbc_trailing > 0u) {                                                       \
                        for (uint _sbc_i = 0u; _sbc_i < _sbc_trailing; _sbc_i += 1u) {              \
                            (lit_buf)[uint(lit_base) + (lit_count) + _sbc_i] =                      \
                                (src_buf)[uint(src_base) + uint(anchor_inout) + _sbc_i];            \
                        }                                                                           \
                        (lit_count) += _sbc_trailing;                                               \
                                                                                                    \
                        emitCmd(token_buf, token_base, token_count,                                 \
                                off16_buf, off16_base, off16_count,                                 \
                                off32_buf, off32_base, off32_pos, off32_count,                      \
                                len_buf, len_base, len_count,                                       \
                                _sbc_trailing, 0u, 0u,                                              \
                                uint(anchor_inout), block2_start_in, int(recent_offset_inout));     \
                                                                                                    \
                        (anchor_inout) = int(_sbc_end_pos);                                         \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    } while (false)

#endif
