// 1:1 port of src/encode/lz_token_emit.cuh.
// Output-stream serializer used by both parsers — turns (literal-run,
// match) pairs into the codec's lit / token / off16 / off32 / len
// sub-streams. The CUDA file passes an `OutputStreams&` struct of raw
// `uint8_t*` pointers; GLSL has no SSBO-pointer parameters, so this file
// expresses the same logic as function-like macros that take the SSBO
// names + cursor identifiers in scope as plain identifiers. The
// arithmetic and emit sequence are byte-identical to CUDA.
//

#ifndef SRCVK_ENCODE_LZ_TOKEN_EMIT_GLSL
#define SRCVK_ENCODE_LZ_TOKEN_EMIT_GLSL

#include "lz_format.glsl"

// CUDA reference: src/encode/lz_token_emit.cuh:13-19. Token bit-layout
// constants.
const uint MAX_INLINE_LITERALS = 7u;
const uint MAX_INLINE_MATCH    = 15u;
const uint TOKEN_MATCH_SHIFT_MUL = 8u;
const uint TOKEN_RECENT_FLAG   = 0x80u;
const uint LITERAL_RUN_LENGTH_THRESHOLD = 64u;
const uint NEAR_CONT_MATCH_MAX = 90u;
const uint LITERAL_SLOT_MODULUS = 7u;

// CUDA reference: src/encode/lz_token_emit.cuh:22-27. Offset-class
// (Step 4 far/long path) token constants.
const int  OFF32_MATCH_LEN_BIAS  = 5;
const int  OFF32_INLINE_DELTA_MAX = 23;
const int  OFF32_FAR_LEN_BIAS = OFF32_MATCH_LEN_BIAS + OFF32_INLINE_DELTA_MAX + 1;
const int  OFF32_NEAR_LEN_BIAS = int(NEAR_CONT_MATCH_MAX) + 1;

// CUDA reference: src/encode/lz_token_emit.cuh:30-32. emitWithLiteral1
// scan window.
const uint LIT1_SCAN_MIN = 8u;
const uint LIT1_SCAN_MAX = 63u;
const uint LIT1_MAX_MATCHES = 32u;

// CUDA reference: src/encode/lz_token_emit.cuh:56-73. writeLengthValue:
// append a length to the len_buf. Values <= LENGTH_INLINE_MAX take one
// byte; larger values take a 1-byte tag plus a 2-byte LE remainder.
//
// VK adaptation: macro form so the SSBO + cursor are in scope. `len_buf`
// is the std430 uint8_t[] SSBO name; `len_base` is the byte offset where
// this sub-chunk's length stream starts; `len_count` is the in-scope
// uint cursor (mutated in place exactly like CUDA's `&len_count`).
#define writeLengthValue(len_buf, len_base, len_count, value)                                      \
    do {                                                                                            \
        uint _wlv_v = uint(value);                                                                  \
        if (_wlv_v <= LENGTH_INLINE_MAX) {                                                          \
            (len_buf)[uint(len_base) + (len_count)] = uint8_t(_wlv_v);                              \
            (len_count) += 1u;                                                                       \
        } else {                                                                                    \
            uint _wlv_low2 = _wlv_v & 3u;                                                            \
            uint _wlv_tag  = (_wlv_low2 - LENGTH_EXT_TAG_BIAS) & 0xFFu;                              \
            (len_buf)[uint(len_base) + (len_count)] = uint8_t(_wlv_tag);                            \
            (len_count) += 1u;                                                                       \
            uint _wlv_rem = (_wlv_v - (_wlv_low2 + (LENGTH_INLINE_MAX + 1u))) >> 2;                  \
            uint _wlv_b0; uint _wlv_b1;                                                              \
            storeU16LE(_wlv_b0, _wlv_b1, _wlv_rem);                                                  \
            (len_buf)[uint(len_base) + (len_count) + 0u] = uint8_t(_wlv_b0);                         \
            (len_buf)[uint(len_base) + (len_count) + 1u] = uint8_t(_wlv_b1);                         \
            (len_count) += 2u;                                                                       \
        }                                                                                            \
    } while (false)

// CUDA reference: src/encode/lz_token_emit.cuh:80-90. writeOffset32:
// append a far offset to the off32 stream. Offsets below
// LARGE_OFFSET_THRESHOLD take 3 bytes; larger ones take 4 bytes via the
// OFF32_LARGE_TAG-marked extended form.
//
// VK adaptation: see writeLengthValue. `off32_pos` is the in-scope uint
// cursor.
#define writeOffset32(off32_buf, off32_base, off32_pos, offset)                                    \
    do {                                                                                            \
        uint _wo32_off = uint(offset);                                                              \
        if (_wo32_off >= LARGE_OFFSET_THRESHOLD) {                                                  \
            uint _wo32_trunc = (_wo32_off & OFF32_LOW22_MASK) | OFF32_LARGE_TAG;                    \
            uint _wo32_b0; uint _wo32_b1; uint _wo32_b2;                                            \
            writeLE24(_wo32_b0, _wo32_b1, _wo32_b2, _wo32_trunc);                                   \
            (off32_buf)[uint(off32_base) + (off32_pos) + 0u] = uint8_t(_wo32_b0);                   \
            (off32_buf)[uint(off32_base) + (off32_pos) + 1u] = uint8_t(_wo32_b1);                   \
            (off32_buf)[uint(off32_base) + (off32_pos) + 2u] = uint8_t(_wo32_b2);                   \
            (off32_pos) += 3u;                                                                       \
            (off32_buf)[uint(off32_base) + (off32_pos)] =                                            \
                uint8_t((_wo32_off - _wo32_trunc) >> OFF32_LOW_BITS);                                \
            (off32_pos) += 1u;                                                                       \
        } else {                                                                                    \
            uint _wo32_b0; uint _wo32_b1; uint _wo32_b2;                                             \
            writeLE24(_wo32_b0, _wo32_b1, _wo32_b2, _wo32_off);                                      \
            (off32_buf)[uint(off32_base) + (off32_pos) + 0u] = uint8_t(_wo32_b0);                    \
            (off32_buf)[uint(off32_base) + (off32_pos) + 1u] = uint8_t(_wo32_b1);                    \
            (off32_buf)[uint(off32_base) + (off32_pos) + 2u] = uint8_t(_wo32_b2);                    \
            (off32_pos) += 3u;                                                                        \
        }                                                                                            \
    } while (false)

// Helper: storeU16 into an off16 SSBO at `entry_idx * 2 + base`.
#define _slzStoreOff16(off16_buf, off16_base, off16_count_in, value)                                \
    do {                                                                                            \
        uint _so16_b0; uint _so16_b1;                                                               \
        storeU16LE(_so16_b0, _so16_b1, uint(value));                                                \
        (off16_buf)[uint(off16_base) + (off16_count_in) * 2u + 0u] = uint8_t(_so16_b0);             \
        (off16_buf)[uint(off16_base) + (off16_count_in) * 2u + 1u] = uint8_t(_so16_b1);             \
    } while (false)

// CUDA reference: src/encode/lz_token_emit.cuh:103-205. emitCmd: emit
// tokens — 1:1 port of CPU writeComplexOffset. Flow:
//   Step 1: fast path (lit<=7, match<=15, off<=NEAR_OFFSET_MAX)
//   Step 2: literal encoding (continuation or 0x00 + len_buf)
//   Step 3: near-offset continuation
//   Step 4: long-match / far-offset
//
// VK adaptation: macro form — every SSBO and every cursor identifier is
// taken by name + base from the caller's scope. The struct
// `OutputStreams &s` in the CUDA source is dissolved into the explicit
// (buf, base, count) triples below; semantics are byte-identical.
#define emitCmd(token_buf, token_base, token_count,                                                 \
                off16_buf, off16_base, off16_count,                                                 \
                off32_buf, off32_base, off32_pos, off32_count,                                      \
                len_buf, len_base, len_count,                                                       \
                lit_len_in, match_len_in, offset_in,                                                \
                anchor_pos, block2_start_in, recent_offset_in)                                      \
    do {                                                                                            \
        uint _ec_remaining_lit = uint(lit_len_in);                                                  \
        uint _ec_match_pos     = uint(anchor_pos) + uint(lit_len_in);                               \
        uint _ec_match_len     = uint(match_len_in);                                                \
        uint _ec_offset        = uint(offset_in);                                                   \
                                                                                                    \
        /* Step 1: Fast path */                                                                     \
        bool _ec_step1 = (_ec_remaining_lit <= MAX_INLINE_LITERALS)                                 \
                       && (_ec_match_len <= MAX_INLINE_MATCH)                                       \
                       && (_ec_offset <= NEAR_OFFSET_MAX);                                          \
        if (_ec_step1) {                                                                            \
            uint _ec_tok = _ec_remaining_lit + TOKEN_MATCH_SHIFT_MUL * _ec_match_len;               \
            if (_ec_offset == 0u) _ec_tok = _ec_tok | TOKEN_RECENT_FLAG;                            \
            (token_buf)[uint(token_base) + (token_count)] = uint8_t(_ec_tok);                       \
            (token_count) += 1u;                                                                     \
            if (_ec_offset != 0u) {                                                                 \
                _slzStoreOff16(off16_buf, off16_base, off16_count, _ec_offset);                     \
                (off16_count) += 1u;                                                                 \
            }                                                                                       \
        } else {                                                                                    \
            /* Step 2: Literal encoding */                                                          \
            bool _ec_match_done = false;                                                            \
            if (_ec_remaining_lit < LITERAL_RUN_LENGTH_THRESHOLD) {                                 \
                while (_ec_remaining_lit > MAX_INLINE_LITERALS) {                                   \
                    (token_buf)[uint(token_base) + (token_count)] =                                 \
                        uint8_t(TOKEN_RECENT_FLAG | MAX_INLINE_LITERALS);                           \
                    (token_count) += 1u;                                                             \
                    _ec_remaining_lit -= MAX_INLINE_LITERALS;                                       \
                }                                                                                   \
            } else {                                                                                \
                writeLengthValue(len_buf, len_base, len_count,                                      \
                                 _ec_remaining_lit - LITERAL_RUN_LENGTH_THRESHOLD);                 \
                (token_buf)[uint(token_base) + (token_count)] = uint8_t(0u);                        \
                (token_count) += 1u;                                                                 \
                _ec_remaining_lit = 0u;                                                             \
                if (_ec_match_len == 0u) _ec_match_done = true;                                     \
            }                                                                                       \
                                                                                                    \
            if (!_ec_match_done) {                                                                  \
                /* Step 3: Near-offset continuation */                                              \
                if ((_ec_offset <= NEAR_OFFSET_MAX) && (_ec_match_len <= NEAR_CONT_MATCH_MAX)) {    \
                    uint _ec_current = (_ec_match_len < MAX_INLINE_MATCH)                           \
                                     ? _ec_match_len : MAX_INLINE_MATCH;                            \
                    uint _ec_tok2 = _ec_remaining_lit + TOKEN_MATCH_SHIFT_MUL * _ec_current;        \
                    if (_ec_offset == 0u) _ec_tok2 = _ec_tok2 | TOKEN_RECENT_FLAG;                  \
                    (token_buf)[uint(token_base) + (token_count)] = uint8_t(_ec_tok2);              \
                    (token_count) += 1u;                                                             \
                    if (_ec_offset != 0u) {                                                         \
                        _slzStoreOff16(off16_buf, off16_base, off16_count, _ec_offset);             \
                        (off16_count) += 1u;                                                         \
                    }                                                                               \
                    uint _ec_remaining_match = _ec_match_len - _ec_current;                         \
                    while (_ec_remaining_match > 0u) {                                              \
                        _ec_current = (_ec_remaining_match < MAX_INLINE_MATCH)                      \
                                    ? _ec_remaining_match : MAX_INLINE_MATCH;                       \
                        (token_buf)[uint(token_base) + (token_count)] =                             \
                            uint8_t(TOKEN_RECENT_FLAG + TOKEN_MATCH_SHIFT_MUL * _ec_current);       \
                        (token_count) += 1u;                                                         \
                        _ec_remaining_match -= _ec_current;                                         \
                    }                                                                               \
                } else {                                                                            \
                    /* Step 4: Long-match / far-offset path */                                      \
                    if (_ec_remaining_lit != 0u) {                                                  \
                        (token_buf)[uint(token_base) + (token_count)] =                             \
                            uint8_t(TOKEN_RECENT_FLAG + _ec_remaining_lit);                         \
                        (token_count) += 1u;                                                         \
                    }                                                                               \
                                                                                                    \
                    uint _ec_eff_off = _ec_offset;                                                  \
                    if (_ec_eff_off == 0u) {                                                        \
                        _ec_eff_off = uint(-int(recent_offset_in));                                 \
                    }                                                                               \
                                                                                                    \
                    uint _ec_octok = 0u;                                                            \
                    bool _ec_write_length = false;                                                  \
                    int  _ec_lv = 0;                                                                \
                                                                                                    \
                    if (_ec_eff_off > NEAR_OFFSET_MAX) {                                            \
                        int _ec_delta = int(_ec_match_len) - OFF32_MATCH_LEN_BIAS;                  \
                        if (_ec_delta >= 0 && _ec_delta <= OFF32_INLINE_DELTA_MAX) {                \
                            _ec_octok = uint(int(_ec_match_len) - OFF32_MATCH_LEN_BIAS);            \
                        } else {                                                                    \
                            _ec_octok = 2u;                                                         \
                            _ec_lv = int(_ec_match_len) - OFF32_FAR_LEN_BIAS;                       \
                            _ec_write_length = true;                                                \
                        }                                                                           \
                    } else {                                                                        \
                        _ec_octok = 1u;                                                             \
                        _ec_lv = int(_ec_match_len) - OFF32_NEAR_LEN_BIAS;                          \
                        _ec_write_length = true;                                                    \
                    }                                                                               \
                                                                                                    \
                    (token_buf)[uint(token_base) + (token_count)] = uint8_t(_ec_octok);             \
                    (token_count) += 1u;                                                             \
                    if (_ec_write_length) {                                                         \
                        uint _ec_lvu = (_ec_lv > 0) ? uint(_ec_lv) : 0u;                            \
                        writeLengthValue(len_buf, len_base, len_count, _ec_lvu);                    \
                    }                                                                               \
                                                                                                    \
                    if (_ec_eff_off > NEAR_OFFSET_MAX) {                                            \
                        uint _ec_adjusted = _ec_eff_off + uint(block2_start_in) - _ec_match_pos;    \
                        writeOffset32(off32_buf, off32_base, off32_pos, _ec_adjusted);              \
                        (off32_count) += 1u;                                                         \
                    } else {                                                                        \
                        _slzStoreOff16(off16_buf, off16_base, off16_count, _ec_eff_off);            \
                        (off16_count) += 1u;                                                         \
                    }                                                                               \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
    } while (false)

// CUDA reference: src/encode/lz_token_emit.cuh:209-213. literalRunSlotCount:
// value mod LITERAL_SLOT_MODULUS for values above the modulus.
uint literalRunSlotCount(uint value) {
    return (value > LITERAL_SLOT_MODULUS)
        ? (((value - 1u) % LITERAL_SLOT_MODULUS) + 1u)
        : value;
}

// CUDA reference: src/encode/lz_token_emit.cuh:220-301. emitWithLiteral1:
// scan literal runs in [LIT1_SCAN_MIN, LIT1_SCAN_MAX] for single-byte
// recent-offset matches and split the run to emit them as
// (lit_count, match=1, offset=0) tokens.
//
// VK adaptation: macro form. `src` is the input-bytes SSBO; the lit
// stream is `lit_buf[lit_base..]` and lit_count cursor name is in scope.
// The `found[]` scratch array is a local fixed-size uint array — GLSL
// supports local arrays so the CUDA `uint32_t found[LIT1_MAX_MATCHES+1]`
// translates verbatim.
#define emitWithLiteral1(src, src_base,                                                             \
                         lit_buf, lit_base, lit_count,                                              \
                         token_buf, token_base, token_count,                                        \
                         off16_buf, off16_base, off16_count,                                        \
                         off32_buf, off32_base, off32_pos, off32_count,                             \
                         len_buf, len_base, len_count,                                              \
                         anchor_in, lit_len_in, match_len_in, offset_in,                            \
                         block2_start_in, recent_offset_in)                                         \
    do {                                                                                            \
        uint _ewl_anchor = uint(anchor_in);                                                         \
        uint _ewl_lit_len = uint(lit_len_in);                                                       \
        uint _ewl_match_len = uint(match_len_in);                                                   \
        uint _ewl_offset = uint(offset_in);                                                         \
        int  _ewl_ro = int(recent_offset_in);                                                       \
                                                                                                    \
        if (_ewl_lit_len < LIT1_SCAN_MIN || _ewl_lit_len > LIT1_SCAN_MAX) {                         \
            for (uint _ewl_i = 0u; _ewl_i < _ewl_lit_len; _ewl_i += 1u) {                           \
                (lit_buf)[uint(lit_base) + (lit_count) + _ewl_i] =                                  \
                    (src)[uint(src_base) + _ewl_anchor + _ewl_i];                                   \
            }                                                                                       \
            (lit_count) += _ewl_lit_len;                                                            \
            emitCmd(token_buf, token_base, token_count,                                             \
                    off16_buf, off16_base, off16_count,                                             \
                    off32_buf, off32_base, off32_pos, off32_count,                                  \
                    len_buf, len_base, len_count,                                                   \
                    _ewl_lit_len, _ewl_match_len, _ewl_offset,                                      \
                    _ewl_anchor, block2_start_in, recent_offset_in);                                \
        } else {                                                                                    \
            uint _ewl_found[LIT1_MAX_MATCHES + 1u];                                                 \
            uint _ewl_found_count = 0u;                                                             \
            uint _ewl_last = 0u;                                                                    \
                                                                                                    \
            for (uint _ewl_i = 1u; _ewl_i < _ewl_lit_len; _ewl_i += 1u) {                           \
                uint _ewl_p = _ewl_anchor + _ewl_i;                                                 \
                int  _ewl_back = int(_ewl_p) + _ewl_ro;                                             \
                if (_ewl_back >= 0 &&                                                               \
                    uint((src)[uint(src_base) + _ewl_p]) ==                                         \
                    uint((src)[uint(src_base) + uint(_ewl_back)])) {                                \
                    if (_ewl_i != _ewl_last) {                                                      \
                        _ewl_found[_ewl_found_count] = _ewl_i - _ewl_last;                          \
                        _ewl_found_count += 1u;                                                     \
                        _ewl_last = _ewl_i + 1u;                                                    \
                    }                                                                               \
                    if (_ewl_found_count >= LIT1_MAX_MATCHES) break;                                \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            if (_ewl_found_count != 0u) {                                                           \
                _ewl_found[_ewl_found_count] = _ewl_lit_len - _ewl_last;                            \
                uint _ewl_cur_anchor = _ewl_anchor;                                                 \
                uint _ewl_cur_lit = _ewl_lit_len;                                                   \
                                                                                                    \
                for (uint _ewl_fi = 0u; _ewl_fi < _ewl_found_count; _ewl_fi += 1u) {                \
                    uint _ewl_current = _ewl_found[_ewl_fi];                                        \
                    uint _ewl_next = _ewl_found[_ewl_fi + 1u];                                      \
                    if (literalRunSlotCount(_ewl_current) +                                         \
                        literalRunSlotCount(_ewl_next) + 1u > MAX_INLINE_LITERALS) {                \
                        for (uint _ewl_i = 0u; _ewl_i < _ewl_current; _ewl_i += 1u) {               \
                            (lit_buf)[uint(lit_base) + (lit_count) + _ewl_i] =                      \
                                (src)[uint(src_base) + _ewl_cur_anchor + _ewl_i];                   \
                        }                                                                           \
                        (lit_count) += _ewl_current;                                                \
                        emitCmd(token_buf, token_base, token_count,                                 \
                                off16_buf, off16_base, off16_count,                                 \
                                off32_buf, off32_base, off32_pos, off32_count,                      \
                                len_buf, len_base, len_count,                                       \
                                _ewl_current, 1u, 0u,                                               \
                                _ewl_cur_anchor, block2_start_in, recent_offset_in);                \
                        _ewl_cur_anchor += _ewl_current + 1u;                                       \
                        _ewl_cur_lit -= _ewl_current + 1u;                                          \
                    } else {                                                                        \
                        _ewl_found[_ewl_fi + 1u] += _ewl_current + 1u;                              \
                    }                                                                               \
                }                                                                                   \
                                                                                                    \
                for (uint _ewl_i = 0u; _ewl_i < _ewl_cur_lit; _ewl_i += 1u) {                       \
                    (lit_buf)[uint(lit_base) + (lit_count) + _ewl_i] =                              \
                        (src)[uint(src_base) + _ewl_cur_anchor + _ewl_i];                           \
                }                                                                                   \
                (lit_count) += _ewl_cur_lit;                                                        \
                emitCmd(token_buf, token_base, token_count,                                         \
                        off16_buf, off16_base, off16_count,                                         \
                        off32_buf, off32_base, off32_pos, off32_count,                              \
                        len_buf, len_base, len_count,                                               \
                        _ewl_cur_lit, _ewl_match_len, _ewl_offset,                                  \
                        _ewl_cur_anchor, block2_start_in, recent_offset_in);                        \
            } else {                                                                                \
                for (uint _ewl_i = 0u; _ewl_i < _ewl_lit_len; _ewl_i += 1u) {                       \
                    (lit_buf)[uint(lit_base) + (lit_count) + _ewl_i] =                              \
                        (src)[uint(src_base) + _ewl_anchor + _ewl_i];                               \
                }                                                                                   \
                (lit_count) += _ewl_lit_len;                                                        \
                emitCmd(token_buf, token_base, token_count,                                         \
                        off16_buf, off16_base, off16_count,                                         \
                        off32_buf, off32_base, off32_pos, off32_count,                              \
                        len_buf, len_base, len_count,                                               \
                        _ewl_lit_len, _ewl_match_len, _ewl_offset,                                  \
                        _ewl_anchor, block2_start_in, recent_offset_in);                            \
            }                                                                                       \
        }                                                                                           \
    } while (false)

#endif
