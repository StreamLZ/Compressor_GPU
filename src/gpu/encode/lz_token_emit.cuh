// ── StreamLZ GPU LZ encode — token / offset / length serializer ──
// The output-stream serializer shared by both parsers. Turns
// (literal-run, match) pairs into the codec's lit / token / off16 /
// off32 / len sub-streams. 1:1 port of the CPU writeComplexOffset
// family.
//
// Included by lz_kernel.cu — see that file for the build line.
#pragma once

#include "lz_format.cuh"

// ── Token bit layout ────────────────────────────────────────────
static constexpr uint32_t MAX_INLINE_LITERALS = 7;   // literals foldable into one token (low 3 bits)
static constexpr uint32_t MAX_INLINE_MATCH    = 15;  // match length foldable into a token nibble
static constexpr uint32_t TOKEN_MATCH_SHIFT_MUL = 8; // match length occupies bits 3+ (× 8)
static constexpr uint32_t TOKEN_RECENT_FLAG   = 0x80;  // high bit: recent-offset / continuation token
static constexpr uint32_t LITERAL_RUN_LENGTH_THRESHOLD = 64;  // run length that spills into len_buf
static constexpr uint32_t NEAR_CONT_MATCH_MAX = 90;  // max match length for near-offset continuation
static constexpr uint32_t LITERAL_SLOT_MODULUS = 7;  // literalRunSlotCount modulus (numerically == MAX_INLINE_LITERALS, distinct concept)

// ── Offset-class token (Step 4 far/long path) ───────────────────
static constexpr int32_t  OFF32_MATCH_LEN_BIAS  = 5;   // offset-class token base bias (match_len - 5)
static constexpr int32_t  OFF32_INLINE_DELTA_MAX = 23; // max delta encodable inline in the offset-class token
// type-2 far token length bias: 29 = OFF32_MATCH_LEN_BIAS + OFF32_INLINE_DELTA_MAX + 1
static constexpr int32_t  OFF32_FAR_LEN_BIAS = OFF32_MATCH_LEN_BIAS + OFF32_INLINE_DELTA_MAX + 1;
// type-1 near-but-long token length bias: 91 = NEAR_CONT_MATCH_MAX + 1
static constexpr int32_t  OFF32_NEAR_LEN_BIAS = (int32_t)NEAR_CONT_MATCH_MAX + 1;

// ── emitWithLiteral1 scan window ────────────────────────────────
static constexpr uint32_t LIT1_SCAN_MIN = 8;    // smallest literal run scanned for recent-offset bytes
static constexpr uint32_t LIT1_SCAN_MAX = 63;   // largest literal run scanned (LITERAL_RUN_LENGTH_THRESHOLD - 1)
static constexpr uint32_t LIT1_MAX_MATCHES = 32;  // max recorded recent-offset matches per run

// ── OutputStreams ───────────────────────────────────────────────
// Bundles the five output sub-streams (literals, tokens, off16,
// off32, lengths) plus their write cursors into one value, so the
// serializer / parser signatures do not have to thread 12 separate
// arguments. Passed by reference; cursors mutate in place. Bundling
// is purely structural — semantics are identical to the previous
// flat-argument form.
struct OutputStreams {
    uint8_t* lit_buf;   uint32_t lit_count;
    uint8_t* token_buf; uint32_t token_count;
    uint8_t* off16_buf; uint32_t off16_count;
    uint8_t* off32_buf; uint32_t off32_pos;  uint32_t off32_count;
    uint8_t* len_buf;   uint32_t length_count;
};

// Store a uint16_t little-endian without alignment assumptions.
__device__ void storeU16LE(uint8_t* dst, uint16_t value) {
    memcpy(dst, &value, 2);
}

// ── writeLengthValue ────────────────────────────────────────────
// Append a length to len_buf. Values <= LENGTH_INLINE_MAX take one
// byte; larger values take a 1-byte tag plus a 2-byte remainder.
// Port of the CPU length-stream encoder.
__device__ void writeLengthValue(uint8_t* len_buf, uint32_t &len_count, uint32_t value) {
    if (value <= LENGTH_INLINE_MAX) {
        len_buf[len_count++] = (uint8_t)value;
    } else {
        uint32_t low2 = value & 3;
        uint8_t tag = (uint8_t)((low2 - LENGTH_EXT_TAG_BIAS) & 0xFF);
        len_buf[len_count++] = tag;
        uint16_t remainder = (uint16_t)((value - (low2 + (LENGTH_INLINE_MAX + 1))) >> 2);
        storeU16LE(len_buf + len_count, remainder);
        len_count += 2;
    }
}

// ── writeOffset32 ───────────────────────────────────────────────
// Append a far offset to off32_buf. Offsets below
// LARGE_OFFSET_THRESHOLD take 3 bytes; larger ones take 4 bytes via
// the OFF32_LARGE_TAG-marked extended form. Port of the CPU
// off32-stream encoder.
__device__ void writeOffset32(uint8_t* off32_buf, uint32_t &off32_pos, uint32_t offset) {
    if (offset >= LARGE_OFFSET_THRESHOLD) {
        uint32_t truncated = (offset & OFF32_LOW22_MASK) | OFF32_LARGE_TAG;
        off32_buf[off32_pos++] = (uint8_t)(truncated & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((truncated >> 8) & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((truncated >> 16) & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((offset - truncated) >> OFF32_LOW_BITS);
    } else {
        off32_buf[off32_pos++] = (uint8_t)(offset & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((offset >> 8) & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((offset >> 16) & 0xFF);
    }
}

// ── emitCmd ─────────────────────────────────────────────────────
// Emit tokens -- 1:1 port of CPU writeComplexOffset.
// Flow matches the CPU exactly:
//   1. Fast path (lit<=7, match<=15, off<=NEAR_OFFSET_MAX)
//   2. Literal encoding (0x80|7 continuation, or 0x00 + len_buf)
//   3. Near-offset continuation (off<=NEAR_OFFSET_MAX, match<=90):
//      remaining_lit folded into the first match token
//   4. Long-match / far-offset: emit remaining_lit separately, then
//      resolve the offset-class token
// anchor_pos is consumed only by the Step-4 far branch (to compute
// match_pos); it is dead weight for near-offset emits.
__device__ void emitCmd(
    OutputStreams &s,
    uint32_t lit_len,
    uint32_t match_len, uint32_t offset,
    uint32_t anchor_pos, uint32_t block2_start,
    int32_t recent_offset
) {
    uint32_t remaining_lit = lit_len;
    uint32_t match_pos = anchor_pos + lit_len;

    // Step 1: Fast path
    if (remaining_lit <= MAX_INLINE_LITERALS && match_len <= MAX_INLINE_MATCH && offset <= NEAR_OFFSET_MAX) {
        uint8_t token = (uint8_t)(remaining_lit + TOKEN_MATCH_SHIFT_MUL * match_len);
        if (offset == 0) token |= TOKEN_RECENT_FLAG;
        s.token_buf[s.token_count++] = token;
        if (offset != 0) {
            storeU16LE(s.off16_buf + s.off16_count * 2, (uint16_t)offset);
            s.off16_count++;
        }
        return;
    }

    // Step 2: Literal encoding
    if (remaining_lit < LITERAL_RUN_LENGTH_THRESHOLD) {
        while (remaining_lit > MAX_INLINE_LITERALS) {
            s.token_buf[s.token_count++] = (uint8_t)(TOKEN_RECENT_FLAG | MAX_INLINE_LITERALS);
            remaining_lit -= MAX_INLINE_LITERALS;
        }
    } else {
        writeLengthValue(s.len_buf, s.length_count, remaining_lit - LITERAL_RUN_LENGTH_THRESHOLD);
        s.token_buf[s.token_count++] = 0x00;
        remaining_lit = 0;
        if (match_len == 0) return;
    }

    // Step 3: Near-offset continuation (remaining_lit folded into first match token)
    if (offset <= NEAR_OFFSET_MAX && match_len <= NEAR_CONT_MATCH_MAX) {
        uint32_t current = (match_len < MAX_INLINE_MATCH) ? match_len : MAX_INLINE_MATCH;
        uint8_t token = (uint8_t)(remaining_lit + TOKEN_MATCH_SHIFT_MUL * current);
        if (offset == 0) token |= TOKEN_RECENT_FLAG;
        s.token_buf[s.token_count++] = token;
        if (offset != 0) {
            storeU16LE(s.off16_buf + s.off16_count * 2, (uint16_t)offset);
            s.off16_count++;
        }
        uint32_t remaining_match = match_len - current;
        while (remaining_match > 0) {
            current = (remaining_match < MAX_INLINE_MATCH) ? remaining_match : MAX_INLINE_MATCH;
            s.token_buf[s.token_count++] = (uint8_t)(TOKEN_RECENT_FLAG + TOKEN_MATCH_SHIFT_MUL * current);
            remaining_match -= current;
        }
        return;
    }

    // Step 4: Long-match / far-offset path
    if (remaining_lit != 0) {
        s.token_buf[s.token_count++] = (uint8_t)(TOKEN_RECENT_FLAG + remaining_lit);
    }

    uint32_t effective_offset = offset;
    if (effective_offset == 0) {
        effective_offset = (uint32_t)(-recent_offset);
    }

    uint8_t offset_class_token = 0;
    bool write_length = false;
    int32_t length_value = 0;

    if (effective_offset > NEAR_OFFSET_MAX) {
        int32_t delta = (int32_t)match_len - OFF32_MATCH_LEN_BIAS;
        if (delta >= 0 && delta <= OFF32_INLINE_DELTA_MAX) {
            offset_class_token = (uint8_t)(match_len - OFF32_MATCH_LEN_BIAS);
        } else {
            offset_class_token = 2;
            length_value = (int32_t)match_len - OFF32_FAR_LEN_BIAS;
            write_length = true;
        }
    } else {
        offset_class_token = 1;
        length_value = (int32_t)match_len - OFF32_NEAR_LEN_BIAS;
        write_length = true;
    }

    s.token_buf[s.token_count++] = offset_class_token;
    if (write_length) {
        uint32_t lv = (length_value > 0) ? (uint32_t)length_value : 0;
        writeLengthValue(s.len_buf, s.length_count, lv);
    }

    if (effective_offset > NEAR_OFFSET_MAX) {
        uint32_t adjusted = effective_offset + block2_start - match_pos;
        writeOffset32(s.off32_buf, s.off32_pos, adjusted);
        s.off32_count++;
    } else {
        storeU16LE(s.off16_buf + s.off16_count * 2, (uint16_t)effective_offset);
        s.off16_count++;
    }
}

// ── literalRunSlotCount ─────────────────────────────────────────
// Port of CPU fast_constants.literalRunSlotCount: value mod 7 for > 7.
__device__ uint32_t literalRunSlotCount(uint32_t value) {
    return (value > LITERAL_SLOT_MODULUS)
        ? ((value - 1) % LITERAL_SLOT_MODULUS + 1)
        : value;
}

// ── emitWithLiteral1 ────────────────────────────────────────────
// Port of CPU writeOffsetWithLiteral1. For literal runs in
// [LIT1_SCAN_MIN, LIT1_SCAN_MAX] bytes, scans for single-byte
// recent-offset matches and splits the run to emit them as
// (lit_count, match=1, offset=0) tokens.
__device__ void emitWithLiteral1(
    const uint8_t* src,
    OutputStreams &s,
    uint32_t anchor, uint32_t lit_len,
    uint32_t match_len, uint32_t offset,
    uint32_t block2_start, int32_t recent_offset
) {
    // Only scan literal runs in [LIT1_SCAN_MIN, LIT1_SCAN_MAX]
    if (lit_len < LIT1_SCAN_MIN || lit_len > LIT1_SCAN_MAX) {
        for (uint32_t i = 0; i < lit_len; i++)
            s.lit_buf[s.lit_count + i] = src[anchor + i];
        s.lit_count += lit_len;
        emitCmd(s, lit_len, match_len, offset,
                anchor, block2_start, recent_offset);
        return;
    }

    // Scan for bytes matching recent-offset pattern.
    // found[] holds up to LIT1_MAX_MATCHES gap entries plus one
    // trailing sentinel (the final remaining-literal count).
    uint32_t found[LIT1_MAX_MATCHES + 1];
    uint32_t found_count = 0;
    uint32_t last = 0;
    int32_t ro = recent_offset; // negative distance

    for (uint32_t i = 1; i < lit_len; i++) {
        uint32_t p = anchor + i;
        int32_t back = (int32_t)p + ro;
        if (back >= 0 && src[p] == src[(uint32_t)back]) {
            if (i != last) {
                // New recorded match: register the gap and advance last
                // past the matched byte. Subsequent consecutive matches
                // (i == last on next iter) are FOLDED into this entry —
                // they must NOT update `last`, otherwise the fold loop
                // below emits a recent-offset match at a non-matching
                // position. (CPU equivalent: only updates last when
                // adding to found[].)
                found[found_count] = i - last;
                found_count++;
                last = i + 1;
            }
            if (found_count >= LIT1_MAX_MATCHES) break;
        }
    }

    if (found_count != 0) {
        found[found_count] = lit_len - last;
        uint32_t cur_anchor = anchor;
        uint32_t cur_lit = lit_len;

        for (uint32_t fi = 0; fi < found_count; fi++) {
            uint32_t current = found[fi];
            uint32_t next_val = found[fi + 1];
            if (literalRunSlotCount(current) + literalRunSlotCount(next_val) + 1 > MAX_INLINE_LITERALS) {
                // Emit: `current` literals + 1 match byte at recent offset
                for (uint32_t i = 0; i < current; i++)
                    s.lit_buf[s.lit_count + i] = src[cur_anchor + i];
                s.lit_count += current;
                emitCmd(s, current, 1, 0,
                        cur_anchor, block2_start, recent_offset);
                cur_anchor += current + 1;
                cur_lit -= current + 1;
            } else {
                found[fi + 1] += current + 1;
            }
        }

        // Emit remaining literals + the actual match
        for (uint32_t i = 0; i < cur_lit; i++)
            s.lit_buf[s.lit_count + i] = src[cur_anchor + i];
        s.lit_count += cur_lit;
        emitCmd(s, cur_lit, match_len, offset,
                cur_anchor, block2_start, recent_offset);
    } else {
        // No recent-offset bytes found — emit normally
        for (uint32_t i = 0; i < lit_len; i++)
            s.lit_buf[s.lit_count + i] = src[anchor + i];
        s.lit_count += lit_len;
        emitCmd(s, lit_len, match_len, offset,
                anchor, block2_start, recent_offset);
    }
}
