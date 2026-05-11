// ── StreamLZ GPU L1 Compress Kernel ─────────────────────────────
// Warp-parallel greedy scan with shared-memory hash table.
// 32-bit hash entries support chunks up to 256KB (sc_group=1).
// 1 warp per block, 16KB hash table (4K x u32) in shared memory.
//
// Two-pass block design: the parser scans block 1 [0..64KB) and
// block 2 [64KB..src_size) separately so that literal runs and
// match extensions never cross the 64KB boundary. This matches
// the CPU encoder's architecture exactly.
//
// Two parser modes:
//   use_chain=0 (default): warp-parallel greedy scan (scanBlock)
//   use_chain=1:           serial chain-hash lazy parser (scanBlockChain)
//
// Build: nvcc -ptx -arch=sm_89 -O3 gpu_encode_kernel.cu

#include <cstdint>

static constexpr uint32_t MIN_MATCH    = 4;
static constexpr uint32_t INITIAL_COPY = 8;
static constexpr uint32_t HASH_EMPTY   = 0xFFFFFFFFu;
static constexpr uint32_t LARGE_OFFSET_THRESHOLD = 0xC00000u;
static constexpr uint32_t BLOCK1_SIZE  = 0x10000u;
static constexpr uint32_t CHAIN_MAX_STEPS = 8;
static constexpr uint32_t NEXT_HASH_SIZE = 65536;  // 2^16 entries of uint16_t (matches CPU c_bits=16)

struct CompressChunkDesc {
    uint32_t src_offset;
    uint32_t src_size;
    uint32_t dst_offset;
    uint32_t dst_capacity;
    uint32_t is_first;
};

// Match result returned by findMatchChain
struct ChainMatch {
    int32_t length;
    int32_t offset;  // 0 = recent-offset reuse, >0 = explicit distance
};

__device__ uint32_t hashKey(uint32_t key, uint32_t hash_bits, uint32_t hash_mask) {
    key *= 0x9E3779B1u;
    return (key >> (32 - hash_bits)) & hash_mask;
}

// Read up to 8 bytes at p, zero-padding if fewer than 8 remain.
__device__ uint64_t read8safe(const uint8_t* p, uint32_t pos, uint32_t src_size) {
    uint64_t v = 0;
    uint32_t avail = (pos + 8 <= src_size) ? 8 : (src_size > pos ? src_size - pos : 0);
    memcpy(&v, p, avail);
    return v;
}

// Hash-A: 8-byte key with 64-bit multiply, matching CPU MatchHasher2.
__device__ uint32_t hashKeyA8(const uint8_t* p, uint32_t hash_bits, uint32_t hash_mask, uint64_t at_src) {
    uint64_t product = (uint64_t)0xB7A5646300000000ULL * at_src;
    uint32_t hi32 = (uint32_t)(product >> 32);
    return (hi32 >> (32 - hash_bits)) & hash_mask;
}

// Hash-B: 8-byte key with Fibonacci 64-bit multiply, matching CPU.
__device__ uint32_t hashKeyB8(const uint8_t* p, uint32_t hash_bits, uint32_t hash_mask, uint64_t at_src) {
    uint64_t product = (uint64_t)0x9E3779B97F4A7C15ULL * at_src;
    uint32_t hi32 = (uint32_t)(product >> 32);
    return (hi32 >> (32 - hash_bits)) & hash_mask;
}

// Hash-B tag: full 32-bit hash value, caller uses & 0x3F.
__device__ uint32_t hashKeyBTag8(const uint8_t* p, uint32_t hash_bits, uint64_t at_src) {
    uint64_t product = (uint64_t)0x9E3779B97F4A7C15ULL * at_src;
    uint32_t hi32 = (uint32_t)(product >> 32);
    return hi32;
}

__device__ void writeLengthValue(uint8_t* len_buf, uint32_t &len_count, uint32_t value) {
    if (value <= 251) {
        len_buf[len_count++] = (uint8_t)value;
    } else {
        uint32_t low2 = value & 3;
        uint8_t tag = (uint8_t)((low2 - 4) & 0xFF);
        len_buf[len_count++] = tag;
        uint16_t remainder = (uint16_t)((value - (low2 + 252)) >> 2);
        memcpy(len_buf + len_count, &remainder, 2);
        len_count += 2;
    }
}

__device__ void writeOffset32(uint8_t* off32_buf, uint32_t &off32_pos, uint32_t offset) {
    if (offset >= LARGE_OFFSET_THRESHOLD) {
        uint32_t truncated = (offset & 0x3FFFFF) | 0xC00000;
        off32_buf[off32_pos++] = (uint8_t)(truncated & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((truncated >> 8) & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((truncated >> 16) & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((offset - truncated) >> 22);
    } else {
        off32_buf[off32_pos++] = (uint8_t)(offset & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((offset >> 8) & 0xFF);
        off32_buf[off32_pos++] = (uint8_t)((offset >> 16) & 0xFF);
    }
}

// Emit tokens -- 1:1 port of CPU writeComplexOffset.
// Flow matches the CPU exactly:
//   1. Fast path (lit<=7, match<=15, off<=0xFFFF)
//   2. Literal encoding (0x87 or 0x00)
//   3. Near-offset continuation (off<=0xFFFF, match<=90): remaining_lit in first match token
//   4. Long-match / far-offset: emit remaining_lit separately, then resolve offset
__device__ void emitCmd(
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* off32_buf, uint32_t &off32_pos, uint32_t &off32_count,
    uint8_t* len_buf, uint32_t &length_count,
    uint32_t lit_len,
    uint32_t match_len, uint32_t offset,
    uint32_t anchor_pos, uint32_t block2_start,
    int32_t recent_offset
) {
    uint32_t remaining_lit = lit_len;
    uint32_t match_pos = anchor_pos + lit_len;

    // Step 1: Fast path
    if (remaining_lit <= 7 && match_len <= 15 && offset <= 0xFFFF) {
        uint8_t token = (uint8_t)(remaining_lit + 8 * match_len);
        if (offset == 0) token |= 0x80;
        cmd_buf[token_count++] = token;
        if (offset != 0) {
            uint16_t off_val = (uint16_t)offset;
            memcpy(off16_buf + off16_count * 2, &off_val, 2);
            off16_count++;
        }
        return;
    }

    // Step 2: Literal encoding
    if (remaining_lit < 64) {
        while (remaining_lit > 7) {
            cmd_buf[token_count++] = 0x87;
            remaining_lit -= 7;
        }
    } else {
        writeLengthValue(len_buf, length_count, remaining_lit - 64);
        cmd_buf[token_count++] = 0x00;
        remaining_lit = 0;
        if (match_len == 0) return;
    }

    // Step 3: Near-offset continuation (remaining_lit folded into first match token)
    if (offset <= 0xFFFF && match_len <= 90) {
        uint32_t current = (match_len < 15) ? match_len : 15;
        uint8_t token = (uint8_t)(remaining_lit + 8 * current);
        if (offset == 0) token |= 0x80;
        cmd_buf[token_count++] = token;
        if (offset != 0) {
            uint16_t off_val = (uint16_t)offset;
            memcpy(off16_buf + off16_count * 2, &off_val, 2);
            off16_count++;
        }
        uint32_t remaining_match = match_len - current;
        while (remaining_match > 0) {
            current = (remaining_match < 15) ? remaining_match : 15;
            cmd_buf[token_count++] = (uint8_t)(0x80 + 8 * current);
            remaining_match -= current;
        }
        return;
    }

    // Step 4: Long-match / far-offset path
    if (remaining_lit != 0) {
        cmd_buf[token_count++] = (uint8_t)(0x80 + remaining_lit);
    }

    uint32_t effective_offset = offset;
    if (effective_offset == 0) {
        effective_offset = (uint32_t)(-recent_offset);
    }

    uint8_t token_byte = 0;
    bool write_length = false;
    int32_t length_value = 0;

    if (effective_offset > 0xFFFF) {
        int32_t delta = (int32_t)match_len - 5;
        if (delta >= 0 && delta <= 23) {
            token_byte = (uint8_t)(match_len - 5);
        } else {
            token_byte = 2;
            length_value = (int32_t)match_len - 29;
            write_length = true;
        }
    } else {
        token_byte = 1;
        length_value = (int32_t)match_len - 91;
        write_length = true;
    }

    cmd_buf[token_count++] = token_byte;
    if (write_length) {
        uint32_t lv = (length_value > 0) ? (uint32_t)length_value : 0;
        writeLengthValue(len_buf, length_count, lv);
    }

    if (effective_offset > 0xFFFF) {
        uint32_t adjusted = effective_offset + block2_start - match_pos;
        writeOffset32(off32_buf, off32_pos, adjusted);
        off32_count++;
    } else {
        uint16_t off_val = (uint16_t)effective_offset;
        memcpy(off16_buf + off16_count * 2, &off_val, 2);
        off16_count++;
    }
}

// ── literalRunSlotCount ─────────────────────────────────────────
// Port of CPU fast_constants.literalRunSlotCount: value mod 7 for > 7.
__device__ uint32_t literalRunSlotCount(uint32_t value) {
    return (value > 7) ? ((value - 1) % 7 + 1) : value;
}

// ── emitWithLiteral1 ───────────────────────────────────────────
// Port of CPU writeOffsetWithLiteral1. For literal runs 8-63 bytes,
// scans for single-byte recent-offset matches and splits the run
// to emit them as (lit_count, match=1, offset=0) tokens.
__device__ void emitWithLiteral1(
    const uint8_t* src,
    uint8_t* lit_buf, uint32_t &lit_count,
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* off32_buf, uint32_t &off32_pos, uint32_t &off32_count,
    uint8_t* len_buf, uint32_t &length_count,
    uint32_t anchor, uint32_t lit_len,
    uint32_t match_len, uint32_t offset,
    uint32_t block2_start, int32_t recent_offset
) {
    // Only scan literal runs in [8, 63]
    if (lit_len < 8 || lit_len > 63) {
        for (uint32_t i = 0; i < lit_len; i++)
            lit_buf[lit_count + i] = src[anchor + i];
        lit_count += lit_len;
        emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                off32_buf, off32_pos, off32_count,
                len_buf, length_count, lit_len, match_len, offset,
                anchor, block2_start, recent_offset);
        return;
    }

    // Scan for bytes matching recent-offset pattern
    uint32_t found[33];
    uint32_t found_count = 0;
    uint32_t last = 0;
    int32_t ro = recent_offset; // negative distance

    for (uint32_t i = 1; i < lit_len; i++) {
        uint32_t p = anchor + i;
        int32_t back = (int32_t)p + ro;
        if (back >= 0 && src[p] == src[(uint32_t)back]) {
            if (i != last) {
                found[found_count] = i - last;
                found_count++;
                last = i + 1;
            } else {
                last = i + 1;
            }
            if (found_count >= 32) break;
        }
    }

    if (found_count != 0) {
        found[found_count] = lit_len - last;
        uint32_t cur_anchor = anchor;
        uint32_t cur_lit = lit_len;

        for (uint32_t fi = 0; fi < found_count; fi++) {
            uint32_t current = found[fi];
            uint32_t next_val = found[fi + 1];
            if (literalRunSlotCount(current) + literalRunSlotCount(next_val) + 1 > 7) {
                // Emit: `current` literals + 1 match byte at recent offset
                for (uint32_t i = 0; i < current; i++)
                    lit_buf[lit_count + i] = src[cur_anchor + i];
                lit_count += current;
                emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                        off32_buf, off32_pos, off32_count,
                        len_buf, length_count, current, 1, 0,
                        cur_anchor, block2_start, recent_offset);
                cur_anchor += current + 1;
                cur_lit -= current + 1;
            } else {
                found[fi + 1] += current + 1;
            }
        }

        // Emit remaining literals + the actual match
        for (uint32_t i = 0; i < cur_lit; i++)
            lit_buf[lit_count + i] = src[cur_anchor + i];
        lit_count += cur_lit;
        emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                off32_buf, off32_pos, off32_count,
                len_buf, length_count, cur_lit, match_len, offset,
                cur_anchor, block2_start, recent_offset);
    } else {
        // No recent-offset bytes found — emit normally
        for (uint32_t i = 0; i < lit_len; i++)
            lit_buf[lit_count + i] = src[anchor + i];
        lit_count += lit_len;
        emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                off32_buf, off32_pos, off32_count,
                len_buf, length_count, lit_len, match_len, offset,
                anchor, block2_start, recent_offset);
    }
}

// ── isLazyMatchBetter ────────────────────────────────────────────
// Returns true if `cand` (found at pos + step) is better than `current`.
// Port of CPU isLazyMatchBetter: 5*(len_diff) - 5 - (bits_diff) > step*4
__device__ bool isLazyMatchBetter(ChainMatch cand, ChainMatch current, int32_t step) {
    int32_t bits_cand = (cand.offset > 0) ? ((cand.offset > 0xFFFF) ? 32 : 16) : 0;
    int32_t bits_cur  = (current.offset > 0) ? ((current.offset > 0xFFFF) ? 32 : 16) : 0;
    return 5 * (cand.length - current.length) - 5 - (bits_cand - bits_cur) > step * 4;
}

// ── isMatchBetter ───────────────────────────────────────────────
// Port of CPU isMatchBetter: accounts for near/far offset cost.
__device__ bool isMatchBetter(int32_t match_length, int32_t match_offset,
                               int32_t best_length, int32_t best_offset) {
    if (match_length == best_length) return match_offset < best_offset;
    if ((match_offset <= 0xFFFF) == (best_offset <= 0xFFFF)) return match_length > best_length;
    if (best_offset <= 0xFFFF) return match_length > best_length + 5;
    return match_length >= best_length - 5;
}

// ── isBetterThanRecentMatch ──────────────────────────────────────
// Returns true if the hash match is preferable to the recent-offset match.
__device__ bool isBetterThanRecentMatch(int32_t recent_match_length, int32_t match_length, int32_t match_offset) {
    return recent_match_length < 2 ||
        (recent_match_length + 1 < match_length && (recent_match_length + 4 < match_length || match_offset < 65536));
}

// ── findMatchChain ───────────────────────────────────────────────
// Chain-hash match finder for the lazy parser. Port of CPU
// findMatchWithChainHasher. Called by lane 0 only.
//
// Parameters:
//   src, src_size, pos  -- source data and current scan position
//   recent_offset       -- recent match offset (negative distance)
//   first_hash          -- chain head table (hash_size entries)
//   long_hash           -- secondary direct-mapped table (hash_size entries)
//   next_hash           -- chain link table (NEXT_HASH_SIZE u16 entries)
//   hash_bits           -- log2 of hash_size
//   hash_mask           -- hash_size - 1
//   end_pos             -- block boundary (do not extend past this)
//
// Returns ChainMatch: {length, offset}
//   offset=0 means recent-offset reuse; offset>0 is explicit distance
__device__ ChainMatch findMatchChain(
    const uint8_t* src, uint32_t src_size,
    uint32_t pos,
    int32_t recent_offset,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    uint32_t end_pos,
    uint32_t lit_run_length
) {
    // Read 4 bytes at current position
    uint32_t bytes_at_pos = (uint32_t)src[pos]
                          | ((uint32_t)src[pos + 1] << 8)
                          | ((uint32_t)src[pos + 2] << 16)
                          | ((uint32_t)src[pos + 3] << 24);

    uint64_t at_src = read8safe(src + pos, pos, src_size);
    uint32_t ha = hashKeyA8(src + pos, hash_bits, hash_mask, at_src);
    uint32_t hb = hashKeyB8(src + pos, hash_bits, hash_mask, at_src);
    uint32_t hb_tag = hashKeyBTag8(src + pos, hash_bits, at_src);

    // (a) Recent-offset match: compare 4 bytes at src[pos] vs src[pos + recent_offset]
    int32_t recent_match_length = 0;
    {
        uint32_t recent_ref = (uint32_t)((int32_t)pos + recent_offset);
        if (recent_ref < pos) {  // valid backward reference
            uint32_t recent_word = (uint32_t)src[recent_ref]
                                 | ((uint32_t)src[recent_ref + 1] << 8)
                                 | ((uint32_t)src[recent_ref + 2] << 16)
                                 | ((uint32_t)src[recent_ref + 3] << 24);
            uint32_t xor_val = bytes_at_pos ^ recent_word;
            if (xor_val == 0) {
                // Full 4-byte match -- extend forward
                uint32_t ml = 4;
                uint32_t max_ext = end_pos - pos;
                while (ml < max_ext && src[pos + ml] == src[recent_ref + ml]) ml++;

                // Insert current position into hash tables
                uint32_t prev_head = first_hash[ha];
                next_hash[pos & 0xFFFF] = (uint16_t)(prev_head & 0xFFFF);
                first_hash[ha] = pos;
                long_hash[hb] = (hb_tag & 0x3F) | (pos << 6);

                return {(int32_t)ml, 0};
            }
            // Partial recent match: count leading matching bytes (0-3)
            recent_match_length = (int32_t)(__ffs(xor_val) - 1) / 8;
        }
    }

    // Min match length bump for long literal runs (matches CPU)
    int32_t minimum_match_length = (int32_t)MIN_MATCH;
    if (lit_run_length >= 64) {
        if (recent_match_length < 3) recent_match_length = 0;
        minimum_match_length += 1;
    }

    int32_t best_offset = 0;
    int32_t best_match_length = 0;

    // (b) Walk first_hash chain (max CHAIN_MAX_STEPS steps)
    {
        uint32_t head = first_hash[ha];
        uint32_t candidate_offset = pos - head;

        if (candidate_offset <= 0xFFFF) {
            if (candidate_offset != 0) {
                uint32_t chain_steps = CHAIN_MAX_STEPS;
                uint32_t hash_value = head;
                while (candidate_offset < 0x10000u) {  // stays within 64KB block
                    if (candidate_offset > 8) {
                        if (candidate_offset <= pos) {
                            uint32_t ref = pos - candidate_offset;
                            uint32_t ref_word = (uint32_t)src[ref]
                                              | ((uint32_t)src[ref + 1] << 8)
                                              | ((uint32_t)src[ref + 2] << 16)
                                              | ((uint32_t)src[ref + 3] << 24);
                            if (ref_word == bytes_at_pos) {
                                // Three-stage filter: check byte at best_match_length
                                bool quick_ok = true;
                                if (best_match_length >= 4) {
                                    uint32_t tail_pos = pos + (uint32_t)best_match_length;
                                    if (tail_pos >= end_pos) {
                                        quick_ok = false;
                                    } else {
                                        if (src[tail_pos] != src[tail_pos - candidate_offset])
                                            quick_ok = false;
                                    }
                                }
                                if (quick_ok) {
                                    uint32_t ml = 4;
                                    uint32_t max_ext = end_pos - pos;
                                    while (ml < max_ext && src[pos + ml] == src[ref + ml]) ml++;
                                    int32_t cand_len = (int32_t)ml;
                                    if (cand_len > best_match_length && cand_len >= minimum_match_length) {
                                        best_match_length = cand_len;
                                        best_offset = (int32_t)candidate_offset;
                                    }
                                }
                            }
                        }
                        chain_steps--;
                        if (chain_steps == 0) break;
                    }
                    uint32_t previous_offset = candidate_offset;
                    hash_value = (uint32_t)next_hash[hash_value & 0xFFFF];
                    // Chain positions are modulo 64K
                    candidate_offset = (uint16_t)(pos - hash_value);
                    if (candidate_offset <= previous_offset) break;
                }
            }
        } else if (candidate_offset <= pos && candidate_offset < 0x10000u) {
            // Far first-hash hit (> 0xFFFF but within dictionary)
            uint32_t ref = pos - candidate_offset;
            uint32_t ref_word = (uint32_t)src[ref]
                              | ((uint32_t)src[ref + 1] << 8)
                              | ((uint32_t)src[ref + 2] << 16)
                              | ((uint32_t)src[ref + 3] << 24);
            if (ref_word == bytes_at_pos) {
                uint32_t ml = 4;
                uint32_t max_ext = end_pos - pos;
                while (ml < max_ext && src[pos + ml] == src[ref + ml]) ml++;
                int32_t cand_len = (int32_t)ml;
                if (cand_len > minimum_match_length && cand_len >= 14) {
                    best_match_length = cand_len;
                    best_offset = (int32_t)candidate_offset;
                }
            }
        }
    }

    // (c) Check long_hash secondary table
    {
        uint32_t lh_value = long_hash[hb];
        if (((hb_tag ^ lh_value) & 0x3F) == 0) {
            uint32_t cand_pos = lh_value >> 6;
            if (cand_pos < pos) {
                uint32_t cand_off = pos - cand_pos;
                if (cand_off >= 8 && cand_off <= 0xFFFF) {
                    uint32_t ref = pos - cand_off;
                    uint32_t ref_word = (uint32_t)src[ref]
                                      | ((uint32_t)src[ref + 1] << 8)
                                      | ((uint32_t)src[ref + 2] << 16)
                                      | ((uint32_t)src[ref + 3] << 24);
                    if (ref_word == bytes_at_pos) {
                        // Three-stage filter: check byte at best_match_length
                        bool quick_ok = true;
                        if (best_match_length >= 4) {
                            uint32_t tail_pos = pos + (uint32_t)best_match_length;
                            if (tail_pos >= end_pos)
                                quick_ok = false;
                            else if (src[tail_pos] != src[tail_pos - cand_off])
                                quick_ok = false;
                        }
                        if (quick_ok) {
                        uint32_t ml = 4;
                        uint32_t max_ext = end_pos - pos;
                        while (ml < max_ext && src[pos + ml] == src[ref + ml]) ml++;
                        int32_t cand_len = (int32_t)ml;
                        if (cand_len >= minimum_match_length &&
                            isMatchBetter(cand_len, (int32_t)cand_off, best_match_length, best_offset)) {
                            best_match_length = cand_len;
                            best_offset = (int32_t)cand_off;
                        }
                        }
                    }
                }
            }
        }
    }

    // (d) Fixed offset-8 fallback
    if (pos >= 8) {
        uint32_t ref = pos - 8;
        uint32_t ref_word = (uint32_t)src[ref]
                          | ((uint32_t)src[ref + 1] << 8)
                          | ((uint32_t)src[ref + 2] << 16)
                          | ((uint32_t)src[ref + 3] << 24);
        if (ref_word == bytes_at_pos) {
            uint32_t ml = 4;
            uint32_t max_ext = end_pos - pos;
            while (ml < max_ext && src[pos + ml] == src[ref + ml]) ml++;
            int32_t cand_len = (int32_t)ml;
            if (cand_len >= best_match_length && cand_len >= minimum_match_length) {
                best_match_length = cand_len;
                best_offset = 8;
            }
        }
    }

    // (e) Insert current position into all three hash tables
    {
        uint32_t prev_head = first_hash[ha];
        next_hash[pos & 0xFFFF] = (uint16_t)(prev_head & 0xFFFF);
        first_hash[ha] = pos;
        long_hash[hb] = (hb_tag & 0x3F) | (pos << 6);
    }

    // (f) Return best match vs recent match (prefer recent if close)
    if (best_offset == 0 || !isBetterThanRecentMatch(recent_match_length, best_match_length, best_offset)) {
        return {recent_match_length, 0};
    }
    return {best_match_length, best_offset};
}

// ── insertChainRange ─────────────────────────────────────────────
// Insert positions [from..to) into the chain hash tables.
// Used after emitting a match to populate hash entries for the
// matched byte range. longHash at exponentially spaced positions;
// firstHash/nextHash at every position.
__device__ void insertChainRange(
    const uint8_t* src, uint32_t src_size,
    uint32_t from, uint32_t to,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask
) {
    // longHash at exponentially spaced positions: i = 0, 1, 3, 7, 15, ...
    {
        uint32_t len = to - from;
        uint32_t i = 0;
        while (i < len) {
            uint32_t p = from + i;
            uint64_t at = read8safe(src + p, p, src_size);
            uint32_t hb = hashKeyB8(src + p, hash_bits, hash_mask, at);
            uint32_t hb_tag = hashKeyBTag8(src + p, hash_bits, at);
            long_hash[hb] = (hb_tag & 0x3F) | (p << 6);
            i = 2 * i + 1;
        }
    }

    // firstHash + nextHash at every position
    for (uint32_t p = from; p < to; p++) {
        uint64_t at = read8safe(src + p, p, src_size);
        uint32_t ha = hashKeyA8(src + p, hash_bits, hash_mask, at);
        uint32_t prev_head = first_hash[ha];
        next_hash[p & 0xFFFF] = (uint16_t)(prev_head & 0xFFFF);
        first_hash[ha] = p;
    }
}

// ── scanBlockChain: serial chain-hash lazy parser for one block ──
// Port of CPU runLazyParserChain. Lane 0 does all work; other lanes
// participate only in the final __syncwarp().
//
// Scans positions [start_pos .. end_pos), using the chain hasher to
// find matches with lazy-1 and lazy-2 evaluation, backward extension,
// and emitting tokens via emitCmd.
__device__ void scanBlockChain(
    const uint8_t* src, uint32_t src_size,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    uint8_t* lit_buf, uint32_t &lit_count,
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* off32_buf, uint32_t &off32_pos, uint32_t &off32_count,
    uint8_t* len_buf, uint32_t &length_count,
    uint32_t &anchor, int32_t &recent_offset,
    uint32_t start_pos, uint32_t end_pos, uint32_t block2_start
) {
    const uint32_t lane = threadIdx.x & 31;

    if (lane != 0) return;  // All work done by lane 0

    uint32_t pos = start_pos;

    // Guard: need at least 5 bytes of lookahead
    if (pos + 5 >= end_pos) {
        // Handle trailing literals and return
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = 0; i < trailing; i++)
                lit_buf[lit_count + i] = src[anchor + i];
            lit_count += trailing;
            emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                    off32_buf, off32_pos, off32_count,
                    len_buf, length_count, trailing, 0, 0,
                    anchor, block2_start, recent_offset);
            anchor = end_pos;
        }
        return;
    }

    while (pos + 5 < end_pos) {
        // Find a match at current position
        uint32_t cur_lit_run = pos - anchor;
        ChainMatch match = findMatchChain(src, src_size, pos, recent_offset,
                                          first_hash, long_hash, next_hash,
                                          hash_bits, hash_mask, end_pos, cur_lit_run);
        if (match.length < 2) {
            pos++;
            continue;
        }

        // Lazy-1: check pos+1 for a better match
        while (pos + 6 < end_pos) {
            ChainMatch lazy1 = findMatchChain(src, src_size, pos + 1, recent_offset,
                                              first_hash, long_hash, next_hash,
                                              hash_bits, hash_mask, end_pos, cur_lit_run + 1);
            if (lazy1.length >= 2 && isLazyMatchBetter(lazy1, match, 0)) {
                pos++;
                match = lazy1;
            } else {
                // Lazy-2: check pos+2 (only if current match length > 2)
                if (pos + 7 >= end_pos || match.length == 2) break;
                ChainMatch lazy2 = findMatchChain(src, src_size, pos + 2, recent_offset,
                                                  first_hash, long_hash, next_hash,
                                                  hash_bits, hash_mask, end_pos, cur_lit_run + 2);
                if (lazy2.length >= 2 && isLazyMatchBetter(lazy2, match, 1)) {
                    pos += 2;
                    match = lazy2;
                } else {
                    break;
                }
            }
        }

        // Resolve actual offset for backward extension
        uint32_t actual_offset;
        if (match.offset == 0)
            actual_offset = (uint32_t)(-recent_offset);
        else
            actual_offset = (uint32_t)match.offset;

        // Enforce minimum match length for far offsets
        uint32_t off_param = (uint32_t)match.offset;
        uint32_t resolved_off = off_param;
        if (resolved_off == 0) resolved_off = (uint32_t)(-recent_offset);
        if (resolved_off > 0xFFFF && (uint32_t)match.length < 14) {
            pos++;
            continue;
        }

        // (e) Backward extension: extend match backward into literal run
        while (pos > anchor && pos > actual_offset) {
            uint32_t prev = pos - 1;
            uint32_t back = prev - actual_offset;
            if (src[prev] != src[back]) break;
            pos--;
            match.length++;
        }

        // Compute literal run and emit with literal-1 splitting
        uint32_t lit_len = pos - anchor;
        emitWithLiteral1(src, lit_buf, lit_count,
                         cmd_buf, token_count, off16_buf, off16_count,
                         off32_buf, off32_pos, off32_count,
                         len_buf, length_count,
                         anchor, lit_len, (uint32_t)match.length, off_param,
                         block2_start, recent_offset);

        // Update recent offset
        recent_offset = -(int32_t)actual_offset;

        // Advance past the match
        uint32_t match_end = pos + (uint32_t)match.length;

        // (g) Insert matched range positions into hash tables
        if (match_end > pos + 1 && match_end + 8 <= src_size) {
            uint32_t insert_end = (match_end + 8 <= src_size) ? match_end : src_size - 8;
            insertChainRange(src, src_size, pos + 1, insert_end, first_hash, long_hash, next_hash,
                             hash_bits, hash_mask);
        }

        anchor = match_end;
        pos = match_end;
    }

    // (h) Trailing literals at end_pos
    {
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = 0; i < trailing; i++)
                lit_buf[lit_count + i] = src[anchor + i];
            lit_count += trailing;

            emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                    off32_buf, off32_pos, off32_count,
                    len_buf, length_count, trailing, 0, 0,
                    anchor, block2_start, recent_offset);

            anchor = end_pos;
        }
    }
}

// ── scanBlock: warp-parallel greedy scan for one block ──────────
// Scans positions [start_pos .. end_pos), finding matches, extending
// them (capped at end_pos), and emitting tokens + trailing literals.
// All counters are passed by reference and accumulate across calls.
__device__ void scanBlock(
    const uint8_t* src, uint32_t src_size,
    uint32_t* ht, uint32_t hash_bits, uint32_t hash_mask,
    uint8_t* lit_buf, uint32_t &lit_count,
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* off32_buf, uint32_t &off32_pos, uint32_t &off32_count,
    uint8_t* len_buf, uint32_t &length_count,
    uint32_t &anchor, int32_t &recent_offset,
    uint32_t start_pos, uint32_t end_pos, uint32_t block2_start
) {
    const uint32_t lane = threadIdx.x & 31;
    uint32_t pos = start_pos;

    while (pos + MIN_MATCH <= end_pos) {
        uint32_t my_pos = pos + lane;
        uint32_t my_byte = (my_pos < src_size) ? (uint32_t)src[my_pos] : 0u;

        uint32_t b1 = __shfl_down_sync(0xFFFFFFFF, my_byte, 1);
        uint32_t b2 = __shfl_down_sync(0xFFFFFFFF, my_byte, 2);
        uint32_t b3 = __shfl_down_sync(0xFFFFFFFF, my_byte, 3);
        uint32_t key4 = my_byte | (b1 << 8) | (b2 << 16) | (b3 << 24);

        uint32_t remaining = end_pos - pos;
        uint32_t active_count;
        if (remaining >= 32) active_count = 29;
        else if (remaining >= 4) active_count = remaining - 3;
        else active_count = 0;
        bool is_active = (lane < active_count);

        uint32_t h = hashKey(key4, hash_bits, hash_mask);
        bool hash_match = false;
        uint32_t hash_ref = 0;

        if (is_active) {
            uint32_t ref_val = ht[h];
            if (ref_val != HASH_EMPTY && ref_val < my_pos && (my_pos - ref_val) <= 0xFFFF) {
                uint32_t rk = (uint32_t)src[ref_val] |
                             ((uint32_t)src[ref_val+1] << 8) |
                             ((uint32_t)src[ref_val+2] << 16) |
                             ((uint32_t)src[ref_val+3] << 24);
                if (rk == key4) {
                    hash_match = true;
                    hash_ref = ref_val;
                }
            }
        }

        uint32_t warp_match = __match_any_sync(0xFFFFFFFF, key4);
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, is_active);
        warp_match &= active_mask;
        uint32_t lower_mask = warp_match & ((1u << lane) - 1);

        bool has_match = hash_match;
        uint32_t my_ref = hash_ref;

        if (is_active && lower_mask != 0) {
            uint32_t intra_lane = __ffs(lower_mask) - 1;
            has_match = true;
            my_ref = pos + intra_lane;
        }

        uint32_t match_ballot = __ballot_sync(0xFFFFFFFF, has_match);

        {
            uint32_t write_limit = (match_ballot != 0)
                ? (uint32_t)(__ffs(match_ballot) - 1) : active_count;
            if (is_active && lane <= write_limit) {
                ht[h] = my_pos;
            }
        }

        if (match_ballot == 0) {
            pos += active_count;
            continue;
        }

        uint32_t first_lane = __ffs(match_ballot) - 1;
        uint32_t match_pos = pos + first_lane;
        uint32_t match_ref = __shfl_sync(0xFFFFFFFF, my_ref, first_lane);

        // Match extension capped at end_pos (block boundary)
        uint32_t match_len = MIN_MATCH;
        uint32_t max_match = end_pos - match_pos;
        for (uint32_t ext = MIN_MATCH; ext < max_match; ext += 32) {
            uint32_t check = ext + lane;
            bool mm;
            if (check >= max_match)
                mm = true;
            else
                mm = (src[match_pos + check] != src[match_ref + check]);
            uint32_t mm_mask = __ballot_sync(0xFFFFFFFF, mm);
            if (mm_mask != 0) {
                match_len = ext + __ffs(mm_mask) - 1;
                goto ext_done;
            }
        }
        match_len = max_match;
        ext_done:

        {
            int32_t neg_off = -(int32_t)(match_pos - match_ref);
            uint32_t off_param = (neg_off == recent_offset) ? 0 : (uint32_t)(-neg_off);

            // Resolve actual offset for fast-path decision
            uint32_t resolved_off = off_param;
            if (resolved_off == 0) resolved_off = (uint32_t)(-recent_offset);

            // Enforce minimum match length for far offsets (CPU mmlt = 14)
            if (resolved_off > 0xFFFF && match_len < 14) {
                pos = match_pos + 1;
                continue;
            }

            uint32_t lit_len = match_pos - anchor;
            for (uint32_t i = lane; i < lit_len; i += 32)
                lit_buf[lit_count + i] = src[anchor + i];
            lit_count += lit_len;

            if (lit_len <= 7 && match_len <= 15 && resolved_off <= 0xFFFF) {
                if (lane == 0) {
                    uint8_t token = (uint8_t)(lit_len + 8 * match_len);
                    if (off_param == 0) token |= 0x80;
                    cmd_buf[token_count++] = token;
                    if (off_param != 0) {
                        uint16_t ov = (uint16_t)off_param;
                        memcpy(off16_buf + off16_count * 2, &ov, 2);
                        off16_count++;
                    }
                }
            } else {
                if (lane == 0) {
                    emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                            off32_buf, off32_pos, off32_count,
                            len_buf, length_count, lit_len, match_len, off_param,
                            anchor, block2_start, recent_offset);
                }
            }

            recent_offset = neg_off;
        }

        anchor = match_pos + match_len;
        pos = anchor;
    }

    // Trailing literals up to end_pos
    {
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = lane; i < trailing; i += 32)
                lit_buf[lit_count + i] = src[anchor + i];
            lit_count += trailing;

            if (lane == 0) {
                emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                        off32_buf, off32_pos, off32_count,
                        len_buf, length_count, trailing, 0, 0,
                        anchor, block2_start, recent_offset);
            }

            anchor = end_pos;
        }
    }
}

extern "C" __global__ void __launch_bounds__(32, 1) slzCompressL1Kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const CompressChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ global_hash,
    uint32_t* __restrict__ comp_sizes,
    uint32_t total_chunks,
    uint32_t hash_bits,
    uint32_t use_chain
) {
    extern __shared__ uint32_t shared_ht[];

    const uint32_t chunk_id = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31;
    if (chunk_id >= total_chunks) return;

    const uint32_t hash_size = 1u << hash_bits;
    const uint32_t hash_mask = hash_size - 1;

    const CompressChunkDesc& desc = descs[chunk_id];
    const uint8_t* src = input + desc.src_offset;
    const uint32_t src_size = desc.src_size;

    uint8_t* dst = output + desc.dst_offset;
    const uint32_t lit_data_start = (desc.is_first ? INITIAL_COPY : 0) + 3;
    uint8_t* lit_buf = dst + lit_data_start;
    uint8_t* cmd_buf = dst + src_size;
    uint8_t* off16_buf = cmd_buf + (src_size / 4);
    uint8_t* off32_buf = off16_buf + (src_size / 2);
    uint8_t* len_buf = off32_buf + (src_size / 4);

    uint32_t lit_count = 0, token_count = 0, off16_count = 0, length_count = 0;
    uint32_t off32_pos = 0, off32_count_block1 = 0, off32_count_block2 = 0, off32_count = 0;
    uint32_t cmd_stream2_offset = 0;
    uint32_t anchor = desc.is_first ? INITIAL_COPY : 0;
    int32_t recent_offset = -8;

    if (use_chain) {
        // ── Chain parser mode ────────────────────────────────────
        // Three hash tables per block laid out contiguously in global_hash:
        //   first_hash: hash_size u32
        //   long_hash:  hash_size u32
        //   next_hash:  NEXT_HASH_SIZE u16 (= NEXT_HASH_SIZE/2 u32 words)
        uint32_t table_stride = hash_size + hash_size + (NEXT_HASH_SIZE / 2);
        uint32_t* base = global_hash + (uint64_t)chunk_id * table_stride;
        uint32_t* chain_first_hash = base;
        uint32_t* chain_long_hash  = base + hash_size;
        uint16_t* chain_next_hash  = (uint16_t*)(base + hash_size + hash_size);

        // Initialize all three tables to 0
        uint32_t total_words = hash_size + hash_size + (NEXT_HASH_SIZE / 2);
        for (uint32_t i = lane; i < total_words; i += 32)
            base[i] = 0;
        __syncwarp();

        // ── Block 1 pass ────
        {
            uint32_t block1_end = (src_size < BLOCK1_SIZE) ? src_size : BLOCK1_SIZE;
            scanBlockChain(src, src_size,
                           chain_first_hash, chain_long_hash, chain_next_hash,
                           hash_bits, hash_mask,
                           lit_buf, lit_count,
                           cmd_buf, token_count,
                           off16_buf, off16_count,
                           off32_buf, off32_pos, off32_count,
                           len_buf, length_count,
                           anchor, recent_offset,
                           anchor, block1_end, /*block2_start=*/0);
            off32_count_block1 = off32_count;
            off32_count = 0;
        }

        // ── Block 2 pass ────
        if (src_size > BLOCK1_SIZE) {
            cmd_stream2_offset = token_count;
            uint32_t block2_start_pos = (anchor > BLOCK1_SIZE) ? anchor : BLOCK1_SIZE;
            scanBlockChain(src, src_size,
                           chain_first_hash, chain_long_hash, chain_next_hash,
                           hash_bits, hash_mask,
                           lit_buf, lit_count,
                           cmd_buf, token_count,
                           off16_buf, off16_count,
                           off32_buf, off32_pos, off32_count,
                           len_buf, length_count,
                           anchor, recent_offset,
                           block2_start_pos, src_size, /*block2_start=*/BLOCK1_SIZE);
            off32_count_block2 = off32_count;
        }
    } else {
        // ── Greedy parser mode (original) ────────────────────────
        // Use shared memory if available, else global memory per-block tables
        uint32_t* ht = (global_hash != nullptr)
            ? global_hash + (uint64_t)chunk_id * hash_size
            : shared_ht;

        for (uint32_t i = lane; i < hash_size; i += 32)
            ht[i] = HASH_EMPTY;
        __syncwarp();

        // ── Block 1 pass ────
        {
            uint32_t block1_end = (src_size < BLOCK1_SIZE) ? src_size : BLOCK1_SIZE;
            scanBlock(src, src_size, ht, hash_bits, hash_mask,
                      lit_buf, lit_count,
                      cmd_buf, token_count,
                      off16_buf, off16_count,
                      off32_buf, off32_pos, off32_count,
                      len_buf, length_count,
                      anchor, recent_offset,
                      anchor, block1_end, /*block2_start=*/0);
            off32_count_block1 = off32_count;
            off32_count = 0;
        }

        // ── Block 2 pass ────
        if (src_size > BLOCK1_SIZE) {
            cmd_stream2_offset = token_count;
            uint32_t block2_start_pos = (anchor > BLOCK1_SIZE) ? anchor : BLOCK1_SIZE;
            scanBlock(src, src_size, ht, hash_bits, hash_mask,
                      lit_buf, lit_count,
                      cmd_buf, token_count,
                      off16_buf, off16_count,
                      off32_buf, off32_pos, off32_count,
                      len_buf, length_count,
                      anchor, recent_offset,
                      block2_start_pos, src_size, /*block2_start=*/BLOCK1_SIZE);
            off32_count_block2 = off32_count;
        }
    }

    __syncwarp();

    if (lane == 0) {
        uint32_t out_pos = 0;
        if (desc.is_first) {
            memcpy(dst, src, INITIAL_COPY);
            out_pos = INITIAL_COPY;
        }

        dst[out_pos] = (uint8_t)((lit_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((lit_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(lit_count & 0xFF);
        out_pos += 3 + lit_count;

        dst[out_pos] = (uint8_t)((token_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((token_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(token_count & 0xFF);
        out_pos += 3;
        memcpy(dst + out_pos, cmd_buf, token_count);
        out_pos += token_count;

        if (src_size > BLOCK1_SIZE) {
            uint16_t cs2o = (cmd_stream2_offset > 0) ? (uint16_t)cmd_stream2_offset : (uint16_t)token_count;
            memcpy(dst + out_pos, &cs2o, 2);
            out_pos += 2;
        }

        dst[out_pos] = (uint8_t)(off16_count & 0xFF);
        dst[out_pos + 1] = (uint8_t)((off16_count >> 8) & 0xFF);
        out_pos += 2;
        memcpy(dst + out_pos, off16_buf, off16_count * 2);
        out_pos += off16_count * 2;

        uint32_t c1 = (off32_count_block1 < 4095) ? off32_count_block1 : 4095;
        uint32_t c2 = (off32_count_block2 < 4095) ? off32_count_block2 : 4095;
        uint32_t packed = (c1 << 12) | c2;
        dst[out_pos++] = (uint8_t)(packed & 0xFF);
        dst[out_pos++] = (uint8_t)((packed >> 8) & 0xFF);
        dst[out_pos++] = (uint8_t)((packed >> 16) & 0xFF);
        if (off32_count_block1 >= 4095) {
            uint16_t ext = (uint16_t)off32_count_block1;
            memcpy(dst + out_pos, &ext, 2);
            out_pos += 2;
        }
        if (off32_count_block2 >= 4095) {
            uint16_t ext = (uint16_t)off32_count_block2;
            memcpy(dst + out_pos, &ext, 2);
            out_pos += 2;
        }
        memcpy(dst + out_pos, off32_buf, off32_pos);
        out_pos += off32_pos;

        memcpy(dst + out_pos, len_buf, length_count);
        out_pos += length_count;

        comp_sizes[chunk_id] = out_pos;
    }
}
