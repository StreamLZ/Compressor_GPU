// ── StreamLZ GPU LZ encode - serial chain-hash lazy parser ──────
// The use_chain=1 parser: a serial chain-hash lazy matcher run
// entirely on lane 0. Port of the CPU runLazyParserChain. Used at
// the higher levels where the driver enables chain mode.
//
// Included by lz_kernel.cu - see that file for the build line.
#pragma once

#include "lz_format.cuh"
#include "lz_token_emit.cuh"

// ── v4 #14 gate-0 instrumentation (compile-time, default off) ────
// Counts the chain parser's actual work shape so the warp-
// parallelization decision is data-driven: findMatchChain calls,
// chain-candidate evaluations, and forward-extend byte compares.
// Same pattern as SLZ_COUNT_PP in lz_decode_raw.cuh: flip to 1,
// `zig build ptx && zig build`, run an L5 encode, read the globals,
// flip back.
#ifndef SLZ_COUNT_CHAIN
#define SLZ_COUNT_CHAIN 0
#endif
#if SLZ_COUNT_CHAIN
__device__ unsigned long long g_slz_chain_calls = 0;
__device__ unsigned long long g_slz_chain_cand = 0;
__device__ unsigned long long g_slz_extend_bytes = 0;
#define SLZ_CHAIN_COUNT(var, n) atomicAdd(&(var), (unsigned long long)(n))
#else
#define SLZ_CHAIN_COUNT(var, n) ((void)0)
#endif

// ── Lazy-match cost weights ─────────────────────────────────────
// Port of CPU isLazyMatchBetter: a candidate found `step` bytes
// ahead must beat the current match by enough length to pay for the
// extra literals it would leave behind.
static constexpr int32_t LAZY_LEN_WEIGHT     = 5;  // weight on the match-length advantage
static constexpr int32_t LAZY_MATCH_PENALTY  = 5;  // flat penalty for taking the later match
static constexpr int32_t LAZY_STEP_WEIGHT    = 4;  // per-step literal cost
static constexpr int32_t NEAR_OFFSET_COST_BITS = 16;  // bit cost of a near (off16) match offset
static constexpr int32_t FAR_OFFSET_COST_BITS  = 32;  // bit cost of a far (off32) match offset

// ── isMatchBetter near/far margins ──────────────────────────────
static constexpr int32_t OFFSET_CLASS_LEN_MARGIN = 5;  // length advantage to switch near/far class

// ── isBetterThanRecentMatch heuristic thresholds ────────────────
static constexpr int32_t RECENT_MATCH_MIN       = 2;  // recent-match length below which the hash match always wins
static constexpr int32_t RECENT_LEN_MARGIN      = 1;  // hash match must be at least this much longer
static constexpr int32_t RECENT_LEN_MARGIN_FAR  = 4;  // extra margin required before allowing a far hash match

// ── chain-parser tuning ─────────────────────────────────────────
static constexpr uint32_t LONG_LIT_RUN_THRESHOLD = 64;  // literal run length that bumps the minimum match
static constexpr uint32_t CHAIN_MIN_LOOKAHEAD = 5;      // bytes of lookahead the chain parser needs

// ── isLazyMatchBetter ───────────────────────────────────────────
// Returns true if `cand` (found at pos + step) is better than `current`.
// Port of CPU isLazyMatchBetter:
//   LAZY_LEN_WEIGHT*len_diff - LAZY_MATCH_PENALTY - bits_diff > step*LAZY_STEP_WEIGHT
__device__ bool isLazyMatchBetter(ChainMatch cand, ChainMatch current, int32_t step) {
    int32_t bits_cand = (cand.offset > 0)
        ? ((cand.offset > (int32_t)NEAR_OFFSET_MAX) ? FAR_OFFSET_COST_BITS : NEAR_OFFSET_COST_BITS) : 0;
    int32_t bits_cur  = (current.offset > 0)
        ? ((current.offset > (int32_t)NEAR_OFFSET_MAX) ? FAR_OFFSET_COST_BITS : NEAR_OFFSET_COST_BITS) : 0;
    return LAZY_LEN_WEIGHT * (cand.length - current.length)
         - LAZY_MATCH_PENALTY - (bits_cand - bits_cur) > step * LAZY_STEP_WEIGHT;
}

// ── isMatchBetter ───────────────────────────────────────────────
// Port of CPU isMatchBetter: accounts for near/far offset cost.
__device__ bool isMatchBetter(int32_t match_length, int32_t match_offset,
                               int32_t best_length, int32_t best_offset) {
    if (match_length == best_length) return match_offset < best_offset;
    if ((match_offset <= (int32_t)NEAR_OFFSET_MAX) == (best_offset <= (int32_t)NEAR_OFFSET_MAX))
        return match_length > best_length;
    if (best_offset <= (int32_t)NEAR_OFFSET_MAX) return match_length > best_length + OFFSET_CLASS_LEN_MARGIN;
    return match_length >= best_length - OFFSET_CLASS_LEN_MARGIN;
}

// ── isBetterThanRecentMatch ─────────────────────────────────────
// Returns true if the hash match is preferable to the recent-offset match.
__device__ bool isBetterThanRecentMatch(int32_t recent_match_length, int32_t match_length, int32_t match_offset) {
    return recent_match_length < RECENT_MATCH_MIN ||
        (recent_match_length + RECENT_LEN_MARGIN < match_length &&
         (recent_match_length + RECENT_LEN_MARGIN_FAR < match_length ||
          match_offset <= (int32_t)NEAR_OFFSET_MAX));
}

// ── findMatchChain ──────────────────────────────────────────────
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
// v4 #16 (HAS_DICT): the preset dictionary is probed as the STRICTLY
// lowest-priority source - only when the window yields nothing usable
// (no chain/long-hash/offset-8 match and no recent reuse), so
// dictionary-less parses are byte-identical by construction. The
// probe reuses the greedy parser's dict table (hashKey6 buckets) and
// gates distances to off16 (<= NEAR_OFFSET_MAX), exactly like the
// greedy integration. The lazy-1/lazy-2 evaluation upstream treats a
// dict match as an ordinary candidate; backward extension never
// applies to dict matches (the `pos > actual_offset` guard is always
// false for them - dict distances exceed the position by design).
template <bool HAS_DICT = false>
__device__ ChainMatch findMatchChain(
    const uint8_t* src, uint32_t src_size,
    uint32_t pos,
    int32_t recent_offset,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    uint32_t end_pos,
    uint32_t lit_run_length,
    const uint8_t* dict = nullptr, uint32_t dict_len = 0,
    const uint32_t* dict_ht = nullptr
) {
    SLZ_CHAIN_COUNT(g_slz_chain_calls, 1);
    // Read 4 bytes at current position
    uint32_t bytes_at_pos = readU32LE(src + pos);

    uint64_t at_src = read8safe(src, pos, src_size);
    uint32_t ha = hashTableA(hash_bits, hash_mask, at_src);
    uint32_t hb = hashTableB(hash_bits, hash_mask, at_src);
    uint32_t hb_tag = hashTagB(at_src);

    // (a) Recent-offset match: compare 4 bytes at src[pos] vs src[pos + recent_offset]
    int32_t recent_match_length = 0;
    {
        uint32_t recent_ref = (uint32_t)((int32_t)pos + recent_offset);
        if (recent_ref < pos) {  // valid backward reference
            uint32_t recent_word = readU32LE(src + recent_ref);
            uint32_t xor_val = bytes_at_pos ^ recent_word;
            if (xor_val == 0) {
                // Full 4-byte match -- extend forward
                uint32_t ml = 4;
                uint32_t max_ext = end_pos - pos;
                while (ml < max_ext && src[pos + ml] == src[recent_ref + ml]) ml++;

                // Insert current position into hash tables. The long_hash
                // entry packs (hb_tag in low 6 bits) | (pos << 6) - the
                // 26-bit pos truncation is safe because chunks are ≤ 256KB
                // (= 2^18), so pos < 2^18 < 2^26 and the top 6 bits of
                // (pos << LONG_HASH_TAG_BITS) are guaranteed zero.
                uint32_t prev_head = first_hash[ha];
                next_hash[pos & NEXT_HASH_INDEX_MASK] = (uint16_t)(prev_head & NEXT_HASH_INDEX_MASK);
                first_hash[ha] = pos;
                long_hash[hb] = (hb_tag & LONG_HASH_TAG_MASK) | (pos << LONG_HASH_TAG_BITS);

                return {(int32_t)ml, 0};
            }
            // Partial recent match: count leading matching bytes (0-3).
            // xor_val has its lowest set bit at the first mismatched bit;
            // __ffs returns a 1-based bit index, so __ffs(x)-1 is the
            // 0-based trailing-zero count. Divided by BITS_PER_BYTE (=8)
            // it counts full matching low bytes - the partial-match length.
            // (Mirrors CPU runChainParser recent-match path.)
            recent_match_length = (int32_t)(__ffs(xor_val) - 1) / (int32_t)BITS_PER_BYTE;
        }
    }

    // Min match length bump for long literal runs (matches CPU
    // runChainParser, src/encode/fast/fast_lz_parser.zig).
    int32_t minimum_match_length = (int32_t)MIN_MATCH;
    if (lit_run_length >= LONG_LIT_RUN_THRESHOLD) {
        if (recent_match_length < 3) recent_match_length = 0;
        minimum_match_length += 1;
    }

    int32_t best_offset = 0;
    int32_t best_match_length = 0;

    // (b) Walk first_hash chain (max CHAIN_MAX_STEPS steps)
    {
        uint32_t head = first_hash[ha];
        uint32_t candidate_offset = pos - head;

        if (candidate_offset <= NEAR_OFFSET_MAX) {
            if (candidate_offset != 0) {
                uint32_t chain_steps = CHAIN_MAX_STEPS;
                uint32_t hash_value = head;
                while (candidate_offset < LZ_BLOCK_SIZE) {  // stays within 64KB block
                    SLZ_CHAIN_COUNT(g_slz_chain_cand, 1);
                    if (candidate_offset > MIN_HASH_MATCH_OFFSET) {
                        if (candidate_offset <= pos) {
                            uint32_t ref = pos - candidate_offset;
                            uint32_t ref_word = readU32LE(src + ref);
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
                                    SLZ_CHAIN_COUNT(g_slz_extend_bytes, ml);
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
                    hash_value = (uint32_t)next_hash[hash_value & NEXT_HASH_INDEX_MASK];
                    // Chain positions are modulo 64K
                    candidate_offset = (uint16_t)(pos - hash_value);
                    if (candidate_offset <= previous_offset) break;
                }
            }
        }
        // Note: there is no reachable "far first-hash" branch here -
        // NEAR_OFFSET_MAX == LZ_BLOCK_SIZE - 1, so any candidate_offset
        // > NEAR_OFFSET_MAX is also >= LZ_BLOCK_SIZE. Far matches are
        // discovered via the long_hash secondary table below.
    }

    // (c) Check long_hash secondary table
    {
        uint32_t lh_value = long_hash[hb];
        if (((hb_tag ^ lh_value) & LONG_HASH_TAG_MASK) == 0) {
            uint32_t cand_pos = lh_value >> LONG_HASH_TAG_BITS;
            if (cand_pos < pos) {
                uint32_t cand_off = pos - cand_pos;
                if (cand_off >= MIN_HASH_MATCH_OFFSET && cand_off <= NEAR_OFFSET_MAX) {
                    uint32_t ref = pos - cand_off;
                    uint32_t ref_word = readU32LE(src + ref);
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
    if (pos >= MIN_HASH_MATCH_OFFSET) {
        uint32_t ref = pos - MIN_HASH_MATCH_OFFSET;
        uint32_t ref_word = readU32LE(src + ref);
        if (ref_word == bytes_at_pos) {
            uint32_t ml = 4;
            uint32_t max_ext = end_pos - pos;
            while (ml < max_ext && src[pos + ml] == src[ref + ml]) ml++;
            int32_t cand_len = (int32_t)ml;
            if (cand_len >= best_match_length && cand_len >= minimum_match_length) {
                best_match_length = cand_len;
                best_offset = (int32_t)MIN_HASH_MATCH_OFFSET;
            }
        }
    }

    // (e) Insert current position into all three hash tables
    {
        uint32_t prev_head = first_hash[ha];
        next_hash[pos & NEXT_HASH_INDEX_MASK] = (uint16_t)(prev_head & NEXT_HASH_INDEX_MASK);
        first_hash[ha] = pos;
        long_hash[hb] = (hb_tag & LONG_HASH_TAG_MASK) | (pos << LONG_HASH_TAG_BITS);
    }

    // (f) Return best match vs recent match (prefer recent if close)
    if (best_offset == 0 || !isBetterThanRecentMatch(recent_match_length, best_match_length, best_offset)) {
        // v4 #16: dictionary probe - only when the window would yield
        // no usable match at all (a recent reuse of length >= 2 is a
        // match the caller takes; dict never competes with it).
        if constexpr (HAS_DICT) {
            if (recent_match_length < 2) {
                uint32_t h6 = hashKey6(at_src, hash_bits, hash_mask);
                uint32_t dv = dict_ht[h6];
                if (dv != HASH_EMPTY) {
                    uint32_t dist = pos + (dict_len - dv);
                    if (dist <= NEAR_OFFSET_MAX && readU32LE(dict + dv) == bytes_at_pos) {
                        uint32_t ml = 4;
                        uint32_t max_ext = end_pos - pos;
                        uint32_t dict_avail = dict_len - dv;
                        if (dict_avail < max_ext) max_ext = dict_avail;
                        while (ml < max_ext && src[pos + ml] == dict[dv + ml]) ml++;
                        if ((int32_t)ml >= minimum_match_length)
                            return {(int32_t)ml, (int32_t)dist};
                    }
                }
            }
        }
        return {recent_match_length, 0};
    }
    return {best_match_length, best_offset};
}

// ── insertChainRange ────────────────────────────────────────────
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
            uint64_t at = read8safe(src, p, src_size);
            uint32_t hb = hashTableB(hash_bits, hash_mask, at);
            uint32_t hb_tag = hashTagB(at);
            long_hash[hb] = (hb_tag & LONG_HASH_TAG_MASK) | (p << LONG_HASH_TAG_BITS);
            i = 2 * i + 1;
        }
    }

    // firstHash + nextHash at every position
    for (uint32_t p = from; p < to; p++) {
        uint64_t at = read8safe(src, p, src_size);
        uint32_t ha = hashTableA(hash_bits, hash_mask, at);
        uint32_t prev_head = first_hash[ha];
        next_hash[p & NEXT_HASH_INDEX_MASK] = (uint16_t)(prev_head & NEXT_HASH_INDEX_MASK);
        first_hash[ha] = p;
    }
}

// ── scanBlockChain: serial chain-hash lazy parser for one block ──
// Port of CPU runLazyParserChain. Lane 0 does all work; the other
// lanes do nothing here - the warp re-converges at the __syncwarp()
// in the kernel after this function returns.
//
// Scans positions [start_pos .. end_pos), using the chain hasher to
// find matches with lazy-1 and lazy-2 evaluation, backward extension,
// and emitting tokens via emitCmd.
template <bool HAS_DICT = false>
__device__ void scanBlockChain(
    const uint8_t* src, uint32_t src_size,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    OutputStreams &s,
    uint32_t &anchor, int32_t &recent_offset,
    uint32_t start_pos, uint32_t end_pos, uint32_t block2_start,
    const uint8_t* dict = nullptr, uint32_t dict_len = 0,
    const uint32_t* dict_ht = nullptr
) {
    const uint32_t lane = threadIdx.x & LANE_MASK;

    if (lane != 0) return;  // All work done by lane 0

    uint32_t pos = start_pos;

    // Guard: need at least CHAIN_MIN_LOOKAHEAD bytes of lookahead
    if (pos + CHAIN_MIN_LOOKAHEAD >= end_pos) {
        // Handle trailing literals and return
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = 0; i < trailing; i++)
                s.lit_buf[s.lit_count + i] = src[anchor + i];
            s.lit_count += trailing;
            emitCmd(s, trailing, 0, 0,
                    anchor, block2_start, recent_offset);
            anchor = end_pos;
        }
        return;
    }

    while (pos + CHAIN_MIN_LOOKAHEAD < end_pos) {
        // Find a match at current position
        uint32_t cur_lit_run = pos - anchor;
        ChainMatch match = findMatchChain<HAS_DICT>(src, src_size, pos, recent_offset,
                                          first_hash, long_hash, next_hash,
                                          hash_bits, hash_mask, end_pos, cur_lit_run,
                                          dict, dict_len, dict_ht);
        if (match.length < 2) {
            pos++;
            continue;
        }

        // Lazy-1: check pos+1 for a better match
        while (pos + CHAIN_MIN_LOOKAHEAD + 1 < end_pos) {
            ChainMatch lazy1 = findMatchChain<HAS_DICT>(src, src_size, pos + 1, recent_offset,
                                              first_hash, long_hash, next_hash,
                                              hash_bits, hash_mask, end_pos, cur_lit_run + 1,
                                              dict, dict_len, dict_ht);
            if (lazy1.length >= 2 && isLazyMatchBetter(lazy1, match, 0)) {
                pos++;
                match = lazy1;
            } else {
                // Lazy-2: check pos+2 (only if current match length > 2)
                if (pos + CHAIN_MIN_LOOKAHEAD + 2 >= end_pos || match.length == 2) break;
                ChainMatch lazy2 = findMatchChain<HAS_DICT>(src, src_size, pos + 2, recent_offset,
                                                  first_hash, long_hash, next_hash,
                                                  hash_bits, hash_mask, end_pos, cur_lit_run + 2,
                                                  dict, dict_len, dict_ht);
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
        if (resolved_off > NEAR_OFFSET_MAX && (uint32_t)match.length < FAR_OFFSET_MIN_MATCH) {
            pos++;
            continue;
        }

        // Backward extension: extend match backward into literal run.
        // Mirrors CPU runChainParser's backward-extension loop
        // (src/encode/fast/fast_lz_parser.zig).
        while (pos > anchor && pos > actual_offset) {
            uint32_t prev = pos - 1;
            uint32_t back = prev - actual_offset;
            if (src[prev] != src[back]) break;
            pos--;
            match.length++;
        }

        // Compute literal run and emit with literal-1 splitting
        uint32_t lit_len = pos - anchor;
        emitWithLiteral1(src, s,
                         anchor, lit_len, (uint32_t)match.length, off_param,
                         block2_start, recent_offset);

        // Update recent offset
        recent_offset = -(int32_t)actual_offset;

        // Advance past the match
        uint32_t match_end = pos + (uint32_t)match.length;

        // Insert matched range positions into hash tables. The outer
        // guard already asserts `match_end + 8 <= src_size`, so the
        // insert range ends at match_end without further clamping.
        if (match_end > pos + 1 && match_end + 8 <= src_size) {
            insertChainRange(src, src_size, pos + 1, match_end, first_hash, long_hash, next_hash,
                             hash_bits, hash_mask);
        }

        anchor = match_end;
        pos = match_end;
    }

    // Trailing literals at end_pos
    {
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = 0; i < trailing; i++)
                s.lit_buf[s.lit_count + i] = src[anchor + i];
            s.lit_count += trailing;

            emitCmd(s, trailing, 0, 0,
                    anchor, block2_start, recent_offset);

            anchor = end_pos;
        }
    }
}
