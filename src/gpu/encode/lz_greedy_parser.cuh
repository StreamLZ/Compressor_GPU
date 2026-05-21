// ── StreamLZ GPU LZ encode — warp-parallel greedy parser ────────
// The use_chain=0 parser: a warp-parallel greedy match scan. All 32
// lanes probe the hash table in lockstep and simulate the CPU's
// serial hash-write order. Port of the CPU runGreedyParser.
//
// Included by lz_kernel.cu — see that file for the build line.
#pragma once

#include "lz_format.cuh"
#include "lz_token_emit.cuh"

// ── greedy-parser tuning ────────────────────────────────────────
static constexpr uint32_t HASH_LOOKAHEAD = 7;  // bytes lane K needs from lanes K+1..K+7 for the 8-byte hash
static constexpr uint32_t XOR_PATTERN_BYTES123_MASK = 0xFFFFFF00u;  // isolates bytes 1..3 for the XOR check
static constexpr uint32_t XOR_MATCH_MIN_LEN = 3;  // bytes guaranteed by a passing XOR pattern check
static constexpr uint32_t NO_MATCH_STEP_THRESHOLD = 128;  // literal-run length above which the scan strides
static constexpr uint32_t NO_MATCH_STEP_SHIFT = 7;        // dist >> 7 component of the adaptive step
static constexpr uint32_t NO_MATCH_STEP_MAX = 16;         // cap on the adaptive no-match step

// ── scanBlock: warp-parallel greedy scan for one block ──────────
// Scans positions [start_pos .. end_pos), finding matches, extending
// them (capped at end_pos), and emitting tokens + trailing literals.
// All stream cursors live in OutputStreams and accumulate across calls.
// enable_match_rehash inserts hash entries inside the just-emitted
// match (the driver's L4-and-up behavior).
__device__ void scanBlock(
    const uint8_t* src, uint32_t src_size,
    uint32_t* ht, uint32_t hash_bits, uint32_t hash_mask,
    OutputStreams &s,
    uint32_t &anchor, int32_t &recent_offset,
    uint32_t start_pos, uint32_t end_pos, uint32_t block2_start,
    uint32_t enable_match_rehash
) {
    const uint32_t lane = threadIdx.x & LANE_MASK;
    uint32_t pos = start_pos;

    while (pos + MIN_MATCH <= end_pos) {
        uint32_t my_pos = pos + lane;
        uint32_t my_byte = (my_pos < src_size) ? (uint32_t)src[my_pos] : 0u;

        // 8-byte lookahead via shfl_down. Matches CPU's runGreedyParser
        // which reads a u64 word for hashing. Note key4 = low 32 bits.
        uint32_t b1 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 1);
        uint32_t b2 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 2);
        uint32_t b3 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 3);
        uint32_t b4 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 4);
        uint32_t b5 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 5);
        uint32_t b6 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 6);
        uint32_t b7 = __shfl_down_sync(FULL_WARP_MASK, my_byte, 7);
        uint32_t key4 = my_byte | (b1 << 8) | (b2 << 16) | (b3 << 24);
        uint64_t key8 = (uint64_t)key4
                      | ((uint64_t)b4 << 32) | ((uint64_t)b5 << 40)
                      | ((uint64_t)b6 << 48) | ((uint64_t)b7 << 56);

        uint32_t remaining = end_pos - pos;
        uint32_t active_count;
        // 8-byte hash needs HASH_LOOKAHEAD-byte lookahead (lane K needs
        // lanes K+1..K+7). active_count = WARP_SIZE - HASH_LOOKAHEAD = 25.
        if (remaining >= WARP_SIZE) active_count = WARP_SIZE - HASH_LOOKAHEAD;
        else if (remaining >= HASH_LOOKAHEAD + 1) active_count = remaining - HASH_LOOKAHEAD;
        else active_count = 0;
        bool is_active = (lane < active_count);

        // CPU uses k=6 hash on text inputs (the common case for -gpu L1).
        // Hashes 8 bytes with Fibonacci multiplier << 16, picking buckets
        // by 6-byte sequences instead of 4-byte. Lower collision rate on
        // text → fewer false-positive hash hits that only extend 4 bytes.
        uint32_t h = hashKey6(key8, hash_bits, hash_mask);

        // ── Simulate CPU's serial hash-write order within the warp ─────
        // CPU at position P does: read ht[h(P)] (returns LAST write to
        // bucket h(P) before P), then write ht[h(P)] = P. So for each lane
        // K, the value CPU would have seen at ht[h(K)] is:
        //   - if any lower active lane K' < K has h(K') == h(K), then
        //     K' (most recent intra-warp same-bucket lane) overwrote the
        //     pre-warp value. The HIGHEST such K' is what CPU sees.
        //     Content match: only if K'.key4 == K.key4 (and K-K' >= 8).
        //   - otherwise, pre-warp ht[h(K)] is what CPU sees.
        //
        // All warp-wide intrinsics here are unconditional with the
        // FULL_WARP_MASK; per-lane logic uses purely-local mask arithmetic.
        uint32_t bucket_same = __match_any_sync(FULL_WARP_MASK, h);
        uint32_t key_same    = __match_any_sync(FULL_WARP_MASK, key4);
        uint32_t active_mask = __ballot_sync(FULL_WARP_MASK, is_active);
        uint32_t lower_same_bucket = bucket_same & active_mask & ((1u << lane) - 1);

        bool hash_match = false;
        uint32_t hash_ref = 0;

        if (is_active) {
            if (lower_same_bucket != 0) {
                // CPU would see the highest lower same-bucket lane's pos.
                // That lane's key4 is "same as me" iff its bit is in key_same.
                uint32_t highest_lower = (WARP_SIZE - 1) - __clz(lower_same_bucket);
                bool same_key = (key_same & (1u << highest_lower)) != 0;
                uint32_t their_pos = pos + highest_lower;
                if (same_key && their_pos + 8 <= my_pos) {
                    hash_match = true;
                    hash_ref = their_pos;
                }
                // else: CPU's content check fails (different key OR
                // offset < 8) — no match.
            } else {
                // No intra-warp overwrite — pre-warp ht[h] is what CPU sees.
                uint32_t ref_val = ht[h];
                if (ref_val != HASH_EMPTY && ref_val + 8 <= my_pos) {
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
        }

        // ── CPU secondary match attempts ──────────────────────────────
        // After a hash miss, CPU runGreedyParser tries (in order):
        //   (1) 1-byte XOR pattern: if bytes 1..3 of cursor match bytes 1..3
        //       of cursor + recent_offset → match at cursor+1 with len ≥ 3,
        //       offset = recent_offset (use_recent flag at decode).
        //   (2) -8 fixed offset: if 4 bytes at cursor match 4 bytes at
        //       cursor - 8 → match at cursor with offset = 8.
        // Each lane checks both independently. Match-type is tracked so we
        // can dispatch correctly when a lane wins the ballot.
        // Encoding for my_match_type: 0 = hash, 1 = xor (cursor+1), 2 = -8.
        bool xor_match = false;
        bool eight_match = false;
        if (is_active && !hash_match && recent_offset < 0) {
            uint32_t ro_dist = (uint32_t)(-recent_offset);
            if (my_pos >= ro_dist) {
                uint32_t ref = my_pos - ro_dist;
                uint32_t recent_word = (uint32_t)src[ref] |
                                      ((uint32_t)src[ref+1] << 8) |
                                      ((uint32_t)src[ref+2] << 16) |
                                      ((uint32_t)src[ref+3] << 24);
                uint32_t xor_v = key4 ^ recent_word;
                if ((xor_v & XOR_PATTERN_BYTES123_MASK) == 0) {
                    // Bytes 1..3 of cursor == bytes 1..3 of cursor+recent_offset.
                    // Match starts at cursor+1, length ≥ 3, offset = recent.
                    // Also need cursor+1+3 ≤ end_pos to safely read the match.
                    if (my_pos + 1 + XOR_MATCH_MIN_LEN <= end_pos) {
                        xor_match = true;
                    }
                }
            }
        }
        if (is_active && !hash_match && !xor_match && my_pos >= MIN_HASH_MATCH_OFFSET) {
            uint32_t eight_ref = my_pos - MIN_HASH_MATCH_OFFSET;
            uint32_t eight_word = (uint32_t)src[eight_ref] |
                                 ((uint32_t)src[eight_ref+1] << 8) |
                                 ((uint32_t)src[eight_ref+2] << 16) |
                                 ((uint32_t)src[eight_ref+3] << 24);
            if (eight_word == key4) eight_match = true;
        }

        bool has_match = hash_match || xor_match || eight_match;
        // Match type for dispatch:
        //   0 = hash    (hash_ref)
        //   1 = xor     (cursor+1, offset=recent)
        //   2 = eight   (offset=8)
        // (Intra-warp same-key cases are folded into hash_match by the
        // unified serial-order simulation above.)
        uint32_t my_match_type = hash_match  ? 0u
                               : xor_match   ? 1u
                               : eight_match ? 2u
                               : 0u;
        uint32_t my_ref = hash_ref;
        uint32_t match_ballot = __ballot_sync(FULL_WARP_MASK, has_match);

        // CPU runGreedyParser advances by an adaptive step on no match:
        //   step = (dist < 128) ? 1 : min((dist >> 7) + 1, 16)
        // where dist = source_cursor - literal_start. When matches happen,
        // step is always 1 (every position gets a hash probe). Only when
        // no match is found does CPU stride forward, accelerating when
        // the literal run is already long.
        uint32_t no_match_step = 1;
        if (match_ballot == 0) {
            uint32_t dist = pos - anchor;
            if (dist >= NO_MATCH_STEP_THRESHOLD) {
                uint32_t step = (dist >> NO_MATCH_STEP_SHIFT) + 1;
                no_match_step = (step > NO_MATCH_STEP_MAX) ? NO_MATCH_STEP_MAX : step;
            }
        }

        // Hash-table writes mirror CPU: write the bucket for every position
        // CPU would have visited (= lane positions up to and including the
        // first-match lane, or up to no_match_step-1 if no match).
        {
            uint32_t write_limit = (match_ballot != 0)
                ? (uint32_t)(__ffs(match_ballot) - 1)
                : (no_match_step - 1);
            if (is_active && lane <= write_limit) {
                ht[h] = my_pos;
            }
        }

        if (match_ballot == 0) {
            pos += no_match_step;
            continue;
        }

        uint32_t first_lane = __ffs(match_ballot) - 1;
        uint32_t winning_type = __shfl_sync(FULL_WARP_MASK, my_match_type, first_lane);
        uint32_t match_pos;
        uint32_t match_ref;
        uint32_t min_match_len;

        if (winning_type == 0) {
            // Hash match (includes intra-warp same-key — unified above).
            match_pos = pos + first_lane;
            match_ref = __shfl_sync(FULL_WARP_MASK, my_ref, first_lane);
            min_match_len = MIN_MATCH;
        } else if (winning_type == 1) {
            // XOR recent-offset: match starts at first_lane + 1,
            // ref at match_pos + recent_offset (= match_pos - |recent_offset|).
            match_pos = pos + first_lane + 1;
            match_ref = match_pos - (uint32_t)(-recent_offset);
            min_match_len = XOR_MATCH_MIN_LEN;  // 3 bytes guaranteed by xor check
        } else {
            // -8 fixed offset: match starts at first_lane with offset 8.
            match_pos = pos + first_lane;
            match_ref = match_pos - MIN_HASH_MATCH_OFFSET;
            min_match_len = MIN_MATCH;
        }

        // Match extension from min_match_len, capped at end_pos.
        uint32_t match_len = min_match_len;
        uint32_t max_match = end_pos - match_pos;
        bool ext_found = false;
        for (uint32_t ext = min_match_len; ext < max_match; ext += WARP_SIZE) {
            uint32_t check = ext + lane;
            bool mm;
            if (check >= max_match)
                mm = true;
            else
                mm = (src[match_pos + check] != src[match_ref + check]);
            uint32_t mm_mask = __ballot_sync(FULL_WARP_MASK, mm);
            if (mm_mask != 0) {
                match_len = ext + __ffs(mm_mask) - 1;
                ext_found = true;
                break;
            }
        }
        if (!ext_found) match_len = max_match;

        {
            int32_t neg_off = -(int32_t)(match_pos - match_ref);
            uint32_t off_param = (neg_off == recent_offset) ? 0 : (uint32_t)(-neg_off);

            // Resolve actual offset for fast-path decision
            uint32_t resolved_off = off_param;
            if (resolved_off == 0) resolved_off = (uint32_t)(-recent_offset);

            // Enforce minimum match length for far offsets (CPU mmlt = 14)
            if (resolved_off > NEAR_OFFSET_MAX && match_len < FAR_OFFSET_MIN_MATCH) {
                pos = match_pos + 1;
                continue;
            }

            // ── Backward extension ──
            // Walk match_pos / match_ref backward while bytes still equal.
            // Reduces literal-run length and grows the match, often
            // collapsing two tokens (literal-bearing + match) into one.
            // Lane-0 serial; cheap because matches typically extend
            // back only a few bytes.
            uint32_t bw_steps = 0;
            if (lane == 0) {
                uint32_t mp = match_pos;
                uint32_t mr = match_ref;
                while (mp > anchor && mr > 0 && src[mp - 1] == src[mr - 1]) {
                    mp--;
                    mr--;
                    bw_steps++;
                }
            }
            bw_steps = __shfl_sync(FULL_WARP_MASK, bw_steps, 0);
            match_pos -= bw_steps;
            match_ref -= bw_steps;
            match_len += bw_steps;

            uint32_t lit_len = match_pos - anchor;
            for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
                s.lit_buf[s.lit_count + i] = src[anchor + i];
            s.lit_count += lit_len;

            if (lit_len <= MAX_INLINE_LITERALS && match_len <= MAX_INLINE_MATCH && resolved_off <= NEAR_OFFSET_MAX) {
                if (lane == 0) {
                    uint8_t token = (uint8_t)(lit_len + TOKEN_MATCH_SHIFT_MUL * match_len);
                    if (off_param == 0) token |= TOKEN_RECENT_FLAG;
                    s.token_buf[s.token_count++] = token;
                    if (off_param != 0) {
                        storeU16LE(s.off16_buf + s.off16_count * 2, (uint16_t)off_param);
                        s.off16_count++;
                    }
                }
            } else {
                if (lane == 0) {
                    emitCmd(s, lit_len, match_len, off_param,
                            anchor, block2_start, recent_offset);
                }
            }

            recent_offset = neg_off;
        }

        // enable_match_rehash: match-range rehash. Insert hash-table
        // entries for positions *inside* the just-emitted match at
        // exponentially spaced offsets, so later positions can
        // reference mid-match. Port of CPU emitMatch's level>=2 rehash
        // (fast_lz_parser.zig runGreedyParser).
        if (enable_match_rehash) {
            uint32_t ri = 1u << lane;
            if (ri < match_len) {
                uint32_t rp = match_pos + ri;
                if (rp + 8 <= src_size) {
                    uint64_t rk = (uint64_t)src[rp]
                                | ((uint64_t)src[rp+1] << 8)
                                | ((uint64_t)src[rp+2] << 16)
                                | ((uint64_t)src[rp+3] << 24)
                                | ((uint64_t)src[rp+4] << 32)
                                | ((uint64_t)src[rp+5] << 40)
                                | ((uint64_t)src[rp+6] << 48)
                                | ((uint64_t)src[rp+7] << 56);
                    ht[hashKey6(rk, hash_bits, hash_mask)] = rp;
                }
            }
        }

        anchor = match_pos + match_len;
        pos = anchor;
    }

    // Trailing literals up to end_pos
    {
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = lane; i < trailing; i += WARP_SIZE)
                s.lit_buf[s.lit_count + i] = src[anchor + i];
            s.lit_count += trailing;

            if (lane == 0) {
                emitCmd(s, trailing, 0, 0,
                        anchor, block2_start, recent_offset);
            }

            anchor = end_pos;
        }
    }
}
