// ── HARNESS V3 (hybrid): warp-parallel scan + chain investigation ──
//
// Strategy:
//   • Warp scans 25 consecutive positions in parallel, probing first_hash
//     for each. Identifies the EARLIEST position with a plausible match
//     (offset valid + 4-byte ref_word equals key) via __ballot_sync.
//   • For positions before that match: emitted as literals (no chain work).
//   • At the match position: lane 0 runs full chain investigation (chain
//     walk + lazy-1/lazy-2 + backward extension), preserving L5 chain
//     semantics for the match itself.
//   • If no match in window: skip 25 positions.
//
// Goal: pay greedy cost (~30 cycles/position via warp) for no-match
// scanning, pay chain cost only for match positions (which are minority).
//
// Output bytes WILL differ from pure chain — the "earliest match in
// window" rule replaces chain's per-position recent+long+chain decision.
// Ratio should be between greedy (47.1%) and chain (40.86%).
//
// Enable: -DHARNESS_HYBRID_CHAIN=1
#pragma once

#include "lz_format.cuh"
#include "lz_token_emit.cuh"
#include "lz_chain_parser.cuh"  // findMatchChain, insertChainRange, isLazyMatchBetter, etc

// Warp scan: each active lane checks ALL 4 chain-style probes at
// (base_pos + lane) to mirror chain's match-finding coverage:
//   • first_hash
//   • long_hash
//   • offset-8 fallback
//   • recent_offset
// Returns has_match if ANY probe finds a 4-byte match. This is more
// expensive per-lane than the first_hash-only variant but recovers
// most of the matches chain would find (so ratio approaches chain's).
__device__ __forceinline__ void hybrid_warp_probe_full(
    const uint8_t* src, uint32_t src_size,
    uint32_t base_pos, uint32_t end_pos,
    int32_t recent_offset,
    uint32_t* first_hash, uint32_t* long_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    bool &out_active, bool &out_match
) {
    const uint32_t lane = threadIdx.x & LANE_MASK;
    uint32_t my_pos = base_pos + lane;
    out_active = (my_pos + CHAIN_MIN_LOOKAHEAD < end_pos) && (lane < 25);
    out_match = false;
    if (!out_active) return;

    uint32_t my_word = readU32LE(src + my_pos);
    uint64_t at = read8safe(src, my_pos, src_size);

    // (a) first_hash probe
    {
        uint32_t ha = hashTableA(hash_bits, hash_mask, at);
        uint32_t ref = first_hash[ha];
        if (ref < my_pos) {
            uint32_t cand_off = my_pos - ref;
            if (cand_off >= MIN_HASH_MATCH_OFFSET && cand_off <= NEAR_OFFSET_MAX) {
                uint32_t ref_word = readU32LE(src + ref);
                if (ref_word == my_word) { out_match = true; return; }
            }
        }
    }
    // (b) long_hash probe
    {
        uint32_t hb = hashTableB(hash_bits, hash_mask, at);
        uint32_t hb_tag = hashTagB(at);
        uint32_t lh_value = long_hash[hb];
        if (((hb_tag ^ lh_value) & LONG_HASH_TAG_MASK) == 0) {
            uint32_t cand_pos = lh_value >> LONG_HASH_TAG_BITS;
            if (cand_pos < my_pos) {
                uint32_t cand_off = my_pos - cand_pos;
                if (cand_off >= MIN_HASH_MATCH_OFFSET && cand_off <= NEAR_OFFSET_MAX) {
                    uint32_t ref = my_pos - cand_off;
                    uint32_t ref_word = readU32LE(src + ref);
                    if (ref_word == my_word) { out_match = true; return; }
                }
            }
        }
    }
    // (c) offset-8 fallback
    if (my_pos >= MIN_HASH_MATCH_OFFSET) {
        uint32_t ref = my_pos - MIN_HASH_MATCH_OFFSET;
        uint32_t ref_word = readU32LE(src + ref);
        if (ref_word == my_word) { out_match = true; return; }
    }
    // (d) recent_offset
    {
        uint32_t recent_ref = (uint32_t)((int32_t)my_pos + recent_offset);
        if (recent_ref < my_pos) {
            uint32_t recent_word = readU32LE(src + recent_ref);
            if (recent_word == my_word) { out_match = true; return; }
        }
    }
}

// Scanner: hybrid of warp-parallel fast-skip + per-match chain investigation.
__device__ void scanBlockHybrid(
    const uint8_t* src, uint32_t src_size,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    OutputStreams &s,
    uint32_t &anchor, int32_t &recent_offset,
    uint32_t start_pos, uint32_t end_pos, uint32_t block2_start
) {
    const uint32_t lane = threadIdx.x & LANE_MASK;
    uint32_t pos = start_pos;

    if (pos + CHAIN_MIN_LOOKAHEAD >= end_pos) {
        if (lane == 0) {
            uint32_t trailing = end_pos - anchor;
            if (trailing > 0) {
                for (uint32_t i = 0; i < trailing; i++)
                    s.lit_buf[s.lit_count + i] = src[anchor + i];
                s.lit_count += trailing;
                emitCmd(s, trailing, 0, 0, anchor, block2_start, recent_offset);
                anchor = end_pos;
            }
        }
        return;
    }

    while (pos + CHAIN_MIN_LOOKAHEAD < end_pos) {
        // ── Phase 1: warp scan from pos, find earliest match position ──
        bool my_active, my_match;
        hybrid_warp_probe_full(src, src_size, pos, end_pos, recent_offset,
                               first_hash, long_hash, hash_bits, hash_mask,
                               my_active, my_match);
        uint32_t match_mask = __ballot_sync(0xffffffff, my_match);
        // (No parallel hash insertion — racy writes break the next_hash
        // linked-list chain. Per-position insertion only inside chain
        // investigation via insertChainRange.)

        if (match_mask == 0) {
            // No match in this 25-position window. Skip ahead, BUT
            // lane 0 must serially insert each skipped position into
            // the chain hash tables — otherwise chain depth degrades
            // for subsequent positions and ratio collapses.
            uint32_t skip = 25;
            uint32_t max_skip = end_pos - CHAIN_MIN_LOOKAHEAD - pos;
            if (skip > max_skip) skip = max_skip;
            if (skip == 0) break;
            if (lane == 0) {
                for (uint32_t k = 0; k < skip; k++) {
                    uint32_t p = pos + k;
                    uint64_t at = read8safe(src, p, src_size);
                    uint32_t ha = hashTableA(hash_bits, hash_mask, at);
                    uint32_t hb = hashTableB(hash_bits, hash_mask, at);
                    uint32_t hb_tag = hashTagB(at);
                    uint32_t prev_head = first_hash[ha];
                    next_hash[p & NEXT_HASH_INDEX_MASK] = (uint16_t)(prev_head & NEXT_HASH_INDEX_MASK);
                    first_hash[ha] = p;
                    long_hash[hb] = (hb_tag & LONG_HASH_TAG_MASK) | (p << LONG_HASH_TAG_BITS);
                }
            }
            pos += skip;
            continue;
        }

        uint32_t earliest_lane = __ffs(match_mask) - 1;
        uint32_t match_pos = pos + earliest_lane;

        // Insert skipped positions [pos, match_pos) into chain hash tables
        // (lane 0 only) so chain investigation at match_pos sees the
        // same hash state as pure chain would have built.
        if (lane == 0 && match_pos > pos) {
            for (uint32_t p = pos; p < match_pos; p++) {
                uint64_t at = read8safe(src, p, src_size);
                uint32_t ha = hashTableA(hash_bits, hash_mask, at);
                uint32_t hb = hashTableB(hash_bits, hash_mask, at);
                uint32_t hb_tag = hashTagB(at);
                uint32_t prev_head = first_hash[ha];
                next_hash[p & NEXT_HASH_INDEX_MASK] = (uint16_t)(prev_head & NEXT_HASH_INDEX_MASK);
                first_hash[ha] = p;
                long_hash[hb] = (hb_tag & LONG_HASH_TAG_MASK) | (p << LONG_HASH_TAG_BITS);
            }
        }

        // ── Phase 2: chain investigation at match_pos (lane 0 owns state) ──
        // Use existing findMatchChain (the serial lane-0 version).
        if (lane == 0) {
            uint32_t cur_lit_run = match_pos - anchor;
            ChainMatch match = findMatchChain(src, src_size, match_pos, recent_offset,
                                              first_hash, long_hash, next_hash,
                                              hash_bits, hash_mask, end_pos, cur_lit_run);

            if (match.length < 2) {
                // Warp scan said "match", chain disagreed (e.g. offset class
                // rules killed it). Advance by 1 byte and retry warp scan.
                // (Lane 0 only mutates pos; we broadcast below.)
                pos = match_pos + 1;
            } else {
#ifndef HARNESS_DISABLE_LAZY
                // Lazy-1 / Lazy-2 (lane 0 only — same as serial chain)
                uint32_t lazy_pos = match_pos;
                while (lazy_pos + CHAIN_MIN_LOOKAHEAD + 1 < end_pos) {
                    ChainMatch lazy1 = findMatchChain(src, src_size, lazy_pos + 1, recent_offset,
                                                     first_hash, long_hash, next_hash,
                                                     hash_bits, hash_mask, end_pos, cur_lit_run + 1);
                    if (lazy1.length >= 2 && isLazyMatchBetter(lazy1, match, 0)) {
                        lazy_pos++;
                        match = lazy1;
                    } else {
                        if (lazy_pos + CHAIN_MIN_LOOKAHEAD + 2 >= end_pos || match.length == 2) break;
                        ChainMatch lazy2 = findMatchChain(src, src_size, lazy_pos + 2, recent_offset,
                                                         first_hash, long_hash, next_hash,
                                                         hash_bits, hash_mask, end_pos, cur_lit_run + 2);
                        if (lazy2.length >= 2 && isLazyMatchBetter(lazy2, match, 1)) {
                            lazy_pos += 2;
                            match = lazy2;
                        } else break;
                    }
                }
                match_pos = lazy_pos;
                cur_lit_run = match_pos - anchor;
#endif

                uint32_t actual_offset;
                if (match.offset == 0) actual_offset = (uint32_t)(-recent_offset);
                else                   actual_offset = (uint32_t)match.offset;

                uint32_t off_param = (uint32_t)match.offset;
                uint32_t resolved_off = off_param;
                if (resolved_off == 0) resolved_off = (uint32_t)(-recent_offset);
                if (resolved_off > NEAR_OFFSET_MAX && (uint32_t)match.length < FAR_OFFSET_MIN_MATCH) {
                    pos = match_pos + 1;
                } else {
                    // Backward extension
                    while (match_pos > anchor && match_pos > actual_offset) {
                        uint32_t prev = match_pos - 1;
                        uint32_t back = prev - actual_offset;
                        if (src[prev] != src[back]) break;
                        match_pos--;
                        match.length++;
                    }

                    uint32_t lit_len = match_pos - anchor;
                    emitWithLiteral1(src, s, anchor, lit_len, (uint32_t)match.length, off_param,
                                     block2_start, recent_offset);
                    recent_offset = -(int32_t)actual_offset;

                    uint32_t match_end = match_pos + (uint32_t)match.length;

#ifndef HARNESS_DISABLE_INSERT_RANGE
                    if (match_end > match_pos + 1 && match_end + 8 <= src_size) {
                        insertChainRange(src, src_size, match_pos + 1, match_end,
                                         first_hash, long_hash, next_hash,
                                         hash_bits, hash_mask);
                    }
#endif
                    anchor = match_end;
                    pos = match_end;
                }
            }
        }
        // Broadcast updated pos from lane 0 to all lanes.
        pos = __shfl_sync(0xffffffff, pos, 0);
    }

    // Trailing literals
    if (lane == 0) {
        uint32_t trailing = end_pos - anchor;
        if (trailing > 0) {
            for (uint32_t i = 0; i < trailing; i++)
                s.lit_buf[s.lit_count + i] = src[anchor + i];
            s.lit_count += trailing;
            emitCmd(s, trailing, 0, 0, anchor, block2_start, recent_offset);
            anchor = end_pos;
        }
    }
}
