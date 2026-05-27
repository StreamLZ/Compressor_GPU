// ── HARNESS V2: warp-cooperative chain parser ──────────────────
// Distributes the 4 independent candidate probes in findMatchChain
// across lanes 0-3 instead of running all on lane 0. Chain walk stays
// serial on lane 0; long_hash / offset-8 / recent_offset go on lanes
// 1-3 in parallel. Result reduction via shfl preserves the serial
// order so output bytes stay deterministic (fingerprint MUST match
// the baseline at the same hash_bits).
//
// scanBlockChain control flow stays uniform across all 32 lanes; only
// lane 0 mutates pos / anchor / recent_offset, and broadcasts via
// __shfl_sync at each step.
//
// Enable: -DHARNESS_WARP_CHAIN=1
#pragma once

#include "lz_format.cuh"
#include "lz_token_emit.cuh"
#include "lz_chain_parser.cuh"  // for isMatchBetter, isBetterThanRecentMatch, isLazyMatchBetter, insertChainRange

// 8-byte XOR forward extension helper (was V1, kept here for warp parser use).
__device__ __forceinline__ uint32_t matchExtendFast(
    const uint8_t* src, uint32_t pos, uint32_t ref,
    uint32_t start_ml, uint32_t max_ext
) {
    uint32_t ml = start_ml;
    while (ml + 8 <= max_ext) {
        uint64_t a = readU64LE(src + pos + ml);
        uint64_t b = readU64LE(src + ref + ml);
        uint64_t x = a ^ b;
        if (x != 0) {
            ml += (uint32_t)((__ffsll((long long)x) - 1) >> 3);
            return ml;
        }
        ml += 8;
    }
    while (ml < max_ext && src[pos + ml] == src[ref + ml]) ml++;
    return ml;
}

// ── findMatchChain_warp ──────────────────────────────────────────
// All 32 lanes call this with the same (pos, recent_offset, ...).
// Returns the same ChainMatch to all lanes via shfl broadcast.
__device__ ChainMatch findMatchChain_warp(
    const uint8_t* src, uint32_t src_size,
    uint32_t pos,
    int32_t recent_offset,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    uint32_t end_pos,
    uint32_t lit_run_length
) {
    const uint32_t lane = threadIdx.x & LANE_MASK;

    // Phase 0: uniform hash computations (every lane does the same)
    uint32_t bytes_at_pos = readU32LE(src + pos);
    uint64_t at_src = read8safe(src, pos, src_size);
    uint32_t ha = hashTableA(hash_bits, hash_mask, at_src);
    uint32_t hb = hashTableB(hash_bits, hash_mask, at_src);
    uint32_t hb_tag = hashTagB(at_src);

    // Phase 1: each lane 0-3 runs ONE probe; lanes 4+ idle.
    // Each lane produces (cand_len, cand_off, [extras]) in its own
    // registers; we reduce later.
    int32_t my_len = 0;
    int32_t my_off = 0;
    int32_t my_recent_match_length = 0;  // only meaningful on lane 3 (recent path)
    bool    my_full_recent_match = false; // only lane 3
    uint32_t my_recent_ref = 0;           // only lane 3

    // ──────────── Lane 3: recent_offset check ────────────
    // Mirrors lz_chain_parser.cuh section (a). On full 4-byte match,
    // this would early-return in the serial version — here we just
    // record a candidate with offset=0 (recent reuse) and let the
    // tie-breaker pick it.
    if (lane == 3) {
        uint32_t recent_ref = (uint32_t)((int32_t)pos + recent_offset);
        my_recent_ref = recent_ref;
        if (recent_ref < pos) {
            uint32_t recent_word = readU32LE(src + recent_ref);
            uint32_t xor_val = bytes_at_pos ^ recent_word;
            if (xor_val == 0) {
                // Full match — extend
                uint32_t max_ext = end_pos - pos;
                uint32_t ml = matchExtendFast(src, pos, recent_ref, 4, max_ext);
                my_len = (int32_t)ml;
                my_off = 0;  // recent reuse
                my_full_recent_match = true;
                my_recent_match_length = (int32_t)ml;
            } else {
                my_recent_match_length = (int32_t)(__ffs(xor_val) - 1) / (int32_t)BITS_PER_BYTE;
            }
        }
    }

    // Broadcast recent_match_length from lane 3 so all lanes can apply
    // the minimum_match_length bump uniformly.
    int32_t recent_match_length = __shfl_sync(0xffffffff, my_recent_match_length, 3);
    int32_t minimum_match_length = (int32_t)MIN_MATCH;
    if (lit_run_length >= LONG_LIT_RUN_THRESHOLD) {
        if (recent_match_length < 3) recent_match_length = 0;
        minimum_match_length += 1;
    }

    // ──────────── Lane 0: walk first_hash chain ──────────
    // Same logic as section (b) in lz_chain_parser.cuh. Serial; lane 0
    // only. The other lanes are doing their own probes concurrently.
    if (lane == 0) {
        int32_t best_match_length = 0;
        int32_t best_offset = 0;
        uint32_t head = first_hash[ha];
        uint32_t candidate_offset = pos - head;
        if (candidate_offset <= NEAR_OFFSET_MAX && candidate_offset != 0) {
            uint32_t chain_steps = CHAIN_MAX_STEPS;
            uint32_t hash_value = head;
            while (candidate_offset < LZ_BLOCK_SIZE) {
                if (candidate_offset > MIN_HASH_MATCH_OFFSET) {
                    if (candidate_offset <= pos) {
                        uint32_t ref = pos - candidate_offset;
                        uint32_t ref_word = readU32LE(src + ref);
                        if (ref_word == bytes_at_pos) {
                            bool quick_ok = true;
                            if (best_match_length >= 4) {
                                uint32_t tail_pos = pos + (uint32_t)best_match_length;
                                if (tail_pos >= end_pos) quick_ok = false;
                                else if (src[tail_pos] != src[tail_pos - candidate_offset]) quick_ok = false;
                            }
                            if (quick_ok) {
                                uint32_t max_ext = end_pos - pos;
                                uint32_t ml = matchExtendFast(src, pos, ref, 4, max_ext);
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
                candidate_offset = (uint16_t)(pos - hash_value);
                if (candidate_offset <= previous_offset) break;
            }
        }
        my_len = best_match_length;
        my_off = best_offset;
    }

    // ──────────── Lane 1: long_hash probe ────────────────
    if (lane == 1) {
        uint32_t lh_value = long_hash[hb];
        if (((hb_tag ^ lh_value) & LONG_HASH_TAG_MASK) == 0) {
            uint32_t cand_pos = lh_value >> LONG_HASH_TAG_BITS;
            if (cand_pos < pos) {
                uint32_t cand_off = pos - cand_pos;
                if (cand_off >= MIN_HASH_MATCH_OFFSET && cand_off <= NEAR_OFFSET_MAX) {
                    uint32_t ref = pos - cand_off;
                    uint32_t ref_word = readU32LE(src + ref);
                    if (ref_word == bytes_at_pos) {
                        // No best_match_length tail-byte check here — we
                        // don't know lane 0's best yet. The tie-break
                        // reduction below uses isMatchBetter so we still
                        // produce the same winner the serial path would.
                        uint32_t max_ext = end_pos - pos;
                        uint32_t ml = matchExtendFast(src, pos, ref, 4, max_ext);
                        int32_t cand_len = (int32_t)ml;
                        if (cand_len >= minimum_match_length) {
                            my_len = cand_len;
                            my_off = (int32_t)cand_off;
                        }
                    }
                }
            }
        }
    }

    // ──────────── Lane 2: offset-8 fallback ──────────────
    if (lane == 2) {
        if (pos >= MIN_HASH_MATCH_OFFSET) {
            uint32_t ref = pos - MIN_HASH_MATCH_OFFSET;
            uint32_t ref_word = readU32LE(src + ref);
            if (ref_word == bytes_at_pos) {
                uint32_t max_ext = end_pos - pos;
                uint32_t ml = matchExtendFast(src, pos, ref, 4, max_ext);
                int32_t cand_len = (int32_t)ml;
                if (cand_len >= minimum_match_length) {
                    my_len = cand_len;
                    my_off = (int32_t)MIN_HASH_MATCH_OFFSET;
                }
            }
        }
    }

    __syncwarp();

    // Phase 2: ordered reduction preserving serial tie-break.
    // Serial: best = chain; if isMatchBetter(long_hash, best) best = long_hash;
    //          if offset8 satisfies "cand >= best.len AND >= min" then take it.
    // Recent is compared via isBetterThanRecentMatch at the very end.
    int32_t l0_len = __shfl_sync(0xffffffff, my_len, 0);
    int32_t l0_off = __shfl_sync(0xffffffff, my_off, 0);
    int32_t l1_len = __shfl_sync(0xffffffff, my_len, 1);
    int32_t l1_off = __shfl_sync(0xffffffff, my_off, 1);
    int32_t l2_len = __shfl_sync(0xffffffff, my_len, 2);
    int32_t l2_off = __shfl_sync(0xffffffff, my_off, 2);
    int32_t l3_len = __shfl_sync(0xffffffff, my_len, 3);
    int32_t l3_off = __shfl_sync(0xffffffff, my_off, 3);
    bool    l3_full = __shfl_sync(0xffffffff, (int)my_full_recent_match, 3) != 0;

    int32_t best_len = l0_len;
    int32_t best_off = l0_off;
    // long_hash: only swap if isMatchBetter
    if (l1_len > 0 && isMatchBetter(l1_len, l1_off, best_len, best_off)) {
        best_len = l1_len;
        best_off = l1_off;
    }
    // offset-8 fallback: simple "cand_len >= best_len" rule from the serial code
    if (l2_len > 0 && l2_len >= best_len) {
        best_len = l2_len;
        best_off = l2_off;
    }

    // Phase 3: hash table insert (lane 0 only — matches serial order)
    if (lane == 0) {
        uint32_t prev_head = first_hash[ha];
        next_hash[pos & NEXT_HASH_INDEX_MASK] = (uint16_t)(prev_head & NEXT_HASH_INDEX_MASK);
        first_hash[ha] = pos;
        long_hash[hb] = (hb_tag & LONG_HASH_TAG_MASK) | (pos << LONG_HASH_TAG_BITS);
    }
    // The serial path has an EARLY return on a full recent match BEFORE
    // the long_hash / offset-8 / chain walks even fire (and BEFORE the
    // hash-table insert). We must reproduce that: if lane 3 saw a full
    // recent match, the result is (recent_len, 0) and the OTHER probes
    // are discarded. To make this fingerprint-equivalent we also need
    // to AVOID inserting pos into the hash tables in that case — the
    // serial code wrote to first_hash/next_hash/long_hash *inside* the
    // full-recent branch (its own write set) and then returned. So the
    // write pattern is actually the same. (See lz_chain_parser.cuh
    // lines 113-121 vs 254-260.)
    // No behavioral fix needed; the writes happen in both cases.

    // Phase 4: final recent vs hash match decision (mirrors serial return)
    if (l3_full) {
        // Serial path would have returned (recent_len, 0) early.
        return { l3_len, 0 };
    }
    if (best_off == 0 || !isBetterThanRecentMatch(recent_match_length, best_len, best_off)) {
        return { recent_match_length, 0 };
    }
    return { best_len, best_off };
}

// ── scanBlockChain_warp: warp-cooperative top-level scanner ─────
// All 32 lanes stay alive throughout. State (pos, anchor, recent_offset)
// lives in lane 0; broadcast at decision points.
__device__ void scanBlockChain_warp(
    const uint8_t* src, uint32_t src_size,
    uint32_t* first_hash, uint32_t* long_hash, uint16_t* next_hash,
    uint32_t hash_bits, uint32_t hash_mask,
    OutputStreams &s,
    uint32_t &anchor, int32_t &recent_offset,
    uint32_t start_pos, uint32_t end_pos, uint32_t block2_start
) {
    const uint32_t lane = threadIdx.x & LANE_MASK;
    uint32_t pos = start_pos;

    // Trailing-literals-only path (uniform control flow on all lanes).
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
        // Broadcast anchor + recent_offset (lane 0 owns them).
        uint32_t anchor_u = __shfl_sync(0xffffffff, anchor, 0);
        int32_t  rec_u    = __shfl_sync(0xffffffff, recent_offset, 0);
        uint32_t cur_lit_run = pos - anchor_u;

        ChainMatch match = findMatchChain_warp(src, src_size, pos, rec_u,
                                               first_hash, long_hash, next_hash,
                                               hash_bits, hash_mask, end_pos, cur_lit_run);
        if (match.length < 2) {
            pos++;
            continue;
        }

        // Lazy-1 / Lazy-2 (uniform control flow — every lane goes around
        // this loop the same number of times, driven by per-iteration
        // findMatchChain_warp results).
#ifndef HARNESS_DISABLE_LAZY
        while (pos + CHAIN_MIN_LOOKAHEAD + 1 < end_pos) {
            ChainMatch lazy1 = findMatchChain_warp(src, src_size, pos + 1, rec_u,
                                                   first_hash, long_hash, next_hash,
                                                   hash_bits, hash_mask, end_pos, cur_lit_run + 1);
            if (lazy1.length >= 2 && isLazyMatchBetter(lazy1, match, 0)) {
                pos++;
                match = lazy1;
            } else {
                if (pos + CHAIN_MIN_LOOKAHEAD + 2 >= end_pos || match.length == 2) break;
                ChainMatch lazy2 = findMatchChain_warp(src, src_size, pos + 2, rec_u,
                                                       first_hash, long_hash, next_hash,
                                                       hash_bits, hash_mask, end_pos, cur_lit_run + 2);
                if (lazy2.length >= 2 && isLazyMatchBetter(lazy2, match, 1)) {
                    pos += 2;
                    match = lazy2;
                } else break;
            }
        }
#endif

        // Resolve actual offset for backward extension
        uint32_t actual_offset;
        if (match.offset == 0) actual_offset = (uint32_t)(-rec_u);
        else                   actual_offset = (uint32_t)match.offset;

        uint32_t off_param = (uint32_t)match.offset;
        uint32_t resolved_off = off_param;
        if (resolved_off == 0) resolved_off = (uint32_t)(-rec_u);
        if (resolved_off > NEAR_OFFSET_MAX && (uint32_t)match.length < FAR_OFFSET_MIN_MATCH) {
            pos++;
            continue;
        }

        // Backward extension — mutates pos, no other lane state. Run on
        // lane 0 only via broadcast at the end. (All lanes follow the
        // same loop count via uniform pos.)
        if (lane == 0) {
            while (pos > anchor && pos > actual_offset) {
                uint32_t prev = pos - 1;
                uint32_t back = prev - actual_offset;
                if (src[prev] != src[back]) break;
                pos--;
                match.length++;
            }
        }
        // After backward extension, re-broadcast pos so all lanes agree.
        pos = __shfl_sync(0xffffffff, pos, 0);
        match.length = __shfl_sync(0xffffffff, match.length, 0);

        if (lane == 0) {
            uint32_t lit_len = pos - anchor;
            emitWithLiteral1(src, s, anchor, lit_len, (uint32_t)match.length, off_param,
                             block2_start, recent_offset);
            recent_offset = -(int32_t)actual_offset;
        }

        uint32_t match_end = pos + (uint32_t)match.length;

#ifndef HARNESS_DISABLE_INSERT_RANGE
        if (match_end > pos + 1 && match_end + 8 <= src_size) {
            // insertChainRange is on lane 0 only (matches the serial path).
            // Future: parallelize across lanes by striding `from` by lane.
            if (lane == 0) {
                insertChainRange(src, src_size, pos + 1, match_end,
                                 first_hash, long_hash, next_hash,
                                 hash_bits, hash_mask);
            }
        }
#endif

        if (lane == 0) {
            anchor = match_end;
        }
        pos = match_end;
    }

    // Trailing literals after the loop
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
