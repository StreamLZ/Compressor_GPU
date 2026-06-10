// ── StreamLZ LZ decode - raw-mode sub-chunk decoder ────────────
// One of two warp-cooperative LZ decoders (the other lives in
// lz_decode_general.cuh). Streamlined single-LZ-block path: no off32,
// no delta literals, no block split. Selected for any sub-chunk with
// off32_count == 0. Register-optimized: uses uint32 offsets instead of
// 64-bit pointers, removes redundant bounds checks and shuffles.
//
// Templated on OFF16_SPLIT so each call site dead-code-eliminates the
// unused off16 branch and the unused off16_hi/lo pointer params. Both
// instantiations are live: <false> for interleaved-u16 off16 (the
// common raw path), <true> for an entropy-coded hi/lo split off16.
//
// Lane contract: warp-cooperative. Token fields parsed on lane 0 are
// broadcast to all lanes; cmd_pos / lit_pos stay coherent because
// `cmd_pos++` and `lit_pos += lit_len` run identically on every lane,
// so no broadcast is needed for them.
//
// Included into the single lz_kernel.cu translation unit. Depends on
// lz_decode_core.cuh for warpScanU32 + warpLiteralCopy + warpMatchCopy.
#pragma once

#include "lz_decode_core.cuh"

template <bool OFF16_SPLIT>
__device__ __noinline__ void decodeSubChunkRawMode(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off16_hi, const uint8_t* __restrict__ off16_lo,
    const uint8_t* __restrict__ length_stream, uint32_t length_remaining,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t dst_offset
) {
    const int lane = threadIdx.x & LANE_MASK;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = INITIAL_RECENT_OFFSET;
    uint32_t length_offset = 0;

    // v4 #1: per-warp staging for the flat batched literal copy in the
    // PP fast path below (one slice per warp; the kernels launch
    // WARPS_PER_BLOCK warps with warp id = threadIdx.y). 512 B per
    // instantiation — the kernels used 0 B shared, so no occupancy cost.
    __shared__ uint32_t s_lit_prefix[WARPS_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t s_dst_adj[WARPS_PER_BLOCK][WARP_SIZE];

    while (cmd_pos < cmd_size) {
        // Parallel-parse fast path. 32 tokens per outer iter: coalesced
        // cmd LDG, per-lane decode, prefix-scan side-stream offsets,
        // sequential cooperative copy. Compile-time dispatch on
        // OFF16_SPLIT for the per-lane off16 load.
        {
            uint32_t remaining = cmd_size - cmd_pos;
            uint32_t batch_size = remaining < WARP_SIZE ? remaining : WARP_SIZE;
            uint8_t my_cmd = ((uint32_t)lane < batch_size) ? cmd[cmd_pos + lane] : 0;
            bool my_is_long = ((uint32_t)lane < batch_size) && (my_cmd < TOKEN_SHORT_MIN);
            uint32_t any_long = __ballot_sync(FULL_WARP_MASK, my_is_long);

            // PP-prefix truncation: a long token at lane j used to force
            // tokens 0..j-1 through the serial path one at a time — the
            // ballot saw the long token in every shifted window until it
            // was finally consumed, so a single long token serialized up
            // to 32 short tokens ahead of it. Truncate the batch to the
            // all-short prefix and PP-process it; the next window then
            // starts AT the long token and takes the serial path exactly
            // once. Inputs with deep redundancy (enwik9-class: longer
            // matches → more TOKEN_LONG_NEAR) hit this constantly;
            // enwik8/silesia rarely. The PP body already handles
            // batch_size < 32 via the `my_valid = lane < batch_size`
            // guards, so the truncated batch needs no other change.
            if (any_long != 0)
                batch_size = (uint32_t)(__ffs(any_long) - 1);

            if (batch_size > 0) {
                bool my_valid = (uint32_t)lane < batch_size;
                uint32_t my_lit_len   = my_valid ? (uint32_t)(my_cmd & TOKEN_LIT_MASK) : 0u;
                uint32_t my_match_len = my_valid ? (uint32_t)((my_cmd >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK) : 0u;
                uint32_t my_use_recent = my_valid ? (uint32_t)((my_cmd >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK) : 0u;
                uint32_t my_consumes_off16 = (my_valid && !my_use_recent) ? 1u : 0u;

                uint32_t my_off16_local, total_off16_used;
                warpScanU32(my_consumes_off16, my_off16_local, total_off16_used);

                int32_t my_match_offset = recent_offset;
                if (my_consumes_off16) {
                    uint32_t entry_idx = off16_pos + my_off16_local;
                    if (entry_idx < off16_count) {
                        uint16_t v;
                        if constexpr (OFF16_SPLIT) {
                            // Entropy-coded off16: lo/hi in separate streams.
                            v = (uint16_t)off16_lo[entry_idx] |
                                ((uint16_t)off16_hi[entry_idx] << 8);
                        } else {
                            v = readU16LE(off16_raw + entry_idx * OFF16_ENTRY_BYTES);
                        }
                        my_match_offset = -(int32_t)v;
                    }
                }

                uint32_t fresh_mask = __ballot_sync(FULL_WARP_MASK, my_consumes_off16 != 0);
                // Compute src_lane on every lane (must call __shfl_sync uniformly).
                // `(1u << (lane + 1)) - 1u` would be undefined when lane == 31
                // (shift count >= u32 width); `(2u << lane) - 1u` produces the
                // same inclusive-prefix mask without UB. The idiom only works
                // because `lane` is in [0, 31] - assert that explicitly.
                static_assert(WARP_SIZE == 32,
                              "(2u << lane) - 1u inclusive-prefix idiom assumes WARP_SIZE == 32");
                uint32_t my_prefix = fresh_mask & ((2u << lane) - 1u);
                int src_lane = (my_prefix != 0) ? lastBitSet(my_prefix) : 0;
                int32_t shuffled_off = __shfl_sync(FULL_WARP_MASK, my_match_offset, src_lane);
                if (my_use_recent && my_prefix != 0) {
                    my_match_offset = shuffled_off;
                }

                uint32_t my_total = my_lit_len + my_match_len;
                uint32_t my_dst_local, total_dst;
                warpScanU32(my_total, my_dst_local, total_dst);
                uint32_t my_lit_local, total_lit;
                warpScanU32(my_lit_len, my_lit_local, total_lit);

                // ── Flat batched literal copy (v4 #1, 2026-06-10) ──
                // The old per-token loop ran warpLiteralCopy once per
                // token: a typical token carries ~5-8 literal bytes, so
                // each copy iteration used ~20% of the warp plus ~10
                // instructions of shuffle/sync overhead, 64 __syncwarp()s
                // per batch. Instead: stage the two prefix sums to shared
                // memory and copy the batch's WHOLE concatenated literal
                // run in one warp-wide pass — every lane owns flat
                // literal byte i, finds its owning token k (largest k
                // with lit_prefix[k] <= i, 5-step binary search over the
                // staged prefixes), and writes dst at the token's output
                // slot. Source is contiguous (lit[lit_pos + i]); dst is
                // piecewise-contiguous. nvCOMP's LZ4 decode ships this
                // exact design (docs/nvcomp_lz4_architecture.md §8).
                //
                // Ordering safety: every byte a match copy reads sits
                // strictly BELOW its own write position (off16 offsets
                // are negative; the overlap path self-feeds in order),
                // and the batch's literal slots are disjoint from all
                // match write ranges — so writing ALL the batch's
                // literals first, then running the matches in token
                // order, produces byte-identical output to the strict
                // interleaved order. One __syncwarp() fences the flat
                // pass before the first match read.
                s_lit_prefix[threadIdx.y][lane] = my_lit_local;
                // dst byte for flat lit i owned by token k:
                //   dst_pos + dst_local[k] + (i - lit_prefix[k])
                // = dst_pos + (dst_local[k] - lit_prefix[k]) + i
                s_dst_adj[threadIdx.y][lane] = my_dst_local - my_lit_local;
                __syncwarp();

                for (uint32_t i = lane; i < total_lit; i += WARP_SIZE) {
                    uint32_t k = 0;
                    #pragma unroll
                    for (uint32_t step = 16; step >= 1; step >>= 1) {
                        uint32_t cand = k + step;
                        if (cand < batch_size && s_lit_prefix[threadIdx.y][cand] <= i)
                            k = cand;
                    }
                    dst[dst_pos + s_dst_adj[threadIdx.y][k] + i] = lit[lit_pos + i];
                }
                __syncwarp();

                // ── Match copies, token order, lit-free ──
                // Ballot the lanes that actually carry a match and walk
                // only those (lit-only tokens cost zero iterations now).
                uint32_t match_mask = __ballot_sync(FULL_WARP_MASK, my_match_len > 0);
                while (match_mask != 0) {
                    uint32_t k = (uint32_t)(__ffs(match_mask) - 1);
                    match_mask &= match_mask - 1;
                    uint32_t k_lit_len   = __shfl_sync(FULL_WARP_MASK, my_lit_len, k);
                    uint32_t k_match_len = __shfl_sync(FULL_WARP_MASK, my_match_len, k);
                    int32_t  k_match_off = __shfl_sync(FULL_WARP_MASK, my_match_offset, k);
                    uint32_t k_dst_local = __shfl_sync(FULL_WARP_MASK, my_dst_local, k);

                    uint32_t copy_dst = dst_pos + k_dst_local + k_lit_len;
                    uint32_t match_src = (uint32_t)((int32_t)copy_dst + k_match_off);
                    warpMatchCopy(dst, copy_dst, match_src, k_match_len, -k_match_off, lane);
                    __syncwarp();
                }

                cmd_pos   += batch_size;
                off16_pos += total_off16_used;
                dst_pos   += total_dst;
                lit_pos   += total_lit;

                if (fresh_mask != 0) {
                    int last_fresh = lastBitSet(fresh_mask);
                    recent_offset = __shfl_sync(FULL_WARP_MASK, my_match_offset, last_fresh);
                }
                continue;
            }
        }

        uint32_t lit_len = 0, match_len = 0;
        int32_t match_offset = recent_offset;
        uint32_t use_recent = 0;

        // ── Token parse (lane 0 only) ──
        if (lane == 0) {
            uint32_t token = cmd[cmd_pos];
            if (token >= TOKEN_SHORT_MIN) {
                lit_len = token & TOKEN_LIT_MASK;
                match_len = (token >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK;
                use_recent = (token >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK;
                if (!use_recent && off16_pos < off16_count) {
                    uint16_t v;
                    if constexpr (OFF16_SPLIT) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        v = readU16LE(off16_raw + off16_pos * OFF16_ENTRY_BYTES);
                    }
                    match_offset = -(int32_t)v;
                    off16_pos++;
                }
            } else if (token == TOKEN_LONG_LITERAL) {
                use_recent = 1;
                lit_len = readLength(length_stream, length_offset, length_remaining)
                        + LONG_LITERAL_BASE;
            } else if (token == TOKEN_LONG_NEAR) {
                match_len = readLength(length_stream, length_offset, length_remaining)
                          + LONG_NEAR_BASE;
                if (off16_pos < off16_count) {
                    uint16_t v;
                    if constexpr (OFF16_SPLIT) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        v = readU16LE(off16_raw + off16_pos * OFF16_ENTRY_BYTES);
                    }
                    match_offset = -(int32_t)v; off16_pos++;
                }
            }
        }
        cmd_pos++;

        // ── Broadcast parsed values from lane 0 to all lanes ──
        lit_len      = __shfl_sync(FULL_WARP_MASK, lit_len, 0);
        match_len    = __shfl_sync(FULL_WARP_MASK, match_len, 0);
        match_offset = __shfl_sync(FULL_WARP_MASK, match_offset, 0);
        use_recent   = __shfl_sync(FULL_WARP_MASK, use_recent, 0);
        off16_pos    = __shfl_sync(FULL_WARP_MASK, off16_pos, 0);

        // ── Warp-cooperative literal copy ──
        if (lit_len > 0) {
            warpLiteralCopy(dst, dst_pos, lit, lit_pos, lit_len, lane);
            __syncwarp();
        }
        dst_pos += lit_len;
        lit_pos += lit_len;

        // ── Warp-cooperative match copy ──
        if (match_len > 0) {
            uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
            warpMatchCopy(dst, dst_pos, match_src, match_len, -match_offset, lane);
            __syncwarp();
        }
        dst_pos += match_len;

        if (!use_recent) recent_offset = match_offset;
        recent_offset = __shfl_sync(FULL_WARP_MASK, recent_offset, 0);
        length_offset = __shfl_sync(FULL_WARP_MASK, length_offset, 0);
    }

    // ── Trailing literals (bytes after the last token) ──
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    for (uint32_t i = lane; i < trailing; i += WARP_SIZE)
        dst[dst_pos + i] = lit[lit_pos + i];

}
