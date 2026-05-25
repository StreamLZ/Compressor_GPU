// ── StreamLZ LZ decode hot loops ───────────────────────────────
// The two warp-cooperative LZ decoders. Lane 0 of the warp parses
// tokens serially; all 32 lanes participate in literal and match byte
// copies. Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"

// Hillis-Steele warp exclusive scan, returns (lane's exclusive prefix, total).
__device__ __forceinline__ void warpScanU32(uint32_t v, uint32_t& exclusive, uint32_t& total) {
    uint32_t inclusive = v;
    const int lane = threadIdx.x & LANE_MASK;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t n = __shfl_up_sync(FULL_WARP_MASK, inclusive, d);
        if (lane >= d) inclusive += n;
    }
    exclusive = inclusive - v;
    total = __shfl_sync(FULL_WARP_MASK, inclusive, 31);
}

// ── Raw-mode sub-chunk decoder ─────────────────────────────────
// Streamlined single-LZ-block decoder: no off32, no delta literals, no
// block split. Selected for any sub-chunk with off32_count == 0.
// Register-optimized: uses uint32 offsets instead of 64-bit pointers,
// removes redundant bounds checks and shuffles.
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

            if (any_long == 0) {
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
                uint32_t my_prefix = fresh_mask & ((1u << (lane + 1)) - 1u);
                int src_lane = (my_prefix != 0) ? (31 - __clz(my_prefix)) : 0;
                int32_t shuffled_off = __shfl_sync(FULL_WARP_MASK, my_match_offset, src_lane);
                if (my_use_recent && my_prefix != 0) {
                    my_match_offset = shuffled_off;
                }

                uint32_t my_total = my_lit_len + my_match_len;
                uint32_t my_dst_local, total_dst;
                warpScanU32(my_total, my_dst_local, total_dst);
                uint32_t my_lit_local, total_lit;
                warpScanU32(my_lit_len, my_lit_local, total_lit);

                #pragma unroll 1
                for (uint32_t k = 0; k < batch_size; k++) {
                    uint32_t k_lit_len   = __shfl_sync(FULL_WARP_MASK, my_lit_len, k);
                    uint32_t k_match_len = __shfl_sync(FULL_WARP_MASK, my_match_len, k);
                    int32_t  k_match_off = __shfl_sync(FULL_WARP_MASK, my_match_offset, k);
                    uint32_t k_dst_local = __shfl_sync(FULL_WARP_MASK, my_dst_local, k);
                    uint32_t k_lit_local = __shfl_sync(FULL_WARP_MASK, my_lit_local, k);

                    uint32_t this_dst_pos = dst_pos + k_dst_local;
                    uint32_t this_lit_pos = lit_pos + k_lit_local;

                    if (k_lit_len > 0) {
                        for (uint32_t i = lane; i < k_lit_len; i += WARP_SIZE)
                            dst[this_dst_pos + i] = lit[this_lit_pos + i];
                        __syncwarp();
                    }
                    uint32_t copy_dst = this_dst_pos + k_lit_len;

                    if (k_match_len > 0) {
                        uint32_t match_src = (uint32_t)((int32_t)copy_dst + k_match_off);
                        int32_t match_dist = -k_match_off;
                        if (match_dist >= (int32_t)k_match_len && k_match_len > MIN_PARALLEL_MATCH_LEN - 1) {
                            for (uint32_t i = lane; i < k_match_len; i += WARP_SIZE)
                                dst[copy_dst + i] = dst[match_src + i];
                        } else {
                            if (lane == 0)
                                for (uint32_t i = 0; i < k_match_len; i++)
                                    dst[copy_dst + i] = dst[match_src + i];
                        }
                        __syncwarp();
                    }
                }

                cmd_pos   += batch_size;
                off16_pos += total_off16_used;
                dst_pos   += total_dst;
                lit_pos   += total_lit;

                if (fresh_mask != 0) {
                    int last_fresh = 31 - __clz(fresh_mask);
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
                    if (OFF16_SPLIT) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        memcpy(&v, off16_raw + off16_pos * OFF16_ENTRY_BYTES, OFF16_ENTRY_BYTES);
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
                    if (OFF16_SPLIT) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        memcpy(&v, off16_raw + off16_pos * OFF16_ENTRY_BYTES, OFF16_ENTRY_BYTES);
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
            for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
                dst[dst_pos + i] = lit[lit_pos + i];
            __syncwarp();
        }
        dst_pos += lit_len;
        lit_pos += lit_len;

        // ── Warp-cooperative match copy ──
        if (match_len > 0) {
            uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
            int32_t match_dist = -match_offset;
            // Parallel copy is only safe for non-overlapping matches:
            // an overlapping match (match_dist < match_len) would have a
            // lane read a dst byte another lane has not written yet.
            if (match_dist >= (int32_t)match_len && match_len > MIN_PARALLEL_MATCH_LEN - 1) {
                for (uint32_t i = lane; i < match_len; i += WARP_SIZE)
                    dst[dst_pos + i] = dst[match_src + i];
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < match_len; i++)
                        dst[dst_pos + i] = dst[match_src + i];
            }
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

// ── General sub-chunk decoder ──────────────────────────────────
// Full-featured path: handles off32, delta literals, and multi-block
// sub-chunks (up to MAX_BLOCKS_PER_SUBCHUNK 64KB LZ blocks each).
// mode bits = (sub-chunk-header >> 19) & 0xF: only mode == 0 (delta
// literals) vs mode != 0 (raw literals) is distinguished here.
//
// Lane contract: warp-cooperative; all token fields and the running
// cursors are broadcast from lane 0 after each token.
__device__ void decodeSubChunkGeneral(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off16_hi, const uint8_t* __restrict__ off16_lo,
    uint32_t off16_split,
    const uint8_t* __restrict__ off32_raw1, uint32_t off32_count1,
    const uint8_t* __restrict__ off32_raw2, uint32_t off32_count2,
    const uint8_t* __restrict__ length_stream, uint32_t length_remaining,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t block2_cmd_offset,
    uint32_t dst_offset,
    uint32_t mode
) {
    const int lane = threadIdx.x & LANE_MASK;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0, off32_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = INITIAL_RECENT_OFFSET;
    uint32_t dst_end_abs = dst_offset + dst_size;
    uint32_t length_offset = 0;
    const uint8_t* off32_block = off32_raw1;
    uint32_t off32_block_count = off32_count1;
    uint32_t block_dst_start = dst_offset;

    uint32_t block_cmd_end = (block2_cmd_offset > 0 && block2_cmd_offset < cmd_size)
        ? block2_cmd_offset : cmd_size;

    // Prefetch first token
    uint32_t prefetched_token = 0;
    if (lane == 0 && cmd_pos < block_cmd_end) prefetched_token = cmd[cmd_pos];

    for (int block_iter = 0; block_iter < MAX_BLOCKS_PER_SUBCHUNK; block_iter++) {
        while (cmd_pos < block_cmd_end) {
            uint32_t token = 0, lit_len = 0, match_len = 0;
            int32_t match_offset = recent_offset;
            uint32_t use_recent = 0, token_type = 0;

            // ── Token parse (lane 0 only) ──
            if (lane == 0) {
                token = prefetched_token;
                cmd_pos++;
                if (cmd_pos < block_cmd_end) prefetched_token = cmd[cmd_pos];
                if (token >= TOKEN_SHORT_MIN) {
                    token_type = 0;
                    lit_len = token & TOKEN_LIT_MASK;
                    match_len = (token >> TOKEN_MATCH_SHIFT) & TOKEN_MATCH_MASK;
                    use_recent = (token >> TOKEN_USE_RECENT_SHIFT) & TOKEN_USE_RECENT_MASK;
                    if (!use_recent && off16_pos < off16_count) {
                        uint16_t v;
                        if (off16_split) {
                            v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                        } else {
                            v = readU16LE(off16_raw + off16_pos * OFF16_ENTRY_BYTES);
                        }
                        match_offset = -(int32_t)v;
                        off16_pos++;
                    }
                } else if (token == TOKEN_LONG_LITERAL) {
                    token_type = 1;
                    lit_len = readLength(length_stream, length_offset, length_remaining)
                            + LONG_LITERAL_BASE;
                } else if (token == TOKEN_LONG_NEAR) {
                    token_type = 2;
                    match_len = readLength(length_stream, length_offset, length_remaining)
                              + LONG_NEAR_BASE;
                    if (off16_pos < off16_count) {
                        uint16_t v;
                        if (off16_split) {
                            v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                        } else {
                            v = readU16LE(off16_raw + off16_pos * OFF16_ENTRY_BYTES);
                        }
                        match_offset = -(int32_t)v; off16_pos++;
                    }
                    use_recent = 0;
                } else if (token == TOKEN_LONG_FAR) {
                    token_type = 4;
                    match_len = readLength(length_stream, length_offset, length_remaining)
                              + LONG_FAR_BASE;
                    if (off32_pos < off32_block_count) {
                        // off32 entries are 3 bytes each (encoder writes byte triples).
                        const uint8_t* p = off32_block + off32_pos * OFF32_ENTRY_BYTES;
                        uint32_t v = readLE24(p);
                        match_offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                        off32_pos++;
                    }
                    use_recent = 0;
                } else {
                    token_type = 3;
                    match_len = token + SHORT_FAR_BASE;
                    if (off32_pos < off32_block_count) {
                        // off32 entries are 3 bytes each (encoder writes byte triples).
                        const uint8_t* p = off32_block + off32_pos * OFF32_ENTRY_BYTES;
                        uint32_t v = readLE24(p);
                        match_offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                        off32_pos++;
                    }
                    use_recent = 0;
                }
            }

            // ── Broadcast parsed values from lane 0 to all lanes ──
            token_type   = __shfl_sync(FULL_WARP_MASK, token_type, 0);
            lit_len      = __shfl_sync(FULL_WARP_MASK, lit_len, 0);
            match_len    = __shfl_sync(FULL_WARP_MASK, match_len, 0);
            match_offset = __shfl_sync(FULL_WARP_MASK, match_offset, 0);
            use_recent   = __shfl_sync(FULL_WARP_MASK, use_recent, 0);
            cmd_pos      = __shfl_sync(FULL_WARP_MASK, cmd_pos, 0);
            lit_pos      = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
            off16_pos    = __shfl_sync(FULL_WARP_MASK, off16_pos, 0);
            off32_pos    = __shfl_sync(FULL_WARP_MASK, off32_pos, 0);

            // ── Warp-cooperative literal copy ──
            if (lit_len > 0) {
                if (mode == 0) {
                    for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
                        if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                            uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                            dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
                        }
                } else {
                    for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
                        if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                            dst[dst_pos + i] = lit[lit_pos + i];
                }
                __syncwarp();
                dst_pos += lit_len;
                lit_pos += lit_len;
            }

            // ── Warp-cooperative match copy ──
            if (match_len > 0) {
                uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
                int32_t match_dist = -match_offset;
                // Parallel copy is only safe for non-overlapping matches:
                // an overlapping match (match_dist < match_len) would have
                // a lane read a dst byte another lane has not written yet.
                if (match_dist >= (int32_t)match_len && match_len > MIN_PARALLEL_MATCH_LEN - 1) {
                    for (uint32_t i = lane; i < match_len; i += WARP_SIZE)
                        if (dst_pos + i < dst_end_abs)
                            dst[dst_pos + i] = dst[match_src + i];
                } else {
                    if (lane == 0)
                        for (uint32_t i = 0; i < match_len; i++)
                            if (dst_pos + i < dst_end_abs)
                                dst[dst_pos + i] = dst[match_src + i];
                }
                __syncwarp();
                dst_pos += match_len;
            }

            dst_pos   = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
            lit_pos   = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
            if (!use_recent && (token_type != 1)) recent_offset = match_offset;
            recent_offset = __shfl_sync(FULL_WARP_MASK, recent_offset, 0);
            length_offset = __shfl_sync(FULL_WARP_MASK, length_offset, 0);
        }

        // ── Per-block trailing literals (at 64KB boundary) ──
        __syncwarp();
        dst_pos = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
        lit_pos = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
        {
            uint32_t block_end = block_dst_start + LZ_BLOCK_SIZE;
            if (block_end > dst_end_abs) block_end = dst_end_abs;
            uint32_t block_trailing = (block_end > dst_pos) ? (block_end - dst_pos) : 0;
            if (mode == 0) {
                for (uint32_t i = lane; i < block_trailing; i += WARP_SIZE)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                        uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                        dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
                    }
            } else {
                for (uint32_t i = lane; i < block_trailing; i += WARP_SIZE)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                        dst[dst_pos + i] = lit[lit_pos + i];
            }
            __syncwarp();
            dst_pos += block_trailing;
            lit_pos += block_trailing;
        }

        // ── Advance to block 2 ──
        if (block_iter + 1 >= MAX_BLOCKS_PER_SUBCHUNK) break;
        block_cmd_end = cmd_size;
        off32_block = off32_raw2;
        off32_block_count = off32_count2;
        off32_pos = 0;
        block_dst_start = dst_pos;
        if (lane == 0 && cmd_pos < block_cmd_end) prefetched_token = cmd[cmd_pos];
    }

    // ── Final trailing literals ──
    __syncwarp();
    dst_pos = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
    lit_pos = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    if (mode == 0) {
        for (uint32_t i = lane; i < trailing; i += WARP_SIZE)
            if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
            }
    } else {
        for (uint32_t i = lane; i < trailing; i += WARP_SIZE)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}
