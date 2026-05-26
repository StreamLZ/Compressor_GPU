// ── StreamLZ LZ decode — general sub-chunk decoder ─────────────
// One of two warp-cooperative LZ decoders (the other lives in
// lz_decode_raw.cuh). Full-featured path: handles off32, delta
// literals, and multi-block sub-chunks (up to MAX_BLOCKS_PER_SUBCHUNK
// 64KB LZ blocks each). mode bits = (sub-chunk-header >> 19) & 0xF:
// only mode == 0 (delta literals) vs mode != 0 (raw literals) is
// distinguished here.
//
// Lane contract: warp-cooperative; all token fields and the running
// cursors are broadcast from lane 0 after each token.
//
// Included into the single lz_kernel.cu translation unit. Depends on
// lz_decode_core.cuh for warpScanU32 + warpLiteralCopy + warpMatchCopy.
#pragma once

#include "lz_decode_core.cuh"

// Intentionally NOT __noinline__: an experiment in this session
// confirmed the attribute is a net negative here. With it,
// slzLzDecodeKernel STACK grew 192 -> 208 and slzLzDecodeRawKernel
// STACK grew 72 -> 80, while REG stayed at 40 in both. The raw
// decoder's sibling banner explains why __noinline__ helps THERE (the
// register-pressure consideration is real for that function); the
// general decoder is large enough that nvcc already places it
// out-of-line implicitly, and forcing the attribute only adds spill
// slots.
//
// The five __shfl_sync broadcasts of dst_pos / lit_pos in the token
// loop and per-block / final trailing-literal sections look formally
// redundant (every lane updates both by the broadcast lit_len /
// match_len count). The K1 cleanup removed them on that argument and
// L3 enwik8 kernel time regressed ~130 µs (+2%) measurably. The PTX
// REG count stays at 40 either way, so the cost is scheduling /
// memory-ordering, not register pressure. Keep the shfls.
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
            // The full broadcast set (including lit_pos / dst_pos shfls
            // below) is required for perf even though some values are
            // already coherent across lanes — see the C9 revert note in
            // the file docstring.
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
                    // Delta-literal mode: dst[i] = lit[i] + dst[i + recent_offset].
                    // Not extractable into a generic helper because of the
                    // additive recent-offset back-reference.
                    for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
                        if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                            uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                            dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
                        }
                } else {
                    warpLiteralCopyBounded(dst, dst_pos, lit, lit_pos,
                                           lit_len, dst_end_abs, lit_size, lane);
                }
                __syncwarp();
                dst_pos += lit_len;
                lit_pos += lit_len;
            }

            // ── Warp-cooperative match copy ──
            if (match_len > 0) {
                uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
                int32_t match_dist = -match_offset;
                warpMatchCopyBounded(dst, dst_pos, match_src, match_len,
                                     match_dist, dst_end_abs, lane);
                __syncwarp();
                dst_pos += match_len;
            }

            // dst_pos / lit_pos are formally coherent here (every lane
            // added the broadcast lit_len / match_len above), but the
            // shfls below are still required for measured perf — see C9
            // revert note in the file docstring.
            dst_pos   = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
            lit_pos   = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
            if (!use_recent && (token_type != 1)) recent_offset = match_offset;
            recent_offset = __shfl_sync(FULL_WARP_MASK, recent_offset, 0);
            length_offset = __shfl_sync(FULL_WARP_MASK, length_offset, 0);
        }

        // ── Per-block trailing literals (at 64KB boundary) ──
        __syncwarp();
        // Lane-broadcast dst_pos / lit_pos required for perf — formally
        // coherent but the shfls act as nvcc reorder barriers. Do NOT
        // remove without re-running `-db -t 1 -r 30` on L3 (this was the
        // exact regression in K1's C9; see file docstring).
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
                warpLiteralCopyBounded(dst, dst_pos, lit, lit_pos,
                                       block_trailing, dst_end_abs, lit_size, lane);
            }
            __syncwarp();
            dst_pos += block_trailing;
            lit_pos += block_trailing;
        }

        // ── Advance to block 2 ──
        // On the last iter (block_iter == MAX-1) these writes are dead —
        // the for-loop bound will exit before any are read. Letting the
        // setup run on the dead iteration trades a never-taken `break`
        // for a few unconditional writes that nvcc dead-code-eliminates.
        block_cmd_end = cmd_size;
        off32_block = off32_raw2;
        off32_block_count = off32_count2;
        off32_pos = 0;
        block_dst_start = dst_pos;
        if (lane == 0 && cmd_pos < block_cmd_end) prefetched_token = cmd[cmd_pos];
    }

    // ── Final trailing literals ──
    __syncwarp();
    // Same perf-required shfls as the per-block trailing block above —
    // see file docstring (C9 revert) before considering removal.
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
        // The final trailing literal copy lacks the lit_pos bound check
        // (the trailing count itself derives from lit_size), so the
        // bounded literal helper would add a guard the original did not.
        for (uint32_t i = lane; i < trailing; i += WARP_SIZE)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}
