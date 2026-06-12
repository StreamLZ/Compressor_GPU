// ── StreamLZ LZ decode - general sub-chunk decoder ─────────────
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

// Per-call destination region for decodeSubChunkGeneral. The decoder
// only writes inside `dst[dst_offset .. dst_offset + dst_size)`; the
// overflow check at function entry derives `dst_end_abs` from these.
struct DecodeOutput {
    uint8_t* __restrict__ dst;
    uint32_t dst_size;
    uint32_t dst_offset;
};

// Token-type tag values assigned by the lane-0 parser and consumed
// after the broadcast. Only LONG_LITERAL is tested by name (it
// suppresses the recent_offset update - long-literal tokens don't
// carry a new offset, so recent_offset must be preserved). The others
// differ only in which offset stream they pull from (off16 vs off32);
// kept named for debuggability.
enum TokenType : uint32_t {
    TOKEN_TYPE_SHORT        = 0, // 1-byte token, inline off16 (or use_recent)
    TOKEN_TYPE_LONG_LITERAL = 1, // 1-byte token + extended literal length
    TOKEN_TYPE_LONG_NEAR    = 2, // 1-byte token + extended match length, off16
    TOKEN_TYPE_SHORT_FAR    = 3, // 1-byte token (match length inline), off32
    TOKEN_TYPE_LONG_FAR     = 4, // 1-byte token + extended match length, off32
};

// Delta-literal mode (sub-chunk mode == 0): each output byte is the
// sum of a literal byte and the byte at `dst[i + recent_offset]`.
// Used in the inner token loop AND the per-block / final trailing-
// literal flushes - same pattern at three sites. `__forceinline__` so
// nvcc inlines it identically to the open-coded version it replaces.
__device__ __forceinline__ void deltaLiteralCopyBounded(
    uint8_t* __restrict__ dst, uint32_t dst_pos,
    const uint8_t* __restrict__ lit, uint32_t lit_pos,
    uint32_t count, uint32_t dst_end_abs, uint32_t lit_size,
    int32_t recent_offset, int lane
) {
    for (uint32_t i = lane; i < count; i += WARP_SIZE)
        if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
            uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
            dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
        }
}

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
// match_len count). A prior cleanup removed them on that argument and
// L3 enwik8 kernel time regressed ~130 µs (+2%) measurably. The PTX
// REG count stays at 40 either way, so the cost is scheduling /
// memory-ordering, not register pressure. Keep the shfls.
template <bool OFF16_SPLIT, bool HAS_DICT = false>
__device__ void decodeSubChunkGeneral(
    const ParsedStreams& ps,
    const DecodeOutput& out,
    uint32_t mode,
    // v4 #16 (HAS_DICT only): preset dictionary; match sources below
    // `out.dst_offset` (the sub-chunk's window base) read the dict tail.
    const uint8_t* __restrict__ dict = nullptr, uint32_t dict_len = 0
) {
    // Hoist field reads into locals so the hot loop addresses registers
    // rather than restating `ps.x` everywhere. nvcc would do this anyway
    // but the explicit form preserves the codegen shape from before the
    // struct-arg refactor.
    const uint8_t* __restrict__ cmd = ps.cmd_ptr;
    const uint32_t cmd_size = ps.cmd_size;
    const uint8_t* __restrict__ lit = ps.lit_ptr;
    const uint32_t lit_size = ps.lit_size;
    // off16_raw is unused when OFF16_SPLIT=true; off16_hi/lo are unused
    // when OFF16_SPLIT=false. [[maybe_unused]] silences nvcc's #177-D
    // "declared but never referenced" warning in each specialization -
    // the warning is in fact evidence the if-constexpr DCE is working.
    [[maybe_unused]] const uint8_t* __restrict__ off16_raw = ps.off16_raw;
    const uint32_t off16_count = ps.off16_count;
    [[maybe_unused]] const uint8_t* __restrict__ off16_hi = ps.off16_hi;
    [[maybe_unused]] const uint8_t* __restrict__ off16_lo = ps.off16_lo;
    const uint8_t* __restrict__ off32_raw1 = ps.off32_raw1;
    const uint32_t off32_count1 = ps.off32_count1;
    const uint8_t* __restrict__ off32_raw2 = ps.off32_raw2;
    const uint32_t off32_count2 = ps.off32_count2;
    const uint8_t* __restrict__ length_stream = ps.len_stream;
    const uint32_t length_remaining = ps.len_avail;
    const uint32_t initial_copy = ps.initial_copy;
    const uint32_t block2_cmd_offset = ps.cmd_stream2_offset;
    uint8_t* __restrict__ dst = out.dst;
    const uint32_t dst_size = out.dst_size;
    const uint32_t dst_offset = out.dst_offset;

    const int lane = threadIdx.x & LANE_MASK;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0, off32_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = INITIAL_RECENT_OFFSET;
    // dst_end_abs is the upper bound on every `dst[i]` store below. If a
    // corrupt descriptor presents `dst_offset + dst_size` that overflows
    // uint32, the sum would wrap small and let writes scribble far past
    // the legal output region. One-time clamp at function entry - costs
    // ~2 instructions per sub-chunk decode, not in any hot loop.
    uint32_t dst_end_abs = (dst_size > 0xFFFFFFFFu - dst_offset)
        ? 0xFFFFFFFFu
        : (dst_offset + dst_size);
    uint32_t length_offset = 0;
    const uint8_t* off32_block = off32_raw1;
    uint32_t off32_block_count = off32_count1;
    uint32_t block_dst_start = dst_offset;

    uint32_t block_cmd_end = (block2_cmd_offset > 0 && block2_cmd_offset < cmd_size)
        ? block2_cmd_offset : cmd_size;

    // Prefetch first token. Hoisted outside the for-loop (both
    // block_iter iterations need the carried value across the boundary;
    // the advance-to-block-2 site at the bottom re-prefetches lane 0).
    uint32_t prefetched_token = 0;
    if (lane == 0 && cmd_pos < block_cmd_end) prefetched_token = cmd[cmd_pos];

    for (int block_iter = 0; block_iter < MAX_BLOCKS_PER_SUBCHUNK; block_iter++) {
        while (cmd_pos < block_cmd_end) {
            uint32_t token = 0, lit_len = 0, match_len = 0;
            int32_t match_offset = recent_offset;
            uint32_t use_recent = 0, token_type = TOKEN_TYPE_SHORT;

            // ── Token parse (lane 0 only) ──
            if (lane == 0) {
                token = prefetched_token;
                cmd_pos++;
                if (cmd_pos < block_cmd_end) prefetched_token = cmd[cmd_pos];
                if (token >= TOKEN_SHORT_MIN) {
                    token_type = TOKEN_TYPE_SHORT;
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
                    token_type = TOKEN_TYPE_LONG_LITERAL;
                    lit_len = readLength(length_stream, length_offset, length_remaining)
                            + LONG_LITERAL_BASE;
                } else if (token == TOKEN_LONG_NEAR) {
                    token_type = TOKEN_TYPE_LONG_NEAR;
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
                    use_recent = 0;
                } else if (token == TOKEN_LONG_FAR) {
                    token_type = TOKEN_TYPE_LONG_FAR;
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
                    token_type = TOKEN_TYPE_SHORT_FAR;
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
            // PERF: lit_pos looks redundant (lane-coherent - every lane
            // does lit_pos += lit_len below). It isn't. Removing this
            // + the dst_pos/lit_pos shfls in the 3 sites below costs
            // ~130 µs (+2%) on L3 enwik8 `-db -t 1 -r 30`. PTX REG=40
            // either way; nvcc uses them as reorder barriers.
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
                    deltaLiteralCopyBounded(dst, dst_pos, lit, lit_pos,
                                            lit_len, dst_end_abs, lit_size,
                                            recent_offset, lane);
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
                warpMatchCopyBoundedD<HAS_DICT>(dst, dst_pos, match_src, match_len,
                                                match_dist, dst_end_abs, lane,
                                                dst_offset, dict, dict_len);
                __syncwarp();
                dst_pos += match_len;
            }

            // PERF: dst_pos/lit_pos look redundant (lane-coherent after
            // the cooperative copies). They aren't - see top-of-loop note.
            dst_pos   = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
            lit_pos   = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
            if (!use_recent && (token_type != TOKEN_TYPE_LONG_LITERAL)) recent_offset = match_offset;
            recent_offset = __shfl_sync(FULL_WARP_MASK, recent_offset, 0);
            length_offset = __shfl_sync(FULL_WARP_MASK, length_offset, 0);
        }

        // ── Per-block trailing literals (at 64KB boundary) ──
        __syncwarp();
        // PERF: see top-of-function "five __shfl_sync broadcasts" note.
        dst_pos = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
        lit_pos = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
        {
            uint32_t block_end = block_dst_start + LZ_BLOCK_SIZE;
            if (block_end > dst_end_abs) block_end = dst_end_abs;
            uint32_t block_trailing = (block_end > dst_pos) ? (block_end - dst_pos) : 0;
            if (mode == 0) {
                deltaLiteralCopyBounded(dst, dst_pos, lit, lit_pos,
                                        block_trailing, dst_end_abs, lit_size,
                                        recent_offset, lane);
            } else {
                warpLiteralCopyBounded(dst, dst_pos, lit, lit_pos,
                                       block_trailing, dst_end_abs, lit_size, lane);
            }
            __syncwarp();
            dst_pos += block_trailing;
            lit_pos += block_trailing;
        }

        // ── Advance to block 2 ──
        // On the last iter (block_iter == MAX-1) these writes are dead -
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
    // PERF: see top-of-function "five __shfl_sync broadcasts" note.
    dst_pos = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);
    lit_pos = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    if (mode == 0) {
        deltaLiteralCopyBounded(dst, dst_pos, lit, lit_pos,
                                trailing, dst_end_abs, lit_size,
                                recent_offset, lane);
    } else {
        // The final trailing literal copy lacks the lit_pos bound check
        // (the trailing count itself derives from lit_size), so the
        // bounded literal helper would add a guard the original did not.
        for (uint32_t i = lane; i < trailing; i += WARP_SIZE)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}
