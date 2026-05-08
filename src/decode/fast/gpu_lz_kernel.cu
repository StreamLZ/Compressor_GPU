// StreamLZ L1 GPU Decompressor Kernel
// Phase 1: batch parallel — 1 warp per SC group, all groups simultaneously.
// CPU does entropy decode (readLzTable), GPU does token decode (processLzRuns).

#include <cstdint>
#include <cstdio>

// Batch descriptor: one per sub-chunk, indexes into packed stream buffers.
struct SlzSubChunkDesc {
    uint32_t cmd_offset;      uint32_t cmd_size;
    uint32_t lit_offset;      uint32_t lit_size;
    uint32_t off16_offset;    uint32_t off16_count;
    uint32_t off32_1_offset;  uint32_t off32_count1;
    uint32_t off32_2_offset;  uint32_t off32_count2;
    uint32_t len_offset;      uint32_t len_avail;
    uint32_t dst_offset;
    uint32_t dst_size;
    uint32_t initial_copy;
    uint32_t cmd_stream2_offset;
};

// Extended length decode
__device__ uint32_t readLength(const uint8_t* &len_stream, const uint8_t* len_end) {
    if (len_stream >= len_end) return 0;
    uint32_t v = *len_stream;
    if (v > 251) {
        if (len_stream + 2 >= len_end) { len_stream++; return v; }
        uint16_t extra;
        memcpy(&extra, len_stream + 1, 2);
        v += (uint32_t)extra * 4;
        len_stream += 2;
    }
    len_stream++;
    return v;
}

// Core sub-chunk decode: used by both serial and batch kernels.
// mode: 0 = delta literals, 1 = raw literals
__device__ void decodeSubChunk(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off32_raw1, uint32_t off32_count1,
    const uint8_t* __restrict__ off32_raw2, uint32_t off32_count2,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t cmd_stream2_offset,
    uint32_t dst_offset,
    uint32_t mode
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0, off32_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = -8;
    uint32_t dst_end_abs = dst_offset + dst_size;
    const uint8_t* len_stream = len_data;
    const uint8_t* len_end = len_data + len_avail;
    const uint8_t* off32_cur = off32_raw1;
    uint32_t off32_count_cur = off32_count1;
    uint32_t block_dst_start = dst_offset;

    uint32_t block_cmd_end = (cmd_stream2_offset > 0 && cmd_stream2_offset < cmd_size)
        ? cmd_stream2_offset : cmd_size;
    int block_iter = 0;

    for (;;) {
    while (cmd_pos < block_cmd_end) {
        uint32_t token = 0, local_lit = 0, local_match = 0;
        int32_t offset = recent_offset;
        uint32_t use_recent = 0, token_type = 0;

        if (lane == 0) {
            token = cmd[cmd_pos++];
            if (token >= 24) {
                token_type = 0;
                local_lit = token & 7;
                local_match = (token >> 3) & 0xF;
                use_recent = (token >> 7) & 1;
                if (!use_recent && off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    offset = -(int32_t)v;
                    off16_pos++;
                }
            } else if (token == 0) {
                token_type = 1;
                local_lit = readLength(len_stream, len_end) + 64;
            } else if (token == 1) {
                token_type = 2;
                local_match = readLength(len_stream, len_end) + 91;
                if (off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    offset = -(int32_t)v; off16_pos++;
                }
                use_recent = 0;
            } else if (token == 2) {
                token_type = 4;
                local_match = readLength(len_stream, len_end) + 29;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            } else {
                token_type = 3;
                local_match = token + 5;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            }
        }

        token_type  = __shfl_sync(0xFFFFFFFF, token_type, 0);
        local_lit   = __shfl_sync(0xFFFFFFFF, local_lit, 0);
        local_match = __shfl_sync(0xFFFFFFFF, local_match, 0);
        offset      = __shfl_sync(0xFFFFFFFF, offset, 0);
        use_recent  = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        cmd_pos     = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        lit_pos     = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        off16_pos   = __shfl_sync(0xFFFFFFFF, off16_pos, 0);
        off32_pos   = __shfl_sync(0xFFFFFFFF, off32_pos, 0);

        if (local_lit > 0) {
            if (mode == 0) {
                // Delta literals: dst[i] = lit[i] + dst[match_src + i]
                for (uint32_t i = lane; i < local_lit; i += 32)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                        uint32_t ms = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                        dst[dst_pos + i] = lit[lit_pos + i] + dst[ms];
                    }
            } else {
                for (uint32_t i = lane; i < local_lit; i += 32)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                        dst[dst_pos + i] = lit[lit_pos + i];
            }
            __syncwarp();
            dst_pos += local_lit;
            lit_pos += local_lit;
        }

        if (local_match > 0) {
            uint32_t ms = (uint32_t)((int32_t)dst_pos + offset);
            int32_t abs_off = -offset;
            if (abs_off >= (int32_t)local_match && local_match > 1) {
                for (uint32_t i = lane; i < local_match; i += 32)
                    if (dst_pos + i < dst_end_abs)
                        dst[dst_pos + i] = dst[ms + i];
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < local_match; i++)
                        if (dst_pos + i < dst_end_abs)
                            dst[dst_pos + i] = dst[ms + i];
            }
            __syncwarp();
            dst_pos += local_match;
        }

        dst_pos   = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
        lit_pos   = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        if (!use_recent && (token_type != 1)) recent_offset = offset;
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
        {
            uint32_t len_off = (uint32_t)((uintptr_t)len_stream - (uintptr_t)len_data);
            len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
            len_stream = len_data + len_off;
        }
    }

    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    {
        uint32_t block_end = block_dst_start + 0x10000;
        if (block_end > dst_end_abs) block_end = dst_end_abs;
        uint32_t block_trailing = (block_end > dst_pos) ? (block_end - dst_pos) : 0;
        if (mode == 0) {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                    uint32_t ms = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                    dst[dst_pos + i] = lit[lit_pos + i] + dst[ms];
                }
        } else {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                    dst[dst_pos + i] = lit[lit_pos + i];
        }
        __syncwarp();
        dst_pos += block_trailing;
        lit_pos += block_trailing;
    }

    block_iter++;
    if (block_iter >= 2) break;
    block_cmd_end = cmd_size;
    off32_cur = off32_raw2;
    off32_count_cur = off32_count2;
    off32_pos = 0;
    block_dst_start = dst_pos;
    }

    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    if (mode == 0) {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                uint32_t ms = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                dst[dst_pos + i] = lit[lit_pos + i] + dst[ms];
            }
    } else {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}

// ────────────────────────────────────────────────────────────────
//  Grouped parallel decode: parse all tokens, then each lane
//  processes TOKENS_PER_LANE tokens independently (lit + match).
// ────────────────────────────────────────────────────────────────

#define TOKENS_PER_LANE 16

struct TokenWork {
    uint32_t lit_len;
    uint32_t match_len;
    int32_t  offset;
    uint32_t lit_src;        // position in literal stream
    uint32_t dst_pos;        // output position for this token's literals
    int32_t  recent_offset;  // recent_offset at time of this token (for delta)
};

__device__ void decodeSubChunkBatched(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off32_raw1, uint32_t off32_count1,
    const uint8_t* __restrict__ off32_raw2, uint32_t off32_count2,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t cmd_stream2_offset,
    uint32_t dst_offset,
    uint32_t mode
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0, off32_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = -8;
    uint32_t dst_end_abs = dst_offset + dst_size;
    const uint8_t* len_stream = len_data;
    const uint8_t* len_end = len_data + len_avail;
    const uint8_t* off32_cur = off32_raw1;
    uint32_t off32_count_cur = off32_count1;
    uint32_t block_dst_start = dst_offset;

    uint32_t block_cmd_end = (cmd_stream2_offset > 0 && cmd_stream2_offset < cmd_size)
        ? cmd_stream2_offset : cmd_size;
    int block_iter = 0;

    const uint32_t BATCH_SZ = 32 * TOKENS_PER_LANE;
    __shared__ TokenWork batch[32 * TOKENS_PER_LANE];
    __shared__ uint32_t batch_n;
    __shared__ uint8_t match_safe[32 * TOKENS_PER_LANE];

    for (;;) {
    while (cmd_pos < block_cmd_end) {
        // ── Pass 1: Parse tokens, set absPos (lane 0, serial) ──
        if (lane == 0) {
            uint32_t n = 0;
            while (n < BATCH_SZ && cmd_pos < block_cmd_end) {
                uint32_t token = cmd[cmd_pos++];
                uint32_t local_lit = 0, local_match = 0;
                int32_t offset = recent_offset;
                uint32_t use_recent = 0;

                if (token >= 24) {
                    local_lit = token & 7;
                    local_match = (token >> 3) & 0xF;
                    use_recent = (token >> 7) & 1;
                    if (!use_recent && off16_pos < off16_count) {
                        uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                        offset = -(int32_t)v;
                        off16_pos++;
                    }
                } else if (token == 0) {
                    local_lit = readLength(len_stream, len_end) + 64;
                } else if (token == 1) {
                    local_match = readLength(len_stream, len_end) + 91;
                    if (off16_pos < off16_count) {
                        uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                        offset = -(int32_t)v; off16_pos++;
                    }
                    use_recent = 0;
                } else if (token == 2) {
                    local_match = readLength(len_stream, len_end) + 29;
                    if (off32_pos < off32_count_cur) {
                        uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                        offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                        off32_pos++;
                    }
                    use_recent = 0;
                } else {
                    local_match = token + 5;
                    if (off32_pos < off32_count_cur) {
                        uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                        offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                        off32_pos++;
                    }
                    use_recent = 0;
                }

                batch[n].lit_len = local_lit;
                batch[n].match_len = local_match;
                batch[n].offset = offset;
                batch[n].lit_src = lit_pos;
                batch[n].dst_pos = dst_pos;
                batch[n].recent_offset = recent_offset;

                if (!use_recent && token != 0)
                    recent_offset = offset;

                lit_pos += local_lit;
                dst_pos += local_lit + local_match;
                n++;
            }
            batch_n = n;
        }
        __syncwarp();
        uint32_t n_tokens = batch_n;
        if (n_tokens == 0) break;

        // ── Pass 2: Resolve all literals (massively parallel) ──
        // Each lane handles TOKENS_PER_LANE tokens. For raw mode: just copy.
        // For delta: lit[i] + dst[src+i] where src is (almost always) pre-existing output.
        for (uint32_t g = 0; g < TOKENS_PER_LANE; g++) {
            uint32_t idx = lane + g * 32;
            if (idx >= n_tokens) continue;
            uint32_t ll = batch[idx].lit_len;
            if (ll == 0) continue;
            uint32_t ls = batch[idx].lit_src;
            uint32_t dp = batch[idx].dst_pos;
            if (mode == 0) {
                int32_t ro = batch[idx].recent_offset;
                for (uint32_t i = 0; i < ll; i++)
                    if (dp + i < dst_end_abs && ls + i < lit_size) {
                        uint32_t ms = (uint32_t)((int32_t)(dp + i) + ro);
                        dst[dp + i] = lit[ls + i] + dst[ms];
                    }
            } else {
                for (uint32_t i = 0; i < ll; i++)
                    if (dp + i < dst_end_abs && ls + i < lit_size)
                        dst[dp + i] = lit[ls + i];
            }
        }
        __syncwarp();

        // ── Pass 3: Flag matches that only rely on literals (lane 0, serial) ──
        // A match is "lit-safe" if its entire source region was written by
        // literals (or is pre-existing output). Match source = [ms, ms+ml).
        // The batch's literal-written region is [batch_start, batch_end) with
        // gaps where matches go. Simple check: source entirely before the
        // batch's first MATCH destination, or before the batch start.
        if (lane == 0) {
            uint32_t batch_start_pos = batch[0].dst_pos;
            for (uint32_t t = 0; t < n_tokens; t++) {
                uint32_t ml = batch[t].match_len;
                if (ml == 0) { match_safe[t] = 1; continue; }
                int32_t off = batch[t].offset;
                uint32_t match_dst_pos = batch[t].dst_pos + batch[t].lit_len;
                uint32_t ms = (uint32_t)((int32_t)match_dst_pos + off);
                int32_t abs_off = -off;
                // Safe if: source entirely before batch output, OR
                // non-overlapping and source in literal-only region
                if (ms + ml <= batch_start_pos) {
                    match_safe[t] = 1;
                } else if (abs_off >= (int32_t)ml) {
                    // Non-overlapping. Check if source region only contains
                    // literals (every byte in [ms, ms+ml) was written by a
                    // literal, not a match). Approximate: source before this
                    // token's dst_pos means it could include prior match output.
                    // Conservative: safe only if source is pre-batch.
                    match_safe[t] = 0;
                } else {
                    match_safe[t] = 0;
                }
            }
        }
        __syncwarp();

        // ── Pass 4: Resolve lit-safe matches (massively parallel) ──
        for (uint32_t g = 0; g < TOKENS_PER_LANE; g++) {
            uint32_t idx = lane + g * 32;
            if (idx >= n_tokens) continue;
            if (!match_safe[idx] || batch[idx].match_len == 0) continue;
            uint32_t ml = batch[idx].match_len;
            int32_t off = batch[idx].offset;
            uint32_t match_dst_pos = batch[idx].dst_pos + batch[idx].lit_len;
            uint32_t ms = (uint32_t)((int32_t)match_dst_pos + off);
            for (uint32_t i = 0; i < ml; i++)
                if (match_dst_pos + i < dst_end_abs)
                    dst[match_dst_pos + i] = dst[ms + i];
        }
        __syncwarp();

        // ── Pass 5: Resolve remaining matches (serial, cooperative warp) ──
        for (uint32_t t = 0; t < n_tokens; t++) {
            if (match_safe[t] || batch[t].match_len == 0) continue;
            uint32_t ml = batch[t].match_len;
            int32_t off = batch[t].offset;
            uint32_t match_dst_pos = batch[t].dst_pos + batch[t].lit_len;
            uint32_t ms = (uint32_t)((int32_t)match_dst_pos + off);
            int32_t abs_off = -off;

            if (abs_off >= (int32_t)ml && ml > 1) {
                for (uint32_t i = lane; i < ml; i += 32)
                    if (match_dst_pos + i < dst_end_abs)
                        dst[match_dst_pos + i] = dst[ms + i];
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < ml; i++)
                        if (match_dst_pos + i < dst_end_abs)
                            dst[match_dst_pos + i] = dst[ms + i];
            }
            __syncwarp();
        }

        // Broadcast updated state from lane 0
        dst_pos   = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
        lit_pos   = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        cmd_pos   = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        off16_pos = __shfl_sync(0xFFFFFFFF, off16_pos, 0);
        off32_pos = __shfl_sync(0xFFFFFFFF, off32_pos, 0);
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
        {
            uint32_t len_off = (uint32_t)((uintptr_t)len_stream - (uintptr_t)len_data);
            len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
            len_stream = len_data + len_off;
        }
    }

    // Per-block trailing literals
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    {
        uint32_t block_end = block_dst_start + 0x10000;
        if (block_end > dst_end_abs) block_end = dst_end_abs;
        uint32_t block_trailing = (block_end > dst_pos) ? (block_end - dst_pos) : 0;
        if (mode == 0) {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                    uint32_t ms = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                    dst[dst_pos + i] = lit[lit_pos + i] + dst[ms];
                }
        } else {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                    dst[dst_pos + i] = lit[lit_pos + i];
        }
        __syncwarp();
        dst_pos += block_trailing;
        lit_pos += block_trailing;
    }

    block_iter++;
    if (block_iter >= 2) break;
    block_cmd_end = cmd_size;
    off32_cur = off32_raw2;
    off32_count_cur = off32_count2;
    off32_pos = 0;
    block_dst_start = dst_pos;
    }

    // Final trailing literals
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    if (mode == 0) {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                uint32_t ms = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                dst[dst_pos + i] = lit[lit_pos + i] + dst[ms];
            }
    } else {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}

// --- Serial kernel (legacy, wraps decodeSubChunk) ---

extern "C" __global__ void slzDecompressL1Kernel(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off32_raw1, uint32_t off32_count1,
    const uint8_t* __restrict__ off32_raw2, uint32_t off32_count2,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t cmd_stream2_offset,
    uint32_t base_offset,
    uint32_t dst_offset
) {
    decodeSubChunk(cmd, cmd_size, lit, lit_size,
        off16_raw, off16_count, off32_raw1, off32_count1,
        off32_raw2, off32_count2, len_data, len_avail,
        dst, dst_size, initial_copy, cmd_stream2_offset, dst_offset, 1);
}

// --- Batch kernel: 1 block per SC group, all groups simultaneously ---

extern "C" __global__ void slzBatchDecompressL1Kernel(
    const uint8_t* __restrict__ cmd_all,
    const uint8_t* __restrict__ lit_all,
    const uint8_t* __restrict__ off16_all,
    const uint8_t* __restrict__ off32_all,
    const uint8_t* __restrict__ len_all,
    uint8_t* __restrict__ dst,
    const SlzSubChunkDesc* __restrict__ descs,
    uint32_t sub_chunks_per_group,
    uint32_t total_sub_chunks
) {
    const uint32_t group_id = blockIdx.x;
    const uint32_t base_idx = group_id * sub_chunks_per_group;

    for (uint32_t sc = 0; sc < sub_chunks_per_group; sc++) {
        uint32_t idx = base_idx + sc;
        if (idx >= total_sub_chunks) return;

        const SlzSubChunkDesc& d = descs[idx];
        if (d.dst_size == 0) continue;

        decodeSubChunk(
            cmd_all + d.cmd_offset,    d.cmd_size,
            lit_all + d.lit_offset,    d.lit_size,
            off16_all + d.off16_offset, d.off16_count,
            off32_all + d.off32_1_offset, d.off32_count1,
            off32_all + d.off32_2_offset, d.off32_count2,
            len_all + d.len_offset,    d.len_avail,
            dst, d.dst_size, d.initial_copy, d.cmd_stream2_offset,
            d.dst_offset, 1
        );
        __syncwarp();
    }
}

// ────────────────────────────────────────────────────────────────
//  Phase 2: Full GPU decode — parse raw compressed chunks on GPU
//  CPU only uploads the raw compressed block + chunk descriptors.
// ────────────────────────────────────────────────────────────────

struct SlzChunkDesc {
    uint32_t src_offset;    // byte offset into compressed block
    uint32_t comp_size;     // compressed payload size
    uint32_t decomp_size;   // decompressed size (usually 256KB)
    uint32_t dst_offset;    // absolute output position
    uint32_t flags;         // bit 0: uncompressed, bit 1: memset
    uint8_t  memset_fill;   // fill byte for memset chunks
    uint8_t  _pad[3];
};

// Parse Type 0 entropy header: returns payload size and advances src.
__device__ uint32_t parseType0Header(const uint8_t* &src) {
    if (src[0] >= 0x80) {
        uint32_t sz = (((uint32_t)src[0] << 8) | src[1]) & 0xFFF;
        src += 2;
        return sz;
    } else {
        uint32_t sz = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
        src += 3;
        return sz;
    }
}

// GPU-side readLzTable + decodeSubChunk for one sub-chunk.
// Parses Type 0 entropy headers to locate streams, then runs token decode.
__device__ void parseAndDecodeSubChunk(
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint32_t mode
) {
    const int lane = threadIdx.x & 31;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    uint32_t initial_copy = 0;
    if (base_offset == 0) {
        // First 8 bytes are raw literals — copy directly to output
        if (lane < 8) dst[dst_offset + lane] = src[lane];
        __syncwarp();
        src += 8;
        initial_copy = 8;
    }

    // Literal stream (Type 0)
    const uint8_t* lit_ptr = src;
    uint32_t lit_size = 0;
    if (lane == 0) {
        lit_size = parseType0Header(src);
        lit_ptr = src;
        src += lit_size;
    }
    lit_size = __shfl_sync(0xFFFFFFFF, lit_size, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        lit_ptr = src - lit_size;
    }

    // Command stream (Type 0)
    const uint8_t* cmd_ptr;
    uint32_t cmd_size = 0;
    if (lane == 0) {
        cmd_size = parseType0Header(src);
        cmd_ptr = src;
        src += cmd_size;
    }
    cmd_size = __shfl_sync(0xFFFFFFFF, cmd_size, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        cmd_ptr = src - cmd_size;
    }

    // cmd_stream2_offset
    uint32_t cmd_stream2_offset = cmd_size;
    if (lane == 0 && sc_decomp_size > 0x10000) {
        uint16_t v; memcpy(&v, src, 2);
        cmd_stream2_offset = v;
        src += 2;
    }
    cmd_stream2_offset = __shfl_sync(0xFFFFFFFF, cmd_stream2_offset, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
    }

    // Off16 stream
    const uint8_t* off16_raw;
    uint32_t off16_count = 0;
    if (lane == 0) {
        uint16_t cnt; memcpy(&cnt, src, 2);
        off16_count = cnt;
        off16_raw = src + 2;
        src += 2 + off16_count * 2;
    }
    off16_count = __shfl_sync(0xFFFFFFFF, off16_count, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        off16_raw = src - off16_count * 2;
    }

    // Off32 stream sizes
    uint32_t off32_count1 = 0, off32_count2 = 0;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
    const uint8_t* len_stream;
    uint32_t len_avail = 0;

    if (lane == 0) {
        uint32_t tmp = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
        src += 3;
        if (tmp != 0) {
            off32_count1 = tmp >> 12;
            off32_count2 = tmp & 0xFFF;
            if (off32_count1 == 4095) { uint16_t v; memcpy(&v, src, 2); off32_count1 = v; src += 2; }
            if (off32_count2 == 4095) { uint16_t v; memcpy(&v, src, 2); off32_count2 = v; src += 2; }
            off32_raw1 = src;
            src += off32_count1 * 3;
            off32_raw2 = src;
            src += off32_count2 * 3;
        } else {
            off32_raw1 = src;
            off32_raw2 = src;
        }
        len_stream = src;
        len_avail = (uint32_t)((uintptr_t)src_end - (uintptr_t)src);
    }
    off32_count1 = __shfl_sync(0xFFFFFFFF, off32_count1, 0);
    off32_count2 = __shfl_sync(0xFFFFFFFF, off32_count2, 0);
    len_avail = __shfl_sync(0xFFFFFFFF, len_avail, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        len_stream = src;
        off32_raw2 = src - off32_count2 * 3;
        off32_raw1 = off32_raw2 - off32_count1 * 3;
    }

    __shared__ uint8_t off32_scratch[2][4096];  // up to 1024 off32 values per block
    uint8_t* off32_exp1 = off32_scratch[0];
    uint8_t* off32_exp2 = off32_scratch[1];

    // Expand 3-byte off32 values to 4-byte
    for (uint32_t i = lane; i < off32_count1; i += 32) {
        uint32_t v = (uint32_t)off32_raw1[i*3] | ((uint32_t)off32_raw1[i*3+1] << 8) | ((uint32_t)off32_raw1[i*3+2] << 16);
        memcpy(off32_exp1 + i*4, &v, 4);
    }
    for (uint32_t i = lane; i < off32_count2; i += 32) {
        uint32_t v = (uint32_t)off32_raw2[i*3] | ((uint32_t)off32_raw2[i*3+1] << 8) | ((uint32_t)off32_raw2[i*3+2] << 16);
        memcpy(off32_exp2 + i*4, &v, 4);
    }
    __syncwarp();

    decodeSubChunkBatched(
        cmd_ptr, cmd_size,
        lit_ptr, lit_size,
        off16_raw, off16_count,
        off32_exp1, off32_count1,
        off32_exp2, off32_count2,
        len_stream, len_avail,
        dst, sc_decomp_size, initial_copy, cmd_stream2_offset,
        dst_offset, mode
    );
}

// Full GPU L1 kernel: 1 block per SC group, parses raw compressed chunks.
extern "C" __global__ void slzFullDecompressL1Kernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    uint32_t total_chunks
) {
    const uint32_t group_id = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const uint32_t base_chunk = group_id * chunks_per_group;

    for (uint32_t c = 0; c < chunks_per_group; c++) {
        uint32_t chunk_idx = base_chunk + c;
        if (chunk_idx >= total_chunks) return;

        const SlzChunkDesc& ch = chunks[chunk_idx];
        if (ch.decomp_size == 0) continue;

        // Uncompressed chunk: warp-cooperative copy
        if (ch.flags & 1) {
            const uint8_t* src = compressed + ch.src_offset;
            for (uint32_t i = lane; i < ch.decomp_size; i += 32)
                dst[ch.dst_offset + i] = src[i];
            __syncwarp();
            continue;
        }

        // Memset chunk
        if (ch.flags & 2) {
            for (uint32_t i = lane; i < ch.decomp_size; i += 32)
                dst[ch.dst_offset + i] = ch.memset_fill;
            __syncwarp();
            continue;
        }

        // LZ-compressed chunk: iterate sub-chunks
        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > 0x20000) sc_size = 0x20000;  // sub_chunk_size = 128KB

            // Parse 3-byte sub-chunk header (big-endian)
            uint32_t chunkhdr = 0;
            if (lane == 0)
                chunkhdr = ((uint32_t)chunk_src[0] << 16) | ((uint32_t)chunk_src[1] << 8) | chunk_src[2];
            chunkhdr = __shfl_sync(0xFFFFFFFF, chunkhdr, 0);

            if (!(chunkhdr & 0x800000)) {
                // Non-LZ sub-chunk (entropy-only) — skip for now
                break;
            }

            uint32_t sc_comp_size = chunkhdr & 0x7FFFF;
            uint32_t sc_mode = (chunkhdr >> 19) & 0xF;
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                uint32_t base_offset_val = sc_dst_off;

                parseAndDecodeSubChunk(
                    sc_payload, sc_comp_size, sc_size,
                    dst, sc_dst_off, base_offset_val, sc_mode
                );
            } else {
                // Uncompressed sub-chunk: copy
                for (uint32_t i = lane; i < sc_size; i += 32)
                    dst[sc_dst_off + i] = sc_payload[i];
            }
            __syncwarp();

            chunk_src += 3 + sc_comp_size;
            sc_dst_off += sc_size;
            sc_remaining -= sc_size;
        }
    }
}

extern "C" {

int slz_gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

struct GpuLzRunsDesc {
    const uint8_t* cmd_data;    uint32_t cmd_size;
    const uint8_t* lit_data;    uint32_t lit_size;
    const uint8_t* off16_data;  uint32_t off16_count;
    const uint8_t* off32_data1; uint32_t off32_count1;
    const uint8_t* off32_data2; uint32_t off32_count2;
    const uint8_t* length_data; uint32_t length_avail;
    uint8_t* dst;               uint32_t dst_size;
    uint32_t initial_copy;
    uint32_t cmd_stream2_offset;
    uint32_t base_offset;
    uint32_t dst_offset;
};

int slz_gpu_process_lz_runs(const GpuLzRunsDesc* desc) {
    uint8_t *d_cmd, *d_lit, *d_off16, *d_off32_1, *d_off32_2, *d_len, *d_dst;
    uint32_t total_dst = desc->base_offset + desc->dst_size;

    cudaMalloc(&d_cmd, desc->cmd_size);
    cudaMalloc(&d_lit, desc->lit_size);
    cudaMalloc(&d_off16, desc->off16_count * 2 + 4);
    cudaMalloc(&d_off32_1, desc->off32_count1 * 4 + 4);
    cudaMalloc(&d_off32_2, desc->off32_count2 * 4 + 4);
    cudaMalloc(&d_len, desc->length_avail + 4);
    cudaMalloc(&d_dst, total_dst + 64);

    cudaMemcpy(d_cmd, desc->cmd_data, desc->cmd_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_lit, desc->lit_data, desc->lit_size, cudaMemcpyHostToDevice);
    if (desc->off16_count > 0)
        cudaMemcpy(d_off16, desc->off16_data, desc->off16_count * 2, cudaMemcpyHostToDevice);
    if (desc->off32_count1 > 0)
        cudaMemcpy(d_off32_1, desc->off32_data1, desc->off32_count1 * 4, cudaMemcpyHostToDevice);
    if (desc->off32_count2 > 0)
        cudaMemcpy(d_off32_2, desc->off32_data2, desc->off32_count2 * 4, cudaMemcpyHostToDevice);
    if (desc->length_avail > 0)
        cudaMemcpy(d_len, desc->length_data, desc->length_avail, cudaMemcpyHostToDevice);

    // Copy prior output for cross-sub-chunk match references
    if (desc->base_offset > 0)
        cudaMemcpy(d_dst, desc->dst - desc->base_offset,
                   desc->base_offset + desc->initial_copy, cudaMemcpyHostToDevice);
    else if (desc->initial_copy > 0)
        cudaMemcpy(d_dst, desc->dst, desc->initial_copy, cudaMemcpyHostToDevice);

    slzDecompressL1Kernel<<<1, 32>>>(
        d_cmd, desc->cmd_size, d_lit, desc->lit_size,
        d_off16, desc->off16_count,
        d_off32_1, desc->off32_count1,
        d_off32_2, desc->off32_count2,
        d_len, desc->length_avail,
        d_dst, desc->dst_size, desc->initial_copy,
        desc->cmd_stream2_offset,
        desc->base_offset, desc->base_offset);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();

    if (err == cudaSuccess)
        cudaMemcpy(desc->dst, d_dst + desc->base_offset,
                   desc->dst_size, cudaMemcpyDeviceToHost);

    cudaFree(d_cmd); cudaFree(d_lit); cudaFree(d_off16);
    cudaFree(d_off32_1); cudaFree(d_off32_2); cudaFree(d_len); cudaFree(d_dst);

    return (err == cudaSuccess) ? 0 : -1;
}

} // extern "C"
