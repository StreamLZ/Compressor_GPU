// ── StreamLZ GPU L1 Compress Kernel ─────────────────────────────
// Warp-parallel hash probe, serial token emission on lane 0.
// Each warp compresses one 64KB chunk independently.
//
// Build: nvcc -ptx -arch=sm_89 -O3 gpu_encode_kernel.cu

#include <cstdint>

static constexpr uint32_t HASH_BITS    = 14;
static constexpr uint32_t HASH_SIZE    = 1 << HASH_BITS;
static constexpr uint32_t HASH_MASK    = HASH_SIZE - 1;
static constexpr uint32_t MIN_MATCH    = 4;
static constexpr uint32_t INITIAL_COPY = 8;

struct CompressChunkDesc {
    uint32_t src_offset;
    uint32_t src_size;
    uint32_t dst_offset;
    uint32_t dst_capacity;
    uint32_t is_first;
};

__device__ uint32_t hashKey(uint32_t key) {
    key *= 0x9E3779B1u;
    return (key >> (32 - HASH_BITS)) & HASH_MASK;
}

__device__ uint32_t writeLength(uint8_t* buf, uint32_t pos, uint32_t value) {
    if (value <= 251) {
        buf[pos] = (uint8_t)value;
        return 1;
    }
    uint16_t extra = (uint16_t)((value - 252) / 4);
    uint8_t base = (uint8_t)(value - extra * 4);
    buf[pos] = base;
    memcpy(buf + pos + 1, &extra, 2);
    return 3;
}

extern "C" __global__ void __launch_bounds__(64, 24) slzCompressL1Kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const CompressChunkDesc* __restrict__ descs,
    uint16_t* __restrict__ hash_tables,
    uint32_t* __restrict__ comp_sizes,
    uint32_t total_chunks
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    const int lane = threadIdx.x & 31;
    if (chunk_id >= total_chunks) return;

    const CompressChunkDesc& desc = descs[chunk_id];
    const uint8_t* src = input + desc.src_offset;
    const uint32_t src_size = desc.src_size;
    uint16_t* ht = hash_tables + chunk_id * HASH_SIZE;

    // Init hash table (all lanes)
    for (uint32_t i = lane; i < HASH_SIZE; i += 32)
        ht[i] = 0xFFFF;
    __syncwarp();

    // Output streams (temp regions past src_size in output buffer)
    uint8_t* dst = output + desc.dst_offset;
    const uint32_t lit_data_start = (desc.is_first ? INITIAL_COPY : 0) + 3;
    uint8_t* lit_buf = dst + lit_data_start;
    uint8_t* cmd_buf = dst + src_size;
    uint8_t* off16_buf = cmd_buf + 16384;
    uint8_t* len_buf = off16_buf + 32768;

    uint32_t lit_count = 0, token_count = 0, off16_count = 0, length_count = 0;
    uint32_t anchor = desc.is_first ? INITIAL_COPY : 0;
    int32_t recent_offset = -8;

    // ── Serial greedy scan on lane 0 ──────────────────────────────
    // Only two token types for correctness:
    //   Standard (>= 24): lit 0-7, match 8-22
    //   Token 0: long literal (>= 64 bytes)
    // Matches > 22 are capped at 22. Matches < 8 are skipped.
    // Literals 8-63 are flushed as long literal (token 0, length = lit-64,
    // but lit < 64 needs special handling: emit as multiple short tokens
    // by advancing without a match, letting literals accumulate to >= 64
    // or drain as part of a short token with a match).
    if (lane == 0) {
        uint32_t pos = anchor;

        while (pos + MIN_MATCH <= src_size) {
            uint32_t key4 = (uint32_t)src[pos] | ((uint32_t)src[pos+1] << 8) |
                           ((uint32_t)src[pos+2] << 16) | ((uint32_t)src[pos+3] << 24);
            uint32_t h = hashKey(key4);
            uint32_t ref = ht[h];
            ht[h] = (uint16_t)pos;

            if (ref >= pos || ref == 0xFFFF) { pos++; continue; }

            uint32_t ref_key = (uint32_t)src[ref] | ((uint32_t)src[ref+1] << 8) |
                              ((uint32_t)src[ref+2] << 16) | ((uint32_t)src[ref+3] << 24);
            if (ref_key != key4) { pos++; continue; }

            // Extend match (cap at 22 for standard tokens only)
            uint32_t match_len = MIN_MATCH;
            uint32_t max_len = src_size - pos;
            if (max_len > 22) max_len = 22;
            while (match_len < max_len && src[pos + match_len] == src[ref + match_len])
                match_len++;

            if (match_len < 8) { pos++; continue; }

            int32_t offset = -(int32_t)(pos - ref);
            uint32_t lit_len = pos - anchor;
            uint32_t use_recent = (offset == recent_offset) ? 1 : 0;

            // Flush long literals (>= 64) as token 0
            while (lit_len >= 64) {
                uint32_t emit = (lit_len >= 64 + 251) ? 64 + 251 : lit_len;
                if (emit < 64) emit = 64;
                // Clamp to available literals
                if (emit > lit_len) emit = lit_len;
                for (uint32_t i = 0; i < emit; i++)
                    lit_buf[lit_count + i] = src[anchor + i];
                lit_count += emit;
                cmd_buf[token_count++] = 0;
                length_count += writeLength(len_buf, length_count, emit - 64);
                anchor += emit;
                lit_len -= emit;
            }

            // Now lit_len < 64. If < 8, emit standard token directly.
            // If 8-63, we can't use token 0 (needs >= 64). Skip this match
            // and let literals accumulate until they reach 64 or drain
            // with a future match that has lit < 8.
            if (lit_len >= 8) {
                pos++;
                continue;
            }

            // Standard short token: lit 0-7, match 8-22
            uint32_t token = (lit_len & 7) | (((match_len - 8) & 0xF) << 3) |
                            (use_recent << 7);
            // token >= 24 is guaranteed when match >= 8 (minimum (0 | 0<<3) = 0
            // but match-8 >= 0 so (0 << 3) = 0, token = lit. Need match >= 11
            // for token >= 24 when lit = 0: (3 << 3) = 24. Actually:
            // match=8 → (0<<3) = 0, token = lit. match=11 → (3<<3) = 24.
            // For match 8-10 with lit 0-2: token < 24. Skip these.
            if (token < 24) {
                pos++;
                continue;
            }

            for (uint32_t i = 0; i < lit_len; i++)
                lit_buf[lit_count + i] = src[anchor + i];
            lit_count += lit_len;
            cmd_buf[token_count++] = (uint8_t)token;
            if (!use_recent) {
                uint16_t off_val = (uint16_t)(-offset);
                memcpy(off16_buf + off16_count * 2, &off_val, 2);
                off16_count++;
            }
            recent_offset = offset;
            anchor = pos + match_len;
            pos = anchor;
        }

        // Trailing literals
        uint32_t trailing = src_size - anchor;
        if (trailing >= 64) {
            // Emit as long literal tokens
            while (trailing >= 64) {
                uint32_t emit = (trailing > 64 + 251) ? 64 + 251 : trailing;
                for (uint32_t i = 0; i < emit; i++)
                    lit_buf[lit_count + i] = src[anchor + i];
                lit_count += emit;
                cmd_buf[token_count++] = 0;
                length_count += writeLength(len_buf, length_count, emit - 64);
                anchor += emit;
                trailing -= emit;
            }
        }
        // Remaining trailing (< 64): just append as raw literals
        for (uint32_t i = 0; i < trailing; i++)
            lit_buf[lit_count + i] = src[anchor + i];
        lit_count += trailing;
    }
    __syncwarp();

    // ── Pack output (lane 0) ────────────────────────────────────
    if (lane == 0) {
        uint32_t out_pos = 0;
        if (desc.is_first) {
            memcpy(dst, src, INITIAL_COPY);
            out_pos = INITIAL_COPY;
        }

        // Literal header (3 bytes BE)
        dst[out_pos] = (uint8_t)((lit_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((lit_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(lit_count & 0xFF);
        out_pos += 3 + lit_count;

        // Token stream
        dst[out_pos] = (uint8_t)((token_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((token_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(token_count & 0xFF);
        out_pos += 3;
        memcpy(dst + out_pos, cmd_buf, token_count);
        out_pos += token_count;

        // Off16
        dst[out_pos] = (uint8_t)(off16_count & 0xFF);
        dst[out_pos + 1] = (uint8_t)((off16_count >> 8) & 0xFF);
        out_pos += 2;
        memcpy(dst + out_pos, off16_buf, off16_count * 2);
        out_pos += off16_count * 2;

        // Off32 (empty)
        dst[out_pos] = 0; dst[out_pos + 1] = 0; dst[out_pos + 2] = 0;
        out_pos += 3;

        // Length stream
        memcpy(dst + out_pos, len_buf, length_count);
        out_pos += length_count;

        comp_sizes[chunk_id] = out_pos;
    }
}
