// ── StreamLZ GPU L1 Compress Kernel ─────────────────────────────
// Warp-parallel greedy scan with shared-memory hash table.
// 32-bit hash entries support chunks up to 256KB (sc_group=1).
// 1 warp per block, 8KB hash table (2K × u32) in shared memory.
//
// Build: nvcc -ptx -arch=sm_89 -O3 gpu_encode_kernel.cu

#include <cstdint>

static constexpr uint32_t HASH_BITS    = 12;
static constexpr uint32_t HASH_SIZE    = 1 << HASH_BITS;
static constexpr uint32_t HASH_MASK    = HASH_SIZE - 1;
static constexpr uint32_t MIN_MATCH    = 4;
static constexpr uint32_t INITIAL_COPY = 8;
static constexpr uint32_t HASH_EMPTY   = 0xFFFFFFFFu;

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

__device__ void writeLengthValue(uint8_t* len_buf, uint32_t &len_count, uint32_t value) {
    if (value <= 251) {
        len_buf[len_count++] = (uint8_t)value;
    } else {
        uint32_t low2 = value & 3;
        uint8_t tag = (uint8_t)((low2 - 4) & 0xFF);
        len_buf[len_count++] = tag;
        uint16_t remainder = (uint16_t)((value - (low2 + 252)) >> 2);
        memcpy(len_buf + len_count, &remainder, 2);
        len_count += 2;
    }
}

__device__ void emitCmd(
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* len_buf, uint32_t &length_count,
    uint32_t lit_len,
    uint32_t match_len, uint32_t offset
) {
    uint32_t remaining_lit = lit_len;

    if (remaining_lit <= 7 && match_len <= 15 && offset <= 0xFFFF) {
        uint8_t token = (uint8_t)(remaining_lit + 8 * match_len);
        if (offset == 0) token |= 0x80;
        cmd_buf[token_count++] = token;
        if (offset != 0) {
            uint16_t off_val = (uint16_t)offset;
            memcpy(off16_buf + off16_count * 2, &off_val, 2);
            off16_count++;
        }
        return;
    }

    if (remaining_lit < 64) {
        while (remaining_lit > 7) {
            cmd_buf[token_count++] = 0x87;
            remaining_lit -= 7;
        }
    } else {
        writeLengthValue(len_buf, length_count, remaining_lit - 64);
        cmd_buf[token_count++] = 0x00;
        remaining_lit = 0;
        if (match_len == 0) return;
    }

    if (offset <= 0xFFFF && match_len <= 15) {
        uint8_t token = (uint8_t)(remaining_lit + 8 * match_len);
        if (offset == 0) token |= 0x80;
        cmd_buf[token_count++] = token;
        if (offset != 0) {
            uint16_t off_val = (uint16_t)offset;
            memcpy(off16_buf + off16_count * 2, &off_val, 2);
            off16_count++;
        }
        return;
    }

    if (offset <= 0xFFFF) {
        uint32_t current = (match_len < 15) ? match_len : 15;
        uint8_t token = (uint8_t)(remaining_lit + 8 * current);
        if (offset == 0) token |= 0x80;
        cmd_buf[token_count++] = token;
        if (offset != 0) {
            uint16_t off_val = (uint16_t)offset;
            memcpy(off16_buf + off16_count * 2, &off_val, 2);
            off16_count++;
        }
        uint32_t remaining_match = match_len - current;
        while (remaining_match > 0) {
            current = (remaining_match < 15) ? remaining_match : 15;
            cmd_buf[token_count++] = (uint8_t)(0x80 + 8 * current);
            remaining_match -= current;
        }
        return;
    }

    if (remaining_lit != 0) {
        cmd_buf[token_count++] = (uint8_t)(0x80 + remaining_lit);
    }
    cmd_buf[token_count++] = 1;
    writeLengthValue(len_buf, length_count, match_len > 91 ? match_len - 91 : 0);
    uint16_t off_val = (uint16_t)offset;
    memcpy(off16_buf + off16_count * 2, &off_val, 2);
    off16_count++;
}

// 1 warp per block, 8KB shared memory hash table (2K × u32)
extern "C" __global__ void __launch_bounds__(32, 6) slzCompressL1Kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const CompressChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ hash_tables_unused,
    uint32_t* __restrict__ comp_sizes,
    uint32_t total_chunks
) {
    __shared__ uint32_t ht[HASH_SIZE];

    const uint32_t chunk_id = blockIdx.x;
    const uint32_t lane = threadIdx.x & 31;
    if (chunk_id >= total_chunks) return;

    const CompressChunkDesc& desc = descs[chunk_id];
    const uint8_t* src = input + desc.src_offset;
    const uint32_t src_size = desc.src_size;

    for (uint32_t i = lane; i < HASH_SIZE; i += 32)
        ht[i] = HASH_EMPTY;
    __syncwarp();

    uint8_t* dst = output + desc.dst_offset;
    const uint32_t lit_data_start = (desc.is_first ? INITIAL_COPY : 0) + 3;
    uint8_t* lit_buf = dst + lit_data_start;
    // Scale temp stream regions by src_size
    uint8_t* cmd_buf = dst + src_size;
    uint8_t* off16_buf = cmd_buf + (src_size / 4);
    uint8_t* len_buf = off16_buf + (src_size / 2);

    uint32_t lit_count = 0, token_count = 0, off16_count = 0, length_count = 0;
    uint32_t cmd_stream2_offset = 0;
    uint32_t anchor = desc.is_first ? INITIAL_COPY : 0;
    int32_t recent_offset = -8;
    uint32_t pos = anchor;

    while (pos + MIN_MATCH <= src_size) {
        uint32_t my_pos = pos + lane;
        uint32_t my_byte = (my_pos < src_size) ? (uint32_t)src[my_pos] : 0u;

        uint32_t b1 = __shfl_down_sync(0xFFFFFFFF, my_byte, 1);
        uint32_t b2 = __shfl_down_sync(0xFFFFFFFF, my_byte, 2);
        uint32_t b3 = __shfl_down_sync(0xFFFFFFFF, my_byte, 3);
        uint32_t key4 = my_byte | (b1 << 8) | (b2 << 16) | (b3 << 24);

        uint32_t remaining = src_size - pos;
        uint32_t active_count;
        if (remaining >= 32) active_count = 29;
        else if (remaining >= 4) active_count = remaining - 3;
        else active_count = 0;
        bool is_active = (lane < active_count);

        uint32_t h = hashKey(key4);
        bool hash_match = false;
        uint32_t hash_ref = 0;

        if (is_active) {
            uint32_t ref_val = ht[h];
            if (ref_val != HASH_EMPTY && ref_val < my_pos && (my_pos - ref_val) <= 0xFFFF) {
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

        uint32_t warp_match = __match_any_sync(0xFFFFFFFF, key4);
        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, is_active);
        warp_match &= active_mask;
        uint32_t lower_mask = warp_match & ((1u << lane) - 1);

        bool has_match = hash_match;
        uint32_t my_ref = hash_ref;

        if (is_active && lower_mask != 0) {
            uint32_t intra_lane = __ffs(lower_mask) - 1;
            has_match = true;
            my_ref = pos + intra_lane;
        }

        uint32_t match_ballot = __ballot_sync(0xFFFFFFFF, has_match);

        {
            uint32_t write_limit = (match_ballot != 0)
                ? (uint32_t)(__ffs(match_ballot) - 1) : active_count;
            if (is_active && lane <= write_limit) {
                ht[h] = my_pos;
            }
        }

        if (match_ballot == 0) {
            pos += active_count;
            continue;
        }

        uint32_t first_lane = __ffs(match_ballot) - 1;
        uint32_t match_pos = pos + first_lane;
        uint32_t match_ref = __shfl_sync(0xFFFFFFFF, my_ref, first_lane);

        uint32_t match_len = MIN_MATCH;
        uint32_t max_match = src_size - match_pos;
        for (uint32_t ext = MIN_MATCH; ext < max_match; ext += 32) {
            uint32_t check = ext + lane;
            bool mm;
            if (check >= max_match)
                mm = true;
            else
                mm = (src[match_pos + check] != src[match_ref + check]);
            uint32_t mm_mask = __ballot_sync(0xFFFFFFFF, mm);
            if (mm_mask != 0) {
                match_len = ext + __ffs(mm_mask) - 1;
                goto ext_done;
            }
        }
        match_len = max_match;
        ext_done:

        {
            uint32_t lit_len = match_pos - anchor;
            for (uint32_t i = lane; i < lit_len; i += 32)
                lit_buf[lit_count + i] = src[anchor + i];

            lit_count += lit_len;

            int32_t neg_off = -(int32_t)(match_pos - match_ref);
            uint32_t off_param = (neg_off == recent_offset) ? 0 : (uint32_t)(-neg_off);

            if (lit_len <= 7 && match_len <= 15 && off_param <= 0xFFFF) {
                if (lane == 0) {
                    uint8_t token = (uint8_t)(lit_len + 8 * match_len);
                    if (off_param == 0) token |= 0x80;
                    cmd_buf[token_count++] = token;
                    if (off_param != 0) {
                        uint16_t ov = (uint16_t)off_param;
                        memcpy(off16_buf + off16_count * 2, &ov, 2);
                        off16_count++;
                    }
                }
            } else {
                if (lane == 0) {
                    emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                            len_buf, length_count, lit_len, match_len, off_param);
                }
            }
            recent_offset = neg_off;

            // Track block boundary for cmd_stream2_offset
            if (cmd_stream2_offset == 0 && match_pos + match_len > 0x10000 && lane == 0) {
                cmd_stream2_offset = token_count;
            }
        }

        anchor = match_pos + match_len;
        pos = anchor;
    }

    {
        uint32_t trailing = src_size - anchor;
        if (trailing > 0) {
            for (uint32_t i = lane; i < trailing; i += 32)
                lit_buf[lit_count + i] = src[anchor + i];
            lit_count += trailing;

            if (lane == 0) {
                emitCmd(cmd_buf, token_count, off16_buf, off16_count,
                        len_buf, length_count, trailing, 0, 0);
            }
        }
    }
    __syncwarp();

    if (lane == 0) {
        uint32_t out_pos = 0;
        if (desc.is_first) {
            memcpy(dst, src, INITIAL_COPY);
            out_pos = INITIAL_COPY;
        }

        dst[out_pos] = (uint8_t)((lit_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((lit_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(lit_count & 0xFF);
        out_pos += 3 + lit_count;

        dst[out_pos] = (uint8_t)((token_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((token_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(token_count & 0xFF);
        out_pos += 3;
        memcpy(dst + out_pos, cmd_buf, token_count);
        out_pos += token_count;

        // cmd_stream2_offset: written for sub-chunks > 64KB
        if (src_size > 0x10000) {
            uint16_t cs2o = (cmd_stream2_offset > 0) ? (uint16_t)cmd_stream2_offset : (uint16_t)token_count;
            memcpy(dst + out_pos, &cs2o, 2);
            out_pos += 2;
        }

        dst[out_pos] = (uint8_t)(off16_count & 0xFF);
        dst[out_pos + 1] = (uint8_t)((off16_count >> 8) & 0xFF);
        out_pos += 2;
        memcpy(dst + out_pos, off16_buf, off16_count * 2);
        out_pos += off16_count * 2;

        dst[out_pos] = 0; dst[out_pos + 1] = 0; dst[out_pos + 2] = 0;
        out_pos += 3;

        memcpy(dst + out_pos, len_buf, length_count);
        out_pos += length_count;

        comp_sizes[chunk_id] = out_pos;
    }
}
