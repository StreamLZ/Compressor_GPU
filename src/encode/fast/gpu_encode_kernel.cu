// ── StreamLZ GPU L1 Compress Kernel ─────────────────────────────
// Exact port of CPU writeOffset/writeComplexOffset token emission.
// Serial greedy scan on lane 0 with hash table in global memory.
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

// ── writeLengthValue: exact port of CPU ─────────────────────────
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

// ── writeComplexOffset: exact port of CPU ───────────────────────
// offset=0 means "use recent offset" (reuse flag)
__device__ void emitComplex(
    uint8_t* lit_buf, uint32_t &lit_count,
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* len_buf, uint32_t &length_count,
    const uint8_t* src,
    uint32_t anchor, uint32_t lit_len,
    uint32_t match_len, uint32_t offset  // offset=0 means recent
) {
    // Copy ALL literals to literal stream
    for (uint32_t i = 0; i < lit_len; i++)
        lit_buf[lit_count + i] = src[anchor + i];
    lit_count += lit_len;

    uint32_t remaining_lit = lit_len;

    if (remaining_lit < 64) {
        // Drain 8-63 with 0x87 continuation tokens (7 lit + 8 match at recent)
        while (remaining_lit > 7) {
            cmd_buf[token_count++] = 0x87;
            remaining_lit -= 7;
        }
    } else {
        // Long literal: token 0 + length value
        writeLengthValue(len_buf, length_count, remaining_lit - 64);
        cmd_buf[token_count++] = 0x00;
        remaining_lit = 0;
        if (match_len == 0) return;
    }

    if (offset <= 0xFFFF && match_len <= 15) {
        // Near-offset match: first token includes remaining literals
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
        // Near-offset, match > 15: first token takes up to 15, then continuations
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

    // Far-offset path (off32) — not used for 64KB chunks
    // but included for completeness
    if (remaining_lit != 0) {
        cmd_buf[token_count++] = (uint8_t)(0x80 + remaining_lit);
    }
    cmd_buf[token_count++] = 1; // TOKEN_LONG_NEAR as fallback
    writeLengthValue(len_buf, length_count, match_len > 91 ? match_len - 91 : 0);
    uint16_t off_val = (uint16_t)offset;
    memcpy(off16_buf + off16_count * 2, &off_val, 2);
    off16_count++;
}

// ── writeOffset: exact port of CPU fast path ────────────────────
__device__ void emitToken(
    uint8_t* lit_buf, uint32_t &lit_count,
    uint8_t* cmd_buf, uint32_t &token_count,
    uint8_t* off16_buf, uint32_t &off16_count,
    uint8_t* len_buf, uint32_t &length_count,
    const uint8_t* src,
    uint32_t anchor, uint32_t lit_len,
    uint32_t match_len, uint32_t offset  // offset=0 means recent
) {
    if (lit_len <= 7 && match_len <= 15 && offset <= 0xFFFF) {
        // Fast path: single token byte
        for (uint32_t i = 0; i < lit_len; i++)
            lit_buf[lit_count + i] = src[anchor + i];
        lit_count += lit_len;

        uint8_t token = (uint8_t)(lit_len + 8 * match_len);
        if (offset == 0) token |= 0x80;
        cmd_buf[token_count++] = token;
        if (offset != 0) {
            uint16_t off_val = (uint16_t)offset;
            memcpy(off16_buf + off16_count * 2, &off_val, 2);
            off16_count++;
        }
        return;
    }
    emitComplex(lit_buf, lit_count, cmd_buf, token_count,
                off16_buf, off16_count, len_buf, length_count,
                src, anchor, lit_len, match_len, offset);
}

// ── Main compress kernel ────────────────────────────────────────
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

    // Output streams
    uint8_t* dst = output + desc.dst_offset;
    const uint32_t lit_data_start = (desc.is_first ? INITIAL_COPY : 0) + 3;
    uint8_t* lit_buf = dst + lit_data_start;
    uint8_t* cmd_buf = dst + src_size;
    uint8_t* off16_buf = cmd_buf + 16384;
    uint8_t* len_buf = off16_buf + 32768;

    uint32_t lit_count = 0, token_count = 0, off16_count = 0, length_count = 0;
    uint32_t anchor = desc.is_first ? INITIAL_COPY : 0;
    int32_t recent_offset = -8;

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

            // Extend match (no cap — emitToken handles arbitrary lengths)
            uint32_t match_len = MIN_MATCH;
            uint32_t max_len = src_size - pos;
            while (match_len < max_len && src[pos + match_len] == src[ref + match_len])
                match_len++;

            uint32_t lit_len = pos - anchor;
            int32_t neg_offset = -(int32_t)(pos - ref);
            // offset=0 means "use recent", otherwise actual distance
            uint32_t offset_param = (neg_offset == recent_offset) ? 0 : (uint32_t)(-neg_offset);

            emitToken(lit_buf, lit_count, cmd_buf, token_count,
                      off16_buf, off16_count, len_buf, length_count,
                      src, anchor, lit_len, match_len, offset_param);

            recent_offset = neg_offset;
            anchor = pos + match_len;
            pos = anchor;
        }

        // Trailing literals — emit as lit-only (match_len=0)
        uint32_t trailing = src_size - anchor;
        if (trailing > 0) {
            // Use emitComplex with match_len=0 to handle long trailing
            emitComplex(lit_buf, lit_count, cmd_buf, token_count,
                       off16_buf, off16_count, len_buf, length_count,
                       src, anchor, trailing, 0, 0);
        }
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

        // Off32 (empty for 64KB chunks)
        dst[out_pos] = 0; dst[out_pos + 1] = 0; dst[out_pos + 2] = 0;
        out_pos += 3;

        // Length stream
        memcpy(dst + out_pos, len_buf, length_count);
        out_pos += length_count;

        comp_sizes[chunk_id] = out_pos;
    }
}
