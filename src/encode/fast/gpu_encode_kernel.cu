// ── StreamLZ GPU L1 Compress Kernel ─────────────────────────────
// Each warp (32 threads) compresses one 64KB chunk independently.
// Two warps per CUDA block for occupancy (blockDim 32,2,1).
//
// Architecture (inspired by nvCOMP LZ4 SASS):
//   1. Each lane loads 1 byte, builds 4-byte rolling key via SHFL.DOWN
//   2. Hash table probe for match candidates (16K × 2-byte entries)
//   3. Warp-parallel match extension via VOTE.ANY
//   4. Serial token emission on lane 0
//
// Output: StreamLZ raw sub-chunk format (same as CPU encodeSubChunkRaw)
//   [initial_copy:8] [lit_hdr:3] [lits] [cmd_hdr:3] [cmds]
//   [off16_count:2] [off16s] [off32_hdr:3(=0)] [lengths]
//
// Build: nvcc -ptx -arch=sm_89 -O3 gpu_encode_kernel.cu
// Embed: @embedFile("gpu_encode_kernel.ptx") in gpu_encoder.zig

#include <cstdint>

// ── Token format constants (same as decode kernel) ──────────────
static constexpr uint32_t TOKEN_SHORT_MIN    = 24;
static constexpr uint32_t TOKEN_LONG_LITERAL = 0;
static constexpr uint32_t TOKEN_LONG_NEAR    = 1;
static constexpr uint32_t LONG_LITERAL_BASE  = 64;
static constexpr uint32_t LONG_NEAR_BASE     = 91;
static constexpr uint32_t INITIAL_COPY       = 8;
static constexpr uint32_t HASH_BITS          = 14;
static constexpr uint32_t HASH_SIZE          = 1 << HASH_BITS;
static constexpr uint32_t HASH_MASK          = HASH_SIZE - 1;
static constexpr uint32_t MIN_MATCH          = 4;

struct CompressChunkDesc {
    uint32_t src_offset;     // byte offset into input buffer
    uint32_t src_size;       // decompressed chunk size (usually 64KB)
    uint32_t dst_offset;     // byte offset into output buffer
    uint32_t dst_capacity;   // max output bytes for this chunk
    uint32_t is_first;       // 1 if first chunk (needs initial_copy)
};

// ── Hash function (same as CPU fast hasher) ─────────────────────
__device__ uint32_t hashKey(uint32_t key) {
    key *= 0x9E3779B1u;
    return (key >> (32 - HASH_BITS)) & HASH_MASK;
}

// ── Build 4-byte key from consecutive bytes via warp shuffle ────
__device__ uint32_t buildKey4(uint8_t my_byte, int lane) {
    uint32_t b0 = my_byte;
    uint32_t b1 = __shfl_down_sync(0xFFFFFFFF, b0, 1);
    uint32_t b2 = __shfl_down_sync(0xFFFFFFFF, b0, 2);
    uint32_t b3 = __shfl_down_sync(0xFFFFFFFF, b0, 3);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// ── Warp-parallel match extension ───────────────────────────────
// Returns the length of the match starting at src[pos] vs src[ref].
// All 32 lanes compare in parallel, 32 bytes per iteration.
__device__ uint32_t extendMatch(
    const uint8_t* src, uint32_t pos, uint32_t ref,
    uint32_t max_len, int lane
) {
    uint32_t len = 0;
    while (len + 32 <= max_len) {
        uint32_t a = (pos + len + lane < pos + max_len) ? src[pos + len + lane] : 0;
        uint32_t b = (ref + len + lane < pos) ? src[ref + len + lane] : 0xFF;
        uint32_t mismatch = (a != b) ? 1 : 0;
        uint32_t mask = __ballot_sync(0xFFFFFFFF, mismatch);
        if (mask != 0) {
            // Find first mismatching lane
            uint32_t first = __ffs(mask) - 1;
            return len + first;
        }
        len += 32;
    }
    // Tail: check remaining bytes on lane 0
    if (lane == 0) {
        while (len < max_len && src[pos + len] == src[ref + len])
            len++;
    }
    return __shfl_sync(0xFFFFFFFF, len, 0);
}

// ── Write extended length to output ─────────────────────────────
__device__ uint32_t writeLength(uint8_t* dst, uint32_t pos, uint32_t value) {
    if (value <= 251) {
        dst[pos] = (uint8_t)value;
        return 1;
    }
    uint16_t extra = (uint16_t)((value - 252) / 4);
    uint8_t base = (uint8_t)(value - extra * 4);
    dst[pos] = base;
    memcpy(dst + pos + 1, &extra, 2);
    return 3;
}

// ── Main compress kernel ────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(64, 24) slzCompressL1Kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const CompressChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ hash_tables,   // HASH_SIZE u16 entries per chunk
    uint32_t* __restrict__ comp_sizes,    // output: compressed size per chunk
    uint32_t total_chunks
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    const int lane = threadIdx.x & 31;
    if (chunk_id >= total_chunks) return;

    const CompressChunkDesc& desc = descs[chunk_id];
    const uint8_t* src = input + desc.src_offset;
    uint8_t* dst = output + desc.dst_offset;
    const uint32_t src_size = desc.src_size;
    const uint32_t dst_cap = desc.dst_capacity;

    // Per-chunk hash table (16K × 2-byte entries, stored as u16 in u32 array)
    uint16_t* ht = (uint16_t*)(hash_tables + chunk_id * (HASH_SIZE / 2));

    // Initialize hash table to 0xFFFF (no match)
    for (uint32_t i = lane; i < HASH_SIZE; i += 32)
        ht[i] = 0xFFFF;
    __syncwarp();

    // ── Streams: accumulate in output buffer ──
    // Layout: [initial_copy] [lit_hdr:3] [lits...] [cmd_hdr:3] [cmds...]
    //         [off16_count:2] [off16s...] [off32_hdr:3] [lengths...]
    // We write streams to separate regions, then pack at the end.
    // Use local arrays for cmd, off16, length streams (small for 64KB chunks).

    // Worst case: every byte is a literal → lit_count = src_size
    // Worst case tokens: src_size / MIN_MATCH tokens
    // For 64KB: ~16K tokens, ~8K off16 values, ~100 length values

    // We'll write directly to output in two passes:
    // Pass 1: compress (accumulate streams in temp buffers)
    // Pass 2: pack streams into output format (lane 0)

    // Temp stream buffers in the output region (we have dst_cap space)
    // Split output into: [packed result] ... [temp streams at the end]
    // Tokens: max ~16K bytes
    // Off16: max ~16K × 2 = 32K bytes
    // Lengths: max ~1K bytes

    uint32_t lit_count = 0;
    uint32_t token_count = 0;
    uint32_t off16_count = 0;
    uint32_t length_count = 0;

    // Literals go directly into the output buffer right after the
    // reserved 3-byte header. Other streams (cmd, off16, length) go
    // to the tail of the output buffer as temp space, then get packed
    // into final position after compression completes.
    // dst_capacity is compress_bound (~src_size + overhead), so the
    // tail has plenty of room for temp streams.
    const uint32_t lit_start = (desc.is_first ? INITIAL_COPY : 0) + 3;
    uint8_t* lit_buf = dst + lit_start;

    // Temp stream buffers at output tail (won't overlap with final output
    // since compressed size < src_size, and these are past src_size offset)
    uint8_t* cmd_buf = dst + src_size;
    uint8_t* off16_buf = cmd_buf + 16384;
    uint8_t* len_buf = off16_buf + 32768;

    // Only lane 0 does the serial compression
    if (lane == 0) {
        uint32_t pos = desc.is_first ? INITIAL_COPY : 0;
        int32_t recent_offset = -8;
        uint32_t anchor = pos; // start of pending literals

        while (pos + MIN_MATCH <= src_size) {
            // Build 4-byte key at current position
            uint32_t key4 = (uint32_t)src[pos] | ((uint32_t)src[pos+1] << 8) |
                           ((uint32_t)src[pos+2] << 16) | ((uint32_t)src[pos+3] << 24);
            uint32_t h = hashKey(key4);

            // Probe hash table
            uint32_t ref = ht[h];
            ht[h] = (uint16_t)pos;

            if (ref == 0xFFFF || ref >= pos) {
                pos++;
                continue;
            }

            // Validate match
            uint32_t ref_key = (uint32_t)src[ref] | ((uint32_t)src[ref+1] << 8) |
                              ((uint32_t)src[ref+2] << 16) | ((uint32_t)src[ref+3] << 24);
            if (ref_key != key4) {
                pos++;
                continue;
            }

            // Extend match
            uint32_t max_match = src_size - pos;
            if (max_match > 65535) max_match = 65535;
            uint32_t match_len = MIN_MATCH;
            while (match_len < max_match && src[pos + match_len] == src[ref + match_len])
                match_len++;

            // Compute offset
            int32_t offset = -(int32_t)(pos - ref);
            uint32_t lit_len = pos - anchor;

            // Emit literals
            for (uint32_t i = 0; i < lit_len; i++)
                lit_buf[lit_count + i] = src[anchor + i];

            // Check if we can use recent offset
            uint32_t use_recent = (offset == recent_offset) ? 1 : 0;

            // Emit token(s)
            if (lit_len < 8 && match_len >= 8 && match_len <= 22) {
                // Standard short token: lit[2:0] match[6:3] recent[7]
                uint32_t token = (lit_len & 7) | (((match_len - 8) & 0xF) << 3) |
                                (use_recent << 7);
                // Ensure token >= 24
                if (token >= TOKEN_SHORT_MIN) {
                    cmd_buf[token_count++] = (uint8_t)token;
                    if (!use_recent) {
                        uint16_t off_val = (uint16_t)(-offset);
                        memcpy(off16_buf + off16_count * 2, &off_val, 2);
                        off16_count++;
                    }
                    lit_count += lit_len;
                    recent_offset = offset;
                    anchor = pos + match_len;
                    pos = anchor;
                    continue;
                }
            }

            // Long literal (token 0): lit_len >= 64
            if (lit_len >= 64 && match_len < 8) {
                // Emit long literal token
                cmd_buf[token_count++] = TOKEN_LONG_LITERAL;
                length_count += writeLength(len_buf, length_count, lit_len - LONG_LITERAL_BASE);
                lit_count += lit_len;
                anchor = pos;
                // Don't advance past — no match emitted
                pos++;
                continue;
            }

            // Fallback: emit literals as individual short tokens with match
            // For simplicity, emit a standard token if possible
            {
                uint32_t emit_lit = (lit_len < 8) ? lit_len : 0;
                uint32_t emit_match = match_len;

                // If lit_len >= 8, flush literals first as long literal
                if (lit_len >= 64) {
                    cmd_buf[token_count++] = TOKEN_LONG_LITERAL;
                    length_count += writeLength(len_buf, length_count, lit_len - LONG_LITERAL_BASE);
                    lit_count += lit_len;
                    emit_lit = 0;
                } else if (lit_len >= 8) {
                    // Emit literals one token at a time as lit-only
                    // Actually, just emit them inline — the decoder handles arbitrary lit counts
                    // by accumulating. For simplicity, emit a long literal.
                    cmd_buf[token_count++] = TOKEN_LONG_LITERAL;
                    length_count += writeLength(len_buf, length_count, lit_len - LONG_LITERAL_BASE);
                    lit_count += lit_len;
                    emit_lit = 0;
                }

                if (emit_match <= 22 && emit_match >= 8) {
                    uint32_t token = (emit_lit & 7) | (((emit_match - 8) & 0xF) << 3) |
                                    (use_recent << 7);
                    if (token >= TOKEN_SHORT_MIN) {
                        cmd_buf[token_count++] = (uint8_t)token;
                        if (!use_recent) {
                            uint16_t off_val = (uint16_t)(-offset);
                            memcpy(off16_buf + off16_count * 2, &off_val, 2);
                            off16_count++;
                        }
                        if (emit_lit > 0) lit_count += emit_lit;
                    } else {
                        // Token < 24: use long near match instead
                        cmd_buf[token_count++] = TOKEN_LONG_NEAR;
                        length_count += writeLength(len_buf, length_count, emit_match - LONG_NEAR_BASE);
                        uint16_t off_val = (uint16_t)(-offset);
                        memcpy(off16_buf + off16_count * 2, &off_val, 2);
                        off16_count++;
                        if (emit_lit > 0) lit_count += emit_lit;
                    }
                } else if (emit_match > 22) {
                    // Long near match (token 1)
                    cmd_buf[token_count++] = TOKEN_LONG_NEAR;
                    length_count += writeLength(len_buf, length_count, emit_match - LONG_NEAR_BASE);
                    uint16_t off_val = (uint16_t)(-offset);
                    memcpy(off16_buf + off16_count * 2, &off_val, 2);
                    off16_count++;
                    if (emit_lit > 0) lit_count += emit_lit;
                } else {
                    // match_len < 8: skip this match, advance by 1
                    pos++;
                    continue;
                }

                recent_offset = offset;
                anchor = pos + match_len;
                pos = anchor;
            }
        }

        // ── Trailing literals ──
        uint32_t trailing = src_size - anchor;
        for (uint32_t i = 0; i < trailing; i++)
            lit_buf[lit_count + i] = src[anchor + i];
        lit_count += trailing;
    }

    // Broadcast counts from lane 0
    lit_count = __shfl_sync(0xFFFFFFFF, lit_count, 0);
    token_count = __shfl_sync(0xFFFFFFFF, token_count, 0);
    off16_count = __shfl_sync(0xFFFFFFFF, off16_count, 0);
    length_count = __shfl_sync(0xFFFFFFFF, length_count, 0);

    // ── Pack output (lane 0) ──
    // Literals are already at dst[lit_start..lit_start+lit_count].
    // Backfill headers and append remaining streams after literals.
    if (lane == 0) {
        uint32_t out_pos = 0;

        // Initial copy (first chunk only)
        if (desc.is_first) {
            memcpy(dst, src, INITIAL_COPY);
            out_pos = INITIAL_COPY;
        }

        // Literal header: 3-byte BE count
        dst[out_pos] = (uint8_t)((lit_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((lit_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(lit_count & 0xFF);
        out_pos += 3;
        // Literal data is already at dst[out_pos..out_pos+lit_count]
        out_pos += lit_count;

        // Token stream: 3-byte BE header + data from temp
        dst[out_pos] = (uint8_t)((token_count >> 16) & 0xFF);
        dst[out_pos + 1] = (uint8_t)((token_count >> 8) & 0xFF);
        dst[out_pos + 2] = (uint8_t)(token_count & 0xFF);
        out_pos += 3;
        memcpy(dst + out_pos, cmd_buf, token_count);
        out_pos += token_count;

        // Off16: 2-byte LE count + data from temp
        dst[out_pos] = (uint8_t)(off16_count & 0xFF);
        dst[out_pos + 1] = (uint8_t)((off16_count >> 8) & 0xFF);
        out_pos += 2;
        memcpy(dst + out_pos, off16_buf, off16_count * 2);
        out_pos += off16_count * 2;

        // Off32 header: 3 bytes = 0 (no off32 in 64KB chunks)
        dst[out_pos] = 0;
        dst[out_pos + 1] = 0;
        dst[out_pos + 2] = 0;
        out_pos += 3;

        // Length stream from temp
        memcpy(dst + out_pos, len_buf, length_count);
        out_pos += length_count;

        comp_sizes[chunk_id] = out_pos;
    }
}
