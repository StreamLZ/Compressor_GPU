// ── StreamLZ GPU Decode Kernel ──────────────────────────────────
// Each warp (32 threads) decodes one StreamLZ chunk independently.
// Two warps are packed per CUDA thread-block for SM occupancy
// (blockDim 32,2,1 — the two warps do NOT cooperate).
// Lane 0 of each warp parses tokens serially; all 32 lanes
// participate in literal and match byte copies.
//
// Decode paths:
//   decodeSubChunkL1      — 64KB raw-mode (no off32, no delta, no block split)
//   decodeSubChunk        — general (off32, delta literals, 2-block sub-chunks)
//   parseAndDecodeSubChunk — parses stream headers, dispatches to above
//
// Build: nvcc -cubin -arch=sm_89 -O3 gpu_decode_kernel.cu
// Embed: @embedFile("gpu_decode_kernel.cubin") in gpu_driver.zig

#include <cstdint>

// ── StreamLZ token format constants ────────────────────────────
// Tokens encode literal-length + match-length + offset type in one byte.
//   token >= 24:  standard — lit[2:0], match[6:3], use_recent[7], off16
//   token == 0:   long literal — length from length stream + 64
//   token == 1:   long near match — length from length stream + 91, off16
//   token == 2:   long far match — length from length stream + 29, off32
//   token 3-23:   short far match — match = token + 5, off32
static constexpr uint32_t TOKEN_SHORT_MIN      = 24;
static constexpr uint32_t TOKEN_LONG_LITERAL   = 0;
static constexpr uint32_t TOKEN_LONG_NEAR      = 1;
static constexpr uint32_t TOKEN_LONG_FAR       = 2;
static constexpr uint32_t LONG_LITERAL_BASE    = 64;
static constexpr uint32_t LONG_NEAR_BASE       = 91;
static constexpr uint32_t LONG_FAR_BASE        = 29;
static constexpr uint32_t SHORT_FAR_BASE       = 5;
static constexpr uint32_t BLOCK_SIZE           = 0x10000;  // 64KB

// ── Chunk descriptor (extern-matched with gpu_driver.zig ChunkDesc) ────
struct SlzChunkDesc {
    uint32_t src_offset;    // byte offset into compressed block
    uint32_t comp_size;     // compressed payload size
    uint32_t decomp_size;   // decompressed size (usually 256KB)
    uint32_t dst_offset;    // absolute output position
    uint32_t flags;         // bit 0: uncompressed, bit 1: memset
    uint8_t  memset_fill;   // fill byte for memset chunks
    uint8_t  _pad[3];
};

// ── Extended length decode ─────────────────────────────────────
// Reads a variable-length count from the length stream. Uses a uint32
// offset from len_data base to avoid 64-bit pointer register pressure.
__device__ uint32_t readLength(const uint8_t* len_data, uint32_t &len_off, uint32_t len_avail) {
    if (len_off >= len_avail) return 0;
    uint32_t v = len_data[len_off];
    if (v > 251) {
        if (len_off + 2 >= len_avail) { len_off++; return v; }
        uint16_t extra;
        memcpy(&extra, len_data + len_off + 1, 2);
        v += (uint32_t)extra * 4;
        len_off += 2;
    }
    len_off++;
    return v;
}

// ── L1 raw-mode sub-chunk decoder ──────────────────────────────
// Streamlined single-block raw decoder for 64KB chunks.
// No off32, no delta, no block split — fastest path for L1 data.
// Register-optimized: uses uint32 offsets instead of 64-bit pointers,
// removes redundant bounds checks and shuffles.
__device__ void decodeSubChunkL1(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off16_hi, const uint8_t* __restrict__ off16_lo,
    uint32_t off16_split,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t dst_offset
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = -8;
    uint32_t len_off = 0;

    // Prefetch first token
    uint32_t next_token = 0;
    if (lane == 0 && cmd_pos < cmd_size) next_token = cmd[cmd_pos];

    while (cmd_pos < cmd_size) {
        uint32_t lit_len = 0, match_len = 0;
        int32_t match_offset = recent_offset;
        uint32_t use_recent = 0;

        // ── Token parse (lane 0 only) ──
        if (lane == 0) {
            uint32_t token = next_token;
            // Prefetch next token while processing this one
            if (cmd_pos + 1 < cmd_size) next_token = cmd[cmd_pos + 1];
            if (token >= TOKEN_SHORT_MIN) {
                lit_len = token & 7;
                match_len = (token >> 3) & 0xF;
                use_recent = (token >> 7) & 1;
                if (!use_recent && off16_pos < off16_count) {
                    uint16_t v;
                    if (off16_split) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        memcpy(&v, off16_raw + off16_pos * 2, 2);
                    }
                    match_offset = -(int32_t)v;
                    off16_pos++;
                }
            } else if (token == TOKEN_LONG_LITERAL) {
                use_recent = 1;
                lit_len = readLength(len_data, len_off, len_avail) + LONG_LITERAL_BASE;
            } else if (token == TOKEN_LONG_NEAR) {
                match_len = readLength(len_data, len_off, len_avail) + LONG_NEAR_BASE;
                if (off16_pos < off16_count) {
                    uint16_t v;
                    if (off16_split) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        memcpy(&v, off16_raw + off16_pos * 2, 2);
                    }
                    match_offset = -(int32_t)v; off16_pos++;
                }
            }
        }
        cmd_pos++;

        // ── Broadcast parsed values from lane 0 to all lanes ──
        lit_len      = __shfl_sync(0xFFFFFFFF, lit_len, 0);
        match_len    = __shfl_sync(0xFFFFFFFF, match_len, 0);
        match_offset = __shfl_sync(0xFFFFFFFF, match_offset, 0);
        use_recent   = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        off16_pos    = __shfl_sync(0xFFFFFFFF, off16_pos, 0);

        // ── Warp-cooperative literal copy ──
        if (lit_len > 0) {
            for (uint32_t i = lane; i < lit_len; i += 32)
                dst[dst_pos + i] = lit[lit_pos + i];
            __syncwarp();
        }
        dst_pos += lit_len;
        lit_pos += lit_len;

        // ── Warp-cooperative match copy ──
        if (match_len > 0) {
            uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
            int32_t match_dist = -match_offset;
            if (match_dist >= (int32_t)match_len && match_len > 1) {
                for (uint32_t i = lane; i < match_len; i += 32)
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
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
        len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
    }

    // ── Trailing literals (bytes after the last token) ──
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    for (uint32_t i = lane; i < trailing; i += 32)
        dst[dst_pos + i] = lit[lit_pos + i];

}

// ── General sub-chunk decoder ──────────────────────────────────
// Full-featured path: handles off32, delta literals, and multi-block
// sub-chunks (up to 2 blocks of 64KB each).
// mode: 0 = delta literals, 1 = raw literals
__device__ void decodeSubChunk(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off16_hi, const uint8_t* __restrict__ off16_lo,
    uint32_t off16_split,
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
    uint32_t len_off = 0;
    const uint8_t* off32_cur = off32_raw1;
    uint32_t off32_count_cur = off32_count1;
    uint32_t block_dst_start = dst_offset;

    uint32_t block_cmd_end = (cmd_stream2_offset > 0 && cmd_stream2_offset < cmd_size)
        ? cmd_stream2_offset : cmd_size;
    int block_iter = 0;

    // Prefetch first token
    uint32_t next_token_g = 0;
    if (lane == 0 && cmd_pos < block_cmd_end) next_token_g = cmd[cmd_pos];

    for (;;) {
    while (cmd_pos < block_cmd_end) {
        uint32_t token = 0, lit_len = 0, match_len = 0;
        int32_t match_offset = recent_offset;
        uint32_t use_recent = 0, token_type = 0;

        // ── Token parse (lane 0 only) ──
        if (lane == 0) {
            token = next_token_g;
            cmd_pos++;
            if (cmd_pos < block_cmd_end) next_token_g = cmd[cmd_pos];
            if (token >= TOKEN_SHORT_MIN) {
                token_type = 0;
                lit_len = token & 7;
                match_len = (token >> 3) & 0xF;
                use_recent = (token >> 7) & 1;
                if (!use_recent && off16_pos < off16_count) {
                    uint16_t v;
                    if (off16_split) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        memcpy(&v, off16_raw + off16_pos * 2, 2);
                    }
                    match_offset = -(int32_t)v;
                    off16_pos++;
                }
            } else if (token == TOKEN_LONG_LITERAL) {
                token_type = 1;
                lit_len = readLength(len_data, len_off, len_avail) + LONG_LITERAL_BASE;
            } else if (token == TOKEN_LONG_NEAR) {
                token_type = 2;
                match_len = readLength(len_data, len_off, len_avail) + LONG_NEAR_BASE;
                if (off16_pos < off16_count) {
                    uint16_t v;
                    if (off16_split) {
                        v = (uint16_t)off16_lo[off16_pos] | ((uint16_t)off16_hi[off16_pos] << 8);
                    } else {
                        memcpy(&v, off16_raw + off16_pos * 2, 2);
                    }
                    match_offset = -(int32_t)v; off16_pos++;
                }
                use_recent = 0;
            } else if (token == TOKEN_LONG_FAR) {
                token_type = 4;
                match_len = readLength(len_data, len_off, len_avail) + LONG_FAR_BASE;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    match_offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            } else {
                token_type = 3;
                match_len = token + SHORT_FAR_BASE;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    match_offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            }
        }

        // ── Broadcast parsed values from lane 0 to all lanes ──
        token_type   = __shfl_sync(0xFFFFFFFF, token_type, 0);
        lit_len      = __shfl_sync(0xFFFFFFFF, lit_len, 0);
        match_len    = __shfl_sync(0xFFFFFFFF, match_len, 0);
        match_offset = __shfl_sync(0xFFFFFFFF, match_offset, 0);
        use_recent   = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        cmd_pos      = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        lit_pos      = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        off16_pos    = __shfl_sync(0xFFFFFFFF, off16_pos, 0);
        off32_pos    = __shfl_sync(0xFFFFFFFF, off32_pos, 0);

        // ── Warp-cooperative literal copy ──
        if (lit_len > 0) {
            if (mode == 0) {
                for (uint32_t i = lane; i < lit_len; i += 32)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                        uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                        dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
                    }
            } else {
                for (uint32_t i = lane; i < lit_len; i += 32)
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
            if (match_dist >= (int32_t)match_len && match_len > 1) {
                for (uint32_t i = lane; i < match_len; i += 32)
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

        dst_pos   = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
        lit_pos   = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        if (!use_recent && (token_type != 1)) recent_offset = match_offset;
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
        len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
    }

    // ── Per-block trailing literals (at 64KB boundary) ──
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    {
        uint32_t block_end = block_dst_start + BLOCK_SIZE;
        if (block_end > dst_end_abs) block_end = dst_end_abs;
        uint32_t block_trailing = (block_end > dst_pos) ? (block_end - dst_pos) : 0;
        if (mode == 0) {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                    uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                    dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
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

    // ── Advance to block 2 ──
    block_iter++;
    if (block_iter >= 2) break;
    block_cmd_end = cmd_size;
    off32_cur = off32_raw2;
    off32_count_cur = off32_count2;
    off32_pos = 0;
    block_dst_start = dst_pos;
    if (lane == 0 && cmd_pos < block_cmd_end) next_token_g = cmd[cmd_pos];
    }

    // ── Final trailing literals ──
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    if (mode == 0) {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
            }
    } else {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}

// ── Type 0 entropy header parser ───────────────────────────────
// Reads a 2- or 3-byte big-endian size prefix from the compressed stream.
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

// ── Parsed sub-chunk streams — passed from header parser to decoder ──
struct ParsedStreams {
    const uint8_t* lit_ptr;
    const uint8_t* cmd_ptr;
    const uint8_t* off16_raw;
    const uint8_t* off16_hi;
    const uint8_t* off16_lo;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
    const uint8_t* len_stream;
    uint32_t lit_size;
    uint32_t cmd_size;
    uint32_t off16_count;
    uint32_t off16_split;  // 1 = hi/lo split format from GPU tANS, 0 = interleaved u16
    uint32_t off32_count1;
    uint32_t off32_count2;
    uint32_t len_avail;
    uint32_t cmd_stream2_offset;
    uint32_t initial_copy;
};

// ── Helper: skip an entropy-coded stream header + payload ──────
// Reads the type-0 or type-1 header at *src, advances src past the
// header + payload, and returns the decompressed size.
__device__ uint32_t skipEntropyStream(const uint8_t* &src) {
    uint32_t ct = (src[0] >> 4) & 0x7;
    if (ct == 0) {
        uint32_t sz = parseType0Header(src);
        src += sz;
        return sz;
    } else if (ct == 1) {
        // tANS header
        uint32_t tans_comp_size, tans_dst_size;
        if (src[0] >= 0x80) {
            uint32_t bits = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
            tans_comp_size = bits & 0x3FF;
            tans_dst_size = tans_comp_size + ((bits >> 10) & 0x3FF) + 1;
            src += 3 + tans_comp_size;
        } else {
            uint32_t bits = ((uint32_t)src[1] << 24) | ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 8) | src[4];
            tans_comp_size = bits & 0x3FFFF;
            tans_dst_size = ((((bits >> 18) | ((uint32_t)src[0] << 14)) & 0x3FFFF)) + 1;
            src += 5 + tans_comp_size;
        }
        return tans_dst_size;
    } else if (ct == 7) {
        // Paired-primary: [0x70][countA:u24][inner type-6 stream]. Returns
        // this unit's count (countA); advances past the whole marker+stream.
        uint32_t count_a = ((uint32_t)src[1] << 16) | ((uint32_t)src[2] << 8) | src[3];
        const uint8_t* inner = src + 4;
        if (inner[0] >= 0x80) {
            uint32_t bits = ((uint32_t)inner[0] << 16) | ((uint32_t)inner[1] << 8) | inner[2];
            src = inner + 3 + (bits & 0x3FF);
        } else {
            uint32_t bits = ((uint32_t)inner[1] << 24) | ((uint32_t)inner[2] << 16) | ((uint32_t)inner[3] << 8) | inner[4];
            src = inner + 5 + (bits & 0x3FFFF);
        }
        return count_a;
    } else if (ct == 5) {
        // Paired-secondary: [0x50][countA:u24][countB:u24], no payload.
        uint32_t count_b = ((uint32_t)src[4] << 16) | ((uint32_t)src[5] << 8) | src[6];
        src += 7;
        return count_b;
    } else {
        // Huffman type 2/4 or tANS32 type 6 — parse header to skip
        uint32_t comp_size, dst_size;
        if (src[0] >= 0x80) {
            uint32_t bits = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
            comp_size = bits & 0x3FF;
            dst_size = comp_size + ((bits >> 10) & 0x3FF) + 1;
            src += 3 + comp_size;
        } else {
            uint32_t bits = ((uint32_t)src[1] << 24) | ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 8) | src[4];
            comp_size = bits & 0x3FFFF;
            dst_size = ((((bits >> 18) | ((uint32_t)src[0] << 14)) & 0x3FFFF)) + 1;
            src += 5 + comp_size;
        }
        return dst_size;
    }
}

// ── Sub-chunk header parser — __noinline__ to free registers ────
// Parses all stream headers, returns pointers/sizes in ParsedStreams.
// Registers used here are freed before the decode hot loop runs.
__device__ __noinline__ void parseSubChunkHeaders(
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint8_t* tans_scratch_chunk,
    uint8_t* tans_tok_scratch_chunk,
    uint8_t* tans_off16_scratch_chunk,
    ParsedStreams& ps
) {
    const int lane = threadIdx.x & 31;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    ps.initial_copy = 0;
    if (base_offset == 0) {
        if (lane < 8) dst[dst_offset + lane] = src[lane];
        __syncwarp();
        src += 8;
        ps.initial_copy = 8;
    }

    // Literal stream
    const uint8_t* lit_ptr = src;
    uint32_t lit_size = 0;
    uint32_t lit_is_tans = 0;
    if (lane == 0) {
        uint32_t chunk_type = (src[0] >> 4) & 0x7;
        if (chunk_type == 0) {
            lit_size = parseType0Header(src);
            lit_ptr = src;
            src += lit_size;
        } else if ((chunk_type == 1 || chunk_type == 6) && tans_scratch_chunk != nullptr) {
            uint32_t tans_comp_size, tans_dst_size;
            if (src[0] >= 0x80) {
                uint32_t bits = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
                tans_comp_size = bits & 0x3FF;
                tans_dst_size = tans_comp_size + ((bits >> 10) & 0x3FF) + 1;
                src += 3 + tans_comp_size;
            } else {
                uint32_t bits = ((uint32_t)src[1] << 24) | ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 8) | src[4];
                tans_comp_size = bits & 0x3FFFF;
                tans_dst_size = ((((bits >> 18) | ((uint32_t)src[0] << 14)) & 0x3FFFF)) + 1;
                src += 5 + tans_comp_size;
            }
            lit_ptr = tans_scratch_chunk;
            lit_size = tans_dst_size;
            lit_is_tans = 1;
        } else if (chunk_type == 7 && tans_scratch_chunk != nullptr) {
            // Paired-primary: [0x70][countA:u24 BE][inner type-6 tANS stream].
            // This unit's literals are countA symbols at the start of the
            // decoded combined buffer (the tANS kernel split-wrote them here).
            uint32_t count_a = ((uint32_t)src[1] << 16) | ((uint32_t)src[2] << 8) | src[3];
            const uint8_t* inner = src + 4;
            if (inner[0] >= 0x80) {
                uint32_t bits = ((uint32_t)inner[0] << 16) | ((uint32_t)inner[1] << 8) | inner[2];
                src = inner + 3 + (bits & 0x3FF);
            } else {
                uint32_t bits = ((uint32_t)inner[1] << 24) | ((uint32_t)inner[2] << 16) | ((uint32_t)inner[3] << 8) | inner[4];
                src = inner + 5 + (bits & 0x3FFFF);
            }
            lit_ptr = tans_scratch_chunk;
            lit_size = count_a;
            lit_is_tans = 1;
        } else if (chunk_type == 5 && tans_scratch_chunk != nullptr) {
            // Paired-secondary: [0x50][countA:u24 BE][countB:u24 BE], no payload.
            // This unit's literals are countB symbols, split-written by the
            // tANS kernel into this chunk's region (dst_offset_b).
            uint32_t count_b = ((uint32_t)src[4] << 16) | ((uint32_t)src[5] << 8) | src[6];
            src += 7;
            lit_ptr = tans_scratch_chunk;
            lit_size = count_b;
            lit_is_tans = 1;
        } else {
            lit_size = 0;
            lit_ptr = src;
        }
    }
    lit_size = __shfl_sync(0xFFFFFFFF, lit_size, 0);
    lit_is_tans = __shfl_sync(0xFFFFFFFF, lit_is_tans, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        if (lit_is_tans) lit_ptr = tans_scratch_chunk;
        else lit_ptr = src - lit_size;
    }
    ps.lit_ptr = lit_ptr;
    ps.lit_size = lit_size;

    // Command stream (tokens)
    const uint8_t* cmd_ptr;
    uint32_t cmd_size = 0;
    uint32_t cmd_is_tans = 0;
    if (lane == 0) {
        uint32_t ct = (src[0] >> 4) & 0x7;
        if (ct == 0) {
            cmd_size = parseType0Header(src);
            cmd_ptr = src;
            src += cmd_size;
        } else if ((ct == 1 || ct == 6) && tans_tok_scratch_chunk != nullptr) {
            // tANS-encoded token stream: skip compressed data, use pre-decoded buffer
            uint32_t tans_comp_size, tans_dst_size;
            if (src[0] >= 0x80) {
                uint32_t bits = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
                tans_comp_size = bits & 0x3FF;
                tans_dst_size = tans_comp_size + ((bits >> 10) & 0x3FF) + 1;
                src += 3 + tans_comp_size;
            } else {
                uint32_t bits = ((uint32_t)src[1] << 24) | ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 8) | src[4];
                tans_comp_size = bits & 0x3FFFF;
                tans_dst_size = ((((bits >> 18) | ((uint32_t)src[0] << 14)) & 0x3FFFF)) + 1;
                src += 5 + tans_comp_size;
            }
            cmd_ptr = tans_tok_scratch_chunk;
            cmd_size = tans_dst_size;
            cmd_is_tans = 1;
        } else if (ct == 7 && tans_tok_scratch_chunk != nullptr) {
            // Paired-primary token stream: [0x70][countA:u24][inner type-6 stream]
            uint32_t count_a = ((uint32_t)src[1] << 16) | ((uint32_t)src[2] << 8) | src[3];
            const uint8_t* inner = src + 4;
            if (inner[0] >= 0x80) {
                uint32_t bits = ((uint32_t)inner[0] << 16) | ((uint32_t)inner[1] << 8) | inner[2];
                src = inner + 3 + (bits & 0x3FF);
            } else {
                uint32_t bits = ((uint32_t)inner[1] << 24) | ((uint32_t)inner[2] << 16) | ((uint32_t)inner[3] << 8) | inner[4];
                src = inner + 5 + (bits & 0x3FFFF);
            }
            cmd_ptr = tans_tok_scratch_chunk;
            cmd_size = count_a;
            cmd_is_tans = 1;
        } else if (ct == 5 && tans_tok_scratch_chunk != nullptr) {
            // Paired-secondary token stream: [0x50][countA:u24][countB:u24]
            uint32_t count_b = ((uint32_t)src[4] << 16) | ((uint32_t)src[5] << 8) | src[6];
            src += 7;
            cmd_ptr = tans_tok_scratch_chunk;
            cmd_size = count_b;
            cmd_is_tans = 1;
        } else {
            // Huffman or other — skip the stream, zero out cmd_size
            cmd_size = skipEntropyStream(src);
            cmd_ptr = src - cmd_size; // won't be used but keep safe
            cmd_size = 0; // can't decode this
        }
    }
    cmd_size = __shfl_sync(0xFFFFFFFF, cmd_size, 0);
    cmd_is_tans = __shfl_sync(0xFFFFFFFF, cmd_is_tans, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        if (cmd_is_tans) cmd_ptr = tans_tok_scratch_chunk;
        else cmd_ptr = src - cmd_size;
    }
    ps.cmd_ptr = cmd_ptr;
    ps.cmd_size = cmd_size;

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
    ps.cmd_stream2_offset = cmd_stream2_offset;

    // Off16 stream
    const uint8_t* off16_raw;
    const uint8_t* off16_hi_ptr = nullptr;
    const uint8_t* off16_lo_ptr = nullptr;
    uint32_t off16_count = 0;
    uint32_t off16_is_entropy = 0;
    uint32_t off16_is_split = 0;
    if (lane == 0) {
        uint16_t cnt; memcpy(&cnt, src, 2);
        if (cnt == 0xFFFF && tans_off16_scratch_chunk != nullptr) {
            // Entropy-coded off16: skip the two encoded sub-streams,
            // read pre-decoded hi/lo bytes from GPU tANS scratch
            src += 2;
            uint32_t hi_size = skipEntropyStream(src);
            uint32_t lo_size = skipEntropyStream(src);
            // hi_size and lo_size should be equal; use hi_size as count
            off16_count = hi_size;
            if (lo_size != hi_size) off16_count = (lo_size < hi_size) ? lo_size : hi_size;
            // GPU tANS layout: hi bytes at offset 0, lo bytes at offset 32768
            off16_hi_ptr = tans_off16_scratch_chunk;
            off16_lo_ptr = tans_off16_scratch_chunk + 65536;
            off16_raw = nullptr;
            off16_is_entropy = 1;
            off16_is_split = 1;
        } else {
            off16_count = cnt;
            off16_raw = src + 2;
            src += 2 + off16_count * 2;
        }
    }
    off16_count = __shfl_sync(0xFFFFFFFF, off16_count, 0);
    off16_is_entropy = __shfl_sync(0xFFFFFFFF, off16_is_entropy, 0);
    off16_is_split = __shfl_sync(0xFFFFFFFF, off16_is_split, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        if (off16_is_entropy) {
            off16_hi_ptr = tans_off16_scratch_chunk;
            off16_lo_ptr = tans_off16_scratch_chunk + 65536;
            off16_raw = nullptr;
        } else {
            off16_raw = src - off16_count * 2;
        }
    }
    ps.off16_raw = off16_raw;
    ps.off16_hi = off16_hi_ptr;
    ps.off16_lo = off16_lo_ptr;
    ps.off16_count = off16_count;
    ps.off16_split = off16_is_split;

    // Off32 stream sizes
    uint32_t off32_count1 = 0, off32_count2 = 0;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
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
        len_avail = (uint32_t)((uintptr_t)src_end - (uintptr_t)src);
    }
    off32_count1 = __shfl_sync(0xFFFFFFFF, off32_count1, 0);
    off32_count2 = __shfl_sync(0xFFFFFFFF, off32_count2, 0);
    len_avail = __shfl_sync(0xFFFFFFFF, len_avail, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        off32_raw2 = src - off32_count2 * 3;
        off32_raw1 = off32_raw2 - off32_count1 * 3;
    }
    ps.off32_raw1 = off32_raw1;
    ps.off32_raw2 = off32_raw2;
    ps.off32_count1 = off32_count1;
    ps.off32_count2 = off32_count2;
    ps.len_stream = src;
    ps.len_avail = len_avail;
}

// ── Sub-chunk decoder dispatch ────────────────────────────────
// Calls parseSubChunkHeaders (__noinline__) then dispatches to
// decodeSubChunkL1 or decodeSubChunk. The header parser's registers
// are freed before the decode hot loop runs.
__device__ void parseAndDecodeSubChunk(
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint32_t mode,
    uint8_t* tans_scratch_chunk,
    uint8_t* tans_tok_scratch_chunk,
    uint8_t* tans_off16_scratch_chunk
) {
    ParsedStreams ps;
    parseSubChunkHeaders(sc_src, sc_comp_size, sc_decomp_size, dst,
                         dst_offset, base_offset, tans_scratch_chunk,
                         tans_tok_scratch_chunk, tans_off16_scratch_chunk, ps);

    if (mode == 1 && ps.off32_count1 == 0 && ps.off32_count2 == 0) {
        decodeSubChunkL1(
            ps.cmd_ptr, ps.cmd_size,
            ps.lit_ptr, ps.lit_size,
            ps.off16_raw, ps.off16_count,
            ps.off16_hi, ps.off16_lo, ps.off16_split,
            ps.len_stream, ps.len_avail,
            dst, sc_decomp_size, ps.initial_copy,
            dst_offset
        );
    } else {
        decodeSubChunk(
            ps.cmd_ptr, ps.cmd_size,
            ps.lit_ptr, ps.lit_size,
            ps.off16_raw, ps.off16_count,
            ps.off16_hi, ps.off16_lo, ps.off16_split,
            ps.off32_raw1, ps.off32_count1,
            ps.off32_raw2, ps.off32_count2,
            ps.len_stream, ps.len_avail,
            dst, sc_decomp_size, ps.initial_copy, ps.cmd_stream2_offset,
            dst_offset, mode
        );
    }
}

// ── Production kernel ──────────────────────────────────────────
// Full GPU LZ kernel: 1 block per SC group, parses raw compressed
// chunks on-GPU. 2 warps per block for SM occupancy.
// No tANS code or shared-memory LUT — tANS is decoded by a separate
// kernel (Pass 1) before this kernel runs (Pass 2).
extern "C" __global__ void __launch_bounds__(64, 24) slzFullDecompressL1Kernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    uint32_t total_chunks,
    uint32_t sub_chunk_cap,
    uint8_t* __restrict__ tans_scratch,
    uint8_t* __restrict__ tans_tok_scratch,
    uint8_t* __restrict__ tans_off16_scratch,
    uint32_t chunk_base,
    // first_subchunk_idx[chunk_idx] = global sub-chunk index for sub-chunk 0
    // of this chunk. Each successive sub-chunk in the chunk uses the next
    // global index. nullptr → fall back to chunk_idx (legacy single-sub-chunk).
    const uint32_t* __restrict__ first_subchunk_idx
) {
    // 2 warps per block: warp 0 = threadIdx.y==0, warp 1 = threadIdx.y==1
    const uint32_t warp_id = threadIdx.y;
    const uint32_t group_id = blockIdx.x * 2 + warp_id;
    const int lane = threadIdx.x & 31;
    if (group_id >= (total_chunks + chunks_per_group - 1) / chunks_per_group) return;
    const uint32_t base_chunk = chunk_base + group_id * chunks_per_group;

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

        // LZ-compressed chunk: iterate sub-chunks. Each sub-chunk gets its
        // own slot in the tANS scratch buffers (indexed by global sub-chunk
        // index). When first_subchunk_idx is nullptr (legacy / no tANS),
        // fall back to chunk_idx for backward compatibility.
        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;
        // Per-sub-chunk slot size = 131072 (= sub_chunk_cap, large enough for
        // the biggest sub-chunk's lit/tok streams). off16-hi at offset 0,
        // off16-lo at offset 65536 within each slot.
        uint32_t global_sub_idx = first_subchunk_idx ? first_subchunk_idx[chunk_idx] : chunk_idx;
        uint8_t* chunk_tans_scratch = tans_scratch ? (tans_scratch + (uint64_t)global_sub_idx * 131072) : nullptr;
        uint8_t* chunk_tok_scratch = tans_tok_scratch ? (tans_tok_scratch + (uint64_t)global_sub_idx * 131072) : nullptr;
        uint8_t* chunk_off16_scratch = tans_off16_scratch ? (tans_off16_scratch + (uint64_t)global_sub_idx * 131072) : nullptr;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

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
                // base_offset is the relative offset within the chunk
                // (0 for the first sub-chunk, triggers initial 8-byte copy)
                uint32_t base_offset_val = sc_dst_off - ch.dst_offset;

                parseAndDecodeSubChunk(
                    sc_payload, sc_comp_size, sc_size,
                    dst, sc_dst_off, base_offset_val, sc_mode,
                    chunk_tans_scratch, chunk_tok_scratch, chunk_off16_scratch
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
            // Advance scratch pointers to next sub-chunk's slot (128KB stride)
            if (chunk_tans_scratch) chunk_tans_scratch += 131072;
            if (chunk_tok_scratch) chunk_tok_scratch += 131072;
            if (chunk_off16_scratch) chunk_off16_scratch += 131072;
        }
    }
}
