// ── StreamLZ GPU Decode Kernel ──────────────────────────────────
// 1 warp per chunk, 2 warps per block. Parses compressed sub-chunk
// headers on-GPU and decodes LZ tokens cooperatively.
//
// Three decode paths:
//   decodeSubChunkL1  — 64KB raw-mode chunks (no off32, no delta)
//   decodeSubChunk    — general path (off32, delta, multi-block)
//   parseAndDecodeSubChunk — parses Type 0 entropy headers, dispatches above
//
// Compiled to CUBIN via: nvcc -cubin -arch=sm_89 -O3 gpu_decode_kernel.cu
// Embedded in Zig via @embedFile("gpu_decode_kernel.cubin")

#include <cstdint>
#include <cstdio>

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
// Reads a variable-length count from the length stream (used for
// tokens 0, 1, 2 whose lengths exceed the inline capacity).
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

// ── L1 raw-mode sub-chunk decoder ──────────────────────────────
// Streamlined single-block raw decoder for 64KB chunks.
// No off32, no delta, no block split — fastest path for L1 data.
__device__ void decodeSubChunkL1(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t dst_offset
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = -8;
    const uint32_t dst_end_abs = dst_offset + dst_size;
    const uint8_t* len_stream = len_data;
    const uint8_t* len_end = len_data + len_avail;

    while (cmd_pos < cmd_size) {
        uint32_t local_lit = 0, local_match = 0;
        int32_t offset = recent_offset;
        uint32_t use_recent = 0, is_lit_only = 0;

        if (lane == 0) {
            uint32_t token = cmd[cmd_pos++];
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
                is_lit_only = 1;
                local_lit = readLength(len_stream, len_end) + 64;
            } else if (token == 1) {
                local_match = readLength(len_stream, len_end) + 91;
                if (off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    offset = -(int32_t)v; off16_pos++;
                }
            }
        }

        local_lit   = __shfl_sync(0xFFFFFFFF, local_lit, 0);
        local_match = __shfl_sync(0xFFFFFFFF, local_match, 0);
        offset      = __shfl_sync(0xFFFFFFFF, offset, 0);
        use_recent  = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        is_lit_only = __shfl_sync(0xFFFFFFFF, is_lit_only, 0);
        cmd_pos     = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        lit_pos     = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        off16_pos   = __shfl_sync(0xFFFFFFFF, off16_pos, 0);

        if (local_lit > 0) {
            for (uint32_t i = lane; i < local_lit; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                    dst[dst_pos + i] = lit[lit_pos + i];
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
        if (!use_recent && !is_lit_only) recent_offset = offset;
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
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    for (uint32_t i = lane; i < trailing; i += 32)
        if (dst_pos + i < dst_end_abs)
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

// ── Sub-chunk header parser + decoder dispatch ─────────────────
// GPU-side readLzTable + decode for one sub-chunk.
// Parses Type 0 entropy headers to locate streams, then dispatches
// to decodeSubChunkL1 (fast path) or decodeSubChunk (general path).
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

    if (mode == 1 && off32_count1 == 0 && off32_count2 == 0) {
        decodeSubChunkL1(
            cmd_ptr, cmd_size,
            lit_ptr, lit_size,
            off16_raw, off16_count,
            len_stream, len_avail,
            dst, sc_decomp_size, initial_copy,
            dst_offset
        );
    } else {
        decodeSubChunk(
            cmd_ptr, cmd_size,
            lit_ptr, lit_size,
            off16_raw, off16_count,
            off32_raw1, off32_count1,
            off32_raw2, off32_count2,
            len_stream, len_avail,
            dst, sc_decomp_size, initial_copy, cmd_stream2_offset,
            dst_offset, mode
        );
    }
}

// ── Production kernel ──────────────────────────────────────────
// Full GPU L1 kernel: 1 block per SC group, parses raw compressed
// chunks on-GPU. 2 warps per block for SM occupancy.
extern "C" __global__ void __launch_bounds__(64, 20) slzFullDecompressL1Kernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    uint32_t total_chunks,
    uint32_t sub_chunk_cap
) {
    // 2 warps per block: warp 0 = threadIdx.y==0, warp 1 = threadIdx.y==1
    const uint32_t warp_id = threadIdx.y;
    const uint32_t group_id = blockIdx.x * 2 + warp_id;
    const int lane = threadIdx.x & 31;
    if (group_id >= (total_chunks + chunks_per_group - 1) / chunks_per_group) return;
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
