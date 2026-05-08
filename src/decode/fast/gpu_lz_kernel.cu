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
    uint32_t dst_offset
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
        for (uint32_t i = lane; i < block_trailing; i += 32)
            if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                dst[dst_pos + i] = lit[lit_pos + i];
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
    for (uint32_t i = lane; i < trailing; i += 32)
        if (dst_pos + i < dst_end_abs)
            dst[dst_pos + i] = lit[lit_pos + i];
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
        dst, dst_size, initial_copy, cmd_stream2_offset, dst_offset);
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
            d.dst_offset
        );
        __syncwarp();
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
