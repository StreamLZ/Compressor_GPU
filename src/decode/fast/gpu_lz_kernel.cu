// StreamLZ L1 GPU Decompressor Kernel
// Replaces processLzRuns: serial token parse on lane 0, warp-cooperative copies.
// CPU does entropy decode + stream extraction (readLzTable), GPU does LZ copy.

#include <cstdint>
#include <cstdio>

extern "C" {

// Batch descriptor for one sub-chunk (matches Zig bridge struct)
struct GpuSubChunkDesc {
    const uint8_t* cmd_data;    uint32_t cmd_size;
    const uint8_t* lit_data;    uint32_t lit_size;
    const uint8_t* off16_data;  uint32_t off16_count;
    uint8_t* dst;               uint32_t dst_size;
    uint32_t initial_copy;      // 8 if base_offset==0, else 0
};

// ── Kernel ──
// One warp per sub-chunk. Lane 0 parses tokens, all lanes copy.

__global__ void slzDecompressL1Kernel(
    const GpuSubChunkDesc* __restrict__ descs,
    int num_descs
) {
    // Each warp handles one descriptor
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_descs) return;

    const GpuSubChunkDesc& desc = descs[warp_id];

    const uint8_t* cmd = desc.cmd_data;
    const uint8_t* lit = desc.lit_data;
    const uint8_t* off16_raw = desc.off16_data;
    uint8_t* dst = desc.dst;

    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0;
    uint32_t dst_pos = 0;
    int32_t recent_offset = -8;

    // Initial 8-byte raw copy (first sub-chunk of first chunk)
    if (desc.initial_copy > 0) {
        // The initial bytes are at the start of the lit stream
        // (readLzTable already advanced dst past them and put them in dst)
        // So we need to account for the 8 bytes already in dst
        dst_pos = desc.initial_copy;
    }

    const uint32_t cmd_end = desc.cmd_size;
    const uint32_t dst_end = desc.dst_size;

    while (cmd_pos < cmd_end) {
        uint32_t token = 0;
        uint32_t local_lit = 0, local_match = 0;
        int32_t offset = recent_offset;
        uint32_t use_recent = 0;
        uint32_t token_valid = 1;

        // ── Token parse (lane 0) ──
        if (lane == 0) {
            token = cmd[cmd_pos++];

            if (token >= 24) {
                // Short token (hot path ~90%)
                local_lit = token & 7;
                local_match = (token >> 3) & 0xF;
                use_recent = (token >> 7) & 1;

                if (!use_recent && off16_pos < desc.off16_count) {
                    uint16_t off_val;
                    memcpy(&off_val, off16_raw + off16_pos * 2, 2);
                    offset = -(int32_t)off_val;
                    off16_pos++;
                }
            } else {
                // Non-short token — stop for now
                // TODO: handle cmd 0 (long literal), 1 (long near match),
                //       2 (long far match), 3-23 (medium matches)
                token_valid = 0;
            }
        }

        // Broadcast from lane 0
        token       = __shfl_sync(0xFFFFFFFF, token, 0);
        token_valid = __shfl_sync(0xFFFFFFFF, token_valid, 0);
        if (!token_valid) break;

        local_lit   = __shfl_sync(0xFFFFFFFF, local_lit, 0);
        local_match = __shfl_sync(0xFFFFFFFF, local_match, 0);
        offset      = __shfl_sync(0xFFFFFFFF, offset, 0);
        use_recent  = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        cmd_pos     = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        lit_pos     = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        off16_pos   = __shfl_sync(0xFFFFFFFF, off16_pos, 0);

        // ── Literal copy (warp-cooperative) ──
        for (uint32_t i = lane; i < local_lit; i += 32) {
            if (dst_pos + i < dst_end && lit_pos + i < desc.lit_size)
                dst[dst_pos + i] = lit[lit_pos + i];
        }
        __syncwarp();
        dst_pos += local_lit;
        lit_pos += local_lit;

        // ── Match copy ──
        if (local_match > 0) {
            uint32_t match_src = (uint32_t)((int32_t)dst_pos + offset);
            int32_t abs_off = -offset;

            if (abs_off >= (int32_t)local_match && local_match > 1) {
                // Non-overlapping: all lanes help
                for (uint32_t i = lane; i < local_match; i += 32)
                    if (dst_pos + i < dst_end)
                        dst[dst_pos + i] = dst[match_src + i];
            } else {
                // Overlapping or very short: lane 0 serial
                if (lane == 0)
                    for (uint32_t i = 0; i < local_match; i++)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[match_src + i];
            }
            __syncwarp();
            dst_pos += local_match;
        }

        dst_pos   = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
        lit_pos   = __shfl_sync(0xFFFFFFFF, lit_pos, 0);

        if (!use_recent) recent_offset = offset;
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
    }
}

// ── Host API ──

int slz_gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0) ? 1 : 0;
}

int slz_gpu_decompress_batch(
    const GpuSubChunkDesc* host_descs,
    int num_descs,
    // Pre-allocated device memory for streams (caller manages)
    // For simplicity, we'll copy everything internally for now
    void* /* reserved */
) {
    if (num_descs <= 0) return 0;

    // Allocate device descriptors
    GpuSubChunkDesc* d_descs;
    cudaMalloc(&d_descs, num_descs * sizeof(GpuSubChunkDesc));

    // For each descriptor, copy streams to device and update pointers
    GpuSubChunkDesc* h_descs_dev = new GpuSubChunkDesc[num_descs];

    for (int i = 0; i < num_descs; i++) {
        const GpuSubChunkDesc& h = host_descs[i];
        GpuSubChunkDesc& d = h_descs_dev[i];
        d = h; // copy metadata

        // Allocate and copy cmd stream
        uint8_t* d_cmd; cudaMalloc(&d_cmd, h.cmd_size);
        cudaMemcpy(d_cmd, h.cmd_data, h.cmd_size, cudaMemcpyHostToDevice);
        d.cmd_data = d_cmd;

        // Allocate and copy lit stream
        uint8_t* d_lit; cudaMalloc(&d_lit, h.lit_size);
        cudaMemcpy(d_lit, h.lit_data, h.lit_size, cudaMemcpyHostToDevice);
        d.lit_data = d_lit;

        // Allocate and copy off16 stream
        uint8_t* d_off16; cudaMalloc(&d_off16, h.off16_count * 2);
        cudaMemcpy(d_off16, h.off16_data, h.off16_count * 2, cudaMemcpyHostToDevice);
        d.off16_data = d_off16;

        // Allocate output buffer
        uint8_t* d_dst; cudaMalloc(&d_dst, h.dst_size);
        // Copy initial bytes if present
        if (h.initial_copy > 0) {
            cudaMemcpy(d_dst, h.dst, h.initial_copy, cudaMemcpyHostToDevice);
        }
        d.dst = d_dst;
    }

    // Copy descriptors to device
    cudaMemcpy(d_descs, h_descs_dev, num_descs * sizeof(GpuSubChunkDesc), cudaMemcpyHostToDevice);

    // Launch: 1 warp per descriptor, 2 warps per block (64 threads)
    int warps_per_block = 2;
    int threads_per_block = warps_per_block * 32;
    int num_blocks = (num_descs + warps_per_block - 1) / warps_per_block;

    slzDecompressL1Kernel<<<num_blocks, threads_per_block>>>(d_descs, num_descs);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        // Cleanup
        for (int i = 0; i < num_descs; i++) {
            cudaFree((void*)h_descs_dev[i].cmd_data);
            cudaFree((void*)h_descs_dev[i].lit_data);
            cudaFree((void*)h_descs_dev[i].off16_data);
            cudaFree(h_descs_dev[i].dst);
        }
        cudaFree(d_descs);
        delete[] h_descs_dev;
        return -1;
    }

    // Copy results back to host
    for (int i = 0; i < num_descs; i++) {
        cudaMemcpy((void*)host_descs[i].dst, h_descs_dev[i].dst,
                   host_descs[i].dst_size, cudaMemcpyDeviceToHost);
    }

    // Cleanup device memory
    for (int i = 0; i < num_descs; i++) {
        cudaFree((void*)h_descs_dev[i].cmd_data);
        cudaFree((void*)h_descs_dev[i].lit_data);
        cudaFree((void*)h_descs_dev[i].off16_data);
        cudaFree(h_descs_dev[i].dst);
    }
    cudaFree(d_descs);
    delete[] h_descs_dev;

    return 0;
}

} // extern "C"
