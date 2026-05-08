// StreamLZ L1 GPU Test — Full token support
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

extern "C" {
    int slz_compress(const void* src, size_t src_len, void* dst, size_t dst_len, int level);
    int slz_decompress(const void* src, size_t src_len, void* dst, size_t dst_len);
    size_t slz_compress_bound(size_t src_len);

    struct GpuStreams {
        const uint8_t* cmd_data;    uint32_t cmd_size;
        const uint8_t* lit_data;    uint32_t lit_size;
        const uint8_t* off16_data;  uint32_t off16_count;
        const uint8_t* off32_data1; uint32_t off32_count1;
        const uint8_t* off32_data2; uint32_t off32_count2;
        const uint8_t* length_data; uint32_t length_avail;
        uint32_t initial_copy;
        uint32_t decomp_size;
        uint32_t cmd_stream2_offset;
        uint32_t base_offset;
        uint32_t dst_offset;
    };

    int slz_extract_gpu_streams(const uint8_t*, size_t, uint8_t*, size_t,
                                 uint8_t*, GpuStreams*, int);
}

// Extended length decode (matches Zig: length_stream byte, if > 251 read u16*4)
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

// Full L1 GPU decompressor kernel — all 5 token types
__global__ void slzDecompressL1Kernel(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off32_raw1, uint32_t off32_count1,
    const uint8_t* __restrict__ off32_raw2, uint32_t off32_count2,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t cmd_stream2_offset,
    uint32_t base_offset,  // group-relative offset (for match bounds)
    uint32_t dst_offset    // absolute offset in the full output buffer
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0, off32_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;  // absolute position in full output
    int32_t recent_offset = -8;
    uint32_t dst_end_abs = dst_offset + dst_size;
    const uint8_t* len_stream = len_data;
    const uint8_t* len_end = len_data + len_avail;
    const uint8_t* off32_cur = off32_raw1;
    uint32_t off32_count_cur = off32_count1;
    uint32_t block_dst_start = dst_offset;  // absolute start of current block in output

    uint32_t block_cmd_end = (cmd_stream2_offset > 0 && cmd_stream2_offset < cmd_size)
        ? cmd_stream2_offset : cmd_size;
    uint32_t block_dst_limit = (dst_size > 0x10000) ? 0x10000 : dst_size;
    int block_iter = 0;

    for (;;) {
    while (cmd_pos < block_cmd_end) {
        uint32_t token = 0;
        uint32_t local_lit = 0, local_match = 0;
        int32_t offset = recent_offset;
        uint32_t use_recent = 0;
        uint32_t token_type = 0; // 0=short, 1=long_lit, 2=long_near, 3=far_short, 4=far_long

        if (lane == 0) {
            token = cmd[cmd_pos++];

            if (token >= 24) {
                // Short token
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
                // Long literal (cmd 0): length from length_stream + 64
                token_type = 1;
                local_lit = readLength(len_stream, len_end) + 64;
                local_match = 0;
            } else if (token == 1) {
                // Long near match (cmd 1): length from length_stream + 91, off16
                token_type = 2;
                local_match = readLength(len_stream, len_end) + 91;
                local_lit = 0;
                if (off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    offset = -(int32_t)v;
                    off16_pos++;
                }
                use_recent = 0;
            } else if (token == 2) {
                // Long far match (cmd 2): length from length_stream + 29, off32
                token_type = 4;
                local_match = readLength(len_stream, len_end) + 29;
                local_lit = 0;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    // off32 'far' is distance from block start in FULL output
                    // match at: (base_offset + block_dst_start - far)
                    // relative to current write pos: above - (base_offset + dst_pos)
                    offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            } else {
                // Medium far match (cmd 3-23): length = cmd + 5, off32
                token_type = 3;
                local_match = token + 5;
                local_lit = 0;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            }
        }

        // Broadcast all state from lane 0
        token_type  = __shfl_sync(0xFFFFFFFF, token_type, 0);
        local_lit   = __shfl_sync(0xFFFFFFFF, local_lit, 0);
        local_match = __shfl_sync(0xFFFFFFFF, local_match, 0);
        offset      = __shfl_sync(0xFFFFFFFF, offset, 0);
        use_recent  = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        cmd_pos     = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        lit_pos     = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        off16_pos   = __shfl_sync(0xFFFFFFFF, off16_pos, 0);
        off32_pos   = __shfl_sync(0xFFFFFFFF, off32_pos, 0);

        // ALL SERIAL ON LANE 0 (debug: bypass warp cooperation)
        if (lane == 0) {
            for (uint32_t i = 0; i < local_lit; i++)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                    dst[dst_pos + i] = lit[lit_pos + i];
            dst_pos += local_lit;
            lit_pos += local_lit;

            if (local_match > 0) {
                uint32_t ms = (uint32_t)((int32_t)dst_pos + offset);
                for (uint32_t i = 0; i < local_match; i++)
                    if (dst_pos + i < dst_end_abs)
                        dst[dst_pos + i] = dst[ms + i];
                dst_pos += local_match;
            }
        }
        __syncwarp();

        dst_pos   = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
        lit_pos   = __shfl_sync(0xFFFFFFFF, lit_pos, 0);

        if (!use_recent && (token_type != 1)) recent_offset = offset;
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);

        // Debug: print token state at the block boundary transition
        if (lane == 0 && dst_pos >= 65520 && dst_pos <= 65550) {
            printf("  [BDY] cmd=%u tok=0x%02x lit=%u match=%u off=%d dst=%u lit_pos=%u o16=%u blk_end=%u\n",
                   cmd_pos-1, token, local_lit, local_match, offset, dst_pos, lit_pos, off16_pos, block_cmd_end);
        }
        if (lane == 0 && token < 24 && token_type != 0) {
            printf("  [GPU] NON-SHORT: cmd_pos=%u token=%u type=%u lit=%u match=%u dst_pos=%u\n",
                   cmd_pos-1, token, token_type, local_lit, local_match, dst_pos);
        }
        if (lane == 0 && token_type == 1) {
            printf("  [GPU] LONG LIT: cmd_pos=%u lit=%u dst_pos=%u lit_pos=%u len_byte=%u\n",
                   cmd_pos-1, local_lit, dst_pos - local_lit, lit_pos - local_lit,
                   (len_stream > len_data) ? len_stream[-1] : 255);
        }
        if (lane == 0 && (cmd_size <= 20 || cmd_pos >= block_cmd_end - 3)) {
            printf("  [GPU] cmd_pos=%u token=0x%02x type=%u lit=%u match=%u dst_pos=%u lit_pos=%u off16=%u\n",
                   cmd_pos-1, token, token_type, local_lit, local_match, dst_pos, lit_pos, off16_pos);
        }
        // Print final state after last command
        if (lane == 0 && cmd_pos >= block_cmd_end) {
            printf("  [GPU] FINAL: dst_pos=%u lit_pos=%u off16_pos=%u/%u recent=%d\n",
                   dst_pos, lit_pos, off16_pos, off16_count, recent_offset);
        }

        // Broadcast len_stream pointer
        {
            uint32_t len_off = (uint32_t)((uintptr_t)len_stream - (uintptr_t)len_data);
            len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
            len_stream = len_data + len_off;
        }
    } // end while (cmd_pos < block_cmd_end)

    // Advance to block 2
    block_iter++;
    if (block_iter >= 2) break;
    block_cmd_end = cmd_size;
    block_dst_limit = dst_size;
    // Swap to block 2's off32 backing
    off32_cur = off32_raw2;
    off32_count_cur = off32_count2;
    off32_pos = 0;
    block_dst_start = dst_pos;  // block 2 starts here
    } // end for (;;)

    // Trailing literals: after all commands, remaining literals are copied verbatim
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    for (uint32_t i = lane; i < trailing; i += 32) {
        if (dst_pos + i < dst_end_abs)
            dst[dst_pos + i] = lit[lit_pos + i];
    }
}

static uint8_t* readFile(const char* path, size_t* sz) {
    FILE* f = fopen(path, "rb");
    if (!f) { *sz = 0; return nullptr; }
    fseek(f, 0, SEEK_END); *sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc(*sz);
    fread(buf, 1, *sz, f); fclose(f);
    return buf;
}

int main(int argc, char** argv) {
    printf("StreamLZ L1 GPU Test — Full Token Support\n");

    size_t src_size;
    const char* input_file = argc > 1 ? argv[1] : "c:\\tmp\\test_50k.txt";
    uint8_t* src = readFile(input_file, &src_size);
    if (!src) { printf("Cannot read source\n"); return 1; }

    size_t comp_cap = slz_compress_bound(src_size);
    uint8_t* comp = (uint8_t*)malloc(comp_cap);
    int comp_size = slz_compress(src, src_size, comp, comp_cap, 1);
    if (comp_size < 0) { printf("Compress failed\n"); return 1; }
    printf("Compressed: %zu -> %d bytes\n", src_size, comp_size);

    uint8_t* cpu_dst = (uint8_t*)calloc(src_size + 64, 1);
    int dec_size = slz_decompress(comp, comp_size, cpu_dst, src_size + 64);
    if (dec_size < 0 || (size_t)dec_size != src_size) { printf("CPU decomp failed\n"); return 1; }
    printf("CPU roundtrip OK\n\n");

    // Extract streams
    size_t scratch_size = src_size * 3 + 1024 * 1024;
    uint8_t* scratch = (uint8_t*)malloc(scratch_size);
    uint8_t* gpu_dst = (uint8_t*)calloc(src_size + 64, 1);
    GpuStreams streams[64];

    int num = slz_extract_gpu_streams((const uint8_t*)comp, comp_size,
                                       scratch, scratch_size, gpu_dst, streams, 64);
    if (num <= 0) { printf("Extract failed: %d\n", num); return 1; }
    printf("Extracted %d sub-chunks\n", num);

    // Allocate single device output buffer for ALL sub-chunks
    uint8_t* d_dst_full;
    cudaMalloc(&d_dst_full, src_size + 256);
    cudaMemset(d_dst_full, 0, src_size + 256);
    // Copy initial bytes for first sub-chunk only (tail prefix applied after kernels)
    if (num > 0 && streams[0].initial_copy > 0)
        cudaMemcpy(d_dst_full, gpu_dst, streams[0].initial_copy, cudaMemcpyHostToDevice);

    uint32_t total_pass = 0, total_fail = 0;
    uint32_t dst_offset = 0;

    for (int i = 0; i < num; i++) {
        const GpuStreams& s = streams[i];
        printf("  SC %d: cmd=%u(split@%u) lit=%u off16=%u off32=%u+%u len=%u init=%u decomp=%u\n",
            i, s.cmd_size, s.cmd_stream2_offset, s.lit_size, s.off16_count,
            s.off32_count1, s.off32_count2,
            s.length_avail, s.initial_copy, s.decomp_size);

        uint8_t *d_cmd, *d_lit, *d_off16, *d_off32_1, *d_off32_2, *d_len;
        cudaMalloc(&d_cmd, s.cmd_size);
        cudaMalloc(&d_lit, s.lit_size);
        cudaMalloc(&d_off16, s.off16_count * 2 + 1);
        cudaMalloc(&d_off32_1, s.off32_count1 * 4 + 4);
        cudaMalloc(&d_off32_2, s.off32_count2 * 4 + 4);
        cudaMalloc(&d_len, s.length_avail + 1);

        cudaMemcpy(d_cmd, s.cmd_data, s.cmd_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lit, s.lit_data, s.lit_size, cudaMemcpyHostToDevice);
        if (s.off16_count > 0)
            cudaMemcpy(d_off16, s.off16_data, s.off16_count * 2, cudaMemcpyHostToDevice);
        if (s.off32_count1 > 0)
            cudaMemcpy(d_off32_1, s.off32_data1, s.off32_count1 * 4, cudaMemcpyHostToDevice);
        if (s.off32_count2 > 0)
            cudaMemcpy(d_off32_2, s.off32_data2, s.off32_count2 * 4, cudaMemcpyHostToDevice);
        if (s.length_avail > 0)
            cudaMemcpy(d_len, s.length_data, s.length_avail, cudaMemcpyHostToDevice);

        // Use shared output buffer — kernel gets the FULL buffer base
        uint8_t* d_dst = d_dst_full;

        slzDecompressL1Kernel<<<1, 32>>>(
            d_cmd, s.cmd_size, d_lit, s.lit_size,
            d_off16, s.off16_count,
            d_off32_1, s.off32_count1,
            d_off32_2, s.off32_count2,
            d_len, s.length_avail,
            d_dst, s.decomp_size, s.initial_copy,
            s.cmd_stream2_offset, s.base_offset, s.dst_offset);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("    CUDA error: %s\n", cudaGetErrorString(err));
            total_fail++;
        } else {
            cudaMemcpy(gpu_dst + dst_offset, d_dst_full + dst_offset, s.decomp_size, cudaMemcpyDeviceToHost);

            uint32_t mis = 0;
            for (uint32_t j = 0; j < s.decomp_size; j++) {
                if (gpu_dst[dst_offset + j] != cpu_dst[dst_offset + j]) {
                    if (mis < 3)
                        printf("    MISMATCH at %u: gpu=0x%02x cpu=0x%02x\n",
                            j, gpu_dst[dst_offset + j], cpu_dst[dst_offset + j]);
                    mis++;
                }
            }
            // After GPU decode, check how far we got
        // Find last non-0xCC byte
        uint32_t last_nonzero = 0;
        for (uint32_t j = 0; j < s.decomp_size; j++) {
            if (gpu_dst[dst_offset + j] != 0) last_nonzero = j + 1;
        }
        printf("    %s (%u bytes, %u mismatches, last_nonzero=%u)\n",
                   mis == 0 ? "PASS" : "FAIL", s.decomp_size, mis, last_nonzero);
            if (mis == 0) total_pass++; else total_fail++;
        }

        cudaFree(d_cmd); cudaFree(d_lit); cudaFree(d_off16);
        cudaFree(d_off32_1); cudaFree(d_off32_2); cudaFree(d_len);
        dst_offset += s.decomp_size;
    }

    // Apply tail prefix restoration on device (first 8 bytes of each chunk > 0)
    // The host buffer gpu_dst has the restored bytes from slz_extract_gpu_streams
    {
        uint32_t chunk_size = 262144;
        uint32_t n_chunks = (uint32_t)((src_size + chunk_size - 1) / chunk_size);
        for (uint32_t ci = 1; ci < n_chunks; ci++) {
            uint32_t off = ci * chunk_size;
            if (off + 8 <= src_size)
                cudaMemcpy(d_dst_full + off, gpu_dst + off, 8, cudaMemcpyHostToDevice);
        }
    }

    // Re-verify after tail prefix restoration
    cudaMemcpy(gpu_dst, d_dst_full, src_size, cudaMemcpyDeviceToHost);
    total_pass = 0; total_fail = 0;
    for (int i = 0; i < num; i++) {
        const GpuStreams& s = streams[i];
        uint32_t mis = 0;
        for (uint32_t j = 0; j < s.decomp_size; j++)
            if (gpu_dst[s.dst_offset + j] != cpu_dst[s.dst_offset + j]) mis++;
        if (mis == 0) total_pass++; else {
            if (total_fail < 3) printf("  SC %d: FAIL after prefix (%u mismatches)\n", i, mis);
            total_fail++;
        }
    }

    printf("\n%u PASS, %u FAIL (after tail prefix)\n", total_pass, total_fail);
    printf("%s\n", total_fail == 0 ? "ALL SUB-CHUNKS PASS" : "SOME FAILED");

    // Quick throughput estimate: total decomp bytes / wall time
    if (total_fail == 0) {
        uint32_t total_decomp = 0;
        for (int i = 0; i < num; i++) total_decomp += streams[i].decomp_size;
        printf("\nTotal decompressed: %u bytes (%.1f MB)\n",
               total_decomp, total_decomp / 1e6);
        printf("Note: this is sequential per-sub-chunk with individual H2D/D2H.\n");
        printf("Real GPU perf requires batched launches across all sub-chunks.\n");
    }

    cudaFree(d_dst_full);
    free(src); free(comp); free(cpu_dst); free(scratch); free(gpu_dst);
    return total_fail == 0 ? 0 : 1;
}
