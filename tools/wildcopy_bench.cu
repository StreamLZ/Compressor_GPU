// Realistic wildcopy benchmark for StreamLZ GPU decode.
// Simulates the actual kernel workload: each warp processes a 64KB chunk
// by doing thousands of small literal+match copies interleaved with
// serial token parsing on lane 0.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define NUM_CHUNKS   1526
#define CHUNK_SIZE   65536
#define WARMUP       50
#define ITERS        500

// Pre-generated token sequence per chunk: lit_len, match_len pairs.
// Stored in global memory, read by lane 0 (simulates cmd stream parsing).
struct Token {
    uint16_t lit_len;
    uint16_t match_len;
    int16_t  offset;
};

// ── Byte-exact copies (current kernel) ─────────────────────────
__global__ void __launch_bounds__(64) decodeByte(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const Token* __restrict__ tokens,
    const uint32_t* __restrict__ token_counts,
    uint32_t num_chunks
) {
    const uint32_t chunk_id = blockIdx.x * 2 + threadIdx.y;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;
    const uint32_t base = chunk_id * CHUNK_SIZE;
    const Token* my_tokens = tokens + chunk_id * 8192;
    const uint32_t n_tokens = token_counts[chunk_id];

    uint32_t dst_pos = base + 8;
    uint32_t lit_pos = base + 8;
    const uint32_t dst_end = base + CHUNK_SIZE;

    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t ll = 0, ml = 0;
        int32_t off = 0;
        if (lane == 0) {
            ll = my_tokens[t].lit_len;
            ml = my_tokens[t].match_len;
            off = my_tokens[t].offset;
        }
        ll  = __shfl_sync(0xFFFFFFFF, ll, 0);
        ml  = __shfl_sync(0xFFFFFFFF, ml, 0);
        off = __shfl_sync(0xFFFFFFFF, off, 0);

        // Literal copy
        for (uint32_t i = lane; i < ll; i += 32)
            if (dst_pos + i < dst_end && lit_pos + i < dst_end)
                dst[dst_pos + i] = src[lit_pos + i];
        __syncwarp();
        dst_pos += ll;
        lit_pos += ll;

        // Match copy
        if (ml > 0) {
            uint32_t ms = (uint32_t)((int32_t)dst_pos + off);
            int32_t abs_off = -off;
            if (abs_off >= (int32_t)ml) {
                for (uint32_t i = lane; i < ml; i += 32)
                    if (dst_pos + i < dst_end)
                        dst[dst_pos + i] = dst[ms + i];
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < ml; i++)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
            }
            __syncwarp();
            dst_pos += ml;
        }
    }
}

// ── 2-byte copies ──────────────────────────────────────────────
__global__ void __launch_bounds__(64) decode2Byte(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const Token* __restrict__ tokens,
    const uint32_t* __restrict__ token_counts,
    uint32_t num_chunks
) {
    const uint32_t chunk_id = blockIdx.x * 2 + threadIdx.y;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;
    const uint32_t base = chunk_id * CHUNK_SIZE;
    const Token* my_tokens = tokens + chunk_id * 8192;
    const uint32_t n_tokens = token_counts[chunk_id];

    uint32_t dst_pos = base + 8;
    uint32_t lit_pos = base + 8;
    const uint32_t dst_end = base + CHUNK_SIZE;

    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t ll = 0, ml = 0;
        int32_t off = 0;
        if (lane == 0) {
            ll = my_tokens[t].lit_len;
            ml = my_tokens[t].match_len;
            off = my_tokens[t].offset;
        }
        ll  = __shfl_sync(0xFFFFFFFF, ll, 0);
        ml  = __shfl_sync(0xFFFFFFFF, ml, 0);
        off = __shfl_sync(0xFFFFFFFF, off, 0);

        // Literal copy — 2 bytes per lane when aligned, else byte
        if ((lit_pos & 1) == 0 && (dst_pos & 1) == 0 && ll >= 4) {
            uint32_t len2 = ll / 2;
            const uint16_t* s2 = (const uint16_t*)(src + lit_pos);
            uint16_t* d2 = (uint16_t*)(dst + dst_pos);
            for (uint32_t i = lane; i < len2; i += 32)
                if (dst_pos + i * 2 + 1 < dst_end)
                    d2[i] = s2[i];
            uint32_t done = len2 * 2;
            for (uint32_t i = done + lane; i < ll; i += 32)
                if (dst_pos + i < dst_end)
                    dst[dst_pos + i] = src[lit_pos + i];
        } else {
            for (uint32_t i = lane; i < ll; i += 32)
                if (dst_pos + i < dst_end)
                    dst[dst_pos + i] = src[lit_pos + i];
        }
        __syncwarp();
        dst_pos += ll;
        lit_pos += ll;

        // Match copy — 2 bytes per lane when aligned (non-overlapping only)
        if (ml > 0) {
            uint32_t ms = (uint32_t)((int32_t)dst_pos + off);
            int32_t abs_off = -off;
            if (abs_off >= (int32_t)ml) {
                if ((ms & 1) == 0 && (dst_pos & 1) == 0 && ml >= 4) {
                    uint32_t len2 = ml / 2;
                    const uint16_t* s2 = (const uint16_t*)(dst + ms);
                    uint16_t* d2 = (uint16_t*)(dst + dst_pos);
                    for (uint32_t i = lane; i < len2; i += 32)
                        if (dst_pos + i * 2 + 1 < dst_end)
                            d2[i] = s2[i];
                    uint32_t done = len2 * 2;
                    for (uint32_t i = done + lane; i < ml; i += 32)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
                } else {
                    for (uint32_t i = lane; i < ml; i += 32)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
                }
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < ml; i++)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
            }
            __syncwarp();
            dst_pos += ml;
        }
    }
}

// ── 4-byte copies ──────────────────────────────────────────────
__global__ void __launch_bounds__(64) decode4Byte(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const Token* __restrict__ tokens,
    const uint32_t* __restrict__ token_counts,
    uint32_t num_chunks
) {
    const uint32_t chunk_id = blockIdx.x * 2 + threadIdx.y;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;
    const uint32_t base = chunk_id * CHUNK_SIZE;
    const Token* my_tokens = tokens + chunk_id * 8192;
    const uint32_t n_tokens = token_counts[chunk_id];

    uint32_t dst_pos = base + 8;
    uint32_t lit_pos = base + 8;
    const uint32_t dst_end = base + CHUNK_SIZE;

    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t ll = 0, ml = 0;
        int32_t off = 0;
        if (lane == 0) {
            ll = my_tokens[t].lit_len;
            ml = my_tokens[t].match_len;
            off = my_tokens[t].offset;
        }
        ll  = __shfl_sync(0xFFFFFFFF, ll, 0);
        ml  = __shfl_sync(0xFFFFFFFF, ml, 0);
        off = __shfl_sync(0xFFFFFFFF, off, 0);

        // Literal copy — 4 bytes per lane when aligned, else byte
        if ((lit_pos & 3) == 0 && (dst_pos & 3) == 0 && ll >= 8) {
            uint32_t len4 = ll / 4;
            const uint32_t* s4 = (const uint32_t*)(src + lit_pos);
            uint32_t* d4 = (uint32_t*)(dst + dst_pos);
            for (uint32_t i = lane; i < len4; i += 32)
                if (dst_pos + i * 4 + 3 < dst_end)
                    d4[i] = s4[i];
            uint32_t done = len4 * 4;
            for (uint32_t i = done + lane; i < ll; i += 32)
                if (dst_pos + i < dst_end)
                    dst[dst_pos + i] = src[lit_pos + i];
        } else {
            for (uint32_t i = lane; i < ll; i += 32)
                if (dst_pos + i < dst_end)
                    dst[dst_pos + i] = src[lit_pos + i];
        }
        __syncwarp();
        dst_pos += ll;
        lit_pos += ll;

        // Match copy — 4 bytes per lane when aligned (non-overlapping only)
        if (ml > 0) {
            uint32_t ms = (uint32_t)((int32_t)dst_pos + off);
            int32_t abs_off = -off;
            if (abs_off >= (int32_t)ml) {
                if ((ms & 3) == 0 && (dst_pos & 3) == 0 && ml >= 8) {
                    uint32_t len4 = ml / 4;
                    const uint32_t* s4 = (const uint32_t*)(dst + ms);
                    uint32_t* d4 = (uint32_t*)(dst + dst_pos);
                    for (uint32_t i = lane; i < len4; i += 32)
                        if (dst_pos + i * 4 + 3 < dst_end)
                            d4[i] = s4[i];
                    uint32_t done = len4 * 4;
                    for (uint32_t i = done + lane; i < ml; i += 32)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
                } else {
                    for (uint32_t i = lane; i < ml; i += 32)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
                }
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < ml; i++)
                        if (dst_pos + i < dst_end)
                            dst[dst_pos + i] = dst[ms + i];
            }
            __syncwarp();
            dst_pos += ml;
        }
    }
}

int main() {
    printf("Realistic StreamLZ GPU Decode — Wildcopy Benchmark\n");
    printf("==================================================\n");
    printf("%d chunks x 64KB, 2 warps/block, %d iterations\n\n", NUM_CHUNKS, ITERS);

    // Generate realistic token sequences.
    // L1 enwik8 stats: ~10 tokens per 64 bytes → ~10000 tokens per 64KB chunk.
    // Average: lit=4, match=10, offset=-50 (recent or near).
    srand(42);
    uint32_t max_tokens_per_chunk = 8192;
    size_t token_buf_size = (size_t)NUM_CHUNKS * max_tokens_per_chunk * sizeof(Token);
    Token* h_tokens = (Token*)malloc(token_buf_size);
    uint32_t* h_counts = (uint32_t*)malloc(NUM_CHUNKS * sizeof(uint32_t));

    for (uint32_t c = 0; c < NUM_CHUNKS; c++) {
        Token* ct = h_tokens + c * max_tokens_per_chunk;
        uint32_t pos = 8; // initial copy
        uint32_t t = 0;
        while (pos < CHUNK_SIZE && t < max_tokens_per_chunk) {
            // Realistic distribution: mostly short tokens
            uint32_t lit = rand() % 8;          // 0-7 literals
            uint32_t match = 8 + (rand() % 8);  // 8-15 match length

            if (pos + lit + match > CHUNK_SIZE) {
                lit = (CHUNK_SIZE > pos) ? CHUNK_SIZE - pos : 0;
                match = 0;
            }
            int32_t off = (pos + lit >= 64) ? -(8 + (rand() % 56)) : -8;
            ct[t].lit_len = (uint16_t)lit;
            ct[t].match_len = (uint16_t)match;
            ct[t].offset = (int16_t)off;
            pos += lit + match;
            t++;
        }
        h_counts[c] = t;
    }

    printf("Avg tokens/chunk: %u\n", h_counts[0]);

    // Allocate device memory
    size_t total = (size_t)NUM_CHUNKS * CHUNK_SIZE;
    uint8_t* h_src = (uint8_t*)malloc(total);
    for (size_t i = 0; i < total; i++) h_src[i] = (uint8_t)(i * 7 + 13);

    uint8_t *d_src, *d_dst;
    Token* d_tokens;
    uint32_t* d_counts;
    cudaMalloc(&d_src, total);
    cudaMalloc(&d_dst, total);
    cudaMalloc(&d_tokens, token_buf_size);
    cudaMalloc(&d_counts, NUM_CHUNKS * sizeof(uint32_t));
    cudaMemcpy(d_src, h_src, total, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tokens, h_tokens, token_buf_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, NUM_CHUNKS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    dim3 block(32, 2, 1);
    dim3 grid((NUM_CHUNKS + 1) / 2, 1, 1);

    // Sanity check each kernel
    auto checkKernel = [&](const char* name, auto launch_fn) -> bool {
        cudaDeviceSynchronize();
        cudaGetLastError(); // clear
        launch_fn<<<grid, block>>>(d_src, d_dst, d_tokens, d_counts, NUM_CHUNKS);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error (%s): %s\n", name, cudaGetErrorString(err));
            return false;
        }
        return true;
    };
    bool byte_ok = checkKernel("byte", decodeByte);
    bool b2_ok = checkKernel("2-byte", decode2Byte);
    bool b4_ok = checkKernel("4-byte", decode4Byte);
    if (!byte_ok) { printf("Byte kernel failed — aborting\n"); return 1; }
    printf("Kernels OK: byte=%d 2byte=%d 4byte=%d\n", byte_ok, b2_ok, b4_ok);

    // Warmup
    for (int i = 0; i < WARMUP; i++)
        decodeByte<<<grid, block>>>(d_src, d_dst, d_tokens, d_counts, NUM_CHUNKS);
    cudaDeviceSynchronize();

    // Benchmark each
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        decodeByte<<<grid, block>>>(d_src, d_dst, d_tokens, d_counts, NUM_CHUNKS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float t_byte = ms / ITERS;

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        decode2Byte<<<grid, block>>>(d_src, d_dst, d_tokens, d_counts, NUM_CHUNKS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float t_2byte = ms / ITERS;

    cudaEventRecord(start);
    for (int i = 0; i < ITERS; i++)
        decode4Byte<<<grid, block>>>(d_src, d_dst, d_tokens, d_counts, NUM_CHUNKS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float t_4byte = ms / ITERS;

    double total_mb = (double)NUM_CHUNKS * CHUNK_SIZE / 1e6;

    printf("\nResults (%.1f MB decoded per iteration):\n", total_mb);
    printf("  1-byte:  %.3f ms  → %.1f GB/s\n", t_byte, total_mb / t_byte / 1e3);
    printf("  2-byte:  %.3f ms  → %.1f GB/s  (%.1f%% vs 1-byte)\n",
           t_2byte, total_mb / t_2byte / 1e3, (t_byte / t_2byte - 1) * 100);
    printf("  4-byte:  %.3f ms  → %.1f GB/s  (%.1f%% vs 1-byte)\n",
           t_4byte, total_mb / t_4byte / 1e3, (t_byte / t_4byte - 1) * 100);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_tokens); cudaFree(d_counts);
    free(h_src); free(h_tokens); free(h_counts);
    return 0;
}
