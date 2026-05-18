// Host harness for the GPU Huffman decoder. Uses the CPU reference
// (`huff_ref.c`) as the oracle for encoding and validates that the
// GPU kernel decodes byte-exact.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

// Pull in GPU kernel + constants first so MAX_CODE_LEN / LUT_SIZE are defined
// before huff_ref.c re-defines them (its #ifndef guard then no-ops).
#include "huff_kernel.cu"

// CPU reference oracle. HUFF_REF_NO_MAIN suppresses its main().
#define HUFF_REF_NO_MAIN
#include "huff_ref.c"

#define CK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ── Test one Huffman block end-to-end ──────────────────────────────────
static int test_gpu_decode_one(const char *name, const uint8_t *src, size_t n) {
    // CPU side: build tables + encode (4-stream)
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n; i++) hist[src[i]]++;

    uint8_t code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    uint16_t *lut = (uint16_t*)malloc(LUT_SIZE * sizeof(uint16_t));

    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) {
        height_limit(hist, code_lengths, MAX_CODE_LEN);
    }
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    size_t enc_cap = n * 2 + 64;
    uint8_t *enc = (uint8_t*)malloc(enc_cap);
    int enc_bytes = huff_encode_4s(src, n, code_lengths, codes, enc, enc_cap);
    if (enc_bytes <= 0) {
        printf("[%s] FAIL CPU encode\n", name);
        free(enc); free(lut); return 1;
    }

    // GPU buffers
    uint8_t *d_comp, *d_out;
    uint16_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, enc_bytes));
    CK(cudaMalloc(&d_out, n));
    CK(cudaMalloc(&d_lut, LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc)));

    CK(cudaMemcpy(d_comp, enc, enc_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut, LUT_BYTES, cudaMemcpyHostToDevice));

    HuffBlockDesc desc = {0, (uint32_t)enc_bytes, 0, (uint32_t)n, 0};
    CK(cudaMemcpy(d_descs, &desc, sizeof(desc), cudaMemcpyHostToDevice));

    // Launch: 1 block, 32 threads, 4KB shared.
    huffDecode4StreamKernel<<<1, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, 1);
    CK(cudaDeviceSynchronize());

    uint8_t *gpu_out = (uint8_t*)malloc(n);
    CK(cudaMemcpy(gpu_out, d_out, n, cudaMemcpyDeviceToHost));

    int fails = 0;
    if (memcmp(src, gpu_out, n) != 0) {
        printf("[%s] FAIL GPU decode mismatch\n", name);
        for (size_t i = 0; i < n; i++) {
            if (src[i] != gpu_out[i]) {
                printf("  first diff at %zu: src=0x%02x gpu=0x%02x\n", i, src[i], gpu_out[i]);
                break;
            }
        }
        fails = 1;
    } else {
        printf("[%s] OK n=%zu enc=%d (%.1f%%) max_len=%d\n",
               name, n, enc_bytes, 100.0 * enc_bytes / (double)n, max_len);
    }

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(enc); free(gpu_out); free(lut);
    return fails;
}

// ── Bench: many parallel blocks ────────────────────────────────────────
static void bench_gpu_decode_parallel(const uint8_t *src, size_t n_per_block, int n_blocks) {
    // Build a single Huffman table for all blocks (same data per block).
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n_per_block; i++) hist[src[i]]++;

    uint8_t code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    uint16_t *lut = (uint16_t*)malloc(LUT_SIZE * sizeof(uint16_t));

    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    size_t enc_cap_per = n_per_block * 2 + 64;
    uint8_t *enc_one = (uint8_t*)malloc(enc_cap_per);
    int enc_bytes_per = huff_encode_4s(src, n_per_block, code_lengths, codes, enc_one, enc_cap_per);

    size_t total_comp = (size_t)enc_bytes_per * n_blocks;
    size_t total_out  = n_per_block * n_blocks;
    uint8_t *comp_buf = (uint8_t*)malloc(total_comp);
    for (int b = 0; b < n_blocks; b++) memcpy(comp_buf + b * enc_bytes_per, enc_one, enc_bytes_per);

    std::vector<HuffBlockDesc> descs(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        descs[b].in_offset = b * enc_bytes_per;
        descs[b].in_size = enc_bytes_per;
        descs[b].out_offset = b * n_per_block;
        descs[b].out_size = (uint32_t)n_per_block;
        descs[b].lut_offset = 0;  // all share same LUT
    }

    uint8_t *d_comp, *d_out;
    uint16_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, total_comp));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));

    CK(cudaMemcpy(d_comp, comp_buf, total_comp, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut, LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    // Warmup.
    huffDecode4StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    // Bench loop.
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    // Verify a few random blocks.
    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(src, gpu_out + b * n_per_block, n_per_block) != 0) {
            printf("  bench: block %d mismatch\n", b);
            verify_fail++;
            if (verify_fail >= 3) break;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    printf("[bench] %d blocks × %zu B = %.1f MB → best %.3f ms = %.1f GB/s%s\n",
           n_blocks, n_per_block, total_out / 1e6, best_ms, gb_per_s,
           verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(enc_one); free(lut); free(gpu_out);
}

// ── Bench: parallel GPU encode ─────────────────────────────────────────
static void bench_gpu_encode_parallel(const uint8_t *src, size_t n_per_block, int n_blocks) {
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n_per_block; i++) hist[src[i]]++;
    uint8_t code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
    assign_canonical_codes(code_lengths, codes);

    size_t total_in = n_per_block * n_blocks;
    size_t scratch_per_stream = (n_per_block / 4 + 64) * 2;
    size_t enc_cap_per = n_per_block * 2 + 64;
    size_t total_scratch = scratch_per_stream * 4 * n_blocks;
    size_t total_enc = enc_cap_per * n_blocks;

    uint8_t *d_input, *d_scratch, *d_enc_out;
    uint16_t *d_lut_unused;
    uint32_t *d_codes;
    uint8_t  *d_code_lengths;
    HuffBlockDesc *d_descs;
    uint32_t *d_out_sizes, *d_out_offsets;

    CK(cudaMalloc(&d_input, total_in));
    CK(cudaMalloc(&d_scratch, total_scratch));
    CK(cudaMalloc(&d_enc_out, total_enc));
    CK(cudaMalloc(&d_lut_unused, LUT_BYTES));
    CK(cudaMalloc(&d_codes, 256 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_code_lengths, 256));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMalloc(&d_out_sizes, sizeof(uint32_t) * n_blocks));
    CK(cudaMalloc(&d_out_offsets, sizeof(uint32_t) * n_blocks));

    // Tile input: copy the same n_per_block buffer n_blocks times.
    for (int b = 0; b < n_blocks; b++)
        CK(cudaMemcpy(d_input + b * n_per_block, src, n_per_block, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_codes, codes, 256 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_code_lengths, code_lengths, 256, cudaMemcpyHostToDevice));

    std::vector<HuffBlockDesc> descs(n_blocks);
    std::vector<uint32_t> out_offsets(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        descs[b] = { (uint32_t)(b * n_per_block), (uint32_t)n_per_block, 0, 0, 0 };
        out_offsets[b] = (uint32_t)(b * enc_cap_per);
    }
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_out_offsets, out_offsets.data(), sizeof(uint32_t) * n_blocks, cudaMemcpyHostToDevice));

    // Warmup.
    huffEncode4StreamKernel<<<n_blocks, 32>>>(d_input, d_descs, d_code_lengths, d_codes,
        d_scratch, d_enc_out, d_out_sizes, d_out_offsets,
        (uint32_t)scratch_per_stream, 256, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffEncode4StreamKernel<<<n_blocks, 32>>>(d_input, d_descs, d_code_lengths, d_codes,
            d_scratch, d_enc_out, d_out_sizes, d_out_offsets,
            (uint32_t)scratch_per_stream, 256, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    double gb_per_s = (double)total_in / (best_ms * 1e6);
    printf("[bench-enc] %d blocks × %zu B = %.1f MB → best %.3f ms = %.1f GB/s\n",
           n_blocks, n_per_block, total_in / 1e6, best_ms, gb_per_s);

    cudaFree(d_input); cudaFree(d_scratch); cudaFree(d_enc_out); cudaFree(d_lut_unused);
    cudaFree(d_codes); cudaFree(d_code_lengths); cudaFree(d_descs);
    cudaFree(d_out_sizes); cudaFree(d_out_offsets);
}

// ── Full GPU roundtrip: GPU encode + GPU decode ───────────────────────
static int test_gpu_encode_decode(const char *name, const uint8_t *src, size_t n) {
    // Build tables CPU-side (table-build kernel comes later).
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n; i++) hist[src[i]]++;
    uint8_t code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    uint16_t *lut = (uint16_t*)malloc(LUT_BYTES);
    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    // GPU buffers.
    uint8_t  *d_input, *d_scratch, *d_enc_out, *d_dec_out;
    uint16_t *d_lut;
    uint32_t *d_codes;
    uint8_t  *d_code_lengths;
    HuffBlockDesc *d_descs_enc, *d_descs_dec;
    uint32_t *d_out_sizes, *d_out_offsets;

    size_t scratch_per_stream = (n / 4 + 64) * 2;  // worst case per stream
    size_t enc_cap = n * 2 + 64;

    CK(cudaMalloc(&d_input, n));
    CK(cudaMalloc(&d_scratch, scratch_per_stream * 4));
    CK(cudaMalloc(&d_enc_out, enc_cap));
    CK(cudaMalloc(&d_dec_out, n));
    CK(cudaMalloc(&d_lut, LUT_BYTES));
    CK(cudaMalloc(&d_codes, 256 * sizeof(uint32_t)));
    CK(cudaMalloc(&d_code_lengths, 256));
    CK(cudaMalloc(&d_descs_enc, sizeof(HuffBlockDesc)));
    CK(cudaMalloc(&d_descs_dec, sizeof(HuffBlockDesc)));
    CK(cudaMalloc(&d_out_sizes, sizeof(uint32_t)));
    CK(cudaMalloc(&d_out_offsets, sizeof(uint32_t)));

    CK(cudaMemcpy(d_input, src, n, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut, LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_codes, codes, 256 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_code_lengths, code_lengths, 256, cudaMemcpyHostToDevice));

    HuffBlockDesc enc_desc = {0, (uint32_t)n, 0, 0, 0};  // out_offset & lut_offset unused
    CK(cudaMemcpy(d_descs_enc, &enc_desc, sizeof(enc_desc), cudaMemcpyHostToDevice));
    uint32_t out_off = 0;
    CK(cudaMemcpy(d_out_offsets, &out_off, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Encode.
    huffEncode4StreamKernel<<<1, 32>>>(
        d_input, d_descs_enc, d_code_lengths, d_codes,
        d_scratch, d_enc_out, d_out_sizes, d_out_offsets,
        (uint32_t)scratch_per_stream, 256, 1);
    CK(cudaDeviceSynchronize());

    uint32_t enc_bytes = 0;
    CK(cudaMemcpy(&enc_bytes, d_out_sizes, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Decode.
    HuffBlockDesc dec_desc = {0, enc_bytes, 0, (uint32_t)n, 0};
    CK(cudaMemcpy(d_descs_dec, &dec_desc, sizeof(dec_desc), cudaMemcpyHostToDevice));
    huffDecode4StreamKernel<<<1, 32, LUT_BYTES>>>(d_enc_out, d_descs_dec, d_lut, d_dec_out, 1);
    CK(cudaDeviceSynchronize());

    uint8_t *gpu_out = (uint8_t*)malloc(n);
    CK(cudaMemcpy(gpu_out, d_dec_out, n, cudaMemcpyDeviceToHost));

    int fails = 0;
    if (memcmp(src, gpu_out, n) != 0) {
        printf("[%s/GPU-RT] FAIL\n", name);
        for (size_t i = 0; i < n; i++) {
            if (src[i] != gpu_out[i]) {
                printf("  first diff at %zu: src=0x%02x gpu=0x%02x\n", i, src[i], gpu_out[i]);
                break;
            }
        }
        fails = 1;
    } else {
        printf("[%s/GPU-RT] OK n=%zu enc=%u (%.1f%%) max_len=%d\n",
               name, n, enc_bytes, 100.0 * enc_bytes / (double)n, max_len);
    }

    cudaFree(d_input); cudaFree(d_scratch); cudaFree(d_enc_out); cudaFree(d_dec_out);
    cudaFree(d_lut); cudaFree(d_codes); cudaFree(d_code_lengths);
    cudaFree(d_descs_enc); cudaFree(d_descs_dec);
    cudaFree(d_out_sizes); cudaFree(d_out_offsets);
    free(lut); free(gpu_out);
    return fails;
}

int main(int argc, char **argv) {
    int fails = 0;

    // ── Correctness tests ──
    {
        size_t n = 65536;
        uint8_t *buf = (uint8_t*)malloc(n);
        srand(12345);
        for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)rand();
        fails += test_gpu_decode_one("uniform-random-64KB", buf, n);
        free(buf);
    }
    {
        size_t n = 4096;
        uint8_t *buf = (uint8_t*)malloc(n);
        for (size_t i = 0; i < n; i++) buf[i] = (i % 100 == 0) ? (uint8_t)(i & 0xFF) : 'A';
        fails += test_gpu_decode_one("skewed-A-dominant", buf, n);
        free(buf);
    }
    {
        const char *s = "Hello, GPU Huffman! abcabcabc";
        fails += test_gpu_decode_one("hello-gpu", (const uint8_t*)s, strlen(s));
    }

    // Full GPU-only roundtrip tests
    {
        size_t n = 65536;
        uint8_t *buf = (uint8_t*)malloc(n);
        srand(12345);
        for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)rand();
        fails += test_gpu_encode_decode("uniform-random-64KB", buf, n);
        free(buf);
    }
    {
        size_t n = 4096;
        uint8_t *buf = (uint8_t*)malloc(n);
        for (size_t i = 0; i < n; i++) buf[i] = (i % 100 == 0) ? (uint8_t)(i & 0xFF) : 'A';
        fails += test_gpu_encode_decode("skewed-A-dominant", buf, n);
        free(buf);
    }

    // Real text
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            size_t cap = sz > (1 << 16) ? (1 << 16) : (size_t)sz;
            uint8_t *buf = (uint8_t*)malloc(cap);
            fread(buf, 1, cap, f);
            fclose(f);
            fails += test_gpu_decode_one("real-64KB", buf, cap);
            fails += test_gpu_encode_decode("real-64KB", buf, cap);
            // Bench with 1500 blocks (simulates ~100MB of literals)
            if (fails == 0) {
                bench_gpu_encode_parallel(buf, cap, 1500);
                bench_gpu_decode_parallel(buf, cap, 1500);
            }
            free(buf);
        }
    }

    printf("\n%d failures\n", fails);
    return fails ? 1 : 0;
}
