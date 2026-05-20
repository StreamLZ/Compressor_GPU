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
    uint32_t *lut = (uint32_t*)malloc(LUT_SIZE * sizeof(uint32_t));

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
    uint32_t *d_lut;
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
    uint32_t *lut = (uint32_t*)malloc(LUT_SIZE * sizeof(uint32_t));

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
    uint32_t *d_lut;
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
    uint32_t *d_lut_unused;
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
    uint32_t *lut = (uint32_t*)malloc(LUT_BYTES);
    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    // GPU buffers.
    uint8_t  *d_input, *d_scratch, *d_enc_out, *d_dec_out;
    uint32_t *d_lut;
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

// ── 32-stream variant: full GPU roundtrip + bench ─────────────────────
static int test_gpu_encode_decode_32s(const char *name, const uint8_t *src, size_t n) {
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n; i++) hist[src[i]]++;
    uint8_t code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    uint32_t *lut = (uint32_t*)malloc(LUT_BYTES);
    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    uint8_t *d_input, *d_scratch, *d_enc_out, *d_dec_out;
    uint32_t *d_lut;
    uint32_t *d_codes;
    uint8_t  *d_code_lengths;
    HuffBlockDesc *d_descs_enc, *d_descs_dec;
    uint32_t *d_out_sizes, *d_out_offsets;

    size_t scratch_per_stream = (n / 32 + 64) * 2;
    size_t enc_cap = n * 2 + 256;

    CK(cudaMalloc(&d_input, n));
    CK(cudaMalloc(&d_scratch, scratch_per_stream * 32));
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

    HuffBlockDesc enc_desc = {0, (uint32_t)n, 0, 0, 0};
    CK(cudaMemcpy(d_descs_enc, &enc_desc, sizeof(enc_desc), cudaMemcpyHostToDevice));
    uint32_t out_off = 0;
    CK(cudaMemcpy(d_out_offsets, &out_off, sizeof(uint32_t), cudaMemcpyHostToDevice));

    huffEncode32StreamKernel<<<1, 32>>>(
        d_input, d_descs_enc, d_code_lengths, d_codes,
        d_scratch, d_enc_out, d_out_sizes, d_out_offsets,
        (uint32_t)scratch_per_stream, 256, 1);
    CK(cudaDeviceSynchronize());

    uint32_t enc_bytes = 0;
    CK(cudaMemcpy(&enc_bytes, d_out_sizes, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    HuffBlockDesc dec_desc = {0, enc_bytes, 0, (uint32_t)n, 0};
    CK(cudaMemcpy(d_descs_dec, &dec_desc, sizeof(dec_desc), cudaMemcpyHostToDevice));
    huffDecode32StreamKernel<<<1, 32, LUT_BYTES>>>(d_enc_out, d_descs_dec, d_lut, d_dec_out, 1);
    CK(cudaDeviceSynchronize());

    uint8_t *gpu_out = (uint8_t*)malloc(n);
    CK(cudaMemcpy(gpu_out, d_dec_out, n, cudaMemcpyDeviceToHost));

    int fails = 0;
    if (memcmp(src, gpu_out, n) != 0) {
        printf("[%s/GPU-RT-32s] FAIL\n", name);
        for (size_t i = 0; i < n; i++) {
            if (src[i] != gpu_out[i]) {
                printf("  first diff at %zu: src=0x%02x gpu=0x%02x\n", i, src[i], gpu_out[i]);
                break;
            }
        }
        fails = 1;
    } else {
        printf("[%s/GPU-RT-32s] OK n=%zu enc=%u (%.1f%%) max_len=%d\n",
               name, n, enc_bytes, 100.0 * enc_bytes / (double)n, max_len);
    }

    cudaFree(d_input); cudaFree(d_scratch); cudaFree(d_enc_out); cudaFree(d_dec_out);
    cudaFree(d_lut); cudaFree(d_codes); cudaFree(d_code_lengths);
    cudaFree(d_descs_enc); cudaFree(d_descs_dec);
    cudaFree(d_out_sizes); cudaFree(d_out_offsets);
    free(lut); free(gpu_out);
    return fails;
}

static void bench_gpu_decode_32s_parallel(const uint8_t *src, size_t n_per_block, int n_blocks) {
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n_per_block; i++) hist[src[i]]++;
    uint8_t code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    uint32_t *lut = (uint32_t*)malloc(LUT_BYTES);
    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    // Encode one block CPU-side (32-stream) using CPU ref oracle.
    size_t enc_cap_per = n_per_block * 2 + 256;
    uint8_t *enc_one = (uint8_t*)malloc(enc_cap_per);
    int enc_bytes_per = huff_encode_ns(32, src, n_per_block, code_lengths, codes, enc_one, enc_cap_per);

    size_t total_comp = (size_t)enc_bytes_per * n_blocks;
    size_t total_out = n_per_block * n_blocks;
    uint8_t *comp_buf = (uint8_t*)malloc(total_comp);
    for (int b = 0; b < n_blocks; b++) memcpy(comp_buf + b * enc_bytes_per, enc_one, enc_bytes_per);

    std::vector<HuffBlockDesc> descs(n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        descs[b].in_offset = b * enc_bytes_per;
        descs[b].in_size = enc_bytes_per;
        descs[b].out_offset = b * n_per_block;
        descs[b].out_size = (uint32_t)n_per_block;
        descs[b].lut_offset = 0;
    }

    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, total_comp));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, total_comp, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut, LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode32StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode32StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(src, gpu_out + b * n_per_block, n_per_block) != 0) {
            verify_fail++;
            if (verify_fail >= 3) break;
        }
    }
    double gb_per_s = (double)total_out / (best_ms * 1e6);
    printf("[bench-dec-32s] %d blocks × %zu B = %.1f MB → best %.3f ms = %.1f GB/s%s\n",
           n_blocks, n_per_block, total_out / 1e6, best_ms, gb_per_s,
           verify_fail ? " [VERIFY FAIL]" : "");
    (void)max_len;

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(enc_one); free(lut); free(gpu_out);
}

// ── Realistic bench: distinct 64 KB blocks from a real file ────────────
// Unlike bench_gpu_decode_parallel (which tiles ONE encoded block N times
// and shares ONE LUT), this splits the actual file into distinct 64 KB
// blocks, builds a per-block Huffman table + LUT, and benches decoding
// every block with its own LUT — i.e. exactly what the real pipeline does.
// Block count is whatever the file yields (enwik8 ~1525, silesia ~3247);
// no artificial tiling, no unrealistic 24000-block runs.
static void bench_gpu_decode_realfile(const uint8_t *file, size_t file_size,
                                      size_t BLK) {
    int n_blocks = (int)(file_size / BLK);  // whole blocks only
    if (n_blocks < 1) { printf("[realfile] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

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

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile] %d blocks x %zu KB (4-stream, %zu B/lane) = %.1f MB, "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, BLK / 1024, BLK / 4, total_out / 1e6, ratio, best_ms, gb_per_s,
           verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: 32-stream variant ─────────────────────────────────
// Same distinct-block / per-block-LUT setup as bench_gpu_decode_realfile,
// but encodes each 64 KB block as 32 sub-streams and decodes with
// huffDecode32StreamKernel — all 32 warp lanes do decode work (vs only
// 4 in the 4-stream kernel). Direct A/B for the 4-of-32-idle-lanes fix.
static void bench_gpu_decode_realfile_32s(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-32s] input < 64 KB — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)256 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_ns(32, src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)256 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-32s] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode32StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode32StreamKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-32s] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-32s] %d distinct 64KB blocks = %.1f MB, per-block LUT, "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, total_out / 1e6, ratio, best_ms, gb_per_s,
           verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: 8-lane variant (2 blocks per warp) ────────────────
// Identical 4-stream wire format and distinct-block / per-block-LUT setup
// as bench_gpu_decode_realfile, but decodes with huffDecode4Stream2xKernel
// — one warp handles 2 blocks (8 active lanes). A/B vs the 4-lane kernel:
// does 2x issue-slot utilization beat the cost of 16KB shared LUT (lower
// occupancy) and half the warp count?
static void bench_gpu_decode_realfile_2x(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-2x] input < 64 KB — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;
        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);
        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-2x] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    int n_warps = (n_blocks + 1) / 2;             // 2 blocks per warp
    size_t smem = 2 * (size_t)LUT_BYTES;          // two LUTs

    huffDecode4Stream2xKernel<<<n_warps, 32, smem>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4Stream2xKernel<<<n_warps, 32, smem>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-2x] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    printf("[realfile-2x] %d blocks, 2/warp (8 lanes), per-block LUT "
           "-> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: 2-lane variant (2 streams per lane) ───────────────
// Same 4-stream wire format / per-block-LUT setup, decoded with
// huffDecode4Stream2LaneKernel — only 2 active lanes, each decoding 2
// streams. The "narrow" end of the active-lane sweep.
static void bench_gpu_decode_realfile_2lane(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-2lane] input < 64 KB — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;
        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);
        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-2lane] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4Stream2LaneKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4Stream2LaneKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-2lane] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    printf("[realfile-2lane] %d blocks, 2 lanes x 2 streams, per-block LUT "
           "-> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: word-interleaved 4-stream variant ─────────────────
// Distinct 64KB blocks, per-block LUT, encoded with the word-interleaved
// 4-stream format and decoded by huffDecode4StreamInterleavedKernel. The
// 4 decode lanes' refill loads coalesce (vs 4 scattered sectors in the
// contiguous layout) — the fix for the 43.5% global-load stall NCU found.
static void bench_gpu_decode_realfile_interleaved(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-il] input < 64 KB — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)256 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;
        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);
        int enc_bytes = huff_encode_4s_interleaved(src, BLK, code_lengths, codes,
                                                   comp_buf + comp_off,
                                                   file_size * 2 + (size_t)256 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-il] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamInterleavedKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamInterleavedKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-il] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-il] %d blocks, word-interleaved 4-stream, per-block LUT, "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: software-prefetched 4-stream variant ──────────────
// Identical contiguous format / setup as bench_gpu_decode_realfile, but
// decodes with huffDecode4StreamPrefetchKernel — the decode core issues
// each 32-bit refill load one refill ahead to hide global-load latency
// (NCU flagged the kernel as latency-bound on the refill load).
static void bench_gpu_decode_realfile_prefetch(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-pf] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-pf] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamPrefetchKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamPrefetchKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-pf] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-pf] %d blocks x 64 KB (4-stream prefetch, per-block LUT), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: u16 single-symbol LUT variant ─────────────────────
// Same contiguous format / setup as bench_gpu_decode_realfile, but the
// per-block LUT is u16 (4 KB) instead of u32 (8 KB), decoded by
// huffDecode4StreamLut16Kernel. Halving shared memory per warp roughly
// doubles the SM's shared-mem-limited occupancy (NCU: 22.9% -> ~46%).
static void bench_gpu_decode_realfile_lut16(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-l16] input < block size — skipping\n"); return; }

    uint16_t *lut_buf  = (uint16_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint16_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut16(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-l16] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint16_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT16_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT16_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamLut16Kernel<<<n_blocks, 32, LUT16_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamLut16Kernel<<<n_blocks, 32, LUT16_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-l16] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-l16] %d blocks x 64 KB (4-stream u16 LUT, per-block LUT), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: wide-output-store variant ─────────────────────────
// Same contiguous format / u32 LUT / setup as bench_gpu_decode_realfile,
// but decoded by huffDecode4StreamWStoreKernel — the decode core flushes
// output as 32-bit stores instead of single bytes, cutting store
// transactions and L2 store traffic ~4×.
static void bench_gpu_decode_realfile_wstore(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-ws] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-ws] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamWStoreKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamWStoreKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-ws] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-ws] %d blocks x 64 KB (4-stream 32-bit stores, per-block LUT), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: 64-bit wide-output-store variant ──────────────────
// Same setup as bench_gpu_decode_realfile_wstore, but decoded by
// huffDecode4StreamWStore64Kernel — 8-byte output stores.
static void bench_gpu_decode_realfile_wstore64(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-w64] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s(src, BLK, code_lengths, codes,
                                       comp_buf + comp_off,
                                       file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-w64] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamWStore64Kernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamWStore64Kernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-w64] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-w64] %d blocks x 64 KB (4-stream 64-bit stores, per-block LUT), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: aligned 32-bit refill + wide-store variant ────────
// Encodes with huff_encode_4s_aligned (12-byte header, each stream padded
// to a 4-byte boundary) and decodes with huffDecode4StreamWStoreAlignedKernel
// — single aligned 32-bit refill load instead of four byte loads.
static void bench_gpu_decode_realfile_wstore_aligned(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-wsa] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN) height_limit(hist, code_lengths, MAX_CODE_LEN);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s_aligned(src, BLK, code_lengths, codes,
                                               comp_buf + comp_off,
                                               file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-wsa] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;  // enc_bytes is a multiple of 4 -> blocks stay 4-aligned
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamWStoreAlignedKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamWStoreAlignedKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-wsa] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-wsa] %d blocks x 64 KB (4-stream aligned refill + 32-bit stores), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: 10-bit-LUT + 11-bit-escape variant ────────────────
// Codes height-limited to 11 (NO ratio loss vs the 11-bit codec), but the
// GPU LUT is 1024-entry (4 KB) with escape entries for length-11 codes —
// keeping the doubled occupancy of the simple 10-bit variant. Decoded by
// huffDecode4StreamWStoreAlignedEscKernel.
static void bench_gpu_decode_realfile_wstore_aligned_esc(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-wse] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        // Height-limit to 11 (not MAX_CODE_LEN=10) — escape LUT covers 11-bit codes.
        if (max_len > MAX_CODE_LEN + 1) height_limit(hist, code_lengths, MAX_CODE_LEN + 1);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut_esc(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s_aligned(src, BLK, code_lengths, codes,
                                               comp_buf + comp_off,
                                               file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-wse] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamWStoreAlignedEscKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamWStoreAlignedEscKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-wse] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-wse] %d blocks x 64 KB (4-stream 10-bit LUT + 11-bit escape), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: persistent atomic-work-queue variant ──────────────
// Same encode/format/LUT as bench_gpu_decode_realfile_wstore_aligned_esc,
// but decoded by the persistent huffDecode4StreamWStoreAlignedEscQueueKernel
// — grid = one wave of CUDA blocks, each draining a global atomic counter.
static void bench_gpu_decode_realfile_wstore_aligned_esc_queue(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-wseq] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN + 1) height_limit(hist, code_lengths, MAX_CODE_LEN + 1);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut_esc(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s_aligned(src, BLK, code_lengths, codes,
                                               comp_buf + comp_off,
                                               file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-wseq] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut, *d_counter;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMalloc(&d_counter, sizeof(uint32_t)));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    // Persistent grid: one full wave of CUDA blocks (SMs × blocks/SM).
    int numSM = 0, blocksPerSM = 0;
    cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM,
        huffDecode4StreamWStoreAlignedEscQueueKernel, 32, LUT_BYTES);
    int grid = numSM * blocksPerSM;
    if (grid < 1) grid = 1;
    if (grid > n_blocks) grid = n_blocks;

    CK(cudaMemset(d_counter, 0, sizeof(uint32_t)));
    huffDecode4StreamWStoreAlignedEscQueueKernel<<<grid, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks, d_counter);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaMemset(d_counter, 0, sizeof(uint32_t));  // before e0 -> not timed
        cudaEventRecord(e0);
        huffDecode4StreamWStoreAlignedEscQueueKernel<<<grid, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks, d_counter);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-wseq] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-wseq] %d blocks, persistent queue (%d-block grid = %d SM x %d), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, grid, numSM, blocksPerSM, ratio, best_ms, gb_per_s,
           verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs); cudaFree(d_counter);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: cp.async ring-buffer-refill variant ───────────────
// Same encode/format/escape-LUT as bench_gpu_decode_realfile_wstore_aligned_esc,
// but decoded by huffDecode4StreamCpAsyncEscKernel — bit-buffer refill fed
// by cp.async (global→shared) instead of a synchronous load.
static void bench_gpu_decode_realfile_cpasync_esc(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-cpa] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN + 1) height_limit(hist, code_lengths, MAX_CODE_LEN + 1);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut_esc(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s_aligned(src, BLK, code_lengths, codes,
                                               comp_buf + comp_off,
                                               file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-cpa] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamCpAsyncEscKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamCpAsyncEscKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-cpa] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-cpa] %d blocks x 64 KB (4-stream cp.async refill + escape LUT), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── Realistic bench: cp.async refill + cold-branch escape ──────────────
// Same encode/format/escape-LUT as bench_gpu_decode_realfile_cpasync_esc,
// decoded by huffDecode4StreamCpAsyncEscCbKernel (escape on a cold branch).
static void bench_gpu_decode_realfile_cpasync_esc_cb(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[realfile-cpb] input < block size — skipping\n"); return; }

    uint32_t *lut_buf  = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));
    uint8_t  *comp_buf = (uint8_t*)malloc(file_size * 2 + (size_t)64 * n_blocks);
    std::vector<HuffBlockDesc> descs(n_blocks);

    size_t comp_off = 0;
    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN + 1) height_limit(hist, code_lengths, MAX_CODE_LEN + 1);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut_esc(code_lengths, codes, lut_buf + (size_t)b * LUT_SIZE);

        int enc_bytes = huff_encode_4s_aligned(src, BLK, code_lengths, codes,
                                               comp_buf + comp_off,
                                               file_size * 2 + (size_t)64 * n_blocks - comp_off);
        if (enc_bytes <= 0) { printf("[realfile-cpb] block %d encode fail\n", b); return; }
        descs[b].in_offset  = (uint32_t)comp_off;
        descs[b].in_size    = (uint32_t)enc_bytes;
        descs[b].out_offset = (uint32_t)((size_t)b * BLK);
        descs[b].out_size   = (uint32_t)BLK;
        descs[b].lut_offset = (uint32_t)((size_t)b * LUT_SIZE);
        comp_off += enc_bytes;
    }

    size_t total_out = (size_t)n_blocks * BLK;
    uint8_t *d_comp, *d_out;
    uint32_t *d_lut;
    HuffBlockDesc *d_descs;
    CK(cudaMalloc(&d_comp, comp_off));
    CK(cudaMalloc(&d_out, total_out));
    CK(cudaMalloc(&d_lut, (size_t)n_blocks * LUT_BYTES));
    CK(cudaMalloc(&d_descs, sizeof(HuffBlockDesc) * n_blocks));
    CK(cudaMemcpy(d_comp, comp_buf, comp_off, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_lut, lut_buf, (size_t)n_blocks * LUT_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(), sizeof(HuffBlockDesc) * n_blocks, cudaMemcpyHostToDevice));

    huffDecode4StreamCpAsyncEscCbKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffDecode4StreamCpAsyncEscCbKernel<<<n_blocks, 32, LUT_BYTES>>>(d_comp, d_descs, d_lut, d_out, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }

    uint8_t *gpu_out = (uint8_t*)malloc(total_out);
    CK(cudaMemcpy(gpu_out, d_out, total_out, cudaMemcpyDeviceToHost));
    int verify_fail = 0;
    for (int b = 0; b < n_blocks; b++) {
        if (memcmp(file + (size_t)b * BLK, gpu_out + (size_t)b * BLK, BLK) != 0) {
            if (verify_fail < 3) printf("  [realfile-cpb] block %d mismatch\n", b);
            verify_fail++;
        }
    }

    double gb_per_s = (double)total_out / (best_ms * 1e6);
    double ratio = 100.0 * (double)comp_off / (double)total_out;
    printf("[realfile-cpb] %d blocks x 64 KB (cp.async refill + cold-branch escape), "
           "enc %.1f%% -> best %.3f ms = %.1f GB/s%s\n",
           n_blocks, ratio, best_ms, gb_per_s, verify_fail ? " [VERIFY FAIL]" : "");

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_lut); cudaFree(d_descs);
    free(comp_buf); free(lut_buf); free(gpu_out);
}

// ── LUT-build bench: serial (lane-0) vs parallel (32-lane) ─────────────
// The production split timer showed the per-block decode LUT *build* is
// ~68% of the Huffman pass (lane-0-serial pass1+pass2). This A/Bs the
// serial build kernel against the 32-lane parallel build, and verifies
// both produce a bit-identical LUT vs the CPU `build_decode_lut_esc`.
static void bench_lut_build(const uint8_t *file, size_t file_size) {
    const size_t BLK = 65536;
    int n_blocks = (int)(file_size / BLK);
    if (n_blocks < 1) { printf("[lut-build] input < block size — skipping\n"); return; }

    uint8_t  *cl_all  = (uint8_t*)malloc((size_t)n_blocks * 256);
    uint32_t *lut_ref = (uint32_t*)malloc((size_t)n_blocks * LUT_SIZE * sizeof(uint32_t));

    uint8_t  code_lengths[256];
    uint32_t codes[256];
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t *src = file + (size_t)b * BLK;
        uint32_t hist[256] = {0};
        for (size_t i = 0; i < BLK; i++) hist[src[i]]++;

        memset(code_lengths, 0, sizeof(code_lengths));
        int max_len = build_code_lengths(hist, code_lengths);
        if (max_len > MAX_CODE_LEN + 1) height_limit(hist, code_lengths, MAX_CODE_LEN + 1);
        memset(codes, 0, sizeof(codes));
        assign_canonical_codes(code_lengths, codes);
        build_decode_lut_esc(code_lengths, codes, lut_ref + (size_t)b * LUT_SIZE);
        memcpy(cl_all + (size_t)b * 256, code_lengths, 256);
    }

    uint8_t  *d_cl;
    uint32_t *d_lut_s, *d_lut_p;
    CK(cudaMalloc(&d_cl,    (size_t)n_blocks * 256));
    CK(cudaMalloc(&d_lut_s, (size_t)n_blocks * LUT_SIZE * sizeof(uint32_t)));
    CK(cudaMalloc(&d_lut_p, (size_t)n_blocks * LUT_SIZE * sizeof(uint32_t)));
    CK(cudaMemcpy(d_cl, cl_all, (size_t)n_blocks * 256, cudaMemcpyHostToDevice));

    // Warmup both.
    huffBuildLutEscSerialKernel<<<n_blocks, 32>>>(d_cl, d_lut_s, n_blocks);
    huffBuildLutEscParallelKernel<<<n_blocks, 32>>>(d_cl, d_lut_p, n_blocks);
    CK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;

    float best_s = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffBuildLutEscSerialKernel<<<n_blocks, 32>>>(d_cl, d_lut_s, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_s) best_s = ms;
    }
    float best_p = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        huffBuildLutEscParallelKernel<<<n_blocks, 32>>>(d_cl, d_lut_p, n_blocks);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_p) best_p = ms;
    }

    // Verify both LUTs bit-identical to the CPU reference.
    size_t lut_bytes = (size_t)n_blocks * LUT_SIZE * sizeof(uint32_t);
    uint32_t *h_lut_s = (uint32_t*)malloc(lut_bytes);
    uint32_t *h_lut_p = (uint32_t*)malloc(lut_bytes);
    CK(cudaMemcpy(h_lut_s, d_lut_s, lut_bytes, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_lut_p, d_lut_p, lut_bytes, cudaMemcpyDeviceToHost));
    int fail_s = memcmp(h_lut_s, lut_ref, lut_bytes) != 0;
    int fail_p = memcmp(h_lut_p, lut_ref, lut_bytes) != 0;
    if (fail_s || fail_p) {
        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < LUT_SIZE; i++) {
                size_t k = (size_t)b * LUT_SIZE + i;
                if (fail_s && h_lut_s[k] != lut_ref[k]) {
                    printf("  [lut-build] serial   blk %d idx %d: gpu %08x ref %08x\n",
                           b, i, h_lut_s[k], lut_ref[k]); fail_s = 2; break;
                }
            }
            if (fail_s == 2) break;
        }
        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < LUT_SIZE; i++) {
                size_t k = (size_t)b * LUT_SIZE + i;
                if (fail_p && h_lut_p[k] != lut_ref[k]) {
                    printf("  [lut-build] parallel blk %d idx %d: gpu %08x ref %08x\n",
                           b, i, h_lut_p[k], lut_ref[k]); fail_p = 2; break;
                }
            }
            if (fail_p == 2) break;
        }
    }

    printf("\n--- LUT-build bench (%d blocks, 64 KB each) ---\n", n_blocks);
    printf("[lut-build] serial   (lane-0): best %.3f ms%s\n",
           best_s, fail_s ? "  [VERIFY FAIL]" : "  [verified]");
    printf("[lut-build] parallel (32-lane): best %.3f ms%s  (%.2fx)\n",
           best_p, fail_p ? "  [VERIFY FAIL]" : "  [verified]",
           best_s / best_p);

    cudaFree(d_cl); cudaFree(d_lut_s); cudaFree(d_lut_p);
    free(cl_all); free(lut_ref); free(h_lut_s); free(h_lut_p);
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
            // Read the whole file so the bench uses real, distinct content.
            uint8_t *full = (uint8_t*)malloc((size_t)sz);
            size_t got = fread(full, 1, (size_t)sz, f);
            fclose(f);
            (void)got;

            // Correctness on the first 64 KB block.
            size_t cap = sz > (1 << 16) ? (1 << 16) : (size_t)sz;
            fails += test_gpu_decode_one("real-64KB", full, cap);
            fails += test_gpu_encode_decode("real-64KB", full, cap);
            fails += test_gpu_encode_decode_32s("real-64KB", full, cap);

            if (fails == 0) {
                // Realistic decode bench: every block is distinct file
                // content with its own Huffman table + LUT.
                printf("\n--- realistic decode bench (per-block LUT, distinct blocks) ---\n");
                bench_gpu_decode_realfile(full, (size_t)sz, 65536);
                bench_gpu_decode_realfile_wstore(full, (size_t)sz);
                bench_gpu_decode_realfile_wstore_aligned(full, (size_t)sz);
                bench_gpu_decode_realfile_wstore_aligned_esc(full, (size_t)sz);
                bench_gpu_decode_realfile_cpasync_esc(full, (size_t)sz);
                bench_gpu_decode_realfile_cpasync_esc_cb(full, (size_t)sz);
                bench_lut_build(full, (size_t)sz);
                bench_gpu_decode_realfile_wstore_aligned_esc_queue(full, (size_t)sz);
                bench_gpu_decode_realfile_wstore64(full, (size_t)sz);
                bench_gpu_decode_realfile_lut16(full, (size_t)sz);
                bench_gpu_decode_realfile_prefetch(full, (size_t)sz);
                bench_gpu_decode_realfile_interleaved(full, (size_t)sz);
                bench_gpu_decode_realfile_2lane(full, (size_t)sz);
                bench_gpu_decode_realfile_2x(full, (size_t)sz);
                bench_gpu_decode_realfile_32s(full, (size_t)sz);
                // Decoupling experiment: 4-stream kernel with smaller
                // blocks. 16KB block -> 4KB/lane, 8KB block -> 2KB/lane
                // (same per-lane work as the 32-stream kernel). If these
                // are also slow, small per-lane streams are the cause,
                // not the count of active lanes.
                printf("--- decouple: 4-stream kernel, smaller blocks ---\n");
                bench_gpu_decode_realfile(full, (size_t)sz, 16384);
                bench_gpu_decode_realfile(full, (size_t)sz, 8192);
            }
            free(full);
        }
    }

    printf("\n%d failures\n", fails);
    return fails ? 1 : 0;
}
