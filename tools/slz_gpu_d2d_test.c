/* End-to-end D2D test for slzCompress + slzDecompress via the C ABI.
 * Reads enwik8.txt (or argv[1]) up to argv[2] bytes (default 1 MB),
 * runs compress + decompress at L1/L3/L5 entirely on the device,
 * byte-exact roundtrip check. Exits 0 on all-pass. */
#include "streamlz_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int roundtrip(slzHandle_t h, const unsigned char* input, size_t n, int level) {
    setvbuf(stdout, NULL, _IONBF, 0);
    slzCompressOpts_t opts = slzCompressDefaultOpts();
    opts.level = level;
    size_t bound = 0;
    if (slzCompressBound(h, n, opts, &bound) != SLZ_SUCCESS) { printf("L%d bound failed\n", level); return 1; }

    void *d_in = NULL, *d_frame = NULL, *d_out = NULL;
    if (cudaMalloc(&d_in, n) || cudaMalloc(&d_frame, bound) || cudaMalloc(&d_out, n + 64)) {
        printf("L%d cudaMalloc failed\n", level); return 1;
    }
    cudaMemcpy(d_in, input, n, cudaMemcpyHostToDevice);

    size_t frame_len = 0;
    slzStatus_t s = slzCompress(h, d_in, n, NULL, 0, d_frame, bound, &frame_len, opts);
    if (s != SLZ_SUCCESS) { printf("L%d compress failed: %s\n", level, slzStatusString(s)); return 1; }

    size_t out_len = 0;
    s = slzDecompress(h, d_frame, frame_len, NULL, 0, d_out, n + 64, &out_len);
    if (s != SLZ_SUCCESS) { printf("L%d decompress failed: %s\n", level, slzStatusString(s)); return 1; }

    unsigned char* result = (unsigned char*)malloc(out_len);
    cudaMemcpy(result, d_out, out_len, cudaMemcpyDeviceToHost);
    int ok = (out_len == n) && memcmp(input, result, n) == 0;
    printf("L%d in=%zu frame=%zu out=%zu %s\n", level, n, frame_len, out_len, ok ? "OK" : "FAIL");
    if (!ok) {
        for (size_t i = 0; i < out_len && i < n; i++) {
            if (input[i] != result[i]) {
                printf("  first mismatch at byte %zu: in=0x%02x out=0x%02x\n",
                       i, input[i], result[i]);
                break;
            }
        }
    }
    cudaFree(d_in); cudaFree(d_frame); cudaFree(d_out);
    free(result);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* path = argc > 1 ? argv[1] : "assets/enwik8.txt";
    size_t want = argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : (1u << 20);

    FILE* f = fopen(path, "rb");
    if (!f) { printf("open %s failed\n", path); return 1; }
    fseek(f, 0, SEEK_END); size_t fsz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    size_t n = want < fsz ? want : fsz;
    unsigned char* input = (unsigned char*)malloc(n);
    if (fread(input, 1, n, f) != n) { printf("read failed\n"); return 1; }
    fclose(f);
    printf("loaded %zu bytes from %s\n", n, path);

    slzHandle_t h = NULL;
    if (slzCreate(&h) != SLZ_SUCCESS) { printf("slzCreate failed\n"); return 1; }
    int rc = 0;
    rc |= roundtrip(h, input, n, 1);
    rc |= roundtrip(h, input, n, 3);
    rc |= roundtrip(h, input, n, 5);
    slzDestroy(h);
    free(input);
    return rc;
}
