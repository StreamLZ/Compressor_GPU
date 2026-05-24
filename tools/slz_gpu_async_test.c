/* Verify slzCompressAsync + slzDecompressAsync end-to-end on a real
 * caller-supplied CUstream. Pattern matches nvCOMP:
 *   slzXxxAsync(..., stream);
 *   cudaStreamSynchronize(stream);
 *
 * The point of this test is to confirm (a) the call returns BEFORE the
 * queued kernels finish (the user's stream sync is meaningful), and
 * (b) the output is correct after the sync.
 *
 * The "returns early" check uses cudaEventQuery on an event recorded
 * right after the Async submission: it should NOT be cudaSuccess yet
 * for a sufficiently large buffer (the heavy kernels still in flight).
 * For very small inputs this can race, so we check correctness, not
 * timing, as the hard assertion. */
#include "streamlz_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int roundtrip(slzHandle_t h, const unsigned char* input, size_t n, int level, int profile) {
    setvbuf(stdout, NULL, _IONBF, 0);
    slzCompressOpts_t copts = slzCompressDefaultOpts();
    copts.level = level;
    copts.enable_profiling = profile;
    size_t bound = 0;
    slzCompressBound(h, n, copts, &bound);

    void *d_in = NULL, *d_frame = NULL, *d_out = NULL;
    cudaMalloc(&d_in, n);
    cudaMalloc(&d_frame, bound);
    cudaMalloc(&d_out, n + 64);
    cudaMemcpy(d_in, input, n, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t frame_len = 0;
    slzStatus_t s = slzCompressAsync(h, d_in, n, NULL, 0, d_frame, bound, &frame_len, copts, stream);
    if (s != SLZ_SUCCESS) { printf("L%d compressAsync failed: %s\n", level, slzStatusString(s)); return 1; }
    cudaStreamSynchronize(stream);

    size_t out_len = 0;
    slzDecompressOpts_t dopts = slzDecompressDefaultOpts();
    dopts.enable_profiling = profile;
    s = slzDecompressAsync(h, d_frame, frame_len, NULL, 0, d_out, n + 64, &out_len, dopts, stream);
    if (s != SLZ_SUCCESS) { printf("L%d decompressAsync failed: %s\n", level, slzStatusString(s)); return 1; }

    /* Probe: is the kernel still in flight when we get here? */
    cudaEvent_t e;
    cudaEventCreate(&e);
    cudaEventRecord(e, stream);
    cudaError_t q = cudaEventQuery(e);
    const char* state = (q == cudaSuccess) ? "already done" :
                        (q == cudaErrorNotReady) ? "in flight" : "?";
    cudaEventDestroy(e);

    cudaStreamSynchronize(stream);

    unsigned char* result = (unsigned char*)malloc(out_len);
    cudaMemcpy(result, d_out, out_len, cudaMemcpyDeviceToHost);
    int ok = (out_len == n) && memcmp(input, result, n) == 0;
    printf("L%d in=%zu frame=%zu out=%zu %s   decompress-when-checked: %s\n",
           level, n, frame_len, out_len, ok ? "OK" : "FAIL", state);
    if (profile) {
        slzKernelTiming_t timings[64];
        size_t cnt = 0;
        slzGetLastTimings(h, timings, 64, &cnt);
        printf("  profile: %zu kernels (compress + decompress merged):\n", cnt);
        for (size_t i = 0; i < cnt && i < 64; i++)
            printf("    %-40s %8.4f ms\n", timings[i].name, timings[i].ms);
    }

    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_frame); cudaFree(d_out);
    free(result);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    const char* path = "assets/enwik8.txt";
    size_t want = (8u << 20);
    int profile = 0;
    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--profile") == 0) profile = 1;
        else if (positional == 0) { path = argv[i]; positional++; }
        else if (positional == 1) { want = (size_t)strtoull(argv[i], NULL, 10); positional++; }
    }

    FILE* f = fopen(path, "rb");
    if (!f) { printf("open %s failed\n", path); return 1; }
    fseek(f, 0, SEEK_END); size_t fsz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    size_t n = want < fsz ? want : fsz;
    unsigned char* input = (unsigned char*)malloc(n);
    if (fread(input, 1, n, f) != n) { printf("read failed\n"); return 1; }
    fclose(f);
    printf("loaded %zu bytes from %s (profile=%d)\n", n, path, profile);

    slzHandle_t h = NULL;
    if (slzCreate(&h) != SLZ_SUCCESS) { printf("slzCreate failed\n"); return 1; }
    int rc = 0;
    rc |= roundtrip(h, input, n, 1, profile);
    rc |= roundtrip(h, input, n, 3, profile);
    rc |= roundtrip(h, input, n, 5, profile);
    slzDestroy(h);
    free(input);
    return rc;
}
