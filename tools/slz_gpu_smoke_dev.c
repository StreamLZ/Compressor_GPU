/* Device->device smoke test for the StreamLZ GPU C ABI.
 * Allocates CUDA device buffers, uploads input, exercises slzCompressAsync /
 * slzDecompressAsync on the default stream, downloads the result, and
 * checks a byte-exact roundtrip. Exits 0 on success. */
#include "streamlz_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slzHandle_t h = NULL;
    slzStatus_t s = slzCreate(&h);
    if (s != SLZ_SUCCESS) { printf("slzCreate failed: %s\n", slzStatusString(s)); return 1; }

    const size_t n = 1u << 20;
    unsigned char *input = (unsigned char *)malloc(n);
    for (size_t i = 0; i < n; i++) input[i] = (unsigned char)('A' + (i % 26));

    slzCompressOpts_t opts = slzCompressDefaultOpts();
    size_t bound = 0;
    if (slzCompressBound(h, n, opts, &bound) != SLZ_SUCCESS) { printf("bound failed\n"); return 1; }

    void *d_in = NULL, *d_frame = NULL, *d_out = NULL;
    if (cudaMalloc(&d_in, n) || cudaMalloc(&d_frame, bound) || cudaMalloc(&d_out, n + 64)) {
        printf("cudaMalloc failed\n"); return 1;
    }
    cudaMemcpy(d_in, input, n, cudaMemcpyHostToDevice);

    size_t frame_len = 0;
    s = slzCompressAsync(h, d_in, n, d_frame, bound, &frame_len, opts, NULL);
    if (s != SLZ_SUCCESS) { printf("slzCompressAsync (d2d) failed: %s\n", slzStatusString(s)); return 1; }
    cudaDeviceSynchronize();
    printf("d2d compressed %zu -> %zu bytes\n", n, frame_len);

    slzDecompressOpts_t dopts = slzDecompressDefaultOpts();
    s = slzDecompressAsync(h, d_frame, frame_len, d_out, n, dopts, NULL);
    if (s != SLZ_SUCCESS) { printf("slzDecompressAsync (d2d) failed: %s\n", slzStatusString(s)); return 1; }
    cudaDeviceSynchronize();

    unsigned char *result = (unsigned char *)malloc(n);
    cudaMemcpy(result, d_out, n, cudaMemcpyDeviceToHost);
    int ok = memcmp(input, result, n) == 0;
    printf("d2d decompressed %zu bytes, roundtrip %s\n", n, ok ? "OK" : "FAIL");

    slzDestroy(h);
    cudaFree(d_in); cudaFree(d_frame); cudaFree(d_out);
    free(input); free(result);
    return ok ? 0 : 1;
}
