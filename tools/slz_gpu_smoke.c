/* Smoke test for the StreamLZ GPU C ABI (streamlz_gpu.dll).
 * Exercises the full host->host path: create, bound, compress,
 * get-decompressed-size, decompress, destroy. Exits 0 on a byte-exact
 * roundtrip, non-zero otherwise. */
#include "streamlz_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    slzHandle_t h = NULL;
    slzStatus_t s = slzCreate(&h);
    if (s != SLZ_SUCCESS) {
        printf("slzCreate failed: %s\n", slzStatusString(s));
        return 1;
    }
    printf("streamlz_gpu version: %s\n", slzVersionString());

    /* 1 MB of compressible input. */
    const size_t n = 1u << 20;
    unsigned char *input = (unsigned char *)malloc(n);
    for (size_t i = 0; i < n; i++) input[i] = (unsigned char)('A' + (i % 26));

    slzCompressOpts_t opts = slzCompressDefaultOpts();

    size_t bound = 0;
    s = slzCompressBound(h, n, opts, &bound);
    if (s != SLZ_SUCCESS) { printf("bound failed: %s\n", slzStatusString(s)); return 1; }

    unsigned char *frame = (unsigned char *)malloc(bound);
    size_t frame_len = 0;
    s = slzCompressHost(h, input, n, frame, bound, &frame_len, opts);
    if (s != SLZ_SUCCESS) { printf("compress failed: %s\n", slzStatusString(s)); return 1; }
    printf("compressed %zu -> %zu bytes\n", n, frame_len);

    size_t dsize = 0;
    s = slzGetDecompressedSize(h, frame, frame_len, &dsize);
    if (s != SLZ_SUCCESS) { printf("get-size failed: %s\n", slzStatusString(s)); return 1; }
    printf("decompressed size from header: %zu\n", dsize);

    unsigned char *output = (unsigned char *)malloc(n + 64);
    size_t out_len = 0;
    s = slzDecompressHost(h, frame, frame_len, output, n + 64, &out_len);
    if (s != SLZ_SUCCESS) { printf("decompress failed: %s\n", slzStatusString(s)); return 1; }

    int ok = (out_len == n) && (memcmp(input, output, n) == 0);
    printf("decompressed %zu bytes, roundtrip %s\n", out_len, ok ? "OK" : "FAIL");

    slzDestroy(h);
    free(input);
    free(frame);
    free(output);
    return ok ? 0 : 1;
}
