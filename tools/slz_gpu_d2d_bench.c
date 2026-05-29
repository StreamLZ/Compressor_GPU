/* tools/slz_gpu_d2d_bench.c
 *
 * D2D-path decode benchmark, the C-ABI analogue of `streamlz -db -r N`.
 * bench_all.bat covers the CLI `-gpu` decode path (host-bounce, CPU
 * scan); this tool covers slzDecompress / slzDecompressAsync (frame
 * device-resident, output device-resident, GPU scan path).
 *
 * Output format mirrors the CLI bencher so the bench_all.bat summary
 * parser can read both with the same regexes:
 *   best:            xx.xxx ms  (xxxx MB/s)
 *   gpu kernel best: xx.xxx ms  (xxxxx MB/s)
 *
 * "best" / "mean"             = submit-to-sync host wall-clock, what
 *                               the calling thread actually sees
 * "gpu kernel best" / "mean"  = cudaEventRecord pair around the entire
 *                               slzDecompressAsync work on the caller's
 *                               stream — the GPU-stream span between
 *                               the queued start and stop events,
 *                               including any idle gaps where the
 *                               stream sat waiting for the host to
 *                               queue the next kernel (e.g. the
 *                               walkMetaToHost D2H block)
 * "kernel active best" / "mean" = sum of per-kernel cudaEventElapsedTime
 *                               (via slzGetLastTimings) — strictly the
 *                               GPU's busy-time, excluding stream-idle
 *                               gaps. The delta vs `gpu kernel best`
 *                               reflects on-stream memcpy time + cuEvent
 *                               record overhead — work that's real but
 *                               not classified as a kernel.
 *
 * Usage:
 *   slz_gpu_d2d_bench <frame.slz> <source.bin> [--runs N]
 *
 * `source.bin` is the expected decompressed bytes (for SHA-style memcmp
 * verify; mismatch → exit 2). N defaults to 30.
 */

#include "streamlz_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

static double host_now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

static unsigned char* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_size = (size_t)sz;
    return buf;
}

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 3) {
        printf("usage: %s <frame.slz> <source.bin> [--runs N]\n", argv[0]);
        return 1;
    }
    const char* frame_path = argv[1];
    const char* src_path = argv[2];
    int runs = 30;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs < 1) runs = 1;
        }
    }

    size_t frame_size = 0, src_size = 0;
    unsigned char* host_frame = read_file(frame_path, &frame_size);
    if (!host_frame) { printf("open %s failed\n", frame_path); return 1; }
    unsigned char* host_src = read_file(src_path, &src_size);
    if (!host_src) { printf("open %s failed\n", src_path); free(host_frame); return 1; }

    /* The decompressed size is the source size; we know this from the
     * host buffer, so we don't need slzGetDecompressedSize here. (A real
     * caller would either know it from outer metadata like this, or call
     * slzGetDecompressedSize once to discover it.) */
    const size_t decomp_size = src_size;

    slzHandle_t h = NULL;
    if (slzCreate(&h) != SLZ_SUCCESS) { printf("slzCreate failed\n"); free(host_frame); free(host_src); return 1; }

    void *d_frame = NULL, *d_out = NULL;
    if (cudaMalloc(&d_frame, frame_size) != cudaSuccess ||
        cudaMalloc(&d_out, decomp_size) != cudaSuccess) {
        printf("cudaMalloc failed\n"); slzDestroy(h); free(host_frame); free(host_src); return 1;
    }
    cudaMemcpy(d_frame, host_frame, frame_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);

    slzDecompressOpts_t dopts = slzDecompressDefaultOpts();
    /* Enable profiling so slzGetLastTimings returns per-kernel cudaEvent
     * timings — we sum them to expose stream-idle gaps as the delta
     * between event-bracketed time and kernel-active time. */
    dopts.enable_profiling = 1;

    /* Warm-up run (mirrors -db which does +1 untimed iteration). */
    slzStatus_t s = slzDecompressAsync(h, d_frame, frame_size, d_out, decomp_size, dopts, stream);
    if (s != SLZ_SUCCESS) {
        printf("warm-up slzDecompressAsync failed: %s\n", slzStatusString(s));
        return 1;
    }
    cudaStreamSynchronize(stream);
    /* Drain the warm-up's pending events so they don't leak into the
     * first measured run's slzGetLastTimings. */
    {
        slzKernelTiming_t warm_timings[64];
        size_t warm_cnt = 0;
        slzGetLastTimings(h, warm_timings, 64, &warm_cnt);
    }

    double best_wall = 1e30, sum_wall = 0.0;
    float best_d2d = 1e30f, sum_d2d = 0.0f;
    double best_kern = 1e30, sum_kern = 0.0;
    /* Per-kernel breakdown captured from the run that produced
     * `best_kern` so the displayed breakdown's sum ties out exactly
     * to the reported `kernel active best`. */
    slzKernelTiming_t breakdown_timings[64];
    size_t breakdown_cnt = 0;

    for (int r = 0; r < runs; r++) {
        const double t0 = host_now_ms();
        cudaEventRecord(e_start, stream);
        s = slzDecompressAsync(h, d_frame, frame_size, d_out, decomp_size, dopts, stream);
        if (s != SLZ_SUCCESS) {
            printf("slzDecompressAsync run %d failed: %s\n", r, slzStatusString(s));
            return 1;
        }
        cudaEventRecord(e_stop, stream);
        cudaStreamSynchronize(stream);
        const double t1 = host_now_ms();

        float d2d_ms = 0.0f;
        cudaEventElapsedTime(&d2d_ms, e_start, e_stop);

        /* Pull this run's per-kernel timings. slzGetLastTimings drains
         * the pending event queue, so this also primes the handle for
         * the next iteration. */
        slzKernelTiming_t timings[64];
        size_t kcnt = 0;
        slzGetLastTimings(h, timings, 64, &kcnt);
        double kern_ms = 0.0;
        for (size_t i = 0; i < kcnt; i++) kern_ms += (double)timings[i].ms;

        const double wall_ms = t1 - t0;
        if (wall_ms < best_wall) best_wall = wall_ms;
        if (d2d_ms < best_d2d) best_d2d = d2d_ms;
        if (kern_ms < best_kern) {
            best_kern = kern_ms;
            /* Snapshot this run's per-kernel breakdown — the displayed
             * sum will tie out to `kernel active best`. */
            const size_t to_copy = kcnt < 64 ? kcnt : 64;
            for (size_t i = 0; i < to_copy; i++) breakdown_timings[i] = timings[i];
            breakdown_cnt = to_copy;
        }
        sum_wall += wall_ms;
        sum_d2d += d2d_ms;
        sum_kern += kern_ms;
    }

    /* SHA-style byte compare against the host source. */
    unsigned char* host_out = (unsigned char*)malloc(decomp_size);
    if (!host_out) { printf("host_out alloc failed\n"); return 1; }
    cudaMemcpy(host_out, d_out, decomp_size, cudaMemcpyDeviceToHost);
    const int verify_ok = memcmp(host_out, host_src, src_size) == 0;

    const double decomp_mb = (double)decomp_size / (1024.0 * 1024.0);
    const double wall_best_mb_s = decomp_mb / (best_wall / 1000.0);
    const double wall_mean_ms = sum_wall / (double)runs;
    const double wall_mean_mb_s = decomp_mb / (wall_mean_ms / 1000.0);
    const double d2d_best_mb_s = decomp_mb / ((double)best_d2d / 1000.0);
    const double d2d_mean_ms = (double)sum_d2d / (double)runs;
    const double d2d_mean_mb_s = decomp_mb / (d2d_mean_ms / 1000.0);
    const double kern_best_mb_s = decomp_mb / (best_kern / 1000.0);
    const double kern_mean_ms = sum_kern / (double)runs;
    const double kern_mean_mb_s = decomp_mb / (kern_mean_ms / 1000.0);
    const double gap_best = (double)best_d2d - best_kern;
    const double gap_mean = d2d_mean_ms - kern_mean_ms;

    /* Output format matches `streamlz -db` so the bench_all.bat summary
     * parser regexes still read best / mean / gpu kernel best/mean,
     * plus the additional kernel-active and gap lines. */
    printf("bench: %s\n", frame_path);
    printf("  src bytes:       %zu\n", frame_size);
    printf("  decompressed:    %zu (%.2f MB)\n", decomp_size, decomp_mb);
    printf("  runs:            %d (plus 1 warm-up)\n", runs);
    printf("  best:            %.3f ms  (%.0f MB/s)\n", best_wall, wall_best_mb_s);
    printf("  mean:            %.3f ms  (%.0f MB/s)\n", wall_mean_ms, wall_mean_mb_s);
    printf("  gpu kernel best: %.3f ms  (%.0f MB/s)\n", (double)best_d2d, d2d_best_mb_s);
    printf("  gpu kernel mean: %.3f ms  (%.0f MB/s)\n", d2d_mean_ms, d2d_mean_mb_s);
    printf("  kernel active best: %.3f ms  (%.0f MB/s)\n", best_kern, kern_best_mb_s);
    printf("  kernel active mean: %.3f ms  (%.0f MB/s)\n", kern_mean_ms, kern_mean_mb_s);
    printf("  stream-idle gap (D2D - kernel-active) best/mean: %.3f / %.3f ms\n", gap_best, gap_mean);
    printf("  verify:          %s\n", verify_ok ? "OK" : "FAIL");
    /* Per-kernel breakdown from the run that produced `kernel active
     * best` above — its sum equals that value. */
    if (breakdown_cnt > 0) {
        printf("  per-kernel breakdown (best-kern run):\n");
        double sum_check = 0.0;
        for (size_t i = 0; i < breakdown_cnt; i++) {
            printf("    %-40s %8.4f ms\n", breakdown_timings[i].name, breakdown_timings[i].ms);
            sum_check += (double)breakdown_timings[i].ms;
        }
        printf("    %-40s %8.4f ms\n", "(sum)", sum_check);
    }

    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    cudaStreamDestroy(stream);
    cudaFree(d_frame);
    cudaFree(d_out);
    free(host_out);
    free(host_frame);
    free(host_src);
    slzDestroy(h);

    return verify_ok ? 0 : 2;
}
