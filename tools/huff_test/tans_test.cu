// Standalone testbed for the GPU tANS-32 decode kernel.
//
// Loads a real snapshot captured from a live `streamlz -l5 -gpu` decode
// (SLZ_DUMP_TANS32=1 — see gpu_driver.zig), runs the production tans32
// decode path in isolation, verifies byte-exact against the production
// output, and benchmarks it. tANS analogue of huff_test.
//
// Production tans32 path is TWO kernels:
//   1. slzTansFseBuildKernel  — parse FSE prob table, build per-chunk LUT
//   2. slzTans32DecodeKernel  — 32-lane decode using the pre-built LUT
// (slzTans32FusedKernel's inline parseTable is the stale Golomb-Rice
//  parser and fails on current FSE-format tables.)
//
// Snapshot format (c:\tmp\tans32_snapshot.bin):
//   [u32 magic 'TN32'][u32 num_chunks][u32 comp_size][u32 scratch_size]
//   [num_chunks * TansDecChunkDesc (28 bytes)]
//   [comp_size bytes  — the compressed frame, = src_buf]
//   [scratch_size bytes — the production decode output, 128KB-strided]
//
// Build: build_tans.bat   (nvcc -O3 -arch=sm_89)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

// Brings in slzTansFseBuildKernel, slzTans32DecodeKernel, TansDecChunkDesc,
// TansLutEnt, TansTableMeta, TANS_OK, and all device helpers.
#include "tans_decode_kernel.cu"

#define CK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(1); \
    } \
} while (0)

static uint8_t* read_file(const char* path, size_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    (void)got;
    *out_size = (size_t)sz;
    return buf;
}

int main(int argc, char** argv) {
    const char* snap_path = (argc > 1) ? argv[1] : "c:/tmp/tans32_snapshot.bin";

    size_t file_size = 0;
    uint8_t* file = read_file(snap_path, &file_size);

    const uint32_t* hdr = (const uint32_t*)file;
    uint32_t magic    = hdr[0];
    uint32_t n_chunks = hdr[1];
    uint32_t comp_size = hdr[2];
    uint32_t scratch_size = hdr[3];
    if (magic != 0x544e3332u) { printf("bad magic %08x\n", magic); return 1; }

    const uint8_t* p = file + 16;
    TansDecChunkDesc* descs = (TansDecChunkDesc*)malloc(sizeof(TansDecChunkDesc) * n_chunks);
    memcpy(descs, p, sizeof(TansDecChunkDesc) * (size_t)n_chunks);
    p += (size_t)n_chunks * sizeof(TansDecChunkDesc);
    const uint8_t* comp = p;             p += comp_size;
    const uint8_t* scratch = p;          // p += scratch_size;

    printf("snapshot: %u chunks, comp %.1f MB, scratch %.1f MB\n",
           n_chunks, comp_size / 1e6, scratch_size / 1e6);

    // Compact the 128KB-strided production output into a contiguous
    // reference, and rewrite each desc's dst_offset to match.
    uint64_t total_dst = 0;
    for (uint32_t i = 0; i < n_chunks; i++) total_dst += descs[i].dst_size;
    uint8_t* ref = (uint8_t*)malloc(total_dst);
    uint64_t acc = 0;
    for (uint32_t i = 0; i < n_chunks; i++) {
        uint32_t dsz = descs[i].dst_size;
        memcpy(ref + acc, scratch + descs[i].dst_offset, dsz);
        descs[i].dst_offset   = (uint32_t)acc;
        descs[i].dst_offset_b = (uint32_t)acc;
        descs[i].split_count  = dsz;
        acc += dsz;
    }
    printf("decoded total %.1f MB\n", total_dst / 1e6);

    // ── Device setup ──
    uint8_t *d_comp, *d_out;
    TansDecChunkDesc* d_descs;
    uint32_t* d_status;
    TansLutEnt* d_lut;
    TansTableMeta* d_meta;
    uint32_t* d_work;
    size_t lut_bytes  = (size_t)n_chunks * 2048 * sizeof(TansLutEnt);
    size_t meta_bytes = (size_t)n_chunks * sizeof(TansTableMeta);
    CK(cudaMalloc(&d_comp, comp_size + 16));  // +16: readLE32_aligned over-reads up to 7B
    CK(cudaMalloc(&d_out, total_dst));
    CK(cudaMalloc(&d_descs, sizeof(TansDecChunkDesc) * (size_t)n_chunks));
    CK(cudaMalloc(&d_status, sizeof(uint32_t) * (size_t)n_chunks));
    CK(cudaMalloc(&d_lut, lut_bytes));
    CK(cudaMalloc(&d_meta, meta_bytes));
    CK(cudaMalloc(&d_work, 4));
    CK(cudaMemcpy(d_comp, comp, comp_size, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs, sizeof(TansDecChunkDesc) * (size_t)n_chunks, cudaMemcpyHostToDevice));
    printf("LUT buf %.1f MB, meta buf %.2f MB\n", lut_bytes / 1e6, meta_bytes / 1e6);

    dim3 block(32, 2, 1);
    dim3 grid((n_chunks + 1) / 2, 1, 1);
    uint32_t fse_grid = (n_chunks + 1) / 2;
    if (fse_grid > 1024) fse_grid = 1024;

    // ── FSE table build (prerequisite — LUTs persist after this) ──
    CK(cudaMemset(d_work, 0, 4));
    slzTansFseBuildKernel<<<fse_grid, block>>>(d_comp, d_descs, d_lut, d_meta, n_chunks, d_work);
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { printf("FSE build launch error: %s\n", cudaGetErrorString(le)); return 1; }
    CK(cudaDeviceSynchronize());

    // ── tANS-32 decode: warmup + correctness ──
    CK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
    CK(cudaMemset(d_out, 0, total_dst));
    slzTans32DecodeKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
    le = cudaGetLastError();
    if (le != cudaSuccess) { printf("decode launch error: %s\n", cudaGetErrorString(le)); return 1; }
    CK(cudaDeviceSynchronize());

    uint8_t* gpu_out = (uint8_t*)malloc(total_dst);
    uint32_t* status = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)n_chunks);
    CK(cudaMemcpy(gpu_out, d_out, total_dst, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(status, d_status, sizeof(uint32_t) * (size_t)n_chunks, cudaMemcpyDeviceToHost));

    int status_bad = 0, mismatch_chunks = 0;
    uint64_t off = 0;
    for (uint32_t i = 0; i < n_chunks; i++) {
        if (status[i] != TANS_OK) {
            if (status_bad < 5) printf("  chunk %u status=%u\n", i, status[i]);
            status_bad++;
        }
        if (memcmp(gpu_out + off, ref + off, descs[i].dst_size) != 0) {
            if (mismatch_chunks < 5) {
                uint32_t j = 0;
                while (j < descs[i].dst_size && gpu_out[off+j] == ref[off+j]) j++;
                printf("  chunk %u mismatch at byte %u/%u: got %02x exp %02x\n",
                       i, j, descs[i].dst_size, gpu_out[off+j], ref[off+j]);
            }
            mismatch_chunks++;
        }
        off += descs[i].dst_size;
    }
    printf("verify: %d/%u status errors, %d/%u chunk mismatches%s\n",
           status_bad, n_chunks, mismatch_chunks, n_chunks,
           (status_bad == 0 && mismatch_chunks == 0) ? "  — BYTE-EXACT" : "");

    // ── Benchmark the decode kernel (LUTs already built) ──
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    const int N_RUNS = 50;
    float best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        slzTans32DecodeKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }
    double gbps = (double)total_dst / (best_ms * 1e6);
    printf("[tans32-decode] %u chunks, %.1f MB decoded -> best %.3f ms = %.1f GB/s\n",
           n_chunks, total_dst / 1e6, best_ms, gbps);

    // ── Variant: shared-memory LUT decode ──
    CK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
    CK(cudaMemset(d_out, 0, total_dst));
    slzTans32DecodeSharedLutKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
    le = cudaGetLastError();
    if (le != cudaSuccess) { printf("sharedlut launch error: %s\n", cudaGetErrorString(le)); return 1; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gpu_out, d_out, total_dst, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(status, d_status, sizeof(uint32_t) * (size_t)n_chunks, cudaMemcpyDeviceToHost));
    {
        int sb = 0, mm = 0;
        uint64_t o = 0;
        for (uint32_t i = 0; i < n_chunks; i++) {
            if (status[i] != TANS_OK) sb++;
            if (memcmp(gpu_out + o, ref + o, descs[i].dst_size) != 0) mm++;
            o += descs[i].dst_size;
        }
        printf("verify (shared-LUT): %d status errors, %d chunk mismatches%s\n",
               sb, mm, (sb == 0 && mm == 0) ? "  — BYTE-EXACT" : "");
    }
    best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        slzTans32DecodeSharedLutKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }
    printf("[tans32-sharedlut] %u chunks, %.1f MB decoded -> best %.3f ms = %.1f GB/s\n",
           n_chunks, total_dst / 1e6, best_ms, (double)total_dst / (best_ms * 1e6));

    // ── Variant: shared LUT + aligned readLE32 refill ──
    CK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
    CK(cudaMemset(d_out, 0, total_dst));
    slzTans32DecodeSharedLutAlignedKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
    le = cudaGetLastError();
    if (le != cudaSuccess) { printf("aligned launch error: %s\n", cudaGetErrorString(le)); return 1; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gpu_out, d_out, total_dst, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(status, d_status, sizeof(uint32_t) * (size_t)n_chunks, cudaMemcpyDeviceToHost));
    {
        int sb = 0, mm = 0;
        uint64_t o = 0;
        for (uint32_t i = 0; i < n_chunks; i++) {
            if (status[i] != TANS_OK) sb++;
            if (memcmp(gpu_out + o, ref + o, descs[i].dst_size) != 0) mm++;
            o += descs[i].dst_size;
        }
        printf("verify (shared-LUT+aligned): %d status errors, %d chunk mismatches%s\n",
               sb, mm, (sb == 0 && mm == 0) ? "  — BYTE-EXACT" : "");
    }
    best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        slzTans32DecodeSharedLutAlignedKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }
    printf("[tans32-sl+aligned] %u chunks, %.1f MB decoded -> best %.3f ms = %.1f GB/s\n",
           n_chunks, total_dst / 1e6, best_ms, (double)total_dst / (best_ms * 1e6));

    // ── Variant: shared LUT + aligned refill + sub-streams staged in shared ──
    CK(cudaMemset(d_status, 0xFF, sizeof(uint32_t) * (size_t)n_chunks));
    CK(cudaMemset(d_out, 0, total_dst));
    slzTans32DecodeStagedKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
    le = cudaGetLastError();
    if (le != cudaSuccess) { printf("staged launch error: %s\n", cudaGetErrorString(le)); return 1; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(gpu_out, d_out, total_dst, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(status, d_status, sizeof(uint32_t) * (size_t)n_chunks, cudaMemcpyDeviceToHost));
    {
        int sb = 0, mm = 0;
        uint64_t o = 0;
        for (uint32_t i = 0; i < n_chunks; i++) {
            if (status[i] != TANS_OK) sb++;
            if (memcmp(gpu_out + o, ref + o, descs[i].dst_size) != 0) mm++;
            o += descs[i].dst_size;
        }
        printf("verify (staged): %d status errors, %d chunk mismatches%s\n",
               sb, mm, (sb == 0 && mm == 0) ? "  — BYTE-EXACT" : "");
    }
    best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        cudaEventRecord(e0);
        slzTans32DecodeStagedKernel<<<grid, block>>>(d_comp, d_out, d_descs, d_status, n_chunks, d_lut, d_meta, 0u);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }
    printf("[tans32-staged] %u chunks, %.1f MB decoded -> best %.3f ms = %.1f GB/s\n",
           n_chunks, total_dst / 1e6, best_ms, (double)total_dst / (best_ms * 1e6));

    // Also time the FSE build kernel for reference.
    best_ms = 1e9f;
    for (int r = 0; r < N_RUNS; r++) {
        CK(cudaMemset(d_work, 0, 4));
        cudaEventRecord(e0);
        slzTansFseBuildKernel<<<fse_grid, block>>>(d_comp, d_descs, d_lut, d_meta, n_chunks, d_work);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }
    printf("[fse-build]     %u chunks -> best %.3f ms\n", n_chunks, best_ms);

    cudaFree(d_comp); cudaFree(d_out); cudaFree(d_descs); cudaFree(d_status);
    cudaFree(d_lut); cudaFree(d_meta); cudaFree(d_work);
    free(file); free(descs); free(ref); free(gpu_out); free(status);
    return 0;
}
