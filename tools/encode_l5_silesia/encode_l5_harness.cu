// ── StreamLZ GPU L5 encode harness — silesia ─────────────────────
//
// Replicates production's GPU L5 encode pipeline for silesia_all.tar
// using ONLY local header copies under include/. Never #includes
// anything from src/.
//
// Production reference (RTX 4060 Ti, baseline 2026-05-27):
//   $ streamlz.exe -c -l 5 -gpu assets/silesia_all.tar -o c:/tmp/sil_l5_prod.slz
//     compressed 212797440 -> 72084526 bytes  (33.9%)  L5
//     GPU kernel: ~4220 ms (~48 MB/s)
//
// The harness measures only the LZ encode kernel (slzLzEncodeKernel).
// Production's "GPU kernel: Xms" print is the same kernel — so we
// compare wall-clock cuEvent ms directly against ~4220 ms.
//
// Chunking (matches src/encode/fast_framed.zig gpu_compress block):
//   sc_grp        = 0.25 (hard-coded for -gpu)
//   eff_chunk     = scGroupSizeToBytes(0.25) = 0.25 * 256KB = 64KB
//   gpu_block     = min(eff_chunk, sub_chunk_size=128KB) = 64KB
//   n_descriptors = ceil(src.len / 64KB), 1 sub-chunk per chunk
//   per_block_cap = gpu_block * 3 = 192KB
//
// Kernel params for L5 (matches src/gpu/encode/{levels,encode_lz}.zig):
//   hash_bits   = 20  (hashBitsForLevel(5))
//   use_chain   = 1   (useChainParser(5) = level >= 5)
//   l4_features = 0   (only set for level >= 4 in greedy mode; chain ignores it)
//
// Build: tools/encode_l5_silesia/build_encode_l5_harness.bat
// Run:   tools/encode_l5_silesia/encode_l5_harness.exe assets/silesia_all.tar

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// Local-only includes — see [[harness-standalone]]
#include "include/lz_kernel.cu"   // includes lz_format/token_emit/greedy/chain transitively

#define CK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(1); \
    } \
} while (0)

// Mirrors src/gpu/encode/encode_context.zig:CompressChunkDesc
struct HostCompressChunkDesc {
    uint32_t src_offset;
    uint32_t src_size;
    uint32_t dst_offset;
    uint32_t dst_capacity;
    uint32_t is_first;
};
static_assert(sizeof(HostCompressChunkDesc) == 20, "ABI");

// Constants mirroring production
static constexpr uint32_t CHUNK_SIZE_BYTES    = 0x40000;  // 256 KB (streamlz_constants.zig)
static constexpr uint32_t SUB_CHUNK_SIZE_BYTES= 0x20000;  // 128 KB
static constexpr float    SC_GROUP_GPU        = 0.25f;    // hard-coded for -gpu
static constexpr uint32_t NEXT_HASH_WORDS     = 32768;    // 65536 u16 entries / 2 (u32 words)

static std::vector<uint8_t> read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if ((long)fread(v.data(), 1, n, f) != n) { printf("read truncated\n"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "assets/silesia_all.tar";
    int runs = (argc > 2) ? atoi(argv[2]) : 5;
    int level = (argc > 3) ? atoi(argv[3]) : 5;
    // Optional: cap input bytes (for NCU profiling on a subset)
    size_t max_bytes = (argc > 4) ? (size_t)atoll(argv[4]) : (size_t)-1;
    // Optional: override hash_bits (default = level policy)
    int hash_bits_override = (argc > 5) ? atoi(argv[5]) : -1;
    // Optional: force parser (0=greedy, 1=chain, -1=level policy)
    int parser_override = (argc > 6) ? atoi(argv[6]) : -1;
    // Optional: sc_group_size (default 0.25 matches production -gpu)
    float sc_arg = (argc > 7) ? atof(argv[7]) : SC_GROUP_GPU;

    printf("== encode_l5_harness ==\n");
    printf("input: %s\n", path);
    int default_hb = (level == 5 || level == 4) ? 20 : (level == 3) ? 19 : (level == 2) ? 18 : 17;
    int hb_actual = (hash_bits_override > 0) ? hash_bits_override : default_hb;
    printf("level: %d (chain=%d, hash_bits=%d%s)\n",
           level, (level >= 5) ? 1 : 0, hb_actual,
           (hash_bits_override > 0) ? " [OVERRIDE]" : "");
    printf("runs:  %d\n", runs);

    auto host_src = read_file(path);
    if (max_bytes < host_src.size()) {
        host_src.resize(max_bytes);
        printf("(truncated input to %zu bytes for profiling)\n", max_bytes);
    }
    printf("src:   %zu bytes (%.2f MB)\n", host_src.size(), host_src.size() / (1024.0*1024.0));

    // Chunking (mirrors fast_framed.zig:1119-1131). sc_arg can override
    // production's 0.25 default to test larger match windows.
    printf("sc:    %.3f\n", sc_arg);
    uint32_t eff_chunk = (uint32_t)((double)sc_arg * CHUNK_SIZE_BYTES);
    if (eff_chunk > CHUNK_SIZE_BYTES) eff_chunk = CHUNK_SIZE_BYTES;
    uint32_t gpu_block = (eff_chunk < SUB_CHUNK_SIZE_BYTES) ? eff_chunk : SUB_CHUNK_SIZE_BYTES;
    uint32_t per_block_cap = gpu_block * 3;

    size_t n_chunks_outer = (host_src.size() + eff_chunk - 1) / eff_chunk;
    size_t n_descs = 0;
    for (size_t ci = 0; ci < n_chunks_outer; ++ci) {
        size_t chunk_size = std::min<size_t>(eff_chunk, host_src.size() - ci * eff_chunk);
        n_descs += (chunk_size + gpu_block - 1) / gpu_block;
    }
    printf("chunks: %zu outer, %zu descriptors (1 sub-chunk per outer chunk)\n",
           n_chunks_outer, n_descs);
    printf("per-block dst cap: %u bytes\n", per_block_cap);

    std::vector<HostCompressChunkDesc> descs(n_descs);
    size_t bi = 0;
    for (size_t ci = 0; ci < n_chunks_outer; ++ci) {
        size_t chunk_start = ci * eff_chunk;
        size_t chunk_size  = std::min<size_t>(eff_chunk, host_src.size() - chunk_start);
        size_t n_subs      = (chunk_size + gpu_block - 1) / gpu_block;
        for (size_t si = 0; si < n_subs; ++si) {
            size_t sub_start = chunk_start + si * gpu_block;
            size_t sub_size  = std::min<size_t>(gpu_block, host_src.size() - sub_start);
            descs[bi].src_offset   = (uint32_t)sub_start;
            descs[bi].src_size     = (uint32_t)sub_size;
            descs[bi].dst_offset   = (uint32_t)(bi * per_block_cap);
            descs[bi].dst_capacity = per_block_cap;
            descs[bi].is_first     = (bi == 0) ? 1u : 0u;
            ++bi;
        }
    }

    // Compute allocation sizes
    uint32_t hash_bits = (uint32_t)hb_actual;
    bool use_chain = (parser_override >= 0) ? (parser_override != 0) : (level >= 5);
    if (parser_override >= 0) {
        printf("parser OVERRIDE: %s\n", use_chain ? "chain" : "greedy");
    }
    // levels.useGlobalHash always true on production
    size_t hash_size = (size_t)1 << hash_bits;
    size_t per_block_hash_words = use_chain
        ? (hash_size + hash_size + NEXT_HASH_WORDS)
        : hash_size;
    size_t hash_bytes = n_descs * per_block_hash_words * 4;
    size_t output_bytes = n_descs * per_block_cap;
    size_t sizes_bytes = n_descs * 4;
    size_t desc_bytes  = n_descs * sizeof(HostCompressChunkDesc);

    // Free memory probe
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    printf("GPU:   free %.2f GB / total %.2f GB\n",
           free_b/(1024.0*1024.0*1024.0), total_b/(1024.0*1024.0*1024.0));
    printf("alloc plan:\n");
    printf("  d_input  : %.2f MB\n", host_src.size() / (1024.0*1024.0));
    printf("  d_output : %.2f MB (n_descs %zu * per_block_cap %u)\n",
           output_bytes / (1024.0*1024.0), n_descs, per_block_cap);
    printf("  d_descs  : %.2f KB\n", desc_bytes / 1024.0);
    printf("  d_hash   : %.2f MB (per-block %.2f MB %s)\n",
           hash_bytes / (1024.0*1024.0),
           per_block_hash_words * 4.0 / (1024.0*1024.0),
           use_chain ? "[chain: first+long+next]" : "[greedy: single table]");
    printf("  d_sizes  : %.2f KB\n", sizes_bytes / 1024.0);
    printf("  TOTAL    : %.2f GB\n",
           (host_src.size() + output_bytes + desc_bytes + hash_bytes + sizes_bytes)
           / (1024.0*1024.0*1024.0));

    if (hash_bytes > free_b) {
        printf("\n*** WARNING: hash_bytes (%.2f GB) > free GPU memory (%.2f GB) ***\n",
               hash_bytes/(1024.0*1024.0*1024.0), free_b/(1024.0*1024.0*1024.0));
        printf("    Production would also need this; if it ran, WDDM may be falling back\n");
        printf("    to system memory via PCIe. Continuing.\n");
    }

    // Allocate
    uint8_t  *d_input  = nullptr;
    uint8_t  *d_output = nullptr;
    HostCompressChunkDesc *d_descs = nullptr;
    uint32_t *d_hash   = nullptr;
    uint32_t *d_sizes  = nullptr;

    CK(cudaMalloc((void**)&d_input,  host_src.size()));
    CK(cudaMalloc((void**)&d_output, output_bytes));
    CK(cudaMalloc((void**)&d_descs,  desc_bytes));
    CK(cudaMalloc((void**)&d_sizes,  sizes_bytes));
    CK(cudaMalloc((void**)&d_hash,   hash_bytes));

    // H2D
    CK(cudaMemcpy(d_input, host_src.data(), host_src.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs, descs.data(),    desc_bytes,      cudaMemcpyHostToDevice));

    uint32_t l4_features = (level >= 4) ? 1u : 0u;
    uint32_t use_chain_u = use_chain ? 1u : 0u;

    // CUDA events for kernel-only timing
    cudaEvent_t e_start, e_stop;
    CK(cudaEventCreate(&e_start));
    CK(cudaEventCreate(&e_stop));

    // Warmup
    printf("\nwarmup launch ...\n");
    CK(cudaMemset(d_sizes, 0, sizes_bytes));
    slzLzEncodeKernel<<<(unsigned)n_descs, 32, 0, 0>>>(
        d_input,
        d_output,
        (const CompressChunkDesc*)d_descs,
        d_hash,
        d_sizes,
        (uint32_t)n_descs,
        hash_bits,
        use_chain_u,
        l4_features
    );
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) { printf("launch err: %s\n", cudaGetErrorString(le)); exit(1); }
    CK(cudaDeviceSynchronize());

    // Pull comp_sizes once to verify
    std::vector<uint32_t> comp_sizes(n_descs);
    CK(cudaMemcpy(comp_sizes.data(), d_sizes, sizes_bytes, cudaMemcpyDeviceToHost));
    uint64_t lz_total = 0;
    uint32_t cs_min = 0xFFFFFFFFu, cs_max = 0;
    for (auto s : comp_sizes) { lz_total += s; if (s < cs_min) cs_min = s; if (s > cs_max) cs_max = s; }
    printf("warmup: lz_total=%llu bytes (%.2f MB), per-block min=%u max=%u avg=%llu\n",
           (unsigned long long)lz_total, lz_total/(1024.0*1024.0),
           cs_min, cs_max, (unsigned long long)(lz_total / n_descs));

    // Fingerprint: pull all LZ kernel output bytes back and FNV-1a hash them.
    // Two harness runs (or harness vs production with same kernel) must
    // produce the same fingerprint — kernel is deterministic. If the hash
    // changes after a kernel edit, output has drifted from baseline.
    std::vector<uint8_t> host_out_compact;
    host_out_compact.reserve(lz_total);
    {
        // Pull each block's compressed bytes into a packed host buffer.
        for (size_t i = 0; i < n_descs; ++i) {
            uint32_t cs = comp_sizes[i];
            if (cs == 0) continue;
            size_t old = host_out_compact.size();
            host_out_compact.resize(old + cs);
            CK(cudaMemcpy(host_out_compact.data() + old,
                          d_output + descs[i].dst_offset,
                          cs, cudaMemcpyDeviceToHost));
        }
    }
    // FNV-1a 64-bit
    uint64_t fnv = 0xcbf29ce484222325ULL;
    for (uint8_t b : host_out_compact) {
        fnv ^= b;
        fnv *= 0x100000001b3ULL;
    }
    // Hash the comp_sizes array too — catches per-block size drift
    // independent of byte order
    uint64_t sz_fnv = 0xcbf29ce484222325ULL;
    for (uint32_t s : comp_sizes) {
        for (int k = 0; k < 4; ++k) { sz_fnv ^= (s >> (k*8)) & 0xff; sz_fnv *= 0x100000001b3ULL; }
    }
    printf("fingerprint: bytes_fnv=0x%016llx sizes_fnv=0x%016llx\n",
           (unsigned long long)fnv, (unsigned long long)sz_fnv);

    // Timed runs — match production timer methodology (encode_lz.zig:121-138):
    // wall-clock around launch + post-launch cuCtxSynchronize. This includes
    // CPU launch overhead so harness ms is directly comparable to production's
    // "GPU kernel: Xms" line.
    printf("\ntimed runs (wall-clock around launch+sync, matches production):\n");
    double best_ms = 1e30, sum_ms = 0;
    for (int r = 0; r < runs; ++r) {
        CK(cudaMemset(d_sizes, 0, sizes_bytes));
        CK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        slzLzEncodeKernel<<<(unsigned)n_descs, 32, 0, 0>>>(
            d_input,
            d_output,
            (const CompressChunkDesc*)d_descs,
            d_hash,
            d_sizes,
            (uint32_t)n_descs,
            hash_bits,
            use_chain_u,
            l4_features
        );
        CK(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // Also capture pure GPU event time for comparison
        CK(cudaMemset(d_sizes, 0, sizes_bytes));
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e_start));
        slzLzEncodeKernel<<<(unsigned)n_descs, 32, 0, 0>>>(
            d_input,
            d_output,
            (const CompressChunkDesc*)d_descs,
            d_hash,
            d_sizes,
            (uint32_t)n_descs,
            hash_bits,
            use_chain_u,
            l4_features
        );
        CK(cudaEventRecord(e_stop));
        CK(cudaEventSynchronize(e_stop));
        float evt_ms = 0;
        CK(cudaEventElapsedTime(&evt_ms, e_start, e_stop));
        double mbps = host_src.size() / (1024.0*1024.0) / (ms / 1000.0);
        printf("  run %d: wall %.1f ms (%.0f MB/s)  gpu-event %.1f ms\n",
               r+1, ms, mbps, evt_ms);
        if (ms < best_ms) best_ms = ms;
        sum_ms += ms;
    }
    double mean_ms = sum_ms / runs;
    double best_mbps = host_src.size() / (1024.0*1024.0) / (best_ms / 1000.0);
    double mean_mbps = host_src.size() / (1024.0*1024.0) / (mean_ms / 1000.0);
    printf("wall best: %.1f ms (%.0f MB/s)\n", best_ms, best_mbps);
    printf("wall mean: %.1f ms (%.0f MB/s)\n", mean_ms, mean_mbps);
    printf("(production baseline 2026-05-27 RTX 4060 Ti: ~4215-4506 ms wall, mean ~4290 ms)\n");

    // Cleanup
    CK(cudaFree(d_input));
    CK(cudaFree(d_output));
    CK(cudaFree(d_descs));
    CK(cudaFree(d_sizes));
    CK(cudaFree(d_hash));
    CK(cudaEventDestroy(e_start));
    CK(cudaEventDestroy(e_stop));

    return 0;
}
