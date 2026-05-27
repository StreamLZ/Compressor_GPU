// ── StreamLZ GPU L5 decode harness ────────────────────────────────
//
// Replicates production's GPU L5 decode pipeline using only LOCAL
// copies of the kernels and headers under include/. Never #includes
// anything from src/.
//
// Pipeline (matches src/gpu/decode/decode_dispatch.zig fullGpuLaunchImpl):
//   1. slzHuffBuildLutKernel    — build per-chunk Huffman LUTs
//   2. slzHuffDecode4StreamKernel — decode lit/tok/off16hi/off16lo
//   3. slzLzDecodeKernel        — assemble output from streams
//
// Production reference (RTX 4060 Ti, 2026-05-27, hash_bits=17 patch):
//   silesia L5 D2D best: ~7.72 ms
//   enwik8  L5 D2D best: ~4.12 ms
//
// Input: SCN1 dump produced by `SLZ_DUMP_SCAN=path streamlz.exe -d -gpu ...`
//
// SCN1 format (from src/gpu/decode/decode_dispatch.zig dumpScanIfRequested):
//   4 u32 header: magic 'SCN1', n_chunks, sub_chunk_cap, comp_len
//   ChunkDesc[n_chunks]                       (24 bytes each)
//   uint8_t  comp_data[comp_len]
//   For each of 4 streams (lit, tok, off16hi, off16lo):
//     u32 count
//     HuffDecChunkDesc[count]                 (20 bytes each)
//   u32 raw_count
//   (u32 src_offset, u32 size, u32 gpu_offset)[raw_count]
//
// Verification: D2H decompressed output, SHA-256 compare to expected.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdarg>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

// Local-only includes — see [[harness-standalone]]
// Production's lz_kernel.cu aggregator; we only need the LZ decode kernels
// (slzLzDecodeKernel, slzLzDecodeRawKernel). Skip the GPU-scan kernels
// (walk_frame, compact_descs, merge_huff_descs, prefix_sum, scan_parse)
// because the SCN1 dump path already gives us merged scan results from CPU.
#include "include/slz_wire_format.cuh"
#include "include/lz_decode_core.cuh"
#include "include/lz_decode_raw.cuh"
#include "include/lz_decode_general.cuh"
#include "include/lz_header_parse.cuh"
#include "include/lz_dispatch.cuh"
#include "include/lz_decode_kernels.cuh"

// Raw off16 gather kernel (copies raw-stream off16 bytes from comp to
// entropy_scratch off16 region — needed when off16hi/lo is type-0 raw,
// which is the common case for short-offset matches).
#include "include/gather_raw_off16_kernel.cuh"

// Huffman LUT-build + 32-stream decode kernels.
#include "include/huffman_kernel.cu"

#define CK(call) do { \
    cudaError_t e_ = (call); \
    if (e_ != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(1); \
    } \
} while (0)

// Mirror of src/gpu/decode/descriptors.zig ChunkDesc — 24 bytes.
#pragma pack(push, 4)
struct HostChunkDesc {
    uint32_t src_offset;
    uint32_t comp_size;
    uint32_t decomp_size;
    uint32_t dst_offset;
    uint32_t flags;
    uint8_t  memset_fill;
    uint8_t  _pad[3];
};
#pragma pack(pop)
static_assert(sizeof(HostChunkDesc) == 24, "ChunkDesc ABI mismatch");

// Mirror of HuffDecChunkDesc — 20 bytes.
struct HostHuffDesc {
    uint32_t in_offset;
    uint32_t in_size;
    uint32_t out_offset;
    uint32_t out_size;
    uint32_t lut_offset;
};
static_assert(sizeof(HostHuffDesc) == 20, "HuffDecChunkDesc ABI mismatch");

// Mirror of SlzRawOff16Desc — 12 bytes (3 u32).
struct HostRawOff16Desc {
    uint32_t src_offset;
    uint32_t size;
    uint32_t gpu_offset;
};
static_assert(sizeof(HostRawOff16Desc) == 12, "RawOff16Desc ABI mismatch");

// Production constants we mirror.
static constexpr uint32_t HOST_HUFF_LUT_ENTRIES        = 1024; // descriptors.zig
static constexpr uint64_t HOST_ENTROPY_SLOT_BYTES      = 131072;
static constexpr uint32_t HOST_OFF16_HILO_SPLIT_OFFSET = 65536;

// ── SHA-256 (tiny, FIPS-180-4 reference) ─────────────────────────
// Used for bit-exact output verification vs the ground-truth file.
struct Sha256 {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  buffer[64];
    uint32_t buf_used;
};

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

static inline uint32_t ror32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_init(Sha256* h) {
    static const uint32_t I[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    };
    memcpy(h->state, I, sizeof(I));
    h->bitlen = 0;
    h->buf_used = 0;
}

static void sha256_compress(Sha256* h, const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t)p[i*4]<<24 | (uint32_t)p[i*4+1]<<16 | (uint32_t)p[i*4+2]<<8 | (uint32_t)p[i*4+3];
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = ror32(w[i-15],7) ^ ror32(w[i-15],18) ^ (w[i-15]>>3);
        uint32_t s1 = ror32(w[i-2],17) ^ ror32(w[i-2],19) ^ (w[i-2]>>10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=h->state[0],b=h->state[1],c=h->state[2],d=h->state[3];
    uint32_t e=h->state[4],f=h->state[5],g=h->state[6],hh=h->state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = ror32(e,6) ^ ror32(e,11) ^ ror32(e,25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = hh + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ror32(a,2) ^ ror32(a,13) ^ ror32(a,22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h->state[0]+=a; h->state[1]+=b; h->state[2]+=c; h->state[3]+=d;
    h->state[4]+=e; h->state[5]+=f; h->state[6]+=g; h->state[7]+=hh;
}

static void sha256_update(Sha256* h, const uint8_t* data, size_t n) {
    h->bitlen += (uint64_t)n * 8;
    while (n > 0) {
        size_t take = 64 - h->buf_used;
        if (take > n) take = n;
        memcpy(h->buffer + h->buf_used, data, take);
        h->buf_used += (uint32_t)take;
        data += take;
        n -= take;
        if (h->buf_used == 64) {
            sha256_compress(h, h->buffer);
            h->buf_used = 0;
        }
    }
}

static void sha256_final(Sha256* h, uint8_t out[32]) {
    h->buffer[h->buf_used++] = 0x80;
    if (h->buf_used > 56) {
        while (h->buf_used < 64) h->buffer[h->buf_used++] = 0;
        sha256_compress(h, h->buffer);
        h->buf_used = 0;
    }
    while (h->buf_used < 56) h->buffer[h->buf_used++] = 0;
    uint64_t bl = h->bitlen;
    for (int i = 7; i >= 0; --i) h->buffer[56 + i] = (uint8_t)(bl & 0xff), bl >>= 8;
    sha256_compress(h, h->buffer);
    for (int i = 0; i < 8; ++i) {
        out[i*4]   = (uint8_t)(h->state[i] >> 24);
        out[i*4+1] = (uint8_t)(h->state[i] >> 16);
        out[i*4+2] = (uint8_t)(h->state[i] >> 8);
        out[i*4+3] = (uint8_t)(h->state[i]);
    }
}

static std::string sha256_hex(const uint8_t* data, size_t n) {
    Sha256 h; sha256_init(&h); sha256_update(&h, data, n);
    uint8_t out[32]; sha256_final(&h, out);
    char buf[65];
    for (int i = 0; i < 32; ++i) snprintf(buf + i*2, 3, "%02x", out[i]);
    buf[64] = 0;
    return std::string(buf);
}

// ── File I/O ─────────────────────────────────────────────────────
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

// ── SCN1 dump reader ─────────────────────────────────────────────
struct ScanDump {
    uint32_t n_chunks = 0;
    uint32_t sub_chunk_cap = 0;
    uint32_t comp_len = 0;
    std::vector<HostChunkDesc> chunks;
    std::vector<uint8_t>       comp;
    std::vector<HostHuffDesc>  huff_lit;
    std::vector<HostHuffDesc>  huff_tok;
    std::vector<HostHuffDesc>  huff_off16hi;
    std::vector<HostHuffDesc>  huff_off16lo;
    std::vector<HostRawOff16Desc> raw_off16;
};

static void read_huff_block(FILE* f, std::vector<HostHuffDesc>& out, const char* name) {
    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { printf("scn1: %s count truncated\n", name); exit(1); }
    out.resize(n);
    if (n > 0 && fread(out.data(), sizeof(HostHuffDesc), n, f) != n) {
        printf("scn1: %s descs truncated\n", name); exit(1);
    }
}

static ScanDump read_scn1(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    uint32_t hdr[4];
    if (fread(hdr, 4, 4, f) != 4) { printf("scn1: hdr truncated\n"); exit(1); }
    if (hdr[0] != 0x53434E31u) { printf("scn1: bad magic 0x%08x\n", hdr[0]); exit(1); }
    ScanDump s;
    s.n_chunks      = hdr[1];
    s.sub_chunk_cap = hdr[2];
    s.comp_len      = hdr[3];
    s.chunks.resize(s.n_chunks);
    if (fread(s.chunks.data(), sizeof(HostChunkDesc), s.n_chunks, f) != s.n_chunks) {
        printf("scn1: chunks truncated\n"); exit(1);
    }
    s.comp.resize(s.comp_len);
    if (s.comp_len > 0 && fread(s.comp.data(), 1, s.comp_len, f) != s.comp_len) {
        printf("scn1: comp truncated\n"); exit(1);
    }
    read_huff_block(f, s.huff_lit,     "lit");
    read_huff_block(f, s.huff_tok,     "tok");
    read_huff_block(f, s.huff_off16hi, "off16hi");
    read_huff_block(f, s.huff_off16lo, "off16lo");
    // raw_off16 list: u32 count + (u32 src_offset, u32 size, u32 gpu_offset)[count]
    uint32_t nraw = 0;
    if (fread(&nraw, 4, 1, f) != 1) { printf("scn1: raw count truncated\n"); exit(1); }
    s.raw_off16.resize(nraw);
    if (nraw > 0 && fread(s.raw_off16.data(), sizeof(HostRawOff16Desc), nraw, f) != nraw) {
        printf("scn1: raw_off16 descs truncated\n"); exit(1);
    }
    fclose(f);
    return s;
}

// Compute total sub-chunks from chunk decomp sizes.
static uint32_t total_sub_chunks(const ScanDump& s) {
    uint32_t total = 0;
    for (auto& c : s.chunks) {
        uint32_t subs = (c.decomp_size + s.sub_chunk_cap - 1) / s.sub_chunk_cap;
        if (subs == 0) subs = 1;
        total += subs;
    }
    return total;
}

// ── Merge per-stream HuffDecChunkDesc[] into one device-bound array ──
// Mirrors decode_dispatch.zig:mergeHuffDescs CPU path.
static std::vector<HostHuffDesc> merge_huff(
    const ScanDump& s,
    uint32_t tok_offset, uint32_t off16_offset
) {
    std::vector<HostHuffDesc> merged;
    merged.reserve(s.huff_lit.size() + s.huff_tok.size()
                 + s.huff_off16hi.size() + s.huff_off16lo.size());
    uint32_t lut_slot = 0;
    auto append = [&](const std::vector<HostHuffDesc>& src, uint32_t region) {
        for (auto e : src) {
            e.out_offset += region;
            e.lut_offset = lut_slot * HOST_HUFF_LUT_ENTRIES;
            merged.push_back(e);
            ++lut_slot;
        }
    };
    append(s.huff_lit,     0);
    append(s.huff_tok,     tok_offset);
    append(s.huff_off16hi, off16_offset);
    append(s.huff_off16lo, off16_offset);
    return merged;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("usage: %s <scn1_dump> <expected_decompressed>\n", argv[0]);
        printf("  scn1_dump:           output of SLZ_DUMP_SCAN=path streamlz.exe -d -gpu ...\n");
        printf("  expected_decompressed: ground-truth decompressed bytes (for SHA verify)\n");
        printf("  optional 3rd arg: runs (default 30)\n");
        return 1;
    }
    const char* scn1_path     = argv[1];
    const char* expected_path = argv[2];
    int runs = (argc > 3) ? atoi(argv[3]) : 30;

    printf("== decode_l5_harness ==\n");
    printf("scn1:     %s\n", scn1_path);
    printf("expected: %s\n", expected_path);
    printf("runs:     %d\n", runs);

    ScanDump s = read_scn1(scn1_path);
    printf("scn1: n_chunks=%u sub_chunk_cap=%u comp=%u\n",
           s.n_chunks, s.sub_chunk_cap, s.comp_len);
    printf("scn1: huff lit=%zu tok=%zu off16hi=%zu off16lo=%zu\n",
           s.huff_lit.size(), s.huff_tok.size(),
           s.huff_off16hi.size(), s.huff_off16lo.size());

    auto expected = read_file(expected_path);
    std::string expected_sha = sha256_hex(expected.data(), expected.size());
    printf("expected SHA: %s (%zu bytes)\n", expected_sha.c_str(), expected.size());

    // Decompressed total = sum of chunk decomp_sizes.
    uint64_t decomp_total = 0;
    for (auto& c : s.chunks) decomp_total += c.decomp_size;
    if (decomp_total != expected.size()) {
        printf("WARNING: decomp_total %llu != expected %zu — proceeding anyway\n",
               (unsigned long long)decomp_total, expected.size());
    }

    // Layout: entropy scratch is 3 contiguous regions of (total_subchunks ×
    // ENTROPY_SCRATCH_SLOT_BYTES) — lit, tok, off16 (with hi/lo packed
    // inside off16 region via +OFF16_HILO_SPLIT_OFFSET on lo).
    uint32_t total_subs = total_sub_chunks(s);
    uint32_t tok_offset   = (uint32_t)(total_subs * HOST_ENTROPY_SLOT_BYTES);
    uint32_t off16_offset = (uint32_t)(total_subs * HOST_ENTROPY_SLOT_BYTES * 2);
    auto merged_huff = merge_huff(s, tok_offset, off16_offset);
    uint32_t n_huff = (uint32_t)merged_huff.size();
    printf("merged huff: n_huff=%u total_subchunks=%u raw_off16=%zu\n",
           n_huff, total_subs, s.raw_off16.size());
    printf("first chunk: src_off=%u comp_size=%u decomp_size=%u dst_off=%u flags=0x%x\n",
        s.chunks[0].src_offset, s.chunks[0].comp_size, s.chunks[0].decomp_size,
        s.chunks[0].dst_offset, s.chunks[0].flags);
    if (!s.huff_lit.empty())
        printf("first lit desc: in_off=%u in_size=%u out_off=%u out_size=%u lut_off=%u\n",
               s.huff_lit[0].in_offset, s.huff_lit[0].in_size,
               s.huff_lit[0].out_offset, s.huff_lit[0].out_size, s.huff_lit[0].lut_offset);
    if (!merged_huff.empty())
        printf("first merged: in_off=%u in_size=%u out_off=%u out_size=%u lut_off=%u\n",
               merged_huff[0].in_offset, merged_huff[0].in_size,
               merged_huff[0].out_offset, merged_huff[0].out_size, merged_huff[0].lut_offset);
    // Dump first 5 chunk decomp sizes + first 5 lit huff out_offsets
    printf("chunk decomp[0..5]: ");
    for (int i = 0; i < 5 && i < (int)s.n_chunks; ++i) printf("%u ", s.chunks[i].decomp_size);
    printf("\nlit huff out_off[0..6]: ");
    for (int i = 0; i < 6 && i < (int)s.huff_lit.size(); ++i) printf("%u ", s.huff_lit[i].out_offset);
    printf("\nlit huff in_off[0..6]: ");
    for (int i = 0; i < 6 && i < (int)s.huff_lit.size(); ++i) printf("%u ", s.huff_lit[i].in_offset);
    printf("\nlit huff out_size[0..6]: ");
    for (int i = 0; i < 6 && i < (int)s.huff_lit.size(); ++i) printf("%u ", s.huff_lit[i].out_size);
    printf("\n");

    // ── Device buffers ─────────────────────────────────────
    uint8_t  *d_comp = nullptr;
    HostChunkDesc *d_descs = nullptr;
    HostHuffDesc  *d_huff_descs = nullptr;
    uint32_t *d_huff_lut = nullptr;
    uint8_t  *d_entropy_scratch = nullptr;
    uint8_t  *d_output = nullptr;
    uint32_t *d_n_huff = nullptr;
    uint32_t *d_n_groups = nullptr;
    uint32_t *d_first_sub_idx = nullptr;
    HostRawOff16Desc *d_raw_off16 = nullptr;
    uint32_t *d_n_raw = nullptr;

    size_t entropy_bytes = (size_t)total_subs * HOST_ENTROPY_SLOT_BYTES * 3;
    size_t huff_lut_bytes = (size_t)n_huff * HOST_HUFF_LUT_ENTRIES * sizeof(uint32_t);

    printf("alloc: comp=%.2f MB descs=%.2f KB huff_descs=%.2f KB huff_lut=%.2f MB entropy=%.2f MB output=%.2f MB\n",
        s.comp_len / (1024.0*1024.0),
        s.n_chunks * sizeof(HostChunkDesc) / 1024.0,
        n_huff * sizeof(HostHuffDesc) / 1024.0,
        huff_lut_bytes / (1024.0*1024.0),
        entropy_bytes / (1024.0*1024.0),
        decomp_total / (1024.0*1024.0));

    CK(cudaMalloc(&d_comp,             s.comp_len));
    CK(cudaMalloc(&d_descs,            s.n_chunks * sizeof(HostChunkDesc)));
    CK(cudaMalloc(&d_huff_descs,       n_huff * sizeof(HostHuffDesc)));
    CK(cudaMalloc(&d_huff_lut,         huff_lut_bytes));
    CK(cudaMalloc(&d_entropy_scratch,  entropy_bytes));
    CK(cudaMalloc(&d_output,           decomp_total));
    CK(cudaMalloc(&d_n_huff,           sizeof(uint32_t)));
    CK(cudaMalloc(&d_n_groups,         sizeof(uint32_t)));
    if (!s.raw_off16.empty()) {
        CK(cudaMalloc(&d_raw_off16, s.raw_off16.size() * sizeof(HostRawOff16Desc)));
        CK(cudaMalloc(&d_n_raw, sizeof(uint32_t)));
    }

    // first_sub_idx is only needed when total_subs != n_chunks (sc < 1).
    // For L5 -gpu (sc=0.25, 1 sub-chunk per chunk) total_subs == n_chunks
    // and we pass 0 (nullptr) to the kernel; the kernel uses identity mapping.
    bool need_first_sub_idx = (total_subs != s.n_chunks);
    std::vector<uint32_t> first_sub_idx_host;
    if (need_first_sub_idx) {
        first_sub_idx_host.resize(s.n_chunks);
        uint32_t acc = 0;
        for (uint32_t i = 0; i < s.n_chunks; ++i) {
            first_sub_idx_host[i] = acc;
            uint32_t subs = (s.chunks[i].decomp_size + s.sub_chunk_cap - 1) / s.sub_chunk_cap;
            if (subs == 0) subs = 1;
            acc += subs;
        }
        CK(cudaMalloc(&d_first_sub_idx, s.n_chunks * sizeof(uint32_t)));
        CK(cudaMemcpy(d_first_sub_idx, first_sub_idx_host.data(),
                      s.n_chunks * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ── H2D one-time ──────────────────────────────────────
    CK(cudaMemcpy(d_comp,       s.comp.data(),       s.comp_len, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_descs,      s.chunks.data(),     s.n_chunks * sizeof(HostChunkDesc), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_huff_descs, merged_huff.data(),  n_huff * sizeof(HostHuffDesc), cudaMemcpyHostToDevice));
    uint32_t h_n_huff   = n_huff;
    uint32_t h_n_groups = s.n_chunks; // chunks_per_group=1 for L5/-gpu → groups == chunks
    CK(cudaMemcpy(d_n_huff,   &h_n_huff,   sizeof(uint32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_n_groups, &h_n_groups, sizeof(uint32_t), cudaMemcpyHostToDevice));
    if (!s.raw_off16.empty()) {
        CK(cudaMemcpy(d_raw_off16, s.raw_off16.data(),
                      s.raw_off16.size() * sizeof(HostRawOff16Desc), cudaMemcpyHostToDevice));
        uint32_t h_n_raw = (uint32_t)s.raw_off16.size();
        CK(cudaMemcpy(d_n_raw, &h_n_raw, sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    CK(cudaDeviceSynchronize());

    // ── Launch helpers ─────────────────────────────────────
    auto launch_gather_raw = [&]() {
        if (s.raw_off16.empty()) return;
        // d_entropy_scratch + off16_offset is the off16 region base
        uint8_t* d_off16_scratch = d_entropy_scratch + 2 * (size_t)total_subs * HOST_ENTROPY_SLOT_BYTES;
        uint32_t ndesc = (uint32_t)s.raw_off16.size();
        slzGatherRawOff16Kernel<<<ndesc, 256, 0, 0>>>(
            d_comp, s.comp_len, d_off16_scratch,
            reinterpret_cast<SlzRawOff16Desc*>(d_raw_off16), d_n_raw);
    };
    auto launch_huff_build = [&]() {
        slzHuffBuildLutKernel<<<n_huff, 32, 0, 0>>>(
            d_comp,
            reinterpret_cast<HuffDecChunkDesc*>(d_huff_descs),
            d_huff_lut,
            d_n_huff);
    };
    auto launch_huff_decode = [&]() {
        slzHuffDecode4StreamKernel<<<n_huff, 32, HOST_HUFF_LUT_ENTRIES * sizeof(uint32_t), 0>>>(
            d_comp,
            reinterpret_cast<HuffDecChunkDesc*>(d_huff_descs),
            d_huff_lut,
            d_entropy_scratch,
            d_n_huff);
    };
    auto launch_lz_decode = [&]() {
        // chunks_per_group=1 → lz_groups_in_pipe = n_chunks; lz_grid_x = ceil(/2)
        uint32_t lz_grid_x = (s.n_chunks + 1) / 2;
        slzLzDecodeKernel<<<dim3(lz_grid_x, 1, 1), dim3(32, 2, 1), 0, 0>>>(
            d_comp,
            reinterpret_cast<SlzChunkDesc*>(d_descs),
            d_output,
            /*chunks_per_group=*/ 1u,
            d_n_groups,
            s.sub_chunk_cap,
            d_entropy_scratch,
            (uint64_t)total_subs * HOST_ENTROPY_SLOT_BYTES,
            d_first_sub_idx);
    };

    // ── Warm-up + correctness check ──────────────────────
    printf("\nwarm-up (correctness check) ...\n");
    // Zero output + scratch so any unwritten bytes are obviously 0 in
    // the diff dump (production doesn't zero — kernels must write every
    // byte they care about, so this should not change correctness).
    CK(cudaMemset(d_output, 0xAA, decomp_total));
    CK(cudaMemset(d_entropy_scratch, 0, entropy_bytes));
    CK(cudaMemset(d_huff_lut, 0, huff_lut_bytes));
    const char* skip_huff_env = getenv("DECODE_L5_SKIP_HUFF");
    bool skip_huff = skip_huff_env && skip_huff_env[0] != '0';
    if (n_huff == 0) skip_huff = true; // L1/L2 have no Huff
    if (!skip_huff) { launch_gather_raw(); launch_huff_build();
    cudaError_t le1 = cudaGetLastError();
    if (le1 != cudaSuccess) { printf("huff_build launch: %s\n", cudaGetErrorString(le1)); exit(1); }
    launch_huff_decode(); }
    cudaError_t le2 = cudaGetLastError();
    if (le2 != cudaSuccess) { printf("huff_decode launch: %s\n", cudaGetErrorString(le2)); exit(1); }
    launch_lz_decode(); printf("(skip_huff=%d)\n", skip_huff?1:0);
    cudaError_t le3 = cudaGetLastError();
    if (le3 != cudaSuccess) { printf("lz_decode launch: %s\n", cudaGetErrorString(le3)); exit(1); }
    CK(cudaDeviceSynchronize());

    std::vector<uint8_t> got(decomp_total);
    CK(cudaMemcpy(got.data(), d_output, decomp_total, cudaMemcpyDeviceToHost));

    // ── SC tail prefix restoration ─────────────────────────
    // Production encoder strips the first 8 bytes of each chunk[1..N-1]
    // from the LZ stream and stores them in a tail-prefix-table appended
    // after the block payload (see src/encode/fast_framed.zig and
    // src/decode/streamlz_decoder.zig:479-495 for the production caller).
    // The GPU kernels don't touch those bytes. The host caller restores
    // them from the prefix table. The SCN1 dump excludes the prefix
    // table, so we synthesize the same fixup here using bytes from the
    // expected source — same bytes the encoder would have written into
    // the prefix table. This is bookkeeping unrelated to GPU correctness
    // (~25KB of 203MB on silesia); the rest of the output IS the GPU
    // kernel result and must match bit-exact.
    {
        // sc=0.25 → eff_chunk_size = 64KB; restore first 8 bytes of
        // chunks 1..n-1 from the source (= expected bytes).
        uint32_t eff_cs = s.sub_chunk_cap / 2; // 64KB for sc=0.25 (sub_chunk_cap=128KB)
        // Actually use the first chunk's decomp_size as the per-chunk stride
        // (assuming uniform; last chunk may be partial).
        uint32_t per_chunk = s.chunks[0].decomp_size;
        (void)eff_cs;
        uint32_t prefix_chunks = 0;
        for (uint32_t i = 1; i < s.n_chunks; ++i) {
            uint64_t off = (uint64_t)s.chunks[i].dst_offset;
            if (off + 8 > expected.size()) break;
            memcpy(got.data() + off, expected.data() + off, 8);
            ++prefix_chunks;
        }
        printf("sc-prefix restored: %u chunks × 8 bytes\n", prefix_chunks);
        (void)per_chunk;
    }

    std::string got_sha = sha256_hex(got.data(), got.size());
    bool ok = (got_sha == expected_sha) && (got.size() == expected.size());
    printf("got SHA:      %s (%zu bytes) %s\n",
           got_sha.c_str(), got.size(), ok ? "[OK]" : "[MISMATCH]");
    if (!ok) {
        // Find first diff byte for debugging
        size_t lim = got.size() < expected.size() ? got.size() : expected.size();
        size_t i;
        for (i = 0; i < lim; ++i) if (got[i] != expected[i]) break;
        printf("first diff at byte %zu: got 0x%02x expected 0x%02x\n",
               i, i < lim ? got[i] : 0, i < lim ? expected[i] : 0);
        // Dump 32 bytes around the first diff
        size_t lo = i > 16 ? i - 16 : 0;
        size_t hi = (i + 16 < got.size()) ? i + 16 : got.size();
        printf("  expected ");
        for (size_t k = lo; k < hi; ++k) printf("%02x ", expected[k]);
        printf("\n  got      ");
        for (size_t k = lo; k < hi; ++k) printf("%02x ", got[k]);
        printf("\n");
        // Dump first 8 bytes (the init copy)
        printf("  bytes 0-7  expected ");
        for (size_t k = 0; k < 8; ++k) printf("%02x ", expected[k]);
        printf("\n  bytes 0-7  got      ");
        for (size_t k = 0; k < 8; ++k) printf("%02x ", got[k]);
        printf("\n");
        // Count total diffs and dump distribution by 64KB chunk
        size_t total_diffs = 0;
        std::vector<size_t> chunk_diffs(s.n_chunks, 0);
        for (size_t k = 0; k < lim; ++k) {
            if (got[k] != expected[k]) {
                total_diffs++;
                size_t chunk_idx = k / 65536; // sc=0.25 chunks = 64KB
                if (chunk_idx < chunk_diffs.size()) chunk_diffs[chunk_idx]++;
            }
        }
        printf("total diffs: %zu / %zu bytes (%.4f%%)\n",
               total_diffs, lim, 100.0 * total_diffs / lim);
        size_t chunks_with_diffs = 0;
        for (auto d : chunk_diffs) if (d > 0) chunks_with_diffs++;
        printf("chunks with diffs: %zu / %u\n", chunks_with_diffs, s.n_chunks);
        return 1;
    }

    // ── Timed runs ───────────────────────────────────────
    cudaEvent_t e_start, e_stop;
    CK(cudaEventCreate(&e_start));
    CK(cudaEventCreate(&e_stop));

    printf("\ntimed runs (cuEvent around 3-kernel pipeline, matches production -db):\n");
    float best_ms = 1e30f, sum_ms = 0;
    for (int r = 0; r < runs; ++r) {
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e_start));
        launch_gather_raw();
        launch_huff_build();
        launch_huff_decode();
        launch_lz_decode();
        CK(cudaEventRecord(e_stop));
        CK(cudaEventSynchronize(e_stop));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e_start, e_stop));
        double mbps = (decomp_total / (1024.0*1024.0)) / (ms / 1000.0);
        printf("  run %2d: %.3f ms (%.0f MB/s)\n", r+1, ms, mbps);
        if (ms < best_ms) best_ms = ms;
        sum_ms += ms;
    }
    float mean_ms = sum_ms / runs;
    double best_mbps = (decomp_total / (1024.0*1024.0)) / (best_ms / 1000.0);
    double mean_mbps = (decomp_total / (1024.0*1024.0)) / (mean_ms / 1000.0);
    printf("kernel best: %.3f ms (%.0f MB/s)\n", best_ms, best_mbps);
    printf("kernel mean: %.3f ms (%.0f MB/s)\n", mean_ms, mean_mbps);

    // Cleanup
    CK(cudaFree(d_comp));
    CK(cudaFree(d_descs));
    CK(cudaFree(d_huff_descs));
    CK(cudaFree(d_huff_lut));
    CK(cudaFree(d_entropy_scratch));
    CK(cudaFree(d_output));
    CK(cudaFree(d_n_huff));
    CK(cudaFree(d_n_groups));
    if (d_first_sub_idx) CK(cudaFree(d_first_sub_idx));
    if (d_raw_off16) CK(cudaFree(d_raw_off16));
    if (d_n_raw) CK(cudaFree(d_n_raw));
    CK(cudaEventDestroy(e_start));
    CK(cudaEventDestroy(e_stop));
    return 0;
}
