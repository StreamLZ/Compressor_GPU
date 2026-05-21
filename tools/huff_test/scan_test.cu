// ── StreamLZ GPU decode-scan harness ────────────────────────────────
// Phase 2 (roadmap 4d): verify a GPU port of scanForTansChunks against
// the real CPU oracle. Input: c:/tmp/scan_dump.bin, produced by the Zig
// decoder run with SLZ_DUMP_SCAN=1 — it carries the scan's input
// (chunk_descs + compressed block) and its output (the four Huffman
// descriptor arrays + the raw-off16 list).
//
// The GPU kernel runs one thread per chunk, walks every sub-chunk's
// literal / token / off16 stream headers, and stages a descriptor per
// stream type per sub-chunk. The host then compacts the staged output
// (drop raw-literal sub-chunks, assign lut_offset) and diffs it against
// the oracle. PASS == the GPU parse reproduces the CPU scan exactly.
//
// Build: tools/huff_test/build_scan.bat  (nvcc -O3 -arch=sm_89)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Wire-format / layout constants (mirror src/gpu/decode/driver.zig) ─
static constexpr uint32_t HUFF_LUT_ENTRIES   = 1024;
static constexpr uint32_t PER_SUBCHUNK_SLOT  = 131072; // sub_dst_off stride
static constexpr uint32_t OFF16_LO_SLOT      = 65536;  // lo half within slot
static constexpr uint32_t SUBCHUNK_64K       = 0x10000;
static constexpr uint32_t OFF16_MARKER       = 0xFFFF;

// ChunkDesc — mirrors the Zig extern struct (24 bytes).
struct ChunkDesc {
    uint32_t src_offset, comp_size, decomp_size, dst_offset, flags;
    uint8_t  memset_fill; uint8_t _pad[3];
};

// HuffDecChunkDesc — mirrors the Zig extern struct (20 bytes).
struct HuffDecChunkDesc {
    uint32_t in_offset, in_size, out_offset, out_size, lut_offset;
};

// RawOff16Desc — 3×u32 (12 bytes).
struct RawOff16Desc { uint32_t src_offset, size, gpu_offset; };

// Staged per-sub-chunk descriptors written by the parse kernel. `valid`
// marks whether the stream is entropy-coded (huff) / raw — the host
// compaction drops invalid slots.
struct StagedHuff { uint32_t in_offset, in_size, out_offset, out_size, valid; };
struct StagedRaw  { uint32_t src_offset, size, gpu_offset, valid; };

#define CK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
    exit(1);} } while(0)

// ── Device header parsers (ports of the Zig scan helpers) ───────────
__device__ __forceinline__ uint32_t readU24BE(const uint8_t* p) {
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
}

// chunk_type=4 Huffman header at `pos` within chunk_src. Fills `out`
// (in_offset block-relative, out_offset = caller's dst_off). lut_offset
// is left 0 — the host assigns it during compaction.
__device__ static bool parseHuffHeaderDev(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos,
    uint32_t src_offset_base, uint32_t dst_off, StagedHuff& out) {
    if (pos >= chunk_len) return false;
    const uint8_t first = chunk_src[pos];
    uint32_t comp_size, dst_size, payload_off;
    if (first >= 0x80) {
        if (pos + 3 > chunk_len) return false;
        const uint32_t bits = ((uint32_t)chunk_src[pos] << 16)
                            | ((uint32_t)chunk_src[pos + 1] << 8)
                            | (uint32_t)chunk_src[pos + 2];
        comp_size = bits & 0x3FF;
        dst_size  = comp_size + ((bits >> 10) & 0x3FF) + 1;
        payload_off = src_offset_base + pos + 3;
    } else {
        if (pos + 5 > chunk_len) return false;
        const uint32_t bits = ((uint32_t)chunk_src[pos + 1] << 24)
                            | ((uint32_t)chunk_src[pos + 2] << 16)
                            | ((uint32_t)chunk_src[pos + 3] << 8)
                            | (uint32_t)chunk_src[pos + 4];
        comp_size = bits & 0x3FFFF;
        dst_size  = (((bits >> 18) | ((uint32_t)chunk_src[pos] << 14)) & 0x3FFFF) + 1;
        payload_off = src_offset_base + pos + 5;
    }
    out.in_offset  = payload_off;
    out.in_size    = comp_size;
    out.out_offset = dst_off;
    out.out_size   = dst_size;
    out.valid      = 1;
    return true;
}

// type-0 (memcpy) stream header → data offset (chunk-relative) + size.
__device__ static bool parseType0Dev(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos,
    uint32_t& data_off, uint32_t& size) {
    if (pos >= chunk_len) return false;
    const uint8_t first = chunk_src[pos];
    if (first >= 0x80) {
        if (pos + 2 > chunk_len) return false;
        size = (((uint32_t)chunk_src[pos] << 8) | (uint32_t)chunk_src[pos + 1]) & 0xFFF;
        data_off = pos + 2;
    } else {
        if (pos + 3 > chunk_len) return false;
        size = readU24BE(chunk_src + pos);
        data_off = pos + 3;
    }
    return data_off != 0;
}

// Skip an entropy/raw stream header + payload. Returns new pos, or
// 0xFFFFFFFF on truncation.
__device__ static uint32_t skipStreamHeaderDev(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos) {
    if (pos >= chunk_len) return 0xFFFFFFFFu;
    const uint8_t first = chunk_src[pos];
    const uint32_t ct = (first >> 4) & 0x7;
    if (ct == 0) {
        if (first >= 0x80) {
            if (pos + 2 > chunk_len) return 0xFFFFFFFFu;
            const uint32_t sz = (((uint32_t)chunk_src[pos] << 8) | (uint32_t)chunk_src[pos + 1]) & 0xFFF;
            return pos + 2 + sz;
        } else {
            if (pos + 3 > chunk_len) return 0xFFFFFFFFu;
            return pos + 3 + readU24BE(chunk_src + pos);
        }
    } else if (ct == 1 || ct == 2 || ct == 4 || ct == 6) {
        if (first >= 0x80) {
            if (pos + 3 > chunk_len) return 0xFFFFFFFFu;
            const uint32_t bits = readU24BE(chunk_src + pos);
            return pos + 3 + (bits & 0x3FF);
        } else {
            if (pos + 5 > chunk_len) return 0xFFFFFFFFu;
            const uint32_t bits = ((uint32_t)chunk_src[pos + 1] << 24)
                                | ((uint32_t)chunk_src[pos + 2] << 16)
                                | ((uint32_t)chunk_src[pos + 3] << 8)
                                | (uint32_t)chunk_src[pos + 4];
            return pos + 5 + (bits & 0x3FFFF);
        }
    } else if (ct == 5) {
        if (pos + 7 > chunk_len) return 0xFFFFFFFFu;
        return pos + 7;
    } else if (ct == 7) {
        if (pos + 4 > chunk_len) return 0xFFFFFFFFu;
        return skipStreamHeaderDev(chunk_src, chunk_len, pos + 4);
    }
    return 0xFFFFFFFFu;
}

// ── Parse kernel — one thread per chunk ─────────────────────────────
// Walks every sub-chunk of the chunk and stages a descriptor per stream
// type per global sub-chunk index. Slots for raw-literal / absent
// streams are left valid=0.
extern "C" __global__ void slzScanParseKernel(
    const uint8_t*  __restrict__ block,
    uint32_t                     block_len,
    const ChunkDesc* __restrict__ chunks,
    const uint32_t* __restrict__ first_sub_idx,
    uint32_t                     n_chunks,
    uint32_t                     sub_chunk_cap,
    StagedHuff* __restrict__ st_lit,
    StagedHuff* __restrict__ st_tok,
    StagedHuff* __restrict__ st_hi,
    StagedHuff* __restrict__ st_lo,
    StagedRaw*  __restrict__ st_raw_hi,
    StagedRaw*  __restrict__ st_raw_lo)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;

    const ChunkDesc ch = chunks[i];
    const uint32_t cap_safe = sub_chunk_cap ? sub_chunk_cap : 65536u;
    const uint32_t chunk_first_sub = first_sub_idx[i];

    // Non-LZ / empty chunk: occupies one sub-chunk slot, no descriptors.
    if (ch.flags != 0 || ch.decomp_size == 0) {
        st_lit[chunk_first_sub].valid = 0;
        st_tok[chunk_first_sub].valid = 0;
        st_hi[chunk_first_sub].valid  = 0;
        st_lo[chunk_first_sub].valid  = 0;
        st_raw_hi[chunk_first_sub].valid = 0;
        st_raw_lo[chunk_first_sub].valid = 0;
        return;
    }
    if (ch.src_offset >= block_len) return;

    uint32_t chunk_end = ch.src_offset + ch.comp_size;
    if (chunk_end > block_len) chunk_end = block_len;
    const uint8_t* chunk_src = block + ch.src_offset;
    const uint32_t chunk_len = chunk_end - ch.src_offset;

    uint32_t sub_pos = 0;
    uint32_t remaining = ch.decomp_size;
    uint32_t sub_local = 0;
    const uint32_t n_subs_expected = (ch.decomp_size + cap_safe - 1) / cap_safe;

    while (remaining > 0 && sub_pos + 3 <= chunk_len) {
        const uint32_t sub_idx = chunk_first_sub + sub_local;
        // Default every slot for this sub-chunk to invalid.
        st_lit[sub_idx].valid = 0;
        st_tok[sub_idx].valid = 0;
        st_hi[sub_idx].valid  = 0;
        st_lo[sub_idx].valid  = 0;
        st_raw_hi[sub_idx].valid = 0;
        st_raw_lo[sub_idx].valid = 0;

        if (chunk_len < 3) break;
        const uint32_t sub_hdr = readU24BE(chunk_src + sub_pos);
        if ((sub_hdr & 0x800000u) == 0) break;          // not an LZ sub-chunk
        const uint32_t sc_comp = sub_hdr & 0x7FFFFu;
        const uint32_t sub_end = sub_pos + 3 + sc_comp;
        if (sub_end > chunk_len) break;
        const uint32_t sub_decomp = remaining < cap_safe ? remaining : cap_safe;

        const uint32_t sub_dst_off = sub_idx * PER_SUBCHUNK_SLOT;
        const uint32_t init_b = (sub_idx == 0) ? 8u : 0u;
        uint32_t pos = sub_pos + 3 + init_b;
        const uint32_t end = sub_end;

        do {
            if (pos >= end) break;
            // Stream 1: literals.
            const uint32_t lit_type = (chunk_src[pos] >> 4) & 0x7;
            if (lit_type == 4)
                parseHuffHeaderDev(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off, st_lit[sub_idx]);
            uint32_t nxt = skipStreamHeaderDev(chunk_src, chunk_len, pos);
            if (nxt == 0xFFFFFFFFu) break;
            pos = nxt;
            if (pos >= end) break;

            // Stream 2: tokens.
            const uint32_t tok_type = (chunk_src[pos] >> 4) & 0x7;
            if (tok_type == 4)
                parseHuffHeaderDev(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off, st_tok[sub_idx]);
            nxt = skipStreamHeaderDev(chunk_src, chunk_len, pos);
            if (nxt == 0xFFFFFFFFu) break;
            pos = nxt;
            if (pos >= end) break;

            // cmd_stream2_offset (2 bytes) for sub-chunks > 64KB.
            if (sub_decomp > SUBCHUNK_64K) {
                if (pos + 2 > end) break;
                pos += 2;
            }
            if (pos + 2 > end) break;

            // Off16: must be entropy-coded (0xFFFF marker) to have hi/lo.
            const uint32_t marker = (uint32_t)chunk_src[pos] | ((uint32_t)chunk_src[pos + 1] << 8);
            if (marker != OFF16_MARKER) break;
            pos += 2;
            if (pos >= end) break;

            // Off16 hi.
            const uint32_t hi_type = (chunk_src[pos] >> 4) & 0x7;
            if (hi_type == 0) {
                uint32_t doff, sz;
                if (parseType0Dev(chunk_src, chunk_len, pos, doff, sz)) {
                    st_raw_hi[sub_idx].src_offset = ch.src_offset + doff;
                    st_raw_hi[sub_idx].size       = sz;
                    st_raw_hi[sub_idx].gpu_offset = sub_dst_off;
                    st_raw_hi[sub_idx].valid      = 1;
                }
            } else if (hi_type == 4) {
                parseHuffHeaderDev(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off, st_hi[sub_idx]);
            }
            nxt = skipStreamHeaderDev(chunk_src, chunk_len, pos);
            if (nxt == 0xFFFFFFFFu) break;
            pos = nxt;
            if (pos >= end) break;

            // Off16 lo.
            const uint32_t lo_type = (chunk_src[pos] >> 4) & 0x7;
            if (lo_type == 0) {
                uint32_t doff, sz;
                if (parseType0Dev(chunk_src, chunk_len, pos, doff, sz)) {
                    st_raw_lo[sub_idx].src_offset = ch.src_offset + doff;
                    st_raw_lo[sub_idx].size       = sz;
                    st_raw_lo[sub_idx].gpu_offset = sub_dst_off + OFF16_LO_SLOT;
                    st_raw_lo[sub_idx].valid      = 1;
                }
            } else if (lo_type == 4) {
                parseHuffHeaderDev(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off + OFF16_LO_SLOT, st_lo[sub_idx]);
            }
        } while (0);

        sub_pos = sub_end;
        remaining -= sub_decomp;
        sub_local++;
        if (sub_local >= n_subs_expected) break;
    }
}

// ── Host helpers ────────────────────────────────────────────────────
static uint8_t* g_file = nullptr;
static size_t   g_pos  = 0;
static size_t   g_len  = 0;

static uint32_t rdU32() {
    uint32_t v; memcpy(&v, g_file + g_pos, 4); g_pos += 4; return v;
}
static const void* rdBytes(size_t n) {
    const void* p = g_file + g_pos; g_pos += n; return p;
}

static int cmpHuff(const char* name, const HuffDecChunkDesc* got, uint32_t ngot,
                   const HuffDecChunkDesc* exp, uint32_t nexp) {
    if (ngot != nexp) {
        printf("  %-14s FAIL  count got=%u expected=%u\n", name, ngot, nexp);
        return 1;
    }
    for (uint32_t i = 0; i < ngot; i++) {
        const HuffDecChunkDesc& a = got[i];
        const HuffDecChunkDesc& b = exp[i];
        if (a.in_offset != b.in_offset || a.in_size != b.in_size ||
            a.out_offset != b.out_offset || a.out_size != b.out_size ||
            a.lut_offset != b.lut_offset) {
            printf("  %-14s FAIL  desc[%u] got{in=%u,sz=%u,out=%u,osz=%u,lut=%u} "
                   "exp{in=%u,sz=%u,out=%u,osz=%u,lut=%u}\n", name, i,
                   a.in_offset,a.in_size,a.out_offset,a.out_size,a.lut_offset,
                   b.in_offset,b.in_size,b.out_offset,b.out_size,b.lut_offset);
            return 1;
        }
    }
    printf("  %-14s PASS  %u descriptors\n", name, ngot);
    return 0;
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "c:/tmp/scan_dump.bin";
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END); g_len = ftell(f); fseek(f, 0, SEEK_SET);
    g_file = (uint8_t*)malloc(g_len);
    if (fread(g_file, 1, g_len, f) != g_len) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    const uint32_t magic = rdU32();
    if (magic != 0x53434E31u) { fprintf(stderr, "bad magic %#x\n", magic); return 1; }
    const uint32_t n_chunks = rdU32();
    const uint32_t sub_chunk_cap = rdU32();
    const uint32_t block_len = rdU32();
    printf("scan dump: n_chunks=%u sub_chunk_cap=%u block_len=%u\n",
           n_chunks, sub_chunk_cap, block_len);

    const ChunkDesc* chunks = (const ChunkDesc*)rdBytes((size_t)n_chunks * sizeof(ChunkDesc));
    const uint8_t* block = (const uint8_t*)rdBytes(block_len);

    HuffDecChunkDesc* oracle[4]; uint32_t oracle_n[4];
    for (int s = 0; s < 4; s++) {
        oracle_n[s] = rdU32();
        oracle[s] = (HuffDecChunkDesc*)rdBytes((size_t)oracle_n[s] * sizeof(HuffDecChunkDesc));
    }
    const uint32_t oracle_nraw = rdU32();
    const RawOff16Desc* oracle_raw = (const RawOff16Desc*)rdBytes((size_t)oracle_nraw * sizeof(RawOff16Desc));
    printf("oracle: lit=%u tok=%u off16hi=%u off16lo=%u raw=%u\n",
           oracle_n[0], oracle_n[1], oracle_n[2], oracle_n[3], oracle_nraw);

    // Host prefix sum: first global sub-chunk index per chunk.
    const uint32_t cap_safe = sub_chunk_cap ? sub_chunk_cap : 65536u;
    uint32_t* first_sub = (uint32_t*)malloc((size_t)n_chunks * 4);
    uint32_t total_subs = 0;
    for (uint32_t i = 0; i < n_chunks; i++) {
        first_sub[i] = total_subs;
        const uint32_t n = (chunks[i].flags != 0 || chunks[i].decomp_size == 0)
                         ? 1u : (chunks[i].decomp_size + cap_safe - 1) / cap_safe;
        total_subs += n;
    }
    printf("total sub-chunks=%u\n", total_subs);

    // Device buffers.
    uint8_t* d_block; ChunkDesc* d_chunks; uint32_t* d_first;
    CK(cudaMalloc(&d_block, block_len));
    CK(cudaMalloc(&d_chunks, (size_t)n_chunks * sizeof(ChunkDesc)));
    CK(cudaMalloc(&d_first, (size_t)n_chunks * 4));
    CK(cudaMemcpy(d_block, block, block_len, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_chunks, chunks, (size_t)n_chunks * sizeof(ChunkDesc), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_first, first_sub, (size_t)n_chunks * 4, cudaMemcpyHostToDevice));

    StagedHuff *d_lit, *d_tok, *d_hi, *d_lo;
    StagedRaw  *d_raw_hi, *d_raw_lo;
    const size_t sh = (size_t)total_subs * sizeof(StagedHuff);
    const size_t sr = (size_t)total_subs * sizeof(StagedRaw);
    CK(cudaMalloc(&d_lit, sh)); CK(cudaMalloc(&d_tok, sh));
    CK(cudaMalloc(&d_hi, sh));  CK(cudaMalloc(&d_lo, sh));
    CK(cudaMalloc(&d_raw_hi, sr)); CK(cudaMalloc(&d_raw_lo, sr));
    CK(cudaMemset(d_lit, 0, sh)); CK(cudaMemset(d_tok, 0, sh));
    CK(cudaMemset(d_hi, 0, sh));  CK(cudaMemset(d_lo, 0, sh));
    CK(cudaMemset(d_raw_hi, 0, sr)); CK(cudaMemset(d_raw_lo, 0, sr));

    const int TPB = 256;
    const int blocks = (n_chunks + TPB - 1) / TPB;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0));
    slzScanParseKernel<<<blocks, TPB>>>(d_block, block_len, d_chunks, d_first,
                                        n_chunks, sub_chunk_cap,
                                        d_lit, d_tok, d_hi, d_lo, d_raw_hi, d_raw_lo);
    CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("parse kernel: %.3f ms\n", ms);

    StagedHuff *h_lit = (StagedHuff*)malloc(sh), *h_tok = (StagedHuff*)malloc(sh);
    StagedHuff *h_hi  = (StagedHuff*)malloc(sh), *h_lo  = (StagedHuff*)malloc(sh);
    StagedRaw  *h_raw_hi = (StagedRaw*)malloc(sr), *h_raw_lo = (StagedRaw*)malloc(sr);
    CK(cudaMemcpy(h_lit, d_lit, sh, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_tok, d_tok, sh, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_hi,  d_hi,  sh, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_lo,  d_lo,  sh, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_raw_hi, d_raw_hi, sr, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_raw_lo, d_raw_lo, sr, cudaMemcpyDeviceToHost));

    // Host compaction: drop invalid slots, assign lut_offset = k * 1024.
    HuffDecChunkDesc* comp[4];
    uint32_t comp_n[4] = {0,0,0,0};
    StagedHuff* staged[4] = { h_lit, h_tok, h_hi, h_lo };
    for (int s = 0; s < 4; s++) {
        comp[s] = (HuffDecChunkDesc*)malloc((size_t)total_subs * sizeof(HuffDecChunkDesc));
        uint32_t k = 0;
        for (uint32_t i = 0; i < total_subs; i++) {
            if (!staged[s][i].valid) continue;
            comp[s][k] = { staged[s][i].in_offset, staged[s][i].in_size,
                           staged[s][i].out_offset, staged[s][i].out_size,
                           k * HUFF_LUT_ENTRIES };
            k++;
        }
        comp_n[s] = k;
    }
    // Raw compaction: per sub-chunk emit hi then lo, in sub-chunk order.
    RawOff16Desc* comp_raw = (RawOff16Desc*)malloc((size_t)total_subs * 2 * sizeof(RawOff16Desc));
    uint32_t comp_nraw = 0;
    for (uint32_t i = 0; i < total_subs; i++) {
        if (h_raw_hi[i].valid)
            comp_raw[comp_nraw++] = { h_raw_hi[i].src_offset, h_raw_hi[i].size, h_raw_hi[i].gpu_offset };
        if (h_raw_lo[i].valid)
            comp_raw[comp_nraw++] = { h_raw_lo[i].src_offset, h_raw_lo[i].size, h_raw_lo[i].gpu_offset };
    }

    printf("\n-- verification --\n");
    int fails = 0;
    const char* names[4] = { "huff_lit", "huff_tok", "huff_off16hi", "huff_off16lo" };
    for (int s = 0; s < 4; s++)
        fails += cmpHuff(names[s], comp[s], comp_n[s], oracle[s], oracle_n[s]);

    if (comp_nraw != oracle_nraw) {
        printf("  %-14s FAIL  count got=%u expected=%u\n", "raw_off16", comp_nraw, oracle_nraw);
        fails++;
    } else {
        int rfail = 0;
        for (uint32_t i = 0; i < comp_nraw; i++) {
            if (comp_raw[i].src_offset != oracle_raw[i].src_offset ||
                comp_raw[i].size       != oracle_raw[i].size ||
                comp_raw[i].gpu_offset != oracle_raw[i].gpu_offset) {
                printf("  %-14s FAIL  desc[%u] got{src=%u,sz=%u,gpu=%u} exp{src=%u,sz=%u,gpu=%u}\n",
                       "raw_off16", i, comp_raw[i].src_offset, comp_raw[i].size, comp_raw[i].gpu_offset,
                       oracle_raw[i].src_offset, oracle_raw[i].size, oracle_raw[i].gpu_offset);
                rfail = 1; break;
            }
        }
        if (rfail) fails++;
        else printf("  %-14s PASS  %u descriptors\n", "raw_off16", comp_nraw);
    }

    printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILED");
    return fails == 0 ? 0 : 1;
}
