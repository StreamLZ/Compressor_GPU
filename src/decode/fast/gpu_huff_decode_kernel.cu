// ── StreamLZ GPU Huffman Decode Kernel ──────────────────────────
// Canonical Huffman with length-limit MAX_CODE_LEN=11, HUF_DECODER_X2
// double-symbol LUT (8KB), 4-stream parallel decode (zstd pattern).
//
// Wire layout for a chunk_type=4 literal stream:
//   [type-4 sub-chunk header parsed by skipEntropyStream — 3 or 5 bytes]
//   [128 B weights, 4 bits per symbol, packed low-nibble-first]
//   [9 B sub-header: 3 × u24 LE — stream sizes 0..2; size3 = remainder]
//   [stream 0 bits | stream 1 | stream 2 | stream 3]
// Bits within each byte are MSB-first.
//
// Two kernels:
//   slzHuffBuildLutKernel  — 1 warp per block, builds 2048-entry X2 LUT
//                            in global luts[] from 128 B weights.
//   slzHuffDecode4StreamKernel — 1 warp per block, 4 active lanes decode
//                                the 4 streams into output[dst_offset].
//
// Build: nvcc -cubin -arch=sm_89 -O3 gpu_huff_decode_kernel.cu

#include <cstdint>

static constexpr int MAX_CODE_LEN = 11;
static constexpr int LUT_SIZE     = 1 << MAX_CODE_LEN;  // 2048

// ── Descriptor — matches Zig HuffDecChunkDesc in gpu_driver.zig ─────
// Unified: in_offset/in_size cover the FULL payload (128 B weights +
// 9 B sub-header + 4 stream payloads). Build kernel reads the leading
// 128 B; decode kernel skips them and works on the remainder.
struct HuffDecChunkDesc {
    uint32_t in_offset;     // byte offset in compressed buffer (points at 128 B weights)
    uint32_t in_size;       // FULL payload bytes (128 + 9 + sum(streams))
    uint32_t out_offset;    // byte offset in literal scratch
    uint32_t out_size;      // expected decompressed bytes
    uint32_t lut_offset;    // entry index into luts[] (LUT_SIZE entries per block)
};

// ── LUT entry packing ───────────────────────────────────────────────
// Same layout as tools/huff_test/huff_ref.c pack_lut_entry():
//   bits  7:0  — sym1
//   bits 15:8  — sym2
//   bits 23:16 — total_len (bits consumed)
//   bits 31:24 — num_syms (1 or 2)
__device__ __forceinline__ uint32_t packLutEntry(uint8_t sym1, uint8_t sym2,
                                                  uint8_t total_len, uint8_t num_syms) {
    return ((uint32_t)num_syms << 24) | ((uint32_t)total_len << 16)
         | ((uint32_t)sym2 << 8) | (uint32_t)sym1;
}

// ── LUT build kernel ────────────────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (32, 1, 1).
// One warp per block. Weights[] is 128 bytes; each byte has two
// nibbles — low nibble = symbol 2*i, high nibble = symbol 2*i+1.
// Pass 1: single-symbol fan-out (lane 0 serial — 256 symbols, ~256 short loops).
// Pass 2: dual-symbol overwrite (lane 0 serial — at most 256*256 inner iters,
// but bounded by L1+L2<=11, which prunes to a small fraction of pairs).
//
// LUT is written to global memory (luts + desc.lut_offset).
extern "C" __global__ void slzHuffBuildLutKernel(
    const uint8_t* __restrict__ comp,
    const HuffDecChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ luts,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    const HuffDecChunkDesc desc = descs[block_id];
    const uint8_t* weights = comp + desc.in_offset;
    uint32_t* lut = luts + desc.lut_offset;

    // Unpack 128 B → 256 lengths into shared memory (one cache line of accesses).
    __shared__ uint8_t code_lengths[256];
    for (int i = lane; i < 128; i += 32) {
        uint8_t b = weights[i];
        code_lengths[i * 2 + 0] = b & 0x0F;
        code_lengths[i * 2 + 1] = (b >> 4) & 0x0F;
    }
    __syncwarp();

    // Compute canonical codes (RFC 1951 style — assigned in length-ascending
    // order, sym-ascending within each length). All lanes derive the same
    // codes[] table so each can do its slice of the fan-out work.
    __shared__ uint32_t codes[256];
    __shared__ uint32_t bl_count[16];
    __shared__ uint32_t next_code[16];

    if (lane < 16) bl_count[lane] = 0;
    __syncwarp();
    // Histogram lengths into bl_count. Each lane handles a stride of 32 symbols.
    for (int s = lane; s < 256; s += 32) {
        uint8_t L = code_lengths[s];
        if (L > 0 && L <= MAX_CODE_LEN) atomicAdd(&bl_count[L], 1);
    }
    __syncwarp();
    // Lane 0 builds next_code[] (serial — only 16 lengths).
    if (lane == 0) {
        uint32_t code = 0;
        next_code[0] = 0;
        for (int L = 1; L <= MAX_CODE_LEN; L++) {
            code = (code + bl_count[L - 1]) << 1;
            next_code[L] = code;
        }
    }
    __syncwarp();
    // Assign codes in (length, symbol) ascending order. Serial in lane 0:
    // only 256 symbols and trivially cheap. Avoids cross-lane sync.
    if (lane == 0) {
        for (int s = 0; s < 256; s++) {
            uint8_t L = code_lengths[s];
            if (L == 0) { codes[s] = 0; continue; }
            codes[s] = next_code[L]++;
        }
    }
    __syncwarp();

    // ── Build LUT in global memory ──
    // Zero the LUT — cooperative across warp.
    for (int i = lane; i < LUT_SIZE; i += 32) lut[i] = 0;
    __syncwarp();

    // Pass 1: single-symbol entries. Lane 0 does all writes; pre-aggregated
    // fan-out is too irregular to parallelize cleanly across 32 lanes.
    if (lane == 0) {
        for (int s = 0; s < 256; s++) {
            int L = code_lengths[s];
            if (L == 0) continue;
            uint32_t aligned = codes[s] << (MAX_CODE_LEN - L);
            uint32_t span = 1u << (MAX_CODE_LEN - L);
            uint32_t entry = packLutEntry((uint8_t)s, 0, (uint8_t)L, 1);
            for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
        }
    }
    __syncwarp();

    // Pass 2: dual-symbol entries — overwrite where L1+L2 <= MAX_CODE_LEN.
    // Lane 0 again; the inner loops are short because L1+L2>11 prunes pairs.
    if (lane == 0) {
        for (int s1 = 0; s1 < 256; s1++) {
            int L1 = code_lengths[s1];
            if (L1 == 0 || L1 >= MAX_CODE_LEN) continue;
            uint32_t C1 = codes[s1];
            for (int s2 = 0; s2 < 256; s2++) {
                int L2 = code_lengths[s2];
                if (L2 == 0) continue;
                int total = L1 + L2;
                if (total > MAX_CODE_LEN) continue;
                uint32_t C2 = codes[s2];
                uint32_t aligned = (C1 << (MAX_CODE_LEN - L1))
                                 | (C2 << (MAX_CODE_LEN - L1 - L2));
                uint32_t span = 1u << (MAX_CODE_LEN - total);
                uint32_t entry = packLutEntry((uint8_t)s1, (uint8_t)s2,
                                               (uint8_t)total, 2);
                for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
            }
        }
    }
}

// ── Single-stream decode core (one lane) ────────────────────────────
// Top-aligned bit buffer; 4-byte big-endian refill; HUF_DECODER_X2.
__device__ __forceinline__ uint32_t decodeStreamOneLane(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;

    // Hot loop: each iter emits 1 or 2 symbols. Refill when bit_count < 11.
    while (out_pos + 2 <= out_size) {
        if (bit_count < MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos]     = (uint8_t)(e & 0xFF);
        out[out_pos + 1] = (uint8_t)((e >> 8) & 0xFF);
        out_pos += num_syms;
        bit_buf <<= total_len;
        bit_count -= total_len;
    }

    // Tail: clamp dual entries to single at end-of-stream.
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── 4-stream decode kernel ──────────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (32, 1, 1) — one warp per block.
// Layout (after the 128 B weights, which this kernel skips via in_offset+128):
//   [9 B sub-header: 3 × u24 LE stream sizes; stream3 = total - sum]
//   [stream 0 | stream 1 | stream 2 | stream 3]
//
// `in_offset` points at the 128 B weights (full payload start); the
// decode kernel skips them internally to reach the 9 B sub-header.
extern "C" __global__ void slzHuffDecode4StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffDecChunkDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    const HuffDecChunkDesc desc = descs[block_id];

    // Cooperative LUT load into shared memory (2048 × 4 = 8 KB).
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    // Skip the 128 B weights to reach the 9 B sub-header.
    const uint8_t* hdr = comp + desc.in_offset + 128;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 128 - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    // This lane's input pointer (exclusive prefix sum of sizes).
    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    // This lane's output region (4 contiguous quarters).
    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    (void)decodeStreamOneLane(my_in, my_in_size, shared_lut, my_out, my_out_size);
}
