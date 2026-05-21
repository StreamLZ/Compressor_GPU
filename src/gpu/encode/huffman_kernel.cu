// GPU Huffman entropy encoder — two kernels.
//
//   slzHuffBuildTablesKernel  — histogram + canonical code-length build.
//   slzHuffEncode4StreamKernel — pack a sub-chunk into a chunk_type=4 body.
//
// Both mirror the production CPU encoder `huffman_encoder.zig` exactly, so
// the output pairs bit-for-bit with the production Huffman decoder. Codes
// are canonical and height-limited to 11 bits.
//
// Developed and verified byte-identical to a CPU oracle in the standalone
// harness tools/huff_test/ before being backported here.

#include <cstdint>

// Canonical-Huffman height limit. Matches huffman_encoder.zig MAX_CODE_LEN.
static constexpr int ENC_MAX_CODE_LEN = 11;

// One encode unit (one 64KB sub-chunk's stream). Mirrors Tans32EncDesc in
// gpu_tans_kernel.cu. `src_stride` is 1 for a contiguous stream (literals,
// tokens) or 2 to encode one byte plane of an interleaved lo/hi off16
// array (`src_offset` picks the lo=+0 or hi=+1 plane).
struct HuffEncDesc {
    uint32_t src_offset;
    uint32_t src_size;      // logical symbol count after stride extraction
    uint32_t src_stride;    // 1 or 2
    uint32_t dst_offset;    // body offset in the output buffer
    uint32_t dst_capacity;
};

// ── Single-stream encode core ─────────────────────────────────────────
// Sequential encode of one stream by one lane. Packs codewords MSB-first
// into `out`. Returns total bytes written (including the trailing partial
// byte). `stride` reads either a contiguous stream or one byte plane.
__device__ __forceinline__ uint32_t encode_stream_one_lane(
    const uint8_t* __restrict__ in, uint32_t in_size, uint32_t stride,
    const uint8_t* __restrict__ code_lengths,    // 256 entries
    const uint32_t* __restrict__ codes,          // 256 entries
    uint8_t* __restrict__ out)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t out_pos = 0;

    for (uint32_t i = 0; i < in_size; i++) {
        uint8_t  sym = in[(uint64_t)i * stride];
        uint32_t code = codes[sym];
        int      len  = code_lengths[sym];
        bit_buf = (bit_buf << len) | (uint64_t)(code & ((1u << len) - 1u));
        bit_count += len;
        while (bit_count >= 8) {
            bit_count -= 8;
            out[out_pos++] = (uint8_t)((bit_buf >> bit_count) & 0xFFu);
        }
    }
    if (bit_count > 0) {
        out[out_pos++] = (uint8_t)((bit_buf << (8 - bit_count)) & 0xFFu);
    }
    return out_pos;
}

// ══════════════════════════════════════════════════════════════════════
// Table builder — one block per sub-chunk, 32 lanes.
// ══════════════════════════════════════════════════════════════════════
//
// 32 lanes histogram the input into shared memory (atomicAdd); lane 0 then
// serially builds canonical code lengths and codes. Mirrors
// huffman_encoder.zig: buildCodeLengths -> heightLimit(11) ->
// assignCanonicalCodes.
//
// Writes code_lengths[256] (u8) + codes[256] (u32) per block, strided by
// `tables_stride` (= 256).
extern "C" __global__ void slzHuffBuildTablesKernel(
    const uint8_t* __restrict__ input,
    const HuffEncDesc* __restrict__ descs,        // src_offset, src_size, src_stride
    uint8_t* __restrict__ code_lengths_out,       // tables_stride × n_blocks
    uint32_t* __restrict__ codes_out,             // tables_stride × n_blocks
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffEncDesc d = descs[block_id];

    __shared__ uint32_t hist[256];
    __shared__ uint32_t weights[512];   // leaves 0..255, internals 256..510
    __shared__ int      parents[512];
    __shared__ int      active[512];
    __shared__ int      idx[256];
    __shared__ uint8_t  cl[256];
    __shared__ uint32_t codes[256];

    // ── Histogram (32 lanes → shared atomics) ──
    for (int i = lane; i < 256; i += 32) hist[i] = 0;
    __syncwarp();
    const uint8_t* in = input + d.src_offset;
    for (uint32_t i = lane; i < d.src_size; i += 32)
        atomicAdd(&hist[in[(uint64_t)i * d.src_stride]], 1u);
    __syncwarp();

    // ── Lane 0: serial tree-build + height-limit + canonical codes ──
    if (lane == 0) {
        for (int i = 0; i < 512; i++) { weights[i] = 0; parents[i] = -1; }
        for (int s = 0; s < 256; s++) { cl[s] = 0; codes[s] = 0; }

        int symbols_used = 0;
        for (int s = 0; s < 256; s++) {
            if (hist[s] > 0) { weights[s] = hist[s]; idx[symbols_used++] = s; }
        }

        if (symbols_used == 1) {
            cl[idx[0]] = 1;
        } else if (symbols_used >= 2) {
            int n_active = symbols_used;
            int next_node = 256;
            for (int i = 0; i < n_active; i++) active[i] = idx[i];
            while (n_active > 1) {
                int a_pos = 0, b_pos = 1;
                if (weights[active[a_pos]] > weights[active[b_pos]]) {
                    int t = a_pos; a_pos = b_pos; b_pos = t;
                }
                for (int i = 2; i < n_active; i++) {
                    uint32_t w = weights[active[i]];
                    if (w < weights[active[a_pos]]) { b_pos = a_pos; a_pos = i; }
                    else if (w < weights[active[b_pos]]) { b_pos = i; }
                }
                int a = active[a_pos], b = active[b_pos];
                weights[next_node] = weights[a] + weights[b];
                parents[a] = next_node;
                parents[b] = next_node;
                int new_pos = (a_pos < b_pos) ? a_pos : b_pos;
                int old_pos = (a_pos < b_pos) ? b_pos : a_pos;
                active[new_pos] = next_node;
                active[old_pos] = active[--n_active];
                next_node++;
            }
            for (int i = 0; i < symbols_used; i++) {
                int s = idx[i];
                int depth = 0, n = s;
                while (parents[n] != -1) { depth++; n = parents[n]; }
                cl[s] = (uint8_t)depth;
            }
        }

        // height-limit to ENC_MAX_CODE_LEN (Kraft-preserving, 30-bit fixed point)
        {
            const uint32_t TARGET = 1u << 30;
            const int limit = ENC_MAX_CODE_LEN;
            uint32_t sum = 0;
            for (int s = 0; s < 256; s++)
                if (cl[s] > 0) sum += (1u << (30 - cl[s]));
            for (int s = 0; s < 256; s++) {
                if (cl[s] > limit) {
                    sum -= (1u << (30 - cl[s]));
                    cl[s] = (uint8_t)limit;
                    sum += (1u << (30 - limit));
                }
            }
            while (sum > TARGET) {
                int best = -1;
                uint32_t best_w = 0xFFFFFFFFu;
                for (int s = 0; s < 256; s++) {
                    if (cl[s] > 0 && cl[s] < limit && hist[s] < best_w) {
                        best_w = hist[s]; best = s;
                    }
                }
                if (best < 0) break;
                sum -= (1u << (30 - cl[best]));
                cl[best]++;
                sum += (1u << (30 - cl[best]));
            }
        }

        // canonical code assignment
        {
            uint32_t length_count[ENC_MAX_CODE_LEN + 2] = {0};
            for (int s = 0; s < 256; s++) length_count[cl[s]]++;
            length_count[0] = 0;
            uint32_t next_code[ENC_MAX_CODE_LEN + 2] = {0};
            uint32_t code = 0;
            for (int L = 1; L <= ENC_MAX_CODE_LEN + 1; L++) {
                code = (code + length_count[L - 1]) << 1;
                next_code[L] = code;
            }
            for (int s = 0; s < 256; s++) {
                int L = cl[s];
                codes[s] = (L != 0) ? next_code[L]++ : 0u;
            }
        }
    }
    __syncwarp();

    // ── Write tables ──
    uint8_t*  CLo = code_lengths_out + (size_t)block_id * tables_stride;
    uint32_t* CDo = codes_out        + (size_t)block_id * tables_stride;
    for (int i = lane; i < 256; i += 32) { CLo[i] = cl[i]; CDo[i] = codes[i]; }
}

// ══════════════════════════════════════════════════════════════════════
// 4-stream encoder — one block per sub-chunk, lanes 0..3 active.
// ══════════════════════════════════════════════════════════════════════
//
// Emits the chunk_type=4 body (huffman_encoder.zig encodeBlock, minus the
// 5-byte chunk header which the frame assembler prepends):
//   [128 B weights — 4 bits/symbol, byte i = cl[2i] | cl[2i+1]<<4]
//   [9 B sub-header — 3 × u24 LE stream sizes; stream 3 derived]
//   [stream 0 | stream 1 | stream 2 | stream 3]
//
// Code tables come from slzHuffBuildTablesKernel. Lanes 0..3 each encode
// one input quarter into per-stream scratch; lane 0 then assembles the
// body at `output + descs[block_id].dst_offset`.
extern "C" __global__ void slzHuffEncode4StreamKernel(
    const uint8_t* __restrict__ input,
    const HuffEncDesc* __restrict__ descs_in,
    const uint8_t* __restrict__ code_lengths,         // 256 × n_blocks
    const uint32_t* __restrict__ codes,               // 256 × n_blocks
    uint8_t* __restrict__ scratch,                    // per-stream scratch
    uint8_t* __restrict__ output,                     // packed bodies
    uint32_t* __restrict__ out_sizes,                 // per-block body size
    uint32_t scratch_per_stream,
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffEncDesc d = descs_in[block_id];

    // Empty descriptor → no Huffman body for this sub-chunk (caller uses
    // the raw fallback). Signalled by out_sizes == 0. d is block-uniform,
    // so all 32 lanes return together — no shuffle/sync divergence.
    if (d.src_size == 0) {
        if (lane == 0) out_sizes[block_id] = 0;
        return;
    }

    // Lanes 0..3 each encode one of the 4 quarters into scratch.
    uint32_t bytes = 0;
    if (lane < 4) {
        uint32_t q = d.src_size / 4;
        uint32_t my_in_start = (lane < 3) ? (lane * q) : (3 * q);
        uint32_t my_in_size  = (lane < 3) ? q : (d.src_size - 3 * q);
        const uint8_t* my_in = input + d.src_offset + (uint64_t)my_in_start * d.src_stride;
        uint8_t* my_scratch = scratch + ((uint64_t)block_id * 4 + lane) * scratch_per_stream;
        const uint8_t*  my_cl = code_lengths + (uint64_t)block_id * tables_stride;
        const uint32_t* my_cd = codes        + (uint64_t)block_id * tables_stride;
        bytes = encode_stream_one_lane(my_in, my_in_size, d.src_stride, my_cl, my_cd, my_scratch);
    }
    __syncwarp();

    // Convergent shuffles — every lane participates.
    uint32_t s0 = __shfl_sync(0xFFFFFFFF, bytes, 0);
    uint32_t s1 = __shfl_sync(0xFFFFFFFF, bytes, 1);
    uint32_t s2 = __shfl_sync(0xFFFFFFFF, bytes, 2);
    uint32_t s3 = __shfl_sync(0xFFFFFFFF, bytes, 3);

    uint8_t* out = output + d.dst_offset;
    const uint8_t* cl = code_lengths + (uint64_t)block_id * tables_stride;

    // 128-byte weights — 32 lanes pack 4 entries each.
    for (int i = lane; i < 128; i += 32)
        out[i] = (uint8_t)((cl[2 * i] & 0x0F) | ((cl[2 * i + 1] & 0x0F) << 4));

    if (lane == 0) {
        // 9-byte sub-header at offset 128 (3 × u24 LE; stream 3 derived).
        uint8_t* h = out + 128;
        h[0] = (uint8_t)(s0 & 0xFF); h[1] = (uint8_t)((s0 >> 8) & 0xFF); h[2] = (uint8_t)((s0 >> 16) & 0xFF);
        h[3] = (uint8_t)(s1 & 0xFF); h[4] = (uint8_t)((s1 >> 8) & 0xFF); h[5] = (uint8_t)((s1 >> 16) & 0xFF);
        h[6] = (uint8_t)(s2 & 0xFF); h[7] = (uint8_t)((s2 >> 8) & 0xFF); h[8] = (uint8_t)((s2 >> 16) & 0xFF);

        uint8_t* dst = out + 137;
        uint8_t* src0 = scratch + ((uint64_t)block_id * 4 + 0) * scratch_per_stream;
        uint8_t* src1 = scratch + ((uint64_t)block_id * 4 + 1) * scratch_per_stream;
        uint8_t* src2 = scratch + ((uint64_t)block_id * 4 + 2) * scratch_per_stream;
        uint8_t* src3 = scratch + ((uint64_t)block_id * 4 + 3) * scratch_per_stream;
        for (uint32_t i = 0; i < s0; i++) dst[i] = src0[i]; dst += s0;
        for (uint32_t i = 0; i < s1; i++) dst[i] = src1[i]; dst += s1;
        for (uint32_t i = 0; i < s2; i++) dst[i] = src2[i]; dst += s2;
        for (uint32_t i = 0; i < s3; i++) dst[i] = src3[i];

        out_sizes[block_id] = 128 + 9 + s0 + s1 + s2 + s3;
    }
}
