// GPU Huffman entropy encoder — two kernels.
//
//   slzHuffBuildTablesKernel  — histogram + canonical code-length build.
//   slzHuffEncode4StreamKernel — pack a sub-chunk into a chunk_type=4 body.
//
// Both mirror the production CPU Huffman encoder exactly, so the output
// pairs bit-for-bit with the production Huffman decoder. Codes are
// canonical and height-limited to `ENC_MAX_CODE_LEN` (11) bits.
//
// Developed and verified byte-identical to a CPU oracle in the standalone
// harness tools/huff_test/ before being backported here.

#include <cstdint>
#include "../common/gpu_warp.cuh"     // WARP_SIZE, FULL_WARP_MASK, BITS_PER_BYTE
#include "../common/gpu_byteio.cuh"   // writeLE24
#include "../common/gpu_huffman.cuh"  // HUFF_* constants, weights pack, buildCanonicalCodes

// Canonical-Huffman height limit — an alias of the shared
// HUFF_MAX_CODE_LEN (11), the same limit the CPU Huffman encoder uses.
// gpu_huffman.cuh ties the LUT index width on the decode side to this
// value with a static_assert, so the old hand-maintained "must equal
// MAX_CODE_LEN + 1" invariant is now compiler-checked.
static constexpr int ENC_MAX_CODE_LEN = HUFF_MAX_CODE_LEN;  // 11

// Encode-specific table sizes. HUFF_* wire constants, the weights
// pack helper and buildCanonicalCodes come from common/gpu_huffman.cuh.
static constexpr int HUFF_TREE_NODES = 2 * HUFF_ALPHABET;  // 256 leaves + up to 255 internals

// Kraft fixed-point precision. The Kraft sum 1.0 is represented as
// 1u << KRAFT_PRECISION_BITS; 30 leaves headroom so `1u << (30 - 1)`
// (shortest code) and the running sum both stay within a u32.
static constexpr int      KRAFT_PRECISION_BITS   = 30;

// ── Encode unit descriptor — matches Zig HuffEncDesc in encode/driver.zig ──
// One encode unit (one 64KB sub-chunk's stream). `src_stride` is 1 for a
// contiguous stream (literals, tokens) or 2 to encode one byte plane of an
// interleaved lo/hi off16 array (`src_offset` picks the lo=+0 or hi=+1 plane).
struct HuffEncDesc {
    uint32_t src_offset;
    uint32_t src_size;      // logical symbol count after stride extraction
    uint32_t src_stride;    // 1 or 2
    uint32_t dst_offset;    // body offset in the output buffer
    uint32_t dst_capacity;
};
static_assert(sizeof(HuffEncDesc) == 20, "ABI: keep in sync with encode/driver.zig");

// ── Single-stream encode core ─────────────────────────────────────────
// Sequential encode of one stream by one lane. Packs codewords MSB-first
// into `out`. Returns total bytes written (including the trailing partial
// byte). `stride` reads either a contiguous stream or one byte plane.
// Capacity invariant: the caller guarantees `out` holds at least
// `in_size * 2` bytes (driver-sized per-stream scratch); this core does
// not bounds-check against dst_capacity.
__device__ __forceinline__ uint32_t encodeStreamOneLane(
    const uint8_t* __restrict__ in, uint32_t in_size, uint32_t stride,
    const uint8_t* __restrict__ code_lengths,    // HUFF_ALPHABET entries
    const uint32_t* __restrict__ codes,          // HUFF_ALPHABET entries
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
        while (bit_count >= BITS_PER_BYTE) {
            bit_count -= BITS_PER_BYTE;
            out[out_pos++] = (uint8_t)((bit_buf >> bit_count) & 0xFFu);
        }
    }
    if (bit_count > 0) {
        out[out_pos++] = (uint8_t)((bit_buf << (BITS_PER_BYTE - bit_count)) & 0xFFu);
    }
    return out_pos;
}

// ── Table builder — one block per sub-chunk, WARP_SIZE lanes ─────────
//
// 32 lanes histogram the input into shared memory (atomicAdd); lane 0 then
// serially builds canonical code lengths and codes. Mirrors the CPU
// Huffman encoder: build code lengths -> Kraft-sum fixed-point height
// limiting (1<<KRAFT_PRECISION_BITS budget) -> assign canonical codes.
//
// Writes code_lengths[256] (u8) + codes[256] (u32) per block, strided by
// `tables_stride` (= HUFF_ALPHABET; a runtime parameter that never varies).
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
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const HuffEncDesc desc = descs[block_id];

    __shared__ uint32_t hist[HUFF_ALPHABET];
    __shared__ uint32_t weights[HUFF_TREE_NODES];   // leaves 0..255, internals 256..510
    __shared__ int      parents[HUFF_TREE_NODES];
    __shared__ int      active[HUFF_TREE_NODES];
    __shared__ int      used_symbols[HUFF_ALPHABET];
    __shared__ uint8_t  code_lengths[HUFF_ALPHABET];
    __shared__ uint32_t codes[HUFF_ALPHABET];

    // ── Histogram (32 lanes → shared atomics) ──
    for (int i = lane; i < HUFF_ALPHABET; i += WARP_SIZE) hist[i] = 0;
    __syncwarp();
    const uint8_t* in = input + desc.src_offset;
    for (uint32_t i = lane; i < desc.src_size; i += WARP_SIZE)
        atomicAdd(&hist[in[(uint64_t)i * desc.src_stride]], 1u);
    __syncwarp();

    // ── Lane 0: serial tree-build + height-limit + canonical codes ──
    if (lane == 0) {
        for (int i = 0; i < HUFF_TREE_NODES; i++) { weights[i] = 0; parents[i] = -1; }
        for (int s = 0; s < HUFF_ALPHABET; s++) { code_lengths[s] = 0; codes[s] = 0; }

        int symbols_used = 0;
        for (int s = 0; s < HUFF_ALPHABET; s++) {
            if (hist[s] > 0) { weights[s] = hist[s]; used_symbols[symbols_used++] = s; }
        }

        if (symbols_used == 1) {
            // Degenerate single-symbol alphabet: a 1-bit code. Deliberate
            // path matching the CPU Huffman encoder; the decoder LUT
            // fan-out handles length-1 codes.
            code_lengths[used_symbols[0]] = 1;
        } else if (symbols_used >= 2) {
            int n_active = symbols_used;
            int next_node = HUFF_ALPHABET;
            for (int i = 0; i < n_active; i++) active[i] = used_symbols[i];
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
                int s = used_symbols[i];
                int depth = 0, n = s;
                while (parents[n] != -1) { depth++; n = parents[n]; }
                code_lengths[s] = (uint8_t)depth;
            }
        }

        // height-limit to ENC_MAX_CODE_LEN (Kraft-preserving, fixed point)
        {
            const uint32_t kraft_budget = 1u << KRAFT_PRECISION_BITS;
            const int max_len = ENC_MAX_CODE_LEN;
            uint32_t sum = 0;
            for (int s = 0; s < HUFF_ALPHABET; s++)
                if (code_lengths[s] > 0) sum += (1u << (KRAFT_PRECISION_BITS - code_lengths[s]));
            for (int s = 0; s < HUFF_ALPHABET; s++) {
                if (code_lengths[s] > max_len) {
                    sum -= (1u << (KRAFT_PRECISION_BITS - code_lengths[s]));
                    code_lengths[s] = (uint8_t)max_len;
                    sum += (1u << (KRAFT_PRECISION_BITS - max_len));
                }
            }
            while (sum > kraft_budget) {
                int best = -1;
                uint32_t best_w = 0xFFFFFFFFu;
                for (int s = 0; s < HUFF_ALPHABET; s++) {
                    if (code_lengths[s] > 0 && code_lengths[s] < max_len && hist[s] < best_w) {
                        best_w = hist[s]; best = s;
                    }
                }
                if (best < 0) break;
                sum -= (1u << (KRAFT_PRECISION_BITS - code_lengths[best]));
                code_lengths[best]++;
                sum += (1u << (KRAFT_PRECISION_BITS - code_lengths[best]));
            }
        }

        // canonical code assignment — RFC-1951 canonical codes, shared
        // with the decoder's LUT builder (common/gpu_huffman.cuh).
        buildCanonicalCodes(code_lengths, codes);
    }
    __syncwarp();

    // ── Write tables ──
    uint8_t*  code_lengths_dst = code_lengths_out + (size_t)block_id * tables_stride;
    uint32_t* codes_dst        = codes_out        + (size_t)block_id * tables_stride;
    for (int i = lane; i < HUFF_ALPHABET; i += WARP_SIZE) {
        code_lengths_dst[i] = code_lengths[i];
        codes_dst[i]        = codes[i];
    }
}

// ── 4-stream encoder — one block per sub-chunk, lanes 0..3 active ────
//
// Emits the chunk_type=4 body (the CPU encoder's encode-block output,
// minus the 5-byte chunk header which the frame assembler prepends):
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
    const uint8_t* __restrict__ code_lengths,         // HUFF_ALPHABET × n_blocks
    const uint32_t* __restrict__ codes,               // HUFF_ALPHABET × n_blocks
    uint8_t* __restrict__ scratch,                    // per-stream scratch
    uint8_t* __restrict__ output,                     // packed bodies
    uint32_t* __restrict__ out_sizes,                 // per-block body size
    uint32_t scratch_per_stream,
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const HuffEncDesc desc = descs_in[block_id];

    // Empty descriptor → no Huffman body for this sub-chunk (caller uses
    // the raw fallback). Signalled by out_sizes == 0. desc is block-uniform,
    // so all 32 lanes return together — no shuffle/sync divergence.
    if (desc.src_size == 0) {
        if (lane == 0) out_sizes[block_id] = 0;
        return;
    }

    // Per-block table base (one buffer, one name — reused for encode and
    // for the weights pack below).
    const uint8_t* block_code_lengths = code_lengths + (uint64_t)block_id * tables_stride;

    // Lanes 0..3 each encode one of the 4 quarters into scratch.
    uint32_t bytes = 0;
    if (lane < HUFF_NUM_STREAMS) {
        uint32_t q = desc.src_size / HUFF_NUM_STREAMS;
        uint32_t my_in_start = (lane < HUFF_NUM_STREAMS - 1)
                             ? (lane * q)
                             : ((HUFF_NUM_STREAMS - 1) * q);
        uint32_t my_in_size  = (lane < HUFF_NUM_STREAMS - 1)
                             ? q
                             : (desc.src_size - (HUFF_NUM_STREAMS - 1) * q);
        const uint8_t* my_in = input + desc.src_offset + (uint64_t)my_in_start * desc.src_stride;
        uint8_t* my_scratch = scratch
            + ((uint64_t)block_id * HUFF_NUM_STREAMS + lane) * scratch_per_stream;
        const uint32_t* my_codes = codes + (uint64_t)block_id * tables_stride;
        bytes = encodeStreamOneLane(my_in, my_in_size, desc.src_stride,
                                    block_code_lengths, my_codes, my_scratch);
    }
    // Barrier covers the lane-0..3 scratch writes before the cross-lane
    // shuffles below; all 32 lanes reach here (the if-block has no return).
    __syncwarp();

    // Convergent shuffles — every lane participates. stream_bytes[s] holds
    // the encoded byte count of stream s.
    uint32_t stream_bytes[HUFF_NUM_STREAMS];
    #pragma unroll
    for (int s = 0; s < HUFF_NUM_STREAMS; s++)
        stream_bytes[s] = __shfl_sync(FULL_WARP_MASK, bytes, s);

    uint8_t* out = output + desc.dst_offset;

    // 128-byte weights — 32 lanes pack 4 entries each.
    for (int i = lane; i < HUFF_WEIGHTS_BYTES; i += WARP_SIZE)
        out[i] = packWeightByte(block_code_lengths[2 * i], block_code_lengths[2 * i + 1]);

    if (lane == 0) {
        // 9-byte sub-header at offset 128 (3 × u24 LE; stream 3 derived).
        uint8_t* subheader = out + HUFF_WEIGHTS_BYTES;
        for (int s = 0; s < HUFF_NUM_STREAMS - 1; s++) {
            writeLE24(subheader + s * HUFF_STREAM_SIZE_BYTES, stream_bytes[s]);
        }

        // Concatenate the 4 per-stream scratch buffers into the body.
        uint8_t* dst = out + HUFF_BODY_HEADER_BYTES;
        uint32_t total_body = 0;
        for (int s = 0; s < HUFF_NUM_STREAMS; s++) {
            const uint8_t* src = scratch
                + ((uint64_t)block_id * HUFF_NUM_STREAMS + s) * scratch_per_stream;
            for (uint32_t i = 0; i < stream_bytes[s]; i++) dst[i] = src[i];
            dst += stream_bytes[s];
            total_body += stream_bytes[s];
        }

        out_sizes[block_id] = HUFF_WEIGHTS_BYTES + HUFF_SUBHEADER_BYTES + total_body;
    }
}
