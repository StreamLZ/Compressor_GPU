// GPU Huffman 4-stream decoder.
//
// Format mirrors the CPU reference (`huff_ref.c`):
//   [9-byte header: 3 × u24 LE stream sizes; stream4 derived]
//   [stream 0 bits]
//   [stream 1 bits]
//   [stream 2 bits]
//   [stream 3 bits]
//
// Codes are MSB-first within bytes; decoder uses a 2048-entry fast LUT
// (11-bit prefix → (length:8, symbol:8) packed in u16).
//
// One block per Huffman block (= one sub-chunk in production). 4 active
// lanes per warp each decode one of the 4 streams independently. Other
// lanes idle. We can revisit lane utilization (e.g., 8 lanes/stream with
// shared bit buffer) once correctness is proven.

#include <cstdint>
#include <cstdio>

static constexpr int MAX_CODE_LEN = 11;
static constexpr int LUT_SIZE     = 1 << MAX_CODE_LEN;  // 2048
static constexpr int LUT_BYTES    = LUT_SIZE * sizeof(uint16_t);  // 4096

struct HuffBlockDesc {
    uint32_t in_offset;     // byte offset in compressed buffer (points at 9-byte header)
    uint32_t in_size;       // total compressed bytes including header
    uint32_t out_offset;    // byte offset in output buffer
    uint32_t out_size;      // expected decompressed bytes
    uint32_t lut_offset;    // entry offset in luts array (LUT_SIZE entries per block)
};

// ── Single-stream decode core ─────────────────────────────────────────
// Decodes one Huffman bit-stream of `in_size` bytes into `out_size` bytes.
// Pure single-thread; called by per-lane dispatch in the warp kernel.
// Returns number of output bytes written (== out_size on success, 0 on
// internal error).
__device__ __forceinline__ uint32_t decode_stream_one_lane(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint16_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;

    while (out_pos < out_size) {
        // Refill: keep ≥ MAX_CODE_LEN bits in buffer.
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf = (bit_buf << 8) | (uint64_t)in[in_pos++];
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) {
            // Pad with zeros for trailing partial codewords.
            bit_buf <<= (MAX_CODE_LEN - bit_count);
            bit_count = MAX_CODE_LEN;
        }
        uint32_t idx = (uint32_t)((bit_buf >> (bit_count - MAX_CODE_LEN)) & (LUT_SIZE - 1));
        uint16_t entry = lut[idx];
        int len = entry >> 8;
        if (len == 0) return 0;  // invalid stream
        out[out_pos++] = (uint8_t)(entry & 0xFF);
        bit_count -= len;
    }
    return out_pos;
}

// ── Single-stream encode core ─────────────────────────────────────────
// Sequential encode of one stream by one lane. Writes packed bits MSB-first
// into `out`. Returns total bytes written (including trailing partial byte).
__device__ __forceinline__ uint32_t encode_stream_one_lane(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint8_t* __restrict__ code_lengths,    // 256 entries
    const uint32_t* __restrict__ codes,          // 256 entries
    uint8_t* __restrict__ out)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t out_pos = 0;

    for (uint32_t i = 0; i < in_size; i++) {
        uint8_t  sym = in[i];
        uint32_t code = codes[sym];
        int      len  = code_lengths[sym];
        bit_buf = (bit_buf << len) | (uint64_t)(code & ((1u << len) - 1u));
        bit_count += len;
        // Drain whole bytes from the top of bit_buf.
        while (bit_count >= 8) {
            bit_count -= 8;
            out[out_pos++] = (uint8_t)((bit_buf >> bit_count) & 0xFFu);
        }
    }
    // Flush trailing partial byte (pad LSB with zeros).
    if (bit_count > 0) {
        out[out_pos++] = (uint8_t)((bit_buf << (8 - bit_count)) & 0xFFu);
    }
    return out_pos;
}

// ── Encoder kernel: 1 block per Huffman block, 4 active lanes ─────────
// Each block encodes ONE Huffman block (sub-chunk) given the input bytes
// and pre-built code tables. Lanes 0..3 each encode one of the 4 quarters
// in parallel into a SCRATCH buffer; then lane 0 writes the 9-byte header
// + concatenates the streams into the final output buffer.
//
// Scratch layout (per block): 4 contiguous regions, each with worst-case
// size = (in_quarter_size * 2 + 64) bytes. Host pre-allocates global mem.
extern "C" __global__ void huffEncode4StreamKernel(
    const uint8_t* __restrict__ input,
    const HuffBlockDesc* __restrict__ descs_in,       // (in_offset, in_size, scratch_off, scratch_per_stream)
    const uint8_t* __restrict__ code_lengths,         // 256 × n_blocks
    const uint32_t* __restrict__ codes,               // 256 × n_blocks
    uint8_t* __restrict__ scratch,                    // per-stream scratch buffer
    uint8_t* __restrict__ output,                     // packed output
    uint32_t* __restrict__ out_sizes,                 // per-block output size
    const uint32_t* __restrict__ out_offsets,         // per-block output offset
    uint32_t scratch_per_stream,                      // per-stream scratch capacity
    uint32_t tables_stride,                           // 256 (per-block table stride)
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc d = descs_in[block_id];

    // All 32 lanes compute `bytes` (lanes 4..31 stay 0). This keeps the
    // following __shfl_sync convergent (all lanes participate).
    uint32_t bytes = 0;
    if (lane < 4) {
        uint32_t q = d.in_size / 4;
        uint32_t my_in_start = (lane < 3) ? (lane * q) : (3 * q);
        uint32_t my_in_size  = (lane < 3) ? q : (d.in_size - 3 * q);
        const uint8_t* my_in = input + d.in_offset + my_in_start;

        // Scratch layout per block: 4 streams × scratch_per_stream bytes,
        // starting at scratch + block_id * 4 * scratch_per_stream.
        uint8_t* my_scratch = scratch + ((uint64_t)block_id * 4 + lane) * scratch_per_stream;

        // Per-block code table base.
        const uint8_t*  my_cl = code_lengths + (uint64_t)block_id * tables_stride;
        const uint32_t* my_cd = codes        + (uint64_t)block_id * tables_stride;

        bytes = encode_stream_one_lane(my_in, my_in_size, my_cl, my_cd, my_scratch);
    }
    __syncwarp();

    // Convergent shuffles — every lane must participate.
    uint32_t bytes_l0 = __shfl_sync(0xFFFFFFFF, bytes, 0);
    uint32_t bytes_l1 = __shfl_sync(0xFFFFFFFF, bytes, 1);
    uint32_t bytes_l2 = __shfl_sync(0xFFFFFFFF, bytes, 2);
    uint32_t bytes_l3 = __shfl_sync(0xFFFFFFFF, bytes, 3);

    if (lane == 0) {
        // Write header + concatenate scratch streams into output[out_offsets[block_id]].
        uint8_t* out = output + out_offsets[block_id];
        uint32_t s0 = bytes_l0, s1 = bytes_l1, s2 = bytes_l2, s3 = bytes_l3;
        out[0] = (uint8_t)(s0 & 0xFF); out[1] = (uint8_t)((s0 >> 8) & 0xFF); out[2] = (uint8_t)((s0 >> 16) & 0xFF);
        out[3] = (uint8_t)(s1 & 0xFF); out[4] = (uint8_t)((s1 >> 8) & 0xFF); out[5] = (uint8_t)((s1 >> 16) & 0xFF);
        out[6] = (uint8_t)(s2 & 0xFF); out[7] = (uint8_t)((s2 >> 8) & 0xFF); out[8] = (uint8_t)((s2 >> 16) & 0xFF);

        uint8_t* dst = out + 9;
        uint8_t* src0 = scratch + ((uint64_t)block_id * 4 + 0) * scratch_per_stream;
        uint8_t* src1 = scratch + ((uint64_t)block_id * 4 + 1) * scratch_per_stream;
        uint8_t* src2 = scratch + ((uint64_t)block_id * 4 + 2) * scratch_per_stream;
        uint8_t* src3 = scratch + ((uint64_t)block_id * 4 + 3) * scratch_per_stream;
        for (uint32_t i = 0; i < s0; i++) dst[i] = src0[i]; dst += s0;
        for (uint32_t i = 0; i < s1; i++) dst[i] = src1[i]; dst += s1;
        for (uint32_t i = 0; i < s2; i++) dst[i] = src2[i]; dst += s2;
        for (uint32_t i = 0; i < s3; i++) dst[i] = src3[i];

        out_sizes[block_id] = 9 + s0 + s1 + s2 + s3;
    }
}

// ── Decoder kernel: 1 block per Huffman block, 4 active lanes ────────
//
// Grid: (n_blocks, 1, 1). Block: (32, 1, 1).
//
// Each block decodes one Huffman block (one sub-chunk). The block's warp
// cooperatively loads the LUT into shared memory (4KB), then 4 lanes
// (0..3) each decode one of the 4 streams. Other lanes idle.
extern "C" __global__ void huffDecode4StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint16_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    // Load LUT cooperatively into shared memory (2048 × 2 = 4096 bytes).
    extern __shared__ uint16_t shared_lut[];
    const uint16_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) {
        shared_lut[i] = src_lut[i];
    }
    __syncwarp();

    if (lane >= 4) return;

    // Parse header: 3 × u24 LE sizes.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    // Compute this lane's input pointer (offset within compressed payload).
    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    // Compute this lane's output region (4 contiguous quarters).
    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane(my_in, my_in_size, shared_lut,
                                              my_out, my_out_size);
    (void)written;  // future: write per-block error flag if written != my_out_size
}
