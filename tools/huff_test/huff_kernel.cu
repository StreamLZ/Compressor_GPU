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
