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
static constexpr int LUT_BYTES    = LUT_SIZE * sizeof(uint32_t);  // 4096

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
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    // Top-aligned bit buffer: bits [63 .. 64-bit_count] are valid, lower
    // are zero. Allows 8-byte big-endian refill when bit_count == 0.
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;

    // ── Hot loop: unrolled 2× — one refill guards two LUT decodes ──
    // Each LUT lookup is indexed by MAX_CODE_LEN (11) bits and consumes
    // total_len ≤ 11 bits while emitting 1 OR 2 symbols. Two decodes
    // consume ≤ 22 bits.
    //
    // The 32-bit refill `bit_buf |= v << (32 - bit_count)` is only valid
    // when bit_count ≤ 32. We refill whenever bit_count < 2*MAX_CODE_LEN
    // (< 22), so the shift amount is in [10, 32] — always valid — and the
    // post-refill bit_count is in [32, 53], i.e. ≥ 22, enough for the two
    // decodes below with no mid-group refill check. This halves the
    // refill-branch / loop-branch / bounds-check overhead per symbol.
    //
    // Each iteration emits up to 4 symbols (2 decodes × ≤2 syms), so the
    // output-room guard is `out_pos + 4 <= out_size`.
    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        out[out_pos]     = (uint8_t)(e0 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e0 >> 8) & 0xFF);
        out_pos += ns0;
        bit_buf <<= len0;
        bit_count -= len0;
        // decode B (≤ 22 bits total consumed this iter; refill above
        // guaranteed ≥ 22 valid bits, so no refill check needed here)
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        out[out_pos]     = (uint8_t)(e1 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e1 >> 8) & 0xFF);
        out_pos += ns1;
        bit_buf <<= len1;
        bit_count -= len1;
    }

    // ── Tail loop: handles last 1-2 bytes; clamps dual entries to single ──
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
        // Even if we clamped to single output, we still consumed total_len
        // bits — see CPU reference rationale (only happens at end-of-stream).
        bit_buf <<= total_len;
        bit_count -= total_len;
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

// ══════════════════════════════════════════════════════════════════════
// 32-STREAM VARIANT — all 32 lanes active, one stream per lane
// ══════════════════════════════════════════════════════════════════════
//
// Format:
//   [93-byte header: 31 × u24 LE stream sizes; stream 31 size derived]
//   [stream 0 bits | stream 1 | ... | stream 31]
//
// Same MSB-first per-byte bit order as the 4-stream version. Each lane
// (0..31) decodes/encodes its own quarter (= 1/32 of output). For 16 KB
// literal streams, each lane handles 512 bytes.

static constexpr int N_STREAMS_32 = 32;
static constexpr int HUFF_32S_HDR_BYTES = (N_STREAMS_32 - 1) * 3;  // 93

// ── Encoder kernel: 1 block per Huffman block, 32 active lanes ───────
extern "C" __global__ void huffEncode32StreamKernel(
    const uint8_t* __restrict__ input,
    const HuffBlockDesc* __restrict__ descs_in,
    const uint8_t* __restrict__ code_lengths,
    const uint32_t* __restrict__ codes,
    uint8_t* __restrict__ scratch,
    uint8_t* __restrict__ output,
    uint32_t* __restrict__ out_sizes,
    const uint32_t* __restrict__ out_offsets,
    uint32_t scratch_per_stream,
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc d = descs_in[block_id];

    // Per-lane quarter (32-way split).
    uint32_t q = d.in_size / 32;
    uint32_t my_in_start = (lane < 31) ? (lane * q) : (31 * q);
    uint32_t my_in_size  = (lane < 31) ? q : (d.in_size - 31 * q);
    const uint8_t* my_in = input + d.in_offset + my_in_start;

    uint8_t* my_scratch = scratch + ((uint64_t)block_id * 32 + lane) * scratch_per_stream;
    const uint8_t*  my_cl = code_lengths + (uint64_t)block_id * tables_stride;
    const uint32_t* my_cd = codes        + (uint64_t)block_id * tables_stride;

    uint32_t bytes = encode_stream_one_lane(my_in, my_in_size, my_cl, my_cd, my_scratch);

    __syncwarp();

    // Gather all 32 stream sizes into every lane (32 shuffles, convergent).
    uint32_t all_sizes[N_STREAMS_32];
    #pragma unroll
    for (int s = 0; s < N_STREAMS_32; s++) {
        all_sizes[s] = __shfl_sync(0xFFFFFFFF, bytes, s);
    }

    // Lane 0 writes header + concatenates payload.
    if (lane == 0) {
        uint8_t* out = output + out_offsets[block_id];
        uint8_t* hdr = out;
        uint32_t total = HUFF_32S_HDR_BYTES;
        for (int s = 0; s < N_STREAMS_32 - 1; s++) {
            uint32_t sz = all_sizes[s];
            hdr[s*3 + 0] = (uint8_t)(sz & 0xFF);
            hdr[s*3 + 1] = (uint8_t)((sz >> 8) & 0xFF);
            hdr[s*3 + 2] = (uint8_t)((sz >> 16) & 0xFF);
            total += sz;
        }
        total += all_sizes[N_STREAMS_32 - 1];

        uint8_t* dst = out + HUFF_32S_HDR_BYTES;
        for (int s = 0; s < N_STREAMS_32; s++) {
            uint32_t sz = all_sizes[s];
            uint8_t* src = scratch + ((uint64_t)block_id * 32 + s) * scratch_per_stream;
            for (uint32_t i = 0; i < sz; i++) dst[i] = src[i];
            dst += sz;
        }
        out_sizes[block_id] = total;
    }
}

// ── Decoder kernel: 1 block per Huffman block, 32 active lanes ───────
extern "C" __global__ void huffDecode32StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    // Cooperative LUT load.
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    // Parse header: 31 × u24 LE sizes + derive stream 31 size.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t my_size;
    if (lane < N_STREAMS_32 - 1) {
        my_size = (uint32_t)hdr[lane*3 + 0]
                | ((uint32_t)hdr[lane*3 + 1] << 8)
                | ((uint32_t)hdr[lane*3 + 2] << 16);
    } else {
        my_size = 0;  // computed via warp reduce below
    }
    // Warp reduce sum of sizes[0..30] to derive size[31].
    uint32_t sum = my_size;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    if (lane == N_STREAMS_32 - 1) {
        my_size = (desc.in_size - HUFF_32S_HDR_BYTES) - sum;
    }
    __syncwarp();

    // Compute per-lane input offset via prefix-sum (exclusive scan).
    uint32_t in_off = my_size;
    for (int off = 1; off < 32; off <<= 1) {
        uint32_t v = __shfl_up_sync(0xFFFFFFFF, in_off, off);
        if (lane >= off) in_off += v;
    }
    in_off -= my_size;  // exclusive

    const uint8_t* my_in = hdr + HUFF_32S_HDR_BYTES + in_off;

    // Per-lane output region.
    uint32_t q = desc.out_size / 32;
    uint32_t my_out_start = (lane < 31) ? (lane * q) : (31 * q);
    uint32_t my_out_size  = (lane < 31) ? q : (desc.out_size - 31 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane(my_in, my_size, shared_lut,
                                              my_out, my_out_size);
    (void)written;
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
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    // Load LUT cooperatively into shared memory (2048 × 2 = 4096 bytes).
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
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

// ── Decoder kernel: 2 Huffman blocks per warp, 8 active lanes ─────────
// Each warp handles a PAIR of blocks: lanes 0-3 decode block (2*pid)'s
// 4 streams, lanes 4-7 decode block (2*pid+1)'s. 8 active lanes amortize
// the warp's per-instruction issue slot 2x vs the 4-lane kernel, while
// staying well below the 32-lane LDS-bank-conflict regime (8 random LUT
// indices into 32 banks rarely collide). Shared memory holds 2 LUTs.
// Launch: <<<(n_blocks + 1) / 2, 32, 2 * LUT_BYTES>>>
extern "C" __global__ void huffDecode4Stream2xKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t pid  = blockIdx.x;
    const uint32_t bidA = pid * 2;
    const uint32_t bidB = bidA + 1;
    if (bidA >= n_blocks) return;
    const bool haveB = (bidB < n_blocks);
    const int lane = threadIdx.x & 31;

    // Shared memory holds two LUTs: [0, LUT_SIZE) for block A,
    // [LUT_SIZE, 2*LUT_SIZE) for block B. All 32 lanes load cooperatively.
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* lutA = luts + descs[bidA].lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = lutA[i];
    if (haveB) {
        const uint32_t* lutB = luts + descs[bidB].lut_offset;
        for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[LUT_SIZE + i] = lutB[i];
    }
    __syncwarp();

    // Active lanes 0-7: lane>>2 picks the block, lane&3 picks the stream.
    if (lane >= 8) return;
    if (lane >= 4 && !haveB) return;
    const uint32_t sub   = (uint32_t)lane >> 2;   // 0 = block A, 1 = block B
    const int      slane = lane & 3;              // stream index 0-3
    const HuffBlockDesc desc = descs[sub ? bidB : bidA];
    const uint32_t* my_lut = shared_lut + sub * LUT_SIZE;

    // Parse header: 3 × u24 LE sizes; stream 3 derived.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    stream_sizes[3] = (desc.in_size - 9)
                    - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t s_in_off = 0;
    for (int s = 0; s < slane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size  = stream_sizes[slane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (slane < 3) ? (slane * q) : (3 * q);
    uint32_t my_out_size  = (slane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane(my_in, my_in_size, my_lut,
                                              my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: 1 block per warp, only 2 active lanes ─────────────
// Each of 2 lanes decodes TWO of the block's 4 streams sequentially.
// Lane 0 -> streams 0,1 ; lane 1 -> streams 2,3. This is the "narrow"
// counterpart to the 4/8/32-lane variants: it minimizes concurrent LUT
// lookups (2 lanes -> 2 indices) at the cost of 2x serial work per lane.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4Stream2LaneKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 2) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    stream_sizes[3] = (desc.in_size - 9)
                    - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t prefix[4];
    prefix[0] = 0;
    prefix[1] = stream_sizes[0];
    prefix[2] = stream_sizes[0] + stream_sizes[1];
    prefix[3] = stream_sizes[0] + stream_sizes[1] + stream_sizes[2];

    const uint32_t q = desc.out_size / 4;

    // This lane owns streams (2*lane) and (2*lane + 1).
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int s = 2 * lane + k;
        const uint8_t* my_in = hdr + 9 + prefix[s];
        uint32_t my_in_size  = stream_sizes[s];
        uint32_t my_out_start = (s < 3) ? (s * q) : (3 * q);
        uint32_t my_out_size  = (s < 3) ? q : (desc.out_size - 3 * q);
        uint8_t* my_out = output + desc.out_offset + my_out_start;
        uint32_t written = decode_stream_one_lane(my_in, my_in_size, shared_lut,
                                                  my_out, my_out_size);
        (void)written;
    }
}
