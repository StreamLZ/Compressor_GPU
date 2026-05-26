// ── StreamLZ GPU Huffman Decode Kernel ──────────────────────────
// Canonical Huffman, code height limited to 11 (`MAX_CODE_LEN + 1`);
// fast-LUT index width `MAX_CODE_LEN = 10`. Double-symbol (X2) LUT
// (4 KB shared), 4-stream parallel decode (zstd pattern).
//
// Wire layout for a chunk_type=4 literal stream:
//   [preceding chunk header, consumed by the frame decoder — 3 or 5 bytes]
//   [128 B weights, 4 bits per symbol, packed low-nibble-first]
//   [9 B sub-header: 3 × u24 LE — stream sizes 0..2; size3 = remainder]
//   [stream 0 bits | stream 1 | stream 2 | stream 3]
// Bits within each byte are MSB-first.
//
// Two kernels:
//   slzHuffBuildLutKernel  — 1 warp per block, builds a 1024-entry
//                            (`LUT_SIZE`) X2 LUT in global luts[] from
//                            128 B weights.
//   slzHuffDecode4StreamKernel — 1 warp per block, 4 active lanes decode
//                                the 4 streams into output[dst_offset].
//
// Built to huffman_kernel.ptx by tools/build_gpu.bat (nvcc);
// decode/driver.zig @embedFile's the PTX.

#include <cstdint>
#include "../common/gpu_warp.cuh"     // WARP_SIZE
#include "../common/gpu_byteio.cuh"   // readLE24
#include "../common/gpu_huffman.cuh"  // HUFF_* constants, LUT pack/unpack, buildCanonicalCodes

// MAX_CODE_LEN is the fast-LUT *index width* (10) — an alias of the
// shared HUFF_LUT_INDEX_BITS. Codes are height-limited to
// MAX_CODE_LEN+1 = HUFF_MAX_CODE_LEN = 11; length-11 codes resolve via
// escape LUT entries (num_syms == LUT_NUM_SYMS_ESCAPE) — a 1024-entry
// LUT (4 KB shared) instead of 2048 (8 KB), roughly doubling
// decode-kernel occupancy.
static constexpr int MAX_CODE_LEN = HUFF_LUT_INDEX_BITS;  // 10
static constexpr int LUT_SIZE     = 1 << MAX_CODE_LEN;    // 1024

// HUFF_* wire constants, the canonical-code builder, the LUT-entry
// pack/unpack helpers and the num_syms tags come from
// common/gpu_huffman.cuh.

// Bit-buffer arithmetic.
static constexpr int      BITBUF_BITS  = 64;                   // bit_buf width in bits
static constexpr int      REFILL_BITS  = 32;                   // bits per 4-byte refill chunk

// ── Descriptor — matches Zig HuffDecChunkDesc in decode/driver.zig ──
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
static_assert(sizeof(HuffDecChunkDesc) == 20, "ABI: keep in sync with decode/driver.zig");

// LUT entry packing/unpacking (packLutEntry, lutSym1/lutSym2/lutSymPair/
// lutTotalLen/lutNumSyms) and the LUT_NUM_SYMS_* / LUT_MAX_SYMS_PER_STEP
// constants come from common/gpu_huffman.cuh.

// ── LUT build kernel ────────────────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (WARP_SIZE, 1, 1).
// One warp per block. Weights[] is HUFF_WEIGHTS_BYTES bytes; each byte
// has two nibbles — low nibble = symbol 2*i, high nibble = symbol 2*i+1.
// Pass 1: single-symbol fan-out (32-lane parallel over symbols).
// Pass 2: dual-symbol overwrite (32-lane parallel over s1; inner loop
// bounded by L1+L2<=10). Prefix-free codes give disjoint spans, so both
// passes are race-free with no atomics. Pass 3 escape stays lane-0.
//
// LUT is written to global memory (luts + desc.lut_offset).
extern "C" __global__ void slzHuffBuildLutKernel(
    const uint8_t* __restrict__ comp,
    const HuffDecChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ luts,
    const uint32_t* __restrict__ d_n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= *d_n_blocks) return;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const HuffDecChunkDesc desc = descs[block_id];
    const uint8_t* weights = comp + desc.in_offset;
    uint32_t* lut = luts + desc.lut_offset;

    // Unpack 128 B → 256 lengths into shared memory (one cache line of accesses).
    __shared__ uint8_t code_lengths[HUFF_ALPHABET];
    for (int i = lane; i < HUFF_WEIGHTS_BYTES; i += WARP_SIZE) {
        unpackWeightByte(weights[i], code_lengths[i * 2 + 0], code_lengths[i * 2 + 1]);
    }
    __syncwarp();

    // Compute canonical codes (RFC 1951 style — assigned in length-ascending
    // order, sym-ascending within each length). buildCanonicalCodes
    // (common/gpu_huffman.cuh) does the histogram + next_code + code
    // assignment serially in lane 0; all lanes then read the same codes[]
    // table for their slice of the fan-out work.
    __shared__ uint32_t codes[HUFF_ALPHABET];
    if (lane == 0) buildCanonicalCodes(code_lengths, codes);
    __syncwarp();

    // ── Build LUT in global memory ──
    // Zero the LUT — cooperative across warp.
    for (int i = lane; i < LUT_SIZE; i += WARP_SIZE) lut[i] = 0;
    __syncwarp();

    // Pass 1: single-symbol entries — parallel over symbols (32 lanes).
    // Canonical codes are prefix-free, so distinct symbols fill mutually
    // disjoint LUT spans — the fan-out is race-free with no atomics.
    for (int s = lane; s < HUFF_ALPHABET; s += WARP_SIZE) {
        int L = code_lengths[s];
        if (L == 0 || L > MAX_CODE_LEN) continue;  // length-11 -> escape pass
        uint32_t aligned = codes[s] << (MAX_CODE_LEN - L);
        uint32_t span = 1u << (MAX_CODE_LEN - L);
        uint32_t entry = packLutEntry((uint8_t)s, 0, (uint8_t)L, LUT_NUM_SYMS_SINGLE);
        for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
    }
    __syncwarp();

    // Pass 2: dual-symbol entries — parallel over s1 (32 lanes). Distinct
    // s1 prefixes fill disjoint spans, so the fan-out stays race-free.
    for (int s1 = lane; s1 < HUFF_ALPHABET; s1 += WARP_SIZE) {
        int L1 = code_lengths[s1];
        if (L1 == 0 || L1 >= MAX_CODE_LEN) continue;
        uint32_t C1 = codes[s1];
        for (int s2 = 0; s2 < HUFF_ALPHABET; s2++) {
            int L2 = code_lengths[s2];
            if (L2 == 0) continue;
            int total = L1 + L2;
            if (total > MAX_CODE_LEN) continue;
            uint32_t C2 = codes[s2];
            uint32_t aligned = (C1 << (MAX_CODE_LEN - L1))
                             | (C2 << (MAX_CODE_LEN - L1 - L2));
            uint32_t span = 1u << (MAX_CODE_LEN - total);
            uint32_t entry = packLutEntry((uint8_t)s1, (uint8_t)s2,
                                           (uint8_t)total, LUT_NUM_SYMS_DUAL);
            for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
        }
    }
    __syncwarp();

    // Pass 3: escape entries for length-(MAX_CODE_LEN+1) codes. A 10-bit
    // prefix can't resolve an 11-bit code; the entry stores both sibling
    // symbols (num_syms == LUT_NUM_SYMS_ESCAPE) and the decoder reads the
    // 11th bit. These 10-bit prefixes are never touched by passes 1-2
    // (prefix-free).
    if (lane == 0) {
        for (int s = 0; s < HUFF_ALPHABET; s++) {
            if (code_lengths[s] != MAX_CODE_LEN + 1) continue;
            uint32_t code = codes[s];
            uint32_t prefix = code >> 1;               // 10-bit prefix = LUT index
            uint32_t entry = lut[prefix];
            uint8_t lo_sym = lutSym1(entry);
            uint8_t hi_sym = lutSym2(entry);
            if (code & 1u) hi_sym = (uint8_t)s; else lo_sym = (uint8_t)s;
            lut[prefix] = packLutEntry(lo_sym, hi_sym,
                                       (uint8_t)(MAX_CODE_LEN + 1), LUT_NUM_SYMS_ESCAPE);
        }
    }
}

// ── Single-stream decode core (one lane) ────────────────────────────
// Top-aligned bit buffer; 4-byte big-endian refill; double-symbol (X2)
// LUT lookup.
__device__ __forceinline__ uint32_t decodeStreamOneLane(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    uint32_t bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;     // total bytes decoded
    uint32_t written = 0;     // bytes flushed to global
    uint64_t acc = 0;         // pending decoded bytes (byte 0 = oldest)
    uint32_t pending = 0;

    // Hot loop: 1 decode/iter, refill keeps >= 11 bits. Decoded bytes are
    // accumulated and flushed as 32-bit stores — the first <=3 flushes
    // byte-store to bring `out + written` to a 4-byte boundary, then the
    // rest are aligned 32-bit stores (cuts L2 store traffic ~4x).
    while (out_pos + LUT_MAX_SYMS_PER_STEP <= out_size) {
        if (bit_count < MAX_CODE_LEN + 1) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);
            bit_count += REFILL_BITS;
            in_pos += 4;
        }
        uint32_t entry = lut[(uint32_t)(bit_buf >> (BITBUF_BITS - MAX_CODE_LEN))];
        int total_len = lutTotalLen(entry);
        int num_syms  = lutNumSyms(entry);
        if (total_len == 0) return 0;
        if (num_syms == LUT_NUM_SYMS_ESCAPE) {  // escape: length-11 code, pick by the 11th bit
            uint32_t b11 = (uint32_t)((bit_buf >> (BITBUF_BITS - 1 - MAX_CODE_LEN)) & 1u);
            uint32_t sym = b11 ? lutSym2(entry) : lutSym1(entry);
            acc |= (uint64_t)sym << (pending * 8);
            pending += 1;
            out_pos += 1;
        } else {
            acc |= (uint64_t)lutSymPair(entry) << (pending * 8);
            pending += num_syms;
            out_pos += num_syms;
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
        while (pending >= sizeof(uint32_t)) {
            if (((uintptr_t)(out + written) & (alignof(uint32_t) - 1)) == 0u) {
                *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
                acc >>= 32;
                written  += (uint32_t)sizeof(uint32_t);
                pending  -= (uint32_t)sizeof(uint32_t);
            } else {
                out[written++] = (uint8_t)(acc & 0xFF);
                acc >>= 8;
                pending -= 1;
            }
        }
    }

    // Drain pending bytes; `written` catches up to `out_pos`.
    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending -= 1;
    }

    // Tail: handles the trailing 1-2 bytes the X2 hot loop leaves;
    // clamps dual entries to single.
    while (out_pos < out_size) {
        if (bit_count <= REFILL_BITS && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);
            bit_count += REFILL_BITS;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN + 1 && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (BITBUF_BITS - 8 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN + 1) bit_count = MAX_CODE_LEN + 1;
        uint32_t entry = lut[(uint32_t)(bit_buf >> (BITBUF_BITS - MAX_CODE_LEN))];
        int total_len = lutTotalLen(entry);
        int num_syms  = lutNumSyms(entry);
        if (total_len == 0) return 0;
        if (num_syms == LUT_NUM_SYMS_ESCAPE) {
            uint32_t b11 = (uint32_t)((bit_buf >> (BITBUF_BITS - 1 - MAX_CODE_LEN)) & 1u);
            out[out_pos++] = b11 ? lutSym2(entry) : lutSym1(entry);
        } else {
            out[out_pos++] = lutSym1(entry);
            if (num_syms == LUT_NUM_SYMS_DUAL && out_pos < out_size) {
                out[out_pos++] = lutSym2(entry);
            }
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── 4-stream decode kernel ──────────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (WARP_SIZE, 1, 1) — one warp per block.
// 32 lanes are launched for the cooperative LUT load; only the first 4
// lanes then decode (one stream each).
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
    const uint32_t* __restrict__ d_n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= *d_n_blocks) return;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const HuffDecChunkDesc desc = descs[block_id];

    // Cooperative LUT load into shared memory (LUT_SIZE × 4 = 1024 × 4 = 4 KB).
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += WARP_SIZE) shared_lut[i] = src_lut[i];
    __syncwarp();

    // Intentional early return: lanes 4..31 finish here. Safe because no
    // __syncwarp() follows — all remaining work is per-lane independent.
    // Do NOT add a warp barrier below this line; it would deadlock.
    if (lane >= HUFF_NUM_STREAMS) return;

    // Skip the 128 B weights to reach the 9 B sub-header.
    const uint8_t* hdr = comp + desc.in_offset + HUFF_WEIGHTS_BYTES;
    uint32_t stream_sizes[HUFF_NUM_STREAMS];
    #pragma unroll
    for (int s = 0; s < HUFF_NUM_STREAMS - 1; s++) {
        stream_sizes[s] = readLE24(hdr + s * HUFF_STREAM_SIZE_BYTES);
    }
    // Stream 3 size is the payload remainder. For a well-formed frame
    // in_size >= HUFF_BODY_HEADER_BYTES and the first three sizes sum to
    // <= total_streams, so the clamp below is a no-op; it only fires on a
    // malformed descriptor, where it prevents an unsigned-underflow OOB read.
    uint32_t total_streams = (desc.in_size >= (uint32_t)HUFF_BODY_HEADER_BYTES)
                           ? (desc.in_size - HUFF_BODY_HEADER_BYTES)
                           : 0u;
    uint32_t stored_sum = stream_sizes[0] + stream_sizes[1] + stream_sizes[2];
    stream_sizes[HUFF_NUM_STREAMS - 1] =
        (stored_sum <= total_streams) ? (total_streams - stored_sum) : 0u;

    // This lane's input pointer (exclusive prefix sum of sizes).
    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + HUFF_SUBHEADER_BYTES + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    // This lane's output region (4 contiguous quarters).
    uint32_t q = desc.out_size / HUFF_NUM_STREAMS;
    uint32_t my_out_start = (lane < HUFF_NUM_STREAMS - 1)
                          ? (lane * q)
                          : ((HUFF_NUM_STREAMS - 1) * q);
    uint32_t my_out_size  = (lane < HUFF_NUM_STREAMS - 1)
                          ? q
                          : (desc.out_size - (HUFF_NUM_STREAMS - 1) * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    (void)decodeStreamOneLane(my_in, my_in_size, shared_lut, my_out, my_out_size);
}
