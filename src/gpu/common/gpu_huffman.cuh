// ── StreamLZ GPU - shared Huffman wire format & primitives ──────
// Shared by the two Huffman kernels (decode/huffman_kernel.cu and
// encode/huffman_kernel.cu). Pure header: #pragma once, constants +
// inline device helpers, no kernels and no translation-unit state.
// #include'd via "../common/gpu_huffman.cuh".
//
// This is the canonical-Huffman *format contract*: the encoder writes
// and the decoder reads exactly these constants and layouts. Keeping
// them in one header removes the split-brain hazard of two hand-kept
// copies that must agree byte-for-byte.
#pragma once

#include <cstdint>

// ── Code-height limit ───────────────────────────────────────────
// Canonical codes are height-limited to HUFF_MAX_CODE_LEN bits
// (the same limit the CPU Huffman encoder uses). The decode fast-LUT is
// indexed by HUFF_LUT_INDEX_BITS bits; an HUFF_MAX_CODE_LEN-bit code
// cannot be resolved by that prefix and instead uses an escape LUT
// entry (the decoder then reads one extra bit). The two constants are
// linked: the LUT index width is one bit less than the height limit.
static constexpr int      HUFF_MAX_CODE_LEN   = 11;
static constexpr int      HUFF_LUT_INDEX_BITS = HUFF_MAX_CODE_LEN - 1;   // 10
static constexpr uint32_t HUFF_LUT_ENTRIES    = 1u << HUFF_LUT_INDEX_BITS; // 1024

// Code-length histogram size: indices 0..HUFF_MAX_CODE_LEN+1 inclusive.
static constexpr int HUFF_LEN_HIST_SIZE = HUFF_MAX_CODE_LEN + 2;    // 13

// ── 32-stream wire-format constants ─────────────────────────────
// chunk_type=4 body layout, where N = HUFF_NUM_STREAMS = 32:
//   [HUFF_WEIGHTS_BYTES weights - 4 bits/symbol, packed low-nibble-first]
//   [HUFF_SUBHEADER_BYTES sub-header - (N-1) × u24 LE stream sizes;
//                                       stream (N-1) size derived from total]
//   [stream 0 | stream 1 | ... | stream N-1]
//
// 32 streams lets all 32 warp lanes decode in parallel (vs 4 lanes active
// in the prior 4-stream format). The cost has two parts:
//   - Sub-header byte overhead: (93-9) / chunk_size = 84 / 65536 ≈ 0.13%
//     at 64 KB sub-chunks.
//   - Stream-boundary entropy loss: 32 boundaries vs 4 means more
//     trailing-byte rounding + more partial-byte flushes. Empirically
//     this dominates the byte overhead.
// Total measured ratio cost ≈ 0.4-0.5 pp at 64 KB sub-chunks on
// enwik8 / silesia L3-L5 (see ratio table in src/gpu/README.md).
// At larger sub-chunks both components shrink linearly with chunk size.
static constexpr int     HUFF_NUM_STREAMS       = 32;           // 32-stream split (was 4)
static constexpr int     HUFF_ALPHABET          = 256;          // 8-bit symbol alphabet
static constexpr int     HUFF_WEIGHTS_BYTES     = 128;          // 256 symbols × 4-bit lengths
static constexpr int     HUFF_STREAM_SIZE_BYTES = 3;            // bytes per u24 stream-size field
static constexpr int     HUFF_SUBHEADER_BYTES   = (HUFF_NUM_STREAMS - 1) * HUFF_STREAM_SIZE_BYTES; // 93
static constexpr int     HUFF_BODY_HEADER_BYTES = HUFF_WEIGHTS_BYTES + HUFF_SUBHEADER_BYTES;       // 221
static constexpr uint8_t HUFF_NIBBLE_MASK       = 0x0F;         // low-nibble mask

// Both Huffman kernels assign one warp lane per stream (encoder lane k
// encodes stream k, decoder lane k decodes stream k). Bumping
// HUFF_NUM_STREAMS to anything other than the NVIDIA warp size (32)
// would require both kernels to be re-architected. Lock the invariant.
static_assert(HUFF_NUM_STREAMS == 32,
              "HUFF_NUM_STREAMS is hard-wired to one lane per warp; "
              "see slzHuffEncode4StreamKernel / slzHuffDecode4StreamKernel");

// ── Weights pack / unpack ───────────────────────────────────────
// 256 4-bit code lengths are stored in HUFF_WEIGHTS_BYTES bytes: byte i
// holds symbol 2*i in the low nibble and symbol 2*i+1 in the high
// nibble. packWeightByte / unpackWeightByte are exact inverses of one
// nibble layout - the encoder packs, the decoder unpacks.
__device__ __forceinline__ uint8_t packWeightByte(uint8_t len_even, uint8_t len_odd) {
    return (uint8_t)((len_even & HUFF_NIBBLE_MASK)
                   | ((len_odd & HUFF_NIBBLE_MASK) << 4));
}

__device__ __forceinline__ void unpackWeightByte(uint8_t b, uint8_t& len_even,
                                                  uint8_t& len_odd) {
    len_even = b & HUFF_NIBBLE_MASK;
    len_odd  = (b >> 4) & HUFF_NIBBLE_MASK;
}

// ── Canonical-Huffman code construction ─────────────────────────
// RFC-1951-style canonical code assignment: codes are assigned in
// (length, symbol) ascending order. One helper shared by the decoder's
// LUT builder and the encoder's table builder - both previously open-
// coded the identical algorithm. Called serially by one lane.
//
//   code_lengths : HUFF_ALPHABET entries; 0 = unused symbol.
//   codes        : HUFF_ALPHABET entries; written with the canonical
//                  code for each used symbol, 0 for unused symbols.
//
// All input code lengths are <= HUFF_MAX_CODE_LEN (the encoder height-
// limits; the decoder's weights come from such an encoder).
__device__ __forceinline__ void buildCanonicalCodes(const uint8_t* code_lengths,
                                                     uint32_t* codes) {
    uint32_t length_count[HUFF_LEN_HIST_SIZE] = {0};
    for (int s = 0; s < HUFF_ALPHABET; s++) {
        uint8_t L = code_lengths[s];
        if (L > 0) length_count[L]++;
    }
    length_count[0] = 0;

    uint32_t next_code[HUFF_LEN_HIST_SIZE] = {0};
    uint32_t code = 0;
    for (int L = 1; L <= HUFF_MAX_CODE_LEN; L++) {
        code = (code + length_count[L - 1]) << 1;
        next_code[L] = code;
    }

    for (int s = 0; s < HUFF_ALPHABET; s++) {
        int L = code_lengths[s];
        codes[s] = (L != 0) ? next_code[L]++ : 0u;
    }
}

// ── LUT entry packing / unpacking ───────────────────────────────
// Same layout as tools/huff_test/huff_ref.c pack_lut_entry():
//   bits  7:0  - sym1
//   bits 15:8  - sym2
//   bits 23:16 - total_len (bits consumed)
//   bits 31:24 - num_syms (1, 2, or LUT_NUM_SYMS_ESCAPE)
static constexpr int LUT_SYM1_SHIFT = 0;
static constexpr int LUT_SYM2_SHIFT = 8;
static constexpr int LUT_LEN_SHIFT  = 16;
static constexpr int LUT_NSYM_SHIFT = 24;

// num_syms field values in a packed LUT entry - these are wire-format
// tag values stored in the LUT's high byte, not symbol counts (though
// SINGLE=1 and DUAL=2 happen to match the count of symbols they emit).
// ESCAPE=3 is chosen so any non-{1,2} value triggers the escape branch;
// no encode-side meaning to the literal `3` beyond being distinct.
static constexpr uint8_t LUT_NUM_SYMS_SINGLE   = 1;   // single-symbol entry
static constexpr uint8_t LUT_NUM_SYMS_DUAL     = 2;   // double-symbol entry
static constexpr uint8_t LUT_NUM_SYMS_ESCAPE   = 3;   // height-limit escape entry

// Distinct from the tag values above: this is the maximum number of
// symbols any LUT entry can emit per decode step (an X2 entry emits 2).
// Used to size per-thread output buffers in the decode loop.
static constexpr int     LUT_MAX_SYMS_PER_STEP = 2;

__device__ __forceinline__ uint32_t packLutEntry(uint8_t sym1, uint8_t sym2,
                                                  uint8_t total_len, uint8_t num_syms) {
    return ((uint32_t)num_syms  << LUT_NSYM_SHIFT)
         | ((uint32_t)total_len << LUT_LEN_SHIFT)
         | ((uint32_t)sym2      << LUT_SYM2_SHIFT)
         | ((uint32_t)sym1      << LUT_SYM1_SHIFT);
}

// Field accessors mirroring packLutEntry - keep unpack centralized.
// The four single-field accessors (lutSym1/lutSym2/lutTotalLen/lutNumSyms)
// return uint8_t for type-consistency with packLutEntry's inputs;
// lutSymPair returns uint16_t because callers consume it as a packed
// (sym1 in low byte, sym2 in high byte) value, e.g. for `*(uint16_t*)dst = lutSymPair(e)`.
__device__ __forceinline__ uint8_t  lutSym1(uint32_t entry)     { return (uint8_t)(entry >> LUT_SYM1_SHIFT); }
__device__ __forceinline__ uint8_t  lutSym2(uint32_t entry)     { return (uint8_t)(entry >> LUT_SYM2_SHIFT); }
__device__ __forceinline__ uint16_t lutSymPair(uint32_t entry)  { return (uint16_t)(entry & 0xFFFFu); }
__device__ __forceinline__ uint8_t  lutTotalLen(uint32_t entry) { return (uint8_t)((entry >> LUT_LEN_SHIFT)  & 0xFF); }
__device__ __forceinline__ uint8_t  lutNumSyms(uint32_t entry)  { return (uint8_t)((entry >> LUT_NSYM_SHIFT) & 0xFF); }
