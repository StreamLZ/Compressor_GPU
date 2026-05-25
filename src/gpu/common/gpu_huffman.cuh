// ── StreamLZ GPU — shared Huffman wire format & primitives ──────
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

// ── 4-stream wire-format constants ──────────────────────────────
// chunk_type=4 body layout:
//   [HUFF_WEIGHTS_BYTES weights — 4 bits/symbol, packed low-nibble-first]
//   [HUFF_SUBHEADER_BYTES sub-header — 3 × u24 LE stream sizes]
//   [stream 0 | stream 1 | stream 2 | stream 3]
static constexpr int     HUFF_NUM_STREAMS       = 4;            // 4-stream split
static constexpr int     HUFF_ALPHABET          = 256;          // 8-bit symbol alphabet
static constexpr int     HUFF_WEIGHTS_BYTES     = 128;          // 256 symbols × 4-bit lengths
static constexpr int     HUFF_STREAM_SIZE_BYTES = 3;            // bytes per u24 stream-size field
static constexpr int     HUFF_SUBHEADER_BYTES   = (HUFF_NUM_STREAMS - 1) * HUFF_STREAM_SIZE_BYTES; // 9
static constexpr int     HUFF_BODY_HEADER_BYTES = HUFF_WEIGHTS_BYTES + HUFF_SUBHEADER_BYTES;       // 137
static constexpr uint8_t HUFF_NIBBLE_MASK       = 0x0F;         // low-nibble mask

// ── Weights pack / unpack ───────────────────────────────────────
// 256 4-bit code lengths are stored in HUFF_WEIGHTS_BYTES bytes: byte i
// holds symbol 2*i in the low nibble and symbol 2*i+1 in the high
// nibble. packWeightByte / unpackWeightByte are exact inverses of one
// nibble layout — the encoder packs, the decoder unpacks.
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
// LUT builder and the encoder's table builder — both previously open-
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
//   bits  7:0  — sym1
//   bits 15:8  — sym2
//   bits 23:16 — total_len (bits consumed)
//   bits 31:24 — num_syms (1, 2, or LUT_NUM_SYMS_ESCAPE)
static constexpr int LUT_SYM1_SHIFT = 0;
static constexpr int LUT_SYM2_SHIFT = 8;
static constexpr int LUT_LEN_SHIFT  = 16;
static constexpr int LUT_NSYM_SHIFT = 24;

// num_syms field values in a packed LUT entry.
static constexpr uint8_t LUT_NUM_SYMS_SINGLE   = 1;   // single-symbol entry
static constexpr uint8_t LUT_NUM_SYMS_DUAL     = 2;   // double-symbol entry
static constexpr uint8_t LUT_NUM_SYMS_ESCAPE   = 3;   // height-limit escape entry
static constexpr int     LUT_MAX_SYMS_PER_STEP = 2;   // max symbols an X2 entry emits

__device__ __forceinline__ uint32_t packLutEntry(uint8_t sym1, uint8_t sym2,
                                                  uint8_t total_len, uint8_t num_syms) {
    return ((uint32_t)num_syms  << LUT_NSYM_SHIFT)
         | ((uint32_t)total_len << LUT_LEN_SHIFT)
         | ((uint32_t)sym2      << LUT_SYM2_SHIFT)
         | ((uint32_t)sym1      << LUT_SYM1_SHIFT);
}

// Field accessors mirroring packLutEntry — keep unpack centralized.
__device__ __forceinline__ uint8_t  lutSym1(uint32_t entry)     { return (uint8_t)(entry >> LUT_SYM1_SHIFT); }
__device__ __forceinline__ uint8_t  lutSym2(uint32_t entry)     { return (uint8_t)(entry >> LUT_SYM2_SHIFT); }
__device__ __forceinline__ uint16_t lutSymPair(uint32_t entry)  { return (uint16_t)(entry & 0xFFFFu); }
__device__ __forceinline__ int      lutTotalLen(uint32_t entry) { return (entry >> LUT_LEN_SHIFT)  & 0xFF; }
__device__ __forceinline__ int      lutNumSyms(uint32_t entry)  { return (entry >> LUT_NSYM_SHIFT) & 0xFF; }
