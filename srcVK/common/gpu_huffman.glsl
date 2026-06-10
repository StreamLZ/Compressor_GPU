// 1:1 port of src/common/gpu_huffman.cuh.
// Shared Huffman wire-format constants: HUFF_NUM_STREAMS,
// HUFF_BODY_HEADER_BYTES, HUFF_*. L2 stub: constants live in this header
// (the Zig host encode_huff.zig references them at compile time); the
// device-side bodies of the Huffman shaders are stubbed.

#ifndef SRCVK_COMMON_GPU_HUFFMAN_GLSL
#define SRCVK_COMMON_GPU_HUFFMAN_GLSL

// CUDA reference: src/common/gpu_huffman.cuh:22-24. Canonical-Huffman
// code-height limit + LUT index width. The two constants are linked:
// the LUT index width is one bit less than the height limit.
const int  HUFF_MAX_CODE_LEN   = 11;
const int  HUFF_LUT_INDEX_BITS = HUFF_MAX_CODE_LEN - 1;   // 10
const uint HUFF_LUT_ENTRIES    = 1u << HUFF_LUT_INDEX_BITS; // 1024

// CUDA reference: src/common/gpu_huffman.cuh:27. Code-length histogram
// size: indices 0..HUFF_MAX_CODE_LEN+1 inclusive.
const int HUFF_LEN_HIST_SIZE = HUFF_MAX_CODE_LEN + 2; // 13

// CUDA reference: src/common/gpu_huffman.cuh:56-64. 32-stream
// bounded-interleaved (BIL) wire-format constants. 32 streams = one
// warp lane per stream.
const int  HUFF_NUM_STREAMS       = 32;
const int  HUFF_ALPHABET          = 256;
const int  HUFF_WEIGHTS_BYTES     = 128; // 256 symbols × 4-bit lengths
const int  HUFF_STREAM_SIZE_BYTES = 3;   // bytes per u24 stream-size field
const int  HUFF_SUBHEADER_BYTES   = HUFF_NUM_STREAMS * HUFF_STREAM_SIZE_BYTES;       // 96
const int  HUFF_BIL_K_BYTES       = 4;   // u32 LE interleaved word count K
const int  HUFF_BODY_HEADER_BYTES = HUFF_WEIGHTS_BYTES + HUFF_SUBHEADER_BYTES + HUFF_BIL_K_BYTES; // 228
const int  HUFF_BIL_ROW_BYTES     = HUFF_NUM_STREAMS * 4; // 128 — one interleaved row
const uint HUFF_NIBBLE_MASK       = 0x0Fu;

// CUDA reference: src/common/gpu_huffman.cuh:130-133. LUT entry packing
// bit shifts.
const int LUT_SYM1_SHIFT = 0;
const int LUT_SYM2_SHIFT = 8;
const int LUT_LEN_SHIFT  = 16;
const int LUT_NSYM_SHIFT = 24;

// CUDA reference: src/common/gpu_huffman.cuh:140-142. num_syms field
// values in a packed LUT entry (wire-format tag values; SINGLE=1 and
// DUAL=2 happen to match the count of symbols they emit, ESCAPE=3 is
// chosen distinct).
const uint LUT_NUM_SYMS_SINGLE  = 1u;
const uint LUT_NUM_SYMS_DUAL    = 2u;
const uint LUT_NUM_SYMS_ESCAPE  = 3u;

// CUDA reference: src/common/gpu_huffman.cuh:147. Maximum symbols any
// LUT entry can emit per decode step (X2 entries emit 2).
const int LUT_MAX_SYMS_PER_STEP = 2;

// ── Helper-function bodies are L2-stubbed ─────────────────────────────
// CUDA reference: src/common/gpu_huffman.cuh:79-88 (pack/unpackWeightByte),
// :102-122 (buildCanonicalCodes), :149-166 (packLutEntry + lut* accessors).
// Bodies will land alongside the Huffman kernel fleshout in Phase 9.
// The constants above are sufficient for the Phase-1 host references in
// encode_huff.zig + the Huffman .comp shells (which are SHELL stubs
// until L2).

#endif
