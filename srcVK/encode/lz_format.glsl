// 1:1 port of src/encode/lz_format.cuh.
// Encoder-private LZ format constants + per-chunk descriptor layout +
// hash-key helpers. Function definitions take pre-loaded scalar args
// (the SSBO byte-load lives in the caller) to sidestep GLSL's lack of
// SSBO-pointer parameters.
//

#ifndef SRCVK_ENCODE_LZ_FORMAT_GLSL
#define SRCVK_ENCODE_LZ_FORMAT_GLSL

#include "gpu_warp.glsl"
#include "gpu_byteio.glsl"
#include "gpu_wire_format.glsl"

// CUDA reference: src/encode/lz_format.cuh:18-21. LZ encode match /
// format constants.
const uint MIN_MATCH        = 4u;
const uint HASH_EMPTY       = 0xFFFFFFFFu;
const uint CHAIN_MAX_STEPS  = 8u;
const uint NEXT_HASH_SIZE   = 65536u;

// CUDA reference: src/encode/lz_format.cuh:28-47. Offset / length stream
// encoding bit layout.
const uint NEAR_OFFSET_MAX        = 0xFFFFu;
const uint OFF32_LARGE_TAG        = 0xC00000u;
const uint LARGE_OFFSET_THRESHOLD = OFF32_LARGE_TAG;
const uint OFF32_LOW22_MASK       = 0x3FFFFFu;
const uint OFF32_LOW_BITS         = 22u;
const uint LENGTH_INLINE_MAX      = 251u;
const uint LENGTH_EXT_TAG_BIAS    = 4u;

// CUDA reference: src/encode/lz_format.cuh:52-57. Chain-parser secondary
// hash + reusable hash-offset thresholds.
const uint LONG_HASH_TAG_BITS    = 6u;
const uint LONG_HASH_TAG_MASK    = 0x3Fu;
const uint NEXT_HASH_INDEX_MASK  = 0xFFFFu;
const uint MIN_HASH_MATCH_OFFSET = 8u;
const uint FAR_OFFSET_MIN_MATCH  = 14u;

// CUDA reference: src/encode/lz_format.cuh:65-73. Hash multipliers
// (Fibonacci + table-A). CUDA exports these as 64-bit constants
// (uint64_t). VK adaptation: GLSL has no 64-bit literal multiplication
// so the hash kernels do the 64-bit multiply with 32-bit primitives —
// scalar _LO/_HI halves feed umulExtended; the verbatim CUDA names
// are exported as uvec2(lo, hi) wrappers below.
const uint HASH_MUL_FIB_K6_LO = 0x7C150000u;
const uint HASH_MUL_FIB_K6_HI = 0x79B97F4Au;
const uint HASH_MUL_FIB_64_LO = 0x7F4A7C15u;
const uint HASH_MUL_FIB_64_HI = 0x9E3779B9u;
const uint HASH_MUL_A_LO      = 0x00000000u;
const uint HASH_MUL_A_HI      = 0xB7A56463u;

// Verbatim CUDA names — uvec2(lo, hi) pairs mirroring the CUDA uint64_t
// symbols. The hash helpers pull the halves out via .x/.y.
const uvec2 HASH_MUL_FIB_K6 = uvec2(HASH_MUL_FIB_K6_LO, HASH_MUL_FIB_K6_HI);
const uvec2 HASH_MUL_FIB_64 = uvec2(HASH_MUL_FIB_64_LO, HASH_MUL_FIB_64_HI);
const uvec2 HASH_MUL_A      = uvec2(HASH_MUL_A_LO,      HASH_MUL_A_HI);

// CUDA reference: src/encode/lz_format.cuh:68-73. Back-compat aliases
// for the prior names. CUDA exports these; mirrored here per 1:1 port.
const uvec2 FIB_HASH_MUL_K6 = HASH_MUL_FIB_K6;
const uvec2 HASH_A_MUL      = HASH_MUL_A;
const uvec2 FIB_HASH_MUL_64 = HASH_MUL_FIB_64;

// CUDA reference: src/encode/lz_format.cuh:79-86. Per-chunk LZ-pass
// descriptor. Mirrors CompressChunkDesc in srcVK/encode/encode_context.zig
// byte-for-byte (20 bytes). Backed by an SSBO of uint[5] entries.
const uint COMPRESS_CHUNK_DESC_U32_STRIDE = 5u;
const uint COMPRESS_CHUNK_DESC_OFF_SRC      = 0u;
const uint COMPRESS_CHUNK_DESC_OFF_SRC_SIZE = 1u;
const uint COMPRESS_CHUNK_DESC_OFF_DST      = 2u;
const uint COMPRESS_CHUNK_DESC_OFF_DST_CAP  = 3u;
const uint COMPRESS_CHUNK_DESC_OFF_FIRST    = 4u;

// CUDA reference: src/encode/lz_format.cuh:79-86. CompressChunkDesc
// struct, mirroring CUDA's struct CompressChunkDesc field order and
// types (std430 layout, 20 bytes).
struct CompressChunkDesc {
    uint src_offset;
    uint src_size;
    uint dst_offset;
    uint dst_capacity;
    uint is_first;
};

// CUDA reference: src/encode/assemble_kernel.cu:44-59. AssembleDesc layout
// constants. Backed by an SSBO of uint[13] entries (52 bytes).
const uint ASSEMBLE_DESC_U32_STRIDE = 13u;
const uint ASSEMBLE_DESC_OFF_RAW_OFFSET        = 0u;
const uint ASSEMBLE_DESC_OFF_RAW_SIZE          = 1u;
const uint ASSEMBLE_DESC_OFF_HUFF_LIT_OFFSET   = 2u;
const uint ASSEMBLE_DESC_OFF_HUFF_LIT_SIZE     = 3u;
const uint ASSEMBLE_DESC_OFF_HUFF_TOK_OFFSET   = 4u;
const uint ASSEMBLE_DESC_OFF_HUFF_TOK_SIZE     = 5u;
const uint ASSEMBLE_DESC_OFF_HUFF_OFF16HI_OFF  = 6u;
const uint ASSEMBLE_DESC_OFF_HUFF_OFF16HI_SIZE = 7u;
const uint ASSEMBLE_DESC_OFF_HUFF_OFF16LO_OFF  = 8u;
const uint ASSEMBLE_DESC_OFF_HUFF_OFF16LO_SIZE = 9u;
const uint ASSEMBLE_DESC_OFF_SUB_DECOMP_SIZE   = 10u;
const uint ASSEMBLE_DESC_OFF_INIT_BYTES        = 11u;
const uint ASSEMBLE_DESC_OFF_OUT_OFFSET        = 12u;

// CUDA reference: src/encode/lz_format.cuh:88-92. Chain-parser match
// result; not used by the L1 greedy path but the type is mirrored for
// symmetry with the CUDA header.
struct ChainMatch {
    int length;
    int offset;
};

// CUDA reference: src/encode/lz_format.cuh:99-103. k=6 hash: 8-byte key
// with HASH_MUL_FIB_K6 (Fibonacci, shifted by 16 for k=6). Returns
// `(word *% mult) >> (64 - hash_bits)` masked by `hash_mask`.
//
// VK adaptation: GLSL has no 64-bit multiply, so the 64×64→64 product is
// built from 32×32→64 partial products via `umulExtended`. word and the
// multiplier are passed as (lo, hi) uvec2 matching readU64LE's return
// shape.
uint hashKey6(uvec2 word8, uint hash_bits, uint hash_mask) {
    uint w_lo = word8.x;
    uint w_hi = word8.y;
    uint m_lo = HASH_MUL_FIB_K6_LO;
    uint m_hi = HASH_MUL_FIB_K6_HI;
    // Build the low 64 bits of word8 * mult, exactly mirroring CUDA's
    // `uint64_t product = word8 * mult;` (mod 2^64).
    uint p00_hi; uint p00_lo;
    umulExtended(w_lo, m_lo, p00_hi, p00_lo);
    // Add (w_lo*m_hi + w_hi*m_lo) << 32 into the high half.
    uint p01_lo = w_lo * m_hi;
    uint p10_lo = w_hi * m_lo;
    uint prod_lo = p00_lo;
    uint prod_hi = p00_hi + p01_lo + p10_lo;
    // shift = U64_BITS - hash_bits; CUDA path: hash_bits in [12..20], so
    // shift in [44..52] — always >= 32, so the result fits in the high
    // word shifted right by (shift - 32).
    uint shift = U64_BITS - hash_bits;
    uint idx = (shift >= 32u) ? (prod_hi >> (shift - 32u))
                              : ((prod_hi << (32u - shift)) | (prod_lo >> shift));
    return idx & hash_mask;
}

// CUDA reference: src/encode/lz_format.cuh:109-113. Hash-A: 8-byte key
// with HASH_MUL_A. Returns `(hi32 >> (32 - hash_bits)) & hash_mask`.
uint hashTableA(uint hash_bits, uint hash_mask, uvec2 at_src) {
    uint w_lo = at_src.x;
    uint w_hi = at_src.y;
    uint m_lo = HASH_MUL_A_LO;
    uint m_hi = HASH_MUL_A_HI;
    uint p00_hi; uint p00_lo;
    umulExtended(w_lo, m_lo, p00_hi, p00_lo);
    uint p01_lo = w_lo * m_hi;
    uint p10_lo = w_hi * m_lo;
    uint prod_hi = p00_hi + p01_lo + p10_lo;
    return (prod_hi >> (U32_BITS - hash_bits)) & hash_mask;
}

// CUDA reference: src/encode/lz_format.cuh:116-120. Hash-B: 8-byte key
// with HASH_MUL_FIB_64. Same shape as hashTableA, different multiplier.
uint hashTableB(uint hash_bits, uint hash_mask, uvec2 at_src) {
    uint w_lo = at_src.x;
    uint w_hi = at_src.y;
    uint m_lo = HASH_MUL_FIB_64_LO;
    uint m_hi = HASH_MUL_FIB_64_HI;
    uint p00_hi; uint p00_lo;
    umulExtended(w_lo, m_lo, p00_hi, p00_lo);
    uint p01_lo = w_lo * m_hi;
    uint p10_lo = w_hi * m_lo;
    uint prod_hi = p00_hi + p01_lo + p10_lo;
    return (prod_hi >> (U32_BITS - hash_bits)) & hash_mask;
}

// CUDA reference: src/encode/lz_format.cuh:123-127. Hash-B tag — the
// full hi32 result; caller masks with LONG_HASH_TAG_MASK.
uint hashTagB(uvec2 at_src) {
    uint w_lo = at_src.x;
    uint w_hi = at_src.y;
    uint m_lo = HASH_MUL_FIB_64_LO;
    uint m_hi = HASH_MUL_FIB_64_HI;
    uint p00_hi; uint p00_lo;
    umulExtended(w_lo, m_lo, p00_hi, p00_lo);
    uint p01_lo = w_lo * m_hi;
    uint p10_lo = w_hi * m_lo;
    uint prod_hi = p00_hi + p01_lo + p10_lo;
    return prod_hi;
}

#endif
