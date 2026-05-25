// ── StreamLZ GPU LZ encode — format constants & hash primitives ──
// Pure format/hash primitives for the LZ encode kernel: stream-format
// constants, the per-chunk descriptor, hash-key helpers and the
// zero-padded tail read. No parser logic lives here.
//
// Included by lz_kernel.cu — see that file for the build line.
#pragma once

#include <cstdint>
#include "../common/gpu_warp.cuh"          // WARP_SIZE, LANE_MASK, U32_BITS, U64_BITS, BITS_PER_BYTE
#include "../common/gpu_byteio.cuh"        // read8safe
#include "../common/gpu_wire_format.cuh"   // LZ_BLOCK_SIZE, INITIAL_LITERAL_COPY_BYTES, OFF32_LONG_ENTRY_TAG, ...

// ── Match / format constants ────────────────────────────────────
// LZ_BLOCK_SIZE (the 64KB block boundary) and INITIAL_LITERAL_COPY_BYTES
// (the 8-byte verbatim prefix on the first chunk) come from
// ../common/gpu_wire_format.cuh — the encode/decode-shared contract.
static constexpr uint32_t MIN_MATCH        = 4;            // shortest LZ match
static constexpr uint32_t HASH_EMPTY       = 0xFFFFFFFFu;  // empty greedy hash-table slot
static constexpr uint32_t CHAIN_MAX_STEPS  = 8;            // chain-parser walk depth
static constexpr uint32_t NEXT_HASH_SIZE   = 65536;        // 2^16 entries of uint16_t (matches CPU c_bits=16)

// Warp layout: the kernel runs exactly one warp per chunk. The warp /
// bit-width constants (WARP_SIZE, LANE_MASK, U32_BITS, U64_BITS,
// BITS_PER_BYTE) come from ../common/gpu_warp.cuh.

// Offset / length stream encoding (see lz_token_emit.cuh).
static constexpr uint32_t NEAR_OFFSET_MAX = 0xFFFFu;        // off16 vs off32 split
static constexpr uint32_t LARGE_OFFSET_THRESHOLD = 0xC00000u;  // 4-byte extended off32 form
static constexpr uint32_t OFF32_LARGE_TAG  = 0xC00000u;     // tag bits OR'd into the truncated offset
static constexpr uint32_t OFF32_LOW22_MASK = 0x3FFFFFu;     // 22-bit low offset mask
static constexpr uint32_t OFF32_LOW_BITS   = 22;            // low-bit width of the extended offset
static constexpr uint32_t LENGTH_INLINE_MAX  = 251;  // max single-byte length value
static constexpr uint32_t LENGTH_EXT_TAG_BIAS = 4;   // tag bias for the 3-byte extended length

// Chain-parser secondary-table tag packed into long_hash.
static constexpr uint32_t LONG_HASH_TAG_BITS = 6;            // tag occupies the low 6 bits
static constexpr uint32_t LONG_HASH_TAG_MASK = 0x3Fu;        // (1 << LONG_HASH_TAG_BITS) - 1
static constexpr uint32_t NEXT_HASH_INDEX_MASK = 0xFFFFu;    // next_hash modular index width

static constexpr uint32_t MIN_HASH_MATCH_OFFSET = 8;   // smallest reusable hash offset (offset-8 is the fallback)
static constexpr uint32_t FAR_OFFSET_MIN_MATCH  = 14;  // CPU mmlt: minimum match length threshold for far offsets

// ── Fibonacci hash multipliers ──────────────────────────────────
static constexpr uint64_t FIB_HASH_MUL_K6 = 0x79B97F4A7C150000ULL;  // k=6 shifted Fibonacci multiplier
static constexpr uint64_t HASH_A_MUL       = 0xB7A5646300000000ULL;  // Hash-A 64-bit multiplier
static constexpr uint64_t FIB_HASH_MUL_64  = 0x9E3779B97F4A7C15ULL;  // 64-bit Fibonacci multiplier

// ── Per-chunk LZ-pass descriptor ────────────────────────────────
// ABI-mirrored by CompressChunkDesc in src/gpu/encode/driver.zig — do
// not reorder fields or change types. is_first triggers the 8-byte
// INITIAL_COPY verbatim prefix at the start of the output.
struct CompressChunkDesc {
    uint32_t src_offset;
    uint32_t src_size;
    uint32_t dst_offset;
    uint32_t dst_capacity;
    uint32_t is_first;
};
static_assert(sizeof(CompressChunkDesc) == 20, "ABI: keep in sync with encode/driver.zig");

// Match result returned by findMatchChain.
struct ChainMatch {
    int32_t length;
    int32_t offset;  // 0 = recent-offset reuse, >0 = explicit distance
};

// k=6 hash: matches CPU FastMatchHasher.init for hasher_k=6.
// CPU's mult is fibonacci_hash_multiplier << ((8-6)*8) = 0x9E3779B97F4A7C15ULL << 16.
// Effective value (u64 wrap): FIB_HASH_MUL_K6 = 0x79B97F4A7C150000ULL.
// Then index = (word *% mult) >> (64 - hash_bits).
// Text inputs use k=6 to reduce hash collisions (groups 6-byte sequences,
// not just 4-byte). Same buckets as CPU runGreedyParser on text data.
__device__ uint32_t hashKey6(uint64_t word8, uint32_t hash_bits, uint32_t hash_mask) {
    uint64_t product = word8 * FIB_HASH_MUL_K6;
    return (uint32_t)(product >> (U64_BITS - hash_bits)) & hash_mask;
}

// read8safe (read up to 8 bytes, zero-padding past src_size) comes
// from ../common/gpu_byteio.cuh.

// Hash-A: 8-byte key with 64-bit multiply, matching CPU MatchHasher2.
__device__ uint32_t hashTableA(uint32_t hash_bits, uint32_t hash_mask, uint64_t at_src) {
    uint64_t product = HASH_A_MUL * at_src;
    uint32_t hi32 = (uint32_t)(product >> U32_BITS);
    return (hi32 >> (U32_BITS - hash_bits)) & hash_mask;
}

// Hash-B: 8-byte key with Fibonacci 64-bit multiply, matching CPU.
__device__ uint32_t hashTableB(uint32_t hash_bits, uint32_t hash_mask, uint64_t at_src) {
    uint64_t product = FIB_HASH_MUL_64 * at_src;
    uint32_t hi32 = (uint32_t)(product >> U32_BITS);
    return (hi32 >> (U32_BITS - hash_bits)) & hash_mask;
}

// Hash-B tag: full 32-bit hash value, caller uses & LONG_HASH_TAG_MASK.
__device__ uint32_t hashTagB(uint64_t at_src) {
    uint64_t product = FIB_HASH_MUL_64 * at_src;
    uint32_t hi32 = (uint32_t)(product >> U32_BITS);
    return hi32;
}
