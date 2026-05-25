// ── StreamLZ wire-format definitions ───────────────────────────
// Shared header for the GPU LZ decode kernel: token-format constants,
// the host-mirrored descriptor structs, the parsed-stream struct, and
// the device-side header parsers used by every decode path.
//
// Included into the single lz_kernel.cu translation unit (see that
// file's banner). Pure header — no kernels.
#pragma once

#include <cstdint>
#include "../common/gpu_warp.cuh"          // WARP_SIZE, LANE_MASK, FULL_WARP_MASK
#include "../common/gpu_byteio.cuh"        // readBE24
#include "../common/gpu_wire_format.cuh"   // LZ_BLOCK_SIZE, sub-chunk header, off16/off32, chunk types

// ── StreamLZ token format constants ────────────────────────────
// Tokens encode literal-length + match-length + offset type in one byte.
//   token >= 24:  standard — lit[2:0], match[6:3], use_recent[7], off16
//   token == 0:   long literal — length from length stream + 64
//   token == 1:   long near match — length from length stream + 91, off16
//   token == 2:   long far match — length from length stream + 29, off32
//   token 3-23:   short far match — match = token + 5, off32
static constexpr uint32_t TOKEN_SHORT_MIN      = 24;
static constexpr uint32_t TOKEN_LONG_LITERAL   = 0;
static constexpr uint32_t TOKEN_LONG_NEAR      = 1;
static constexpr uint32_t TOKEN_LONG_FAR       = 2;
static constexpr uint32_t LONG_LITERAL_BASE    = 64;
static constexpr uint32_t LONG_NEAR_BASE       = 91;
static constexpr uint32_t LONG_FAR_BASE        = 29;
static constexpr uint32_t SHORT_FAR_BASE       = 5;

// Standard-token bit-field extraction (token >= TOKEN_SHORT_MIN).
static constexpr uint32_t TOKEN_LIT_MASK       = 7;     // lit length = token & 7
static constexpr uint32_t TOKEN_MATCH_SHIFT    = 3;     // match nibble at bits 3..6
static constexpr uint32_t TOKEN_MATCH_MASK     = 0xF;   // match length = (token >> 3) & 0xF
static constexpr uint32_t TOKEN_USE_RECENT_SHIFT = 7;   // bit 7 = use recent offset
static constexpr uint32_t TOKEN_USE_RECENT_MASK  = 1;

// LZ_BLOCK_SIZE, MAX_BLOCKS_PER_SUBCHUNK, INITIAL_LITERAL_COPY_BYTES,
// INITIAL_RECENT_OFFSET come from ../common/gpu_wire_format.cuh — the
// encode/decode-shared format contract. Warp / lane constants
// (WARP_SIZE, LANE_MASK, FULL_WARP_MASK) come from ../common/gpu_warp.cuh.
//
// A warp-parallel match copy is only used when the match does not
// overlap the destination (match_dist >= match_len) and is long enough
// to be worth splitting across lanes.
static constexpr uint32_t MIN_PARALLEL_MATCH_LEN = 2;

// Extended length-stream encoding: a byte value > 251 carries a 2-byte
// uint16_t extension that is scaled by 4.
static constexpr uint32_t EXT_LENGTH_THRESHOLD   = 251;
static constexpr uint32_t EXT_LENGTH_EXTRA_BYTES = 2;
static constexpr uint32_t EXT_LENGTH_SCALE       = 4;

// Off16 entries are uint16_t; off32 entries are 3-byte triples.
static constexpr uint32_t OFF16_ENTRY_BYTES = 2;
static constexpr uint32_t OFF32_ENTRY_BYTES = 3;

// Stream-header bit layout: byte 0 high bit selects the short form
// (3-byte entropy / 2-byte type-0) vs the long form (5-byte / 3-byte).
// CHUNK_TYPE_SHIFT / CHUNK_TYPE_MASK come from ../common/gpu_wire_format.cuh.
static constexpr uint32_t HEADER_LONG_FORM_BIT  = 0x80;

// Type-0 (raw) header sizes.
static constexpr uint32_t TYPE0_SHORT_SIZE_MASK = 0xFFF;  // 12-bit size, short form

// Entropy chunk-header field layout. GPU emits only type 4 (Huffman); the
// same short/long format is also used by legacy types 1/6 that the decoder
// must still skip when reading older frames.
static constexpr uint32_t ENTROPY_HEADER_SHORT_BYTES = 3;
static constexpr uint32_t ENTROPY_HEADER_LONG_BYTES  = 5;
static constexpr uint32_t ENTROPY_SHORT_COMP_MASK    = 0x3FF;   // 10-bit comp size
static constexpr uint32_t ENTROPY_SHORT_DELTA_SHIFT  = 10;      // 10-bit dst-size delta
static constexpr uint32_t ENTROPY_SHORT_DELTA_MASK   = 0x3FF;
static constexpr uint32_t ENTROPY_LONG_SIZE_MASK     = 0x3FFFF; // 18-bit size field
static constexpr uint32_t ENTROPY_LONG_DELTA_SHIFT   = 18;      // dst-size reconstruction
static constexpr uint32_t ENTROPY_LONG_HI_SHIFT      = 14;

// Paired-stream header sizes.
//   paired-primary  : [0x70][countA:u24][inner type-6 stream]
//   paired-secondary: [0x50][countA:u24][countB:u24], no payload
static constexpr uint32_t PAIRED_PRIMARY_HEADER_BYTES   = 4;
static constexpr uint32_t PAIRED_SECONDARY_HEADER_BYTES = 7;

// Off16 stream: OFF16_ENTROPY_MARKER (0xFFFF in ../common/gpu_wire_format.cuh)
// signals an entropy-coded off16 pair (hi/lo split). The hi/lo halves
// live at these offsets in the per-sub-chunk off16 scratch slot.
static constexpr uint32_t OFF16_HILO_SPLIT_OFFSET = 65536;

// Off32 stream sizes header (3-byte LE field): count1 in the high 12
// bits, count2 in the low 12 bits. A 12-bit count of OFF32_COUNT_PACK_MAX
// (in ../common/gpu_wire_format.cuh) escapes to a following u16.
static constexpr uint32_t OFF32_COUNT1_SHIFT  = OFF32_COUNT_FIELD_BITS; // 12
static constexpr uint32_t OFF32_COUNT2_MASK   = OFF32_COUNT_PACK_MAX;   // 0xFFF

// SUBCHUNK_LZ_FLAG_BIT / SUBCHUNK_COMP_SIZE_MASK / SUBCHUNK_MODE_SHIFT /
// SUBCHUNK_MODE_MASK come from ../common/gpu_wire_format.cuh.

// Per-sub-chunk entropy-scratch slot size in bytes (128KB). One slot
// holds the lit / tok / off16 sub-buffers (see slzLzDecodeKernel).
static constexpr uint64_t ENTROPY_SCRATCH_SLOT_BYTES = 131072;

// Kernel launch geometry: 2 warps (64 threads) per block.
static constexpr uint32_t WARPS_PER_BLOCK            = 2;
static constexpr int      LZ_KERNEL_BLOCK_THREADS    = 64;
static constexpr int      LZ_KERNEL_MIN_BLOCKS_PER_SM = 24;

// ── Chunk descriptor ───────────────────────────────────────────
// ABI struct — must stay byte-identical to `pub const ChunkDesc` in
// src/gpu/decode/driver.zig (which uses an explicit `_pad: [3]u8`).
struct SlzChunkDesc {
    uint32_t src_offset;    // byte offset into compressed block
    uint32_t comp_size;     // compressed payload size
    uint32_t decomp_size;   // decompressed size (usually 256KB)
    uint32_t dst_offset;    // absolute output position
    uint32_t flags;         // bit 0: uncompressed, bit 1: memset
    uint8_t  memset_fill;   // fill byte for memset chunks
    uint8_t  _pad[3];       // alignment pad, keep in sync with Zig ChunkDesc
};
static_assert(sizeof(SlzChunkDesc) == 24, "ABI: keep in sync with decode/driver.zig");

// ── Raw off16 gather descriptor ────────────────────────────────
// ABI struct — must stay byte-identical to `RawOff16Desc` in
// src/gpu/decode/driver.zig.
struct SlzRawOff16Desc {
    uint32_t src_offset;   // byte offset into comp_base
    uint32_t size;         // bytes to copy
    uint32_t gpu_offset;   // byte offset into scratch_base
};
static_assert(sizeof(SlzRawOff16Desc) == 12, "ABI: keep in sync with decode/driver.zig");

// ── Parsed sub-chunk streams ───────────────────────────────────
// Filled by parseSubChunkHeaders, consumed by the decoders. Lives on
// the stack of parseAndDecodeSubChunk and is passed by reference; the
// __noinline__ header parser frees its registers before the hot loop.
//
// Pointer fields may point into the compressed blob OR into an entropy
// scratch buffer, depending on the matching *_is_pre_decoded flag at
// parse time (e.g. lit_ptr → compressed for type-0, → scratch otherwise).
struct ParsedStreams {
    const uint8_t* lit_ptr;       // literal-byte stream
    const uint8_t* cmd_ptr;       // token (command) stream
    const uint8_t* off16_raw;     // interleaved u16 off16 stream (non-split)
    const uint8_t* off16_hi;      // hi-byte half of a split off16 stream
    const uint8_t* off16_lo;      // lo-byte half of a split off16 stream
    const uint8_t* off32_raw1;    // off32 triples for LZ block 0
    const uint8_t* off32_raw2;    // off32 triples for LZ block 1
    const uint8_t* len_stream;    // extended-length side stream
    uint32_t lit_size;            // literal stream length in bytes
    uint32_t cmd_size;            // token stream length in bytes
    uint32_t off16_count;         // number of off16 entries
    uint32_t off16_split;         // 1 = hi/lo split format, 0 = interleaved u16
    uint32_t off32_count1;        // off32 entry count for LZ block 0
    uint32_t off32_count2;        // off32 entry count for LZ block 1
    uint32_t len_avail;           // bytes available in len_stream
    uint32_t cmd_stream2_offset;  // byte offset in cmd stream where block 1 begins
    uint32_t initial_copy;        // 8 if this sub-chunk did the initial copy, else 0
};

// readBE24 (24-bit big-endian read) comes from ../common/gpu_byteio.cuh.

// ── Extended length decode ─────────────────────────────────────
// Reads a variable-length count from the length stream. Uses a uint32
// offset from length_stream base to avoid 64-bit pointer register pressure.
__device__ inline uint32_t readLength(const uint8_t* length_stream,
                                      uint32_t& length_offset,
                                      uint32_t length_remaining) {
    if (length_offset >= length_remaining) return 0;
    uint32_t v = length_stream[length_offset];
    if (v > EXT_LENGTH_THRESHOLD) {
        if (length_offset + EXT_LENGTH_EXTRA_BYTES >= length_remaining) {
            length_offset++;
            return v;
        }
        const uint16_t extra = readU16LE(length_stream + length_offset + 1);
        v += (uint32_t)extra * EXT_LENGTH_SCALE;
        length_offset += EXT_LENGTH_EXTRA_BYTES;
    }
    length_offset++;
    return v;
}

// ── Type-0 (raw) stream-size parser ────────────────────────────
// Reads a 2- or 3-byte big-endian size prefix from the compressed
// stream, advances src past the prefix, and returns the raw size.
__device__ inline uint32_t parseRawStreamSize(const uint8_t*& src) {
    if (src[0] >= HEADER_LONG_FORM_BIT) {
        uint32_t sz = (((uint32_t)src[0] << 8) | src[1]) & TYPE0_SHORT_SIZE_MASK;
        src += 2;
        return sz;
    } else {
        uint32_t sz = readBE24(src);
        src += 3;
        return sz;
    }
}

// ── 3/5-byte entropy header parser ─────────────────────────────
// Parses the short (3-byte) or long (5-byte) header shared by every
// entropy-coded chunk type (GPU emits type 4 Huffman; decoder also
// accepts legacy types 1 and 6 in older frames). Advances src past the
// header + payload and returns the decompressed size; the compressed
// size is written to out_comp_size.
__device__ inline uint32_t parseEntropyHeader(const uint8_t*& src,
                                              uint32_t& out_comp_size) {
    uint32_t comp_size, dst_size;
    if (src[0] >= HEADER_LONG_FORM_BIT) {
        uint32_t bits = readBE24(src);
        comp_size = bits & ENTROPY_SHORT_COMP_MASK;
        dst_size = comp_size
                 + ((bits >> ENTROPY_SHORT_DELTA_SHIFT) & ENTROPY_SHORT_DELTA_MASK) + 1;
        src += ENTROPY_HEADER_SHORT_BYTES + comp_size;
    } else {
        uint32_t bits = readU32BE(src + 1);
        comp_size = bits & ENTROPY_LONG_SIZE_MASK;
        dst_size = ((((bits >> ENTROPY_LONG_DELTA_SHIFT)
                   | ((uint32_t)src[0] << ENTROPY_LONG_HI_SHIFT)) & ENTROPY_LONG_SIZE_MASK)) + 1;
        src += ENTROPY_HEADER_LONG_BYTES + comp_size;
    }
    out_comp_size = comp_size;
    return dst_size;
}

// ── Paired-primary inner-stream skip ───────────────────────────
// For a paired-primary header [0x70][countA:u24][inner type-6 stream]:
// reads countA, advances src past the whole marker + inner stream, and
// returns countA.
__device__ inline uint32_t skipPairedPrimary(const uint8_t*& src) {
    uint32_t count_a = readBE24(src + 1);
    const uint8_t* inner = src + PAIRED_PRIMARY_HEADER_BYTES;
    if (inner[0] >= HEADER_LONG_FORM_BIT) {
        uint32_t bits = readBE24(inner);
        src = inner + ENTROPY_HEADER_SHORT_BYTES + (bits & ENTROPY_SHORT_COMP_MASK);
    } else {
        uint32_t bits = readU32BE(inner + 1);
        src = inner + ENTROPY_HEADER_LONG_BYTES + (bits & ENTROPY_LONG_SIZE_MASK);
    }
    return count_a;
}

// ── Helper: skip an entropy-coded stream header + payload ──────
// Reads the header at *src, advances src past the header + payload, and
// returns the decompressed (or count) size for the stream's chunk type:
//   type 0    raw                          — returns raw size
//   type 4    Huffman                      — returns dst size (GPU encoder emits)
//   type 1, 2, 6  legacy entropy (decoder-skip-only) — returns dst size
//   type 7    paired-primary               — returns countA
//   type 5    paired-secondary             — returns countB
__device__ inline uint32_t skipEntropyStream(const uint8_t*& src) {
    uint32_t ct = (src[0] >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;
    if (ct == 0) {
        uint32_t sz = parseRawStreamSize(src);
        src += sz;
        return sz;
    } else if (ct == 1) {
        uint32_t comp_size;
        return parseEntropyHeader(src, comp_size);
    } else if (ct == 7) {
        return skipPairedPrimary(src);
    } else if (ct == 5) {
        // Paired-secondary: [0x50][countA:u24][countB:u24], no payload.
        uint32_t count_b = readBE24(src + 4);
        src += PAIRED_SECONDARY_HEADER_BYTES;
        return count_b;
    } else {
        // Huffman type 4 (GPU emits) or legacy entropy types 2/6 — parse header to skip.
        uint32_t comp_size;
        return parseEntropyHeader(src, comp_size);
    }
}
