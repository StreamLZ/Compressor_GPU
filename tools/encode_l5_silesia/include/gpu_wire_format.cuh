// ── StreamLZ GPU - shared LZ wire-format constants ──────────────────
// The contract between encode-side (lz_kernel.cu, assemble_kernel.cu)
// and decode-side (lz_kernel.cu, huffman_kernel.cu) kernels. Anything
// that the encoder writes and the decoder reads (or vice versa) lives
// here so the two paths cannot drift.
//
// Pure header: #pragma once, constants only, no kernels, no host code.
// Decoder-private helpers (parsers, descriptor structs) stay in
// decode/slz_wire_format.cuh; encoder-private hash constants stay in
// encode/lz_format.cuh.
#pragma once

#include <cstdint>

// ── Block geometry ──────────────────────────────────────────────────
// LZ matches and literal runs never cross a 64KB block boundary; off32
// offsets reset, and per-block trailing literals flush at each multiple
// of LZ_BLOCK_SIZE. A single sub-chunk decodes into at most
// MAX_BLOCKS_PER_SUBCHUNK such blocks (i.e. up to 128KB).
static constexpr uint32_t LZ_BLOCK_SIZE              = 0x10000u; // 64KB
static constexpr int      MAX_BLOCKS_PER_SUBCHUNK    = 2;

// First 8 bytes of the frame's first sub-chunk are raw literals copied
// straight to the output. INITIAL_RECENT_OFFSET is the matching seed
// value for `recent_offset` so the first token can use it.
static constexpr uint32_t INITIAL_LITERAL_COPY_BYTES = 8;
static constexpr int32_t  INITIAL_RECENT_OFFSET      = -8;

// ── Chunk types (chunk_header byte high nibble >> CHUNK_TYPE_SHIFT) ──
// On GPU we only emit raw (implicitly type 0, no named constant) and
// Huffman (type 4). The raw type is the default-zero value so no
// RAW_CHUNK_TYPE constant exists; code reads "ct == 0" inline.
static constexpr uint32_t CHUNK_TYPE_SHIFT      = 4;
static constexpr uint32_t CHUNK_TYPE_MASK       = 7;
static constexpr uint8_t  HUFF_CHUNK_TYPE       = 4;

// ── Sub-chunk header (3-byte big-endian) ────────────────────────────
// Bit 23      : LZ-compressed (set by encoder, gates LZ decode path)
// Bits 22..19 : decode mode (4 bits, identifies entropy form)
// Bits 18..0  : compressed size (19 bits = up to 512KB)
static constexpr uint32_t SUBCHUNK_HDR_BYTES        = 3;
static constexpr uint32_t SUBCHUNK_LZ_FLAG_BIT      = 0x800000u;
static constexpr uint32_t SUBCHUNK_MODE_SHIFT       = 19;
static constexpr uint32_t SUBCHUNK_MODE_MASK        = 0xFu;
static constexpr uint32_t SUBCHUNK_COMP_SIZE_MASK   = 0x7FFFFu;

// Field accessors over the 3-byte sub-chunk header (read as readBE24).
__device__ __forceinline__ bool     subchunkIsLz      (uint32_t hdr) { return (hdr & SUBCHUNK_LZ_FLAG_BIT) != 0; }
__device__ __forceinline__ uint32_t subchunkMode      (uint32_t hdr) { return (hdr >> SUBCHUNK_MODE_SHIFT) & SUBCHUNK_MODE_MASK; }
__device__ __forceinline__ uint32_t subchunkCompSize  (uint32_t hdr) { return hdr & SUBCHUNK_COMP_SIZE_MASK; }

// ── Off16 stream - entropy-coded marker ─────────────────────────────
// A leading count of OFF16_ENTROPY_MARKER tells the parser the off16
// stream is split into hi/lo halves (each its own entropy chunk).
static constexpr uint32_t OFF16_ENTROPY_MARKER      = 0xFFFFu;

// ── Off32 stream - packed count header ──────────────────────────────
// The 3-byte off32 header packs two 12-bit counts (block-0 in the high
// 12 bits, block-1 in the low 12 bits). A count field of
// OFF32_COUNT_PACK_MAX (= 4095) escapes to a following u16 carrying the
// real count.
static constexpr uint32_t OFF32_COUNT_FIELD_BITS    = 12;
static constexpr uint32_t OFF32_COUNT_PACK_MAX      = (1u << OFF32_COUNT_FIELD_BITS) - 1; // 4095

// Off32 entries are 3-byte triples by default. The "long-entry" tag byte
// (high byte of the third byte, value >= OFF32_LONG_ENTRY_TAG) flags a
// 4-byte entry carrying an extended 22-bit offset. The relationship to
// the encode-side mask is:
//   OFF32_LONG_ENTRY_TAG == (OFF32_LARGE_TAG >> 16)
// where OFF32_LARGE_TAG lives in encode/lz_format.cuh.
static constexpr uint8_t  OFF32_LONG_ENTRY_TAG      = 0xC0;

// ── Frame-level wire constants (ABI surface for the decoder) ────────
// Matched on the Zig side by `decode/driver.zig::FrameWalkStatus` and
// the frame writer in `encode/fast_framed.zig`. Currently only the
// walk-frame kernel consumes them on the GPU, but they are the
// encode/decode contract and so live in the shared header.
static constexpr uint32_t SLZ_FRAME_MAGIC           = 0x534C5A31u;
static constexpr uint8_t  SLZ_FRAME_VERSION         = 2;
static constexpr uint8_t  SLZ_CODEC_FAST_LZ         = 1; // Codec.fast_lz
static constexpr uint32_t SLZ_FRAME_MIN_HDR_SIZE    = 14;
static constexpr uint32_t SLZ_FRAME_END_MARK        = 0;

// Frame-header flag bits (byte 5 of the frame header).
static constexpr uint8_t  SLZ_FRAME_FLAG_CONTENT_SIZE_PRESENT = 0x01;
static constexpr uint8_t  SLZ_FRAME_FLAG_DICT_ID_PRESENT      = 0x02;

// Block-header bits (4-byte LE word at block start).
static constexpr uint32_t SLZ_BLOCK_UNCOMP_FLAG     = 0x80000000u;
static constexpr uint32_t SLZ_BLOCK_PDM_FLAG        = 0x40000000u;
static constexpr uint32_t SLZ_CHUNK_SIZE_MASK       = 0x3FFFFu;   // 256KB - 1
static constexpr uint32_t SLZ_CHUNK_TYPE_SHIFT      = 18;
static constexpr uint32_t SLZ_CHUNK_TYPE_MASK       = 3u << SLZ_CHUNK_TYPE_SHIFT;
static constexpr uint8_t  SLZ_INT_BLOCK_MAGIC       = 0x05;
static constexpr uint8_t  SLZ_BLOCK_HDR_MAGIC_MASK   = 0x0F; // low nibble of internal-header byte 0
static constexpr uint8_t  SLZ_BLOCK_HDR_DECODER_MASK = 0x7F; // low 7 bits of internal-header byte 1
static constexpr uint8_t  SLZ_BLOCK_HDR_CHECKSUM_FLAG = 0x80; // top bit of internal-header byte 1
static constexpr uint8_t  SLZ_DECODER_FAST          = 1;
static constexpr uint8_t  SLZ_DECODER_TURBO         = 2;

// Total source-chunk size in bytes (sc_group_size=1.0). Matches the CPU
// encoder's `constants.chunk_size`.
static constexpr uint32_t SLZ_CHUNK_SIZE_BYTES      = 262144u; // 256KB

// Chunk descriptor flag bits (SlzChunkDesc::flags).
static constexpr uint32_t CHUNK_FLAG_UNCOMPRESSED   = 0x1u;
static constexpr uint32_t CHUNK_FLAG_MEMSET         = 0x2u;

// Default sub-chunk slot size when caller passes 0. Equal to LZ_BLOCK_SIZE
// by coincidence; named separately because the semantic is "fallback
// scratch-slot stride for the scan kernel", not "block size".
static constexpr uint32_t DEFAULT_SUB_CHUNK_CAP     = 0x10000u; // 64KB
static_assert(DEFAULT_SUB_CHUNK_CAP == LZ_BLOCK_SIZE,
              "naming coincidence - both are 64KB; see comment above");

// LZ length-stream extended marker: bytes whose raw value exceeds
// EXT_LENGTH_THRESHOLD trigger a 2-byte extended-length read. Encoder
// emits against this; decoder reads it.
static constexpr uint32_t EXT_LENGTH_THRESHOLD       = 251;

// ── LZ-substream header sizes (encode side writes, decode side reads) ──
// LZ_SUBSTREAM_COUNT_HDR_BYTES = bytes for the per-substream count prefix
// (BE u24). STREAM_HEADER_BYTES is the old name retained for back-compat
// with the existing encode references; prefer the new name in new code.
static constexpr uint32_t LZ_SUBSTREAM_COUNT_HDR_BYTES = 3;
// DEPRECATED - use LZ_SUBSTREAM_COUNT_HDR_BYTES in new code.
static constexpr uint32_t STREAM_HEADER_BYTES       = LZ_SUBSTREAM_COUNT_HDR_BYTES;
static constexpr uint32_t OFF16_HEADER_BYTES        = 2; // little-endian u16

// ── 5-byte non-compact entropy chunk header (assemble_kernel.cu) ────
// Layout (encoder writes, decoder reads):
//   byte 0:           [type:4 | dm1_hi4:4]   - type in high nibble,
//                     top 4 bits of dst_size_minus_1 in low nibble
//   bytes 1..4 (BE):  [dm1_low14:14 | comp_size:18]
// where dst_size_minus_1 reconstructs as (dm1_hi4 << 14) | dm1_low14.
// The 32-bit BE word at bytes 1..4 (readU32BE) yields comp_size in the
// low 18 bits and dm1_low14 in bits 18..31.
static constexpr uint32_t ENTROPY_HDR_DM1_HIGH4_SHIFT = 14;
static constexpr uint32_t ENTROPY_HDR_DM1_HIGH_MASK   = 0xF;
static constexpr uint32_t ENTROPY_HDR_DM1_LOW_MASK    = 0x3FFF;
static constexpr uint32_t ENTROPY_HDR_COMP_BITS       = 18;
