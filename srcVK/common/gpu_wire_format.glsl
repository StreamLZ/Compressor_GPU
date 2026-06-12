// 1:1 port of src/common/gpu_wire_format.cuh.
// Shared device-side wire-format constants used by every encode + decode
// compute shader. Must agree byte-for-byte with srcVK/format/streamlz_constants.zig
// and srcVK/decode/slz_wire_format.glsl.

#ifndef SRCVK_COMMON_GPU_WIRE_FORMAT_GLSL
#define SRCVK_COMMON_GPU_WIRE_FORMAT_GLSL

// CUDA reference: src/common/gpu_wire_format.cuh:20-21. Block geometry —
// LZ matches and literal runs never cross a 64 KB block boundary. A
// single sub-chunk decodes into at most MAX_BLOCKS_PER_SUBCHUNK such
// blocks (up to 128 KB).
const uint LZ_BLOCK_SIZE              = 0x10000u; // 64 KB
const uint MAX_BLOCKS_PER_SUBCHUNK    = 2u;

// CUDA reference: src/common/gpu_wire_format.cuh:26-27. First 8 bytes of
// the frame's first sub-chunk are raw literals copied straight to the
// output; INITIAL_RECENT_OFFSET seeds `recent_offset`.
const uint INITIAL_LITERAL_COPY_BYTES = 8u;
const int  INITIAL_RECENT_OFFSET      = -8;

// CUDA reference: src/common/gpu_wire_format.cuh:33-35. Chunk-type field
// of the outer chunk header (raw chunks have no named constant; ct==0).
const uint CHUNK_TYPE_SHIFT           = 4u;
const uint CHUNK_TYPE_MASK            = 7u;
const uint HUFF_CHUNK_TYPE            = 4u;

// CUDA reference: src/common/gpu_wire_format.cuh:41-45. 3-byte
// big-endian sub-chunk header layout.
const uint SUBCHUNK_HDR_BYTES         = 3u;
const uint SUBCHUNK_LZ_FLAG_BIT       = 0x800000u;
const uint SUBCHUNK_MODE_SHIFT        = 19u;
const uint SUBCHUNK_MODE_MASK         = 0xFu;
const uint SUBCHUNK_COMP_SIZE_MASK    = 0x7FFFFu;

// CUDA reference: src/common/gpu_wire_format.cuh:55. A leading count of
// OFF16_ENTROPY_MARKER tells the parser the off16 stream is split into
// hi/lo halves (each its own entropy chunk).
const uint OFF16_ENTROPY_MARKER       = 0xFFFFu;

// CUDA reference: src/common/gpu_wire_format.cuh:62-71. Off32 stream —
// packed count header (two 12-bit count fields per chunk pair) + 4-byte
// long-entry tag.
const uint OFF32_COUNT_FIELD_BITS     = 12u;
const uint OFF32_COUNT_PACK_MAX       = 0xFFFu; // (1<<12) - 1
const uint OFF32_LONG_ENTRY_TAG       = 0xC0u;

// CUDA reference: src/common/gpu_wire_format.cuh:78-82. Frame-level wire
// constants — ABI surface for the decoder.
const uint SLZ_FRAME_MAGIC            = 0x534C5A31u;
const uint SLZ_FRAME_VERSION          = 2u;
const uint SLZ_CODEC_FAST_LZ          = 1u; // Codec.fast_lz
const uint SLZ_FRAME_MIN_HDR_SIZE     = 14u;
const uint SLZ_FRAME_END_MARK         = 0u;

// CUDA reference: src/common/gpu_wire_format.cuh:92-93. Frame-header
// flag bits (byte 5 of the frame header). DICT_ID was wrongly 0x02
// (the content-checksum bit) until the v4 #16 VK port wave - the same
// latent bug CUDA fixed when its walk kernel started parsing
// dictionary frames. Harmless while nothing tested the bit against a
// dict frame; catastrophic once the 4-byte skip depends on it.
const uint SLZ_FRAME_FLAG_CONTENT_SIZE_PRESENT = 0x01u;
const uint SLZ_FRAME_FLAG_DICT_ID_PRESENT      = 0x08u;
// v4 #20 frame-header flag bit 6: chunk-size table footer present.
const uint SLZ_FRAME_FLAG_CHUNK_TABLE          = 0x40u;

// CUDA reference: src/common/gpu_wire_format.cuh:96-111. Block-header
// bits (4-byte LE word at block start) + internal-block-header masks.
const uint SLZ_BLOCK_UNCOMP_FLAG      = 0x80000000u;
const uint SLZ_BLOCK_PDM_FLAG         = 0x40000000u;
const uint SLZ_CHUNK_SIZE_MASK        = 0x3FFFFu; // 256 KB - 1
const uint SLZ_CHUNK_TYPE_SHIFT       = 18u;
const uint SLZ_CHUNK_TYPE_MASK        = 3u << 18; // 3u << SLZ_CHUNK_TYPE_SHIFT
const uint SLZ_INT_BLOCK_MAGIC        = 0x05u;
const uint SLZ_BLOCK_HDR_MAGIC_MASK    = 0x0Fu;
const uint SLZ_BLOCK_HDR_DECODER_MASK  = 0x7Fu;
const uint SLZ_BLOCK_HDR_CHECKSUM_FLAG = 0x80u;
// Internal-header byte 0 high bits (above the magic nibble):
const uint SLZ_BLOCK_HDR_SC_FLAG       = 0x10u; // self-contained chunks (SC tail present)
const uint SLZ_INT_BLOCK_UNCOMP_FLAG   = 0x80u; // per-chunk uncompressed form: no 4-byte chunk hdr
const uint SLZ_DECODER_FAST           = 1u;
const uint SLZ_DECODER_TURBO          = 2u;

// CUDA reference: src/common/gpu_wire_format.cuh:103. Total source-chunk
// size in bytes (sc_group_size=1.0).
const uint SLZ_CHUNK_SIZE_BYTES       = 262144u; // 256 KB

// CUDA reference: src/common/gpu_wire_format.cuh:106-107. Chunk
// descriptor flag bits (SlzChunkDesc::flags).
const uint CHUNK_FLAG_UNCOMPRESSED    = 0x1u;
const uint CHUNK_FLAG_MEMSET          = 0x2u;

// CUDA reference: src/common/gpu_wire_format.cuh:112. Default sub-chunk
// slot size when caller passes 0. Equal to LZ_BLOCK_SIZE by coincidence;
// distinct naming for the "fallback scratch-slot stride" semantic.
const uint DEFAULT_SUB_CHUNK_CAP      = 0x10000u; // 64 KB

// CUDA reference: src/common/gpu_wire_format.cuh:119. LZ length-stream
// extended marker — bytes whose raw value exceeds EXT_LENGTH_THRESHOLD
// trigger a 2-byte extended-length read.
const uint EXT_LENGTH_THRESHOLD       = 251u;

// CUDA reference: src/common/gpu_wire_format.cuh:125-126. LZ substream
// count-prefix byte counts (BE u24 + LE u16 for off16).
const uint LZ_SUBSTREAM_COUNT_HDR_BYTES = 3u;
const uint OFF16_HEADER_BYTES         = 2u;

// CUDA reference: src/common/gpu_wire_format.cuh:136-139. Non-compact
// 5-byte entropy chunk header bit layout (assemble_kernel.cu writes,
// decoder reads).
const uint ENTROPY_HDR_DM1_HIGH4_SHIFT = 14u;
const uint ENTROPY_HDR_DM1_HIGH_MASK   = 0xFu;
const uint ENTROPY_HDR_DM1_LOW_MASK    = 0x3FFFu;
const uint ENTROPY_HDR_COMP_BITS       = 18u;

// CUDA reference: src/common/gpu_wire_format.cuh:48-50.
// Field accessors over the 3-byte sub-chunk header (read as readBE24).
bool subchunkIsLz(uint subchunk_hdr) {
    return (subchunk_hdr & SUBCHUNK_LZ_FLAG_BIT) != 0u;
}

uint subchunkMode(uint subchunk_hdr) {
    return (subchunk_hdr >> SUBCHUNK_MODE_SHIFT) & SUBCHUNK_MODE_MASK;
}

uint subchunkCompSize(uint subchunk_hdr) {
    return subchunk_hdr & SUBCHUNK_COMP_SIZE_MASK;
}

#endif
