// ── StreamLZ Vulkan decode pipeline: shared types + constants ──────
// GLSL header included by the 5 device-side decode-orchestration shells
// that mirror the CUDA decode pipeline kernels:
//
//   walk_frame.comp           ← src/decode/walk_frame_kernel.cuh
//   prefix_sum_chunks.comp    ← src/decode/prefix_sum_chunks_kernel.cuh
//   scan_parse.comp           ← src/decode/scan_parse_kernel.cuh
//   compact_huff_descs.comp   ← src/decode/compact_descs_kernels.cuh (huff)
//   compact_raw_descs.comp    ← src/decode/compact_descs_kernels.cuh (raw)
//   gather_raw_off16.comp     ← src/decode/gather_raw_off16_kernel.cuh
//
// Provides:
//   * Wire-format constants (mirror of src/common/gpu_wire_format.cuh)
//   * Decoder-private parser constants (mirror of decode/slz_wire_format.cuh)
//   * Frame-walk status codes (must match decode/driver.zig FrameWalkStatus)
//   * Descriptor structs as u32-strided slot layouts (the host-side
//     `extern struct ChunkDesc / HuffDecChunkDesc / RawOff16Desc /
//     ScanHuffDesc / ScanRawDesc` in src/decode/descriptors.zig +
//     src/decode/slz_wire_format.cuh +
//     src/decode/compact_descs_kernels.cuh).
//   * Helper functions for byte-addressed loads from u32-typed SSBOs
//     (the convention every existing Vulkan shader in this tree uses —
//     keeps the decoder portable to drivers without VK_KHR_8bit_storage
//     for the small descriptor / staged buffers; the dst byte buffer
//     in lz_decode.comp / lz_decode_raw.comp keeps its uint8_t[] form).
//   * Header parsers (parseEntropyHdrFields / parseType0HdrFields /
//     scanSkipStreamHeader / readBE24 / readU16LE / readU32LE) — exact
//     ports of the CUDA helpers in slz_wire_format.cuh + gpu_byteio.cuh.
//
// LAYOUT CONTRACT (must stay byte-identical to the CUDA ABI):
//
//   ChunkDesc          — 24 bytes / 6 u32 slots  (sizeof(SlzChunkDesc))
//   HuffDecChunkDesc   — 20 bytes / 5 u32 slots  (sizeof(SlzHuffDecChunkDesc))
//   RawOff16Desc       — 12 bytes / 3 u32 slots  (sizeof(SlzRawOff16Desc))
//   ScanHuffDesc       — 20 bytes / 5 u32 slots  (sizeof(SlzScanHuffDesc))
//   ScanRawDesc        — 16 bytes / 4 u32 slots  (sizeof(SlzScanRawDesc))
//
// We deliberately do NOT declare GLSL `struct ChunkDesc { uint a; uint b; ... };`
// blocks because the existing Vulkan codec idiom (lz_encode.comp +
// lz_decode.comp + assemble_*.comp + wire_format_gpu.zig) stores
// descriptor arrays as flat `uint d[]` SSBOs and uses named slot
// accessors. Mirroring that idiom keeps the new kernels uniform with
// the rest of the tree and avoids the std430 vs scalar layout-quirk
// minefield (scalar would require VK_EXT_scalar_block_layout, which
// the Tier-2 path on Vulkan 1.2 may not enable).
//
// Per-descriptor SLOT MACROS below give each field a name so kernel
// code reads `descs_buf.d[base + CHUNK_SRC_OFFSET_SLOT]` rather than
// open-coding the offset. Verified slot-by-slot against
// src/decode/descriptors.zig + slz_wire_format.cuh.

#ifndef STREAMLZ_DECODE_PIPELINE_SHARED_GLSL
#define STREAMLZ_DECODE_PIPELINE_SHARED_GLSL

// ── Block / chunk geometry (mirror src/common/gpu_wire_format.cuh) ──
const uint LZ_BLOCK_SIZE              = 0x10000u;   // 64 KiB
const uint MAX_BLOCKS_PER_SUBCHUNK    = 2u;
const uint INITIAL_LITERAL_COPY_BYTES = 8u;
const int  INITIAL_RECENT_OFFSET      = -8;

// ── Chunk type field (high nibble of chunk-header byte) ──────────────
// On the GPU decode path we only emit raw (ct == 0) and Huffman (ct ==
// HUFF_CHUNK_TYPE). Other types are rejected as malformed.
const uint CHUNK_TYPE_SHIFT           = 4u;
const uint CHUNK_TYPE_MASK            = 7u;
const uint HUFF_CHUNK_TYPE            = 4u;

// ── Sub-chunk header (3-byte BE u24) ────────────────────────────────
const uint SUBCHUNK_HDR_BYTES         = 3u;
const uint SUBCHUNK_LZ_FLAG_BIT       = 0x800000u;
const uint SUBCHUNK_MODE_SHIFT        = 19u;
const uint SUBCHUNK_MODE_MASK         = 0xFu;
const uint SUBCHUNK_COMP_SIZE_MASK    = 0x7FFFFu;

// ── Off16 / Off32 stream markers ────────────────────────────────────
const uint OFF16_ENTRY_BYTES          = 2u;
const uint OFF32_ENTRY_BYTES          = 3u;
const uint OFF16_ENTROPY_MARKER       = 0xFFFFu;
const uint OFF32_COUNT_FIELD_BITS     = 12u;
const uint OFF32_COUNT_PACK_MAX       = 0xFFFu;
const uint OFF32_LONG_ENTRY_TAG       = 0xC0u;

// ── Stream-header bit-field constants (decoder-private) ─────────────
// Mirror src/decode/slz_wire_format.cuh — these are the bit-field
// extraction constants the scan kernel + per-sub-chunk parsers use.
const uint HEADER_LONG_FORM_BIT       = 0x80u;
const uint TYPE0_SHORT_SIZE_MASK      = 0xFFFu;          // 12-bit size, short-form
const uint ENTROPY_HEADER_SHORT_BYTES = 3u;
const uint ENTROPY_HEADER_LONG_BYTES  = 5u;
const uint ENTROPY_SHORT_COMP_MASK    = 0x3FFu;          // 10-bit comp size
const uint ENTROPY_SHORT_DELTA_SHIFT  = 10u;
const uint ENTROPY_SHORT_DELTA_MASK   = 0x3FFu;
const uint ENTROPY_LONG_SIZE_MASK     = 0x3FFFFu;        // 18-bit size field
const uint ENTROPY_LONG_DELTA_SHIFT   = 18u;
const uint ENTROPY_LONG_HI_SHIFT      = 14u;

// ── Per-sub-chunk entropy-scratch slot ──────────────────────────────
// One slot holds the lit / tok / off16-hi / off16-lo bytes for one
// global sub-chunk. ENTROPY_SCRATCH_SLOT_BYTES is u64 on the CUDA side
// because it multiplies the global sub-chunk index in the per-warp LZ
// kernel; for the scan/compact/gather kernels here we only use it as a
// 32-bit stride so the u32 alias matches scan_parse_kernel.cuh's
// SCAN_SUBCHUNK_SLOT.
const uint ENTROPY_SCRATCH_SLOT_BYTES = 131072u;  // 128 KiB
const uint OFF16_HILO_SPLIT_OFFSET    = 65536u;   // = LZ_BLOCK_SIZE

// ── Frame-level wire constants (mirror gpu_wire_format.cuh) ─────────
// These drive the walk-frame kernel's header parse.
const uint SLZ_FRAME_MAGIC                          = 0x534C5A31u;
const uint SLZ_FRAME_VERSION                        = 2u;
const uint SLZ_CODEC_FAST_LZ                        = 1u;
const uint SLZ_FRAME_MIN_HDR_SIZE                   = 14u;
const uint SLZ_FRAME_END_MARK                       = 0u;
const uint SLZ_FRAME_FLAG_CONTENT_SIZE_PRESENT      = 0x01u;
const uint SLZ_FRAME_FLAG_DICT_ID_PRESENT           = 0x02u;
const uint SLZ_BLOCK_UNCOMP_FLAG                    = 0x80000000u;
const uint SLZ_BLOCK_PDM_FLAG                       = 0x40000000u;
const uint SLZ_CHUNK_SIZE_MASK                      = 0x3FFFFu;  // 256 KiB - 1
const uint SLZ_CHUNK_TYPE_SHIFT                     = 18u;
const uint SLZ_CHUNK_TYPE_MASK                      = (3u << 18); // bits 18..19
const uint SLZ_INT_BLOCK_MAGIC                      = 0x05u;
const uint SLZ_BLOCK_HDR_MAGIC_MASK                 = 0x0Fu;
const uint SLZ_BLOCK_HDR_DECODER_MASK               = 0x7Fu;
const uint SLZ_BLOCK_HDR_CHECKSUM_FLAG              = 0x80u;
const uint SLZ_DECODER_FAST                         = 1u;
const uint SLZ_DECODER_TURBO                        = 2u;
const uint SLZ_CHUNK_SIZE_BYTES                     = 262144u;   // 256 KiB

// ── Chunk-flag bits (SlzChunkDesc::flags) ───────────────────────────
const uint CHUNK_FLAG_UNCOMPRESSED    = 0x1u;
const uint CHUNK_FLAG_MEMSET          = 0x2u;

// Default sub-chunk slot stride when the caller passes 0. Equal to
// LZ_BLOCK_SIZE today but kept as a separate name (matches the comment
// on the CUDA constant).
const uint DEFAULT_SUB_CHUNK_CAP      = 0x10000u;

// ── Frame-walk kernel status codes ─────────────────────────────────
// MUST match decode/driver.zig FrameWalkStatus and the walk-frame CUDA
// kernel banner. Written into d_walk_meta[status] (= offset 20 within
// the 24-byte meta buffer).
const uint FW_STATUS_OK                       = 0u;
const uint FW_STATUS_BAD_MAGIC                = 1u;
const uint FW_STATUS_UNSUPPORTED_VERSION      = 2u;
const uint FW_STATUS_UNSUPPORTED_CODEC        = 3u;
const uint FW_STATUS_DICT_PRESENT             = 4u;
const uint FW_STATUS_PDM_OR_CHECKSUM          = 5u;
const uint FW_STATUS_BLOCK_HEADER_TRUNCATED   = 6u;
const uint FW_STATUS_BAD_INTERNAL_BLOCK_MAGIC = 7u;
const uint FW_STATUS_BAD_DECODER_TYPE         = 8u;
const uint FW_STATUS_CHUNK_HEADER_TRUNCATED   = 9u;
const uint FW_STATUS_BAD_CHUNK_TYPE           = 10u;
const uint FW_STATUS_CHUNK_BUFFER_OVERFLOW    = 11u;
const uint FW_STATUS_FRAME_TRUNCATED          = 12u;
const uint FW_STATUS_MULTI_BLOCK_UNSUPPORTED  = 13u;

// ── Per-context capacity bounds (mirror decode/descriptors.zig) ─────
const uint WALK_MAX_CHUNKS            = 16384u;
const uint MAX_SUB_CHUNKS_PER_CHUNK   = 4u;

// ── Walk-meta layout (u32 slot offsets, NOT byte offsets) ──────────
// d_walk_meta is a 6-u32 buffer (24 bytes). The walk kernel writes
// into it; downstream kernels (LZ decode self-gate) read n_chunks
// straight from the same buffer.
const uint WALK_META_N_CHUNKS_SLOT       = 0u;
const uint WALK_META_DECOMP_SIZE_SLOT    = 1u;
const uint WALK_META_SUB_CHUNK_CAP_SLOT  = 2u;
const uint WALK_META_BLOCK_START_SLOT    = 3u;
const uint WALK_META_BLOCK_SIZE_SLOT     = 4u;
const uint WALK_META_STATUS_SLOT         = 5u;
const uint WALK_META_U32_COUNT           = 6u;

// ── ChunkDesc — 6 u32 slots / 24 bytes ─────────────────────────────
// Mirrors `extern struct ChunkDesc` in src/decode/descriptors.zig
// and `SlzChunkDesc` in src/decode/slz_wire_format.cuh. The trailing
// 3-byte pad sits inside slot 5 with `memset_fill` (the low byte of
// slot 5).
const uint CHUNK_SRC_OFFSET_SLOT      = 0u;
const uint CHUNK_COMP_SIZE_SLOT       = 1u;
const uint CHUNK_DECOMP_SIZE_SLOT     = 2u;
const uint CHUNK_DST_OFFSET_SLOT      = 3u;
const uint CHUNK_FLAGS_SLOT           = 4u;
const uint CHUNK_MEMSET_FILL_SLOT     = 5u;   // low byte = memset_fill, upper 24b reserved (pad)
const uint CHUNK_DESC_U32_COUNT       = 6u;

// ── HuffDecChunkDesc — 5 u32 slots / 20 bytes ──────────────────────
// Mirrors `HuffDecChunkDesc` in descriptors.zig + huffman_kernel.cu.
// The compact kernels write into this layout.
const uint HUFFDESC_IN_OFFSET_SLOT    = 0u;
const uint HUFFDESC_IN_SIZE_SLOT      = 1u;
const uint HUFFDESC_OUT_OFFSET_SLOT   = 2u;
const uint HUFFDESC_OUT_SIZE_SLOT     = 3u;
const uint HUFFDESC_LUT_OFFSET_SLOT   = 4u;
const uint HUFFDESC_U32_COUNT         = 5u;

// ── RawOff16Desc — 3 u32 slots / 12 bytes ──────────────────────────
const uint RAWDESC_SRC_OFFSET_SLOT    = 0u;
const uint RAWDESC_SIZE_SLOT          = 1u;
const uint RAWDESC_GPU_OFFSET_SLOT    = 2u;
const uint RAWDESC_U32_COUNT          = 3u;

// ── ScanHuffDesc — 5 u32 slots / 20 bytes (staged) ─────────────────
// Mirrors `SlzScanHuffDesc` in compact_descs_kernels.cuh. The scan
// kernel writes valid=1 for the slots it filled; the compact kernel
// drops valid=0 entries.
const uint SCANHUFF_IN_OFFSET_SLOT    = 0u;
const uint SCANHUFF_IN_SIZE_SLOT      = 1u;
const uint SCANHUFF_OUT_OFFSET_SLOT   = 2u;
const uint SCANHUFF_OUT_SIZE_SLOT     = 3u;
const uint SCANHUFF_VALID_SLOT        = 4u;
const uint SCANHUFF_U32_COUNT         = 5u;

// ── ScanRawDesc — 4 u32 slots / 16 bytes (staged) ──────────────────
// Mirrors `SlzScanRawDesc` in compact_descs_kernels.cuh.
const uint SCANRAW_SRC_OFFSET_SLOT    = 0u;
const uint SCANRAW_SIZE_SLOT          = 1u;
const uint SCANRAW_GPU_OFFSET_SLOT    = 2u;
const uint SCANRAW_VALID_SLOT         = 3u;
const uint SCANRAW_U32_COUNT          = 4u;

// ── Huffman shared constants (decoder-side) ─────────────────────────
// HUFF_LUT_ENTRIES mirrors src/common/gpu_huffman.cuh. The compact
// huff-descs kernel writes lut_offset = k * HUFF_LUT_ENTRIES per the
// sequential-assignment contract.
const uint HUFF_LUT_ENTRIES           = 1024u;

// ────────────────────────────────────────────────────────────────────
//   Byte-addressed loads from u32-typed SSBOs
// ────────────────────────────────────────────────────────────────────
//
// Convention copied from lz_encode.comp / lz_decode.comp / assemble_*.comp:
// the compressed frame, prefix-bytes buffer, and entropy scratch are all
// u32-typed SSBOs (`buffer X { uint w[]; }`) on the host side because
// VK_KHR_8bit_storage is a Tier-2 feature and the Tier-1 NV path doesn't
// always honor it. Helpers below give callers a single named load per
// width and forward to the right gather.
//
// All loads are unchecked: caller must have verified the byte address
// is in-bounds against the parent buffer. Mirror of the CUDA helpers
// in src/common/gpu_byteio.cuh.

// Reads one byte from `buf` at byte-address `byte_pos`.
//   uint loadByteU32(buf, byte_pos)
//
// Pattern:
//   uint w = buf.w[byte_pos >> 2];
//   uint b = (w >> ((byte_pos & 3u) * 8u)) & 0xFFu;
//
// Because GLSL lacks templates and passing SSBO refs around is
// awkward, each kernel that needs byte loads against a fresh buffer
// declares a one-line wrapper around its own buffer:
//
//   uint loadFrameByte(uint p) { uint w = frame_buf.w[p >> 2]; return (w >> ((p & 3u) * 8u)) & 0xFFu; }
//
// (Same pattern lz_decode.comp:148 uses.) The helpers below operate on
// scalar uint values that the kernel has already extracted, so they
// can be shared without per-buffer instantiation.

// Little-endian 16-bit pack from two byte loads. Caller supplies the
// already-loaded byte values via two `loadByte` calls of its own.
uint readU16LEFromBytes(uint b0, uint b1) {
    return b0 | (b1 << 8);
}

// Little-endian 32-bit pack from four byte loads.
uint readU32LEFromBytes(uint b0, uint b1, uint b2, uint b3) {
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

// Big-endian 24-bit pack from three byte loads. Used for sub-chunk
// headers (readBE24).
uint readBE24FromBytes(uint b0, uint b1, uint b2) {
    return (b0 << 16) | (b1 << 8) | b2;
}

// Big-endian 32-bit pack from four byte loads. Used by the entropy
// long-form parser (readU32BE on bytes 1..4 of the 5-byte header).
uint readU32BEFromBytes(uint b0, uint b1, uint b2, uint b3) {
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
}

// ── Sub-chunk header field accessors ────────────────────────────────
// Apply to the result of readBE24 over the 3-byte sub-chunk header.
bool  subchunkIsLz       (uint hdr) { return (hdr & SUBCHUNK_LZ_FLAG_BIT) != 0u; }
uint  subchunkMode       (uint hdr) { return (hdr >> SUBCHUNK_MODE_SHIFT) & SUBCHUNK_MODE_MASK; }
uint  subchunkCompSize   (uint hdr) { return hdr & SUBCHUNK_COMP_SIZE_MASK; }

// ── Cursor-free header field parsers ────────────────────────────────
// Mirror parseType0HdrFields / parseEntropyHdrFields in slz_wire_format.cuh.
// The caller pre-loads the first `need` header bytes from its source
// buffer and packs them via the byte helpers above; these accessors
// decode the packed fields. The two struct returns are u32 packs:
//
//   parseType0HdrFields → ivec2(size, header_bytes)
//   parseEntropyHdrFields → ivec3(comp_size, dst_size, header_bytes)
//
// We pass the bytes already-loaded rather than taking a buffer ref so
// each call site can use its own SSBO without GLSL generic-callback
// gymnastics. Tradeoff: callers do the byte loads explicitly. Benefit:
// the parsers stay buffer-agnostic, which matches how scan_parse and
// walk_frame each have their own compressed-bytes buffer.
//
// short-form vs long-form selection: byte 0 >= HEADER_LONG_FORM_BIT
// (0x80) means SHORT form (3-byte entropy / 2-byte type-0), CLEAR means
// LONG form. This is named confusingly on the CUDA side too — see the
// comment in scan_parse_kernel.cuh:49.

// Returns (size, header_bytes). Caller supplies the first 3 header
// bytes (b0, b1, b2) — the long-form path uses all three; the short-
// form path uses only b0+b1 but the third byte is required for the
// long path and the call site has already loaded it.
uvec2 parseType0HdrFields(uint b0, uint b1, uint b2) {
    if (b0 >= HEADER_LONG_FORM_BIT) {
        // CUDA reference (slz_wire_format.cuh parseType0HdrFields):
        //   sz = (((u32)p[0] << 8) | p[1]) & TYPE0_SHORT_SIZE_MASK;
        // i.e. b0 is the HIGH byte of a 12-bit field packed BE in the
        // two header bytes; the size mask drops the long-form-bit. The
        // third byte b2 is unread in short form; the call site loads it
        // unconditionally because the long-form path always needs it.
        uint sz = (((b0 << 8) | b1) & TYPE0_SHORT_SIZE_MASK);
        return uvec2(sz, 2u);
    } else {
        return uvec2(readBE24FromBytes(b0, b1, b2), 3u);
    }
}

// Returns (comp_size, dst_size, header_bytes). Caller supplies the
// first 5 header bytes (b0..b4) — short form uses b0..b2, long form
// uses all five.
uvec3 parseEntropyHdrFields(uint b0, uint b1, uint b2, uint b3, uint b4) {
    if (b0 >= HEADER_LONG_FORM_BIT) {
        uint bits = readBE24FromBytes(b0, b1, b2);
        uint comp = bits & ENTROPY_SHORT_COMP_MASK;
        uint dst = comp
                 + ((bits >> ENTROPY_SHORT_DELTA_SHIFT) & ENTROPY_SHORT_DELTA_MASK)
                 + 1u;
        return uvec3(comp, dst, ENTROPY_HEADER_SHORT_BYTES);
    } else {
        // bytes 1..4 as a BE u32; comp_size = low 18 bits.
        uint bits = readU32BEFromBytes(b1, b2, b3, b4);
        uint comp = bits & ENTROPY_LONG_SIZE_MASK;
        uint dst  = (((bits >> ENTROPY_LONG_DELTA_SHIFT)
                    | (b0 << ENTROPY_LONG_HI_SHIFT)) & ENTROPY_LONG_SIZE_MASK) + 1u;
        return uvec3(comp, dst, ENTROPY_HEADER_LONG_BYTES);
    }
}

#endif // STREAMLZ_DECODE_PIPELINE_SHARED_GLSL
