//! Host-side SLZ1 frame walk for the host-input L1 decode path.
//!
//! Mirrors CUDA's host-input shape at
//! `src/decode/streamlz_decoder.zig:215-317 (decompressFrameInner)` and
//! `src/decode/streamlz_decoder.zig:386-481 (buildChunkDescriptors)`.
//! In CUDA the CPU parses the SLZ1 frame header + per-block headers, then
//! `buildChunkDescriptors` fills a stack-allocated array of `ChunkDesc`
//! (6 u32 each — frame-level fields the LZ kernel consumes). Only the
//! `block_payload` and the `chunk_descs[]` ever cross the H2D boundary.
//! No GPU walk_frame, no GPU l1_unwrap, no scan/compact/gather kernels.
//!
//! VK additionally requires a second descriptor array because the
//! `lz_decode.comp` reader (binding 5) consumes a 16-u32 per-chunk record
//! with byte offsets into each of the cmd / lit / off16 / length / off32
//! streams. CUDA's `lz_decode_raw` kernel does that per-chunk
//! sub-chunk-header parse INSIDE the warp on-device (every CUDA warp
//! re-parses its own chunk's sub-chunk headers in registers); the VK
//! shader was rewritten to consume pre-computed byte bases produced by
//! `l1_unwrap.comp`. To keep the VK shader unchanged while skipping
//! `l1_unwrap` on host-input, this module also writes those 16-u32 byte
//! bases CPU-side.
//!
//! Together the two outputs are exactly what `walk_frame.comp` (binding 7
//! / WalkChunks) and `l1_unwrap.comp` (binding 5 / ChunkDescs) would have
//! produced on the D2D path. The byte-base values are relative to the
//! start of `block_payload` (which is what gets uploaded to the
//! pipeline_result.frame buffer on this path, mirroring CUDA's
//! `req.compressed_block` parameter at
//! `src/decode/decode_dispatch.zig:565-571`).
//!
//! Single-block SLZ1 frames only — same constraint as
//! `wire_format.unwrapSlz1ToL1Streams`. The GPU encoder always emits a
//! single block; multi-block / Huffman frames are an L≥3 concern that
//! routes through a different decode path (NOT the host-input L1 entry).

const std = @import("std");

const wire_constants = @import("wire_constants.zig");

// Frame-format helpers from the CPU codec. Reached via the
// per-target root files (e.g. `cli_vk_root.zig`) that widen the
// package boundary the same way `wire_format.zig` and
// `wire_format_gpu.zig` already do.
const frame = @import("../src/format/frame_format.zig");
const block_header = @import("../src/format/block_header.zig");
const constants = @import("../src/format/streamlz_constants.zig");

const LZ_BLOCK_SIZE: u32 = wire_constants.LZ_BLOCK_SIZE;
const INITIAL_LITERAL_COPY_BYTES: u32 = wire_constants.INITIAL_LITERAL_COPY_BYTES;
const SUBCHUNK_HDR_BYTES: u32 = wire_constants.SUBCHUNK_HDR_BYTES;
const SUBCHUNK_LZ_FLAG_BIT: u32 = wire_constants.SUBCHUNK_LZ_FLAG_BIT;
const SUBCHUNK_COMP_SIZE_MASK: u32 = wire_constants.SUBCHUNK_COMP_SIZE_MASK;
const OFF16_ENTRY_BYTES: u32 = wire_constants.OFF16_ENTRY_BYTES;
const OFF32_ENTRY_BYTES: u32 = wire_constants.OFF32_ENTRY_BYTES;
const OFF32_COUNT_FIELD_BITS: u5 = wire_constants.OFF32_COUNT_FIELD_BITS;
const OFF32_COUNT_PACK_MAX: u32 = wire_constants.OFF32_COUNT_PACK_MAX;
const SC_TAIL_PER_CHUNK_BYTES: u32 = wire_constants.SC_TAIL_PER_CHUNK_BYTES;

// Slot constants — mirror `wire_constants.zig` and the shared GLSL
// header `decode_pipeline_shared.glsl`. The 6-u32 WalkChunks layout
// uses CHUNK_SRC_OFFSET_SLOT/CHUNK_COMP_SIZE_SLOT/CHUNK_DECOMP_SIZE_SLOT/
// CHUNK_DST_OFFSET_SLOT/CHUNK_FLAGS_SLOT/CHUNK_MEMSET_FILL_SLOT.
const CHUNK_DESC_U32_COUNT: u32 = wire_constants.CHUNK_DESC_U32_COUNT;
const CHUNK_SRC_OFFSET_SLOT: u32 = wire_constants.CHUNK_SRC_OFFSET_SLOT;
const CHUNK_COMP_SIZE_SLOT: u32 = wire_constants.CHUNK_COMP_SIZE_SLOT;
const CHUNK_DECOMP_SIZE_SLOT: u32 = wire_constants.CHUNK_DECOMP_SIZE_SLOT;
const CHUNK_DST_OFFSET_SLOT: u32 = wire_constants.CHUNK_DST_OFFSET_SLOT;
const CHUNK_FLAGS_SLOT: u32 = wire_constants.CHUNK_FLAGS_SLOT;
const CHUNK_MEMSET_FILL_SLOT: u32 = wire_constants.CHUNK_MEMSET_FILL_SLOT;
const CHUNK_FLAG_UNCOMPRESSED: u32 = wire_constants.CHUNK_FLAG_UNCOMPRESSED;

pub const HostWalkError = error{
    BadFrame,
    BadHeader,
    UnknownDictionary,
    ContentSizeTooLarge,
    Truncated,
    InvalidInternalHeader,
    BadSubChunkHeader,
    BadChunkHeader,
    TooManyChunks,
};

pub const HostWalkResult = struct {
    /// Total chunks parsed (equals what would land in `walk_meta.n_chunks`).
    n_chunks: u32,
    /// Decompressed bytes (from the outer block header).
    decomp_size: u32,
    /// Byte offset (into the original `slz_bytes`) where the block
    /// payload starts. The caller uses this to upload exactly the
    /// block-payload bytes (without the SC tail prefix) to the GPU
    /// `frame` buffer — mirrors CUDA's `block_payload` slice at
    /// `src/decode/streamlz_decoder.zig:344`.
    block_payload_off: u32,
    /// Bytes uploaded to the GPU `frame` buffer (= block payload sans
    /// SC tail prefix). All byte_base fields in `chunks_buf_16u32` are
    /// relative to byte 0 of this uploaded region.
    block_payload_size: u32,
    /// Byte offset (into `slz_bytes`) of the SC tail prefix bytes. The
    /// caller copies the (n_chunks - 1) * 8 bytes starting here into
    /// `dst` post-dispatch (CPU @memcpy — mirrors CUDA's
    /// `src/decode/streamlz_decoder.zig:294-308`).
    sc_tail_off: u32,
    /// Number of SC tail-prefix bytes (= (n_chunks - 1) * 8 for the
    /// self-contained case the GPU encoder emits, else 0).
    sc_tail_size: u32,
};

/// CPU port of CUDA's host-input L1 path. Parses an SLZ1 frame and
/// writes BOTH descriptor arrays the VK `lz_decode.comp` reader needs:
///
///   * `walk_chunks_buf_6u32[ci * 6 + slot]` — same layout as
///     walk_frame.comp's WalkChunks output. Slots: src_offset (UNUSED
///     post-CPU walk — we set it to the relative byte base of the
///     chunk's internal block header so debug introspection still
///     yields a meaningful offset, but lz_decode does not read it on
///     this path), comp_size, decomp_size, dst_offset, flags,
///     memset_fill.
///   * `chunks_buf_16u32[ci * 16 + slot]` — same layout as
///     l1_unwrap.comp's ChunkDescs output. Slots: 0/1 unused, 2..15
///     populated with byte_base + size for every stream that
///     `lz_decode.comp` reads (cmd / lit / off16 / length / off32 +
///     initial-copy + cmd_stream2_offset).
///
/// All byte_base fields are relative to byte 0 of the block payload
/// region (`out_result.block_payload_off`..`+ block_payload_size`),
/// matching what `l1_unwrap.comp` would have written had it run on
/// `block_start = 0` (which is the layout CUDA uses for the host-input
/// path — `block_payload` is passed verbatim and chunk src_offsets are
/// payload-relative).
///
/// Capacity contract: both output slices must be sized for at least
/// `wire_constants.MAX_CHUNKS` chunks (== CUDA's `WALK_MAX_CHUNKS`).
pub fn buildHostDescriptors(
    slz_bytes: []const u8,
    walk_chunks_buf_6u32: []u32,
    chunks_buf_16u32: []u32,
    out_result: *HostWalkResult,
) HostWalkError!void {
    if (slz_bytes.len == 0) return error.BadFrame;

    // ── 1. Frame header. ─────────────────────────────────────────────
    const fhdr = frame.parseHeader(slz_bytes) catch return error.BadHeader;

    if (fhdr.dictionary_id) |_| return error.UnknownDictionary;
    const cs = fhdr.content_size orelse return error.BadFrame;
    if (cs > std.math.maxInt(u32)) return error.ContentSizeTooLarge;
    const decomp_size: u32 = @intCast(cs);

    // ── 2. Single outer block. (GPU encoder emits exactly one block.) ─
    var pos: usize = fhdr.header_size;
    if (pos + 8 > slz_bytes.len) return error.Truncated;

    const bhdr = frame.parseBlockHeader(slz_bytes[pos..]) catch return error.InvalidInternalHeader;
    if (bhdr.isEndMark()) return error.BadFrame;
    if (bhdr.uncompressed or bhdr.parallel_decode_metadata) return error.InvalidInternalHeader;
    pos += 8;

    if (pos + bhdr.compressed_size > slz_bytes.len) return error.Truncated;
    const block_compressed_size = bhdr.compressed_size;
    const block_decompressed_size = bhdr.decompressed_size;
    if (block_decompressed_size != decomp_size) return error.BadFrame;

    // ── 3. Strip SC tail prefix from the block payload. ─────────────
    // Mirrors CUDA `src/decode/streamlz_decoder.zig:342-344`. The SC
    // tail prefix carries the 8-byte init prefix the decoder kernel
    // doesn't write into chunks 1..n-1 (no `Copy64` fires when
    // base_offset != 0). CPU @memcpy after dispatch restores those
    // bytes; lz_decode never sees them.
    const eff_chunk_size: u32 = @intCast(@min(
        frame.scGroupSizeToBytes(fhdr.sc_group_size),
        constants.chunk_size,
    ));
    const n_chunks_u64: u64 = (@as(u64, decomp_size) + eff_chunk_size - 1) / eff_chunk_size;
    if (n_chunks_u64 > wire_constants.MAX_CHUNKS) return error.TooManyChunks;
    const n_chunks: u32 = @intCast(n_chunks_u64);

    // Peek the internal block header (2 bytes) for the self_contained
    // flag — gates whether the SC tail prefix is present at all.
    if (block_compressed_size < 2) return error.Truncated;
    const peek = block_header.parseBlockHeader(slz_bytes[pos..]) catch
        return error.InvalidInternalHeader;
    if (peek.decoder_type != .fast and peek.decoder_type != .turbo)
        return error.InvalidInternalHeader;

    const prefix_sz: u32 = if (peek.self_contained and n_chunks > 1)
        (n_chunks - 1) * SC_TAIL_PER_CHUNK_BYTES
    else
        0;
    if (prefix_sz >= block_compressed_size) return error.Truncated;

    const block_payload_off: u32 = @intCast(pos);
    const block_payload_size: u32 = block_compressed_size - prefix_sz;
    const sc_tail_off: u32 = block_payload_off + block_payload_size;

    if (walk_chunks_buf_6u32.len < n_chunks * CHUNK_DESC_U32_COUNT) return error.TooManyChunks;
    if (chunks_buf_16u32.len < n_chunks * 16) return error.TooManyChunks;

    // Zero the active region of both descriptor arrays. The 16-u32
    // descriptor's slots 0/1 are UNUSED on this path (lz_decode reads
    // dst_offset + decomp_size from the 6-u32 binding-7 buffer instead),
    // but they're left as zero rather than uninitialised so any future
    // reader sees deterministic values. CUDA's
    // `streamlz_decoder.zig:358 @memset(std.mem.sliceAsBytes(chunk_descs), 0)`
    // does the same.
    @memset(walk_chunks_buf_6u32[0 .. n_chunks * CHUNK_DESC_U32_COUNT], 0);
    @memset(chunks_buf_16u32[0 .. n_chunks * 16], 0);

    // ── 4. Per-chunk walk. ──────────────────────────────────────────
    // Mirrors a fused version of CUDA `buildChunkDescriptors`
    // (src/decode/streamlz_decoder.zig:386-481) + the VK
    // `wire_format.unwrapSlz1ToL1Streams` per-chunk sub-chunk parse
    // (src_vulkan/wire_format.zig:752-941). The fusion writes both
    // descriptor outputs from a single pass over the block bytes.
    var src_pos: u32 = 0; // relative to block_payload start (i.e. the
    // bytes that will live at frame buffer offset 0..block_payload_size).
    var dst_off: u32 = 0;
    var ci: u32 = 0;
    var internal_hdr: ?block_header.BlockHeader = null;

    while (ci < n_chunks) : (ci += 1) {
        const at_chunk_boundary = (dst_off % eff_chunk_size) == 0;
        if (at_chunk_boundary or internal_hdr == null) {
            if (src_pos + 2 > block_payload_size) return error.Truncated;
            internal_hdr = block_header.parseBlockHeader(
                slz_bytes[block_payload_off + src_pos ..],
            ) catch return error.InvalidInternalHeader;
            src_pos += 2;
        }
        const ihdr = internal_hdr.?;

        const decomp_remaining: u32 = decomp_size - dst_off;
        const chunk_decomp_size: u32 = @min(eff_chunk_size, decomp_remaining);

        // ── 4a. Uncompressed chunk. ─────────────────────────────────
        // Payload = `chunk_decomp_size` raw bytes immediately after the
        // 2-byte internal header (no 4-byte chunk header). Matches
        // `streamlz_decoder.zig:412-426`. lz_decode treats it as one
        // long literal token via `writeUncompressedChunkDesc` equivalent.
        if (ihdr.uncompressed) {
            if (src_pos + chunk_decomp_size > block_payload_size) return error.Truncated;
            writeWalkChunkDesc(walk_chunks_buf_6u32, ci, .{
                .src_offset = src_pos,
                .comp_size = chunk_decomp_size,
                .decomp_size = chunk_decomp_size,
                .dst_offset = dst_off,
                .flags = CHUNK_FLAG_UNCOMPRESSED,
                .memset_fill = 0,
            });
            writeUncompressedChunkDescriptor(chunks_buf_16u32, ci, src_pos, chunk_decomp_size);
            dst_off += chunk_decomp_size;
            src_pos += chunk_decomp_size;
            continue;
        }

        // ── 4b. 4-byte chunk header. ─────────────────────────────────
        const chdr = block_header.parseChunkHeader(
            slz_bytes[block_payload_off + src_pos ..],
            ihdr.use_checksums,
        ) catch return error.BadChunkHeader;
        src_pos += @intCast(chdr.bytes_consumed);

        if (chdr.is_memset) {
            // Memset fast path not implemented on host-input today.
            // The GPU encoder doesn't currently emit memset chunks for
            // L1; defensively flag as unsupported so the codec returns
            // a real error rather than producing wrong output.
            return error.BadChunkHeader;
        }

        const chunk_payload_start = src_pos;
        const chunk_payload_end = src_pos + chdr.compressed_size;
        if (chunk_payload_end > block_payload_size) return error.Truncated;

        // Whole-chunk shortcut (comp_size == decomp_size): the chunk is
        // raw bytes verbatim. Mirrors the CUDA
        // `streamlz_decoder.zig:449-466` short circuit; emit it as an
        // uncompressed descriptor so the LZ kernel's trailing-literal
        // path copies the bytes through.
        if (chdr.compressed_size == chunk_decomp_size) {
            writeWalkChunkDesc(walk_chunks_buf_6u32, ci, .{
                .src_offset = chunk_payload_start,
                .comp_size = chdr.compressed_size,
                .decomp_size = chunk_decomp_size,
                .dst_offset = dst_off,
                .flags = CHUNK_FLAG_UNCOMPRESSED,
                .memset_fill = 0,
            });
            writeUncompressedChunkDescriptor(chunks_buf_16u32, ci, chunk_payload_start, chunk_decomp_size);
            dst_off += chunk_decomp_size;
            src_pos = chunk_payload_end;
            continue;
        }

        // ── 4c. 3-byte BE sub-chunk header. ─────────────────────────
        if (chunk_payload_start + SUBCHUNK_HDR_BYTES > chunk_payload_end) return error.Truncated;
        const sc_word: u32 = readBE24(slz_bytes[block_payload_off + src_pos ..][0..3]);
        const sc_payload_size: u32 = sc_word & SUBCHUNK_COMP_SIZE_MASK;
        const sc_is_lz: bool = (sc_word & SUBCHUNK_LZ_FLAG_BIT) != 0;
        if (!sc_is_lz) return error.BadSubChunkHeader;
        src_pos += SUBCHUNK_HDR_BYTES;
        if (src_pos + sc_payload_size > chunk_payload_end) return error.BadSubChunkHeader;
        const sc_end: u32 = src_pos + sc_payload_size;

        // ── 4d. Initial verbatim copy (chunk 0 only). ───────────────
        var initial_copy: u32 = 0;
        const init_byte_base: u32 = src_pos;
        if (ci == 0) {
            if (src_pos + INITIAL_LITERAL_COPY_BYTES > sc_end) return error.Truncated;
            initial_copy = INITIAL_LITERAL_COPY_BYTES;
            src_pos += INITIAL_LITERAL_COPY_BYTES;
        }

        // ── 4e. Literal stream (type-0 raw, 3-byte BE size header). ──
        if (src_pos + 3 > sc_end) return error.Truncated;
        // Validate chunk type = 0 (high nibble of byte 0).
        if ((slz_bytes[block_payload_off + src_pos] & 0x80) != 0) return error.BadSubChunkHeader;
        if (((slz_bytes[block_payload_off + src_pos] >> 4) & 0x07) != 0) return error.BadSubChunkHeader;
        const lit_size: u32 = readBE24(slz_bytes[block_payload_off + src_pos ..][0..3]);
        src_pos += 3;
        if (src_pos + lit_size > sc_end) return error.Truncated;
        const lit_byte_base: u32 = src_pos;
        src_pos += lit_size;

        // ── 4f. Command stream (type-0 raw, 3-byte BE size header). ──
        if (src_pos + 3 > sc_end) return error.Truncated;
        if ((slz_bytes[block_payload_off + src_pos] & 0x80) != 0) return error.BadSubChunkHeader;
        if (((slz_bytes[block_payload_off + src_pos] >> 4) & 0x07) != 0) return error.BadSubChunkHeader;
        const cmd_size: u32 = readBE24(slz_bytes[block_payload_off + src_pos ..][0..3]);
        src_pos += 3;
        if (src_pos + cmd_size > sc_end) return error.Truncated;
        const cmd_byte_base: u32 = src_pos;
        src_pos += cmd_size;

        // ── 4g. cmd_stream2_offset (2 bytes LE) if chunk > 64 KiB. ──
        var cmd_stream2_offset: u32 = 0;
        if (chunk_decomp_size > LZ_BLOCK_SIZE) {
            if (src_pos + 2 > sc_end) return error.Truncated;
            cmd_stream2_offset = std.mem.readInt(u16, slz_bytes[block_payload_off + src_pos ..][0..2], .little);
            src_pos += 2;
        }

        // ── 4h. off16 stream: 2-byte LE count + count*2 raw bytes. ──
        if (src_pos + 2 > sc_end) return error.Truncated;
        const off16_count: u32 = std.mem.readInt(u16, slz_bytes[block_payload_off + src_pos ..][0..2], .little);
        src_pos += 2;
        if (off16_count == wire_constants.OFF16_ENTROPY_MARKER) return error.BadSubChunkHeader;
        const off16_bytes: u32 = off16_count * OFF16_ENTRY_BYTES;
        if (src_pos + off16_bytes > sc_end) return error.Truncated;
        const off16_byte_base: u32 = src_pos;
        src_pos += off16_bytes;

        // ── 4i. off32 packed counts header (3 bytes LE). ────────────
        if (src_pos + 3 > sc_end) return error.Truncated;
        const off32_packed: u32 = @as(u32, slz_bytes[block_payload_off + src_pos + 0]) |
            (@as(u32, slz_bytes[block_payload_off + src_pos + 1]) << 8) |
            (@as(u32, slz_bytes[block_payload_off + src_pos + 2]) << 16);
        src_pos += 3;
        var off32_count1: u32 = off32_packed >> OFF32_COUNT_FIELD_BITS;
        var off32_count2: u32 = off32_packed & OFF32_COUNT_PACK_MAX;
        if (off32_count1 == OFF32_COUNT_PACK_MAX) {
            if (src_pos + 2 > sc_end) return error.Truncated;
            off32_count1 = std.mem.readInt(u16, slz_bytes[block_payload_off + src_pos ..][0..2], .little);
            src_pos += 2;
        }
        if (off32_count2 == OFF32_COUNT_PACK_MAX) {
            if (src_pos + 2 > sc_end) return error.Truncated;
            off32_count2 = std.mem.readInt(u16, slz_bytes[block_payload_off + src_pos ..][0..2], .little);
            src_pos += 2;
        }
        const off32_total_bytes: u32 = (off32_count1 + off32_count2) * OFF32_ENTRY_BYTES;
        if (src_pos + off32_total_bytes > sc_end) return error.Truncated;
        const off32_byte_base: u32 = src_pos;
        src_pos += off32_total_bytes;

        // ── 4j. Length stream — everything left in the sub-chunk. ───
        const length_byte_base: u32 = src_pos;
        const length_size: u32 = sc_end - src_pos;
        src_pos = sc_end;

        // Sanity: the chunk header's compressed_size should land us at
        // chunk_payload_end. (`unwrapSlz1ToL1Streams` enforces the same
        // contract at line 943-949.)
        if (src_pos != chunk_payload_end) return error.BadSubChunkHeader;

        // ── 4k. Emit walk_chunks descriptor (6 u32). ─────────────────
        writeWalkChunkDesc(walk_chunks_buf_6u32, ci, .{
            .src_offset = chunk_payload_start,
            .comp_size = chdr.compressed_size,
            .decomp_size = chunk_decomp_size,
            .dst_offset = dst_off,
            .flags = 0,
            .memset_fill = 0,
        });

        // ── 4l. Emit lz_decode chunk descriptor (16 u32). ───────────
        // Slot semantics mirror what `l1_unwrap.comp::writeChunkDesc`
        // (src_vulkan/shaders/l1_unwrap.comp:116-141) would have written.
        writeLzChunkDescriptor(chunks_buf_16u32, ci, .{
            .cmd_byte_base = cmd_byte_base,
            .cmd_size = cmd_size,
            .lit_byte_base = lit_byte_base,
            .lit_size = lit_size,
            .off16_byte_base = off16_byte_base,
            .off16_count = off16_count,
            .length_byte_base = length_byte_base,
            .length_size = length_size,
            .initial_copy = initial_copy,
            .init_byte_base = init_byte_base,
            .off32_byte_base = off32_byte_base,
            .off32_count1 = off32_count1,
            .off32_count2 = off32_count2,
            .cmd_stream2_offset = cmd_stream2_offset,
        });

        dst_off += chunk_decomp_size;
    }

    if (dst_off != decomp_size) return error.BadFrame;
    if (src_pos != block_payload_size) return error.Truncated;

    out_result.* = .{
        .n_chunks = n_chunks,
        .decomp_size = decomp_size,
        .block_payload_off = block_payload_off,
        .block_payload_size = block_payload_size,
        .sc_tail_off = sc_tail_off,
        .sc_tail_size = prefix_sz,
    };
}

const WalkDescFields = struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u32,
};

fn writeWalkChunkDesc(buf: []u32, ci: u32, f: WalkDescFields) void {
    const base = ci * CHUNK_DESC_U32_COUNT;
    buf[base + CHUNK_SRC_OFFSET_SLOT] = f.src_offset;
    buf[base + CHUNK_COMP_SIZE_SLOT] = f.comp_size;
    buf[base + CHUNK_DECOMP_SIZE_SLOT] = f.decomp_size;
    buf[base + CHUNK_DST_OFFSET_SLOT] = f.dst_offset;
    buf[base + CHUNK_FLAGS_SLOT] = f.flags;
    buf[base + CHUNK_MEMSET_FILL_SLOT] = f.memset_fill;
}

const LzDescFields = struct {
    cmd_byte_base: u32,
    cmd_size: u32,
    lit_byte_base: u32,
    lit_size: u32,
    off16_byte_base: u32,
    off16_count: u32,
    length_byte_base: u32,
    length_size: u32,
    initial_copy: u32,
    init_byte_base: u32,
    off32_byte_base: u32,
    off32_count1: u32,
    off32_count2: u32,
    cmd_stream2_offset: u32,
};

fn writeLzChunkDescriptor(buf: []u32, ci: u32, f: LzDescFields) void {
    const base = ci * 16;
    // Slots 0/1 (dst_offset / dst_size) are unused on the host-input
    // path — lz_decode reads dst_offset + decomp_size from
    // walk_chunks_buf (binding 7) instead. Left zero for determinism.
    buf[base + 0] = 0;
    buf[base + 1] = 0;
    buf[base + 2] = f.cmd_byte_base;
    buf[base + 3] = f.cmd_size;
    buf[base + 4] = f.lit_byte_base;
    buf[base + 5] = f.lit_size;
    buf[base + 6] = f.off16_byte_base;
    buf[base + 7] = f.off16_count;
    buf[base + 8] = f.length_byte_base;
    buf[base + 9] = f.length_size;
    buf[base + 10] = f.initial_copy;
    buf[base + 11] = f.off32_byte_base;
    buf[base + 12] = f.off32_count1;
    buf[base + 13] = f.off32_count2;
    buf[base + 14] = f.cmd_stream2_offset;
    buf[base + 15] = f.init_byte_base;
}

/// Mirrors `l1_unwrap.comp::writeUncompressedChunkDesc`: all stream
/// bases point at the raw payload, cmd_size = 0 so the decoder falls
/// through to the trailing-literal copy that flushes `decomp_size`
/// bytes from lit_buf into dst.
fn writeUncompressedChunkDescriptor(buf: []u32, ci: u32, payload_byte_base: u32, decomp_size: u32) void {
    const base = ci * 16;
    buf[base + 0] = 0;
    buf[base + 1] = 0;
    buf[base + 2] = payload_byte_base;
    buf[base + 3] = 0; // cmd_size = 0
    buf[base + 4] = payload_byte_base;
    buf[base + 5] = decomp_size; // lit_size
    buf[base + 6] = payload_byte_base;
    buf[base + 7] = 0; // off16_count
    buf[base + 8] = payload_byte_base;
    buf[base + 9] = 0; // length_size
    buf[base + 10] = 0; // initial_copy
    buf[base + 11] = payload_byte_base;
    buf[base + 12] = 0;
    buf[base + 13] = 0;
    buf[base + 14] = 0;
    buf[base + 15] = payload_byte_base;
}

inline fn readBE24(b: *const [3]u8) u32 {
    return (@as(u32, b[0]) << 16) | (@as(u32, b[1]) << 8) | @as(u32, b[2]);
}
