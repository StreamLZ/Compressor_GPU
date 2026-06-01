//! SLZ1 wire-format wrapper for Vulkan L1 streams.
//!
//! Phase: CPU-side framing only. The Vulkan L1 codec
//! (`src_vulkan/l1_codec.zig`) emits four raw streams + per-chunk sizes
//! per chunk (lit / cmd / off16 / length). This module wraps those
//! streams into a real `.slz` file that the CUDA decoder
//! (`zig-out/bin/streamlz.exe -d`) can consume — and conversely unwraps
//! a CUDA-produced `.slz` back into the per-chunk stream view the
//! Vulkan decoder needs.
//!
//! Wire-format reuse strategy: every outer header (frame header, block
//! header, end mark) goes through the CPU codec's `src/format/...`
//! parsers and writers (imported via the absolute-path-from-tests
//! pattern that `tests_root.zig` already uses).  Sub-chunk and stream
//! headers — which live device-side on the CUDA encode path and so
//! have no Zig writer — are emitted by hand here per the layouts in
//! `src/decode/slz_wire_format.cuh` and
//! `src/common/gpu_wire_format.cuh`.
//!
//! ── Layout one wrapped frame emits (sc_group_size = 0.5) ───────────
//!
//!   Frame header (LE):
//!     14 bytes fixed + 8 bytes content_size = 22 bytes
//!     codec = Fast (1), level = 1, block_size = 256 KiB,
//!     sc_group_size = 0.5f, content_size_present = true.
//!
//!   One block covers the whole input.
//!   Block header (8 bytes, LE):
//!     compressed_size = payload + (n_chunks - 1) * 8     (SC tail
//!                                                          prefix bytes)
//!     decompressed_size = original input bytes
//!     uncompressed flag = false, parallel_decode_metadata = false.
//!
//!   Per chunk (the L1 codec's CHUNK_SIZE = 128 KiB == the wire-format
//!   sub_chunk_size at sc_group_size = 0.5, so each chunk maps to
//!   exactly one sub-chunk):
//!     [2-byte internal block header] (BE-positional, byte 0 = flags,
//!                                     byte 1 = decoder_type)
//!         byte 0 = 0x55 = SLZ_INT_BLOCK_MAGIC (0x05)
//!                       | self_contained (0x10) | restart_decoder (0x40)
//!         byte 1 = 0x01 = decoder Fast (= CodecType.fast).
//!     [4-byte LE chunk header]
//!         low 18 bits = compressed_size - 1
//!                       (compressed_size = SUBCHUNK_HDR_BYTES (3)
//!                                          + sub-chunk payload bytes,
//!                        i.e. everything the sub-chunk contributes to
//!                        the chunk body).
//!     [3-byte BE sub-chunk header]
//!         bits 0-18 = sub-chunk payload size
//!         bits 19-22 = mode = 1 (LZ-with-entropy decoder path; the
//!                                CUDA decoder routes mode 0 and 1
//!                                through the same per-stream type-0/4
//!                                dispatch, and the assembler picks 1)
//!         bit 23 = SUBCHUNK_LZ_FLAG_BIT.
//!     [sub-chunk payload]:
//!         [8 verbatim source bytes]    only for the first chunk
//!                                       (INITIAL_LITERAL_COPY_BYTES)
//!         [3-byte BE size][lit bytes]   type-0 (raw) chunk header
//!                                        (high nibble of byte 0 = 0)
//!         [3-byte BE size][cmd bytes]
//!         [2-byte LE cmd_stream2_offset]   present iff sub_decomp_size
//!                                          > LZ_BLOCK_SIZE (64 KiB).
//!                                          For L1 with 128 KiB
//!                                          sub-chunks this is ALWAYS
//!                                          present.  Setting it to
//!                                          cmd_size puts every token
//!                                          in LZ block 0, which the
//!                                          decoder's block_iter loop
//!                                          handles correctly because
//!                                          dst_pos lands at
//!                                          sub_decomp_size after the
//!                                          encoder's trailing
//!                                          long-literal token.
//!         [2-byte LE off16_count][off16 bytes interleaved u16 LE]
//!                                          raw form, since the L1
//!                                          codec emits at most a
//!                                          handful of off16 entries
//!                                          per sub-chunk relative to
//!                                          OFF16_ENTROPY_MIN (32) —
//!                                          this is the "<32 entries"
//!                                          branch in the assembler.
//!         [3 bytes 0x00] off32 packed counts = 0, no off32 entries
//!         [length bytes]                   raw length stream.
//!
//!   (n_chunks - 1) * 8 bytes SC tail prefix:
//!     For each chunk i in [1..n_chunks): 8 source bytes starting at
//!     i * chunk_size, padded with zeros if past EOF.  Restores the
//!     per-chunk init copy that the decoder kernel doesn't write into
//!     non-first chunks.
//!
//!   [4-byte LE end mark = 0]
//!
//! ── Why off32 emission is out of scope ─────────────────────────────
//! The Vulkan L1 encoder rejects far-offset matches (§11.4 of the spec)
//! — when a resolved offset > 0xFFFF, the parser advances and tries
//! again.  Therefore off32_count is always zero and the 3-byte packed
//! header trivially serializes as `[0x00, 0x00, 0x00]`.
//!
//! ── Why chunk 0 is emitted uncompressed ────────────────────────────
//! The CUDA decoder unconditionally copies the first 8 bytes of the
//! first sub-chunk's payload to dst[0..8] before processing any token
//! (`base_offset == 0` branch in
//! `src/decode/lz_dispatch.cuh::parseAndDecodeSubChunkRaw`).  The
//! Vulkan L1 encoder, by spec §1 line 15, OMITS this 8-byte prefix
//! (anchor = 0 from the start) — its token stream decodes positions
//! 0..src_size including those first 8 bytes.  The two cannot be
//! reconciled without either re-targeting tokens (offset-by-8 fixups
//! on every emitted literal-length token) or rebuilding the encoder.
//! Both are out of scope for this CPU-side wrap.  As the simpler-and-
//! correct alternative, chunk 0 is emitted as a whole-chunk
//! uncompressed block: bit 7 of `internal_hdr0` is set, no 4-byte
//! chunk header follows, and the 2-byte internal block header is
//! immediately followed by `chunk_size` raw source bytes.  Chunks
//! 1..n-1 round-trip through the LZ path normally — for them
//! `base_offset > 0` so the decoder skips the initial copy, AND the
//! SC tail prefix table after the last chunk OVERWRITES whatever the
//! Vulkan decoder wrote into the first 8 bytes of each non-first
//! chunk.  Net cost: 128 KiB of raw bytes per frame (chunk 0).  All
//! subsequent chunks compress.

const std = @import("std");

const l1_codec = @import("l1_codec.zig");

// ── Imports of the CPU codec's wire-format helpers ────────────────
// Imported via relative path.  This module compiles only as part of
// the `vk-wire-format-test` build target, whose root file at the repo
// top (`wire_format_test_root.zig`) widens the package boundary so
// `../src/...` resolves through `src_vulkan/wire_format.zig`.
const frame = @import("../src/format/frame_format.zig");
const block_header = @import("../src/format/block_header.zig");
const constants = @import("../src/format/streamlz_constants.zig");

// ── Wire-format constants mirrored from common/gpu_wire_format.cuh ─
// Kept here as private constants so we don't need to pull the .cuh
// header into the Zig build graph.  The static_assert in the .cuh
// validates these against the encoder side at CUDA build time; this
// file's tests validate them against the decoder side at runtime.

/// LZ block boundary inside a sub-chunk (64 KiB).  Sub-chunks larger
/// than this trigger the cmd_stream2_offset prefix.
const LZ_BLOCK_SIZE: u32 = 0x10000;

/// First sub-chunk in a frame carries this many verbatim source bytes
/// before the lit-stream header (matches reencodeGpuWithEntropy's
/// `initial_bytes` and lz_kernel.cu's `is_first` branch).
const INITIAL_LITERAL_COPY_BYTES: u32 = 8;

/// 3-byte BE sub-chunk header: bit 23 set = LZ-coded sub-chunk.
const SUBCHUNK_LZ_FLAG_BIT: u32 = 0x800000;
/// 3-byte BE sub-chunk header: bits 19-22 = decode mode nibble.  The
/// assembler writes mode = 1 (LZ-with-entropy); the decoder treats
/// 0 and 1 identically because the per-stream type byte already says
/// raw-vs-Huffman.
const SUBCHUNK_MODE_SHIFT: u5 = 19;
const SUBCHUNK_HDR_BYTES: u32 = 3;

/// Per-chunk header overhead (mirrors encode_context.zig):
///   2-byte internal block header + 4-byte chunk header.
const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;

/// SC tail prefix bytes per (non-first) chunk.
const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;

/// L1 codec chunk size — 128 KiB.  Maps 1:1 to the wire-format
/// sub_chunk_size at sc_group_size = 0.5.
pub const VK_CHUNK_SIZE: u32 = l1_codec.CHUNK_SIZE;

// ── PerChunkStreams: host-visible per-chunk view of the L1 streams ─
// `wrapL1ToSlz1` consumes one of these; `unwrapSlz1ToL1Streams`
// produces one.  The byte slices are owned by the caller for the
// wrap path (typically scratch buffers filled by mapping the Vulkan
// SSBOs); they are owned by the unwrap path (allocator-allocated)
// for the decode path.

pub const PerChunkStreams = struct {
    /// One entry per chunk; lit_bytes[i] is the chunk-local literal
    /// byte run (`per_chunk_lit_size[i]` bytes).
    lit_bytes: [][]const u8,
    cmd_bytes: [][]const u8,
    /// off16_bytes[i] is the raw interleaved u16-LE byte slice
    /// (`per_chunk_off16_count[i] * 2` bytes).
    off16_bytes: [][]const u8,
    length_bytes: [][]const u8,
    /// Original source bytes — needed for the per-chunk init prefix
    /// (chunk 0's first 8 verbatim bytes) and the SC tail prefix
    /// (8 bytes per non-first chunk from the source).  Wrap path only;
    /// the unwrap path leaves this null.
    src_bytes: ?[]const u8,
    /// Chunk geometry.
    n_chunks: u32,
    chunk_size: u32 = VK_CHUNK_SIZE,
    /// Original input length.  Stored on the bundle so wrap can stamp
    /// the frame header's content_size without a separate parameter.
    original_size: u64,
};

pub const UnwrapResult = struct {
    n_chunks: u32,
    chunk_size: u32,
    original_size: u64,
    /// Per-chunk decompressed sizes (last chunk may be < chunk_size).
    per_chunk_decomp_size: [l1_codec.MAX_CHUNKS]u32,
};

pub const WrapError = error{
    OutOfMemory,
    OutputTooSmall,
    TooManyChunks,
    /// A frame-header parameter (level, block_size, sc_group_size)
    /// outside the writer's accepted range.  Indirectly bubbles up
    /// from `frame.writeHeader`.
    BadHeader,
};

pub const UnwrapError = error{
    OutOfMemory,
    Truncated,
    BadMagic,
    UnsupportedVersion,
    BadCodec,
    BadBlockSize,
    BadScGroupSize,
    BadInternalHeader,
    BadChunkHeader,
    BadSubChunkHeader,
    TooManyChunks,
    /// frame's content size missing or contradicts the parsed block(s).
    MissingContentSize,
};

// ── Helpers ───────────────────────────────────────────────────────

inline fn writeBE24(dst: []u8, v: u32) void {
    dst[0] = @intCast((v >> 16) & 0xFF);
    dst[1] = @intCast((v >> 8) & 0xFF);
    dst[2] = @intCast(v & 0xFF);
}

inline fn readBE24(src: []const u8) u32 {
    return (@as(u32, src[0]) << 16) | (@as(u32, src[1]) << 8) | @as(u32, src[2]);
}

inline fn writeLE24(dst: []u8, v: u32) void {
    dst[0] = @intCast(v & 0xFF);
    dst[1] = @intCast((v >> 8) & 0xFF);
    dst[2] = @intCast((v >> 16) & 0xFF);
}

// ── Wrap: per-chunk streams → SLZ1 bytes ─────────────────────────

/// Size upper bound for `wrapL1ToSlz1`.  Caller allocates `out` with
/// at least this many bytes.  Actual usage runs ~bounded by sum of
/// stream sizes + per-chunk header overhead, well under the bound
/// below.
pub fn wrapBound(streams: PerChunkStreams) usize {
    var total: usize = frame.max_header_size + 8; // frame + block header
    var ci: usize = 0;
    while (ci < streams.n_chunks) : (ci += 1) {
        if (ci == 0) {
            // Chunk 0 is emitted uncompressed (see header comment) —
            // 2-byte internal block header + chunk_size raw bytes,
            // capped to the original input length for inputs smaller
            // than one chunk.
            total += 2;
            total += streams.chunk_size;
            continue;
        }
        total += CHUNK_INTERNAL_HDR_BYTES; // 2-byte internal + 4-byte chunk
        total += SUBCHUNK_HDR_BYTES;        // 3-byte sub-chunk header
        total += 3 + streams.lit_bytes[ci].len;           // type-0 lit
        total += 3 + streams.cmd_bytes[ci].len;           // type-0 cmd
        total += 2;                                        // cmd_stream2_offset
        total += 2 + streams.off16_bytes[ci].len;          // raw off16
        total += 3;                                        // off32 packed = 0
        total += streams.length_bytes[ci].len;             // length stream
    }
    if (streams.n_chunks > 1) {
        total += @as(usize, streams.n_chunks - 1) * SC_TAIL_PER_CHUNK_BYTES;
    }
    total += 4; // end mark
    return total;
}

/// Walk the per-chunk streams and emit a complete SLZ1 frame into
/// `out`.  Returns the number of bytes written.  `streams.src_bytes`
/// must be set — wrap reads the per-chunk init prefix (chunk 0's first
/// 8 bytes) and the SC tail prefix (8 bytes per non-first chunk) from
/// it directly.
pub fn wrapL1ToSlz1(
    streams: PerChunkStreams,
    out: []u8,
) WrapError!usize {
    if (streams.n_chunks == 0 or streams.n_chunks > l1_codec.MAX_CHUNKS) {
        return error.TooManyChunks;
    }
    const src = streams.src_bytes orelse return error.BadHeader;
    if (out.len < wrapBound(streams)) return error.OutputTooSmall;

    var pos: usize = 0;

    // ── 1. Frame header ──────────────────────────────────────────
    const hdr_n = frame.writeHeader(out[pos..], .{
        .codec = .fast,
        .level = 1,
        .block_size = constants.chunk_size, // 256 KiB
        .sc_group_size = 0.5,
        .content_size = streams.original_size,
        .dictionary_id = null,
        .content_checksum = false,
    }) catch return error.BadHeader;
    pos += hdr_n;

    // ── 2. Block header placeholder (filled after walking chunks) ─
    const block_hdr_pos = pos;
    pos += 8;

    const block_payload_start = pos;

    // ── 3. Per-chunk wrapping ───────────────────────────────────
    var ci: u32 = 0;
    while (ci < streams.n_chunks) : (ci += 1) {
        const chunk_dst_off: u64 = @as(u64, ci) * streams.chunk_size;
        const chunk_decomp_size: u32 = blk: {
            const remaining = streams.original_size - chunk_dst_off;
            if (remaining >= streams.chunk_size) break :blk streams.chunk_size;
            break :blk @intCast(remaining);
        };

        // 3a. 2-byte internal block header.
        //   byte 0 = magic(0x05) | self_contained(0x10) | restart_decoder(0x40)
        //          | (uncompressed(0x80) iff chunk 0)
        //   byte 1 = decoder_type = Fast (1).
        const uncomp_flag: u8 = if (ci == 0) 0x80 else 0;
        const internal_hdr0: u8 = 0x05 | 0x10 | 0x40 | uncomp_flag;
        const internal_hdr1: u8 = @intFromEnum(block_header.CodecType.fast);
        out[pos + 0] = internal_hdr0;
        out[pos + 1] = internal_hdr1;
        pos += 2;

        if (ci == 0) {
            // 3b/c/d (uncompressed). Per `streamlz_decoder.zig::
            // buildChunkDescriptors`, an uncompressed chunk has NO
            // 4-byte chunk header — its payload sits immediately after
            // the 2-byte internal header, sized by the outer
            // `eff_chunk_size` (= our `chunk_size`) capped to the
            // remaining source bytes.
            std.debug.assert(src.len >= chunk_dst_off + chunk_decomp_size);
            @memcpy(out[pos..][0..chunk_decomp_size],
                src[@intCast(chunk_dst_off)..][0..chunk_decomp_size]);
            pos += chunk_decomp_size;
            continue;
        }

        // 3b. 4-byte LE chunk header (compressed_size - 1 in low 18 bits).
        //   compressed_size = 3-byte sub-chunk header + sub-chunk payload bytes.
        const sub_payload_size: u32 = subChunkPayloadSize(
            ci,
            chunk_decomp_size,
            streams.lit_bytes[ci].len,
            streams.cmd_bytes[ci].len,
            streams.off16_bytes[ci].len,
            streams.length_bytes[ci].len,
        );
        const chunk_compressed_size: u32 = SUBCHUNK_HDR_BYTES + sub_payload_size;
        std.mem.writeInt(u32, out[pos..][0..4], chunk_compressed_size - 1, .little);
        pos += 4;

        // 3c. 3-byte BE sub-chunk header.
        const sc_hdr_word: u32 = sub_payload_size |
            (@as(u32, 1) << SUBCHUNK_MODE_SHIFT) |
            SUBCHUNK_LZ_FLAG_BIT;
        writeBE24(out[pos..][0..3], sc_hdr_word);
        pos += 3;

        // 3d. Sub-chunk payload (NO initial 8-byte verbatim copy for
        //     chunk index > 0 — `base_offset > 0` on the decoder side
        //     skips that prefix).

        // 3d.i Literal stream (type-0 raw).
        writeBE24(out[pos..][0..3], @intCast(streams.lit_bytes[ci].len));
        pos += 3;
        @memcpy(out[pos..][0..streams.lit_bytes[ci].len], streams.lit_bytes[ci]);
        pos += streams.lit_bytes[ci].len;

        // 3d.ii Command/token stream (type-0 raw).
        writeBE24(out[pos..][0..3], @intCast(streams.cmd_bytes[ci].len));
        pos += 3;
        @memcpy(out[pos..][0..streams.cmd_bytes[ci].len], streams.cmd_bytes[ci]);
        pos += streams.cmd_bytes[ci].len;

        // 3d.iii cmd_stream2_offset (2 bytes LE) — present because
        //         sub_decomp_size > LZ_BLOCK_SIZE.  Set to cmd_size so
        //         every token goes through the CUDA decoder's block-0
        //         loop iteration; block-1 finds zero tokens and the
        //         block-trailing routine quickly observes
        //         dst_pos >= block_end with no fill needed.
        if (chunk_decomp_size > LZ_BLOCK_SIZE) {
            const cs2o: u16 = @intCast(streams.cmd_bytes[ci].len);
            std.mem.writeInt(u16, out[pos..][0..2], cs2o, .little);
            pos += 2;
        }

        // 3d.iv off16 stream (small raw form: count < OFF16_ENTROPY_MIN).
        const off16_count: u32 = @intCast(streams.off16_bytes[ci].len / 2);
        std.mem.writeInt(u16, out[pos..][0..2], @intCast(off16_count), .little);
        pos += 2;
        @memcpy(out[pos..][0..streams.off16_bytes[ci].len], streams.off16_bytes[ci]);
        pos += streams.off16_bytes[ci].len;

        // 3d.v off32 packed counts header — all zero (L1 emits no off32).
        out[pos + 0] = 0;
        out[pos + 1] = 0;
        out[pos + 2] = 0;
        pos += 3;

        // 3d.vi length stream (raw).
        @memcpy(out[pos..][0..streams.length_bytes[ci].len], streams.length_bytes[ci]);
        pos += streams.length_bytes[ci].len;
    }

    // ── 4. SC tail prefix (n_chunks - 1 entries of 8 bytes each). ─
    //       Each entry restores the 8 verbatim source bytes the decoder
    //       skips at the head of every non-first chunk.
    if (streams.n_chunks > 1) {
        var prefix_idx: u32 = 0;
        while (prefix_idx < streams.n_chunks - 1) : (prefix_idx += 1) {
            const src_off: u64 = @as(u64, prefix_idx + 1) * streams.chunk_size;
            var i: usize = 0;
            while (i < SC_TAIL_PER_CHUNK_BYTES) : (i += 1) {
                const abs: u64 = src_off + i;
                out[pos + i] = if (abs < src.len) src[@intCast(abs)] else 0;
            }
            pos += SC_TAIL_PER_CHUNK_BYTES;
        }
    }

    // ── 5. Backfill block header now that we know the payload size. ─
    const block_payload_size: u32 = @intCast(pos - block_payload_start);
    frame.writeBlockHeader(out[block_hdr_pos..][0..8], .{
        .compressed_size = block_payload_size,
        .decompressed_size = @intCast(streams.original_size),
        .uncompressed = false,
        .parallel_decode_metadata = false,
    });

    // ── 6. End mark. ────────────────────────────────────────────
    frame.writeEndMark(out[pos..][0..4]);
    pos += 4;

    return pos;
}

/// Compute the assembled sub-chunk payload size for one (compressed)
/// chunk.  Chunks emitted as uncompressed use this only for the
/// upper-bound calculation in `wrapBound` and skip this code path at
/// emit time.  `chunk_idx` is currently unused (chunk 0 never reaches
/// here in the live path), but kept in the signature so the helper
/// generalizes to a future encoder that DOES support `is_first`-aware
/// sub-chunks.
fn subChunkPayloadSize(
    chunk_idx: u32,
    chunk_decomp_size: u32,
    lit_size: usize,
    cmd_size: usize,
    off16_bytes: usize,
    length_size: usize,
) u32 {
    _ = chunk_idx;
    var n: u32 = 0;
    n += 3 + @as(u32, @intCast(lit_size)); // type-0 lit
    n += 3 + @as(u32, @intCast(cmd_size)); // type-0 cmd
    if (chunk_decomp_size > LZ_BLOCK_SIZE) n += 2; // cmd_stream2_offset
    n += 2 + @as(u32, @intCast(off16_bytes)); // off16 count + bytes
    n += 3; // off32 packed = 0
    n += @as(u32, @intCast(length_size));
    return n;
}

// ── Unwrap: SLZ1 bytes → per-chunk streams ────────────────────────

/// Storage owned by `unwrapSlz1ToL1Streams`.  Caller frees via
/// `freeUnwrapStorage` once done reading the per-chunk byte slices.
pub const UnwrapStorage = struct {
    lit_arena: []u8,
    cmd_arena: []u8,
    off16_arena: []u8,
    length_arena: []u8,
    lit_views: [][]const u8,
    cmd_views: [][]const u8,
    off16_views: [][]const u8,
    length_views: [][]const u8,
};

pub fn freeUnwrapStorage(allocator: std.mem.Allocator, s: *UnwrapStorage) void {
    allocator.free(s.lit_arena);
    allocator.free(s.cmd_arena);
    allocator.free(s.off16_arena);
    allocator.free(s.length_arena);
    allocator.free(s.lit_views);
    allocator.free(s.cmd_views);
    allocator.free(s.off16_views);
    allocator.free(s.length_views);
}

/// Parse an SLZ1 frame produced by either the CUDA encoder or the
/// `wrapL1ToSlz1` writer above, and reassemble the per-chunk stream
/// views the Vulkan L1 decoder needs.  Returns metadata in
/// `out_result`; `out_streams` is filled with view slices that borrow
/// from the returned `UnwrapStorage`.
pub const UnwrapBundle = struct {
    result: UnwrapResult,
    storage: UnwrapStorage,
};

pub fn unwrapSlz1ToL1Streams(
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    out_streams: *PerChunkStreams,
) UnwrapError!UnwrapBundle {
    // ── 1. Frame header. ─────────────────────────────────────────
    const fhdr = frame.parseHeader(slz_bytes) catch |err| switch (err) {
        error.BadMagic => return error.BadMagic,
        error.UnsupportedVersion => return error.UnsupportedVersion,
        error.BadCodec => return error.BadCodec,
        error.BadBlockSize => return error.BadBlockSize,
        error.BadScGroupSize => return error.BadScGroupSize,
        error.Truncated => return error.Truncated,
    };
    var pos: usize = fhdr.header_size;
    const original_size: u64 = fhdr.content_size orelse return error.MissingContentSize;
    const eff_chunk_size: u32 = @intCast(@min(
        frame.scGroupSizeToBytes(fhdr.sc_group_size),
        constants.chunk_size,
    ));

    const n_chunks_u64: u64 = (original_size + eff_chunk_size - 1) / eff_chunk_size;
    if (n_chunks_u64 > l1_codec.MAX_CHUNKS) return error.TooManyChunks;
    const n_chunks: u32 = @intCast(n_chunks_u64);

    // ── 2. Walk the (single) compressed block. ───────────────────
    //       The CUDA encoder and our wrap both emit one block for the
    //       whole input; the unwrap path enforces that — multi-block
    //       inputs are an L≥3 / Huffman concern out of scope here.
    const bhdr = frame.parseBlockHeader(slz_bytes[pos..]) catch |err| switch (err) {
        error.Truncated => return error.Truncated,
        error.BadMagic, error.UnsupportedVersion, error.BadCodec,
        error.BadBlockSize, error.BadScGroupSize => return error.BadInternalHeader,
    };
    if (bhdr.isEndMark()) return error.Truncated;
    pos += 8;
    if (bhdr.uncompressed or bhdr.parallel_decode_metadata) {
        return error.BadInternalHeader;
    }
    if (pos + bhdr.compressed_size > slz_bytes.len) return error.Truncated;

    const block_end = pos + bhdr.compressed_size;

    // Strip the SC tail prefix bytes off the end of the block payload —
    // they sit after the last chunk and aren't part of any chunk's
    // sub-chunk payload.
    const prefix_sz: usize = if (n_chunks > 1)
        @as(usize, n_chunks - 1) * SC_TAIL_PER_CHUNK_BYTES
    else
        0;
    if (prefix_sz >= bhdr.compressed_size) return error.Truncated;
    const chunks_end: usize = block_end - prefix_sz;

    // ── 3. Allocate storage for the per-chunk views. ─────────────
    //       Maximum per-chunk stream size = sub_chunk_size (128 KiB)
    //       — the L1 codec's worst case is `2 * src_size + 16`, but
    //       the wire-format header is the size source of truth.
    //       Bumped lit_arena to `2 * original_size` so an uncompressed
    //       chunk (whose entire 128 KiB payload moves into the lit
    //       stream) plus the chunk-0 8-byte prefix splice both fit
    //       without an additional sizing pass.
    var storage: UnwrapStorage = .{
        .lit_arena = try allocator.alloc(u8, original_size * 2 + 1024),
        .cmd_arena = try allocator.alloc(u8, original_size + 1024),
        .off16_arena = try allocator.alloc(u8, original_size + 1024),
        .length_arena = try allocator.alloc(u8, original_size + 1024),
        .lit_views = try allocator.alloc([]const u8, n_chunks),
        .cmd_views = try allocator.alloc([]const u8, n_chunks),
        .off16_views = try allocator.alloc([]const u8, n_chunks),
        .length_views = try allocator.alloc([]const u8, n_chunks),
    };
    errdefer freeUnwrapStorage(allocator, &storage);

    var lit_arena_pos: usize = 0;
    var cmd_arena_pos: usize = 0;
    var off16_arena_pos: usize = 0;
    var length_arena_pos: usize = 0;

    var result: UnwrapResult = .{
        .n_chunks = n_chunks,
        .chunk_size = eff_chunk_size,
        .original_size = original_size,
        .per_chunk_decomp_size = @splat(0),
    };

    // ── 4. Per-chunk walk. ───────────────────────────────────────
    //       The unwrap path tolerates BOTH layouts on a per-chunk
    //       basis: uncompressed (the layout `wrapL1ToSlz1` emits for
    //       chunk 0) and LZ-compressed (every other chunk plus the
    //       layout the CUDA encoder emits for every chunk).  For
    //       uncompressed chunks we copy the raw source bytes into the
    //       literal stream and synthesise a single long-literal token
    //       that consumes them, so the Vulkan L1 decoder reproduces
    //       the bytes as if they had been LZ-encoded.
    var ci: u32 = 0;
    while (ci < n_chunks) : (ci += 1) {
        if (pos + 2 > chunks_end) return error.Truncated;
        const ihdr = block_header.parseBlockHeader(slz_bytes[pos..]) catch
            return error.BadInternalHeader;
        if (ihdr.decoder_type != .fast and ihdr.decoder_type != .turbo) {
            return error.BadInternalHeader;
        }
        pos += 2;

        // Per-chunk decompressed size (last chunk may be < eff_chunk_size).
        //   Lifted out of the LZ branch so the uncompressed branch can
        //   use it for its raw-byte copy size.
        const decomp_remaining_top: u64 = original_size - @as(u64, ci) * eff_chunk_size;
        const chunk_decomp_size_top: u32 = @intCast(@min(
            decomp_remaining_top,
            @as(u64, eff_chunk_size),
        ));
        result.per_chunk_decomp_size[ci] = chunk_decomp_size_top;

        if (ihdr.uncompressed) {
            // Uncompressed chunk: payload = chunk_decomp_size raw
            // source bytes immediately after the 2-byte internal
            // header (no 4-byte chunk header).  Materialise a single
            // long-literal token so the Vulkan decoder treats the
            // chunk as one LZ run of literals.
            if (pos + chunk_decomp_size_top > chunks_end) return error.Truncated;
            const raw_view = storage.lit_arena[lit_arena_pos..][0..chunk_decomp_size_top];
            @memcpy(raw_view, slz_bytes[pos..][0..chunk_decomp_size_top]);
            storage.lit_views[ci] = raw_view;
            lit_arena_pos += chunk_decomp_size_top;
            pos += chunk_decomp_size_top;

            // Empty cmd stream — the Vulkan decoder's trailing-literal
            // routine flushes the remaining lit bytes when cmd has no
            // tokens to process.
            const empty_cmd = storage.cmd_arena[cmd_arena_pos..][0..0];
            storage.cmd_views[ci] = empty_cmd;
            const empty_off16 = storage.off16_arena[off16_arena_pos..][0..0];
            storage.off16_views[ci] = empty_off16;
            const empty_len = storage.length_arena[length_arena_pos..][0..0];
            storage.length_views[ci] = empty_len;
            continue;
        }

        const chdr = block_header.parseChunkHeader(slz_bytes[pos..], ihdr.use_checksums) catch
            return error.BadChunkHeader;
        pos += chdr.bytes_consumed;
        if (chdr.is_memset) return error.BadChunkHeader;

        const chunk_payload_start = pos;
        const chunk_payload_end = pos + chdr.compressed_size;
        if (chunk_payload_end > chunks_end) return error.Truncated;

        // 4a. Sub-chunk header.
        if (chunk_payload_start + SUBCHUNK_HDR_BYTES > chunk_payload_end) {
            return error.Truncated;
        }
        const sc_word = readBE24(slz_bytes[pos..][0..3]);
        const sc_payload_size = sc_word & 0x7FFFF; // SUBCHUNK_COMP_SIZE_MASK
        const sc_is_lz = (sc_word & SUBCHUNK_LZ_FLAG_BIT) != 0;
        if (!sc_is_lz) return error.BadSubChunkHeader;
        pos += SUBCHUNK_HDR_BYTES;
        if (pos + sc_payload_size > chunk_payload_end) {
            return error.BadSubChunkHeader;
        }
        const sc_end = pos + sc_payload_size;

        // Per-chunk decompressed size hoisted above the uncompressed
        // branch — reused here without recomputing.
        const chunk_decomp_size = chunk_decomp_size_top;

        // 4b. Capture the 8 initial bytes on the first sub-chunk so we
        //     can splice them into the lit stream for the Vulkan
        //     decoder.  The CUDA encoder seeds its anchor at 8 for
        //     chunk 0 — the lit stream covers positions 8..src_size —
        //     and the decoder restores dst[0..8] via the sub-chunk
        //     prefix.  The Vulkan decoder has no equivalent prefix
        //     copy (anchor starts at 0 in the encoder per the spec),
        //     so we prepend the 8 prefix bytes + two short-form tokens
        //     that emit them as a literal run before any encoder
        //     tokens execute.  The synthetic tokens carry the
        //     "use_recent" flag (so they don't perturb the
        //     decoder-side `recent_offset` from its INITIAL_RECENT_OFFSET
        //     = -8 seed value) and have `match_len = 0` so no match
        //     copy fires.
        var first_chunk_prefix: [INITIAL_LITERAL_COPY_BYTES]u8 = undefined;
        const has_initial_prefix = (ci == 0);
        if (has_initial_prefix) {
            if (pos + INITIAL_LITERAL_COPY_BYTES > sc_end) return error.Truncated;
            @memcpy(first_chunk_prefix[0..], slz_bytes[pos..][0..INITIAL_LITERAL_COPY_BYTES]);
            pos += INITIAL_LITERAL_COPY_BYTES;
        }

        // 4c. Literal stream (type-0 raw, 3-byte BE size header).
        //     For chunk 0 we prepend the 8 prefix bytes captured in 4b
        //     so the Vulkan decoder reproduces them at dst[0..8].
        if (pos + 3 > sc_end) return error.Truncated;
        // Validate chunk type = 0 (the high nibble of byte 0 carries it).
        if ((slz_bytes[pos] & 0x80) != 0) return error.BadSubChunkHeader;
        if (((slz_bytes[pos] >> 4) & 0x07) != 0) return error.BadSubChunkHeader;
        const lit_size: u32 = readBE24(slz_bytes[pos..][0..3]);
        pos += 3;
        if (pos + lit_size > sc_end) return error.Truncated;
        const lit_prefix_bytes: u32 = if (has_initial_prefix) INITIAL_LITERAL_COPY_BYTES else 0;
        const total_lit: u32 = lit_size + lit_prefix_bytes;
        const lit_view = storage.lit_arena[lit_arena_pos..][0..total_lit];
        if (has_initial_prefix) {
            @memcpy(lit_view[0..INITIAL_LITERAL_COPY_BYTES], first_chunk_prefix[0..]);
            @memcpy(lit_view[INITIAL_LITERAL_COPY_BYTES..], slz_bytes[pos..][0..lit_size]);
        } else {
            @memcpy(lit_view, slz_bytes[pos..][0..lit_size]);
        }
        storage.lit_views[ci] = lit_view;
        lit_arena_pos += total_lit;
        pos += lit_size;

        // 4d. Command stream (type-0 raw, 3-byte BE size header).
        //     For chunk 0 we prepend two short-form tokens that consume
        //     the 8 prefix literal bytes above (`0x87` emits 7 lits
        //     with recent_flag set + match_len 0; `0x81` emits 1 lit
        //     with the same flags).  These don't perturb the encoder-
        //     side `recent_offset = INITIAL_RECENT_OFFSET = -8` seed
        //     because the use_recent path leaves recent_offset
        //     unchanged, and `match_len = 0` means no off16 read.
        if (pos + 3 > sc_end) return error.Truncated;
        if ((slz_bytes[pos] & 0x80) != 0) return error.BadSubChunkHeader;
        if (((slz_bytes[pos] >> 4) & 0x07) != 0) return error.BadSubChunkHeader;
        const cmd_size: u32 = readBE24(slz_bytes[pos..][0..3]);
        pos += 3;
        if (pos + cmd_size > sc_end) return error.Truncated;
        const cmd_prefix_bytes: u32 = if (has_initial_prefix) 2 else 0;
        const total_cmd: u32 = cmd_size + cmd_prefix_bytes;
        const cmd_view = storage.cmd_arena[cmd_arena_pos..][0..total_cmd];
        if (has_initial_prefix) {
            cmd_view[0] = 0x87; // lit=7, match=0, use_recent=1
            cmd_view[1] = 0x81; // lit=1, match=0, use_recent=1
            @memcpy(cmd_view[2..], slz_bytes[pos..][0..cmd_size]);
        } else {
            @memcpy(cmd_view, slz_bytes[pos..][0..cmd_size]);
        }
        storage.cmd_views[ci] = cmd_view;
        cmd_arena_pos += total_cmd;
        pos += cmd_size;

        // 4e. cmd_stream2_offset (2 bytes LE) if sub-chunk > 64 KiB.
        if (chunk_decomp_size > LZ_BLOCK_SIZE) {
            if (pos + 2 > sc_end) return error.Truncated;
            pos += 2; // value not retained — the Vulkan decoder doesn't need it.
        }

        // 4f. off16 stream.
        if (pos + 2 > sc_end) return error.Truncated;
        const off16_count: u32 = std.mem.readInt(u16, slz_bytes[pos..][0..2], .little);
        pos += 2;
        if (off16_count == 0xFFFF) {
            // Entropy-coded off16 — L1 never emits this; reject early so the
            // Vulkan decoder isn't fed a bogus offset stream.
            return error.BadSubChunkHeader;
        }
        const off16_bytes: u32 = off16_count * 2;
        if (pos + off16_bytes > sc_end) return error.Truncated;
        const off16_view = storage.off16_arena[off16_arena_pos..][0..off16_bytes];
        @memcpy(off16_view, slz_bytes[pos..][0..off16_bytes]);
        storage.off16_views[ci] = off16_view;
        off16_arena_pos += off16_bytes;
        pos += off16_bytes;

        // 4g. off32 packed counts header.
        if (pos + 3 > sc_end) return error.Truncated;
        const off32_packed: u32 = @as(u32, slz_bytes[pos + 0]) |
            (@as(u32, slz_bytes[pos + 1]) << 8) |
            (@as(u32, slz_bytes[pos + 2]) << 16);
        pos += 3;
        if (off32_packed != 0) {
            // L1 never emits off32 entries; reject so the Vulkan decoder
            // doesn't silently lose data.
            return error.BadSubChunkHeader;
        }

        // 4h. Length stream — everything left in the sub-chunk.
        const length_size: u32 = @intCast(sc_end - pos);
        const length_view = storage.length_arena[length_arena_pos..][0..length_size];
        @memcpy(length_view, slz_bytes[pos..][0..length_size]);
        storage.length_views[ci] = length_view;
        length_arena_pos += length_size;
        pos += length_size;

        // 4i. Verify the chunk's compressed_size matched what we consumed.
        if (pos != chunk_payload_end) {
            // The chunk header advertised more bytes than the sub-chunk
            // walk consumed — would mean a stale chdr.compressed_size
            // vs. sc_payload_size disagreement.  Treat as corruption.
            return error.BadSubChunkHeader;
        }
    }

    // ── 5. Final layout sanity checks. ──────────────────────────
    if (pos != chunks_end) return error.Truncated;

    out_streams.* = .{
        .lit_bytes = storage.lit_views,
        .cmd_bytes = storage.cmd_views,
        .off16_bytes = storage.off16_views,
        .length_bytes = storage.length_views,
        .src_bytes = null,
        .n_chunks = n_chunks,
        .chunk_size = eff_chunk_size,
        .original_size = original_size,
    };

    return .{ .result = result, .storage = storage };
}
