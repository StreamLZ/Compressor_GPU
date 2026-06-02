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
const wire_constants = @import("wire_constants.zig");

// ── Imports of the CPU codec's wire-format helpers ────────────────
// Imported via relative path.  This module compiles only as part of
// the `vk-wire-format-test` build target, whose root file at the repo
// top (`wire_format_test_root.zig`) widens the package boundary so
// `../src/...` resolves through `src_vulkan/wire_format.zig`.
const frame = @import("../src/format/frame_format.zig");
const block_header = @import("../src/format/block_header.zig");
const constants = @import("../src/format/streamlz_constants.zig");

// ── Wire-format constants ─────────────────────────────────────────
// Sourced from `wire_constants.zig` (Cluster H consolidation). Kept as
// local re-bindings so the existing code reads unchanged.

const LZ_BLOCK_SIZE: u32 = wire_constants.LZ_BLOCK_SIZE;
const INITIAL_LITERAL_COPY_BYTES: u32 = wire_constants.INITIAL_LITERAL_COPY_BYTES;
const SUBCHUNK_LZ_FLAG_BIT: u32 = wire_constants.SUBCHUNK_LZ_FLAG_BIT;
const SUBCHUNK_MODE_SHIFT: u5 = wire_constants.SUBCHUNK_MODE_SHIFT;
const SUBCHUNK_HDR_BYTES: u32 = wire_constants.SUBCHUNK_HDR_BYTES;
const CHUNK_INTERNAL_HDR_BYTES: u32 = wire_constants.CHUNK_INTERNAL_HDR_BYTES;
const OFF32_ENTRY_BYTES: u32 = wire_constants.OFF32_ENTRY_BYTES;
const OFF32_COUNT_FIELD_BITS: u5 = wire_constants.OFF32_COUNT_FIELD_BITS;
const OFF32_COUNT_PACK_MAX: u32 = wire_constants.OFF32_COUNT_PACK_MAX;
const OFF32_COUNT1_SHIFT: u5 = OFF32_COUNT_FIELD_BITS; // = 12
const OFF32_COUNT2_MASK: u32 = OFF32_COUNT_PACK_MAX;   // low 12 bits
const SC_TAIL_PER_CHUNK_BYTES: u32 = wire_constants.SC_TAIL_PER_CHUNK_BYTES;

/// L1 codec chunk size — 64 KiB.  Maps 1:1 to the wire-format
/// sub_chunk_size at sc_group_size = 0.25 (matches CUDA defaults).
pub const VK_CHUNK_SIZE: u32 = l1_codec.CHUNK_SIZE;

// ── PerChunkStreams: host-visible per-chunk view of the L1 streams ─
// `wrapL1ToSlz1` consumes one of these; `unwrapSlz1ToL1Streams`
// produces one.  The byte slices are owned by the caller for the
// wrap path (typically scratch buffers filled by mapping the Vulkan
// SSBOs); they are owned by the unwrap path (allocator-allocated)
// for the decode path.

pub const PerChunkStreams = struct {
    /// One entry per chunk; lit_bytes[i] is the chunk-local literal
    /// byte run (`per_chunk_lit_size[i]` bytes). When
    /// `per_chunk_initial_copy[i] != 0`, the first
    /// INITIAL_LITERAL_COPY_BYTES bytes of lit_bytes[i] are the
    /// verbatim prefix the wrap path emits as the sub-chunk header's
    /// initial copy (and the unwrap path strips off before exposing
    /// lit_bytes to the Vulkan decoder via lit_views).
    lit_bytes: [][]const u8,
    cmd_bytes: [][]const u8,
    /// off16_bytes[i] is the raw interleaved u16-LE byte slice
    /// (`per_chunk_off16_count[i] * 2` bytes).
    off16_bytes: [][]const u8,
    length_bytes: [][]const u8,
    /// off32_bytes[i] is the chunk's concatenated off32 stream:
    /// `off32_count_block1[i] * OFF32_ENTRY_BYTES` bytes for block 0
    /// followed immediately by `off32_count_block2[i] * OFF32_ENTRY_BYTES`
    /// bytes for block 1. Empty slice when the chunk emitted no
    /// far-offset matches.
    off32_bytes: ?[][]const u8 = null,
    /// Per-chunk off32 entry counts (block 0 / block 1). When `null`,
    /// the wrap path assumes zero off32 entries per chunk (legacy).
    per_chunk_off32_count1: ?[]const u32 = null,
    per_chunk_off32_count2: ?[]const u32 = null,
    /// Per-chunk cmd_stream2_offset — token-index boundary between LZ
    /// blocks 0 and 1. Set non-zero when the chunk spans both blocks
    /// AND emitted off32 entries (else the wrap path collapses it to
    /// the legacy `cmd_size` filler so the decoder treats the whole
    /// stream as block 0). The Vulkan decoder reads this from chunk
    /// descriptor slot 14 (see lz_decode.comp head comment).
    per_chunk_cmd_stream2_offset: ?[]const u32 = null,
    /// Original source bytes — needed for the SC tail prefix
    /// (8 bytes per non-first chunk from the source).  Wrap path only;
    /// the unwrap path leaves this null.
    src_bytes: ?[]const u8,
    /// Per-chunk initial-copy byte count (0 or INITIAL_LITERAL_COPY_BYTES).
    /// Mirrors `l1_codec.L1Streams.per_chunk_initial_copy`.  When non-zero
    /// for chunk i, lit_bytes[i][0..n] is the verbatim sub-chunk prefix.
    /// Allocator-owned by the unwrap path (set to a non-null slice of
    /// `n_chunks` entries); wrap callers either pass an existing slice
    /// (the encoder's `per_chunk_initial_copy`) or leave it null to mean
    /// "chunk 0 has 8 prefix bytes, all others 0" (legacy single-source
    /// default).
    per_chunk_initial_copy: ?[]const u32 = null,
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
        const init_copy: usize = chunkInitialCopy(streams, @intCast(ci));
        // Each chunk takes the larger of the LZ-compressed bound and
        // the uncompressed-fallback bound (used when the LZ payload would
        // exceed the chunk's decompressed size — see wrapL1ToSlz1).
        // LZ bound (matches the LZ-path emission below).
        var lz: usize = CHUNK_INTERNAL_HDR_BYTES; // 2-byte internal + 4-byte chunk
        lz += SUBCHUNK_HDR_BYTES;
        lz += init_copy;
        const advertised_lit_len: usize = streams.lit_bytes[ci].len - init_copy;
        lz += 3 + advertised_lit_len;                   // type-0 lit
        lz += 3 + streams.cmd_bytes[ci].len;            // type-0 cmd
        lz += 2;                                         // cmd_stream2_offset
        lz += 2 + streams.off16_bytes[ci].len;           // raw off16
        lz += 3 + 4;                                     // off32 packed counts + 2× u16 ext (worst case)
        const off32_bytes: usize = if (streams.off32_bytes) |arr| arr[ci].len else 0;
        lz += off32_bytes;
        lz += streams.length_bytes[ci].len;              // length stream
        // Uncompressed-fallback bound: 2-byte internal hdr + chunk_size raw bytes.
        const chunk_dst_off: u64 = @as(u64, ci) * streams.chunk_size;
        const decomp: usize = if (streams.original_size > chunk_dst_off)
            @min(@as(usize, @intCast(streams.original_size - chunk_dst_off)), streams.chunk_size)
        else
            0;
        const uncomp: usize = 2 + decomp;
        total += if (lz > uncomp) lz else uncomp;
    }
    if (streams.n_chunks > 1) {
        total += @as(usize, streams.n_chunks - 1) * SC_TAIL_PER_CHUNK_BYTES;
    }
    total += 4; // end mark
    return total;
}

/// Per-chunk initial-copy byte count. Honors the caller-supplied
/// `per_chunk_initial_copy` slice when present; otherwise applies the
/// legacy default (chunk 0 carries 8 bytes when the chunk is large
/// enough, every other chunk carries none).
fn chunkInitialCopy(streams: PerChunkStreams, ci: u32) usize {
    if (streams.per_chunk_initial_copy) |pci| {
        if (ci < pci.len) return pci[ci];
    }
    if (ci == 0 and streams.lit_bytes[0].len >= INITIAL_LITERAL_COPY_BYTES) {
        return INITIAL_LITERAL_COPY_BYTES;
    }
    return 0;
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
    // src_bytes only consumed for the (n_chunks > 1) SC tail prefix; for
    // single-chunk encodes we can wrap without a source slice because the
    // 8-byte initial copy comes from lit_buf, not from src directly.
    const src_opt = streams.src_bytes;
    if (streams.n_chunks > 1 and src_opt == null) return error.BadHeader;
    if (out.len < wrapBound(streams)) return error.OutputTooSmall;

    var pos: usize = 0;

    // ── 1. Frame header ──────────────────────────────────────────
    const hdr_n = frame.writeHeader(out[pos..], .{
        .codec = .fast,
        .level = 1,
        .block_size = constants.chunk_size, // 256 KiB
        .sc_group_size = wire_constants.SC_GROUP_SIZE,
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

        const init_copy: u32 = @intCast(chunkInitialCopy(streams, ci));

        // Pre-compute the LZ sub-chunk payload size so we can decide
        // between LZ-compressed and raw emission for this chunk.
        const advertised_lit_len: usize = streams.lit_bytes[ci].len - init_copy;
        const off32_count1_ci: u32 = if (streams.per_chunk_off32_count1) |a| a[ci] else 0;
        const off32_count2_ci: u32 = if (streams.per_chunk_off32_count2) |a| a[ci] else 0;
        const off32_bytes_ci: usize = if (streams.off32_bytes) |a| a[ci].len else 0;
        // CUDA's packed-count header has a 12-bit field per block;
        // counts ≥ 4095 spill into a u16 extension after the 3-byte
        // packed word (one extension per block that overflows).
        const off32_ext_bytes: usize =
            (@as(usize, if (off32_count1_ci >= OFF32_COUNT_PACK_MAX) 2 else 0)) +
            (@as(usize, if (off32_count2_ci >= OFF32_COUNT_PACK_MAX) 2 else 0));
        const sub_payload_size: u32 = subChunkPayloadSize(
            ci,
            chunk_decomp_size,
            advertised_lit_len,
            streams.cmd_bytes[ci].len,
            streams.off16_bytes[ci].len,
            streams.length_bytes[ci].len,
            init_copy,
            off32_bytes_ci,
            off32_ext_bytes,
        );

        // ── Emit-as-uncompressed path ───────────────────────────────
        // CUDA's per-warp decoder picks the raw-copy fast path whenever
        // the sub-chunk's compressed size is >= the decompressed size
        // (`if (sc_comp_size < sc_size)` → LZ; else raw, in
        // lz_decode_kernels.cuh::slzLzDecodeRawKernel). If the encoder
        // produced an LZ payload that doesn't actually compress (which
        // happens on high-entropy chunks — silesia x-ray data is the
        // canonical case), the LZ-path bytes get reinterpreted as raw
        // source bytes and the decoded chunk is garbage. The fix mirrors
        // what the CUDA encoder does for the same chunk: emit it as a
        // whole-chunk uncompressed block. The internal block header has
        // bit 7 (uncompressed) set, no 4-byte chunk header follows, and
        // the 2-byte hdr is immediately followed by `chunk_decomp_size`
        // raw source bytes. The chunk's per-stream buffers (lit / cmd /
        // ...) are dropped — they're now unused.
        const lz_payload_size: u32 = SUBCHUNK_HDR_BYTES + sub_payload_size;
        const emit_uncompressed = sub_payload_size >= chunk_decomp_size and
            src_opt != null and
            chunk_dst_off + chunk_decomp_size <= src_opt.?.len;
        if (emit_uncompressed) {
            // 2-byte internal block header with uncompressed flag set.
            const u_hdr0: u8 = 0x05 | 0x10 | 0x40 | 0x80;
            const u_hdr1: u8 = @intFromEnum(block_header.CodecType.fast);
            out[pos + 0] = u_hdr0;
            out[pos + 1] = u_hdr1;
            pos += 2;
            // Raw chunk_decomp_size source bytes.
            const src = src_opt.?;
            @memcpy(
                out[pos..][0..chunk_decomp_size],
                src[@intCast(chunk_dst_off)..][0..chunk_decomp_size],
            );
            pos += chunk_decomp_size;
            continue;
        }

        // 3a. 2-byte internal block header.
        //   byte 0 = magic(0x05) | self_contained(0x10) | restart_decoder(0x40)
        //   byte 1 = decoder_type = Fast (1).
        const internal_hdr0: u8 = 0x05 | 0x10 | 0x40;
        const internal_hdr1: u8 = @intFromEnum(block_header.CodecType.fast);
        out[pos + 0] = internal_hdr0;
        out[pos + 1] = internal_hdr1;
        pos += 2;

        // 3b. 4-byte LE chunk header (compressed_size - 1 in low 18 bits).
        //   compressed_size = 3-byte sub-chunk header + sub-chunk payload bytes.
        const chunk_compressed_size: u32 = lz_payload_size;
        std.mem.writeInt(u32, out[pos..][0..4], chunk_compressed_size - 1, .little);
        pos += 4;

        // 3c. 3-byte BE sub-chunk header.
        const sc_hdr_word: u32 = sub_payload_size |
            (@as(u32, 1) << SUBCHUNK_MODE_SHIFT) |
            SUBCHUNK_LZ_FLAG_BIT;
        writeBE24(out[pos..][0..3], sc_hdr_word);
        pos += 3;

        // 3d.0 Initial verbatim copy (chunk 0 only — INITIAL_LITERAL_COPY_BYTES
        //      source bytes that the CUDA decoder writes straight to
        //      dst[0..8] via the `base_offset == 0` branch). Encoder put
        //      these as the first 8 bytes of lit_buf; wrap pulls them
        //      off the front so the lit-stream header advertises only
        //      the tokens-driven literals.
        if (init_copy != 0) {
            @memcpy(out[pos..][0..init_copy], streams.lit_bytes[ci][0..init_copy]);
            pos += init_copy;
        }

        // 3d.i Literal stream (type-0 raw) — excludes the initial copy.
        writeBE24(out[pos..][0..3], @intCast(advertised_lit_len));
        pos += 3;
        @memcpy(out[pos..][0..advertised_lit_len], streams.lit_bytes[ci][init_copy..]);
        pos += advertised_lit_len;

        // 3d.ii Command/token stream (type-0 raw).
        writeBE24(out[pos..][0..3], @intCast(streams.cmd_bytes[ci].len));
        pos += 3;
        @memcpy(out[pos..][0..streams.cmd_bytes[ci].len], streams.cmd_bytes[ci]);
        pos += streams.cmd_bytes[ci].len;

        // 3d.iii cmd_stream2_offset (2 bytes LE) — present because
        //         sub_decomp_size > LZ_BLOCK_SIZE. Use the encoder's
        //         actual block-0/1 token boundary so the CUDA decoder
        //         routes off32 entries to the correct block. When the
        //         encoder didn't split (single-block chunk encoded as
        //         128 KiB anyway, e.g. last chunk smaller than 64 KiB
        //         wouldn't even reach this branch), fall back to
        //         cmd_size — "everything is block 0" — which CUDA
        //         tolerates because the block-1 trailing path then sees
        //         dst_pos >= block_end and emits nothing.
        if (chunk_decomp_size > LZ_BLOCK_SIZE) {
            const cs2o_raw: u32 = if (streams.per_chunk_cmd_stream2_offset) |arr| arr[ci] else 0;
            const cs2o: u16 = if (cs2o_raw != 0 and cs2o_raw < streams.cmd_bytes[ci].len)
                @intCast(cs2o_raw)
            else
                @intCast(streams.cmd_bytes[ci].len);
            std.mem.writeInt(u16, out[pos..][0..2], cs2o, .little);
            pos += 2;
        }

        // 3d.iv off16 stream (small raw form: count < OFF16_ENTROPY_MIN).
        const off16_count: u32 = @intCast(streams.off16_bytes[ci].len / 2);
        std.mem.writeInt(u16, out[pos..][0..2], @intCast(off16_count), .little);
        pos += 2;
        @memcpy(out[pos..][0..streams.off16_bytes[ci].len], streams.off16_bytes[ci]);
        pos += streams.off16_bytes[ci].len;

        // 3d.v off32 packed counts header (3 bytes LE) — `(count1 << 12)
        //      | count2`, clamped to OFF32_COUNT_PACK_MAX with the
        //      overflow spilling into a trailing u16 LE per block.
        const c1_clamped: u32 = if (off32_count1_ci < OFF32_COUNT_PACK_MAX) off32_count1_ci else OFF32_COUNT_PACK_MAX;
        const c2_clamped: u32 = if (off32_count2_ci < OFF32_COUNT_PACK_MAX) off32_count2_ci else OFF32_COUNT_PACK_MAX;
        const packed_counts: u32 = (c1_clamped << OFF32_COUNT1_SHIFT) | c2_clamped;
        writeLE24(out[pos..][0..3], packed_counts);
        pos += 3;
        if (off32_count1_ci >= OFF32_COUNT_PACK_MAX) {
            std.mem.writeInt(u16, out[pos..][0..2], @intCast(off32_count1_ci), .little);
            pos += 2;
        }
        if (off32_count2_ci >= OFF32_COUNT_PACK_MAX) {
            std.mem.writeInt(u16, out[pos..][0..2], @intCast(off32_count2_ci), .little);
            pos += 2;
        }
        if (streams.off32_bytes) |off32_arr| {
            @memcpy(out[pos..][0..off32_arr[ci].len], off32_arr[ci]);
            pos += off32_arr[ci].len;
        }

        // 3d.vi length stream (raw).
        @memcpy(out[pos..][0..streams.length_bytes[ci].len], streams.length_bytes[ci]);
        pos += streams.length_bytes[ci].len;
    }

    // ── 4. SC tail prefix (n_chunks - 1 entries of 8 bytes each). ─
    //       Each entry restores the 8 verbatim source bytes the decoder
    //       skips at the head of every non-first chunk.
    if (streams.n_chunks > 1) {
        const src = src_opt.?;
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
    init_copy: u32,
    off32_bytes: usize,
    off32_ext_bytes: usize,
) u32 {
    _ = chunk_idx;
    var n: u32 = init_copy; // INITIAL_LITERAL_COPY_BYTES verbatim prefix (chunk 0)
    n += 3 + @as(u32, @intCast(lit_size)); // type-0 lit
    n += 3 + @as(u32, @intCast(cmd_size)); // type-0 cmd
    if (chunk_decomp_size > LZ_BLOCK_SIZE) n += 2; // cmd_stream2_offset
    n += 2 + @as(u32, @intCast(off16_bytes)); // off16 count + bytes
    n += 3 + @as(u32, @intCast(off32_ext_bytes)); // off32 packed counts + optional u16 ext per block
    n += @as(u32, @intCast(off32_bytes));         // off32 entry bytes
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
    off32_arena: []u8,
    lit_views: [][]const u8,
    cmd_views: [][]const u8,
    off16_views: [][]const u8,
    length_views: [][]const u8,
    off32_views: [][]const u8,
    /// Per-chunk initial-copy byte count. The Vulkan decoder consumes
    /// this via the chunk descriptor (slot 10) to pre-write dst[0..n]
    /// from the head of lit_views[i] before any tokens execute.
    initial_copy: []u32,
    off32_count1: []u32,
    off32_count2: []u32,
    cmd_stream2_offset: []u32,
};

pub fn freeUnwrapStorage(allocator: std.mem.Allocator, s: *UnwrapStorage) void {
    allocator.free(s.lit_arena);
    allocator.free(s.cmd_arena);
    allocator.free(s.off16_arena);
    allocator.free(s.length_arena);
    allocator.free(s.off32_arena);
    allocator.free(s.lit_views);
    allocator.free(s.cmd_views);
    allocator.free(s.off16_views);
    allocator.free(s.length_views);
    allocator.free(s.off32_views);
    allocator.free(s.initial_copy);
    allocator.free(s.off32_count1);
    allocator.free(s.off32_count2);
    allocator.free(s.cmd_stream2_offset);
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
        .off32_arena = try allocator.alloc(u8, original_size + 1024),
        .lit_views = try allocator.alloc([]const u8, n_chunks),
        .cmd_views = try allocator.alloc([]const u8, n_chunks),
        .off16_views = try allocator.alloc([]const u8, n_chunks),
        .length_views = try allocator.alloc([]const u8, n_chunks),
        .off32_views = try allocator.alloc([]const u8, n_chunks),
        .initial_copy = try allocator.alloc(u32, n_chunks),
        .off32_count1 = try allocator.alloc(u32, n_chunks),
        .off32_count2 = try allocator.alloc(u32, n_chunks),
        .cmd_stream2_offset = try allocator.alloc(u32, n_chunks),
    };
    errdefer freeUnwrapStorage(allocator, &storage);
    @memset(storage.initial_copy, 0);
    @memset(storage.off32_count1, 0);
    @memset(storage.off32_count2, 0);
    @memset(storage.cmd_stream2_offset, 0);

    var lit_arena_pos: usize = 0;
    var cmd_arena_pos: usize = 0;
    var off16_arena_pos: usize = 0;
    var length_arena_pos: usize = 0;
    var off32_arena_pos: usize = 0;

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

            // Empty cmd / off16 / off32 / length streams — the Vulkan
            // decoder's trailing-literal routine flushes the remaining
            // lit bytes when cmd has no tokens to process.
            const empty_cmd = storage.cmd_arena[cmd_arena_pos..][0..0];
            storage.cmd_views[ci] = empty_cmd;
            const empty_off16 = storage.off16_arena[off16_arena_pos..][0..0];
            storage.off16_views[ci] = empty_off16;
            const empty_off32 = storage.off32_arena[off32_arena_pos..][0..0];
            storage.off32_views[ci] = empty_off32;
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

        // 4b. Capture the 8 initial bytes on the first sub-chunk and
        //     splice them onto the front of lit_view so the Vulkan
        //     decoder can reproduce them at dst[0..8] via its slot-10
        //     initial-copy descriptor field. (Earlier ports synthesised
        //     two `0x8X` literal-only tokens at the head of cmd_view to
        //     consume the prefix; that workaround is gone now that the
        //     decoder shader handles initial_copy natively.)
        var first_chunk_prefix: [INITIAL_LITERAL_COPY_BYTES]u8 = undefined;
        const has_initial_prefix = (ci == 0);
        if (has_initial_prefix) {
            if (pos + INITIAL_LITERAL_COPY_BYTES > sc_end) return error.Truncated;
            @memcpy(first_chunk_prefix[0..], slz_bytes[pos..][0..INITIAL_LITERAL_COPY_BYTES]);
            pos += INITIAL_LITERAL_COPY_BYTES;
            storage.initial_copy[ci] = INITIAL_LITERAL_COPY_BYTES;
        }

        // 4c. Literal stream (type-0 raw, 3-byte BE size header).
        //     For chunk 0 we prepend the 8 prefix bytes captured in 4b
        //     so the decoder's initial-copy step reads them from the
        //     head of lit_view.
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
        if (pos + 3 > sc_end) return error.Truncated;
        if ((slz_bytes[pos] & 0x80) != 0) return error.BadSubChunkHeader;
        if (((slz_bytes[pos] >> 4) & 0x07) != 0) return error.BadSubChunkHeader;
        const cmd_size: u32 = readBE24(slz_bytes[pos..][0..3]);
        pos += 3;
        if (pos + cmd_size > sc_end) return error.Truncated;
        const cmd_view = storage.cmd_arena[cmd_arena_pos..][0..cmd_size];
        @memcpy(cmd_view, slz_bytes[pos..][0..cmd_size]);
        storage.cmd_views[ci] = cmd_view;
        cmd_arena_pos += cmd_size;
        pos += cmd_size;

        // 4e. cmd_stream2_offset (2 bytes LE) if sub-chunk > 64 KiB.
        if (chunk_decomp_size > LZ_BLOCK_SIZE) {
            if (pos + 2 > sc_end) return error.Truncated;
            const cs2o: u32 = std.mem.readInt(u16, slz_bytes[pos..][0..2], .little);
            storage.cmd_stream2_offset[ci] = cs2o;
            pos += 2;
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

        // 4g. off32 packed counts header (3 bytes LE). Bits [23:12] =
        //     count1, bits [11:0] = count2 (each clamped to 0xFFF in
        //     the header; ≥ 0xFFF values spill into a trailing u16 LE
        //     extension per block).
        if (pos + 3 > sc_end) return error.Truncated;
        const off32_packed: u32 = @as(u32, slz_bytes[pos + 0]) |
            (@as(u32, slz_bytes[pos + 1]) << 8) |
            (@as(u32, slz_bytes[pos + 2]) << 16);
        pos += 3;
        var off32_count1: u32 = off32_packed >> OFF32_COUNT1_SHIFT;
        var off32_count2: u32 = off32_packed & OFF32_COUNT2_MASK;
        if (off32_count1 == OFF32_COUNT_PACK_MAX) {
            if (pos + 2 > sc_end) return error.Truncated;
            off32_count1 = std.mem.readInt(u16, slz_bytes[pos..][0..2], .little);
            pos += 2;
        }
        if (off32_count2 == OFF32_COUNT_PACK_MAX) {
            if (pos + 2 > sc_end) return error.Truncated;
            off32_count2 = std.mem.readInt(u16, slz_bytes[pos..][0..2], .little);
            pos += 2;
        }
        const off32_total_bytes: u32 = (off32_count1 + off32_count2) * OFF32_ENTRY_BYTES;
        if (pos + off32_total_bytes > sc_end) return error.Truncated;
        const off32_view = storage.off32_arena[off32_arena_pos..][0..off32_total_bytes];
        @memcpy(off32_view, slz_bytes[pos..][0..off32_total_bytes]);
        storage.off32_views[ci] = off32_view;
        storage.off32_count1[ci] = off32_count1;
        storage.off32_count2[ci] = off32_count2;
        off32_arena_pos += off32_total_bytes;
        pos += off32_total_bytes;

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
        .off32_bytes = storage.off32_views,
        .per_chunk_off32_count1 = storage.off32_count1,
        .per_chunk_off32_count2 = storage.off32_count2,
        .per_chunk_cmd_stream2_offset = storage.cmd_stream2_offset,
        .src_bytes = null,
        .per_chunk_initial_copy = storage.initial_copy,
        .n_chunks = n_chunks,
        .chunk_size = eff_chunk_size,
        .original_size = original_size,
    };

    return .{ .result = result, .storage = storage };
}
