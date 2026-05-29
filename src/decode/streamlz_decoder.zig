//! StreamLZ framed decompressor — public API.
//!
//! The decoder walks an SLZ1 frame's outer block list, parses each block's
//! internal header, builds per-chunk descriptors, and hands the descriptor
//! list to the GPU dispatch (`gpu_driver.fullGpuLaunchImpl`).
//!
//! Two entry points exist:
//!
//!   * `decompressFramed` / `decompressFramedThreaded` — host-input,
//!     host-output. The decoder fills the caller's `dst` slice; the
//!     compressed `src` and the chunk descriptors are H2D-copied inside
//!     the dispatch.
//!   * `decompressFramedFromDevice` — device-input, device-output (the
//!     pure-D2D path used by the v3 C ABI). The frame and output never
//!     leave VRAM; the chunk walk runs on the GPU via
//!     `gpu_driver.gpuWalkFrameImpl`.

const std = @import("std");

const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const constants = @import("../format/streamlz_constants.zig");
const gpu_driver = @import("driver.zig");

/// Bytes the decoder is allowed to overshoot past the requested output
/// length. Several copy helpers prefetch and write ahead by up to this
/// many bytes; the caller-supplied buffer must include the slack.
pub const safe_space = constants.safe_space;

/// Upper bound on `content_size` the decoder will honor. A malicious frame
/// header can claim absurd sizes; the decoder rejects anything above this
/// cap rather than allocating into hostile territory.
pub const max_content_size: u64 = 4 * 1024 * 1024 * 1024;

/// Errors the decoder can surface. Members embed `gpu_driver.GpuError`
/// (CUDA-side failures) and `std.mem.Allocator.Error` (host alloc), so a
/// `try`-chain through the GPU dispatch propagates without re-wrapping.
pub const DecompressError = error{
    /// Frame header failed magic / version / codec / flag validation.
    BadFrame,
    /// Source ran out of bytes before the parser could complete a header
    /// field. The caller did not pass the whole frame.
    Truncated,
    /// A nested length field disagrees with its container - e.g. the
    /// frame's `content_size` is smaller than the sum of decompressed
    /// block sizes.
    SizeMismatch,
    /// The 2-byte SLZ internal block header (magic / decoder / checksum
    /// bit) is malformed.
    InvalidBlockHeader,
    /// The 4-byte chunk header word carries a reserved type or sets
    /// a reserved bit.
    InvalidInternalHeader,
    /// A chunk header's `compressed_size` field is invalid (zero or
    /// larger than the surrounding block payload).
    BadChunkHeader,
    /// The block claimed more compressed bytes than the frame contains.
    BlockDataTruncated,
    /// `dst` is smaller than the frame's declared / computed
    /// decompressed-byte count plus `safe_space`.
    OutputTooSmall,
    /// A per-block or per-chunk CRC24 / content checksum mismatched
    /// (currently parsed-but-not-verified; reserved for a future
    /// strict-mode flag).
    ChecksumMismatch,
    /// A chunk's `decompressed_size` disagrees with its dst-stride.
    ChunkSizeMismatch,
    /// The frame carries a `dictionary_id`; the GPU decoder has no
    /// dictionary store. Always rejected.
    UnknownDictionary,
    /// `content_size` exceeds `max_content_size` (4 GiB). Either the
    /// frame is hostile or it was produced by an encoder that exceeds
    /// this decoder's cap.
    ContentSizeTooLarge,
} || gpu_driver.GpuError || std.mem.Allocator.Error;

pub const DecompressResult = struct {
    written: usize,
    /// Always 0 in the GPU-only codec. Reserved for compatibility with
    /// the original dictionary-prefix offset returned by the CPU decoder
    /// — the field is still read by some callers.
    offset: usize = 0,
};

/// Decompress one SLZ1 frame from host bytes `src` into host bytes
/// `dst`. The compressed bytes are H2D-copied inside the GPU dispatch;
/// the decompressed bytes are D2H'd back into `dst`.
///
/// Returns the number of bytes written to `dst` (always equal to the
/// frame's declared `content_size` when the call succeeds).
///
/// Caller owns `dec_ctx`. The context is reused across calls so its
/// device buffers can grow; the caller is responsible for calling
/// `dec_ctx.deinit()` when done with the context.
pub fn decompressFramed(
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
) DecompressError!usize {
    const r = try decompressFrameInner(src, dst, dec_ctx, null, null);
    return r.written;
}

/// Same contract as `decompressFramed`, plus optional `std.Io` for
/// telemetry (per-phase elapsed times via `SLZ_E2E_TIMER` /
/// `SLZ_SPLIT_TIMER` are gated on `io != null`). Returns the structured
/// `DecompressResult` so the offset field is observable to callers that
/// want it; `decompressFramed` is a `.written`-only convenience.
///
/// The `allocator` parameter is currently unused (the dispatch takes
/// the allocator from the encode/decode context). Plumbed in for ABI
/// compat with the previous CPU codec entry point; callers can pass
/// `undefined`.
pub fn decompressFramedThreaded(
    allocator: std.mem.Allocator,
    io: ?std.Io,
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
) DecompressError!DecompressResult {
    _ = allocator;
    return decompressFrameInner(src, dst, dec_ctx, null, io);
}

/// True-D2D decompress for the v3 C ABI. Compressed frame and output
/// stay on device throughout. `decomp_size` is the caller-supplied
/// decompressed-byte count (from `slzGetDecompressedSize` or caller
/// metadata) — the v3 contract requires it.
///
/// Accepts only the slzCompress-shape frame: Fast codec, single block,
/// content_size present, sc_group ≤ 1.0, no dictionary. Anything else
/// returns `error.BadMode` and the C ABI falls back to the host-bounce
/// entry points.
pub fn decompressFramedFromDevice(
    io: ?std.Io,
    d_frame: u64,
    frame_size: u32,
    d_output: u64,
    dec_ctx: *gpu_driver.DecodeContext,
    decomp_size: u32,
) DecompressError!u32 {
    // Frame layout: [hdr 14][content_size 8][block_hdr 8][block][end 4]
    const fixed_overhead: u32 = 14 + 8 + 8 + 4;
    if (frame_size < fixed_overhead + 1) return error.BadMode;
    const block_start: u32 = 14 + 8 + 8;
    const block_size: u32 = frame_size - fixed_overhead;

    // sc_group=0.25 ⇒ eff_chunk_size = 64 KB. The walk kernel writes the
    // real chunk count to d_meta+0; the bound below just sizes the
    // descriptor + grid space.
    const eff_chunk_64k: u32 = 0x10000;
    const n_chunks_bound: u32 = (decomp_size + eff_chunk_64k - 1) / eff_chunk_64k;
    if (n_chunks_bound == 0 or n_chunks_bound > gpu_driver.WALK_MAX_CHUNKS) return error.BadMode;

    const dev = try gpu_driver.gpuWalkFrameImpl(dec_ctx, d_frame, frame_size);

    // Stub slices: the kernel reads from `d_chunk_descs_override` and
    // `d_compressed_src`; the host slices are length-only.
    const stub_chunks_ptr: [*]const gpu_driver.ChunkDesc = @ptrFromInt(0x10);
    const chunks_stub: []const gpu_driver.ChunkDesc = stub_chunks_ptr[0..n_chunks_bound];
    const stub_bytes_ptr: [*]const u8 = @ptrFromInt(0x10);
    const compressed_block: []const u8 = stub_bytes_ptr[0..block_size];

    var dst_dummy: u8 = 0;
    try gpu_driver.fullGpuLaunchImpl(dec_ctx, .{
        .chunk_descs = chunks_stub,
        .compressed_block = compressed_block,
        .dst_full = @ptrCast(&dst_dummy),
        .dst_start_off = 0,
        .decompressed_size = decomp_size,
        .chunks_per_group = 1,
        .sub_chunk_cap = @intCast(gpu_driver.ENTROPY_SCRATCH_SLOT_BYTES),
        .io = io,
        .d_output_target = d_output,
        .d_compressed_src = d_frame + block_start,
        .d_chunk_descs_override = dev.d_chunk_descs,
        .d_n_chunks_dev = dev.d_meta + gpu_driver.walk_meta_offsets.n_chunks,
    });
    return decomp_size;
}

pub const DecompressContext = struct {
    allocator: std.mem.Allocator,
    io: ?std.Io,
    dec_ctx: *gpu_driver.DecodeContext = &gpu_driver.g_default,

    pub fn init(allocator: std.mem.Allocator) DecompressContext {
        return .{ .allocator = allocator, .io = null };
    }

    pub fn initWithIo(allocator: std.mem.Allocator, io: std.Io) DecompressContext {
        return .{ .allocator = allocator, .io = io };
    }

    pub fn decompress(self: *DecompressContext, src: []const u8, dst: []u8) DecompressError!DecompressResult {
        return decompressFrameInner(src, dst, self.dec_ctx, null, self.io);
    }

    pub fn deinit(self: *DecompressContext) void {
        _ = self;
    }
};

// ────────────────────────────────────────────────────────────────────────
//  Frame walk
// ────────────────────────────────────────────────────────────────────────

fn decompressFrameInner(
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
) DecompressError!DecompressResult {
    if (src.len == 0) return .{ .written = 0 };

    const hdr = frame.parseHeader(src) catch return error.BadFrame;

    // GPU codec never produces dictionary frames. The dictionary subsystem
    // was removed during the GPU-only strip; any frame carrying a
    // dictionary_id originated from the legacy CPU encoder.
    if (hdr.dictionary_id != null) return error.UnknownDictionary;

    if (hdr.content_size) |cs| {
        if (cs > max_content_size) return error.ContentSizeTooLarge;
        if (d_output_target == null and dst.len < @as(usize, @intCast(cs + safe_space))) return error.OutputTooSmall;
    }

    var pos: usize = hdr.header_size;
    var dst_off: usize = 0;

    while (pos + 4 <= src.len) {
        const first_word = std.mem.readInt(u32, src[pos..][0..4], .little);
        if (first_word == frame.end_mark) {
            pos += 4;
            break;
        }

        const block_hdr = frame.parseBlockHeader(src[pos..]) catch return error.InvalidBlockHeader;
        if (block_hdr.isEndMark()) {
            pos += 8;
            break;
        }
        pos += 8;

        // Skip any legacy sidecar (parallel-decode metadata) blocks the
        // wire format may still carry from older encoders. The GPU codec
        // does not emit them; bytes contribute 0 to dst.
        if (block_hdr.parallel_decode_metadata) {
            if (pos + block_hdr.compressed_size > src.len) return error.BlockDataTruncated;
            pos += block_hdr.compressed_size;
            continue;
        }

        if (block_hdr.uncompressed) {
            if (pos + block_hdr.decompressed_size > src.len) return error.BlockDataTruncated;
            if (d_output_target) |dev_target| {
                if (!gpu_driver.copyHostToDevice(dev_target + dst_off, src[pos..][0..block_hdr.decompressed_size]))
                    return error.BadMode;
            } else {
                if (dst_off + block_hdr.decompressed_size > dst.len) return error.OutputTooSmall;
                @memcpy(dst[dst_off..][0..block_hdr.decompressed_size], src[pos..][0..block_hdr.decompressed_size]);
            }
            dst_off += block_hdr.decompressed_size;
            pos += block_hdr.compressed_size;
            continue;
        }

        const block_src = src[pos .. pos + block_hdr.compressed_size];
        try dispatchCompressedBlock(
            block_src,
            dst,
            dst_off,
            block_hdr.decompressed_size,
            hdr.sc_group_size,
            dec_ctx,
            d_output_target,
            io,
        );
        // Restore the 8-byte SC tail prefix bytes that the encoder
        // appended at the end of the block. The decoder kernel writes
        // garbage into the first 8 bytes of every chunk past the first
        // (no Copy64 fires when base_offset != 0).
        const eff_cs = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
        const num_chunks = (block_hdr.decompressed_size + eff_cs - 1) / eff_cs;
        const prefix_sz: usize = if (num_chunks > 1) (num_chunks - 1) * 8 else 0;
        if (prefix_sz != 0) {
            const prefix_base: [*]const u8 = block_src[block_src.len - prefix_sz ..].ptr;
            for (0..num_chunks - 1) |prefix_idx| {
                const chunk_dst_off = dst_off + (prefix_idx + 1) * eff_cs;
                var copy_size: usize = 8;
                if ((prefix_idx + 1) * eff_cs + copy_size > block_hdr.decompressed_size)
                    copy_size = block_hdr.decompressed_size - (prefix_idx + 1) * eff_cs;
                if (d_output_target) |dev_target| {
                    if (!gpu_driver.copyHostToDevice(dev_target + chunk_dst_off, prefix_base[prefix_idx * 8 ..][0..copy_size]))
                        return error.BadMode;
                } else {
                    @memcpy(dst[chunk_dst_off..][0..copy_size], prefix_base[prefix_idx * 8 ..][0..copy_size]);
                }
            }
        }
        dst_off += block_hdr.decompressed_size;
        pos += block_hdr.compressed_size;
    }

    if (hdr.content_size) |cs| {
        if (dst_off != cs) return error.SizeMismatch;
    }
    return .{ .written = dst_off };
}

// ────────────────────────────────────────────────────────────────────────
//  Per-block GPU dispatch
// ────────────────────────────────────────────────────────────────────────

fn dispatchCompressedBlock(
    block_src: []const u8,
    dst: []u8,
    dst_start_off: usize,
    decompressed_size: usize,
    sc_group_size: f32,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
) DecompressError!void {
    const eff_chunk_size = @min(frame.scGroupSizeToBytes(sc_group_size), constants.chunk_size);
    const num_chunks = (decompressed_size + eff_chunk_size - 1) / eff_chunk_size;
    if (num_chunks == 0) return;
    // Strip the SC tail prefix bytes before parsing chunks — they sit
    // after the last chunk and are not part of any chunk's payload.
    if (block_src.len < 2) return error.Truncated;
    const peek = block_header.parseBlockHeader(block_src) catch return error.InvalidInternalHeader;
    const is_fast = peek.decoder_type == .fast or peek.decoder_type == .turbo;
    if (!is_fast) return error.InvalidInternalHeader;
    const prefix_sz: usize = if (peek.self_contained and num_chunks > 1) (num_chunks - 1) * 8 else 0;
    if (prefix_sz >= block_src.len) return error.Truncated;
    const block_payload = block_src[0 .. block_src.len - prefix_sz];

    const sc_grp_chunks: usize = if (sc_group_size >= 1.0)
        @max(1, @as(usize, @intFromFloat(sc_group_size)))
    else
        1;
    const chunks_per_group: u32 = if (peek.self_contained and sc_grp_chunks < num_chunks)
        @intCast(sc_grp_chunks)
    else
        @intCast(num_chunks);

    var chunk_descs_buf: [gpu_driver.WALK_MAX_CHUNKS]gpu_driver.ChunkDesc = undefined;
    if (num_chunks > chunk_descs_buf.len) return error.BadFrame;
    const chunk_descs = chunk_descs_buf[0..num_chunks];
    @memset(std.mem.sliceAsBytes(chunk_descs), 0);

    try buildChunkDescriptors(
        block_payload,
        chunk_descs,
        eff_chunk_size,
        decompressed_size,
        dst_start_off,
    );

    try gpu_driver.fullGpuLaunchImpl(dec_ctx, .{
        .chunk_descs = chunk_descs,
        .compressed_block = block_payload,
        .dst_full = dst.ptr,
        .dst_start_off = dst_start_off,
        .decompressed_size = decompressed_size,
        .chunks_per_group = chunks_per_group,
        .sub_chunk_cap = @intCast(constants.sub_chunk_size),
        .io = io,
        .d_output_target = d_output_target,
        .d_compressed_src = null,
    });
}

/// Walk the internal block to fill `chunk_descs` (one entry per chunk).
/// Sets flags=2 for memset chunks, flags=1 for "whole match equals chunk"
/// shortcuts, and flags=0 for LZ-compressed chunks. The GPU decoder reads
/// these flags to pick its per-chunk fast path.
fn buildChunkDescriptors(
    block_payload: []const u8,
    chunk_descs: []gpu_driver.ChunkDesc,
    eff_chunk_size: usize,
    decompressed_size: usize,
    dst_start_off: usize,
) DecompressError!void {
    var src_pos: usize = 0;
    var dst_remaining: usize = decompressed_size;
    var internal_hdr: ?block_header.BlockHeader = null;
    var chunk_idx: usize = 0;
    var dst_off: usize = dst_start_off;

    while (dst_remaining > 0 and chunk_idx < chunk_descs.len) {
        const at_chunk_boundary = ((dst_off - dst_start_off) % eff_chunk_size) == 0;
        if (at_chunk_boundary or internal_hdr == null) {
            if (src_pos + 2 > block_payload.len) return error.Truncated;
            internal_hdr = block_header.parseBlockHeader(block_payload[src_pos..]) catch
                return error.InvalidInternalHeader;
            src_pos += 2;
        }
        const hdr = internal_hdr.?;

        var dst_this_chunk: usize = @min(eff_chunk_size, constants.chunk_size);
        if (dst_this_chunk > dst_remaining) dst_this_chunk = dst_remaining;

        if (hdr.uncompressed) {
            chunk_descs[chunk_idx] = .{
                .src_offset = @intCast(src_pos),
                .comp_size = @intCast(dst_this_chunk),
                .decomp_size = @intCast(dst_this_chunk),
                .dst_offset = @intCast(dst_off),
                .flags = 1,
                .memset_fill = 0,
            };
            dst_off += dst_this_chunk;
            dst_remaining -= dst_this_chunk;
            src_pos += dst_this_chunk;
            chunk_idx += 1;
            continue;
        }

        const ch = block_header.parseChunkHeader(block_payload[src_pos..], hdr.use_checksums) catch
            return error.BadChunkHeader;
        src_pos += ch.bytes_consumed;

        if (ch.is_memset) {
            chunk_descs[chunk_idx] = .{
                .src_offset = 0,
                .comp_size = 0,
                .decomp_size = @intCast(dst_this_chunk),
                .dst_offset = @intCast(dst_off),
                .flags = 2,
                .memset_fill = ch.memset_fill,
            };
            dst_off += dst_this_chunk;
            dst_remaining -= dst_this_chunk;
            chunk_idx += 1;
            continue;
        }

        if (src_pos + ch.compressed_size > block_payload.len) return error.Truncated;

        // Whole-chunk shortcut: comp_size == decomp_size means the chunk
        // is raw bytes verbatim (no LZ tokens). The GPU decoder copies
        // src to dst directly.
        if (ch.compressed_size == dst_this_chunk) {
            chunk_descs[chunk_idx] = .{
                .src_offset = @intCast(src_pos),
                .comp_size = @intCast(ch.compressed_size),
                .decomp_size = @intCast(dst_this_chunk),
                .dst_offset = @intCast(dst_off),
                .flags = 1,
                .memset_fill = 0,
            };
            dst_off += dst_this_chunk;
            dst_remaining -= dst_this_chunk;
            src_pos += ch.compressed_size;
            chunk_idx += 1;
            continue;
        }

        chunk_descs[chunk_idx] = .{
            .src_offset = @intCast(src_pos),
            .comp_size = @intCast(ch.compressed_size),
            .decomp_size = @intCast(dst_this_chunk),
            .dst_offset = @intCast(dst_off),
            .flags = 0,
            .memset_fill = 0,
        };
        dst_off += dst_this_chunk;
        dst_remaining -= dst_this_chunk;
        src_pos += ch.compressed_size;
        chunk_idx += 1;
    }
}
