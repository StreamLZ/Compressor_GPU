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
const dictionary = @import("../dict/dictionary.zig");
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
    /// The frame carries a `dictionary_id` that does not resolve in
    /// the dictionary registry (`src/dict/dictionary.zig`). Only
    /// registry built-ins resolve today; custom-dictionary
    /// registration lands with the C ABI surface.
    UnknownDictionary,
    /// `content_size` exceeds `max_content_size` (4 GiB). Either the
    /// frame is hostile or it was produced by an encoder that exceeds
    /// this decoder's cap.
    ContentSizeTooLarge,
} || gpu_driver.GpuError || std.mem.Allocator.Error;

pub const DecompressResult = struct {
    /// Number of bytes written into `dst`. Always equal to the frame's
    /// `content_size` on success.
    written: usize,
    /// Always 0 in the GPU-only codec. Kept for shape compatibility
    /// with the legacy CPU decoder, which used to return a non-zero
    /// dictionary-prefix offset; no in-tree caller reads it.
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
/// `DecompressResult`; `decompressFramed` is a `.written`-only
/// convenience.
///
/// The `allocator` parameter is currently unused (the dispatch takes
/// the allocator from the encode/decode context). Plumbed in for
/// shape compatibility with the previous CPU codec entry point;
/// callers can pass `undefined`.
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
/// content_size present, sc_group ≤ 1.0. Dictionary frames are
/// supported (v4 #16): the ID is read from the ≤64-byte header D2H and
/// resolved against the context's registered dictionaries + builtins.
/// Other shapes return `error.BadMode` and the C ABI falls back to the
/// host-bounce entry points.
pub fn decompressFramedFromDevice(
    io: ?std.Io,
    d_frame: u64,
    frame_size: u32,
    d_output: u64,
    dec_ctx: *gpu_driver.DecodeContext,
    decomp_size: u32,
) DecompressError!u32 {
    // Frame layout: [hdr 14][content_size 8][dict_id 4?][block_hdr 8]
    // [block][end 4]. The header is variable (dict frames carry 4 more
    // bytes), so the block offsets are derived from the parsed header
    // below - the ≤64-byte D2H already happens for the codec level.
    const min_overhead: u32 = 14 + 8 + 8 + 4;
    if (frame_size < min_overhead + 1) return error.BadMode;

    // sc_group=0.25 → effective chunk size of 64 KB, the smallest the
    // GPU encoder produces. The walk kernel writes the real chunk count
    // to d_meta+0; this bound only sizes the descriptor + grid space.
    const min_eff_chunk_size: u32 = 0x10000; // 64 KB
    const n_chunks_bound: u32 = (decomp_size + min_eff_chunk_size - 1) / min_eff_chunk_size;
    if (n_chunks_bound == 0 or n_chunks_bound > gpu_driver.WALK_MAX_CHUNKS) return error.BadMode;

    // D2H the (tiny) frame header to learn the codec level. The L3+
    // entropy stages in fullGpuLaunchImpl gate on `req.level >= 3`;
    // before 2026-06-10 this request left `.level` at its default of 1,
    // so Huffman frames silently misrouted down the raw path and the
    // true-D2D entry failed at L3+ (found by tools\bench_d2d.bat).
    // Same bug class as srcVK iter-4b (`bcaa1f1`). A ≤64-byte D2H is
    // ~tens of µs against a ≥multi-ms decode.
    var hdr_buf: [64]u8 = @splat(0);
    const hdr_n: u32 = @min(frame_size, 64);
    if (!gpu_driver.copyDeviceToHost(hdr_buf[0..hdr_n], d_frame)) return error.BadMode;
    const parsed_hdr = frame.parseHeader(hdr_buf[0..]) catch return error.BadMode;

    // Block offsets follow the variable-length header (dict frames
    // carry 4 extra ID bytes; header_size accounts for it).
    const block_start: u32 = @intCast(parsed_hdr.header_size + 8);
    if (frame_size < block_start + 4 + 1) return error.BadMode;
    const block_size: u32 = frame_size - block_start - 4;

    // v4 #16: resolve the frame's dictionary (registered store first,
    // then builtins) and make it device-resident. An unresolvable ID
    // is a real error, not a fall-back shape - the host-bounce path
    // could not resolve it either.
    var d_dict: u64 = 0;
    var dict_len: u32 = 0;
    if (parsed_hdr.dictionary_id) |did| {
        const data = dictionary.resolve(dec_ctx.registered_dicts.items, did) orelse
            return error.UnknownDictionary;
        try gpu_driver.ensureDictOnDevice(dec_ctx, did, data);
        d_dict = dec_ctx.d_dict;
        dict_len = dec_ctx.dict_cached_len;
    }

    // v4 #20: when the frame carries the chunk-size table footer, the
    // walk runs as a parallel table-mode kernel instead of the serial
    // chunk chain. The table position is computable from the header:
    // frame_end - trailers - 3 * n_chunks (n_chunks from content_size
    // and eff_chunk; trailer sizes announced by flag bits).
    var table_info: ?gpu_driver.ChunkTableInfo = null;
    if (parsed_hdr.chunk_size_table) {
        const eff: u32 = @intCast(@min(frame.scGroupSizeToBytes(parsed_hdr.sc_group_size), constants.chunk_size));
        if (eff == 0) return error.BadMode;
        const n_tab: u32 = (decomp_size + eff - 1) / eff;
        var trailer_bytes: u32 = 0;
        if (parsed_hdr.chunk_merkle) trailer_bytes += 4;
        if (parsed_hdr.content_checksum) trailer_bytes += 4;
        const table_bytes: u32 = n_tab * 3;
        if (frame_size < trailer_bytes + table_bytes) return error.BadMode;
        table_info = .{
            .table_off = frame_size - trailer_bytes - table_bytes,
            .n_chunks = n_tab,
        };
    }

    const dev = try gpu_driver.gpuWalkFrame(dec_ctx, d_frame, frame_size, table_info);

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
        .level = parsed_hdr.level,
        .io = io,
        .d_output_target = d_output,
        .d_compressed_src = d_frame + block_start,
        .d_chunk_descs_override = dev.d_chunk_descs,
        .d_n_chunks_dev = dev.d_meta + gpu_driver.walk_meta_offsets.n_chunks,
        .d_dict = d_dict,
        .dict_len = dict_len,
    });

    // v4 #20: surface walk-kernel rejections. The downstream kernels
    // self-gate on the walk's n_chunks, so a non-zero status (hostile
    // frame, inconsistent chunk table, truncation) would otherwise
    // return SUCCESS with undecoded output. Read after the dispatch's
    // own sync - zero extra synchronization in sync mode. Async-stream
    // callers sync on their own stream; carrying the status to them
    // belongs to the #19 verdict surface.
    if (dec_ctx.work_stream == 0) {
        var wstatus: u32 = 0;
        if (!gpu_driver.copyDeviceToHost(std.mem.asBytes(&wstatus), dev.d_meta + gpu_driver.walk_meta_offsets.status))
            return error.CopyFailed;
        if (wstatus != 0) return error.BadMode;
    }
    return decomp_size;
}

/// Per-call decompress wrapper used by the CLI. Bundles the allocator
/// and an optional `std.Io` for the SLZ_E2E_TIMER / SLZ_SPLIT_TIMER
/// telemetry plumbing, and delegates to the module-level
/// `g_default` GPU context. Library callers that hold their own
/// `gpu_driver.DecodeContext` should call `decompressFramed*` directly.
pub const DecompressContext = struct {
    allocator: std.mem.Allocator,
    io: ?std.Io,
    dec_ctx: *gpu_driver.DecodeContext = &gpu_driver.g_default,

    pub fn initWithIo(allocator: std.mem.Allocator, io: std.Io) DecompressContext {
        return .{ .allocator = allocator, .io = io };
    }

    pub fn decompress(self: *DecompressContext, src: []const u8, dst: []u8) DecompressError!DecompressResult {
        return decompressFrameInner(src, dst, self.dec_ctx, null, self.io);
    }

    /// No-op today: the wrapper owns no allocations of its own (the
    /// `g_default` GPU context outlives every call). Defined so call
    /// sites can use the canonical `initWithIo` / `defer deinit()`
    /// pattern without remembering the wrapper is stateless.
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

    // v4 #16: resolve the frame's dictionary in the registry (unknown
    // ID rejects immediately - the wire contract). The device upload
    // is LAZY: it happens at the first compressed-block dispatch,
    // cached by ID on the context so a batch of frames sharing one
    // dictionary uploads once. Uncompressed-body dict frames (tiny
    // inputs) therefore decode without touching the GPU at all - the
    // eager-upload variant failed them with BackendNotAvailable
    // before any dispatch had initialized the driver.
    var dict_bytes: ?[]const u8 = null;
    var dict_id: u32 = 0;
    if (hdr.dictionary_id) |did| {
        dict_bytes = dictionary.resolve(dec_ctx.registered_dicts.items, did) orelse
            return error.UnknownDictionary;
        dict_id = did;
    }

    if (hdr.content_size) |cs| {
        if (cs > max_content_size) return error.ContentSizeTooLarge;
        if (d_output_target == null and dst.len < @as(usize, @intCast(cs + safe_space))) return error.OutputTooSmall;
    }

    var pos: usize = hdr.header_size;
    var dst_off: usize = 0;
    // v4 #19 device-only verification: per-chunk hashing, prefix
    // application, root roll-up and the compare all run ON DEVICE;
    // the host contributes the expected root as a launch scalar
    // (parsed from the frame tail it already holds) and reads one
    // 4-byte verdict alongside the final sync. Needs the content
    // size (our encoder always writes it) to pre-size the device
    // hash buffer; falls back to the parallel host hash otherwise.
    // verdict: 0 = device path did not run, 1 = match, 2 = mismatch.
    var merkle_total: usize = 0;
    var merkle_filled: usize = 0;
    var merkle_verdict: u32 = 0;
    var merkle_expected: u32 = 0;
    if (hdr.chunk_merkle and gpu_driver.hasChunkHashKernel()) {
        if (hdr.content_size) |csz| {
            const eff0 = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
            merkle_total = (@as(usize, @intCast(csz)) + eff0 - 1) / eff0;
            if (src.len >= 4)
                merkle_expected = std.mem.readInt(u32, src[src.len - 4 ..][0..4], .little);
        }
    }

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
        // v4 #16: first compressed block - make the dictionary
        // device-resident now (cached; see the resolution above).
        var d_dict: u64 = 0;
        var dict_len: u32 = 0;
        if (dict_bytes) |data| {
            try gpu_driver.ensureDictOnDevice(dec_ctx, dict_id, data);
            d_dict = dec_ctx.d_dict;
            dict_len = dec_ctx.dict_cached_len;
        }
        // v4 #19: per-block device-hash geometry.
        var blk_merkle_n: u32 = 0;
        var blk_merkle_base: u32 = 0;
        var blk_run_verdict = false;
        if (merkle_total > 0) {
            const eff_cs0 = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
            const n_blk = (block_hdr.decompressed_size + eff_cs0 - 1) / eff_cs0;
            blk_merkle_n = @intCast(n_blk);
            blk_merkle_base = @intCast(merkle_filled);
            merkle_filled += n_blk;
            blk_run_verdict = merkle_filled == merkle_total;
        }
        try dispatchCompressedBlock(
            block_src,
            dst,
            dst_off,
            block_hdr.decompressed_size,
            hdr.sc_group_size,
            hdr.level,
            dec_ctx,
            d_output_target,
            io,
            blk_merkle_n,
            blk_merkle_base,
            @intCast(merkle_total),
            blk_run_verdict,
            merkle_expected,
            &merkle_verdict,
            d_dict,
            dict_len,
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

    // v4 #20: the chunk-size table footer sits between the end mark
    // and the trailers - step the forward cursor over it so the
    // trailer reads below land on the right bytes. (The host decode
    // path itself never reads the table; it walks the chunk chain.)
    if (hdr.chunk_size_table) {
        const eff_t = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
        if (eff_t > 0) pos += ((dst_off + eff_t - 1) / eff_t) * 3;
    }

    // v4 #13 (2026-06-10): XXH32 content checksum verification. The 4-byte
    // LE hash follows the end-mark when the frame's ContentChecksum flag is
    // set. Computed over the decompressed output, not the compressed frame.
    // For D2D (d_output_target != null) the output lives on the device and
    // there's no host copy to hash — skip verification (the D2D caller can
    // opt into their own check if needed).
    if (hdr.content_checksum and d_output_target == null) {
        if (pos + 4 > src.len) return error.Truncated;
        const expected = std.mem.readInt(u32, src[pos..][0..4], .little);
        const xxh = @import("../format/xxhash32.zig");
        const actual = xxh.xxhash32(dst[0..dst_off]);
        if (actual != expected) return error.ChecksumMismatch;
    }
    // v4 #19: chunk-Merkle trailer - follows the content trailer when
    // both flags are set. Verified with the parallel per-chunk hash
    // (~1 ms/100 MB across threads), so default-ON integrity does not
    // move the e2e headline. D2D outputs live on device; skipped there
    // like the content checksum (the GPU-side verify is the v2 plan).
    if (hdr.chunk_merkle) {
        const merkle_pos = pos + @as(usize, if (hdr.content_checksum) 4 else 0);
        if (merkle_pos + 4 > src.len) return error.Truncated;
        const expected = std.mem.readInt(u32, src[merkle_pos..][0..4], .little);
        const xxh = @import("../format/xxhash32.zig");
        if (merkle_verdict == 1) {
            // Device verdict: match. Pin the tail-derived scalar the
            // device compared against to the canonical trailer slot.
            if (merkle_expected != expected) return error.ChecksumMismatch;
        } else if (merkle_verdict == 2) {
            return error.ChecksumMismatch;
        } else if (d_output_target == null) {
            // Device path did not run (kernels missing / no content
            // size): parallel host hash fallback over the host copy.
            const eff = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
            const actual = xxh.chunkMerkleRoot(std.heap.page_allocator, dst[0..dst_off], eff) catch
                return error.ChecksumMismatch;
            if (actual != expected) return error.ChecksumMismatch;
        }
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
    level: u8,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
    // v4 #19 device-only verify geometry (see DecodeRequest docs).
    merkle_n: u32,
    merkle_base: u32,
    merkle_total: u32,
    merkle_run_verdict: bool,
    merkle_expected: u32,
    merkle_verdict_out: *u32,
    // v4 #16: device-resident preset dictionary (0/0 = none).
    d_dict: u64,
    dict_len: u32,
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
        // v4 #19: upload the FULL block (incl. the SC tail prefix
        // table) - descriptors never reference the tail, and the
        // prefix-apply kernel reads it in place from d_comp.
        .compressed_block = block_src,
        .dst_full = dst.ptr,
        .dst_start_off = dst_start_off,
        .decompressed_size = decompressed_size,
        .chunks_per_group = chunks_per_group,
        .sub_chunk_cap = @intCast(constants.sub_chunk_size),
        .io = io,
        .d_output_target = d_output_target,
        .d_compressed_src = null,
        .level = level,
        .merkle_n_chunks = merkle_n,
        .merkle_chunk_base = merkle_base,
        .merkle_total_chunks = merkle_total,
        .merkle_eff_chunk = @intCast(eff_chunk_size),
        .merkle_prefix_off = if (prefix_sz != 0 and merkle_n > 0) @as(u64, @intCast(block_payload.len)) else 0,
        .merkle_run_verdict = merkle_run_verdict,
        .merkle_expected_root = merkle_expected,
        .merkle_verdict_out = if (merkle_run_verdict) merkle_verdict_out else null,
        .d_dict = d_dict,
        .dict_len = dict_len,
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
