//! 1:1 port of src/decode/streamlz_decoder.zig.
//!
//! Top-level frame decompressor. Parses the SLZ1 frame header, walks
//! the block list, builds the chunk descriptors on the CPU, and
//! dispatches each block into the GPU pipeline
//! (gpu_driver.fullGpuLaunchImpl).

const std = @import("std");
const constants = @import("../format/streamlz_constants.zig");
const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const gpu_driver = @import("driver.zig");
const vk = @import("vulkan_api.zig");

/// CUDA reference: src/decode/streamlz_decoder.zig:28. Bytes the decoder
/// is allowed to overshoot past the requested output length.
pub const safe_space = constants.safe_space;

/// CUDA reference: src/decode/streamlz_decoder.zig:33. Upper bound on
/// content_size the decoder will honor.
pub const max_content_size: u64 = 4 * 1024 * 1024 * 1024;

/// CUDA reference: src/decode/streamlz_decoder.zig:38-75. Decompress
/// failure modes.
pub const DecompressError = error{
    BadFrame,
    Truncated,
    SizeMismatch,
    InvalidBlockHeader,
    InvalidInternalHeader,
    BadChunkHeader,
    BlockDataTruncated,
    OutputTooSmall,
    ChecksumMismatch,
    ChunkSizeMismatch,
    UnknownDictionary,
    ContentSizeTooLarge,
} || gpu_driver.GpuError || std.mem.Allocator.Error;

/// CUDA reference: src/decode/streamlz_decoder.zig:77-85. Structured
/// decompress result.
pub const DecompressResult = struct {
    written: usize,
    offset: usize = 0,
};

/// CUDA reference: src/decode/streamlz_decoder.zig:97-104. Decompress
/// host->host. Bytes-written variant.
pub fn decompressFramed(
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
) DecompressError!usize {
    const r = try decompressFrameInner(src, dst, dec_ctx, null, null);
    return r.written;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:116-125. Same as
/// decompressFramed plus optional std.Io for telemetry.
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

/// CUDA reference: src/decode/streamlz_decoder.zig:136-182. True-D2D
/// decompress for the v3 C ABI. Single-block, slzCompress-shape frames
/// only. Drives the GPU frame-walk kernel + the device-resident dispatch
/// path so the CPU never touches the compressed bytes.
pub fn decompressFramedFromDevice(
    io: ?std.Io,
    d_frame: u64,
    frame_size: u32,
    d_output: u64,
    dec_ctx: *gpu_driver.DecodeContext,
    decomp_size: u32,
) DecompressError!u32 {
    // CUDA reference: src/decode/streamlz_decoder.zig:149-154 (v4 #16
    // revision). Frame layout: [hdr 14][content_size 8][dict_id 4?]
    // [block_hdr 8][block][end 4]. The header is variable (dict frames
    // carry 4 more bytes), so the block offsets are derived from the
    // parsed header below - the <=64-byte D2H already happens for the
    // codec level.
    const min_overhead: u32 = 14 + 8 + 8 + 4;
    if (frame_size < min_overhead + 1) return error.BadMode;

    // CUDA reference: src/decode/streamlz_decoder.zig:156-161.
    // sc_group=0.25 → effective chunk size of 64 KB, the smallest the
    // GPU encoder produces. The walk kernel writes the real chunk count
    // to d_meta+0; this bound only sizes descriptor + grid space.
    const min_eff_chunk_size: u32 = 0x10000; // 64 KB
    const n_chunks_bound: u32 = (decomp_size + min_eff_chunk_size - 1) / min_eff_chunk_size;
    if (n_chunks_bound == 0 or n_chunks_bound > gpu_driver.WALK_MAX_CHUNKS) return error.BadMode;

    // CUDA reference: src/decode/streamlz_decoder.zig (2026-06-10 fix).
    // D2H the (tiny) frame header to learn the codec level - the L3+
    // entropy stages gate on `req.level >= 3`, and the previous
    // hardcoded `.level = 1` silently misrouted Huffman frames down
    // the raw path on the true-D2D entry (same bug class as iter-4b
    // `bcaa1f1`; found on the CUDA side by tools/bench_d2d.bat).
    var hdr_buf: [64]u8 = @splat(0);
    const hdr_n: u32 = @min(frame_size, 64);
    if (!gpu_driver.copyDeviceToHost(hdr_buf[0..hdr_n], d_frame)) return error.BadMode;
    const parsed_hdr = frame.parseHeader(hdr_buf[0..]) catch return error.BadMode;

    // Block offsets follow the variable-length header (dict frames
    // carry 4 extra ID bytes; header_size accounts for it).
    const block_start: u32 = @intCast(parsed_hdr.header_size + 8);
    if (frame_size < block_start + 4 + 1) return error.BadMode;
    const block_size: u32 = frame_size - block_start - 4;

    // v4 #16 (CUDA-mirror): resolve the frame's dictionary (registered
    // store first, then builtins) and make it device-resident. An
    // unresolvable ID is a real error, not a fall-back shape - the
    // host-bounce path could not resolve it either.
    var d_dict: u64 = 0;
    var dict_len: u32 = 0;
    if (parsed_hdr.dictionary_id) |did| {
        const dictionary = @import("../dict/dictionary.zig");
        const data = dictionary.resolve(dec_ctx.registered_dicts.items, did) orelse
            return error.UnknownDictionary;
        gpu_driver.ensureDictOnDevice(dec_ctx, did, data) catch return error.BadMode;
        d_dict = dec_ctx.d_dict;
        dict_len = dec_ctx.dict_cached_len;
    }

    // v4 #20 (CUDA-mirror): when the frame carries the chunk-size table
    // footer, the walk runs as a parallel table-mode kernel instead of
    // the serial chunk chain. The table position is computable from the
    // header: frame_end - trailers - 3 * n_chunks (n_chunks from
    // content_size and eff_chunk; trailer sizes announced by flag bits).
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

    // A-026 (2026-06-10): batch the WHOLE D2D call onto pipeline_stream.
    // Without this, the walk launch and the compressed-block D2D copy
    // each pay their own vkQueueSubmit + fence round-trip (the WDDM
    // per-submit floor) before fullGpuLaunchImpl's internal sync-mode
    // promotion kicks in - measured as a ~3.4 ms wall-vs-kernel gap on
    // enwik8 L1. Promote here instead, mirroring fullGpuLaunchImpl's
    // iter-4 pattern (it sees work_stream != 0 and skips its own
    // promotion, mutex, and end-of-call drain - so this fn owns all
    // three). The dispatcher mutex serializes against concurrent
    // sync-mode decodes sharing the singleton pipeline_stream cmdbuf.
    const was_sync = dec_ctx.work_stream == 0;
    if (was_sync) {
        try gpu_driver.ensurePipelineStream(dec_ctx);
        gpu_driver.lockDispatcherMutex();
        dec_ctx.work_stream = dec_ctx.pipeline_stream;
    }
    defer if (was_sync) {
        dec_ctx.work_stream = 0;
        gpu_driver.unlockDispatcherMutex();
    };

    const dev = try gpu_driver.gpuWalkFrame(dec_ctx, d_frame, frame_size, table_info);
    // v4 #12 RESOLVED (2026-06-10): the walk now batches into the same
    // cmdbuf as the rest of the pipeline - the former "early submit"
    // workaround is gone. The mystery was never the walk: the chunk-desc
    // D2D copy (uploadInputAndPrefixSum) recorded into the TRANSFER
    // cmdbuf, which streamEndAndWait submits BEFORE the compute leg, so
    // it read the descs before the batched walk wrote them. Stream-path
    // D2D is now compute-cmdbuf-ordered (procD2DOffset); only the
    // explicitly input-side copy keeps the transfer-leg DMA fast path
    // (procs.d2d_input_offset).

    // CUDA reference: src/decode/streamlz_decoder.zig:158-164. Stub
    // host-side slices: the kernel reads from `d_chunk_descs_override`
    // and `d_compressed_src`; the host slices are length-only and never
    // dereferenced.
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
        .d_compressed_src = d_frame,
        .d_compressed_src_offset = block_start,
        .d_chunk_descs_override = dev.d_chunk_descs,
        .d_n_chunks_dev = dev.d_meta + gpu_driver.walk_meta_offsets.n_chunks,
        .level = parsed_hdr.level,
        .d_dict = d_dict,
        .dict_len = dict_len,
    });

    // A-026: drain the batch - ONE submit+wait for walk + copy + prefix
    // + LZ (fullGpuLaunchImpl skipped its sync-mode drain because we
    // promoted first), then materialize the per-kernel timings the
    // bench reads from last_timings.
    if (was_sync) {
        const sync_fn = vk.procs.stream_sync orelse return error.BackendNotAvailable;
        if (sync_fn(dec_ctx.work_stream) != vk.VK_SUCCESS_RC) return error.SyncFailed;
        gpu_driver.finalizeProfiling(&dec_ctx.pending_timings, &dec_ctx.last_timings);

        // v4 #20 (CUDA-mirror): surface walk-kernel rejections. The
        // downstream kernels self-gate on the walk's n_chunks, so a
        // non-zero status (hostile frame, inconsistent chunk table,
        // truncation) would otherwise return SUCCESS with undecoded
        // output. Read after the batch drain - zero extra
        // synchronization. Async-stream callers sync on their own
        // stream; carrying the status to them belongs to the #19
        // verdict surface.
        // VK adaptation: VkDeviceBuffer handles take no +offset
        // arithmetic, so D2H the whole 1536-byte meta block and index
        // the status slot on the host.
        var meta_buf: [gpu_driver.walk_meta_offsets.bytes]u8 = undefined;
        if (!gpu_driver.copyDeviceToHost(meta_buf[0..], dev.d_meta))
            return error.CopyFailed;
        const wstatus = std.mem.readInt(u32, meta_buf[gpu_driver.walk_meta_offsets.status..][0..4], .little);
        if (wstatus != 0) return error.BadMode;
    }
    return decomp_size;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:189-209. Per-call
/// decompress wrapper used by the CLI.
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

    pub fn deinit(self: *DecompressContext) void {
        _ = self;
    }
};

/// CUDA reference: src/decode/streamlz_decoder.zig:215-317. Frame-walk
/// inner driver. Parses the SLZ1 header, walks the block list, and
/// dispatches each compressed block into the GPU.
pub fn decompressFrameInner(
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
) DecompressError!DecompressResult {
    if (src.len == 0) return .{ .written = 0 };

    const hdr = frame.parseHeader(src) catch return error.BadFrame;

    // v4 #16 (CUDA-mirror): resolve the frame's dictionary in the
    // registry (unknown ID rejects immediately - the wire contract).
    // The device upload is LAZY: it happens at the first
    // compressed-block dispatch, cached by ID on the context so a batch
    // of frames sharing one dictionary uploads once. Uncompressed-body
    // dict frames (tiny inputs) therefore decode without touching the
    // GPU at all.
    var dict_bytes: ?[]const u8 = null;
    var dict_id: u32 = 0;
    if (hdr.dictionary_id) |did| {
        const dictionary = @import("../dict/dictionary.zig");
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
        // v4 #16 (CUDA-mirror): first compressed block - make the
        // dictionary device-resident now (cached; see the resolution
        // above).
        var d_dict: u64 = 0;
        var dict_len: u32 = 0;
        if (dict_bytes) |data| {
            gpu_driver.ensureDictOnDevice(dec_ctx, dict_id, data) catch return error.BadMode;
            d_dict = dec_ctx.d_dict;
            dict_len = dec_ctx.dict_cached_len;
        }
        // VK adaptation: only thread level>=3 to trigger fullGpuLaunchImpl's
        // L2 gate (the Huffman pre-decode chain + general LZ kernel). L2
        // frames have no Huffman streams, and the upstream scan/compact/
        // merge kernel host bindings still use offset-on-handle arithmetic
        // that the VK launcher doesn't support — forcing level>=2 there
        // would regress the currently-passing L2 path (which decodes via
        // the L1 raw kernel). Once the launcher gains per-binding offset
        // support, this can become `hdr.level` unconditionally.
        const dispatch_level: u8 = if (hdr.level >= 3) hdr.level else 1;
        try dispatchCompressedBlock(
            block_src,
            dst,
            dst_off,
            block_hdr.decompressed_size,
            hdr.sc_group_size,
            dispatch_level,
            dec_ctx,
            d_output_target,
            io,
            d_dict,
            dict_len,
        );
        // Restore the 8-byte SC tail prefix bytes that the encoder
        // appended at the end of the block. The decoder kernel writes
        // garbage into the first 8 bytes of every chunk past the first.
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

    // v4 #20 (CUDA-mirror): the chunk-size table footer sits between
    // the end mark and the trailers - step the forward cursor over it
    // so the trailer reads below land on the right bytes. (The host
    // decode path itself never reads the table; it walks the chain.)
    if (hdr.chunk_size_table) {
        const eff_t = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
        if (eff_t > 0) pos += ((dst_off + eff_t - 1) / eff_t) * 3;
    }

    // v4 #13 (2026-06-10, CUDA-mirror): XXH32 content checksum
    // verification. The 4-byte LE hash follows the end-mark when the
    // frame's ContentChecksum flag is set. Computed over the
    // decompressed output, not the compressed frame. See
    // src/decode/streamlz_decoder.zig for the CUDA reference.
    if (hdr.content_checksum) {
        if (pos + 4 > src.len) return error.Truncated;
        const expected = std.mem.readInt(u32, src[pos..][0..4], .little);
        const xxh = @import("../format/xxhash32.zig");
        const actual = xxh.xxhash32(dst[0..dst_off]);
        if (actual != expected) return error.ChecksumMismatch;
    }
    // v4 #19 (CUDA-mirror): chunk-Merkle trailer, after the content
    // trailer when both flags are set. Parallel per-chunk hash; see
    // src/decode/streamlz_decoder.zig for the CUDA reference.
    if (hdr.chunk_merkle) {
        const merkle_pos = pos + @as(usize, if (hdr.content_checksum) 4 else 0);
        if (merkle_pos + 4 > src.len) return error.Truncated;
        const expected = std.mem.readInt(u32, src[merkle_pos..][0..4], .little);
        const xxh = @import("../format/xxhash32.zig");
        const eff = @min(frame.scGroupSizeToBytes(hdr.sc_group_size), constants.chunk_size);
        const actual = xxh.chunkMerkleRoot(std.heap.page_allocator, dst[0..dst_off], eff) catch
            return error.ChecksumMismatch;
        if (actual != expected) return error.ChecksumMismatch;
    }
    return .{ .written = dst_off };
}

/// CUDA reference: src/decode/streamlz_decoder.zig:323-380. Per-block
/// GPU dispatch. Parses the internal block header on the CPU, builds the
/// chunk descriptors on the CPU, and hands the descriptor slice to the
/// GPU pipeline.
///
/// VK adaptation: `level` is threaded from the parsed FrameHeader so
/// fullGpuLaunchImpl's L2 gate (req.level >= 2) fires for L3+ frames
/// that contain Huffman-coded streams. Without it, n_huff stays 0 and
/// runLzPipeline routes to the L1 raw kernel, which produces wrong
/// output on huff-coded inputs (the L3+ wrong-output bug fixed here).
pub fn dispatchCompressedBlock(
    block_src: []const u8,
    dst: []u8,
    dst_start_off: usize,
    decompressed_size: usize,
    sc_group_size: f32,
    level: u8,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
    // v4 #16 (CUDA-mirror): preset dictionary, already device-resident
    // on the context's d_dict cache. 0/0 on dictionary-less frames.
    d_dict: u64,
    dict_len: u32,
) DecompressError!void {
    const eff_chunk_size = @min(frame.scGroupSizeToBytes(sc_group_size), constants.chunk_size);
    const num_chunks = (decompressed_size + eff_chunk_size - 1) / eff_chunk_size;
    if (num_chunks == 0) return;
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
        // VK adaptation: thread the frame's level so fullGpuLaunchImpl's
        // L2 gate fires for L3+ frames with Huffman-coded streams. Without
        // this the default level=1 keeps n_huff=0 and the dispatcher
        // routes to the L1 raw kernel — produces wrong output past byte 8.
        .level = level,
        .d_dict = d_dict,
        .dict_len = dict_len,
    });
}

/// CUDA reference: src/decode/streamlz_decoder.zig:386-481. CPU walk over
/// the block payload to fill `chunk_descs` (one entry per chunk). Parses
/// the internal block header + each 4-byte (or 7-byte w/ checksum)
/// chunk header from CPU memory and produces a host slice of ChunkDesc.
/// Sets flags=2 for memset chunks, flags=1 for whole-match shortcuts,
/// and flags=0 for LZ-compressed chunks.
pub fn buildChunkDescriptors(
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
