//! 1:1 port of src/encode/fast_framed.zig.
//!
//! L1 encode orchestrator. Entry point compressFramedOne() called by
//! srcVK/encode/streamlz_encoder.zig::compressFramedWithIo. Owns the
//! `opts.level >= 3` gate that funnels L3-L5 frames into the Huffman
//! pipeline; on L1/L2 the gate skips Huffman and emits raw streams via
//! the LZ-encode + assemble-measure + assemble-write + frame-assemble
//! kernel chain.

const std = @import("std");

const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const lz_constants = @import("../format/streamlz_constants.zig");

const encoder = @import("streamlz_encoder.zig");
const Options = encoder.Options;
const CompressError = encoder.CompressError;

const gpu_enc = @import("driver.zig");
const vk_api = @import("../decode/vulkan_api.zig");
const decode_module_loader = @import("../decode/module_loader.zig");

/// CUDA reference: src/encode/fast_framed.zig:27. Inputs at or below this
/// size are too small for the LZ kernel's per-warp setup cost to ever
/// produce a smaller output; emit them as a whole-frame uncompressed
/// body.
const min_source_length: usize = 128;

/// CUDA reference: src/encode/fast_framed.zig:40. Number of decoder warps
/// each SM can host simultaneously.
const decoder_warps_per_sm: usize = 48;

/// CUDA reference: src/encode/fast_framed.zig:43. Bytes each decoder warp
/// consumes per sub-chunk at sc_group=0.5.
const sc05_bytes_per_warp: usize = 128 * 1024;

/// CUDA reference: src/encode/fast_framed.zig:49. Per-sub-chunk staging
/// headroom for the raw LZ payload.
const gpu_block_capacity_multiplier: usize = 3;

/// CUDA reference: src/encode/fast_framed.zig:55. Fallback SM count.
const sm_count_fallback: usize = 34;

/// CUDA reference: src/encode/fast_framed.zig:61. Maximum chunk count
/// the frame-assemble kernel accepts in a single grid.
const assembly_chunk_cap: usize = 16384;

/// CUDA reference: src/encode/fast_framed.zig:66-74. Pick the
/// sc_group_size to advertise in the frame header. Honors a caller
/// override; otherwise picks 0.25 below the GPU saturation threshold
/// and 0.5 at or above it.
fn resolveScGroupSize(src_len: usize, override: ?f32) f32 {
    if (override) |ov| return ov;
    // VK adaptation: vk_api.sm_count is 0 until a future
    // VkPhysicalDeviceProperties query populates it; the fallback path
    // preserves CUDA's behavior when the driver did not report SM count.
    const sm_count: usize = if (vk_api.sm_count > 0)
        @intCast(vk_api.sm_count)
    else
        sm_count_fallback;
    const saturation_bytes = sm_count * decoder_warps_per_sm * sc05_bytes_per_warp;
    return if (src_len >= saturation_bytes) 0.5 else 0.25;
}

/// CUDA reference: src/encode/fast_framed.zig:80-89. Map the unified
/// L1-L5 user level to the codec level written in the frame header.
fn codecLevelFor(user_level: u8) u8 {
    return switch (user_level) {
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 5,
        5 => 6,
        else => unreachable,
    };
}

/// CUDA reference: src/encode/fast_framed.zig:91-139. Compress one frame
/// from src into dst. Returns the number of bytes written.
pub fn compressFramedOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
) CompressError!usize {
    if (!gpu_enc.isAvailable()) return error.DestinationTooSmall;

    // VK adaptation: ptest_vk's 16-worker test runner dispatches multiple
    // concurrent encodes through enc_driver.g_default. The frame orchestration
    // below (gpuCompressImpl + gpuAssembleFrameImpl + gpuFrameAssembleImpl)
    // grows EncodeContext's persistent device buffers via
    // encode_context.ensureBuf (destroy+create), and stages intra-frame
    // state on enc_ctx fields (d_input_override, d_output_override,
    // assembled_offsets/sizes, output_written_to_device). A sibling worker
    // running mid-frame clobbers the device handles and surfaces as
    // DestinationTooSmall or all-zero decode output. Serialize the whole
    // frame compress through the encode dispatcher mutex in sync mode
    // (work_stream == 0). Async-mode callers (per-worker EncodeContext +
    // stream) skip the lock and serialize per-stream upstream.
    const enc_is_sync = enc_ctx.work_stream == 0;
    if (enc_is_sync) decode_module_loader.lockEncodeDispatcherMutex();
    defer if (enc_is_sync) decode_module_loader.unlockEncodeDispatcherMutex();

    var pos: usize = 0;
    const sc_grp = resolveScGroupSize(src.len, opts.sc_group_size_override);

    const hdr_len = frame.writeHeader(dst, .{
        .codec = .fast,
        .level = codecLevelFor(opts.level),
        .block_size = opts.block_size,
        .sc_group_size = sc_grp,
        .content_size = if (opts.include_content_size) @as(u64, @intCast(src.len)) else null,
        .dictionary_id = null,
        .content_checksum = false,
    }) catch return error.DestinationTooSmall;
    pos += hdr_len;

    if (src.len == 0) {
        if (pos + 4 > dst.len) return error.DestinationTooSmall;
        frame.writeEndMark(dst[pos..]);
        return pos + 4;
    }

    const frame_block_hdr_pos: usize = pos;
    const frame_block_start: usize = pos + 8;

    const can_compress = src.len > min_source_length;
    if (!can_compress) {
        return writeUncompressedFrame(dst, frame_block_hdr_pos, frame_block_start, src);
    }

    return gpuEncodeAndAssemble(
        allocator,
        io,
        src,
        dst,
        opts,
        enc_ctx,
        sc_grp,
        frame_block_hdr_pos,
    );
}

/// CUDA reference: src/encode/fast_framed.zig:141-157. Emit a whole-frame
/// uncompressed body.
fn writeUncompressedFrame(
    dst: []u8,
    frame_block_hdr_pos: usize,
    frame_block_start: usize,
    src: []const u8,
) CompressError!usize {
    if (frame_block_start + src.len + 4 > dst.len) return error.DestinationTooSmall;
    frame.writeBlockHeader(dst[frame_block_hdr_pos..], .{
        .compressed_size = @intCast(src.len),
        .decompressed_size = @intCast(src.len),
        .uncompressed = true,
        .parallel_decode_metadata = false,
    });
    @memcpy(dst[frame_block_start..][0..src.len], src);
    frame.writeEndMark(dst[frame_block_start + src.len ..]);
    return frame_block_start + src.len + 4;
}

/// CUDA reference: src/encode/fast_framed.zig:163-276. Runs the GPU
/// encode pipeline and the device-resident assembly chain.
fn gpuEncodeAndAssemble(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
    sc_grp: f32,
    frame_block_hdr_pos: usize,
) CompressError!usize {
    const eff_chunk = @min(frame.scGroupSizeToBytes(sc_grp), lz_constants.chunk_size);
    const sub_chunk_cap: usize = lz_constants.sub_chunk_size;
    const gpu_block: usize = @min(eff_chunk, sub_chunk_cap);
    const n_chunks = (src.len + eff_chunk - 1) / eff_chunk;

    var n_gpu_blocks: usize = 0;
    for (0..n_chunks) |ci| {
        const chunk_size = @min(eff_chunk, src.len - ci * eff_chunk);
        n_gpu_blocks += (chunk_size + gpu_block - 1) / gpu_block;
    }
    if (n_chunks > assembly_chunk_cap) {
        return writeUncompressedFrame(dst, frame_block_hdr_pos, frame_block_hdr_pos + 8, src);
    }

    const owns_wrap_input = enc_ctx.d_input_override == 0;
    const owns_wrap_output = enc_ctx.d_output_override == 0;
    if (owns_wrap_input) {
        if (!gpu_enc.ensureBuf(&enc_ctx.d_host_wrap_input, &enc_ctx.d_host_wrap_input_size, src.len))
            return error.DestinationTooSmall;
        if (!gpu_enc.copyHostToDevice(enc_ctx.d_host_wrap_input, src))
            return error.DestinationTooSmall;
        enc_ctx.d_input_override = enc_ctx.d_host_wrap_input;
    }
    if (owns_wrap_output) {
        const bound = encoder.compressBound(src.len);
        if (!gpu_enc.ensureBuf(&enc_ctx.d_host_wrap_output, &enc_ctx.d_host_wrap_output_size, bound))
            return error.DestinationTooSmall;
        enc_ctx.d_output_override = enc_ctx.d_host_wrap_output;
        enc_ctx.output_written_to_device = false;
    }
    defer {
        if (owns_wrap_input) enc_ctx.d_input_override = 0;
        if (owns_wrap_output) {
            enc_ctx.d_output_override = 0;
            enc_ctx.output_written_to_device = false;
        }
    }

    const descs = try allocator.alloc(gpu_enc.CompressChunkDesc, n_gpu_blocks);
    defer allocator.free(descs);
    const comp_sizes = try allocator.alloc(u32, n_gpu_blocks);
    defer allocator.free(comp_sizes);
    const per_block_cap = gpu_block * gpu_block_capacity_multiplier;
    // VK adaptation (encode D2H gather): use the EncodeContext's
    // persistent page-aligned host buffer so the iter-8 LRU import
    // cache in procD2HOffsetGather hits on every call after the first
    // (same host_ptr across encodes). Pre-fix this was a per-call
    // allocator.alloc which produced a fresh host_ptr each encode and
    // paid a fresh ~1 ms VK_EXT_external_memory_host import on top of
    // the per-region submit-floor cost. Falls back to allocator.alloc
    // when ensureGpuOutBuf returns null (backend unavailable /
    // malloc_host null) — preserves the pre-fix behavior for
    // init-failed contexts.
    var gpu_out_fallback: ?[]u8 = null;
    defer if (gpu_out_fallback) |fb| allocator.free(fb);
    const gpu_out: []u8 = if (gpu_enc.ensureGpuOutBuf(enc_ctx, n_gpu_blocks * per_block_cap)) |buf|
        buf
    else blk: {
        const fb = try allocator.alloc(u8, n_gpu_blocks * per_block_cap);
        gpu_out_fallback = fb;
        break :blk fb;
    };

    {
        var bi: usize = 0;
        for (0..n_chunks) |ci| {
            const chunk_start = ci * eff_chunk;
            const chunk_size = @min(eff_chunk, src.len - chunk_start);
            const n_subs = (chunk_size + gpu_block - 1) / gpu_block;
            for (0..n_subs) |si| {
                const sub_start = chunk_start + si * gpu_block;
                const sub_size = @min(gpu_block, src.len - sub_start);
                descs[bi] = .{
                    .src_offset = @intCast(sub_start),
                    .src_size = @intCast(sub_size),
                    .dst_offset = @intCast(bi * per_block_cap),
                    .dst_capacity = @intCast(per_block_cap),
                    .is_first = if (bi == 0) @as(u32, 1) else 0,
                };
                bi += 1;
            }
        }
    }

    if (!gpu_enc.gpuCompressImpl(enc_ctx, src, gpu_out, descs, comp_sizes, io, opts.level))
        return error.DestinationTooSmall;

    const did_huff_lit = opts.level >= 3 and gpu_enc.gpuEncodeLiteralsHuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_lit) freeHuffLit(allocator, enc_ctx);
    const did_huff_tok = opts.level >= 3 and gpu_enc.gpuEncodeTokensHuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_tok) freeHuffTok(allocator, enc_ctx);
    const did_huff_off16 = opts.level >= 3 and gpu_enc.gpuEncodeOff16HuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_off16) freeHuffOff16(allocator, enc_ctx);

    if (!gpu_enc.gpuAssembleFrameImpl(enc_ctx, allocator, descs, comp_sizes))
        return error.DestinationTooSmall;
    defer freeAssembled(allocator, enc_ctx);

    const frame_size = try assembleFrame(
        enc_ctx,
        allocator,
        src,
        dst,
        eff_chunk,
        gpu_block,
        n_chunks,
        comp_sizes,
        frame_block_hdr_pos,
    );

    if (owns_wrap_output) {
        if (dst.len < frame_size) return error.DestinationTooSmall;
        if (!gpu_enc.copyDeviceToHost(dst[0..frame_size], enc_ctx.d_host_wrap_output))
            return error.DestinationTooSmall;
    }
    return frame_size;
}

/// CUDA reference: src/encode/fast_framed.zig:282-378. Launch
/// `slzFrameAssembleKernel`. Per-chunk uncompressed handling is signalled
/// via UNCOMPRESSED_CHUNK_MARKER in the chunk's offset slot.
fn assembleFrame(
    enc_ctx: *gpu_enc.EncodeContext,
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    eff_chunk: usize,
    gpu_block: usize,
    n_chunks: usize,
    comp_sizes: []const u32,
    frame_block_hdr_pos: usize,
) CompressError!usize {
    const sizes = enc_ctx.assembled_sizes orelse return error.DestinationTooSmall;
    const offsets = enc_ctx.assembled_offsets orelse return error.DestinationTooSmall;
    if (sizes.len < n_chunks or offsets.len < n_chunks) return error.DestinationTooSmall;

    std.debug.assert(comp_sizes.len == n_chunks);

    const internal_block_flags: u8 = 0x05 | 0x10 | 0x40; // magic | self_contained | keyframe
    const internal_block_codec: u8 = @intFromEnum(block_header.CodecType.fast);

    const per_chunk_asm_size_buf = try allocator.alloc(u32, n_chunks);
    defer allocator.free(per_chunk_asm_size_buf);
    const per_chunk_asm_off_buf = try allocator.alloc(u32, n_chunks);
    defer allocator.free(per_chunk_asm_off_buf);

    var total_chunk_bytes: usize = 0;
    var sub_idx: usize = 0;
    for (0..n_chunks) |ci| {
        const chunk_src_size: u32 = @intCast(@min(eff_chunk, src.len - ci * eff_chunk));
        var any_failed = false;
        var remaining: u32 = chunk_src_size;
        while (remaining > 0) {
            if (sub_idx >= comp_sizes.len) break;
            const sub_src: u32 = @min(@as(u32, @intCast(gpu_block)), remaining);
            if (comp_sizes[sub_idx] >= sub_src) any_failed = true;
            remaining -= sub_src;
            sub_idx += 1;
        }
        if (any_failed) {
            per_chunk_asm_off_buf[ci] = gpu_enc.UNCOMPRESSED_CHUNK_MARKER;
            per_chunk_asm_size_buf[ci] = chunk_src_size;
            total_chunk_bytes += gpu_enc.UNCOMPRESSED_CHUNK_HDR_BYTES + chunk_src_size;
        } else {
            per_chunk_asm_off_buf[ci] = offsets[ci];
            per_chunk_asm_size_buf[ci] = sizes[ci];
            total_chunk_bytes += gpu_enc.CHUNK_INTERNAL_HDR_BYTES + sizes[ci];
        }
    }

    const sc_tail_bytes: usize = if (n_chunks > 1) (n_chunks - 1) * gpu_enc.SC_TAIL_PER_CHUNK_BYTES else 0;
    const block_payload_size: usize = total_chunk_bytes + sc_tail_bytes;

    var block_hdr_buf: [8]u8 = undefined;
    frame.writeBlockHeader(&block_hdr_buf, .{
        .compressed_size = @intCast(block_payload_size),
        .decompressed_size = @intCast(src.len),
        .uncompressed = false,
        .parallel_decode_metadata = false,
    });

    const prefix_size: usize = frame_block_hdr_pos + 8;
    var prefix_buf: [128]u8 = undefined;
    if (prefix_size > prefix_buf.len) return error.DestinationTooSmall;
    @memcpy(prefix_buf[0..frame_block_hdr_pos], dst[0..frame_block_hdr_pos]);
    @memcpy(prefix_buf[frame_block_hdr_pos..][0..8], &block_hdr_buf);

    const frame_size = gpu_enc.gpuFrameAssembleImpl(
        enc_ctx,
        allocator,
        .{
            .n_chunks = @intCast(n_chunks),
            .eff_chunk_size = @intCast(eff_chunk),
            .src_len = @intCast(src.len),
            .per_chunk_asm_off = per_chunk_asm_off_buf,
            .per_chunk_asm_size = per_chunk_asm_size_buf,
        },
        .{
            .prefix_bytes = prefix_buf[0..prefix_size],
            .internal_hdr0 = internal_block_flags,
            .internal_hdr1 = internal_block_codec,
        },
        enc_ctx.d_input_override,
        enc_ctx.d_output_override,
    ) orelse return error.DestinationTooSmall;
    enc_ctx.output_written_to_device = true;
    return @intCast(frame_size);
}

fn freeHuffLit(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.huff_lit_sizes) |s| a.free(s);
    if (c.huff_lit_offsets) |o| a.free(o);
    c.huff_lit_sizes = null;
    c.huff_lit_offsets = null;
}

fn freeHuffTok(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.huff_tok_sizes) |s| a.free(s);
    if (c.huff_tok_offsets) |o| a.free(o);
    c.huff_tok_sizes = null;
    c.huff_tok_offsets = null;
}

fn freeHuffOff16(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.huff_off16hi_sizes) |s| a.free(s);
    if (c.huff_off16lo_sizes) |s| a.free(s);
    if (c.huff_off16hi_offsets) |o| a.free(o);
    if (c.huff_off16lo_offsets) |o| a.free(o);
    c.huff_off16hi_sizes = null;
    c.huff_off16hi_offsets = null;
    c.huff_off16lo_sizes = null;
    c.huff_off16lo_offsets = null;
}

fn freeAssembled(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.assembled_offsets) |o| a.free(o);
    if (c.assembled_sizes) |s| a.free(s);
    c.assembled_offsets = null;
    c.assembled_sizes = null;
}
