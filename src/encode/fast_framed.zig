//! Fast codec (L1-L5) frame builder, GPU-only.
//!
//! One path: GPU LZ encode → Huffman pass (L3-L5) → assembly kernel
//! chain. The final SLZ1 frame is written directly into a device buffer
//! by `slzFrameAssembleKernel`. When the caller did not pass a device
//! input or output, this file allocates persistent "host-wrap" device
//! buffers on the `EncodeContext`, H2Ds the source into the wrap input,
//! and D2Hs the assembled frame back to the host `dst` after the kernels
//! finish. The slzCompressAsync entry point passes its own device
//! buffers and skips the wrap entirely.
//!
//! Per-chunk uncompressed handling lives in the assembly kernel: when
//! any sub-chunk in a chunk fails to beat raw, the host writes
//! `UNCOMPRESSED_CHUNK_MARKER` into that chunk's offset slot, and the
//! kernel switches its per-chunk write path to copy raw bytes from the
//! source device buffer instead of the assembled payload buffer.

const std = @import("std");

const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const lz_constants = @import("../format/streamlz_constants.zig");

/// Inputs at or below this size are too small for the LZ kernel's
/// per-warp setup cost to ever produce a smaller output; emit them as a
/// whole-frame uncompressed body.
const min_source_length: usize = 128;

const encoder = @import("streamlz_encoder.zig");
const Options = encoder.Options;
const CompressError = encoder.CompressError;

const gpu_enc = @import("driver.zig");
const cuda_api = @import("../decode/cuda_api.zig");
const enc_phase = @import("enc_phase.zig");

/// Number of decoder warps each SM can host simultaneously on sm_8x /
/// sm_9x. Used to size the saturation threshold below which sc_group=0.25
/// (more sub-chunks → more parallel warps) beats sc_group=0.5 (larger
/// sub-chunks → better ratio).
const decoder_warps_per_sm: usize = 48;
/// Bytes each decoder warp consumes per sub-chunk at sc_group=0.5
/// (the 128 KB sub-chunk size — sc_group encodes a fraction of 256 KB).
const sc05_bytes_per_warp: usize = 128 * 1024;

/// Per-sub-chunk staging headroom for the raw LZ payload. Three bytes
/// per stream count, two for the off16 count, plus slack for off32 and
/// length overshoot. `× 3` of `gpu_block` is empirically enough across
/// enwik8 + silesia at all five levels.
const gpu_block_capacity_multiplier: usize = 3;

/// Fallback SM count when the CUDA driver did not populate
/// `cuda_api.sm_count` (very old driver builds). Conservative for
/// consumer hardware; sc=0.5 only kicks in above the saturation
/// threshold, which large inputs reach regardless.
const sm_count_fallback: usize = 34;

/// Maximum chunk count `slzFrameAssembleKernel` accepts in a single
/// grid. Inputs larger than `assembly_chunk_cap × eff_chunk_size`
/// (currently 1 GB at sc=0.25) get emitted as a whole-frame
/// uncompressed body on host without launching the kernel chain.
const assembly_chunk_cap: usize = 16384;

/// Pick the `sc_group_size` to advertise in the frame header. Honors a
/// caller override; otherwise always 0.25 (64 KB sub-chunks).
///
/// 2026-06-09: was "0.25 below the GPU saturation threshold, 0.5 at or
/// above it" on the theory that once the GPU is saturated, bigger
/// sub-chunks buy ratio for free. Measurement on 1 GB enwik9 (RTX 4060
/// Ti) disproved the "free": sc=0.5 halves the warp count and doubles
/// every warp's serial decode chain, costing 1.8× decode wall-clock
/// (L1 44.2 ms → 24.3 ms at sc=0.25; L3-L5 ~2×). The ratio cost of
/// 0.25 is ~2.3 pp at L1 / ~1.4 pp at L3 on enwik9 — and 0.25 is the
/// configuration that beats nvCOMP LZ4 (L1) and nvCOMP Zstd (L5) on
/// BOTH ratio and decode speed simultaneously. Callers that want
/// maximum ratio pass `--sc 0.5` (the decoder handles any sc value;
/// it's stamped in the frame header).
fn resolveScGroupSize(src_len: usize, override: ?f32) f32 {
    _ = src_len;
    if (override) |ov| return ov;
    return 0.25;
}

/// Map the unified L1-L5 user level to the codec level written in the
/// frame header. Mirrors the historical mapping; internal level 4 is
/// skipped because the parser variant it would select lost to internal
/// 5 on every workload measured.
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

/// v4 #19: the frame's effective chunk size for a given input length
/// and sc override - the chunk grid the Merkle root is defined over.
/// MUST match the eff_chunk used by compressFramedOne below (and the
/// decoder's walk grid).
pub fn effChunkFor(src_len: usize, sc_override: ?f32) usize {
    const sc = resolveScGroupSize(src_len, sc_override);
    return @min(frame.scGroupSizeToBytes(sc), lz_constants.chunk_size);
}

pub fn compressFramedOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
) CompressError!usize {
    if (!gpu_enc.isAvailable()) return error.DestinationTooSmall;

    var pos: usize = 0;
    const sc_grp = resolveScGroupSize(src.len, opts.sc_group_size_override);

    const hdr_len = frame.writeHeader(dst, .{
        .codec = .fast,
        .level = codecLevelFor(opts.level),
        .block_size = opts.block_size,
        .sc_group_size = sc_grp,
        .content_size = if (opts.include_content_size) @as(u64, @intCast(src.len)) else null,
        .dictionary_id = null,
        .content_checksum = opts.content_checksum,
        .chunk_merkle = opts.chunk_checksum,
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

/// Runs the GPU encode pipeline and the device-resident assembly chain.
/// When the caller did not pass `d_input_override` / `d_output_override`,
/// allocates persistent host-wrap buffers on `enc_ctx`, H2Ds the source,
/// runs the kernels, then D2Hs the assembled frame back into `dst`.
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
        // Input too large for the kernel grid. Fall back to a whole-
        // frame uncompressed body. Rare: at sc=0.25 the threshold is
        // 1 GB; users encoding more than that should use multi-frame
        // batching at a higher level.
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
    const gpu_out = try allocator.alloc(u8, n_gpu_blocks * per_block_cap);
    defer allocator.free(gpu_out);

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

    // SLZ_PROFILE_PHASES checkpoints (srcVK profiler backport) — each
    // begin() is 0 when profiling is off and add() then no-ops.
    var t_ph = enc_phase.begin();
    if (!gpu_enc.gpuCompressImpl(enc_ctx, src, gpu_out, descs, comp_sizes, io, opts.level))
        return error.DestinationTooSmall;
    enc_phase.add(.lz_total, t_ph);

    t_ph = enc_phase.begin();
    const did_huff_lit = opts.level >= 3 and gpu_enc.gpuEncodeLiteralsHuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_lit) freeHuffLit(allocator, enc_ctx);
    enc_phase.add(.huff_lit, t_ph);
    t_ph = enc_phase.begin();
    const did_huff_tok = opts.level >= 3 and gpu_enc.gpuEncodeTokensHuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_tok) freeHuffTok(allocator, enc_ctx);
    enc_phase.add(.huff_tok, t_ph);
    t_ph = enc_phase.begin();
    const did_huff_off16 = opts.level >= 3 and gpu_enc.gpuEncodeOff16HuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_off16) freeHuffOff16(allocator, enc_ctx);
    enc_phase.add(.huff_off16, t_ph);

    t_ph = enc_phase.begin();
    if (!gpu_enc.gpuAssembleFrameImpl(enc_ctx, allocator, descs, comp_sizes))
        return error.DestinationTooSmall;
    defer freeAssembled(allocator, enc_ctx);
    enc_phase.add(.asm_device, t_ph);

    t_ph = enc_phase.begin();
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
    enc_phase.add(.asm_host, t_ph);

    if (owns_wrap_output) {
        if (dst.len < frame_size) return error.DestinationTooSmall;
        t_ph = enc_phase.begin();
        if (!gpu_enc.copyDeviceToHost(dst[0..frame_size], enc_ctx.d_host_wrap_output))
            return error.DestinationTooSmall;
        enc_phase.add(.wrap_d2h, t_ph);
    }
    return frame_size;
}

/// Launch `slzFrameAssembleKernel`. Per-chunk uncompressed handling is
/// signaled to the kernel via `UNCOMPRESSED_CHUNK_MARKER` in the chunk's
/// offset slot; the size slot then carries the chunk's raw source byte
/// count instead of its assembled payload size.
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

    // `assembled_offsets`/`assembled_sizes` are indexed per SUB-CHUNK by
    // `gpuAssembleFrameImpl`, but the walk below indexes them by chunk
    // index `ci`. That is correct only when every chunk has exactly one
    // sub-chunk — which holds for any `sc_group_size <= 0.5` (the
    // current adaptive picker's range). A future override of
    // `sc_group_size >= 1.0` would let `eff_chunk > sub_chunk_cap` and
    // produce more than one sub-chunk per chunk, breaking the 1:1
    // index mapping. Assert it loudly here.
    std.debug.assert(comp_sizes.len == n_chunks);

    const internal_block_flags: u8 = 0x05 | 0x10 | 0x40; // magic | self_contained | keyframe
    const internal_block_codec: u8 = @intFromEnum(block_header.CodecType.fast);

    const per_chunk_asm_size_buf = try allocator.alloc(u32, n_chunks);
    defer allocator.free(per_chunk_asm_size_buf);
    const per_chunk_asm_off_buf = try allocator.alloc(u32, n_chunks);
    defer allocator.free(per_chunk_asm_off_buf);

    // Walk sub-chunks chunk-by-chunk. A chunk is emitted uncompressed
    // when any of its sub-chunks failed to beat raw.
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
