//! Fast codec frame builder. Produces a single SLZ1-framed block for
//! L1-L5, GPU-encoded.
//!
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.
//!
//! Flow:
//!
//!   1. Plan: derive `sc_group_size`, count sub-chunks, allocate per-block
//!      descriptors and the raw output staging buffer.
//!   2. LZ: launch `slzLzEncodeKernel`. Result is a packed [literals,
//!      tokens, off16, off32, lengths] payload per sub-chunk.
//!   3. Huffman (L3-L5): run the three Huffman-encode kernels over the
//!      raw lit / tok / off16 streams. Each produces chunk_type=4 bodies.
//!   4. Assembly: pick a path —
//!       a. Pure-D2D (only when the caller passed a device output and
//!          every sub-chunk maps 1:1 to a chunk and compresses below
//!          raw): launch `slzFrameAssembleKernel` and return.
//!       b. Otherwise: host walk. For each sub-chunk, splice the
//!          device-assembled wire block when available, else fall back to
//!          `gpu_assembly.reencodeGpuWithEntropy` which builds the wire
//!          block on the host from the raw streams + GPU-Huffman bodies.
//!   5. SC tail prefix table + end mark.

const std = @import("std");

const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const lz_constants = @import("../format/streamlz_constants.zig");
const fast_constants = @import("fast/fast_constants.zig");

const encoder = @import("streamlz_encoder.zig");
const Options = encoder.Options;
const CompressError = encoder.CompressError;
const entropyOptionsForLevel = encoder.entropyOptionsForLevel;

const gpu_assembly = @import("gpu_stream_assembly.zig");
const gpu_enc = @import("driver.zig");
const cuda_api = @import("../decode/cuda_api.zig");

const areAllBytesEqual = block_header.areAllBytesEqual;

/// SM count at which sc=0.5 (128 KB sub-chunks) starts outperforming
/// sc=0.25 (64 KB) on the decode side: every decoder warp gets exactly
/// one sub-chunk, so saturation needs `sm_count × warps_per_sm` warps.
/// `48` is the architectural per-SM warp cap on sm_8x / sm_9x.
const decoder_warps_per_sm: usize = 48;
const sc05_bytes_per_warp: usize = 128 * 1024;

/// Per-sub-chunk staging headroom for the raw LZ payload. Three bytes
/// per stream count, two bytes for the off16 count, plus generous slack
/// for off32 and length overshoot. `× 3` of `gpu_block` is empirically
/// enough across enwik8 + silesia at all five levels.
const gpu_block_capacity_multiplier: usize = 3;

/// Pick the `sc_group_size` to advertise in the frame header. Honors a
/// caller override; otherwise picks 0.25 (64 KB sub-chunks) below the GPU
/// saturation threshold and 0.5 (128 KB) at or above it.
///
/// The threshold matters because every decoder warp processes one
/// sub-chunk. Below saturation, more sub-chunks → more parallel work for
/// the decode kernel; above saturation, larger sub-chunks → better
/// compression ratio without losing decode throughput.
///
/// Falls back to assuming an RTX 4060 Ti-class GPU (34 SMs) when
/// `cuda_api.sm_count` hasn't been populated yet.
fn resolveScGroupSize(src_len: usize, override: ?f32) f32 {
    if (override) |ov| return ov;
    const sm_count_fallback: usize = 34;
    const sm_count: usize = if (cuda_api.sm_count > 0)
        @intCast(cuda_api.sm_count)
    else
        sm_count_fallback;
    const saturation_bytes = sm_count * decoder_warps_per_sm * sc05_bytes_per_warp;
    return if (src_len >= saturation_bytes) 0.5 else 0.25;
}

/// Map the unified L1-L5 user level to the codec level written in the
/// frame header. Mirrors the original CPU codec's `MapLevel`: user 4 is
/// skipped on the way down because the parser variant it would select
/// (Fast 4) consistently lost to Fast 5 on every workload we measured.
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
        .content_checksum = false,
    }) catch return error.DestinationTooSmall;
    pos += hdr_len;

    if (src.len == 0) {
        if (pos + 4 > dst.len) return error.DestinationTooSmall;
        frame.writeEndMark(dst[pos..]);
        return pos + 4;
    }

    const can_compress = src.len > fast_constants.min_source_length;
    const self_contained: bool = opts.self_contained or opts.two_phase or true; // L1-L5 are always SC
    const sc_flag_bit: u8 = if (self_contained) 0x10 else 0;
    const two_phase_flag_bit: u8 = if (opts.two_phase) 0x20 else 0;

    const frame_block_hdr_pos: usize = pos;
    pos += 8;
    const frame_block_start: usize = pos;

    if (!can_compress) {
        return writeUncompressedFrame(dst, frame_block_hdr_pos, frame_block_start, src);
    }

    pos = try gpuEncodeAndAssemble(
        allocator,
        io,
        src,
        dst,
        pos,
        opts,
        enc_ctx,
        sc_grp,
        sc_flag_bit,
        two_phase_flag_bit,
        frame_block_hdr_pos,
        frame_block_start,
    );
    if (enc_ctx.output_written_to_device) {
        // Pure-D2D path returned the final size as `pos` directly; the
        // frame is fully on device, no host post-processing.
        return pos;
    }

    const block_payload_size = pos - frame_block_start;
    frame.writeBlockHeader(dst[frame_block_hdr_pos..], .{
        .compressed_size = @intCast(block_payload_size),
        .decompressed_size = @intCast(src.len),
        .uncompressed = false,
        .parallel_decode_metadata = false,
    });

    if (pos + 4 > dst.len) return error.DestinationTooSmall;
    frame.writeEndMark(dst[pos..]);
    pos += 4;
    return pos;
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

/// Returns the next write position after assembling the encoded body. When
/// the pure-D2D kernel path runs, returns the final total frame size and
/// sets `enc_ctx.output_written_to_device = true`; the caller short-circuits
/// every host-side write that would follow.
fn gpuEncodeAndAssemble(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    pos_in: usize,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
    sc_grp: f32,
    sc_flag_bit: u8,
    two_phase_flag_bit: u8,
    frame_block_hdr_pos: usize,
    frame_block_start: usize,
) CompressError!usize {
    var pos = pos_in;

    const eff_chunk = @min(frame.scGroupSizeToBytes(sc_grp), lz_constants.chunk_size);
    const sub_chunk_cap: usize = lz_constants.sub_chunk_size;
    const gpu_block: usize = @min(eff_chunk, sub_chunk_cap);
    const n_chunks = (src.len + eff_chunk - 1) / eff_chunk;

    var n_gpu_blocks: usize = 0;
    for (0..n_chunks) |ci| {
        const chunk_size = @min(eff_chunk, src.len - ci * eff_chunk);
        n_gpu_blocks += (chunk_size + gpu_block - 1) / gpu_block;
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
                // Only the very first sub-chunk of the frame writes the
                // 8 raw init bytes. Later chunks rely on the SC prefix
                // table emitted at the end of the block.
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

    if (!gpu_enc.gpuCompressImpl(enc_ctx, src, gpu_out, descs, comp_sizes, io, opts.level)) {
        return error.DestinationTooSmall;
    }

    // L3-L5 add the Huffman pass. Each call leaves the encoded bodies on
    // the host (and optionally also on the device when huff_keep_device
    // is set, used by the pure-D2D assembly path below). The defer blocks
    // free the host buffers regardless of which assembly path runs.
    const want_gpu_assemble: bool =
        opts.level >= 3 and
        (std.c.getenv("SLZ_GPU_ASSEMBLE") != null or enc_ctx.d_output_override != 0);
    enc_ctx.huff_keep_device = want_gpu_assemble;

    const did_huff_lit = opts.level >= 3 and gpu_enc.gpuEncodeLiteralsHuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_lit) freeHuffLit(allocator, enc_ctx);
    const did_huff_tok = opts.level >= 3 and gpu_enc.gpuEncodeTokensHuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_tok) freeHuffTok(allocator, enc_ctx);
    const did_huff_off16 = opts.level >= 3 and gpu_enc.gpuEncodeOff16HuffImpl(enc_ctx, allocator, gpu_out, descs, comp_sizes);
    defer if (did_huff_off16) freeHuffOff16(allocator, enc_ctx);

    const did_gpu_assemble: bool = if (want_gpu_assemble)
        gpu_enc.gpuAssembleFrameImpl(enc_ctx, allocator, descs, comp_sizes)
    else
        false;
    defer if (did_gpu_assemble) freeAssembled(allocator, enc_ctx);

    // Pure-D2D fast path: the assembly kernel can write the entire frame
    // straight into the caller's device buffer when sc=0.25 (so each chunk
    // is one sub-chunk), no dictionary, and every chunk compressed below
    // its raw size (the kernel doesn't model the uncompressed fallback).
    if (try tryPureD2D(
        enc_ctx,
        allocator,
        src,
        dst,
        opts,
        eff_chunk,
        gpu_block,
        n_chunks,
        comp_sizes,
        did_gpu_assemble,
        sc_flag_bit,
        two_phase_flag_bit,
        frame_block_hdr_pos,
    )) |frame_size| return frame_size;

    pos = try assembleHostSide(
        allocator,
        src,
        dst,
        pos,
        opts,
        enc_ctx,
        eff_chunk,
        gpu_block,
        n_chunks,
        per_block_cap,
        gpu_out,
        comp_sizes,
        did_huff_lit,
        did_huff_tok,
        did_huff_off16,
        sc_flag_bit,
        two_phase_flag_bit,
    );

    // SC tail prefix table: (n_chunks - 1) × 8 bytes carrying the first 8
    // raw bytes of every chunk past the first, used by the parallel
    // decoder to restore the corrupted leading bytes of each chunk after
    // independent decode.
    if (n_chunks > 1) {
        for (1..n_chunks) |ci| {
            const chunk_start = ci * eff_chunk;
            if (chunk_start >= src.len) break;
            const copy_size: usize = @min(@as(usize, 8), src.len - chunk_start);
            if (pos + 8 > dst.len) return error.DestinationTooSmall;
            @memset(dst[pos..][0..8], 0);
            if (enc_ctx.d_input_override != 0) {
                if (!gpu_enc.copyDeviceToHost(dst[pos..][0..copy_size], enc_ctx.d_input_override + chunk_start))
                    return error.DestinationTooSmall;
            } else {
                @memcpy(dst[pos..][0..copy_size], src[chunk_start..][0..copy_size]);
            }
            pos += 8;
        }
    }

    _ = frame_block_start;
    return pos;
}

fn freeHuffLit(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.huff_lit_sizes) |s| a.free(s);
    if (c.huff_lit_data) |d| a.free(d);
    if (c.huff_lit_offsets) |o| a.free(o);
    c.huff_lit_sizes = null;
    c.huff_lit_data = null;
    c.huff_lit_offsets = null;
}

fn freeHuffTok(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.huff_tok_sizes) |s| a.free(s);
    if (c.huff_tok_data) |d| a.free(d);
    if (c.huff_tok_offsets) |o| a.free(o);
    c.huff_tok_sizes = null;
    c.huff_tok_data = null;
    c.huff_tok_offsets = null;
}

fn freeHuffOff16(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    // hi_data OWNS the shared buffer; lo_data is an alias that points
    // into the same allocation. Asserting the alias contract keeps any
    // future change that hands lo_data its own allocation honest.
    if (c.huff_off16hi_data) |hi| if (c.huff_off16lo_data) |lo|
        std.debug.assert(hi.ptr == lo.ptr);
    if (c.huff_off16hi_sizes) |s| a.free(s);
    if (c.huff_off16lo_sizes) |s| a.free(s);
    if (c.huff_off16hi_offsets) |o| a.free(o);
    if (c.huff_off16lo_offsets) |o| a.free(o);
    if (c.huff_off16hi_data) |d| a.free(d);
    c.huff_off16hi_sizes = null;
    c.huff_off16hi_data = null;
    c.huff_off16hi_offsets = null;
    c.huff_off16lo_sizes = null;
    c.huff_off16lo_data = null;
    c.huff_off16lo_offsets = null;
}

fn freeAssembled(a: std.mem.Allocator, c: *gpu_enc.EncodeContext) void {
    if (c.assembled_data) |d| a.free(d);
    if (c.assembled_offsets) |o| a.free(o);
    if (c.assembled_sizes) |s| a.free(s);
    c.assembled_data = null;
    c.assembled_offsets = null;
    c.assembled_sizes = null;
}

fn tryPureD2D(
    enc_ctx: *gpu_enc.EncodeContext,
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    opts: Options,
    eff_chunk: usize,
    gpu_block: usize,
    n_chunks: usize,
    comp_sizes: []const u32,
    did_gpu_assemble: bool,
    sc_flag_bit: u8,
    two_phase_flag_bit: u8,
    frame_block_hdr_pos: usize,
) CompressError!?usize {
    if (enc_ctx.d_output_override == 0) return null;
    if (!did_gpu_assemble) return null;
    if (gpu_block != eff_chunk) return null; // sc=0.25 → 1 sub per chunk
    if (enc_ctx.d_input_override == 0) return null; // need device src for SC tail
    const sizes = enc_ctx.assembled_sizes orelse return null;
    const offsets = enc_ctx.assembled_offsets orelse return null;
    if (sizes.len < n_chunks or offsets.len < n_chunks) return null;
    // slzFrameAssembleKernel's `per_chunk_*` arrays live on a fixed-size
    // device buffer; the cap is a kernel implementation detail.
    if (n_chunks > 16384) return null;

    // Every chunk must have beaten raw — the assembly kernel does not
    // model the uncompressed fallback. Bail to the host path on any miss.
    for (0..n_chunks) |ci| {
        const chunk_size = @min(eff_chunk, src.len - ci * eff_chunk);
        if (comp_sizes[ci] >= chunk_size) return null;
    }

    const flags0_d2d: u8 = 0x05 | sc_flag_bit | two_phase_flag_bit | 0x40;
    const codec_byte: u8 = @intFromEnum(block_header.CodecType.fast);

    var per_chunk_asm_size_buf = try allocator.alloc(u32, n_chunks);
    defer allocator.free(per_chunk_asm_size_buf);
    var per_chunk_asm_off_buf = try allocator.alloc(u32, n_chunks);
    defer allocator.free(per_chunk_asm_off_buf);

    var total_chunk_bytes: usize = 0;
    for (0..n_chunks) |ci| {
        per_chunk_asm_off_buf[ci] = offsets[ci];
        per_chunk_asm_size_buf[ci] = sizes[ci];
        total_chunk_bytes += gpu_enc.CHUNK_INTERNAL_HDR_BYTES + sizes[ci];
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
    if (prefix_size > prefix_buf.len) return null;
    @memcpy(prefix_buf[0..frame_block_hdr_pos], dst[0..frame_block_hdr_pos]);
    @memcpy(prefix_buf[frame_block_hdr_pos..][0..8], &block_hdr_buf);

    const total_frame_size = gpu_enc.gpuFrameAssembleImpl(
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
            .internal_hdr0 = flags0_d2d,
            .internal_hdr1 = codec_byte,
        },
        enc_ctx.d_input_override,
        enc_ctx.d_output_override,
    ) orelse return null;
    enc_ctx.output_written_to_device = true;
    _ = opts;
    return total_frame_size;
}

fn assembleHostSide(
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    pos_in: usize,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
    eff_chunk: usize,
    gpu_block: usize,
    n_chunks: usize,
    per_block_cap: usize,
    gpu_out: []u8,
    comp_sizes: []const u32,
    did_huff_lit: bool,
    did_huff_tok: bool,
    did_huff_off16: bool,
    sc_flag_bit: u8,
    two_phase_flag_bit: u8,
) CompressError!usize {
    var pos = pos_in;
    var soff: usize = 0;
    var gpu_bi: usize = 0;

    for (0..n_chunks) |ci| {
        const chunk_size = @min(eff_chunk, src.len - ci * eff_chunk);
        const n_subs = (chunk_size + gpu_block - 1) / gpu_block;

        if (pos + 2 > dst.len) return error.DestinationTooSmall;
        const flags0: u8 = 0x05 | sc_flag_bit | two_phase_flag_bit | 0x40;

        var all_compressed = true;
        for (0..n_subs) |si| {
            const sub_size = @min(gpu_block, chunk_size - si * gpu_block);
            if (comp_sizes[gpu_bi + si] >= sub_size) {
                all_compressed = false;
                break;
            }
        }

        if (!all_compressed) {
            dst[pos] = flags0 | 0x80;
            dst[pos + 1] = @intFromEnum(block_header.CodecType.fast);
            pos += 2;
            if (enc_ctx.d_input_override != 0) {
                if (!gpu_enc.copyDeviceToHost(dst[pos..][0..chunk_size], enc_ctx.d_input_override + soff))
                    return error.DestinationTooSmall;
            } else {
                if (pos + chunk_size > dst.len) return error.DestinationTooSmall;
                @memcpy(dst[pos..][0..chunk_size], src[soff..][0..chunk_size]);
            }
            pos += chunk_size;
        } else {
            dst[pos] = flags0;
            dst[pos + 1] = @intFromEnum(block_header.CodecType.fast);
            pos += 2;

            if (pos + 4 > dst.len) return error.DestinationTooSmall;
            const chunk_hdr_pos = pos;
            pos += 4;

            for (0..n_subs) |si| {
                pos = try writeSubChunk(
                    allocator,
                    dst,
                    pos,
                    opts,
                    enc_ctx,
                    chunk_size,
                    gpu_block,
                    si,
                    gpu_bi + si,
                    gpu_out,
                    comp_sizes,
                    per_block_cap,
                    did_huff_lit,
                    did_huff_tok,
                    did_huff_off16,
                );
            }

            const chunk_payload = pos - chunk_hdr_pos - 4;
            std.mem.writeInt(u32, dst[chunk_hdr_pos..][0..4], @intCast(chunk_payload - 1), .little);
        }

        soff += chunk_size;
        gpu_bi += n_subs;
    }
    return pos;
}

fn writeSubChunk(
    allocator: std.mem.Allocator,
    dst: []u8,
    pos_in: usize,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
    chunk_size: usize,
    gpu_block: usize,
    si: usize,
    gpu_bi_idx: usize,
    gpu_out: []u8,
    comp_sizes: []const u32,
    per_block_cap: usize,
    did_huff_lit: bool,
    did_huff_tok: bool,
    did_huff_off16: bool,
) CompressError!usize {
    var pos = pos_in;
    const raw_cs = comp_sizes[gpu_bi_idx];
    const raw_payload = gpu_out[gpu_bi_idx * per_block_cap ..][0..raw_cs];

    if (enc_ctx.assembled_data) |ad| if (enc_ctx.assembled_offsets) |ao| if (enc_ctx.assembled_sizes) |asz| {
        if (gpu_bi_idx < asz.len) {
            const block = ad[ao[gpu_bi_idx]..][0..asz[gpu_bi_idx]];
            if (pos + block.len > dst.len) return error.DestinationTooSmall;
            @memcpy(dst[pos..][0..block.len], block);
            return pos + block.len;
        }
    };

    const init_bytes: usize = if (gpu_bi_idx == 0) 8 else 0;

    if (opts.level >= 3) {
        const sub_size = @min(gpu_block, chunk_size - si * gpu_block);
        var entropy_options = entropyOptionsForLevel(opts.level);
        entropy_options.allow_tans = false;
        entropy_options.allow_tans32 = false;

        const huff_streams: gpu_assembly.GpuHuffStreams = .{
            .lit = if (did_huff_lit) gpuHuffSlice(enc_ctx.huff_lit_sizes, enc_ctx.huff_lit_data, enc_ctx.huff_lit_offsets, gpu_bi_idx) else null,
            .tok = if (did_huff_tok) gpuHuffSlice(enc_ctx.huff_tok_sizes, enc_ctx.huff_tok_data, enc_ctx.huff_tok_offsets, gpu_bi_idx) else null,
            .off16_hi = if (did_huff_off16) gpuHuffSlice(enc_ctx.huff_off16hi_sizes, enc_ctx.huff_off16hi_data, enc_ctx.huff_off16hi_offsets, gpu_bi_idx) else null,
            .off16_lo = if (did_huff_off16) gpuHuffSlice(enc_ctx.huff_off16lo_sizes, enc_ctx.huff_off16lo_data, enc_ctx.huff_off16lo_offsets, gpu_bi_idx) else null,
        };

        var enc_buf = try allocator.alloc(u8, raw_cs + 4096);
        defer allocator.free(enc_buf);

        const enc_n = gpu_assembly.reencodeGpuWithEntropy(
            allocator,
            raw_payload,
            enc_buf,
            entropy_options,
            0.0,
            init_bytes,
            sub_size,
            huff_streams,
            .none,
            .none,
            .none,
        ) catch 0;

        if (enc_n > 0 and enc_n < raw_cs and pos + 3 + enc_n <= dst.len) {
            const sc_hdr: u32 = @as(u32, @intCast(enc_n)) |
                (@as(u32, 1) << lz_constants.sub_chunk_type_shift) |
                lz_constants.chunk_header_compressed_flag;
            block_header.writeBE24(dst[pos..].ptr, sc_hdr);
            pos += 3;
            @memcpy(dst[pos..][0..enc_n], enc_buf[0..enc_n]);
            return pos + enc_n;
        }
    }

    // Raw sub-chunk fallback (L1-L2 always lands here, L3+ only when the
    // entropy-encoded form didn't beat raw).
    if (pos + 3 + raw_cs > dst.len) return error.DestinationTooSmall;
    const sc_hdr: u32 = @as(u32, @intCast(raw_cs)) |
        (@as(u32, 1) << lz_constants.sub_chunk_type_shift) |
        lz_constants.chunk_header_compressed_flag;
    block_header.writeBE24(dst[pos..].ptr, sc_hdr);
    pos += 3;
    @memcpy(dst[pos..][0..raw_cs], raw_payload);
    return pos + raw_cs;
}

fn gpuHuffSlice(
    sizes_opt: ?[]const u32,
    data_opt: ?[]const u8,
    offsets_opt: ?[]const u32,
    idx: usize,
) ?[]const u8 {
    const sizes = sizes_opt orelse return null;
    const data = data_opt orelse return null;
    const offsets = offsets_opt orelse return null;
    if (idx >= sizes.len) return null;
    const sz = sizes[idx];
    if (sz == 0) return null;
    const off = offsets[idx];
    if (off + sz > data.len) return null;
    return data[off..][0..sz];
}
