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

// VK adaptation: per-phase QPC profiling for the encode hot path. Mirrors
// the decode-side SLZ_VK_PROFILE_PHASES machinery (srcVK/decode/decode_dispatch.zig
// :g_phase_*_ns + phaseProfileInit + printAndResetPhaseProfile). Pure
// host-side measurement around the existing dispatch / H2D / D2H / sync
// call sites — adds ~50-200 ns per checkpoint (negligible vs the 332 ms
// warm encode wall). Used by bench_compress.zig to localize where the
// residual ~190 ms gap vs CUDA lives. Counters are zeroed by
// encPhaseProfileInit(); printed + reset by printAndResetEncodePhaseProfile().
// Reads SLZ_VK_PROFILE_PHASES env var on init() so the toggle matches
// the decode-side gate.
pub var g_enc_phase_profile_enabled: bool = false;
pub var g_enc_phase_count: u64 = 0;
pub var g_enc_phase_cpu_descs_build_ns: i64 = 0;
pub var g_enc_phase_wrap_input_h2d_ns: i64 = 0;
pub var g_enc_phase_ensure_gpu_out_buf_ns: i64 = 0;
pub var g_enc_phase_lz_h2d_input_ns: i64 = 0;
pub var g_enc_phase_lz_h2d_descs_init_ns: i64 = 0;
pub var g_enc_phase_lz_pre_kernel_sync_ns: i64 = 0;
pub var g_enc_phase_lz_kernel_ns: i64 = 0;
pub var g_enc_phase_lz_d2h_comp_sizes_ns: i64 = 0;
pub var g_enc_phase_lz_d2h_gather_ns: i64 = 0;
pub var g_enc_phase_asm_h2d_descs_ns: i64 = 0;
pub var g_enc_phase_asm_measure_dispatch_ns: i64 = 0;
pub var g_enc_phase_asm_d2h_enc_sizes_ns: i64 = 0;
pub var g_enc_phase_asm_cpu_prefix_sum_ns: i64 = 0;
pub var g_enc_phase_asm_h2d_descs_second_ns: i64 = 0;
pub var g_enc_phase_asm_write_dispatch_ns: i64 = 0;
pub var g_enc_phase_frame_asm_h2ds_ns: i64 = 0;
pub var g_enc_phase_frame_asm_dispatch_ns: i64 = 0;
pub var g_enc_phase_d2h_final_ns: i64 = 0;
pub var g_enc_phase_host_finalize_ns: i64 = 0;

pub fn encPhaseProfileInit() void {
    g_enc_phase_profile_enabled = std.c.getenv("SLZ_VK_PROFILE_PHASES") != null;
    g_enc_phase_count = 0;
    g_enc_phase_cpu_descs_build_ns = 0;
    g_enc_phase_wrap_input_h2d_ns = 0;
    g_enc_phase_ensure_gpu_out_buf_ns = 0;
    g_enc_phase_lz_h2d_input_ns = 0;
    g_enc_phase_lz_h2d_descs_init_ns = 0;
    g_enc_phase_lz_pre_kernel_sync_ns = 0;
    g_enc_phase_lz_kernel_ns = 0;
    g_enc_phase_lz_d2h_comp_sizes_ns = 0;
    g_enc_phase_lz_d2h_gather_ns = 0;
    g_enc_phase_asm_h2d_descs_ns = 0;
    g_enc_phase_asm_measure_dispatch_ns = 0;
    g_enc_phase_asm_d2h_enc_sizes_ns = 0;
    g_enc_phase_asm_cpu_prefix_sum_ns = 0;
    g_enc_phase_asm_h2d_descs_second_ns = 0;
    g_enc_phase_asm_write_dispatch_ns = 0;
    g_enc_phase_frame_asm_h2ds_ns = 0;
    g_enc_phase_frame_asm_dispatch_ns = 0;
    g_enc_phase_d2h_final_ns = 0;
    g_enc_phase_host_finalize_ns = 0;
}

inline fn qpcDeltaNs(start: i64, end: i64) i64 {
    return @intFromFloat(vk_api.qpcMs(start, end) * 1e6);
}

pub fn printAndResetEncodePhaseProfile(w: anytype) void {
    if (!g_enc_phase_profile_enabled or g_enc_phase_count == 0) return;
    const n = @as(f64, @floatFromInt(g_enc_phase_count));
    const ns_to_ms_per = struct {
        fn f(total_ns: i64, count: f64) f64 {
            return (@as(f64, @floatFromInt(total_ns)) / 1e6) / count;
        }
    }.f;
    w.print("phase: enc.count                  {d}\n", .{g_enc_phase_count}) catch {};
    w.print("phase: enc.cpu_descs_build        {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_cpu_descs_build_ns, n)}) catch {};
    w.print("phase: enc.wrap_input_h2d         {d:.4} ms/enc  (~95 MB enwik8 wrap-input H2D)\n", .{ns_to_ms_per(g_enc_phase_wrap_input_h2d_ns, n)}) catch {};
    w.print("phase: enc.ensure_gpu_out_buf     {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_ensure_gpu_out_buf_ns, n)}) catch {};
    w.print("phase: enc.lz.h2d_input           {d:.4} ms/enc  (d2d when wrap-input, else H2D)\n", .{ns_to_ms_per(g_enc_phase_lz_h2d_input_ns, n)}) catch {};
    w.print("phase: enc.lz.h2d_descs_init      {d:.4} ms/enc  (descs H2D + memset d_sizes)\n", .{ns_to_ms_per(g_enc_phase_lz_h2d_descs_init_ns, n)}) catch {};
    w.print("phase: enc.lz.pre_kernel_sync     {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_lz_pre_kernel_sync_ns, n)}) catch {};
    w.print("phase: enc.lz.kernel              {d:.4} ms/enc  (launch + post-kernel sync)\n", .{ns_to_ms_per(g_enc_phase_lz_kernel_ns, n)}) catch {};
    w.print("phase: enc.lz.d2h_comp_sizes      {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_lz_d2h_comp_sizes_ns, n)}) catch {};
    w.print("phase: enc.lz.d2h_gather          {d:.4} ms/enc  (~50 MB iter-3 gather)\n", .{ns_to_ms_per(g_enc_phase_lz_d2h_gather_ns, n)}) catch {};
    w.print("phase: enc.asm.h2d_descs          {d:.4} ms/enc  (assemble descs H2D + sync)\n", .{ns_to_ms_per(g_enc_phase_asm_h2d_descs_ns, n)}) catch {};
    w.print("phase: enc.asm.measure_dispatch   {d:.4} ms/enc  (measure launch + sync)\n", .{ns_to_ms_per(g_enc_phase_asm_measure_dispatch_ns, n)}) catch {};
    w.print("phase: enc.asm.d2h_enc_sizes      {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_asm_d2h_enc_sizes_ns, n)}) catch {};
    w.print("phase: enc.asm.cpu_prefix_sum     {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_asm_cpu_prefix_sum_ns, n)}) catch {};
    w.print("phase: enc.asm.h2d_descs_2nd      {d:.4} ms/enc  (second descs H2D after prefix)\n", .{ns_to_ms_per(g_enc_phase_asm_h2d_descs_second_ns, n)}) catch {};
    w.print("phase: enc.asm.write_dispatch     {d:.4} ms/enc  (write launch + sync)\n", .{ns_to_ms_per(g_enc_phase_asm_write_dispatch_ns, n)}) catch {};
    w.print("phase: enc.frame.h2ds             {d:.4} ms/enc  (4 small H2Ds: chunk_dst+asm_off+asm_sz+prefix)\n", .{ns_to_ms_per(g_enc_phase_frame_asm_h2ds_ns, n)}) catch {};
    w.print("phase: enc.frame.dispatch         {d:.4} ms/enc  (frame_assemble launch + sync)\n", .{ns_to_ms_per(g_enc_phase_frame_asm_dispatch_ns, n)}) catch {};
    w.print("phase: enc.d2h_final              {d:.4} ms/enc  (~50 MB compressed frame D2H)\n", .{ns_to_ms_per(g_enc_phase_d2h_final_ns, n)}) catch {};
    w.print("phase: enc.host_finalize          {d:.4} ms/enc\n", .{ns_to_ms_per(g_enc_phase_host_finalize_ns, n)}) catch {};
    const totalled =
        g_enc_phase_cpu_descs_build_ns + g_enc_phase_wrap_input_h2d_ns +
        g_enc_phase_ensure_gpu_out_buf_ns + g_enc_phase_lz_h2d_input_ns +
        g_enc_phase_lz_h2d_descs_init_ns + g_enc_phase_lz_pre_kernel_sync_ns +
        g_enc_phase_lz_kernel_ns + g_enc_phase_lz_d2h_comp_sizes_ns +
        g_enc_phase_lz_d2h_gather_ns + g_enc_phase_asm_h2d_descs_ns +
        g_enc_phase_asm_measure_dispatch_ns + g_enc_phase_asm_d2h_enc_sizes_ns +
        g_enc_phase_asm_cpu_prefix_sum_ns + g_enc_phase_asm_h2d_descs_second_ns +
        g_enc_phase_asm_write_dispatch_ns + g_enc_phase_frame_asm_h2ds_ns +
        g_enc_phase_frame_asm_dispatch_ns + g_enc_phase_d2h_final_ns +
        g_enc_phase_host_finalize_ns;
    w.print("phase: enc.SUM                    {d:.4} ms/enc\n", .{ns_to_ms_per(totalled, n)}) catch {};
    encPhaseProfileInit();
}

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

/// CUDA reference: src/encode/fast_framed.zig::resolveScGroupSize
/// (updated 2026-06-09). Pick the sc_group_size to advertise in the
/// frame header. Honors a caller override; otherwise always 0.25.
/// The previous saturation-threshold switch to 0.5 doubled every
/// decode warp's serial chain at 1 GB scale for a measured 1.8×
/// decode-time cost; sc=0.25 is the configuration that beats nvCOMP
/// LZ4/Zstd on both ratio and decode speed. Ported in step.
fn resolveScGroupSize(src_len: usize, override: ?f32) f32 {
    _ = src_len;
    if (override) |ov| return ov;
    return 0.25;
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
/// v4 #19 (CUDA-mirror): the frame's effective chunk size - the
/// Merkle chunk grid. MUST match compressFramedOne's eff_chunk.
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
        .dictionary_id = opts.dictionary_id,
        .content_checksum = false,
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
    // VK adaptation: per-phase profile checkpoint (cpu_descs_build /
    // wrap_input_h2d / ensure_gpu_out_buf). Pure measurement — does not
    // change any dispatch / H2D / sync logic.
    const _prof = g_enc_phase_profile_enabled;
    const _t_wall0 = if (_prof) vk_api.qpcNow() else 0;
    const _t_setup0 = _t_wall0;
    // Snapshot the GPU-phase accumulators that nest inside this setup
    // span so cpu_descs_build (computed at the end of setup) can subtract
    // them out and remain pure-CPU. Both phases run zero or once per encode.
    const _snap_wrap_in = if (_prof) g_enc_phase_wrap_input_h2d_ns else 0;
    const _snap_gob = if (_prof) g_enc_phase_ensure_gpu_out_buf_ns else 0;
    // Snapshot every other phase accumulator so host_finalize (residual)
    // can compute wall - all-attributed-phase-deltas at end of function.
    const _snap_lz_h2d_in = if (_prof) g_enc_phase_lz_h2d_input_ns else 0;
    const _snap_lz_h2d_di = if (_prof) g_enc_phase_lz_h2d_descs_init_ns else 0;
    const _snap_lz_presync = if (_prof) g_enc_phase_lz_pre_kernel_sync_ns else 0;
    const _snap_lz_kern = if (_prof) g_enc_phase_lz_kernel_ns else 0;
    const _snap_lz_dsz = if (_prof) g_enc_phase_lz_d2h_comp_sizes_ns else 0;
    const _snap_lz_gather = if (_prof) g_enc_phase_lz_d2h_gather_ns else 0;
    const _snap_asm_h2d = if (_prof) g_enc_phase_asm_h2d_descs_ns else 0;
    const _snap_asm_meas = if (_prof) g_enc_phase_asm_measure_dispatch_ns else 0;
    const _snap_asm_desz = if (_prof) g_enc_phase_asm_d2h_enc_sizes_ns else 0;
    const _snap_asm_psum = if (_prof) g_enc_phase_asm_cpu_prefix_sum_ns else 0;
    const _snap_asm_h2d2 = if (_prof) g_enc_phase_asm_h2d_descs_second_ns else 0;
    const _snap_asm_wr = if (_prof) g_enc_phase_asm_write_dispatch_ns else 0;
    const _snap_fa_h2ds = if (_prof) g_enc_phase_frame_asm_h2ds_ns else 0;
    const _snap_fa_disp = if (_prof) g_enc_phase_frame_asm_dispatch_ns else 0;
    const _snap_d2h_final = if (_prof) g_enc_phase_d2h_final_ns else 0;
    const _snap_cpu_descs = if (_prof) g_enc_phase_cpu_descs_build_ns else 0;
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
    // VK adaptation A-023 (companion): when the CLI invoked this path
    // (override == 0 → owns_wrap_*) we historically allocated a
    // d_host_wrap_input device buffer and H2D'd src into it, then let
    // gpuCompressImpl D2D from d_input_override (= d_host_wrap_input)
    // into d_input_persist (CUDA-mirror src/encode/fast_framed.zig:191-
    // 199). On enwik9 L5 that doubles input residency (2 × 1 GB) on top
    // of the already-tight hash + d_output_persist + huff_* allocations,
    // and the downstream d_asm_out alloc fails. The two-buffer pattern
    // is a CUDA-allocator concession (paged WDDM2 memory hides the
    // duplication); on VK we instead let gpuCompressImpl H2D directly
    // into d_input_persist (its `d_input_override == 0` branch already
    // handles that), then point d_input_override at d_input_persist
    // BEFORE the frame_assemble call so it can read source bytes for
    // uncompressed chunks. Net VRAM savings: 1 GB for enwik9, scales
    // with src.len.
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
    // VK adaptation: time ensureGpuOutBuf separately — first call per
    // size grows the persistent buffer (ensureBuf destroy+create), warm
    // calls just check the size.
    const _t_gob0 = if (_prof) vk_api.qpcNow() else 0;
    var gpu_out_fallback: ?[]u8 = null;
    defer if (gpu_out_fallback) |fb| allocator.free(fb);
    const gpu_out: []u8 = if (gpu_enc.ensureGpuOutBuf(enc_ctx, n_gpu_blocks * per_block_cap)) |buf|
        buf
    else blk: {
        const fb = try allocator.alloc(u8, n_gpu_blocks * per_block_cap);
        gpu_out_fallback = fb;
        break :blk fb;
    };
    if (_prof) g_enc_phase_ensure_gpu_out_buf_ns += qpcDeltaNs(_t_gob0, vk_api.qpcNow());

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
    // VK adaptation: cpu_descs_build = total setup span - already-attributed
    // wrap_input_h2d + ensure_gpu_out_buf (those checkpoints sat inside
    // this same span and bumped their own accumulators; subtract the
    // per-encode delta via the entry-snapshot so SUM stays = wall).
    if (_prof) {
        const _t_setup_end = vk_api.qpcNow();
        const setup_total = qpcDeltaNs(_t_setup0, _t_setup_end);
        const sub_phases = (g_enc_phase_wrap_input_h2d_ns - _snap_wrap_in) +
            (g_enc_phase_ensure_gpu_out_buf_ns - _snap_gob);
        g_enc_phase_cpu_descs_build_ns += setup_total - sub_phases;
    }

    if (!gpu_enc.gpuCompressImpl(enc_ctx, src, gpu_out, descs, comp_sizes, io, opts.level))
        return error.DestinationTooSmall;
    // VK adaptation A-023 (companion): now that d_input_persist holds
    // the H2D'd source (gpuCompressImpl's `override == 0` branch did
    // that for us), expose it via d_input_override so the downstream
    // frame_assemble dispatch (line 642 below) reads the same source
    // bytes from there. Same bytes, no second device buffer.
    if (owns_wrap_input) {
        enc_ctx.d_input_override = enc_ctx.d_input_persist;
    }

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

    // VK adaptation: d2h_final = the ~50 MB compressed-frame D2H readback.
    // host_finalize = any post-D2H bookkeeping that runs before the
    // function returns. The count++ bumps g_enc_phase_count by exactly 1
    // per encode so per-encode averages are sound (every other accumulator
    // sums total ns across all encodes since encPhaseProfileInit).
    //
    // VK adaptation (iter-4): final-frame D2H routed through transfer queue
    // + VK_EXT_external_memory_host import of dst (iter-7+11+15 pattern from
    // decode side, already proven on enc.lz.d2h_gather in iter-3). Per the
    // encode phase profiler (d83ea21), this single call was 71% of warm-mode
    // encode wallclock at 0.21 GB/s on the main queue. Transfer queue gather
    // gets ~10 GB/s for the same payload. Strictly downstream of frame_assemble
    // dispatch — does not touch any kernel sync point.
    //
    // The caller-provided `dst` slice comes from a std.heap allocator (see
    // bench_compress.zig:52) and is NOT page-aligned, so the import path
    // would refuse it. We gather into the persistent d2h_final_buf
    // (procMallocHost-backed, page-aligned, stable host_ptr → iter-8 LRU
    // import cache hits after the first call) and @memcpy to `dst` at the
    // end. The memcpy cost (~10 ms for 50 MB on DDR5) is ~12x cheaper than
    // the prior sync staging D2H on the main queue (~245 ms / 0.21 GB/s).
    // Fallback: if the pinned-buffer alloc fails OR the gather slot is
    // unbound, take the pre-fix copyDeviceToHost path (preserves correctness
    // on every backend including init-failed contexts).
    if (owns_wrap_output) {
        if (dst.len < frame_size) return error.DestinationTooSmall;
        const _t_d2h0 = if (_prof) vk_api.qpcNow() else 0;

        var did_fast_path: bool = false;
        if (vk_api.procs.d2h_offset_gather) |gather_fn| {
            if (gpu_enc.ensureD2hFinalBuf(enc_ctx, frame_size)) |pinned| {
                const region = vk_api.VkBufferCopyRegion{
                    .src_offset = 0,
                    .dst_offset = 0,
                    .size = frame_size,
                };
                const rc = gather_fn(@ptrCast(pinned.ptr), enc_ctx.d_host_wrap_output, @ptrCast(&region), 1);
                if (rc == vk_api.VK_SUCCESS_RC) {
                    @memcpy(dst[0..frame_size], pinned[0..frame_size]);
                    did_fast_path = true;
                }
            }
        }
        if (!did_fast_path) {
            if (!gpu_enc.copyDeviceToHost(dst[0..frame_size], enc_ctx.d_host_wrap_output))
                return error.DestinationTooSmall;
        }
        if (_prof) g_enc_phase_d2h_final_ns += qpcDeltaNs(_t_d2h0, vk_api.qpcNow());
    }
    // VK adaptation: host_finalize = wall - every attributed phase delta.
    // Captures host stretches not explicitly bracketed (allocator overhead,
    // assembleFrame CPU prep, defers, etc.) so SUM == wall by construction.
    if (_prof) {
        const wall = qpcDeltaNs(_t_wall0, vk_api.qpcNow());
        const attributed =
            (g_enc_phase_cpu_descs_build_ns - _snap_cpu_descs) +
            (g_enc_phase_wrap_input_h2d_ns - _snap_wrap_in) +
            (g_enc_phase_ensure_gpu_out_buf_ns - _snap_gob) +
            (g_enc_phase_lz_h2d_input_ns - _snap_lz_h2d_in) +
            (g_enc_phase_lz_h2d_descs_init_ns - _snap_lz_h2d_di) +
            (g_enc_phase_lz_pre_kernel_sync_ns - _snap_lz_presync) +
            (g_enc_phase_lz_kernel_ns - _snap_lz_kern) +
            (g_enc_phase_lz_d2h_comp_sizes_ns - _snap_lz_dsz) +
            (g_enc_phase_lz_d2h_gather_ns - _snap_lz_gather) +
            (g_enc_phase_asm_h2d_descs_ns - _snap_asm_h2d) +
            (g_enc_phase_asm_measure_dispatch_ns - _snap_asm_meas) +
            (g_enc_phase_asm_d2h_enc_sizes_ns - _snap_asm_desz) +
            (g_enc_phase_asm_cpu_prefix_sum_ns - _snap_asm_psum) +
            (g_enc_phase_asm_h2d_descs_second_ns - _snap_asm_h2d2) +
            (g_enc_phase_asm_write_dispatch_ns - _snap_asm_wr) +
            (g_enc_phase_frame_asm_h2ds_ns - _snap_fa_h2ds) +
            (g_enc_phase_frame_asm_dispatch_ns - _snap_fa_disp) +
            (g_enc_phase_d2h_final_ns - _snap_d2h_final);
        g_enc_phase_host_finalize_ns += wall - attributed;
    }
    if (_prof) g_enc_phase_count += 1;
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
