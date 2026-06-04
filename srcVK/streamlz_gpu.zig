//! 1:1 port of src/streamlz_gpu.zig.
//!
//! StreamLZ GPU compression library — C ABI implementation. Implements
//! include/streamlz_gpu.h: a handle-based, synchronous GPU-backed
//! compressor exposed to C callers. Each handle owns a private
//! EncodeContext + DecodeContext.

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_driver = @import("decode/driver.zig");
const frame = @import("format/frame_format.zig");
const version = @import("version.zig");
const vk_api = @import("decode/vulkan_api.zig");

const allocator = std.heap.c_allocator;

/// CUDA reference: src/streamlz_gpu.zig:32. Worker-thread stack.
const worker_stack_size: usize = 32 << 20;

/// CUDA reference: src/streamlz_gpu.zig:44. Non-dereferenceable address
/// embedded in the host slice that slzCompressAsync hands to CompressJob
/// when the input lives on the device.
const device_only_host_stub_addr: usize = 0x10;

/// CUDA reference: src/streamlz_gpu.zig:54-57.
fn deviceOnlySrcStub(len: usize) []const u8 {
    const ptr: [*]const u8 = @ptrFromInt(device_only_host_stub_addr);
    return ptr[0..len];
}

// ── Status codes ───────────────────────────────────────────────────────
// CUDA reference: src/streamlz_gpu.zig:63-74. Mirror of `slzStatus_t` in
// include/streamlz_gpu.h.
const SLZ_SUCCESS: c_int = 0;
const SLZ_ERROR_INVALID_HANDLE: c_int = 1;
const SLZ_ERROR_INVALID_ARG: c_int = 2;
const SLZ_ERROR_BUFFER_TOO_SMALL: c_int = 3;
const SLZ_ERROR_CORRUPT_FRAME: c_int = 4;
const SLZ_ERROR_UNSUPPORTED: c_int = 5;
const SLZ_ERROR_CUDA: c_int = 6;
const SLZ_ERROR_OUT_OF_MEMORY: c_int = 7;
const SLZ_ERROR_DEVICE_LOST: c_int = 8;
const SLZ_ERROR_VK_FEATURE_MISSING: c_int = 9;

comptime {
    std.debug.assert(SLZ_SUCCESS == 0);
    std.debug.assert(SLZ_ERROR_OUT_OF_MEMORY == 7);
    std.debug.assert(SLZ_ERROR_DEVICE_LOST == 8);
    std.debug.assert(SLZ_ERROR_VK_FEATURE_MISSING == 9);
}

pub const pseudo_kernel_compressed_size: [*:0]const u8 = "__compressed_size__";
pub const pseudo_kernel_decompressed_size: [*:0]const u8 = "__decompressed_size__";

/// CUDA reference: src/streamlz_gpu.zig:105-110. Mirror of slzCompressOpts_t.
const CompressOpts = extern struct {
    level: c_int = 5,
    enable_profiling: c_int = 0,
    effective_level_out: c_int = 0,
    reserved: [5]c_int = @splat(0),
};

/// CUDA reference: src/streamlz_gpu.zig:113-116. Mirror of slzDecompressOpts_t.
const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = @splat(0),
};

/// CUDA reference: src/streamlz_gpu.zig:119-122. Mirror of slzKernelTiming_t.
const KernelTimingC = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

comptime {
    std.debug.assert(@sizeOf(CompressOpts) == 8 * @sizeOf(c_int));
    std.debug.assert(@sizeOf(DecompressOpts) == 8 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "level") == 0);
    std.debug.assert(@offsetOf(CompressOpts, "enable_profiling") == @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "effective_level_out") == 2 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "reserved") == 3 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(DecompressOpts, "enable_profiling") == 0);
    std.debug.assert(@offsetOf(DecompressOpts, "reserved") == @sizeOf(c_int));
    std.debug.assert(@offsetOf(KernelTimingC, "name") == 0);
    std.debug.assert(@offsetOf(KernelTimingC, "ms") == @sizeOf(*anyopaque));
    std.debug.assert(@sizeOf(KernelTimingC) == 2 * @sizeOf(*anyopaque));
}

/// CUDA reference: src/streamlz_gpu.zig:147-150. Library handle.
const Context = struct {
    enc: gpu_encoder.EncodeContext = .{},
    dec: gpu_driver.DecodeContext = .{},
};

/// CUDA reference: src/streamlz_gpu.zig:170-184. Map an
/// encoder.CompressError to the C ABI status code.
fn mapCompressError(err: anyerror) c_int {
    return switch (err) {
        error.BadLevel, error.BadScGroupSize, error.BadBlockSize => SLZ_ERROR_UNSUPPORTED,
        error.DestinationTooSmall => SLZ_ERROR_BUFFER_TOO_SMALL,
        error.OutOfMemory, error.OutOfDeviceMemory => SLZ_ERROR_OUT_OF_MEMORY,
        error.BackendNotAvailable,
        error.KernelLaunchFailed,
        error.SyncFailed,
        error.CopyFailed,
        error.KernelMissing,
        error.BadMode,
        => SLZ_ERROR_CUDA,
        else => SLZ_ERROR_CUDA,
    };
}

/// CUDA reference: src/streamlz_gpu.zig:200-225. Map a
/// decoder.DecompressError to the C ABI status code.
fn mapDecompressError(err: anyerror) c_int {
    return switch (err) {
        error.OutputTooSmall => SLZ_ERROR_BUFFER_TOO_SMALL,
        error.OutOfMemory, error.OutOfDeviceMemory => SLZ_ERROR_OUT_OF_MEMORY,
        error.BadFrame,
        error.Truncated,
        error.SizeMismatch,
        error.InvalidBlockHeader,
        error.InvalidInternalHeader,
        error.BadChunkHeader,
        error.BlockDataTruncated,
        error.ChecksumMismatch,
        error.ChunkSizeMismatch,
        error.UnknownDictionary,
        error.ContentSizeTooLarge,
        => SLZ_ERROR_CORRUPT_FRAME,
        error.BadMode, error.NotImplementedL2 => SLZ_ERROR_UNSUPPORTED,
        error.BackendNotAvailable,
        error.KernelLaunchFailed,
        error.SyncFailed,
        error.CopyFailed,
        error.KernelMissing,
        => SLZ_ERROR_CUDA,
        else => SLZ_ERROR_CUDA,
    };
}

/// CUDA reference: src/streamlz_gpu.zig:235-254. True when `err` belongs
/// to gpu_driver.GpuError. The device-resident decode path uses this to
/// route any GPU-side failure into the host-bounce fallback.
fn isGpuFallbackError(err: anyerror) bool {
    comptime {
        const expected_members: usize = @typeInfo(gpu_driver.GpuError).error_set.?.len;
        // VK port: GpuError gained NotImplementedL2 + NotYetPorted on top
        // of CUDA's seven members.
        if (expected_members != 9) @compileError(
            "gpu_driver.GpuError member count changed; update the switch below " ++
            "and the assertion to match the new count.",
        );
    }
    return switch (err) {
        error.BadMode,
        error.BackendNotAvailable,
        error.OutOfDeviceMemory,
        error.KernelLaunchFailed,
        error.SyncFailed,
        error.CopyFailed,
        error.KernelMissing,
        error.NotImplementedL2,
        error.NotYetPorted,
        => true,
        else => false,
    };
}

// ── Codec cores — run on the worker thread ────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:258-269.
fn compressCore(h: *Context, input: []const u8, output: []u8, opts: CompressOpts) c_int {
    const n = encoder.compressFramed(
        allocator,
        input,
        output,
        .{ .level = @intCast(opts.level) },
        &h.enc,
    ) catch |err| return mapCompressError(err);
    return @intCast(n);
}

/// CUDA reference: src/streamlz_gpu.zig:271-282.
fn decompressCore(h: *Context, frame_bytes: []const u8, output: []u8, opts: DecompressOpts) c_int {
    h.dec.enable_profiling = opts.enable_profiling != 0;
    defer h.dec.enable_profiling = false;
    const r = decoder.decompressFramedThreaded(
        allocator,
        null,
        frame_bytes,
        output,
        &h.dec,
    ) catch |err| return mapDecompressError(err);
    return @intCast(r.written);
}

// ── Worker jobs ───────────────────────────────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:288-345.
const CompressJob = struct {
    h: *Context,
    src: []const u8,
    dst: []u8,
    opts: CompressOpts,
    d_src: u64 = 0,
    d_dst: u64 = 0,
    work_stream: usize = 0,
    result: c_int = SLZ_ERROR_CUDA,

    fn run(j: *CompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        if (j.d_src != 0) {
            j.h.enc.d_input_override = j.d_src;
        }
        if (j.d_dst != 0) {
            j.h.enc.d_output_override = j.d_dst;
            j.h.enc.output_written_to_device = false;
        }
        j.h.enc.enable_profiling = j.opts.enable_profiling != 0;
        j.h.enc.work_stream = j.work_stream;
        defer {
            j.h.enc.d_input_override = 0;
            j.h.enc.d_output_override = 0;
            j.h.enc.output_written_to_device = false;
            j.h.enc.work_stream = 0;
            if (j.work_stream == 0)
                gpu_driver.finalizeProfiling(&j.h.enc.pending_timings, &j.h.enc.last_timings);
            j.h.enc.enable_profiling = false;
        }
        const rc = compressCore(j.h, j.src, j.dst, j.opts);
        if (rc < 0) {
            j.result = rc;
            return;
        }
        if (j.d_dst != 0 and !j.h.enc.output_written_to_device) {
            if (!gpu_driver.copyHostToDevice(j.d_dst, j.dst[0..@intCast(rc)])) {
                j.result = SLZ_ERROR_CUDA;
                return;
            }
        }
        j.result = rc;
    }
};

/// CUDA reference: src/streamlz_gpu.zig:347-361.
const DecompressJob = struct {
    h: *Context,
    src: []const u8,
    dst: []u8,
    opts: DecompressOpts = .{},
    result: c_int = SLZ_ERROR_CUDA,

    fn run(j: *DecompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        j.result = decompressCore(j.h, j.src, j.dst, j.opts);
    }
};

/// CUDA reference: src/streamlz_gpu.zig:368-413. TRUE D2D decompress
/// worker.
const DecompressJobTrueD2D = struct {
    h: *Context,
    d_src: u64,
    src_size: u32,
    d_dst: u64,
    opts: DecompressOpts = .{},
    work_stream: usize = 0,
    decomp_size: u32 = 0,
    result: c_int = SLZ_ERROR_CUDA,
    fall_back: bool = false,

    fn run(j: *DecompressJobTrueD2D) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        j.h.dec.enable_profiling = j.opts.enable_profiling != 0;
        j.h.dec.work_stream = j.work_stream;
        defer {
            j.h.dec.enable_profiling = false;
            j.h.dec.work_stream = 0;
        }
        const n = decoder.decompressFramedFromDevice(
            null,
            j.d_src,
            j.src_size,
            j.d_dst,
            &j.h.dec,
            j.decomp_size,
        ) catch |err| {
            if (isGpuFallbackError(err)) {
                j.fall_back = true;
                j.result = SLZ_ERROR_CUDA;
            } else {
                j.result = mapDecompressError(err);
            }
            return;
        };
        j.result = @intCast(n);
    }
};

/// CUDA reference: src/streamlz_gpu.zig:418-423. Run `job` on a fresh
/// worker thread with a large stack, blocking until it finishes.
fn runOnWorker(comptime Job: type, job: *Job) c_int {
    const t = std.Thread.spawn(.{ .stack_size = worker_stack_size }, Job.run, .{job}) catch
        return SLZ_ERROR_OUT_OF_MEMORY;
    t.join();
    return job.result;
}

// ── Diagnostics ───────────────────────────────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:426-440.
export fn slzStatusString(status: c_int) [*:0]const u8 {
    return switch (status) {
        SLZ_SUCCESS => "success",
        SLZ_ERROR_INVALID_HANDLE => "invalid handle",
        SLZ_ERROR_INVALID_ARG => "invalid argument",
        SLZ_ERROR_BUFFER_TOO_SMALL => "output or temp buffer too small",
        SLZ_ERROR_CORRUPT_FRAME => "corrupt or unrecognized frame",
        SLZ_ERROR_UNSUPPORTED => "unsupported level or option",
        SLZ_ERROR_CUDA => "underlying CUDA call failed",
        SLZ_ERROR_OUT_OF_MEMORY => "out of memory",
        SLZ_ERROR_DEVICE_LOST => "device lost",
        SLZ_ERROR_VK_FEATURE_MISSING => "required Vulkan feature missing",
        else => "unknown status",
    };
}

/// CUDA reference: src/streamlz_gpu.zig:444-446.
export fn slzVersionString() [*:0]const u8 {
    return version.string;
}

// ── Handle lifecycle ──────────────────────────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:449-457.
export fn slzCreate(out_handle: ?*?*Context) c_int {
    const slot = out_handle orelse return SLZ_ERROR_INVALID_ARG;
    if (!gpu_encoder.isAvailable()) return SLZ_ERROR_CUDA;
    const ctx = allocator.create(Context) catch return SLZ_ERROR_OUT_OF_MEMORY;
    ctx.* = .{};
    slot.* = ctx;
    return SLZ_SUCCESS;
}

/// CUDA reference: src/streamlz_gpu.zig:459-469.
export fn slzDestroy(handle: ?*Context) c_int {
    if (handle) |h| {
        h.enc.deinit(allocator);
        h.dec.deinit();
        allocator.destroy(h);
    }
    return SLZ_SUCCESS;
}

// ── Options ───────────────────────────────────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:472-474.
export fn slzCompressDefaultOpts() CompressOpts {
    return .{};
}

/// CUDA reference: src/streamlz_gpu.zig:476-478.
export fn slzDecompressDefaultOpts() DecompressOpts {
    return .{};
}

// ── Size queries ──────────────────────────────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:489-500.
export fn slzCompressBound(
    handle: ?*Context,
    input_size: usize,
    opts: CompressOpts,
    max_output_size: ?*usize,
) c_int {
    _ = opts;
    _ = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const out = max_output_size orelse return SLZ_ERROR_INVALID_ARG;
    out.* = encoder.compressBound(input_size);
    return SLZ_SUCCESS;
}

/// CUDA reference: src/streamlz_gpu.zig:508-527.
export fn slzGetDecompressedSize(
    handle: ?*Context,
    host_bytes: ?*const anyopaque,
    decompressed_size: ?*usize,
) c_int {
    _ = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const hdr_ptr = host_bytes orelse return SLZ_ERROR_INVALID_ARG;
    const out = decompressed_size orelse return SLZ_ERROR_INVALID_ARG;
    const bytes: [*]const u8 = @ptrCast(hdr_ptr);
    const parsed = frame.parseHeader(bytes[0..64]) catch return SLZ_ERROR_CORRUPT_FRAME;
    const size = parsed.content_size orelse return SLZ_ERROR_CORRUPT_FRAME;
    if (size > decoder.max_content_size) return SLZ_ERROR_CORRUPT_FRAME;
    out.* = size;
    return SLZ_SUCCESS;
}

// ── Compress / decompress: host -> host ───────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:530-556.
export fn slzCompressHost(
    handle: ?*Context,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
    opts: CompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const in_ptr = input orelse return SLZ_ERROR_INVALID_ARG;
    const out_ptr = output orelse return SLZ_ERROR_INVALID_ARG;
    const out_size = output_size orelse return SLZ_ERROR_INVALID_ARG;
    const in: [*]const u8 = @ptrCast(in_ptr);
    const out: [*]u8 = @ptrCast(out_ptr);

    var job = CompressJob{
        .h = h,
        .src = in[0..input_size],
        .dst = out[0..output_capacity],
        .opts = opts,
    };
    const rc = runOnWorker(CompressJob, &job);
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

/// CUDA reference: src/streamlz_gpu.zig:558-584.
export fn slzDecompressHost(
    handle: ?*Context,
    frame_in: ?*const anyopaque,
    frame_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
    opts: DecompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const frame_ptr = frame_in orelse return SLZ_ERROR_INVALID_ARG;
    const out_ptr = output orelse return SLZ_ERROR_INVALID_ARG;
    const out_size = output_size orelse return SLZ_ERROR_INVALID_ARG;
    const fr: [*]const u8 = @ptrCast(frame_ptr);
    const out: [*]u8 = @ptrCast(out_ptr);

    var job = DecompressJob{
        .h = h,
        .src = fr[0..frame_size],
        .dst = out[0..output_capacity],
        .opts = opts,
    };
    const rc = runOnWorker(DecompressJob, &job);
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

// ── Async (stream-taking) device-to-device entry points ─────────────
/// CUDA reference: src/streamlz_gpu.zig:590-649.
export fn slzCompressAsync(
    handle: ?*Context,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    max_compressed_size: usize,
    compressed_size: ?*usize,
    opts: CompressOpts,
    stream: ?*anyopaque,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const in_dev = d_input orelse return SLZ_ERROR_INVALID_ARG;
    const out_dev = d_output orelse return SLZ_ERROR_INVALID_ARG;
    const out_size = compressed_size orelse return SLZ_ERROR_INVALID_ARG;

    const async_min_input: usize = 128 + 1;
    const async_max_input: usize = @as(usize, 16384) * 65536;
    if (input_size < async_min_input or input_size > async_max_input)
        return SLZ_ERROR_UNSUPPORTED;

    const host_out = allocator.alloc(u8, max_compressed_size) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_out);

    var job = CompressJob{
        .h = h,
        .src = deviceOnlySrcStub(input_size),
        .dst = host_out,
        .opts = opts,
        .d_src = @intFromPtr(in_dev),
        .d_dst = @intFromPtr(out_dev),
        .work_stream = @intFromPtr(stream),
    };
    CompressJob.run(&job);
    const rc = job.result;
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

/// CUDA reference: src/streamlz_gpu.zig:651-692.
export fn slzDecompressAsync(
    handle: ?*Context,
    d_frame: ?*const anyopaque,
    frame_size: usize,
    d_output: ?*anyopaque,
    output_size: usize,
    opts: DecompressOpts,
    stream: ?*anyopaque,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const frame_dev = d_frame orelse return SLZ_ERROR_INVALID_ARG;
    const out_dev = d_output orelse return SLZ_ERROR_INVALID_ARG;

    var true_job = DecompressJobTrueD2D{
        .h = h,
        .d_src = @intFromPtr(frame_dev),
        .src_size = @intCast(frame_size),
        .d_dst = @intFromPtr(out_dev),
        .opts = opts,
        .work_stream = @intFromPtr(stream),
        .decomp_size = @intCast(output_size),
    };
    DecompressJobTrueD2D.run(&true_job);
    const rc = true_job.result;
    if (rc >= 0) return SLZ_SUCCESS;
    if (true_job.fall_back) return SLZ_ERROR_UNSUPPORTED;
    return rc;
}

// ── Profiling ─────────────────────────────────────────────────────────
/// CUDA reference: src/streamlz_gpu.zig:699-732.
export fn slzGetLastTimings(
    handle: ?*Context,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const cnt = count_out orelse return SLZ_ERROR_INVALID_ARG;
    gpu_driver.finalizeProfiling(&h.enc.pending_timings, &h.enc.last_timings);
    gpu_driver.finalizeProfiling(&h.dec.pending_timings, &h.dec.last_timings);
    const enc_list = h.enc.last_timings.items;
    const dec_list = h.dec.last_timings.items;
    const total = enc_list.len + dec_list.len;
    cnt.* = total;
    if (timings) |t| {
        var i: usize = 0;
        for (enc_list) |kt| {
            if (i >= capacity) break;
            t[i] = .{ .name = kt.name, .ms = kt.ms };
            i += 1;
        }
        for (dec_list) |kt| {
            if (i >= capacity) break;
            t[i] = .{ .name = kt.name, .ms = kt.ms };
            i += 1;
        }
    }
    return SLZ_SUCCESS;
}

/// CUDA reference: src/streamlz_gpu.zig:742-754.
export fn slzWaitAndGetLastTimings(
    handle: ?*Context,
    stream: ?*anyopaque,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    if (stream) |s| {
        // VK adaptation: stream sync via vk_api.procs.stream_sync (CUDA
        // path uses cuda_api.cuStreamSync_fn).
        const sync_fn = vk_api.procs.stream_sync orelse return SLZ_ERROR_CUDA;
        if (sync_fn(@intFromPtr(s)) != vk_api.VK_SUCCESS_RC) return SLZ_ERROR_CUDA;
    }
    return slzGetLastTimings(handle, timings, capacity, count_out);
}
