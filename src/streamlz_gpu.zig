//! StreamLZ GPU compression library — C ABI implementation.
//!
//! Implements the contract in `include/streamlz_gpu.h`: a handle-based,
//! synchronous, GPU-backed compressor exposed to C callers. Each handle
//! owns a private `EncodeContext` + `DecodeContext`, so distinct handles
//! never share mutable GPU state.
//!
//! Buffer models:
//!   - host->host  (slzCompressHost / slzDecompressHost): plain host
//!     buffers; the underlying pipeline does its own H2D/D2H.
//!   - device->device (slzCompress / slzDecompress): the caller's data is
//!     GPU-resident. v1 bridges to the host pipeline via an internal
//!     host bounce; a later revision moves this fully device-resident.
//!
//! Every compress/decompress runs on a library-owned worker thread (a
//! generous stack, with the CUDA context bound) so the call is safe from
//! any caller thread regardless of that thread's stack size.

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const gpu_encoder = @import("gpu/encode/driver.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_driver = @import("gpu/decode/driver.zig");
const frame = @import("format/frame_format.zig");

const allocator = std.heap.c_allocator;

/// Worker-thread stack. The compress orchestration has multi-MB stack
/// frames; this keeps callers safe even on small-stack threads.
const worker_stack_size: usize = 32 << 20;

// ── Status codes — must match streamlz_gpu.h slzStatus_t ──────────────
const SLZ_SUCCESS: c_int = 0;
const SLZ_ERROR_INVALID_HANDLE: c_int = 1;
const SLZ_ERROR_INVALID_ARG: c_int = 2;
const SLZ_ERROR_BUFFER_TOO_SMALL: c_int = 3;
const SLZ_ERROR_CORRUPT_FRAME: c_int = 4;
const SLZ_ERROR_UNSUPPORTED: c_int = 5;
const SLZ_ERROR_CUDA: c_int = 6;
const SLZ_ERROR_OUT_OF_MEMORY: c_int = 7;

// ── slzCompressOpts_t — must match the header struct layout ───────────
const CompressOpts = extern struct {
    level: c_int = 5,
    reserved: [7]c_int = .{0} ** 7,
};

/// Library handle. Owns the per-operation GPU contexts so concurrent
/// handles cannot corrupt one another.
const Context = struct {
    enc: gpu_encoder.EncodeContext = .{},
    dec: gpu_driver.DecodeContext = .{},
};

fn mapCompressError(err: anyerror) c_int {
    return switch (err) {
        error.DestinationTooSmall => SLZ_ERROR_BUFFER_TOO_SMALL,
        error.OutOfMemory => SLZ_ERROR_OUT_OF_MEMORY,
        error.BadLevel, error.BadScGroupSize => SLZ_ERROR_UNSUPPORTED,
        else => SLZ_ERROR_CUDA,
    };
}

fn mapDecompressError(err: anyerror) c_int {
    return switch (err) {
        error.OutputTooSmall => SLZ_ERROR_BUFFER_TOO_SMALL,
        error.OutOfMemory => SLZ_ERROR_OUT_OF_MEMORY,
        else => SLZ_ERROR_CORRUPT_FRAME,
    };
}

// ── Codec cores — run on the worker thread ────────────────────────────
// Each returns the byte count (>= 0) or a negative SLZ_ERROR_* code.
fn compressCore(h: *Context, input: []const u8, output: []u8, opts: CompressOpts) c_int {
    if (opts.level < 1 or opts.level > 5) return SLZ_ERROR_UNSUPPORTED;
    const n = encoder.compressFramed(
        allocator,
        input,
        output,
        .{ .level = @intCast(opts.level), .gpu_mode = true },
        &h.enc,
    ) catch |err| return mapCompressError(err);
    return @intCast(n);
}

fn decompressCore(h: *Context, frame_bytes: []const u8, output: []u8) c_int {
    const r = decoder.decompressFramedParallelThreaded(
        allocator,
        null,
        frame_bytes,
        output,
        0,
        &h.dec,
    ) catch |err| return mapDecompressError(err);
    if (r.offset > 0 and r.written > 0) {
        std.mem.copyForwards(u8, output[0..r.written], output[r.offset..][0..r.written]);
    }
    return @intCast(r.written);
}

/// 4d Phase 3 device-resident decode: the decoded bytes are D2D-copied
/// straight into the caller's device output buffer (no host bounce on
/// the output). The CPU still needs the frame bytes (host-readable) to
/// walk frame/block/chunk headers — full GPU-side header walk is a
/// follow-up. Returns -1 on the GPU-only contract failing so the caller
/// can fall back to the host-bounce path.
fn decompressCoreD2D(h: *Context, frame_bytes: []const u8, d_output: u64) c_int {
    const r = decoder.decompressFramedParallelToDevice(
        allocator,
        null,
        frame_bytes,
        d_output,
        0,
        &h.dec,
    ) catch |err| return switch (err) {
        error.BadMode => -1, // signal "fall back"
        else => mapDecompressError(err),
    };
    // For a frame with a dictionary prefix, the decoded content sits at
    // d_output[r.offset..r.offset+r.written]. The library API contract
    // returns the *content* size, so callers expect the bytes at
    // d_output[0..]. Dictionary frames aren't supported on the D2D path
    // for now — return -1 to fall back.
    if (r.offset > 0) return -1;
    return @intCast(r.written);
}

// ── Worker jobs ───────────────────────────────────────────────────────
// `d_src`/`d_dst` are 0 for the host->host path, or device addresses for
// the device->device path (then `src` aliases an internal bounce buffer
// that the worker fills via D2H before the core runs).
const CompressJob = struct {
    h: *Context,
    src: []const u8,
    dst: []u8,
    opts: CompressOpts,
    d_src: u64 = 0,
    d_dst: u64 = 0,
    result: c_int = SLZ_ERROR_CUDA,

    fn run(j: *CompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        if (j.d_src != 0 and !gpu_driver.copyDeviceToHost(@constCast(j.src), j.d_src)) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        const rc = compressCore(j.h, j.src, j.dst, j.opts);
        if (rc < 0) {
            j.result = rc;
            return;
        }
        if (j.d_dst != 0 and !gpu_driver.copyHostToDevice(j.d_dst, j.dst[0..@intCast(rc)])) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        j.result = rc;
    }
};

const DecompressJob = struct {
    h: *Context,
    src: []const u8,
    dst: []u8,
    d_src: u64 = 0,
    d_dst: u64 = 0,
    result: c_int = SLZ_ERROR_CUDA,

    fn run(j: *DecompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        if (j.d_src != 0 and !gpu_driver.copyDeviceToHost(@constCast(j.src), j.d_src)) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        const rc = decompressCore(j.h, j.src, j.dst);
        if (rc < 0) {
            j.result = rc;
            return;
        }
        if (j.d_dst != 0 and !gpu_driver.copyHostToDevice(j.d_dst, j.dst[0..@intCast(rc)])) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        j.result = rc;
    }
};

/// 4d Phase 3 D2D decompress worker: D2H the frame into a host scratch
/// (CPU header walk needs it), then decompress directly to the caller's
/// device output via decompressCoreD2D (D2D output, no host bounce).
/// On a -1 return the caller falls back to `DecompressJob`'s host path.
const DecompressJobD2D = struct {
    h: *Context,
    host_frame: []u8,
    d_src: u64,
    d_dst: u64,
    result: c_int = SLZ_ERROR_CUDA,
    fall_back: bool = false,

    fn run(j: *DecompressJobD2D) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        if (!gpu_driver.copyDeviceToHost(j.host_frame, j.d_src)) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        const rc = decompressCoreD2D(j.h, j.host_frame, j.d_dst);
        if (rc == -1) {
            j.fall_back = true;
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        j.result = rc;
    }
};

/// Run `job` on a fresh worker thread with a large stack, blocking until
/// it finishes. Returns the job's `result` (byte count >= 0, or negative
/// SLZ_ERROR_*); SLZ_ERROR_OUT_OF_MEMORY if the thread cannot be spawned.
fn runOnWorker(comptime Job: type, job: *Job) c_int {
    const t = std.Thread.spawn(.{ .stack_size = worker_stack_size }, Job.run, .{job}) catch
        return SLZ_ERROR_OUT_OF_MEMORY;
    t.join();
    return job.result;
}

// ── Diagnostics ───────────────────────────────────────────────────────
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
        else => "unknown status",
    };
}

export fn slzVersionString() [*:0]const u8 {
    return "2.0.0";
}

// ── Handle lifecycle ──────────────────────────────────────────────────
export fn slzCreate(out_handle: ?*?*Context) c_int {
    const slot = out_handle orelse return SLZ_ERROR_INVALID_ARG;
    // The library is GPU-backed — fail fast if no usable GPU.
    if (!gpu_encoder.isAvailable()) return SLZ_ERROR_CUDA;
    const ctx = allocator.create(Context) catch return SLZ_ERROR_OUT_OF_MEMORY;
    ctx.* = .{};
    slot.* = ctx;
    return SLZ_SUCCESS;
}

export fn slzDestroy(handle: ?*Context) c_int {
    if (handle) |h| allocator.destroy(h);
    return SLZ_SUCCESS;
}

// ── Options ───────────────────────────────────────────────────────────
export fn slzCompressDefaultOpts() CompressOpts {
    return .{};
}

// ── Size queries ──────────────────────────────────────────────────────
export fn slzCompressBound(
    handle: ?*Context,
    input_size: usize,
    opts: CompressOpts,
    max_output_size: ?*usize,
) c_int {
    _ = opts;
    if (handle == null) return SLZ_ERROR_INVALID_HANDLE;
    const out = max_output_size orelse return SLZ_ERROR_INVALID_ARG;
    out.* = encoder.compressBound(input_size);
    return SLZ_SUCCESS;
}

// v1 manages all device scratch internally, so callers need no temp
// buffer for either path. Both queries report zero.
export fn slzCompressGetTempSize(
    handle: ?*Context,
    input_size: usize,
    opts: CompressOpts,
    temp_size: ?*usize,
) c_int {
    _ = input_size;
    _ = opts;
    if (handle == null) return SLZ_ERROR_INVALID_HANDLE;
    const out = temp_size orelse return SLZ_ERROR_INVALID_ARG;
    out.* = 0;
    return SLZ_SUCCESS;
}

export fn slzDecompressGetTempSize(
    handle: ?*Context,
    frame_size: usize,
    temp_size: ?*usize,
) c_int {
    _ = frame_size;
    if (handle == null) return SLZ_ERROR_INVALID_HANDLE;
    const out = temp_size orelse return SLZ_ERROR_INVALID_ARG;
    out.* = 0;
    return SLZ_SUCCESS;
}

export fn slzGetDecompressedSize(
    handle: ?*Context,
    frame_header: ?*const anyopaque,
    header_len: usize,
    decompressed_size: ?*usize,
) c_int {
    if (handle == null) return SLZ_ERROR_INVALID_HANDLE;
    const hdr_ptr = frame_header orelse return SLZ_ERROR_INVALID_ARG;
    const out = decompressed_size orelse return SLZ_ERROR_INVALID_ARG;
    const bytes: [*]const u8 = @ptrCast(hdr_ptr);
    const parsed = frame.parseHeader(bytes[0..header_len]) catch return SLZ_ERROR_CORRUPT_FRAME;
    out.* = parsed.content_size orelse return SLZ_ERROR_CORRUPT_FRAME;
    return SLZ_SUCCESS;
}

// ── Compress / decompress: host -> host ───────────────────────────────
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

export fn slzDecompressHost(
    handle: ?*Context,
    frame_in: ?*const anyopaque,
    frame_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
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
    };
    const rc = runOnWorker(DecompressJob, &job);
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

// ── Compress / decompress: device -> device ───────────────────────────
// v1 bridges to the host pipeline through an internal host bounce: the
// worker thread copies the device input down, runs the host pipeline
// (which itself drives the GPU), and copies the result back up.
// Functionally device->device for the caller; a later revision keeps it
// device-resident throughout.
export fn slzCompress(
    handle: ?*Context,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_temp: ?*anyopaque,
    temp_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
    opts: CompressOpts,
) c_int {
    _ = d_temp;
    _ = temp_size;
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const in_dev = d_input orelse return SLZ_ERROR_INVALID_ARG;
    const out_dev = d_output orelse return SLZ_ERROR_INVALID_ARG;
    const out_size = output_size orelse return SLZ_ERROR_INVALID_ARG;

    const host_in = allocator.alloc(u8, input_size) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_in);
    const host_out = allocator.alloc(u8, output_capacity) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_out);

    var job = CompressJob{
        .h = h,
        .src = host_in,
        .dst = host_out,
        .opts = opts,
        .d_src = @intFromPtr(in_dev),
        .d_dst = @intFromPtr(out_dev),
    };
    const rc = runOnWorker(CompressJob, &job);
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

export fn slzDecompress(
    handle: ?*Context,
    d_frame: ?*const anyopaque,
    frame_size: usize,
    d_temp: ?*anyopaque,
    temp_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
) c_int {
    _ = d_temp;
    _ = temp_size;
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const frame_dev = d_frame orelse return SLZ_ERROR_INVALID_ARG;
    const out_dev = d_output orelse return SLZ_ERROR_INVALID_ARG;
    const out_size = output_size orelse return SLZ_ERROR_INVALID_ARG;

    const host_frame = allocator.alloc(u8, frame_size) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_frame);

    // 4d Phase 3 D2D path: D2H the frame to host (CPU header walk needs
    // it), then decompress directly to the caller's device output — no
    // host output bounce. Falls back to the legacy host-bounce path if
    // the frame requires a CPU-only block decoder or carries a dict.
    var d2d_job = DecompressJobD2D{
        .h = h,
        .host_frame = host_frame,
        .d_src = @intFromPtr(frame_dev),
        .d_dst = @intFromPtr(out_dev),
    };
    const rc_d2d = runOnWorker(DecompressJobD2D, &d2d_job);
    if (rc_d2d >= 0) {
        out_size.* = @intCast(rc_d2d);
        return SLZ_SUCCESS;
    }
    if (!d2d_job.fall_back) return rc_d2d;

    // Fallback: host-bounce output. Reuses the already-D2H'd host_frame.
    const host_out = allocator.alloc(u8, output_capacity) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_out);

    var job = DecompressJob{
        .h = h,
        .src = host_frame,
        .dst = host_out,
        .d_src = 0, // host_frame already populated by the D2D try
        .d_dst = @intFromPtr(out_dev),
    };
    const rc = runOnWorker(DecompressJob, &job);
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}
