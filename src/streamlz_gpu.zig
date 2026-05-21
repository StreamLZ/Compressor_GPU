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

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const gpu_encoder = @import("encode/fast/gpu_encoder.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_driver = @import("decode/fast/gpu_driver.zig");
const frame = @import("format/frame_format.zig");

const allocator = std.heap.c_allocator;

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

// ── Core host->host compress / decompress ─────────────────────────────
fn compressToHost(
    h: *Context,
    input: []const u8,
    output: []u8,
    opts: CompressOpts,
) c_int {
    if (opts.level < 1 or opts.level > 5) return SLZ_ERROR_UNSUPPORTED;
    const n = encoder.compressFramed(
        allocator,
        input,
        output,
        .{ .level = @intCast(opts.level), .gpu_mode = true },
        &h.enc,
    ) catch |err| return mapCompressError(err);
    return @intCast(n); // bytes written (>= 0); never exceeds output.len
}

fn decompressToHost(h: *Context, frame_bytes: []const u8, output: []u8) c_int {
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
    const rc = compressToHost(h, in[0..input_size], out[0..output_capacity], opts);
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
    const rc = decompressToHost(h, fr[0..frame_size], out[0..output_capacity]);
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

// ── Device->device compress / decompress ──────────────────────────────
// v1 bridges to the host pipeline through an internal host bounce: the
// device input is copied down, processed on the host pipeline (which
// itself drives the GPU), and the result is copied back up. Functionally
// device->device for the caller; a later revision keeps it device-resident.
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

    if (!gpu_driver.copyDeviceToHost(host_in, @intFromPtr(in_dev))) return SLZ_ERROR_CUDA;
    const rc = compressToHost(h, host_in, host_out, opts);
    if (rc < 0) return rc;
    const n: usize = @intCast(rc);
    if (!gpu_driver.copyHostToDevice(@intFromPtr(out_dev), host_out[0..n])) return SLZ_ERROR_CUDA;
    out_size.* = n;
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
    const host_out = allocator.alloc(u8, output_capacity) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_out);

    if (!gpu_driver.copyDeviceToHost(host_frame, @intFromPtr(frame_dev))) return SLZ_ERROR_CUDA;
    const rc = decompressToHost(h, host_frame, host_out);
    if (rc < 0) return rc;
    const n: usize = @intCast(rc);
    if (!gpu_driver.copyHostToDevice(@intFromPtr(out_dev), host_out[0..n])) return SLZ_ERROR_CUDA;
    out_size.* = n;
    return SLZ_SUCCESS;
}
