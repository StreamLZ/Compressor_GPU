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
    enable_profiling: c_int = 0,
    reserved: [6]c_int = .{0} ** 6,
};

// ── slzDecompressOpts_t — must match the header struct layout ─────────
const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = .{0} ** 7,
};

// ── slzKernelTiming_t — must match the header struct layout ───────────
const KernelTimingC = extern struct {
    name: [*:0]const u8,
    ms: f32,
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

/// Returns true iff `err` is one of `gpu_driver.GpuError`'s members.
/// Used by both GPU C-ABI entry points to map any GpuError uniformly
/// to "host fallback" (SLZ_ERROR_CUDA / -1) rather than misclassify a
/// CUDA failure as a corrupt frame via `mapDecompressError`.
///
/// The comptime block locks the listed members against
/// `descriptors.zig:GpuError`: adding a new member there without
/// updating the switch below is a compile error, so silent drift is
/// impossible.
fn isGpuFallbackError(err: anyerror) bool {
    comptime {
        const expected_members: usize = @typeInfo(gpu_driver.GpuError).error_set.?.len;
        if (expected_members != 7) @compileError(
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
        => true,
        else => false,
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

fn decompressCore(h: *Context, frame_bytes: []const u8, output: []u8, opts: DecompressOpts) c_int {
    h.dec.enable_profiling = opts.enable_profiling != 0;
    defer h.dec.enable_profiling = false;
    const r = decoder.decompressFramedParallelThreaded(
        allocator,
        null,
        frame_bytes,
        output,
        0,
        &h.dec,
        true, // C ABI slzDecompress: always allow GPU when compiled in
    ) catch |err| return mapDecompressError(err);
    if (r.offset > 0 and r.written > 0) {
        std.mem.copyForwards(u8, output[0..r.written], output[r.offset..][0..r.written]);
    }
    return @intCast(r.written);
}

/// 4d Phase 3 device-resident decode: the decoded bytes are D2D-copied
/// straight into the caller's device output buffer (no host bounce on
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
    /// Caller-supplied CUstream for the async variant; 0 = default stream.
    /// When non-zero, the encode driver leaves the final frame-assembly
    /// kernel queued on this stream rather than blocking on it.
    work_stream: usize = 0,
    result: c_int = SLZ_ERROR_CUDA,

    fn run(j: *CompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = SLZ_ERROR_CUDA;
            return;
        }
        // 4d Phase 3 encode-input D2D: the encoder's CPU paths never
        // dereference `input` for -gpu mode (the GPU LZ kernel + the M2
        // assembly kernel handle every byte access), so we leave the
        // host `src` slice's bytes uninitialized. gpuCompressImpl reads
        // the data from d_input_override (D2D copy from the caller's
        // device input) instead of H2D-ing the host bounce.
        if (j.d_src != 0) {
            j.h.enc.d_input_override = j.d_src;
        }
        // 4d step 8: when the caller asked for a device output AND the
        // encode hits the slzFrameAssembleKernel path, the kernel writes
        // the frame straight to d_dst — no host bounce. The fallback
        // host->device H2D below stays as the safety net for paths that
        // didn't take the pure-D2D branch.
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
            // Drain cuEvent pairs → last_timings (sync mode only — async
            // leaves events pending until the user syncs + calls
            // slzGetLastTimings).
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

/// 4d Phase 3 TRUE D2D decompress worker: frame stays on device, walk
/// runs on GPU, output stays on device — zero host bounce of payload
/// or frame bytes. On -1 (slzCompress-shape constraint failed, e.g.
/// dict / PDM / multi-block frame) the caller falls back to the host-
/// frame D2D path; on -2 to the full host-bounce path.
const DecompressJobTrueD2D = struct {
    h: *Context,
    d_src: u64,
    src_size: u32,
    d_dst: u64,
    opts: DecompressOpts = .{},
    /// Caller-supplied CUstream for the async variant; 0 = default stream.
    /// When non-zero, the decode driver queues huff + LZ + output D2D on
    /// this stream and skips its final sync — the caller is the sync.
    work_stream: usize = 0,
    /// Caller-supplied decompressed byte count, required by the v3 ABI
    /// (slzGetDecompressedSize gives it once per archive).
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
    return "3.0.0";
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

export fn slzDecompressDefaultOpts() DecompressOpts {
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

/// v3 contract: host_bytes points to >= 64 bytes of valid StreamLZ frame
/// (or the whole frame if shorter — every header we produce is well
/// under 64 bytes, and every realistic caller has the whole frame in
/// host RAM). The library reads only what its parser needs; if the
/// frame is shorter than the header field at hand, parseHeader returns
/// Truncated and we map to SLZ_ERROR_CORRUPT_FRAME.
export fn slzGetDecompressedSize(
    handle: ?*Context,
    host_bytes: ?*const anyopaque,
    decompressed_size: ?*usize,
) c_int {
    if (handle == null) return SLZ_ERROR_INVALID_HANDLE;
    const hdr_ptr = host_bytes orelse return SLZ_ERROR_INVALID_ARG;
    const out = decompressed_size orelse return SLZ_ERROR_INVALID_ARG;
    const bytes: [*]const u8 = @ptrCast(hdr_ptr);
    // 64-byte slice is safely longer than any header we produce (max ~26 B)
    // and the caller contract requires that many bytes be readable.
    const parsed = frame.parseHeader(bytes[0..64]) catch return SLZ_ERROR_CORRUPT_FRAME;
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
// Mirror nvCOMP's pattern: submit on `stream`, return; caller is the
// sync. NULL stream is the default stream and makes these behave like
// the sync entry points.
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

    const host_out = allocator.alloc(u8, max_compressed_size) catch return SLZ_ERROR_OUT_OF_MEMORY;
    defer allocator.free(host_out);

    const sentinel_ptr: [*]const u8 = @ptrFromInt(0x10);
    const src_stub: []const u8 = sentinel_ptr[0..input_size];

    var job = CompressJob{
        .h = h,
        .src = src_stub,
        .dst = host_out,
        .opts = opts,
        .d_src = @intFromPtr(in_dev),
        .d_dst = @intFromPtr(out_dev),
        .work_stream = @intFromPtr(stream),
    };
    // Run inline on the caller's thread (nvCOMP-style — no per-call
    // thread spawn). The host front half (setup + measure-pass D2H +
    // headers) blocks the caller; the heavy back-half kernels +
    // final D2D copy ride `stream` and complete after the caller's
    // cudaStreamSynchronize.
    CompressJob.run(&job);
    const rc = job.result;
    if (rc < 0) return rc;
    out_size.* = @intCast(rc);
    return SLZ_SUCCESS;
}

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
    // Run inline on the caller's thread (nvCOMP-style — no per-call
    // thread spawn). The host front half (walk + scan + prefix-sum +
    // compact + merge) blocks the caller; the heavy back-half Huff +
    // LZ kernels and the final D2D output copy ride `stream` and
    // complete after the caller's cudaStreamSynchronize.
    //
    // v3: caller already knows output_size (from slzGetDecompressedSize
    // or their own metadata) so we don't need to report it back. The
    // size param is used to validate d_output capacity inside the
    // pipeline; passing wrong value is undefined behaviour.
    DecompressJobTrueD2D.run(&true_job);
    const rc = true_job.result;
    if (rc >= 0) return SLZ_SUCCESS;
    // Async path doesn't fall back. If the TrueD2D shape isn't satisfied
    // (dict frame / PDM / multi-block), the caller should use the sync
    // slzDecompress entry point which handles fallback through the
    // half-D2D and full-host paths.
    if (true_job.fall_back) return SLZ_ERROR_UNSUPPORTED;
    return rc;
}

// ── Profiling ─────────────────────────────────────────────────────────
// Reads the per-kernel timings captured during the most recent compress
// or decompress call on this handle. Combines encode + decode timings
// (encode first if both ran, then decode). `timings` may be NULL — the
// call always writes *count and returns SLZ_SUCCESS.
export fn slzGetLastTimings(
    handle: ?*Context,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const cnt = count_out orelse return SLZ_ERROR_INVALID_ARG;
    // Drain any pending cuEvent pairs. Idempotent: no-op when pending is
    // empty (already finalized by sync wrapper or another call). For
    // slzCompressAsync/slzDecompressAsync callers this is where timings
    // materialize — they must have synced the user stream first.
    // Convenience: slzWaitAndGetLastTimings does the stream sync for them.
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

// Wait on the caller's CUstream and then drain per-kernel timings.
// Async convenience: avoids the caller's "cudaStreamSynchronize, then
// slzGetLastTimings" pair. Passing stream=NULL behaves like the
// synchronous slzGetLastTimings (no stream sync; assumes already
// synced).
const cuda_api = @import("gpu/decode/cuda_api.zig");
const CUDA_SUCCESS: c_int = 0;
export fn slzWaitAndGetLastTimings(
    handle: ?*Context,
    stream: ?*anyopaque,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    _ = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    _ = count_out orelse return SLZ_ERROR_INVALID_ARG;
    if (stream) |s| {
        const sync_fn = cuda_api.cuStreamSync_fn orelse return SLZ_ERROR_CUDA;
        if (sync_fn(@intFromPtr(s)) != CUDA_SUCCESS) return SLZ_ERROR_CUDA;
    }
    return slzGetLastTimings(handle, timings, capacity, count_out);
}
