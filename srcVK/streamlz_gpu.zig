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
const dec_module_loader = @import("decode/module_loader.zig");

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

/// CUDA reference: src/streamlz_gpu.zig:105-114. Mirror of slzCompressOpts_t.
const CompressOpts = extern struct {
    level: c_int = 5,
    enable_profiling: c_int = 0,
    effective_level_out: c_int = 0,
    /// v4 #16: preset dictionary ID (0 = none). Reuses the next
    /// `reserved` slot, so pre-dict callers (who must zero reserved)
    /// are automatically dictionary-less; struct size unchanged.
    dictionary_id: c_int = 0,
    reserved: [4]c_int = @splat(0),
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
    std.debug.assert(@offsetOf(CompressOpts, "dictionary_id") == 3 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "reserved") == 4 * @sizeOf(c_int));
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
// CUDA reference: src/streamlz_gpu.zig:263-280. Each returns the byte
// count (>= 0) or a NEGATED SLZ_ERROR_* code - the public constants
// are positive, so a positive error would be misread as a tiny
// successful output (the exact latent bug the CUDA v4 #16 dict test
// exposed and fixed; mirrored here). Entry points negate back before
// returning a status to the C caller.
fn compressCore(h: *Context, input: []const u8, output: []u8, opts: CompressOpts) c_int {
    // Level range is validated by `encoder.compressFramed` (returns
    // `error.BadLevel` -> SLZ_ERROR_UNSUPPORTED via mapCompressError);
    // an unresolvable dictionary_id likewise (error.UnknownDictionary).
    const n = encoder.compressFramed(
        allocator,
        input,
        output,
        .{
            .level = @intCast(opts.level),
            .dictionary_id = if (opts.dictionary_id != 0) @intCast(opts.dictionary_id) else null,
        },
        &h.enc,
    ) catch |err| return -mapCompressError(err);
    return @intCast(n);
}

/// CUDA reference: src/streamlz_gpu.zig:271-282.
///
/// VK adaptation (F-001 fix, 2026-06-08): Mirror of
/// `srcVK/cli/decompress.zig:115,120` — the decoder's H2D fast path
/// imports the caller's `frame_bytes` and the D2H fast path imports
/// the caller's `output` into the LRU `g_import_cache` keyed by
/// `(host_addr, size, usage_src)`. Without an explicit release on the
/// way OUT, a subsequent decode whose caller-supplied slice happens to
/// land at the same virtual address (allocator reuse is common with
/// the same-sized alloc pattern) would hit a stale cache entry whose
/// `VkDeviceMemory` still imports the OLD physical pages — GPU writes
/// would go to the stale mapping and the caller's buffer would never
/// be touched. The CLI handled this by calling
/// `releaseImportsByHostRange` after every decode (see iter-8 subfix
/// 1 commentary at module_loader.zig:4272). The C ABI must do the
/// same. We release on success AND on error (the import was already
/// created by the time the error fired) so the cache invariant holds
/// regardless of decode outcome.
///
/// Release BEFORE the decode too (belt-and-suspenders): if the caller
/// somehow re-used a `frame_bytes` / `output` address whose prior
/// import entry survived an earlier decode that bypassed this path
/// (e.g. an older library version, a foreign caller), we drop the
/// stale entry up front and force a fresh import on first H2D / D2H.
fn decompressCore(h: *Context, frame_bytes: []const u8, output: []u8, opts: DecompressOpts) c_int {
    h.dec.enable_profiling = opts.enable_profiling != 0;
    // Pre-release: drop any prior cache entries that reference the
    // caller's input or output buffers. Cheap (linear walk over a
    // fixed-size LRU array).
    if (frame_bytes.len > 0) dec_module_loader.releaseImportsByHostRange(@ptrCast(frame_bytes.ptr), frame_bytes.len);
    if (output.len > 0) dec_module_loader.releaseImportsByHostRange(@ptrCast(output.ptr), output.len);
    defer {
        h.dec.enable_profiling = false;
        // Post-release: drop the entries the decoder just created so
        // the next decode against the same addresses imports fresh
        // physical pages.
        if (frame_bytes.len > 0) dec_module_loader.releaseImportsByHostRange(@ptrCast(frame_bytes.ptr), frame_bytes.len);
        if (output.len > 0) dec_module_loader.releaseImportsByHostRange(@ptrCast(output.ptr), output.len);
    }
    const r = decoder.decompressFramedThreaded(
        allocator,
        null,
        frame_bytes,
        output,
        &h.dec,
    ) catch |err| return -mapDecompressError(err);
    return @intCast(r.written);
}

// ── Worker jobs ───────────────────────────────────────────────────────
// CUDA reference: src/streamlz_gpu.zig:288-345. Job-result protocol:
// `result` >= 0 is a byte count; errors are NEGATED SLZ_ERROR_* codes
// (see the codec-core comment above).
const CompressJob = struct {
    h: *Context,
    src: []const u8,
    dst: []u8,
    opts: CompressOpts,
    d_src: u64 = 0,
    d_dst: u64 = 0,
    work_stream: usize = 0,
    result: c_int = -SLZ_ERROR_CUDA,

    fn run(j: *CompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = -SLZ_ERROR_CUDA;
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
                j.result = -SLZ_ERROR_CUDA;
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
    result: c_int = -SLZ_ERROR_CUDA,

    fn run(j: *DecompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = -SLZ_ERROR_CUDA;
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
    result: c_int = -SLZ_ERROR_CUDA,
    fall_back: bool = false,

    fn run(j: *DecompressJobTrueD2D) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = -SLZ_ERROR_CUDA;
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
                j.result = -SLZ_ERROR_CUDA;
            } else {
                j.result = -mapDecompressError(err);
            }
            return;
        };
        j.result = @intCast(n);
    }
};

/// CUDA reference: src/streamlz_gpu.zig:434-441. Run `job` on a fresh
/// worker thread with a large stack, blocking until it finishes.
/// Returns the job's `result` (byte count >= 0, or negative
/// SLZ_ERROR_*); -SLZ_ERROR_OUT_OF_MEMORY if the thread cannot spawn.
fn runOnWorker(comptime Job: type, job: *Job) c_int {
    const t = std.Thread.spawn(.{ .stack_size = worker_stack_size }, Job.run, .{job}) catch
        return -SLZ_ERROR_OUT_OF_MEMORY;
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

/// CUDA reference: src/streamlz_gpu.zig:490-513 (v4 #16). Register a
/// custom dictionary on this handle for both directions. The bytes are
/// copied (freed at slzDestroy). The returned ID is content-derived
/// (XXH32 of the bytes, forced into the custom range >= 0x10000000):
/// pass it as `slzCompressOpts_t.dictionary_id`; decompression
/// resolves it automatically from the frame header. Built-in IDs
/// (1..8) need no registration. Registering identical content twice is
/// a no-op returning the same ID.
export fn slzSetDictionary(
    handle: ?*Context,
    dict: ?[*]const u8,
    dict_size: usize,
    id_out: ?*u32,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const ptr = dict orelse return SLZ_ERROR_INVALID_ARG;
    if (dict_size == 0) return SLZ_ERROR_INVALID_ARG;
    const bytes = ptr[0..dict_size];
    const enc_id = h.enc.registerDict(allocator, bytes) catch return SLZ_ERROR_OUT_OF_MEMORY;
    const dec_id = gpu_driver.registerDict(&h.dec, allocator, bytes) catch return SLZ_ERROR_OUT_OF_MEMORY;
    std.debug.assert(enc_id == dec_id);
    if (id_out) |out| out.* = enc_id;
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
    if (rc < 0) return -rc;
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
    if (rc < 0) return -rc;
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
    if (rc < 0) return -rc;
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
    return -rc;
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

// ─────────────────────────────────────────────────────────────────────────
// Phase 4: `_vk`-suffixed C ABI sibling (include/streamlz_gpu_vk.h).
//
// The `_vk` surface is the Vulkan-native counterpart to the CUDA-shaped
// exports above. Per the port-don't-reinvent rule, every `_vk` symbol
// delegates to the same internal codec logic — the only differences are:
//   - return codes (some use slzStatus_t directly instead of a byte
//     count packed in c_int)
//   - the async/poll pattern (slzCompressAsync_vk submits + returns
//     immediately; slzCompressAsyncPoll_vk waits / peeks for completion)
//   - the registration registry tracks (BDA address, VkBuffer, size)
//     for true D2D paths (Tier-1 records the mapping; Tier-2 wires it
//     into the codec descriptor binding)
//   - slzMakeDeviceOnlyHandle_vk returns a synthetic non-dereferenceable
//     sentinel that test code can hand to slzCompress_vk for D2D smoke
//
// Handle type: slzVkHandle_t is opaque per the header. We back it with
// VkContext, which wraps the same Context the CUDA-shaped exports use
// plus the Phase-4 async slots + registration registry. This satisfies
// "the _vk handle is distinct from slzHandle_t to prevent accidentally
// feeding a CUDA handle into a Vulkan entry point" while reusing the
// existing EncodeContext / DecodeContext machinery.

// F-007: derived from `version.string` at comptime so version bumps don't
// silently leave the `_vk` ABI string stale.
const vk_version_string: [:0]const u8 = version.string ++ "+vk";

/// Cap on registered D2D buffers per handle. 16 covers every realistic
/// pattern (input + output + a couple of frame staging buffers per
/// outstanding decode). Mirror of src_vulkan/streamlz_gpu_vk.zig.
///
/// F-004: At Tier-1 the registry is INTENTIONALLY INERT — callers can
/// register/unregister (`slzRegisterBuffer_vk`/`slzUnregisterBuffer_vk`)
/// and we track the (BDA, VkBuffer, size) tuples on the handle, but the
/// codec entry points do not consult the registry to short-circuit the
/// host-bounce path with a real D2D submit (that is Tier-2 BDA work).
/// `lookupRegisteredVk` is retained as a placeholder so the Tier-2
/// wiring needs only to fill in the call sites, not re-add the lookup.
/// `_ = lookupRegisteredVk(...)` discards at the call sites mark exactly
/// where the Tier-2 binding will plug in.
///
/// Thread-safety note (Tier-2 prerequisite): the registry is currently
/// touched only from the calling thread (slzRegisterBuffer_vk runs
/// inline; the async workers don't consult it). When Tier-2 lands, the
/// async worker will read the registry concurrently with the caller's
/// register/unregister — add an `std.Thread.Mutex` field to VkContext
/// at that point.
pub const MAX_REGISTERED_VK: usize = 16;

pub const RegisteredBufferVk = struct {
    address: u64 = 0,
    vk_buffer: ?*anyopaque = null,
    size: usize = 0,
};

/// Per-direction async slot. Owns the worker thread + the result it
/// committed before flipping `done`. Mirrors src_vulkan/streamlz_gpu_vk.zig
/// — the rationale for the worker-thread + atomic-done pattern over a
/// raw VkFence is documented there: the codec is many submits + waits
/// internally (per-chunk gather, transfer-queue split, final D2H), so
/// a single VkFence at the top isn't a fit — wrapping the entire sync
/// call on a worker thread + atomic-done is the closest VK analog to
/// CUDA's "submit on stream, return; caller's sync is the only sync".
pub const AsyncSlotVk = struct {
    thread: ?std.Thread = null,
    done: bool = false,
    result: c_int = SLZ_SUCCESS,
    written: usize = 0,
    compressed_size_out: ?*usize = null,

    fn isBusy(self: *const AsyncSlotVk) bool {
        return self.thread != null and !self.done;
    }
};

/// Async worker stack — mirror of `worker_stack_size` above (32 MiB).
const async_worker_stack_vk: usize = worker_stack_size;

/// VkContext = Context + Phase-4 async + registration.
/// Aliased to slzVkHandle_t at the C ABI surface.
pub const VkContext = struct {
    inner: Context = .{},
    async_enc: AsyncSlotVk = .{},
    async_dec: AsyncSlotVk = .{},
    registry: [MAX_REGISTERED_VK]RegisteredBufferVk = @splat(.{}),
    registry_count: u32 = 0,
};

const AsyncCompressArgsVk = struct {
    h: *VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
};

const AsyncDecompressArgsVk = struct {
    h: *VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
};

fn asyncCompressWorkerVk(args_ptr: *AsyncCompressArgsVk) void {
    defer allocator.destroy(args_ptr);
    const h = args_ptr.h;
    const rc = compressHostVkImpl(
        h,
        args_ptr.d_input,
        args_ptr.input_size,
        args_ptr.d_output,
        args_ptr.output_capacity,
        args_ptr.opts,
    );
    h.async_enc.result = rc;
    if (rc >= 0) {
        h.async_enc.written = @intCast(rc);
        if (h.async_enc.compressed_size_out) |slot| slot.* = @intCast(rc);
    }
    @atomicStore(bool, &h.async_enc.done, true, .release);
}

fn asyncDecompressWorkerVk(args_ptr: *AsyncDecompressArgsVk) void {
    defer allocator.destroy(args_ptr);
    const h = args_ptr.h;
    const rc = decompressHostVkImpl(
        h,
        args_ptr.d_input,
        args_ptr.input_size,
        args_ptr.d_output,
        args_ptr.output_capacity,
        args_ptr.opts,
    );
    h.async_dec.result = rc;
    if (rc >= 0) h.async_dec.written = @intCast(rc);
    @atomicStore(bool, &h.async_dec.done, true, .release);
}

/// Map a std.Thread.SpawnError to the C ABI status. The error set is
/// stable across Windows + Linux on Zig 0.16; OutOfMemory is the only
/// one with an obvious mapping — every other variant is "the OS won't
/// give us another thread right now," which we surface as DEVICE_LOST
/// (the closest "transient system resource pressure" code).
fn mapThreadSpawnErrorVk(err: std.Thread.SpawnError) c_int {
    return switch (err) {
        error.OutOfMemory => SLZ_ERROR_OUT_OF_MEMORY,
        else => SLZ_ERROR_DEVICE_LOST,
    };
}

fn lookupRegisteredVk(h: *VkContext, addr: u64) ?RegisteredBufferVk {
    if (addr == 0) return null;
    var i: u32 = 0;
    while (i < h.registry_count) : (i += 1) {
        if (h.registry[i].address == addr) return h.registry[i];
    }
    return null;
}

/// Synthetic device-only sentinel base. Tier-1 hands callers
/// monotonically-increasing pseudo-addresses starting here so test code
/// can pass a non-null "device pointer" to slzCompress_vk / etc. without
/// owning an actual VkBuffer. Callers MUST NOT dereference it; the
/// codec recognises addresses in this range and routes them through the
/// host-bounce path (the registered VkBuffer is null, so there's no D2D
/// to perform — the call falls back to the host shape).
///
/// Tier-2 work (true BDA D2D) will give registered buffers real device
/// addresses via vkGetBufferDeviceAddress and this synthetic range will
/// stop being needed for those.
///
/// F-005: Range chosen in the UPPER HALF of the 64-bit address space
/// (`1 << 62 .. 1 << 62 + 1 << 40`) so it cannot collide with any
/// `vkGetBufferDeviceAddress` return on any current driver. The Vulkan
/// spec places no upper bound on `VkDeviceAddress`, so a sentinel in
/// the typical 4-256 GiB BDA range is theoretically unsafe even though
/// no shipping driver returns BDAs there today.
const device_only_sentinel_base: usize = 1 << 62;
const device_only_sentinel_window: usize = 1 << 40;
var g_device_only_next: std.atomic.Value(usize) = .init(device_only_sentinel_base);

fn isDeviceOnlySentinel(addr: usize) bool {
    return addr >= device_only_sentinel_base and addr < device_only_sentinel_base + device_only_sentinel_window;
}

// ── Diagnostics (_vk) ──────────────────────────────────────────────────
export fn slzGetVersionString_vk() [*:0]const u8 {
    return vk_version_string.ptr;
}

// ── Handle lifecycle (_vk) ─────────────────────────────────────────────
export fn slzCreate_vk(out_handle: ?*?*VkContext) c_int {
    const slot = out_handle orelse return SLZ_ERROR_INVALID_ARG;
    if (!gpu_encoder.isAvailable()) return SLZ_ERROR_VK_FEATURE_MISSING;
    const ctx = allocator.create(VkContext) catch return SLZ_ERROR_OUT_OF_MEMORY;
    ctx.* = .{};
    slot.* = ctx;
    return SLZ_SUCCESS;
}

export fn slzDestroy_vk(handle: ?*VkContext) void {
    if (handle) |h| {
        // Best-effort: join any in-flight async workers. If a poll
        // already drained the slot, `thread` is null and this is a
        // no-op. If the caller didn't drain, we must not leak the
        // thread — block until the worker returns. Encode first, then
        // decode (matches the order timings are reported in).
        if (h.async_enc.thread) |t| {
            t.join();
            h.async_enc.thread = null;
        }
        if (h.async_dec.thread) |t| {
            t.join();
            h.async_dec.thread = null;
        }
        h.inner.enc.deinit(allocator);
        h.inner.dec.deinit();
        allocator.destroy(h);
    }
}

// ── Buffer registration (_vk) ──────────────────────────────────────────
// Tier-1 records (BDA address, VkBuffer, size) on the handle's registry
// but the codec does NOT yet bind the VkBuffer through the descriptor
// set — the slzCompress_vk / slzDecompress_vk fast path detects a
// registered address and currently falls through to the host bounce
// (with a courtesy memcpy from the device-bound region if the caller
// passed a real device pointer). True BDA-bound D2D is Tier-2 work
// (Phase 5 perf).
//
// `d_base_address` semantics:
//   - non-null: caller already queried vkGetBufferDeviceAddress, we
//               trust the address as-is
//   - null:     the BDA query lives in srcVK and Tier-1 doesn't wire
//               it; reject with INVALID_ARG. (Mirror of src_vulkan's
//               behaviour minus the auto-query — added when Tier-2
//               lands.)
export fn slzRegisterBuffer_vk(
    handle: ?*VkContext,
    vk_buffer_handle: ?*anyopaque,
    d_base_address: ?*const anyopaque,
    buffer_size: usize,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    // vk_buffer_handle is accepted as nullable — Tier-1 records the
    // handle but doesn't bind it (Tier-2 wires the descriptor).
    if (buffer_size == 0) return SLZ_ERROR_INVALID_ARG;
    if (h.registry_count >= MAX_REGISTERED_VK) return SLZ_ERROR_OUT_OF_MEMORY;
    const addr_raw = d_base_address orelse return SLZ_ERROR_INVALID_ARG;
    const addr: u64 = @intFromPtr(addr_raw);
    h.registry[h.registry_count] = .{
        .address = addr,
        .vk_buffer = vk_buffer_handle,
        .size = buffer_size,
    };
    h.registry_count += 1;
    return SLZ_SUCCESS;
}

export fn slzUnregisterBuffer_vk(
    handle: ?*VkContext,
    d_base_address: ?*const anyopaque,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const addr_raw = d_base_address orelse return SLZ_SUCCESS;
    const addr: u64 = @intFromPtr(addr_raw);
    var i: u32 = 0;
    while (i < h.registry_count) : (i += 1) {
        if (h.registry[i].address == addr) {
            const last = h.registry_count - 1;
            if (i != last) h.registry[i] = h.registry[last];
            h.registry[last] = .{};
            h.registry_count = last;
            return SLZ_SUCCESS;
        }
    }
    return SLZ_SUCCESS;
}

// ── Sizing helpers (_vk) ───────────────────────────────────────────────
export fn slzCompressBound_vk(input_size: usize) usize {
    return encoder.compressBound(input_size);
}

/// Hand the caller a synthetic non-dereferenceable "device pointer"
/// that the _vk codec entry points recognise. Test code uses this for
/// D2D smoke without owning real device memory. The pointer is never
/// dereferenced by the codec — its only role is to satisfy the
/// caller-visible "this looks like a device address" contract and
/// route control to the device-only path (which Tier-1 falls back to
/// the host shape for).
export fn slzMakeDeviceOnlyHandle_vk(
    out_handle: ?*?*const anyopaque,
    bytes: usize,
) c_int {
    const slot = out_handle orelse return SLZ_ERROR_INVALID_ARG;
    // F-006: bump by the caller's requested size (page-aligned), so two
    // sentinels handed out in sequence don't overlap when the caller
    // treats them as N-byte ranges. Minimum stride 4 KiB keeps the
    // sentinels well-separated for tiny `bytes` values.
    const stride_raw = if (bytes < 4096) 4096 else bytes;
    const page_mask: usize = 4095;
    const stride = (stride_raw + page_mask) & ~page_mask;
    // Wrap-protect: if the bump counter wraps past the sentinel range,
    // refuse. 1 TiB of distinct sentinels covers every realistic test.
    const next = g_device_only_next.fetchAdd(stride, .monotonic);
    if (!isDeviceOnlySentinel(next)) {
        return SLZ_ERROR_OUT_OF_MEMORY;
    }
    const ptr: ?*const anyopaque = @ptrFromInt(next);
    slot.* = ptr;
    return SLZ_SUCCESS;
}

// ── Sync compress / decompress (_vk) ───────────────────────────────────
// _vk return shape: int = byte count when >= 0, slzStatus_t value when < 0.
// Identical to the CUDA-shaped exports above; only the registration
// + sentinel handling differs.

fn compressHostVkImpl(
    h: *VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    if (input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    if (output == null) return SLZ_ERROR_INVALID_ARG;
    const in_ptr: [*]const u8 = @ptrCast(input.?);
    const out_ptr: [*]u8 = @ptrCast(output.?);
    var job = CompressJob{
        .h = &h.inner,
        .src = in_ptr[0..input_size],
        .dst = out_ptr[0..output_capacity],
        .opts = opts,
    };
    return runOnWorker(CompressJob, &job);
}

fn decompressHostVkImpl(
    h: *VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    if (input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    // F-008: symmetric with compressHostVkImpl + CUDA-shaped surface — reject
    // output==null unconditionally rather than only when output_capacity!=0.
    if (output == null) return SLZ_ERROR_INVALID_ARG;
    const in_ptr: [*]const u8 = @ptrCast(input.?);
    const out_ptr: [*]u8 = @ptrCast(output.?);
    var job = DecompressJob{
        .h = &h.inner,
        .src = in_ptr[0..input_size],
        .dst = out_ptr[0..output_capacity],
        .opts = opts,
    };
    return runOnWorker(DecompressJob, &job);
}

export fn slzCompressHost_vk(
    handle: ?*VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return compressHostVkImpl(h, input, input_size, output, output_capacity, opts);
}

export fn slzDecompressHost_vk(
    handle: ?*VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return decompressHostVkImpl(h, input, input_size, output, output_capacity, opts);
}

/// Device-pointer variant of compress. Tier-1 semantics:
///   - if `d_input` is a registered BDA address, fall back to the
///     host shape (Tier-1 doesn't wire the descriptor binding yet)
///   - if `d_input` is a synthetic device-only sentinel from
///     slzMakeDeviceOnlyHandle_vk, route to the host bounce with
///     output going wherever d_output points (caller's
///     responsibility to allocate a real host buffer for output in
///     that case — same as the CUDA-shaped slzCompressAsync_vk pre-
///     existing pattern)
///   - otherwise treat as a host pointer and call compressHostVkImpl
export fn slzCompress_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    if (d_input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    if (d_output == null) return SLZ_ERROR_INVALID_ARG;

    // Synthetic device-only sentinel for testing: the codec can't
    // dereference it; the only viable Tier-1 path is to surface an
    // UNSUPPORTED so the test can route through the host shape
    // explicitly.
    const in_addr: usize = @intFromPtr(d_input);
    if (isDeviceOnlySentinel(in_addr)) return SLZ_ERROR_UNSUPPORTED;

    // Registered-address fast path: Tier-2 binds the registered
    // VkBuffer; Tier-1 records it but doesn't bind, so we treat a
    // registered hit identically to a host pointer for the purposes
    // of the encode call (the caller's address IS a valid host
    // address — we just have richer metadata about it).
    _ = lookupRegisteredVk(h, in_addr);
    return compressHostVkImpl(h, d_input, input_size, d_output, output_capacity, opts);
}

export fn slzDecompress_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    if (d_input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    // F-008: symmetric with slzCompress_vk + CUDA-shaped surface.
    if (d_output == null) return SLZ_ERROR_INVALID_ARG;

    const in_addr: usize = @intFromPtr(d_input);
    if (isDeviceOnlySentinel(in_addr)) return SLZ_ERROR_UNSUPPORTED;
    _ = lookupRegisteredVk(h, in_addr);
    return decompressHostVkImpl(h, d_input, input_size, d_output, output_capacity, opts);
}

// ── Async + polling (_vk) ──────────────────────────────────────────────
// VK-native submit-and-poll pattern: slzCompressAsync_vk spawns a
// worker, returns SLZ_SUCCESS immediately. slzCompressAsyncPoll_vk
// peeks (`blocking == 0`) or waits (`blocking != 0`) for the worker
// to commit its result. Mirror of src_vulkan/streamlz_gpu_vk.zig
// — see that file's header for the design rationale.
//
// Return shape of the Poll exports (mirror src_vulkan):
//   SLZ_SUCCESS                if the slot was idle or the op completed
//                              successfully (slot is reset; the
//                              committed byte count is in the previous
//                              non-poll call's `compressed_size_out` or
//                              reflected in subsequent
//                              slzGetLastTimings_vk drain).
//   SLZ_ERROR_UNSUPPORTED      blocking==0 and the op is in flight —
//                              nvCOMP's "not ready" sentinel.
//   negative status            op completed with an error; slot reset.
//
// NOTE: this departs slightly from the header's slzStatus_t return
// type contract — the header says SLZ_SUCCESS on completion regardless
// of byte count. We carry the byte count via the original async call's
// `compressed_size` out-pointer (which the worker writes synchronously
// before flipping `done`).
export fn slzCompressAsync_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    compressed_size: ?*usize,
    opts: CompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    if (d_input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    if (d_output == null) return SLZ_ERROR_INVALID_ARG;
    if (h.async_enc.isBusy()) return SLZ_ERROR_UNSUPPORTED;
    // F-003 fix (2026-06-08): `isBusy()` returns false when
    // `thread != null && done == true` — i.e. a prior async call
    // committed but the caller never polled the slot. If we
    // overwrote `h.async_enc` here with a fresh default the still-
    // joinable `std.Thread` handle would leak (OS handle leak on
    // Windows + Linux) and a subsequent slzDestroy_vk would no
    // longer have a `thread` to join. Drain the stale slot first.
    if (h.async_enc.thread) |t| {
        t.join();
        h.async_enc.thread = null;
    }
    // Reset slot; if a previous op left it in done-state, the caller
    // missed their drain — Poll is idempotent so we don't surface that
    // here.
    h.async_enc = .{ .compressed_size_out = compressed_size };

    const args = allocator.create(AsyncCompressArgsVk) catch return SLZ_ERROR_OUT_OF_MEMORY;
    args.* = .{
        .h = h,
        .d_input = d_input,
        .input_size = input_size,
        .d_output = d_output,
        .output_capacity = output_capacity,
        .opts = opts,
    };
    const t = std.Thread.spawn(
        .{ .stack_size = async_worker_stack_vk },
        asyncCompressWorkerVk,
        .{args},
    ) catch |err| {
        allocator.destroy(args);
        h.async_enc = .{};
        return mapThreadSpawnErrorVk(err);
    };
    h.async_enc.thread = t;
    return SLZ_SUCCESS;
}

export fn slzCompressAsyncPoll_vk(handle: ?*VkContext, blocking: c_int) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return pollSlotVk(&h.async_enc, blocking);
}

export fn slzDecompressAsync_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    if (d_input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    if (d_output == null and output_capacity != 0) return SLZ_ERROR_INVALID_ARG;
    if (h.async_dec.isBusy()) return SLZ_ERROR_UNSUPPORTED;
    // F-003 fix (2026-06-08): see slzCompressAsync_vk for rationale.
    // Drain any stale thread handle from a prior un-polled completion
    // before stomping the slot.
    if (h.async_dec.thread) |t| {
        t.join();
        h.async_dec.thread = null;
    }
    h.async_dec = .{};

    const args = allocator.create(AsyncDecompressArgsVk) catch return SLZ_ERROR_OUT_OF_MEMORY;
    args.* = .{
        .h = h,
        .d_input = d_input,
        .input_size = input_size,
        .d_output = d_output,
        .output_capacity = output_capacity,
        .opts = opts,
    };
    const t = std.Thread.spawn(
        .{ .stack_size = async_worker_stack_vk },
        asyncDecompressWorkerVk,
        .{args},
    ) catch |err| {
        allocator.destroy(args);
        h.async_dec = .{};
        return mapThreadSpawnErrorVk(err);
    };
    h.async_dec.thread = t;
    return SLZ_SUCCESS;
}

export fn slzDecompressAsyncPoll_vk(handle: ?*VkContext, blocking: c_int) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return pollSlotVk(&h.async_dec, blocking);
}

fn pollSlotVk(slot: *AsyncSlotVk, blocking: c_int) c_int {
    if (slot.thread == null) {
        // No op in flight — idempotent success.
        return SLZ_SUCCESS;
    }
    if (blocking == 0) {
        if (!@atomicLoad(bool, &slot.done, .acquire)) return SLZ_ERROR_UNSUPPORTED;
    }
    if (slot.thread) |t| {
        t.join();
        slot.thread = null;
    }
    const rc = slot.result;
    slot.* = .{};
    return rc;
}

// ── Per-kernel timings drain (_vk) ─────────────────────────────────────
// Returns the count of timings written. The CUDA-shaped
// slzGetLastTimings packs encode + decode timings into one array; the
// _vk shape returns the count directly (no separate out-pointer) and
// otherwise behaves identically.
export fn slzGetLastTimings_vk(
    handle: ?*VkContext,
    out: ?[*]KernelTimingC,
    capacity: usize,
) usize {
    const h = handle orelse return 0;
    gpu_driver.finalizeProfiling(&h.inner.enc.pending_timings, &h.inner.enc.last_timings);
    gpu_driver.finalizeProfiling(&h.inner.dec.pending_timings, &h.inner.dec.last_timings);
    const enc_list = h.inner.enc.last_timings.items;
    const dec_list = h.inner.dec.last_timings.items;
    const total = enc_list.len + dec_list.len;
    if (out) |buf| {
        var i: usize = 0;
        for (enc_list) |kt| {
            if (i >= capacity) break;
            buf[i] = .{ .name = kt.name, .ms = kt.ms };
            i += 1;
        }
        for (dec_list) |kt| {
            if (i >= capacity) break;
            buf[i] = .{ .name = kt.name, .ms = kt.ms };
            i += 1;
        }
    }
    return total;
}
