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
const gpu_encoder = @import("encode/driver.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_driver = @import("decode/driver.zig");
const frame = @import("format/frame_format.zig");
const version = @import("version.zig");
const cuda_api = @import("decode/cuda_api.zig");

const allocator = std.heap.c_allocator;

/// Worker-thread stack. The compress orchestration has multi-MB stack
/// frames; this keeps callers safe even on small-stack threads.
const worker_stack_size: usize = 32 << 20;

/// Non-dereferenceable address embedded in the host slice that
/// `slzCompressAsync` hands to `CompressJob` when the input lives on the
/// device. The pipeline reads source bytes from `d_input_override`
/// instead; the slice's `.ptr` exists only because the job carries the
/// `.len` for size calculations.
///
/// Address 0x10 is never a valid user-space pointer on any supported
/// platform; dereferencing it crashes immediately and visibly, which is
/// the desired behaviour if anything downstream forgets the contract
/// and tries to read the bytes.
const device_only_host_stub_addr: usize = 0x10;

/// Build a sentinel host slice whose `.ptr` is `device_only_host_stub_addr`
/// and whose `.len` is `len`. Encoder paths that consult `.len` (for
/// chunk-count math, descriptor sizing, etc.) work normally; any
/// accidental read of the bytes will fault on a guaranteed-invalid
/// pointer rather than corrupting silently. The encoder's only
/// remaining host-read site (`fast_framed.writeUncompressedFrame`)
/// is unreachable for async-D2D inputs thanks to the size guards in
/// `slzCompressAsync` below.
fn deviceOnlySrcStub(len: usize) []const u8 {
    const ptr: [*]const u8 = @ptrFromInt(device_only_host_stub_addr);
    return ptr[0..len];
}

// ── Status codes ───────────────────────────────────────────────────────
// Mirror of `slzStatus_t` in `include/streamlz_gpu.h`. The comptime
// asserts below lock the values; adding or reordering the C enum
// without updating these is a compile-time error here.
const SLZ_SUCCESS: c_int = 0;
const SLZ_ERROR_INVALID_HANDLE: c_int = 1;
const SLZ_ERROR_INVALID_ARG: c_int = 2;
const SLZ_ERROR_BUFFER_TOO_SMALL: c_int = 3;
const SLZ_ERROR_CORRUPT_FRAME: c_int = 4;
const SLZ_ERROR_UNSUPPORTED: c_int = 5;
const SLZ_ERROR_CUDA: c_int = 6;
const SLZ_ERROR_OUT_OF_MEMORY: c_int = 7;
// Vulkan-port additions (shared ABI; CUDA backend never returns
// these but the slot reservation prevents future renumbering).
const SLZ_ERROR_DEVICE_LOST: c_int = 8;
const SLZ_ERROR_VK_FEATURE_MISSING: c_int = 9;

comptime {
    // If the C enum's numeric values ever change, the explicit
    // constants above and the switch in slzStatusString must be
    // updated together. Pin the values here.
    std.debug.assert(SLZ_SUCCESS == 0);
    std.debug.assert(SLZ_ERROR_OUT_OF_MEMORY == 7);
    std.debug.assert(SLZ_ERROR_DEVICE_LOST == 8);
    std.debug.assert(SLZ_ERROR_VK_FEATURE_MISSING == 9);
}

/// Pseudo-kernel timing slot names. The library emits these into the
/// timings array when `enable_profiling` is set on the call; the `ms`
/// field carries a `uint32_t` byte count via `@bitCast`, not a wall-
/// clock duration. See the slzKernelTiming_t docs in
/// `include/streamlz_gpu.h`.
pub const pseudo_kernel_compressed_size: [*:0]const u8 = "__compressed_size__";
pub const pseudo_kernel_decompressed_size: [*:0]const u8 = "__decompressed_size__";

/// Mirror of `slzCompressOpts_t`. Default `level = 5` matches the C
/// header doc; the Zig-side encoder `Options.level` defaults to 1
/// because the Zig CLI prioritizes speed. Both surfaces are public; the
/// difference is intentional and the per-call value carries through.
///
/// `effective_level_out` is an OUT slot the CUDA path writes with
/// `opts.level` (CUDA never clamps); the Vulkan Tier-2 path will
/// write the post-clamp level when mobile-VRAM forces L5->L3. See
/// `include/streamlz_gpu.h` slzCompressOpts_t docs. Struct size is
/// still 32 B (8 * c_int) — the slot reuses the first `reserved`
/// entry; the trailing `reserved` array shrinks to [5]c_int.
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

/// Mirror of `slzDecompressOpts_t`.
const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = @splat(0),
};

/// Mirror of `slzKernelTiming_t`.
const KernelTimingC = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

comptime {
    // Both Opts structs are `8 * sizeof(c_int)` = 32 bytes on every
    // platform we target (LP64 / LLP64 with 32-bit `int`). Lock the
    // size so a stray field addition that breaks the C ABI fails to
    // compile.
    std.debug.assert(@sizeOf(CompressOpts) == 8 * @sizeOf(c_int));
    std.debug.assert(@sizeOf(DecompressOpts) == 8 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "level") == 0);
    std.debug.assert(@offsetOf(CompressOpts, "enable_profiling") == @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "effective_level_out") == 2 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "dictionary_id") == 3 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(CompressOpts, "reserved") == 4 * @sizeOf(c_int));
    std.debug.assert(@offsetOf(DecompressOpts, "enable_profiling") == 0);
    std.debug.assert(@offsetOf(DecompressOpts, "reserved") == @sizeOf(c_int));
    // `slzKernelTiming_t` is `const char* + float` with trailing pad to
    // pointer alignment on 64-bit. Lock the offset, leave the trailing
    // padding to the platform.
    std.debug.assert(@offsetOf(KernelTimingC, "name") == 0);
    std.debug.assert(@offsetOf(KernelTimingC, "ms") == @sizeOf(*anyopaque));
    std.debug.assert(@sizeOf(KernelTimingC) == 2 * @sizeOf(*anyopaque));
}

/// Library handle. Owns the per-operation GPU contexts so concurrent
/// handles cannot corrupt one another.
const Context = struct {
    enc: gpu_encoder.EncodeContext = .{},
    dec: gpu_driver.DecodeContext = .{},
};

/// Map an `encoder.CompressError` (which embeds `gpu_driver.GpuError`
/// and `Allocator.Error`) to the C ABI status code.
///
/// Categories:
///   - shape rejected by the encoder (`BadLevel`, `BadScGroupSize`,
///     `BadBlockSize`) → `SLZ_ERROR_UNSUPPORTED`
///   - host alloc / GPU alloc failure → `SLZ_ERROR_OUT_OF_MEMORY`
///   - output buffer too small → `SLZ_ERROR_BUFFER_TOO_SMALL`
///   - every CUDA-side failure (`GpuError.{KernelLaunchFailed,
///     SyncFailed, CopyFailed, KernelMissing, BackendNotAvailable,
///     BadMode}`) → `SLZ_ERROR_CUDA`
///
/// The parameter is `anyerror` (not `CompressError`) so the Zig
/// compiler does NOT flag a new member added to `CompressError` or
/// the embedded `GpuError` upstream — such a variant silently falls
/// through to `else => SLZ_ERROR_CUDA`. The arms below enumerate
/// every current member explicitly; audit this switch whenever a
/// variant is added.
fn mapCompressError(err: anyerror) c_int {
    return switch (err) {
        // UnknownDictionary: the caller named a dictionary_id the
        // registry cannot resolve (v4 #16).
        error.BadLevel, error.BadScGroupSize, error.BadBlockSize, error.UnknownDictionary => SLZ_ERROR_UNSUPPORTED,
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

/// Map a `decoder.DecompressError` (which embeds `gpu_driver.GpuError`
/// and `Allocator.Error`) to the C ABI status code.
///
/// Categories:
///   - frame-parsing failure (header, block, chunk, checksum, dictionary,
///     content-size cap) → `SLZ_ERROR_CORRUPT_FRAME`
///   - caller output buffer too small → `SLZ_ERROR_BUFFER_TOO_SMALL`
///   - host alloc / GPU alloc failure → `SLZ_ERROR_OUT_OF_MEMORY`
///   - every CUDA-side failure → `SLZ_ERROR_CUDA`
///
/// `BadMode` here means the frame shape is not supported by the
/// device-resident decoder (multi-block, dictionary, etc.); the C ABI
/// can fall back to the host-bounce entry point, so the caller sees
/// `SLZ_ERROR_UNSUPPORTED`.
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
        error.BadMode => SLZ_ERROR_UNSUPPORTED,
        error.BackendNotAvailable,
        error.KernelLaunchFailed,
        error.SyncFailed,
        error.CopyFailed,
        error.KernelMissing,
        => SLZ_ERROR_CUDA,
        else => SLZ_ERROR_CUDA,
    };
}

/// Returns true when `err` belongs to `gpu_driver.GpuError`. The
/// device-resident decode path (`DecompressJobTrueD2D`) uses this to
/// route any GPU-side failure into the host-bounce fallback rather
/// than letting `mapDecompressError` misclassify it.
///
/// The comptime block locks the listed members against
/// `descriptors.zig:GpuError`: adding a new member there without
/// updating the switch below is a compile error.
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
    // Level range is validated by `encoder.compressFramed` (returns
    // `error.BadLevel` → SLZ_ERROR_UNSUPPORTED via mapCompressError);
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

fn decompressCore(h: *Context, frame_bytes: []const u8, output: []u8, opts: DecompressOpts) c_int {
    h.dec.enable_profiling = opts.enable_profiling != 0;
    defer h.dec.enable_profiling = false;
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
// `d_src`/`d_dst` are 0 for the host->host path, or device addresses for
// the device->device path (then `src` aliases an internal bounce buffer
// that the worker fills via D2H before the core runs).
//
// Job-result protocol: `result` >= 0 is a byte count; errors are
// NEGATED SLZ_ERROR_* codes (the public constants are positive byte
// counts' namespace-mates, so a positive error would be misread as a
// tiny successful output - which is exactly the latent bug the v4 #16
// dict test exposed: every codec error on these paths used to report
// success with output_size = the error code). Entry points negate
// back before returning to the C caller.
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
    result: c_int = -SLZ_ERROR_CUDA,

    fn run(j: *CompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = -SLZ_ERROR_CUDA;
            return;
        }
        // Device-resident input: the encoder reads source bytes from
        // d_input_override (a caller-supplied device pointer) instead of
        // H2D-ing them; the `src` host slice is left untouched.
        if (j.d_src != 0) {
            j.h.enc.d_input_override = j.d_src;
        }
        // Device-resident output: the frame-assembly kernel writes the
        // SLZ1 frame straight into d_output_override, no host bounce.
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
                j.result = -SLZ_ERROR_CUDA;
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
    result: c_int = -SLZ_ERROR_CUDA,

    fn run(j: *DecompressJob) void {
        if (!gpu_driver.bindContextToCallingThread()) {
            j.result = -SLZ_ERROR_CUDA;
            return;
        }
        j.result = decompressCore(j.h, j.src, j.dst, j.opts);
    }
};

/// TRUE D2D decompress worker: frame stays on device, walk runs on
/// GPU, output stays on device — zero host bounce of payload or frame
/// bytes. Sets `fall_back = true` if the frame shape doesn't fit the
/// slzCompress invariant (dict / PDM / multi-block / size out of bounds)
/// so the C ABI entry point can return SLZ_ERROR_UNSUPPORTED.
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
        SLZ_ERROR_DEVICE_LOST => "device lost",
        SLZ_ERROR_VK_FEATURE_MISSING => "required Vulkan feature missing",
        else => "unknown status",
    };
}

/// Library version: `MAJOR.MINOR.PATCH`, null-terminated, valid for
/// the lifetime of the process.
export fn slzVersionString() [*:0]const u8 {
    return version.string;
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
    if (handle) |h| {
        // Free every device + host buffer the encode/decode contexts
        // own. Without this, every `cuMemAlloc` and `cuMemAllocHost`
        // made on the handle's behalf leaks until the process exits.
        h.enc.deinit(allocator);
        h.dec.deinit();
        allocator.destroy(h);
    }
    return SLZ_SUCCESS;
}

/// v4 #16: register a custom dictionary on this handle for both
/// directions. The bytes are copied (freed at slzDestroy). The
/// returned ID is content-derived (XXH32 of the bytes, forced into
/// the custom range >= 0x10000000): pass it as
/// `slzCompressOpts_t.dictionary_id`; decompression resolves it
/// automatically from the frame header. Built-in IDs (1..8) need no
/// registration. Registering identical content twice is a no-op
/// returning the same ID.
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
export fn slzCompressDefaultOpts() CompressOpts {
    return .{};
}

export fn slzDecompressDefaultOpts() DecompressOpts {
    return .{};
}

// ── Size queries ──────────────────────────────────────────────────────

/// Worst-case compressed-frame size for `input_size` input bytes. The
/// bound is level-independent: it sizes the worst-case
/// uncompressed-body shape (verbatim source + all frame / block /
/// chunk / sub-chunk headers + the SC tail prefix table). `opts` is
/// therefore intentionally unread - callers may pass
/// `slzCompressDefaultOpts()` or a per-call configuration without
/// affecting the result.
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
    _ = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const hdr_ptr = host_bytes orelse return SLZ_ERROR_INVALID_ARG;
    const out = decompressed_size orelse return SLZ_ERROR_INVALID_ARG;
    const bytes: [*]const u8 = @ptrCast(hdr_ptr);
    // 64-byte slice is safely longer than any header we produce (max ~26 B)
    // and the caller contract requires that many bytes be readable.
    const parsed = frame.parseHeader(bytes[0..64]) catch return SLZ_ERROR_CORRUPT_FRAME;
    const size = parsed.content_size orelse return SLZ_ERROR_CORRUPT_FRAME;
    // Match the decoder's `DecompressError.ContentSizeTooLarge` gate so
    // the caller learns about the cap here rather than allocating
    // a >4 GiB device buffer that the dispatch will then reject.
    if (size > decoder.max_content_size) return SLZ_ERROR_CORRUPT_FRAME;
    out.* = size;
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
    if (rc < 0) return -rc;
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
    if (rc < 0) return -rc;
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

    // The encode pipeline writes an uncompressed-body frame on the host
    // when the input is too small or too large to feed the LZ kernel
    // grid. Both paths `@memcpy` from the host `src` slice, which is
    // the device-only sentinel here. Reject upfront so the caller
    // retries on the host-bounce entry point; the path is not safe to
    // take silently.
    //
    // The thresholds match the gates inside `fast_framed.compressFramedOne`
    // (`min_source_length = 128`) and `gpuEncodeAndAssemble`
    // (`assembly_chunk_cap = 16384` * minimum 64 KB sub-chunk).
    // ToDo.md: we could D2H from `d_input` instead of bouncing the
    // whole call. Not done yet — the threshold rejects only inputs no
    // realistic D2D caller would feed.
    const async_min_input: usize = 128 + 1;
    // 16384 sub-chunks × 64 KB minimum sub-chunk = 1 GiB. Inputs above
    // this would either overflow the assembly chunk cap or, at larger
    // sc_group, are gated by a higher per-call ceiling we don't apply
    // here (callers chunk on the API side for that case).
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
    // Run inline on the caller's thread (nvCOMP-style — no per-call
    // thread spawn). The host front half (setup + measure-pass D2H +
    // headers) blocks the caller; the heavy back-half kernels +
    // final D2D copy ride `stream` and complete after the caller's
    // cudaStreamSynchronize.
    CompressJob.run(&job);
    const rc = job.result;
    if (rc < 0) return -rc;
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
    // The async path can't bounce through the host on the caller's
    // behalf. If the TrueD2D shape isn't satisfied (PDM / multi-block /
    // size out of bounds - dictionary frames ARE supported since
    // v4 #16, given the dictionary is built-in or was registered via
    // slzSetDictionary), return SLZ_ERROR_UNSUPPORTED so the caller
    // can retry via slzDecompressHost.
    if (true_job.fall_back) return SLZ_ERROR_UNSUPPORTED;
    return -rc;
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

/// Wait on the caller's CUstream and then drain per-kernel timings.
/// Async convenience: avoids the caller's `cudaStreamSynchronize` +
/// `slzGetLastTimings` pair. Passing `stream = NULL` behaves like the
/// synchronous `slzGetLastTimings` (no stream sync; assumes the
/// caller has already synced).
///
/// Handle and argument validation are delegated to `slzGetLastTimings`
/// to avoid double-validating.
export fn slzWaitAndGetLastTimings(
    handle: ?*Context,
    stream: ?*anyopaque,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    if (stream) |s| {
        const sync_fn = cuda_api.cuStreamSync_fn orelse return SLZ_ERROR_CUDA;
        if (sync_fn(@intFromPtr(s)) != cuda_api.CUDA_SUCCESS) return SLZ_ERROR_CUDA;
    }
    return slzGetLastTimings(handle, timings, capacity, count_out);
}
