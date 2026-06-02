//! StreamLZ Vulkan-backend C ABI (M4 scaffolding).
//!
//! Implements the contract in `include/streamlz_gpu_vk.h`: a handle-based
//! Vulkan compressor exported as `streamlz_vk.dll` (+ `streamlz_vk.lib`).
//! Mirrors the shape of `src/streamlz_gpu.zig`; the per-call codec paths
//! are stubbed to `SLZ_ERROR_UNSUPPORTED` until the per-kernel
//! milestones (M11..M27) wire them.
//!
//! Currently wired:
//!   - slzCreate_vk / slzDestroy_vk           (driver bring-up + tier probe)
//!   - slzGetVersionString_vk                 (version.string + "+vk")
//!   - slzRegisterBuffer_vk / Unregister_vk   (no-op on Tier-1; M6+ for T2)
//!
//! Every other entry point returns SLZ_ERROR_UNSUPPORTED (== 5).

const std = @import("std");

const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const slz1_codec = @import("slz1_codec.zig");
const l1_codec = @import("l1_codec.zig");

const allocator = std.heap.c_allocator;

/// Process-singleton `std.Io.Threaded` instance for SPV-blob filesystem
/// loads. Initialised lazily on the first compress / decompress call —
/// the smoke-test path that only exercises slzCreate_vk + slzDestroy_vk
/// avoids the threaded io chassis. Never torn down (a future
/// `slzShutdown_vk` could do this); the OS reclaims the thread pool at
/// process exit.
var g_io_threaded: std.Io.Threaded = undefined;
var g_io_inited: bool = false;

fn ensureIo() std.Io {
    if (!g_io_inited) {
        g_io_threaded = std.Io.Threaded.init(allocator, .{});
        g_io_inited = true;
    }
    return g_io_threaded.io();
}

// Version string for the Vulkan backend. Kept in sync with
// `src/version.zig` by the CHANGELOG bump checklist; the two are not
// physically linked because the Vulkan module's package root lives at
// `src_vulkan/` and pulling `../src/version.zig` would escape that root.
// The "+vk" build-metadata suffix follows SemVer 2.0 §10.
const vk_version_string: [:0]const u8 = "3.0.0+vk";

// ── Status codes ──────────────────────────────────────────────────────
// Mirror of `slzStatus_t` from `include/streamlz_gpu.h`. Kept in sync
// with `src/streamlz_gpu.zig`; the comptime asserts below pin the
// values so a future renumber on either side fails to compile.
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
    std.debug.assert(SLZ_ERROR_UNSUPPORTED == 5);
    std.debug.assert(SLZ_ERROR_DEVICE_LOST == 8);
    std.debug.assert(SLZ_ERROR_VK_FEATURE_MISSING == 9);
}

// ── ABI structs (shared layout with the CUDA path) ────────────────────
pub const CompressOpts = extern struct {
    level: c_int = 5,
    enable_profiling: c_int = 0,
    effective_level_out: c_int = 0,
    reserved: [5]c_int = @splat(0),
};

pub const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = @splat(0),
};

pub const KernelTimingC = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

comptime {
    std.debug.assert(@sizeOf(CompressOpts) == 8 * @sizeOf(c_int));
    std.debug.assert(@sizeOf(DecompressOpts) == 8 * @sizeOf(c_int));
    std.debug.assert(@sizeOf(KernelTimingC) == 2 * @sizeOf(*anyopaque));
}

// ── Handle ────────────────────────────────────────────────────────────
/// Vulkan-backend handle. Owns the per-handle classification result
/// (vendor, tier) so dispatch sites don't re-probe. The shared Vulkan
/// instance + device live on the process-wide `driver.g_default`.
pub const VkContext = struct {
    tier: probe_mod.Tier = .unsupported,
    vendor_id: u32 = 0,
    device_id: u32 = 0,
    subgroup_size: u32 = 0,
    /// Last sync (or completed-async) compress / decompress timings,
    /// stored in nanoseconds (taken from
    /// `l1_codec.last_{encode,decode}_dispatch_ns` plus the per-phase
    /// decode breakdown). `slzGetLastTimings_vk` translates them into
    /// the `KernelTimingC` array the caller drains. A value of 0 means
    /// "no timing captured" — the call that wrote it didn't dispatch
    /// that kernel.
    last_encode_dispatch_ns: u64 = 0,
    last_decode_dispatch_ns: u64 = 0,
    /// Async state (managed by `slzCompressAsync_vk` /
    /// `slzCompressAsyncPoll_vk` and the decode pair). A simple
    /// worker-thread + flag scheme matches the CUDA backend's
    /// per-call worker spawn (`runOnWorker`); Vulkan timeline
    /// semaphores would be lower latency but the overhead of a
    /// per-call thread spawn is dominated by the GPU dispatch in
    /// every case our async callers care about.
    async_enc: AsyncSlot = .{},
    async_dec: AsyncSlot = .{},
};

/// Per-direction async slot. Owns the in-flight worker thread (if any)
/// and the most recent result. The slot is "busy" while `thread` is
/// non-null and `done` is false; `slzCompressAsyncPoll_vk` /
/// `slzDecompressAsyncPoll_vk` joins the thread, captures the result,
/// and clears the slot when polled with `blocking != 0`.
pub const AsyncSlot = struct {
    thread: ?std.Thread = null,
    done: bool = false,
    result: c_int = SLZ_SUCCESS,
    written: usize = 0,
    /// If non-null, where to write the byte count back when the worker
    /// completes (matches the CUDA `compressed_size` slot — but the
    /// Vulkan path writes it synchronously from the worker so the
    /// poller doesn't need to know about it).
    compressed_size_out: ?*usize = null,

    fn isBusy(self: *const AsyncSlot) bool {
        return self.thread != null and !self.done;
    }
};

// ── Async worker contexts (heap-allocated; the worker owns its own
//    copy of the args so the caller's slzCompressAsync_vk frame can
//    return immediately).
const AsyncCompressArgs = struct {
    handle: *VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
};
const AsyncDecompressArgs = struct {
    handle: *VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
};

fn asyncCompressWorker(args_ptr: *AsyncCompressArgs) void {
    defer allocator.destroy(args_ptr);
    const h = args_ptr.handle;
    // The worker calls the synchronous path which already drives the
    // Vulkan submit + fence wait + readback. From the caller's
    // perspective the call is async because the spawn returns
    // immediately; the worker thread blocks here until the GPU finishes.
    const rc = slzCompressHost_vk(
        h,
        args_ptr.input,
        args_ptr.input_size,
        args_ptr.output,
        args_ptr.output_capacity,
        args_ptr.opts,
    );
    h.async_enc.result = rc;
    if (rc >= 0) {
        h.async_enc.written = @intCast(rc);
        if (h.async_enc.compressed_size_out) |slot| slot.* = @intCast(rc);
    }
    // `done` is set last so a concurrent poll observing `done == true`
    // can rely on `result` / `written` already being committed.
    @atomicStore(bool, &h.async_enc.done, true, .release);
}

fn asyncDecompressWorker(args_ptr: *AsyncDecompressArgs) void {
    defer allocator.destroy(args_ptr);
    const h = args_ptr.handle;
    const rc = slzDecompressHost_vk(
        h,
        args_ptr.input,
        args_ptr.input_size,
        args_ptr.output,
        args_ptr.output_capacity,
        args_ptr.opts,
    );
    h.async_dec.result = rc;
    if (rc >= 0) h.async_dec.written = @intCast(rc);
    @atomicStore(bool, &h.async_dec.done, true, .release);
}

/// Sentinel value handed back by `slzMakeDeviceOnlyHandle_vk`. Today
/// (Phase 1) this is just a non-null marker pointer — the codec paths
/// don't yet know how to dereference it as a device address. Phase 2
/// (A2 — BDA wiring in `slzCompress_vk` / `slzDecompress_vk`) will
/// teach the codec to recognize it and skip the H2D copy. The address
/// is intentionally invalid for host dereference (matches CUDA's
/// `device_only_host_stub_addr = 0x10` pattern) so any path that
/// accidentally reads through it faults loudly.
const device_only_sentinel_addr: usize = 0x10;

/// Test whether a host pointer is the device-only sentinel from
/// `slzMakeDeviceOnlyHandle_vk`. Phase 1: the codec rejects sentinels
/// with `SLZ_ERROR_UNSUPPORTED` (no D2D path yet). Phase 2 makes this
/// the trigger that flips the codec into device-address mode.
fn isDeviceOnlySentinel(p: ?*const anyopaque) bool {
    if (p == null) return false;
    return @intFromPtr(p) == device_only_sentinel_addr;
}

// ── Diagnostics ───────────────────────────────────────────────────────
pub export fn slzGetVersionString_vk() [*:0]const u8 {
    return vk_version_string.ptr;
}

// ── Handle lifecycle ──────────────────────────────────────────────────
pub export fn slzCreate_vk(out_handle: ?*?*VkContext) c_int {
    const slot = out_handle orelse return SLZ_ERROR_INVALID_ARG;

    // Bring up the loader + instance + (device, queue). Idempotent —
    // multiple concurrent slzCreate_vk callers re-enter and the second
    // returns immediately with the existing g_default.
    driver.ensureInit() catch return SLZ_ERROR_VK_FEATURE_MISSING;

    // Classify the device that ensureInit() picked. If the probe says
    // `unsupported`, the device is below our Vulkan 1.2 floor (or some
    // entry point was missing); fail init with VK_FEATURE_MISSING so
    // the caller doesn't get a half-live handle.
    const result = probe_mod.probe(driver.g_default.inst, driver.g_default.pd);
    if (result.tier == .unsupported) return SLZ_ERROR_VK_FEATURE_MISSING;

    const ctx = allocator.create(VkContext) catch return SLZ_ERROR_OUT_OF_MEMORY;
    ctx.* = .{
        .tier = result.tier,
        .vendor_id = result.vendor_id,
        .device_id = result.device_id,
        .subgroup_size = result.subgroup_size,
    };
    slot.* = ctx;
    return SLZ_SUCCESS;
}

pub export fn slzDestroy_vk(handle: ?*VkContext) void {
    if (handle) |h| {
        allocator.destroy(h);
    }
    // The shared driver bundle is intentionally NOT torn down per-handle;
    // teardown happens at process exit (or via a future explicit
    // `slzShutdown_vk`). Multiple handles share one VkDevice in M4.
}

// ── Buffer registration ───────────────────────────────────────────────
// Tier-1 (BDA): both calls are no-ops returning SLZ_SUCCESS. Tier-2
// (M6+) will record the (VkBuffer, VkDeviceAddress, size) mapping so
// dispatchers can resolve a caller-supplied device address into a
// descriptor-set binding without re-querying the loader. The signature
// is stable now so portable callers can wire register/unregister calls
// even before Tier-2 is live.
pub export fn slzRegisterBuffer_vk(
    handle: ?*VkContext,
    vk_buffer_handle: ?*anyopaque,
    d_base_address: ?*const anyopaque,
    buffer_size: usize,
) c_int {
    _ = vk_buffer_handle;
    _ = d_base_address;
    _ = buffer_size;
    _ = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return SLZ_SUCCESS;
}

pub export fn slzUnregisterBuffer_vk(
    handle: ?*VkContext,
    d_base_address: ?*const anyopaque,
) c_int {
    _ = d_base_address;
    _ = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return SLZ_SUCCESS;
}

// ── Sync compress / decompress (STUBBED) ──────────────────────────────
// All return SLZ_ERROR_UNSUPPORTED until the per-kernel milestones
// (Wave 1: M11..M21 decode; Wave 2: M22..M27 encode) wire the pipelines.
// Returns `int` (not slzStatus_t) to match the header — the byte count
// surface mirrors the CUDA contract once implemented.

/// v1 device-pointer variant: dereferences `d_input` and `d_output`
/// directly as host pointers — i.e. the caller is expected to pass
/// already-mapped or unified-memory pointers. True device-to-device
/// dispatch via `slzMakeDeviceOnlyHandle_vk` sentinels is Phase 2
/// (A2 — BDA wiring); today a sentinel pointer is rejected with
/// SLZ_ERROR_UNSUPPORTED so the caller learns the codec hasn't been
/// taught the device-address path yet. Behaviour is otherwise
/// identical to `slzCompressHost_vk`.
pub export fn slzCompress_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    // Phase 1: reject device-only sentinel pointers — the codec
    // doesn't have a BDA path yet, so dereferencing the sentinel as
    // a host pointer would fault. Phase 2 lifts this guard and
    // routes the call through the BDA codec.
    if (isDeviceOnlySentinel(d_input) or isDeviceOnlySentinel(d_output))
        return SLZ_ERROR_UNSUPPORTED;
    return slzCompressHost_vk(handle, d_input, input_size, d_output, output_capacity, opts);
}

pub export fn slzCompressHost_vk(
    handle: ?*VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    if (input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    if (output == null) return SLZ_ERROR_INVALID_ARG;
    if (opts.level != 1) return SLZ_ERROR_UNSUPPORTED;

    const src_ptr: [*]const u8 = @ptrCast(input.?);
    const src: []const u8 = src_ptr[0..input_size];
    const out_ptr: [*]u8 = @ptrCast(output.?);
    const out: []u8 = out_ptr[0..output_capacity];

    const io = ensureIo();
    // Reset the prior decode timing — the caller's most-recent call
    // becomes the source of truth for `slzGetLastTimings_vk`.
    l1_codec.last_encode_dispatch_ns = 0;
    const written = slz1_codec.encodeL1ToSlz1(&driver.g_default, io, allocator, src, out) catch |err|
        return mapEncodeError(err);
    // Snapshot the dispatch ns into the handle. The l1_codec globals
    // are racy across handles (two calls on different handles will
    // clobber each other) — the per-handle snapshot taken here keeps
    // each handle's `slzGetLastTimings_vk` self-consistent.
    h.last_encode_dispatch_ns = l1_codec.last_encode_dispatch_ns;
    return @intCast(written);
}

/// v1 device-pointer variant: dereferences `d_input` and `d_output`
/// directly as host pointers. Device-only sentinels from
/// `slzMakeDeviceOnlyHandle_vk` are rejected with SLZ_ERROR_UNSUPPORTED
/// until Phase 2 wires the BDA path.
pub export fn slzDecompress_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    if (isDeviceOnlySentinel(d_input) or isDeviceOnlySentinel(d_output))
        return SLZ_ERROR_UNSUPPORTED;
    return slzDecompressHost_vk(handle, d_input, input_size, d_output, output_capacity, opts);
}

pub export fn slzDecompressHost_vk(
    handle: ?*VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    _ = opts;
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    if (input == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
    if (output == null and output_capacity != 0) return SLZ_ERROR_INVALID_ARG;

    const src_ptr: [*]const u8 = @ptrCast(input.?);
    const src: []const u8 = src_ptr[0..input_size];
    const out_ptr: [*]u8 = @ptrCast(output.?);
    const out: []u8 = out_ptr[0..output_capacity];

    const io = ensureIo();
    // Same per-handle snapshot pattern as `slzCompressHost_vk` — the
    // l1_codec global is reset before the call and the resulting
    // dispatch ns is captured into the handle after the call returns.
    l1_codec.last_decode_dispatch_ns = 0;
    const written = slz1_codec.decodeSlz1ToBytes(&driver.g_default, io, allocator, src, out) catch |err|
        return mapDecodeError(err);
    h.last_decode_dispatch_ns = l1_codec.last_decode_dispatch_ns;
    return @intCast(written);
}

pub export fn slzCompressBound_vk(input_size: usize) usize {
    // Loose upper bound covering the worst-case L1-encoded SLZ1 frame:
    // ~2x stream expansion (spec §4.1) plus per-chunk headers and a
    // global 8 KiB pad. Matches the formula in `slz1_codec.slz1Bound`;
    // see that comment for the breakdown.
    return slz1_codec.slz1Bound(input_size);
}

// ── Error mapping helpers ─────────────────────────────────────────────

fn mapEncodeError(err: slz1_codec.Slz1Error) c_int {
    return switch (err) {
        error.OutputTooSmall => SLZ_ERROR_BUFFER_TOO_SMALL,
        error.OutOfMemory => SLZ_ERROR_OUT_OF_MEMORY,
        error.UnsupportedTier => SLZ_ERROR_VK_FEATURE_MISSING,
        error.NoSpvForTier => SLZ_ERROR_VK_FEATURE_MISSING,
        else => SLZ_ERROR_UNSUPPORTED,
    };
}

fn mapDecodeError(err: slz1_codec.Slz1Error) c_int {
    return switch (err) {
        error.OutputTooSmall => SLZ_ERROR_BUFFER_TOO_SMALL,
        error.OutOfMemory => SLZ_ERROR_OUT_OF_MEMORY,
        error.UnsupportedTier => SLZ_ERROR_VK_FEATURE_MISSING,
        error.NoSpvForTier => SLZ_ERROR_VK_FEATURE_MISSING,
        error.BadMagic,
        error.UnsupportedVersion,
        error.BadCodec,
        error.BadBlockSize,
        error.BadScGroupSize,
        error.BadInternalHeader,
        error.BadChunkHeader,
        error.BadSubChunkHeader,
        error.MissingContentSize,
        error.Truncated,
        error.TooManyChunks,
        error.BadFrame,
        => SLZ_ERROR_CORRUPT_FRAME,
        else => SLZ_ERROR_UNSUPPORTED,
    };
}

/// Build a sentinel pointer the codec recognizes as "this argument is
/// already a device-resident buffer; skip the H2D copy and consume the
/// pointer as a device address." Phase 1: returns the sentinel address
/// (`0x10`) but the codec rejects it — calling `slzCompress_vk` /
/// `slzDecompress_vk` with the sentinel currently returns
/// SLZ_ERROR_UNSUPPORTED. Phase 2 wires the actual BDA path. The
/// sentinel address is intentionally non-dereferenceable so any path
/// that forgets the contract and reads through it faults loudly
/// instead of corrupting silently (matches the CUDA backend's
/// `device_only_host_stub_addr` pattern in `src/streamlz_gpu.zig`).
pub export fn slzMakeDeviceOnlyHandle_vk(
    out_handle: ?*?*const anyopaque,
    bytes: usize,
) c_int {
    _ = bytes;
    const slot = out_handle orelse return SLZ_ERROR_INVALID_ARG;
    const sentinel: *const anyopaque = @ptrFromInt(device_only_sentinel_addr);
    slot.* = sentinel;
    return SLZ_SUCCESS;
}

// ── Async / polling ───────────────────────────────────────────────────
// Worker-thread + flag scheme (mirrors `runOnWorker` in the CUDA
// backend, minus the per-call thread spawn — the Vulkan path keeps the
// worker alive until polled and joined). Synchronization is one atomic
// store/load on `done`; the rest of the slot is owned by the worker
// until `done == true` and by the polling caller thereafter.

/// Stack-size for async workers. Mirrors `worker_stack_size = 32 MiB`
/// in `src/streamlz_gpu.zig` — the L1 codec orchestration plus host-
/// side wire-format wrap allocates multi-MB frames; this keeps the
/// caller's thread choice (worker-pool, libuv, ...) safe.
const async_worker_stack: usize = 32 << 20;

pub export fn slzCompressAsync_vk(
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
    if (opts.level != 1) return SLZ_ERROR_UNSUPPORTED;
    // Phase 1: no BDA path yet — reject the device-only sentinel
    // explicitly so the caller doesn't get a worker thread launched
    // against an unreadable host pointer. Phase 2 makes this a real
    // device-address dispatch.
    if (isDeviceOnlySentinel(d_input) or isDeviceOnlySentinel(d_output))
        return SLZ_ERROR_UNSUPPORTED;
    // Refuse to start a second op while one is already in flight —
    // the slot owns the worker thread and stomping on it leaks the
    // join.
    if (h.async_enc.isBusy()) return SLZ_ERROR_UNSUPPORTED;
    // A previous completed op might still be sitting in the slot with
    // `done = true`. Clear before reuse — the caller is expected to
    // have read the result via Poll(blocking=1).
    h.async_enc = .{ .compressed_size_out = compressed_size };

    const args = allocator.create(AsyncCompressArgs) catch return SLZ_ERROR_OUT_OF_MEMORY;
    args.* = .{
        .handle = h,
        .input = d_input,
        .input_size = input_size,
        .output = d_output,
        .output_capacity = output_capacity,
        .opts = opts,
    };
    const t = std.Thread.spawn(
        .{ .stack_size = async_worker_stack },
        asyncCompressWorker,
        .{args},
    ) catch {
        allocator.destroy(args);
        h.async_enc = .{};
        return SLZ_ERROR_OUT_OF_MEMORY;
    };
    h.async_enc.thread = t;
    return SLZ_SUCCESS;
}

pub export fn slzCompressAsyncPoll_vk(handle: ?*VkContext, blocking: c_int) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return pollSlot(&h.async_enc, blocking);
}

pub export fn slzDecompressAsync_vk(
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
    if (isDeviceOnlySentinel(d_input) or isDeviceOnlySentinel(d_output))
        return SLZ_ERROR_UNSUPPORTED;
    if (h.async_dec.isBusy()) return SLZ_ERROR_UNSUPPORTED;
    h.async_dec = .{};

    const args = allocator.create(AsyncDecompressArgs) catch return SLZ_ERROR_OUT_OF_MEMORY;
    args.* = .{
        .handle = h,
        .input = d_input,
        .input_size = input_size,
        .output = d_output,
        .output_capacity = output_capacity,
        .opts = opts,
    };
    const t = std.Thread.spawn(
        .{ .stack_size = async_worker_stack },
        asyncDecompressWorker,
        .{args},
    ) catch {
        allocator.destroy(args);
        h.async_dec = .{};
        return SLZ_ERROR_OUT_OF_MEMORY;
    };
    h.async_dec.thread = t;
    return SLZ_SUCCESS;
}

pub export fn slzDecompressAsyncPoll_vk(handle: ?*VkContext, blocking: c_int) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    return pollSlot(&h.async_dec, blocking);
}

/// Implementation shared by both Poll exports. Returns:
///   SLZ_SUCCESS (0)            if the slot is idle or the op completed
///                              successfully (the caller drained the
///                              result; the slot is reset).
///   SLZ_ERROR_UNSUPPORTED      if `blocking == 0` and the op is still
///                              in flight — translates to nvCOMP's
///                              "not ready" sentinel for the
///                              non-blocking caller's poll loop.
///   any other negative code    if the op completed with an error;
///                              the slot is reset and the caller can
///                              re-arm.
fn pollSlot(slot: *AsyncSlot, blocking: c_int) c_int {
    if (slot.thread == null) {
        // No op in flight — idempotent success matches the CUDA
        // backend's behaviour for a polled-but-never-armed handle.
        return SLZ_SUCCESS;
    }
    // Non-blocking poll: peek `done` and bail if not ready. The
    // memory ordering matches the worker's `release` store; we use
    // `acquire` so the rest of the slot (`result`, `written`) is
    // visible once we observe `done == true`.
    if (blocking == 0) {
        if (!@atomicLoad(bool, &slot.done, .acquire)) return SLZ_ERROR_UNSUPPORTED;
    }
    // Blocking poll OR done-already: join the worker, capture the
    // result, reset the slot.
    if (slot.thread) |t| {
        t.join();
        slot.thread = null;
    }
    const rc = slot.result;
    // Leave `written` reset for the next op. The caller already has
    // it (the worker also wrote it into `compressed_size_out` for
    // encode); the decode path's byte count is reported via the
    // poll's return value directly (which is `written` when rc >= 0).
    const out_rc: c_int = if (rc >= 0) @intCast(slot.written) else rc;
    slot.* = .{};
    return out_rc;
}

// ── Per-kernel timings drain ──────────────────────────────────────────
// Translates the per-handle `last_{encode,decode}_dispatch_ns`
// snapshot captured by the sync entry points into the
// `KernelTimingC` array the C caller drains. Names mirror the
// production GLSL shader entry points (`lz_encode`, `lz_decode`); a
// 0-ns slot is suppressed so the caller never sees a phantom kernel
// that didn't actually run on the most-recent call.
pub export fn slzGetLastTimings_vk(
    handle: ?*VkContext,
    out: ?[*]KernelTimingC,
    capacity: usize,
) usize {
    const h = handle orelse return 0;
    // Build candidate list (name + ns), then translate to ms.
    var candidates: [2]struct { name: [*:0]const u8, ns: u64 } = .{
        .{ .name = "lz_encode", .ns = h.last_encode_dispatch_ns },
        .{ .name = "lz_decode", .ns = h.last_decode_dispatch_ns },
    };

    var total: usize = 0;
    for (&candidates) |c| {
        if (c.ns == 0) continue;
        if (out) |buf| {
            if (total < capacity) {
                const ms_f: f32 = @as(f32, @floatFromInt(c.ns)) / 1_000_000.0;
                buf[total] = .{ .name = c.name, .ms = ms_f };
            }
        }
        total += 1;
    }
    return total;
}
