//! StreamLZ Vulkan-backend C ABI.
//!
//! Implements the contract in `include/streamlz_gpu_vk.h`: a handle-based
//! Vulkan compressor exported as `streamlz_vk.dll` (+ `streamlz_vk.lib`).
//! Mirrors the shape of `src/streamlz_gpu.zig`.
//!
//! Wired entry points (level=1 only):
//!   - slzCreate_vk / slzDestroy_vk           — driver bring-up + tier probe.
//!   - slzGetVersionString_vk                 — version.string + "+vk".
//!   - slzRegisterBuffer_vk / Unregister_vk   — D2D registry for
//!                                              BDA-backed VkBuffer
//!                                              arguments to {Compress,
//!                                              Decompress}_vk.
//!   - slzBufferGetDeviceAddress_vk           — public BDA query helper.
//!   - slzCompress_vk / slzCompressHost_vk    — synchronous L1 encode +
//!                                              wire-format wrap. Device-
//!                                              pointer variant resolves
//!                                              registered VkBuffer args
//!                                              and skips the host bounce.
//!   - slzDecompress_vk / slzDecompressHost_vk — synchronous L1 decode of
//!                                              an SLZ1 frame; same D2D
//!                                              registry resolution.
//!   - slzCompressBound_vk                    — worst-case .slz size for
//!                                              an L1-encoded payload.
//!   - slzCompressAsync_vk / Poll, decode counterparts — worker-thread
//!                                              wrappers around the sync
//!                                              entry points.
//!   - slzGetLastTimings_vk                   — per-handle kernel-ns
//!                                              snapshot drain.
//!
//! Always-error entry point:
//!   - slzMakeDeviceOnlyHandle_vk             — returns
//!                                              SLZ_ERROR_UNSUPPORTED.
//!                                              The CUDA backend uses
//!                                              `device_only_host_stub_addr`
//!                                              (src/streamlz_gpu.zig:44)
//!                                              as an INTERNAL sentinel
//!                                              for async-D2D inputs; it
//!                                              is never exposed via the
//!                                              CUDA C ABI. The Vulkan
//!                                              header reserves the symbol
//!                                              for ABI compat but there
//!                                              is no wired D2D-only
//!                                              path the sentinel could
//!                                              flow into — every codec
//!                                              entry point that saw the
//!                                              old `0x10` sentinel
//!                                              rejected it as
//!                                              UNSUPPORTED. Returning
//!                                              the error directly tells
//!                                              callers honestly rather
//!                                              than handing back a
//!                                              sentinel that nothing
//!                                              accepts.

const std = @import("std");

const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const slz1_codec = @import("slz1_codec.zig");
const l1_codec = @import("l1_codec.zig");
const vk = @import("vk_api.zig");

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

    /// D2D buffer registry. Phase 2 (TODO A2) — `slzRegisterBuffer_vk`
    /// adds (device-address u64, VkBuffer, size) tuples here so the
    /// codec can look up the caller's VkBuffer when given the address
    /// as the `d_input` / `d_output` argument to `slzCompress_vk` or
    /// `slzDecompress_vk`. A flat slot array (no hash map) is fine —
    /// any realistic D2D caller registers a handful of buffers per
    /// handle (input + output + maybe a frame staging buffer); a
    /// linear scan is faster than a map at that size.
    registry: [MAX_REGISTERED]RegisteredBuffer = @splat(.{}),
    registry_count: u32 = 0,
};

/// Cap on registered D2D buffers per handle. Bumped if real-world
/// callers register more than this; 16 covers every test/bench shape
/// we have today.
pub const MAX_REGISTERED: usize = 16;

pub const RegisteredBuffer = struct {
    address: u64 = 0,
    buffer: vk.VkBuffer = null,
    size: usize = 0,
};

/// Look up a registered buffer by address. Returns null if the address
/// is zero, has never been registered, or has since been unregistered.
fn lookupRegistered(h: *VkContext, addr: u64) ?RegisteredBuffer {
    if (addr == 0) return null;
    var i: u32 = 0;
    while (i < h.registry_count) : (i += 1) {
        if (h.registry[i].address == addr) return h.registry[i];
    }
    return null;
}

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

// The Vulkan backend does NOT implement a host-sentinel-for-device
// argument pattern. The CUDA backend uses
// `device_only_host_stub_addr = 0x10` (src/streamlz_gpu.zig:44) but only
// as an INTERNAL sentinel inside its async compress path — it is never
// exposed via the CUDA C ABI. The Vulkan backend's D2D path resolves
// caller VkBuffer arguments through the `slzRegisterBuffer_vk` registry
// (BDA u64 lookup), not via a host-sentinel pointer. The
// `slzMakeDeviceOnlyHandle_vk` ABI symbol survives only to preserve the
// C-header ABI surface; its implementation returns SLZ_ERROR_UNSUPPORTED
// (see below).

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

// ── Buffer registration (Phase 2 / TODO A2 — true D2D via BDA) ────────
// Caller supplies a VkBuffer (cast to `void*`) plus optional pre-
// queried device address. The function records (address, VkBuffer,
// size) on the handle's registry so the codec can resolve a
// caller-supplied device address into the corresponding VkBuffer
// when invoked via `slzCompress_vk` / `slzDecompress_vk`.
//
// `d_base_address` semantics:
//   * non-null  — caller already queried `vkGetBufferDeviceAddress`
//                 and is asserting the address. We trust them.
//   * null      — we call vkGetBufferDeviceAddress on the supplied
//                 VkBuffer ourselves. The caller can read the result
//                 back via the companion `slzBufferGetDeviceAddress_vk`
//                 helper below; without that, the registry mapping
//                 is opaque and the caller has nothing to pass back.
//                 This branch is mainly here for symmetry with the
//                 CUDA-side `slzRegisterBuffer` (where the address
//                 is the caller's CUDA-side allocation pointer and
//                 always known).
//
// The VkBuffer MUST be created with
// `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT` (Phase 2 enables
// `bufferDeviceAddress` at device-create time in `device.zig`).
pub export fn slzRegisterBuffer_vk(
    handle: ?*VkContext,
    vk_buffer_handle: ?*anyopaque,
    d_base_address: ?*const anyopaque,
    buffer_size: usize,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const vk_buf_raw = vk_buffer_handle orelse return SLZ_ERROR_INVALID_ARG;
    if (buffer_size == 0) return SLZ_ERROR_INVALID_ARG;
    if (h.registry_count >= MAX_REGISTERED) return SLZ_ERROR_OUT_OF_MEMORY;

    const vk_buf: vk.VkBuffer = @ptrCast(vk_buf_raw);

    // Resolve address: either trust the caller (CUDA-symmetry path)
    // or query via vkGetBufferDeviceAddress.
    const addr: u64 = if (d_base_address) |p|
        @intFromPtr(p)
    else blk: {
        const a = queryBufferAddress(vk_buf) catch return SLZ_ERROR_VK_FEATURE_MISSING;
        if (a == 0) return SLZ_ERROR_VK_FEATURE_MISSING;
        break :blk a;
    };

    h.registry[h.registry_count] = .{
        .address = addr,
        .buffer = vk_buf,
        .size = buffer_size,
    };
    h.registry_count += 1;
    return SLZ_SUCCESS;
}

/// Removes an address from the registry. Idempotent — unregistering
/// an address that is not registered returns SLZ_SUCCESS.
pub export fn slzUnregisterBuffer_vk(
    handle: ?*VkContext,
    d_base_address: ?*const anyopaque,
) c_int {
    const h = handle orelse return SLZ_ERROR_INVALID_HANDLE;
    const addr_raw = d_base_address orelse return SLZ_SUCCESS;
    const addr = @intFromPtr(addr_raw);
    var i: u32 = 0;
    while (i < h.registry_count) : (i += 1) {
        if (h.registry[i].address == addr) {
            // Swap-with-last so the registry stays compact. Order is
            // not exposed to callers so this is safe.
            const last = h.registry_count - 1;
            if (i != last) h.registry[i] = h.registry[last];
            h.registry[last] = .{};
            h.registry_count = last;
            return SLZ_SUCCESS;
        }
    }
    return SLZ_SUCCESS;
}

/// Query the device address of a VkBuffer. Internal helper used by
/// `slzRegisterBuffer_vk` when the caller didn't pre-query, AND
/// exposed publicly (below) so test code can call it without
/// importing all of vk_api.
fn queryBufferAddress(buf: vk.VkBuffer) !u64 {
    // Lazy resolve.
    if (vk.vkGetBufferDeviceAddress_fn == null) {
        if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
            if (gdpa(driver.g_default.dev, "vkGetBufferDeviceAddress")) |raw| {
                vk.vkGetBufferDeviceAddress_fn = @ptrCast(@alignCast(raw));
            }
        }
        // Some drivers only expose the KHR-suffixed name on Vulkan
        // 1.1 hosts; the Tier floor here is 1.2 so the core name is
        // always present, but try the KHR alias as a belt.
        if (vk.vkGetBufferDeviceAddress_fn == null) {
            if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
                if (gdpa(driver.g_default.dev, "vkGetBufferDeviceAddressKHR")) |raw| {
                    vk.vkGetBufferDeviceAddress_fn = @ptrCast(@alignCast(raw));
                }
            }
        }
    }
    const get = vk.vkGetBufferDeviceAddress_fn orelse return error.NoBdaFn;
    const info: vk.VkBufferDeviceAddressInfo = .{ .buffer = buf };
    return get(driver.g_default.dev, &info);
}

/// Public helper: query the device address of an already-created
/// VkBuffer. Caller passes the VkBuffer handle cast to `void*`;
/// returns the BDA u64 or 0 on failure. Surfaced through the C ABI
/// so test code can build BDA-typed D2D round-trips without pulling
/// in vk_api directly.
pub export fn slzBufferGetDeviceAddress_vk(
    handle: ?*VkContext,
    vk_buffer_handle: ?*anyopaque,
) u64 {
    _ = handle orelse return 0;
    const buf: vk.VkBuffer = @ptrCast(vk_buffer_handle orelse return 0);
    return queryBufferAddress(buf) catch 0;
}

// ── Sync compress / decompress (STUBBED) ──────────────────────────────
// All return SLZ_ERROR_UNSUPPORTED until the per-kernel milestones
// (Wave 1: M11..M21 decode; Wave 2: M22..M27 encode) wire the pipelines.
// Returns `int` (not slzStatus_t) to match the header — the byte count
// surface mirrors the CUDA contract once implemented.

/// Device-pointer variant: if `d_input` is the device address of a
/// previously-registered VkBuffer (via `slzRegisterBuffer_vk`), the
/// encoder binds that buffer directly into the descriptor set and
/// skips its internal HOST_VISIBLE staging + memcpy. Otherwise the
/// call falls through to `slzCompressHost_vk` (treats the pointer
/// as a host pointer).
///
/// `d_output` is always host-side for compress today — the L1
/// encoder's output goes through the CPU wire-format wrap before
/// landing in `d_output`. Registering `d_output` as a device buffer
/// is accepted (we copy the wrapped frame into it via staging) for
/// API symmetry but is NOT a perf win at this phase.
pub export fn slzCompress_vk(
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
    if (opts.level != 1) return SLZ_ERROR_UNSUPPORTED;

    // Look up the input address in the registry. If found, branch
    // through the D2D path; otherwise treat as a host pointer and
    // delegate to the host-pointer entry point.
    const in_addr: u64 = @intFromPtr(d_input);
    const out_addr: u64 = @intFromPtr(d_output);
    const in_reg = lookupRegistered(h, in_addr);
    const out_reg = lookupRegistered(h, out_addr);
    if (in_reg == null and out_reg == null) {
        return slzCompressHost_vk(handle, d_input, input_size, d_output, output_capacity, opts);
    }

    // D2D path. The host-side wire-format wrap still runs in CPU,
    // so we always need a host buffer for the compressed frame. If
    // d_output is a registered device buffer, we encode into a host
    // scratch buffer then stage the result into the device buffer.
    return d2dCompress(
        h,
        in_reg,
        d_input,
        input_size,
        out_reg,
        d_output,
        output_capacity,
        opts,
    );
}

/// D2D-aware compress backend. Builds the codec options from the
/// registered-buffer lookup and either bounces the output through a
/// host scratch (when d_output is device-resident) or writes
/// straight to the caller's host pointer.
fn d2dCompress(
    h: *VkContext,
    in_reg: ?RegisteredBuffer,
    d_input_host: ?*const anyopaque,
    input_size: usize,
    out_reg: ?RegisteredBuffer,
    d_output_host: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    _ = opts;
    // Phase-2 perf win lives in the src side: skip the HOST_VISIBLE
    // staging + memcpy when the caller's source is already on
    // device. The encoder kernel then reads the caller's VkBuffer
    // directly via descriptor binding. BUT the CPU wire-format
    // wrap downstream of the encoder also needs the source bytes
    // (the SC tail prefix copies 8 verbatim bytes per non-first
    // chunk from the source into the output frame). For Phase 2
    // we satisfy that by reading the bytes back from the device
    // buffer into a host scratch before encode. Net effect: we
    // trade one H2D for one D2H — perf-neutral for now but the
    // path lets the encoder kernel skip the descriptor's H2D wait
    // and Phase 3 (GPU frame assembly) drops the D2H entirely.
    const src_buffer_override: ?vk.VkBuffer = if (in_reg) |r| r.buffer else null;
    var src_view: []const u8 = undefined;
    var d2h_scratch: ?[]u8 = null;
    defer if (d2h_scratch) |b| allocator.free(b);
    if (in_reg) |reg| {
        if (input_size > reg.size) return SLZ_ERROR_INVALID_ARG;
        const scratch = allocator.alloc(u8, input_size) catch return SLZ_ERROR_OUT_OF_MEMORY;
        d2h_scratch = scratch;
        if (!stageBytesFromDeviceBuffer(reg.buffer, scratch))
            return SLZ_ERROR_VK_FEATURE_MISSING;
        src_view = scratch;
    } else {
        if (d_input_host == null and input_size != 0) return SLZ_ERROR_INVALID_ARG;
        const ptr: [*]const u8 = @ptrCast(d_input_host.?);
        src_view = ptr[0..input_size];
    }

    // Output side: if the caller's output is a registered device
    // buffer, encode into a host scratch and stage the result over
    // to the device buffer via vkCmdCopyBuffer. The bound size
    // (caller's registered_size) caps capacity.
    if (out_reg) |reg| {
        if (output_capacity == 0) return SLZ_ERROR_INVALID_ARG;
        const cap = @min(output_capacity, reg.size);
        const scratch = allocator.alloc(u8, cap) catch return SLZ_ERROR_OUT_OF_MEMORY;
        defer allocator.free(scratch);

        const io = ensureIo();
        l1_codec.last_encode_dispatch_ns = 0;
        const written_or_err = slz1_codec.encodeL1ToSlz1Ex(
            &driver.g_default,
            io,
            allocator,
            src_view,
            scratch,
            .{ .src_buffer_override = src_buffer_override },
        );
        const written: usize = written_or_err catch |err| return mapEncodeError(err);
        h.last_encode_dispatch_ns = l1_codec.last_encode_dispatch_ns;

        // Stage scratch → caller's device buffer. Cheap CPU-side
        // copy through a host-visible staging buffer then a
        // GPU-side copy into the registered buffer.
        const ok = stageBytesToDeviceBuffer(reg.buffer, scratch[0..written]);
        if (!ok) return SLZ_ERROR_OUT_OF_MEMORY;
        return @intCast(written);
    }

    // Output is a host pointer; encode directly into it.
    if (d_output_host == null) return SLZ_ERROR_INVALID_ARG;
    const out_ptr: [*]u8 = @ptrCast(d_output_host.?);
    const out: []u8 = out_ptr[0..output_capacity];

    const io = ensureIo();
    l1_codec.last_encode_dispatch_ns = 0;
    const written = slz1_codec.encodeL1ToSlz1Ex(
        &driver.g_default,
        io,
        allocator,
        src_view,
        out,
        .{ .src_buffer_override = src_buffer_override },
    ) catch |err| return mapEncodeError(err);
    h.last_encode_dispatch_ns = l1_codec.last_encode_dispatch_ns;
    return @intCast(written);
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

/// Device-pointer variant. If `d_output` is a registered VkBuffer's
/// device address, the decoder writes decoded bytes directly into the
/// caller's VkBuffer without the internal HOST_VISIBLE staging copy +
/// host @memcpy chain. If `d_input` is registered, the CPU
/// wire-format unwrap still runs (the CPU path needs the bytes); we
/// read them back from the device buffer via a staging copy. This is
/// correctness-preserving but not a perf win for input D2D at this
/// phase — Phase 4 (GPU decode pipeline) is the path to real D2D for
/// input.
pub export fn slzDecompress_vk(
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

    const in_addr: u64 = @intFromPtr(d_input);
    const out_addr: u64 = @intFromPtr(d_output);
    const in_reg = lookupRegistered(h, in_addr);
    const out_reg = lookupRegistered(h, out_addr);
    if (in_reg == null and out_reg == null) {
        return slzDecompressHost_vk(handle, d_input, input_size, d_output, output_capacity, opts);
    }
    return d2dDecompress(h, in_reg, d_input, input_size, out_reg, d_output, output_capacity, opts);
}

fn d2dDecompress(
    h: *VkContext,
    in_reg: ?RegisteredBuffer,
    d_input_host: ?*const anyopaque,
    input_size: usize,
    out_reg: ?RegisteredBuffer,
    d_output_host: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    _ = opts;
    // Input: if registered, pull bytes off the device into a host
    // scratch (the CPU wire-format unwrap needs them). For host
    // input, alias the caller's slice.
    var input_slice: []const u8 = undefined;
    var input_owned_scratch: ?[]u8 = null;
    defer if (input_owned_scratch) |b| allocator.free(b);
    if (in_reg) |reg| {
        if (input_size > reg.size) return SLZ_ERROR_INVALID_ARG;
        const scratch = allocator.alloc(u8, input_size) catch return SLZ_ERROR_OUT_OF_MEMORY;
        input_owned_scratch = scratch;
        if (!stageBytesFromDeviceBuffer(reg.buffer, scratch)) return SLZ_ERROR_VK_FEATURE_MISSING;
        input_slice = scratch;
    } else {
        if (d_input_host == null) return SLZ_ERROR_INVALID_ARG;
        const ptr: [*]const u8 = @ptrCast(d_input_host.?);
        input_slice = ptr[0..input_size];
    }

    // Output: if registered, the decoder writes straight into the
    // caller's buffer via dst_buffer_override. For host output,
    // alias the caller's slice as before.
    const dst_buffer_override: ?vk.VkBuffer = if (out_reg) |r| r.buffer else null;
    var dst_view: []u8 = undefined;
    if (out_reg != null) {
        // The codec only writes to the device buffer; pass a zero-
        // length host slice so the unused @memcpy path is skipped.
        dst_view = &.{};
    } else {
        if (d_output_host == null) return SLZ_ERROR_INVALID_ARG;
        const ptr: [*]u8 = @ptrCast(d_output_host.?);
        dst_view = ptr[0..output_capacity];
    }

    const io = ensureIo();
    l1_codec.last_decode_dispatch_ns = 0;
    const written = slz1_codec.decodeSlz1ToBytesEx(
        &driver.g_default,
        io,
        allocator,
        input_slice,
        dst_view,
        .{ .dst_buffer_override = dst_buffer_override },
    ) catch |err| return mapDecodeError(err);
    h.last_decode_dispatch_ns = l1_codec.last_decode_dispatch_ns;
    return @intCast(written);
}

/// Stage `bytes` into a caller-owned device VkBuffer via an
/// internal HOST_VISIBLE staging buffer and a vkCmdCopyBuffer.
/// Returns true on success.
fn stageBytesToDeviceBuffer(dst: vk.VkBuffer, bytes: []const u8) bool {
    // `to_device` direction only reads `host_bytes` — the @memcpy
    // inside `stageDeviceBufferCopy` clones the source bytes into
    // the staging buffer's mapped memory before submit. A
    // `@constCast` is fine here because the helper never writes
    // to the slice in this direction.
    const writable: []u8 = @constCast(bytes);
    return stageDeviceBufferCopy(dst, writable, .to_device);
}

/// Read bytes from a caller-owned device VkBuffer into a host
/// scratch (size = `out.len`). Returns true on success.
fn stageBytesFromDeviceBuffer(src: vk.VkBuffer, out: []u8) bool {
    return stageDeviceBufferCopy(src, out, .from_device);
}

const StageDir = enum { to_device, from_device };

/// Helper shared by stageBytesTo/FromDeviceBuffer. Allocates a
/// HOST_VISIBLE staging buffer the same size as `bytes` and submits
/// a single vkCmdCopyBuffer in either direction. Reuses the
/// existing dispatch chassis on `driver.g_default` so the fence
/// chassis doesn't get duplicated.
fn stageDeviceBufferCopy(dev_buf: vk.VkBuffer, host_bytes: []u8, dir: StageDir) bool {
    if (host_bytes.len == 0) return true;
    const ctx = &driver.g_default;
    l1_codec.ensureBufferFnSlots(ctx);

    // Create the staging buffer.
    const create_buf = vk.vkCreateBuffer_fn orelse return false;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse return false;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return false;
    const free_mem = vk.vkFreeMemory_fn orelse return false;
    const bind = vk.vkBindBufferMemory_fn orelse return false;
    const destroy_buf = vk.vkDestroyBuffer_fn orelse return false;
    const map = vk.vkMapMemory_fn orelse return false;
    const unmap = vk.vkUnmapMemory_fn orelse return false;
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return false;

    const usage: vk.VkBufferUsageFlags = switch (dir) {
        .to_device => vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .from_device => vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    const bci: vk.VkBufferCreateInfo = .{
        .size = host_bytes.len,
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var stage_buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &stage_buf) != vk.VK_SUCCESS) return false;
    defer destroy_buf(ctx.dev, stage_buf, null);

    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, stage_buf, &req);

    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(ctx.pd, &mem_props);
    const want = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    var mt_idx: u32 = std.math.maxInt(u32);
    {
        var i: u32 = 0;
        while (i < mem_props.memoryTypeCount) : (i += 1) {
            const ok = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
            if (ok and (mem_props.memoryTypes[i].propertyFlags & want) == want) {
                mt_idx = i;
                break;
            }
        }
    }
    if (mt_idx == std.math.maxInt(u32)) return false;

    const mai: vk.VkMemoryAllocateInfo = .{
        .allocationSize = req.size,
        .memoryTypeIndex = mt_idx,
    };
    var stage_mem: vk.VkDeviceMemory = null;
    if (alloc_mem(ctx.dev, &mai, null, &stage_mem) != vk.VK_SUCCESS) return false;
    defer free_mem(ctx.dev, stage_mem, null);
    if (bind(ctx.dev, stage_buf, stage_mem, 0) != vk.VK_SUCCESS) return false;

    var raw: ?*anyopaque = null;
    if (map(ctx.dev, stage_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS) return false;
    defer unmap(ctx.dev, stage_mem);
    const mapped: [*]u8 = @ptrCast(@alignCast(raw.?));

    // For to_device, fill the staging buffer; for from_device, read
    // it after the GPU copy completes (post-submit).
    if (dir == .to_device) {
        @memcpy(mapped[0..host_bytes.len], host_bytes);
    }

    // Submit a one-shot vkCmdCopyBuffer using the driver chassis.
    if (!recordAndSubmitCopy(ctx, stage_buf, dev_buf, host_bytes.len, dir)) return false;

    if (dir == .from_device) {
        @memcpy(host_bytes, mapped[0..host_bytes.len]);
    }
    return true;
}

/// Inline cmdbuf record + submit + wait for a single
/// vkCmdCopyBuffer. Reuses the dispatch chassis primary command
/// buffer + fence on `ctx`.
fn recordAndSubmitCopy(
    ctx: *driver.Context,
    stage_buf: vk.VkBuffer,
    dev_buf: vk.VkBuffer,
    size: usize,
    dir: StageDir,
) bool {
    // Ensure the chassis (cmd_pool, cmd_buf, fence, query_pool) is up.
    // dispatch.submitOne does this lazily on first dispatch; the D2D
    // stage-copy path may run before any codec dispatch, so prep it
    // explicitly here.
    const dispatch_mod = @import("dispatch.zig");
    dispatch_mod.ensureChassisPub(ctx) catch return false;

    const reset_cb = vk.vkResetCommandBuffer_fn orelse return false;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return false;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return false;
    const cmd_copy = vk.vkCmdCopyBuffer_fn orelse return false;
    const reset_fence = vk.vkResetFences_fn orelse return false;
    const submit = vk.vkQueueSubmit_fn orelse return false;
    const wait_fence = vk.vkWaitForFences_fn orelse return false;

    // The chassis is set up by `dispatch.submitOne` on first call —
    // we expect at least one codec dispatch has run by this point
    // (the encoder/decoder runs before we get here), so cmd_buf +
    // fence are already populated. If not, bail.
    if (ctx.cmd_buf == null or ctx.fence == null) return false;

    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return false;
    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) return false;
    const region: vk.VkBufferCopy = .{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = @intCast(size),
    };
    const regions: [1]vk.VkBufferCopy = .{region};
    switch (dir) {
        .to_device => cmd_copy(ctx.cmd_buf, stage_buf, dev_buf, 1, @ptrCast(&regions)),
        .from_device => cmd_copy(ctx.cmd_buf, dev_buf, stage_buf, 1, @ptrCast(&regions)),
    }
    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return false;

    const fences: [1]vk.VkFence = .{ctx.fence};
    if (reset_fence(ctx.dev, 1, @ptrCast(&fences)) != vk.VK_SUCCESS) return false;
    const cmd_bufs: [1]vk.VkCommandBuffer = .{ctx.cmd_buf};
    const submit_info: vk.VkSubmitInfo = .{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cmd_bufs),
    };
    const submits: [1]vk.VkSubmitInfo = .{submit_info};
    if (submit(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) return false;
    return wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS) == vk.VK_SUCCESS;
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
        // Cluster B (F004) collapsed the decoder's CPU-unwrap error
        // set into the two device-side failures the GPU pipeline can
        // surface: a frame the GPU couldn't parse (`BadFrame`) and
        // chunk-count overflow (`TooManyChunks`). The fine-grained
        // BadMagic / UnsupportedVersion / BadCodec / etc. used to come
        // from `wire_format.UnwrapError`, which is no longer reachable
        // from the production decode path (the CPU unwrap was deleted
        // in commit `146c5a6`; the toggle that fell back to it was
        // deleted in Cluster B's F004 commit).
        error.BadFrame, error.TooManyChunks => SLZ_ERROR_CORRUPT_FRAME,
        else => SLZ_ERROR_UNSUPPORTED,
    };
}

/// Always returns SLZ_ERROR_UNSUPPORTED. The Vulkan backend exposes
/// device-resident I/O via the `slzRegisterBuffer_vk` registry (the
/// caller passes the BDA u64 as the `d_input`/`d_output` argument and
/// the codec resolves it to the registered VkBuffer). There is no
/// host-sentinel pattern equivalent to the CUDA backend's INTERNAL
/// `device_only_host_stub_addr` (src/streamlz_gpu.zig:44) — that
/// sentinel is not exposed in the CUDA C ABI either, so a CUDA caller
/// using `slzMakeDeviceOnlyHandle` does not exist. The Vulkan
/// `slzMakeDeviceOnlyHandle_vk` ABI symbol is retained for header
/// compatibility but reports unsupported so callers branch on the
/// real D2D path (`slzRegisterBuffer_vk` + BDA address argument).
pub export fn slzMakeDeviceOnlyHandle_vk(
    out_handle: ?*?*const anyopaque,
    bytes: usize,
) c_int {
    _ = bytes;
    if (out_handle) |slot| slot.* = null;
    return SLZ_ERROR_UNSUPPORTED;
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
