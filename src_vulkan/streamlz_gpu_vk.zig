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

const allocator = std.heap.c_allocator;

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
const CompressOpts = extern struct {
    level: c_int = 5,
    enable_profiling: c_int = 0,
    effective_level_out: c_int = 0,
    reserved: [5]c_int = @splat(0),
};

const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = @splat(0),
};

const KernelTimingC = extern struct {
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
};

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

pub export fn slzCompress_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    _ = handle;
    _ = d_input;
    _ = input_size;
    _ = d_output;
    _ = output_capacity;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzCompressHost_vk(
    handle: ?*VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: CompressOpts,
) c_int {
    _ = handle;
    _ = input;
    _ = input_size;
    _ = output;
    _ = output_capacity;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzDecompress_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    _ = handle;
    _ = d_input;
    _ = input_size;
    _ = d_output;
    _ = output_capacity;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzDecompressHost_vk(
    handle: ?*VkContext,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    _ = handle;
    _ = input;
    _ = input_size;
    _ = output;
    _ = output_capacity;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzCompressBound_vk(input_size: usize) usize {
    // Mirrors the CUDA-side `slzCompressBound` formula at high level —
    // worst-case is bounded by `encoder.compressBound`. M4 returns a
    // safe overestimate (input_size + 8 KiB headers + 6.25 % expansion
    // pad). The encode wave-2 milestones replace this with the exact
    // shared formula from `src/encode/streamlz_encoder.zig`.
    const expansion_pad: usize = input_size / 16; // 6.25 %
    const fixed_header_pad: usize = 8 * 1024;
    return input_size + expansion_pad + fixed_header_pad;
}

pub export fn slzMakeDeviceOnlyHandle_vk(
    out_handle: ?*?*const anyopaque,
    bytes: usize,
) c_int {
    _ = bytes;
    if (out_handle) |s| s.* = null;
    return SLZ_ERROR_UNSUPPORTED;
}

// ── Async / polling (STUBBED) ─────────────────────────────────────────
pub export fn slzCompressAsync_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    compressed_size: ?*usize,
    opts: CompressOpts,
) c_int {
    _ = handle;
    _ = d_input;
    _ = input_size;
    _ = d_output;
    _ = output_capacity;
    _ = compressed_size;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzCompressAsyncPoll_vk(handle: ?*VkContext, blocking: c_int) c_int {
    _ = handle;
    _ = blocking;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzDecompressAsync_vk(
    handle: ?*VkContext,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    output_capacity: usize,
    opts: DecompressOpts,
) c_int {
    _ = handle;
    _ = d_input;
    _ = input_size;
    _ = d_output;
    _ = output_capacity;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

pub export fn slzDecompressAsyncPoll_vk(handle: ?*VkContext, blocking: c_int) c_int {
    _ = handle;
    _ = blocking;
    return SLZ_ERROR_UNSUPPORTED;
}

// ── Per-kernel timings drain (STUBBED to 0 entries in M4) ─────────────
pub export fn slzGetLastTimings_vk(
    handle: ?*VkContext,
    out: ?[*]KernelTimingC,
    capacity: usize,
) usize {
    _ = handle;
    _ = out;
    _ = capacity;
    return 0;
}
