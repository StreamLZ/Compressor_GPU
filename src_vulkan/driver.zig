//! Vulkan driver facade — mirrors src/decode/driver.zig's role for the
//! CUDA pipeline. Owns the process-singleton `g_default` Context that
//! every later milestone's encode/decode dispatcher will read.
//!
//! `ensureInit` is idempotent: it loads vulkan-1.dll, creates a VkInstance
//! (with optional validation), picks the first compute-capable physical
//! device, creates a logical device + queue, and stashes everything on
//! `g_default`. `deinit` tears the bundle down in reverse order.

const std = @import("std");

const vk = @import("vk_api.zig");
const instance_mod = @import("instance.zig");
const device_mod = @import("device.zig");
const dispatch_mod = @import("dispatch.zig");
const probe_mod = @import("probe.zig");

pub const Context = struct {
    inst: vk.VkInstance = null,
    pd: vk.VkPhysicalDevice = null,
    dev: vk.VkDevice = null,
    queue: vk.VkQueue = null,
    qfi: u32 = 0,
    initialized: bool = false,

    // ── M8a: lazy-allocated dispatch + timing chassis ────────────────
    // Created on first `dispatch.submitOne` call so contexts that never
    // dispatch (probe-only callers, error paths between createInstance
    // and createDevice) don't pay the pool/fence/query-pool cost.
    // `dispatch.zig` is the sole owner — these fields are mutated only
    // from there. `deinit` calls `dispatch.releaseContextChassis` to
    // tear them down in the correct order (free cmdbuf → destroy pool +
    // fence + query pool) before destroyDevice.
    cmd_pool: vk.VkCommandPool = null,
    cmd_buf: vk.VkCommandBuffer = null,
    fence: vk.VkFence = null,
    query_pool: vk.VkQueryPool = null,
    timestamp_period_ns: f32 = 0.0,

    // ── M8c: probed sync2 capability (read-only after ensureInit) ────
    // sync.zig consults this to pick `vkCmdPipelineBarrier2` vs the
    // sync1 fallback. Populated from `probe.probe()` during ensureInit.
    // Default false keeps the sync1 fallback path safe on Contexts that
    // were never brought up via ensureInit (test code, error paths).
    has_synchronization2: bool = false,
};

pub var g_default: Context = .{};

pub const DriverError = error{
    LoaderInitFailed,
} || instance_mod.InstanceError || device_mod.DeviceError;

/// Bring up the full Vulkan bootstrap into `g_default`. Safe to call any
/// number of times — subsequent calls after the first success are no-ops.
pub fn ensureInit() DriverError!void {
    if (g_default.initialized) return;

    if (!vk.init()) return error.LoaderInitFailed;

    // Validation default: off. Caller (CLI flag, env var elsewhere) flips
    // the want_validation argument; instance.zig still requires SLZ_VK_
    // VALIDATION=1 in the env as a belt-and-suspenders.
    const inst = try instance_mod.createInstance(true);
    errdefer instance_mod.destroyInstance(inst);

    const pd = try device_mod.pickPhysicalDevice(inst);
    const bundle = try device_mod.createDevice(pd);
    errdefer device_mod.destroyDevice(bundle.dev);

    // Probe the device once at bring-up so the M8c sync wrapper can read
    // `has_synchronization2` off the Context without re-running the full
    // pNext-chain query on every barrier emission. Other tier-classifying
    // call sites still re-probe — this is a single cached scalar.
    const pr = probe_mod.probe(inst, pd);

    g_default = .{
        .inst = inst,
        .pd = pd,
        .dev = bundle.dev,
        .queue = bundle.queue,
        .qfi = bundle.queue_family_index,
        .initialized = true,
        .has_synchronization2 = pr.has_synchronization2,
    };
}

pub fn deinit() void {
    if (!g_default.initialized) return;
    // M8a chassis teardown must happen BEFORE destroyDevice — command
    // pools, fences, and query pools are device-owned and the spec
    // requires they be destroyed before their parent VkDevice.
    dispatch_mod.releaseContextChassis(&g_default);
    device_mod.destroyDevice(g_default.dev);
    instance_mod.destroyInstance(g_default.inst);
    g_default = .{};
}
