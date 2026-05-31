//! M8c: production 68-slot timestamp query pool per architecture §15.
//!
//! Architecture §15: VkQueryPool sized **2 × 17 × in_flight_count = 68
//! slots** where in_flight_count = 2. Each kernel takes a (begin, end)
//! pair at slot index `2 * kernel_idx + 0..1 + 34 * (current_buffer_idx)`.
//! The M8a chassis sized at 2 slots for the bare-minimum dispatch path;
//! this module owns the production-grade 68-slot pool that the per-
//! milestone kernel dispatchers (M11+) will reserve pairs from.
//!
//! Reservation strategy (M8c): a monotonic `next` counter handed out in
//! pairs of 2. When the counter reaches 68 the pool is "full" and the
//! caller must (a) drain pending timings, (b) reset the pool's slots,
//! (c) reset `next` to 0. Architecture-§15 indexing (the kernel_idx +
//! buffer_idx scheme) is the M11+ caller's responsibility — at the
//! pool-management level all this module sees is a flat free counter.
//!
//! No allocations: the pool descriptor is one VkQueryPool handle plus
//! two u32 counters; per-call timing reads are stack-allocated u64[2].

const std = @import("std");

const vk = @import("vk_api.zig");
const driver_mod = @import("driver.zig");
const probe_mod = @import("probe.zig");

/// Total timestamp slots in the production pool, per architecture §15:
///   2 (begin/end) × 17 (kernels) × 2 (in_flight) = 68.
/// Bumping the in_flight count or adding a kernel slot above 17 changes
/// this constant — the entire downstream slot-indexing scheme depends on
/// it staying in sync with the kernel count.
pub const QUERY_POOL_SLOTS: u32 = 68;

pub const TimingError = error{
    LoaderNotReady,
    QueryPoolCreateFailed,
    PoolFull,
    InvalidPair,
    QueryReadFailed,
};

/// Owns the 68-slot VkQueryPool plus a monotonic `next` cursor that
/// `reserve2` carves pairs out of. `slots` is the pool's capacity (stored
/// here so the pool can be resized in M11+ without breaking the call
/// sites). `next` starts at 0 and increments by 2 per reservation; when
/// `next + 2 > slots`, `reserve2` returns null and the caller must
/// `reset()` before reserving again.
pub const QueryPool = struct {
    pool: vk.VkQueryPool = null,
    slots: u32 = 0,
    next: u32 = 0,
};

/// Same lazy device-fn resolution pattern as dispatch.zig.
fn resolveDeviceFn(comptime T: type, dev: vk.VkDevice, name: [*:0]const u8) ?T {
    if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
        if (gdpa(dev, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    if (vk.vkGetInstanceProcAddr_fn) |gipa| {
        const inst = driver_mod.g_default.inst;
        if (gipa(inst, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    return null;
}

/// Populate the timestamp-pool entry points. Same pattern as M8a's
/// dispatch.zig — most of these were already filled there, but timing.zig
/// shouldn't depend on dispatch.zig having been called first.
fn ensureFnSlots(dev: vk.VkDevice) TimingError!void {
    if (dev == null) return error.LoaderNotReady;
    if (vk.vkCreateQueryPool_fn == null)
        vk.vkCreateQueryPool_fn = resolveDeviceFn(vk.FnCreateQueryPool, dev, "vkCreateQueryPool");
    if (vk.vkDestroyQueryPool_fn == null)
        vk.vkDestroyQueryPool_fn = resolveDeviceFn(vk.FnDestroyQueryPool, dev, "vkDestroyQueryPool");
    if (vk.vkCmdResetQueryPool_fn == null)
        vk.vkCmdResetQueryPool_fn = resolveDeviceFn(vk.FnCmdResetQueryPool, dev, "vkCmdResetQueryPool");
    if (vk.vkGetQueryPoolResults_fn == null)
        vk.vkGetQueryPoolResults_fn = resolveDeviceFn(vk.FnGetQueryPoolResults, dev, "vkGetQueryPoolResults");
}

/// Create a fresh 68-slot timestamp pool. Caller owns the returned
/// `QueryPool` and must call `deinit` before destroying the VkDevice.
pub fn init(ctx: *driver_mod.Context) TimingError!QueryPool {
    try ensureFnSlots(ctx.dev);
    const create = vk.vkCreateQueryPool_fn orelse return error.LoaderNotReady;

    const ci: vk.VkQueryPoolCreateInfo = .{
        .queryType = vk.VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = QUERY_POOL_SLOTS,
    };
    var pool: vk.VkQueryPool = null;
    if (create(ctx.dev, &ci, null, &pool) != vk.VK_SUCCESS) {
        return error.QueryPoolCreateFailed;
    }
    return .{
        .pool = pool,
        .slots = QUERY_POOL_SLOTS,
        .next = 0,
    };
}

/// Destroy the pool's VkQueryPool. Idempotent on a zeroed `qp`.
pub fn deinit(ctx: *driver_mod.Context, qp: *QueryPool) void {
    if (ctx.dev == null) return;
    if (qp.pool != null) {
        if (vk.vkDestroyQueryPool_fn) |destroy| destroy(ctx.dev, qp.pool, null);
    }
    qp.* = .{};
}

/// Reserve the next available (begin, end) slot pair. Returns null when
/// the pool is full and the caller must `reset()` (typically after
/// draining the previous round of timings into `readPairNs`).
///
/// Pair indices are returned in source order: `pair[0]` is the begin
/// slot, `pair[1]` is the end slot. Architecture-§15 layout (the
/// `2 * kernel_idx + 34 * buffer_idx` scheme) is the caller's
/// responsibility; this module only hands out flat sequential pairs.
pub fn reserve2(qp: *QueryPool) ?[2]u32 {
    if (qp.next + 2 > qp.slots) return null;
    const begin = qp.next;
    qp.next += 2;
    return .{ begin, begin + 1 };
}

/// Reset every slot in the pool via vkCmdResetQueryPool and zero the
/// `next` cursor. Must be called inside a command-buffer recording
/// scope (it emits a command). Architecture §15 uses the host-side
/// vkResetQueryPool (Vulkan 1.2 core via VK_EXT_host_query_reset) when
/// available — that's an M11+ optimization; M8c uses the in-cmdbuf path
/// for portability.
pub fn reset(ctx: *driver_mod.Context, qp: *QueryPool, cmd: vk.VkCommandBuffer) void {
    _ = ctx;
    if (qp.pool == null) return;
    if (vk.vkCmdResetQueryPool_fn) |cmd_reset| {
        cmd_reset(cmd, qp.pool, 0, qp.slots);
    }
    qp.next = 0;
}

/// Read back the (begin, end) timestamp pair at `pair` and return the
/// delta scaled to nanoseconds via the device's `timestampPeriod`.
/// Caller MUST have ensured the GPU work that wrote the timestamps has
/// completed (typically via vkWaitForFences on the cmdbuf's fence) —
/// passes VK_QUERY_RESULT_WAIT_BIT as belt-and-suspenders but the spec
/// only guarantees a finite wait when the work is already submitted.
pub fn readPairNs(ctx: *driver_mod.Context, qp: *QueryPool, pair: [2]u32) TimingError!u64 {
    if (pair[0] >= qp.slots or pair[1] >= qp.slots) return error.InvalidPair;
    const get_results = vk.vkGetQueryPoolResults_fn orelse return error.LoaderNotReady;

    var ts: [2]u64 = .{ 0, 0 };
    const ok = get_results(
        ctx.dev,
        qp.pool,
        pair[0],
        2,
        @sizeOf(@TypeOf(ts)),
        @ptrCast(&ts),
        @sizeOf(u64),
        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
    );
    if (ok != vk.VK_SUCCESS) return error.QueryReadFailed;

    const delta_ticks: u64 = if (ts[1] >= ts[0]) ts[1] - ts[0] else 0;
    // timestampPeriod is read into the Context by the M8a chassis or by
    // the probe; default to 1.0 if it's never been set so the math is
    // well-defined.
    const period_raw: f32 = ctx.timestamp_period_ns;
    const period: f64 = if (period_raw > 0.0) @as(f64, period_raw) else 1.0;
    const ns_f: f64 = @as(f64, @floatFromInt(delta_ticks)) * period;
    if (ns_f <= 0.0) return 0;
    return @intFromFloat(ns_f);
}

// Public re-export for callers that want to refresh `ctx.timestamp_period_ns`
// without pulling in probe_mod directly.
pub fn refreshPeriod(ctx: *driver_mod.Context) void {
    ctx.timestamp_period_ns = probe_mod.readTimestampPeriod(ctx.pd);
}

// ── Tests ────────────────────────────────────────────────────────
// Pure cursor bookkeeping; no device required.

test "reserve2 hands out sequential pairs until full" {
    var qp: QueryPool = .{ .pool = null, .slots = 6, .next = 0 };
    const a = reserve2(&qp).?;
    try std.testing.expectEqual(@as(u32, 0), a[0]);
    try std.testing.expectEqual(@as(u32, 1), a[1]);
    const b = reserve2(&qp).?;
    try std.testing.expectEqual(@as(u32, 2), b[0]);
    try std.testing.expectEqual(@as(u32, 3), b[1]);
    const c = reserve2(&qp).?;
    try std.testing.expectEqual(@as(u32, 4), c[0]);
    try std.testing.expectEqual(@as(u32, 5), c[1]);
    try std.testing.expect(reserve2(&qp) == null);
}

test "QUERY_POOL_SLOTS matches architecture §15 (68)" {
    try std.testing.expectEqual(@as(u32, 68), QUERY_POOL_SLOTS);
}
