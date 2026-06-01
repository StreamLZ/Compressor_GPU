//! M8a: minimal one-shot dispatch + timing chassis.
//!
//! Owns the bare-bones command/sync/query plumbing that M5's match_any
//! microbenchmark and M8b/M8c production paths build on. The full
//! production substrate (shape-keyed primary command-buffer cache,
//! sync2 timestamp writes, 68-slot VkQueryPool with arch §15 indexing,
//! tier1_nv barrier cache) lands in M8b/M8c — this milestone is the
//! "one buffer, one submit, one fence, two timestamps" floor that
//! unblocks M5 from M8b's dependency chain.
//!
//! Lifetime:
//!   • `submitOne` lazy-creates a per-Context VkCommandPool + one primary
//!     VkCommandBuffer + one VkFence + a 2-slot timestamp VkQueryPool on
//!     first call, then reuses them on every subsequent call (reset the
//!     buffer, reset the fence, reset the query pool, re-record, submit).
//!   • `releaseContextChassis` (called from driver.deinit) frees them in
//!     spec-required reverse-creation order: free command buffers →
//!     destroy command pool → destroy fence → destroy query pool.
//!
//! Not yet handled (deferred):
//!   • Multi-dispatch batching (M8c builds the 68-slot per-arch §15 pool).
//!   • Sync2 timestamp writes (M8c — uses vkCmdWriteTimestamp2KHR when
//!     synchronization2 is supported, falls back to the 1.0 entry point
//!     otherwise; M8a unconditionally uses the 1.0 path).
//!   • Resolve of fn-pointer slots — M8a does it lazily on first call.
//!     Production (M8b) hoists this into device.zig's createDevice path.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver_mod = @import("driver.zig");
const probe_mod = @import("probe.zig");

pub const DispatchError = error{
    LoaderNotReady,
    CommandPoolCreateFailed,
    CommandBufferAllocateFailed,
    FenceCreateFailed,
    QueryPoolCreateFailed,
    BeginCommandBufferFailed,
    EndCommandBufferFailed,
    ResetCommandBufferFailed,
    ResetFenceFailed,
    SubmitFailed,
    FenceWaitTimeout,
    FenceWaitFailed,
    QueryReadFailed,
};

pub const DispatchResult = struct {
    /// GPU-side duration measured between the TOP_OF_PIPE timestamp written
    /// before vkCmdDispatch and the BOTTOM_OF_PIPE timestamp written after.
    /// Already scaled by VkPhysicalDeviceLimits.timestampPeriod, so the
    /// unit is nanoseconds regardless of vendor tick rate. Zero on devices
    /// that report timestampValidBits==0 (no timestamp support — we record
    /// the pair anyway so the cmdbuf shape stays uniform, but the delta
    /// will be garbage; M8c surfaces a hard error via the probe instead).
    ns: u64,
};

/// Slot indices within the 2-slot timestamp pool. Encoded as constants so
/// the M8c bump to 68 slots is a single-file change here — production
/// indexing (per arch §15) becomes `2 * kernel_idx + slot + 34 * buffer`.
const TS_SLOT_BEGIN: u32 = 0;
const TS_SLOT_END: u32 = 1;
const TS_SLOT_COUNT: u32 = 2;

/// Resolve a device-level entry point; prefer vkGetDeviceProcAddr (one
/// fewer dispatch hop) but fall back to the instance-level thunk via
/// vkGetInstanceProcAddr for completeness. M8a calls this once per slot
/// on first submitOne; M8b will hoist resolution into device.zig.
fn resolveDeviceFn(comptime T: type, dev: vk.VkDevice, name: [*:0]const u8) ?T {
    if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
        if (gdpa(dev, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    if (vk.vkGetInstanceProcAddr_fn) |gipa| {
        // Per spec, vkGetInstanceProcAddr returns device-level fns only
        // for instances; for true device-level lookup it's a fallback
        // that goes through the loader's dispatch trampoline. Acceptable
        // for M8a; M8b switches to per-device dispatch tables.
        const inst = driver_mod.g_default.inst;
        if (gipa(inst, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    return null;
}

/// Populate the vk_api.zig slots this milestone needs. Idempotent —
/// each `orelse resolve(...)` short-circuits when the slot is already
/// non-null, so a device.zig that knows about M8a fns can pre-populate
/// without us double-resolving.
fn ensureFnSlots(dev: vk.VkDevice) DispatchError!void {
    if (dev == null) return error.LoaderNotReady;

    if (vk.vkCreateCommandPool_fn == null)
        vk.vkCreateCommandPool_fn = resolveDeviceFn(vk.FnCreateCommandPool, dev, "vkCreateCommandPool");
    if (vk.vkDestroyCommandPool_fn == null)
        vk.vkDestroyCommandPool_fn = resolveDeviceFn(vk.FnDestroyCommandPool, dev, "vkDestroyCommandPool");
    if (vk.vkAllocateCommandBuffers_fn == null)
        vk.vkAllocateCommandBuffers_fn = resolveDeviceFn(vk.FnAllocateCommandBuffers, dev, "vkAllocateCommandBuffers");
    if (vk.vkFreeCommandBuffers_fn == null)
        vk.vkFreeCommandBuffers_fn = resolveDeviceFn(vk.FnFreeCommandBuffers, dev, "vkFreeCommandBuffers");
    if (vk.vkResetCommandBuffer_fn == null)
        vk.vkResetCommandBuffer_fn = resolveDeviceFn(vk.FnResetCommandBuffer, dev, "vkResetCommandBuffer");
    if (vk.vkBeginCommandBuffer_fn == null)
        vk.vkBeginCommandBuffer_fn = resolveDeviceFn(vk.FnBeginCommandBuffer, dev, "vkBeginCommandBuffer");
    if (vk.vkEndCommandBuffer_fn == null)
        vk.vkEndCommandBuffer_fn = resolveDeviceFn(vk.FnEndCommandBuffer, dev, "vkEndCommandBuffer");
    if (vk.vkCmdBindPipeline_fn == null)
        vk.vkCmdBindPipeline_fn = resolveDeviceFn(vk.FnCmdBindPipeline, dev, "vkCmdBindPipeline");
    if (vk.vkCmdBindDescriptorSets_fn == null)
        vk.vkCmdBindDescriptorSets_fn = resolveDeviceFn(vk.FnCmdBindDescriptorSets, dev, "vkCmdBindDescriptorSets");
    if (vk.vkCmdPushConstants_fn == null)
        vk.vkCmdPushConstants_fn = resolveDeviceFn(vk.FnCmdPushConstants, dev, "vkCmdPushConstants");
    if (vk.vkCmdDispatch_fn == null)
        vk.vkCmdDispatch_fn = resolveDeviceFn(vk.FnCmdDispatch, dev, "vkCmdDispatch");
    if (vk.vkQueueSubmit_fn == null)
        vk.vkQueueSubmit_fn = resolveDeviceFn(vk.FnQueueSubmit, dev, "vkQueueSubmit");
    if (vk.vkQueueWaitIdle_fn == null)
        vk.vkQueueWaitIdle_fn = resolveDeviceFn(vk.FnQueueWaitIdle, dev, "vkQueueWaitIdle");
    if (vk.vkCreateFence_fn == null)
        vk.vkCreateFence_fn = resolveDeviceFn(vk.FnCreateFence, dev, "vkCreateFence");
    if (vk.vkDestroyFence_fn == null)
        vk.vkDestroyFence_fn = resolveDeviceFn(vk.FnDestroyFence, dev, "vkDestroyFence");
    if (vk.vkResetFences_fn == null)
        vk.vkResetFences_fn = resolveDeviceFn(vk.FnResetFences, dev, "vkResetFences");
    if (vk.vkWaitForFences_fn == null)
        vk.vkWaitForFences_fn = resolveDeviceFn(vk.FnWaitForFences, dev, "vkWaitForFences");
    if (vk.vkCreateQueryPool_fn == null)
        vk.vkCreateQueryPool_fn = resolveDeviceFn(vk.FnCreateQueryPool, dev, "vkCreateQueryPool");
    if (vk.vkDestroyQueryPool_fn == null)
        vk.vkDestroyQueryPool_fn = resolveDeviceFn(vk.FnDestroyQueryPool, dev, "vkDestroyQueryPool");
    if (vk.vkCmdResetQueryPool_fn == null)
        vk.vkCmdResetQueryPool_fn = resolveDeviceFn(vk.FnCmdResetQueryPool, dev, "vkCmdResetQueryPool");
    if (vk.vkCmdWriteTimestamp_fn == null)
        vk.vkCmdWriteTimestamp_fn = resolveDeviceFn(vk.FnCmdWriteTimestamp, dev, "vkCmdWriteTimestamp");
    if (vk.vkGetQueryPoolResults_fn == null)
        vk.vkGetQueryPoolResults_fn = resolveDeviceFn(vk.FnGetQueryPoolResults, dev, "vkGetQueryPoolResults");
}

/// Lazy-create the chassis state on `ctx` if not already present. Safe
/// to call repeatedly — second+ calls short-circuit on the already-
/// populated cmd_pool field.
fn ensureChassis(ctx: *driver_mod.Context) DispatchError!void {
    if (ctx.cmd_pool != null) return;
    try ensureFnSlots(ctx.dev);

    // 1. VkCommandPool — RESET_COMMAND_BUFFER_BIT so we can recycle the
    //    one buffer in-place; TRANSIENT_BIT as a driver hint (the M8a
    //    buffer's lifetime is one submit-and-wait, which fits the
    //    "short-lived" semantic the flag describes).
    const pool_create = vk.vkCreateCommandPool_fn orelse return error.LoaderNotReady;
    const pool_ci: vk.VkCommandPoolCreateInfo = .{
        .flags = vk.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = ctx.qfi,
    };
    var pool: vk.VkCommandPool = null;
    if (pool_create(ctx.dev, &pool_ci, null, &pool) != vk.VK_SUCCESS) {
        return error.CommandPoolCreateFailed;
    }
    errdefer if (vk.vkDestroyCommandPool_fn) |destroy| destroy(ctx.dev, pool, null);

    // 2. VkCommandBuffer — single primary.
    const alloc_bufs = vk.vkAllocateCommandBuffers_fn orelse return error.LoaderNotReady;
    const alloc_ci: vk.VkCommandBufferAllocateInfo = .{
        .commandPool = pool,
        .level = vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var cmd_bufs: [1]vk.VkCommandBuffer = .{null};
    if (alloc_bufs(ctx.dev, &alloc_ci, @ptrCast(&cmd_bufs)) != vk.VK_SUCCESS) {
        return error.CommandBufferAllocateFailed;
    }
    // No errdefer for the buffer — destroying the pool will free the
    // buffer automatically per spec; the cleaner shutdown path is
    // implicit. (We still call vkFreeCommandBuffers from releaseContext-
    // Chassis on a happy-path teardown to keep validation layers happy.)

    // 3. VkFence — created unsignaled. submitOne's first iteration
    //    expects it that way (we reset-then-submit-then-wait).
    const fence_create = vk.vkCreateFence_fn orelse return error.LoaderNotReady;
    const fence_ci: vk.VkFenceCreateInfo = .{ .flags = 0 };
    var fence: vk.VkFence = null;
    if (fence_create(ctx.dev, &fence_ci, null, &fence) != vk.VK_SUCCESS) {
        return error.FenceCreateFailed;
    }
    errdefer if (vk.vkDestroyFence_fn) |destroy| destroy(ctx.dev, fence, null);

    // 4. VkQueryPool — 2 timestamp slots (begin/end). M8c bumps to 68
    //    per arch §15; the slot-count constant is in one place above
    //    so the bump is localized.
    const qpool_create = vk.vkCreateQueryPool_fn orelse return error.LoaderNotReady;
    const qpool_ci: vk.VkQueryPoolCreateInfo = .{
        .queryType = vk.VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = TS_SLOT_COUNT,
    };
    var qpool: vk.VkQueryPool = null;
    if (qpool_create(ctx.dev, &qpool_ci, null, &qpool) != vk.VK_SUCCESS) {
        return error.QueryPoolCreateFailed;
    }
    // Past the last fallible step — commit to ctx.

    ctx.cmd_pool = pool;
    ctx.cmd_buf = cmd_bufs[0];
    ctx.fence = fence;
    ctx.query_pool = qpool;
    if (ctx.timestamp_period_ns == 0.0) {
        ctx.timestamp_period_ns = probe_mod.readTimestampPeriod(ctx.pd);
    }
}

/// Tear down the chassis state on `ctx`. Called from `driver.deinit`
/// strictly before `device_mod.destroyDevice` — every M8a object is a
/// device child and must be destroyed first per the Vulkan spec.
///
/// Safe to call on a Context that never had submitOne invoked (every
/// field starts null and the null-guards below short-circuit).
pub fn releaseContextChassis(ctx: *driver_mod.Context) void {
    if (ctx.dev == null) return;

    // Best-effort idle the queue. If submitOne's last call left work in
    // flight (it shouldn't — submitOne waits before returning — but a
    // panic between submit and wait would skip the wait), this drains
    // it so the implicit "all submitted work finished" precondition for
    // destroying the buffer/pool/fence/qpool is satisfied.
    if (vk.vkQueueWaitIdle_fn) |idle| {
        if (ctx.queue != null) _ = idle(ctx.queue);
    }

    // Free the command buffer explicitly before destroying its parent
    // pool. Validation layers complain about leaked allocations when
    // the pool is destroyed with live children even though the spec
    // permits it (pool destruction implicitly frees children).
    if (ctx.cmd_buf != null and ctx.cmd_pool != null) {
        if (vk.vkFreeCommandBuffers_fn) |free_bufs| {
            const bufs: [1]vk.VkCommandBuffer = .{ctx.cmd_buf};
            free_bufs(ctx.dev, ctx.cmd_pool, 1, @ptrCast(&bufs));
        }
        ctx.cmd_buf = null;
    }

    if (ctx.cmd_pool != null) {
        if (vk.vkDestroyCommandPool_fn) |destroy| destroy(ctx.dev, ctx.cmd_pool, null);
        ctx.cmd_pool = null;
    }
    if (ctx.fence != null) {
        if (vk.vkDestroyFence_fn) |destroy| destroy(ctx.dev, ctx.fence, null);
        ctx.fence = null;
    }
    if (ctx.query_pool != null) {
        if (vk.vkDestroyQueryPool_fn) |destroy| destroy(ctx.dev, ctx.query_pool, null);
        ctx.query_pool = null;
    }
    ctx.timestamp_period_ns = 0.0;
}

/// Submit a single compute dispatch with a fence and a timestamp pair.
///
/// Records the command buffer with:
///   reset query pool → write TOP_OF_PIPE @ slot 0 → bind pipeline
///   → bind descriptor set 0 → push constants (if any) → dispatch
///   → write BOTTOM_OF_PIPE @ slot 1
///
/// Submits to ctx.queue with ctx.fence, vkWaitForFences with the budget
/// in vk_api.VK_M8A_FENCE_WAIT_NS (60 s — generous enough for a 200 MB
/// L1 silesia encode on an Intel iGPU, tight enough to surface real
/// hangs in CI), then reads the timestamp pair and returns the scaled
/// delta.
///
/// `push_constants_bytes.len == 0` skips the push-constant call entirely
/// (the spec accepts a zero-byte push, but skipping avoids the dispatch
/// table entry and validation chatter on layouts with no push range).
///
/// `descriptor_set == null` skips the bind — useful for kernels whose
/// pipeline layout has zero descriptor sets (M5's match_any microbench
/// is one such case once it's wired). When non-null, we bind it to set
/// index 0; multi-set support is M8b.
///
/// `group_count` is the [X, Y, Z] dispatch dim passed straight through
/// to vkCmdDispatch. Caller is responsible for not exceeding
/// VkPhysicalDeviceLimits.maxComputeWorkGroupCount (a Tier-1 NVIDIA
/// device gives 2^31-1 in X, so this is hard to hit accidentally).
pub fn submitOne(
    ctx: *driver_mod.Context,
    pipeline: vk.VkPipeline,
    pipeline_layout: vk.VkPipelineLayout,
    descriptor_set: vk.VkDescriptorSet,
    push_constants_bytes: []const u8,
    group_count: [3]u32,
) DispatchError!DispatchResult {
    if (!ctx.initialized or ctx.dev == null or ctx.queue == null) {
        return error.LoaderNotReady;
    }
    try ensureChassis(ctx);

    // Capture the fn-ptr handles up front so the recording block reads
    // top-to-bottom without `orelse` clutter at every hop.
    const reset_cb = vk.vkResetCommandBuffer_fn orelse return error.LoaderNotReady;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return error.LoaderNotReady;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return error.LoaderNotReady;
    const cmd_reset_qp = vk.vkCmdResetQueryPool_fn orelse return error.LoaderNotReady;
    const cmd_write_ts = vk.vkCmdWriteTimestamp_fn orelse return error.LoaderNotReady;
    const cmd_bind_pl = vk.vkCmdBindPipeline_fn orelse return error.LoaderNotReady;
    const cmd_dispatch = vk.vkCmdDispatch_fn orelse return error.LoaderNotReady;
    const reset_fence = vk.vkResetFences_fn orelse return error.LoaderNotReady;
    const submit = vk.vkQueueSubmit_fn orelse return error.LoaderNotReady;
    const wait_fence = vk.vkWaitForFences_fn orelse return error.LoaderNotReady;
    const get_results = vk.vkGetQueryPoolResults_fn orelse return error.LoaderNotReady;

    // Reset the command buffer in place (RESET_COMMAND_BUFFER_BIT on the
    // pool authorizes this).
    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return error.ResetCommandBufferFailed;

    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) {
        return error.BeginCommandBufferFailed;
    }

    // Reset the query pool inside the cmdbuf. vkCmdResetQueryPool is
    // mandatory before every reuse (the host-side vkResetQueryPool from
    // VK_EXT_host_query_reset / Vulkan 1.2 core is M8c's optimization).
    cmd_reset_qp(ctx.cmd_buf, ctx.query_pool, 0, TS_SLOT_COUNT);

    // Write the "before" timestamp at TOP_OF_PIPE. The spec defines this
    // as "the earliest stage the timestamp may be captured at"; for a
    // pure compute submit on a compute-only queue it lands right at the
    // start of the dispatch. M8c flips this to a sync2 vkCmdWriteTimestamp2.
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_BEGIN);

    cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    // Bind descriptor set 0 only if the caller provided one. M5's
    // match_any benchmark uses no buffers (subgroup-only); production
    // kernels always pass a set.
    if (descriptor_set != null and pipeline_layout != null) {
        const cmd_bind_ds = vk.vkCmdBindDescriptorSets_fn orelse return error.LoaderNotReady;
        const sets: [1]vk.VkDescriptorSet = .{descriptor_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0, // firstSet
            1, // descriptorSetCount
            @ptrCast(&sets),
            0,
            null,
        );
    }

    // Push constants, if any. We assume the pipeline layout declares one
    // compute-stage range starting at offset 0; M8b's descriptor factory
    // is what defines that range, and the M5 microbench layout matches.
    if (push_constants_bytes.len > 0) {
        const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
        if (pipeline_layout == null) return error.LoaderNotReady;
        cmd_push(
            ctx.cmd_buf,
            pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_constants_bytes.len),
            @ptrCast(push_constants_bytes.ptr),
        );
    }

    cmd_dispatch(ctx.cmd_buf, group_count[0], group_count[1], group_count[2]);

    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_END);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;

    // Reset the fence to unsignaled before submit. submitOne starts from
    // a signaled fence on the second+ call (vkWaitForFences leaves it
    // signaled); the spec requires reset before reuse.
    const fences: [1]vk.VkFence = .{ctx.fence};
    if (reset_fence(ctx.dev, 1, @ptrCast(&fences)) != vk.VK_SUCCESS) {
        return error.ResetFenceFailed;
    }

    const cmd_bufs: [1]vk.VkCommandBuffer = .{ctx.cmd_buf};
    const submit_info: vk.VkSubmitInfo = .{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cmd_bufs),
    };
    const submits: [1]vk.VkSubmitInfo = .{submit_info};
    if (submit(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) {
        return error.SubmitFailed;
    }

    // Wait with the budget in VK_M8A_FENCE_WAIT_NS (60 s). The original
    // 2 s value was sized for M8a microbenches; production-scale L1
    // encodes (200 MB silesia, Intel iGPU) routinely exceed that.
    // 60 s still surfaces a typed timeout fast enough that a genuinely
    // hung GPU does not deadlock the test process.
    const wait_result = wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS);
    if (wait_result == vk.VK_TIMEOUT) return error.FenceWaitTimeout;
    if (wait_result != vk.VK_SUCCESS) return error.FenceWaitFailed;

    // Pull both timestamps in one call. 64-bit slots, WAIT_BIT for
    // belt-and-suspenders (the fence guarantee already makes results
    // available; WAIT_BIT just means the driver won't return VK_NOT_READY
    // if it ever did otherwise).
    var ts: [TS_SLOT_COUNT]u64 = .{ 0, 0 };
    const result = get_results(
        ctx.dev,
        ctx.query_pool,
        0,
        TS_SLOT_COUNT,
        @sizeOf(@TypeOf(ts)),
        @ptrCast(&ts),
        @sizeOf(u64),
        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
    );
    if (result != vk.VK_SUCCESS) return error.QueryReadFailed;

    // Delta scaled by timestampPeriod (ns/tick). Saturating sub guards
    // against the (spec-permitted) case where TOP > BOTTOM on some
    // drivers when the dispatch is shorter than one tick.
    const delta_ticks: u64 = if (ts[TS_SLOT_END] >= ts[TS_SLOT_BEGIN])
        ts[TS_SLOT_END] - ts[TS_SLOT_BEGIN]
    else
        0;
    const period = if (ctx.timestamp_period_ns > 0.0) ctx.timestamp_period_ns else 1.0;
    // u64 → f64 → u64 round-trip; one tick at 1ns precision is plenty
    // for the µs-scale dispatches M8a will report.
    const ns_f: f64 = @as(f64, @floatFromInt(delta_ticks)) * @as(f64, period);
    const ns: u64 = if (ns_f <= 0.0) 0 else @intFromFloat(ns_f);

    return .{ .ns = ns };
}
