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

// QueryPerformanceCounter bracketing for `submit_wait_wall_ns`. Used by
// the readback-cost diagnostic to attribute the host-wall time around
// vkQueueSubmit + vkWaitForFences vs. host-side recording / fence read.
const win32 = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};

inline fn qpcNow() i64 {
    var c: i64 = 0;
    _ = win32.QueryPerformanceCounter(&c);
    return c;
}

inline fn qpcDeltaNs(from: i64, to: i64) u64 {
    var freq: i64 = 0;
    _ = win32.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

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
    /// GPU-side duration of the in-cmdbuf vkCmdCopyBuffer (set only by
    /// `submitOneWithCopy`; zero for plain `submitOne`). Measured with
    /// a second TOP_OF_PIPE/BOTTOM_OF_PIPE timestamp pair written either
    /// side of cmd_copy. Used by the readback-cost diagnostic in
    /// slz1_codec to separate the GPU dst_b → dst_stage transfer from
    /// the host @memcpy(out, dst_stage.mapped).
    copy_ns: u64 = 0,
    /// Host-wall duration measured around `vkQueueSubmit` ... `vkWaitForFences`.
    /// Includes the GPU's actual work (kernel + barrier + copy on
    /// submitOneWithCopy) plus any host scheduling / fence-poll cost.
    /// On a single-queue path with a wait-fence right after submit this
    /// is dominated by GPU work time. Diagnostic-only — not used for
    /// any control flow.
    submit_wait_wall_ns: u64 = 0,
    /// Host-wall around the cmdbuf recording phase (reset → begin →
    /// record dispatch / barriers / copy → end). Diagnostic-only.
    record_wall_ns: u64 = 0,
    /// Host-wall around the `vkQueueSubmit` call alone (no wait).
    submit_call_ns: u64 = 0,
    /// Host-wall around the `vkWaitForFences` call alone (this is the
    /// "GPU work + driver scheduling" cost from the host's POV).
    wait_call_ns: u64 = 0,
    /// Host-wall around the `vkGetQueryPoolResults` call (timestamp
    /// readback). With VK_QUERY_RESULT_WAIT_BIT this can include extra
    /// availability-wait time, though all queries should already be
    /// resolved by the time we get here (the fence wait above blocked
    /// until the GPU finished, which makes all timestamps available).
    query_read_ns: u64 = 0,
};

/// Slot indices within the timestamp pool. The first 4 slots
/// (BEGIN/END/COPY_BEGIN/COPY_END) are owned by the chassis submit
/// helpers (submitOne writes the first pair; submitOneWithCopy writes
/// all four). Slots 4..35 are the per-kernel-decode breakdown slots
/// reserved for `recordDecodePipelineInto` + `recordAndSubmitMerged
/// Decode` so the SLZ_VK_PROFILE_DECODE=1 printer can report per-
/// kernel GPU ms on installs where nsys --trace=vulkan produces an
/// empty Vulkan trace (the prior agent verified this is the case on
/// Nsight Systems 2025.5.2 on Windows). The decode pipeline writes
/// BEGIN/END pairs for every dispatched kernel (walk_frame, prefix_
/// sum, scan_parse, 4× compact_huff, compact_raw, gather_raw_off16,
/// l1_unwrap) plus 3 transfer-stage spans (frame_staging→frame DMA,
/// 5× vkCmdFillBuffer lumped together, meta→n_chunks_scratch DMA);
/// the existing lz_decode kernel + dst→host copy still ride
/// slots 0-3 in the chassis-level reservation. Total = 36 slots.
pub const TS_SLOT_BEGIN: u32 = 0;
pub const TS_SLOT_END: u32 = 1;
pub const TS_SLOT_COPY_BEGIN: u32 = 2;
pub const TS_SLOT_COPY_END: u32 = 3;
/// Slot count for the chassis-level `submitOne` / `submitOneWithCopy`
/// helpers — they only write slots 0..3 and must only `vkGetQueryPool
/// Results` for that range (a larger range would wait forever on
/// VK_QUERY_RESULT_WAIT_BIT for slots that were never written). The
/// decode-pipeline per-kernel slots live in 4..29 and are read in a
/// separate call from `recordAndSubmitMergedDecode`.
pub const TS_SLOT_CHASSIS_COUNT: u32 = 4;

// Per-kernel decode pipeline slots (begin/end pairs). Indices match the
// dispatch order in `recordDecodePipelineInto` so a reader of the
// SLZ_VK_PROFILE_DECODE=1 line can follow the slot numbers in source
// order. SLOT_DEC_KERNEL_FIRST_BEGIN = 4 keeps the first 4 slots
// reserved for the chassis-level BEGIN/END/COPY_BEGIN/COPY_END.
pub const TS_SLOT_DEC_FRAME_DMA_BEGIN: u32 = 4; // frame_staging → frame
pub const TS_SLOT_DEC_FRAME_DMA_END: u32 = 5;
pub const TS_SLOT_DEC_FILLS_BEGIN: u32 = 6; // 5× vkCmdFillBuffer lumped
pub const TS_SLOT_DEC_FILLS_END: u32 = 7;
pub const TS_SLOT_DEC_WALK_BEGIN: u32 = 8;
pub const TS_SLOT_DEC_WALK_END: u32 = 9;
pub const TS_SLOT_DEC_PREFIX_BEGIN: u32 = 10;
pub const TS_SLOT_DEC_PREFIX_END: u32 = 11;
pub const TS_SLOT_DEC_META_COPY_BEGIN: u32 = 12; // meta → n_chunks_scratch
pub const TS_SLOT_DEC_META_COPY_END: u32 = 13;
pub const TS_SLOT_DEC_SCAN_BEGIN: u32 = 14;
pub const TS_SLOT_DEC_SCAN_END: u32 = 15;
pub const TS_SLOT_DEC_COMPACT_LIT_BEGIN: u32 = 16;
pub const TS_SLOT_DEC_COMPACT_LIT_END: u32 = 17;
pub const TS_SLOT_DEC_COMPACT_TOK_BEGIN: u32 = 18;
pub const TS_SLOT_DEC_COMPACT_TOK_END: u32 = 19;
pub const TS_SLOT_DEC_COMPACT_HI_BEGIN: u32 = 20;
pub const TS_SLOT_DEC_COMPACT_HI_END: u32 = 21;
pub const TS_SLOT_DEC_COMPACT_LO_BEGIN: u32 = 22;
pub const TS_SLOT_DEC_COMPACT_LO_END: u32 = 23;
pub const TS_SLOT_DEC_COMPACT_RAW_BEGIN: u32 = 24;
pub const TS_SLOT_DEC_COMPACT_RAW_END: u32 = 25;
pub const TS_SLOT_DEC_GATHER_BEGIN: u32 = 26;
pub const TS_SLOT_DEC_GATHER_END: u32 = 27;
pub const TS_SLOT_DEC_UNWRAP_BEGIN: u32 = 28;
pub const TS_SLOT_DEC_UNWRAP_END: u32 = 29;
// Slots 30..35 reserved for future per-kernel splits (e.g. splitting
// the 5× vkCmdFillBuffer into per-buffer spans for finer attribution).
pub const TS_SLOT_COUNT: u32 = 36;

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
    // The discrete-GPU readback path uses a vkCmdCopyBuffer at the tail
    // of the decode cmdbuf, plus a vkCmdPipelineBarrier for the
    // compute-write → transfer-read sync. Both are core Vulkan 1.0 fns;
    // resolve here so submitOneWithCopy doesn't need a sibling ensure.
    if (vk.vkCmdCopyBuffer_fn == null)
        vk.vkCmdCopyBuffer_fn = resolveDeviceFn(vk.FnCmdCopyBuffer, dev, "vkCmdCopyBuffer");
    if (vk.vkCmdPipelineBarrier_fn == null)
        vk.vkCmdPipelineBarrier_fn = resolveDeviceFn(vk.FnCmdPipelineBarrier, dev, "vkCmdPipelineBarrier");
    // CPU-built host-input decode path stages the host-known n_chunks
    // (4 bytes) into the GPU-side `n_chunks_scratch` via vkCmdUpdateBuffer
    // inline in the recorded cmdbuf — mirrors CUDA's 4 B H2D at
    // `src/decode/decode_dispatch.zig:617-619`.
    if (vk.vkCmdUpdateBuffer_fn == null)
        vk.vkCmdUpdateBuffer_fn = resolveDeviceFn(vk.FnCmdUpdateBuffer, dev, "vkCmdUpdateBuffer");

    // VK_EXT_debug_utils label entry points — pure instrumentation. Slots
    // stay null on devices/loaders without the extension; the label-emit
    // helper below null-checks and silently no-ops. Resolved here (lazy
    // on first dispatch) so we don't pay the GetDeviceProcAddr cost
    // before the first decode/encode runs.
    if (vk.vkCmdBeginDebugUtilsLabelEXT_fn == null)
        vk.vkCmdBeginDebugUtilsLabelEXT_fn = resolveDeviceFn(vk.FnCmdBeginDebugUtilsLabelEXT, dev, "vkCmdBeginDebugUtilsLabelEXT");
    if (vk.vkCmdEndDebugUtilsLabelEXT_fn == null)
        vk.vkCmdEndDebugUtilsLabelEXT_fn = resolveDeviceFn(vk.FnCmdEndDebugUtilsLabelEXT, dev, "vkCmdEndDebugUtilsLabelEXT");
}

// ── VK_EXT_debug_utils label helpers ─────────────────────────────
// `beginLabel` / `endLabel` wrap the cmd-buffer label entry points
// with a null-check (silent no-op when the extension was not loaded).
// All four submit* helpers below call these around the recorded
// `vkCmdDispatch` (and `vkCmdCopyBuffer` for the staging copy) so
// Nsight Systems' Vulkan trace can attribute per-kernel GPU intervals
// to a human-readable name like "lz_decode" / "dst_b->dst_stage".
//
// `name` is a non-null-terminated slice; we copy it into a fixed
// stack buffer + zero-terminator so the Vulkan call sees a [*:0]
// without needing the caller to provide a sentinel-terminated string.
// 64-byte cap covers every label we emit ("prefix_sum_chunks" at 17
// bytes is the longest); longer names are truncated.
inline fn beginLabel(cmd_buf: vk.VkCommandBuffer, name: ?[]const u8) void {
    const fn_ptr = vk.vkCmdBeginDebugUtilsLabelEXT_fn orelse return;
    const n = name orelse return;
    if (n.len == 0) return;
    var buf: [64]u8 = @splat(0);
    const cap: usize = @min(n.len, buf.len - 1);
    @memcpy(buf[0..cap], n[0..cap]);
    buf[cap] = 0;
    const info: vk.VkDebugUtilsLabelEXT = .{
        .pLabelName = @ptrCast(&buf[0]),
        .color = .{ 0, 0, 0, 0 },
    };
    fn_ptr(cmd_buf, &info);
}

inline fn endLabel(cmd_buf: vk.VkCommandBuffer, name: ?[]const u8) void {
    const fn_ptr = vk.vkCmdEndDebugUtilsLabelEXT_fn orelse return;
    const n = name orelse return;
    if (n.len == 0) return;
    fn_ptr(cmd_buf);
}

/// Lazy-create the chassis state on `ctx` if not already present. Safe
/// to call repeatedly — second+ calls short-circuit on the already-
/// populated cmd_pool field.
///
/// `ensureChassisPub` exposes this to peer modules (the streamlz_gpu_vk
/// D2D staging path uses it to prep the cmd_buf + fence + queue
/// pointers before recording its own one-shot copies, without needing
/// to bounce through a full submitOne dispatch).
pub fn ensureChassisPub(ctx: *driver_mod.Context) DispatchError!void {
    return ensureChassis(ctx);
}

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
    return submitOneLabeled(ctx, pipeline, pipeline_layout, descriptor_set, push_constants_bytes, group_count, null);
}

/// Same as `submitOne` but emits a `VK_EXT_debug_utils` label region
/// around the recorded `vkCmdDispatch`. `label` is a non-null-terminated
/// slice (the helper null-terminates into a stack buffer); pass `null`
/// for no label (silent no-op when the extension is absent). Nsight
/// Systems' Vulkan trace picks up the label and attributes the GPU-side
/// kernel time to the human-readable name in vulkan_gpu_marker_sum.
pub fn submitOneLabeled(
    ctx: *driver_mod.Context,
    pipeline: vk.VkPipeline,
    pipeline_layout: vk.VkPipelineLayout,
    descriptor_set: vk.VkDescriptorSet,
    push_constants_bytes: []const u8,
    group_count: [3]u32,
    label: ?[]const u8,
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
    const t_record_begin = qpcNow();
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

    beginLabel(ctx.cmd_buf, label);
    cmd_dispatch(ctx.cmd_buf, group_count[0], group_count[1], group_count[2]);
    endLabel(ctx.cmd_buf, label);

    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_END);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;
    const record_ns = qpcDeltaNs(t_record_begin, qpcNow());

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
    const t_submit_begin = qpcNow();
    if (submit(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) {
        return error.SubmitFailed;
    }
    const t_submit_end = qpcNow();
    const submit_call_ns = qpcDeltaNs(t_submit_begin, t_submit_end);

    // Wait with the budget in VK_M8A_FENCE_WAIT_NS (60 s). The original
    // 2 s value was sized for M8a microbenches; production-scale L1
    // encodes (200 MB silesia, Intel iGPU) routinely exceed that.
    // 60 s still surfaces a typed timeout fast enough that a genuinely
    // hung GPU does not deadlock the test process.
    const wait_result = wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS);
    const t_wait_end = qpcNow();
    const wait_call_ns = qpcDeltaNs(t_submit_end, t_wait_end);
    const submit_wait_ns = qpcDeltaNs(t_submit_begin, t_wait_end);
    if (wait_result == vk.VK_TIMEOUT) return error.FenceWaitTimeout;
    if (wait_result != vk.VK_SUCCESS) return error.FenceWaitFailed;

    // Pull only the BEGIN/END timestamps; plain `submitOne` doesn't
    // write the COPY_BEGIN/COPY_END pair so reading them back with
    // VK_QUERY_RESULT_WAIT_BIT would hang (waits for availability of
    // queries that were never recorded). The wider read happens only
    // in `submitOneWithCopy`, which writes all four slots.
    var ts: [2]u64 = .{ 0, 0 };
    const t_query_begin = qpcNow();
    const result = get_results(
        ctx.dev,
        ctx.query_pool,
        0,
        2,
        @sizeOf(@TypeOf(ts)),
        @ptrCast(&ts),
        @sizeOf(u64),
        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
    );
    const query_read_ns = qpcDeltaNs(t_query_begin, qpcNow());
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

    return .{
        .ns = ns,
        .copy_ns = 0,
        .submit_wait_wall_ns = submit_wait_ns,
        .record_wall_ns = record_ns,
        .submit_call_ns = submit_call_ns,
        .wait_call_ns = wait_call_ns,
        .query_read_ns = query_read_ns,
    };
}

/// One follow-on buffer copy queued after the dispatch in the SAME
/// command buffer. Used by the decoder's discrete-GPU readback path:
/// the kernel writes into a DEVICE_LOCAL-only Dst buffer and this copy
/// stages it into a HOST_VISIBLE buffer for fast CPU readback. Keeping
/// dispatch + copy in one cmdbuf avoids a second submit/wait round-trip
/// — the implicit pipeline barrier between the compute write and the
/// transfer read is provided via the explicit vkCmdPipelineBarrier
/// call below.
pub const CopyOp = struct {
    src: vk.VkBuffer,
    dst: vk.VkBuffer,
    size: vk.VkDeviceSize,
    src_offset: vk.VkDeviceSize = 0,
    dst_offset: vk.VkDeviceSize = 0,
};

/// Same as `submitOne` but also records a vkCmdCopyBuffer after the
/// dispatch (with a SHADER_WRITE → TRANSFER_READ barrier so the copy
/// sees the kernel's writes). All other behavior is identical to
/// submitOne: TOP_OF_PIPE timestamp BEFORE the dispatch, BOTTOM_OF_PIPE
/// AFTER the dispatch (the copy is NOT counted in the returned ns, so
/// the caller can still attribute kernel time vs. copy time). Used by
/// the L1 decoder; encoder doesn't need it because its readback stream
/// buffers stay host-visible.
pub fn submitOneWithCopy(
    ctx: *driver_mod.Context,
    pipeline: vk.VkPipeline,
    pipeline_layout: vk.VkPipelineLayout,
    descriptor_set: vk.VkDescriptorSet,
    push_constants_bytes: []const u8,
    group_count: [3]u32,
    copy: CopyOp,
) DispatchError!DispatchResult {
    return submitOneWithCopyLabeled(ctx, pipeline, pipeline_layout, descriptor_set, push_constants_bytes, group_count, copy, null, null);
}

/// Same as `submitOneWithCopy` but emits `VK_EXT_debug_utils` label
/// regions around the recorded `vkCmdDispatch` (named by
/// `dispatch_label`) and `vkCmdCopyBuffer` (named by `copy_label`).
/// Pass `null` for either to skip the corresponding label (silent
/// no-op when the extension is absent).
pub fn submitOneWithCopyLabeled(
    ctx: *driver_mod.Context,
    pipeline: vk.VkPipeline,
    pipeline_layout: vk.VkPipelineLayout,
    descriptor_set: vk.VkDescriptorSet,
    push_constants_bytes: []const u8,
    group_count: [3]u32,
    copy: CopyOp,
    dispatch_label: ?[]const u8,
    copy_label: ?[]const u8,
) DispatchError!DispatchResult {
    if (!ctx.initialized or ctx.dev == null or ctx.queue == null) {
        return error.LoaderNotReady;
    }
    try ensureChassis(ctx);

    const reset_cb = vk.vkResetCommandBuffer_fn orelse return error.LoaderNotReady;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return error.LoaderNotReady;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return error.LoaderNotReady;
    const cmd_reset_qp = vk.vkCmdResetQueryPool_fn orelse return error.LoaderNotReady;
    const cmd_write_ts = vk.vkCmdWriteTimestamp_fn orelse return error.LoaderNotReady;
    const cmd_bind_pl = vk.vkCmdBindPipeline_fn orelse return error.LoaderNotReady;
    const cmd_dispatch = vk.vkCmdDispatch_fn orelse return error.LoaderNotReady;
    const cmd_copy = vk.vkCmdCopyBuffer_fn orelse return error.LoaderNotReady;
    const cmd_barrier = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
    const reset_fence = vk.vkResetFences_fn orelse return error.LoaderNotReady;
    const submit = vk.vkQueueSubmit_fn orelse return error.LoaderNotReady;
    const wait_fence = vk.vkWaitForFences_fn orelse return error.LoaderNotReady;
    const get_results = vk.vkGetQueryPoolResults_fn orelse return error.LoaderNotReady;

    const t_record_begin = qpcNow();
    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return error.ResetCommandBufferFailed;

    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) {
        return error.BeginCommandBufferFailed;
    }

    cmd_reset_qp(ctx.cmd_buf, ctx.query_pool, 0, TS_SLOT_COUNT);
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_BEGIN);

    cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    if (descriptor_set != null and pipeline_layout != null) {
        const cmd_bind_ds = vk.vkCmdBindDescriptorSets_fn orelse return error.LoaderNotReady;
        const sets: [1]vk.VkDescriptorSet = .{descriptor_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
    }

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

    beginLabel(ctx.cmd_buf, dispatch_label);
    cmd_dispatch(ctx.cmd_buf, group_count[0], group_count[1], group_count[2]);
    endLabel(ctx.cmd_buf, dispatch_label);

    // BOTTOM_OF_PIPE timestamp captures pure kernel time — the copy
    // below is bracketed by its own COPY_BEGIN/COPY_END timestamp pair
    // so the host can attribute kernel ns vs vkCmdCopyBuffer GPU ns
    // independently.
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_END);

    // Buffer barrier: SHADER_WRITE in the compute stage must complete
    // and be visible to the upcoming TRANSFER_READ from the transfer
    // stage on the src buffer. dst buffer doesn't need a barrier here
    // because it has no prior GPU writes (host writes were flushed by
    // vkQueueSubmit's implicit host→queue domain transition).
    const bbarrier: vk.VkBufferMemoryBarrier = .{
        .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = vk.VK_ACCESS_TRANSFER_READ_BIT,
        .srcQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
        .buffer = copy.src,
        .offset = copy.src_offset,
        .size = copy.size,
    };
    const bbarriers: [1]vk.VkBufferMemoryBarrier = .{bbarrier};
    cmd_barrier(
        ctx.cmd_buf,
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        null,
        1,
        @ptrCast(&bbarriers),
        0,
        null,
    );

    // TOP_OF_PIPE timestamp pre-copy (after the barrier so the timer
    // starts at the moment the transfer stage actually picks the work
    // up). The copy is a single vkCmdCopyBuffer of `copy.size` bytes
    // from a DEVICE_LOCAL buffer to a HOST_VISIBLE (host_visible_sysmem
    // mode in l1_codec.createBufferEx) buffer; on a discrete GPU this
    // is GPU-PCIe-WRITE bandwidth-bound (~10 GB/s on a 4060 Ti slot).
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_COPY_BEGIN);

    const region: vk.VkBufferCopy = .{
        .srcOffset = copy.src_offset,
        .dstOffset = copy.dst_offset,
        .size = copy.size,
    };
    const regions: [1]vk.VkBufferCopy = .{region};
    beginLabel(ctx.cmd_buf, copy_label);
    cmd_copy(ctx.cmd_buf, copy.src, copy.dst, 1, @ptrCast(&regions));
    endLabel(ctx.cmd_buf, copy_label);

    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_COPY_END);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;
    const record_ns = qpcDeltaNs(t_record_begin, qpcNow());

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
    const t_submit_begin = qpcNow();
    if (submit(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) {
        return error.SubmitFailed;
    }
    const t_submit_end = qpcNow();
    const submit_call_ns = qpcDeltaNs(t_submit_begin, t_submit_end);

    const wait_result = wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS);
    const t_wait_end = qpcNow();
    const wait_call_ns = qpcDeltaNs(t_submit_end, t_wait_end);
    const submit_wait_ns = qpcDeltaNs(t_submit_begin, t_wait_end);
    if (wait_result == vk.VK_TIMEOUT) return error.FenceWaitTimeout;
    if (wait_result != vk.VK_SUCCESS) return error.FenceWaitFailed;

    var ts: [TS_SLOT_CHASSIS_COUNT]u64 = .{ 0, 0, 0, 0 };
    const t_query_begin = qpcNow();
    const result = get_results(
        ctx.dev,
        ctx.query_pool,
        0,
        TS_SLOT_CHASSIS_COUNT,
        @sizeOf(@TypeOf(ts)),
        @ptrCast(&ts),
        @sizeOf(u64),
        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
    );
    const query_read_ns = qpcDeltaNs(t_query_begin, qpcNow());
    if (result != vk.VK_SUCCESS) return error.QueryReadFailed;

    const delta_ticks: u64 = if (ts[TS_SLOT_END] >= ts[TS_SLOT_BEGIN])
        ts[TS_SLOT_END] - ts[TS_SLOT_BEGIN]
    else
        0;
    const period = if (ctx.timestamp_period_ns > 0.0) ctx.timestamp_period_ns else 1.0;
    const ns_f: f64 = @as(f64, @floatFromInt(delta_ticks)) * @as(f64, period);
    const ns: u64 = if (ns_f <= 0.0) 0 else @intFromFloat(ns_f);

    const copy_ticks: u64 = if (ts[TS_SLOT_COPY_END] >= ts[TS_SLOT_COPY_BEGIN])
        ts[TS_SLOT_COPY_END] - ts[TS_SLOT_COPY_BEGIN]
    else
        0;
    const copy_ns_f: f64 = @as(f64, @floatFromInt(copy_ticks)) * @as(f64, period);
    const copy_ns: u64 = if (copy_ns_f <= 0.0) 0 else @intFromFloat(copy_ns_f);

    return .{
        .ns = ns,
        .copy_ns = copy_ns,
        .submit_wait_wall_ns = submit_wait_ns,
        .record_wall_ns = record_ns,
        .submit_call_ns = submit_call_ns,
        .wait_call_ns = wait_call_ns,
        .query_read_ns = query_read_ns,
    };
}

/// Description of one compute dispatch within a multi-dispatch command
/// buffer. `descriptor_set == null` skips the bind; `push_constants_bytes
/// .len == 0` skips the push. `label` (when non-null) is the
/// `VK_EXT_debug_utils` region name emitted around the recorded
/// `vkCmdDispatch`; pass `null` for no label (silent no-op when the
/// extension is absent).
pub const DispatchSpec = struct {
    pipeline: vk.VkPipeline,
    pipeline_layout: vk.VkPipelineLayout,
    descriptor_set: vk.VkDescriptorSet,
    push_constants_bytes: []const u8,
    group_count: [3]u32,
    label: ?[]const u8 = null,
};

/// Two-dispatch + buffer-copy in a single command buffer + single submit.
///
/// Replaces the pair `submitOne(unwrap)` + `submitOneWithCopy(decode,copy)`
/// the decoder used pre-merge. Eliminates one vkQueueSubmit and one
/// vkWaitForFences round-trip per decode call.
///
/// Recorded shape:
///   reset query pool
///   first dispatch (bind pipe → bind ds → push consts → dispatch)
///   buffer-memory barrier on `inter_barrier_buf` (SHADER_WRITE →
///     SHADER_READ, compute → compute) so the second dispatch sees the
///     first dispatch's writes
///   TS_SLOT_BEGIN write (TOP_OF_PIPE) — brackets ONLY the second dispatch
///   second dispatch (bind pipe → bind ds → push consts → dispatch)
///   TS_SLOT_END write (BOTTOM_OF_PIPE)
///   buffer-memory barrier on `copy.src` (SHADER_WRITE → TRANSFER_READ,
///     compute → transfer) so the copy sees the second dispatch's writes
///   TS_SLOT_COPY_BEGIN write
///   vkCmdCopyBuffer
///   TS_SLOT_COPY_END write
///
/// `ns` in the returned DispatchResult is the GPU-side duration of the
/// SECOND dispatch only (matching `submitOneWithCopy`'s semantics so the
/// caller's `gpu_kernel_ns` reporting stays comparable across the
/// pre/post-merge code).
///
/// `inter_barrier_buf` is the buffer the first dispatch writes that the
/// second dispatch reads (in the L1 decoder, this is the per-stream
/// chunks descriptor that l1_unwrap.comp writes and lz_decode.comp reads).
/// If `inter_barrier_buf == null` no intermediate barrier is emitted —
/// the caller is responsible for the chain producing/consuming on
/// distinct buffers or no aliasing.
pub fn submitTwoWithCopy(
    ctx: *driver_mod.Context,
    first: DispatchSpec,
    second: DispatchSpec,
    inter_barrier_buf: vk.VkBuffer,
    inter_barrier_size: vk.VkDeviceSize,
    copy: CopyOp,
) DispatchError!DispatchResult {
    return submitTwoWithCopyLabeled(ctx, first, second, inter_barrier_buf, inter_barrier_size, copy, null);
}

/// Same as `submitTwoWithCopy` but takes an additional `copy_label`
/// for the trailing `vkCmdCopyBuffer`. The per-dispatch labels come
/// from `first.label` / `second.label` (DispatchSpec fields). All
/// labels are `VK_EXT_debug_utils` regions; `null` skips the
/// corresponding emit (silent no-op when the extension is absent).
pub fn submitTwoWithCopyLabeled(
    ctx: *driver_mod.Context,
    first: DispatchSpec,
    second: DispatchSpec,
    inter_barrier_buf: vk.VkBuffer,
    inter_barrier_size: vk.VkDeviceSize,
    copy: CopyOp,
    copy_label: ?[]const u8,
) DispatchError!DispatchResult {
    if (!ctx.initialized or ctx.dev == null or ctx.queue == null) {
        return error.LoaderNotReady;
    }
    try ensureChassis(ctx);

    const reset_cb = vk.vkResetCommandBuffer_fn orelse return error.LoaderNotReady;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return error.LoaderNotReady;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return error.LoaderNotReady;
    const cmd_reset_qp = vk.vkCmdResetQueryPool_fn orelse return error.LoaderNotReady;
    const cmd_write_ts = vk.vkCmdWriteTimestamp_fn orelse return error.LoaderNotReady;
    const cmd_bind_pl = vk.vkCmdBindPipeline_fn orelse return error.LoaderNotReady;
    const cmd_dispatch = vk.vkCmdDispatch_fn orelse return error.LoaderNotReady;
    const cmd_copy = vk.vkCmdCopyBuffer_fn orelse return error.LoaderNotReady;
    const cmd_barrier = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
    const cmd_bind_ds = vk.vkCmdBindDescriptorSets_fn orelse return error.LoaderNotReady;
    const reset_fence = vk.vkResetFences_fn orelse return error.LoaderNotReady;
    const submit = vk.vkQueueSubmit_fn orelse return error.LoaderNotReady;
    const wait_fence = vk.vkWaitForFences_fn orelse return error.LoaderNotReady;
    const get_results = vk.vkGetQueryPoolResults_fn orelse return error.LoaderNotReady;

    const t_record_begin = qpcNow();
    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return error.ResetCommandBufferFailed;

    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) {
        return error.BeginCommandBufferFailed;
    }

    cmd_reset_qp(ctx.cmd_buf, ctx.query_pool, 0, TS_SLOT_COUNT);

    // ── First dispatch (no timestamps — fast/trivial in the L1 path) ──
    cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, first.pipeline);
    if (first.descriptor_set != null and first.pipeline_layout != null) {
        const sets: [1]vk.VkDescriptorSet = .{first.descriptor_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            first.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
    }
    if (first.push_constants_bytes.len > 0) {
        const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
        if (first.pipeline_layout == null) return error.LoaderNotReady;
        cmd_push(
            ctx.cmd_buf,
            first.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(first.push_constants_bytes.len),
            @ptrCast(first.push_constants_bytes.ptr),
        );
    }
    beginLabel(ctx.cmd_buf, first.label);
    cmd_dispatch(ctx.cmd_buf, first.group_count[0], first.group_count[1], first.group_count[2]);
    endLabel(ctx.cmd_buf, first.label);

    // ── Inter-dispatch barrier (compute write → compute read) ──
    // The first dispatch writes `inter_barrier_buf`; the second reads
    // it. A buffer-memory barrier with SHADER_WRITE → SHADER_READ on
    // COMPUTE_SHADER → COMPUTE_SHADER is the minimal correct sync.
    if (inter_barrier_buf != null) {
        const bbarrier: vk.VkBufferMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .buffer = inter_barrier_buf,
            .offset = 0,
            .size = inter_barrier_size,
        };
        const bbarriers: [1]vk.VkBufferMemoryBarrier = .{bbarrier};
        cmd_barrier(
            ctx.cmd_buf,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            null,
            1,
            @ptrCast(&bbarriers),
            0,
            null,
        );
    }

    // ── Second dispatch (timestamp-bracketed) ──
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_BEGIN);
    cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, second.pipeline);
    if (second.descriptor_set != null and second.pipeline_layout != null) {
        const sets: [1]vk.VkDescriptorSet = .{second.descriptor_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            second.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
    }
    if (second.push_constants_bytes.len > 0) {
        const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
        if (second.pipeline_layout == null) return error.LoaderNotReady;
        cmd_push(
            ctx.cmd_buf,
            second.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(second.push_constants_bytes.len),
            @ptrCast(second.push_constants_bytes.ptr),
        );
    }
    beginLabel(ctx.cmd_buf, second.label);
    cmd_dispatch(ctx.cmd_buf, second.group_count[0], second.group_count[1], second.group_count[2]);
    endLabel(ctx.cmd_buf, second.label);
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_END);

    // ── Compute → Transfer barrier + vkCmdCopyBuffer (skipped when
    // copy.size == 0 — used by the D2D override path that writes the
    // decoded bytes straight into the caller's VkBuffer). The COPY_BEGIN
    // / COPY_END timestamps are still written so the get_results call
    // below has a defined value to read; they collapse to a zero copy_ns
    // when nothing was recorded between them. ──
    const has_copy = copy.size != 0;
    if (has_copy) {
        const cbarrier: vk.VkBufferMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .buffer = copy.src,
            .offset = copy.src_offset,
            .size = copy.size,
        };
        const cbarriers: [1]vk.VkBufferMemoryBarrier = .{cbarrier};
        cmd_barrier(
            ctx.cmd_buf,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            null,
            1,
            @ptrCast(&cbarriers),
            0,
            null,
        );
    }

    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_COPY_BEGIN);
    if (has_copy) {
        const region: vk.VkBufferCopy = .{
            .srcOffset = copy.src_offset,
            .dstOffset = copy.dst_offset,
            .size = copy.size,
        };
        const regions: [1]vk.VkBufferCopy = .{region};
        beginLabel(ctx.cmd_buf, copy_label);
        cmd_copy(ctx.cmd_buf, copy.src, copy.dst, 1, @ptrCast(&regions));
        endLabel(ctx.cmd_buf, copy_label);
    }
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, TS_SLOT_COPY_END);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;
    const record_ns = qpcDeltaNs(t_record_begin, qpcNow());

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
    const t_submit_begin = qpcNow();
    if (submit(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) {
        return error.SubmitFailed;
    }
    const t_submit_end = qpcNow();
    const submit_call_ns = qpcDeltaNs(t_submit_begin, t_submit_end);

    const wait_result = wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS);
    const t_wait_end = qpcNow();
    const wait_call_ns = qpcDeltaNs(t_submit_end, t_wait_end);
    const submit_wait_ns = qpcDeltaNs(t_submit_begin, t_wait_end);
    if (wait_result == vk.VK_TIMEOUT) return error.FenceWaitTimeout;
    if (wait_result != vk.VK_SUCCESS) return error.FenceWaitFailed;

    var ts: [TS_SLOT_CHASSIS_COUNT]u64 = .{ 0, 0, 0, 0 };
    const t_query_begin = qpcNow();
    const result = get_results(
        ctx.dev,
        ctx.query_pool,
        0,
        TS_SLOT_CHASSIS_COUNT,
        @sizeOf(@TypeOf(ts)),
        @ptrCast(&ts),
        @sizeOf(u64),
        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
    );
    const query_read_ns = qpcDeltaNs(t_query_begin, qpcNow());
    if (result != vk.VK_SUCCESS) return error.QueryReadFailed;

    const delta_ticks: u64 = if (ts[TS_SLOT_END] >= ts[TS_SLOT_BEGIN])
        ts[TS_SLOT_END] - ts[TS_SLOT_BEGIN]
    else
        0;
    const period = if (ctx.timestamp_period_ns > 0.0) ctx.timestamp_period_ns else 1.0;
    const ns_f: f64 = @as(f64, @floatFromInt(delta_ticks)) * @as(f64, period);
    const ns: u64 = if (ns_f <= 0.0) 0 else @intFromFloat(ns_f);

    const copy_ticks: u64 = if (ts[TS_SLOT_COPY_END] >= ts[TS_SLOT_COPY_BEGIN])
        ts[TS_SLOT_COPY_END] - ts[TS_SLOT_COPY_BEGIN]
    else
        0;
    const copy_ns_f: f64 = @as(f64, @floatFromInt(copy_ticks)) * @as(f64, period);
    const copy_ns: u64 = if (copy_ns_f <= 0.0) 0 else @intFromFloat(copy_ns_f);

    return .{
        .ns = ns,
        .copy_ns = copy_ns,
        .submit_wait_wall_ns = submit_wait_ns,
        .record_wall_ns = record_ns,
        .submit_call_ns = submit_call_ns,
        .wait_call_ns = wait_call_ns,
        .query_read_ns = query_read_ns,
    };
}
