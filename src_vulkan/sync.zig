//! M8c: sync2/sync1 buffer-barrier wrapper.
//!
//! Architecture §8 / §9 calls for a single VkQueue compute pipeline whose
//! intra-command-buffer ordering is enforced by per-buffer barriers
//! between dependent dispatches. Tier-1 devices that report
//! `synchronization2 == VK_TRUE` use `vkCmdPipelineBarrier2` with
//! 64-bit-mask `VkBufferMemoryBarrier2` (carrying per-barrier stage masks);
//! Tier-2 / sync2-absent devices fall back to `vkCmdPipelineBarrier` with
//! the 32-bit-mask sync1 `VkBufferMemoryBarrier`.
//!
//! Callers always speak in sync1-shape `VkBufferMemoryBarrier` plus
//! 32-bit `src_stage` / `dst_stage` (the bit values for COMPUTE_SHADER,
//! TOP_OF_PIPE, BOTTOM_OF_PIPE, TRANSFER are stable between sync1 and
//! sync2). The wrapper translates to sync2's per-barrier stage masks
//! when the probe flag is set.
//!
//! No allocations: the sync2 translation array is stack-bounded at
//! `MAX_BUFFER_BARRIERS`. Per-dispatch barrier counts in production are
//! ≤ 12 (mergeHuffDescs is the worst case at architecture review time);
//! 16 is the conservative cap.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver_mod = @import("driver.zig");

/// Stack-allocated cap on the per-call buffer-barrier count. Architecture
/// audit at M8c-time shows 12 as the worst-case (mergeHuffDescs); 16 is
/// the conservative upper bound. Bumping this changes only the stack
/// array in `barrier` — no API churn.
pub const MAX_BUFFER_BARRIERS: usize = 16;

pub const SyncError = error{
    LoaderNotReady,
    TooManyBarriers,
};

/// Resolve a device-level entry point — same fallback chain as dispatch.zig
/// / descriptors.zig: prefer vkGetDeviceProcAddr, fall back to the
/// instance-level thunk via vkGetInstanceProcAddr.
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

/// Populate the sync1 + (when supported) sync2 entry points. Idempotent.
/// Picks `vkCmdPipelineBarrier2` (Vulkan 1.3 core) by name; on a Vulkan
/// 1.2 device the KHR-suffixed entry point exists as a fallback (when
/// the extension was enabled at device-create) — we do NOT try the KHR
/// name at M8c because device.zig does not enable VK_KHR_synchronization2
/// yet. Production milestones that need sync2 on Vulkan 1.2 must enable
/// the extension AND fall back to "vkCmdPipelineBarrier2KHR" here.
fn ensureFnSlots(ctx: *driver_mod.Context) SyncError!void {
    if (ctx.dev == null) return error.LoaderNotReady;

    if (vk.vkCmdPipelineBarrier_fn == null)
        vk.vkCmdPipelineBarrier_fn = resolveDeviceFn(vk.FnCmdPipelineBarrier, ctx.dev, "vkCmdPipelineBarrier");

    if (vk.vkCmdPipelineBarrier2_fn == null) {
        vk.vkCmdPipelineBarrier2_fn = resolveDeviceFn(vk.FnCmdPipelineBarrier2, ctx.dev, "vkCmdPipelineBarrier2");
    }
}

/// Map a sync1 access bit to its sync2 equivalent. The named bit values
/// happen to be identical for the bits both versions define (SHADER_READ,
/// SHADER_WRITE, TRANSFER_READ, TRANSFER_WRITE, HOST_READ, HOST_WRITE),
/// so the conversion is a zero-extend.
fn accessSync1To2(a: vk.VkAccessFlags) vk.VkAccessFlags2 {
    return @as(vk.VkAccessFlags2, a);
}

/// Map a sync1 pipeline-stage bit to its sync2 equivalent. Same
/// stability argument as accessSync1To2 above.
fn stageSync1To2(s: vk.VkPipelineStageFlags) vk.VkPipelineStageFlags2 {
    return @as(vk.VkPipelineStageFlags2, s);
}

/// Emit a buffer barrier across `src_stage → dst_stage` for every
/// barrier in `buffer_barriers`. Picks `vkCmdPipelineBarrier2` when the
/// probe says `has_synchronization2`; otherwise falls back to
/// `vkCmdPipelineBarrier` (sync1).
///
/// The caller-facing surface is sync1-shaped — `VkBufferMemoryBarrier`
/// + the 32-bit stage masks — so production code doesn't have to thread
/// the probe flag through every call site. The wrapper translates as
/// needed.
///
/// `buffer_barriers.len == 0` is a no-op (no barrier emitted) rather
/// than an error — useful for code paths that compute the barrier list
/// dynamically and may have nothing to emit.
pub fn barrier(
    ctx: *driver_mod.Context,
    cmd: vk.VkCommandBuffer,
    src_stage: u32,
    dst_stage: u32,
    buffer_barriers: []const vk.VkBufferMemoryBarrier,
) SyncError!void {
    if (ctx.dev == null) return error.LoaderNotReady;
    if (buffer_barriers.len == 0) return;
    if (buffer_barriers.len > MAX_BUFFER_BARRIERS) return error.TooManyBarriers;

    try ensureFnSlots(ctx);

    // Pick sync2 only when (a) the probe says it's supported AND
    // (b) the entry point actually resolved. Either failing → sync1.
    if (ctx.has_synchronization2) {
        if (vk.vkCmdPipelineBarrier2_fn) |barrier2| {
            // Translate each sync1 barrier into a sync2 barrier with
            // per-barrier stage masks (cloned from the call-site stage
            // args). Production milestones that need different stages
            // per barrier will plumb VkBufferMemoryBarrier2 directly
            // through a separate entry point; M8c keeps the surface
            // simple (one stage pair for the whole batch).
            var sync2_bars: [MAX_BUFFER_BARRIERS]vk.VkBufferMemoryBarrier2 = @splat(.{});
            const src_stage2 = stageSync1To2(src_stage);
            const dst_stage2 = stageSync1To2(dst_stage);
            var i: usize = 0;
            while (i < buffer_barriers.len) : (i += 1) {
                const b = buffer_barriers[i];
                sync2_bars[i] = .{
                    .srcStageMask = src_stage2,
                    .srcAccessMask = accessSync1To2(b.srcAccessMask),
                    .dstStageMask = dst_stage2,
                    .dstAccessMask = accessSync1To2(b.dstAccessMask),
                    .srcQueueFamilyIndex = b.srcQueueFamilyIndex,
                    .dstQueueFamilyIndex = b.dstQueueFamilyIndex,
                    .buffer = b.buffer,
                    .offset = b.offset,
                    .size = b.size,
                };
            }
            const dep_info: vk.VkDependencyInfo = .{
                .bufferMemoryBarrierCount = @intCast(buffer_barriers.len),
                .pBufferMemoryBarriers = @ptrCast(&sync2_bars),
            };
            barrier2(cmd, &dep_info);
            return;
        }
        // Fell through — sync2 was probed-supported but the entry point
        // didn't resolve. This shouldn't happen on a well-formed driver
        // but the sync1 fallback below handles it gracefully.
    }

    const barrier1 = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
    barrier1(
        cmd,
        src_stage,
        dst_stage,
        0, // dependencyFlags
        0, // memoryBarrierCount
        null,
        @intCast(buffer_barriers.len),
        @ptrCast(buffer_barriers.ptr),
        0, // imageMemoryBarrierCount
        null,
    );
}

// ── Tests ────────────────────────────────────────────────────────
// The translation helpers are pure functions; the dispatch path needs a
// live VkDevice and lives in the M8c integration suite.

test "stage/access sync1→sync2 is a zero-extend" {
    const s = stageSync1To2(vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    try std.testing.expectEqual(@as(vk.VkPipelineStageFlags2, 0x800), s);
    const a = accessSync1To2(vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT);
    try std.testing.expectEqual(@as(vk.VkAccessFlags2, 0x60), a);
}
