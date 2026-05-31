//! M5 — `__match_any_sync` emulation microbenchmark.
//!
//! Foundation R1 retirement gate: measure how slow the 32-broadcast OR-
//! reduction is on Vulkan vs CUDA's 1-instruction `__match_any_sync`
//! (~1 ns/call per the foundation deep-dive). The result feeds the
//! milestone-plan rip-cord in §6: AMD/Intel must sustain ≥ 65% throughput
//! before M22a/M22b can target lz_encode site #1 on those vendors. NVIDIA
//! Tier-1+NV uses the partitioned-XOR fast path elsewhere; here we measure
//! the portable (non-NV) base emulation that AMD/Intel also run.
//!
//! Shape (mirrors src_vulkan/dispatch_test.zig step-for-step):
//!   1. driver.ensureInit               (instance + device + queue)
//!   2. probe                           (pick tier — tier1 / tier1_nv / tier2)
//!   3. loadSpv("match_any_bench", tier)
//!   4. Two 128-byte device-local + host-visible SSBOs (32 lanes × 4 B):
//!        binding 0 = read-only input keys, prefilled to identity
//!        binding 1 = write-only output masks, ignored after the run
//!   5. descriptors.getOrCreate(..., n_storage_buffers=2, push_const_size=4)
//!   6. 10 warmup dispatches + 100 measured dispatches @ pc.n_iters = 1024
//!   7. Per dispatch GPU ns from the M8a timestamp pair / 1024 emulations
//!   8. Print "match_any_bench: best <X>ns/call, mean <Y>ns/call,
//!            slowdown <Z>x vs CUDA(~1ns)"
//!
//! No allocator: every host-side buffer is stack-bounded. The .spv is
//! loaded into a 1-MiB stack array via the same loadSpv helper shape as
//! dispatch_test.zig.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");

// ── Tuning ──────────────────────────────────────────────────────────

/// One 32-wide subgroup × 4 B per lane = 128 B per SSBO.
const SSBO_BYTES: vk.VkDeviceSize = 128;
const LANE_COUNT: u32 = 32;

/// Number of emulations per dispatch — amortizes Vulkan submit/fence
/// overhead (~µs) over many shader-internal emulations so the per-call
/// number reflects GPU work, not submit latency.
const N_ITERS_PER_DISPATCH: u32 = 1024;

const WARMUP_DISPATCHES: u32 = 10;
const MEASURED_DISPATCHES: u32 = 100;

/// CUDA reference from foundation deep-dive `match-any-sync-callsites-
/// and-emulation` § R1: a single `__match_any_sync` instruction on
/// NVIDIA hardware retires in ~1 ns at typical clock rates.
const CUDA_REF_NS_PER_CALL: f64 = 1.0;

const MAX_SPV_BYTES: usize = 1 << 20; // 1 MiB — well above the shell sizes.
const SPV_DIR_REL: []const u8 = "zig-out/shaders";

const BenchError = error{
    NoBlobForTier,
    UnsupportedTier,
    SpvOpenFailed,
    SpvReadFailed,
    SpvTooLarge,
    BufferCreateFailed,
    MemoryTypeNotFound,
    MemoryAllocateFailed,
    BindBufferFailed,
    MapMemoryFailed,
};

// ── Helpers (copied 1:1 from dispatch_test.zig where applicable) ────

fn probeTierName(t: probe_mod.Tier) ?[]const u8 {
    return switch (t) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => null,
    };
}

fn loadSpv(io: std.Io, kernel: []const u8, tier_name: []const u8, dest: []u8) ![]u8 {
    var path_buf: [256]u8 = undefined;
    const filename = try std.fmt.bufPrint(&path_buf, "{s}/{s}.{s}.spv", .{ SPV_DIR_REL, kernel, tier_name });
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch return error.SpvOpenFailed;
    defer file.close(io);
    const body = dest[256..];
    const n = file.readPositionalAll(io, body, 0) catch return error.SpvReadFailed;
    if (n == body.len) return error.SpvTooLarge;
    return body[0..n];
}

/// Mirrors dispatch_test.zig::resolveDeviceFn — same fallback chain.
fn resolveDeviceFn(comptime T: type, dev: vk.VkDevice, name: [*:0]const u8) ?T {
    if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
        if (gdpa(dev, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    if (vk.vkGetInstanceProcAddr_fn) |gipa| {
        const inst = driver.g_default.inst;
        if (gipa(inst, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    return null;
}

fn ensureBufferFnSlots(ctx: *driver.Context) void {
    if (vk.vkCreateBuffer_fn == null)
        vk.vkCreateBuffer_fn = resolveDeviceFn(vk.FnCreateBuffer, ctx.dev, "vkCreateBuffer");
    if (vk.vkDestroyBuffer_fn == null)
        vk.vkDestroyBuffer_fn = resolveDeviceFn(vk.FnDestroyBuffer, ctx.dev, "vkDestroyBuffer");
    if (vk.vkAllocateMemory_fn == null)
        vk.vkAllocateMemory_fn = resolveDeviceFn(vk.FnAllocateMemory, ctx.dev, "vkAllocateMemory");
    if (vk.vkFreeMemory_fn == null)
        vk.vkFreeMemory_fn = resolveDeviceFn(vk.FnFreeMemory, ctx.dev, "vkFreeMemory");
    if (vk.vkBindBufferMemory_fn == null)
        vk.vkBindBufferMemory_fn = resolveDeviceFn(vk.FnBindBufferMemory, ctx.dev, "vkBindBufferMemory");
    if (vk.vkMapMemory_fn == null)
        vk.vkMapMemory_fn = resolveDeviceFn(vk.FnMapMemory, ctx.dev, "vkMapMemory");
    if (vk.vkUnmapMemory_fn == null)
        vk.vkUnmapMemory_fn = resolveDeviceFn(vk.FnUnmapMemory, ctx.dev, "vkUnmapMemory");
    if (vk.vkGetBufferMemoryRequirements_fn == null)
        vk.vkGetBufferMemoryRequirements_fn = resolveDeviceFn(vk.FnGetBufferMemoryRequirements, ctx.dev, "vkGetBufferMemoryRequirements");
    if (vk.vkGetPhysicalDeviceMemoryProperties_fn == null) {
        if (vk.vkGetInstanceProcAddr_fn) |gipa| {
            if (gipa(driver.g_default.inst, "vkGetPhysicalDeviceMemoryProperties")) |raw| {
                vk.vkGetPhysicalDeviceMemoryProperties_fn = @ptrCast(@alignCast(raw));
            }
        }
    }
}

fn findMemoryType(
    pd: vk.VkPhysicalDevice,
    type_bits_mask: u32,
    required_flags: vk.VkMemoryPropertyFlags,
) ?u32 {
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return null;
    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(pd, &mem_props);
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (type_bits_mask & (@as(u32, 1) << @intCast(i))) != 0;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        const has_required = (flags & required_flags) == required_flags;
        if (supported and has_required) return i;
    }
    return null;
}

/// One owned buffer + its backing memory + the host map. Identical shape
/// to dispatch_test.zig's TestBuffer — the bench uses two such buffers
/// (one for the input keys, one for the per-lane mask output).
const TestBuffer = struct {
    buf: vk.VkBuffer = null,
    mem: vk.VkDeviceMemory = null,
    mapped: ?[*]u8 = null,
    size: vk.VkDeviceSize = 0,
};

fn createTestBuffer(ctx: *driver.Context, size: vk.VkDeviceSize) !TestBuffer {
    ensureBufferFnSlots(ctx);
    const create_buf = vk.vkCreateBuffer_fn orelse return error.BufferCreateFailed;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse return error.BufferCreateFailed;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return error.MemoryAllocateFailed;
    const bind = vk.vkBindBufferMemory_fn orelse return error.BindBufferFailed;
    const map = vk.vkMapMemory_fn orelse return error.MapMemoryFailed;

    const bci: vk.VkBufferCreateInfo = .{
        .size = size,
        .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) {
        return error.BufferCreateFailed;
    }

    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    const ideal = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const fallback = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const mt_idx = findMemoryType(ctx.pd, req.memoryTypeBits, ideal) orelse
        findMemoryType(ctx.pd, req.memoryTypeBits, fallback) orelse {
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MemoryTypeNotFound;
    };

    const mai: vk.VkMemoryAllocateInfo = .{
        .allocationSize = req.size,
        .memoryTypeIndex = mt_idx,
    };
    var mem: vk.VkDeviceMemory = null;
    if (alloc_mem(ctx.dev, &mai, null, &mem) != vk.VK_SUCCESS) {
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MemoryAllocateFailed;
    }
    if (bind(ctx.dev, buf, mem, 0) != vk.VK_SUCCESS) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.BindBufferFailed;
    }

    var mapped_raw: ?*anyopaque = null;
    if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &mapped_raw) != vk.VK_SUCCESS) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MapMemoryFailed;
    }
    const mapped_bytes: [*]u8 = @ptrCast(@alignCast(mapped_raw orelse return error.MapMemoryFailed));
    @memset(mapped_bytes[0..size], 0);
    return .{ .buf = buf, .mem = mem, .mapped = mapped_bytes, .size = size };
}

fn destroyTestBuffer(ctx: *driver.Context, tb: *TestBuffer) void {
    if (tb.mapped != null) {
        if (vk.vkUnmapMemory_fn) |u| u(ctx.dev, tb.mem);
        tb.mapped = null;
    }
    if (tb.buf != null) {
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, tb.buf, null);
        tb.buf = null;
    }
    if (tb.mem != null) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, tb.mem, null);
        tb.mem = null;
    }
}

// ── Entry point ─────────────────────────────────────────────────────

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // 1. Loader + instance + device + queue.
    try driver.ensureInit();
    defer driver.deinit();
    const ctx = &driver.g_default;

    // 2. Probe to learn the tier; same dispatch as dispatch_test.zig.
    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    const tier_name = probeTierName(pr.tier) orelse {
        try w.print("match_any_bench FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    // 3. Load the match_any_bench .spv blob for the probed tier.
    var spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const spv = loadSpv(io, "match_any_bench", tier_name, &spv_storage) catch |err| {
        try w.print("match_any_bench FAIL spv_load err={s} tier={s}\n", .{ @errorName(err), tier_name });
        return err;
    };

    // 4. Create two 128-byte SSBOs (32 lanes × 4 B). HOST_VISIBLE +
    //    HOST_COHERENT so we can preload the input identity vector
    //    without a staging copy, and so we can sanity-check the output
    //    when validating.
    var in_buf = try createTestBuffer(ctx, SSBO_BYTES);
    defer destroyTestBuffer(ctx, &in_buf);
    var out_buf = try createTestBuffer(ctx, SSBO_BYTES);
    defer destroyTestBuffer(ctx, &out_buf);

    // Preload input keys with the identity 0..31 — each lane sees a
    // unique key so the emulation's expected group_mask for lane k is
    // exactly `(1 << k)` (lane k matches only itself). This makes
    // post-run validation trivial.
    {
        const in_words: [*]u32 = @ptrCast(@alignCast(in_buf.mapped.?));
        var i: u32 = 0;
        while (i < LANE_COUNT) : (i += 1) in_words[i] = i;
    }

    // 5. Build (or fetch from cache) the pipeline + descriptor-set
    //    layout. 2 storage buffers + a 4-byte push constant (n_iters).
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);
    const cached = try descriptors.getOrCreate(
        ctx,
        &cache,
        "match_any_bench",
        pr.tier,
        spv,
        2, // binding 0 = in_buf, binding 1 = out_buf
        @sizeOf(u32), // push constant: pc.n_iters
    );

    // 6. Allocate a descriptor set once and reuse across every dispatch.
    //    The buffers don't change between iterations; only the push
    //    constant changes (and even that is constant across the run).
    const buf_infos: [2]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = in_buf.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = out_buf.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dset = try descriptors.allocSet(ctx, cached, buf_infos[0..]);

    // Push-constant value — fixed across the run; just copy the u32 into
    // a 4-byte byte slice for the dispatch.submitOne signature.
    const n_iters: u32 = N_ITERS_PER_DISPATCH;
    var pc_bytes: [4]u8 = undefined;
    @memcpy(pc_bytes[0..], std.mem.asBytes(&n_iters));

    // 7. Warmup — burn 10 dispatches to amortize first-call overhead
    //    (shader compile lazily JITs on first dispatch on some drivers,
    //    descriptor pool fills, etc.).
    var wi: u32 = 0;
    while (wi < WARMUP_DISPATCHES) : (wi += 1) {
        _ = try dispatch.submitOne(
            ctx,
            cached.pipeline,
            cached.pipeline_layout,
            dset,
            pc_bytes[0..],
            .{ 1, 1, 1 },
        );
    }

    // 8. Measured runs — record per-dispatch GPU ns from the M8a
    //    timestamp pair. Track best (min) + sum for mean. Best is the
    //    "no scheduling noise" number; mean is the "what the user will
    //    actually see" number.
    var best_ns: u64 = std.math.maxInt(u64);
    var sum_ns: u64 = 0;
    var mi: u32 = 0;
    while (mi < MEASURED_DISPATCHES) : (mi += 1) {
        const result = try dispatch.submitOne(
            ctx,
            cached.pipeline,
            cached.pipeline_layout,
            dset,
            pc_bytes[0..],
            .{ 1, 1, 1 },
        );
        if (result.ns < best_ns) best_ns = result.ns;
        sum_ns += result.ns;
    }

    // 9. Compute per-emulation cost: each dispatch does N_ITERS_PER_DISPATCH
    //    emulations across the warp. The 32 lanes do the same emulation
    //    in lockstep — we deliberately report per-call cost (one
    //    emulation across the whole warp), NOT per-lane, since CUDA's
    //    `__match_any_sync` is also a per-warp operation.
    const iters_f: f64 = @floatFromInt(N_ITERS_PER_DISPATCH);
    const best_per_call: f64 = @as(f64, @floatFromInt(best_ns)) / iters_f;
    const mean_total_ns: f64 = @as(f64, @floatFromInt(sum_ns)) / @as(f64, @floatFromInt(MEASURED_DISPATCHES));
    const mean_per_call: f64 = mean_total_ns / iters_f;
    const slowdown: f64 = best_per_call / CUDA_REF_NS_PER_CALL;

    // 10. Print. Format is one machine-parseable line for the milestone
    //     plan's pre-committed-floor check + a human-readable summary on
    //     a separate line for log scanning.
    try w.print(
        "match_any_bench: best {d:.3}ns/call, mean {d:.3}ns/call, slowdown {d:.2}x vs CUDA(~1ns)\n",
        .{ best_per_call, mean_per_call, slowdown },
    );
    try w.print(
        "  tier={s} device=\"{s}\" subgroup={d} iters_per_dispatch={d} warmup={d} measured={d}\n",
        .{
            tier_name,
            pr.device_name,
            pr.subgroup_size,
            N_ITERS_PER_DISPATCH,
            WARMUP_DISPATCHES,
            MEASURED_DISPATCHES,
        },
    );

    // Optional: a one-line sanity check on the output buffer.
    //
    // With identity input keys (lane k = k), every lane k sees:
    //   • Iter 0:                group_mask  = (1 << k)        (only lane k matches its own key)
    //   • Iter i ∈ [1..n_iters): sink         = (1 << k)        (lane k still uniquely matches k+i)
    //                            group_mask ^= sink             (XOR flips bit k)
    // So bit k is toggled `n_iters` times total starting from 0
    // (one set + (n_iters − 1) XOR-toggles ⇒ n_iters net toggles).
    // The final per-lane mask is therefore:
    //   • (1 << k) when n_iters is odd
    //   • 0        when n_iters is even   (our N_ITERS_PER_DISPATCH = 1024)
    const out_words: [*]const u32 = @ptrCast(@alignCast(out_buf.mapped.?));
    var ok = true;
    var k: u32 = 0;
    while (k < LANE_COUNT) : (k += 1) {
        const expected: u32 = if ((N_ITERS_PER_DISPATCH & 1) == 1)
            (@as(u32, 1) << @intCast(k))
        else
            0;
        if (out_words[k] != expected) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        try w.print("  WARN: output mask mismatch — emulation may be incorrect on this device\n", .{});
    }
}
