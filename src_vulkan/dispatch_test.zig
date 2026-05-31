//! M8c end-to-end dispatch test.
//!
//! The full vertical slice for M8c's "ship it" gate:
//!   1. driver.ensureInit                       (instance + device + queue)
//!   2. probe                                   (pick tier — tier1/tier1_nv/tier2)
//!   3. spv_blobs.find("lz_encode", tier)       (the M6 shell)
//!   4. Create a 4 KiB HOST_VISIBLE + DEVICE_LOCAL VkBuffer with
//!      STORAGE_BUFFER usage (single-buffer read-back simplicity).
//!   5. descriptors.getOrCreate(..., n_storage_buffers=1)
//!   6. descriptors.allocSet(...) writing the full buffer as binding 0.
//!   7. dispatch.submitOne(group_count=.{1,1,1})
//!   8. Read back data[0] via the host-mapped pointer.
//!   9. Assert == 0xDEADBEEF — the magic the lz_encode shell writes
//!      (see src_vulkan/shaders/lz_encode.comp).
//!
//! Output line on success: `dispatch_test PASS ns=<elapsed>`.
//! Output line on failure: `dispatch_test FAIL got=0x<hex>`.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");

const BUFFER_BYTES: vk.VkDeviceSize = 4096;
const EXPECTED_MAGIC: u32 = 0xDEADBEEF;
const MAX_SPV_BYTES: usize = 1 << 20; // 1 MiB; the M6 shells top out ~1 KiB

/// Hard-coded relative path to the shader directory `vk-shaders` produces.
/// dispatch_test is intentionally not wired through spv_blobs.zig (which
/// uses `@embedFile` and depends on the M4 streamlz_vk module's eventual
/// embed-dir wiring): loading from disk at runtime is fine for a test and
/// sidesteps the build-system embed-path plumbing entirely.
const SPV_DIR_REL: []const u8 = "zig-out/shaders";

const TestError = error{
    NoBlobForTier,
    MemoryTypeNotFound,
    BufferCreateFailed,
    MemoryAllocateFailed,
    BindBufferFailed,
    MapMemoryFailed,
    UnsupportedTier,
    SpvOpenFailed,
    SpvReadFailed,
    SpvTooLarge,
};

/// Convert a probe.Tier to the "tier1" / "tier1_nv" / "tier2" filename
/// substring `vk-shaders` uses. Returns null for `unsupported` since no
/// .spv is built for that tier.
fn probeTierName(t: probe_mod.Tier) ?[]const u8 {
    return switch (t) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => null,
    };
}

/// Load `<SPV_DIR_REL>/<kernel>.<tier>.spv` into `dest` and return the
/// populated slice. dest is the caller-owned read buffer (stack-allocated
/// at MAX_SPV_BYTES in main); we never allocate.
///
/// The path scratch (first 256 B of `dest`) holds the formatted filename;
/// the SPV body fills the tail starting at offset 256, which preserves
/// the caller's 4-byte alignment requirement (256 % 4 == 0).
fn loadSpv(io: std.Io, kernel: []const u8, tier_name: []const u8, dest: []u8) ![]u8 {
    var path_buf: [256]u8 = undefined;
    const filename = try std.fmt.bufPrint(&path_buf, "{s}/{s}.{s}.spv", .{ SPV_DIR_REL, kernel, tier_name });

    // Open relative to the process cwd — `zig build vk-dispatch-test`
    // runs from the project root, so the relative path resolves.
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch return error.SpvOpenFailed;
    defer file.close(io);

    // Read into the tail of dest (skipping the 256-byte path scratch at
    // the head) so the body stays 4-byte aligned for SPV.
    const body = dest[256..];
    const n = file.readPositionalAll(io, body, 0) catch return error.SpvReadFailed;
    if (n == body.len) return error.SpvTooLarge;
    return body[0..n];
}

/// Resolve every M8c device-level fn we need that wasn't pre-populated
/// by an earlier dispatch.zig / descriptors.zig call. Same lazy pattern.
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

    // vkGetPhysicalDeviceMemoryProperties is instance-level — resolve via
    // vkGetInstanceProcAddr.
    if (vk.vkGetPhysicalDeviceMemoryProperties_fn == null) {
        if (vk.vkGetInstanceProcAddr_fn) |gipa| {
            if (gipa(driver.g_default.inst, "vkGetPhysicalDeviceMemoryProperties")) |raw| {
                vk.vkGetPhysicalDeviceMemoryProperties_fn = @ptrCast(@alignCast(raw));
            }
        }
    }
}

/// Find a memory type index whose bit is set in `type_bits_mask` AND whose
/// propertyFlags include every bit in `required_flags`. Returns null when
/// no such type exists (caller should retry with weaker flags).
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

/// Owns the one VkBuffer + VkDeviceMemory the test uses for read-back.
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

    // 1. Create the buffer — STORAGE_BUFFER for the shader binding,
    //    TRANSFER_DST for completeness (some drivers prefer it on a
    //    buffer that will be written from the GPU side too).
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

    // 2. Query the buffer's memory requirements (size, alignment,
    //    allowed memory-type mask).
    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    // 3. Pick a memory type that's HOST_VISIBLE + HOST_COHERENT (so we
    //    can map+read without a flush) AND ideally DEVICE_LOCAL (so the
    //    dispatch hits fast memory). Most desktop drivers expose at
    //    least one type that's all three; fall back to host-visible
    //    only if not.
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

    // 4. Map the memory for host-side read-back. We zero the buffer
    //    first so a "shader didn't write" failure shows up as 0 (not
    //    leftover memory) for clearer diagnostics.
    var mapped_raw: ?*anyopaque = null;
    if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &mapped_raw) != vk.VK_SUCCESS) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MapMemoryFailed;
    }
    const mapped_bytes: [*]u8 = @ptrCast(@alignCast(mapped_raw orelse return error.MapMemoryFailed));
    // Zero the buffer up to the caller-requested size (req.size may be
    // padded above `size` by the driver; only zero what we promised).
    @memset(mapped_bytes[0..size], 0);

    return .{
        .buf = buf,
        .mem = mem,
        .mapped = mapped_bytes,
        .size = size,
    };
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

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // 1. Bring up the loader + instance + (device, queue).
    try driver.ensureInit();
    defer driver.deinit();
    const ctx = &driver.g_default;

    // 2. Probe to learn the tier (which selects the .spv variant).
    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    const tier_name = probeTierName(pr.tier) orelse {
        try w.print("dispatch_test FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    // 3. Load the lz_encode .spv blob for the probed tier from disk. The
    //    `vk-shaders` step deposits the SPVs under `zig-out/shaders/` and
    //    the test's build rule depends on it, so they're present.
    //    Read buffer is split inside loadSpv: head 256 B for the path
    //    string, tail for the SPV body. SPV bodies for the M6 shells
    //    are well under MAX_SPV_BYTES.
    var spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const spv = loadSpv(io, "lz_encode", tier_name, &spv_storage) catch |err| {
        try w.print("dispatch_test FAIL spv_load err={s} tier={s}\n", .{ @errorName(err), tier_name });
        return err;
    };

    // 4. Create a 4 KiB host-visible + device-local buffer for the
    //    shader to write into.
    var tb = try createTestBuffer(ctx, BUFFER_BYTES);
    defer destroyTestBuffer(ctx, &tb);

    // 5. Build (or fetch from cache) the pipeline + descriptor-set
    //    layout for lz_encode at the probed tier.
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);
    const cached = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_encode",
        pr.tier,
        spv,
        1, // n_storage_buffers — lz_encode shell binds 1 SSBO at binding=0
        0, // push_const_size — none
    );

    // 6. Allocate a descriptor set and bind the whole test buffer to
    //    binding=0.
    const buf_infos: [1]vk.VkDescriptorBufferInfo = .{.{
        .buffer = tb.buf,
        .offset = 0,
        .range = vk.VK_WHOLE_SIZE,
    }};
    const dset = try descriptors.allocSet(ctx, cached, buf_infos[0..]);

    // 7. Submit one dispatch (1 workgroup is enough — the shell only
    //    writes data[0] when gl_GlobalInvocationID.x == 0).
    const result = try dispatch.submitOne(
        ctx,
        cached.pipeline,
        cached.pipeline_layout,
        dset,
        &.{}, // no push constants
        .{ 1, 1, 1 },
    );

    // 8. Read back data[0] and check the magic. HOST_COHERENT means
    //    we don't need a vkInvalidateMappedMemoryRanges; the fence
    //    wait inside submitOne is the final ordering guarantee for
    //    the read.
    const got_u32: *const u32 = @ptrCast(@alignCast(tb.mapped.?));
    const got = got_u32.*;

    // 9. Print PASS/FAIL.
    if (got == EXPECTED_MAGIC) {
        try w.print("dispatch_test PASS ns={d}\n", .{result.ns});
    } else {
        try w.print("dispatch_test FAIL got=0x{x}\n", .{got});
        return error.MagicMismatch;
    }
}
