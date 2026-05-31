//! M2 smoke test: enumerate every Vulkan physical device on the host,
//! run the feature probe, print one summary line per device.
//!
//! Wired in build.zig as `zig build vk-probe`. Exits non-zero only on
//! loader-bring-up failure (no vulkan-1.dll, no instance) — per-device
//! probe results are reported but never fail the binary, because a box
//! with one Tier-1 dGPU + one Tier-2 iGPU should still print both.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const instance_mod = @import("instance.zig");
const probe_mod = @import("probe.zig");

const MAX_PHYSICAL_DEVICES: u32 = 16;

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // We can't reuse driver.ensureInit() because it picks ONE device and
    // creates a logical device on it — for the probe we want to enumerate
    // every physical device the loader reports without committing. Bring
    // up the loader + instance manually; skip device creation.
    if (!vk.init()) {
        try w.print("[vk-probe] FATAL: vulkan-1.dll loader init failed\n", .{});
        return error.LoaderInitFailed;
    }

    // Validation off — probe paths shouldn't surface any spurious warnings,
    // and we want this binary safe to run on machines without the layer.
    const inst = instance_mod.createInstance(false) catch |err| {
        try w.print("[vk-probe] FATAL: createInstance: {s}\n", .{@errorName(err)});
        return err;
    };
    defer instance_mod.destroyInstance(inst);

    const enumerate = vk.vkEnumeratePhysicalDevices_fn orelse {
        try w.print("[vk-probe] FATAL: vkEnumeratePhysicalDevices unresolved\n", .{});
        return error.MissingProc;
    };

    var count: u32 = 0;
    if (enumerate(inst, &count, null) != vk.VK_SUCCESS) {
        try w.print("[vk-probe] FATAL: enumeratePhysicalDevices(count) failed\n", .{});
        return error.EnumerateFailed;
    }
    if (count == 0) {
        try w.print("[vk-probe] no Vulkan physical devices reported\n", .{});
        return;
    }
    if (count > MAX_PHYSICAL_DEVICES) count = MAX_PHYSICAL_DEVICES;

    var devices: [MAX_PHYSICAL_DEVICES]vk.VkPhysicalDevice = @splat(null);
    const r = enumerate(inst, &count, @ptrCast(&devices));
    if (r != vk.VK_SUCCESS and r != vk.VK_INCOMPLETE) {
        try w.print("[vk-probe] FATAL: enumeratePhysicalDevices(fill) failed\n", .{});
        return error.EnumerateFailed;
    }

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const pd = devices[i];
        if (pd == null) continue;
        const result = probe_mod.probe(inst, pd);
        try printDeviceLine(w, i, result);
    }
}

fn printDeviceLine(w: anytype, index: u32, r: probe_mod.ProbeResult) !void {
    const major = vk.VK_API_VERSION_MAJOR(r.api_version);
    const minor = vk.VK_API_VERSION_MINOR(r.api_version);
    const patch = vk.VK_API_VERSION_PATCH(r.api_version);
    try w.print(
        "device[{d}]: {s} vendor=0x{X:0>4} api={d}.{d}.{d} subgroup={d}[{d}..{d}] tier={s} bda={s} int64={s} ts={s} sync2={s} int8={s} 8stor={s} nvpart={s}\n",
        .{
            index,
            r.device_name,
            r.vendor_id,
            major,
            minor,
            patch,
            r.subgroup_size,
            r.subgroup_size_min,
            r.subgroup_size_max,
            probe_mod.tierName(r.tier),
            yn(r.has_buffer_device_address),
            yn(r.has_shader_int64),
            yn(r.has_timeline_semaphore),
            yn(r.has_synchronization2),
            yn(r.has_shader_int8),
            yn(r.has_8bit_storage),
            yn(r.has_nv_subgroup_partitioned),
        },
    );
}

fn yn(b: bool) []const u8 {
    return if (b) "y" else "n";
}
