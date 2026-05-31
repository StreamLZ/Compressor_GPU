//! Smoke test for the M1 Vulkan loader: bring up vulkan-1.dll, create an
//! instance + logical device, query device properties, print one line of
//! identifying info, tear everything down.
//!
//! Wired in build.zig as `zig build hello-vulkan`. Exits non-zero on any
//! bring-up failure so CI catches a missing vulkan-1.dll / no compute-
//! capable device immediately.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    try driver.ensureInit();
    defer driver.deinit();

    // Query device properties for the print line. `vkGetPhysicalDeviceProperties`
    // is one of the instance-level fns resolved by instance.zig.
    const get_props = vk.vkGetPhysicalDeviceProperties_fn orelse {
        try w.print("[hello_vulkan] FATAL: vkGetPhysicalDeviceProperties unresolved\n", .{});
        return error.MissingProc;
    };
    var props: vk.VkPhysicalDeviceProperties = .{};
    get_props(driver.g_default.pd, &props);

    // deviceName is a fixed 256-byte buffer; slice off at the first NUL
    // before printing.
    const name_slice = blk: {
        var n: usize = 0;
        while (n < props.deviceName.len and props.deviceName[n] != 0) : (n += 1) {}
        break :blk props.deviceName[0..n];
    };

    const api = props.apiVersion;
    try w.print(
        "[hello_vulkan] device='{s}'  api={d}.{d}.{d}  queue_family_index={d}\n",
        .{
            name_slice,
            vk.VK_API_VERSION_MAJOR(api),
            vk.VK_API_VERSION_MINOR(api),
            vk.VK_API_VERSION_PATCH(api),
            driver.g_default.qfi,
        },
    );
}
