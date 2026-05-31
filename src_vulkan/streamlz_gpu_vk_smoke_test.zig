//! M4 smoke test for the Vulkan C ABI: invokes slzCreate_vk +
//! slzDestroy_vk through a Zig import (no DLL load) and prints a
//! one-line PASS/FAIL summary. Wired in build.zig as `zig build vk-smoke`.
//!
//! Success path: brings up the vulkan-1.dll loader, picks a device,
//! classifies its tier, allocates + frees a handle, prints
//! `vk smoke OK`. On any error (no Vulkan driver, no compute device,
//! tier == unsupported), prints `vk smoke FAIL: <status>` and exits 1
//! so CI catches the regression.

const std = @import("std");

const vk_abi = @import("streamlz_gpu_vk.zig");

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;

    var stdout_buf: [256]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    var handle: ?*vk_abi.VkContext = null;
    const rc = vk_abi.slzCreate_vk(&handle);
    if (rc != 0 or handle == null) {
        try w.print("vk smoke FAIL: {d}\n", .{rc});
        return error.CreateFailed;
    }
    defer vk_abi.slzDestroy_vk(handle);

    const ver = vk_abi.slzGetVersionString_vk();
    // Convert the C string back to a Zig slice for printing — span()
    // walks to the NUL.
    const ver_slice = std.mem.span(ver);
    try w.print("vk smoke OK (version={s})\n", .{ver_slice});
}
