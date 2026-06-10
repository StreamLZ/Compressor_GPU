//! 1:1 port of src/cli.zig.
//!
//! StreamLZ command-line interface dispatcher. Routes -c / -d / -b / -db
//! / -ba / -i modes to the per-mode handlers under srcVK/cli/.
//!
//! VK adaptations beyond the CUDA dispatcher:
//!  * Accepts `--device <N|name>` to select a Vulkan physical device.
//!  * Reads `SLZ_VK_DEVICE_INDEX` as an env fallback.
//!  * Prints the bound `vkPhysicalDeviceProperties.deviceName` once at
//!    startup (`Device: NVIDIA GeForce RTX 4060 Ti`) before dispatching.
//!  * `--probe` mode enumerates every physical device and exits.

const std = @import("std");

const util = @import("cli/util.zig");
const compress_cmd = @import("cli/compress.zig");
const decompress_cmd = @import("cli/decompress.zig");
const bench_compress_cmd = @import("cli/bench_compress.zig");
const bench_decompress_cmd = @import("cli/bench_decompress.zig");
const bench_all_cmd = @import("cli/bench_all.zig");
const info_cmd = @import("cli/info.zig");
const decode_module_loader = @import("decode/module_loader.zig");
const decode_driver = @import("decode/driver.zig");

/// VK adaptation: stage device selection + bring up the GPU driver, then
/// print the bound deviceName. CUDA's CLI does none of this (cuInit
/// happens lazily inside the driver on first kernel launch).
fn ensureGpuInitAndPrintDevice(w: *std.Io.Writer, args: util.Args) !void {
    const sel = util.selectorFromSpec(args.device_spec);
    decode_module_loader.setDeviceSelector(switch (sel) {
        .default => .default,
        .by_index => |i| .{ .by_index = i },
        .by_name => |n| .{ .by_name = n },
    });
    if (!decode_driver.init()) {
        try w.print("error: vulkan init failed (device_spec={?s})\n", .{args.device_spec});
        try w.flush();
        std.process.exit(2);
    }
    var name_buf: [256]u8 = undefined;
    const name = decode_module_loader.readBoundDeviceName(name_buf[0..]);
    if (name.len > 0) {
        try w.print("Device: {s}\n", .{name});
    } else {
        try w.writeAll("Device: (unknown)\n");
    }
}

/// VK adaptation: --probe mode. Stand up just enough of the Vulkan
/// loader to enumerate physical devices, print one line per device, exit.
fn runProbe(w: *std.Io.Writer) !void {
    var devices: [16]decode_module_loader.ProbedDevice = undefined;
    const n_opt = decode_module_loader.enumerateDevicesForProbe(devices[0..]);
    const n = n_opt orelse {
        try w.writeAll("error: cannot enumerate Vulkan physical devices\n");
        try w.flush();
        std.process.exit(2);
    };
    if (n == 0) {
        try w.writeAll("no Vulkan physical devices reported\n");
        return;
    }
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const d = devices[i];
        const name_slice = d.name_buf[0..d.name_len];
        const major = (d.api_version >> 22) & 0x7F;
        const minor = (d.api_version >> 12) & 0x3FF;
        const patch = d.api_version & 0xFFF;
        try w.print("device[{d}]: {s} type={s} vendor=0x{X:0>4} api={d}.{d}.{d}", .{
            i,
            name_slice,
            decode_module_loader.deviceTypeName(d.device_type),
            d.vendor_id,
            major, minor, patch,
        });
        // Surface the WARP_SIZE=32 contract verdict per device so an
        // incompatible machine is diagnosable from --probe alone.
        if (d.max_subgroup != 0) {
            const ok = d.min_subgroup <= 32 and d.max_subgroup >= 32;
            try w.print(" subgroup=[{d},{d}]{s}\n", .{
                d.min_subgroup,
                d.max_subgroup,
                if (ok) "" else "  UNSUPPORTED (requires subgroupSize 32)",
            });
        } else {
            try w.writeAll(" subgroup=[?]\n");
        }
    }
}

/// CUDA reference: src/cli.zig:38-end. Top-level CLI dispatcher.
pub fn run(process_init: std.process.Init) !void {
    const allocator = process_init.gpa;
    const io = process_init.io;

    var args_it = try process_init.minimal.args.iterateAllocator(allocator);
    defer args_it.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(allocator);
    while (args_it.next()) |arg| try args_list.append(allocator, arg);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    if (args_list.items.len < 2) {
        try util.printUsage(w);
        return;
    }
    const args = util.parseArgs(args_list.items, w);

    switch (args.mode) {
        .version => try util.printVersion(w),
        .help => try util.printUsage(w),
        .probe => try runProbe(w),
        // The -i (info) mode is host-only; CUDA's CLI never touches the
        // GPU for it. Mirror that here so SLZ files can be inspected on
        // machines without a working Vulkan loader.
        .info => try info_cmd.run(allocator, io, w, args),
        .compress => {
            try ensureGpuInitAndPrintDevice(w, args);
            try compress_cmd.run(allocator, io, w, args);
        },
        .decompress => {
            try ensureGpuInitAndPrintDevice(w, args);
            try decompress_cmd.run(allocator, io, w, args);
        },
        .bench => {
            try ensureGpuInitAndPrintDevice(w, args);
            try bench_compress_cmd.run(allocator, io, w, args);
        },
        .bench_decompress => {
            try ensureGpuInitAndPrintDevice(w, args);
            try bench_decompress_cmd.run(allocator, io, w, args);
        },
        .bench_all => {
            try ensureGpuInitAndPrintDevice(w, args);
            try bench_all_cmd.run(allocator, io, w, args);
        },
    }

    if (args.report_mem) {
        const mem = util.getMemInfo();
        try w.print("MEMORY: {d:.0} MB peak RSS, {d:.0} MB peak commit\n", .{ mem.peak_rss_mb, mem.commit_mb });
    }
}
