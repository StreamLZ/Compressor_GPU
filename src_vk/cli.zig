//! StreamLZ Vulkan-backed command-line interface (foundation wave).
//!
//! VK PORT NOTE: ports src/cli.zig. The full mode dispatcher
//! (compress / decompress / bench_*) lands once the L1 codec ports
//! reach src_vk/decode/. For the foundation wave we expose `--version`
//! and a clear "not yet wired" message for every other mode so the
//! binary is invokable end-to-end without crashing.

const std = @import("std");

const version = @import("version.zig");

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

    // Foundation-wave surface: --version is real, every other mode
    // prints a placeholder. Mirrors the CUDA CLI's --version path so
    // tooling that scrapes the version string interprets both binaries
    // as interchangeable codecs.
    if (args_list.items.len >= 2) {
        const arg1 = args_list.items[1];
        if (std.mem.eql(u8, arg1, "--version") or std.mem.eql(u8, arg1, "-v")) {
            try w.print("streamlz_vk {s}\n", .{version.string});
            return;
        }
    }

    try w.writeAll(
        \\streamlz_vk: foundation-wave binary.
        \\
        \\This build links the src_vk/ tree but the encode/decode
        \\dispatch is not yet wired — a subsequent wave brings the L1
        \\codec online. Run with --version to confirm the binary.
        \\
        \\For the CUDA-backed CLI, run `streamlz` instead.
        \\
    );
}
