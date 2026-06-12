//! `streamlz -c <file>` — compress one file to one SLZ1 frame on disk.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const gpu_enc_driver = @import("../encode/driver.zig");
const mmap_helpers = @import("../mmap.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);
    util.checkLevel(args.level, w);

    const in_file = std.Io.Dir.cwd().openFile(io, in_path, .{}) catch |err| {
        try w.print("error: cannot open '{s}': {s}\n", .{ in_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer in_file.close(io);

    const in_size = in_file.length(io) catch |err| {
        try w.print("error: cannot stat '{s}': {s}\n", .{ in_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    if (in_size == 0) {
        try w.writeAll("error: input file is empty\n");
        try w.flush();
        std.process.exit(1);
    }

    var in_map = mmap_helpers.mapFileRead(in_file, in_size) orelse {
        try w.writeAll("error: cannot memory-map input file\n");
        try w.flush();
        std.process.exit(1);
    };
    defer in_map.unmap();
    const src = in_map.sliceConst();

    const derived = if (args.output == null) try util.deriveCompressOutput(allocator, in_path) else null;
    defer if (derived) |d| allocator.free(d);
    const out_path = args.output orelse derived.?;

    const bound = encoder.compressBound(src.len);
    const out_file = std.Io.Dir.cwd().createFile(io, out_path, .{ .read = true }) catch |err| {
        try w.print("error: cannot create '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer out_file.close(io);

    out_file.setLength(io, bound) catch |err| {
        try w.print("error: cannot pre-size output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var out_map = mmap_helpers.mapFileReadWrite(out_file, bound) orelse {
        try w.writeAll("error: cannot memory-map output file\n");
        try w.flush();
        std.process.exit(1);
    };

    const written = encoder.compressFramedWithIo(allocator, io, src, out_map.slice(), .{
        .level = args.level,
        .sc_group_size_override = args.sc_group,
        .content_checksum = args.checksum,
        .dictionary_id = util.resolveDictionary(args.dictionary, in_path, w),
    }, &gpu_enc_driver.g_default) catch |err| {
        out_map.unmap();
        try w.print("error: compression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    out_map.unmap();
    out_file.setLength(io, written) catch |err| {
        try w.print("error: cannot truncate output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    const ratio: f64 = @as(f64, @floatFromInt(written)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0;
    try w.print("compressed {d} -> {d} bytes  ({d:.1}%)  L{d}  ({s} -> {s})\n", .{
        src.len, written, ratio, args.level, in_path, out_path,
    });

    if (gpu_enc_driver.last_kernel_ns > 0) {
        const kms: f64 = @as(f64, @floatFromInt(gpu_enc_driver.last_kernel_ns)) / 1e6;
        const kmbps: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0) * 1e9 /
            @as(f64, @floatFromInt(gpu_enc_driver.last_kernel_ns));
        try w.print("  GPU kernel: {d:.1}ms ({d:.0} MB/s)\n", .{ kms, kmbps });
    }
}
