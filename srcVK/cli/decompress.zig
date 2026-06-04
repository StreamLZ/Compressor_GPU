//! 1:1 port of src/cli/decompress.zig.
//!
//! `streamlz_vk -d <file>` handler. Reads a .slz frame, runs the GPU
//! decoder, writes the recovered bytes.

const std = @import("std");
const util = @import("util.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_dec_driver = @import("../decode/driver.zig");
const frame = @import("../format/frame_format.zig");
const mmap_helpers = @import("../mmap.zig");

/// CUDA reference: src/cli/decompress.zig:11-end. -d mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);

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

    const hdr = frame.parseHeader(src) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    const content_size: usize = if (hdr.content_size) |cs| blk: {
        if (cs > decoder.max_content_size) {
            try w.print("error: frame claims {d} bytes uncompressed, exceeds {d}\n", .{ cs, decoder.max_content_size });
            try w.flush();
            std.process.exit(1);
        }
        break :blk @intCast(cs);
    } else {
        try w.writeAll("error: frame has no content size; streaming mode unsupported\n");
        try w.flush();
        std.process.exit(1);
    };

    const out_size = content_size + decoder.safe_space;
    const derived = if (args.output == null) try util.deriveDecompressOutput(allocator, in_path) else null;
    defer if (derived) |d| allocator.free(d);
    const out_path = args.output orelse derived.?;

    const out_file = std.Io.Dir.cwd().createFile(io, out_path, .{ .read = true }) catch |err| {
        try w.print("error: cannot create '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer out_file.close(io);

    out_file.setLength(io, out_size) catch |err| {
        try w.print("error: cannot pre-size output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var out_map = mmap_helpers.mapFileReadWrite(out_file, out_size) orelse {
        try w.writeAll("error: cannot memory-map output file\n");
        try w.flush();
        std.process.exit(1);
    };

    const result = decoder.decompressFramedThreaded(allocator, io, src, out_map.slice(), &gpu_dec_driver.g_default) catch |err| {
        out_map.unmap();
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    out_map.unmap();

    out_file.setLength(io, result.written) catch |err| {
        try w.print("error: cannot truncate output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    try w.print("decompressed {d} -> {d} bytes  ({s} -> {s})\n", .{
        src.len, result.written, in_path, out_path,
    });
}
