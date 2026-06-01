//! StreamLZ Vulkan-backend command line — sibling of `src/cli.zig`.
//!
//! Scope: level=1 only, single-file compress / decompress. The flag set
//! mirrors the CUDA CLI's so existing scripts can swap `streamlz.exe`
//! for `streamlz_vk.exe`; flags that only make sense for the CUDA
//! pipeline (`--sc`, `-b` benchmark modes, etc.) are not implemented
//! in v1.
//!
//!   streamlz_vk -c <file> [-l N] -o <out.slz>   compress
//!   streamlz_vk -d <file> -o <out>              decompress
//!   streamlz_vk -h | --help                     usage
//!   streamlz_vk -V | --version                  version line
//!
//! Built as `zig build streamlz_vk`; installed to `zig-out/bin/streamlz_vk.exe`.

const std = @import("std");

const driver = @import("driver.zig");
const slz1_codec = @import("slz1_codec.zig");
const wire_format = @import("wire_format.zig");
const frame = @import("../src/format/frame_format.zig");

const VK_SAFE_SPACE: usize = 64;

const Mode = enum { compress, decompress, version, help };

const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
};

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn die(w: *std.Io.Writer, comptime fmt: []const u8, args: anytype) noreturn {
    w.print(fmt, args) catch {};
    w.flush() catch {};
    std.process.exit(2);
}

fn printUsage(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\Usage: streamlz_vk [options] <file>
        \\
        \\Modes:
        \\  -c            Compress (default)
        \\  -d            Decompress
        \\
        \\Options:
        \\  -l N          Compression level (only 1 supported in v1)
        \\  -o PATH       Output path
        \\  -h, --help    Show this help
        \\  -V, --version Show version
        \\
        \\
    );
}

fn printVersion(w: *std.Io.Writer) !void {
    try w.print("streamlz_vk 3.0.0+vk (Vulkan, L1-only, level=1)\n", .{});
}

fn parseArgs(raw: []const []const u8, w: *std.Io.Writer) Args {
    var result: Args = .{};
    var i: usize = 1;
    while (i < raw.len) : (i += 1) {
        const arg = raw[i];
        if (eql(arg, "-V") or eql(arg, "--version")) return .{ .mode = .version };
        if (eql(arg, "-h") or eql(arg, "--help")) return .{ .mode = .help };
        if (eql(arg, "-c")) {
            result.mode = .compress;
            continue;
        }
        if (eql(arg, "-d")) {
            result.mode = .decompress;
            continue;
        }
        if (eql(arg, "-l")) {
            i += 1;
            if (i >= raw.len) die(w, "error: -l requires a value\n", .{});
            result.level = std.fmt.parseInt(u8, raw[i], 10) catch
                die(w, "error: invalid -l value '{s}'\n", .{raw[i]});
            continue;
        }
        if (eql(arg, "-o")) {
            i += 1;
            if (i >= raw.len) die(w, "error: -o requires a value\n", .{});
            result.output = raw[i];
            continue;
        }
        if (arg.len > 0 and arg[0] == '-') {
            die(w, "error: unknown flag '{s}'\n", .{arg});
        }
        if (result.input == null) {
            result.input = arg;
        } else {
            die(w, "error: unexpected argument '{s}'\n", .{arg});
        }
    }
    return result;
}

fn readFileAll(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const size: usize = @intCast(stat.size);
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const n = try file.readPositionalAll(io, buf, 0);
    if (n != size) return error.ShortRead;
    return buf;
}

fn writeFileAll(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, bytes, 0);
}

fn runCompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    if (args.level != 1) {
        try w.print("error: only level=1 is supported in v1 (got -l {d})\n", .{args.level});
        return error.UnsupportedLevel;
    }
    const in_path = args.input orelse {
        try w.writeAll("error: -c requires an input file\n");
        return error.NoInput;
    };
    const out_path = args.output orelse {
        try w.writeAll("error: -o <path> is required for compress\n");
        return error.NoOutput;
    };

    const src = readFileAll(allocator, io, in_path) catch |err| {
        try w.print("error: cannot read '{s}': {s}\n", .{ in_path, @errorName(err) });
        return err;
    };
    defer allocator.free(src);
    if (src.len == 0) {
        try w.writeAll("error: input file is empty\n");
        return error.EmptyInput;
    }

    try driver.ensureInit();
    defer driver.deinit();

    const bound = slz1_codec.slz1Bound(src.len);
    const out_buf = try allocator.alloc(u8, bound);
    defer allocator.free(out_buf);

    const written = slz1_codec.encodeL1ToSlz1(&driver.g_default, io, allocator, src, out_buf) catch |err| {
        try w.print("error: compression failed: {s}\n", .{@errorName(err)});
        return err;
    };

    writeFileAll(io, out_path, out_buf[0..written]) catch |err| {
        try w.print("error: cannot write '{s}': {s}\n", .{ out_path, @errorName(err) });
        return err;
    };

    const ratio: f64 = @as(f64, @floatFromInt(written)) /
        @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0;
    try w.print("compressed {s} -> {d} bytes ({d:.1}%) L1 -> {s}\n", .{
        in_path, written, ratio, out_path,
    });
}

fn runDecompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = args.input orelse {
        try w.writeAll("error: -d requires an input file\n");
        return error.NoInput;
    };
    const out_path = args.output orelse {
        try w.writeAll("error: -o <path> is required for decompress\n");
        return error.NoOutput;
    };

    const src = readFileAll(allocator, io, in_path) catch |err| {
        try w.print("error: cannot read '{s}': {s}\n", .{ in_path, @errorName(err) });
        return err;
    };
    defer allocator.free(src);
    if (src.len == 0) {
        try w.writeAll("error: input file is empty\n");
        return error.EmptyInput;
    }

    // Peek at the frame header to size the output buffer. The Vulkan
    // decoder writes exactly `content_size` bytes, but we add a small
    // pad to mirror the CUDA decompress contract.
    const hdr = frame.parseHeader(src) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        return err;
    };
    const content_size: usize = if (hdr.content_size) |cs| @intCast(cs) else {
        try w.writeAll("error: frame missing content_size (streaming mode unsupported)\n");
        return error.NoContentSize;
    };

    try driver.ensureInit();
    defer driver.deinit();

    const out_buf = try allocator.alloc(u8, content_size + VK_SAFE_SPACE);
    defer allocator.free(out_buf);

    const written = slz1_codec.decodeSlz1ToBytes(&driver.g_default, io, allocator, src, out_buf) catch |err| {
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        return err;
    };

    writeFileAll(io, out_path, out_buf[0..written]) catch |err| {
        try w.print("error: cannot write '{s}': {s}\n", .{ out_path, @errorName(err) });
        return err;
    };

    try w.print("decompressed {s} -> {d} bytes ({s} -> {s})\n", .{
        in_path, written, in_path, out_path,
    });
}

pub fn main(process_init: std.process.Init) !void {
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
        try printUsage(w);
        return;
    }
    const args = parseArgs(args_list.items, w);

    switch (args.mode) {
        .version => try printVersion(w),
        .help => try printUsage(w),
        .compress => try runCompress(allocator, io, w, args),
        .decompress => try runDecompress(allocator, io, w, args),
    }
}
