//! StreamLZ command-line interface. GPU-only.
//!
//! Supported modes:
//!
//!   `streamlz -c <file>`        compress to <file>.slz
//!   `streamlz -d <file>`        decompress to <file> (.slz stripped)
//!   `streamlz -b <file>`        compress + decompress with verification
//!   `streamlz -db <file>`       decompress benchmark only
//!   `streamlz -ba <file>`       sweep all levels, ratio + throughput table
//!   `streamlz -i  <file>`       dump frame header + block list
//!
//! Options:
//!   `-l <1..5>`   level                    `-r <N>`     bench runs
//!   `-o <path>`   output path              `-gpu`       accepted, no-op
//!   `--sc <f>`    sc_group override
//!   `-mem`        print peak process memory at exit
//!
//! The `-gpu` flag is accepted as a no-op for backwards compatibility with
//! older scripts (the GPU is the only backend now). Levels outside 1..5
//! return `error.BadLevel` from the encoder.

const std = @import("std");
const builtin = @import("builtin");

const frame = @import("format/frame_format.zig");
const encoder = @import("encode/streamlz_encoder.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const gpu_driver = @import("decode/driver.zig");
const mmap_helpers = @import("mmap.zig");

const version_string = "3.0.0";

// ────────────────────────────────────────────────────────────────────────
//  Argument parsing
// ────────────────────────────────────────────────────────────────────────

const Mode = enum { compress, decompress, bench, bench_decompress, bench_all, info, version, help };

const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    runs: ?u32 = null,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    report_mem: bool = false,
    sc_group: ?f32 = null,
};

fn parseArgs(raw: []const []const u8, w: *std.Io.Writer) Args {
    var result: Args = .{};
    var i: usize = 1;
    while (i < raw.len) : (i += 1) {
        const arg = raw[i];
        if (eql(arg, "-V") or eql(arg, "--version")) return .{ .mode = .version };
        if (eql(arg, "-h") or eql(arg, "--help")) return .{ .mode = .help };
        if (eql(arg, "-c")) { result.mode = .compress; continue; }
        if (eql(arg, "-d")) { result.mode = .decompress; continue; }
        if (eql(arg, "-b")) { result.mode = .bench; continue; }
        if (eql(arg, "-db")) { result.mode = .bench_decompress; continue; }
        if (eql(arg, "-ba")) { result.mode = .bench_all; continue; }
        if (eql(arg, "-i")) { result.mode = .info; continue; }
        if (eql(arg, "-mem")) { result.report_mem = true; continue; }
        // `-gpu` is accepted but ignored — kept to avoid breaking caller
        // scripts that learned it before the GPU became the only backend.
        if (eql(arg, "-gpu")) continue;
        if (eql(arg, "-l")) { i += 1; result.level = parseInt(u8, expect(raw, i, "-l", w), w, "-l"); continue; }
        if (eql(arg, "-r")) { i += 1; result.runs = parseInt(u32, expect(raw, i, "-r", w), w, "-r"); continue; }
        if (eql(arg, "-o")) { i += 1; result.output = expect(raw, i, "-o", w); continue; }
        if (eql(arg, "--sc")) {
            i += 1;
            const v = expect(raw, i, "--sc", w);
            result.sc_group = std.fmt.parseFloat(f32, v) catch die(w, "error: --sc value must be a float\n");
            continue;
        }
        if (arg.len > 0 and arg[0] == '-') {
            w.print("error: unknown flag '{s}'\n\n", .{arg}) catch {};
            printUsage(w) catch {};
            w.flush() catch {};
            std.process.exit(2);
        }
        if (result.input == null) {
            result.input = arg;
        } else {
            w.print("error: unexpected argument '{s}'\n\n", .{arg}) catch {};
            printUsage(w) catch {};
            w.flush() catch {};
            std.process.exit(2);
        }
    }
    return result;
}

fn eql(a: []const u8, b: []const u8) bool { return std.mem.eql(u8, a, b); }

fn expect(raw: []const []const u8, i: usize, flag: []const u8, w: *std.Io.Writer) []const u8 {
    if (i >= raw.len) {
        w.print("error: {s} requires a value\n", .{flag}) catch {};
        w.flush() catch {};
        std.process.exit(2);
    }
    return raw[i];
}

fn parseInt(comptime T: type, s: []const u8, w: *std.Io.Writer, flag: []const u8) T {
    return std.fmt.parseInt(T, s, 10) catch {
        w.print("error: invalid {s} value '{s}'\n", .{ flag, s }) catch {};
        w.flush() catch {};
        std.process.exit(2);
    };
}

fn die(w: *std.Io.Writer, msg: []const u8) noreturn {
    w.writeAll(msg) catch {};
    w.flush() catch {};
    std.process.exit(2);
}

// ────────────────────────────────────────────────────────────────────────
//  Entry point
// ────────────────────────────────────────────────────────────────────────

pub fn run(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(allocator);
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
        .bench => try runBenchCompress(allocator, io, w, args),
        .bench_decompress => try runBenchDecompress(allocator, io, w, args),
        .bench_all => try runBenchAll(allocator, io, w, args),
        .info => try runInfo(allocator, io, w, args),
    }

    if (args.report_mem) {
        const mem = getMemInfo();
        try w.print("MEMORY: {d:.0} MB peak commit\n", .{mem.commit_mb});
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Output / help
// ────────────────────────────────────────────────────────────────────────

fn printVersion(w: *std.Io.Writer) !void {
    try w.print("streamlz {s} (GPU-only, Zig {f}, {s}-{s})\n", .{
        version_string, builtin.zig_version,
        @tagName(builtin.target.cpu.arch), @tagName(builtin.target.os.tag),
    });
}

fn printUsage(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\Usage: streamlz [options] <input-file>
        \\
        \\Mode (default: -c):
        \\  -c              Compress
        \\  -d              Decompress
        \\  -b              Compress + decompress + round-trip verify
        \\  -db             Decompress benchmark on a .slz file
        \\  -ba             Sweep levels L1-L5: compress + decompress ratio/speed table
        \\  -i              Dump frame header + block list
        \\
        \\Options:
        \\  -l <1..5>       Compression level (default: 1)
        \\  -r <runs>       Benchmark runs (default: 3 for -b, 10 for -db)
        \\  -o <file>       Output path
        \\  --sc <float>    sc_group_size override (0.25 = 64 KB sub-chunks)
        \\  -gpu            Accepted, no-op (GPU is the only backend)
        \\  -mem            Print peak process memory at exit
        \\  -V, --version   Print version
        \\  -h, --help      Print help
        \\
    );
}

// ────────────────────────────────────────────────────────────────────────
//  Helpers
// ────────────────────────────────────────────────────────────────────────

fn deriveCompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    const result = try allocator.alloc(u8, input.len + 4);
    @memcpy(result[0..input.len], input);
    @memcpy(result[input.len..][0..4], ".slz");
    return result;
}

fn deriveDecompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    if (input.len > 4 and eql(input[input.len - 4 ..], ".slz")) {
        const result = try allocator.alloc(u8, input.len - 4);
        @memcpy(result, input[0 .. input.len - 4]);
        return result;
    }
    const result = try allocator.alloc(u8, input.len + 4);
    @memcpy(result[0..input.len], input);
    @memcpy(result[input.len..][0..4], ".dec");
    return result;
}

fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, w: *std.Io.Writer) []const u8 {
    const max_bytes: usize = 1 << 31;
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, @enumFromInt(max_bytes)) catch |err| {
        w.print("error: cannot read '{s}': {s}\n", .{ path, @errorName(err) }) catch {};
        w.flush() catch {};
        std.process.exit(1);
    };
}

fn requireInput(args: Args, w: *std.Io.Writer) []const u8 {
    return args.input orelse {
        w.writeAll("error: no input file specified\n\n") catch {};
        printUsage(w) catch {};
        w.flush() catch {};
        std.process.exit(2);
    };
}

fn checkLevel(level: u8, w: *std.Io.Writer) void {
    if (level < 1 or level > 5) {
        w.print("error: level must be 1..5 (got {d})\n", .{level}) catch {};
        w.flush() catch {};
        std.process.exit(2);
    }
}

/// Median for short slices; mean for long ones (avoids an O(n log n)
/// sort blowing the small fixed stack buffer).
fn medianOrMean(times: []const u64) u64 {
    var buf: [256]u64 = undefined;
    const n = times.len;
    if (n == 0) return 0;
    if (n > buf.len) {
        var sum: u128 = 0;
        for (times) |t| sum += t;
        return @intCast(sum / n);
    }
    @memcpy(buf[0..n], times);
    std.mem.sort(u64, buf[0..n], {}, std.sort.asc(u64));
    if (n % 2 == 1) return buf[n / 2];
    return (buf[n / 2 - 1] + buf[n / 2]) / 2;
}

fn fmtBytes(buf: []u8, value: usize) []const u8 {
    var raw: [32]u8 = undefined;
    const raw_slice = std.fmt.bufPrint(&raw, "{d}", .{value}) catch return "?";
    const len = raw_slice.len;
    if (len <= 3) {
        @memcpy(buf[0..len], raw_slice);
        return buf[0..len];
    }
    const commas = (len - 1) / 3;
    const total = len + commas;
    if (total > buf.len) return raw_slice;
    var out: usize = total;
    var src_i: usize = len;
    var group: usize = 0;
    while (src_i > 0) {
        src_i -= 1;
        out -= 1;
        buf[out] = raw_slice[src_i];
        group += 1;
        if (group == 3 and src_i > 0) {
            out -= 1;
            buf[out] = ',';
            group = 0;
        }
    }
    return buf[0..total];
}

fn fmtMbps(buf: []u8, value: f64) []const u8 {
    if (value >= 100.0) return std.fmt.bufPrint(buf, "{d:>.0}", .{value}) catch "?";
    return std.fmt.bufPrint(buf, "{d:>.1}", .{value}) catch "?";
}

// ────────────────────────────────────────────────────────────────────────
//  Compress
// ────────────────────────────────────────────────────────────────────────

fn runCompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    checkLevel(args.level, w);

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

    const derived = if (args.output == null) try deriveCompressOutput(allocator, in_path) else null;
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
    }, &gpu_encoder.g_default) catch |err| {
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

    if (gpu_encoder.last_kernel_ns > 0) {
        const kms: f64 = @as(f64, @floatFromInt(gpu_encoder.last_kernel_ns)) / 1e6;
        const kmbps: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0) * 1e9 /
            @as(f64, @floatFromInt(gpu_encoder.last_kernel_ns));
        try w.print("  GPU kernel: {d:.1}ms ({d:.0} MB/s)\n", .{ kms, kmbps });
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Decompress
// ────────────────────────────────────────────────────────────────────────

fn runDecompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);

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
    const derived = if (args.output == null) try deriveDecompressOutput(allocator, in_path) else null;
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

    const result = decoder.decompressFramedThreaded(allocator, io, src, out_map.slice(), &gpu_driver.g_default) catch |err| {
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

// ────────────────────────────────────────────────────────────────────────
//  Benchmark: compress + decompress + verify (-b)
// ────────────────────────────────────────────────────────────────────────

fn runBenchCompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const runs = args.runs orelse 3;
    checkLevel(args.level, w);
    if (runs == 0) die(w, "error: runs must be >= 1\n");

    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(decompressed);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("Input: {s} ({d} bytes, {d:.2} MB)\n", .{ in_path, src.len, mb });

    const comp_opts: encoder.Options = .{
        .level = args.level,
        .sc_group_size_override = args.sc_group,
    };

    var comp_size: usize = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_encoder.g_default);
    {
        const t0 = std.Io.Clock.awake.now(io);
        comp_size = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_encoder.g_default);
        const ns = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        try w.print("  Compress: {d:.0}ms ({d:.1} MB/s)\n", .{ ms, mb * 1000.0 / ms });
    }
    try w.print("Level {d}: {d} -> {d} bytes ({d:.1}%)\n\n", .{
        args.level, src.len, comp_size,
        @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(src.len)) * 100.0,
    });

    var dec_ctx = decoder.DecompressContext.initWithIo(allocator, io);
    defer dec_ctx.deinit();

    _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed); // warm-up

    const dec_times = try allocator.alloc(u64, runs);
    defer allocator.free(dec_times);
    const kern_times = try allocator.alloc(i64, runs);
    defer allocator.free(kern_times);
    @memset(kern_times, 0);

    var r: u32 = 0;
    while (r < runs) : (r += 1) {
        const t0 = std.Io.Clock.awake.now(io);
        _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
        dec_times[r] = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        kern_times[r] = gpu_driver.last_kernel_ns;
        const ms = @as(f64, @floatFromInt(dec_times[r])) / 1_000_000.0;
        const kms = @as(f64, @floatFromInt(kern_times[r])) / 1_000_000.0;
        if (kern_times[r] > 0) {
            try w.print("  Decompress run {d}: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n", .{
                r + 1, ms, mb * 1000.0 / ms, kms, mb * 1000.0 / kms,
            });
        } else {
            try w.print("  Decompress run {d}: {d:.0}ms ({d:.1} MB/s)\n", .{ r + 1, ms, mb * 1000.0 / ms });
        }
    }

    const dec_med_ns = medianOrMean(dec_times);
    const dec_med_ms = @as(f64, @floatFromInt(dec_med_ns)) / 1_000_000.0;
    var nonzero_kerns: [256]i64 = undefined;
    var nz_count: usize = 0;
    for (kern_times) |k| if (k > 0 and nz_count < nonzero_kerns.len) {
        nonzero_kerns[nz_count] = k;
        nz_count += 1;
    };
    if (nz_count > 0) {
        std.mem.sort(i64, nonzero_kerns[0..nz_count], {}, std.sort.asc(i64));
        const kms = @as(f64, @floatFromInt(nonzero_kerns[nz_count / 2])) / 1_000_000.0;
        try w.print("  Decompress median: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n\n", .{
            dec_med_ms, mb * 1000.0 / dec_med_ms, kms, mb * 1000.0 / kms,
        });
    } else {
        try w.print("  Decompress median: {d:.0}ms ({d:.1} MB/s)\n\n", .{ dec_med_ms, mb * 1000.0 / dec_med_ms });
    }

    if (std.mem.eql(u8, src, decompressed[0..src.len])) {
        try w.writeAll("Round-trip: PASS\n");
    } else {
        var first_fail: usize = 0;
        var fail_count: usize = 0;
        for (0..src.len) |bi| if (src[bi] != decompressed[bi]) {
            if (fail_count == 0) first_fail = bi;
            fail_count += 1;
        };
        try w.print("Round-trip: FAIL  first_diff={d} total_diffs={d}\n", .{ first_fail, fail_count });
        try w.flush();
        std.process.exit(1);
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Benchmark: decompress only (-db)
// ────────────────────────────────────────────────────────────────────────

fn runBenchDecompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const runs = args.runs orelse 10;

    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const hdr = frame.parseHeader(src) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    const content_size: usize = if (hdr.content_size) |cs| @intCast(cs) else {
        try w.writeAll("error: frame has no content size; bench needs a sized frame\n");
        try w.flush();
        std.process.exit(1);
    };

    // Pin the output buffer so the D2H runs at full PCIe bandwidth. The
    // pinned allocation falls back to a normal heap allocation if CUDA
    // can't satisfy it (offline-development convenience).
    const dst_size = content_size + decoder.safe_space;
    const DstBuf = struct { buf: []u8, pinned: bool };
    const dh: DstBuf = blk: {
        if (gpu_driver.allocHost(dst_size)) |p| break :blk .{ .buf = p, .pinned = true };
        break :blk .{ .buf = allocator.alloc(u8, dst_size) catch |err| {
            try w.print("error: cannot allocate {d} bytes: {s}\n", .{ dst_size, @errorName(err) });
            try w.flush();
            std.process.exit(1);
        }, .pinned = false };
    };
    const dst = dh.buf;
    defer if (dh.pinned) gpu_driver.freeHost(dst) else allocator.free(dst);

    var dec_ctx = decoder.DecompressContext.initWithIo(allocator, io);
    defer dec_ctx.deinit();

    _ = dec_ctx.decompress(src, dst) catch |err| {
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var best_ns: u64 = std.math.maxInt(u64);
    var total_ns: u64 = 0;
    var best_kern_ns: i64 = std.math.maxInt(i64);
    var total_kern_ns: i64 = 0;
    var best_lz_ns: i64 = std.math.maxInt(i64);
    var total_lz_ns: i64 = 0;
    var best_huff_ns: i64 = std.math.maxInt(i64);
    var total_huff_ns: i64 = 0;

    var run_i: u32 = 0;
    while (run_i < runs) : (run_i += 1) {
        const t0 = std.Io.Clock.awake.now(io);
        _ = try dec_ctx.decompress(src, dst);
        const elapsed = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        if (elapsed < best_ns) best_ns = elapsed;
        total_ns += elapsed;
        if (gpu_driver.last_kernel_ns > 0) {
            if (gpu_driver.last_kernel_ns < best_kern_ns) best_kern_ns = gpu_driver.last_kernel_ns;
            total_kern_ns += gpu_driver.last_kernel_ns;
        }
        if (gpu_driver.last_lz_kernel_ns > 0) {
            if (gpu_driver.last_lz_kernel_ns < best_lz_ns) best_lz_ns = gpu_driver.last_lz_kernel_ns;
            total_lz_ns += gpu_driver.last_lz_kernel_ns;
        }
        if (gpu_driver.last_huff_kernel_ns > 0) {
            if (gpu_driver.last_huff_kernel_ns < best_huff_ns) best_huff_ns = gpu_driver.last_huff_kernel_ns;
            total_huff_ns += gpu_driver.last_huff_kernel_ns;
        }
    }

    const mean_ns: u64 = total_ns / runs;
    const mb: f64 = @as(f64, @floatFromInt(content_size)) / (1024.0 * 1024.0);
    try w.print("bench: {s}\n", .{in_path});
    try w.print("  src bytes:       {d}\n", .{src.len});
    try w.print("  decompressed:    {d} ({d:.2} MB)\n", .{ content_size, mb });
    try w.print("  runs:            {d} (plus 1 warm-up)\n", .{runs});
    try w.print("  best:            {d:.3} ms  ({d:.0} MB/s)\n", .{
        @as(f64, @floatFromInt(best_ns)) / 1_000_000.0, mb * 1e9 / @as(f64, @floatFromInt(best_ns)),
    });
    try w.print("  mean:            {d:.3} ms  ({d:.0} MB/s)\n", .{
        @as(f64, @floatFromInt(mean_ns)) / 1_000_000.0, mb * 1e9 / @as(f64, @floatFromInt(mean_ns)),
    });
    if (best_kern_ns < std.math.maxInt(i64)) {
        const mean_kern_ns = @divTrunc(total_kern_ns, @as(i64, @intCast(runs)));
        const best_kern_ms = @as(f64, @floatFromInt(best_kern_ns)) / 1_000_000.0;
        const mean_kern_ms = @as(f64, @floatFromInt(mean_kern_ns)) / 1_000_000.0;
        try w.print("  gpu kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{ best_kern_ms, mb * 1000.0 / best_kern_ms });
        try w.print("  gpu kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{ mean_kern_ms, mb * 1000.0 / mean_kern_ms });
        if (best_lz_ns < std.math.maxInt(i64)) {
            const mean_lz_ns = @divTrunc(total_lz_ns, @as(i64, @intCast(runs)));
            const best_lz_ms = @as(f64, @floatFromInt(best_lz_ns)) / 1_000_000.0;
            const mean_lz_ms = @as(f64, @floatFromInt(mean_lz_ns)) / 1_000_000.0;
            try w.print("  lz kernel best:   {d:.3} ms  ({d:.0} MB/s)\n", .{ best_lz_ms, mb * 1000.0 / best_lz_ms });
            try w.print("  lz kernel mean:   {d:.3} ms  ({d:.0} MB/s)\n", .{ mean_lz_ms, mb * 1000.0 / mean_lz_ms });
        }
        if (best_huff_ns < std.math.maxInt(i64)) {
            const mean_huff_ns = @divTrunc(total_huff_ns, @as(i64, @intCast(runs)));
            const best_huff_ms = @as(f64, @floatFromInt(best_huff_ns)) / 1_000_000.0;
            const mean_huff_ms = @as(f64, @floatFromInt(mean_huff_ns)) / 1_000_000.0;
            try w.print("  huff kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{ best_huff_ms, mb * 1000.0 / best_huff_ms });
            try w.print("  huff kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{ mean_huff_ms, mb * 1000.0 / mean_huff_ms });
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Benchmark: sweep L1-L5 (-ba)
// ────────────────────────────────────────────────────────────────────────

fn runBenchAll(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const runs = args.runs orelse 3;
    if (runs == 0) die(w, "error: runs must be >= 1\n");

    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("streamlz bench-all: {s} ({d} bytes, {d} decompress runs)\n", .{
        in_path, src.len, runs,
    });

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(decompressed);

    const Result = struct {
        level: u8,
        comp_size: usize,
        ratio: f64,
        comp_mbps: f64,
        dec_mbps: f64,
        pass: bool,
    };
    var results: [5]Result = undefined;

    var level: u8 = 1;
    while (level <= 5) : (level += 1) {
        const idx = level - 1;
        const comp_opts: encoder.Options = .{
            .level = level,
            .sc_group_size_override = args.sc_group,
        };
        const t_comp = std.Io.Clock.awake.now(io);
        const comp_size = encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_encoder.g_default) catch |err| {
            try w.print("  L{d}: compress failed: {s}\n", .{ level, @errorName(err) });
            results[idx] = .{ .level = level, .comp_size = 0, .ratio = 0, .comp_mbps = 0, .dec_mbps = 0, .pass = false };
            continue;
        };
        const best_comp_ns = @as(u64, @intCast(t_comp.untilNow(io, .awake).toNanoseconds()));

        var dec_ctx = decoder.DecompressContext.initWithIo(allocator, io);
        defer dec_ctx.deinit();

        _ = dec_ctx.decompress(compressed[0..comp_size], decompressed) catch |err| {
            try w.print("  L{d}: decompress failed: {s}\n", .{ level, @errorName(err) });
            results[idx] = .{ .level = level, .comp_size = comp_size, .ratio = 0, .comp_mbps = 0, .dec_mbps = 0, .pass = false };
            continue;
        };

        var best_dec_ns: u64 = std.math.maxInt(u64);
        var r: u32 = 0;
        while (r < runs) : (r += 1) {
            const t0 = std.Io.Clock.awake.now(io);
            _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
            const elapsed = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
            if (elapsed < best_dec_ns) best_dec_ns = elapsed;
        }

        const pass = std.mem.eql(u8, src, decompressed[0..src.len]);
        const ratio = @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0;
        const comp_ms = @as(f64, @floatFromInt(best_comp_ns)) / 1_000_000.0;
        const dec_ms = @as(f64, @floatFromInt(best_dec_ns)) / 1_000_000.0;
        results[idx] = .{
            .level = level,
            .comp_size = comp_size,
            .ratio = ratio,
            .comp_mbps = mb * 1000.0 / comp_ms,
            .dec_mbps = mb * 1000.0 / dec_ms,
            .pass = pass,
        };
        try w.print("  L{d} done ({d:.1}%)\n", .{ level, ratio });
        try w.flush();
    }

    try w.writeAll("\nLevel | Compressed         | Ratio  | Compress   | Decompress\n");
    try w.writeAll("------+--------------------+--------+------------+-----------\n");
    for (&results) |res| {
        var bytes_buf: [32]u8 = undefined;
        var cmbps_buf: [16]u8 = undefined;
        var dmbps_buf: [16]u8 = undefined;
        const pass_str: []const u8 = if (res.pass) "" else " FAIL";
        try w.print("L{d:<2}   | {s:>14} bytes | {d:>5.1}% | {s:>7} MB/s | {s:>7} MB/s{s}\n", .{
            res.level,
            fmtBytes(&bytes_buf, res.comp_size),
            res.ratio,
            fmtMbps(&cmbps_buf, res.comp_mbps),
            fmtMbps(&dmbps_buf, res.dec_mbps),
            pass_str,
        });
    }
}

// ────────────────────────────────────────────────────────────────────────
//  Info (-i)
// ────────────────────────────────────────────────────────────────────────

fn runInfo(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const path = requireInput(args, w);
    const data = readFile(allocator, io, path, w);
    defer allocator.free(data);

    const hdr = frame.parseHeader(data) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    try w.print("file: {s}\n", .{path});
    try w.print("  size on disk:    {d} bytes\n", .{data.len});
    try w.print("  magic:           SLZ1\n", .{});
    try w.print("  version:         {d}\n", .{hdr.version});
    try w.print("  codec:           {s} ({d})\n", .{ hdr.codec.name(), @intFromEnum(hdr.codec) });
    try w.print("  level:           {d}  (internal)\n", .{hdr.level});
    try w.print("  block_size:      {d} ({d} KB)\n", .{ hdr.block_size, hdr.block_size / 1024 });
    try w.print("  header_size:     {d} bytes\n", .{hdr.header_size});
    try w.print("  flags:\n", .{});
    try w.print("    content_size_present:  {}\n", .{hdr.flags.content_size_present});
    try w.print("    content_checksum:      {}\n", .{hdr.flags.content_checksum});
    try w.print("    block_checksums:       {}\n", .{hdr.flags.block_checksums});
    try w.print("    dictionary_id_present: {}\n", .{hdr.flags.dictionary_id_present});
    if (hdr.content_size) |cs| try w.print("  content_size:    {d} bytes\n", .{cs});
    if (hdr.dictionary_id) |id| try w.print("  dictionary_id:   0x{x:0>8}\n", .{id});

    try w.print("  blocks:\n", .{});
    var pos: usize = hdr.header_size;
    var block_index: usize = 0;
    var total_decompressed: u64 = 0;
    while (pos + 4 <= data.len) {
        const block_hdr = frame.parseBlockHeader(data[pos..]) catch |err| {
            try w.print("    [#{d}] invalid block header at pos={d}: {s}\n", .{ block_index, pos, @errorName(err) });
            break;
        };
        if (block_hdr.isEndMark()) {
            try w.print("    end_mark at pos={d}\n", .{pos});
            pos += 4;
            break;
        }
        try w.print("    [#{d}] pos={d} comp={d} decomp={d}{s}\n", .{
            block_index, pos, block_hdr.compressed_size, block_hdr.decompressed_size,
            if (block_hdr.uncompressed) " UNCOMPRESSED" else "",
        });
        total_decompressed += block_hdr.decompressed_size;
        pos += 8 + block_hdr.compressed_size;
        block_index += 1;
    }
    try w.print("  total blocks:    {d}\n", .{block_index});
    try w.print("  total decomp:    {d} bytes\n", .{total_decompressed});
    try w.print("  trailing bytes:  {d}\n", .{data.len -| pos});
}

// ────────────────────────────────────────────────────────────────────────
//  Process memory
// ────────────────────────────────────────────────────────────────────────

const MemInfo = struct { peak_rss_mb: f64, commit_mb: f64 };

fn getMemInfo() MemInfo {
    const os = builtin.os.tag;
    if (os == .windows) {
        const PROCESS_MEMORY_COUNTERS = extern struct {
            cb: u32 = @sizeOf(@This()),
            PageFaultCount: u32 = 0,
            PeakWorkingSetSize: usize = 0,
            WorkingSetSize: usize = 0,
            QuotaPeakPagedPoolUsage: usize = 0,
            QuotaPagedPoolUsage: usize = 0,
            QuotaPeakNonPagedPoolUsage: usize = 0,
            QuotaNonPagedPoolUsage: usize = 0,
            PagefileUsage: usize = 0,
            PeakPagefileUsage: usize = 0,
        };
        const k32 = struct {
            extern "kernel32" fn K32GetProcessMemoryInfo(
                hProcess: std.os.windows.HANDLE,
                ppsmemCounters: *PROCESS_MEMORY_COUNTERS,
                cb: u32,
            ) callconv(.winapi) std.os.windows.BOOL;
        };
        var info: PROCESS_MEMORY_COUNTERS = .{};
        if (k32.K32GetProcessMemoryInfo(std.os.windows.GetCurrentProcess(), &info, @sizeOf(PROCESS_MEMORY_COUNTERS)) != .FALSE) {
            return .{
                .peak_rss_mb = @as(f64, @floatFromInt(info.PeakWorkingSetSize)) / (1024.0 * 1024.0),
                .commit_mb = @as(f64, @floatFromInt(info.PeakPagefileUsage)) / (1024.0 * 1024.0),
            };
        }
    } else if (os == .linux or os == .macos or os == .ios) {
        var usage: std.c.rusage = undefined;
        if (std.c.getrusage(0, &usage) == 0) {
            const peak_kb: u64 = @intCast(@max(@as(isize, 0), usage.maxrss));
            const divisor: f64 = if (os == .macos or os == .ios) (1024.0 * 1024.0) else 1024.0;
            return .{
                .peak_rss_mb = @as(f64, @floatFromInt(peak_kb)) / divisor,
                .commit_mb = 0,
            };
        }
    }
    return .{ .peak_rss_mb = 0, .commit_mb = 0 };
}
