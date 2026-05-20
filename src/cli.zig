const std = @import("std");
const builtin = @import("builtin");
const frame = @import("format/frame_format.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const encoder = @import("encode/streamlz_encoder.zig");
const dict_mod = @import("dict/dictionary.zig");

const version_string = "2.0.0";

// ─── Argument parsing ────────────────────────────────────────────────

const trainer = @import("dict/trainer.zig");

const build_options = @import("build_options");
const enable_bench = build_options.enable_bench;
const zstd = if (enable_bench) @import("compare/zstd.zig") else struct {};
const lz4 = if (enable_bench) @import("compare/lz4.zig") else struct {};
const lzt = if (enable_bench) @import("compare/lzturbo.zig") else struct {};

const forward_lz = @import("encode/forward_lz.zig");

const mmap_helpers = @import("platform/mmap.zig");
const cache_detect = @import("platform/cache_detect.zig");

const Mode = enum {
    compress,
    decompress,
    bench,
    bench_decompress,
    bench_all,
    bench_compare,
    bench_compare_fast,
    forward_analyze,
    train,
    info,
    version,
    help,
};

const Args = struct {
    mode: Mode,
    level: u8,
    runs: ?u32, // null = use default for mode
    threads: u32,
    input: ?[]const u8,
    output: ?[]const u8,
    dict_name: ?[]const u8 = null, // -D name or path
    no_dict: bool = false, // --no-dict disables auto-detection
    report_mem: bool = false, // -mem: print peak commit at exit
    gpu: bool = false, // -gpu: encode for GPU decode (sc_group=0.25, raw mode)
    sc_group: ?f32 = null, // --sc <float>: override sc_group_size
    engine: Engine = .slz, // --zstd or --lz4 to use external compressor
};

const Engine = enum { slz, zstd_engine, lz4_engine };

fn parseArgs(raw: []const []const u8, w: *std.Io.Writer) Args {
    var result: Args = .{
        .mode = .compress,
        .level = 1,
        .runs = null,
        .threads = 0,
        .input = null,
        .output = null,
    };

    var i: usize = 1; // skip argv[0]
    while (i < raw.len) : (i += 1) {
        const arg = raw[i];

        if (eql(arg, "-V") or eql(arg, "--version")) {
            result.mode = .version;
            return result;
        }
        if (eql(arg, "-h") or eql(arg, "--help")) {
            result.mode = .help;
            return result;
        }
        if (eql(arg, "-c")) {
            result.mode = .compress;

            continue;
        }
        if (eql(arg, "-d")) {
            result.mode = .decompress;

            continue;
        }
        if (eql(arg, "-b")) {
            result.mode = .bench;

            continue;
        }
        if (eql(arg, "-db")) {
            result.mode = .bench_decompress;

            continue;
        }
        if (eql(arg, "-bc")) {
            result.mode = .bench_compare;
            continue;
        }
        if (eql(arg, "-bcf")) {
            result.mode = .bench_compare_fast;
            continue;
        }
        if (eql(arg, "-ba")) {
            result.mode = .bench_all;

            continue;
        }
        if (eql(arg, "-i")) {
            result.mode = .info;
            continue;
        }
        if (eql(arg, "--train")) {
            result.mode = .train;
            continue;
        }
        if (eql(arg, "--forward")) {
            result.mode = .forward_analyze;
            continue;
        }
        if (eql(arg, "-l")) {
            if (i + 1 >= raw.len) die(w, "error: -l requires a value\n");
            i += 1;
            result.level = parseInt(u8, raw[i], w, "-l");
            continue;
        }
        if (eql(arg, "-r")) {
            if (i + 1 >= raw.len) die(w, "error: -r requires a value\n");
            i += 1;
            result.runs = parseInt(u32, raw[i], w, "-r");
            continue;
        }
        if (eql(arg, "-t")) {
            if (i + 1 >= raw.len) die(w, "error: -t requires a value\n");
            i += 1;
            result.threads = parseInt(u32, raw[i], w, "-t");
            continue;
        }
        if (eql(arg, "-o")) {
            if (i + 1 >= raw.len) die(w, "error: -o requires a value\n");
            i += 1;
            result.output = raw[i];
            continue;
        }
        if (eql(arg, "-D")) {
            if (i + 1 >= raw.len) die(w, "error: -D requires a dictionary name or path\n");
            i += 1;
            result.dict_name = raw[i];
            continue;
        }
        if (eql(arg, "--no-dict")) {
            result.no_dict = true;
            continue;
        }
        if (eql(arg, "-mem")) {
            result.report_mem = true;
            continue;
        }
        if (eql(arg, "-gpu")) {
            result.gpu = true;
            continue;
        }
        if (eql(arg, "--sc")) {
            if (i + 1 >= raw.len) die(w, "error: --sc requires a value\n");
            i += 1;
            result.sc_group = std.fmt.parseFloat(f32, raw[i]) catch {
                die(w, "error: --sc value must be a float\n");
            };
            continue;
        }
        if (eql(arg, "--zstd")) {
            result.engine = .zstd_engine;
            continue;
        }
        if (eql(arg, "--lz4")) {
            result.engine = .lz4_engine;
            continue;
        }
        // Starts with '-' but not recognized
        if (arg.len > 0 and arg[0] == '-') {
            w.print("error: unknown flag '{s}'\n\n", .{arg}) catch {};
            printUsage(w) catch {};
            w.flush() catch {};
            std.process.exit(2);
        }
        // Positional: input file
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

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
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

// ─── Entry point ─────────────────────────────────────────────────────

pub fn run(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(allocator);
    defer args_it.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(allocator);
    while (args_it.next()) |arg| {
        try args_list.append(allocator, arg);
    }
    const raw_args = args_list.items;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    if (raw_args.len < 2) {
        try printUsage(w);
        return;
    }

    const args = parseArgs(raw_args, w);

    switch (args.mode) {
        .version => try printVersion(w),
        .help => try printUsage(w),
        .compress => try runCompress(allocator, io, w, args),
        .decompress => try runDecompress(allocator, io, w, args),
        .bench => try runBenchCompress(allocator, io, w, args),
        .bench_decompress => try runBenchDecompress(allocator, io, w, args),
        .bench_all => try runBenchAll(allocator, io, w, args),
        .bench_compare => if (enable_bench) try runBenchCompare(allocator, io, w, args, false) else {
            try w.writeAll("error: -bc requires building with -Dbench=true\n");
            try w.flush();
            std.process.exit(2);
        },
        .bench_compare_fast => if (enable_bench) try runBenchCompare(allocator, io, w, args, true) else {
            try w.writeAll("error: -bfast requires building with -Dbench=true\n");
            try w.flush();
            std.process.exit(2);
        },
        .forward_analyze => try runForwardAnalyze(allocator, io, w, args),
        .info => try runInfo(allocator, io, w, args),
        .train => try runTrain(allocator, io, w, args),
    }

    if (args.report_mem) {
        const mem = getMemInfo();
        try w.print("MEMORY: {d:.0} MB peak commit\n", .{mem.commit_mb});
    }
}

// ─── Help / version ──────────────────────────────────────────────────

fn printVersion(w: *std.Io.Writer) !void {
    try w.print("streamlz {s} (Zig {f}, {s}-{s})\n", .{
        version_string,
        builtin.zig_version,
        @tagName(builtin.target.cpu.arch),
        @tagName(builtin.target.os.tag),
    });
}

fn printUsage(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\Usage: streamlz [options] <input-file>
        \\
        \\Mode flags (default: -c):
        \\  -c              Compress
        \\  -d              Decompress
        \\  -b              Benchmark (compress + decompress, verify round-trip)
        \\  -db             Decompress benchmark (input is pre-compressed .slz file)
        \\  -ba             Bench all levels L1-L11 (compress only, shows ratio table)
        \\  -bc             Comparison benchmark vs zstd + LZ4 (8 threads)
        \\  --train         Train a dictionary from input files (-o dict.bin file1 file2 ...)
        \\  -i              Info (dump SLZ1 frame header + block list)
        \\
        \\Options:
        \\  -l <level>      Compression level 1-11 (default: 1)
        \\  -r <runs>       Benchmark runs (default: 3 for -b, 10 for -db)
        \\  -t <threads>    Threads (0=auto, default: 0)
        \\  -o <file>       Output file
        \\  -D <name|path>  Dictionary (built-in: json, html, text, xml, css, js)
        \\  --no-dict       Disable auto-detected dictionary
        \\  -V, --version   Print version
        \\  -h, --help      Print help
        \\
    );
}

// ─── Helpers ─────────────────────────────────────────────────────────

/// Derive the output path for compress: input + ".slz"
fn deriveCompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    const result = try allocator.alloc(u8, input.len + 4);
    @memcpy(result[0..input.len], input);
    @memcpy(result[input.len..][0..4], ".slz");
    return result;
}

/// Derive the output path for decompress: strip .slz or append .dec
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

/// Median of a slice of nanosecond durations (falls back to mean for n > 256).
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

/// Format a byte count with thousands separators (e.g. 1,234,567).
fn fmtBytes(buf: []u8, value: usize) []const u8 {
    // First, render the raw number.
    var raw: [32]u8 = undefined;
    const raw_slice = std.fmt.bufPrint(&raw, "{d}", .{value}) catch return "?";
    const len = raw_slice.len;
    if (len <= 3) {
        @memcpy(buf[0..len], raw_slice);
        return buf[0..len];
    }
    // Insert commas.
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

// ─── Compress ────────────────────────────────────────────────────────

fn runCompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const level = args.level;

    if (args.engine == .slz and (level < 1 or level > 11)) {
        try w.print("error: level must be 1..11 (got {d})\n", .{level});
        try w.flush();
        std.process.exit(2);
    }

    if (args.engine != .slz) {
        if (!enable_bench) {
            try w.writeAll("error: zstd/lz4 engines require building with -Dbench=true\n");
            try w.flush();
            std.process.exit(2);
        }
        const src = readFile(allocator, io, in_path, w);
        defer allocator.free(src);
        const threads: usize = if (args.threads == 0) @max(1, std.Thread.getCpuCount() catch 1) else args.threads;

        switch (args.engine) {
            .zstd_engine => {
                var result = zstd.compressBlocksMt(allocator, io, src, threads, @intCast(level)) catch |err| {
                    try w.print("error: zstd compression failed: {s}\n", .{@errorName(err)});
                    try w.flush();
                    std.process.exit(1);
                };
                defer result.deinit();
                try w.print("compressed {d} -> {d} bytes  ({d:.1}%)  zstd {d} MT  ({s})\n", .{
                    src.len,
                    result.total_compressed,
                    @as(f64, @floatFromInt(result.total_compressed)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0,
                    level,
                    in_path,
                });
            },
            .lz4_engine => {
                const hc_level: ?c_int = if (level > 1) @intCast(level) else null;
                var result = lz4.compressMt(allocator, io, src, threads, hc_level) catch |err| {
                    try w.print("error: lz4 compression failed: {s}\n", .{@errorName(err)});
                    try w.flush();
                    std.process.exit(1);
                };
                defer result.deinit();
                try w.print("compressed {d} -> {d} bytes  ({d:.1}%)  LZ4{s} MT  ({s})\n", .{
                    src.len,
                    result.total_compressed,
                    @as(f64, @floatFromInt(result.total_compressed)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0,
                    if (hc_level != null) " HC" else "",
                    in_path,
                });
            },
            .slz => unreachable,
        }
        return;
    }

    // ── SLZ compress: mmap input, mmap output ──

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

    // Derive output path
    const derived = if (args.output == null) try deriveCompressOutput(allocator, in_path) else null;
    defer if (derived) |d| allocator.free(d);
    const out_path = args.output orelse derived.?;

    // Dictionary resolution
    var dict_data: ?[]const u8 = null;
    var dict_id: ?u32 = null;
    if (args.dict_name) |name| {
        if (dict_mod.findByName(name)) |d| {
            dict_data = d.data;
            dict_id = d.id;
            try w.print("  dictionary: {s} (built-in, {d} bytes)\n", .{ d.name, d.data.len });
        } else {
            try w.print("error: unknown dictionary '{s}'\n", .{name});
            try w.flush();
            std.process.exit(2);
        }
    } else if (!args.no_dict and !args.gpu) {
        // GPU mode never uses a dictionary: the GPU encode path produces
        // self-contained 64KB blocks with no cross-frame back-reference
        // window, and the L1/L2 GPU encoder mis-sizes its output when a
        // dict prefix is present (DestinationTooSmall). Auto-detection is
        // suppressed here so `-gpu` is dict-free regardless of file
        // extension.
        if (dict_mod.findByExtension(in_path)) |d| {
            dict_data = d.data;
            dict_id = d.id;
            try w.print("  dictionary: {s} (auto-detected, {d} bytes)\n", .{ d.name, d.data.len });
        }
    }

    const threads: usize = if (args.threads == 0) @max(1, std.Thread.getCpuCount() catch 1) else args.threads;
    const bound = encoder.compressBound(src.len);

    // Create output file, pre-size to compress bound, mmap for direct write.
    const out_file = std.Io.Dir.cwd().createFile(io, out_path, .{ .read = true }) catch |err| {
        try w.print("error: cannot create '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer out_file.close(io);

    out_file.setLength(io, bound) catch |err| {
        try w.print("error: cannot pre-size output file: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var out_map = mmap_helpers.mapFileReadWrite(out_file, bound) orelse {
        try w.writeAll("error: cannot memory-map output file\n");
        try w.flush();
        std.process.exit(1);
    };

    const dst = out_map.slice();

    const written = encoder.compressFramedWithIo(allocator, io, src, dst, .{
        .level = level,
        .num_threads = @intCast(threads),
        .dictionary = dict_data,
        .dictionary_id = dict_id,
        .gpu_mode = args.gpu,
        .sc_group_size_override = args.sc_group,
    }) catch |err| {
        out_map.unmap();
        try w.print("error: compression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    out_map.unmap();

    // Truncate to actual compressed size.
    out_file.setLength(io, written) catch |err| {
        try w.print("error: cannot truncate output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    const ratio: f64 = @as(f64, @floatFromInt(written)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0;
    try w.print("compressed {d} -> {d} bytes  ({d:.1}%)  L{d}  ({s} -> {s})\n", .{
        src.len, written, ratio, level, in_path, out_path,
    });

    if (comptime blk: {
        const bo = @import("build_options");
        break :blk if (@hasDecl(bo, "gpu")) bo.gpu else false;
    }) {
        const gpu_enc = @import("encode/fast/gpu_encoder.zig");
        if (gpu_enc.last_kernel_ns > 0) {
            const kms: f64 = @as(f64, @floatFromInt(gpu_enc.last_kernel_ns)) / 1e6;
            const kmbps: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0) * 1e9 / @as(f64, @floatFromInt(gpu_enc.last_kernel_ns));
            try w.print("  GPU kernel: {d:.1}ms ({d:.0} MB/s)\n", .{ kms, kmbps });
        }
        if (std.c.getenv("SLZ_STREAM_DBG") != null) {
            const flz = @import("encode/fast/fast_lz_encoder.zig");
            const total = flz.stream_dbg_lit + flz.stream_dbg_tok + flz.stream_dbg_off16 + flz.stream_dbg_off32 + flz.stream_dbg_len;
            if (total > 0) {
                try w.print("  raw stream bytes:\n", .{});
                try w.print("    literal: {d:>12} ({d:>5.2}%)\n", .{ flz.stream_dbg_lit, 100.0 * @as(f64, @floatFromInt(flz.stream_dbg_lit)) / @as(f64, @floatFromInt(total)) });
                try w.print("    token:   {d:>12} ({d:>5.2}%)\n", .{ flz.stream_dbg_tok, 100.0 * @as(f64, @floatFromInt(flz.stream_dbg_tok)) / @as(f64, @floatFromInt(total)) });
                try w.print("    off16:   {d:>12} ({d:>5.2}%)\n", .{ flz.stream_dbg_off16, 100.0 * @as(f64, @floatFromInt(flz.stream_dbg_off16)) / @as(f64, @floatFromInt(total)) });
                try w.print("    off32:   {d:>12} ({d:>5.2}%)\n", .{ flz.stream_dbg_off32, 100.0 * @as(f64, @floatFromInt(flz.stream_dbg_off32)) / @as(f64, @floatFromInt(total)) });
                try w.print("    length:  {d:>12} ({d:>5.2}%)\n", .{ flz.stream_dbg_len, 100.0 * @as(f64, @floatFromInt(flz.stream_dbg_len)) / @as(f64, @floatFromInt(total)) });
                try w.print("    TOTAL:   {d:>12}\n", .{total});
            }
        }
    }
}

// ─── Decompress ──────────────────────────────────────────────────────

fn runDecompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);

    if (args.engine != .slz) {
        if (!enable_bench) {
            try w.writeAll("error: zstd/lz4 engines require building with -Dbench=true\n");
            try w.flush();
            std.process.exit(2);
        }
        const src = readFile(allocator, io, in_path, w);
        defer allocator.free(src);
        const threads: usize = if (args.threads == 0) @max(1, std.Thread.getCpuCount() catch 1) else args.threads;
        const level: c_int = @intCast(args.level);

        switch (args.engine) {
            .zstd_engine => {
                var result = zstd.compressBlocksMt(allocator, io, src, threads, level) catch |err| {
                    try w.print("error: zstd compress failed: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
                defer result.deinit();
                const dec_buf = try allocator.alloc(u8, src.len);
                defer allocator.free(dec_buf);
                zstd.decompressBlocksMt(io, src, dec_buf, &result, threads) catch |err| {
                    try w.print("error: zstd decompress failed: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
                try w.print("decompressed {d} bytes  zstd {d} MT\n", .{ src.len, level });
            },
            .lz4_engine => {
                const hc_level: ?c_int = if (level > 1) level else null;
                var result = lz4.compressMt(allocator, io, src, threads, hc_level) catch |err| {
                    try w.print("error: lz4 compress failed: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
                defer result.deinit();
                const dec_buf = try allocator.alloc(u8, src.len);
                defer allocator.free(dec_buf);
                lz4.decompressMt(io, src, dec_buf, &result, threads) catch |err| {
                    try w.print("error: lz4 decompress failed: {s}\n", .{@errorName(err)});
                    std.process.exit(1);
                };
                try w.print("decompressed {d} bytes  LZ4{s} MT\n", .{
                    src.len,
                    if (hc_level != null) " HC" else "",
                });
            },
            .slz => unreachable,
        }
        return;
    }

    // ── SLZ decompress: mmap input, mmap output ──

    // Open + mmap input file (read-only)
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

    // Parse frame header to get content size
    const hdr = frame.parseHeader(src) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    const content_size: usize = if (hdr.content_size) |cs| blk: {
        if (cs > decoder.max_content_size) {
            try w.print("error: frame claims {d} bytes uncompressed, exceeds maximum {d}\n", .{ cs, decoder.max_content_size });
            try w.flush();
            std.process.exit(1);
        }
        break :blk @intCast(cs);
    } else {
        try w.writeAll("error: frame has no content size; streaming mode not yet supported\n");
        try w.flush();
        std.process.exit(1);
    };

    const dict_overhead: usize = if (hdr.dictionary_id) |did|
        if (dict_mod.findById(did)) |d| d.data.len + decoder.safe_space else 0
    else
        0;
    const out_size: usize = content_size + decoder.safe_space + dict_overhead;

    // Derive output path
    const derived = if (args.output == null) try deriveDecompressOutput(allocator, in_path) else null;
    defer if (derived) |d| allocator.free(d);
    const out_path = args.output orelse derived.?;

    // Create output file, pre-size it, and mmap for direct decompress.
    const out_file = std.Io.Dir.cwd().createFile(io, out_path, .{ .read = true }) catch |err| {
        try w.print("error: cannot create '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer out_file.close(io);

    out_file.setLength(io, out_size) catch |err| {
        try w.print("error: cannot pre-size output file: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var out_map = mmap_helpers.mapFileReadWrite(out_file, out_size) orelse {
        try w.writeAll("error: cannot memory-map output file\n");
        try w.flush();
        std.process.exit(1);
    };

    const dst = out_map.slice();

    const dec_result = decoder.decompressFramedParallelThreaded(allocator, io, src, dst, args.threads) catch |err| {
        out_map.unmap();
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    const written = dec_result.written;

    // Shift content past the dictionary prefix (if any) to file offset 0.
    if (dec_result.offset > 0) {
        const content = dst[dec_result.offset..][0..written];
        std.mem.copyForwards(u8, dst[0..written], content);
    }

    out_map.unmap();

    // Truncate to exact decompressed size.
    out_file.setLength(io, written) catch |err| {
        try w.print("error: cannot truncate output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    try w.print("decompressed {d} -> {d} bytes  ({s} -> {s})\n", .{
        src.len, written, in_path, out_path,
    });

    // Forward-LZ diagnostic report (if any High codec chunks were decoded).
    const high_runs = @import("decode/high/high_lz_token_executor.zig");
    high_runs.reportFwdDiag();
}

// ─── Benchmark: compress + decompress (-b) ───────────────────────────

fn runBenchCompress(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const level = args.level;
    const runs = args.runs orelse 3;

    if (level < 1 or level > 11) {
        try w.print("error: level must be 1..11 (got {d})\n", .{level});
        try w.flush();
        std.process.exit(2);
    }
    if (runs == 0) {
        die(w, "error: runs must be >= 1\n");
    }

    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("Input: {s} ({d} bytes, {d:.2} MB)\n", .{ in_path, src.len, mb });

    // Dictionary resolution (same logic as runCompress).
    var dict_data_b: ?[]const u8 = null;
    var dict_id_b: ?u32 = null;
    if (args.dict_name) |name| {
        if (dict_mod.findByName(name)) |d| { dict_data_b = d.data; dict_id_b = d.id; }
    } else if (!args.no_dict) {
        if (dict_mod.findByExtension(in_path)) |d| { dict_data_b = d.data; dict_id_b = d.id; }
    }

    const dict_buf_extra: usize = if (dict_data_b) |d| d.len + decoder.safe_space else 0;
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space + dict_buf_extra);
    defer allocator.free(decompressed);

    // Warm-up compress.
    const comp_opts: encoder.Options = .{ .level = level, .dictionary = dict_data_b, .dictionary_id = dict_id_b, .num_threads = args.threads };
    var comp_size: usize = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts);

    // Compress (single timed run — -r only affects decompress).
    {
        const timer_start = std.Io.Clock.awake.now(io);
        comp_size = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts);
        const ns = @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds()));
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        const mbps = mb * 1000.0 / ms;
        try w.print("  Compress: {d:.0}ms ({d:.1} MB/s)\n", .{ ms, mbps });
    }
    try w.print("Level {d}: {d} -> {d} bytes ({d:.1}%)\n\n", .{
        level,
        src.len,
        comp_size,
        @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(src.len)) * 100.0,
    });

    // Persistent decompress context.
    var dec_ctx = decoder.DecompressContext.initThreadedWithIo(allocator, io, args.threads);
    defer dec_ctx.deinit();

    // Warm-up decompress.
    var dec_off: usize = 0;
    {
        const wr = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
        dec_off = wr.offset;
    }

    // Decompress benchmark.
    const dec_times = try allocator.alloc(u64, runs);
    defer allocator.free(dec_times);
    var r: u32 = 0;
    while (r < runs) : (r += 1) {
        const timer_start = std.Io.Clock.awake.now(io);
        _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
        dec_times[r] = @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds()));
        const run_ms = @as(f64, @floatFromInt(dec_times[r])) / 1_000_000.0;
        const run_mbps = mb * 1000.0 / run_ms;
        try w.print("  Decompress run {d}: {d:.0}ms ({d:.1} MB/s)\n", .{ r + 1, run_ms, run_mbps });
    }
    const dec_median_ns = medianOrMean(dec_times);
    const dec_median_ms = @as(f64, @floatFromInt(dec_median_ns)) / 1_000_000.0;
    const dec_median_mbps = mb * 1000.0 / dec_median_ms;
    try w.print("  Decompress median: {d:.0}ms ({d:.1} MB/s)\n\n", .{ dec_median_ms, dec_median_mbps });

    // Round-trip check.
    if (std.mem.eql(u8, src, decompressed[dec_off..][0..src.len])) {
        try w.writeAll("Round-trip: PASS\n");
    } else {
        var first_fail: usize = 0;
        var fail_count: usize = 0;
        for (0..src.len) |bi| {
            if (src[bi] != decompressed[dec_off + bi]) {
                if (fail_count == 0) first_fail = bi;
                fail_count += 1;
            }
        }
        try w.print("Round-trip: FAIL  first_diff={d} total_diffs={d} chunk={d} rel={d}\n", .{
            first_fail, fail_count, first_fail / 262144, first_fail % 262144,
        });
        try w.flush();
        std.process.exit(1);
    }

}

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
        if (std.c.getrusage(0, &usage) == 0) { // 0 == RUSAGE_SELF
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

// ─── Benchmark: decompress only (-db) ────────────────────────────────

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

    const db_dict_overhead: usize = if (hdr.dictionary_id) |did|
        if (dict_mod.findById(did)) |d| d.data.len + decoder.safe_space else 0
    else
        0;
    const dst = allocator.alloc(u8, content_size + decoder.safe_space + db_dict_overhead) catch |err| {
        try w.print("error: cannot allocate {d} bytes: {s}\n", .{ content_size + decoder.safe_space + db_dict_overhead, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer allocator.free(dst);

    var dec_ctx = decoder.DecompressContext.initThreadedWithIo(allocator, io, args.threads);
    defer dec_ctx.deinit();

    // Warm-up.
    _ = dec_ctx.decompress(src, dst) catch |err| {
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    const use_gpu_bench = comptime blk: {
        const bo = @import("build_options");
        break :blk if (@hasDecl(bo, "gpu")) bo.gpu else false;
    };

    var best_ns: u64 = std.math.maxInt(u64);
    var total_ns: u64 = 0;
    var best_kern_ns: i64 = std.math.maxInt(i64);
    var total_kern_ns: i64 = 0;
    var best_tans_ns: i64 = std.math.maxInt(i64);
    var total_tans_ns: i64 = 0;
    var best_lz_ns: i64 = std.math.maxInt(i64);
    var total_lz_ns: i64 = 0;
    var best_huff_ns: i64 = std.math.maxInt(i64);
    var total_huff_ns: i64 = 0;
    var run_i: u32 = 0;
    while (run_i < runs) : (run_i += 1) {
        const timer_start = std.Io.Clock.awake.now(io);
        _ = try dec_ctx.decompress(src, dst);
        const elapsed = @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds()));
        if (elapsed < best_ns) best_ns = elapsed;
        total_ns += elapsed;
        if (use_gpu_bench) {
            const gpu = @import("decode/fast/gpu_driver.zig");
            const vk = @import("decode/fast/vk_driver.zig");
            const kns = if (gpu.last_kernel_ns > 0) gpu.last_kernel_ns else vk.last_kernel_ns;
            if (kns > 0) {
                if (kns < best_kern_ns) best_kern_ns = kns;
                total_kern_ns += kns;
            }
            const tns = gpu.last_tans_kernel_ns;
            if (tns > 0) {
                if (tns < best_tans_ns) best_tans_ns = tns;
                total_tans_ns += tns;
            }
            const lns = gpu.last_lz_kernel_ns;
            if (lns > 0) {
                if (lns < best_lz_ns) best_lz_ns = lns;
                total_lz_ns += lns;
            }
            const hns = gpu.last_huff_kernel_ns;
            if (hns > 0) {
                if (hns < best_huff_ns) best_huff_ns = hns;
                total_huff_ns += hns;
            }
        }
    }

    const mean_ns: u64 = total_ns / runs;
    const mb: f64 = @as(f64, @floatFromInt(content_size)) / (1024.0 * 1024.0);
    const best_mbps: f64 = mb * 1e9 / @as(f64, @floatFromInt(best_ns));
    const mean_mbps: f64 = mb * 1e9 / @as(f64, @floatFromInt(mean_ns));

    try w.print("bench: {s}\n", .{in_path});
    try w.print("  src bytes:       {d}\n", .{src.len});
    try w.print("  decompressed:    {d} ({d:.2} MB)\n", .{ content_size, mb });
    try w.print("  runs:            {d} (plus 1 warm-up)\n", .{runs});
    if (args.threads > 0) {
        try w.print("  threads:         {d}\n", .{args.threads});
    }
    try w.print("  best:            {d:.3} ms  ({d:.0} MB/s)\n", .{
        @as(f64, @floatFromInt(best_ns)) / 1_000_000.0,
        best_mbps,
    });
    try w.print("  mean:            {d:.3} ms  ({d:.0} MB/s)\n", .{
        @as(f64, @floatFromInt(mean_ns)) / 1_000_000.0,
        mean_mbps,
    });
    if (use_gpu_bench and best_kern_ns < std.math.maxInt(i64)) {
        const mean_kern_ns = @divTrunc(total_kern_ns, @as(i64, @intCast(runs)));
        const best_kern_ms: f64 = @as(f64, @floatFromInt(best_kern_ns)) / 1_000_000.0;
        const mean_kern_ms: f64 = @as(f64, @floatFromInt(mean_kern_ns)) / 1_000_000.0;
        try w.print("  gpu kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{
            best_kern_ms,
            mb * 1000.0 / best_kern_ms,
        });
        try w.print("  gpu kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{
            mean_kern_ms,
            mb * 1000.0 / mean_kern_ms,
        });
        if (best_tans_ns < std.math.maxInt(i64)) {
            const mean_tans_ns = @divTrunc(total_tans_ns, @as(i64, @intCast(runs)));
            const best_tans_ms: f64 = @as(f64, @floatFromInt(best_tans_ns)) / 1_000_000.0;
            const mean_tans_ms: f64 = @as(f64, @floatFromInt(mean_tans_ns)) / 1_000_000.0;
            try w.print("  tans kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{
                best_tans_ms,
                mb * 1000.0 / best_tans_ms,
            });
            try w.print("  tans kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{
                mean_tans_ms,
                mb * 1000.0 / mean_tans_ms,
            });
        }
        if (best_lz_ns < std.math.maxInt(i64)) {
            const mean_lz_ns = @divTrunc(total_lz_ns, @as(i64, @intCast(runs)));
            const best_lz_ms: f64 = @as(f64, @floatFromInt(best_lz_ns)) / 1_000_000.0;
            const mean_lz_ms: f64 = @as(f64, @floatFromInt(mean_lz_ns)) / 1_000_000.0;
            try w.print("  lz kernel best:   {d:.3} ms  ({d:.0} MB/s)\n", .{
                best_lz_ms,
                mb * 1000.0 / best_lz_ms,
            });
            try w.print("  lz kernel mean:   {d:.3} ms  ({d:.0} MB/s)\n", .{
                mean_lz_ms,
                mb * 1000.0 / mean_lz_ms,
            });
        }
        if (best_huff_ns < std.math.maxInt(i64)) {
            const mean_huff_ns = @divTrunc(total_huff_ns, @as(i64, @intCast(runs)));
            const best_huff_ms: f64 = @as(f64, @floatFromInt(best_huff_ns)) / 1_000_000.0;
            const mean_huff_ms: f64 = @as(f64, @floatFromInt(mean_huff_ns)) / 1_000_000.0;
            try w.print("  huff kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{
                best_huff_ms,
                mb * 1000.0 / best_huff_ms,
            });
            try w.print("  huff kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{
                mean_huff_ms,
                mb * 1000.0 / mean_huff_ms,
            });
        }
    }
}

// ─── Benchmark: all levels (-ba) ─────────────────────────────────────

fn runBenchAll(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const runs = args.runs orelse 3;

    if (runs == 0) {
        die(w, "error: runs must be >= 1\n");
    }

    const num_threads: usize = if (args.threads == 0) (std.Thread.getCpuCount() catch 1) else args.threads;

    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    cache_detect.printSystemInfo(w, io);
    try w.print("streamlz bench-all: {s} ({d} bytes, {d} threads, {d} decompress runs)\n", .{ in_path, src.len, num_threads, runs });

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(decompressed);

    // Collect results for all levels, then print table.
    const Result = struct {
        level: u8,
        comp_size: usize,
        ratio: f64,
        comp_mbps: f64,
        dec_mbps: f64,
        pass: bool,
    };
    var results: [11]Result = undefined;

    var level: u8 = 1;
    while (level <= 11) : (level += 1) {
        const idx = level - 1;

        // Compress once, timed.
        const comp_opts: encoder.Options = .{ .level = level, .num_threads = @intCast(num_threads) };
        const comp_start = std.Io.Clock.awake.now(io);
        const comp_size = encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts) catch |err| {
            try w.print("  L{d}: compress failed: {s}\n", .{ level, @errorName(err) });
            results[idx] = .{ .level = level, .comp_size = 0, .ratio = 0, .comp_mbps = 0, .dec_mbps = 0, .pass = false };
            continue;
        };
        const best_comp_ns = @as(u64, @intCast(comp_start.untilNow(io, .awake).toNanoseconds()));

        // Decompress context.
        var dec_ctx = decoder.DecompressContext.initThreadedWithIo(allocator, io, num_threads);
        defer dec_ctx.deinit();

        // Warm-up decompress.
        _ = dec_ctx.decompress(compressed[0..comp_size], decompressed) catch |err| {
            try w.print("  L{d}: decompress failed: {s}\n", .{ level, @errorName(err) });
            results[idx] = .{ .level = level, .comp_size = comp_size, .ratio = 0, .comp_mbps = 0, .dec_mbps = 0, .pass = false };
            continue;
        };

        // Decompress: best of N runs.
        var best_dec_ns: u64 = std.math.maxInt(u64);
        var r: u32 = 0;
        while (r < runs) : (r += 1) {
            const timer_start = std.Io.Clock.awake.now(io);
            _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
            const elapsed = @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds()));
            if (elapsed < best_dec_ns) best_dec_ns = elapsed;
        }

        // Round-trip verification.
        const pass = std.mem.eql(u8, src, decompressed[0..src.len]);

        const ratio = @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0;
        const comp_ms = @as(f64, @floatFromInt(best_comp_ns)) / 1_000_000.0;
        const dec_ms = @as(f64, @floatFromInt(best_dec_ns)) / 1_000_000.0;
        const comp_mbps = mb * 1000.0 / comp_ms;
        const dec_mbps = mb * 1000.0 / dec_ms;

        results[idx] = .{
            .level = level,
            .comp_size = comp_size,
            .ratio = ratio,
            .comp_mbps = comp_mbps,
            .dec_mbps = dec_mbps,
            .pass = pass,
        };

        // Progress indicator (levels can be slow).
        try w.print("  L{d} done ({d:.1}%)\n", .{ level, ratio });
        try w.flush();
    }

    // Print the table.
    try w.writeAll("\nLevel | Compressed         | Ratio  | Compress   | Decompress\n");
    try w.writeAll("------+--------------------+--------+------------+-----------\n");

    for (results) |res| {
        var bytes_buf: [32]u8 = undefined;
        const bytes_str = fmtBytes(&bytes_buf, res.comp_size);

        // Format MB/s with appropriate precision.
        var comp_mbps_buf: [16]u8 = undefined;
        const comp_mbps_str = fmtMbps(&comp_mbps_buf, res.comp_mbps);
        var dec_mbps_buf: [16]u8 = undefined;
        const dec_mbps_str = fmtMbps(&dec_mbps_buf, res.dec_mbps);

        const pass_str: []const u8 = if (res.pass) "" else " FAIL";

        try w.print("L{d:<2}   | {s:>14} bytes | {d:>5.1}% | {s:>7} MB/s | {s:>7} MB/s{s}\n", .{
            res.level,
            bytes_str,
            res.ratio,
            comp_mbps_str,
            dec_mbps_str,
            pass_str,
        });
    }
}

// ─── Benchmark: comparison vs zstd + LZ4 (-bc) ─────────────────────

fn flushMemory() void {
    if (builtin.os.tag == .windows) {
        const k32 = struct {
            extern "kernel32" fn SetProcessWorkingSetSize(
                hProcess: std.os.windows.HANDLE,
                dwMin: usize,
                dwMax: usize,
            ) callconv(.winapi) std.os.windows.BOOL;
            extern "kernel32" fn HeapCompact(
                hHeap: std.os.windows.HANDLE,
                dwFlags: u32,
            ) callconv(.winapi) usize;
            extern "kernel32" fn GetProcessHeap() callconv(.winapi) std.os.windows.HANDLE;
        };
        _ = k32.SetProcessWorkingSetSize(
            std.os.windows.GetCurrentProcess(),
            std.math.maxInt(usize),
            std.math.maxInt(usize),
        );
        _ = k32.HeapCompact(k32.GetProcessHeap(), 0);
    }
}

const BenchCompareRow = struct {
    name_buf: [16]u8 = undefined,
    name_len: usize = 0,
    comp_size: usize = 0,
    ratio: f64 = 0,
    comp_mbps: f64 = 0,
    dec_mbps: f64 = 0,

    fn name(self: *const @This()) []const u8 {
        return self.name_buf[0..self.name_len];
    }

    fn setName(self: *@This(), n: []const u8) void {
        @memcpy(self.name_buf[0..n.len], n);
        self.name_len = n.len;
    }
};

/// Decompression dispatch — wraps the different codec decompress APIs behind
/// a single `run()` method so the timed best-of-N loop can be shared.
const BenchDecomp = union(enum) {
    lz4_st: struct { dst: []u8, comp: []const u8, orig_size: usize },
    lz4_mt: struct { io: std.Io, src: []const u8, dst: []u8, mt: *const lz4.MtResult, threads: usize },
    lzt_st: struct { dst: []u8, comp: []const u8, orig_size: usize },
    lzt_mt: struct { io: std.Io, src: []const u8, dst: []u8, mt: *const lzt.MtResult, threads: usize },
    zstd_st: struct { dst: []u8, comp: []const u8 },
    zstd_mt: struct { io: std.Io, src: []const u8, dst: []u8, mt: *const zstd.MtResult, threads: usize },
    slz: struct { ctx: *decoder.DecompressContext, comp: []const u8, dst: []u8 },

    fn run(self: BenchDecomp) !void {
        switch (self) {
            .lz4_st => |s| _ = try lz4.decompress(s.dst, s.comp, s.orig_size),
            .lz4_mt => |s| try lz4.decompressMt(s.io, s.src, s.dst, s.mt, s.threads),
            .lzt_st => |s| _ = try lzt.decompress(s.dst, s.comp, s.orig_size),
            .lzt_mt => |s| try lzt.decompressMt(s.io, s.src, s.dst, s.mt, s.threads),
            .zstd_st => |s| _ = try zstd.decompress(s.dst, s.comp),
            .zstd_mt => |s| try zstd.decompressBlocksMt(s.io, s.src, s.dst, s.mt, s.threads),
            .slz => |s| _ = try s.ctx.decompress(s.comp, s.dst),
        }
    }
};

/// Runs a warmup decompress, then a best-of-N timed loop, and records the
/// result row (compressed size, ratio, MB/s for compress and decompress).
fn benchmarkCodec(
    codec_name: []const u8,
    comp_size: usize,
    comp_ns: u64,
    decomp: BenchDecomp,
    src_len: usize,
    mb: f64,
    runs: usize,
    io: std.Io,
    w: *std.Io.Writer,
    results: []BenchCompareRow,
    result_count: *usize,
) !void {
    // Warmup decompress.
    try decomp.run();
    // Timed best-of-N decompress.
    var best_dec_ns: u64 = std.math.maxInt(u64);
    for (0..runs) |_| {
        const timer_start = std.Io.Clock.awake.now(io);
        try decomp.run();
        const elapsed = @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds()));
        if (elapsed < best_dec_ns) best_dec_ns = elapsed;
    }
    // Record result.
    const comp_ms = @as(f64, @floatFromInt(comp_ns)) / 1_000_000.0;
    const dec_ms = @as(f64, @floatFromInt(best_dec_ns)) / 1_000_000.0;
    results[result_count.*].setName(codec_name);
    results[result_count.*].comp_size = comp_size;
    results[result_count.*].ratio = @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(src_len)) * 100.0;
    results[result_count.*].comp_mbps = mb * 1000.0 / comp_ms;
    results[result_count.*].dec_mbps = mb * 1000.0 / dec_ms;
    result_count.* += 1;
    try w.writeAll(" done\n");
}

fn runBenchCompare(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args, fast_only: bool) !void {
    const in_path = requireInput(args, w);
    const runs = args.runs orelse 3;
    const threads: c_int = if (args.threads > 0) @intCast(args.threads) else @intCast(std.Thread.getCpuCount() catch 8);

    if (runs == 0) die(w, "error: runs must be >= 1\n");

    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    cache_detect.printSystemInfo(w, io);
    try w.print("comparison benchmark: {s} ({d} bytes, {d:.2} MB)\n", .{ in_path, src.len, mb });
    try w.print("threads: {d}, runs: {d}\n\n", .{ threads, runs });
    try w.flush();

    var results: [16]BenchCompareRow = @splat(.{});
    var result_count: usize = 0;

    // Shared buffers — allocate to the max bound across all compressors.
    const zstd_bound = zstd.compressBound(src.len);
    const lz4_bound = lz4.compressBound(src.len);
    const lzt_bound = lzt.compressBound(src.len);
    const slz_bound = encoder.compressBound(src.len);
    const max_bound = @max(zstd_bound, @max(lz4_bound, @max(lzt_bound, slz_bound)));
    const compressed = try allocator.alloc(u8, max_bound);
    defer allocator.free(compressed);
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(decompressed);

    // ── LZ4 default ──
    {
        const is_single = threads == 1;
        const lz4_label: []const u8 = if (is_single) "LZ4" else "LZ4 MT";
        try w.print("  {s} ...", .{lz4_label});
        try w.flush();
        if (is_single) {
            const comp_timer = std.Io.Clock.awake.now(io);
            const comp_size = lz4.compress(compressed, src) catch 0;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (comp_size > 0) {
                try benchmarkCodec(lz4_label, comp_size, comp_ns,
                    .{ .lz4_st = .{ .dst = decompressed[0..src.len], .comp = compressed[0..comp_size], .orig_size = src.len } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        } else {
            const comp_timer = std.Io.Clock.awake.now(io);
            var mt_result: ?lz4.MtResult = lz4.compressMt(allocator, io, src, @intCast(threads), null) catch null;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (mt_result) |*mr| {
                defer mr.deinit();
                try benchmarkCodec(lz4_label, mr.total_compressed, comp_ns,
                    .{ .lz4_mt = .{ .io = io, .src = src, .dst = decompressed[0..src.len], .mt = mr, .threads = @intCast(threads) } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        }
        try w.flush();
    }

    flushMemory();

    // ── LZ4 HC levels ──
    const lz4_hc_levels_fast = [_]c_int{9};
    const lz4_hc_levels_full = [_]c_int{ 4, 9, 12 };
    const lz4_hc_levels: []const c_int = if (fast_only) &lz4_hc_levels_fast else &lz4_hc_levels_full;
    for (lz4_hc_levels) |hc_level| {
        var name_buf: [16]u8 = undefined;
        const is_single = threads == 1;
        const label = if (is_single)
            std.fmt.bufPrint(&name_buf, "LZ4 HC {d}", .{hc_level}) catch "LZ4 HC ?"
        else
            std.fmt.bufPrint(&name_buf, "LZ4 HC {d} MT", .{hc_level}) catch "LZ4 HC ?";
        try w.print("  {s} ...", .{label});
        try w.flush();
        if (is_single) {
            const comp_timer = std.Io.Clock.awake.now(io);
            const comp_size = lz4.compressHc(compressed, src, hc_level) catch 0;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (comp_size > 0) {
                try benchmarkCodec(label, comp_size, comp_ns,
                    .{ .lz4_st = .{ .dst = decompressed[0..src.len], .comp = compressed[0..comp_size], .orig_size = src.len } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        } else {
            const comp_timer = std.Io.Clock.awake.now(io);
            var mt_result: ?lz4.MtResult = lz4.compressMt(allocator, io, src, @intCast(threads), hc_level) catch null;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (mt_result) |*mr| {
                defer mr.deinit();
                try benchmarkCodec(label, mr.total_compressed, comp_ns,
                    .{ .lz4_mt = .{ .io = io, .src = src, .dst = decompressed[0..src.len], .mt = mr, .threads = @intCast(threads) } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        }
        try w.flush();
    }

    flushMemory();

    // ── LZTurbo 10 ──
    {
        const is_single = threads == 1;
        const lzt_label: []const u8 = if (is_single) "LZTurbo 10" else "LZTurbo 10 MT";
        try w.print("  {s} ...", .{lzt_label});
        try w.flush();
        if (is_single) {
            const comp_timer = std.Io.Clock.awake.now(io);
            const comp_size = lzt.compress(compressed, src) catch 0;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (comp_size > 0) {
                try benchmarkCodec(lzt_label, comp_size, comp_ns,
                    .{ .lzt_st = .{ .dst = decompressed[0..src.len], .comp = compressed[0..comp_size], .orig_size = src.len } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        } else {
            const comp_timer = std.Io.Clock.awake.now(io);
            var mt_result: ?lzt.MtResult = lzt.compressMt(allocator, io, src, @intCast(threads)) catch null;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (mt_result) |*mr| {
                defer mr.deinit();
                try benchmarkCodec(lzt_label, mr.total_compressed, comp_ns,
                    .{ .lzt_mt = .{ .io = io, .src = src, .dst = decompressed[0..src.len], .mt = mr, .threads = @intCast(threads) } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        }
        try w.flush();
    }

    flushMemory();

    // ── zstd levels ──
    const zstd_levels_fast = [_]c_int{ 1, 3, 19 };
    const zstd_levels_full = [_]c_int{ 1, 3, 9, 19 };
    const zstd_levels: []const c_int = if (fast_only) &zstd_levels_fast else &zstd_levels_full;
    for (zstd_levels) |zstd_level| {
        var name_buf: [16]u8 = undefined;
        const is_single = threads == 1;
        const label = if (is_single)
            std.fmt.bufPrint(&name_buf, "zstd {d}", .{zstd_level}) catch "zstd ?"
        else
            std.fmt.bufPrint(&name_buf, "zstd {d} MT", .{zstd_level}) catch "zstd ?";
        try w.print("  {s} ...", .{label});
        try w.flush();
        if (is_single) {
            const comp_timer = std.Io.Clock.awake.now(io);
            const comp_size = zstd.compress(compressed, src, zstd_level) catch 0;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (comp_size > 0) {
                try benchmarkCodec(label, comp_size, comp_ns,
                    .{ .zstd_st = .{ .dst = decompressed[0..src.len], .comp = compressed[0..comp_size] } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        } else {
            const comp_timer = std.Io.Clock.awake.now(io);
            var mt_result: ?zstd.MtResult = zstd.compressBlocksMt(allocator, io, src, @intCast(threads), zstd_level) catch null;
            const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
            if (mt_result) |*mr| {
                defer mr.deinit();
                try benchmarkCodec(label, mr.total_compressed, comp_ns,
                    .{ .zstd_mt = .{ .io = io, .src = src, .dst = decompressed[0..src.len], .mt = mr, .threads = @intCast(threads) } },
                    src.len, mb, runs, io, w, &results, &result_count);
            } else {
                try w.writeAll(" FAILED\n");
            }
        }
        try w.flush();
    }

    flushMemory();

    // ── StreamLZ levels 1, 3, 5, 6, 8, 9, 11 (with threads) ──
    const slz_levels_fast = [_]u8{ 1, 3, 5 };
    const slz_levels_full = [_]u8{ 1, 3, 5, 6, 8, 9, 11 };
    const slz_levels: []const u8 = if (fast_only) &slz_levels_fast else &slz_levels_full;
    for (slz_levels) |slz_level| {
        flushMemory();
        var name_buf: [16]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "SLZ L{d}", .{slz_level}) catch "SLZ ?";
        try w.print("  {s} ...", .{label});
        try w.flush();

        const opts: encoder.Options = .{ .level = slz_level, .num_threads = @intCast(threads) };
        const comp_timer = std.Io.Clock.awake.now(io);
        const comp_size = encoder.compressFramedWithIo(allocator, io, src, compressed, opts) catch 0;
        const comp_ns = @as(u64, @intCast(comp_timer.untilNow(io, .awake).toNanoseconds()));
        if (comp_size > 0) {
            var dec_ctx = decoder.DecompressContext.initThreadedWithIo(allocator, io, @intCast(threads));
            defer dec_ctx.deinit();
            _ = dec_ctx.decompress(compressed[0..comp_size], decompressed) catch {
                try w.writeAll(" decompress FAILED\n");
                try w.flush();
                continue;
            };
            try benchmarkCodec(label, comp_size, comp_ns,
                .{ .slz = .{ .ctx = &dec_ctx, .comp = compressed[0..comp_size], .dst = decompressed } },
                src.len, mb, runs, io, w, &results, &result_count);
        } else {
            try w.writeAll(" FAILED\n");
        }
        try w.flush();
    }

    // ── Print results table ──
    try w.writeAll("\nCompressor    | Compressed         | Ratio  | Compress   | Decompress\n");
    try w.writeAll("--------------+--------------------+--------+------------+-----------\n");

    for (results[0..result_count]) |*res| {
        var bytes_buf: [32]u8 = undefined;
        const bytes_str = fmtBytes(&bytes_buf, res.comp_size);
        var comp_mbps_buf: [16]u8 = undefined;
        const comp_mbps_str = fmtMbps(&comp_mbps_buf, res.comp_mbps);
        var dec_mbps_buf: [16]u8 = undefined;
        const dec_mbps_str = fmtMbps(&dec_mbps_buf, res.dec_mbps);

        try w.print("{s:<13} | {s:>14} bytes | {d:>5.1}% | {s:>7} MB/s | {s:>7} MB/s\n", .{
            res.name(),
            bytes_str,
            res.ratio,
            comp_mbps_str,
            dec_mbps_str,
        });
    }

    try w.writeAll("\nVersions: LZ4 1.10.0, zstd 1.5.7, LZTurbo 1.2 (Zig reimpl)\n");
    if (threads == 1) {
        try w.writeAll("Single-threaded: all compressors use full-stream mode (no block splitting).\n");
    } else {
        try w.print("\nThreading ({d} threads, 4 MB independent blocks):\n", .{threads});
        try w.writeAll("  LZ4:      compress MT, decompress MT\n");
        try w.writeAll("  LZTurbo:  compress MT, decompress MT\n");
        try w.writeAll("  zstd:     compress MT, decompress MT\n");
        try w.writeAll("  StreamLZ: compress MT (L1, L6+), decompress MT (all levels)\n");
    }
}

fn fmtMbps(buf: []u8, value: f64) []const u8 {
    if (value >= 100.0) {
        return std.fmt.bufPrint(buf, "{d:>.0}", .{value}) catch "?";
    } else {
        return std.fmt.bufPrint(buf, "{d:>.1}", .{value}) catch "?";
    }
}

// ─── Info ────────────────────────────────────────────────────────────

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

    // Walk block headers.
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
            block_index,
            pos,
            block_hdr.compressed_size,
            block_hdr.decompressed_size,
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

// ─── Train ─────────────────────────────────────────────────────────

// ─── Forward-LZ analysis ────────────────────────────────────────────

fn runForwardAnalyze(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const src = readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("Forward-LZ analysis: {s} ({d} bytes, {d:.2} MB)\n\n", .{ in_path, src.len, mb });
    try w.flush();

    const timer_start = std.Io.Clock.awake.now(io);
    const result = forward_lz.analyzeForwardLz(allocator, src) catch |err| {
        try w.print("error: forward-LZ analysis failed: {s}\n", .{@errorName(err)});
        return;
    };
    const elapsed_ns = @as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds()));
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    const ratio = @as(f64, @floatFromInt(result.total_size)) / @as(f64, @floatFromInt(src.len)) * 100.0;

    try w.print("Encode time: {d:.1} ms\n\n", .{elapsed_ms});
    try w.print("Compressed: {d:>12} bytes ({d:.1}%)\n", .{ result.total_size, ratio });
    try w.print("  patterns:   {d:>12} bytes (tANS)\n", .{result.pattern_stream_size});
    try w.print("  positions:  {d:>12} bytes (tANS)\n", .{result.position_stream_size});
    try w.print("  literals:   {d:>12} bytes (tANS)\n", .{result.literal_stream_size});
    try w.print("  lit_pos:    {d:>12} bytes (tANS)\n", .{result.literal_pos_stream_size});
    try w.print("  control:    {d:>12} bytes (tANS)\n", .{result.control_stream_size});
    try w.print("\n", .{});
    try w.print("Forward refs: {d:>12} (patterns appearing 2+ times)\n", .{result.num_forward_refs});
    try w.print("Singletons:   {d:>12} (unique matches)\n", .{result.num_singletons});
    try w.print("Scatter writes:{d:>11} (total dest positions)\n", .{result.num_scatter_writes});
    try w.print("Literals:     {d:>12} (uncovered bytes)\n", .{result.num_literals});

    // Compare to StreamLZ L1
    const bound = encoder.compressBound(src.len);
    const comp_buf = try allocator.alloc(u8, bound);
    defer allocator.free(comp_buf);
    const slz_size = encoder.compressFramedWithIo(allocator, io, src, comp_buf, .{ .level = 1 }) catch 0;
    if (slz_size > 0) {
        const slz_ratio = @as(f64, @floatFromInt(slz_size)) / @as(f64, @floatFromInt(src.len)) * 100.0;
        try w.print("\nStreamLZ L1: {d:>12} bytes ({d:.1}%)\n", .{ slz_size, slz_ratio });
        try w.print("Forward-LZ:  {d:>12} bytes ({d:.1}%)\n", .{ result.total_size, ratio });
        if (result.total_size < slz_size) {
            try w.print("Forward wins by {d} bytes\n", .{slz_size - result.total_size});
        } else {
            try w.print("Backward wins by {d} bytes\n", .{result.total_size - slz_size});
        }
    }
}

fn runTrain(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = requireInput(args, w);
    const out_path = args.output orelse "dictionary.bin";
    const dict_size: usize = 32768;

    // Read all files from the input directory as training samples.
    var dir = std.Io.Dir.cwd().openDir(io, in_path, .{ .iterate = true }) catch |err| {
        try w.print("error: cannot open directory '{s}': {s}\n", .{ in_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer dir.close(io);

    var samples: std.ArrayList([]const u8) = .empty;
    defer {
        for (samples.items) |s| allocator.free(s);
        samples.deinit(allocator);
    }

    var total_bytes: usize = 0;
    var file_count: usize = 0;
    var iter = dir.iterate();
    while (iter.next(io) catch null) |entry| {
        if (entry.kind != .file) continue;
        const data = dir.readFileAlloc(io, entry.name, allocator, @enumFromInt(64 * 1024 * 1024)) catch continue;
        if (data.len < 16) {
            allocator.free(data);
            continue;
        }
        try samples.append(allocator, data);
        total_bytes += data.len;
        file_count += 1;
    }

    if (file_count == 0) {
        try w.print("error: no usable files found in '{s}'\n", .{in_path});
        try w.flush();
        std.process.exit(1);
    }

    try w.print("training: {d} files, {d:.1} KB total\n", .{
        file_count,
        @as(f64, @floatFromInt(total_bytes)) / 1024.0,
    });

    const timer_start = std.Io.Clock.awake.now(io);
    var result = trainer.train(allocator, samples.items, .{
        .dict_size = dict_size,
    }) catch |err| {
        try w.print("error: training failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    defer result.deinit();
    const train_ms = @as(f64, @floatFromInt(@as(u64, @intCast(timer_start.untilNow(io, .awake).toNanoseconds())))) / 1_000_000.0;

    // Write dictionary.
    const out_file = std.Io.Dir.cwd().createFile(io, out_path, .{}) catch |err| {
        try w.print("error: cannot create '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer out_file.close(io);
    out_file.writeStreamingAll(io, result.dict) catch |err| {
        try w.print("error: cannot write '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };

    try w.print("trained: {d} bytes dictionary -> {s}  ({d:.0} ms)\n", .{
        result.dict.len,
        out_path,
        train_ms,
    });

    // Quick quality check: compress each sample with and without dictionary.
    var total_no_dict: usize = 0;
    var total_with_dict: usize = 0;
    for (samples.items) |sample| {
        if (sample.len < 64) continue;
        const bound = encoder.compressBound(sample.len);
        const comp_buf = allocator.alloc(u8, bound) catch continue;
        defer allocator.free(comp_buf);

        // Serial intentional: small sample, parallel overhead not worth it.
        const no_dict_size = encoder.compressFramed(allocator, sample, comp_buf, .{
            .level = 3,
        }) catch continue;
        total_no_dict += no_dict_size;

        // Serial intentional: small sample, parallel overhead not worth it.
        const with_dict_size = encoder.compressFramed(allocator, sample, comp_buf, .{
            .level = 3,
            .dictionary = result.dict,
            .dictionary_id = 0x10000001,
        }) catch continue;
        total_with_dict += with_dict_size;
    }

    if (total_no_dict > 0) {
        const improvement = @as(f64, @floatFromInt(total_no_dict)) - @as(f64, @floatFromInt(total_with_dict));
        const pct = improvement / @as(f64, @floatFromInt(total_no_dict)) * 100.0;
        try w.print("quality: no_dict={d} bytes, with_dict={d} bytes ({d:.2}% improvement)\n", .{
            total_no_dict,
            total_with_dict,
            pct,
        });
    }
}
