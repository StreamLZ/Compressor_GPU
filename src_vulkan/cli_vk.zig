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
const builtin = @import("builtin");

const driver = @import("driver.zig");
const slz1_codec = @import("slz1_codec.zig");
const wire_format = @import("wire_format.zig");
const frame = @import("../src/format/frame_format.zig");
const vk = @import("vk_api.zig");
const instance_mod = @import("instance.zig");
const probe_mod = @import("probe.zig");
const device_mod = @import("device.zig");
const l1_codec = @import("l1_codec.zig");

const VK_SAFE_SPACE: usize = 64;

const Mode = enum { compress, decompress, version, help, probe, bench, decompress_bench };

const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    /// Bench-mode run count (`-r N`). Defaults to 3 to match the CUDA
    /// `streamlz -b` default. Only consulted in `.bench` mode.
    runs: u32 = 3,
    /// Raw `--device <arg>` payload (lifetime: argv). Parsed at run-time:
    /// pure-digit strings become `.by_index`; anything else becomes
    /// `.by_name` (case-insensitive substring match against deviceName).
    /// `null` means "no flag" → driver consults SLZ_VK_DEVICE_INDEX env
    /// var, else falls back to first compute-capable device.
    device_spec: ?[]const u8 = null,
};

// ── Bench-mode QPC helpers ────────────────────────────────────────────
// Windows has the highest-resolution clock the CUDA-side bench uses
// for its `e2e` timings; mirror it here so the two backends report
// directly comparable numbers. On non-Windows the fallback resolution
// is whatever std.time.nanoTimestamp gives — typically still
// sub-microsecond and good enough for the ms-scale bench output.
const win32_qpc = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};
fn qpcNow() i64 {
    if (builtin.os.tag != .windows) return @intCast(std.time.nanoTimestamp());
    var c: i64 = 0;
    _ = win32_qpc.QueryPerformanceCounter(&c);
    return c;
}
fn qpcNs(from: i64, to: i64) u64 {
    if (builtin.os.tag != .windows) {
        const d = to - from;
        return if (d > 0) @intCast(d) else 0;
    }
    var freq: i64 = 0;
    _ = win32_qpc.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

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
        \\  -c                  Compress (default)
        \\  -d                  Decompress
        \\  -b                  Benchmark — compress + N decompress runs, report
        \\                      e2e (wall) and d2d (kernel) timings in the
        \\                      same format as the CUDA-backend `streamlz -b`
        \\  -p, --probe         List all Vulkan physical devices and exit
        \\
        \\Options:
        \\  -l N                Compression level (only 1 supported in v1)
        \\  -r N                Bench: number of decompress runs (default 3)
        \\  -o PATH             Output path (compress / decompress modes)
        \\  --device <N|name>   Select Vulkan device by zero-based index
        \\                      (matches --probe order) or by case-insensitive
        \\                      substring of the device name. Env var
        \\                      SLZ_VK_DEVICE_INDEX=<N> is used as a fallback
        \\                      when --device is not passed.
        \\  -h, --help          Show this help
        \\  -V, --version       Show version
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
        if (eql(arg, "-V") or eql(arg, "--version")) {
            result.mode = .version;
            continue;
        }
        if (eql(arg, "-h") or eql(arg, "--help")) {
            result.mode = .help;
            continue;
        }
        if (eql(arg, "-c")) {
            result.mode = .compress;
            continue;
        }
        if (eql(arg, "-d")) {
            result.mode = .decompress;
            continue;
        }
        if (eql(arg, "-db")) {
            result.mode = .decompress_bench;
            continue;
        }
        if (eql(arg, "-b")) {
            result.mode = .bench;
            continue;
        }
        if (eql(arg, "-r")) {
            i += 1;
            if (i >= raw.len) die(w, "error: -r requires a value\n", .{});
            result.runs = std.fmt.parseInt(u32, raw[i], 10) catch
                die(w, "error: invalid -r value '{s}'\n", .{raw[i]});
            if (result.runs == 0) die(w, "error: -r must be >= 1\n", .{});
            continue;
        }
        if (eql(arg, "-p") or eql(arg, "--probe")) {
            result.mode = .probe;
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
        if (eql(arg, "--device")) {
            i += 1;
            if (i >= raw.len) die(w, "error: --device requires a value (index or name substring)\n", .{});
            result.device_spec = raw[i];
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

/// Translate the `--device <arg>` string into a `DeviceSelector`. Pure-
/// digit strings become `.by_index`; everything else is treated as a
/// case-insensitive substring of the device name. Returns `.default` for
/// a null spec so callers can unconditionally call `driver.setSelector`.
fn selectorFromSpec(spec: ?[]const u8) device_mod.DeviceSelector {
    const s = spec orelse return .default;
    if (s.len == 0) return .default;
    // All-digits → by index. Anything else (incl. "0x...", "1a") → by name.
    var all_digits = true;
    for (s) |c| {
        if (c < '0' or c > '9') {
            all_digits = false;
            break;
        }
    }
    if (all_digits) {
        const n = std.fmt.parseInt(u32, s, 10) catch return .{ .by_name = s };
        return .{ .by_index = n };
    }
    return .{ .by_name = s };
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

/// Print the deviceName of the currently-bound Vulkan physical device.
/// Must be called AFTER `ensureInitWithSelector` so `driver.g_default.pd`
/// is populated. The line goes to the regular CLI writer (not gated
/// behind SLZ_VK_PROFILE_DECODE=1) so every bench / compress / decompress
/// run makes the bound hardware impossible to miss — the most common
/// cause of "regression" reports has been measuring on the wrong device
/// (Intel iGPU vs NVIDIA dGPU) without realizing it.
fn printBoundDevice(w: *std.Io.Writer) !void {
    if (driver.g_default.pd == null) return;
    var name_buf: [vk.VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = @splat(0);
    const name = device_mod.readDeviceName(driver.g_default.pd, name_buf[0..]);
    try w.print("Device: {s}\n", .{name});
}

fn ensureInitWithSelector(w: *std.Io.Writer, args: Args) !void {
    driver.setSelector(selectorFromSpec(args.device_spec));
    driver.ensureInit() catch |err| {
        // Selector errors are user-facing: print a friendly message and
        // exit with status 2 (matching `die`) so the shell sees failure
        // without a stack trace polluting the terminal. Other errors
        // (LoaderInitFailed, CreateInstanceFailed, ...) fall through to
        // the default trace-printing path because they indicate a real
        // bug, not a CLI usage mistake.
        switch (err) {
            error.DeviceIndexOutOfRange => die(
                w,
                "error: device {?s} not found, --probe to list available devices\n",
                .{args.device_spec},
            ),
            error.DeviceNameNotFound => die(
                w,
                "error: no device matched '{?s}', --probe to list available devices\n",
                .{args.device_spec},
            ),
            error.DeviceNameAmbiguous => die(
                w,
                "error: multiple devices matched '{?s}', --probe to list and pick a more specific substring or use --device <N>\n",
                .{args.device_spec},
            ),
            else => {
                try w.print("error: Vulkan init failed: {s}\n", .{@errorName(err)});
                return err;
            },
        }
    };
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

    try ensureInitWithSelector(w, args);
    defer driver.deinit();
    try printBoundDevice(w);

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

    try ensureInitWithSelector(w, args);
    defer driver.deinit();
    try printBoundDevice(w);

    // Page-align so the SLZ1 decoder takes the
    // VK_EXT_external_memory_host fast path (see bench comment in
    // runBenchOrPrintHelp for the rationale).
    const out_buf_size = ((content_size + VK_SAFE_SPACE + 4095) & ~@as(usize, 4095));
    const out_buf = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), out_buf_size);
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

/// Enumerate every Vulkan physical device the loader reports and print one
/// line per device in the same format as `vk-probe`. Stays in this file
/// (rather than reusing probe_main.zig's `main`) because the CLI binary
/// can't fork into another `main` — we instead duplicate the ~30 lines of
/// loader/instance bring-up + per-device print loop here.
fn runProbe(w: *std.Io.Writer) !void {
    if (!vk.init()) {
        try w.writeAll("error: vulkan-1.dll loader init failed\n");
        return error.LoaderInitFailed;
    }
    const inst = instance_mod.createInstance(false) catch |err| {
        try w.print("error: createInstance failed: {s}\n", .{@errorName(err)});
        return err;
    };
    defer instance_mod.destroyInstance(inst);

    const enumerate = vk.vkEnumeratePhysicalDevices_fn orelse {
        try w.writeAll("error: vkEnumeratePhysicalDevices unresolved\n");
        return error.MissingProc;
    };
    var count: u32 = 0;
    if (enumerate(inst, &count, null) != vk.VK_SUCCESS) {
        try w.writeAll("error: enumeratePhysicalDevices(count) failed\n");
        return error.EnumerateFailed;
    }
    if (count == 0) {
        try w.writeAll("no Vulkan physical devices reported\n");
        return;
    }
    const MAX_PD: u32 = 16;
    if (count > MAX_PD) count = MAX_PD;
    var devices: [MAX_PD]vk.VkPhysicalDevice = @splat(null);
    const r = enumerate(inst, &count, @ptrCast(&devices));
    if (r != vk.VK_SUCCESS and r != vk.VK_INCOMPLETE) {
        try w.writeAll("error: enumeratePhysicalDevices(fill) failed\n");
        return error.EnumerateFailed;
    }

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const pd = devices[i];
        if (pd == null) continue;
        const result = probe_mod.probe(inst, pd);
        const major = vk.VK_API_VERSION_MAJOR(result.api_version);
        const minor = vk.VK_API_VERSION_MINOR(result.api_version);
        const patch = vk.VK_API_VERSION_PATCH(result.api_version);
        try w.print(
            "device[{d}]: {s} vendor=0x{X:0>4} api={d}.{d}.{d} subgroup={d}[{d}..{d}] tier={s} bda={s} int64={s} ts={s} sync2={s} int8={s} 8stor={s} nvpart={s}\n",
            .{
                i,
                result.device_name,
                result.vendor_id,
                major,
                minor,
                patch,
                result.subgroup_size,
                result.subgroup_size_min,
                result.subgroup_size_max,
                probe_mod.tierName(result.tier),
                yn(result.has_buffer_device_address),
                yn(result.has_shader_int64),
                yn(result.has_timeline_semaphore),
                yn(result.has_synchronization2),
                yn(result.has_shader_int8),
                yn(result.has_8bit_storage),
                yn(result.has_nv_subgroup_partitioned),
            },
        );
    }
}

fn yn(b: bool) []const u8 {
    return if (b) "y" else "n";
}

/// Return the median of `xs` (mutates the slice; sorts in place).
/// Empty slice returns 0.
fn medianU64(xs: []u64) u64 {
    if (xs.len == 0) return 0;
    std.mem.sort(u64, xs, {}, std.sort.asc(u64));
    return xs[xs.len / 2];
}

/// `streamlz_vk -b <file>` — mirrors the CUDA `streamlz -b` output
/// format so cross-backend benchmarks read line-for-line
/// comparably. Format:
///
///   Input: <path> (<bytes> bytes, <MiB> MB)
///     Compress: <ms>ms (<MB/s> MB/s)
///   Level 1: <bytes> -> <bytes> (<%>)
///
///     Decompress run 1: e2e <ms>ms (<MB/s>)  d2d <ms>ms (<MB/s>)
///     ...
///     Decompress median: e2e <ms>ms (<MB/s>)  d2d <ms>ms (<MB/s>)
///
///   Round-trip: PASS|FAIL
///
/// `e2e` = QPC wall-clock around the synchronous decompress call
/// (host setup + dispatch + GPU exec + readback). `d2d` = pure
/// GPU dispatch time, pulled from slzGetLastTimings_vk after each
/// run.
fn runBench(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    if (args.level != 1) {
        try w.print("error: bench requires -l 1 (got -l {d})\n", .{args.level});
        return error.UnsupportedLevel;
    }
    const in_path = args.input orelse {
        try w.writeAll("error: -b requires an input file\n");
        return error.NoInput;
    };
    const runs: u32 = args.runs;

    const src = readFileAll(allocator, io, in_path) catch |err| {
        try w.print("error: cannot read '{s}': {s}\n", .{ in_path, @errorName(err) });
        return err;
    };
    defer allocator.free(src);
    if (src.len == 0) {
        try w.writeAll("error: input file is empty\n");
        return error.EmptyInput;
    }

    try ensureInitWithSelector(w, args);
    defer driver.deinit();
    try printBoundDevice(w);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("Input: {s} ({d} bytes, {d:.2} MB)\n", .{ in_path, src.len, mb });

    // ── Compress (one timed pass, after a warm-up of equal shape). ──
    const bound = slz1_codec.slz1Bound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);

    // Warm-up encode to populate the pipeline cache; the timed run
    // mirrors the second call so the cache effect doesn't skew the
    // headline number.
    _ = try slz1_codec.encodeL1ToSlz1(&driver.g_default, io, allocator, src, compressed);

    var comp_size: usize = 0;
    {
        const t0 = qpcNow();
        comp_size = try slz1_codec.encodeL1ToSlz1(&driver.g_default, io, allocator, src, compressed);
        const t1 = qpcNow();
        const ns = qpcNs(t0, t1);
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        try w.print("  Compress: {d:.0}ms ({d:.1} MB/s)\n", .{ ms, mb * 1000.0 / ms });
    }
    try w.print("Level 1: {d} -> {d} bytes ({d:.1}%)\n\n", .{
        src.len, comp_size, @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(src.len)) * 100.0,
    });

    // ── Decompress runs ──────────────────────────────────────────
    // Page-align the decode output buffer AND round the length up to a
    // multiple of the page so the SLZ1 decoder can take the
    // VK_EXT_external_memory_host fast path (the import requires
    // pointer + size both aligned to `minImportedHostPointerAlignment`
    // — typically 4096). Without alignment the decoder falls back to
    // the dst_stage staging copy + post-submit host @memcpy, which was
    // what made enwik8 e2e ~7 ms slower than CUDA's cuMemcpyDtoH_v2.
    const dec_buf_alloc_size = ((src.len + VK_SAFE_SPACE + 4095) & ~@as(usize, 4095));
    const dec_buf = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), dec_buf_alloc_size);
    defer allocator.free(dec_buf);

    // Warm-up decompress (same rationale as the encode warm-up).
    _ = try slz1_codec.decodeSlz1ToBytes(&driver.g_default, io, allocator, compressed[0..comp_size], dec_buf);

    const dec_e2e_ns = try allocator.alloc(u64, runs);
    defer allocator.free(dec_e2e_ns);
    const dec_d2d_ns = try allocator.alloc(u64, runs);
    defer allocator.free(dec_d2d_ns);

    var r: u32 = 0;
    while (r < runs) : (r += 1) {
        // Reset the per-process dispatch-timing global to 0 so we
        // know the per-run d2d came from THIS call (the codec
        // resets it on entry but this is belt-and-suspenders).
        l1_codec.last_decode_dispatch_ns = 0;
        const t0 = qpcNow();
        _ = try slz1_codec.decodeSlz1ToBytes(&driver.g_default, io, allocator, compressed[0..comp_size], dec_buf);
        const t1 = qpcNow();
        dec_e2e_ns[r] = qpcNs(t0, t1);
        dec_d2d_ns[r] = l1_codec.last_decode_dispatch_ns;
        const e2e_ms = @as(f64, @floatFromInt(dec_e2e_ns[r])) / 1_000_000.0;
        const d2d_ms = @as(f64, @floatFromInt(dec_d2d_ns[r])) / 1_000_000.0;
        if (dec_d2d_ns[r] > 0) {
            try w.print(
                "  Decompress run {d}: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n",
                .{ r + 1, e2e_ms, mb * 1000.0 / e2e_ms, d2d_ms, mb * 1000.0 / d2d_ms },
            );
        } else {
            try w.print(
                "  Decompress run {d}: {d:.0}ms ({d:.1} MB/s)\n",
                .{ r + 1, e2e_ms, mb * 1000.0 / e2e_ms },
            );
        }
        // Optional profile breakdown — enable with SLZ_VK_PROFILE_DECODE=1.
        const want_profile: bool = blk: {
            const raw = std.c.getenv("SLZ_VK_PROFILE_DECODE") orelse break :blk false;
            const s = std.mem.span(raw);
            break :blk s.len > 0 and s[0] != '0';
        };
        if (want_profile) {
            try w.print(
                "    profile: unwrap={d:.2}ms alloc={d:.2}ms memset={d:.2}ms fill={d:.2}ms descset={d:.2}ms dispatch={d:.2}ms readback={d:.2}ms\n",
                .{
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_unwrap_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_alloc_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_memset_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_fill_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_descset_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_dispatch_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_readback_ns)) / 1_000_000.0,
                },
            );
            // dispatch sub-phase breakdown: GPU kernel ns + GPU copy ns
            // (vkCmdCopyBuffer) + host-wall around vkQueueSubmit/wait.
            // readback_ns above is purely the host @memcpy because
            // submitOneWithCopy already waited on the fence inside.
            try w.print(
                "      sub: gpu_kernel={d:.2}ms gpu_copy={d:.2}ms submit_wait_wall={d:.2}ms\n",
                .{
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_gpu_kernel_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_gpu_copy_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_submit_wait_wall_ns)) / 1_000_000.0,
                },
            );
            try w.print(
                "      unwrap: wall={d:.2}ms rec={d:.2}ms sub={d:.2}ms wait={d:.2}ms qry={d:.2}ms\n",
                .{
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_unwrap_submit_wait_wall_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_unwrap_record_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_unwrap_submit_call_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_unwrap_wait_call_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_unwrap_query_read_ns)) / 1_000_000.0,
                },
            );
            try w.print(
                "      decode: rec={d:.2}ms sub={d:.2}ms wait={d:.2}ms qry={d:.2}ms\n",
                .{
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_dec_record_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_dec_submit_call_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_dec_wait_call_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_slz_dec_query_read_ns)) / 1_000_000.0,
                },
            );
            // Caller-import path telemetry — exposes the gate decision
            // from slz1_codec so the bench reader can confirm the
            // VK_EXT_external_memory_host fast path is actually firing
            // (vs the two-buffer staging-copy fallback). taken=1 means
            // the kernel wrote into dst_b and vkCmdCopyBuffer'd straight
            // into the caller's pageable `out` buffer (no host @memcpy).
            try w.print(
                "      import: taken={d} has_ext={d} ptr_aligned={d} align={d} size={d}\n",
                .{
                    slz1_codec.last_decode_slz_caller_import_taken,
                    slz1_codec.last_decode_slz_caller_has_ext,
                    slz1_codec.last_decode_slz_caller_ptr_aligned,
                    slz1_codec.last_decode_slz_caller_import_align,
                    slz1_codec.last_decode_slz_caller_import_size,
                },
            );
            // Per-kernel GPU breakdown from the VkQueryPool timestamps
            // written around each dispatch in recordDecodePipelineInto +
            // recordAndSubmitMergedDecode. Format matches what the
            // CUDA-side `nsys stats --report cuda_gpu_kern_sum` table
            // gives for the corresponding kernels so a parity comparison
            // can be done by eye. `lz_decode` and `dst_copy` reuse the
            // existing `sub: gpu_kernel`/`gpu_copy` numbers above; this
            // line covers the front-half pipeline kernels + transfers.
            try w.print(
                "      kper: walk={d:.3} prefix={d:.3} scan={d:.3} compact_lit={d:.3} compact_tok={d:.3} compact_hi={d:.3} compact_lo={d:.3} compact_raw={d:.3} gather={d:.3} unwrap={d:.3} frame_dma={d:.3} fills={d:.3} meta_copy={d:.3}\n",
                .{
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_walk_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_prefix_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_scan_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_compact_lit_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_compact_tok_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_compact_hi_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_compact_lo_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_compact_raw_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_gather_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_unwrap_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_frame_dma_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_fills_ns)) / 1_000_000.0,
                    @as(f64, @floatFromInt(slz1_codec.last_decode_per_kernel_meta_copy_ns)) / 1_000_000.0,
                },
            );
        }
    }

    // Print the one-shot readback diagnostic after all runs (it ran
    // once internally on the first decode call when the profile env
    // knob was on). Lets the reviewer correlate the per-run readback
    // ns with the underlying memory-type choice + host @memcpy ceiling.
    if (slz1_codec.diag_ran) {
        try w.print("readback-diag: memory_types[{d}]:\n", .{slz1_codec.diag_n_memory_types});
        var i: u32 = 0;
        while (i < slz1_codec.diag_n_memory_types and i < 32) : (i += 1) {
            const f = slz1_codec.diag_memory_type_flags[i];
            try w.print("    type[{d:2}] flags=0x{x:0>3} (", .{ i, f });
            if (f & 0x1 != 0) try w.writeAll("DEVICE_LOCAL ");
            if (f & 0x2 != 0) try w.writeAll("HOST_VISIBLE ");
            if (f & 0x4 != 0) try w.writeAll("HOST_COHERENT ");
            if (f & 0x8 != 0) try w.writeAll("HOST_CACHED ");
            if (f & 0x10 != 0) try w.writeAll("LAZILY_ALLOCATED ");
            try w.writeAll(")\n");
        }
        try w.print(
            "  dst_stage resolved: memoryTypeIndex={d} flags=0x{x:0>3}\n",
            .{ slz1_codec.diag_dst_stage_mt_index, slz1_codec.diag_dst_stage_mt_flags },
        );
        const sysmem_gbps = @as(f64, @floatFromInt(slz1_codec.diag_sysmem_memcpy_GBps_x100)) / 100.0;
        const stage_gbps = @as(f64, @floatFromInt(slz1_codec.diag_bar_like_memcpy_GBps_x100)) / 100.0;
        try w.print(
            "  sysmem→sysmem @memcpy: {d:.2} GB/s    dst_stage.mapped→sysmem @memcpy: {d:.2} GB/s\n",
            .{ sysmem_gbps, stage_gbps },
        );
    }

    // Median across the runs. Slice the e2e and d2d arrays
    // separately — independent medians give the same shape as the
    // CUDA bench's output.
    const e2e_med = medianU64(dec_e2e_ns);
    const e2e_med_ms = @as(f64, @floatFromInt(e2e_med)) / 1_000_000.0;
    // For d2d, drop any zero entries (could happen on a device that
    // reports timestampValidBits == 0; unlikely on supported tiers).
    var nz_d2d: std.ArrayList(u64) = .empty;
    defer nz_d2d.deinit(allocator);
    for (dec_d2d_ns) |x| {
        if (x > 0) try nz_d2d.append(allocator, x);
    }
    if (nz_d2d.items.len > 0) {
        const d2d_med = medianU64(nz_d2d.items);
        const d2d_med_ms = @as(f64, @floatFromInt(d2d_med)) / 1_000_000.0;
        try w.print(
            "  Decompress median: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n\n",
            .{ e2e_med_ms, mb * 1000.0 / e2e_med_ms, d2d_med_ms, mb * 1000.0 / d2d_med_ms },
        );
    } else {
        try w.print(
            "  Decompress median: {d:.0}ms ({d:.1} MB/s)\n\n",
            .{ e2e_med_ms, mb * 1000.0 / e2e_med_ms },
        );
    }

    // ── Round-trip verify. ──────────────────────────────────────
    if (std.mem.eql(u8, src, dec_buf[0..src.len])) {
        try w.writeAll("Round-trip: PASS\n");
    } else {
        var first_fail: usize = 0;
        var fail_count: usize = 0;
        for (0..src.len) |bi| if (src[bi] != dec_buf[bi]) {
            if (fail_count == 0) first_fail = bi;
            fail_count += 1;
        };
        try w.print("Round-trip: FAIL  first_diff={d} total_diffs={d}\n", .{ first_fail, fail_count });
        try w.flush();
        std.process.exit(1);
    }
}

/// Decompress-only bench. Takes a pre-compressed .slz file and runs
/// `runs` timed decompresses against it (+ 1 warmup). Mirrors CUDA's
/// `streamlz -db` shape so nsys traces can be compared apples-to-apples
/// without the encoder's allocations polluting the per-API counts.
fn runDecompressBench(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: Args) !void {
    const in_path = args.input orelse {
        try w.writeAll("error: -db requires an input file (.slz)\n");
        return error.NoInput;
    };
    const runs: u32 = args.runs;

    const compressed = readFileAll(allocator, io, in_path) catch |err| {
        try w.print("error: cannot read '{s}': {s}\n", .{ in_path, @errorName(err) });
        return err;
    };
    defer allocator.free(compressed);
    if (compressed.len == 0) {
        try w.writeAll("error: input file is empty\n");
        return error.EmptyInput;
    }

    // Peek at the frame header for content_size so we can size dec_buf.
    const hdr = frame.parseHeader(compressed) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        return err;
    };
    const content_size: usize = if (hdr.content_size) |cs| @intCast(cs) else {
        try w.writeAll("error: frame missing content_size (streaming mode unsupported)\n");
        return error.NoContentSize;
    };

    try ensureInitWithSelector(w, args);
    defer driver.deinit();
    try printBoundDevice(w);

    const mb: f64 = @as(f64, @floatFromInt(content_size)) / (1024.0 * 1024.0);
    try w.print("Input: {s} ({d} bytes compressed, {d} bytes decompressed, {d:.2} MB)\n", .{
        in_path, compressed.len, content_size, mb,
    });

    // Page-align dec_buf so the SLZ1 decoder takes the
    // VK_EXT_external_memory_host fast path.
    const dec_buf_alloc_size = ((content_size + VK_SAFE_SPACE + 4095) & ~@as(usize, 4095));
    const dec_buf = try allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(4096), dec_buf_alloc_size);
    defer allocator.free(dec_buf);

    // Warm-up decompress.
    _ = try slz1_codec.decodeSlz1ToBytes(&driver.g_default, io, allocator, compressed, dec_buf);

    const dec_e2e_ns = try allocator.alloc(u64, runs);
    defer allocator.free(dec_e2e_ns);
    const dec_d2d_ns = try allocator.alloc(u64, runs);
    defer allocator.free(dec_d2d_ns);

    var r: u32 = 0;
    while (r < runs) : (r += 1) {
        l1_codec.last_decode_dispatch_ns = 0;
        const t0 = qpcNow();
        _ = try slz1_codec.decodeSlz1ToBytes(&driver.g_default, io, allocator, compressed, dec_buf);
        const t1 = qpcNow();
        dec_e2e_ns[r] = qpcNs(t0, t1);
        dec_d2d_ns[r] = l1_codec.last_decode_dispatch_ns;
        const e2e_ms = @as(f64, @floatFromInt(dec_e2e_ns[r])) / 1_000_000.0;
        const d2d_ms = @as(f64, @floatFromInt(dec_d2d_ns[r])) / 1_000_000.0;
        if (dec_d2d_ns[r] > 0) {
            try w.print(
                "  Decompress run {d}: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n",
                .{ r + 1, e2e_ms, mb * 1000.0 / e2e_ms, d2d_ms, mb * 1000.0 / d2d_ms },
            );
        } else {
            try w.print(
                "  Decompress run {d}: {d:.0}ms ({d:.1} MB/s)\n",
                .{ r + 1, e2e_ms, mb * 1000.0 / e2e_ms },
            );
        }
    }

    const e2e_med = medianU64(dec_e2e_ns);
    const e2e_med_ms = @as(f64, @floatFromInt(e2e_med)) / 1_000_000.0;
    var nz_d2d: std.ArrayList(u64) = .empty;
    defer nz_d2d.deinit(allocator);
    for (dec_d2d_ns) |x| {
        if (x > 0) try nz_d2d.append(allocator, x);
    }
    if (nz_d2d.items.len > 0) {
        const d2d_med = medianU64(nz_d2d.items);
        const d2d_med_ms = @as(f64, @floatFromInt(d2d_med)) / 1_000_000.0;
        try w.print(
            "  Decompress median: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n",
            .{ e2e_med_ms, mb * 1000.0 / e2e_med_ms, d2d_med_ms, mb * 1000.0 / d2d_med_ms },
        );
    } else {
        try w.print(
            "  Decompress median: {d:.0}ms ({d:.1} MB/s)\n",
            .{ e2e_med_ms, mb * 1000.0 / e2e_med_ms },
        );
    }
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
        .bench => try runBench(allocator, io, w, args),
        .decompress_bench => try runDecompressBench(allocator, io, w, args),
        .probe => try runProbe(w),
    }
}
