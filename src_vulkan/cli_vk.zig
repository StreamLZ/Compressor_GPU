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
const vk = @import("vk_api.zig");
const instance_mod = @import("instance.zig");
const probe_mod = @import("probe.zig");
const device_mod = @import("device.zig");

const VK_SAFE_SPACE: usize = 64;

const Mode = enum { compress, decompress, version, help, probe };

const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    /// Raw `--device <arg>` payload (lifetime: argv). Parsed at run-time:
    /// pure-digit strings become `.by_index`; anything else becomes
    /// `.by_name` (case-insensitive substring match against deviceName).
    /// `null` means "no flag" → driver consults SLZ_VK_DEVICE_INDEX env
    /// var, else falls back to first compute-capable device.
    device_spec: ?[]const u8 = null,
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
        \\  -c                  Compress (default)
        \\  -d                  Decompress
        \\  -p, --probe         List all Vulkan physical devices and exit
        \\
        \\Options:
        \\  -l N                Compression level (only 1 supported in v1)
        \\  -o PATH             Output path
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
        .probe => try runProbe(w),
    }
}
