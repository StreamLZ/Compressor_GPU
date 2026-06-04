//! 1:1 port of src/cli/util.zig.
//!
//! Shared CLI plumbing: argument parsing, output-path derivation, file
//! I/O, formatting helpers, process-memory reporting. No GPU calls.
//!
//! VK adaptation: extends the CUDA-side `Args` with `device_spec`
//! (`--device <N|name>`) so the operator can target a specific
//! Vulkan physical device. Every other field, flag, and helper mirrors
//! the CUDA CLI verbatim per audit Section 3.

const std = @import("std");
const builtin = @import("builtin");
const version_mod = @import("../version.zig");

const version_string = version_mod.string;

/// CUDA reference: src/cli/util.zig:12. Top-level CLI mode.
pub const Mode = enum { compress, decompress, bench, bench_decompress, bench_all, info, version, help, probe };

/// CUDA reference: src/cli/util.zig:14-22. Parsed CLI arguments.
/// VK adaptation: adds `device_spec` (lifetime: argv) for `--device <N|name>`.
pub const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    runs: ?u32 = null,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    report_mem: bool = false,
    sc_group: ?f32 = null,
    /// VK adaptation: raw `--device <arg>` payload. Pure-digit strings
    /// are interpreted as `by_index`; otherwise as `by_name` (case-
    /// insensitive substring of vkPhysicalDeviceProperties.deviceName).
    /// Null means: consult SLZ_VK_DEVICE_INDEX, else default priority.
    device_spec: ?[]const u8 = null,
};

/// CUDA reference: src/cli/util.zig:24-65. Parse argv into an Args value.
pub fn parseArgs(raw: []const []const u8, w: *std.Io.Writer) Args {
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
        // VK adaptation: --probe enumerates Vulkan physical devices. CUDA
        // has no analogue (it hardcodes device 0); mirrors src_vulkan/cli_vk.zig.
        if (eql(arg, "--probe") or eql(arg, "-p")) { result.mode = .probe; continue; }
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
        // VK adaptation: --device <N|name> selects a Vulkan physical device.
        if (eql(arg, "--device")) {
            i += 1;
            result.device_spec = expect(raw, i, "--device", w);
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

/// CUDA reference: src/cli/util.zig:87-91. Print msg and abort.
pub fn die(w: *std.Io.Writer, msg: []const u8) noreturn {
    w.writeAll(msg) catch {};
    w.flush() catch {};
    std.process.exit(2);
}

/// CUDA reference: src/cli/util.zig:93-98. Print version line.
pub fn printVersion(w: *std.Io.Writer) !void {
    try w.print("streamlz_vk {s} (Vulkan, Zig {f}, {s}-{s})\n", .{
        version_string, builtin.zig_version,
        @tagName(builtin.target.cpu.arch), @tagName(builtin.target.os.tag),
    });
}

/// CUDA reference: src/cli/util.zig:100-123. Print the CLI help text.
pub fn printUsage(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\Usage: streamlz_vk [options] <input-file>
        \\
        \\Mode (default: -c):
        \\  -c              Compress
        \\  -d              Decompress
        \\  -b              Compress + decompress + round-trip verify
        \\  -db             Decompress benchmark on a .slz file
        \\  -ba             Sweep levels L1-L5: compress + decompress ratio/speed table (-l ignored)
        \\  -i              Dump frame header + block list
        \\  --probe, -p     List all Vulkan physical devices and exit
        \\
        \\Options:
        \\  -l <1..5>       Compression level (default: 1)
        \\  -r <runs>       Benchmark runs (default: 3 for -b, 10 for -db)
        \\  -o <file>       Output path
        \\  --sc <float>    sc_group_size override (0.25 = 64 KB sub-chunks)
        \\  --device <N|name>  Select Vulkan physical device by zero-based index
        \\                  or case-insensitive substring of deviceName.
        \\                  Env fallback: SLZ_VK_DEVICE_INDEX=<N>.
        \\  -gpu            Accepted, no-op (GPU is the only backend)
        \\  -mem            Print peak process memory at exit
        \\  -V, --version   Print version
        \\  -h, --help      Print help
        \\
    );
}

// ── Output path derivation ─────────────────────────────────────────────

/// CUDA reference: src/cli/util.zig:127-133. Append ".slz" to input.
pub fn deriveCompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    const result = try allocator.alloc(u8, input.len + 4);
    @memcpy(result[0..input.len], input);
    @memcpy(result[input.len..][0..4], ".slz");
    return result;
}

/// CUDA reference: src/cli/util.zig:134-144. Strip ".slz" suffix or
/// append ".dec".
pub fn deriveDecompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
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

// ── File I/O helpers ───────────────────────────────────────────────────

/// CUDA reference: src/cli/util.zig:148-155. Read entire file into a
/// freshly-allocated buffer.
pub fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, w: *std.Io.Writer) []const u8 {
    const max_bytes: usize = 1 << 31;
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, @enumFromInt(max_bytes)) catch |err| {
        w.print("error: cannot read '{s}': {s}\n", .{ path, @errorName(err) }) catch {};
        w.flush() catch {};
        std.process.exit(1);
    };
}

/// CUDA reference: src/cli/util.zig:157-164. Pull the input path out of
/// args or die.
pub fn requireInput(args: Args, w: *std.Io.Writer) []const u8 {
    return args.input orelse {
        w.writeAll("error: no input file specified\n\n") catch {};
        printUsage(w) catch {};
        w.flush() catch {};
        std.process.exit(2);
    };
}

/// CUDA reference: src/cli/util.zig:166-172. Reject levels outside 1..5.
pub fn checkLevel(level: u8, w: *std.Io.Writer) void {
    if (level < 1 or level > 5) {
        w.print("error: level must be 1..5 (got {d})\n", .{level}) catch {};
        w.flush() catch {};
        std.process.exit(2);
    }
}

// ── Bench helpers ──────────────────────────────────────────────────────

/// CUDA reference: src/cli/util.zig:178-191. Median for short slices;
/// mean for long ones (avoids an O(n log n) sort blowing the small fixed
/// stack buffer).
pub fn medianOrMean(times: []const u64) u64 {
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

/// CUDA reference: src/cli/util.zig:193-219. Format byte counts with
/// thousand-separator commas.
pub fn fmtBytes(buf: []u8, value: usize) []const u8 {
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

/// CUDA reference: src/cli/util.zig:221-224. Format MB/s throughput.
pub fn fmtMbps(buf: []u8, value: f64) []const u8 {
    if (value >= 100.0) return std.fmt.bufPrint(buf, "{d:>.0}", .{value}) catch "?";
    return std.fmt.bufPrint(buf, "{d:>.1}", .{value}) catch "?";
}

// ── Process memory ─────────────────────────────────────────────────────

pub const MemInfo = struct { peak_rss_mb: f64, commit_mb: f64 };

/// CUDA reference: src/cli/util.zig:230-272. Query peak RSS + working set.
pub fn getMemInfo() MemInfo {
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

// ── Device selection (VK port carve-out) ──────────────────────────────

/// VK adaptation: how a CLI invocation wants to pick the Vulkan physical
/// device. Mirrors src_vulkan/device.zig's DeviceSelector shape.
pub const DeviceSelector = union(enum) {
    /// No spec: consult SLZ_VK_DEVICE_INDEX env var; if unset, fall back
    /// to the priority ordering (discrete > integrated > virtual > cpu > other).
    default,
    /// Zero-based index into the vkEnumeratePhysicalDevices order.
    by_index: u32,
    /// Case-insensitive substring of vkPhysicalDeviceProperties.deviceName.
    by_name: []const u8,
};

/// VK adaptation: parse `--device <spec>` into a DeviceSelector.
/// All-digit strings → `by_index`; everything else (incl. "0x...") → `by_name`.
pub fn selectorFromSpec(spec: ?[]const u8) DeviceSelector {
    const s = spec orelse return .default;
    if (s.len == 0) return .default;
    var all_digits = true;
    for (s) |c| {
        if (c < '0' or c > '9') { all_digits = false; break; }
    }
    if (all_digits) {
        const n = std.fmt.parseInt(u32, s, 10) catch return .{ .by_name = s };
        return .{ .by_index = n };
    }
    return .{ .by_name = s };
}

/// VK adaptation: pull SLZ_VK_DEVICE_INDEX out of the environment.
/// Returns null when the var is unset or unparseable.
pub fn envDeviceIndex() ?u32 {
    const raw = std.c.getenv("SLZ_VK_DEVICE_INDEX") orelse return null;
    const s = std.mem.span(raw);
    if (s.len == 0) return null;
    return std.fmt.parseInt(u32, s, 10) catch null;
}

/// VK adaptation: device priority score; higher beats lower. Mirrors
/// CUDA's implicit "discrete > integrated > virtual > cpu > other"
/// ordering documented in src_vulkan/probe.zig.
pub fn devicePriorityScore(device_type: c_int) u32 {
    // VkPhysicalDeviceType enum values per the Vulkan spec:
    //   OTHER=0, INTEGRATED_GPU=1, DISCRETE_GPU=2, VIRTUAL_GPU=3, CPU=4
    return switch (device_type) {
        2 => 4, // DISCRETE_GPU
        1 => 3, // INTEGRATED_GPU
        3 => 2, // VIRTUAL_GPU
        4 => 1, // CPU
        else => 0, // OTHER / unknown
    };
}

/// VK adaptation: lowercase a single ASCII byte for case-insensitive
/// deviceName matching. Used by the `--device <name>` substring code.
pub fn asciiToLower(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

/// VK adaptation: case-insensitive substring search over ASCII bytes.
/// True if `needle` appears anywhere in `haystack`. Empty needle matches.
pub fn asciiContainsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (asciiToLower(haystack[i + j]) != asciiToLower(needle[j])) { match = false; break; }
        }
        if (match) return true;
    }
    return false;
}

/// VK adaptation: null-terminated bytes → slice.
pub fn cstrLen(bytes: []const u8) []const u8 {
    for (bytes, 0..) |c, i| if (c == 0) return bytes[0..i];
    return bytes;
}
