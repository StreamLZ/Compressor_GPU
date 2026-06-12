//! Shared CLI plumbing: argument parsing, error/exit helpers, output-
//! path derivation, file I/O, formatting helpers, and process memory
//! reporting. Imported by both `src/cli.zig` (the dispatcher) and the
//! per-mode handlers under `src/cli/`.

const std = @import("std");
const builtin = @import("builtin");
const version_mod = @import("../version.zig");

const version_string = version_mod.string;

pub const Mode = enum { compress, decompress, bench, bench_decompress, bench_all, info, version, help };

pub const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    runs: ?u32 = null,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    report_mem: bool = false,
    sc_group: ?f32 = null,
    checksum: bool = false,
    /// v4 #16: preset dictionary for compression - a registry name
    /// ("json", "html", ...), a numeric ID, or "auto" (select by the
    /// input file's extension). Decompression needs no flag: the
    /// frame header names its dictionary.
    dictionary: ?[]const u8 = null,
};

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
        // `-gpu` is accepted but ignored — kept to avoid breaking caller
        // scripts that learned it before the GPU became the only backend.
        if (eql(arg, "-gpu")) continue;
        if (eql(arg, "-l")) { i += 1; result.level = parseInt(u8, expect(raw, i, "-l", w), w, "-l"); continue; }
        if (eql(arg, "-r")) { i += 1; result.runs = parseInt(u32, expect(raw, i, "-r", w), w, "-r"); continue; }
        if (eql(arg, "-o")) { i += 1; result.output = expect(raw, i, "-o", w); continue; }
        if (eql(arg, "--checksum")) { result.checksum = true; continue; }
        if (eql(arg, "-D")) { i += 1; result.dictionary = expect(raw, i, "-D", w); continue; }
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

pub fn die(w: *std.Io.Writer, msg: []const u8) noreturn {
    w.writeAll(msg) catch {};
    w.flush() catch {};
    std.process.exit(2);
}

/// v4 #16: resolve the `-D` value to a dictionary ID for COMPRESSION.
/// Accepts, in order: a registry name, a numeric builtin ID, "auto"
/// (select by the input path's extension), or a path to a custom
/// dictionary file - the file's bytes are registered on `enc_ctx`
/// and the content-derived ID is returned. Exits with a message
/// naming the available dictionaries on failure.
pub fn resolveDictionary(
    allocator: std.mem.Allocator,
    io: std.Io,
    spec: ?[]const u8,
    input_path: []const u8,
    w: *std.Io.Writer,
    enc_ctx: *@import("../encode/driver.zig").EncodeContext,
) ?u32 {
    const dictionary = @import("../dict/dictionary.zig");
    const s = spec orelse return null;
    if (eql(s, "auto")) {
        if (dictionary.findByExtension(input_path)) |d| return d.id;
        return null;
    }
    if (dictionary.findByName(s)) |d| return d.id;
    if (std.fmt.parseInt(u32, s, 10) catch null) |id| {
        if (dictionary.findById(id)) |d| return d.id;
    }
    if (loadDictFile(allocator, io, s)) |bytes| {
        defer allocator.free(bytes);
        const id = enc_ctx.registerDict(allocator, bytes) catch
            die(w, "error: out of memory registering dictionary\n");
        w.print("dictionary: {s} ({d} bytes, id 0x{x:0>8})\n", .{ s, bytes.len, id }) catch {};
        return id;
    }
    w.print("error: unknown dictionary '{s}'; available:", .{s}) catch {};
    for (dictionary.builtin_dicts) |*d| w.print(" {s}", .{d.name}) catch {};
    w.writeAll(" auto, or a path to a dictionary file\n") catch {};
    w.flush() catch {};
    std.process.exit(2);
}

/// v4 #16: decode-side `-D` - supply a dictionary for the frame about
/// to be decoded and VERIFY it against the frame's dictionary ID
/// (mismatch exits naming both IDs, far clearer than the wrong-bytes
/// ChecksumMismatch a blind decode would produce). Builtins resolve
/// automatically from the frame header with no flag; without -D, a
/// frame needing a custom dictionary gets a message naming the ID to
/// go find.
pub fn supplyDecodeDictionary(
    allocator: std.mem.Allocator,
    io: std.Io,
    spec: ?[]const u8,
    frame_dict_id: ?u32,
    w: *std.Io.Writer,
    dec_ctx: *@import("../decode/driver.zig").DecodeContext,
) void {
    const dictionary = @import("../dict/dictionary.zig");
    const gpu_dec = @import("../decode/driver.zig");
    const s = spec orelse {
        if (frame_dict_id) |fid| {
            if (dictionary.findById(fid) == null) {
                w.print(
                    "error: frame requires dictionary 0x{x:0>8} (not a built-in); supply it with -D <file>\n",
                    .{fid},
                ) catch {};
                w.flush() catch {};
                std.process.exit(1);
            }
        }
        return;
    };
    const fid = frame_dict_id orelse {
        w.print("note: -D ignored - the frame carries no dictionary\n", .{}) catch {};
        return;
    };
    var supplied_id: ?u32 = null;
    if (dictionary.findByName(s)) |d| {
        supplied_id = d.id;
    } else if (std.fmt.parseInt(u32, s, 10) catch null) |nid| {
        if (dictionary.findById(nid)) |d| supplied_id = d.id;
    }
    if (supplied_id == null) {
        if (loadDictFile(allocator, io, s)) |bytes| {
            defer allocator.free(bytes);
            supplied_id = gpu_dec.registerDict(dec_ctx, allocator, bytes) catch
                die(w, "error: out of memory registering dictionary\n");
        } else {
            w.print("error: unknown dictionary '{s}' (not a built-in name/ID or readable file)\n", .{s}) catch {};
            w.flush() catch {};
            std.process.exit(2);
        }
    }
    if (supplied_id.? != fid) {
        w.print(
            "error: frame requires dictionary 0x{x:0>8}, supplied dictionary is 0x{x:0>8}\n",
            .{ fid, supplied_id.? },
        ) catch {};
        w.flush() catch {};
        std.process.exit(1);
    }
}

/// Load a custom dictionary file (raw bytes, as `dict_gate0` trains
/// them). Returns null when the path does not name a readable file.
/// Size cap: dictionaries beyond off16 reach are useless to the
/// match finder, so reject absurd files early.
pub fn loadDictFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ?[]u8 {
    const max_dict_file: usize = 16 * 1024 * 1024;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, @enumFromInt(max_dict_file)) catch return null;
    if (bytes.len == 0) {
        allocator.free(bytes);
        return null;
    }
    return bytes;
}

pub fn printVersion(w: *std.Io.Writer) !void {
    try w.print("streamlz {s} (GPU-only, Zig {f}, {s}-{s})\n", .{
        version_string, builtin.zig_version,
        @tagName(builtin.target.cpu.arch), @tagName(builtin.target.os.tag),
    });
}

pub fn printUsage(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\Usage: streamlz [options] <input-file>
        \\
        \\Mode (default: -c):
        \\  -c              Compress
        \\  -d              Decompress
        \\  -b              Compress + decompress + round-trip verify
        \\  -db             Decompress benchmark on a .slz file
        \\  -ba             Sweep levels L1-L5: compress + decompress ratio/speed table (-l ignored)
        \\  -i              Dump frame header + block list
        \\
        \\Options:
        \\  -l <1..5>       Compression level (default: 1)
        \\  -r <runs>       Benchmark runs (default: 3 for -b, 10 for -db)
        \\  -o <file>       Output path
        \\  -D <dict>       Preset dictionary: name (json, html, text, xml,
        \\                  css, js, general), numeric ID, or "auto" (by
        \\                  input extension). Decode reads the frame header.
        \\  --sc <float>    sc_group_size override (0.25 = 64 KB sub-chunks)
        \\  -gpu            Accepted, no-op (GPU is the only backend)
        \\  -mem            Print peak process memory at exit
        \\  -V, --version   Print version
        \\  -h, --help      Print help
        \\
    );
}

// ── Output path derivation ─────────────────────────────────────────────

pub fn deriveCompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    const result = try allocator.alloc(u8, input.len + 4);
    @memcpy(result[0..input.len], input);
    @memcpy(result[input.len..][0..4], ".slz");
    return result;
}

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

pub fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, w: *std.Io.Writer) []const u8 {
    const max_bytes: usize = 1 << 31;
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, @enumFromInt(max_bytes)) catch |err| {
        w.print("error: cannot read '{s}': {s}\n", .{ path, @errorName(err) }) catch {};
        w.flush() catch {};
        std.process.exit(1);
    };
}

pub fn requireInput(args: Args, w: *std.Io.Writer) []const u8 {
    return args.input orelse {
        w.writeAll("error: no input file specified\n\n") catch {};
        printUsage(w) catch {};
        w.flush() catch {};
        std.process.exit(2);
    };
}

pub fn checkLevel(level: u8, w: *std.Io.Writer) void {
    if (level < 1 or level > 5) {
        w.print("error: level must be 1..5 (got {d})\n", .{level}) catch {};
        w.flush() catch {};
        std.process.exit(2);
    }
}

// ── Bench helpers ──────────────────────────────────────────────────────

/// Median for short slices; mean for long ones (avoids an O(n log n)
/// sort blowing the small fixed stack buffer).
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

pub fn fmtMbps(buf: []u8, value: f64) []const u8 {
    if (value >= 100.0) return std.fmt.bufPrint(buf, "{d:>.0}", .{value}) catch "?";
    return std.fmt.bufPrint(buf, "{d:>.1}", .{value}) catch "?";
}

// ── Process memory ─────────────────────────────────────────────────────

const MemInfo = struct { peak_rss_mb: f64, commit_mb: f64 };

pub fn getMemInfo() MemInfo {
    const os = builtin.os.tag;
    if (os == .windows) {
        const PROCESS_MEMORY_COUNTERS = extern struct {
            // Win32 ABI mirror; field names match PSAPI's `PROCESS_MEMORY_COUNTERS`.
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
