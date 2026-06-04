//! 1:1 port of src/cli/util.zig.
//!
//! Shared CLI plumbing: argument parsing, output-path derivation, file
//! I/O, formatting helpers, process-memory reporting. No GPU calls.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const version = @import("../version.zig");
const mmap = @import("../mmap.zig");

/// CUDA reference: src/cli/util.zig:12. Top-level CLI mode.
pub const Mode = enum { compress, decompress, bench, bench_decompress, bench_all, info, version, help };

/// CUDA reference: src/cli/util.zig:14-22. Parsed CLI arguments.
pub const Args = struct {
    mode: Mode = .compress,
    level: u8 = 1,
    runs: ?u32 = null,
    input: ?[]const u8 = null,
    output: ?[]const u8 = null,
    report_mem: bool = false,
    sc_group: ?f32 = null,
};

const MemInfo = struct { peak_rss_mb: f64, commit_mb: f64 };

/// CUDA reference: src/cli/util.zig:24-85. Parse argv into an Args value.
pub fn parseArgs(raw: []const []const u8, w: *std.Io.Writer) Args {
    _ = raw;
    _ = w;
    return .{};
}

/// CUDA reference: src/cli/util.zig:87-91. Print msg and abort.
pub fn die(w: *std.Io.Writer, msg: []const u8) noreturn {
    _ = w;
    _ = msg;
    @panic("die");
}

/// CUDA reference: src/cli/util.zig:93-98. Print "streamlz vX.Y.Z".
pub fn printVersion(w: *std.Io.Writer) !void {
    _ = w;
    return error.NotYetPorted;
}

/// CUDA reference: src/cli/util.zig:100-126. Print the CLI help text.
pub fn printUsage(w: *std.Io.Writer) !void {
    _ = w;
    return error.NotYetPorted;
}

/// CUDA reference: src/cli/util.zig:127-133. Derive <input>.slz from
/// the compression input path.
pub fn deriveCompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    _ = allocator;
    _ = input;
    return error.NotYetPorted;
}

/// CUDA reference: src/cli/util.zig:134-147. Derive the decompressed
/// output path (strip .slz suffix).
pub fn deriveDecompressOutput(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    _ = allocator;
    _ = input;
    return error.NotYetPorted;
}

/// CUDA reference: src/cli/util.zig:148-156. Read entire file into a
/// freshly-allocated buffer.
pub fn readFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, w: *std.Io.Writer) []const u8 {
    _ = allocator;
    _ = io;
    _ = path;
    _ = w;
    return &.{};
}

/// CUDA reference: src/cli/util.zig:157-165. Pull the input path out of
/// args or die.
pub fn requireInput(args: Args, w: *std.Io.Writer) []const u8 {
    _ = args;
    _ = w;
    return &.{};
}

/// CUDA reference: src/cli/util.zig:166-177. Reject level == 0 or > 9.
pub fn checkLevel(level: u8, w: *std.Io.Writer) void {
    _ = level;
    _ = w;
}

/// CUDA reference: src/cli/util.zig:178-192. Statistical helper for bench
/// timings.
pub fn medianOrMean(times: []const u64) u64 {
    _ = times;
    return 0;
}

/// CUDA reference: src/cli/util.zig:193-220. Format byte counts with a
/// unit suffix (KB/MB/GB).
pub fn fmtBytes(buf: []u8, value: usize) []const u8 {
    _ = buf;
    _ = value;
    return &.{};
}

/// CUDA reference: src/cli/util.zig:221-229. Format MB/s throughput.
pub fn fmtMbps(buf: []u8, value: f64) []const u8 {
    _ = buf;
    _ = value;
    return &.{};
}

/// CUDA reference: src/cli/util.zig:230-end. Query peak RSS + working set.
pub fn getMemInfo() MemInfo {
    return .{ .peak_rss_mb = 0, .commit_mb = 0 };
}
