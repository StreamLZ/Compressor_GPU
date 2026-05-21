const std = @import("std");
const slz = @import("streamlz");

/// Fuzz harness for the StreamLZ decompressor.
///
/// Usage:  fuzz-decompress <input-file>
///
/// Reads the file as compressed input, calls decompressFramed with a
/// fixed-size output buffer, and swallows decode errors (expected for
/// fuzz-generated input). Memory safety violations will panic, which
/// ReleaseSafe catches via bounds/overflow checks.
///
/// Works with AFL (@@), honggfuzz, or manual corpus files.
pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(allocator);
    defer args_it.deinit();
    _ = args_it.next(); // skip argv[0]
    const path = args_it.next() orelse return;

    const input = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, @enumFromInt(1 << 24)) catch return;
    defer allocator.free(input);

    var dst: [1 << 24]u8 = undefined;
    _ = slz.decompressFramed(input, &dst, slz.default_decode_context) catch return;
}
