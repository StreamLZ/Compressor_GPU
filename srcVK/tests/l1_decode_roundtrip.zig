//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Golden L1 frames (encoded via CUDA reference or a fresh VK encode)
//! decoded via VK port; byte-compare against original. Test bodies
//! added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const testing = std.testing;
const decoder = @import("../decode/streamlz_decoder.zig");
const driver = @import("../decode/driver.zig");
const frame = @import("../format/frame_format.zig");

const max_corpus_bytes: usize = 1 << 30; // 1 GiB cap

fn makeIo() std.Io.Threaded {
    return std.Io.Threaded.init(std.mem.Allocator.failing, .{});
}

fn readFileAlloc(allocator: std.mem.Allocator, io: std.Io, rel_path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(
        io,
        rel_path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    );
}

fn decodeAndCompare(allocator: std.mem.Allocator, golden_path: []const u8, original_path: []const u8) !void {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    const golden = readFileAlloc(allocator, io, golden_path) catch return error.SkipZigTest;
    defer allocator.free(golden);
    const original = readFileAlloc(allocator, io, original_path) catch return error.SkipZigTest;
    defer allocator.free(original);

    const hdr = try frame.parseHeader(golden);
    const want_size: usize = if (hdr.content_size) |cs| @intCast(cs) else original.len;
    try testing.expectEqual(original.len, want_size);

    const dst = try allocator.alloc(u8, want_size + 64);
    defer allocator.free(dst);
    @memset(dst, 0);

    const written = try decoder.decompressFramed(golden, dst, &driver.g_default);
    try testing.expectEqual(original.len, written);
    try testing.expectEqualSlices(u8, original, dst[0..written]);
}

test "L1 decode: small frame (web.txt ~4.5 MB) byte-equal round-trip" {
    try decodeAndCompare(
        std.testing.allocator,
        "tests/goldens/web.txt.L1.slz",
        "assets/web.txt",
    );
}

test "L1 decode: large frame (enwik8 ~100 MB) byte-equal round-trip" {
    try decodeAndCompare(
        std.testing.allocator,
        "tests/goldens/enwik8.txt.L1.slz",
        "assets/enwik8.txt",
    );
}

test "L1 decode: extra-large frame (silesia ~200 MB) byte-equal round-trip" {
    try decodeAndCompare(
        std.testing.allocator,
        "tests/goldens/silesia_all.tar.L1.slz",
        "assets/silesia_all.tar",
    );
}

test "L1 decode: empty src returns 0 written without touching dst" {
    var dst: [16]u8 = @splat(0xCD);
    const written = try decoder.decompressFramed(&[_]u8{}, &dst, &driver.g_default);
    try testing.expectEqual(@as(usize, 0), written);
    // dst must be untouched.
    for (dst) |b| try testing.expectEqual(@as(u8, 0xCD), b);
}

test "L1 decode: bad magic header returns BadFrame" {
    var bogus: [frame.min_header_size]u8 = @splat(0xAA);
    var dst: [64]u8 = undefined;
    const err = decoder.decompressFramed(&bogus, &dst, &driver.g_default);
    try testing.expectError(error.BadFrame, err);
}
