//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Unit tests for srcVK/encode/streamlz_encoder.zig: compressBound math,
//! Options validation, writeUncompressedFrame path for tiny inputs. No
//! GPU. Test bodies added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const testing = std.testing;
const encoder = @import("../encode/streamlz_encoder.zig");
const frame = @import("../format/frame_format.zig");
const constants = @import("../format/streamlz_constants.zig");

// ── compressBound ──────────────────────────────────────────────────────

test "compressBound: empty input still includes frame header + end mark" {
    const b = encoder.compressBound(0);
    // Header + end mark + safety margin — must always be > 0.
    try testing.expect(b >= frame.max_header_size + 4);
}

test "compressBound: 1-byte input bound is strictly greater than input + frame header" {
    const b = encoder.compressBound(1);
    try testing.expect(b > 1 + frame.max_header_size);
}

test "compressBound: bound is monotone non-decreasing as input grows" {
    const a = encoder.compressBound(1024);
    const c = encoder.compressBound(64 * 1024);
    const d = encoder.compressBound(4 * 1024 * 1024);
    try testing.expect(a <= c);
    try testing.expect(c <= d);
}

test "compressBound: bound for 1 MiB exceeds input by reasonable margin" {
    const src_len: usize = 1 * 1024 * 1024;
    const b = encoder.compressBound(src_len);
    // The formula adds: frame header + end mark + chunk_count * 14 +
    // sub_chunks * 270 + src_len + 64 + sc_prefix_upper_bound.
    // At minimum bound > src_len + 64.
    try testing.expect(b > src_len + 64);
    // Sanity ceiling — should not balloon past 2x for 1 MiB.
    try testing.expect(b < src_len * 2);
}

// ── Options validation (via compressFramedWithIo guard rails) ─────────

test "compressFramedWithIo: level=0 returns BadLevel" {
    var dst: [64]u8 = undefined;
    const src = [_]u8{0xAA} ** 16;
    // EncodeContext pointer is never dereferenced on the level-check path,
    // so passing undefined is safe; the BadLevel return fires first.
    const opts: encoder.Options = .{ .level = 0 };
    var ctx_undef: @import("../encode/driver.zig").EncodeContext = undefined;
    const err = encoder.compressFramedWithIo(
        std.testing.allocator,
        std.Io.failing,
        &src,
        &dst,
        opts,
        &ctx_undef,
    );
    try testing.expectError(error.BadLevel, err);
}

test "compressFramedWithIo: level=6 returns BadLevel" {
    var dst: [64]u8 = undefined;
    const src = [_]u8{0xAA} ** 16;
    const opts: encoder.Options = .{ .level = 6 };
    var ctx_undef: @import("../encode/driver.zig").EncodeContext = undefined;
    const err = encoder.compressFramedWithIo(
        std.testing.allocator,
        std.Io.failing,
        &src,
        &dst,
        opts,
        &ctx_undef,
    );
    try testing.expectError(error.BadLevel, err);
}

test "compressFramedWithIo: dst too small returns DestinationTooSmall" {
    // dst smaller than compressBound(src.len) -> DestinationTooSmall before
    // the GPU path is touched.
    const src = [_]u8{0xAA} ** 4096;
    var dst: [4]u8 = undefined; // way too small
    const opts: encoder.Options = .{ .level = 1 };
    var ctx_undef: @import("../encode/driver.zig").EncodeContext = undefined;
    const err = encoder.compressFramedWithIo(
        std.testing.allocator,
        std.Io.failing,
        &src,
        &dst,
        opts,
        &ctx_undef,
    );
    try testing.expectError(error.DestinationTooSmall, err);
}

// ── frame.writeHeader Options validation (mirrors encoder Options gates) ─

test "writeHeader: level=0 returns BadLevel" {
    var dst: [frame.max_header_size]u8 = undefined;
    const err = frame.writeHeader(&dst, .{
        .codec = .fast,
        .level = 0,
        .block_size = @intCast(constants.chunk_size),
    });
    try testing.expectError(error.BadLevel, err);
}

test "writeHeader: block_size=0 returns BadBlockSize" {
    var dst: [frame.max_header_size]u8 = undefined;
    const err = frame.writeHeader(&dst, .{
        .codec = .fast,
        .level = 1,
        .block_size = 0,
    });
    try testing.expectError(error.BadBlockSize, err);
}

test "writeHeader: non-power-of-2 block_size returns BadBlockSize" {
    var dst: [frame.max_header_size]u8 = undefined;
    const err = frame.writeHeader(&dst, .{
        .codec = .fast,
        .level = 1,
        .block_size = 65537, // not a power of 2
    });
    try testing.expectError(error.BadBlockSize, err);
}

test "writeHeader: sc_group_size <= 0 returns BadScGroupSize" {
    var dst: [frame.max_header_size]u8 = undefined;
    const err = frame.writeHeader(&dst, .{
        .codec = .fast,
        .level = 1,
        .block_size = @intCast(constants.chunk_size),
        .sc_group_size = 0.0,
    });
    try testing.expectError(error.BadScGroupSize, err);
}

test "writeHeader: valid options writes the documented byte count" {
    var dst: [frame.max_header_size]u8 = undefined;
    const written = try frame.writeHeader(&dst, .{
        .codec = .fast,
        .level = 1,
        .block_size = @intCast(constants.chunk_size),
        .sc_group_size = 4.0,
    });
    // Without content_size and without dictionary_id, writeHeader emits
    // 14 bytes (== min_header_size).
    try testing.expectEqual(@as(usize, frame.min_header_size), written);
}
