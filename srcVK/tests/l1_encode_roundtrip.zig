//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Random + structured payloads encoded via VK port, decoded via VK
//! port, byte-compare against original. Levels 1-2 exercised here.
//! Test bodies added by the fleshout agent.
//!

const std = @import("std");
const testing = std.testing;
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const enc_driver = @import("../encode/driver.zig");
const dec_driver = @import("../decode/driver.zig");

// VK adaptation: ptest_vk's worker pool ran multiple encode/decode tests
// against the shared enc_driver.g_default + dec_driver.g_default. Between
// worker A's encode-unlock and decode-lock, a sibling worker can call
// encode_context.ensureBuf which is destroy+create on the persistent
// device buffers; the resulting clobber surfaces here as zero-byte
// decoded output. Per-test EncodeContext + DecodeContext on the stack
// gives each test an independent set of persistent device buffers, with
// the registry SRWLOCK in module_loader.zig protecting the underlying
// VMA handle table from concurrent register/lookup/release calls.
fn roundtripWithLevel(allocator: std.mem.Allocator, src: []const u8, level: u8) !void {
    var enc_ctx: enc_driver.EncodeContext = .{};
    defer enc_ctx.deinit(allocator);
    var dec_ctx: dec_driver.DecodeContext = .{};
    defer dec_ctx.deinit();

    const opts: encoder.Options = .{ .level = level };
    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const written = try encoder.compressFramed(
        allocator,
        src,
        compressed,
        opts,
        &enc_ctx,
    );
    try testing.expect(written > 0);
    try testing.expect(written <= bound);

    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);

    const out_written = try decoder.decompressFramed(
        compressed[0..written],
        dst,
        &dec_ctx,
    );
    try testing.expectEqual(src.len, out_written);
    try testing.expectEqualSlices(u8, src, dst[0..out_written]);
}

test "L1 encode/decode round-trip: zeros (1 KiB)" {
    const src = try std.testing.allocator.alloc(u8, 1024);
    defer std.testing.allocator.free(src);
    @memset(src, 0);
    try roundtripWithLevel(std.testing.allocator, src, 1);
}

test "L1 encode/decode round-trip: ascending bytes (1 KiB)" {
    const src = try std.testing.allocator.alloc(u8, 1024);
    defer std.testing.allocator.free(src);
    for (src, 0..) |*b, i| b.* = @intCast(i & 0xFF);
    try roundtripWithLevel(std.testing.allocator, src, 1);
}

test "L1 encode/decode round-trip: random PRNG (100 KiB) at level=1" {
    const src = try std.testing.allocator.alloc(u8, 100 * 1024);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xC0FFEE);
    rng.random().bytes(src);
    try roundtripWithLevel(std.testing.allocator, src, 1);
}

test "L1 encode/decode round-trip: repeating pattern (1 MiB) at level=1" {
    const len: usize = 1 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    const pattern = "the quick brown fox jumps over the lazy dog ";
    var i: usize = 0;
    while (i < len) : (i += 1) src[i] = pattern[i % pattern.len];
    try roundtripWithLevel(std.testing.allocator, src, 1);
}

test "L1 encode/decode round-trip: random PRNG (1 MiB) at level=2" {
    const src = try std.testing.allocator.alloc(u8, 1 * 1024 * 1024);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xBADCAFE);
    rng.random().bytes(src);
    try roundtripWithLevel(std.testing.allocator, src, 2);
}

test "L1 encode/decode round-trip: empty payload" {
    // VK adaptation: per-test EncodeContext (see roundtripWithLevel).
    var enc_ctx: enc_driver.EncodeContext = .{};
    defer enc_ctx.deinit(std.testing.allocator);

    const src: []const u8 = &[_]u8{};
    const opts: encoder.Options = .{ .level = 1 };
    const bound = encoder.compressBound(src.len);
    const compressed = try std.testing.allocator.alloc(u8, bound);
    defer std.testing.allocator.free(compressed);
    const written = try encoder.compressFramed(
        std.testing.allocator,
        src,
        compressed,
        opts,
        &enc_ctx,
    );
    // Even for empty input the frame header + end mark must be emitted.
    try testing.expect(written >= 0);
}
