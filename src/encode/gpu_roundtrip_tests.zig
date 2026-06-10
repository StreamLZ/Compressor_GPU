//! GPU encode + decode roundtrip integration tests.
//!
//! The parallel test runner can fire multiple tests onto the same CUDA
//! context concurrently, which is unsafe — we serialize the entire GPU
//! payload behind a single test function. Each shape runs the
//! compress → decompress → byte-compare cycle and reports the first
//! failure with a label. The whole test skips (not fails) when CUDA is
//! unavailable at test time.

const std = @import("std");
const testing = std.testing;
const encoder = @import("streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_encoder = @import("driver.zig");
const gpu_driver = @import("../decode/driver.zig");

// 2026-06-10: the module_loader init() race used to make all but the
// first GPU test fn skip — which accidentally serialized GPU work.
// With init properly locked, every GPU test fn RUNS, and the parallel
// runner would fire them concurrently onto the shared g_default
// encode/decode contexts (persistent-buffer races → bogus
// DestinationTooSmall and worse). Serialize every GPU test fn body
// behind one process-wide lock; host-only tests stay parallel.
const SRWLOCK = extern struct { ptr: ?*anyopaque = null };
extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
var g_gpu_test_lock: SRWLOCK = .{};

pub fn lockGpuTests() void {
    AcquireSRWLockExclusive(&g_gpu_test_lock);
}

pub fn unlockGpuTests() void {
    ReleaseSRWLockExclusive(&g_gpu_test_lock);
}

const Case = struct {
    label: []const u8,
    bytes: []const u8,
    level: u8,
};

fn roundtripOne(allocator: std.mem.Allocator, case: Case) !void {
    const bound = encoder.compressBound(case.bytes.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);

    const n = encoder.compressFramed(
        allocator,
        case.bytes,
        compressed,
        .{ .level = case.level },
        &gpu_encoder.g_default,
    ) catch |err| {
        std.debug.print("compress failed for {s}: {s}\n", .{ case.label, @errorName(err) });
        return err;
    };
    try testing.expect(n > 0);
    try testing.expect(n <= bound);

    const dst = try allocator.alloc(u8, case.bytes.len + decoder.safe_space);
    defer allocator.free(dst);
    const written = decoder.decompressFramed(compressed[0..n], dst, &gpu_driver.g_default) catch |err| {
        std.debug.print("decompress failed for {s}: {s}\n", .{ case.label, @errorName(err) });
        return err;
    };
    if (written != case.bytes.len) {
        std.debug.print("size mismatch for {s}: expected {d}, got {d}\n", .{ case.label, case.bytes.len, written });
        return error.TestUnexpectedResult;
    }
    if (!std.mem.eql(u8, case.bytes, dst[0..written])) {
        var first_diff: usize = 0;
        while (first_diff < case.bytes.len and case.bytes[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        std.debug.print("byte mismatch for {s} at offset {d}\n", .{ case.label, first_diff });
        return error.TestUnexpectedResult;
    }
}

test "GPU roundtrip: every shape and level" {
    lockGpuTests();
    defer unlockGpuTests();
    const allocator = testing.allocator;

    if (!gpu_encoder.isAvailable()) {
        std.debug.print("(skipping GPU tests — CUDA not available)\n", .{});
        return error.SkipZigTest;
    }
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;

    // Bind the CUDA context to whichever runner thread picked this up.
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;

    // Generate the per-shape payloads once, reuse across levels.
    const tiny = "Hello, world!\n";

    const small_repeating = try allocator.alloc(u8, 4096);
    defer allocator.free(small_repeating);
    for (small_repeating, 0..) |*b, i| b.* = @intCast('A' + (i % 26));

    const medium_text = try allocator.alloc(u8, 100 * 1024);
    defer allocator.free(medium_text);
    {
        const pattern = "The quick brown fox jumps over a lazy dog. ";
        var i: usize = 0;
        while (i < medium_text.len) : (i += 1) medium_text[i] = pattern[i % pattern.len];
    }

    const sub_chunk_boundary = try allocator.alloc(u8, 128 * 1024 + 1);
    defer allocator.free(sub_chunk_boundary);
    for (sub_chunk_boundary, 0..) |*b, i| b.* = @intCast('a' + (i % 7));

    const incompressible = try allocator.alloc(u8, 32 * 1024);
    defer allocator.free(incompressible);
    {
        var state: u32 = 0xC0FFEE;
        for (incompressible) |*b| {
            state = state *% 1103515245 +% 12345;
            b.* = @intCast((state >> 16) & 0xFF);
        }
    }

    const cases = [_]Case{
        .{ .label = "tiny L1", .bytes = tiny, .level = 1 },
        .{ .label = "tiny L3", .bytes = tiny, .level = 3 },
        .{ .label = "tiny L5", .bytes = tiny, .level = 5 },
        .{ .label = "small repeating L1", .bytes = small_repeating, .level = 1 },
        .{ .label = "small repeating L3", .bytes = small_repeating, .level = 3 },
        .{ .label = "medium text L1", .bytes = medium_text, .level = 1 },
        .{ .label = "medium text L2", .bytes = medium_text, .level = 2 },
        .{ .label = "medium text L3", .bytes = medium_text, .level = 3 },
        .{ .label = "medium text L4", .bytes = medium_text, .level = 4 },
        .{ .label = "medium text L5", .bytes = medium_text, .level = 5 },
        .{ .label = "sub-chunk boundary L1", .bytes = sub_chunk_boundary, .level = 1 },
        .{ .label = "sub-chunk boundary L5", .bytes = sub_chunk_boundary, .level = 5 },
        .{ .label = "incompressible L1", .bytes = incompressible, .level = 1 },
        .{ .label = "incompressible L5", .bytes = incompressible, .level = 5 },
    };

    for (cases) |c| try roundtripOne(allocator, c);
}

test "GPU roundtrip: empty input at every level" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    var level: u8 = 1;
    while (level <= 5) : (level += 1) {
        try roundtripOne(allocator, .{ .label = "empty", .bytes = &.{}, .level = level });
    }
}

test "compressBound is a strict upper bound on real frames" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const sizes = [_]usize{ 0, 1, 1024, 4096, 64 * 1024, 256 * 1024, 1024 * 1024 };
    for (sizes) |sz| {
        const buf = try allocator.alloc(u8, sz);
        defer allocator.free(buf);
        for (buf, 0..) |*b, i| b.* = @intCast(i & 0xFF);
        const bound = encoder.compressBound(sz);
        const dst = try allocator.alloc(u8, bound);
        defer allocator.free(dst);
        const n = try encoder.compressFramed(allocator, buf, dst, .{ .level = 3 }, &gpu_encoder.g_default);
        try testing.expect(n <= bound);
    }
}

test "compressFramed rejects level outside 1..5" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    var dst: [4096]u8 = undefined;
    const src = "hello";
    try testing.expectError(error.BadLevel, encoder.compressFramed(allocator, src, &dst, .{ .level = 0 }, &gpu_encoder.g_default));
    try testing.expectError(error.BadLevel, encoder.compressFramed(allocator, src, &dst, .{ .level = 6 }, &gpu_encoder.g_default));
    try testing.expectError(error.BadLevel, encoder.compressFramed(allocator, src, &dst, .{ .level = 12 }, &gpu_encoder.g_default));
}

test "compressFramed rejects undersized destination" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    var dst: [4]u8 = undefined;
    const src = "hello world, this needs more than 4 bytes of output space";
    try testing.expectError(
        error.DestinationTooSmall,
        encoder.compressFramed(allocator, src, &dst, .{ .level = 1 }, &gpu_encoder.g_default),
    );
}
