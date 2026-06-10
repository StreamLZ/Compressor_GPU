//! GPU regression tests backported from the srcVK suite (2026-06-10).
//!
//! Two coverage holes the VK port filled that the CUDA side never got:
//!
//! 1. A-023 batched LZ dispatch: the batched path only triggers
//!    organically when the per-chunk hash exceeds the cuMemGetInfo
//!    budget (~8 GB+ of demand), so CI never exercises it. The
//!    `g_force_batch_count_for_test` hook (mirroring VK's) drives the
//!    batched loop on a small input; output must be BYTE-IDENTICAL to
//!    the unbatched encode because every chunk re-initialises its own
//!    hash region at kernel entry.
//!
//! 2. Real-corpus roundtrips: the synthetic shapes in
//!    gpu_roundtrip_tests.zig passed for weeks on the VK port while
//!    real corpus data failed at byte 65544 — distinct content per
//!    64 KB block exercises entropy paths, block transitions, and
//!    recent-offset carry that repeating patterns cannot. Mirrors the
//!    srcVK l3_l4_encode_roundtrip corpus cases against assets/web.txt.
//!
//! Same conventions as gpu_roundtrip_tests.zig: every test fn skips
//! (not fails) without CUDA, binds the context to the runner thread,
//! and bundles its GPU cases serially inside one test fn.

const std = @import("std");
const testing = std.testing;
const encoder = @import("streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const encode_lz = @import("encode_lz.zig");
const gpu_encoder = @import("driver.zig");
const gpu_driver = @import("../decode/driver.zig");
const gpu_roundtrip = @import("gpu_roundtrip_tests.zig");

fn roundtripLabeled(allocator: std.mem.Allocator, label: []const u8, bytes: []const u8, level: u8) !void {
    const bound = encoder.compressBound(bytes.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const n = encoder.compressFramed(allocator, bytes, compressed, .{ .level = level }, &gpu_encoder.g_default) catch |err| {
        std.debug.print("compress failed for {s}: {s}\n", .{ label, @errorName(err) });
        return err;
    };
    const dst = try allocator.alloc(u8, bytes.len + decoder.safe_space);
    defer allocator.free(dst);
    const written = decoder.decompressFramed(compressed[0..n], dst, &gpu_driver.g_default) catch |err| {
        std.debug.print("decompress failed for {s}: {s}\n", .{ label, @errorName(err) });
        return err;
    };
    if (written != bytes.len) {
        std.debug.print("size mismatch for {s}: expected {d}, got {d}\n", .{ label, bytes.len, written });
        return error.TestUnexpectedResult;
    }
    if (!std.mem.eql(u8, bytes, dst[0..written])) {
        var first_diff: usize = 0;
        while (first_diff < bytes.len and bytes[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        var total_diffs: usize = 0;
        for (bytes, 0..) |b, i| {
            if (b != dst[i]) total_diffs += 1;
        }
        std.debug.print(
            "byte mismatch for {s}: first_diff={d} total_diffs={d} len={d} got={x:0>2} exp={x:0>2}\n",
            .{ label, first_diff, total_diffs, bytes.len, dst[first_diff], bytes[first_diff] },
        );
        return error.TestUnexpectedResult;
    }
}

test "A-023: forced-batch LZ dispatch is byte-identical to unbatched" {
    gpu_roundtrip.lockGpuTests();
    defer gpu_roundtrip.unlockGpuTests();
    if (!gpu_encoder.isAvailable()) {
        std.debug.print("(A-023 skip: encoder unavailable)\n", .{});
        return error.SkipZigTest;
    }
    if (!gpu_driver.isAvailable()) {
        std.debug.print("(A-023 skip: decoder unavailable)\n", .{});
        return error.SkipZigTest;
    }
    if (!gpu_driver.bindContextToCallingThread()) {
        std.debug.print("(A-023 skip: bindContext failed)\n", .{});
        return error.SkipZigTest;
    }
    const allocator = testing.allocator;

    // 3 MB of mixed compressible content → 48 LZ chunks at 64 KB, so a
    // forced batch of 7 runs the per-batch descs-upload / launch / D2H
    // loop 7 times and batch=1 runs it 48 times.
    const src = try allocator.alloc(u8, 3 * 1024 * 1024);
    defer allocator.free(src);
    {
        const pattern = "the quick brown fox jumps over the lazy dog, again and again; ";
        var state: u32 = 0xBEEF;
        for (src, 0..) |*b, i| {
            state = state *% 1103515245 +% 12345;
            // Mostly repeating text with a sprinkle of noise so every
            // level finds matches AND the chunks differ from each other.
            b.* = if ((state >> 24) < 24) @intCast((state >> 16) & 0xFF) else pattern[(i + (state >> 28)) % pattern.len];
        }
    }

    const bound = encoder.compressBound(src.len);
    const baseline = try allocator.alloc(u8, bound);
    defer allocator.free(baseline);
    const forced = try allocator.alloc(u8, bound);
    defer allocator.free(forced);

    // Always reset the hook, even if an expect below fails.
    defer encode_lz.g_force_batch_count_for_test = 0;

    // L1 = greedy single-table hash; L5 = chain parser 3-table stride.
    const levels = [_]u8{ 1, 5 };
    const batch_counts = [_]u32{ 7, 1 };
    for (levels) |level| {
        encode_lz.g_force_batch_count_for_test = 0;
        const n_base = try encoder.compressFramed(allocator, src, baseline, .{ .level = level }, &gpu_encoder.g_default);

        for (batch_counts) |bc| {
            encode_lz.g_force_batch_count_for_test = bc;
            const n_forced = try encoder.compressFramed(allocator, src, forced, .{ .level = level }, &gpu_encoder.g_default);
            encode_lz.g_force_batch_count_for_test = 0;

            if (n_forced != n_base or !std.mem.eql(u8, baseline[0..n_base], forced[0..n_forced])) {
                var first_diff: usize = 0;
                const min_n = @min(n_base, n_forced);
                while (first_diff < min_n and baseline[first_diff] == forced[first_diff]) : (first_diff += 1) {}
                std.debug.print(
                    "A-023 L{d} batch={d}: sizes {d} vs {d}, first_diff={d}\n",
                    .{ level, bc, n_base, n_forced, first_diff },
                );
                return error.TestUnexpectedResult;
            }
            // The batched output must also round-trip.
            try roundtripLabeled(allocator, "A-023 forced-batch roundtrip", src, level);
        }
    }
}

test "real-corpus roundtrips (assets/web.txt) at every level" {
    gpu_roundtrip.lockGpuTests();
    defer gpu_roundtrip.unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;

    // Real corpus: distinct content per 64 KB block. Skip cleanly when
    // the asset is not present (e.g. a sparse checkout). File access in
    // this Zig goes through std.Io (same pattern as srcVK/tests).
    var threaded = std.Io.Threaded.init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const max_len = 8 * 1024 * 1024;
    const web = std.Io.Dir.cwd().readFileAlloc(
        io,
        "assets/web.txt",
        allocator,
        @enumFromInt(max_len),
    ) catch {
        std.debug.print("(skipping corpus tests — assets/web.txt not found)\n", .{});
        return error.SkipZigTest;
    };
    defer allocator.free(web);
    try testing.expect(web.len > 1024 * 1024);

    // Full file at every level — the case class that caught the VK
    // byte-65544 bug while every synthetic shape passed.
    var level: u8 = 1;
    while (level <= 5) : (level += 1) {
        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "web full L{d}", .{level});
        try roundtripLabeled(allocator, label, web, level);
    }

    // Sliced windows: odd offsets + odd lengths vary chunk-boundary
    // alignment, sub-chunk counts, and trailing-literal shapes.
    const Slice = struct { label: []const u8, off: usize, len: usize, level: u8 };
    const slices = [_]Slice{
        .{ .label = "web slice mid L3", .off = 1_000_001, .len = 300_000, .level = 3 },
        .{ .label = "web slice odd L5", .off = 2_222_222, .len = 777_777, .level = 5 },
        .{ .label = "web slice tail L4", .off = 4_000_000, .len = 0, .level = 4 }, // len 0 = to EOF
        .{ .label = "web slice 64K+1 L2", .off = 333_333, .len = 64 * 1024 + 1, .level = 2 },
    };
    for (slices) |s| {
        if (s.off >= web.len) continue;
        const end = if (s.len == 0) web.len else @min(web.len, s.off + s.len);
        try roundtripLabeled(allocator, s.label, web[s.off..end], s.level);
    }
}
