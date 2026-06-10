//! L5 chain-parser hardening roundtrips — backport of the four
//! adversarial patterns from srcVK L5 hardening (`f08713d`,
//! BACKPORTS.md D wave 2).
//!
//! The VK originals are cross-backend SHA tests (ptest_vk spawns both
//! binaries and diffs frames). Those still run — but they require the
//! VK tree and only fire on cross-backend divergence. These mirrors
//! run the SAME patterns (same seeds, same marker positions) through
//! the in-process CUDA encode→decode path so a CUDA-side chain-parser
//! regression fails HERE, locally, with the pattern named.
//!
//! Why these patterns: the pre-existing L5 tests use a repeated 1 KiB
//! random block — a degenerate self-referencing input that exercises
//! none of the parser's decision boundaries. Each case below sits on
//! one:
//!   1. long-literal-run threshold (LONG_LIT_RUN_THRESHOLD=64)
//!   2. hash-chain walk truncation (CHAIN_MAX_STEPS=8)
//!   3. near/far offset-class scoring (OFFSET_CLASS_LEN_MARGIN)
//!   4. block 1→block 2 recent_offset carryover at LZ_BLOCK_SIZE=64 KiB

const std = @import("std");
const testing = std.testing;

const encoder = @import("streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_encoder = @import("driver.zig");
const gpu_decoder = @import("../decode/driver.zig");
const gpu_roundtrip_tests = @import("gpu_roundtrip_tests.zig");

fn roundtripL5(allocator: std.mem.Allocator, label: []const u8, src: []const u8) !void {
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const n = encoder.compressFramed(allocator, src, compressed, .{ .level = 5 }, &gpu_encoder.g_default) catch |err| {
        std.debug.print("L5 hardening compress failed ({s}): {s}\n", .{ label, @errorName(err) });
        return err;
    };
    try testing.expect(n > 0);
    try testing.expect(n <= bound);

    const dst = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(dst);
    const written = decoder.decompressFramed(compressed[0..n], dst, &gpu_decoder.g_default) catch |err| {
        std.debug.print("L5 hardening decompress failed ({s}): {s}\n", .{ label, @errorName(err) });
        return err;
    };
    try testing.expectEqual(src.len, written);
    if (!std.mem.eql(u8, src, dst[0..written])) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        std.debug.print("L5 hardening mismatch ({s}) at byte {d}\n", .{ label, first_diff });
        return error.TestUnexpectedResult;
    }
}

test "L5 hardening: long-literal-run threshold (LONG_LIT_RUN_THRESHOLD=64)" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    // 128 KiB. "ABCD" markers separated by exactly 70 non-matching bytes:
    // every match candidate is gated by a literal run crossing the
    // 64-byte threshold; each marker has a 4-byte recent-match candidate
    // 74 bytes back. Same seed/stride as the VK cross-backend case.
    const total: usize = 128 * 1024;
    const stride: usize = 74;
    const src = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xA1B1_C1D1_E1F1_0011);
    rng.random().bytes(src);
    var off: usize = 0;
    while (off + 4 <= total) : (off += stride) {
        @memcpy(src[off..][0..4], "ABCD");
    }
    try roundtripL5(testing.allocator, "longlitrun", src);
}

test "L5 hardening: chain walk truncation (CHAIN_MAX_STEPS=8)" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    // 64 KiB. "WXYZ" prefix every 16 bytes with unique extensions: deep
    // hash chains force the walk to truncate at CHAIN_MAX_STEPS and pick
    // the best of the first 8 candidates. Same seed/stride as VK.
    const total: usize = 64 * 1024;
    const stride: usize = 16;
    const src = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xB2C2_D2E2_F200_2233);
    rng.random().bytes(src);
    var off: usize = 0;
    while (off + stride <= total) : (off += stride) {
        @memcpy(src[off..][0..4], "WXYZ");
    }
    try roundtripL5(testing.allocator, "chaintrunc", src);
}

test "L5 hardening: mixed near/far offset classes" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    // 256 KiB (4 LZ blocks). 8-byte signature at positions giving the
    // same match both near (<64 KiB) and far (>=64 KiB) offsets —
    // exercises isMatchBetter's offset-class margin. Same seed/positions
    // as VK.
    const total: usize = 256 * 1024;
    const src = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xC3D3_E3F3_0033_4455);
    rng.random().bytes(src);
    const sig = [8]u8{ 0xFE, 0xDC, 0xBA, 0x09, 0x87, 0x65, 0x43, 0x21 };
    const positions = [_]usize{ 100, 200, 70000, 70200, 200100, 200200 };
    for (positions) |p| {
        @memcpy(src[p..][0..sig.len], &sig);
    }
    try roundtripL5(testing.allocator, "mixedoffset", src);
}

test "L5 hardening: block1->block2 boundary recent_offset carryover" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    // 128 KiB spanning the LZ_BLOCK_SIZE=64 KiB boundary. Signature at
    // 100/200 establishes recent_offset=-100 in block 1; the 65440/65540
    // pair re-tests that offset just past the boundary in block 2. Same
    // seed/positions as VK.
    const total: usize = 128 * 1024;
    const src = try testing.allocator.alloc(u8, total);
    defer testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xD4E4_F400_4455_66AA);
    rng.random().bytes(src);
    const sig = [8]u8{ 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 };
    const positions = [_]usize{ 100, 200, 65440, 65540 };
    for (positions) |p| {
        @memcpy(src[p..][0..sig.len], &sig);
    }
    try roundtripL5(testing.allocator, "blockcarry", src);
}
