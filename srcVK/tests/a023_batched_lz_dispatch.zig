//! VK adaptation A-023 regression coverage. See
//! `srcVK/PortAdaptations.md::A-023`. The batched LZ dispatch is a
//! VRAM-pressure adaptation: when `ensureBuf` fails on the full
//! `num_chunks * hash_size * 4` allocation, `gpuCompressImpl` halves
//! `batch_count` and dispatches the LZ kernel in back-to-back batches
//! that each reuse a smaller hash buffer. Per-chunk results stay byte-
//! identical because each chunk reinitialises its own hash region to
//! HASH_EMPTY at entry.
//!
//! The organic VRAM-pressure trigger needs ~14 GiB hash (enwik9 L3 on
//! a 16 GiB device), well above CI hardware. To exercise the batched
//! path without enwik9 we expose `g_force_batch_count_for_test` on
//! `srcVK/encode/encode_lz.zig` — when set to a positive value before
//! the encode, `batch_count` is capped to that value and the batched
//! path runs even on small inputs.
//!
//! These tests assert: VK encode at L3 and L5 with the cap set produces
//! BYTE-IDENTICAL output to VK encode without the cap, AND the decoded
//! payload round-trips to the original source. That guarantees the
//! per-batch H2D / sizes-memset / launch / D2H / gather choreography
//! does not corrupt or reorder bytes vs the single-shot path.
//!
//! Each test runs in-process so the `g_force_batch_count_for_test`
//! global toggle is safe — the bracketing `defer` resets it to 0 at
//! end of the test so sibling tests see the production default.

const std = @import("std");
const testing = std.testing;
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const enc_driver = @import("../encode/driver.zig");
const dec_driver = @import("../decode/driver.zig");
const encode_lz = @import("../encode/encode_lz.zig");

/// Encode `src` at `level` with an optional `force_batch_count` cap
/// (0 = production default, no cap). Returns an owned compressed slice.
fn encodeOnce(
    allocator: std.mem.Allocator,
    src: []const u8,
    level: u8,
    force_batch_count: u32,
) ![]u8 {
    var enc_ctx: enc_driver.EncodeContext = .{};
    defer enc_ctx.deinit(allocator);

    const opts: encoder.Options = .{ .level = level };
    const bound = encoder.compressBound(src.len);
    const buf = try allocator.alloc(u8, bound);
    errdefer allocator.free(buf);

    const saved_force = encode_lz.g_force_batch_count_for_test;
    encode_lz.g_force_batch_count_for_test = force_batch_count;
    defer encode_lz.g_force_batch_count_for_test = saved_force;

    const written = try encoder.compressFramed(allocator, src, buf, opts, &enc_ctx);
    return try allocator.realloc(buf, written);
}

/// Asserts that:
///   * VK encode at `level` succeeds both with and without batching
///   * the two compressed payloads are byte-identical (proves batched
///     path produces same bytes as single-shot)
///   * decode of the batched payload reproduces the source exactly
fn assertBatchedIdentity(
    allocator: std.mem.Allocator,
    src: []const u8,
    level: u8,
    force_batch_count: u32,
) !void {
    const baseline = try encodeOnce(allocator, src, level, 0);
    defer allocator.free(baseline);
    const batched = try encodeOnce(allocator, src, level, force_batch_count);
    defer allocator.free(batched);

    if (baseline.len != batched.len) {
        std.debug.print(
            "A-023 L{d} batched vs single-shot LENGTH MISMATCH: baseline={d} batched={d} src.len={d} force={d}\n",
            .{ level, baseline.len, batched.len, src.len, force_batch_count },
        );
        return error.BatchedLengthMismatch;
    }
    if (!std.mem.eql(u8, baseline, batched)) {
        var first_diff: usize = 0;
        while (first_diff < baseline.len and baseline[first_diff] == batched[first_diff]) : (first_diff += 1) {}
        std.debug.print(
            "A-023 L{d} batched vs single-shot BYTE MISMATCH: first_diff={d} src.len={d} force={d}\n",
            .{ level, first_diff, src.len, force_batch_count },
        );
        return error.BatchedBytesMismatch;
    }

    var dec_ctx: dec_driver.DecodeContext = .{};
    defer dec_ctx.deinit();
    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);
    const out_written = try decoder.decompressFramed(batched, dst, &dec_ctx);
    if (out_written != src.len) {
        std.debug.print(
            "A-023 L{d} batched round-trip SIZE MISMATCH: out_written={d} src.len={d}\n",
            .{ level, out_written, src.len },
        );
        return error.RoundtripSizeMismatch;
    }
    if (!std.mem.eql(u8, src, dst[0..src.len])) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        std.debug.print(
            "A-023 L{d} batched round-trip BYTE MISMATCH: first_diff={d} src.len={d}\n",
            .{ level, first_diff, src.len },
        );
        return error.RoundtripBytesMismatch;
    }
}

/// Repeating-1 KiB-random-block payload — same shape as
/// `l3_l4_encode_roundtrip.zig::fillCompressible`. Compressible enough
/// that L3/L5 produce a real LZ stream (not all-raw), so the per-batch
/// descs/sizes/gather/D2H paths see non-trivial traffic.
fn fillCompressible(buf: []u8, seed: u64) void {
    var block: [1024]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().bytes(&block);
    var i: usize = 0;
    while (i < buf.len) : (i += 1) buf[i] = block[i & 1023];
}

// =====================================================================
// L3 batched-dispatch regression
// =====================================================================
//
// 1 MiB at force_batch_count=4 means num_chunks=16 splits into 4 batches
// of 4 chunks each. Covers the per-batch H2D / memset / D2H / gather
// path 4 times in one test — enough to surface any inter-batch state
// bleed.

test "A-023: L3 batched dispatch byte-identical to single-shot (1 MiB compressible)" {
    const len: usize = 1 * 1024 * 1024;
    const src = try testing.allocator.alloc(u8, len);
    defer testing.allocator.free(src);
    fillCompressible(src, 0xA023_3333_3333_3333);
    try assertBatchedIdentity(testing.allocator, src, 3, 4);
}

test "A-023: L3 batched dispatch byte-identical to single-shot (4 MiB compressible, batch=2)" {
    // Smaller batch_count forces MORE batches over the same chunk count
    // (~64 chunks split 32 ways). Stresses the per-batch loop start /
    // boundary handling vs the single-shot path.
    const len: usize = 4 * 1024 * 1024;
    const src = try testing.allocator.alloc(u8, len);
    defer testing.allocator.free(src);
    fillCompressible(src, 0xA023_3344_3344_3344);
    try assertBatchedIdentity(testing.allocator, src, 3, 2);
}

// =====================================================================
// L5 batched-dispatch regression
// =====================================================================
//
// L5 uses the chain parser (`levels.useChainParser(5) == true`) so the
// hash_stride is `2*hash_size + NEXT_HASH_ENTRIES/2` instead of just
// `hash_size`. The batched-dispatch path computes the hash budget from
// `hash_stride` so the chain variant needs its own regression to ensure
// the stride calculation is correct.

test "A-023: L5 batched dispatch byte-identical to single-shot (1 MiB compressible)" {
    const len: usize = 1 * 1024 * 1024;
    const src = try testing.allocator.alloc(u8, len);
    defer testing.allocator.free(src);
    fillCompressible(src, 0xA023_5555_5555_5555);
    try assertBatchedIdentity(testing.allocator, src, 5, 4);
}
