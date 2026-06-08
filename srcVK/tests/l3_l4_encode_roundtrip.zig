//! NEW per Exception 3 (no CUDA counterpart). Phase 2A-decoder iter 4c
//! correctness gate.
//!
//! VK-encode -> VK-decode round-trip at levels 3 and 4. Mirrors the
//! L2-shape harness in l2_encode_roundtrip.zig but expands the size
//! matrix to provoke the two iter-4c bug regimes documented below.
//!
//! WHY THIS FILE EXISTS — iter 4c gap audit:
//!   ptest_vk shipped 88/0/0 GREEN after iter 4c yet end-to-end L3+
//!   decode is broken in two distinct ways. The green signal was a
//!   FALSE POSITIVE because the suite had ZERO L3 or L4 round-trip
//!   coverage. This file closes that gap so the next workflow has a
//!   reproducible failing signal to debug.
//!
//! IRON-LAW BUG SHAPES (expected to fire with iter 4c on the working tree):
//!   1. SILENT CORRUPTION AT BYTE 65544 — inputs larger than the
//!      65,536-byte LZ_BLOCK_SIZE decode the first 65,544 bytes correctly
//!      then go to zero. Suspected: block-1 -> block-2 transition in
//!      lz_decode_general.glsl:415-450. Hit it with any input
//!      > LZ_BLOCK_SIZE; we use 128 KiB and 1 MiB.
//!   2. LOUD KernelLaunchFailed ON SMALL INPUTS — inputs around 32-70 KiB
//!      trip a runtime guard or alignment failure inside procLaunchKernel.
//!      We probe with 32 KiB (< LZ_BLOCK_SIZE) so the bug surfaces as a
//!      Zig error rather than a byte mismatch.
//!
//! Per CLAUDE.md the next workflow OWNS the fix. This file's job is to
//! produce a clear failing signal: first_diff index, observed length,
//! first 16 bytes of expected/got — gold for the bug-fix workflow.

const std = @import("std");
const testing = std.testing;
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const enc_driver = @import("../encode/driver.zig");
const dec_driver = @import("../decode/driver.zig");
const frame = @import("../format/frame_format.zig");

// VK adaptation: corpus subsets — cap at 1 GiB. See l1_decode_roundtrip.zig
// for the same value. Real-corpus tests load from assets/ at fixed prefix
// lengths so a 100 MB enwik8.txt is read once and trimmed.
const max_corpus_bytes: usize = 1 << 30;

fn makeCorpusIo() std.Io.Threaded {
    // VK adaptation: identical to l1_decode_roundtrip.zig::makeIo. The
    // failing allocator suffices because the corpus reader path only uses
    // the testing.allocator we pass explicitly, no synchronous IO buffers.
    return std.Io.Threaded.init(std.mem.Allocator.failing, .{});
}

fn readCorpusHead(allocator: std.mem.Allocator, rel_path: []const u8, head_bytes: usize) ![]u8 {
    // VK adaptation: read whole file then trim to head_bytes. The corpus
    // files are mmap-friendly and we already pay the 100 MB read in the
    // L1 file, so this is no worse. If the file is missing we propagate
    // FileNotFound; callers translate that to error.SkipZigTest so a
    // corpus-less dev box still reports a clean run.
    var io_inst = makeCorpusIo();
    defer io_inst.deinit();
    const all = try std.Io.Dir.cwd().readFileAlloc(
        io_inst.io(),
        rel_path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    );
    if (all.len <= head_bytes) return all;
    // Realloc-shrink to the requested head so callers don't carry the
    // full 100 MB enwik8 backing store for a 256 KiB test.
    const head = try allocator.alloc(u8, head_bytes);
    @memcpy(head, all[0..head_bytes]);
    allocator.free(all);
    return head;
}

// VK adaptation: per-test EncodeContext + DecodeContext (the established
// l1/l2_encode_roundtrip.zig pattern). The shared g_default contexts in
// the parallel worker pool race against each other on
// encode_context.ensureBuf (destroy+create of persistent device buffers
// between worker A's unlock and worker B's lock); per-test contexts
// sidestep that hazard so the diagnostic we surface here is the actual
// L3/L4 decode bug — not a worker-pool clobber masquerading as one.
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

    // Diagnostic gold for the bug-fix workflow: when the decode comes
    // back short or with byte mismatches, print exactly where it went
    // sideways. Bug 1 (silent block-1->block-2 corruption) typically
    // shows first_diff ≈ 65544 with the tail all zeros; bug 2 (small
    // inputs) tends to fault inside decompressFramed itself and never
    // reach this block.
    if (out_written != src.len) {
        std.debug.print(
            "L{d} round-trip SIZE MISMATCH: src.len={d} out_written={d} written_frame={d}\n",
            .{ level, src.len, out_written, written },
        );
        return error.DecodedSizeMismatch;
    }
    if (!std.mem.eql(u8, src, dst[0..out_written])) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        var total_diffs: usize = 0;
        var j: usize = 0;
        while (j < src.len) : (j += 1) if (src[j] != dst[j]) {
            total_diffs += 1;
        };
        // Tail-zero probe: bug 1 zeroes everything after first_diff.
        var trailing_zeros: usize = 0;
        var k: usize = src.len;
        while (k > 0) {
            k -= 1;
            if (dst[k] != 0) break;
            trailing_zeros += 1;
        }
        std.debug.print(
            "L{d} round-trip BYTE MISMATCH: src.len={d} first_diff={d} total_diffs={d} trailing_zeros_in_dst={d}\n",
            .{ level, src.len, first_diff, total_diffs, trailing_zeros },
        );
        // Spot-print the 16-byte window centred on first_diff so the
        // bug-fix workflow can eyeball block-boundary corruption.
        const lo = if (first_diff >= 8) first_diff - 8 else 0;
        const hi = @min(first_diff + 8, src.len);
        std.debug.print("  src [{d}..{d}]:", .{ lo, hi });
        for (src[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n  dst [{d}..{d}]:", .{ lo, hi });
        for (dst[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n", .{});
        return error.DecodedBytesMismatch;
    }
}

// VK adaptation: deterministic compressible payload. Repeating a 1 KiB
// random block gives the encoder real matches to find at L3/L4 while
// keeping the source content reproducible across runs. Matches the
// shape used by shaByteIdentityL2 in cross_backend_roundtrip.zig.
fn fillCompressible(buf: []u8, seed: u64) void {
    var block: [1024]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().bytes(&block);
    var i: usize = 0;
    while (i < buf.len) : (i += 1) buf[i] = block[i & 1023];
}

// =====================================================================
// L3 round-trips
// =====================================================================

test "L3 encode/decode round-trip: 32 KiB (small-input KernelLaunchFailed probe)" {
    // VK adaptation: 32 KiB sits well below LZ_BLOCK_SIZE (65,536) and
    // inside the band that iter 4c made fragile. If the small-input
    // KernelLaunchFailed bug fires, decompressFramed bubbles a Zig
    // error and the test reports it via runner. Expected to FAIL until
    // the iter 4c regression in procLaunchKernel is addressed.
    const len: usize = 32 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    fillCompressible(src, 0x3232_3232_3232_3232);
    try roundtripWithLevel(std.testing.allocator, src, 3);
}

test "L3 encode/decode round-trip: 128 KiB (byte-65544 block-transition probe)" {
    // VK adaptation: 128 KiB = 2 × LZ_BLOCK_SIZE. Inputs above 65,536
    // expose the iter 4c block-1 -> block-2 corruption: decode returns
    // the right length but bytes [65544..end] are zeroed. Expected to
    // FAIL with first_diff ≈ 65544 and a large trailing_zeros_in_dst.
    const len: usize = 128 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    fillCompressible(src, 0xC0DE_C0DE_C0DE_C0DE);
    try roundtripWithLevel(std.testing.allocator, src, 3);
}

test "L3 encode/decode round-trip: 1 MiB (multi-block stress)" {
    // VK adaptation: 1 MiB = 16 × LZ_BLOCK_SIZE; even if block-2 partly
    // works the later boundaries can fail in different ways. Crosses
    // the iter-3/4 multi-chunk boundary class as well.
    const len: usize = 1 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    fillCompressible(src, 0x1111_2222_3333_4444);
    try roundtripWithLevel(std.testing.allocator, src, 3);
}

// =====================================================================
// L4 round-trips
// =====================================================================
//
// L4 is L3's parser with hash_bits capped at 17 (see encode/levels.zig).
// Both levels share the same decode path so they're expected to share
// the same iter-4c failure modes; covering both guards against either
// level being independently regressed in a later fix.

test "L4 encode/decode round-trip: 32 KiB (small-input KernelLaunchFailed probe)" {
    const len: usize = 32 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    fillCompressible(src, 0x4444_5555_6666_7777);
    try roundtripWithLevel(std.testing.allocator, src, 4);
}

test "L4 encode/decode round-trip: 128 KiB (byte-65544 block-transition probe)" {
    const len: usize = 128 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    fillCompressible(src, 0x8888_9999_AAAA_BBBB);
    try roundtripWithLevel(std.testing.allocator, src, 4);
}

test "L4 encode/decode round-trip: 1 MiB (multi-block stress)" {
    const len: usize = 1 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    fillCompressible(src, 0xFFEE_DDCC_BBAA_9988);
    try roundtripWithLevel(std.testing.allocator, src, 4);
}

// =====================================================================
// REAL CORPUS round-trips — iter 4c bug surfaces on real data, not
// synthetic. The synthetic compressible payload above (1 KiB repeating
// random block) produces narrow Huffman distributions with 1-2 hot
// symbols that exercise only a tiny corner of the decoder. Real English
// text has 30-50 active symbols and exercises the full LUT + 4-stream
// parallel path; real binary tarballs have near-maximum entropy and
// exercise the larger sub-chunk count + the KernelLaunchFailed path.
//
// EMPIRICAL EVIDENCE (parent agent, iter 4c working tree, NVIDIA):
//   VK->VK L3 web.txt (4.5 MB):    DIFFER (silent corruption)
//   VK->VK L4 web.txt:             DIFFER (silent corruption)
//   VK->VK L3 enwik8.txt (100 MB): DIFFER (silent corruption)
//   VK->VK L4 enwik8.txt:          FAIL (KernelLaunchFailed)
//
// These tests subset the heads of the corpus files at sizes intended to
// cross LZ_BLOCK_SIZE (65,536) so the block-1->block-2 silent corruption
// has a chance to fire. If the small subset doesn't reproduce, escalate
// to a larger head (1 MiB, then full file).
// =====================================================================

fn roundtripCorpusWithLevel(allocator: std.mem.Allocator, corpus_path: []const u8, head_bytes: usize, level: u8) !void {
    // VK adaptation: thin wrapper around roundtripWithLevel that loads a
    // corpus subset, skips on missing-asset, and frees the source after.
    // Bridges the synthetic/real-corpus signal split documented above
    // without duplicating the round-trip diagnostic body.
    const src = readCorpusHead(allocator, corpus_path, head_bytes) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(src);
    try roundtripWithLevel(allocator, src, level);
}

test "L3 encode/decode round-trip: web.txt 128 KiB head (real text corpus)" {
    // VK adaptation: 128 KiB head of assets/web.txt — 2 x LZ_BLOCK_SIZE,
    // realistic English text with 30-50 active symbols. Expected to FAIL
    // with first_diff ~ 65544 + large trailing_zeros on iter 4c (block-1
    // -> block-2 silent corruption on the real-distribution decoder
    // path that the synthetic 1-KiB-block payload doesn't exercise).
    try roundtripCorpusWithLevel(std.testing.allocator, "assets/web.txt", 128 * 1024, 3);
}

test "L3 encode/decode round-trip: web.txt 1 MiB head (real text corpus)" {
    // VK adaptation: 1 MiB head of web.txt. If the 128 KiB head doesn't
    // reproduce the silent corruption (the decoder's faulty path may be
    // input-size-dependent), the 1 MiB head crosses 16 block boundaries
    // and should reliably trip the bug. Per parent agent reproduction
    // the full web.txt at 4.5 MB DIFFERs at iter 4c.
    try roundtripCorpusWithLevel(std.testing.allocator, "assets/web.txt", 1 * 1024 * 1024, 3);
}

test "L3 encode/decode round-trip: enwik8.txt 256 KiB head (real text corpus)" {
    // VK adaptation: 256 KiB head of assets/enwik8.txt — Wikipedia text
    // dump, broader symbol distribution than web.txt and known to trip
    // the iter 4c silent corruption at full size (100 MB DIFFERs per
    // parent agent). 256 KiB = 4 x LZ_BLOCK_SIZE, enough boundary
    // transitions to reproduce without a 100 MB read on every test run.
    try roundtripCorpusWithLevel(std.testing.allocator, "assets/enwik8.txt", 256 * 1024, 3);
}

test "L4 encode/decode round-trip: web.txt 128 KiB head (real text corpus)" {
    // VK adaptation: L4 mirror of the web.txt 128 KiB L3 test. Same bug
    // class — L4 is L3's parser with hash_bits capped at 17, same
    // decode path. Per parent agent reproduction VK->VK L4 web.txt
    // DIFFERs at full size.
    try roundtripCorpusWithLevel(std.testing.allocator, "assets/web.txt", 128 * 1024, 4);
}

test "L4 encode/decode round-trip: enwik8.txt 256 KiB head (real text corpus)" {
    // VK adaptation: L4 mirror of enwik8 256 KiB. At iter 4c the full
    // enwik8 L4 surfaces KernelLaunchFailed (loud failure) rather than
    // silent corruption, so this test may surface a Zig error from
    // decompressFramed rather than a byte mismatch. Either failure
    // shape gives the bug-fix workflow the signal it needs.
    try roundtripCorpusWithLevel(std.testing.allocator, "assets/enwik8.txt", 256 * 1024, 4);
}

// =====================================================================
// GOLDEN DECODE — CUDA-encoded .slz files decoded in-process via VK.
// Mirrors the l1_decode_roundtrip.zig pattern but at L3/L4. Distinguishes
// decode-side bug (golden->VK fails) from encode-side bug (golden->VK
// passes but corpus encode->VK fails above) without paying CUDA
// subprocess overhead.
// =====================================================================

fn goldenDecodeAndCompare(allocator: std.mem.Allocator, golden_path: []const u8, original_path: []const u8) !void {
    var io_inst = makeCorpusIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    const golden = std.Io.Dir.cwd().readFileAlloc(
        io,
        golden_path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(golden);
    const original = std.Io.Dir.cwd().readFileAlloc(
        io,
        original_path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(original);

    const hdr = try frame.parseHeader(golden);
    const want_size: usize = if (hdr.content_size) |cs| @intCast(cs) else original.len;
    try testing.expectEqual(original.len, want_size);

    const dst = try allocator.alloc(u8, want_size + 64);
    defer allocator.free(dst);
    @memset(dst, 0);

    var dec_ctx: dec_driver.DecodeContext = .{};
    defer dec_ctx.deinit();

    // The decode may itself error (bug 2: KernelLaunchFailed); catch and
    // print the error name so the bug-fix workflow knows which surface
    // is firing without re-running.
    const written = decoder.decompressFramed(golden, dst, &dec_ctx) catch |e| {
        std.debug.print(
            "golden decode FAIL ({s}): error.{s} on frame_size={d}\n",
            .{ golden_path, @errorName(e), golden.len },
        );
        return e;
    };

    if (written != original.len) {
        std.debug.print(
            "golden decode SIZE MISMATCH ({s}): original.len={d} written={d}\n",
            .{ golden_path, original.len, written },
        );
        return error.DecodedSizeMismatch;
    }
    if (!std.mem.eql(u8, original, dst[0..written])) {
        var first_diff: usize = 0;
        while (first_diff < original.len and original[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        var total_diffs: usize = 0;
        var j: usize = 0;
        while (j < original.len) : (j += 1) if (original[j] != dst[j]) {
            total_diffs += 1;
        };
        var trailing_zeros: usize = 0;
        var k: usize = original.len;
        while (k > 0) {
            k -= 1;
            if (dst[k] != 0) break;
            trailing_zeros += 1;
        }
        std.debug.print(
            "golden decode BYTE MISMATCH ({s}): original.len={d} first_diff={d} total_diffs={d} trailing_zeros={d}\n",
            .{ golden_path, original.len, first_diff, total_diffs, trailing_zeros },
        );
        const lo = if (first_diff >= 8) first_diff - 8 else 0;
        const hi = @min(first_diff + 8, original.len);
        std.debug.print("  original [{d}..{d}]:", .{ lo, hi });
        for (original[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n  decoded  [{d}..{d}]:", .{ lo, hi });
        for (dst[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n", .{});
        return error.DecodedBytesMismatch;
    }
}

test "L3 golden decode: web.txt CUDA-encoded -> VK decode (real frame)" {
    // VK adaptation: pre-existing CUDA-encoded L3 golden at
    // tests/goldens/web.txt.L3.slz (Jun 2 baseline). In-process VK
    // decode, byte-compared against assets/web.txt. Isolates the
    // decode-side half of the iter 4c regression: if THIS fails the bug
    // is in the decoder; if THIS passes but the corpus encode round-trip
    // above fails, the bug is encode-side.
    try goldenDecodeAndCompare(
        std.testing.allocator,
        "tests/goldens/web.txt.L3.slz",
        "assets/web.txt",
    );
}

test "L4 golden decode: web.txt CUDA-encoded -> VK decode (real frame)" {
    // VK adaptation: L4 mirror of the L3 golden-decode test above.
    try goldenDecodeAndCompare(
        std.testing.allocator,
        "tests/goldens/web.txt.L4.slz",
        "assets/web.txt",
    );
}
