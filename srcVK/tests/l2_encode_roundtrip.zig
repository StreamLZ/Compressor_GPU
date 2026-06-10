//! NEW per Exception 3 (no CUDA counterpart).
//!
//! L2-specific encode/decode round-trip + anti-H2-regression sanity gate.
//!
//! L2 = L1's greedy encoder with hash_bits=18 (vs L1's 17). Same kernel,
//! same wire format, no Huffman. The encoder kernel plumbs hash_bits via
//! levels.hashBitsForLevel(level=2) at encode_lz.zig:56.
//!
//! H2 LESSON (mandatory gate): the H2 disaster was a code path that
//! silently produced VALID-SLZ ALL-RAW output. A pure roundtrip test
//! passes that case because the all-raw frame decodes correctly, BUT the
//! point of compression is lost. So every L2 test in this file MUST also
//! assert one of:
//!   1. SHA(VK output) == SHA(CUDA output)   -- gold standard
//!   2. compressed_size  <  input_size * R   -- weaker, catches all-raw
//!
//! In-process tests here use option (2) because spawning the CUDA binary
//! from every in-process worker would inflate test wall-clock. The
//! cross-backend tests (cross_backend_roundtrip.zig) carry the full SHA
//! byte-identity gate.
//!

const std = @import("std");
const testing = std.testing;
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const enc_driver = @import("../encode/driver.zig");
const dec_driver = @import("../decode/driver.zig");

// VK adaptation: see l1_encode_roundtrip.zig::roundtripWithLevel for the
// per-test EncodeContext / DecodeContext rationale (concurrent worker
// pool + persistent device buffer clobber). Identical pattern here.
fn roundtripL2(allocator: std.mem.Allocator, src: []const u8) !usize {
    var enc_ctx: enc_driver.EncodeContext = .{};
    defer enc_ctx.deinit(allocator);
    var dec_ctx: dec_driver.DecodeContext = .{};
    defer dec_ctx.deinit();

    const opts: encoder.Options = .{ .level = 2 };
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
    return written;
}

// H2-regression gate: compressible input MUST shrink. If the encoder
// silently goes all-raw (the H2 failure mode) the compressed size would
// be ~src.len + per-sub-chunk overhead, which blows past this bound.
// The threshold is set generously (75% of input) so legitimate L2 output
// on text-like data sits well below it; an all-raw frame on the same
// input would sit at ~100%+.
fn assertCompressionWorked(input_size: usize, compressed_size: usize, max_ratio_pct: u32) !void {
    const max_bytes: usize = (input_size * max_ratio_pct) / 100;
    if (compressed_size > max_bytes) {
        std.debug.print(
            "L2 H2-regression gate FAIL: compressed={d} bytes > {d}% of input ({d} bytes). " ++
                "Encoder may be emitting all-raw frames.\n",
            .{ compressed_size, max_ratio_pct, max_bytes },
        );
        return error.L2CompressionTooWeak;
    }
}

test "L2 encode/decode round-trip: ascending bytes (1 MiB) + ratio gate" {
    // Highly compressible: repeating 0..255 cycle → L2 must emit matches,
    // not raw. A real L2 frame for this input is < 10% of input; we set
    // the H2-regression gate at 50% which is wildly above the real
    // expected ratio yet catches an all-raw regression at 100%.
    const len: usize = 1 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    for (src, 0..) |*b, i| b.* = @intCast(i & 0xFF);
    const compressed = try roundtripL2(std.testing.allocator, src);
    try assertCompressionWorked(len, compressed, 50);
}

test "L2 encode/decode round-trip: repeating English pattern (4 MiB) + ratio gate" {
    // 4 MiB = 16 × chunk_size (256 KiB); exercises multi-chunk
    // boundaries that the L2-specific iter-3/4 fixes addressed.
    // Repeating-text pattern compresses extremely well at L2; the 30%
    // gate is loose enough to absorb any legitimate ratio drift while
    // still catching an H2-class all-raw regression at ~100%.
    const len: usize = 4 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    const pattern = "the quick brown fox jumps over the lazy dog ";
    var i: usize = 0;
    while (i < len) : (i += 1) src[i] = pattern[i % pattern.len];
    const compressed = try roundtripL2(std.testing.allocator, src);
    try assertCompressionWorked(len, compressed, 30);
}

test "L2 encode/decode round-trip: zeros (2 MiB) + ratio gate" {
    // All-zeros is the most compressible input possible; real L2 ratio
    // is < 1%. Threshold at 25% gives huge headroom while still
    // catching all-raw at 100%.
    const len: usize = 2 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    @memset(src, 0);
    const compressed = try roundtripL2(std.testing.allocator, src);
    try assertCompressionWorked(len, compressed, 25);
}

test "L2 encode/decode round-trip: random PRNG (100 KiB) round-trip only" {
    // Random input is INCOMPRESSIBLE: a correct L2 encoder will emit
    // mostly raw subchunks because there are no matches. So we cannot
    // apply the ratio gate here; this test is the basic-correctness
    // companion to the ratio-gated tests above. The other tests in this
    // file catch the H2 all-raw regression class.
    const src = try std.testing.allocator.alloc(u8, 100 * 1024);
    defer std.testing.allocator.free(src);
    var rng = std.Random.DefaultPrng.init(0xC0FFEEC0DE);
    rng.random().bytes(src);
    _ = try roundtripL2(std.testing.allocator, src);
}

test "L2 encode/decode round-trip: 8 MiB above-threshold (PRNG-seeded text) + ratio gate" {
    // 8 MiB is the input size that exposed the L2-specific iter-3/4 bug
    // class per ToDo / project memory ("above-threshold regression
    // class"). We synthesize a deterministically-compressible payload
    // by repeating a 1 KiB random block 8 K times; this gives L2's hash
    // table real work yet leaves obvious matches the encoder must find.
    const len: usize = 8 * 1024 * 1024;
    const src = try std.testing.allocator.alloc(u8, len);
    defer std.testing.allocator.free(src);
    var block: [1024]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(0xDEADBEEFCAFE);
    rng.random().bytes(&block);
    var i: usize = 0;
    while (i < len) : (i += 1) src[i] = block[i & 1023];
    const compressed = try roundtripL2(std.testing.allocator, src);
    // Repeating 1 KiB block over 8 MiB → real L2 ratio is tiny;
    // 20% threshold is loose enough yet catches all-raw at 100%.
    try assertCompressionWorked(len, compressed, 20);
}
