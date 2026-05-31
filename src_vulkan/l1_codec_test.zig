//! Phase-4 L1 codec round-trip test.
//!
//! Vertical slice exercising the full Vulkan L1 path:
//!
//!   1. driver.ensureInit                       (instance + device + queue)
//!   2. probe                                   (pick tier — tier1/tier1_nv/tier2)
//!   3. loadSpv("lz_encode" / "lz_decode")      (runtime load from zig-out/)
//!   4. Generate test input bytes
//!   5. l1_codec.encodeL1Sync(ctx, src, enc_spv) → EncodeResult
//!   6. l1_codec.decodeL1Sync(ctx, streams, src.len, dst, dec_spv)
//!   7. Compare dst vs src byte-by-byte
//!   8. Print one PASS/FAIL line per test case
//!   9. freeStreams + driver.deinit
//!
//! Two cases run in sequence:
//!   * Small (256 B): repetitive "ABCDABCD…" — easier failure-mode
//!     debugging since the expected compression ratio is huge.
//!   * Large (~64 KiB): mix of 16 KiB repetitive + 48 KiB of real
//!     corpus bytes from `assets/web.txt` (falls back to deterministic
//!     LCG random bytes if the file is missing).
//!
//! The larger case only fires if the smaller case passes — failing
//! fast on the small one is the more productive debug path.
//!
//! No allocator: every host-side buffer is stack-bounded. SPV is read
//! into a 1 MiB stack array per kernel; the src + dst buffers live in
//! a single 64 KiB stack array each. Total stack footprint < 200 KiB.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");

// ── Tuning ─────────────────────────────────────────────────────────

const MAX_SPV_BYTES: usize = 1 << 20; // 1 MiB; the shaders top out ~10 KiB.
const SPV_DIR_REL: []const u8 = "zig-out/shaders";

/// Small test case: 256 B repetitive — fast feedback loop on failure.
const SMALL_SIZE: usize = 256;

/// Large test case: 64 KiB (= LZ_BLOCK_SIZE) of mixed input. Stays under
/// 128 KB so the phase-1 two-block design covers it without needing the
/// (phase-2) chunk-boundary header wrapping.
const LARGE_SIZE: usize = 64 * 1024;
const LARGE_REPETITIVE_BYTES: usize = 16 * 1024;
const LARGE_CORPUS_BYTES: usize = LARGE_SIZE - LARGE_REPETITIVE_BYTES;

/// Path to the corpus we'll splice into the large case. Falls back to
/// a deterministic LCG when missing so the test can run in a clean clone.
const CORPUS_PATH: []const u8 = "assets/web.txt";

const TestError = error{
    SpvOpenFailed,
    SpvReadFailed,
    SpvTooLarge,
    UnsupportedTier,
    Mismatch,
};

// ── Helpers (same SPV-load shape as dispatch_test / match_any_bench) ─

fn probeTierName(t: probe_mod.Tier) ?[]const u8 {
    return switch (t) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => null,
    };
}

/// Load `<SPV_DIR_REL>/<kernel>.<tier>.spv` into the tail of `dest` and
/// return the populated slice. Head 256 B is scratch for the path
/// string; tail starts 4-byte aligned (256 % 4 == 0) per the SPV
/// pCode alignment requirement.
fn loadSpv(io: std.Io, kernel: []const u8, tier_name: []const u8, dest: []u8) ![]u8 {
    var path_buf: [256]u8 = undefined;
    const filename = try std.fmt.bufPrint(&path_buf, "{s}/{s}.{s}.spv", .{ SPV_DIR_REL, kernel, tier_name });
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch return error.SpvOpenFailed;
    defer file.close(io);
    const body = dest[256..];
    const n = file.readPositionalAll(io, body, 0) catch return error.SpvReadFailed;
    if (n == body.len) return error.SpvTooLarge;
    return body[0..n];
}

// ── Test-input generators ─────────────────────────────────────────

/// Fill `dst` with repeating "ABCDABCD…" — every 4 bytes the same
/// 4-byte pattern. The L1 encoder should land a giant single match
/// after the first 4 bytes, exercising the fast-inline-emit + match-
/// extension paths.
fn fillRepetitive(dst: []u8) void {
    const pat: [4]u8 = .{ 'A', 'B', 'C', 'D' };
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        dst[i] = pat[i & 3];
    }
}

/// Deterministic LCG fallback for when assets/web.txt is missing. Same
/// constants the M9 conformance harness uses for its synthetic case.
fn fillDeterministic(dst: []u8, seed: u64) void {
    var s = seed;
    var i: usize = 0;
    while (i < dst.len) : (i += 1) {
        s = s *% 6364136223846793005 +% 1442695040888963407;
        // Keep it ASCII-ish so a hex dump on failure is readable.
        dst[i] = @intCast(((s >> 33) & 0x7F) | 0x20);
    }
}

/// Try to splat the first `dst.len` bytes of CORPUS_PATH into `dst`.
/// Returns false on any I/O failure so the caller can fall back.
fn fillFromCorpus(io: std.Io, dst: []u8) bool {
    var file = std.Io.Dir.cwd().openFile(io, CORPUS_PATH, .{}) catch return false;
    defer file.close(io);
    const n = file.readPositionalAll(io, dst, 0) catch return false;
    // Pad the tail with deterministic bytes if the corpus was shorter
    // than requested (assets/web.txt is ~1 MiB so this shouldn't fire).
    if (n < dst.len) fillDeterministic(dst[n..], 0x1234);
    return true;
}

/// Compose the large case: head = repetitive ABCD pattern; tail = first
/// LARGE_CORPUS_BYTES of `assets/web.txt` (or deterministic LCG).
fn fillLargeCase(io: std.Io, dst: []u8) void {
    fillRepetitive(dst[0..LARGE_REPETITIVE_BYTES]);
    if (!fillFromCorpus(io, dst[LARGE_REPETITIVE_BYTES..])) {
        fillDeterministic(dst[LARGE_REPETITIVE_BYTES..], 0xC0DE);
    }
}

// ── Round-trip driver ─────────────────────────────────────────────

const RoundTripResult = struct {
    pass: bool,
    /// First mismatching byte offset on failure; undefined on pass.
    first_diff: usize = 0,
    /// Total mismatch count on failure; 0 on pass.
    total_diffs: usize = 0,
    comp_bytes: u32 = 0,
};

/// Run encode + decode + compare for a single payload. The dst slice
/// must be at least src.len bytes long; only src.len bytes are touched.
fn roundTrip(
    ctx: *driver.Context,
    enc_spv: []const u8,
    dec_spv: []const u8,
    src: []const u8,
    dst: []u8,
) !RoundTripResult {
    std.debug.assert(dst.len >= src.len);

    var enc = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
    defer l1_codec.freeStreams(ctx, &enc.streams);

    try l1_codec.decodeL1Sync(ctx, enc.streams, src.len, dst[0..src.len], dec_spv);

    var first_diff: usize = 0;
    var total_diffs: usize = 0;
    var found_first = false;
    var i: usize = 0;
    while (i < src.len) : (i += 1) {
        if (src[i] != dst[i]) {
            if (!found_first) {
                first_diff = i;
                found_first = true;
            }
            total_diffs += 1;
        }
    }
    return .{
        .pass = (total_diffs == 0),
        .first_diff = first_diff,
        .total_diffs = total_diffs,
        .comp_bytes = enc.comp_total_bytes,
    };
}

fn printResult(
    w: *std.Io.Writer,
    label: []const u8,
    src_len: usize,
    r: RoundTripResult,
) !void {
    if (r.pass) {
        const ratio_pct: f64 = if (src_len > 0)
            (100.0 * @as(f64, @floatFromInt(r.comp_bytes))) / @as(f64, @floatFromInt(src_len))
        else
            0.0;
        try w.print(
            "L1_ROUNDTRIP PASS {s} bytes={d} comp_bytes={d} ratio={d:.1}%\n",
            .{ label, src_len, r.comp_bytes, ratio_pct },
        );
    } else {
        try w.print(
            "L1_ROUNDTRIP FAIL {s} bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src_len, r.first_diff, r.total_diffs },
        );
    }
}

// ── main ──────────────────────────────────────────────────────────

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // 1. Bring up the loader + instance + device + queue.
    try driver.ensureInit();
    defer driver.deinit();
    const ctx = &driver.g_default;

    // 2. Probe the tier — needed for the SPV path component.
    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    const tier_name = probeTierName(pr.tier) orelse {
        try w.print("L1_ROUNDTRIP FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    // 3. Load both SPVs. The dest buffers are 1-MiB stack arrays —
    //    the M8c shaders top out ~10 KiB so this is generous.
    var enc_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    var dec_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const enc_spv = loadSpv(io, "lz_encode", tier_name, &enc_spv_storage) catch |err| {
        try w.print("L1_ROUNDTRIP FAIL enc_spv err={s} tier={s}\n", .{ @errorName(err), tier_name });
        return err;
    };
    const dec_spv = loadSpv(io, "lz_decode", tier_name, &dec_spv_storage) catch |err| {
        try w.print("L1_ROUNDTRIP FAIL dec_spv err={s} tier={s}\n", .{ @errorName(err), tier_name });
        return err;
    };

    try w.print(
        "L1 codec round-trip test — tier={s} device=\"{s}\" subgroup={d}\n",
        .{ tier_name, pr.device_name, pr.subgroup_size },
    );
    try w.flush();

    // ── Case 1: 256 B repetitive ABCDABCD ──────────────────────────
    var small_src: [SMALL_SIZE]u8 = undefined;
    var small_dst: [SMALL_SIZE]u8 = undefined;
    fillRepetitive(small_src[0..]);

    const small_r = try roundTrip(ctx, enc_spv, dec_spv, small_src[0..], small_dst[0..]);
    try printResult(w, "small_repetitive", SMALL_SIZE, small_r);
    try w.flush();

    if (!small_r.pass) {
        // Don't run the large case if the small one already failed — the
        // smaller failure mode is the productive one to debug first.
        return error.Mismatch;
    }

    // ── Case 2: 64 KiB mixed (repetitive head + corpus tail) ───────
    // 64 KiB stack allocations (one each for src + dst) keep the test
    // self-contained without an allocator. Total = 128 KiB extra on top
    // of the SPV storage; main()'s 4-MB Windows default stack covers it.
    var large_src: [LARGE_SIZE]u8 = undefined;
    var large_dst: [LARGE_SIZE]u8 = undefined;
    fillLargeCase(io, large_src[0..]);

    const large_r = try roundTrip(ctx, enc_spv, dec_spv, large_src[0..], large_dst[0..]);
    try printResult(w, "large_mixed", LARGE_SIZE, large_r);
    try w.flush();

    // DEBUG: if the large case failed, dump 32 src/dst bytes around first_diff
    // to help eyeball whether the issue is offset drift, garbage data, or zero
    // fill — all three look different at a glance.
    if (!large_r.pass) {
        const d = large_r.first_diff;
        const start = if (d >= 16) d - 16 else 0;
        const end = if (d + 16 < LARGE_SIZE) d + 16 else LARGE_SIZE;
        try w.print("DEBUG src[{d}..{d}] = ", .{ start, end });
        for (large_src[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG dst[{d}..{d}] = ", .{ start, end });
        for (large_dst[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 3: pure corpus (no ABCD head) ─────────────────────────
    // Helps isolate whether the failure is at the repetitive→corpus
    // boundary or in the corpus path itself.
    var corpus_src: [LARGE_SIZE]u8 = undefined;
    var corpus_dst: [LARGE_SIZE]u8 = undefined;
    if (!fillFromCorpus(io, corpus_src[0..])) fillDeterministic(corpus_src[0..], 0x1234);
    const corpus_r = try roundTrip(ctx, enc_spv, dec_spv, corpus_src[0..], corpus_dst[0..]);
    try printResult(w, "pure_corpus", LARGE_SIZE, corpus_r);
    try w.flush();

    if (!corpus_r.pass) {
        const d = corpus_r.first_diff;
        const start = if (d >= 16) d - 16 else 0;
        const end = if (d + 16 < LARGE_SIZE) d + 16 else LARGE_SIZE;
        try w.print("DEBUG corpus src[{d}..{d}] = ", .{ start, end });
        for (corpus_src[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG corpus dst[{d}..{d}] = ", .{ start, end });
        for (corpus_dst[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 4: ALL-ZERO src ──────────────────────────────────────
    // Should compress to nearly nothing. Tests basic match handling.
    var zero_src: [256]u8 = [_]u8{0} ** 256;
    var zero_dst: [256]u8 = undefined;
    const zero_r = try roundTrip(ctx, enc_spv, dec_spv, zero_src[0..], zero_dst[0..]);
    try printResult(w, "all_zero", 256, zero_r);
    try w.flush();
    if (!zero_r.pass) {
        const d = zero_r.first_diff;
        const start = if (d >= 8) d - 8 else 0;
        const end = if (d + 16 < 256) d + 16 else 256;
        try w.print("DEBUG zero dst[{d}..{d}] = ", .{ start, end });
        for (zero_dst[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 5: 32-byte literal sequence (forces pure-literal path) ──
    // 32 distinct bytes with no chance of LZ matches. Tests the trailing-
    // literal emit path for short inputs.
    var lit32_src: [32]u8 = undefined;
    var lit32_dst: [32]u8 = undefined;
    for (lit32_src[0..], 0..) |*b, i| b.* = @intCast(i);
    const lit32_r = try roundTrip(ctx, enc_spv, dec_spv, lit32_src[0..], lit32_dst[0..]);
    try printResult(w, "lit32_sequence", 32, lit32_r);
    try w.flush();
    if (!lit32_r.pass) {
        try w.print("DEBUG lit32 src = ", .{});
        for (lit32_src[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG lit32 dst = ", .{});
        for (lit32_dst[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 6: small corpus prefix (128 bytes) ────────────────────
    var corpus128_src: [128]u8 = undefined;
    var corpus128_dst: [128]u8 = undefined;
    if (!fillFromCorpus(io, corpus128_src[0..])) fillDeterministic(corpus128_src[0..], 0x1234);

    // Dedicated stream-dumping run for corpus_128 — needs separate encode
    // because roundTrip frees the streams before returning.
    {
        var enc = try l1_codec.encodeL1Sync(ctx, corpus128_src[0..], enc_spv);
        defer l1_codec.freeStreams(ctx, &enc.streams);

        var cmd_buf: [256]u8 = undefined;
        var lit_buf: [256]u8 = undefined;
        var off16_buf: [256]u8 = undefined;
        var length_buf: [256]u8 = undefined;
        const ns = l1_codec.debugReadStreams(ctx, enc.streams, &cmd_buf, &lit_buf, &off16_buf, &length_buf);
        try w.print("DEBUG corpus128 cmd[0..{d}] = ", .{ns.cmd_n});
        for (cmd_buf[0..ns.cmd_n]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.print("DEBUG corpus128 lit[0..{d}] = ", .{ns.lit_n});
        for (lit_buf[0..ns.lit_n]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.print("DEBUG corpus128 off16[0..{d}] = ", .{ns.off16_n});
        for (off16_buf[0..ns.off16_n]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.print("DEBUG corpus128 length[0..{d}] = ", .{ns.length_n});
        for (length_buf[0..ns.length_n]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    const corpus128_r = try roundTrip(ctx, enc_spv, dec_spv, corpus128_src[0..], corpus128_dst[0..]);
    try printResult(w, "corpus_128", 128, corpus128_r);
    try w.flush();
    if (!corpus128_r.pass) {
        try w.print("DEBUG corpus128 src = ", .{});
        for (corpus128_src[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG corpus128 dst = ", .{});
        for (corpus128_dst[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 7: small corpus prefix (64 bytes) ─────────────────────
    var corpus64_src: [64]u8 = undefined;
    var corpus64_dst: [64]u8 = undefined;
    if (!fillFromCorpus(io, corpus64_src[0..])) fillDeterministic(corpus64_src[0..], 0x1234);
    const corpus64_r = try roundTrip(ctx, enc_spv, dec_spv, corpus64_src[0..], corpus64_dst[0..]);
    try printResult(w, "corpus_64", 64, corpus64_r);
    try w.flush();
    if (!corpus64_r.pass) {
        try w.print("DEBUG corpus64 src = ", .{});
        for (corpus64_src[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG corpus64 dst = ", .{});
        for (corpus64_dst[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 8: corpus prefix 96 ─────────────────────────────────
    var corpus96_src: [96]u8 = undefined;
    var corpus96_dst: [96]u8 = undefined;
    if (!fillFromCorpus(io, corpus96_src[0..])) fillDeterministic(corpus96_src[0..], 0x1234);
    const corpus96_r = try roundTrip(ctx, enc_spv, dec_spv, corpus96_src[0..], corpus96_dst[0..]);
    try printResult(w, "corpus_96", 96, corpus96_r);
    try w.flush();
    if (!corpus96_r.pass) {
        try w.print("DEBUG corpus96 src = ", .{});
        for (corpus96_src[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG corpus96 dst = ", .{});
        for (corpus96_dst[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 9: corpus prefix 112 ───────────────────────────────
    var corpus112_src: [112]u8 = undefined;
    var corpus112_dst: [112]u8 = undefined;
    if (!fillFromCorpus(io, corpus112_src[0..])) fillDeterministic(corpus112_src[0..], 0x1234);
    const corpus112_r = try roundTrip(ctx, enc_spv, dec_spv, corpus112_src[0..], corpus112_dst[0..]);
    try printResult(w, "corpus_112", 112, corpus112_r);
    try w.flush();
    if (!corpus112_r.pass) {
        try w.print("DEBUG corpus112 src = ", .{});
        for (corpus112_src[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG corpus112 dst = ", .{});
        for (corpus112_dst[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Case 10: corpus prefix 120 ───────────────────────────────
    var corpus120_src: [120]u8 = undefined;
    var corpus120_dst: [120]u8 = undefined;
    if (!fillFromCorpus(io, corpus120_src[0..])) fillDeterministic(corpus120_src[0..], 0x1234);
    const corpus120_r = try roundTrip(ctx, enc_spv, dec_spv, corpus120_src[0..], corpus120_dst[0..]);
    try printResult(w, "corpus_120", 120, corpus120_r);
    try w.flush();
    if (!corpus120_r.pass) {
        try w.print("DEBUG corpus120 src = ", .{});
        for (corpus120_src[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG corpus120 dst = ", .{});
        for (corpus120_dst[0..]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
        try w.flush();
    }

    // ── Cases 11-15: corpus prefix size sweep ────────────────────
    inline for ([_]usize{ 622, 626, 1024, 4096, 16384, 32768 }) |sz| {
        var s: [sz]u8 = undefined;
        var d: [sz]u8 = undefined;
        if (!fillFromCorpus(io, s[0..])) fillDeterministic(s[0..], 0x1234);
        const r = try roundTrip(ctx, enc_spv, dec_spv, s[0..], d[0..]);
        var label_buf: [32]u8 = undefined;
        const label = try std.fmt.bufPrint(&label_buf, "corpus_{d}", .{sz});
        try printResult(w, label, sz, r);
        try w.flush();
    }


    // Phase-1 evidence: the round-trip test exercises every code path
    // (literal-only, single match, multi-match, fast path, slow path)
    // across a size sweep.
}
