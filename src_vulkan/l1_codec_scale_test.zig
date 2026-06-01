//! L1 codec large-corpus scale test.
//!
//! Exercises the Vulkan L1 codec at sizes well above the existing
//! l1_codec_test.zig surface (which tops out at ~4.5 MB / 35 chunks):
//!
//!   * enwik8 prefixes: 16 / 32 / 64 MiB, plus the full 100 MB file
//!   * silesia_all.tar slices: 16 / 64 MiB, plus the full 200 MB tar
//!
//! Each case: VK-encode -> VK-decode -> byte-equal compare. Prints a
//! PASS line with bytes, comp_bytes, compression ratio, and per-stage
//! ns/byte timings. Failures print first-diff offset and total
//! mismatches.
//!
//! Wired to `zig build vk-l1-scale-test` — invoked explicitly (NOT
//! folded into `zig build test`) because the full silesia case touches
//! 200 MB of input + the corresponding GPU buffers (~3 GB of host-
//! visible memory) and takes seconds to minutes per run.
//!
//! Reads SPV from `zig-out/shaders/` at runtime (same pattern as
//! l1_codec_test.zig). The build target adds the hard dependency on
//! the `vk-shaders` top-level step so the .spv files exist first.

const std = @import("std");

const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");

const MAX_SPV_BYTES: usize = 1 << 20; // 1 MiB
const SPV_DIR_REL: []const u8 = "zig-out/shaders";

const ENWIK8_PATH: []const u8 = "assets/enwik8.txt";
const SILESIA_PATH: []const u8 = "assets/silesia_all.tar";

// ── Win32 QPC for ns timing (matches l1_perf_bench.zig pattern) ─────

const win32 = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};

fn qpcNow() i64 {
    var c: i64 = 0;
    _ = win32.QueryPerformanceCounter(&c);
    return c;
}

fn qpcNs(from: i64, to: i64) u64 {
    var freq: i64 = 0;
    _ = win32.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

// ── SPV loader (same shape as l1_codec_test.zig) ────────────────────

fn probeTierName(t: probe_mod.Tier) ?[]const u8 {
    return switch (t) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => null,
    };
}

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

// ── One scale case ────────────────────────────────────────────────

const CaseResult = struct {
    pass: bool,
    bytes: usize,
    comp_bytes: u32,
    n_chunks: u32,
    encode_ns: u64,
    decode_ns: u64,
    first_diff: usize = 0,
    total_diffs: usize = 0,
    /// True if the input was loaded successfully. False = file open /
    /// short read; the case is reported as SKIP, not FAIL.
    loaded: bool = true,
    /// On encode failure (e.g. TooManyChunks, vkAllocateMemory failure),
    /// the error name. Empty string on success or roundtrip-mismatch.
    err_name: []const u8 = "",
};

/// Read up to `size_cap` bytes of `path` into a freshly allocated
/// buffer. Returns null on open failure; returns a short slice on
/// short read.
fn loadCorpusSlice(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    size_cap: usize,
) !?[]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    const stat = file.stat(io) catch return null;
    const stat_size: usize = @intCast(stat.size);
    if (stat_size == 0) return null;
    const want: usize = if (size_cap == 0) stat_size else @min(size_cap, stat_size);
    const buf = try allocator.alloc(u8, want);
    errdefer allocator.free(buf);
    const n = file.readPositionalAll(io, buf, 0) catch 0;
    if (n != want) {
        allocator.free(buf);
        return null;
    }
    return buf;
}

fn runOne(
    allocator: std.mem.Allocator,
    ctx: *driver.Context,
    enc_spv: []const u8,
    dec_spv: []const u8,
    src: []const u8,
) CaseResult {
    var r: CaseResult = .{
        .pass = false,
        .bytes = src.len,
        .comp_bytes = 0,
        .n_chunks = 0,
        .encode_ns = 0,
        .decode_ns = 0,
    };

    // Encode (timed).
    const t0 = qpcNow();
    var enc = l1_codec.encodeL1Sync(ctx, src, enc_spv) catch |err| {
        r.err_name = @errorName(err);
        return r;
    };
    const t1 = qpcNow();
    defer l1_codec.freeStreams(ctx, &enc.streams);
    r.encode_ns = qpcNs(t0, t1);
    r.comp_bytes = enc.comp_total_bytes;
    r.n_chunks = enc.streams.n_chunks;

    // Decode (timed). Allocate dst once.
    const dst = allocator.alloc(u8, src.len) catch |err| {
        r.err_name = @errorName(err);
        return r;
    };
    defer allocator.free(dst);
    @memset(dst, 0);

    const t2 = qpcNow();
    l1_codec.decodeL1Sync(ctx, enc.streams, src.len, dst, dec_spv) catch |err| {
        r.err_name = @errorName(err);
        return r;
    };
    const t3 = qpcNow();
    r.decode_ns = qpcNs(t2, t3);

    // Compare.
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
    r.first_diff = first_diff;
    r.total_diffs = total_diffs;
    r.pass = (total_diffs == 0);
    return r;
}

fn printResult(w: *std.Io.Writer, label: []const u8, r: CaseResult) !void {
    if (!r.loaded) {
        try w.print("L1_SCALE SKIP {s} (load fail)\n", .{label});
        return;
    }
    if (r.err_name.len != 0) {
        try w.print(
            "L1_SCALE FAIL {s} bytes={d} err={s}\n",
            .{ label, r.bytes, r.err_name },
        );
        return;
    }
    if (r.pass) {
        const ratio_pct: f64 = if (r.bytes > 0)
            (100.0 * @as(f64, @floatFromInt(r.comp_bytes))) / @as(f64, @floatFromInt(r.bytes))
        else
            0.0;
        const enc_ns_per_b: f64 = if (r.bytes > 0)
            @as(f64, @floatFromInt(r.encode_ns)) / @as(f64, @floatFromInt(r.bytes))
        else
            0.0;
        const dec_ns_per_b: f64 = if (r.bytes > 0)
            @as(f64, @floatFromInt(r.decode_ns)) / @as(f64, @floatFromInt(r.bytes))
        else
            0.0;
        try w.print(
            "L1_SCALE PASS {s} bytes={d} chunks={d} comp_bytes={d} ratio={d:.1}% enc_ns={d} dec_ns={d} enc_ns/B={d:.2} dec_ns/B={d:.2}\n",
            .{ label, r.bytes, r.n_chunks, r.comp_bytes, ratio_pct, r.encode_ns, r.decode_ns, enc_ns_per_b, dec_ns_per_b },
        );
    } else {
        try w.print(
            "L1_SCALE FAIL {s} bytes={d} chunks={d} comp_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, r.bytes, r.n_chunks, r.comp_bytes, r.first_diff, r.total_diffs },
        );
    }
}

const CaseSpec = struct {
    label: []const u8,
    path: []const u8,
    size_cap: usize, // 0 = read full file
};

// ── main ─────────────────────────────────────────────────────────

pub fn main(process_init: std.process.Init) !void {
    const allocator = process_init.gpa;
    const io = process_init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    try driver.ensureInit();
    defer driver.deinit();
    const ctx = &driver.g_default;

    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    const tier_name = probeTierName(pr.tier) orelse {
        try w.print("L1_SCALE FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    var enc_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    var dec_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const enc_spv = try loadSpv(io, "lz_encode", tier_name, &enc_spv_storage);
    const dec_spv = try loadSpv(io, "lz_decode", tier_name, &dec_spv_storage);

    try w.print("L1 codec scale test — tier={s} subgroup={d}\n", .{ tier_name, pr.subgroup_size });
    try w.flush();

    // Case order: ascending size. enwik8 first (text, well-compressible),
    // then silesia (binary mix, harder). For each, run the size sweep.
    const cases = [_]CaseSpec{
        // enwik8 prefixes + full file (100 MB).
        .{ .label = "enwik8_first_16mb", .path = ENWIK8_PATH, .size_cap = 16 * 1024 * 1024 },
        .{ .label = "enwik8_first_32mb", .path = ENWIK8_PATH, .size_cap = 32 * 1024 * 1024 },
        .{ .label = "enwik8_first_64mb", .path = ENWIK8_PATH, .size_cap = 64 * 1024 * 1024 },
        .{ .label = "enwik8_full", .path = ENWIK8_PATH, .size_cap = 0 },
        // silesia (binary-heavy) prefixes + full tar (200 MB). The smaller
        // 1/4/8/11 MiB cases narrow down the chunk where the binary-data
        // failure first appears — observed first_diff is at byte 10,206,968
        // (chunk 77 interior) when running silesia_first_16mb, and the
        // smaller cases either pass or fail in the same chunk depending
        // on whether the chunk is included.
        .{ .label = "silesia_first_1mb", .path = SILESIA_PATH, .size_cap = 1 * 1024 * 1024 },
        .{ .label = "silesia_first_4mb", .path = SILESIA_PATH, .size_cap = 4 * 1024 * 1024 },
        .{ .label = "silesia_first_8mb", .path = SILESIA_PATH, .size_cap = 8 * 1024 * 1024 },
        .{ .label = "silesia_first_10mb", .path = SILESIA_PATH, .size_cap = 10 * 1024 * 1024 },
        .{ .label = "silesia_first_11mb", .path = SILESIA_PATH, .size_cap = 11 * 1024 * 1024 },
        .{ .label = "silesia_first_16mb", .path = SILESIA_PATH, .size_cap = 16 * 1024 * 1024 },
        .{ .label = "silesia_first_64mb", .path = SILESIA_PATH, .size_cap = 64 * 1024 * 1024 },
        .{ .label = "silesia_full", .path = SILESIA_PATH, .size_cap = 0 },
    };

    // Also include a 1 MiB + 4 MiB enwik8 entry to ground the perf table
    // — these overlap l1_codec_test.zig coverage but anchor the low end
    // of the size sweep in the same harness, same warmup state, same
    // timing path. Without them the perf table starts at 16 MiB and the
    // user has to cross-reference two test outputs to see the trend.
    const baseline_cases = [_]CaseSpec{
        .{ .label = "enwik8_first_1mb", .path = ENWIK8_PATH, .size_cap = 1 * 1024 * 1024 },
        .{ .label = "enwik8_first_4mb", .path = ENWIK8_PATH, .size_cap = 4 * 1024 * 1024 },
    };

    // ── Warmup: one tiny pass so we don't pay shader-create cost in
    //    the first measured case. (encodeL1Sync builds its descriptor
    //    cache lazily; first call always pays one pipeline-create.)
    {
        var tiny = [_]u8{ 'A', 'B', 'C', 'D' } ** 64; // 256 B
        var tiny_dst: [256]u8 = undefined;
        var enc = try l1_codec.encodeL1Sync(ctx, tiny[0..], enc_spv);
        defer l1_codec.freeStreams(ctx, &enc.streams);
        try l1_codec.decodeL1Sync(ctx, enc.streams, tiny.len, tiny_dst[0..], dec_spv);
    }

    // ── Run baseline (1 / 4 MiB) cases first ──────────────────────
    for (baseline_cases) |c| {
        const src_opt = loadCorpusSlice(allocator, io, c.path, c.size_cap) catch null;
        if (src_opt) |src| {
            defer allocator.free(src);
            const r = runOne(allocator, ctx, enc_spv, dec_spv, src);
            try printResult(w, c.label, r);
        } else {
            try w.print("L1_SCALE SKIP {s} (load fail)\n", .{c.label});
        }
        try w.flush();
    }

    // ── Run the big cases ──────────────────────────────────────────
    for (cases) |c| {
        const src_opt = loadCorpusSlice(allocator, io, c.path, c.size_cap) catch |err| blk: {
            try w.print("L1_SCALE FAIL {s} load_err={s}\n", .{ c.label, @errorName(err) });
            break :blk null;
        };
        if (src_opt) |src| {
            defer allocator.free(src);
            const r = runOne(allocator, ctx, enc_spv, dec_spv, src);
            try printResult(w, c.label, r);
        } else {
            try w.print("L1_SCALE SKIP {s} (load fail)\n", .{c.label});
        }
        try w.flush();
    }

    try w.print("L1_SCALE done.\n", .{});
}
