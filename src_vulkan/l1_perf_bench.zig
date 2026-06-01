//! Quick wall-clock perf measurement for the Vulkan L1 codec.
//!
//! Reads assets/web.txt once, runs encode + decode through the Vulkan
//! path, and prints `encode_ns_per_byte=X decode_ns_per_byte=Y` lines.
//! Used as the before/after measurement aid for piece 2 (decoder
//! fast-batch) and piece 3 (encoder warp-parallel match extension).
//!
//! Loads SPV via spv_blobs (compile-time @embedFile), so the binary is
//! self-contained — same SPV loading path as production CLI.

const std = @import("std");

const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const spv_blobs = @import("spv_blobs");

const CORPUS_PATH: []const u8 = "assets/web.txt";

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

fn tierName(t: probe_mod.Tier) ?spv_blobs.Tier {
    return switch (t) {
        .tier1 => .tier1,
        .tier1_nv => .tier1_nv,
        .tier2 => .tier2,
        .unsupported => null,
    };
}

pub fn main(process_init: std.process.Init) !void {
    const allocator = process_init.gpa;
    const io = process_init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // Load corpus.
    var file = std.Io.Dir.cwd().openFile(io, CORPUS_PATH, .{}) catch |err| {
        try w.print("error: cannot open {s}: {s}\n", .{ CORPUS_PATH, @errorName(err) });
        return;
    };
    defer file.close(io);
    const stat = try file.stat(io);
    const size: usize = @intCast(stat.size);
    const src = try allocator.alloc(u8, size);
    defer allocator.free(src);
    const n_read = try file.readPositionalAll(io, src, 0);
    if (n_read != size) {
        try w.print("error: short read on {s}\n", .{CORPUS_PATH});
        return;
    }

    try driver.ensureInit();
    defer driver.deinit();

    const ctx = &driver.g_default;
    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    const tier_b = tierName(pr.tier) orelse {
        try w.writeAll("error: unsupported tier\n");
        return;
    };

    const enc_spv_raw = spv_blobs.find("lz_encode", tier_b) orelse {
        try w.writeAll("error: no lz_encode spv\n");
        return;
    };
    const dec_spv_raw = spv_blobs.find("lz_decode", tier_b) orelse {
        try w.writeAll("error: no lz_decode spv\n");
        return;
    };
    // SPV needs 4-byte alignment for vkCreateShaderModule. @embedFile()
    // returns alignment-1 data; copy into aligned buffers.
    const enc_spv = try allocator.alignedAlloc(u8, .@"4", enc_spv_raw.len);
    defer allocator.free(enc_spv);
    @memcpy(enc_spv, enc_spv_raw);
    const dec_spv = try allocator.alignedAlloc(u8, .@"4", dec_spv_raw.len);
    defer allocator.free(dec_spv);
    @memcpy(dec_spv, dec_spv_raw);

    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);

    // Warmup: one encode + decode pass (pipeline create, JIT warm-up).
    {
        var enc = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
        defer l1_codec.freeStreams(ctx, &enc.streams);
        try l1_codec.decodeL1Sync(ctx, enc.streams, src.len, dst[0..src.len], dec_spv);
    }

    // Measured encode: 3 trials, take min.
    var encode_ns_min: u64 = std.math.maxInt(u64);
    var trial: u32 = 0;
    while (trial < 3) : (trial += 1) {
        const t0 = qpcNow();
        var enc = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
        const t1 = qpcNow();
        defer l1_codec.freeStreams(ctx, &enc.streams);
        const dt = qpcNs(t0, t1);
        if (dt < encode_ns_min) encode_ns_min = dt;
    }

    // Measured decode: 3 trials, take min. Each trial needs its own
    // encode result since decodeL1Sync consumes streams from a fresh
    // pipeline run. Encode once outside loop to keep encode noise out.
    var enc_for_decode = try l1_codec.encodeL1Sync(ctx, src, enc_spv);
    defer l1_codec.freeStreams(ctx, &enc_for_decode.streams);
    var decode_ns_min: u64 = std.math.maxInt(u64);
    trial = 0;
    while (trial < 3) : (trial += 1) {
        @memset(dst, 0);
        const t0 = qpcNow();
        try l1_codec.decodeL1Sync(ctx, enc_for_decode.streams, src.len, dst[0..src.len], dec_spv);
        const t1 = qpcNow();
        const dt = qpcNs(t0, t1);
        if (dt < decode_ns_min) decode_ns_min = dt;
    }

    const enc_ns_per_byte: f64 = @as(f64, @floatFromInt(encode_ns_min)) / @as(f64, @floatFromInt(src.len));
    const dec_ns_per_byte: f64 = @as(f64, @floatFromInt(decode_ns_min)) / @as(f64, @floatFromInt(src.len));

    try w.print(
        "VK_L1_PERF src_bytes={d} encode_ns={d} decode_ns={d} encode_ns_per_byte={d:.3} decode_ns_per_byte={d:.3}\n",
        .{ src.len, encode_ns_min, decode_ns_min, enc_ns_per_byte, dec_ns_per_byte },
    );
}
