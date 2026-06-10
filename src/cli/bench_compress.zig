//! `streamlz -b <file>` — compress, decompress, verify the round-trip.
//! Reports compress wall-clock, per-run decompress times, decompress
//! median + GPU-kernel breakdown when timings are available.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_enc_driver = @import("../encode/driver.zig");
const gpu_dec_driver = @import("../decode/driver.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);
    const runs = args.runs orelse 3;
    util.checkLevel(args.level, w);
    if (runs == 0) util.die(w, "error: runs must be >= 1\n");

    const src = util.readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(decompressed);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("Input: {s} ({d} bytes, {d:.2} MB)\n", .{ in_path, src.len, mb });

    const comp_opts: encoder.Options = .{
        .level = args.level,
        .sc_group_size_override = args.sc_group,
    };

    var comp_size: usize = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_enc_driver.g_default);
    {
        const t0 = std.Io.Clock.awake.now(io);
        comp_size = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_enc_driver.g_default);
        const ns = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        try w.print("  Compress: {d:.0}ms ({d:.1} MB/s)\n", .{ ms, mb * 1000.0 / ms });
    }
    try w.print("Level {d}: {d} -> {d} bytes ({d:.1}%)\n\n", .{
        args.level, src.len, comp_size,
        @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(src.len)) * 100.0,
    });

    // 2026-06-10: trim the encoder's persistent device buffers before
    // the decompress phase. After a 1 GB L3 encode they hold ~13 GB
    // (hash + LZ output + input + wrap buffers); the decoder needs
    // ~7.5 GB more. Vulkan's strict allocator fails outright; CUDA's
    // WDDM pages silently and pollutes the decompress timings (the
    // same frame measured up to 30× slower here than via `-db`). The
    // compressed bytes are already on the host; nothing below touches
    // the encoder, and a later encode would transparently re-allocate.
    gpu_enc_driver.g_default.releaseDeviceBuffers();

    var dec_ctx = decoder.DecompressContext.initWithIo(allocator, io);
    defer dec_ctx.deinit();

    _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed); // warm-up

    const dec_times = try allocator.alloc(u64, runs);
    defer allocator.free(dec_times);
    const kern_times = try allocator.alloc(i64, runs);
    defer allocator.free(kern_times);
    @memset(kern_times, 0);

    var r: u32 = 0;
    while (r < runs) : (r += 1) {
        const t0 = std.Io.Clock.awake.now(io);
        _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
        dec_times[r] = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        kern_times[r] = gpu_dec_driver.last_kernel_ns;
        const ms = @as(f64, @floatFromInt(dec_times[r])) / 1_000_000.0;
        const kms = @as(f64, @floatFromInt(kern_times[r])) / 1_000_000.0;
        if (kern_times[r] > 0) {
            try w.print("  Decompress run {d}: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n", .{
                r + 1, ms, mb * 1000.0 / ms, kms, mb * 1000.0 / kms,
            });
        } else {
            try w.print("  Decompress run {d}: {d:.0}ms ({d:.1} MB/s)\n", .{ r + 1, ms, mb * 1000.0 / ms });
        }
    }

    const dec_med_ns = util.medianOrMean(dec_times);
    const dec_med_ms = @as(f64, @floatFromInt(dec_med_ns)) / 1_000_000.0;
    const nonzero_kerns = try allocator.alloc(i64, runs);
    defer allocator.free(nonzero_kerns);
    var nz_count: usize = 0;
    for (kern_times) |k| if (k > 0) {
        nonzero_kerns[nz_count] = k;
        nz_count += 1;
    };
    if (nz_count > 0) {
        std.mem.sort(i64, nonzero_kerns[0..nz_count], {}, std.sort.asc(i64));
        const kms = @as(f64, @floatFromInt(nonzero_kerns[nz_count / 2])) / 1_000_000.0;
        try w.print("  Decompress median: e2e {d:.0}ms ({d:.1} MB/s)  d2d {d:.2}ms ({d:.1} MB/s)\n\n", .{
            dec_med_ms, mb * 1000.0 / dec_med_ms, kms, mb * 1000.0 / kms,
        });
    } else {
        try w.print("  Decompress median: {d:.0}ms ({d:.1} MB/s)\n\n", .{ dec_med_ms, mb * 1000.0 / dec_med_ms });
    }

    if (std.mem.eql(u8, src, decompressed[0..src.len])) {
        try w.writeAll("Round-trip: PASS\n");
    } else {
        var first_fail: usize = 0;
        var fail_count: usize = 0;
        for (0..src.len) |bi| if (src[bi] != decompressed[bi]) {
            if (fail_count == 0) first_fail = bi;
            fail_count += 1;
        };
        try w.print("Round-trip: FAIL  first_diff={d} total_diffs={d}\n", .{ first_fail, fail_count });
        try w.flush();
        std.process.exit(1);
    }
}
