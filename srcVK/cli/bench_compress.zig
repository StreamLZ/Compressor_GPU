//! 1:1 port of src/cli/bench_compress.zig.
//!
//! `streamlz_vk -b <file>` — compress, decompress, verify the round-trip.
//! Reports compress wall-clock, per-run decompress times, decompress
//! median + GPU-kernel breakdown when timings are available.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_enc_driver = @import("../encode/driver.zig");
const gpu_dec_driver = @import("../decode/driver.zig");
// VK adaptation: per-phase QPC profile for the encode hot path lives on
// fast_framed.zig (the encode orchestrator). Wire init + print here so
// SLZ_VK_PROFILE_PHASES=1 localizes the per-phase breakdown the same
// way the decode side does (phaseProfileInit / printAndResetPhaseProfile).
const enc_phase = @import("../encode/fast_framed.zig");

/// CUDA reference: src/cli/bench_compress.zig:12-end. -b mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);
    const runs = args.runs orelse 3;
    util.checkLevel(args.level, w);
    if (runs == 0) util.die(w, "error: runs must be >= 1\n");

    const src_raw = util.readFile(allocator, io, in_path, w);
    defer allocator.free(src_raw);

    // VK adaptation (H3): pin the encode input through procMallocHost so
    // its base address is page-aligned. iter-12's
    // VK_EXT_external_memory_host import inside procH2DAsync requires the
    // surrounding page-aligned region be owned by the caller — readFileAlloc
    // returns a non-page-aligned slice from the general-purpose allocator
    // (rejected by tryImportHostBuffer). Mirrors bench_decompress.zig
    // lines 30-43 for the decompress src pin. Falls back to a regular
    // allocation (and the iter-4 staging path inside procH2DAsync) when
    // the GPU host-alloc slot is unavailable.
    const SrcBuf = struct { buf: []u8, pinned: bool };
    const sh: SrcBuf = blk: {
        if (gpu_dec_driver.allocHost(src_raw.len)) |p| {
            @memcpy(p[0..src_raw.len], src_raw);
            break :blk .{ .buf = p[0..src_raw.len], .pinned = true };
        }
        const dup = allocator.alloc(u8, src_raw.len) catch break :blk .{ .buf = @constCast(src_raw), .pinned = false };
        @memcpy(dup, src_raw);
        break :blk .{ .buf = dup, .pinned = false };
    };
    const src: []const u8 = sh.buf;
    defer if (sh.pinned) gpu_dec_driver.freeHost(sh.buf) else if (sh.buf.ptr != src_raw.ptr) allocator.free(sh.buf);

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
        .dictionary_id = util.resolveDictionary(allocator, io, args.dictionary, in_path, w, &gpu_enc_driver.g_default),
    };

    // VK adaptation (H3): reset the phase / import-cache counters after
    // the warm-up compress so only the measured run accumulates. Mirrors
    // bench_decompress.zig's phaseProfileInit() pattern; same global
    // counters (g_h2d_path_* + g_import_cache_*) live in
    // decode/module_loader.zig and are shared by both procH2DAsync paths.
    var comp_size: usize = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_enc_driver.g_default);
    gpu_dec_driver.phaseProfileInit();
    // VK adaptation: reset the encode-side per-phase accumulators after
    // the untimed warm-up so only the timed Compress: run accumulates.
    // Mirrors the decode pattern (phaseProfileInit/printAndResetPhaseProfile).
    enc_phase.encPhaseProfileInit();
    {
        const t0 = std.Io.Clock.awake.now(io);
        comp_size = try encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_enc_driver.g_default);
        const ns = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        const ms = @as(f64, @floatFromInt(ns)) / 1_000_000.0;
        try w.print("  Compress: {d:.0}ms ({d:.1} MB/s)\n", .{ ms, mb * 1000.0 / ms });
    }
    // VK adaptation (H3): surface h2d_paths + import_cache counters
    // immediately after the timed compress so SLZ_VK_PROFILE_PHASES=1
    // confirms the 95 MB input H2D took the inline_import branch (iter-12)
    // rather than falling back to the iter-4 staging copy. Printed BEFORE
    // the decode runs below so the encode-side numbers stand alone. The
    // generic phase printer (printAndResetPhaseProfile) gates on
    // g_phase_decode_count which is 0 here — use the encode-only variant.
    try w.flush();
    gpu_dec_driver.printEncodeImportTelemetry(w);
    // VK adaptation: print + reset the per-phase encode breakdown
    // (cpu_descs_build / wrap_input_h2d / lz.* / asm.* / frame.* / d2h_final
    // / host_finalize). Gated on SLZ_VK_PROFILE_PHASES env var by the
    // printer itself; no-op when unset.
    enc_phase.printAndResetEncodePhaseProfile(w);
    try w.flush();
    try w.print("Level {d}: {d} -> {d} bytes ({d:.1}%)\n\n", .{
        args.level, src.len, comp_size,
        @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(src.len)) * 100.0,
    });

    // CUDA-mirror (2026-06-10): trim the encoder's persistent device
    // buffers before the decompress phase. After a 1 GB L3 encode they
    // hold ~13 GB; the decoder needs ~7.5 GB more, which under strict
    // VMA is a hard OutOfDeviceMemory on a 16 GB card. The compressed
    // bytes are already on the host; a later encode re-allocates.
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
