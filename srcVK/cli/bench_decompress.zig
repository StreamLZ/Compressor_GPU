//! 1:1 port of src/cli/bench_decompress.zig.
//!
//! `streamlz_vk -db <file>` — decompress-only benchmark with best+mean
//! wall-clock and per-kernel breakdown (GPU total, LZ, Huff) reported
//! when the decode driver populates last_*_kernel_ns.

const std = @import("std");
const util = @import("util.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_dec_driver = @import("../decode/driver.zig");
const frame = @import("../format/frame_format.zig");

/// CUDA reference: src/cli/bench_decompress.zig:11-end. -db mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);
    const runs = args.runs orelse 10;
    if (runs == 0) util.die(w, "error: runs must be >= 1\n");

    const src = util.readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const hdr = frame.parseHeader(src) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    const content_size: usize = if (hdr.content_size) |cs| @intCast(cs) else {
        try w.writeAll("error: frame has no content size; bench needs a sized frame\n");
        try w.flush();
        std.process.exit(1);
    };

    // Pin the output buffer so the D2H runs at full PCIe bandwidth.
    // Falls back to a normal allocation when the GPU host-alloc path is
    // unavailable, so the CLI still works for inspection without a GPU.
    const dst_size = content_size + decoder.safe_space;
    const DstBuf = struct { buf: []u8, pinned: bool };
    const dh: DstBuf = blk: {
        if (gpu_dec_driver.allocHost(dst_size)) |p| break :blk .{ .buf = p, .pinned = true };
        break :blk .{ .buf = allocator.alloc(u8, dst_size) catch |err| {
            try w.print("error: cannot allocate {d} bytes: {s}\n", .{ dst_size, @errorName(err) });
            try w.flush();
            std.process.exit(1);
        }, .pinned = false };
    };
    const dst = dh.buf;
    defer if (dh.pinned) gpu_dec_driver.freeHost(dst) else allocator.free(dst);

    var dec_ctx = decoder.DecompressContext.initWithIo(allocator, io);
    defer dec_ctx.deinit();

    _ = dec_ctx.decompress(src, dst) catch |err| {
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var best_ns: u64 = std.math.maxInt(u64);
    var total_ns: u64 = 0;
    var best_kern_ns: i64 = std.math.maxInt(i64);
    var total_kern_ns: i64 = 0;
    var best_lz_ns: i64 = std.math.maxInt(i64);
    var total_lz_ns: i64 = 0;
    var best_huff_ns: i64 = std.math.maxInt(i64);
    var total_huff_ns: i64 = 0;

    var run_i: u32 = 0;
    while (run_i < runs) : (run_i += 1) {
        const t0 = std.Io.Clock.awake.now(io);
        _ = try dec_ctx.decompress(src, dst);
        const elapsed = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
        if (elapsed < best_ns) best_ns = elapsed;
        total_ns += elapsed;
        if (gpu_dec_driver.last_kernel_ns > 0) {
            if (gpu_dec_driver.last_kernel_ns < best_kern_ns) best_kern_ns = gpu_dec_driver.last_kernel_ns;
            total_kern_ns += gpu_dec_driver.last_kernel_ns;
        }
        if (gpu_dec_driver.last_lz_kernel_ns > 0) {
            if (gpu_dec_driver.last_lz_kernel_ns < best_lz_ns) best_lz_ns = gpu_dec_driver.last_lz_kernel_ns;
            total_lz_ns += gpu_dec_driver.last_lz_kernel_ns;
        }
        if (gpu_dec_driver.last_huff_kernel_ns > 0) {
            if (gpu_dec_driver.last_huff_kernel_ns < best_huff_ns) best_huff_ns = gpu_dec_driver.last_huff_kernel_ns;
            total_huff_ns += gpu_dec_driver.last_huff_kernel_ns;
        }
    }

    const mean_ns: u64 = total_ns / runs;
    const mb: f64 = @as(f64, @floatFromInt(content_size)) / (1024.0 * 1024.0);
    try w.print("bench: {s}\n", .{in_path});
    try w.print("  src bytes:       {d}\n", .{src.len});
    try w.print("  decompressed:    {d} ({d:.2} MB)\n", .{ content_size, mb });
    try w.print("  runs:            {d} (plus 1 warm-up)\n", .{runs});
    try w.print("  best:            {d:.3} ms  ({d:.0} MB/s)\n", .{
        @as(f64, @floatFromInt(best_ns)) / 1_000_000.0, mb * 1e9 / @as(f64, @floatFromInt(best_ns)),
    });
    try w.print("  mean:            {d:.3} ms  ({d:.0} MB/s)\n", .{
        @as(f64, @floatFromInt(mean_ns)) / 1_000_000.0, mb * 1e9 / @as(f64, @floatFromInt(mean_ns)),
    });
    if (best_kern_ns < std.math.maxInt(i64)) {
        const mean_kern_ns = @divTrunc(total_kern_ns, @as(i64, @intCast(runs)));
        const best_kern_ms = @as(f64, @floatFromInt(best_kern_ns)) / 1_000_000.0;
        const mean_kern_ms = @as(f64, @floatFromInt(mean_kern_ns)) / 1_000_000.0;
        try w.print("  gpu kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{ best_kern_ms, mb * 1000.0 / best_kern_ms });
        try w.print("  gpu kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{ mean_kern_ms, mb * 1000.0 / mean_kern_ms });
        if (best_lz_ns < std.math.maxInt(i64)) {
            const mean_lz_ns = @divTrunc(total_lz_ns, @as(i64, @intCast(runs)));
            const best_lz_ms = @as(f64, @floatFromInt(best_lz_ns)) / 1_000_000.0;
            const mean_lz_ms = @as(f64, @floatFromInt(mean_lz_ns)) / 1_000_000.0;
            try w.print("  lz kernel best:   {d:.3} ms  ({d:.0} MB/s)\n", .{ best_lz_ms, mb * 1000.0 / best_lz_ms });
            try w.print("  lz kernel mean:   {d:.3} ms  ({d:.0} MB/s)\n", .{ mean_lz_ms, mb * 1000.0 / mean_lz_ms });
        }
        if (best_huff_ns < std.math.maxInt(i64)) {
            const mean_huff_ns = @divTrunc(total_huff_ns, @as(i64, @intCast(runs)));
            const best_huff_ms = @as(f64, @floatFromInt(best_huff_ns)) / 1_000_000.0;
            const mean_huff_ms = @as(f64, @floatFromInt(mean_huff_ns)) / 1_000_000.0;
            try w.print("  huff kernel best: {d:.3} ms  ({d:.0} MB/s)\n", .{ best_huff_ms, mb * 1000.0 / best_huff_ms });
            try w.print("  huff kernel mean: {d:.3} ms  ({d:.0} MB/s)\n", .{ mean_huff_ms, mb * 1000.0 / mean_huff_ms });
        }
    }
}
