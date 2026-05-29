//! `streamlz -ba <file>` — sweep levels L1..L5 and print a single
//! ratio + compress/decompress throughput table.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_encoder = @import("../encode/driver.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);
    const runs = args.runs orelse 3;
    if (runs == 0) util.die(w, "error: runs must be >= 1\n");

    const src = util.readFile(allocator, io, in_path, w);
    defer allocator.free(src);

    const mb: f64 = @as(f64, @floatFromInt(src.len)) / (1024.0 * 1024.0);
    try w.print("streamlz bench-all: {s} ({d} bytes, {d} decompress runs)\n", .{
        in_path, src.len, runs,
    });

    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const decompressed = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(decompressed);

    const Result = struct {
        level: u8,
        comp_size: usize,
        ratio: f64,
        comp_mbps: f64,
        dec_mbps: f64,
        pass: bool,
    };
    var results: [5]Result = undefined;

    var level: u8 = 1;
    while (level <= 5) : (level += 1) {
        const idx = level - 1;
        const comp_opts: encoder.Options = .{
            .level = level,
            .sc_group_size_override = args.sc_group,
        };
        const t_comp = std.Io.Clock.awake.now(io);
        const comp_size = encoder.compressFramedWithIo(allocator, io, src, compressed, comp_opts, &gpu_encoder.g_default) catch |err| {
            try w.print("  L{d}: compress failed: {s}\n", .{ level, @errorName(err) });
            results[idx] = .{ .level = level, .comp_size = 0, .ratio = 0, .comp_mbps = 0, .dec_mbps = 0, .pass = false };
            continue;
        };
        const best_comp_ns = @as(u64, @intCast(t_comp.untilNow(io, .awake).toNanoseconds()));

        var dec_ctx = decoder.DecompressContext.initWithIo(allocator, io);
        defer dec_ctx.deinit();

        _ = dec_ctx.decompress(compressed[0..comp_size], decompressed) catch |err| {
            try w.print("  L{d}: decompress failed: {s}\n", .{ level, @errorName(err) });
            results[idx] = .{ .level = level, .comp_size = comp_size, .ratio = 0, .comp_mbps = 0, .dec_mbps = 0, .pass = false };
            continue;
        };

        var best_dec_ns: u64 = std.math.maxInt(u64);
        var r: u32 = 0;
        while (r < runs) : (r += 1) {
            const t0 = std.Io.Clock.awake.now(io);
            _ = try dec_ctx.decompress(compressed[0..comp_size], decompressed);
            const elapsed = @as(u64, @intCast(t0.untilNow(io, .awake).toNanoseconds()));
            if (elapsed < best_dec_ns) best_dec_ns = elapsed;
        }

        const pass = std.mem.eql(u8, src, decompressed[0..src.len]);
        const ratio = @as(f64, @floatFromInt(comp_size)) / @as(f64, @floatFromInt(@max(src.len, 1))) * 100.0;
        const comp_ms = @as(f64, @floatFromInt(best_comp_ns)) / 1_000_000.0;
        const dec_ms = @as(f64, @floatFromInt(best_dec_ns)) / 1_000_000.0;
        results[idx] = .{
            .level = level,
            .comp_size = comp_size,
            .ratio = ratio,
            .comp_mbps = mb * 1000.0 / comp_ms,
            .dec_mbps = mb * 1000.0 / dec_ms,
            .pass = pass,
        };
        try w.print("  L{d} done ({d:.1}%)\n", .{ level, ratio });
        try w.flush();
    }

    try w.writeAll("\nLevel | Compressed         | Ratio  | Compress   | Decompress\n");
    try w.writeAll("------+--------------------+--------+------------+-----------\n");
    for (&results) |res| {
        var bytes_buf: [32]u8 = undefined;
        var cmbps_buf: [16]u8 = undefined;
        var dmbps_buf: [16]u8 = undefined;
        const pass_str: []const u8 = if (res.pass) "" else " FAIL";
        try w.print("L{d:<2}   | {s:>14} bytes | {d:>5.1}% | {s:>7} MB/s | {s:>7} MB/s{s}\n", .{
            res.level,
            util.fmtBytes(&bytes_buf, res.comp_size),
            res.ratio,
            util.fmtMbps(&cmbps_buf, res.comp_mbps),
            util.fmtMbps(&dmbps_buf, res.dec_mbps),
            pass_str,
        });
    }
}
