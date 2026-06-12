//! v4 #16: per-record preset-dictionary benchmark.
//!
//! The dictionary use case is many small records, each its own frame
//! (the GPU batch shape): every record starts cold without a dict and
//! warm with one. This tool encodes each input record twice - plain
//! and with a registry dictionary - then decodes the dictionary
//! frames and byte-verifies them, reporting ratio and throughput per
//! level.
//!
//! Usage: dict_bench <records.jsonl> [--dict json] [--levels 1,3,5]
//!                   [--max-records N]
//!
//! Note on dictionary quality: the registry built-ins are shared with
//! the CPU sibling project and are NOT trained on any particular
//! corpus - ratio lift here demonstrates the machinery. Per-corpus
//! training (the gate-0 knee study projected 2.3-3.5x on github_users
//! with a trained dict) is a registry-asset decision, not a codec one.

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const gpu_driver = @import("decode/driver.zig");
const dictionary = @import("dict/dictionary.zig");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(gpa);
    defer args_it.deinit();
    _ = args_it.next();

    var records_path: ?[]const u8 = null;
    var dict_name: []const u8 = "json";
    var levels_buf: [8]u8 = .{ 1, 3, 5, 0, 0, 0, 0, 0 };
    var levels: []const u8 = levels_buf[0..3];
    var max_records: usize = std.math.maxInt(usize);
    var eval_split = false;
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dict")) {
            dict_name = try gpa.dupe(u8, args_it.next() orelse return usage());
        } else if (std.mem.eql(u8, arg, "--eval-split")) {
            // Keep only odd-indexed records - the half dict_gate0 holds
            // out from training, so a gate-0-trained dictionary is never
            // benchmarked on its own training data.
            eval_split = true;
        } else if (std.mem.eql(u8, arg, "--levels")) {
            const v = args_it.next() orelse return usage();
            var n: usize = 0;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |part| : (n += 1) {
                if (n == levels_buf.len) return usage();
                levels_buf[n] = try std.fmt.parseInt(u8, part, 10);
            }
            levels = levels_buf[0..n];
        } else if (std.mem.eql(u8, arg, "--max-records")) {
            max_records = try std.fmt.parseInt(usize, args_it.next() orelse return usage(), 10);
        } else if (records_path == null) {
            records_path = try gpa.dupe(u8, arg);
        } else return usage();
    }
    const path = records_path orelse return usage();
    const dict = dictionary.findByName(dict_name) orelse {
        std.debug.print("unknown dictionary '{s}'\n", .{dict_name});
        return 1;
    };

    // Load records (one per line).
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    var records: std.ArrayList([]const u8) = .empty;
    defer records.deinit(gpa);
    var it = std.mem.splitScalar(u8, data, '\n');
    var line_idx: usize = 0;
    while (it.next()) |line| {
        if (line.len == 0) continue;
        defer line_idx += 1;
        if (eval_split and line_idx % 2 == 0) continue;
        if (records.items.len == max_records) break;
        try records.append(gpa, line);
    }
    var raw_total: usize = 0;
    var max_rec: usize = 0;
    for (records.items) |r| {
        raw_total += r.len;
        max_rec = @max(max_rec, r.len);
    }
    if (gpu_driver.isAvailable())
        std.debug.print("Device: {s}\n", .{gpu_driver.deviceName()});
    std.debug.print(
        "records: {d}, raw {d} bytes (avg {d} B), dict '{s}' ({d} B)\n\n",
        .{ records.items.len, raw_total, raw_total / records.items.len, dict.name, dict.data.len },
    );

    const frame_buf = try gpa.alloc(u8, encoder.compressBound(max_rec));
    defer gpa.free(frame_buf);
    const out_buf = try gpa.alloc(u8, max_rec + decoder.safe_space);
    defer gpa.free(out_buf);
    // Frames retained for the decode pass.
    var frames: std.ArrayList([]u8) = .empty;
    defer {
        for (frames.items) |f| gpa.free(f);
        frames.deinit(gpa);
    }

    // Warm-up.
    _ = try encoder.compressFramed(gpa, records.items[0], frame_buf, .{ .level = 1 }, &gpu_encoder.g_default);

    std.debug.print(
        "| Level | Variant | Total bytes | Ratio | Encode | Decode+verify |\n" ++
            "|-------|---------|------------:|------:|-------:|--------------:|\n",
        .{},
    );

    for (levels) |level| {
        for ([_]?u32{ null, dict.id }) |dict_id| {
            for (frames.items) |f| gpa.free(f);
            frames.clearRetainingCapacity();

            const t_enc = std.Io.Clock.awake.now(io);
            var total: usize = 0;
            for (records.items) |rec| {
                const n = try encoder.compressFramed(
                    gpa,
                    rec,
                    frame_buf,
                    .{ .level = level, .dictionary_id = dict_id },
                    &gpu_encoder.g_default,
                );
                total += n;
                try frames.append(gpa, try gpa.dupe(u8, frame_buf[0..n]));
            }
            const enc_ms = nsToMs(t_enc.untilNow(io, .awake).toNanoseconds());

            const t_dec = std.Io.Clock.awake.now(io);
            for (frames.items, 0..) |f, i| {
                const written = try decoder.decompressFramed(f, out_buf, &gpu_driver.g_default);
                if (written != records.items[i].len or
                    !std.mem.eql(u8, records.items[i], out_buf[0..written]))
                {
                    std.debug.print("VERIFY FAILED at record {d} (L{d}, dict={any})\n", .{ i, level, dict_id });
                    return 1;
                }
            }
            const dec_ms = nsToMs(t_dec.untilNow(io, .awake).toNanoseconds());

            const ratio = @as(f64, @floatFromInt(total)) * 100.0 / @as(f64, @floatFromInt(raw_total));
            std.debug.print("| L{d} | {s} | {d} | {d:.1}% | {d:.0} ms | {d:.0} ms |\n", .{
                level,
                if (dict_id == null) "plain" else "dict",
                total,
                ratio,
                enc_ms,
                dec_ms,
            });
        }
    }
    return 0;
}

fn nsToMs(ns: i96) f64 {
    return @as(f64, @floatFromInt(@as(i64, @intCast(ns)))) / 1e6;
}

fn usage() u8 {
    std.debug.print("usage: dict_bench <records.jsonl> [--dict json] [--levels 1,3,5] [--max-records N]\n", .{});
    return 1;
}
