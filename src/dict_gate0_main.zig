//! v4 #16 gate-0: dictionary ratio measurement, no codec changes.
//!
//! Measures the compression-ratio lift a preset dictionary would give
//! on a small-records corpus, using ONLY the existing encoder: the
//! dictionary semantics ("dict bytes are reachable history at record
//! start") are simulated by encoding `dict ++ record` inside a single
//! parse window and charging the record its MARGINAL bytes:
//!
//!     marginal(record) = |encode(dict ++ record)| - |encode(dict)|
//!
//! All encodes run with sc_group_size_override = 1.0 (256 KB parse
//! window) so dict + record always share one chunk - the same
//! reachability a real dict implementation would provide. The
//! subtraction cancels frame overhead, so a per-frame overhead
//! estimate (measured by encoding a 1-byte input) is added back to
//! keep the dict column comparable with the cold column.
//!
//! Method limits, stated once: at L3+ the Huffman tables of the
//! combined encode are trained on dict+record jointly, so L3/L5
//! numbers are indicative rather than exact; L1 (LZ-only) is the
//! clean signal. Records are split train/eval by alternating index
//! (even = train, odd = eval) so the dictionary is never measured on
//! the records that trained it.
//!
//! Usage: dict_gate0 <records.jsonl | records-dir>
//!            [--sizes 2,4,8,16,32,64]   dictionary sizes, KB
//!            [--levels 1,3,5]           encode levels
//!            [--max-records N]          cap the record count
//!            [--out-dir <dir>]          write trained dicts there
//!
//! A .jsonl input is one record per line; a directory input is one
//! record per file (read in name order for determinism).

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const trainer = @import("dict/trainer.zig");

const max_dict_kb = 1024;

const Args = struct {
    records_path: []const u8,
    sizes_kb: []const usize,
    levels: []const u8,
    max_records: usize,
    out_dir: ?[]const u8,
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;

    var sizes_buf: [32]usize = undefined;
    var levels_buf: [8]u8 = undefined;
    const args = parseArgs(init, gpa, &sizes_buf, &levels_buf) catch {
        std.debug.print(
            "usage: dict_gate0 <records.jsonl|records-dir> [--sizes 2,4,8,16,32,64] " ++
                "[--levels 1,3,5] [--max-records N] [--out-dir <dir>]\n",
            .{},
        );
        return 1;
    };

    // ── Load records ────────────────────────────────────────────────
    const records = try loadRecords(gpa, io, args.records_path, args.max_records);
    if (records.items.len < 4) {
        std.debug.print("error: need at least 4 records, got {d}\n", .{records.items.len});
        return 1;
    }

    // Alternating train/eval split: even index trains, odd evaluates.
    var train_set: std.ArrayList([]const u8) = .empty;
    defer train_set.deinit(gpa);
    var eval_set: std.ArrayList([]const u8) = .empty;
    defer eval_set.deinit(gpa);
    for (records.items, 0..) |rec, i| {
        try (if (i % 2 == 0) &train_set else &eval_set).append(gpa, rec);
    }

    var raw_eval_bytes: usize = 0;
    for (eval_set.items) |r| raw_eval_bytes += r.len;
    std.debug.print(
        "records: {d} total ({d} train / {d} eval), eval raw {d} bytes, avg {d} B/record\n",
        .{
            records.items.len,           train_set.items.len,
            eval_set.items.len,          raw_eval_bytes,
            raw_eval_bytes / eval_set.items.len,
        },
    );

    // ── Train one dictionary per size ───────────────────────────────
    var dicts: [32]trainer.TrainResult = undefined;
    var dict_count: usize = 0;
    defer for (dicts[0..dict_count]) |*d| d.deinit();
    for (args.sizes_kb) |kb| {
        const result = try trainer.train(gpa, train_set.items, .{ .dict_size = kb * 1024 });
        dicts[dict_count] = result;
        dict_count += 1;
        std.debug.print("trained {d} KB dict: {d} bytes\n", .{ kb, result.dict.len });
        if (args.out_dir) |dir| {
            var name_buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}/github_users_{d}k.dict", .{ dir, kb });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = result.dict });
        }
    }

    // ── Measure ─────────────────────────────────────────────────────
    // One shared frame buffer sized for the largest combined input,
    // and one concat staging buffer. sc=1.0 keeps dict + record in a
    // single parse window (256 KB chunk); see the module header.
    const opts_base: encoder.Options = .{ .sc_group_size_override = 1.0 };
    var max_record: usize = 0;
    for (eval_set.items) |r| max_record = @max(max_record, r.len);
    const max_combined = max_dict_kb * 1024 + max_record;
    const frame = try gpa.alloc(u8, encoder.compressBound(max_combined));
    defer gpa.free(frame);
    const concat = try gpa.alloc(u8, max_combined);
    defer gpa.free(concat);

    // Warm-up encode (context init, buffer growth) before measuring.
    _ = try compress(gpa, eval_set.items[0], frame, opts_base, 1);

    std.debug.print(
        "\n| Level | Dict | Eval compressed (B) | Ratio | Lift vs cold |\n" ++
            "|-------|------|--------------------:|------:|-------------:|\n",
        .{},
    );

    for (args.levels) |level| {
        // Per-frame overhead estimate: a 1-byte input's frame is all
        // header/trailer. Added back onto every marginal total so the
        // dict rows stay comparable with the cold row.
        const one_byte = [_]u8{' '};
        const overhead = (try compress(gpa, &one_byte, frame, opts_base, level)) - 1;

        // Cold baseline: each eval record encoded alone.
        var cold_total: usize = 0;
        for (eval_set.items) |r| {
            cold_total += try compress(gpa, r, frame, opts_base, level);
        }
        printRow(level, null, cold_total, raw_eval_bytes, cold_total);

        for (args.sizes_kb, 0..) |kb, di| {
            const dict = dicts[di].dict;
            @memcpy(concat[0..dict.len], dict);
            const dict_only = try compress(gpa, dict, frame, opts_base, level);

            var marginal_total: usize = 0;
            for (eval_set.items) |r| {
                @memcpy(concat[dict.len..][0..r.len], r);
                const combined = try compress(gpa, concat[0 .. dict.len + r.len], frame, opts_base, level);
                // A dict can only help; a tiny negative marginal would
                // mean header jitter, not signal. Clamp at zero.
                marginal_total += if (combined > dict_only) combined - dict_only else 0;
            }
            const dict_total = marginal_total + eval_set.items.len * overhead;
            printRow(level, kb, dict_total, raw_eval_bytes, cold_total);
        }
    }

    std.debug.print(
        "\nratio = compressed/raw of the eval half ({d} records). Dict rows charge each\n" ++
            "record its marginal bytes plus the measured per-frame overhead. L1 is the\n" ++
            "clean LZ-only signal; L3/L5 share Huffman tables with the dict (indicative).\n",
        .{eval_set.items.len},
    );
    return 0;
}

fn compress(
    gpa: std.mem.Allocator,
    src: []const u8,
    frame: []u8,
    opts_base: encoder.Options,
    level: u8,
) !usize {
    var opts = opts_base;
    opts.level = level;
    return encoder.compressFramed(gpa, src, frame, opts, &gpu_encoder.g_default);
}

fn printRow(level: u8, dict_kb: ?usize, total: usize, raw: usize, cold_total: usize) void {
    const ratio = @as(f64, @floatFromInt(total)) * 100.0 / @as(f64, @floatFromInt(raw));
    if (dict_kb) |kb| {
        const lift = @as(f64, @floatFromInt(cold_total)) / @as(f64, @floatFromInt(total));
        std.debug.print(
            "| L{d} | {d} KB | {d} | {d:.1}% | {d:.2}x |\n",
            .{ level, kb, total, ratio, lift },
        );
    } else {
        std.debug.print(
            "| L{d} | cold | {d} | {d:.1}% | 1.00x |\n",
            .{ level, total, ratio },
        );
    }
}

fn parseArgs(
    init: std.process.Init,
    gpa: std.mem.Allocator,
    sizes_buf: *[32]usize,
    levels_buf: *[8]u8,
) !Args {
    var args_it = try init.minimal.args.iterateAllocator(gpa);
    defer args_it.deinit();
    _ = args_it.next();

    var result: Args = .{
        .records_path = "",
        .sizes_kb = &.{ 2, 4, 8, 16, 32, 64 },
        .levels = &.{ 1, 3, 5 },
        .max_records = std.math.maxInt(usize),
        .out_dir = null,
    };

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--sizes")) {
            const v = args_it.next() orelse return error.BadArgs;
            result.sizes_kb = try parseList(usize, v, sizes_buf, max_dict_kb);
        } else if (std.mem.eql(u8, arg, "--levels")) {
            const v = args_it.next() orelse return error.BadArgs;
            result.levels = try parseList(u8, v, levels_buf, 5);
        } else if (std.mem.eql(u8, arg, "--max-records")) {
            const v = args_it.next() orelse return error.BadArgs;
            result.max_records = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, arg, "--out-dir")) {
            result.out_dir = try gpa.dupe(u8, args_it.next() orelse return error.BadArgs);
        } else if (result.records_path.len == 0) {
            result.records_path = try gpa.dupe(u8, arg);
        } else {
            return error.BadArgs;
        }
    }
    if (result.records_path.len == 0) return error.BadArgs;
    return result;
}

fn parseList(comptime T: type, csv: []const u8, buf: []T, max: usize) ![]T {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |part| {
        if (part.len == 0) continue;
        if (n == buf.len) return error.BadArgs;
        const v = try std.fmt.parseInt(T, part, 10);
        if (v == 0 or v > max) return error.BadArgs;
        buf[n] = v;
        n += 1;
    }
    if (n == 0) return error.BadArgs;
    return buf[0..n];
}

/// Load records from a .jsonl file (one record per line, newline kept)
/// or a directory (one record per file, name order). Record memory is
/// owned by the process for its lifetime; no per-record frees.
fn loadRecords(
    gpa: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    max_records: usize,
) !std.ArrayList([]const u8) {
    var records: std.ArrayList([]const u8) = .empty;

    if (std.mem.endsWith(u8, path, ".jsonl")) {
        const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
        var it = std.mem.splitScalar(u8, data, '\n');
        while (it.next()) |line| {
            if (line.len == 0) continue;
            if (records.items.len == max_records) break;
            try records.append(gpa, line);
        }
        return records;
    }

    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        try names.append(gpa, try gpa.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    for (names.items) |name| {
        if (records.items.len == max_records) break;
        const data = try dir.readFileAlloc(io, name, gpa, .unlimited);
        try records.append(gpa, data);
    }
    return records;
}
