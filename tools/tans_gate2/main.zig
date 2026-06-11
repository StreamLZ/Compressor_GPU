//! v4 #11 gate-2 driver: replay SLZ_TANS_GATE2_DUMP records through
//! the resurrected host tANS-32 encoder and compare REAL per-chunk
//! tANS wire sizes against the exact production Huffman wire sizes.
//!
//! Record format (written by encode_huff.dumpGate2Records):
//!   "TG2R" u32 LE | plane u8 | raw_len u32 LE | huff_len u32 LE | raw bytes
//! Planes: 0=lit 1=tok 2=off16hi 3=off16lo.
//!
//! Usage: tans_gate2 <dump-file> [frame-bytes]
//! frame-bytes (optional) scales the selector saving to end-ratio pp.

const std = @import("std");
const tans = @import("tans_encoder.zig");
const hist_mod = @import("byte_histogram.zig");

const PLANE_NAMES = [_][]const u8{ "lit", "tok", "off16hi", "off16lo" };

const Stat = struct {
    n: u64 = 0,
    raw: u64 = 0,
    huff: u64 = 0,
    tans: u64 = 0, // sum of min(tans, huff) fallback when tANS errors
    tans_ok: u64 = 0, // records where tANS produced a size
    min_hb: u64 = 0, // per-record min(huff, tans)
    flips: u64 = 0, // records where tans < huff
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(gpa);
    defer args_it.deinit();
    _ = args_it.next(); // exe
    const dump_path = args_it.next() orelse {
        std.debug.print("usage: tans_gate2 <dump-file> [frame-bytes]\n", .{});
        return 1;
    };
    const frame_bytes: u64 = if (args_it.next()) |s|
        std.fmt.parseInt(u64, s, 10) catch 0
    else
        0;

    const data = try std.Io.Dir.cwd().readFileAlloc(io, dump_path, gpa, .unlimited);
    defer gpa.free(data);

    var stats: [4]Stat = .{ .{}, .{}, .{}, .{} };
    var dst_buf: []u8 = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(dst_buf);

    var pos: usize = 0;
    var bad_magic: u64 = 0;
    while (pos + 13 <= data.len) {
        const magic = std.mem.readInt(u32, data[pos..][0..4], .little);
        if (magic != 0x52324754) {
            bad_magic += 1;
            break;
        }
        const plane = data[pos + 4];
        const raw_len = std.mem.readInt(u32, data[pos + 5 ..][0..4], .little);
        const huff_len = std.mem.readInt(u32, data[pos + 9 ..][0..4], .little);
        pos += 13;
        if (pos + raw_len > data.len or plane >= 4) break;
        const raw = data[pos..][0..raw_len];
        pos += raw_len;

        if (dst_buf.len < raw_len * 2 + 4096) {
            dst_buf = try gpa.realloc(dst_buf, raw_len * 2 + 4096);
        }

        var histo: hist_mod.ByteHistogram = .{};
        histo.countBytes(raw);
        var cost: f32 = 0;
        const tans_len: ?usize = tans.encodeArrayU8Tans32(
            gpa, dst_buf, raw, &histo, 0.0, &cost, null,
        ) catch null;

        var st = &stats[plane];
        st.n += 1;
        st.raw += raw_len;
        st.huff += huff_len;
        if (tans_len) |tl| {
            st.tans_ok += 1;
            st.tans += tl;
            if (tl < huff_len) {
                st.min_hb += tl;
                st.flips += 1;
            } else {
                st.min_hb += huff_len;
            }
        } else {
            st.tans += huff_len; // no tANS form -> huff stands in
            st.min_hb += huff_len;
        }
    }

    var tot: Stat = .{};
    std.debug.print(
        "{s:<8}{s:>9}{s:>13}{s:>13}{s:>13}{s:>13}{s:>8}{s:>9}\n",
        .{ "plane", "records", "raw", "huff", "tans", "min(h,t)", "flips%", "save%" },
    );
    for (0..4) |p| {
        const st = stats[p];
        if (st.n == 0) continue;
        tot.n += st.n;
        tot.raw += st.raw;
        tot.huff += st.huff;
        tot.tans += st.tans;
        tot.min_hb += st.min_hb;
        tot.flips += st.flips;
        tot.tans_ok += st.tans_ok;
        const save = @as(f64, @floatFromInt(st.huff - st.min_hb)) * 100.0 / @as(f64, @floatFromInt(st.huff));
        const flippct = @as(f64, @floatFromInt(st.flips)) * 100.0 / @as(f64, @floatFromInt(st.n));
        std.debug.print(
            "{s:<8}{d:>9}{d:>13}{d:>13}{d:>13}{d:>13}{d:>8.1}{d:>9.3}\n",
            .{ PLANE_NAMES[p], st.n, st.raw, st.huff, st.tans, st.min_hb, flippct, save },
        );
    }
    const tsave = @as(f64, @floatFromInt(tot.huff - tot.min_hb)) * 100.0 / @as(f64, @floatFromInt(tot.huff));
    const tflip = @as(f64, @floatFromInt(tot.flips)) * 100.0 / @as(f64, @floatFromInt(tot.n));
    std.debug.print(
        "{s:<8}{d:>9}{d:>13}{d:>13}{d:>13}{d:>13}{d:>8.1}{d:>9.3}\n",
        .{ "TOTAL", tot.n, tot.raw, tot.huff, tot.tans, tot.min_hb, tflip, tsave },
    );
    if (bad_magic != 0) std.debug.print("WARN: stopped at bad magic (records truncated?)\n", .{});
    if (frame_bytes > 0) {
        const saved = tot.huff - tot.min_hb;
        const pp = @as(f64, @floatFromInt(saved)) * 100.0 / @as(f64, @floatFromInt(frame_bytes));
        std.debug.print(
            "selector saves {d} bytes = {d:.3} pp of end ratio (frame {d} bytes)\n",
            .{ saved, pp, frame_bytes },
        );
    }
    return 0;
}
