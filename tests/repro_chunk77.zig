//! Minimal repro for the silesia chunk-77 bug.
//!
//! Loads chunk 77 of silesia_all.tar (bytes [10092544, 10223616), exactly
//! 128 KiB) into a stack-resident buffer, runs the Vulkan L1 encode +
//! decode round-trip, and prints PASS/FAIL with the first-differing
//! offset and total mismatch count.
//!
//! Same SPV / pipeline shape as src_vulkan/l1_codec_test.zig — the goal
//! is byte-equal round-trip on the single chunk that the scale test
//! observed mis-encoding at intra-chunk offset 114424.
//!
//! Wired as `zig build vk-repro-chunk77`. Requires assets/silesia_all.tar.

const std = @import("std");

const driver = @import("../src_vulkan/driver.zig");
const probe_mod = @import("../src_vulkan/probe.zig");
const l1_codec = @import("../src_vulkan/l1_codec.zig");

const MAX_SPV_BYTES: usize = 1 << 20;
const SPV_DIR_REL: []const u8 = "zig-out/shaders";
const SILESIA_PATH: []const u8 = "assets/silesia_all.tar";

// To run a SMALLER window inside the chunk (e.g. just the failure region),
// adjust CHUNK_OFFSET / CHUNK_BYTES. The chunk-77 origin in silesia_all.tar
// is byte 10,092,544 and the failure was observed at intra-chunk offset
// 114424 (= silesia byte 10,206,968). To reproduce the failure with a
// smaller test input, use a sub-range that still includes the failure
// site, padding to a multiple of 4 to satisfy the encoder's u32 packing.
const CHUNK_OFFSET: u64 = 10092544; // chunk 77 byte 0 in silesia_all.tar
const CHUNK_BYTES: usize = 131072; // exactly 128 KiB == L1 CHUNK_SIZE

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

pub fn main(process_init: std.process.Init) !void {
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
        try w.print("REPRO FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    var enc_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    var dec_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const enc_spv = try loadSpv(io, "lz_encode", tier_name, &enc_spv_storage);
    const dec_spv = try loadSpv(io, "lz_decode", tier_name, &dec_spv_storage);

    try w.print("REPRO chunk77 — tier={s} subgroup={d}\n", .{ tier_name, pr.subgroup_size });
    try w.flush();

    // ── Load chunk 77 from silesia ─────────────────────────────────
    var src_buf: [CHUNK_BYTES]u8 = undefined;
    var dst_buf: [CHUNK_BYTES]u8 = undefined;

    {
        var file = std.Io.Dir.cwd().openFile(io, SILESIA_PATH, .{}) catch {
            try w.print("REPRO FAIL silesia open\n", .{});
            return error.MissingAsset;
        };
        defer file.close(io);
        const n = file.readPositionalAll(io, src_buf[0..], CHUNK_OFFSET) catch {
            try w.print("REPRO FAIL silesia read\n", .{});
            return error.ReadFail;
        };
        if (n != CHUNK_BYTES) {
            try w.print("REPRO FAIL short_read {d}\n", .{n});
            return error.ShortRead;
        }
    }

    // Optional: encode only a sub-window of the chunk. Override via
    // cmdline args: `vk_repro_chunk77 LEN [START]` encodes chunk[START..START+LEN].
    // Defaults to encoding the full 128 KiB chunk from offset 0.
    var slice_len: usize = CHUNK_BYTES;
    var slice_start: usize = 0;
    {
        const allocator = process_init.gpa;
        var args_it = process_init.minimal.args.iterateAllocator(allocator) catch null;
        if (args_it) |*it| {
            defer it.deinit();
            _ = it.next(); // exe path
            if (it.next()) |first| {
                slice_len = std.fmt.parseInt(usize, first, 10) catch CHUNK_BYTES;
            }
            if (it.next()) |second| {
                slice_start = std.fmt.parseInt(usize, second, 10) catch 0;
            }
        }
        if (slice_start > CHUNK_BYTES) slice_start = CHUNK_BYTES;
        if (slice_start + slice_len > CHUNK_BYTES) slice_len = CHUNK_BYTES - slice_start;
    }
    try w.print("REPRO encoding slice_start={d} slice_len={d}\n", .{ slice_start, slice_len });

    // ── Encode + decode the chunk ────────────────────────────────
    const slice = src_buf[slice_start .. slice_start + slice_len];
    var enc = try l1_codec.encodeL1Sync(ctx, slice, enc_spv);
    defer l1_codec.freeStreams(ctx, &enc.streams);
    @memset(dst_buf[0..], 0);
    try l1_codec.decodeL1Sync(ctx, enc.streams, slice_len, dst_buf[0..slice_len], dec_spv);

    // ── Compare ───────────────────────────────────────────────────
    var first_diff: usize = 0;
    var total_diffs: usize = 0;
    var found_first = false;
    var i: usize = 0;
    while (i < slice_len) : (i += 1) {
        if (slice[i] != dst_buf[i]) {
            if (!found_first) {
                first_diff = i;
                found_first = true;
            }
            total_diffs += 1;
        }
    }

    if (total_diffs == 0) {
        try w.print("REPRO PASS chunk77 bytes={d} comp_bytes={d}\n", .{ slice_len, enc.comp_total_bytes });
    } else {
        try w.print(
            "REPRO FAIL chunk77 bytes={d} comp_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ slice_len, enc.comp_total_bytes, first_diff, total_diffs },
        );

        // Find last diff
        var last_diff: usize = first_diff;
        var j: usize = first_diff;
        while (j < slice_len) : (j += 1) {
            if (slice[j] != dst_buf[j]) last_diff = j;
        }
        try w.print("last_diff={d}\n", .{last_diff});

        // Print every diff position
        var k: usize = first_diff;
        while (k <= last_diff) : (k += 1) {
            if (slice[k] != dst_buf[k]) {
                try w.print("DIFF @ {d}: src={x:0>2} dst={x:0>2}\n", .{ k, slice[k], dst_buf[k] });
            }
        }
    }
}
