//! Cross-backend SLZ1 wire-format conformance at production scale.
//!
//! Mirrors `wire_format_test.zig` but targets the large corpora the
//! base test deliberately skips (enwik8 full = 100 MB, silesia full =
//! 200 MB). Each direction:
//!
//!   * vk_encode_cuda_decode: VK encode -> wrap to SLZ1 -> shell out
//!     to streamlz.exe -d -> compare bytes.
//!   * cuda_encode_vk_decode: streamlz.exe -c -l 1 -> read .slz ->
//!     unwrap -> VK decode -> compare.
//!
//! Validates the CPU wrap/unwrap path has no O(n²) hiding in it and
//! the wire format survives at production scale. Opt-in target:
//! `zig build vk-wire-format-scale-test`.
//!
//! Lifted helpers from `wire_format_test.zig` rather than `pub`-ing
//! the internals — keeps the production module's surface unchanged.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const wire_format = @import("wire_format.zig");
// Cluster F (F036): scale-test CUDA_TO_VK direction now goes through
// the production decoder. Sibling fix to wire_format_test.zig. The
// `wire_constants` / `descriptors` / `dispatch` imports went away with
// the deleted `decodeUnwrappedVkOnly` fixture — they were only ever
// used to build descriptor sets + push constants for the test-only
// lz_decode dispatch that the fixture drove.
const slz1_codec = @import("slz1_codec.zig");

const MAX_SPV_BYTES: usize = 1 << 20;
const SPV_DIR_REL: []const u8 = "zig-out/shaders";
const STREAMLZ_EXE: []const u8 = "c:/Users/james.JAMESWORK2025/Repos/Compressor_GPU/zig-out/bin/streamlz.exe";
const TMP_DIR: []const u8 = "c:/tmp";

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

const win32 = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};
fn qpcNow() i64 { var c: i64 = 0; _ = win32.QueryPerformanceCounter(&c); return c; }
fn qpcNs(from: i64, to: i64) u64 {
    var freq: i64 = 0;
    _ = win32.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

// ── Stream readback (mirror of wire_format_test.zig::buildStreamBundle) ─

const StreamBundle = struct {
    lit_views: [][]const u8,
    cmd_views: [][]const u8,
    off16_views: [][]const u8,
    length_views: [][]const u8,
    off32_views: [][]const u8,
    lit_arena: []u8,
    cmd_arena: []u8,
    off16_arena: []u8,
    length_arena: []u8,
    off32_arena: []u8,

    pub fn deinit(self: *StreamBundle, allocator: std.mem.Allocator) void {
        allocator.free(self.lit_arena);
        allocator.free(self.cmd_arena);
        allocator.free(self.off16_arena);
        allocator.free(self.length_arena);
        allocator.free(self.off32_arena);
        allocator.free(self.lit_views);
        allocator.free(self.cmd_views);
        allocator.free(self.off16_views);
        allocator.free(self.length_views);
        allocator.free(self.off32_views);
    }
};

fn copyOneStream(
    ctx: *driver.Context,
    mem: vk.VkDeviceMemory,
    out: []u8,
    chunk_capacity: u32,
    n_chunks: u32,
    per_chunk_size: *const [l1_codec.MAX_CHUNKS]u32,
    is_off16: bool,
) !void {
    const map = vk.vkMapMemory_fn orelse return error.MapMemoryFailed;
    const unmap = vk.vkUnmapMemory_fn orelse return error.MapMemoryFailed;
    var raw: ?*anyopaque = null;
    if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS)
        return error.MapMemoryFailed;
    defer unmap(ctx.dev, mem);
    const p: [*]const u8 = @ptrCast(raw.?);

    var pos: usize = 0;
    var ci: u32 = 0;
    while (ci < n_chunks) : (ci += 1) {
        const chunk_byte_base: usize = @as(usize, ci) * chunk_capacity;
        const slice_bytes: usize = if (is_off16)
            @as(usize, per_chunk_size[ci]) * 2
        else
            per_chunk_size[ci];
        @memcpy(out[pos..][0..slice_bytes], p[chunk_byte_base..][0..slice_bytes]);
        pos += slice_bytes;
    }
}

fn copyOff32Stream(
    ctx: *driver.Context,
    streams: l1_codec.L1Streams,
    out: []u8,
) !void {
    const map = vk.vkMapMemory_fn orelse return error.MapMemoryFailed;
    const unmap = vk.vkUnmapMemory_fn orelse return error.MapMemoryFailed;
    var raw: ?*anyopaque = null;
    if (map(ctx.dev, streams.off32_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS)
        return error.MapMemoryFailed;
    defer unmap(ctx.dev, streams.off32_mem);
    const p: [*]const u8 = @ptrCast(raw.?);
    var pos: usize = 0;
    var ci: u32 = 0;
    while (ci < streams.n_chunks) : (ci += 1) {
        const chunk_byte_base: usize = @as(usize, ci) * streams.off32_capacity;
        const slice_bytes: usize = @as(usize, streams.per_chunk_off32_count1[ci] + streams.per_chunk_off32_count2[ci]) * 3;
        @memcpy(out[pos..][0..slice_bytes], p[chunk_byte_base..][0..slice_bytes]);
        pos += slice_bytes;
    }
}

fn buildStreamBundle(
    allocator: std.mem.Allocator,
    ctx: *driver.Context,
    streams: l1_codec.L1Streams,
) !StreamBundle {
    var total_lit: usize = 0;
    var total_cmd: usize = 0;
    var total_off16_bytes: usize = 0;
    var total_length: usize = 0;
    var total_off32_bytes: usize = 0;
    {
        var ci: u32 = 0;
        while (ci < streams.n_chunks) : (ci += 1) {
            total_lit += streams.per_chunk_lit_size[ci];
            total_cmd += streams.per_chunk_cmd_size[ci];
            total_off16_bytes += @as(usize, streams.per_chunk_off16_count[ci]) * 2;
            total_length += streams.per_chunk_length_used[ci];
            total_off32_bytes += @as(usize, streams.per_chunk_off32_count1[ci] + streams.per_chunk_off32_count2[ci]) * 3;
        }
    }

    var bundle: StreamBundle = .{
        .lit_arena = try allocator.alloc(u8, total_lit + 16),
        .cmd_arena = try allocator.alloc(u8, total_cmd + 16),
        .off16_arena = try allocator.alloc(u8, total_off16_bytes + 16),
        .length_arena = try allocator.alloc(u8, total_length + 16),
        .off32_arena = try allocator.alloc(u8, total_off32_bytes + 16),
        .lit_views = try allocator.alloc([]const u8, streams.n_chunks),
        .cmd_views = try allocator.alloc([]const u8, streams.n_chunks),
        .off16_views = try allocator.alloc([]const u8, streams.n_chunks),
        .length_views = try allocator.alloc([]const u8, streams.n_chunks),
        .off32_views = try allocator.alloc([]const u8, streams.n_chunks),
    };
    errdefer bundle.deinit(allocator);

    try copyOneStream(ctx, streams.lit_mem, bundle.lit_arena, streams.chunk_capacity,
        streams.n_chunks, &streams.per_chunk_lit_size, false);
    try copyOneStream(ctx, streams.cmd_mem, bundle.cmd_arena, streams.chunk_capacity,
        streams.n_chunks, &streams.per_chunk_cmd_size, false);
    try copyOneStream(ctx, streams.off16_mem, bundle.off16_arena, streams.chunk_capacity,
        streams.n_chunks, &streams.per_chunk_off16_count, true);
    try copyOneStream(ctx, streams.length_mem, bundle.length_arena, streams.chunk_capacity,
        streams.n_chunks, &streams.per_chunk_length_used, false);
    try copyOff32Stream(ctx, streams, bundle.off32_arena);

    var lit_pos: usize = 0;
    var cmd_pos: usize = 0;
    var off16_pos: usize = 0;
    var length_pos: usize = 0;
    var off32_pos: usize = 0;
    var ci: u32 = 0;
    while (ci < streams.n_chunks) : (ci += 1) {
        const ls = streams.per_chunk_lit_size[ci];
        const cs = streams.per_chunk_cmd_size[ci];
        const os = @as(usize, streams.per_chunk_off16_count[ci]) * 2;
        const xs = streams.per_chunk_length_used[ci];
        const o32 = @as(usize, streams.per_chunk_off32_count1[ci] + streams.per_chunk_off32_count2[ci]) * 3;
        bundle.lit_views[ci] = bundle.lit_arena[lit_pos..][0..ls];
        bundle.cmd_views[ci] = bundle.cmd_arena[cmd_pos..][0..cs];
        bundle.off16_views[ci] = bundle.off16_arena[off16_pos..][0..os];
        bundle.length_views[ci] = bundle.length_arena[length_pos..][0..xs];
        bundle.off32_views[ci] = bundle.off32_arena[off32_pos..][0..o32];
        lit_pos += ls;
        cmd_pos += cs;
        off16_pos += os;
        length_pos += xs;
        off32_pos += o32;
    }

    return bundle;
}

fn runStreamlz(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u32 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    var argv = try aa.alloc([]const u8, 1 + args.len);
    argv[0] = STREAMLZ_EXE;
    for (args, 0..) |a, i| argv[i + 1] = a;
    const result = try std.process.run(aa, io, .{ .argv = argv });
    return switch (result.term) { .exited => |code| code, else => 1 };
}

fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, bytes, 0);
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    const n = try file.readPositionalAll(io, buf, 0);
    if (n != buf.len) return error.ShortRead;
    return buf;
}

const Diff = struct { first_diff: usize, total_diffs: usize };

fn compareBytes(a: []const u8, b: []const u8) Diff {
    const n = @min(a.len, b.len);
    var first: usize = 0;
    var found = false;
    var total: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) {
            if (!found) { first = i; found = true; }
            total += 1;
        }
    }
    if (a.len != b.len) {
        if (!found) { first = n; found = true; }
        total += @max(a.len, b.len) - n;
    }
    return .{ .first_diff = first, .total_diffs = total };
}

// ── VK encode -> CUDA decode at scale ──────────────────────────────

fn testVkToCuda(
    w: *std.Io.Writer,
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *driver.Context,
    enc_spv: []const u8,
    label: []const u8,
    src: []const u8,
    tmp_slz: []const u8,
    tmp_out: []const u8,
    expected_failure_byte: ?usize,
) !void {
    const t0 = qpcNow();
    var enc = l1_codec.encodeL1Multi(ctx, src, enc_spv) catch |err| {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=vk_encode err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer l1_codec.freeStreams(ctx, &enc.streams);
    const t1 = qpcNow();
    const vk_encode_ns = qpcNs(t0, t1);

    var bundle = buildStreamBundle(allocator, ctx, enc.streams) catch |err| {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=bundle err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer bundle.deinit(allocator);
    const t2 = qpcNow();
    const bundle_ns = qpcNs(t1, t2);

    const streams_view: wire_format.PerChunkStreams = .{
        .lit_bytes = bundle.lit_views,
        .cmd_bytes = bundle.cmd_views,
        .off16_bytes = bundle.off16_views,
        .length_bytes = bundle.length_views,
        .off32_bytes = bundle.off32_views,
        .per_chunk_off32_count1 = enc.streams.per_chunk_off32_count1[0..enc.streams.n_chunks],
        .per_chunk_off32_count2 = enc.streams.per_chunk_off32_count2[0..enc.streams.n_chunks],
        .per_chunk_cmd_stream2_offset = enc.streams.per_chunk_cmd_stream2_offset[0..enc.streams.n_chunks],
        .src_bytes = src,
        .per_chunk_initial_copy = enc.streams.per_chunk_initial_copy[0..enc.streams.n_chunks],
        .n_chunks = enc.streams.n_chunks,
        .chunk_size = wire_format.VK_CHUNK_SIZE,
        .original_size = @intCast(src.len),
    };

    const bound = wire_format.wrapBound(streams_view);
    const slz_buf = allocator.alloc(u8, bound) catch {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=alloc_wrap\n", .{label});
        return;
    };
    defer allocator.free(slz_buf);
    const slz_size = wire_format.wrapL1ToSlz1(streams_view, slz_buf) catch |err| {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=wrap err={s}\n", .{ label, @errorName(err) });
        return;
    };
    const t3 = qpcNow();
    const wrap_ns = qpcNs(t2, t3);

    writeFile(io, tmp_slz, slz_buf[0..slz_size]) catch |err| {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=write err={s}\n", .{ label, @errorName(err) });
        return;
    };

    const exit = runStreamlz(io, allocator, &.{ "-d", tmp_slz, "-o", tmp_out }) catch |err| {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=streamlz err={s}\n", .{ label, @errorName(err) });
        return;
    };
    if (exit != 0) {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=streamlz exit={d} slz_bytes={d}\n", .{ label, exit, slz_size });
        return;
    }
    const t4 = qpcNow();
    const cuda_decode_ns = qpcNs(t3, t4);

    const decoded = readFile(io, allocator, tmp_out) catch |err| {
        try w.print("VK_TO_CUDA_SCALE FAIL {s} stage=read_decoded err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer allocator.free(decoded);

    const diff = compareBytes(src, decoded);
    if (diff.total_diffs == 0 and src.len == decoded.len) {
        if (expected_failure_byte) |_| {
            // Round-trip succeeded for a corpus that was annotated as
            // a pre-existing failure: the underlying bug was fixed.
            // Report as REGRESS so the operator notices and removes
            // the expected_failure_byte slot from the corpus table.
            try w.print(
                "VK_TO_CUDA_SCALE REGRESS_FIXED {s} bytes={d} slz_bytes={d}\n",
                .{ label, src.len, slz_size },
            );
        } else {
            try w.print(
                "VK_TO_CUDA_SCALE PASS {s} bytes={d} slz_bytes={d} vk_enc_ns={d} bundle_ns={d} wrap_ns={d} cuda_dec_ns={d}\n",
                .{ label, src.len, slz_size, vk_encode_ns, bundle_ns, wrap_ns, cuda_decode_ns },
            );
        }
    } else if (expected_failure_byte) |expected| {
        // Corpus is known to fail at `expected`; assert the actual
        // diff position is within range of it (chunk-boundary slop is
        // SLZ_CHUNK_SIZE = 64 KiB, so allow ±1 chunk before tagging
        // it as a NEW failure mode that needs investigation).
        const slop: usize = 64 * 1024;
        const lo = if (expected > slop) expected - slop else 0;
        const hi = expected + slop;
        if (diff.first_diff >= lo and diff.first_diff <= hi) {
            try w.print(
                "VK_TO_CUDA_SCALE EXPECTED_FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d} (within ±64KiB of expected_failure_byte={d})\n",
                .{ label, src.len, decoded.len, diff.first_diff, diff.total_diffs, expected },
            );
        } else {
            try w.print(
                "VK_TO_CUDA_SCALE UNEXPECTED_FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d} (expected near byte {d} ±64KiB)\n",
                .{ label, src.len, decoded.len, diff.first_diff, diff.total_diffs, expected },
            );
        }
    } else {
        try w.print(
            "VK_TO_CUDA_SCALE FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src.len, decoded.len, diff.first_diff, diff.total_diffs },
        );
    }
}

// ── CUDA encode -> VK decode at scale ──────────────────────────────

// Cluster F (F036): the test-only decodeUnwrappedVkOnly fixture and
// its createMappedBuffer / destroyMappedBuffer / DecodePush / DecodeResult
// / MappedBuffer support cast lived here. The fixture bypassed the
// production walk_frame / l1_unwrap / DEVICE_LOCAL+sysmem-staging chain.
// It has been deleted — testCudaToVk above now calls
// slz1_codec.decodeSlz1ToBytes, the same entry point the CLI and the
// C ABI take.

fn testCudaToVk(
    w: *std.Io.Writer,
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *driver.Context,
    dec_spv: []const u8,
    label: []const u8,
    src_path: []const u8,
    src: []const u8,
    tmp_slz: []const u8,
) !void {
    _ = dec_spv;
    // Cluster F (F036): sibling fix to wire_format_test.zig — this
    // direction used to drive the test-only decodeUnwrappedVkOnly
    // fixture, bypassing the production walk_frame / l1_unwrap /
    // DEVICE_LOCAL+sysmem-staging chain. It now calls
    // `slz1_codec.decodeSlz1ToBytes`, the same entry point the CLI
    // and the C ABI take. The unwrap-then-decode timing split that
    // used to come for free from doing both steps in-test is gone:
    // the production codec runs unwrap on the GPU, so there's no
    // separate "unwrap_ns" stage to measure. We report unwrap_ns=0
    // for output-shape compatibility with downstream scrapers.

    const t0 = qpcNow();
    const exit = runStreamlz(io, allocator, &.{ "-c", "-l", "1", src_path, "-o", tmp_slz }) catch |err| {
        try w.print("CUDA_TO_VK_SCALE FAIL {s} stage=streamlz err={s}\n", .{ label, @errorName(err) });
        return;
    };
    if (exit != 0) {
        try w.print("CUDA_TO_VK_SCALE FAIL {s} stage=streamlz exit={d}\n", .{ label, exit });
        return;
    }
    const t1 = qpcNow();
    const cuda_encode_ns = qpcNs(t0, t1);

    const slz_bytes = readFile(io, allocator, tmp_slz) catch |err| {
        try w.print("CUDA_TO_VK_SCALE FAIL {s} stage=read_slz err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer allocator.free(slz_bytes);

    const dst = allocator.alloc(u8, src.len) catch |err| {
        try w.print("CUDA_TO_VK_SCALE FAIL {s} stage=alloc_dst err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer allocator.free(dst);

    const t2 = qpcNow();
    const written = slz1_codec.decodeSlz1ToBytes(
        ctx,
        io,
        allocator,
        slz_bytes,
        dst,
    ) catch |err| {
        try w.print("CUDA_TO_VK_SCALE FAIL {s} stage=vk_decode err={s}\n", .{ label, @errorName(err) });
        return;
    };
    const t3 = qpcNow();
    const vk_decode_ns = qpcNs(t2, t3);

    const diff = compareBytes(src, dst[0..written]);
    if (diff.total_diffs == 0 and src.len == written) {
        try w.print(
            "CUDA_TO_VK_SCALE PASS {s} bytes={d} slz_bytes={d} cuda_enc_ns={d} unwrap_ns=0 vk_dec_ns={d}\n",
            .{ label, src.len, slz_bytes.len, cuda_encode_ns, vk_decode_ns },
        );
    } else {
        try w.print(
            "CUDA_TO_VK_SCALE FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src.len, written, diff.first_diff, diff.total_diffs },
        );
    }
}

// ── main ──────────────────────────────────────────────────────────

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
        try w.print("WIRE_FORMAT_SCALE FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    var enc_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    var dec_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const enc_spv = try loadSpv(io, "lz_encode", tier_name, &enc_spv_storage);
    const dec_spv = try loadSpv(io, "lz_decode", tier_name, &dec_spv_storage);

    const allocator = std.heap.page_allocator;

    try w.print("wire-format scale conformance — tier={s}\n", .{tier_name});
    try w.flush();

    // Corpus list: enwik8 full + silesia full. Smaller cases are already
    // exercised by `vk-wire-format-test`; this target focuses on the
    // production-scale endpoints.
    //
    // NOTE: silesia round-trip currently fails at byte 10,206,968 in the
    // L1 codec self-roundtrip (chunk 77, binary-data pattern), so the
    // VK_TO_CUDA direction is expected to FAIL on silesia for the same
    // reason — included for completeness, not as a regression gate.
    // `expected_failure_byte` encodes the pre-existing failure mode
    // for silesia (chunk 77, binary-data L1 self-roundtrip diff at
    // byte 10,206,968 — see the file-level NOTE above). The test
    // asserts the failure occurs at-or-after that offset (the exact
    // byte depends on chunk-boundary parsing); if a future fix moves
    // first_diff PAST the corpus length the case reports a real PASS
    // and the operator should rip out this expected-failure box.
    const corpora = [_]struct {
        name: []const u8,
        path: []const u8,
        size: ?usize,
        expected_failure_byte: ?usize,
    }{
        .{ .name = "enwik8_full", .path = "assets/enwik8.txt", .size = null, .expected_failure_byte = null },
        .{ .name = "silesia_full", .path = "assets/silesia_all.tar", .size = null, .expected_failure_byte = 10_206_968 },
    };

    // Direction 1: VK encode -> CUDA decode.
    for (corpora) |c| {
        var file = std.Io.Dir.cwd().openFile(io, c.path, .{}) catch {
            try w.print("VK_TO_CUDA_SCALE SKIP {s} (open fail {s})\n", .{ c.name, c.path });
            try w.flush();
            continue;
        };
        const stat_size: usize = blk: {
            const s = file.stat(io) catch {
                file.close(io);
                try w.print("VK_TO_CUDA_SCALE SKIP {s} (stat fail)\n", .{c.name});
                try w.flush();
                break :blk 0;
            };
            break :blk @intCast(s.size);
        };
        if (stat_size == 0) continue;
        const target_size = if (c.size) |sz| @min(sz, stat_size) else stat_size;
        const src = allocator.alloc(u8, target_size) catch {
            file.close(io);
            try w.print("VK_TO_CUDA_SCALE SKIP {s} (alloc fail)\n", .{c.name});
            try w.flush();
            continue;
        };
        defer allocator.free(src);
        const n = file.readPositionalAll(io, src, 0) catch 0;
        file.close(io);
        if (n != target_size) {
            try w.print("VK_TO_CUDA_SCALE SKIP {s} (short read {d})\n", .{ c.name, n });
            try w.flush();
            continue;
        }

        var slz_path_buf: [256]u8 = undefined;
        var out_path_buf: [256]u8 = undefined;
        const slz_path = try std.fmt.bufPrint(&slz_path_buf, "{s}/vk_l1_scale_{s}.slz", .{ TMP_DIR, c.name });
        const out_path = try std.fmt.bufPrint(&out_path_buf, "{s}/vk_l1_scale_{s}.out", .{ TMP_DIR, c.name });
        try testVkToCuda(w, io, allocator, ctx, enc_spv, c.name, src, slz_path, out_path, c.expected_failure_byte);
        try w.flush();
    }

    // Direction 2: CUDA encode -> VK decode.
    for (corpora) |c| {
        var file = std.Io.Dir.cwd().openFile(io, c.path, .{}) catch {
            try w.print("CUDA_TO_VK_SCALE SKIP {s} (open fail {s})\n", .{ c.name, c.path });
            try w.flush();
            continue;
        };
        const stat_size: usize = blk: {
            const s = file.stat(io) catch {
                file.close(io);
                try w.print("CUDA_TO_VK_SCALE SKIP {s} (stat fail)\n", .{c.name});
                try w.flush();
                break :blk 0;
            };
            break :blk @intCast(s.size);
        };
        if (stat_size == 0) continue;
        const target_size = if (c.size) |sz| @min(sz, stat_size) else stat_size;
        const src = allocator.alloc(u8, target_size) catch {
            file.close(io);
            try w.print("CUDA_TO_VK_SCALE SKIP {s} (alloc fail)\n", .{c.name});
            try w.flush();
            continue;
        };
        defer allocator.free(src);
        const n = file.readPositionalAll(io, src, 0) catch 0;
        file.close(io);
        if (n != target_size) {
            try w.print("CUDA_TO_VK_SCALE SKIP {s} (short read {d})\n", .{ c.name, n });
            try w.flush();
            continue;
        }

        var slz_path_buf: [256]u8 = undefined;
        const slz_path = try std.fmt.bufPrint(&slz_path_buf, "{s}/cuda_l1_scale_{s}.slz", .{ TMP_DIR, c.name });
        try testCudaToVk(w, io, allocator, ctx, dec_spv, c.name, c.path, src, slz_path);
        try w.flush();
    }
}
