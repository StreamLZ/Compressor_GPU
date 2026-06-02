//! Cross-backend SLZ1 wire-format conformance test.
//!
//! Two directions, three corpora:
//!
//!   * vk_encode_cuda_decode: Vulkan L1 encode -> wrap to SLZ1 -> write
//!     to a temp .slz -> shell out to streamlz.exe -d -> compare the
//!     decoded bytes against the source.
//!
//!   * cuda_encode_vk_decode: streamlz.exe -c -l 1 to produce an .slz
//!     -> read the bytes -> unwrap SLZ1 -> Vulkan L1 decode -> compare.
//!
//! Both tests print a one-line PASS/FAIL summary per corpus, with the
//! first mismatch offset on failure.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const wire_format = @import("wire_format.zig");
// Cluster F (F036): the CUDA_TO_VK direction now goes through the
// production decoder so the test exercises the same code path the CLI
// and the C ABI take. The `wire_constants` / `descriptors` / `dispatch`
// imports went away with the deleted `decodeUnwrappedVkOnly` fixture —
// they were only ever used to build descriptor sets + push constants
// for the test-only lz_decode dispatch that the fixture drove.
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
    const filename = try std.fmt.bufPrint(
        &path_buf,
        "{s}/{s}.{s}.spv",
        .{ SPV_DIR_REL, kernel, tier_name },
    );
    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch return error.SpvOpenFailed;
    defer file.close(io);
    const body = dest[256..];
    const n = file.readPositionalAll(io, body, 0) catch return error.SpvReadFailed;
    if (n == body.len) return error.SpvTooLarge;
    return body[0..n];
}

// ── Vulkan stream readback: copy each chunk's slice from device memory ─
// `encodeL1Multi` leaves the four stream buffers device-resident with the
// memory unmapped.  We re-map each, snapshot only the per-chunk-prefix
// bytes, and concatenate into per-chunk byte slices for wrapping.

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

// ── Shell-out helper for streamlz.exe ─────────────────────────────────

fn runStreamlz(io: std.Io, allocator: std.mem.Allocator, args: []const []const u8) !u32 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    var argv = try aa.alloc([]const u8, 1 + args.len);
    argv[0] = STREAMLZ_EXE;
    for (args, 0..) |a, i| argv[i + 1] = a;

    const result = try std.process.run(aa, io, .{ .argv = argv });
    return switch (result.term) {
        .exited => |code| code,
        else => 1,
    };
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

// ── Comparison helper ─────────────────────────────────────────────────

const Diff = struct {
    first_diff: usize,
    total_diffs: usize,
};

fn compareBytes(a: []const u8, b: []const u8) Diff {
    const n = @min(a.len, b.len);
    var first: usize = 0;
    var found = false;
    var total: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) {
            if (!found) {
                first = i;
                found = true;
            }
            total += 1;
        }
    }
    if (a.len != b.len) {
        if (!found) {
            first = n;
            found = true;
        }
        total += @max(a.len, b.len) - n;
    }
    return .{ .first_diff = first, .total_diffs = total };
}

// ── vk_encode_cuda_decode test ─────────────────────────────────────────

fn testVkEncodeCudaDecode(
    w: *std.Io.Writer,
    io: std.Io,
    allocator: std.mem.Allocator,
    ctx: *driver.Context,
    enc_spv: []const u8,
    label: []const u8,
    src: []const u8,
    tmp_slz: []const u8,
    tmp_out: []const u8,
) !void {
    // 1. Vulkan encode.
    var enc = l1_codec.encodeL1Multi(ctx, src, enc_spv) catch |err| {
        try w.print("VK_TO_CUDA FAIL {s} stage=vk_encode err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer l1_codec.freeStreams(ctx, &enc.streams);

    // 2. Read per-chunk streams from device memory.
    var bundle = buildStreamBundle(allocator, ctx, enc.streams) catch |err| {
        try w.print("VK_TO_CUDA FAIL {s} stage=bundle err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer bundle.deinit(allocator);

    // 3. Wrap into SLZ1.
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
        .chunk_size = enc.streams.chunk_capacity, // unused for sizing — wrap uses VK_CHUNK_SIZE
        .original_size = @intCast(src.len),
    };
    // Override chunk_size to the L1 codec's CHUNK_SIZE (128 KiB) — the
    // chunk_capacity field on the L1Streams bundle is the per-stream
    // device-buffer reservation, not the source-side chunk granularity.
    var streams_view_fixed = streams_view;
    streams_view_fixed.chunk_size = wire_format.VK_CHUNK_SIZE;

    const bound = wire_format.wrapBound(streams_view_fixed);
    const slz_buf = allocator.alloc(u8, bound) catch {
        try w.print("VK_TO_CUDA FAIL {s} stage=alloc_wrap\n", .{label});
        return;
    };
    defer allocator.free(slz_buf);
    const slz_size = wire_format.wrapL1ToSlz1(streams_view_fixed, slz_buf) catch |err| {
        try w.print("VK_TO_CUDA FAIL {s} stage=wrap err={s}\n", .{ label, @errorName(err) });
        return;
    };

    // 4. Write to disk.
    writeFile(io, tmp_slz, slz_buf[0..slz_size]) catch |err| {
        try w.print("VK_TO_CUDA FAIL {s} stage=write err={s}\n", .{ label, @errorName(err) });
        return;
    };

    // 5. Shell out to streamlz.exe -d.
    const exit = runStreamlz(io, allocator, &.{ "-d", tmp_slz, "-o", tmp_out }) catch |err| {
        try w.print("VK_TO_CUDA FAIL {s} stage=streamlz err={s}\n", .{ label, @errorName(err) });
        return;
    };
    if (exit != 0) {
        try w.print(
            "VK_TO_CUDA FAIL {s} stage=streamlz exit_code={d} slz_bytes={d}\n",
            .{ label, exit, slz_size },
        );
        return;
    }

    // 6. Read decoded bytes.
    const decoded = readFile(io, allocator, tmp_out) catch |err| {
        try w.print("VK_TO_CUDA FAIL {s} stage=read_decoded err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer allocator.free(decoded);

    // 7. Compare.
    const diff = compareBytes(src, decoded);
    if (diff.total_diffs == 0 and src.len == decoded.len) {
        try w.print(
            "VK_TO_CUDA PASS {s} bytes={d} slz_bytes={d}\n",
            .{ label, src.len, slz_size },
        );
    } else {
        try w.print(
            "VK_TO_CUDA FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src.len, decoded.len, diff.first_diff, diff.total_diffs },
        );
        const d = diff.first_diff;
        const start = if (d >= 16) d - 16 else 0;
        const end = @min(d + 16, @min(src.len, decoded.len));
        try w.print("DEBUG src[{d}..{d}] = ", .{ start, end });
        for (src[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG dec[{d}..{d}] = ", .{ start, end });
        for (decoded[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
    }
}

// ── cuda_encode_vk_decode test ─────────────────────────────────────────

fn testCudaEncodeVkDecode(
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
    // Cluster F (F036): this test used to drive a test-only
    // `decodeUnwrappedVkOnly` fixture that bypassed the GPU pipeline
    // (no walk_frame, no l1_unwrap, no DEVICE_LOCAL+sysmem staging
    // readback). That meant the CUDA_TO_VK direction NEVER exercised
    // the production decoder — the one the CLI and the C ABI call —
    // so a real regression there would have gone undetected. We now
    // drive `slz1_codec.decodeSlz1ToBytes`, the same entry point
    // `slzDecompressHost_vk` uses.

    // 1. Shell out to streamlz.exe -c -l 1 (default sc = 0.5).
    //    The Vulkan L1 decoder now consumes off32 entries natively,
    //    so the previous `--sc 0.25` workaround (which pinned the
    //    CUDA encoder to 64 KiB sub-chunks to avoid off32) is no
    //    longer required.
    const exit = runStreamlz(io, allocator, &.{ "-c", "-l", "1", src_path, "-o", tmp_slz }) catch |err| {
        try w.print("CUDA_TO_VK FAIL {s} stage=streamlz err={s}\n", .{ label, @errorName(err) });
        return;
    };
    if (exit != 0) {
        try w.print("CUDA_TO_VK FAIL {s} stage=streamlz exit={d}\n", .{ label, exit });
        return;
    }

    // 2. Read the CUDA-produced .slz bytes.
    const slz_bytes = readFile(io, allocator, tmp_slz) catch |err| {
        try w.print("CUDA_TO_VK FAIL {s} stage=read_slz err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer allocator.free(slz_bytes);

    // 3. Drive the production VK decoder. Output buffer is sized to
    //    src.len exactly — `decodeSlz1ToBytes` returns the original
    //    size from the frame header, which must equal src.len for a
    //    correct round-trip.
    const dst = allocator.alloc(u8, src.len) catch |err| {
        try w.print("CUDA_TO_VK FAIL {s} stage=alloc_dst err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer allocator.free(dst);

    const written = slz1_codec.decodeSlz1ToBytes(
        ctx,
        io,
        allocator,
        slz_bytes,
        dst,
    ) catch |err| {
        try w.print("CUDA_TO_VK FAIL {s} stage=vk_decode err={s}\n", .{ label, @errorName(err) });
        return;
    };

    // 4. Compare.
    const diff = compareBytes(src, dst[0..written]);
    if (diff.total_diffs == 0 and src.len == written) {
        try w.print(
            "CUDA_TO_VK PASS {s} bytes={d} slz_bytes={d}\n",
            .{ label, src.len, slz_bytes.len },
        );
    } else {
        try w.print(
            "CUDA_TO_VK FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src.len, written, diff.first_diff, diff.total_diffs },
        );
        const d = diff.first_diff;
        const start = if (d >= 16) d - 16 else 0;
        const end = @min(d + 16, @min(src.len, written));
        try w.print("DEBUG src[{d}..{d}] = ", .{ start, end });
        for (src[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\nDEBUG dec[{d}..{d}] = ", .{ start, end });
        for (dst[start..end]) |b| try w.print("{x:0>2} ", .{b});
        try w.print("\n", .{});
    }
}

// Cluster F (F036): the `decodeUnwrappedVkOnly` fixture that used to
// live here drove the lz_decode shader directly from host-uploaded
// per-chunk stream slices, bypassing the production GPU walk_frame /
// l1_unwrap / DEVICE_LOCAL+sysmem-staging chain. It has been deleted —
// `testCudaEncodeVkDecode` above now calls `slz1_codec.decodeSlz1ToBytes`,
// the same entry point the CLI and the C ABI take. The chunk-descriptor
// + bind helpers that supported the fixture (createMappedBuffer,
// destroyMappedBuffer, DecodePush, DecodeResult, MappedBuffer) were
// deleted with it; their absence is enforced by the unused-decl warnings
// in the test compile.

// ── main ──────────────────────────────────────────────────────────────

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
        try w.print("WIRE_FORMAT FAIL tier=unsupported\n", .{});
        return error.UnsupportedTier;
    };

    var enc_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    var dec_spv_storage: [MAX_SPV_BYTES]u8 align(4) = undefined;
    const enc_spv = try loadSpv(io, "lz_encode", tier_name, &enc_spv_storage);
    const dec_spv = try loadSpv(io, "lz_decode", tier_name, &dec_spv_storage);

    const allocator = std.heap.page_allocator;

    try w.print("wire-format conformance — tier={s}\n", .{tier_name});
    try w.flush();

    // ── Corpus list ─────────────────────────────────────────────────
    const corpora = [_]struct { name: []const u8, path: []const u8, size: ?usize }{
        .{ .name = "web_full", .path = "assets/web.txt", .size = null },
        .{ .name = "enwik8_1mb", .path = "assets/enwik8.txt", .size = 1024 * 1024 },
        .{ .name = "enwik8_4mb", .path = "assets/enwik8.txt", .size = 4 * 1024 * 1024 },
    };

    // ── Direction 1: vk_encode â†’ cuda_decode ─────────────────────────
    for (corpora) |c| {
        var file = std.Io.Dir.cwd().openFile(io, c.path, .{}) catch {
            try w.print("VK_TO_CUDA SKIP {s} (open fail {s})\n", .{ c.name, c.path });
            try w.flush();
            continue;
        };
        const stat_size: usize = blk: {
            const s = file.stat(io) catch {
                file.close(io);
                try w.print("VK_TO_CUDA SKIP {s} (stat fail)\n", .{c.name});
                try w.flush();
                break :blk 0;
            };
            break :blk @intCast(s.size);
        };
        if (stat_size == 0) continue;
        const target_size = if (c.size) |sz| @min(sz, stat_size) else stat_size;
        const src = allocator.alloc(u8, target_size) catch {
            file.close(io);
            try w.print("VK_TO_CUDA SKIP {s} (alloc fail)\n", .{c.name});
            try w.flush();
            continue;
        };
        defer allocator.free(src);
        const n = file.readPositionalAll(io, src, 0) catch 0;
        file.close(io);
        if (n != target_size) {
            try w.print("VK_TO_CUDA SKIP {s} (short read {d})\n", .{ c.name, n });
            try w.flush();
            continue;
        }

        var slz_path_buf: [256]u8 = undefined;
        var out_path_buf: [256]u8 = undefined;
        const slz_path = try std.fmt.bufPrint(&slz_path_buf, "{s}/vk_l1_{s}.slz", .{ TMP_DIR, c.name });
        const out_path = try std.fmt.bufPrint(&out_path_buf, "{s}/vk_l1_{s}.out", .{ TMP_DIR, c.name });
        try testVkEncodeCudaDecode(w, io, allocator, ctx, enc_spv, c.name, src, slz_path, out_path);
        try w.flush();
    }

    // ── Direction 2: cuda_encode â†’ vk_decode ─────────────────────────
    for (corpora) |c| {
        var file = std.Io.Dir.cwd().openFile(io, c.path, .{}) catch {
            try w.print("CUDA_TO_VK SKIP {s} (open fail {s})\n", .{ c.name, c.path });
            try w.flush();
            continue;
        };
        const stat_size: usize = blk: {
            const s = file.stat(io) catch {
                file.close(io);
                try w.print("CUDA_TO_VK SKIP {s} (stat fail)\n", .{c.name});
                try w.flush();
                break :blk 0;
            };
            break :blk @intCast(s.size);
        };
        if (stat_size == 0) continue;
        const target_size = if (c.size) |sz| @min(sz, stat_size) else stat_size;
        const src = allocator.alloc(u8, target_size) catch {
            file.close(io);
            try w.print("CUDA_TO_VK SKIP {s} (alloc fail)\n", .{c.name});
            try w.flush();
            continue;
        };
        defer allocator.free(src);
        const n = file.readPositionalAll(io, src, 0) catch 0;
        file.close(io);
        if (n != target_size) {
            try w.print("CUDA_TO_VK SKIP {s} (short read {d})\n", .{ c.name, n });
            try w.flush();
            continue;
        }

        // streamlz.exe -c reads from disk; for prefix-sized tests we'd
        // need a temp source file.  For web_full we can use the corpus
        // directly; for the enwik8 prefixes write a temp truncated copy.
        var slz_path_buf: [256]u8 = undefined;
        const slz_path = try std.fmt.bufPrint(&slz_path_buf, "{s}/cuda_l1_{s}.slz", .{ TMP_DIR, c.name });
        var input_path_buf: [256]u8 = undefined;
        var input_path: []const u8 = c.path;
        if (c.size) |_| {
            input_path = try std.fmt.bufPrint(
                &input_path_buf,
                "{s}/src_{s}.bin",
                .{ TMP_DIR, c.name },
            );
            writeFile(io, input_path, src) catch {
                try w.print("CUDA_TO_VK SKIP {s} (write temp src fail)\n", .{c.name});
                try w.flush();
                continue;
            };
        }
        try testCudaEncodeVkDecode(w, io, allocator, ctx, dec_spv, c.name, input_path, src, slz_path);
        try w.flush();
    }
}
