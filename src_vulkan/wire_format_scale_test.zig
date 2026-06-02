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
const wire_constants = @import("wire_constants.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");

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
        try w.print(
            "VK_TO_CUDA_SCALE PASS {s} bytes={d} slz_bytes={d} vk_enc_ns={d} bundle_ns={d} wrap_ns={d} cuda_dec_ns={d}\n",
            .{ label, src.len, slz_size, vk_encode_ns, bundle_ns, wrap_ns, cuda_decode_ns },
        );
    } else {
        try w.print(
            "VK_TO_CUDA_SCALE FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src.len, decoded.len, diff.first_diff, diff.total_diffs },
        );
    }
}

// ── CUDA encode -> VK decode at scale ──────────────────────────────

const DecodePush = extern struct { n_chunks: u32 };

const MappedBuffer = struct { buf: vk.VkBuffer, mem: vk.VkDeviceMemory, mapped: [*]u8, size: vk.VkDeviceSize };

fn createMappedBuffer(ctx: *driver.Context, size: vk.VkDeviceSize) !MappedBuffer {
    const create_buf = vk.vkCreateBuffer_fn orelse return error.BufferCreateFailed;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse return error.BufferCreateFailed;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return error.MemoryAllocateFailed;
    const bind = vk.vkBindBufferMemory_fn orelse return error.BindBufferFailed;
    const map = vk.vkMapMemory_fn orelse return error.MapMemoryFailed;
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return error.MemoryTypeNotFound;

    const bci: vk.VkBufferCreateInfo = .{
        .size = size,
        .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) return error.BufferCreateFailed;
    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(ctx.pd, &mem_props);
    var mt_idx: u32 = std.math.maxInt(u32);
    const want_flags = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const fallback_flags = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        if (supported and (flags & want_flags) == want_flags) { mt_idx = i; break; }
    }
    if (mt_idx == std.math.maxInt(u32)) {
        var j: u32 = 0;
        while (j < mem_props.memoryTypeCount) : (j += 1) {
            const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(j))) != 0;
            const flags = mem_props.memoryTypes[j].propertyFlags;
            if (supported and (flags & fallback_flags) == fallback_flags) { mt_idx = j; break; }
        }
    }
    if (mt_idx == std.math.maxInt(u32)) return error.MemoryTypeNotFound;

    const mai: vk.VkMemoryAllocateInfo = .{ .allocationSize = req.size, .memoryTypeIndex = mt_idx };
    var mem: vk.VkDeviceMemory = null;
    if (alloc_mem(ctx.dev, &mai, null, &mem) != vk.VK_SUCCESS) return error.MemoryAllocateFailed;
    if (bind(ctx.dev, buf, mem, 0) != vk.VK_SUCCESS) return error.BindBufferFailed;
    var raw: ?*anyopaque = null;
    if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS) return error.MapMemoryFailed;
    const mapped: [*]u8 = @ptrCast(@alignCast(raw.?));
    return .{ .buf = buf, .mem = mem, .mapped = mapped, .size = size };
}

fn destroyMappedBuffer(ctx: *driver.Context, b: *MappedBuffer) void {
    if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, b.buf, null);
    if (vk.vkFreeMemory_fn) |f| f(ctx.dev, b.mem, null);
    b.buf = null;
    b.mem = null;
}

const DecodeResult = struct { bytes: []u8 };

fn decodeUnwrappedVkOnly(
    allocator: std.mem.Allocator,
    ctx: *driver.Context,
    dec_spv: []const u8,
    unwrap: *const wire_format.UnwrapBundle,
    stream_view: *const wire_format.PerChunkStreams,
) !?DecodeResult {
    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const chunk_capacity = l1_codec.CHUNK_STREAM_CAPACITY;
    const n_chunks = unwrap.result.n_chunks;
    const stream_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * chunk_capacity;

    var lit_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &lit_b);
    var cmd_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &cmd_b);
    var off16_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &off16_b);
    var length_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &length_b);
    const off32_capacity = l1_codec.CHUNK_OFF32_CAPACITY;
    const off32_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * off32_capacity;
    var off32_b = try createMappedBuffer(ctx, off32_cap_total);
    defer destroyMappedBuffer(ctx, &off32_b);

    @memset(lit_b.mapped[0..@intCast(lit_b.size)], 0);
    @memset(cmd_b.mapped[0..@intCast(cmd_b.size)], 0);
    @memset(off16_b.mapped[0..@intCast(off16_b.size)], 0);
    @memset(length_b.mapped[0..@intCast(length_b.size)], 0);
    @memset(off32_b.mapped[0..@intCast(off32_b.size)], 0);

    {
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base: usize = @as(usize, ci) * chunk_capacity;
            const off32_base: usize = @as(usize, ci) * off32_capacity;
            const ls = stream_view.lit_bytes[ci].len;
            const cs = stream_view.cmd_bytes[ci].len;
            const ob = stream_view.off16_bytes[ci].len;
            const xs = stream_view.length_bytes[ci].len;
            if (ls != 0) @memcpy(lit_b.mapped[base..][0..ls], stream_view.lit_bytes[ci]);
            if (cs != 0) @memcpy(cmd_b.mapped[base..][0..cs], stream_view.cmd_bytes[ci]);
            if (ob != 0) @memcpy(off16_b.mapped[base..][0..ob], stream_view.off16_bytes[ci]);
            if (xs != 0) @memcpy(length_b.mapped[base..][0..xs], stream_view.length_bytes[ci]);
            if (stream_view.off32_bytes) |arr| {
                const ob32 = arr[ci].len;
                if (ob32 != 0) @memcpy(off32_b.mapped[off32_base..][0..ob32], arr[ci]);
            }
        }
    }

    const dst_total: u32 = @intCast(unwrap.result.original_size);
    const dst_buf_size: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 4),
        (@as(vk.VkDeviceSize, dst_total) + 3) & ~@as(vk.VkDeviceSize, 3),
    );
    var dst_b = try createMappedBuffer(ctx, dst_buf_size);
    defer destroyMappedBuffer(ctx, &dst_b);
    @memset(dst_b.mapped[0..@intCast(dst_b.size)], 0);

    var chunks_b = try createMappedBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 16);
    defer destroyMappedBuffer(ctx, &chunks_b);
    @memset(chunks_b.mapped[0..@intCast(chunks_b.size)], 0);
    {
        const words: [*]u32 = @ptrCast(@alignCast(chunks_b.mapped));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const dst_off: u32 = ci * unwrap.result.chunk_size;
            const dst_size: u32 = unwrap.result.per_chunk_decomp_size[ci];
            // BYTE base (post-l1_unwrap port — see lz_decode.comp head
            // comment; slots 2/4/6/8/11 are byte offsets into the
            // corresponding stream SSBO, lz_decode no longer multiplies
            // by 4).
            const stream_byte_base: u32 = ci * chunk_capacity;
            const off32_byte_base: u32 = ci * off32_capacity;
            const base = ci * 16;
            const ic: u32 = if (stream_view.per_chunk_initial_copy) |pci|
                (if (ci < pci.len) pci[ci] else 0)
            else
                0;
            words[base + 0] = dst_off;
            words[base + 1] = dst_size;
            words[base + 2] = stream_byte_base;
            words[base + 3] = @intCast(stream_view.cmd_bytes[ci].len);
            // CPU unwrap spliced the init prefix at the head of
            // lit_bytes[ci]. lit_byte_base / lit_size point at the
            // token-loop slice (after prefix); init_byte_base = prefix.
            words[base + 4] = stream_byte_base + ic;
            words[base + 5] = @intCast(stream_view.lit_bytes[ci].len - ic);
            words[base + 6] = stream_byte_base;
            words[base + 7] = @intCast(stream_view.off16_bytes[ci].len / 2);
            words[base + 8] = stream_byte_base;
            words[base + 9] = @intCast(stream_view.length_bytes[ci].len);
            words[base + 10] = ic;
            words[base + 11] = off32_byte_base;
            if (stream_view.per_chunk_off32_count1) |a| if (ci < a.len) { words[base + 12] = a[ci]; };
            if (stream_view.per_chunk_off32_count2) |a| if (ci < a.len) { words[base + 13] = a[ci]; };
            if (stream_view.per_chunk_cmd_stream2_offset) |a| if (ci < a.len) { words[base + 14] = a[ci]; };
            // Slot 15 = init_byte_base. Prefix sits at the head of the
            // per-chunk lit slice.
            words[base + 15] = stream_byte_base;
        }
    }

    // Cluster A wiring: walk-format chunks buffer for lz_decode binding 7.
    var walk_chunks_b = try createMappedBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * wire_constants.CHUNK_DESC_U32_COUNT,
    );
    defer destroyMappedBuffer(ctx, &walk_chunks_b);
    @memset(walk_chunks_b.mapped[0..@intCast(walk_chunks_b.size)], 0);
    {
        const wcw: [*]u32 = @ptrCast(@alignCast(walk_chunks_b.mapped));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const dst_off: u32 = ci * unwrap.result.chunk_size;
            const dst_size: u32 = unwrap.result.per_chunk_decomp_size[ci];
            const wbase = ci * wire_constants.CHUNK_DESC_U32_COUNT;
            wcw[wbase + wire_constants.CHUNK_DECOMP_SIZE_SLOT] = dst_size;
            wcw[wbase + wire_constants.CHUNK_DST_OFFSET_SLOT] = dst_off;
        }
    }

    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_dec = try descriptors.getOrCreate(
        ctx, &cache, "lz_decode", tier, dec_spv, 8, @sizeOf(DecodePush),
    );

    const dec_bindings: [8]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = cmd_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = lit_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off16_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = length_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off32_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = walk_chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    const dec_push: DecodePush = .{ .n_chunks = n_chunks };
    var dec_push_bytes: [@sizeOf(DecodePush)]u8 = undefined;
    @memcpy(dec_push_bytes[0..], std.mem.asBytes(&dec_push));
    _ = try dispatch.submitOne(
        ctx, cached_dec.pipeline, cached_dec.pipeline_layout, dec_set,
        dec_push_bytes[0..], .{ n_chunks, 1, 1 },
    );

    const dst_bytes = try allocator.alloc(u8, dst_total);
    errdefer allocator.free(dst_bytes);
    @memcpy(dst_bytes, dst_b.mapped[0..dst_total]);
    return DecodeResult{ .bytes = dst_bytes };
}

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

    var stream_view: wire_format.PerChunkStreams = undefined;
    var unwrap = wire_format.unwrapSlz1ToL1Streams(allocator, slz_bytes, &stream_view) catch |err| {
        try w.print("CUDA_TO_VK_SCALE FAIL {s} stage=unwrap err={s}\n", .{ label, @errorName(err) });
        return;
    };
    defer wire_format.freeUnwrapStorage(allocator, &unwrap.storage);
    const t2 = qpcNow();
    const unwrap_ns = qpcNs(t1, t2);

    if (try decodeUnwrappedVkOnly(allocator, ctx, dec_spv, &unwrap, &stream_view)) |result| {
        defer allocator.free(result.bytes);
        const t3 = qpcNow();
        const vk_decode_ns = qpcNs(t2, t3);

        const diff = compareBytes(src, result.bytes);
        if (diff.total_diffs == 0 and src.len == result.bytes.len) {
            try w.print(
                "CUDA_TO_VK_SCALE PASS {s} bytes={d} slz_bytes={d} cuda_enc_ns={d} unwrap_ns={d} vk_dec_ns={d}\n",
                .{ label, src.len, slz_bytes.len, cuda_encode_ns, unwrap_ns, vk_decode_ns },
            );
        } else {
            try w.print(
                "CUDA_TO_VK_SCALE FAIL {s} bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
                .{ label, src.len, result.bytes.len, diff.first_diff, diff.total_diffs },
            );
        }
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
    const corpora = [_]struct { name: []const u8, path: []const u8, size: ?usize }{
        .{ .name = "enwik8_full", .path = "assets/enwik8.txt", .size = null },
        .{ .name = "silesia_full", .path = "assets/silesia_all.tar", .size = null },
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
        try testVkToCuda(w, io, allocator, ctx, enc_spv, c.name, src, slz_path, out_path);
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
