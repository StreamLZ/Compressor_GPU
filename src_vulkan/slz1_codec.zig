//! End-to-end SLZ1-format codec built on top of `l1_codec` (raw stream
//! encode/decode) + `wire_format` (CPU-side SLZ1 wrap/unwrap).
//!
//! Public API (level=1 only — the only mode the Vulkan port implements
//! today):
//!
//!   pub fn encodeL1ToSlz1(ctx, src, out) !usize
//!     Vulkan-encode `src`, wrap into a real .slz frame, write into `out`.
//!     Returns the number of bytes written. `out.len` must be at least
//!     `slz1Bound(src.len)`.
//!
//!   pub fn decodeSlz1ToBytes(ctx, allocator, slz_bytes, out) !usize
//!     Parse `slz_bytes` as an SLZ1 frame, run the Vulkan L1 decoder,
//!     write decoded bytes into `out`. Returns the byte count written.
//!     `out.len` must be at least `original_size` advertised by the
//!     frame header.
//!
//!   pub fn slz1Bound(input_size: usize) usize
//!     Worst-case .slz size for an L1-encoded input of `input_size` bytes.
//!     Loose upper bound: original size + 6.25% expansion pad + 8 KiB of
//!     wire-format headers + per-chunk overhead.
//!
//! Implementation notes:
//!
//!   * SPV blobs are loaded at runtime from `zig-out/shaders/<kernel>.<tier>.spv`
//!     via the same pattern the wire-format test uses. The C ABI layer
//!     that wraps these functions therefore inherits the requirement that
//!     callers either run with cwd set to the repo root, or that the SPV
//!     blobs are deployed alongside the binary.
//!   * `decodeSlz1ToBytes` reproduces the wire-format test's
//!     `decodeUnwrappedVkOnly` path: it uploads the unwrapped per-chunk
//!     stream bytes into fresh device buffers and drives the lz_decode
//!     shader directly. Splitting that logic out of the test file means
//!     both the test harness AND the C ABI / CLI consumers go through
//!     the same code path.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const wire_format = @import("wire_format.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");

const MAX_SPV_BYTES: usize = 1 << 20;
const SPV_DIR_REL: []const u8 = "zig-out/shaders";

pub const Slz1Error = error{
    UnsupportedTier,
    SpvOpenFailed,
    SpvReadFailed,
    SpvTooLarge,
    OutOfMemory,
    OutputTooSmall,
    BufferCreateFailed,
    MemoryAllocateFailed,
    MemoryTypeNotFound,
    BindBufferFailed,
    MapMemoryFailed,
    BadFrame,
    BadLevel,
} ||
    l1_codec.L1Error ||
    wire_format.WrapError ||
    wire_format.UnwrapError ||
    descriptors.DescriptorError ||
    dispatch.DispatchError ||
    driver.DriverError;

/// Loose upper bound for the .slz size of an L1-encoded input. Reuses
/// the same formula as `slzCompressBound_vk` and is a safe overestimate
/// for `wire_format.wrapBound` once the input is actually encoded — we
/// don't know the encoder-side per-chunk stream sizes upfront, so we
/// take the worst-case `2 * src + overhead` budget the L1 codec spec
/// reserves and round it up to a friendly headroom.
pub fn slz1Bound(input_size: usize) usize {
    // Worst-case wire-format overhead per chunk (lit_size + cmd_size +
    // off16 + length + headers) is dominated by the 2*src factor from
    // the L1 codec's spec §4.1; in practice the actual ratio is well
    // under 1.2x for compressible data. We pre-allocate 2x + 8 KiB
    // global headers to cover the worst case without forcing the
    // encoder to spill.
    const stream_bound = input_size * 2;
    const fixed = 8 * 1024;
    // Per-chunk headers (~24 bytes each), assume ceil(input/CHUNK)+1.
    const n_chunks_est = (input_size / l1_codec.CHUNK_SIZE) + 2;
    const per_chunk_hdr = n_chunks_est * 64;
    return stream_bound + fixed + per_chunk_hdr;
}

fn tierName(t: probe_mod.Tier) ?[]const u8 {
    return switch (t) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => null,
    };
}

fn loadSpv(io: std.Io, kernel: []const u8, tier_name: []const u8, dest: []u8) Slz1Error![]u8 {
    var path_buf: [256]u8 = undefined;
    const filename = std.fmt.bufPrint(
        &path_buf,
        "{s}/{s}.{s}.spv",
        .{ SPV_DIR_REL, kernel, tier_name },
    ) catch return error.SpvOpenFailed;

    var file = std.Io.Dir.cwd().openFile(io, filename, .{}) catch return error.SpvOpenFailed;
    defer file.close(io);

    const n = file.readPositionalAll(io, dest, 0) catch return error.SpvReadFailed;
    if (n == dest.len) return error.SpvTooLarge;
    return dest[0..n];
}

// ── Encode path: src → SLZ1 ──────────────────────────────────────────

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

    fn deinit(self: *StreamBundle, allocator: std.mem.Allocator) void {
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
) Slz1Error!void {
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
) Slz1Error!void {
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
) Slz1Error!StreamBundle {
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

/// Vulkan-encode `src` into the L1 streams, wrap into SLZ1, write into `out`.
/// Returns bytes written. Uses `driver.g_default` after ensureInit.
pub fn encodeL1ToSlz1(
    ctx: *driver.Context,
    io: std.Io,
    allocator: std.mem.Allocator,
    src: []const u8,
    out: []u8,
) Slz1Error!usize {
    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const tier_name = tierName(tier) orelse return error.UnsupportedTier;

    const enc_spv_storage = try allocator.alignedAlloc(u8, .@"4", MAX_SPV_BYTES);
    defer allocator.free(enc_spv_storage);
    const enc_spv = try loadSpv(io, "lz_encode", tier_name, enc_spv_storage);

    var enc = try l1_codec.encodeL1Multi(ctx, src, enc_spv);
    defer l1_codec.freeStreams(ctx, &enc.streams);

    var bundle = try buildStreamBundle(allocator, ctx, enc.streams);
    defer bundle.deinit(allocator);

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
    if (out.len < bound) return error.OutputTooSmall;
    return try wire_format.wrapL1ToSlz1(streams_view, out);
}

// ── Decode path: SLZ1 → src ──────────────────────────────────────────

const DecodePush = extern struct {
    n_chunks: u32,
};

const MappedBuffer = struct {
    buf: vk.VkBuffer,
    mem: vk.VkDeviceMemory,
    mapped: [*]u8,
    size: vk.VkDeviceSize,
};

fn createMappedBuffer(ctx: *driver.Context, size: vk.VkDeviceSize) Slz1Error!MappedBuffer {
    // Make sure the per-device function pointer slots (vkCreateBuffer
    // etc.) are populated. They get filled lazily on first use — on a
    // decode-only process invocation (e.g. `streamlz_vk -d`) the encode
    // path that normally fills them via l1_codec.encodeL1Multi never
    // runs, so the slots are still null at this point. Idempotent —
    // subsequent calls are no-ops.
    l1_codec.ensureBufferFnSlots(ctx);
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
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const fallback_flags = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        if (supported and (flags & want_flags) == want_flags) {
            mt_idx = i;
            break;
        }
    }
    if (mt_idx == std.math.maxInt(u32)) {
        var j: u32 = 0;
        while (j < mem_props.memoryTypeCount) : (j += 1) {
            const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(j))) != 0;
            const flags = mem_props.memoryTypes[j].propertyFlags;
            if (supported and (flags & fallback_flags) == fallback_flags) {
                mt_idx = j;
                break;
            }
        }
    }
    if (mt_idx == std.math.maxInt(u32)) return error.MemoryTypeNotFound;

    const mai: vk.VkMemoryAllocateInfo = .{
        .allocationSize = req.size,
        .memoryTypeIndex = mt_idx,
    };
    var mem: vk.VkDeviceMemory = null;
    if (alloc_mem(ctx.dev, &mai, null, &mem) != vk.VK_SUCCESS) return error.MemoryAllocateFailed;
    if (bind(ctx.dev, buf, mem, 0) != vk.VK_SUCCESS) return error.BindBufferFailed;

    var raw: ?*anyopaque = null;
    if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS)
        return error.MapMemoryFailed;
    const mapped: [*]u8 = @ptrCast(@alignCast(raw.?));

    return .{ .buf = buf, .mem = mem, .mapped = mapped, .size = size };
}

fn destroyMappedBuffer(ctx: *driver.Context, b: *MappedBuffer) void {
    if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, b.buf, null);
    if (vk.vkFreeMemory_fn) |f| f(ctx.dev, b.mem, null);
    b.buf = null;
    b.mem = null;
}

/// Unwrap an SLZ1 frame, run the Vulkan L1 decoder, write decoded bytes
/// into `out`. Returns the byte count written (== original_size).
pub fn decodeSlz1ToBytes(
    ctx: *driver.Context,
    io: std.Io,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    out: []u8,
) Slz1Error!usize {
    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const tier_name = tierName(tier) orelse return error.UnsupportedTier;

    const dec_spv_storage = try allocator.alignedAlloc(u8, .@"4", MAX_SPV_BYTES);
    defer allocator.free(dec_spv_storage);
    const dec_spv = try loadSpv(io, "lz_decode", tier_name, dec_spv_storage);

    var stream_view: wire_format.PerChunkStreams = undefined;
    var unwrap = try wire_format.unwrapSlz1ToL1Streams(allocator, slz_bytes, &stream_view);
    defer wire_format.freeUnwrapStorage(allocator, &unwrap.storage);

    const dst_total: u32 = @intCast(unwrap.result.original_size);
    if (out.len < dst_total) return error.OutputTooSmall;

    const chunk_capacity = l1_codec.CHUNK_STREAM_CAPACITY;
    const n_chunks = unwrap.result.n_chunks;
    const stream_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * chunk_capacity;
    const off32_capacity = l1_codec.CHUNK_OFF32_CAPACITY;
    const off32_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * off32_capacity;

    var lit_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &lit_b);
    var cmd_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &cmd_b);
    var off16_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &off16_b);
    var length_b = try createMappedBuffer(ctx, stream_cap_total);
    defer destroyMappedBuffer(ctx, &length_b);
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
        const cap_words: u32 = chunk_capacity / 4;
        const off32_cap_words: u32 = off32_capacity / 4;
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const dst_off: u32 = ci * unwrap.result.chunk_size;
            const dst_size: u32 = unwrap.result.per_chunk_decomp_size[ci];
            const word_base: u32 = ci * cap_words;
            const base = ci * 16;
            words[base + 0] = dst_off;
            words[base + 1] = dst_size;
            words[base + 2] = word_base;
            words[base + 3] = @intCast(stream_view.cmd_bytes[ci].len);
            words[base + 4] = word_base;
            words[base + 5] = @intCast(stream_view.lit_bytes[ci].len);
            words[base + 6] = word_base;
            words[base + 7] = @intCast(stream_view.off16_bytes[ci].len / 2);
            words[base + 8] = word_base;
            words[base + 9] = @intCast(stream_view.length_bytes[ci].len);
            if (stream_view.per_chunk_initial_copy) |pci| {
                if (ci < pci.len) words[base + 10] = pci[ci];
            }
            words[base + 11] = ci * off32_cap_words;
            if (stream_view.per_chunk_off32_count1) |a| if (ci < a.len) { words[base + 12] = a[ci]; };
            if (stream_view.per_chunk_off32_count2) |a| if (ci < a.len) { words[base + 13] = a[ci]; };
            if (stream_view.per_chunk_cmd_stream2_offset) |a| if (ci < a.len) { words[base + 14] = a[ci]; };
        }
    }

    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_dec = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_decode",
        tier,
        dec_spv,
        7,
        @sizeOf(DecodePush),
    );

    const dec_bindings: [7]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = cmd_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = lit_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off16_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = length_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off32_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    const dec_push: DecodePush = .{ .n_chunks = n_chunks };
    var dec_push_bytes: [@sizeOf(DecodePush)]u8 = undefined;
    @memcpy(dec_push_bytes[0..], std.mem.asBytes(&dec_push));

    _ = try dispatch.submitOne(
        ctx,
        cached_dec.pipeline,
        cached_dec.pipeline_layout,
        dec_set,
        dec_push_bytes[0..],
        .{ n_chunks, 1, 1 },
    );

    @memcpy(out[0..dst_total], dst_b.mapped[0..dst_total]);
    return dst_total;
}
