//! GPU-side SLZ1 wire-format wrap.
//!
//! Mirrors the CUDA encoder's device-resident frame-assembly path.
//! The CPU reference is `wire_format.wrapL1ToSlz1` (in `wire_format.zig`) —
//! this module produces byte-identical output without staging the per-chunk
//! LZ streams or any frame/block header bytes through host memory.
//!
//! Pipeline (mirrors `src/encode/encode_assemble.zig`):
//!
//!   1. Build `AssembleMeasureDesc` (8 u32/chunk) from per-chunk
//!      stream sizes + chunk geometry, upload, dispatch
//!      `assemble_measure.comp` → per-chunk sub-chunk payload size.
//!   2. Dispatch `frame_layout.comp` (single-threaded GPU prefix-sum):
//!      reads measure output, writes `AssembleWriteDesc` slot-0 +
//!      `FrameChunkDesc` (all 4 slots) + `FrameMeta` (total_frame_size,
//!      sc_tail_off, end_mark_off, asm_total, block_payload_size).
//!   3. Dispatch `assemble_write.comp` → assembled sub-chunk payloads
//!      laid out back-to-back in `asm_out_buf`.
//!   4. Dispatch `frame_assemble.comp` (n_chunks + 1 workgroups) → splice
//!      sub-chunk payloads into per-chunk slots AND synthesize the frame
//!      header + block header + SC tail prefix + end mark directly into
//!      `frame_out_buf` (no host prefix upload).
//!   5. Read back `frame_meta_buf[0]` (total_frame_size) and memcpy
//!      `frame_out_buf[0..total_frame_size]` → `out_host`.
//!
//! No host-side computation of per-chunk offsets, no host header bytes
//! uploaded, no host fallback toggle — the GPU pipeline IS the path.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");
const wire_constants = @import("wire_constants.zig");
const spv_blobs = @import("spv_blobs");

// Frame-format constants (codec / level / block_size / sc_group_size /
// flags / block_size_log2 base) — imported via the per-target root
// (e.g. `wire_format_test_root.zig`) that widens the package boundary.
// The frame header is now SYNTHESIZED on the GPU; the only thing we
// pull out of the CPU module is the codec / level / block-size /
// sc-group-size constants the host bakes into push constants for the
// frame_assemble kernel.
const frame_format = @import("../src/format/frame_format.zig");
const block_header = @import("../src/format/block_header.zig");
const constants = @import("../src/format/streamlz_constants.zig");

// ── Wire-format constants (sourced from wire_constants.zig) ──────────

const LZ_BLOCK_SIZE: u32 = wire_constants.LZ_BLOCK_SIZE;
const INITIAL_LITERAL_COPY_BYTES: u32 = wire_constants.INITIAL_LITERAL_COPY_BYTES;
const SUBCHUNK_HDR_BYTES: u32 = wire_constants.SUBCHUNK_HDR_BYTES;
const CHUNK_INTERNAL_HDR_BYTES: u32 = wire_constants.CHUNK_INTERNAL_HDR_BYTES;
const UNCOMPRESSED_CHUNK_HDR_BYTES: u32 = wire_constants.UNCOMPRESSED_CHUNK_HDR_BYTES;
const OFF32_COUNT_PACK_MAX: u32 = wire_constants.OFF32_COUNT_PACK_MAX;
const SC_TAIL_PER_CHUNK_BYTES: u32 = wire_constants.SC_TAIL_PER_CHUNK_BYTES;

// ── Errors ────────────────────────────────────────────────────────────

pub const GpuWrapError = error{
    OutOfMemory,
    OutputTooSmall,
    BadHeader,
    UnsupportedTier,
    NoSpvForTier,
    TooManyChunks,
    MapMemoryFailed,
    BufferCreateFailed,
    MemoryAllocateFailed,
    MemoryTypeNotFound,
    BindBufferFailed,
} ||
    l1_codec.L1Error ||
    descriptors.DescriptorError ||
    dispatch.DispatchError;

// ── Push-constant layouts ────────────────────────────────────────────

const MeasurePush = extern struct {
    n_chunks: u32,
};

const LayoutPush = extern struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    prefix_size: u32,
};

const WritePush = extern struct {
    n_chunks: u32,
};

const FramePush = extern struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    prefix_size: u32,
    packed_hdrs: u32,
    codec_byte: u32,
    level_byte: u32,
    block_size_log2_offset: u32,
    sc_group_size_bits: u32,
    flags_byte: u32,
};

// ── Buffer helper (host-visible, mirror l1_codec's pattern) ──────────

const Buffer = struct {
    buf: vk.VkBuffer = null,
    mem: vk.VkDeviceMemory = null,
    mapped: ?[*]u8 = null,
    size: vk.VkDeviceSize = 0,
};

fn findMemoryType(
    pd: vk.VkPhysicalDevice,
    type_bits_mask: u32,
    required_flags: vk.VkMemoryPropertyFlags,
) ?u32 {
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return null;
    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(pd, &mem_props);
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (type_bits_mask & (@as(u32, 1) << @intCast(i))) != 0;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        if (supported and (flags & required_flags) == required_flags) return i;
    }
    return null;
}

fn createHostVisibleBuffer(ctx: *driver.Context, size: vk.VkDeviceSize) GpuWrapError!Buffer {
    l1_codec.ensureBufferFnSlots(ctx);
    const create_buf = vk.vkCreateBuffer_fn orelse return error.BufferCreateFailed;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse return error.BufferCreateFailed;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return error.MemoryAllocateFailed;
    const bind = vk.vkBindBufferMemory_fn orelse return error.BindBufferFailed;
    const map = vk.vkMapMemory_fn orelse return error.MapMemoryFailed;

    const bci: vk.VkBufferCreateInfo = .{
        .size = @max(size, 4),
        .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) return error.BufferCreateFailed;

    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    const ideal = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const fallback = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const mt_idx: u32 = blk: {
        if (findMemoryType(ctx.pd, req.memoryTypeBits, ideal)) |i| break :blk i;
        if (findMemoryType(ctx.pd, req.memoryTypeBits, fallback)) |i| break :blk i;
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MemoryTypeNotFound;
    };

    const mai: vk.VkMemoryAllocateInfo = .{
        .allocationSize = req.size,
        .memoryTypeIndex = mt_idx,
    };
    var mem: vk.VkDeviceMemory = null;
    if (alloc_mem(ctx.dev, &mai, null, &mem) != vk.VK_SUCCESS) {
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MemoryAllocateFailed;
    }
    if (bind(ctx.dev, buf, mem, 0) != vk.VK_SUCCESS) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.BindBufferFailed;
    }
    var raw: ?*anyopaque = null;
    if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
        return error.MapMemoryFailed;
    }
    return .{
        .buf = buf,
        .mem = mem,
        .mapped = @ptrCast(@alignCast(raw.?)),
        .size = req.size,
    };
}

fn destroyBuffer(ctx: *driver.Context, b: *Buffer) void {
    if (ctx.dev == null) return;
    if (b.mapped != null) {
        if (vk.vkUnmapMemory_fn) |u| u(ctx.dev, b.mem);
        b.mapped = null;
    }
    if (b.buf != null) {
        if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, b.buf, null);
        b.buf = null;
    }
    if (b.mem != null) {
        if (vk.vkFreeMemory_fn) |f| f(ctx.dev, b.mem, null);
        b.mem = null;
    }
}

// ── SPV lookup ───────────────────────────────────────────────────────

fn tierBlob(t: probe_mod.Tier) ?spv_blobs.Tier {
    return switch (t) {
        .tier1 => .tier1,
        .tier1_nv => .tier1_nv,
        .tier2 => .tier2,
        .unsupported => null,
    };
}

fn dupAlignedSpv(allocator: std.mem.Allocator, spv: []const u8) GpuWrapError![]align(4) u8 {
    const buf = allocator.alignedAlloc(u8, .@"4", spv.len) catch return error.OutOfMemory;
    @memcpy(buf, spv);
    return buf;
}

// ── Frame-header constants (mirror writeHeader's encoder-side config) ─
//
// The GPU `frame_assemble.comp` kernel synthesizes the SLZ1 frame +
// block header bytes directly into `frame_out_buf`. The encoder's
// option set is fixed at compile time — codec=fast, level=1,
// block_size=256 KiB, sc_group_size=0.25, content_size_present=true,
// no dictionary, no content_checksum — so the corresponding header
// fields are precomputed once here and passed to the kernel via push
// constants. Mirrors `writeHeader` in `src/format/frame_format.zig`.

const FRAME_CODEC_BYTE: u32 = @intFromEnum(frame_format.Codec.fast);
const FRAME_LEVEL_BYTE: u32 = 1;
const FRAME_BLOCK_SIZE_LOG2_OFFSET: u32 = blk: {
    const min_log2 = std.math.log2_int(usize, frame_format.min_block_size);
    const this_log2 = std.math.log2_int(usize, constants.chunk_size);
    break :blk this_log2 - min_log2;
};
const FRAME_FLAGS_BYTE: u32 = blk: {
    const f: frame_format.FrameFlags = .{ .content_size_present = true };
    break :blk @as(u32, @intCast(@as(u8, @bitCast(f))));
};

/// Fixed prefix size (frame header + 8-byte block header) for the
/// encoder's frozen option set. magic(4) + version(1) + flags(1) +
/// codec(1) + level(1) + block_size_log2(1) + sc_group_size_f32(4) +
/// reserved(1) + content_size_i64(8) + block_compressed_raw(4) +
/// block_decomp(4) = 30 bytes. Asserted at compile time below.
const PREFIX_SIZE: u32 = 14 + 8 + 8;

comptime {
    // Sanity-check against the CPU writer's bookkeeping — if anyone
    // ever flips a flag bit or adds an optional field, the SPV kernel's
    // inline header writer needs the matching change.
    std.debug.assert(PREFIX_SIZE == 30);
    std.debug.assert(FRAME_CODEC_BYTE == @intFromEnum(block_header.CodecType.fast));
}

// ── Frame-out / asm-out conservative upper bounds ───────────────────
//
// The host no longer participates in the per-chunk layout pass — the
// GPU is the source of truth for per-chunk offsets, asm_total, and
// total_frame_size. To size the buffers up-front (Vulkan needs the
// allocation size at vkCreateBuffer time) we compute conservative
// upper bounds the GPU kernel can never exceed.
//
// Per-chunk encoded ceiling:
//   uncompressed: UNCOMPRESSED_CHUNK_HDR_BYTES + chunk_size
//   compressed:   CHUNK_INTERNAL_HDR_BYTES + SUBCHUNK_HDR_BYTES + sps_max
//                  where sps_max == subChunkPayloadSize at every chunk
//                  capacity bound; using 2*chunk_size + 64 covers it
//                  with margin (2*chunk_size comes from the L1 codec's
//                  worst-case lit+cmd+off16+length budget; +64 covers
//                  every per-chunk header byte the wire format adds).
//
// asm_out per-chunk ceiling: the chunk's sub-chunk payload only (no
// per-chunk frame header overhead), bounded by sps_max above.

fn perChunkFrameBound(chunk_size: u32) u32 {
    const compressed_max: u32 = CHUNK_INTERNAL_HDR_BYTES + SUBCHUNK_HDR_BYTES + (2 * chunk_size) + 64;
    const uncompressed_max: u32 = UNCOMPRESSED_CHUNK_HDR_BYTES + chunk_size;
    return @max(compressed_max, uncompressed_max);
}

fn perChunkAsmBound(chunk_size: u32) u32 {
    return (2 * chunk_size) + 64;
}

// ── Public API ────────────────────────────────────────────────────────

/// Wrap `streams` (already device-resident from `l1_codec.encodeL1Multi`)
/// into a complete SLZ1 frame; write into `out_host`. The source bytes
/// are needed for the uncompressed-chunk fallback and the SC tail
/// prefix — pass them via `src_host`. Returns the byte count written.
///
/// Byte-identical to `wire_format.wrapL1ToSlz1` on every L1 corpus
/// (verified by `wire_format_test`).
pub fn wrapL1ToSlz1Gpu(
    ctx: *driver.Context,
    allocator: std.mem.Allocator,
    streams: l1_codec.L1Streams,
    src_host: []const u8,
    out_host: []u8,
) GpuWrapError!usize {
    const n_chunks = streams.n_chunks;
    if (n_chunks == 0 or n_chunks > l1_codec.MAX_CHUNKS) return error.TooManyChunks;

    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const tier_b = tierBlob(tier) orelse return error.UnsupportedTier;

    const original_size: u32 = @intCast(streams.dst_size);
    const chunk_size: u32 = l1_codec.CHUNK_SIZE;
    if (n_chunks > 1) {
        // Sanity: chunk_size * (n_chunks - 1) must be < original_size.
        if (@as(u64, chunk_size) * @as(u64, n_chunks - 1) >= original_size) {
            return error.BadHeader;
        }
    }

    // ── 1. Compute conservative upper bounds for the device buffers ─
    // The GPU layout kernel writes the actual sizes into frame_meta_buf;
    // the host reads them back after frame_assemble completes.
    const sc_tail_bytes: u32 = if (n_chunks > 1) (n_chunks - 1) * SC_TAIL_PER_CHUNK_BYTES else 0;
    const per_chunk_bound = perChunkFrameBound(chunk_size);
    const per_chunk_asm_bound = perChunkAsmBound(chunk_size);
    const total_frame_bound: u32 =
        PREFIX_SIZE + (n_chunks * per_chunk_bound) + sc_tail_bytes + 4;
    const asm_out_bound: u32 = n_chunks * per_chunk_asm_bound;

    if (out_host.len < PREFIX_SIZE) return error.OutputTooSmall;

    // ── 2. Allocate device buffers ───────────────────────────────────
    var measure_descs_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 8 * 4);
    defer destroyBuffer(ctx, &measure_descs_b);
    var measure_out_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 4);
    defer destroyBuffer(ctx, &measure_out_b);

    var write_descs_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 16 * 4);
    defer destroyBuffer(ctx, &write_descs_b);
    var asm_out_b = try createHostVisibleBuffer(ctx, @max(@as(vk.VkDeviceSize, asm_out_bound), 4));
    defer destroyBuffer(ctx, &asm_out_b);

    var frame_descs_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 4 * 4);
    defer destroyBuffer(ctx, &frame_descs_b);

    // 5 u32: total_frame_size, sc_tail_off, end_mark_off, asm_total,
    // block_payload_size — written by frame_layout, read by host (slot 0)
    // and by frame_assemble (slots 1, 2, 4).
    var frame_meta_b = try createHostVisibleBuffer(ctx, 5 * 4);
    defer destroyBuffer(ctx, &frame_meta_b);

    // Source buffer for the GPU side — uncompressed-fallback chunks
    // copy `chunk_decomp_size` raw source bytes via frame_assemble;
    // SC tail entries copy the first 8 source bytes of each non-first
    // chunk via the same kernel.
    const src_buf_size: vk.VkDeviceSize = @max(@as(vk.VkDeviceSize, src_host.len), 4);
    var src_b = try createHostVisibleBuffer(ctx, src_buf_size);
    defer destroyBuffer(ctx, &src_b);

    var frame_out_b = try createHostVisibleBuffer(ctx, @max(@as(vk.VkDeviceSize, total_frame_bound), 4));
    defer destroyBuffer(ctx, &frame_out_b);

    // ── 3. Upload src + initialise device buffers ────────────────────
    if (src_host.len > 0) {
        @memcpy(src_b.mapped.?[0..src_host.len], src_host);
    }
    @memset(measure_out_b.mapped.?[0..@intCast(measure_out_b.size)], 0);
    @memset(frame_meta_b.mapped.?[0..@intCast(frame_meta_b.size)], 0);
    @memset(asm_out_b.mapped.?[0..@intCast(asm_out_b.size)], 0);
    @memset(frame_out_b.mapped.?[0..@intCast(frame_out_b.size)], 0);

    // ── 4. Populate AssembleMeasureDesc (8 u32/chunk) ────────────────
    {
        const d: [*]u32 = @ptrCast(@alignCast(measure_descs_b.mapped.?));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 8;
            const init_copy = streams.per_chunk_initial_copy[ci];
            const chunk_dst_off: u64 = @as(u64, ci) * chunk_size;
            const decomp: u32 = blk: {
                const remaining = original_size - chunk_dst_off;
                if (remaining >= chunk_size) break :blk chunk_size;
                break :blk @intCast(remaining);
            };
            const c1 = streams.per_chunk_off32_count1[ci];
            const c2 = streams.per_chunk_off32_count2[ci];
            const lit_size = streams.per_chunk_lit_size[ci] - init_copy;
            d[base + 0] = lit_size;
            d[base + 1] = streams.per_chunk_cmd_size[ci];
            d[base + 2] = streams.per_chunk_off16_count[ci];
            d[base + 3] = streams.per_chunk_length_used[ci];
            d[base + 4] = (c1 + c2) * 3;
            d[base + 5] = c1;
            d[base + 6] = c2;
            const sub_decomp_gt_lzb: u32 = if (decomp > LZ_BLOCK_SIZE) 1 else 0;
            d[base + 7] = init_copy | (sub_decomp_gt_lzb << 16);
        }
    }

    // ── 5. Pre-fill AssembleWriteDesc slots 1..15 (slot 0 = layout's) ─
    {
        const d: [*]u32 = @ptrCast(@alignCast(write_descs_b.mapped.?));
        const cap_words: u32 = streams.chunk_capacity / 4;
        const off32_cap_words: u32 = streams.off32_capacity / 4;
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 16;
            const init_copy = streams.per_chunk_initial_copy[ci];
            const chunk_dst_off: u64 = @as(u64, ci) * chunk_size;
            const decomp: u32 = blk: {
                const remaining = original_size - chunk_dst_off;
                if (remaining >= chunk_size) break :blk chunk_size;
                break :blk @intCast(remaining);
            };
            const c1 = streams.per_chunk_off32_count1[ci];
            const c2 = streams.per_chunk_off32_count2[ci];

            // Slot 0 = asm_out_offset (or UNCOMPRESSED_CHUNK_MARKER).
            // Written by frame_layout.comp; pre-init to a poison value
            // (max u32 - 1) so a missed layout dispatch surfaces as a
            // visible OOB rather than silently appearing valid.
            d[base + 0]  = 0xFFFF_FFFE;
            d[base + 1]  = ci * cap_words;
            d[base + 2]  = streams.per_chunk_lit_size[ci];
            d[base + 3]  = ci * cap_words;
            d[base + 4]  = streams.per_chunk_cmd_size[ci];
            d[base + 5]  = ci * cap_words;
            d[base + 6]  = streams.per_chunk_off16_count[ci];
            d[base + 7]  = ci * cap_words;
            d[base + 8]  = streams.per_chunk_length_used[ci];
            d[base + 9]  = ci * off32_cap_words;
            d[base + 10] = (c1 + c2) * 3;
            d[base + 11] = c1;
            d[base + 12] = c2;
            d[base + 13] = 0; // src_byte_base — unused for L1 (init bytes come from lit_buf)
            const sub_decomp_gt_lzb: u32 = if (decomp > LZ_BLOCK_SIZE) 1 else 0;
            d[base + 14] = init_copy | (sub_decomp_gt_lzb << 16);
            d[base + 15] = 0;
        }
    }

    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    // ── 6. Measure dispatch ──────────────────────────────────────────
    {
        const spv_raw = spv_blobs.find("assemble_measure", tier_b) orelse return error.NoSpvForTier;
        const spv = try dupAlignedSpv(allocator, spv_raw);
        defer allocator.free(spv);
        const cached = try descriptors.getOrCreate(
            ctx, &cache, "assemble_measure", tier, spv, 2, @sizeOf(MeasurePush),
        );
        const bindings: [2]vk.VkDescriptorBufferInfo = .{
            .{ .buffer = measure_descs_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = measure_out_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const set = try descriptors.allocSet(ctx, cached, bindings[0..]);
        const push: MeasurePush = .{ .n_chunks = n_chunks };
        var push_bytes: [@sizeOf(MeasurePush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        _ = try dispatch.submitOne(
            ctx, cached.pipeline, cached.pipeline_layout, set, push_bytes[0..],
            .{ n_chunks, 1, 1 },
        );
    }

    // ── 7. Layout dispatch (GPU prefix-sum) ──────────────────────────
    // Consumes measure output, writes AssembleWriteDesc slot 0,
    // FrameChunkDesc (4 u32/chunk), FrameMeta (5 u32). After this
    // dispatch fences, frame_meta_b is the source of truth for
    // total_frame_size / sc_tail_off / end_mark_off / block_payload_size.
    {
        const spv_raw = spv_blobs.find("frame_layout", tier_b) orelse return error.NoSpvForTier;
        const spv = try dupAlignedSpv(allocator, spv_raw);
        defer allocator.free(spv);
        const cached = try descriptors.getOrCreate(
            ctx, &cache, "frame_layout", tier, spv, 4, @sizeOf(LayoutPush),
        );
        const bindings: [4]vk.VkDescriptorBufferInfo = .{
            .{ .buffer = measure_out_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = write_descs_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = frame_descs_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = frame_meta_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const set = try descriptors.allocSet(ctx, cached, bindings[0..]);
        const push: LayoutPush = .{
            .n_chunks = n_chunks,
            .eff_chunk_size = chunk_size,
            .src_len = @intCast(src_host.len),
            .prefix_size = PREFIX_SIZE,
        };
        var push_bytes: [@sizeOf(LayoutPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        _ = try dispatch.submitOne(
            ctx, cached.pipeline, cached.pipeline_layout, set, push_bytes[0..],
            .{ 1, 1, 1 },
        );
    }

    // ── 8. Write dispatch ────────────────────────────────────────────
    {
        const spv_raw = spv_blobs.find("assemble_write", tier_b) orelse return error.NoSpvForTier;
        const spv = try dupAlignedSpv(allocator, spv_raw);
        defer allocator.free(spv);
        const cached = try descriptors.getOrCreate(
            ctx, &cache, "assemble_write", tier, spv, 8, @sizeOf(WritePush),
        );
        const bindings: [8]vk.VkDescriptorBufferInfo = .{
            .{ .buffer = write_descs_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = streams.lit_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = streams.cmd_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = streams.off16_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = streams.length_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = streams.off32_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = src_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = asm_out_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const set = try descriptors.allocSet(ctx, cached, bindings[0..]);
        const push: WritePush = .{ .n_chunks = n_chunks };
        var push_bytes: [@sizeOf(WritePush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        _ = try dispatch.submitOne(
            ctx, cached.pipeline, cached.pipeline_layout, set, push_bytes[0..],
            .{ n_chunks, 1, 1 },
        );
    }

    // ── 9. Frame-assemble dispatch ───────────────────────────────────
    // Splices per-chunk payloads into frame_out AND synthesizes the
    // frame + block headers + SC tail prefix + end mark. The host no
    // longer uploads any prefix bytes — every byte in frame_out_buf is
    // written by this kernel.
    const internal_hdr0: u32 = 0x05 | 0x10 | 0x40; // magic | self_contained | restart_decoder
    const internal_hdr1: u32 = @intFromEnum(block_header.CodecType.fast);
    const sc_group_size_bits: u32 = @bitCast(wire_constants.SC_GROUP_SIZE);
    {
        const spv_raw = spv_blobs.find("frame_assemble", tier_b) orelse return error.NoSpvForTier;
        const spv = try dupAlignedSpv(allocator, spv_raw);
        defer allocator.free(spv);
        const cached = try descriptors.getOrCreate(
            ctx, &cache, "frame_assemble", tier, spv, 5, @sizeOf(FramePush),
        );
        const bindings: [5]vk.VkDescriptorBufferInfo = .{
            .{ .buffer = frame_descs_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = asm_out_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = src_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = frame_meta_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = frame_out_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const set = try descriptors.allocSet(ctx, cached, bindings[0..]);
        const packed_hdrs: u32 = internal_hdr0 | (internal_hdr1 << 8);
        const push: FramePush = .{
            .n_chunks = n_chunks,
            .eff_chunk_size = chunk_size,
            .src_len = @intCast(src_host.len),
            .prefix_size = PREFIX_SIZE,
            .packed_hdrs = packed_hdrs,
            .codec_byte = FRAME_CODEC_BYTE,
            .level_byte = FRAME_LEVEL_BYTE,
            .block_size_log2_offset = FRAME_BLOCK_SIZE_LOG2_OFFSET,
            .sc_group_size_bits = sc_group_size_bits,
            .flags_byte = FRAME_FLAGS_BYTE,
        };
        var push_bytes: [@sizeOf(FramePush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        _ = try dispatch.submitOne(
            ctx, cached.pipeline, cached.pipeline_layout, set, push_bytes[0..],
            .{ n_chunks + 1, 1, 1 },
        );
    }

    // ── 10. Read total_frame_size, stage frame_out → host ────────────
    const meta_ptr: [*]u32 = @ptrCast(@alignCast(frame_meta_b.mapped.?));
    const total_frame_size: u32 = meta_ptr[0];
    if (total_frame_size == 0 or total_frame_size > total_frame_bound) {
        // The GPU layout kernel never published a sane size — either
        // the dispatch never ran or a constant divergence corrupted the
        // running sum. Either way the frame bytes are unsafe to publish.
        return error.BadHeader;
    }
    if (out_host.len < total_frame_size) return error.OutputTooSmall;
    @memcpy(out_host[0..total_frame_size], frame_out_b.mapped.?[0..total_frame_size]);
    return total_frame_size;
}
