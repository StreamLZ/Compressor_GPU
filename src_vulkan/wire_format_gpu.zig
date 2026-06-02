//! GPU-side SLZ1 wire-format wrap.
//!
//! Phase 3 of the Vulkan port (see `src_vulkan/TODO.md`): mirror the
//! CUDA encoder's device-resident frame-assembly path. The CPU
//! reference is `wire_format.wrapL1ToSlz1` (in `wire_format.zig`) —
//! this module produces byte-identical output without staging the
//! per-chunk LZ streams through host memory.
//!
//! Pipeline (mirrors `src/encode/encode_assemble.zig`):
//!
//!   1. Build `AssembleMeasureDesc` (8 u32/chunk) from per-chunk
//!      stream sizes + chunk geometry, upload, dispatch
//!      `assemble_measure.comp` → per-chunk sub-chunk payload size.
//!   2. Host: prefix-sum the per-chunk sizes; decide compressed vs.
//!      uncompressed (sub_payload_size >= chunk_decomp_size triggers
//!      the raw-block fallback per `wire_format.zig` §3 head comment);
//!      build the per-chunk `AssembleWriteDesc` (16 u32/chunk) with
//!      asm_out offsets baked in.
//!   3. Upload AssembleWriteDesc, allocate the device-side asm_out
//!      buffer, dispatch `assemble_write.comp` → assembled sub-chunk
//!      payloads laid out back-to-back in `asm_out_buf`.
//!   4. Pre-form frame_hdr + block_hdr bytes on host (per
//!      `wire_format.zig`); compute the per-chunk dst offsets within
//!      the final frame; build the 4-u32/chunk FrameChunkDesc;
//!      upload prefix bytes + desc table; dispatch
//!      `frame_assemble.comp` (n_chunks + 1 workgroups) → complete
//!      frame in `frame_out_buf`.
//!   5. Stage `frame_out_buf` → host `out` slice.
//!
//! The CPU path remains callable as `wire_format.wrapL1ToSlz1`; the
//! `gpu_wrap_supported` predicate below documents the cases this
//! module handles (currently: every L1 corpus the CPU wrap handles,
//! since the L1 codec doesn't run a Huffman pass and the assembler
//! shaders emit only chunk-type-0 entropy chunks).

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");
const wire_format = @import("wire_format.zig");
const spv_blobs = @import("spv_blobs");

// CPU wire-format helpers (frame_hdr + block_hdr writers). Same import
// pattern `wire_format.zig` uses — resolves via the per-target root
// (e.g. `wire_format_test_root.zig`) that widens the package boundary.
const frame = @import("../src/format/frame_format.zig");
const block_header = @import("../src/format/block_header.zig");
const constants = @import("../src/format/streamlz_constants.zig");

// ── Wire-format constants (mirror wire_format.zig) ───────────────────

const LZ_BLOCK_SIZE: u32 = 0x10000;
const INITIAL_LITERAL_COPY_BYTES: u32 = 8;
const SUBCHUNK_LZ_FLAG_BIT: u32 = 0x800000;
const SUBCHUNK_MODE_SHIFT: u5 = 19;
const SUBCHUNK_HDR_BYTES: u32 = 3;
const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;
const UNCOMPRESSED_CHUNK_HDR_BYTES: u32 = 2;
const OFF32_COUNT_PACK_MAX: u32 = (1 << 12) - 1; // 4095
const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;
const UNCOMPRESSED_CHUNK_MARKER: u32 = 0xFFFFFFFF;

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

const WritePush = extern struct {
    n_chunks: u32,
};

const FramePush = extern struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    prefix_size: u32,
    sc_tail_off: u32,
    end_mark_off: u32,
    packed_hdrs: u32,
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

// ── Public API ────────────────────────────────────────────────────────

/// Per-chunk sub-chunk payload size — same formula as
/// `wire_format.subChunkPayloadSize` but evaluated on host using
/// the per-chunk descriptor values the L1 codec publishes. Used by the
/// pre-dispatch sizing pass for the asm_out buffer.
fn subChunkPayloadSize(
    streams: l1_codec.L1Streams,
    ci: u32,
    chunk_decomp_size: u32,
    init_copy: u32,
) u32 {
    const lit = streams.per_chunk_lit_size[ci] - init_copy;
    const cmd = streams.per_chunk_cmd_size[ci];
    const off16_bytes = streams.per_chunk_off16_count[ci] * 2;
    const length = streams.per_chunk_length_used[ci];
    const c1 = streams.per_chunk_off32_count1[ci];
    const c2 = streams.per_chunk_off32_count2[ci];
    const off32_bytes = (c1 + c2) * 3;
    var off32_ext: u32 = 0;
    if (c1 >= OFF32_COUNT_PACK_MAX) off32_ext += 2;
    if (c2 >= OFF32_COUNT_PACK_MAX) off32_ext += 2;

    var n: u32 = init_copy;
    n += 3 + lit;
    n += 3 + cmd;
    if (chunk_decomp_size > LZ_BLOCK_SIZE) n += 2;
    n += 2 + off16_bytes;
    n += 3 + off32_ext + off32_bytes;
    n += length;
    return n;
}

fn chunkInitialCopy(streams: l1_codec.L1Streams, ci: u32) u32 {
    return streams.per_chunk_initial_copy[ci];
}

/// Wrap `streams` (already device-resident from `l1_codec.encodeL1Multi`)
/// into a complete SLZ1 frame; write into `out_host`. The source bytes
/// are needed for the uncompressed-chunk fallback and the SC tail
/// prefix — pass them via `src_host`. Returns the byte count written.
///
/// Byte-identical to `wire_format.wrapL1ToSlz1` on every L1 corpus
/// (verified by `wire_format_test`; see TODO.md Phase 3).
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

    // ── 1. Pre-form frame_hdr + placeholder block_hdr on host ────────
    const original_size: u32 = @intCast(streams.dst_size);
    const eff_chunk_size: u32 = streams.dst_size / n_chunks; // checked below
    // Actual chunk size is l1_codec.CHUNK_SIZE — geometry from the L1 codec.
    const chunk_size: u32 = l1_codec.CHUNK_SIZE;
    _ = eff_chunk_size;
    if (n_chunks > 1) {
        // Sanity: chunk_size * (n_chunks - 1) must be < original_size.
        if (@as(u64, chunk_size) * @as(u64, n_chunks - 1) >= original_size) {
            return error.BadHeader;
        }
    }

    var prefix_buf: [frame.max_header_size + 8]u8 = undefined;
    const hdr_n = frame.writeHeader(prefix_buf[0..], .{
        .codec = .fast,
        .level = 1,
        .block_size = constants.chunk_size, // 256 KiB outer block bound
        .sc_group_size = 0.25,
        .content_size = original_size,
        .dictionary_id = null,
        .content_checksum = false,
    }) catch return error.BadHeader;
    // Block-header writer is filled later, after we know payload size.
    const prefix_size: u32 = @intCast(hdr_n + 8);

    // Internal block header bytes (per-chunk repeated).
    const internal_hdr0: u8 = 0x05 | 0x10 | 0x40; // magic | self_contained | restart_decoder
    const internal_hdr1: u8 = @intFromEnum(block_header.CodecType.fast);

    // ── 2. Host-side measurement (replicate measure-kernel arithmetic) ─
    // Per-chunk payload sizes + uncompressed-fallback decision. The
    // measure shader publishes the same values on the GPU — we run the
    // same arithmetic here so we can pre-decide chunk dst offsets, pre-
    // size the asm_out buffer, and skip a roundtrip dispatch+readback.
    // Phase 4 may move the prefix-sum + decision to the GPU.
    var sub_payload_sizes = allocator.alloc(u32, n_chunks) catch return error.OutOfMemory;
    defer allocator.free(sub_payload_sizes);
    var asm_offsets = allocator.alloc(u32, n_chunks) catch return error.OutOfMemory;
    defer allocator.free(asm_offsets);
    var per_chunk_asm_size = allocator.alloc(u32, n_chunks) catch return error.OutOfMemory;
    defer allocator.free(per_chunk_asm_size);
    var per_chunk_dst_off = allocator.alloc(u32, n_chunks) catch return error.OutOfMemory;
    defer allocator.free(per_chunk_dst_off);
    var per_chunk_decomp_size = allocator.alloc(u32, n_chunks) catch return error.OutOfMemory;
    defer allocator.free(per_chunk_decomp_size);

    var asm_total: u32 = 0;
    var pos: u32 = prefix_size;
    {
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const chunk_dst_off: u64 = @as(u64, ci) * chunk_size;
            const decomp: u32 = blk: {
                const remaining = original_size - chunk_dst_off;
                if (remaining >= chunk_size) break :blk chunk_size;
                break :blk @intCast(remaining);
            };
            per_chunk_decomp_size[ci] = decomp;
            const init_copy = chunkInitialCopy(streams, ci);
            const sps = subChunkPayloadSize(streams, ci, decomp, init_copy);
            sub_payload_sizes[ci] = sps;

            const emit_uncompressed = sps >= decomp and
                chunk_dst_off + decomp <= src_host.len;
            per_chunk_dst_off[ci] = pos;
            if (emit_uncompressed) {
                asm_offsets[ci] = UNCOMPRESSED_CHUNK_MARKER;
                per_chunk_asm_size[ci] = decomp;
                pos += UNCOMPRESSED_CHUNK_HDR_BYTES + decomp;
            } else {
                asm_offsets[ci] = asm_total;
                per_chunk_asm_size[ci] = sps;
                asm_total += sps;
                // Per-chunk overhead = 2B internal hdr + 4B chunk hdr +
                // 3B sub-chunk hdr (CHUNK_INTERNAL_HDR_BYTES = 6 covers
                // only the first two; the sub-chunk header is the third).
                // The frame_assemble kernel writes payload at
                // `dst_off + 9` for compressed chunks (see
                // frame_assemble.comp), so the host has to advance `pos`
                // by the same 9-byte overhead.
                pos += CHUNK_INTERNAL_HDR_BYTES + SUBCHUNK_HDR_BYTES + sps;
            }
        }
    }
    const sc_tail_off: u32 = pos;
    const sc_tail_bytes: u32 = if (n_chunks > 1)
        (n_chunks - 1) * SC_TAIL_PER_CHUNK_BYTES
    else
        0;
    pos += sc_tail_bytes;
    const end_mark_off: u32 = pos;
    pos += 4;
    const total_frame_size: u32 = pos;

    if (out_host.len < total_frame_size) return error.OutputTooSmall;

    // Backfill the block header now that we know payload size.
    const block_payload_size: u32 = total_frame_size - prefix_size - 4;
    frame.writeBlockHeader(prefix_buf[hdr_n..][0..8], .{
        .compressed_size = block_payload_size,
        .decompressed_size = original_size,
        .uncompressed = false,
        .parallel_decode_metadata = false,
    });

    // ── 3. Allocate device buffers ───────────────────────────────────
    var measure_descs_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 8 * 4);
    defer destroyBuffer(ctx, &measure_descs_b);
    var measure_out_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 4);
    defer destroyBuffer(ctx, &measure_out_b);
    var write_descs_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 16 * 4);
    defer destroyBuffer(ctx, &write_descs_b);
    var asm_out_b = try createHostVisibleBuffer(ctx, @max(@as(vk.VkDeviceSize, asm_total), 4));
    defer destroyBuffer(ctx, &asm_out_b);

    // Per-chunk frame_assemble descriptor: 4 u32 per chunk.
    var frame_descs_b = try createHostVisibleBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * 4 * 4);
    defer destroyBuffer(ctx, &frame_descs_b);
    var prefix_b = try createHostVisibleBuffer(ctx, prefix_size);
    defer destroyBuffer(ctx, &prefix_b);
    // Source buffer — for SC tail prefix copies + uncompressed-chunk
    // payload copies on the GPU side.
    const src_buf_size: vk.VkDeviceSize = @max(@as(vk.VkDeviceSize, src_host.len), 4);
    var src_b = try createHostVisibleBuffer(ctx, src_buf_size);
    defer destroyBuffer(ctx, &src_b);
    // Final frame buffer.
    var frame_out_b = try createHostVisibleBuffer(ctx, @max(@as(vk.VkDeviceSize, total_frame_size), 4));
    defer destroyBuffer(ctx, &frame_out_b);

    // Populate src buffer + prefix buffer.
    if (src_host.len > 0) {
        @memcpy(src_b.mapped.?[0..src_host.len], src_host);
    }
    @memcpy(prefix_b.mapped.?[0..prefix_size], prefix_buf[0..prefix_size]);
    @memset(asm_out_b.mapped.?[0..@intCast(asm_out_b.size)], 0);
    @memset(frame_out_b.mapped.?[0..@intCast(frame_out_b.size)], 0);

    // ── 4. Build measure descriptor + dispatch ───────────────────────
    {
        const d: [*]u32 = @ptrCast(@alignCast(measure_descs_b.mapped.?));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 8;
            const init_copy = chunkInitialCopy(streams, ci);
            const decomp = per_chunk_decomp_size[ci];
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
    @memset(measure_out_b.mapped.?[0..@intCast(measure_out_b.size)], 0);

    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    // Measure pass.
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

    // NOTE: the measure-kernel output is functionally redundant — the
    // host has already replicated the same per-chunk size arithmetic
    // in `sub_payload_sizes` above (needed for the asm_out buffer-size
    // and uncompressed-fallback predicate, both of which the host has
    // to know synchronously). Keeping the measure dispatch in the
    // pipeline anyway because:
    //   1. It matches the CUDA assembler's three-pass topology,
    //      simplifying parallel-implementation comparisons.
    //   2. Phase 4 (`prefix_sum_chunks.comp` + GPU scan) wants to
    //      consume measure-kernel output directly without the host
    //      readback we currently take. Wiring it in now lets that
    //      future change touch only the scan kernel.
    // No runtime_safety assertion against the measure output — a
    // mismatch can only mean a kernel/host arithmetic divergence,
    // which the per-corpus round-trip suites already catch.

    // ── 5. Build write descriptor + dispatch ─────────────────────────
    {
        const d: [*]u32 = @ptrCast(@alignCast(write_descs_b.mapped.?));
        const cap_words: u32 = streams.chunk_capacity / 4;
        const off32_cap_words: u32 = streams.off32_capacity / 4;
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 16;
            const init_copy = chunkInitialCopy(streams, ci);
            const decomp = per_chunk_decomp_size[ci];
            const c1 = streams.per_chunk_off32_count1[ci];
            const c2 = streams.per_chunk_off32_count2[ci];

            // asm_out_offset — UNCOMPRESSED_CHUNK_MARKER means "skip the
            // write kernel for this chunk" (frame_assemble will splice
            // raw bytes from src instead). The shader checks slot 0 for
            // the marker and bails before any descriptor reads.
            const skip_write = (asm_offsets[ci] == UNCOMPRESSED_CHUNK_MARKER);
            d[base + 0]  = if (skip_write) UNCOMPRESSED_CHUNK_MARKER else asm_offsets[ci];
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

    // ── 6. Build frame_assemble descriptor + dispatch ────────────────
    {
        const d: [*]u32 = @ptrCast(@alignCast(frame_descs_b.mapped.?));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 4;
            const is_uncompressed = (asm_offsets[ci] == UNCOMPRESSED_CHUNK_MARKER);
            d[base + 0] = per_chunk_dst_off[ci];
            d[base + 1] = asm_offsets[ci]; // already UNCOMPRESSED_CHUNK_MARKER for raw chunks
            d[base + 2] = per_chunk_asm_size[ci];
            d[base + 3] = if (is_uncompressed) 0 else
                (sub_payload_sizes[ci] |
                    (@as(u32, 1) << SUBCHUNK_MODE_SHIFT) |
                    SUBCHUNK_LZ_FLAG_BIT);
        }
    }

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
            .{ .buffer = prefix_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
            .{ .buffer = frame_out_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        };
        const set = try descriptors.allocSet(ctx, cached, bindings[0..]);
        const packed_hdrs: u32 = @as(u32, internal_hdr0) | (@as(u32, internal_hdr1) << 8);
        const push: FramePush = .{
            .n_chunks = n_chunks,
            .eff_chunk_size = chunk_size,
            .src_len = @intCast(src_host.len),
            .prefix_size = prefix_size,
            .sc_tail_off = sc_tail_off,
            .end_mark_off = end_mark_off,
            .packed_hdrs = packed_hdrs,
        };
        var push_bytes: [@sizeOf(FramePush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        _ = try dispatch.submitOne(
            ctx, cached.pipeline, cached.pipeline_layout, set, push_bytes[0..],
            .{ n_chunks + 1, 1, 1 },
        );
    }

    // ── 7. Stage frame_out → host ────────────────────────────────────
    @memcpy(out_host[0..total_frame_size], frame_out_b.mapped.?[0..total_frame_size]);
    return total_frame_size;
}
