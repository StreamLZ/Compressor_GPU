//! Phase-1 / phase-1.5 Vulkan L1 codec module.
//!
//! Production host-side glue for the GLSL L1 encoder + decoder shaders
//! in `src_vulkan/shaders/lz_{encode,decode}.comp`. Spec source:
//! `docs/vulkan_l1_codec_spec.md` §9 (host-side interface).
//!
//! Public API:
//!
//!   pub const L1Streams = struct {
//!       lit_buf, cmd_buf, off16_buf, length_buf  — device-side VkBuffers
//!       lit_size, cmd_size, off16_count, length_used — totals across all chunks
//!       n_chunks, chunk_capacity, dst_size       — multi-chunk geometry
//!       per_chunk_sizes_*                        — per-chunk size arrays
//!   };
//!
//!   pub const EncodeResult = struct { streams: L1Streams,
//!                                     comp_total_bytes: u32 };
//!
//!   pub fn encodeL1Sync(ctx, src_host) !EncodeResult
//!     Thin wrapper over encodeL1Multi for single-chunk callers.
//!
//!   pub fn encodeL1Multi(ctx, src_host) !EncodeResult
//!     Slices src into ceil(len/CHUNK_SIZE) chunks and dispatches one
//!     workgroup per chunk in a single vkCmdDispatch.
//!
//!   pub fn decodeL1Sync(ctx, streams, original_size, dst_host) !void
//!     Drives the per-chunk decoder dispatch and writes original_size
//!     bytes into dst_host.
//!
//!   pub fn freeStreams(ctx, streams: *L1Streams) void
//!
//! Implementation notes:
//!
//!   * Multi-chunk dispatch mirrors the CUDA path (src/encode/lz_kernel.cu
//!     :slzLzEncodeKernel). Each workgroup reads its `ChunkDesc` from a
//!     dedicated SSBO, computes its slice of every output stream via
//!     `chunk_id * chunk_capacity_words`, and writes per-chunk sizes
//!     back into a 4-u32-per-chunk sidecar.
//!   * Streams are over-provisioned at `2 * CHUNK_SIZE + 16` per chunk
//!     (spec §4.1 worst case). The total stream-buffer size is
//!     `n_chunks * chunk_capacity` — for the 35-chunk web.txt case
//!     (~4.5 MB src) the per-stream buffer is ~9 MB, which the host
//!     allocator handles without staging.
//!   * All buffers are HOST_VISIBLE + HOST_COHERENT + DEVICE_LOCAL where
//!     supported; falls back to HOST_VISIBLE only. The hash buffer is
//!     DEVICE_LOCAL only (sized to n_chunks * HASH_SIZE_BYTES; per-chunk
//!     slot to avoid cross-chunk reference contamination).

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");

// ── Constants ─────────────────────────────────────────────────────

/// Per-chunk hash-table size for L1 (hash_bits = 17 → 1<<17 entries × u32
/// = 512 KB). Spec §1.1.
pub const HASH_BITS: u32 = 17;
pub const HASH_SIZE_BYTES: vk.VkDeviceSize = (@as(vk.VkDeviceSize, 1) << HASH_BITS) * @sizeOf(u32);

/// Per-chunk input slice size. Matches `src/format/streamlz_constants.zig`
/// sub_chunk_size = 0x20000 (128 KiB). Each shader workgroup processes
/// one chunk; the encoder still uses the 2-block scanBlock interior
/// (64 KiB blocks), so a 128 KiB chunk = 2 blocks per workgroup.
pub const CHUNK_SIZE: u32 = 0x20000;

/// Per-chunk reservation in each output stream. Worst-case stream size
/// per spec §4.1 is `2 * src_size`; we add 16 bytes of slack so lane-0
/// RMW byte stores into the tail u32 word never write past the slice.
/// Must be a multiple of 4 (shader byte-load helpers assume word-aligned
/// per-chunk bases — see lz_{encode,decode}.comp head-of-file comment).
pub const CHUNK_STREAM_CAPACITY: u32 = (CHUNK_SIZE * 2) + 16;

/// Encoder push constants (12 bytes — see lz_encode.comp).
const EncodePush = extern struct {
    n_chunks: u32,
    hash_bits: u32,
    chunk_capacity: u32,
};

/// Decoder push constants (4 bytes — see lz_decode.comp).
const DecodePush = extern struct {
    n_chunks: u32,
};

// ── Errors ────────────────────────────────────────────────────────

pub const L1Error = error{
    LoaderNotReady,
    NoSpvForTier,
    UnsupportedTier,
    BufferCreateFailed,
    MemoryAllocateFailed,
    MemoryTypeNotFound,
    BindBufferFailed,
    MapMemoryFailed,
    BeginRecordFailed,
    EndRecordFailed,
    SubmitFailed,
    FenceWaitFailed,
    TooManyChunks,
} ||
    descriptors.DescriptorError ||
    dispatch.DispatchError;

/// Hard cap on chunks per encode. Bounds the per-chunk size sidecar
/// arrays so we never need heap allocation in the codec module. 256
/// chunks × 128 KiB = 32 MiB max input — comfortably above the 4.5 MiB
/// web.txt smoke test and the 4 MiB enwik8 prefix test.
pub const MAX_CHUNKS: u32 = 256;

// ── Public types ──────────────────────────────────────────────────

/// One device-side L1 stream bundle. All four buffers are owned —
/// callers must `freeStreams` exactly once. The size fields sum across
/// every chunk; per-chunk slices into the streams are reconstructed via
/// `chunk_capacity` + the per-chunk count arrays.
pub const L1Streams = struct {
    lit_buf: vk.VkBuffer = null,
    lit_mem: vk.VkDeviceMemory = null,
    lit_size: u32 = 0,

    cmd_buf: vk.VkBuffer = null,
    cmd_mem: vk.VkDeviceMemory = null,
    cmd_size: u32 = 0,

    off16_buf: vk.VkBuffer = null,
    off16_mem: vk.VkDeviceMemory = null,
    off16_count: u32 = 0,

    length_buf: vk.VkBuffer = null,
    length_mem: vk.VkDeviceMemory = null,
    length_used: u32 = 0,

    /// Multi-chunk geometry.
    n_chunks: u32 = 0,
    chunk_capacity: u32 = CHUNK_STREAM_CAPACITY,
    /// Original input length in bytes. Stored on the bundle so the
    /// decoder can rebuild per-chunk dst slices without the caller
    /// re-asserting it.
    dst_size: u32 = 0,

    /// Per-chunk sizes (filled in by the encoder's Sizes sidecar).
    /// Indices [0..n_chunks) are valid.
    per_chunk_lit_size: [MAX_CHUNKS]u32 = @splat(0),
    per_chunk_cmd_size: [MAX_CHUNKS]u32 = @splat(0),
    per_chunk_off16_count: [MAX_CHUNKS]u32 = @splat(0),
    per_chunk_length_used: [MAX_CHUNKS]u32 = @splat(0),

    /// Total bytes worth of compressed payload (lit + cmd + 2*off16 + length)
    /// summed across all chunks. Convenience for the round-trip test's
    /// compression-ratio report.
    pub fn compressedTotalBytes(self: L1Streams) u32 {
        return self.lit_size + self.cmd_size + (self.off16_count * 2) + self.length_used;
    }
};

pub const EncodeResult = struct {
    streams: L1Streams,
    /// Sum of the four stream sizes across every chunk. Equal to
    /// `streams.compressedTotalBytes()`.
    comp_total_bytes: u32,
};

// ── Internal: low-level buffer helpers ────────────────────────────

fn resolveDeviceFn(comptime T: type, dev: vk.VkDevice, name: [*:0]const u8) ?T {
    if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
        if (gdpa(dev, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    if (vk.vkGetInstanceProcAddr_fn) |gipa| {
        const inst = driver.g_default.inst;
        if (gipa(inst, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    return null;
}

fn ensureBufferFnSlots(ctx: *driver.Context) void {
    if (vk.vkCreateBuffer_fn == null)
        vk.vkCreateBuffer_fn = resolveDeviceFn(vk.FnCreateBuffer, ctx.dev, "vkCreateBuffer");
    if (vk.vkDestroyBuffer_fn == null)
        vk.vkDestroyBuffer_fn = resolveDeviceFn(vk.FnDestroyBuffer, ctx.dev, "vkDestroyBuffer");
    if (vk.vkAllocateMemory_fn == null)
        vk.vkAllocateMemory_fn = resolveDeviceFn(vk.FnAllocateMemory, ctx.dev, "vkAllocateMemory");
    if (vk.vkFreeMemory_fn == null)
        vk.vkFreeMemory_fn = resolveDeviceFn(vk.FnFreeMemory, ctx.dev, "vkFreeMemory");
    if (vk.vkBindBufferMemory_fn == null)
        vk.vkBindBufferMemory_fn = resolveDeviceFn(vk.FnBindBufferMemory, ctx.dev, "vkBindBufferMemory");
    if (vk.vkMapMemory_fn == null)
        vk.vkMapMemory_fn = resolveDeviceFn(vk.FnMapMemory, ctx.dev, "vkMapMemory");
    if (vk.vkUnmapMemory_fn == null)
        vk.vkUnmapMemory_fn = resolveDeviceFn(vk.FnUnmapMemory, ctx.dev, "vkUnmapMemory");
    if (vk.vkGetBufferMemoryRequirements_fn == null)
        vk.vkGetBufferMemoryRequirements_fn = resolveDeviceFn(vk.FnGetBufferMemoryRequirements, ctx.dev, "vkGetBufferMemoryRequirements");

    // vkGetPhysicalDeviceMemoryProperties is instance-level — resolve via
    // vkGetInstanceProcAddr only.
    if (vk.vkGetPhysicalDeviceMemoryProperties_fn == null) {
        if (vk.vkGetInstanceProcAddr_fn) |gipa| {
            if (gipa(driver.g_default.inst, "vkGetPhysicalDeviceMemoryProperties")) |raw| {
                vk.vkGetPhysicalDeviceMemoryProperties_fn = @ptrCast(@alignCast(raw));
            }
        }
    }
}

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
        const has_required = (flags & required_flags) == required_flags;
        if (supported and has_required) return i;
    }
    return null;
}

const Buffer = struct {
    buf: vk.VkBuffer = null,
    mem: vk.VkDeviceMemory = null,
    mapped: ?[*]u8 = null,
    size: vk.VkDeviceSize = 0,
};

fn createBuffer(
    ctx: *driver.Context,
    size: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
    host_visible: bool,
) L1Error!Buffer {
    ensureBufferFnSlots(ctx);
    const create_buf = vk.vkCreateBuffer_fn orelse return error.BufferCreateFailed;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse return error.BufferCreateFailed;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return error.MemoryAllocateFailed;
    const bind = vk.vkBindBufferMemory_fn orelse return error.BindBufferFailed;

    const bci: vk.VkBufferCreateInfo = .{
        .size = size,
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) {
        return error.BufferCreateFailed;
    }

    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    const mt_idx: u32 = blk: {
        if (host_visible) {
            const ideal = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            const fallback = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (findMemoryType(ctx.pd, req.memoryTypeBits, ideal)) |i| break :blk i;
            if (findMemoryType(ctx.pd, req.memoryTypeBits, fallback)) |i| break :blk i;
        } else {
            if (findMemoryType(ctx.pd, req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) |i| break :blk i;
        }
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

    var mapped: ?[*]u8 = null;
    if (host_visible) {
        const map = vk.vkMapMemory_fn orelse {
            if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
            if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
            return error.MapMemoryFailed;
        };
        var raw: ?*anyopaque = null;
        if (map(ctx.dev, mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) != vk.VK_SUCCESS) {
            if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
            if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
            return error.MapMemoryFailed;
        }
        mapped = @ptrCast(@alignCast(raw orelse {
            if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);
            if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);
            return error.MapMemoryFailed;
        }));
    }

    return .{ .buf = buf, .mem = mem, .mapped = mapped, .size = size };
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

// ── Helpers shared by encode + decode ─────────────────────────────

fn pickTier(ctx: *driver.Context) L1Error!probe_mod.Tier {
    const pr = probe_mod.probe(ctx.inst, ctx.pd);
    return switch (pr.tier) {
        .tier1, .tier1_nv, .tier2 => pr.tier,
        .unsupported => error.UnsupportedTier,
    };
}

/// Number of chunks for an input of `src_size` bytes. Equivalent to
/// `ceil(src_size / CHUNK_SIZE)` with a hard minimum of 1.
fn computeNChunks(src_size: u32) u32 {
    if (src_size == 0) return 1;
    return (src_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
}

// ── Public API: encode ────────────────────────────────────────────

/// Thin wrapper retained for single-chunk callers. Identical to
/// `encodeL1Multi`; the previous phase-1 single-chunk restriction is
/// gone now that the shader respects gl_WorkGroupID.x.
pub fn encodeL1Sync(
    ctx: *driver.Context,
    src_host: []const u8,
    encode_spv: []const u8,
) L1Error!EncodeResult {
    return encodeL1Multi(ctx, src_host, encode_spv);
}

/// Encode `src_host` through the Vulkan L1 encoder, slicing into
/// `ceil(src_host.len / CHUNK_SIZE)` independent chunks. Dispatches a
/// single workgroup-per-chunk kernel and reads per-chunk stream sizes
/// back via the Sizes sidecar.
pub fn encodeL1Multi(
    ctx: *driver.Context,
    src_host: []const u8,
    encode_spv: []const u8,
) L1Error!EncodeResult {
    const src_size: u32 = @intCast(src_host.len);
    const n_chunks = computeNChunks(src_size);
    if (n_chunks > MAX_CHUNKS) return error.TooManyChunks;

    const tier = try pickTier(ctx);
    const spv = encode_spv;

    // Per-chunk slice reservation, in bytes. Total stream buffer size =
    // n_chunks * chunk_capacity. CHUNK_STREAM_CAPACITY is already a
    // multiple of 4 (264208 = 0x40810), keeping per-chunk word bases
    // u32-aligned.
    const chunk_capacity: u32 = CHUNK_STREAM_CAPACITY;
    const stream_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * chunk_capacity;

    // ── Allocations ──────────────────────────────────────────────
    var src_b = try createBuffer(
        ctx,
        @max(@as(vk.VkDeviceSize, 4), @as(vk.VkDeviceSize, src_size)),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        true,
    );
    errdefer destroyBuffer(ctx, &src_b);

    var lit_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &lit_b);
    var cmd_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &cmd_b);
    var off16_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &off16_b);
    var length_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &length_b);

    // Hash buffer: n_chunks × HASH_SIZE_BYTES. Device-local only — the
    // host never reads or writes it.
    var hash_b = try createBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * HASH_SIZE_BYTES, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    defer destroyBuffer(ctx, &hash_b);

    // Sizes sidecar: 4 u32 per chunk.
    var sizes_b = try createBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 4,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &sizes_b);

    // ChunkDescs: 3 u32 per chunk (src_offset, src_size, reserved).
    var chunks_b = try createBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 3,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &chunks_b);

    // ── Upload src + chunk descriptors ───────────────────────────
    if (src_size > 0) {
        const src_mapped = src_b.mapped orelse return error.MapMemoryFailed;
        @memcpy(src_mapped[0..src_size], src_host);
    }
    @memset(lit_b.mapped.?[0..@intCast(lit_b.size)], 0);
    @memset(cmd_b.mapped.?[0..@intCast(cmd_b.size)], 0);
    @memset(off16_b.mapped.?[0..@intCast(off16_b.size)], 0);
    @memset(length_b.mapped.?[0..@intCast(length_b.size)], 0);
    @memset(sizes_b.mapped.?[0..@intCast(sizes_b.size)], 0);

    // Fill chunk descriptors: each chunk gets [src_offset, src_size, 0].
    {
        const chunks_words: [*]u32 = @ptrCast(@alignCast(chunks_b.mapped.?));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const off = ci * CHUNK_SIZE;
            const this_size: u32 = if (off + CHUNK_SIZE <= src_size)
                CHUNK_SIZE
            else if (off < src_size)
                src_size - off
            else
                0;
            chunks_words[ci * 3 + 0] = off;
            chunks_words[ci * 3 + 1] = this_size;
            chunks_words[ci * 3 + 2] = 0;
        }
    }

    // ── Build pipeline + descriptor set ──────────────────────────
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_enc = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_encode",
        tier,
        spv,
        8, // bindings 0..7: src, lit, cmd, off16, length, hash, sizes, chunks
        @sizeOf(EncodePush),
    );

    const enc_bindings: [8]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = src_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = lit_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = cmd_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off16_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = length_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = hash_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = sizes_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const enc_set = try descriptors.allocSet(ctx, cached_enc, enc_bindings[0..]);

    // ── Submit encode dispatch (n_chunks workgroups) ─────────────
    const enc_push: EncodePush = .{
        .n_chunks = n_chunks,
        .hash_bits = HASH_BITS,
        .chunk_capacity = chunk_capacity,
    };
    var enc_push_bytes: [@sizeOf(EncodePush)]u8 = undefined;
    @memcpy(enc_push_bytes[0..], std.mem.asBytes(&enc_push));

    _ = try dispatch.submitOne(
        ctx,
        cached_enc.pipeline,
        cached_enc.pipeline_layout,
        enc_set,
        enc_push_bytes[0..],
        .{ n_chunks, 1, 1 },
    );

    // ── Read back per-chunk sizes ────────────────────────────────
    const sizes_words: [*]const u32 = @ptrCast(@alignCast(sizes_b.mapped.?));
    var streams: L1Streams = .{
        .n_chunks = n_chunks,
        .chunk_capacity = chunk_capacity,
        .dst_size = src_size,
    };
    var lit_total: u32 = 0;
    var cmd_total: u32 = 0;
    var off16_total: u32 = 0;
    var length_total: u32 = 0;
    {
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 4;
            const ls = sizes_words[base + 0];
            const cs = sizes_words[base + 1];
            const os = sizes_words[base + 2];
            const xs = sizes_words[base + 3];
            streams.per_chunk_lit_size[ci] = ls;
            streams.per_chunk_cmd_size[ci] = cs;
            streams.per_chunk_off16_count[ci] = os;
            streams.per_chunk_length_used[ci] = xs;
            lit_total += ls;
            cmd_total += cs;
            off16_total += os;
            length_total += xs;
        }
    }

    // Drop the src buffer — encode is done.
    destroyBuffer(ctx, &src_b);

    streams.lit_buf = lit_b.buf;
    streams.lit_mem = lit_b.mem;
    streams.lit_size = lit_total;
    streams.cmd_buf = cmd_b.buf;
    streams.cmd_mem = cmd_b.mem;
    streams.cmd_size = cmd_total;
    streams.off16_buf = off16_b.buf;
    streams.off16_mem = off16_b.mem;
    streams.off16_count = off16_total;
    streams.length_buf = length_b.buf;
    streams.length_mem = length_b.mem;
    streams.length_used = length_total;

    // Mapped pointers are forgotten by L1Streams (the buffer handles are
    // enough for the decoder bind + host readback). Unmap and zero the
    // local Buffer handles so the errdefer chain becomes a no-op on
    // the success return.
    if (vk.vkUnmapMemory_fn) |u| {
        u(ctx.dev, lit_b.mem);
        u(ctx.dev, cmd_b.mem);
        u(ctx.dev, off16_b.mem);
        u(ctx.dev, length_b.mem);
    }
    lit_b.buf = null;
    lit_b.mem = null;
    lit_b.mapped = null;
    cmd_b.buf = null;
    cmd_b.mem = null;
    cmd_b.mapped = null;
    off16_b.buf = null;
    off16_b.mem = null;
    off16_b.mapped = null;
    length_b.buf = null;
    length_b.mem = null;
    length_b.mapped = null;

    return .{
        .streams = streams,
        .comp_total_bytes = streams.compressedTotalBytes(),
    };
}

// ── Public API: decode ────────────────────────────────────────────

/// Run the Vulkan L1 decoder on `streams` and copy the decoded bytes
/// into `dst_host`. `dst_host.len` must equal `original_size`.
pub fn decodeL1Sync(
    ctx: *driver.Context,
    streams: L1Streams,
    original_size: usize,
    dst_host: []u8,
    decode_spv: []const u8,
) L1Error!void {
    std.debug.assert(dst_host.len == original_size);
    ensureBufferFnSlots(ctx);

    const dst_size: u32 = @intCast(original_size);
    const n_chunks = streams.n_chunks;
    if (n_chunks == 0 or n_chunks > MAX_CHUNKS) return error.TooManyChunks;
    const chunk_capacity = streams.chunk_capacity;

    const tier = try pickTier(ctx);
    const spv = decode_spv;

    // Dst buffer: rounded up to 4 bytes for the u32-packed RMW pattern.
    // Each chunk's slice is at byte offset `chunk_id * CHUNK_SIZE` and is
    // sized by the per-chunk dst_size descriptor entry.
    const dst_buf_size: vk.VkDeviceSize = @max(@as(vk.VkDeviceSize, 4), (@as(vk.VkDeviceSize, dst_size) + 3) & ~@as(vk.VkDeviceSize, 3));
    var dst_b = try createBuffer(
        ctx,
        dst_buf_size,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &dst_b);
    @memset(dst_b.mapped.?[0..@intCast(dst_b.size)], 0);

    // ChunkDescs: 12 u32 per chunk. See lz_decode.comp head comment for
    // the slot layout. The host writes word bases (in u32 words); the
    // shader multiplies by 4 to get byte bases for the per-byte load
    // helpers.
    var chunks_b = try createBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 12,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &chunks_b);
    @memset(chunks_b.mapped.?[0..@intCast(chunks_b.size)], 0);
    {
        const chunks_words: [*]u32 = @ptrCast(@alignCast(chunks_b.mapped.?));
        const cap_words: u32 = chunk_capacity / 4;
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const dst_off: u32 = ci * CHUNK_SIZE;
            const this_dst_size: u32 = if (dst_off + CHUNK_SIZE <= dst_size)
                CHUNK_SIZE
            else if (dst_off < dst_size)
                dst_size - dst_off
            else
                0;
            const stream_word_base: u32 = ci * cap_words;
            const base = ci * 12;
            chunks_words[base + 0] = dst_off;
            chunks_words[base + 1] = this_dst_size;
            chunks_words[base + 2] = stream_word_base; // cmd
            chunks_words[base + 3] = streams.per_chunk_cmd_size[ci];
            chunks_words[base + 4] = stream_word_base; // lit
            chunks_words[base + 5] = streams.per_chunk_lit_size[ci];
            chunks_words[base + 6] = stream_word_base; // off16 (byte addr = base*4)
            chunks_words[base + 7] = streams.per_chunk_off16_count[ci];
            chunks_words[base + 8] = stream_word_base; // length
            chunks_words[base + 9] = streams.per_chunk_length_used[ci];
            // slots 10..11 reserved.
        }
    }

    // ── Build decode pipeline + descriptor set ───────────────────
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_dec = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_decode",
        tier,
        spv,
        6, // bindings 0..5: cmd, lit, off16, length, dst, chunks
        @sizeOf(DecodePush),
    );

    const dec_bindings: [6]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = streams.cmd_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.lit_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.off16_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.length_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    // ── Submit decode dispatch (n_chunks workgroups) ─────────────
    const dec_push: DecodePush = .{
        .n_chunks = n_chunks,
    };
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

    // Copy dst out via the host map.
    if (dst_size > 0) {
        const dst_mapped = dst_b.mapped orelse return error.MapMemoryFailed;
        @memcpy(dst_host[0..dst_size], dst_mapped[0..dst_size]);
    }
}

// ── Debug helper: read back stream bytes ─────────────────────────
// Single-chunk debug aid — maps each stream buffer and copies up to
// `max_bytes` of CHUNK 0's slice into the caller-supplied buffers.
// Useful for the per-128-byte regression cases in l1_codec_test.zig.
pub fn debugReadStreams(
    ctx: *driver.Context,
    streams: L1Streams,
    out_cmd: []u8,
    out_lit: []u8,
    out_off16: []u8,
    out_length: []u8,
) struct { cmd_n: usize, lit_n: usize, off16_n: usize, length_n: usize } {
    ensureBufferFnSlots(ctx);
    const map = vk.vkMapMemory_fn.?;
    const unmap = vk.vkUnmapMemory_fn.?;

    var raw: ?*anyopaque = null;
    var cmd_n: usize = 0;
    var lit_n: usize = 0;
    var off16_n: usize = 0;
    var length_n: usize = 0;

    // Chunk 0's slice starts at byte 0; per-chunk_0 sizes are the totals
    // when n_chunks == 1, which is the only case this helper is exercised.
    const ch0_cmd = if (streams.n_chunks > 0) streams.per_chunk_cmd_size[0] else 0;
    const ch0_lit = if (streams.n_chunks > 0) streams.per_chunk_lit_size[0] else 0;
    const ch0_off16 = if (streams.n_chunks > 0) streams.per_chunk_off16_count[0] else 0;
    const ch0_length = if (streams.n_chunks > 0) streams.per_chunk_length_used[0] else 0;

    if (map(ctx.dev, streams.cmd_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        cmd_n = @min(out_cmd.len, ch0_cmd);
        @memcpy(out_cmd[0..cmd_n], p[0..cmd_n]);
        unmap(ctx.dev, streams.cmd_mem);
    }
    if (map(ctx.dev, streams.lit_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        lit_n = @min(out_lit.len, ch0_lit);
        @memcpy(out_lit[0..lit_n], p[0..lit_n]);
        unmap(ctx.dev, streams.lit_mem);
    }
    if (map(ctx.dev, streams.off16_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        off16_n = @min(out_off16.len, ch0_off16 * 2);
        @memcpy(out_off16[0..off16_n], p[0..off16_n]);
        unmap(ctx.dev, streams.off16_mem);
    }
    if (map(ctx.dev, streams.length_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        length_n = @min(out_length.len, ch0_length);
        @memcpy(out_length[0..length_n], p[0..length_n]);
        unmap(ctx.dev, streams.length_mem);
    }
    return .{ .cmd_n = cmd_n, .lit_n = lit_n, .off16_n = off16_n, .length_n = length_n };
}

// ── Public API: cleanup ───────────────────────────────────────────

/// Destroy every device-side handle in `streams` and zero the struct.
/// Idempotent — repeat calls are no-ops because each field reads as null
/// after the first call.
pub fn freeStreams(ctx: *driver.Context, streams: *L1Streams) void {
    if (ctx.dev == null) return;

    inline for (.{
        .{ &streams.lit_buf, &streams.lit_mem },
        .{ &streams.cmd_buf, &streams.cmd_mem },
        .{ &streams.off16_buf, &streams.off16_mem },
        .{ &streams.length_buf, &streams.length_mem },
    }) |pair| {
        if (pair[0].* != null) {
            if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, pair[0].*, null);
            pair[0].* = null;
        }
        if (pair[1].* != null) {
            if (vk.vkFreeMemory_fn) |f| f(ctx.dev, pair[1].*, null);
            pair[1].* = null;
        }
    }
    streams.lit_size = 0;
    streams.cmd_size = 0;
    streams.off16_count = 0;
    streams.length_used = 0;
    streams.n_chunks = 0;
    streams.dst_size = 0;
}
