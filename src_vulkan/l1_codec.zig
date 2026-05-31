//! Phase-1 Vulkan L1 codec module.
//!
//! Production host-side glue for the GLSL L1 encoder + decoder shaders
//! in `src_vulkan/shaders/lz_{encode,decode}.comp`. Spec source:
//! `docs/vulkan_l1_codec_spec.md` §9 (host-side interface).
//!
//! Public API (locked by the spec):
//!
//!   pub const L1Streams = struct {
//!       lit_buf, cmd_buf, off16_buf, length_buf  — device-side VkBuffers
//!       lit_size, cmd_size, off16_count, length_used — bytes/entries used
//!   };
//!
//!   pub const EncodeResult = struct { streams: L1Streams,
//!                                     comp_total_bytes: u32 };
//!
//!   pub fn encodeL1Sync(ctx, src_host) !EncodeResult
//!   pub fn decodeL1Sync(ctx, streams, original_size, dst_host) !void
//!   pub fn freeStreams(ctx, streams: *L1Streams) void
//!
//! Implementation notes:
//!
//!   * Phase 1 = single chunk dispatch (group_count = .{1,1,1}). Multi-
//!     chunk extension is phase 2.
//!   * Streams are over-provisioned at `2 * src_size` per the spec §4.1.
//!     Phase 2 will tighten this once header wrapping is wired.
//!   * All buffers are HOST_VISIBLE + HOST_COHERENT + DEVICE_LOCAL (the
//!     "ideal" memory type that desktop NVIDIA and AMD always expose).
//!     Falls back to HOST_VISIBLE + HOST_COHERENT only. This avoids the
//!     staging-buffer + vkCmdCopyBuffer round-trip the spec mentions in
//!     §9 "Internals" — for the phase-1 single-shot case the simpler
//!     direct-map path is plenty fast and removes a chunk of host glue.
//!   * The descriptors.Cache is owned by the codec (one per `init` call).
//!     Pipelines are keyed on (kernel, tier); the encoder + decoder share
//!     one cache so a single dispatch sequence pays the build cost twice
//!     and the second sequence pays zero.
//!   * `freeStreams` destroys every device-side handle and zeroes the
//!     struct. Calling encodeL1Sync without a prior freeStreams leaks the
//!     buffers from the previous result — by design, since the test
//!     driver consumes a single EncodeResult per encode call.

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

/// Encoder push constants (8 bytes — see lz_encode.comp).
const EncodePush = extern struct {
    src_size: u32,
    hash_bits: u32,
};

/// Decoder push constants (20 bytes — see lz_decode.comp).
const DecodePush = extern struct {
    cmd_size: u32,
    lit_size: u32,
    off16_count: u32,
    length_remaining: u32,
    dst_size: u32,
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
} ||
    descriptors.DescriptorError ||
    dispatch.DispatchError;

// ── Public types ──────────────────────────────────────────────────

/// One device-side L1 stream bundle. All four buffers are owned —
/// callers must `freeStreams` exactly once. `lit_size`/`cmd_size` are
/// in bytes; `off16_count` is u16 entries (each = 2 bytes); `length_used`
/// is in bytes.
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

    /// Total bytes worth of compressed payload (lit + cmd + 2*off16 + length).
    /// Convenience for the round-trip test's compression-ratio report.
    pub fn compressedTotalBytes(self: L1Streams) u32 {
        return self.lit_size + self.cmd_size + (self.off16_count * 2) + self.length_used;
    }
};

pub const EncodeResult = struct {
    streams: L1Streams,
    /// Sum of the four stream sizes. Equal to `streams.compressedTotalBytes()`.
    comp_total_bytes: u32,
};

// ── Internal: low-level buffer helpers ────────────────────────────
// These mirror dispatch_test.zig's helpers but live here so the codec
// has no test-module dependency. The longer-term plan (spec §9 and
// the phase-2 buffers.zig refactor) lifts this into a shared module.

/// Resolve every device-level fn the codec needs. Idempotent — each
/// `orelse resolve` short-circuits once the slot is non-null. Called
/// from `encodeL1Sync` and `decodeL1Sync` on first use; cheap to call
/// repeatedly.
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

/// Find a memory type whose bit is in `type_bits_mask` AND whose
/// propertyFlags include every bit in `required_flags`. Returns null
/// when no matching type exists (caller can retry with weaker flags).
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

/// A device-side VkBuffer + its backing VkDeviceMemory + an optional
/// host-mapped pointer for HOST_VISIBLE bindings. `size` is the
/// caller-requested logical size; the driver's `req.size` may be
/// padded above this for alignment, but we never expose the pad to
/// callers.
const Buffer = struct {
    buf: vk.VkBuffer = null,
    mem: vk.VkDeviceMemory = null,
    mapped: ?[*]u8 = null,
    size: vk.VkDeviceSize = 0,
};

/// Create a buffer + backing memory. `host_visible = true` selects a
/// HOST_VISIBLE + HOST_COHERENT memory type (prefers also DEVICE_LOCAL),
/// `false` selects DEVICE_LOCAL only (the hash table — no host I/O).
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

    // 1. Create the VkBuffer with the requested usage.
    const bci: vk.VkBufferCreateInfo = .{
        .size = size,
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) {
        return error.BufferCreateFailed;
    }

    // 2. Query memory requirements.
    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    // 3. Pick a memory type.
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

    // 4. Allocate + bind.
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

    // 5. Map if requested. We hold the map for the buffer's lifetime —
    //    Vulkan permits persistent mapping of HOST_VISIBLE memory, and
    //    the phase-1 codec's short-lived host accesses don't justify
    //    the unmap-after-each-use churn.
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

// ── Public API: encode ────────────────────────────────────────────

/// Encode `src_host` through the Vulkan L1 encoder. Allocates one
/// device-side stream bundle (lit/cmd/off16/length), uploads the source
/// bytes, dispatches the encode kernel, reads the produced sizes back,
/// and returns the bundle. Caller owns the bundle and must call
/// `freeStreams` on it exactly once.
///
/// `src_host.len` must be ≤ 128 KB in phase 1 (single-chunk two-block
/// design — see spec §1.2). Larger inputs are rejected by the GLSL
/// shader (the parser only handles two LZ blocks).
///
/// `encode_spv` is the lz_encode SPIR-V bytes for the device's probed
/// tier. The caller resolves the tier and loads the bytes (typically
/// from `zig-out/shaders/lz_encode.<tier>.spv`) and threads them in.
/// The codec module deliberately does NOT load SPV itself — that lets
/// the test driver share the same load helper as dispatch_test and
/// match_any_bench, and sidesteps spv_blobs.zig's @embedFile path
/// resolution (see build.zig comment §M6).
pub fn encodeL1Sync(
    ctx: *driver.Context,
    src_host: []const u8,
    encode_spv: []const u8,
) L1Error!EncodeResult {
    const src_size: u32 = @intCast(src_host.len);

    // 1. Pick the tier (used as the cache key so a future shared-cache
    //    instance bucketizes per-device variants correctly).
    const tier = try pickTier(ctx);
    const spv = encode_spv;

    // 2. Worst-case stream sizes per spec §4.1. Each side stream caps at
    //    `2 * src_size`. We over-allocate by a small constant (16 B) so
    //    the encoder's lane-0 RMW byte stores into the tail u32 word
    //    never write past the buffer end.
    const stream_cap: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 64),
        (@as(vk.VkDeviceSize, src_size) * 2) + 16,
    );

    // 3. Allocate the src buffer + four output streams + the hash table
    //    + the sizes sidecar. All HOST_VISIBLE so the host can preload
    //    src and read back sizes; hash is DEVICE_LOCAL only (no host
    //    contact).
    var src_b = try createBuffer(
        ctx,
        @max(@as(vk.VkDeviceSize, 4), @as(vk.VkDeviceSize, src_size)),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        true,
    );
    errdefer destroyBuffer(ctx, &src_b);

    var lit_b = try createBuffer(ctx, stream_cap, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &lit_b);
    var cmd_b = try createBuffer(ctx, stream_cap, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &cmd_b);
    var off16_b = try createBuffer(ctx, stream_cap, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &off16_b);
    var length_b = try createBuffer(ctx, stream_cap, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &length_b);

    var hash_b = try createBuffer(ctx, HASH_SIZE_BYTES, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    defer destroyBuffer(ctx, &hash_b); // hash buffer is per-encode scratch

    var sizes_b = try createBuffer(
        ctx,
        @sizeOf(u32) * 4,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &sizes_b); // sizes are read back into the result struct

    // 4. Upload src bytes into src_b. HOST_COHERENT means no flush.
    if (src_size > 0) {
        const src_mapped = src_b.mapped orelse return error.MapMemoryFailed;
        @memcpy(src_mapped[0..src_size], src_host);
    }

    // 5. Zero the four output streams + sizes sidecar. The encoder's
    //    byte-packed RMW stores assume the destination word starts at
    //    zero (lz_encode.comp:170 comment). Hash is initialized inside
    //    the shader to HASH_EMPTY (0xFFFFFFFF), so we don't touch it.
    @memset(lit_b.mapped.?[0..@intCast(lit_b.size)], 0);
    @memset(cmd_b.mapped.?[0..@intCast(cmd_b.size)], 0);
    @memset(off16_b.mapped.?[0..@intCast(off16_b.size)], 0);
    @memset(length_b.mapped.?[0..@intCast(length_b.size)], 0);
    @memset(sizes_b.mapped.?[0..@intCast(sizes_b.size)], 0);

    // 6. Build the encode pipeline + descriptor set.
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_enc = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_encode",
        tier,
        spv,
        7, // bindings 0..6: src, lit, cmd, off16, length, hash, sizes
        @sizeOf(EncodePush),
    );

    const enc_bindings: [7]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = src_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = lit_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = cmd_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off16_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = length_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = hash_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = sizes_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const enc_set = try descriptors.allocSet(ctx, cached_enc, enc_bindings[0..]);

    // 7. Submit the encode dispatch (one workgroup per chunk = 1).
    const enc_push: EncodePush = .{
        .src_size = src_size,
        .hash_bits = HASH_BITS,
    };
    var enc_push_bytes: [@sizeOf(EncodePush)]u8 = undefined;
    @memcpy(enc_push_bytes[0..], std.mem.asBytes(&enc_push));

    _ = try dispatch.submitOne(
        ctx,
        cached_enc.pipeline,
        cached_enc.pipeline_layout,
        enc_set,
        enc_push_bytes[0..],
        .{ 1, 1, 1 },
    );

    // 8. Read back the sizes sidecar. HOST_COHERENT + the fence-wait
    //    inside submitOne is enough — no flush/invalidate needed.
    const sizes_words: [*]const u32 = @ptrCast(@alignCast(sizes_b.mapped.?));
    const lit_size = sizes_words[0];
    const cmd_size = sizes_words[1];
    const off16_count = sizes_words[2];
    const length_used = sizes_words[3];

    // 9. We no longer need the src buffer — the encoder is done. Free it
    //    here so the EncodeResult owns only the four output streams.
    destroyBuffer(ctx, &src_b);

    const streams: L1Streams = .{
        .lit_buf = lit_b.buf,
        .lit_mem = lit_b.mem,
        .lit_size = lit_size,
        .cmd_buf = cmd_b.buf,
        .cmd_mem = cmd_b.mem,
        .cmd_size = cmd_size,
        .off16_buf = off16_b.buf,
        .off16_mem = off16_b.mem,
        .off16_count = off16_count,
        .length_buf = length_b.buf,
        .length_mem = length_b.mem,
        .length_used = length_used,
    };
    // Mapped pointers are forgotten by L1Streams (the buffer handles
    // alone are enough for the decoder bind + the test's host readback —
    // host accesses go through fresh vkMapMemory calls). Drop the maps
    // here so the errdefer chain above doesn't double-unmap on success.
    if (vk.vkUnmapMemory_fn) |u| {
        u(ctx.dev, lit_b.mem);
        u(ctx.dev, cmd_b.mem);
        u(ctx.dev, off16_b.mem);
        u(ctx.dev, length_b.mem);
    }
    // Null out the local Buffer handles so the four `errdefer destroy
    // Buffer` calls above become no-ops on the success path. Belt-and-
    // suspenders: there's no `try` between here and the return, but a
    // future edit could insert one and we don't want to double-free.
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
///
/// `decode_spv` is the lz_decode SPIR-V bytes for the device's probed
/// tier — same caller-loads-it pattern as `encodeL1Sync`.
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

    // 1. Tier (used as cache key).
    const tier = try pickTier(ctx);
    const spv = decode_spv;

    // 2. Allocate the dst output buffer (HOST_VISIBLE so we can read
    //    back; size = original_size rounded up to 4 bytes so the
    //    decoder's u32-packed byte stores have room to RMW the tail
    //    word without overrunning).
    const dst_buf_size: vk.VkDeviceSize = @max(@as(vk.VkDeviceSize, 4), (@as(vk.VkDeviceSize, dst_size) + 3) & ~@as(vk.VkDeviceSize, 3));
    var dst_b = try createBuffer(
        ctx,
        dst_buf_size,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &dst_b);
    @memset(dst_b.mapped.?[0..@intCast(dst_b.size)], 0);

    // 3. Build the decode pipeline + descriptor set. The four input
    //    bindings reference the existing stream buffers (no copy).
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_dec = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_decode",
        tier,
        spv,
        5, // bindings 0..4: cmd, lit, off16, length, dst
        @sizeOf(DecodePush),
    );

    const dec_bindings: [5]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = streams.cmd_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.lit_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.off16_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.length_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    // 4. Submit the decode dispatch.
    const dec_push: DecodePush = .{
        .cmd_size = streams.cmd_size,
        .lit_size = streams.lit_size,
        .off16_count = streams.off16_count,
        .length_remaining = streams.length_used,
        .dst_size = dst_size,
    };
    var dec_push_bytes: [@sizeOf(DecodePush)]u8 = undefined;
    @memcpy(dec_push_bytes[0..], std.mem.asBytes(&dec_push));

    _ = try dispatch.submitOne(
        ctx,
        cached_dec.pipeline,
        cached_dec.pipeline_layout,
        dec_set,
        dec_push_bytes[0..],
        .{ 1, 1, 1 },
    );

    // 5. Copy dst out via the host map.
    if (dst_size > 0) {
        const dst_mapped = dst_b.mapped orelse return error.MapMemoryFailed;
        @memcpy(dst_host[0..dst_size], dst_mapped[0..dst_size]);
    }
}

// ── Debug helper: read back stream bytes ─────────────────────────
// Maps each stream buffer, copies up to `max_bytes` into the caller-
// supplied buffers, and unmaps. Returned slices reflect the actual
// (clamped) length. Returns null if mapping fails. Test-only helper —
// not production code.
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

    if (map(ctx.dev, streams.cmd_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        cmd_n = @min(out_cmd.len, streams.cmd_size);
        @memcpy(out_cmd[0..cmd_n], p[0..cmd_n]);
        unmap(ctx.dev, streams.cmd_mem);
    }
    if (map(ctx.dev, streams.lit_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        lit_n = @min(out_lit.len, streams.lit_size);
        @memcpy(out_lit[0..lit_n], p[0..lit_n]);
        unmap(ctx.dev, streams.lit_mem);
    }
    if (map(ctx.dev, streams.off16_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        off16_n = @min(out_off16.len, streams.off16_count * 2);
        @memcpy(out_off16[0..off16_n], p[0..off16_n]);
        unmap(ctx.dev, streams.off16_mem);
    }
    if (map(ctx.dev, streams.length_mem, 0, vk.VK_WHOLE_SIZE, 0, &raw) == vk.VK_SUCCESS) {
        const p: [*]const u8 = @ptrCast(raw.?);
        length_n = @min(out_length.len, streams.length_used);
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
}
