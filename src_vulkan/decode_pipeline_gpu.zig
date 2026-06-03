//! GPU-side SLZ1 decode-pipeline orchestration (Phase 2 integration).
//!
//! Drives the 5 newly-ported Vulkan kernels in a single command buffer:
//!
//!   1. walk_frame         — parse SLZ1 frame on GPU → ChunkDescs + Meta.
//!   2. prefix_sum_chunks  — exclusive prefix sum of per-chunk sub-chunk
//!                            counts → first_sub_idx[] + total_subchunks.
//!   3. scan_parse         — walk each chunk's sub-chunks → staged huff
//!                            descs (lit/tok/hi/lo) + staged raw off16
//!                            descs (hi/lo).
//!   4a-4d. compact_huff_descs (×4)
//!                         — compact each of the four staged huff arrays
//!                            into a dense per-stream array + count.
//!   4e. compact_raw_descs — compact staged raw hi/lo into one dense
//!                            array + count.
//!   5. gather_raw_off16   — scatter raw off16 sub-streams from the
//!                            compressed bytes into the off16 entropy
//!                            scratch.
//!
//! Mirrors the CUDA orchestration in `src/decode/decode_dispatch.zig::
//! fullGpuLaunchImpl` and `src/decode/scan_gpu.zig::{gpuWalkFrameImpl,
//! gpuPrefixSumChunksImpl, gpuScanChunks}` + `gatherRawOff16` —
//! see those files for the per-launch parameter shapes the CUDA
//! reference uses.
//!
//! All five dispatches share one VkCommandBuffer (one submit, one fence).
//! Inter-dispatch synchronization rides VkCmdPipelineBarrier with
//! srcStage=dstStage=COMPUTE and srcAccessMask=SHADER_WRITE +
//! dstAccessMask=SHADER_READ on a single VkMemoryBarrier — cheaper than
//! per-buffer barriers and sufficient because the kernels read
//! disjoint regions of the same set of buffers.
//!
//! Buffer placement (P1 / P4 fix from `slz1_codec.zig` decode):
//!   * Intermediate device-only SSBOs (Chunks, FirstSubIdx, TotalSubs,
//!     Staged*, Compact*, NCounts, Off16Scratch) are .device_local_only —
//!     they're never read by host.
//!   * Frame (compressed bytes input) lives in host_visible_prefer_device_
//!     local so the host can populate it cheaply; subsequent reads are
//!     all device-side.
//!   * Meta (6 u32) lives in host_visible_sysmem because the host needs
//!     to read n_chunks + status + decomp_size before queueing the back-
//!     half lz_decode dispatch.
//!
//! For L1 inputs (no Huffman, no raw off16 sub-streams) every kernel's
//! "real work" portion is a no-op — scan_parse writes valid=0 to every
//! staged slot, the four compact-huff kernels each emit 0 entries, the
//! compact-raw kernel emits 0 entries, and gather_raw_off16 self-gates
//! to nothing. The kernels still dispatch (matching the CUDA orchestrator
//! exactly) so a) the pipeline shape stays uniform across L1/L2/L3+
//! frames and b) future Phase 3 work that introduces Huffman-encoded
//! L3+ frames lights up automatically.
//!
//! Public API:
//!   pub fn runDecodePipeline(ctx, slz_bytes, allocator) !DecodeResult
//!     Runs the 5-kernel chain, returns a DecodeResult bundle. The
//!     caller reads the populated Meta fields (n_chunks, decomp_size,
//!     status) host-side to decide what to do next. On the L1 path the
//!     caller still drives `lz_decode.comp` via the CPU-unwrapped
//!     per-stream byte arrays from `wire_format.unwrapSlz1ToL1Streams`
//!     — the new GPU outputs (ChunkDescs + entropy scratch) are
//!     consumed by Phase 3+ kernels.
//!
//!   pub fn destroyDecodeResult(ctx, result)
//!     Tears down the device buffers the pipeline allocated.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const l1_codec = @import("l1_codec.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");
const wire_constants = @import("wire_constants.zig");
const spv_blobs = @import("spv_blobs");

// ── Constants that mirror the shared GLSL header ─────────────────────
//
// Sourced from `wire_constants.zig` (Cluster H consolidation). The GLSL
// header `decode_pipeline_shared.glsl` carries identical values; the
// comptime checks in `wire_constants` enforce host/GLSL coherence.

/// Max chunks the walk-frame kernel is sized to accept (matches the
/// `WALK_MAX_CHUNKS` constant in `decode_pipeline_shared.glsl`).
pub const WALK_MAX_CHUNKS: u32 = wire_constants.WALK_MAX_CHUNKS;

/// Worst-case sub-chunks per chunk. Matches `MAX_SUB_CHUNKS_PER_CHUNK`
/// in the shared GLSL header.
pub const MAX_SUB_CHUNKS_PER_CHUNK: u32 = wire_constants.MAX_SUB_CHUNKS_PER_CHUNK;

/// Per-sub-chunk entropy-scratch slot stride. Matches
/// `ENTROPY_SCRATCH_SLOT_BYTES` (128 KiB).
pub const ENTROPY_SCRATCH_SLOT_BYTES: u32 = wire_constants.ENTROPY_SCRATCH_SLOT_BYTES;

/// Outer-block boundary in bytes (256 KiB) for an SLZ1 frame. Mirrors
/// `SLZ_CHUNK_SIZE_BYTES` in the shared GLSL header.
pub const SLZ_CHUNK_SIZE_BYTES: u32 = wire_constants.SLZ_CHUNK_SIZE_BYTES;

/// Slot widths (u32 counts) of the descriptor structs the 5 kernels
/// exchange. Must stay byte-identical to the shared GLSL header.
pub const CHUNK_DESC_U32_COUNT: u32 = 6;
pub const SCANHUFF_U32_COUNT: u32 = 5;
pub const SCANRAW_U32_COUNT: u32 = 4;
pub const HUFFDESC_U32_COUNT: u32 = 5;
pub const RAWDESC_U32_COUNT: u32 = 3;
pub const WALK_META_U32_COUNT: u32 = 6;

/// Meta-slot offsets — host reads `meta[WALK_META_*_SLOT]` after the
/// dispatch chain completes.
pub const WALK_META_N_CHUNKS_SLOT: u32 = 0;
pub const WALK_META_DECOMP_SIZE_SLOT: u32 = 1;
pub const WALK_META_SUB_CHUNK_CAP_SLOT: u32 = 2;
pub const WALK_META_BLOCK_START_SLOT: u32 = 3;
pub const WALK_META_BLOCK_SIZE_SLOT: u32 = 4;
pub const WALK_META_STATUS_SLOT: u32 = 5;

pub const PipelineError = error{
    UnsupportedTier,
    NoSpvForTier,
    OutOfMemory,
    BufferCreateFailed,
    MemoryAllocateFailed,
    MemoryTypeNotFound,
    BindBufferFailed,
    MapMemoryFailed,
    BadFrame,
    TooManyChunks,
    LoaderNotReady,
    CommandPoolCreateFailed,
    CommandBufferAllocateFailed,
    FenceCreateFailed,
    QueryPoolCreateFailed,
    BeginCommandBufferFailed,
    EndCommandBufferFailed,
    ResetCommandBufferFailed,
    ResetFenceFailed,
    SubmitFailed,
    FenceWaitTimeout,
    FenceWaitFailed,
    QueryReadFailed,
} ||
    l1_codec.L1Error ||
    descriptors.DescriptorError ||
    dispatch.DispatchError;

// ── Push-constant blocks (mirror the .comp shaders) ──────────────────

const WalkPush = extern struct {
    frame_size: u32,
    max_chunks: u32,
};

const PrefixPush = extern struct {
    n_chunks: u32,
    sub_chunk_cap: u32,
};

const ScanPush = extern struct {
    block_len: u32,
    sub_chunk_cap: u32,
};

const GatherPush = extern struct {
    comp_len: u32,
};

// ── Memory-type helpers ──────────────────────────────────────────────

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

fn findHostVisibleNonDeviceLocal(
    pd: vk.VkPhysicalDevice,
    type_bits_mask: u32,
) ?u32 {
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return null;
    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(pd, &mem_props);
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (type_bits_mask & (@as(u32, 1) << @intCast(i))) != 0;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        const host_visible = (flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        const host_coherent = (flags & vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        const device_local = (flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (supported and host_visible and host_coherent and !device_local) return i;
    }
    return null;
}

// ── Buffer helper ────────────────────────────────────────────────────

const BufferMode = enum {
    /// Pure device memory — fastest GPU access, no host mapping.
    device_local_only,
    /// Host visible, prefer device-local (rebar-style on NVIDIA).
    host_visible_prefer_device_local,
    /// Host visible, explicitly NOT device-local — sysmem reads are
    /// driver-cached, important for the small Meta buffer the host
    /// reads back.
    host_visible_sysmem,
};

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
    mode: BufferMode,
) PipelineError!Buffer {
    l1_codec.ensureBufferFnSlots(ctx);
    const create_buf = vk.vkCreateBuffer_fn orelse return error.BufferCreateFailed;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse return error.BufferCreateFailed;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return error.MemoryAllocateFailed;
    const bind = vk.vkBindBufferMemory_fn orelse return error.BindBufferFailed;

    const bci: vk.VkBufferCreateInfo = .{
        .size = @max(size, 4),
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) return error.BufferCreateFailed;

    var req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &req);

    const mt_idx: u32 = blk: {
        switch (mode) {
            .device_local_only => {
                if (findMemoryType(ctx.pd, req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) |i| break :blk i;
            },
            .host_visible_prefer_device_local => {
                const ideal = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                const fallback = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if (findMemoryType(ctx.pd, req.memoryTypeBits, ideal)) |i| break :blk i;
                if (findMemoryType(ctx.pd, req.memoryTypeBits, fallback)) |i| break :blk i;
            },
            .host_visible_sysmem => {
                if (findHostVisibleNonDeviceLocal(ctx.pd, req.memoryTypeBits)) |i| break :blk i;
                const fallback = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if (findMemoryType(ctx.pd, req.memoryTypeBits, fallback)) |i| break :blk i;
            },
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
    if (mode != .device_local_only) {
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

    return .{ .buf = buf, .mem = mem, .mapped = mapped, .size = req.size };
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

// ── SPV helpers ──────────────────────────────────────────────────────

fn tierBlob(t: probe_mod.Tier) ?spv_blobs.Tier {
    return switch (t) {
        .tier1 => .tier1,
        .tier1_nv => .tier1_nv,
        .tier2 => .tier2,
        .unsupported => null,
    };
}

fn dupAlignedSpv(allocator: std.mem.Allocator, spv: []const u8) PipelineError![]align(4) u8 {
    const buf = allocator.alignedAlloc(u8, .@"4", spv.len) catch return error.OutOfMemory;
    @memcpy(buf, spv);
    return buf;
}

// ── Public result type ───────────────────────────────────────────────

pub const DecodeResult = struct {
    // Device buffers — caller owns lifetime via destroyDecodeResult.
    frame: Buffer = .{},
    chunks: Buffer = .{},
    meta: Buffer = .{},
    first_sub_idx: Buffer = .{},
    total_subs: Buffer = .{},
    n_chunks_scratch: Buffer = .{},
    staged: Buffer = .{},
    compact_lit: Buffer = .{},
    compact_tok: Buffer = .{},
    compact_hi: Buffer = .{},
    compact_lo: Buffer = .{},
    compact_raw: Buffer = .{},
    compact_counts: Buffer = .{},
    off16_scratch: Buffer = .{},

    // Host-side aggregate (read from `meta.mapped` after the dispatch
    // chain completes). All 6 u32 are read out to a local copy so the
    // caller doesn't have to chase the host pointer.
    n_chunks: u32 = 0,
    decomp_size: u32 = 0,
    sub_chunk_cap: u32 = 0,
    block_start: u32 = 0,
    block_size: u32 = 0,
    status: u32 = 0,

    // Geometry constants the staged/compact buffers were sized against —
    // future Phase-3 kernels need these so they can index into the
    // device-resident arrays.
    total_subchunks: u32 = 0,
};

pub fn destroyDecodeResult(ctx: *driver.Context, r: *DecodeResult) void {
    destroyBuffer(ctx, &r.frame);
    destroyBuffer(ctx, &r.chunks);
    destroyBuffer(ctx, &r.meta);
    destroyBuffer(ctx, &r.first_sub_idx);
    destroyBuffer(ctx, &r.total_subs);
    destroyBuffer(ctx, &r.n_chunks_scratch);
    destroyBuffer(ctx, &r.staged);
    destroyBuffer(ctx, &r.compact_lit);
    destroyBuffer(ctx, &r.compact_tok);
    destroyBuffer(ctx, &r.compact_hi);
    destroyBuffer(ctx, &r.compact_lo);
    destroyBuffer(ctx, &r.compact_raw);
    destroyBuffer(ctx, &r.compact_counts);
    destroyBuffer(ctx, &r.off16_scratch);
}

// ── Internal pipeline-cache holder ───────────────────────────────────

const Pipelines = struct {
    walk: descriptors.CachedPipeline,
    prefix: descriptors.CachedPipeline,
    scan: descriptors.CachedPipeline,
    compact_huff: descriptors.CachedPipeline,
    compact_raw: descriptors.CachedPipeline,
    gather: descriptors.CachedPipeline,
};

fn loadAllPipelines(
    ctx: *driver.Context,
    cache: *descriptors.Cache,
    tier: probe_mod.Tier,
    tier_b: spv_blobs.Tier,
    allocator: std.mem.Allocator,
) PipelineError!Pipelines {
    const walk_spv_raw = spv_blobs.find("walk_frame", tier_b) orelse return error.NoSpvForTier;
    const walk_spv = try dupAlignedSpv(allocator, walk_spv_raw);
    defer allocator.free(walk_spv);
    const walk = try descriptors.getOrCreate(ctx, cache, "walk_frame", tier, walk_spv, 3, @sizeOf(WalkPush));

    const prefix_spv_raw = spv_blobs.find("prefix_sum_chunks", tier_b) orelse return error.NoSpvForTier;
    const prefix_spv = try dupAlignedSpv(allocator, prefix_spv_raw);
    defer allocator.free(prefix_spv);
    const prefix = try descriptors.getOrCreate(ctx, cache, "prefix_sum_chunks", tier, prefix_spv, 3, @sizeOf(PrefixPush));

    const scan_spv_raw = spv_blobs.find("scan_parse", tier_b) orelse return error.NoSpvForTier;
    const scan_spv = try dupAlignedSpv(allocator, scan_spv_raw);
    defer allocator.free(scan_spv);
    const scan = try descriptors.getOrCreate(ctx, cache, "scan_parse", tier, scan_spv, 10, @sizeOf(ScanPush));

    const compact_huff_spv_raw = spv_blobs.find("compact_huff_descs", tier_b) orelse return error.NoSpvForTier;
    const compact_huff_spv = try dupAlignedSpv(allocator, compact_huff_spv_raw);
    defer allocator.free(compact_huff_spv);
    const compact_huff = try descriptors.getOrCreate(ctx, cache, "compact_huff_descs", tier, compact_huff_spv, 4, 0);

    const compact_raw_spv_raw = spv_blobs.find("compact_raw_descs", tier_b) orelse return error.NoSpvForTier;
    const compact_raw_spv = try dupAlignedSpv(allocator, compact_raw_spv_raw);
    defer allocator.free(compact_raw_spv);
    const compact_raw = try descriptors.getOrCreate(ctx, cache, "compact_raw_descs", tier, compact_raw_spv, 5, 0);

    const gather_spv_raw = spv_blobs.find("gather_raw_off16", tier_b) orelse return error.NoSpvForTier;
    const gather_spv = try dupAlignedSpv(allocator, gather_spv_raw);
    defer allocator.free(gather_spv);
    const gather = try descriptors.getOrCreate(ctx, cache, "gather_raw_off16", tier, gather_spv, 4, @sizeOf(GatherPush));

    return .{
        .walk = walk,
        .prefix = prefix,
        .scan = scan,
        .compact_huff = compact_huff,
        .compact_raw = compact_raw,
        .gather = gather,
    };
}

// ── VK_EXT_debug_utils label helpers ─────────────────────────────────
//
// `beginLabel` / `endLabel` wrap `vkCmd{Begin,End}DebugUtilsLabelEXT`
// with a null-check (silent no-op when the extension was not loaded,
// e.g. on a clean box without the Vulkan SDK installed). All five
// inline `cmd_dispatch` calls below are wrapped so Nsight Systems'
// Vulkan trace can attribute the per-kernel GPU intervals in the
// vulkan_gpu_marker_sum report.
//
// 64-byte stack buffer caps the label length (longest emitted here
// is "prefix_sum_chunks" at 17 bytes; "compact_huff_*" at 14 bytes).
// dispatch.zig has the same helper for its submit* wrappers; the
// duplicate here exists because runDecodePipelineEx records the
// cmdbuf inline rather than going through dispatch.submitOne.
inline fn beginLabel(cmd_buf: vk.VkCommandBuffer, name: []const u8) void {
    const fn_ptr = vk.vkCmdBeginDebugUtilsLabelEXT_fn orelse return;
    if (name.len == 0) return;
    var buf: [64]u8 = @splat(0);
    const cap: usize = @min(name.len, buf.len - 1);
    @memcpy(buf[0..cap], name[0..cap]);
    buf[cap] = 0;
    const info: vk.VkDebugUtilsLabelEXT = .{
        .pLabelName = @ptrCast(&buf[0]),
        .color = .{ 0, 0, 0, 0 },
    };
    fn_ptr(cmd_buf, &info);
}

inline fn endLabel(cmd_buf: vk.VkCommandBuffer, name: []const u8) void {
    const fn_ptr = vk.vkCmdEndDebugUtilsLabelEXT_fn orelse return;
    if (name.len == 0) return;
    fn_ptr(cmd_buf);
}

// ── Inter-dispatch barrier ───────────────────────────────────────────
//
// All five logical fences (Barriers A..E in the Phase-1 plan) use the
// same shape: a single global VkMemoryBarrier with
// srcAccessMask=SHADER_WRITE → dstAccessMask=SHADER_READ at the COMPUTE
// stage on both sides. A global memory barrier is cheaper than 6
// per-buffer barriers and sufficient because the kernels write
// disjoint regions.
//
// F067: returns an error rather than silently no-oping when
// `vkCmdPipelineBarrier_fn` is unresolved. A missing barrier in a
// kernel chain corrupts results undetectably; better to fail loud than
// to ship incorrect output.
fn recordComputeBarrier(cmd_buf: vk.VkCommandBuffer) PipelineError!void {
    const cmd_barrier = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
    const mb: vk.VkMemoryBarrier = .{
        .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
    };
    const mbs: [1]vk.VkMemoryBarrier = .{mb};
    cmd_barrier(
        cmd_buf,
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        @ptrCast(&mbs),
        0,
        null,
        0,
        null,
    );
}

// ── Public entry point ───────────────────────────────────────────────

/// Optional caller hints for buffer sizing — when the caller has
/// already CPU-parsed the frame (`wire_format.unwrapSlz1ToL1Streams`),
/// the real `n_chunks` and `decomp_size` are passed here so the
/// pipeline doesn't over-allocate against the WALK_MAX_CHUNKS upper
/// bound. The off16 entropy scratch in particular is
/// `n_chunks * MAX_SUB_CHUNKS_PER_CHUNK * 128 KiB` per the CUDA
/// reference; at WALK_MAX_CHUNKS = 16384 that would balloon to 8 GiB
/// of VRAM, which exceeds the available VRAM on every Tier-2 device
/// and many Tier-1 devices. Sizing against the real `n_chunks` keeps
/// the allocation proportional to the input.
pub const PipelineHints = struct {
    /// Real `n_chunks` value the CPU unwrap reported. 0 means "no
    /// hint" — the pipeline sizes against a conservative upper bound
    /// derived from `slz_bytes.len / SLZ_CHUNK_SIZE_BYTES + 2`.
    n_chunks_hint: u32 = 0,
};

/// Run the 5-kernel GPU decode pipeline on `slz_bytes`. Returns a
/// DecodeResult bundle containing all device-side outputs and the
/// host-readable Meta. Caller MUST call `destroyDecodeResult` to free.
///
/// On success the meta status will be zero (FW_STATUS_OK); a non-zero
/// status is the walk-kernel's parse error code (FW_STATUS_* per
/// `decode_pipeline_shared.glsl`). The caller maps walk-kernel
/// statuses to host error codes — this function leaves it raw.
pub fn runDecodePipeline(
    ctx: *driver.Context,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
) PipelineError!DecodeResult {
    return runDecodePipelineEx(ctx, allocator, slz_bytes, .{});
}

pub fn runDecodePipelineEx(
    ctx: *driver.Context,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    hints: PipelineHints,
) PipelineError!DecodeResult {
    if (slz_bytes.len == 0) return error.BadFrame;
    if (slz_bytes.len > std.math.maxInt(u32)) return error.BadFrame;

    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const tier_b = tierBlob(tier) orelse return error.UnsupportedTier;

    // Right-size the staged / compact / scratch buffers against the
    // caller's hint (or a frame-size-derived upper bound). Off16 scratch
    // is `n_chunks_max * MAX_SUB_CHUNKS_PER_CHUNK * 128 KiB`; at
    // WALK_MAX_CHUNKS that's 8 GiB, which exceeds available VRAM on
    // every Tier-2 device. The CUDA reference (`decode_dispatch.zig`
    // line 541-549) does the same right-sizing pass using a
    // host-known n_chunks.
    //
    // SLZ_CHUNK_SIZE_BYTES = 256 KiB is the maximum compressed-block
    // boundary the wire format admits; the worst-case n_chunks is
    // `(decomp_size / SLZ_CHUNK_SIZE_BYTES) + 2`. We use `slz_bytes.len`
    // as a proxy upper bound for decomp_size (compressed is always
    // ≤ decompressed) plus a safety pad.
    const slz_chunk_size_bytes: u32 = SLZ_CHUNK_SIZE_BYTES; // 256 KiB (wire-format outer block bound)
    const n_chunks_max: u32 = if (hints.n_chunks_hint != 0)
        @max(hints.n_chunks_hint, 1)
    else
        @min(
            WALK_MAX_CHUNKS,
            @as(u32, @intCast(@max(slz_bytes.len / slz_chunk_size_bytes + 2, 4))),
        );
    if (n_chunks_max > WALK_MAX_CHUNKS) return error.TooManyChunks;
    const max_total_subchunks: u32 = n_chunks_max * MAX_SUB_CHUNKS_PER_CHUNK;

    var result: DecodeResult = .{};
    errdefer destroyDecodeResult(ctx, &result);

    // ── 1. Allocate buffers ──────────────────────────────────────────

    // Frame: u32-packed compressed SLZ1 bytes. host_visible_prefer_
    // device_local so the host fill below stays cheap on both NVIDIA
    // (rebar) and Intel iGPU (shared heap).
    const frame_bytes: vk.VkDeviceSize = @intCast((slz_bytes.len + 3) & ~@as(usize, 3));
    result.frame = try createBuffer(
        ctx,
        @max(frame_bytes, 4),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .host_visible_prefer_device_local,
    );

    // Chunks: WALK_MAX_CHUNKS * 6 u32 (walk_frame is unconditionally
    // sized at WALK_MAX_CHUNKS because the kernel's max_chunks bound
    // protects it from overflow). At 16384 chunks this is ~400 KiB —
    // small enough to leave at the worst case regardless of hint.
    const chunks_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, WALK_MAX_CHUNKS) * CHUNK_DESC_U32_COUNT * 4;
    result.chunks = try createBuffer(ctx, chunks_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);

    // Meta: 6 u32. host_visible_sysmem because we read the count + status
    // on the host immediately after the dispatch chain.
    const meta_bytes: vk.VkDeviceSize = WALK_META_U32_COUNT * 4;
    result.meta = try createBuffer(
        ctx,
        meta_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .host_visible_sysmem,
    );

    // FirstSubIdx: WALK_MAX_CHUNKS u32.
    const first_sub_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, WALK_MAX_CHUNKS) * 4;
    result.first_sub_idx = try createBuffer(ctx, first_sub_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);

    // TotalSubs: 1 u32.
    result.total_subs = try createBuffer(ctx, 4, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);

    // NChunks scratch: 1 u32 (for the scan kernel self-gate). Initialised
    // by a tiny vkCmdCopyBuffer from the meta buffer below — see the
    // dispatch sequence.
    result.n_chunks_scratch = try createBuffer(
        ctx,
        4,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .device_local_only,
    );

    // Staged: [lit][tok][hi][lo] huff + [raw_hi][raw_lo] raw, each at
    // max_total_subchunks count. Single device buffer; the four huff
    // bindings + two raw bindings split it at offsets.
    const huff_arr_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * SCANHUFF_U32_COUNT * 4;
    const raw_arr_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * SCANRAW_U32_COUNT * 4;
    const staged_bytes: vk.VkDeviceSize = huff_arr_bytes * 4 + raw_arr_bytes * 2;
    // Bind separate VkBuffers for each sub-region (descriptors need disjoint
    // VkBuffer handles) — one device allocation with VkBuffers at different
    // offsets would work too, but each `createBuffer` call allocates its own
    // VkDeviceMemory anyway. Sized at the worst-case per-region count.
    result.staged = try createBuffer(ctx, staged_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);

    // Compact buffers: one per stream type. Sized against worst-case
    // per-stream output counts (matches CUDA's `huff_compact_max`).
    const huff_compact_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * HUFFDESC_U32_COUNT * 4;
    const raw_compact_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks * 2) * RAWDESC_U32_COUNT * 4;
    result.compact_lit = try createBuffer(ctx, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_tok = try createBuffer(ctx, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_hi = try createBuffer(ctx, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_lo = try createBuffer(ctx, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_raw = try createBuffer(ctx, raw_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);

    // Compact counts: [n_lit, n_tok, n_hi, n_lo, n_raw, n_merged].
    // The 5 GLSL compact kernels each take a writeonly 1-u32 NOut SSBO;
    // we sub-bind into this 6-u32 buffer via per-binding `VkDescriptor
    // BufferInfo.offset` so all five writes land in one device alloc.
    result.compact_counts = try createBuffer(
        ctx,
        6 * 4,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .device_local_only,
    );

    // Off16 entropy scratch — sized at worst-case total_subchunks slots,
    // each slot ENTROPY_SCRATCH_SLOT_BYTES wide.
    const off16_scratch_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * ENTROPY_SCRATCH_SLOT_BYTES;
    result.off16_scratch = try createBuffer(
        ctx,
        off16_scratch_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .device_local_only,
    );

    // ── 2. Populate frame buffer from slz_bytes ──────────────────────
    if (result.frame.mapped) |m| {
        @memcpy(m[0..slz_bytes.len], slz_bytes);
        // Pad to 4-byte alignment of frame_buf.w[]; bytes past
        // slz_bytes.len are never read (walk_frame guards against
        // frame_size > pc.frame_size).
        if (frame_bytes > slz_bytes.len) {
            @memset(m[slz_bytes.len..@intCast(frame_bytes)], 0);
        }
    } else {
        return error.MapMemoryFailed;
    }

    // Pre-zero meta so the host read sees 0 if the dispatch chain
    // somehow skipped writing.
    if (result.meta.mapped) |m| @memset(m[0..@intCast(meta_bytes)], 0);

    // ── 3. Build pipelines + descriptor sets ─────────────────────────

    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const pipes = try loadAllPipelines(ctx, &cache, tier, tier_b, allocator);

    // walk_frame: 3 bindings [Frame, Chunks, Meta] + push (frame_size, max_chunks).
    const walk_bindings: [3]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.meta.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const walk_set = try descriptors.allocSet(ctx, pipes.walk, walk_bindings[0..]);

    // prefix_sum_chunks: 3 bindings [Chunks, FirstSubIdx, TotalSubs] + push.
    const prefix_bindings: [3]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.first_sub_idx.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const prefix_set = try descriptors.allocSet(ctx, pipes.prefix, prefix_bindings[0..]);

    // scan_parse: 10 bindings + push.
    //   Bindings 4..7 (staged huff lit/tok/hi/lo) take the same staged
    //   buffer at different offsets; binding 8/9 take the staged buffer
    //   at the raw-hi/raw-lo offsets. All entries are 4-byte aligned
    //   (descriptor offsets must satisfy
    //   `minStorageBufferOffsetAlignment` — a few hundred bytes on every
    //   sane device; the worst-case alignment is 256 on a few mobile
    //   parts, which the multi-megabyte huff/raw arrays satisfy
    //   trivially).
    const scan_bindings: [10]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.first_sub_idx.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.n_chunks_scratch.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.staged.buf, .offset = 0, .range = huff_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes, .range = huff_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 2, .range = huff_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 3, .range = huff_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 4, .range = raw_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 4 + raw_arr_bytes, .range = raw_arr_bytes },
    };
    const scan_set = try descriptors.allocSet(ctx, pipes.scan, scan_bindings[0..]);

    // compact_huff_descs (×4): each needs (StagedIn, TotalSubs, Out, NOut).
    const compact_lit_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.staged.buf, .offset = 0, .range = huff_arr_bytes },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_lit.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 0, .range = 4 },
    };
    const compact_lit_set = try descriptors.allocSet(ctx, pipes.compact_huff, compact_lit_bindings[0..]);

    const compact_tok_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes, .range = huff_arr_bytes },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_tok.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 4, .range = 4 },
    };
    const compact_tok_set = try descriptors.allocSet(ctx, pipes.compact_huff, compact_tok_bindings[0..]);

    const compact_hi_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 2, .range = huff_arr_bytes },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_hi.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 8, .range = 4 },
    };
    const compact_hi_set = try descriptors.allocSet(ctx, pipes.compact_huff, compact_hi_bindings[0..]);

    const compact_lo_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 3, .range = huff_arr_bytes },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_lo.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 12, .range = 4 },
    };
    const compact_lo_set = try descriptors.allocSet(ctx, pipes.compact_huff, compact_lo_bindings[0..]);

    // compact_raw_descs: (StagedHi, StagedLo, TotalSubs, Out, NOut).
    const compact_raw_bindings: [5]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 4, .range = raw_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 4 + raw_arr_bytes, .range = raw_arr_bytes },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_raw.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 16, .range = 4 },
    };
    const compact_raw_set = try descriptors.allocSet(ctx, pipes.compact_raw, compact_raw_bindings[0..]);

    // gather_raw_off16: (Comp, Scratch, Descs, NDescs) + push (comp_len).
    const gather_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.off16_scratch.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_raw.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 16, .range = 4 },
    };
    const gather_set = try descriptors.allocSet(ctx, pipes.gather, gather_bindings[0..]);

    // ── 4. Record the command buffer ─────────────────────────────────

    try dispatch.ensureChassisPub(ctx);

    const reset_cb = vk.vkResetCommandBuffer_fn orelse return error.LoaderNotReady;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return error.LoaderNotReady;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return error.LoaderNotReady;
    const cmd_bind_pl = vk.vkCmdBindPipeline_fn orelse return error.LoaderNotReady;
    const cmd_bind_ds = vk.vkCmdBindDescriptorSets_fn orelse return error.LoaderNotReady;
    const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
    const cmd_dispatch = vk.vkCmdDispatch_fn orelse return error.LoaderNotReady;
    const cmd_copy = vk.vkCmdCopyBuffer_fn orelse return error.LoaderNotReady;
    const reset_fence = vk.vkResetFences_fn orelse return error.LoaderNotReady;
    const submit = vk.vkQueueSubmit_fn orelse return error.LoaderNotReady;
    const wait_fence = vk.vkWaitForFences_fn orelse return error.LoaderNotReady;

    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return error.ResetCommandBufferFailed;

    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) return error.BeginCommandBufferFailed;

    // 0. Zero-clear the chunks + total_subs buffers via vkCmdFillBuffer.
    //    walk_frame only writes the first n_chunks slots of `chunks`;
    //    prefix_sum_chunks iterates the full WALK_MAX_CHUNKS range (we
    //    pass WALK_MAX_CHUNKS as n_chunks because we don't yet know the
    //    walk-frame-determined value host-side). Zeroing every chunk
    //    slot guarantees flagged-or-empty interpretations for the
    //    over-counted tail, which falls into the `n_subs=1` branch and
    //    keeps total_subchunks well-defined. Vulkan does NOT guarantee
    //    zero-initialized VkDeviceMemory (spec §11.4.2), so the fill is
    //    not optional — observed pre-fix on Intel Arc as garbage flags
    //    in slots past n_chunks.
    if (vk.vkCmdFillBuffer_fn) |cmd_fill| {
        cmd_fill(ctx.cmd_buf, result.chunks.buf, 0, chunks_bytes, 0);
        cmd_fill(ctx.cmd_buf, result.total_subs.buf, 0, 4, 0);
        cmd_fill(ctx.cmd_buf, result.first_sub_idx.buf, 0, first_sub_bytes, 0);
        cmd_fill(ctx.cmd_buf, result.compact_counts.buf, 0, 6 * 4, 0);
        // Staged-buffer pre-clear: scan_parse self-clears `valid=0`
        // every iteration but only touches slots [0, n_chunks * MAX_SUB_
        // CHUNKS_PER_CHUNK). Slots past that range MUST be `valid==0`
        // for the compact kernels to drop them. Vulkan device memory is
        // not zero-init; clear it explicitly here.
        cmd_fill(ctx.cmd_buf, result.staged.buf, 0, staged_bytes, 0);
    } else {
        // vkCmdFillBuffer is core Vulkan 1.0; missing slot indicates a
        // broken loader. Bail rather than silently emit a kernel chain
        // that reads uninitialized SSBO contents.
        return error.LoaderNotReady;
    }
    // Barrier 0 — transfer (fill) → compute reads on chunks /
    // total_subs / first_sub_idx / staged / compact_counts.
    {
        const cmd_barrier_fn = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
        const mb: vk.VkMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        };
        const mbs: [1]vk.VkMemoryBarrier = .{mb};
        cmd_barrier_fn(
            ctx.cmd_buf,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            @ptrCast(&mbs),
            0,
            null,
            0,
            null,
        );
    }

    // 1. walk_frame.
    {
        cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.walk.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{walk_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipes.walk.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
        const push: WalkPush = .{
            .frame_size = @intCast(slz_bytes.len),
            .max_chunks = WALK_MAX_CHUNKS,
        };
        var push_bytes: [@sizeOf(WalkPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(
            ctx.cmd_buf,
            pipes.walk.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_bytes.len),
            @ptrCast(&push_bytes),
        );
        beginLabel(ctx.cmd_buf, "walk_frame");
        cmd_dispatch(ctx.cmd_buf, 1, 1, 1);
        endLabel(ctx.cmd_buf, "walk_frame");
    }

    // BARRIER A — walk_frame writes → prefix_sum_chunks reads.
    try recordComputeBarrier(ctx.cmd_buf);

    // 2. prefix_sum_chunks. Push uses worst-case n_chunks = WALK_MAX_CHUNKS
    //    minus a loop early-exit: the kernel iterates `i < n_chunks` but
    //    we don't know n_chunks until after walk_frame ran. We compromise
    //    by passing WALK_MAX_CHUNKS as n_chunks; the kernel will iterate
    //    over the entire range but flagged-or-empty chunks beyond the
    //    real n_chunks all read flags=0 and decomp_size=0 (we pre-zeroed
    //    the device meta buffer above), so they fall into the `n_subs=1`
    //    branch and bump first_sub_idx[i] by 1 each. That over-counts
    //    total_subchunks for slots past n_chunks, but downstream kernels
    //    self-gate on the real value (compact_*'s loop reads from
    //    total_subs which mismatches reality, BUT each pre-zeroed staged
    //    slot has valid=0 so the compact kernels emit no entries from
    //    the over-counted slots — exact match for the L1 fast path).
    //
    //    NOTE: this is a correctness-preserving over-count. The CUDA
    //    reference instead H2D's the real n_chunks before launching
    //    prefix_sum (`gpuPrefixSumChunksImpl(self, ..., n_chunks, ...)`).
    //    We can't do that here because we want the walk_frame result on
    //    GPU. A future patch should split this into two command-buffer
    //    submissions (walk → fence → readback n_chunks → prefix_sum...)
    //    or add a meta→push-constant indirection kernel.
    //
    //    For the L1 fast path this is harmless: every over-counted
    //    sub-chunk has valid=0 (chunks_buf slot 4 was zeroed) and the
    //    compact kernels reject them.
    //
    //    The chunks buffer is pre-zeroed by the walk_frame kernel — it
    //    only writes the first n_chunks slots, leaving slots
    //    [n_chunks, WALK_MAX_CHUNKS) at the device-local-allocate-time
    //    zero (Vulkan does not zero VkDeviceMemory by default, but
    //    `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` allocations land in the
    //    same zero-initialized device pages on every conforming driver
    //    we exercise — verified by the L1 round-trip suite and the
    //    Phase 2 scan-parse kernel's clearAllValid semantics).
    //
    //    Belt-and-suspenders: a vkCmdFillBuffer here would guarantee the
    //    zero state but adds another barrier hop. The walk_frame kernel
    //    explicitly writes slot 4 (flags) and slot 2 (decomp_size) for
    //    every chunk it emits, so the over-count loop only reads
    //    not-yet-written slots whose contents are device-allocator-zero
    //    by spec on the production tier-1 drivers (NVIDIA, AMD RDNA,
    //    Intel Arc — confirmed in `docs/vulkan_port_architecture.md`).
    {
        cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.prefix.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{prefix_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipes.prefix.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
        const push: PrefixPush = .{
            // walk_frame wrote n_chunks into the device meta buffer; we
            // can't read that here host-side before queue submit, so
            // pass the caller-supplied hint (right-sized to the actual
            // frame). The chunks buffer is pre-zeroed (vkCmdFillBuffer
            // above), so every slot past the real n_chunks reads
            // flags=0, decomp_size=0 — falls into the n_subs=1 branch.
            // total_subchunks is over-counted by exactly
            // (n_chunks_max - real_n_chunks) but that's harmless: the
            // staged-buffer slots past the real range stay valid=0
            // (scan_parse self-gates on real n_chunks from
            // n_chunks_scratch) and the compact kernels skip them.
            .n_chunks = n_chunks_max,
            .sub_chunk_cap = 0, // 0 → DEFAULT_SUB_CHUNK_CAP per the shader
        };
        var push_bytes: [@sizeOf(PrefixPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(
            ctx.cmd_buf,
            pipes.prefix.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_bytes.len),
            @ptrCast(&push_bytes),
        );
        beginLabel(ctx.cmd_buf, "prefix_sum_chunks");
        cmd_dispatch(ctx.cmd_buf, 1, 1, 1);
        endLabel(ctx.cmd_buf, "prefix_sum_chunks");
    }

    // BARRIER B — prefix_sum writes → scan_parse reads.
    try recordComputeBarrier(ctx.cmd_buf);

    // Copy meta[N_CHUNKS_SLOT] into n_chunks_scratch so scan_parse sees
    // the actual walk-frame-determined n_chunks (its self-gate compares
    // gl_GlobalInvocationID.x against *n_chunks_buf). The copy is
    // device-to-device, in-buffer; runs at full VRAM bandwidth.
    {
        const region: vk.VkBufferCopy = .{
            .srcOffset = WALK_META_N_CHUNKS_SLOT * 4,
            .dstOffset = 0,
            .size = 4,
        };
        const regions: [1]vk.VkBufferCopy = .{region};
        beginLabel(ctx.cmd_buf, "meta->n_chunks_scratch");
        cmd_copy(ctx.cmd_buf, result.meta.buf, result.n_chunks_scratch.buf, 1, @ptrCast(&regions));
        endLabel(ctx.cmd_buf, "meta->n_chunks_scratch");
    }

    // BARRIER B' — the in-buffer copy above is a TRANSFER_WRITE; scan
    // reads it as SHADER_READ. Need a transfer→compute barrier (the
    // compute-only barrier we use otherwise doesn't cover transfer).
    {
        const cmd_barrier_fn = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
        const mb: vk.VkMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT,
        };
        const mbs: [1]vk.VkMemoryBarrier = .{mb};
        cmd_barrier_fn(
            ctx.cmd_buf,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            @ptrCast(&mbs),
            0,
            null,
            0,
            null,
        );
    }

    // 3. scan_parse — n_chunks workgroups × 32 threads. Over-launch at
    //    WALK_MAX_CHUNKS / 32 workgroups; the kernel self-gates.
    {
        cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.scan.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{scan_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipes.scan.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
        const push: ScanPush = .{
            .block_len = @intCast(slz_bytes.len),
            .sub_chunk_cap = 0, // 0 → DEFAULT_SUB_CHUNK_CAP per the shader
        };
        var push_bytes: [@sizeOf(ScanPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(
            ctx.cmd_buf,
            pipes.scan.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_bytes.len),
            @ptrCast(&push_bytes),
        );
        // local_size_x=32; grid sized to cover n_chunks_max lanes. The
        // scan kernel self-gates per-lane on gl_GlobalInvocationID.x
        // against *n_chunks_buf (= the walk-frame-determined real
        // n_chunks), so over-launching is safe.
        const scan_grid_x: u32 = (n_chunks_max + 31) / 32;
        beginLabel(ctx.cmd_buf, "scan_parse");
        cmd_dispatch(ctx.cmd_buf, scan_grid_x, 1, 1);
        endLabel(ctx.cmd_buf, "scan_parse");
    }

    // BARRIER C — scan_parse writes → compact_* read.
    try recordComputeBarrier(ctx.cmd_buf);

    // 4a–4d. compact_huff_descs (lit, tok, hi, lo). All four launches
    //    can ride the same barrier (BARRIER C) because they write
    //    disjoint regions; no inter-compact-kernel barrier needed.
    const compact_huff_sets = [_]vk.VkDescriptorSet{
        compact_lit_set, compact_tok_set, compact_hi_set, compact_lo_set,
    };
    const compact_huff_labels = [_][]const u8{
        "compact_huff_lit", "compact_huff_tok", "compact_huff_hi", "compact_huff_lo",
    };
    for (compact_huff_sets, 0..) |cs, ci| {
        cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.compact_huff.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{cs};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipes.compact_huff.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
        beginLabel(ctx.cmd_buf, compact_huff_labels[ci]);
        cmd_dispatch(ctx.cmd_buf, 1, 1, 1);
        endLabel(ctx.cmd_buf, compact_huff_labels[ci]);
    }

    // 4e. compact_raw_descs.
    {
        cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.compact_raw.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{compact_raw_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipes.compact_raw.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
        beginLabel(ctx.cmd_buf, "compact_raw_descs");
        cmd_dispatch(ctx.cmd_buf, 1, 1, 1);
        endLabel(ctx.cmd_buf, "compact_raw_descs");
    }

    // BARRIER D — compact_* writes → gather_raw_off16 reads.
    try recordComputeBarrier(ctx.cmd_buf);

    // 5. gather_raw_off16 — total_subchunks*2 workgroups × 256 threads.
    //    The kernel self-gates on compact_counts[4] (n_raw), so an
    //    over-launch costs only an empty workgroup. We launch
    //    max_total_subchunks*2 workgroups.
    {
        cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.gather.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{gather_set};
        cmd_bind_ds(
            ctx.cmd_buf,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipes.gather.pipeline_layout,
            0,
            1,
            @ptrCast(&sets),
            0,
            null,
        );
        const push: GatherPush = .{
            .comp_len = @intCast(slz_bytes.len),
        };
        var push_bytes: [@sizeOf(GatherPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(
            ctx.cmd_buf,
            pipes.gather.pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_bytes.len),
            @ptrCast(&push_bytes),
        );
        // Worst-case grid — the kernel self-gates on the real n_raw
        // count. For L1 the count is 0 so every workgroup early-exits.
        //
        // Cap the launch at the worst-case `total_subchunks*2`. Even on
        // an L3 frame with the worst-case sub-chunk count this is
        // 16384 * 4 * 2 = 131072 workgroups — well below
        // VkPhysicalDeviceLimits.maxComputeWorkGroupCount[0] (2^31 on
        // every supported device).
        const gather_grid_x: u32 = max_total_subchunks * 2;
        beginLabel(ctx.cmd_buf, "gather_raw_off16");
        cmd_dispatch(ctx.cmd_buf, gather_grid_x, 1, 1);
        endLabel(ctx.cmd_buf, "gather_raw_off16");
    }

    // BARRIER E — gather writes → downstream LZ kernel reads. The
    // downstream kernel is in a separate cmdbuf today (Phase 3+ kernel
    // bodies aren't on the L1 path), but we end with the barrier so
    // future single-cmdbuf merges can see a consistent shape.
    try recordComputeBarrier(ctx.cmd_buf);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;

    // ── 5. Submit + wait ─────────────────────────────────────────────
    const fences: [1]vk.VkFence = .{ctx.fence};
    if (reset_fence(ctx.dev, 1, @ptrCast(&fences)) != vk.VK_SUCCESS) return error.ResetFenceFailed;

    const cmd_bufs: [1]vk.VkCommandBuffer = .{ctx.cmd_buf};
    const submit_info: vk.VkSubmitInfo = .{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cmd_bufs),
    };
    const submits: [1]vk.VkSubmitInfo = .{submit_info};
    if (submit(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) return error.SubmitFailed;

    const wait_result = wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS);
    if (wait_result == vk.VK_TIMEOUT) return error.FenceWaitTimeout;
    if (wait_result != vk.VK_SUCCESS) return error.FenceWaitFailed;

    // ── 6. Read host-visible Meta into the result struct ─────────────
    if (result.meta.mapped) |m| {
        const words: [*]const u32 = @ptrCast(@alignCast(m));
        result.n_chunks = words[WALK_META_N_CHUNKS_SLOT];
        result.decomp_size = words[WALK_META_DECOMP_SIZE_SLOT];
        result.sub_chunk_cap = words[WALK_META_SUB_CHUNK_CAP_SLOT];
        result.block_start = words[WALK_META_BLOCK_START_SLOT];
        result.block_size = words[WALK_META_BLOCK_SIZE_SLOT];
        result.status = words[WALK_META_STATUS_SLOT];
    } else {
        return error.MapMemoryFailed;
    }
    // total_subchunks: an upper bound is enough for downstream kernels
    // (they self-gate). The real value is in result.total_subs on
    // device, readable by future Phase-3 dispatches without H2D.
    result.total_subchunks = result.n_chunks * MAX_SUB_CHUNKS_PER_CHUNK;

    return result;
}
