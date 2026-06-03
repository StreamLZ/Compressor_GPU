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
const decode_workspace = @import("decode_workspace.zig");

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

// Use l1_codec.Buffer directly so DecodeResult fields can hold by-value
// copies of the per-context workspace's `Slot.buffer` field (which is
// also l1_codec.Buffer). The two types were already structurally
// identical pre-workspace; aliasing removes the duplication and lets
// the workspace own the lifetime.
const Buffer = l1_codec.Buffer;

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
    // Device buffers — VIEWS into the per-context DecodeWorkspace pool
    // (`driver.Context.decode_workspace`). The workspace owns the
    // VkBuffer + VkDeviceMemory lifetime across decode calls; this
    // struct's fields are by-value Buffer copies populated by
    // runDecodePipelineEx from `ws.<slot>.buffer`. `destroyDecodeResult`
    // is therefore a no-op — the workspace frees everything once at
    // `driver.deinit`. Mirrors CUDA's pattern where the dispatch reads
    // `self.d_<slot>` straight off the long-lived DecodeContext
    // (src/decode/decode_dispatch.zig:411, 519, 549).
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

/// No-op since the workspace pool was introduced: every Buffer field
/// in DecodeResult is a view into a workspace slot whose lifetime is
/// owned by `driver.Context.decode_workspace`. Kept around so existing
/// callers (slz1_codec.decodeSlz1ToBytesEx via its `defer
/// destroyDecodeResult` line) compile unchanged and so the documented
/// API contract doesn't break for future re-use cases.
///
/// Mirrors CUDA's per-call pattern in `src/decode/decode_dispatch.zig`
/// which never frees the per-decode workspace slots — they're owned by
/// `DecodeContext` for the lifetime of the context. See
/// `src/decode/decode_context.zig:320` for the once-per-context teardown.
pub fn destroyDecodeResult(ctx: *driver.Context, r: *DecodeResult) void {
    _ = ctx;
    _ = r;
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
    // S002: pass the process-lifetime VkPipelineCache so the driver
    // can dedupe SPV→ISA work across the 6 pipelines built here
    // (and the 2 more built by slz1_codec — they share the cache).
    const vk_pl_cache = driver.getOrCreateVkPipelineCache(ctx);

    const walk_spv_raw = spv_blobs.find("walk_frame", tier_b) orelse return error.NoSpvForTier;
    const walk_spv = try dupAlignedSpv(allocator, walk_spv_raw);
    defer allocator.free(walk_spv);
    const walk = try descriptors.getOrCreateWithPipelineCache(ctx, cache, "walk_frame", tier, walk_spv, 3, @sizeOf(WalkPush), vk_pl_cache);

    const prefix_spv_raw = spv_blobs.find("prefix_sum_chunks", tier_b) orelse return error.NoSpvForTier;
    const prefix_spv = try dupAlignedSpv(allocator, prefix_spv_raw);
    defer allocator.free(prefix_spv);
    const prefix = try descriptors.getOrCreateWithPipelineCache(ctx, cache, "prefix_sum_chunks", tier, prefix_spv, 3, @sizeOf(PrefixPush), vk_pl_cache);

    const scan_spv_raw = spv_blobs.find("scan_parse", tier_b) orelse return error.NoSpvForTier;
    const scan_spv = try dupAlignedSpv(allocator, scan_spv_raw);
    defer allocator.free(scan_spv);
    const scan = try descriptors.getOrCreateWithPipelineCache(ctx, cache, "scan_parse", tier, scan_spv, 10, @sizeOf(ScanPush), vk_pl_cache);

    const compact_huff_spv_raw = spv_blobs.find("compact_huff_descs", tier_b) orelse return error.NoSpvForTier;
    const compact_huff_spv = try dupAlignedSpv(allocator, compact_huff_spv_raw);
    defer allocator.free(compact_huff_spv);
    const compact_huff = try descriptors.getOrCreateWithPipelineCache(ctx, cache, "compact_huff_descs", tier, compact_huff_spv, 4, 0, vk_pl_cache);

    const compact_raw_spv_raw = spv_blobs.find("compact_raw_descs", tier_b) orelse return error.NoSpvForTier;
    const compact_raw_spv = try dupAlignedSpv(allocator, compact_raw_spv_raw);
    defer allocator.free(compact_raw_spv);
    const compact_raw = try descriptors.getOrCreateWithPipelineCache(ctx, cache, "compact_raw_descs", tier, compact_raw_spv, 5, 0, vk_pl_cache);

    const gather_spv_raw = spv_blobs.find("gather_raw_off16", tier_b) orelse return error.NoSpvForTier;
    const gather_spv = try dupAlignedSpv(allocator, gather_spv_raw);
    defer allocator.free(gather_spv);
    const gather = try descriptors.getOrCreateWithPipelineCache(ctx, cache, "gather_raw_off16", tier, gather_spv, 4, @sizeOf(GatherPush), vk_pl_cache);

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

/// S003: prepared state shared between buffer allocation + recording.
/// Holds every workspace-resident buffer view, every descriptor set, the
/// loaded pipeline cache entries, and the per-call sizing derived from
/// `slz_bytes.len`. `prepareDecodePipeline` populates this; either
/// `runDecodePipelineEx` (legacy stand-alone driver) or `recordDecodePipelineInto`
/// (S003 single-submit caller in slz1_codec.zig) consumes it. The split
/// lets the new caller record the GPU decode pipeline + l1_unwrap +
/// lz_decode + dst→host copy into one cmdbuf with one vkQueueSubmit
/// instead of the old two-submit pattern.
pub const PreparedDecodePipeline = struct {
    result: DecodeResult,
    pipes: Pipelines,
    walk_set: vk.VkDescriptorSet,
    prefix_set: vk.VkDescriptorSet,
    scan_set: vk.VkDescriptorSet,
    compact_lit_set: vk.VkDescriptorSet,
    compact_tok_set: vk.VkDescriptorSet,
    compact_hi_set: vk.VkDescriptorSet,
    compact_lo_set: vk.VkDescriptorSet,
    compact_raw_set: vk.VkDescriptorSet,
    gather_set: vk.VkDescriptorSet,
    // Sizing derived from slz_bytes.len + hint — needed at record time
    // for grid sizing, vkCmdFillBuffer ranges, and descriptor offsets.
    n_chunks_max: u32,
    max_total_subchunks: u32,
    huff_arr_bytes: vk.VkDeviceSize,
    raw_arr_bytes: vk.VkDeviceSize,
    staged_bytes: vk.VkDeviceSize,
    chunks_bytes: vk.VkDeviceSize,
    first_sub_bytes: vk.VkDeviceSize,
    meta_bytes: vk.VkDeviceSize,
    frame_bytes: vk.VkDeviceSize,
    slz_len_u32: u32,
};

/// S003: allocate every workspace slot this pipeline needs, build/cache
/// the 6 pipelines + 9 descriptor sets, and stage the compressed input
/// into the host-visible `frame_staging` slot. Does NOT record any
/// commands and does NOT submit; the caller invokes
/// `recordDecodePipelineInto` against an already-begun command buffer.
///
/// This is the post-S003 entry point that lets slz1_codec.zig record the
/// pipeline phases + l1_unwrap + lz_decode + dst→host copy into a single
/// command buffer with one vkQueueSubmit, replacing the legacy
/// `runDecodePipelineEx` (which still exists for any caller that wants
/// the stand-alone submit-and-wait shape).
pub fn prepareDecodePipeline(
    ctx: *driver.Context,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    hints: PipelineHints,
) PipelineError!PreparedDecodePipeline {
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

    const slz_chunk_size_bytes: u32 = SLZ_CHUNK_SIZE_BYTES;
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
    const ws = driver.getOrCreateDecodeWorkspace(ctx) catch return error.OutOfMemory;

    const frame_bytes: vk.VkDeviceSize = @intCast((slz_bytes.len + 3) & ~@as(usize, 3));
    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.frame,
        @max(frame_bytes, 4),
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .device_local_only,
    );
    result.frame = ws.frame.buffer;
    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.frame_staging,
        @max(frame_bytes, 4),
        vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .host_visible_sysmem,
    );

    const chunks_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, WALK_MAX_CHUNKS) * CHUNK_DESC_U32_COUNT * 4;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.chunks, chunks_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.chunks = ws.chunks.buffer;

    const meta_bytes: vk.VkDeviceSize = WALK_META_U32_COUNT * 4;
    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.meta,
        meta_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .host_visible_sysmem,
    );
    result.meta = ws.meta.buffer;

    const first_sub_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, WALK_MAX_CHUNKS) * 4;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.first_sub_idx, first_sub_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.first_sub_idx = ws.first_sub_idx.buffer;

    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.total_subs, 4, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.total_subs = ws.total_subs.buffer;

    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.n_chunks_scratch,
        4,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .device_local_only,
    );
    result.n_chunks_scratch = ws.n_chunks_scratch.buffer;

    const huff_arr_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * SCANHUFF_U32_COUNT * 4;
    const raw_arr_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * SCANRAW_U32_COUNT * 4;
    const staged_bytes: vk.VkDeviceSize = huff_arr_bytes * 4 + raw_arr_bytes * 2;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.staged, staged_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.staged = ws.staged.buffer;

    const huff_compact_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * HUFFDESC_U32_COUNT * 4;
    const raw_compact_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks * 2) * RAWDESC_U32_COUNT * 4;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.compact_lit, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_lit = ws.compact_lit.buffer;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.compact_tok, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_tok = ws.compact_tok.buffer;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.compact_hi, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_hi = ws.compact_hi.buffer;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.compact_lo, huff_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_lo = ws.compact_lo.buffer;
    try decode_workspace.ensureWorkspaceBuf(ctx, &ws.compact_raw, raw_compact_bytes, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .device_local_only);
    result.compact_raw = ws.compact_raw.buffer;

    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.compact_counts,
        6 * 4,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .device_local_only,
    );
    result.compact_counts = ws.compact_counts.buffer;

    const off16_scratch_bytes: vk.VkDeviceSize = @as(vk.VkDeviceSize, max_total_subchunks) * ENTROPY_SCRATCH_SLOT_BYTES;
    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.off16_scratch,
        off16_scratch_bytes,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .device_local_only,
    );
    result.off16_scratch = ws.off16_scratch.buffer;

    if (ws.frame_staging.buffer.mapped) |m| {
        @memcpy(m[0..slz_bytes.len], slz_bytes);
        if (frame_bytes > slz_bytes.len) {
            @memset(m[slz_bytes.len..@intCast(frame_bytes)], 0);
        }
    } else {
        return error.MapMemoryFailed;
    }

    if (result.meta.mapped) |m| @memset(m[0..@intCast(meta_bytes)], 0);

    const cache = driver.getOrCreateDecodePipelineCache(ctx);
    const pipes = try loadAllPipelines(ctx, cache, tier, tier_b, allocator);

    const walk_bindings: [3]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.meta.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const walk_set = try descriptors.allocSet(ctx, pipes.walk, walk_bindings[0..]);

    const prefix_bindings: [3]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.first_sub_idx.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const prefix_set = try descriptors.allocSet(ctx, pipes.prefix, prefix_bindings[0..]);

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

    const compact_raw_bindings: [5]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 4, .range = raw_arr_bytes },
        .{ .buffer = result.staged.buf, .offset = huff_arr_bytes * 4 + raw_arr_bytes, .range = raw_arr_bytes },
        .{ .buffer = result.total_subs.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_raw.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 16, .range = 4 },
    };
    const compact_raw_set = try descriptors.allocSet(ctx, pipes.compact_raw, compact_raw_bindings[0..]);

    const gather_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.off16_scratch.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_raw.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = result.compact_counts.buf, .offset = 16, .range = 4 },
    };
    const gather_set = try descriptors.allocSet(ctx, pipes.gather, gather_bindings[0..]);

    return .{
        .result = result,
        .pipes = pipes,
        .walk_set = walk_set,
        .prefix_set = prefix_set,
        .scan_set = scan_set,
        .compact_lit_set = compact_lit_set,
        .compact_tok_set = compact_tok_set,
        .compact_hi_set = compact_hi_set,
        .compact_lo_set = compact_lo_set,
        .compact_raw_set = compact_raw_set,
        .gather_set = gather_set,
        .n_chunks_max = n_chunks_max,
        .max_total_subchunks = max_total_subchunks,
        .huff_arr_bytes = huff_arr_bytes,
        .raw_arr_bytes = raw_arr_bytes,
        .staged_bytes = staged_bytes,
        .chunks_bytes = chunks_bytes,
        .first_sub_bytes = first_sub_bytes,
        .meta_bytes = meta_bytes,
        .frame_bytes = frame_bytes,
        .slz_len_u32 = @intCast(slz_bytes.len),
    };
}

/// S003: record the 5-kernel GPU decode pipeline (frame_staging→frame
/// DMA + walk_frame + prefix_sum_chunks + scan_parse + 4× compact_huff +
/// compact_raw + gather_raw_off16) plus the inter-dispatch barriers
/// into a CALLER's already-begun command buffer. Does NOT begin/end the
/// command buffer, does NOT submit, does NOT wait — the caller stitches
/// in additional dispatches (l1_unwrap + lz_decode + dst→host copy) and
/// drives a single vkQueueSubmit + vkWaitForFences at the end.
///
/// Mirrors CUDA's pattern at src/decode/decode_dispatch.zig:614-621:
/// every back-half kernel reads its self-gate count from the device-
/// resident meta buffer (`d_n_chunks_dev = d_walk_meta + walk_meta_offsets
/// .n_chunks`), so no host fence wait is needed between phases.
///
/// Terminates with BARRIER E (SHADER_WRITE → SHADER_READ at COMPUTE) so
/// the caller's subsequent dispatch reads from `result.meta`/`result.chunks`
/// see the GPU pipeline's writes.
pub fn recordDecodePipelineInto(
    ctx: *driver.Context,
    cmd_buf: vk.VkCommandBuffer,
    prepared: PreparedDecodePipeline,
) PipelineError!void {
    const cmd_bind_pl = vk.vkCmdBindPipeline_fn orelse return error.LoaderNotReady;
    const cmd_bind_ds = vk.vkCmdBindDescriptorSets_fn orelse return error.LoaderNotReady;
    const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
    const cmd_dispatch = vk.vkCmdDispatch_fn orelse return error.LoaderNotReady;
    const cmd_copy = vk.vkCmdCopyBuffer_fn orelse return error.LoaderNotReady;

    const pipes = prepared.pipes;
    const result = prepared.result;
    const n_chunks_max = prepared.n_chunks_max;
    const max_total_subchunks = prepared.max_total_subchunks;

    // S007: DMA-copy compressed input from sysmem staging → device-local
    // frame via the GPU copy engine. The staging buffer lives on the
    // workspace alongside the device-local frame buffer (= result.frame).
    const ws = driver.getOrCreateDecodeWorkspace(ctx) catch return error.OutOfMemory;
    {
        const region: vk.VkBufferCopy = .{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = prepared.frame_bytes,
        };
        const regions: [1]vk.VkBufferCopy = .{region};
        beginLabel(cmd_buf, "frame_staging->frame (S007)");
        cmd_copy(cmd_buf, ws.frame_staging.buffer.buf, result.frame.buf, 1, @ptrCast(&regions));
        endLabel(cmd_buf, "frame_staging->frame (S007)");
    }

    if (vk.vkCmdFillBuffer_fn) |cmd_fill| {
        cmd_fill(cmd_buf, result.chunks.buf, 0, prepared.chunks_bytes, 0);
        cmd_fill(cmd_buf, result.total_subs.buf, 0, 4, 0);
        cmd_fill(cmd_buf, result.first_sub_idx.buf, 0, prepared.first_sub_bytes, 0);
        cmd_fill(cmd_buf, result.compact_counts.buf, 0, 6 * 4, 0);
        cmd_fill(cmd_buf, result.staged.buf, 0, prepared.staged_bytes, 0);
    } else {
        return error.LoaderNotReady;
    }

    {
        const cmd_barrier_fn = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
        const mb: vk.VkMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        };
        const mbs: [1]vk.VkMemoryBarrier = .{mb};
        cmd_barrier_fn(
            cmd_buf,
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
        cmd_bind_pl(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.walk.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{prepared.walk_set};
        cmd_bind_ds(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.walk.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
        const push: WalkPush = .{ .frame_size = prepared.slz_len_u32, .max_chunks = WALK_MAX_CHUNKS };
        var push_bytes: [@sizeOf(WalkPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(cmd_buf, pipes.walk.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_bytes.len), @ptrCast(&push_bytes));
        beginLabel(cmd_buf, "walk_frame");
        cmd_dispatch(cmd_buf, 1, 1, 1);
        endLabel(cmd_buf, "walk_frame");
    }
    try recordComputeBarrier(cmd_buf);

    // 2. prefix_sum_chunks.
    {
        cmd_bind_pl(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.prefix.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{prepared.prefix_set};
        cmd_bind_ds(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.prefix.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
        const push: PrefixPush = .{ .n_chunks = n_chunks_max, .sub_chunk_cap = 0 };
        var push_bytes: [@sizeOf(PrefixPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(cmd_buf, pipes.prefix.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_bytes.len), @ptrCast(&push_bytes));
        beginLabel(cmd_buf, "prefix_sum_chunks");
        cmd_dispatch(cmd_buf, 1, 1, 1);
        endLabel(cmd_buf, "prefix_sum_chunks");
    }
    try recordComputeBarrier(cmd_buf);

    {
        const region: vk.VkBufferCopy = .{
            .srcOffset = WALK_META_N_CHUNKS_SLOT * 4,
            .dstOffset = 0,
            .size = 4,
        };
        const regions: [1]vk.VkBufferCopy = .{region};
        beginLabel(cmd_buf, "meta->n_chunks_scratch");
        cmd_copy(cmd_buf, result.meta.buf, result.n_chunks_scratch.buf, 1, @ptrCast(&regions));
        endLabel(cmd_buf, "meta->n_chunks_scratch");
    }

    {
        const cmd_barrier_fn = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
        const mb: vk.VkMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT,
        };
        const mbs: [1]vk.VkMemoryBarrier = .{mb};
        cmd_barrier_fn(
            cmd_buf,
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

    // 3. scan_parse.
    {
        cmd_bind_pl(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.scan.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{prepared.scan_set};
        cmd_bind_ds(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.scan.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
        const push: ScanPush = .{ .block_len = prepared.slz_len_u32, .sub_chunk_cap = 0 };
        var push_bytes: [@sizeOf(ScanPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(cmd_buf, pipes.scan.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_bytes.len), @ptrCast(&push_bytes));
        const scan_grid_x: u32 = (n_chunks_max + 31) / 32;
        beginLabel(cmd_buf, "scan_parse");
        cmd_dispatch(cmd_buf, scan_grid_x, 1, 1);
        endLabel(cmd_buf, "scan_parse");
    }
    try recordComputeBarrier(cmd_buf);

    // 4a-4d. compact_huff_descs.
    const compact_huff_sets = [_]vk.VkDescriptorSet{
        prepared.compact_lit_set, prepared.compact_tok_set, prepared.compact_hi_set, prepared.compact_lo_set,
    };
    const compact_huff_labels = [_][]const u8{
        "compact_huff_lit", "compact_huff_tok", "compact_huff_hi", "compact_huff_lo",
    };
    for (compact_huff_sets, 0..) |cs, ci| {
        cmd_bind_pl(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.compact_huff.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{cs};
        cmd_bind_ds(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.compact_huff.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
        beginLabel(cmd_buf, compact_huff_labels[ci]);
        cmd_dispatch(cmd_buf, 1, 1, 1);
        endLabel(cmd_buf, compact_huff_labels[ci]);
    }

    // 4e. compact_raw_descs.
    {
        cmd_bind_pl(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.compact_raw.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{prepared.compact_raw_set};
        cmd_bind_ds(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.compact_raw.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
        beginLabel(cmd_buf, "compact_raw_descs");
        cmd_dispatch(cmd_buf, 1, 1, 1);
        endLabel(cmd_buf, "compact_raw_descs");
    }
    try recordComputeBarrier(cmd_buf);

    // 5. gather_raw_off16.
    {
        cmd_bind_pl(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.gather.pipeline);
        const sets: [1]vk.VkDescriptorSet = .{prepared.gather_set};
        cmd_bind_ds(cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipes.gather.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
        const push: GatherPush = .{ .comp_len = prepared.slz_len_u32 };
        var push_bytes: [@sizeOf(GatherPush)]u8 = undefined;
        @memcpy(push_bytes[0..], std.mem.asBytes(&push));
        cmd_push(cmd_buf, pipes.gather.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(push_bytes.len), @ptrCast(&push_bytes));
        const gather_grid_x: u32 = max_total_subchunks * 2;
        beginLabel(cmd_buf, "gather_raw_off16");
        cmd_dispatch(cmd_buf, gather_grid_x, 1, 1);
        endLabel(cmd_buf, "gather_raw_off16");
    }
    try recordComputeBarrier(cmd_buf);
}

/// Legacy stand-alone driver kept for callers that want the pre-S003
/// shape (own command buffer, own submit, own fence wait, meta read on
/// return). The new in-process callers in slz1_codec.zig drive
/// `prepareDecodePipeline` + `recordDecodePipelineInto` directly so the
/// GPU pipeline + l1_unwrap + lz_decode + dst→host copy ride one
/// vkQueueSubmit. This implementation now delegates to the same two
/// helpers to keep one source of truth for the dispatch shape.
pub fn runDecodePipelineEx(
    ctx: *driver.Context,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    hints: PipelineHints,
) PipelineError!DecodeResult {
    var prepared = try prepareDecodePipeline(ctx, allocator, slz_bytes, hints);

    try dispatch.ensureChassisPub(ctx);

    const reset_cb = vk.vkResetCommandBuffer_fn orelse return error.LoaderNotReady;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return error.LoaderNotReady;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return error.LoaderNotReady;
    const reset_fence = vk.vkResetFences_fn orelse return error.LoaderNotReady;
    const submit = vk.vkQueueSubmit_fn orelse return error.LoaderNotReady;
    const wait_fence = vk.vkWaitForFences_fn orelse return error.LoaderNotReady;

    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return error.ResetCommandBufferFailed;

    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) return error.BeginCommandBufferFailed;

    try recordDecodePipelineInto(ctx, ctx.cmd_buf, prepared);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;

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

    if (prepared.result.meta.mapped) |m| {
        const words: [*]const u32 = @ptrCast(@alignCast(m));
        prepared.result.n_chunks = words[WALK_META_N_CHUNKS_SLOT];
        prepared.result.decomp_size = words[WALK_META_DECOMP_SIZE_SLOT];
        prepared.result.sub_chunk_cap = words[WALK_META_SUB_CHUNK_CAP_SLOT];
        prepared.result.block_start = words[WALK_META_BLOCK_START_SLOT];
        prepared.result.block_size = words[WALK_META_BLOCK_SIZE_SLOT];
        prepared.result.status = words[WALK_META_STATUS_SLOT];
    } else {
        return error.MapMemoryFailed;
    }
    prepared.result.total_subchunks = prepared.result.n_chunks * MAX_SUB_CHUNKS_PER_CHUNK;

    return prepared.result;
}
