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
const builtin = @import("builtin");

const vk = @import("vk_api.zig");
const driver = @import("driver.zig");
const probe_mod = @import("probe.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");

// ── QPC helpers (Windows-only; degrade to monotonic on other OSes) ──
// Used by the diagnostic decode-phase timing to attribute host-overhead
// ns to specific decode steps for TODO-P1 diagnosis.
const win32_qpc = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};
fn qpcNow() i64 {
    if (builtin.os.tag != .windows) return std.time.nanoTimestamp();
    var c: i64 = 0;
    _ = win32_qpc.QueryPerformanceCounter(&c);
    return c;
}
fn qpcNs(from: i64, to: i64) u64 {
    if (builtin.os.tag != .windows) return @intCast(@max(0, to - from));
    var freq: i64 = 0;
    _ = win32_qpc.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

// ── Constants ─────────────────────────────────────────────────────

/// Per-chunk hash-table size for L1 (hash_bits = 17 → 1<<17 entries × u32
/// = 512 KB). Spec §1.1.
pub const HASH_BITS: u32 = 17;
pub const HASH_SIZE_BYTES: vk.VkDeviceSize = (@as(vk.VkDeviceSize, 1) << HASH_BITS) * @sizeOf(u32);

/// Per-chunk input slice size. Matches `src/format/streamlz_constants.zig`
/// sub_chunk_size = 0x10000 (64 KiB) at sc_group_size = 0.25 — the same
/// sub-chunk size the CUDA encoder defaults to. Each shader workgroup
/// processes one chunk; the encoder uses the 2-block scanBlock interior
/// (64 KiB blocks), so a 64 KiB chunk = 1 block per workgroup.
pub const CHUNK_SIZE: u32 = 0x10000;

/// Number of source bytes written verbatim to the head of chunk 0's
/// lit_buf and skipped in its token stream. Mirrors CUDA's
/// `INITIAL_LITERAL_COPY_BYTES` from `src/common/gpu_wire_format.cuh`
/// (matches the "anchor = 8 when is_first" rule in lz_kernel.cu and the
/// `base_offset == 0` prefix copy in lz_dispatch.cuh).
pub const INITIAL_LITERAL_COPY_BYTES: u32 = 8;

/// Per-chunk reservation in each output stream. Worst-case stream size
/// per spec §4.1 is `2 * src_size`; we add 16 bytes of slack so lane-0
/// RMW byte stores into the tail u32 word never write past the slice.
/// Must be a multiple of 4 (shader byte-load helpers assume word-aligned
/// per-chunk bases — see lz_{encode,decode}.comp head-of-file comment).
pub const CHUNK_STREAM_CAPACITY: u32 = (CHUNK_SIZE * 2) + 16;

/// Per-chunk reservation for the off32 stream. Bounded by
/// `2 * src_size` (every byte slot in a chunk could in the worst case
/// drive a 3-byte off32 entry; the *2 keeps the headroom symmetric with
/// the other streams), and we pad to a multiple of 4. With CHUNK_SIZE =
/// 128 KiB this works out to ~512 KiB per chunk — generous, but the
/// decoder/encoder allocate on demand only when a chunk actually emits
/// far-offset matches.
pub const CHUNK_OFF32_CAPACITY: u32 = (CHUNK_SIZE * 2) + 16;

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

/// Per-process stash for the most recent encode / decode GPU-side
/// dispatch nanoseconds (as returned by `dispatch.submitOne` from the
/// VkQueryPool TIMESTAMP pair). Updated by encodeL1Multi / decodeL1Sync
/// right after their submitOne calls. Diagnostic-only — production
/// callers should plumb a proper out-param when needed (TODO P1's
/// `slzGetLastTimings_vk` is the spec'd path for that). Lives at module
/// scope so the perf bench can read it without changing the public API.
pub var last_encode_dispatch_ns: u64 = 0;
pub var last_decode_dispatch_ns: u64 = 0;

/// Per-process fine-grained decode-phase wall ns (QPC). Populated by
/// decodeL1Sync to split where the host overhead actually goes — at
/// minimum: buffer create+map+zero, descriptor build, dispatch (subsumes
/// GPU + fence wait), final readback @memcpy. Diagnostic-only; used by
/// l1_perf_bench to render the breakdown table for TODO P1.
pub var last_decode_dst_alloc_ns: u64 = 0;
pub var last_decode_dst_memset_ns: u64 = 0;
pub var last_decode_descriptors_ns: u64 = 0;
pub var last_decode_dispatch_wall_ns: u64 = 0;
pub var last_decode_readback_ns: u64 = 0;

/// Hard cap on chunks per encode. Bounds the per-chunk size sidecar
/// arrays so we never need heap allocation in the codec module. 4096
/// chunks × 128 KiB = 512 MiB max input — covers the silesia 200 MB
/// scale test plus headroom for ~2x growth without re-tuning. Bumped
/// from the original 256 (= 32 MiB cap) when the L1 scale test
/// required full-corpus runs on enwik8 (100 MB) + silesia (200 MB).
///
/// Per-chunk sidecar growth from this bump: L1Streams gains
/// (4096-256) * 4 bytes * 8 arrays = ~120 KiB, total ~128 KiB struct.
/// L1Streams is passed by value in a few host paths — the Windows
/// 4 MiB default stack absorbs it without trouble.
pub const MAX_CHUNKS: u32 = 4096;

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

    off32_buf: vk.VkBuffer = null,
    off32_mem: vk.VkDeviceMemory = null,
    /// Sum across all chunks of (off32_count_block1 + off32_count_block2)
    /// — used for the round-trip ratio report and for size accounting
    /// on the host side. Each entry is `OFF32_ENTRY_BYTES = 3` bytes.
    off32_total: u32 = 0,

    /// Multi-chunk geometry.
    n_chunks: u32 = 0,
    chunk_capacity: u32 = CHUNK_STREAM_CAPACITY,
    off32_capacity: u32 = CHUNK_OFF32_CAPACITY,
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
    /// `INITIAL_LITERAL_COPY_BYTES` (8) for chunk 0, 0 otherwise. Matches
    /// the CUDA `CompressChunkDesc::is_first` flag — when non-zero, the
    /// encoder skipped the first 8 source bytes in its token output and
    /// wrote them as the first 8 literals of this chunk's lit_buf; the
    /// decoder reproduces them directly to dst[0..8].
    per_chunk_initial_copy: [MAX_CHUNKS]u32 = @splat(0),

    /// Per-chunk off32 entry counts (block-0 + block-1). The sub-chunk
    /// payload's off32 block carries `count1` entries followed by
    /// `count2` entries, each `OFF32_ENTRY_BYTES = 3` bytes.
    per_chunk_off32_count1: [MAX_CHUNKS]u32 = @splat(0),
    per_chunk_off32_count2: [MAX_CHUNKS]u32 = @splat(0),
    /// Per-chunk cmd_stream2_offset — token-index boundary between
    /// blocks 0 and 1 in the cmd stream. 0 means "no split" (chunk
    /// fits in one LZ block). Used by the decoder to switch the active
    /// off32 cursor at the right token boundary.
    per_chunk_cmd_stream2_offset: [MAX_CHUNKS]u32 = @splat(0),

    /// Total bytes worth of compressed payload (lit + cmd + 2*off16 +
    /// 3*off32 + length) summed across all chunks. Convenience for the
    /// round-trip test's compression-ratio report.
    pub fn compressedTotalBytes(self: L1Streams) u32 {
        return self.lit_size + self.cmd_size + (self.off16_count * 2) +
            (self.off32_total * 3) + self.length_used;
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

pub fn ensureBufferFnSlots(ctx: *driver.Context) void {
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

/// Find a memory type that is HOST_VISIBLE+HOST_COHERENT and EXPLICITLY
/// NOT DEVICE_LOCAL. On discrete GPUs with resizable BAR, the rebar
/// region is reported as both DEVICE_LOCAL and HOST_VISIBLE — and the
/// CPU reads from it are uncached PCIe BAR reads (very slow). Plain
/// sysmem (HOST_VISIBLE only) is driver-cached and reads at full memory
/// bandwidth. Returns null on integrated GPUs where every host-visible
/// heap is also device-local — callers fall back to the regular
/// HOST_VISIBLE search.
fn findHostVisibleNonDeviceLocal(
    pd: vk.VkPhysicalDevice,
    type_bits_mask: u32,
) ?u32 {
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return null;
    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(pd, &mem_props);

    const want = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (type_bits_mask & (@as(u32, 1) << @intCast(i))) != 0;
        if (!supported) continue;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        const has_want = (flags & want) == want;
        const is_device_local = (flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        if (has_want and !is_device_local) return i;
    }
    return null;
}

const Buffer = struct {
    buf: vk.VkBuffer = null,
    mem: vk.VkDeviceMemory = null,
    mapped: ?[*]u8 = null,
    size: vk.VkDeviceSize = 0,
};

/// Memory-placement hint passed to `createBuffer`. Names describe the
/// CPU↔GPU access pattern the caller actually wants.
pub const MemMode = enum {
    /// Prefer DEVICE_LOCAL+HOST_VISIBLE (rebar/BAR); fall back to plain
    /// HOST_VISIBLE+HOST_COHERENT (sysmem). Best for buffers the host
    /// fills once and the GPU reads many times (encoder input streams,
    /// encoder output streams the host reads in the no-rebar fallback).
    host_visible_prefer_device_local,
    /// DEVICE_LOCAL only — no host mapping. Best for buffers the GPU
    /// writes/reads hot (decoder output, hash table). On discrete GPUs
    /// this lives in VRAM and gets full bandwidth.
    device_local_only,
    /// Plain HOST_VISIBLE+HOST_COHERENT — explicitly NOT DEVICE_LOCAL.
    /// Best for staging buffers the host reads frequently after the
    /// GPU writes them (decoder output staging). On a discrete GPU
    /// with resizable BAR, the "ideal" host-visible+device-local
    /// region is actually slow to read from the CPU (uncached PCIe
    /// BAR reads → 20–60 MB/s on RTX 4060 Ti); plain sysmem is
    /// driver-cached and the CPU reads at full memory bandwidth.
    host_visible_sysmem,
};

fn createBuffer(
    ctx: *driver.Context,
    size: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
    host_visible: bool,
) L1Error!Buffer {
    const mode: MemMode = if (host_visible) .host_visible_prefer_device_local else .device_local_only;
    return createBufferEx(ctx, size, usage, mode);
}

fn createBufferEx(
    ctx: *driver.Context,
    size: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
    mode: MemMode,
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
        switch (mode) {
            .host_visible_prefer_device_local => {
                const ideal = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                const fallback = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
                if (findMemoryType(ctx.pd, req.memoryTypeBits, ideal)) |i| break :blk i;
                if (findMemoryType(ctx.pd, req.memoryTypeBits, fallback)) |i| break :blk i;
            },
            .device_local_only => {
                if (findMemoryType(ctx.pd, req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) |i| break :blk i;
            },
            .host_visible_sysmem => {
                // Walk memory types and pick the first HOST_VISIBLE+
                // HOST_COHERENT type that is NOT also DEVICE_LOCAL.
                // On discrete GPUs that's the sysmem heap; on an iGPU
                // every host-visible heap is also device-local, so we
                // fall back to the rebar-style ideal.
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
    const wants_mapping = (mode != .device_local_only);
    if (wants_mapping) {
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
    const off32_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * CHUNK_OFF32_CAPACITY;
    var off32_b = try createBuffer(ctx, off32_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &off32_b);

    // Hash buffer: n_chunks × HASH_SIZE_BYTES. Device-local only — the
    // host never reads or writes it.
    var hash_b = try createBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * HASH_SIZE_BYTES, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
    defer destroyBuffer(ctx, &hash_b);

    // Sizes sidecar: 8 u32 per chunk:
    //   [0] lit_count   [1] cmd_count
    //   [2] off16_count [3] length_count
    //   [4] off32_count_block1   [5] off32_count_block2
    //   [6] cmd_stream2_offset   [7] reserved
    var sizes_b = try createBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 8,
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
    @memset(off32_b.mapped.?[0..@intCast(off32_b.size)], 0);
    @memset(sizes_b.mapped.?[0..@intCast(sizes_b.size)], 0);

    // Fill chunk descriptors: each chunk gets [src_offset, src_size, is_first].
    // is_first=1 for chunk 0 makes the encoder treat the first 8 source
    // bytes as a direct lit_buf copy and skip them in the token parser
    // (matches CUDA's INITIAL_LITERAL_COPY_BYTES handling). Subsequent
    // chunks always have is_first=0 (no per-chunk initial prefix).
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
            const is_first: u32 = if (ci == 0 and this_size >= INITIAL_LITERAL_COPY_BYTES) 1 else 0;
            chunks_words[ci * 3 + 0] = off;
            chunks_words[ci * 3 + 1] = this_size;
            chunks_words[ci * 3 + 2] = is_first;
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
        9, // bindings 0..8: src, lit, cmd, off16, length, hash, sizes, chunks, off32
        @sizeOf(EncodePush),
    );

    const enc_bindings: [9]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = src_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = lit_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = cmd_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off16_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = length_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = hash_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = sizes_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off32_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
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

    const enc_dispatch_result = try dispatch.submitOne(
        ctx,
        cached_enc.pipeline,
        cached_enc.pipeline_layout,
        enc_set,
        enc_push_bytes[0..],
        .{ n_chunks, 1, 1 },
    );
    last_encode_dispatch_ns = enc_dispatch_result.ns;

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
    var off32_total_count: u32 = 0;
    {
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const base = ci * 8;
            const ls = sizes_words[base + 0];
            const cs = sizes_words[base + 1];
            const os = sizes_words[base + 2];
            const xs = sizes_words[base + 3];
            const o32c1 = sizes_words[base + 4];
            const o32c2 = sizes_words[base + 5];
            const cs2o = sizes_words[base + 6];
            streams.per_chunk_lit_size[ci] = ls;
            streams.per_chunk_cmd_size[ci] = cs;
            streams.per_chunk_off16_count[ci] = os;
            streams.per_chunk_length_used[ci] = xs;
            streams.per_chunk_off32_count1[ci] = o32c1;
            streams.per_chunk_off32_count2[ci] = o32c2;
            streams.per_chunk_cmd_stream2_offset[ci] = cs2o;
            // is_first reflects exactly the chunk-descriptor value we
            // wrote pre-dispatch. Carrying it on the streams bundle lets
            // the wire-format wrapper and the decoder agree on which
            // chunks start with an 8-byte verbatim prefix in lit_buf.
            const init_copy: u32 = if (ci == 0 and ls >= INITIAL_LITERAL_COPY_BYTES)
                INITIAL_LITERAL_COPY_BYTES
            else
                0;
            streams.per_chunk_initial_copy[ci] = init_copy;
            lit_total += ls;
            cmd_total += cs;
            off16_total += os;
            length_total += xs;
            off32_total_count += o32c1 + o32c2;
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
    streams.off32_buf = off32_b.buf;
    streams.off32_mem = off32_b.mem;
    streams.off32_total = off32_total_count;
    streams.off32_capacity = CHUNK_OFF32_CAPACITY;

    // Mapped pointers are forgotten by L1Streams (the buffer handles are
    // enough for the decoder bind + host readback). Unmap and zero the
    // local Buffer handles so the errdefer chain becomes a no-op on
    // the success return.
    if (vk.vkUnmapMemory_fn) |u| {
        u(ctx.dev, lit_b.mem);
        u(ctx.dev, cmd_b.mem);
        u(ctx.dev, off16_b.mem);
        u(ctx.dev, length_b.mem);
        u(ctx.dev, off32_b.mem);
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
    off32_b.buf = null;
    off32_b.mem = null;
    off32_b.mapped = null;

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
    // Discrete-GPU readback pattern (TODO P1 fix):
    //   dst_b      = DEVICE_LOCAL-only — the kernel writes here at full
    //                VRAM bandwidth instead of through PCIe BAR.
    //   dst_stage  = HOST_VISIBLE | HOST_COHERENT (no DEVICE_LOCAL
    //                preference) — lives in driver-cached sysmem, so
    //                CPU readback is plain cached-memory bandwidth
    //                (10+ GB/s) rather than uncached PCIe BAR reads
    //                (20–60 MB/s observed on RTX 4060 Ti before the
    //                fix, see commit before this one).
    // The kernel writes Dst, then a vkCmdCopyBuffer in the SAME cmdbuf
    // stages it into dst_stage; the host @memcpy at the end then reads
    // from dst_stage at cached-memory speed. On Intel iGPU the same
    // pattern is a tiny regression (one extra GPU copy at iGPU memory
    // bandwidth) because shared memory makes the discrete-GPU asymmetry
    // a non-issue — but the regression is in the 10s of µs and
    // dwarfed by the wins everywhere else.
    const t_alloc0 = qpcNow();
    var dst_b = try createBufferEx(
        ctx,
        dst_buf_size,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .device_local_only,
    );
    defer destroyBuffer(ctx, &dst_b);
    // Staging buffer for readback — explicitly pinned to HOST_VISIBLE+
    // sysmem (NOT DEVICE_LOCAL). On NVIDIA RTX 4060 Ti the rebar region
    // is reported as HOST_VISIBLE+DEVICE_LOCAL, but CPU reads from it
    // are uncached PCIe BAR reads (~20-60 MB/s, measured pre-fix).
    // Sysmem-backed HOST_VISIBLE is driver-cached and reads at full
    // memory bandwidth (4-10 GB/s). On Intel iGPU every host-visible
    // heap IS device-local (shared memory), so the .sysmem mode falls
    // back to the rebar-style ideal — identical behavior on iGPU,
    // dramatic win on discrete.
    var dst_stage = try createBufferEx(
        ctx,
        dst_buf_size,
        vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .host_visible_sysmem,
    );
    defer destroyBuffer(ctx, &dst_stage);
    const t_alloc1 = qpcNow();
    last_decode_dst_alloc_ns = qpcNs(t_alloc0, t_alloc1);
    // No @memset needed — the shader writes every output byte that
    // the host will read. The 1–3 trailing bytes between dst_size and
    // dst_buf_size (the round-up-to-4 padding) may contain garbage,
    // but those bytes are not in [0..dst_size) and the @memcpy below
    // only reads dst_size bytes.
    last_decode_dst_memset_ns = 0;

    // ChunkDescs: 16 u32 per chunk. See lz_decode.comp head comment for
    // the slot layout (slots 12..14 plumb off32 counts + the cmd-stream
    // boundary; slot 15 is reserved for future alignment).
    const off32_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * streams.off32_capacity;
    var chunks_b = try createBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 16,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &chunks_b);
    @memset(chunks_b.mapped.?[0..@intCast(chunks_b.size)], 0);
    {
        const chunks_words: [*]u32 = @ptrCast(@alignCast(chunks_b.mapped.?));
        const cap_words: u32 = chunk_capacity / 4;
        const off32_cap_words: u32 = streams.off32_capacity / 4;
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
            // For the Vulkan encoder's own decode path the cmd-stream
            // block-2 boundary is at `cmd_size` for sub-chunks larger
            // than LZ_BLOCK_SIZE — every token gets attributed to block-1
            // (CUDA writes the boundary at the actual encoder-side token
            // count for block 1; we don't track that yet, so the
            // boundary collapses to "the rest is block 1"). Multi-chunk
            // round-trip with default chunks still works because
            // `cmd_stream2_offset = 0` means "no block-2 boundary",
            // i.e. everything is block-0 — and the encoder's
            // off32_count_block1/2 split matches that view.
            //
            const base = ci * 16;
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
            // Slot 10 = initial_copy bytes (0 or 8). Mirrors the
            // sub-chunk prefix the CUDA decoder writes from
            // SUBCHUNK_PAYLOAD[0..8] to dst[0..8] for chunk 0.
            chunks_words[base + 10] = streams.per_chunk_initial_copy[ci];
            // Slot 11 = u32-word base of the per-chunk off32 slice.
            chunks_words[base + 11] = ci * off32_cap_words;
            // Slots 12..14 = off32_count_block1 / off32_count_block2 /
            // cmd_stream2_offset.
            chunks_words[base + 12] = streams.per_chunk_off32_count1[ci];
            chunks_words[base + 13] = streams.per_chunk_off32_count2[ci];
            chunks_words[base + 14] = streams.per_chunk_cmd_stream2_offset[ci];
            // slot 15 reserved.
        }
    }
    _ = off32_cap_total;

    // ── Build decode pipeline + descriptor set ───────────────────
    const t_desc0 = qpcNow();
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_dec = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_decode",
        tier,
        spv,
        7, // bindings 0..6: cmd, lit, off16, length, dst, chunks, off32
        @sizeOf(DecodePush),
    );

    const dec_bindings: [7]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = streams.cmd_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.lit_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.off16_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.length_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.off32_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);
    const t_desc1 = qpcNow();
    last_decode_descriptors_ns = qpcNs(t_desc0, t_desc1);

    // ── Submit decode dispatch (n_chunks workgroups) ─────────────
    const dec_push: DecodePush = .{
        .n_chunks = n_chunks,
    };
    var dec_push_bytes: [@sizeOf(DecodePush)]u8 = undefined;
    @memcpy(dec_push_bytes[0..], std.mem.asBytes(&dec_push));

    const t_disp0 = qpcNow();
    const dec_dispatch_result = try dispatch.submitOneWithCopy(
        ctx,
        cached_dec.pipeline,
        cached_dec.pipeline_layout,
        dec_set,
        dec_push_bytes[0..],
        .{ n_chunks, 1, 1 },
        .{
            .src = dst_b.buf,
            .dst = dst_stage.buf,
            .size = @as(vk.VkDeviceSize, dst_size),
        },
    );
    const t_disp1 = qpcNow();
    last_decode_dispatch_ns = dec_dispatch_result.ns;
    last_decode_dispatch_wall_ns = qpcNs(t_disp0, t_disp1);

    // Copy dst out via the host map. Reads from the host-visible
    // staging buffer (which the GPU just filled via vkCmdCopyBuffer),
    // not from BAR-mapped device memory.
    const t_read0 = qpcNow();
    if (dst_size > 0) {
        const stage_mapped = dst_stage.mapped orelse return error.MapMemoryFailed;
        @memcpy(dst_host[0..dst_size], stage_mapped[0..dst_size]);
    }
    const t_read1 = qpcNow();
    last_decode_readback_ns = qpcNs(t_read0, t_read1);
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
        .{ &streams.off32_buf, &streams.off32_mem },
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
    streams.off32_total = 0;
    streams.n_chunks = 0;
    streams.dst_size = 0;
}
