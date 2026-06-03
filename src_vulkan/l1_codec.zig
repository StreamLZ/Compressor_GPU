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
const wire_constants = @import("wire_constants.zig");

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
//
// Re-exports of the cross-cutting wire-format constants that live in
// `wire_constants.zig`. Cluster H (Phase 4 / `recon2.md`) consolidated
// these here so the L1 codec, wire-format wrap/unwrap, decode-pipeline
// orchestrator, and GLSL kernels all agree at compile time. The comptime
// invariants are in `wire_constants.zig`; this file just exposes the
// names with the doc-strings that the rest of the codec module uses.

/// Per-chunk hash-table size for L1 (hash_bits = 17 → 1<<17 entries × u32
/// = 512 KB). Spec §1.1.
pub const HASH_BITS: u32 = wire_constants.HASH_BITS;
pub const HASH_SIZE_BYTES: vk.VkDeviceSize = wire_constants.HASH_SIZE_BYTES;

/// Per-chunk input slice size. Matches `src/format/streamlz_constants.zig`
/// sub_chunk_size = 0x10000 (64 KiB) at sc_group_size = 0.25 — the same
/// sub-chunk size the CUDA encoder defaults to. Each shader workgroup
/// processes one chunk; the encoder uses the 2-block scanBlock interior
/// (64 KiB blocks), so a 64 KiB chunk = 1 block per workgroup.
pub const CHUNK_SIZE: u32 = wire_constants.CHUNK_SIZE;

/// Number of source bytes written verbatim to the head of chunk 0's
/// lit_buf and skipped in its token stream. Mirrors CUDA's
/// `INITIAL_LITERAL_COPY_BYTES` from `src/common/gpu_wire_format.cuh`
/// (matches the "anchor = 8 when is_first" rule in lz_kernel.cu and the
/// `base_offset == 0` prefix copy in lz_dispatch.cuh).
pub const INITIAL_LITERAL_COPY_BYTES: u32 = wire_constants.INITIAL_LITERAL_COPY_BYTES;

/// Per-chunk reservation in each output stream. Worst-case stream size
/// per spec §4.1 is `2 * src_size`; we add 16 bytes of slack so lane-0
/// RMW byte stores into the tail u32 word never write past the slice.
/// Must be a multiple of 4 (shader byte-load helpers assume word-aligned
/// per-chunk bases — see lz_{encode,decode}.comp head-of-file comment).
pub const CHUNK_STREAM_CAPACITY: u32 = wire_constants.CHUNK_STREAM_CAPACITY;

/// Per-chunk reservation for the off32 stream. Bounded by
/// `2 * src_size` (every byte slot in a chunk could in the worst case
/// drive a 3-byte off32 entry; the *2 keeps the headroom symmetric with
/// the other streams), and we pad to a multiple of 4. With CHUNK_SIZE =
/// 128 KiB this works out to ~512 KiB per chunk — generous, but the
/// decoder/encoder allocate on demand only when a chunk actually emits
/// far-offset matches.
pub const CHUNK_OFF32_CAPACITY: u32 = wire_constants.CHUNK_OFF32_CAPACITY;

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

/// Hard cap on chunks per encode. Now (post-F052) just a runtime guard:
/// L1Streams holds allocator-owned slices for per-chunk sidecars so the
/// cap doesn't bound stack-static array sizes. 4096 × 64 KiB = 256 MiB
/// max input — covers the silesia 200 MB scale test plus headroom.
///
/// Sourced from `wire_constants.MAX_CHUNKS` so the orchestrator
/// (`decode_pipeline_gpu.zig::WALK_MAX_CHUNKS`) and the codec agree on a
/// single number. Re-exported here for callers (tests, wire_format.zig)
/// that historically named it `l1_codec.MAX_CHUNKS`.
pub const MAX_CHUNKS: u32 = wire_constants.MAX_CHUNKS;

// ── Public types ──────────────────────────────────────────────────

/// One device-side L1 stream bundle. All four buffers are owned —
/// callers must `freeStreams` exactly once. The size fields sum across
/// every chunk; per-chunk slices into the streams are reconstructed via
/// `chunk_capacity` + the per-chunk count arrays.
///
/// `MAX_CHUNKS` (= `wire_constants.MAX_CHUNKS`) IS load-bearing here:
/// the per-chunk arrays below are inline `[MAX_CHUNKS]u32`. Total
/// struct size = 5 + 3 inline arrays × `MAX_CHUNKS * 4 B`, currently
/// 8 × 4096 × 4 = 128 KiB. The struct is occasionally passed by value
/// across boundaries (the unit-test helpers and `slz1_codec` decode
/// path); the Windows 4 MiB default stack absorbs it but any bump of
/// `MAX_CHUNKS` beyond ~8192 will start to crowd typical 64 KiB stack
/// budgets in alternative configurations. The runtime cap check in
/// `encodeL1Multi` (`if (n_chunks > MAX_CHUNKS)`) provides the user-
/// visible error if input exceeds capacity (256 MiB at the current
/// `CHUNK_SIZE = 64 KiB`).
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
    // vkCmdFillBuffer is core Vulkan 1.0 — used by the decode-pipeline
    // orchestration (`decode_pipeline_gpu.zig`) to zero the chunks
    // buffer before walk_frame writes its (n_chunks) prefix. Resolves
    // lazily here so the decode-only entry path doesn't have to wire
    // its own resolver.
    if (vk.vkCmdFillBuffer_fn == null)
        vk.vkCmdFillBuffer_fn = resolveDeviceFn(vk.FnCmdFillBuffer, ctx.dev, "vkCmdFillBuffer");

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

/// Find a memory type that is HOST_VISIBLE+HOST_COHERENT+HOST_CACHED.
/// On NVIDIA discrete with resizable BAR, the BAR window is exposed as
/// two separate memory types: an uncached one (HOST_VISIBLE+HOST_COHERENT,
/// 0x006) and a driver-cached one (HOST_VISIBLE+HOST_COHERENT+HOST_CACHED,
/// 0x00e). CPU reads from the uncached type issue per-cache-line PCIe
/// reads (~0.28 GB/s on RTX 4060 Ti). The cached variant lets the CPU
/// hit its own L1/L2/L3 on subsequent reads and runs at full DRAM
/// bandwidth (~6–14 GB/s). On Intel integrated GPUs, all heaps are
/// DEVICE_LOCAL but type 0x00f (DEVICE_LOCAL+HOST_VISIBLE+HOST_COHERENT+
/// HOST_CACHED) exists alongside an uncached 0x007 — same speedup applies.
///
/// HOST_COHERENT is required: a HOST_CACHED-only memory type would force
/// callers to add vkInvalidateMappedMemoryRanges before every CPU read.
/// On every device we've seen the cached variant also carries
/// HOST_COHERENT, so the conjunction is the right thing to ask for; on
/// devices where it doesn't, the caller falls through to the next
/// helper rather than picking up new invalidate plumbing.
fn findHostCachedHostVisible(
    pd: vk.VkPhysicalDevice,
    type_bits_mask: u32,
) ?u32 {
    const get_mem_props = vk.vkGetPhysicalDeviceMemoryProperties_fn orelse return null;
    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    get_mem_props(pd, &mem_props);

    const want = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (type_bits_mask & (@as(u32, 1) << @intCast(i))) != 0;
        if (!supported) continue;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        const has_want = (flags & want) == want;
        if (has_want) return i;
    }
    return null;
}

/// Returns true iff the physical device exposes a memory type that is
/// simultaneously DEVICE_LOCAL+HOST_VISIBLE+HOST_COHERENT+HOST_CACHED
/// — i.e. a memory pool the GPU treats as local VRAM (full bandwidth
/// for kernel stores) AND the CPU reads through its own cache (no per-
/// line PCIe round-trips).
///
/// When true (NVIDIA discrete with rebar enabled, AMD, some Apple Silicon
/// IGPs), the SLZ1 decoder can let lz_decode write straight into the
/// readback buffer and skip the dst→stage vkCmdCopyBuffer entirely.
///
/// When false (Intel iGPU's HOST_CACHED type[3] is NOT DEVICE_LOCAL — the
/// GPU writes into uncached sysmem there, ~100× slower than its
/// DEVICE_LOCAL type[1]), the decoder must keep the two-buffer pattern:
/// kernel writes to a DEVICE_LOCAL-only dst, then a vkCmdCopyBuffer
/// stages it into the HOST_CACHED readback buffer.
///
/// Probed against the SLZ1 decoder's intended buffer usage so the result
/// reflects the actual type that allocation would land on.
pub fn deviceHasDeviceLocalHostCached(ctx: *driver.Context) bool {
    ensureBufferFnSlots(ctx);
    if (vk.vkGetPhysicalDeviceMemoryProperties_fn == null) return false;
    if (vk.vkGetBufferMemoryRequirements_fn == null) return false;
    if (vk.vkCreateBuffer_fn == null) return false;
    if (vk.vkDestroyBuffer_fn == null) return false;

    // A 4-byte probe buffer is the cheapest way to read the device's
    // memoryTypeBits mask for the intended STORAGE_BUFFER+TRANSFER_SRC
    // usage. Some drivers restrict the type mask based on usage flags,
    // so probing with the actual flags gives the correct answer.
    const bci: vk.VkBufferCreateInfo = .{
        .size = 4,
        .usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var probe_buf: vk.VkBuffer = null;
    if (vk.vkCreateBuffer_fn.?(ctx.dev, &bci, null, &probe_buf) != vk.VK_SUCCESS) return false;
    defer vk.vkDestroyBuffer_fn.?(ctx.dev, probe_buf, null);

    var req: vk.VkMemoryRequirements = .{};
    vk.vkGetBufferMemoryRequirements_fn.?(ctx.dev, probe_buf, &req);

    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    vk.vkGetPhysicalDeviceMemoryProperties_fn.?(ctx.pd, &mem_props);

    const want = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
        if (!supported) continue;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        if ((flags & want) == want) return true;
    }
    return false;
}

pub const Buffer = struct {
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

pub fn createBufferEx(
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
                // Priority order — the CPU is going to @memcpy this
                // buffer out after the GPU writes it, so what we want
                // is whichever memory type lets CPU reads hit the
                // cache hierarchy instead of issuing per-line PCIe
                // round trips:
                //
                //   1. HOST_VISIBLE+HOST_COHERENT+HOST_CACHED — driver-
                //      cached BAR on NVIDIA discrete (type 0x00e) or
                //      cached host-visible on Intel iGPU (type 0x00f).
                //      CPU reads hit L1/L2/L3 on subsequent passes;
                //      sustained ~6–14 GB/s vs 0.28 GB/s for uncached.
                //   2. HOST_VISIBLE+HOST_COHERENT and NOT DEVICE_LOCAL —
                //      the existing "plain sysmem on iGPU/AMD" path. Only
                //      reached when the device offers no HOST_CACHED
                //      host-visible heap at all.
                //   3. HOST_VISIBLE+HOST_COHERENT fallback — uncached BAR;
                //      what we used to pick on NVIDIA. Kept as the last
                //      resort because the alternative is failing the
                //      allocation.
                if (findHostCachedHostVisible(ctx.pd, req.memoryTypeBits)) |i| break :blk i;
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

pub fn destroyBuffer(ctx: *driver.Context, b: *Buffer) void {
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

/// Wrap a caller-owned pageable host pointer as a VkBuffer + imported
/// VkDeviceMemory pair. The returned Buffer's `mem` is owned by Vulkan
/// (vkFreeMemory tears it down) and DOES NOT free the caller's host
/// allocation — the host pointer's lifetime must outlive the import.
/// `mapped` is left null because the caller already owns the host
/// mapping directly.
///
/// Mirrors CUDA's cuMemcpyDtoH_v2 transfer pattern
/// (src/decode/decode_dispatch.zig:485): the GPU's copy engine DMAs
/// straight from VRAM into the caller's pageable host buffer, with no
/// intermediate staging copy and no host-side @memcpy. The Vulkan
/// equivalent is to register the host pointer via
/// `VkImportMemoryHostPointerInfoEXT`, bind a transient VkBuffer to
/// that VkDeviceMemory, and use that VkBuffer as the destination of
/// vkCmdCopyBuffer.
///
/// Preconditions (caller's responsibility):
///   * `ctx.has_external_memory_host == true`
///   * `host_ptr` is aligned to `ctx.external_memory_host_alignment`
///   * `size_bytes` is a multiple of `ctx.external_memory_host_alignment`
///     (round the caller's requested size up to the alignment boundary;
///     extra trailing bytes are OK since the caller's buffer is the
///     copy destination and we only ever copy `dst_total` bytes into it).
///   * `host_ptr[0 .. size_bytes]` outlives the returned Buffer.
pub fn importHostPointerBuffer(
    ctx: *driver.Context,
    host_ptr: *anyopaque,
    size_bytes: vk.VkDeviceSize,
    usage: vk.VkBufferUsageFlags,
) L1Error!Buffer {
    ensureBufferFnSlots(ctx);
    if (!ctx.has_external_memory_host) return error.MemoryTypeNotFound;
    const get_props = vk.vkGetMemoryHostPointerPropertiesEXT_fn orelse
        return error.MemoryTypeNotFound;
    const create_buf = vk.vkCreateBuffer_fn orelse return error.BufferCreateFailed;
    const get_req = vk.vkGetBufferMemoryRequirements_fn orelse
        return error.BufferCreateFailed;
    const alloc_mem = vk.vkAllocateMemory_fn orelse return error.MemoryAllocateFailed;
    const bind = vk.vkBindBufferMemory_fn orelse return error.BindBufferFailed;

    // Query which memory types are compatible with importing this host
    // pointer. Drivers typically return the HOST_VISIBLE+HOST_COHERENT
    // (no DEVICE_LOCAL) types — i.e. plain sysmem on the host side that
    // the GPU can DMA into.
    var host_ptr_props: vk.VkMemoryHostPointerPropertiesEXT = .{};
    if (get_props(
        ctx.dev,
        vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        host_ptr,
        &host_ptr_props,
    ) != vk.VK_SUCCESS) return error.MemoryTypeNotFound;
    if (host_ptr_props.memoryTypeBits == 0) return error.MemoryTypeNotFound;

    // Create the VkBuffer with the usage flags the caller needs (typically
    // TRANSFER_DST_BIT so vkCmdCopyBuffer can target it).
    const bci: vk.VkBufferCreateInfo = .{
        .size = size_bytes,
        .usage = usage,
        .sharingMode = vk.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buf: vk.VkBuffer = null;
    if (create_buf(ctx.dev, &bci, null, &buf) != vk.VK_SUCCESS) {
        return error.BufferCreateFailed;
    }
    errdefer if (vk.vkDestroyBuffer_fn) |d| d(ctx.dev, buf, null);

    // Pick a memory type that's BOTH supported for this VkBuffer's usage
    // AND compatible with importing the host pointer. The intersection
    // mask `buf_req.memoryTypeBits & host_ptr_props.memoryTypeBits` is
    // what we search.
    var buf_req: vk.VkMemoryRequirements = .{};
    get_req(ctx.dev, buf, &buf_req);
    const compatible_mask = buf_req.memoryTypeBits & host_ptr_props.memoryTypeBits;
    if (compatible_mask == 0) return error.MemoryTypeNotFound;

    // Among compatible types, prefer one that is NOT DEVICE_LOCAL — the
    // imported memory IS the caller's sysmem so DEVICE_LOCAL types make
    // no sense here. In practice drivers only ever advertise HOST_VISIBLE
    // types in `host_ptr_props.memoryTypeBits`, but we keep the explicit
    // preference for clarity.
    var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
    vk.vkGetPhysicalDeviceMemoryProperties_fn.?(ctx.pd, &mem_props);
    var picked: ?u32 = null;
    var i: u32 = 0;
    while (i < mem_props.memoryTypeCount) : (i += 1) {
        const supported = (compatible_mask & (@as(u32, 1) << @intCast(i))) != 0;
        if (!supported) continue;
        const flags = mem_props.memoryTypes[i].propertyFlags;
        // Need HOST_VISIBLE to bind the pointer at all.
        if ((flags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
        picked = i;
        break;
    }
    const mt_idx: u32 = picked orelse return error.MemoryTypeNotFound;

    // Chain VkImportMemoryHostPointerInfoEXT off the allocate-info pNext.
    // `allocationSize` MUST exactly equal the imported region's size
    // (per spec) — the caller has already rounded `size_bytes` up to
    // `minImportedHostPointerAlignment`.
    var import_info: vk.VkImportMemoryHostPointerInfoEXT = .{
        .handleType = vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = host_ptr,
    };
    const mai: vk.VkMemoryAllocateInfo = .{
        .pNext = @ptrCast(&import_info),
        .allocationSize = size_bytes,
        .memoryTypeIndex = mt_idx,
    };
    var mem: vk.VkDeviceMemory = null;
    if (alloc_mem(ctx.dev, &mai, null, &mem) != vk.VK_SUCCESS) {
        return error.MemoryAllocateFailed;
    }
    errdefer if (vk.vkFreeMemory_fn) |f| f(ctx.dev, mem, null);

    if (bind(ctx.dev, buf, mem, 0) != vk.VK_SUCCESS) {
        return error.BindBufferFailed;
    }

    // `mapped` stays null: the caller already owns the host pointer
    // directly, so destroyBuffer doesn't try to unmap.
    return .{ .buf = buf, .mem = mem, .mapped = null, .size = size_bytes };
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

/// Optional knobs for `encodeL1MultiEx`. Phase 2 (TODO A2) adds the
/// `src_buffer_override` slot — when non-null, the encoder skips its
/// own HOST_VISIBLE src allocation + @memcpy and binds the caller's
/// VkBuffer directly into the descriptor set's src slot. The caller
/// is responsible for the buffer's contents being valid for the
/// duration of the dispatch (typically: the caller filled it via
/// staging copy or mapped memory before invoking the codec).
pub const EncodeOptions = struct {
    src_buffer_override: ?vk.VkBuffer = null,
};

/// Encode `src_host` through the Vulkan L1 encoder, slicing into
/// `ceil(src_host.len / CHUNK_SIZE)` independent chunks. Dispatches a
/// single workgroup-per-chunk kernel and reads per-chunk stream sizes
/// back via the Sizes sidecar.
pub fn encodeL1Multi(
    ctx: *driver.Context,
    src_host: []const u8,
    encode_spv: []const u8,
) L1Error!EncodeResult {
    return encodeL1MultiEx(ctx, src_host, encode_spv, .{});
}

pub fn encodeL1MultiEx(
    ctx: *driver.Context,
    src_host: []const u8,
    encode_spv: []const u8,
    opts: EncodeOptions,
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
    // Phase 2 (TODO A2): when the caller registered a D2D source
    // buffer via `slzRegisterBuffer_vk`, the encoder reuses it
    // directly via `opts.src_buffer_override`. Skip the H2D copy
    // (the bytes are already where the GPU needs them) and skip
    // our own allocation. `src_b` stays zero-initialized so the
    // teardown path is a no-op.
    var src_b: Buffer = .{};
    const use_src_override = opts.src_buffer_override != null;
    if (!use_src_override) {
        src_b = try createBuffer(
            ctx,
            @max(@as(vk.VkDeviceSize, 4), @as(vk.VkDeviceSize, src_size)),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            true,
        );
    }
    errdefer if (!use_src_override) destroyBuffer(ctx, &src_b);

    // F061 guard: every encoded chunk reserves CHUNK_STREAM_CAPACITY
    // bytes (= 2*CHUNK_SIZE + 16, ~128 KiB at 64 KiB chunks) in each
    // of 5 output streams. The encoder genuinely needs the worst-case
    // strided buffer because each chunk_id writes into its slice at
    // `chunk_id * chunk_capacity` and the actual per-chunk sizes are
    // a dispatch output. For very large corpora (silesia 200 MB =
    // 3248 chunks → ~2 GiB per stream, ~10 GiB across all 5) this
    // can exceed available VRAM. Fail loud rather than letting the
    // driver return an opaque VK_ERROR_OUT_OF_DEVICE_MEMORY mid-
    // allocation.
    if (stream_cap_total > wire_constants.ENCODE_STREAM_VRAM_MAX_BYTES) {
        return error.MemoryAllocateFailed;
    }

    var lit_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &lit_b);
    var cmd_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &cmd_b);
    var off16_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &off16_b);
    var length_b = try createBuffer(ctx, stream_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &length_b);
    const off32_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * CHUNK_OFF32_CAPACITY;
    if (off32_cap_total > wire_constants.ENCODE_STREAM_VRAM_MAX_BYTES) {
        return error.MemoryAllocateFailed;
    }
    var off32_b = try createBuffer(ctx, off32_cap_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    errdefer destroyBuffer(ctx, &off32_b);

    // Hash buffer: n_chunks × HASH_SIZE_BYTES. Device-local only — the
    // host never reads or writes it.
    //
    // F058 guard: per-chunk hash is 512 KiB, so n_chunks=4096 demands
    // 2 GiB of VRAM purely for hash slots. On a 6 GB iGPU or a 4060 Ti
    // (8 GB total) running the L1 scale test alongside other allocations
    // (lit/cmd/off16/length/off32 streams at ~512 KiB per chunk = ~10 GiB
    // total worst case) this is the buffer that puts us over the cliff.
    // Fail loud with `MemoryAllocateFailed` if the requested hash VRAM
    // exceeds the configured ceiling so callers see a real error rather
    // than an opaque VK_ERROR_OUT_OF_DEVICE_MEMORY from the driver.
    const hash_bytes_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * HASH_SIZE_BYTES;
    if (hash_bytes_total > wire_constants.HASH_VRAM_MAX_BYTES) {
        return error.MemoryAllocateFailed;
    }
    var hash_b = try createBuffer(ctx, hash_bytes_total, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, false);
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
    // Skip the H2D when the caller's src_buffer_override is in use —
    // the bytes are already on device. `src_host.len` is still the
    // logical input size (the codec uses it to compute chunk
    // descriptors etc.).
    if (src_size > 0 and !use_src_override) {
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

    const src_binding_buf: vk.VkBuffer = if (opts.src_buffer_override) |b| b else src_b.buf;
    const enc_bindings: [9]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = src_binding_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
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

    // Drop the src buffer — encode is done. When src_buffer_override
    // was in use, `src_b` is a zero-init Buffer and destroyBuffer is
    // a no-op; the caller owns the override's lifetime.
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

    // Cluster A (F001): walk-format chunks descriptor (6 u32 per chunk,
    // mirror of `decode_pipeline_shared.glsl::SlzChunkDesc`). For the
    // direct codec path there's no walk_frame kernel — the encoder owns
    // the per-chunk geometry — so we mirror walk_frame's output shape
    // CPU-side and bind THIS buffer (not the 16-u32 chunks_b) as the
    // source of truth for dst_offset + decomp_size in lz_decode.comp.
    var walk_chunks_b = try createBuffer(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * wire_constants.CHUNK_DESC_U32_COUNT,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        true,
    );
    defer destroyBuffer(ctx, &walk_chunks_b);
    @memset(walk_chunks_b.mapped.?[0..@intCast(walk_chunks_b.size)], 0);
    {
        const w: [*]u32 = @ptrCast(@alignCast(walk_chunks_b.mapped.?));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const dst_off: u32 = ci * CHUNK_SIZE;
            const this_dst_size: u32 = if (dst_off + CHUNK_SIZE <= dst_size)
                CHUNK_SIZE
            else if (dst_off < dst_size)
                dst_size - dst_off
            else
                0;
            const wbase = ci * wire_constants.CHUNK_DESC_U32_COUNT;
            // SlzChunkDesc layout (see decode_pipeline_shared.glsl):
            //   slot 0: src_offset (unused on this path — encoder owns it)
            //   slot 1: comp_size  (unused on this path)
            //   slot 2: decomp_size  ← consumed by lz_decode
            //   slot 3: dst_offset   ← consumed by lz_decode
            //   slot 4: flags        (zero — direct path has no GPU walk)
            //   slot 5: memset_fill  (zero)
            w[wbase + 0] = 0; // src_offset
            w[wbase + 1] = this_dst_size; // comp_size
            w[wbase + 2] = this_dst_size; // CHUNK_DECOMP_SIZE_SLOT
            w[wbase + 3] = dst_off;       // CHUNK_DST_OFFSET_SLOT
            w[wbase + 4] = 0; // flags
            w[wbase + 5] = 0; // memset_fill (+ pad)
        }
    }
    @memset(chunks_b.mapped.?[0..@intCast(chunks_b.size)], 0);
    {
        const chunks_words: [*]u32 = @ptrCast(@alignCast(chunks_b.mapped.?));
        // lz_decode.comp's slots 2/4/6/8/11 are BYTE-base offsets into
        // their respective stream SSBOs (post-l1_unwrap port). On the
        // direct encode-roundtrip path each stream has its own
        // chunk_capacity-strided SSBO, so the byte base is
        // `ci * chunk_capacity` (already 4-aligned because
        // chunk_capacity is a multiple of 4 — comptime-checked in
        // wire_constants.zig).
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const stream_byte_base: u32 = ci * chunk_capacity;
            const off32_byte_base: u32 = ci * streams.off32_capacity;
            // Cluster C (F018): cmd-stream block-0/1 boundary IS tracked
            // correctly by the encoder. lz_encode.comp writes
            // `cmd_stream2_offset = token_count` at the block-0/1
            // boundary (line 595) when g_src_size > LZ_BLOCK_SIZE,
            // and leaves it as 0 (init value) when the chunk fits in a
            // single LZ block (line 580-584 early-break). At L1 default
            // geometry CHUNK_SIZE == LZ_BLOCK_SIZE so every chunk is
            // single-block and the boundary stays 0 — which is the
            // correct sentinel value the decoder reads (slot 14 → "no
            // block-2 boundary, all tokens are block-0").
            //
            // For multi-chunk corpora the encoder's per-chunk
            // off32_count_block1 / block2 split (read from sizes_buf
            // slots 4 / 5) and the boundary above stay coherent
            // because both are written in the same critical section of
            // lz_encode.comp's main(). See sizes_buf write at
            // shaders/lz_encode.comp:988.
            //
            const base = ci * 16;
            // Cluster A (F002): slots 0/1 (dst_offset, dst_size) are
            // no longer consumed by lz_decode.comp — those fields come
            // from the walk-format chunks buffer at binding 7
            // (`walk_chunks_b` built just above). Zero the slots so a
            // future regression that re-reads them surfaces as
            // immediate output corruption.
            chunks_words[base + 0] = 0;
            chunks_words[base + 1] = 0;
            chunks_words[base + 2] = stream_byte_base; // cmd  byte base
            chunks_words[base + 3] = streams.per_chunk_cmd_size[ci];
            // lit_byte_base / lit_size semantics (post-l1_unwrap port):
            //   lit_byte_base points at the FIRST byte the token loop
            //   reads (after any init prefix). lit_size is just the
            //   chunk's token-driven lit-byte count.
            // The encoder placed the 8-byte init prefix at the head of
            // the per-chunk lit slice — so we bump lit_byte_base by
            // initial_copy and trim lit_size accordingly. The init
            // prefix itself is reachable via init_byte_base (slot 15).
            const ic: u32 = streams.per_chunk_initial_copy[ci];
            chunks_words[base + 4] = stream_byte_base + ic; // lit byte base (after prefix)
            chunks_words[base + 5] = streams.per_chunk_lit_size[ci] - ic;
            chunks_words[base + 6] = stream_byte_base; // off16 byte base
            chunks_words[base + 7] = streams.per_chunk_off16_count[ci];
            chunks_words[base + 8] = stream_byte_base; // length byte base
            chunks_words[base + 9] = streams.per_chunk_length_used[ci];
            // Slot 10 = initial_copy bytes (0 or 8). Mirrors the
            // sub-chunk prefix the CUDA decoder writes from
            // SUBCHUNK_PAYLOAD[0..8] to dst[0..8] for chunk 0.
            chunks_words[base + 10] = ic;
            // Slot 11 = byte base of the per-chunk off32 slice.
            chunks_words[base + 11] = off32_byte_base;
            // Slots 12..14 = off32_count_block1 / off32_count_block2 /
            // cmd_stream2_offset.
            chunks_words[base + 12] = streams.per_chunk_off32_count1[ci];
            chunks_words[base + 13] = streams.per_chunk_off32_count2[ci];
            chunks_words[base + 14] = streams.per_chunk_cmd_stream2_offset[ci];
            // Slot 15 = init_byte_base: where the init prefix sits.
            // Encoder places it at the head of the per-chunk lit slice,
            // so this equals the unbumped stream_byte_base.
            chunks_words[base + 15] = stream_byte_base;
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
        8, // bindings 0..7: cmd, lit, off16, length, dst, chunks, off32, walk_chunks
        @sizeOf(DecodePush),
    );

    const dec_bindings: [8]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = streams.cmd_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.lit_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.off16_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.length_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = streams.off32_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        // Cluster A (F001): walk-format chunks descriptor — source of
        // truth for dst_offset + decomp_size that lz_decode.comp reads.
        .{ .buffer = walk_chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
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
    // Mirror CUDA src/decode/decode_dispatch.zig:400-401
    //   lz_groups = (total_chunks + chunks_per_group - 1) / chunks_per_group;
    //   lz_grid_x = (lz_groups + 1) / 2;            // = ceil(lz_groups / WARPS_PER_BLOCK)
    // and src/decode/decode_dispatch.zig:430/451 which launches the kernel
    // with `block(32, 2, 1)` = WARPS_PER_BLOCK = 2 warps per block.
    //
    // Each `n_chunks` here is one 64 KiB CHUNK_SIZE unit (src_vulkan/
    // l1_codec.zig:101 + computeNChunks), the same shape walk_frame
    // produces from sc_group_size = 0.25. lz_decode.comp dispatches
    // 2 chunks per workgroup via gl_SubgroupID indexing.
    const WARPS_PER_BLOCK: u32 = 2;
    const lz_grid_x: u32 = (n_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const dec_dispatch_result = try dispatch.submitOneWithCopy(
        ctx,
        cached_dec.pipeline,
        cached_dec.pipeline_layout,
        dec_set,
        dec_push_bytes[0..],
        .{ lz_grid_x, 1, 1 },
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
