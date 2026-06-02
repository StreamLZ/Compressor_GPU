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
const wire_format_gpu = @import("wire_format_gpu.zig");
const decode_pipeline_gpu = @import("decode_pipeline_gpu.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");
const wire_constants = @import("wire_constants.zig");
const spv_blobs = @import("spv_blobs");

/// When true (the default), `encodeL1ToSlz1` uses the GPU-side
/// frame-assembly path (`wire_format_gpu.wrapL1ToSlz1Gpu`); when false
/// it falls back to the CPU `wire_format.wrapL1ToSlz1`. Exposed at
/// module scope so tests can A/B compare the two; production callers
/// should leave it at the default.
pub var use_gpu_wrap: bool = true;

/// When true (the default), `decodeSlz1ToBytes` runs the Phase-2 GPU
/// decode pipeline (walk_frame → prefix_sum_chunks → scan_parse →
/// compact_huff_descs ×4 → compact_raw_descs → gather_raw_off16) in
/// the same VkQueue ahead of the lz_decode dispatch.
///
/// Cluster A wiring (recon2.md) — walk_frame's `chunks` device buffer
/// is bound DIRECTLY into lz_decode's binding-7 descriptor. lz_decode
/// reads `dst_offset` + `decomp_size` from walk_frame's GPU output;
/// the previous "GPU dispatch runs then we throw the result away"
/// pattern is gone. The load-bearing proof is in the cluster's commit
/// message: zeroing walk_frame's chunks output causes vk-l1-test (via
/// streamlz_vk -d) to fail byte-equality.
///
/// What still rides on the CPU unwrap path: the per-stream byte split
/// (lit / cmd / off16 / length / off32) of each chunk's compressed
/// payload, and the per-stream metadata (cmd_stream2_offset, off32
/// counts, initial_copy). There is no GPU producer for this at L1
/// today — `scan_parse` emits valid=0 on every L1 sub-chunk because
/// L1 has no Huffman pass. A full GPU "L1 unwrap" kernel would be a
/// new addition and is tracked separately.
///
/// When `false` the 5-kernel pipeline is skipped and a CPU mirror
/// (`walk_chunks_cpu`, built from `unwrap.result.per_chunk_decomp_size`)
/// is bound in walk_frame's place. Used by the A/B tests.
pub var use_gpu_unwrap: bool = true;

pub const Slz1Error = error{
    UnsupportedTier,
    NoSpvForTier,
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

fn tierBlob(t: probe_mod.Tier) ?spv_blobs.Tier {
    return switch (t) {
        .tier1 => .tier1,
        .tier1_nv => .tier1_nv,
        .tier2 => .tier2,
        .unsupported => null,
    };
}

/// Copy an @embedFile()'d SPV blob into a freshly allocated 4-byte-aligned
/// buffer. `@embedFile` returns alignment-1 bytes; vkCreateShaderModule
/// requires pCode to be 4-byte aligned (and codeSize a multiple of 4).
/// Callers must free the returned slice via the same allocator.
fn dupAlignedSpv(allocator: std.mem.Allocator, spv: []const u8) Slz1Error![]align(4) u8 {
    const buf = try allocator.alignedAlloc(u8, .@"4", spv.len);
    @memcpy(buf, spv);
    return buf;
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

/// Optional knobs for the D2D paths added in Phase 2 (TODO A2).
/// Both fields default to null — the L1 codec falls through to the
/// existing HOST_VISIBLE allocate + memcpy path. When either is
/// populated the codec uses the caller's VkBuffer for that endpoint
/// directly (no internal allocation, no host bounce for THAT
/// endpoint). The other endpoint still rides the host path.
pub const Slz1Options = struct {
    /// Encode-side: caller's VkBuffer holding `src` already on
    /// device. The encoder binds this as the src SSBO instead of
    /// staging the host bytes through its own buffer.
    src_buffer_override: ?vk.VkBuffer = null,
    /// Decode-side: caller's VkBuffer to receive decoded bytes
    /// directly. The decoder uses this as the dst SSBO, skipping
    /// the host-mapped staging buffer. When set, the `out` slice
    /// passed to `decodeSlz1ToBytes` may be empty — the codec
    /// only writes to the device buffer.
    dst_buffer_override: ?vk.VkBuffer = null,
};

/// Vulkan-encode `src` into the L1 streams, wrap into SLZ1, write into `out`.
/// Returns bytes written. Uses `driver.g_default` after ensureInit.
///
/// SPV blobs are baked into the binary via `spv_blobs.zig` at compile time,
/// so the `io` parameter is unused on the SPV side. Kept for ABI stability
/// with the prior runtime-load shape and for any future host-side IO needs.
pub fn encodeL1ToSlz1(
    ctx: *driver.Context,
    io: std.Io,
    allocator: std.mem.Allocator,
    src: []const u8,
    out: []u8,
) Slz1Error!usize {
    return encodeL1ToSlz1Ex(ctx, io, allocator, src, out, .{});
}

pub fn encodeL1ToSlz1Ex(
    ctx: *driver.Context,
    io: std.Io,
    allocator: std.mem.Allocator,
    src: []const u8,
    out: []u8,
    opts: Slz1Options,
) Slz1Error!usize {
    _ = io;
    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const tier_b = tierBlob(tier) orelse return error.UnsupportedTier;
    const enc_spv_raw = spv_blobs.find("lz_encode", tier_b) orelse return error.NoSpvForTier;
    const enc_spv = try dupAlignedSpv(allocator, enc_spv_raw);
    defer allocator.free(enc_spv);

    var enc = try l1_codec.encodeL1MultiEx(ctx, src, enc_spv, .{
        .src_buffer_override = opts.src_buffer_override,
    });
    defer l1_codec.freeStreams(ctx, &enc.streams);

    // GPU wrap path (Phase 3). Default — pipelines the wire-format
    // wrap behind the same VkQueue the encoder already drives, so the
    // host never sees the per-chunk LZ streams. Falls back to the CPU
    // path when `use_gpu_wrap` is toggled off (testing / debugging).
    if (use_gpu_wrap) {
        return wire_format_gpu.wrapL1ToSlz1Gpu(ctx, allocator, enc.streams, src, out) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.OutputTooSmall => return error.OutputTooSmall,
            error.BadHeader => return error.BadHeader,
            error.UnsupportedTier => return error.UnsupportedTier,
            error.NoSpvForTier => return error.NoSpvForTier,
            error.TooManyChunks => return error.OutOfMemory,
            else => return error.OutOfMemory,
        };
    }

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
///
/// SPV blobs are baked into the binary via `spv_blobs.zig` at compile time,
/// so the `io` parameter is unused on the SPV side. Kept for ABI stability.
pub fn decodeSlz1ToBytes(
    ctx: *driver.Context,
    io: std.Io,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    out: []u8,
) Slz1Error!usize {
    return decodeSlz1ToBytesEx(ctx, io, allocator, slz_bytes, out, .{});
}

// Per-process per-phase profile globals (ns) for decodeSlz1ToBytes.
// Always populated; cli_vk.zig prints them when SLZ_VK_PROFILE_DECODE=1.
// Mirror the same shape as l1_codec.last_decode_dst_*_ns so the bench
// harness can pull per-phase numbers for the SLZ1 unwrap path.
// Used by the P4 diagnosis (commit message) to confirm that the
// dst readback through BAR was the dominant cost (~88 ms of a 122 ms
// decode wall-time on web.txt 4.5 MB on NVIDIA RTX 4060 Ti).
pub var last_decode_slz_unwrap_ns: u64 = 0;
pub var last_decode_slz_alloc_ns: u64 = 0;
pub var last_decode_slz_memset_ns: u64 = 0;
pub var last_decode_slz_fill_ns: u64 = 0;
pub var last_decode_slz_descset_ns: u64 = 0;
pub var last_decode_slz_dispatch_ns: u64 = 0;
pub var last_decode_slz_readback_ns: u64 = 0;

const win32 = struct {
    extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};

inline fn qpcNow() i64 {
    var c: i64 = 0;
    _ = win32.QueryPerformanceCounter(&c);
    return c;
}

inline fn qpcNs(from: i64, to: i64) u64 {
    var freq: i64 = 0;
    _ = win32.QueryPerformanceFrequency(&freq);
    if (freq <= 0) freq = 1;
    const delta = if (to > from) to - from else 0;
    const ns = @divTrunc(@as(i128, delta) * 1_000_000_000, @as(i128, freq));
    return @intCast(ns);
}

pub fn decodeSlz1ToBytesEx(
    ctx: *driver.Context,
    io: std.Io,
    allocator: std.mem.Allocator,
    slz_bytes: []const u8,
    out: []u8,
    opts: Slz1Options,
) Slz1Error!usize {
    _ = io;
    const tier = blk: {
        const pr = probe_mod.probe(ctx.inst, ctx.pd);
        switch (pr.tier) {
            .tier1, .tier1_nv, .tier2 => break :blk pr.tier,
            .unsupported => return error.UnsupportedTier,
        }
    };
    const tier_b = tierBlob(tier) orelse return error.UnsupportedTier;
    const dec_spv_raw = spv_blobs.find("lz_decode", tier_b) orelse return error.NoSpvForTier;
    const dec_spv = try dupAlignedSpv(allocator, dec_spv_raw);
    defer allocator.free(dec_spv);

    const t_unwrap_begin = qpcNow();
    var stream_view: wire_format.PerChunkStreams = undefined;
    var unwrap = try wire_format.unwrapSlz1ToL1Streams(allocator, slz_bytes, &stream_view);
    defer wire_format.freeUnwrapStorage(allocator, &unwrap.storage);
    last_decode_slz_unwrap_ns = qpcNs(t_unwrap_begin, qpcNow());

    // Phase-2 GPU decode pipeline. Runs the 5 newly-ported kernels
    // (walk_frame → prefix_sum_chunks → scan_parse → compact_huff ×4 →
    // compact_raw → gather_raw_off16) on `slz_bytes` directly, producing
    // device-resident ChunkDescs + an entropy off16 scratch.
    //
    // On the L1 path the pipeline's outputs aren't (yet) consumed by
    // `lz_decode.comp` — that shader still reads the CPU-unwrapped
    // per-stream bytes from `unwrap.storage`. The GPU dispatch is run
    // anyway because a) the 5 kernels need integration test coverage on
    // every L1 round-trip so regressions surface promptly, and b) future
    // Phase-3 Huffman work will graft into this exact pipeline shape.
    //
    // The 5-kernel chain shares one VkCommandBuffer + fence (see
    // `decode_pipeline_gpu.zig`). Total wall-time on L1 web.txt
    // (4.5 MB, 18 chunks): <1 ms — every kernel is launch-overhead-
    // dominated on the L1 fast path since scan/compact produce zero
    // entries and gather sees `n_raw = 0` at the self-gate. Profiled
    // delta vs use_gpu_unwrap=false on RTX 4060 Ti is +0.4 ms.
    // Cluster A (F001/F002/F003): the GPU pipeline result (specifically
    // walk_frame's `chunks` device buffer) is bound DIRECTLY into the
    // downstream lz_decode dispatch. Previously this struct's lifetime
    // ended at the `if (use_gpu_unwrap)` scope exit before lz_decode
    // ran, which made every GPU output dead-code. Lifting it to the
    // function-body scope keeps the buffers alive across the second
    // dispatch.
    var pipeline_result: decode_pipeline_gpu.DecodeResult = .{};
    defer if (use_gpu_unwrap) decode_pipeline_gpu.destroyDecodeResult(ctx, &pipeline_result);
    if (use_gpu_unwrap) {
        // Hand the CPU-unwrap n_chunks to the pipeline as a sizing hint
        // so the off16 entropy scratch (n_chunks * 4 * 128 KiB) doesn't
        // balloon to 8 GiB against the WALK_MAX_CHUNKS upper bound. The
        // hint is advisory — the GPU kernels still self-gate on
        // walk_frame's `n_chunks` output (read via the
        // `n_chunks_scratch` 4-byte SSBO).
        pipeline_result = decode_pipeline_gpu.runDecodePipelineEx(
            ctx,
            allocator,
            slz_bytes,
            .{ .n_chunks_hint = unwrap.result.n_chunks },
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.UnsupportedTier => return error.UnsupportedTier,
            error.NoSpvForTier => return error.NoSpvForTier,
            error.BadFrame => return error.BadFrame,
            error.TooManyChunks => return error.OutOfMemory,
            else => return error.OutOfMemory,
        };
        // Surface the walk-frame parse status as a frame-level error if
        // the GPU said the frame is malformed. Status==0 (FW_STATUS_OK)
        // is the happy path; non-zero statuses are FW_STATUS_BAD_MAGIC=1
        // .. FW_STATUS_MULTI_BLOCK_UNSUPPORTED=13 per
        // `decode_pipeline_shared.glsl`. The CPU unwrap above already
        // validated the frame, so any non-zero GPU status here means a
        // CPU-vs-GPU parser divergence — fail loud as BadFrame.
        if (pipeline_result.status != 0) return error.BadFrame;
        // Optional integrity check: GPU n_chunks and decomp_size must
        // match CPU. Discrepancies fail BadFrame; CPU has already passed
        // the frame so a mismatch is a kernel-port bug we want to
        // surface immediately. Guarded — disabled in production if the
        // diff prints become noisy.
        if (pipeline_result.n_chunks != unwrap.result.n_chunks or
            pipeline_result.decomp_size != unwrap.result.original_size)
        {
            std.debug.print(
                "[decode_pipeline_gpu] CPU/GPU disagreement: cpu n_chunks={d} decomp_size={d} | gpu n_chunks={d} decomp_size={d}\n",
                .{ unwrap.result.n_chunks, unwrap.result.original_size, pipeline_result.n_chunks, pipeline_result.decomp_size },
            );
            return error.BadFrame;
        }
    }

    const dst_total: u32 = @intCast(unwrap.result.original_size);
    // When dst_buffer_override is set, the `out` host slice may be
    // empty — the codec writes only to the device buffer. The
    // OutputTooSmall guard still fires for host-output paths.
    const use_dst_override = opts.dst_buffer_override != null;
    if (!use_dst_override and out.len < dst_total) return error.OutputTooSmall;

    const chunk_capacity = l1_codec.CHUNK_STREAM_CAPACITY;
    const n_chunks = unwrap.result.n_chunks;
    const stream_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * chunk_capacity;
    const off32_capacity = l1_codec.CHUNK_OFF32_CAPACITY;
    const off32_cap_total: vk.VkDeviceSize = @as(vk.VkDeviceSize, n_chunks) * off32_capacity;

    const t_alloc_begin = qpcNow();
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
    last_decode_slz_alloc_ns = qpcNs(t_alloc_begin, qpcNow());

    // No @memset on the stream buffers. The lz_decode shader reads
    // each stream by byte offset bounded by per-chunk descriptor
    // sizes (g_<stream>_size), so out-of-range bytes are never
    // consumed. Pre-fix, this memset cleared 5 × stream_cap_total of
    // BAR-mapped memory on NVIDIA at write-combine speed (~12 ms on
    // 4.5 MB web.txt) for no functional benefit. Word loads at the
    // tail of a stream may read up to 3 bytes past the last
    // initialized byte, but the byte-shift+mask in loadCmdByte and
    // friends discards them. Verified by the round-trip check below
    // and the full vk-l1-test suite.
    last_decode_slz_memset_ns = 0;

    const t_fill_begin = qpcNow();
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
    last_decode_slz_fill_ns = qpcNs(t_fill_begin, qpcNow());

    const dst_buf_size: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 4),
        (@as(vk.VkDeviceSize, dst_total) + 3) & ~@as(vk.VkDeviceSize, 3),
    );
    // Discrete-GPU readback pattern (mirrors the P1 fix in
    // l1_codec.decodeL1Sync — commits 45613bd + 0fa1afb). The SLZ1
    // decode path was missing this pattern: the previous
    // createMappedBuffer call was placing dst in NVIDIA's resizable
    // BAR region (DEVICE_LOCAL+HOST_VISIBLE), and the trailing
    // @memcpy(out, dst_b.mapped) read at uncached PCIe BAR speed
    // (~20-60 MB/s) — measured 88 ms of a 122 ms decode wall-time
    // on web.txt 4.5 MB. We allocate:
    //   dst_b     = DEVICE_LOCAL-only (no host mapping) so the kernel
    //               writes at full VRAM bandwidth, and
    //   dst_stage = HOST_VISIBLE+HOST_COHERENT explicitly NOT
    //               DEVICE_LOCAL — driver-cached sysmem, so the host
    //               @memcpy reads at full cached-memory bandwidth.
    // A vkCmdCopyBuffer in the same cmdbuf as the dispatch (via
    // dispatch.submitOneWithCopy) stages dst_b → dst_stage.
    // On Intel iGPU every host-visible heap is also device-local; the
    // .host_visible_sysmem mode then falls back to the rebar-style
    // ideal and the extra copy is a tiny iGPU-bandwidth penalty
    // (microseconds, dwarfed by the rest of the path).
    var dst_b: l1_codec.Buffer = .{};
    var dst_stage: l1_codec.Buffer = .{};
    if (!use_dst_override) {
        dst_b = try l1_codec.createBufferEx(
            ctx,
            dst_buf_size,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            .device_local_only,
        );
        dst_stage = try l1_codec.createBufferEx(
            ctx,
            dst_buf_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .host_visible_sysmem,
        );
    }
    defer if (!use_dst_override) l1_codec.destroyBuffer(ctx, &dst_b);
    defer if (!use_dst_override) l1_codec.destroyBuffer(ctx, &dst_stage);

    var chunks_b = try createMappedBuffer(ctx, @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 16);
    defer destroyMappedBuffer(ctx, &chunks_b);
    @memset(chunks_b.mapped[0..@intCast(chunks_b.size)], 0);
    {
        const words: [*]u32 = @ptrCast(@alignCast(chunks_b.mapped));
        const cap_words: u32 = chunk_capacity / 4;
        const off32_cap_words: u32 = off32_capacity / 4;
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const word_base: u32 = ci * cap_words;
            const base = ci * 16;
            // F002: slots 0 (dst_offset) and 1 (dst_size) are NO LONGER
            // consumed by lz_decode.comp — those fields come from the
            // GPU walk_frame chunks buffer bound at descriptor binding 7
            // (or its CPU mirror when use_gpu_unwrap=false). Leaving the
            // slots zero so a leftover-read regression in lz_decode would
            // surface as immediate corruption rather than silent fallback.
            words[base + 0] = 0;
            words[base + 1] = 0;
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

    // Cluster A (F001/F002): walk-format chunks descriptor (6 u32/chunk).
    // When the GPU pipeline ran (`use_gpu_unwrap = true`, default), we
    // BIND walk_frame's GPU-resident output directly into lz_decode's
    // binding-7 descriptor — making the previously-dead GPU dispatch
    // load-bearing. lz_decode reads dst_offset and decomp_size from this
    // buffer; if walk_frame's output is corrupt the decoder writes to
    // wrong offsets and the round-trip cmp fails (verified by the
    // load-bearing proof in cluster A).
    //
    // When `use_gpu_unwrap = false` (the legacy A/B path tests can flip),
    // there's no GPU producer for the walk chunks — we build a CPU mirror
    // from `unwrap.result.per_chunk_decomp_size` so lz_decode still has a
    // valid binding. This fallback exists purely to keep the toggle
    // honest; production callers should leave use_gpu_unwrap=true.
    var walk_chunks_cpu: l1_codec.Buffer = .{};
    defer if (!use_gpu_unwrap) l1_codec.destroyBuffer(ctx, &walk_chunks_cpu);
    if (!use_gpu_unwrap) {
        walk_chunks_cpu = try l1_codec.createBufferEx(
            ctx,
            @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * wire_constants.CHUNK_DESC_U32_COUNT,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            .host_visible_prefer_device_local,
        );
        @memset(walk_chunks_cpu.mapped.?[0..@intCast(walk_chunks_cpu.size)], 0);
        const wcw: [*]u32 = @ptrCast(@alignCast(walk_chunks_cpu.mapped.?));
        var ci: u32 = 0;
        while (ci < n_chunks) : (ci += 1) {
            const dst_off: u32 = ci * unwrap.result.chunk_size;
            const dst_size: u32 = unwrap.result.per_chunk_decomp_size[ci];
            const wbase = ci * wire_constants.CHUNK_DESC_U32_COUNT;
            wcw[wbase + wire_constants.CHUNK_DECOMP_SIZE_SLOT] = dst_size;
            wcw[wbase + wire_constants.CHUNK_DST_OFFSET_SLOT] = dst_off;
        }
    }
    const walk_chunks_bind: vk.VkBuffer = if (use_gpu_unwrap)
        pipeline_result.chunks.buf
    else
        walk_chunks_cpu.buf;

    const t_descset_begin = qpcNow();
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_dec = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_decode",
        tier,
        dec_spv,
        8,
        @sizeOf(DecodePush),
    );

    const dst_bind_buf: vk.VkBuffer = if (opts.dst_buffer_override) |b| b else dst_b.buf;
    const dec_bindings: [8]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = cmd_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = lit_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off16_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = length_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_bind_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = off32_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        // Cluster A (F001/F002): walk_frame's GPU chunks output bound
        // here makes the 5-kernel decode pipeline load-bearing.
        .{ .buffer = walk_chunks_bind, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    const dec_push: DecodePush = .{ .n_chunks = n_chunks };
    var dec_push_bytes: [@sizeOf(DecodePush)]u8 = undefined;
    @memcpy(dec_push_bytes[0..], std.mem.asBytes(&dec_push));

    last_decode_slz_descset_ns = qpcNs(t_descset_begin, qpcNow());

    const t_dispatch_begin = qpcNow();
    // Submit decode dispatch. When the caller did NOT supply a D2D
    // dst override, queue a vkCmdCopyBuffer(dst_b → dst_stage) in
    // the same cmdbuf so the host readback below reads from sysmem
    // (cached) instead of BAR-mapped VRAM (uncached). The D2D path
    // skips the copy because the caller's VkBuffer is its own dst
    // — they don't go through the staging buffer at all.
    const dec_dispatch_result = blk: {
        if (use_dst_override or dst_total == 0) {
            break :blk try dispatch.submitOne(
                ctx,
                cached_dec.pipeline,
                cached_dec.pipeline_layout,
                dec_set,
                dec_push_bytes[0..],
                .{ n_chunks, 1, 1 },
            );
        }
        break :blk try dispatch.submitOneWithCopy(
            ctx,
            cached_dec.pipeline,
            cached_dec.pipeline_layout,
            dec_set,
            dec_push_bytes[0..],
            .{ n_chunks, 1, 1 },
            .{
                .src = dst_b.buf,
                .dst = dst_stage.buf,
                .size = @as(vk.VkDeviceSize, dst_total),
            },
        );
    };
    last_decode_slz_dispatch_ns = qpcNs(t_dispatch_begin, qpcNow());
    // Phase 4: surface the GPU-side dispatch ns so the CLI bench
    // and `slzGetLastTimings_vk` callers can report `d2d` numbers
    // for the SLZ1 decode path. `decodeL1Sync` writes the same
    // global from its own dispatch site (the lower-level direct
    // codec); this path is the one the SLZ1 wire-format unwrap
    // takes, so wiring it here closes the missing path.
    l1_codec.last_decode_dispatch_ns = dec_dispatch_result.ns;

    // Host readback only when the host buffer was actually the
    // dst — D2D callers skip it entirely (the bytes are already
    // in their device buffer). Reads from dst_stage (sysmem,
    // driver-cached) instead of dst_b (DEVICE_LOCAL VRAM).
    const t_readback_begin = qpcNow();
    if (!use_dst_override and dst_total > 0) {
        const stage_mapped = dst_stage.mapped orelse return error.MapMemoryFailed;
        @memcpy(out[0..dst_total], stage_mapped[0..dst_total]);
    }
    last_decode_slz_readback_ns = qpcNs(t_readback_begin, qpcNow());
    return dst_total;
}
