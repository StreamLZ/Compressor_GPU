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

/// Push constants for the GPU L1 unwrap kernel — must match the
/// `PC { uint frame_size; }` block in `shaders/l1_unwrap.comp`.
const UnwrapPush = extern struct {
    frame_size: u32,
};

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

    const unwrap_spv_raw = spv_blobs.find("l1_unwrap", tier_b) orelse return error.NoSpvForTier;
    const unwrap_spv = try dupAlignedSpv(allocator, unwrap_spv_raw);
    defer allocator.free(unwrap_spv);

    // ── GPU decode pipeline: walk_frame → ... → gather_raw_off16 ──
    // Produces pipeline_result.frame (compressed bytes uploaded
    // device-side), pipeline_result.chunks (walk_frame's 6-u32 per
    // chunk SlzChunkDesc), pipeline_result.meta (n_chunks /
    // decomp_size / block_start). All three are LOAD-BEARING for the
    // lz_decode dispatch below — there is no CPU unwrap path anymore.
    //
    // The pipeline is run with no caller-side n_chunks hint because
    // the host has not parsed the frame; the pipeline self-sizes from
    // a frame-bytes-derived upper bound. The host reads `n_chunks`
    // out of the meta buffer after `runDecodePipelineEx` returns
    // (it does a fence-wait internally) and uses it to size the
    // chunks descriptor + drive the lz_decode grid.
    last_decode_slz_unwrap_ns = 0; // legacy CPU-unwrap stage is gone
    var pipeline_result = decode_pipeline_gpu.runDecodePipelineEx(
        ctx,
        allocator,
        slz_bytes,
        .{},
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.UnsupportedTier => return error.UnsupportedTier,
        error.NoSpvForTier => return error.NoSpvForTier,
        error.BadFrame => return error.BadFrame,
        error.TooManyChunks => return error.OutOfMemory,
        else => return error.OutOfMemory,
    };
    defer decode_pipeline_gpu.destroyDecodeResult(ctx, &pipeline_result);

    if (pipeline_result.status != 0) return error.BadFrame;
    if (pipeline_result.n_chunks == 0) return error.BadFrame;

    const n_chunks: u32 = pipeline_result.n_chunks;
    const dst_total: u32 = pipeline_result.decomp_size;

    // When dst_buffer_override is set, the `out` host slice may be
    // empty — the codec writes only to the device buffer. The
    // OutputTooSmall guard still fires for host-output paths.
    const use_dst_override = opts.dst_buffer_override != null;
    if (!use_dst_override and out.len < dst_total) return error.OutputTooSmall;

    // ── 16-u32 ChunkDescs (device-only) ─────────────────────────────
    // This is the buffer l1_unwrap.comp WRITES and lz_decode.comp
    // READS at binding 5. Per-chunk slots (see lz_decode.comp head
    // comment): cmd/lit/off16/length/off32 byte bases + sizes,
    // initial_copy, off32 counts, cmd_stream2_offset. Device-only —
    // host never touches it.
    const t_alloc_begin = qpcNow();
    var chunks_b = try l1_codec.createBufferEx(
        ctx,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 16,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .device_local_only,
    );
    defer l1_codec.destroyBuffer(ctx, &chunks_b);
    last_decode_slz_alloc_ns = qpcNs(t_alloc_begin, qpcNow());

    // No per-stream allocations and no host fill: lz_decode reads
    // every stream directly from pipeline_result.frame at the byte
    // offsets l1_unwrap.comp computed. These two telemetry slots
    // stay at 0 — the work they used to measure no longer happens.
    last_decode_slz_memset_ns = 0;
    last_decode_slz_fill_ns = 0;

    // ── dst buffers (unchanged readback pattern) ───────────────────
    // Discrete-GPU readback pattern (mirrors the P1 fix in
    // l1_codec.decodeL1Sync). dst_b is DEVICE_LOCAL-only so the kernel
    // writes at full VRAM bandwidth; dst_stage is HOST_VISIBLE-sysmem
    // so the host @memcpy reads cached. A vkCmdCopyBuffer in the
    // dispatch cmdbuf stages dst_b → dst_stage.
    const dst_buf_size: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 4),
        (@as(vk.VkDeviceSize, dst_total) + 3) & ~@as(vk.VkDeviceSize, 3),
    );
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

    // ── Descriptor sets ────────────────────────────────────────────
    const t_descset_begin = qpcNow();
    var cache: descriptors.Cache = .{};
    defer descriptors.invalidateAll(ctx, &cache);

    const cached_unwrap = try descriptors.getOrCreate(
        ctx,
        &cache,
        "l1_unwrap",
        tier,
        unwrap_spv,
        4,
        @sizeOf(UnwrapPush),
    );

    const unwrap_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.meta.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const unwrap_set = try descriptors.allocSet(ctx, cached_unwrap, unwrap_bindings[0..]);

    const cached_dec = try descriptors.getOrCreate(
        ctx,
        &cache,
        "lz_decode",
        tier,
        dec_spv,
        8,
        @sizeOf(DecodePush),
    );

    // lz_decode bindings 0..3 (cmd/lit/off16/length) and 6 (off32) all
    // point at pipeline_result.frame — the new descriptor produced by
    // l1_unwrap.comp encodes byte offsets INTO that single compressed
    // buffer for every stream. No per-stream materialized buffers
    // anywhere on this path.
    const dst_bind_buf: vk.VkBuffer = if (opts.dst_buffer_override) |b| b else dst_b.buf;
    const dec_bindings: [8]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = dst_bind_buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        // Cluster A (F001/F002): walk_frame's GPU chunks output —
        // source of truth for dst_offset + decomp_size.
        .{ .buffer = pipeline_result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    const dec_push: DecodePush = .{ .n_chunks = n_chunks };
    var dec_push_bytes: [@sizeOf(DecodePush)]u8 = undefined;
    @memcpy(dec_push_bytes[0..], std.mem.asBytes(&dec_push));

    last_decode_slz_descset_ns = qpcNs(t_descset_begin, qpcNow());

    // ── Dispatch #1: l1_unwrap (n_chunks workgroups, 1 thread each) ─
    const t_dispatch_begin = qpcNow();
    const unwrap_push: UnwrapPush = .{ .frame_size = @intCast(slz_bytes.len) };
    var unwrap_push_bytes: [@sizeOf(UnwrapPush)]u8 = undefined;
    @memcpy(unwrap_push_bytes[0..], std.mem.asBytes(&unwrap_push));

    _ = try dispatch.submitOne(
        ctx,
        cached_unwrap.pipeline,
        cached_unwrap.pipeline_layout,
        unwrap_set,
        unwrap_push_bytes[0..],
        .{ n_chunks, 1, 1 },
    );

    // ── Dispatch #2: lz_decode reading from compressed frame ────────
    // Submit lz_decode. When the caller did NOT supply a D2D dst
    // override, queue a vkCmdCopyBuffer(dst_b → dst_stage) in the
    // same cmdbuf so the host readback below reads from sysmem
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
