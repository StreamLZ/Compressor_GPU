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
const wire_format_gpu = @import("wire_format_gpu.zig");
const decode_pipeline_gpu = @import("decode_pipeline_gpu.zig");
const decode_workspace = @import("decode_workspace.zig");
const descriptors = @import("descriptors.zig");
const dispatch = @import("dispatch.zig");
const wire_constants = @import("wire_constants.zig");
const spv_blobs = @import("spv_blobs");

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
    BadHeader,
    TooManyChunks,
} ||
    l1_codec.L1Error ||
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

    // GPU wrap is the ONLY path. The per-chunk LZ streams stay device-
    // resident; the host never sees them. Frame + block headers are
    // synthesized on-GPU (see wire_format_gpu.zig + shaders/frame_assemble.comp).
    return wire_format_gpu.wrapL1ToSlz1Gpu(ctx, allocator, enc.streams, src, out) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OutputTooSmall => return error.OutputTooSmall,
        error.BadHeader => return error.BadHeader,
        error.UnsupportedTier => return error.UnsupportedTier,
        error.NoSpvForTier => return error.NoSpvForTier,
        error.TooManyChunks => return error.TooManyChunks,
        else => return error.OutOfMemory,
    };
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

// Sub-phase breakdown for the dispatch_ns and readback_ns globals
// above. Populated from `dispatch.DispatchResult` (kernel + GPU copy
// GPU-side timestamps, host-wall around submit+fence-wait) on every
// decode invocation; all in nanoseconds. cli_vk.zig prints them when
// SLZ_VK_PROFILE_DECODE=1.
//
//   gpu_kernel_ns       = TS_SLOT_END - TS_SLOT_BEGIN (lz_decode kernel)
//   gpu_copy_ns         = TS_SLOT_COPY_END - TS_SLOT_COPY_BEGIN
//                         (the vkCmdCopyBuffer dst_b → dst_stage)
//   submit_wait_wall_ns = QPC host-wall around vkQueueSubmit through
//                         vkWaitForFences (kernel + GPU copy + driver
//                         scheduling overhead from the host's POV)
//
// readback_ns above is purely the host @memcpy(out, dst_stage.mapped)
// because submitOneWithCopy waits on the fence INSIDE before returning,
// so all GPU-side work is done before the t_readback_begin QPC sample.
pub var last_decode_slz_gpu_kernel_ns: u64 = 0;
pub var last_decode_slz_gpu_copy_ns: u64 = 0;
pub var last_decode_slz_submit_wait_wall_ns: u64 = 0;

// Finer-grained host-side attribution for the decode dispatch phase.
// Populated when SLZ_VK_PROFILE_DECODE=1. All in nanoseconds. The
// l1_unwrap submit is the small `dispatch.submitOne` call before the
// lz_decode `dispatch.submitOneWithCopy`. Together with the
// gpu_kernel/gpu_copy/submit_wait_wall fields above this gives a
// per-call breakdown of where `last_decode_slz_dispatch_ns` is spent.
pub var last_decode_slz_unwrap_submit_wait_wall_ns: u64 = 0;
pub var last_decode_slz_unwrap_record_ns: u64 = 0;
pub var last_decode_slz_unwrap_submit_call_ns: u64 = 0;
pub var last_decode_slz_unwrap_wait_call_ns: u64 = 0;
pub var last_decode_slz_unwrap_query_read_ns: u64 = 0;
pub var last_decode_slz_dec_record_ns: u64 = 0;
pub var last_decode_slz_dec_submit_call_ns: u64 = 0;
pub var last_decode_slz_dec_wait_call_ns: u64 = 0;
pub var last_decode_slz_dec_query_read_ns: u64 = 0;

// One-shot diagnostic outputs filled by `runReadbackDiagnostics`. The
// helper runs once per process the first time decodeSlz1ToBytesEx is
// called with SLZ_VK_PROFILE_DECODE=1, then sets `_ran` to disable
// further runs. Used by the readback-cost investigation.
//
//   dst_stage_mt_index   = the VkMemoryType index findHostVisibleNonDeviceLocal
//                          actually resolved on this device for the
//                          dst_stage buffer (95 MB shape, TRANSFER_DST).
//   dst_stage_mt_flags   = the propertyFlags bitmask of that memory type
//                          (DEVICE_LOCAL=0x1, HOST_VISIBLE=0x2,
//                           HOST_COHERENT=0x4, HOST_CACHED=0x8).
//   sysmem_memcpy_GBps_x100 = standalone sysmem→sysmem @memcpy throughput
//                          (× 100, integer GB/s) on a 95 MB buffer pair.
//                          Theoretical ceiling for the dst_stage → out
//                          memcpy step in the readback path.
//   bar_like_memcpy_GBps_x100 = same shape, but src is a freshly mapped
//                          dst_stage-style HOST_VISIBLE Vulkan buffer.
//                          If this is similar to the sysmem rate the
//                          backing memory really is sysmem-cached; if
//                          it's an order of magnitude slower it's
//                          BAR-mapped.
pub var diag_ran: bool = false;
pub var diag_dst_stage_mt_index: i32 = -1;
pub var diag_dst_stage_mt_flags: u32 = 0;
pub var diag_sysmem_memcpy_GBps_x100: u32 = 0;
pub var diag_bar_like_memcpy_GBps_x100: u32 = 0;
pub var diag_n_memory_types: u32 = 0;
pub var diag_memory_type_flags: [vk.VK_MAX_MEMORY_TYPES]u32 = @splat(0);

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

/// One-shot diagnostic for the readback path. Runs the first time
/// `decodeSlz1ToBytesEx` is called with SLZ_VK_PROFILE_DECODE=1. Writes
/// findings into the `diag_*` globals; cli_vk.zig prints them.
///
/// Three measurements:
///   1. Enumerate all VkMemoryType slots on the physical device and
///      record their propertyFlags. Cross-references whatever
///      findHostVisibleNonDeviceLocal happens to pick on this device.
///   2. Allocate a real dst_stage-shaped (TRANSFER_DST, 95 MB) buffer
///      using the same .host_visible_sysmem mode as the production
///      decode path, read back its resolved memoryTypeIndex via the
///      same heuristic, and record its propertyFlags. Confirms whether
///      "host_visible_sysmem" actually got a sysmem-only HOST_VISIBLE
///      type or whether the helper fell through to a DEVICE_LOCAL +
///      HOST_VISIBLE rebar slot.
///   3. Run a 95 MB host @memcpy from one sysmem (Zig allocator) buffer
///      to another to measure the theoretical ceiling for the final
///      `@memcpy(out, dst_stage.mapped)` step in the readback path.
///   4. Run a 95 MB host @memcpy from the freshly mapped dst_stage
///      buffer's `.mapped` pointer into the sysmem dst buffer. This is
///      the actual operation the readback path performs. If (4) << (3)
///      the backing memory is BAR-mapped, not sysmem.
fn runReadbackDiagnostics(
    ctx: *driver.Context,
    allocator: std.mem.Allocator,
    sample_size: usize,
) void {
    if (diag_ran) return;
    diag_ran = true;

    // 1. Enumerate memory types so we can print them.
    if (vk.vkGetPhysicalDeviceMemoryProperties_fn) |get_mem_props| {
        var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
        get_mem_props(ctx.pd, &mem_props);
        diag_n_memory_types = mem_props.memoryTypeCount;
        var i: u32 = 0;
        while (i < mem_props.memoryTypeCount and i < vk.VK_MAX_MEMORY_TYPES) : (i += 1) {
            diag_memory_type_flags[i] = mem_props.memoryTypes[i].propertyFlags;
        }
    }

    // 2. Build a dst_stage-shaped buffer (same size, same usage, same
    //    MemMode) and inspect what memory type it actually got. We
    //    can't directly read `mem_type_index` from the l1_codec.Buffer
    //    struct (it isn't stored), so we replay the lookup with the
    //    real buffer's memory requirements.
    const dst_buf_size: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 4),
        (@as(vk.VkDeviceSize, sample_size) + 3) & ~@as(vk.VkDeviceSize, 3),
    );
    var probe_buf = l1_codec.createBufferEx(
        ctx,
        dst_buf_size,
        vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .host_visible_sysmem,
    ) catch return;
    defer l1_codec.destroyBuffer(ctx, &probe_buf);

    if (vk.vkGetBufferMemoryRequirements_fn) |get_req| {
        if (vk.vkGetPhysicalDeviceMemoryProperties_fn) |get_mem_props| {
            var req: vk.VkMemoryRequirements = .{};
            get_req(ctx.dev, probe_buf.buf, &req);
            var mem_props: vk.VkPhysicalDeviceMemoryProperties = .{};
            get_mem_props(ctx.pd, &mem_props);
            // Mirror createBufferEx(.host_visible_sysmem)'s priority
            // order so the diag prints the type the buffer actually
            // got — not the helper-by-helper internals:
            //   1. HOST_VISIBLE+HOST_COHERENT+HOST_CACHED
            //   2. HOST_VISIBLE+HOST_COHERENT and NOT DEVICE_LOCAL
            //   3. HOST_VISIBLE+HOST_COHERENT (any)
            const hv_co = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            const hv_co_cached = hv_co | vk.VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

            // Priority 1: HOST_CACHED.
            var i: u32 = 0;
            while (i < mem_props.memoryTypeCount) : (i += 1) {
                const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
                if (!supported) continue;
                const flags = mem_props.memoryTypes[i].propertyFlags;
                if ((flags & hv_co_cached) == hv_co_cached) {
                    diag_dst_stage_mt_index = @intCast(i);
                    diag_dst_stage_mt_flags = flags;
                    break;
                }
            }
            // Priority 2: non-DEVICE_LOCAL HOST_VISIBLE+HOST_COHERENT.
            if (diag_dst_stage_mt_index < 0) {
                i = 0;
                while (i < mem_props.memoryTypeCount) : (i += 1) {
                    const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
                    if (!supported) continue;
                    const flags = mem_props.memoryTypes[i].propertyFlags;
                    const has_want = (flags & hv_co) == hv_co;
                    const is_device_local = (flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                    if (has_want and !is_device_local) {
                        diag_dst_stage_mt_index = @intCast(i);
                        diag_dst_stage_mt_flags = flags;
                        break;
                    }
                }
            }
            // Priority 3: any HOST_VISIBLE+HOST_COHERENT.
            if (diag_dst_stage_mt_index < 0) {
                i = 0;
                while (i < mem_props.memoryTypeCount) : (i += 1) {
                    const supported = (req.memoryTypeBits & (@as(u32, 1) << @intCast(i))) != 0;
                    if (!supported) continue;
                    const flags = mem_props.memoryTypes[i].propertyFlags;
                    if ((flags & hv_co) == hv_co) {
                        diag_dst_stage_mt_index = @intCast(i);
                        diag_dst_stage_mt_flags = flags;
                        break;
                    }
                }
            }
        }
    }

    // 3. Sysmem→sysmem @memcpy throughput on this host.
    const sysmem_src = allocator.alloc(u8, sample_size) catch return;
    defer allocator.free(sysmem_src);
    const sysmem_dst = allocator.alloc(u8, sample_size) catch return;
    defer allocator.free(sysmem_dst);
    // Touch every page on the dst to fault them in (avoid first-touch
    // page-fault overhead inflating the @memcpy time).
    @memset(sysmem_src, 0xAA);
    @memset(sysmem_dst, 0x55);
    {
        const t0 = qpcNow();
        @memcpy(sysmem_dst, sysmem_src);
        const ns = qpcNs(t0, qpcNow());
        const gbps = (@as(f64, @floatFromInt(sample_size)) / 1_073_741_824.0) /
            (@as(f64, @floatFromInt(ns)) / 1_000_000_000.0);
        diag_sysmem_memcpy_GBps_x100 = @intFromFloat(gbps * 100.0);
    }

    // 4. dst_stage.mapped → sysmem @memcpy. This is the exact operation
    //    the production readback performs (out is allocator-backed
    //    sysmem, dst_stage.mapped is the Vulkan-mapped buffer).
    if (probe_buf.mapped) |stage_mapped| {
        // Touch the mapped region by writing zeroes through it. If it
        // really is sysmem this is a sysmem write (~10 GB/s); if it's
        // BAR-mapped this is a PCIe write (~3 GB/s typical).
        @memset(stage_mapped[0..sample_size], 0);
        // Run the read twice and keep the fastest — the first iteration
        // can be polluted by cache cold-start.
        var best_ns: u64 = std.math.maxInt(u64);
        var r: usize = 0;
        while (r < 2) : (r += 1) {
            const t0 = qpcNow();
            @memcpy(sysmem_dst, stage_mapped[0..sample_size]);
            const ns = qpcNs(t0, qpcNow());
            if (ns < best_ns) best_ns = ns;
        }
        const gbps = (@as(f64, @floatFromInt(sample_size)) / 1_073_741_824.0) /
            (@as(f64, @floatFromInt(best_ns)) / 1_000_000_000.0);
        diag_bar_like_memcpy_GBps_x100 = @intFromFloat(gbps * 100.0);
    }
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

    const unwrap_spv_raw = spv_blobs.find("l1_unwrap", tier_b) orelse return error.NoSpvForTier;
    const unwrap_spv = try dupAlignedSpv(allocator, unwrap_spv_raw);
    defer allocator.free(unwrap_spv);

    // S001 housekeeping: reset every descriptor pool in the
    // process-lifetime decode cache at the start of every decode call.
    // The pools themselves stay alive, so this is a constant-time call
    // per pipeline rather than the destroy+rebuild traffic the old
    // per-call invalidateAll paid. Without this, the per-pipeline pool
    // (MAX_SETS_PER_POOL = 16) would fill after ~16 / sets-per-call
    // decode invocations and vkAllocateDescriptorSets would start
    // failing with OUT_OF_POOL_MEMORY. Mirrors CUDA's per-call kernel-
    // arg pattern (every cuLaunchKernel re-supplies args at zero cost).
    descriptors.resetAllPools(ctx, driver.getOrCreateDecodePipelineCache(ctx));

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
    //
    // Pooled in the per-context decode workspace (`ws.chunks_16u32`)
    // alongside the 14 pipeline-internal buffers and the 2 dst buffers
    // below. Mirrors CUDA's `d_descs_persist` field
    // (src/decode/decode_context.zig:207), grown lazily via
    // `ensureDeviceBuf` at src/decode/decode_dispatch.zig:520 — same
    // shape here. The alloc_ns timing slot reports the
    // ensureWorkspaceBuf cost (~0 on warm calls; the first decode pays
    // the buffer grow).
    const t_alloc_begin = qpcNow();
    const ws = driver.getOrCreateDecodeWorkspace(ctx) catch return error.OutOfMemory;
    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.chunks_16u32,
        @as(vk.VkDeviceSize, n_chunks) * @sizeOf(u32) * 16,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .device_local_only,
    );
    const chunks_b: l1_codec.Buffer = ws.chunks_16u32.buffer;
    last_decode_slz_alloc_ns = qpcNs(t_alloc_begin, qpcNow());

    // No per-stream allocations and no host fill: lz_decode reads
    // every stream directly from pipeline_result.frame at the byte
    // offsets l1_unwrap.comp computed. These two telemetry slots
    // stay at 0 — the work they used to measure no longer happens.
    last_decode_slz_memset_ns = 0;
    last_decode_slz_fill_ns = 0;

    // ── dst buffer(s) ──────────────────────────────────────────────
    //
    // Two shapes are supported, picked by device capability:
    //
    //   single-buffer (NVIDIA discrete with rebar, AMD, Apple Silicon
    //   IGP — `deviceHasDeviceLocalHostCached(ctx) == true`):
    //     One buffer in DEVICE_LOCAL+HOST_VISIBLE+HOST_COHERENT+
    //     HOST_CACHED memory (NVIDIA type[2] = 0x00f). The kernel
    //     writes directly into this buffer at VRAM bandwidth and the
    //     host @memcpy reads the same memory through the CPU cache.
    //     No vkCmdCopyBuffer is needed — the dst→stage transfer the
    //     old two-buffer path emitted was pure overhead.
    //
    //   two-buffer (Intel iGPU — DEVICE_LOCAL+HOST_CACHED is not a
    //   single memory type there; type[3] = HOST_CACHED is sysmem and
    //   GPU writes to it are ~100× slower than to type[1] = DEVICE_LOCAL):
    //     `dst_b`     in DEVICE_LOCAL-only memory (fast kernel writes)
    //     `dst_stage` in HOST_VISIBLE+HOST_CACHED memory (cached CPU
    //                 reads, ~14 GB/s on every device measured)
    //     A vkCmdCopyBuffer(dst_b → dst_stage) inside the dispatch
    //     cmdbuf stages the result for readback.
    //
    // The shape decision lives here (per-call, cheap probe) so the
    // submit pattern downstream stays uniform: it's always one merged
    // submitTwoWithCopy, with copy.size==0 (no copy emitted) for the
    // single-buffer shape and copy.size==dst_total (full GPU staging
    // copy with the compute→transfer barrier) for the two-buffer shape.
    const dst_buf_size: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 4),
        (@as(vk.VkDeviceSize, dst_total) + 3) & ~@as(vk.VkDeviceSize, 3),
    );
    const direct_write: bool = !use_dst_override and
        l1_codec.deviceHasDeviceLocalHostCached(ctx);
    // Both dst_b and dst_stage are pooled in the per-context workspace
    // — mirrors CUDA's `d_output` field and pinned-host mirror
    // (src/decode/decode_context.zig:195-201). Per-call free + realloc
    // was the largest single contributor to the 23.7 ms alloc overhead
    // the Nsight Systems trace flagged (dst_b is the ~95 MB
    // DEVICE_LOCAL slot, dst_stage is the ~95 MB HOST_VISIBLE+
    // HOST_CACHED mirror — both vkAllocateMemory calls on the same
    // memory heap). After this change, the first decode pays both
    // grows; subsequent calls of the same-or-smaller decompressed
    // size are no-ops.
    var dst_b: l1_codec.Buffer = .{};
    var dst_stage: l1_codec.Buffer = .{};
    if (!use_dst_override) {
        if (direct_write) {
            // Single-buffer: dst_stage IS the kernel write target.
            try decode_workspace.ensureWorkspaceBuf(
                ctx,
                &ws.dst_stage,
                dst_buf_size,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .host_visible_sysmem,
            );
            dst_stage = ws.dst_stage.buffer;
        } else {
            try decode_workspace.ensureWorkspaceBuf(
                ctx,
                &ws.dst_b,
                dst_buf_size,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .device_local_only,
            );
            dst_b = ws.dst_b.buffer;
            try decode_workspace.ensureWorkspaceBuf(
                ctx,
                &ws.dst_stage,
                dst_buf_size,
                vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                .host_visible_sysmem,
            );
            dst_stage = ws.dst_stage.buffer;
        }
    }
    // No defer destroyBuffer — the workspace owns dst_b / dst_stage
    // across decode calls (frees once in driver.deinit).

    // One-shot readback-cost diagnostic (controlled by SLZ_VK_PROFILE_DECODE=1).
    // Runs before any per-decode timing is captured so the diag work
    // does not pollute the dispatch/readback ns globals. Internally
    // gated by `diag_ran` so subsequent decode calls are no-ops.
    if (!use_dst_override and dst_total > 0) {
        const want_profile: bool = blk2: {
            const raw = std.c.getenv("SLZ_VK_PROFILE_DECODE") orelse break :blk2 false;
            const s = std.mem.span(raw);
            break :blk2 s.len > 0 and s[0] != '0';
        };
        if (want_profile) runReadbackDiagnostics(ctx, allocator, dst_total);
    }

    // ── Descriptor sets ────────────────────────────────────────────
    // S001: cache is process-lifetime (owned by `ctx.decode_pipeline_cache`)
    // so the unwrap + decode pipelines are built once on the first call
    // and reused thereafter. Mirrors CUDA's pattern of loading every
    // kernel once at process init (src/decode/module_loader.zig:140-141,
    // 169-173). The teardown happens in driver.deinit before
    // destroyDevice; no per-call invalidateAll.
    const t_descset_begin = qpcNow();
    const cache = driver.getOrCreateDecodePipelineCache(ctx);

    const vk_pl_cache = driver.getOrCreateVkPipelineCache(ctx);
    const cached_unwrap = try descriptors.getOrCreateWithPipelineCache(
        ctx,
        cache,
        "l1_unwrap",
        tier,
        unwrap_spv,
        4,
        @sizeOf(UnwrapPush),
        vk_pl_cache,
    );

    const unwrap_bindings: [4]vk.VkDescriptorBufferInfo = .{
        .{ .buffer = pipeline_result.frame.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.chunks.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = pipeline_result.meta.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
        .{ .buffer = chunks_b.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const unwrap_set = try descriptors.allocSet(ctx, cached_unwrap, unwrap_bindings[0..]);

    const cached_dec = try descriptors.getOrCreateWithPipelineCache(
        ctx,
        cache,
        "lz_decode",
        tier,
        dec_spv,
        8,
        @sizeOf(DecodePush),
        vk_pl_cache,
    );

    // lz_decode bindings 0..3 (cmd/lit/off16/length) and 6 (off32) all
    // point at pipeline_result.frame — the new descriptor produced by
    // l1_unwrap.comp encodes byte offsets INTO that single compressed
    // buffer for every stream. No per-stream materialized buffers
    // anywhere on this path.
    const dst_bind_buf: vk.VkBuffer = if (opts.dst_buffer_override) |b|
        b
    else if (direct_write)
        // Single-buffer: kernel writes directly into the host-cached
        // staging buffer (no GPU copy needed).
        dst_stage.buf
    else
        // Two-buffer: kernel writes into dst_b (DEVICE_LOCAL VRAM);
        // a vkCmdCopyBuffer below stages it into dst_stage for
        // host readback.
        dst_b.buf;
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

    // ── Merged dispatch: l1_unwrap + lz_decode + dst→stage copy in
    // one cmdbuf + one submit + one fence wait. Saves one
    // vkQueueSubmit/vkWaitForFences round-trip per decode call vs the
    // previous two-submit pattern (measured ~0.4 ms / call on NVIDIA
    // RTX 4060 Ti, ~4 ms / call on Intel iGPU at the time of the
    // merge). When the caller supplied a D2D dst override the copy
    // step collapses to size=0 (no barrier, no vkCmdCopyBuffer) and
    // the kernel's writes land directly in the caller's VkBuffer. ─
    const t_dispatch_begin = qpcNow();
    const unwrap_push: UnwrapPush = .{ .frame_size = @intCast(slz_bytes.len) };
    var unwrap_push_bytes: [@sizeOf(UnwrapPush)]u8 = undefined;
    @memcpy(unwrap_push_bytes[0..], std.mem.asBytes(&unwrap_push));

    // copy_size==0 → single-buffer path: kernel wrote straight into
    // dst_stage (HOST_CACHED), no GPU staging copy needed; the COPY
    // timestamp pair gets written but the resulting copy_ns is 0.
    // copy_size==dst_total → two-buffer path: stage dst_b → dst_stage
    // with the compute→transfer barrier.
    // copy_size==0 when use_dst_override → D2D-style: caller's buffer
    // is the kernel target, no copy needed.
    const copy_size: vk.VkDeviceSize = if (use_dst_override or dst_total == 0 or direct_write)
        0
    else
        @as(vk.VkDeviceSize, dst_total);
    const copy_src: vk.VkBuffer = if (copy_size == 0) null else dst_b.buf;
    const copy_dst: vk.VkBuffer = if (copy_size == 0) null else dst_stage.buf;

    // Mirror CUDA src/decode/decode_dispatch.zig:400-401
    //   lz_groups = (total_chunks + chunks_per_group - 1) / chunks_per_group;
    //   lz_grid_x = (lz_groups + 1) / 2;            // = ceil(lz_groups / WARPS_PER_BLOCK)
    // and src/decode/decode_dispatch.zig:430/451 which launches the kernel
    // with `block(32, 2, 1)` = WARPS_PER_BLOCK = 2 warps per block.
    //
    // In the Vulkan port each walk_frame ChunkDesc already represents one
    // 64 KiB sub-chunk (sc_group_size = 0.25; src_vulkan/wire_constants.zig:42),
    // so the wire-format equivalent of `chunks_per_group` is 1 and
    // `lz_groups == n_chunks`. The lz_decode.comp shader handles 2 chunks
    // per workgroup via `chunk_id = gl_WorkGroupID.x * WARPS_PER_BLOCK + gl_SubgroupID`,
    // matching CUDA src/decode/lz_decode_kernels.cuh:47.
    const WARPS_PER_BLOCK: u32 = 2;
    const lz_grid_x: u32 = (n_chunks + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const dec_dispatch_result = try dispatch.submitTwoWithCopyLabeled(
        ctx,
        .{
            .pipeline = cached_unwrap.pipeline,
            .pipeline_layout = cached_unwrap.pipeline_layout,
            .descriptor_set = unwrap_set,
            .push_constants_bytes = unwrap_push_bytes[0..],
            .group_count = .{ n_chunks, 1, 1 },
            .label = "l1_unwrap",
        },
        .{
            .pipeline = cached_dec.pipeline,
            .pipeline_layout = cached_dec.pipeline_layout,
            .descriptor_set = dec_set,
            .push_constants_bytes = dec_push_bytes[0..],
            .group_count = .{ lz_grid_x, 1, 1 },
            .label = "lz_decode",
        },
        // Inter-dispatch barrier on the chunks buffer that l1_unwrap
        // writes (binding 3 in its set) and lz_decode reads (binding 5
        // in its set). All other lz_decode inputs are either already
        // visible from `vkQueueSubmit`'s implicit host→queue domain
        // transition (the compressed frame uploaded before submit) or
        // produced by the GPU decode pipeline that ran with its own
        // fence wait inside `runDecodePipelineEx`.
        chunks_b.buf,
        vk.VK_WHOLE_SIZE,
        .{
            .src = copy_src,
            .dst = copy_dst,
            .size = copy_size,
        },
        "dst_b->dst_stage",
    );
    last_decode_slz_dispatch_ns = qpcNs(t_dispatch_begin, qpcNow());

    // The merged submit collapses the previous unwrap-submit's
    // per-call wall (record/submit/wait/query) into the single
    // submitTwoWithCopy call below. There is no separate l1_unwrap
    // submit any more, so the unwrap_* globals stay at zero — kept
    // around so the SLZ_VK_PROFILE_DECODE=1 printer doesn't need an
    // out-of-band conditional.
    last_decode_slz_unwrap_submit_wait_wall_ns = 0;
    last_decode_slz_unwrap_record_ns = 0;
    last_decode_slz_unwrap_submit_call_ns = 0;
    last_decode_slz_unwrap_wait_call_ns = 0;
    last_decode_slz_unwrap_query_read_ns = 0;
    // Phase 4: surface the GPU-side dispatch ns so the CLI bench
    // and `slzGetLastTimings_vk` callers can report `d2d` numbers
    // for the SLZ1 decode path. `decodeL1Sync` writes the same
    // global from its own dispatch site (the lower-level direct
    // codec); this path is the one the SLZ1 wire-format unwrap
    // takes, so wiring it here closes the missing path.
    l1_codec.last_decode_dispatch_ns = dec_dispatch_result.ns;
    // Readback-cost diagnostic globals: kernel GPU ns, GPU-side
    // vkCmdCopyBuffer ns, host-wall around submit+fence-wait. All
    // sourced from the per-submitOneWithCopy `DispatchResult` that
    // dispatch.zig fills via the 4-slot timestamp pool + QPC.
    last_decode_slz_gpu_kernel_ns = dec_dispatch_result.ns;
    last_decode_slz_gpu_copy_ns = dec_dispatch_result.copy_ns;
    last_decode_slz_submit_wait_wall_ns = dec_dispatch_result.submit_wait_wall_ns;
    last_decode_slz_dec_record_ns = dec_dispatch_result.record_wall_ns;
    last_decode_slz_dec_submit_call_ns = dec_dispatch_result.submit_call_ns;
    last_decode_slz_dec_wait_call_ns = dec_dispatch_result.wait_call_ns;
    last_decode_slz_dec_query_read_ns = dec_dispatch_result.query_read_ns;

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
