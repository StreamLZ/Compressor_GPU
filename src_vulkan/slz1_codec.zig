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

// Path-selection telemetry (updated every decode call). 0/1 booleans
// for the gates and the resolved alignment + import size. Used by the
// bench harness to confirm the VK_EXT_external_memory_host fast path
// is engaged after the staging-copy elimination.
pub var last_decode_slz_caller_import_taken: u64 = 0;
pub var last_decode_slz_caller_import_align: u64 = 0;
pub var last_decode_slz_caller_ptr_aligned: u64 = 0;
pub var last_decode_slz_caller_has_ext: u64 = 0;
pub var last_decode_slz_caller_import_size: u64 = 0;

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
    //
    // S003: PREPARE the pipeline (buffer alloc + descriptor sets +
    // pipeline build), but do NOT record/submit yet. The 5 GPU decode
    // kernels' dispatches will be recorded into the SAME command buffer
    // as l1_unwrap + lz_decode + dst→host copy below, riding ONE
    // vkQueueSubmit + ONE vkWaitForFences. Mirrors the CUDA pattern at
    // src/decode/decode_dispatch.zig:614-621 where the LZ kernel reads
    // its self-gate count from the device-resident meta buffer
    // (`d_n_chunks_dev = d_walk_meta + walk_meta_offsets.n_chunks`),
    // eliminating the host fence wait between phases.
    last_decode_slz_unwrap_ns = 0; // legacy CPU-unwrap stage is gone
    // S003: derive an n_chunks upper bound from the caller's out.len
    // (host contract: out.len >= original_size). Each walk_frame
    // ChunkDesc represents one 64 KiB sub-chunk
    // (wire_constants.CHUNK_SIZE), so worst-case n_chunks =
    // (out.len / 64 KiB) + 2. The frame-bytes-derived fallback in
    // `prepareDecodePipeline` uses SLZ_CHUNK_SIZE_BYTES = 256 KiB which
    // is the outer-block bound, NOT the sub-chunk bound — that's a
    // 4× under-count for actual n_chunks. We override here with the
    // tighter sub-chunk bound so the lz_decode grid covers every real
    // chunk. For D2D-override callers `out.len` is 0 → fall back to
    // the prepareDecodePipeline default.
    const n_chunks_hint: u32 = if (opts.dst_buffer_override != null)
        0
    else
        @intCast(@min(
            @as(u64, decode_pipeline_gpu.WALK_MAX_CHUNKS),
            (out.len / wire_constants.CHUNK_SIZE) + 2,
        ));
    var prepared = decode_pipeline_gpu.prepareDecodePipeline(
        ctx,
        allocator,
        slz_bytes,
        .{ .n_chunks_hint = n_chunks_hint },
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.UnsupportedTier => return error.UnsupportedTier,
        error.NoSpvForTier => return error.NoSpvForTier,
        error.BadFrame => return error.BadFrame,
        error.TooManyChunks => return error.OutOfMemory,
        else => return error.OutOfMemory,
    };
    const pipeline_result = prepared.result;

    // Worst-case sizing inputs derived from the frame size before submit.
    // The real n_chunks and decomp_size live in `pipeline_result.meta`
    // after the GPU dispatch chain runs; we read them post-fence at the
    // bottom of this function to drive the host @memcpy. For all
    // host-side ALLOC / DISPATCH / GRID decisions we use the worst-case
    // upper bound (matches CUDA's `d_descs_persist` grow-only sizing at
    // src/decode/decode_context.zig:35-43 — same shape: never shrink).
    const n_chunks_max: u32 = prepared.n_chunks_max;

    // dst_total_max: upper bound on decompressed bytes for buffer
    // sizing. When the caller passes a host `out` slice we trust their
    // contract (out.len >= original_size); when they pass an
    // override-only D2D path with empty `out`, dst sizing is irrelevant
    // (the kernel writes to the caller's VkBuffer directly). The
    // post-fence read of meta gives the real `decomp_size` for the
    // host @memcpy size.
    const use_dst_override = opts.dst_buffer_override != null;
    const dst_total_max: u32 = if (use_dst_override)
        0
    else
        @intCast(@min(out.len, std.math.maxInt(u32)));

    // ── 16-u32 ChunkDescs (device-only) ─────────────────────────────
    // S003: sized against worst-case n_chunks_max so the alloc happens
    // BEFORE the GPU pipeline runs (we don't know the real n_chunks
    // host-side yet). Mirrors CUDA's grow-only `ensureDeviceBuf` at
    // src/decode/decode_context.zig:35: never shrink, only grow.
    const t_alloc_begin = qpcNow();
    const ws = driver.getOrCreateDecodeWorkspace(ctx) catch return error.OutOfMemory;
    try decode_workspace.ensureWorkspaceBuf(
        ctx,
        &ws.chunks_16u32,
        @as(vk.VkDeviceSize, n_chunks_max) * @sizeOf(u32) * 16,
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
    // dispatch chain, with copy.size==0 (no copy emitted) for the
    // single-buffer shape and copy.size==dst_total_max (full GPU staging
    // copy with the compute→transfer barrier) for the two-buffer shape.
    //
    // S003: sized against `dst_total_max` (= out.len, the host
    // contract) since we can't read the real decomp_size until after
    // the merged fence wait. ensureWorkspaceBuf is grow-only so
    // subsequent same-or-smaller calls are no-ops.
    //
    // PAD: lz_decode's warp-cooperative batched writes can land up to
    // 32 bytes past the chunk's true decomp_size (the encoder pre-pads
    // stream capacities so the unbounded `warpLiteralCopy` doesn't
    // emit a bounds-check on every byte). Reserve 64 bytes of tail
    // slack in dst_b so the kernel's over-write doesn't truncate the
    // last batch when `out.len` is not 32-aligned. The CLI bench
    // already pads dec_buf this way (cli_vk.zig:29 VK_SAFE_SPACE = 64
    // → dec_buf_alloc_size = (src.len + 64 + 4095) & ~4095) which is
    // why the bench passed but the L1 test failed before this pad.
    const DST_TAIL_PAD: vk.VkDeviceSize = 64;
    const dst_buf_size: vk.VkDeviceSize = @max(
        @as(vk.VkDeviceSize, 4),
        (@as(vk.VkDeviceSize, dst_total_max) + DST_TAIL_PAD + 3) & ~@as(vk.VkDeviceSize, 3),
    );

    // ── Caller-pointer import path (preferred when supported) ──────
    //
    // CUDA's readback at `src/decode/decode_dispatch.zig:485` issues a
    // single cuMemcpyDtoH_v2 from VRAM straight into the caller's
    // pageable host buffer — no GPU staging copy, no host @memcpy. The
    // Vulkan analog is VK_EXT_external_memory_host: register the
    // caller's `out` buffer as a VkDeviceMemory, bind a transient
    // VkBuffer to it, and make THAT the destination of vkCmdCopyBuffer.
    //
    // Eligibility (all must hold):
    //   * extension enabled at device-creation time
    //   * caller didn't pass a device-side dst override (D2D path)
    //   * dst_total > 0 (skip empty decodes)
    //   * caller's `out` pointer is page-aligned per
    //     `minImportedHostPointerAlignment` AND we can pad the import
    //     size up to a multiple of that same alignment without
    //     overflowing `out.len`.
    //
    // When eligible: kernel writes into dst_b (DEVICE_LOCAL VRAM, full
    // bandwidth) → vkCmdCopyBuffer(dst_b → caller_imported) inside the
    // same submit → caller's bytes land directly in `out`. No host
    // @memcpy. Saves ~7 ms / 95 MB enwik8 on NVIDIA RTX 4060 Ti where
    // no memory type satisfies DEVICE_LOCAL+HOST_VISIBLE+HOST_COHERENT+
    // HOST_CACHED simultaneously (the gate `deviceHasDeviceLocalHostCached`
    // checks); previously that device fell into the two-buffer slow
    // path which paid both a 7.58 ms GPU staging copy AND a 9.50 ms
    // host @memcpy.
    // S003: import_size is sized against `dst_total_max` (= out.len)
    // because we don't know the real decomp_size host-side until after
    // the merged fence wait. Caller's contract is out.len >= original
    // bytes; the import covers the whole user buffer and the GPU only
    // writes the first `decomp_size` bytes — the rest stay as-is.
    const ext_align: u64 = ctx.external_memory_host_alignment;
    const caller_ptr_aligned: bool = ext_align > 0 and
        (@intFromPtr(out.ptr) % ext_align) == 0;
    // S003 tail-byte fix: import_size MUST cover the GPU copy size
    // (= dst_total_max = out.len) exactly, with no rounding gap.
    // Mirrors pre-S003 gate semantics (padded_up(decomp_size) <= out.len)
    // collapsed to the post-S003 worst-case-sizing reality where
    // dst_total_max == out.len: the import region has to cover every
    // byte the GPU copy will write, so we only take the caller-import
    // path when out.len is itself a multiple of ext_align (no slack to
    // pad over). Tight-sized callers (e.g. the L1 codec test with
    // out.len = src.len, not page-padded) fall through to the two-
    // buffer slow path, which is what the pre-S003 code also did
    // (its gate was `padded_up(decomp_size) <= out.len`, which fails
    // whenever out.len is exactly decomp_size and decomp_size isn't
    // ext_align-multiple). Page-padded bench callers (cli_vk.zig:526
    // `(src.len + 64 + 4095) & ~4095` allocation) still hit the fast
    // path because their out.len is page-aligned.
    const out_len_aligned: bool = ext_align > 0 and
        (@as(u64, out.len) % ext_align) == 0;
    const import_size: vk.VkDeviceSize = if (out_len_aligned)
        @as(vk.VkDeviceSize, dst_total_max)
    else
        0;
    const use_caller_import: bool = !use_dst_override and
        dst_total_max > 0 and
        ctx.has_external_memory_host and
        caller_ptr_aligned and
        out_len_aligned and
        import_size > 0;

    // Probe diagnostic globals — updated every decode call so callers
    // can verify which path was taken. Cheap (just stores u64s).
    last_decode_slz_caller_import_taken = if (use_caller_import) 1 else 0;
    last_decode_slz_caller_import_align = ext_align;
    last_decode_slz_caller_ptr_aligned = if (caller_ptr_aligned) 1 else 0;
    last_decode_slz_caller_has_ext = if (ctx.has_external_memory_host) 1 else 0;
    last_decode_slz_caller_import_size = @intCast(import_size);

    const direct_write: bool = !use_dst_override and
        !use_caller_import and
        l1_codec.deviceHasDeviceLocalHostCached(ctx);
    // dst_b and dst_stage are pooled in the per-context workspace
    // — mirrors CUDA's `d_output` field and pinned-host mirror
    // (src/decode/decode_context.zig:195-201). Per-call free + realloc
    // was the largest single contributor to the 23.7 ms alloc overhead
    // the Nsight Systems trace flagged (dst_b is the ~95 MB
    // DEVICE_LOCAL slot, dst_stage is the ~95 MB HOST_VISIBLE+
    // HOST_CACHED mirror — both vkAllocateMemory calls on the same
    // memory heap). After this change, the first decode pays both
    // grows; subsequent calls of the same-or-smaller decompressed
    // size are no-ops.
    //
    // `caller_imported` is cache-fronted via the workspace
    // `ImportedSlot` (see decode_workspace.zig::ensureImportedHostBuf).
    // The IMPORT itself is NOT "just a binding" as the prior comment
    // claimed — vkAllocateMemory + vkCreateBuffer + vkBindBufferMemory
    // each take real driver time (measured ~7 ms / call on NVIDIA
    // RTX 4060 Ti for the 95 MB enwik8 import). Caching keyed by
    // `(caller_ptr, import_size)` collapses every same-buffer repeat
    // call to zero driver work — mirror of CUDA's grow-only ensure
    // pattern at src/decode/decode_context.zig:35-43. The CACHED
    // import is owned by `ws.caller_imported` and freed once in
    // `driver.deinit` (decode_workspace.deinit), NOT per call.
    var dst_b: l1_codec.Buffer = .{};
    var dst_stage: l1_codec.Buffer = .{};
    var caller_imported: l1_codec.Buffer = .{};
    if (!use_dst_override) {
        if (use_caller_import) {
            // dst_b is the kernel's DEVICE_LOCAL VRAM target; the imported
            // VkBuffer is the vkCmdCopyBuffer destination.
            try decode_workspace.ensureWorkspaceBuf(
                ctx,
                &ws.dst_b,
                dst_buf_size,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                .device_local_only,
            );
            dst_b = ws.dst_b.buffer;
            caller_imported = decode_workspace.ensureImportedHostBuf(
                ctx,
                &ws.caller_imported,
                @ptrCast(out.ptr),
                import_size,
                vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            ) catch |err| switch (err) {
                // If the import fails at runtime (unexpected — we
                // already gated on alignment + extension support), fall
                // through to the existing two-buffer slow path rather
                // than failing the decode. On error,
                // `ensureImportedHostBuf` has already cleared the
                // slot, so no stale cached buffer is leaked.
                error.MemoryTypeNotFound,
                error.BufferCreateFailed,
                error.MemoryAllocateFailed,
                error.BindBufferFailed,
                error.MapMemoryFailed,
                => l1_codec.Buffer{},
                else => return error.OutOfMemory,
            };
        } else if (direct_write) {
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
    // Workspace owns dst_b / dst_stage AND caller_imported across
    // decode calls (frees once in driver.deinit). The caller-import
    // pool was previously per-call but is now cached via
    // `ws.caller_imported` keyed on (out.ptr, import_size) — see
    // decode_workspace.ensureImportedHostBuf for the cache logic.
    // The import-path fallback (failed runtime import despite the
    // eligibility gate) leaves `caller_imported.buf == null` and means
    // we have neither dst_stage NOR a target — recover by allocating
    // dst_stage on demand so the readback path still has a destination.
    const caller_import_active: bool = use_caller_import and caller_imported.buf != null;
    if (use_caller_import and !caller_import_active) {
        try decode_workspace.ensureWorkspaceBuf(
            ctx,
            &ws.dst_stage,
            dst_buf_size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .host_visible_sysmem,
        );
        dst_stage = ws.dst_stage.buffer;
    }

    // One-shot readback-cost diagnostic (controlled by SLZ_VK_PROFILE_DECODE=1).
    // Runs before any per-decode timing is captured so the diag work
    // does not pollute the dispatch/readback ns globals. Internally
    // gated by `diag_ran` so subsequent decode calls are no-ops.
    if (!use_dst_override and dst_total_max > 0) {
        const want_profile: bool = blk2: {
            const raw = std.c.getenv("SLZ_VK_PROFILE_DECODE") orelse break :blk2 false;
            const s = std.mem.span(raw);
            break :blk2 s.len > 0 and s[0] != '0';
        };
        if (want_profile) runReadbackDiagnostics(ctx, allocator, dst_total_max);
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

    // S003: lz_decode now reads n_chunks from a device-resident SSBO
    // (binding 8 = WalkMeta) instead of a host push constant; push
    // constant block removed from the shader. Pipeline-layout push
    // size is therefore 0 and the binding count bumps 8 → 9.
    const cached_dec = try descriptors.getOrCreateWithPipelineCache(
        ctx,
        cache,
        "lz_decode",
        tier,
        dec_spv,
        9,
        0,
        vk_pl_cache,
    );

    // lz_decode bindings 0..3 (cmd/lit/off16/length) and 6 (off32) all
    // point at pipeline_result.frame — the new descriptor produced by
    // l1_unwrap.comp encodes byte offsets INTO that single compressed
    // buffer for every stream. No per-stream materialized buffers
    // anywhere on this path.
    const dst_bind_buf: vk.VkBuffer = if (opts.dst_buffer_override) |b|
        b
    else if (caller_import_active)
        // Caller-import path: kernel writes into dst_b (DEVICE_LOCAL VRAM
        // for full bandwidth); a vkCmdCopyBuffer below stages dst_b into
        // the caller's imported VkBuffer (no host @memcpy after submit).
        dst_b.buf
    else if (direct_write)
        // Single-buffer: kernel writes directly into the host-cached
        // staging buffer (no GPU copy needed).
        dst_stage.buf
    else
        // Two-buffer: kernel writes into dst_b (DEVICE_LOCAL VRAM);
        // a vkCmdCopyBuffer below stages it into dst_stage for
        // host readback.
        dst_b.buf;
    const dec_bindings: [9]vk.VkDescriptorBufferInfo = .{
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
        // S003: WalkMeta (binding 8) — n_chunks self-gate source.
        // Mirrors CUDA's `d_n_chunks_dev = d_walk_meta + walk_meta_offsets
        // .n_chunks` pattern (src/decode/decode_dispatch.zig:614-621).
        // Bound to `n_chunks_scratch` (device-local 1×u32) rather than the
        // host-visible meta buffer; the GPU pipeline copies meta[0] →
        // n_chunks_scratch as part of recordDecodePipelineInto, so the
        // value is identical and lz_decode reads from VRAM at full
        // bandwidth (the sysmem meta buffer would otherwise cost a
        // through-PCIe read per subgroup).
        .{ .buffer = pipeline_result.n_chunks_scratch.buf, .offset = 0, .range = vk.VK_WHOLE_SIZE },
    };
    const dec_set = try descriptors.allocSet(ctx, cached_dec, dec_bindings[0..]);

    last_decode_slz_descset_ns = qpcNs(t_descset_begin, qpcNow());

    // ── S003 single-submit merged dispatch ──────────────────────────
    //
    // Records the entire decode pipeline + l1_unwrap + lz_decode +
    // dst→host copy into ONE command buffer with ONE vkQueueSubmit and
    // ONE vkWaitForFences. Saves one host↔GPU round-trip vs the prior
    // two-submit pattern (~5-8 ms / 95 MB enwik8 on NVIDIA RTX 4060 Ti).
    // lz_decode's self-gate count is read from WalkMeta (binding 8) so
    // the host no longer needs to know n_chunks before recording.
    const t_dispatch_begin = qpcNow();
    const unwrap_push: UnwrapPush = .{ .frame_size = @intCast(slz_bytes.len) };
    var unwrap_push_bytes: [@sizeOf(UnwrapPush)]u8 = undefined;
    @memcpy(unwrap_push_bytes[0..], std.mem.asBytes(&unwrap_push));

    // copy_size selection: 0 for single-buffer (kernel wrote into
    // dst_stage directly) or D2D override; otherwise the worst-case
    // dst_total_max bytes (= out.len). The actual decomp_size is only
    // known post-fence — we over-copy a few bytes when decomp_size <
    // out.len, but those bytes are either tail-zero in the dst_b
    // buffer or pre-existing in the caller's buffer. We trim by using
    // the actual decomp_size for the host @memcpy at the bottom.
    const copy_size: vk.VkDeviceSize = if (use_dst_override or dst_total_max == 0 or direct_write)
        0
    else
        @as(vk.VkDeviceSize, dst_total_max);
    const copy_src: vk.VkBuffer = if (copy_size == 0) null else dst_b.buf;
    const copy_dst: vk.VkBuffer = if (copy_size == 0)
        null
    else if (caller_import_active)
        caller_imported.buf
    else
        dst_stage.buf;

    // Mirror CUDA src/decode/decode_dispatch.zig:400-401 grid sizing
    // (lz_groups / WARPS_PER_BLOCK). With chunks_per_group = 1 and
    // WARPS_PER_BLOCK = 2, the grid_x is ceil(n_chunks / 2). We use
    // `n_chunks_max` here because the host doesn't know the real
    // n_chunks (still living in pipeline_result.meta, read post-fence).
    // The lz_decode shader self-gates on `chunk_id >= walk_meta_buf.w[0]`
    // so extra workgroups beyond real n_chunks early-exit.
    const WARPS_PER_BLOCK: u32 = 2;
    const lz_grid_x: u32 = (n_chunks_max + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    const dec_dispatch_result = try recordAndSubmitMergedDecode(
        ctx,
        &prepared,
        chunks_b,
        .{
            .pipeline = cached_unwrap.pipeline,
            .pipeline_layout = cached_unwrap.pipeline_layout,
            .descriptor_set = unwrap_set,
            .push_constants_bytes = unwrap_push_bytes[0..],
            .group_count = .{ n_chunks_max, 1, 1 },
            .label = "l1_unwrap",
        },
        .{
            .pipeline = cached_dec.pipeline,
            .pipeline_layout = cached_dec.pipeline_layout,
            .descriptor_set = dec_set,
            .push_constants_bytes = &.{},
            .group_count = .{ lz_grid_x, 1, 1 },
            .label = "lz_decode",
        },
        .{
            .src = copy_src,
            .dst = copy_dst,
            .size = copy_size,
        },
    );
    last_decode_slz_dispatch_ns = qpcNs(t_dispatch_begin, qpcNow());

    // ── Post-fence: read meta to get the real n_chunks + decomp_size
    // for status validation + host @memcpy sizing. Meta is host-visible
    // sysmem (mapped pointer); the read is a few cached u32 loads.
    const meta_words: [*]const u32 = if (pipeline_result.meta.mapped) |m|
        @ptrCast(@alignCast(m))
    else
        return error.MapMemoryFailed;
    const actual_n_chunks: u32 = meta_words[decode_pipeline_gpu.WALK_META_N_CHUNKS_SLOT];
    const actual_decomp_size: u32 = meta_words[decode_pipeline_gpu.WALK_META_DECOMP_SIZE_SLOT];
    const actual_status: u32 = meta_words[decode_pipeline_gpu.WALK_META_STATUS_SLOT];
    if (actual_status != 0) return error.BadFrame;
    if (actual_n_chunks == 0) return error.BadFrame;
    if (!use_dst_override and out.len < actual_decomp_size) return error.OutputTooSmall;

    // Unwrap-submit globals stay at zero — the merged submit collapses
    // the previous unwrap-submit's per-call wall into the single
    // recordAndSubmitMergedDecode call above.
    last_decode_slz_unwrap_submit_wait_wall_ns = 0;
    last_decode_slz_unwrap_record_ns = 0;
    last_decode_slz_unwrap_submit_call_ns = 0;
    last_decode_slz_unwrap_wait_call_ns = 0;
    last_decode_slz_unwrap_query_read_ns = 0;
    l1_codec.last_decode_dispatch_ns = dec_dispatch_result.ns;
    last_decode_slz_gpu_kernel_ns = dec_dispatch_result.ns;
    last_decode_slz_gpu_copy_ns = dec_dispatch_result.copy_ns;
    last_decode_slz_submit_wait_wall_ns = dec_dispatch_result.submit_wait_wall_ns;
    last_decode_slz_dec_record_ns = dec_dispatch_result.record_wall_ns;
    last_decode_slz_dec_submit_call_ns = dec_dispatch_result.submit_call_ns;
    last_decode_slz_dec_wait_call_ns = dec_dispatch_result.wait_call_ns;
    last_decode_slz_dec_query_read_ns = dec_dispatch_result.query_read_ns;

    // Host readback only when the host buffer was actually the
    // dst — D2D callers skip it entirely (the bytes are already
    // in their device buffer). Caller-import path also skips: the
    // GPU's copy engine already DMA'd straight into `out` via the
    // imported VkBuffer (the analog of CUDA's cuMemcpyDtoH_v2 at
    // src/decode/decode_dispatch.zig:485). Two-buffer slow path
    // still reads from dst_stage (sysmem, driver-cached).
    const t_readback_begin = qpcNow();
    if (!use_dst_override and actual_decomp_size > 0 and !caller_import_active) {
        const stage_mapped = dst_stage.mapped orelse return error.MapMemoryFailed;
        @memcpy(out[0..actual_decomp_size], stage_mapped[0..actual_decomp_size]);
    }
    last_decode_slz_readback_ns = qpcNs(t_readback_begin, qpcNow());

    return actual_decomp_size;
}

/// S003: record the GPU decode pipeline + l1_unwrap + lz_decode +
/// dst→host copy into ONE command buffer; submit; wait. Replaces the
/// pre-S003 split of (runDecodePipelineEx fence wait) + (dispatch
/// .submitTwoWithCopyLabeled fence wait) so the host pays ONE
/// vkQueueSubmit + ONE vkWaitForFences per decode call.
///
/// Mirrors the CUDA pattern at src/decode/decode_dispatch.zig:614-621:
/// the back-half LZ kernel reads its self-gate count from device-
/// resident meta memory — no host round-trip between phases.
fn recordAndSubmitMergedDecode(
    ctx: *driver.Context,
    prepared: *decode_pipeline_gpu.PreparedDecodePipeline,
    chunks_b: l1_codec.Buffer,
    unwrap_spec: dispatch.DispatchSpec,
    dec_spec: dispatch.DispatchSpec,
    copy: dispatch.CopyOp,
) dispatch.DispatchError!dispatch.DispatchResult {
    try dispatch.ensureChassisPub(ctx);

    const reset_cb = vk.vkResetCommandBuffer_fn orelse return error.LoaderNotReady;
    const begin_cb = vk.vkBeginCommandBuffer_fn orelse return error.LoaderNotReady;
    const end_cb = vk.vkEndCommandBuffer_fn orelse return error.LoaderNotReady;
    const cmd_reset_qp = vk.vkCmdResetQueryPool_fn orelse return error.LoaderNotReady;
    const cmd_write_ts = vk.vkCmdWriteTimestamp_fn orelse return error.LoaderNotReady;
    const cmd_bind_pl = vk.vkCmdBindPipeline_fn orelse return error.LoaderNotReady;
    const cmd_bind_ds = vk.vkCmdBindDescriptorSets_fn orelse return error.LoaderNotReady;
    const cmd_dispatch = vk.vkCmdDispatch_fn orelse return error.LoaderNotReady;
    const cmd_copy = vk.vkCmdCopyBuffer_fn orelse return error.LoaderNotReady;
    const cmd_barrier = vk.vkCmdPipelineBarrier_fn orelse return error.LoaderNotReady;
    const reset_fence = vk.vkResetFences_fn orelse return error.LoaderNotReady;
    const submit_fn = vk.vkQueueSubmit_fn orelse return error.LoaderNotReady;
    const wait_fence = vk.vkWaitForFences_fn orelse return error.LoaderNotReady;
    const get_results = vk.vkGetQueryPoolResults_fn orelse return error.LoaderNotReady;

    const t_record_begin = qpcNow();
    if (reset_cb(ctx.cmd_buf, 0) != vk.VK_SUCCESS) return error.ResetCommandBufferFailed;

    const begin_info: vk.VkCommandBufferBeginInfo = .{
        .flags = vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (begin_cb(ctx.cmd_buf, &begin_info) != vk.VK_SUCCESS) return error.BeginCommandBufferFailed;

    cmd_reset_qp(ctx.cmd_buf, ctx.query_pool, 0, dispatch.TS_SLOT_COUNT);

    // ── Phase 1: GPU decode pipeline (frame_staging→frame DMA + walk_
    //    frame + prefix_sum + scan_parse + 4× compact_huff + compact_raw
    //    + gather_raw_off16). recordDecodePipelineInto leaves a
    //    SHADER_WRITE→SHADER_READ barrier on COMPUTE_SHADER at its end. ─
    decode_pipeline_gpu.recordDecodePipelineInto(ctx, ctx.cmd_buf, prepared.*) catch |err| switch (err) {
        error.LoaderNotReady => return error.LoaderNotReady,
        // recordDecodePipelineInto can only fail with LoaderNotReady or
        // OutOfMemory (latter only if workspace re-init races, which
        // can't happen here since `prepareDecodePipeline` already ran).
        // Fold every other variant into LoaderNotReady so this function
        // can stay in dispatch.DispatchError without leaking pipeline-
        // specific error variants.
        else => return error.LoaderNotReady,
    };

    // ── Phase 2: l1_unwrap (untimed — fast, the legacy code didn't
    //    time it either when riding submitTwoWithCopy). ─
    cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, unwrap_spec.pipeline);
    {
        const sets: [1]vk.VkDescriptorSet = .{unwrap_spec.descriptor_set};
        cmd_bind_ds(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, unwrap_spec.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
    }
    if (unwrap_spec.push_constants_bytes.len > 0) {
        const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
        cmd_push(ctx.cmd_buf, unwrap_spec.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(unwrap_spec.push_constants_bytes.len), @ptrCast(unwrap_spec.push_constants_bytes.ptr));
    }
    cmd_dispatch(ctx.cmd_buf, unwrap_spec.group_count[0], unwrap_spec.group_count[1], unwrap_spec.group_count[2]);

    // Inter-dispatch barrier on chunks_b (l1_unwrap writes; lz_decode
    // reads at binding 5). Also covers the WalkMeta read at lz_decode's
    // binding 8 — the global compute→compute SHADER_WRITE→SHADER_READ
    // mb from recordDecodePipelineInto's tail already covered the
    // meta visibility, but a per-buffer chunks_b barrier here is the
    // minimal correct sync for the new chunks descriptor.
    {
        const bbarrier: vk.VkBufferMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_SHADER_READ_BIT,
            .srcQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .buffer = chunks_b.buf,
            .offset = 0,
            .size = vk.VK_WHOLE_SIZE,
        };
        const bbarriers: [1]vk.VkBufferMemoryBarrier = .{bbarrier};
        cmd_barrier(
            ctx.cmd_buf,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            null,
            1,
            @ptrCast(&bbarriers),
            0,
            null,
        );
    }

    // ── Phase 3: lz_decode (timestamp-bracketed for kernel ns) ─
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, dispatch.TS_SLOT_BEGIN);
    cmd_bind_pl(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, dec_spec.pipeline);
    {
        const sets: [1]vk.VkDescriptorSet = .{dec_spec.descriptor_set};
        cmd_bind_ds(ctx.cmd_buf, vk.VK_PIPELINE_BIND_POINT_COMPUTE, dec_spec.pipeline_layout, 0, 1, @ptrCast(&sets), 0, null);
    }
    if (dec_spec.push_constants_bytes.len > 0) {
        const cmd_push = vk.vkCmdPushConstants_fn orelse return error.LoaderNotReady;
        cmd_push(ctx.cmd_buf, dec_spec.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, @intCast(dec_spec.push_constants_bytes.len), @ptrCast(dec_spec.push_constants_bytes.ptr));
    }
    cmd_dispatch(ctx.cmd_buf, dec_spec.group_count[0], dec_spec.group_count[1], dec_spec.group_count[2]);
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, dispatch.TS_SLOT_END);

    // ── Phase 4: dst→host copy (compute→transfer barrier, skipped when
    //    copy.size == 0). The COPY_BEGIN/END timestamps are written
    //    regardless so the query read below has defined values. ─
    const has_copy = copy.size != 0;
    if (has_copy) {
        const cbarrier: vk.VkBufferMemoryBarrier = .{
            .srcAccessMask = vk.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.VK_ACCESS_TRANSFER_READ_BIT,
            .srcQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = vk.VK_QUEUE_FAMILY_IGNORED,
            .buffer = copy.src,
            .offset = copy.src_offset,
            .size = copy.size,
        };
        const cbarriers: [1]vk.VkBufferMemoryBarrier = .{cbarrier};
        cmd_barrier(
            ctx.cmd_buf,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            null,
            1,
            @ptrCast(&cbarriers),
            0,
            null,
        );
    }

    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ctx.query_pool, dispatch.TS_SLOT_COPY_BEGIN);
    if (has_copy) {
        const region: vk.VkBufferCopy = .{
            .srcOffset = copy.src_offset,
            .dstOffset = copy.dst_offset,
            .size = copy.size,
        };
        const regions: [1]vk.VkBufferCopy = .{region};
        cmd_copy(ctx.cmd_buf, copy.src, copy.dst, 1, @ptrCast(&regions));
    }
    cmd_write_ts(ctx.cmd_buf, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ctx.query_pool, dispatch.TS_SLOT_COPY_END);

    if (end_cb(ctx.cmd_buf) != vk.VK_SUCCESS) return error.EndCommandBufferFailed;
    const record_ns = qpcNs(t_record_begin, qpcNow());

    // ── Single submit + single wait ─
    const fences: [1]vk.VkFence = .{ctx.fence};
    if (reset_fence(ctx.dev, 1, @ptrCast(&fences)) != vk.VK_SUCCESS) return error.ResetFenceFailed;
    const cmd_bufs: [1]vk.VkCommandBuffer = .{ctx.cmd_buf};
    const submit_info: vk.VkSubmitInfo = .{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cmd_bufs),
    };
    const submits: [1]vk.VkSubmitInfo = .{submit_info};
    const t_submit_begin = qpcNow();
    if (submit_fn(ctx.queue, 1, @ptrCast(&submits), ctx.fence) != vk.VK_SUCCESS) return error.SubmitFailed;
    const t_submit_end = qpcNow();
    const submit_call_ns = qpcNs(t_submit_begin, t_submit_end);

    const wait_result = wait_fence(ctx.dev, 1, @ptrCast(&fences), vk.VK_TRUE, vk.VK_M8A_FENCE_WAIT_NS);
    const t_wait_end = qpcNow();
    const wait_call_ns = qpcNs(t_submit_end, t_wait_end);
    const submit_wait_ns = qpcNs(t_submit_begin, t_wait_end);
    if (wait_result == vk.VK_TIMEOUT) return error.FenceWaitTimeout;
    if (wait_result != vk.VK_SUCCESS) return error.FenceWaitFailed;

    var ts: [dispatch.TS_SLOT_COUNT]u64 = .{ 0, 0, 0, 0 };
    const t_query_begin = qpcNow();
    const res = get_results(
        ctx.dev,
        ctx.query_pool,
        0,
        dispatch.TS_SLOT_COUNT,
        @sizeOf(@TypeOf(ts)),
        @ptrCast(&ts),
        @sizeOf(u64),
        vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
    );
    const query_read_ns = qpcNs(t_query_begin, qpcNow());
    if (res != vk.VK_SUCCESS) return error.QueryReadFailed;

    const delta_ticks: u64 = if (ts[dispatch.TS_SLOT_END] >= ts[dispatch.TS_SLOT_BEGIN])
        ts[dispatch.TS_SLOT_END] - ts[dispatch.TS_SLOT_BEGIN]
    else
        0;
    const period = if (ctx.timestamp_period_ns > 0.0) ctx.timestamp_period_ns else 1.0;
    const ns_f: f64 = @as(f64, @floatFromInt(delta_ticks)) * @as(f64, period);
    const ns: u64 = if (ns_f <= 0.0) 0 else @intFromFloat(ns_f);

    const copy_ticks: u64 = if (ts[dispatch.TS_SLOT_COPY_END] >= ts[dispatch.TS_SLOT_COPY_BEGIN])
        ts[dispatch.TS_SLOT_COPY_END] - ts[dispatch.TS_SLOT_COPY_BEGIN]
    else
        0;
    const copy_ns_f: f64 = @as(f64, @floatFromInt(copy_ticks)) * @as(f64, period);
    const copy_ns: u64 = if (copy_ns_f <= 0.0) 0 else @intFromFloat(copy_ns_f);

    return .{
        .ns = ns,
        .copy_ns = copy_ns,
        .submit_wait_wall_ns = submit_wait_ns,
        .record_wall_ns = record_ns,
        .submit_call_ns = submit_call_ns,
        .wait_call_ns = wait_call_ns,
        .query_read_ns = query_read_ns,
    };
}
