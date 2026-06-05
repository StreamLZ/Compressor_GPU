//! 1:1 port of src/decode/decode_dispatch.zig.
//!
//! Top-level decode dispatcher. Owns the L2 gate (per project
//! EXCEPTION 2): for `req.level == 1` decodes the dispatcher must skip
//! the Huff/scan/compact/merge/gather paths and run the L1 raw
//! `slzLzDecodeRawKernel` directly off the host-computed chunk descs +
//! prefix sum.

const std = @import("std");
const vk = @import("vulkan_api.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const module_loader = @import("module_loader.zig");
const scan_gpu = @import("scan_gpu.zig");

const VkDeviceBuffer = vk.VkDeviceBuffer;
const VkStream = vk.VkStream;
const VkResult = vk.VkResult;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;

const ChunkDesc = descriptors.ChunkDesc;
const HuffDecChunkDesc = descriptors.HuffDecChunkDesc;
const RawOff16Desc = descriptors.RawOff16Desc;
const ScanResult = descriptors.ScanResult;
const GpuError = descriptors.GpuError;
const vkCall = descriptors.vkCall;
const HUFF_LUT_ENTRIES = descriptors.HUFF_LUT_ENTRIES;

const DecodeContext = decode_context.DecodeContext;
const ensureDeviceBuf = decode_context.ensureDeviceBuf;
const ensureDeviceOutput = decode_context.ensureDeviceOutput;
const beginKernelTiming = decode_context.beginKernelTiming;
const endKernelTiming = decode_context.endKernelTiming;
const finalizeProfiling = decode_context.finalizeProfiling;

// ── Per-phase host-overhead profiling (SLZ_VK_PROFILE_PHASES=1) ──────
// Accumulates wall-clock QPC time spent in each logical sub-phase of
// fullGpuLaunchImpl across calls. Read+printed by bench_decompress
// via printAndResetPhaseProfile(). Orthogonal to SLZ_VK_PROFILE_DECODE
// (per-kernel GPU timestamps): this captures HOST overhead between
// vk_proc.* calls so we can localize where the 1.6x e2e gap comes from.
pub var g_phase_profile_enabled: bool = false;
pub var g_phase_decode_count: u64 = 0;
pub var g_phase_upload_ns: i64 = 0;          // entire uploadInputAndPrefixSum
pub var g_phase_upload_h2d_ns: i64 = 0;      // just the h2d_async() calls
pub var g_phase_upload_prefixsum_ns: i64 = 0; // gpuPrefixSumChunksImpl
pub var g_phase_upload_misc_ns: i64 = 0;     // everything else (ensureDeviceBuf etc)
pub var g_phase_backhalf_ns: i64 = 0;        // entire runBackHalf
pub var g_phase_backhalf_h2d_count_ns: i64 = 0; // 4B n_chunks h2d
pub var g_phase_backhalf_lz_launch_ns: i64 = 0; // runLzPipeline (just the launch call)
pub var g_phase_backhalf_fence_wait_ns: i64 = 0; // end-of-back-half stream_sync
pub var g_phase_finalize_ns: i64 = 0;        // finalizeOutput + post-finalize sync
pub var g_phase_finalize_d2h_ns: i64 = 0;    // just the d2h_async() call
pub var g_phase_finalize_sync_ns: i64 = 0;   // post-finalize stream_sync (flushes deferred D2H memcpy)
pub var g_phase_resolve_ns: i64 = 0;         // VkProcs.resolve()
pub var g_phase_other_ns: i64 = 0;           // everything else inside fullGpuLaunchImpl
// Iter 8 subfix 3: prepareImportHostBuffer time (moved out of finalize
// so it overlaps with runBackHalf on the GPU). Reported separately so
// we can verify subfix 3 is actually overlapping (the value here is
// the would-have-been serialized cost; it's still spent — just not on
// the finalize critical path).
pub var g_phase_import_prep_ns: i64 = 0;

pub fn phaseProfileInit() void {
    g_phase_profile_enabled = std.c.getenv("SLZ_VK_PROFILE_PHASES") != null;
    g_phase_decode_count = 0;
    g_phase_upload_ns = 0;
    g_phase_upload_h2d_ns = 0;
    g_phase_upload_prefixsum_ns = 0;
    g_phase_upload_misc_ns = 0;
    g_phase_backhalf_ns = 0;
    g_phase_backhalf_h2d_count_ns = 0;
    g_phase_backhalf_lz_launch_ns = 0;
    g_phase_backhalf_fence_wait_ns = 0;
    g_phase_finalize_ns = 0;
    g_phase_finalize_d2h_ns = 0;
    g_phase_finalize_sync_ns = 0;
    g_phase_resolve_ns = 0;
    g_phase_other_ns = 0;
    g_phase_import_prep_ns = 0;
}

inline fn qpcNs() i64 {
    // Convert QPC ticks to ns via qpcMs (ms * 1e6). Stable across threads.
    const t = vk.qpcNow();
    return t;
}
inline fn qpcDeltaNs(start: i64, end: i64) i64 {
    // Use qpcMs to ticks-to-ns: ms = (end-start)*1000/freq → ns = ms*1e6.
    // Avoid a second freq query by inlining: qpcMs is f64 already.
    return @intFromFloat(vk.qpcMs(start, end) * 1e6);
}

pub fn printAndResetPhaseProfile(w: anytype) void {
    if (!g_phase_profile_enabled or g_phase_decode_count == 0) return;
    const n = @as(f64, @floatFromInt(g_phase_decode_count));
    const ns_to_ms_per = struct {
        fn f(total_ns: i64, count: f64) f64 {
            return (@as(f64, @floatFromInt(total_ns)) / 1e6) / count;
        }
    }.f;
    w.print("phase: decode_count            {d}\n", .{g_phase_decode_count}) catch {};
    w.print("phase: VkProcs.resolve         {d:.4} ms/decode\n", .{ns_to_ms_per(g_phase_resolve_ns, n)}) catch {};
    w.print("phase: upload_total            {d:.4} ms/decode\n", .{ns_to_ms_per(g_phase_upload_ns, n)}) catch {};
    w.print("phase:   upload_h2d            {d:.4} ms/decode  (chunk_descs+comp_input)\n", .{ns_to_ms_per(g_phase_upload_h2d_ns, n)}) catch {};
    w.print("phase:   upload_prefixsum      {d:.4} ms/decode  (gpuPrefixSumChunksImpl)\n", .{ns_to_ms_per(g_phase_upload_prefixsum_ns, n)}) catch {};
    w.print("phase:   upload_misc           {d:.4} ms/decode  (ensureDeviceBuf+sizing)\n", .{ns_to_ms_per(g_phase_upload_misc_ns, n)}) catch {};
    w.print("phase: backhalf_total          {d:.4} ms/decode\n", .{ns_to_ms_per(g_phase_backhalf_ns, n)}) catch {};
    w.print("phase:   backhalf_h2d_count    {d:.4} ms/decode  (4B n_chunks)\n", .{ns_to_ms_per(g_phase_backhalf_h2d_count_ns, n)}) catch {};
    w.print("phase:   backhalf_lz_launch    {d:.4} ms/decode  (runLzPipeline launch)\n", .{ns_to_ms_per(g_phase_backhalf_lz_launch_ns, n)}) catch {};
    w.print("phase:   backhalf_fence_wait   {d:.4} ms/decode  (vkWaitForFences)\n", .{ns_to_ms_per(g_phase_backhalf_fence_wait_ns, n)}) catch {};
    w.print("phase: finalize_total          {d:.4} ms/decode\n", .{ns_to_ms_per(g_phase_finalize_ns, n)}) catch {};
    w.print("phase:   finalize_d2h          {d:.4} ms/decode  (d2h_async submit)\n", .{ns_to_ms_per(g_phase_finalize_d2h_ns, n)}) catch {};
    w.print("phase:   finalize_sync         {d:.4} ms/decode  (stream_sync drain)\n", .{ns_to_ms_per(g_phase_finalize_sync_ns, n)}) catch {};
    w.print("phase: import_prep             {d:.4} ms/decode  (subfix 3: overlapped with back-half; cache_hits={d}, misses={d})\n", .{
        ns_to_ms_per(g_phase_import_prep_ns, n),
        module_loader.g_import_prep_cache_hits,
        module_loader.g_import_prep_cache_misses,
    }) catch {};
    w.print("phase: other                   {d:.4} ms/decode\n", .{ns_to_ms_per(g_phase_other_ns, n)}) catch {};
    const totalled =
        g_phase_resolve_ns + g_phase_upload_ns + g_phase_backhalf_ns +
        g_phase_finalize_ns + g_phase_other_ns;
    w.print("phase: SUM                     {d:.4} ms/decode\n", .{ns_to_ms_per(totalled, n)}) catch {};
    phaseProfileInit();
}

// Local function-pointer typedefs mirroring CUDA's FnMemcpyHtoD /
// FnMemcpyDtoH / FnLaunchKernel / FnCtxSync / FnStreamSync /
// FnMemcpyDtoDAsync. Substitute CUdeviceptr → VkDeviceBuffer and
// CUstream → VkStream; argument order preserved.
const FnH2D = *const fn (VkDeviceBuffer, *const anyopaque, usize) callconv(.c) VkResult;
const FnD2H = *const fn (*anyopaque, VkDeviceBuffer, usize) callconv(.c) VkResult;
const FnH2DAsync = *const fn (VkDeviceBuffer, *const anyopaque, usize, VkStream) callconv(.c) VkResult;
const FnD2HAsync = *const fn (*anyopaque, VkDeviceBuffer, usize, VkStream) callconv(.c) VkResult;
const FnLaunchKernel = *const fn (
    usize,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    VkStream,
    [*]?*anyopaque,
    [*]?*anyopaque,
) callconv(.c) VkResult;
const FnCtxSync = *const fn () callconv(.c) VkResult;
const FnStreamSync = *const fn (VkStream) callconv(.c) VkResult;
const FnD2D = *const fn (VkDeviceBuffer, VkDeviceBuffer, usize, VkStream) callconv(.c) VkResult;

/// CUDA reference: src/decode/decode_dispatch.zig:47-65. Renamed from
/// CudaProcs per Section B. Bundle of the function-pointer slots the
/// pipeline uses, resolved once at fullGpuLaunchImpl entry and threaded
/// into the pipeline helpers (`runHuffBuildAndDecode`, `runLzPipeline`,
/// `finalizeOutput`) so they don't each re-resolve the same
/// `vk.procs.h2d orelse return error.BackendNotAvailable` patterns.
/// `d2d` is optional because only the D2D source and D2D output paths
/// consult it; each site unwraps with its own `orelse return
/// error.BackendNotAvailable`.
///
/// Field NAMES stay verbatim from CUDA's CudaProcs (h2d / d2h / launch /
/// sync / stream_sync / d2d) so call sites read identical across the
/// two ports.
pub const VkProcs = struct {
    h2d: FnH2D,
    d2h: FnD2H,
    h2d_async: ?FnH2DAsync,
    d2h_async: ?FnD2HAsync,
    launch: FnLaunchKernel,
    sync: FnCtxSync,
    stream_sync: FnStreamSync,
    d2d: ?FnD2D,

    fn resolve() GpuError!VkProcs {
        return .{
            .h2d = vk.procs.h2d orelse return error.BackendNotAvailable,
            .d2h = vk.procs.d2h orelse return error.BackendNotAvailable,
            // Iter 4: h2d_async/d2h_async are populated by module_loader
            // (see :1407-1408) and required by the single-submit batching
            // path; null fallback keeps the codec working against legacy
            // backends that didn't expose them.
            .h2d_async = vk.procs.h2d_async,
            .d2h_async = vk.procs.d2h_async,
            .launch = vk.procs.launch_kernel orelse return error.BackendNotAvailable,
            .sync = vk.procs.ctx_sync orelse return error.BackendNotAvailable,
            .stream_sync = vk.procs.stream_sync orelse return error.BackendNotAvailable,
            .d2d = vk.procs.d2d,
        };
    }
};

/// CUDA reference: src/decode/decode_dispatch.zig:70-78. The six params
/// shared by both LZ-decode kernel variants (raw and general). Held as
/// a struct so each field has a stable address for the kernel-params
/// array (which takes `&field` pointers).
const LzCommonParams = struct {
    comp: VkDeviceBuffer,
    descs_dev: VkDeviceBuffer,
    dst: VkDeviceBuffer,
    chunks_per_group: u32,
    /// Device pointer to the LZ kernel's self-gate count.
    total: VkDeviceBuffer,
    sub_chunk_cap: u32,
};

/// CUDA reference: src/decode/decode_dispatch.zig:83-96.
/// `slzMergeHuffDescsKernel` parameter bundle.
const MergeHuffParams = struct {
    lit: u64,
    tok: u64,
    hi: u64,
    lo: u64,
    n_lit: u64,
    n_tok: u64,
    n_hi: u64,
    n_lo: u64,
    tok_region: u32,
    off16_region: u32,
    merged: u64,
    n_merged: u64,
};

/// CUDA reference: src/decode/decode_dispatch.zig:105-140. Pre-decode
/// preparation: combine the four per-stream-type Huffman descriptor
/// arrays into one merged device array in `self.d_huff_descs`, with
/// out_offsets adjusted by the per-stream-type region offset and
/// lut_offsets assigned sequentially. Driven entirely by the GPU merge
/// kernel: the compact kernels in `gpuScanChunks` already wrote the
/// four per-stream device buffers (`d_compact_*`), and the merge kernel
/// reads its per-stream counts from `d_compact_counts` and self-gates.
pub fn mergeHuffDescs(
    self: *DecodeContext,
    tok_offset: usize,
    off16_offset: usize,
    launch_fn: FnLaunchKernel,
) GpuError!void {
    var args = MergeHuffParams{
        .lit = self.d_compact_lit,
        .tok = self.d_compact_tok,
        .hi = self.d_compact_hi,
        .lo = self.d_compact_lo,
        .n_lit = self.d_compact_counts,
        .n_tok = self.d_compact_counts + 4,
        .n_hi = self.d_compact_counts + 8,
        .n_lo = self.d_compact_counts + 12,
        .tok_region = @intCast(tok_offset),
        .off16_region = @intCast(off16_offset),
        .merged = self.d_huff_descs,
        .n_merged = self.d_compact_counts + 20,
    };
    var m_params = [_]?*anyopaque{
        @ptrCast(&args.lit),        @ptrCast(&args.tok),          @ptrCast(&args.hi),       @ptrCast(&args.lo),
        @ptrCast(&args.n_lit),      @ptrCast(&args.n_tok),        @ptrCast(&args.n_hi),     @ptrCast(&args.n_lo),
        @ptrCast(&args.tok_region), @ptrCast(&args.off16_region),
        @ptrCast(&args.merged),     @ptrCast(&args.n_merged),
    };
    var m_extra = [_]?*anyopaque{null};
    const stream = self.work_stream;
    const t_merge = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzMergeHuffDescsKernel", stream);
    try vkCall(launch_fn(module_loader.merge_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, stream, &m_params, &m_extra), .launch);
    endKernelTiming(t_merge, stream);
    // No post-launch sync: downstream kernels (huff_build, huff_decode)
    // are queued on the same stream and see merge's output via stream
    // ordering. In sync mode the pre-back-half cross-stream sync in
    // `fullGpuLaunchImpl` covers it.
}

/// CUDA reference: src/decode/decode_dispatch.zig:148-180. Scatter the
/// raw (type 0) off16 sub-streams from `d_comp_persist` into the
/// per-sub-chunk entropy scratch. All inputs are already device-
/// resident: the descriptor list lives in `d_compact_raw`, the count in
/// `d_compact_counts[4]`, and the source bytes in `d_comp_persist`. A
/// single `slzGatherRawOff16Kernel` launch copies every stream in
/// parallel.
pub fn gatherRawOff16(
    self: *DecodeContext,
    scan: ScanResult,
    comp_len: u32,
    launch_fn: FnLaunchKernel,
) GpuError!void {
    // Defensive guard: `num_raw_off16` is unconditionally set to
    // `total_subchunks * 2` by `gpuScanChunks`, and `total_subchunks
    // == 0` is filtered earlier in the dispatch. The zero-check is
    // kept against a future scan implementation that does report real
    // raw counts; today it never fires.
    if (scan.num_raw_off16 == 0) return;
    var p_comp: u64 = self.d_comp_persist;
    var p_comp_len: u32 = comp_len;
    var p_scratch: u64 = self.d_entropy_off16_scratch;
    var p_descs: u64 = self.d_compact_raw;
    var p_count: u64 = self.d_compact_counts + 16;
    var params = [_]?*anyopaque{
        @ptrCast(&p_comp), @ptrCast(&p_comp_len), @ptrCast(&p_scratch),
        @ptrCast(&p_descs), @ptrCast(&p_count),
    };
    var extra = [_]?*anyopaque{null};
    // Grid size = worst-case sub-chunk count (`num_raw_off16` is the
    // dispatch's upper bound); the kernel self-gates on `*p_count` so
    // over-launching is safe.
    const grid_x: u32 = scan.num_raw_off16;
    const stream = self.work_stream;
    const t_gather = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzGatherRawOff16Kernel", stream);
    defer endKernelTiming(t_gather, stream);
    try vkCall(launch_fn(module_loader.gather_off16_fn, grid_x, 1, 1, 256, 1, 1, 0, stream, &params, &extra), .launch);
    // No post-launch sync: downstream LZ kernel reads
    // d_entropy_off16_scratch from the same stream.
}

/// CUDA reference: src/decode/decode_dispatch.zig:185-200. SLZ_E2E_TIMER
/// cumulative timestamps, each measured as elapsed nanoseconds since
/// `t0`. Populated incrementally between phases of `fullGpuLaunchImpl`
/// so `emitE2eTrace` can recover per-phase deltas.
const E2eCumulative = struct {
    h2d: i64 = 0,
    postscan: i64 = 0,
    scan: i64 = 0,
    predh: i64 = 0,
};

/// Nanoseconds elapsed between `t0` and now. `t0` is a
/// `std.Io.Clock.Awake` reading captured at function entry; the
/// `anytype` is because the Clock reading type is not directly
/// nameable at the use site.
inline fn nsSince(t0: anytype, io: std.Io) i64 {
    return @intCast(t0.untilNow(io, .awake).toNanoseconds());
}

inline fn nsToMs(ns: i64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Iter 4 host-perf helper: prefer h2d_async on the supplied stream so
/// the dispatcher batches all H2D copies + kernel launches + the D2H
/// readback into a single per-stream cmdbuf + single vkQueueSubmit +
/// single vkWaitForFences per decoded block. Falls back to the sync
/// h2d slot when no h2d_async is available (legacy backends) or when
/// stream == 0 (caller explicitly opted out of batching).
inline fn h2dRoute(procs: *const VkProcs, stream: VkStream, dst: VkDeviceBuffer, src: *const anyopaque, size: usize) VkResult {
    if (stream != 0) {
        if (procs.h2d_async) |f| return f(dst, src, size, stream);
    }
    return procs.h2d(dst, src, size);
}

/// Iter 4 host-perf helper (paired with h2dRoute). Routes D2H through
/// d2h_async on the supplied stream when available. The staging → host
/// memcpy is deferred to the next streamEndAndWait so the caller MUST
/// invoke procs.stream_sync(stream) before reading the destination
/// pointer — fullGpuLaunchImpl's back-half sync (runBackHalf line 665)
/// covers this for the dispatcher's only D2H caller (finalizeOutput).
inline fn d2hRoute(procs: *const VkProcs, stream: VkStream, dst: *anyopaque, src: VkDeviceBuffer, size: usize) VkResult {
    if (stream != 0) {
        if (procs.d2h_async) |f| return f(dst, src, size, stream);
    }
    return procs.d2h(dst, src, size);
}

/// CUDA reference: src/decode/decode_dispatch.zig:216-241. SLZ_E2E_TIMER
/// per-phase breakdown print at the end of the decode call. `t0` is the
/// start clock reading captured at function entry.
pub fn emitE2eTrace(
    t0: anytype,
    io: std.Io,
    cum: E2eCumulative,
    last_kernel: i64,
) void {
    const cum_end_ns: i64 = nsSince(t0, io);
    // After the phase extraction the `preBlk` slice is empty (the
    // prefix-sum kernel queue moved into `uploadInputAndPrefixSum`,
    // so there is no code between the `h2d` and `postscan` boundaries).
    // Keep the column for trace-format stability; it reads as 0.000.
    const preblk_ns: i64 = 0;
    const scanfn_ns = cum.postscan - cum.h2d;
    const rawcopy_ns = cum.scan - cum.postscan;
    const prep_ns = (cum.predh - cum.scan) - last_kernel;
    std.debug.print("  [e2e] setup+H2D {d:.3}  preBlk {d:.3}  scanFn {d:.3}  rawD2D {d:.3}  prep {d:.3}  kernels {d:.3}  D2H {d:.3}  total {d:.3} ms\n", .{
        nsToMs(cum.h2d),
        nsToMs(preblk_ns),
        nsToMs(scanfn_ns),
        nsToMs(rawcopy_ns),
        nsToMs(prep_ns),
        nsToMs(last_kernel),
        nsToMs(cum_end_ns - cum.predh),
        nsToMs(cum_end_ns),
    });
}

/// CUDA reference: src/decode/decode_dispatch.zig:254-290. Per-call
/// request bundle. The VK port adds a `level: u8` field so the L2 gate
/// inside `fullGpuLaunchImpl` fires per request (audit Section C.5.1
/// commentary).
///
/// `d_compressed_src` (pure-D2D source): when non-null the compressed
/// block is already device-resident at this address; the source is
/// D2D-copied into `d_comp_persist` (no PCIe). `compressed_block.ptr`
/// is then unused (caller may pass an undefined slice with the
/// correct `.len`).
pub const DecodeRequest = struct {
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    io: ?std.Io,
    d_output_target: ?u64,
    d_compressed_src: ?u64,
    /// When non-null, the chunk descs already live on the device at
    /// this address (`chunk_descs.len * sizeof(ChunkDesc)` bytes) and
    /// the H2D from the host slice is skipped. Used by
    /// `decompressFramedFromDevice` to reuse the walk-kernel's
    /// `d_walk_chunks` directly.
    d_chunk_descs_override: ?u64 = null,
    /// When non-null, the LZ kernel reads its self-gate count
    /// (`*p_total`) directly from this device address. Used by
    /// `decompressFramedFromDevice` to point at
    /// `d_walk_meta + walk_meta_offsets.n_chunks` so the walk kernel's
    /// output is consumed by the LZ kernel via stream ordering with no
    /// host round-trip. Null preserves the CLI / host-bounce behavior
    /// (`total_chunks` H2D'd into `d_n_groups_scratch`).
    d_n_chunks_dev: ?u64 = null,
    /// VK port addition (audit Section C.5.1): explicit level so the L2
    /// gate inside `fullGpuLaunchImpl` fires per request rather than per
    /// loaded-module-set.
    level: u8 = 1,

    /// True when the LZ kernel can write decompressed bytes straight
    /// into the caller's device buffer (no `dst_start_off` prefix to
    /// splice in, and a `d_output_target` was supplied). On this path
    /// the back half routes the LZ kernel's `dst` to `d_output_target`
    /// and the finalize phase skips the D2D copy that would otherwise
    /// stage through `self.d_output`. Used by `runBackHalf` and
    /// `fullGpuLaunchImpl` to keep the predicate in one place.
    pub fn writesDirectlyToTarget(self: DecodeRequest) bool {
        return self.d_output_target != null and self.dst_start_off == 0;
    }
};

/// CUDA reference: src/decode/decode_dispatch.zig:300-358. Launches the
/// Huffman LUT-build kernel and the 4-stream decode kernel into
/// `heavy_stream`. Both kernels self-gate on `*d_n_huff`, so it is safe
/// to launch them with the worst-case chunk count even if some chunks
/// have no Huffman streams.
///
/// L2 stub: Phase 7 mirrors the CUDA orchestration but the underlying
/// huff-build / huff-decode kernels are L2-only (Phase 9). Returns
/// `error.NotImplementedL2`; never invoked on the L1 path because the
/// caller gates on `have_huff` which is false when `huff_build_fn == 0`.
pub fn runHuffBuildAndDecode(
    self: *DecodeContext,
    procs: *const VkProcs,
    n_huff: u32,
    heavy_stream: usize,
    split_timer: bool,
    t_huff_start: anytype,
    io: ?std.Io,
) GpuError!i64 {
    _ = self;
    _ = procs;
    _ = n_huff;
    _ = heavy_stream;
    _ = split_timer;
    _ = t_huff_start;
    _ = io;
    return error.NotImplementedL2;
}

/// CUDA reference: src/decode/decode_dispatch.zig:366-465. Launches the
/// LZ decode kernel. Uses the lean raw kernel when no Huffman streams
/// are present (L1/L2 path); otherwise the general kernel that consumes
/// `d_entropy_scratch`.
///
/// Returns the kernel's elapsed nanoseconds when `split_timer` is set;
/// otherwise zero.
pub fn runLzPipeline(
    self: *DecodeContext,
    procs: *const VkProcs,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    n_huff: u32,
    total_chunks: u32,
    total_subchunks: u32,
    heavy_stream: usize,
    split_timer: bool,
    io: ?std.Io,
    /// Device address holding the self-gate count. On the D2D path
    /// this is `d_walk_meta + offset(n_chunks)` (no H2D round-trip);
    /// otherwise `self.d_n_groups_scratch` after the host-staging H2D
    /// in `fullGpuLaunchImpl`.
    lz_total_count_dev: VkDeviceBuffer,
    /// Base device address the LZ kernel writes decompressed bytes
    /// into. Normally `self.d_output`; on the direct-write fast path
    /// the caller routes `req.d_output_target` here so the kernel
    /// skips the finalize D2D copy.
    lz_dst_base_dev: VkDeviceBuffer,
) GpuError!i64 {
    const launch_fn = procs.launch;
    const stream_sync_fn = procs.stream_sync;
    const stream = heavy_stream;

    var split_lz_ns: i64 = 0;
    if (total_chunks == 0) return split_lz_ns;

    const t_lz_start = if (split_timer)
        if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
    else
        null;

    const lz_groups = (total_chunks + chunks_per_group - 1) / chunks_per_group;
    const lz_grid_x = (lz_groups + 1) / 2;

    // Fast path: no entropy in this scan → use lean L1/L2 raw kernel.
    // Huffman literals require the general kernel (it reads entropy_scratch).
    const use_raw_kernel = n_huff == 0 and module_loader.kernel_raw_fn != 0;

    // The six shared params (comp, descs_dev, dst, cpg, total, sc_cap)
    // live in a small struct so each field has a stable address for the
    // kernel-params array.
    var common = LzCommonParams{
        .comp = self.d_comp_persist,
        .descs_dev = self.d_descs_persist,
        .dst = lz_dst_base_dev,
        .chunks_per_group = chunks_per_group,
        .total = lz_total_count_dev,
        .sub_chunk_cap = sub_chunk_cap,
    };

    if (use_raw_kernel) {
        // VK adaptation: params[] layout per module_loader KERNEL_DECLS:
        // first n_bindings entries are pointers to VkDeviceBuffer handles
        // populating descriptor set bindings 0..n_bindings-1; remaining
        // entries pack into push_constant_size bytes in declaration order.
        // lz_decode_raw_kernel.comp: n_bindings=4 (CompressedBuf, ChunksBuf,
        // DstBuf, TotalChunksBuf), push_constant_size=8 (chunks_per_group,
        // sub_chunk_cap as 2× u32).
        var raw_params = [_]?*anyopaque{
            @ptrCast(&common.comp),
            @ptrCast(&common.descs_dev),
            @ptrCast(&common.dst),
            @ptrCast(&common.total),
            @ptrCast(&common.chunks_per_group),
            @ptrCast(&common.sub_chunk_cap),
        };
        var raw_extra = [_]?*anyopaque{null};
        const t_lzr = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeRawKernel", stream);
        try vkCall(launch_fn(module_loader.kernel_raw_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &raw_params, &raw_extra), .launch);
        endKernelTiming(t_lzr, stream);
    } else {
        var p_entropy_scratch: u64 = self.d_entropy_scratch;
        var p_entropy_slot_stride: u64 = @as(u64, total_subchunks) * descriptors.ENTROPY_SCRATCH_SLOT_BYTES;
        var p_first_sub_idx: VkDeviceBuffer = self.d_first_subchunk_idx;

        var lz_params = [_]?*anyopaque{
            @ptrCast(&common.comp),
            @ptrCast(&common.descs_dev),
            @ptrCast(&common.dst),
            @ptrCast(&common.chunks_per_group),
            @ptrCast(&common.total),
            @ptrCast(&common.sub_chunk_cap),
            @ptrCast(&p_entropy_scratch),
            @ptrCast(&p_entropy_slot_stride),
            @ptrCast(&p_first_sub_idx),
        };
        var lz_extra = [_]?*anyopaque{null};

        const t_lz = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeKernel", stream);
        try vkCall(launch_fn(module_loader.kernel_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &lz_params, &lz_extra), .launch);
        endKernelTiming(t_lz, stream);
    }

    if (split_timer) {
        try vkCall(stream_sync_fn(stream), .sync);
        if (t_lz_start) |ts_start| {
            if (io) |io_val| {
                split_lz_ns += nsSince(ts_start, io_val);
            }
        }
    }

    return split_lz_ns;
}

/// CUDA reference: src/decode/decode_dispatch.zig:479-487. Writes the
/// decompressed output to the caller-supplied destination.
/// Two paths:
///   - D2D: the caller passed a device-resident target; issue a D2D
///     async copy on `heavy_stream` so it serializes after the LZ
///     kernel on the same stream. The caller's stream-sync on that
///     stream waits for the result.
///   - D2H: the caller wants the result on the host; the synchronous
///     `procs.d2h` blocks until the copy completes.
///
/// In sync mode (work_stream == 0) the D2D path also issues a
/// ctx-wide sync to ensure the caller observes a settled output
/// when this function returns.
pub fn finalizeOutput(
    self: *DecodeContext,
    procs: *const VkProcs,
    req: DecodeRequest,
    heavy_stream: usize,
) GpuError!void {
    if (req.d_output_target) |dev_target| {
        const d2d = procs.d2d orelse return error.BackendNotAvailable;
        try vkCall(d2d(dev_target + req.dst_start_off, self.d_output + req.dst_start_off, req.decompressed_size, heavy_stream), .copy);
        if (self.work_stream == 0) try vkCall(procs.sync(), .sync);
    } else {
        // Iter 4: route D2H via d2h_async on heavy_stream when available
        // so it batches into the same cmdbuf that ran the LZ kernel.
        // The staging→host memcpy is deferred to streamEndAndWait which
        // fullGpuLaunchImpl triggers via the back-half stream_sync.
        const _t_d2h0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        try vkCall(d2hRoute(procs, heavy_stream, @ptrCast(req.dst_full + req.dst_start_off), self.d_output + req.dst_start_off, req.decompressed_size), .copy);
        if (g_phase_profile_enabled) g_phase_finalize_d2h_ns += qpcDeltaNs(_t_d2h0, vk.qpcNow());
    }
}

/// CUDA reference: src/decode/decode_dispatch.zig:493-497. Result of
/// the input-upload + prefix-sum phase: the worst-case sub-chunk count
/// drives entropy-scratch sizing, and the per-stream region offsets
/// within the scratch must match the merge-kernel writer and the
/// LZ-kernel reader.
const FrameLayout = struct {
    total_subchunks: u32,
    tok_offset: usize,
    off16_offset: usize,
};

/// CUDA reference: src/decode/decode_dispatch.zig:503-579. Upload
/// phase: ensure persistent buffers, H2D/D2D the chunk descs, H2D/D2D
/// the compressed block, run the device-side prefix sum, and size the
/// entropy scratch. Returns the FrameLayout the back-half kernels need.
pub fn uploadInputAndPrefixSum(
    self: *DecodeContext,
    req: DecodeRequest,
    procs: *const VkProcs,
) GpuError!FrameLayout {
    const _t_upload_misc_start = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    const total_output = req.dst_start_off + req.decompressed_size;
    try ensureDeviceOutput(self, total_output + 64);
    // Iter 4: route H2D copies via h2d_async on the work_stream when
    // available. Sync semantics still hold on the dispatcher's caller
    // because fullGpuLaunchImpl drains the stream before return; the
    // batching collapses N submit+wait cycles into 1 per block.
    if (req.dst_start_off > 0) {
        const _t_h2d0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        try vkCall(h2dRoute(procs, self.work_stream, self.d_output, @ptrCast(req.dst_full), req.dst_start_off), .copy);
        if (g_phase_profile_enabled) g_phase_upload_h2d_ns += qpcDeltaNs(_t_h2d0, vk.qpcNow());
    }

    // `procs.malloc_device(0)` is rejected by VMA (mirrors CUDA's
    // `cuMemAlloc(0)`); route an empty-input frame through a small dummy
    // allocation so the ensureDeviceBuf below succeeds and the empty
    // input flows through to the end-mark emit without a special case.
    const comp_bytes = if (req.compressed_block.len > 0) req.compressed_block.len else 4;
    const desc_bytes = req.chunk_descs.len * @sizeOf(ChunkDesc);
    try ensureDeviceBuf(&self.d_comp_persist, &self.d_comp_persist_size, comp_bytes + 16);
    try ensureDeviceBuf(&self.d_descs_persist, &self.d_descs_persist_size, desc_bytes);

    // Chunk descs: D2D from a caller-supplied buffer when the walk
    // kernel already wrote them to d_walk_chunks, else H2D from the
    // host slice. H2D blocks on the host; D2D queues on work_stream
    // and serializes with the prefix-sum kernel via stream ordering.
    if (req.d_chunk_descs_override) |dev_ptr| {
        const d2d = procs.d2d orelse return error.BackendNotAvailable;
        try vkCall(d2d(self.d_descs_persist, dev_ptr, desc_bytes, self.work_stream), .copy);
    } else {
        const _t_h2d_descs0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        try vkCall(h2dRoute(procs, self.work_stream, self.d_descs_persist, @ptrCast(req.chunk_descs.ptr), desc_bytes), .copy);
        if (g_phase_profile_enabled) g_phase_upload_h2d_ns += qpcDeltaNs(_t_h2d_descs0, vk.qpcNow());
    }

    // Account misc time up to here (everything before prefix-sum).
    if (g_phase_profile_enabled) g_phase_upload_misc_ns += qpcDeltaNs(_t_upload_misc_start, vk.qpcNow());

    // `gpuPrefixSumChunksImpl` returns `GpuError!T` (not `?T`) because
    // there is no host fallback — every GPU decode path needs it.
    const _t_ps0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    _ = try scan_gpu.gpuPrefixSumChunksImpl(self, self.d_descs_persist, @intCast(req.chunk_descs.len), req.sub_chunk_cap);
    if (g_phase_profile_enabled) g_phase_upload_prefixsum_ns += qpcDeltaNs(_t_ps0, vk.qpcNow());
    const _t_upload_misc_resume = if (g_phase_profile_enabled) vk.qpcNow() else 0;

    // Skip the D2H of `total_subchunks` and use a host-computed
    // worst-case bound: `chunk_count * ceil(chunk_size / sub_chunk_cap)`.
    // Over-allocating entropy_scratch is safe — the kernels self-gate
    // on the actual counts via d_total_subchunks_buf / d_compact_counts.
    const chunk_size_bytes: u32 = 0x40000; // = `constants.chunk_size`, inlined to avoid the format-layer import
    const max_sub_per_chunk: u32 = if (req.sub_chunk_cap == 0) 4 else @max(1, (chunk_size_bytes + req.sub_chunk_cap - 1) / req.sub_chunk_cap);
    const total_subchunks: u32 = @as(u32, @intCast(req.chunk_descs.len)) * max_sub_per_chunk;
    // descriptors.ENTROPY_SCRATCH_SLOT_BYTES holds the largest sub-chunk's
    // lit/tok streams; off16-hi at +0, off16-lo at +OFF16_HILO_SPLIT_OFFSET
    // within each slot. Layout: [lit: total*slot] [tok: total*slot] [off16: total*slot].
    const per_subchunk_scratch: usize = @intCast(descriptors.ENTROPY_SCRATCH_SLOT_BYTES);
    const entropy_scratch_bytes = @as(usize, total_subchunks) * per_subchunk_scratch * 3;
    try ensureDeviceBuf(&self.d_entropy_scratch, &self.d_entropy_scratch_size, entropy_scratch_bytes);
    const tok_offset = @as(usize, total_subchunks) * per_subchunk_scratch;
    const off16_offset = @as(usize, total_subchunks) * per_subchunk_scratch * 2;
    self.d_entropy_off16_scratch = self.d_entropy_scratch + off16_offset;

    // Always pass the device prefix-sum to the LZ kernel. When
    // `sub_chunk_cap >= chunk_size` the prefix sum is `[0, 1, 2, ...]`
    // and the kernel reads identity values.
    self.d_first_subchunk_idx = self.d_first_sub_idx_persist;
    if (req.chunk_descs.len > descriptors.WALK_MAX_CHUNKS) return error.BadMode;

    // Source-side input: D2D when the bytes are already device-resident,
    // H2D otherwise. The D2D copy issues on `self.work_stream` so the
    // post-copy stream sync only has to wait on the caller's stream;
    // the H2D path uses h2d_async into the same per-stream cmdbuf when
    // available so it batches with the surrounding kernels (iter 4).
    if (req.compressed_block.len > 0) {
        if (req.d_compressed_src) |dev_src| {
            const d2d = procs.d2d orelse return error.BackendNotAvailable;
            try vkCall(d2d(self.d_comp_persist, dev_src, req.compressed_block.len, self.work_stream), .copy);
        } else {
            const _t_h2d_comp0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
            try vkCall(h2dRoute(procs, self.work_stream, self.d_comp_persist, @ptrCast(req.compressed_block.ptr), req.compressed_block.len), .copy);
            if (g_phase_profile_enabled) g_phase_upload_h2d_ns += qpcDeltaNs(_t_h2d_comp0, vk.qpcNow());
        }
    }

    if (g_phase_profile_enabled) g_phase_upload_misc_ns += qpcDeltaNs(_t_upload_misc_resume, vk.qpcNow());

    return .{
        .total_subchunks = total_subchunks,
        .tok_offset = tok_offset,
        .off16_offset = off16_offset,
    };
}

/// CUDA reference: src/decode/decode_dispatch.zig:586-686. Back-half
/// phase: pick the heavy stream, stage the LZ self-gate count, run the
/// Huffman predecode (if any), run the LZ pipeline, then sync. Records
/// `last_*_kernel_ns` onto the facade on the way out. The caller is
/// responsible for the e2e timer slice that surrounds this call.
pub fn runBackHalf(
    self: *DecodeContext,
    req: DecodeRequest,
    procs: *const VkProcs,
    layout: FrameLayout,
    n_huff: u32,
    have_huff: bool,
    total_chunks: u32,
    heavy_stream: usize,
    facade: anytype,
    /// Iter 4: true when the dispatcher entered in sync mode (caller's
    /// work_stream was 0 before the iter-4 promotion to pipeline_stream).
    /// Gates the end-of-back-half stream_sync that flushes the batched
    /// cmdbuf and deferred D2H memcpys.
    is_sync_mode: bool,
) GpuError!void {
    const io = req.io;
    const stream_sync_fn = procs.stream_sync;

    // Cross-stream barrier: in sync mode `heavy_stream` is the library-
    // owned `pipeline_stream` so the front-half work queued on
    // stream 0 has to drain before the back half can read its results.
    if (heavy_stream != self.work_stream) {
        try vkCall(stream_sync_fn(self.work_stream), .sync);
    }

    // KERNEL TIMER: only pure GPU kernel time from here.
    const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    // SLZ_SPLIT_TIMER: separate Huff predecode from LZ decode timing.
    const split_timer = std.c.getenv("SLZ_SPLIT_TIMER") != null;

    // Stage the LZ kernel's self-gate count. D2D path: the caller
    // supplies `d_n_chunks_dev` (typically `d_walk_meta + offset`).
    // Else stage `total_chunks` into `d_n_groups_scratch` via 4 B H2D.
    //
    // Iter 4: route through h2d_async on heavy_stream so this trailing
    // 4 B copy batches with the LZ kernel that consumes it instead of
    // triggering its own submit+wait round-trip. `host_total_chunks`
    // must outlive h2dRoute's call — h2d_async memcpys into staging
    // immediately so the local-variable lifetime is fine.
    const lz_total_count_dev: u64 = if (req.d_n_chunks_dev) |dev| dev else blk: {
        try ensureDeviceBuf(&self.d_n_groups_scratch, &self.d_n_groups_scratch_size, 4);
        var host_total_chunks: u32 = total_chunks;
        const _t_h2d_n0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        try vkCall(h2dRoute(procs, heavy_stream, self.d_n_groups_scratch, @ptrCast(&host_total_chunks), 4), .copy);
        if (g_phase_profile_enabled) g_phase_backhalf_h2d_count_ns += qpcDeltaNs(_t_h2d_n0, vk.qpcNow());
        break :blk self.d_n_groups_scratch;
    };

    // Direct-write-to-output fast path: when the caller supplied a
    // device output target AND there's no host prefix to splice in,
    // the LZ kernel writes straight to the caller's buffer, skipping
    // the finalize D2D copy.
    const lz_dst_base_dev: VkDeviceBuffer = if (req.writesDirectlyToTarget()) req.d_output_target.? else self.d_output;

    const t_huff_start = if (split_timer and have_huff)
        if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
    else
        null;
    var split_huff_build_ns: i64 = 0;
    if (have_huff) {
        split_huff_build_ns = try runHuffBuildAndDecode(self, procs, n_huff, heavy_stream, split_timer, t_huff_start, io);
    }

    var split_huff_ns: i64 = 0;

    // Close the Huff time slice: sync `heavy_stream` so the LZ
    // measurement excludes Huff time. `split_timer` rules out a future
    // graph-mode capture, so the inner sync is safe.
    if (split_timer and have_huff) {
        try vkCall(stream_sync_fn(heavy_stream), .sync);
        if (t_huff_start) |hs| {
            if (io) |io_val| {
                split_huff_ns = nsSince(hs, io_val);
            }
        }
        std.debug.print("  [huff split] build {d:.3} ms  decode {d:.3} ms\n", .{
            @as(f64, @floatFromInt(split_huff_build_ns)) / 1e6,
            @as(f64, @floatFromInt(split_huff_ns - split_huff_build_ns)) / 1e6,
        });
    }

    const _t_lz0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    const split_lz_ns = try runLzPipeline(
        self, procs,
        req.chunks_per_group, req.sub_chunk_cap, n_huff,
        total_chunks, layout.total_subchunks,
        heavy_stream, split_timer, io,
        lz_total_count_dev, lz_dst_base_dev,
    );
    if (g_phase_profile_enabled) g_phase_backhalf_lz_launch_ns += qpcDeltaNs(_t_lz0, vk.qpcNow());

    // Back-half stream sync: skip in async mode (caller's stream
    // carries the queued work; they sync themselves).
    //
    // Iter 4: gate on the captured `is_sync_mode` instead of the live
    // `self.work_stream == 0` check — the dispatcher's iter-4 batching
    // path promotes work_stream to pipeline_stream for the duration of
    // the call, so the original check would now mis-classify sync-mode
    // decodes as async and skip the flush.
    if (is_sync_mode) {
        const _t_fw0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        const sync_rc = stream_sync_fn(heavy_stream);
        if (g_phase_profile_enabled) g_phase_backhalf_fence_wait_ns += qpcDeltaNs(_t_fw0, vk.qpcNow());
        if (sync_rc != VK_SUCCESS_RC) {
            std.debug.print("GPU heavy stream: sync FAILED rc={d}\n", .{sync_rc});
            return error.SyncFailed;
        }
    }

    if (t_before_kern) |t_start| {
        if (io) |io_val| {
            facade.last_kernel_ns = nsSince(t_start, io_val);
            if (split_timer) {
                facade.last_lz_kernel_ns = split_lz_ns;
                facade.last_huff_kernel_ns = split_huff_ns;
            } else {
                facade.last_lz_kernel_ns = 0;
                facade.last_huff_kernel_ns = 0;
            }
        }
    }
}

/// CUDA reference: src/decode/decode_dispatch.zig:688-775. Top-level
/// decode entry. Orchestrates upload + prefix-sum, optional L2
/// scan/compact/merge/gather/huff, back-half (LZ pipeline + sync +
/// timing), finalize.
///
/// VK iter 4 host-perf adaptation: when the caller is in sync mode
/// (req.work_stream == 0 on entry → self.work_stream == 0 here), the
/// dispatcher temporarily reassigns self.work_stream = pipeline_stream
/// so every internal procs.{h2d_async, d2h_async, d2d, launch_kernel,
/// prefix_sum} call records into the same per-stream cmdbuf instead of
/// triggering its own vkQueueSubmit + vkWaitForFences. The whole batch
/// drains in ONE submit at the end. Original work_stream is restored
/// before return so the caller-visible state is unchanged. This single
/// change collapses ~7 submit/wait round-trips per decode block to 1,
/// recovering the host overhead that put VK at 21x CUDA on enwik8.
pub fn fullGpuLaunchImpl(self: *DecodeContext, req: DecodeRequest) GpuError!void {
    if (!module_loader.init()) return error.BackendNotAvailable;
    if (module_loader.kernel_fn == 0 and module_loader.kernel_raw_fn == 0) return error.KernelMissing;
    try module_loader.ensurePipelineStream(self);

    const facade = @import("driver.zig");
    const io = req.io;

    // SLZ_E2E_TIMER: end-to-end decode phase breakdown.
    const e2e_timer = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_e2e0 = if (e2e_timer)
        (if (io) |io_val| std.Io.Clock.awake.now(io_val) else null)
    else
        null;
    var e2e_cum: E2eCumulative = .{};

    // Iter 4 host-perf fix: capture sync-mode entry state, then promote
    // work_stream to pipeline_stream so all in-decode procs.* ops batch
    // through one cmdbuf. The `is_sync_mode` flag preserves the original
    // gate semantics for the `work_stream == 0` checks below (sync-mode
    // call needs the end-of-decode stream_sync + profiling drain to run).
    const is_sync_mode = self.work_stream == 0;
    const original_work_stream = self.work_stream;
    if (is_sync_mode) self.work_stream = self.pipeline_stream;
    defer self.work_stream = original_work_stream;

    // Iter 4 parallel-safety: when running in sync mode the dispatcher
    // routes all work through the singleton pipeline_stream's cmdbuf.
    // Two concurrent threads (e.g. ptest_vk's 16-worker test runner)
    // would otherwise interleave their cmdbuf records and corrupt each
    // other's batched submission. Serialize through this dispatcher
    // mutex; async-mode callers using their own VkStream skip it and
    // serialize per-stream inside module_loader instead.
    if (is_sync_mode) {
        module_loader.lockDispatcherMutex();
    }
    defer if (is_sync_mode) module_loader.unlockDispatcherMutex();

    const _t_resolve0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    const procs = try VkProcs.resolve();
    if (g_phase_profile_enabled) g_phase_resolve_ns += qpcDeltaNs(_t_resolve0, vk.qpcNow());

    // Phase 1: upload + prefix sum (always runs — needed for chunk
    // descriptor sizing on L1 and L2+).
    const _t_upload0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    const layout = try uploadInputAndPrefixSum(self, req, &procs);
    if (g_phase_profile_enabled) g_phase_upload_ns += qpcDeltaNs(_t_upload0, vk.qpcNow());
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.h2d = nsSince(t0, io_val);
        e2e_cum.postscan = e2e_cum.h2d;
    };

    // Phase 2-3: L2 scan/compact/merge/gather + Huffman descriptor merge.
    //
    // VK adaptation: gated on level>=2 to skip L2 work on L1 frames.
    // CUDA currently dispatches unconditionally (no-op on L1); upstream
    // TODO to backport this gate.
    var scan: ScanResult = .{ .num_raw_off16 = 0 };
    var n_huff: u32 = 0;
    var have_huff = false;
    if (req.level >= 2) {
        // Phase 2: scan + raw-off16 gather. The GPU scan kernel reads
        // from `d_comp_persist` (filled by the prior H2D / D2D), so the
        // same code path handles both decode entry points.
        if (module_loader.huff_build_fn != 0) {
            if (req.chunk_descs.len > descriptors.WALK_MAX_CHUNKS) return error.BadMode;

            // Allocate L2-only scratch slots before the scan kernel
            // runs. Each ensureDeviceBuf is a grow-only no-op on
            // subsequent decodes of the same level.
            try ensureDeviceBuf(&self.d_compact_counts, &self.d_compact_counts_size, 32);
            try ensureDeviceBuf(&self.d_scan_staged, &self.d_scan_staged_size, @as(usize, layout.total_subchunks) * @sizeOf(descriptors.ScanHuffDesc));
            try ensureDeviceBuf(&self.d_first_sub_idx_persist, &self.d_first_sub_idx_persist_size, @as(usize, descriptors.WALK_MAX_CHUNKS) * 4);

            scan = try scan_gpu.gpuScanChunks(
                self,
                req.chunk_descs,
                req.compressed_block,
                req.sub_chunk_cap,
                layout.total_subchunks,
            );
            if (t_e2e0) |t0| if (io) |io_val| {
                e2e_cum.postscan = nsSince(t0, io_val);
            };
            try gatherRawOff16(self, scan, @intCast(req.compressed_block.len), procs.launch);
        }

        // Phase 3: Huffman descriptor merge (LUT + descriptor allocations).
        n_huff = scan.num_huff_lit + scan.num_huff_tok +
            scan.num_huff_off16hi + scan.num_huff_off16lo;
        have_huff = n_huff > 0 and module_loader.huff_build_fn != 0 and module_loader.huff_decode_fn != 0;
        if (have_huff) {
            const huff_desc_bytes = @as(usize, n_huff) * @sizeOf(HuffDecChunkDesc);
            const huff_lut_bytes = @as(usize, n_huff) * HUFF_LUT_ENTRIES * @sizeOf(u32);
            try ensureDeviceBuf(&self.d_huff_descs, &self.d_huff_descs_size, huff_desc_bytes);
            try ensureDeviceBuf(&self.d_huff_lut, &self.d_huff_lut_size, huff_lut_bytes);
            try mergeHuffDescs(self, layout.tok_offset, layout.off16_offset, procs.launch);
        }
    }
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.scan = nsSince(t0, io_val);
    };

    // `heavy_stream` carries the back half (huff + LZ + finalize). In
    // async mode the caller's stream IS the heavy stream; in sync mode
    // we use the library-owned `pipeline_stream` (guaranteed populated
    // by the unconditional `ensurePipelineStream` call above).
    const heavy_stream: usize = if (self.work_stream != 0) self.work_stream else self.pipeline_stream;
    const total_chunks: u32 = @intCast(req.chunk_descs.len);

    // Iter 8 subfix 3: pre-create / pre-allocate the imported host
    // buffer that the eventual finalize-phase D2H will target. Doing
    // this BEFORE runBackHalf overlaps the ~1 ms
    // vkCreateBuffer+vkAllocateMemory cost (cache miss) or ~0 ms
    // (cache hit) with the LZ kernel's GPU execution. The prep is
    // stashed on the heavy_stream so procD2HAsync inside
    // finalizeOutput consumes it without re-running the alloc.
    //
    // Only run when the D2H path will actually execute: skipped when
    // writesDirectlyToTarget() is true (LZ kernel writes straight to
    // d_output_target — no D2H). The dst pointer arithmetic mirrors
    // finalizeOutput's d2hRoute call below.
    if (!req.writesDirectlyToTarget()) {
        const _t_prep0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        const dst_ptr: *anyopaque = @ptrCast(req.dst_full + req.dst_start_off);
        if (module_loader.prepareImportHostBuffer(dst_ptr, req.decompressed_size)) |prep| {
            _ = module_loader.stashPreparedImportForStream(heavy_stream, prep);
        }
        if (g_phase_profile_enabled) g_phase_import_prep_ns += qpcDeltaNs(_t_prep0, vk.qpcNow());
    }

    // Phase 4: back half (Huff predecode + LZ pipeline + sync + timing).
    const _t_bh0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    try runBackHalf(self, req, &procs, layout, n_huff, have_huff, total_chunks, heavy_stream, facade, is_sync_mode);
    if (g_phase_profile_enabled) g_phase_backhalf_ns += qpcDeltaNs(_t_bh0, vk.qpcNow());
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.predh = nsSince(t0, io_val);
    };

    // Phase 5: finalize, profiling drain, e2e emit. Skip the finalize
    // D2D when the LZ kernel already wrote straight to the caller's
    // device buffer — same predicate as the LZ dst-pick in runBackHalf.
    //
    // Iter 4: finalizeOutput records D2H into the batched cmdbuf via
    // d2h_async; the runBackHalf stream_sync above already flushed it,
    // BUT in the L1 path finalize is called AFTER the sync, so the
    // d2h_async records into a freshly-reset cmdbuf and needs another
    // flush. Issue an extra stream_sync after finalize to drain it.
    const _t_fin0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    if (!req.writesDirectlyToTarget()) {
        try finalizeOutput(self, &procs, req, heavy_stream);
        if (is_sync_mode) {
            const _t_fsync0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
            try vkCall(procs.stream_sync(heavy_stream), .sync);
            if (g_phase_profile_enabled) g_phase_finalize_sync_ns += qpcDeltaNs(_t_fsync0, vk.qpcNow());
        }
    }
    if (g_phase_profile_enabled) {
        g_phase_finalize_ns += qpcDeltaNs(_t_fin0, vk.qpcNow());
        g_phase_decode_count += 1;
    }
    // Iter 4: gate profiling drain on captured is_sync_mode (not the
    // live work_stream check, which now reads non-zero due to the
    // dispatcher's pipeline_stream promotion).
    if (is_sync_mode) {
        finalizeProfiling(&self.pending_timings, &self.last_timings);
    }
    if (t_e2e0) |t0| if (io) |io_val| {
        emitE2eTrace(t0, io_val, e2e_cum, facade.last_kernel_ns);
    };
}
