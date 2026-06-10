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
    // Iter 12: H2D path-taken counters live in module_loader.
    module_loader.g_h2d_path_prepared_import = 0;
    module_loader.g_h2d_path_inline_import = 0;
    module_loader.g_h2d_path_staging = 0;
    // Iter 14 (B): LRU import-cache telemetry counters live in module_loader.
    module_loader.g_import_cache_hits.store(0, .seq_cst);
    module_loader.g_import_cache_misses.store(0, .seq_cst);
    module_loader.g_import_cache_evictions.store(0, .seq_cst);
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

// VK adaptation (H3): encode-side phase printer. Only the h2d_paths /
// import_cache counters mean anything for the compress hot path — every
// decode-specific phase accumulator (upload_h2d, backhalf_*, finalize_*)
// stays at zero because fullGpuLaunchImpl never runs. Called from
// bench_compress around the timed encode pass to confirm the 95 MB
// enwik8 input H2D auto-fires the iter-12 VK_EXT_external_memory_host
// import path inside procH2DAsync. Counters live in module_loader and
// are shared with the decode path; reset by phaseProfileInit().
pub fn printEncodeImportTelemetry(w: anytype) void {
    if (!g_phase_profile_enabled) return;
    w.print("phase: h2d_paths               prepared_import={d}, inline_import={d}, staging={d}\n", .{
        module_loader.g_h2d_path_prepared_import,
        module_loader.g_h2d_path_inline_import,
        module_loader.g_h2d_path_staging,
    }) catch {};
    w.print("phase: import_cache            hits={d}, misses={d}, evictions={d}\n", .{
        module_loader.g_import_cache_hits.load(.seq_cst),
        module_loader.g_import_cache_misses.load(.seq_cst),
        module_loader.g_import_cache_evictions.load(.seq_cst),
    }) catch {};
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
    // Iter 12: which H2D path did procH2DAsync actually take? prepared_import
    // = consumed a pre-stashed prep (cheapest); inline_import = imported
    // mid-call (no host @memcpy, ~200us alloc on cache miss); staging =
    // iter-4 fallback (pays host @memcpy of the full size). Confirms that
    // the iter-12 re-enable of H2D import is actually firing.
    w.print("phase: h2d_paths               prepared_import={d}, inline_import={d}, staging={d}\n", .{
        module_loader.g_h2d_path_prepared_import,
        module_loader.g_h2d_path_inline_import,
        module_loader.g_h2d_path_staging,
    }) catch {};
    // Iter 14 (B): surface whether iter-8's 16-slot LRU import cache is
    // hitting reliably in steady state. Misses > 0 in steady state
    // suggest src/dst pointers churn between decodes — caller
    // re-maps or moves the buffer; worth investigating for the
    // small-file (web.txt) regime where the 200-500us cache-miss cost
    // dominates the residual e2e gap vs CUDA.
    w.print("phase: import_cache            hits={d}, misses={d}, evictions={d}\n", .{
        module_loader.g_import_cache_hits.load(.seq_cst),
        module_loader.g_import_cache_misses.load(.seq_cst),
        module_loader.g_import_cache_evictions.load(.seq_cst),
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
    ?[*]const u64, // iter 4c: per-binding byte offsets (null = all-zero)
) callconv(.c) VkResult;
const FnCtxSync = *const fn () callconv(.c) VkResult;
const FnStreamSync = *const fn (VkStream) callconv(.c) VkResult;
const FnD2D = *const fn (VkDeviceBuffer, VkDeviceBuffer, usize, VkStream) callconv(.c) VkResult;
const FnD2DOffset = *const fn (VkDeviceBuffer, VkDeviceBuffer, usize, usize, VkStream) callconv(.c) VkResult;

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
    // Iter 13: VK adaptation — early-submit the transfer cmdbuf at the
    // end of uploadInputAndPrefixSum so the dedicated DMA engine starts
    // the H2D copies in parallel with runBackHalf's host-side prep.
    // No CUDA analogue (cuMemcpyHtoDAsync auto-issues on the call).
    // Optional so legacy backends without the slot still resolve.
    stream_flush_transfer: ?FnStreamSync,
    d2d: ?FnD2D,
    // A-025: D2D with source byte offset (true-D2D compressed-block copy).
    d2d_offset: ?FnD2DOffset,
    // v4 #12: input-side transfer-leg D2D (source not written by batched compute).
    d2d_input_offset: ?FnD2DOffset,
    // VK adaptation: COMPUTE_SHADER_WRITE → COMPUTE_SHADER_READ pipeline
    // barrier on the stream's open cmdbuf. Required at the Huffman-decode
    // → LZ-decode boundary where the LZ general kernel reads the
    // entropy_scratch slots that huff_decode_4stream just wrote. CUDA
    // collapses this dependency into cuLaunchKernel's per-stream
    // implicit ordering; VK has no such auto-ordering between
    // consecutive vkCmdDispatch calls. Optional so legacy backends still
    // resolve.
    compute_to_compute_barrier: ?FnStreamSync,

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
            .stream_flush_transfer = vk.procs.stream_flush_transfer,
            .d2d = vk.procs.d2d,
            .d2d_offset = vk.procs.d2d_offset,
            .d2d_input_offset = vk.procs.d2d_input_offset,
            .compute_to_compute_barrier = vk.procs.compute_to_compute_barrier,
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
    // Iter 4c: per-binding offsets replace the pre-iter-4c handle
    // arithmetic on d_compact_counts (each n_* count is at
    // COUNTS_STRIDE*idx within the same buffer; pre-iter-4c the code
    // did `self.d_compact_counts + 4/8/12/20` which is meaningless
    // because VkDeviceBuffer is a registry handle, not a VA). The
    // four per-stream count slots and the n_merged slot all bind to
    // the same base handle; their offsets land in binding_offsets[].
    const COUNTS_STRIDE: u64 = 256;
    var args = MergeHuffParams{
        .lit = self.d_compact_lit,
        .tok = self.d_compact_tok,
        .hi = self.d_compact_hi,
        .lo = self.d_compact_lo,
        .n_lit = self.d_compact_counts,
        .n_tok = self.d_compact_counts,
        .n_hi = self.d_compact_counts,
        .n_lo = self.d_compact_counts,
        .tok_region = @intCast(tok_offset),
        .off16_region = @intCast(off16_offset),
        .merged = self.d_huff_descs,
        .n_merged = self.d_compact_counts,
    };
    // VK adaptation: params[] layout per module_loader KERNEL_DECLS contract
    // (see scan_gpu.zig:58-64 and decode_dispatch.zig:597-604 for working
    // reference). First n_bindings entries are pointers to VkDeviceBuffer
    // handles populating descriptor set bindings 0..n_bindings-1; remaining
    // entries pack into push_constant_size bytes in declaration order.
    // merge_huff_descs_kernel.comp: n_bindings=10 (LitBuf, TokBuf, HiBuf,
    // LoBuf, NLitBuf, NTokBuf, NHiBuf, NLoBuf, MergedBuf, NMergedBuf),
    // push_constant_size=8 (tok_region_off, off16_region_off as 2× u32).
    // CUDA's kernel argument list is unchanged; only the VK host packing
    // order changes — buffer-pointer params first, then push-constant scalars.
    var m_params = [_]?*anyopaque{
        @ptrCast(&args.lit),        @ptrCast(&args.tok),          @ptrCast(&args.hi),       @ptrCast(&args.lo),
        @ptrCast(&args.n_lit),      @ptrCast(&args.n_tok),        @ptrCast(&args.n_hi),     @ptrCast(&args.n_lo),
        @ptrCast(&args.merged),     @ptrCast(&args.n_merged),
        @ptrCast(&args.tok_region), @ptrCast(&args.off16_region),
    };
    var m_extra = [_]?*anyopaque{null};
    // Bindings 0-3 (lit/tok/hi/lo) + 8 (merged) bind whole buffers at
    // offset 0; bindings 4-7 + 9 are sub-region binds of d_compact_counts
    // at strided per-stream slots.
    const m_offs = [_]u64{
        0, 0, 0, 0,
        COUNTS_STRIDE * 0,
        COUNTS_STRIDE * 1,
        COUNTS_STRIDE * 2,
        COUNTS_STRIDE * 3,
        0,
        COUNTS_STRIDE * 5,
    };
    const stream = self.work_stream;
    // A-021 (2026-06-10, CUDA-mirror of src/decode/decode_dispatch.zig
    // mergeHuffDescs): prefer the 4-block parallel merge (wall ~= the
    // largest region instead of the sum of all four); the serial kernel
    // stays as a fallback when the par pipeline failed to build.
    const use_par = module_loader.merge_huff_descs_par_fn != 0;
    const merge_fn = if (use_par) module_loader.merge_huff_descs_par_fn else module_loader.merge_huff_descs_fn;
    const grid_x: u32 = if (use_par) 4 else 1;
    const label: [*:0]const u8 = if (use_par) "slzMergeHuffDescsParKernel (x4)" else "slzMergeHuffDescsKernel";
    const t_merge = beginKernelTiming(self.enable_profiling, &self.pending_timings, label, stream);
    try vkCall(launch_fn(merge_fn, grid_x, 1, 1, 1, 1, 1, 0, stream, &m_params, &m_extra, &m_offs), .launch);
    endKernelTiming(t_merge, stream);
    // Iter 4c: merge wrote d_huff_descs + n_merged. huff_build_lut +
    // huff_decode_4stream read both downstream. VK requires the explicit
    // compute→compute barrier here; CUDA's per-stream cuLaunchKernel
    // ordering covers it implicitly. Pre-iter-4c the merge launch never
    // actually wrote anything (lookupAlloc on `base + N` returned -1 →
    // launch returned -1 → KernelLaunchFailed) so the missing barrier
    // was latent — iter 4c surfaces the dependency.
    if (vk.procs.compute_to_compute_barrier) |bf| try vkCall(bf(stream), .sync);
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
    // Iter 4c: n_raw lives in d_compact_counts slot 4 under the strided
    // layout (see scan_gpu.zig::gpuScanChunks). Pre-iter-4c the code did
    // `self.d_compact_counts + 16` arithmetic on the handle — fix by
    // passing the base + routing the strided offset through
    // procLaunchKernel.binding_offsets.
    const COUNTS_STRIDE: u64 = 256;
    var p_count: u64 = self.d_compact_counts;
    // VK adaptation: params[] layout per module_loader KERNEL_DECLS contract
    // (see scan_gpu.zig:58-64 and decode_dispatch.zig:597-604 for working
    // reference). First n_bindings entries are pointers to VkDeviceBuffer
    // handles populating descriptor set bindings 0..n_bindings-1; remaining
    // entries pack into push_constant_size bytes in declaration order.
    // gather_raw_off16_kernel.comp: n_bindings=4 (CompBaseBuf, ScratchBaseBuf,
    // DescsBuf, CountBuf), push_constant_size=4 (comp_len as u32).
    // CUDA's kernel argument list is unchanged; only the VK host packing
    // order changes — buffer-pointer params first, then push-constant scalars.
    var params = [_]?*anyopaque{
        @ptrCast(&p_comp), @ptrCast(&p_scratch), @ptrCast(&p_descs), @ptrCast(&p_count),
        @ptrCast(&p_comp_len),
    };
    var extra = [_]?*anyopaque{null};
    // Binding 1 (ScratchBaseBuf) is the d_entropy_off16 sub-region of
    // d_entropy_scratch (offset = off16_offset captured at setup).
    const gather_offs = [_]u64{ 0, self.d_entropy_off16_offset, 0, COUNTS_STRIDE * 4 };
    // Grid size = worst-case sub-chunk count (`num_raw_off16` is the
    // dispatch's upper bound); the kernel self-gates on `*p_count` so
    // over-launching is safe.
    const grid_x: u32 = scan.num_raw_off16;
    const stream = self.work_stream;
    const t_gather = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzGatherRawOff16Kernel", stream);
    defer endKernelTiming(t_gather, stream);
    try vkCall(launch_fn(module_loader.gather_off16_fn, grid_x, 1, 1, 256, 1, 1, 0, stream, &params, &extra, &gather_offs), .launch);
    // Iter 4c: gather wrote into d_entropy_off16_scratch (sub-region of
    // d_entropy_scratch). The downstream LZ general kernel reads from
    // d_entropy_scratch. CUDA's per-stream implicit ordering covers
    // this; VK needs the explicit barrier. Without it the LZ kernel
    // can observe stale entropy_scratch and emit garbage / zeros.
    if (vk.procs.compute_to_compute_barrier) |bf| try vkCall(bf(stream), .sync);
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
    /// A-025: byte offset of the compressed block within
    /// `d_compressed_src`. CUDA folds this into the pointer; VK needs it
    /// separate because VkDeviceBuffer is a registry index.
    d_compressed_src_offset: u64 = 0,
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
/// CUDA orchestration mirrored verbatim: launch
/// `slzHuffBuildLutKernel` (writes the per-block LUT into
/// `self.d_huff_lut`), optionally split-fence for telemetry, then
/// launch `slzHuffDecode4StreamKernel` (consumes the LUT + descriptors,
/// writes decoded literal bytes to `self.d_entropy_scratch`). Both
/// dispatches use the merge kernel's output count at
/// `self.d_compact_counts + 20` as the per-kernel self-gate.
pub fn runHuffBuildAndDecode(
    self: *DecodeContext,
    procs: *const VkProcs,
    n_huff: u32,
    heavy_stream: usize,
    split_timer: bool,
    t_huff_start: anytype,
    io: ?std.Io,
) GpuError!i64 {
    const launch_fn = procs.launch;
    const huff_stream = heavy_stream;
    var split_huff_build_ns: i64 = 0;

    // CUDA reference: src/decode/decode_dispatch.zig:317. The huff
    // kernels self-gate on `*d_n_huff`. The merge kernel wrote
    // `n_merged` to `d_compact_counts[5]`; under the iter 4c strided
    // layout slot 5 lives at COUNTS_STRIDE*5 within the buffer (vs the
    // packed 5*4=20 offset CUDA uses). Bind the base handle here and
    // route the strided offset through procLaunchKernel.binding_offsets
    // below — exactly the pattern mergeHuffDescs / gatherRawOff16 use.
    const COUNTS_STRIDE: u64 = 256;
    const N_MERGED_SLOT_OFF: u64 = COUNTS_STRIDE * 5;
    const d_n_huff: u64 = self.d_compact_counts;

    // ── 1. slzHuffBuildLutKernel ────────────────────────────────
    // CUDA reference: src/decode/decode_dispatch.zig:318-331. Builds the
    // per-block 1024-entry LUT from the 128 B weights at the head of
    // each Huffman body.
    {
        var p_comp: u64 = self.d_comp_persist;
        var p_descs: u64 = self.d_huff_descs;
        var p_lut: u64 = self.d_huff_lut;
        var p_n: u64 = d_n_huff;
        // VK adaptation: params[] layout per module_loader KERNEL_DECLS
        // contract (see scan_gpu.zig:58-64 and decode_dispatch.zig
        // lz_decode_raw at :616-624 for working reference). First
        // n_bindings entries are pointers to VkDeviceBuffer handles
        // populating descriptor set bindings 0..n_bindings-1; remaining
        // entries pack into push_constant_size bytes in declaration
        // order. huff_build_lut_kernel.comp: n_bindings=4 (CompBuf,
        // DescsBuf, LutsBuf, NBlocksBuf), push_constant_size=0.
        // CUDA's kernel argument list is unchanged; only the VK host
        // packing order changes — buffer-pointer params first (no
        // scalar push consts here, but the convention still applies).
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs),
            @ptrCast(&p_lut),  @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        const t_hb = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffBuildLutKernel", huff_stream);
        // Iter 4c: binding 3 (NBlocksBuf) is d_compact_counts slot 5.
        const hb_offs = [_]u64{ 0, 0, 0, N_MERGED_SLOT_OFF };
        try vkCall(launch_fn(module_loader.huff_build_fn, n_huff, 1, 1, 32, 1, 1, 0, huff_stream, &params, &extra, &hb_offs), .launch);
        endKernelTiming(t_hb, huff_stream);
    }
    // Iter 4c: huff_build wrote d_huff_lut (the per-block LUT) that
    // huff_decode_4stream reads next. Without the VK barrier the decode
    // kernel can race on the LUT. CUDA's per-launch stream ordering
    // covers this implicitly; VK requires the explicit barrier.
    if (procs.compute_to_compute_barrier) |bf| try vkCall(bf(huff_stream), .sync);
    // CUDA reference: src/decode/decode_dispatch.zig:333-339. Split
    // fence: time the LUT build separately from the decode.
    if (split_timer) {
        try vkCall(procs.stream_sync(huff_stream), .sync);
        if (t_huff_start) |hs| {
            if (io) |io_val|
                split_huff_build_ns = nsSince(hs, io_val);
        }
    }
    // ── 2. slzHuffDecode4StreamKernel ───────────────────────────
    // CUDA reference: src/decode/decode_dispatch.zig:340-356. 32-stream
    // parallel decode; consumes the LUT + descriptors, writes decoded
    // literal bytes into the per-chunk entropy scratch slots.
    {
        var p_comp: u64 = self.d_comp_persist;
        var p_descs: u64 = self.d_huff_descs;
        var p_lut: u64 = self.d_huff_lut;
        var p_out: u64 = self.d_entropy_scratch;
        var p_n: u64 = d_n_huff;
        // Binding 5: u32-aliased view of the same VkBuffer as binding 3
        // (OutputBuf). Used by the Phase 2 hot loop's 4-byte store path
        // to issue a single 32-bit transaction instead of four 8-bit
        // ones — the SPIR-V→NVIDIA driver does not coalesce sequential
        // uint8_t stores so the byte path costs ~4× the memory traffic.
        // See srcVK/decode/huff_decode_4stream_kernel.comp:79-93.
        var p_out_u32: u64 = self.d_entropy_scratch;
        // A-024: binding 6 = d_compact_counts. The kernel uses n_lit /
        // n_lit+n_tok as boundaries to pick which region a block_id
        // belongs to, and adds the matching region offset (passed below
        // as a push constant) instead of trusting the u32 fold the
        // merge kernel used to do (which silently truncated at ~6553
        // sub-chunks).
        var p_counts: u64 = self.d_compact_counts;
        // A-024 (2026-06-10 revision): THREE dispatches, one per region
        // (0 = lit, 1 = tok, 2 = off16). Each binds the output views
        // (bindings 3 + 5) AT the region's byte offset and passes the
        // region id as the single u32 push constant; blocks outside the
        // selected region exit before the LUT load. In-shader write
        // offsets stay region-relative (≤ 2.14 GB), keeping the ~6 GB
        // 1 GB-input scratch under both u32 arithmetic and the
        // per-binding maxStorageBufferRange. CUDA keeps ONE dispatch
        // with u64 region params — divergence documented in
        // PortAdaptations A-024. Region bases are multiples of
        // ENTROPY_SCRATCH_SLOT_BYTES (128 KB), satisfying
        // minStorageBufferOffsetAlignment everywhere.
        //
        // params[] layout per module_loader KERNEL_DECLS: n_bindings=7
        // (CompBuf, DescsBuf, LutsBuf, OutputBuf, NBlocksBuf,
        // OutputBufU32, CompactCountsBuf), push_constant_size=4
        // (region_select). The shared-memory allocation CUDA passes as
        // the 8th launch arg is replaced by the static
        // `shared uint shared_lut[LUT_SIZE]` inside the .comp.
        var p_region: u32 = 0;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp),   @ptrCast(&p_descs),
            @ptrCast(&p_lut),    @ptrCast(&p_out),
            @ptrCast(&p_n),      @ptrCast(&p_out_u32),
            @ptrCast(&p_counts),
            @ptrCast(&p_region),
        };
        var extra = [_]?*anyopaque{null};
        const region_bases = [_]u64{ 0, self.last_tok_offset, self.last_off16_offset };
        const t_hd = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffDecode4StreamKernel", huff_stream);
        for (region_bases, 0..) |region_base, region_idx| {
            p_region = @intCast(region_idx);
            // Iter 4c: binding 4 (NBlocksBuf) is d_compact_counts slot 5.
            // Bindings 3 / 5 (byte + u32 views of entropy scratch) bind
            // at this region's base; binding 6 (CompactCountsBuf) at 0.
            const hd_offs = [_]u64{ 0, 0, 0, region_base, N_MERGED_SLOT_OFF, region_base, 0 };
            try vkCall(launch_fn(module_loader.huff_decode_fn, n_huff, 1, 1, 32, 1, 1, 0, huff_stream, &params, &extra, &hd_offs), .launch);
        }
        endKernelTiming(t_hd, huff_stream);
    }
    return split_huff_build_ns;
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
        try vkCall(launch_fn(module_loader.kernel_raw_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &raw_params, &raw_extra, null), .launch);
        endKernelTiming(t_lzr, stream);
    } else {
        var p_entropy_scratch: u64 = self.d_entropy_scratch;
        var p_entropy_slot_stride: u64 = @as(u64, total_subchunks) * descriptors.ENTROPY_SCRATCH_SLOT_BYTES;
        var p_first_sub_idx: VkDeviceBuffer = self.d_first_subchunk_idx;

        // VK adaptation: params[] layout per module_loader KERNEL_DECLS contract
        // (see scan_gpu.zig:58-64 and the lz_decode_raw_kernel sibling at
        // decode_dispatch.zig:597-604 for working reference). First n_bindings
        // entries are pointers to VkDeviceBuffer handles populating descriptor
        // set bindings 0..n_bindings-1; remaining entries pack into
        // push_constant_size bytes in declaration order. lz_decode_kernel.comp:
        // n_bindings=6 (CompressedBuf, ChunksBuf, DstBuf, TotalChunksBuf,
        // EntropyScratchBuf, FirstSubChunkIdxBuf), push_constant_size=16
        // (chunks_per_group as u32, sub_chunk_cap as u32, entropy_slot_stride
        // as uvec2/u64). CUDA's kernel argument list is unchanged; only the
        // VK host packing order changes — buffer-pointer params first, then
        // push-constant scalars.
        // 2026-06-10 (A-024 revision): bindings 6 / 7 are the tok / off16
        // region views of the SAME entropy-scratch buffer, bound at
        // last_tok_offset / last_off16_offset so every in-shader byte
        // offset stays region-relative (≤ 2.14 GB). The stride push
        // constant remains in the ABI but the kernel no longer reads it.
        var p_entropy_scratch_tok: u64 = self.d_entropy_scratch;
        var p_entropy_scratch_off16: u64 = self.d_entropy_scratch;
        var lz_params = [_]?*anyopaque{
            @ptrCast(&common.comp),
            @ptrCast(&common.descs_dev),
            @ptrCast(&common.dst),
            @ptrCast(&common.total),
            @ptrCast(&p_entropy_scratch),
            @ptrCast(&p_first_sub_idx),
            @ptrCast(&p_entropy_scratch_tok),
            @ptrCast(&p_entropy_scratch_off16),
            @ptrCast(&common.chunks_per_group),
            @ptrCast(&common.sub_chunk_cap),
            @ptrCast(&p_entropy_slot_stride),
        };
        var lz_extra = [_]?*anyopaque{null};
        const lz_offs = [_]u64{ 0, 0, 0, 0, 0, 0, self.last_tok_offset, self.last_off16_offset };

        const t_lz = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeKernel", stream);
        try vkCall(launch_fn(module_loader.kernel_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &lz_params, &lz_extra, &lz_offs), .launch);
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
    // Iter 13: VK adaptation — uploadInputAndPrefixSum stages the LZ
    // kernel's self-gate count (the 4 B n_chunks H2D) before the
    // early-submit of the transfer cmdbuf, so runBackHalf no longer
    // owns this copy. The resolved device address (either the caller's
    // `d_n_chunks_dev` or `self.d_n_groups_scratch`) is passed through
    // on the FrameLayout for the LZ kernel's `total` param.
    //
    // Pre-iter-13 runBackHalf did this 4 B H2D inline. Moving it up
    // ensures all H2D vkCmdCopyBuffers are recorded into the transfer
    // cmdbuf BEFORE procs.stream_flush_transfer end+submits it, so
    // the dedicated DMA engine can start the H2D in parallel with the
    // host-side kernel-record + descriptor-binding work in runBackHalf.
    lz_total_count_dev: VkDeviceBuffer,
};

/// CUDA reference: src/decode/decode_dispatch.zig:503-579. Upload
/// phase: ensure persistent buffers, H2D/D2D the chunk descs, H2D/D2D
/// the compressed block, run the device-side prefix sum, and size the
/// entropy scratch. Returns the FrameLayout the back-half kernels need.
pub fn uploadInputAndPrefixSum(
    self: *DecodeContext,
    req: DecodeRequest,
    procs: *const VkProcs,
    /// Iter 13: VK adaptation — when true, after recording all H2Ds
    /// into the work_stream's transfer cmdbuf, this function end+submits
    /// it EARLY via procs.stream_flush_transfer so the dedicated DMA
    /// engine starts the H2D copies in parallel with runBackHalf's host
    /// prep. Gated on sync mode (mirrors iter-4 batching scope).
    is_sync_mode: bool,
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

    // Skip the D2H of `total_subchunks` and compute the count on host.
    // 2026-06-10 (CUDA-mirror): when the host can read the chunk descs
    // (the CLI / host-bounce path), sum the EXACT per-chunk sub-chunk
    // count, mirroring slzPrefixSumChunksKernel's loop term for term —
    // the old worst-case bound (chunk_count × ceil(256 KB / cap)) was
    // 2× the actual at sc=0.25 (64 KB chunks → 1 sub-chunk each), which
    // ballooned the 1 GB L3+ entropy scratch to 12 GB (a hard VMA
    // failure under strict DEVICE_LOCAL) and pushed the off16 region
    // offset past the 4 GiB u32 push-constant ceiling (A-024 residual).
    // With the exact count, 1 GB sc=0.25 sits at ~6 GB scratch and the
    // region offsets stay under 2^32. The D2D path
    // (d_chunk_descs_override != null) passes an undefined host slice,
    // so it keeps the worst-case bound.
    const chunk_size_bytes: u32 = 0x40000; // = `constants.chunk_size`, inlined to avoid the format-layer import
    const total_subchunks: u32 = blk: {
        const max_sub_per_chunk: u32 = if (req.sub_chunk_cap == 0) 4 else @max(1, (chunk_size_bytes + req.sub_chunk_cap - 1) / req.sub_chunk_cap);
        if (req.d_chunk_descs_override != null or req.sub_chunk_cap == 0)
            break :blk @as(u32, @intCast(req.chunk_descs.len)) * max_sub_per_chunk;
        var total: u32 = 0;
        for (req.chunk_descs) |ch| {
            // Mirror slzPrefixSumChunksKernel: non-LZ chunks
            // (uncompressed / memset / empty) occupy one slot.
            const n_subs: u32 = if (ch.flags != 0 or ch.decomp_size == 0)
                1
            else
                (ch.decomp_size + req.sub_chunk_cap - 1) / req.sub_chunk_cap;
            total += n_subs;
        }
        break :blk total;
    };
    // descriptors.ENTROPY_SCRATCH_SLOT_BYTES holds the largest sub-chunk's
    // lit/tok streams; off16-hi at +0, off16-lo at +OFF16_HILO_SPLIT_OFFSET
    // within each slot. Layout: [lit: total*slot] [tok: total*slot] [off16: total*slot].
    const per_subchunk_scratch: usize = @intCast(descriptors.ENTROPY_SCRATCH_SLOT_BYTES);
    const entropy_scratch_bytes = @as(usize, total_subchunks) * per_subchunk_scratch * 3;
    const tok_offset = @as(usize, total_subchunks) * per_subchunk_scratch;
    const off16_offset = @as(usize, total_subchunks) * per_subchunk_scratch * 2;
    // L1/L2 gate (CUDA-mirror, 2026-06-10): entropy_scratch is consumed
    // only by the Huffman predecode → gather → LZ-general chain
    // (level >= 3). L1/L2 frames contain no type-4 chunks, so the
    // pointer is never dereferenced; skip the multi-GB allocation.
    if (req.level >= 3) {
        try ensureDeviceBuf(&self.d_entropy_scratch, &self.d_entropy_scratch_size, entropy_scratch_bytes);
        // Iter 4c: store the base handle + offset; the gather kernel binds
        // the sub-region via procLaunchKernel.binding_offsets.
        self.d_entropy_off16_scratch = self.d_entropy_scratch;
        self.d_entropy_off16_offset = off16_offset;
    }
    // A-024: stash region offsets for the Huff decode kernel push
    // constants. Must match the entropy_scratch layout above.
    self.last_tok_offset = @intCast(tok_offset);
    self.last_off16_offset = @intCast(off16_offset);

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
            // A-025: the compressed block sits at a byte offset inside
            // the caller's frame buffer; the offset travels separately
            // (registry-index handles cannot carry pointer arithmetic).
            const d2d_off = procs.d2d_input_offset orelse procs.d2d_offset orelse return error.BackendNotAvailable;
            try vkCall(d2d_off(self.d_comp_persist, dev_src, req.d_compressed_src_offset, req.compressed_block.len, self.work_stream), .copy);
        } else {
            const _t_h2d_comp0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
            try vkCall(h2dRoute(procs, self.work_stream, self.d_comp_persist, @ptrCast(req.compressed_block.ptr), req.compressed_block.len), .copy);
            if (g_phase_profile_enabled) g_phase_upload_h2d_ns += qpcDeltaNs(_t_h2d_comp0, vk.qpcNow());
        }
    }

    // Iter 13: VK adaptation — stage the LZ self-gate count (4 B
    // n_chunks H2D) HERE rather than in runBackHalf. Combined with the
    // procs.stream_flush_transfer call below, this gives the dedicated
    // DMA engine the full H2D batch up-front so it can run in parallel
    // with the host-side prep work runBackHalf does (kernel record,
    // descriptor binding, prepareImportedOutputBuffer). The resolved
    // device address is returned via FrameLayout so runBackHalf doesn't
    // need to know which path was taken.
    //
    // host_total_chunks must outlive the h2dRoute call only until the
    // procH2DAsync staging-bump memcpy completes (immediately on
    // return) — local-variable lifetime here is fine because the H2D
    // either consumes the host bytes synchronously (staging path) or
    // imports a separate page-aligned buffer (the comp_input import
    // does not apply at 4 B sizes — below H2D_IMPORT_THRESHOLD).
    const lz_total_count_dev: VkDeviceBuffer = if (req.d_n_chunks_dev) |dev| dev else blk: {
        try ensureDeviceBuf(&self.d_n_groups_scratch, &self.d_n_groups_scratch_size, 4);
        var host_total_chunks: u32 = @intCast(req.chunk_descs.len);
        const _t_h2d_n0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        try vkCall(h2dRoute(procs, self.work_stream, self.d_n_groups_scratch, @ptrCast(&host_total_chunks), 4), .copy);
        if (g_phase_profile_enabled) g_phase_backhalf_h2d_count_ns += qpcDeltaNs(_t_h2d_n0, vk.qpcNow());
        break :blk self.d_n_groups_scratch;
    };

    if (g_phase_profile_enabled) g_phase_upload_misc_ns += qpcDeltaNs(_t_upload_misc_resume, vk.qpcNow());

    // VK adaptation: end+submit the transfer cmdbuf at end of upload so
    // the dedicated DMA engine starts H2D in parallel with the host
    // runBackHalf prep (kernel record + descriptor binding +
    // prepareImportedOutputBuffer). Mirrors CUDA cuMemcpyHtoD's natural
    // ordering where the synchronous host call has already drained the
    // copy engine by the time the back-half stream_sync fires.
    //
    // Only fired in sync mode — async callers manage their own stream
    // and may queue further H2Ds after this point. In sync mode
    // self.work_stream is the dispatcher-promoted pipeline_stream
    // (iter 4) and we control every subsequent procs.* call, so the
    // "no more H2Ds after flush" invariant is preserved (the only
    // post-upload H2D in the L1 path — the 4 B n_chunks copy — was
    // moved into this function above).
    if (is_sync_mode) {
        if (procs.stream_flush_transfer) |flush| {
            _ = flush(self.work_stream);
        }
    }

    return .{
        .total_subchunks = total_subchunks,
        .tok_offset = tok_offset,
        .off16_offset = off16_offset,
        .lz_total_count_dev = lz_total_count_dev,
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

    // Iter 13: VK adaptation — the LZ self-gate count was staged in
    // uploadInputAndPrefixSum (see FrameLayout.lz_total_count_dev) so
    // the 4 B H2D is part of the transfer cmdbuf the early-submit
    // flush dispatched on the DMA engine. runBackHalf only reads the
    // resolved device address here.
    const lz_total_count_dev: u64 = layout.lz_total_count_dev;

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
        // VK adaptation: Vulkan needs an explicit COMPUTE→COMPUTE pipeline
        // barrier between the Huffman decode kernel (which writes
        // entropy_scratch) and the LZ general decoder kernel below
        // (which reads from it). CUDA's per-launch implicit ordering on
        // the same stream covers this dependency invisibly; the VK port
        // must record it on the cmdbuf. Without it the LZ kernel can
        // observe stale entropy_scratch and emit garbage output.
        if (procs.compute_to_compute_barrier) |barrier_fn| {
            try vkCall(barrier_fn(heavy_stream), .sync);
        }
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

    // Iter 15: VK adaptation — back-half stream sync REMOVED. Pre-iter-15
    // this was a stream_sync_fn(heavy_stream) call that ended+submitted
    // the compute cmdbuf and vkWaitForFences'd the result, then
    // finalizeOutput opened a FRESH cmdbuf for the D2H — 2 submits + 2
    // waits per decode. Iter 15 leaves the compute cmdbuf OPEN here so
    // finalizeOutput records the D2H into it; fullGpuLaunchImpl's single
    // post-finalize stream_sync submits + waits once.
    //
    // The COMPUTE_SHADER_WRITE (LZ kernel output) → TRANSFER_READ (D2H
    // vkCmdCopyBuffer) dependency that used to be enforced by the submit
    // boundary now needs an explicit vkCmdPipelineBarrier on the cmdbuf;
    // module_loader.procD2HAsync records it as its first cmd before any
    // vkCmdCopyBuffer (recordComputeToTransferBarrier). The
    // backhalf_fence_wait phase profile field stays zero — that wait
    // moves entirely into finalize_sync, which already accounts for it.
    //
    // Async-mode (work_stream != 0) callers manage their own cmdbuf
    // lifecycle, so the iter-15 change only affects the sync path.
    // `is_sync_mode` and `stream_sync_fn` (captured at function entry)
    // remain unused on the back-half tail path; they retain their roles
    // in the front-half cross-stream barrier above and in the split_timer
    // Huff sync slice.
    _ = is_sync_mode;

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

    // Iter 9: pre-create / pre-fetch the imported host SOURCE buffer
    // wrapping the caller's compressed-input pointer. The actual H2D
    // runs inside uploadInputAndPrefixSum on self.work_stream; doing
    // the import (vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory)
    // BEFORE that overlaps the ~200 us cache-miss cost with VkProcs
    // resolution and any other dispatcher prep, and is ~0 on cache hit.
    // Stashed on self.work_stream so procH2DAsync matches on consumption.
    // Gated by the same eligibility constraints as the inline path
    // (alignment / extension support / size threshold-equivalent — we
    // only stash for the big comp_input, the tiny chunk_descs H2D will
    // bypass the import even if a stash existed).
    //
    // Iter 12: RE-ENABLED. The iter-9 regression that motivated the
    // disable was that the import-path H2D landed on the universal
    // COMPUTE queue (per iter-4 batching), bypassing NVIDIA's dedicated
    // copy engine. Iter 11 added a dedicated VK_QUEUE_TRANSFER_BIT-only
    // queue and routes every H2D vkCmdCopyBuffer through it, so the
    // imported-memory DMA now executes on the same fast path the
    // staging copy was using — minus the 3.5 ms host @memcpy of
    // the 58 MB block.
    if (req.d_compressed_src == null and req.compressed_block.len >= (4 * 1024 * 1024)) {
        const _t_prep_h2d0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        const src_ptr_const: *const anyopaque = @ptrCast(req.compressed_block.ptr);
        const src_addr = @intFromPtr(src_ptr_const);
        if (module_loader.prepareImportHostBufferForUpload(src_ptr_const, req.compressed_block.len)) |prep| {
            _ = module_loader.stashPreparedUploadImportForStream(
                self.work_stream,
                src_addr,
                req.compressed_block.len,
                prep,
            );
        }
        if (g_phase_profile_enabled) g_phase_import_prep_ns += qpcDeltaNs(_t_prep_h2d0, vk.qpcNow());
    }

    // Phase 1: upload + prefix sum (always runs — needed for chunk
    // descriptor sizing on L1 and L2+).
    const _t_upload0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    const layout = try uploadInputAndPrefixSum(self, req, &procs, is_sync_mode);
    if (g_phase_profile_enabled) g_phase_upload_ns += qpcDeltaNs(_t_upload0, vk.qpcNow());
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.h2d = nsSince(t0, io_val);
        e2e_cum.postscan = e2e_cum.h2d;
    };

    // Phase 2-3: scan/compact/merge/gather + Huffman descriptor merge.
    //
    // 2026-06-10 (CUDA-mirror): gate tightened from level>=2 to
    // level>=3 — the encoder front door only runs the Huffman pass at
    // level >= 3, so L2 frames have no entropy streams either and the
    // whole chain ran as worst-case-grid no-ops on L2. CUDA backported
    // the original VK gate and tightened it (db1e061); this re-syncs.
    var scan: ScanResult = .{ .num_raw_off16 = 0 };
    var n_huff: u32 = 0;
    var have_huff = false;
    if (req.level >= 3) {
        // Phase 2: scan + raw-off16 gather. The GPU scan kernel reads
        // from `d_comp_persist` (filled by the prior H2D / D2D), so the
        // same code path handles both decode entry points.
        if (module_loader.huff_build_fn != 0) {
            if (req.chunk_descs.len > descriptors.WALK_MAX_CHUNKS) return error.BadMode;

            // Allocate L2-only scratch slots before the scan kernel
            // runs. Each ensureDeviceBuf is a grow-only no-op on
            // subsequent decodes of the same level.
            // Iter 4c: 6 strided u32 slots (256 B each = SSBO_ALIGN) so
            // each per-stream count binds at a Vulkan-aligned offset via
            // procLaunchKernel.binding_offsets. Matches the layout
            // scan_gpu.zig::gpuScanChunks computes (COUNTS_STRIDE * 6).
            try ensureDeviceBuf(&self.d_compact_counts, &self.d_compact_counts_size, 256 * 6);
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
    // Iter 15: VK adaptation — single-submit consolidation. Pre-iter-15
    // runBackHalf fired stream_sync at the end of the back half (submit
    // 1 + wait 1), finalizeOutput opened a fresh cmdbuf and recorded its
    // D2H, then this block fired the second stream_sync (submit 2 + wait
    // 2). Iter 15 keeps the cmdbuf open through runBackHalf so
    // finalizeOutput's d2h_async records into the same cmdbuf as the LZ
    // kernel; the single stream_sync below submits + waits once.
    //
    // The sync ALSO has to fire on the writesDirectlyToTarget path:
    // finalizeOutput is skipped there (LZ writes straight to req.
    // d_output_target) but the LZ kernel still needs to land before this
    // call returns to the host caller. Pre-iter-15 the back-half
    // stream_sync covered it; iter-15 needs the post-finalize sync to
    // handle it instead.
    const _t_fin0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
    if (!req.writesDirectlyToTarget()) {
        try finalizeOutput(self, &procs, req, heavy_stream);
    }
    if (is_sync_mode) {
        const _t_fsync0 = if (g_phase_profile_enabled) vk.qpcNow() else 0;
        try vkCall(procs.stream_sync(heavy_stream), .sync);
        if (g_phase_profile_enabled) g_phase_finalize_sync_ns += qpcDeltaNs(_t_fsync0, vk.qpcNow());
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
