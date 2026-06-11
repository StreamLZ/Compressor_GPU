//! Top-level decode pipeline orchestration: walk → prefix-sum → scan →
//! compact/merge → Huff predecode → LZ decode → finalize.
//!
//! Holds the per-decode helpers extracted from the original monolithic
//! `fullGpuLaunchImpl`: `emitE2eTrace` (SLZ_E2E_TIMER print),
//! `gatherRawOff16` (raw off16 scatter), and `mergeHuffDescs` (Huff
//! descriptor merge launch).
//!
//! Reads/writes the singleton `g_default` and the `last_*_kernel_ns`
//! telemetry vars on the facade (`driver.zig`) so external callers
//! continue to see them at `gpu_decode.X`.

const std = @import("std");

const cuda = @import("cuda_api.zig");
const module_loader = @import("module_loader.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const scan_gpu = @import("scan_gpu.zig");

const CUdeviceptr = cuda.CUdeviceptr;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;

const ChunkDesc = descriptors.ChunkDesc;
const HuffDecChunkDesc = descriptors.HuffDecChunkDesc;
const RawOff16Desc = descriptors.RawOff16Desc;
const ScanResult = descriptors.ScanResult;
const GpuError = descriptors.GpuError;
const cudaCall = descriptors.cudaCall;
const HUFF_LUT_ENTRIES = descriptors.HUFF_LUT_ENTRIES;

const DecodeContext = decode_context.DecodeContext;
const ensureDeviceBuf = decode_context.ensureDeviceBuf;
const ensureDeviceOutput = decode_context.ensureDeviceOutput;
const beginKernelTiming = decode_context.beginKernelTiming;
const endKernelTiming = decode_context.endKernelTiming;
const finalizeProfiling = decode_context.finalizeProfiling;

/// Bundle of the CUDA Driver API function pointers used by the GPU
/// decode pipeline. Resolved once at `fullGpuLaunchImpl` entry and
/// threaded into the pipeline helpers (`runHuffBuildAndDecode`,
/// `runLzPipeline`, `finalizeOutput`) so they don't each re-resolve the
/// same `cuMemcpyHtoD_fn orelse return error.BackendNotAvailable`
/// patterns. `d2d` is optional because only the D2D source and D2D
/// output paths consult it; each site unwraps with its own `orelse
/// return error.BackendNotAvailable`.
const CudaProcs = struct {
    h2d: cuda.FnMemcpyHtoD,
    d2h: cuda.FnMemcpyDtoH,
    launch: cuda.FnLaunchKernel,
    sync: cuda.FnCtxSync,
    stream_sync: cuda.FnStreamSync,
    d2d: ?cuda.FnMemcpyDtoDAsync,

    fn resolve() GpuError!CudaProcs {
        return .{
            .h2d = cuda.cuMemcpyHtoD_fn orelse return error.BackendNotAvailable,
            .d2h = cuda.cuMemcpyDtoH_fn orelse return error.BackendNotAvailable,
            .launch = cuda.cuLaunchKernel_fn orelse return error.BackendNotAvailable,
            .sync = cuda.cuCtxSynchronize_fn orelse return error.BackendNotAvailable,
            .stream_sync = cuda.cuStreamSync_fn orelse return error.BackendNotAvailable,
            .d2d = cuda.cuMemcpyDtoDAsync_fn,
        };
    }
};

/// The six params shared by both LZ-decode kernel variants (raw and
/// general). Held as a struct so each field has a stable address for
/// the CUDA kernel-params array (which takes `&field` pointers).
const LzCommonParams = struct {
    comp: CUdeviceptr,
    descs_dev: CUdeviceptr,
    dst: CUdeviceptr,
    chunks_per_group: u32,
    /// Device pointer to the LZ kernel's self-gate count.
    total: CUdeviceptr,
    sub_chunk_cap: u32,
};

/// `slzMergeHuffDescsKernel` parameter bundle. Held as a struct so
/// every field has a stable address for the CUDA kernel-params array
/// (which takes `&field` pointers).
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

/// Pre-decode preparation: combine the four per-stream-type Huffman
/// descriptor arrays into one merged device array in `self.d_huff_descs`,
/// with out_offsets adjusted by the per-stream-type region offset and
/// lut_offsets assigned sequentially. Driven entirely by the GPU merge
/// kernel: the compact kernels in `gpuScanChunks` already wrote the
/// four per-stream device buffers (`d_compact_*`), and the merge kernel
/// reads its per-stream counts from `d_compact_counts` and self-gates.
fn mergeHuffDescs(
    self: *DecodeContext,
    tok_offset: usize,
    off16_offset: usize,
    launch_fn: anytype,
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
    // B2 (2026-06-10): prefer the 4-block parallel merge (wall ~= the
    // largest region instead of the sum of all four); the serial
    // kernel stays as a fallback for older PTX.
    const use_par = module_loader.merge_huff_descs_par_fn != 0;
    const merge_fn = if (use_par) module_loader.merge_huff_descs_par_fn else module_loader.merge_huff_descs_fn;
    const grid_x: c_uint = if (use_par) 4 else 1;
    const label: [*:0]const u8 = if (use_par) "slzMergeHuffDescsParKernel (x4)" else "slzMergeHuffDescsKernel";
    const t_merge = beginKernelTiming(self.enable_profiling, &self.pending_timings, label, stream);
    try cudaCall(launch_fn(merge_fn, grid_x, 1, 1, 1, 1, 1, 0, stream, &m_params, &m_extra), .launch);
    endKernelTiming(t_merge, stream);
    // No post-launch sync: downstream kernels (huff_build, huff_decode)
    // are queued on the same stream and see merge's output via stream
    // ordering. In sync mode the pre-back-half cross-stream sync in
    // `fullGpuLaunchImpl` covers it.
}

/// Scatter the raw (type 0) off16 sub-streams from `d_comp_persist`
/// into the per-sub-chunk entropy scratch. All inputs are already
/// device-resident: the descriptor list lives in `d_compact_raw`, the
/// count in `d_compact_counts[4]`, and the source bytes in
/// `d_comp_persist`. A single `slzGatherRawOff16Kernel` launch copies
/// every stream in parallel.
fn gatherRawOff16(
    self: *DecodeContext,
    scan: ScanResult,
    comp_len: u32,
    launch_fn: anytype,
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
    // B2 overlap (2026-06-10): run the gather on a dedicated stream,
    // concurrent with the merge + huff chain on work_stream. Safe
    // because the gather writes RAW sub-chunks' scratch slots while
    // huff_decode writes HUFF sub-chunks' slots (disjoint by
    // sub-chunk). Ordering: gather_stream waits ev_compact_done
    // (compact wrote d_compact_raw + counts on work_stream); the LZ
    // launch waits ev_gather_done (runLzPipeline). Falls back to the
    // legacy inline launch when the event procs or aux objects are
    // unavailable.
    var stream = self.work_stream;
    if (cuda.cuStreamWaitEvent_fn != null and cuda.cuEventRecord_fn != null and ensureGatherOverlap(self)) {
        _ = cuda.cuEventRecord_fn.?(self.ev_compact_done, self.work_stream);
        _ = cuda.cuStreamWaitEvent_fn.?(self.gather_stream, self.ev_compact_done, 0);
        stream = self.gather_stream;
    }
    const t_gather = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzGatherRawOff16Kernel", stream);
    defer endKernelTiming(t_gather, stream);
    try cudaCall(launch_fn(module_loader.gather_off16_fn, grid_x, 1, 1, 256, 1, 1, 0, stream, &params, &extra), .launch);
    if (stream == self.gather_stream and stream != self.work_stream) {
        _ = cuda.cuEventRecord_fn.?(self.ev_gather_done, self.gather_stream);
        self.gather_event_pending = true;
    }
    // No post-launch sync: downstream LZ kernel reads
    // d_entropy_off16_scratch from the same stream.
}

/// SLZ_E2E_TIMER cumulative timestamps, each measured as elapsed
/// nanoseconds since `t0`. Populated incrementally between phases of
/// `fullGpuLaunchImpl` so `emitE2eTrace` can recover per-phase deltas.
const E2eCumulative = struct {
    /// End of upload + prefix-sum (the H2D / D2D of the chunk descs and
    /// compressed bytes plus the device-side prefix-sum kernel). The
    /// `preBlk` column in the trace is the slice from here to
    /// `postscan` (the in-line scan kernel queue), which after the
    /// phase extraction is empty — the column is kept so emitted
    /// columns line up with historical traces.
    h2d: i64 = 0,
    /// Just after the scan kernel, before raw-off16 gather.
    postscan: i64 = 0,
    /// End of the scan phase (after raw-off16 gather).
    scan: i64 = 0,
    /// End of the back half (Huff predecode + LZ pipeline), before
    /// the finalize copy.
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

/// SLZ_E2E_TIMER: per-phase breakdown print at the end of the decode
/// call. `t0` is the start clock reading captured at function entry.
fn emitE2eTrace(
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

/// Bundle of the per-call inputs to `fullGpuLaunchImpl`. Replaces an
/// 11-parameter signature with `(self, req)`. Passed by value; the
/// struct's slice / pointer fields make it ~100 bytes, but it's
/// constructed once per decompress call so the copy cost is invisible
/// next to the kernel work.
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
    /// L1 gate (backported from VK, audit Section C.5.1): for `level == 1`
    /// frames the dispatcher skips the entire scan/compact/merge/gather
    /// pipeline + the Huffman pre-decode kernels (which all self-gate but
    /// still launch with worst-case grid sizes ≈ 4× total_subchunks). L1
    /// has no entropy streams, so none of that work is needed; the LZ
    /// kernel's lean raw path (`slzLzDecodeRawKernel`) reads off16 etc.
    /// directly from the compressed buffer. Default `1` keeps existing
    /// callers behavioral on raw-input paths that don't know the level.
    level: u8 = 1,
    /// v4 #19 device-only verify. When merkle_n_chunks > 0 the
    /// dispatch runs, on device, after the LZ kernel:
    ///   1. slzScPrefixApplyKernel - writes the true first-8 bytes of
    ///      chunks 1+ into the output (source = the SC tail table
    ///      inside the uploaded compressed block at
    ///      `merkle_prefix_off`), making the output FINAL in VRAM;
    ///   2. slzChunkHashKernel - per-chunk XXH32s into the persistent
    ///      device hash buffer at chunk index `merkle_chunk_base`
    ///      (sized for `merkle_total_chunks` across the frame);
    ///   3. on the frame's LAST block (`merkle_run_verdict`):
    ///      slzMerkleVerdictKernel rolls the whole array into the
    ///      root, compares against `merkle_expected_root` (a launch
    ///      scalar read from the frame header area by the host - not
    ///      payload traffic), and the 4-byte verdict is read into
    ///      `merkle_verdict_out.*` alongside the existing readbacks.
    merkle_n_chunks: u32 = 0,
    merkle_chunk_base: u32 = 0,
    merkle_total_chunks: u32 = 0,
    merkle_eff_chunk: u32 = 0,
    merkle_prefix_off: u64 = 0, // 0 = no prefix table (single-chunk block)
    merkle_run_verdict: bool = false,
    merkle_expected_root: u32 = 0,
    merkle_verdict_out: ?*u32 = null,

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

/// Launches the Huffman LUT-build kernel and the 4-stream decode
/// kernel into `heavy_stream`. Both kernels self-gate on `*d_n_huff`,
/// so it is safe to launch them with the worst-case chunk count even
/// if some chunks have no Huffman streams.
///
/// Returns the LUT-build elapsed nanoseconds when `split_timer` is set
/// (driven by `SLZ_SPLIT_TIMER`); otherwise zero. The caller computes
/// the full huff-total time after the back-half stream sync.
fn runHuffBuildAndDecode(
    self: *DecodeContext,
    procs: *const CudaProcs,
    n_huff: u32,
    heavy_stream: usize,
    split_timer: bool,
    t_huff_start: anytype,
    io: ?std.Io,
) GpuError!i64 {
    const launch_fn = procs.launch;

    const huff_stream = heavy_stream;
    var split_huff_build_ns: i64 = 0;

    // The huff kernels self-gate on `*d_n_blocks`. The merge kernel
    // wrote `n_merged` to `d_compact_counts[5]` (offset 20 bytes); the
    // huff kernels read it from there.
    const d_n_huff: u64 = self.d_compact_counts + 20;
    {
        var p_comp: u64 = self.d_comp_persist;
        var p_descs: u64 = self.d_huff_descs;
        var p_lut: u64 = self.d_huff_lut;
        var p_n: u64 = d_n_huff;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs),
            @ptrCast(&p_lut),  @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        const t_hb = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffBuildLutKernel", huff_stream);
        try cudaCall(launch_fn(module_loader.huff_build_fn, n_huff, 1, 1, 32, 1, 1, 0, huff_stream, &params, &extra), .launch);
        endKernelTiming(t_hb, huff_stream);
    }
    // Split fence: time the LUT build separately from the decode.
    if (split_timer) {
        try cudaCall(procs.stream_sync(huff_stream), .sync);
        if (t_huff_start) |hs| {
            if (io) |io_val|
                split_huff_build_ns = nsSince(hs, io_val);
        }
    }
    {
        var p_comp: u64 = self.d_comp_persist;
        var p_descs: u64 = self.d_huff_descs;
        var p_lut: u64 = self.d_huff_lut;
        var p_out: u64 = self.d_entropy_scratch;
        var p_n: u64 = d_n_huff;
        // A-024: pass region offsets as u64 + d_compact_counts pointer so
        // each block reads its slot's region from there. The merge kernel
        // no longer folds these into desc.out_offset (which is u32 and
        // overflowed at ~6553 sub-chunks).
        var p_counts: u64 = self.d_compact_counts;
        var p_tok_off: u64 = self.last_tok_offset;
        var p_off16_off: u64 = self.last_off16_offset;
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs),
            @ptrCast(&p_lut),  @ptrCast(&p_out),
            @ptrCast(&p_n),
            @ptrCast(&p_counts),
            @ptrCast(&p_tok_off),
            @ptrCast(&p_off16_off),
        };
        var extra = [_]?*anyopaque{null};
        const shared_bytes: c_uint = HUFF_LUT_ENTRIES * @sizeOf(u32);
        const t_hd = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffDecode4StreamKernel", huff_stream);
        try cudaCall(launch_fn(module_loader.huff_decode_fn, n_huff, 1, 1, 32, 1, 1, shared_bytes, huff_stream, &params, &extra), .launch);
        endKernelTiming(t_hd, huff_stream);
    }
    return split_huff_build_ns;
}

/// Launches the LZ decode kernel. Uses the lean raw kernel when no
/// Huffman streams are present (L1/L2 path); otherwise the general
/// kernel that consumes `d_entropy_scratch`.
///
/// Returns the kernel's elapsed nanoseconds when `split_timer` is set;
/// otherwise zero.
fn runLzPipeline(
    self: *DecodeContext,
    procs: *const CudaProcs,
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
    lz_total_count_dev: CUdeviceptr,
    /// Base device address the LZ kernel writes decompressed bytes
    /// into. Normally `self.d_output`; on the direct-write fast path
    /// the caller routes `req.d_output_target` here so the kernel
    /// skips the finalize D2D copy.
    lz_dst_base_dev: CUdeviceptr,
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

    // B2 gather-overlap: the LZ kernel reads the raw-off16 scratch
    // slots the gather wrote on gather_stream - make this stream wait
    // for it before the LZ launch.
    if (self.gather_event_pending) {
        self.gather_event_pending = false;
        if (cuda.cuStreamWaitEvent_fn) |wf| {
            try cudaCall(wf(stream, self.ev_gather_done, 0), .sync);
        }
    }

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
        // v4 #15 (2026-06-11): pipelined kernel — the warps of a block
        // cooperate on the SAME group (warp 0 parses batch N+1 while a
        // 3-warp copier team executes batch N's copies from a shared
        // double buffer; team-internal order uses a named barrier so
        // the parser never waits mid-batch), grid doubled to
        // compensate, blockDim.y = 4 (128 threads). Hides the
        // long-scoreboard memory-latency stall the post-#1/#2 NCU
        // profile exposed (52.7% SM, 22.9 stall). Default ON;
        // SLZ_NO_PIPELINE=1 is the debug escape back to the
        // single-warp kernel.
        const use_pipeline = module_loader.kernel_raw_pipeline_fn != 0 and
            std.c.getenv("SLZ_NO_PIPELINE") == null;
        const raw_kernel = if (use_pipeline) module_loader.kernel_raw_pipeline_fn else module_loader.kernel_raw_fn;
        const raw_grid_x: u32 = if (use_pipeline) lz_grid_x * 2 else lz_grid_x;
        const raw_block_y: u32 = if (use_pipeline) 4 else 2;
        var raw_params = [_]?*anyopaque{
            @ptrCast(&common.comp),
            @ptrCast(&common.descs_dev),
            @ptrCast(&common.dst),
            @ptrCast(&common.chunks_per_group),
            @ptrCast(&common.total),
            @ptrCast(&common.sub_chunk_cap),
        };
        var raw_extra = [_]?*anyopaque{null};
        const label: [*:0]const u8 = if (use_pipeline) "slzLzDecodeRawPipelinedKernel" else "slzLzDecodeRawKernel";
        const t_lzr = beginKernelTiming(self.enable_profiling, &self.pending_timings, label, stream);
        try cudaCall(launch_fn(raw_kernel, raw_grid_x, 1, 1, 32, raw_block_y, 1, 0, stream, &raw_params, &raw_extra), .launch);
        endKernelTiming(t_lzr, stream);
    } else {
        var p_entropy_scratch: u64 = self.d_entropy_scratch;
        var p_entropy_slot_stride: u64 = @as(u64, total_subchunks) * descriptors.ENTROPY_SCRATCH_SLOT_BYTES;
        var p_first_sub_idx: CUdeviceptr = self.d_first_subchunk_idx;

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

        // v4 #15 L3+ port (2026-06-11): same K=4 pipeline as the raw
        // branch above — one group per block, parser warp + 3-warp
        // copier team, blockDim (32,4), grid doubled. Sub-chunks that
        // are not mode-1/off32-free fall back to the warp-level
        // general decoder inside the kernel. SLZ_NO_PIPELINE=1 escapes.
        const use_gen_pipeline = module_loader.kernel_general_pipeline_fn != 0 and
            std.c.getenv("SLZ_NO_PIPELINE") == null;
        const gen_kernel = if (use_gen_pipeline) module_loader.kernel_general_pipeline_fn else module_loader.kernel_fn;
        const gen_grid_x: u32 = if (use_gen_pipeline) lz_grid_x * 2 else lz_grid_x;
        const gen_block_y: u32 = if (use_gen_pipeline) 4 else 2;
        const gen_label: [*:0]const u8 = if (use_gen_pipeline) "slzLzDecodeGeneralPipelinedKernel" else "slzLzDecodeKernel";
        const t_lz = beginKernelTiming(self.enable_profiling, &self.pending_timings, gen_label, stream);
        try cudaCall(launch_fn(gen_kernel, gen_grid_x, 1, 1, 32, gen_block_y, 1, 0, stream, &lz_params, &lz_extra), .launch);
        endKernelTiming(t_lz, stream);
    }

    if (split_timer) {
        try cudaCall(stream_sync_fn(stream), .sync);
        if (t_lz_start) |ts_start| {
            if (io) |io_val| {
                split_lz_ns += nsSince(ts_start, io_val);
            }
        }
    }

    return split_lz_ns;
}

/// Writes the decompressed output to the caller-supplied destination.
/// Two paths:
///   - D2D: the caller passed a device-resident target; issue a D2D
///     async copy on `heavy_stream` so it serializes after the LZ
///     kernel on the same stream. The caller's `cudaStreamSynchronize`
///     on that stream waits for the result.
///   - D2H: the caller wants the result on the host; the synchronous
///     `cuMemcpyDtoH` blocks until the copy completes.
///
/// In sync mode (work_stream == 0) the D2D path also issues a
/// ctx-wide sync to ensure the caller observes a settled output
/// when this function returns.
fn finalizeOutput(self: *DecodeContext, procs: *const CudaProcs, req: DecodeRequest, heavy_stream: usize) GpuError!void {
    if (req.d_output_target) |dev_target| {
        const d2d = procs.d2d orelse return error.BackendNotAvailable;
        try cudaCall(d2d(dev_target + req.dst_start_off, self.d_output + req.dst_start_off, req.decompressed_size, heavy_stream), .copy);
        if (self.work_stream == 0) try cudaCall(procs.sync(), .sync);
    } else {
        try cudaCall(procs.d2h(@ptrCast(req.dst_full + req.dst_start_off), self.d_output + req.dst_start_off, req.decompressed_size), .copy);
    }
}

/// Result of the input-upload + prefix-sum phase: the worst-case
/// sub-chunk count drives entropy-scratch sizing, and the per-stream
/// region offsets within the scratch must match the merge-kernel
/// writer and the LZ-kernel reader.
const FrameLayout = struct {
    total_subchunks: u32,
    tok_offset: usize,
    off16_offset: usize,
};

/// Upload phase: ensure persistent buffers, H2D/D2D the chunk descs,
/// H2D/D2D the compressed block, run the device-side prefix sum, and
/// size the entropy scratch. Returns the FrameLayout the back-half
/// kernels need.
fn uploadInputAndPrefixSum(
    self: *DecodeContext,
    req: DecodeRequest,
    procs: *const CudaProcs,
) GpuError!FrameLayout {
    const total_output = req.dst_start_off + req.decompressed_size;
    try ensureDeviceOutput(self, total_output + 64);
    if (req.dst_start_off > 0)
        try cudaCall(procs.h2d(self.d_output, @ptrCast(req.dst_full), req.dst_start_off), .copy);

    // `cuMemAlloc(0)` is a CUDA error; route an empty-input frame
    // through a small dummy allocation so the ensureDeviceBuf below
    // succeeds and the empty input flows through to the end-mark
    // emit without a special case.
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
        try cudaCall(d2d(self.d_descs_persist, dev_ptr, desc_bytes, self.work_stream), .copy);
    } else {
        try cudaCall(procs.h2d(self.d_descs_persist, @ptrCast(req.chunk_descs.ptr), desc_bytes), .copy);
    }

    // `gpuPrefixSumChunksImpl` returns `GpuError!T` (not `?T`) because
    // there is no host fallback - every GPU decode path needs it.
    _ = try scan_gpu.gpuPrefixSumChunksImpl(self, self.d_descs_persist, @intCast(req.chunk_descs.len), req.sub_chunk_cap);

    // Skip the D2H of `total_subchunks` and compute the count on host.
    // 2026-06-10: when the host can read the chunk descs (the CLI /
    // host-bounce path), sum the EXACT per-chunk sub-chunk count,
    // mirroring slzPrefixSumChunksKernel term for term — the old
    // worst-case bound (chunk_count × ceil(256 KB / cap)) was 2× the
    // actual at sc=0.25 (64 KB chunks → 1 sub-chunk each), sizing the
    // 1 GB L3+ entropy scratch at 12 GB instead of 6 GB. Also what
    // keeps the VK port's u32 region offsets under 2^32 at 1 GB
    // (A-024 residual). The D2D path (d_chunk_descs_override != null)
    // passes an undefined host slice, so it keeps the worst-case bound.
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
    // L1/L2 gate: entropy_scratch is consumed only by the Huffman
    // predecode → gather → LZ-general chain (level >= 3). L1/L2 decode
    // through the raw kernel and never dereference it, but the previous
    // unconditional allocation still cost real VRAM from the BOUND-based
    // sizing: 3 GB at 1 GB sc=0.5, 12 GB(!) at 1 GB sc=0.25. Skip it.
    // The raw/general kernels treat a null scratch pointer as "no
    // entropy" (L1/L2 frames contain no type-4 chunks, so the pointer
    // is never dereferenced even on the general-kernel fallback path).
    if (req.level >= 3) {
        try ensureDeviceBuf(&self.d_entropy_scratch, &self.d_entropy_scratch_size, entropy_scratch_bytes);
        self.d_entropy_off16_scratch = self.d_entropy_scratch + off16_offset;
    }
    // A-024: stash region offsets as u64 for slzHuffDecode4StreamKernel.
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
    // the H2D path is host-synchronous, so there is no async hand-off
    // to wait on.
    if (req.compressed_block.len > 0) {
        if (req.d_compressed_src) |dev_src| {
            const d2d = procs.d2d orelse return error.BackendNotAvailable;
            try cudaCall(d2d(self.d_comp_persist, dev_src, req.compressed_block.len, self.work_stream), .copy);
        } else {
            try cudaCall(procs.h2d(self.d_comp_persist, @ptrCast(req.compressed_block.ptr), req.compressed_block.len), .copy);
        }
    }

    return .{
        .total_subchunks = total_subchunks,
        .tok_offset = tok_offset,
        .off16_offset = off16_offset,
    };
}

/// Back-half phase: pick the heavy stream, stage the LZ self-gate
/// count, run the Huffman predecode (if any), run the LZ pipeline,
/// then sync. Records `last_*_kernel_ns` onto the facade on the way
/// out. The caller is responsible for the e2e timer slice that
/// surrounds this call.
fn runBackHalf(
    self: *DecodeContext,
    req: DecodeRequest,
    procs: *const CudaProcs,
    layout: FrameLayout,
    n_huff: u32,
    have_huff: bool,
    total_chunks: u32,
    heavy_stream: usize,
    facade: anytype,
) GpuError!void {
    const io = req.io;
    const stream_sync_fn = procs.stream_sync;

    // Cross-stream barrier: in sync mode `heavy_stream` is the library-
    // owned `pipeline_stream` so the front-half work queued on
    // stream 0 has to drain before the back half can read its results.
    if (heavy_stream != self.work_stream) {
        try cudaCall(stream_sync_fn(self.work_stream), .sync);
    }

    // KERNEL TIMER: only pure GPU kernel time from here.
    const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    // SLZ_SPLIT_TIMER: separate Huff predecode from LZ decode timing.
    const split_timer = std.c.getenv("SLZ_SPLIT_TIMER") != null;

    // Stage the LZ kernel's self-gate count. D2D path: the caller
    // supplies `d_n_chunks_dev` (typically `d_walk_meta + offset`).
    // Else stage `total_chunks` into `d_n_groups_scratch` via 4 B H2D.
    const lz_total_count_dev: u64 = if (req.d_n_chunks_dev) |dev| dev else blk: {
        try ensureDeviceBuf(&self.d_n_groups_scratch, &self.d_n_groups_scratch_size, 4);
        var host_total_chunks: u32 = total_chunks;
        try cudaCall(procs.h2d(self.d_n_groups_scratch, @ptrCast(&host_total_chunks), 4), .copy);
        break :blk self.d_n_groups_scratch;
    };

    // Direct-write-to-output fast path: when the caller supplied a
    // device output target AND there's no host prefix to splice in,
    // the LZ kernel writes straight to the caller's buffer, skipping
    // the finalize D2D copy.
    const lz_dst_base_dev: CUdeviceptr = if (req.writesDirectlyToTarget()) req.d_output_target.? else self.d_output;

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
        try cudaCall(stream_sync_fn(heavy_stream), .sync);
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

    const split_lz_ns = try runLzPipeline(
        self, procs,
        req.chunks_per_group, req.sub_chunk_cap, n_huff,
        total_chunks, layout.total_subchunks,
        heavy_stream, split_timer, io,
        lz_total_count_dev, lz_dst_base_dev,
    );

    // Back-half stream sync: skip in async mode (caller's stream
    // carries the queued work; they sync themselves).
    if (self.work_stream == 0) {
        const sync_rc = stream_sync_fn(heavy_stream);
        if (sync_rc != CUDA_SUCCESS) {
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

pub fn fullGpuLaunchImpl(self: *DecodeContext, req: DecodeRequest) GpuError!void {
    if (!module_loader.init()) return error.BackendNotAvailable;
    if (module_loader.kernel_fn == 0) return error.KernelMissing;
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

    const procs = try CudaProcs.resolve();

    // Phase 1: upload + prefix sum.
    const layout = try uploadInputAndPrefixSum(self, req, &procs);
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.h2d = nsSince(t0, io_val);
        // `postscan` defaults to the same boundary so the trace columns
        // collapse cleanly when the scan-skipped path runs (rare:
        // requires `huff_build_fn == 0`).
        e2e_cum.postscan = e2e_cum.h2d;
    };

    // Phase 2-3: scan + raw-off16 gather + Huffman descriptor merge.
    //
    // L1/L2 gate (tightened from the VK port's `level >= 2`): L1 AND L2
    // frames have no entropy streams — the encoder front-door
    // (`fast_framed.zig:247`) only runs the Huffman pass at `level >= 3`.
    // The LZ raw kernel reads everything directly from the compressed
    // buffer. The scan + compact + gather + merge + huff_build +
    // huff_decode kernels all run as expensive no-ops on L1/L2 (each
    // launches with worst-case grid ≈ 4 × total_subchunks and returns
    // immediately via its self-gate). Skip them outright. The VK port
    // currently gates on `level >= 2`; the L2 path there has the same
    // wasteful launches and should be tightened in step.
    var scan: ScanResult = .{ .num_raw_off16 = 0 };
    if (req.level >= 3 and module_loader.huff_build_fn != 0) {
        if (req.chunk_descs.len > descriptors.WALK_MAX_CHUNKS) return error.BadMode;
        scan = scan_gpu.gpuScanChunks(
            self,
            req.chunk_descs,
            req.compressed_block,
            req.sub_chunk_cap,
            layout.total_subchunks,
        ) orelse return error.BackendNotAvailable;
        if (t_e2e0) |t0| if (io) |io_val| {
            e2e_cum.postscan = nsSince(t0, io_val);
        };
        try gatherRawOff16(self, scan, @intCast(req.compressed_block.len), procs.launch);
    }
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.scan = nsSince(t0, io_val);
    };

    // Phase 3: Huffman descriptor merge (LUT + descriptor allocations).
    const n_huff: u32 = scan.num_huff_lit + scan.num_huff_tok +
        scan.num_huff_off16hi + scan.num_huff_off16lo;
    const have_huff = req.level >= 3 and n_huff > 0 and module_loader.huff_build_fn != 0 and module_loader.huff_decode_fn != 0;
    if (have_huff) {
        const huff_desc_bytes = @as(usize, n_huff) * @sizeOf(HuffDecChunkDesc);
        const huff_lut_bytes = @as(usize, n_huff) * HUFF_LUT_ENTRIES * @sizeOf(u32);
        try ensureDeviceBuf(&self.d_huff_descs, &self.d_huff_descs_size, huff_desc_bytes);
        try ensureDeviceBuf(&self.d_huff_lut, &self.d_huff_lut_size, huff_lut_bytes);
        try mergeHuffDescs(self, layout.tok_offset, layout.off16_offset, procs.launch);
    }

    // `heavy_stream` carries the back half (huff + LZ + finalize). In
    // async mode the caller's stream IS the heavy stream; in sync mode
    // we use the library-owned `pipeline_stream` (guaranteed populated
    // by the unconditional `ensurePipelineStream` call above).
    const heavy_stream: usize = if (self.work_stream != 0) self.work_stream else self.pipeline_stream;
    const total_chunks: u32 = @intCast(req.chunk_descs.len);

    // Phase 4: back half (Huff predecode + LZ pipeline + sync + timing).
    try runBackHalf(self, req, &procs, layout, n_huff, have_huff, total_chunks, heavy_stream, facade);
    if (t_e2e0) |t0| if (io) |io_val| {
        e2e_cum.predh = nsSince(t0, io_val);
    };

    // Phase 5: finalize, profiling drain, e2e emit. Skip the finalize
    // D2D when the LZ kernel already wrote straight to the caller's
    // device buffer — same predicate as the LZ dst-pick in runBackHalf.
    if (!req.writesDirectlyToTarget()) {
        // v4 #19 device-only verify (see DecodeRequest field docs).
        // The hash pass re-reads the whole decoded output (~1.1 ms at
        // 100 MB - bandwidth-bound), so it runs on the AUX stream,
        // overlapped under the finalize D2H of the same buffer
        // (read-read, safe): LZ-done event -> aux waits -> hash +
        // combine + verdict on aux, while the D2H streams on the
        // heavy stream. The verdict D2H at the end syncs aux only.
        var merkle_stream: usize = heavy_stream;
        if (req.merkle_n_chunks > 0 and
            module_loader.seg_hash_fn != 0 and
            module_loader.chunk_combine_fn != 0 and
            module_loader.sc_prefix_apply_fn != 0 and
            module_loader.merkle_verdict_fn != 0)
        {
            if (cuda.cuStreamWaitEvent_fn != null and cuda.cuEventRecord_fn != null and
                ensureGatherOverlap(self) and ensureMerkleEvent(self))
            {
                _ = cuda.cuEventRecord_fn.?(self.ev_lz_done_merkle, heavy_stream);
                _ = cuda.cuStreamWaitEvent_fn.?(self.gather_stream, self.ev_lz_done_merkle, 0);
                merkle_stream = self.gather_stream;
            }
            const dst_dev: u64 = if (req.d_output_target) |t_dev| t_dev else self.d_output;
            var p_total_sz: u64 = @intCast(req.decompressed_size);
            // 1. SC prefixes -> output is final on device.
            if (req.merkle_prefix_off != 0 and req.merkle_n_chunks > 1) {
                var a_dst: u64 = dst_dev;
                var a_comp: u64 = self.d_comp_persist;
                var a_off: u64 = req.merkle_prefix_off;
                var a_n: u32 = req.merkle_n_chunks;
                var a_eff: u32 = req.merkle_eff_chunk;
                var ap = [_]?*anyopaque{
                    @ptrCast(&a_dst), @ptrCast(&a_comp), @ptrCast(&a_off),
                    @ptrCast(&a_n), @ptrCast(&a_eff), @ptrCast(&p_total_sz),
                };
                var ax = [_]?*anyopaque{null};
                const a_grid: u32 = (req.merkle_n_chunks + 127) / 128;
                try cudaCall(procs.launch(module_loader.sc_prefix_apply_fn, a_grid, 1, 1, 128, 1, 1, 0, merkle_stream, &ap, &ax), .launch);
            }
            // 2. Hierarchical per-chunk hashes (4 KiB segments -> chunk
            // hash) appended at the frame-global base. One thread per
            // SEGMENT: ~24k threads at 100 MB.
            const spc: u32 = (req.merkle_eff_chunk + 1023) / 1024; // KEEP IN SYNC: SLZ_MERKLE_SEG_BYTES
            const total_hash_bytes = @as(usize, req.merkle_total_chunks) * 4;
            const seg_bytes = @as(usize, req.merkle_n_chunks) * spc * 4;
            try ensureDeviceBuf(&self.d_merkle_hashes, &self.d_merkle_hashes_size, total_hash_bytes);
            try ensureDeviceBuf(&self.d_merkle_seghashes, &self.d_merkle_seghashes_size, seg_bytes);
            var h_data: u64 = dst_dev;
            var h_n: u32 = req.merkle_n_chunks;
            var h_eff: u32 = req.merkle_eff_chunk;
            var h_segs: u64 = self.d_merkle_seghashes;
            var h_prefix: u64 = 0;
            var sp = [_]?*anyopaque{
                @ptrCast(&h_data), @ptrCast(&h_n), @ptrCast(&h_eff),
                @ptrCast(&p_total_sz), @ptrCast(&h_segs), @ptrCast(&h_prefix),
            };
            var sx = [_]?*anyopaque{null};
            const n_segs: u32 = req.merkle_n_chunks * spc;
            const s_grid: u32 = (n_segs + 127) / 128;
            const t_h = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzSegHashKernel", merkle_stream);
            try cudaCall(procs.launch(module_loader.seg_hash_fn, s_grid, 1, 1, 128, 1, 1, 0, merkle_stream, &sp, &sx), .launch);
            endKernelTiming(t_h, heavy_stream);
            var c_out: u64 = self.d_merkle_hashes + @as(u64, req.merkle_chunk_base) * 4;
            var cp = [_]?*anyopaque{
                @ptrCast(&h_segs), @ptrCast(&h_n), @ptrCast(&h_eff),
                @ptrCast(&p_total_sz), @ptrCast(&c_out),
            };
            var cx = [_]?*anyopaque{null};
            const c_grid: u32 = (req.merkle_n_chunks + 127) / 128;
            try cudaCall(procs.launch(module_loader.chunk_combine_fn, c_grid, 1, 1, 128, 1, 1, 0, merkle_stream, &cp, &cx), .launch);
            // 3. Verdict on the frame's last block.
            if (req.merkle_run_verdict) {
                try ensureDeviceBuf(&self.d_merkle_verdict, &self.d_merkle_verdict_size, 4);
                var v_hashes: u64 = self.d_merkle_hashes;
                var v_n: u32 = req.merkle_total_chunks;
                var v_exp: u32 = req.merkle_expected_root;
                var v_out: u64 = self.d_merkle_verdict;
                var vp = [_]?*anyopaque{
                    @ptrCast(&v_hashes), @ptrCast(&v_n), @ptrCast(&v_exp), @ptrCast(&v_out),
                };
                var vx = [_]?*anyopaque{null};
                try cudaCall(procs.launch(module_loader.merkle_verdict_fn, 1, 1, 1, 1, 1, 1, 0, merkle_stream, &vp, &vx), .launch);
                if (req.merkle_verdict_out) |vo| {
                    try cudaCall(procs.stream_sync(merkle_stream), .sync);
                    try cudaCall(procs.d2h(@ptrCast(vo), self.d_merkle_verdict, 4), .copy);
                }
            }
        }
        try finalizeOutput(self, &procs, req, heavy_stream);
    }
    if (self.work_stream == 0) {
        finalizeProfiling(&self.pending_timings, &self.last_timings);
    }
    if (t_e2e0) |t0| if (io) |io_val| {
        emitE2eTrace(t0, io_val, e2e_cum, facade.last_kernel_ns);
    };
}

/// v4 #19: lazily create the LZ-done event the Merkle verify chain
/// waits on (sync-only, like the B2 events).
fn ensureMerkleEvent(self: *DecodeContext) bool {
    if (self.ev_lz_done_merkle != 0) return true;
    const ecreate = cuda.cuEventCreate_fn orelse return false;
    return ecreate(&self.ev_lz_done_merkle, 0x2) == CUDA_SUCCESS;
}

/// B2 gather-overlap: lazily create the aux stream + the two
/// sync-only events (CU_EVENT_DISABLE_TIMING = 0x2). Returns false
/// (caller falls back to the inline gather) when any create fails.
fn ensureGatherOverlap(self: *DecodeContext) bool {
    if (self.gather_stream != 0 and self.ev_compact_done != 0 and self.ev_gather_done != 0) return true;
    const screate = cuda.cuStreamCreate_fn orelse return false;
    const ecreate = cuda.cuEventCreate_fn orelse return false;
    if (self.gather_stream == 0) {
        if (screate(&self.gather_stream, 0) != CUDA_SUCCESS) return false;
    }
    if (self.ev_compact_done == 0) {
        if (ecreate(&self.ev_compact_done, 0x2) != CUDA_SUCCESS) return false;
    }
    if (self.ev_gather_done == 0) {
        if (ecreate(&self.ev_gather_done, 0x2) != CUDA_SUCCESS) return false;
    }
    return true;
}
