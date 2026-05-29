//! Top-level decode pipeline orchestration: walk → prefix-sum → scan →
//! compact/merge → Huff predecode → LZ decode → finalize.
//!
//! Holds the per-decode helpers extracted from the original monolithic
//! `fullGpuLaunchImpl`: `dumpScanIfRequested` (debug snapshot),
//! `emitE2eTrace` (SLZ_E2E_TIMER print), `gatherRawOff16` (raw-off16
//! scratch fill), and `mergeHuffDescs` (GPU or CPU Huff-desc merge).
//!
//! Reads/writes the singleton `g_default` and the `last_*_kernel_ns`
//! telemetry vars on the facade (`driver.zig`) so external callers
//! continue to see them at `gpu_decode.X`.

const std = @import("std");

const cuda = @import("cuda_api.zig");
const ml = @import("module_loader.zig");
const d = @import("descriptors.zig");
const dec_ctx = @import("decode_context.zig");
const scan_gpu_mod = @import("scan_gpu.zig");

const CUdeviceptr = cuda.CUdeviceptr;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;
const NUM_PIPELINE_STREAMS = cuda.NUM_PIPELINE_STREAMS;

const ChunkDesc = d.ChunkDesc;
const HuffDecChunkDesc = d.HuffDecChunkDesc;
const RawOff16Desc = d.RawOff16Desc;
const ScanResult = d.ScanResult;
const GpuError = d.GpuError;
const cudaCall = d.cudaCall;
const HUFF_LUT_ENTRIES = d.HUFF_LUT_ENTRIES;

const DecodeContext = dec_ctx.DecodeContext;
const ensureDeviceBuf = dec_ctx.ensureDeviceBuf;
const ensureDeviceOutput = dec_ctx.ensureDeviceOutput;
const beginKernelTiming = dec_ctx.beginKernelTiming;
const endKernelTiming = dec_ctx.endKernelTiming;
const finalizeProfiling = dec_ctx.finalizeProfiling;

/// Bundle of the CUDA Driver API function pointers used by the GPU
/// decode pipeline. Resolved once at `fullGpuLaunchImpl` entry and
/// threaded into the three pipeline helpers (`runHuffPredecode`,
/// `runLzPipeline`, `finalizeOutput`) so they don't each re-resolve the
/// same `cuMemcpyHtoD_fn orelse return error.BackendNotAvailable`
/// patterns. The required fields are non-optional; `d2d` stays optional
/// because the host-bounce paths work without it - only the D2D-source,
/// D2D-output, and gather-off16 fallback paths consult it, and each
/// site unwraps with its own `orelse return error.BackendNotAvailable`.
const Fns = struct {
    h2d: cuda.FnMemcpyHtoD,
    d2h: cuda.FnMemcpyDtoH,
    launch: cuda.FnLaunchKernel,
    sync: cuda.FnCtxSync,
    stream_sync: cuda.FnStreamSync,
    d2d: ?cuda.FnMemcpyDtoDAsync,

    fn resolve() GpuError!Fns {
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
/// general). Kept as a struct so each field has a stable address for the
/// CUDA kernel-params array (which takes `&field` pointers).
const LzCommonParams = struct {
    comp: CUdeviceptr,
    descs_dev: CUdeviceptr,
    dst: CUdeviceptr,
    cpg: u32, // chunks_per_group
    total: CUdeviceptr, // device pointer to the LZ kernel self-gate count
    sc_cap: u32, // sub_chunk_cap
};

/// Pre-decode preparation: combine the four per-stream-type Huffman
/// descriptor arrays (lit / tok / off16-hi / off16-lo) into one merged
/// device array in `self.d_huff_descs`, with out_offsets adjusted by
/// the per-stream-type region offset and lut_offsets assigned sequentially.
/// Two paths:
///   GPU merge (pure-D2D, when the device compact populated the d_compact_*
///   buffers): launches slzMergeHuffDescsKernel - no host arrays touched.
///   CPU merge (CPU-scan fallback): walks the host-side huff_*_host_buf
///   arrays and uploads the merged result via one H2D.
fn mergeHuffDescs(
    self: *DecodeContext,
    scan: ScanResult,
    tok_offset: usize,
    off16_offset: usize,
    h2d_fn: anytype,
    launch_fn: anytype,
) GpuError!void {
    if (scan.device_compact_populated and ml.merge_huff_descs_fn != 0) {
        // GPU merge: the compact kernels already populated the four
        // per-stream device buffers. Launch the merge kernel - it writes
        // straight to self.d_huff_descs and updates the n_merged slot in
        // d_compact_counts. No host arrays, no H2D.
        var k_lit: u64 = self.d_compact_lit;
        var k_tok: u64 = self.d_compact_tok;
        var k_hi: u64 = self.d_compact_hi;
        var k_lo: u64 = self.d_compact_lo;
        var k_n_lit: u64 = self.d_compact_counts + 0;
        var k_n_tok: u64 = self.d_compact_counts + 4;
        var k_n_hi: u64 = self.d_compact_counts + 8;
        var k_n_lo: u64 = self.d_compact_counts + 12;
        var k_tok_region: u32 = @intCast(tok_offset);
        var k_off16_region: u32 = @intCast(off16_offset);
        var k_merged: u64 = self.d_huff_descs;
        var k_n_merged: u64 = self.d_compact_counts + 20;
        var m_params = [_]?*anyopaque{
            @ptrCast(&k_lit),         @ptrCast(&k_tok),          @ptrCast(&k_hi),       @ptrCast(&k_lo),
            @ptrCast(&k_n_lit),       @ptrCast(&k_n_tok),        @ptrCast(&k_n_hi),     @ptrCast(&k_n_lo),
            @ptrCast(&k_tok_region),  @ptrCast(&k_off16_region),
            @ptrCast(&k_merged),      @ptrCast(&k_n_merged),
        };
        var m_extra = [_]?*anyopaque{null};
        // Phase 2: stream-targeted launch on caller's work_stream.
        const stream = self.work_stream;
        const t_merge = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzMergeHuffDescsKernel", stream);
        try cudaCall(launch_fn(ml.merge_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, stream, &m_params, &m_extra), .launch);
        endKernelTiming(t_merge, stream);
        // No post-launch sync: downstream kernels (huff_build, huff_decode)
        // are queued on the same stream and see merge's output via stream
        // ordering. In sync mode the pre-back-half cross-stream sync in
        // `fullGpuLaunchImpl` covers it. Saves one ~50-300 µs host wait
        // per call.
        return;
    }
    // CPU merge fallback: used when the GPU compact path didn't run
    // (no compact-kernel symbol, or any compact step failed). Storage
    // lives on the `DecodeContext.merged_huff_buf` so the dispatch
    // frame stays small.
    const merged_huff: []HuffDecChunkDesc = self.merged_huff_buf[0..];

    // Upfront capacity check: the sum of per-stream counts must fit in
    // `merged_huff` (sized `MAX_HUFF_DESCS_PER_STREAM × 4`). Each per-
    // stream-type input buffer is sized `MAX_HUFF_DESCS_PER_STREAM`, so
    // the merge can only overflow if a future change relaxes those
    // per-stream bounds. Reject loudly rather than silently truncating.
    const total_huff: usize = @as(usize, scan.num_huff_lit) + scan.num_huff_tok +
        scan.num_huff_off16hi + scan.num_huff_off16lo;
    if (total_huff > merged_huff.len) return error.BadMode;

    var m: u32 = 0;
    var lut_slot: u32 = 0;
    const append = struct {
        fn run(dst: []HuffDecChunkDesc, m_ptr: *u32, lut_ptr: *u32,
               src: []const HuffDecChunkDesc, region_off: u32) void {
            for (src) |s| {
                // Guard is now redundant (total_huff bound above), but
                // kept as a defense-in-depth assert against a future
                // refactor that bypasses the upfront check.
                if (m_ptr.* >= dst.len) return;
                var entry = s;
                entry.out_offset += region_off;
                entry.lut_offset = lut_ptr.* * @as(u32, @intCast(HUFF_LUT_ENTRIES));
                dst[m_ptr.*] = entry;
                m_ptr.* += 1;
                lut_ptr.* += 1;
            }
        }
    }.run;
    append(merged_huff, &m, &lut_slot, self.huff_lit_host_buf[0..scan.num_huff_lit], 0);
    append(merged_huff, &m, &lut_slot, self.huff_tok_host_buf[0..scan.num_huff_tok], @intCast(tok_offset));
    append(merged_huff, &m, &lut_slot, self.huff_off16hi_host_buf[0..scan.num_huff_off16hi], @intCast(off16_offset));
    append(merged_huff, &m, &lut_slot, self.huff_off16lo_host_buf[0..scan.num_huff_off16lo], @intCast(off16_offset));

    try cudaCall(h2d_fn(self.d_huff_descs, @ptrCast(merged_huff.ptr), @as(usize, m) * @sizeOf(HuffDecChunkDesc)), .copy);
    // h2d_fn is sync-on-host (cuMemcpyHtoD); no further sync needed.
}

/// Place raw (type 0) off16 sub-streams into the off16 scratch. The bytes
/// are already on the GPU (d_comp_persist holds the whole compressed blob).
/// Preferred: upload the descriptor list in one H2D and run
/// slzGatherRawOff16Kernel - one launch copies every stream in parallel.
/// Fallbacks: async device-to-device loop, then a plain host-upload loop.
fn gatherRawOff16(
    self: *DecodeContext,
    scan: ScanResult,
    compressed_block: []const u8,
    h2d_fn: anytype,
    launch_fn: anytype,
) GpuError!void {
    if (scan.num_raw_off16 == 0) return;
    if (ml.gather_off16_fn != 0) gather_blk: {
        const ndesc: u32 = scan.num_raw_off16;
        // Pure-D2D path: descs already device-resident in d_compact_raw;
        // skip the H2D and feed the kernel directly.
        const desc_dev: u64 = if (scan.device_compact_populated)
            self.d_compact_raw
        else d_raw: {
            const dbytes: usize = @as(usize, ndesc) * @sizeOf(RawOff16Desc);
            if (!ensureDeviceBuf(&self.d_raw_off16_descs, &self.d_raw_off16_descs_size, dbytes))
                break :gather_blk;
            try cudaCall(h2d_fn(self.d_raw_off16_descs, @ptrCast(&self.raw_off16_buf), dbytes), .copy);
            break :d_raw self.d_raw_off16_descs;
        };
        // Step 7: kernel self-gates on `*d_count`. Use the n_raw slot of
        // d_compact_counts when the GPU compact ran (no D2H needed); else
        // stage the host-known count in d_n_raw_scratch and pass that.
        const d_count: u64 = if (scan.device_compact_populated)
            self.d_compact_counts + 16
        else d_cnt: {
            if (!ensureDeviceBuf(&self.d_n_raw_scratch, &self.d_n_raw_scratch_size, 4))
                break :gather_blk;
            var host_n: u32 = ndesc;
            try cudaCall(h2d_fn(self.d_n_raw_scratch, @ptrCast(&host_n), 4), .copy);
            break :d_cnt self.d_n_raw_scratch;
        };
        {
            var p_comp: u64 = self.d_comp_persist;
            var p_comp_len: u32 = @intCast(compressed_block.len);
            var p_scratch: u64 = self.d_entropy_off16_scratch;
            var p_descs: u64 = desc_dev;
            var p_count: u64 = d_count;
            var params = [_]?*anyopaque{
                @ptrCast(&p_comp), @ptrCast(&p_comp_len), @ptrCast(&p_scratch),
                @ptrCast(&p_descs), @ptrCast(&p_count),
            };
            var extra = [_]?*anyopaque{null};
            // ndesc is exact (host already knows the count); no over-launch.
            // The self-gate inside the kernel makes over-launch safe regardless.
            //
            // The D2D/H2D fallback below is KEPT and the launch treated
            // as a best-effort fast path. Justification: the two failure
            // modes are disjoint - the kernel-launch path needs
            // `cuLaunchKernel + scratch_base + descs`; the fallback path
            // needs only `cuMemcpyDtoDAsync` (or `cuMemcpyHtoD`) + the
            // same buffers. A driver glitch that breaks launch (e.g. the
            // optional kernel slot got loaded but the launch fails due
            // to grid limits) does not break a plain memcpy. So a launch
            // failure is "slower, not wrong"; the fallback recovers
            // byte-equivalent output. If you ever change that contract
            // (i.e. fold the fallback into a single trusted path),
            // replace the if-success-return below with
            // `try cudaCall(launch_fn(...))`.
            const grid_x: u32 = ndesc;
            const stream = self.work_stream;
            const t_gather = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzGatherRawOff16Kernel", stream);
            const launch_rc = launch_fn(ml.gather_off16_fn, grid_x, 1, 1, 256, 1, 1, 0, stream, &params, &extra);
            if (launch_rc == CUDA_SUCCESS) {
                endKernelTiming(t_gather, stream);
                // No post-launch sync: downstream LZ kernel reads
                // d_entropy_off16_scratch from the same stream.
                return;
            }
            // Best-effort path failed; log so a genuine misconfiguration
            // surfaces rather than silently falling back. The D2D/H2D
            // fallback below still completes correctness; the print just
            // makes the slow-path diagnosable.
            std.debug.print("GPU gatherRawOff16: launch failed rc={d}; falling back to D2D/H2D\n", .{launch_rc});
            endKernelTiming(t_gather, stream);
        }
    }
    // Fallback: async device-to-device, else host upload.
    if (cuda.cuMemcpyDtoDAsync_fn) |d2d| {
        const stream = self.work_stream;
        for (0..scan.num_raw_off16) |ri| {
            const rd = self.raw_off16_buf[ri];
            if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len)
                try cudaCall(d2d(self.d_entropy_off16_scratch + rd.gpu_offset, self.d_comp_persist + rd.src_offset, rd.size, stream), .copy);
        }
        // No post-D2D sync: same stream as downstream LZ kernel.
    } else {
        for (0..scan.num_raw_off16) |ri| {
            const rd = self.raw_off16_buf[ri];
            if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len)
                try cudaCall(h2d_fn(self.d_entropy_off16_scratch + rd.gpu_offset, @ptrCast(compressed_block.ptr + rd.src_offset), rd.size), .copy);
        }
    }
}

/// SLZ_E2E_TIMER trace fields, populated incrementally by fullGpuLaunchImpl.
const E2eCumulative = struct {
    h2d: i64 = 0,
    prescan: i64 = 0,
    postscan: i64 = 0,
    scan: i64 = 0,
    predh: i64 = 0,
};

/// Nanoseconds elapsed between `t0` (a `std.Io.Clock.awake.now(...)`
/// reading) and now, as i64. `anytype` because the Clock reading type
/// isn't directly nameable at the call site.
inline fn nsSince(t0: anytype, iv: std.Io) i64 {
    return @intCast(t0.untilNow(iv, .awake).toNanoseconds());
}

/// SLZ_E2E_TIMER: phase breakdown print at the end of the decode call.
/// `t0` is the start clock reading captured at function entry - `anytype`
/// because the std.Io.Clock reading type is not directly nameable here.
fn emitE2eTrace(
    t0: anytype,
    iv: std.Io,
    cum: E2eCumulative,
    last_kernel: i64,
) void {
    const cum_end_ns: i64 = nsSince(t0, iv);
    const ms = struct {
        fn f(ns: i64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.f;
    const preblk_ns = cum.prescan - cum.h2d;
    const scanfn_ns = cum.postscan - cum.prescan;
    const rawcopy_ns = cum.scan - cum.postscan;
    const prep_ns = (cum.predh - cum.scan) - last_kernel;
    std.debug.print("  [e2e] setup+H2D {d:.3}  preBlk {d:.3}  scanFn {d:.3}  rawD2D {d:.3}  prep {d:.3}  kernels {d:.3}  D2H {d:.3}  total {d:.3} ms\n", .{
        ms(cum.h2d),
        ms(preblk_ns),
        ms(scanfn_ns),
        ms(rawcopy_ns),
        ms(prep_ns),
        ms(last_kernel),
        ms(cum_end_ns - cum.predh),
        ms(cum_end_ns),
    });
}

/// SLZ_DUMP_SCAN: dumps the scan input/output to c:/tmp/scan_dump.bin for
/// the GPU scan-kernel testbed in tools/huff_test/. Inert without the env var.
fn dumpScanIfRequested(
    self: *DecodeContext,
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    scan: ScanResult,
) void {
    // Dev-debug-only path gated on SLZ_DUMP_SCAN. Uses libc fopen so
    // no `std.Io` plumbing is needed (the rest of this fn signature has
    // no io param). The Zig 0.16 `std.Io.Dir.cwd().createFile` API does
    // require an io param; threading one through is a bigger refactor
    // than this debug helper warrants. Path is the value of SLZ_DUMP_SCAN
    // if it looks like a path, else "scan_dump.bin" in cwd (was
    // hard-coded "c:/tmp/scan_dump.bin", Windows-only).
    const env = std.c.getenv("SLZ_DUMP_SCAN") orelse return;
    const cio = @cImport({
        @cInclude("stdio.h");
    });
    // Treat a value of "1" / "true" / "yes" as the default path; anything
    // else is the literal path the user wants.
    const env_slice = std.mem.sliceTo(env, 0);
    const path: [*:0]const u8 = if (env_slice.len <= 1 or
        std.mem.eql(u8, env_slice, "true") or
        std.mem.eql(u8, env_slice, "yes"))
        "scan_dump.bin"
    else
        env;
    const fp = cio.fopen(path, "wb") orelse return;
    defer _ = cio.fclose(fp);
    var hdr = [_]u32{
        0x53434E31, // 'SCN1'
        @intCast(chunk_descs.len),
        sub_chunk_cap,
        @intCast(compressed_block.len),
    };
    _ = cio.fwrite(&hdr, 4, 4, fp);
    _ = cio.fwrite(chunk_descs.ptr, @sizeOf(ChunkDesc), chunk_descs.len, fp);
    if (compressed_block.len > 0)
        _ = cio.fwrite(compressed_block.ptr, 1, compressed_block.len, fp);
    const huff_streams = [_]struct { buf: []const HuffDecChunkDesc, n: u32 }{
        .{ .buf = &self.huff_lit_host_buf, .n = scan.num_huff_lit },
        .{ .buf = &self.huff_tok_host_buf, .n = scan.num_huff_tok },
        .{ .buf = &self.huff_off16hi_host_buf, .n = scan.num_huff_off16hi },
        .{ .buf = &self.huff_off16lo_host_buf, .n = scan.num_huff_off16lo },
    };
    for (huff_streams) |hs| {
        var n = [_]u32{hs.n};
        _ = cio.fwrite(&n, 4, 1, fp);
        _ = cio.fwrite(hs.buf.ptr, @sizeOf(HuffDecChunkDesc), hs.n, fp);
    }
    var nraw = [_]u32{scan.num_raw_off16};
    _ = cio.fwrite(&nraw, 4, 1, fp);
    for (0..scan.num_raw_off16) |ri| {
        const rd = self.raw_off16_buf[ri];
        var t = [_]u32{ rd.src_offset, rd.size, rd.gpu_offset };
        _ = cio.fwrite(&t, 4, 3, fp);
    }
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
    /// Phase 3: when non-null, the chunk descs already live on the
    /// device at this address (size: chunk_descs.len * sizeof(ChunkDesc))
    /// and the H2D from the host slice is skipped. `chunk_descs.len`
    /// is still authoritative for the count. Used by the D2D path
    /// (decompressFramedFromDevice) to reuse the walk-kernel's
    /// `d_walk_chunks` directly instead of D2H'ing + re-H2D'ing the
    /// same bytes.
    d_chunk_descs_override: ?u64 = null,
    /// Phase 3-final: when non-null, the LZ kernel reads its self-gate
    /// count (`*p_total`) from this device address instead of from
    /// `self.d_n_groups_scratch`. Used by `decompressFramedFromDevice` to
    /// point at `d_walk_meta + walk_meta_offsets.n_chunks` so the walk
    /// kernel's output is consumed directly by the LZ kernel via stream
    /// ordering — no host D2H, no host H2D restage. Null preserves the
    /// CLI / host-bounce behavior (stage `total_chunks` into
    /// `d_n_groups_scratch` host-side).
    d_n_chunks_dev: ?u64 = null,
};

/// Launches the Huffman LUT-build kernel and the 4-stream decode
/// kernel into `heavy_stream`. Both kernels self-gate on `*d_n_huff`,
/// so it is safe to launch them with the chunk count even if some
/// chunks have no Huffman streams.
///
/// Returns the LUT-build elapsed nanoseconds when `split_timer` is
/// set (driven by SLZ_SPLIT_TIMER); otherwise zero. The caller is
/// responsible for computing the full huff-total time after the
/// pipeline-stream sync.
fn runHuffPredecode(
    self: *DecodeContext,
    fns: *const Fns,
    scan: ScanResult,
    n_huff: u32,
    heavy_stream: usize,
    split_timer: bool,
    t_huff_start: anytype,
    io: ?std.Io,
) GpuError!i64 {
    const h2d_fn = fns.h2d;
    const launch_fn = fns.launch;

    const huff_stream = heavy_stream;
    var split_huff_build_ns: i64 = 0;

    // Step 7: huff kernels self-gate on `*d_n_blocks`. The GPU
    // merge kernel writes n_merged to d_compact_counts+20; reuse
    // that slot when GPU merge ran. CPU-merge fallback stages
    // n_huff into d_n_huff_scratch via a 4 B H2D.
    const d_n_huff: u64 = if (scan.device_compact_populated and ml.merge_huff_descs_fn != 0)
        self.d_compact_counts + 20
    else d_nh: {
        if (!ensureDeviceBuf(&self.d_n_huff_scratch, &self.d_n_huff_scratch_size, 4)) return error.OutOfDeviceMemory;
        var host_n_huff: u32 = n_huff;
        try cudaCall(h2d_fn(self.d_n_huff_scratch, @ptrCast(&host_n_huff), 4), .copy);
        break :d_nh self.d_n_huff_scratch;
    };
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
        try cudaCall(launch_fn(ml.huff_build_fn, n_huff, 1, 1, 32, 1, 1, 0, huff_stream, &params, &extra), .launch);
        endKernelTiming(t_hb, huff_stream);
    }
    // Split fence: time the LUT build separately from the decode.
    if (split_timer) {
        try cudaCall(fns.stream_sync(huff_stream), .sync);
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
        var params = [_]?*anyopaque{
            @ptrCast(&p_comp), @ptrCast(&p_descs),
            @ptrCast(&p_lut),  @ptrCast(&p_out),
            @ptrCast(&p_n),
        };
        var extra = [_]?*anyopaque{null};
        const shared_bytes: c_uint = HUFF_LUT_ENTRIES * @sizeOf(u32);
        const t_hd = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffDecode4StreamKernel", huff_stream);
        try cudaCall(launch_fn(ml.huff_decode_fn, n_huff, 1, 1, 32, 1, 1, shared_bytes, huff_stream, &params, &extra), .launch);
        endKernelTiming(t_hd, huff_stream);
    }
    return split_huff_build_ns;
}

/// Launches the per-pipeline-stream LZ decode kernels. Uses the lean
/// raw kernel when no Huffman streams are present (L1/L2 path);
/// otherwise the general kernel that consumes `d_entropy_scratch`.
/// Operations within a stream are ordered; across streams they can
/// overlap.
///
/// Returns the per-launch summed LZ elapsed nanoseconds when
/// `split_timer` is set; otherwise zero.
fn runLzPipeline(
    self: *DecodeContext,
    fns: *const Fns,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    n_huff: u32,
    total_chunks: u32,
    total_subchunks: u32,
    pipe_chunk_count: u32,
    heavy_stream: usize,
    split_timer: bool,
    io: ?std.Io,
    /// Device address holding the self-gate count (`*p_total` for the
    /// LZ kernel). When the D2D path supplies `req.d_n_chunks_dev`,
    /// this is `d_walk_meta + offset(n_chunks)` (no H2D round-trip);
    /// otherwise `self.d_n_groups_scratch` after the host-staging H2D
    /// in `fullGpuLaunchImpl`.
    total_dev: CUdeviceptr,
    /// Base device address the LZ kernel writes decompressed bytes
    /// into. Normally `self.d_output` (library scratch). On the D2D
    /// fast path the caller routes the caller-supplied
    /// `req.d_output_target` here so the kernel writes straight to
    /// the caller's buffer — skips the ~1.1 ms / 100 MB finalize D2D
    /// copy that otherwise stages through d_output. Requires
    /// `req.dst_start_off == 0` (no host-side prefix to splice in).
    dst_base: CUdeviceptr,
) GpuError!i64 {
    const launch_fn = fns.launch;
    const stream_sync_fn = fns.stream_sync;

    var split_lz_ns: i64 = 0;

    for (0..NUM_PIPELINE_STREAMS) |g| {
        const stream: usize = if (comptime NUM_PIPELINE_STREAMS == 1) heavy_stream else self.pipeline_streams[g];
        const chunk_start = @as(u32, @intCast(g)) * pipe_chunk_count;
        const chunk_end = @min(chunk_start + pipe_chunk_count, total_chunks);
        const group_chunks = chunk_end - chunk_start;
        if (group_chunks == 0) continue;

        const t_lz_start = if (split_timer)
            if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
        else
            null;

        // chunks pointer = full array (NOT offset), chunk_base = chunk_start
        // total = chunk_end so the kernel's bounds check (chunk_idx >= total) works
        const lz_groups_in_pipe = (group_chunks + chunks_per_group - 1) / chunks_per_group;
        const lz_grid_x = (lz_groups_in_pipe + 1) / 2;

        // The LZ kernel's self-gate count comes via the `total_dev`
        // parameter; fullGpuLaunchImpl is responsible for arranging the
        // backing storage (either `d_n_groups_scratch` staged by host
        // H2D, or `d_walk_meta + offset(n_chunks)` on the D2D path).
        // At NUM_PIPELINE_STREAMS == 1 this hoist is correct because
        // chunk_end - chunk_start == total_chunks when the loop runs once;
        // a future bump to N > 1 needs N device counters.

        // Fast path: no entropy in this scan → use lean L1/L2 raw kernel.
        // Huffman literals require the general kernel (it reads entropy_scratch).
        const use_raw_kernel = n_huff == 0 and ml.kernel_raw_fn != 0;

        // The six shared params (comp, descs_dev, dst, cpg, total, sc_cap)
        // live in a small struct so each field has a stable address for
        // the kernel-params array.
        var common = LzCommonParams{
            .comp = self.d_comp_persist,
            .descs_dev = self.d_descs_persist + @as(u64, chunk_start) * @sizeOf(ChunkDesc),
            .dst = dst_base,
            .cpg = chunks_per_group,
            .total = total_dev,
            .sc_cap = sub_chunk_cap,
        };

        if (use_raw_kernel) {
            var raw_params = [_]?*anyopaque{
                @ptrCast(&common.comp),
                @ptrCast(&common.descs_dev),
                @ptrCast(&common.dst),
                @ptrCast(&common.cpg),
                @ptrCast(&common.total),
                @ptrCast(&common.sc_cap),
            };
            var raw_extra = [_]?*anyopaque{null};
            const t_lzr = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeRawKernel", stream);
            try cudaCall(launch_fn(ml.kernel_raw_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &raw_params, &raw_extra), .launch);
            endKernelTiming(t_lzr, stream);
        } else {
            // General kernel adds entropy_scratch + slot_stride + first_sub_idx.
            var p_entropy_scratch: u64 = self.d_entropy_scratch;
            var p_entropy_slot_stride: u64 = @as(u64, total_subchunks) * d.ENTROPY_SCRATCH_SLOT_BYTES;
            var p_first_sub_idx: CUdeviceptr = self.d_first_subchunk_idx +
                if (self.d_first_subchunk_idx != 0) @as(u64, chunk_start) * @sizeOf(u32) else 0;

            var lz_params = [_]?*anyopaque{
                @ptrCast(&common.comp),
                @ptrCast(&common.descs_dev),
                @ptrCast(&common.dst),
                @ptrCast(&common.cpg),
                @ptrCast(&common.total),
                @ptrCast(&common.sc_cap),
                @ptrCast(&p_entropy_scratch),
                @ptrCast(&p_entropy_slot_stride),
                @ptrCast(&p_first_sub_idx),
            };
            var lz_extra = [_]?*anyopaque{null};

            const t_lz = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeKernel", stream);
            try cudaCall(launch_fn(ml.kernel_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &lz_params, &lz_extra), .launch);
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
fn finalizeOutput(self: *DecodeContext, fns: *const Fns, req: DecodeRequest, heavy_stream: usize) GpuError!void {
    if (req.d_output_target) |dev_target| {
        const d2d = fns.d2d orelse return error.BackendNotAvailable;
        try cudaCall(d2d(dev_target + req.dst_start_off, self.d_output + req.dst_start_off, req.decompressed_size, heavy_stream), .copy);
        if (self.work_stream == 0) try cudaCall(fns.sync(), .sync);
    } else {
        try cudaCall(fns.d2h(@ptrCast(req.dst_full + req.dst_start_off), self.d_output + req.dst_start_off, req.decompressed_size), .copy);
    }
}

pub fn fullGpuLaunchImpl(self: *DecodeContext, req: DecodeRequest) GpuError!void {
    // Shadow request fields as locals so the function body below - which
    // predates the struct-arg refactor and references the bare names -
    // doesn't need a full rename. Zig folds these into the same registers
    // it would have used for the old direct params.
    const chunk_descs = req.chunk_descs;
    const compressed_block = req.compressed_block;
    const dst_full = req.dst_full;
    const dst_start_off = req.dst_start_off;
    const decompressed_size = req.decompressed_size;
    const chunks_per_group = req.chunks_per_group;
    const sub_chunk_cap = req.sub_chunk_cap;
    const io = req.io;
    const d_compressed_src = req.d_compressed_src;

    if (!ml.init()) return error.BackendNotAvailable;
    if (ml.kernel_fn == 0) return error.KernelMissing;
    try ml.ensurePipelineStreams(self);

    const facade = @import("driver.zig");

    // SLZ_E2E_TIMER: end-to-end decode phase breakdown - setup+H2D /
    // host scan+prep / kernels / D2H. Off by default.
    const e2e_timer = std.c.getenv("SLZ_E2E_TIMER") != null;
    // SLZ_HUFF_DBG: cache once per call (otherwise getenv would run on
    // every decode in this loop's caller). NOTE: scope is per-CALL, not
    // per-process - the env var is still re-read on every fullGpuLaunchImpl
    // invocation. For a true once-per-process cache, lift to a module-
    // level std.once.Once. The per-call cache is enough for the dev-only
    // SLZ_HUFF_DBG path; flipping it across calls in one process is rare.
    const huff_dbg_this_call = std.c.getenv("SLZ_HUFF_DBG") != null;
    const t_e2e0 = if (e2e_timer)
        (if (io) |iv| std.Io.Clock.awake.now(iv) else null)
    else
        null;
    var e2e_cum: E2eCumulative = .{};

    const fns = try Fns.resolve();
    const h2d_fn = fns.h2d;
    const launch_fn = fns.launch;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(self, total_output + 64)) return error.OutOfDeviceMemory;

    if (dst_start_off > 0)
        try cudaCall(h2d_fn(self.d_output, @ptrCast(dst_full), dst_start_off), .copy);

    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    if (!ensureDeviceBuf(&self.d_comp_persist, &self.d_comp_persist_size, comp_bytes + 16)) return error.OutOfDeviceMemory;
    if (!ensureDeviceBuf(&self.d_descs_persist, &self.d_descs_persist_size, desc_bytes)) return error.OutOfDeviceMemory;

    // Chunk descs land in d_descs_persist either by D2D-copy from a
    // caller-supplied device buffer (Phase 3 — saves a D2H+H2D round
    // trip on the pure-D2D path where the walk-frame kernel already
    // wrote them to d_walk_chunks) or by H2D from the host slice
    // (host-bounce path, no device chunks available). h2d_fn is
    // sync-on-host (cuMemcpyHtoD), and cuMemcpyDtoDAsync on the
    // caller's work_stream serializes with the prefix-sum kernel below
    // via stream ordering.
    if (req.d_chunk_descs_override) |dev_ptr| {
        const d2d = fns.d2d orelse return error.BackendNotAvailable;
        try cudaCall(d2d(self.d_descs_persist, dev_ptr, desc_bytes, self.work_stream), .copy);
    } else {
        try cudaCall(h2d_fn(self.d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes), .copy);
    }

    // Prefix-sum runs on device. `gpuPrefixSumChunksImpl` returns
    // `GpuError!T` (not `?T`) because there is no host fallback -
    // every GPU decode path needs it; specific failure variants
    // (OutOfDeviceMemory / KernelLaunchFailed / SyncFailed /
    // BackendNotAvailable / KernelMissing) propagate to the caller.
    _ = try scan_gpu_mod.gpuPrefixSumChunksImpl(self, self.d_descs_persist, @intCast(chunk_descs.len), sub_chunk_cap);
    // Skip the D2H of `total_subchunks` and use a host-computed
    // worst-case bound: max sub-chunks per chunk = ceil(chunk_size /
    // sub_chunk_cap), clamped to MAX_BLOCKS_PER_SUBCHUNK*2 = 4 in the
    // tightest 64 KB sub-chunk case. Over-allocating `entropy_scratch`
    // is safe — the kernels self-gate on the actual counts via
    // `d_total_subchunks_buf` / `d_compact_counts`. Saves a 4-byte D2H
    // plus the implicit kernel sync it would force.
    // Wire format: SLZ_CHUNK_SIZE_BYTES = 256 KB (matches src/gpu/common/gpu_wire_format.cuh)
    const CHUNK_SIZE_BYTES: u32 = 0x40000;
    const max_sub_per_chunk: u32 = if (sub_chunk_cap == 0) 4 else @max(1, (CHUNK_SIZE_BYTES + sub_chunk_cap - 1) / sub_chunk_cap);
    // Worst-case upper bound on total sub-chunks across the frame; used
    // for entropy_scratch sizing + region offsets that must match between
    // the merge-kernel writer and the LZ-kernel reader. Naming kept as
    // `total_subchunks` so downstream call sites are unchanged.
    const total_subchunks: u32 = @as(u32, @intCast(chunk_descs.len)) * max_sub_per_chunk;
    // d.ENTROPY_SCRATCH_SLOT_BYTES holds the largest sub-chunk's lit/tok
    // streams; off16-hi at +0, off16-lo at +d.OFF16_HILO_SPLIT_OFFSET
    // within each slot. Layout: [lit: total*slot] [tok: total*slot] [off16: total*slot].
    const per_subchunk_scratch: usize = @intCast(d.ENTROPY_SCRATCH_SLOT_BYTES);
    const entropy_scratch_bytes = @as(usize, total_subchunks) * per_subchunk_scratch * 3;
    if (!ensureDeviceBuf(&self.d_entropy_scratch, &self.d_entropy_scratch_size, entropy_scratch_bytes)) return error.OutOfDeviceMemory;
    const tok_offset = @as(usize, total_subchunks) * per_subchunk_scratch;
    const off16_offset = @as(usize, total_subchunks) * per_subchunk_scratch * 2;
    self.d_entropy_off16_scratch = self.d_entropy_scratch + off16_offset;

    // Always pass the device prefix-sum to the LZ kernel. When
    // `sub_chunk_cap >= chunk_size` the prefix sum is `[0, 1, 2, ...]`
    // and the kernel reads identity values — same effective behavior as
    // the prior null-pointer / kernel-side identity branch.
    self.d_first_subchunk_idx = self.d_first_sub_idx_persist;
    if (chunk_descs.len > d.WALK_MAX_CHUNKS) return error.BadMode;

    // Source-side input: D2D when the bytes are already device-resident,
    // H2D otherwise. The D2D copy issues on `self.work_stream` so the
    // post-copy stream sync below only has to wait on the caller's
    // stream; the H2D path blocks on the host side, so there is no
    // async hand-off to wait on.
    if (compressed_block.len > 0) {
        if (d_compressed_src) |dev_src| {
            const d2d = fns.d2d orelse return error.BackendNotAvailable;
            // No post-D2D sync: every downstream consumer of d_comp_persist
            // (gpuPrefixSumChunksImpl, gpuScanChunks, runHuffPredecode,
            // runLzPipeline) runs on the same work_stream and sees the copy
            // via stream ordering. The explicit sync added in commit ebce084
            // was a stream-0-era leftover; removing it lets the front-half
            // kernels queue while the D2D is still in flight on the same
            // stream's command buffer.
            try cudaCall(d2d(self.d_comp_persist, dev_src, compressed_block.len, self.work_stream), .copy);
        } else {
            try cudaCall(h2d_fn(self.d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len), .copy);
        }
    }
    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum.h2d = nsSince(t0, iv);
        e2e_cum.prescan = nsSince(t0, iv);
    };

    // ── Scan for entropy chunks (Huffman + raw off16) ─────────
    // The GPU scan kernel reads from `d_comp_persist` (filled by either
    // the H2D from the host `compressed_block` slice or the D2D copy
    // from the caller's device pointer); the host bytes themselves are
    // never touched here, so the same code path handles both decode
    // entry points.
    var scan: ScanResult = .{ .num_raw_off16 = 0 };

    if (ml.huff_build_fn != 0) {
        if (chunk_descs.len > d.WALK_MAX_CHUNKS) return error.BadMode;
        scan = scan_gpu_mod.gpuScanChunks(
            self,
            chunk_descs,
            compressed_block,
            sub_chunk_cap,
            total_subchunks,
            &self.huff_lit_host_buf,
            &self.huff_tok_host_buf,
            &self.huff_off16hi_host_buf,
            &self.huff_off16lo_host_buf,
            &self.raw_off16_buf,
        ) orelse return error.BackendNotAvailable;

        dumpScanIfRequested(self, chunk_descs, compressed_block, sub_chunk_cap, scan);

        if (t_e2e0) |t0| if (io) |iv| {
            e2e_cum.postscan = nsSince(t0, iv);
        };

        try gatherRawOff16(self, scan, compressed_block, h2d_fn, launch_fn);
    }

    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum.scan = nsSince(t0, iv);
    };

    // ── Huffman pre-decode (Pass 1.5): merge per-stream-type descriptors
    // into a single device array with correct out_offsets, then upload.
    const n_huff: u32 = scan.num_huff_lit + scan.num_huff_tok +
        scan.num_huff_off16hi + scan.num_huff_off16lo;
    if (huff_dbg_this_call) {
        std.debug.print("huff scan: lit={d} tok={d} hi={d} lo={d} total={d}\n", .{ scan.num_huff_lit, scan.num_huff_tok, scan.num_huff_off16hi, scan.num_huff_off16lo, n_huff });
    }
    const have_huff = n_huff > 0 and ml.huff_build_fn != 0 and ml.huff_decode_fn != 0;
    if (have_huff) {
        const huff_desc_bytes = @as(usize, n_huff) * @sizeOf(HuffDecChunkDesc);
        const huff_lut_bytes = @as(usize, n_huff) * HUFF_LUT_ENTRIES * @sizeOf(u32);
        if (!ensureDeviceBuf(&self.d_huff_descs, &self.d_huff_descs_size, huff_desc_bytes)) return error.OutOfDeviceMemory;
        if (!ensureDeviceBuf(&self.d_huff_lut, &self.d_huff_lut_size, huff_lut_bytes)) return error.OutOfDeviceMemory;

        try mergeHuffDescs(self, scan, tok_offset, off16_offset, h2d_fn, launch_fn);
    }

    // ── Pipeline: split into N groups, overlap entropy with LZ ───
    // The pipeline path overlaps Huffman pre-decode with the LZ kernel.
    const total_chunks: u32 = @intCast(chunk_descs.len);
    // Defensive: `try ml.ensurePipelineStreams(self)` at fullGpuLaunchImpl
    // entry already guarantees `pipeline_streams_created == true` (it
    // throws BackendNotAvailable on stream-create failure). This duplicate
    // check is kept as a belt-and-suspenders against future refactors
    // that might reorder init steps; the early return is unreachable on
    // any path that gets here through the normal entry point.
    if (!self.pipeline_streams_created) return error.BackendNotAvailable;
    // When the caller is slzDecompressAsync, work_stream is the caller's
    // CUstream so its cudaStreamSynchronize waits for the decompress to
    // land in d_output. The sync wrapper leaves work_stream at 0 and
    // falls back to the library's own pipeline_streams[0].
    const heavy_stream: usize = if (self.work_stream != 0) self.work_stream else self.pipeline_streams[0];
    {
        const pipe_chunk_count = (total_chunks + NUM_PIPELINE_STREAMS - 1) / NUM_PIPELINE_STREAMS;
        // Front-half work (walk + prefix-sum + scan + compact + merge +
        // gather + descs D2D) is queued on self.work_stream. Back-half
        // (huff + LZ + finalize) is queued on heavy_stream.
        //
        // In async mode (self.work_stream != 0) the caller's stream is
        // both work_stream and heavy_stream — same stream, so stream
        // ordering alone gives the back-half its view of front-half
        // results. The sync is pure host-wait overhead. nsys measured
        // 12 µs median, 1.6 ms worst case on the bench.
        //
        // In sync mode (work_stream == 0) heavy_stream is
        // pipeline_streams[0], a different stream — the back-half won't
        // see the front-half's writes without an explicit barrier.
        if (heavy_stream != self.work_stream) {
            try cudaCall(fns.stream_sync(self.work_stream), .sync);
        }

        // ── KERNEL TIMER: only pure GPU kernel time from here ──
        const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

        // SLZ_SPLIT_TIMER hoisted above the Huff launches so we can fence
        // around them too. Without split, Huff time gets pipelined into LZ.
        const split_timer = std.c.getenv("SLZ_SPLIT_TIMER") != null;

        // Stage the LZ kernel's self-gate count. Three modes:
        //   1. req.d_n_chunks_dev non-null (Phase 3-final D2D path): point
        //      the LZ kernel directly at the caller-supplied device counter
        //      (typically `d_walk_meta + walk_meta_offsets.n_chunks`, which
        //      the walk kernel populated on this stream). No H2D, no D2H.
        //   2. Otherwise (CLI / host-bounce): allocate / restage
        //      d_n_groups_scratch with `total_chunks`. Hoisted out of
        //      runLzPipeline so a graph-mode capture of the back half
        //      doesn't see a sync H2D inside the captured region.
        // See runLzPipeline comment for the NUM_PIPELINE_STREAMS == 1
        // caveat.
        const lz_total_dev: u64 = if (req.d_n_chunks_dev) |dev| dev else blk: {
            if (!ensureDeviceBuf(&self.d_n_groups_scratch, &self.d_n_groups_scratch_size, 4)) return error.OutOfDeviceMemory;
            var host_total_chunks: u32 = total_chunks;
            try cudaCall(h2d_fn(self.d_n_groups_scratch, @ptrCast(&host_total_chunks), 4), .copy);
            break :blk self.d_n_groups_scratch;
        };

        // ── Direct-write-to-output fast path ───────────────────────────
        // When the caller supplied a device output buffer AND there's no
        // host-side prefix to splice in (dst_start_off == 0), point the
        // LZ kernel directly at that buffer. The finalize-D2D copy
        // (~1.1 ms on a 100 MB output measured by nsys) is then skipped
        // — the LZ kernel was already going to write `decompressed_size`
        // bytes; writing them to the caller's address instead of our
        // scratch saves that whole copy.
        //
        // The legacy path (dst_start_off > 0, or host D2H output) still
        // stages through self.d_output: the host needs to H2D the prefix
        // bytes into our buffer first, and the D2H output path needs a
        // device source to D2H from anyway.
        const write_direct: bool = req.d_output_target != null and req.dst_start_off == 0;
        const lz_dst_base: CUdeviceptr = if (write_direct) req.d_output_target.? else self.d_output;

        const t_huff_start = if (split_timer and have_huff)
            if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
        else
            null;
        var split_huff_build_ns: i64 = 0;
        if (have_huff) {
            split_huff_build_ns = try runHuffPredecode(self, &fns, scan, n_huff, heavy_stream, split_timer, t_huff_start, io);
        }

        // Launch pipelined groups: one LZ launch per pipeline stream.
        // Operations within a stream are ordered; across streams they can overlap.
        const stream_sync_fn = fns.stream_sync;

        var split_lz_ns: i64 = 0;
        var split_huff_ns: i64 = 0;

        // Close Huff time slice: sync the pipeline stream so the LZ
        // measurement excludes Huff time. (split_timer rules out graph
        // mode above; the inner sync is safe.)
        if (split_timer and have_huff) {
            try cudaCall(stream_sync_fn(self.pipeline_streams[0]), .sync);
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

        split_lz_ns = try runLzPipeline(
            self,
            &fns,
            chunks_per_group,
            sub_chunk_cap,
            n_huff,
            total_chunks,
            total_subchunks,
            pipe_chunk_count,
            heavy_stream,
            split_timer,
            io,
            lz_total_dev,
            lz_dst_base,
        );

        // Sync all pipeline streams - UNLESS the caller is async
        // (work_stream set). In async mode we leave the queued work on
        // the user's stream and let them sync; blocking here would
        // defeat the purpose.
        //
        // Failure handling: first stream's sync failure aborts and
        // returns; remaining streams' work is left running on the
        // device. Safe at NUM_PIPELINE_STREAMS == 1 (only one stream
        // exists). If the constant is ever bumped to > 1, revisit:
        // either drain remaining streams before returning (loses the
        // specific failing stream's rc in the print) or accept the
        // existing first-fail-wins semantics with a clearer comment.
        if (self.work_stream == 0) {
            for (0..NUM_PIPELINE_STREAMS) |g| {
                const sync_rc = stream_sync_fn(self.pipeline_streams[g]);
                if (sync_rc != CUDA_SUCCESS) {
                    std.debug.print("GPU pipe[{d}]: stream sync FAILED rc={d}\n", .{ g, sync_rc });
                    return error.SyncFailed;
                }
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

    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum.predh = nsSince(t0, iv);
    };
    // Skip finalizeOutput when the LZ kernel wrote straight to the
    // caller's device buffer (req.d_output_target with dst_start_off
    // == 0). Same condition as `write_direct` inside the back-half
    // block; inlined here because `write_direct` was scoped to that
    // block. Saves the ~1.1 ms / 100 MB D2D copy the legacy path did.
    const skip_finalize = req.d_output_target != null and req.dst_start_off == 0;
    if (!skip_finalize) {
        try finalizeOutput(self, &fns, req, heavy_stream);
    }

    // Profiling: drain pending cuEvent pairs into last_timings. Skip in
    // async mode (caller hasn't synced yet; the events may still be
    // pending). slzGetLastTimings finalizes on demand.
    if (self.work_stream == 0) {
        finalizeProfiling(&self.pending_timings, &self.last_timings);
    }
    if (t_e2e0) |t0| if (io) |iv| {
        emitE2eTrace(t0, iv, e2e_cum, facade.last_kernel_ns);
    };
}
