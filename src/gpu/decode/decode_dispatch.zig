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
const scan_host_mod = @import("scan_host.zig");
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

/// Pre-decode preparation: combine the four per-stream-type Huffman
/// descriptor arrays (lit / tok / off16-hi / off16-lo) into one merged
/// device array in `self.d_huff_descs`, with out_offsets adjusted by
/// the per-stream-type region offset and lut_offsets assigned sequentially.
/// Two paths:
///   GPU merge (pure-D2D, when the device compact populated the d_compact_*
///   buffers): launches slzMergeHuffDescsKernel — no host arrays touched.
///   CPU merge (CPU-scan fallback): walks the host-side huff_*_host_buf
///   arrays and uploads the merged result via one H2D.
fn mergeHuffDescs(
    self: *DecodeContext,
    scan: ScanResult,
    tok_offset: usize,
    off16_offset: usize,
    h2d_fn: anytype,
    sync_fn: anytype,
    launch_fn: anytype,
) GpuError!void {
    if (scan.device_compact_populated and ml.merge_huff_descs_fn != 0) {
        // Step 6c: the compact kernels already populated the four per-stream
        // device buffers in step 6b. Launch the merge kernel — it writes
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
        const t_merge = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzMergeHuffDescsKernel", 0);
        try cudaCall(launch_fn(ml.merge_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, 0, &m_params, &m_extra));
        endKernelTiming(t_merge, 0);
        try cudaCall(sync_fn());
        return;
    }
    // CPU merge fallback: used by the non-pure-D2D / CPU-scan paths.
    // Storage lives on the DecodeContext (see merged_huff_buf there) so
    // the dispatch frame stays small.
    const merged_huff: []HuffDecChunkDesc = self.merged_huff_buf[0..];
    var m: u32 = 0;
    var lut_slot: u32 = 0;
    const append = struct {
        fn run(dst: []HuffDecChunkDesc, m_ptr: *u32, lut_ptr: *u32,
               src: []const HuffDecChunkDesc, region_off: u32) void {
            for (src) |s| {
                if (m_ptr.* >= dst.len) return;
                var dd = s;
                dd.out_offset += region_off;
                dd.lut_offset = lut_ptr.* * @as(u32, @intCast(HUFF_LUT_ENTRIES));
                dst[m_ptr.*] = dd;
                m_ptr.* += 1;
                lut_ptr.* += 1;
            }
        }
    }.run;
    append(merged_huff, &m, &lut_slot, self.huff_lit_host_buf[0..scan.num_huff_lit], 0);
    append(merged_huff, &m, &lut_slot, self.huff_tok_host_buf[0..scan.num_huff_tok], @intCast(tok_offset));
    append(merged_huff, &m, &lut_slot, self.huff_off16hi_host_buf[0..scan.num_huff_off16hi], @intCast(off16_offset));
    append(merged_huff, &m, &lut_slot, self.huff_off16lo_host_buf[0..scan.num_huff_off16lo], @intCast(off16_offset));

    try cudaCall(h2d_fn(self.d_huff_descs, @ptrCast(merged_huff.ptr), @as(usize, m) * @sizeOf(HuffDecChunkDesc)));
    try cudaCall(sync_fn());
}

/// Place raw (type 0) off16 sub-streams into the off16 scratch. The bytes
/// are already on the GPU (d_comp_persist holds the whole compressed blob).
/// Preferred: upload the descriptor list in one H2D and run
/// slzGatherRawOff16Kernel — one launch copies every stream in parallel.
/// Fallbacks: async device-to-device loop, then a plain host-upload loop.
fn gatherRawOff16(
    self: *DecodeContext,
    scan: ScanResult,
    compressed_block: []const u8,
    h2d_fn: anytype,
    sync_fn: anytype,
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
            try cudaCall(h2d_fn(self.d_raw_off16_descs, @ptrCast(&self.raw_off16_buf), dbytes));
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
            try cudaCall(h2d_fn(self.d_n_raw_scratch, @ptrCast(&host_n), 4));
            break :d_cnt self.d_n_raw_scratch;
        };
        {
            var p_comp = self.d_comp_persist;
            var p_comp_len: u32 = @intCast(compressed_block.len);
            var p_scratch = self.d_entropy_off16_scratch;
            var p_descs = desc_dev;
            var p_count = d_count;
            var params = [_]?*anyopaque{
                @ptrCast(&p_comp), @ptrCast(&p_comp_len), @ptrCast(&p_scratch),
                @ptrCast(&p_descs), @ptrCast(&p_count),
            };
            var extra = [_]?*anyopaque{null};
            // ndesc is exact (host already knows the count); no over-launch.
            // The self-gate inside the kernel makes over-launch safe regardless.
            //
            // K5.9 decision: KEEP the D2D/H2D fallback below and treat the
            // launch as a best-effort fast path. Justification: the two
            // failure modes are disjoint — the kernel-launch path needs
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
            const t_gather = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzGatherRawOff16Kernel", 0);
            if (launch_fn(ml.gather_off16_fn, grid_x, 1, 1, 256, 1, 1, 0, 0, &params, &extra) == CUDA_SUCCESS) {
                endKernelTiming(t_gather, 0);
                try cudaCall(sync_fn());
                return;
            }
            endKernelTiming(t_gather, 0);
        }
    }
    // Fallback: async device-to-device, else host upload.
    if (cuda.cuMemcpyDtoDAsync_fn) |d2d| {
        for (0..scan.num_raw_off16) |ri| {
            const rd = self.raw_off16_buf[ri];
            if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len)
                try cudaCall(d2d(self.d_entropy_off16_scratch + rd.gpu_offset, self.d_comp_persist + rd.src_offset, rd.size, 0));
        }
        try cudaCall(sync_fn());
    } else {
        for (0..scan.num_raw_off16) |ri| {
            const rd = self.raw_off16_buf[ri];
            if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len)
                try cudaCall(h2d_fn(self.d_entropy_off16_scratch + rd.gpu_offset, @ptrCast(compressed_block.ptr + rd.src_offset), rd.size));
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

/// SLZ_E2E_TIMER: phase breakdown print at the end of the decode call.
/// `t0` is the start clock reading captured at function entry — `anytype`
/// because the std.Io.Clock reading type is not directly nameable here.
fn emitE2eTrace(
    t0: anytype,
    iv: std.Io,
    cum: E2eCumulative,
    last_kernel: i64,
) void {
    const cum_end_ns: i64 = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
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
    if (std.c.getenv("SLZ_DUMP_SCAN") == null) return;
    const cio = @cImport({
        @cInclude("stdio.h");
    });
    const fp = cio.fopen("c:/tmp/scan_dump.bin", "wb") orelse return;
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

pub fn fullGpuLaunch(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    io: ?std.Io,
) GpuError!void {
    return fullGpuLaunchImpl(
        &@import("driver.zig").g_default,
        chunk_descs,
        compressed_block,
        dst_full,
        dst_start_off,
        decompressed_size,
        chunks_per_group,
        sub_chunk_cap,
        io,
        null,
        null,
    );
}

pub fn fullGpuLaunchImpl(
    self: *DecodeContext,
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    io: ?std.Io,
    d_output_target: ?u64,
    /// 4d Phase 3 D2D: when non-null the compressed block is already
    /// device-resident at this address; the source is D2D-copied into
    /// d_comp_persist (no PCIe). `compressed_block.ptr` is then unused
    /// (caller may pass an undefined slice with the correct `.len`).
    d_compressed_src: ?u64,
) GpuError!void {
    if (!ml.init() or ml.kernel_fn == 0) return error.BadMode;
    try ml.ensurePipelineStreams(self);

    const facade = @import("driver.zig");

    // SLZ_E2E_TIMER: end-to-end decode phase breakdown — setup+H2D /
    // host scan+prep / kernels / D2H. Off by default.
    const e2e_timer = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_e2e0 = if (e2e_timer)
        (if (io) |iv| std.Io.Clock.awake.now(iv) else null)
    else
        null;
    var e2e_cum: E2eCumulative = .{};

    const h2d_fn = cuda.cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuda.cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuda.cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuda.cuCtxSynchronize_fn orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(self, total_output + 64)) return error.BadMode;

    if (dst_start_off > 0)
        try cudaCall(h2d_fn(self.d_output, @ptrCast(dst_full), dst_start_off));

    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    if (!ensureDeviceBuf(&self.d_comp_persist, &self.d_comp_persist_size, comp_bytes + 16)) return error.BadMode;
    if (!ensureDeviceBuf(&self.d_descs_persist, &self.d_descs_persist_size, desc_bytes)) return error.BadMode;

    // Chunk descs must be on device before the prefix-sum kernel reads
    // them. The lower H2D path is still present and unconditionally
    // re-uploads; this hoist is what feeds the prefix-sum kernel.
    try cudaCall(h2d_fn(self.d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes));
    try cudaCall(sync_fn());

    // 4d Phase 3 step 6: prefix-sum runs on device. The 4-byte D2H of
    // total_subchunks below is launch-plumbing — needed to size the
    // entropy scratch + compute region offsets host-side; no per-chunk
    // CPU work remains.
    _ = scan_gpu_mod.gpuPrefixSumChunksImpl(self, self.d_descs_persist, @intCast(chunk_descs.len), sub_chunk_cap) orelse return error.BadMode;
    var total_subchunks: u32 = 0;
    try cudaCall(d2h_fn(@ptrCast(&total_subchunks), self.d_total_subchunks_buf, 4));
    self.d_first_subchunk_idx = self.d_first_sub_idx_persist;
    // CPU mirror for the pipeline branch — NUM_PIPELINE_STREAMS==1
    // reads only index 0 (= 0). Storage lives on the DecodeContext;
    // zero-init at first use is enough (subsequent calls overwrite as
    // needed). Bumping the stream count would need a selective D2H of
    // group boundaries from d_first_sub_idx_persist.
    const first_subchunk_idx_buf: []u32 = &self.first_subchunk_idx_buf;
    // d.ENTROPY_SCRATCH_SLOT_BYTES holds the largest sub-chunk's lit/tok
    // streams; off16-hi at +0, off16-lo at +d.OFF16_HILO_SPLIT_OFFSET
    // within each slot. Layout: [lit: total*slot] [tok: total*slot] [off16: total*slot].
    const per_subchunk_scratch: usize = @intCast(d.ENTROPY_SCRATCH_SLOT_BYTES);
    const entropy_scratch_bytes = @as(usize, total_subchunks) * per_subchunk_scratch * 3;
    if (!ensureDeviceBuf(&self.d_entropy_scratch, &self.d_entropy_scratch_size, entropy_scratch_bytes)) return error.BadMode;
    const tok_offset = @as(usize, total_subchunks) * per_subchunk_scratch;
    const off16_offset = @as(usize, total_subchunks) * per_subchunk_scratch * 2;
    self.d_entropy_off16_scratch = self.d_entropy_scratch + off16_offset;

    const need_first_sub_idx = total_subchunks != @as(u32, @intCast(chunk_descs.len));
    if (need_first_sub_idx) {
        const fs_bytes: usize = chunk_descs.len * @sizeOf(u32);
        try cudaCall(d2h_fn(@ptrCast(first_subchunk_idx_buf.ptr), self.d_first_sub_idx_persist, fs_bytes));
    } else {
        self.d_first_subchunk_idx = 0;
    }

    // 4d Phase 3 D2D: source bytes already device-resident → D2D-copy
    // them into d_comp_persist (no PCIe). Else H2D from host.
    if (compressed_block.len > 0) {
        if (d_compressed_src) |dev_src| {
            const d2d = cuda.cuMemcpyDtoDAsync_fn orelse return error.BadMode;
            try cudaCall(d2d(self.d_comp_persist, dev_src, compressed_block.len, 0));
        } else {
            try cudaCall(h2d_fn(self.d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len));
        }
    }
    try cudaCall(sync_fn());
    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum.h2d = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
        e2e_cum.prescan = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };

    // ── Scan for entropy chunks (Huffman + raw off16) ─────────
    // scanForEntropyChunks fills both the Huffman descriptor buffers and the
    // raw-off16 gather list. GPU produces only Huffman + raw outputs.
    var scan: ScanResult = .{ .num_raw_off16 = 0 };

    if (ml.huff_build_fn != 0) {
        // SLZ_GPU_SCAN routes the sub-chunk header walk onto the GPU.
        // True D2D (d_compressed_src set) forces it: the CPU path would
        // dereference the sentinel `compressed_block` host slice and
        // segfault. The GPU scan reads from d_comp_persist (the D2D-
        // copied data), which is what we want.
        const want_gpu_scan = ml.scan_parse_fn != 0 and
            chunk_descs.len <= first_subchunk_idx_buf.len and
            (d_compressed_src != null or std.c.getenv("SLZ_GPU_SCAN") != null);
        const gpu_scan: ?ScanResult = if (want_gpu_scan)
            scan_gpu_mod.gpuScanChunks(
                self,
                chunk_descs,
                compressed_block,
                sub_chunk_cap,
                first_subchunk_idx_buf[0..chunk_descs.len],
                total_subchunks,
                &self.huff_lit_host_buf,
                &self.huff_tok_host_buf,
                &self.huff_off16hi_host_buf,
                &self.huff_off16lo_host_buf,
                &self.raw_off16_buf,
            )
        else
            null;

        scan = gpu_scan orelse scan_host_mod.scanForEntropyChunks(
            chunk_descs,
            compressed_block,
            sub_chunk_cap,
            &self.raw_off16_buf,
            &self.huff_lit_host_buf,
            &self.huff_tok_host_buf,
            &self.huff_off16hi_host_buf,
            &self.huff_off16lo_host_buf,
            io,
        );

        dumpScanIfRequested(self, chunk_descs, compressed_block, sub_chunk_cap, scan);

        if (t_e2e0) |t0| if (io) |iv| {
            e2e_cum.postscan = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
        };

        try gatherRawOff16(self, scan, compressed_block, h2d_fn, sync_fn, launch_fn);
    }

    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum.scan = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };

    // ── Huffman pre-decode (Pass 1.5): merge per-stream-type descriptors
    // into a single device array with correct out_offsets, then upload.
    const n_huff: u32 = scan.num_huff_lit + scan.num_huff_tok +
        scan.num_huff_off16hi + scan.num_huff_off16lo;
    if (std.c.getenv("SLZ_HUFF_DBG") != null) {
        std.debug.print("huff scan: lit={d} tok={d} hi={d} lo={d} total={d}\n", .{ scan.num_huff_lit, scan.num_huff_tok, scan.num_huff_off16hi, scan.num_huff_off16lo, n_huff });
    }
    const have_huff = n_huff > 0 and ml.huff_build_fn != 0 and ml.huff_decode_fn != 0;
    if (have_huff) {
        const huff_desc_bytes = @as(usize, n_huff) * @sizeOf(HuffDecChunkDesc);
        const huff_lut_bytes = @as(usize, n_huff) * HUFF_LUT_ENTRIES * @sizeOf(u32);
        if (!ensureDeviceBuf(&self.d_huff_descs, &self.d_huff_descs_size, huff_desc_bytes)) return error.BadMode;
        if (!ensureDeviceBuf(&self.d_huff_lut, &self.d_huff_lut_size, huff_lut_bytes)) return error.BadMode;

        try mergeHuffDescs(self, scan, tok_offset, off16_offset, h2d_fn, sync_fn, launch_fn);
    }

    // ── Pipeline: split into N groups, overlap entropy with LZ ───
    // The pipeline path overlaps Huffman pre-decode with the LZ kernel.
    const total_chunks: u32 = @intCast(chunk_descs.len);
    // pipeline_streams_created is the contract for the launch path below:
    // it owns the Huffman pre-decode and the LZ kernel launch. Without it,
    // Huffman would silently skip and L3+ would decode zero literals. The
    // caller (fullGpuLaunch via ensurePipelineStreams) creates the streams
    // lazily; this is just the guard against a stream-create failure.
    if (!self.pipeline_streams_created) return error.BadMode;
    // When the caller is slzDecompressAsync, work_stream is the caller's
    // CUstream so its cudaStreamSynchronize waits for the decompress to
    // land in d_output. The sync wrapper leaves work_stream at 0 and
    // falls back to the library's own pipeline_streams[0].
    const heavy_stream: usize = if (self.work_stream != 0) self.work_stream else self.pipeline_streams[0];
    {
        const pipe_chunk_count = (total_chunks + NUM_PIPELINE_STREAMS - 1) / NUM_PIPELINE_STREAMS;
        try cudaCall(sync_fn());

        // ── KERNEL TIMER: only pure GPU kernel time from here ──
        const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

        // SLZ_SPLIT_TIMER hoisted above the Huff launches so we can fence
        // around them too. Without split, Huff time gets pipelined into LZ.
        const split_timer = std.c.getenv("SLZ_SPLIT_TIMER") != null;

        const t_huff_start = if (split_timer and have_huff)
            if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
        else
            null;
        var split_huff_build_ns: i64 = 0;
        if (have_huff) {
            const huff_stream = heavy_stream;
            // Step 7: huff kernels self-gate on `*d_n_blocks`. The GPU
            // merge kernel writes n_merged to d_compact_counts+20; reuse
            // that slot when GPU merge ran. CPU-merge fallback stages
            // n_huff into d_n_huff_scratch via a 4 B H2D.
            const d_n_huff: u64 = if (scan.device_compact_populated and ml.merge_huff_descs_fn != 0)
                self.d_compact_counts + 20
            else d_nh: {
                if (!ensureDeviceBuf(&self.d_n_huff_scratch, &self.d_n_huff_scratch_size, 4)) return error.BadMode;
                var host_n_huff: u32 = n_huff;
                try cudaCall(h2d_fn(self.d_n_huff_scratch, @ptrCast(&host_n_huff), 4));
                break :d_nh self.d_n_huff_scratch;
            };
            {
                var p_comp = self.d_comp_persist;
                var p_descs = self.d_huff_descs;
                var p_lut = self.d_huff_lut;
                var p_n = d_n_huff;
                var params = [_]?*anyopaque{
                    @ptrCast(&p_comp), @ptrCast(&p_descs),
                    @ptrCast(&p_lut), @ptrCast(&p_n),
                };
                var extra = [_]?*anyopaque{null};
                const t_hb = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffBuildLutKernel", huff_stream);
                try cudaCall(launch_fn(ml.huff_build_fn, n_huff, 1, 1, 32, 1, 1, 0, huff_stream, &params, &extra));
                endKernelTiming(t_hb, huff_stream);
            }
            // Split fence: time the LUT build separately from the decode.
            if (split_timer) {
                if (cuda.cuStreamSync_fn) |sf| try cudaCall(sf(huff_stream));
                if (t_huff_start) |hs| {
                    if (io) |io_val|
                        split_huff_build_ns = @intCast(hs.untilNow(io_val, .awake).toNanoseconds());
                }
            }
            {
                var p_comp = self.d_comp_persist;
                var p_descs = self.d_huff_descs;
                var p_lut = self.d_huff_lut;
                var p_out = self.d_entropy_scratch;
                var p_n = d_n_huff;
                var params = [_]?*anyopaque{
                    @ptrCast(&p_comp), @ptrCast(&p_descs),
                    @ptrCast(&p_lut), @ptrCast(&p_out), @ptrCast(&p_n),
                };
                var extra = [_]?*anyopaque{null};
                const shared_bytes: c_uint = HUFF_LUT_ENTRIES * @sizeOf(u32);
                const t_hd = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzHuffDecode4StreamKernel", huff_stream);
                try cudaCall(launch_fn(ml.huff_decode_fn, n_huff, 1, 1, 32, 1, 1, shared_bytes, huff_stream, &params, &extra));
                endKernelTiming(t_hd, huff_stream);
            }
        }

        // Launch pipelined groups: one LZ launch per pipeline stream.
        // Operations within a stream are ordered; across streams they can overlap.
        const stream_sync_fn = cuda.cuStreamSync_fn orelse return error.BadMode;

        var split_lz_ns: i64 = 0;
        var split_huff_ns: i64 = 0;

        // Close Huff time slice: sync the pipeline stream so the LZ
        // measurement excludes Huff time.
        if (split_timer and have_huff) {
            try cudaCall(stream_sync_fn(self.pipeline_streams[0]));
            if (t_huff_start) |hs| {
                if (io) |io_val| {
                    split_huff_ns = @intCast(hs.untilNow(io_val, .awake).toNanoseconds());
                }
            }
            std.debug.print("  [huff split] build {d:.3} ms  decode {d:.3} ms\n", .{
                @as(f64, @floatFromInt(split_huff_build_ns)) / 1e6,
                @as(f64, @floatFromInt(split_huff_ns - split_huff_build_ns)) / 1e6,
            });
        }

        for (0..NUM_PIPELINE_STREAMS) |g| {
            const stream: usize = if (NUM_PIPELINE_STREAMS == 1) heavy_stream else self.pipeline_streams[g];
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

            // Step 7: LZ kernels self-gate on `*d_total_chunks`. Stage the
            // per-pipeline-group count into d_n_groups_scratch (4 B H2D).
            if (!ensureDeviceBuf(&self.d_n_groups_scratch, &self.d_n_groups_scratch_size, 4)) return error.BadMode;
            var host_per_group_total: u32 = chunk_end - chunk_start;
            try cudaCall(h2d_fn(self.d_n_groups_scratch, @ptrCast(&host_per_group_total), 4));

            // Fast path: no entropy in this scan → use lean L1/L2 raw kernel.
            // Huffman literals require the general kernel (it reads entropy_scratch).
            const use_raw_kernel = n_huff == 0 and ml.kernel_raw_fn != 0;

            if (use_raw_kernel) {
                var p_comp = self.d_comp_persist;
                var p_descs_dev = self.d_descs_persist + @as(u64, chunk_start) * @sizeOf(ChunkDesc);
                var p_dst = self.d_output;
                var p_cpg = chunks_per_group;
                var p_total = self.d_n_groups_scratch;
                var p_sc_cap = sub_chunk_cap;
                var raw_params = [_]?*anyopaque{
                    @ptrCast(&p_comp),
                    @ptrCast(&p_descs_dev),
                    @ptrCast(&p_dst),
                    @ptrCast(&p_cpg),
                    @ptrCast(&p_total),
                    @ptrCast(&p_sc_cap),
                };
                var raw_extra = [_]?*anyopaque{null};
                const t_lzr = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeRawKernel", stream);
                try cudaCall(launch_fn(ml.kernel_raw_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &raw_params, &raw_extra));
                endKernelTiming(t_lzr, stream);
            } else {
                var p_comp = self.d_comp_persist;
                var p_descs_dev = self.d_descs_persist + @as(u64, chunk_start) * @sizeOf(ChunkDesc);
                var p_dst = self.d_output;
                var p_cpg = chunks_per_group;
                var p_total = self.d_n_groups_scratch;
                var p_sc_cap = sub_chunk_cap;
                var p_entropy_scratch = self.d_entropy_scratch;
                var p_entropy_slot_stride: u64 = @as(u64, total_subchunks) * d.ENTROPY_SCRATCH_SLOT_BYTES;
                var p_first_sub_idx: CUdeviceptr = self.d_first_subchunk_idx +
                    if (self.d_first_subchunk_idx != 0) @as(u64, chunk_start) * @sizeOf(u32) else 0;

                var lz_params = [_]?*anyopaque{
                    @ptrCast(&p_comp),
                    @ptrCast(&p_descs_dev),
                    @ptrCast(&p_dst),
                    @ptrCast(&p_cpg),
                    @ptrCast(&p_total),
                    @ptrCast(&p_sc_cap),
                    @ptrCast(&p_entropy_scratch),
                    @ptrCast(&p_entropy_slot_stride),
                    @ptrCast(&p_first_sub_idx),
                };
                var lz_extra = [_]?*anyopaque{null};

                const t_lz = beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzDecodeKernel", stream);
                try cudaCall(launch_fn(ml.kernel_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &lz_params, &lz_extra));
                endKernelTiming(t_lz, stream);
            }

            if (split_timer) {
                try cudaCall(stream_sync_fn(stream));
                if (t_lz_start) |ts_start| {
                    if (io) |io_val| {
                        split_lz_ns += @intCast(ts_start.untilNow(io_val, .awake).toNanoseconds());
                    }
                }
            }
        }

        // Sync all pipeline streams — UNLESS the caller is async
        // (work_stream set). In async mode we leave the queued work on
        // the user's stream and let them sync; blocking here would
        // defeat the purpose.
        if (self.work_stream == 0) {
            for (0..NUM_PIPELINE_STREAMS) |g| {
                const sync_rc = stream_sync_fn(self.pipeline_streams[g]);
                if (sync_rc != CUDA_SUCCESS) {
                    std.debug.print("GPU pipe[{d}]: stream sync FAILED rc={d}\n", .{ g, sync_rc });
                    return error.BadMode;
                }
            }
        }

        if (t_before_kern) |t_start| {
            if (io) |io_val| {
                facade.last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
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
        e2e_cum.predh = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };
    // Device-resident output (4d Phase 3): D2D-copy the decoded bytes
    // straight to the caller's device buffer, skipping the host bounce.
    // Use heavy_stream so the copy serializes after the LZ kernel on
    // the same stream (and so the user's cudaStreamSynchronize on that
    // stream waits for the result).
    if (d_output_target) |dev_target| {
        if (cuda.cuMemcpyDtoDAsync_fn) |d2d| {
            try cudaCall(d2d(dev_target + dst_start_off, self.d_output + dst_start_off, decompressed_size, heavy_stream));
            if (self.work_stream == 0) try cudaCall(sync_fn());
        } else {
            return error.BadMode;
        }
    } else {
        try cudaCall(d2h_fn(@ptrCast(dst_full + dst_start_off), self.d_output + dst_start_off, decompressed_size));
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
