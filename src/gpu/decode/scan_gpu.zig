//! Device-side decode-scan kernels: frame walker, per-chunk prefix sum,
//! per-sub-chunk header parser + compaction (roadmap 4d Phase 2/3).
//!
//! Mirror of `scan_host.zig` but with the parsing work on the GPU and the
//! compacted descriptor arrays staying device-resident wherever possible.
//! Each entry point gracefully returns null when its kernel symbol isn't
//! loaded so the host-scan fallback in `decode_dispatch.zig` can pick up
//! without a rebuild.
//!
//! Convention note: most entry points in this file use the `?T` return
//! + raw `if (rc != CUDA_SUCCESS) return null;` pattern intentionally,
//! NOT `GpuError!T` + `try cudaCall(...)` like `decode_dispatch.zig`.
//! Reason: those scan_gpu functions are best-effort fast paths - null
//! means "fall back to host scan" (or skip this optimization), not
//! "the decode has failed". The caller in `fullGpuLaunchImpl` always
//! has the `scan_host.zig` fallback available. Promoting CUDA failures
//! to errors there would force the caller to either re-fallback in a
//! catch (back to null-equivalent) or fail the whole decode on what is
//! recoverable.
//!
//! Exception: `gpuPrefixSumChunksImpl` returns `GpuError!T` because the
//! prefix-sum has NO host fallback - every GPU decode path requires it.
//! Null-on-failure would have silently masked alloc/launch/sync failures
//! as the generic `error.BadMode` at the call site. K5.1's error fan-out
//! should reach this function too.

const std = @import("std");

const cuda = @import("cuda_api.zig");
const ml = @import("module_loader.zig");
const d = @import("descriptors.zig");
const dec_ctx = @import("decode_context.zig");
const desc_err = @import("descriptors.zig");

const CUdeviceptr = cuda.CUdeviceptr;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;
const DecodeContext = dec_ctx.DecodeContext;
const ensureDeviceBuf = dec_ctx.ensureDeviceBuf;
const GpuError = desc_err.GpuError;
const cudaCall = desc_err.cudaCall;

/// 4d Phase 3 GPU frame walk - device-only output. Launches the walk
/// kernel and returns the device pointers it wrote to. NO D2H. Caller
/// either passes the device pointers to downstream kernels (true D2D
/// path) or invokes `walkResultToHost` to copy what it needs out.
///
/// Returns specific GpuError variants like `gpuPrefixSumChunksImpl`:
/// frame-walk is on the device-resident decode hot path with no host
/// fallback, so distinguishing alloc / launch / sync failures from
/// missing-symbol failures at the caller matters.
pub fn gpuWalkFrameImpl(
    self: *DecodeContext,
    d_frame: u64,
    frame_size: u32,
) GpuError!d.WalkFrameResultDev {
    if (!ml.init()) return error.BackendNotAvailable;
    if (ml.walk_frame_fn == 0) return error.KernelMissing;
    const launch = cuda.cuLaunchKernel_fn orelse return error.BackendNotAvailable;
    const memset = cuda.cuMemsetD8_fn orelse return error.BackendNotAvailable;

    const chunks_bytes: usize = @as(usize, d.WALK_MAX_CHUNKS) * @sizeOf(d.ChunkDesc);
    if (!ensureDeviceBuf(&self.d_walk_chunks, &self.d_walk_chunks_size, chunks_bytes)) return error.OutOfDeviceMemory;
    if (!ensureDeviceBuf(&self.d_walk_meta, &self.d_walk_meta_size, d.walk_meta_offsets.bytes)) return error.OutOfDeviceMemory;
    try cudaCall(memset(self.d_walk_meta, 0, d.walk_meta_offsets.bytes), .copy);

    // Phase 3c: launch on work_stream (sync=0, async=caller's stream).
    // No post-kernel sync here — the caller's walkMetaToHostAsync (or
    // any subsequent kernel on the same stream) serializes naturally.
    const stream = self.work_stream;
    var k_frame = d_frame;
    var k_size = frame_size;
    var k_chunks = self.d_walk_chunks;
    var k_max = d.WALK_MAX_CHUNKS;
    var k_meta_n: u64 = self.d_walk_meta + d.walk_meta_offsets.n_chunks;
    var k_meta_decomp: u64 = self.d_walk_meta + d.walk_meta_offsets.decomp_size;
    var k_meta_sccap: u64 = self.d_walk_meta + d.walk_meta_offsets.sub_chunk_cap;
    var k_meta_bstart: u64 = self.d_walk_meta + d.walk_meta_offsets.block_start;
    var k_meta_bsize: u64 = self.d_walk_meta + d.walk_meta_offsets.block_size;
    var k_meta_status: u64 = self.d_walk_meta + d.walk_meta_offsets.status;
    var params = [_]?*anyopaque{
        @ptrCast(&k_frame),       @ptrCast(&k_size),
        @ptrCast(&k_chunks),      @ptrCast(&k_max),
        @ptrCast(&k_meta_n),      @ptrCast(&k_meta_decomp),
        @ptrCast(&k_meta_sccap),  @ptrCast(&k_meta_bstart),
        @ptrCast(&k_meta_bsize),  @ptrCast(&k_meta_status),
    };
    var extra = [_]?*anyopaque{null};
    const t_walk = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzWalkFrameKernel", stream);
    try cudaCall(launch(ml.walk_frame_fn, 1, 1, 1, 1, 1, 1, 0, stream, &params, &extra), .launch);
    dec_ctx.endKernelTiming(t_walk, stream);

    return .{
        .d_chunk_descs = self.d_walk_chunks,
        .d_meta = self.d_walk_meta,
    };
}

/// Per-chunk prefix sum of sub-chunk counts. Required by every GPU
/// decode path - there is no host fallback (unlike the rest of this
/// file's `?T`-returning functions, which DO have host fallbacks).
/// Returns specific GpuError variants so the caller can surface
/// out-of-memory / launch-failure / sync-failure distinctly instead
/// of flattening them to error.BadMode.
pub fn gpuPrefixSumChunksImpl(
    self: *DecodeContext,
    d_chunk_descs: u64,
    n_chunks: u32,
    sub_chunk_cap: u32,
) GpuError!d.PrefixSumResultDev {
    if (!ml.init()) return error.BackendNotAvailable;
    if (ml.prefix_sum_chunks_fn == 0) return error.KernelMissing;
    const launch = cuda.cuLaunchKernel_fn orelse return error.BackendNotAvailable;

    const first_bytes: usize = @as(usize, d.WALK_MAX_CHUNKS) * 4;
    if (!ensureDeviceBuf(&self.d_first_sub_idx_persist, &self.d_first_sub_idx_persist_size, first_bytes)) return error.OutOfDeviceMemory;
    if (!ensureDeviceBuf(&self.d_total_subchunks_buf, &self.d_total_subchunks_buf_size, 4)) return error.OutOfDeviceMemory;

    // Phase 2: launch on caller's stream when async (0 otherwise).
    // Phase 3: no internal sync — downstream consumers are kernels on
    // the same stream and serialize naturally. Result device pointers
    // (d_first_sub_idx, d_total_subchunks_buf) are read by kernels, not
    // by host.
    const stream = self.work_stream;
    var k_chunks = d_chunk_descs;
    var k_n = n_chunks;
    var k_cap = sub_chunk_cap;
    var k_first = self.d_first_sub_idx_persist;
    var k_total = self.d_total_subchunks_buf;
    var params = [_]?*anyopaque{
        @ptrCast(&k_chunks), @ptrCast(&k_n), @ptrCast(&k_cap),
        @ptrCast(&k_first), @ptrCast(&k_total),
    };
    var extra = [_]?*anyopaque{null};
    const t_prefix = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzPrefixSumChunksKernel", stream);
    try cudaCall(launch(ml.prefix_sum_chunks_fn, 1, 1, 1, 1, 1, 1, 0, stream, &params, &extra), .launch);
    dec_ctx.endKernelTiming(t_prefix, stream);

    return .{
        .d_first_sub_idx = self.d_first_sub_idx_persist,
        .d_total_subchunks = self.d_total_subchunks_buf,
    };
}

/// D2H the walk metadata struct from the device.
/// Phase 3c: now async on `stream` + stream-targeted sync — preserves
/// caller's other-stream parallelism (the prior synchronous cuMemcpyDtoH
/// implicitly stalled stream 0 against every other stream in the context).
/// Status check is the caller's responsibility — the GPU decode contract
/// is that the input frame came from the GPU encode path; CPU-produced
/// frames (dict / multi-block / PDM / checksumed) are out of scope and
/// allowed to fail loudly (see feedback_cpu_gpu_separate_formats).
pub fn walkMetaToHost(d_meta: u64, stream: usize) GpuError!d.WalkMeta {
    const d2h_async = cuda.cuMemcpyDtoHAsync_fn orelse return error.BackendNotAvailable;
    const stream_sync = cuda.cuStreamSync_fn orelse return error.BackendNotAvailable;
    var m: [6]u32 = .{0} ** 6;
    try cudaCall(d2h_async(@ptrCast(&m), d_meta, d.walk_meta_offsets.bytes, stream), .copy);
    try cudaCall(stream_sync(stream), .sync);
    return .{
        .n_chunks = m[0], .decomp_size = m[1], .sub_chunk_cap = m[2],
        .block_start = m[3], .block_size = m[4], .status = m[5],
    };
}

pub fn gpuScanChunks(
    self: *DecodeContext,
    chunk_descs: []const d.ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    total_subchunks: u32,
    huff_lit_descs: []d.HuffDecChunkDesc,
    huff_tok_descs: []d.HuffDecChunkDesc,
    huff_off16hi_descs: []d.HuffDecChunkDesc,
    huff_off16lo_descs: []d.HuffDecChunkDesc,
    raw_off16_descs: []d.RawOff16Desc,
) ?d.ScanResult {
    if (ml.scan_parse_fn == 0) return null;
    const n: u32 = @intCast(chunk_descs.len);
    if (n == 0 or total_subchunks == 0) return null;

    const d2h = cuda.cuMemcpyDtoH_fn orelse return null;
    const h2d = cuda.cuMemcpyHtoD_fn orelse return null;
    const launch = cuda.cuLaunchKernel_fn orelse return null;
    const stream_sync = cuda.cuStreamSync_fn orelse return null;
    const memset = cuda.cuMemsetD8_fn orelse return null;
    // Phase 2: launch on caller's work_stream when async, stream 0 when
    // sync. Stream-targeted sync below preserves caller's other-stream
    // parallelism.
    const stream = self.work_stream;
    // The scan kernel reads the device-resident prefix sum
    // (d_first_sub_idx_persist) directly; no host first_subchunk_idx
    // mirror is needed here.

    // Staged buffer: [lit][tok][hi][lo] ScanHuffDesc, then [raw_hi][raw_lo]
    // ScanRawDesc - one entry per global sub-chunk index per stream type.
    const huff_arr_bytes: usize = @as(usize, total_subchunks) * @sizeOf(d.ScanHuffDesc);
    const raw_arr_bytes: usize = @as(usize, total_subchunks) * @sizeOf(d.ScanRawDesc);
    const staged_bytes: usize = huff_arr_bytes * 4 + raw_arr_bytes * 2;
    if (!ensureDeviceBuf(&self.d_scan_staged, &self.d_scan_staged_size, staged_bytes)) return null;
    // Zero so sub-chunk slots that no thread reaches keep valid=0.
    if (memset(self.d_scan_staged, 0, staged_bytes) != CUDA_SUCCESS) return null;

    const base = self.d_scan_staged;
    var k_block = self.d_comp_persist;
    var k_blen: u32 = @intCast(compressed_block.len);
    var k_chunks = self.d_descs_persist;
    // Use the device-resident prefix-sum output directly. The legacy
    // host array (`first_subchunk_idx`) is zero-filled for the
    // `total_subchunks == n_chunks` case in fullGpuLaunchImpl and
    // would feed the scan kernel a bogus prefix sum.
    var k_first = self.d_first_sub_idx_persist;
    // Step 7: scan kernel self-gates on `*d_n_chunks`. Stage host n
    // into d_n_chunks_scratch (4 B H2D).
    if (!ensureDeviceBuf(&self.d_n_chunks_scratch, &self.d_n_chunks_scratch_size, 4)) return null;
    var host_n_chunks: u32 = n;
    if (h2d(self.d_n_chunks_scratch, @ptrCast(&host_n_chunks), 4) != CUDA_SUCCESS) return null;
    var k_n: u64 = self.d_n_chunks_scratch;
    var k_cap = sub_chunk_cap;
    var k_lit = base;
    var k_tok = base + huff_arr_bytes;
    var k_hi = base + huff_arr_bytes * 2;
    var k_lo = base + huff_arr_bytes * 3;
    var k_rhi = base + huff_arr_bytes * 4;
    var k_rlo = base + huff_arr_bytes * 4 + raw_arr_bytes;
    var params = [_]?*anyopaque{
        @ptrCast(&k_block), @ptrCast(&k_blen),  @ptrCast(&k_chunks),
        @ptrCast(&k_first), @ptrCast(&k_n),     @ptrCast(&k_cap),
        @ptrCast(&k_lit),   @ptrCast(&k_tok),   @ptrCast(&k_hi),
        @ptrCast(&k_lo),    @ptrCast(&k_rhi),   @ptrCast(&k_rlo),
    };
    var extra = [_]?*anyopaque{null};
    const tpb: u32 = 256;
    const blocks: u32 = (n + tpb - 1) / tpb;
    const t_scan = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzScanParseKernel", stream);
    if (launch(ml.scan_parse_fn, blocks, 1, 1, tpb, 1, 1, 0, stream, &params, &extra) != CUDA_SUCCESS) return null;
    dec_ctx.endKernelTiming(t_scan, stream);
    if (stream_sync(stream) != CUDA_SUCCESS) return null;

    // ── Step 6b: device-side compaction ─────────────────────────────
    // 4 × slzCompactHuffDescsKernel + 1 × slzCompactRawDescsKernel
    // produce compacted per-stream arrays + counts entirely on device.
    // Total D2H from this fn: 5 × u32 counts (20 B). Falls back to the
    // host-side compact path if any compact-kernel symbol is missing.
    const gpu_compact_ok =
        ml.compact_huff_descs_fn != 0 and ml.compact_raw_descs_fn != 0;

    var num_lit: u32 = 0;
    var num_tok: u32 = 0;
    var num_hi: u32 = 0;
    var num_lo: u32 = 0;
    var num_raw: u32 = 0;

    if (gpu_compact_ok) {
        // Size compact buffers. n_huff bound: WALK_MAX_CHUNKS * 4
        // (sub-chunks per chunk at sc>=1). n_raw bound: 2 × that.
        const huff_compact_max = @as(usize, d.WALK_MAX_CHUNKS) * 4;
        const huff_compact_bytes = huff_compact_max * @sizeOf(d.HuffDecChunkDesc);
        const raw_compact_max = huff_compact_max * 2;
        const raw_compact_bytes = raw_compact_max * @sizeOf(d.RawOff16Desc);
        if (!ensureDeviceBuf(&self.d_compact_lit, &self.d_compact_lit_size, huff_compact_bytes)) return null;
        if (!ensureDeviceBuf(&self.d_compact_tok, &self.d_compact_tok_size, huff_compact_bytes)) return null;
        if (!ensureDeviceBuf(&self.d_compact_hi, &self.d_compact_hi_size, huff_compact_bytes)) return null;
        if (!ensureDeviceBuf(&self.d_compact_lo, &self.d_compact_lo_size, huff_compact_bytes)) return null;
        if (!ensureDeviceBuf(&self.d_compact_raw, &self.d_compact_raw_size, raw_compact_bytes)) return null;
        // 6 u32: [n_lit, n_tok, n_hi, n_lo, n_raw, n_merged]. n_merged is
        // written by slzMergeHuffDescsKernel in step 6c.
        if (!ensureDeviceBuf(&self.d_compact_counts, &self.d_compact_counts_size, 6 * 4)) return null;
        if (memset(self.d_compact_counts, 0, 6 * 4) != CUDA_SUCCESS) return null;

        const huff_streams = [_]struct { staged_off: usize, dst: u64, n_off: u32 }{
            .{ .staged_off = 0,                  .dst = self.d_compact_lit, .n_off = 0 },
            .{ .staged_off = huff_arr_bytes,     .dst = self.d_compact_tok, .n_off = 4 },
            .{ .staged_off = huff_arr_bytes * 2, .dst = self.d_compact_hi,  .n_off = 8 },
            .{ .staged_off = huff_arr_bytes * 3, .dst = self.d_compact_lo,  .n_off = 12 },
        };
        const compact_names = [_][*:0]const u8{
            "slzCompactHuffDescsKernel (lit)",
            "slzCompactHuffDescsKernel (tok)",
            "slzCompactHuffDescsKernel (hi)",
            "slzCompactHuffDescsKernel (lo)",
        };
        for (huff_streams, 0..) |hs, ci| {
            var k_staged: u64 = base + hs.staged_off;
            var k_total: u64 = self.d_total_subchunks_buf;
            var k_dst: u64 = hs.dst;
            var k_count: u64 = self.d_compact_counts + hs.n_off;
            var c_params = [_]?*anyopaque{
                @ptrCast(&k_staged), @ptrCast(&k_total),
                @ptrCast(&k_dst), @ptrCast(&k_count),
            };
            var c_extra = [_]?*anyopaque{null};
            const t_ch = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, compact_names[ci], stream);
            if (launch(ml.compact_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, stream, &c_params, &c_extra) != CUDA_SUCCESS) return null;
            dec_ctx.endKernelTiming(t_ch, stream);
        }
        // Raw compact.
        {
            var cr_hi: u64 = base + huff_arr_bytes * 4;
            var cr_lo: u64 = base + huff_arr_bytes * 4 + raw_arr_bytes;
            var cr_total: u64 = self.d_total_subchunks_buf;
            var cr_dst: u64 = self.d_compact_raw;
            var cr_count: u64 = self.d_compact_counts + 16;
            var cr_params = [_]?*anyopaque{
                @ptrCast(&cr_hi), @ptrCast(&cr_lo), @ptrCast(&cr_total),
                @ptrCast(&cr_dst), @ptrCast(&cr_count),
            };
            var cr_extra = [_]?*anyopaque{null};
            const t_cr = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzCompactRawDescsKernel", stream);
            if (launch(ml.compact_raw_descs_fn, 1, 1, 1, 1, 1, 1, 0, stream, &cr_params, &cr_extra) != CUDA_SUCCESS) return null;
            dec_ctx.endKernelTiming(t_cr, stream);
        }
        if (ml.merge_huff_descs_fn != 0) {
            // Phase 3d (pure-D2D merge path, the common case): skip the
            // compact-counts D2H + the cuStreamSynchronize that gated
            // it. The merge kernel + huff_build + huff_decode + the
            // raw-gather kernel all read their counts from
            // d_compact_counts as device pointers (already wired up
            // below) and self-gate. Downstream allocations use
            // worst-case bounds (= total_subchunks per Huff stream type
            // + 2× total_subchunks for the two raw streams).
            //
            // Cost: on L1/L2 frames (no entropy → actual counts 0) the
            // merge / huff_build / huff_decode kernels still launch
            // with worst-case grids and early-exit on
            // *d_n_merged == 0. Estimated ~15 μs additional GPU time
            // on the L1/L2 D2D path. Phase 4 (CUDA Graphs) recovers
            // this and more by eliminating per-call launch latency on
            // the captured pipeline.
            num_lit = total_subchunks;
            num_tok = total_subchunks;
            num_hi = total_subchunks;
            num_lo = total_subchunks;
            num_raw = total_subchunks * 2;
        } else {
            // Legacy host-merge fallback (merge_huff_descs_fn unloaded):
            // need actual counts host-side to size the host-buf appends
            // + D2H the compacted descriptors into the host buffers.
            // Hot path doesn't take this branch on any in-tree build —
            // it's defensive against a future driver build dropping the
            // merge kernel.
            if (stream_sync(stream) != CUDA_SUCCESS) return null;
            var counts: [5]u32 = .{ 0, 0, 0, 0, 0 };
            if (d2h(@ptrCast(&counts), self.d_compact_counts, 5 * 4) != CUDA_SUCCESS) return null;
            num_lit = counts[0];
            num_tok = counts[1];
            num_hi = counts[2];
            num_lo = counts[3];
            num_raw = counts[4];

            if (num_lit > 0 and num_lit <= huff_lit_descs.len)
                if (d2h(@ptrCast(huff_lit_descs.ptr), self.d_compact_lit, @as(usize, num_lit) * @sizeOf(d.HuffDecChunkDesc)) != CUDA_SUCCESS) return null;
            if (num_tok > 0 and num_tok <= huff_tok_descs.len)
                if (d2h(@ptrCast(huff_tok_descs.ptr), self.d_compact_tok, @as(usize, num_tok) * @sizeOf(d.HuffDecChunkDesc)) != CUDA_SUCCESS) return null;
            if (num_hi > 0 and num_hi <= huff_off16hi_descs.len)
                if (d2h(@ptrCast(huff_off16hi_descs.ptr), self.d_compact_hi, @as(usize, num_hi) * @sizeOf(d.HuffDecChunkDesc)) != CUDA_SUCCESS) return null;
            if (num_lo > 0 and num_lo <= huff_off16lo_descs.len)
                if (d2h(@ptrCast(huff_off16lo_descs.ptr), self.d_compact_lo, @as(usize, num_lo) * @sizeOf(d.HuffDecChunkDesc)) != CUDA_SUCCESS) return null;
            if (num_raw > 0 and num_raw <= raw_off16_descs.len)
                if (d2h(@ptrCast(raw_off16_descs.ptr), self.d_compact_raw, @as(usize, num_raw) * @sizeOf(d.RawOff16Desc)) != CUDA_SUCCESS) return null;
        }
    } else {
        // Fallback: D2H the staged arrays and compact on host.
        const alloc = std.heap.page_allocator;
        const staged = alloc.alloc(u8, staged_bytes) catch return null;
        defer alloc.free(staged);
        if (d2h(@ptrCast(staged.ptr), base, staged_bytes) != CUDA_SUCCESS) return null;

        const lit_st: [*]const d.ScanHuffDesc = @ptrCast(@alignCast(staged.ptr));
        const tok_st: [*]const d.ScanHuffDesc = @ptrCast(@alignCast(staged.ptr + huff_arr_bytes));
        const hi_st: [*]const d.ScanHuffDesc = @ptrCast(@alignCast(staged.ptr + huff_arr_bytes * 2));
        const lo_st: [*]const d.ScanHuffDesc = @ptrCast(@alignCast(staged.ptr + huff_arr_bytes * 3));
        const rhi_st: [*]const d.ScanRawDesc = @ptrCast(@alignCast(staged.ptr + huff_arr_bytes * 4));
        const rlo_st: [*]const d.ScanRawDesc = @ptrCast(@alignCast(staged.ptr + huff_arr_bytes * 4 + raw_arr_bytes));

        const compactHuff = struct {
            fn run(st: [*]const d.ScanHuffDesc, tot: u32, out: []d.HuffDecChunkDesc) u32 {
                var k: u32 = 0;
                var i: u32 = 0;
                while (i < tot) : (i += 1) {
                    if (st[i].valid == 0) continue;
                    if (k >= out.len) break;
                    out[k] = .{
                        .in_offset = st[i].in_offset,
                        .in_size = st[i].in_size,
                        .out_offset = st[i].out_offset,
                        .out_size = st[i].out_size,
                        .lut_offset = k * @as(u32, @intCast(d.HUFF_LUT_ENTRIES)),
                    };
                    k += 1;
                }
                return k;
            }
        }.run;
        num_lit = compactHuff(lit_st, total_subchunks, huff_lit_descs);
        num_tok = compactHuff(tok_st, total_subchunks, huff_tok_descs);
        num_hi = compactHuff(hi_st, total_subchunks, huff_off16hi_descs);
        num_lo = compactHuff(lo_st, total_subchunks, huff_off16lo_descs);

        var si: u32 = 0;
        while (si < total_subchunks) : (si += 1) {
            if (rhi_st[si].valid != 0 and num_raw < raw_off16_descs.len) {
                raw_off16_descs[num_raw] = .{ .src_offset = rhi_st[si].src_offset, .size = rhi_st[si].size, .gpu_offset = rhi_st[si].gpu_offset };
                num_raw += 1;
            }
            if (rlo_st[si].valid != 0 and num_raw < raw_off16_descs.len) {
                raw_off16_descs[num_raw] = .{ .src_offset = rlo_st[si].src_offset, .size = rlo_st[si].size, .gpu_offset = rlo_st[si].gpu_offset };
                num_raw += 1;
            }
        }
    }

    return .{
        .num_raw_off16 = num_raw,
        .num_huff_lit = num_lit,
        .num_huff_tok = num_tok,
        .num_huff_off16hi = num_hi,
        .num_huff_off16lo = num_lo,
        .device_compact_populated = gpu_compact_ok,
    };
}
