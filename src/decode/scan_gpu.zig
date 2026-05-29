//! Device-side decode-scan kernels: frame walker, per-chunk prefix sum,
//! per-sub-chunk header parser + compaction.
//!
//! The compacted descriptor arrays stay device-resident wherever
//! possible; the host gets only the small per-stream counts.
//!
//! Convention: most entry points return `?T` (null on failure) because
//! callers in `decode_dispatch.zig` translate the failure to an outer
//! `error.BackendNotAvailable`. The exception is `gpuPrefixSumChunksImpl`,
//! which returns `GpuError!T` directly because there is no fallback for
//! the device-side prefix sum and the dispatch needs to distinguish
//! alloc / launch / sync failures from missing-symbol failures.

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

/// GPU frame walk - device-only output. Launches the walk kernel and
/// returns the device pointers it wrote to; NO D2H. Caller either passes
/// the device pointers to downstream kernels (the true D2D path) or
/// invokes `walkResultToHost` to copy what it needs out.
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

    // Launch on `work_stream` (= 0 in the sync wrapper, caller's stream
    // in the async wrapper).
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

    // Launch on caller's stream (= 0 in the sync wrapper, caller's
    // stream in the async wrapper). No internal sync: downstream
    // consumers are kernels queued on the same stream that serialize
    // naturally; the result device pointers
    // (`d_first_sub_idx_persist`, `d_total_subchunks_buf`) are read by
    // kernels, never by host.
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

pub fn gpuScanChunks(
    self: *DecodeContext,
    chunk_descs: []const d.ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    total_subchunks: u32,
) ?d.ScanResult {
    if (ml.scan_parse_fn == 0) return null;
    if (ml.compact_huff_descs_fn == 0 or ml.compact_raw_descs_fn == 0) return null;
    const n: u32 = @intCast(chunk_descs.len);
    if (n == 0 or total_subchunks == 0) return null;

    const h2d = cuda.cuMemcpyHtoD_fn orelse return null;
    const launch = cuda.cuLaunchKernel_fn orelse return null;
    const memset = cuda.cuMemsetD8_fn orelse return null;
    const stream = self.work_stream;

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
    var k_first = self.d_first_sub_idx_persist;
    // Scan kernel self-gates on `*d_n_chunks` - stage host n into a
    // device-resident 4 B slot.
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
    // No post-scan sync: the compact kernels below queue on the same
    // stream and see scan_parse's writes to d_scan_staged via stream
    // ordering.

    // Device-side compaction: 4 × slzCompactHuffDescsKernel + 1 ×
    // slzCompactRawDescsKernel produce compacted per-stream arrays
    // + per-stream counts entirely on device. Nothing comes back to
    // the host here - downstream kernels (merge, huff_build, gather)
    // read straight from d_compact_*.
    {
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
        if (ml.merge_huff_descs_fn == 0) return null;
    }

    // Counts stay device-resident in d_compact_counts; downstream merge,
    // huff_build, huff_decode, and gather kernels read them as device
    // pointers and self-gate. The ScanResult reports worst-case bounds
    // (= total_subchunks per Huffman stream type + 2× for the two raw
    // streams) so the dispatch's allocator sizes scratch correctly; the
    // actual non-zero work happens at kernel time. On L1/L2 frames (no
    // entropy → real counts of 0) the downstream kernels still launch
    // with worst-case grids and early-exit on the first thread's check
    // (~15 μs of empty GPU time).
    return .{
        .num_raw_off16    = total_subchunks * 2,
        .num_huff_lit     = total_subchunks,
        .num_huff_tok     = total_subchunks,
        .num_huff_off16hi = total_subchunks,
        .num_huff_off16lo = total_subchunks,
    };
}
