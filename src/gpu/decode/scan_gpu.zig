//! Device-side decode-scan kernels: frame walker, per-chunk prefix sum,
//! per-sub-chunk header parser + compaction (roadmap 4d Phase 2/3).
//!
//! Mirror of `scan_host.zig` but with the parsing work on the GPU and the
//! compacted descriptor arrays staying device-resident wherever possible.
//! Each entry point gracefully returns null when its kernel symbol isn't
//! loaded so the host-scan fallback in `decode_dispatch.zig` can pick up
//! without a rebuild.

const std = @import("std");

const cuda = @import("cuda_api.zig");
const ml = @import("module_loader.zig");
const d = @import("descriptors.zig");
const dec_ctx = @import("decode_context.zig");

const CUdeviceptr = cuda.CUdeviceptr;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;
const DecodeContext = dec_ctx.DecodeContext;
const ensureDeviceBuf = dec_ctx.ensureDeviceBuf;

/// 4d Phase 3 GPU frame walk — device-only output. Launches the walk
/// kernel and returns the device pointers it wrote to. NO D2H. Caller
/// either passes the device pointers to downstream kernels (true D2D
/// path) or invokes `walkResultToHost` to copy what it needs out.
pub fn gpuWalkFrameImpl(
    self: *DecodeContext,
    d_frame: u64,
    frame_size: u32,
) ?d.WalkFrameResultDev {
    if (!ml.init()) return null;
    if (ml.walk_frame_fn == 0) return null;
    const launch = cuda.cuLaunchKernel_fn orelse return null;
    const sync = cuda.cuCtxSynchronize_fn orelse return null;
    const memset = cuda.cuMemsetD8_fn orelse return null;

    const chunks_bytes: usize = @as(usize, d.walk_max_chunks) * @sizeOf(d.ChunkDesc);
    if (!ensureDeviceBuf(&self.d_walk_chunks, &self.d_walk_chunks_size, chunks_bytes)) return null;
    if (!ensureDeviceBuf(&self.d_walk_meta, &self.d_walk_meta_size, d.walk_meta_offsets.bytes)) return null;
    if (memset(self.d_walk_meta, 0, d.walk_meta_offsets.bytes) != CUDA_SUCCESS) return null;

    var k_frame = d_frame;
    var k_size = frame_size;
    var k_chunks = self.d_walk_chunks;
    var k_max = d.walk_max_chunks;
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
    const t_walk = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzWalkFrameKernel", 0);
    if (launch(ml.walk_frame_fn, 1, 1, 1, 1, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS) return null;
    dec_ctx.endKernelTiming(t_walk, 0);
    if (sync() != CUDA_SUCCESS) return null;

    return .{
        .d_chunk_descs = self.d_walk_chunks,
        .d_meta = self.d_walk_meta,
    };
}

pub fn gpuPrefixSumChunksImpl(
    self: *DecodeContext,
    d_chunk_descs: u64,
    n_chunks: u32,
    sub_chunk_cap: u32,
) ?d.PrefixSumResultDev {
    if (!ml.init()) return null;
    if (ml.prefix_sum_chunks_fn == 0) return null;
    const launch = cuda.cuLaunchKernel_fn orelse return null;
    const sync = cuda.cuCtxSynchronize_fn orelse return null;

    const first_bytes: usize = @as(usize, d.walk_max_chunks) * 4;
    if (!ensureDeviceBuf(&self.d_first_sub_idx_persist, &self.d_first_sub_idx_persist_size, first_bytes)) return null;
    if (!ensureDeviceBuf(&self.d_total_subchunks_buf, &self.d_total_subchunks_buf_size, 4)) return null;

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
    const t_prefix = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzPrefixSumChunksKernel", 0);
    if (launch(ml.prefix_sum_chunks_fn, 1, 1, 1, 1, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS) return null;
    dec_ctx.endKernelTiming(t_prefix, 0);
    if (sync() != CUDA_SUCCESS) return null;

    return .{
        .d_first_sub_idx = self.d_first_sub_idx_persist,
        .d_total_subchunks = self.d_total_subchunks_buf,
    };
}

pub fn walkMetaToHost(d_meta: u64) ?d.WalkMeta {
    const d2h = cuda.cuMemcpyDtoH_fn orelse return null;
    var m: [6]u32 = .{0} ** 6;
    if (d2h(@ptrCast(&m), d_meta, d.walk_meta_offsets.bytes) != CUDA_SUCCESS) return null;
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
    const sync = cuda.cuCtxSynchronize_fn orelse return null;
    const memset = cuda.cuMemsetD8_fn orelse return null;
    // The scan kernel reads the device-resident prefix sum
    // (d_first_sub_idx_persist) directly; no host first_subchunk_idx
    // mirror is needed here.

    // Staged buffer: [lit][tok][hi][lo] ScanHuffDesc, then [raw_hi][raw_lo]
    // ScanRawDesc — one entry per global sub-chunk index per stream type.
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
    const t_scan = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzScanParseKernel", 0);
    if (launch(ml.scan_parse_fn, blocks, 1, 1, tpb, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS) return null;
    dec_ctx.endKernelTiming(t_scan, 0);
    if (sync() != CUDA_SUCCESS) return null;

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
        // Size compact buffers. n_huff bound: walk_max_chunks * 4
        // (sub-chunks per chunk at sc>=1). n_raw bound: 2 × that.
        const huff_compact_max = @as(usize, d.walk_max_chunks) * 4;
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
            const t_ch = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, compact_names[ci], 0);
            if (launch(ml.compact_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, 0, &c_params, &c_extra) != CUDA_SUCCESS) return null;
            dec_ctx.endKernelTiming(t_ch, 0);
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
            const t_cr = dec_ctx.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzCompactRawDescsKernel", 0);
            if (launch(ml.compact_raw_descs_fn, 1, 1, 1, 1, 1, 1, 0, 0, &cr_params, &cr_extra) != CUDA_SUCCESS) return null;
            dec_ctx.endKernelTiming(t_cr, 0);
        }
        if (sync() != CUDA_SUCCESS) return null;

        // The 5 counts come back as a single 20 B D2H.
        var counts: [5]u32 = .{ 0, 0, 0, 0, 0 };
        if (d2h(@ptrCast(&counts), self.d_compact_counts, 5 * 4) != CUDA_SUCCESS) return null;
        num_lit = counts[0];
        num_tok = counts[1];
        num_hi = counts[2];
        num_lo = counts[3];
        num_raw = counts[4];

        // The legacy CPU-merge fallback consumes huff_*_host_buf /
        // raw_off16_descs. The pure-D2D merge kernel reads d_compact_*
        // directly, so the host arrays are only needed when the merge
        // kernel itself is missing. Skip the ~120 KB D2H in the common
        // case; populate host arrays only as a true fallback path.
        if (ml.merge_huff_descs_fn == 0) {
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
