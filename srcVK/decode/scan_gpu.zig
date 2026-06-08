//! 1:1 port of src/decode/scan_gpu.zig.
//!
//! Three GPU dispatch entry points used by the decode path: the L2
//! `slzWalkFrameKernel` launch, the L1 `slzPrefixSumChunksKernel` launch
//! (always fires — required by every decode), and the L2 scan/compact
//! chain (`slzScanParseKernel` + `slzCompactHuffDescsKernel` +
//! `slzCompactRawDescsKernel`). All device ops funnel through
//! `vulkan_api.procs.*` — no direct vk*/vma calls in this file.

const std = @import("std");
const vk = @import("vulkan_api.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const module_loader = @import("module_loader.zig");

const DecodeContext = decode_context.DecodeContext;
const GpuError = descriptors.GpuError;
const ensureDeviceBuf = decode_context.ensureDeviceBuf;
const vkCall = descriptors.vkCall;

/// CUDA reference: src/decode/scan_gpu.zig:37-82. GPU frame walk —
/// device-only output. Launches the walk kernel and returns the device
/// pointers it wrote to; NO D2H. Used by `decompressFramedFromDevice`
/// to keep the entire decode pipeline device-resident on the L1 raw
/// (and future L2+) path.
pub fn gpuWalkFrameImpl(
    self: *DecodeContext,
    d_frame: u64,
    frame_size: u32,
) GpuError!descriptors.WalkFrameResultDev {
    if (!module_loader.init()) return error.BackendNotAvailable;
    if (module_loader.walk_frame_fn == 0) return error.KernelMissing;
    const launch = vk.procs.launch_kernel orelse return error.BackendNotAvailable;
    const memset = vk.procs.memset_d8 orelse return error.BackendNotAvailable;

    const chunks_bytes: usize = @as(usize, descriptors.WALK_MAX_CHUNKS) * @sizeOf(descriptors.ChunkDesc);
    try ensureDeviceBuf(&self.d_walk_chunks, &self.d_walk_chunks_size, chunks_bytes);
    try ensureDeviceBuf(&self.d_walk_meta, &self.d_walk_meta_size, descriptors.walk_meta_offsets.bytes);
    try vkCall(memset(self.d_walk_meta, 0, descriptors.walk_meta_offsets.bytes), .copy);

    // Launch on caller's `work_stream` (= 0 in sync, caller's stream in
    // async). No post-kernel sync — any subsequent kernel queued on the
    // same stream serializes via stream ordering.
    const stream = self.work_stream;
    var k_frame: u64 = d_frame;
    var k_chunks: u64 = self.d_walk_chunks;
    var k_meta_n: u64 = self.d_walk_meta + descriptors.walk_meta_offsets.n_chunks;
    var k_meta_decomp: u64 = self.d_walk_meta + descriptors.walk_meta_offsets.decomp_size;
    var k_meta_sccap: u64 = self.d_walk_meta + descriptors.walk_meta_offsets.sub_chunk_cap;
    var k_meta_bstart: u64 = self.d_walk_meta + descriptors.walk_meta_offsets.block_start;
    var k_meta_bsize: u64 = self.d_walk_meta + descriptors.walk_meta_offsets.block_size;
    var k_meta_status: u64 = self.d_walk_meta + descriptors.walk_meta_offsets.status;
    var k_size: u32 = frame_size;
    var k_max: u32 = descriptors.WALK_MAX_CHUNKS;
    // walk_frame: n_bindings=8 (FrameBuf, ChunksBuf, NChunksBuf,
    // DecompressedSizeBuf, SubChunkCapBuf, BlockStartBuf, BlockSizeBuf,
    // StatusBuf), push_constant_size=8 (frame_size, max_chunks).
    var params = [_]?*anyopaque{
        @ptrCast(&k_frame),       @ptrCast(&k_chunks),
        @ptrCast(&k_meta_n),      @ptrCast(&k_meta_decomp),
        @ptrCast(&k_meta_sccap),  @ptrCast(&k_meta_bstart),
        @ptrCast(&k_meta_bsize),  @ptrCast(&k_meta_status),
        @ptrCast(&k_size),        @ptrCast(&k_max),
    };
    var extra = [_]?*anyopaque{null};
    const t_walk = decode_context.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzWalkFrameKernel", stream);
    try vkCall(launch(module_loader.walk_frame_fn, 1, 1, 1, 1, 1, 1, 0, stream, &params, &extra), .launch);
    decode_context.endKernelTiming(t_walk, stream);

    return .{
        .d_chunk_descs = self.d_walk_chunks,
        .d_meta = self.d_walk_meta,
    };
}

/// CUDA reference: src/decode/scan_gpu.zig:90-129. Per-chunk prefix sum
/// of sub-chunk counts. Required by every GPU decode path — there is no
/// host fallback. Returns specific GpuError variants so the caller can
/// surface out-of-memory / launch-failure distinctly.
pub fn gpuPrefixSumChunksImpl(
    self: *DecodeContext,
    d_chunk_descs: u64,
    n_chunks: u32,
    sub_chunk_cap: u32,
) GpuError!descriptors.PrefixSumResultDev {
    if (!module_loader.init()) return error.BackendNotAvailable;
    if (module_loader.prefix_sum_chunks_fn == 0) return error.KernelMissing;
    const launch = vk.procs.launch_kernel orelse return error.BackendNotAvailable;

    const first_bytes: usize = @as(usize, descriptors.WALK_MAX_CHUNKS) * 4;
    try ensureDeviceBuf(&self.d_first_sub_idx_persist, &self.d_first_sub_idx_persist_size, first_bytes);
    try ensureDeviceBuf(&self.d_total_subchunks_buf, &self.d_total_subchunks_buf_size, 4);

    // Launch on caller's stream (= 0 in the sync wrapper, caller's
    // stream in the async wrapper). No internal sync: downstream
    // consumers are kernels queued on the same stream that serialize
    // naturally; the result device pointers
    // (`d_first_sub_idx_persist`, `d_total_subchunks_buf`) are read by
    // kernels, never by host.
    const stream = self.work_stream;
    var k_chunks: u64 = d_chunk_descs;
    var k_first: u64 = self.d_first_sub_idx_persist;
    var k_total: u64 = self.d_total_subchunks_buf;
    var k_n: u32 = n_chunks;
    var k_cap: u32 = sub_chunk_cap;
    // params layout per procs.launch_kernel contract: the first
    // n_bindings entries are pointers to VkDeviceBuffer handles (the
    // shader's std430 buffer bindings 0..n_bindings-1), the remaining
    // entries are pointers to push-constant scalars packed into the
    // push_constant_size byte buffer in declaration order.
    // prefix_sum_chunks: n_bindings=3 (ChunksBuf, FirstSubIdxBuf,
    // TotalSubchunksBuf), push_constant_size=8 (n_chunks, sub_chunk_cap).
    var params = [_]?*anyopaque{
        @ptrCast(&k_chunks), @ptrCast(&k_first), @ptrCast(&k_total),
        @ptrCast(&k_n),      @ptrCast(&k_cap),
    };
    var extra = [_]?*anyopaque{null};
    const t_prefix = decode_context.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzPrefixSumChunksKernel", stream);
    try vkCall(launch(module_loader.prefix_sum_chunks_fn, 1, 1, 1, 1, 1, 1, 0, stream, &params, &extra), .launch);
    decode_context.endKernelTiming(t_prefix, stream);

    return .{
        .d_first_sub_idx = self.d_first_sub_idx_persist,
        .d_total_subchunks = self.d_total_subchunks_buf,
    };
}

/// CUDA reference: src/decode/scan_gpu.zig:131-275. Sizes the staged
/// descriptor buffer ([lit][tok][hi][lo] ScanHuffDesc + [raw_hi][raw_lo]
/// ScanRawDesc), memsets it, stages the host `n_chunks` to a device
/// scratch slot via H2D, launches `slzScanParseKernel` with
/// grid=(n+255)/256, then drives 4× `slzCompactHuffDescsKernel` (one per
/// stream type: lit/tok/hi/lo) + 1× `slzCompactRawDescsKernel`. Returns
/// per-stream upper-bound counts as ScanResult; the real counts stay
/// device-resident in `d_compact_counts` and are read by downstream
/// kernels at launch time.
///
/// VK adaptation: CUDA returns `?ScanResult` and the caller flattens to
/// `error.BackendNotAvailable`; the VK port already widened the
/// signature to `GpuError!ScanResult` (matching `gpuPrefixSumChunksImpl`)
/// so failures surface distinctly (alloc / launch / copy). All return
/// statements use `error.*` instead of `null` accordingly.
pub fn gpuScanChunks(
    self: *DecodeContext,
    chunk_descs: []const descriptors.ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    total_subchunks: u32,
) GpuError!descriptors.ScanResult {
    if (!module_loader.init()) return error.BackendNotAvailable;
    if (module_loader.scan_parse_fn == 0) return error.KernelMissing;
    if (module_loader.compact_huff_descs_fn == 0 or module_loader.compact_raw_descs_fn == 0)
        return error.KernelMissing;
    if (module_loader.merge_huff_descs_fn == 0) return error.KernelMissing;

    const n: u32 = @intCast(chunk_descs.len);
    // CUDA returns null when n==0 or total_subchunks==0; mirror with
    // BadMode (the dispatcher pre-filters these — defensive).
    if (n == 0 or total_subchunks == 0) return error.BadMode;

    const h2d = vk.procs.h2d orelse return error.BackendNotAvailable;
    const launch = vk.procs.launch_kernel orelse return error.BackendNotAvailable;
    const memset = vk.procs.memset_d8 orelse return error.BackendNotAvailable;
    const stream = self.work_stream;

    // CUDA reference: src/decode/scan_gpu.zig:148-155. Staged buffer:
    // [lit][tok][hi][lo] ScanHuffDesc, then [raw_hi][raw_lo] ScanRawDesc
    // — one entry per global sub-chunk index per stream type.
    const huff_arr_bytes: usize = @as(usize, total_subchunks) * @sizeOf(descriptors.ScanHuffDesc);
    const raw_arr_bytes: usize = @as(usize, total_subchunks) * @sizeOf(descriptors.ScanRawDesc);
    const staged_bytes: usize = huff_arr_bytes * 4 + raw_arr_bytes * 2;
    try ensureDeviceBuf(&self.d_scan_staged, &self.d_scan_staged_size, staged_bytes);
    // Zero so sub-chunk slots that no thread reaches keep valid=0.
    try vkCall(memset(self.d_scan_staged, 0, staged_bytes), .copy);

    // CUDA reference: src/decode/scan_gpu.zig:157-168. Scan kernel self-
    // gates on `*d_n_chunks` — stage host `n` into a device-resident 4 B
    // slot via a sync H2D. VK h2d (synchronous) is the sync analogue of
    // CUDA cuMemcpyHtoD; we keep it sync (matching CUDA's pre-launch
    // pattern) since later kernels on the same stream pick up the value
    // by stream ordering.
    try ensureDeviceBuf(&self.d_n_chunks_scratch, &self.d_n_chunks_scratch_size, 4);
    var host_n_chunks: u32 = n;
    try vkCall(h2d(self.d_n_chunks_scratch, @ptrCast(&host_n_chunks), 4), .copy);

    // VK adaptation: params[] layout per module_loader KERNEL_DECLS
    // contract — first n_bindings entries are pointers to VkDeviceBuffer
    // handles populating descriptor bindings 0..n_bindings-1; remaining
    // entries pack into push_constant_size bytes in declaration order.
    // scan_parse_kernel.comp: n_bindings=10 (BlockBuf, ChunksBuf,
    // FirstSubIdxBuf, NChunksBuf, StLitBuf, StTokBuf, StHiBuf, StLoBuf,
    // StRawHiBuf, StRawLoBuf), push_constant_size=8 (block_len,
    // sub_chunk_cap as 2× u32). CUDA arg order: block, blen, chunks,
    // first, n, cap, lit, tok, hi, lo, rhi, rlo.
    var k_block: u64 = self.d_comp_persist;
    var k_chunks: u64 = self.d_descs_persist;
    var k_first: u64 = self.d_first_sub_idx_persist;
    var k_n: u64 = self.d_n_chunks_scratch;
    const base: u64 = self.d_scan_staged;
    var k_lit: u64 = base;
    var k_tok: u64 = base + huff_arr_bytes;
    var k_hi: u64 = base + huff_arr_bytes * 2;
    var k_lo: u64 = base + huff_arr_bytes * 3;
    var k_rhi: u64 = base + huff_arr_bytes * 4;
    var k_rlo: u64 = base + huff_arr_bytes * 4 + raw_arr_bytes;
    var k_blen: u32 = @intCast(compressed_block.len);
    var k_cap: u32 = sub_chunk_cap;
    var params = [_]?*anyopaque{
        @ptrCast(&k_block), @ptrCast(&k_chunks),
        @ptrCast(&k_first), @ptrCast(&k_n),
        @ptrCast(&k_lit),   @ptrCast(&k_tok),
        @ptrCast(&k_hi),    @ptrCast(&k_lo),
        @ptrCast(&k_rhi),   @ptrCast(&k_rlo),
        @ptrCast(&k_blen),  @ptrCast(&k_cap),
    };
    var extra = [_]?*anyopaque{null};
    const tpb: u32 = 256;
    const blocks: u32 = (n + tpb - 1) / tpb;
    const t_scan = decode_context.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzScanParseKernel", stream);
    try vkCall(launch(module_loader.scan_parse_fn, blocks, 1, 1, tpb, 1, 1, 0, stream, &params, &extra), .launch);
    decode_context.endKernelTiming(t_scan, stream);
    // No post-scan sync: the compact kernels below queue on the same
    // stream and see scan_parse's writes to d_scan_staged via stream
    // ordering. CUDA reference: src/decode/scan_gpu.zig:187-189.

    // CUDA reference: src/decode/scan_gpu.zig:196-256. Device-side
    // compaction: 4 × compactHuffDescs (one per stream type) + 1 ×
    // compactRawDescs produce compacted per-stream arrays + per-stream
    // counts entirely on device. Nothing comes back to the host here —
    // downstream kernels (merge, huff_build, gather) read straight from
    // d_compact_*.
    {
        // CUDA reference: src/decode/scan_gpu.zig:200-212. Size compact
        // buffers. `n_huff` bound: WALK_MAX_CHUNKS * MAX_SUB_CHUNKS_PER_CHUNK
        // (worst-case sub-chunks per chunk at sc >= 1). `n_raw` bound: 2×
        // that (hi + lo planes).
        const huff_compact_max = @as(usize, descriptors.WALK_MAX_CHUNKS) * @as(usize, descriptors.MAX_SUB_CHUNKS_PER_CHUNK);
        const huff_compact_bytes = huff_compact_max * @sizeOf(descriptors.HuffDecChunkDesc);
        const raw_compact_max = huff_compact_max * 2;
        const raw_compact_bytes = raw_compact_max * @sizeOf(descriptors.RawOff16Desc);
        try ensureDeviceBuf(&self.d_compact_lit, &self.d_compact_lit_size, huff_compact_bytes);
        try ensureDeviceBuf(&self.d_compact_tok, &self.d_compact_tok_size, huff_compact_bytes);
        try ensureDeviceBuf(&self.d_compact_hi, &self.d_compact_hi_size, huff_compact_bytes);
        try ensureDeviceBuf(&self.d_compact_lo, &self.d_compact_lo_size, huff_compact_bytes);
        try ensureDeviceBuf(&self.d_compact_raw, &self.d_compact_raw_size, raw_compact_bytes);
        // 6 u32: [n_lit, n_tok, n_hi, n_lo, n_raw, n_merged]. n_merged
        // is written by slzMergeHuffDescsKernel in step 6c. CUDA reference:
        // src/decode/scan_gpu.zig:209-212.
        try ensureDeviceBuf(&self.d_compact_counts, &self.d_compact_counts_size, 6 * 4);
        try vkCall(memset(self.d_compact_counts, 0, 6 * 4), .copy);

        // CUDA reference: src/decode/scan_gpu.zig:214-239. 4 launches of
        // slzCompactHuffDescsKernel (one per stream type) — same kernel,
        // different bindings (staged source / dst / count slot).
        // compact_huff_descs_kernel.comp: n_bindings=4 (StagedBuf,
        // TotalSubsBuf, OutBuf, NOutBuf), push_constant_size=0.
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
                @ptrCast(&k_dst),    @ptrCast(&k_count),
            };
            var c_extra = [_]?*anyopaque{null};
            const t_ch = decode_context.beginKernelTiming(self.enable_profiling, &self.pending_timings, compact_names[ci], stream);
            try vkCall(launch(module_loader.compact_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, stream, &c_params, &c_extra), .launch);
            decode_context.endKernelTiming(t_ch, stream);
        }
        // CUDA reference: src/decode/scan_gpu.zig:241-255. Raw compact —
        // 1 launch of slzCompactRawDescsKernel reading hi+lo staged
        // arrays into d_compact_raw, count at offset 16 (= 4 u32 slots
        // for the four Huffman streams).
        // compact_raw_descs_kernel.comp: n_bindings=5 (StagedHiBuf,
        // StagedLoBuf, TotalSubsBuf, OutBuf, NOutBuf), push_constant_size=0.
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
            const t_cr = decode_context.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzCompactRawDescsKernel", stream);
            try vkCall(launch(module_loader.compact_raw_descs_fn, 1, 1, 1, 1, 1, 1, 0, stream, &cr_params, &cr_extra), .launch);
            decode_context.endKernelTiming(t_cr, stream);
        }
    }

    // CUDA reference: src/decode/scan_gpu.zig:259-274. Counts stay
    // device-resident in d_compact_counts; downstream merge, huff_build,
    // huff_decode, and gather kernels read them as device pointers and
    // self-gate. The ScanResult reports worst-case bounds (= total_subchunks
    // per Huffman stream type + 2× for the two raw streams) so the
    // dispatch's allocator sizes scratch correctly; the actual non-zero
    // work happens at kernel time.
    return .{
        .num_raw_off16    = total_subchunks * 2,
        .num_huff_lit     = total_subchunks,
        .num_huff_tok     = total_subchunks,
        .num_huff_off16hi = total_subchunks,
        .num_huff_off16lo = total_subchunks,
    };
}
