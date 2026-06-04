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

/// CUDA reference: src/decode/scan_gpu.zig:37-82. L2 stub.
pub fn gpuWalkFrameImpl(
    self: *DecodeContext,
    d_frame: u64,
    frame_size: u32,
) GpuError!descriptors.WalkFrameResultDev {
    _ = self;
    _ = d_frame;
    _ = frame_size;
    return error.NotImplementedL2;
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

/// CUDA reference: src/decode/scan_gpu.zig:131-275. L2 stub.
pub fn gpuScanChunks(
    self: *DecodeContext,
    chunk_descs: []const descriptors.ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    total_subchunks: u32,
) GpuError!descriptors.ScanResult {
    _ = self;
    _ = chunk_descs;
    _ = compressed_block;
    _ = sub_chunk_cap;
    _ = total_subchunks;
    return error.NotImplementedL2;
}
