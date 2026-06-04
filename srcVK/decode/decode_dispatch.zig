//! 1:1 port of src/decode/decode_dispatch.zig.
//!
//! Top-level decode dispatcher. Owns the L2 gate (per project
//! EXCEPTION 2): for `req.level == 1` decodes the dispatcher must skip
//! the Huff/scan/compact/merge/gather paths and run the L1 raw
//! `slzLzDecodeRawKernel` directly off the host-computed chunk descs +
//! prefix sum.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const vk = @import("vulkan_api.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const module_loader = @import("module_loader.zig");
const scan_gpu = @import("scan_gpu.zig");

const DecodeContext = decode_context.DecodeContext;
const ChunkDesc = descriptors.ChunkDesc;
const GpuError = descriptors.GpuError;

/// CUDA reference: src/decode/decode_dispatch.zig:47-65. Renamed from
/// CudaProcs per Section B. Bundle of the function-pointer slots the
/// pipeline uses, resolved once at fullGpuLaunchImpl entry.
pub const VkProcs = struct {
    h2d: ?*const anyopaque = null,
    d2h: ?*const anyopaque = null,
    launch: ?*const anyopaque = null,
    sync: ?*const anyopaque = null,
    stream_sync: ?*const anyopaque = null,
    d2d: ?*const anyopaque = null,
};

/// CUDA reference: src/decode/decode_dispatch.zig:254-290. Per-call
/// request bundle. The VK port adds a `level` field so the L2 gate fires
/// per request (audit Section C.5.1 commentary).
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
    d_chunk_descs_override: ?u64 = null,
    d_n_chunks_dev: ?u64 = null,
    /// VK port addition (audit Section C.5.1): explicit level so the L2
    /// gate inside fullGpuLaunchImpl fires per request rather than per
    /// loaded-module-set.
    level: u8 = 1,

    pub fn writesDirectlyToTarget(self: DecodeRequest) bool {
        return self.d_output_target != null and self.dst_start_off == 0;
    }
};

/// CUDA reference: src/decode/decode_dispatch.zig:105-140. Launches
/// slzMergeHuffDescsKernel. L2 stub.
pub fn mergeHuffDescs(
    self: *DecodeContext,
    tok_offset: usize,
    off16_offset: usize,
    launch_fn: anytype,
) GpuError!void {
    _ = self;
    _ = tok_offset;
    _ = off16_offset;
    _ = launch_fn;
    return error.NotImplementedL2;
}

/// CUDA reference: src/decode/decode_dispatch.zig:148-215. Launches
/// slzGatherRawOff16Kernel. L2 stub.
pub fn gatherRawOff16(self: *DecodeContext) GpuError!void {
    _ = self;
    return error.NotImplementedL2;
}

/// CUDA reference: src/decode/decode_dispatch.zig:216-253. SLZ_E2E_TIMER
/// trace emitter.
pub fn emitE2eTrace(self: *DecodeContext) void {
    _ = self;
}

/// CUDA reference: src/decode/decode_dispatch.zig:300-365. Launches the
/// Huffman LUT-build + 4-stream decode kernels. L2 stub.
pub fn runHuffBuildAndDecode(
    self: *DecodeContext,
    procs: *const VkProcs,
    n_huff: u32,
) GpuError!void {
    _ = self;
    _ = procs;
    _ = n_huff;
    return error.NotImplementedL2;
}

/// CUDA reference: src/decode/decode_dispatch.zig:366-478. Launches the
/// LZ-decode kernel (raw on L1, general on L2+). L1 hot path.
pub fn runLzPipeline(
    self: *DecodeContext,
    procs: *const VkProcs,
    req: DecodeRequest,
    heavy_stream: usize,
) GpuError!void {
    _ = self;
    _ = procs;
    _ = req;
    _ = heavy_stream;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/decode_dispatch.zig:479-502. Final
/// D2D / D2H stage of the dispatch.
pub fn finalizeOutput(
    self: *DecodeContext,
    procs: *const VkProcs,
    req: DecodeRequest,
    heavy_stream: usize,
) GpuError!void {
    _ = self;
    _ = procs;
    _ = req;
    _ = heavy_stream;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/decode_dispatch.zig:503-585. H2D-uploads
/// the compressed block + the chunk-desc array and launches the
/// prefix-sum kernel. L1 hot path.
pub fn uploadInputAndPrefixSum(
    self: *DecodeContext,
    procs: *const VkProcs,
    req: DecodeRequest,
) GpuError!void {
    _ = self;
    _ = procs;
    _ = req;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/decode_dispatch.zig:586-687. Heavy-half
/// kernel orchestrator (Huff build/decode + LZ decode + finalize).
pub fn runBackHalf(
    self: *DecodeContext,
    procs: *const VkProcs,
    req: DecodeRequest,
    heavy_stream: usize,
) GpuError!void {
    _ = self;
    _ = procs;
    _ = req;
    _ = heavy_stream;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/decode_dispatch.zig:688-end. Top-level
/// dispatch entry. L2 gate (audit EXCEPTION 2): for req.level == 1 skip
/// the Huff/scan/compact/merge/gather paths entirely and run the raw
/// LZ-decode kernel off the host-built chunk descs + prefix sum.
pub fn fullGpuLaunchImpl(self: *DecodeContext, req: DecodeRequest) GpuError!void {
    _ = self;
    _ = req;
    return error.NotYetPorted;
}
