//! 1:1 port of src/encode/encode_assemble.zig.
//!
//! L1 hot frame-assembly orchestrator. gpuAssembleFrameImpl runs the
//! per-sub-chunk measure + write passes (assemble_measure_kernel.comp +
//! assemble_write_kernel.comp); gpuFrameAssembleImpl drives the final
//! device-resident frame writer (frame_assemble_kernel.comp).
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const vulkan_ffi = @import("vulkan_ffi.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;

/// CUDA reference: src/encode/encode_assemble.zig:33-185. Two-pass
/// per-sub-chunk assembly (measure + write). L1 hot path.
pub fn gpuAssembleFrameImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    _ = self;
    _ = allocator;
    _ = chunk_descs;
    _ = comp_sizes;
    return false;
}

/// CUDA reference: src/encode/encode_assemble.zig:186-190. Frame-prefix
/// bytes the frame-assemble kernel splices into its output.
pub const FramePreamble = struct {
    prefix_bytes: []const u8,
    internal_hdr0: u8,
    internal_hdr1: u8,
};

/// CUDA reference: src/encode/encode_assemble.zig:196-202. Layout
/// description for the per-chunk asm payloads.
pub const ChunkLayout = struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    per_chunk_asm_off: []const u32,
    per_chunk_asm_size: []const u32,
};

/// CUDA reference: src/encode/encode_assemble.zig:204-end. Final
/// device-resident frame writer. L1 hot path.
pub fn gpuFrameAssembleImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    layout: ChunkLayout,
    preamble: FramePreamble,
    d_input_dev: u64,
    d_output: u64,
) ?u32 {
    _ = self;
    _ = allocator;
    _ = layout;
    _ = preamble;
    _ = d_input_dev;
    _ = d_output;
    return null;
}
