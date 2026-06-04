//! 1:1 port of src/encode/encode_huff.zig.
//!
//! L2 stub: GPU Huffman encode (L3+ only on the codec level scale).
//! Symbols MUST exist because fast_framed.zig and driver.zig reference
//! them; bodies return false on L1 so the `opts.level >= 3` gate in
//! fast_framed.zig naturally skips emitted Huffman bodies.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const vulkan_ffi = @import("vulkan_ffi.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;
const HuffEncDesc = encode_context.HuffEncDesc;
const VkDeviceBuffer = vulkan_ffi.VkDeviceBuffer;

/// CUDA reference: src/encode/encode_huff.zig:52-224. L2 stub: shared
/// Huffman build+encode dispatcher.
pub fn gpuEncodeHuffImpl(
    self: *EncodeContext,
    descs: []const HuffEncDesc,
    out_sizes: []u32,
    out_dev: VkDeviceBuffer,
    profile_names: [2][*:0]const u8,
) bool {
    _ = self;
    _ = descs;
    _ = out_sizes;
    _ = out_dev;
    _ = profile_names;
    return false;
}

/// CUDA reference: src/encode/encode_huff.zig:225-461. L2 stub: Huffman
/// encode for the per-sub-chunk off16 byte plane (hi + lo).
pub fn gpuEncodeOff16HuffImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    _ = self;
    _ = allocator;
    _ = output;
    _ = chunk_descs;
    _ = comp_sizes;
    return false;
}

/// CUDA reference: src/encode/encode_huff.zig:462-475. L2 stub: Huffman
/// encode for the per-sub-chunk literal byte plane.
pub fn gpuEncodeLiteralsHuffImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    _ = self;
    _ = allocator;
    _ = output;
    _ = chunk_descs;
    _ = comp_sizes;
    return false;
}

/// CUDA reference: src/encode/encode_huff.zig:478-end. L2 stub: Huffman
/// encode for the per-sub-chunk token byte plane.
pub fn gpuEncodeTokensHuffImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    _ = self;
    _ = allocator;
    _ = output;
    _ = chunk_descs;
    _ = comp_sizes;
    return false;
}
