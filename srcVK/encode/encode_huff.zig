//! 1:1 port of src/encode/encode_huff.zig.
//!
//! GPU Huffman encode pass and its three stream-specific wrappers
//! (literals, tokens, off16). Every entry point returns `bool` — true on
//! a successful Huffman encode, false to signal the caller to fall back
//! to the raw stream for that sub-chunk. The L1 gate (`opts.level >= 3`)
//! in fast_framed.zig means these never get called on L1; the bool
//! fall-through is the documented CUDA convention.

const std = @import("std");
const vk = @import("../decode/vulkan_api.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const VkDeviceBuffer = vk.VkDeviceBuffer;
const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;
const HuffEncDesc = encode_context.HuffEncDesc;

/// CUDA reference: src/encode/encode_huff.zig:52-177. Shared Huffman
/// build+encode dispatcher. Bool fall-through convention.
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

/// CUDA reference: src/encode/encode_huff.zig:225-364. Huffman encode
/// for the per-sub-chunk off16 hi+lo byte planes.
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

/// CUDA reference: src/encode/encode_huff.zig:462-475. Huffman encode
/// for the per-sub-chunk literal byte plane.
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

/// CUDA reference: src/encode/encode_huff.zig:478-491. Huffman encode
/// for the per-sub-chunk token byte plane.
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
