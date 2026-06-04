//! 1:1 port of src/encode/fast_framed.zig.
//!
//! L1 encode orchestrator. Entry point compressFramedOne() called by
//! srcVK/encode/streamlz_encoder.zig::compressFramedWithIo. Owns the
//! `opts.level >= 3` gate that funnels L3-L5 frames into the Huffman
//! pipeline; on L1/L2 the gate skips Huffman and emits raw streams via
//! the LZ-encode + assemble-measure + assemble-write + frame-assemble
//! kernel chain.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");

const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const lz_constants = @import("../format/streamlz_constants.zig");

const encoder = @import("streamlz_encoder.zig");
const Options = encoder.Options;
const CompressError = encoder.CompressError;

const gpu_enc = @import("driver.zig");
const vk_api = @import("../decode/vulkan_api.zig");

/// CUDA reference: src/encode/fast_framed.zig:91-end. Compress one frame
/// from src into dst. Returns the number of bytes written.
pub fn compressFramedOne(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_enc.EncodeContext,
) CompressError!usize {
    _ = allocator;
    _ = io;
    _ = src;
    _ = dst;
    _ = opts;
    _ = enc_ctx;
    return error.NotYetPorted;
}
