//! 1:1 port of src/encode/streamlz_encoder.zig.
//!
//! Public encoder facade: compressBound + compressFramed +
//! compressFramedWithIo. Delegates to fast_framed.compressFramedOne for
//! the actual GPU dispatch.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const frame = @import("../format/frame_format.zig");
const lz_constants = @import("../format/streamlz_constants.zig");
const fast_framed = @import("fast_framed.zig");
const gpu_encoder = @import("driver.zig");

/// CUDA reference: src/encode/streamlz_encoder.zig:24-41. Encoder error
/// set.
pub const CompressError = error{
    BadLevel,
    BadBlockSize,
    BadScGroupSize,
    DestinationTooSmall,
    NotYetPorted,
} || std.mem.Allocator.Error;

/// CUDA reference: src/encode/streamlz_encoder.zig:43-64. Encoder
/// options.
pub const Options = struct {
    level: u8 = 1,
    include_content_size: bool = true,
    block_size: u32 = lz_constants.chunk_size,
    sc_group_size_override: ?f32 = null,
};

/// CUDA reference: src/encode/streamlz_encoder.zig:70-84. Upper bound on
/// the compressed-output size for a given input length.
pub fn compressBound(src_len: usize) usize {
    _ = src_len;
    return 0;
}

/// CUDA reference: src/encode/streamlz_encoder.zig:97-105. Host→host
/// frame compress. Bytes-written variant.
pub fn compressFramed(
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    _ = allocator;
    _ = src;
    _ = dst;
    _ = opts;
    _ = enc_ctx;
    return error.NotYetPorted;
}

/// CUDA reference: src/encode/streamlz_encoder.zig:112-end. Same as
/// compressFramed plus a std.Io for telemetry.
pub fn compressFramedWithIo(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    _ = allocator;
    _ = io;
    _ = src;
    _ = dst;
    _ = opts;
    _ = enc_ctx;
    return error.NotYetPorted;
}
