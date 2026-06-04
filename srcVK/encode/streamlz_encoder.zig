//! 1:1 port of src/encode/streamlz_encoder.zig.
//!
//! Public encoder facade: compressBound + compressFramed +
//! compressFramedWithIo. Delegates to fast_framed.compressFramedOne for
//! the actual GPU dispatch.

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
    const chunk_count: usize = (src_len + lz_constants.chunk_size - 1) / lz_constants.chunk_size;
    const sub_chunks: usize = (src_len + lz_constants.sub_chunk_size - 1) / lz_constants.sub_chunk_size;
    const per_sub_chunk_overhead: usize = 3 + 8 + 3 + 256;
    const sc_prefix_upper_bound: usize = chunk_count * 8;
    return frame.max_header_size + 4
        + chunk_count * (8 + 2 + 4)
        + sub_chunks * per_sub_chunk_overhead
        + src_len
        + 64
        + sc_prefix_upper_bound;
}

/// CUDA reference: src/encode/streamlz_encoder.zig:97-105. Host->host
/// frame compress. Bytes-written variant.
pub fn compressFramed(
    allocator: std.mem.Allocator,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    return compressFramedWithIo(allocator, std.Io.failing, src, dst, opts, enc_ctx);
}

/// CUDA reference: src/encode/streamlz_encoder.zig:112-123. Same as
/// compressFramed plus a std.Io for telemetry.
pub fn compressFramedWithIo(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    dst: []u8,
    opts: Options,
    enc_ctx: *gpu_encoder.EncodeContext,
) CompressError!usize {
    if (opts.level < 1 or opts.level > 5) return error.BadLevel;
    if (dst.len < compressBound(src.len)) return error.DestinationTooSmall;
    return fast_framed.compressFramedOne(allocator, io, src, dst, opts, enc_ctx);
}
