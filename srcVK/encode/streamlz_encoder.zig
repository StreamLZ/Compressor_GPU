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
    /// v4 #16 (CUDA-mirror): `opts.dictionary_id` does not resolve in
    /// the dictionary registry (`srcVK/dict/dictionary.zig`).
    UnknownDictionary,
} || std.mem.Allocator.Error;

/// CUDA reference: src/encode/streamlz_encoder.zig:43-64. Encoder
/// options.
pub const Options = struct {
    level: u8 = 1,
    include_content_size: bool = true,
    block_size: u32 = lz_constants.chunk_size,
    sc_group_size_override: ?f32 = null,
    /// v4 #19 (CUDA-mirror): chunk-Merkle checksum, default ON.
    chunk_checksum: bool = true,
    /// v4 #16 (CUDA-mirror): preset-dictionary ID, flag bit 3 + 4 ID
    /// bytes in the frame header. Phase 1: wire surface only - the
    /// match finder does not search the dictionary yet.
    dictionary_id: ?u32 = null,
    /// v4 #20 (CUDA-mirror): emit the chunk-size table footer (flag
    /// bit 6) - 3 bytes per chunk after the end mark, before the
    /// trailers. Lets the device-resident decode path locate every
    /// chunk with one parallel read instead of the serial chunk-chain
    /// walk. Decoders that predate the bit ignore the footer. Not
    /// emitted on uncompressed-body frames or when the effective chunk
    /// size is under 64 KB.
    chunk_size_table: bool = false,
};

/// CUDA reference: src/encode/streamlz_encoder.zig:70-84. Upper bound on
/// the compressed-output size for a given input length.
pub fn compressBound(src_len: usize) usize {
    const chunk_count: usize = (src_len + lz_constants.chunk_size - 1) / lz_constants.chunk_size;
    const sub_chunks: usize = (src_len + lz_constants.sub_chunk_size - 1) / lz_constants.sub_chunk_size;
    const per_sub_chunk_overhead: usize = 3 + 8 + 3 + 256;
    const sc_prefix_upper_bound: usize = chunk_count * 8;
    // v4 #20 chunk-size table: 3 B per effective chunk. The table is
    // only emitted for eff_chunk >= 64 KB, so ceil(len/64K) bounds the
    // entry count at every legal sc setting.
    const chunk_table_upper_bound: usize = (src_len / 65536 + 2) * 3;
    return frame.max_header_size + 4 + 4 // (+4: v4 #19 merkle trailer)
        + chunk_count * (8 + 2 + 4)
        + sub_chunks * per_sub_chunk_overhead
        + src_len
        + 64
        + sc_prefix_upper_bound
        + chunk_table_upper_bound;
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
    if (opts.dictionary_id) |did| {
        const dict = @import("../dict/dictionary.zig");
        if (dict.resolve(enc_ctx.registered_dicts.items, did) == null) return error.UnknownDictionary;
    }
    var frame_len = try fast_framed.compressFramedOne(allocator, io, src, dst, opts, enc_ctx);
    // v4 #19 (CUDA-mirror): chunk-Merkle trailer after the end mark.
    // d_input_override != 0 means the caller's data is DEVICE-resident
    // and the host `src` slice is a length-only sentinel (see
    // EncodeContext.d_input_override) - it MUST NOT be dereferenced.
    // v1 therefore skips the Merkle trailer on the D2D path (flag
    // stays clear; decoders handle both). v2: hash on device via the
    // slzChunkHashKernel already staged in the PTX modules.
    if (opts.chunk_checksum and enc_ctx.d_input_override == 0) {
        if (frame_len + 4 > dst.len) return error.DestinationTooSmall;
        const xxh = @import("../format/xxhash32.zig");
        const eff = fast_framed.effChunkFor(src.len, opts.sc_group_size_override);
        const root = xxh.chunkMerkleRoot(allocator, src, eff) catch return error.OutOfMemory;
        std.mem.writeInt(u32, dst[frame_len..][0..4], root, .little);
        frame_len += 4;
    }
    return frame_len;
}
