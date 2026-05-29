//! Host-side entropy wrappers used by the GPU orchestration path.
//!
//! Three primitives, in order of preference at each call site:
//!
//!   * `encodeArrayU8Huffman` — try a chunk-type=4 Huffman block via the
//!     canonical-Huffman encoder; fall back to memcpy when Huffman does
//!     not beat raw. Used for off16 hi/lo planes when the GPU Huffman
//!     pass did not produce a body that beats raw.
//!   * `encodeArrayU8Memcpy`  — raw bytes wrapped in a chunk-type=0
//!     3-byte BE header. Always succeeds when dst has room.
//!   * `writeNonCompactChunkHeader` — splice a 5-byte non-compact chunk
//!     header in front of a pre-computed body (used to wrap the GPU
//!     Huffman bodies that `gpu_stream_assembly` receives from the
//!     encode-side Huffman kernel).
//!
//! The returned byte count includes the chunk header.

const std = @import("std");
const huffman = @import("huffman_encoder.zig");

pub const EncodeError = error{
    DestinationTooSmall,
} || std.mem.Allocator.Error;

/// Write a 3-byte BE chunk-type=0 header followed by the raw bytes.
/// Returns the total bytes written.
pub fn encodeArrayU8Memcpy(dst: []u8, src: []const u8) EncodeError!usize {
    if (src.len + 3 > dst.len) return error.DestinationTooSmall;
    dst[0] = @intCast((src.len >> 16) & 0xFF);
    dst[1] = @intCast((src.len >> 8) & 0xFF);
    dst[2] = @intCast(src.len & 0xFF);
    @memcpy(dst[3 .. 3 + src.len], src);
    return src.len + 3;
}

/// Write a 5-byte non-compact chunk header. `decompressed_size` and
/// `compressed_size` must each fit in 18 bits; `chunk_type` must fit in 4
/// bits. Layout (`dm1` = `decompressed_size - 1`):
///
///   byte 0:           [chunk_type:4 | dm1_high4:4]
///   bytes 1..4 (BE):  [dm1_low14:14 | comp_size:18]
pub fn writeNonCompactChunkHeader(dst: []u8, chunk_type: u8, compressed_size: u32, decompressed_size: u32) void {
    const dst_minus_1: u32 = decompressed_size - 1;
    dst[0] = @intCast((@as(u32, chunk_type) << 4) | ((dst_minus_1 >> 14) & 0xF));
    const bits: u32 = compressed_size | ((dst_minus_1 & 0x3FFF) << 18);
    dst[1] = @intCast((bits >> 24) & 0xFF);
    dst[2] = @intCast((bits >> 16) & 0xFF);
    dst[3] = @intCast((bits >> 8) & 0xFF);
    dst[4] = @intCast(bits & 0xFF);
}

/// Encode `src` as a Huffman chunk-type=4 block when that beats a raw
/// memcpy chunk; otherwise emit a memcpy chunk. Used for off16 hi/lo
/// planes when the GPU Huffman pass did not produce a body for this
/// sub-chunk.
pub fn encodeArrayU8Huffman(dst: []u8, src: []const u8) EncodeError!usize {
    const min_huffman_dst = 5 + 128 + 9;
    if (src.len >= 32 and dst.len > min_huffman_dst) {
        if (huffman.encodeBlock(dst, src) catch null) |n| {
            if (n < src.len + 3) return n;
        }
    }
    return encodeArrayU8Memcpy(dst, src);
}
