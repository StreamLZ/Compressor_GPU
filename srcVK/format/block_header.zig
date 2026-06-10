//! 1:1 port of src/format/block_header.zig.
//!
//! Internal 2-byte StreamLZ block header + 4-byte chunk header parsers.
//! Pure host code, no CUDA tokens, no device dependencies.
//!

const std = @import("std");
const constants = @import("streamlz_constants.zig");

/// CUDA reference: src/format/block_header.zig:33-38. Decoder-type field
/// of the block header. The open `_` member is required because the wire
/// field is a `u8` (any non-listed value parses as
/// `error.BadDecoderType`).
pub const CodecType = enum(u8) {
    high = 0,
    fast = 1,
    turbo = 2,
    _,
};

/// CUDA reference: src/format/block_header.zig:40-65. Parsed 2-byte
/// block header.
pub const BlockHeader = struct {
    decoder_type: CodecType,
    restart_decoder: bool,
    uncompressed: bool,
    use_checksums: bool,
    self_contained: bool,
    two_phase: bool,

    pub const size: usize = 2;
};

/// CUDA reference: src/format/block_header.zig:67-80. Parsed 4-byte (or
/// 7-byte with checksum) chunk header.
pub const ChunkHeader = struct {
    compressed_size: u32,
    checksum: u32,
    whole_match_distance: u32,
    bytes_consumed: usize,
    is_memset: bool,
    memset_fill: u8,
};

/// CUDA reference: src/format/block_header.zig:82-97. Parse failure
/// enumeration.
pub const ParseError = error{
    TooShort,
    BadMagic,
    BadDecoderType,
    BadChunkType,
};

/// CUDA reference: src/format/block_header.zig:99-120. Parse a 2-byte
/// block header from src.
pub fn parseBlockHeader(src: []const u8) ParseError!BlockHeader {
    if (src.len < BlockHeader.size) return error.TooShort;
    const b0 = src[0];
    const b1 = src[1];
    if ((b0 & 0x0F) != 0x5) return error.BadMagic;

    const decoder_byte: u8 = b1 & 0x7F;
    const decoder_type: CodecType = @enumFromInt(decoder_byte);
    switch (decoder_type) {
        .high, .fast, .turbo => {},
        else => return error.BadDecoderType,
    }

    return .{
        .decoder_type = decoder_type,
        .two_phase = ((b0 >> 5) & 1) != 0,
        .self_contained = ((b0 >> 4) & 1) != 0,
        .restart_decoder = ((b0 >> 6) & 1) != 0,
        .uncompressed = ((b0 >> 7) & 1) != 0,
        .use_checksums = (b1 >> 7) != 0,
    };
}

/// CUDA reference: src/format/block_header.zig:122-166. Parse a 4-byte
/// (or 7-byte) chunk header from src.
pub fn parseChunkHeader(src: []const u8, use_checksum: bool) ParseError!ChunkHeader {
    const min_bytes: usize = if (use_checksum) 7 else 4;
    if (src.len < min_bytes) return error.TooShort;

    const v = std.mem.readInt(u32, src[0..4], .little);
    const size = v & constants.chunk_size_mask;
    const chunk_type = (v >> constants.chunk_type_shift) & 3;

    switch (chunk_type) {
        0 => {
            if (use_checksum) {
                const cs: u32 = (@as(u32, src[4]) << 16) | (@as(u32, src[5]) << 8) | @as(u32, src[6]);
                return .{
                    .compressed_size = size + 1,
                    .checksum = cs,
                    .whole_match_distance = 0,
                    .bytes_consumed = 7,
                    .is_memset = false,
                    .memset_fill = 0,
                };
            }
            return .{
                .compressed_size = size + 1,
                .checksum = 0,
                .whole_match_distance = 0,
                .bytes_consumed = 4,
                .is_memset = false,
                .memset_fill = 0,
            };
        },
        1 => {
            // Memset chunk: 1 extra byte for the fill value (no checksum path).
            if (src.len < 5) return error.TooShort;
            return .{
                .compressed_size = 0,
                .checksum = src[4],
                .whole_match_distance = 0,
                .bytes_consumed = 5,
                .is_memset = true,
                .memset_fill = src[4],
            };
        },
        else => return error.BadChunkType,
    }
}
