//! Internal 2-byte StreamLZ block header + 4-byte chunk header parsers.
//!
//! Layout of the 2-byte block header (big-endian byte order):
//!   byte 0:
//!     [3:0]  magic nibble, must be 0x5
//!     [4]    self_contained
//!     [5]    two_phase
//!     [6]    restart_decoder
//!     [7]    uncompressed
//!   byte 1:
//!     [6:0]  decoder_type (0=High, 1=Fast, 2=Turbo)
//!     [7]    use_checksums
//!
//! Layout of the 4-byte LE chunk header:
//!   bits [17:0]  compressed_size - 1
//!   bits [19:18] type (0=normal, 1=memset, 2+ reserved)
//!   bits [31:20] reserved (must be 0 on write)
//!
//! When `use_checksums` is set, 3 extra bytes follow the 4-byte chunk header
//! (big-endian CRC24). The decoder currently parses but does not verify them.

const std = @import("std");
const constants = @import("streamlz_constants.zig");

/// Decoder-type field of the block header. The GPU encoder only emits
/// `.fast`; `.high` and `.turbo` are accepted by the decoder for
/// backward compatibility with frames produced by the now-deleted CPU
/// codec. The open `_` member is required because the wire field is
/// a `u8` (any non-listed value parses as `error.BadDecoderType`).
pub const CodecType = enum(u8) {
    high = 0,
    fast = 1,
    turbo = 2,
    _,
};

pub const BlockHeader = struct {
    /// Which codec produced the block. Drives kernel selection at decode.
    decoder_type: CodecType,
    /// Bit 6: when set, the decoder discards any cross-block carry
    /// state and treats the block as a fresh start. Always true for
    /// frames the GPU codec produces (every block is self-contained).
    restart_decoder: bool,
    /// Bit 7: block body is verbatim source bytes (no LZ encoding).
    /// The decoder copies `decompressed_size` bytes from the body
    /// directly into `dst`.
    uncompressed: bool,
    /// Top bit of byte 1: per-chunk CRC24 checksums are present after
    /// each 4-byte chunk header. The GPU decoder parses but does not
    /// verify them today (reserved for a future strict mode).
    use_checksums: bool,
    /// Bit 4: every sub-chunk in this block decodes independently
    /// (no cross-sub-chunk back-references). Always true for the GPU
    /// codec.
    self_contained: bool,
    /// Bit 5: a phase-1 parallel-decode sidecar block precedes this
    /// block in the frame. Legacy CPU-codec hint; never set by the
    /// GPU codec, ignored by the GPU decoder.
    two_phase: bool,

    pub const size: usize = 2;
};

pub const ChunkHeader = struct {
    /// Real compressed size (header field is stored as size-1).
    compressed_size: u32,
    /// Optional 3-byte CRC24. Zero when checksums are disabled.
    checksum: u32,
    /// Whole-chunk match-copy distance (non-zero only for special memset/wholematch).
    whole_match_distance: u32,
    /// How many bytes of `src` the header occupies (4 or 7).
    bytes_consumed: usize,
    /// True when this chunk is a memset fill (type==1 in the 4-byte header).
    is_memset: bool,
    /// For memset chunks, the fill byte (or first byte after the header).
    memset_fill: u8,
};

pub const ParseError = error{
    /// `src` was shorter than the 2-byte block header, or shorter than
    /// the 4-byte (or 7-byte with checksum) chunk header.
    TooShort,
    /// Block header's low nibble of byte 0 was not `0x5` (the SLZ
    /// internal block magic).
    BadMagic,
    /// Decoder-type field is not `High` / `Fast` / `Turbo`; the GPU
    /// codec only emits `Fast` but the decoder also accepts the other
    /// two for backward compatibility.
    BadDecoderType,
    /// Chunk-header type field is not `0` (LZ-compressed) or `1`
    /// (memset fill). Types 2+ are reserved.
    BadChunkType,
};

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


// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;
test "parseBlockHeader rejects bad magic" {
    try testing.expectError(error.BadMagic, parseBlockHeader(&[_]u8{ 0x00, 0x01 }));
}

test "parseBlockHeader accepts Fast codec" {
    // magic 0x5 in low nibble + Fast (1) in low 7 bits of byte 1
    const hdr = try parseBlockHeader(&[_]u8{ 0x05, 0x01 });
    try testing.expectEqual(CodecType.fast, hdr.decoder_type);
    try testing.expect(!hdr.uncompressed);
    try testing.expect(!hdr.self_contained);
    try testing.expect(!hdr.two_phase);
    try testing.expect(!hdr.restart_decoder);
    try testing.expect(!hdr.use_checksums);
}

test "parseBlockHeader parses all flag bits" {
    // byte 0: magic 0x5 | self_contained(4) | two_phase(5) | restart(6) | uncomp(7) = 0xF5
    // byte 1: decoder=0 (High) | use_checksums(7) = 0x80
    const hdr = try parseBlockHeader(&[_]u8{ 0xF5, 0x80 });
    try testing.expectEqual(CodecType.high, hdr.decoder_type);
    try testing.expect(hdr.uncompressed);
    try testing.expect(hdr.restart_decoder);
    try testing.expect(hdr.two_phase);
    try testing.expect(hdr.self_contained);
    try testing.expect(hdr.use_checksums);
}

test "parseBlockHeader rejects invalid decoder type" {
    // decoder = 5 is not High/Fast/Turbo
    try testing.expectError(error.BadDecoderType, parseBlockHeader(&[_]u8{ 0x05, 0x05 }));
}

test "parseChunkHeader parses normal chunk, no checksum" {
    // size field = 1023 → compressed_size = 1024, type = 0, high bits 0
    const value: u32 = 1023; // (compressed_size - 1)
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, value, .little);
    const ch = try parseChunkHeader(&buf, false);
    try testing.expectEqual(@as(u32, 1024), ch.compressed_size);
    try testing.expect(!ch.is_memset);
    try testing.expectEqual(@as(usize, 4), ch.bytes_consumed);
}

test "parseChunkHeader parses normal chunk with checksum" {
    const value: u32 = 99; // compressed_size = 100, type = 0
    var buf: [7]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], value, .little);
    buf[4] = 0xAA;
    buf[5] = 0xBB;
    buf[6] = 0xCC;
    const ch = try parseChunkHeader(&buf, true);
    try testing.expectEqual(@as(u32, 100), ch.compressed_size);
    try testing.expectEqual(@as(u32, 0xAABBCC), ch.checksum);
    try testing.expectEqual(@as(usize, 7), ch.bytes_consumed);
}

test "parseChunkHeader parses memset chunk" {
    const value: u32 = constants.chunk_type_memset; // type=1, size=0
    var buf: [5]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], value, .little);
    buf[4] = 0x42;
    const ch = try parseChunkHeader(&buf, false);
    try testing.expect(ch.is_memset);
    try testing.expectEqual(@as(u8, 0x42), ch.memset_fill);
    try testing.expectEqual(@as(usize, 5), ch.bytes_consumed);
}

// ────────────────────────────────────────────────────────────
//  Write-side tests
// ────────────────────────────────────────────────────────────

