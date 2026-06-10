//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Unit tests for srcVK/decode/streamlz_decoder.zig: parseHeader /
//! parseBlockHeader / parseChunkHeader edge cases; pure host-side, no
//! GPU. Test bodies added by the fleshout agent.
//!

const std = @import("std");
const testing = std.testing;
const decoder = @import("../decode/streamlz_decoder.zig");
const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const constants = @import("../format/streamlz_constants.zig");
const descriptors = @import("../decode/descriptors.zig");

// ── parseHeader ────────────────────────────────────────────────────────

test "parseHeader: valid SLZ1 frame header round-trips writeHeader" {
    var buf: [frame.max_header_size]u8 = undefined;
    const opts: frame.WriteHeaderOptions = .{
        .codec = .fast,
        .level = 1,
        .block_size = @intCast(constants.chunk_size),
        .sc_group_size = constants.default_sc_group_size,
        .content_size = 12345,
    };
    const written = try frame.writeHeader(&buf, opts);
    try testing.expect(written >= frame.min_header_size);

    const hdr = try frame.parseHeader(buf[0..written]);
    try testing.expectEqual(@as(u8, frame.version), hdr.version);
    try testing.expectEqual(frame.Codec.fast, hdr.codec);
    try testing.expectEqual(@as(u8, 1), hdr.level);
    try testing.expectEqual(@as(u32, @intCast(constants.chunk_size)), hdr.block_size);
    try testing.expectEqual(constants.default_sc_group_size, hdr.sc_group_size);
    try testing.expectEqual(@as(?u64, 12345), hdr.content_size);
    try testing.expectEqual(written, hdr.header_size);
}

test "parseHeader: bad magic returns BadMagic" {
    var buf: [frame.min_header_size]u8 = @splat(0);
    // Wrong magic (0 instead of 0x534C5A31).
    try testing.expectError(error.BadMagic, frame.parseHeader(&buf));
}

test "parseHeader: truncated input returns Truncated" {
    var buf: [frame.min_header_size - 1]u8 = @splat(0);
    try testing.expectError(error.Truncated, frame.parseHeader(&buf));
}

test "parseHeader: unsupported version returns UnsupportedVersion" {
    var buf: [frame.min_header_size]u8 = @splat(0);
    std.mem.writeInt(u32, buf[0..4], frame.magic, .little);
    buf[4] = 99; // bogus version
    try testing.expectError(error.UnsupportedVersion, frame.parseHeader(&buf));
}

test "parseHeader: bad codec returns BadCodec" {
    var buf: [frame.min_header_size]u8 = @splat(0);
    std.mem.writeInt(u32, buf[0..4], frame.magic, .little);
    buf[4] = frame.version;
    buf[5] = 0; // flags
    buf[6] = 0xEE; // invalid codec
    try testing.expectError(error.BadCodec, frame.parseHeader(&buf));
}

test "parseHeader: out-of-range block size encoded log2 returns BadBlockSize" {
    var buf: [frame.min_header_size]u8 = @splat(0);
    std.mem.writeInt(u32, buf[0..4], frame.magic, .little);
    buf[4] = frame.version;
    buf[5] = 0;
    buf[6] = @intFromEnum(frame.Codec.fast);
    buf[7] = 1;
    // Encoded block-size log2 of 0xFF is well past the valid window
    // (min_log2=16, max_log2=22).
    buf[8] = 0xFF;
    // valid sc_group_size payload (4.0 as f32) so we reach the block-size
    // check before the sc_group_size check.
    std.mem.writeInt(u32, buf[9..][0..4], @bitCast(@as(f32, 4.0)), .little);
    buf[13] = 0;
    try testing.expectError(error.BadBlockSize, frame.parseHeader(&buf));
}

// ── parseBlockHeader (frame-level) ─────────────────────────────────────

test "parseBlockHeader: end-mark (4 zero bytes) round-trips" {
    var buf: [8]u8 = @splat(0);
    const hdr = try frame.parseBlockHeader(&buf);
    try testing.expect(hdr.isEndMark());
    try testing.expectEqual(@as(u32, 0), hdr.compressed_size);
}

test "parseBlockHeader: uncompressed block round-trips writeBlockHeader" {
    var buf: [8]u8 = @splat(0);
    const want: frame.BlockHeader = .{
        .compressed_size = 4096,
        .decompressed_size = 4096,
        .uncompressed = true,
        .parallel_decode_metadata = false,
    };
    frame.writeBlockHeader(&buf, want);
    const got = try frame.parseBlockHeader(&buf);
    try testing.expectEqual(want.compressed_size, got.compressed_size);
    try testing.expectEqual(want.decompressed_size, got.decompressed_size);
    try testing.expect(got.uncompressed);
    try testing.expect(!got.parallel_decode_metadata);
}

test "parseBlockHeader: compressed block round-trips writeBlockHeader" {
    var buf: [8]u8 = @splat(0);
    const want: frame.BlockHeader = .{
        .compressed_size = 9000,
        .decompressed_size = 32768,
        .uncompressed = false,
        .parallel_decode_metadata = false,
    };
    frame.writeBlockHeader(&buf, want);
    const got = try frame.parseBlockHeader(&buf);
    try testing.expectEqual(want.compressed_size, got.compressed_size);
    try testing.expectEqual(want.decompressed_size, got.decompressed_size);
    try testing.expect(!got.uncompressed);
    try testing.expect(!got.isEndMark());
}

test "parseBlockHeader: truncated returns Truncated" {
    var buf: [2]u8 = .{ 0x55, 0xAA };
    try testing.expectError(error.Truncated, frame.parseBlockHeader(&buf));
}

// ── parseBlockHeader (internal 2-byte header) ──────────────────────────

test "internal parseBlockHeader: fast codec uncompressed flag bit" {
    // b0 = magic 0x5 in low nibble, uncompressed bit (b7) set
    const b0: u8 = 0x80 | 0x05;
    const b1: u8 = @intFromEnum(block_header.CodecType.fast);
    const buf = [_]u8{ b0, b1 };
    const got = try block_header.parseBlockHeader(&buf);
    try testing.expectEqual(block_header.CodecType.fast, got.decoder_type);
    try testing.expect(got.uncompressed);
}

test "internal parseBlockHeader: bad magic in low nibble returns BadMagic" {
    const buf = [_]u8{ 0x00, @intFromEnum(block_header.CodecType.fast) };
    try testing.expectError(error.BadMagic, block_header.parseBlockHeader(&buf));
}

test "internal parseBlockHeader: bad decoder type returns BadDecoderType" {
    const buf = [_]u8{ 0x05, 0x7E }; // decoder byte = 0x7E (>= 3, no checksums)
    try testing.expectError(error.BadDecoderType, block_header.parseBlockHeader(&buf));
}

// ── parseChunkHeader ───────────────────────────────────────────────────

test "parseChunkHeader: LZ chunk (type 0) parses size+1 and 4 bytes consumed" {
    // type=0, raw size field encodes (compressed_size - 1).
    const compressed_size: u32 = 1234;
    const v: u32 = (0 << constants.chunk_type_shift) | (compressed_size - 1);
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    const ch = try block_header.parseChunkHeader(&buf, false);
    try testing.expectEqual(compressed_size, ch.compressed_size);
    try testing.expectEqual(@as(usize, 4), ch.bytes_consumed);
    try testing.expect(!ch.is_memset);
}

test "parseChunkHeader: LZ chunk with checksum returns 7 bytes consumed" {
    const compressed_size: u32 = 999;
    const v: u32 = (0 << constants.chunk_type_shift) | (compressed_size - 1);
    var buf: [7]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], v, .little);
    buf[4] = 0xAB;
    buf[5] = 0xCD;
    buf[6] = 0xEF;
    const ch = try block_header.parseChunkHeader(&buf, true);
    try testing.expectEqual(compressed_size, ch.compressed_size);
    try testing.expectEqual(@as(usize, 7), ch.bytes_consumed);
    const expected_cs: u32 = (@as(u32, 0xAB) << 16) | (@as(u32, 0xCD) << 8) | @as(u32, 0xEF);
    try testing.expectEqual(expected_cs, ch.checksum);
}

test "parseChunkHeader: memset chunk (type 1) decodes fill byte" {
    // type=1 (memset), size field ignored.
    const v: u32 = (1 << constants.chunk_type_shift);
    var buf: [5]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], v, .little);
    buf[4] = 0x7F;
    const ch = try block_header.parseChunkHeader(&buf, false);
    try testing.expect(ch.is_memset);
    try testing.expectEqual(@as(u8, 0x7F), ch.memset_fill);
    try testing.expectEqual(@as(usize, 5), ch.bytes_consumed);
}

test "parseChunkHeader: truncated input returns TooShort" {
    var buf: [3]u8 = .{ 0, 0, 0 };
    try testing.expectError(error.TooShort, block_header.parseChunkHeader(&buf, false));
}

test "parseChunkHeader: bad chunk type (2 or 3) returns BadChunkType" {
    // type=2 (reserved/invalid).
    const v: u32 = (2 << constants.chunk_type_shift);
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try testing.expectError(error.BadChunkType, block_header.parseChunkHeader(&buf, false));
}

// ── buildChunkDescriptors ──────────────────────────────────────────────

test "buildChunkDescriptors: empty/zero decompressed_size produces no chunks" {
    var descs: [4]descriptors.ChunkDesc = undefined;
    @memset(std.mem.sliceAsBytes(descs[0..]), 0);
    const empty_payload = [_]u8{};
    try decoder.buildChunkDescriptors(
        &empty_payload,
        descs[0..0],
        constants.chunk_size,
        0,
        0,
    );
    // Nothing to assert beyond the call succeeding with len==0.
}

test "buildChunkDescriptors: single uncompressed chunk sets flags=1, dst/comp/decomp sizes" {
    // Construct a 1-chunk block where the internal block header says
    // "uncompressed": b0 with uncompressed bit (0x80) + magic 0x5; b1 fast.
    const decompressed_size: usize = 256;
    var payload: [2 + 256]u8 = undefined;
    payload[0] = 0x80 | 0x05; // uncompressed = true, magic 0x5
    payload[1] = @intFromEnum(block_header.CodecType.fast);
    // Fill the 256 raw bytes with anything.
    for (payload[2..], 0..) |*b, i| b.* = @intCast(i & 0xFF);

    var descs: [1]descriptors.ChunkDesc = undefined;
    @memset(std.mem.sliceAsBytes(descs[0..]), 0);
    try decoder.buildChunkDescriptors(
        &payload,
        &descs,
        constants.chunk_size,
        decompressed_size,
        0,
    );
    try testing.expectEqual(@as(u32, 1), descs[0].flags); // uncompressed shortcut
    try testing.expectEqual(@as(u32, decompressed_size), descs[0].decomp_size);
    try testing.expectEqual(@as(u32, decompressed_size), descs[0].comp_size);
    try testing.expectEqual(@as(u32, 0), descs[0].dst_offset);
    // After 2 internal hdr bytes, src_offset == 2.
    try testing.expectEqual(@as(u32, 2), descs[0].src_offset);
}

test "buildChunkDescriptors: memset chunk sets flags=2 and carries fill byte" {
    // Internal block hdr (fast, NOT uncompressed at block scope), then a
    // memset chunk header (4-byte word with type=1) + 1 fill byte.
    var payload: [2 + 5]u8 = undefined;
    payload[0] = 0x05; // magic, no uncompressed
    payload[1] = @intFromEnum(block_header.CodecType.fast);
    const v: u32 = (1 << constants.chunk_type_shift);
    std.mem.writeInt(u32, payload[2..][0..4], v, .little);
    payload[6] = 0x42; // fill byte

    const decompressed_size: usize = 1024;
    var descs: [1]descriptors.ChunkDesc = undefined;
    @memset(std.mem.sliceAsBytes(descs[0..]), 0);
    try decoder.buildChunkDescriptors(
        &payload,
        &descs,
        constants.chunk_size,
        decompressed_size,
        0,
    );
    try testing.expectEqual(@as(u32, 2), descs[0].flags); // memset
    try testing.expectEqual(@as(u8, 0x42), descs[0].memset_fill);
    try testing.expectEqual(@as(u32, decompressed_size), descs[0].decomp_size);
    try testing.expectEqual(@as(u32, 0), descs[0].comp_size);
}

test "buildChunkDescriptors: truncated input returns Truncated" {
    const payload = [_]u8{0x05}; // only 1 byte; need 2 for internal hdr
    var descs: [1]descriptors.ChunkDesc = undefined;
    try testing.expectError(error.Truncated, decoder.buildChunkDescriptors(
        &payload,
        &descs,
        constants.chunk_size,
        128,
        0,
    ));
}
