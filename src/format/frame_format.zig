//! SLZ frame format (v2).
//!
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.
//!
//! Wire layout (little-endian) — StreamLZ v2:
//!   [4] magic = 0x534C5A31 ('SLZ1' as a big-endian mnemonic;
//!                           on-disk bytes are 0x31, 0x5A, 0x4C, 0x53)
//!   [1] version = 2                         ← bumped from 1; breaking change
//!   [1] flags
//!   [1] codec (0=High, 1=Fast, 2=Turbo)
//!   [1] level (internal: High L5/L7/L9 or Fast L1/L2/L3/L5/L6)
//!   [1] block_size_log2 (offset from log2(min_block_size) = 16)
//!   [4] sc_group_size (LE f32, units of 256 KB chunks; e.g. 0.25 / 1.0 / 4.0)
//!   [1] reserved (must be zero on write)
//!   [8] content_size (LE i64, present iff flags.ContentSize)
//!   [4] dictionary_id (LE u32, present iff flags.DictionaryId)
//!
//! Changes from v1:
//!   * version byte is now 2 (was 1). v1 files are rejected by v2 decoders.
//!   * The old 1-byte reserved slot is replaced by a 4-byte LE f32
//!     `sc_group_size` plus 1 reserved byte (10 → 14 byte fixed-header
//!     size).
//!   * FrameFlags gains bit 4 = parallel_decode_metadata_present, signaling
//!     the presence of a phase-1 sidecar block for Fast L1-L4 parallel decode.
//!
//! Then blocks:
//!   [4] compressed_size (LE u32; high bit = uncompressed; == 0 terminates)
//!   [4] decompressed_size (LE u32)
//!   [compressed_size] block payload
//!
//! After end mark (4 zeros):
//!   [4] XXH32 content checksum (LE u32, if flags.ContentChecksum)

const std = @import("std");
const constants = @import("streamlz_constants.zig");

pub const magic: u32 = 0x534C5A31;
pub const version: u8 = 2;
pub const end_mark: u32 = 0;

pub const block_uncompressed_flag: u32 = 0x80000000;

/// v2: bit 30 of the block header's `compressed_size` field marks the
/// block as a parallel-decode metadata (sidecar) block. Legacy CPU-codec
/// hint; the GPU decoder skips these blocks via the check at
/// `decode/streamlz_decoder.zig:dispatchCompressedBlock` and the GPU
/// encoder never emits the bit.
pub const block_parallel_decode_metadata_flag: u32 = 0x40000000;

// v2 header is 14 bytes fixed (up from 10 in v1): magic(4) + version(1)
// + flags(1) + codec(1) + level(1) + block_size_log2(1) + sc_group_size(4)
// + reserved(1). With optional content_size(8) + dictionary_id(4) the
// fixed-portion max is 26 bytes (up from 22 in v1).
pub const min_header_size: usize = 14;
pub const max_header_size: usize = 26;

pub const default_block_size: usize = constants.chunk_size;
pub const min_block_size: usize = 0x10000; // 64 KB
pub const max_block_size: usize = 0x400000; // 4 MB
pub const max_decompressed_block_size: usize = 512 * 1024 * 1024;

pub const default_window_size: usize = 128 * 1024 * 1024;
pub const max_window_size: usize = constants.max_dictionary_size;

pub const FrameFlags = packed struct(u8) {
    /// Bit 0: the 8 bytes immediately after the flags carry the
    /// decompressed content size as a little-endian u64. The GPU
    /// encoder always sets this; `slzGetDecompressedSize` requires it.
    content_size_present: bool = false,
    /// Bit 1: a content-wide checksum trailer follows the end-mark.
    /// Parsed but not verified by the GPU decoder today.
    content_checksum: bool = false,
    /// Bit 2: every block carries a CRC24 trailer. Parsed but not
    /// verified today.
    block_checksums: bool = false,
    /// Bit 3: a 4-byte `dictionary_id` follows the optional content
    /// size. The GPU codec has no dictionary store, so any frame
    /// carrying one is rejected with `error.UnknownDictionary`.
    dictionary_id_present: bool = false,
    /// Bit 4: the frame carries a parallel-decode metadata sidecar
    /// block for the legacy CPU codec's parallel Fast L1-L4 path.
    /// The GPU decoder skips these blocks; the GPU encoder never
    /// sets the flag (`WriteHeaderOptions` plumbs it through for
    /// symmetry, but no current call site supplies `true`).
    parallel_decode_metadata_present: bool = false,
    /// Bit 5 (v4 #19): a 4-byte chunk-Merkle checksum trailer follows
    /// the end-mark (after the bit-1 content checksum trailer when both
    /// are set): XXH32 over the concatenated per-chunk XXH32s of the
    /// decompressed content, chunk grid = the frame's eff_chunk. A
    /// self-verification root, NOT a plain content hash - external
    /// XXH32 tools must not compare against it. Default-ON since
    /// 2026-06-11.
    chunk_merkle: bool = false,
    /// Bits 6-7: reserved. Must be zero on write; non-zero values
    /// reject with `error.BadFrame`.
    _reserved: u2 = 0,
};

pub const Codec = enum(u8) {
    high = 0,
    fast = 1,
    turbo = 2,

    pub fn name(self: Codec) []const u8 {
        return switch (self) {
            .high => "High",
            .fast => "Fast",
            .turbo => "Turbo",
        };
    }
};

pub const FrameHeader = struct {
    version: u8,
    flags: FrameFlags,
    codec: Codec,
    level: u8,
    block_size: u32,
    /// SC group size in units of 256KB chunks. Fractional values < 1.0
    /// mean sub-chunk-level SC (0.25 = 64KB blocks, 0.5 = 128KB sub-chunks).
    /// Wire format: u8 byte × 0.25 (byte=1 → 0.25, byte=4 → 1.0, byte=16 → 4.0).
    sc_group_size: f32,
    content_size: ?u64,
    dictionary_id: ?u32,
    content_checksum: bool,
    chunk_merkle: bool,
    header_size: usize,
};

pub const ParseError = error{
    /// First 4 bytes did not match the SLZ1 frame magic (`0x534C5A31`).
    BadMagic,
    /// `version` field is not the current `version` constant (= 2).
    /// Frames produced by older encoders hit this; future wire-format
    /// bumps will reuse it.
    UnsupportedVersion,
    /// `codec` byte is not one of `.high` / `.fast` / `.turbo`. The
    /// GPU encoder only emits `.fast`; the other two are accepted for
    /// backward compat with frames produced by the now-retired CPU
    /// codec.
    BadCodec,
    /// Advertised `block_size` is not a power of two in
    /// `[min_block_size, max_block_size]`.
    BadBlockSize,
    /// `sc_group_size` (an `f32` post-v2) is `<= 0`.
    BadScGroupSize,
    /// `src` was shorter than the parser needed to reach the next
    /// header field.
    Truncated,
};

/// Reads and parses a frame header from `src`. Returns the parsed header
/// and the number of bytes consumed (header_size).
pub fn parseHeader(src: []const u8) ParseError!FrameHeader {
    if (src.len < min_header_size) return error.Truncated;

    const got_magic = std.mem.readInt(u32, src[0..4], .little);
    if (got_magic != magic) return error.BadMagic;

    var pos: usize = 4;
    const ver = src[pos];
    pos += 1;
    if (ver != version) return error.UnsupportedVersion;

    const raw_flags: FrameFlags = @bitCast(src[pos]);
    pos += 1;
    const codec: Codec = std.enums.fromInt(Codec, src[pos]) orelse return error.BadCodec;
    pos += 1;
    const lvl = src[pos];
    pos += 1;

    const min_log2 = std.math.log2_int(usize, min_block_size);
    const max_log2 = std.math.log2_int(usize, max_block_size);
    const block_size_log2_encoded = src[pos];
    pos += 1;
    const block_size_log2 = @as(u8, @intCast(min_log2)) + block_size_log2_encoded;
    if (block_size_log2 < min_log2 or block_size_log2 > max_log2) return error.BadBlockSize;
    const block_size: u32 = @as(u32, 1) << @intCast(block_size_log2);

    // sc_group_size as f32 (4 bytes LE) + 1 reserved byte = 5 bytes total.
    const sc_group_size: f32 = @bitCast(std.mem.readInt(u32, src[pos..][0..4], .little));
    pos += 4;
    pos += 1; // reserved
    if (!(sc_group_size > 0)) return error.BadScGroupSize;

    var content_size: ?u64 = null;
    if (raw_flags.content_size_present) {
        if (src.len < pos + 8) return error.Truncated;
        const cs_raw = std.mem.readInt(i64, src[pos..][0..8], .little);
        content_size = if (cs_raw >= 0) @intCast(cs_raw) else null;
        pos += 8;
    }

    var dict_id: ?u32 = null;
    if (raw_flags.dictionary_id_present) {
        if (src.len < pos + 4) return error.Truncated;
        dict_id = std.mem.readInt(u32, src[pos..][0..4], .little);
        pos += 4;
    }

    return .{
        .version = ver,
        .flags = raw_flags,
        .codec = codec,
        .level = lvl,
        .block_size = block_size,
        .sc_group_size = sc_group_size,
        .content_size = content_size,
        .dictionary_id = dict_id,
        .content_checksum = raw_flags.content_checksum,
        .chunk_merkle = raw_flags.chunk_merkle,
        .header_size = pos,
    };
}

/// Convert sc_group_size (float, in units of 256KB chunks) to bytes.
pub fn scGroupSizeToBytes(sc_group_size: f32) usize {
    return @intFromFloat(sc_group_size * @as(f32, @floatFromInt(constants.chunk_size)));
}

/// Effective sub-chunk size for a given sc_group_size.
/// For fractional groups (< 1.0), sub-chunks are the group size.
/// For integer groups (>= 1.0), sub-chunks use the default 128KB.
pub fn scGroupSubChunkSize(sc_group_size: f32) usize {
    if (sc_group_size < 1.0) {
        // Cap at 0xFFFF (65535) so all match offsets fit in off16.
        // No off32 values produced → no OffsetOutOfBounds risk.
        return @min(scGroupSizeToBytes(sc_group_size), 0xFFFF);
    }
    return constants.sub_chunk_size;
}

pub const WriteHeaderOptions = struct {
    codec: Codec,
    level: u8,
    block_size: u32 = default_block_size,
    /// SC group size in units of 256KB chunks. 4.0 = default (1MB groups).
    /// 0.25 = 64KB blocks (maximum GPU parallelism).
    sc_group_size: f32 = constants.default_sc_group_size,
    parallel_decode_metadata_present: bool = false,
    content_size: ?u64 = null,
    content_checksum: bool = false,
    chunk_merkle: bool = false,
    block_checksums: bool = false,
    dictionary_id: ?u32 = null,
};

pub const WriteError = error{
    /// `opts.level` is outside `[1, 9]` (the frame header's level field
    /// is 4 bits; this catches an overflow before the bit-pack).
    BadLevel,
    /// `opts.block_size` is not a power of two in
    /// `[min_block_size, max_block_size]`.
    BadBlockSize,
    /// `opts.sc_group_size` is `<= 0`.
    BadScGroupSize,
};

/// Writes a frame header to `dst` and returns the number of bytes written.
/// Caller must ensure `dst.len >= max_header_size`.
pub fn writeHeader(dst: []u8, opts: WriteHeaderOptions) WriteError!usize {
    if (opts.level < 1 or opts.level > 9) return error.BadLevel;
    if (opts.block_size < min_block_size or opts.block_size > max_block_size) return error.BadBlockSize;
    if (!std.math.isPowerOfTwo(opts.block_size)) return error.BadBlockSize;
    if (opts.sc_group_size <= 0) return error.BadScGroupSize;

    const flags: FrameFlags = .{
        .content_size_present = opts.content_size != null,
        .content_checksum = opts.content_checksum,
        .chunk_merkle = opts.chunk_merkle,
        .block_checksums = opts.block_checksums,
        .dictionary_id_present = opts.dictionary_id != null,
        .parallel_decode_metadata_present = opts.parallel_decode_metadata_present,
    };

    var pos: usize = 0;
    std.mem.writeInt(u32, dst[pos..][0..4], magic, .little);
    pos += 4;
    dst[pos] = version;
    pos += 1;
    dst[pos] = @bitCast(flags);
    pos += 1;
    dst[pos] = @intFromEnum(opts.codec);
    pos += 1;
    dst[pos] = opts.level;
    pos += 1;

    const min_log2: u8 = @intCast(std.math.log2_int(usize, min_block_size));
    const this_log2: u8 = @intCast(std.math.log2_int(u32, opts.block_size));
    dst[pos] = this_log2 - min_log2;
    pos += 1;
    // sc_group_size as f32 (4 bytes LE) + 1 reserved byte = 5 bytes total
    // (replaces the old 1-byte sc_group_size + 4 reserved bytes).
    std.mem.writeInt(u32, dst[pos..][0..4], @bitCast(opts.sc_group_size), .little);
    pos += 4;
    dst[pos] = 0;
    pos += 1;

    if (opts.content_size) |cs| {
        std.mem.writeInt(i64, dst[pos..][0..8], @intCast(cs), .little);
        pos += 8;
    }
    if (opts.dictionary_id) |id| {
        std.mem.writeInt(u32, dst[pos..][0..4], id, .little);
        pos += 4;
    }
    return pos;
}

pub const BlockHeader = struct {
    compressed_size: u32,
    decompressed_size: u32,
    uncompressed: bool,
    /// v2: this block carries parallel-decode sidecar metadata, not
    /// decompressible data. Its compressed_size bytes are the sidecar
    /// payload; its decompressed_size is 0. Serial decoders skip it.
    parallel_decode_metadata: bool,

    /// True when this is the end-of-stream sentinel (compressed_size == 0).
    pub fn isEndMark(self: BlockHeader) bool {
        return self.compressed_size == 0 and !self.parallel_decode_metadata;
    }
};

pub fn parseBlockHeader(src: []const u8) ParseError!BlockHeader {
    // End mark is just 4 zero bytes — no decompressed size field.
    if (src.len < 4) return error.Truncated;
    const raw = std.mem.readInt(u32, src[0..4], .little);
    if (raw == end_mark) {
        return .{
            .compressed_size = 0,
            .decompressed_size = 0,
            .uncompressed = false,
            .parallel_decode_metadata = false,
        };
    }
    if (src.len < 8) return error.Truncated;
    const decompressed = std.mem.readInt(u32, src[4..8], .little);
    const size_mask: u32 = ~(block_uncompressed_flag | block_parallel_decode_metadata_flag);
    return .{
        .compressed_size = raw & size_mask,
        .decompressed_size = decompressed,
        .uncompressed = (raw & block_uncompressed_flag) != 0,
        .parallel_decode_metadata = (raw & block_parallel_decode_metadata_flag) != 0,
    };
}

pub fn writeBlockHeader(dst: []u8, hdr: BlockHeader) void {
    var raw: u32 = hdr.compressed_size;
    if (hdr.uncompressed) raw |= block_uncompressed_flag;
    if (hdr.parallel_decode_metadata) raw |= block_parallel_decode_metadata_flag;
    std.mem.writeInt(u32, dst[0..4], raw, .little);
    std.mem.writeInt(u32, dst[4..8], hdr.decompressed_size, .little);
}

pub fn writeEndMark(dst: []u8) void {
    std.mem.writeInt(u32, dst[0..4], end_mark, .little);
}

// ────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────

const testing = std.testing;

test "parseHeader rejects non-SLZ1 magic" {
    // min_header_size bytes of garbage with the wrong magic up front.
    const bogus = [_]u8{ 'N', 'O', 'P', 'E' } ++ @as([min_header_size - 4]u8, @splat(0));
    try testing.expectError(error.BadMagic, parseHeader(&bogus));
}

test "parseHeader rejects truncated input" {
    const tiny = [_]u8{ 0x31, 0x5A, 0x4C, 0x53 };
    try testing.expectError(error.Truncated, parseHeader(&tiny));
}

test "parseHeader rejects v1 frames (breaking change for v2)" {
    // Original v1 fixture (18 bytes) — v2 decoders must reject.
    const v1_fixture = [_]u8{
        0x31, 0x5A, 0x4C, 0x53, // magic 'SLZ1'
        0x01, // version = 1 (v1, not supported by v2 decoder)
        0x01, // flags: content_size_present
        0x01, // codec: Fast
        0x01, // level: 1
        0x02, // blockSizeLog2
        0x00, // v1 reserved
        0x29, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // content_size
    };
    try testing.expectError(error.UnsupportedVersion, parseHeader(&v1_fixture));
}

test "parseHeader rejects sc_group_size == 0" {
    // Handcrafted v2 header with a zero sc_group_size.
    const bad = [_]u8{
        0x31, 0x5A, 0x4C, 0x53, // magic
        0x02, // version = 2
        0x00, // flags: none
        0x01, // codec: Fast
        0x01, // level: 1
        0x02, // block_size_log2
        0x00, 0x00, 0x00, 0x00, // sc_group_size = 0.0f32  ← invalid
        0x00, // reserved
    };
    try testing.expectError(error.BadScGroupSize, parseHeader(&bad));
}

test "writeHeader / parseHeader roundtrip, minimal flags" {
    var buf: [max_header_size]u8 = undefined;
    const n = try writeHeader(&buf, .{
        .codec = .fast,
        .level = 1,
        .block_size = default_block_size,
    });
    const hdr = try parseHeader(buf[0..n]);
    try testing.expectEqual(@as(u8, 2), hdr.version);
    try testing.expectEqual(Codec.fast, hdr.codec);
    try testing.expectEqual(@as(u8, 1), hdr.level);
    try testing.expectEqual(@as(u32, default_block_size), hdr.block_size);
    try testing.expectEqual(constants.default_sc_group_size, hdr.sc_group_size);
    try testing.expect(!hdr.flags.parallel_decode_metadata_present);
    try testing.expect(hdr.content_size == null);
    try testing.expectEqual(n, hdr.header_size);
}

test "writeHeader / parseHeader roundtrip, with content_size" {
    var buf: [max_header_size]u8 = undefined;
    const n = try writeHeader(&buf, .{
        .codec = .high,
        .level = 9,
        .block_size = 1 << 18,
        .content_size = 1234567,
    });
    const hdr = try parseHeader(buf[0..n]);
    try testing.expectEqual(Codec.high, hdr.codec);
    try testing.expectEqual(@as(u8, 9), hdr.level);
    try testing.expectEqual(@as(?u64, 1234567), hdr.content_size);
    try testing.expect(hdr.flags.content_size_present);
}

test "writeHeader / parseHeader roundtrip, custom sc_group_size + parallel flag" {
    var buf: [max_header_size]u8 = undefined;
    const n = try writeHeader(&buf, .{
        .codec = .fast,
        .level = 3,
        .block_size = default_block_size,
        .sc_group_size = 8,
        .parallel_decode_metadata_present = true,
    });
    const hdr = try parseHeader(buf[0..n]);
    try testing.expectEqual(@as(u8, 2), hdr.version);
    try testing.expectEqual(@as(f32, 8.0), hdr.sc_group_size);
    try testing.expect(hdr.flags.parallel_decode_metadata_present);
}

test "block header end mark detected" {
    const em = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const block_hdr = try parseBlockHeader(&em);
    try testing.expect(block_hdr.isEndMark());
}

test "block header uncompressed flag roundtrip" {
    var buf: [8]u8 = undefined;
    writeBlockHeader(&buf, .{
        .compressed_size = 41,
        .decompressed_size = 41,
        .uncompressed = true,
        .parallel_decode_metadata = false,
    });
    const block_hdr = try parseBlockHeader(&buf);
    try testing.expect(block_hdr.uncompressed);
    try testing.expect(!block_hdr.parallel_decode_metadata);
    try testing.expectEqual(@as(u32, 41), block_hdr.compressed_size);
    try testing.expectEqual(@as(u32, 41), block_hdr.decompressed_size);
}

test "block header parallel_decode_metadata flag roundtrip" {
    var buf: [8]u8 = undefined;
    writeBlockHeader(&buf, .{
        .compressed_size = 123,
        .decompressed_size = 0,
        .uncompressed = false,
        .parallel_decode_metadata = true,
    });
    const block_hdr = try parseBlockHeader(&buf);
    try testing.expect(block_hdr.parallel_decode_metadata);
    try testing.expect(!block_hdr.uncompressed);
    try testing.expectEqual(@as(u32, 123), block_hdr.compressed_size);
    try testing.expectEqual(@as(u32, 0), block_hdr.decompressed_size);
    try testing.expect(!block_hdr.isEndMark());
}
