//! 1:1 port of src/format/frame_format.zig.
//!
//! StreamLZ frame format parser / writer. Pure host code with no CUDA
//! dependencies — the wire format is shared verbatim between the CUDA
//! and Vulkan backends.
//!

const std = @import("std");
const constants = @import("streamlz_constants.zig");

pub const magic: u32 = 0x534C5A31;
pub const version: u8 = 2;
pub const end_mark: u32 = 0;

pub const block_uncompressed_flag: u32 = 0x80000000;
pub const block_parallel_decode_metadata_flag: u32 = 0x40000000;

pub const min_header_size: usize = 14;
pub const max_header_size: usize = 26;

pub const default_block_size: usize = constants.chunk_size;
pub const min_block_size: usize = 0x10000;
pub const max_block_size: usize = 0x400000;
pub const max_decompressed_block_size: usize = 512 * 1024 * 1024;

pub const default_window_size: usize = 128 * 1024 * 1024;
pub const max_window_size: usize = constants.max_dictionary_size;

/// CUDA reference: src/format/frame_format.zig:65-89. Frame header flag
/// bits packed into a single u8.
pub const FrameFlags = packed struct(u8) {
    content_size_present: bool = false,
    content_checksum: bool = false,
    block_checksums: bool = false,
    dictionary_id_present: bool = false,
    parallel_decode_metadata_present: bool = false,
    _reserved: u3 = 0,
};

/// CUDA reference: src/format/frame_format.zig:91-103. Codec enum.
/// Re-uses the name "Codec" verbatim. Note: src/format/block_header.zig
/// also exports a `CodecType` enum (different file/different concept).
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

/// CUDA reference: src/format/frame_format.zig:105-118. Parsed v2 frame
/// header.
pub const FrameHeader = struct {
    version: u8,
    flags: FrameFlags,
    codec: Codec,
    level: u8,
    block_size: u32,
    sc_group_size: f32,
    content_size: ?u64,
    dictionary_id: ?u32,
    header_size: usize,
};

/// CUDA reference: src/format/frame_format.zig:120-140. parseHeader
/// failure modes.
pub const ParseError = error{
    BadMagic,
    UnsupportedVersion,
    BadCodec,
    BadBlockSize,
    BadScGroupSize,
    Truncated,
};

/// CUDA reference: src/format/frame_format.zig:144-202. Parse a frame
/// header out of src.
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
        .header_size = pos,
    };
}

/// CUDA reference: src/format/frame_format.zig:205-207. Convert
/// sc_group_size (float, in units of 256KB chunks) to bytes.
pub fn scGroupSizeToBytes(sc_group_size: f32) usize {
    return @intFromFloat(sc_group_size * @as(f32, @floatFromInt(constants.chunk_size)));
}

/// CUDA reference: src/format/frame_format.zig:212-219. Effective
/// sub-chunk size for a given sc_group_size. For fractional groups
/// (< 1.0), sub-chunks are the group size (capped at 0xFFFF so all
/// match offsets fit in off16); for integer groups (>= 1.0), sub-chunks
/// use the default 128 KB.
pub fn scGroupSubChunkSize(sc_group_size: f32) usize {
    if (sc_group_size < 1.0) {
        return @min(scGroupSizeToBytes(sc_group_size), 0xFFFF);
    }
    return constants.sub_chunk_size;
}

/// CUDA reference: src/format/frame_format.zig:221-233. Options for
/// writeHeader.
pub const WriteHeaderOptions = struct {
    codec: Codec,
    level: u8,
    block_size: u32 = default_block_size,
    sc_group_size: f32 = constants.default_sc_group_size,
    parallel_decode_metadata_present: bool = false,
    content_size: ?u64 = null,
    content_checksum: bool = false,
    block_checksums: bool = false,
    dictionary_id: ?u32 = null,
};

/// CUDA reference: src/format/frame_format.zig:235-244. writeHeader
/// failure modes.
pub const WriteError = error{
    BadLevel,
    BadBlockSize,
    BadScGroupSize,
};

/// CUDA reference: src/format/frame_format.zig:248-294. Serialise a
/// frame header to dst.
pub fn writeHeader(dst: []u8, opts: WriteHeaderOptions) WriteError!usize {
    if (opts.level < 1 or opts.level > 9) return error.BadLevel;
    if (opts.block_size < min_block_size or opts.block_size > max_block_size) return error.BadBlockSize;
    if (!std.math.isPowerOfTwo(opts.block_size)) return error.BadBlockSize;
    if (opts.sc_group_size <= 0) return error.BadScGroupSize;

    const flags: FrameFlags = .{
        .content_size_present = opts.content_size != null,
        .content_checksum = opts.content_checksum,
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
    // sc_group_size as f32 (4 bytes LE) + 1 reserved byte = 5 bytes total.
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

/// CUDA reference: src/format/frame_format.zig:296-309. Per-block 8-byte
/// header.
pub const BlockHeader = struct {
    compressed_size: u32,
    decompressed_size: u32,
    uncompressed: bool,
    parallel_decode_metadata: bool,

    pub fn isEndMark(self: BlockHeader) bool {
        return self.compressed_size == 0 and !self.parallel_decode_metadata;
    }
};

/// CUDA reference: src/format/frame_format.zig:311-332. Parse a block
/// header out of src.
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

/// CUDA reference: src/format/frame_format.zig:334-340. Serialise an
/// 8-byte block header to dst.
pub fn writeBlockHeader(dst: []u8, hdr: BlockHeader) void {
    var raw: u32 = hdr.compressed_size;
    if (hdr.uncompressed) raw |= block_uncompressed_flag;
    if (hdr.parallel_decode_metadata) raw |= block_parallel_decode_metadata_flag;
    std.mem.writeInt(u32, dst[0..4], raw, .little);
    std.mem.writeInt(u32, dst[4..8], hdr.decompressed_size, .little);
}

/// CUDA reference: src/format/frame_format.zig:342-344. Stamp the
/// 4-byte end-of-frame sentinel at dst[0..4].
pub fn writeEndMark(dst: []u8) void {
    std.mem.writeInt(u32, dst[0..4], end_mark, .little);
}
