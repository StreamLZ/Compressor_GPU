// SPIR-V blob registry.
//
// 18 kernel shells × 3 variants (tier1, tier1_nv, tier2) = 54 embedded blobs.
// `zig build vk-shaders` produces a WriteFiles directory containing every
// <kernel>.<variant>.spv as a flat name; consumers add that directory as an
// embed path via `Module.addEmbedPath`, after which @embedFile() resolves
// the bare filenames below at zig compile time.

const std = @import("std");

pub const Tier = enum { tier1, tier1_nv, tier2 };

pub const Blob = struct {
    name: []const u8,
    tier: Tier,
    bytes: []const u8,
};

pub const blobs = [_]Blob{
    // ── encode (6) ──────────────────────────────────────────────────────
    .{ .name = "lz_encode", .tier = .tier1, .bytes = @embedFile("lz_encode.tier1.spv") },
    .{ .name = "lz_encode", .tier = .tier1_nv, .bytes = @embedFile("lz_encode.tier1_nv.spv") },
    .{ .name = "lz_encode", .tier = .tier2, .bytes = @embedFile("lz_encode.tier2.spv") },

    .{ .name = "huff_build_tables", .tier = .tier1, .bytes = @embedFile("huff_build_tables.tier1.spv") },
    .{ .name = "huff_build_tables", .tier = .tier1_nv, .bytes = @embedFile("huff_build_tables.tier1_nv.spv") },
    .{ .name = "huff_build_tables", .tier = .tier2, .bytes = @embedFile("huff_build_tables.tier2.spv") },

    .{ .name = "huff_encode_4stream", .tier = .tier1, .bytes = @embedFile("huff_encode_4stream.tier1.spv") },
    .{ .name = "huff_encode_4stream", .tier = .tier1_nv, .bytes = @embedFile("huff_encode_4stream.tier1_nv.spv") },
    .{ .name = "huff_encode_4stream", .tier = .tier2, .bytes = @embedFile("huff_encode_4stream.tier2.spv") },

    .{ .name = "assemble_measure", .tier = .tier1, .bytes = @embedFile("assemble_measure.tier1.spv") },
    .{ .name = "assemble_measure", .tier = .tier1_nv, .bytes = @embedFile("assemble_measure.tier1_nv.spv") },
    .{ .name = "assemble_measure", .tier = .tier2, .bytes = @embedFile("assemble_measure.tier2.spv") },

    .{ .name = "assemble_write", .tier = .tier1, .bytes = @embedFile("assemble_write.tier1.spv") },
    .{ .name = "assemble_write", .tier = .tier1_nv, .bytes = @embedFile("assemble_write.tier1_nv.spv") },
    .{ .name = "assemble_write", .tier = .tier2, .bytes = @embedFile("assemble_write.tier2.spv") },

    .{ .name = "frame_assemble", .tier = .tier1, .bytes = @embedFile("frame_assemble.tier1.spv") },
    .{ .name = "frame_assemble", .tier = .tier1_nv, .bytes = @embedFile("frame_assemble.tier1_nv.spv") },
    .{ .name = "frame_assemble", .tier = .tier2, .bytes = @embedFile("frame_assemble.tier2.spv") },

    // ── decode (11) ─────────────────────────────────────────────────────
    .{ .name = "walk_frame", .tier = .tier1, .bytes = @embedFile("walk_frame.tier1.spv") },
    .{ .name = "walk_frame", .tier = .tier1_nv, .bytes = @embedFile("walk_frame.tier1_nv.spv") },
    .{ .name = "walk_frame", .tier = .tier2, .bytes = @embedFile("walk_frame.tier2.spv") },

    .{ .name = "prefix_sum_chunks", .tier = .tier1, .bytes = @embedFile("prefix_sum_chunks.tier1.spv") },
    .{ .name = "prefix_sum_chunks", .tier = .tier1_nv, .bytes = @embedFile("prefix_sum_chunks.tier1_nv.spv") },
    .{ .name = "prefix_sum_chunks", .tier = .tier2, .bytes = @embedFile("prefix_sum_chunks.tier2.spv") },

    .{ .name = "scan_parse", .tier = .tier1, .bytes = @embedFile("scan_parse.tier1.spv") },
    .{ .name = "scan_parse", .tier = .tier1_nv, .bytes = @embedFile("scan_parse.tier1_nv.spv") },
    .{ .name = "scan_parse", .tier = .tier2, .bytes = @embedFile("scan_parse.tier2.spv") },

    .{ .name = "compact_huff_descs", .tier = .tier1, .bytes = @embedFile("compact_huff_descs.tier1.spv") },
    .{ .name = "compact_huff_descs", .tier = .tier1_nv, .bytes = @embedFile("compact_huff_descs.tier1_nv.spv") },
    .{ .name = "compact_huff_descs", .tier = .tier2, .bytes = @embedFile("compact_huff_descs.tier2.spv") },

    .{ .name = "compact_raw_descs", .tier = .tier1, .bytes = @embedFile("compact_raw_descs.tier1.spv") },
    .{ .name = "compact_raw_descs", .tier = .tier1_nv, .bytes = @embedFile("compact_raw_descs.tier1_nv.spv") },
    .{ .name = "compact_raw_descs", .tier = .tier2, .bytes = @embedFile("compact_raw_descs.tier2.spv") },

    .{ .name = "gather_raw_off16", .tier = .tier1, .bytes = @embedFile("gather_raw_off16.tier1.spv") },
    .{ .name = "gather_raw_off16", .tier = .tier1_nv, .bytes = @embedFile("gather_raw_off16.tier1_nv.spv") },
    .{ .name = "gather_raw_off16", .tier = .tier2, .bytes = @embedFile("gather_raw_off16.tier2.spv") },

    .{ .name = "merge_huff_descs", .tier = .tier1, .bytes = @embedFile("merge_huff_descs.tier1.spv") },
    .{ .name = "merge_huff_descs", .tier = .tier1_nv, .bytes = @embedFile("merge_huff_descs.tier1_nv.spv") },
    .{ .name = "merge_huff_descs", .tier = .tier2, .bytes = @embedFile("merge_huff_descs.tier2.spv") },

    .{ .name = "huff_build_lut", .tier = .tier1, .bytes = @embedFile("huff_build_lut.tier1.spv") },
    .{ .name = "huff_build_lut", .tier = .tier1_nv, .bytes = @embedFile("huff_build_lut.tier1_nv.spv") },
    .{ .name = "huff_build_lut", .tier = .tier2, .bytes = @embedFile("huff_build_lut.tier2.spv") },

    .{ .name = "huff_decode_4stream", .tier = .tier1, .bytes = @embedFile("huff_decode_4stream.tier1.spv") },
    .{ .name = "huff_decode_4stream", .tier = .tier1_nv, .bytes = @embedFile("huff_decode_4stream.tier1_nv.spv") },
    .{ .name = "huff_decode_4stream", .tier = .tier2, .bytes = @embedFile("huff_decode_4stream.tier2.spv") },

    .{ .name = "lz_decode", .tier = .tier1, .bytes = @embedFile("lz_decode.tier1.spv") },
    .{ .name = "lz_decode", .tier = .tier1_nv, .bytes = @embedFile("lz_decode.tier1_nv.spv") },
    .{ .name = "lz_decode", .tier = .tier2, .bytes = @embedFile("lz_decode.tier2.spv") },

    .{ .name = "lz_decode_raw", .tier = .tier1, .bytes = @embedFile("lz_decode_raw.tier1.spv") },
    .{ .name = "lz_decode_raw", .tier = .tier1_nv, .bytes = @embedFile("lz_decode_raw.tier1_nv.spv") },
    .{ .name = "lz_decode_raw", .tier = .tier2, .bytes = @embedFile("lz_decode_raw.tier2.spv") },

    .{ .name = "l1_unwrap", .tier = .tier1, .bytes = @embedFile("l1_unwrap.tier1.spv") },
    .{ .name = "l1_unwrap", .tier = .tier1_nv, .bytes = @embedFile("l1_unwrap.tier1_nv.spv") },
    .{ .name = "l1_unwrap", .tier = .tier2, .bytes = @embedFile("l1_unwrap.tier2.spv") },
};

pub fn find(name: []const u8, tier: Tier) ?[]const u8 {
    for (blobs) |b| {
        if (b.tier == tier and std.mem.eql(u8, b.name, name)) return b.bytes;
    }
    return null;
}

test "blob count is 18 kernels x 3 variants = 54" {
    try std.testing.expectEqual(@as(usize, 54), blobs.len);
}

test "every blob is non-empty and SPIR-V (magic 0x07230203)" {
    for (blobs) |b| {
        try std.testing.expect(b.bytes.len >= 4);
        const magic = std.mem.readInt(u32, b.bytes[0..4], .little);
        try std.testing.expectEqual(@as(u32, 0x07230203), magic);
    }
}

test "find returns matching blob" {
    const found = find("lz_encode", .tier1) orelse return error.NotFound;
    try std.testing.expect(found.len > 0);
    try std.testing.expect(find("nonexistent", .tier1) == null);
}
