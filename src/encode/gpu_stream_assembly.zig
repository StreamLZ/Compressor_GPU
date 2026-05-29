//! Host-side assembly of the SLZ1 wire format from raw GPU encoder output.
//!
//! The GPU LZ kernel emits each sub-chunk as a packed sequence of raw byte
//! streams: literals, tokens, off16, off32, lengths. This module re-encodes
//! that raw payload into the final wire format that the GPU decoder reads
//! — choosing per stream between Huffman (chunk_type=4), tANS (chunk_type=6),
//! or a literal memcpy, mirroring what the original CPU encoder would have
//! produced.
//!
//! Two coding paths arrive here:
//!
//!   * Huffman bodies pre-computed on the GPU. The orchestrator hands us
//!     a body slice and we splice it in with a 5-byte chunk-4 header.
//!   * Per-pair tANS streams. When two adjacent 64 KB sub-chunks share a
//!     combined literal / token / off16 distribution, tANS-encoding the
//!     concatenation beats two independent passes. The "primary" sub-chunk
//!     carries the combined stream; the "secondary" emits only a pointer.
//!
//! Everything else (off32, length) is copied through verbatim — the GPU
//! kernel already produced the wire bytes for those streams.

const std = @import("std");

const fast_constants = @import("fast/fast_constants.zig");
const entropy_enc = @import("entropy/entropy_encoder.zig");
const tans_enc = @import("entropy/tans_encoder.zig");
const ByteHistogram = @import("entropy/byte_histogram.zig").ByteHistogram;

pub const EntropyOptions = entropy_enc.EntropyOptions;

/// Slice triplet extracted from a raw GPU sub-chunk payload.
pub const RawStreams = struct {
    literals: []const u8,
    tokens: []const u8,
    /// Interleaved u16 pairs (lo, hi, lo, hi, ...). Empty when the payload
    /// already advertises entropy-coded off16 via the sentinel marker.
    off16_data: []const u8,
    off16_count: usize,
};

/// Parse the literal / token / off16 fields off the front of a raw GPU
/// payload. Used by the paired-tANS pre-pass — the per-chunk wire writer
/// has its own walk that consumes the remaining off32 / length data.
pub fn extractRawStreams(raw: []const u8, init_bytes: usize) ?RawStreams {
    var rp: usize = init_bytes;
    if (rp + 3 > raw.len) return null;
    const lit_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + lit_count > raw.len) return null;
    const literals = raw[rp..][0..lit_count];
    rp += lit_count;

    if (rp + 3 > raw.len) return null;
    const token_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + token_count > raw.len) return null;
    const tokens = raw[rp..][0..token_count];
    rp += token_count;

    // cmd_stream2_offset is present only for sub-chunks > 64 KB. With
    // sc_group=0.25 every unit is exactly 64 KB so it is never present
    // on the input we serve.

    if (rp + 2 > raw.len) return null;
    const off16_count: usize = @as(usize, std.mem.readInt(u16, raw[rp..][0..2], .little));
    rp += 2;
    if (off16_count == fast_constants.entropy_coded_16_marker) {
        return .{ .literals = literals, .tokens = tokens, .off16_data = &.{}, .off16_count = 0 };
    }
    const off16_bytes = off16_count * 2;
    if (rp + off16_bytes > raw.len) return null;
    return .{
        .literals = literals,
        .tokens = tokens,
        .off16_data = raw[rp..][0..off16_bytes],
        .off16_count = off16_count,
    };
}

pub const CombinedTansResult = struct { stream: []u8, count_a: u32 };

/// Concatenate two byte streams and tANS32-encode them as ONE wire chunk
/// (chunk_type=6 non-compact). The result owns its buffer — caller frees.
/// Returns null when either input is too small to be worth pairing or when
/// the combined size would overflow the u24 count fields.
pub fn encodeCombinedTans32(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) ?CombinedTansResult {
    if (a.len < 32 or b.len < 32) return null;
    if (a.len + b.len > 0xFFFFFF) return null; // count_a / count_b are u24

    const combined = allocator.alloc(u8, a.len + b.len) catch return null;
    defer allocator.free(combined);
    @memcpy(combined[0..a.len], a);
    @memcpy(combined[a.len..], b);

    var histogram: ByteHistogram = .{};
    histogram.countBytes(combined);

    var cost: f32 = std.math.inf(f32);
    const enc = allocator.alloc(u8, combined.len + 256) catch return null;
    const n = tans_enc.encodeArrayU8Tans32(allocator, enc[5..], combined, &histogram, 0.0, &cost, null) catch {
        allocator.free(enc);
        return null;
    };
    if (n == 0 or n + 5 >= combined.len) {
        allocator.free(enc);
        return null;
    }
    entropy_enc.writeNonCompactChunkHeader(enc, 6, @intCast(n), @intCast(combined.len));
    return .{ .stream = enc[0 .. 5 + n], .count_a = @intCast(a.len) };
}

/// Per-stream pairing override. When two adjacent 64 KB sub-chunks pair on a
/// stream, the primary carries the combined tANS body and the secondary
/// emits a pointer that names countA so the decoder can split the bytes
/// back across the pair.
pub const LitOverride = union(enum) {
    none,
    primary: struct { combined_stream: []const u8, count_a: u32 },
    secondary: struct { count_a: u32 },
};

/// Pre-encoded streams the GPU may hand us per sub-chunk. Any non-null
/// field is a chunk-type=4 Huffman body (no 5-byte header — we add it).
/// Null fields fall back to whatever the host-side encoder would have
/// done (typically: try CPU Huffman / tANS, then memcpy if no win).
pub const GpuHuffStreams = struct {
    lit: ?[]const u8 = null,
    tok: ?[]const u8 = null,
    off16_hi: ?[]const u8 = null,
    off16_lo: ?[]const u8 = null,
};

/// Re-encode a raw GPU sub-chunk payload into the SLZ1 wire format,
/// writing into `dst`. Returns bytes written, or 0 on any failure.
///
/// `init_bytes` carries the 8-byte raw header on the very first sub-chunk
/// of the frame (matches the CPU encoder convention; the GPU LZ kernel
/// only emits these for chunk 0). `src_size` is the decompressed sub-chunk
/// length (gates the `cmd_stream2_offset` presence test).
pub fn reencodeGpuWithEntropy(
    allocator: std.mem.Allocator,
    raw: []const u8,
    dst: []u8,
    options: EntropyOptions,
    speed_tradeoff: f32,
    init_bytes: usize,
    src_size: usize,
    huff: GpuHuffStreams,
    lit_override: LitOverride,
    tok_override: LitOverride,
    off16hi_override: LitOverride,
) !usize {
    var rp: usize = 0;
    var wp: usize = 0;

    // 8-byte raw init bytes prefix the very first sub-chunk of the frame.
    if (init_bytes > 0) {
        if (rp + init_bytes > raw.len) return 0;
        if (wp + init_bytes > dst.len) return 0;
        @memcpy(dst[wp..][0..init_bytes], raw[rp..][0..init_bytes]);
        rp += init_bytes;
        wp += init_bytes;
    }

    // ── Parse the raw payload's framed streams ───────────────────────────
    if (rp + 3 > raw.len) return 0;
    const lit_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + lit_count > raw.len) return 0;
    const literals = raw[rp..][0..lit_count];
    rp += lit_count;
    entropy_enc.measureStream(allocator, 0, literals);

    if (rp + 3 > raw.len) return 0;
    const token_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + token_count > raw.len) return 0;
    const tokens = raw[rp..][0..token_count];
    rp += token_count;
    entropy_enc.measureStream(allocator, 1, tokens);

    var cmd_stream2_data: ?[2]u8 = null;
    if (src_size > 0x10000) {
        if (rp + 2 > raw.len) return 0;
        cmd_stream2_data = raw[rp..][0..2].*;
        rp += 2;
    }

    if (rp + 2 > raw.len) return 0;
    const off16_count: usize = @as(usize, std.mem.readInt(u16, raw[rp..][0..2], .little));
    rp += 2;
    const off16_bytes = off16_count * 2;
    if (rp + off16_bytes > raw.len) return 0;
    const off16_data = raw[rp..][0..off16_bytes];
    rp += off16_bytes;

    if (rp + 3 > raw.len) return 0;
    const off32_hdr = raw[rp..][0..3];
    const off32_packed: u32 = @as(u32, off32_hdr[0]) | (@as(u32, off32_hdr[1]) << 8) | (@as(u32, off32_hdr[2]) << 16);
    rp += 3;
    var off32_c1: usize = (off32_packed >> 12) & 0xFFF;
    var off32_c2: usize = off32_packed & 0xFFF;
    var off32_extra: usize = 0;
    if (off32_c1 >= 4095) {
        if (rp + 2 > raw.len) return 0;
        off32_c1 = std.mem.readInt(u16, raw[rp..][0..2], .little);
        rp += 2;
        off32_extra += 2;
    }
    if (off32_c2 >= 4095) {
        if (rp + 2 > raw.len) return 0;
        off32_c2 = std.mem.readInt(u16, raw[rp..][0..2], .little);
        rp += 2;
        off32_extra += 2;
    }
    // off32 entries are either 3 or 4 bytes; the high byte of the last
    // triple sets `>= 0xC0` to flag the 4-byte form. Scan to size precisely.
    const off32_byte_count: usize = blk: {
        var count: usize = 0;
        var scan = rp;
        const total_entries = off32_c1 + off32_c2;
        for (0..total_entries) |_| {
            if (scan + 3 > raw.len) break :blk count;
            if (raw[scan + 2] >= 0xC0) {
                if (scan + 4 > raw.len) break :blk count;
                count += 4;
                scan += 4;
            } else {
                count += 3;
                scan += 3;
            }
        }
        break :blk count;
    };
    if (rp + off32_byte_count > raw.len) return 0;
    rp += off32_byte_count;

    const length_data = raw[rp..];

    // ── Emit literals ────────────────────────────────────────────────────
    const lit_n = blk: {
        switch (lit_override) {
            .primary => |p| {
                if (wp + 4 + p.combined_stream.len > dst.len) break :blk @as(usize, 0);
                dst[wp] = 0x70;
                dst[wp + 1] = @intCast((p.count_a >> 16) & 0xFF);
                dst[wp + 2] = @intCast((p.count_a >> 8) & 0xFF);
                dst[wp + 3] = @intCast(p.count_a & 0xFF);
                @memcpy(dst[wp + 4 ..][0..p.combined_stream.len], p.combined_stream);
                break :blk 4 + p.combined_stream.len;
            },
            .secondary => |s| {
                if (wp + 7 > dst.len) break :blk @as(usize, 0);
                const count_b: u32 = @intCast(literals.len);
                dst[wp] = 0x50;
                dst[wp + 1] = @intCast((s.count_a >> 16) & 0xFF);
                dst[wp + 2] = @intCast((s.count_a >> 8) & 0xFF);
                dst[wp + 3] = @intCast(s.count_a & 0xFF);
                dst[wp + 4] = @intCast((count_b >> 16) & 0xFF);
                dst[wp + 5] = @intCast((count_b >> 8) & 0xFF);
                dst[wp + 6] = @intCast(count_b & 0xFF);
                break :blk 7;
            },
            .none => {},
        }
        if (huff.lit) |pre| {
            if (pre.len > 0 and pre.len + 5 < literals.len + 3) {
                if (wp + 5 + pre.len > dst.len) break :blk @as(usize, 0);
                entropy_enc.writeNonCompactChunkHeader(dst[wp..], 4, @intCast(pre.len), @intCast(literals.len));
                @memcpy(dst[wp + 5 ..][0..pre.len], pre);
                break :blk 5 + pre.len;
            }
            break :blk entropy_enc.encodeArrayU8Memcpy(dst[wp..], literals) catch 0;
        }
        if (options.allow_tans32) {
            break :blk entropy_enc.encodeArrayU8(allocator, dst[wp..], literals, options, speed_tradeoff, null, 0, null) catch 0;
        }
        break :blk entropy_enc.encodeArrayU8Memcpy(dst[wp..], literals) catch 0;
    };
    wp += lit_n;

    // ── Emit tokens ──────────────────────────────────────────────────────
    const tok_n = blk_tok: {
        switch (tok_override) {
            .primary => |p| {
                if (wp + 4 + p.combined_stream.len > dst.len) return 0;
                dst[wp] = 0x70;
                dst[wp + 1] = @intCast((p.count_a >> 16) & 0xFF);
                dst[wp + 2] = @intCast((p.count_a >> 8) & 0xFF);
                dst[wp + 3] = @intCast(p.count_a & 0xFF);
                @memcpy(dst[wp + 4 ..][0..p.combined_stream.len], p.combined_stream);
                break :blk_tok 4 + p.combined_stream.len;
            },
            .secondary => |s| {
                if (wp + 7 > dst.len) return 0;
                const count_b: u32 = @intCast(tokens.len);
                dst[wp] = 0x50;
                dst[wp + 1] = @intCast((s.count_a >> 16) & 0xFF);
                dst[wp + 2] = @intCast((s.count_a >> 8) & 0xFF);
                dst[wp + 3] = @intCast(s.count_a & 0xFF);
                dst[wp + 4] = @intCast((count_b >> 16) & 0xFF);
                dst[wp + 5] = @intCast((count_b >> 8) & 0xFF);
                dst[wp + 6] = @intCast(count_b & 0xFF);
                break :blk_tok 7;
            },
            .none => {},
        }
        if (huff.tok) |pre| {
            if (pre.len > 0 and pre.len + 5 < tokens.len + 3) {
                if (wp + 5 + pre.len > dst.len) return 0;
                entropy_enc.writeNonCompactChunkHeader(dst[wp..], 4, @intCast(pre.len), @intCast(tokens.len));
                @memcpy(dst[wp + 5 ..][0..pre.len], pre);
                break :blk_tok 5 + pre.len;
            }
            break :blk_tok entropy_enc.encodeArrayU8Memcpy(dst[wp..], tokens) catch return 0;
        }
        break :blk_tok entropy_enc.encodeArrayU8(allocator, dst[wp..], tokens, options, speed_tradeoff, null, 0, null) catch return 0;
    };
    wp += tok_n;

    if (cmd_stream2_data) |cs2d| {
        if (wp + 2 > dst.len) return 0;
        dst[wp] = cs2d[0];
        dst[wp + 1] = cs2d[1];
        wp += 2;
    }

    // ── Emit off16 (split hi/lo, entropy-encode each plane) ──────────────
    if (off16hi_override != .none) {
        // Paired primary carries the combined hi stream; the lo stream is
        // per-unit. The pre-pass guarantees off16_count >= 32.
        const lo_bytes = allocator.alloc(u8, off16_count) catch return 0;
        defer allocator.free(lo_bytes);
        for (0..off16_count) |i| lo_bytes[i] = off16_data[i * 2];

        if (wp + 2 > dst.len) return 0;
        std.mem.writeInt(u16, dst[wp..][0..2], fast_constants.entropy_coded_16_marker, .little);
        wp += 2;
        switch (off16hi_override) {
            .primary => |p| {
                if (wp + 4 + p.combined_stream.len > dst.len) return 0;
                dst[wp] = 0x70;
                dst[wp + 1] = @intCast((p.count_a >> 16) & 0xFF);
                dst[wp + 2] = @intCast((p.count_a >> 8) & 0xFF);
                dst[wp + 3] = @intCast(p.count_a & 0xFF);
                @memcpy(dst[wp + 4 ..][0..p.combined_stream.len], p.combined_stream);
                wp += 4 + p.combined_stream.len;
            },
            .secondary => |s| {
                if (wp + 7 > dst.len) return 0;
                const count_b: u32 = @intCast(off16_count);
                dst[wp] = 0x50;
                dst[wp + 1] = @intCast((s.count_a >> 16) & 0xFF);
                dst[wp + 2] = @intCast((s.count_a >> 8) & 0xFF);
                dst[wp + 3] = @intCast(s.count_a & 0xFF);
                dst[wp + 4] = @intCast((count_b >> 16) & 0xFF);
                dst[wp + 5] = @intCast((count_b >> 8) & 0xFF);
                dst[wp + 6] = @intCast(count_b & 0xFF);
                wp += 7;
            },
            .none => unreachable,
        }
        entropy_enc.measureStream(allocator, 3, lo_bytes);
        const lo_n = entropy_enc.encodeArrayU8(allocator, dst[wp..], lo_bytes, options, speed_tradeoff, null, 0, null) catch return 0;
        wp += lo_n;
    } else if (off16_count >= 32) {
        if (huff.off16_hi != null and huff.off16_lo != null) {
            // GPU-Huffman path. Per-plane chunk-type=4 body beats a memcpy
            // chunk (3 B hdr + raw) when `body + 2 < count`. Skip the split
            // entirely if the combined coded size doesn't beat raw off16.
            const hi_body = huff.off16_hi.?;
            const lo_body = huff.off16_lo.?;
            const hi_use_huff = hi_body.len + 2 < off16_count;
            const lo_use_huff = lo_body.len + 2 < off16_count;
            const hi_chunk_len: usize = if (hi_use_huff) 5 + hi_body.len else off16_count + 3;
            const lo_chunk_len: usize = if (lo_use_huff) 5 + lo_body.len else off16_count + 3;
            const split_total = hi_chunk_len + lo_chunk_len;
            if (split_total < off16_bytes) {
                if (wp + 2 + split_total > dst.len) return 0;
                std.mem.writeInt(u16, dst[wp..][0..2], fast_constants.entropy_coded_16_marker, .little);
                wp += 2;
                if (hi_use_huff) {
                    entropy_enc.writeNonCompactChunkHeader(dst[wp..], 4, @intCast(hi_body.len), @intCast(off16_count));
                    @memcpy(dst[wp + 5 ..][0..hi_body.len], hi_body);
                    wp += 5 + hi_body.len;
                } else {
                    dst[wp] = @intCast((off16_count >> 16) & 0xFF);
                    dst[wp + 1] = @intCast((off16_count >> 8) & 0xFF);
                    dst[wp + 2] = @intCast(off16_count & 0xFF);
                    for (0..off16_count) |i| dst[wp + 3 + i] = off16_data[i * 2 + 1];
                    wp += off16_count + 3;
                }
                if (lo_use_huff) {
                    entropy_enc.writeNonCompactChunkHeader(dst[wp..], 4, @intCast(lo_body.len), @intCast(off16_count));
                    @memcpy(dst[wp + 5 ..][0..lo_body.len], lo_body);
                    wp += 5 + lo_body.len;
                } else {
                    dst[wp] = @intCast((off16_count >> 16) & 0xFF);
                    dst[wp + 1] = @intCast((off16_count >> 8) & 0xFF);
                    dst[wp + 2] = @intCast(off16_count & 0xFF);
                    for (0..off16_count) |i| dst[wp + 3 + i] = off16_data[i * 2];
                    wp += off16_count + 3;
                }
            } else {
                if (wp + 2 + off16_bytes > dst.len) return 0;
                std.mem.writeInt(u16, dst[wp..][0..2], @intCast(off16_count), .little);
                wp += 2;
                @memcpy(dst[wp..][0..off16_bytes], off16_data);
                wp += off16_bytes;
            }
        } else {
            // CPU-Huffman path. The byte-exact output matches the
            // GPU-Huffman path above, which is why the GPU decoder accepts
            // a frame from either producer.
            const split_buf = allocator.alloc(u8, off16_count * 2) catch return 0;
            defer allocator.free(split_buf);
            const lo_bytes = split_buf[0..off16_count];
            const hi_bytes = split_buf[off16_count..][0..off16_count];
            for (0..off16_count) |i| {
                lo_bytes[i] = off16_data[i * 2];
                hi_bytes[i] = off16_data[i * 2 + 1];
            }
            const split_enc = allocator.alloc(u8, off16_bytes + 512) catch return 0;
            defer allocator.free(split_enc);
            const hi_n = entropy_enc.encodeArrayU8Huffman(split_enc, hi_bytes) catch return 0;
            const lo_n = entropy_enc.encodeArrayU8Huffman(split_enc[hi_n..], lo_bytes) catch return 0;
            const split_total = hi_n + lo_n;
            if (split_total < off16_bytes) {
                if (wp + 2 + split_total > dst.len) return 0;
                std.mem.writeInt(u16, dst[wp..][0..2], fast_constants.entropy_coded_16_marker, .little);
                wp += 2;
                @memcpy(dst[wp..][0..split_total], split_enc[0..split_total]);
                wp += split_total;
            } else {
                if (wp + 2 + off16_bytes > dst.len) return 0;
                std.mem.writeInt(u16, dst[wp..][0..2], @intCast(off16_count), .little);
                wp += 2;
                @memcpy(dst[wp..][0..off16_bytes], off16_data);
                wp += off16_bytes;
            }
        }
    } else {
        if (wp + 2 + off16_bytes > dst.len) return 0;
        std.mem.writeInt(u16, dst[wp..][0..2], @intCast(off16_count), .little);
        wp += 2;
        @memcpy(dst[wp..][0..off16_bytes], off16_data);
        wp += off16_bytes;
    }

    // ── Emit off32 and length verbatim ───────────────────────────────────
    if (wp + 3 + off32_extra + off32_byte_count > dst.len) return 0;
    @memcpy(dst[wp..][0..3], off32_hdr);
    wp += 3;
    if (off32_extra > 0) {
        @memcpy(dst[wp..][0..off32_extra], raw[rp - off32_byte_count - off32_extra ..][0..off32_extra]);
        wp += off32_extra;
    }
    if (off32_byte_count > 0) {
        @memcpy(dst[wp..][0..off32_byte_count], raw[rp - off32_byte_count ..][0..off32_byte_count]);
        wp += off32_byte_count;
    }

    if (wp + length_data.len > dst.len) return 0;
    @memcpy(dst[wp..][0..length_data.len], length_data);
    wp += length_data.len;

    return wp;
}
