//! Host-side assembly of the SLZ1 wire format from raw GPU encoder output.
//!
//! The GPU LZ kernel emits each sub-chunk as a packed sequence of raw byte
//! streams: literals, tokens, off16, off32, lengths. This module re-encodes
//! that raw payload into the final wire format by splicing in the Huffman
//! bodies the GPU Huffman kernel produced, falling back to memcpy or
//! CPU-side Huffman when the GPU body did not beat raw.
//!
//! Output is identical to what the device-resident assembly kernel would
//! have produced; this path runs only when the caller did not opt into the
//! pure-D2D assembly (no device output buffer, or one or more sub-chunks
//! failed to compress below their raw size).

const std = @import("std");

const fast_constants = @import("fast/fast_constants.zig");
const entropy_enc = @import("entropy/entropy_encoder.zig");

/// Pre-encoded chunk-type=4 Huffman bodies the GPU may hand us per
/// sub-chunk. Each non-null field is a body (no 5-byte chunk header —
/// this module adds it). A null field means the GPU Huffman pass either
/// did not run or its output did not beat raw; the literal / token / off16
/// emitter falls back accordingly.
pub const GpuHuffStreams = struct {
    lit: ?[]const u8 = null,
    tok: ?[]const u8 = null,
    off16_hi: ?[]const u8 = null,
    off16_lo: ?[]const u8 = null,
};

/// Re-encode a raw GPU sub-chunk payload into the SLZ1 wire format,
/// writing into `dst`. Returns the byte count written, or 0 on any
/// failure (typically a destination overflow — the caller falls back to
/// emitting the raw sub-chunk as-is).
///
/// `init_bytes` carries the 8-byte raw header on the very first sub-chunk
/// of the frame; the GPU LZ kernel emits these for chunk 0 only.
/// `src_size` is the decompressed sub-chunk length and gates the
/// `cmd_stream2_offset` presence test (set for sub-chunks > 64 KB, which
/// the current sc_group_size choices never produce, but kept for
/// completeness).
pub fn reencodeGpuWithEntropy(
    allocator: std.mem.Allocator,
    raw: []const u8,
    dst: []u8,
    init_bytes: usize,
    src_size: usize,
    huff: GpuHuffStreams,
) !usize {
    var rp: usize = 0;
    var wp: usize = 0;

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

    if (rp + 3 > raw.len) return 0;
    const token_count: usize = (@as(usize, raw[rp]) << 16) | (@as(usize, raw[rp + 1]) << 8) | raw[rp + 2];
    rp += 3;
    if (rp + token_count > raw.len) return 0;
    const tokens = raw[rp..][0..token_count];
    rp += token_count;

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
    // Each off32 entry is 3 or 4 bytes; the high byte of the third byte
    // sets `>= 0xC0` to flag the 4-byte form. Scan to size precisely.
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
    wp += try emitHuffOrMemcpy(dst[wp..], literals, huff.lit);

    // ── Emit tokens ──────────────────────────────────────────────────────
    wp += try emitHuffOrMemcpy(dst[wp..], tokens, huff.tok);

    if (cmd_stream2_data) |cs2d| {
        if (wp + 2 > dst.len) return 0;
        dst[wp] = cs2d[0];
        dst[wp + 1] = cs2d[1];
        wp += 2;
    }

    // ── Emit off16 (split hi/lo, entropy-encode each plane) ──────────────
    wp += try emitOff16(allocator, dst[wp..], off16_data, off16_count, off16_bytes, huff);

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

/// Splice a chunk-type=4 Huffman body (with its 5-byte non-compact header)
/// when the GPU produced one and it beats raw; otherwise emit the raw
/// bytes via the chunk-type=0 memcpy header.
fn emitHuffOrMemcpy(dst: []u8, src: []const u8, huff_body: ?[]const u8) !usize {
    if (huff_body) |body| {
        // chunk-type=4 wins over a raw chunk (3-byte hdr + count bytes)
        // exactly when 5 + body.len < 3 + src.len, i.e. body.len + 2 < src.len.
        if (body.len > 0 and body.len + 2 < src.len) {
            if (5 + body.len > dst.len) return 0;
            entropy_enc.writeNonCompactChunkHeader(dst, 4, @intCast(body.len), @intCast(src.len));
            @memcpy(dst[5..][0..body.len], body);
            return 5 + body.len;
        }
    }
    return entropy_enc.encodeArrayU8Memcpy(dst, src) catch 0;
}

/// Emit the off16 stream. When off16_count >= 32, try splitting into hi
/// and lo byte planes and entropy-coding each: prefer the GPU's Huffman
/// bodies, fall back to CPU Huffman, fall back to raw if neither wins.
fn emitOff16(
    allocator: std.mem.Allocator,
    dst: []u8,
    off16_data: []const u8,
    off16_count: usize,
    off16_bytes: usize,
    huff: GpuHuffStreams,
) !usize {
    if (off16_count < 32) {
        if (2 + off16_bytes > dst.len) return 0;
        std.mem.writeInt(u16, dst[0..2], @intCast(off16_count), .little);
        @memcpy(dst[2..][0..off16_bytes], off16_data);
        return 2 + off16_bytes;
    }

    if (huff.off16_hi != null and huff.off16_lo != null) {
        return emitOff16GpuHuffman(dst, off16_data, off16_count, off16_bytes, huff.off16_hi.?, huff.off16_lo.?);
    }
    return emitOff16CpuHuffman(allocator, dst, off16_data, off16_count, off16_bytes);
}

fn emitOff16GpuHuffman(
    dst: []u8,
    off16_data: []const u8,
    off16_count: usize,
    off16_bytes: usize,
    hi_body: []const u8,
    lo_body: []const u8,
) !usize {
    // A chunk-type=4 plane (5 B hdr + body) beats a raw plane chunk
    // (3 B hdr + count bytes) when body + 2 < count. Decide per plane.
    const hi_use_huff = hi_body.len + 2 < off16_count;
    const lo_use_huff = lo_body.len + 2 < off16_count;
    const hi_chunk_len: usize = if (hi_use_huff) 5 + hi_body.len else off16_count + 3;
    const lo_chunk_len: usize = if (lo_use_huff) 5 + lo_body.len else off16_count + 3;
    const split_total = hi_chunk_len + lo_chunk_len;

    if (split_total >= off16_bytes) {
        if (2 + off16_bytes > dst.len) return 0;
        std.mem.writeInt(u16, dst[0..2], @intCast(off16_count), .little);
        @memcpy(dst[2..][0..off16_bytes], off16_data);
        return 2 + off16_bytes;
    }

    if (2 + split_total > dst.len) return 0;
    std.mem.writeInt(u16, dst[0..2], fast_constants.entropy_coded_16_marker, .little);
    var wp: usize = 2;
    wp += writePlane(dst[wp..], off16_data, off16_count, hi_use_huff, hi_body, 1);
    wp += writePlane(dst[wp..], off16_data, off16_count, lo_use_huff, lo_body, 0);
    return wp;
}

fn emitOff16CpuHuffman(
    allocator: std.mem.Allocator,
    dst: []u8,
    off16_data: []const u8,
    off16_count: usize,
    off16_bytes: usize,
) !usize {
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

    if (split_total >= off16_bytes) {
        if (2 + off16_bytes > dst.len) return 0;
        std.mem.writeInt(u16, dst[0..2], @intCast(off16_count), .little);
        @memcpy(dst[2..][0..off16_bytes], off16_data);
        return 2 + off16_bytes;
    }
    if (2 + split_total > dst.len) return 0;
    std.mem.writeInt(u16, dst[0..2], fast_constants.entropy_coded_16_marker, .little);
    @memcpy(dst[2..][0..split_total], split_enc[0..split_total]);
    return 2 + split_total;
}

/// Emit one off16 byte plane: either a chunk-type=4 Huffman block or a
/// chunk-type=0 raw block. `plane_offset` is 0 for the lo plane (even
/// bytes) and 1 for the hi plane (odd bytes).
fn writePlane(
    dst: []u8,
    off16_data: []const u8,
    off16_count: usize,
    use_huff: bool,
    body: []const u8,
    plane_offset: usize,
) usize {
    if (use_huff) {
        entropy_enc.writeNonCompactChunkHeader(dst, 4, @intCast(body.len), @intCast(off16_count));
        @memcpy(dst[5..][0..body.len], body);
        return 5 + body.len;
    }
    dst[0] = @intCast((off16_count >> 16) & 0xFF);
    dst[1] = @intCast((off16_count >> 8) & 0xFF);
    dst[2] = @intCast(off16_count & 0xFF);
    for (0..off16_count) |i| dst[3 + i] = off16_data[i * 2 + plane_offset];
    return off16_count + 3;
}
