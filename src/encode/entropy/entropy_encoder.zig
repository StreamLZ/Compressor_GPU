//! High-level entropy encoder wrapper.
//! Used by: Fast and High codecs
//!
//! Two primary paths:
//!
//!   * `encodeArrayU8Memcpy` — raw memcpy with a 3-byte BE24 chunk type 0
//!     header. Always valid, always falls back to this.
//!   * `encodeArrayU8` — try tANS (if allowed), pick the cheaper of
//!     {tANS, memcpy}. Writes either a 5-byte non-compact header (tANS)
//!     or the 3-byte memcpy header.
//!
//! The returned byte count includes the chunk header. Negative / error
//! returns mean "doesn't fit" (caller should fall back elsewhere).

const std = @import("std");
const hist_mod = @import("byte_histogram.zig");
const tans = @import("tans_encoder.zig");

const ByteHistogram = hist_mod.ByteHistogram;

/// Entropy option bit flags.
pub const EntropyOptions = packed struct(u8) {
    allow_tans: bool = false,
    allow_rle_entropy: bool = false,
    allow_double_huffman: bool = false,
    allow_rle: bool = false,
    allow_multi_array: bool = false,
    supports_new_huffman: bool = false,
    allow_tans32: bool = false,
    supports_short_memset: bool = false,

    pub fn raw(self: EntropyOptions) u8 {
        return @bitCast(self);
    }
};

pub const EncodeError = error{
    DestinationTooSmall,
    EntropyNotBeneficial,
} || std.mem.Allocator.Error;

/// Write a 3-byte BE24 chunk type 0 memcpy header + the raw bytes.
/// Returns total bytes written.
pub fn encodeArrayU8Memcpy(dst: []u8, src: []const u8) EncodeError!usize {
    if (src.len + 3 > dst.len) return error.DestinationTooSmall;
    dst[0] = @intCast((src.len >> 16) & 0xFF);
    dst[1] = @intCast((src.len >> 8) & 0xFF);
    dst[2] = @intCast(src.len & 0xFF);
    @memcpy(dst[3 .. 3 + src.len], src);
    return src.len + 3;
}

/// Write a 5-byte non-compact chunk header. `decompressed_size` and
/// `compressed_size` must each fit in 18 bits; `chunk_type` in 4 bits.
pub fn writeNonCompactChunkHeader(dst: []u8, chunk_type: u8, compressed_size: u32, decompressed_size: u32) void {
    const dst_minus_1: u32 = decompressed_size - 1;
    dst[0] = @intCast((@as(u32, chunk_type) << 4) | ((dst_minus_1 >> 14) & 0xF));
    const bits: u32 = compressed_size | ((dst_minus_1 & 0x3FFF) << 18);
    dst[1] = @intCast((bits >> 24) & 0xFF);
    dst[2] = @intCast((bits >> 16) & 0xFF);
    dst[3] = @intCast((bits >> 8) & 0xFF);
    dst[4] = @intCast(bits & 0xFF);
}

/// Convert a fresh 5-byte non-compact header (or Type-0 3-byte memcpy
/// header) into a compact variant when possible. Decrements `cost_ptr`
/// by 1 (memcpy shrink) or 2 (compressed shrink) when the header compacts.
/// Returns the new total byte count.
fn makeCompactChunkHdr(dst: []u8, total_n: usize, cost_ptr: ?*f32) usize {
    const chunk_type: u32 = (dst[0] >> 4) & 0x7;
    if (chunk_type == 0) {
        // Memcpy: try a compact 2-byte header.
        const src_size: u32 = (@as(u32, dst[0]) << 16) | (@as(u32, dst[1]) << 8) | @as(u32, dst[2]);
        if (src_size <= 0xFFF) {
            const hdr_val: u32 = 0x8000 | src_size;
            const h0: u8 = @intCast((hdr_val >> 8) & 0xFF);
            const h1: u8 = @intCast(hdr_val & 0xFF);
            // Shift payload 1 byte left (from dst[3..] to dst[2..]).
            std.mem.copyForwards(u8, dst[2 .. 2 + src_size], dst[3 .. 3 + src_size]);
            dst[0] = h0;
            dst[1] = h1;
            if (cost_ptr) |c| c.* -= 1;
            return total_n - 1;
        }
        return total_n;
    }

    // Non-memcpy chunk: try a compact 3-byte header.
    const bits5: u32 = (@as(u32, dst[1]) << 24) |
        (@as(u32, dst[2]) << 16) |
        (@as(u32, dst[3]) << 8) |
        @as(u32, dst[4]);
    const src_size: u32 = bits5 & 0x3FFFF;
    const dst_size: u32 = (((bits5 >> 18) | (@as(u32, dst[0]) << 14)) & 0x3FFFF) + 1;
    if (dst_size <= src_size) return total_n;
    const delta: u32 = dst_size - src_size - 1;
    if (src_size <= 0x3FF and delta <= 0x3FF) {
        const bits3: u32 = src_size |
            (delta << 10) |
            (chunk_type << 20) |
            (@as(u32, 1) << 23);
        const b0: u8 = @intCast((bits3 >> 16) & 0xFF);
        const b1: u8 = @intCast((bits3 >> 8) & 0xFF);
        const b2: u8 = @intCast(bits3 & 0xFF);
        // Shift payload 2 bytes left (from dst[5..] to dst[3..]).
        std.mem.copyForwards(u8, dst[3 .. 3 + src_size], dst[5 .. 5 + src_size]);
        dst[0] = b0;
        dst[1] = b1;
        dst[2] = b2;
        if (cost_ptr) |c| c.* -= 2;
        return total_n - 2;
    }
    return total_n;
}

/// Encode a byte array picking the cheaper of tANS and memcpy. Writes
/// the chunk header (non-compact 5 bytes for compressed, 3 bytes for
/// memcpy). Returns the total byte count written.
///
/// `speed_tradeoff`
/// biases tANS vs memcpy cost comparison, `cost_out` receives the
/// rate-distortion cost of the chosen encoding (used by callers to
/// pick between multiple candidates), `level` gates level-dependent
/// algorithm choices. `histo_out` optionally receives the computed
/// histogram so callers can reuse it.
pub fn encodeArrayU8(
    allocator: std.mem.Allocator,
    dst: []u8,
    src: []const u8,
    options: EntropyOptions,
    speed_tradeoff: f32,
    cost_out: ?*f32,
    level: u32,
    histo_out: ?*ByteHistogram,
) EncodeError!usize {
    if (src.len <= 32) {
        if (histo_out) |h| h.countBytes(src);
        if (cost_out) |c| c.* = @floatFromInt(src.len + 3);
        return encodeArrayU8Memcpy(dst, src);
    }

    var histo: ByteHistogram = .{};
    histo.countBytes(src);
    if (histo_out) |h| h.* = histo;

    return encodeArrayU8CoreWithHisto(allocator, dst, src, &histo, options, speed_tradeoff, cost_out, level);
}

/// Encode `src` as a Huffman chunk_type=4 block when that beats a raw
/// memcpy chunk; otherwise emit a memcpy chunk. Unlike `encodeArrayU8`
/// this never attempts tANS — used for GPU off16 hi/lo streams, which
/// the GPU tANS off16 encoder mis-encodes (see the gpu-silesia-off16-bug
/// note) while CPU Huffman GPU-decodes byte-exact.
pub fn encodeArrayU8Huffman(dst: []u8, src: []const u8) EncodeError!usize {
    if (src.len >= 32 and dst.len > 5 + 128 + 9) {
        const huff = @import("huffman_encoder.zig");
        if (huff.encodeBlock(dst, src) catch null) |n| {
            if (n < src.len + 3) return n;
        }
    }
    return encodeArrayU8Memcpy(dst, src);
}

/// Core encode with pre-computed histogram. Takes the histogram by
/// value (copies internally so the caller's copy is untouched by any
/// in-place adjustments the encoders make).
pub fn encodeArrayU8WithHisto(
    allocator: std.mem.Allocator,
    dst: []u8,
    src: []const u8,
    histo: ByteHistogram,
    options: EntropyOptions,
    speed_tradeoff: f32,
    cost_out: ?*f32,
    level: u32,
) EncodeError!usize {
    if (src.len <= 32) {
        if (cost_out) |c| c.* = @floatFromInt(src.len + 3);
        return encodeArrayU8Memcpy(dst, src);
    }
    var histo_copy = histo;
    return encodeArrayU8CoreWithHisto(allocator, dst, src, &histo_copy, options, speed_tradeoff, cost_out, level);
}

fn encodeArrayU8CoreWithHisto(
    allocator: std.mem.Allocator,
    dst: []u8,
    src: []const u8,
    histo: *ByteHistogram,
    options: EntropyOptions,
    speed_tradeoff: f32,
    cost_out: ?*f32,
    level: u32,
) EncodeError!usize {
    _ = level; // currently unused; reserved for Huffman / multi-array paths

    const memcpy_cost: f32 = @floatFromInt(src.len + 3);

    // SLZ_HUFF_LIT=1 — replace tANS attempts with a Huffman type-4 attempt.
    // Used as a direct A/B vs tANS on the same stream. Falls back to memcpy
    // if Huffman doesn't beat raw or the destination is too small.
    if (std.c.getenv("SLZ_HUFF_LIT") != null and dst.len > 5 + 128 + 9 and src.len >= 32) {
        const huff = @import("huffman_encoder.zig");
        const h_n = huff.encodeBlock(dst, src) catch null;
        if (h_n) |n| {
            if (n < src.len + 3) {
                if (cost_out) |c| c.* = @floatFromInt(n);
                return n;
            }
        }
        if (cost_out) |c| c.* = memcpy_cost;
        return encodeArrayU8Memcpy(dst, src);
    }

    // Try 32-lane tANS if allowed (GPU-optimized, chunk_type=6).
    if (options.allow_tans32 and dst.len > 5 + 128) {
        var tans32_cost: f32 = memcpy_cost;
        const tans32_n = tans.encodeArrayU8Tans32(allocator, dst[5..], src, histo, speed_tradeoff, &tans32_cost, null) catch |err| switch (err) {
            error.TansNotBeneficial, error.TooFewSymbols, error.DestinationTooSmall => null,
            error.BadParameters => return error.DestinationTooSmall,
            error.OutOfMemory => return error.OutOfMemory,
        };
        if (tans32_n) |n| {
            if (n > 0 and n < src.len and tans32_cost < memcpy_cost) {
                writeNonCompactChunkHeader(dst, 6, @intCast(n), @intCast(src.len));
                if (cost_out) |c| c.* = tans32_cost;
                return 5 + n;
            }
        }
    }

    // Try tANS if allowed (5-state interleaved, chunk_type=1).
    if (options.allow_tans and !options.allow_tans32 and dst.len > 5 + 8) {
        var tans_cost: f32 = memcpy_cost;
        const tans_n = tans.encodeArrayU8Tans(allocator, dst[5..], src, histo, speed_tradeoff, &tans_cost) catch |err| switch (err) {
            error.TansNotBeneficial, error.TooFewSymbols, error.DestinationTooSmall => {
                if (cost_out) |c| c.* = memcpy_cost;
                return encodeArrayU8Memcpy(dst, src);
            },
            error.BadParameters => return error.DestinationTooSmall,
            error.OutOfMemory => return error.OutOfMemory,
        };
        if (tans_n > 0 and tans_n < src.len and tans_cost < memcpy_cost) {
            writeNonCompactChunkHeader(dst, 1, @intCast(tans_n), @intCast(src.len));
            if (cost_out) |c| c.* = tans_cost;
            return 5 + tans_n;
        }
    }
    // Fall back to memcpy.
    if (cost_out) |c| c.* = memcpy_cost;
    return encodeArrayU8Memcpy(dst, src);
}

/// Same as `encodeArrayU8` but converts the result to a compact
/// chunk header where possible (saving 1–2 bytes). When the header
/// shrinks, `cost_out` is decremented to reflect the smaller output.
pub fn encodeArrayU8CompactHeader(
    allocator: std.mem.Allocator,
    dst: []u8,
    src: []const u8,
    options: EntropyOptions,
    speed_tradeoff: f32,
    cost_out: ?*f32,
    level: u32,
    histo_out: ?*ByteHistogram,
) EncodeError!usize {
    const n = try encodeArrayU8(allocator, dst, src, options, speed_tradeoff, cost_out, level, histo_out);
    return makeCompactChunkHdr(dst, n, cost_out);
}

/// Encode with a pre-built shared tANS32 context. Produces chunk_type 6
/// WITHOUT embedded LUT — the GPU decoder uses a separate shared LUT upload.
/// Encode with a pre-built shared tANS32 context. Produces chunk_type 6.
/// Uses shared encode table (consistent across all streams in the frame).
/// LUT is NOT embedded — the GPU decoder uses a separately-uploaded shared LUT.
pub fn encodeArrayU8Tans32Shared(
    allocator: std.mem.Allocator,
    dst: []u8,
    src: []const u8,
    shared_ctx: *const tans.Tans32SharedCtx,
) EncodeError!usize {
    if (src.len < 64 or dst.len < 5 + 128) return encodeArrayU8Memcpy(dst, src);

    var histo: ByteHistogram = .{};
    histo.countBytes(src);
    var tans_cost: f32 = @floatFromInt(src.len + 3);
    const n = tans.encodeArrayU8Tans32(allocator, dst[5..], src, &histo, 0.0, &tans_cost, shared_ctx) catch {
        return encodeArrayU8Memcpy(dst, src);
    };
    if (n > 0 and n < src.len) {
        writeNonCompactChunkHeader(dst, 6, @intCast(n), @intCast(src.len));
        return 5 + n;
    }
    return encodeArrayU8Memcpy(dst, src);
}

/// Debug (SLZ_DUMP_STREAMS=1): measure how a raw stream would compress
/// under Huffman vs tANS-32, appending "<kind>,<raw>,<huff>,<tans>" to
/// c:\tmp\slz_streams.csv. kind: 0=lit 1=tok 2=off16hi 3=off16lo. All
/// three sizes include the chunk header; a coder's size falls back to
/// the raw memcpy size when it isn't beneficial. Inert without the env
/// var.
pub fn measureStream(allocator: std.mem.Allocator, kind: u8, src: []const u8) void {
    if (std.c.getenv("SLZ_DUMP_STREAMS") == null) return;
    if (src.len < 32) return;
    const huff = @import("huffman_encoder.zig");
    const cap = src.len * 2 + 4096;
    const buf = allocator.alloc(u8, cap) catch return;
    defer allocator.free(buf);

    var huff_n: usize = src.len + 3;
    if (huff.encodeBlock(buf, src) catch null) |hn| {
        if (hn < huff_n) huff_n = hn;
    }

    var histo: ByteHistogram = .{};
    histo.countBytes(src);
    var tcost: f32 = @floatFromInt(src.len + 3);
    var tans_n: usize = src.len + 3;
    if (tans.encodeArrayU8Tans32(allocator, buf, src, &histo, 0.0, &tcost, null) catch null) |tn| {
        if (tn + 5 < tans_n) tans_n = tn + 5;
    }

    const cio = @cImport({ @cInclude("stdio.h"); });
    const fp = cio.fopen("c:/tmp/slz_streams.csv", "ab") orelse return;
    defer _ = cio.fclose(fp);
    var line: [96]u8 = undefined;
    const s = std.fmt.bufPrint(&line, "{d},{d},{d},{d}\n", .{ kind, src.len, huff_n, tans_n }) catch return;
    _ = cio.fwrite(s.ptr, 1, s.len, fp);
}

