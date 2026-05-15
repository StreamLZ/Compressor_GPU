//! Frame-wide shared probability tables for the GPU 32-lane tANS pipeline.
//!
//! Phase 2 specialisation: instead of every sub-chunk embedding its own
//! probability table, the encoder builds 4 frame-wide histograms (one
//! per stream type — literals, tokens, off16-hi, off16-lo), normalises
//! each to a single shared probability table, and writes those 4 tables
//! into the frame header. Sub-chunks then emit chunk_type=3 streams
//! that omit the embedded table; the decoder builds 4 decode LUTs once
//! per frame and reuses them for every sub-chunk.
//!
//! This module is CPU-only — the build work runs once per frame and is
//! cheap (~bytes of data to count, ~256-entry heap-based normalise).

const std = @import("std");
const ByteHistogram = @import("../entropy/byte_histogram.zig").ByteHistogram;
const tans_enc = @import("../entropy/tans_encoder.zig");
const frame_format = @import("../../format/frame_format.zig");

const CompressChunkDesc = @import("gpu_encoder.zig").CompressChunkDesc;

pub const StreamType = enum(u8) {
    lit = 0,
    tok = 1,
    off16_hi = 2,
    off16_lo = 3,
};

/// Encoding table built from a shared probability table. Suitable for
/// uploading to GPU and consuming inside slzTans32EncodeSharedKernel.
pub const Tans32EncodeTable = extern struct {
    /// 256 × 8 byte TansEncEntry. Kept extern-compatible so the GPU
    /// kernel can map the same layout directly.
    te_packed: [256 * 8]u8 = @splat(0),
    /// 2048 × u16 next-state table.
    te_data: [2048]u16 = @splat(0),
    log_table_bits: u32 = 0,
    used_symbols: u32 = 0,
    _pad: u32 = 0,
};

pub const SharedLutBuildResult = struct {
    table_bytes: []u8, // bit-packed probability table; caller owns
    enc_table: Tans32EncodeTable,
};

/// Emit the FSE-style bit-packed probability table that the GPU
/// decoder (and our 32-lane kernel) consumes. Mirrors the inline code
/// in tans_encoder.encodeArrayU8Tans32 at the per-stream level.
/// Returns bytes written into `table_buf`.
fn emitTans32ProbabilityTable(
    table_buf: []u8,
    weights: []const u32,
    num_symbols: usize,
    log_table_bits: u32,
) usize {
    var wp: usize = 0;
    table_buf[wp] = @intCast(log_table_bits);
    wp = 1;

    var tbl_bits: u64 = 0;
    var tbl_bit_count: u32 = 0;
    var remaining: i32 = @as(i32, 1) << @intCast(log_table_bits);
    var tbl_sym: usize = 0;

    while (remaining > 0 and tbl_sym < num_symbols) {
        const w: i32 = @intCast(weights[tbl_sym]);
        const value: u32 = @intCast(w + 1);

        const rem_plus_1: u32 = @intCast(remaining + 1);
        const log2_rem: u32 = if (rem_plus_1 > 1) (31 - @clz(rem_plus_1)) else 0;
        const nb: u32 = log2_rem + 1;
        const half: u32 = @as(u32, 1) << @intCast(nb - 1);
        const max_val: u32 = (half << 1) - @as(u32, @intCast(remaining)) - 2;

        if (value < max_val) {
            tbl_bits |= @as(u64, value) << @intCast(tbl_bit_count);
            tbl_bit_count += nb - 1;
        } else if (value < half) {
            tbl_bits |= @as(u64, value) << @intCast(tbl_bit_count);
            tbl_bit_count += nb;
        } else {
            tbl_bits |= @as(u64, value + max_val) << @intCast(tbl_bit_count);
            tbl_bit_count += nb;
        }

        while (tbl_bit_count >= 8) {
            table_buf[wp] = @intCast(tbl_bits & 0xFF);
            tbl_bits >>= 8;
            tbl_bit_count -= 8;
            wp += 1;
        }

        if (w > 0) remaining -= w;
        tbl_sym += 1;
    }

    if (tbl_bit_count > 0) {
        table_buf[wp] = @intCast(tbl_bits & 0xFF);
        wp += 1;
    }

    return wp;
}

/// Build the shared LUT for one stream type from its aggregated
/// frame-wide histogram. Returns null when the histogram has fewer
/// than 2 unique symbols (kernel rejects; caller should disable
/// shared-LUT mode for that stream type and fall back to embedded).
pub fn buildSharedLut(
    allocator: std.mem.Allocator,
    histo: *const ByteHistogram,
    total_count: u32,
) ?SharedLutBuildResult {
    if (total_count < 64) return null;

    var num_symbols: usize = 256;
    while (num_symbols > 0 and histo.count[num_symbols - 1] == 0) num_symbols -= 1;
    if (num_symbols <= 1) return null;

    // log_table_bits = clamp(ilog2Round(total_count) - 2, 8, 11)
    var log_table_bits: u32 = ilog2Round(total_count);
    if (log_table_bits >= 2) log_table_bits -= 2 else log_table_bits = 0;
    if (log_table_bits < 8) log_table_bits = 8;
    if (log_table_bits > 11) log_table_bits = 11;

    var weights_buf: [256]u32 = @splat(0);
    var histo_copy = histo.*;
    const used = tans_enc.tansNormalizeCounts(
        &weights_buf,
        @as(u32, 1) << @intCast(log_table_bits),
        &histo_copy,
        total_count,
        num_symbols,
    );
    if (used <= 1) return null;

    var table_buf_static: [512]u8 = @splat(0);
    const table_len = emitTans32ProbabilityTable(&table_buf_static, &weights_buf, num_symbols, log_table_bits);

    const table_owned = allocator.alloc(u8, table_len) catch return null;
    @memcpy(table_owned, table_buf_static[0..table_len]);

    var enc: Tans32EncodeTable = .{};
    var te_typed: [256]tans_enc.TansEncEntry = undefined;
    tans_enc.tansInitTable(&te_typed, @ptrCast(&enc.te_data), &weights_buf, num_symbols, log_table_bits);
    // Pack te_typed into the extern-friendly te_packed (8 bytes per entry
    // with explicit layout — matches `struct TansEncEntry` in the kernel).
    var i: usize = 0;
    while (i < 256) : (i += 1) {
        const e = te_typed[i];
        const base: usize = i * 8;
        // 4 bytes base_offset (i32 LE), 2 bytes thres (u16 LE), 1 byte num_bits, 1 byte pad
        std.mem.writeInt(i32, enc.te_packed[base..][0..4], e.base_offset, .little);
        std.mem.writeInt(u16, enc.te_packed[base + 4 ..][0..2], e.thres, .little);
        enc.te_packed[base + 6] = e.num_bits;
        enc.te_packed[base + 7] = 0;
    }
    enc.log_table_bits = log_table_bits;
    enc.used_symbols = @intCast(used);

    return .{
        .table_bytes = table_owned,
        .enc_table = enc,
    };
}

/// Aggregate frame-wide histograms for the four GPU entropy streams by
/// walking the raw GPU output once. For each sub-chunk, parse its
/// header to locate the lit/tok/off16-hi/off16-lo bytes and count them
/// into the matching histogram.
///
/// Returns the total byte count for each stream type (for normalisation).
pub const FrameHistograms = struct {
    lit: ByteHistogram = .{},
    tok: ByteHistogram = .{},
    off16_hi: ByteHistogram = .{},
    off16_lo: ByteHistogram = .{},
    lit_total: u32 = 0,
    tok_total: u32 = 0,
    off16_total: u32 = 0,
};

pub fn buildFrameHistograms(
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) FrameHistograms {
    var fh: FrameHistograms = .{};
    for (chunk_descs, 0..) |cd, i| {
        const cs = comp_sizes[i];
        const base: u32 = cd.dst_offset;
        const init_b: u32 = if (cd.is_first != 0) 8 else 0;
        if (cs < init_b + 3) continue;

        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        const lit_start: u32 = lit_hdr + 3;
        if (lit_start + lit_count > base + cs) continue;
        // Accumulate (countBytes resets, which we don't want).
        for (output[lit_start..][0..lit_count]) |b| fh.lit.count[b] += 1;
        fh.lit_total += lit_count;

        const tok_hdr: u32 = lit_start + lit_count;
        if (tok_hdr + 3 > base + cs) continue;
        const tok_count: u32 =
            (@as(u32, output[tok_hdr]) << 16) |
            (@as(u32, output[tok_hdr + 1]) << 8) |
            @as(u32, output[tok_hdr + 2]);
        const tok_start: u32 = tok_hdr + 3;
        if (tok_start + tok_count > base + cs) continue;
        for (output[tok_start..][0..tok_count]) |b| fh.tok.count[b] += 1;
        fh.tok_total += tok_count;

        const cmd2_size: u32 = if (cd.src_size > 0x10000) 2 else 0;
        const off16_hdr: u32 = tok_start + tok_count + cmd2_size;
        if (off16_hdr + 2 > base + cs) continue;
        const off16_count: u32 =
            @as(u32, output[off16_hdr]) | (@as(u32, output[off16_hdr + 1]) << 8);
        if (off16_count < 32) continue; // matches CPU encoder >=32 gate
        const off16_data: u32 = off16_hdr + 2;
        if (off16_data + off16_count * 2 > base + cs) continue;

        // Interleaved lo/hi byte planes.
        var j: u32 = 0;
        while (j < off16_count) : (j += 1) {
            fh.off16_lo.count[output[off16_data + j * 2]] += 1;
            fh.off16_hi.count[output[off16_data + j * 2 + 1]] += 1;
        }
        fh.off16_total += off16_count;
    }
    return fh;
}

// Local ilog2Round mirroring the one used in tans_encoder.
fn ilog2Round(v: u32) u32 {
    if (v == 0) return 0;
    const bsr: u32 = 31 - @clz(v);
    const lower: u32 = @as(u32, 1) << @intCast(bsr);
    const upper: u32 = lower << 1;
    return if (v - lower >= upper - v) bsr + 1 else bsr;
}
