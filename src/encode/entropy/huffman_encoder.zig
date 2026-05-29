//! Canonical Huffman 4-stream encoder, length-limited to 11 bits.
//! Wire format (chunk_type=4, matches gpu_huff_decode_kernel.cu):
//!
//!   [3 or 5 byte standard non-compact chunk header]
//!   [128 B weights — 4 bits/symbol, packed low-nibble-first]
//!   [9 B sub-header — 3 × u24 LE stream sizes; stream 3 size derived]
//!   [stream 0 bits | stream 1 | stream 2 | stream 3]
//!
//! Bits within each byte are MSB-first.
//!
//! Ported from tools/huff_test/huff_ref.c.

const std = @import("std");
const entropy_enc = @import("entropy_encoder.zig");

pub const MAX_CODE_LEN: u8 = 11;
pub const WEIGHTS_BYTES: usize = 128; // 256 symbols × 4 bits / 8 = 128
pub const SUB_HEADER_BYTES: usize = 9; // 3 × u24 LE

pub const EncodeError = error{
    DestinationTooSmall,
    TooFewSymbols,
    StreamTooLarge,
    BitBufferOverflow,
};

// ── Code-length builder (binary heap, returns max length pre-limit) ────
fn buildCodeLengths(hist: *const [256]u32, code_lengths: *[256]u8) struct { max_len: u8, symbols_used: u32 } {
    // Up to 256 leaves + 255 internals = 511 nodes.
    var weights: [512]u32 = @splat(0);
    var parents: [512]i32 = @splat(-1);

    var symbols_used: u32 = 0;
    var idx: [256]u16 = undefined;
    for (0..256) |s| {
        if (hist[s] > 0) {
            weights[s] = hist[s];
            idx[symbols_used] = @intCast(s);
            symbols_used += 1;
        }
    }

    @memset(code_lengths, 0);
    if (symbols_used == 0) return .{ .max_len = 0, .symbols_used = 0 };
    if (symbols_used == 1) {
        code_lengths[idx[0]] = 1;
        return .{ .max_len = 1, .symbols_used = 1 };
    }

    var active: [512]u16 = undefined;
    for (0..symbols_used) |i| active[i] = idx[i];
    var n_active: u32 = symbols_used;
    var next_node: u32 = 256;

    while (n_active > 1) {
        // Find two smallest weights via linear scan.
        var a_pos: u32 = 0;
        var b_pos: u32 = 1;
        if (weights[active[a_pos]] > weights[active[b_pos]]) {
            const t = a_pos;
            a_pos = b_pos;
            b_pos = t;
        }
        var i: u32 = 2;
        while (i < n_active) : (i += 1) {
            const w = weights[active[i]];
            if (w < weights[active[a_pos]]) {
                b_pos = a_pos;
                a_pos = i;
            } else if (w < weights[active[b_pos]]) {
                b_pos = i;
            }
        }
        const a = active[a_pos];
        const b = active[b_pos];

        weights[next_node] = weights[a] + weights[b];
        parents[a] = @intCast(next_node);
        parents[b] = @intCast(next_node);

        const new_pos = if (a_pos < b_pos) a_pos else b_pos;
        const old_pos = if (a_pos < b_pos) b_pos else a_pos;
        active[new_pos] = @intCast(next_node);
        n_active -= 1;
        active[old_pos] = active[n_active];
        next_node += 1;
    }

    // Walk parents from each leaf to compute depth.
    var max_len: u8 = 0;
    for (0..symbols_used) |i| {
        const s = idx[i];
        var depth: u8 = 0;
        var n: i32 = s;
        while (parents[@intCast(n)] != -1) {
            depth += 1;
            n = parents[@intCast(n)];
        }
        code_lengths[s] = depth;
        if (depth > max_len) max_len = depth;
    }
    return .{ .max_len = max_len, .symbols_used = symbols_used };
}

// ── Kraft-preserving height limit ──────────────────────────────────────
fn heightLimit(hist: *const [256]u32, code_lengths: *[256]u8, limit: u8) void {
    const TARGET: u32 = @as(u32, 1) << 30;
    var sum: u32 = 0;
    for (0..256) |s| {
        if (code_lengths[s] > 0) sum += (@as(u32, 1) << @intCast(30 - code_lengths[s]));
    }

    // Demote everything longer than limit.
    for (0..256) |s| {
        if (code_lengths[s] > limit) {
            sum -= (@as(u32, 1) << @intCast(30 - code_lengths[s]));
            code_lengths[s] = limit;
            sum += (@as(u32, 1) << @intCast(30 - limit));
        }
    }

    // Rebalance: lengthen lowest-weight code until Kraft sum == target.
    while (sum > TARGET) {
        var best: i32 = -1;
        var best_w: u32 = std.math.maxInt(u32);
        for (0..256) |s| {
            if (code_lengths[s] > 0 and code_lengths[s] < limit and hist[s] < best_w) {
                best_w = hist[s];
                best = @intCast(s);
            }
        }
        if (best < 0) break;
        const bi: usize = @intCast(best);
        sum -= (@as(u32, 1) << @intCast(30 - code_lengths[bi]));
        code_lengths[bi] += 1;
        sum += (@as(u32, 1) << @intCast(30 - code_lengths[bi]));
    }
}

// ── Canonical code assignment ─────────────────────────────────────────
fn assignCanonicalCodes(code_lengths: *const [256]u8, codes_out: *[256]u32) void {
    var length_count: [MAX_CODE_LEN + 2]u32 = @splat(0);
    for (0..256) |s| length_count[code_lengths[s]] += 1;
    length_count[0] = 0;

    var next_code: [MAX_CODE_LEN + 2]u32 = @splat(0);
    var code: u32 = 0;
    for (1..MAX_CODE_LEN + 2) |L| {
        code = (code + length_count[L - 1]) << 1;
        next_code[L] = code;
    }

    for (0..256) |s| {
        const len = code_lengths[s];
        if (len != 0) {
            codes_out[s] = next_code[len];
            next_code[len] += 1;
        } else {
            codes_out[s] = 0;
        }
    }
}

// ── Single-stream bit packer ──────────────────────────────────────────
// Packs codewords MSB-first into `out`. Returns bytes written or error.
fn encodeStream(
    in: []const u8,
    code_lengths: *const [256]u8,
    codes: *const [256]u32,
    out: []u8,
) EncodeError!usize {
    var bit_buf: u64 = 0;
    var bit_count: u6 = 0;
    var out_pos: usize = 0;

    for (in) |sym| {
        const code = codes[sym];
        const len: u6 = @intCast(code_lengths[sym]);
        if (len == 0) return EncodeError.BitBufferOverflow; // symbol with zero code length but appears in input
        if (@as(u32, bit_count) + @as(u32, len) > 64) return EncodeError.BitBufferOverflow;
        bit_buf = (bit_buf << len) | (@as(u64, code) & ((@as(u64, 1) << len) - 1));
        bit_count += len;
        while (bit_count >= 8) {
            if (out_pos >= out.len) return EncodeError.DestinationTooSmall;
            bit_count -= 8;
            out[out_pos] = @intCast((bit_buf >> bit_count) & 0xFF);
            out_pos += 1;
        }
    }
    if (bit_count > 0) {
        if (out_pos >= out.len) return EncodeError.DestinationTooSmall;
        const pad: u6 = @intCast(@as(u32, 8) - @as(u32, bit_count));
        out[out_pos] = @intCast((bit_buf << pad) & 0xFF);
        out_pos += 1;
    }
    return out_pos;
}

// ── Public API: encode a byte block as chunk_type=4 ──────────────────
// Writes the standard non-compact 5-byte chunk header + 128 B weights +
// 9 B sub-header + 4 streams into `dst`. Returns total bytes written.
// dst must have capacity ≥ src.len + 200 to be safe.
pub fn encodeBlock(dst: []u8, src: []const u8) EncodeError!usize {
    if (src.len == 0) return EncodeError.TooFewSymbols;
    if (src.len > (1 << 18) - 1) return EncodeError.StreamTooLarge;

    var hist: [256]u32 = @splat(0);
    for (src) |b| hist[b] += 1;

    var code_lengths: [256]u8 = @splat(0);
    var codes: [256]u32 = @splat(0);
    const built = buildCodeLengths(&hist, &code_lengths);
    if (built.symbols_used < 1) return EncodeError.TooFewSymbols;

    if (built.max_len > MAX_CODE_LEN) {
        heightLimit(&hist, &code_lengths, MAX_CODE_LEN);
    }
    assignCanonicalCodes(&code_lengths, &codes);

    // Reserve space for 5 B header + 128 B weights + 9 B sub-header.
    const fixed_prefix: usize = 5 + WEIGHTS_BYTES + SUB_HEADER_BYTES;
    if (dst.len < fixed_prefix) return EncodeError.DestinationTooSmall;

    // ── Pack weights (nibble pairs) ──
    const weights_off: usize = 5;
    var i: usize = 0;
    while (i < 128) : (i += 1) {
        const lo = code_lengths[i * 2];
        const hi = code_lengths[i * 2 + 1];
        dst[weights_off + i] = (lo & 0x0F) | ((hi & 0x0F) << 4);
    }

    // ── Encode 4 streams ──
    const sub_hdr_off: usize = weights_off + WEIGHTS_BYTES;
    var stream_off: usize = sub_hdr_off + SUB_HEADER_BYTES;
    var sizes: [4]u32 = @splat(0);
    const q = src.len / 4;
    var stream_idx: usize = 0;
    while (stream_idx < 4) : (stream_idx += 1) {
        const start = stream_idx * q;
        const end = if (stream_idx == 3) src.len else (stream_idx + 1) * q;
        const n = try encodeStream(src[start..end], &code_lengths, &codes, dst[stream_off..]);
        if (n >= (1 << 24)) return EncodeError.StreamTooLarge;
        sizes[stream_idx] = @intCast(n);
        stream_off += n;
    }

    // ── Sub-header (3 × u24 LE) ──
    for (0..3) |s| {
        dst[sub_hdr_off + s * 3 + 0] = @intCast(sizes[s] & 0xFF);
        dst[sub_hdr_off + s * 3 + 1] = @intCast((sizes[s] >> 8) & 0xFF);
        dst[sub_hdr_off + s * 3 + 2] = @intCast((sizes[s] >> 16) & 0xFF);
    }

    // ── Non-compact 5-byte chunk header (type 4) ──
    const payload_size: u32 = @intCast(stream_off - 5);
    const decomp_size: u32 = @intCast(src.len);
    if (payload_size > 0x3FFFF) return EncodeError.StreamTooLarge;
    if (decomp_size - 1 > 0x3FFFF) return EncodeError.StreamTooLarge;
    entropy_enc.writeNonCompactChunkHeader(dst[0..5], 4, payload_size, decomp_size);

    return stream_off;
}
