//! Forward-LZ experimental codec.
//!
//! Encoder: greedy parse → group matches by content → emit:
//!   1. Pattern table (unique patterns with tANS-coded bytes)
//!   2. Position lists (delta-varint per pattern, tANS-coded)
//!   3. Literal stream (uncovered bytes, tANS-coded)
//!   4. Literal position deltas (tANS-coded)
//!   5. Control stream (pattern lengths + counts)
//!
//! Decoder: scatter-write each pattern to its positions, fill literals.
//! No backward reads from the output buffer — all writes are from
//! known pattern bytes or the literal stream.

const std = @import("std");
const lz_constants = @import("../format/streamlz_constants.zig");
const FastMatchHasher = @import("fast/fast_match_hasher.zig").FastMatchHasher;
const fast_constants = @import("fast/fast_constants.zig");
const entropy_enc = @import("entropy/entropy_encoder.zig");
const tans_enc = @import("entropy/tans_encoder.zig");

const ByteHistogram = @import("entropy/byte_histogram.zig").ByteHistogram;

pub const ForwardLzResult = struct {
    total_size: usize,
    pattern_stream_size: usize,
    position_stream_size: usize,
    literal_stream_size: usize,
    literal_pos_stream_size: usize,
    control_stream_size: usize,
    num_forward_refs: usize,
    num_singletons: usize,
    num_literals: usize,
    num_scatter_writes: usize,
};

const MatchInfo = struct {
    dest_pos: u32,
    length: u16,
};

const PatternGroup = struct {
    content_hash: u64,
    first_dest: u32,
    length: u16,
    count: u32,
    positions: std.ArrayList(u32),
};

pub fn analyzeForwardLz(
    allocator: std.mem.Allocator,
    src: []const u8,
) !ForwardLzResult {
    // ── Step 1: Greedy parse (reuse FastMatchHasher) ──
    var hasher = FastMatchHasher(u16).init(allocator, .{
        .hash_bits = 14,
        .min_match_length = 4,
    }) catch return error.OutOfMemory;
    defer hasher.deinit();

    const hash_table = hasher.hash_table;
    const hash_mult = hasher.hash_mult;
    const hash_shift = hasher.hash_shift;

    var matches = std.ArrayList(MatchInfo).empty;
    defer matches.deinit(allocator);

    var covered = try allocator.alloc(u8, src.len);
    defer allocator.free(covered);
    @memset(covered, 0);

    var pos: usize = 0;
    while (pos + 8 <= src.len) {
        const word: u64 = std.mem.readInt(u64, src[pos..][0..8], .little);
        const hi: usize = @intCast((word *% hash_mult) >> hash_shift);
        const stored: u16 = hash_table[hi];
        const cur: u16 = @truncate(pos);
        hash_table[hi] = cur;

        const offset: u16 = cur -% stored;
        if (offset >= 8) {
            const src_pos = pos -% offset;
            var mlen: usize = 0;
            const limit = @min(@as(usize, 256), src.len - pos, src.len - src_pos);
            while (mlen < limit and src[src_pos + mlen] == src[pos + mlen]) {
                mlen += 1;
            }
            if (mlen >= 4) {
                try matches.append(allocator, .{
                    .dest_pos = @intCast(pos),
                    .length = @intCast(mlen),
                });
                @memset(covered[pos..][0..mlen], 1);
                pos += mlen;
                continue;
            }
        }
        pos += 1;
    }

    // ── Step 2: Group by content hash ──
    var groups = std.AutoHashMap(u64, *PatternGroup).init(allocator);
    defer {
        var it = groups.valueIterator();
        while (it.next()) |g| {
            g.*.positions.deinit(allocator);
            allocator.destroy(g.*);
        }
        groups.deinit();
    }

    for (matches.items) |m| {
        const content = src[m.dest_pos..][0..m.length];
        var h: u64 = 0;
        for (content) |b| {
            h = h *% 0x100000001B3 +% b;
        }
        h ^= @as(u64, m.length) << 48;

        const gop = try groups.getOrPut(h);
        if (!gop.found_existing) {
            const g = try allocator.create(PatternGroup);
            g.* = .{
                .content_hash = h,
                .first_dest = m.dest_pos,
                .length = m.length,
                .count = 1,
                .positions = std.ArrayList(u32).empty,
            };
            try g.positions.append(allocator, m.dest_pos);
            gop.value_ptr.* = g;
        } else {
            const g = gop.value_ptr.*;
            g.count += 1;
            try g.positions.append(allocator, m.dest_pos);
        }
    }

    // ── Step 3: Build streams ──
    var pattern_bytes = std.ArrayList(u8).empty;
    defer pattern_bytes.deinit(allocator);
    var position_deltas = std.ArrayList(u8).empty;
    defer position_deltas.deinit(allocator);
    var control_bytes = std.ArrayList(u8).empty;
    defer control_bytes.deinit(allocator);

    var num_forward: usize = 0;
    var num_singleton: usize = 0;
    var num_scatters: usize = 0;

    var git = groups.valueIterator();
    while (git.next()) |gp| {
        const g = gp.*;
        const content = src[g.first_dest..][0..g.length];
        try pattern_bytes.appendSlice(allocator, content);

        // Control: pattern length as varint
        try appendVarint(&control_bytes, allocator, g.length);

        if (g.count >= 2) {
            num_forward += 1;
            num_scatters += g.count;
            // Control: count as varint
            try appendVarint(&control_bytes, allocator, g.count);
            // Positions: sorted delta-varint
            std.mem.sort(u32, g.positions.items, {}, std.sort.asc(u32));
            var prev: u32 = 0;
            for (g.positions.items) |p| {
                try appendVarint(&position_deltas, allocator, p - prev);
                prev = p;
            }
        } else {
            num_singleton += 1;
            try appendVarint(&control_bytes, allocator, 1);
            try appendVarint(&position_deltas, allocator, g.positions.items[0]);
        }
    }

    // Literal stream + literal position deltas
    var lit_bytes = std.ArrayList(u8).empty;
    defer lit_bytes.deinit(allocator);
    var lit_pos_deltas = std.ArrayList(u8).empty;
    defer lit_pos_deltas.deinit(allocator);

    var num_lits: usize = 0;
    var prev_lit_pos: u32 = 0;
    for (covered, 0..) |c, i| {
        if (c == 0) {
            try lit_bytes.append(allocator, src[i]);
            try appendVarint(&lit_pos_deltas, allocator, @as(u32, @intCast(i)) - prev_lit_pos);
            prev_lit_pos = @intCast(i);
            num_lits += 1;
        }
    }

    // ── Step 4: tANS-encode each stream ──
    const pat_enc = try tansEncodeStream(allocator, pattern_bytes.items);
    defer allocator.free(pat_enc);
    const pos_enc = try tansEncodeStream(allocator, position_deltas.items);
    defer allocator.free(pos_enc);
    const lit_enc = try tansEncodeStream(allocator, lit_bytes.items);
    defer allocator.free(lit_enc);
    const litpos_enc = try tansEncodeStream(allocator, lit_pos_deltas.items);
    defer allocator.free(litpos_enc);
    const ctrl_enc = try tansEncodeStream(allocator, control_bytes.items);
    defer allocator.free(ctrl_enc);

    const total = pat_enc.len + pos_enc.len + lit_enc.len + litpos_enc.len + ctrl_enc.len + 20;

    return .{
        .total_size = total,
        .pattern_stream_size = pat_enc.len,
        .position_stream_size = pos_enc.len,
        .literal_stream_size = lit_enc.len,
        .literal_pos_stream_size = litpos_enc.len,
        .control_stream_size = ctrl_enc.len,
        .num_forward_refs = num_forward,
        .num_singletons = num_singleton,
        .num_literals = num_lits,
        .num_scatter_writes = num_scatters,
    };
}

fn appendVarint(list: *std.ArrayList(u8), alloc: std.mem.Allocator, value: u32) !void {
    var v = value;
    while (v >= 0x80) {
        try list.append(alloc, @intCast((v & 0x7F) | 0x80));
        v >>= 7;
    }
    try list.append(alloc, @intCast(v));
}

fn tansEncodeStream(allocator: std.mem.Allocator, src: []const u8) ![]u8 {
    if (src.len <= 32) {
        const out = try allocator.alloc(u8, src.len + 3);
        // memcpy header
        out[0] = @intCast((src.len >> 16) & 0xFF);
        out[1] = @intCast((src.len >> 8) & 0xFF);
        out[2] = @intCast(src.len & 0xFF);
        @memcpy(out[3..][0..src.len], src);
        return out;
    }

    const bound = src.len + 512;
    const dst = try allocator.alloc(u8, bound);
    errdefer allocator.free(dst);

    var histo: ByteHistogram = .{};
    histo.countBytes(src);

    var cost: f32 = 0;
    const n = tans_enc.encodeArrayU8Tans(allocator, dst[5..], src, &histo, 0.0, &cost) catch {
        // tANS failed — fall back to memcpy
        const out = try allocator.alloc(u8, src.len + 3);
        out[0] = @intCast((src.len >> 16) & 0xFF);
        out[1] = @intCast((src.len >> 8) & 0xFF);
        out[2] = @intCast(src.len & 0xFF);
        @memcpy(out[3..][0..src.len], src);
        allocator.free(dst);
        return out;
    };

    if (n > 0 and n < src.len) {
        // tANS header: type(1) + compressed_size(2) + decompressed_size(2)
        dst[0] = 1; // tANS
        dst[1] = @intCast((n >> 8) & 0xFF);
        dst[2] = @intCast(n & 0xFF);
        dst[3] = @intCast((src.len >> 8) & 0xFF);
        dst[4] = @intCast(src.len & 0xFF);
        const result = try allocator.alloc(u8, 5 + n);
        @memcpy(result, dst[0 .. 5 + n]);
        allocator.free(dst);
        return result;
    }

    // tANS didn't help
    allocator.free(dst);
    const out = try allocator.alloc(u8, src.len + 3);
    out[0] = @intCast((src.len >> 16) & 0xFF);
    out[1] = @intCast((src.len >> 8) & 0xFF);
    out[2] = @intCast(src.len & 0xFF);
    @memcpy(out[3..][0..src.len], src);
    return out;
}

// ── Test ──
test "forward-LZ analysis on small input" {
    const allocator = std.testing.allocator;
    const src = "the quick brown fox jumps over the lazy dog and the quick brown fox";
    const result = try analyzeForwardLz(allocator, src);
    try std.testing.expect(result.total_size > 0);
    try std.testing.expect(result.num_forward_refs > 0 or result.num_singletons > 0);
}
