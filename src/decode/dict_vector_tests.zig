//! v4 #16 phase 2: hand-crafted dictionary-frame decode vectors.
//!
//! The encoder cannot emit dictionary-reaching matches yet (phase 3),
//! so the decoder's dictionary reach is verified against the FORMAT
//! spec directly: each vector is a declarative list of token ops from
//! which the test generates BOTH the wire bytes (a complete one-chunk
//! L1 frame, raw mode-1 sub-chunk) and the expected output (through a
//! sequential reference model where absolute position p < 0 reads
//! `dict[dict_len + p]`, and reads below the dictionary clamp to 0).
//! The two never share code with the GPU decoder, so agreement means
//! the kernels implement the spec, not merely their own encoder.
//!
//! Coverage by construction:
//!   - pure dict match, warp-parallel copy (dist >= len)
//!   - straddle: source starts in dict, crosses into the window's own
//!     output, self-overlapping (serial per-byte path)
//!   - dict reach through the recent-offset (two tokens in one PP batch)
//!   - long-near token (the pipelined kernel's serial drain path)
//!   - off32 short-far dict ref (the general decoder body; only
//!     reachable with SLZ_NO_PIPELINE=1 - the pipelined raw bridge
//!     assumes off32-free sub-chunks, which GPU frames satisfy)
//!   - hostile reach below the dictionary (clamps to 0x00, no fault)
//!
//! Every vector runs on the default pipelined kernel AND with
//! SLZ_NO_PIPELINE=1 (single-warp kernels), like-for-like.

const std = @import("std");
const testing = std.testing;
const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const constants = @import("../format/streamlz_constants.zig");
const dictionary = @import("../dict/dictionary.zig");
const decoder = @import("streamlz_decoder.zig");
const gpu_driver = @import("driver.zig");
const gpu_tests = @import("../encode/gpu_roundtrip_tests.zig");

extern "c" fn _putenv(pair: [*:0]const u8) c_int;

// ── Token ops: the declarative vector form ───────────────────────────

const Op = union(enum) {
    /// Short token: `lits.len` literal bytes (0-7) then a match of
    /// `match_len` (0-15) at `dist` bytes back (off16, 1..65535).
    short: struct { lits: []const u8, match_len: u5, dist: u16 },
    /// Short token reusing the most recent offset (no off16 entry).
    short_recent: struct { lits: []const u8, match_len: u5 },
    /// TOKEN_LONG_NEAR: match of `len` (>= 91) at `dist` back (off16).
    long_near: struct { len: u32, dist: u16 },
    /// Short-far token: match of `match_len` (5..28) whose off32 entry
    /// is `v` bytes BELOW the window base (v > 0 reaches the dict).
    short_far: struct { match_len: u8, v: u24 },
};

const token_short_min = 24;
const token_long_near = 1;
const long_near_base = 91;
const short_far_base = 5;

/// Sequential reference model. Absolute output position p reads:
/// p >= 0 -> out[p]; -dict_len <= p < 0 -> dict[dict_len + p]; else 0.
fn refByte(out: []const u8, dict: []const u8, p: i64) u8 {
    if (p >= 0) return out[@intCast(p)];
    const below = -p;
    if (below <= dict.len) return dict[dict.len - @as(usize, @intCast(below))];
    return 0;
}

const Generated = struct {
    sc_payload: std.ArrayList(u8) = .empty,
    expected: std.ArrayList(u8) = .empty,

    fn deinit(self: *Generated, gpa: std.mem.Allocator) void {
        self.sc_payload.deinit(gpa);
        self.expected.deinit(gpa);
    }
};

/// Generate the mode-1 sub-chunk payload streams AND the expected
/// decode output for an op list. Layout per FORMAT.md: [8 init bytes]
/// [lit stream][cmd stream][off16][off32][length stream].
fn generate(gpa: std.mem.Allocator, dict: []const u8, init8: *const [8]u8, ops: []const Op) !Generated {
    var g: Generated = .{};
    errdefer g.deinit(gpa);

    var lits: std.ArrayList(u8) = .empty;
    defer lits.deinit(gpa);
    var cmd: std.ArrayList(u8) = .empty;
    defer cmd.deinit(gpa);
    var off16: std.ArrayList(u8) = .empty;
    defer off16.deinit(gpa);
    var off32: std.ArrayList(u8) = .empty;
    defer off32.deinit(gpa);
    var lengths: std.ArrayList(u8) = .empty;
    defer lengths.deinit(gpa);

    try g.expected.appendSlice(gpa, init8);
    var recent: i64 = -8; // INITIAL_RECENT_OFFSET

    for (ops) |op| switch (op) {
        .short => |s| {
            std.debug.assert(s.lits.len <= 7 and s.match_len <= 15 and s.dist >= 1);
            try cmd.append(gpa, @intCast((@as(u8, s.match_len) << 3) | @as(u8, @intCast(s.lits.len))));
            try lits.appendSlice(gpa, s.lits);
            try off16.append(gpa, @intCast(s.dist & 0xFF));
            try off16.append(gpa, @intCast(s.dist >> 8));
            recent = -@as(i64, s.dist);
            try g.expected.appendSlice(gpa, s.lits);
            for (0..s.match_len) |_| {
                const p = @as(i64, @intCast(g.expected.items.len)) + recent;
                try g.expected.append(gpa, refByte(g.expected.items, dict, p));
            }
        },
        .short_recent => |s| {
            // use_recent bit 7 set; consumes no off16 entry.
            try cmd.append(gpa, 0x80 | (@as(u8, s.match_len) << 3) | @as(u8, @intCast(s.lits.len)));
            try lits.appendSlice(gpa, s.lits);
            try g.expected.appendSlice(gpa, s.lits);
            for (0..s.match_len) |_| {
                const p = @as(i64, @intCast(g.expected.items.len)) + recent;
                try g.expected.append(gpa, refByte(g.expected.items, dict, p));
            }
        },
        .long_near => |l| {
            std.debug.assert(l.len >= long_near_base and l.len - long_near_base <= 251);
            try cmd.append(gpa, token_long_near);
            try lengths.append(gpa, @intCast(l.len - long_near_base));
            try off16.append(gpa, @intCast(l.dist & 0xFF));
            try off16.append(gpa, @intCast(l.dist >> 8));
            recent = -@as(i64, l.dist);
            for (0..l.len) |_| {
                const p = @as(i64, @intCast(g.expected.items.len)) + recent;
                try g.expected.append(gpa, refByte(g.expected.items, dict, p));
            }
        },
        .short_far => |f| {
            // token = match_len - 5, valid token range 3..23.
            std.debug.assert(f.match_len >= 8 and f.match_len <= 28);
            try cmd.append(gpa, f.match_len - short_far_base);
            try off32.append(gpa, @intCast(f.v & 0xFF));
            try off32.append(gpa, @intCast((f.v >> 8) & 0xFF));
            try off32.append(gpa, @intCast(f.v >> 16));
            // off32 entries encode "v bytes below the window base":
            // source byte i sits at absolute position -v + i.
            const src0: i64 = -@as(i64, f.v);
            for (0..f.match_len) |i| {
                try g.expected.append(gpa, refByte(g.expected.items, dict, src0 + @as(i64, @intCast(i))));
            }
        },
    };

    // Assemble the sub-chunk payload.
    try g.sc_payload.appendSlice(gpa, init8);
    try appendRawStream(gpa, &g.sc_payload, lits.items);
    try appendRawStream(gpa, &g.sc_payload, cmd.items);
    // off16: 2-byte LE count + entries.
    const n_off16: u16 = @intCast(off16.items.len / 2);
    try g.sc_payload.append(gpa, @intCast(n_off16 & 0xFF));
    try g.sc_payload.append(gpa, @intCast(n_off16 >> 8));
    try g.sc_payload.appendSlice(gpa, off16.items);
    // off32: 3-byte packed counts (count1 << 12 | count2), entries.
    const n_off32: u24 = @intCast(off32.items.len / 3);
    const packed_counts: u24 = @as(u24, n_off32) << 12;
    try g.sc_payload.append(gpa, @intCast(packed_counts & 0xFF));
    try g.sc_payload.append(gpa, @intCast((packed_counts >> 8) & 0xFF));
    try g.sc_payload.append(gpa, @intCast(packed_counts >> 16));
    try g.sc_payload.appendSlice(gpa, off32.items);
    try g.sc_payload.appendSlice(gpa, lengths.items);
    return g;
}

/// Type-0 raw stream: 3-byte big-endian size, then the bytes.
fn appendRawStream(gpa: std.mem.Allocator, dst: *std.ArrayList(u8), bytes: []const u8) !void {
    std.debug.assert(bytes.len < 0x800000);
    try dst.append(gpa, @intCast((bytes.len >> 16) & 0x7F));
    try dst.append(gpa, @intCast((bytes.len >> 8) & 0xFF));
    try dst.append(gpa, @intCast(bytes.len & 0xFF));
    try dst.appendSlice(gpa, bytes);
}

/// Wrap a sub-chunk payload into a complete one-chunk dictionary frame.
fn buildFrame(gpa: std.mem.Allocator, dict_id: u32, content_size: usize, sc_payload: []const u8) ![]u8 {
    const buf = try gpa.alloc(u8, 256 + sc_payload.len);
    errdefer gpa.free(buf);

    var pos = try frame.writeHeader(buf, .{
        .codec = .fast,
        .level = 1,
        .block_size = constants.chunk_size,
        .sc_group_size = 0.25,
        .content_size = @intCast(content_size),
        .dictionary_id = dict_id,
        .content_checksum = false,
        .chunk_merkle = false,
    });

    // Block payload = [2B internal hdr][4B chunk hdr][3B sub-chunk hdr][payload].
    const block_payload_len = 2 + 4 + 3 + sc_payload.len;
    frame.writeBlockHeader(buf[pos..], .{
        .compressed_size = @intCast(block_payload_len),
        .decompressed_size = @intCast(content_size),
        .uncompressed = false,
        .parallel_decode_metadata = false,
    });
    pos += 8;

    // Internal header: magic | self_contained | keyframe, codec fast
    // (mirrors encode_assemble's internal_hdr0/1).
    buf[pos] = 0x05 | 0x10 | 0x40;
    buf[pos + 1] = @intFromEnum(block_header.CodecType.fast);
    pos += 2;

    // 4-byte LE chunk header: (compressed_size - 1) in bits 0..17, type 0.
    const chunk_comp: u32 = @intCast(3 + sc_payload.len);
    std.mem.writeInt(u32, buf[pos..][0..4], chunk_comp - 1, .little);
    pos += 4;

    // 3-byte BE sub-chunk header: LZ flag (bit 23) | mode 1 (bits 19-22)
    // | payload size (bits 0-18).
    const sch: u32 = (1 << 23) | (1 << 19) | @as(u32, @intCast(sc_payload.len));
    buf[pos] = @intCast((sch >> 16) & 0xFF);
    buf[pos + 1] = @intCast((sch >> 8) & 0xFF);
    buf[pos + 2] = @intCast(sch & 0xFF);
    pos += 3;

    @memcpy(buf[pos..][0..sc_payload.len], sc_payload);
    pos += sc_payload.len;

    frame.writeEndMark(buf[pos..]);
    pos += 4;
    // Shrink to the exact frame size so the caller's free matches the
    // allocation (the debug allocator validates slice identity).
    return gpa.realloc(buf, pos);
}

fn decodeAndCheck(gpa: std.mem.Allocator, frame_bytes: []const u8, expected: []const u8, label: []const u8) !void {
    const dst = try gpa.alloc(u8, expected.len + decoder.safe_space);
    defer gpa.free(dst);
    @memset(dst, 0xAA);
    const written = decoder.decompressFramed(frame_bytes, dst, &gpu_driver.g_default) catch |err| {
        std.debug.print("dict vector '{s}': decode failed: {s}\n", .{ label, @errorName(err) });
        return err;
    };
    if (written != expected.len) {
        std.debug.print("dict vector '{s}': wrote {d}, expected {d}\n", .{ label, written, expected.len });
        return error.TestUnexpectedResult;
    }
    if (!std.mem.eql(u8, expected, dst[0..written])) {
        var off: usize = 0;
        while (off < expected.len and expected[off] == dst[off]) off += 1;
        std.debug.print(
            "dict vector '{s}': mismatch at {d}: got 0x{x:0>2} want 0x{x:0>2}\n  got:  {x}\n  want: {x}\n",
            .{ label, off, dst[off], expected[off], dst[0..@min(written, 32)], expected[0..@min(expected.len, 32)] },
        );
        return error.TestUnexpectedResult;
    }
}

test "v4 #16 phase 2: dictionary decode vectors (spec-derived, both kernel paths)" {
    gpu_tests.lockGpuTests();
    defer gpu_tests.unlockGpuTests();
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const gpa = testing.allocator;

    // The two-path sweep relies on _putenv being visible to the
    // dispatch's std.c.getenv (same CRT). Probe it loudly - a silent
    // mismatch would run every vector on the default kernel twice.
    try testing.expect(_putenv("SLZ_DICT_VEC_PROBE=1") == 0);
    try testing.expect(std.c.getenv("SLZ_DICT_VEC_PROBE") != null);
    _ = _putenv("SLZ_DICT_VEC_PROBE=");

    const dict = dictionary.findById(dictionary.id_json).?.data;
    const init8 = "ABCDEFGH";

    const Vector = struct {
        label: []const u8,
        ops: []const Op,
        /// Skipped on the pipelined kernel: the raw-pipeline bridge
        /// assumes off32-free sub-chunks (GPU frames always are).
        off32: bool = false,
    };
    const vectors = [_]Vector{
        .{ .label = "pure dict match, warp-parallel", .ops = &.{
            .{ .short = .{ .lits = "", .match_len = 15, .dist = 108 } },
        } },
        .{ .label = "straddle + self-overlap, serial per-byte", .ops = &.{
            .{ .short = .{ .lits = "", .match_len = 15, .dist = 10 } },
        } },
        .{ .label = "lits + dict + recent-offset reuse in one batch", .ops = &.{
            .{ .short = .{ .lits = "abc", .match_len = 12, .dist = 50 } },
            .{ .short_recent = .{ .lits = "xy", .match_len = 10 } },
        } },
        .{ .label = "long-near dict straddle (serial drain path)", .ops = &.{
            .{ .long_near = .{ .len = 100, .dist = 58 } },
        } },
        .{ .label = "hostile reach below the dictionary clamps to zero", .ops = &.{
            .{ .short = .{ .lits = "", .match_len = 15, .dist = 60000 } },
        } },
        // match_len 25 keeps decomp (33) above the payload size (23) so
        // the sub-chunk stays on the LZ path (comp == decomp would route
        // to the uncompressed memcpy branch).
        .{ .label = "off32 short-far dict ref (general body)", .off32 = true, .ops = &.{
            .{ .short_far = .{ .match_len = 25, .v = 200 } },
        } },
    };

    // Both kernel paths: default (pipelined) and SLZ_NO_PIPELINE=1
    // (single-warp; also the only route to the off32/general body).
    for ([_]bool{ false, true }) |no_pipeline| {
        _ = _putenv(if (no_pipeline) "SLZ_NO_PIPELINE=1" else "SLZ_NO_PIPELINE=");
        defer _ = _putenv("SLZ_NO_PIPELINE=");

        for (vectors) |v| {
            if (v.off32 and !no_pipeline) continue;
            var g = try generate(gpa, dict, init8, v.ops);
            defer g.deinit(gpa);
            const fb = try buildFrame(gpa, dictionary.id_json, g.expected.items.len, g.sc_payload.items);
            defer gpa.free(fb);
            try decodeAndCheck(gpa, fb, g.expected.items, v.label);
        }
    }
}
