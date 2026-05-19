//! Standalone CPU tANS / tANS32 round-trip harness.
//!
//! Usage:
//!   zig run src/tans_standalone.zig -lc -- <file> [tans|tans32]
//!
//! Reads <file> bytes, encodes via tANS (5-state, chunk_type=1) or
//! tANS32 (32-stream, chunk_type=6), then decodes via highDecodeBytes
//! (which sees the chunk header and routes to the right decoder).
//! Compares byte-for-byte and reports first mismatch.

const std = @import("std");
const tans_enc = @import("encode/entropy/tans_encoder.zig");
const entropy_enc = @import("encode/entropy/entropy_encoder.zig");
const entropy_dec = @import("decode/entropy/entropy_decoder.zig");
const byte_hist = @import("encode/entropy/byte_histogram.zig");

const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
});

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Read from env vars: SLZ_TANS_IN, SLZ_TANS_MODE.
    const path_env = c.getenv("SLZ_TANS_IN") orelse {
        std.debug.print("set SLZ_TANS_IN=<file> [and SLZ_TANS_MODE=tans|tans32]\n", .{});
        return;
    };
    const path = std.mem.span(path_env);
    const mode_env = c.getenv("SLZ_TANS_MODE");
    const use_tans32 = if (mode_env) |m|
        std.mem.eql(u8, std.mem.span(m), "tans32")
    else
        true;

    // Read input via libc
    const path_c = try allocator.dupeZ(u8, path);
    const f = c.fopen(path_c, "rb") orelse {
        std.debug.print("cannot open {s}\n", .{path});
        return;
    };
    defer _ = c.fclose(f);
    _ = c.fseek(f, 0, c.SEEK_END);
    const size: usize = @intCast(c.ftell(f));
    _ = c.fseek(f, 0, c.SEEK_SET);
    const src = try allocator.alloc(u8, size);
    const got = c.fread(src.ptr, 1, size, f);
    if (got != size) {
        std.debug.print("short read: {d}/{d}\n", .{ got, size });
        return;
    }

    // Histogram
    var histo: byte_hist.ByteHistogram = .{};
    histo.countBytes(src);

    // Encode (writes the 5-byte non-compact chunk header + body via encodeArrayU8 wrapper).
    const enc_cap = src.len + 1024;
    const enc_buf = try allocator.alloc(u8, enc_cap);
    var cost: f32 = @floatFromInt(src.len + 3);

    const enc_n = if (use_tans32)
        try tans_enc.encodeArrayU8Tans32(allocator, enc_buf[5..], src, &histo, 0.0, &cost, null)
    else
        try tans_enc.encodeArrayU8Tans(allocator, enc_buf[5..], src, &histo, 0.0, &cost);
    entropy_enc.writeNonCompactChunkHeader(enc_buf[0..5], if (use_tans32) 6 else 1, @intCast(enc_n), @intCast(src.len));
    const total_enc = 5 + enc_n;

    std.debug.print("mode={s}  src={d}  enc={d} ({d:.2}%)\n", .{
        if (use_tans32) "tans32" else "tans",
        src.len,
        total_enc,
        100.0 * @as(f64, @floatFromInt(total_enc)) / @as(f64, @floatFromInt(src.len)),
    });

    // Decode via highDecodeBytes (reads chunk header, dispatches).
    const dec_buf = try allocator.alloc(u8, src.len);
    const scratch = try allocator.alloc(u8, 16 * 1024 * 1024);
    const scratch_end = scratch.ptr + scratch.len;
    const res = try entropy_dec.highDecodeBytes(
        dec_buf.ptr,
        dec_buf.len,
        enc_buf[0..total_enc],
        true,
        scratch.ptr,
        scratch_end,
    );
    std.debug.print("decoded {d} bytes (consumed {d})\n", .{ res.decoded_size, res.bytes_consumed });

    if (res.decoded_size != src.len) {
        std.debug.print("SIZE MISMATCH\n", .{});
        return;
    }

    var first_diff: usize = src.len;
    var n_diff: usize = 0;
    for (0..src.len) |i| {
        if (dec_buf[i] != src[i]) {
            if (first_diff == src.len) first_diff = i;
            n_diff += 1;
        }
    }
    if (n_diff == 0) {
        std.debug.print("OK (round-trip clean)\n", .{});
    } else {
        std.debug.print("MISMATCH: {d} bytes differ; first at byte {d}: src=0x{x:0>2} dec=0x{x:0>2}\n", .{
            n_diff,
            first_diff,
            src[first_diff],
            dec_buf[first_diff],
        });
    }
}
