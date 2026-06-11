//! v4 #18 stress harness: in-process encode+decode roundtrip loop of
//! the exact shape that failed once in the suite (medium text L1,
//! mismatch at chunk1+8), reproducing the TEST conditions the CLI
//! loop lacks: one long-lived process, the same g_default
//! encode/decode contexts the suite uses, buffer reuse across
//! iterations, and the suite's case ordering (a tiny warm-up case
//! first). Reports first failing iteration + offset.
//!
//! Usage: stress18 <iters> [level]

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const gpu_driver = @import("decode/driver.zig");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;

    var args_it = try init.minimal.args.iterateAllocator(gpa);
    defer args_it.deinit();
    _ = args_it.next();
    const iters: u32 = if (args_it.next()) |s| std.fmt.parseInt(u32, s, 10) catch 200 else 200;
    const level: u8 = if (args_it.next()) |s| std.fmt.parseInt(u8, s, 10) catch 1 else 1;

    const src = try gpa.alloc(u8, 100 * 1024);
    defer gpa.free(src);
    const pattern = "The quick brown fox jumps over a lazy dog. ";
    for (src, 0..) |*b, i| b.* = pattern[i % pattern.len];

    const tiny = try gpa.alloc(u8, 1024);
    defer gpa.free(tiny);
    for (tiny, 0..) |*b, i| b.* = @intCast(i % 251);

    const bound = encoder.compressBound(src.len);
    const frame = try gpa.alloc(u8, bound);
    defer gpa.free(frame);
    const out = try gpa.alloc(u8, src.len + decoder.safe_space);
    defer gpa.free(out);

    // Warm-up roundtrip on tiny (mirrors the suite's case order).
    {
        const tb = encoder.compressBound(tiny.len);
        const tf = try gpa.alloc(u8, tb);
        defer gpa.free(tf);
        const to = try gpa.alloc(u8, tiny.len + decoder.safe_space);
        defer gpa.free(to);
        const fl = try encoder.compressFramed(gpa, tiny, tf, .{ .level = level }, &gpu_encoder.g_default);
        _ = try decoder.decompressFramed(tf[0..fl], to, &gpu_driver.g_default);
    }

    var fails: u32 = 0;
    var it: u32 = 0;
    while (it < iters) : (it += 1) {
        const frame_len = try encoder.compressFramed(gpa, src, frame, .{ .level = level }, &gpu_encoder.g_default);
        @memset(out[0..src.len], 0xAA);
        const n = try decoder.decompressFramed(frame[0..frame_len], out, &gpu_driver.g_default);
        if (n != src.len) {
            std.debug.print("iter {d}: LENGTH MISMATCH {d} != {d}\n", .{ it, n, src.len });
            fails += 1;
            continue;
        }
        if (!std.mem.eql(u8, out[0..src.len], src)) {
            var off: usize = 0;
            while (off < src.len and out[off] == src[off]) off += 1;
            std.debug.print(
                "iter {d}: BYTE MISMATCH at offset {d} (chunk {d} + {d}): got 0x{x:0>2} want 0x{x:0>2}\n",
                .{ it, off, off / 65536, off % 65536, out[off], src[off] },
            );
            fails += 1;
        }
    }
    std.debug.print("stress18 L{d}: {d} iters, {d} failures\n", .{ level, iters, fails });
    return if (fails == 0) 0 else 1;
}
