//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Kernel conformance: feed known sub-chunk inputs into the L1 hot
//! kernels (slzLzDecodeRawKernel, slzPrefixSumChunksKernel,
//! slzLzEncodeKernel, slzAssembleMeasureKernel, slzAssembleWriteKernel,
//! slzFrameAssembleKernel) via VK; assert byte-identical outputs vs
//! CUDA goldens stored in tests/goldens/.
//!
//! The CUDA repo currently stores end-to-end frame goldens (full .slz
//! files) rather than per-kernel I/O blobs. This file exercises those
//! goldens through the full VK decode pipeline as the closest available
//! kernel-conformance probe — a mismatch isolates either an upload, a
//! kernel, or a finalize bug.
//!

const std = @import("std");
const testing = std.testing;
const decoder = @import("../decode/streamlz_decoder.zig");
const driver = @import("../decode/driver.zig");
const frame = @import("../format/frame_format.zig");

const max_corpus_bytes: usize = 1 << 30; // 1 GiB cap

fn makeIo() std.Io.Threaded {
    return std.Io.Threaded.init(std.mem.Allocator.failing, .{});
}

fn readGoldenAlloc(allocator: std.mem.Allocator, io: std.Io, rel_path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(
        io,
        rel_path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    );
}

test "kernel conformance: web.txt.L1.slz parses as a valid SLZ1 frame" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = readGoldenAlloc(arena.allocator(), io, "tests/goldens/web.txt.L1.slz") catch return error.SkipZigTest;

    const hdr = try frame.parseHeader(golden);
    try testing.expectEqual(@as(u8, frame.version), hdr.version);
    try testing.expectEqual(@as(u8, 1), hdr.level);
    // L1 always emits the .fast codec.
    try testing.expectEqual(frame.Codec.fast, hdr.codec);
    // content_size_present is the L1 encoder default.
    try testing.expect(hdr.content_size != null);
}

test "kernel conformance: enwik8.txt.L1.slz parses as a valid SLZ1 frame" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = readGoldenAlloc(arena.allocator(), io, "tests/goldens/enwik8.txt.L1.slz") catch return error.SkipZigTest;

    const hdr = try frame.parseHeader(golden);
    try testing.expectEqual(@as(u8, frame.version), hdr.version);
    try testing.expectEqual(@as(u8, 1), hdr.level);
    try testing.expectEqual(frame.Codec.fast, hdr.codec);
    try testing.expect(hdr.content_size != null);
    try testing.expect(hdr.content_size.? > 0);
}

test "kernel conformance: web.txt.L1.slz block walk reaches end-mark without truncation" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const golden = readGoldenAlloc(arena.allocator(), io, "tests/goldens/web.txt.L1.slz") catch return error.SkipZigTest;

    const hdr = try frame.parseHeader(golden);
    var pos: usize = hdr.header_size;
    var blocks_seen: usize = 0;
    while (pos + 4 <= golden.len) {
        const first_word = std.mem.readInt(u32, golden[pos..][0..4], .little);
        if (first_word == frame.end_mark) {
            pos += 4;
            break;
        }
        const bh = try frame.parseBlockHeader(golden[pos..]);
        if (bh.isEndMark()) {
            pos += 8;
            break;
        }
        pos += 8;
        // Block payload must lie entirely inside the frame.
        try testing.expect(pos + bh.compressed_size <= golden.len);
        pos += bh.compressed_size;
        blocks_seen += 1;
        if (blocks_seen > 100_000) break; // safety bail
    }
    try testing.expect(blocks_seen > 0);
}

test "kernel conformance: GPU-decode web.txt.L1.slz round-trips byte-equal vs source asset" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const golden = readGoldenAlloc(al, io, "tests/goldens/web.txt.L1.slz") catch return error.SkipZigTest;
    const original = readGoldenAlloc(al, io, "assets/web.txt") catch return error.SkipZigTest;

    const hdr = try frame.parseHeader(golden);
    const want_size: usize = if (hdr.content_size) |cs| @intCast(cs) else original.len;
    try testing.expectEqual(original.len, want_size);

    const dst = try al.alloc(u8, want_size + 64);
    @memset(dst, 0);

    // VK adaptation: per-test DecodeContext. The shared dec_driver.g_default
    // gets clobbered by sibling parallel workers calling ensureDeviceBuf
    // (destroy+create) on its persistent device buffers.
    var dec_ctx: driver.DecodeContext = .{};
    defer dec_ctx.deinit();

    const written = try decoder.decompressFramed(golden, dst, &dec_ctx);
    try testing.expectEqual(original.len, written);
    try testing.expectEqualSlices(u8, original, dst[0..written]);
}
