//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Smoke test the VK binary: `streamlz_vk -c file -o out.slz` then
//! `streamlz_vk -d out.slz -o roundtrip` and compare to original.
//! Exercises every CLI mode at L1. Test bodies added by the fleshout
//! agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const testing = std.testing;

const VK_BIN = "zig-out/bin/streamlz_vk.exe";

const max_corpus_bytes: usize = 1 << 30;

fn makeIo() std.Io.Threaded {
    // Use the testing allocator so std.process.run + std.Io.Dir.writeFile
    // can satisfy their internal allocations (pipe buffers, etc.).
    return std.Io.Threaded.init(std.testing.allocator, .{});
}

fn fileExists(io: std.Io, rel: []const u8) bool {
    var f = std.Io.Dir.cwd().openFile(io, rel, .{}) catch return false;
    f.close(io);
    return true;
}

fn writeBytes(io: std.Io, path: []const u8, data: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{
        .sub_path = path,
        .data = data,
    });
}

fn readAll(allocator: std.mem.Allocator, io: std.Io, rel: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(
        io,
        rel,
        allocator,
        @enumFromInt(max_corpus_bytes),
    );
}

fn rmIfExists(io: std.Io, path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(io, path) catch {};
}

fn fileSize(io: std.Io, rel: []const u8) !u64 {
    var f = try std.Io.Dir.cwd().openFile(io, rel, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    return stat.size;
}

fn runCmd(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !std.process.RunResult {
    return std.process.run(allocator, io, .{ .argv = argv });
}

fn exitCode(term: std.process.Child.Term) u32 {
    return switch (term) {
        .exited => |c| c,
        else => 1,
    };
}

test "CLI smoke: -V prints version string and exits 0" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;
    const res = try runCmd(std.testing.allocator, io, &.{ VK_BIN, "-V" });
    defer {
        std.testing.allocator.free(res.stdout);
        std.testing.allocator.free(res.stderr);
    }
    try testing.expectEqual(@as(u32, 0), exitCode(res.term));
    try testing.expect(std.mem.indexOf(u8, res.stdout, "streamlz_vk") != null);
}

test "CLI smoke: --probe lists at least one device and exits 0" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;
    const res = try runCmd(std.testing.allocator, io, &.{ VK_BIN, "--probe" });
    defer {
        std.testing.allocator.free(res.stdout);
        std.testing.allocator.free(res.stderr);
    }
    try testing.expectEqual(@as(u32, 0), exitCode(res.term));
    try testing.expect(std.mem.indexOf(u8, res.stdout, "device[") != null);
}

test "CLI smoke: -c then -d round-trips a 4 KiB input byte-equal" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    var rng = std.Random.DefaultPrng.init(0x5EED_F00D);
    const src = try al.alloc(u8, 4 * 1024);
    rng.random().bytes(src);

    const in_path = "tmp_cli_smoke_in.bin";
    const slz_path = "tmp_cli_smoke.slz";
    const out_path = "tmp_cli_smoke_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // Compress.
    {
        const res = try runCmd(al, io, &.{ VK_BIN, "-c", in_path, "-o", slz_path });
        if (exitCode(res.term) != 0) {
            std.debug.print("compress failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.CompressFailed;
        }
    }
    // .slz must exist + be non-empty.
    const slz_sz = try fileSize(io, slz_path);
    try testing.expect(slz_sz > 0);

    // Decompress.
    {
        const res = try runCmd(al, io, &.{ VK_BIN, "-d", slz_path, "-o", out_path });
        if (exitCode(res.term) != 0) {
            std.debug.print("decompress failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.DecompressFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "CLI smoke: -i dumps a frame summary and exits 0" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const src = try al.alloc(u8, 2048);
    for (src, 0..) |*b, i| b.* = @intCast(i & 0xFF);

    const in_path = "tmp_cli_info_in.bin";
    const slz_path = "tmp_cli_info.slz";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);

    try writeBytes(io, in_path, src);

    {
        const res = try runCmd(al, io, &.{ VK_BIN, "-c", in_path, "-o", slz_path });
        if (exitCode(res.term) != 0) {
            std.debug.print("compress for -i failed: stderr={s}\n", .{res.stderr});
            return error.CompressFailed;
        }
    }
    const res = try runCmd(al, io, &.{ VK_BIN, "-i", slz_path });
    defer {
        al.free(res.stdout);
        al.free(res.stderr);
    }
    if (exitCode(res.term) != 0) {
        std.debug.print("info failed: stderr={s}\n", .{res.stderr});
        return error.InfoFailed;
    }
    // -i prints a frame summary — should be non-empty.
    try testing.expect(res.stdout.len > 0);
}

test "CLI smoke: -b runs a 1-round bench cleanly" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const src = try al.alloc(u8, 16 * 1024);
    for (src, 0..) |*b, i| b.* = @intCast((i * 7 + 13) & 0xFF);

    const in_path = "tmp_cli_bench_in.bin";
    defer rmIfExists(io, in_path);

    try writeBytes(io, in_path, src);

    const res = try runCmd(al, io, &.{ VK_BIN, "-b", "-r", "1", in_path });
    defer {
        al.free(res.stdout);
        al.free(res.stderr);
    }
    if (exitCode(res.term) != 0) {
        std.debug.print("bench failed: stderr={s}\n", .{res.stderr});
        return error.BenchFailed;
    }
    try testing.expect(res.stdout.len > 0);
}

test "CLI smoke: --help prints usage and exits 0" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;
    const res = try runCmd(std.testing.allocator, io, &.{ VK_BIN, "--help" });
    defer {
        std.testing.allocator.free(res.stdout);
        std.testing.allocator.free(res.stderr);
    }
    try testing.expectEqual(@as(u32, 0), exitCode(res.term));
    try testing.expect(std.mem.indexOf(u8, res.stdout, "Usage") != null);
}
