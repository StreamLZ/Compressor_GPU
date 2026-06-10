//! CLI smoke tests — backport of `srcVK/tests/cli_smoke.zig`
//! (BACKPORTS.md D, wave 2).
//!
//! Spawns the installed `streamlz.exe` as a subprocess and exercises
//! every CLI mode end-to-end: -V, --help, -c/-d roundtrip, -i, -b.
//! These catch the class of bug unit tests can't: argument parsing,
//! file I/O plumbing, exit codes, and the child-process environment
//! (the VK side's empty-env spawn once hid the discrete GPU from the
//! child — see makeIo below).
//!
//! CUDA adaptations vs the VK original: no `--probe`/`--device NVIDIA`
//! splicing (the CUDA driver API enumerates the dGPU directly; there is
//! no Intel-iGPU default-selection trap), and no `[serial_first]`
//! marker (CUDA contexts have no cross-process bring-up contention; the
//! GPU-touching cases take the shared in-process GPU lock instead).

const std = @import("std");
const testing = std.testing;

const gpu_roundtrip_tests = @import("encode/gpu_roundtrip_tests.zig");

const BIN = "zig-out/bin/streamlz.exe";
const max_read_bytes: usize = 1 << 30;

fn makeIo() std.Io.Threaded {
    // `.environ = .empty` would hand the child an EMPTY Windows
    // environment block (no PATH / SystemRoot) — the VK-side lesson:
    // driver discovery in the child needs the parent environment.
    // Forward the parent's PEB block verbatim.
    return std.Io.Threaded.init(std.testing.allocator, .{
        .environ = .{ .block = .global },
    });
}

fn fileExists(io: std.Io, rel: []const u8) bool {
    var f = std.Io.Dir.cwd().openFile(io, rel, .{}) catch return false;
    f.close(io);
    return true;
}

fn writeBytes(io: std.Io, path: []const u8, data: []const u8) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

fn readAll(allocator: std.mem.Allocator, io: std.Io, rel: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, rel, allocator, @enumFromInt(max_read_bytes));
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

    if (!fileExists(io, BIN)) return error.SkipZigTest;
    const res = try runCmd(testing.allocator, io, &.{ BIN, "-V" });
    defer {
        testing.allocator.free(res.stdout);
        testing.allocator.free(res.stderr);
    }
    try testing.expectEqual(@as(u32, 0), exitCode(res.term));
    try testing.expect(std.mem.indexOf(u8, res.stdout, "streamlz") != null);
}

test "CLI smoke: --help prints usage and exits 0" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, BIN)) return error.SkipZigTest;
    const res = try runCmd(testing.allocator, io, &.{ BIN, "--help" });
    defer {
        testing.allocator.free(res.stdout);
        testing.allocator.free(res.stderr);
    }
    try testing.expectEqual(@as(u32, 0), exitCode(res.term));
    try testing.expect(std.mem.indexOf(u8, res.stdout, "Usage") != null);
}

test "CLI smoke: -c then -d round-trips a 4 KiB input byte-equal" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    var rng = std.Random.DefaultPrng.init(0x5EED_F00D);
    const src = try al.alloc(u8, 4 * 1024);
    rng.random().bytes(src);

    const in_path = "tmp_cuda_cli_smoke_in.bin";
    const slz_path = "tmp_cuda_cli_smoke.slz";
    const out_path = "tmp_cuda_cli_smoke_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    {
        const res = try runCmd(al, io, &.{ BIN, "-c", in_path, "-o", slz_path });
        if (exitCode(res.term) != 0) {
            std.debug.print("compress failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.CompressFailed;
        }
    }
    const slz_sz = try fileSize(io, slz_path);
    try testing.expect(slz_sz > 0);

    {
        const res = try runCmd(al, io, &.{ BIN, "-d", slz_path, "-o", out_path });
        if (exitCode(res.term) != 0) {
            std.debug.print("decompress failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.DecompressFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "CLI smoke: -i dumps a frame summary and exits 0" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const src = try al.alloc(u8, 2048);
    for (src, 0..) |*b, i| b.* = @intCast(i & 0xFF);

    const in_path = "tmp_cuda_cli_info_in.bin";
    const slz_path = "tmp_cuda_cli_info.slz";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);

    try writeBytes(io, in_path, src);

    {
        const res = try runCmd(al, io, &.{ BIN, "-c", in_path, "-o", slz_path });
        if (exitCode(res.term) != 0) {
            std.debug.print("compress for -i failed: stdout={s}\n stderr={s}\n", .{ res.stdout, res.stderr });
            return error.CompressFailed;
        }
    }
    // -i is a host-only dump; no GPU work.
    const res = try runCmd(al, io, &.{ BIN, "-i", slz_path });
    if (exitCode(res.term) != 0) {
        std.debug.print("info failed: stdout={s}\n stderr={s}\n", .{ res.stdout, res.stderr });
        return error.InfoFailed;
    }
    try testing.expect(res.stdout.len > 0);
}

test "CLI smoke: -b runs a 1-round bench cleanly" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const src = try al.alloc(u8, 16 * 1024);
    for (src, 0..) |*b, i| b.* = @intCast((i * 7 + 13) & 0xFF);

    const in_path = "tmp_cuda_cli_bench_in.bin";
    defer rmIfExists(io, in_path);

    try writeBytes(io, in_path, src);

    const res = try runCmd(al, io, &.{ BIN, "-b", "-r", "1", in_path });
    if (exitCode(res.term) != 0) {
        std.debug.print("bench failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
        return error.BenchFailed;
    }
    try testing.expect(res.stdout.len > 0);
}
