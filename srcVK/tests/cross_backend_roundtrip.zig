//! NEW per Exception 3 (no CUDA counterpart).
//!
//! CUDA<->VK cross-backend round-trip matrix: encode via CUDA, decode via
//! VK; encode via VK, decode via CUDA. Levels 1-2 full; levels 3-5 once
//! Huffman lands.
//!
//! Drives the live `streamlz.exe` (CUDA) and `streamlz_vk.exe` (VK)
//! binaries via std.process.run. When either binary is missing the
//! tests skip with SkipZigTest so a CUDA-toolchain-less dev box still
//! reports a clean run.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const testing = std.testing;

const CUDA_BIN = "zig-out/bin/streamlz.exe";
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

fn runCmd(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !std.process.RunResult {
    return std.process.run(allocator, io, .{ .argv = argv });
}

fn exitCode(term: std.process.Child.Term) u32 {
    return switch (term) {
        .exited => |c| c,
        else => 1,
    };
}

test "cross-backend: CUDA encode -> VK decode (level=1, 64 KiB random)" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    var rng = std.Random.DefaultPrng.init(0x1234_5678);
    const src = try al.alloc(u8, 64 * 1024);
    rng.random().bytes(src);

    const in_path = "tmp_cb_c2v_in.bin";
    const slz_path = "tmp_cb_c2v.slz";
    const out_path = "tmp_cb_c2v_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // CUDA encode.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", "1", in_path, "-o", slz_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA encode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.CudaEncodeFailed;
        }
    }
    // VK decode.
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK decode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: VK encode -> CUDA decode (level=1, 64 KiB random)" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    var rng = std.Random.DefaultPrng.init(0xAABB_CCDD);
    const src = try al.alloc(u8, 64 * 1024);
    rng.random().bytes(src);

    const in_path = "tmp_cb_v2c_in.bin";
    const slz_path = "tmp_cb_v2c.slz";
    const out_path = "tmp_cb_v2c_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // VK encode.
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "1", in_path, "-o", slz_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK encode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.VkEncodeFailed;
        }
    }
    // CUDA decode.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA decode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.CudaDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: VK encode -> VK decode (level=1, ascending pattern)" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const src = try al.alloc(u8, 8 * 1024);
    for (src, 0..) |*b, i| b.* = @intCast(i & 0xFF);

    const in_path = "tmp_cb_vk_self_in.bin";
    const slz_path = "tmp_cb_vk_self.slz";
    const out_path = "tmp_cb_vk_self_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // VK encode.
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "1", in_path, "-o", slz_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK encode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.VkEncodeFailed;
        }
    }
    // VK decode.
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK decode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: CUDA encode -> VK decode (level=2, 64 KiB random)" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    var rng = std.Random.DefaultPrng.init(0xBEEF_0001);
    const src = try al.alloc(u8, 64 * 1024);
    rng.random().bytes(src);

    const in_path = "tmp_cb_c2v_l2_in.bin";
    const slz_path = "tmp_cb_c2v_l2.slz";
    const out_path = "tmp_cb_c2v_l2_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // CUDA encode at level 2.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", "2", in_path, "-o", slz_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA encode L2 failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.CudaEncodeFailed;
        }
    }
    // VK decode at L2 (expected to work or surface NotImplementedL2).
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK decode L2 failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}
