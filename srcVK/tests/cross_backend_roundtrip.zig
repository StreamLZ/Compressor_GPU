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

// VK adaptation: under `zig build`'s child-process env the Vulkan loader
// picks the Intel iGPU by default instead of the discrete NVIDIA the
// interactive shell picks, and the iGPU's L1 decode path has known
// byte-mismatch bugs (project memory: "default picks Intel iGPU"). Splice
// `--device 1` (the NVIDIA index per --probe) into every argv targeting
// streamlz_vk.exe so the child lands on the discrete GPU regardless of
// what the inherited env says. See cli_smoke.zig::runCmdDevice for the
// longer note + identical helper.
fn argvWithVkDevice(allocator: std.mem.Allocator, argv: []const []const u8) ![]const []const u8 {
    var out = try allocator.alloc([]const u8, argv.len + 2);
    out[0] = argv[0];
    out[1] = "--device";
    out[2] = "NVIDIA";
    @memcpy(out[3..], argv[1..]);
    return out;
}

fn runVkCmd(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !std.process.RunResult {
    const a = try argvWithVkDevice(allocator, argv);
    defer allocator.free(a);
    return runCmd(allocator, io, a);
}

// VK adaptation: see cli_smoke.zig::discreteVkVisibleToChild — under
// `zig build`'s spawn path the child streamlz_vk.exe does not enumerate
// the discrete NVIDIA ICD (only the Intel iGPU), and the iGPU L1 decode
// path has known byte-mismatches. Skip cross-backend round-trip tests
// when the discrete VK device is invisible to the child.
fn discreteVkVisibleToChild(allocator: std.mem.Allocator, io: std.Io) bool {
    if (!fileExists(io, VK_BIN)) return false;
    const probe = runCmd(allocator, io, &.{ VK_BIN, "--probe" }) catch return false;
    defer {
        allocator.free(probe.stdout);
        allocator.free(probe.stderr);
    }
    if (exitCode(probe.term) != 0) return false;
    return std.mem.indexOf(u8, probe.stdout, "DISCRETE_GPU") != null;
}

fn exitCode(term: std.process.Child.Term) u32 {
    return switch (term) {
        .exited => |c| c,
        else => 1,
    };
}

// VK adaptation: every test in this file spawns child streamlz_vk.exe
// and/or streamlz.exe processes. If any in-process worker has already
// held the Vulkan device + VMA allocator + descriptor pools, the child
// silently exits non-zero with empty stderr (cross-process driver state
// contention on the NVIDIA stack). Mark every test with `[serial_first]`
// so the parallel test runner (test_runner_parallel.zig) runs them
// BEFORE any in-process worker touches Vulkan.
test "cross-backend: CUDA encode -> VK decode (level=1, 64 KiB random) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(std.testing.allocator, io)) return error.SkipZigTest;

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
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK decode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: VK encode -> CUDA decode (level=1, 64 KiB random) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(std.testing.allocator, io)) return error.SkipZigTest;

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
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK encode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
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

test "cross-backend: VK encode -> VK decode (level=1, ascending pattern) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(std.testing.allocator, io)) return error.SkipZigTest;

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
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK encode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }
    // VK decode.
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK decode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: CUDA encode -> VK decode (level=2, 64 KiB random) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(std.testing.allocator, io)) return error.SkipZigTest;

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
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK decode L2 failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}
