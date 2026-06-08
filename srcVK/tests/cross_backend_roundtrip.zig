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
    //
    // VK adaptation: see cli_smoke.zig::makeIo for the full diagnosis.
    // TL;DR: default `environ = .empty` makes `createWindowsBlock` hand
    // the spawned child an EMPTY env block, which stops the Vulkan
    // loader's PnP/registry ICD discovery from locating the NVIDIA
    // driver. Passing `.{ .block = .global }` forwards the parent PEB
    // environment verbatim and restores discrete-GPU visibility.
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

// VK adaptation: L2-specific helper. Compute SHA256 over a file's bytes
// (used by the cross-backend L2 byte-identity gate below). Reuses the
// already-proven readAll path so we don't reinvent file IO.
fn sha256OfFile(allocator: std.mem.Allocator, io: std.Io, rel_path: []const u8, out_hex: *[64]u8) !void {
    const bytes = try readAll(allocator, io, rel_path);
    defer allocator.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const hex = "0123456789abcdef";
    for (digest, 0..) |b, i| {
        out_hex[i * 2] = hex[b >> 4];
        out_hex[i * 2 + 1] = hex[b & 0xF];
    }
}

// VK adaptation: L2 SHA-byte-identity gate (H2 lesson). Encodes the same
// input via BOTH backends at the same level and asserts the produced
// .slz frames are byte-identical (via SHA256). Catches the H2-class
// "encoder silently produces valid-but-uncompressed output" regression
// — that frame would roundtrip cleanly but its SHA would diverge from
// CUDA's real-match output. `size` is the input length; `seed` selects
// the PRNG; `compressible` controls whether we feed PRNG bytes (random,
// incompressible) or a repeating-block payload (compressible — exercises
// the match-emission code path that H2 broke).
fn shaByteIdentityL2(
    io: std.Io,
    allocator: std.mem.Allocator,
    tag: []const u8,
    size: usize,
    seed: u64,
    compressible: bool,
) !void {
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    const src = try allocator.alloc(u8, size);
    defer allocator.free(src);
    if (compressible) {
        // Repeating 1 KiB random block — gives the encoder real matches
        // to find. An all-raw H2-regression encoder would emit ~size
        // bytes; the real encoder emits a tiny fraction.
        var block: [1024]u8 = undefined;
        var rng = std.Random.DefaultPrng.init(seed);
        rng.random().bytes(&block);
        var i: usize = 0;
        while (i < size) : (i += 1) src[i] = block[i & 1023];
    } else {
        var rng = std.Random.DefaultPrng.init(seed);
        rng.random().bytes(src);
    }

    // tmp paths keyed by tag to stay unique across parallel-ish runs.
    var in_path_buf: [128]u8 = undefined;
    var cu_slz_buf: [128]u8 = undefined;
    var vk_slz_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_l2sha_{s}_in.bin", .{tag});
    const cu_slz = try std.fmt.bufPrint(&cu_slz_buf, "tmp_cb_l2sha_{s}_cu.slz", .{tag});
    const vk_slz = try std.fmt.bufPrint(&vk_slz_buf, "tmp_cb_l2sha_{s}_vk.slz", .{tag});
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, cu_slz);
    defer rmIfExists(io, vk_slz);

    try writeBytes(io, in_path, src);

    // CUDA L2 encode.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", "2", in_path, "-o", cu_slz };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA L2 encode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.CudaEncodeFailed;
        }
    }
    // VK L2 encode.
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "2", in_path, "-o", vk_slz };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L2 encode failed: term={any} stdout={s} stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }

    // SHA byte-identity gate.
    var cu_hex: [64]u8 = undefined;
    var vk_hex: [64]u8 = undefined;
    try sha256OfFile(allocator, io, cu_slz, &cu_hex);
    try sha256OfFile(allocator, io, vk_slz, &vk_hex);
    if (!std.mem.eql(u8, &cu_hex, &vk_hex)) {
        std.debug.print(
            "L2 SHA byte-identity FAIL ({s}, size={d}): CUDA={s} VK={s}\n",
            .{ tag, size, cu_hex, vk_hex },
        );
        return error.L2ByteIdentityMismatch;
    }

    // Sanity: VK round-trip via VK decode also passes (the H2 trap was
    // valid-but-uncompressed; the SHA check above already covers it,
    // but a roundtrip is cheap defense-in-depth).
    var out_path_buf: [128]u8 = undefined;
    const out_path = try std.fmt.bufPrint(&out_path_buf, "tmp_cb_l2sha_{s}_out.bin", .{tag});
    defer rmIfExists(io, out_path);
    {
        const argv = [_][]const u8{ VK_BIN, "-d", vk_slz, "-o", out_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L2 decode failed: term={any} stdout={s} stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(allocator, io, out_path);
    defer allocator.free(got);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: VK L2 == CUDA L2 SHA byte-identity (256 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try shaByteIdentityL2(io, arena.allocator(), "256k", 256 * 1024, 0xA1B2C3D4, true);
}

test "cross-backend: VK L2 == CUDA L2 SHA byte-identity (1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try shaByteIdentityL2(io, arena.allocator(), "1m", 1 * 1024 * 1024, 0x5E5E5E5E, true);
}

test "cross-backend: VK L2 == CUDA L2 SHA byte-identity (4 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    // 4 MiB exercises the multi-chunk boundary class the L2 iter-3/4
    // fixes addressed (chunk_size = 256 KiB → 16 chunks).
    try shaByteIdentityL2(io, arena.allocator(), "4m", 4 * 1024 * 1024, 0x77777777, true);
}

test "cross-backend: VK encode -> CUDA decode (level=2, 64 KiB random) [serial_first]" {
    // VK adaptation: mirror of the L1 V→C case at the top of this file
    // for the L2 wire-compatibility direction. The SHA-identity tests
    // above already prove VK output matches CUDA byte-for-byte, but this
    // explicitly exercises CUDA's L2 decode against a VK-produced frame.
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(std.testing.allocator, io)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    var rng = std.Random.DefaultPrng.init(0xC0FE_C0FE);
    const src = try al.alloc(u8, 64 * 1024);
    rng.random().bytes(src);

    const in_path = "tmp_cb_v2c_l2_in.bin";
    const slz_path = "tmp_cb_v2c_l2.slz";
    const out_path = "tmp_cb_v2c_l2_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // VK encode at level 2.
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "2", in_path, "-o", slz_path };
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L2 encode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }
    // CUDA decode.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA L2 decode failed: term={any} stderr={s}\n", .{ res.term, res.stderr });
            return error.CudaDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}
