//! NEW per Exception 3 (no CUDA counterpart). Phase 2A-decoder iter 4c
//! cross-backend correctness gate.
//!
//! CUDA-encode -> VK-decode round-trip at L3 + L4. Mirrors the L2 cross-
//! backend pattern at cross_backend_roundtrip.zig (which landed at
//! 7bbcf77) and extends it to the higher levels that iter 4c was
//! supposed to unlock. See l3_l4_encode_roundtrip.zig for the full
//! bug-shape preamble; this file is the cross-backend half.
//!
//! Why CUDA-encode -> VK-decode specifically:
//!   The in-process VK->VK file (l3_l4_encode_roundtrip.zig) catches
//!   the iter 4c bugs from the encode-side perspective. The cross-
//!   backend tests here decode a known-good CUDA-produced .slz frame
//!   through the same VK decode path, which DISTINGUISHES:
//!     * decode-side bug  -> CUDA->VK fails matching the VK->VK signal
//!     * encode-side bug  -> CUDA->VK passes but VK->VK fails
//!   That signal split is what makes the cross-backend coverage worth
//!   the subprocess overhead.
//!
//! Per the parent runner convention, every test in this file marks
//! `[serial_first]` because each spawns child streamlz.exe + streamlz_vk.exe.
//! The cross-process Vulkan driver state contention bug documented in
//! cross_backend_roundtrip.zig applies here verbatim.

const std = @import("std");
const testing = std.testing;

const CUDA_BIN = "zig-out/bin/streamlz.exe";
const VK_BIN = "zig-out/bin/streamlz_vk.exe";

const max_corpus_bytes: usize = 1 << 30;

// VK adaptation: identical to cross_backend_roundtrip.zig::makeIo. We
// inherit the parent PEB environment so the Vulkan loader's PnP/registry
// ICD discovery can locate the discrete NVIDIA driver in the child.
fn makeIo() std.Io.Threaded {
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

// VK adaptation: see cross_backend_roundtrip.zig::argvWithVkDevice for
// the long-form rationale. TL;DR: under `zig build` the inherited env
// causes the Vulkan loader to default-pick the Intel iGPU; we splice
// `--device NVIDIA` so the child lands on the discrete GPU regardless.
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

// VK adaptation: see cross_backend_roundtrip.zig::discreteVkVisibleToChild.
// Skip cross-backend round-trip tests when the discrete VK device is
// invisible to the child process — otherwise the iGPU L1 decode bugs
// would dirty this file's signal.
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

// VK adaptation: shared body for the L3/L4 cross-backend tests. Lifted
// to a helper so each test case is one call + size + level + tag and
// the diagnostic shape stays consistent across them. `tag` keys the tmp
// filenames so parallel/sequenced runs don't stomp each other.
fn cudaEncodeVkDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    tag: []const u8,
    size: usize,
    level: u8,
    seed: u64,
) !void {
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    // Compressible deterministic payload: repeating 1 KiB random block.
    // Gives the L3/L4 encoder real matches to find so the .slz isn't
    // just an all-raw frame (which would mask decode-side bugs).
    const src = try allocator.alloc(u8, size);
    defer allocator.free(src);
    var block: [1024]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().bytes(&block);
    var i: usize = 0;
    while (i < size) : (i += 1) src[i] = block[i & 1023];

    var in_path_buf: [128]u8 = undefined;
    var slz_path_buf: [128]u8 = undefined;
    var out_path_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_l{d}_{s}_in.bin", .{ level, tag });
    const slz_path = try std.fmt.bufPrint(&slz_path_buf, "tmp_cb_l{d}_{s}.slz", .{ level, tag });
    const out_path = try std.fmt.bufPrint(&out_path_buf, "tmp_cb_l{d}_{s}_out.bin", .{ level, tag });
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // CUDA encode at requested level.
    {
        var lvl_buf: [4]u8 = undefined;
        const lvl_str = try std.fmt.bufPrint(&lvl_buf, "{d}", .{level});
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", lvl_str, in_path, "-o", slz_path };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "CUDA L{d} encode failed ({s}, size={d}): term={any} stderr={s}\n",
                .{ level, tag, size, res.term, res.stderr },
            );
            return error.CudaEncodeFailed;
        }
    }
    // VK decode of the CUDA-produced frame.
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            // Bug 2 (small-input KernelLaunchFailed) typically surfaces
            // here as a non-zero exit. Capture stdout/stderr verbatim
            // so the next workflow can grep for the dispatcher error
            // string without re-running.
            std.debug.print(
                "VK L{d} decode failed ({s}, size={d}): term={any}\n stdout={s}\n stderr={s}\n",
                .{ level, tag, size, res.term, res.stdout, res.stderr },
            );
            return error.VkDecodeFailed;
        }
    }

    const got = try readAll(allocator, io, out_path);
    defer allocator.free(got);

    if (got.len != src.len) {
        std.debug.print(
            "L{d} cross-backend SIZE MISMATCH ({s}): src.len={d} got.len={d}\n",
            .{ level, tag, src.len, got.len },
        );
        return error.DecodedSizeMismatch;
    }
    if (!std.mem.eql(u8, src, got)) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == got[first_diff]) : (first_diff += 1) {}
        var total_diffs: usize = 0;
        var j: usize = 0;
        while (j < src.len) : (j += 1) if (src[j] != got[j]) {
            total_diffs += 1;
        };
        var trailing_zeros: usize = 0;
        var k: usize = got.len;
        while (k > 0) {
            k -= 1;
            if (got[k] != 0) break;
            trailing_zeros += 1;
        }
        std.debug.print(
            "L{d} cross-backend BYTE MISMATCH ({s}): src.len={d} first_diff={d} total_diffs={d} trailing_zeros_in_got={d}\n",
            .{ level, tag, src.len, first_diff, total_diffs, trailing_zeros },
        );
        const lo = if (first_diff >= 8) first_diff - 8 else 0;
        const hi = @min(first_diff + 8, src.len);
        std.debug.print("  src [{d}..{d}]:", .{ lo, hi });
        for (src[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n  got [{d}..{d}]:", .{ lo, hi });
        for (got[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n", .{});
        return error.DecodedBytesMismatch;
    }
}

// =====================================================================
// L3 cross-backend cases
// =====================================================================
//
// Skip the < 64 KiB size class here: the in-process L3 file already
// probes the small-input KernelLaunchFailed regime, and the subprocess
// overhead of CUDA-encode + VK-decode for a 32 KiB payload is wasteful
// when the in-process variant gives a tighter, faster failure signal.

test "cross-backend: CUDA encode -> VK decode (L3, 128 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecode(arena.allocator(), io_inst.io(), "128k", 128 * 1024, 3, 0xC0DE_3333);
}

test "cross-backend: CUDA encode -> VK decode (L3, 1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecode(arena.allocator(), io_inst.io(), "1m", 1 * 1024 * 1024, 3, 0xC0DE_3001);
}

// =====================================================================
// L4 cross-backend cases
// =====================================================================

test "cross-backend: CUDA encode -> VK decode (L4, 128 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecode(arena.allocator(), io_inst.io(), "128k", 128 * 1024, 4, 0xC0DE_4444);
}

test "cross-backend: CUDA encode -> VK decode (L4, 1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecode(arena.allocator(), io_inst.io(), "1m", 1 * 1024 * 1024, 4, 0xC0DE_4001);
}

// =====================================================================
// REAL CORPUS cross-backend cases. Synthetic compressible payloads
// (repeating 1 KiB random block) produce narrow Huffman distributions
// that exercise only a tiny corner of the decoder. Real English text
// has 30-50 active symbols and exercises the full LUT + 4-stream
// parallel path. The parent agent's empirical reproduction showed
// VK->VK at L3/L4 DIFFERs (silent corruption) on real corpus but passes
// on synthetic — the same false-positive trap iter 4c fell into.
//
// These cross-backend cases load a head of assets/{web,enwik8}.txt,
// write it to a tmp file, run CUDA encode + VK decode, and byte-compare.
// =====================================================================

fn cudaEncodeVkDecodeCorpus(
    allocator: std.mem.Allocator,
    io: std.Io,
    tag: []const u8,
    corpus_rel: []const u8,
    head_bytes: usize,
    level: u8,
) !void {
    // VK adaptation: real-corpus cross-backend driver. Mirrors
    // cudaEncodeVkDecode but sources the input from a corpus head rather
    // than a synthetic compressible payload. Loads the full file via
    // readFileAlloc then writes only the head_bytes prefix as the test
    // input so the CUDA binary sees a normal file path.
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    const full = std.Io.Dir.cwd().readFileAlloc(
        io,
        corpus_rel,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(full);
    const take = @min(head_bytes, full.len);
    const src = full[0..take];

    var in_path_buf: [128]u8 = undefined;
    var slz_path_buf: [128]u8 = undefined;
    var out_path_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_l{d}_corpus_{s}_in.bin", .{ level, tag });
    const slz_path = try std.fmt.bufPrint(&slz_path_buf, "tmp_cb_l{d}_corpus_{s}.slz", .{ level, tag });
    const out_path = try std.fmt.bufPrint(&out_path_buf, "tmp_cb_l{d}_corpus_{s}_out.bin", .{ level, tag });
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // CUDA encode at requested level.
    {
        var lvl_buf: [4]u8 = undefined;
        const lvl_str = try std.fmt.bufPrint(&lvl_buf, "{d}", .{level});
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", lvl_str, in_path, "-o", slz_path };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "CUDA L{d} corpus encode failed ({s} from {s}, head={d}): term={any} stderr={s}\n",
                .{ level, tag, corpus_rel, take, res.term, res.stderr },
            );
            return error.CudaEncodeFailed;
        }
    }
    // VK decode of the CUDA-produced frame.
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "VK L{d} corpus decode failed ({s} from {s}, head={d}): term={any}\n stdout={s}\n stderr={s}\n",
                .{ level, tag, corpus_rel, take, res.term, res.stdout, res.stderr },
            );
            return error.VkDecodeFailed;
        }
    }

    const got = try readAll(allocator, io, out_path);
    defer allocator.free(got);

    if (got.len != src.len) {
        std.debug.print(
            "L{d} corpus cross-backend SIZE MISMATCH ({s} from {s}): src.len={d} got.len={d}\n",
            .{ level, tag, corpus_rel, src.len, got.len },
        );
        return error.DecodedSizeMismatch;
    }
    if (!std.mem.eql(u8, src, got)) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == got[first_diff]) : (first_diff += 1) {}
        var total_diffs: usize = 0;
        var j: usize = 0;
        while (j < src.len) : (j += 1) if (src[j] != got[j]) {
            total_diffs += 1;
        };
        var trailing_zeros: usize = 0;
        var k: usize = got.len;
        while (k > 0) {
            k -= 1;
            if (got[k] != 0) break;
            trailing_zeros += 1;
        }
        std.debug.print(
            "L{d} corpus cross-backend BYTE MISMATCH ({s} from {s}): src.len={d} first_diff={d} total_diffs={d} trailing_zeros={d}\n",
            .{ level, tag, corpus_rel, src.len, first_diff, total_diffs, trailing_zeros },
        );
        const lo = if (first_diff >= 8) first_diff - 8 else 0;
        const hi = @min(first_diff + 8, src.len);
        std.debug.print("  src [{d}..{d}]:", .{ lo, hi });
        for (src[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n  got [{d}..{d}]:", .{ lo, hi });
        for (got[lo..hi]) |b| std.debug.print(" {x:0>2}", .{b});
        std.debug.print("\n", .{});
        return error.DecodedBytesMismatch;
    }
}

test "cross-backend: CUDA encode -> VK decode (L3, web.txt 128 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecodeCorpus(arena.allocator(), io_inst.io(), "web128k", "assets/web.txt", 128 * 1024, 3);
}

test "cross-backend: CUDA encode -> VK decode (L3, enwik8.txt 256 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecodeCorpus(arena.allocator(), io_inst.io(), "enwik256k", "assets/enwik8.txt", 256 * 1024, 3);
}

test "cross-backend: CUDA encode -> VK decode (L4, web.txt 128 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecodeCorpus(arena.allocator(), io_inst.io(), "web128k", "assets/web.txt", 128 * 1024, 4);
}

// =====================================================================
// Phase 5: VK encode -> CUDA decode reverse direction (L3 + L4 + L5)
// and L5 cross-backend coverage.
//
// Background: prior iterations of this file only covered CUDA->VK. The
// SHA byte-identity gate (encode side) is already covered in
// cross_backend_roundtrip.zig at the L2 SHA helper level for L2; this
// section adds the equivalent reverse-direction round-trip gate for L3,
// L4, and L5, plus a dedicated L5 SHA byte-identity helper paralleling
// shaByteIdentityL2 from cross_backend_roundtrip.zig.
// =====================================================================

// VK adaptation: shared body for VK encode -> CUDA decode at level N.
// Mirrors cudaEncodeVkDecode but swaps the encode/decode binaries so the
// VK-produced .slz is the wire format the CUDA decoder sees. Same tag /
// size / level signature so the call sites stay symmetric.
fn vkEncodeCudaDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    tag: []const u8,
    size: usize,
    level: u8,
    seed: u64,
) !void {
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    const src = try allocator.alloc(u8, size);
    defer allocator.free(src);
    var block: [1024]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().bytes(&block);
    var i: usize = 0;
    while (i < size) : (i += 1) src[i] = block[i & 1023];

    var in_path_buf: [128]u8 = undefined;
    var slz_path_buf: [128]u8 = undefined;
    var out_path_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_v2c_l{d}_{s}_in.bin", .{ level, tag });
    const slz_path = try std.fmt.bufPrint(&slz_path_buf, "tmp_cb_v2c_l{d}_{s}.slz", .{ level, tag });
    const out_path = try std.fmt.bufPrint(&out_path_buf, "tmp_cb_v2c_l{d}_{s}_out.bin", .{ level, tag });
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // VK encode at requested level.
    {
        var lvl_buf: [4]u8 = undefined;
        const lvl_str = try std.fmt.bufPrint(&lvl_buf, "{d}", .{level});
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", lvl_str, in_path, "-o", slz_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "VK L{d} encode failed ({s}, size={d}): term={any}\n stdout={s}\n stderr={s}\n",
                .{ level, tag, size, res.term, res.stdout, res.stderr },
            );
            return error.VkEncodeFailed;
        }
    }
    // CUDA decode of the VK-produced frame.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "CUDA L{d} decode failed ({s}, size={d}): term={any} stderr={s}\n",
                .{ level, tag, size, res.term, res.stderr },
            );
            return error.CudaDecodeFailed;
        }
    }

    const got = try readAll(allocator, io, out_path);
    defer allocator.free(got);

    if (got.len != src.len) {
        std.debug.print(
            "L{d} VK->CUDA SIZE MISMATCH ({s}): src.len={d} got.len={d}\n",
            .{ level, tag, src.len, got.len },
        );
        return error.DecodedSizeMismatch;
    }
    if (!std.mem.eql(u8, src, got)) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == got[first_diff]) : (first_diff += 1) {}
        std.debug.print(
            "L{d} VK->CUDA BYTE MISMATCH ({s}): src.len={d} first_diff={d}\n",
            .{ level, tag, src.len, first_diff },
        );
        return error.DecodedBytesMismatch;
    }
}

// VK adaptation: real-corpus variant of vkEncodeCudaDecode. Mirrors
// cudaEncodeVkDecodeCorpus but swaps backends. Used for L3/L4/L5 corpus
// reverse coverage on web.txt + enwik8.txt heads.
fn vkEncodeCudaDecodeCorpus(
    allocator: std.mem.Allocator,
    io: std.Io,
    tag: []const u8,
    corpus_rel: []const u8,
    head_bytes: usize,
    level: u8,
) !void {
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    const full = std.Io.Dir.cwd().readFileAlloc(
        io,
        corpus_rel,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(full);
    const take = @min(head_bytes, full.len);
    const src = full[0..take];

    var in_path_buf: [128]u8 = undefined;
    var slz_path_buf: [128]u8 = undefined;
    var out_path_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_v2c_l{d}_corp_{s}_in.bin", .{ level, tag });
    const slz_path = try std.fmt.bufPrint(&slz_path_buf, "tmp_cb_v2c_l{d}_corp_{s}.slz", .{ level, tag });
    const out_path = try std.fmt.bufPrint(&out_path_buf, "tmp_cb_v2c_l{d}_corp_{s}_out.bin", .{ level, tag });
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    // VK encode.
    {
        var lvl_buf: [4]u8 = undefined;
        const lvl_str = try std.fmt.bufPrint(&lvl_buf, "{d}", .{level});
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", lvl_str, in_path, "-o", slz_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "VK L{d} corpus encode failed ({s} from {s}, head={d}): term={any}\n stdout={s}\n stderr={s}\n",
                .{ level, tag, corpus_rel, take, res.term, res.stdout, res.stderr },
            );
            return error.VkEncodeFailed;
        }
    }
    // CUDA decode.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-d", slz_path, "-o", out_path };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print(
                "CUDA L{d} corpus decode failed ({s} from {s}, head={d}): term={any} stderr={s}\n",
                .{ level, tag, corpus_rel, take, res.term, res.stderr },
            );
            return error.CudaDecodeFailed;
        }
    }

    const got = try readAll(allocator, io, out_path);
    defer allocator.free(got);

    if (got.len != src.len or !std.mem.eql(u8, src, got)) {
        std.debug.print(
            "L{d} corpus VK->CUDA MISMATCH ({s} from {s}): src.len={d} got.len={d}\n",
            .{ level, tag, corpus_rel, src.len, got.len },
        );
        return error.DecodedBytesMismatch;
    }
}

// VK adaptation: SHA helper mirroring sha256OfFile in
// cross_backend_roundtrip.zig — same loader, same digest, same hex
// encode. Kept local so we don't reach across test files.
fn sha256OfFileLocal(allocator: std.mem.Allocator, io: std.Io, rel_path: []const u8, out_hex: *[64]u8) !void {
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

// VK adaptation: L5 SHA byte-identity gate. Mirrors shaByteIdentityL2 in
// cross_backend_roundtrip.zig but at level 5 (chain parser + Huffman),
// which exercises the entire Phase 2B chain-parser port (A-013/A-014).
// VK and CUDA L5 encoders must produce byte-identical .slz frames on
// compressible inputs.
fn shaByteIdentityL5(
    io: std.Io,
    allocator: std.mem.Allocator,
    tag: []const u8,
    size: usize,
    seed: u64,
) !void {
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    const src = try allocator.alloc(u8, size);
    defer allocator.free(src);
    var block: [1024]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().bytes(&block);
    var i: usize = 0;
    while (i < size) : (i += 1) src[i] = block[i & 1023];

    var in_path_buf: [128]u8 = undefined;
    var cu_slz_buf: [128]u8 = undefined;
    var vk_slz_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_l5sha_{s}_in.bin", .{tag});
    const cu_slz = try std.fmt.bufPrint(&cu_slz_buf, "tmp_cb_l5sha_{s}_cu.slz", .{tag});
    const vk_slz = try std.fmt.bufPrint(&vk_slz_buf, "tmp_cb_l5sha_{s}_vk.slz", .{tag});
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, cu_slz);
    defer rmIfExists(io, vk_slz);

    try writeBytes(io, in_path, src);

    // CUDA L5 encode.
    {
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", "5", in_path, "-o", cu_slz };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA L5 encode failed ({s}): term={any} stderr={s}\n", .{ tag, res.term, res.stderr });
            return error.CudaEncodeFailed;
        }
    }
    // VK L5 encode.
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "5", in_path, "-o", vk_slz };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L5 encode failed ({s}): term={any} stdout={s} stderr={s}\n", .{ tag, res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }

    var cu_hex: [64]u8 = undefined;
    var vk_hex: [64]u8 = undefined;
    try sha256OfFileLocal(allocator, io, cu_slz, &cu_hex);
    try sha256OfFileLocal(allocator, io, vk_slz, &vk_hex);
    if (!std.mem.eql(u8, &cu_hex, &vk_hex)) {
        std.debug.print(
            "L5 SHA byte-identity FAIL ({s}, size={d}): CUDA={s} VK={s}\n",
            .{ tag, size, cu_hex, vk_hex },
        );
        return error.L5ByteIdentityMismatch;
    }
}

// =====================================================================
// L3 / L4 VK encode -> CUDA decode reverse direction (Phase 5)
// =====================================================================

test "cross-backend: VK encode -> CUDA decode (L3, 128 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecode(arena.allocator(), io_inst.io(), "128k", 128 * 1024, 3, 0xC0DE_3133);
}

test "cross-backend: VK encode -> CUDA decode (L3, 1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecode(arena.allocator(), io_inst.io(), "1m", 1 * 1024 * 1024, 3, 0xC0DE_3134);
}

test "cross-backend: VK encode -> CUDA decode (L3, web.txt 128 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecodeCorpus(arena.allocator(), io_inst.io(), "web128k", "assets/web.txt", 128 * 1024, 3);
}

test "cross-backend: VK encode -> CUDA decode (L3, enwik8.txt 256 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecodeCorpus(arena.allocator(), io_inst.io(), "enwik256k", "assets/enwik8.txt", 256 * 1024, 3);
}

test "cross-backend: VK encode -> CUDA decode (L4, 128 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecode(arena.allocator(), io_inst.io(), "128k", 128 * 1024, 4, 0xC0DE_4144);
}

test "cross-backend: VK encode -> CUDA decode (L4, 1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecode(arena.allocator(), io_inst.io(), "1m", 1 * 1024 * 1024, 4, 0xC0DE_4145);
}

test "cross-backend: VK encode -> CUDA decode (L4, web.txt 128 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecodeCorpus(arena.allocator(), io_inst.io(), "web128k", "assets/web.txt", 128 * 1024, 4);
}

test "cross-backend: VK encode -> CUDA decode (L4, enwik8.txt 256 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecodeCorpus(arena.allocator(), io_inst.io(), "enwik256k", "assets/enwik8.txt", 256 * 1024, 4);
}

// =====================================================================
// L5 cross-backend (Phase 5 — entirely new coverage)
// L5 exercises the chain parser (Phase 2B) + Huffman pipeline together.
// =====================================================================

test "cross-backend: CUDA encode -> VK decode (L5, 128 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecode(arena.allocator(), io_inst.io(), "128k", 128 * 1024, 5, 0xC0DE_5555);
}

test "cross-backend: CUDA encode -> VK decode (L5, 1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecode(arena.allocator(), io_inst.io(), "1m", 1 * 1024 * 1024, 5, 0xC0DE_5001);
}

test "cross-backend: VK encode -> CUDA decode (L5, 128 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecode(arena.allocator(), io_inst.io(), "128k", 128 * 1024, 5, 0xC0DE_5256);
}

test "cross-backend: VK encode -> CUDA decode (L5, 1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkEncodeCudaDecode(arena.allocator(), io_inst.io(), "1m", 1 * 1024 * 1024, 5, 0xC0DE_5257);
}

test "cross-backend: VK encode -> VK decode (L5, 1 MiB ascending pattern) [serial_first]" {
    // VK adaptation: parallels the L1 VK->VK ascending-pattern test in
    // cross_backend_roundtrip.zig:214. Ensures L5 in-process round trip
    // works on a non-random deterministic payload (catches encoder
    // off-by-one / pattern-degenerate regressions that random payloads
    // mask). 1 MiB rather than 8 KiB so the chain parser meaningfully
    // exercises the rehash + recent-offset paths.
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(std.testing.allocator, io)) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    const src = try al.alloc(u8, 1 * 1024 * 1024);
    for (src, 0..) |*b, i| b.* = @intCast(i & 0xFF);

    const in_path = "tmp_cb_vk_self_l5_in.bin";
    const slz_path = "tmp_cb_vk_self_l5.slz";
    const out_path = "tmp_cb_vk_self_l5_out.bin";
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "5", in_path, "-o", slz_path };
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L5 encode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runVkCmd(al, io, &argv);
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L5 decode failed: term={any}\n stdout={s}\n stderr={s}\n", .{ res.term, res.stdout, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = try readAll(al, io, out_path);
    try testing.expectEqualSlices(u8, src, got);
}

test "cross-backend: VK L5 == CUDA L5 SHA byte-identity (256 KiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try shaByteIdentityL5(io, arena.allocator(), "256k", 256 * 1024, 0xA5B5C5D5);
}

test "cross-backend: VK L5 == CUDA L5 SHA byte-identity (1 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try shaByteIdentityL5(io, arena.allocator(), "1m", 1 * 1024 * 1024, 0x5F5F5F5F);
}

test "cross-backend: VK L5 == CUDA L5 SHA byte-identity (4 MiB compressible) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try shaByteIdentityL5(io, arena.allocator(), "4m", 4 * 1024 * 1024, 0x88888888);
}

test "cross-backend: CUDA encode -> VK decode (L5, web.txt 128 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecodeCorpus(arena.allocator(), io_inst.io(), "web128k", "assets/web.txt", 128 * 1024, 5);
}

test "cross-backend: CUDA encode -> VK decode (L5, enwik8.txt 256 KiB head, real corpus) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try cudaEncodeVkDecodeCorpus(arena.allocator(), io_inst.io(), "enwik256k", "assets/enwik8.txt", 256 * 1024, 5);
}

// =====================================================================
// L5 chain-parser edge-case hardening (2026-06-08)
//
// The shaByteIdentityL5 tests above use a 1 KiB random block repeated to
// fill the buffer — degenerate self-referencing pattern (every block past
// the first is one big LZ match). That exercises far less of the chain
// parser than real text. These tests construct synthetic inputs that
// specifically target the chain parser's threshold branches:
//
//   1. LONG_LIT_RUN_THRESHOLD (=64) — bumps minimum_match_length and
//      zeros recent_match_length when a literal run crosses 64 bytes.
//   2. CHAIN_MAX_STEPS (=8) — truncates the first_hash chain walk after
//      8 candidates; if VK and CUDA pick different "best" matches when
//      the chain is deeper than 8, output diverges.
//   3. Mixed near/far offsets — isMatchBetter's OFFSET_CLASS_LEN_MARGIN
//      logic decides between a near match and a longer far match.
//   4. Block 1 → Block 2 boundary — recent_offset must survive the
//      block transition; if either backend resets it, output diverges.
//
// Each test ASSERTS VK encode SHA == CUDA encode SHA on a constructed
// input. Both backends must take the same code-path through the chain
// parser to produce byte-identical output. Acts as a regression guard
// against future chain-parser changes that might subtly mis-handle one
// of these branches.

/// SHA byte-identity test taking a caller-supplied buffer (vs the
/// seeded-RNG generator in shaByteIdentityL5). Used by the chain-parser
/// edge-case hardening tests below.
fn shaByteIdentityL5WithSrc(
    io: std.Io,
    allocator: std.mem.Allocator,
    tag: []const u8,
    src: []const u8,
) !void {
    if (!fileExists(io, CUDA_BIN) or !fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    var in_path_buf: [128]u8 = undefined;
    var cu_slz_buf: [128]u8 = undefined;
    var vk_slz_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_l5harden_{s}_in.bin", .{tag});
    const cu_slz = try std.fmt.bufPrint(&cu_slz_buf, "tmp_cb_l5harden_{s}_cu.slz", .{tag});
    const vk_slz = try std.fmt.bufPrint(&vk_slz_buf, "tmp_cb_l5harden_{s}_vk.slz", .{tag});
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, cu_slz);
    defer rmIfExists(io, vk_slz);

    try writeBytes(io, in_path, src);

    {
        const argv = [_][]const u8{ CUDA_BIN, "-c", "-l", "5", in_path, "-o", cu_slz };
        const res = try runCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("CUDA L5 harden encode failed ({s}): term={any} stderr={s}\n", .{ tag, res.term, res.stderr });
            return error.CudaEncodeFailed;
        }
    }
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", "5", in_path, "-o", vk_slz };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("VK L5 harden encode failed ({s}): term={any} stdout={s} stderr={s}\n", .{ tag, res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }

    var cu_hex: [64]u8 = undefined;
    var vk_hex: [64]u8 = undefined;
    try sha256OfFileLocal(allocator, io, cu_slz, &cu_hex);
    try sha256OfFileLocal(allocator, io, vk_slz, &vk_hex);
    if (!std.mem.eql(u8, &cu_hex, &vk_hex)) {
        std.debug.print(
            "L5 harden SHA byte-identity FAIL ({s}, size={d}): CUDA={s} VK={s}\n",
            .{ tag, src.len, cu_hex, vk_hex },
        );
        return error.L5HardenByteIdentityMismatch;
    }
}

test "cross-backend: L5 chain-parser edge — long-literal-run threshold (LONG_LIT_RUN_THRESHOLD=64) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    // Pattern: 128 KiB. Recurring "ABCD" markers separated by exactly 70
    // non-matching bytes — every match candidate is gated by a literal
    // run crossing the 64-byte threshold. Each "ABCD" at offset O has a
    // potential 4-byte recent-match against the previous "ABCD" at O-74.
    // The threshold bump changes whether the encoder takes the match or
    // continues as literals; VK and CUDA must agree.
    const total: usize = 128 * 1024;
    const stride: usize = 74; // 4 marker bytes + 70 literal bytes
    const src = try al.alloc(u8, total);
    var rng = std.Random.DefaultPrng.init(0xA1B1_C1D1_E1F1_0011);
    rng.random().bytes(src);
    var off: usize = 0;
    while (off + 4 <= total) : (off += stride) {
        src[off + 0] = 'A';
        src[off + 1] = 'B';
        src[off + 2] = 'C';
        src[off + 3] = 'D';
    }
    try shaByteIdentityL5WithSrc(io, al, "longlitrun", src);
}

test "cross-backend: L5 chain-parser edge — chain walk truncation (CHAIN_MAX_STEPS=8) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    // Pattern: 64 KiB. Same 4-byte prefix "WXYZ" repeated at 32 positions
    // densely packed (every 16 bytes), each followed by 12 unique bytes
    // so the extension is different at every candidate. The 33rd "WXYZ"
    // sees a hash chain of length 32 → walks at most CHAIN_MAX_STEPS=8
    // before truncating → picks the best of the first 8 candidates,
    // which may not be the globally best match. VK and CUDA must walk
    // the chain in the same order to pick the same candidate.
    const total: usize = 64 * 1024;
    const stride: usize = 16;
    const src = try al.alloc(u8, total);
    var rng = std.Random.DefaultPrng.init(0xB2C2_D2E2_F200_2233);
    rng.random().bytes(src);
    var off: usize = 0;
    while (off + stride <= total) : (off += stride) {
        src[off + 0] = 'W';
        src[off + 1] = 'X';
        src[off + 2] = 'Y';
        src[off + 3] = 'Z';
    }
    try shaByteIdentityL5WithSrc(io, al, "chaintrunc", src);
}

test "cross-backend: L5 chain-parser edge — mixed near/far offset classes [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    // Pattern: 256 KiB (4 LZ blocks). Recurring 8-byte signature
    // "FEDCBA09" placed at positions chosen so the same signature
    // appears at both near (<64 KiB) and far (>=64 KiB) offsets from
    // the third copy. Exercises isMatchBetter's near-vs-far class
    // comparison + OFFSET_CLASS_LEN_MARGIN logic.
    //
    // Positions: 100, 200 (near, +100 from first), 70000 (far, +69900
    // from first; near 70000-200=69800 from second), 70200 (near
    // +200 from 70000, far +70100 from first), 200100 (far from
    // every prior copy), 200200 (near +100 from 200100, far from rest).
    const total: usize = 256 * 1024;
    const src = try al.alloc(u8, total);
    var rng = std.Random.DefaultPrng.init(0xC3D3_E3F3_0033_4455);
    rng.random().bytes(src);
    const sig = [8]u8{ 0xFE, 0xDC, 0xBA, 0x09, 0x87, 0x65, 0x43, 0x21 };
    const positions = [_]usize{ 100, 200, 70000, 70200, 200100, 200200 };
    for (positions) |p| {
        if (p + sig.len <= total) {
            @memcpy(src[p..][0..sig.len], &sig);
        }
    }
    try shaByteIdentityL5WithSrc(io, al, "mixedoffset", src);
}

test "cross-backend: L5 chain-parser edge — block1→block2 boundary recent_offset carryover [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    const io = io_inst.io();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const al = arena.allocator();

    // Pattern: 128 KiB (spans LZ_BLOCK_SIZE=64 KiB boundary). 8-byte
    // signature placed at four positions that establish a recent_offset
    // in block 1 and then exercise it across the block boundary in
    // block 2:
    //   - pos 100: first occurrence (no recent_offset yet)
    //   - pos 200: second occurrence (sets recent_offset = -100)
    //   - pos 65540: 4 bytes past block boundary; pos-100 = 65440 (also
    //     in block 2) is where recent_offset would point. If the
    //     encoder PRESERVES recent_offset across the boundary, the
    //     recent-match candidate at pos 65540 against pos 65440 is
    //     considered. We also place the signature at pos 65440 to make
    //     the recent-match valid.
    //   - pos 65440: companion to pos 65540
    //
    // VK and CUDA must make the same decision about whether to carry
    // recent_offset across the boundary.
    const total: usize = 128 * 1024;
    const src = try al.alloc(u8, total);
    var rng = std.Random.DefaultPrng.init(0xD4E4_F400_4455_66AA);
    rng.random().bytes(src);
    const sig = [8]u8{ 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0 };
    const positions = [_]usize{ 100, 200, 65440, 65540 };
    for (positions) |p| {
        if (p + sig.len <= total) {
            @memcpy(src[p..][0..sig.len], &sig);
        }
    }
    try shaByteIdentityL5WithSrc(io, al, "blockboundary", src);
}

// =====================================================================
// A-022 regression: sc=0.5 path on L1/L2 raw decode (2026-06-09)
//
// The Phase 5 perf-sweep + ptest matrix capped at enwik8 (95 MB) and
// silesia (203 MB), both BELOW the saturation_bytes threshold
// (`resolveScGroupSize`, ~208 MB on RTX 4060 Ti) that flips
// sc_group_size from 0.25 → 0.5. With sc=0.25 each sub-chunk is 64 KiB
// = LZ_BLOCK_SIZE so off32 stays empty and the raw decode kernel takes
// the lean `decodeSubChunkRawMode` path. With sc=0.5 the sub-chunk
// decomp_size becomes 128 KiB and the dispatcher routes to
// `decodeSubChunkGeneral_false` — where A-022 (3 sites in
// lz_decode_general.glsl that compared an absolute byte offset against
// a relative stream size) silently dropped every bounded literal write.
//
// Repro on enwik9 (1 GB, > 200 MB) showed deterministic 2-byte zeros
// at every `[[xx:LANG]]` wiki-link token past the first sub-chunk
// boundary (byte 130906 = ~128 KiB). CUDA decoded the same .slz
// correctly; only VK was wrong. Catalogued + fixed at A-022.
//
// This regression test forces sc=0.5 via the CLI `--sc 0.5` flag on a
// 4 MiB enwik8 head (text-like content, plenty of off32 entries) at L1
// and L2 — both routes through `decodeSubChunkGeneral_false`.

fn vkRoundtripWithScOverride(
    allocator: std.mem.Allocator,
    io: std.Io,
    tag: []const u8,
    corpus_rel: []const u8,
    head_bytes: usize,
    level: u8,
    sc_override: []const u8,
) !void {
    if (!fileExists(io, VK_BIN)) return error.SkipZigTest;
    if (!discreteVkVisibleToChild(allocator, io)) return error.SkipZigTest;

    const full = std.Io.Dir.cwd().readFileAlloc(
        io,
        corpus_rel,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(full);
    const take = @min(head_bytes, full.len);
    const src = full[0..take];

    var in_path_buf: [128]u8 = undefined;
    var slz_path_buf: [128]u8 = undefined;
    var out_path_buf: [128]u8 = undefined;
    const in_path = try std.fmt.bufPrint(&in_path_buf, "tmp_cb_a022_{s}_in.bin", .{tag});
    const slz_path = try std.fmt.bufPrint(&slz_path_buf, "tmp_cb_a022_{s}.slz", .{tag});
    const out_path = try std.fmt.bufPrint(&out_path_buf, "tmp_cb_a022_{s}_out.bin", .{tag});
    defer rmIfExists(io, in_path);
    defer rmIfExists(io, slz_path);
    defer rmIfExists(io, out_path);

    try writeBytes(io, in_path, src);

    var lvl_buf: [4]u8 = undefined;
    const lvl_str = try std.fmt.bufPrint(&lvl_buf, "{d}", .{level});
    {
        const argv = [_][]const u8{ VK_BIN, "-c", "-l", lvl_str, "--sc", sc_override, in_path, "-o", slz_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("VK A-022 encode failed ({s}): term={any} stdout={s} stderr={s}\n", .{ tag, res.term, res.stdout, res.stderr });
            return error.VkEncodeFailed;
        }
    }
    {
        const argv = [_][]const u8{ VK_BIN, "-d", slz_path, "-o", out_path };
        const res = try runVkCmd(allocator, io, &argv);
        defer {
            allocator.free(res.stdout);
            allocator.free(res.stderr);
        }
        if (exitCode(res.term) != 0) {
            std.debug.print("VK A-022 decode failed ({s}): term={any} stdout={s} stderr={s}\n", .{ tag, res.term, res.stdout, res.stderr });
            return error.VkDecodeFailed;
        }
    }
    const got = std.Io.Dir.cwd().readFileAlloc(io, out_path, allocator, @enumFromInt(max_corpus_bytes)) catch return error.ReadOutputFailed;
    defer allocator.free(got);
    if (got.len != src.len) {
        std.debug.print("A-022 size mismatch ({s}): got={d} want={d}\n", .{ tag, got.len, src.len });
        return error.A022SizeMismatch;
    }
    if (!std.mem.eql(u8, src, got)) {
        var first_diff: usize = 0;
        while (first_diff < src.len and src[first_diff] == got[first_diff]) : (first_diff += 1) {}
        std.debug.print("A-022 byte mismatch ({s}): first_diff={d} (=0x{X})\n", .{ tag, first_diff, first_diff });
        return error.A022ByteMismatch;
    }
}

test "A-022 regression: VK L1 enwik8 4 MiB head with --sc 0.5 (sc=0.5 raw decode path) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkRoundtripWithScOverride(arena.allocator(), io_inst.io(), "enwik8_4m_l1_sc05", "assets/enwik8.txt", 4 * 1024 * 1024, 1, "0.5");
}

test "A-022 regression: VK L2 enwik8 4 MiB head with --sc 0.5 (sc=0.5 raw decode path) [serial_first]" {
    var io_inst = makeIo();
    defer io_inst.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    try vkRoundtripWithScOverride(arena.allocator(), io_inst.io(), "enwik8_4m_l2_sc05", "assets/enwik8.txt", 4 * 1024 * 1024, 2, "0.5");
}
