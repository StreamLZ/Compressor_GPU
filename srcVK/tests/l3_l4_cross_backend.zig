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
