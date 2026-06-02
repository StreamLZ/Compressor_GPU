//! CLI integration test for `streamlz_vk.exe`.
//!
//! Spawns the freshly-built `streamlz_vk.exe` (and the CUDA-backed
//! `streamlz.exe`) via std.process.Child and checks three round-trip
//! directions on `assets/web.txt`:
//!
//!   * VK_SELF:    streamlz_vk -c -> streamlz_vk -d              (must match)
//!   * VK_TO_CUDA: streamlz_vk -c -> streamlz.exe -d             (interop)
//!   * CUDA_TO_VK: streamlz.exe -c -l 1 -> streamlz_vk -d        (interop)
//!
//! Each direction prints a one-line `CLI_<dir> PASS|FAIL` summary.

const std = @import("std");

const STREAMLZ_VK_EXE: []const u8 = "c:/Users/james.JAMESWORK2025/Repos/Compressor_GPU/zig-out/bin/streamlz_vk.exe";
const STREAMLZ_EXE: []const u8 = "c:/Users/james.JAMESWORK2025/Repos/Compressor_GPU/zig-out/bin/streamlz.exe";
const CORPUS_PATH: []const u8 = "assets/web.txt";
const TMP_DIR: []const u8 = "c:/tmp";

fn runExe(io: std.Io, allocator: std.mem.Allocator, argv: []const []const u8) !u32 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    // Copy argv into the arena so the caller doesn't need to keep them
    // alive across the spawn.
    const argv_copy = try aa.alloc([]const u8, argv.len);
    for (argv, 0..) |a, i| argv_copy[i] = a;

    const result = try std.process.run(aa, io, .{ .argv = argv_copy });
    return switch (result.term) {
        .exited => |code| code,
        else => 1,
    };
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const buf = try allocator.alloc(u8, @intCast(stat.size));
    const n = try file.readPositionalAll(io, buf, 0);
    if (n != buf.len) return error.ShortRead;
    return buf;
}

const Diff = struct {
    first_diff: usize,
    total_diffs: usize,
};

fn compareBytes(a: []const u8, b: []const u8) Diff {
    const n = @min(a.len, b.len);
    var first: usize = 0;
    var found = false;
    var total: usize = 0;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (a[i] != b[i]) {
            if (!found) {
                first = i;
                found = true;
            }
            total += 1;
        }
    }
    if (a.len != b.len) {
        if (!found) {
            first = n;
            found = true;
        }
        total += @max(a.len, b.len) - n;
    }
    return .{ .first_diff = first, .total_diffs = total };
}

fn runOne(
    w: *std.Io.Writer,
    io: std.Io,
    allocator: std.mem.Allocator,
    label: []const u8,
    src: []const u8,
    encode_argv: []const []const u8,
    decode_argv: []const []const u8,
    decoded_path: []const u8,
) !bool {
    const encode_exit = runExe(io, allocator, encode_argv) catch |err| {
        try w.print("{s} FAIL stage=spawn_encode err={s}\n", .{ label, @errorName(err) });
        return false;
    };
    if (encode_exit != 0) {
        try w.print("{s} FAIL stage=encode exit={d}\n", .{ label, encode_exit });
        return false;
    }
    const decode_exit = runExe(io, allocator, decode_argv) catch |err| {
        try w.print("{s} FAIL stage=spawn_decode err={s}\n", .{ label, @errorName(err) });
        return false;
    };
    if (decode_exit != 0) {
        try w.print("{s} FAIL stage=decode exit={d}\n", .{ label, decode_exit });
        return false;
    }
    const decoded = readFile(io, allocator, decoded_path) catch |err| {
        try w.print("{s} FAIL stage=read_decoded err={s}\n", .{ label, @errorName(err) });
        return false;
    };
    defer allocator.free(decoded);

    const diff = compareBytes(src, decoded);
    if (diff.total_diffs != 0 or src.len != decoded.len) {
        try w.print(
            "{s} FAIL bytes={d} decoded_bytes={d} first_diff={d} total_diffs={d}\n",
            .{ label, src.len, decoded.len, diff.first_diff, diff.total_diffs },
        );
        return false;
    }
    try w.print("{s} PASS bytes={d}\n", .{ label, src.len });
    return true;
}

/// Probe whether the CUDA-side streamlz.exe binary is present on disk
/// AND runnable. The interop pair (VK_TO_CUDA + CUDA_TO_VK) is the
/// load-bearing half of this test — VK_SELF (VK encode → VK decode)
/// can pass with symmetric corruption because both halves share the
/// same buggy state — so we need to know up-front whether the CUDA
/// binary is available, and refuse to pass the test if it isn't.
fn cudaInteropAvailable(io: std.Io) bool {
    var file = std.Io.Dir.cwd().openFile(io, STREAMLZ_EXE, .{}) catch return false;
    file.close(io);
    return true;
}

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    const allocator = std.heap.c_allocator;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    const src = readFile(io, allocator, CORPUS_PATH) catch |err| {
        try w.print("CLI_SETUP FAIL stage=read_corpus path={s} err={s}\n", .{ CORPUS_PATH, @errorName(err) });
        return error.ReadCorpus;
    };
    defer allocator.free(src);
    try w.print("CLI smoke — corpus={s} bytes={d}\n", .{ CORPUS_PATH, src.len });
    try w.flush();

    // Cluster F (F040): VK_SELF (VK encode → VK decode) can pass with
    // symmetric corruption — if both halves of the codec are broken the
    // same way, the byte-equal round-trip still holds. The interop pair
    // (VK_TO_CUDA + CUDA_TO_VK) is the only load-bearing direction in
    // this test. We refuse to declare PASS if the pair didn't run.
    const interop_up = cudaInteropAvailable(io);
    if (interop_up) {
        try w.print("CLI cuda_interop=available exe={s}\n", .{STREAMLZ_EXE});
    } else {
        try w.print(
            "CLI cuda_interop=UNAVAILABLE exe={s} (VK_SELF alone is INSUFFICIENT)\n",
            .{STREAMLZ_EXE},
        );
    }
    try w.flush();

    // Paths.
    const vk_self_slz = TMP_DIR ++ "/vk_self.slz";
    const vk_self_out = TMP_DIR ++ "/vk_self.out";
    const vk_to_cuda_slz = TMP_DIR ++ "/vk_to_cuda.slz";
    const vk_to_cuda_out = TMP_DIR ++ "/vk_to_cuda.out";
    const cuda_to_vk_slz = TMP_DIR ++ "/cuda_to_vk.slz";
    const cuda_to_vk_out = TMP_DIR ++ "/cuda_to_vk.out";

    var any_fail = false;
    var vk_self_ok = false;
    var vk_to_cuda_ok = false;
    var cuda_to_vk_ok = false;

    // ── 1. VK_SELF: vk_encode + vk_decode. ───────────────────────────
    {
        const enc_argv = [_][]const u8{ STREAMLZ_VK_EXE, "-c", CORPUS_PATH, "-o", vk_self_slz };
        const dec_argv = [_][]const u8{ STREAMLZ_VK_EXE, "-d", vk_self_slz, "-o", vk_self_out };
        vk_self_ok = try runOne(w, io, allocator, "CLI_VK_SELF", src, &enc_argv, &dec_argv, vk_self_out);
        if (!vk_self_ok) any_fail = true;
        try w.flush();
    }

    // ── 2. VK_TO_CUDA: vk_encode + cuda_decode. ──────────────────────
    if (interop_up) {
        const enc_argv = [_][]const u8{ STREAMLZ_VK_EXE, "-c", CORPUS_PATH, "-o", vk_to_cuda_slz };
        const dec_argv = [_][]const u8{ STREAMLZ_EXE, "-d", vk_to_cuda_slz, "-o", vk_to_cuda_out };
        vk_to_cuda_ok = try runOne(w, io, allocator, "CLI_VK_TO_CUDA", src, &enc_argv, &dec_argv, vk_to_cuda_out);
        if (!vk_to_cuda_ok) any_fail = true;
    } else {
        try w.writeAll("CLI_VK_TO_CUDA SKIP cuda_interop_unavailable\n");
    }
    try w.flush();

    // ── 3. CUDA_TO_VK: cuda_encode -l 1 + vk_decode. ────────────────
    if (interop_up) {
        const enc_argv = [_][]const u8{ STREAMLZ_EXE, "-c", "-l", "1", CORPUS_PATH, "-o", cuda_to_vk_slz };
        const dec_argv = [_][]const u8{ STREAMLZ_VK_EXE, "-d", cuda_to_vk_slz, "-o", cuda_to_vk_out };
        cuda_to_vk_ok = try runOne(w, io, allocator, "CLI_CUDA_TO_VK", src, &enc_argv, &dec_argv, cuda_to_vk_out);
        if (!cuda_to_vk_ok) any_fail = true;
    } else {
        try w.writeAll("CLI_CUDA_TO_VK SKIP cuda_interop_unavailable\n");
    }
    try w.flush();

    // Verdict logic (F040 — VK_SELF cannot be the sole gate):
    //   * interop_up + any_fail  -> FAIL (a real regression).
    //   * interop_up + all pass  -> PASS.
    //   * !interop_up + VK_SELF pass -> INSUFFICIENT (exits non-zero
    //     so a CI run on a CUDA-less box doesn't ship a green light
    //     for a VK codec that might be symmetrically broken).
    //   * !interop_up + VK_SELF fail -> FAIL (still a real regression).
    if (!interop_up) {
        if (!vk_self_ok) {
            try w.writeAll("CLI_OVERALL FAIL (VK_SELF mismatch on CUDA-less box)\n");
            return error.OneOrMoreFailed;
        }
        try w.writeAll(
            "CLI_OVERALL INSUFFICIENT — VK_SELF passed but interop pair (VK_TO_CUDA + CUDA_TO_VK) did not run; pass requires CUDA-side streamlz.exe present\n",
        );
        return error.InteropPairUnavailable;
    }

    if (any_fail) {
        try w.writeAll("CLI_OVERALL FAIL\n");
        return error.OneOrMoreFailed;
    }
    // Belt: even with interop_up + any_fail clean, refuse to declare
    // PASS unless the interop pair both ran AND passed. This is the
    // F040 invariant — VK_SELF is never the sole gate.
    if (!(vk_to_cuda_ok and cuda_to_vk_ok)) {
        try w.writeAll("CLI_OVERALL FAIL — interop pair did not pass\n");
        return error.OneOrMoreFailed;
    }
    try w.writeAll("CLI_OVERALL PASS\n");
}
