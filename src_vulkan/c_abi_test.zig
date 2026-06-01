//! C-ABI round-trip test: compresses + decompresses a representative
//! buffer through the public `_vk` entry points and asserts byte-equal
//! round trip. Wired in build.zig as `zig build vk-abi-test`.
//!
//! Exercises:
//!   * slzCreate_vk / slzDestroy_vk
//!   * slzCompressBound_vk
//!   * slzCompressHost_vk (level=1)
//!   * slzDecompressHost_vk
//!
//! Prints a single one-line `C_ABI_ROUNDTRIP PASS bytes=<n> slz_bytes=<m>`
//! summary on success and exits 0, or `FAIL stage=<x> err=<y>` and exits 1
//! on any failure. CI / the parent workflow grep for the PASS prefix.

const std = @import("std");

const vk_abi = @import("streamlz_gpu_vk.zig");

const CORPUS_PATH: []const u8 = "assets/web.txt";
const PREFIX_SIZE: usize = 1024 * 1024; // 1 MiB prefix — enough to exercise multi-chunk + non-trivial ratios.

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    const allocator = std.heap.c_allocator;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    // ── 1. Allocate input (1 MiB prefix of assets/web.txt). ─────────
    const input: []u8 = blk: {
        var file = std.Io.Dir.cwd().openFile(io, CORPUS_PATH, .{}) catch {
            try w.print("C_ABI_ROUNDTRIP FAIL stage=open_corpus path={s}\n", .{CORPUS_PATH});
            return error.OpenFailed;
        };
        defer file.close(io);
        const stat = file.stat(io) catch {
            try w.print("C_ABI_ROUNDTRIP FAIL stage=stat_corpus\n", .{});
            return error.StatFailed;
        };
        const want: usize = @min(PREFIX_SIZE, @as(usize, @intCast(stat.size)));
        const buf = try allocator.alloc(u8, want);
        const n = file.readPositionalAll(io, buf, 0) catch {
            allocator.free(buf);
            try w.print("C_ABI_ROUNDTRIP FAIL stage=read_corpus\n", .{});
            return error.ReadFailed;
        };
        if (n != want) {
            allocator.free(buf);
            try w.print("C_ABI_ROUNDTRIP FAIL stage=short_read want={d} got={d}\n", .{ want, n });
            return error.ShortRead;
        }
        break :blk buf;
    };
    defer allocator.free(input);

    // ── 2. Create handle. ───────────────────────────────────────────
    var handle: ?*vk_abi.VkContext = null;
    const cr = vk_abi.slzCreate_vk(&handle);
    if (cr != 0 or handle == null) {
        try w.print("C_ABI_ROUNDTRIP FAIL stage=create rc={d}\n", .{cr});
        return error.CreateFailed;
    }
    defer vk_abi.slzDestroy_vk(handle);

    // ── 3. Compress. ────────────────────────────────────────────────
    const bound = vk_abi.slzCompressBound_vk(input.len);
    const slz_buf = try allocator.alloc(u8, bound);
    defer allocator.free(slz_buf);

    const opts: vk_abi.CompressOpts = .{ .level = 1 };

    const cz = vk_abi.slzCompressHost_vk(
        handle,
        input.ptr,
        input.len,
        slz_buf.ptr,
        slz_buf.len,
        opts,
    );
    if (cz < 0) {
        try w.print("C_ABI_ROUNDTRIP FAIL stage=compress rc={d}\n", .{cz});
        return error.CompressFailed;
    }
    const slz_bytes: usize = @intCast(cz);

    // ── 4. Decompress. ──────────────────────────────────────────────
    // Pre-allocate input.len + 64 bytes of safe-space slack; the
    // Vulkan decoder writes exactly `original_size` bytes, but
    // matching the CUDA decompress contract makes the test buffer-
    // shape independent.
    const decompressed = try allocator.alloc(u8, input.len + 64);
    defer allocator.free(decompressed);

    const dz = vk_abi.slzDecompressHost_vk(
        handle,
        slz_buf.ptr,
        slz_bytes,
        decompressed.ptr,
        decompressed.len,
        .{},
    );
    if (dz < 0) {
        try w.print("C_ABI_ROUNDTRIP FAIL stage=decompress rc={d}\n", .{dz});
        return error.DecompressFailed;
    }
    const decoded_bytes: usize = @intCast(dz);

    // ── 5. Compare. ─────────────────────────────────────────────────
    if (decoded_bytes != input.len) {
        try w.print(
            "C_ABI_ROUNDTRIP FAIL stage=length input_len={d} decoded_len={d}\n",
            .{ input.len, decoded_bytes },
        );
        return error.LengthMismatch;
    }
    if (!std.mem.eql(u8, input, decompressed[0..decoded_bytes])) {
        // Find first diff to aid debugging.
        var first: usize = 0;
        var i: usize = 0;
        while (i < input.len) : (i += 1) {
            if (input[i] != decompressed[i]) {
                first = i;
                break;
            }
        }
        try w.print(
            "C_ABI_ROUNDTRIP FAIL stage=compare first_diff={d}\n",
            .{first},
        );
        return error.BytesDiffer;
    }

    try w.print(
        "C_ABI_ROUNDTRIP PASS bytes={d} slz_bytes={d}\n",
        .{ input.len, slz_bytes },
    );
}
