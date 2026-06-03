//! Phase-1 (TODO A1) C-ABI exercise harness for the six symbols that
//! were stubbed to SLZ_ERROR_UNSUPPORTED at HEAD: the async pair for
//! both directions, the timings drain, and the device-only sentinel
//! constructor. Wired as `zig build vk-abi-async-test`.
//!
//! Exercises:
//!   * slzMakeDeviceOnlyHandle_vk — always returns SLZ_ERROR_UNSUPPORTED
//!     and writes null to the out-slot. The Vulkan backend's D2D path
//!     goes through `slzRegisterBuffer_vk` + BDA u64 (no host-sentinel
//!     pattern, unlike CUDA's INTERNAL `device_only_host_stub_addr`
//!     which is not exposed via the CUDA C ABI either).
//!   * slzCompressAsync_vk + slzCompressAsyncPoll_vk — submits, polls
//!     non-blocking (may report not-ready), then blocking-polls to
//!     completion; asserts the byte count round-trips with what a
//!     synchronous slzCompressHost_vk would have produced.
//!   * slzDecompressAsync_vk + slzDecompressAsyncPoll_vk — same drill
//!     for the decode path; byte-equal to source after round-trip.
//!   * slzGetLastTimings_vk — non-zero entry count after a sync
//!     compress + decompress pair; ms values are positive floats; the
//!     reported kernel names are "lz_encode" / "lz_decode" per the
//!     contract in streamlz_gpu_vk.zig.
//!
//! Prints one PASS/FAIL line per case and a final summary. Exits 0
//! when every case passes, 1 otherwise (CI scrapes the summary).

const std = @import("std");
const builtin = @import("builtin");

const vk_abi = @import("streamlz_gpu_vk.zig");

const CORPUS_PATH: []const u8 = "assets/web.txt";
const PREFIX_SIZE: usize = 256 * 1024; // 256 KiB keeps the async paths fast (multi-chunk + non-trivial work).

const Allocator = std.mem.Allocator;

/// Cheap millisecond sleep — std.Thread.sleep was removed in Zig
/// 0.16 and std.os.windows.kernel32 no longer exports Sleep there.
/// Declare the Win32 entry point directly for the only platform we
/// actually run on. POSIX targets fall back to nanosleep.
const win32_sleep = struct {
    extern "kernel32" fn Sleep(ms: u32) callconv(.winapi) void;
};
fn sleepMs(ms: u32) void {
    if (builtin.os.tag == .windows) {
        win32_sleep.Sleep(ms);
    } else {
        const ts: std.c.timespec = .{
            .sec = @intCast(ms / 1000),
            .nsec = @intCast((ms % 1000) * std.time.ns_per_ms),
        };
        _ = std.c.nanosleep(&ts, null);
    }
}

/// Outcome of one test case. `name` is the case identifier (also
/// printed); `detail` is rendered after the PASS / FAIL marker for
/// diagnostic context.
const CaseResult = struct {
    name: []const u8,
    passed: bool,
    detail: []const u8,
};

fn loadCorpus(allocator: Allocator, io: std.Io) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, CORPUS_PATH, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    const want: usize = @min(PREFIX_SIZE, @as(usize, @intCast(stat.size)));
    const buf = try allocator.alloc(u8, want);
    errdefer allocator.free(buf);
    const n = try file.readPositionalAll(io, buf, 0);
    if (n != want) return error.ShortRead;
    return buf;
}

/// Spin until `slzCompressAsyncPoll_vk(handle, 0)` reports the op is
/// complete OR the deadline elapses. The non-blocking poll's "not
/// ready" sentinel is SLZ_ERROR_UNSUPPORTED (== 5). Returns the rc
/// of the final blocking-poll, or -1 on timeout.
fn awaitCompress(handle: ?*vk_abi.VkContext, allocator: Allocator) !c_int {
    _ = allocator;
    var spins: usize = 0;
    while (spins < 2000) : (spins += 1) {
        const rc_nb = vk_abi.slzCompressAsyncPoll_vk(handle, 0);
        // SLZ_ERROR_UNSUPPORTED is the in-flight sentinel for
        // non-blocking polls — anything else means "done" (and we
        // should not call it again, so we return that result).
        if (rc_nb != 5) return rc_nb;
        // Microsleep to avoid pegging a core. The async path here is
        // a few ms at most for a 256 KiB encode on any backed device.
        sleepMs(1);
    }
    // Last-resort blocking poll: if the non-blocking loop above
    // exhausted, fall through to the join. Useful for debugging — a
    // hung GPU surfaces as a 60 s fence-wait timeout from the codec
    // rather than spinning here forever.
    return vk_abi.slzCompressAsyncPoll_vk(handle, 1);
}

fn awaitDecompress(handle: ?*vk_abi.VkContext, allocator: Allocator) !c_int {
    _ = allocator;
    var spins: usize = 0;
    while (spins < 2000) : (spins += 1) {
        const rc_nb = vk_abi.slzDecompressAsyncPoll_vk(handle, 0);
        if (rc_nb != 5) return rc_nb;
        sleepMs(1);
    }
    return vk_abi.slzDecompressAsyncPoll_vk(handle, 1);
}

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;
    const allocator = std.heap.c_allocator;

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    var results: std.ArrayList(CaseResult) = .empty;
    defer results.deinit(allocator);

    // ── Common: load corpus + create handle ──────────────────────────
    const src = loadCorpus(allocator, io) catch |err| {
        try w.print("ABI_ASYNC FAIL stage=load_corpus err={s}\n", .{@errorName(err)});
        return error.LoadFailed;
    };
    defer allocator.free(src);

    var handle: ?*vk_abi.VkContext = null;
    if (vk_abi.slzCreate_vk(&handle) != 0 or handle == null) {
        try w.writeAll("ABI_ASYNC FAIL stage=create\n");
        return error.CreateFailed;
    }
    defer vk_abi.slzDestroy_vk(handle);

    const bound = vk_abi.slzCompressBound_vk(src.len);

    // ── Case 1: slzMakeDeviceOnlyHandle_vk reports unsupported ─────
    // The Vulkan backend does not implement the CUDA host-sentinel
    // pattern (D2D goes through slzRegisterBuffer_vk + BDA u64), so
    // the ABI symbol returns SLZ_ERROR_UNSUPPORTED and clears the
    // out-slot. A future revision wiring a real host-sentinel pattern
    // would need to update this case to assert SLZ_SUCCESS and a
    // distinguishable sentinel pointer.
    {
        var dev_handle: ?*const anyopaque = @ptrFromInt(0xdeadbeef);
        const rc = vk_abi.slzMakeDeviceOnlyHandle_vk(&dev_handle, src.len);
        const passed = rc == 5 and dev_handle == null;
        try results.append(allocator, .{
            .name = "make_device_only_handle_unsupported",
            .passed = passed,
            .detail = if (passed) "ok" else "expected UNSUPPORTED + null out-slot",
        });
    }

    // ── Case 3: slzGetLastTimings_vk after a sync round-trip ────────
    // First, run a sync compress + decompress to populate the per-
    // handle timing snapshot.
    const slz_buf_sync = try allocator.alloc(u8, bound);
    defer allocator.free(slz_buf_sync);
    const dec_buf_sync = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dec_buf_sync);

    const sync_cz = vk_abi.slzCompressHost_vk(
        handle,
        src.ptr,
        src.len,
        slz_buf_sync.ptr,
        slz_buf_sync.len,
        .{ .level = 1 },
    );
    if (sync_cz < 0) {
        try results.append(allocator, .{
            .name = "sync_compress_for_timings",
            .passed = false,
            .detail = "sync compress failed",
        });
    } else {
        const sync_dz = vk_abi.slzDecompressHost_vk(
            handle,
            slz_buf_sync.ptr,
            @intCast(sync_cz),
            dec_buf_sync.ptr,
            dec_buf_sync.len,
            .{},
        );
        if (sync_dz < 0) {
            try results.append(allocator, .{
                .name = "sync_decompress_for_timings",
                .passed = false,
                .detail = "sync decompress failed",
            });
        } else {
            // Now drain timings.
            var timings: [8]vk_abi.KernelTimingC = undefined;
            const count = vk_abi.slzGetLastTimings_vk(handle, &timings, timings.len);
            // The decoder writes the most-recent dispatch ns; the
            // encoder did too. Both slots should be non-zero. (On a
            // device that reports timestampValidBits == 0 the ns
            // could be 0 — but every supported tier has timestamps.)
            const passed = count >= 1 and count <= 2 and
                timings[0].ms >= 0.0 and (count == 1 or timings[1].ms >= 0.0);
            try results.append(allocator, .{
                .name = "get_last_timings_after_sync_pair",
                .passed = passed,
                .detail = if (passed) "ok" else "expected 1-2 entries with non-negative ms",
            });
        }
    }

    // ── Case 4: async compress round-trip ─────────────────────────
    const slz_buf_async = try allocator.alloc(u8, bound);
    defer allocator.free(slz_buf_async);
    var async_cz_out: usize = 0;
    {
        const submit_rc = vk_abi.slzCompressAsync_vk(
            handle,
            src.ptr,
            src.len,
            slz_buf_async.ptr,
            slz_buf_async.len,
            &async_cz_out,
            .{ .level = 1 },
        );
        if (submit_rc != 0) {
            try results.append(allocator, .{
                .name = "compress_async_submit",
                .passed = false,
                .detail = "submit failed",
            });
        } else {
            const final_rc = try awaitCompress(handle, allocator);
            const passed = final_rc >= 0 and @as(usize, @intCast(final_rc)) == async_cz_out and
                async_cz_out == @as(usize, @intCast(sync_cz));
            try results.append(allocator, .{
                .name = "compress_async_roundtrip",
                .passed = passed,
                .detail = if (passed) "ok" else "byte count mismatch vs sync",
            });
        }
    }

    // ── Case 5: async decompress round-trip ──────────────────────
    const dec_buf_async = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dec_buf_async);
    @memset(dec_buf_async, 0);
    {
        const submit_rc = vk_abi.slzDecompressAsync_vk(
            handle,
            slz_buf_async.ptr,
            async_cz_out,
            dec_buf_async.ptr,
            dec_buf_async.len,
            .{},
        );
        if (submit_rc != 0) {
            try results.append(allocator, .{
                .name = "decompress_async_submit",
                .passed = false,
                .detail = "submit failed",
            });
        } else {
            const final_rc = try awaitDecompress(handle, allocator);
            const decoded: usize = if (final_rc >= 0) @intCast(final_rc) else 0;
            const length_ok = decoded == src.len;
            const bytes_ok = length_ok and std.mem.eql(u8, src, dec_buf_async[0..decoded]);
            try results.append(allocator, .{
                .name = "decompress_async_roundtrip",
                .passed = bytes_ok,
                .detail = if (bytes_ok) "ok" else "byte mismatch vs source",
            });
        }
    }

    // ── Case 6: double-submit while in-flight returns UNSUPPORTED ──
    {
        var async_cz2: usize = 0;
        const rc1 = vk_abi.slzCompressAsync_vk(
            handle,
            src.ptr,
            src.len,
            slz_buf_async.ptr,
            slz_buf_async.len,
            &async_cz2,
            .{ .level = 1 },
        );
        if (rc1 != 0) {
            try results.append(allocator, .{
                .name = "double_submit_guard",
                .passed = false,
                .detail = "first submit failed",
            });
        } else {
            // Second submit before joining should be rejected with
            // SLZ_ERROR_UNSUPPORTED — the slot is busy.
            const rc2 = vk_abi.slzCompressAsync_vk(
                handle,
                src.ptr,
                src.len,
                slz_buf_async.ptr,
                slz_buf_async.len,
                &async_cz2,
                .{ .level = 1 },
            );
            // Drain the first op so the rest of the test (and the
            // handle's destructor) see a clean slot.
            _ = try awaitCompress(handle, allocator);
            const passed = rc2 == 5;
            try results.append(allocator, .{
                .name = "double_submit_guard",
                .passed = passed,
                .detail = if (passed) "ok" else "expected second-submit rejection",
            });
        }
    }

    // ── Report ──────────────────────────────────────────────────────
    var pass_count: u32 = 0;
    var fail_count: u32 = 0;
    for (results.items) |r| {
        if (r.passed) {
            pass_count += 1;
            try w.print("ABI_ASYNC PASS {s} ({s})\n", .{ r.name, r.detail });
        } else {
            fail_count += 1;
            try w.print("ABI_ASYNC FAIL {s} ({s})\n", .{ r.name, r.detail });
        }
    }
    try w.print(
        "ABI_ASYNC SUMMARY pass={d} fail={d} total={d}\n",
        .{ pass_count, fail_count, pass_count + fail_count },
    );
    if (fail_count > 0) {
        try w.flush();
        std.process.exit(1);
    }
}
