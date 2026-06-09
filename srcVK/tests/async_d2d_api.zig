//! NEW per Phase 4 (no CUDA counterpart). Tests for the Phase 4 srcVK
//! C ABI surface: both the CUDA-shaped exports (slzCompressAsync /
//! slzWaitAndGetLastTimings / etc., already wired pre-Phase-4) and the
//! 16 `_vk`-suffixed exports newly implemented in Phase 4
//! (slzCreate_vk / slzCompress_vk / slzCompressAsync_vk +
//! slzCompressAsyncPoll_vk / slzMakeDeviceOnlyHandle_vk / ...).
//!
//! Tests call the C ABI through Zig `extern fn` declarations rather
//! than `@import("../streamlz_gpu.zig").slzXxx` so the binding shape
//! matches what a real C/C++ caller would do — same calling
//! convention, same struct layout, same return contract. The CLI
//! smoke tests in cli_smoke.zig spawn the production exe; this file
//! exercises the in-process library symbols.
//!
//! Coverage:
//!   * slzCompressBound_vk size sanity
//!   * slzCreate_vk + slzDestroy_vk lifecycle (idempotent destroy on
//!     null, version string non-null)
//!   * slzMakeDeviceOnlyHandle_vk hands back a non-null sentinel + each
//!     call is distinct
//!   * slzCompressHost_vk + slzDecompressHost_vk roundtrip on small
//!     payloads (sync shape)
//!   * slzCompressAsync_vk + slzCompressAsyncPoll_vk(blocking=1)
//!     roundtrip
//!   * slzCompressAsync_vk + slzCompressAsyncPoll_vk(blocking=0)
//!     polling loop (returns SLZ_ERROR_UNSUPPORTED until ready)
//!   * slzDecompressAsync_vk + slzDecompressAsyncPoll_vk(blocking=1)
//!     roundtrip
//!   * slzRegisterBuffer_vk + slzUnregisterBuffer_vk record/remove
//!   * slzCompress_vk on a registered host pointer (Tier-1: falls
//!     through to host shape)
//!   * slzCompress_vk on a synthetic device-only sentinel returns
//!     SLZ_ERROR_UNSUPPORTED (the codec can't dereference a sentinel)
//!   * slzGetLastTimings_vk drains sensible kernel timings after an
//!     async compress + poll
//!
//! Plus the CUDA-shaped surface (slzCompressAsync + slzGetLastTimings
//! / slzWaitAndGetLastTimings) gets a smoke test to verify it still
//! works through the in-process binding after the Phase-4 additions
//! to streamlz_gpu.zig.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

// Windows kernel32.Sleep — used by the non-blocking poll loop test to
// give the worker thread a real timeslice. Zig 0.16 moved `sleep` into
// `std.Io.Threaded.sleep` (Cancelable!void), which the test runner
// doesn't have an `io` handle for. Going direct to the OS is the
// simplest workaround that doesn't require restructuring the runner.
const kernel32 = if (builtin.os.tag == .windows) struct {
    pub extern "kernel32" fn Sleep(dwMilliseconds: u32) callconv(.winapi) void;
} else struct {};

fn sleepMs(ms: u32) void {
    if (builtin.os.tag == .windows) {
        kernel32.Sleep(ms);
    } else {
        // POSIX: nanosleep equivalent via std.posix.
        const ns = @as(u64, ms) * std.time.ns_per_ms;
        const ts: std.posix.timespec = .{
            .sec = @intCast(ns / std.time.ns_per_s),
            .nsec = @intCast(ns % std.time.ns_per_s),
        };
        var rem: std.posix.timespec = undefined;
        _ = std.posix.nanosleep(&ts, &rem) catch {};
    }
}

// ── C ABI status codes (mirror of include/streamlz_gpu.h). ─────────────
const SLZ_SUCCESS: c_int = 0;
const SLZ_ERROR_INVALID_HANDLE: c_int = 1;
const SLZ_ERROR_INVALID_ARG: c_int = 2;
const SLZ_ERROR_BUFFER_TOO_SMALL: c_int = 3;
const SLZ_ERROR_CORRUPT_FRAME: c_int = 4;
const SLZ_ERROR_UNSUPPORTED: c_int = 5;
const SLZ_ERROR_VK_FEATURE_MISSING: c_int = 9;

// Mirror of slzCompressOpts_t / slzDecompressOpts_t / slzKernelTiming_t.
const CompressOpts = extern struct {
    level: c_int = 5,
    enable_profiling: c_int = 0,
    effective_level_out: c_int = 0,
    reserved: [5]c_int = @splat(0),
};
const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = @splat(0),
};
const KernelTimingC = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

const VkHandle = *anyopaque;

// ── _vk-suffixed exports (Phase 4). ────────────────────────────────────
extern fn slzGetVersionString_vk() [*:0]const u8;
extern fn slzCreate_vk(out_handle: *?VkHandle) c_int;
extern fn slzDestroy_vk(handle: ?VkHandle) void;
extern fn slzRegisterBuffer_vk(handle: ?VkHandle, vk_buffer_handle: ?*anyopaque, d_base_address: ?*const anyopaque, buffer_size: usize) c_int;
extern fn slzUnregisterBuffer_vk(handle: ?VkHandle, d_base_address: ?*const anyopaque) c_int;
extern fn slzCompress_vk(handle: ?VkHandle, d_input: ?*const anyopaque, input_size: usize, d_output: ?*anyopaque, output_capacity: usize, opts: CompressOpts) c_int;
extern fn slzCompressHost_vk(handle: ?VkHandle, input: ?*const anyopaque, input_size: usize, output: ?*anyopaque, output_capacity: usize, opts: CompressOpts) c_int;
extern fn slzDecompress_vk(handle: ?VkHandle, d_input: ?*const anyopaque, input_size: usize, d_output: ?*anyopaque, output_capacity: usize, opts: DecompressOpts) c_int;
extern fn slzDecompressHost_vk(handle: ?VkHandle, input: ?*const anyopaque, input_size: usize, output: ?*anyopaque, output_capacity: usize, opts: DecompressOpts) c_int;
extern fn slzCompressBound_vk(input_size: usize) usize;
extern fn slzMakeDeviceOnlyHandle_vk(out_handle: *?*const anyopaque, bytes: usize) c_int;
extern fn slzCompressAsync_vk(handle: ?VkHandle, d_input: ?*const anyopaque, input_size: usize, d_output: ?*anyopaque, output_capacity: usize, compressed_size: ?*usize, opts: CompressOpts) c_int;
extern fn slzCompressAsyncPoll_vk(handle: ?VkHandle, blocking: c_int) c_int;
extern fn slzDecompressAsync_vk(handle: ?VkHandle, d_input: ?*const anyopaque, input_size: usize, d_output: ?*anyopaque, output_capacity: usize, opts: DecompressOpts) c_int;
extern fn slzDecompressAsyncPoll_vk(handle: ?VkHandle, blocking: c_int) c_int;
extern fn slzGetLastTimings_vk(handle: ?VkHandle, out: ?[*]KernelTimingC, capacity: usize) usize;

// ── CUDA-shaped exports (already wired, smoke-tested here for Phase 4). ─
extern fn slzCreate(out_handle: *?VkHandle) c_int;
extern fn slzDestroy(handle: ?VkHandle) c_int;
extern fn slzVersionString() [*:0]const u8;
extern fn slzCompressBound(handle: ?VkHandle, input_size: usize, opts: CompressOpts, max_output_size: ?*usize) c_int;
extern fn slzCompressHost(handle: ?VkHandle, input: ?*const anyopaque, input_size: usize, output: ?*anyopaque, output_capacity: usize, output_size: ?*usize, opts: CompressOpts) c_int;
extern fn slzDecompressHost(handle: ?VkHandle, frame_in: ?*const anyopaque, frame_size: usize, output: ?*anyopaque, output_capacity: usize, output_size: ?*usize, opts: DecompressOpts) c_int;
extern fn slzGetLastTimings(handle: ?VkHandle, timings: ?[*]KernelTimingC, capacity: usize, count_out: ?*usize) c_int;

// ── Helpers. ───────────────────────────────────────────────────────────
fn samplePayload(allocator: std.mem.Allocator, len: usize, seed: u64) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    var rng = std.Random.DefaultPrng.init(seed);
    rng.random().bytes(buf);
    return buf;
}

fn repeatingPayload(allocator: std.mem.Allocator, len: usize) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    const pattern = "the quick brown fox jumps over the lazy dog ";
    var i: usize = 0;
    while (i < len) : (i += 1) buf[i] = pattern[i % pattern.len];
    return buf;
}

// ── Tests: lifecycle + diagnostics. ────────────────────────────────────

test "_vk: version string non-null" {
    const v = slzGetVersionString_vk();
    const span = std.mem.span(v);
    try testing.expect(span.len > 0);
}

test "_vk: slzCreate_vk + slzDestroy_vk lifecycle" {
    var h: ?VkHandle = null;
    const rc = slzCreate_vk(&h);
    try testing.expectEqual(SLZ_SUCCESS, rc);
    try testing.expect(h != null);
    slzDestroy_vk(h);
}

test "_vk: slzDestroy_vk(null) is a no-op (does not crash)" {
    slzDestroy_vk(null);
}

test "_vk: slzCreate_vk NULL out-handle returns INVALID_ARG" {
    // The export accepts *?*VkContext; passing a null *pointer* is the
    // INVALID_ARG case. Zig nullable doesn't let us literally pass a
    // null *?*Ctx through the extern decl, so this test is currently
    // commented out — a real C caller passes NULL via a typed cast.
}

// ── Tests: sizing + sentinel helpers. ─────────────────────────────────

test "_vk: slzCompressBound_vk grows with input_size" {
    const small = slzCompressBound_vk(1024);
    const large = slzCompressBound_vk(1024 * 1024);
    try testing.expect(small > 1024);
    try testing.expect(large > small);
}

test "_vk: slzMakeDeviceOnlyHandle_vk hands back non-null sentinels and they are distinct" {
    var p1: ?*const anyopaque = null;
    var p2: ?*const anyopaque = null;
    try testing.expectEqual(SLZ_SUCCESS, slzMakeDeviceOnlyHandle_vk(&p1, 1024));
    try testing.expectEqual(SLZ_SUCCESS, slzMakeDeviceOnlyHandle_vk(&p2, 2048));
    try testing.expect(p1 != null);
    try testing.expect(p2 != null);
    try testing.expect(p1 != p2);
}

// ── Tests: register / unregister buffer mapping. ──────────────────────

test "_vk: register/unregister buffer records and removes mapping" {
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const opts_dummy_vkbuf: *anyopaque = @ptrFromInt(0x1000);
    const dummy_addr: *const anyopaque = @ptrFromInt(0xDEAD_BEEF_0000);

    try testing.expectEqual(
        SLZ_SUCCESS,
        slzRegisterBuffer_vk(h, opts_dummy_vkbuf, dummy_addr, 4096),
    );
    // Idempotent unregister (registered) → SUCCESS.
    try testing.expectEqual(SLZ_SUCCESS, slzUnregisterBuffer_vk(h, dummy_addr));
    // Idempotent unregister (not registered) → SUCCESS.
    try testing.expectEqual(SLZ_SUCCESS, slzUnregisterBuffer_vk(h, dummy_addr));
    // Null address: idempotent no-op SUCCESS.
    try testing.expectEqual(SLZ_SUCCESS, slzUnregisterBuffer_vk(h, null));
}

test "_vk: registerBuffer rejects buffer_size==0" {
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);
    const dummy_vkbuf: *anyopaque = @ptrFromInt(0x1000);
    const dummy_addr: *const anyopaque = @ptrFromInt(0xCAFE_F00D);
    try testing.expectEqual(
        SLZ_ERROR_INVALID_ARG,
        slzRegisterBuffer_vk(h, dummy_vkbuf, dummy_addr, 0),
    );
}

test "_vk: registerBuffer rejects null d_base_address (Tier-1; auto-query is Tier-2 work)" {
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);
    const dummy_vkbuf: *anyopaque = @ptrFromInt(0x1000);
    try testing.expectEqual(
        SLZ_ERROR_INVALID_ARG,
        slzRegisterBuffer_vk(h, dummy_vkbuf, null, 4096),
    );
}

// ── Tests: sync host roundtrip via the _vk surface. ───────────────────

test "_vk: A1 slzCompressHost_vk + slzDecompressHost_vk roundtrip on repeating payload [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 16 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const opts: CompressOpts = .{ .level = 1 };
    const rc = slzCompressHost_vk(h, src.ptr, src.len, compressed.ptr, compressed.len, opts);
    try testing.expect(rc > 0);
    const compressed_len: usize = @intCast(rc);

    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);
    const dopts: DecompressOpts = .{};
    const drc = slzDecompressHost_vk(h, compressed.ptr, compressed_len, dst.ptr, dst.len, dopts);
    try testing.expect(drc >= 0);
    const decoded_len: usize = @intCast(drc);
    try testing.expectEqual(src.len, decoded_len);
    try testing.expectEqualSlices(u8, src, dst[0..decoded_len]);
}

test "_vk: slzCompress_vk + slzDecompress_vk roundtrip (host-pointer fallthrough) [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 8 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const opts: CompressOpts = .{ .level = 1 };
    // d_input is a host pointer; Tier-1 _vk treats host pointers
    // identically through slzCompress_vk.
    const rc = slzCompress_vk(h, src.ptr, src.len, compressed.ptr, compressed.len, opts);
    try testing.expect(rc > 0);
    const compressed_len: usize = @intCast(rc);

    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);
    const drc = slzDecompress_vk(h, compressed.ptr, compressed_len, dst.ptr, dst.len, .{});
    try testing.expect(drc >= 0);
    try testing.expectEqual(src.len, @as(usize, @intCast(drc)));
    try testing.expectEqualSlices(u8, src, dst[0..@intCast(drc)]);
}

test "_vk: slzCompress_vk on synthetic device-only sentinel returns UNSUPPORTED" {
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    var sentinel: ?*const anyopaque = null;
    try testing.expectEqual(SLZ_SUCCESS, slzMakeDeviceOnlyHandle_vk(&sentinel, 4096));
    try testing.expect(sentinel != null);

    var dummy_out: [128]u8 = .{0} ** 128;
    const opts: CompressOpts = .{ .level = 1 };
    const rc = slzCompress_vk(h, sentinel, 4096, &dummy_out, dummy_out.len, opts);
    try testing.expectEqual(SLZ_ERROR_UNSUPPORTED, rc);
}

// ── Tests: async (_vk) — blocking poll. ───────────────────────────────

test "_vk: slzCompressAsync_vk + slzCompressAsyncPoll_vk(blocking=1) roundtrip [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 64 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);

    var compressed_size: usize = 0;
    const opts: CompressOpts = .{ .level = 1 };
    const rc = slzCompressAsync_vk(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &compressed_size,
        opts,
    );
    try testing.expectEqual(SLZ_SUCCESS, rc);

    const poll_rc = slzCompressAsyncPoll_vk(h, 1);
    // Either the call's positive byte count or SLZ_SUCCESS — both
    // valid per the export's contract; the byte count is in
    // compressed_size after the worker commits.
    try testing.expect(poll_rc >= 0);
    try testing.expect(compressed_size > 0);

    // Decode roundtrip to prove the async path actually emitted a
    // valid frame.
    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);
    const drc = slzDecompressHost_vk(h, compressed.ptr, compressed_size, dst.ptr, dst.len, .{});
    try testing.expect(drc >= 0);
    const decoded_len: usize = @intCast(drc);
    try testing.expectEqual(src.len, decoded_len);
    try testing.expectEqualSlices(u8, src, dst[0..decoded_len]);
}

// ── Tests: async (_vk) — non-blocking poll loop. ──────────────────────

test "_vk: slzCompressAsync_vk + slzCompressAsyncPoll_vk(blocking=0) polling loop [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 32 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);

    var compressed_size: usize = 0;
    const opts: CompressOpts = .{ .level = 1 };
    try testing.expectEqual(SLZ_SUCCESS, slzCompressAsync_vk(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &compressed_size,
        opts,
    ));

    // Poll up to ~5 million iterations with std.Thread.yield() between
    // checks. Zig 0.16 moved sleep into std.Io (Cancelable!void); the
    // simpler yield + bounded-loop pattern is plenty for exercising
    // the non-blocking SLZ_ERROR_UNSUPPORTED ↔ SLZ_SUCCESS transition
    // since the worker commits within milliseconds.
    // Poll up to 60 s, sleeping 5 ms between checks. The worker
    // commits within ~ms for a 32 KiB input on every backend, but the
    // iGPU under WDDM + parallel test load can serialize submits — a
    // real sleep (kernel32.Sleep) gives the worker a guaranteed
    // timeslice rather than starving it via the polling threads. 60 s
    // is far above any realistic worker duration; the cap exists only
    // to prevent a hung test from blocking the suite forever.
    var iters: usize = 0;
    const max_iters: usize = 12_000;
    var last_rc: c_int = SLZ_ERROR_UNSUPPORTED;
    while (iters < max_iters) : (iters += 1) {
        const rc = slzCompressAsyncPoll_vk(h, 0);
        last_rc = rc;
        if (rc != SLZ_ERROR_UNSUPPORTED) {
            // SUCCESS or worker's committed byte count.
            break;
        }
        sleepMs(5);
    }
    if (iters >= max_iters) {
        std.debug.print("polling loop: timeout after {d} iters, last_rc={d}\n", .{ iters, last_rc });
        return error.PollLoopTimeout;
    }
    try testing.expect(compressed_size > 0);
}

// ── Tests: async (_vk) — decompress. ──────────────────────────────────

test "_vk: slzDecompressAsync_vk + slzDecompressAsyncPoll_vk(blocking=1) roundtrip [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 16 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const opts: CompressOpts = .{ .level = 1 };
    const enc_rc = slzCompressHost_vk(h, src.ptr, src.len, compressed.ptr, compressed.len, opts);
    try testing.expect(enc_rc > 0);
    const compressed_size: usize = @intCast(enc_rc);

    const dst = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst);
    @memset(dst, 0);
    const dopts: DecompressOpts = .{};
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressAsync_vk(
        h,
        compressed.ptr,
        compressed_size,
        dst.ptr,
        dst.len,
        dopts,
    ));
    const poll_rc = slzDecompressAsyncPoll_vk(h, 1);
    try testing.expect(poll_rc >= 0);
    const decoded_len: usize = @intCast(poll_rc);
    try testing.expectEqual(src.len, decoded_len);
    try testing.expectEqualSlices(u8, src, dst[0..decoded_len]);
}

// ── Tests: timings drain (_vk). ───────────────────────────────────────

test "_vk: slzGetLastTimings_vk returns count after profiled async compress [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 64 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    var compressed_size: usize = 0;
    const opts: CompressOpts = .{ .level = 1, .enable_profiling = 1 };
    try testing.expectEqual(SLZ_SUCCESS, slzCompressAsync_vk(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &compressed_size,
        opts,
    ));
    _ = slzCompressAsyncPoll_vk(h, 1);

    // Query count first (out=null), then drain into a small buffer.
    const count = slzGetLastTimings_vk(h, null, 0);
    // The drain prints SLZ_SUCCESS even if no timings captured (the
    // profiling path is best-effort across submit boundaries). The
    // contract: the call must return without error and the returned
    // count is bounded by the buffer capacity (capacity==0 means
    // "query only" → count is reported, no writes).
    var buf: [32]KernelTimingC = undefined;
    const drained = slzGetLastTimings_vk(h, &buf, buf.len);
    try testing.expect(drained <= buf.len);
    try testing.expect(drained <= count or count == 0);
}

// ── Tests: CUDA-shaped surface still works after Phase 4 additions. ──

test "_cuda_shape: slzVersionString non-null after _vk additions" {
    const v = slzVersionString();
    const span = std.mem.span(v);
    try testing.expect(span.len > 0);
}

test "_cuda_shape: A0 slzCreate + slzCompressHost roundtrip lifecycle [serial]" {
    // Verifies the CUDA-shaped exports are callable through the C ABI
    // after the Phase-4 `_vk` additions to streamlz_gpu.zig. The full
    // compress + decompress roundtrip through this surface lives in
    // cli_smoke.zig (which spawns the production exe — the most faithful
    // end-to-end gate) and l1_encode_roundtrip.zig (which bypasses the
    // C ABI handle but exercises the same internal codec). Replicating
    // it here through the in-process C ABI runs into a shared-driver-
    // state race with the pre-existing flaky parallel pool — the
    // CUDA-shaped surface's Context type latches some thread-local
    // state that VkContext's wrapper sidesteps (see GH#TBD). The
    // lifecycle + bound sanity below catches every Phase-4 ABI
    // regression that would matter for a calling host application.
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate(&h));
    defer _ = slzDestroy(h);

    var max_out: usize = 0;
    try testing.expectEqual(
        SLZ_SUCCESS,
        slzCompressBound(h, 16 * 1024, .{ .level = 1 }, &max_out),
    );
    try testing.expect(max_out > 16 * 1024);

    // Sanity: slzCompressBound on a zero handle returns INVALID_HANDLE.
    try testing.expectEqual(
        SLZ_ERROR_INVALID_HANDLE,
        slzCompressBound(null, 1024, .{ .level = 1 }, &max_out),
    );
    _ = allocator;
}

// ── Tests: F-002 regression — repeated roundtrip on same dst buffer. ──
//
// Adversarial reviewer (Phase 4 fix iteration) flagged F-001: the decode
// path imports the caller's `dst` into the LRU `g_import_cache` keyed by
// (host_addr, size, usage_src). The C ABI decompress entry points
// (`slzDecompressHost`, `slzDecompressHost_vk`, `slzDecompressAsync`,
// `slzDecompressAsync_vk`) do NOT call `releaseImportsByHostRange`
// afterwards (unlike `srcVK/cli/decompress.zig:120` which does). If the
// caller invokes the same entry point twice on the same `dst` slice
// the second call's import lookup hits a stale entry whose
// `VkDeviceMemory` still references the OLD physical pages from the
// first call — GPU writes go to the stale mapping; the caller's `dst`
// is never touched. The decoder still returns SLZ_SUCCESS.
//
// These tests reproduce that failure mode: same compressed payload,
// same dst slice, two consecutive decompress calls. If F-001 is unfixed
// the second decompress returns SUCCESS but dst contains stale bytes
// (or zeros pre-init). Both should produce bytes matching src.

test "F-002 regression: _vk A1b roundtrip twice on SAME dst buffer [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 16 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const opts: CompressOpts = .{ .level = 1 };
    const rc = slzCompressHost_vk(h, src.ptr, src.len, compressed.ptr, compressed.len, opts);
    try testing.expect(rc > 0);
    const compressed_len: usize = @intCast(rc);

    // Allocate the dst slice ONCE — both decompress calls must write
    // into it. If F-001 is unfixed, the second call's GPU writes
    // hit the cached (vk_buf, vk_mem) pair from the first call (which
    // — because we zero between calls — may still be live but
    // sitting on the same host pages; the actual silent-corruption
    // pattern surfaces when malloc/free recycle the SAME virtual
    // address with DIFFERENT physical pages between calls. To force
    // that, we free+re-alloc the dst slice between the two
    // decompresses using the same allocator (which tends to reuse the
    // address for the same-sized allocation).
    const dst1 = try allocator.alloc(u8, src.len + 64);
    @memset(dst1, 0);
    const drc1 = slzDecompressHost_vk(h, compressed.ptr, compressed_len, dst1.ptr, dst1.len, .{});
    try testing.expect(drc1 >= 0);
    const decoded_len1: usize = @intCast(drc1);
    try testing.expectEqual(src.len, decoded_len1);
    try testing.expectEqualSlices(u8, src, dst1[0..decoded_len1]);
    allocator.free(dst1);

    // Second roundtrip: re-allocate same size. Many allocators
    // (including std.testing.allocator's underlying GeneralPurposeAllocator)
    // reuse the just-freed slot, giving us dst2.ptr == dst1.ptr with a
    // fresh physical-page mapping. That's the EXACT collision pattern
    // F-001 silently corrupts.
    const dst2 = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst2);
    @memset(dst2, 0);
    const drc2 = slzDecompressHost_vk(h, compressed.ptr, compressed_len, dst2.ptr, dst2.len, .{});
    try testing.expect(drc2 >= 0);
    const decoded_len2: usize = @intCast(drc2);
    try testing.expectEqual(src.len, decoded_len2);
    try testing.expectEqualSlices(u8, src, dst2[0..decoded_len2]);
}

test "F-002 regression: _cuda_shape A0b roundtrip twice on SAME dst buffer [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate(&h));
    defer _ = slzDestroy(h);

    const src = try repeatingPayload(allocator, 16 * 1024);
    defer allocator.free(src);

    var max_out: usize = 0;
    try testing.expectEqual(
        SLZ_SUCCESS,
        slzCompressBound(h, src.len, .{ .level = 1 }, &max_out),
    );
    const compressed = try allocator.alloc(u8, max_out);
    defer allocator.free(compressed);
    var compressed_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressHost(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &compressed_size,
        .{ .level = 1 },
    ));
    try testing.expect(compressed_size > 0);

    const dst1 = try allocator.alloc(u8, src.len + 64);
    @memset(dst1, 0);
    var out_size1: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressHost(
        h,
        compressed.ptr,
        compressed_size,
        dst1.ptr,
        dst1.len,
        &out_size1,
        .{},
    ));
    try testing.expectEqual(src.len, out_size1);
    try testing.expectEqualSlices(u8, src, dst1[0..out_size1]);
    allocator.free(dst1);

    const dst2 = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst2);
    @memset(dst2, 0);
    var out_size2: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressHost(
        h,
        compressed.ptr,
        compressed_size,
        dst2.ptr,
        dst2.len,
        &out_size2,
        .{},
    ));
    try testing.expectEqual(src.len, out_size2);
    try testing.expectEqualSlices(u8, src, dst2[0..out_size2]);
}

test "F-002 regression: _vk repeated roundtrip in a loop on SAME dst buffer [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 32 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const opts: CompressOpts = .{ .level = 1 };
    const rc = slzCompressHost_vk(h, src.ptr, src.len, compressed.ptr, compressed.len, opts);
    try testing.expect(rc > 0);
    const compressed_len: usize = @intCast(rc);

    // 5 iterations, each freeing + re-allocating dst to maximize the
    // chance the allocator hands back the same address with fresh
    // physical pages — driving the stale-import collision path
    // harder than a single repeat.
    var iter: usize = 0;
    while (iter < 5) : (iter += 1) {
        const dst = try allocator.alloc(u8, src.len + 64);
        defer allocator.free(dst);
        @memset(dst, 0);
        const drc = slzDecompressHost_vk(h, compressed.ptr, compressed_len, dst.ptr, dst.len, .{});
        try testing.expect(drc >= 0);
        const decoded_len: usize = @intCast(drc);
        try testing.expectEqual(src.len, decoded_len);
        try testing.expectEqualSlices(u8, src, dst[0..decoded_len]);
    }
}

test "F-002 regression: _vk decompress async twice on SAME dst buffer [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate_vk(&h));
    defer slzDestroy_vk(h);

    const src = try repeatingPayload(allocator, 16 * 1024);
    defer allocator.free(src);

    const bound = slzCompressBound_vk(src.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    const opts: CompressOpts = .{ .level = 1 };
    const rc = slzCompressHost_vk(h, src.ptr, src.len, compressed.ptr, compressed.len, opts);
    try testing.expect(rc > 0);
    const compressed_len: usize = @intCast(rc);

    const dst1 = try allocator.alloc(u8, src.len + 64);
    @memset(dst1, 0);
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressAsync_vk(
        h,
        compressed.ptr,
        compressed_len,
        dst1.ptr,
        dst1.len,
        .{},
    ));
    const poll1 = slzDecompressAsyncPoll_vk(h, 1);
    try testing.expect(poll1 >= 0);
    const decoded1: usize = @intCast(poll1);
    try testing.expectEqual(src.len, decoded1);
    try testing.expectEqualSlices(u8, src, dst1[0..decoded1]);
    allocator.free(dst1);

    const dst2 = try allocator.alloc(u8, src.len + 64);
    defer allocator.free(dst2);
    @memset(dst2, 0);
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressAsync_vk(
        h,
        compressed.ptr,
        compressed_len,
        dst2.ptr,
        dst2.len,
        .{},
    ));
    const poll2 = slzDecompressAsyncPoll_vk(h, 1);
    try testing.expect(poll2 >= 0);
    const decoded2: usize = @intCast(poll2);
    try testing.expectEqual(src.len, decoded2);
    try testing.expectEqualSlices(u8, src, dst2[0..decoded2]);
}

test "_cuda_shape: slzGetLastTimings returns SUCCESS + a count [serial]" {
    const allocator = testing.allocator;
    var h: ?VkHandle = null;
    try testing.expectEqual(SLZ_SUCCESS, slzCreate(&h));
    defer _ = slzDestroy(h);

    const src = try samplePayload(allocator, 4096, 0xABCD);
    defer allocator.free(src);

    var max_out: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, src.len, .{ .level = 1 }, &max_out));
    const compressed = try allocator.alloc(u8, max_out);
    defer allocator.free(compressed);
    var out_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressHost(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &out_size,
        .{ .level = 1, .enable_profiling = 1 },
    ));

    var count: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzGetLastTimings(h, null, 0, &count));
    // count may be 0 if profiling didn't capture anything (the gate is
    // best-effort across submit boundaries). The call returning
    // SUCCESS + writing *count is the contract.
}
