//! C ABI tests for the streamlz_gpu library surface — backport of
//! `srcVK/tests/async_d2d_api.zig` (BACKPORTS.md D2, the last zero-
//! coverage area: the old ABI tests belonged to the deleted
//! src_vulkan tree).
//!
//! Tests call the C ABI through Zig `extern fn` declarations rather
//! than importing the implementation's Zig identifiers, so the binding
//! shape matches what a real C/C++ caller does — same calling
//! convention, same struct layout, same return contract. The export
//! definitions are linked into the test binary via the
//! `_ = @import("streamlz_gpu.zig")` line in main.zig's test block.
//!
//! Coverage (CUDA-shaped surface; the _vk-suffixed exports are VK-only):
//!   * slzVersionString / slzStatusString string contracts
//!   * slzCreate / slzDestroy lifecycle (+ null-arg validation)
//!   * slzCompressBound contract (grows with size, handle/arg checks)
//!   * default-opts ABI values
//!   * slzCompressHost + slzDecompressHost roundtrips (L1 + L5)
//!   * slzGetDecompressedSize on a real frame + corrupt-header reject
//!   * null-pointer argument validation on the hot entry points
//!   * repeated decompress into re-allocated same-size dst (the VK
//!     F-002 stale-import regression shape; CUDA has no import cache
//!     but the contract must hold regardless)
//!   * slzCompressAsync + slzDecompressAsync true-D2D roundtrip on
//!     device buffers (NULL stream) + tiny-input UNSUPPORTED gate
//!   * slzGetLastTimings / slzWaitAndGetLastTimings drain contract

const std = @import("std");
const testing = std.testing;

const cuda = @import("decode/cuda_api.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_roundtrip_tests = @import("encode/gpu_roundtrip_tests.zig");

// ── Status codes (mirror of include/streamlz_gpu.h) ────────────────
const SLZ_SUCCESS: c_int = 0;
const SLZ_ERROR_INVALID_HANDLE: c_int = 1;
const SLZ_ERROR_INVALID_ARG: c_int = 2;
const SLZ_ERROR_CORRUPT_FRAME: c_int = 4;
const SLZ_ERROR_UNSUPPORTED: c_int = 5;

// ── ABI structs (mirror of slzCompressOpts_t / slzDecompressOpts_t /
// slzKernelTiming_t; the implementation comptime-asserts 32-byte size
// for both opts structs) ───────────────────────────────────────────
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

const Handle = ?*anyopaque;

// ── Extern declarations (the C-caller view) ────────────────────────
extern fn slzVersionString() [*:0]const u8;
extern fn slzStatusString(status: c_int) [*:0]const u8;
extern fn slzCreate(out_handle: ?*Handle) c_int;
extern fn slzDestroy(handle: Handle) c_int;
extern fn slzCompressDefaultOpts() CompressOpts;
extern fn slzDecompressDefaultOpts() DecompressOpts;
extern fn slzCompressBound(handle: Handle, input_size: usize, opts: CompressOpts, max_output_size: ?*usize) c_int;
extern fn slzGetDecompressedSize(handle: Handle, host_bytes: ?*const anyopaque, decompressed_size: ?*usize) c_int;
extern fn slzCompressHost(handle: Handle, input: ?*const anyopaque, input_size: usize, output: ?*anyopaque, output_capacity: usize, output_size: ?*usize, opts: CompressOpts) c_int;
extern fn slzDecompressHost(handle: Handle, frame_in: ?*const anyopaque, frame_size: usize, output: ?*anyopaque, output_capacity: usize, output_size: ?*usize, opts: DecompressOpts) c_int;
extern fn slzCompressAsync(handle: Handle, d_input: ?*const anyopaque, input_size: usize, d_output: ?*anyopaque, max_compressed_size: usize, compressed_size: ?*usize, opts: CompressOpts, stream: ?*anyopaque) c_int;
extern fn slzDecompressAsync(handle: Handle, d_frame: ?*const anyopaque, frame_size: usize, d_output: ?*anyopaque, output_size: usize, opts: DecompressOpts, stream: ?*anyopaque) c_int;
extern fn slzGetLastTimings(handle: Handle, timings: ?[*]KernelTimingC, capacity: usize, count_out: ?*usize) c_int;
extern fn slzWaitAndGetLastTimings(handle: Handle, stream: ?*anyopaque, timings: ?[*]KernelTimingC, capacity: usize, count_out: ?*usize) c_int;

// ── Helpers ────────────────────────────────────────────────────────
fn repeatingPayload(allocator: std.mem.Allocator, len: usize) ![]u8 {
    const buf = try allocator.alloc(u8, len);
    const pattern = "the quick brown fox jumps over the lazy dog ";
    for (buf, 0..) |*b, i| b.* = pattern[i % pattern.len];
    return buf;
}

/// Create a handle or skip the test (no GPU). Caller must slzDestroy.
fn createOrSkip() !Handle {
    var h: Handle = null;
    const rc = slzCreate(&h);
    if (rc != SLZ_SUCCESS) return error.SkipZigTest;
    try testing.expect(h != null);
    return h;
}

fn roundtripHostOnce(allocator: std.mem.Allocator, h: Handle, src: []const u8, level: c_int) !void {
    var bound: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, src.len, .{ .level = level }, &bound));
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);

    var comp_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressHost(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &comp_size,
        .{ .level = level },
    ));
    try testing.expect(comp_size > 0);
    try testing.expect(comp_size <= bound);

    // Size query must agree with the original input length.
    var content_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzGetDecompressedSize(h, compressed.ptr, &content_size));
    try testing.expectEqual(src.len, content_size);

    const dst = try allocator.alloc(u8, src.len + decoder.safe_space);
    defer allocator.free(dst);
    @memset(dst, 0xAA);
    var out_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressHost(
        h,
        compressed.ptr,
        comp_size,
        dst.ptr,
        dst.len,
        &out_size,
        .{},
    ));
    try testing.expectEqual(src.len, out_size);
    try testing.expectEqualSlices(u8, src, dst[0..out_size]);
}

// ── String + options contracts (host-only) ─────────────────────────

test "C ABI: slzVersionString and slzStatusString contracts" {
    const v = std.mem.span(slzVersionString());
    try testing.expect(v.len > 0);
    try testing.expect(std.mem.count(u8, v, ".") >= 2); // MAJOR.MINOR.PATCH

    try testing.expectEqualStrings("success", std.mem.span(slzStatusString(SLZ_SUCCESS)));
    try testing.expectEqualStrings("invalid handle", std.mem.span(slzStatusString(SLZ_ERROR_INVALID_HANDLE)));
    try testing.expectEqualStrings("unknown status", std.mem.span(slzStatusString(999)));
}

test "C ABI: default opts match the documented ABI values" {
    const c = slzCompressDefaultOpts();
    try testing.expectEqual(@as(c_int, 5), c.level);
    try testing.expectEqual(@as(c_int, 0), c.enable_profiling);
    const d = slzDecompressDefaultOpts();
    try testing.expectEqual(@as(c_int, 0), d.enable_profiling);
    // ABI freeze: both opts structs are exactly 32 bytes.
    try testing.expectEqual(@as(usize, 32), @sizeOf(CompressOpts));
    try testing.expectEqual(@as(usize, 32), @sizeOf(DecompressOpts));
}

// ── Lifecycle + validation ─────────────────────────────────────────

test "C ABI: slzCreate/slzDestroy lifecycle and null-arg validation" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();

    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzCreate(null));
    // Destroy of a null handle is a harmless no-op returning SUCCESS.
    try testing.expectEqual(SLZ_SUCCESS, slzDestroy(null));

    const h = try createOrSkip();
    try testing.expectEqual(SLZ_SUCCESS, slzDestroy(h));
}

test "C ABI: slzCompressBound contract" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    var small: usize = 0;
    var large: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, 1024, .{}, &small));
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, 1024 * 1024, .{}, &large));
    try testing.expect(small > 1024);
    try testing.expect(large > small);

    try testing.expectEqual(SLZ_ERROR_INVALID_HANDLE, slzCompressBound(null, 1024, .{}, &small));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzCompressBound(h, 1024, .{}, null));
}

test "C ABI: null-pointer validation on hot entry points" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    var buf: [256]u8 = undefined;
    var out_size: usize = 0;
    // slzCompressHost: each missing pointer is INVALID_ARG; missing
    // handle is INVALID_HANDLE.
    try testing.expectEqual(SLZ_ERROR_INVALID_HANDLE, slzCompressHost(null, &buf, buf.len, &buf, buf.len, &out_size, .{}));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzCompressHost(h, null, buf.len, &buf, buf.len, &out_size, .{}));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzCompressHost(h, &buf, buf.len, null, buf.len, &out_size, .{}));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzCompressHost(h, &buf, buf.len, &buf, buf.len, null, .{}));
    // slzDecompressHost: same shape.
    try testing.expectEqual(SLZ_ERROR_INVALID_HANDLE, slzDecompressHost(null, &buf, buf.len, &buf, buf.len, &out_size, .{}));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzDecompressHost(h, null, buf.len, &buf, buf.len, &out_size, .{}));
    // slzGetDecompressedSize: null bytes / null out.
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzGetDecompressedSize(h, null, &out_size));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzGetDecompressedSize(h, &buf, null));
}

test "C ABI: slzGetDecompressedSize rejects a corrupt header" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    var garbage: [64]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(0xBAD_F00D);
    rng.random().bytes(&garbage);
    garbage[0] = 0xFF; // guarantee a wrong magic
    var content_size: usize = 0;
    try testing.expectEqual(SLZ_ERROR_CORRUPT_FRAME, slzGetDecompressedSize(h, &garbage, &content_size));
}

// ── Host roundtrips ────────────────────────────────────────────────

test "C ABI: slzCompressHost + slzDecompressHost roundtrip at L1 and L5" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const allocator = testing.allocator;
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    const src = try repeatingPayload(allocator, 64 * 1024);
    defer allocator.free(src);
    try roundtripHostOnce(allocator, h, src, 1);
    try roundtripHostOnce(allocator, h, src, 5);
}

test "C ABI: repeated decompress into re-allocated same-size dst (F-002 shape)" {
    // The VK side's F-001 bug: a stale device-import cache entry keyed
    // by host address silently redirected the second decompress's GPU
    // writes. CUDA's ABI has no import cache, but the caller-visible
    // contract — N consecutive decompresses into freshly allocated
    // same-size buffers all produce correct bytes — must hold on both
    // backends, so the regression shape runs here too.
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const allocator = testing.allocator;
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    const src = try repeatingPayload(allocator, 32 * 1024);
    defer allocator.free(src);

    var bound: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, src.len, .{ .level = 1 }, &bound));
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    var comp_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressHost(h, src.ptr, src.len, compressed.ptr, compressed.len, &comp_size, .{ .level = 1 }));

    for (0..5) |_| {
        const dst = try allocator.alloc(u8, src.len + decoder.safe_space);
        defer allocator.free(dst);
        @memset(dst, 0);
        var out_size: usize = 0;
        try testing.expectEqual(SLZ_SUCCESS, slzDecompressHost(h, compressed.ptr, comp_size, dst.ptr, dst.len, &out_size, .{}));
        try testing.expectEqual(src.len, out_size);
        try testing.expectEqualSlices(u8, src, dst[0..out_size]);
    }
}

// ── Async true-D2D ─────────────────────────────────────────────────

test "C ABI: slzCompressAsync + slzDecompressAsync D2D roundtrip (NULL stream)" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const allocator = testing.allocator;
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    const malloc_fn = cuda.cuMemAlloc_fn orelse return error.SkipZigTest;
    const free_fn = cuda.cuMemFree_fn orelse return error.SkipZigTest;
    const h2d_fn = cuda.cuMemcpyHtoD_fn orelse return error.SkipZigTest;
    const d2h_fn = cuda.cuMemcpyDtoH_fn orelse return error.SkipZigTest;
    const sync_fn = cuda.cuCtxSynchronize_fn orelse return error.SkipZigTest;

    const src = try repeatingPayload(allocator, 64 * 1024);
    defer allocator.free(src);

    var bound: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, src.len, .{ .level = 1 }, &bound));

    var d_in: u64 = 0;
    if (malloc_fn(&d_in, src.len) != cuda.CUDA_SUCCESS) return error.SkipZigTest;
    defer _ = free_fn(d_in);
    var d_comp: u64 = 0;
    if (malloc_fn(&d_comp, bound) != cuda.CUDA_SUCCESS) return error.SkipZigTest;
    defer _ = free_fn(d_comp);
    var d_out: u64 = 0;
    if (malloc_fn(&d_out, src.len + decoder.safe_space) != cuda.CUDA_SUCCESS) return error.SkipZigTest;
    defer _ = free_fn(d_out);

    if (h2d_fn(d_in, @ptrCast(src.ptr), src.len) != cuda.CUDA_SUCCESS) return error.CopyFailed;

    // Tiny inputs must be rejected upfront (the encode pipeline's
    // host-side small-input path can't read a device pointer).
    var comp_size: usize = 0;
    try testing.expectEqual(SLZ_ERROR_UNSUPPORTED, slzCompressAsync(
        h,
        @ptrFromInt(d_in),
        64,
        @ptrFromInt(d_comp),
        bound,
        &comp_size,
        .{ .level = 1 },
        null,
    ));

    // Full D2D compress on the NULL stream (defaults to sync behavior).
    try testing.expectEqual(SLZ_SUCCESS, slzCompressAsync(
        h,
        @ptrFromInt(d_in),
        src.len,
        @ptrFromInt(d_comp),
        bound,
        &comp_size,
        .{ .level = 1 },
        null,
    ));
    try testing.expect(comp_size > 0);
    try testing.expect(comp_size <= bound);
    if (sync_fn() != cuda.CUDA_SUCCESS) return error.SyncFailed;

    // True-D2D decompress of the device-resident frame.
    try testing.expectEqual(SLZ_SUCCESS, slzDecompressAsync(
        h,
        @ptrFromInt(d_comp),
        comp_size,
        @ptrFromInt(d_out),
        src.len,
        .{},
        null,
    ));
    if (sync_fn() != cuda.CUDA_SUCCESS) return error.SyncFailed;

    const dst = try allocator.alloc(u8, src.len);
    defer allocator.free(dst);
    @memset(dst, 0x55);
    if (d2h_fn(@ptrCast(dst.ptr), d_out, src.len) != cuda.CUDA_SUCCESS) return error.CopyFailed;
    try testing.expectEqualSlices(u8, src, dst);
}

// ── Timings drain ──────────────────────────────────────────────────

test "C ABI: slzGetLastTimings and slzWaitAndGetLastTimings drain contract" {
    gpu_roundtrip_tests.lockGpuTests();
    defer gpu_roundtrip_tests.unlockGpuTests();
    const allocator = testing.allocator;
    const h = try createOrSkip();
    defer _ = slzDestroy(h);

    const src = try repeatingPayload(allocator, 64 * 1024);
    defer allocator.free(src);
    var bound: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressBound(h, src.len, .{}, &bound));
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);
    var comp_size: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzCompressHost(
        h,
        src.ptr,
        src.len,
        compressed.ptr,
        compressed.len,
        &comp_size,
        .{ .level = 5, .enable_profiling = 1 },
    ));

    // Count-only query (timings = NULL) must succeed and report the
    // profiled kernel count.
    var count: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzGetLastTimings(h, null, 0, &count));
    try testing.expect(count > 0); // L5 compress launches several kernels

    // Drain into a capacity-limited buffer: never writes past capacity.
    var buf: [4]KernelTimingC = undefined;
    var count2: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzGetLastTimings(h, &buf, buf.len, &count2));
    try testing.expectEqual(count, count2);
    const written = @min(count2, buf.len);
    for (buf[0..written]) |kt| {
        try testing.expect(std.mem.span(kt.name).len > 0);
        try testing.expect(kt.ms >= 0);
    }

    // Null-handle / null-count validation.
    try testing.expectEqual(SLZ_ERROR_INVALID_HANDLE, slzGetLastTimings(null, null, 0, &count));
    try testing.expectEqual(SLZ_ERROR_INVALID_ARG, slzGetLastTimings(h, null, 0, null));

    // The wait-variant with a NULL stream behaves like the plain call.
    var count3: usize = 0;
    try testing.expectEqual(SLZ_SUCCESS, slzWaitAndGetLastTimings(h, null, null, 0, &count3));
}
