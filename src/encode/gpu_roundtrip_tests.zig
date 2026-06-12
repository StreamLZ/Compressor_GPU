//! GPU encode + decode roundtrip integration tests.
//!
//! The parallel test runner can fire multiple tests onto the same CUDA
//! context concurrently, which is unsafe — we serialize the entire GPU
//! payload behind a single test function. Each shape runs the
//! compress → decompress → byte-compare cycle and reports the first
//! failure with a label. The whole test skips (not fails) when CUDA is
//! unavailable at test time.

const std = @import("std");
const testing = std.testing;
const encoder = @import("streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_encoder = @import("driver.zig");
const gpu_driver = @import("../decode/driver.zig");

// 2026-06-10: the module_loader init() race used to make all but the
// first GPU test fn skip — which accidentally serialized GPU work.
// With init properly locked, every GPU test fn RUNS, and the parallel
// runner would fire them concurrently onto the shared g_default
// encode/decode contexts (persistent-buffer races → bogus
// DestinationTooSmall and worse). Serialize every GPU test fn body
// behind one process-wide lock; host-only tests stay parallel.
const SRWLOCK = extern struct { ptr: ?*anyopaque = null };
extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
var g_gpu_test_lock: SRWLOCK = .{};

pub fn lockGpuTests() void {
    AcquireSRWLockExclusive(&g_gpu_test_lock);
    // The CUDA context is per-thread current, and ptest schedules tests
    // onto arbitrary worker threads. A GPU test landing on a thread
    // where the context was never made current fails its first raw
    // driver call with CUDA_ERROR_INVALID_CONTEXT (201) — seen as the
    // deterministic longlitrun H2D failure and the intermittent "every
    // shape and level" flake (2026-06-10). Binding here covers every
    // GPU test in one place. Benign no-op when the context doesn't
    // exist yet (the first test's init() creates it current).
    _ = gpu_driver.bindContextToCallingThread();
}

pub fn unlockGpuTests() void {
    ReleaseSRWLockExclusive(&g_gpu_test_lock);
}

const Case = struct {
    label: []const u8,
    bytes: []const u8,
    level: u8,
};

fn roundtripOne(allocator: std.mem.Allocator, case: Case) !void {
    const bound = encoder.compressBound(case.bytes.len);
    const compressed = try allocator.alloc(u8, bound);
    defer allocator.free(compressed);

    const n = encoder.compressFramed(
        allocator,
        case.bytes,
        compressed,
        .{ .level = case.level },
        &gpu_encoder.g_default,
    ) catch |err| {
        std.debug.print("compress failed for {s}: {s}\n", .{ case.label, @errorName(err) });
        return err;
    };
    try testing.expect(n > 0);
    try testing.expect(n <= bound);

    const dst = try allocator.alloc(u8, case.bytes.len + decoder.safe_space);
    defer allocator.free(dst);
    const written = decoder.decompressFramed(compressed[0..n], dst, &gpu_driver.g_default) catch |err| {
        std.debug.print("decompress failed for {s}: {s}\n", .{ case.label, @errorName(err) });
        return err;
    };
    if (written != case.bytes.len) {
        std.debug.print("size mismatch for {s}: expected {d}, got {d}\n", .{ case.label, case.bytes.len, written });
        return error.TestUnexpectedResult;
    }
    if (!std.mem.eql(u8, case.bytes, dst[0..written])) {
        var first_diff: usize = 0;
        while (first_diff < case.bytes.len and case.bytes[first_diff] == dst[first_diff]) : (first_diff += 1) {}
        std.debug.print("byte mismatch for {s} at offset {d}\n", .{ case.label, first_diff });
        // v4 #18: this mismatch is intermittent (~1 in 30+ suite runs;
        // unreproduced by 1100+ targeted in-process/CLI roundtrips of
        // the same shape via `zig build stress18`) — capture the
        // evidence the moment it fires again: the exact frame and the
        // wrong output, replayable offline (`streamlz -d` the .frame,
        // diff against the .out). Capture failure is non-fatal.
        captureMismatchArtifacts(case.label, compressed[0..n], dst[0..written]);
        return error.TestUnexpectedResult;
    }
}

/// v4 #18 evidence capture (see the call site above). libc I/O — no
/// std.Io plumbing exists in this test path.
fn captureMismatchArtifacts(label: []const u8, frame: []const u8, out: []const u8) void {
    var name_buf: [256]u8 = undefined;
    var safe_label_buf: [64]u8 = undefined;
    var sl: usize = 0;
    for (label) |c| {
        if (sl >= safe_label_buf.len) break;
        safe_label_buf[sl] = if (c == ' ') '_' else c;
        sl += 1;
    }
    const f1 = std.fmt.bufPrintZ(&name_buf, "v18_mismatch_{s}.frame", .{safe_label_buf[0..sl]}) catch return;
    if (std.c.fopen(f1, "wb")) |fh| {
        _ = std.c.fwrite(frame.ptr, 1, frame.len, fh);
        _ = std.c.fclose(fh);
    }
    const f2 = std.fmt.bufPrintZ(&name_buf, "v18_mismatch_{s}.out", .{safe_label_buf[0..sl]}) catch return;
    if (std.c.fopen(f2, "wb")) |fh| {
        _ = std.c.fwrite(out.ptr, 1, out.len, fh);
        _ = std.c.fclose(fh);
    }
    std.debug.print("v4 #18 artifacts captured: v18_mismatch_{s}.frame/.out\n", .{safe_label_buf[0..sl]});
}

test "GPU roundtrip: every shape and level" {
    lockGpuTests();
    defer unlockGpuTests();
    const allocator = testing.allocator;

    if (!gpu_encoder.isAvailable()) {
        std.debug.print("(skipping GPU tests — CUDA not available)\n", .{});
        return error.SkipZigTest;
    }
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;

    // Bind the CUDA context to whichever runner thread picked this up.
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;

    // Generate the per-shape payloads once, reuse across levels.
    const tiny = "Hello, world!\n";

    const small_repeating = try allocator.alloc(u8, 4096);
    defer allocator.free(small_repeating);
    for (small_repeating, 0..) |*b, i| b.* = @intCast('A' + (i % 26));

    const medium_text = try allocator.alloc(u8, 100 * 1024);
    defer allocator.free(medium_text);
    {
        const pattern = "The quick brown fox jumps over a lazy dog. ";
        var i: usize = 0;
        while (i < medium_text.len) : (i += 1) medium_text[i] = pattern[i % pattern.len];
    }

    const sub_chunk_boundary = try allocator.alloc(u8, 128 * 1024 + 1);
    defer allocator.free(sub_chunk_boundary);
    for (sub_chunk_boundary, 0..) |*b, i| b.* = @intCast('a' + (i % 7));

    const incompressible = try allocator.alloc(u8, 32 * 1024);
    defer allocator.free(incompressible);
    {
        var state: u32 = 0xC0FFEE;
        for (incompressible) |*b| {
            state = state *% 1103515245 +% 12345;
            b.* = @intCast((state >> 16) & 0xFF);
        }
    }

    const cases = [_]Case{
        .{ .label = "tiny L1", .bytes = tiny, .level = 1 },
        .{ .label = "tiny L3", .bytes = tiny, .level = 3 },
        .{ .label = "tiny L5", .bytes = tiny, .level = 5 },
        .{ .label = "small repeating L1", .bytes = small_repeating, .level = 1 },
        .{ .label = "small repeating L3", .bytes = small_repeating, .level = 3 },
        .{ .label = "medium text L1", .bytes = medium_text, .level = 1 },
        .{ .label = "medium text L2", .bytes = medium_text, .level = 2 },
        .{ .label = "medium text L3", .bytes = medium_text, .level = 3 },
        .{ .label = "medium text L4", .bytes = medium_text, .level = 4 },
        .{ .label = "medium text L5", .bytes = medium_text, .level = 5 },
        .{ .label = "sub-chunk boundary L1", .bytes = sub_chunk_boundary, .level = 1 },
        .{ .label = "sub-chunk boundary L5", .bytes = sub_chunk_boundary, .level = 5 },
        .{ .label = "incompressible L1", .bytes = incompressible, .level = 1 },
        .{ .label = "incompressible L5", .bytes = incompressible, .level = 5 },
    };

    for (cases) |c| try roundtripOne(allocator, c);
}

test "GPU roundtrip: empty input at every level" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    var level: u8 = 1;
    while (level <= 5) : (level += 1) {
        try roundtripOne(allocator, .{ .label = "empty", .bytes = &.{}, .level = level });
    }
}

test "compressBound is a strict upper bound on real frames" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const sizes = [_]usize{ 0, 1, 1024, 4096, 64 * 1024, 256 * 1024, 1024 * 1024 };
    for (sizes) |sz| {
        const buf = try allocator.alloc(u8, sz);
        defer allocator.free(buf);
        for (buf, 0..) |*b, i| b.* = @intCast(i & 0xFF);
        const bound = encoder.compressBound(sz);
        const dst = try allocator.alloc(u8, bound);
        defer allocator.free(dst);
        const n = try encoder.compressFramed(allocator, buf, dst, .{ .level = 3 }, &gpu_encoder.g_default);
        try testing.expect(n <= bound);
    }
}

test "compressFramed rejects level outside 1..5" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    var dst: [4096]u8 = undefined;
    const src = "hello";
    try testing.expectError(error.BadLevel, encoder.compressFramed(allocator, src, &dst, .{ .level = 0 }, &gpu_encoder.g_default));
    try testing.expectError(error.BadLevel, encoder.compressFramed(allocator, src, &dst, .{ .level = 6 }, &gpu_encoder.g_default));
    try testing.expectError(error.BadLevel, encoder.compressFramed(allocator, src, &dst, .{ .level = 12 }, &gpu_encoder.g_default));
}

test "compressFramed rejects undersized destination" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    var dst: [4]u8 = undefined;
    const src = "hello world, this needs more than 4 bytes of output space";
    try testing.expectError(
        error.DestinationTooSmall,
        encoder.compressFramed(allocator, src, &dst, .{ .level = 1 }, &gpu_encoder.g_default),
    );
}

test "v4 #16: dictionary frames roundtrip, compress better, enforce the registry" {
    lockGpuTests();
    defer unlockGpuTests();
    if (!gpu_encoder.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.isAvailable()) return error.SkipZigTest;
    if (!gpu_driver.bindContextToCallingThread()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const dictionary = @import("../dict/dictionary.zig");
    const frame_format = @import("../format/frame_format.zig");

    // Small-records payload built FROM dictionary bytes: 120 records,
    // each a slice of the json dictionary's tail plus a unique
    // decoration. Dictionary matches are guaranteed by construction,
    // so the ratio assertions below are deterministic.
    const dict_data = dictionary.findById(dictionary.id_json).?.data;
    var payload_list: std.ArrayList(u8) = .empty;
    defer payload_list.deinit(allocator);
    for (0..120) |i| {
        var dec_buf: [32]u8 = undefined;
        const dec = std.fmt.bufPrint(&dec_buf, "#rec{d}:{d};", .{ i, i * 31 }) catch unreachable;
        try payload_list.appendSlice(allocator, dec);
        const start = dict_data.len - 2048 + (i * 13) % 1600;
        try payload_list.appendSlice(allocator, dict_data[start .. start + 192]);
    }
    const payload = payload_list.items;

    const bound = encoder.compressBound(payload.len);
    const plain = try allocator.alloc(u8, bound);
    defer allocator.free(plain);
    const with_dict = try allocator.alloc(u8, bound);
    defer allocator.free(with_dict);

    for ([_]u8{ 1, 2, 3, 4, 5 }) |level| {
        const n_plain = try encoder.compressFramed(allocator, payload, plain, .{ .level = level }, &gpu_encoder.g_default);
        const n_dict = try encoder.compressFramed(
            allocator,
            payload,
            with_dict,
            .{ .level = level, .dictionary_id = dictionary.id_json },
            &gpu_encoder.g_default,
        );

        const hdr = frame_format.parseHeader(with_dict[0..n_dict]) catch unreachable;
        try testing.expectEqual(@as(?u32, dictionary.id_json), hdr.dictionary_id);
        const id_pos = hdr.header_size - 4;

        // Every level searches the dictionary (greedy at L1-4, the
        // chain parser at L5): on this dict-derived payload the dict
        // frame must be MEANINGFULLY smaller (the cold-start content
        // is dict-matchable by construction). The bar is 20% for the
        // greedy levels and 10% for L5 - the lazy parser already
        // captures most inter-record redundancy on its own, so the
        // dictionary's marginal lift is structurally smaller there.
        const min_cut: usize = if (level == 5) n_plain / 10 else n_plain / 5;
        if (n_dict >= n_plain - min_cut) {
            std.debug.print(
                "dict ratio L{d}: plain {d} B, dict {d} B - reduction below the bar\n",
                .{ level, n_plain, n_dict },
            );
            return error.TestUnexpectedResult;
        }

        // Dict frames roundtrip byte-exact through the dict-aware
        // decode kernels at every level.
        const dst = try allocator.alloc(u8, payload.len + decoder.safe_space);
        defer allocator.free(dst);
        const written = try decoder.decompressFramed(with_dict[0..n_dict], dst, &gpu_driver.g_default);
        try testing.expectEqual(payload.len, written);
        try testing.expectEqualSlices(u8, payload, dst[0..written]);

        // Unknown IDs are rejected on both sides. The decoder check
        // patches the ID field in place - no checksum covers header
        // bytes, so the frame stays otherwise valid.
        try testing.expectError(error.UnknownDictionary, encoder.compressFramed(
            allocator,
            payload,
            plain,
            .{ .level = level, .dictionary_id = 999 },
            &gpu_encoder.g_default,
        ));
        std.mem.writeInt(u32, with_dict[id_pos..][0..4], 0xDEAD_BEEF, .little);
        try testing.expectError(
            error.UnknownDictionary,
            decoder.decompressFramed(with_dict[0..n_dict], dst, &gpu_driver.g_default),
        );
    }
}
