//! M9 — Cross-backend conformance harness (4-direction matrix).
//!
//! Architecture: `docs/vulkan_port_architecture.md` §17 + port map §9.2 (3).
//!
//! The matrix is intentionally over-sized for the day-one substrate: only
//! direction D1 (CUDA-encode → CUDA-decode roundtrip) runs real codec; D2
//! (CUDA-encode → VK-decode), D3 (VK-encode → CUDA-decode) and D4
//! (VK-encode → VK-decode) all return `.skipped_unimplemented` until the
//! corresponding Vulkan production kernels land in waves 1 and 2.
//!
//! Existence of the harness is the M9 deliverable — every later Vulkan
//! kernel milestone flips one or more cells from `skipped_unimplemented`
//! to `pass` (or `fail`, which then blocks the milestone). The
//! pass/fail/skipped tally line at the bottom of the run is the visible
//! progress meter described in §3 of the milestone plan.
//!
//! The harness owns its three corpora at `assets/{web.txt, enwik8.txt,
//! silesia_all.tar}` and asserts byte-identical roundtrip per cell. The
//! `cellDiff` helper bracket-prints the first divergence with ±32-byte
//! context so a regression is debuggable from the test output alone.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

// Relative imports work because the enclosing test module is rooted at
// `tests_root.zig` (repo root), widening the package boundary to cover
// both `src/` and `tests/`. See `tests_root.zig` for the rationale.
const encoder = @import("../src/encode/streamlz_encoder.zig");
const decoder = @import("../src/decode/streamlz_decoder.zig");
const gpu_encoder = @import("../src/encode/driver.zig");
const gpu_driver = @import("../src/decode/driver.zig");

const Direction = enum {
    /// CUDA encode → CUDA decode (real, ships today).
    d1_cuda_cuda,
    /// CUDA encode → Vulkan decode (gated on Wave-1 / M10..M20).
    d2_cuda_vk,
    /// Vulkan encode → CUDA decode (gated on Wave-2 / M22..M26).
    d3_vk_cuda,
    /// Vulkan encode → Vulkan decode (gated on Wave-2 completion).
    d4_vk_vk,
};

const Cell = struct {
    corpus: []const u8,
    level: u8,
    direction: Direction,
    tier: []const u8, // "tier1", "tier1_nv", or "tier2"
};

const CellResult = enum {
    pass,
    fail,
    /// Backend not implemented yet (D2/D3/D4 today). Counts toward the
    /// skipped tally and the milestone-progress reporter; not a failure.
    skipped_unimplemented,
    /// The dev machine lacks the device for this direction (no Vulkan
    /// loader, no NVIDIA GPU, etc.). Counts toward the skipped tally.
    skipped_device_mismatch,
};

// ── Corpora and levels matrix ─────────────────────────────────────────
const corpora = [_][]const u8{
    "assets/web.txt",
    "assets/enwik8.txt",
    "assets/silesia_all.tar",
};

const levels = [_]u8{ 1, 2, 3, 4, 5 };

const directions = [_]Direction{
    .d1_cuda_cuda,
    .d2_cuda_vk,
    .d3_vk_cuda,
    .d4_vk_vk,
};

// ── Corpus cache ──────────────────────────────────────────────────────
// Each corpus is read at most once per test run. The cache is consulted
// before every D1 cell so the 5 levels × 3 corpora = 15 real roundtrips
// avoid 15 separate file reads of the 200+ MiB silesia tarball.

const CorpusEntry = struct {
    bytes: ?[]u8 = null,
    /// `true` once the file was attempted (whether successful or not).
    /// Subsequent attempts use the cached `bytes` (which may be `null` if
    /// the file is missing — then the cell skips as device_mismatch).
    tried: bool = false,
};

var corpus_cache: [corpora.len]CorpusEntry = @splat(.{});

fn loadCorpus(allocator: std.mem.Allocator, io: std.Io, path: []const u8) ?[]const u8 {
    const idx = for (corpora, 0..) |c, i| {
        if (std.mem.eql(u8, c, path)) break i;
    } else return null;

    const entry = &corpus_cache[idx];
    if (entry.tried) return entry.bytes;
    entry.tried = true;

    // Hard cap on per-corpus size — `silesia_all.tar` is ~213 MiB, so 1 GiB
    // gives generous headroom while still bounding a hostile / mis-named
    // file. The cap is a sanity check, not a feature gate.
    const max_corpus_bytes: usize = 1 << 30;
    const buf = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch return null;
    entry.bytes = buf;
    return buf;
}

fn freeCorpusCache(allocator: std.mem.Allocator) void {
    for (&corpus_cache) |*entry| {
        if (entry.bytes) |b| allocator.free(b);
        entry.* = .{};
    }
}

// ── Cell diff reporter ────────────────────────────────────────────────
// First-byte-offset divergence + ±32 bytes of context on each side. Run
// only on .fail; the goal is to make a regression actionable from the
// test log alone without re-running the harness under a debugger.
fn cellDiff(expected: []const u8, actual: []const u8) void {
    if (expected.len != actual.len) {
        std.debug.print(
            "  size mismatch: expected {d} bytes, got {d}\n",
            .{ expected.len, actual.len },
        );
    }
    const min_len = @min(expected.len, actual.len);
    var off: usize = 0;
    while (off < min_len) : (off += 1) {
        if (expected[off] != actual[off]) break;
    }
    if (off == min_len and expected.len == actual.len) return;

    const ctx: usize = 32;
    const lo = if (off > ctx) off - ctx else 0;
    const exp_hi = @min(expected.len, off + ctx);
    const act_hi = @min(actual.len, off + ctx);
    std.debug.print("  first diff at byte offset {d} (0x{x})\n", .{ off, off });
    std.debug.print("    expected[{d}..{d}]:", .{ lo, exp_hi });
    for (expected[lo..exp_hi]) |b| std.debug.print(" {x:0>2}", .{b});
    std.debug.print("\n", .{});
    std.debug.print("    actual  [{d}..{d}]:", .{ lo, act_hi });
    for (actual[lo..act_hi]) |b| std.debug.print(" {x:0>2}", .{b});
    std.debug.print("\n", .{});
}

// ── CUDA → CUDA real cell (D1) ────────────────────────────────────────
// Single-pass roundtrip on the supplied corpus + level using the same
// gpu_encoder / gpu_driver singletons the production CLI exercises.
// Failures report the first-byte divergence via cellDiff and return
// `.fail`; missing corpus returns `.skipped_device_mismatch`.
fn runD1(allocator: std.mem.Allocator, io: std.Io, corpus_path: []const u8, level: u8) CellResult {
    const src = loadCorpus(allocator, io, corpus_path) orelse return .skipped_device_mismatch;

    const bound = encoder.compressBound(src.len);
    const compressed = allocator.alloc(u8, bound) catch return .fail;
    defer allocator.free(compressed);

    const n = encoder.compressFramed(
        allocator,
        src,
        compressed,
        .{ .level = level },
        &gpu_encoder.g_default,
    ) catch |err| {
        std.debug.print(
            "  D1 compress failed: corpus={s} level={d} err={s}\n",
            .{ corpus_path, level, @errorName(err) },
        );
        return .fail;
    };

    const dst = allocator.alloc(u8, src.len + decoder.safe_space) catch return .fail;
    defer allocator.free(dst);

    const written = decoder.decompressFramed(compressed[0..n], dst, &gpu_driver.g_default) catch |err| {
        std.debug.print(
            "  D1 decompress failed: corpus={s} level={d} err={s}\n",
            .{ corpus_path, level, @errorName(err) },
        );
        return .fail;
    };
    if (written != src.len) {
        std.debug.print(
            "  D1 size mismatch: corpus={s} level={d} expected={d} got={d}\n",
            .{ corpus_path, level, src.len, written },
        );
        return .fail;
    }
    if (!std.mem.eql(u8, src, dst[0..written])) {
        std.debug.print("  D1 byte mismatch: corpus={s} level={d}\n", .{ corpus_path, level });
        cellDiff(src, dst[0..written]);
        return .fail;
    }
    return .pass;
}

// ── Cell dispatch ─────────────────────────────────────────────────────
// D1 runs real; D2/D3/D4 return `skipped_unimplemented` until the
// matching Vulkan kernels land. The Vulkan-tagged cells will fan out to
// their own runD2/runD3/runD4 once src_vulkan exposes a compress /
// decompress entry point (currently every `_vk` ABI symbol returns
// SLZ_ERROR_UNSUPPORTED per M2).
fn runCell(allocator: std.mem.Allocator, io: std.Io, cell: Cell) CellResult {
    return switch (cell.direction) {
        .d1_cuda_cuda => runD1(allocator, io, cell.corpus, cell.level),
        .d2_cuda_vk => .skipped_unimplemented,
        .d3_vk_cuda => .skipped_unimplemented,
        .d4_vk_vk => .skipped_unimplemented,
    };
}

// ── Tier probe ────────────────────────────────────────────────────────
// Placeholder until src_vulkan/probe.zig is wired through; the matrix is
// stamped with this value so post-M21 dashboards can group results by
// tier without rerunning. M9 only runs D1 cells so the choice is cosmetic.
fn probeTier() []const u8 {
    return "tier1"; // M4 will replace with a real loader+probe call.
}

// ── The conformance test ──────────────────────────────────────────────
test "cross-backend conformance matrix" {
    const allocator = testing.allocator;
    defer freeCorpusCache(allocator);

    // Stand up a local threaded `std.Io` instance — Zig 0.16 routes every
    // file I/O through an `Io` vtable, and the parallel test runner does
    // not hand one out. `Allocator.failing` is OK because the corpus
    // loader never calls async / concurrent operations.
    var io_inst: std.Io.Threaded = .init(std.mem.Allocator.failing, .{});
    defer io_inst.deinit();
    const io = io_inst.io();

    // D1 needs the CUDA backend up. If absent, every D1 cell drops to
    // .skipped_device_mismatch; the test still runs the matrix (it's a
    // dashboard, not a gate) and asserts only that no cell hit .fail.
    const cuda_up = gpu_encoder.isAvailable() and
        gpu_driver.isAvailable() and
        gpu_driver.bindContextToCallingThread();

    const tier = probeTier();

    var pass: u32 = 0;
    var fail: u32 = 0;
    var skipped: u32 = 0;

    for (corpora) |corpus| {
        for (levels) |lvl| {
            for (directions) |dir| {
                const cell: Cell = .{
                    .corpus = corpus,
                    .level = lvl,
                    .direction = dir,
                    .tier = tier,
                };
                const r = if (!cuda_up and dir == .d1_cuda_cuda)
                    CellResult.skipped_device_mismatch
                else
                    runCell(allocator, io, cell);
                switch (r) {
                    .pass => pass += 1,
                    .fail => fail += 1,
                    .skipped_unimplemented, .skipped_device_mismatch => skipped += 1,
                }
            }
        }
    }

    std.debug.print(
        "conformance: {d} pass / {d} fail / {d} skipped\n",
        .{ pass, fail, skipped },
    );
    try testing.expectEqual(@as(u32, 0), fail);
}
