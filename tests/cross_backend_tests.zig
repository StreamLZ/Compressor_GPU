//! M9 — Cross-backend conformance harness (4-direction matrix).
//!
//! Architecture: `docs/vulkan_port_architecture.md` §17 + port map §9.2 (3).
//!
//! Today's matrix:
//!   * D1 = CUDA-encode → CUDA-decode (production code path, runs when CUDA is up).
//!   * D2 = CUDA-encode → VK-decode (production path; CUDA encoder + VK
//!     `slz1_codec.decodeSlz1ToBytes`).
//!   * D3 = VK-encode → CUDA-decode (production path; VK
//!     `slz1_codec.encodeL1ToSlz1` + CUDA decoder).
//!   * D4 = VK-encode → VK-decode (production path; both halves Vulkan).
//!
//! Cluster F fix (F010): the matrix used to hard-code D2/D3/D4 to
//! `skipped_unimplemented` even though the Vulkan production codec has
//! been live for several milestones. This file now drives every cell
//! through the same production entry points the CLI binaries call,
//! producing real PASS/FAIL signal on the cross-backend matrix.
//!
//! All Vulkan cells are L1-only (the VK codec is L1-level only today),
//! so for D2/D3/D4 only `level == 1` runs real codec; other levels
//! report `.skipped_unimplemented` until the higher-level Vulkan path
//! lands. D1 still runs all 5 levels.
//!
//! The harness owns its three corpora at `assets/{web.txt, enwik8.txt,
//! silesia_all.tar}` and asserts byte-identical roundtrip per cell.
//! `cellDiff` brackets the first divergence with ±32 bytes of context.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

// Relative imports work because the enclosing test module is rooted at
// `tests_root.zig` (repo root), widening the package boundary to cover
// `src/`, `src_vulkan/`, and `tests/`. See `tests_root.zig`.
const encoder = @import("../src/encode/streamlz_encoder.zig");
const decoder = @import("../src/decode/streamlz_decoder.zig");
const gpu_encoder = @import("../src/encode/driver.zig");
const gpu_driver = @import("../src/decode/driver.zig");

const vk_driver = @import("../src_vulkan/driver.zig");
const vk_probe = @import("../src_vulkan/probe.zig");
const vk_slz1 = @import("../src_vulkan/slz1_codec.zig");

const Direction = enum {
    /// CUDA encode → CUDA decode.
    d1_cuda_cuda,
    /// CUDA encode → Vulkan decode.
    d2_cuda_vk,
    /// Vulkan encode → CUDA decode.
    d3_vk_cuda,
    /// Vulkan encode → Vulkan decode.
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
    /// Backend pair not implemented yet at this level. The Vulkan
    /// codec is L1-only today, so D2/D3/D4 at level > 1 still hit
    /// this terminal state. D1 never reports it.
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
// before every cell so the 5 levels × 3 corpora × 4 directions = 60
// real roundtrips avoid 60 separate file reads of the 200+ MiB silesia
// tarball.

const CorpusEntry = struct {
    bytes: ?[]u8 = null,
    /// Set once we've attempted the corpus; further attempts re-use the
    /// cached `bytes` (which can be `null` when `missing == true`).
    tried: bool = false,
    /// `true` if the file is legitimately absent (open returned
    /// FileNotFound). The cell maps this to `.skipped_device_mismatch`.
    missing: bool = false,
};

var corpus_cache: [corpora.len]CorpusEntry = @splat(.{});

/// Outcome of a corpus load:
///   * `.ok(bytes)` — bytes ready (cached for the rest of the run).
///   * `.missing` — file legitimately absent (FileNotFound), skip cell.
///   * `.io_error` — some other IO failure (permission, corruption,
///     short read); the caller MUST fail loudly because this is a real
///     bug, not "the dev box doesn't have the corpus".
const CorpusLoad = union(enum) {
    ok: []const u8,
    missing,
    io_error: anyerror,
};

fn loadCorpus(allocator: std.mem.Allocator, io: std.Io, path: []const u8) CorpusLoad {
    const idx = for (corpora, 0..) |c, i| {
        if (std.mem.eql(u8, c, path)) break i;
    } else return .{ .io_error = error.UnknownCorpus };

    const entry = &corpus_cache[idx];
    if (entry.tried) {
        if (entry.missing) return .missing;
        if (entry.bytes) |b| return .{ .ok = b };
        // Tried but neither cached nor flagged missing — treat as IO
        // error reproducer (a prior call hit this path and burned the
        // entry; surfacing the failure is the point).
        return .{ .io_error = error.CorpusLoadFailed };
    }
    entry.tried = true;

    // Hard cap on per-corpus size — `silesia_all.tar` is ~213 MiB, so
    // 1 GiB gives generous headroom while still bounding a hostile /
    // mis-named file. The cap is a sanity check, not a feature gate.
    const max_corpus_bytes: usize = 1 << 30;
    const buf = std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        @enumFromInt(max_corpus_bytes),
    ) catch |err| switch (err) {
        // Cluster F fix (F050): distinguish "the dev box doesn't have
        // this corpus" (FileNotFound → skip cell, dashboard-friendly)
        // from real IO errors (permission denied, corrupt file, short
        // read, etc.) which used to be silently collapsed into a skip
        // and let real failures go undetected.
        // FileNotFound is the only "legitimately absent" branch we
        // can take here; the other variants in the upstream error set
        // (PermissionDenied, InputOutput, IsDir, NameTooLong, etc.)
        // all indicate either a hostile filesystem or a misconfigured
        // dev box, and the F050 fix demands we surface them loudly.
        error.FileNotFound => {
            entry.missing = true;
            return .missing;
        },
        else => return .{ .io_error = err },
    };
    entry.bytes = buf;
    return .{ .ok = buf };
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

// ── CUDA helpers ──────────────────────────────────────────────────────

fn cudaEncode(
    allocator: std.mem.Allocator,
    src: []const u8,
    level: u8,
) ![]u8 {
    const bound = encoder.compressBound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    errdefer allocator.free(compressed);
    const n = try encoder.compressFramed(
        allocator,
        src,
        compressed,
        .{ .level = level },
        &gpu_encoder.g_default,
    );
    return allocator.realloc(compressed, n) catch compressed[0..n];
}

fn cudaDecode(
    allocator: std.mem.Allocator,
    compressed: []const u8,
    expected_size: usize,
) ![]u8 {
    const dst = try allocator.alloc(u8, expected_size + decoder.safe_space);
    errdefer allocator.free(dst);
    const written = try decoder.decompressFramed(compressed, dst, &gpu_driver.g_default);
    if (written != expected_size) return error.SizeMismatch;
    return allocator.realloc(dst, written) catch dst[0..written];
}

// ── Vulkan helpers ────────────────────────────────────────────────────

fn vkEncode(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
) ![]u8 {
    const bound = vk_slz1.slz1Bound(src.len);
    const compressed = try allocator.alloc(u8, bound);
    errdefer allocator.free(compressed);
    const n = try vk_slz1.encodeL1ToSlz1(
        &vk_driver.g_default,
        io,
        allocator,
        src,
        compressed,
    );
    return allocator.realloc(compressed, n) catch compressed[0..n];
}

fn vkDecode(
    allocator: std.mem.Allocator,
    io: std.Io,
    compressed: []const u8,
    expected_size: usize,
) ![]u8 {
    const dst = try allocator.alloc(u8, expected_size);
    errdefer allocator.free(dst);
    const written = try vk_slz1.decodeSlz1ToBytes(
        &vk_driver.g_default,
        io,
        allocator,
        compressed,
        dst,
    );
    if (written != expected_size) return error.SizeMismatch;
    return dst[0..written];
}

// ── Direction implementations ─────────────────────────────────────────
//
// All four run the production code paths. D1 reuses the original CUDA
// pipeline; D2/D3/D4 route through `slz1_codec.encodeL1ToSlz1` /
// `decodeSlz1ToBytes` — the same entry points `streamlz_vk.exe` and
// `slzCompressHost_vk` / `slzDecompressHost_vk` call. No test-only
// fixtures; no shim decode helpers. If a redirected cell fails it's a
// real production bug, not a test plumbing issue.

fn runD1(allocator: std.mem.Allocator, src: []const u8, corpus_path: []const u8, level: u8) CellResult {
    const compressed = cudaEncode(allocator, src, level) catch |err| {
        std.debug.print(
            "  D1 compress failed: corpus={s} level={d} err={s}\n",
            .{ corpus_path, level, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(compressed);

    const dst = cudaDecode(allocator, compressed, src.len) catch |err| {
        std.debug.print(
            "  D1 decompress failed: corpus={s} level={d} err={s}\n",
            .{ corpus_path, level, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(dst);

    if (!std.mem.eql(u8, src, dst)) {
        std.debug.print("  D1 byte mismatch: corpus={s} level={d}\n", .{ corpus_path, level });
        cellDiff(src, dst);
        return .fail;
    }
    return .pass;
}

fn runD2(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    corpus_path: []const u8,
    level: u8,
) CellResult {
    // CUDA-encode → VK-decode. Only level == 1 runs real codec; the VK
    // decoder is L1-only.
    const compressed = cudaEncode(allocator, src, level) catch |err| {
        std.debug.print(
            "  D2 CUDA-encode failed: corpus={s} level={d} err={s}\n",
            .{ corpus_path, level, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(compressed);

    const dst = vkDecode(allocator, io, compressed, src.len) catch |err| {
        std.debug.print(
            "  D2 VK-decode failed: corpus={s} level={d} err={s}\n",
            .{ corpus_path, level, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(dst);

    if (!std.mem.eql(u8, src, dst)) {
        std.debug.print("  D2 byte mismatch: corpus={s} level={d}\n", .{ corpus_path, level });
        cellDiff(src, dst);
        return .fail;
    }
    return .pass;
}

fn runD3(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    corpus_path: []const u8,
    level: u8,
) CellResult {
    // VK-encode → CUDA-decode. Level fixed at 1 (VK encoder is L1-only).
    _ = level;
    const compressed = vkEncode(allocator, io, src) catch |err| {
        std.debug.print(
            "  D3 VK-encode failed: corpus={s} err={s}\n",
            .{ corpus_path, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(compressed);

    const dst = cudaDecode(allocator, compressed, src.len) catch |err| {
        std.debug.print(
            "  D3 CUDA-decode failed: corpus={s} err={s}\n",
            .{ corpus_path, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(dst);

    if (!std.mem.eql(u8, src, dst)) {
        std.debug.print("  D3 byte mismatch: corpus={s}\n", .{corpus_path});
        cellDiff(src, dst);
        return .fail;
    }
    return .pass;
}

fn runD4(
    allocator: std.mem.Allocator,
    io: std.Io,
    src: []const u8,
    corpus_path: []const u8,
    level: u8,
) CellResult {
    _ = level;
    const compressed = vkEncode(allocator, io, src) catch |err| {
        std.debug.print(
            "  D4 VK-encode failed: corpus={s} err={s}\n",
            .{ corpus_path, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(compressed);

    const dst = vkDecode(allocator, io, compressed, src.len) catch |err| {
        std.debug.print(
            "  D4 VK-decode failed: corpus={s} err={s}\n",
            .{ corpus_path, @errorName(err) },
        );
        return .fail;
    };
    defer allocator.free(dst);

    if (!std.mem.eql(u8, src, dst)) {
        std.debug.print("  D4 byte mismatch: corpus={s}\n", .{corpus_path});
        cellDiff(src, dst);
        return .fail;
    }
    return .pass;
}

// ── Cell dispatch ─────────────────────────────────────────────────────

const Backends = struct {
    cuda_up: bool,
    vk_up: bool,
};

fn runCell(
    allocator: std.mem.Allocator,
    io: std.Io,
    cell: Cell,
    backends: Backends,
    src: []const u8,
) CellResult {
    return switch (cell.direction) {
        .d1_cuda_cuda => if (!backends.cuda_up)
            .skipped_device_mismatch
        else
            runD1(allocator, src, cell.corpus, cell.level),
        .d2_cuda_vk => blk: {
            if (!backends.cuda_up or !backends.vk_up) break :blk .skipped_device_mismatch;
            // VK decoder is L1-only; higher levels stay parked.
            if (cell.level != 1) break :blk .skipped_unimplemented;
            break :blk runD2(allocator, io, src, cell.corpus, cell.level);
        },
        .d3_vk_cuda => blk: {
            if (!backends.cuda_up or !backends.vk_up) break :blk .skipped_device_mismatch;
            if (cell.level != 1) break :blk .skipped_unimplemented;
            break :blk runD3(allocator, io, src, cell.corpus, cell.level);
        },
        .d4_vk_vk => blk: {
            if (!backends.vk_up) break :blk .skipped_device_mismatch;
            if (cell.level != 1) break :blk .skipped_unimplemented;
            break :blk runD4(allocator, io, src, cell.corpus, cell.level);
        },
    };
}

// ── Tier probe ────────────────────────────────────────────────────────
// Cluster F fix (F010): used to hard-code `"tier1"`. Now drives a real
// `src_vulkan/probe.zig::probe()` call and maps the returned tier into
// the dashboard's tier string. Falls back to "unsupported" when the
// Vulkan loader is absent or the device probes below our tier-2 floor.
fn probeTier() []const u8 {
    if (vk_driver.ensureInit()) |_| {} else |_| {
        return "unsupported";
    }
    const r = vk_probe.probe(vk_driver.g_default.inst, vk_driver.g_default.pd);
    return switch (r.tier) {
        .tier1 => "tier1",
        .tier1_nv => "tier1_nv",
        .tier2 => "tier2",
        .unsupported => "unsupported",
    };
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
    // `.skipped_device_mismatch`. D2/D3 also need CUDA; D4 just needs VK.
    const cuda_up = gpu_encoder.isAvailable() and
        gpu_driver.isAvailable() and
        gpu_driver.bindContextToCallingThread();

    // Drive ensureInit through probeTier (it's the one place that
    // gates everything else on Vulkan bring-up). If the loader is
    // missing, every D2/D3/D4 cell drops to `.skipped_device_mismatch`.
    const tier = probeTier();
    const vk_up = !std.mem.eql(u8, tier, "unsupported");

    const backends: Backends = .{ .cuda_up = cuda_up, .vk_up = vk_up };

    var pass: u32 = 0;
    var fail: u32 = 0;
    var skipped: u32 = 0;

    for (corpora) |corpus| {
        // Resolve the corpus once per outer loop and let every direction
        // share the bytes. The cache makes this O(1) after the first hit;
        // pulling it up here also lets us turn a real IO error into a
        // visible test failure instead of a swallowed skip.
        const corpus_bytes: ?[]const u8 = switch (loadCorpus(allocator, io, corpus)) {
            .ok => |b| b,
            .missing => null,
            .io_error => |err| {
                std.debug.print(
                    "  corpus IO error: path={s} err={s}\n",
                    .{ corpus, @errorName(err) },
                );
                // Cluster F fix (F050): an IO error that is NOT
                // FileNotFound is a real bug, not "corpus not present
                // on dev box". Make it loud.
                try testing.expect(false);
                return;
            },
        };

        for (levels) |lvl| {
            for (directions) |dir| {
                const cell: Cell = .{
                    .corpus = corpus,
                    .level = lvl,
                    .direction = dir,
                    .tier = tier,
                };
                const r = if (corpus_bytes == null)
                    CellResult.skipped_device_mismatch
                else
                    runCell(allocator, io, cell, backends, corpus_bytes.?);
                switch (r) {
                    .pass => pass += 1,
                    .fail => fail += 1,
                    .skipped_unimplemented, .skipped_device_mismatch => skipped += 1,
                }
            }
        }
    }

    std.debug.print(
        "conformance: tier={s} {d} pass / {d} fail / {d} skipped\n",
        .{ tier, pass, fail, skipped },
    );
    try testing.expectEqual(@as(u32, 0), fail);
}
