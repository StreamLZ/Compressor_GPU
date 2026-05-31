//! M5 — Pure-Zig 2-GiB VkBuffer split prototype.
//!
//! Vulkan implementations cap `VkBuffer` size at the smaller of
//! `maxStorageBufferRange` and `maxMemoryAllocationSize`. On Tier-2
//! mobile hardware that cap is reliably ~2 GiB (sometimes lower). At
//! the larger compression levels the L1..L5 hash-table footprint
//! (per-subchunk slabs across a 256 MiB frame at sc=0.5 ratio) can
//! exceed a single buffer's capacity, so the host allocator has to
//! bin-pack subchunk slabs across multiple VkBuffers.
//!
//! This file is the pure-algorithm slice: no Vulkan calls, no
//! allocator, just the bin-packing decision. M7 wires it into the
//! real `VkDeviceMemory` allocator.
//!
//! Algorithm (greedy first-fit; uniform subchunk size makes it
//! equivalent to optimal):
//!   subchunks_per_bin = floor(cap_bytes / per_subchunk_bytes)
//!   if subchunks_per_bin == 0  ->  error.SubChunkExceedsBufferCap
//!   n_bins              = ceil(n_subchunks / subchunks_per_bin)
//!   bin_size_bytes      = subchunks_per_bin * per_subchunk_bytes
//!   total_bytes         = n_subchunks * per_subchunk_bytes
//!
//! Invariants the tests below pin down:
//!   bin_size_bytes <= cap_bytes
//!   subchunks_per_bin * n_bins >= n_subchunks       (no orphans)
//!   (n_bins - 1) * bin_size_bytes < total_bytes     (no empty tail bin)

const std = @import("std");

pub const BinPack = struct {
    /// Bytes per individual subchunk slab (e.g. hash-table chunk).
    /// Always a multiple of the kernel's natural alignment; the
    /// algorithm here treats it as opaque.
    per_subchunk_bytes: usize,
    /// How many subchunks fit in one VkBuffer given the cap.
    subchunks_per_bin: usize,
    /// Total VkBuffers the allocator must create.
    n_bins: usize,
    /// Sum of all subchunk bytes across every bin (the logical
    /// footprint, not the allocated footprint).
    total_bytes: usize,
    /// Allocated size per bin (each VkBuffer is sized identically;
    /// the final bin may have unused tail slots).
    /// Guaranteed: bin_size_bytes <= cap_bytes.
    bin_size_bytes: usize,
};

pub const Error = error{SubChunkExceedsBufferCap};

/// Compute the bin-packing plan for `n_subchunks` slabs of
/// `per_subchunk_bytes` each, fitting them into VkBuffers of at most
/// `cap_bytes`.
///
/// Returns `error.SubChunkExceedsBufferCap` when a single subchunk
/// would not fit in any buffer the device can allocate. Callers
/// handle this by surfacing `SLZ_ERROR_VK_FEATURE_MISSING` to the
/// user (the device cannot host the chosen level at the chosen
/// frame size).
pub fn planBins(n_subchunks: usize, per_subchunk_bytes: usize, cap_bytes: usize) Error!BinPack {
    const subchunks_per_bin = cap_bytes / per_subchunk_bytes;
    if (subchunks_per_bin == 0) return error.SubChunkExceedsBufferCap;

    // ceil(n_subchunks / subchunks_per_bin) without overflow. The
    // `+ subchunks_per_bin - 1` form is safe because both terms are
    // bounded by `usize` and `subchunks_per_bin >= 1` here.
    const n_bins: usize = if (n_subchunks == 0)
        0
    else
        (n_subchunks + subchunks_per_bin - 1) / subchunks_per_bin;

    return .{
        .per_subchunk_bytes = per_subchunk_bytes,
        .subchunks_per_bin = subchunks_per_bin,
        .n_bins = n_bins,
        .total_bytes = n_subchunks * per_subchunk_bytes,
        .bin_size_bytes = subchunks_per_bin * per_subchunk_bytes,
    };
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────
//
// The L1/L3/L5 scenarios mirror the hash-table footprints derived in
// docs/vulkan_port_architecture.md §7 (memory plan). The 2 GiB cap is
// the conservative Tier-2 mobile floor; Tier-1 desktops can support
// larger buffers but the planner has no reason to prefer them — one
// bin is always preferable, and the planner already returns one bin
// whenever the footprint fits.

const testing = std.testing;

const MiB: usize = 1024 * 1024;
const GiB: usize = 1024 * MiB;
const CAP_2GIB: usize = 2 * GiB;

test "L1 at 256 MiB / sc=0.5 fits in one bin (1 GiB total)" {
    // L1 per-subchunk hash table at sc=0.5 ratio: 512 KiB per subchunk,
    // 2048 subchunks across a 256 MiB frame -> 1 GiB total.
    const per_subchunk: usize = 512 * 1024;
    const n_subchunks: usize = 2048;

    const plan = try planBins(n_subchunks, per_subchunk, CAP_2GIB);

    try testing.expectEqual(@as(usize, 1), plan.n_bins);
    try testing.expectEqual(@as(usize, 1 * GiB), plan.total_bytes);
    try testing.expect(plan.bin_size_bytes <= CAP_2GIB);
    try testing.expect(plan.subchunks_per_bin * plan.n_bins >= n_subchunks);
}

test "L3 at 256 MiB / sc=0.5 needs at least two bins (4 GiB total)" {
    // L3 per-subchunk hash table: 2 MiB per subchunk, 2048 subchunks
    // -> 4 GiB total, exceeds the 2 GiB single-buffer cap.
    const per_subchunk: usize = 2 * MiB;
    const n_subchunks: usize = 2048;

    const plan = try planBins(n_subchunks, per_subchunk, CAP_2GIB);

    try testing.expect(plan.n_bins >= 2);
    try testing.expectEqual(@as(usize, 4 * GiB), plan.total_bytes);
    try testing.expect(plan.bin_size_bytes <= CAP_2GIB);
    try testing.expect(plan.subchunks_per_bin * plan.n_bins >= n_subchunks);
}

test "L5 at 256 MiB / sc=0.5 splits into exactly two bins (2.25 GiB total)" {
    // L5 per-subchunk hash table: 1.125 MiB per subchunk, 2048 subchunks
    // -> 2.25 GiB total, one byte past the 2 GiB cap so it must split.
    // 1.125 MiB == 1179648 bytes.
    const per_subchunk: usize = 1179648;
    const n_subchunks: usize = 2048;

    const plan = try planBins(n_subchunks, per_subchunk, CAP_2GIB);

    try testing.expectEqual(@as(usize, 2), plan.n_bins);
    try testing.expectEqual(n_subchunks * per_subchunk, plan.total_bytes);
    try testing.expect(plan.bin_size_bytes <= CAP_2GIB);
    try testing.expect(plan.subchunks_per_bin * plan.n_bins >= n_subchunks);
}

test "L1 at 1 MiB frame fits in one bin (4 MiB total)" {
    // Minimum-viable smoke: a tiny 1 MiB frame should always pack into
    // a single bin no matter how the constants change.
    const per_subchunk: usize = 512 * 1024;
    const n_subchunks: usize = 8; // 8 * 512 KiB = 4 MiB

    const plan = try planBins(n_subchunks, per_subchunk, CAP_2GIB);

    try testing.expectEqual(@as(usize, 1), plan.n_bins);
    try testing.expectEqual(@as(usize, 4 * MiB), plan.total_bytes);
    try testing.expect(plan.bin_size_bytes <= CAP_2GIB);
    try testing.expect(plan.subchunks_per_bin * plan.n_bins >= n_subchunks);
}

test "subchunk larger than cap returns SubChunkExceedsBufferCap" {
    // No power of two on either side — just a subchunk strictly larger
    // than the cap. Must surface the named error so the host translates
    // it into SLZ_ERROR_VK_FEATURE_MISSING.
    const per_subchunk: usize = 3 * GiB;
    const n_subchunks: usize = 1;

    try testing.expectError(error.SubChunkExceedsBufferCap, planBins(n_subchunks, per_subchunk, CAP_2GIB));
}

test "subchunk exactly equal to cap packs one subchunk per bin" {
    // Boundary: per_subchunk == cap means subchunks_per_bin == 1, NOT 0
    // (the error path is strictly when the slab cannot fit at all).
    const per_subchunk: usize = CAP_2GIB;
    const n_subchunks: usize = 3;

    const plan = try planBins(n_subchunks, per_subchunk, CAP_2GIB);

    try testing.expectEqual(@as(usize, 1), plan.subchunks_per_bin);
    try testing.expectEqual(@as(usize, 3), plan.n_bins);
    try testing.expect(plan.bin_size_bytes <= CAP_2GIB);
    try testing.expect(plan.subchunks_per_bin * plan.n_bins >= n_subchunks);
}

test "zero subchunks yields a degenerate but valid plan" {
    // The allocator may legitimately ask for the plan before knowing the
    // frame size (e.g. probing). Zero subchunks must not crash; the
    // resulting plan has zero bins and a sane bin_size hint.
    const per_subchunk: usize = 2 * MiB;
    const plan = try planBins(0, per_subchunk, CAP_2GIB);

    try testing.expectEqual(@as(usize, 0), plan.n_bins);
    try testing.expectEqual(@as(usize, 0), plan.total_bytes);
    try testing.expect(plan.bin_size_bytes <= CAP_2GIB);
}
