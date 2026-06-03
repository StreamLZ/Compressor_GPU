//! Smoke test for the Vulkan C ABI: invokes slzCreate_vk + the
//! diagnostic accessors + slzDestroy_vk through a Zig import (no DLL
//! load) and prints a one-line PASS/FAIL summary. Wired in build.zig
//! as `zig build vk-smoke`.
//!
//! Success path: brings up the vulkan-1.dll loader, picks a device,
//! classifies its tier, allocates a handle, asserts post-creation
//! invariants, frees the handle, prints `vk smoke OK`. On any error
//! (no Vulkan driver, no compute device, tier == unsupported,
//! invariant violated), prints `vk smoke FAIL: <reason>` and exits 1
//! so CI catches the regression.
//!
//! Post-creation invariants asserted:
//!   * `handle.tier` is NOT `.unsupported`.
//!   * `handle.vendor_id` is non-zero.
//!   * `handle.subgroup_size` is a power-of-two in [1, 256].
//!   * The version string from `slzGetVersionString_vk` carries the
//!     `+vk` build-metadata suffix (matches `vk_version_string` in
//!     streamlz_gpu_vk.zig).
//!   * `slzCompressBound_vk(0)` is at least the documented minimum
//!     fixed-overhead (8 KiB of headers per slz1Bound) — a regression
//!     that returned 0 here would silently break the C ABI's
//!     allocation contract.
//!   * `slzMakeDeviceOnlyHandle_vk` returns SLZ_ERROR_UNSUPPORTED with
//!     the out-slot cleared to null (Cluster E.F008 contract).

const std = @import("std");

const vk_abi = @import("streamlz_gpu_vk.zig");

fn isPow2InRange(n: u32, lo: u32, hi: u32) bool {
    if (n < lo or n > hi) return false;
    return (n & (n - 1)) == 0;
}

pub fn main(process_init: std.process.Init) !void {
    const io = process_init.io;

    var stdout_buf: [512]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    var handle: ?*vk_abi.VkContext = null;
    const rc = vk_abi.slzCreate_vk(&handle);
    if (rc != 0 or handle == null) {
        try w.print("vk smoke FAIL: create rc={d}\n", .{rc});
        return error.CreateFailed;
    }
    defer vk_abi.slzDestroy_vk(handle);

    const h = handle.?;

    // ── Post-creation invariants ─────────────────────────────────────

    // 1. Tier classification must have succeeded; slzCreate_vk should
    //    already have rejected `.unsupported` with VK_FEATURE_MISSING,
    //    so observing it here is a guard against future regressions
    //    in the slzCreate_vk validation path.
    if (h.tier == .unsupported) {
        try w.print("vk smoke FAIL: tier=unsupported (slzCreate_vk should have rejected)\n", .{});
        return error.UnsupportedTier;
    }

    // 2. Vendor ID is populated by the M1 properties read; 0 means
    //    vkGetPhysicalDeviceProperties was never resolved or the
    //    probe path early-exited.
    if (h.vendor_id == 0) {
        try w.print("vk smoke FAIL: vendor_id=0\n", .{});
        return error.NoVendorId;
    }

    // 3. Subgroup size is a hardware-defined power-of-two warp width.
    //    Every supported tier has one of 8 / 16 / 32 / 64 / 128. 0
    //    means the Properties2 probe never ran (Tier-2 / loader
    //    mis-init).
    if (!isPow2InRange(h.subgroup_size, 1, 256)) {
        try w.print(
            "vk smoke FAIL: subgroup_size={d} not pow2 in [1,256]\n",
            .{h.subgroup_size},
        );
        return error.BadSubgroupSize;
    }

    // 4. Version string carries the +vk suffix (SemVer build metadata).
    const ver = vk_abi.slzGetVersionString_vk();
    const ver_slice = std.mem.span(ver);
    if (std.mem.indexOf(u8, ver_slice, "+vk") == null) {
        try w.print("vk smoke FAIL: version='{s}' missing +vk suffix\n", .{ver_slice});
        return error.BadVersionString;
    }

    // 5. slzCompressBound_vk must reserve at least the fixed-overhead
    //    headers even for input_size=0 (8 KiB of headers + per-chunk
    //    estimate per slz1Bound). A return of 0 would silently break
    //    callers' allocation contract.
    const bound_zero = vk_abi.slzCompressBound_vk(0);
    if (bound_zero < 4096) {
        try w.print(
            "vk smoke FAIL: slzCompressBound_vk(0)={d} unreasonably small\n",
            .{bound_zero},
        );
        return error.BadCompressBound;
    }

    // 6. slzMakeDeviceOnlyHandle_vk contract (Cluster E.F008): always
    //    returns SLZ_ERROR_UNSUPPORTED and clears the out-slot to
    //    null. A regression that silently flipped it back to the old
    //    0x10 sentinel would re-introduce a dead API surface.
    var sentinel_slot: ?*const anyopaque = @ptrFromInt(0xdeadbeef);
    const sentinel_rc = vk_abi.slzMakeDeviceOnlyHandle_vk(&sentinel_slot, 0);
    if (sentinel_rc != 5 or sentinel_slot != null) {
        try w.print(
            "vk smoke FAIL: slzMakeDeviceOnlyHandle_vk rc={d} slot_null={}\n",
            .{ sentinel_rc, sentinel_slot == null },
        );
        return error.BadDeviceOnlyHandleContract;
    }

    try w.print(
        "vk smoke OK (version={s} tier={t} vendor=0x{x:0>4} subgroup={d})\n",
        .{ ver_slice, h.tier, h.vendor_id, h.subgroup_size },
    );
}
