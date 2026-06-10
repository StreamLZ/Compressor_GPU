//! Encode-side per-phase wall-clock profiler.
//!
//! Backport of srcVK's `SLZ_VK_PROFILE_PHASES` QPC accumulators (the
//! tool that located the 238 ms d2h_final bottleneck behind VK's 3.2×
//! encode win). Gated on the `SLZ_PROFILE_PHASES` env var: zero
//! overhead when unset beyond one cached getenv + a branch per
//! checkpoint pair.
//!
//! Usage: callers bracket a phase with
//!   `const t0 = enc_phase.begin();`
//!   ... work ...
//!   `enc_phase.add(.huff_lit, t0);`
//! `begin()` returns 0 when profiling is off and `add` is then a no-op.
//! The CLI (`-b`) calls `reset()` after the warm-up compress and
//! `printAndReset(w)` after the timed one.

const std = @import("std");
const cuda = @import("../decode/cuda_api.zig");

pub const Phase = enum(u8) {
    lz_total,
    huff_lit,
    huff_tok,
    huff_off16,
    asm_device,
    asm_host,
    wrap_d2h,
};

const phase_count = @typeInfo(Phase).@"enum".fields.len;
const phase_labels = [phase_count][]const u8{
    "lz_total (h2d + kernel + d2h)",
    "huff_lit",
    "huff_tok",
    "huff_off16",
    "asm_device (measure + write)",
    "asm_host (frame splice)",
    "wrap_d2h (final frame D2H)",
};

var enabled_checked: bool = false;
var enabled: bool = false;
var ticks: [phase_count]i64 = @splat(0);

pub fn profilingEnabled() bool {
    if (!enabled_checked) {
        enabled_checked = true;
        enabled = std.c.getenv("SLZ_PROFILE_PHASES") != null;
    }
    return enabled;
}

/// Start-of-phase timestamp; 0 when profiling is off.
pub fn begin() i64 {
    if (!profilingEnabled()) return 0;
    return cuda.qpcNow();
}

/// Accumulate the elapsed ticks since `t0` into `phase`. No-op when
/// profiling is off (t0 == 0).
pub fn add(phase: Phase, t0: i64) void {
    if (t0 == 0) return;
    ticks[@intFromEnum(phase)] += cuda.qpcNow() - t0;
}

pub fn reset() void {
    ticks = @splat(0);
}

/// Print accumulated per-phase milliseconds and reset. No output when
/// profiling is off or nothing accumulated.
pub fn printAndReset(w: *std.Io.Writer) !void {
    if (!profilingEnabled()) return;
    var total_ticks: i64 = 0;
    for (ticks) |t| total_ticks += t;
    if (total_ticks == 0) return;
    try w.writeAll("  encode phases (QPC ms):\n");
    inline for (0..phase_count) |i| {
        if (ticks[i] != 0)
            try w.print("    {s:<32} {d:8.3} ms\n", .{ phase_labels[i], cuda.qpcMs(0, ticks[i]) });
    }
    try w.print("    {s:<32} {d:8.3} ms\n", .{ "(sum)", cuda.qpcMs(0, total_ticks) });
    reset();
}
