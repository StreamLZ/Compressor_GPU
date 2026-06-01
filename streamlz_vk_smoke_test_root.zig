//! Executable root for `zig build vk-smoke`.
//!
//! Rooted at the repo top so `src_vulkan/streamlz_gpu_vk.zig` (which the
//! smoke test imports) can transitively reach `src/format/...` through
//! `src_vulkan/wire_format.zig`. Same package-boundary trick as
//! `wire_format_test_root.zig`.

const std = @import("std");
const smoke = @import("src_vulkan/streamlz_gpu_vk_smoke_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return smoke.main(process_init);
}
