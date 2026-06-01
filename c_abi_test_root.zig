//! Executable root for `zig build vk-abi-test`.
//!
//! Rooted at the repo top so `src_vulkan/streamlz_gpu_vk.zig` (which
//! `src_vulkan/c_abi_test.zig` imports) can transitively reach
//! `src/format/...` through `src_vulkan/wire_format.zig`. Same package-
//! boundary trick as `wire_format_test_root.zig`.

const std = @import("std");
const t = @import("src_vulkan/c_abi_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return t.main(process_init);
}
