//! Executable root for `zig build vk-cli-test`.
//!
//! Rooted at the repo top so `src_vulkan/cli_vk_test.zig` resolves
//! cleanly. The test doesn't itself need any `../src/...` imports
//! (it just spawns sub-processes), but the consistent root pattern
//! keeps the build wiring uniform.

const std = @import("std");
const t = @import("src_vulkan/cli_vk_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return t.main(process_init);
}
