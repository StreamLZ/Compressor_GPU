//! Executable root for `streamlz_vk.exe`.
//!
//! Rooted at the repo top so `src_vulkan/cli_vk.zig` (which imports
//! `src_vulkan/slz1_codec.zig` -> `src_vulkan/wire_format.zig` ->
//! `../src/format/...`) all resolve inside the module's package boundary.
//! Same package-boundary trick as `wire_format_test_root.zig`.

const std = @import("std");
const cli = @import("src_vulkan/cli_vk.zig");

pub fn main(process_init: std.process.Init) !void {
    return cli.main(process_init);
}
