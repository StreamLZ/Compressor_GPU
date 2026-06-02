//! Executable root for `zig build vk-abi-async-test`. Same shape as
//! `c_abi_test_root.zig`: package boundary at the repo root so the
//! Vulkan-side `src/format/...` reaches through `wire_format.zig`.

const std = @import("std");
const t = @import("src_vulkan/c_abi_async_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return t.main(process_init);
}
