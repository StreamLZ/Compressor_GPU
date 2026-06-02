//! Executable root for `zig build vk-l1-d2d-test`. Mirrors the
//! pattern of `c_abi_test_root.zig` — package boundary at the repo
//! root so transitively imported `src/format/...` paths resolve.

const std = @import("std");
const t = @import("src_vulkan/c_abi_d2d_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return t.main(process_init);
}
