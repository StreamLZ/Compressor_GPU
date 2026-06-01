//! Executable root for `zig build vk-wire-format-scale-test`.
//!
//! Rooted at the repo top so `src_vulkan/wire_format.zig` (which the
//! test imports transitively) can reach into `src/format/...` via
//! relative `@import` paths. Same package-boundary trick as
//! `wire_format_test_root.zig` and `tests_root.zig`; see those files
//! for the rationale.
//!
//! Forwards `main` to `src_vulkan/wire_format_scale_test.zig`.

const std = @import("std");
const wire_scale = @import("src_vulkan/wire_format_scale_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return wire_scale.main(process_init);
}
