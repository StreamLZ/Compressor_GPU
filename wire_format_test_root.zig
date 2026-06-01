//! Executable root for `zig build vk-wire-format-test`.
//!
//! Rooted at the repo top so `src_vulkan/wire_format.zig` (which the
//! test imports transitively) can reach into `src/format/...` via
//! relative `@import` paths.  Same package-boundary trick as
//! `tests_root.zig` (the M9 conformance harness root); see that file
//! for the rationale.
//!
//! Forwards `main` to `src_vulkan/wire_format_test.zig`.

const std = @import("std");
const wire_test = @import("src_vulkan/wire_format_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return wire_test.main(process_init);
}
