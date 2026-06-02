//! Executable root for `zig build vk-l1-test`.
//!
//! Rooted at the repo top so the transitive imports the production
//! `slz1_codec` chain pulls in (specifically `wire_format_gpu.zig`'s
//! `../src/format/...` paths) stay inside the module's package
//! boundary. Same package-boundary trick as `tests_root.zig` and
//! `wire_format_test_root.zig`; see those for the rationale.
//!
//! Cluster F (F037) made the test add the slz1 sweep on top of the
//! existing raw-kernel sweep; that's what newly drags in the cross-
//! tree imports through `slz1_codec.zig -> wire_format_gpu.zig`.
//!
//! Forwards `main` to `src_vulkan/l1_codec_test.zig`.

const std = @import("std");
const l1_test = @import("src_vulkan/l1_codec_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return l1_test.main(process_init);
}
