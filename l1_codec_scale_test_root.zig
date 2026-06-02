//! Executable root for `zig build vk-l1-scale-test`.
//!
//! Mirror of `l1_codec_test_root.zig` for the scale-test sibling. The
//! Cluster F (F037) slz1 sweep here drags in the `slz1_codec ->
//! wire_format_gpu -> ../src/format/...` import chain, which requires
//! a repo-top root.
//!
//! Forwards `main` to `src_vulkan/l1_codec_scale_test.zig`.

const std = @import("std");
const scale_test = @import("src_vulkan/l1_codec_scale_test.zig");

pub fn main(process_init: std.process.Init) !void {
    return scale_test.main(process_init);
}
