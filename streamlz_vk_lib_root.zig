//! Library root for `streamlz_vk.dll` / `streamlz_vk.lib`.
//!
//! Rooted at the repo top so the C ABI module's transitive imports
//! (`src_vulkan/slz1_codec.zig` -> `src_vulkan/wire_format.zig` ->
//! `src/format/...`) all resolve inside the module's package boundary.
//! Same trick as `wire_format_test_root.zig` and `tests_root.zig`.
//!
//! The body is intentionally empty: `pub export fn` declarations in
//! `src_vulkan/streamlz_gpu_vk.zig` propagate into the link product when
//! the module is referenced from this root via a `comptime` use site.

const std = @import("std");

comptime {
    // Force the C ABI module to be analysed so every `pub export fn`
    // declaration gets emitted into the link product. Without this,
    // an unused-import elision could strip the exports.
    _ = @import("src_vulkan/streamlz_gpu_vk.zig");
}
