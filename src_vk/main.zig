//! streamlz_vk entry point (foundation wave).
//!
//! VK PORT NOTE: ports src/main.zig. The CLI dispatcher
//! (`src_vk/cli.zig`) is a foundation-wave stub today — it acknowledges
//! the args and exits with a "not yet wired" message so the binary
//! links and can be invoked end-to-end while subsequent waves bring
//! the encode/decode handlers online. The `test {}` block below pulls
//! every foundation module into the test build so `zig build test`
//! exercises them.

const std = @import("std");
const cli = @import("cli.zig");

pub fn main(process_init: std.process.Init) !void {
    try cli.run(process_init);
}

test {
    // Foundation-wave module pull-in. Mirrors src/main.zig's test block
    // — every module that needs to compile clean in isolation is
    // referenced here so `zig build test-vk` (later wave) picks them
    // up. L2 stubs are deliberately excluded; they only typecheck when
    // their imports are wired by the L1 dispatch wave.
    _ = @import("error.zig");
    _ = @import("version.zig");
    _ = @import("mmap.zig");
    _ = @import("vma.zig");
    _ = @import("format/streamlz_constants.zig");
    _ = @import("format/frame_format.zig");
    _ = @import("format/block_header.zig");
    _ = @import("decode/vulkan_api.zig");
    _ = @import("decode/descriptors.zig");
    _ = @import("decode/decode_context.zig");
    _ = @import("decode/module_loader.zig");
    _ = @import("decode/driver.zig");
}
