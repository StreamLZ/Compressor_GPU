//! 1:1 port of src/main.zig.
//!
//! Entry point for the streamlz_vk CLI. Delegates to cli.zig::run, which
//! routes to the per-mode handlers under srcVK/cli/.

const std = @import("std");
const cli = @import("cli.zig");

/// CUDA reference: src/main.zig:4-6. Process entry — delegates to cli.run.
pub fn main(process_init: std.process.Init) !void {
    try cli.run(process_init);
}

test {
    _ = @import("cli.zig");
    _ = @import("mmap.zig");
    _ = @import("format/frame_format.zig");
    _ = @import("format/streamlz_constants.zig");
    _ = @import("format/block_header.zig");
    _ = @import("decode/streamlz_decoder.zig");
    _ = @import("encode/streamlz_encoder.zig");
    _ = @import("encode/fast_framed.zig");
    _ = @import("encode/gpu_roundtrip_tests.zig");
    // Phase 13 fleshout: 8 NEW test files under srcVK/tests/.
    _ = @import("tests/decoder_unit.zig");
    _ = @import("tests/encoder_unit.zig");
    _ = @import("tests/dispatch_unit.zig");
    _ = @import("tests/kernel_conformance.zig");
    _ = @import("tests/l1_decode_roundtrip.zig");
    _ = @import("tests/l1_encode_roundtrip.zig");
    _ = @import("tests/l2_encode_roundtrip.zig");
    _ = @import("tests/cross_backend_roundtrip.zig");
    _ = @import("tests/cli_smoke.zig");
}
