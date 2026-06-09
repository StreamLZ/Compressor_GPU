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
    // Phase 4: pull the C ABI module into the test build so the
    // CUDA-shaped + `_vk`-suffixed exports get codegen'd and the
    // async/D2D ABI tests below can call them.
    _ = @import("streamlz_gpu.zig");
    // Phase 13 fleshout: 8 NEW test files under srcVK/tests/.
    _ = @import("tests/decoder_unit.zig");
    _ = @import("tests/encoder_unit.zig");
    _ = @import("tests/dispatch_unit.zig");
    _ = @import("tests/kernel_conformance.zig");
    _ = @import("tests/l1_decode_roundtrip.zig");
    _ = @import("tests/l1_encode_roundtrip.zig");
    _ = @import("tests/l2_encode_roundtrip.zig");
    // Phase 2A-decoder iter 4c gap fix: L3 + L4 round-trip coverage.
    // ptest_vk shipped 88/0/0 GREEN after iter 4c yet end-to-end L3/L4
    // decode is broken in two distinct ways (byte-65544 silent
    // corruption + small-input KernelLaunchFailed). The new files below
    // close the coverage gap so the next workflow has a failing signal
    // to debug against. See each file's header for the bug-shape
    // catalogue and per-test diagnostics.
    _ = @import("tests/l3_l4_encode_roundtrip.zig");
    _ = @import("tests/l3_l4_cross_backend.zig");
    _ = @import("tests/cross_backend_roundtrip.zig");
    _ = @import("tests/cli_smoke.zig");
    // VK adaptation A-023: batched LZ dispatch byte-identity regression.
    // Forces the per-batch hash/H2D/D2H/gather path via the test hook on
    // encode_lz.g_force_batch_count_for_test and asserts the result is
    // byte-identical to the single-shot path. See file header.
    _ = @import("tests/a023_batched_lz_dispatch.zig");
    // Phase 2A-decoder iter 4 safety net: isolated kernel conformance
    // test for the Huffman decode chain (huff_build_lut +
    // huff_decode_4stream). See srcVK/tests/huff_decode_conformance.zig
    // header for the bug class this guards against (iter 3 bitbufRefill
    // hi/lo swap, fix at ac6696f).
    _ = @import("tests/huff_decode_conformance.zig");
    // Phase 4: C ABI async + D2D + `_vk`-suffixed surface tests.
    _ = @import("tests/async_d2d_api.zig");
}
