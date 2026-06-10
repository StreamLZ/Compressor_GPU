//! 1:1 port of src/encode/gpu_roundtrip_tests.zig.
//!
//! L1 round-trip test cases: encode via the VK encoder, decode via the
//! VK decoder, byte-compare against the original. Bodies stubbed in
//! this skeleton pass; fleshout adds the real test {} blocks.
//!

const std = @import("std");
const encoder = @import("streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_encoder = @import("driver.zig");
const gpu_driver = @import("../decode/driver.zig");

// Fleshout agent adds the test {} blocks from src/encode/gpu_roundtrip_tests.zig.
