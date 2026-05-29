const std = @import("std");
const cli = @import("cli.zig");

pub fn main(init: std.process.Init) !void {
    try cli.run(init);
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
    _ = @import("encode/gpu_stream_assembly.zig");
    _ = @import("encode/gpu_roundtrip_tests.zig");
    _ = @import("encode/entropy/byte_histogram.zig");
    _ = @import("encode/entropy/entropy_encoder.zig");
    _ = @import("encode/fast/fast_constants.zig");
}
