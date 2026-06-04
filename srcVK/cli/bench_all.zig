//! 1:1 port of src/cli/bench_all.zig.
//!
//! `streamlz -ba <file>` handler. Sweeps encoder levels 1..5 emitting a
//! ratio+throughput table. Levels 3..5 fall back to the level-2 codec
//! on the VK port until the Huffman/chain L2 stubs land.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_enc_driver = @import("../encode/driver.zig");
const gpu_driver = @import("../decode/driver.zig");
const mmap = @import("../mmap.zig");

/// CUDA reference: src/cli/bench_all.zig:run. -ba mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    _ = allocator;
    _ = io;
    _ = w;
    _ = args;
    return error.NotYetPorted;
}
