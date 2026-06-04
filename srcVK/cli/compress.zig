//! 1:1 port of src/cli/compress.zig.
//!
//! `streamlz -c <file>` handler. Loads the input, runs the GPU
//! encoder, writes the resulting .slz frame.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const gpu_enc_driver = @import("../encode/driver.zig");
const mmap = @import("../mmap.zig");

/// CUDA reference: src/cli/compress.zig:9-end. -c mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    _ = allocator;
    _ = io;
    _ = w;
    _ = args;
    return error.NotYetPorted;
}
