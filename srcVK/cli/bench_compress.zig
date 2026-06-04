//! 1:1 port of src/cli/bench_compress.zig.
//!
//! `streamlz -b <file>` handler. Compress+decompress with verification
//! and per-stage timing.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const util = @import("util.zig");
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_enc_driver = @import("../encode/driver.zig");
const gpu_driver = @import("../decode/driver.zig");
const mmap = @import("../mmap.zig");

/// CUDA reference: src/cli/bench_compress.zig:run. -b mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    _ = allocator;
    _ = io;
    _ = w;
    _ = args;
    return error.NotYetPorted;
}
