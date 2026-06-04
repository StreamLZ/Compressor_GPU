//! 1:1 port of src/cli/bench_decompress.zig.
//!
//! `streamlz -db <file>` handler. Decompress-only benchmark on a .slz
//! input.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const util = @import("util.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_driver = @import("../decode/driver.zig");
const mmap = @import("../mmap.zig");

/// CUDA reference: src/cli/bench_decompress.zig:run. -db mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    _ = allocator;
    _ = io;
    _ = w;
    _ = args;
    return error.NotYetPorted;
}
