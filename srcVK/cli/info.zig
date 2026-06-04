//! 1:1 port of src/cli/info.zig.
//!
//! `streamlz -i <file>` handler. Pure format-side frame dumper — prints
//! the parsed frame header and block list. No GPU calls.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const util = @import("util.zig");
const frame_format = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const mmap = @import("../mmap.zig");

/// CUDA reference: src/cli/info.zig:run. -i mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    _ = allocator;
    _ = io;
    _ = w;
    _ = args;
    return error.NotYetPorted;
}
