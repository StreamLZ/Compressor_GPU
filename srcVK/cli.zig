//! 1:1 port of src/cli.zig.
//!
//! StreamLZ command-line interface dispatcher. Routes -c / -d / -b / -db
//! / -ba / -i modes to the per-mode handlers under srcVK/cli/.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");

const util = @import("cli/util.zig");
const compress_cmd = @import("cli/compress.zig");
const decompress_cmd = @import("cli/decompress.zig");
const bench_compress_cmd = @import("cli/bench_compress.zig");
const bench_decompress_cmd = @import("cli/bench_decompress.zig");
const bench_all_cmd = @import("cli/bench_all.zig");
const info_cmd = @import("cli/info.zig");

/// CUDA reference: src/cli.zig:38-end. Top-level CLI dispatcher.
pub fn run(process_init: std.process.Init) !void {
    _ = process_init;
    return error.NotYetPorted;
}
