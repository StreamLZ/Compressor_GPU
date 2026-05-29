//! StreamLZ command-line interface. GPU-only.
//!
//! Supported modes:
//!
//!   `streamlz -c <file>`        compress to <file>.slz
//!   `streamlz -d <file>`        decompress to <file> (.slz stripped)
//!   `streamlz -b <file>`        compress + decompress with verification
//!   `streamlz -db <file>`       decompress benchmark only
//!   `streamlz -ba <file>`       sweep all levels, ratio + throughput table
//!   `streamlz -i  <file>`       dump frame header + block list
//!
//! Options:
//!   `-l <1..5>`   level                    `-r <N>`     bench runs
//!   `-o <path>`   output path              `-gpu`       accepted, no-op
//!   `--sc <f>`    sc_group override
//!   `-mem`        print peak process memory at exit
//!
//! The `-gpu` flag is accepted as a no-op for backwards compatibility with
//! older scripts (the GPU is the only backend now). Levels outside 1..5
//! return `error.BadLevel` from the encoder.
//!
//! Per-mode handlers live under `src/cli/`:
//!   compress.zig / decompress.zig / bench_compress.zig /
//!   bench_decompress.zig / bench_all.zig / info.zig
//! Shared plumbing (argument parsing, output-path derivation, file I/O,
//! formatting, process memory query) lives in `src/cli/util.zig`.

const std = @import("std");

const util = @import("cli/util.zig");
const compress_cmd = @import("cli/compress.zig");
const decompress_cmd = @import("cli/decompress.zig");
const bench_compress_cmd = @import("cli/bench_compress.zig");
const bench_decompress_cmd = @import("cli/bench_decompress.zig");
const bench_all_cmd = @import("cli/bench_all.zig");
const info_cmd = @import("cli/info.zig");

pub fn run(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var args_it = try init.minimal.args.iterateAllocator(allocator);
    defer args_it.deinit();
    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(allocator);
    while (args_it.next()) |arg| try args_list.append(allocator, arg);

    var stdout_buf: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout();
    var stdout_writer = stdout_file.writer(io, &stdout_buf);
    const w = &stdout_writer.interface;
    defer w.flush() catch {};

    if (args_list.items.len < 2) {
        try util.printUsage(w);
        return;
    }
    const args = util.parseArgs(args_list.items, w);

    switch (args.mode) {
        .version => try util.printVersion(w),
        .help => try util.printUsage(w),
        .compress => try compress_cmd.run(allocator, io, w, args),
        .decompress => try decompress_cmd.run(allocator, io, w, args),
        .bench => try bench_compress_cmd.run(allocator, io, w, args),
        .bench_decompress => try bench_decompress_cmd.run(allocator, io, w, args),
        .bench_all => try bench_all_cmd.run(allocator, io, w, args),
        .info => try info_cmd.run(allocator, io, w, args),
    }

    if (args.report_mem) {
        const mem = util.getMemInfo();
        try w.print("MEMORY: {d:.0} MB peak commit\n", .{mem.commit_mb});
    }
}
