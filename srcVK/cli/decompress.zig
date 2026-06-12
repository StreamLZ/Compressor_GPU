//! 1:1 port of src/cli/decompress.zig.
//!
//! `streamlz_vk -d <file>` handler. Reads a .slz frame, runs the GPU
//! decoder, writes the recovered bytes.

const std = @import("std");
const util = @import("util.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
const gpu_dec_driver = @import("../decode/driver.zig");
const dec_module_loader = @import("../decode/module_loader.zig");
const frame = @import("../format/frame_format.zig");
const mmap_helpers = @import("../mmap.zig");

/// CUDA reference: src/cli/decompress.zig:11-end. -d mode entry point.
pub fn run(allocator: std.mem.Allocator, io: std.Io, w: *std.Io.Writer, args: util.Args) !void {
    const in_path = util.requireInput(args, w);

    const in_file = std.Io.Dir.cwd().openFile(io, in_path, .{}) catch |err| {
        try w.print("error: cannot open '{s}': {s}\n", .{ in_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer in_file.close(io);

    const in_size = in_file.length(io) catch |err| {
        try w.print("error: cannot stat '{s}': {s}\n", .{ in_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    if (in_size == 0) {
        try w.writeAll("error: input file is empty\n");
        try w.flush();
        std.process.exit(1);
    }

    var in_map = mmap_helpers.mapFileRead(in_file, in_size) orelse {
        try w.writeAll("error: cannot memory-map input file\n");
        try w.flush();
        std.process.exit(1);
    };
    defer in_map.unmap();
    const src = in_map.sliceConst();

    const hdr = frame.parseHeader(src) catch |err| {
        try w.print("error: not a valid SLZ1 frame: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    // v4 #16 (CUDA-mirror): supply/verify a custom dictionary before decoding.
    util.supplyDecodeDictionary(allocator, io, args.dictionary, hdr.dictionary_id, w, &gpu_dec_driver.g_default);

    const content_size: usize = if (hdr.content_size) |cs| blk: {
        if (cs > decoder.max_content_size) {
            try w.print("error: frame claims {d} bytes uncompressed, exceeds {d}\n", .{ cs, decoder.max_content_size });
            try w.flush();
            std.process.exit(1);
        }
        break :blk @intCast(cs);
    } else {
        try w.writeAll("error: frame has no content size; streaming mode unsupported\n");
        try w.flush();
        std.process.exit(1);
    };

    const out_size = content_size + decoder.safe_space;
    const derived = if (args.output == null) try util.deriveDecompressOutput(allocator, in_path) else null;
    defer if (derived) |d| allocator.free(d);
    const out_path = args.output orelse derived.?;

    // VK adaptation (2026-06-08, Intel iGPU file-truncate fix): the GPU
    // decoder's VK_EXT_external_memory_host fast path imports the dst
    // pointer as a VkDeviceMemory wrapping the user-supplied host pages.
    // On NVIDIA discrete, vkFreeMemory synchronously drops the kernel-mode
    // page lock — so the iter-8 sequence (mmap the output file, decode
    // into the mmap, release imports, unmap, SetEndOfFile) works.
    // On Intel iGPU's WDDM driver the lock is held past vkFreeMemory and
    // is not released until process exit; SetEndOfFile then fails with
    // AccessDenied (the file's clusters are still "in use" by the
    // kernel-mode allocation). Verified empirically: vkDeviceWaitIdle
    // before/after vkFreeMemory, close+reopen of the file handle, and
    // explicit cache eviction all still fail on Intel.
    //
    // Fix: decode into a plain heap-allocated buffer (page-allocator
    // pages — anonymous, not file-backed) and write the result to disk
    // via the regular file-write path. The import still happens against
    // the heap buffer, but the deferred-release lock there only affects
    // process-lifetime memory accounting, not any file's truncate
    // operation. This mirrors the encoder's CLI shape, which has always
    // decode-into-allocHost-then-write because compressBound is an upper
    // bound rather than the exact written size.
    //
    // Cost: one extra ~content_size memcpy at the end (host->disk).
    // For 100 MB enwik8 on Intel iGPU this is <50 ms — comfortably
    // below the GPU decode time and well within the CLI's perf budget
    // (CLI output is not a hot path; decode bench `-db` runs are
    // unaffected since they use the in-memory decoder directly).
    const out_buf = allocator.alloc(u8, out_size) catch |err| {
        try w.print("error: cannot allocate decode buffer: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    defer allocator.free(out_buf);

    var out_file = std.Io.Dir.cwd().createFile(io, out_path, .{ .read = true }) catch |err| {
        try w.print("error: cannot create '{s}': {s}\n", .{ out_path, @errorName(err) });
        try w.flush();
        std.process.exit(1);
    };
    defer out_file.close(io);

    const result = decoder.decompressFramedThreaded(allocator, io, src, out_buf, &gpu_dec_driver.g_default) catch |err| {
        // Release any cached imports that referenced the heap buffer
        // before it goes out of scope. Otherwise a subsequent decode in
        // the same process could see a stale (vk_buf, vk_mem) pair
        // pointing at freed memory (same lifecycle invariant the
        // pre-iter-8 mmap path violated).
        dec_module_loader.releaseImportsByHostRange(@ptrCast(out_buf.ptr), out_buf.len);
        try w.print("error: decompression failed: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    dec_module_loader.releaseImportsByHostRange(@ptrCast(out_buf.ptr), out_buf.len);

    out_file.setLength(io, result.written) catch |err| {
        try w.print("error: cannot pre-size output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    var write_buf: [4096]u8 = undefined;
    var out_writer = out_file.writer(io, &write_buf);
    out_writer.interface.writeAll(out_buf[0..result.written]) catch |err| {
        try w.print("error: cannot write output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };
    out_writer.interface.flush() catch |err| {
        try w.print("error: cannot flush output: {s}\n", .{@errorName(err)});
        try w.flush();
        std.process.exit(1);
    };

    try w.print("decompressed {d} -> {d} bytes  ({s} -> {s})\n", .{
        src.len, result.written, in_path, out_path,
    });
}
