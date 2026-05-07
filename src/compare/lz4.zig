//! LZ4/LZ4HC C library bindings for benchmark comparison.

const std = @import("std");

const c = struct {
    extern fn LZ4_compress_default(src: [*c]const u8, dst: [*c]u8, srcSize: c_int, dstCapacity: c_int) c_int;
    extern fn LZ4_compress_HC(src: [*c]const u8, dst: [*c]u8, srcSize: c_int, dstCapacity: c_int, compressionLevel: c_int) c_int;
    extern fn LZ4_decompress_safe(src: [*c]const u8, dst: [*c]u8, compressedSize: c_int, dstCapacity: c_int) c_int;
    extern fn LZ4_compressBound(inputSize: c_int) c_int;
};

pub const block_size: usize = 4 * 1024 * 1024; // 4 MB, same as LZ4 CLI default

pub fn compress(dst: []u8, src: []const u8) !usize {
    const ret = c.LZ4_compress_default(
        @ptrCast(src.ptr),
        @ptrCast(dst.ptr),
        @intCast(src.len),
        @intCast(dst.len),
    );
    if (ret <= 0) return error.Lz4CompressError;
    return @intCast(ret);
}

pub fn compressHc(dst: []u8, src: []const u8, level: c_int) !usize {
    const ret = c.LZ4_compress_HC(
        @ptrCast(src.ptr),
        @ptrCast(dst.ptr),
        @intCast(src.len),
        @intCast(dst.len),
        level,
    );
    if (ret <= 0) return error.Lz4CompressError;
    return @intCast(ret);
}

pub fn decompress(dst: []u8, src: []const u8, original_size: usize) !usize {
    const ret = c.LZ4_decompress_safe(
        @ptrCast(src.ptr),
        @ptrCast(dst.ptr),
        @intCast(src.len),
        @intCast(original_size),
    );
    if (ret < 0) return error.Lz4DecompressError;
    return @intCast(ret);
}

pub fn compressBound(src_size: usize) usize {
    return @intCast(c.LZ4_compressBound(@intCast(src_size)));
}

// ── Multi-threaded block compress/decompress ──

const MtShared = struct {
    src: []const u8,
    comp_bufs: [][]u8,
    comp_sizes: []usize,
    num_blocks: usize,
    next_block: std.atomic.Value(usize),
    error_flag: std.atomic.Value(u32),
    hc_level: ?c_int,
};

fn mtCompressWorker(shared: *MtShared) void {
    while (true) {
        const idx = shared.next_block.fetchAdd(1, .monotonic);
        if (idx >= shared.num_blocks) return;
        if (shared.error_flag.load(.acquire) != 0) return;

        const src_off = idx * block_size;
        const blk_len = @min(shared.src.len - src_off, block_size);
        const blk_src = shared.src[src_off..][0..blk_len];

        const ret = if (shared.hc_level) |level|
            c.LZ4_compress_HC(
                @ptrCast(blk_src.ptr),
                @ptrCast(shared.comp_bufs[idx].ptr),
                @intCast(blk_len),
                @intCast(shared.comp_bufs[idx].len),
                level,
            )
        else
            c.LZ4_compress_default(
                @ptrCast(blk_src.ptr),
                @ptrCast(shared.comp_bufs[idx].ptr),
                @intCast(blk_len),
                @intCast(shared.comp_bufs[idx].len),
            );

        if (ret <= 0) {
            shared.error_flag.store(1, .release);
            return;
        }
        shared.comp_sizes[idx] = @intCast(ret);
    }
}

const MtDecompShared = struct {
    src: []const u8,
    comp_bufs: [][]u8,
    comp_sizes: []usize,
    dst: []u8,
    num_blocks: usize,
    next_block: std.atomic.Value(usize),
    error_flag: std.atomic.Value(u32),
};

fn mtDecompressWorker(shared: *MtDecompShared) void {
    while (true) {
        const idx = shared.next_block.fetchAdd(1, .monotonic);
        if (idx >= shared.num_blocks) return;
        if (shared.error_flag.load(.acquire) != 0) return;

        const dst_off = idx * block_size;
        const blk_len = @min(shared.src.len - dst_off, block_size);

        const ret = c.LZ4_decompress_safe(
            @ptrCast(shared.comp_bufs[idx].ptr),
            @ptrCast(shared.dst[dst_off..].ptr),
            @intCast(shared.comp_sizes[idx]),
            @intCast(blk_len),
        );
        if (ret < 0) {
            shared.error_flag.store(1, .release);
            return;
        }
    }
}

pub const MtResult = struct {
    comp_bufs: [][]u8,
    comp_sizes: []usize,
    num_blocks: usize,
    total_compressed: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MtResult) void {
        for (self.comp_bufs) |b| self.allocator.free(b);
        self.allocator.free(self.comp_bufs);
        self.allocator.free(self.comp_sizes);
    }
};

pub fn compressMt(allocator: std.mem.Allocator, io: std.Io, src: []const u8, num_threads: usize, hc_level: ?c_int) !MtResult {
    const num_blocks = (src.len + block_size - 1) / block_size;

    const comp_bufs = try allocator.alloc([]u8, num_blocks);
    for (comp_bufs) |*b| b.* = &[_]u8{};
    errdefer {
        for (comp_bufs) |b| if (b.len != 0) allocator.free(b);
        allocator.free(comp_bufs);
    }
    for (comp_bufs, 0..) |*b, i| {
        const blk_len = @min(src.len - i * block_size, block_size);
        b.* = try allocator.alloc(u8, compressBound(blk_len));
    }

    const comp_sizes = try allocator.alloc(usize, num_blocks);
    errdefer allocator.free(comp_sizes);
    @memset(comp_sizes, 0);

    var shared: MtShared = .{
        .src = src,
        .comp_bufs = comp_bufs,
        .comp_sizes = comp_sizes,
        .num_blocks = num_blocks,
        .next_block = std.atomic.Value(usize).init(0),
        .error_flag = std.atomic.Value(u32).init(0),
        .hc_level = hc_level,
    };

    const worker_count = @min(num_threads, num_blocks);
    if (worker_count <= 1) {
        mtCompressWorker(&shared);
    } else {
        var group: std.Io.Group = .init;
        for (0..worker_count) |_| {
            group.concurrent(io, mtCompressWorker, .{&shared}) catch |err| switch (err) {
                error.ConcurrencyUnavailable => mtCompressWorker(&shared),
            };
        }
        group.await(io) catch {};
    }

    if (shared.error_flag.load(.acquire) != 0) return error.Lz4CompressError;

    var total: usize = 0;
    for (comp_sizes[0..num_blocks]) |s| total += s;

    return .{
        .comp_bufs = comp_bufs,
        .comp_sizes = comp_sizes,
        .num_blocks = num_blocks,
        .total_compressed = total,
        .allocator = allocator,
    };
}

pub fn decompressMt(io: std.Io, src: []const u8, dst: []u8, result: *const MtResult, num_threads: usize) !void {
    var shared: MtDecompShared = .{
        .src = src,
        .comp_bufs = result.comp_bufs,
        .comp_sizes = result.comp_sizes,
        .dst = dst,
        .num_blocks = result.num_blocks,
        .next_block = std.atomic.Value(usize).init(0),
        .error_flag = std.atomic.Value(u32).init(0),
    };

    const worker_count = @min(num_threads, result.num_blocks);
    if (worker_count <= 1) {
        mtDecompressWorker(&shared);
    } else {
        var group: std.Io.Group = .init;
        for (0..worker_count) |_| {
            group.concurrent(io, mtDecompressWorker, .{&shared}) catch |err| switch (err) {
                error.ConcurrencyUnavailable => mtDecompressWorker(&shared),
            };
        }
        group.await(io) catch {};
    }

    if (shared.error_flag.load(.acquire) != 0) return error.Lz4DecompressError;
}
