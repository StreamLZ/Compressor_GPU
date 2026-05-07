//! Zstd C library bindings for benchmark comparison.

const std = @import("std");
const c = @cImport({
    @cInclude("zstd.h");
});

pub const block_size: usize = 4 * 1024 * 1024; // 4 MB independent blocks

pub fn compress(dst: []u8, src: []const u8, level: c_int) !usize {
    const ret = c.ZSTD_compress(dst.ptr, dst.len, src.ptr, src.len, level);
    if (c.ZSTD_isError(ret) != 0) return error.ZstdCompressError;
    return ret;
}

pub fn compressMt(dst: []u8, src: []const u8, level: c_int, threads: c_int) !usize {
    const cctx = c.ZSTD_createCCtx() orelse return error.ZstdAllocError;
    defer _ = c.ZSTD_freeCCtx(cctx);
    if (c.ZSTD_isError(c.ZSTD_CCtx_setParameter(cctx, c.ZSTD_c_compressionLevel, level)) != 0)
        return error.ZstdParameterError;
    if (c.ZSTD_isError(c.ZSTD_CCtx_setParameter(cctx, c.ZSTD_c_nbWorkers, threads)) != 0)
        return error.ZstdParameterError;
    const ret = c.ZSTD_compress2(cctx, dst.ptr, dst.len, src.ptr, src.len);
    if (c.ZSTD_isError(ret) != 0) return error.ZstdCompressError;
    return ret;
}

pub fn decompress(dst: []u8, src: []const u8) !usize {
    const ret = c.ZSTD_decompress(dst.ptr, dst.len, src.ptr, src.len);
    if (c.ZSTD_isError(ret) != 0) return error.ZstdDecompressError;
    return ret;
}

pub fn compressBound(src_size: usize) usize {
    return c.ZSTD_compressBound(src_size);
}

// ── Multi-threaded block compress/decompress (4 MB independent blocks) ──

const MtShared = struct {
    src: []const u8,
    comp_bufs: [][]u8,
    comp_sizes: []usize,
    num_blocks: usize,
    next_block: std.atomic.Value(usize),
    error_flag: std.atomic.Value(u32),
    level: c_int,
};

fn mtCompressWorker(shared: *MtShared) void {
    while (true) {
        const idx = shared.next_block.fetchAdd(1, .monotonic);
        if (idx >= shared.num_blocks) return;
        if (shared.error_flag.load(.acquire) != 0) return;

        const src_off = idx * block_size;
        const blk_len = @min(shared.src.len - src_off, block_size);

        const ret = c.ZSTD_compress(
            shared.comp_bufs[idx].ptr,
            shared.comp_bufs[idx].len,
            shared.src.ptr + src_off,
            blk_len,
            shared.level,
        );
        if (c.ZSTD_isError(ret) != 0) {
            shared.error_flag.store(1, .release);
            return;
        }
        shared.comp_sizes[idx] = ret;
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

        const ret = c.ZSTD_decompress(
            shared.dst.ptr + dst_off,
            blk_len,
            shared.comp_bufs[idx].ptr,
            shared.comp_sizes[idx],
        );
        if (c.ZSTD_isError(ret) != 0) {
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

pub fn compressBlocksMt(allocator: std.mem.Allocator, io: std.Io, src: []const u8, num_threads: usize, level: c_int) !MtResult {
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
        .level = level,
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

    if (shared.error_flag.load(.acquire) != 0) return error.ZstdCompressError;

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

pub fn decompressBlocksMt(io: std.Io, src: []const u8, dst: []u8, result: *const MtResult, num_threads: usize) !void {
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

    if (shared.error_flag.load(.acquire) != 0) return error.ZstdDecompressError;
}
