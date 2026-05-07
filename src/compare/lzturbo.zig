//! LZTurbo level-10 codec for benchmark comparison.
//! Compressor reverse-engineered from lzturbo.exe (VTune + ASM matching).
//! Decompressor verified against lzturbo.exe output on enwik8 (100MB, byte-exact).

const std = @import("std");

pub const block_size: usize = 4 * 1024 * 1024;

// ── Compress ──

pub fn compress(dst: []u8, src: []const u8) !usize {
    const hash_bits = 17;
    const shift: u5 = 32 - hash_bits;
    var hash_table: [1 << hash_bits]u32 = @splat(0);
    return compressBlock(src.ptr, src.len, src.ptr, &hash_table, shift, dst.ptr);
}

fn compressBlock(
    src_start: [*]const u8,
    src_len: usize,
    base: [*]const u8,
    hash_table: *[1 << 17]u32,
    shift: u5,
    out: [*]u8,
) usize {
    const src_end: [*]const u8 = src_start + src_len;
    const safe_end: [*]const u8 = if (src_len >= 32) src_end - 32 else src_start;

    var cursor: [*]const u8 = src_start;
    var literal_start: [*]const u8 = src_start;
    var out_cursor: [*]u8 = out;

    while (@intFromPtr(cursor) < @intFromPtr(safe_end)) {
        const bytes: u32 = @bitCast(cursor[0..4].*);
        const cur_pos: u32 = @intCast(@intFromPtr(cursor) - @intFromPtr(base));
        const hash_idx: u32 = (bytes *% 0x75bcd17) >> shift;
        const stored_pos: u32 = hash_table[hash_idx];
        hash_table[hash_idx] = cur_pos;

        const match_addr: [*]const u8 = base + stored_pos;
        if (bytes != @as(u32, @bitCast(match_addr[0..4].*))) {
            const dist: usize = @intFromPtr(cursor) - @intFromPtr(literal_start);
            if (dist < 128) {
                cursor += 1;
            } else {
                const skip = @min((dist >> 7) + 1, 16);
                cursor += skip;
            }
            continue;
        }

        const offset: u32 = cur_pos - stored_pos;
        if (offset < 8 or offset > 0xFFFF) {
            cursor += 1;
            continue;
        }

        var match_len: u32 = 4;
        {
            var fwd: [*]const u8 = cursor + 4;
            var fwd_match: [*]const u8 = match_addr + 4;
            while (@intFromPtr(fwd) + 8 <= @intFromPtr(safe_end)) {
                const a: u64 = @bitCast(fwd[0..8].*);
                const b: u64 = @bitCast(fwd_match[0..8].*);
                const xor = a ^ b;
                if (xor != 0) {
                    match_len += @intCast(@ctz(xor) >> 3);
                    break;
                }
                match_len += 8;
                fwd += 8;
                fwd_match += 8;
            }
        }

        var back_cursor = cursor;
        var back_match = match_addr;
        while (@intFromPtr(back_cursor) > @intFromPtr(literal_start) and
            @intFromPtr(back_match) > @intFromPtr(base))
        {
            const bc: [*]const u8 = @ptrFromInt(@intFromPtr(back_cursor) - 1);
            const bm: [*]const u8 = @ptrFromInt(@intFromPtr(back_match) - 1);
            if (bc[0] != bm[0]) break;
            back_cursor = bc;
            back_match = bm;
            match_len += 1;
        }

        const lit_len: u32 = @intCast(@intFromPtr(back_cursor) - @intFromPtr(literal_start));
        const ml_minus4: u32 = match_len - 4;
        const lit_hi: u8 = @intCast(@min(lit_len, 15));
        const ml_hi: u8 = @intCast(@min(ml_minus4, 15));
        out_cursor[0] = (lit_hi << 4) | ml_hi;
        out_cursor[1] = @intCast(offset & 0xFF);
        out_cursor[2] = @intCast(offset >> 8);
        out_cursor += 3;

        if (ml_minus4 >= 15) {
            out_cursor = writeExtLen(out_cursor, ml_minus4);
        }
        if (lit_len >= 15) {
            out_cursor = writeExtLen(out_cursor, lit_len);
        }

        var li: usize = 0;
        while (li + 16 <= lit_len) : (li += 16) {
            const s: *const [16]u8 = @ptrCast(literal_start + li);
            const d: *[16]u8 = @ptrCast(out_cursor);
            d.* = s.*;
            out_cursor += 16;
        }
        while (li < lit_len) : (li += 1) {
            out_cursor[0] = (literal_start + li)[0];
            out_cursor += 1;
        }

        const match_end: [*]const u8 = back_cursor + match_len;
        if (@intFromPtr(match_end) + 4 <= @intFromPtr(src_end)) {
            const rp1: u32 = @intCast(@intFromPtr(match_end) - 1 - @intFromPtr(base));
            const rh1: u32 = (@as(u32, @bitCast((match_end - 1)[0..4].*)) *% 0x75bcd17) >> shift;
            hash_table[rh1] = rp1;
            const rp2: u32 = @intCast(@intFromPtr(match_end) - 2 - @intFromPtr(base));
            const rh2: u32 = (@as(u32, @bitCast((match_end - 2)[0..4].*)) *% 0x75bcd17) >> shift;
            hash_table[rh2] = rp2;
        }

        cursor = match_end;
        literal_start = match_end;
    }

    // Trailing literals
    const remaining: u32 = @intCast(@intFromPtr(src_end) - @intFromPtr(literal_start));
    if (remaining > 0) {
        const lit_hi: u8 = @intCast(@min(remaining, 15));
        out_cursor[0] = (lit_hi << 4); // match_len=0
        out_cursor[1] = 0; // offset=0 (terminal)
        out_cursor[2] = 0;
        out_cursor += 3;
        if (remaining >= 15) {
            out_cursor = writeExtLen(out_cursor, remaining);
        }
        for (0..remaining) |i| {
            out_cursor[0] = (literal_start + i)[0];
            out_cursor += 1;
        }
    } else {
        out_cursor[0] = 0; // empty terminal
        out_cursor[1] = 0;
        out_cursor[2] = 0;
        out_cursor += 3;
    }

    return @intFromPtr(out_cursor) - @intFromPtr(out);
}

fn writeExtLen(out: [*]u8, value: u32) [*]u8 {
    var p = out;
    if (value < 255) {
        p[0] = @intCast(value);
        return p + 1;
    }
    p[0] = 255;
    const rem = value - 255;
    if (rem < 128) {
        p[1] = @intCast(rem * 2); // bit 0 = 0
        return p + 2;
    }
    if (rem < 0x4000) {
        const v: u16 = @intCast(rem * 4 + 1); // bit 0 = 1, bit 1 = 0
        @as(*align(1) u16, @ptrCast(p + 1)).* = v;
        return p + 3;
    }
    if (rem < 0x200000) {
        const v: u16 = @intCast(@as(u32, @truncate(rem * 8 + 3))); // bits 0-1 = 1, bit 2 = 0
        @as(*align(1) u16, @ptrCast(p + 1)).* = v;
        p[3] = @intCast(rem >> 13);
        return p + 4;
    }
    if (rem < 0x10000000) {
        const v: u32 = rem * 16 + 7; // bits 0-2 = 1, bit 3 = 0
        @as(*align(1) u32, @ptrCast(p + 1)).* = v;
        return p + 5;
    }
    const v: u32 = @truncate(rem * 16 + 15); // bits 0-3 = 1
    @as(*align(1) u32, @ptrCast(p + 1)).* = v;
    p[5] = @intCast(rem >> 28);
    return p + 6;
}

// ── Decompress ──

inline fn readExtLen(src: *[*]const u8) u32 {
    const first = src.*[0];
    if (first != 255) {
        src.* += 1;
        return first;
    }
    const b = src.*[1];
    if ((b & 1) == 0) {
        src.* += 2;
        return (@as(u32, b) >> 1) + 255;
    }
    if ((b & 2) == 0) {
        const v: u16 = @as(*align(1) const u16, @ptrCast(src.*[1..3])).*;
        src.* += 3;
        return (@as(u32, v) >> 2) + 255;
    }
    if ((b & 4) == 0) {
        const w: u16 = @as(*align(1) const u16, @ptrCast(src.*[1..3])).*;
        const hi: u32 = src.*[3];
        src.* += 4;
        return ((@as(u32, w) >> 3) | (hi << 13)) + 255;
    }
    if ((b & 8) == 0) {
        const v: u32 = @as(*align(1) const u32, @ptrCast(src.*[1..5])).*;
        src.* += 5;
        return (v >> 4) + 255;
    }
    const lo: u32 = @as(*align(1) const u32, @ptrCast(src.*[1..5])).*;
    const hi: u32 = src.*[5];
    src.* += 6;
    return ((lo >> 4) | (hi << 28)) + 255;
}

inline fn copy16(dst: [*]u8, src: [*]const u8) void {
    const s: *const [16]u8 = @ptrCast(src);
    const d: *[16]u8 = @ptrCast(dst);
    d.* = s.*;
}

pub fn decompress(dst: []u8, src: []const u8, original_size: usize) !usize {
    _ = original_size;
    const overlap_tbl = [8]u8{ 0, 0, 1, 1, 3, 2, 1, 0 };
    var sp = src.ptr;
    var dp = dst.ptr;

    while (true) {
        const token: u32 = sp[0];
        const offset: u16 = @as(*align(1) const u16, @ptrCast(sp[1..3])).*;
        var match_len: u32 = token & 0xF;
        var lit_len: u32 = token >> 4;
        sp += 3;

        if (match_len == 15) {
            match_len = readExtLen(&sp);
        }

        if (lit_len == 15) {
            lit_len = readExtLen(&sp);
            const lit_end = dp + lit_len;
            while (true) {
                copy16(dp, sp);
                dp += 16;
                sp += 16;
                if (@intFromPtr(dp) >= @intFromPtr(lit_end)) break;
            }
            const overshoot = @intFromPtr(dp) - @intFromPtr(lit_end);
            sp -= overshoot;
            dp = lit_end;
        } else {
            copy16(dp, sp);
            sp += lit_len;
            dp += lit_len;
        }

        const total_match: usize = @as(usize, match_len) + 4;
        const match_src: [*]const u8 = @ptrFromInt(@intFromPtr(dp) -% @as(usize, offset));
        const match_end = dp + total_match;

        if (@as(u32, offset) > 7) {
            var d = dp;
            var ms = match_src;
            while (true) {
                copy16(d, ms);
                d += 16;
                ms += 16;
                if (@intFromPtr(d) >= @intFromPtr(match_end)) break;
            }
        } else if (offset == 0) {
            return @intFromPtr(dp) - @intFromPtr(dst.ptr);
        } else {
            dp[0] = match_src[0];
            dp[1] = match_src[1];
            dp[2] = match_src[2];
            dp[3] = match_src[3];
            dp[4] = match_src[4];
            dp[5] = match_src[5];
            dp[6] = match_src[6];
            const stride: usize = overlap_tbl[@as(usize, offset)];
            var d = dp + 7;
            var ms: [*]const u8 = match_src + stride;
            while (true) {
                copy16(d, ms);
                d += 16;
                ms += 16;
                if (@intFromPtr(d) >= @intFromPtr(match_end)) break;
            }
        }

        dp = match_end;
    }
}

pub fn compressBound(src_size: usize) usize {
    return src_size + (src_size / 255) + 16 + 3;
}

// ── Multi-threaded block compress/decompress ──

const MtShared = struct {
    src: []const u8,
    comp_bufs: [][]u8,
    comp_sizes: []usize,
    num_blocks: usize,
    next_block: std.atomic.Value(usize),
    error_flag: std.atomic.Value(u32),
};

fn mtCompressWorker(shared: *MtShared) void {
    while (true) {
        const idx = shared.next_block.fetchAdd(1, .monotonic);
        if (idx >= shared.num_blocks) return;
        if (shared.error_flag.load(.acquire) != 0) return;

        const src_off = idx * block_size;
        const blk_len = @min(shared.src.len - src_off, block_size);
        const blk_src = shared.src[src_off..][0..blk_len];

        const ret = compress(shared.comp_bufs[idx], blk_src) catch {
            shared.error_flag.store(1, .release);
            return;
        };
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

        _ = decompress(shared.dst[dst_off..][0..blk_len], shared.comp_bufs[idx][0..shared.comp_sizes[idx]], blk_len) catch {
            shared.error_flag.store(1, .release);
            return;
        };
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

pub fn compressMt(allocator: std.mem.Allocator, io: std.Io, src: []const u8, num_threads: usize) !MtResult {
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

    if (shared.error_flag.load(.acquire) != 0) return error.LztCompressError;

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

    if (shared.error_flag.load(.acquire) != 0) return error.LztDecompressError;
}
