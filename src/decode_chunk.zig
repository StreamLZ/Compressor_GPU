const std = @import("std");
const entropy_dec = @import("decode/entropy/entropy_decoder.zig");
const c = @cImport({ @cInclude("stdio.h"); @cInclude("stdlib.h"); });

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path_c = c.getenv("CHUNK_IN") orelse return;
    const path = std.mem.span(path_c);
    const path_z = try a.dupeZ(u8, path);
    const f = c.fopen(path_z, "rb") orelse return;
    defer _ = c.fclose(f);
    _ = c.fseek(f, 0, c.SEEK_END);
    const sz: usize = @intCast(c.ftell(f));
    _ = c.fseek(f, 0, c.SEEK_SET);
    const buf = try a.alloc(u8, sz);
    _ = c.fread(buf.ptr, 1, sz, f);

    const dst = try a.alloc(u8, 1 << 20);
    const scratch = try a.alloc(u8, 1 << 24);
    const res = entropy_dec.highDecodeBytes(dst.ptr, dst.len, buf, true, scratch.ptr, scratch.ptr + scratch.len) catch |err| {
        std.debug.print("decode err: {s}\n", .{@errorName(err)});
        return;
    };
    std.debug.print("decoded {d} bytes (consumed {d}), file_size={d}\n", .{ res.decoded_size, res.bytes_consumed, sz });
    if (res.bytes_consumed != sz) std.debug.print("UNDER/OVER-CONSUMED\n", .{});
}
