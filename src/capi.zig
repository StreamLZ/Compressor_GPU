const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const frame = @import("format/frame_format.zig");

const allocator = std.heap.c_allocator;

export fn slz_compress(
    src: [*]const u8,
    src_len: usize,
    dst: [*]u8,
    dst_len: usize,
    level: c_int,
) usize {
    if (src_len == 0 or dst_len == 0) return 0;
    const result = encoder.compressFramed(
        allocator,
        src[0..src_len],
        dst[0..dst_len],
        .{ .level = @intCast(std.math.clamp(level, 1, 11)) },
    ) catch return 0;
    return result;
}

export fn slz_decompress(
    src: [*]const u8,
    src_len: usize,
    dst: [*]u8,
    dst_len: usize,
) usize {
    if (src_len == 0 or dst_len == 0) return 0;
    const result = decoder.decompressFramedParallelThreaded(
        allocator,
        null,
        src[0..src_len],
        dst[0..dst_len],
        0,
    ) catch return 0;
    return result.written;
}

export fn slz_compress_bound(src_len: usize) usize {
    return encoder.compressBound(src_len);
}

export fn slz_content_size(src: [*]const u8, src_len: usize) u64 {
    if (src_len < 14) return 0;
    const hdr = frame.parseHeader(src[0..src_len]) catch return 0;
    return hdr.content_size orelse 0;
}

export fn slz_max_content_size() u64 {
    return decoder.max_content_size;
}

export fn slz_version_string() [*:0]const u8 {
    return "2.0.0";
}
