const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const frame = @import("format/frame_format.zig");

const allocator = std.heap.c_allocator;

/// C-ABI error codes. Negative values indicate specific errors.
pub const SLZ_ERROR_DST_TOO_SMALL: c_int = -1;
pub const SLZ_ERROR_CORRUPT: c_int = -2;
pub const SLZ_ERROR_OOM: c_int = -3;
pub const SLZ_ERROR_BAD_LEVEL: c_int = -4;
pub const SLZ_ERROR_UNKNOWN: c_int = -5;

fn mapCompressError(err: encoder.CompressError) c_int {
    return switch (err) {
        error.DestinationTooSmall => SLZ_ERROR_DST_TOO_SMALL,
        error.OutOfMemory => SLZ_ERROR_OOM,
        error.BadLevel => SLZ_ERROR_BAD_LEVEL,
        else => SLZ_ERROR_UNKNOWN,
    };
}

fn mapDecompressError(err: decoder.DecompressError) c_int {
    return switch (err) {
        error.OutputTooSmall => SLZ_ERROR_DST_TOO_SMALL,
        error.OutOfMemory => SLZ_ERROR_OOM,
        else => SLZ_ERROR_CORRUPT,
    };
}

/// Compress `src` into `dst` at the given level (1-11).
/// Returns bytes written (>= 0) on success, or a negative SLZ_ERROR_* code on failure.
export fn slz_compress(
    src: [*]const u8,
    src_len: usize,
    dst: [*]u8,
    dst_len: usize,
    level: c_int,
) c_int {
    if (src_len == 0 or dst_len == 0) return 0;
    if (level < 1 or level > 11) return SLZ_ERROR_BAD_LEVEL;
    const result = encoder.compressFramed(
        allocator,
        src[0..src_len],
        dst[0..dst_len],
        .{ .level = @intCast(level) },
    ) catch |err| return mapCompressError(err);
    return @intCast(result);
}

/// Decompress an SLZ1 frame from `src` into `dst`.
/// Returns bytes written (>= 0) on success, or a negative SLZ_ERROR_* code on failure.
export fn slz_decompress(
    src: [*]const u8,
    src_len: usize,
    dst: [*]u8,
    dst_len: usize,
) c_int {
    if (src_len == 0 or dst_len == 0) return 0;
    const result = decoder.decompressFramedParallelThreaded(
        allocator,
        null,
        src[0..src_len],
        dst[0..dst_len],
        0,
    ) catch |err| return mapDecompressError(err);
    return @intCast(result.written);
}

/// Return the worst-case compressed size for a given source length.
export fn slz_compress_bound(src_len: usize) usize {
    return encoder.compressBound(src_len);
}

/// Read the uncompressed content size from a frame header, or 0 if absent/invalid.
export fn slz_content_size(src: [*]const u8, src_len: usize) u64 {
    if (src_len < 14) return 0;
    const hdr = frame.parseHeader(src[0..src_len]) catch return 0;
    return hdr.content_size orelse 0;
}

/// Return the maximum decompressible content size (4 GB cap).
export fn slz_max_content_size() u64 {
    return decoder.max_content_size;
}

/// Return a null-terminated library version string.
export fn slz_version_string() [*:0]const u8 {
    return "2.0.0";
}
