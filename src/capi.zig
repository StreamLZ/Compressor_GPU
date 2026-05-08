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
    if (dst_len == 0) return SLZ_ERROR_DST_TOO_SMALL;
    if (level < 1 or level > 11) return SLZ_ERROR_BAD_LEVEL;
    const result = encoder.compressFramed(
        allocator,
        src[0..src_len],
        dst[0..dst_len],
        .{ .level = @intCast(level) },
    ) catch |err| return mapCompressError(err);
    if (result > std.math.maxInt(c_int)) return SLZ_ERROR_UNKNOWN;
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
    if (src_len == 0) return 0;
    if (dst_len == 0) return SLZ_ERROR_DST_TOO_SMALL;
    const result = decoder.decompressFramedParallelThreaded(
        allocator,
        null,
        src[0..src_len],
        dst[0..dst_len],
        0,
    ) catch |err| return mapDecompressError(err);
    if (result.offset > 0) {
        std.mem.copyForwards(u8, dst[0..result.written], dst[result.offset..][0..result.written]);
    }
    if (result.written > std.math.maxInt(c_int)) return SLZ_ERROR_UNKNOWN;
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

// ── GPU stream extraction API ──

const fast_dec = @import("decode/fast/fast_lz_decoder.zig");
const block_hdr = @import("format/block_header.zig");
const constants = @import("format/streamlz_constants.zig");

/// Extracted stream pointers for GPU kernel consumption.
const GpuStreams = extern struct {
    cmd_data: [*]const u8,
    cmd_size: u32,
    lit_data: [*]const u8,
    lit_size: u32,
    off16_data: [*]const u8,
    off16_count: u32,
    off32_data1: [*]const u8,
    off32_count1: u32,
    off32_data2: [*]const u8,
    off32_count2: u32,
    length_data: [*]const u8,
    length_avail: u32,
    initial_copy: u32,
    decomp_size: u32,
    cmd_stream2_offset: u32,
    base_offset: u32,
    dst_offset: u32,
};

/// Extract L1 GPU-ready streams from a compressed SLZ1 frame.
/// Decompresses entropy framing on CPU, returns raw stream pointers
/// into `scratch` memory. Caller must keep `scratch` alive while using streams.
///
/// `comp` / `comp_len`: compressed SLZ1 frame
/// `scratch` / `scratch_len`: scratch buffer (must be >= 2 * decomp_size + 64KB)
/// `dst`: output buffer where initial 8 bytes are written
/// `out_streams`: array to receive extracted stream info (one per sub-chunk)
/// `max_streams`: capacity of out_streams array
///
/// Returns number of sub-chunks extracted, or negative error code.
export fn slz_extract_gpu_streams(
    comp: [*]const u8,
    comp_len: usize,
    scratch: [*]u8,
    scratch_len: usize,
    dst: [*]u8,
    out_streams: [*]GpuStreams,
    max_streams: c_int,
) c_int {
    const comp_slice = comp[0..comp_len];
    const hdr = frame.parseHeader(comp_slice) catch return -10;
    var pos: usize = hdr.header_size;

    // Block header (8 bytes)
    const blk = frame.parseBlockHeader(comp_slice[pos..]) catch return -11;
    pos += 8;
    if (blk.compressed_size == 0) return 0;

    // Compute SC prefix size (tail bytes at the end of the block)
    var cur_int_hdr = block_hdr.parseBlockHeader(comp_slice[pos..]) catch return -12;
    pos += 2;

    const num_chunks: usize = (blk.decompressed_size + constants.chunk_size - 1) / constants.chunk_size;
    const prefix_size: usize = if (cur_int_hdr.self_contained and num_chunks > 1) (num_chunks - 1) * 8 else 0;
    const block_end: usize = hdr.header_size + 8 + blk.compressed_size - prefix_size;

    var stream_count: c_int = 0;
    var decomp_offset: usize = 0;
    var dst_ptr = dst;
    var chunk_idx: usize = 0;
    var group_base_offset: usize = 0;
    const sc_group_size: usize = hdr.sc_group_size;

    // Walk chunks — new 2-byte internal header at each chunk boundary
    while (pos + 4 <= block_end and
        decomp_offset < blk.decompressed_size and
        stream_count < max_streams)
    {
        // At each chunk boundary, read a new internal block header
        const at_chunk_boundary = (decomp_offset & (constants.chunk_size - 1)) == 0;
        if (at_chunk_boundary and decomp_offset > 0) {
            cur_int_hdr = block_hdr.parseBlockHeader(comp_slice[pos..]) catch break;
            pos += 2;
        }

        // Track SC group boundaries — reset base_offset at group start
        if (cur_int_hdr.self_contained and (chunk_idx % sc_group_size) == 0) {
            group_base_offset = decomp_offset;
        }

        // Uncompressed chunk — raw copy, skip
        if (cur_int_hdr.uncompressed) {
            const chunk_decomp = @min(constants.chunk_size, blk.decompressed_size - decomp_offset);
            pos += chunk_decomp;
            decomp_offset += chunk_decomp;
            dst_ptr += chunk_decomp;
            chunk_idx += 1;
            continue;
        }

        const chunk_parsed = block_hdr.parseChunkHeader(comp_slice[pos..], cur_int_hdr.use_checksums) catch break;
        pos += chunk_parsed.bytes_consumed;

        if (chunk_parsed.is_memset) {
            pos += 1;
            decomp_offset += @min(constants.chunk_size, blk.decompressed_size - decomp_offset);
            continue;
        }

        const chunk_end = pos + chunk_parsed.compressed_size;
        var sc_remaining: usize = @min(constants.chunk_size, blk.decompressed_size - decomp_offset);

        // Walk sub-chunks within chunk
        while (pos + 3 <= chunk_end and sc_remaining > 0 and stream_count < max_streams) {
            const sc_hdr_val: u32 = (@as(u32, comp_slice[pos]) << 16) | (@as(u32, comp_slice[pos + 1]) << 8) | comp_slice[pos + 2];
            const sc_compressed = (sc_hdr_val & constants.chunk_header_compressed_flag) != 0;

            if (!sc_compressed) break;

            const sc_comp_size: usize = sc_hdr_val & 0x7FFFF;
            const sc_mode: u32 = (sc_hdr_val >> constants.sub_chunk_type_shift) & 0xF;
            const sc_decomp: usize = @min(constants.sub_chunk_size, sc_remaining);
            pos += 3;

            if (sc_comp_size < sc_decomp) {
                // Compressed sub-chunk — extract streams via readLzTable
                const scratch_offset: usize = @as(usize, @intCast(stream_count)) * (constants.sub_chunk_size * 2 + 256);
                if (scratch_offset + constants.sub_chunk_size * 2 + 256 > scratch_len) return SLZ_ERROR_OOM;

                const inner_scratch = scratch + scratch_offset + @sizeOf(fast_dec.FastLzTable);
                const inner_scratch_end = scratch + scratch_offset + constants.sub_chunk_size * 2 + 256;
                const lz_ptr: *fast_dec.FastLzTable = @ptrCast(@alignCast(scratch + scratch_offset));

                var dst_slot = dst_ptr;
                const base_offset: i64 = @intCast(decomp_offset - group_base_offset);

                const sc_src_end = comp_slice[pos + sc_comp_size ..].ptr;
                fast_dec.readLzTable(
                    sc_mode,
                    comp_slice[pos..].ptr,
                    sc_src_end,
                    &dst_slot,
                    @intCast(sc_decomp),
                    base_offset,
                    inner_scratch,
                    inner_scratch_end,
                    lz_ptr,
                ) catch break;
                lz_ptr.src_end = sc_src_end;

                const initial: u32 = @intCast(@intFromPtr(dst_slot) - @intFromPtr(dst_ptr));

                out_streams[@intCast(stream_count)] = .{
                    .cmd_data = lz_ptr.cmd_start,
                    .cmd_size = @intCast(@intFromPtr(lz_ptr.cmd_end) - @intFromPtr(lz_ptr.cmd_start)),
                    .lit_data = lz_ptr.lit_start,
                    .lit_size = @intCast(@intFromPtr(lz_ptr.lit_end) - @intFromPtr(lz_ptr.lit_start)),
                    .off16_data = @ptrCast(lz_ptr.off16_start),
                    .off16_count = @intCast((@intFromPtr(lz_ptr.off16_end) - @intFromPtr(lz_ptr.off16_start)) / 2),
                    .off32_data1 = @ptrCast(lz_ptr.off32_backing1),
                    .off32_count1 = lz_ptr.off32_count1,
                    .off32_data2 = @ptrCast(lz_ptr.off32_backing2),
                    .off32_count2 = lz_ptr.off32_count2,
                    .length_data = lz_ptr.length_stream,
                    .length_avail = @intCast(@intFromPtr(lz_ptr.src_end) -| @intFromPtr(lz_ptr.length_stream)),
                    .cmd_stream2_offset = lz_ptr.cmd_stream2_offset,
                    .base_offset = @intCast(decomp_offset - group_base_offset),
                    .dst_offset = @intCast(decomp_offset),
                    .initial_copy = initial,
                    .decomp_size = @intCast(sc_decomp),
                };
                stream_count += 1;
            }

            pos += sc_comp_size;
            decomp_offset += sc_decomp;
            sc_remaining -= sc_decomp;
            dst_ptr += sc_decomp;
        }

        pos = chunk_end;
        chunk_idx += 1;
    }

    return stream_count;
}

