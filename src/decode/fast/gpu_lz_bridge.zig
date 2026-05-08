const std = @import("std");
const fast_dec = @import("fast_lz_decoder.zig");
const constants = @import("../../format/streamlz_constants.zig");

/// C-ABI struct matching the CUDA kernel's expected inputs.
/// Passed to slz_gpu_process_lz_runs() which copies to device and launches kernel.
const GpuLzRunsDesc = extern struct {
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
    dst: [*]u8,
    dst_size: u32,
    initial_copy: u32,
    cmd_stream2_offset: u32,
    base_offset: u32,
    dst_offset: u32,
};

/// Extern CUDA host function — implemented in gpu_lz_kernel.cu
extern fn slz_gpu_process_lz_runs(desc: *const GpuLzRunsDesc) c_int;
extern fn slz_gpu_available() c_int;

pub fn isAvailable() bool {
    return slz_gpu_available() != 0;
}

/// Drop-in replacement for processLzRuns that runs on GPU.
/// Called from decodeChunk after readLzTable succeeds.
pub fn processLzRunsGpu(
    mode: u32,
    src_end: [*]const u8,
    dst_in: [*]u8,
    dst_size: usize,
    base_offset: u64,
    lz: *fast_dec.FastLzTable,
) fast_dec.DecodeError!void {
    _ = mode; // GPU kernel handles both modes

    const desc = GpuLzRunsDesc{
        .cmd_data = lz.cmd_start,
        .cmd_size = @intCast(@intFromPtr(lz.cmd_end) - @intFromPtr(lz.cmd_start)),
        .lit_data = lz.lit_start,
        .lit_size = @intCast(@intFromPtr(lz.lit_end) - @intFromPtr(lz.lit_start)),
        .off16_data = @ptrCast(lz.off16_start),
        .off16_count = @intCast((@intFromPtr(lz.off16_end) - @intFromPtr(lz.off16_start)) / 2),
        .off32_data1 = @ptrCast(lz.off32_backing1),
        .off32_count1 = lz.off32_count1,
        .off32_data2 = @ptrCast(lz.off32_backing2),
        .off32_count2 = lz.off32_count2,
        .length_data = lz.length_stream,
        .length_avail = @intCast(@intFromPtr(src_end) - @intFromPtr(lz.length_stream)),
        .dst = dst_in,
        .dst_size = @intCast(dst_size),
        .initial_copy = if (base_offset == 0) 8 else 0,
        .cmd_stream2_offset = lz.cmd_stream2_offset,
        .base_offset = @intCast(base_offset),
        .dst_offset = @intCast(@intFromPtr(dst_in) - @intFromPtr(dst_in) + base_offset), // TODO: need full output base
    };

    const ret = slz_gpu_process_lz_runs(&desc);
    if (ret != 0) return error.BadMode;
}
