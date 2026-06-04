//! 1:1 port of src/streamlz_gpu.zig.
//!
//! StreamLZ GPU compression library — C ABI implementation. Implements
//! include/streamlz_gpu.h: a handle-based, synchronous GPU-backed
//! compressor exposed to C callers. Each handle owns a private
//! EncodeContext + DecodeContext.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const encoder = @import("encode/streamlz_encoder.zig");
const gpu_encoder = @import("encode/driver.zig");
const decoder = @import("decode/streamlz_decoder.zig");
const gpu_driver = @import("decode/driver.zig");
const frame = @import("format/frame_format.zig");
const version = @import("version.zig");
const vk_api = @import("decode/vulkan_api.zig");

const allocator = std.heap.c_allocator;

// ── Status codes ───────────────────────────────────────────────────────
const SLZ_SUCCESS: c_int = 0;
const SLZ_ERROR_INVALID_HANDLE: c_int = 1;
const SLZ_ERROR_INVALID_ARG: c_int = 2;
const SLZ_ERROR_BUFFER_TOO_SMALL: c_int = 3;
const SLZ_ERROR_CORRUPT_FRAME: c_int = 4;
const SLZ_ERROR_UNSUPPORTED: c_int = 5;
const SLZ_ERROR_CUDA: c_int = 6;
const SLZ_ERROR_OUT_OF_MEMORY: c_int = 7;
const SLZ_ERROR_DEVICE_LOST: c_int = 8;
const SLZ_ERROR_VK_FEATURE_MISSING: c_int = 9;

pub const pseudo_kernel_compressed_size: [*:0]const u8 = "__compressed_size__";
pub const pseudo_kernel_decompressed_size: [*:0]const u8 = "__decompressed_size__";

/// CUDA reference: src/streamlz_gpu.zig:105-110. Mirror of slzCompressOpts_t.
const CompressOpts = extern struct {
    level: c_int = 5,
    enable_profiling: c_int = 0,
    effective_level_out: c_int = 0,
    reserved: [5]c_int = @splat(0),
};

/// CUDA reference: src/streamlz_gpu.zig:113-116. Mirror of slzDecompressOpts_t.
const DecompressOpts = extern struct {
    enable_profiling: c_int = 0,
    reserved: [7]c_int = @splat(0),
};

/// CUDA reference: src/streamlz_gpu.zig:119-122. Mirror of slzKernelTiming_t.
const KernelTimingC = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

/// CUDA reference: src/streamlz_gpu.zig:147-150. Library handle.
const Context = struct {
    enc: gpu_encoder.EncodeContext = .{},
    dec: gpu_driver.DecodeContext = .{},
};

/// CUDA reference: src/streamlz_gpu.zig:426-440. Map status code → string.
export fn slzStatusString(status: c_int) [*:0]const u8 {
    _ = status;
    return "unknown status";
}

/// CUDA reference: src/streamlz_gpu.zig:444-446. Library version string.
export fn slzVersionString() [*:0]const u8 {
    return version.string;
}

/// CUDA reference: src/streamlz_gpu.zig:449-457. Create a handle.
export fn slzCreate(out_handle: ?*?*Context) c_int {
    _ = out_handle;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:459-469. Destroy a handle.
export fn slzDestroy(handle: ?*Context) c_int {
    _ = handle;
    return SLZ_SUCCESS;
}

/// CUDA reference: src/streamlz_gpu.zig:472-474.
export fn slzCompressDefaultOpts() CompressOpts {
    return .{};
}

/// CUDA reference: src/streamlz_gpu.zig:476-478.
export fn slzDecompressDefaultOpts() DecompressOpts {
    return .{};
}

/// CUDA reference: src/streamlz_gpu.zig:489-500. compressBound query.
export fn slzCompressBound(
    handle: ?*Context,
    input_size: usize,
    opts: CompressOpts,
    max_output_size: ?*usize,
) c_int {
    _ = handle;
    _ = input_size;
    _ = opts;
    _ = max_output_size;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:508-527.
export fn slzGetDecompressedSize(
    handle: ?*Context,
    host_bytes: ?*const anyopaque,
    decompressed_size: ?*usize,
) c_int {
    _ = handle;
    _ = host_bytes;
    _ = decompressed_size;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:530-556.
export fn slzCompressHost(
    handle: ?*Context,
    input: ?*const anyopaque,
    input_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
    opts: CompressOpts,
) c_int {
    _ = handle;
    _ = input;
    _ = input_size;
    _ = output;
    _ = output_capacity;
    _ = output_size;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:558-584.
export fn slzDecompressHost(
    handle: ?*Context,
    frame_in: ?*const anyopaque,
    frame_size: usize,
    output: ?*anyopaque,
    output_capacity: usize,
    output_size: ?*usize,
    opts: DecompressOpts,
) c_int {
    _ = handle;
    _ = frame_in;
    _ = frame_size;
    _ = output;
    _ = output_capacity;
    _ = output_size;
    _ = opts;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:590-649. Async D2D compress.
export fn slzCompressAsync(
    handle: ?*Context,
    d_input: ?*const anyopaque,
    input_size: usize,
    d_output: ?*anyopaque,
    max_compressed_size: usize,
    compressed_size: ?*usize,
    opts: CompressOpts,
    stream: ?*anyopaque,
) c_int {
    _ = handle;
    _ = d_input;
    _ = input_size;
    _ = d_output;
    _ = max_compressed_size;
    _ = compressed_size;
    _ = opts;
    _ = stream;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:651-end. Async D2D decompress.
export fn slzDecompressAsync(
    handle: ?*Context,
    d_frame: ?*const anyopaque,
    frame_size: usize,
    d_output: ?*anyopaque,
    output_size: usize,
    opts: DecompressOpts,
    stream: ?*anyopaque,
) c_int {
    _ = handle;
    _ = d_frame;
    _ = frame_size;
    _ = d_output;
    _ = output_size;
    _ = opts;
    _ = stream;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:699-732.
export fn slzGetLastTimings(
    handle: ?*Context,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    _ = handle;
    _ = timings;
    _ = capacity;
    _ = count_out;
    return SLZ_ERROR_UNSUPPORTED;
}

/// CUDA reference: src/streamlz_gpu.zig:742-end.
export fn slzWaitAndGetLastTimings(
    handle: ?*Context,
    stream: ?*anyopaque,
    timings: ?[*]KernelTimingC,
    capacity: usize,
    count_out: ?*usize,
) c_int {
    _ = handle;
    _ = stream;
    _ = timings;
    _ = capacity;
    _ = count_out;
    return SLZ_ERROR_UNSUPPORTED;
}
