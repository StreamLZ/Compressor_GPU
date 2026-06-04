//! 1:1 port of src/decode/streamlz_decoder.zig.
//!
//! Top-level frame decompressor. Parses the SLZ1 frame header, walks
//! the block list, builds the chunk descriptors, and dispatches each
//! block into the GPU pipeline (gpu_driver.fullGpuLaunchImpl).
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const constants = @import("../format/streamlz_constants.zig");
const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
const gpu_driver = @import("driver.zig");

pub const safe_space = constants.safe_space;

pub const max_content_size: u64 = 4 * 1024 * 1024 * 1024;

/// CUDA reference: src/decode/streamlz_decoder.zig:38-75. Decompress
/// failure modes.
pub const DecompressError = error{
    BadFrame,
    Truncated,
    SizeMismatch,
    InvalidBlockHeader,
    InvalidInternalHeader,
    BadChunkHeader,
    BlockDataTruncated,
    OutputTooSmall,
    ChecksumMismatch,
    ChunkSizeMismatch,
    UnknownDictionary,
    ContentSizeTooLarge,
} || gpu_driver.GpuError || std.mem.Allocator.Error;

/// CUDA reference: src/decode/streamlz_decoder.zig:77-85. Structured
/// decompress result.
pub const DecompressResult = struct {
    written: usize,
    offset: usize = 0,
};

/// CUDA reference: src/decode/streamlz_decoder.zig:97-104. Decompress
/// host→host. Bytes-written variant; delegates to decompressFrameInner.
pub fn decompressFramed(
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
) DecompressError!usize {
    _ = src;
    _ = dst;
    _ = dec_ctx;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:116-125. Same as
/// decompressFramed plus optional std.Io for telemetry.
pub fn decompressFramedThreaded(
    allocator: std.mem.Allocator,
    io: ?std.Io,
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
) DecompressError!DecompressResult {
    _ = allocator;
    _ = io;
    _ = src;
    _ = dst;
    _ = dec_ctx;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:136-182. True-D2D
/// decompress for the v3 C ABI. L2-stub-partial — uses walk_frame_fn.
pub fn decompressFramedFromDevice(
    io: ?std.Io,
    d_frame: u64,
    frame_size: u32,
    d_output: u64,
    dec_ctx: *gpu_driver.DecodeContext,
    decomp_size: u32,
) DecompressError!u32 {
    _ = io;
    _ = d_frame;
    _ = frame_size;
    _ = d_output;
    _ = dec_ctx;
    _ = decomp_size;
    return error.NotImplementedL2;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:189-end. Per-call
/// decompress wrapper used by the CLI.
pub const DecompressContext = struct {
    allocator: std.mem.Allocator,
    io: ?std.Io,
    dec_ctx: *gpu_driver.DecodeContext = &gpu_driver.g_default,

    pub fn initWithIo(allocator: std.mem.Allocator, io: std.Io) DecompressContext {
        return .{ .allocator = allocator, .io = io };
    }

    pub fn decompress(self: *DecompressContext, src: []const u8, dst: []u8) DecompressError!DecompressResult {
        _ = self;
        _ = src;
        _ = dst;
        return error.NotYetPorted;
    }

    pub fn deinit(self: *DecompressContext) void {
        _ = self;
    }
};

/// CUDA reference: src/decode/streamlz_decoder.zig:215-322. Frame-shape
/// inner driver. Audit lists this as an export.
pub fn decompressFrameInner(
    src: []const u8,
    dst: []u8,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
) DecompressError!DecompressResult {
    _ = src;
    _ = dst;
    _ = dec_ctx;
    _ = d_output_target;
    _ = io;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:323-380. Dispatches
/// one compressed block into the GPU pipeline.
pub fn dispatchCompressedBlock(
    block_src: []const u8,
    dst: []u8,
    dst_start_off: usize,
    decompressed_size: usize,
    sc_group_size: f32,
    dec_ctx: *gpu_driver.DecodeContext,
    d_output_target: ?u64,
    io: ?std.Io,
) DecompressError!void {
    _ = block_src;
    _ = dst;
    _ = dst_start_off;
    _ = decompressed_size;
    _ = sc_group_size;
    _ = dec_ctx;
    _ = d_output_target;
    _ = io;
    return error.NotYetPorted;
}

/// CUDA reference: src/decode/streamlz_decoder.zig:386-end. Build the
/// per-chunk descriptor array from the block payload.
pub fn buildChunkDescriptors(
    block_payload: []const u8,
    chunk_descs: []gpu_driver.ChunkDesc,
    eff_chunk_size: usize,
    decompressed_size: usize,
    dst_start_off: usize,
) DecompressError!void {
    _ = block_payload;
    _ = chunk_descs;
    _ = eff_chunk_size;
    _ = decompressed_size;
    _ = dst_start_off;
    return error.NotYetPorted;
}
