//! StreamLZ public library API.
//!
//! Library consumers import this module:
//!   const slz = @import("streamlz");
//!   const n = try slz.compressFramed(allocator, src, dst, .{ .level = 3 });
//!   const m = try slz.decompressFramed(compressed, output);

pub const compressFramed = @import("encode/streamlz_encoder.zig").compressFramed;
pub const compressFramedWithIo = @import("encode/streamlz_encoder.zig").compressFramedWithIo;
pub const compressBound = @import("encode/streamlz_encoder.zig").compressBound;
pub const CompressOptions = @import("encode/streamlz_encoder.zig").CompressOptions;

pub const decompressFramed = @import("decode/streamlz_decoder.zig").decompressFramed;
pub const decompressFramedParallel = @import("decode/streamlz_decoder.zig").decompressFramedParallel;
pub const decompressFramedParallelThreaded = @import("decode/streamlz_decoder.zig").decompressFramedParallelThreaded;
pub const DecompressContext = @import("decode/streamlz_decoder.zig").DecompressContext;
pub const DecompressError = @import("decode/streamlz_decoder.zig").DecompressError;

/// Per-handle GPU decode context. The decode entry points take a
/// `*DecodeContext`; pass `&default_decode_context` for the legacy
/// module-global behavior, or allocate one per library handle.
pub const DecodeContext = @import("decode/fast/gpu_driver.zig").DecodeContext;
pub const default_decode_context = &@import("decode/fast/gpu_driver.zig").g_default;

pub const safe_space = @import("decode/streamlz_decoder.zig").safe_space;
