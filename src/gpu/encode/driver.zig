//! GPU encode driver — thin facade.
//!
//! The encode driver was split into focused sub-modules during the GPU
//! cleanup pass (roadmap item 5). External callers continue to import
//! `gpu/encode/driver.zig` and reach every public symbol unchanged; this
//! file owns the two `pub var` singletons (`g_default`, `last_kernel_ns`)
//! plus the thin wrappers that delegate to per-handle `*Impl` functions
//! in the sub-modules.
//!
//! Sub-module layout:
//!   cuda_ffi.zig         — nvcuda.dll handle, CU* typedefs, FnXxx + cu*_fn slots, getProc
//!   module_loader.zig    — PTX load, kernel-handle pub vars, init(), isAvailable()
//!   encode_context.zig   — EncodeContext, CompressChunkDesc / AssembleDesc /
//!                          HuffEncDesc, ensureBuf, copyDeviceToHost
//!   levels.zig           — hashBitsForLevel / useGlobalHash / useChainParser
//!   encode_lz.zig        — gpuCompressImpl (LZ launcher)
//!   encode_huff.zig      — gpuEncodeHuffImpl + per-stream {Literals,Tokens,Off16} Impls
//!   encode_assemble.zig  — gpuAssembleFrameImpl, gpuFrameAssembleImpl (4d device-resident)

const std = @import("std");
const module_loader = @import("module_loader.zig");
const ec = @import("encode_context.zig");
const encode_lz = @import("encode_lz.zig");
const encode_huff = @import("encode_huff.zig");
const encode_assemble = @import("encode_assemble.zig");

// ── Module lifecycle ───────────────────────────────────────────
pub const init = module_loader.init;
pub const isAvailable = module_loader.isAvailable;

// ── Shared types ──────────────────────────────────────────────
pub const CompressChunkDesc = ec.CompressChunkDesc;
pub const AssembleDesc = ec.AssembleDesc;
pub const HuffEncDesc = ec.HuffEncDesc;
pub const EncodeContext = ec.EncodeContext;

pub const copyDeviceToHost = ec.copyDeviceToHost;

// Wire-format byte sizes re-exported for the CPU-side framer in
// src/encode/fast_framed.zig (which orchestrates GPU encode + frame
// assembly). See encode_context.zig for the underlying definitions.
pub const INITIAL_LITERAL_COPY_BYTES = ec.INITIAL_LITERAL_COPY_BYTES;
pub const SC_TAIL_PER_CHUNK_BYTES = ec.SC_TAIL_PER_CHUNK_BYTES;
pub const CHUNK_INTERNAL_HDR_BYTES = ec.CHUNK_INTERNAL_HDR_BYTES;

// ── Singletons ────────────────────────────────────────────────
// `g_default` and `last_kernel_ns` live on the facade so external callers
// reading `gpu_enc.g_default` / `gpu_enc.last_kernel_ns` keep working
// unchanged. Sub-modules that need to write `last_kernel_ns` import this
// facade back (the @import cycle is fine for `pub var` access).

/// Default context used by the thin public wrappers. A future library
/// API will hand each handle its own `EncodeContext`.
pub var g_default: EncodeContext = .{};

/// Last LZ-encode kernel duration in nanoseconds (set when caller passes
/// an `io` clock to `gpuCompress` / `gpuCompressImpl`). Written by
/// `encode_lz.gpuCompressImpl` via `@import("driver.zig").last_kernel_ns`.
pub var last_kernel_ns: i64 = 0;

// ── LZ-encode ─────────────────────────────────────────────────
pub fn gpuCompress(
    input: []const u8,
    output: []u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes_out: []u32,
    io: ?std.Io,
    level: u8,
) bool {
    return encode_lz.gpuCompressImpl(&g_default, input, output, chunk_descs, comp_sizes_out, io, level);
}
pub const gpuCompressImpl = encode_lz.gpuCompressImpl;

// ── GPU Huffman encode ────────────────────────────────────────
pub fn gpuEncodeHuff(
    descs: []const HuffEncDesc,
    total_dst_bytes: usize,
    out_sizes: []u32,
    out_bytes: []u8,
) bool {
    return encode_huff.gpuEncodeHuffImpl(&g_default, descs, total_dst_bytes, out_sizes, out_bytes, 0,
        .{ "huff/build", "huff/encode" });
}
pub const gpuEncodeHuffImpl = encode_huff.gpuEncodeHuffImpl;

pub fn gpuEncodeLiteralsHuff(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    return encode_huff.gpuEncodeLiteralsHuffImpl(&g_default, allocator, output, chunk_descs, comp_sizes);
}
pub const gpuEncodeLiteralsHuffImpl = encode_huff.gpuEncodeLiteralsHuffImpl;

pub fn gpuEncodeTokensHuff(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    return encode_huff.gpuEncodeTokensHuffImpl(&g_default, allocator, output, chunk_descs, comp_sizes);
}
pub const gpuEncodeTokensHuffImpl = encode_huff.gpuEncodeTokensHuffImpl;

pub fn gpuEncodeOff16Huff(
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    return encode_huff.gpuEncodeOff16HuffImpl(&g_default, allocator, output, chunk_descs, comp_sizes);
}
pub const gpuEncodeOff16HuffImpl = encode_huff.gpuEncodeOff16HuffImpl;

// ── Frame assembly (4d device-resident compress) ──────────────
pub const gpuAssembleFrameImpl = encode_assemble.gpuAssembleFrameImpl;
pub const gpuFrameAssembleImpl = encode_assemble.gpuFrameAssembleImpl;
