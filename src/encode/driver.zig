//! GPU encode driver - thin facade.
//!
//! External callers import `encode/driver.zig` and reach every public
//! symbol unchanged; this file owns the two `pub var` singletons
//! (`g_default`, `last_kernel_ns`) and re-exports the per-handle `*Impl`
//! functions from the sub-modules.
//!
//! Sub-module layout:
//!   cuda_ffi.zig         - nvcuda.dll handle, CU* typedefs, FnXxx + cu*_fn slots, getProc
//!   module_loader.zig    - PTX load, kernel-handle pub vars, init(), isAvailable()
//!   encode_context.zig   - EncodeContext, CompressChunkDesc / AssembleDesc /
//!                          HuffEncDesc, ensureBuf, copyDeviceToHost
//!   levels.zig           - hashBitsForLevel / useChainParser
//!   encode_lz.zig        - gpuCompressImpl (LZ launcher)
//!   encode_huff.zig      - gpuEncodeHuffImpl + per-stream {Literals,Tokens,Off16} Impls
//!   encode_assemble.zig  - gpuAssembleFrameImpl, gpuFrameAssembleImpl

const std = @import("std");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const encode_lz = @import("encode_lz.zig");
const encode_huff = @import("encode_huff.zig");
const encode_assemble = @import("encode_assemble.zig");

// ── Module lifecycle ───────────────────────────────────────────
pub const init = module_loader.init;
pub const isAvailable = module_loader.isAvailable;

// ── Shared types ──────────────────────────────────────────────
pub const CompressChunkDesc = encode_context.CompressChunkDesc;
pub const EncodeContext = encode_context.EncodeContext;

pub const copyDeviceToHost = encode_context.copyDeviceToHost;
pub const copyHostToDevice = encode_context.copyHostToDevice;
pub const ensureBuf = encode_context.ensureBuf;

// Wire-format byte sizes re-exported for the CPU-side framer in
// src/encode/fast_framed.zig (which orchestrates GPU encode + frame
// assembly). See encode_context.zig for the underlying definitions.
pub const SC_TAIL_PER_CHUNK_BYTES = encode_context.SC_TAIL_PER_CHUNK_BYTES;
pub const CHUNK_INTERNAL_HDR_BYTES = encode_context.CHUNK_INTERNAL_HDR_BYTES;
pub const UNCOMPRESSED_CHUNK_HDR_BYTES = encode_context.UNCOMPRESSED_CHUNK_HDR_BYTES;
pub const UNCOMPRESSED_CHUNK_MARKER = encode_context.UNCOMPRESSED_CHUNK_MARKER;

// ── Singletons ────────────────────────────────────────────────
// `g_default` and `last_kernel_ns` live on the facade so external callers
// reading `gpu_enc.g_default` / `gpu_enc.last_kernel_ns` keep working
// unchanged. Sub-modules can `@import("driver.zig")` to read the
// singletons; not a cycle.

/// Default context used by the thin public wrappers. A future library
/// API will hand each handle its own `EncodeContext`.
pub var g_default: EncodeContext = .{};

/// Last LZ-encode kernel duration in nanoseconds (set when caller passes
/// an `io` clock to `gpuCompressImpl`). Written by `encode_lz.gpuCompressImpl`
/// via `@import("driver.zig").last_kernel_ns`. Separate from the decode
/// driver's `last_kernel_ns` - the two are not interchangeable; reading
/// this after a decode (or vice versa) returns a stale value.
pub var last_kernel_ns: i64 = 0;

// ── Per-handle entrypoints (re-exported from sub-modules) ─────
// Every caller threads its own `*EncodeContext`; the `g_default` singleton
// above remains the conventional handle used by the CLI / C ABI today.
pub const gpuCompressImpl = encode_lz.gpuCompressImpl;
pub const ensureDictOnDevice = encode_lz.ensureDictOnDevice;
pub const enc_module_loader = module_loader;

/// v4 #19 device-only: launch slzMerkleRootWriteKernel - rolls the
/// collected per-chunk hashes into the root and writes the 4-byte LE
/// trailer at `trailer_off` inside the device-resident frame. Sync
/// launch on the default stream (the subsequent frame D2H orders
/// after it). Returns false when the launch cannot run (caller falls
/// back to the host trailer).
pub fn launchMerkleRootWrite(enc_ctx: *encode_context.EncodeContext, d_frame: u64, trailer_off: usize) bool {
    const cuda_ffi = @import("cuda_ffi.zig");
    const launch_fn = cuda_ffi.cuLaunchKernel_fn orelse return false;
    const sync_fn = cuda_ffi.cuCtxSynchronize_fn orelse return false;
    if (enc_ctx.d_merkle_hashes == 0 or enc_ctx.merkle_total == 0) return false;
    var p_hashes: u64 = enc_ctx.d_merkle_hashes;
    var p_n: u32 = enc_ctx.merkle_total;
    var p_frame: u64 = d_frame;
    var p_off: u64 = @intCast(trailer_off);
    var params = [_]?*anyopaque{
        @ptrCast(&p_hashes), @ptrCast(&p_n), @ptrCast(&p_frame), @ptrCast(&p_off),
    };
    var extra = [_]?*anyopaque{null};
    if (launch_fn(module_loader.merkle_root_write_fn, 1, 1, 1, 1, 1, 1, 0, 0, &params, &extra) != cuda_ffi.CUDA_SUCCESS)
        return false;
    return sync_fn() == cuda_ffi.CUDA_SUCCESS;
}
pub const gpuEncodeLiteralsHuffImpl = encode_huff.gpuEncodeLiteralsHuffImpl;
pub const gpuEncodeTokensHuffImpl = encode_huff.gpuEncodeTokensHuffImpl;
pub const gpuEncodeOff16HuffImpl = encode_huff.gpuEncodeOff16HuffImpl;

// ── Frame assembly (device-resident compress) ────────────────────────
pub const gpuAssembleFrameImpl = encode_assemble.gpuAssembleFrameImpl;
pub const gpuFrameAssembleImpl = encode_assemble.gpuFrameAssembleImpl;
