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
const ec = @import("encode_context.zig");
const encode_lz = @import("encode_lz.zig");
const encode_huff = @import("encode_huff.zig");
const encode_assemble = @import("encode_assemble.zig");

// ── Module lifecycle ───────────────────────────────────────────
pub const init = module_loader.init;
pub const isAvailable = module_loader.isAvailable;

// ── Shared types ──────────────────────────────────────────────
pub const CompressChunkDesc = ec.CompressChunkDesc;
pub const EncodeContext = ec.EncodeContext;

pub const copyDeviceToHost = ec.copyDeviceToHost;
pub const copyHostToDevice = ec.copyHostToDevice;
pub const ensureBuf = ec.ensureBuf;

// Wire-format byte sizes re-exported for the CPU-side framer in
// src/encode/fast_framed.zig (which orchestrates GPU encode + frame
// assembly). See encode_context.zig for the underlying definitions.
pub const SC_TAIL_PER_CHUNK_BYTES = ec.SC_TAIL_PER_CHUNK_BYTES;
pub const CHUNK_INTERNAL_HDR_BYTES = ec.CHUNK_INTERNAL_HDR_BYTES;
pub const UNCOMPRESSED_CHUNK_HDR_BYTES = ec.UNCOMPRESSED_CHUNK_HDR_BYTES;
pub const UNCOMPRESSED_CHUNK_MARKER = ec.UNCOMPRESSED_CHUNK_MARKER;

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
pub const gpuEncodeLiteralsHuffImpl = encode_huff.gpuEncodeLiteralsHuffImpl;
pub const gpuEncodeTokensHuffImpl = encode_huff.gpuEncodeTokensHuffImpl;
pub const gpuEncodeOff16HuffImpl = encode_huff.gpuEncodeOff16HuffImpl;

// ── Frame assembly (4d device-resident compress) ──────────────
pub const gpuAssembleFrameImpl = encode_assemble.gpuAssembleFrameImpl;
pub const gpuFrameAssembleImpl = encode_assemble.gpuFrameAssembleImpl;
