//! 1:1 port of src/encode/driver.zig.
//!
//! Encode driver facade. Owns g_default + last_kernel_ns; re-exports
//! every public symbol from the encode sub-modules.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const encode_lz = @import("encode_lz.zig");
const encode_huff = @import("encode_huff.zig");
const encode_assemble = @import("encode_assemble.zig");

// ── Module lifecycle ───────────────────────────────────────────────────
pub const init = module_loader.init;
pub const isAvailable = module_loader.isAvailable;

// ── Descriptor / context types ─────────────────────────────────────────
pub const CompressChunkDesc = encode_context.CompressChunkDesc;
pub const EncodeContext = encode_context.EncodeContext;

pub const copyDeviceToHost = encode_context.copyDeviceToHost;
pub const copyHostToDevice = encode_context.copyHostToDevice;
pub const ensureBuf = encode_context.ensureBuf;

// ── Wire-format constants exposed at the facade ────────────────────────
pub const SC_TAIL_PER_CHUNK_BYTES = encode_context.SC_TAIL_PER_CHUNK_BYTES;
pub const CHUNK_INTERNAL_HDR_BYTES = encode_context.CHUNK_INTERNAL_HDR_BYTES;
pub const UNCOMPRESSED_CHUNK_HDR_BYTES = encode_context.UNCOMPRESSED_CHUNK_HDR_BYTES;
pub const UNCOMPRESSED_CHUNK_MARKER = encode_context.UNCOMPRESSED_CHUNK_MARKER;

// ── Singletons ─────────────────────────────────────────────────────────
pub var g_default: EncodeContext = .{};

pub var last_kernel_ns: i64 = 0;

// ── Encode kernel entry points ─────────────────────────────────────────
pub const gpuCompressImpl = encode_lz.gpuCompressImpl;
pub const gpuEncodeLiteralsHuffImpl = encode_huff.gpuEncodeLiteralsHuffImpl;
pub const gpuEncodeTokensHuffImpl = encode_huff.gpuEncodeTokensHuffImpl;
pub const gpuEncodeOff16HuffImpl = encode_huff.gpuEncodeOff16HuffImpl;

pub const gpuAssembleFrameImpl = encode_assemble.gpuAssembleFrameImpl;
pub const gpuFrameAssembleImpl = encode_assemble.gpuFrameAssembleImpl;
