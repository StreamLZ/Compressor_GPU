//! 1:1 port of src/decode/descriptors.zig.
//!
//! POD descriptor structs the decode pipeline plumbs between Zig host
//! and SPIR-V shaders, plus the `GpuError` set, the `vkCall` wrapper
//! (renamed from `cudaCall` per Section B), and the per-kernel timing
//! types.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");

/// CUDA reference: src/decode/descriptors.zig:14-26. Per-chunk descriptor.
pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    reserved: [3]u8 = @splat(0),
};

/// CUDA reference: src/decode/descriptors.zig:33-39. Per-chunk Huffman
/// descriptor.
pub const HuffDecChunkDesc = extern struct {
    in_offset: u32,
    in_size: u32,
    out_offset: u32,
    out_size: u32,
    lut_offset: u32,
};

pub const HUFF_LUT_ENTRIES: usize = 1024;

pub const MAX_SUB_CHUNKS_PER_CHUNK: u32 = 4;
pub const MAX_HUFF_DESCS_PER_STREAM: u32 = WALK_MAX_CHUNKS / MAX_SUB_CHUNKS_PER_CHUNK;
pub const MAX_RAW_OFF16_DESCS: u32 = 8192;

pub const ENTROPY_SCRATCH_SLOT_BYTES: u64 = 131072;
pub const OFF16_HILO_SPLIT_OFFSET: u32 = 65536;

pub const CHUNK_TYPE_SHIFT: u3 = 4;
pub const CHUNK_TYPE_MASK: u8 = 0x7;

/// CUDA reference: src/decode/descriptors.zig:81-84. Per-kernel timing
/// surfaced across the C ABI.
pub const KernelTiming = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

/// CUDA reference: src/decode/descriptors.zig:90-94. In-flight per-kernel
/// timing entry held internally on a DecodeContext.
pub const PendingTiming = struct {
    name: [*:0]const u8,
    start_event: usize,
    end_event: usize,
};

/// CUDA reference: src/decode/descriptors.zig:102-108. Staged decode-scan
/// output (Huffman variant).
pub const ScanHuffDesc = extern struct {
    in_offset: u32 = 0,
    in_size: u32 = 0,
    out_offset: u32 = 0,
    out_size: u32 = 0,
    valid: u32 = 0,
};

/// CUDA reference: src/decode/descriptors.zig:109-114. Staged decode-scan
/// output (raw variant).
pub const ScanRawDesc = extern struct {
    src_offset: u32 = 0,
    size: u32 = 0,
    gpu_offset: u32 = 0,
    valid: u32 = 0,
};

/// CUDA reference: src/decode/descriptors.zig:117-121. Raw off16 sub-stream
/// host-side descriptor.
pub const RawOff16Desc = struct {
    src_offset: u32,
    size: u32,
    gpu_offset: u32,
};

/// CUDA reference: src/decode/descriptors.zig:123-129. Aggregated scan
/// result counts.
pub const ScanResult = struct {
    num_raw_off16: u32 = 0,
    num_huff_lit: u32 = 0,
    num_huff_tok: u32 = 0,
    num_huff_off16hi: u32 = 0,
    num_huff_off16lo: u32 = 0,
};

/// CUDA reference: src/decode/descriptors.zig:136-139. Device-only walk
/// result handles.
pub const WalkFrameResultDev = struct {
    d_chunk_descs: u64,
    d_meta: u64,
};

pub const WALK_MAX_CHUNKS: u32 = 16384;

/// CUDA reference: src/decode/descriptors.zig:141-149. Field offsets into
/// the walk-result meta buffer.
pub const walk_meta_offsets = struct {
    pub const n_chunks: u32 = 0;
    pub const decomp_size: u32 = 4;
    pub const sub_chunk_cap: u32 = 8;
    pub const block_start: u32 = 12;
    pub const block_size: u32 = 16;
    pub const status: u32 = 20;
    pub const bytes: usize = 24;
};

/// CUDA reference: src/decode/descriptors.zig:154-157. Device-only
/// prefix-sum result handles.
pub const PrefixSumResultDev = struct {
    d_first_sub_idx: u64,
    d_total_subchunks: u64,
};

/// CUDA reference: src/decode/descriptors.zig:163-190. GPU decode-path
/// error set. The VK port adds NotImplementedL2 (cross-cutting per
/// srcVK/error.zig).
pub const GpuError = error{
    BackendNotAvailable,
    OutOfDeviceMemory,
    KernelLaunchFailed,
    SyncFailed,
    CopyFailed,
    KernelMissing,
    BadMode,
    NotImplementedL2,
    NotYetPorted,
};

/// CUDA reference: src/decode/descriptors.zig:195-206. Categorises what
/// the wrapped Vulkan call was doing so vkCall returns the matching
/// GpuError member.
pub const ErrorKind = enum {
    launch,
    sync,
    copy,
    alloc,
    init,
};

/// CUDA reference: src/decode/descriptors.zig:212-end. Renamed from
/// cudaCall per Section B. Funnels a Vulkan return code into the
/// GpuError surface.
pub fn vkCall(rc: c_int, comptime kind: ErrorKind) GpuError!void {
    if (rc == 0) return;
    return switch (kind) {
        .launch => error.KernelLaunchFailed,
        .sync => error.SyncFailed,
        .copy => error.CopyFailed,
        .alloc => error.OutOfDeviceMemory,
        .init => error.BackendNotAvailable,
    };
}
