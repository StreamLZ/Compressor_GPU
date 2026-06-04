//! 1:1 port of src/encode/encode_context.zig.
//!
//! Per-encode-operation mutable state. Every CUdeviceptr field becomes
//! a VkDeviceBuffer per Section B. Huff-related fields exist but go
//! unused on L1; the L1 hot path skips them via the `opts.level >= 3`
//! gate in fast_framed.zig.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const vulkan_ffi = @import("vulkan_ffi.zig");
const module_loader = @import("module_loader.zig");
const gpu_decode = @import("../decode/driver.zig");

const VkDeviceBuffer = vulkan_ffi.VkDeviceBuffer;

pub const INITIAL_LITERAL_COPY_BYTES: u32 = 8;
pub const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;
pub const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;
pub const UNCOMPRESSED_CHUNK_HDR_BYTES: u32 = 2;
pub const NEXT_HASH_ENTRIES: usize = 65536;
pub const UNCOMPRESSED_CHUNK_MARKER: u32 = 0xFFFFFFFF;

/// CUDA reference: src/encode/encode_context.zig:50-56. Per-chunk encode
/// descriptor.
pub const CompressChunkDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    dst_offset: u32,
    dst_capacity: u32,
    is_first: u32,
};

/// CUDA reference: src/encode/encode_context.zig:60-74. Per-sub-chunk
/// frame-assembly descriptor.
pub const AssembleDesc = extern struct {
    raw_offset: u32,
    raw_size: u32,
    huff_lit_offset: u32,
    huff_lit_size: u32,
    huff_tok_offset: u32,
    huff_tok_size: u32,
    huff_off16hi_offset: u32,
    huff_off16hi_size: u32,
    huff_off16lo_offset: u32,
    huff_off16lo_size: u32,
    sub_decomp_size: u32,
    init_bytes: u32,
    out_offset: u32,
};

/// CUDA reference: src/encode/encode_context.zig:79-85. Per-stream
/// Huffman encode descriptor.
pub const HuffEncDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    src_stride: u32,
    dst_offset: u32,
    dst_capacity: u32,
};

/// CUDA reference: src/encode/encode_context.zig:92-277. Per-encode
/// context. Every CUdeviceptr slot becomes VkDeviceBuffer.
pub const EncodeContext = struct {
    enable_profiling: bool = false,
    pending_timings: std.ArrayListUnmanaged(gpu_decode.PendingTiming) = .empty,
    last_timings: std.ArrayListUnmanaged(gpu_decode.KernelTiming) = .empty,

    work_stream: usize = 0,

    d_input_override: u64 = 0,
    d_output_override: u64 = 0,
    output_written_to_device: bool = false,

    d_host_wrap_input: VkDeviceBuffer = 0,
    d_host_wrap_input_size: usize = 0,
    d_host_wrap_output: VkDeviceBuffer = 0,
    d_host_wrap_output_size: usize = 0,

    d_frame_chunk_dst: VkDeviceBuffer = 0,
    d_frame_chunk_dst_size: usize = 0,
    d_frame_asm_offsets: VkDeviceBuffer = 0,
    d_frame_asm_offsets_size: usize = 0,
    d_frame_asm_chunk_sz: VkDeviceBuffer = 0,
    d_frame_asm_chunk_sz_size: usize = 0,
    d_frame_prefix_bytes: VkDeviceBuffer = 0,
    d_frame_prefix_bytes_size: usize = 0,

    d_input_persist: VkDeviceBuffer = 0,
    d_input_size: usize = 0,
    d_output_persist: VkDeviceBuffer = 0,
    d_output_size: usize = 0,
    d_descs_persist: VkDeviceBuffer = 0,
    d_descs_size: usize = 0,
    d_hash_persist: VkDeviceBuffer = 0,
    d_hash_size: usize = 0,
    d_sizes_persist: VkDeviceBuffer = 0,
    d_sizes_size: usize = 0,

    d_huff_descs_persist: VkDeviceBuffer = 0,
    d_huff_descs_size: usize = 0,
    d_huff_cl_persist: VkDeviceBuffer = 0,
    d_huff_cl_size: usize = 0,
    d_huff_codes_persist: VkDeviceBuffer = 0,
    d_huff_codes_size: usize = 0,
    d_huff_scratch_persist: VkDeviceBuffer = 0,
    d_huff_scratch_size: usize = 0,
    d_huff_sizes_persist: VkDeviceBuffer = 0,
    d_huff_sizes_size: usize = 0,

    d_asm_huff_lit: VkDeviceBuffer = 0,
    d_asm_huff_lit_size: usize = 0,
    d_asm_huff_tok: VkDeviceBuffer = 0,
    d_asm_huff_tok_size: usize = 0,
    d_asm_huff_off16: VkDeviceBuffer = 0,
    d_asm_huff_off16_size: usize = 0,
    d_asm_descs: VkDeviceBuffer = 0,
    d_asm_descs_size: usize = 0,
    d_asm_out: VkDeviceBuffer = 0,
    d_asm_out_size: usize = 0,
    d_asm_sizes: VkDeviceBuffer = 0,
    d_asm_sizes_size: usize = 0,

    assembled_offsets: ?[]u32 = null,
    assembled_sizes: ?[]u32 = null,

    huff_off16hi_sizes: ?[]u32 = null,
    huff_off16hi_offsets: ?[]u32 = null,
    huff_off16lo_sizes: ?[]u32 = null,
    huff_off16lo_offsets: ?[]u32 = null,

    huff_lit_sizes: ?[]u32 = null,
    huff_lit_offsets: ?[]u32 = null,
    huff_tok_sizes: ?[]u32 = null,
    huff_tok_offsets: ?[]u32 = null,

    /// CUDA reference: src/encode/encode_context.zig:223-277. Free every
    /// owned device + host buffer and reset every field to its default.
    pub fn deinit(self: *EncodeContext, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }
};

/// CUDA reference: src/encode/encode_context.zig:284-293. Grow a device
/// buffer to at least `needed` bytes. Returns false on alloc failure
/// (encode-side bool convention).
pub fn ensureBuf(ptr: *VkDeviceBuffer, current_size: *usize, needed: usize) bool {
    _ = ptr;
    _ = current_size;
    _ = needed;
    return false;
}

/// CUDA reference: src/encode/encode_context.zig:296-300. Blocking D2H
/// copy.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    _ = dst;
    _ = src_device;
    return false;
}

/// CUDA reference: src/encode/encode_context.zig:303-end. Blocking H2D
/// copy.
pub fn copyHostToDevice(dst_device: u64, src: []const u8) bool {
    _ = dst_device;
    _ = src;
    return false;
}
