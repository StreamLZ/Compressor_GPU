//! 1:1 port of src/encode/encode_context.zig.
//!
//! Per-encode-operation mutable state. Every CUdeviceptr field becomes
//! a VkDeviceBuffer per Section B. All device ops route through
//! `vulkan_api.procs.*` so call sites stay CUDA-shaped (no direct
//! vk*/vma calls in this file).

const std = @import("std");
const vk = @import("../decode/vulkan_api.zig");
const module_loader = @import("module_loader.zig");
const gpu_decode = @import("../decode/driver.zig");

const VkDeviceBuffer = vk.VkDeviceBuffer;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;

// ── Wire-format byte sizes (Zig mirrors of common/gpu_wire_format.cuh) ──
//
// CUDA reference: src/encode/encode_context.zig:31-39.
pub const INITIAL_LITERAL_COPY_BYTES: u32 = 8;
pub const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;
pub const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;
/// CUDA reference: src/encode/encode_context.zig:38.
pub const UNCOMPRESSED_CHUNK_HDR_BYTES: u32 = 2;
pub const NEXT_HASH_ENTRIES: usize = 65536;

/// CUDA reference: src/encode/encode_context.zig:47. Sentinel for an
/// uncompressed chunk's `per_chunk_asm_off[i]` slot.
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
/// frame-assembly descriptor. Mirrors AssembleDesc in CUDA's
/// assemble_kernel.cu byte-for-byte.
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

    /// CUDA reference: src/encode/encode_context.zig:223-276. Free every
    /// owned device + host buffer and reset every field to its default.
    pub fn deinit(self: *EncodeContext, allocator: std.mem.Allocator) void {
        const free_dev = struct {
            fn f(ptr: *VkDeviceBuffer, sz: *usize) void {
                if (ptr.* != 0) {
                    if (vk.procs.free_device) |free_fn| _ = free_fn(ptr.*);
                    ptr.* = 0;
                }
                sz.* = 0;
            }
        }.f;
        free_dev(&self.d_frame_chunk_dst, &self.d_frame_chunk_dst_size);
        free_dev(&self.d_frame_asm_offsets, &self.d_frame_asm_offsets_size);
        free_dev(&self.d_frame_asm_chunk_sz, &self.d_frame_asm_chunk_sz_size);
        free_dev(&self.d_frame_prefix_bytes, &self.d_frame_prefix_bytes_size);
        free_dev(&self.d_input_persist, &self.d_input_size);
        free_dev(&self.d_output_persist, &self.d_output_size);
        free_dev(&self.d_host_wrap_input, &self.d_host_wrap_input_size);
        free_dev(&self.d_host_wrap_output, &self.d_host_wrap_output_size);
        free_dev(&self.d_descs_persist, &self.d_descs_size);
        free_dev(&self.d_hash_persist, &self.d_hash_size);
        free_dev(&self.d_sizes_persist, &self.d_sizes_size);
        free_dev(&self.d_huff_descs_persist, &self.d_huff_descs_size);
        free_dev(&self.d_huff_cl_persist, &self.d_huff_cl_size);
        free_dev(&self.d_huff_codes_persist, &self.d_huff_codes_size);
        free_dev(&self.d_huff_scratch_persist, &self.d_huff_scratch_size);
        free_dev(&self.d_huff_sizes_persist, &self.d_huff_sizes_size);
        free_dev(&self.d_asm_huff_lit, &self.d_asm_huff_lit_size);
        free_dev(&self.d_asm_huff_tok, &self.d_asm_huff_tok_size);
        free_dev(&self.d_asm_huff_off16, &self.d_asm_huff_off16_size);
        free_dev(&self.d_asm_descs, &self.d_asm_descs_size);
        free_dev(&self.d_asm_out, &self.d_asm_out_size);
        free_dev(&self.d_asm_sizes, &self.d_asm_sizes_size);

        if (self.assembled_offsets) |s| { allocator.free(s); self.assembled_offsets = null; }
        if (self.assembled_sizes) |s| { allocator.free(s); self.assembled_sizes = null; }

        if (self.huff_off16hi_sizes) |s| { allocator.free(s); self.huff_off16hi_sizes = null; }
        if (self.huff_off16hi_offsets) |s| { allocator.free(s); self.huff_off16hi_offsets = null; }
        if (self.huff_off16lo_sizes) |s| { allocator.free(s); self.huff_off16lo_sizes = null; }
        if (self.huff_off16lo_offsets) |s| { allocator.free(s); self.huff_off16lo_offsets = null; }

        if (self.huff_lit_sizes) |s| { allocator.free(s); self.huff_lit_sizes = null; }
        if (self.huff_lit_offsets) |s| { allocator.free(s); self.huff_lit_offsets = null; }
        if (self.huff_tok_sizes) |s| { allocator.free(s); self.huff_tok_sizes = null; }
        if (self.huff_tok_offsets) |s| { allocator.free(s); self.huff_tok_offsets = null; }

        self.pending_timings.deinit(std.heap.page_allocator);
        self.last_timings.deinit(std.heap.page_allocator);
    }
};

/// CUDA reference: src/encode/encode_context.zig:284-293. Grow-only:
/// returns early when `current_size.*` already fits `needed`. Otherwise
/// frees the old slot (if any) and allocates a fresh one through
/// `procs.malloc_device`. Returns false only on a backend / allocation
/// failure (encode-side bool convention).
pub fn ensureBuf(ptr: *VkDeviceBuffer, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    const free_fn = vk.procs.free_device orelse return false;
    const alloc_fn = vk.procs.malloc_device orelse return false;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    current_size.* = 0;
    if (alloc_fn(ptr, needed) != VK_SUCCESS_RC) return false;
    current_size.* = needed;
    return true;
}

/// CUDA reference: src/encode/encode_context.zig:296-300. Blocking D2H
/// copy through `procs.d2h`.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    if (!module_loader.init()) return false;
    const f = vk.procs.d2h orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == VK_SUCCESS_RC;
}

/// CUDA reference: src/encode/encode_context.zig:303-307. Blocking H2D
/// copy through `procs.h2d`.
pub fn copyHostToDevice(dst_device: u64, src: []const u8) bool {
    if (!module_loader.init()) return false;
    const f = vk.procs.h2d orelse return false;
    return f(dst_device, @ptrCast(src.ptr), src.len) == VK_SUCCESS_RC;
}
