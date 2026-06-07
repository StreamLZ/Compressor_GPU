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

    // VK adaptation (encode D2H gather): persistent page-aligned host
    // buffer that backs `gpu_out` in fast_framed.compressFramedOne. The
    // pre-fix path alloc'd ~285 MB via std.heap per-encode (`defer
    // free`), which produced a fresh host_ptr every call — defeating the
    // iter-8 LRU import cache so procD2HOffsetGather would pay a fresh
    // ~1 ms VK_EXT_external_memory_host import (vkCreateBuffer +
    // vkAllocateMemory) on every encode.
    //
    // Backed by gpu_dec_driver.allocHost (which routes through
    // procs.malloc_host = procMallocHost → page-aligned page_allocator
    // alloc) so the host_ptr satisfies VkImportMemoryHostPointerInfoEXT's
    // allocation-size + alignment contract (minImportedHostPointerAlignment
    // == 4 KiB on every desktop driver). Grow-only via ensureGpuOutBuf;
    // freed in deinit. On grow the prior LRU import cache entry that
    // pointed at the old host pages is released BEFORE freeHost so a
    // recycled page address doesn't collide with a stale cache entry.
    gpu_out_buf: ?[]u8 = null,
    gpu_out_buf_size: usize = 0,

    // VK adaptation (iter-4): persistent page-aligned host buffer for the
    // final-frame D2H readback. Mirrors gpu_out_buf above but is the
    // destination for the ~50 MB compressed-frame D2H out of
    // d_host_wrap_output. Backed by gpu_dec_driver.allocHost
    // (procMallocHost → page-aligned page_allocator alloc) so
    // VK_EXT_external_memory_host import in procD2HOffsetGather can
    // succeed and the iter-8 LRU import cache hits on every call after
    // the first. The caller's `dst` slice (from std.heap allocator at
    // bench_compress.zig) is NOT page-aligned, so we gather into this
    // pinned scratch and @memcpy to dst at the end. The memcpy cost
    // (~10 ms for 50 MB on DDR5) is ~12x cheaper than the prior on-main-
    // queue staging D2H (~245 ms at 0.21 GB/s).
    d2h_final_buf: ?[]u8 = null,
    d2h_final_buf_size: usize = 0,

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

        // VK adaptation (encode D2H gather): release the LRU import
        // entry that points at these host pages BEFORE freeHost, or the
        // cache slot would outlive the underlying allocation — a
        // subsequent procMallocHost happening to reuse the same address
        // would silently reuse the stale (vk_buf, vk_mem) pair and
        // vkCmdCopyBuffer would target the prior physical pages
        // (caller would read zeros).
        if (self.gpu_out_buf) |buf| {
            @import("../decode/module_loader.zig").releaseImportsByHostRange(@ptrCast(buf.ptr), buf.len);
            if (vk.procs.free_host) |free_host_fn| _ = free_host_fn(@ptrCast(buf.ptr));
            self.gpu_out_buf = null;
            self.gpu_out_buf_size = 0;
        }

        // VK adaptation (iter-4): symmetric to gpu_out_buf — release any
        // LRU import entry that points at these host pages BEFORE freeHost
        // so a recycled page address can't collide with a stale cache entry.
        if (self.d2h_final_buf) |buf| {
            @import("../decode/module_loader.zig").releaseImportsByHostRange(@ptrCast(buf.ptr), buf.len);
            if (vk.procs.free_host) |free_host_fn| _ = free_host_fn(@ptrCast(buf.ptr));
            self.d2h_final_buf = null;
            self.d2h_final_buf_size = 0;
        }

        self.pending_timings.deinit(std.heap.page_allocator);
        self.last_timings.deinit(std.heap.page_allocator);
    }
};

/// VK adaptation (encode D2H gather): grow-only host buffer ensure.
/// Returns the live slice (sized to `needed`) on success, null on
/// backend failure. Mirrors `ensureBuf` shape but for the page-aligned
/// host allocation backing `gpu_out` in fast_framed.compressFramedOne.
/// On grow the LRU import cache entry that referenced the old host
/// pages is released BEFORE freeHost so a future procD2HOffsetGather
/// against this buffer doesn't hit a stale (vk_buf, vk_mem) pair
/// pointing at recycled memory.
pub fn ensureGpuOutBuf(self: *EncodeContext, needed: usize) ?[]u8 {
    if (self.gpu_out_buf) |existing| {
        if (existing.len >= needed) return existing[0..needed];
    }
    const allocHost = @import("../decode/decode_context.zig").allocHost;
    if (self.gpu_out_buf) |old| {
        @import("../decode/module_loader.zig").releaseImportsByHostRange(@ptrCast(old.ptr), old.len);
        if (vk.procs.free_host) |free_host_fn| _ = free_host_fn(@ptrCast(old.ptr));
        self.gpu_out_buf = null;
        self.gpu_out_buf_size = 0;
    }
    const fresh = allocHost(needed) orelse return null;
    self.gpu_out_buf = fresh;
    self.gpu_out_buf_size = fresh.len;
    return fresh[0..needed];
}

/// VK adaptation (iter-4): grow-only persistent page-aligned host buffer
/// for the final-frame D2H readback. Symmetric to ensureGpuOutBuf above
/// (which serves the LZ-pass D2H gather). On grow the LRU import cache
/// entry that referenced the old host pages is released BEFORE freeHost
/// so a future procD2HOffsetGather against this buffer doesn't hit a
/// stale (vk_buf, vk_mem) pair pointing at recycled memory.
pub fn ensureD2hFinalBuf(self: *EncodeContext, needed: usize) ?[]u8 {
    if (self.d2h_final_buf) |existing| {
        if (existing.len >= needed) return existing[0..needed];
    }
    const allocHost = @import("../decode/decode_context.zig").allocHost;
    if (self.d2h_final_buf) |old| {
        @import("../decode/module_loader.zig").releaseImportsByHostRange(@ptrCast(old.ptr), old.len);
        if (vk.procs.free_host) |free_host_fn| _ = free_host_fn(@ptrCast(old.ptr));
        self.d2h_final_buf = null;
        self.d2h_final_buf_size = 0;
    }
    const fresh = allocHost(needed) orelse return null;
    self.d2h_final_buf = fresh;
    self.d2h_final_buf_size = fresh.len;
    return fresh[0..needed];
}

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
