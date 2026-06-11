//! Per-encode-operation mutable state + the C-ABI descriptor structs
//! that the host shares with the encode kernels.
//!
//! `EncodeContext` holds every persistent device-buffer pointer the LZ,
//! Huffman, and frame-assembly passes reuse across calls. Splitting this
//! out lets every encode sub-module (encode_lz, encode_huff,
//! encode_assemble) take a `*EncodeContext` without a circular import on
//! the top-level driver.

const std = @import("std");
const cuda_ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = cuda_ffi.CUdeviceptr;

// ── Wire-format byte sizes (Zig mirrors of common/gpu_wire_format.cuh) ──
//
//   INITIAL_LITERAL_COPY_BYTES - 8 verbatim literal bytes the very first
//     sub-chunk emits as a raw prefix; per-sub-chunk init copies are
//     restored from the SC-tail prefix table post-decode.
//   SC_TAIL_PER_CHUNK_BYTES    - bytes-per-entry in the SC-tail prefix
//     table that follows the assembled blocks (entry i holds the first 8
//     bytes of source chunk i+1, so the decoder can restore the init copy
//     without re-encoding it inside each chunk).
//   CHUNK_INTERNAL_HDR_BYTES   - per-chunk internal block header on disk:
//     2-byte SLZ internal block header + 4-byte chunk-size word = 6 bytes
//     prefixing each chunk's assembled payload.
//   NEXT_HASH_ENTRIES          - chain-parser next-hash modular index size
//     (2^16). Matches NEXT_HASH_SIZE in encode/lz_format.cuh.
pub const INITIAL_LITERAL_COPY_BYTES: u32 = 8;
pub const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;
pub const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;
/// Per-chunk header overhead for a compressed chunk: 2-byte internal
/// block header + 4-byte chunk header word. An uncompressed chunk
/// carries only the internal block header (with the uncompressed flag
/// set in byte 0), so its overhead is `UNCOMPRESSED_CHUNK_HDR_BYTES`.
pub const UNCOMPRESSED_CHUNK_HDR_BYTES: u32 = 2;
pub const NEXT_HASH_ENTRIES: usize = 65536;

/// Sentinel value the host writes into `per_chunk_asm_off[i]` when chunk
/// `i` is being emitted uncompressed. The frame-assembly kernel reads
/// this exact value and switches its per-chunk write path: instead of
/// copying from `d_asm_out`, it copies `per_chunk_asm_size[i]` raw bytes
/// from `d_input` at offset `i * eff_chunk_size`. Must stay byte-equal
/// to `UNCOMPRESSED_CHUNK_MARKER` in assemble_kernel.cu.
pub const UNCOMPRESSED_CHUNK_MARKER: u32 = 0xFFFFFFFF;

// ── Chunk descriptor (matches CUDA struct) ──────────────────────
pub const CompressChunkDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    dst_offset: u32,
    dst_capacity: u32,
    is_first: u32,
};

// Per-sub-chunk descriptor for the frame-assembly kernels. Mirrors
// `AssembleDesc` in assemble_kernel.cu - keep field order/types in sync.
pub const AssembleDesc = extern struct {
    raw_offset: u32, // sub-chunk raw payload offset in d_output
    raw_size: u32, // raw payload byte count (comp_sizes[i])
    huff_lit_offset: u32,
    huff_lit_size: u32, // 0 = no Huffman body -> stream is raw
    huff_tok_offset: u32,
    huff_tok_size: u32,
    huff_off16hi_offset: u32,
    huff_off16hi_size: u32,
    huff_off16lo_offset: u32,
    huff_off16lo_size: u32,
    sub_decomp_size: u32, // decompressed size of this sub-chunk
    init_bytes: u32, // 8 for the frame's first sub-chunk (verbatim prefix), else 0
    out_offset: u32, // assembled [hdr+payload] destination (filled pass 2)
};

/// GPU Huffman encode descriptor - matches `HuffEncDesc` in
/// gpu_huff_kernel.cu. `src_stride` is 1 for a contiguous stream or 2
/// to encode one off16 byte plane.
pub const HuffEncDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    src_stride: u32,
    dst_offset: u32,
    dst_capacity: u32,
};

/// Per-encode-operation mutable state. Every persistent device-buffer
/// pointer and its buffer-size companion formerly held as a module-global
/// lives here so a future library API can hand each handle its own
/// context. Load-once module handles, kernel/driver function pointers,
/// and `pub var` result slices stay module-global on purpose.
pub const EncodeContext = struct {
    // Per-kernel timing (slzCompressOpts_t.enable_profiling). When true,
    // major encode kernels record a cuEvent pair and append to
    // `pending_timings`; `finalizeProfiling` (in the decode driver, called
    // after the encode's final sync) drains pending → `last_timings`.
    enable_profiling: bool = false,
    pending_timings: std.ArrayListUnmanaged(gpu_decode.PendingTiming) = .empty,
    last_timings: std.ArrayListUnmanaged(gpu_decode.KernelTiming) = .empty,

    // CUDA stream used for the heavy-phase encode kernel launches (LZ
    // compress, huff encode, assemble write, frame assemble) + the final
    // frame writer. slzCompressAsync sets it to the caller's stream so
    // cudaStreamSynchronize on that stream waits for the compress to
    // complete. The sync wrapper leaves it at 0 (default stream).
    work_stream: usize = 0,

    /// When non-zero, the encode pipeline reads source bytes from this
    /// device address (D2D copy into `d_input_persist`) instead of H2D-ing
    /// from the host `input` slice. Set by `slzCompressAsync` (caller's
    /// device input) and by `compressFramedOne` when it wraps a host
    /// input through the internal pure-D2D path. Reset to 0 by the caller
    /// after the compress call.
    d_input_override: u64 = 0,

    /// When non-zero, the frame-assembly kernel writes the entire SLZ1
    /// frame directly into this device buffer; no host-side frame
    /// assembly runs. Set by `slzCompressAsync` (caller's device output)
    /// and by `compressFramedOne` when it wraps. Reset to 0 by the
    /// caller after the compress call.
    d_output_override: u64 = 0,
    output_written_to_device: bool = false,

    /// Persistent device buffer holding the host source bytes when the
    /// caller did not provide a device input. Reused across calls;
    /// grown via `ensureBuf` when a larger input arrives.
    d_host_wrap_input: CUdeviceptr = 0,
    d_host_wrap_input_size: usize = 0,

    /// Persistent device buffer holding the assembled frame when the
    /// caller did not provide a device output. Reused across calls.
    d_host_wrap_output: CUdeviceptr = 0,
    d_host_wrap_output_size: usize = 0,

    // Frame-assembly scratch — small device buffers H2D'd per call to
    // feed slzFrameAssembleKernel:
    //   d_frame_chunk_dst    - per-chunk dst offsets (n_chunks × u32)
    //   d_frame_asm_offsets  - per-chunk asm-out start offsets (n_chunks × u32)
    //   d_frame_asm_chunk_sz - per-chunk total asm size (n_chunks × u32)
    //   d_frame_prefix_bytes - pre-formed frame_hdr + block_hdr (~40 B)
    d_frame_chunk_dst: CUdeviceptr = 0,
    d_frame_chunk_dst_size: usize = 0,
    d_frame_asm_offsets: CUdeviceptr = 0,
    d_frame_asm_offsets_size: usize = 0,
    d_frame_asm_chunk_sz: CUdeviceptr = 0,
    d_frame_asm_chunk_sz_size: usize = 0,
    d_frame_prefix_bytes: CUdeviceptr = 0,
    d_frame_prefix_bytes_size: usize = 0,

    // ── LZ-encode persistent device buffers ────────────────────
    d_input_persist: CUdeviceptr = 0,
    d_input_size: usize = 0,
    d_output_persist: CUdeviceptr = 0,
    d_output_size: usize = 0,
    d_descs_persist: CUdeviceptr = 0,
    d_descs_size: usize = 0,
    d_hash_persist: CUdeviceptr = 0,
    d_hash_size: usize = 0,
    d_sizes_persist: CUdeviceptr = 0,
    d_sizes_size: usize = 0,

    // ── v4 #17: pinned host staging for the LZ compressed-chunk
    // gather (reverse-port of VK's ensureD2hFinalBuf shape). Grow-only,
    // freed in deinit via cuMemFreeHost. The gather queues one
    // cuMemcpyDtoHAsync per chunk into this buffer (true DMA — the
    // copy engine pipelines the regions) and syncs ONCE, replacing the
    // per-chunk synchronous pageable D2H loop that cost ~17 ms at
    // enwik8 scale (382 chunks x ~45 us each).
    h_gather_pinned: ?[*]u8 = null,
    h_gather_size: usize = 0,

    // ── GPU Huffman encode persistent device buffers ───────────
    d_huff_descs_persist: CUdeviceptr = 0,
    d_huff_descs_size: usize = 0,
    d_huff_cl_persist: CUdeviceptr = 0,
    d_huff_cl_size: usize = 0,
    d_huff_codes_persist: CUdeviceptr = 0,
    d_huff_codes_size: usize = 0,
    d_huff_scratch_persist: CUdeviceptr = 0,
    d_huff_scratch_size: usize = 0,
    d_huff_sizes_persist: CUdeviceptr = 0,
    d_huff_sizes_size: usize = 0,

    // ── Frame-assembly (device-resident compress) device buffers ─────
    d_asm_huff_lit: CUdeviceptr = 0,
    d_asm_huff_lit_size: usize = 0,
    d_asm_huff_tok: CUdeviceptr = 0,
    d_asm_huff_tok_size: usize = 0,
    d_asm_huff_off16: CUdeviceptr = 0,
    d_asm_huff_off16_size: usize = 0,
    d_asm_descs: CUdeviceptr = 0,
    d_asm_descs_size: usize = 0,
    d_asm_out: CUdeviceptr = 0,
    d_asm_out_size: usize = 0,
    d_asm_sizes: CUdeviceptr = 0,
    d_asm_sizes_size: usize = 0,

    // Per-SUB-CHUNK index tables published by `gpuAssembleFrameImpl`:
    // where each sub-chunk's assembled-block bytes live inside
    // `d_asm_out` (offset) and how many bytes long the sub-chunk's
    // slice is (size). The frame writer consults both when computing
    // per-chunk dst positions for `slzFrameAssembleKernel`; the
    // assembled bytes themselves stay on the GPU.
    //
    // The current encoder produces exactly one sub-chunk per chunk
    // (`resolveScGroupSize` returns 0.25 or 0.5), so the frame writer
    // indexes these tables with chunk index. A future `sc_group_size
    // >= 1.0` override would invalidate that 1:1 mapping; see the
    // assertion in `fast_framed.assembleFrame`.
    assembled_offsets: ?[]u32 = null,
    assembled_sizes: ?[]u32 = null,

    // ── Per-sub-chunk Huffman body size + dst-offset tables ──
    // The bodies themselves stay device-resident in d_asm_huff_{lit,tok,
    // off16}; only these small index tables come back to the host and feed
    // the AssembleDesc construction in `encode_assemble.zig`. hi/lo for
    // off16 share one device byte buffer, hence two offset tables.
    huff_off16hi_sizes: ?[]u32 = null,
    huff_off16hi_offsets: ?[]u32 = null,
    huff_off16lo_sizes: ?[]u32 = null,
    huff_off16lo_offsets: ?[]u32 = null,

    huff_lit_sizes: ?[]u32 = null,
    huff_lit_offsets: ?[]u32 = null,
    huff_tok_sizes: ?[]u32 = null,
    huff_tok_offsets: ?[]u32 = null,

    /// Free every owned DEVICE buffer and zero the size slots, leaving
    /// the context fully reusable — `ensureBuf` is grow-only from a
    /// zero size, so the next encode transparently re-allocates.
    ///
    /// 2026-06-10: added for the CLI `-b` / `-ba` flows, which encode
    /// and then decode in ONE process. After a 1 GB L3 encode the
    /// persistent set holds ~13 GB (8 GB hash + 2.9 GB LZ output +
    /// 1 GB input + wrap/huffman buffers); the decoder then needs
    /// ~7.5 GB more. On Vulkan's strict allocator that is a hard
    /// OutOfDeviceMemory; on CUDA, WDDM silently pages and poisons the
    /// decompress timings (in-process L3+ decode measured 10-30×
    /// slower than the same frame via `-db`). Trimming between the
    /// phases fixes the VK failure and makes CUDA `-b` numbers honest.
    pub fn releaseDeviceBuffers(self: *EncodeContext) void {
        const free_dev = struct {
            fn f(ptr: *CUdeviceptr, sz: *usize) void {
                if (ptr.* != 0) {
                    // Free failure on a pointer we know is valid (we
                    // allocated it via cuMemAlloc and never freed it)
                    // means the driver/context is dying - there's
                    // nothing useful we can do here, so ignore the
                    // CUresult.
                    if (cuda_ffi.cuMemFree_fn) |free_fn| _ = free_fn(ptr.*);
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
        if (self.h_gather_pinned) |p| {
            if (cuda_ffi.cuMemFreeHost_fn) |free_host| _ = free_host(@ptrCast(p));
            self.h_gather_pinned = null;
            self.h_gather_size = 0;
        }
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
    }

    /// Free every owned device + host buffer and reset every field to
    /// its default. Called by `slzDestroy` on the per-handle
    /// `Context.enc`. The CLI also has a `driver.g_default` singleton
    /// that is intentionally never deinit'd — its lifetime is the
    /// process.
    pub fn deinit(self: *EncodeContext, allocator: std.mem.Allocator) void {
        self.releaseDeviceBuffers();

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

/// Reallocate the device buffer at `ptr` if its current size is below
/// `needed`; otherwise reuse. Caller passes the companion
/// `current_size` slot which is updated on success. Returns `false`
/// only when the CUDA allocation fails (matches the encode side's
/// `bool` convention; the decode side returns `GpuError!void`).
pub fn ensureBuf(ptr: *CUdeviceptr, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    const free_fn = cuda_ffi.cuMemFree_fn orelse return false;
    const alloc_fn = cuda_ffi.cuMemAlloc_fn orelse return false;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    current_size.* = 0;
    if (alloc_fn(ptr, needed) != cuda_ffi.CUDA_SUCCESS) return false;
    current_size.* = needed;
    return true;
}

/// v4 #17: grow-only pinned host staging (cuMemAllocHost). Returns the
/// staging base or null when pinned allocation is unavailable/fails —
/// callers fall back to the synchronous pageable path.
pub fn ensurePinnedGather(self: *EncodeContext, needed: usize) ?[*]u8 {
    if (self.h_gather_size >= needed) return self.h_gather_pinned;
    const alloc_host = cuda_ffi.cuMemAllocHost_fn orelse return null;
    if (self.h_gather_pinned) |p| {
        if (cuda_ffi.cuMemFreeHost_fn) |free_host| _ = free_host(@ptrCast(p));
        self.h_gather_pinned = null;
        self.h_gather_size = 0;
    }
    var raw: ?*anyopaque = null;
    if (alloc_host(&raw, needed) != cuda_ffi.CUDA_SUCCESS) return null;
    self.h_gather_pinned = @ptrCast(raw.?);
    self.h_gather_size = needed;
    return self.h_gather_pinned;
}

/// Copy `dst.len` bytes from a device address into the host slice `dst`.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    if (!module_loader.init()) return false;
    const f = cuda_ffi.cuMemcpyDtoH_fn orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == cuda_ffi.CUDA_SUCCESS;
}

/// Copy `src.len` bytes from the host slice `src` to a device address.
pub fn copyHostToDevice(dst_device: u64, src: []const u8) bool {
    if (!module_loader.init()) return false;
    const f = cuda_ffi.cuMemcpyHtoD_fn orelse return false;
    return f(dst_device, @ptrCast(src.ptr), src.len) == cuda_ffi.CUDA_SUCCESS;
}
