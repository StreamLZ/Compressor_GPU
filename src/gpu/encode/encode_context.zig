//! Per-encode-operation mutable state + the C-ABI descriptor structs
//! that the host shares with the encode kernels.
//!
//! `EncodeContext` holds every persistent device-buffer pointer the LZ,
//! Huffman, and frame-assembly passes reuse across calls. Splitting this
//! out lets every encode sub-module (encode_lz, encode_huff,
//! encode_assemble) take a `*EncodeContext` without a circular import on
//! the top-level driver.

const std = @import("std");
const ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = ffi.CUdeviceptr;

// ── Wire-format byte sizes (Zig mirrors of common/gpu_wire_format.cuh) ──
//
//   INITIAL_LITERAL_COPY_BYTES — 8 verbatim literal bytes the very first
//     sub-chunk emits as a raw prefix; per-sub-chunk init copies are
//     restored from the SC-tail prefix table post-decode.
//   SC_TAIL_PER_CHUNK_BYTES    — bytes-per-entry in the SC-tail prefix
//     table that follows the assembled blocks (entry i holds the first 8
//     bytes of source chunk i+1, so the decoder can restore the init copy
//     without re-encoding it inside each chunk).
//   CHUNK_INTERNAL_HDR_BYTES   — per-chunk internal block header on disk:
//     2-byte SLZ internal block header + 4-byte chunk-size word = 6 bytes
//     prefixing each chunk's assembled payload.
//   NEXT_HASH_ENTRIES          — chain-parser next-hash modular index size
//     (2^16). Matches NEXT_HASH_SIZE in encode/lz_format.cuh.
pub const INITIAL_LITERAL_COPY_BYTES: u32 = 8;
pub const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;
pub const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;
pub const NEXT_HASH_ENTRIES: usize = 65536;

// ── Chunk descriptor (matches CUDA struct) ──────────────────────
pub const CompressChunkDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    dst_offset: u32,
    dst_capacity: u32,
    is_first: u32,
};

// Per-sub-chunk descriptor for the frame-assembly kernels. Mirrors
// `AssembleDesc` in assemble_kernel.cu — keep field order/types in sync.
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

/// GPU Huffman encode descriptor — matches `HuffEncDesc` in
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

    // 4d Phase 3: when set to a non-zero device address, gpuCompressImpl
    // populates d_input_persist via a D2D copy from this pointer instead
    // of the H2D from the host `input` slice — the caller's data is
    // already GPU-resident (slzCompress D2D path). The caller resets it
    // to 0 after the compress call.
    d_input_override: u64 = 0,

    // 4d step 8: when set to a non-zero device address, the frame-assembly
    // path writes the full StreamLZ frame straight to this device buffer
    // (slzFrameAssembleKernel), skipping the host frame-build loop and
    // the wrapper's H2D bounce. Caller (slzCompress C-ABI) inspects
    // `output_written_to_device` after the encode to decide whether the
    // H2D fallback is needed.
    d_output_override: u64 = 0,
    output_written_to_device: bool = false,

    // 4d step 8 scratch — small device buffers populated host-side per call:
    //   d_frame_chunk_dst    — per-chunk dst offsets (n_chunks × u32)
    //   d_frame_asm_offsets  — per-chunk asm-out start offsets (n_chunks × u32)
    //   d_frame_asm_chunk_sz — per-chunk total asm size (n_chunks × u32)
    //   d_frame_prefix_bytes — pre-formed frame_hdr + block_hdr (~40 B)
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

    // ── GPU Huffman encode persistent device buffers ───────────
    d_huff_descs_persist: CUdeviceptr = 0,
    d_huff_descs_size: usize = 0,
    d_huff_cl_persist: CUdeviceptr = 0,
    d_huff_cl_size: usize = 0,
    d_huff_codes_persist: CUdeviceptr = 0,
    d_huff_codes_size: usize = 0,
    d_huff_scratch_persist: CUdeviceptr = 0,
    d_huff_scratch_size: usize = 0,
    d_huff_out_persist: CUdeviceptr = 0,
    d_huff_out_size: usize = 0,
    d_huff_sizes_persist: CUdeviceptr = 0,
    d_huff_sizes_size: usize = 0,

    // ── Frame-assembly (4d device-resident compress) device buffers ──
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
    // Host-side assembled result — packed [3-byte sub-chunk hdr][payload]
    // blocks; the frame assembler splices block i from
    // assembled_data[assembled_offsets[i]..][0..assembled_sizes[i]].
    assembled_data: ?[]u8 = null,
    assembled_offsets: ?[]u32 = null,
    assembled_sizes: ?[]u32 = null,
    // When set, the three GPU Huffman passes keep their bodies device-
    // resident in d_asm_huff_{lit,tok,off16} (no host bounce) so the
    // frame-assembly kernels read them directly. Set by the encoder
    // before the Huffman passes when SLZ_GPU_ASSEMBLE is active.
    huff_keep_device: bool = false,

    // ── Result slices — formerly module-global `pub var`. Each encode
    // operation writes its downloaded host-side payloads here; the frame
    // assembler reads them back. Moved into the context so the compress
    // path is reentrant per handle.
    // OWNERSHIP RULE: huff_off16hi_data and huff_off16lo_data share
    // one allocation (a flat byte buffer indexed by hi_offsets and
    // lo_offsets respectively). `huff_off16hi_data` is the owner;
    // `huff_off16lo_data` is a non-owning alias of the SAME slice. The
    // only legal teardown is:
    //   1. `allocator.free(huff_off16hi_data)` (the single real free), then
    //   2. null out ALL SIX off16 slots so a follow-up encode lands on a
    //      clean context — sizes/data/offsets for hi AND lo:
    //        huff_off16hi_sizes, huff_off16hi_data, huff_off16hi_offsets,
    //        huff_off16lo_sizes, huff_off16lo_data, huff_off16lo_offsets.
    // See `deinit` below and `fast_framed.zig` for the canonical pattern.
    // Anyone introducing a new free site must honor that rule or the alias
    // double-frees.
    huff_off16hi_sizes: ?[]u32 = null,
    huff_off16hi_data: ?[]u8 = null, // OWNS the shared buffer
    huff_off16hi_offsets: ?[]u32 = null,
    huff_off16lo_sizes: ?[]u32 = null,
    huff_off16lo_data: ?[]u8 = null, // NON-OWNING alias of huff_off16hi_data
    huff_off16lo_offsets: ?[]u32 = null,

    huff_lit_sizes: ?[]u32 = null,
    huff_lit_data: ?[]u8 = null,
    huff_lit_offsets: ?[]u32 = null,
    huff_tok_sizes: ?[]u32 = null,
    huff_tok_data: ?[]u8 = null,
    huff_tok_offsets: ?[]u32 = null,

    /// Free every owned device + host buffer and reset every field to
    /// its default. Intended for a per-handle library API teardown; the
    /// long-lived `driver.g_default` singleton in the current CLI / C ABI
    /// never calls this (its lifetime is the process).
    ///
    /// Honors the huff_off16 OWNERSHIP RULE above: hi owns the shared
    /// allocation, lo is a non-owning alias.
    pub fn deinit(self: *EncodeContext, allocator: std.mem.Allocator) void {
        const free_dev = struct {
            fn f(ptr: *CUdeviceptr, sz: *usize) void {
                if (ptr.* != 0) {
                    // Free failure on a pointer we know is valid (we
                    // allocated it via cuMemAlloc and never freed it)
                    // means the driver/context is dying — there's
                    // nothing useful we can do in a deinit path, so
                    // ignore the CUresult.
                    if (ffi.cuMemFree_fn) |free_fn| _ = free_fn(ptr.*);
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
        free_dev(&self.d_descs_persist, &self.d_descs_size);
        free_dev(&self.d_hash_persist, &self.d_hash_size);
        free_dev(&self.d_sizes_persist, &self.d_sizes_size);
        free_dev(&self.d_huff_descs_persist, &self.d_huff_descs_size);
        free_dev(&self.d_huff_cl_persist, &self.d_huff_cl_size);
        free_dev(&self.d_huff_codes_persist, &self.d_huff_codes_size);
        free_dev(&self.d_huff_scratch_persist, &self.d_huff_scratch_size);
        free_dev(&self.d_huff_out_persist, &self.d_huff_out_size);
        free_dev(&self.d_huff_sizes_persist, &self.d_huff_sizes_size);
        free_dev(&self.d_asm_huff_lit, &self.d_asm_huff_lit_size);
        free_dev(&self.d_asm_huff_tok, &self.d_asm_huff_tok_size);
        free_dev(&self.d_asm_huff_off16, &self.d_asm_huff_off16_size);
        free_dev(&self.d_asm_descs, &self.d_asm_descs_size);
        free_dev(&self.d_asm_out, &self.d_asm_out_size);
        free_dev(&self.d_asm_sizes, &self.d_asm_sizes_size);

        if (self.assembled_data) |s| { allocator.free(s); self.assembled_data = null; }
        if (self.assembled_offsets) |s| { allocator.free(s); self.assembled_offsets = null; }
        if (self.assembled_sizes) |s| { allocator.free(s); self.assembled_sizes = null; }

        // huff_off16: hi OWNS the shared byte buffer; lo is a non-owning
        // alias. Drop the alias FIRST, then free the owner.
        self.huff_off16lo_data = null;
        if (self.huff_off16hi_data) |s| { allocator.free(s); self.huff_off16hi_data = null; }
        if (self.huff_off16hi_sizes) |s| { allocator.free(s); self.huff_off16hi_sizes = null; }
        if (self.huff_off16hi_offsets) |s| { allocator.free(s); self.huff_off16hi_offsets = null; }
        if (self.huff_off16lo_sizes) |s| { allocator.free(s); self.huff_off16lo_sizes = null; }
        if (self.huff_off16lo_offsets) |s| { allocator.free(s); self.huff_off16lo_offsets = null; }

        if (self.huff_lit_sizes) |s| { allocator.free(s); self.huff_lit_sizes = null; }
        if (self.huff_lit_data) |s| { allocator.free(s); self.huff_lit_data = null; }
        if (self.huff_lit_offsets) |s| { allocator.free(s); self.huff_lit_offsets = null; }
        if (self.huff_tok_sizes) |s| { allocator.free(s); self.huff_tok_sizes = null; }
        if (self.huff_tok_data) |s| { allocator.free(s); self.huff_tok_data = null; }
        if (self.huff_tok_offsets) |s| { allocator.free(s); self.huff_tok_offsets = null; }

        self.pending_timings.deinit(std.heap.page_allocator);
        self.last_timings.deinit(std.heap.page_allocator);
    }
};

/// Reallocate the device buffer at `ptr` if its current size is below
/// `needed`; otherwise reuse. Caller passes companion `cur` (bytes) which
/// we update on success. Returns false only when the CUDA alloc fails.
pub fn ensureBuf(ptr: *CUdeviceptr, cur: *usize, needed: usize) bool {
    if (cur.* >= needed) return true;
    const free_fn = ffi.cuMemFree_fn orelse return false;
    const alloc_fn = ffi.cuMemAlloc_fn orelse return false;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    cur.* = 0;
    if (alloc_fn(ptr, needed) != ffi.CUDA_SUCCESS) return false;
    cur.* = needed;
    return true;
}

/// Copy `dst.len` bytes from a device address into the host slice `dst`.
/// Used by 4d Phase 3 D2D encode fallback paths that need a few bytes
/// of the device-resident input on the host.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    if (!module_loader.init()) return false;
    const f = ffi.cuMemcpyDtoH_fn orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == ffi.CUDA_SUCCESS;
}
