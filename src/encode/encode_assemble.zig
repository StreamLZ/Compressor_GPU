//! Device-resident frame assembly.
//!
//! Two entrypoints called back-to-back from `fast_framed.compressFramedOne`:
//!   * `gpuAssembleFrameImpl` - measures every sub-chunk's assembled
//!     payload size, prefix-sums the sizes, then writes [3-byte hdr]
//!     [payload] for every sub-chunk into `d_asm_out`. Publishes the
//!     per-chunk offset/size tables on the context; the bytes themselves
//!     stay on the GPU.
//!   * `gpuFrameAssembleImpl` - pure-D2D writer that splices the
//!     device-resident assembled blocks plus host-precomputed per-chunk
//!     layout into a complete StreamLZ frame straight into the caller's
//!     device buffer.

const std = @import("std");
const cuda_ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = cuda_ffi.CUdeviceptr;
const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;
const AssembleDesc = encode_context.AssembleDesc;

/// Device-resident frame assembly. Runs after `gpuCompressImpl` (raw
/// streams resident in `d_output_persist`) and the three GPU Huffman
/// passes (bodies resident in `d_asm_huff_{lit,tok,off16}`). Assembles
/// every sub-chunk's `[3-byte header][payload]` on the GPU into
/// `d_asm_out` and publishes the per-chunk offset/size index tables in
/// `assembled_offsets` / `assembled_sizes` for the frame writer.
/// Returns false on any prerequisite failure - the caller treats this
/// as "destination too small" and propagates.
pub fn gpuAssembleFrameImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    if (!module_loader.init()) return false;
    if (module_loader.assemble_measure_fn == 0 or module_loader.assemble_write_fn == 0) return false;
    const n: u32 = @intCast(chunk_descs.len);
    if (n == 0) return false;

    // Per-stream Huffman metadata. L1/L2 do not run the Huffman pass so
    // all six per-stream slots are null; in that case the assembly
    // kernel sees `huff_*_size == 0` for every sub-chunk and emits raw
    // (chunk-type=0) for that stream. Zero-fill scratch arrays let us
    // keep one kernel-side branch instead of two distinct code paths.
    const zero_offs = allocator.alloc(u32, n) catch return false;
    defer allocator.free(zero_offs);
    @memset(zero_offs, 0);
    const zero_sizes = allocator.alloc(u32, n) catch return false;
    defer allocator.free(zero_sizes);
    @memset(zero_sizes, 0);

    const huff_lit_offsets = self.huff_lit_offsets orelse zero_offs;
    const huff_lit_sizes = self.huff_lit_sizes orelse zero_sizes;
    const huff_tok_offsets = self.huff_tok_offsets orelse zero_offs;
    const huff_tok_sizes = self.huff_tok_sizes orelse zero_sizes;
    const huff_off16hi_offsets = self.huff_off16hi_offsets orelse zero_offs;
    const huff_off16hi_sizes = self.huff_off16hi_sizes orelse zero_sizes;
    const huff_off16lo_offsets = self.huff_off16lo_offsets orelse zero_offs;
    const huff_off16lo_sizes = self.huff_off16lo_sizes orelse zero_sizes;
    if (huff_lit_offsets.len < n or huff_tok_offsets.len < n or
        huff_off16hi_offsets.len < n or huff_off16lo_offsets.len < n) return false;

    const h2d = cuda_ffi.cuMemcpyHtoD_fn orelse return false;
    const d2h = cuda_ffi.cuMemcpyDtoH_fn orelse return false;
    const launch = cuda_ffi.cuLaunchKernel_fn orelse return false;
    const sync = cuda_ffi.cuCtxSynchronize_fn orelse return false;

    // Build per-sub-chunk descriptors (out_offset filled after pass 1).
    var descs = allocator.alloc(AssembleDesc, n) catch return false;
    defer allocator.free(descs);
    for (0..n) |i| {
        descs[i] = .{
            .raw_offset = chunk_descs[i].dst_offset,
            .raw_size = comp_sizes[i],
            .huff_lit_offset = huff_lit_offsets[i],
            .huff_lit_size = huff_lit_sizes[i],
            .huff_tok_offset = huff_tok_offsets[i],
            .huff_tok_size = huff_tok_sizes[i],
            .huff_off16hi_offset = huff_off16hi_offsets[i],
            .huff_off16hi_size = huff_off16hi_sizes[i],
            .huff_off16lo_offset = huff_off16lo_offsets[i],
            .huff_off16lo_size = huff_off16lo_sizes[i],
            .sub_decomp_size = chunk_descs[i].src_size,
            .init_bytes = if (chunk_descs[i].is_first != 0) encode_context.INITIAL_LITERAL_COPY_BYTES else 0,
            .out_offset = 0,
        };
    }

    const desc_bytes: usize = @as(usize, n) * @sizeOf(AssembleDesc);
    const sizes_bytes: usize = @as(usize, n) * 4;
    if (!encode_context.ensureBuf(&self.d_asm_descs, &self.d_asm_descs_size, desc_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_asm_sizes, &self.d_asm_sizes_size, sizes_bytes)) return false;
    if (h2d(self.d_asm_descs, @ptrCast(descs.ptr), desc_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    if (sync() != cuda_ffi.CUDA_SUCCESS) return false;

    // Kernel-arg locals. Each parameter is taken by `&local`, so every
    // value the kernel reads needs its own addressable slot here.
    var arg_raw = self.d_output_persist;
    var arg_huff_lit = self.d_asm_huff_lit;
    var arg_huff_tok = self.d_asm_huff_tok;
    var arg_huff_off16 = self.d_asm_huff_off16;
    var arg_descs = self.d_asm_descs;
    var arg_sizes = self.d_asm_sizes;
    var arg_n = n;
    var extra = [_]?*anyopaque{null};

    // Pass 1 - measure each sub-chunk's assembled payload size.
    var measure_params = [_]?*anyopaque{
        @ptrCast(&arg_raw),       @ptrCast(&arg_huff_lit), @ptrCast(&arg_huff_tok),
        @ptrCast(&arg_huff_off16),@ptrCast(&arg_descs),    @ptrCast(&arg_sizes),
        @ptrCast(&arg_n),
    };
    const t_am = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzAssembleMeasureKernel", 0);
    defer gpu_decode.endKernelTiming(t_am, 0);
    if (launch(module_loader.assemble_measure_fn, n, 1, 1, 32, 1, 1, 0, 0, &measure_params, &extra) != cuda_ffi.CUDA_SUCCESS) return false;
    if (sync() != cuda_ffi.CUDA_SUCCESS) return false;

    const enc_sizes = allocator.alloc(u32, n) catch return false;
    defer allocator.free(enc_sizes);
    if (d2h(@ptrCast(enc_sizes.ptr), self.d_asm_sizes, sizes_bytes) != cuda_ffi.CUDA_SUCCESS) return false;

    // Prefix-sum: each sub-chunk block is 3 (header) + enc_n bytes.
    var total: u32 = 0;
    for (0..n) |i| {
        // `enc_sizes[i] == 0` is unambiguously a measure-kernel parse
        // failure: the measure pass writes at least the framing bytes
        // for any valid sub-chunk descriptor (raw_size > 0 here - we
        // only reach this path with non-empty chunks fed from
        // gpuCompressImpl). A zero size means the kernel couldn't parse
        // the raw stream layout (corrupted header bytes), so bail out
        // rather than write a malformed block. No "empty sub-chunk"
        // interpretation: zero-length sub-chunks never get encoded.
        if (enc_sizes[i] == 0) return false;
        descs[i].out_offset = total;
        total += 3 + enc_sizes[i];
    }

    // Pass 2 - write [3-byte header][payload] for every sub-chunk.
    if (h2d(self.d_asm_descs, @ptrCast(descs.ptr), desc_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    if (!encode_context.ensureBuf(&self.d_asm_out, &self.d_asm_out_size, @max(total, 1))) return false;
    var arg_out = self.d_asm_out;
    var write_params = [_]?*anyopaque{
        @ptrCast(&arg_raw),       @ptrCast(&arg_huff_lit), @ptrCast(&arg_huff_tok),
        @ptrCast(&arg_huff_off16),@ptrCast(&arg_descs),    @ptrCast(&arg_out),
        @ptrCast(&arg_n),
    };
    const t_aw = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzAssembleWriteKernel", 0);
    defer gpu_decode.endKernelTiming(t_aw, 0);
    if (launch(module_loader.assemble_write_fn, n, 1, 1, 32, 1, 1, 0, 0, &write_params, &extra) != cuda_ffi.CUDA_SUCCESS) return false;
    if (sync() != cuda_ffi.CUDA_SUCCESS) return false;

    // Publish the per-chunk offset+size index tables; the assembled
    // bytes themselves stay in `self.d_asm_out` for the frame writer
    // to read in place.
    const off = allocator.alloc(u32, n) catch return false;
    const sz = allocator.alloc(u32, n) catch {
        allocator.free(off);
        return false;
    };
    for (0..n) |i| {
        off[i] = descs[i].out_offset;
        sz[i] = 3 + enc_sizes[i];
    }
    self.assembled_offsets = off;
    self.assembled_sizes = sz;
    return true;
}

/// Pure-D2D frame writer. Takes the device-resident assembled sub-chunk
/// blocks (already in `self.d_asm_out` from `gpuAssembleFrameImpl`) plus
/// host-precomputed per-chunk layout, launches `slzFrameAssembleKernel`,
/// and writes the complete StreamLZ frame straight into `d_output` on
/// device. Returns the total frame byte count, or null on failure.
///
/// Self-contained mode (the only mode `slzCompress` produces) is implied
/// by `n_chunks > 1`: the kernel emits an `(n_chunks - 1) *
/// SC_TAIL_PER_CHUNK_BYTES` SC tail block right after the per-chunk
/// payloads.

/// Bytes the frame assembler prepends before the per-chunk payloads -
/// pre-formed by the caller (frame header + block header) plus the
/// 2-byte internal block header that gets repeated at each chunk boundary.
pub const FramePreamble = struct {
    prefix_bytes: []const u8,
    internal_hdr0: u8,
    internal_hdr1: u8,
};

/// Layout description of the per-chunk asm payloads the kernel will splice.
/// `per_chunk_asm_off` and `per_chunk_asm_size` are parallel arrays of
/// length `n_chunks`. `eff_chunk_size` and `src_len` describe the source
/// chunking so the kernel can rebuild the SC tail prefix table.
pub const ChunkLayout = struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    per_chunk_asm_off: []const u32,
    per_chunk_asm_size: []const u32,
};

pub fn gpuFrameAssembleImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    layout: ChunkLayout,
    preamble: FramePreamble,
    d_input_dev: u64,
    d_output: u64,
) ?u32 {
    if (!module_loader.init()) return null;
    if (module_loader.frame_assemble_fn == 0) return null;
    if (self.d_asm_out == 0) return null;
    const n_chunks = layout.n_chunks;
    const eff_chunk_size = layout.eff_chunk_size;
    const src_len = layout.src_len;
    const per_chunk_asm_off = layout.per_chunk_asm_off;
    const per_chunk_asm_size = layout.per_chunk_asm_size;
    const prefix_bytes = preamble.prefix_bytes;
    const internal_hdr0 = preamble.internal_hdr0;
    const internal_hdr1 = preamble.internal_hdr1;
    if (n_chunks == 0 or per_chunk_asm_off.len < n_chunks or per_chunk_asm_size.len < n_chunks) return null;

    const h2d = cuda_ffi.cuMemcpyHtoD_fn orelse return null;
    const launch = cuda_ffi.cuLaunchKernel_fn orelse return null;
    const sync = cuda_ffi.cuCtxSynchronize_fn orelse return null;

    // Build per-chunk dst offset table on host. Compressed chunks pay
    // `CHUNK_INTERNAL_HDR_BYTES` (2-byte internal block header + 4-byte
    // chunk header); uncompressed chunks — flagged by the sentinel
    // `UNCOMPRESSED_CHUNK_MARKER` in `per_chunk_asm_off` — pay only
    // `UNCOMPRESSED_CHUNK_HDR_BYTES`.
    const per_chunk_dst_buf = allocator.alloc(u32, n_chunks) catch return null;
    defer allocator.free(per_chunk_dst_buf);
    var pos: u32 = @intCast(prefix_bytes.len);
    for (0..n_chunks) |i| {
        per_chunk_dst_buf[i] = pos;
        const overhead: u32 = if (per_chunk_asm_off[i] == encode_context.UNCOMPRESSED_CHUNK_MARKER)
            encode_context.UNCOMPRESSED_CHUNK_HDR_BYTES
        else
            encode_context.CHUNK_INTERNAL_HDR_BYTES;
        pos += overhead + per_chunk_asm_size[i];
    }
    const sc_tail_off: u32 = pos;
    const sc_tail_bytes: u32 = if (n_chunks > 1) (n_chunks - 1) * encode_context.SC_TAIL_PER_CHUNK_BYTES else 0;
    pos += sc_tail_bytes;
    const end_mark_off: u32 = pos;
    pos += 4;
    const total_frame_size: u32 = pos;

    const ent_bytes: usize = @as(usize, n_chunks) * 4;
    if (!encode_context.ensureBuf(&self.d_frame_chunk_dst, &self.d_frame_chunk_dst_size, ent_bytes)) return null;
    if (!encode_context.ensureBuf(&self.d_frame_asm_offsets, &self.d_frame_asm_offsets_size, ent_bytes)) return null;
    if (!encode_context.ensureBuf(&self.d_frame_asm_chunk_sz, &self.d_frame_asm_chunk_sz_size, ent_bytes)) return null;
    if (!encode_context.ensureBuf(&self.d_frame_prefix_bytes, &self.d_frame_prefix_bytes_size, prefix_bytes.len)) return null;

    if (h2d(self.d_frame_chunk_dst, @ptrCast(per_chunk_dst_buf.ptr), ent_bytes) != cuda_ffi.CUDA_SUCCESS) return null;
    if (h2d(self.d_frame_asm_offsets, @ptrCast(per_chunk_asm_off.ptr), ent_bytes) != cuda_ffi.CUDA_SUCCESS) return null;
    if (h2d(self.d_frame_asm_chunk_sz, @ptrCast(per_chunk_asm_size.ptr), ent_bytes) != cuda_ffi.CUDA_SUCCESS) return null;
    if (h2d(self.d_frame_prefix_bytes, @ptrCast(prefix_bytes.ptr), prefix_bytes.len) != cuda_ffi.CUDA_SUCCESS) return null;

    var k_input = d_input_dev;
    var k_asm_out = self.d_asm_out;
    var k_asm_offs = self.d_frame_asm_offsets;
    var k_asm_sizes = self.d_frame_asm_chunk_sz;
    var k_chunk_dst = self.d_frame_chunk_dst;
    var k_prefix = self.d_frame_prefix_bytes;
    var k_prefix_size: u32 = @intCast(prefix_bytes.len);
    var k_hdr0 = internal_hdr0;
    var k_hdr1 = internal_hdr1;
    var k_n = n_chunks;
    var k_eff = eff_chunk_size;
    var k_src_len = src_len;
    var k_sc_off = sc_tail_off;
    var k_end_off = end_mark_off;
    var k_dst = d_output;
    var params = [_]?*anyopaque{
        @ptrCast(&k_input),    @ptrCast(&k_asm_out),  @ptrCast(&k_asm_offs),
        @ptrCast(&k_asm_sizes),@ptrCast(&k_chunk_dst),@ptrCast(&k_prefix),
        @ptrCast(&k_prefix_size),
        @ptrCast(&k_hdr0),     @ptrCast(&k_hdr1),
        @ptrCast(&k_n),        @ptrCast(&k_eff),      @ptrCast(&k_src_len),
        @ptrCast(&k_sc_off),   @ptrCast(&k_end_off),  @ptrCast(&k_dst),
    };
    var extra = [_]?*anyopaque{null};

    // Grid: n_chunks blocks for per-chunk writes + 1 block for prefix/tail/end mark.
    // 128 threads per block - enough for cooperative copies up to ~few KB/iter.
    // Defensive u32-overflow guard: WALK_MAX_CHUNKS caps n_chunks lower
    // (16384), but a future bump could theoretically reach u32 max here.
    if (n_chunks == std.math.maxInt(u32)) return null;
    const grid_x: u32 = n_chunks + 1;
    // Heavy phase: ride the caller's stream when slzCompressAsync set
    // work_stream so cudaStreamSynchronize on it waits for the frame
    // bytes to be in d_output.
    const heavy_stream: usize = self.work_stream;
    const t_fa = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzFrameAssembleKernel", heavy_stream);
    defer gpu_decode.endKernelTiming(t_fa, heavy_stream);
    if (launch(module_loader.frame_assemble_fn, grid_x, 1, 1, 128, 1, 1, 0, heavy_stream, &params, &extra) != cuda_ffi.CUDA_SUCCESS) return null;
    // In async mode we leave the kernel in flight on the caller's stream
    // (their cudaStreamSynchronize is the sync point); else block here.
    if (heavy_stream == 0) {
        if (sync() != cuda_ffi.CUDA_SUCCESS) return null;
    }

    return total_frame_size;
}
