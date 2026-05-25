//! Device-resident frame assembly (roadmap 4d).
//!
//! Two entrypoints:
//!   * `gpuAssembleFrameImpl` — runs after LZ + the three Huffman passes
//!     left their bodies device-resident in d_asm_huff_{lit,tok,off16};
//!     measures, prefix-sums, then writes [3-byte hdr][payload] for every
//!     sub-chunk and publishes the host-side packed result.
//!   * `gpuFrameAssembleImpl` — 4d step 8 pure-D2D writer that splices
//!     the assembled sub-chunk blocks plus host-precomputed layout into
//!     a complete StreamLZ frame straight into the caller's device buffer.

const std = @import("std");
const ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const ec = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = ffi.CUdeviceptr;
const EncodeContext = ec.EncodeContext;
const CompressChunkDesc = ec.CompressChunkDesc;
const AssembleDesc = ec.AssembleDesc;

/// Device-resident frame assembly (roadmap 4d). Runs after gpuCompressImpl
/// (raw streams resident in d_output) and the three GPU Huffman passes
/// (run with huff_keep_device — bodies resident in d_asm_huff_*, no host
/// bounce). Assembles every sub-chunk's [3-byte header][payload] on the
/// GPU and publishes the packed host-side result in `assembled_*` for the
/// frame assembler to splice. Returns false — caller keeps the CPU path —
/// if any prerequisite is absent.
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

    // Per-stream metadata (offsets + sizes) — always host-side, small;
    // present iff the matching Huffman pass succeeded.
    const hl_off = self.huff_lit_offsets orelse return false;
    const hl_sz = self.huff_lit_sizes orelse return false;
    const ht_off = self.huff_tok_offsets orelse return false;
    const ht_sz = self.huff_tok_sizes orelse return false;
    const hoh_off = self.huff_off16hi_offsets orelse return false;
    const hoh_sz = self.huff_off16hi_sizes orelse return false;
    const hol_off = self.huff_off16lo_offsets orelse return false;
    const hol_sz = self.huff_off16lo_sizes orelse return false;
    if (hl_off.len < n or ht_off.len < n or hoh_off.len < n or hol_off.len < n) return false;

    const h2d = ffi.cuMemcpyHtoD_fn orelse return false;
    const d2h = ffi.cuMemcpyDtoH_fn orelse return false;
    const launch = ffi.cuLaunchKernel_fn orelse return false;
    const sync = ffi.cuCtxSynchronize_fn orelse return false;

    // The three GPU Huffman passes already left their bodies device-
    // resident in d_asm_huff_{lit,tok,off16} (huff_keep_device set by the
    // encoder before they ran) — no host bounce.

    // Build per-sub-chunk descriptors (out_offset filled after pass 1).
    var descs = allocator.alloc(AssembleDesc, n) catch return false;
    defer allocator.free(descs);
    for (0..n) |i| {
        descs[i] = .{
            .raw_offset = chunk_descs[i].dst_offset,
            .raw_size = comp_sizes[i],
            .huff_lit_offset = hl_off[i],
            .huff_lit_size = hl_sz[i],
            .huff_tok_offset = ht_off[i],
            .huff_tok_size = ht_sz[i],
            .huff_off16hi_offset = hoh_off[i],
            .huff_off16hi_size = hoh_sz[i],
            .huff_off16lo_offset = hol_off[i],
            .huff_off16lo_size = hol_sz[i],
            .sub_decomp_size = chunk_descs[i].src_size,
            .init_bytes = if (chunk_descs[i].is_first != 0) 8 else 0,
            .out_offset = 0,
        };
    }

    const desc_bytes: usize = @as(usize, n) * @sizeOf(AssembleDesc);
    const sizes_bytes: usize = @as(usize, n) * 4;
    if (!ec.ensureBuf(&self.d_asm_descs, &self.d_asm_descs_size, desc_bytes)) return false;
    if (!ec.ensureBuf(&self.d_asm_sizes, &self.d_asm_sizes_size, sizes_bytes)) return false;
    _ = h2d(self.d_asm_descs, @ptrCast(descs.ptr), desc_bytes);
    _ = sync();

    var p_raw = self.d_output_persist;
    var p_hl = self.d_asm_huff_lit;
    var p_ht = self.d_asm_huff_tok;
    var p_ho = self.d_asm_huff_off16;
    var p_descs = self.d_asm_descs;
    var p_sizes = self.d_asm_sizes;
    var p_n = n;
    var extra = [_]?*anyopaque{null};

    // Pass 1 — measure each sub-chunk's assembled payload size.
    var m_params = [_]?*anyopaque{
        @ptrCast(&p_raw),   @ptrCast(&p_hl),    @ptrCast(&p_ht),
        @ptrCast(&p_ho),    @ptrCast(&p_descs), @ptrCast(&p_sizes),
        @ptrCast(&p_n),
    };
    const t_am = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzAssembleMeasureKernel", 0);
    if (launch(module_loader.assemble_measure_fn, n, 1, 1, 32, 1, 1, 0, 0, &m_params, &extra) != ffi.CUDA_SUCCESS) return false;
    gpu_decode.endKernelTiming(t_am, 0);
    if (sync() != ffi.CUDA_SUCCESS) return false;

    const enc_sizes = allocator.alloc(u32, n) catch return false;
    defer allocator.free(enc_sizes);
    _ = d2h(@ptrCast(enc_sizes.ptr), self.d_asm_sizes, sizes_bytes);

    // Prefix-sum: each sub-chunk block is 3 (header) + enc_n bytes.
    var total: u32 = 0;
    for (0..n) |i| {
        if (enc_sizes[i] == 0) return false; // kernel parse error
        descs[i].out_offset = total;
        total += 3 + enc_sizes[i];
    }

    // Pass 2 — write [3-byte header][payload] for every sub-chunk.
    _ = h2d(self.d_asm_descs, @ptrCast(descs.ptr), desc_bytes);
    if (!ec.ensureBuf(&self.d_asm_out, &self.d_asm_out_size, @max(total, 1))) return false;
    var p_out = self.d_asm_out;
    var w_params = [_]?*anyopaque{
        @ptrCast(&p_raw),   @ptrCast(&p_hl),    @ptrCast(&p_ht),
        @ptrCast(&p_ho),    @ptrCast(&p_descs), @ptrCast(&p_out),
        @ptrCast(&p_n),
    };
    const t_aw = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzAssembleWriteKernel", 0);
    if (launch(module_loader.assemble_write_fn, n, 1, 1, 32, 1, 1, 0, 0, &w_params, &extra) != ffi.CUDA_SUCCESS) return false;
    gpu_decode.endKernelTiming(t_aw, 0);
    if (sync() != ffi.CUDA_SUCCESS) return false;

    // Download the assembled blocks; publish the host-side result.
    const assembled = allocator.alloc(u8, total) catch return false;
    const off = allocator.alloc(u32, n) catch {
        allocator.free(assembled);
        return false;
    };
    const sz = allocator.alloc(u32, n) catch {
        allocator.free(assembled);
        allocator.free(off);
        return false;
    };
    _ = d2h(@ptrCast(assembled.ptr), self.d_asm_out, total);
    for (0..n) |i| {
        off[i] = descs[i].out_offset;
        sz[i] = 3 + enc_sizes[i];
    }
    self.assembled_data = assembled;
    self.assembled_offsets = off;
    self.assembled_sizes = sz;
    return true;
}

/// 4d step 8 — Pure-D2D frame writer. Takes the device-resident assembled
/// sub-chunk blocks (already in `self.d_asm_out` from gpuAssembleFrameImpl)
/// plus host-precomputed per-chunk layout info, launches
/// slzFrameAssembleKernel, and writes the complete StreamLZ frame to
/// `d_output` on device. Returns the total frame byte count or null on
/// failure.
///
/// Caller must supply:
///   `prefix_bytes` — pre-formed frame_hdr + block_hdr (~30-40 B).
///   `per_chunk_asm_off`  — start of each chunk's sub-chunk(s) in d_asm_out.
///   `per_chunk_asm_size` — total asm bytes for each chunk (sum across its sub-chunks).
///   `internal_hdr0/1`    — the 2-byte internal block header (same for every chunk).
///   `eff_chunk_size`     — source chunk stride in bytes (for SC tail src offsets).
///   `d_input_dev`        — device source (for SC tail prefix bytes).
///
/// Self-contained (`sc_tail_count = n_chunks - 1` when n_chunks > 1, else 0)
/// is inferred from the chunk count; SC mode is the only one slzCompress
/// produces.
pub fn gpuFrameAssembleImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    prefix_bytes: []const u8,
    internal_hdr0: u8,
    internal_hdr1: u8,
    per_chunk_asm_off: []const u32,
    per_chunk_asm_size: []const u32,
    d_input_dev: u64,
    d_output: u64,
) ?u32 {
    if (!module_loader.init()) return null;
    if (module_loader.frame_assemble_fn == 0) return null;
    if (self.d_asm_out == 0) return null;
    if (n_chunks == 0 or per_chunk_asm_off.len < n_chunks or per_chunk_asm_size.len < n_chunks) return null;

    const h2d = ffi.cuMemcpyHtoD_fn orelse return null;
    const launch = ffi.cuLaunchKernel_fn orelse return null;
    const sync = ffi.cuCtxSynchronize_fn orelse return null;

    // Build per-chunk dst offset table on host (prefix sum of 6 + asm_size).
    const per_chunk_dst_buf = allocator.alloc(u32, n_chunks) catch return null;
    defer allocator.free(per_chunk_dst_buf);
    var pos: u32 = @intCast(prefix_bytes.len);
    for (0..n_chunks) |i| {
        per_chunk_dst_buf[i] = pos;
        pos += 6 + per_chunk_asm_size[i];
    }
    const sc_tail_off: u32 = pos;
    const sc_tail_bytes: u32 = if (n_chunks > 1) (n_chunks - 1) * 8 else 0;
    pos += sc_tail_bytes;
    const end_mark_off: u32 = pos;
    pos += 4;
    const total_frame_size: u32 = pos;

    const ent_bytes: usize = @as(usize, n_chunks) * 4;
    if (!ec.ensureBuf(&self.d_frame_chunk_dst, &self.d_frame_chunk_dst_size, ent_bytes)) return null;
    if (!ec.ensureBuf(&self.d_frame_asm_offsets, &self.d_frame_asm_offsets_size, ent_bytes)) return null;
    if (!ec.ensureBuf(&self.d_frame_asm_chunk_sz, &self.d_frame_asm_chunk_sz_size, ent_bytes)) return null;
    if (!ec.ensureBuf(&self.d_frame_prefix_bytes, &self.d_frame_prefix_bytes_size, prefix_bytes.len)) return null;

    _ = h2d(self.d_frame_chunk_dst, @ptrCast(per_chunk_dst_buf.ptr), ent_bytes);
    _ = h2d(self.d_frame_asm_offsets, @ptrCast(per_chunk_asm_off.ptr), ent_bytes);
    _ = h2d(self.d_frame_asm_chunk_sz, @ptrCast(per_chunk_asm_size.ptr), ent_bytes);
    _ = h2d(self.d_frame_prefix_bytes, @ptrCast(prefix_bytes.ptr), prefix_bytes.len);

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
    // 128 threads per block — enough for cooperative copies up to ~few KB/iter.
    const grid_x: u32 = n_chunks + 1;
    // Heavy phase: ride the caller's stream when slzCompressAsync set
    // work_stream so cudaStreamSynchronize on it waits for the frame
    // bytes to be in d_output.
    const heavy_stream: usize = self.work_stream;
    const t_fa = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzFrameAssembleKernel", heavy_stream);
    if (launch(module_loader.frame_assemble_fn, grid_x, 1, 1, 128, 1, 1, 0, heavy_stream, &params, &extra) != ffi.CUDA_SUCCESS) return null;
    gpu_decode.endKernelTiming(t_fa, heavy_stream);
    // In async mode we leave the kernel in flight on the caller's stream
    // (their cudaStreamSynchronize is the sync point); else block here.
    if (heavy_stream == 0) {
        if (sync() != ffi.CUDA_SUCCESS) return null;
    }

    return total_frame_size;
}
