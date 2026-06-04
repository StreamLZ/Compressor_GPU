//! 1:1 port of src/encode/encode_assemble.zig.
//!
//! Two entrypoints called back-to-back from the CPU framer:
//!   * `gpuAssembleFrameImpl` — runs the per-sub-chunk measure + write
//!     passes (assemble_measure + assemble_write).
//!   * `gpuFrameAssembleImpl` — dispatches the final frame writer
//!     (frame_assemble) that splices device-resident assembled blocks
//!     plus host-precomputed per-chunk layout into a complete StreamLZ
//!     frame straight into the caller's device buffer.
//!
//! All device ops funnel through `vulkan_api.procs.*` — no direct
//! vk*/vma calls in this file.

const std = @import("std");
const vk = @import("../decode/vulkan_api.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const VkDeviceBuffer = vk.VkDeviceBuffer;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;
const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;
const AssembleDesc = encode_context.AssembleDesc;

/// CUDA reference: src/encode/encode_assemble.zig:33-170. Device-resident
/// per-sub-chunk assembly: pass A measures, pass B writes
/// `[3-byte header][payload]` for every sub-chunk into `d_asm_out`. The
/// per-chunk offset / size index tables get published in
/// `assembled_offsets` / `assembled_sizes`. Returns false on any
/// prerequisite failure.
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
    // all six per-stream slots are null; the assembly kernel sees
    // `huff_*_size == 0` for every sub-chunk and emits raw (chunk-type=0)
    // for that stream. A single zero-filled scratch buffer covers both
    // the offsets and sizes orelse fallbacks.
    const zero_buf = allocator.alloc(u32, n) catch return false;
    defer allocator.free(zero_buf);
    @memset(zero_buf, 0);

    const huff_lit_offsets = self.huff_lit_offsets orelse zero_buf;
    const huff_lit_sizes = self.huff_lit_sizes orelse zero_buf;
    const huff_tok_offsets = self.huff_tok_offsets orelse zero_buf;
    const huff_tok_sizes = self.huff_tok_sizes orelse zero_buf;
    const huff_off16hi_offsets = self.huff_off16hi_offsets orelse zero_buf;
    const huff_off16hi_sizes = self.huff_off16hi_sizes orelse zero_buf;
    const huff_off16lo_offsets = self.huff_off16lo_offsets orelse zero_buf;
    const huff_off16lo_sizes = self.huff_off16lo_sizes orelse zero_buf;
    if (huff_lit_offsets.len < n or huff_tok_offsets.len < n or
        huff_off16hi_offsets.len < n or huff_off16lo_offsets.len < n) return false;

    const h2d = vk.procs.h2d orelse return false;
    const d2h = vk.procs.d2h orelse return false;
    const launch = vk.procs.launch_kernel orelse return false;
    const sync = vk.procs.ctx_sync orelse return false;

    // Build per-sub-chunk descriptors (out_offset filled after pass A).
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
    if (h2d(self.d_asm_descs, @ptrCast(descs.ptr), desc_bytes) != VK_SUCCESS_RC) return false;
    if (sync() != VK_SUCCESS_RC) return false;

    // Kernel-arg locals. Each parameter is taken by `&local`, so every
    // value the kernel reads needs its own addressable slot here.
    // VK adaptation: at L1 the huff_* device pointers are 0 (no Huffman
    // encode pass populated them). CUDA's cuLaunchKernel happily passes
    // null device pointers; Vulkan demands a real bound VkBuffer at
    // every SSBO slot the .comp declares. For unused huff slots
    // (binding 1..3) substitute `d_output_persist` (always non-null at
    // this point); the kernel reads from these only when the matching
    // `huff_*_size` desc field is non-zero — at L1 they're all zero so
    // the substituted buffer is never actually read.
    var arg_raw = self.d_output_persist;
    var arg_huff_lit = if (self.d_asm_huff_lit != 0) self.d_asm_huff_lit else self.d_output_persist;
    var arg_huff_tok = if (self.d_asm_huff_tok != 0) self.d_asm_huff_tok else self.d_output_persist;
    var arg_huff_off16 = if (self.d_asm_huff_off16 != 0) self.d_asm_huff_off16 else self.d_output_persist;
    var arg_descs = self.d_asm_descs;
    var arg_sizes = self.d_asm_sizes;
    var arg_n = n;
    var extra = [_]?*anyopaque{null};

    // Pass A — measure each sub-chunk's assembled payload size.
    // assemble_measure: n_bindings=7 (d_raw, d_huff_lit, d_huff_tok,
    // d_huff_off16, descs, enc_sizes, scratch_u8 placeholder),
    // push_constant_size=4 (n_subchunks).
    // VK adaptation: binding 6 is the scratch_u8 placeholder the .comp
    // declares purely to satisfy glslc's static l-value check on the
    // disabled-write branches in assembleSubChunk. The kernel never
    // touches it at runtime, but Vulkan still requires a real SSBO
    // bound to the slot — reuse `self.d_asm_sizes` as a non-null
    // placeholder (its real data is also bound at slot 5; aliasing a
    // never-written buffer is harmless).
    var arg_scratch = self.d_asm_sizes;
    var measure_params = [_]?*anyopaque{
        @ptrCast(&arg_raw),       @ptrCast(&arg_huff_lit), @ptrCast(&arg_huff_tok),
        @ptrCast(&arg_huff_off16),@ptrCast(&arg_descs),    @ptrCast(&arg_sizes),
        @ptrCast(&arg_scratch),
        @ptrCast(&arg_n),
    };
    const t_am = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzAssembleMeasureKernel", 0);
    defer gpu_decode.endKernelTiming(t_am, 0);
    if (launch(module_loader.assemble_measure_fn, n, 1, 1, 32, 1, 1, 0, 0, &measure_params, &extra) != VK_SUCCESS_RC) return false;
    if (sync() != VK_SUCCESS_RC) return false;

    const enc_sizes = allocator.alloc(u32, n) catch return false;
    defer allocator.free(enc_sizes);
    if (d2h(@ptrCast(enc_sizes.ptr), self.d_asm_sizes, sizes_bytes) != VK_SUCCESS_RC) return false;

    // Prefix-sum: each sub-chunk block is 3 (header) + enc_n bytes.
    var total: u32 = 0;
    for (0..n) |i| {
        // `enc_sizes[i] == 0` is unambiguously a measure-kernel parse
        // failure — bail rather than write a malformed block.
        if (enc_sizes[i] == 0) return false;
        descs[i].out_offset = total;
        total += 3 + enc_sizes[i];
    }

    // Pass B — write [3-byte header][payload] for every sub-chunk.
    // assemble_write: n_bindings=6 (d_raw, d_huff_lit, d_huff_tok,
    // d_huff_off16, descs, d_frame), push_constant_size=4 (n_subchunks).
    if (h2d(self.d_asm_descs, @ptrCast(descs.ptr), desc_bytes) != VK_SUCCESS_RC) return false;
    if (!encode_context.ensureBuf(&self.d_asm_out, &self.d_asm_out_size, @max(total, 1))) return false;
    var arg_out = self.d_asm_out;
    var write_params = [_]?*anyopaque{
        @ptrCast(&arg_raw),       @ptrCast(&arg_huff_lit), @ptrCast(&arg_huff_tok),
        @ptrCast(&arg_huff_off16),@ptrCast(&arg_descs),    @ptrCast(&arg_out),
        @ptrCast(&arg_n),
    };
    const t_aw = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzAssembleWriteKernel", 0);
    defer gpu_decode.endKernelTiming(t_aw, 0);
    if (launch(module_loader.assemble_write_fn, n, 1, 1, 32, 1, 1, 0, 0, &write_params, &extra) != VK_SUCCESS_RC) return false;
    if (sync() != VK_SUCCESS_RC) return false;

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

/// CUDA reference: src/encode/encode_assemble.zig:186-190. Pre-formed
/// frame-header + block-header bytes the frame writer splices ahead of
/// the per-chunk payloads.
pub const FramePreamble = struct {
    prefix_bytes: []const u8,
    internal_hdr0: u8,
    internal_hdr1: u8,
};

/// CUDA reference: src/encode/encode_assemble.zig:196-202. Layout
/// description of the per-chunk asm payloads the kernel will splice.
pub const ChunkLayout = struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    per_chunk_asm_off: []const u32,
    per_chunk_asm_size: []const u32,
};

/// CUDA reference: src/encode/encode_assemble.zig:204-308. Pass C — pure
/// D2D frame writer. Splices the device-resident assembled blocks plus
/// host-precomputed per-chunk layout into a complete StreamLZ frame on
/// the device. Returns the total frame byte count, or null on failure.
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

    const h2d = vk.procs.h2d orelse return null;
    const launch = vk.procs.launch_kernel orelse return null;
    const sync = vk.procs.ctx_sync orelse return null;

    // Build per-chunk dst offset table on host. Compressed chunks pay
    // `CHUNK_INTERNAL_HDR_BYTES`; uncompressed chunks pay only
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

    if (h2d(self.d_frame_chunk_dst, @ptrCast(per_chunk_dst_buf.ptr), ent_bytes) != VK_SUCCESS_RC) return null;
    if (h2d(self.d_frame_asm_offsets, @ptrCast(per_chunk_asm_off.ptr), ent_bytes) != VK_SUCCESS_RC) return null;
    if (h2d(self.d_frame_asm_chunk_sz, @ptrCast(per_chunk_asm_size.ptr), ent_bytes) != VK_SUCCESS_RC) return null;
    if (h2d(self.d_frame_prefix_bytes, @ptrCast(prefix_bytes.ptr), prefix_bytes.len) != VK_SUCCESS_RC) return null;

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
    // frame_assemble: n_bindings=7 (d_input, d_asm_out, d_asm_offsets,
    // d_asm_chunk_sizes, d_chunk_dst, d_prefix_bytes, d_output),
    // push_constant_size=32 (prefix_size, hdr0, hdr1, n_chunks,
    // eff_chunk_size, src_len, sc_tail_off, end_mark_off).
    //
    // VK adaptation: the CUDA call site (which this is a 1:1 port of)
    // places d_output at the END of the arg list because cuLaunchKernel
    // packs every arg into a single kernel-parameter blob regardless of
    // pointer-vs-scalar order. procs.launch_kernel here reads params
    // [0..n_bindings) as SSBO handles in slot order then params
    // [n_bindings..) as push-constant scalars in the .comp's declared
    // order — so d_output (SSBO slot 6) must come BEFORE the push
    // constant block, and prefix_size moves into the push constant
    // block at the head (matching the .comp's Push struct order).
    var params = [_]?*anyopaque{
        @ptrCast(&k_input),    @ptrCast(&k_asm_out),  @ptrCast(&k_asm_offs),
        @ptrCast(&k_asm_sizes),@ptrCast(&k_chunk_dst),@ptrCast(&k_prefix),
        @ptrCast(&k_dst),
        @ptrCast(&k_prefix_size),
        @ptrCast(&k_hdr0),     @ptrCast(&k_hdr1),
        @ptrCast(&k_n),        @ptrCast(&k_eff),      @ptrCast(&k_src_len),
        @ptrCast(&k_sc_off),   @ptrCast(&k_end_off),
    };
    var extra = [_]?*anyopaque{null};

    // Grid: n_chunks blocks for per-chunk writes + 1 block for
    // prefix/tail/end mark. 128 threads per block — enough for
    // cooperative copies up to ~few KB/iter.
    if (n_chunks == std.math.maxInt(u32)) return null;
    const grid_x: u32 = n_chunks + 1;
    // Heavy phase: ride the caller's stream when slzCompressAsync set
    // work_stream so a stream-sync on it waits for the frame bytes
    // to be in d_output.
    const heavy_stream: usize = self.work_stream;
    const t_fa = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzFrameAssembleKernel", heavy_stream);
    defer gpu_decode.endKernelTiming(t_fa, heavy_stream);
    if (launch(module_loader.frame_assemble_fn, grid_x, 1, 1, 128, 1, 1, 0, heavy_stream, &params, &extra) != VK_SUCCESS_RC) return null;
    // In async mode the kernel stays in flight on the caller's stream
    // (their stream-sync is the sync point); otherwise block here.
    if (heavy_stream == 0) {
        if (sync() != VK_SUCCESS_RC) return null;
    }

    return total_frame_size;
}
