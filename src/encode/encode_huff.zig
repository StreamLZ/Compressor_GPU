//! GPU Huffman encode pass and its three stream-specific wrappers
//! (literals, tokens, off16). All four entrypoints delegate to
//! `gpuEncodeHuffImpl`, which builds per-block Huffman tables then runs
//! the 4-stream encoder, leaving the bodies device-resident in the
//! caller's `d_asm_huff_*` buffer for the frame-assembly kernels to
//! consume directly.

const std = @import("std");
const cuda_ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = cuda_ffi.CUdeviceptr;
const EncodeContext = encode_context.EncodeContext;
const HuffEncDesc = encode_context.HuffEncDesc;
const CompressChunkDesc = encode_context.CompressChunkDesc;

// Per-block dst capacity fixed component for the BIL Huffman encoder.
// Each entropy-stream descriptor sizes its `dst_capacity` as
// `bilDstCap(count)` which both covers the per-stream entropy bits
// (≤ 2 bytes/symbol since codes are ≤ 11 bits) and rounds the result
// up to a 4-byte boundary so consecutive blocks' `dst_offset` values
// stay 4-aligned. The BIL encode kernel writes the interleaved area
// with `*(uint32_t*)dst = ...` where `dst = out + dst_offset + 228 +
// w*128 + lane*4`; a misaligned `dst_offset` makes the store trap
// (CUDA_ERROR_MISALIGNED_ADDRESS / rc=716) the second a sub-chunk has
// an odd input `count`.
//
// Breakdown of the fixed component:
//   - HUFF_BODY_HEADER_BYTES = 228 (weights 128 + sub-header 96 + K 4)
//   - 4 × N rounding tax: each of the 32 streams rounds its encoded
//     byte count up to a 4-byte BIL word, costing ≤ 3 bytes per stream
//     = ≤ 96 bytes per block.
//   - 32 bytes of slack against minor estimate drift.
// Total = 228 + 96 + 32 = 356. KEEP IN SYNC with HUFF_BODY_HEADER_BYTES
// and HUFF_NUM_STREAMS in src/gpu/common/gpu_huffman.cuh.
const HUFF_BODY_FIXED_BYTES: u32 = 356;

/// Per-block dst slot size, 4-byte aligned so cumulative `dst_offset`
/// stays aligned for the BIL kernel's `*(uint32_t*)dst = ...` writes.
inline fn bilDstCap(count: u32) u32 {
    return (count * 2 + HUFF_BODY_FIXED_BYTES + 3) & ~@as(u32, 3);
}

// ── GPU Huffman encode pass ────────────────────────────────────

/// The encoder writes Huffman bodies straight into `out_dev` (a
/// caller-owned device buffer holding ≥ `total_dst_bytes`) and
/// downloads only the per-block size table into `out_sizes`. The
/// frame-assembly kernels then read the bodies in place.
pub fn gpuEncodeHuffImpl(
    self: *EncodeContext,
    descs: []const HuffEncDesc,
    out_sizes: []u32,
    out_dev: CUdeviceptr,
    /// Two static, null-terminated kernel-name strings the profiler will
    /// store in last_timings. Index 0 is the build kernel, 1 is the encode
    /// kernel. Pointers must live for the library's lifetime - use
    /// string literals (see gpuEncodeLiteralsHuffImpl etc).
    profile_names: [2][*:0]const u8,
) bool {
    if (!module_loader.init()) return false;
    if (module_loader.huff_tables_kernel_fn == 0 or module_loader.huff_encode_kernel_fn == 0) return false;

    const h2d_fn = cuda_ffi.cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuda_ffi.cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuda_ffi.cuLaunchKernel_fn orelse return false;
    const sync_fn = cuda_ffi.cuCtxSynchronize_fn orelse return false;

    const n: u32 = @intCast(descs.len);
    // Convention across all four entrypoints: empty input → false (no
    // bodies produced; caller falls back to the raw stream). Matches the
    // `n == 0 → return false` early-outs in the wrappers below.
    if (n == 0) return false;

    // Per-stream scratch sized from the largest descriptor (one slab fits
    // every block). NUM_STREAMS mirrors HUFF_NUM_STREAMS in
    // src/gpu/common/gpu_huffman.cuh - the encoder kernel uses that
    // constant directly; this Zig side has to match so the scratch slab
    // has the right per-stream stride. Each stream gets src_size/N
    // symbols of ≤ 11 bits; 2 bytes/symbol is a safe bound.
    //
    // KEEP IN SYNC: HUFF_NUM_STREAMS in src/gpu/common/gpu_huffman.cuh.
    // There is NO compile-time check that this constant matches the .cuh
    // side. If they drift, the encoder splits the input into the wrong
    // number of slices while the kernel reads/writes the right number,
    // producing silently-wrong output that the decoder may still accept
    // as a "valid" but corrupt frame. A future safeguard would be a
    // build-step that codegens this constant from the .cuh.
    const NUM_STREAMS: usize = 32;
    var max_src: u32 = 0;
    for (descs) |d| {
        if (d.src_size > max_src) max_src = d.src_size;
    }
    // +64 covers the trailing-byte flush (encoder writes one partial byte
    // at end of stream) + height-limit rounding (codes are ≤11 bits, so
    // one extra byte per stream is enough headroom). At N=32, per-stream
    // slices average ~src/32 bytes, so the +64 is generous (~32x the
    // slack the 4-stream era had at the same constant).
    //
    // BIL alignment: the encode kernel's interleaved-row write reads each
    // lane's scratch as `*(uint32_t*)(my_scratch + w*4)`. `my_scratch`
    // strides by `scratch_per_stream` per lane, so any non-multiple-of-4
    // stride leaves odd lanes 2 bytes off from a 4-byte boundary — and
    // PTX `ld.u32` on a runtime-derived misaligned pointer is undefined
    // (compiler emits a single-instruction load assuming alignment).
    // `(x + 64) * 2` is `2x + 128`, which is `2 mod 4` whenever x is odd.
    // Round up to 16 for L2 sector alignment (free, and the kernel reads
    // 128 bytes per row anyway).
    const scratch_per_stream: usize = std.mem.alignForward(usize, (@as(usize, max_src) / NUM_STREAMS + 64) * 2, 16);

    const desc_bytes: usize = descs.len * @sizeOf(HuffEncDesc);
    const sizes_bytes: usize = descs.len * 4;
    const cl_bytes: usize = descs.len * 256;
    const codes_bytes: usize = descs.len * 256 * 4;
    const scratch_bytes: usize = descs.len * NUM_STREAMS * scratch_per_stream;

    if (!encode_context.ensureBuf(&self.d_huff_descs_persist, &self.d_huff_descs_size, desc_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_huff_cl_persist, &self.d_huff_cl_size, cl_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_huff_codes_persist, &self.d_huff_codes_size, codes_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_huff_scratch_persist, &self.d_huff_scratch_size, scratch_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_huff_sizes_persist, &self.d_huff_sizes_size, sizes_bytes)) return false;

    if (h2d_fn(self.d_huff_descs_persist, @ptrCast(descs.ptr), desc_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    // (No memset of d_huff_sizes_persist needed: slzHuffEncode4StreamKernel
    // writes out_sizes[block_id] for every block_id < n_blocks, including
    // empty descs (which write 0). The downstream d2h reads exactly
    // sizes_bytes for this call's descs.len, so stale tail bytes can't
    // leak in.)
    if (sync_fn() != cuda_ffi.CUDA_SUCCESS) return false;

    // Kernel 1: build per-block Huffman tables from the source streams.
    // Huffman source = LZ output (raw streams written by gpuCompressImpl
    // into d_output_persist; descriptor src_offsets point into the same
    // buffer at the lit/tok/off16 sub-stream offsets).
    var p_src = self.d_output_persist;
    var p_descs = self.d_huff_descs_persist;
    var p_cl = self.d_huff_cl_persist;
    var p_codes = self.d_huff_codes_persist;
    var p_stride: u32 = 256;
    var p_n: u32 = n;
    var tbl_params = [_]?*anyopaque{
        @ptrCast(&p_src),   @ptrCast(&p_descs),  @ptrCast(&p_cl),
        @ptrCast(&p_codes), @ptrCast(&p_stride), @ptrCast(&p_n),
    };
    var extra = [_]?*anyopaque{null};
    const t_htbl = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, profile_names[0], 0);
    // Defer so the begin event always pairs with an end record, even on
    // launch failure - finalizeProfiling otherwise blocks on the unrecorded
    // end event.
    defer gpu_decode.endKernelTiming(t_htbl, 0);
    if (launch_fn(module_loader.huff_tables_kernel_fn, n, 1, 1, 32, 1, 1, 0, 0, &tbl_params, &extra) != cuda_ffi.CUDA_SUCCESS)
        return false;

    // Kernel 2: pack each sub-chunk into a chunk_type=4 body. The
    // bodies land directly in the caller's `out_dev` buffer; only the
    // small size table is D2H'd back at the end of this function.
    var p_scratch = self.d_huff_scratch_persist;
    var p_out: CUdeviceptr = out_dev;
    var p_sizes = self.d_huff_sizes_persist;
    var p_sps: u32 = @intCast(scratch_per_stream);
    var enc_params = [_]?*anyopaque{
        @ptrCast(&p_src),   @ptrCast(&p_descs),   @ptrCast(&p_cl),
        @ptrCast(&p_codes), @ptrCast(&p_scratch), @ptrCast(&p_out),
        @ptrCast(&p_sizes), @ptrCast(&p_sps),     @ptrCast(&p_stride),
        @ptrCast(&p_n),
    };
    const t_henc = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, profile_names[1], 0);
    defer gpu_decode.endKernelTiming(t_henc, 0);
    if (launch_fn(module_loader.huff_encode_kernel_fn, n, 1, 1, 32, 1, 1, 0, 0, &enc_params, &extra) != cuda_ffi.CUDA_SUCCESS)
        return false;
    if (sync_fn() != cuda_ffi.CUDA_SUCCESS) return false;

    if (d2h_fn(@ptrCast(out_sizes.ptr), self.d_huff_sizes_persist, sizes_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    return true;
}

/// Launch the GPU Huffman encoder over a finished descriptor list,
/// publish the resulting per-sub-chunk size + offset tables on the
/// context, and own all error-path cleanup. The helper takes
/// ownership of `offsets` (the caller pre-allocates it while filling
/// `descs`); on failure the helper frees it before returning false.
/// On success the caller's `out_sizes_field` and `out_offsets_field`
/// point at fresh allocations the deinit / freeHuff* paths will reap.
fn encodeStreamAndPublish(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    descs: []const HuffEncDesc,
    offsets: []u32,
    total: u32,
    d_out: *CUdeviceptr,
    d_out_size: *usize,
    out_sizes_field: *?[]u32,
    out_offsets_field: *?[]u32,
    profile_names: [2][*:0]const u8,
) bool {
    if (total == 0) {
        allocator.free(offsets);
        return false;
    }
    const sizes = allocator.alloc(u32, descs.len) catch {
        allocator.free(offsets);
        return false;
    };
    if (!encode_context.ensureBuf(d_out, d_out_size, total)) {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    }
    if (!gpuEncodeHuffImpl(self, descs, sizes, d_out.*, profile_names)) {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    }
    out_sizes_field.* = sizes;
    out_offsets_field.* = offsets;
    return true;
}

/// Entropy-codes off16 hi/lo byte planes with the GPU Huffman encoder
/// (chunk_type=4). The `>= 32` gate skips streams too short to amortize
/// the 5-byte Huffman chunk header. Populates `huff_off16{hi,lo}_*` on
/// the context; the bodies carry no header (the frame assembler prepends it).
pub fn gpuEncodeOff16HuffImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    if (!module_loader.init()) return false;
    if (module_loader.huff_tables_kernel_fn == 0 or module_loader.huff_encode_kernel_fn == 0) return false;
    const n = chunk_descs.len;
    if (n == 0) return false;

    const num_descs = n * 2;
    var descs = allocator.alloc(HuffEncDesc, num_descs) catch return false;
    defer allocator.free(descs);
    // Function returns `bool`, not `!bool`, so an errdefer would never
    // fire. Cleanup on subsequent failure is handled by the explicit
    // `allocator.free(...)` calls in each `catch { ... }` block below.
    var hi_offsets = allocator.alloc(u32, n) catch return false;
    var lo_offsets = allocator.alloc(u32, n) catch return false;

    var total: u32 = 0;
    for (0..n) |i| {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) encode_context.INITIAL_LITERAL_COPY_BYTES else 0;

        // Default to empty (no entropy coding for this sub-chunk).
        hi_offsets[i] = total;
        lo_offsets[i] = total;
        descs[i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 2, .dst_offset = total, .dst_capacity = 0 };
        descs[n + i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 2, .dst_offset = total, .dst_capacity = 0 };

        if (cs < init_b + 6) continue;
        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        const tok_hdr: u32 = lit_hdr + 3 + lit_count;
        if (tok_hdr + 3 > base + cs) continue;
        const tok_count: u32 =
            (@as(u32, output[tok_hdr]) << 16) |
            (@as(u32, output[tok_hdr + 1]) << 8) |
            @as(u32, output[tok_hdr + 2]);

        // cmd_stream2_offset (2 bytes) is present when sub-chunk > 64KB.
        const after_tok: u32 = tok_hdr + 3 + tok_count;
        const cmd2_size: u32 = if (chunk_descs[i].src_size > 0x10000) 2 else 0;
        const off16_hdr: u32 = after_tok + cmd2_size;
        if (off16_hdr + 2 > base + cs) continue;

        const off16_count: u32 =
            @as(u32, output[off16_hdr]) |
            (@as(u32, output[off16_hdr + 1]) << 8);
        // Below this count, the per-stream sub-header overhead would
        // dwarf the entropy savings. CPU oracle uses the same `>= 32`
        // gate; the literal `32` is a policy threshold, NOT a stream
        // count (the coincidence with HUFF_NUM_STREAMS is unrelated).
        const OFF16_HUFFMAN_MIN_COUNT: u32 = 32;
        if (off16_count < OFF16_HUFFMAN_MIN_COUNT) continue;
        const off16_data: u32 = off16_hdr + 2;

        // Huffman BIL body worst case = HUFF_BODY_FIXED_BYTES (228 header
        // + 96 rounding + 32 slack = 356) + count×11/8 per-stream bits.
        // See the HUFF_BODY_FIXED_BYTES constant docstring above.
        const hi_cap: u32 = bilDstCap(off16_count);
        descs[i] = .{
            .src_offset = off16_data + 1, // hi plane: odd bytes
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = hi_cap,
        };
        hi_offsets[i] = total;
        total += hi_cap;

        const lo_cap: u32 = bilDstCap(off16_count);
        descs[n + i] = .{
            .src_offset = off16_data, // lo plane: even bytes
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = lo_cap,
        };
        lo_offsets[i] = total;
        total += lo_cap;
    }

    if (total == 0) {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        return false;
    }

    // Run the encoder over the combined 2n descriptors, capturing one
    // size array that we then split into hi (first n) and lo (next n).
    // We can't use `encodeStreamAndPublish` directly because it
    // publishes one (sizes, offsets) pair, and off16 needs two.
    const all_sizes = allocator.alloc(u32, num_descs) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        return false;
    };
    if (!encode_context.ensureBuf(&self.d_asm_huff_off16, &self.d_asm_huff_off16_size, total)) {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        return false;
    }
    if (!gpuEncodeHuffImpl(self, descs, all_sizes, self.d_asm_huff_off16, .{ "huff-off16/build", "huff-off16/encode" })) {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        return false;
    }

    const hi_sizes = allocator.alloc(u32, n) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        return false;
    };
    const lo_sizes = allocator.alloc(u32, n) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(hi_sizes);
        allocator.free(all_sizes);
        return false;
    };
    @memcpy(hi_sizes, all_sizes[0..n]);
    @memcpy(lo_sizes, all_sizes[n..]);
    allocator.free(all_sizes);

    // No-clobber assert: catch a double-encode-without-free at the
    // write site (where we still know whose buffer would leak) instead
    // of at the free site (where the double-free is harder to trace).
    std.debug.assert(self.huff_off16hi_sizes == null);
    std.debug.assert(self.huff_off16hi_offsets == null);
    std.debug.assert(self.huff_off16lo_sizes == null);
    std.debug.assert(self.huff_off16lo_offsets == null);
    self.huff_off16hi_sizes = hi_sizes;
    self.huff_off16hi_offsets = hi_offsets;
    self.huff_off16lo_sizes = lo_sizes;
    self.huff_off16lo_offsets = lo_offsets;
    return true;
}

/// Walk a sub-chunk's stream-header chain. Returns the (src_offset,
/// count) pair for the requested stream, or null when the chain is
/// truncated / the stream is absent / too small. `which == .lit` stops
/// after the literal header; `.tok` walks past it to the token header.
const SubStream = enum { lit, tok };
const StreamSlice = struct { src: u32, count: u32 };
inline fn walkStream(
    output: []const u8,
    base: u32,
    cs: u32,
    init_b: u32,
    which: SubStream,
) ?StreamSlice {
    const min_chain: u32 = switch (which) { .lit => 3, .tok => 6 };
    if (cs < init_b + min_chain) return null;
    const lit_hdr: u32 = base + init_b;
    const lit_count: u32 =
        (@as(u32, output[lit_hdr]) << 16) |
        (@as(u32, output[lit_hdr + 1]) << 8) |
        @as(u32, output[lit_hdr + 2]);
    switch (which) {
        .lit => {
            if (lit_count == 0) return null;
            const lit_src: u32 = lit_hdr + 3;
            if (lit_src + lit_count > base + cs) return null;
            return .{ .src = lit_src, .count = lit_count };
        },
        .tok => {
            const tok_hdr: u32 = lit_hdr + 3 + lit_count;
            if (tok_hdr + 3 > base + cs) return null;
            const tok_count: u32 =
                (@as(u32, output[tok_hdr]) << 16) |
                (@as(u32, output[tok_hdr + 1]) << 8) |
                @as(u32, output[tok_hdr + 2]);
            if (tok_count == 0) return null;
            const tok_src: u32 = tok_hdr + 3;
            if (tok_src + tok_count > base + cs) return null;
            return .{ .src = tok_src, .count = tok_count };
        },
    }
}

/// Entropy-code one byte-stride sub-stream per sub-chunk (literals or
/// tokens). The two callable wrappers differ only by which stream they
/// pull out of the LZ output, which device buffer they target, and
/// which context fields they publish to.
fn encodeByteStreamHuff(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
    which: SubStream,
    d_out: *CUdeviceptr,
    d_out_size: *usize,
    out_sizes_field: *?[]u32,
    out_offsets_field: *?[]u32,
    profile_names: [2][*:0]const u8,
) bool {
    if (!module_loader.init()) return false;
    if (module_loader.huff_tables_kernel_fn == 0 or module_loader.huff_encode_kernel_fn == 0) return false;
    const n = chunk_descs.len;
    if (n == 0) return false;

    var descs = allocator.alloc(HuffEncDesc, n) catch return false;
    defer allocator.free(descs);
    var offsets = allocator.alloc(u32, n) catch return false;

    var total: u32 = 0;
    for (0..n) |i| {
        const init_b: u32 = if (chunk_descs[i].is_first != 0) encode_context.INITIAL_LITERAL_COPY_BYTES else 0;
        offsets[i] = total;
        descs[i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 1, .dst_offset = total, .dst_capacity = 0 };

        const slice = walkStream(output, chunk_descs[i].dst_offset, comp_sizes[i], init_b, which) orelse continue;
        // Huffman BIL body worst case = HUFF_BODY_FIXED_BYTES (228 header
        // + 96 rounding + 32 slack = 356) + count×11/8 per-stream bits.
        // See the HUFF_BODY_FIXED_BYTES constant docstring above.
        const dst_cap: u32 = bilDstCap(slice.count);
        descs[i] = .{
            .src_offset = slice.src,
            .src_size = slice.count,
            .src_stride = 1,
            .dst_offset = total,
            .dst_capacity = dst_cap,
        };
        total += dst_cap;
    }

    return encodeStreamAndPublish(
        self, allocator, descs, offsets, total,
        d_out, d_out_size, out_sizes_field, out_offsets_field, profile_names,
    );
}

/// Entropy-codes each sub-chunk's literal stream. Populates `huff_lit_*`.
pub fn gpuEncodeLiteralsHuffImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    return encodeByteStreamHuff(
        self, allocator, output, chunk_descs, comp_sizes, .lit,
        &self.d_asm_huff_lit, &self.d_asm_huff_lit_size,
        &self.huff_lit_sizes, &self.huff_lit_offsets,
        .{ "huff-lit/build", "huff-lit/encode" },
    );
}

/// Entropy-codes each sub-chunk's token stream. Populates `huff_tok_*`.
pub fn gpuEncodeTokensHuffImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    output: []const u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes: []const u32,
) bool {
    return encodeByteStreamHuff(
        self, allocator, output, chunk_descs, comp_sizes, .tok,
        &self.d_asm_huff_tok, &self.d_asm_huff_tok_size,
        &self.huff_tok_sizes, &self.huff_tok_offsets,
        .{ "huff-tok/build", "huff-tok/encode" },
    );
}
