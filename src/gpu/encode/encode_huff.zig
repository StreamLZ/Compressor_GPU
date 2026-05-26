//! GPU Huffman encode pass and its three stream-specific wrappers
//! (literals, tokens, off16). All four entrypoints delegate to
//! `gpuEncodeHuffImpl`, which builds per-block Huffman tables then runs
//! the 4-stream encoder, optionally leaving the bodies device-resident
//! for the frame-assembly kernels to consume directly.

const std = @import("std");
const ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const ec = @import("encode_context.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = ffi.CUdeviceptr;
const EncodeContext = ec.EncodeContext;
const HuffEncDesc = ec.HuffEncDesc;
const CompressChunkDesc = ec.CompressChunkDesc;

// ── GPU Huffman encode pass ────────────────────────────────────

/// `out_dev`: when non-zero, the encoder writes Huffman bodies straight
/// into that caller-owned device buffer (which must hold ≥
/// `total_dst_bytes`) and skips the host download — `out_bytes` is then
/// unused. This is the 4d device-resident path: the bodies stay on the
/// GPU for the frame-assembly kernels. When zero, bodies land in the
/// shared persist buffer and are downloaded into `out_bytes` as before.
pub fn gpuEncodeHuffImpl(
    self: *EncodeContext,
    descs: []const HuffEncDesc,
    total_dst_bytes: usize,
    out_sizes: []u32,
    out_bytes: []u8,
    out_dev: CUdeviceptr,
    /// Two static, null-terminated kernel-name strings the profiler will
    /// store in last_timings. Index 0 is the build kernel, 1 is the encode
    /// kernel. Pointers must live for the library's lifetime — use
    /// string literals (see gpuEncodeLiteralsHuffImpl etc).
    profile_names: [2][*:0]const u8,
) bool {
    if (!module_loader.init()) return false;
    if (module_loader.huff_tables_kernel_fn == 0 or module_loader.huff_encode_kernel_fn == 0) return false;

    const h2d_fn = ffi.cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = ffi.cuMemcpyDtoH_fn orelse return false;
    const launch_fn = ffi.cuLaunchKernel_fn orelse return false;
    const sync_fn = ffi.cuCtxSynchronize_fn orelse return false;
    const memset_fn = ffi.cuMemsetD8_fn orelse return false;

    const n: u32 = @intCast(descs.len);
    if (n == 0) return true;

    // Per-stream scratch: each 4-way split quarter holds src_size/4 symbols
    // of ≤ 11 bits; 2 bytes/symbol is a safe bound. Size from the largest
    // descriptor so one slab fits every block.
    var max_src: u32 = 0;
    for (descs) |d| {
        if (d.src_size > max_src) max_src = d.src_size;
    }
    const scratch_per_stream: usize = (@as(usize, max_src) / 4 + 64) * 2;

    const desc_bytes: usize = descs.len * @sizeOf(HuffEncDesc);
    const sizes_bytes: usize = descs.len * 4;
    const cl_bytes: usize = descs.len * 256;
    const codes_bytes: usize = descs.len * 256 * 4;
    const scratch_bytes: usize = descs.len * 4 * scratch_per_stream;

    if (!ec.ensureBuf(&self.d_huff_descs_persist, &self.d_huff_descs_size, desc_bytes)) return false;
    if (!ec.ensureBuf(&self.d_huff_cl_persist, &self.d_huff_cl_size, cl_bytes)) return false;
    if (!ec.ensureBuf(&self.d_huff_codes_persist, &self.d_huff_codes_size, codes_bytes)) return false;
    if (!ec.ensureBuf(&self.d_huff_scratch_persist, &self.d_huff_scratch_size, scratch_bytes)) return false;
    if (out_dev == 0 and !ec.ensureBuf(&self.d_huff_out_persist, &self.d_huff_out_size, total_dst_bytes)) return false;
    if (!ec.ensureBuf(&self.d_huff_sizes_persist, &self.d_huff_sizes_size, sizes_bytes)) return false;

    if (h2d_fn(self.d_huff_descs_persist, @ptrCast(descs.ptr), desc_bytes) != ffi.CUDA_SUCCESS) return false;
    if (memset_fn(self.d_huff_sizes_persist, 0, sizes_bytes) != ffi.CUDA_SUCCESS) return false;
    if (sync_fn() != ffi.CUDA_SUCCESS) return false;

    // Kernel 1: build per-block Huffman tables from the source streams.
    var p_src = self.d_output_persist;
    var p_descs = self.d_huff_descs_persist;
    var p_cl = self.d_huff_cl_persist;
    var p_codes = self.d_huff_codes_persist;
    var p_stride: u32 = 256;
    var p_n: u32 = n;
    var tbl_params = [_]?*anyopaque{
        @ptrCast(&p_src),   @ptrCast(&p_descs), @ptrCast(&p_cl),
        @ptrCast(&p_codes), @ptrCast(&p_stride), @ptrCast(&p_n),
    };
    var extra = [_]?*anyopaque{null};
    const t_htbl = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, profile_names[0], 0);
    // Defer so the begin event always pairs with an end record, even on
    // launch failure — finalizeProfiling otherwise blocks on the unrecorded
    // end event.
    defer gpu_decode.endKernelTiming(t_htbl, 0);
    if (launch_fn(module_loader.huff_tables_kernel_fn, n, 1, 1, 32, 1, 1, 0, 0, &tbl_params, &extra) != ffi.CUDA_SUCCESS)
        return false;

    // Kernel 2: pack each sub-chunk into a chunk_type=4 body. Device-
    // resident mode writes straight into the caller's buffer.
    var p_scratch = self.d_huff_scratch_persist;
    var p_out: CUdeviceptr = if (out_dev != 0) out_dev else self.d_huff_out_persist;
    var p_sizes = self.d_huff_sizes_persist;
    var p_sps: u32 = @intCast(scratch_per_stream);
    var enc_params = [_]?*anyopaque{
        @ptrCast(&p_src),     @ptrCast(&p_descs), @ptrCast(&p_cl),
        @ptrCast(&p_codes),   @ptrCast(&p_scratch), @ptrCast(&p_out),
        @ptrCast(&p_sizes),   @ptrCast(&p_sps), @ptrCast(&p_stride),
        @ptrCast(&p_n),
    };
    const t_henc = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, profile_names[1], 0);
    defer gpu_decode.endKernelTiming(t_henc, 0);
    if (launch_fn(module_loader.huff_encode_kernel_fn, n, 1, 1, 32, 1, 1, 0, 0, &enc_params, &extra) != ffi.CUDA_SUCCESS)
        return false;
    if (sync_fn() != ffi.CUDA_SUCCESS) return false;

    if (d2h_fn(@ptrCast(out_sizes.ptr), self.d_huff_sizes_persist, sizes_bytes) != ffi.CUDA_SUCCESS) return false;
    // Device-resident mode leaves the bodies on the GPU — only the small
    // sizes array comes back. Else download the full body buffer.
    if (out_dev == 0)
        if (d2h_fn(@ptrCast(out_bytes.ptr), self.d_huff_out_persist, total_dst_bytes) != ffi.CUDA_SUCCESS) return false;
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
        const init_b: u32 = if (chunk_descs[i].is_first != 0) ec.INITIAL_LITERAL_COPY_BYTES else 0;

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
        if (off16_count < 32) continue; // matches CPU `>= 32` gate
        const off16_data: u32 = off16_hdr + 2;

        // Huffman body worst case ≈ 137 fixed + count×11/8; count*2 + 256
        // is a safe capacity bound.
        const hi_cap: u32 = off16_count * 2 + 256;
        descs[i] = .{
            .src_offset = off16_data + 1, // hi plane: odd bytes
            .src_size = off16_count,
            .src_stride = 2,
            .dst_offset = total,
            .dst_capacity = hi_cap,
        };
        hi_offsets[i] = total;
        total += hi_cap;

        const lo_cap: u32 = off16_count * 2 + 256;
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

    const all_sizes = allocator.alloc(u32, num_descs) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        return false;
    };

    // Device-resident mode: hi+lo bodies share one assembly buffer
    // (d_asm_huff_off16); no host bytes, no D2H bounce.
    const resident = self.huff_keep_device;
    var out_dev: CUdeviceptr = 0;
    if (resident) {
        if (!ec.ensureBuf(&self.d_asm_huff_off16, &self.d_asm_huff_off16_size, total)) {
            allocator.free(hi_offsets);
            allocator.free(lo_offsets);
            allocator.free(all_sizes);
            return false;
        }
        out_dev = self.d_asm_huff_off16;
    }
    const bytes: []u8 = if (resident) &[_]u8{} else (allocator.alloc(u8, total) catch {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        return false;
    });

    if (!gpuEncodeHuffImpl(self, descs, total, all_sizes, bytes, out_dev,
        .{ "huff-off16/build", "huff-off16/encode" })) {
        allocator.free(hi_offsets);
        allocator.free(lo_offsets);
        allocator.free(all_sizes);
        if (!resident) allocator.free(bytes);
        return false;
    }

    // Split sizes into hi (first n) and lo (next n).
    const hi_sizes = allocator.alloc(u32, n) catch {
        allocator.free(hi_offsets); allocator.free(lo_offsets);
        allocator.free(all_sizes);
        if (!resident) allocator.free(bytes);
        return false;
    };
    const lo_sizes = allocator.alloc(u32, n) catch {
        allocator.free(hi_offsets); allocator.free(lo_offsets);
        allocator.free(hi_sizes);
        allocator.free(all_sizes);
        if (!resident) allocator.free(bytes);
        return false;
    };
    @memcpy(hi_sizes, all_sizes[0..n]);
    @memcpy(lo_sizes, all_sizes[n..]);
    allocator.free(all_sizes);

    // No-clobber assert: catch the double-encode-without-free bug at the
    // write site (where we still know whose buffer would leak) instead of
    // at the free site (where the double-free is harder to trace).
    std.debug.assert(self.huff_off16hi_sizes == null);
    std.debug.assert(self.huff_off16hi_data == null);
    std.debug.assert(self.huff_off16hi_offsets == null);
    std.debug.assert(self.huff_off16lo_sizes == null);
    std.debug.assert(self.huff_off16lo_data == null);
    std.debug.assert(self.huff_off16lo_offsets == null);
    self.huff_off16hi_sizes = hi_sizes;
    self.huff_off16hi_data = if (resident) null else bytes; // shared buffer; both hi and lo offsets index into it
    self.huff_off16hi_offsets = hi_offsets;
    self.huff_off16lo_sizes = lo_sizes;
    self.huff_off16lo_data = if (resident) null else bytes; // SAME pointer — only one of {hi,lo} should free it
    self.huff_off16lo_offsets = lo_offsets;
    return true;
}

/// Entropy-codes each sub-chunk's literal stream with the GPU Huffman
/// encoder (chunk_type=4). Populates `huff_lit_*` on the context.
pub fn gpuEncodeLiteralsHuffImpl(
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

    var descs = allocator.alloc(HuffEncDesc, n) catch return false;
    defer allocator.free(descs);
    // bool-return function — errdefer would never fire; cleanup lives
    // in the explicit catch blocks below.
    var offsets = allocator.alloc(u32, n) catch return false;

    var total: u32 = 0;
    for (0..n) |i| {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) ec.INITIAL_LITERAL_COPY_BYTES else 0;
        offsets[i] = total;
        descs[i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 1, .dst_offset = total, .dst_capacity = 0 };

        if (cs < init_b + 3) continue;
        const lit_hdr: u32 = base + init_b;
        const lit_count: u32 =
            (@as(u32, output[lit_hdr]) << 16) |
            (@as(u32, output[lit_hdr + 1]) << 8) |
            @as(u32, output[lit_hdr + 2]);
        if (lit_count == 0) continue;
        const lit_src: u32 = lit_hdr + 3;
        if (lit_src + lit_count > base + cs) continue;

        // Huffman body worst case ≈ 137 fixed + count×11/8; count*2 + 256
        // is a safe capacity bound.
        const dst_cap: u32 = lit_count * 2 + 256;
        descs[i] = .{
            .src_offset = lit_src,
            .src_size = lit_count,
            .src_stride = 1,
            .dst_offset = total,
            .dst_capacity = dst_cap,
        };
        total += dst_cap;
    }

    if (total == 0) {
        allocator.free(offsets);
        return false;
    }

    const sizes = allocator.alloc(u32, n) catch {
        allocator.free(offsets);
        return false;
    };

    // Device-resident mode: write Huffman bodies straight into the
    // assembly buffer (no host bytes, no D2H bounce).
    const resident = self.huff_keep_device;
    var out_dev: CUdeviceptr = 0;
    if (resident) {
        if (!ec.ensureBuf(&self.d_asm_huff_lit, &self.d_asm_huff_lit_size, total)) {
            allocator.free(offsets);
            allocator.free(sizes);
            return false;
        }
        out_dev = self.d_asm_huff_lit;
    }
    const bytes: []u8 = if (resident) &[_]u8{} else (allocator.alloc(u8, total) catch {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    });

    if (!gpuEncodeHuffImpl(self, descs, total, sizes, bytes, out_dev,
        .{ "huff-lit/build", "huff-lit/encode" })) {
        allocator.free(offsets);
        allocator.free(sizes);
        if (!resident) allocator.free(bytes);
        return false;
    }

    self.huff_lit_sizes = sizes;
    self.huff_lit_data = if (resident) null else bytes;
    self.huff_lit_offsets = offsets;
    return true;
}

/// Entropy-codes each sub-chunk's token stream with the GPU Huffman
/// encoder (chunk_type=4). Populates `huff_tok_*` on the context.
pub fn gpuEncodeTokensHuffImpl(
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

    var descs = allocator.alloc(HuffEncDesc, n) catch return false;
    defer allocator.free(descs);
    // bool-return function — errdefer would never fire; cleanup lives
    // in the explicit catch blocks below.
    var offsets = allocator.alloc(u32, n) catch return false;

    var total: u32 = 0;
    for (0..n) |i| {
        const cs = comp_sizes[i];
        const base: u32 = chunk_descs[i].dst_offset;
        const init_b: u32 = if (chunk_descs[i].is_first != 0) ec.INITIAL_LITERAL_COPY_BYTES else 0;
        offsets[i] = total;
        descs[i] = .{ .src_offset = 0, .src_size = 0, .src_stride = 1, .dst_offset = total, .dst_capacity = 0 };

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
        if (tok_count == 0) continue;
        const tok_src: u32 = tok_hdr + 3;
        if (tok_src + tok_count > base + cs) continue;

        const dst_cap: u32 = tok_count * 2 + 256;
        descs[i] = .{
            .src_offset = tok_src,
            .src_size = tok_count,
            .src_stride = 1,
            .dst_offset = total,
            .dst_capacity = dst_cap,
        };
        total += dst_cap;
    }

    if (total == 0) {
        allocator.free(offsets);
        return false;
    }

    const sizes = allocator.alloc(u32, n) catch {
        allocator.free(offsets);
        return false;
    };

    // Device-resident mode: write Huffman bodies straight into the
    // assembly buffer (no host bytes, no D2H bounce).
    const resident = self.huff_keep_device;
    var out_dev: CUdeviceptr = 0;
    if (resident) {
        if (!ec.ensureBuf(&self.d_asm_huff_tok, &self.d_asm_huff_tok_size, total)) {
            allocator.free(offsets);
            allocator.free(sizes);
            return false;
        }
        out_dev = self.d_asm_huff_tok;
    }
    const bytes: []u8 = if (resident) &[_]u8{} else (allocator.alloc(u8, total) catch {
        allocator.free(offsets);
        allocator.free(sizes);
        return false;
    });

    if (!gpuEncodeHuffImpl(self, descs, total, sizes, bytes, out_dev,
        .{ "huff-tok/build", "huff-tok/encode" })) {
        allocator.free(offsets);
        allocator.free(sizes);
        if (!resident) allocator.free(bytes);
        return false;
    }

    self.huff_tok_sizes = sizes;
    self.huff_tok_data = if (resident) null else bytes;
    self.huff_tok_offsets = offsets;
    return true;
}
