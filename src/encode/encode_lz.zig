//! GPU LZ-encode launcher.
//!
//! Owns the host side of the per-chunk LZ kernel launch: uploads input,
//! descriptors, and (level-dependent) global hash tables, fires
//! `slzLzEncodeKernel`, then downloads `comp_sizes` plus the actual
//! compressed bytes per block. Reads the per-level policy from
//! `levels.zig` and the persistent device buffers from the supplied
//! `EncodeContext`.

const std = @import("std");
const cuda_ffi = @import("cuda_ffi.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const levels = @import("levels.zig");
const gpu_decode = @import("../decode/driver.zig");

const CUdeviceptr = cuda_ffi.CUdeviceptr;
const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;

pub fn gpuCompressImpl(
    self: *EncodeContext,
    input: []const u8,
    output: []u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes_out: []u32,
    io: ?std.Io,
    level: u8,
) bool {
    if (!module_loader.init()) return false;

    // FFI fn resolution convention: entrypoints that this path REQUIRES
    // (no fallback) are resolved upfront so the whole launch bails out
    // cleanly if any one is missing. Optional fns (used only on a
    // conditional branch) are resolved inline at the use site - see
    // `cuMemsetD8_fn` below, which is wrapped in an `if (...) |fn|` so
    // its absence doesn't kill the encode (zeroing isn't strictly
    // required when the kernel writes every output byte).
    const h2d_fn = cuda_ffi.cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuda_ffi.cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuda_ffi.cuLaunchKernel_fn orelse return false;
    const sync_fn = cuda_ffi.cuCtxSynchronize_fn orelse return false;

    const num_chunks: u32 = @intCast(chunk_descs.len);
    const desc_bytes = chunk_descs.len * @sizeOf(CompressChunkDesc);
    const sizes_bytes = @as(usize, num_chunks) * 4;
    const hash_bits: u32 = levels.hashBitsForLevel(level);
    const hash_size: usize = @as(usize, 1) << @intCast(hash_bits);
    const chain = levels.useChainParser(level);

    if (!encode_context.ensureBuf(&self.d_input_persist, &self.d_input_size, input.len)) return false;
    if (!encode_context.ensureBuf(&self.d_output_persist, &self.d_output_size, output.len)) return false;
    if (!encode_context.ensureBuf(&self.d_descs_persist, &self.d_descs_size, desc_bytes)) return false;
    if (!encode_context.ensureBuf(&self.d_sizes_persist, &self.d_sizes_size, sizes_bytes)) return false;

    // Global hash tables. Chain mode uses 3 tables per block (first_hash +
    // long_hash + next_hash); non-chain modes use a single hash table
    // (L1 needs more than CUDA shared-mem allows, and global also dodges
    // the shared-mem-hash corruption seen at L2 sc>=0.5).
    if (chain) {
        const next_hash_words: usize = encode_context.NEXT_HASH_ENTRIES / 2; // u16 entries packed into u32 words
        const table_stride = hash_size + hash_size + next_hash_words;
        const hash_bytes = @as(usize, num_chunks) * table_stride * 4;
        if (!encode_context.ensureBuf(&self.d_hash_persist, &self.d_hash_size, hash_bytes)) return false;
    } else {
        const hash_bytes = @as(usize, num_chunks) * hash_size * 4;
        if (!encode_context.ensureBuf(&self.d_hash_persist, &self.d_hash_size, hash_bytes)) return false;
    }

    const d_input = self.d_input_persist;
    const d_output = self.d_output_persist;
    const d_descs = self.d_descs_persist;
    const d_sizes = self.d_sizes_persist;

    // Upload input + descriptors. When the caller's data is already
    // device-resident at `d_input_override`, populate `d_input_persist`
    // via D2D (no PCIe) instead of an H2D from the host `input` slice.
    if (self.d_input_override != 0) {
        // Caller's data is already GPU-resident; the host `input` slice
        // may be a sentinel (per `EncodeContext.d_input_override` doc).
        // If D2D-async is unavailable, fail rather than H2D-ing a stub.
        const d2d = cuda_ffi.cuMemcpyDtoDAsync_fn orelse return false;
        if (d2d(d_input, self.d_input_override, input.len, 0) != cuda_ffi.CUDA_SUCCESS) return false;
    } else {
        if (h2d_fn(d_input, @ptrCast(input.ptr), input.len) != cuda_ffi.CUDA_SUCCESS) return false;
    }
    if (h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    if (cuda_ffi.cuMemsetD8_fn) |memset_fn|
        if (memset_fn(d_sizes, 0, sizes_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    if (sync_fn() != cuda_ffi.CUDA_SUCCESS) return false;

    const t_before = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_input = d_input;
    var p_output = d_output;
    var p_descs = d_descs;
    var p_global_hash: CUdeviceptr = self.d_hash_persist;
    var p_sizes = d_sizes;
    var p_total = num_chunks;
    var p_hash_bits = hash_bits;
    var p_use_chain: u32 = if (chain) 1 else 0;
    // L4+ enables the greedy parser's match-range rehash (CPU engine_level>=2).
    // L3 stays without it - that is the L3/L4 distinction. L5 uses the chain
    // parser so the flag is inert there.
    var p_l4: u32 = if (level >= 4) 1 else 0;

    var params = [_]?*anyopaque{
        @ptrCast(&p_input),
        @ptrCast(&p_output),
        @ptrCast(&p_descs),
        @ptrCast(&p_global_hash),
        @ptrCast(&p_sizes),
        @ptrCast(&p_total),
        @ptrCast(&p_hash_bits),
        @ptrCast(&p_use_chain),
        @ptrCast(&p_l4),
    };
    var extra = [_]?*anyopaque{null};

    // Hash tables always live in global memory; no dynamic shared mem needed.
    const shared_bytes: u32 = 0;
    const t_lz = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzEncodeKernel", 0);
    // Defer endKernelTiming so the pending begin event always gets a
    // matching end record - even on launch failure. Otherwise
    // finalizeProfiling would block on an unrecorded end event.
    defer gpu_decode.endKernelTiming(t_lz, 0);
    if (launch_fn(module_loader.kernel_fn, num_chunks, 1, 1, 32, 1, 1, shared_bytes, 0, &params, &extra) != cuda_ffi.CUDA_SUCCESS)
        return false;

    if (sync_fn() != cuda_ffi.CUDA_SUCCESS) return false;

    if (t_before) |t_start| {
        if (io) |io_val| {
            // Storage lives on the facade so external callers reach it
            // unchanged via `gpu_enc.last_kernel_ns`.
            const driver = @import("driver.zig");
            driver.last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // Download comp_sizes first, then only the actual compressed bytes per block
    if (d2h_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes) != cuda_ffi.CUDA_SUCCESS) return false;

    for (0..chunk_descs.len) |i| {
        const cs = comp_sizes_out[i];
        if (cs > 0) {
            const dst_off = chunk_descs[i].dst_offset;
            if (d2h_fn(@ptrCast(output.ptr + dst_off), d_output + dst_off, cs) != cuda_ffi.CUDA_SUCCESS) return false;
        }
    }

    return true;
}
