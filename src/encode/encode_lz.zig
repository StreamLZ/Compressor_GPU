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

/// A-023 (backported from srcVK, 2026-06-09) testing hook: caps the
/// initial batch_count regardless of how much VRAM the device has, so
/// regression tests can drive the batched-dispatch path on inputs that
/// would otherwise fit in a single dispatch. 0 = inactive (production).
/// The SLZ_FORCE_BATCH env var mirrors this for command-line runs.
pub var g_force_batch_count_for_test: u32 = 0;

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
    //
    // A-023 (backported from srcVK/encode/encode_lz.zig, 2026-06-09):
    // batched dispatch when the full per-chunk hash does not fit VRAM.
    // At sc=0.25 a 1 GB input is 15,259 chunks; the L5 chain-parser
    // tables then total ~18 GB — cuMemAlloc under WDDM accepts that and
    // silently pages over PCIe, collapsing encode to ~50 MB/s. The
    // kernel already addresses a per-chunk hash region (base = chunk_id
    // × stride) and re-initialises it at entry, so dispatching the grid
    // in batches over a smaller hash buffer produces BYTE-IDENTICAL
    // output. VK discovers the cap by VMA allocation failure (strict
    // DEVICE_LOCAL); CUDA never fails the alloc, so the cap comes from
    // a cuMemGetInfo budget (3/4 of free VRAM — the remainder is
    // headroom for the downstream Huffman + assemble buffers), with the
    // VK-style ensureBuf-halving loop kept as a backstop.
    const next_hash_words: usize = encode_context.NEXT_HASH_ENTRIES / 2; // u16 entries packed into u32 words
    const hash_stride: usize = if (chain)
        hash_size + hash_size + next_hash_words
    else
        hash_size;

    var batch_count: u32 = num_chunks;
    if (g_force_batch_count_for_test != 0 and g_force_batch_count_for_test < batch_count)
        batch_count = g_force_batch_count_for_test;
    if (std.c.getenv("SLZ_FORCE_BATCH")) |raw| {
        const span = std.mem.span(raw);
        if (std.fmt.parseInt(u32, span, 10) catch null) |forced| {
            if (forced > 0 and forced < batch_count) batch_count = forced;
        }
    }
    if (cuda_ffi.cuMemGetInfo_fn) |getinfo_fn| {
        var free_b: usize = 0;
        var total_b: usize = 0;
        if (getinfo_fn(&free_b, &total_b) == cuda_ffi.CUDA_SUCCESS) {
            const budget = free_b - (free_b / 4);
            const per_chunk_bytes = hash_stride * 4;
            const fit_usize = @max(@as(usize, 1), budget / per_chunk_bytes);
            const fit: u32 = @intCast(@min(fit_usize, @as(usize, num_chunks)));
            if (fit < batch_count) batch_count = fit;
        }
    }
    while (true) {
        const hash_bytes = @as(usize, batch_count) * hash_stride * 4;
        if (encode_context.ensureBuf(&self.d_hash_persist, &self.d_hash_size, hash_bytes)) break;
        if (batch_count <= 1) return false;
        batch_count = (batch_count + 1) / 2;
    }
    const batched_dispatch = batch_count < num_chunks;

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
    // A-023: when not batching, upload the full descs / clear the full
    // sizes array up-front (the historical shape). When batching, the
    // per-batch loop below uploads only the active batch's descs and
    // clears only the active batch's sizes slots.
    if (!batched_dispatch) {
        if (h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
        if (cuda_ffi.cuMemsetD8_fn) |memset_fn|
            if (memset_fn(d_sizes, 0, sizes_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    }
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
    // v4 #6 experiment (2026-06-10): L2 = greedy+rehash WITHOUT entropy -
    // re-differentiates L2 from L1 (hb=17-everywhere had collapsed them).
    var p_l4: u32 = if (level >= 4 or level == 2) 1 else 0;

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

    // A-023: batched kernel dispatch loop. When batch_count ==
    // num_chunks (the common case), this runs ONCE with the exact
    // historical launch shape. When batch_count < num_chunks (the
    // VRAM-pressure fallback), each iteration uploads the batch's descs
    // to offset 0 of the persistent buffers — the kernel then reads
    // them with local chunk_id = 0..bc, and its hash region index
    // (chunk_id × stride) stays inside the smaller hash buffer. The
    // descs carry GLOBAL src_offset / dst_offset values, so input reads
    // and output writes land at the right absolute positions without
    // further translation. Per-chunk results are BYTE-IDENTICAL to the
    // unbatched dispatch because every chunk re-initialises its own
    // hash region at kernel entry.
    var batch_start: u32 = 0;
    while (batch_start < num_chunks) {
        const batch_end = @min(batch_start + batch_count, num_chunks);
        const bc: u32 = batch_end - batch_start;

        if (batched_dispatch) {
            const batch_desc_bytes = @as(usize, bc) * @sizeOf(CompressChunkDesc);
            if (h2d_fn(d_descs, @ptrCast(&chunk_descs[batch_start]), batch_desc_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
            if (cuda_ffi.cuMemsetD8_fn) |memset_fn|
                if (memset_fn(d_sizes, 0, @as(usize, bc) * 4) != cuda_ffi.CUDA_SUCCESS) return false;
            if (sync_fn() != cuda_ffi.CUDA_SUCCESS) return false;
        }

        p_total = bc;
        if (launch_fn(module_loader.kernel_fn, bc, 1, 1, 32, 1, 1, shared_bytes, 0, &params, &extra) != cuda_ffi.CUDA_SUCCESS)
            return false;
        if (sync_fn() != cuda_ffi.CUDA_SUCCESS) return false;

        if (batched_dispatch) {
            // Per-batch D2H: comp_sizes land at their GLOBAL positions
            // so downstream passes (Huffman / assemble) read them at
            // the same index they would have in the unbatched path.
            // Payload bytes only at L3+ (see the unbatched note - the
            // Huffman passes are the sole host-side consumers).
            if (d2h_fn(@ptrCast(comp_sizes_out.ptr + batch_start), d_sizes, @as(usize, bc) * 4) != cuda_ffi.CUDA_SUCCESS) return false;
            if (level >= 3) {
                for (batch_start..batch_end) |i| {
                    const cs = comp_sizes_out[i];
                    if (cs > 0) {
                        const dst_off = chunk_descs[i].dst_offset;
                        if (d2h_fn(@ptrCast(output.ptr + dst_off), d_output + dst_off, cs) != cuda_ffi.CUDA_SUCCESS) return false;
                    }
                }
            }
        }

        batch_start = batch_end;
    }

    if (t_before) |t_start| {
        if (io) |io_val| {
            // Storage lives on the facade so external callers reach it
            // unchanged via `gpu_enc.last_kernel_ns`.
            const driver = @import("driver.zig");
            driver.last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // Unbatched path: download comp_sizes first (always needed - the
    // assemble descs and frame splice are built from them), then the
    // compressed payload bytes - but ONLY at L3+: the Huffman passes
    // are the sole consumers of the host-side LZ bytes. At L1/L2 the
    // device-resident assemble reads d_output directly on the GPU and
    // the finished frame returns via the single wrap_d2h copy, so the
    // per-chunk payload gather was pure waste (~12-17 ms at enwik8
    // scale; v4 #17 measurement).
    if (!batched_dispatch) {
        if (d2h_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes) != cuda_ffi.CUDA_SUCCESS) return false;
    }
    if (!batched_dispatch and level >= 3) {
        // v4 #17 fast path: queue one cuMemcpyDtoHAsync per chunk into
        // persistent PINNED staging (true DMA — the copy engine
        // pipelines all regions), sync ONCE, then splice the spans to
        // the caller's pageable `output`. Replaces ~N synchronous
        // pageable D2H calls (~45 us each; ~17 ms at enwik8 scale,
        // vs VK's 4.5 ms multi-region gather — the A-018 reference).
        var gathered = false;
        if (cuda_ffi.cuMemcpyDtoHAsync_fn) |d2h_async| {
            if (cuda_ffi.cuStreamSync_fn) |stream_sync| {
                if (encode_context.ensurePinnedGather(self, output.len)) |pinned| {
                    var ok = true;
                    for (0..chunk_descs.len) |i| {
                        const cs = comp_sizes_out[i];
                        if (cs > 0) {
                            const dst_off = chunk_descs[i].dst_offset;
                            if (d2h_async(@ptrCast(pinned + dst_off), d_output + dst_off, cs, 0) != cuda_ffi.CUDA_SUCCESS) {
                                ok = false;
                                break;
                            }
                        }
                    }
                    if (ok and stream_sync(0) == cuda_ffi.CUDA_SUCCESS) {
                        for (0..chunk_descs.len) |i| {
                            const cs = comp_sizes_out[i];
                            if (cs > 0) {
                                const dst_off = chunk_descs[i].dst_offset;
                                @memcpy(output[dst_off..][0..cs], pinned[dst_off..][0..cs]);
                            }
                        }
                        gathered = true;
                    } else {
                        // A failed async queue leaves copies in flight;
                        // drain before the sync fallback reuses d_output.
                        _ = stream_sync(0);
                    }
                }
            }
        }
        if (!gathered) {
            for (0..chunk_descs.len) |i| {
                const cs = comp_sizes_out[i];
                if (cs > 0) {
                    const dst_off = chunk_descs[i].dst_offset;
                    if (d2h_fn(@ptrCast(output.ptr + dst_off), d_output + dst_off, cs) != cuda_ffi.CUDA_SUCCESS) return false;
                }
            }
        }
    }

    // A-023 companion: when the dispatch was batched, release the hash
    // buffer so the downstream Huffman + assemble passes have VRAM room
    // (~multi-GB at 1 GB inputs). It re-allocates on the next call; for
    // unbatched dispatches the buffer stays persistent so warm encodes
    // pay zero alloc overhead.
    if (batched_dispatch) {
        if (cuda_ffi.cuMemFree_fn) |free_fn| {
            if (self.d_hash_persist != 0) {
                _ = free_fn(self.d_hash_persist);
                self.d_hash_persist = 0;
                self.d_hash_size = 0;
            }
        }
    }

    return true;
}
