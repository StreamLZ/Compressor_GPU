//! 1:1 port of src/encode/encode_lz.zig.
//!
//! L1 hot path: dispatches the LZ encode pipeline (lz_encode_kernel.comp)
//! to produce per-chunk compressed streams. All device ops funnel through
//! `vulkan_api.procs.*` — no direct vk*/vma calls in this file.

const std = @import("std");
const vk = @import("../decode/vulkan_api.zig");
const module_loader = @import("module_loader.zig");
const encode_context = @import("encode_context.zig");
const levels = @import("levels.zig");
const gpu_decode = @import("../decode/driver.zig");

const VkDeviceBuffer = vk.VkDeviceBuffer;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;
const EncodeContext = encode_context.EncodeContext;
const CompressChunkDesc = encode_context.CompressChunkDesc;

/// CUDA reference: src/encode/encode_lz.zig:21-153. L1 hot LZ encode
/// launcher. Returns true on success, false on FFI / GPU failure.
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

    // VK adaptation: parallel-test serialization for this entry is
    // applied at compressFramedOne (fast_framed.zig) — it must span
    // gpuCompressImpl + the assemble trio so the inter-call shared
    // device buffers on enc_driver.g_default don't get clobbered by
    // a sibling worker between calls. See lockEncodeDispatcherMutex
    // comment in decode/module_loader.zig.

    // VK adaptation (H2): when `self.work_stream` is non-zero the
    // EncodeContext has been promoted onto a pipeline_stream by
    // compressFramedOne; route every H2D/D2H/sync through the stream
    // variants so the encode pipeline batches into one cmdbuf/submit
    // instead of the 20 submit+wait round-trips the singleton
    // g_command_buffer path took (each ~11 ms per nsys trace). The
    // launch_fn always rides the same stream — when 0 it falls back
    // to the singleton command buffer.
    const work_stream = self.work_stream;
    const use_stream = work_stream != 0;
    const h2d_fn = vk.procs.h2d orelse return false;
    const h2d_async_fn = vk.procs.h2d_async orelse return false;
    const d2h_fn = vk.procs.d2h orelse return false;
    const d2h_async_fn = vk.procs.d2h_async orelse return false;
    const d2h_offset_fn = vk.procs.d2h_offset orelse return false;
    const launch_fn = vk.procs.launch_kernel orelse return false;
    const sync_fn = vk.procs.ctx_sync orelse return false;
    const stream_sync_fn = vk.procs.stream_sync orelse return false;

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
    // long_hash + next_hash); non-chain modes use a single hash table.
    if (chain) {
        const next_hash_words: usize = encode_context.NEXT_HASH_ENTRIES / 2;
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
    // via D2D (no host bounce) instead of an H2D from the host `input`
    // slice.
    //
    // VK adaptation (H2): on the stream path the descriptor + input
    // H2D + memset_d8 all queue onto the stream's transfer / compute
    // cmdbufs. The big enwik8 95 MB input H2D auto-fires the iter-12
    // VK_EXT_external_memory_host import path inside procH2DAsync
    // when host_addr is page-aligned and size >= H2D_IMPORT_THRESHOLD
    // (4 MiB). No procs.stream_sync between H2D + launch — the
    // semaphore handshake the split-submit emits already orders the
    // compute submit behind the transfer signal.
    if (self.d_input_override != 0) {
        const d2d = vk.procs.d2d orelse return false;
        if (d2d(d_input, self.d_input_override, input.len, work_stream) != VK_SUCCESS_RC) return false;
    } else if (use_stream) {
        if (h2d_async_fn(d_input, @ptrCast(input.ptr), input.len, work_stream) != VK_SUCCESS_RC) return false;
    } else {
        if (h2d_fn(d_input, @ptrCast(input.ptr), input.len) != VK_SUCCESS_RC) return false;
    }
    if (use_stream) {
        if (h2d_async_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes, work_stream) != VK_SUCCESS_RC) return false;
        if (vk.procs.memset_d8_async) |memset_async_fn| {
            if (memset_async_fn(d_sizes, 0, sizes_bytes, work_stream) != VK_SUCCESS_RC) return false;
        } else if (vk.procs.memset_d8) |memset_fn| {
            // Fallback: sync memset and then nothing — the stream H2Ds
            // above will still serialize behind it because procs.memset_d8
            // submits+waits inline.
            if (memset_fn(d_sizes, 0, sizes_bytes) != VK_SUCCESS_RC) return false;
        }
    } else {
        if (h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes) != VK_SUCCESS_RC) return false;
        if (vk.procs.memset_d8) |memset_fn|
            if (memset_fn(d_sizes, 0, sizes_bytes) != VK_SUCCESS_RC) return false;
        if (sync_fn() != VK_SUCCESS_RC) return false;
    }

    const t_before = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_input = d_input;
    var p_output = d_output;
    var p_descs = d_descs;
    var p_global_hash: VkDeviceBuffer = self.d_hash_persist;
    var p_sizes = d_sizes;
    var p_total = num_chunks;
    var p_hash_bits = hash_bits;
    var p_use_chain: u32 = if (chain) 1 else 0;
    // L4+ enables the greedy parser's match-range rehash (CPU engine_level>=2).
    // L3 stays without it - that is the L3/L4 distinction. L5 uses the chain
    // parser so the flag is inert there.
    var p_l4: u32 = if (level >= 4) 1 else 0;

    // params layout per procs.launch_kernel contract: the first
    // n_bindings entries are pointers to VkDeviceBuffer handles
    // (the shader's std430 buffer bindings 0..n_bindings-1), the
    // remaining entries are pointers to push-constant scalars packed
    // into the push_constant_size byte buffer in declaration order.
    // lz_encode: n_bindings=5 (Input, Output, Descs, GlobalHash,
    // CompSizes), push_constant_size=16 (total_chunks, hash_bits,
    // use_chain, l4_features).
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

    const shared_bytes: c_uint = 0;
    const t_lz = gpu_decode.beginKernelTiming(self.enable_profiling, &self.pending_timings, "slzLzEncodeKernel", work_stream);
    // Defer endKernelTiming so the pending begin event always gets a
    // matching end record - even on launch failure.
    defer gpu_decode.endKernelTiming(t_lz, work_stream);
    if (launch_fn(module_loader.kernel_fn, num_chunks, 1, 1, 32, 1, 1, shared_bytes, work_stream, &params, &extra) != VK_SUCCESS_RC)
        return false;

    // VK adaptation (H2): the post-LZ D2H of comp_sizes needs to wait
    // on the LZ kernel. On the stream path it queues into the same
    // stream cmdbuf as a TRANSFER_READ following the compute write;
    // the post-loop stream_sync below joins everything. On the sync
    // path we still need the explicit ctx_sync before the D2H so the
    // comp_sizes buffer has landed.
    if (!use_stream) {
        if (sync_fn() != VK_SUCCESS_RC) return false;
    }

    if (t_before) |t_start| {
        if (io) |io_val| {
            // Storage lives on the encode driver facade so external callers
            // reach it unchanged via `gpu_enc.last_kernel_ns`.
            const driver = @import("driver.zig");
            driver.last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // Download comp_sizes first, then only the actual compressed bytes per block.
    //
    // VK adaptation (H2): on the stream path comp_sizes_out's D2H
    // queues into the same stream cmdbuf as the LZ kernel; we then
    // FLUSH with one stream_sync so the CPU can branch on per-chunk
    // sizes for the loop below. The per-chunk D2H still uses the
    // sync d2h_offset slot (no async variant currently exists with a
    // srcOffset arg) — those copies fire after the stream is
    // already drained so the dependency is implicit.
    if (use_stream) {
        if (d2h_async_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes, work_stream) != VK_SUCCESS_RC) return false;
        // Single submit/wait batching every queued H2D + the LZ
        // kernel + the comp_sizes D2H — this is the win the H2 plumbing
        // exists for. Replaces the ~20 ctx_sync round-trips that
        // procH2D/procD2H/procLaunchKernel paid through the singleton
        // g_command_buffer.
        if (stream_sync_fn(work_stream) != VK_SUCCESS_RC) return false;
    } else {
        if (d2h_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes) != VK_SUCCESS_RC) return false;
    }

    // VK adaptation: mirrors CUDA's per-chunk D2H loop at
    // src/encode/encode_lz.zig:144-150 verbatim. CUDA's
    // cuMemcpyDtoH(host + dst_off, d_output + dst_off, cs) folds the
    // device-side offset into pointer arithmetic on the real device VA;
    // on Vulkan VkDeviceBuffer is a registry handle, so the device-side
    // offset travels through the procs.d2h_offset slot's srcOffset
    // argument and ends up in VkBufferCopy.srcOffset. Net result: only
    // sum(comp_sizes) bytes leave the device instead of the full
    // n_gpu_blocks * per_block_cap output capacity (763 sub-blocks
    // × 384 KB = ~285 MB on enwik8 fell to ~50 MB compressed).
    for (0..chunk_descs.len) |i| {
        const cs = comp_sizes_out[i];
        if (cs > 0) {
            const dst_off = chunk_descs[i].dst_offset;
            if (d2h_offset_fn(@ptrCast(output.ptr + dst_off), d_output, cs, dst_off) != VK_SUCCESS_RC) return false;
        }
    }

    return true;
}
