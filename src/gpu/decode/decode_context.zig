//! Per-decode-operation mutable state plus the helpers that pull bytes
//! across the H/D boundary and the kernel-profiling event plumbing.
//!
//! The library exposes one `g_default` context for module-level callers;
//! per-handle library users (streamlz_gpu.zig) instantiate their own
//! `DecodeContext` alongside the matching encode context so a future
//! multi-stream API does not have to thread mutable globals.

const std = @import("std");

const cuda = @import("cuda_api.zig");
const d = @import("descriptors.zig");

const CUdeviceptr = cuda.CUdeviceptr;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;

// The four `last_*_ns` telemetry vars and `g_default` live in driver.zig
// (the facade) because Zig cannot expose a `pub var` from another module
// through a `pub const` alias — external callers reading
// `gpu_decode.last_kernel_ns` must reach actual storage in the facade.
// Sub-modules write back via `@import("driver.zig").last_kernel_ns = ...`.

pub fn ensureDeviceBuf(ptr: *CUdeviceptr, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    if (ptr.* != 0) _ = (cuda.cuMemFree_fn orelse return false)(ptr.*);
    current_size.* = 0;
    if ((cuda.cuMemAlloc_fn orelse return false)(ptr, needed) != CUDA_SUCCESS) return false;
    current_size.* = needed;
    return true;
}

pub fn ensureDeviceOutput(self: *DecodeContext, size: usize) bool {
    return ensureDeviceBuf(&self.d_output, &self.d_output_size, size);
}

/// Page-locked (pinned) host allocation. D2H/H2D against pinned memory runs
/// at full PCIe bandwidth (~2x pageable, which the driver stages chunk-wise
/// through an internal pinned buffer) and is genuinely async-capable.
/// Returns null when CUDA is unavailable — caller falls back to a normal
/// allocation, so this is always safe to attempt.
pub fn allocHost(size: usize) ?[]u8 {
    if (!@import("module_loader.zig").init()) return null;
    const f = cuda.cuMemAllocHost_fn orelse return null;
    var p: ?*anyopaque = null;
    if (f(&p, size) != CUDA_SUCCESS) return null;
    const base = p orelse return null;
    return @as([*]u8, @ptrCast(base))[0..size];
}

pub fn freeHost(buf: []u8) void {
    const f = cuda.cuMemFreeHost_fn orelse return;
    _ = f(@ptrCast(buf.ptr));
}

/// Copy `dst.len` bytes from a device address into the host slice `dst`.
/// Requires init() to have succeeded. Returns false on any CUDA failure.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    if (dst.len == 0) return true;
    const f = cuda.cuMemcpyDtoH_fn orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == CUDA_SUCCESS;
}

/// Copy the host slice `src` to a device address. Requires init() to have
/// succeeded. Returns false on any CUDA failure.
pub fn copyHostToDevice(dst_device: u64, src: []const u8) bool {
    if (src.len == 0) return true;
    const f = cuda.cuMemcpyHtoD_fn orelse return false;
    return f(dst_device, @ptrCast(src.ptr), src.len) == CUDA_SUCCESS;
}

/// Make the library's CUDA context current on the calling thread. A
/// driver-API context is current per-thread, so any thread that issues
/// GPU work (e.g. a library-owned worker thread) must call this first.
/// Requires init() to have succeeded.
pub fn bindContextToCallingThread() bool {
    if (cuda.ctx == 0) return false;
    const f = cuda.cuCtxSetCurrent_fn orelse return false;
    return f(cuda.ctx) == CUDA_SUCCESS;
}

/// Allocate + record a start event for a kernel about to launch on `stream`.
/// Returns the end-event handle to pass to `endKernelTiming`, or null when
/// profiling is disabled or any CUDA call failed (timing is best-effort —
/// never blocks the encode/decode path).
pub fn beginKernelTiming(
    ctx_enabled: bool,
    pending: *std.ArrayListUnmanaged(d.PendingTiming),
    name: [*:0]const u8,
    stream: usize,
) ?usize {
    if (!ctx_enabled) return null;
    const create_fn = cuda.cuEventCreate_fn orelse return null;
    const record_fn = cuda.cuEventRecord_fn orelse return null;
    var start: usize = 0;
    var end: usize = 0;
    if (create_fn(&start, 0) != CUDA_SUCCESS) return null;
    if (create_fn(&end, 0) != CUDA_SUCCESS) {
        if (cuda.cuEventDestroy_fn) |dd| _ = dd(start);
        return null;
    }
    if (record_fn(start, stream) != CUDA_SUCCESS) {
        if (cuda.cuEventDestroy_fn) |dd| { _ = dd(start); _ = dd(end); }
        return null;
    }
    pending.append(std.heap.page_allocator, .{
        .name = name, .start_event = start, .end_event = end,
    }) catch {
        if (cuda.cuEventDestroy_fn) |dd| { _ = dd(start); _ = dd(end); }
        return null;
    };
    return end;
}

pub fn endKernelTiming(end_event: ?usize, stream: usize) void {
    const e = end_event orelse return;
    const record_fn = cuda.cuEventRecord_fn orelse return;
    _ = record_fn(e, stream);
}

/// Synchronize on each pending event pair and compute elapsed times into
/// `last_timings`. Destroys the events. Idempotent (safe to call when
/// pending is empty).
pub fn finalizeProfiling(
    pending: *std.ArrayListUnmanaged(d.PendingTiming),
    last_timings: *std.ArrayListUnmanaged(d.KernelTiming),
) void {
    // Idempotent when pending is empty: leave last_timings untouched so the
    // values populated by the prior call survive. slzGetLastTimings calls
    // this from the main thread after the worker's fullGpuLaunchImpl has
    // already drained pending into last_timings — wiping here would zero
    // those timings out.
    if (pending.items.len == 0) return;
    last_timings.clearRetainingCapacity();
    const sync_fn = cuda.cuEventSynchronize_fn orelse {
        pending.clearRetainingCapacity();
        return;
    };
    const elapsed_fn = cuda.cuEventElapsedTime_fn orelse {
        pending.clearRetainingCapacity();
        return;
    };
    const destroy_fn = cuda.cuEventDestroy_fn orelse {
        pending.clearRetainingCapacity();
        return;
    };
    for (pending.items) |p| {
        _ = sync_fn(p.end_event);
        var ms: f32 = 0.0;
        _ = elapsed_fn(&ms, p.start_event, p.end_event);
        last_timings.append(std.heap.page_allocator, .{ .name = p.name, .ms = ms }) catch {};
        _ = destroy_fn(p.start_event);
        _ = destroy_fn(p.end_event);
    }
    pending.clearRetainingCapacity();
}

/// Per-decode-operation mutable state. Every device-buffer pointer, buffer
/// size, host scratch array, and per-operation flag formerly held as a
/// module-global lives here so a future library API can hand each handle
/// its own context. Load-once module handles, kernel/driver function
/// pointers, and `pub var` telemetry stay module-global on purpose.
pub const DecodeContext = struct {
    // ── Output buffer ──────────────────────────────────────────
    d_output: CUdeviceptr = 0,
    d_output_size: usize = 0,

    // ── Persistent compressed-input + descriptor buffers ───────
    d_comp_persist: CUdeviceptr = 0,
    d_comp_persist_size: usize = 0,
    d_descs_persist: CUdeviceptr = 0,
    d_descs_persist_size: usize = 0,

    // ── Entropy-decoded scratch ────────────────────────────────
    // Holds the Huffman pre-decode output that the LZ kernel reads.
    // Layout per sub-chunk slot: [lit][tok at +tok_offset][off16 at +off16_offset].
    // off16 sub-region: hi bytes at chunk_idx*131072, lo bytes at +65536.
    d_entropy_scratch: CUdeviceptr = 0,
    d_entropy_scratch_size: usize = 0,

    h_pinned_output: ?[*]u8 = null,
    h_pinned_output_size: usize = 0,

    // Off16 scratch view = d_entropy_scratch + off16_offset (set in fullGpuLaunchImpl).
    // Used by the raw-off16 gather kernel and the D2D/H2D fallback loops.
    d_entropy_off16_scratch: CUdeviceptr = 0,
    d_entropy_off16_scratch_size: usize = 0,

    // Raw off16 gather descriptors (one per raw off16 sub-stream).
    d_raw_off16_descs: CUdeviceptr = 0,
    d_raw_off16_descs_size: usize = 0,

    // Per-chunk first-subchunk-index buffer (multi-sub-chunk-per-chunk support).
    // At sc>0.5 chunks have multiple sub-chunks each; each sub-chunk needs its
    // own entropy scratch slot. This array maps chunk_idx → global sub-chunk index.
    d_first_subchunk_idx: CUdeviceptr = 0,
    d_first_subchunk_idx_size: usize = 0,

    raw_off16_buf: [8192]d.RawOff16Desc = undefined,

    // GPU decode-scan staged buffers (roadmap 4d Phase 2). d_scan_staged
    // packs the six staged arrays (lit/tok/hi/lo huff + raw hi/lo);
    // d_scan_first_sub holds the per-chunk first-sub-chunk prefix sum.
    d_scan_staged: CUdeviceptr = 0,
    d_scan_staged_size: usize = 0,
    d_scan_first_sub: CUdeviceptr = 0,
    d_scan_first_sub_size: usize = 0,

    // GPU frame-walk kernel scratch (roadmap 4d Phase 3). d_walk_chunks
    // holds the kernel-produced ChunkDesc array; d_walk_meta is a single
    // 6-u32 region for (n_chunks, decomp_size, sub_chunk_cap,
    // block_start, block_size, status).
    d_walk_chunks: CUdeviceptr = 0,
    d_walk_chunks_size: usize = 0,
    d_walk_meta: CUdeviceptr = 0,
    d_walk_meta_size: usize = 0,

    // Pure-D2D prefix-sum scratch (step 2): d_first_sub_idx holds the
    // per-chunk first-sub-chunk index; d_total_subchunks_buf is a single
    // u32 with the running total (device-resident, never D2H'd on the
    // pure path).
    d_first_sub_idx_persist: CUdeviceptr = 0,
    d_first_sub_idx_persist_size: usize = 0,
    d_total_subchunks_buf: CUdeviceptr = 0,
    d_total_subchunks_buf_size: usize = 0,

    // Pure-D2D compaction outputs (step 6b). Each compact buffer holds
    // a compacted per-stream HuffDecChunkDesc array (slzCompactHuffDescs-
    // Kernel output). d_compact_raw holds the interleaved hi/lo
    // RawOff16Descs (slzCompactRawDescsKernel output). d_compact_counts
    // is a 6 u32 region: [n_lit, n_tok, n_hi, n_lo, n_raw, n_merged].
    d_compact_lit: CUdeviceptr = 0,
    d_compact_lit_size: usize = 0,
    d_compact_tok: CUdeviceptr = 0,
    d_compact_tok_size: usize = 0,
    d_compact_hi: CUdeviceptr = 0,
    d_compact_hi_size: usize = 0,
    d_compact_lo: CUdeviceptr = 0,
    d_compact_lo_size: usize = 0,
    d_compact_raw: CUdeviceptr = 0,
    d_compact_raw_size: usize = 0,
    d_compact_counts: CUdeviceptr = 0,
    d_compact_counts_size: usize = 0,

    // Step 7 launch-plumbing scratch: device-resident 4 B counters used
    // to feed kernel self-gates when the GPU compact path didn't run
    // (CPU-scan fallback). The pure-D2D path consumes d_compact_counts
    // slots directly and never touches these.
    d_n_raw_scratch: CUdeviceptr = 0,
    d_n_raw_scratch_size: usize = 0,
    d_n_huff_scratch: CUdeviceptr = 0,
    d_n_huff_scratch_size: usize = 0,
    d_n_chunks_scratch: CUdeviceptr = 0,
    d_n_chunks_scratch_size: usize = 0,
    d_n_groups_scratch: CUdeviceptr = 0,
    d_n_groups_scratch_size: usize = 0,

    // Huffman descriptors + LUT.
    d_huff_descs: CUdeviceptr = 0,
    d_huff_descs_size: usize = 0,
    d_huff_lut: CUdeviceptr = 0,
    d_huff_lut_size: usize = 0,
    // One host buffer per stream type — scanner appends to each; fullGpuLaunch
    // merges them into one device array with per-type out_offset added.
    huff_lit_host_buf: [4096]d.HuffDecChunkDesc = undefined,
    huff_tok_host_buf: [4096]d.HuffDecChunkDesc = undefined,
    huff_off16hi_host_buf: [4096]d.HuffDecChunkDesc = undefined,
    huff_off16lo_host_buf: [4096]d.HuffDecChunkDesc = undefined,

    // Pipeline streams (persistent, created once in init)
    pipeline_streams: [cuda.NUM_PIPELINE_STREAMS]usize = .{0} ** cuda.NUM_PIPELINE_STREAMS,
    pipeline_streams_created: bool = false,

    // Per-kernel timing (slzDecompressOpts_t.enable_profiling). When true,
    // every kernel launch in fullGpuLaunchImpl records a cuEvent pair and
    // appends to `pending_timings`. `finalizeProfiling` (called after the
    // final sync) drains pending → `last_timings`, which slzGetLastTimings
    // reads out via the C ABI.
    enable_profiling: bool = false,
    pending_timings: std.ArrayListUnmanaged(d.PendingTiming) = .empty,
    last_timings: std.ArrayListUnmanaged(d.KernelTiming) = .empty,

    // CUDA stream used for the heavy-phase kernel launches (huff build/
    // decode, LZ decode) and the final D2D output copy. slzDecompressAsync
    // sets it to the caller's stream so cudaStreamSynchronize on that
    // stream waits for the decompress to complete. The sync slzDecompress
    // wrapper leaves it at 0 (default stream) and waits with cuCtxSync.
    // The walk + scan + compact + merge phases still serialize on stream 0
    // (they share host-dependent values like total_subchunks); only the
    // back half rides the caller's stream.
    work_stream: usize = 0,
};

// `g_default` lives in driver.zig (the facade) — see the telemetry comment
// at the top of this file for why storage of `pub var`s must be there.
