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
// through a `pub const` alias - external callers reading
// `gpu_decode.last_kernel_ns` must reach actual storage in the facade.
// Sub-modules write back via `@import("driver.zig").last_kernel_ns = ...`.

pub fn ensureDeviceBuf(ptr: *CUdeviceptr, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    // The free's result is intentionally dropped: at this point we
    // are about to grow the buffer regardless, and a failed free leaks at
    // worst a single allocation (cleaned up at process exit / context
    // destroy). Surfacing the failure here would force every caller to
    // distinguish "alloc failed" from "free failed before alloc", which
    // adds no actionable signal.
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
/// Returns null when CUDA is unavailable - caller falls back to a normal
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
/// profiling is disabled or any CUDA call failed (timing is best-effort -
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
        if (cuda.cuEventDestroy_fn) |destroy_fn| _ = destroy_fn(start);
        return null;
    }
    if (record_fn(start, stream) != CUDA_SUCCESS) {
        if (cuda.cuEventDestroy_fn) |destroy_fn| { _ = destroy_fn(start); _ = destroy_fn(end); }
        return null;
    }
    pending.append(std.heap.page_allocator, .{
        .name = name, .start_event = start, .end_event = end,
    }) catch {
        if (cuda.cuEventDestroy_fn) |destroy_fn| { _ = destroy_fn(start); _ = destroy_fn(end); }
        return null;
    };
    return end;
}

/// Record the end event of a kernel-timing pair. The pair was appended to
/// `pending_timings` by `beginKernelTiming`; this is the matching second
/// half.
///
/// Failure contract: if `cuEventRecord` fails here (e.g. invalid stream,
/// context lost), the end-event is never armed, so the matching
/// `cuEventSynchronize` in `finalizeProfiling` will block forever waiting
/// for an event that the driver will never complete. We accept this risk
/// because profiling is opt-in (only enabled when the caller sets
/// `enable_profiling`) and any CUDA failure at this point already implies
/// the surrounding decode launch is in a non-recoverable state - the next
/// `cuCtxSynchronize` in the dispatch path will surface the real error
/// before `finalizeProfiling` is reached.
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
    // already drained pending into last_timings - wiping here would zero
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
    // The four CUDA-result drops below (sync / elapsed / destroy ×2)
    // are swallowed by design - profiling is opt-in and a transient driver
    // hiccup here only loses a single ms reading, never decode correctness.
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
    // Pinned host mirror of d_output for the D2H copy on the synchronous
    // decompress path. Grouped here so output-related state lives
    // together.
    h_pinned_output: ?[*]u8 = null,
    h_pinned_output_size: usize = 0,

    // ── Persistent compressed-input + descriptor buffers ───────
    d_comp_persist: CUdeviceptr = 0,
    d_comp_persist_size: usize = 0,
    d_descs_persist: CUdeviceptr = 0,
    d_descs_persist_size: usize = 0,

    // ── Entropy-decoded scratch ────────────────────────────────
    // Holds the Huffman pre-decode output that the LZ kernel reads.
    // Layout per sub-chunk slot: [lit][tok at +tok_offset][off16 at +off16_offset].
    // off16 sub-region: hi bytes at chunk_idx*d.ENTROPY_SCRATCH_SLOT_BYTES,
    // lo bytes at +d.OFF16_HILO_SPLIT_OFFSET.
    d_entropy_scratch: CUdeviceptr = 0,
    d_entropy_scratch_size: usize = 0,

    // Off16 scratch view = d_entropy_scratch + off16_offset (set in fullGpuLaunchImpl).
    // The raw-off16 gather kernel scatters compressed raw bytes here.
    d_entropy_off16_scratch: CUdeviceptr = 0,
    d_entropy_off16_scratch_size: usize = 0,

    // Per-chunk first-subchunk-index buffer (multi-sub-chunk-per-chunk support).
    // At sc>0.5 chunks have multiple sub-chunks each; each sub-chunk needs its
    // own entropy scratch slot. This array maps chunk_idx → global sub-chunk index.
    d_first_subchunk_idx: CUdeviceptr = 0,
    d_first_subchunk_idx_size: usize = 0,

    // GPU decode-scan staged buffers. d_scan_staged packs the six staged
    // arrays the scan kernel writes (lit/tok/hi/lo huff descriptor lists
    // + raw hi/lo gather lists); d_scan_first_sub holds the per-chunk
    // first-sub-chunk prefix sum.
    d_scan_staged: CUdeviceptr = 0,
    d_scan_staged_size: usize = 0,
    d_scan_first_sub: CUdeviceptr = 0,
    d_scan_first_sub_size: usize = 0,

    // GPU frame-walk kernel scratch. d_walk_chunks holds the
    // kernel-produced ChunkDesc array; d_walk_meta is a 6-u32 region
    // (n_chunks, decomp_size, sub_chunk_cap, block_start, block_size,
    // status).
    d_walk_chunks: CUdeviceptr = 0,
    d_walk_chunks_size: usize = 0,
    d_walk_meta: CUdeviceptr = 0,
    d_walk_meta_size: usize = 0,

    // Pure-D2D prefix-sum scratch: d_first_sub_idx holds the per-chunk
    // first-sub-chunk index; d_total_subchunks_buf is a single u32 with
    // the running total (device-resident, never D2H'd on the pure path).
    d_first_sub_idx_persist: CUdeviceptr = 0,
    d_first_sub_idx_persist_size: usize = 0,
    d_total_subchunks_buf: CUdeviceptr = 0,
    d_total_subchunks_buf_size: usize = 0,

    // Pure-D2D compaction outputs. Each `d_compact_*` buffer holds a
    // compacted per-stream HuffDecChunkDesc array (output of
    // slzCompactHuffDescsKernel). `d_compact_raw` holds the interleaved
    // hi/lo RawOff16Descs (output of slzCompactRawDescsKernel).
    // `d_compact_counts` is a 6 × u32 region: [n_lit, n_tok, n_hi, n_lo,
    // n_raw, n_merged].
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

    // Single-slot launch-plumbing scratch: device-resident 4 B counters
    // staged via H2D so the kernel can self-gate. `d_n_chunks_scratch`
    // feeds the scan kernel; `d_n_groups_scratch` feeds the LZ pipeline
    // dispatch with the per-group chunk count.
    d_n_chunks_scratch: CUdeviceptr = 0,
    d_n_chunks_scratch_size: usize = 0,
    d_n_groups_scratch: CUdeviceptr = 0,
    d_n_groups_scratch_size: usize = 0,

    // Huffman descriptors + LUT.
    d_huff_descs: CUdeviceptr = 0,
    d_huff_descs_size: usize = 0,
    d_huff_lut: CUdeviceptr = 0,
    d_huff_lut_size: usize = 0,

    // Library-owned CUstream used as `heavy_stream` whenever the caller
    // didn't provide one (sync wrapper). Created once at init and
    // destroyed in deinit.
    pipeline_stream: usize = 0,
    pipeline_stream_created: bool = false,



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

    /// Free every owned device + host buffer and reset every field to its
    /// default. Intended for a per-handle library API teardown; the
    /// long-lived `driver.g_default` singleton in the current CLI / C ABI
    /// never calls this (its lifetime is the process).
    pub fn deinit(self: *DecodeContext) void {
        const free_dev = struct {
            fn f(ptr: *CUdeviceptr, sz: *usize) void {
                if (ptr.* != 0) {
                    if (cuda.cuMemFree_fn) |free_fn| _ = free_fn(ptr.*);
                    ptr.* = 0;
                }
                sz.* = 0;
            }
        }.f;
        free_dev(&self.d_output, &self.d_output_size);
        free_dev(&self.d_comp_persist, &self.d_comp_persist_size);
        free_dev(&self.d_descs_persist, &self.d_descs_persist_size);
        free_dev(&self.d_entropy_scratch, &self.d_entropy_scratch_size);
        free_dev(&self.d_entropy_off16_scratch, &self.d_entropy_off16_scratch_size);
        free_dev(&self.d_first_subchunk_idx, &self.d_first_subchunk_idx_size);
        free_dev(&self.d_scan_staged, &self.d_scan_staged_size);
        free_dev(&self.d_scan_first_sub, &self.d_scan_first_sub_size);
        free_dev(&self.d_walk_chunks, &self.d_walk_chunks_size);
        free_dev(&self.d_walk_meta, &self.d_walk_meta_size);
        free_dev(&self.d_first_sub_idx_persist, &self.d_first_sub_idx_persist_size);
        free_dev(&self.d_total_subchunks_buf, &self.d_total_subchunks_buf_size);
        free_dev(&self.d_compact_lit, &self.d_compact_lit_size);
        free_dev(&self.d_compact_tok, &self.d_compact_tok_size);
        free_dev(&self.d_compact_hi, &self.d_compact_hi_size);
        free_dev(&self.d_compact_lo, &self.d_compact_lo_size);
        free_dev(&self.d_compact_raw, &self.d_compact_raw_size);
        free_dev(&self.d_compact_counts, &self.d_compact_counts_size);
        free_dev(&self.d_n_chunks_scratch, &self.d_n_chunks_scratch_size);
        free_dev(&self.d_n_groups_scratch, &self.d_n_groups_scratch_size);
        free_dev(&self.d_huff_descs, &self.d_huff_descs_size);
        free_dev(&self.d_huff_lut, &self.d_huff_lut_size);

        // Pinned host output (cuMemAllocHost / cuMemFreeHost).
        if (self.h_pinned_output) |p| {
            if (cuda.cuMemFreeHost_fn) |free_fn| _ = free_fn(@ptrCast(p));
            self.h_pinned_output = null;
        }
        self.h_pinned_output_size = 0;

        // Persistent pipeline stream created in `ensurePipelineStreams`.
        if (self.pipeline_stream_created) {
            if (cuda.cuStreamDestroy_fn) |destroy_fn| {
                if (self.pipeline_stream != 0) _ = destroy_fn(self.pipeline_stream);
            }
            self.pipeline_stream = 0;
            self.pipeline_stream_created = false;
        }

        self.pending_timings.deinit(std.heap.page_allocator);
        self.last_timings.deinit(std.heap.page_allocator);
    }
};

// `g_default` lives in driver.zig (the facade) - see the telemetry comment
// at the top of this file for why storage of `pub var`s must be there.
