//! Per-decode-operation mutable state plus the helpers that pull bytes
//! across the H/D boundary and the kernel-profiling event plumbing.
//!
//! The library exposes one `g_default` context for module-level callers;
//! per-handle library users (streamlz_gpu.zig) instantiate their own
//! `DecodeContext` alongside the matching encode context so a future
//! multi-stream API does not have to thread mutable globals.
//!
//! VK PORT NOTE: ports src/decode/decode_context.zig. The struct shape,
//! field names, helper names, and per-field semantics stay verbatim;
//! the only adaptations are:
//!   * `CUdeviceptr` → `VkDeviceBuffer` (both u64 opaque handles).
//!   * `cuda.cuMemFree_fn` / `cuMemAlloc_fn` → `procs.free_device_fn` /
//!     `malloc_device_fn` through src_vk/decode/vulkan_api.zig.
//!   * `cuda.cuMemAllocHost_fn` → `procs.malloc_host_fn`.
//!   * `cuda.cuMemcpy*` → the matching procs.* shim.
//!   * `cuda.cuStreamCreate_fn` / `cuStreamDestroy_fn` → `procs.stream_create_fn`
//!     / `procs.stream_destroy_fn`. `work_stream` keeps its name and `usize`
//!     shape so codec call sites read identical across the two ports.
//!
//! Timing helpers (`beginKernelTiming`, `endKernelTiming`,
//! `finalizeProfiling`) keep their CUDA-shaped surface — callers pass a
//! `PendingTiming` ArrayList and read back `KernelTiming` slots. The
//! foundation wave leaves the bodies stubbed against the VkQueryPool
//! infrastructure that lands in a subsequent wave; today they return
//! null / no-op so L1 decode (which does not enable profiling) is
//! unaffected.

const std = @import("std");

const vulkan = @import("vulkan_api.zig");
const descriptors = @import("descriptors.zig");

const VkDeviceBuffer = vulkan.VkDeviceBuffer;
const VkStream = vulkan.VkStream;
const VK_SUCCESS_RC = vulkan.VK_SUCCESS_RC;
const procs = vulkan.procs;

// The four `last_*_ns` telemetry vars and `g_default` live in driver.zig
// (the facade) because Zig cannot expose a `pub var` from another module
// through a `pub const` alias — external callers reading
// `gpu_decode.last_kernel_ns` must reach actual storage in the facade.
// Sub-modules write back via `@import("driver.zig").last_kernel_ns = ...`.

/// Grow `*ptr` to hold at least `needed` bytes. No-op when the
/// current allocation is already large enough. Used everywhere the
/// dispatch needs scratch sized to a per-frame quantity; the
/// `GpuError!void` shape matches every other GPU call in the
/// pipeline so the caller's `try`-chain is uniform.
///
/// The free's result is intentionally dropped: at this point we are
/// about to grow the buffer regardless, and a failed free leaks at
/// worst a single allocation (cleaned up at process exit / context
/// destroy). Surfacing the failure here would force every caller to
/// distinguish "alloc failed" from "free failed before alloc", which
/// adds no actionable signal.
pub fn ensureDeviceBuf(ptr: *VkDeviceBuffer, current_size: *usize, needed: usize) descriptors.GpuError!void {
    if (current_size.* >= needed) return;
    const free_fn = procs.free_device_fn orelse return error.BackendNotAvailable;
    const alloc_fn = procs.malloc_device_fn orelse return error.BackendNotAvailable;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    current_size.* = 0;
    if (alloc_fn(ptr, needed) != VK_SUCCESS_RC) return error.OutOfDeviceMemory;
    current_size.* = needed;
}

/// Convenience: grow `self.d_output` to `size`. Same shape as
/// `ensureDeviceBuf` — `try`-friendly from the dispatch.
pub fn ensureDeviceOutput(self: *DecodeContext, size: usize) descriptors.GpuError!void {
    return ensureDeviceBuf(&self.d_output, &self.d_output_size, size);
}

/// Page-locked (pinned) host allocation. D2H/H2D against pinned memory
/// runs at full PCIe bandwidth (~2× pageable, which the driver stages
/// chunk-wise through an internal pinned buffer) and is genuinely
/// async-capable. Returns null when Vulkan is unavailable — caller falls
/// back to a normal allocation, so this is always safe to attempt.
///
/// VK PORT NOTE: CUDA's cuMemAllocHost maps onto a VMA-backed
/// HOST_VISIBLE+HOST_COHERENT VkBuffer with the mapped pointer surfaced
/// directly. The `procs.malloc_host_fn` shim hides the VMA call so the
/// caller stays CUDA-shaped (returns `?[]u8`).
pub fn allocHost(size: usize) ?[]u8 {
    if (!@import("module_loader.zig").init()) return null;
    const f = procs.malloc_host_fn orelse return null;
    var p: ?*anyopaque = null;
    if (f(&p, size) != VK_SUCCESS_RC) return null;
    const base = p orelse return null;
    return @as([*]u8, @ptrCast(base))[0..size];
}

pub fn freeHost(buf: []u8) void {
    const f = procs.free_host_fn orelse return;
    _ = f(@ptrCast(buf.ptr));
}

/// Copy `dst.len` bytes from a device address into the host slice `dst`.
/// Requires init() to have succeeded. Returns false on any Vulkan
/// failure.
pub fn copyDeviceToHost(dst: []u8, src_device: VkDeviceBuffer) bool {
    if (dst.len == 0) return true;
    const f = procs.d2h_fn orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == VK_SUCCESS_RC;
}

/// Copy the host slice `src` to a device address. Requires init() to
/// have succeeded. Returns false on any Vulkan failure.
pub fn copyHostToDevice(dst_device: VkDeviceBuffer, src: []const u8) bool {
    if (src.len == 0) return true;
    const f = procs.h2d_fn orelse return false;
    return f(dst_device, @ptrCast(src.ptr), src.len) == VK_SUCCESS_RC;
}

/// Make the library's Vulkan device current on the calling thread.
///
/// VK PORT NOTE: CUDA's driver API exposes per-thread current-context
/// state — every thread that issues GPU work must call
/// `cuCtxSetCurrent` first. Vulkan has no equivalent: the VkDevice
/// handle and its VkQueue are thread-shareable as long as the caller
/// externally synchronizes queue submission. We retain the entry point
/// (and the `bool` return shape) so caller code reads identical across
/// the two ports; on the VK side it just verifies the bring-up
/// succeeded.
pub fn bindContextToCallingThread() bool {
    return vulkan.init_state == .ready;
}

/// Allocate + record a start event for a kernel about to launch on
/// `stream`. Returns the end-event handle to pass to `endKernelTiming`,
/// or null when profiling is disabled or any backend call failed (timing
/// is best-effort — never blocks the encode/decode path).
///
/// VK PORT NOTE: CUDA backs this with a cuEvent pair; the Vulkan port
/// will back it with a pair of VkQueryPool timestamps. The PendingTiming
/// shape exposes two `usize` slots so the device-side query indices
/// fit without changing the host-visible struct. The foundation wave
/// returns `null` from every entry point because no timestamps are
/// recorded — L1 decode (which never sets `enable_profiling`) is
/// unaffected; a subsequent wave wires the VkQueryPool implementation.
pub fn beginKernelTiming(
    ctx_enabled: bool,
    pending: *std.ArrayListUnmanaged(descriptors.PendingTiming),
    name: [*:0]const u8,
    stream: VkStream,
) ?usize {
    _ = pending;
    _ = name;
    _ = stream;
    if (!ctx_enabled) return null;
    // VK PORT NOTE: VkQueryPool timestamp pairs come online in a later
    // wave. Today we surface null so the codec records no timing — the
    // C ABI's `slzGetLastTimings_vk` returns 0 entries until that wave
    // lands. Mirrors the CUDA fast-path when `cuEventCreate_fn` is null.
    return null;
}

/// Record the end event of a kernel-timing pair. The pair was appended
/// to `pending_timings` by `beginKernelTiming`; this is the matching
/// second half.
///
/// VK PORT NOTE: same VkQueryPool-pending caveat as `beginKernelTiming`.
/// Foundation wave is a no-op.
pub fn endKernelTiming(end_event: ?usize, stream: VkStream) void {
    _ = end_event;
    _ = stream;
}

/// Synchronize on each pending event pair and compute elapsed times
/// into `last_timings`. Destroys the events. Idempotent (safe to call
/// when pending is empty).
///
/// VK PORT NOTE: Foundation-wave bodies leave `last_timings` untouched
/// when `pending` is empty (matches CUDA's guard so prior calls'
/// readings survive). The VkQueryPool sweep lands in a later wave.
pub fn finalizeProfiling(
    pending: *std.ArrayListUnmanaged(descriptors.PendingTiming),
    last_timings: *std.ArrayListUnmanaged(descriptors.KernelTiming),
) void {
    _ = last_timings;
    if (pending.items.len == 0) return;
    pending.clearRetainingCapacity();
}

/// Per-decode-operation mutable state. Every device-buffer pointer,
/// buffer size, host scratch array, and per-operation flag formerly held
/// as a module-global lives here so a future library API can hand each
/// handle its own context. Load-once module handles, kernel/driver
/// function pointers, and `pub var` telemetry stay module-global on
/// purpose.
///
/// VK PORT NOTE: ports CUDA's `DecodeContext` field-for-field. Every
/// `CUdeviceptr` slot becomes `VkDeviceBuffer` (also u64); field NAMES
/// stay verbatim so the dispatch can call `d_walk_meta`, `d_huff_lut`,
/// etc. across both ports without renames. `work_stream` keeps its
/// `usize` shape (VkStream is an alias for usize today; the pure-D2D
/// path will plumb it through to a per-handle VkQueue).
pub const DecodeContext = struct {
    // ── Output buffer ──────────────────────────────────────────
    d_output: VkDeviceBuffer = 0,
    d_output_size: usize = 0,
    // Pinned host mirror of d_output for the D2H copy on the synchronous
    // decompress path. Grouped here so output-related state lives
    // together.
    h_pinned_output: ?[*]u8 = null,
    h_pinned_output_size: usize = 0,

    // ── Persistent compressed-input + descriptor buffers ───────
    d_comp_persist: VkDeviceBuffer = 0,
    d_comp_persist_size: usize = 0,
    d_descs_persist: VkDeviceBuffer = 0,
    d_descs_persist_size: usize = 0,

    // ── Entropy-decoded scratch ────────────────────────────────
    // Holds the Huffman pre-decode output that the LZ kernel reads.
    // Layout per sub-chunk slot: [lit][tok at +tok_offset][off16 at +off16_offset].
    // off16 sub-region: hi bytes at chunk_idx*descriptors.ENTROPY_SCRATCH_SLOT_BYTES,
    // lo bytes at +descriptors.OFF16_HILO_SPLIT_OFFSET.
    d_entropy_scratch: VkDeviceBuffer = 0,
    d_entropy_scratch_size: usize = 0,

    // Off16 scratch VIEW (not owned): set in fullGpuLaunchImpl to
    // `d_entropy_scratch + off16_offset`. The raw-off16 gather kernel
    // scatters compressed raw bytes here. Not freed by deinit because
    // `d_entropy_scratch` owns the allocation.
    //
    // VK PORT NOTE: CUDA's CUdeviceptr is a real VA, so `base + offset`
    // works directly. Vulkan's VkBuffer is an opaque handle; the
    // "view" semantics translate to a (buffer, offset) pair the kernel
    // dispatches through. The dispatch wave stores the buffer handle
    // here and the offset elsewhere (mirrors the CUDA call site
    // verbatim — the buffer slot keeps its name).
    d_entropy_off16_scratch: VkDeviceBuffer = 0,

    // Per-chunk first-subchunk-index ALIAS (not owned): the dispatch
    // sets this to `d_first_sub_idx_persist` so the LZ kernel reads the
    // device-resident prefix sum directly. Not freed by deinit because
    // `d_first_sub_idx_persist` owns the allocation.
    d_first_subchunk_idx: VkDeviceBuffer = 0,

    // GPU decode-scan staged buffers. d_scan_staged packs the six staged
    // arrays the scan kernel writes (lit/tok/hi/lo huff descriptor lists
    // + raw hi/lo gather lists); d_scan_first_sub holds the per-chunk
    // first-sub-chunk prefix sum.
    d_scan_staged: VkDeviceBuffer = 0,
    d_scan_staged_size: usize = 0,
    d_scan_first_sub: VkDeviceBuffer = 0,
    d_scan_first_sub_size: usize = 0,

    // GPU frame-walk kernel scratch. d_walk_chunks holds the
    // kernel-produced ChunkDesc array; d_walk_meta is a 6-u32 region
    // (n_chunks, decomp_size, sub_chunk_cap, block_start, block_size,
    // status).
    d_walk_chunks: VkDeviceBuffer = 0,
    d_walk_chunks_size: usize = 0,
    d_walk_meta: VkDeviceBuffer = 0,
    d_walk_meta_size: usize = 0,

    // Pure-D2D prefix-sum scratch: d_first_sub_idx holds the per-chunk
    // first-sub-chunk index; d_total_subchunks_buf is a single u32 with
    // the running total (device-resident, never D2H'd on the pure path).
    d_first_sub_idx_persist: VkDeviceBuffer = 0,
    d_first_sub_idx_persist_size: usize = 0,
    d_total_subchunks_buf: VkDeviceBuffer = 0,
    d_total_subchunks_buf_size: usize = 0,

    // Pure-D2D compaction outputs. Each `d_compact_*` buffer holds a
    // compacted per-stream HuffDecChunkDesc array (output of
    // slzCompactHuffDescsKernel). `d_compact_raw` holds the interleaved
    // hi/lo RawOff16Descs (output of slzCompactRawDescsKernel).
    // `d_compact_counts` is a 6 × u32 region: [n_lit, n_tok, n_hi, n_lo,
    // n_raw, n_merged].
    d_compact_lit: VkDeviceBuffer = 0,
    d_compact_lit_size: usize = 0,
    d_compact_tok: VkDeviceBuffer = 0,
    d_compact_tok_size: usize = 0,
    d_compact_hi: VkDeviceBuffer = 0,
    d_compact_hi_size: usize = 0,
    d_compact_lo: VkDeviceBuffer = 0,
    d_compact_lo_size: usize = 0,
    d_compact_raw: VkDeviceBuffer = 0,
    d_compact_raw_size: usize = 0,
    d_compact_counts: VkDeviceBuffer = 0,
    d_compact_counts_size: usize = 0,

    // Single-slot launch-plumbing scratch: device-resident 4 B counters
    // staged via H2D so the kernel can self-gate. `d_n_chunks_scratch`
    // feeds the scan kernel; `d_n_groups_scratch` feeds the LZ pipeline
    // dispatch with the per-group chunk count.
    d_n_chunks_scratch: VkDeviceBuffer = 0,
    d_n_chunks_scratch_size: usize = 0,
    d_n_groups_scratch: VkDeviceBuffer = 0,
    d_n_groups_scratch_size: usize = 0,

    // Huffman descriptors + LUT.
    d_huff_descs: VkDeviceBuffer = 0,
    d_huff_descs_size: usize = 0,
    d_huff_lut: VkDeviceBuffer = 0,
    d_huff_lut_size: usize = 0,

    // Library-owned stream used as `heavy_stream` whenever the caller
    // didn't provide one (sync wrapper). Created once at init and
    // destroyed in deinit.
    //
    // VK PORT NOTE: CUDA's CUstream is a per-context FIFO of work.
    // Vulkan's equivalent is a VkQueue (or, for fine-grained async, a
    // VkCommandBuffer pool keyed off a queue). The slot keeps its
    // `usize` shape so the dispatch can pass it through to procs.* —
    // the VK runtime knows how to decode the handle into the real
    // Vulkan object.
    pipeline_stream: VkStream = 0,
    pipeline_stream_created: bool = false,

    // Per-kernel timing (slzDecompressOpts_t.enable_profiling). When
    // true, every kernel launch in fullGpuLaunchImpl records a
    // VkQueryPool timestamp pair and appends to `pending_timings`.
    // `finalizeProfiling` (called after the final sync) drains pending
    // → `last_timings`, which slzGetLastTimings reads out via the C ABI.
    enable_profiling: bool = false,
    pending_timings: std.ArrayListUnmanaged(descriptors.PendingTiming) = .empty,
    last_timings: std.ArrayListUnmanaged(descriptors.KernelTiming) = .empty,

    // Stream used for the heavy-phase kernel launches (huff build /
    // decode, LZ decode) and the final D2D output copy.
    // slzDecompressAsync sets it to the caller's stream so a stream
    // sync on that stream waits for the decompress to complete. The
    // sync slzDecompress wrapper leaves it at 0 (default stream) and
    // waits with a context-wide sync.
    work_stream: VkStream = 0,

    /// Free every owned device + host buffer and reset every field to
    /// its default. Intended for a per-handle library API teardown;
    /// the long-lived `driver.g_default` singleton in the current CLI
    /// / C ABI never calls this (its lifetime is the process).
    pub fn deinit(self: *DecodeContext) void {
        const free_dev = struct {
            fn f(ptr: *VkDeviceBuffer, sz: *usize) void {
                if (ptr.* != 0) {
                    if (procs.free_device_fn) |free_fn| _ = free_fn(ptr.*);
                    ptr.* = 0;
                }
                sz.* = 0;
            }
        }.f;
        free_dev(&self.d_output, &self.d_output_size);
        free_dev(&self.d_comp_persist, &self.d_comp_persist_size);
        free_dev(&self.d_descs_persist, &self.d_descs_persist_size);
        free_dev(&self.d_entropy_scratch, &self.d_entropy_scratch_size);
        // `d_entropy_off16_scratch` and `d_first_subchunk_idx` are views
        // / aliases into other owned buffers (see field docs). Zero them
        // so the alias does not survive teardown, but do NOT free.
        self.d_entropy_off16_scratch = 0;
        self.d_first_subchunk_idx = 0;
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

        // Pinned host output (procs.malloc_host / procs.free_host).
        if (self.h_pinned_output) |p| {
            if (procs.free_host_fn) |free_fn| _ = free_fn(@ptrCast(p));
            self.h_pinned_output = null;
        }
        self.h_pinned_output_size = 0;

        // Persistent pipeline stream created in `ensurePipelineStream`.
        if (self.pipeline_stream_created) {
            if (procs.stream_destroy_fn) |destroy_fn| {
                if (self.pipeline_stream != 0) _ = destroy_fn(self.pipeline_stream);
            }
            self.pipeline_stream = 0;
            self.pipeline_stream_created = false;
        }

        self.pending_timings.deinit(std.heap.page_allocator);
        self.last_timings.deinit(std.heap.page_allocator);
    }
};

// `g_default` lives in driver.zig (the facade) — see the telemetry comment
// at the top of this file for why storage of `pub var`s must be there.

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "DecodeContext default-initializes every device slot to 0" {
    // Codec deinit paths rely on the `if (ptr.* != 0)` guard, which
    // means every slot MUST start at 0. Mirrors the CUDA contract.
    const d: DecodeContext = .{};
    try testing.expectEqual(@as(VkDeviceBuffer, 0), d.d_output);
    try testing.expectEqual(@as(VkDeviceBuffer, 0), d.d_walk_meta);
    try testing.expectEqual(@as(VkDeviceBuffer, 0), d.d_huff_lut);
    try testing.expectEqual(@as(usize, 0), d.d_output_size);
    try testing.expect(!d.pipeline_stream_created);
}

test "ensureDeviceBuf surfaces BackendNotAvailable when procs are unset" {
    // Foundation wave: procs.* slots are null until the runtime is
    // brought up. The error surface must be `BackendNotAvailable` so
    // the codec's existing `try`-chain plumbs it through unchanged.
    var ptr: VkDeviceBuffer = 0;
    var size: usize = 0;
    try testing.expectError(error.BackendNotAvailable, ensureDeviceBuf(&ptr, &size, 64));
}
