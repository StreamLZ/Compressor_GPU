//! 1:1 port of src/decode/decode_context.zig.
//!
//! Per-context state for the GPU decode pipeline. Owns every device
//! buffer the pipeline consumes (compressed input, descriptor lists,
//! entropy scratch, walk/scan/compact/merge outputs, the LUT, and the
//! decode output), the pinned host output mirror, the persistent VK
//! pipeline stream, and the per-kernel timing slots.
//!
//! Every CUdeviceptr slot from the CUDA original becomes a VkDeviceBuffer
//! per Section B; every device op funnels through the
//! `vulkan_api.procs.*` surface (no direct vk*/vma calls in this file).

const std = @import("std");
const vk = @import("vulkan_api.zig");
const descriptors = @import("descriptors.zig");

const VkDeviceBuffer = vk.VkDeviceBuffer;
const VK_SUCCESS_RC = vk.VK_SUCCESS_RC;

/// CUDA reference: src/decode/decode_context.zig:35-43. Grow `*ptr` to hold
/// at least `needed` bytes. No-op when the current allocation already fits
/// (grow-only — never frees on size match). All device ops route through
/// `vk.procs.*` so call sites stay CUDA-shaped.
pub fn ensureDeviceBuf(ptr: *VkDeviceBuffer, current_size: *usize, needed: usize) descriptors.GpuError!void {
    if (current_size.* >= needed) return;
    const free_fn = vk.procs.free_device orelse return error.BackendNotAvailable;
    const alloc_fn = vk.procs.malloc_device orelse return error.BackendNotAvailable;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    current_size.* = 0;
    if (alloc_fn(ptr, needed) != VK_SUCCESS_RC) return error.OutOfDeviceMemory;
    current_size.* = needed;
}

/// CUDA reference: src/decode/decode_context.zig:47-49. Grow `self.d_output`
/// to at least `size` bytes.
pub fn ensureDeviceOutput(self: *DecodeContext, size: usize) descriptors.GpuError!void {
    return ensureDeviceBuf(&self.d_output, &self.d_output_size, size);
}

/// CUDA reference: src/decode/decode_context.zig:61-71 (v4 #16).
/// Register a custom dictionary on this context. The bytes are copied
/// (context-owned; freed by deinit via the captured allocator). Returns
/// the content-derived ID - the value frames carry. Registering the
/// same content again is a no-op returning the same ID.
pub fn registerDict(self: *DecodeContext, allocator: std.mem.Allocator, data: []const u8) !u32 {
    const dictionary = @import("../dict/dictionary.zig");
    const id = dictionary.customId(data);
    for (self.registered_dicts.items) |r| {
        if (r.id == id) return id;
    }
    const copy = try allocator.dupe(u8, data);
    errdefer allocator.free(copy);
    try self.registered_dicts.append(allocator, .{ .id = id, .data = copy });
    self.dict_store_alloc = allocator;
    return id;
}

/// CUDA reference: src/decode/decode_context.zig:73-83 (v4 #16). Make
/// the preset dictionary `data` (identified by `id`) resident in the
/// context's `d_dict` buffer. The upload runs only when the cached ID
/// changes - the batch use case decodes thousands of frames against one
/// dictionary with a single upload.
pub fn ensureDictOnDevice(self: *DecodeContext, id: u32, data: []const u8) descriptors.GpuError!void {
    if (self.dict_cached_id == id and self.dict_cached_len == data.len) return;
    // Callers may reach this before any dispatch initialized the
    // driver (the upload is lazy - see decompressFrameInner).
    if (!@import("module_loader.zig").init()) return error.BackendNotAvailable;
    self.dict_cached_id = 0;
    try ensureDeviceBuf(&self.d_dict, &self.d_dict_size, data.len);
    if (!copyHostToDevice(self.d_dict, data)) return error.CopyFailed;
    self.dict_cached_id = id;
    self.dict_cached_len = @intCast(data.len);
}

/// CUDA reference: src/decode/decode_context.zig:56-63. Page-locked
/// (host-visible / host-coherent) host allocation through
/// `procs.malloc_host`. Returns null when the backend is unavailable so
/// callers fall back to a normal allocation.
pub fn allocHost(size: usize) ?[]u8 {
    if (!@import("module_loader.zig").init()) return null;
    const f = vk.procs.malloc_host orelse return null;
    var p: ?*anyopaque = null;
    if (f(&p, size) != VK_SUCCESS_RC) return null;
    const base = p orelse return null;
    return @as([*]u8, @ptrCast(base))[0..size];
}

/// CUDA reference: src/decode/decode_context.zig:65-68. Free a buffer
/// previously returned by `allocHost` via `procs.free_host`.
pub fn freeHost(buf: []u8) void {
    const f = vk.procs.free_host orelse return;
    _ = f(@ptrCast(buf.ptr));
}

/// CUDA reference: src/decode/decode_context.zig:72-76. Copy `dst.len`
/// bytes from `src_device` into the host slice `dst` via `procs.d2h`.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    if (dst.len == 0) return true;
    const f = vk.procs.d2h orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == VK_SUCCESS_RC;
}

/// CUDA reference: src/decode/decode_context.zig:80-84. Copy the host
/// slice `src` to a device buffer via `procs.h2d`.
pub fn copyHostToDevice(dst_device: u64, src: []const u8) bool {
    if (src.len == 0) return true;
    const f = vk.procs.h2d orelse return false;
    return f(dst_device, @ptrCast(src.ptr), src.len) == VK_SUCCESS_RC;
}

/// CUDA reference: src/decode/decode_context.zig:90-94. Bind the active
/// VK device to the calling thread through `procs.ctx_set_current`.
pub fn bindContextToCallingThread() bool {
    if (vk.ctx == 0) return false;
    const f = vk.procs.ctx_set_current orelse return false;
    return f(vk.ctx) == VK_SUCCESS_RC;
}

/// CUDA reference: src/decode/decode_context.zig:100-127. Begin a
/// per-kernel timing window when profiling is enabled. Returns the
/// end-event handle to pass to `endKernelTiming`, or null when profiling
/// is disabled or any procs.event_* call failed (timing is best-effort).
pub fn beginKernelTiming(
    ctx_enabled: bool,
    pending: *std.ArrayListUnmanaged(descriptors.PendingTiming),
    name: [*:0]const u8,
    stream: usize,
) ?usize {
    if (!ctx_enabled) return null;
    const create_fn = vk.procs.event_create orelse return null;
    const record_fn = vk.procs.event_record orelse return null;
    var start: vk.VkEvent = 0;
    var end: vk.VkEvent = 0;
    if (create_fn(&start, 0) != VK_SUCCESS_RC) return null;
    if (create_fn(&end, 0) != VK_SUCCESS_RC) {
        if (vk.procs.event_destroy) |destroy_fn| _ = destroy_fn(start);
        return null;
    }
    if (record_fn(start, stream) != VK_SUCCESS_RC) {
        if (vk.procs.event_destroy) |destroy_fn| {
            _ = destroy_fn(start);
            _ = destroy_fn(end);
        }
        return null;
    }
    pending.append(std.heap.page_allocator, .{
        .name = name,
        .start_event = @intCast(start),
        .end_event = @intCast(end),
    }) catch {
        if (vk.procs.event_destroy) |destroy_fn| {
            _ = destroy_fn(start);
            _ = destroy_fn(end);
        }
        return null;
    };
    return @intCast(end);
}

/// CUDA reference: src/decode/decode_context.zig:142-146. Close a
/// per-kernel timing window opened by `beginKernelTiming`.
pub fn endKernelTiming(end_event: ?usize, stream: usize) void {
    const e = end_event orelse return;
    const record_fn = vk.procs.event_record orelse return;
    _ = record_fn(@intCast(e), stream);
}

/// CUDA reference: src/decode/decode_context.zig:151-186. Drain pending
/// timestamp pairs into `last_timings` after a final sync. Idempotent
/// when `pending` is empty.
pub fn finalizeProfiling(
    pending: *std.ArrayListUnmanaged(descriptors.PendingTiming),
    last_timings: *std.ArrayListUnmanaged(descriptors.KernelTiming),
) void {
    if (pending.items.len == 0) return;
    last_timings.clearRetainingCapacity();
    const sync_fn = vk.procs.event_synchronize orelse {
        pending.clearRetainingCapacity();
        return;
    };
    const elapsed_fn = vk.procs.event_elapsed_time orelse {
        pending.clearRetainingCapacity();
        return;
    };
    const destroy_fn = vk.procs.event_destroy orelse {
        pending.clearRetainingCapacity();
        return;
    };
    for (pending.items) |p| {
        _ = sync_fn(@intCast(p.end_event));
        var ms: f32 = 0.0;
        _ = elapsed_fn(&ms, @intCast(p.start_event), @intCast(p.end_event));
        last_timings.append(std.heap.page_allocator, .{ .name = p.name, .ms = ms }) catch {};
        _ = destroy_fn(@intCast(p.start_event));
        _ = destroy_fn(@intCast(p.end_event));
    }
    pending.clearRetainingCapacity();
}

/// CUDA reference: src/decode/decode_context.zig:193-374. Per-decode
/// mutable state. Every `d_*` field is a VkDeviceBuffer (CUDA:
/// CUdeviceptr) — same null sentinel (0) and same grow-only `*_size`
/// twin slot pattern.
pub const DecodeContext = struct {
    // ── Output buffer ──────────────────────────────────────────
    d_output: VkDeviceBuffer = 0,
    d_output_size: usize = 0,
    // Pinned host mirror of d_output for the D2H copy on the synchronous
    // decompress path.
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
    d_entropy_scratch: VkDeviceBuffer = 0,
    d_entropy_scratch_size: usize = 0,

    // Off16 scratch VIEW (not owned): set in fullGpuLaunchImpl to
    // Iter 4c: pre-iter-4c this was `d_entropy_scratch + off16_offset`
    // — broken handle arithmetic (the codec routed the arithmetic as a
    // u64 device-address, but VkDeviceBuffer is a registry index). Now
    // the alias is the BASE handle d_entropy_scratch and the per-bind
    // byte offset travels through procLaunchKernel.binding_offsets in
    // d_entropy_off16_offset below. Not freed by deinit because
    // `d_entropy_scratch` owns the allocation.
    d_entropy_off16_scratch: VkDeviceBuffer = 0,
    d_entropy_off16_offset: u64 = 0,

    // Per-chunk first-subchunk-index ALIAS (not owned): the dispatch
    // sets this to `d_first_sub_idx_persist` so the LZ kernel reads the
    // device-resident prefix sum directly. Not freed by deinit because
    // `d_first_sub_idx_persist` owns the allocation.
    d_first_subchunk_idx: VkDeviceBuffer = 0,

    // GPU decode-scan staged buffers.
    d_scan_staged: VkDeviceBuffer = 0,
    d_scan_staged_size: usize = 0,
    d_scan_first_sub: VkDeviceBuffer = 0,
    d_scan_first_sub_size: usize = 0,

    // GPU frame-walk kernel scratch.
    d_walk_chunks: VkDeviceBuffer = 0,
    d_walk_chunks_size: usize = 0,
    d_walk_meta: VkDeviceBuffer = 0,
    d_walk_meta_size: usize = 0,

    // Pure-D2D prefix-sum scratch.
    d_first_sub_idx_persist: VkDeviceBuffer = 0,
    d_first_sub_idx_persist_size: usize = 0,
    d_total_subchunks_buf: VkDeviceBuffer = 0,
    d_total_subchunks_buf_size: usize = 0,

    // A-024: per-region byte offsets into d_entropy_scratch (BOUND-based,
    // matching the layout uploadInputAndPrefixSum sets up). Stashed for
    // the Huff decode kernel push constants so the kernel can apply
    // region_off as u32 — the merge kernel used to fold them into
    // desc.out_offset but that u32 add overflowed at ~6553 sub-chunks.
    last_tok_offset: u64 = 0,
    last_off16_offset: u64 = 0,

    // Pure-D2D compaction outputs.
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

    // Single-slot launch-plumbing scratch.
    d_n_chunks_scratch: VkDeviceBuffer = 0,
    d_n_chunks_scratch_size: usize = 0,
    d_n_groups_scratch: VkDeviceBuffer = 0,
    d_n_groups_scratch_size: usize = 0,

    // Huffman descriptors + LUT.
    d_huff_descs: VkDeviceBuffer = 0,
    d_huff_descs_size: usize = 0,
    d_huff_lut: VkDeviceBuffer = 0,
    d_huff_lut_size: usize = 0,

    // Library-owned VkStream used as `heavy_stream` whenever the caller
    // did not provide one. Created once at init via `procs.stream_create`
    // and destroyed in deinit via `procs.stream_destroy`.
    pipeline_stream: usize = 0,
    pipeline_stream_created: bool = false,

    // Per-kernel timing (slzDecompressOpts_t.enable_profiling).
    enable_profiling: bool = false,
    pending_timings: std.ArrayListUnmanaged(descriptors.PendingTiming) = .empty,
    last_timings: std.ArrayListUnmanaged(descriptors.KernelTiming) = .empty,

    // VkStream used for the heavy-phase kernel launches. slzDecompressAsync
    // sets it to the caller's stream so procs.stream_sync on that stream
    // waits for the decompress to complete. The sync slzDecompress
    // wrapper leaves it at 0 (default stream).
    work_stream: usize = 0,

    // ── v4 #16: preset dictionary (CUDA reference:
    // src/decode/decode_context.zig:299-311) ────────────────────
    // Device-resident copy of the frame's dictionary, cached by ID and
    // reused across calls. `dict_cached_id` keys the cache (0 = empty).
    d_dict: VkDeviceBuffer = 0,
    d_dict_size: usize = 0,
    dict_cached_id: u32 = 0,
    dict_cached_len: u32 = 0,
    // Caller-registered custom dictionaries (slzSetDictionary / CLI -D
    // <file>). Context-owned copies; freed in deinit.
    dict_store_alloc: ?std.mem.Allocator = null,
    registered_dicts: std.ArrayList(@import("../dict/dictionary.zig").RegisteredDict) = .empty,

    /// CUDA reference: src/decode/decode_context.zig:320-374. Free every
    /// owned device + host buffer and reset every field to its default.
    pub fn deinit(self: *DecodeContext) void {
        const free_dev = struct {
            fn f(ptr: *VkDeviceBuffer, sz: *usize) void {
                if (ptr.* != 0) {
                    if (vk.procs.free_device) |free_fn| _ = free_fn(ptr.*);
                    ptr.* = 0;
                }
                sz.* = 0;
            }
        }.f;
        free_dev(&self.d_output, &self.d_output_size);
        free_dev(&self.d_comp_persist, &self.d_comp_persist_size);
        free_dev(&self.d_descs_persist, &self.d_descs_persist_size);
        free_dev(&self.d_entropy_scratch, &self.d_entropy_scratch_size);
        // `d_entropy_off16_scratch` and `d_first_subchunk_idx` are views /
        // aliases into other owned buffers (see field docs). Zero them so
        // the alias does not survive teardown, but do NOT free.
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
        // v4 #16: dictionary cache + registered custom dictionaries.
        free_dev(&self.d_dict, &self.d_dict_size);
        self.dict_cached_id = 0;
        self.dict_cached_len = 0;
        if (self.dict_store_alloc) |alloc| {
            for (self.registered_dicts.items) |r| alloc.free(r.data);
            self.registered_dicts.deinit(alloc);
            self.registered_dicts = .empty;
            self.dict_store_alloc = null;
        }

        // Pinned host output (procs.malloc_host / procs.free_host).
        if (self.h_pinned_output) |p| {
            if (vk.procs.free_host) |free_fn| _ = free_fn(@ptrCast(p));
            self.h_pinned_output = null;
        }
        self.h_pinned_output_size = 0;

        // Persistent pipeline stream created in `ensurePipelineStream`.
        if (self.pipeline_stream_created) {
            if (vk.procs.stream_destroy) |destroy_fn| {
                if (self.pipeline_stream != 0) _ = destroy_fn(self.pipeline_stream);
            }
            self.pipeline_stream = 0;
            self.pipeline_stream_created = false;
        }

        self.pending_timings.deinit(std.heap.page_allocator);
        self.last_timings.deinit(std.heap.page_allocator);
    }
};
