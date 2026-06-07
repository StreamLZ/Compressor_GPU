//! 1:1 port of src/decode/cuda_api.zig (RENAMED per Section B).
//!
//! Vulkan Driver-equivalent API + Win32 surface used by the GPU-decode
//! pipeline. Holds the VkResult / VkDeviceBuffer aliases, the `procs`
//! function-pointer slots, the dlopen state for vulkan-1.dll, and the
//! QPC clock - bundled here so module_loader / decode_context / etc. do
//! not duplicate FFI boilerplate.
//!
//! THE CODEC NEVER CALLS Vulkan ENTRY POINTS DIRECTLY. Every device-side
//! allocation, host-bounce copy, and synchronization point funnels
//! through one of the function-pointer slots on the `procs` struct
//! below, so call sites read identically to the CUDA codec's:
//!
//!     CUDA: rc = cuda_api.cuMemAlloc_fn.?(&dptr, size);
//!     VK  : rc = vulkan_api.procs.malloc_device.?(&buf, size);
//!
//! Module-loader populates the slots after bringing up the VkInstance /
//! VkDevice / VkQueue / VMA allocator; until then the slots are null and
//! every call surfaces as a non-zero rc the codec maps to
//! BackendNotAvailable.
//!
//! Function-pointer slots are `pub var` (not `pub const`) because
//! module_loader.zig fills them at init time and the rest of the
//! pipeline reads them on every launch.

const std = @import("std");

pub const win32 = struct {
    pub extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    pub extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};

/// CUDA reference: src/decode/cuda_api.zig:25-29. Monotonic timestamp
/// from QueryPerformanceCounter. Pair with qpcMs() for elapsed-
/// millisecond timing without an std.Io handle - used by the
/// SLZ_E2E_TIMER instrumentation in init() and the CLI.
pub fn qpcNow() i64 {
    var c: i64 = 0;
    _ = win32.QueryPerformanceCounter(&c);
    return c;
}

// QueryPerformanceFrequency is fixed for the process lifetime - cache
// the first read so qpcMs avoids the syscall on every call. Atomic
// because qpcMs can be called from any thread; the value is idempotent
// so a benign race that re-queries is harmless.
var cached_qpc_freq: std.atomic.Value(i64) = .init(0);

/// CUDA reference: src/decode/cuda_api.zig:35-43. Elapsed milliseconds
/// between two qpcNow samples.
pub fn qpcMs(from: i64, to: i64) f64 {
    var freq = cached_qpc_freq.load(.monotonic);
    if (freq == 0) {
        _ = win32.QueryPerformanceFrequency(&freq);
        if (freq == 0) freq = 1;
        cached_qpc_freq.store(freq, .monotonic);
    }
    return @as(f64, @floatFromInt(to - from)) * 1000.0 / @as(f64, @floatFromInt(freq));
}

// ── Vulkan-side type aliases (replace CU* per Section B) ──────────────

/// CUDA reference: src/decode/cuda_api.zig:45. Underlying driver result
/// code. Mirrors VkResult (c_int) - 0 is success, negatives are errors.
pub const VkResult = c_int;

/// CUDA reference: src/decode/cuda_api.zig:46. Logical device index.
/// CUDA exposes integer device indices; the VK port keeps the same
/// shape so call sites stay uniform.
pub const VkDevice = c_int;

/// CUDA reference: src/decode/cuda_api.zig:47 (CUdeviceptr → VkDeviceBuffer).
/// Opaque device-buffer handle. CUDA gives a real device VA; Vulkan
/// gives a (VkBuffer, VmaAllocation, size, optional BDA) tuple. This
/// u64 is the registry key into the table the VMA-backed module_loader
/// maintains. The codec only ever passes it through procs.* so call
/// sites stay CUDA-shaped (pointer-sized opaque handle, 0 = null /
/// unallocated). Codec deinit paths use `if (buf != 0)` to skip frees
/// for unallocated slots - matches CUDA's null-check shape verbatim.
pub const VkDeviceBuffer = u64;

/// Null sentinel for VkDeviceBuffer. Mirrors CUDA's implicit
/// `CUdeviceptr == 0` null pattern.
pub const VK_NULL_BUFFER: VkDeviceBuffer = 0;

/// CUDA reference: CUstream (an opaque pointer in CUDA). The VK port
/// models a stream as a registry key into a queue + per-queue command
/// buffer + fence triple that module_loader maintains. 0 = the default
/// (per-context shared) stream.
pub const VkStream = usize;

/// CUDA reference: src/decode/cuda_api.zig:48 (CUDA_SUCCESS → VK_SUCCESS_RC).
/// Numeric 0 sentinel preserved so the `rc == 0` fast-path branch reads
/// identical across backends. Renamed from VK_SUCCESS to avoid collision
/// with the vulkan-1.dll bare VK_SUCCESS constant some sibling modules
/// already expose.
pub const VK_SUCCESS_RC: VkResult = 0;

// ── Module state (LoadLibraryA handle + current device + VMA) ─────────
// Mirrors CUDA's `lib` + `ctx` slots but with the extra Vulkan-specific
// bring-up state (instance, physical device, queue family) Vulkan needs
// to thread between dispatch sites.

/// CUDA reference: src/decode/cuda_api.zig:51 (lib). vulkan-1.dll
/// dlopen handle. Populated by module_loader.init via LoadLibraryA.
pub var lib: ?*anyopaque = null;

/// CUDA reference: src/decode/cuda_api.zig:52 (ctx). Current Vulkan
/// device handle (VkDevice cast to usize for shape parity with CUDA's
/// CUcontext slot). Module_loader populates this after vkCreateDevice.
pub var ctx: usize = 0;

/// VK adaptation: Vulkan needs an explicit VkInstance separate from the
/// device; CUDA collapses both into the cuContext. Module_loader writes
/// the handle here after vkCreateInstance.
pub var instance: usize = 0;

/// VK adaptation: the physical device the logical device was created
/// on. Module_loader writes after vkEnumeratePhysicalDevices.
pub var physical_device: usize = 0;

/// VK adaptation: the compute queue handle module_loader pulled from
/// the chosen queue family. procs.* dispatches submits through this.
pub var compute_queue: usize = 0;

/// VK adaptation: queue family index for the compute_queue above.
/// Needed by vkCreateCommandPool calls module_loader stages.
pub var compute_queue_family: u32 = 0;

/// VK adaptation: VMA allocator handle. Module_loader writes after
/// vmaCreateAllocator on the (instance, physical, device) triple. Typed
/// as opaque-pointer-sized usize so this module stays independent of
/// the vma.zig import chain (the encode-side ffi module links to the
/// same VMA TU; they share allocators via this slot).
pub var vma_allocator: usize = 0;

/// CUDA reference: src/decode/cuda_api.zig:58. Module-loader bring-up
/// state. `uninit` is the initial value; `init` advances through
/// `in_progress` while it runs (lets a re-entry detect a loop) and
/// finishes at `ready` on success or `failed` if any step bailed.
pub const InitState = enum { uninit, in_progress, ready, failed };
pub var init_state: InitState = .uninit;

/// CUDA reference: src/decode/cuda_api.zig:101. SM-count analogue for
/// fast_framed.resolveScGroupSize and the adaptive sc_group_size
/// threshold. The CUDA path reads cuda_api.sm_count populated from
/// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT; the VK port populates it
/// from one of the options in audit.md Section H Q2 (queue-family
/// compute-unit count fallback constant on most drivers).
/// 0 until init has run successfully.
pub var sm_count: u32 = 0;

/// CUDA reference: src/decode/cuda_api.zig:104
/// (CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT). The CUDA path queries
/// the driver for the SM count via this attribute id. The VK port has
/// no direct equivalent - module_loader resolves sm_count via the
/// VkPhysicalDeviceProperties query instead. The constant is kept here
/// (matching CUDA's slot) so call sites that mirror cuda_api token by
/// token compile against the same surface.
pub const VK_DEVICE_PROP_MULTIPROCESSOR_COUNT_PLACEHOLDER: c_int = 16;

// ── procs.* shim surface - replaces the cu*_fn slot bank ──────────────
// CUDA exposes one `pub var cu<Op>_fn: ?Fn<Op>` per driver entry point.
// Section B of audit.md folds that bank into a single procs struct so
// the VK port can park VMA / vkCmd* implementations behind the slots
// without exposing Vulkan-flavored names to the codec.
//
// Procs has 23 slots: malloc_device / free_device / h2d / d2h /
// h2d_async / d2h_async / d2d / memset_d8 / memset_d8_async /
// malloc_host / free_host / stream_sync / ctx_sync / ctx_set_current /
// ctx_get_current / stream_create / stream_destroy / event_create /
// event_record / event_synchronize / event_elapsed_time / event_destroy /
// launch_kernel.
//
// Call shape stays CUDA-shaped:
//   CUDA: try cudaCall(cuda_api.cuMemAlloc_fn.?(&dptr, size), .alloc);
//   VK  : try vkCall(vulkan_api.procs.malloc_device.?(&buf, size), .alloc);

// ── Fn typedefs for the four shim slots added in this fix-up ──────────
// These mirror the CUDA Fn aliases at src/decode/cuda_api.zig:78-85 with
// the CU* types swapped for their VkDeviceBuffer / VkStream / VkContext-
// shaped analogues per Section B of the audit.

/// CUDA reference: src/decode/cuda_api.zig:85 (FnCtxSetCurrent).
/// CUDA signature: `*const fn (ctx: CUcontext) callconv(.c) CUresult`.
/// VK port replaces CUcontext with usize (the same shape `vulkan_api.ctx`
/// uses for the VkDevice handle).
pub const FnCtxSetCurrent = *const fn (usize) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:84 (FnCtxGetCurrent).
/// CUDA signature: `*const fn (ctx_out: *CUcontext) callconv(.c) CUresult`.
pub const FnCtxGetCurrent = *const fn (*usize) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:78 (FnMemcpyHtoDAsync).
/// CUDA signature:
///   `*const fn (CUdeviceptr, *const anyopaque, usize, CUstream)
///       callconv(.c) CUresult`.
/// VK port substitutes VkDeviceBuffer for CUdeviceptr and VkStream for
/// CUstream while preserving argument order.
pub const FnMemcpyHtoDAsync = *const fn (VkDeviceBuffer, *const anyopaque, usize, VkStream) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:79 (FnMemcpyDtoHAsync).
/// CUDA signature:
///   `*const fn (*anyopaque, CUdeviceptr, usize, CUstream)
///       callconv(.c) CUresult`.
pub const FnMemcpyDtoHAsync = *const fn (*anyopaque, VkDeviceBuffer, usize, VkStream) callconv(.c) VkResult;

/// VK adaptation: opaque event handle = u64 index into a VkQueryPool slot.
/// CUDA's CUevent is an opaque driver pointer; the VK port models each
/// event as a single VK_QUERY_TYPE_TIMESTAMP slot index. 0 is reserved
/// so the null check matches CUDA's `event != 0` shape.
pub const VkEvent = u64;

/// CUDA reference: src/decode/cuda_api.zig:86 (FnEventCreate).
/// CUDA signature: `*const fn (*CUevent, c_uint) callconv(.c) CUresult`.
/// VK port swaps CUevent → VkEvent.
pub const FnEventCreate = *const fn (*VkEvent, c_uint) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:87 (FnEventRecord).
/// CUDA signature: `*const fn (CUevent, CUstream) callconv(.c) CUresult`.
/// VK port swaps CUevent → VkEvent and CUstream → VkStream.
pub const FnEventRecord = *const fn (VkEvent, VkStream) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:88 (FnEventSynchronize).
/// CUDA signature: `*const fn (CUevent) callconv(.c) CUresult`.
pub const FnEventSynchronize = *const fn (VkEvent) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:89 (FnEventElapsedTime).
/// CUDA signature: `*const fn (*f32, CUevent, CUevent) callconv(.c) CUresult`.
pub const FnEventElapsedTime = *const fn (*f32, VkEvent, VkEvent) callconv(.c) VkResult;

/// CUDA reference: src/decode/cuda_api.zig:90 (FnEventDestroy).
/// CUDA signature: `*const fn (CUevent) callconv(.c) CUresult`.
pub const FnEventDestroy = *const fn (VkEvent) callconv(.c) VkResult;

// VK adaptation: VkDevice is not thread-local; no per-thread context
// binding needed. Returns VK_SUCCESS_RC unconditionally.
fn ctxSetCurrentNoop(_: usize) callconv(.c) VkResult {
    return VK_SUCCESS_RC;
}

// VK adaptation: VkDevice is not thread-local; codec reads the live
// handle off `vulkan_api.ctx` directly. Writes 0 to `ctx_out` so the
// success-path shape matches CUDA's "ctx_out populated, rc = 0".
fn ctxGetCurrentNoop(ctx_out: *usize) callconv(.c) VkResult {
    ctx_out.* = 0;
    return VK_SUCCESS_RC;
}

/// `procs` table layout. Every slot is an optional function pointer;
/// null on startup, filled by module_loader after vkCreateDevice +
/// VMA bring-up. Codec call sites use `procs.<name>.?(args)` to invoke
/// the slot, matching CUDA's `cu<Op>_fn.?(args)` pattern.
pub const Procs = struct {
    /// CUDA reference: cuMemAlloc_fn. Allocate `size` device bytes.
    /// VK adaptation: calls vma.createBuffer with DEVICE_LOCAL +
    /// STORAGE_BUFFER + TRANSFER_SRC + TRANSFER_DST usage.
    malloc_device: ?*const fn (*VkDeviceBuffer, usize) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemFree_fn. Free a device allocation.
    /// VK adaptation: calls vma.destroyBuffer on the (buffer,
    /// allocation) tuple keyed off `buf`.
    free_device: ?*const fn (VkDeviceBuffer) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemcpyHtoD_fn. Host → device blocking copy.
    /// VK adaptation: writes via a per-context staging buffer +
    /// vkCmdCopyBuffer + vkQueueWaitIdle. Synchronous w.r.t. caller.
    h2d: ?*const fn (VkDeviceBuffer, *const anyopaque, usize) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemcpyDtoH_fn. Device → host blocking copy.
    /// VK adaptation: vkCmdCopyBuffer to staging + vkQueueWaitIdle +
    /// memcpy out of the mapped staging buffer.
    d2h: ?*const fn (*anyopaque, VkDeviceBuffer, usize) callconv(.c) VkResult = null,

    /// CUDA reference: src/decode/cuda_api.zig:117 (cuMemcpyHtoDAsync_fn).
    /// Host → device async copy on `stream`.
    /// VK adaptation: vkCmdCopyBuffer from the persistent staging buffer
    /// into the destination, recorded on the stream's cmdbuf.
    h2d_async: ?FnMemcpyHtoDAsync = null,

    /// CUDA reference: src/decode/cuda_api.zig:118 (cuMemcpyDtoHAsync_fn).
    /// Device → host async copy on `stream`.
    /// VK adaptation: vkCmdCopyBuffer to the staging buffer + memcpy
    /// from the persistent mapping after the stream join.
    d2h_async: ?FnMemcpyDtoHAsync = null,

    /// CUDA reference: cuMemcpyDtoDAsync_fn. Device → device async copy
    /// on `stream`. VK adaptation: vkCmdCopyBuffer recorded into the
    /// stream's command buffer; no wait.
    d2d: ?*const fn (VkDeviceBuffer, VkDeviceBuffer, usize, VkStream) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemsetD8_fn. Device memset (8-bit).
    /// VK adaptation: vkCmdFillBuffer with the byte broadcast to u32 +
    /// vkQueueWaitIdle.
    memset_d8: ?*const fn (VkDeviceBuffer, u8, usize) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemsetD8Async_fn. Async on `stream`.
    memset_d8_async: ?*const fn (VkDeviceBuffer, u8, usize, VkStream) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemAllocHost_fn. Page-locked host allocation.
    /// VK adaptation: vma.createBuffer with HOST_VISIBLE +
    /// HOST_COHERENT + MAPPED, returning the persistent mapping.
    malloc_host: ?*const fn (*?*anyopaque, usize) callconv(.c) VkResult = null,

    /// CUDA reference: cuMemFreeHost_fn.
    free_host: ?*const fn (*anyopaque) callconv(.c) VkResult = null,

    /// CUDA reference: cuStreamSynchronize_fn.
    /// VK adaptation: vkWaitForFences on the stream's fence.
    stream_sync: ?*const fn (VkStream) callconv(.c) VkResult = null,

    /// VK adaptation: NEW SLOT (no CUDA analogue). End+submit the
    /// stream's transfer cmdbuf EARLY so the dedicated DMA engine can
    /// start the queued H2D copies in parallel with subsequent host-
    /// side prep (kernel record, descriptor binding, host imports).
    /// CUDA's cuMemcpyHtoDAsync auto-issues on the call; the VK port
    /// batches them into a transfer cmdbuf that streamEndAndWait
    /// usually submits at sync time. This slot lets the dispatcher
    /// flush the transfer cmdbuf at the end of uploadInputAndPrefixSum
    /// (iter 13) so its H2D overlaps with runBackHalf's host prep.
    /// streamEndAndWait then skips the transfer end+submit but still
    /// makes the compute submit wait on the already-in-flight h2d_sem.
    stream_flush_transfer: ?*const fn (VkStream) callconv(.c) VkResult = null,

    /// CUDA reference: cuCtxSynchronize_fn.
    /// VK adaptation: vkDeviceWaitIdle.
    ctx_sync: ?*const fn () callconv(.c) VkResult = null,

    /// CUDA reference: src/decode/cuda_api.zig:124 (cuCtxSetCurrent_fn).
    /// VK adaptation: VkDevice is not thread-local; no per-thread context
    /// binding needed. Returns VK_SUCCESS_RC unconditionally.
    ctx_set_current: ?FnCtxSetCurrent = ctxSetCurrentNoop,

    /// CUDA reference: src/decode/cuda_api.zig:123 (cuCtxGetCurrent_fn).
    /// VK adaptation: VkDevice is not thread-local; the codec already
    /// reads the live handle off `vulkan_api.ctx`. The slot writes 0 to
    /// `ctx_out` and returns VK_SUCCESS_RC so callers get the same
    /// success-path shape as CUDA.
    ctx_get_current: ?FnCtxGetCurrent = ctxGetCurrentNoop,

    /// CUDA reference: cuStreamCreate_fn.
    /// VK adaptation: allocates a VkCommandPool + VkCommandBuffer +
    /// VkFence triple and returns a registry key into the table.
    stream_create: ?*const fn (*VkStream, c_uint) callconv(.c) VkResult = null,

    /// CUDA reference: cuStreamDestroy_fn.
    stream_destroy: ?*const fn (VkStream) callconv(.c) VkResult = null,

    /// CUDA reference: src/decode/cuda_api.zig:125 (cuEventCreate_fn).
    /// Allocate a timing event (slot index into a VkQueryPool of
    /// VK_QUERY_TYPE_TIMESTAMP). The c_uint argument matches CUDA's
    /// `flags` parameter (ignored on the VK side; timestamps always
    /// queue-stamped on cmdbuf execution).
    event_create: ?FnEventCreate = null,

    /// CUDA reference: src/decode/cuda_api.zig:126 (cuEventRecord_fn).
    /// Record `event` on `stream`. VK adaptation: emits a
    /// vkCmdResetQueryPool + vkCmdWriteTimestamp pair into the stream's
    /// cmdbuf at VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT.
    event_record: ?FnEventRecord = null,

    /// CUDA reference: src/decode/cuda_api.zig:127 (cuEventSynchronize_fn).
    /// Block until `event` has completed on the device.
    /// VK adaptation: vkDeviceWaitIdle (per-event sync has no VK analogue).
    event_synchronize: ?FnEventSynchronize = null,

    /// CUDA reference: src/decode/cuda_api.zig:128 (cuEventElapsedTime_fn).
    /// Read both timestamps via vkGetQueryPoolResults and convert ticks →
    /// ms using the cached VkPhysicalDeviceLimits.timestampPeriod.
    event_elapsed_time: ?FnEventElapsedTime = null,

    /// CUDA reference: src/decode/cuda_api.zig:129 (cuEventDestroy_fn).
    /// Release the query slot back to the free list for reuse.
    event_destroy: ?FnEventDestroy = null,

    /// CUDA reference: cuLaunchKernel_fn. Dispatches the underlying
    /// pipeline.
    /// VK adaptation: wraps vkCmdBindPipeline + vkCmdBindDescriptorSets
    /// + vkCmdPushConstants + vkCmdDispatch into one slot. The `f`
    /// handle is the VkPipeline cast to usize; module_loader stages
    /// each compiled SPV blob into one of these.
    launch_kernel: ?*const fn (
        usize, // pipeline (VkPipeline cast to usize)
        c_uint, // grid_x
        c_uint, // grid_y
        c_uint, // grid_z
        c_uint, // block_x (workgroup local size; baked into SPV)
        c_uint, // block_y
        c_uint, // block_z
        c_uint, // shared_bytes (maps to a workgroup-local SSBO)
        VkStream, // stream
        [*]?*anyopaque, // params
        [*]?*anyopaque, // extra (trailing-null sentinel CUDA uses)
    ) callconv(.c) VkResult = null,
};

/// Single global procs table the codec call sites read from. Module_loader
/// fills the slots after vkCreateDevice + VMA bring-up; until then every
/// call surfaces as a non-zero rc that vkCall maps to BackendNotAvailable.
pub var procs: Procs = .{};

/// CUDA reference: src/decode/cuda_api.zig:132-136. GetProcAddress
/// wrapper that casts to T. Used by the bootstrap lookup of
/// `vkGetInstanceProcAddr` against the vulkan-1.dll handle; every other
/// pointer in the VK loader chain resolves via vkGetInstanceProcAddr /
/// vkGetDeviceProcAddr in module_loader.zig.
pub fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;

// VK adaptation: tagged `[serial_first]` so the parallel test runner
// (srcVK/test_runner_parallel.zig) runs this BEFORE any worker spins up
// — once module_loader.init() runs (e.g. via a sibling decode roundtrip),
// procs.* are non-null and this assertion would fail.
test "procs slots default to null (rejects unset surface) [serial_first]" {
    // Module_loader has not run; every call site that reaches procs.*
    // before init must see a null slot so vkCall can route to
    // BackendNotAvailable. Mirrors CUDA's "cu*_fn is null until init"
    // contract.
    try testing.expect(procs.malloc_device == null);
    try testing.expect(procs.free_device == null);
    try testing.expect(procs.h2d == null);
    try testing.expect(procs.d2h == null);
    try testing.expect(procs.launch_kernel == null);
}

test "VkDeviceBuffer null sentinel is 0" {
    // Codec deinit paths use `if (buf != 0)` to skip the free for
    // unallocated slots - same shape as CUDA.
    try testing.expectEqual(@as(VkDeviceBuffer, 0), VK_NULL_BUFFER);
}

test "qpcMs round-trips a positive delta" {
    const a = qpcNow();
    const b = qpcNow() + 1;
    try testing.expect(qpcMs(a, b) >= 0.0);
}
