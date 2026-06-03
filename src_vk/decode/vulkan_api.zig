//! Vulkan Driver API + Win32 surface used by the GPU-decode pipeline.
//!
//! Why this file exists: every sub-module needs the VkResult/VkDeviceBuffer
//! aliases and one or more of the `vk*_fn` slots — bundling the typedefs,
//! function-pointer slots, dlopen state, and the QPC clock here keeps
//! `module_loader.zig`/`decode_context.zig`/etc. free of duplicate FFI
//! boilerplate.
//!
//! VK PORT NOTE: ports src/decode/cuda_api.zig. CUDA's `CUdeviceptr` (u64)
//! becomes `VkDeviceBuffer` (u64) — also pointer-sized, opaque to the codec.
//! Internally VkDeviceBuffer is a registry key into the VMA-backed
//! allocation table; the codec only ever passes it around through the
//! `procs.*` surface below, which preserves CUDA's surface verbatim:
//!     procs.malloc_device(&buf, size)   <- cuMemAlloc(&dptr, size)
//!     procs.h2d(buf, src_host, size)    <- cuMemcpyHtoD(dptr, src, size)
//!     procs.d2h(dst_host, buf, size)    <- cuMemcpyDtoH(dst, dptr, size)
//!     procs.d2d(dst_buf, src_buf, size, stream)   <- cuMemcpyDtoDAsync(...)
//!     procs.malloc_host(&ptr, size)     <- cuMemAllocHost(&ptr, size)
//!     procs.free_host(ptr)              <- cuMemFreeHost(ptr)
//!     procs.stream_sync(stream)         <- cuStreamSynchronize(stream)
//! Codec call sites read identically to CUDA's, just s/cu/procs./.
//!
//! Function-pointer slots are `pub var` (not `pub const`) because `init()`
//! in module_loader.zig fills them after `LoadLibraryA("vulkan-1.dll")`
//! and `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`, then the rest of
//! the pipeline reads them on every launch.
//!
//! VMA (Vulkan Memory Allocator) is the underlying allocator the foundation
//! routes `malloc_device` / `free_device` through. The Zig bindings live in
//! src_vk/vma.zig; this module re-exports the bare minimum the codec uses
//! so call sites do not have to drag two FFI imports around.

const std = @import("std");

const error_mod = @import("../error.zig");
const vma_mod = @import("../vma.zig");

// VK PORT NOTE: Re-export the L2 stub error so every L2 file under
// decode/ can `try error.NotImplementedL2` without an extra import hop.
pub const NotImplementedL2 = error_mod.NotImplementedL2;

// VK PORT NOTE: Re-export the VMA bindings under `vma` so codec call sites
// reach the C ABI through `vulkan_api.vma.*` (mirrors how the CUDA side
// reaches the driver through `cuda.*`). Tests for the binding live in
// src_vk/vma.zig itself.
pub const vma = vma_mod;

pub const win32 = struct {
    pub extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn QueryPerformanceCounter(count: *i64) callconv(.c) c_int;
    pub extern "kernel32" fn QueryPerformanceFrequency(freq: *i64) callconv(.c) c_int;
};

/// Monotonic high-resolution timestamp (QueryPerformanceCounter). Pair
/// with qpcMs() for elapsed-millisecond timing without an std.Io handle
/// — used by the SLZ_E2E_TIMER instrumentation in init() and the CLI.
pub fn qpcNow() i64 {
    var c: i64 = 0;
    _ = win32.QueryPerformanceCounter(&c);
    return c;
}
// QueryPerformanceFrequency is fixed for the process lifetime — cache
// the first read so qpcMs avoids the syscall on every call. Atomic
// because qpcMs can be called from any thread; the value is idempotent
// so a benign race that re-queries is harmless.
var cached_qpc_freq: std.atomic.Value(i64) = .init(0);
pub fn qpcMs(from: i64, to: i64) f64 {
    var freq = cached_qpc_freq.load(.monotonic);
    if (freq == 0) {
        _ = win32.QueryPerformanceFrequency(&freq);
        if (freq == 0) freq = 1;
        cached_qpc_freq.store(freq, .monotonic);
    }
    return @as(f64, @floatFromInt(to - from)) * 1000.0 / @as(f64, @floatFromInt(freq));
}

// ── Vulkan result code surface ─────────────────────────────────
// Mirrors CUresult / CUDA_SUCCESS. VkResult is the underlying Vulkan
// VkResult enum (c_int). VK_SUCCESS_RC is the all-good sentinel that
// every wrapper returns on success — named with the `_RC` suffix so it
// does not collide with the VK_SUCCESS constant the rest of src_vulkan/
// already exposes through `vk_api.zig`.
pub const VkResult = c_int;
pub const VK_SUCCESS_RC: VkResult = 0;

/// Opaque device-buffer handle the codec passes around in lieu of
/// CUdeviceptr. CUDA gives a real device VA; Vulkan gives a (VkBuffer,
/// VmaAllocation, size, optional BDA) tuple. VkDeviceBuffer is the
/// registry key the VMA-backed module_loader stores those tuples under;
/// the codec only ever consults `procs.*` to translate the handle into
/// a real Vulkan op, so the call sites stay CUDA-shaped (u64 pointer-
/// sized opaque handle, 0 = null / unallocated).
///
/// VK PORT NOTE: 0 stays the null sentinel — matches CUDA's `if (ptr.*
/// != 0)` patterns under `decode_context.deinit` so the codec can reuse
/// the same null-check shape.
pub const VkDeviceBuffer = u64;
pub const VK_NULL_BUFFER: VkDeviceBuffer = 0;

/// Opaque stream / queue handle the codec passes through to procs.*.
/// CUDA's CUstream is `usize`; Vulkan's per-thread work queue is also
/// pointer-sized. 0 = default queue.
pub const VkStream = usize;

// ── Module state (dlopen handle + current device) ───────────────
// Mirrors `cuda.lib` and `cuda.ctx`. `lib` is the vulkan-1.dll dlopen
// handle, `device` holds the bring-up's chosen VkDevice (cast to usize
// for parity with cuda.ctx's pointer-sized slot). `instance` holds the
// matching VkInstance.
pub var lib: ?*anyopaque = null;
pub var instance: usize = 0;
pub var physical_device: usize = 0;
pub var device: usize = 0;
pub var compute_queue: usize = 0;
pub var compute_queue_family: u32 = 0;
pub var vma_allocator: ?vma.VmaAllocator = null;

/// Module-loader bring-up state. `uninit` is the initial value; `init`
/// advances through `in_progress` while it runs (lets a re-entry detect a
/// loop) and finishes at `ready` on success or `failed` if any step
/// bailed. `init` returns true iff `init_state == .ready` afterwards.
pub const InitState = enum { uninit, in_progress, ready, failed };
pub var init_state: InitState = .uninit;

// ── Loader function-pointer slots ────────────────────────────────
// The bootstrap pointer comes from GetProcAddress("vkGetInstanceProcAddr")
// on vulkan-1.dll. Every other function pointer is resolved through
// vkGetInstanceProcAddr (instance-level) or vkGetDeviceProcAddr
// (device-level) after the matching handle exists.
pub const FnGetInstanceProcAddr = *const fn (instance: usize, name: [*:0]const u8) callconv(.c) ?*const fn () callconv(.c) void;
pub const FnGetDeviceProcAddr = *const fn (device: usize, name: [*:0]const u8) callconv(.c) ?*const fn () callconv(.c) void;
pub var vkGetInstanceProcAddr_fn: ?FnGetInstanceProcAddr = null;
pub var vkGetDeviceProcAddr_fn: ?FnGetDeviceProcAddr = null;

// ── procs.* — CUDA-shaped allocation + copy surface ──────────────
// The codec NEVER calls vkAllocateMemory, vkCmdCopyBuffer, or
// vmaCreateBuffer directly. Every device-side allocation, host bounce,
// and synchronization point funnels through one of these function
// pointers so the call sites read identical to the CUDA codec's:
//
//   CUDA  : if (cuMemAlloc_fn.?(&buf, size) != CUDA_SUCCESS) ...
//   VK    : if (procs.malloc_device_fn.?(&buf, size) != VK_SUCCESS_RC) ...
//
// VK PORT NOTE: the foundation wave installs null function pointers
// here. A subsequent wave wires them to the VMA-backed implementations
// (vmaCreateBuffer + a staging-buffer bounce for h2d/d2h, vkCmdCopyBuffer
// for d2d, vkQueueWaitIdle for stream_sync). L1 codec ports build against
// this surface, so the signatures must be locked in now.

/// Allocate `size` bytes of device-local memory. Writes the registry-key
/// handle into `out_buf` on success. Mirrors `cuMemAlloc(&dptr, size)`.
pub const FnMallocDevice = *const fn (out_buf: *VkDeviceBuffer, size: usize) callconv(.c) VkResult;
/// Free a previously-allocated device buffer. Mirrors `cuMemFree(dptr)`.
pub const FnFreeDevice = *const fn (buf: VkDeviceBuffer) callconv(.c) VkResult;
/// Copy `size` bytes from host pointer `src` into device buffer `dst`.
/// Synchronous w.r.t. the calling thread. Mirrors `cuMemcpyHtoD(dst,
/// src, size)`.
pub const FnH2D = *const fn (dst: VkDeviceBuffer, src: *const anyopaque, size: usize) callconv(.c) VkResult;
/// Copy `size` bytes from device buffer `src` into host pointer `dst`.
/// Synchronous w.r.t. the calling thread. Mirrors `cuMemcpyDtoH(dst,
/// src, size)`.
pub const FnD2H = *const fn (dst: *anyopaque, src: VkDeviceBuffer, size: usize) callconv(.c) VkResult;
/// Copy `size` bytes from `src` to `dst` on `stream`, asynchronously.
/// Mirrors `cuMemcpyDtoDAsync(dst, src, size, stream)`.
pub const FnD2D = *const fn (dst: VkDeviceBuffer, src: VkDeviceBuffer, size: usize, stream: VkStream) callconv(.c) VkResult;
/// Memset `size` bytes of `dst` to `value`. Synchronous. Mirrors
/// `cuMemsetD8(dst, value, size)`.
pub const FnMemsetD8 = *const fn (dst: VkDeviceBuffer, value: u8, size: usize) callconv(.c) VkResult;
/// Memset `size` bytes of `dst` to `value` on `stream`. Async.
/// Mirrors `cuMemsetD8Async(dst, value, size, stream)`.
pub const FnMemsetD8Async = *const fn (dst: VkDeviceBuffer, value: u8, size: usize, stream: VkStream) callconv(.c) VkResult;
/// Allocate `size` bytes of page-locked (host-visible, host-coherent)
/// host memory. Mirrors `cuMemAllocHost(&ptr, size)`.
pub const FnMallocHost = *const fn (out_ptr: *?*anyopaque, size: usize) callconv(.c) VkResult;
/// Free a previously-allocated pinned host buffer. Mirrors
/// `cuMemFreeHost(ptr)`.
pub const FnFreeHost = *const fn (ptr: *anyopaque) callconv(.c) VkResult;
/// Block the calling thread until every command previously submitted to
/// `stream` has retired. Mirrors `cuStreamSynchronize(stream)`.
pub const FnStreamSync = *const fn (stream: VkStream) callconv(.c) VkResult;
/// Block the calling thread until every command previously submitted on
/// every queue of the bound device has retired. Mirrors
/// `cuCtxSynchronize()`.
pub const FnCtxSync = *const fn () callconv(.c) VkResult;
/// Create a new persistent stream. Mirrors `cuStreamCreate(&s, flags)`.
pub const FnStreamCreate = *const fn (out_stream: *VkStream, flags: c_uint) callconv(.c) VkResult;
/// Destroy a stream created via `stream_create`. Mirrors
/// `cuStreamDestroy(stream)`.
pub const FnStreamDestroy = *const fn (stream: VkStream) callconv(.c) VkResult;

/// procs.* surface: CUDA-shaped allocation + copy entry points.
///
/// Every codec module reaches device memory through this single namespace.
/// The foundation wave leaves the slots null; a subsequent wave wires them
/// to VMA-backed implementations.
pub const procs = struct {
    pub var malloc_device_fn: ?FnMallocDevice = null;
    pub var free_device_fn: ?FnFreeDevice = null;
    pub var h2d_fn: ?FnH2D = null;
    pub var d2h_fn: ?FnD2H = null;
    pub var d2d_fn: ?FnD2D = null;
    pub var memset_d8_fn: ?FnMemsetD8 = null;
    pub var memset_d8_async_fn: ?FnMemsetD8Async = null;
    pub var malloc_host_fn: ?FnMallocHost = null;
    pub var free_host_fn: ?FnFreeHost = null;
    pub var stream_sync_fn: ?FnStreamSync = null;
    pub var ctx_sync_fn: ?FnCtxSync = null;
    pub var stream_create_fn: ?FnStreamCreate = null;
    pub var stream_destroy_fn: ?FnStreamDestroy = null;

    // ── Thin call-site shims ──────────────────────────────────
    // Codec call sites prefer `procs.malloc_device(&buf, size)` over
    // `(procs.malloc_device_fn orelse return error.BackendNotAvailable)
    // (&buf, size)` so a single function call surfaces both the
    // backend-down check and the actual op. These wrappers MUST stay
    // shaped identical to CUDA's `cu*_fn.?(args)` pattern — same
    // argument order, same return-code convention (VK_SUCCESS_RC == 0).

    pub fn malloc_device(out_buf: *VkDeviceBuffer, size: usize) VkResult {
        const f = malloc_device_fn orelse return -1;
        return f(out_buf, size);
    }
    pub fn free_device(buf: VkDeviceBuffer) VkResult {
        const f = free_device_fn orelse return -1;
        return f(buf);
    }
    pub fn h2d(dst: VkDeviceBuffer, src: *const anyopaque, size: usize) VkResult {
        const f = h2d_fn orelse return -1;
        return f(dst, src, size);
    }
    pub fn d2h(dst: *anyopaque, src: VkDeviceBuffer, size: usize) VkResult {
        const f = d2h_fn orelse return -1;
        return f(dst, src, size);
    }
    pub fn d2d(dst: VkDeviceBuffer, src: VkDeviceBuffer, size: usize, stream: VkStream) VkResult {
        const f = d2d_fn orelse return -1;
        return f(dst, src, size, stream);
    }
    pub fn memset_d8(dst: VkDeviceBuffer, value: u8, size: usize) VkResult {
        const f = memset_d8_fn orelse return -1;
        return f(dst, value, size);
    }
    pub fn memset_d8_async(dst: VkDeviceBuffer, value: u8, size: usize, stream: VkStream) VkResult {
        const f = memset_d8_async_fn orelse return -1;
        return f(dst, value, size, stream);
    }
    pub fn malloc_host(out_ptr: *?*anyopaque, size: usize) VkResult {
        const f = malloc_host_fn orelse return -1;
        return f(out_ptr, size);
    }
    pub fn free_host(ptr: *anyopaque) VkResult {
        const f = free_host_fn orelse return -1;
        return f(ptr);
    }
    pub fn stream_sync(stream: VkStream) VkResult {
        const f = stream_sync_fn orelse return -1;
        return f(stream);
    }
    pub fn ctx_sync() VkResult {
        const f = ctx_sync_fn orelse return -1;
        return f();
    }
    pub fn stream_create(out_stream: *VkStream, flags: c_uint) VkResult {
        const f = stream_create_fn orelse return -1;
        return f(out_stream, flags);
    }
    pub fn stream_destroy(stream: VkStream) VkResult {
        const f = stream_destroy_fn orelse return -1;
        return f(stream);
    }
};

/// SM count of the active GPU device, populated by module_loader.init.
/// CUDA reads this from `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`; the
/// Vulkan port derives it from `VkPhysicalDeviceProperties`. Read by
/// callers that need to size launch geometry to the GPU.
/// 0 until init has run successfully.
///
/// VK PORT NOTE: Same field name as CUDA so call sites (encoder's
/// adaptive sc threshold, decode dispatch grid math) read identical
/// across the two ports.
pub var sm_count: u32 = 0;

/// Resolve a function pointer out of the vulkan-1.dll handle. Used only
/// for the bootstrap `vkGetInstanceProcAddr` lookup — every other
/// resolved pointer comes from `vkGetInstanceProcAddr` or
/// `vkGetDeviceProcAddr`. Mirrors `cuda.getProc`.
pub fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "procs.* surface compiles and rejects unset slots" {
    // Foundation wave leaves every slot null; the call-site shims must
    // surface that as a non-zero result code (mirrors CUDA's "backend
    // unavailable" surface where the cu*_fn slot is null).
    var buf: VkDeviceBuffer = 0;
    const rc = procs.malloc_device(&buf, 64);
    try testing.expect(rc != VK_SUCCESS_RC);
}

test "VkDeviceBuffer null sentinel is 0" {
    // Codec deinit paths use `if (ptr.* != 0)` to skip the free for
    // unallocated slots — same shape as CUDA. The 0 sentinel is part
    // of the call-site contract.
    try testing.expectEqual(@as(VkDeviceBuffer, 0), VK_NULL_BUFFER);
}
