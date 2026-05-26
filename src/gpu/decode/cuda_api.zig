//! CUDA Driver API + Win32 surface used by the GPU-decode pipeline.
//!
//! Why this file exists: every sub-module needs the CUresult/CUdeviceptr
//! aliases and one or more of the `cuXxx_fn` slots — bundling the
//! typedefs, function-pointer slots, dlopen state, and the QPC clock
//! here keeps `module_loader.zig`/`decode_context.zig`/etc. free of
//! duplicate FFI boilerplate.
//!
//! Function-pointer slots are `pub var` (not `pub const`) because
//! `init()` in module_loader.zig fills them after `LoadLibraryA` and
//! the rest of the pipeline reads them on every launch.

const std = @import("std");

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
// the first read so qpcMs avoids the syscall on every call.
var cached_qpc_freq: i64 = 0;
pub fn qpcMs(from: i64, to: i64) f64 {
    var freq = cached_qpc_freq;
    if (freq == 0) {
        _ = win32.QueryPerformanceFrequency(&freq);
        if (freq == 0) freq = 1;
        cached_qpc_freq = freq;
    }
    return @as(f64, @floatFromInt(to - from)) * 1000.0 / @as(f64, @floatFromInt(freq));
}

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUdeviceptr = u64;
pub const CUDA_SUCCESS: CUresult = 0;

// ── Module state (dlopen handle + current context) ─────────────
pub var lib: ?*anyopaque = null;
pub var ctx: usize = 0;
pub var initialized: bool = false;

// ── Driver API function signatures ──────────────────────────────
pub const FnInit = *const fn (c_uint) callconv(.c) CUresult;
pub const FnDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
pub const FnCtxCreate = *const fn (*usize, c_uint, CUdevice) callconv(.c) CUresult;
pub const FnModuleLoadData = *const fn (*usize, [*]const u8) callconv(.c) CUresult;
pub const FnModuleGetFunction = *const fn (*usize, usize, [*:0]const u8) callconv(.c) CUresult;
pub const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
pub const FnMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
pub const FnMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
pub const FnMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
pub const FnLaunchKernel = *const fn (usize, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, usize, [*]?*anyopaque, [*]?*anyopaque) callconv(.c) CUresult;
pub const FnCtxSync = *const fn () callconv(.c) CUresult;
pub const FnMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult;
pub const FnStreamCreate = *const fn (*usize, c_uint) callconv(.c) CUresult;
pub const FnStreamDestroy = *const fn (usize) callconv(.c) CUresult;
pub const FnStreamSync = *const fn (usize) callconv(.c) CUresult;
pub const FnMemcpyHtoDAsync = *const fn (CUdeviceptr, *const anyopaque, usize, usize) callconv(.c) CUresult;
pub const FnMemcpyDtoDAsync = *const fn (CUdeviceptr, CUdeviceptr, usize, usize) callconv(.c) CUresult;
pub const FnMemsetD8Async = *const fn (CUdeviceptr, u8, usize, usize) callconv(.c) CUresult;
pub const FnMemAllocHost = *const fn (*?*anyopaque, usize) callconv(.c) CUresult;
pub const FnMemFreeHost = *const fn (*anyopaque) callconv(.c) CUresult;
pub const FnCtxGetCurrent = *const fn (*usize) callconv(.c) CUresult;
pub const FnCtxSetCurrent = *const fn (usize) callconv(.c) CUresult;
pub const FnEventCreate = *const fn (*usize, c_uint) callconv(.c) CUresult;
pub const FnEventRecord = *const fn (usize, usize) callconv(.c) CUresult;
pub const FnEventSynchronize = *const fn (usize) callconv(.c) CUresult;
pub const FnEventElapsedTime = *const fn (*f32, usize, usize) callconv(.c) CUresult;
pub const FnEventDestroy = *const fn (usize) callconv(.c) CUresult;

pub var cuInit_fn: ?FnInit = null;
pub var cuDeviceGet_fn: ?FnDeviceGet = null;
pub var cuCtxCreate_fn: ?FnCtxCreate = null;
pub var cuModuleLoadData_fn: ?FnModuleLoadData = null;
pub var cuModuleGetFunction_fn: ?FnModuleGetFunction = null;
pub var cuMemAlloc_fn: ?FnMemAlloc = null;
pub var cuMemFree_fn: ?FnMemFree = null;
pub var cuMemcpyHtoD_fn: ?FnMemcpyHtoD = null;
pub var cuMemcpyDtoH_fn: ?FnMemcpyDtoH = null;
pub var cuLaunchKernel_fn: ?FnLaunchKernel = null;
pub var cuCtxSynchronize_fn: ?FnCtxSync = null;
pub var cuMemsetD8_fn: ?FnMemsetD8 = null;
pub var cuStreamCreate_fn: ?FnStreamCreate = null;
pub var cuStreamDestroy_fn: ?FnStreamDestroy = null;
pub var cuStreamSync_fn: ?FnStreamSync = null;
pub var cuMemcpyHtoDAsync_fn: ?FnMemcpyHtoDAsync = null;
pub var cuMemcpyDtoDAsync_fn: ?FnMemcpyDtoDAsync = null;
pub var cuMemsetD8Async_fn: ?FnMemsetD8Async = null;
pub var cuMemAllocHost_fn: ?FnMemAllocHost = null;
pub var cuMemFreeHost_fn: ?FnMemFreeHost = null;
pub var cuCtxGetCurrent_fn: ?FnCtxGetCurrent = null;
pub var cuCtxSetCurrent_fn: ?FnCtxSetCurrent = null;
pub var cuEventCreate_fn: ?FnEventCreate = null;
pub var cuEventRecord_fn: ?FnEventRecord = null;
pub var cuEventSynchronize_fn: ?FnEventSynchronize = null;
pub var cuEventElapsedTime_fn: ?FnEventElapsedTime = null;
pub var cuEventDestroy_fn: ?FnEventDestroy = null;

/// Pipeline streams (persistent, created once in init). Owned by each
/// DecodeContext; the numeric width drives `DecodeContext.pipeline_streams`.
pub const NUM_PIPELINE_STREAMS = 1;

pub fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}
