//! CUDA Driver API FFI shim - shared by every src/encode/*.zig module.
//!
//! Holds the Win32 LoadLibrary handles, the CUresult/CUdevice typedefs,
//! the function-pointer typedefs (`FnXxx`), and the `pub var` slots the
//! loader (`module_loader.zig`) populates at init time. Every other
//! sub-module reads the function pointers from here so we have exactly
//! one definition site for each driver entrypoint.
//!
//! The encode side does NOT own CUDA context creation: `module_loader.init`
//! calls into the decode driver first so that exactly one cuCtxCreate runs
//! per process (a second context would clobber the decode side's
//! allocations). Encode then loads its own `nvcuda.dll` handle + function
//! pointers because every encode sub-module imports `cuda_ffi.zig` for
//! its function-pointer slots — this is a per-side namespace
//! duplication, not a capability gap; the decode side already resolves
//! the same nvcuda entries into its own slots. There are no
//! `cuInit_fn`, `cuDeviceGet_fn`, or `cuCtxCreate_fn` slots here on
//! purpose — encode never resolves them.

const std = @import("std");

pub const win32 = struct {
    pub extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
};

pub const CUresult = c_int;
pub const CUdevice = c_int;
pub const CUdeviceptr = u64;
pub const CUDA_SUCCESS: CUresult = 0;

// Opaque CUDA handles. The driver API treats `CUmodule` and `CUfunction`
// as opaque pointers; we model them as `usize` so module_loader can use
// the typed names instead of bare `usize` slots.
pub const CUmodule = usize;
pub const CUfunction = usize;

// nvcuda.dll handle - populated by module_loader.init(). Underscore-
// prefixed because it's an implementation detail; only this file and
// the encode module_loader touch it.
pub var _lib: ?*anyopaque = null;

pub const FnModuleLoadData = *const fn (*CUmodule, [*]const u8) callconv(.c) CUresult;
pub const FnModuleGetFunction = *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.c) CUresult;
pub const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
pub const FnMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
pub const FnMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
pub const FnMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
pub const FnMemcpyDtoDAsync = *const fn (CUdeviceptr, CUdeviceptr, usize, usize) callconv(.c) CUresult;
pub const FnLaunchKernel = *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, usize, [*]?*anyopaque, [*]?*anyopaque) callconv(.c) CUresult;
pub const FnCtxSync = *const fn () callconv(.c) CUresult;
pub const FnMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult;
pub const FnMemGetInfo = *const fn (*usize, *usize) callconv(.c) CUresult;

pub var cuModuleLoadData_fn: ?FnModuleLoadData = null;
pub var cuModuleGetFunction_fn: ?FnModuleGetFunction = null;
pub var cuMemAlloc_fn: ?FnMemAlloc = null;
pub var cuMemFree_fn: ?FnMemFree = null;
pub var cuMemcpyHtoD_fn: ?FnMemcpyHtoD = null;
pub var cuMemcpyDtoH_fn: ?FnMemcpyDtoH = null;
pub var cuMemcpyDtoDAsync_fn: ?FnMemcpyDtoDAsync = null;
pub var cuLaunchKernel_fn: ?FnLaunchKernel = null;
pub var cuCtxSynchronize_fn: ?FnCtxSync = null;
pub var cuMemsetD8_fn: ?FnMemsetD8 = null;
/// A-023 backport: free/total VRAM query for the batched-dispatch
/// budget. Optional — when missing, the encode dispatch runs unbatched
/// (pre-A-023 behavior, WDDM paging risk at 1 GB chain-parser scale).
pub var cuMemGetInfo_fn: ?FnMemGetInfo = null;

/// Resolve a single exported function from the already-loaded nvcuda
/// handle. Returns null if either the handle or the symbol is missing -
/// the caller decides whether that's fatal (cuModuleLoadData etc.) or
/// merely disables an optional kernel path (huffman, assemble).
pub fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = _lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}
