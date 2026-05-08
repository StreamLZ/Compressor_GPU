//! CUDA Driver API bridge for GPU LZ decompression.
//! Loads nvcuda.dll at runtime — no compile-time CUDA dependency.
//! PTX kernel is embedded via @embedFile.

const std = @import("std");
const windows = std.os.windows;

const CUresult = c_int;
const CUdevice = c_int;
const CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

var lib: ?windows.HMODULE = null;
var ctx: usize = 0;
var module: usize = 0;
var kernel_fn: usize = 0;
var initialized = false;

// Function pointer types
const FnInit = *const fn (c_uint) callconv(.C) CUresult;
const FnDeviceGet = *const fn (*CUdevice, c_int) callconv(.C) CUresult;
const FnCtxCreate = *const fn (*usize, c_uint, CUdevice) callconv(.C) CUresult;
const FnModuleLoadData = *const fn (*usize, [*]const u8) callconv(.C) CUresult;
const FnModuleGetFunction = *const fn (*usize, usize, [*:0]const u8) callconv(.C) CUresult;
const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.C) CUresult;
const FnMemFree = *const fn (CUdeviceptr) callconv(.C) CUresult;
const FnMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.C) CUresult;
const FnMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.C) CUresult;
const FnLaunchKernel = *const fn (usize, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, usize, [*]?*anyopaque, [*]?*anyopaque) callconv(.C) CUresult;
const FnCtxSync = *const fn () callconv(.C) CUresult;
const FnMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.C) CUresult;

var cuInit_fn: ?FnInit = null;
var cuDeviceGet_fn: ?FnDeviceGet = null;
var cuCtxCreate_fn: ?FnCtxCreate = null;
var cuModuleLoadData_fn: ?FnModuleLoadData = null;
var cuModuleGetFunction_fn: ?FnModuleGetFunction = null;
var cuMemAlloc_fn: ?FnMemAlloc = null;
var cuMemFree_fn: ?FnMemFree = null;
var cuMemcpyHtoD_fn: ?FnMemcpyHtoD = null;
var cuMemcpyDtoH_fn: ?FnMemcpyDtoH = null;
var cuLaunchKernel_fn: ?FnLaunchKernel = null;
var cuCtxSynchronize_fn: ?FnCtxSync = null;
var cuMemsetD8_fn: ?FnMemsetD8 = null;

fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    return @ptrCast(windows.kernel32.GetProcAddress(h, name));
}

pub fn init() bool {
    if (initialized) return kernel_fn != 0;

    initialized = true;

    lib = windows.kernel32.LoadLibraryA("nvcuda.dll") orelse return false;

    cuInit_fn = getProc(FnInit, "cuInit");
    cuDeviceGet_fn = getProc(FnDeviceGet, "cuDeviceGet");
    cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate_v2");
    cuModuleLoadData_fn = getProc(FnModuleLoadData, "cuModuleLoadData");
    cuModuleGetFunction_fn = getProc(FnModuleGetFunction, "cuModuleGetFunction");
    cuMemAlloc_fn = getProc(FnMemAlloc, "cuMemAlloc_v2");
    cuMemFree_fn = getProc(FnMemFree, "cuMemFree_v2");
    cuMemcpyHtoD_fn = getProc(FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_fn = getProc(FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuLaunchKernel_fn = getProc(FnLaunchKernel, "cuLaunchKernel");
    cuCtxSynchronize_fn = getProc(FnCtxSync, "cuCtxSynchronize");
    cuMemsetD8_fn = getProc(FnMemsetD8, "cuMemsetD8_v2");

    const init_fn = cuInit_fn orelse return false;
    if (init_fn(0) != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    const dev_fn = cuDeviceGet_fn orelse return false;
    if (dev_fn(&dev, 0) != CUDA_SUCCESS) return false;

    const ctx_fn = cuCtxCreate_fn orelse return false;
    if (ctx_fn(&ctx, 0, dev) != CUDA_SUCCESS) return false;

    // Load PTX module
    const ptx = @embedFile("slz_kernel.ptx");
    const mod_fn = cuModuleLoadData_fn orelse return false;
    if (mod_fn(&module, ptx.ptr) != CUDA_SUCCESS) return false;

    const get_fn = cuModuleGetFunction_fn orelse return false;
    if (get_fn(&kernel_fn, module, "slzDecompressL1Kernel") != CUDA_SUCCESS) return false;

    return true;
}

pub fn isAvailable() bool {
    return init();
}
