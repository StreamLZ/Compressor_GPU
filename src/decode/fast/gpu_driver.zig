//! CUDA Driver API bridge for GPU LZ decompression.
//! Loads PTX at runtime via nvcuda.dll — no nvcc link dependency.
//! The PTX is embedded as a comptime string constant.

const std = @import("std");

// CUDA Driver API types
const CUresult = c_int;
const CUdevice = c_int;
const CUcontext = ?*anyopaque;
const CUmodule = ?*anyopaque;
const CUfunction = ?*anyopaque;
const CUdeviceptr = u64;
const CUstream = ?*anyopaque;

// CUDA Driver API functions (loaded dynamically)
const CudaAPI = struct {
    cuInit: *const fn (c_uint) callconv(.C) CUresult,
    cuDeviceGet: *const fn (*CUdevice, c_int) callconv(.C) CUresult,
    cuCtxCreate_v2: *const fn (*CUcontext, c_uint, CUdevice) callconv(.C) CUresult,
    cuModuleLoadData: *const fn (*CUmodule, [*]const u8) callconv(.C) CUresult,
    cuModuleGetFunction: *const fn (*CUfunction, CUmodule, [*:0]const u8) callconv(.C) CUresult,
    cuMemAlloc_v2: *const fn (*CUdeviceptr, usize) callconv(.C) CUresult,
    cuMemFree_v2: *const fn (CUdeviceptr) callconv(.C) CUresult,
    cuMemcpyHtoD_v2: *const fn (CUdeviceptr, [*]const u8, usize) callconv(.C) CUresult,
    cuMemcpyDtoH_v2: *const fn ([*]u8, CUdeviceptr, usize) callconv(.C) CUresult,
    cuLaunchKernel: *const fn (CUfunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CUstream, [*]?*anyopaque, [*]?*anyopaque) callconv(.C) CUresult,
    cuCtxSynchronize: *const fn () callconv(.C) CUresult,
    cuMemsetD8_v2: *const fn (CUdeviceptr, u8, usize) callconv(.C) CUresult,
};

var cuda_api: ?CudaAPI = null;
var cuda_ctx: CUcontext = null;
var cuda_module: CUmodule = null;
var cuda_kernel: CUfunction = null;

// Embed the PTX at compile time
const kernel_ptx = @embedFile("../../../zig-cache/slz_kernel.ptx");

pub fn init() bool {
    // TODO: Load nvcuda.dll, init context, load PTX module
    // This is the runtime initialization
    return false;
}

pub fn isAvailable() bool {
    return cuda_api != null;
}
