//! GPU L1 compression driver — CUDA Driver API.
//! Loads the compress kernel PTX, launches across all 64KB chunks,
//! reads back per-chunk compressed payloads for the CPU to assemble
//! into the final StreamLZ frame.

const std = @import("std");

const win32 = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
};

const CUresult = c_int;
const CUdevice = c_int;
const CUdeviceptr = u64;
const CUDA_SUCCESS: CUresult = 0;

// ── Module state ────────────────────────────────────────────────
var lib: ?*anyopaque = null;
var ctx: usize = 0;
var module: usize = 0;
var kernel_fn: usize = 0;
var initialized = false;

const FnInit = *const fn (c_uint) callconv(.c) CUresult;
const FnDeviceGet = *const fn (*CUdevice, c_int) callconv(.c) CUresult;
const FnCtxCreate = *const fn (*usize, c_uint, CUdevice) callconv(.c) CUresult;
const FnModuleLoadData = *const fn (*usize, [*]const u8) callconv(.c) CUresult;
const FnModuleGetFunction = *const fn (*usize, usize, [*:0]const u8) callconv(.c) CUresult;
const FnMemAlloc = *const fn (*CUdeviceptr, usize) callconv(.c) CUresult;
const FnMemFree = *const fn (CUdeviceptr) callconv(.c) CUresult;
const FnMemcpyHtoD = *const fn (CUdeviceptr, *const anyopaque, usize) callconv(.c) CUresult;
const FnMemcpyDtoH = *const fn (*anyopaque, CUdeviceptr, usize) callconv(.c) CUresult;
const FnLaunchKernel = *const fn (usize, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, usize, [*]?*anyopaque, [*]?*anyopaque) callconv(.c) CUresult;
const FnCtxSync = *const fn () callconv(.c) CUresult;
const FnMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult;

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
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

pub fn init() bool {
    if (initialized) return kernel_fn != 0;
    initialized = true;

    // Reuse CUDA context from the decode driver if available.
    // This avoids creating a second context which would clobber the first.
    const dec_gpu = @import("../../decode/fast/gpu_driver.zig");
    if (!dec_gpu.init()) return false;

    lib = win32.LoadLibraryA("nvcuda.dll");
    if (lib == null) return false;

    cuModuleLoadData_fn = getProc(FnModuleLoadData, "cuModuleLoadData");
    cuModuleGetFunction_fn = getProc(FnModuleGetFunction, "cuModuleGetFunction");
    cuMemAlloc_fn = getProc(FnMemAlloc, "cuMemAlloc_v2");
    cuMemFree_fn = getProc(FnMemFree, "cuMemFree_v2");
    cuMemcpyHtoD_fn = getProc(FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_fn = getProc(FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuLaunchKernel_fn = getProc(FnLaunchKernel, "cuLaunchKernel");
    cuCtxSynchronize_fn = getProc(FnCtxSync, "cuCtxSynchronize");
    cuMemsetD8_fn = getProc(FnMemsetD8, "cuMemsetD8_v2");

    const ptx = @embedFile("gpu_encode_kernel.ptx") ++ "\x00";
    if ((cuModuleLoadData_fn orelse return false)(&module, ptx.ptr) != CUDA_SUCCESS) return false;

    const get_fn = cuModuleGetFunction_fn orelse return false;
    if (get_fn(&kernel_fn, module, "slzCompressL1Kernel") != CUDA_SUCCESS) return false;

    return true;
}

pub fn isAvailable() bool {
    return init();
}

// ── Chunk descriptor (matches CUDA struct) ──────────────────────
pub const CompressChunkDesc = extern struct {
    src_offset: u32,
    src_size: u32,
    dst_offset: u32,
    dst_capacity: u32,
    is_first: u32,
};

const HASH_SIZE = 2048;

pub var last_kernel_ns: i64 = 0;

/// Compress all chunks on GPU. Returns per-chunk compressed sizes.
/// Caller provides input data, output buffer, and chunk layout.
pub fn gpuCompress(
    input: []const u8,
    output: []u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes_out: []u32,
    io: ?std.Io,
) bool {
    if (!init()) return false;

    const alloc_fn = cuMemAlloc_fn orelse return false;
    const free_fn = cuMemFree_fn orelse return false;
    const h2d_fn = cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuLaunchKernel_fn orelse return false;
    const sync_fn = cuCtxSynchronize_fn orelse return false;

    const num_chunks: u32 = @intCast(chunk_descs.len);
    const desc_bytes = chunk_descs.len * @sizeOf(CompressChunkDesc);
    const hash_bytes = @as(usize, num_chunks) * HASH_SIZE * 4; // u32 per entry
    const sizes_bytes = @as(usize, num_chunks) * 4;

    // Allocate device buffers
    var d_input: CUdeviceptr = 0;
    var d_output: CUdeviceptr = 0;
    var d_descs: CUdeviceptr = 0;
    var d_hash: CUdeviceptr = 0;
    var d_sizes: CUdeviceptr = 0;

    if (alloc_fn(&d_input, input.len) != CUDA_SUCCESS) return false;
    if (alloc_fn(&d_output, output.len) != CUDA_SUCCESS) { _ = free_fn(d_input); return false; }
    if (alloc_fn(&d_descs, desc_bytes) != CUDA_SUCCESS) { _ = free_fn(d_input); _ = free_fn(d_output); return false; }
    if (alloc_fn(&d_hash, hash_bytes) != CUDA_SUCCESS) { _ = free_fn(d_input); _ = free_fn(d_output); _ = free_fn(d_descs); return false; }
    if (alloc_fn(&d_sizes, sizes_bytes) != CUDA_SUCCESS) { _ = free_fn(d_input); _ = free_fn(d_output); _ = free_fn(d_descs); _ = free_fn(d_hash); return false; }
    defer {
        _ = free_fn(d_input);
        _ = free_fn(d_output);
        _ = free_fn(d_descs);
        _ = free_fn(d_hash);
        _ = free_fn(d_sizes);
    }

    // Upload input + descriptors, zero sizes
    _ = h2d_fn(d_input, @ptrCast(input.ptr), input.len);
    _ = h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes);
    if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(d_sizes, 0, sizes_bytes);
    _ = sync_fn();

    const t_before = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    // Launch: 1 warp per block, shared memory hash table
    var p_input = d_input;
    var p_output = d_output;
    var p_descs = d_descs;
    var p_hash = d_hash;
    var p_sizes = d_sizes;
    var p_total = num_chunks;

    var params = [_]?*anyopaque{
        @ptrCast(&p_input),
        @ptrCast(&p_output),
        @ptrCast(&p_descs),
        @ptrCast(&p_hash),
        @ptrCast(&p_sizes),
        @ptrCast(&p_total),
    };
    var extra = [_]?*anyopaque{null};

    if (launch_fn(kernel_fn, num_chunks, 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
        return false;

    if (sync_fn() != CUDA_SUCCESS) return false;

    if (t_before) |t_start| {
        if (io) |io_val| {
            last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    _ = d2h_fn(@ptrCast(output.ptr), d_output, output.len);
    _ = d2h_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes);

    return true;
}
