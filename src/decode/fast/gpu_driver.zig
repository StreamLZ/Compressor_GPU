//! CUDA Driver API bridge for GPU LZ decompression.
//! Loads nvcuda.dll at runtime — no compile-time CUDA dependency.
//! Pre-compiled CUBIN is embedded via @embedFile.

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

// ── Driver API function signatures ──────────────────────────────
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

fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

// ── Initialization ──────────────────────────────────────────────

pub fn init() bool {
    if (initialized) return kernel_fn != 0;
    initialized = true;

    lib = win32.LoadLibraryA("nvcuda.dll");
    if (lib == null) return false;

    cuInit_fn = getProc(FnInit, "cuInit");
    cuDeviceGet_fn = getProc(FnDeviceGet, "cuDeviceGet");
    cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate_v2") orelse getProc(FnCtxCreate, "cuCtxCreate");
    cuModuleLoadData_fn = getProc(FnModuleLoadData, "cuModuleLoadData");
    cuModuleGetFunction_fn = getProc(FnModuleGetFunction, "cuModuleGetFunction");
    cuMemAlloc_fn = getProc(FnMemAlloc, "cuMemAlloc_v2");
    cuMemFree_fn = getProc(FnMemFree, "cuMemFree_v2");
    cuMemcpyHtoD_fn = getProc(FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuMemcpyDtoH_fn = getProc(FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuLaunchKernel_fn = getProc(FnLaunchKernel, "cuLaunchKernel");
    cuCtxSynchronize_fn = getProc(FnCtxSync, "cuCtxSynchronize");

    if ((cuInit_fn orelse return false)(0) != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    if ((cuDeviceGet_fn orelse return false)(&dev, 0) != CUDA_SUCCESS) return false;

    cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate_v2");
    if (cuCtxCreate_fn == null) cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate");
    if ((cuCtxCreate_fn orelse return false)(&ctx, 0, dev) != CUDA_SUCCESS) return false;

    // PTX is portable across all NVIDIA GPUs — the driver JIT-compiles it
    // for the specific hardware at load time. launch_bounds(64,24) in the
    // source emits .minnctapersm 24 which the JIT clamps per-architecture.
    const ptx = @embedFile("gpu_decode_kernel.ptx") ++ "\x00";
    if ((cuModuleLoadData_fn orelse return false)(&module, ptx.ptr) != CUDA_SUCCESS) return false;

    const get_fn = cuModuleGetFunction_fn orelse return false;
    if (get_fn(&kernel_fn, module, "slzFullDecompressL1Kernel") != CUDA_SUCCESS) return false;

    return true;
}

pub fn isAvailable() bool {
    return init();
}

// ── Device buffer management ────────────────────────────────────

const fast_dec = @import("fast_lz_decoder.zig");

var d_output: CUdeviceptr = 0;
var d_output_size: usize = 0;
var d_comp_persist: CUdeviceptr = 0;
var d_comp_persist_size: usize = 0;
var d_descs_persist: CUdeviceptr = 0;
var d_descs_persist_size: usize = 0;

pub var last_kernel_ns: i64 = 0;

fn ensureDeviceBuf(ptr: *CUdeviceptr, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    if (ptr.* != 0) _ = (cuMemFree_fn orelse return false)(ptr.*);
    current_size.* = 0;
    if ((cuMemAlloc_fn orelse return false)(ptr, needed) != CUDA_SUCCESS) return false;
    current_size.* = needed;
    return true;
}

fn ensureDeviceOutput(size: usize) bool {
    return ensureDeviceBuf(&d_output, &d_output_size, size);
}

// ── Full GPU decode ─────────────────────────────────────────────
// Uploads the raw compressed block + chunk descriptors, launches
// slzFullDecompressL1Kernel which parses sub-chunk headers on-GPU.

pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

var d_tans_scratch: CUdeviceptr = 0;
var d_tans_scratch_size: usize = 0;

pub fn fullGpuLaunch(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    num_groups: u32,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    io: ?std.Io,
) fast_dec.DecodeError!void {
    if (!init() or kernel_fn == 0) return error.BadMode;

    const h2d_fn = cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuCtxSynchronize_fn orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(total_output + 64)) return error.BadMode;

    if (dst_start_off > 0)
        _ = h2d_fn(d_output, @ptrCast(dst_full), dst_start_off);

    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    if (!ensureDeviceBuf(&d_comp_persist, &d_comp_persist_size, comp_bytes)) return error.BadMode;
    if (!ensureDeviceBuf(&d_descs_persist, &d_descs_persist_size, desc_bytes)) return error.BadMode;

    // Allocate tANS scratch: 65536 bytes per chunk for decoded tANS literals
    const tans_scratch_bytes = chunk_descs.len * 65536;
    if (!ensureDeviceBuf(&d_tans_scratch, &d_tans_scratch_size, tans_scratch_bytes)) return error.BadMode;

    if (compressed_block.len > 0)
        _ = h2d_fn(d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len);
    _ = h2d_fn(d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes);
    _ = sync_fn();

    const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_comp = d_comp_persist;
    var p_descs_dev = d_descs_persist;
    var p_dst = d_output;
    var p_cpg = chunks_per_group;
    var p_total: u32 = @intCast(chunk_descs.len);
    var p_sc_cap = sub_chunk_cap;
    var p_tans_scratch = d_tans_scratch;

    var params = [_]?*anyopaque{
        @ptrCast(&p_comp),
        @ptrCast(&p_descs_dev),
        @ptrCast(&p_dst),
        @ptrCast(&p_cpg),
        @ptrCast(&p_total),
        @ptrCast(&p_sc_cap),
        @ptrCast(&p_tans_scratch),
    };
    var extra = [_]?*anyopaque{null};

    const grid_x = (num_groups + 1) / 2;
    if (launch_fn(kernel_fn, grid_x, 1, 1, 32, 2, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
        return error.BadMode;

    if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

    if (t_before_kern) |t_start| {
        if (io) |io_val| {
            last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    _ = d2h_fn(@ptrCast(dst_full + dst_start_off), d_output + dst_start_off, decompressed_size);
}
