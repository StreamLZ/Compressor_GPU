//! CUDA Driver API bridge for GPU LZ decompression.
//! Loads nvcuda.dll at runtime — no compile-time CUDA dependency.
//! PTX kernel is embedded via @embedFile.

const std = @import("std");
const win32 = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.c) c_int;
    extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.c) c_int;
    extern "kernel32" fn GetStdHandle(nStdHandle: u32) callconv(.c) ?*anyopaque;
    extern "kernel32" fn WriteFile(hFile: *anyopaque, lpBuffer: [*]const u8, nNumberOfBytesToWrite: u32, lpNumberOfBytesWritten: ?*u32, lpOverlapped: ?*anyopaque) callconv(.c) c_int;
};

pub fn writeStderr(msg: []const u8) void {
    std.debug.print("{s}", .{msg});
}

fn qpcMs() i64 {
    var count: i64 = 0;
    var freq: i64 = 0;
    _ = win32.QueryPerformanceCounter(&count);
    _ = win32.QueryPerformanceFrequency(&freq);
    return @divTrunc(count * 1000, freq);
}

const CUresult = c_int;
const CUdevice = c_int;
const CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

var lib: ?*anyopaque = null;
var ctx: usize = 0;
var module: usize = 0;
var kernel_fn: usize = 0;
var batch_kernel_fn: usize = 0;
var full_kernel_fn: usize = 0;
var initialized = false;

// Function pointer types
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
const FnCtxSetCurrent = *const fn (usize) callconv(.c) CUresult;

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
    cuMemsetD8_fn = getProc(FnMemsetD8, "cuMemsetD8_v2");

    const init_fn = cuInit_fn orelse return false;
    const init_result = init_fn(0);
    if (init_result != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    const dev_fn = cuDeviceGet_fn orelse return false;
    const dev_result = dev_fn(&dev, 0);
    if (dev_result != CUDA_SUCCESS) return false;

    cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate_v2");
    if (cuCtxCreate_fn == null) cuCtxCreate_fn = getProc(FnCtxCreate, "cuCtxCreate");
    const ctx_fn = cuCtxCreate_fn orelse return false;
    const ctx_result = ctx_fn(&ctx, 0, dev);
    if (ctx_result != CUDA_SUCCESS) return false;

    // Load PTX module
    const ptx = @embedFile("slz_kernel.ptx") ++ "\x00";
    const mod_fn = cuModuleLoadData_fn orelse return false;
    const mod_result = mod_fn(&module, ptx.ptr);
    if (mod_result != CUDA_SUCCESS) return false;

    const get_fn = cuModuleGetFunction_fn orelse return false;
    const fn_result = get_fn(&kernel_fn, module, "slzDecompressL1Kernel");
    if (fn_result != CUDA_SUCCESS) return false;

    const batch_res = get_fn(&batch_kernel_fn, module, "slzBatchDecompressL1Kernel");
    const full_res = get_fn(&full_kernel_fn, module, "slzFullDecompressL1Kernel");
    if (batch_res != CUDA_SUCCESS or full_res != CUDA_SUCCESS) {
        std.debug.print("batch_kernel={d} full_kernel={d} batch_err={d} full_err={d}\n", .{ batch_kernel_fn, full_kernel_fn, batch_res, full_res });
    }

    return true;
}

pub fn isAvailable() bool {
    return init();
}

const fast_dec = @import("fast_lz_decoder.zig");
const constants = @import("../../format/streamlz_constants.zig");

// Persistent device buffers (grow as needed, never shrink)
var d_output: CUdeviceptr = 0;
var d_output_size: usize = 0;
var d_comp_persist: CUdeviceptr = 0;
var d_comp_persist_size: usize = 0;
var d_descs_persist: CUdeviceptr = 0;
var d_descs_persist_size: usize = 0;

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

pub fn processLzRunsGpu(
    lz: *fast_dec.FastLzTable,
    dst: [*]u8,
    dst_size: usize,
    base_offset: u64,
    src_end: [*]const u8,
    dst_start: [*]const u8,
) fast_dec.DecodeError!void {
    if (!init()) return error.BadMode;

    const alloc_fn = cuMemAlloc_fn orelse return error.BadMode;
    const free_fn = cuMemFree_fn orelse return error.BadMode;
    const h2d_fn = cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuCtxSynchronize_fn orelse return error.BadMode;

    // Stream sizes
    const cmd_size: u32 = @intCast(@intFromPtr(lz.cmd_end) - @intFromPtr(lz.cmd_start));
    const lit_size: u32 = @intCast(@intFromPtr(lz.lit_end) - @intFromPtr(lz.lit_start));
    const off16_count: u32 = @intCast((@intFromPtr(lz.off16_end) - @intFromPtr(lz.off16_start)) / 2);
    const off32_count1: u32 = lz.off32_count1;
    const off32_count2: u32 = lz.off32_count2;
    const len_avail: u32 = @intCast(@intFromPtr(src_end) -| @intFromPtr(lz.length_stream));
    const initial_copy: u32 = if (base_offset == 0) 8 else 0;
    const cmd_stream2_offset: u32 = lz.cmd_stream2_offset;
    const base_off: u32 = @intCast(base_offset);

    // The absolute position of this sub-chunk in the full output
    const dst_offset: u32 = @intCast(@intFromPtr(dst) - @intFromPtr(dst_start));

    // Ensure device output buffer is large enough
    const total_output = dst_offset + @as(u32, @intCast(dst_size));
    if (!ensureDeviceOutput(total_output + 64)) return error.BadMode;

    // Allocate device stream buffers
    var d_cmd: CUdeviceptr = 0;
    var d_lit: CUdeviceptr = 0;
    var d_off16: CUdeviceptr = 0;
    var d_off32_1: CUdeviceptr = 0;
    var d_off32_2: CUdeviceptr = 0;
    var d_len: CUdeviceptr = 0;

    if (alloc_fn(&d_cmd, cmd_size) != CUDA_SUCCESS) return error.BadMode;
    if (alloc_fn(&d_lit, lit_size) != CUDA_SUCCESS) { _ = free_fn(d_cmd); return error.BadMode; }
    if (alloc_fn(&d_off16, off16_count * 2 + 4) != CUDA_SUCCESS) { _ = free_fn(d_cmd); _ = free_fn(d_lit); return error.BadMode; }
    if (alloc_fn(&d_off32_1, off32_count1 * 4 + 4) != CUDA_SUCCESS) { _ = free_fn(d_cmd); _ = free_fn(d_lit); _ = free_fn(d_off16); return error.BadMode; }
    if (alloc_fn(&d_off32_2, off32_count2 * 4 + 4) != CUDA_SUCCESS) { _ = free_fn(d_cmd); _ = free_fn(d_lit); _ = free_fn(d_off16); _ = free_fn(d_off32_1); return error.BadMode; }
    if (alloc_fn(&d_len, len_avail + 4) != CUDA_SUCCESS) { _ = free_fn(d_cmd); _ = free_fn(d_lit); _ = free_fn(d_off16); _ = free_fn(d_off32_1); _ = free_fn(d_off32_2); return error.BadMode; }
    defer { _ = free_fn(d_cmd); _ = free_fn(d_lit); _ = free_fn(d_off16); _ = free_fn(d_off32_1); _ = free_fn(d_off32_2); _ = free_fn(d_len); }

    // Copy streams to device
    _ = h2d_fn(d_cmd, @ptrCast(lz.cmd_start), cmd_size);
    _ = h2d_fn(d_lit, @ptrCast(lz.lit_start), lit_size);
    if (off16_count > 0) _ = h2d_fn(d_off16, @ptrCast(lz.off16_start), off16_count * 2);
    if (off32_count1 > 0) _ = h2d_fn(d_off32_1, @ptrCast(lz.off32_backing1), off32_count1 * 4);
    if (off32_count2 > 0) _ = h2d_fn(d_off32_2, @ptrCast(lz.off32_backing2), off32_count2 * 4);
    if (len_avail > 0) _ = h2d_fn(d_len, @ptrCast(lz.length_stream), len_avail);

    // Sync host output to device — ensures prior sub-chunks' data is available
    // (including CPU-fallback sub-chunks and initial bytes)
    _ = h2d_fn(d_output, @ptrCast(dst_start), dst_offset + initial_copy);

    // Set up kernel parameters
    // The kernel signature: (cmd, cmd_size, lit, lit_size, off16, off16_count,
    //   off32_1, off32_count1, off32_2, off32_count2, len, len_avail,
    //   dst, dst_size, initial_copy, cmd_stream2_offset, base_offset, dst_offset)
    var dst_size_u32: u32 = @intCast(dst_size);

    var p_cmd = d_cmd;
    var p_cmd_size = cmd_size;
    var p_lit = d_lit;
    var p_lit_size = lit_size;
    var p_off16 = d_off16;
    var p_off16_count = off16_count;
    var p_off32_1 = d_off32_1;
    var p_off32_count1 = off32_count1;
    var p_off32_2 = d_off32_2;
    var p_off32_count2 = off32_count2;
    var p_len = d_len;
    var p_len_avail = len_avail;
    var p_dst = d_output;
    var p_initial = initial_copy;
    var p_split = cmd_stream2_offset;
    var p_base = base_off;
    var p_dstoff = dst_offset;

    var params = [_]?*anyopaque{
        @ptrCast(&p_cmd), @ptrCast(&p_cmd_size),
        @ptrCast(&p_lit), @ptrCast(&p_lit_size),
        @ptrCast(&p_off16), @ptrCast(&p_off16_count),
        @ptrCast(&p_off32_1), @ptrCast(&p_off32_count1),
        @ptrCast(&p_off32_2), @ptrCast(&p_off32_count2),
        @ptrCast(&p_len), @ptrCast(&p_len_avail),
        @ptrCast(&p_dst), @ptrCast(&dst_size_u32),
        @ptrCast(&p_initial), @ptrCast(&p_split),
        @ptrCast(&p_base), @ptrCast(&p_dstoff),
    };
    var extra = [_]?*anyopaque{null};

    // Launch: 1 block, 32 threads (1 warp)
    if (launch_fn(kernel_fn, 1, 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
        return error.BadMode;

    if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

    // Copy result back — just this sub-chunk's output
    _ = d2h_fn(@ptrCast(dst), d_output + dst_offset, dst_size);
}

// ────────────────────────────────────────────────────────────
//  Batch parallel: 1 warp per SC group, all groups at once
// ────────────────────────────────────────────────────────────

pub const SubChunkDesc = extern struct {
    cmd_offset: u32,
    cmd_size: u32,
    lit_offset: u32,
    lit_size: u32,
    off16_offset: u32,
    off16_count: u32,
    off32_1_offset: u32,
    off32_count1: u32,
    off32_2_offset: u32,
    off32_count2: u32,
    len_offset: u32,
    len_avail: u32,
    dst_offset: u32,
    dst_size: u32,
    initial_copy: u32,
    cmd_stream2_offset: u32,
};

pub fn batchLaunch(
    descs: []const SubChunkDesc,
    cmd_packed: []const u8,
    lit_packed: []const u8,
    off16_packed: []const u8,
    off32_packed: []const u8,
    len_packed: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    num_groups: u32,
    sub_chunks_per_group: u32,
) fast_dec.DecodeError!void {
    if (!init() or batch_kernel_fn == 0) return error.BadMode;

    const alloc_fn = cuMemAlloc_fn orelse return error.BadMode;
    const free_fn = cuMemFree_fn orelse return error.BadMode;
    const h2d_fn = cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuCtxSynchronize_fn orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(total_output + 64)) return error.BadMode;

    // Upload dictionary / initial bytes to device output buffer
    if (dst_start_off > 0) {
        _ = h2d_fn(d_output, @ptrCast(dst_full), dst_start_off);
    }

    const desc_bytes = descs.len * @sizeOf(SubChunkDesc);
    const cmd_bytes = if (cmd_packed.len > 0) cmd_packed.len else 4;
    const lit_bytes = if (lit_packed.len > 0) lit_packed.len else 4;
    const off16_bytes = if (off16_packed.len > 0) off16_packed.len else 4;
    const off32_bytes = if (off32_packed.len > 0) off32_packed.len else 4;
    const len_bytes = if (len_packed.len > 0) len_packed.len else 4;

    var d_descs: CUdeviceptr = 0;
    var d_cmd: CUdeviceptr = 0;
    var d_lit: CUdeviceptr = 0;
    var d_off16: CUdeviceptr = 0;
    var d_off32: CUdeviceptr = 0;
    var d_len: CUdeviceptr = 0;

    if (alloc_fn(&d_descs, desc_bytes) != CUDA_SUCCESS) return error.BadMode;
    if (alloc_fn(&d_cmd, cmd_bytes) != CUDA_SUCCESS) { _ = free_fn(d_descs); return error.BadMode; }
    if (alloc_fn(&d_lit, lit_bytes) != CUDA_SUCCESS) { _ = free_fn(d_descs); _ = free_fn(d_cmd); return error.BadMode; }
    if (alloc_fn(&d_off16, off16_bytes) != CUDA_SUCCESS) { _ = free_fn(d_descs); _ = free_fn(d_cmd); _ = free_fn(d_lit); return error.BadMode; }
    if (alloc_fn(&d_off32, off32_bytes) != CUDA_SUCCESS) { _ = free_fn(d_descs); _ = free_fn(d_cmd); _ = free_fn(d_lit); _ = free_fn(d_off16); return error.BadMode; }
    if (alloc_fn(&d_len, len_bytes) != CUDA_SUCCESS) { _ = free_fn(d_descs); _ = free_fn(d_cmd); _ = free_fn(d_lit); _ = free_fn(d_off16); _ = free_fn(d_off32); return error.BadMode; }
    defer {
        _ = free_fn(d_descs);
        _ = free_fn(d_cmd);
        _ = free_fn(d_lit);
        _ = free_fn(d_off16);
        _ = free_fn(d_off32);
        _ = free_fn(d_len);
    }

    _ = h2d_fn(d_descs, @ptrCast(descs.ptr), desc_bytes);
    if (cmd_packed.len > 0) _ = h2d_fn(d_cmd, @ptrCast(cmd_packed.ptr), cmd_packed.len);
    if (lit_packed.len > 0) _ = h2d_fn(d_lit, @ptrCast(lit_packed.ptr), lit_packed.len);
    if (off16_packed.len > 0) _ = h2d_fn(d_off16, @ptrCast(off16_packed.ptr), off16_packed.len);
    if (off32_packed.len > 0) _ = h2d_fn(d_off32, @ptrCast(off32_packed.ptr), off32_packed.len);
    if (len_packed.len > 0) _ = h2d_fn(d_len, @ptrCast(len_packed.ptr), len_packed.len);

    var p_cmd = d_cmd;
    var p_lit = d_lit;
    var p_off16 = d_off16;
    var p_off32 = d_off32;
    var p_len = d_len;
    var p_dst = d_output;
    var p_descs_dev = d_descs;
    var p_sc_per_group = sub_chunks_per_group;
    var p_total_sc: u32 = @intCast(descs.len);

    var params = [_]?*anyopaque{
        @ptrCast(&p_cmd),
        @ptrCast(&p_lit),
        @ptrCast(&p_off16),
        @ptrCast(&p_off32),
        @ptrCast(&p_len),
        @ptrCast(&p_dst),
        @ptrCast(&p_descs_dev),
        @ptrCast(&p_sc_per_group),
        @ptrCast(&p_total_sc),
    };
    var extra = [_]?*anyopaque{null};

    if (launch_fn(batch_kernel_fn, num_groups, 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
        return error.BadMode;

    if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

    // D2H: decompressed output (skip dictionary prefix)
    _ = d2h_fn(@ptrCast(dst_full + dst_start_off), d_output + dst_start_off, decompressed_size);
}

// ────────────────────────────────────────────────────────────
//  Phase 2: Full GPU decode — raw compressed block upload
// ────────────────────────────────────────────────────────────

pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

pub fn fullGpuLaunch(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    num_groups: u32,
    chunks_per_group: u32,
) fast_dec.DecodeError!void {
    if (!init() or full_kernel_fn == 0) return error.BadMode;

    const alloc_fn = cuMemAlloc_fn orelse return error.BadMode;
    const free_fn = cuMemFree_fn orelse return error.BadMode;
    const h2d_fn = cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuCtxSynchronize_fn orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(total_output + 64)) return error.BadMode;

    if (dst_start_off > 0)
        _ = h2d_fn(d_output, @ptrCast(dst_full), dst_start_off);

    _ = alloc_fn;
    _ = free_fn;

    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    if (!ensureDeviceBuf(&d_comp_persist, &d_comp_persist_size, comp_bytes)) return error.BadMode;
    if (!ensureDeviceBuf(&d_descs_persist, &d_descs_persist_size, desc_bytes)) return error.BadMode;

    if (compressed_block.len > 0)
        _ = h2d_fn(d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len);
    _ = h2d_fn(d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes);

    var p_comp = d_comp_persist;
    var p_descs_dev = d_descs_persist;
    var p_dst = d_output;
    var p_cpg = chunks_per_group;
    var p_total: u32 = @intCast(chunk_descs.len);

    var params = [_]?*anyopaque{
        @ptrCast(&p_comp),
        @ptrCast(&p_descs_dev),
        @ptrCast(&p_dst),
        @ptrCast(&p_cpg),
        @ptrCast(&p_total),
    };
    var extra = [_]?*anyopaque{null};

    if (launch_fn(full_kernel_fn, num_groups, 1, 1, 32, 1, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
        return error.BadMode;

    if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

    _ = d2h_fn(@ptrCast(dst_full + dst_start_off), d_output + dst_start_off, decompressed_size);
}
