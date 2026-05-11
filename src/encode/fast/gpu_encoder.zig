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

/// Hash bits per level. Sized for 64KB blocks (sc_group=0.25):
/// larger tables don't help ratio but waste VRAM and slow down init.
/// L1-L2 use shared memory; L3+ use global memory.
fn hashBitsForLevel(level: u8) u32 {
    return switch (level) {
        1 => 11,
        2 => 14,
        3 => 17,
        4 => 18,
        5 => 18,
        else => 11,
    };
}

fn useGlobalHash(level: u8) bool {
    return level >= 3;
}

fn useChainParser(level: u8) bool {
    return level >= 5;
}

pub var last_kernel_ns: i64 = 0;

// ── Persistent device buffers ──────────────────────────────────
var d_input_persist: CUdeviceptr = 0;
var d_input_size: usize = 0;
var d_output_persist: CUdeviceptr = 0;
var d_output_size: usize = 0;
var d_descs_persist: CUdeviceptr = 0;
var d_descs_size: usize = 0;
var d_hash_persist: CUdeviceptr = 0;
var d_hash_size: usize = 0;
var d_sizes_persist: CUdeviceptr = 0;
var d_sizes_size: usize = 0;

fn ensureBuf(ptr: *CUdeviceptr, cur: *usize, needed: usize) bool {
    if (cur.* >= needed) return true;
    const free_fn = cuMemFree_fn orelse return false;
    const alloc_fn = cuMemAlloc_fn orelse return false;
    if (ptr.* != 0) _ = free_fn(ptr.*);
    cur.* = 0;
    if (alloc_fn(ptr, needed) != CUDA_SUCCESS) return false;
    cur.* = needed;
    return true;
}

/// Compress all chunks on GPU. Returns per-chunk compressed sizes.
/// Caller provides input data, output buffer, and chunk layout.
pub fn gpuCompress(
    input: []const u8,
    output: []u8,
    chunk_descs: []const CompressChunkDesc,
    comp_sizes_out: []u32,
    io: ?std.Io,
    level: u8,
) bool {
    if (!init()) return false;

    const h2d_fn = cuMemcpyHtoD_fn orelse return false;
    const d2h_fn = cuMemcpyDtoH_fn orelse return false;
    const launch_fn = cuLaunchKernel_fn orelse return false;
    const sync_fn = cuCtxSynchronize_fn orelse return false;

    const num_chunks: u32 = @intCast(chunk_descs.len);
    const desc_bytes = chunk_descs.len * @sizeOf(CompressChunkDesc);
    const sizes_bytes = @as(usize, num_chunks) * 4;
    const hash_bits: u32 = hashBitsForLevel(level);
    const hash_size: usize = @as(usize, 1) << @intCast(hash_bits);
    const global = useGlobalHash(level);
    const chain = useChainParser(level);

    if (!ensureBuf(&d_input_persist, &d_input_size, input.len)) return false;
    if (!ensureBuf(&d_output_persist, &d_output_size, output.len)) return false;
    if (!ensureBuf(&d_descs_persist, &d_descs_size, desc_bytes)) return false;
    if (!ensureBuf(&d_sizes_persist, &d_sizes_size, sizes_bytes)) return false;

    // Global hash tables — chain mode uses 3 tables per block:
    //   first_hash (hash_size u32) + long_hash (hash_size u32) + next_hash (32768 u16 = 16384 u32)
    if (chain) {
        const next_hash_words: usize = 65536 / 2; // 65536 u16 entries = 32768 u32 words
        const table_stride = hash_size + hash_size + next_hash_words;
        const hash_bytes = @as(usize, num_chunks) * table_stride * 4;
        if (!ensureBuf(&d_hash_persist, &d_hash_size, hash_bytes)) return false;
    } else if (global) {
        const hash_bytes = @as(usize, num_chunks) * hash_size * 4;
        if (!ensureBuf(&d_hash_persist, &d_hash_size, hash_bytes)) return false;
    }

    const d_input = d_input_persist;
    const d_output = d_output_persist;
    const d_descs = d_descs_persist;
    const d_sizes = d_sizes_persist;

    // Upload input + descriptors, zero sizes
    _ = h2d_fn(d_input, @ptrCast(input.ptr), input.len);
    _ = h2d_fn(d_descs, @ptrCast(chunk_descs.ptr), desc_bytes);
    if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(d_sizes, 0, sizes_bytes);
    _ = sync_fn();

    const t_before = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

    var p_input = d_input;
    var p_output = d_output;
    var p_descs = d_descs;
    var p_global_hash: CUdeviceptr = if (global or chain) d_hash_persist else 0;
    var p_sizes = d_sizes;
    var p_total = num_chunks;
    var p_hash_bits = hash_bits;
    var p_use_chain: u32 = if (chain) 1 else 0;

    var params = [_]?*anyopaque{
        @ptrCast(&p_input),
        @ptrCast(&p_output),
        @ptrCast(&p_descs),
        @ptrCast(&p_global_hash),
        @ptrCast(&p_sizes),
        @ptrCast(&p_total),
        @ptrCast(&p_hash_bits),
        @ptrCast(&p_use_chain),
    };
    var extra = [_]?*anyopaque{null};

    const shared_bytes: u32 = if (global or chain) 0 else @intCast(hash_size * 4);
    if (launch_fn(kernel_fn, num_chunks, 1, 1, 32, 1, 1, shared_bytes, 0, &params, &extra) != CUDA_SUCCESS)
        return false;

    if (sync_fn() != CUDA_SUCCESS) return false;

    if (t_before) |t_start| {
        if (io) |io_val| {
            last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
        }
    }

    // Download comp_sizes first, then only the actual compressed bytes per block
    _ = d2h_fn(@ptrCast(comp_sizes_out.ptr), d_sizes, sizes_bytes);

    for (0..chunk_descs.len) |i| {
        const cs = comp_sizes_out[i];
        if (cs > 0) {
            const dst_off = chunk_descs[i].dst_offset;
            _ = d2h_fn(@ptrCast(output.ptr + dst_off), d_output + dst_off, cs);
        }
    }

    return true;
}
