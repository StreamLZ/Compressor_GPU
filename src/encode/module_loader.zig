//! One-shot CUDA module loader for the GPU encoder.
//!
//! Owns the module-level kernel handles for the three encode PTX blobs
//! (lz_kernel, huffman_kernel, assemble_kernel). `init()` is idempotent
//! and resolves only once per process; every sub-module that needs to
//! launch a kernel reads its handle from this file (e.g.
//! `module_loader.kernel_fn`).
//!
//! Encode `init()` always defers to the decode driver's `init()` first so
//! exactly one CUDA context is created per process. A second
//! `cuCtxCreate` would clobber decode's existing allocations. After that,
//! encode loads its own `nvcuda.dll` handle and resolves the driver
//! entrypoints it uses into its parallel `cuda_ffi.zig` slots — the
//! decode side resolves the same entries (including `cuMemsetD8` and
//! `cuMemcpyDtoDAsync`) into its own slots. The duplication is a
//! per-side namespace, not a capability split.
//!
//! Consequence: encode cannot stand alone. Importing only `encode/`
//! still works because the encode init explicitly imports
//! `../decode/driver.zig`, but the binary still links the decode driver.
//!
//! The PTX `@embedFile` calls *must* live in a file located in
//! `src/encode/` so the relative paths resolve to the .ptx blobs
//! that the build emits alongside the .cu sources.

const std = @import("std");
const cuda_ffi = @import("cuda_ffi.zig");

/// Embed a `.ptx` file as a null-terminated byte slice. The CUDA Driver
/// API's `cuModuleLoadData` takes a C-string pointer; embedding the file
/// raw gives a sized byte array that is NOT null-terminated, hence the
/// `++ "\x00"`. The .ptr field of the returned slice is `[*:0]const u8`.
fn nullTerminatedPtx(comptime name: []const u8) [:0]const u8 {
    return @embedFile(name) ++ "\x00";
}

// Per-module handles. CUDA context comes from the decode driver - see
// the file-level doc above.
pub var module: cuda_ffi.CUmodule = 0;
pub var kernel_fn: cuda_ffi.CUfunction = 0;
// v4 #19 device-only Merkle (optional - host fallback when 0):
pub var seg_hash_fn: cuda_ffi.CUfunction = 0;
pub var chunk_combine_fn: cuda_ffi.CUfunction = 0;
pub var merkle_root_write_fn: cuda_ffi.CUfunction = 0;
pub var huff_module: cuda_ffi.CUmodule = 0;
pub var huff_tables_kernel_fn: cuda_ffi.CUfunction = 0;
pub var huff_encode_kernel_fn: cuda_ffi.CUfunction = 0;
pub var assemble_module: cuda_ffi.CUmodule = 0;
pub var assemble_measure_fn: cuda_ffi.CUfunction = 0;
pub var assemble_write_fn: cuda_ffi.CUfunction = 0;
pub var frame_assemble_fn: cuda_ffi.CUfunction = 0;
pub var initialized: bool = false;

// 2026-06-10 (VK lesson backport — srcVK uses g_encode_init_lock):
// the latch was set BEFORE the bring-up work, so a concurrent caller
// saw initialized=true with kernel_fn still 0 and concluded the
// encoder is unavailable. Late arrivals now block on the mutex until
// the winner finishes, then read the settled kernel_fn.
// Separate encode-init lock so chaining into decode's init() (which
// takes its own g_init_lock) never recursively acquires the same
// non-recursive SRWLOCK — mirrors srcVK's g_encode_init_lock split.
const SRWLOCK = extern struct { ptr: ?*anyopaque = null };
extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
var g_encode_init_lock: SRWLOCK = .{};

pub fn init() bool {
    if (initialized) return kernel_fn != 0; // fast path once settled
    AcquireSRWLockExclusive(&g_encode_init_lock);
    defer ReleaseSRWLockExclusive(&g_encode_init_lock);
    if (initialized) return kernel_fn != 0;
    // Latch on exit (success or failure) — AFTER kernel_fn has settled,
    // so the unlocked fast path above can never observe a torn state.
    defer initialized = true;

    // Reuse the CUDA context the decode driver creates - see the
    // file-level doc on why encode does not own context creation.
    const dec_gpu = @import("../decode/driver.zig");
    if (!dec_gpu.init()) return false;

    cuda_ffi._lib = cuda_ffi.win32.LoadLibraryA("nvcuda.dll");
    if (cuda_ffi._lib == null) return false;

    cuda_ffi.cuModuleLoadData_fn = cuda_ffi.getProc(cuda_ffi.FnModuleLoadData, "cuModuleLoadData");
    cuda_ffi.cuModuleGetFunction_fn = cuda_ffi.getProc(cuda_ffi.FnModuleGetFunction, "cuModuleGetFunction");
    cuda_ffi.cuMemAlloc_fn = cuda_ffi.getProc(cuda_ffi.FnMemAlloc, "cuMemAlloc_v2");
    cuda_ffi.cuMemFree_fn = cuda_ffi.getProc(cuda_ffi.FnMemFree, "cuMemFree_v2");
    cuda_ffi.cuMemcpyHtoD_fn = cuda_ffi.getProc(cuda_ffi.FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuda_ffi.cuMemcpyDtoH_fn = cuda_ffi.getProc(cuda_ffi.FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuda_ffi.cuMemcpyDtoDAsync_fn = cuda_ffi.getProc(cuda_ffi.FnMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    cuda_ffi.cuLaunchKernel_fn = cuda_ffi.getProc(cuda_ffi.FnLaunchKernel, "cuLaunchKernel");
    cuda_ffi.cuCtxSynchronize_fn = cuda_ffi.getProc(cuda_ffi.FnCtxSync, "cuCtxSynchronize");
    cuda_ffi.cuMemsetD8_fn = cuda_ffi.getProc(cuda_ffi.FnMemsetD8, "cuMemsetD8_v2");
    cuda_ffi.cuMemGetInfo_fn = cuda_ffi.getProc(cuda_ffi.FnMemGetInfo, "cuMemGetInfo_v2");
    // v4 #17: optional - the async pinned gather falls back to the
    // sync per-chunk loop when any of these are missing.
    cuda_ffi.cuMemAllocHost_fn = cuda_ffi.getProc(cuda_ffi.FnMemAllocHost, "cuMemAllocHost_v2");
    cuda_ffi.cuMemFreeHost_fn = cuda_ffi.getProc(cuda_ffi.FnMemFreeHost, "cuMemFreeHost");
    cuda_ffi.cuMemcpyDtoHAsync_fn = cuda_ffi.getProc(cuda_ffi.FnMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    cuda_ffi.cuStreamSync_fn = cuda_ffi.getProc(cuda_ffi.FnStreamSync, "cuStreamSynchronize");

    const ptx = nullTerminatedPtx("lz_kernel.ptx");
    if ((cuda_ffi.cuModuleLoadData_fn orelse return false)(&module, ptx.ptr) != cuda_ffi.CUDA_SUCCESS) return false;

    const get_fn = cuda_ffi.cuModuleGetFunction_fn orelse return false;
    if (get_fn(&kernel_fn, module, "slzLzEncodeKernel") != cuda_ffi.CUDA_SUCCESS) return false;
    _ = get_fn(&seg_hash_fn, module, "slzSegHashKernel");
    _ = get_fn(&chunk_combine_fn, module, "slzChunkCombineKernel");
    _ = get_fn(&merkle_root_write_fn, module, "slzMerkleRootWriteKernel");

    // GPU Huffman encoder (chunk_type=4). Optional - if the module or
    // either kernel is missing, gpuEncode*Huff returns false and the
    // caller falls back to the CPU Huffman encoder.
    const huff_ptx = nullTerminatedPtx("huffman_kernel.ptx");
    if ((cuda_ffi.cuModuleLoadData_fn orelse return false)(&huff_module, huff_ptx.ptr) == cuda_ffi.CUDA_SUCCESS) {
        _ = get_fn(&huff_tables_kernel_fn, huff_module, "slzHuffBuildTablesKernel");
        _ = get_fn(&huff_encode_kernel_fn, huff_module, "slzHuffEncode4StreamKernel");
    }

    // GPU frame-assembly kernels (chunk_type=4 device-resident compress
    // tail). Optional - gpuAssembleFrameImpl returns false if absent.
    const asm_ptx = nullTerminatedPtx("assemble_kernel.ptx");
    if ((cuda_ffi.cuModuleLoadData_fn orelse return false)(&assemble_module, asm_ptx.ptr) == cuda_ffi.CUDA_SUCCESS) {
        _ = get_fn(&assemble_measure_fn, assemble_module, "slzAssembleMeasureKernel");
        _ = get_fn(&assemble_write_fn, assemble_module, "slzAssembleWriteKernel");
        _ = get_fn(&frame_assemble_fn, assemble_module, "slzFrameAssembleKernel");
    }

    return true;
}

pub fn isAvailable() bool {
    return init();
}
