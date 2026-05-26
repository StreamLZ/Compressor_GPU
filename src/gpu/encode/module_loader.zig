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
//! encode loads its own `nvcuda.dll` handle and resolves encode-specific
//! entries (cuMemsetD8, cuMemcpyDtoDAsync) that decode does not use.
//!
//! Consequence: encode cannot stand alone. Importing only `gpu/encode/`
//! still works because the encode init explicitly imports
//! `../decode/driver.zig`, but the binary still links the decode driver.
//!
//! The PTX `@embedFile` calls *must* live in a file located in
//! `src/gpu/encode/` so the relative paths resolve to the .ptx blobs
//! that the build emits alongside the .cu sources.

const std = @import("std");
const ffi = @import("cuda_ffi.zig");

/// Embed a `.ptx` file as a null-terminated byte slice. The CUDA Driver
/// API's `cuModuleLoadData` takes a C-string pointer; embedding the file
/// raw gives a sized byte array that is NOT null-terminated, hence the
/// `++ "\x00"`. The .ptr field of the returned slice is `[*:0]const u8`.
fn nullTerminatedPtx(comptime name: []const u8) [:0]const u8 {
    return @embedFile(name) ++ "\x00";
}

// Per-module handles. CUDA context comes from the decode driver — see
// the file-level doc above.
pub var module: ffi.CUmodule = 0;
pub var kernel_fn: ffi.CUfunction = 0;
pub var huff_module: ffi.CUmodule = 0;
pub var huff_tables_kernel_fn: ffi.CUfunction = 0;
pub var huff_encode_kernel_fn: ffi.CUfunction = 0;
pub var assemble_module: ffi.CUmodule = 0;
pub var assemble_measure_fn: ffi.CUfunction = 0;
pub var assemble_write_fn: ffi.CUfunction = 0;
pub var frame_assemble_fn: ffi.CUfunction = 0;
pub var initialized: bool = false;

pub fn init() bool {
    if (initialized) return kernel_fn != 0;
    initialized = true;

    // Reuse the CUDA context the decode driver creates — see the
    // file-level doc on why encode does not own context creation.
    const dec_gpu = @import("../decode/driver.zig");
    if (!dec_gpu.init()) return false;

    ffi._lib = ffi.win32.LoadLibraryA("nvcuda.dll");
    if (ffi._lib == null) return false;

    ffi.cuModuleLoadData_fn = ffi.getProc(ffi.FnModuleLoadData, "cuModuleLoadData");
    ffi.cuModuleGetFunction_fn = ffi.getProc(ffi.FnModuleGetFunction, "cuModuleGetFunction");
    ffi.cuMemAlloc_fn = ffi.getProc(ffi.FnMemAlloc, "cuMemAlloc_v2");
    ffi.cuMemFree_fn = ffi.getProc(ffi.FnMemFree, "cuMemFree_v2");
    ffi.cuMemcpyHtoD_fn = ffi.getProc(ffi.FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    ffi.cuMemcpyDtoH_fn = ffi.getProc(ffi.FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    ffi.cuMemcpyDtoDAsync_fn = ffi.getProc(ffi.FnMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    ffi.cuLaunchKernel_fn = ffi.getProc(ffi.FnLaunchKernel, "cuLaunchKernel");
    ffi.cuCtxSynchronize_fn = ffi.getProc(ffi.FnCtxSync, "cuCtxSynchronize");
    ffi.cuMemsetD8_fn = ffi.getProc(ffi.FnMemsetD8, "cuMemsetD8_v2");

    const ptx = nullTerminatedPtx("lz_kernel.ptx");
    if ((ffi.cuModuleLoadData_fn orelse return false)(&module, ptx.ptr) != ffi.CUDA_SUCCESS) return false;

    const get_fn = ffi.cuModuleGetFunction_fn orelse return false;
    if (get_fn(&kernel_fn, module, "slzLzEncodeKernel") != ffi.CUDA_SUCCESS) return false;

    // GPU Huffman encoder (chunk_type=4). Optional — if the module or
    // either kernel is missing, gpuEncode*Huff returns false and the
    // caller falls back to the CPU Huffman encoder.
    const huff_ptx = nullTerminatedPtx("huffman_kernel.ptx");
    if ((ffi.cuModuleLoadData_fn orelse return false)(&huff_module, huff_ptx.ptr) == ffi.CUDA_SUCCESS) {
        _ = get_fn(&huff_tables_kernel_fn, huff_module, "slzHuffBuildTablesKernel");
        _ = get_fn(&huff_encode_kernel_fn, huff_module, "slzHuffEncode4StreamKernel");
    }

    // GPU frame-assembly kernels (chunk_type=4 device-resident compress
    // tail). Optional — gpuAssembleFrameImpl returns false if absent.
    const asm_ptx = nullTerminatedPtx("assemble_kernel.ptx");
    if ((ffi.cuModuleLoadData_fn orelse return false)(&assemble_module, asm_ptx.ptr) == ffi.CUDA_SUCCESS) {
        _ = get_fn(&assemble_measure_fn, assemble_module, "slzAssembleMeasureKernel");
        _ = get_fn(&assemble_write_fn, assemble_module, "slzAssembleWriteKernel");
        _ = get_fn(&frame_assemble_fn, assemble_module, "slzFrameAssembleKernel");
    }

    return true;
}

pub fn isAvailable() bool {
    return init();
}
