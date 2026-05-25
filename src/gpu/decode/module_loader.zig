//! One-time CUDA driver bring-up plus the module-level kernel handles
//! every decode path launches against.
//!
//! Kernel handles are `pub var` so `decode_dispatch.zig`, `scan_gpu.zig`,
//! and `decode_context.zig` can dispatch against the slot directly
//! (loaded lazily by `init()`). Optional kernels stay at 0 when their
//! `@embedFile` symbol is missing; call sites check the slot before
//! launch and fall back to the older paths.

const std = @import("std");

const cuda = @import("cuda_api.zig");
const dec_ctx = @import("decode_context.zig");

const CUresult = cuda.CUresult;
const CUdevice = cuda.CUdevice;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;

// ── Kernel module + function handles ─────────────────────────────
pub var module: usize = 0;
pub var kernel_fn: usize = 0;
pub var kernel_raw_fn: usize = 0;
pub var gather_off16_fn: usize = 0;
pub var scan_parse_fn: usize = 0;
pub var walk_frame_fn: usize = 0;
pub var prefix_sum_chunks_fn: usize = 0;
pub var compact_huff_descs_fn: usize = 0;
pub var compact_raw_descs_fn: usize = 0;
pub var merge_huff_descs_fn: usize = 0;
pub var huff_module: usize = 0;
pub var huff_build_fn: usize = 0;
pub var huff_decode_fn: usize = 0;

pub fn init() bool {
    if (cuda.initialized) return kernel_fn != 0;
    cuda.initialized = true;

    if (std.c.getenv("SLZ_NO_CUDA") != null) return false;

    // SLZ_E2E_TIMER: break down cold CUDA bring-up (the dominant fixed
    // cost of a one-shot decode — context creation + PTX JIT).
    const init_dbg = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_init0 = cuda.qpcNow();

    cuda.lib = cuda.win32.LoadLibraryA("nvcuda.dll");
    if (cuda.lib == null) return false;

    cuda.cuInit_fn = cuda.getProc(cuda.FnInit, "cuInit");
    cuda.cuDeviceGet_fn = cuda.getProc(cuda.FnDeviceGet, "cuDeviceGet");
    cuda.cuCtxCreate_fn = cuda.getProc(cuda.FnCtxCreate, "cuCtxCreate_v2") orelse cuda.getProc(cuda.FnCtxCreate, "cuCtxCreate");
    cuda.cuModuleLoadData_fn = cuda.getProc(cuda.FnModuleLoadData, "cuModuleLoadData");
    cuda.cuModuleGetFunction_fn = cuda.getProc(cuda.FnModuleGetFunction, "cuModuleGetFunction");
    cuda.cuMemAlloc_fn = cuda.getProc(cuda.FnMemAlloc, "cuMemAlloc_v2");
    cuda.cuMemFree_fn = cuda.getProc(cuda.FnMemFree, "cuMemFree_v2");
    cuda.cuMemcpyHtoD_fn = cuda.getProc(cuda.FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuda.cuMemcpyDtoH_fn = cuda.getProc(cuda.FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuda.cuLaunchKernel_fn = cuda.getProc(cuda.FnLaunchKernel, "cuLaunchKernel");
    cuda.cuCtxSynchronize_fn = cuda.getProc(cuda.FnCtxSync, "cuCtxSynchronize");
    cuda.cuMemsetD8_fn = cuda.getProc(cuda.FnMemsetD8, "cuMemsetD8_v2");
    cuda.cuStreamCreate_fn = cuda.getProc(cuda.FnStreamCreate, "cuStreamCreate_v2") orelse cuda.getProc(cuda.FnStreamCreate, "cuStreamCreate");
    cuda.cuStreamSync_fn = cuda.getProc(cuda.FnStreamSync, "cuStreamSynchronize_v2") orelse cuda.getProc(cuda.FnStreamSync, "cuStreamSynchronize");
    cuda.cuMemcpyHtoDAsync_fn = cuda.getProc(cuda.FnMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    cuda.cuMemcpyDtoDAsync_fn = cuda.getProc(cuda.FnMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    cuda.cuMemsetD8Async_fn = cuda.getProc(cuda.FnMemsetD8Async, "cuMemsetD8Async");
    cuda.cuMemAllocHost_fn = cuda.getProc(cuda.FnMemAllocHost, "cuMemAllocHost_v2");
    cuda.cuMemFreeHost_fn = cuda.getProc(cuda.FnMemFreeHost, "cuMemFreeHost");
    cuda.cuCtxGetCurrent_fn = cuda.getProc(cuda.FnCtxGetCurrent, "cuCtxGetCurrent");
    cuda.cuCtxSetCurrent_fn = cuda.getProc(cuda.FnCtxSetCurrent, "cuCtxSetCurrent");
    cuda.cuEventCreate_fn = cuda.getProc(cuda.FnEventCreate, "cuEventCreate");
    cuda.cuEventRecord_fn = cuda.getProc(cuda.FnEventRecord, "cuEventRecord");
    cuda.cuEventSynchronize_fn = cuda.getProc(cuda.FnEventSynchronize, "cuEventSynchronize");
    cuda.cuEventElapsedTime_fn = cuda.getProc(cuda.FnEventElapsedTime, "cuEventElapsedTime");
    cuda.cuEventDestroy_fn = cuda.getProc(cuda.FnEventDestroy, "cuEventDestroy_v2");
    cuda.cuFuncSetAttribute_fn = cuda.getProc(cuda.FnFuncSetAttribute, "cuFuncSetAttribute");

    if ((cuda.cuInit_fn orelse return false)(0) != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    if ((cuda.cuDeviceGet_fn orelse return false)(&dev, 0) != CUDA_SUCCESS) return false;

    // Prefer the caller's already-current CUDA context — a library should
    // interoperate with the caller's CUDA / nvCOMP work rather than create
    // a rival context. Only create our own when no context is current
    // (CLI / standalone use).
    if (cuda.cuCtxGetCurrent_fn) |get_current| {
        var existing: usize = 0;
        if (get_current(&existing) == CUDA_SUCCESS and existing != 0) cuda.ctx = existing;
    }
    if (cuda.ctx == 0) {
        if ((cuda.cuCtxCreate_fn orelse return false)(&cuda.ctx, 0, dev) != CUDA_SUCCESS) return false;
    }
    const t_ctx = cuda.qpcNow();

    const load_fn = cuda.cuModuleLoadData_fn orelse return false;
    const get_fn = cuda.cuModuleGetFunction_fn orelse return false;

    // Load LZ decode kernel (Pass 2)
    const ptx = @embedFile("lz_kernel.ptx") ++ "\x00";
    if (load_fn(&module, ptx.ptr) != CUDA_SUCCESS) return false;
    if (get_fn(&kernel_fn, module, "slzLzDecodeKernel") != CUDA_SUCCESS) return false;
    // Optional raw-off16 gather kernel — driver falls back to D2D copies
    // if absent.
    _ = get_fn(&gather_off16_fn, module, "slzGatherRawOff16Kernel");
    // Optional lean L1/L2-raw kernel — driver routes to it when no entropy
    // is present. Failing to load is fine; falls back to general kernel.
    _ = get_fn(&kernel_raw_fn, module, "slzLzDecodeRawKernel");
    // Optional GPU decode-scan kernel (roadmap 4d Phase 2). Absent → the
    // driver keeps the CPU scanForTansChunks path.
    _ = get_fn(&scan_parse_fn, module, "slzScanParseKernel");
    // Optional GPU frame-walk kernel (roadmap 4d Phase 3). Absent → the
    // D2D decompress path falls back to host bounce.
    _ = get_fn(&walk_frame_fn, module, "slzWalkFrameKernel");
    // Optional GPU prefix-sum-chunks kernel (4d Phase 3 step 2). Absent
    // → pure-D2D pipeline disabled, host computes first_sub_idx.
    _ = get_fn(&prefix_sum_chunks_fn, module, "slzPrefixSumChunksKernel");
    // Optional pure-D2D compaction + merge kernels (4d Phase 3 steps 4-5).
    _ = get_fn(&compact_huff_descs_fn, module, "slzCompactHuffDescsKernel");
    _ = get_fn(&compact_raw_descs_fn, module, "slzCompactRawDescsKernel");
    _ = get_fn(&merge_huff_descs_fn, module, "slzMergeHuffDescsKernel");
    const t_lz = cuda.qpcNow();

    // Load Huffman decode kernels (Pass 1.5, for chunk_type=4 literals)
    const huff_ptx = @embedFile("huffman_kernel.ptx") ++ "\x00";
    if (load_fn(&huff_module, huff_ptx.ptr) == CUDA_SUCCESS) {
        _ = get_fn(&huff_build_fn, huff_module, "slzHuffBuildLutKernel");
        _ = get_fn(&huff_decode_fn, huff_module, "slzHuffDecode4StreamKernel");
    }
    const t_huff = cuda.qpcNow();
    if (init_dbg) {
        std.debug.print("[gpu-init] dll+cuInit+ctx {d:.2}  lz-module(PTX JIT) {d:.2}  huff-module(PTX JIT) {d:.2}  total {d:.2} ms\n", .{
            cuda.qpcMs(t_init0, t_ctx), cuda.qpcMs(t_ctx, t_lz), cuda.qpcMs(t_lz, t_huff), cuda.qpcMs(t_init0, t_huff),
        });
    }

    // Create persistent pipeline streams (CU_STREAM_NON_BLOCKING = 1)
    // on the module-default context. Per-handle DecodeContexts get the
    // same treatment lazily in `ensurePipelineStreams` below.
    ensurePipelineStreams(&@import("driver.zig").g_default);

    return true;
}

/// Lazily allocate the persistent pipeline streams on `ctx`. Called from
/// init() for g_default and from fullGpuLaunchImpl for any per-handle
/// context that hasn't created them yet. Without this, h.dec contexts
/// fall through to the non-pipelined branch of fullGpuLaunchImpl, which
/// never launches the Huffman kernels and silently produces zero literals.
pub fn ensurePipelineStreams(d_ctx: *dec_ctx.DecodeContext) void {
    if (d_ctx.pipeline_streams_created) return;
    const create_fn = cuda.cuStreamCreate_fn orelse return;
    for (0..cuda.NUM_PIPELINE_STREAMS) |i| {
        if (create_fn(&d_ctx.pipeline_streams[i], 1) != CUDA_SUCCESS) return;
    }
    d_ctx.pipeline_streams_created = true;
}

pub fn isAvailable() bool {
    return init();
}
