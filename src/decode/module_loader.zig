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
const decode_context = @import("decode_context.zig");
const descriptors = @import("descriptors.zig");

const CUresult = cuda.CUresult;
const CUdevice = cuda.CUdevice;
const CUDA_SUCCESS = cuda.CUDA_SUCCESS;

// ── Kernel module + function handles ─────────────────────────────
/// PTX filenames embedded by `nullTerminatedPtx` below. Named here so
/// a future versioned-PTX scheme (e.g. `lz_kernel.sm89.ptx`) only has
/// to update these two strings, not the two call sites.
const LZ_PTX_NAME: []const u8 = "lz_kernel.ptx";
const HUFF_PTX_NAME: []const u8 = "huffman_kernel.ptx";

/// Embed a `.ptx` file as a null-terminated byte slice. The CUDA Driver
/// API's `cuModuleLoadData` takes a C-string pointer; embedding the file
/// raw gives a sized byte array that is NOT null-terminated, hence the
/// `++ "\x00"`. The .ptr field of the returned slice is `[*:0]const u8`.
fn nullTerminatedPtx(comptime name: []const u8) [:0]const u8 {
    return @embedFile(name) ++ "\x00";
}

pub var module: usize = 0;
pub var kernel_fn: usize = 0;
pub var kernel_raw_fn: usize = 0;
pub var kernel_raw_pipeline_fn: usize = 0;
pub var kernel_general_pipeline_fn: usize = 0;
pub var seg_hash_fn: usize = 0;
pub var chunk_combine_fn: usize = 0;
pub var sc_prefix_apply_fn: usize = 0;
pub var merkle_verdict_fn: usize = 0;
pub var gather_off16_fn: usize = 0;
pub var scan_parse_fn: usize = 0;
pub var walk_frame_fn: usize = 0;
pub var walk_frame_table_fn: usize = 0;
pub var prefix_sum_chunks_fn: usize = 0;
pub var compact_huff_descs_fn: usize = 0;
pub var compact_all_descs_fn: usize = 0;
pub var compact_raw_descs_fn: usize = 0;
pub var merge_huff_descs_fn: usize = 0;
pub var merge_huff_descs_par_fn: usize = 0;
pub var huff_module: usize = 0;
pub var huff_build_fn: usize = 0;
pub var huff_decode_fn: usize = 0;

// 2026-06-10 (VK lesson backport — srcVK uses g_init_lock for exactly
// this): init() was not thread-safe. A second thread arriving while the
// first was mid-bring-up saw `.in_progress` and returned FALSE, i.e.
// reported CUDA unavailable on a working machine. Under the parallel
// test runner that made all but the first GPU test silently skip.
// The mutex makes late arrivals BLOCK until the winner settles the
// state, then read the real result.
const SRWLOCK = extern struct { ptr: ?*anyopaque = null };
extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
var g_init_lock: SRWLOCK = .{};

pub fn init() bool {
    // Fast path once settled (benign unsynchronized read — the state
    // only ever moves monotonically to .ready or .failed).
    if (cuda.init_state == .ready) return true;
    if (cuda.init_state == .failed) return false;
    AcquireSRWLockExclusive(&g_init_lock);
    defer ReleaseSRWLockExclusive(&g_init_lock);
    switch (cuda.init_state) {
        .ready => return true,
        .failed, .in_progress => return false, // .in_progress catches same-thread re-entry
        .uninit => {},
    }
    cuda.init_state = .in_progress;
    // Common bail-out: any path that exits this function while still
    // `.in_progress` transitions to `.failed` so the next init() call
    // doesn't re-try the bring-up. The success path sets `.ready`
    // explicitly at the bottom; defer then sees `.ready` and is a no-op.
    defer if (cuda.init_state == .in_progress) { cuda.init_state = .failed; };

    if (std.c.getenv("SLZ_NO_CUDA") != null) return false;

    // SLZ_E2E_TIMER: break down cold CUDA bring-up (the dominant fixed
    // cost of a one-shot decode - context creation + PTX JIT).
    const init_dbg = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_init0 = cuda.qpcNow();

    cuda.lib = cuda.win32.LoadLibraryA("nvcuda.dll");
    if (cuda.lib == null) return false;

    cuda.cuInit_fn = cuda.getProc(cuda.FnInit, "cuInit");
    cuda.cuDeviceGet_fn = cuda.getProc(cuda.FnDeviceGet, "cuDeviceGet");
    cuda.cuDeviceGetAttribute_fn = cuda.getProc(cuda.FnDeviceGetAttribute, "cuDeviceGetAttribute");
    cuda.cuDeviceGetName_fn = cuda.getProc(cuda.FnDeviceGetName, "cuDeviceGetName");
    cuda.cuCtxCreate_fn = cuda.getProc(cuda.FnCtxCreate, "cuCtxCreate_v2") orelse cuda.getProc(cuda.FnCtxCreate, "cuCtxCreate");
    cuda.cuModuleLoadData_fn = cuda.getProc(cuda.FnModuleLoadData, "cuModuleLoadData");
    cuda.cuModuleLoadDataEx_fn = cuda.getProc(cuda.FnModuleLoadDataEx, "cuModuleLoadDataEx");
    cuda.cuModuleGetFunction_fn = cuda.getProc(cuda.FnModuleGetFunction, "cuModuleGetFunction");
    cuda.cuModuleGetGlobal_fn = cuda.getProc(cuda.FnModuleGetGlobal, "cuModuleGetGlobal_v2");
    cuda.cuMemAlloc_fn = cuda.getProc(cuda.FnMemAlloc, "cuMemAlloc_v2");
    cuda.cuMemFree_fn = cuda.getProc(cuda.FnMemFree, "cuMemFree_v2");
    cuda.cuMemcpyHtoD_fn = cuda.getProc(cuda.FnMemcpyHtoD, "cuMemcpyHtoD_v2");
    cuda.cuMemcpyDtoH_fn = cuda.getProc(cuda.FnMemcpyDtoH, "cuMemcpyDtoH_v2");
    cuda.cuLaunchKernel_fn = cuda.getProc(cuda.FnLaunchKernel, "cuLaunchKernel");
    cuda.cuCtxSynchronize_fn = cuda.getProc(cuda.FnCtxSync, "cuCtxSynchronize");
    cuda.cuMemsetD8_fn = cuda.getProc(cuda.FnMemsetD8, "cuMemsetD8_v2");
    cuda.cuStreamCreate_fn = cuda.getProc(cuda.FnStreamCreate, "cuStreamCreate_v2") orelse cuda.getProc(cuda.FnStreamCreate, "cuStreamCreate");
    cuda.cuStreamDestroy_fn = cuda.getProc(cuda.FnStreamDestroy, "cuStreamDestroy_v2") orelse cuda.getProc(cuda.FnStreamDestroy, "cuStreamDestroy");
    cuda.cuStreamSync_fn = cuda.getProc(cuda.FnStreamSync, "cuStreamSynchronize_v2") orelse cuda.getProc(cuda.FnStreamSync, "cuStreamSynchronize");
    cuda.cuMemcpyHtoDAsync_fn = cuda.getProc(cuda.FnMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    cuda.cuMemcpyDtoHAsync_fn = cuda.getProc(cuda.FnMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    cuda.cuMemcpyDtoDAsync_fn = cuda.getProc(cuda.FnMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    cuda.cuMemsetD8Async_fn = cuda.getProc(cuda.FnMemsetD8Async, "cuMemsetD8Async");
    cuda.cuMemAllocHost_fn = cuda.getProc(cuda.FnMemAllocHost, "cuMemAllocHost_v2");
    cuda.cuMemFreeHost_fn = cuda.getProc(cuda.FnMemFreeHost, "cuMemFreeHost");
    cuda.cuCtxGetCurrent_fn = cuda.getProc(cuda.FnCtxGetCurrent, "cuCtxGetCurrent");
    cuda.cuCtxSetCurrent_fn = cuda.getProc(cuda.FnCtxSetCurrent, "cuCtxSetCurrent");
    cuda.cuEventCreate_fn = cuda.getProc(cuda.FnEventCreate, "cuEventCreate");
    cuda.cuEventRecord_fn = cuda.getProc(cuda.FnEventRecord, "cuEventRecord");
    cuda.cuStreamWaitEvent_fn = cuda.getProc(cuda.FnStreamWaitEvent, "cuStreamWaitEvent");
    cuda.cuEventSynchronize_fn = cuda.getProc(cuda.FnEventSynchronize, "cuEventSynchronize");
    cuda.cuEventElapsedTime_fn = cuda.getProc(cuda.FnEventElapsedTime, "cuEventElapsedTime");
    cuda.cuEventDestroy_fn = cuda.getProc(cuda.FnEventDestroy, "cuEventDestroy_v2");

    if ((cuda.cuInit_fn orelse return false)(0) != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    if ((cuda.cuDeviceGet_fn orelse return false)(&dev, 0) != CUDA_SUCCESS) return false;

    // Cache SM count so callers (fast_framed.zig adaptive sc threshold)
    // can size launch geometry to the actual GPU. Optional: cuDeviceGetAttribute
    // may not resolve on very old driver builds; fall back to 0 (callers
    // treat 0 as "unknown" and use a conservative default).
    if (cuda.cuDeviceGetAttribute_fn) |get_attr| {
        var sm: c_int = 0;
        if (get_attr(&sm, cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev) == CUDA_SUCCESS and sm > 0) {
            cuda.sm_count = @intCast(sm);
        }
    }

    // Cache the device name (VK-parity: the CLI prints `Device: <name>`
    // next to every benchmark so perf numbers stay attributable).
    if (cuda.cuDeviceGetName_fn) |get_name| {
        if (get_name(&cuda.device_name_buf, @intCast(cuda.device_name_buf.len), dev) == CUDA_SUCCESS) {
            cuda.device_name_len = std.mem.indexOfScalar(u8, &cuda.device_name_buf, 0) orelse cuda.device_name_buf.len;
        }
    }

    // Prefer the caller's already-current CUDA context - a library should
    // interoperate with the caller's CUDA / nvCOMP work rather than create
    // a rival context. Only create our own when no context is current
    // (CLI / standalone use).
    if (cuda.cuCtxGetCurrent_fn) |get_current| {
        var existing: usize = 0;
        if (get_current(&existing) == CUDA_SUCCESS and existing != 0) cuda.ctx = existing;
    }
    // Surface which branch the e2e timer measured so the "ctx" cost
    // in the print below is interpretable (piggyback skips cuCtxCreate
    // entirely; standalone pays the full ~40 ms context cost).
    const piggyback_ctx = cuda.ctx != 0;
    if (cuda.ctx == 0) {
        if ((cuda.cuCtxCreate_fn orelse return false)(&cuda.ctx, 0, dev) != CUDA_SUCCESS) return false;
    }
    const t_ctx = cuda.qpcNow();

    const load_fn = cuda.cuModuleLoadData_fn orelse return false;
    const get_fn = cuda.cuModuleGetFunction_fn orelse return false;

    // Load LZ decode kernel (Pass 2)
    const ptx = nullTerminatedPtx(LZ_PTX_NAME);
    // SLZ_JIT_LINEINFO=1 debug hook: ask the JIT to keep line info so
    // compute-sanitizer maps SASS offsets to source lines (the PTX
    // must also be built with -lineinfo; see CLAUDE.md for the manual
    // nvcc/vcvarsall recipe). CU_JIT_GENERATE_LINE_INFO = 13.
    if (std.c.getenv("SLZ_JIT_LINEINFO") != null) {
        if (cuda.cuModuleLoadDataEx_fn) |load_ex| {
            var jit_opts = [_]c_uint{13};
            var jit_vals = [_]?*anyopaque{@ptrFromInt(1)};
            if (load_ex(&module, ptx.ptr, 1, &jit_opts, &jit_vals) != CUDA_SUCCESS) return false;
        } else if (load_fn(&module, ptx.ptr) != CUDA_SUCCESS) return false;
    } else if (load_fn(&module, ptx.ptr) != CUDA_SUCCESS) return false;
    if (get_fn(&kernel_fn, module, "slzLzDecodeKernel") != CUDA_SUCCESS) return false;
    // Optional raw-off16 gather kernel - driver falls back to D2D copies
    // if absent.
    _ = get_fn(&gather_off16_fn, module, "slzGatherRawOff16Kernel");
    // Optional lean L1/L2-raw kernel - driver routes to it when no entropy
    // is present. Failing to load is fine; falls back to general kernel.
    _ = get_fn(&kernel_raw_fn, module, "slzLzDecodeRawKernel");
    _ = get_fn(&kernel_raw_pipeline_fn, module, "slzLzDecodeRawPipelinedKernel");
    _ = get_fn(&kernel_general_pipeline_fn, module, "slzLzDecodeGeneralPipelinedKernel");
    _ = get_fn(&seg_hash_fn, module, "slzSegHashKernel");
    _ = get_fn(&chunk_combine_fn, module, "slzChunkCombineKernel");
    _ = get_fn(&sc_prefix_apply_fn, module, "slzScPrefixApplyKernel");
    _ = get_fn(&merkle_verdict_fn, module, "slzMerkleVerdictKernel");
    // GPU decode-scan kernel. Required: the decode dispatch fails
    // BackendNotAvailable when this symbol is missing (the CPU scan
    // path has been retired).
    _ = get_fn(&scan_parse_fn, module, "slzScanParseKernel");
    // Optional GPU frame-walk kernel. Absent → the D2D decompress path
    // falls back to host bounce.
    _ = get_fn(&walk_frame_fn, module, "slzWalkFrameKernel");
    _ = get_fn(&walk_frame_table_fn, module, "slzWalkFrameTableKernel");
    // Optional GPU prefix-sum-chunks kernel. Absent → pure-D2D pipeline
    // disabled, host computes `first_sub_idx`.
    _ = get_fn(&prefix_sum_chunks_fn, module, "slzPrefixSumChunksKernel");
    // Optional pure-D2D compaction + merge kernels.
    _ = get_fn(&compact_huff_descs_fn, module, "slzCompactHuffDescsKernel");
    _ = get_fn(&compact_raw_descs_fn, module, "slzCompactRawDescsKernel");
    // A-017 backport (2026-06-10): fused 5-block compaction replaces
    // the five sequential single-thread launches above on the hot path;
    // the unfused kernels stay resolved as a reference/fallback.
    _ = get_fn(&compact_all_descs_fn, module, "slzCompactAllDescsKernel");
    _ = get_fn(&merge_huff_descs_fn, module, "slzMergeHuffDescsKernel");
    // B2 (2026-06-10): 4-block parallel merge replaces the serial
    // single-thread merge on the hot path; serial kept as reference.
    _ = get_fn(&merge_huff_descs_par_fn, module, "slzMergeHuffDescsParKernel");
    const t_lz = cuda.qpcNow();

    // Load Huffman decode kernels (Pass 1.5, for chunk_type=4 literals).
    // `slzHuffDecode4StreamKernel`: the "4Stream" suffix is historical
    // (Zig dispatch ABI introduced by the prior 4-stream design); the
    // kernel now decodes HUFF_NUM_STREAMS = 32 streams in parallel
    // (one per warp lane). See src/common/gpu_huffman.cuh.
    const huff_ptx = nullTerminatedPtx(HUFF_PTX_NAME);
    if (load_fn(&huff_module, huff_ptx.ptr) == CUDA_SUCCESS) {
        _ = get_fn(&huff_build_fn, huff_module, "slzHuffBuildLutKernel");
        _ = get_fn(&huff_decode_fn, huff_module, "slzHuffDecode4StreamKernel");
    }
    const t_huff = cuda.qpcNow();
    if (init_dbg) {
        const ctx_kind: [*:0]const u8 = if (piggyback_ctx) "ctx=piggyback" else "ctx=create";
        std.debug.print("[gpu-init] dll+cuInit+{s} {d:.2}  lz-module(PTX JIT) {d:.2}  huff-module(PTX JIT) {d:.2}  total {d:.2} ms\n", .{
            ctx_kind, cuda.qpcMs(t_init0, t_ctx), cuda.qpcMs(t_ctx, t_lz), cuda.qpcMs(t_lz, t_huff), cuda.qpcMs(t_init0, t_huff),
        });
    }

    // Create the persistent pipeline stream (CU_STREAM_NON_BLOCKING = 1)
    // on the module-default context. Per-handle DecodeContexts get the
    // same treatment lazily in `ensurePipelineStream` below. Failure
    // here transitions the loader to `.failed` (the errdefer above doesn't
    // fire because the call returns void - propagate manually).
    ensurePipelineStream(&@import("driver.zig").g_default) catch {
        cuda.init_state = .failed;
        return false;
    };

    cuda.init_state = .ready;
    return true;
}

/// Lazily allocate the persistent pipeline stream on `ctx`. Called from
/// init() for g_default and from fullGpuLaunchImpl for any per-handle
/// context that hasn't created its stream yet.
pub fn ensurePipelineStream(d_ctx: *decode_context.DecodeContext) descriptors.GpuError!void {
    if (d_ctx.pipeline_stream_created) return;
    const create_fn = cuda.cuStreamCreate_fn orelse return error.BackendNotAvailable;
    if (create_fn(&d_ctx.pipeline_stream, 1) != CUDA_SUCCESS) return error.BackendNotAvailable;
    d_ctx.pipeline_stream_created = true;
}

pub fn isAvailable() bool {
    return init();
}
