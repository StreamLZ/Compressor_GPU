//! CUDA Driver API bridge for GPU LZ decompression.
//! Loads nvcuda.dll at runtime — no compile-time CUDA dependency.
//! Pre-compiled PTX is embedded via @embedFile.
//!
//! Two-pass decode pipeline:
//!   Pass 1: slzTansDecodeKernel — decodes tANS literal streams to temp buffer
//!   Pass 2: slzFullDecompressL1Kernel — LZ decode, reads pre-decoded literals

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
var kernel_raw_fn: usize = 0;
var gather_off16_fn: usize = 0;
var tans_module: usize = 0;
var tans_kernel_fn: usize = 0;
var tans_build_fn: usize = 0;
var tans32_kernel_fn: usize = 0;
var tans_parse_fn: usize = 0;
var tans_initlut_fn: usize = 0;
var tans_fused_fn: usize = 0;
var tans_fse_build_fn: usize = 0;
var huff_module: usize = 0;
var huff_build_fn: usize = 0;
var huff_decode_fn: usize = 0;
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
const FnMemsetD8 = *const fn (CUdeviceptr, u8, usize) callconv(.c) CUresult;
const FnStreamCreate = *const fn (*usize, c_uint) callconv(.c) CUresult;
const FnStreamSync = *const fn (usize) callconv(.c) CUresult;
const FnMemcpyHtoDAsync = *const fn (CUdeviceptr, *const anyopaque, usize, usize) callconv(.c) CUresult;
const FnMemcpyDtoDAsync = *const fn (CUdeviceptr, CUdeviceptr, usize, usize) callconv(.c) CUresult;
const FnMemsetD8Async = *const fn (CUdeviceptr, u8, usize, usize) callconv(.c) CUresult;
const FnMemAllocHost = *const fn (*?*anyopaque, usize) callconv(.c) CUresult;
const FnMemFreeHost = *const fn (*anyopaque) callconv(.c) CUresult;
const FnCtxGetCurrent = *const fn (*usize) callconv(.c) CUresult;
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
var cuStreamCreate_fn: ?FnStreamCreate = null;
var cuStreamSync_fn: ?FnStreamSync = null;
var cuMemcpyHtoDAsync_fn: ?FnMemcpyHtoDAsync = null;
var cuMemcpyDtoDAsync_fn: ?FnMemcpyDtoDAsync = null;
var cuMemsetD8Async_fn: ?FnMemsetD8Async = null;
var cuMemAllocHost_fn: ?FnMemAllocHost = null;
var cuMemFreeHost_fn: ?FnMemFreeHost = null;
var cuCtxGetCurrent_fn: ?FnCtxGetCurrent = null;
var cuCtxSetCurrent_fn: ?FnCtxSetCurrent = null;

// Pipeline streams (persistent, created once in init)
const NUM_PIPELINE_STREAMS = 1;

fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(raw);
}

// ── Initialization ──────────────────────────────────────────────

pub fn init() bool {
    if (initialized) return kernel_fn != 0;
    initialized = true;

    if (std.c.getenv("SLZ_NO_CUDA") != null) return false;

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
    cuStreamCreate_fn = getProc(FnStreamCreate, "cuStreamCreate_v2") orelse getProc(FnStreamCreate, "cuStreamCreate");
    cuStreamSync_fn = getProc(FnStreamSync, "cuStreamSynchronize_v2") orelse getProc(FnStreamSync, "cuStreamSynchronize");
    cuMemcpyHtoDAsync_fn = getProc(FnMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    cuMemcpyDtoDAsync_fn = getProc(FnMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    cuMemsetD8Async_fn = getProc(FnMemsetD8Async, "cuMemsetD8Async");
    cuMemAllocHost_fn = getProc(FnMemAllocHost, "cuMemAllocHost_v2");
    cuMemFreeHost_fn = getProc(FnMemFreeHost, "cuMemFreeHost");
    cuCtxGetCurrent_fn = getProc(FnCtxGetCurrent, "cuCtxGetCurrent");
    cuCtxSetCurrent_fn = getProc(FnCtxSetCurrent, "cuCtxSetCurrent");

    if ((cuInit_fn orelse return false)(0) != CUDA_SUCCESS) return false;

    var dev: CUdevice = 0;
    if ((cuDeviceGet_fn orelse return false)(&dev, 0) != CUDA_SUCCESS) return false;

    // Prefer the caller's already-current CUDA context — a library should
    // interoperate with the caller's CUDA / nvCOMP work rather than create
    // a rival context. Only create our own when no context is current
    // (CLI / standalone use).
    if (cuCtxGetCurrent_fn) |get_current| {
        var existing: usize = 0;
        if (get_current(&existing) == CUDA_SUCCESS and existing != 0) ctx = existing;
    }
    if (ctx == 0) {
        if ((cuCtxCreate_fn orelse return false)(&ctx, 0, dev) != CUDA_SUCCESS) return false;
    }

    const load_fn = cuModuleLoadData_fn orelse return false;
    const get_fn = cuModuleGetFunction_fn orelse return false;

    // Load LZ decode kernel (Pass 2)
    const ptx = @embedFile("gpu_decode_kernel.ptx") ++ "\x00";
    if (load_fn(&module, ptx.ptr) != CUDA_SUCCESS) return false;
    if (get_fn(&kernel_fn, module, "slzFullDecompressL1Kernel") != CUDA_SUCCESS) return false;
    // Optional raw-off16 gather kernel — driver falls back to D2D copies
    // if absent.
    _ = get_fn(&gather_off16_fn, module, "slzGatherRawOff16Kernel");
    // Optional lean L1/L2-raw kernel — driver routes to it when no entropy
    // is present. Failing to load is fine; falls back to general kernel.
    _ = get_fn(&kernel_raw_fn, module, "slzFullDecompressL1KernelRaw");

    // Load tANS decode kernel (Pass 1)
    const tans_ptx = @embedFile("gpu_tans_decode_kernel.ptx") ++ "\x00";
    if (load_fn(&tans_module, tans_ptx.ptr) == CUDA_SUCCESS) {
        _ = get_fn(&tans_kernel_fn, tans_module, "slzTansDecodeKernel");
        _ = get_fn(&tans_build_fn, tans_module, "slzTansBuildTablesKernel");
        _ = get_fn(&tans32_kernel_fn, tans_module, "slzTans32DecodeKernel");
        _ = get_fn(&tans_parse_fn, tans_module, "slzTansParseTablesKernel");
        _ = get_fn(&tans_initlut_fn, tans_module, "slzTansInitLutKernel");
        _ = get_fn(&tans_fused_fn, tans_module, "slzTans32FusedKernel");
        _ = get_fn(&tans_fse_build_fn, tans_module, "slzTansFseBuildKernel");
    }

    // Load Huffman decode kernels (Pass 1.5, for chunk_type=4 literals)
    const huff_ptx = @embedFile("gpu_huff_decode_kernel.ptx") ++ "\x00";
    if (load_fn(&huff_module, huff_ptx.ptr) == CUDA_SUCCESS) {
        _ = get_fn(&huff_build_fn, huff_module, "slzHuffBuildLutKernel");
        _ = get_fn(&huff_decode_fn, huff_module, "slzHuffDecode4StreamKernel");
    }

    // Create persistent pipeline streams (CU_STREAM_NON_BLOCKING = 1)
    if (!g_default.pipeline_streams_created) {
        if (cuStreamCreate_fn) |create_fn| {
            var all_ok = true;
            for (0..NUM_PIPELINE_STREAMS) |i| {
                if (create_fn(&g_default.pipeline_streams[i], 1) != CUDA_SUCCESS) {
                    all_ok = false;
                    break;
                }
            }
            if (all_ok) g_default.pipeline_streams_created = true;
        }
    }

    return true;
}

pub fn isAvailable() bool {
    return init();
}

// ── Device buffer management ────────────────────────────────────

const fast_dec = @import("fast_lz_decoder.zig");

pub var last_kernel_ns: i64 = 0;
pub var last_tans_kernel_ns: i64 = 0;
/// Set when SLZ_SPLIT_TIMER=1 — separated Pass 1 (tANS pre-decode) vs
/// Pass 2 (LZ kernel) wall time. last_tans_kernel_ns = tANS only,
/// last_lz_kernel_ns = LZ only. Costs a stream sync between them.
pub var last_lz_kernel_ns: i64 = 0;
pub var last_huff_kernel_ns: i64 = 0;

fn ensureDeviceBuf(ptr: *CUdeviceptr, current_size: *usize, needed: usize) bool {
    if (current_size.* >= needed) return true;
    if (ptr.* != 0) _ = (cuMemFree_fn orelse return false)(ptr.*);
    current_size.* = 0;
    if ((cuMemAlloc_fn orelse return false)(ptr, needed) != CUDA_SUCCESS) return false;
    current_size.* = needed;
    return true;
}

fn ensureDeviceOutput(self: *DecodeContext, size: usize) bool {
    return ensureDeviceBuf(&self.d_output, &self.d_output_size, size);
}

/// Page-locked (pinned) host allocation. D2H/H2D against pinned memory runs
/// at full PCIe bandwidth (~2x pageable, which the driver stages chunk-wise
/// through an internal pinned buffer) and is genuinely async-capable.
/// Returns null when CUDA is unavailable — caller falls back to a normal
/// allocation, so this is always safe to attempt.
pub fn allocHost(size: usize) ?[]u8 {
    if (!init()) return null;
    const f = cuMemAllocHost_fn orelse return null;
    var p: ?*anyopaque = null;
    if (f(&p, size) != CUDA_SUCCESS) return null;
    const base = p orelse return null;
    return @as([*]u8, @ptrCast(base))[0..size];
}

pub fn freeHost(buf: []u8) void {
    const f = cuMemFreeHost_fn orelse return;
    _ = f(@ptrCast(buf.ptr));
}

/// Copy `dst.len` bytes from a device address into the host slice `dst`.
/// Requires init() to have succeeded. Returns false on any CUDA failure.
pub fn copyDeviceToHost(dst: []u8, src_device: u64) bool {
    if (dst.len == 0) return true;
    const f = cuMemcpyDtoH_fn orelse return false;
    return f(@ptrCast(dst.ptr), src_device, dst.len) == CUDA_SUCCESS;
}

/// Copy the host slice `src` to a device address. Requires init() to have
/// succeeded. Returns false on any CUDA failure.
pub fn copyHostToDevice(dst_device: u64, src: []const u8) bool {
    if (src.len == 0) return true;
    const f = cuMemcpyHtoD_fn orelse return false;
    return f(dst_device, @ptrCast(src.ptr), src.len) == CUDA_SUCCESS;
}

/// Make the library's CUDA context current on the calling thread. A
/// driver-API context is current per-thread, so any thread that issues
/// GPU work (e.g. a library-owned worker thread) must call this first.
/// Requires init() to have succeeded.
pub fn bindContextToCallingThread() bool {
    if (ctx == 0) return false;
    const f = cuCtxSetCurrent_fn orelse return false;
    return f(ctx) == CUDA_SUCCESS;
}

// ── Full GPU decode ─────────────────────────────────────────────
// Two-pass pipeline:
//   Pass 1: Launch tANS decode kernel for chunks with tANS literals
//   Pass 2: Launch LZ decode kernel (reads pre-decoded tANS literals)

pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

// tANS chunk descriptor — matches gpu_tans_decode_kernel.cu TansDecChunkDesc
const TansDecChunkDesc = extern struct {
    src_offset: u32, // offset of tANS payload in compressed buffer
    src_size: u32, // tANS compressed size
    dst_offset: u32, // offset in tans_scratch for symbols [0, split_count)
    dst_size: u32, // total decompressed symbol count (countA + countB for paired)
    dst_offset_b: u32, // offset for symbols [split_count, dst_size) — paired unit B
    split_count: u32, // symbols < this go to dst_offset, rest to dst_offset_b
    /// Index into the LUT buffer. Equals the descriptor's index in the
    /// array (== chunk_id) — the scan leaves the sentinel value
    /// `0xFFFFFFFF` here and the merge step backfills it with the final
    /// descriptor index.
    lut_id: u32 = 0xFFFFFFFF,
};

// Huffman literal descriptors — matches gpu_huff_decode_kernel.cu HuffDecChunkDesc.
// in_offset/in_size cover the FULL payload (128 B weights + 9 B sub-header +
// 4 stream payloads). Build kernel reads first 128 B; decode kernel skips 128.
const HuffDecChunkDesc = extern struct {
    in_offset: u32,
    in_size: u32,
    out_offset: u32,
    out_size: u32,
    lut_offset: u32,
};

const HUFF_LUT_ENTRIES: usize = 1024; // matches MAX_CODE_LEN=10 (10-bit escape LUT) in kernel

/// Per-decode-operation mutable state. Every device-buffer pointer, buffer
/// size, host scratch array, and per-operation flag formerly held as a
/// module-global lives here so a future library API can hand each handle
/// its own context. Load-once module handles, kernel/driver function
/// pointers, and `pub var` telemetry stay module-global on purpose.
pub const DecodeContext = struct {
    // ── Output buffer ──────────────────────────────────────────
    d_output: CUdeviceptr = 0,
    d_output_size: usize = 0,

    // ── Persistent compressed-input + descriptor buffers ───────
    d_comp_persist: CUdeviceptr = 0,
    d_comp_persist_size: usize = 0,
    d_descs_persist: CUdeviceptr = 0,
    d_descs_persist_size: usize = 0,

    // ── tANS literal scratch ───────────────────────────────────
    d_tans_scratch: CUdeviceptr = 0,
    d_tans_scratch_size: usize = 0,

    // Debug: one-shot capture of the tans32 decode input (SLZ_DUMP_TANS32=1).
    tans32_dumped: bool = false,

    d_build_timing: CUdeviceptr = 0,
    d_build_timing_size: usize = 0,
    d_parsed_weights: CUdeviceptr = 0,
    d_parsed_weights_size: usize = 0,
    d_work_counter: CUdeviceptr = 0,
    d_work_counter_size: usize = 0,
    h_pinned_output: ?[*]u8 = null,
    h_pinned_output_size: usize = 0,
    d_tans_descs_persist: CUdeviceptr = 0,
    d_tans_descs_persist_size: usize = 0,
    d_tans_status_persist: CUdeviceptr = 0,
    d_tans_status_persist_size: usize = 0,

    // Token tANS scratch + descriptors
    d_tans_tok_scratch: CUdeviceptr = 0,
    d_tans_tok_scratch_size: usize = 0,
    d_tans_tok_descs_persist: CUdeviceptr = 0,
    d_tans_tok_descs_persist_size: usize = 0,
    d_tans_tok_status_persist: CUdeviceptr = 0,
    d_tans_tok_status_persist_size: usize = 0,

    // Off16 scratch (hi bytes at chunk_idx*65536, lo bytes at chunk_idx*65536+32768)
    d_tans_off16_scratch: CUdeviceptr = 0,
    d_tans_off16_scratch_size: usize = 0,

    // Raw off16 gather descriptors (one per raw off16 sub-stream).
    d_raw_off16_descs: CUdeviceptr = 0,
    d_raw_off16_descs_size: usize = 0,

    // Off16 hi/lo tANS scratch + descriptors
    d_tans_off16hi_descs: CUdeviceptr = 0,
    d_tans_off16hi_descs_size: usize = 0,
    d_tans_off16hi_status: CUdeviceptr = 0,
    d_tans_off16hi_status_size: usize = 0,
    d_tans_off16lo_descs: CUdeviceptr = 0,
    d_tans_off16lo_descs_size: usize = 0,
    d_tans_off16lo_status: CUdeviceptr = 0,
    d_tans_off16lo_status_size: usize = 0,

    // Global LUT buffer for tANS decode (replaces shared memory)
    d_tans_lut: CUdeviceptr = 0,
    d_tans_lut_size: usize = 0,

    // Table metadata buffer (table-build kernel → decode kernel)
    d_tans_meta: CUdeviceptr = 0,
    d_tans_meta_size: usize = 0,

    // Per-chunk first-subchunk-index buffer (multi-sub-chunk-per-chunk support).
    // At sc>0.5 chunks have multiple sub-chunks each; each sub-chunk needs its
    // own tANS scratch slot. This array maps chunk_idx → global sub-chunk index.
    d_first_subchunk_idx: CUdeviceptr = 0,
    d_first_subchunk_idx_size: usize = 0,

    // Host-side tANS descriptor buffers (avoids 64KB stack allocations)
    tans_host_buf: [4096]TansDecChunkDesc = undefined,
    tans_tok_host_buf: [4096]TansDecChunkDesc = undefined,
    tans_off16hi_host_buf: [4096]TansDecChunkDesc = undefined,
    tans_off16lo_host_buf: [4096]TansDecChunkDesc = undefined,
    raw_off16_buf: [8192]RawOff16Desc = undefined,

    // Huffman descriptors + LUT.
    d_huff_descs: CUdeviceptr = 0,
    d_huff_descs_size: usize = 0,
    d_huff_lut: CUdeviceptr = 0,
    d_huff_lut_size: usize = 0,
    // One host buffer per stream type — scanner appends to each; fullGpuLaunch
    // merges them into one device array with per-type out_offset added.
    huff_lit_host_buf: [4096]HuffDecChunkDesc = undefined,
    huff_tok_host_buf: [4096]HuffDecChunkDesc = undefined,
    huff_off16hi_host_buf: [4096]HuffDecChunkDesc = undefined,
    huff_off16lo_host_buf: [4096]HuffDecChunkDesc = undefined,

    // Pipeline streams (persistent, created once in init)
    pipeline_streams: [16]usize = .{0} ** 16,
    pipeline_streams_created: bool = false,
};

/// Default decode context backing the module-level public API. Library
/// callers will eventually allocate their own DecodeContext per handle.
pub var g_default: DecodeContext = .{};

// ── tANS header scanning ───────────────────────────────────────
// Scans the host-side compressed data to find tANS literal, token,
// and off16 hi/lo headers. Builds TansDecChunkDesc descriptors for
// the tANS kernel. Raw (type 0) off16 sub-streams are recorded
// for direct H2D copy.

// Describes a raw (type 0 memcpy) off16 sub-stream that needs H2D copy
const RawOff16Desc = struct {
    src_offset: u32, // offset of raw bytes in compressed buffer
    size: u32, // number of bytes
    gpu_offset: u32, // offset in d_tans_off16_scratch
};

const ScanResult = struct {
    num_lit: u32,
    num_tok: u32,
    num_off16hi: u32,
    num_off16lo: u32,
    num_raw_off16: u32,
    num_huff_lit: u32 = 0,
    num_huff_tok: u32 = 0,
    num_huff_off16hi: u32 = 0,
    num_huff_off16lo: u32 = 0,
    use_tans32: bool = false,
};

fn scanForTansChunks(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    sub_chunk_cap: u32,
    tans_lit_descs: []TansDecChunkDesc,
    tans_tok_descs: []TansDecChunkDesc,
    tans_off16hi_descs: []TansDecChunkDesc,
    tans_off16lo_descs: []TansDecChunkDesc,
    raw_off16_descs: []RawOff16Desc,
    huff_lit_descs: []HuffDecChunkDesc,
    huff_tok_descs: []HuffDecChunkDesc,
    huff_off16hi_descs: []HuffDecChunkDesc,
    huff_off16lo_descs: []HuffDecChunkDesc,
    io: ?std.Io,
) ScanResult {
    var num_lit: u32 = 0;
    var num_tok: u32 = 0;
    var num_off16hi: u32 = 0;
    var num_off16lo: u32 = 0;
    var num_raw: u32 = 0;
    var num_huff_lit: u32 = 0;
    var num_huff_tok: u32 = 0;
    var num_huff_hi: u32 = 0;
    var num_huff_lo: u32 = 0;
    var use_tans32: bool = false;
    var cur_sub_idx: u32 = 0; // global sub-chunk index — mirrors driver prefix sum
    const cap_safe: u32 = if (sub_chunk_cap == 0) 65536 else sub_chunk_cap;

    // SLZ_E2E_TIMER: confirm whether this walk is the ~15 ms cost.
    const scan_dbg = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_scan0 = if (scan_dbg)
        (if (io) |iv| std.Io.Clock.awake.now(iv) else null)
    else
        null;
    var dbg_subchunks: u32 = 0;

    for (chunk_descs) |ch| {
        // Compute expected n_subs for this chunk so cur_sub_idx stays in sync
        // with the driver's first_subchunk_idx prefix sum even if we skip.
        const n_subs_expected: u32 = if (ch.flags != 0 or ch.decomp_size == 0) 1
            else (ch.decomp_size + cap_safe - 1) / cap_safe;
        const chunk_first_sub: u32 = cur_sub_idx;
        // Default: advance by full expected count (we'll override only if we
        // successfully walk every sub-chunk).
        defer cur_sub_idx = chunk_first_sub + n_subs_expected;

        if (ch.flags != 0) continue;
        if (ch.decomp_size == 0) continue;
        if (ch.src_offset >= compressed_block.len) continue;

        const chunk_end = @min(ch.src_offset + ch.comp_size, @as(u32, @intCast(compressed_block.len)));
        const chunk_src = compressed_block[ch.src_offset..chunk_end];
        if (chunk_src.len < 3) continue;

        // Walk all sub-chunks within this chunk.
        var sub_pos: u32 = 0;
        var first_sub: bool = true;
        var remaining_decomp: u32 = ch.decomp_size;
        var sub_local_idx: u32 = 0;
        while (remaining_decomp > 0 and sub_pos + 3 <= chunk_src.len) {
            const sub_hdr: u32 = (@as(u32, chunk_src[sub_pos]) << 16) |
                (@as(u32, chunk_src[sub_pos + 1]) << 8) |
                @as(u32, chunk_src[sub_pos + 2]);
            if ((sub_hdr & 0x800000) == 0) break; // not LZ
            const sc_comp_size: u32 = sub_hdr & 0x7FFFF;
            const sub_end = sub_pos + 3 + sc_comp_size;
            if (sub_end > chunk_src.len) break;
            const sub_decomp: u32 = @min(remaining_decomp, cap_safe);

            // Global sub-chunk index for this sub-chunk's tANS descriptors.
            // Slot size = 131072 (sub_chunk_cap) so the largest sub-chunk's
            // literal/token streams fit. off16-lo lives at +65536 within slot.
            const sub_idx: u32 = chunk_first_sub + sub_local_idx;
            const sub_dst_off: usize = @as(usize, sub_idx) * 131072;

            // 8 init bytes only on the very first sub-chunk of the FRAME
            // (sub_idx == 0). All later sub-chunks (including each chunk's
            // own first sub-chunk) get their first 8 bytes restored from
            // the SC prefix table post-decode, not from the stream.
            const init_b: u32 = if (sub_idx == 0) 8 else 0;
            var pos: u32 = sub_pos + 3 + init_b;
            const sub_payload_end: u32 = sub_end;

            walk: {
                if (pos >= sub_payload_end) break :walk;

                // ── Stream 1: Literals ──
                const lit_first = chunk_src[pos];
                const lit_type = (lit_first >> 4) & 0x7;
                if (lit_type == 7) {
                    const lit_before = num_lit;
                    _ = parseTansHeaderPaired(chunk_src, pos, ch.src_offset, sub_idx, tans_lit_descs, &num_lit);
                    use_tans32 = true;
                    if (num_lit > lit_before) {
                        tans_lit_descs[num_lit - 1].src_offset += 128;
                        tans_lit_descs[num_lit - 1].src_size -|= 128;
                    }
                } else if (lit_type == 5) {
                    use_tans32 = true;
                } else if (lit_type == 1 or lit_type == 6) {
                    const lit_before = num_lit;
                    _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, sub_dst_off, tans_lit_descs, &num_lit);
                    if (lit_type == 6) {
                        use_tans32 = true;
                        if (num_lit > lit_before) {
                            tans_lit_descs[num_lit - 1].src_offset += 128;
                            tans_lit_descs[num_lit - 1].src_size -|= 128;
                        }
                    }
                } else if (lit_type == 4) {
                    // Huffman literal stream — parse 3 or 5 byte header
                    // (same convention as tANS type 1/6). Payload after the
                    // header is [128 B weights][9 B sub-header][4 streams].
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off,
                                    huff_lit_descs, &num_huff_lit);
                }
                const lit_next = skipStreamHeader(chunk_src, pos);
                if (lit_next == null) break :walk;
                pos = lit_next.?;

                if (pos >= sub_payload_end) break :walk;

                // ── Stream 2: Tokens (command stream) ──
                const tok_first = chunk_src[pos];
                const tok_type = (tok_first >> 4) & 0x7;
                if (tok_type == 7) {
                    const tok_before = num_tok;
                    _ = parseTansHeaderPaired(chunk_src, pos, ch.src_offset, sub_idx, tans_tok_descs, &num_tok);
                    use_tans32 = true;
                    if (num_tok > tok_before) {
                        tans_tok_descs[num_tok - 1].src_offset += 128;
                        tans_tok_descs[num_tok - 1].src_size -|= 128;
                    }
                } else if (tok_type == 5) {
                    use_tans32 = true;
                } else if (tok_type == 6) {
                    const tok_before = num_tok;
                    _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, sub_dst_off, tans_tok_descs, &num_tok);
                    use_tans32 = true;
                    if (num_tok > tok_before) {
                        tans_tok_descs[num_tok - 1].src_offset += 128;
                        tans_tok_descs[num_tok - 1].src_size -|= 128;
                    }
                } else if (tok_type == 1 and !use_tans32) {
                    _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, sub_dst_off, tans_tok_descs, &num_tok);
                } else if (tok_type == 4) {
                    // Huffman token stream — record sub_dst_off; driver adds
                    // tok_offset later when merging into the unified descriptor array.
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off,
                                    huff_tok_descs, &num_huff_tok);
                }
                const tok_next = skipStreamHeader(chunk_src, pos);
                if (tok_next == null) break :walk;
                pos = tok_next.?;

                if (pos >= sub_payload_end) break :walk;

                // Skip cmd_stream2_offset if sub-chunk > 64KB
                if (sub_decomp > 0x10000) {
                    if (pos + 2 > sub_payload_end) break :walk;
                    pos += 2;
                }

                // ── Off16 stream ──
                if (pos + 2 > sub_payload_end) break :walk;
                const off16_count: u32 = @as(u32, chunk_src[pos]) | (@as(u32, chunk_src[pos + 1]) << 8);
                if (off16_count != 0xFFFF) break :walk; // not entropy-coded
                pos += 2;

                if (pos >= sub_payload_end) break :walk;

                // ── Off16 hi stream ──
                const hi_first = chunk_src[pos];
                const hi_type = (hi_first >> 4) & 0x7;
                if (hi_type == 7) {
                    const hi_before = num_off16hi;
                    _ = parseTansHeaderPaired(chunk_src, pos, ch.src_offset, sub_idx, tans_off16hi_descs, &num_off16hi);
                    use_tans32 = true;
                    if (num_off16hi > hi_before) {
                        tans_off16hi_descs[num_off16hi - 1].src_offset += 128;
                        tans_off16hi_descs[num_off16hi - 1].src_size -|= 128;
                    }
                } else if (hi_type == 5) {
                    use_tans32 = true;
                } else if (hi_type == 1 or hi_type == 6) {
                    const hi_before = num_off16hi;
                    _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, sub_dst_off, tans_off16hi_descs, &num_off16hi);
                    if (hi_type == 6) {
                        use_tans32 = true;
                        if (num_off16hi > hi_before) {
                            tans_off16hi_descs[num_off16hi - 1].src_offset += 128;
                            tans_off16hi_descs[num_off16hi - 1].src_size -|= 128;
                        }
                    }
                } else if (hi_type == 0) {
                    const raw_info = parseType0StreamInfo(chunk_src, pos);
                    if (raw_info.data_offset != 0 and num_raw < raw_off16_descs.len) {
                        raw_off16_descs[num_raw] = .{
                            .src_offset = ch.src_offset + raw_info.data_offset,
                            .size = raw_info.size,
                            .gpu_offset = @intCast(sub_dst_off),
                        };
                        num_raw += 1;
                    }
                } else if (hi_type == 4) {
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off,
                                    huff_off16hi_descs, &num_huff_hi);
                }
                const hi_next = skipStreamHeader(chunk_src, pos);
                if (hi_next == null) break :walk;
                pos = hi_next.?;

                if (pos >= sub_payload_end) break :walk;

                // ── Off16 lo stream ──
                const lo_first = chunk_src[pos];
                const lo_type = (lo_first >> 4) & 0x7;
                if (lo_type == 1 or lo_type == 6) {
                    const lo_before = num_off16lo;
                    _ = parseTansHeaderWithDstOffset(chunk_src, pos, ch.src_offset, sub_dst_off + 65536, tans_off16lo_descs, &num_off16lo);
                    if (lo_type == 6) {
                        use_tans32 = true;
                        if (num_off16lo > lo_before) {
                            tans_off16lo_descs[num_off16lo - 1].src_offset += 128;
                            tans_off16lo_descs[num_off16lo - 1].src_size -|= 128;
                        }
                    }
                } else if (lo_type == 0) {
                    const raw_info = parseType0StreamInfo(chunk_src, pos);
                    if (raw_info.data_offset != 0 and num_raw < raw_off16_descs.len) {
                        raw_off16_descs[num_raw] = .{
                            .src_offset = ch.src_offset + raw_info.data_offset,
                            .size = raw_info.size,
                            .gpu_offset = @intCast(sub_dst_off + 65536),
                        };
                        num_raw += 1;
                    }
                } else if (lo_type == 4) {
                    // Huff lo stream → scratch slot + 65536 (lo half of off16 slot).
                    // Encode sub_dst_off + 65536 here; merge phase adds off16_offset.
                    parseHuffHeader(chunk_src, pos, ch.src_offset, sub_dst_off + 65536,
                                    huff_off16lo_descs, &num_huff_lo);
                }
            } // walk

            // Advance to the next sub-chunk
            sub_pos = sub_end;
            remaining_decomp -= sub_decomp;
            sub_local_idx += 1;
            first_sub = false;
            dbg_subchunks += 1;
        } // while sub-chunks
    } // for chunks

    if (t_scan0) |t0| if (io) |iv| {
        const el: i64 = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
        std.debug.print("  [scan] {d} chunks, {d} sub-chunks, {d} raw-off16, {d:.3} ms ({d:.0} ns/sub-chunk)\n", .{
            chunk_descs.len, dbg_subchunks, num_raw,
            @as(f64, @floatFromInt(el)) / 1e6,
            @as(f64, @floatFromInt(el)) / @as(f64, @floatFromInt(@max(dbg_subchunks, 1))),
        });
    };

    return .{
        .num_lit = num_lit, .num_tok = num_tok, .num_off16hi = num_off16hi,
        .num_off16lo = num_off16lo, .num_raw_off16 = num_raw,
        .num_huff_lit = num_huff_lit,
        .num_huff_tok = num_huff_tok,
        .num_huff_off16hi = num_huff_hi,
        .num_huff_off16lo = num_huff_lo,
        .use_tans32 = use_tans32,
    };
}

/// Parse a chunk_type=4 Huffman literal header at `lit_off` within chunk_src.
/// Writes a HuffDecChunkDesc whose in_offset points at the FULL payload
/// (128 B weights + 9 B sub-header + 4 streams). lut_offset is assigned by
/// index — each descriptor owns 2048 contiguous LUT entries.
fn parseHuffHeader(
    chunk_src: []const u8,
    lit_off: u32,
    src_offset_base: u32,
    dst_offset: usize,
    huff_descs_out: []HuffDecChunkDesc,
    num_huff: *u32,
) void {
    if (lit_off >= chunk_src.len) return;
    const first_byte = chunk_src[lit_off];
    var comp_size: u32 = 0;
    var dst_size: u32 = 0;
    var payload_off: u32 = 0;
    if (first_byte >= 0x80) {
        if (lit_off + 3 > chunk_src.len) return;
        const bits: u32 = (@as(u32, chunk_src[lit_off]) << 16) |
            (@as(u32, chunk_src[lit_off + 1]) << 8) |
            @as(u32, chunk_src[lit_off + 2]);
        comp_size = bits & 0x3FF;
        dst_size = comp_size + ((bits >> 10) & 0x3FF) + 1;
        payload_off = src_offset_base + lit_off + 3;
    } else {
        if (lit_off + 5 > chunk_src.len) return;
        const bits: u32 = (@as(u32, chunk_src[lit_off + 1]) << 24) |
            (@as(u32, chunk_src[lit_off + 2]) << 16) |
            (@as(u32, chunk_src[lit_off + 3]) << 8) |
            @as(u32, chunk_src[lit_off + 4]);
        comp_size = bits & 0x3FFFF;
        dst_size = (((bits >> 18) | (@as(u32, chunk_src[lit_off]) << 14)) & 0x3FFFF) + 1;
        payload_off = src_offset_base + lit_off + 5;
    }
    if (num_huff.* >= huff_descs_out.len) return;
    huff_descs_out[num_huff.*] = .{
        .in_offset = payload_off,
        .in_size = comp_size,
        .out_offset = @intCast(dst_offset),
        .out_size = dst_size,
        .lut_offset = num_huff.* * @as(u32, @intCast(HUFF_LUT_ENTRIES)),
    };
    num_huff.* += 1;
}

/// Parse a type 0 (memcpy) stream header, returning the data offset
/// (relative to chunk start) and the raw byte count.
const Type0Info = struct { data_offset: u32, size: u32 };
fn parseType0StreamInfo(chunk_src: []const u8, pos: u32) Type0Info {
    if (pos >= chunk_src.len) return .{ .data_offset = 0, .size = 0 };
    const first_byte = chunk_src[pos];
    if (first_byte >= 0x80) {
        if (pos + 2 > chunk_src.len) return .{ .data_offset = 0, .size = 0 };
        const sz: u32 = ((@as(u32, chunk_src[pos]) << 8) | @as(u32, chunk_src[pos + 1])) & 0xFFF;
        return .{ .data_offset = pos + 2, .size = sz };
    } else {
        if (pos + 3 > chunk_src.len) return .{ .data_offset = 0, .size = 0 };
        const sz: u32 = (@as(u32, chunk_src[pos]) << 16) | (@as(u32, chunk_src[pos + 1]) << 8) | @as(u32, chunk_src[pos + 2]);
        return .{ .data_offset = pos + 3, .size = sz };
    }
}

/// Skip past an entropy-coded stream header + payload, returning
/// the new position, or null if the data is truncated.
fn skipStreamHeader(chunk_src: []const u8, pos: u32) ?u32 {
    if (pos >= chunk_src.len) return null;
    const first_byte = chunk_src[pos];
    const ct = (first_byte >> 4) & 0x7;

    if (ct == 0) {
        // Type 0: memcpy, 2 or 3 byte header
        if (first_byte >= 0x80) {
            if (pos + 2 > chunk_src.len) return null;
            const sz: u32 = ((@as(u32, chunk_src[pos]) << 8) | @as(u32, chunk_src[pos + 1])) & 0xFFF;
            return pos + 2 + sz;
        } else {
            if (pos + 3 > chunk_src.len) return null;
            const sz: u32 = (@as(u32, chunk_src[pos]) << 16) | (@as(u32, chunk_src[pos + 1]) << 8) | @as(u32, chunk_src[pos + 2]);
            return pos + 3 + sz;
        }
    } else if (ct == 1 or ct == 2 or ct == 4 or ct == 6) {
        // tANS (1) / Huffman (2, 4) / 32-lane tANS (6):
        // 3 or 5 byte header + compressed payload.
        if (first_byte >= 0x80) {
            if (pos + 3 > chunk_src.len) return null;
            const bits: u32 = (@as(u32, chunk_src[pos]) << 16) | (@as(u32, chunk_src[pos + 1]) << 8) | @as(u32, chunk_src[pos + 2]);
            const comp_size = bits & 0x3FF;
            return pos + 3 + comp_size;
        } else {
            if (pos + 5 > chunk_src.len) return null;
            const bits: u32 = (@as(u32, chunk_src[pos + 1]) << 24) | (@as(u32, chunk_src[pos + 2]) << 16) | (@as(u32, chunk_src[pos + 3]) << 8) | @as(u32, chunk_src[pos + 4]);
            const comp_size = bits & 0x3FFFF;
            return pos + 5 + comp_size;
        }
    } else if (ct == 5) {
        // Paired-secondary marker: [0x50][countA:u24][countB:u24] = 7 bytes
        if (pos + 7 > chunk_src.len) return null;
        return pos + 7;
    } else if (ct == 7) {
        // Paired-primary marker: [0x70][countA:u24][embedded type-6 stream]
        if (pos + 4 > chunk_src.len) return null;
        return skipStreamHeader(chunk_src, pos + 4);
    }
    return null;
}

fn parseTansHeaderWithDstOffset(
    chunk_src: []const u8,
    lit_off: u32,
    src_offset_base: u32,
    dst_offset: usize,
    tans_descs_out: []TansDecChunkDesc,
    num_tans: *u32,
) bool {
    if (lit_off >= chunk_src.len) return false;
    const first_byte = chunk_src[lit_off];

    if (first_byte >= 0x80) {
        if (lit_off + 3 > chunk_src.len) return false;
        const hdr3: u32 = (@as(u32, chunk_src[lit_off]) << 16) |
            (@as(u32, chunk_src[lit_off + 1]) << 8) |
            @as(u32, chunk_src[lit_off + 2]);
        const tans_comp_size = hdr3 & 0x3FF;
        const tans_dst_size = tans_comp_size + ((hdr3 >> 10) & 0x3FF) + 1;
        const tans_payload_off = src_offset_base + lit_off + 3;

        if (num_tans.* < tans_descs_out.len) {
            tans_descs_out[num_tans.*] = .{
                .src_offset = @intCast(tans_payload_off),
                .src_size = @intCast(tans_comp_size),
                .dst_offset = @intCast(dst_offset),
                .dst_size = @intCast(tans_dst_size),
                .dst_offset_b = @intCast(dst_offset),
                .split_count = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    } else {
        if (lit_off + 5 > chunk_src.len) return false;
        const bits: u32 = (@as(u32, chunk_src[lit_off + 1]) << 24) |
            (@as(u32, chunk_src[lit_off + 2]) << 16) |
            (@as(u32, chunk_src[lit_off + 3]) << 8) |
            @as(u32, chunk_src[lit_off + 4]);
        const tans_comp_size = bits & 0x3FFFF;
        const tans_dst_size = (((bits >> 18) | (@as(u32, chunk_src[lit_off]) << 14)) & 0x3FFFF) + 1;
        const tans_payload_off = src_offset_base + lit_off + 5;

        if (num_tans.* < tans_descs_out.len) {
            tans_descs_out[num_tans.*] = .{
                .src_offset = @intCast(tans_payload_off),
                .src_size = @intCast(tans_comp_size),
                .dst_offset = @intCast(dst_offset),
                .dst_size = @intCast(tans_dst_size),
                .dst_offset_b = @intCast(dst_offset),
                .split_count = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    }
    return false;
}

fn parseTansHeader(
    chunk_src: []const u8,
    lit_off: u32,
    src_offset_base: u32,
    chunk_idx: usize,
    tans_descs_out: []TansDecChunkDesc,
    num_tans: *u32,
) bool {
    if (lit_off >= chunk_src.len) return false;
    const first_byte = chunk_src[lit_off];

    if (first_byte >= 0x80) {
        // Compact 3-byte header
        if (lit_off + 3 > chunk_src.len) return false;
        const hdr3: u32 = (@as(u32, chunk_src[lit_off]) << 16) |
            (@as(u32, chunk_src[lit_off + 1]) << 8) |
            @as(u32, chunk_src[lit_off + 2]);
        const tans_comp_size = hdr3 & 0x3FF;
        const tans_dst_size = tans_comp_size + ((hdr3 >> 10) & 0x3FF) + 1;
        const tans_payload_off = src_offset_base + lit_off + 3;

        if (num_tans.* < tans_descs_out.len) {
            tans_descs_out[num_tans.*] = .{
                .src_offset = @intCast(tans_payload_off),
                .src_size = @intCast(tans_comp_size),
                .dst_offset = @intCast(chunk_idx * 65536),
                .dst_size = @intCast(tans_dst_size),
                .dst_offset_b = @intCast(chunk_idx * 65536),
                .split_count = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    } else {
        // Non-compact 5-byte header
        if (lit_off + 5 > chunk_src.len) return false;
        const bits: u32 = (@as(u32, chunk_src[lit_off + 1]) << 24) |
            (@as(u32, chunk_src[lit_off + 2]) << 16) |
            (@as(u32, chunk_src[lit_off + 3]) << 8) |
            @as(u32, chunk_src[lit_off + 4]);
        const tans_comp_size = bits & 0x3FFFF;
        const tans_dst_size = (((bits >> 18) | (@as(u32, chunk_src[lit_off]) << 14)) & 0x3FFFF) + 1;
        const tans_payload_off = src_offset_base + lit_off + 5;

        if (num_tans.* < tans_descs_out.len) {
            tans_descs_out[num_tans.*] = .{
                .src_offset = @intCast(tans_payload_off),
                .src_size = @intCast(tans_comp_size),
                .dst_offset = @intCast(chunk_idx * 65536),
                .dst_size = @intCast(tans_dst_size),
                .dst_offset_b = @intCast(chunk_idx * 65536),
                .split_count = @intCast(tans_dst_size),
            };
            num_tans.* += 1;
            return true;
        }
    }
    return false;
}

/// Parse a type-7 paired-primary literal marker:
///   [0x70][countA:u24 BE][embedded type-6 chunk stream]
/// Creates ONE descriptor: symbols [0,countA) → unit A's region (chunk_idx),
/// symbols [countA,countA+countB) → unit B's region (chunk_idx+1).
fn parseTansHeaderPaired(
    chunk_src: []const u8,
    marker_off: u32,
    src_offset_base: u32,
    chunk_idx: usize,
    tans_descs_out: []TansDecChunkDesc,
    num_tans: *u32,
) bool {
    if (marker_off + 4 > chunk_src.len) return false;
    const count_a: u32 = (@as(u32, chunk_src[marker_off + 1]) << 16) |
        (@as(u32, chunk_src[marker_off + 2]) << 8) |
        @as(u32, chunk_src[marker_off + 3]);
    const inner_off = marker_off + 4;
    if (inner_off >= chunk_src.len) return false;
    const inner_first = chunk_src[inner_off];

    var tans_comp_size: u32 = undefined;
    var tans_dst_size: u32 = undefined;
    var tans_payload_off: u32 = undefined;
    if (inner_first >= 0x80) {
        if (inner_off + 3 > chunk_src.len) return false;
        const hdr3: u32 = (@as(u32, chunk_src[inner_off]) << 16) |
            (@as(u32, chunk_src[inner_off + 1]) << 8) |
            @as(u32, chunk_src[inner_off + 2]);
        tans_comp_size = hdr3 & 0x3FF;
        tans_dst_size = tans_comp_size + ((hdr3 >> 10) & 0x3FF) + 1;
        tans_payload_off = src_offset_base + inner_off + 3;
    } else {
        if (inner_off + 5 > chunk_src.len) return false;
        const bits: u32 = (@as(u32, chunk_src[inner_off + 1]) << 24) |
            (@as(u32, chunk_src[inner_off + 2]) << 16) |
            (@as(u32, chunk_src[inner_off + 3]) << 8) |
            @as(u32, chunk_src[inner_off + 4]);
        tans_comp_size = bits & 0x3FFFF;
        tans_dst_size = (((bits >> 18) | (@as(u32, chunk_src[inner_off]) << 14)) & 0x3FFFF) + 1;
        tans_payload_off = src_offset_base + inner_off + 5;
    }

    if (num_tans.* < tans_descs_out.len) {
        tans_descs_out[num_tans.*] = .{
            .src_offset = tans_payload_off,
            .src_size = tans_comp_size,
            .dst_offset = @intCast(chunk_idx * 131072),
            .dst_size = tans_dst_size,
            .dst_offset_b = @intCast((chunk_idx + 1) * 131072),
            .split_count = count_a,
        };
        num_tans.* += 1;
        return true;
    }
    return false;
}

/// Public entry point — delegates to the default context. External callers
/// (streamlz_decoder.zig) keep this exact name + signature.
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
    return fullGpuLaunchImpl(
        &g_default,
        chunk_descs,
        compressed_block,
        dst_full,
        dst_start_off,
        decompressed_size,
        num_groups,
        chunks_per_group,
        sub_chunk_cap,
        io,
    );
}

pub fn fullGpuLaunchImpl(
    self: *DecodeContext,
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

    // SLZ_E2E_TIMER: end-to-end decode phase breakdown — setup+H2D /
    // host scan+prep / kernels / D2H. Off by default.
    const e2e_timer = std.c.getenv("SLZ_E2E_TIMER") != null;
    const t_e2e0 = if (e2e_timer)
        (if (io) |iv| std.Io.Clock.awake.now(iv) else null)
    else
        null;
    var e2e_cum_h2d_ns: i64 = 0;
    var e2e_cum_scan_ns: i64 = 0;
    var e2e_cum_predh_ns: i64 = 0;

    const h2d_fn = cuMemcpyHtoD_fn orelse return error.BadMode;
    const d2h_fn = cuMemcpyDtoH_fn orelse return error.BadMode;
    const launch_fn = cuLaunchKernel_fn orelse return error.BadMode;
    const sync_fn = cuCtxSynchronize_fn orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    if (!ensureDeviceOutput(self, total_output + 64)) return error.BadMode;

    if (dst_start_off > 0)
        _ = h2d_fn(self.d_output, @ptrCast(dst_full), dst_start_off);

    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    // +16 slack: slzTans32DecodeKernel's readLE32_aligned over-reads up to 7B.
    if (!ensureDeviceBuf(&self.d_comp_persist, &self.d_comp_persist_size, comp_bytes + 16)) return error.BadMode;
    if (!ensureDeviceBuf(&self.d_descs_persist, &self.d_descs_persist_size, desc_bytes)) return error.BadMode;

    // Compute per-chunk first-subchunk index (prefix sum of n_subs_per_chunk).
    // At sc>0.5 chunks have multiple sub-chunks; each needs its own scratch slot.
    var first_subchunk_idx_buf: [16384]u32 = undefined;
    var total_subchunks: u32 = 0;
    {
        const cap: u32 = sub_chunk_cap;
        const cap_safe: u32 = if (cap == 0) 65536 else cap;
        for (chunk_descs, 0..) |ch, i| {
            if (i >= first_subchunk_idx_buf.len) break;
            first_subchunk_idx_buf[i] = total_subchunks;
            // For non-LZ chunks (uncompressed/memset) reserve 1 slot for indexing
            const n_subs: u32 = if (ch.flags != 0 or ch.decomp_size == 0) 1
                else (ch.decomp_size + cap_safe - 1) / cap_safe;
            total_subchunks += n_subs;
        }
    }
    // Allocate scratch sized for total sub-chunks across all stream types.
    // Slot size = 131072 (sub_chunk_cap) to fit the largest sub-chunk's lit/tok
    // streams; off16-hi at +0, off16-lo at +65536 within each slot.
    // Layout: [lit: total*128K] [tok: total*128K] [off16: total*128K]
    const per_subchunk_scratch: usize = 131072;
    const tans_scratch_bytes = @as(usize, total_subchunks) * per_subchunk_scratch * 3;
    if (!ensureDeviceBuf(&self.d_tans_scratch, &self.d_tans_scratch_size, tans_scratch_bytes)) return error.BadMode;
    const tok_offset = @as(usize, total_subchunks) * per_subchunk_scratch;
    const off16_offset = @as(usize, total_subchunks) * per_subchunk_scratch * 2;
    self.d_tans_tok_scratch = self.d_tans_scratch + tok_offset;
    self.d_tans_off16_scratch = self.d_tans_scratch + off16_offset;

    // Upload first_subchunk_idx array if multi-sub-chunk (otherwise pass nullptr)
    const need_first_sub_idx = total_subchunks != @as(u32, @intCast(chunk_descs.len));
    if (need_first_sub_idx) {
        const fs_bytes: usize = chunk_descs.len * @sizeOf(u32);
        if (!ensureDeviceBuf(&self.d_first_subchunk_idx, &self.d_first_subchunk_idx_size, fs_bytes)) return error.BadMode;
        _ = h2d_fn(self.d_first_subchunk_idx, @ptrCast(&first_subchunk_idx_buf), fs_bytes);
    } else {
        self.d_first_subchunk_idx = 0;
    }

    // Global LUT buffer: max_descs * 2048 entries * 8 bytes
    // At sc>=1.0 each chunk has multiple sub-chunks; size by total sub-chunks.
    const max_tans_descs = @as(usize, total_subchunks) * 4; // up to 4 streams per sub-chunk
    const lut_bytes = max_tans_descs * 2048 * 8;
    if (!ensureDeviceBuf(&self.d_tans_lut, &self.d_tans_lut_size, lut_bytes)) return error.BadMode;

    if (compressed_block.len > 0)
        _ = h2d_fn(self.d_comp_persist, @ptrCast(compressed_block.ptr), compressed_block.len);
    _ = h2d_fn(self.d_descs_persist, @ptrCast(chunk_descs.ptr), desc_bytes);
    _ = sync_fn();
    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum_h2d_ns = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };

    var e2e_cum_prescan_ns: i64 = 0;
    var e2e_cum_postscan_ns: i64 = 0;
    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum_prescan_ns = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };

    // ── Scan for tANS chunks ──────────────────────────────────
    var merged_count: u32 = 0;
    var scan: ScanResult = .{ .num_lit = 0, .num_tok = 0, .num_off16hi = 0, .num_off16lo = 0, .num_raw_off16 = 0 };

    if (tans_kernel_fn != 0) {
        // Cap by total_subchunks (each sub-chunk can contribute one descriptor per
        // stream type), not chunk_descs.len, because at sc>=1.0 a chunk has
        // multiple sub-chunks and each gets its own tANS stream.
        const max_tans = @min(@as(usize, total_subchunks), self.tans_host_buf.len);

        scan = scanForTansChunks(
            chunk_descs,
            compressed_block,
            sub_chunk_cap,
            self.tans_host_buf[0..max_tans],
            self.tans_tok_host_buf[0..max_tans],
            self.tans_off16hi_host_buf[0..max_tans],
            self.tans_off16lo_host_buf[0..max_tans],
            &self.raw_off16_buf,
            &self.huff_lit_host_buf,
            &self.huff_tok_host_buf,
            &self.huff_off16hi_host_buf,
            &self.huff_off16lo_host_buf,
            io,
        );

        merged_count = scan.num_lit + scan.num_tok + scan.num_off16hi + scan.num_off16lo;

        if (t_e2e0) |t0| if (io) |iv| {
            e2e_cum_postscan_ns = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
        };

        // Place raw (type 0) off16 sub-streams into the off16 scratch.
        // The bytes are already on the GPU (d_comp_persist holds the whole
        // compressed blob). Preferred: upload the descriptor list in one
        // H2D and run slzGatherRawOff16Kernel — one launch copies every
        // stream in parallel. Fallbacks: async device-to-device loop, then
        // a plain host-upload loop.
        if (scan.num_raw_off16 > 0) gather_blk: {
            if (gather_off16_fn != 0) {
                const ndesc: u32 = scan.num_raw_off16;
                const dbytes: usize = @as(usize, ndesc) * @sizeOf(RawOff16Desc);
                if (ensureDeviceBuf(&self.d_raw_off16_descs, &self.d_raw_off16_descs_size, dbytes)) {
                    _ = h2d_fn(self.d_raw_off16_descs, @ptrCast(&self.raw_off16_buf), dbytes);
                    var p_comp = self.d_comp_persist;
                    var p_comp_len: u32 = @intCast(compressed_block.len);
                    var p_scratch = self.d_tans_off16_scratch;
                    var p_descs = self.d_raw_off16_descs;
                    var p_count = ndesc;
                    var params = [_]?*anyopaque{
                        @ptrCast(&p_comp), @ptrCast(&p_comp_len), @ptrCast(&p_scratch),
                        @ptrCast(&p_descs), @ptrCast(&p_count),
                    };
                    var extra = [_]?*anyopaque{null};
                    if (launch_fn(gather_off16_fn, ndesc, 1, 1, 256, 1, 1, 0, 0, &params, &extra) == CUDA_SUCCESS) {
                        _ = sync_fn();
                        break :gather_blk;
                    }
                }
            }
            // Fallback: async device-to-device, else host upload.
            if (cuMemcpyDtoDAsync_fn) |d2d| {
                for (0..scan.num_raw_off16) |ri| {
                    const rd = self.raw_off16_buf[ri];
                    if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len)
                        _ = d2d(self.d_tans_off16_scratch + rd.gpu_offset, self.d_comp_persist + rd.src_offset, rd.size, 0);
                }
                _ = sync_fn();
            } else {
                for (0..scan.num_raw_off16) |ri| {
                    const rd = self.raw_off16_buf[ri];
                    if (rd.size > 0 and rd.src_offset + rd.size <= compressed_block.len)
                        _ = h2d_fn(self.d_tans_off16_scratch + rd.gpu_offset, @ptrCast(compressed_block.ptr + rd.src_offset), rd.size);
                }
            }
        }
    }

    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum_scan_ns = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };

    // ── Huffman pre-decode (Pass 1.5): merge per-stream-type descriptors
    // into a single device array with correct out_offsets, then upload.
    // Scanner left out_offsets in sub-chunk-slot space; here we add the
    // per-stream-type region offset (tok_offset, off16_offset) and assign
    // sequential lut_offsets.
    const n_huff: u32 = scan.num_huff_lit + scan.num_huff_tok +
        scan.num_huff_off16hi + scan.num_huff_off16lo;
    if (std.c.getenv("SLZ_HUFF_DBG") != null) {
        std.debug.print("huff scan: lit={d} tok={d} hi={d} lo={d} total={d}\n", .{ scan.num_huff_lit, scan.num_huff_tok, scan.num_huff_off16hi, scan.num_huff_off16lo, n_huff });
        std.debug.print("tans scan: lit={d} tok={d} hi={d} lo={d}\n", .{ scan.num_lit, scan.num_tok, scan.num_off16hi, scan.num_off16lo });
    }
    const have_huff = n_huff > 0 and huff_build_fn != 0 and huff_decode_fn != 0;
    if (have_huff) {
        // Merge into a host scratch then upload. Capacity = 4 buffers × 4096.
        var merged_huff: [4096 * 4]HuffDecChunkDesc = undefined;
        var m: u32 = 0;
        var lut_slot: u32 = 0;
        const append = struct {
            fn run(dst: []HuffDecChunkDesc, m_ptr: *u32, lut_ptr: *u32,
                   src: []const HuffDecChunkDesc, region_off: u32) void {
                for (src) |s| {
                    if (m_ptr.* >= dst.len) return;
                    var d = s;
                    d.out_offset += region_off;
                    d.lut_offset = lut_ptr.* * @as(u32, @intCast(HUFF_LUT_ENTRIES));
                    dst[m_ptr.*] = d;
                    m_ptr.* += 1;
                    lut_ptr.* += 1;
                }
            }
        }.run;
        append(&merged_huff, &m, &lut_slot, self.huff_lit_host_buf[0..scan.num_huff_lit], 0);
        append(&merged_huff, &m, &lut_slot, self.huff_tok_host_buf[0..scan.num_huff_tok], @intCast(tok_offset));
        append(&merged_huff, &m, &lut_slot, self.huff_off16hi_host_buf[0..scan.num_huff_off16hi], @intCast(off16_offset));
        append(&merged_huff, &m, &lut_slot, self.huff_off16lo_host_buf[0..scan.num_huff_off16lo], @intCast(off16_offset));

        const huff_desc_bytes = @as(usize, m) * @sizeOf(HuffDecChunkDesc);
        const huff_lut_bytes = @as(usize, m) * HUFF_LUT_ENTRIES * @sizeOf(u32);
        if (!ensureDeviceBuf(&self.d_huff_descs, &self.d_huff_descs_size, huff_desc_bytes)) return error.BadMode;
        if (!ensureDeviceBuf(&self.d_huff_lut, &self.d_huff_lut_size, huff_lut_bytes)) return error.BadMode;
        _ = h2d_fn(self.d_huff_descs, @ptrCast(&merged_huff), huff_desc_bytes);
        _ = sync_fn();
    }

    // ── Pipeline: split into N groups, overlap tANS with LZ ───
    const total_chunks: u32 = @intCast(chunk_descs.len);
    const use_pipeline = self.pipeline_streams_created and tans_kernel_fn != 0 and tans_build_fn != 0 and total_chunks >= NUM_PIPELINE_STREAMS;
    if (use_pipeline) {
        // Merge tANS descriptors grouped by pipeline stage
        // Pipeline group g handles chunks [g*pipe_chunk_count .. (g+1)*pipe_chunk_count)
        const pipe_chunk_count = (total_chunks + NUM_PIPELINE_STREAMS - 1) / NUM_PIPELINE_STREAMS;

        var merged_buf: [4096 * 4]TansDecChunkDesc = undefined;
        var group_tans_start: [16]u32 = .{0} ** 16;
        var group_tans_count: [16]u32 = .{0} ** 16;
        merged_count = 0;

        // For each pipeline group, collect all tANS descriptors belonging to its chunks.
        // Descriptor dst_offset is in sub-chunk space (sub_idx * per_subchunk_scratch),
        // so convert pipeline group boundaries (chunk-space) to sub-chunk-space.
        for (0..NUM_PIPELINE_STREAMS) |g| {
            const chunk_start = @as(u32, @intCast(g)) * pipe_chunk_count;
            const chunk_end = @min(chunk_start + pipe_chunk_count, total_chunks);
            const sub_start: u32 = if (chunk_start < first_subchunk_idx_buf.len and chunk_start < total_chunks)
                first_subchunk_idx_buf[chunk_start]
            else
                total_subchunks;
            const sub_end: u32 = if (chunk_end < first_subchunk_idx_buf.len and chunk_end < total_chunks)
                first_subchunk_idx_buf[chunk_end]
            else
                total_subchunks;
            group_tans_start[g] = merged_count;

            const slot: u32 = @intCast(per_subchunk_scratch);
            // Lit descriptors: dst_offset = sub_idx * slot
            for (0..scan.num_lit) |i| {
                const sidx: u32 = @intCast(self.tans_host_buf[i].dst_offset / slot);
                if (sidx >= sub_start and sidx < sub_end) {
                    var d = self.tans_host_buf[i];
                    // In per-stream (legacy) mode lut_id == chunk_id ==
                    // descriptor index. In shared mode the scan already set
                    // lut_id to one of {0..3}; preserve it.
                    if (d.lut_id >= 4) d.lut_id = merged_count - group_tans_start[g];
                    merged_buf[merged_count] = d;
                    merged_count += 1;
                }
            }
            // Tok descriptors: dst_offset = sub_idx * slot (before tok_offset add)
            for (0..scan.num_tok) |i| {
                const sidx: u32 = @intCast(self.tans_tok_host_buf[i].dst_offset / slot);
                if (sidx >= sub_start and sidx < sub_end) {
                    var d = self.tans_tok_host_buf[i];
                    d.dst_offset += @intCast(tok_offset);
                    d.dst_offset_b += @intCast(tok_offset);
                    if (d.lut_id >= 4) d.lut_id = merged_count - group_tans_start[g];
                    merged_buf[merged_count] = d;
                    merged_count += 1;
                }
            }
            // Off16hi descriptors: dst_offset = sub_idx * slot
            for (0..scan.num_off16hi) |i| {
                const sidx: u32 = @intCast(self.tans_off16hi_host_buf[i].dst_offset / slot);
                if (sidx >= sub_start and sidx < sub_end) {
                    var d = self.tans_off16hi_host_buf[i];
                    d.dst_offset += @intCast(off16_offset);
                    d.dst_offset_b += @intCast(off16_offset);
                    if (d.lut_id >= 4) d.lut_id = merged_count - group_tans_start[g];
                    merged_buf[merged_count] = d;
                    merged_count += 1;
                }
            }
            // Off16lo descriptors: dst_offset = sub_idx * slot + 65536
            for (0..scan.num_off16lo) |i| {
                const sidx: u32 = @intCast((self.tans_off16lo_host_buf[i].dst_offset -| 65536) / slot);
                if (sidx >= sub_start and sidx < sub_end) {
                    var d = self.tans_off16lo_host_buf[i];
                    d.dst_offset += @intCast(off16_offset);
                    d.dst_offset_b += @intCast(off16_offset);
                    if (d.lut_id >= 4) d.lut_id = merged_count - group_tans_start[g];
                    merged_buf[merged_count] = d;
                    merged_count += 1;
                }
            }

            group_tans_count[g] = merged_count - group_tans_start[g];
        }

        // Upload ALL merged tANS descriptors at once (single H2D before timer)
        if (merged_count > 0) {
            const mdesc_bytes = merged_count * @sizeOf(TansDecChunkDesc);
            const mstatus_bytes = merged_count * @sizeOf(u32);
            if (!ensureDeviceBuf(&self.d_tans_descs_persist, &self.d_tans_descs_persist_size, mdesc_bytes))
                return error.BadMode;
            if (!ensureDeviceBuf(&self.d_tans_status_persist, &self.d_tans_status_persist_size, mstatus_bytes))
                return error.BadMode;
            _ = h2d_fn(self.d_tans_descs_persist, @ptrCast(@as([*]TansDecChunkDesc, &merged_buf)), mdesc_bytes);
            if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(self.d_tans_status_persist, 0, mstatus_bytes);
        }

        // Allocate meta buffer for all descriptors
        if (merged_count > 0) {
            const meta_bytes = @as(usize, merged_count) * 16;
            if (!ensureDeviceBuf(&self.d_tans_meta, &self.d_tans_meta_size, meta_bytes)) return error.BadMode;
        }

        // Allocate build timing buffer if SLZ_BUILD_TIMING env var is set
        const want_timing = std.c.getenv("SLZ_BUILD_TIMING") != null;
        if (want_timing and merged_count > 0) {
            const timing_bytes = @as(usize, merged_count) * 16; // 4 x u32 per stream
            if (!ensureDeviceBuf(&self.d_build_timing, &self.d_build_timing_size, timing_bytes)) return error.BadMode;
        } else {
            self.d_build_timing = 0; // nullptr = no timing
        }

        _ = sync_fn();

        // ── KERNEL TIMER: only pure GPU kernel time from here ──
        const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

        // SLZ_SPLIT_TIMER hoisted above the Huff launches so we can fence
        // around them too. Without split, Huff time gets pipelined into LZ.
        const split_timer = std.c.getenv("SLZ_SPLIT_TIMER") != null;

        // Launch Huffman pre-decode (Pass 1.5). On split_timer we sync
        // after so the Huff time is attributed separately.
        const t_huff_start = if (split_timer and have_huff)
            if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
        else
            null;
        // SLZ_SPLIT_TIMER: build-vs-decode breakdown of the Huff pass.
        var split_huff_build_ns: i64 = 0;
        if (have_huff) {
            const huff_stream = self.pipeline_streams[0];
            {
                var p_comp = self.d_comp_persist;
                var p_descs = self.d_huff_descs;
                var p_lut = self.d_huff_lut;
                var p_n = n_huff;
                var params = [_]?*anyopaque{
                    @ptrCast(&p_comp), @ptrCast(&p_descs),
                    @ptrCast(&p_lut), @ptrCast(&p_n),
                };
                var extra = [_]?*anyopaque{null};
                if (launch_fn(huff_build_fn, n_huff, 1, 1, 32, 1, 1, 0, huff_stream, &params, &extra) != CUDA_SUCCESS)
                    return error.BadMode;
            }
            // Split fence: time the LUT build separately from the decode.
            if (split_timer) {
                if (cuStreamSync_fn) |sf| _ = sf(huff_stream);
                if (t_huff_start) |hs| {
                    if (io) |io_val|
                        split_huff_build_ns = @intCast(hs.untilNow(io_val, .awake).toNanoseconds());
                }
            }
            {
                var p_comp = self.d_comp_persist;
                var p_descs = self.d_huff_descs;
                var p_lut = self.d_huff_lut;
                var p_out = self.d_tans_scratch;
                var p_n = n_huff;
                var params = [_]?*anyopaque{
                    @ptrCast(&p_comp), @ptrCast(&p_descs),
                    @ptrCast(&p_lut), @ptrCast(&p_out), @ptrCast(&p_n),
                };
                var extra = [_]?*anyopaque{null};
                const shared_bytes: c_uint = HUFF_LUT_ENTRIES * @sizeOf(u32);
                if (launch_fn(huff_decode_fn, n_huff, 1, 1, 32, 1, 1, shared_bytes, huff_stream, &params, &extra) != CUDA_SUCCESS)
                    return error.BadMode;
            }
        }

        // Launch pipelined groups: for each group, launch tANS build → tANS decode → LZ in its own stream
        // Operations within a stream are ordered; across streams they can overlap
        const stream_sync_fn = cuStreamSync_fn orelse return error.BadMode;

        var split_tans_ns: i64 = 0;
        var split_lz_ns: i64 = 0;
        var split_huff_ns: i64 = 0;

        // Close Huff time slice now: sync the pipeline stream so the next
        // group's tANS_start measurement excludes Huff time.
        if (split_timer and have_huff) {
            _ = stream_sync_fn(self.pipeline_streams[0]);
            if (t_huff_start) |hs| {
                if (io) |io_val| {
                    split_huff_ns = @intCast(hs.untilNow(io_val, .awake).toNanoseconds());
                }
            }
            std.debug.print("  [huff split] build {d:.3} ms  decode {d:.3} ms\n", .{
                @as(f64, @floatFromInt(split_huff_build_ns)) / 1e6,
                @as(f64, @floatFromInt(split_huff_ns - split_huff_build_ns)) / 1e6,
            });
        }

        for (0..NUM_PIPELINE_STREAMS) |g| {
            const stream = self.pipeline_streams[g];
            const chunk_start = @as(u32, @intCast(g)) * pipe_chunk_count;
            const chunk_end = @min(chunk_start + pipe_chunk_count, total_chunks);
            const group_chunks = chunk_end - chunk_start;
            if (group_chunks == 0) continue;

            const t_tans_start = if (split_timer)
                if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
            else
                null;

            // tANS build + decode for this group's descriptors
            const tc = group_tans_count[g];
            if (tc > 0) {
                const ts = group_tans_start[g];
                const tans_grid = (tc + 1) / 2;

                // Pointers into the uploaded arrays at group offset
                var bp_comp = self.d_comp_persist;
                var bp_descs = self.d_tans_descs_persist + @as(u64, ts) * @sizeOf(TansDecChunkDesc);
                var bp_lut = self.d_tans_lut + @as(u64, ts) * 2048 * 8;
                var bp_meta = self.d_tans_meta + @as(u64, ts) * 16;
                var bp_num = tc;

                // Use split parse+initlut kernels when available, else legacy combined.
                if (scan.use_tans32 and tans_fse_build_fn != 0 and tans32_kernel_fn != 0) {
                    // FSE build kernel with atomicAdd work claiming
                    if (!ensureDeviceBuf(&self.d_work_counter, &self.d_work_counter_size, 4)) return error.BadMode;
                    if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(self.d_work_counter, 0, 4);
                    var bp_wc = self.d_work_counter;
                    var fse_build_params = [_]?*anyopaque{
                        @ptrCast(&bp_comp), @ptrCast(&bp_descs),
                        @ptrCast(&bp_lut), @ptrCast(&bp_meta), @ptrCast(&bp_num),
                        @ptrCast(&bp_wc),
                    };
                    var fse_build_extra = [_]?*anyopaque{null};
                    // Persistent grid: fill GPU (24 blocks/SM × 34 SMs ≈ 816)
                    const fse_grid = @min(tans_grid, 1024);
                    if (launch_fn(tans_fse_build_fn, fse_grid, 1, 1, 32, 2, 1, 0, stream, &fse_build_params, &fse_build_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                    // Sync to check FSE build kernel for errors
                    const fse_sync = stream_sync_fn(stream);
                    if (fse_sync != CUDA_SUCCESS) {
                        std.debug.print("FSE BUILD kernel FAILED rc={d}\n", .{fse_sync});
                        return error.BadMode;
                    }

                    // Then 32-lane decode (same as before)
                    var tp_comp2 = self.d_comp_persist;
                    var tp_scratch2 = self.d_tans_scratch;
                    var tp_descs2 = bp_descs;
                    var tp_status2 = self.d_tans_status_persist + @as(u64, ts) * @sizeOf(u32);
                    var tp_num2 = tc;
                    var tp_lut2 = bp_lut;
                    var tp_meta2 = bp_meta;
                    var tp_shared: u32 = 0; // per-stream LUT mode
                    var tans32_params = [_]?*anyopaque{
                        @ptrCast(&tp_comp2), @ptrCast(&tp_scratch2),
                        @ptrCast(&tp_descs2), @ptrCast(&tp_status2), @ptrCast(&tp_num2),
                        @ptrCast(&tp_lut2), @ptrCast(&tp_meta2),
                        @ptrCast(&tp_shared),
                    };
                    var tans32_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans32_kernel_fn, tans_grid, 1, 1, 32, 2, 1, 0, stream, &tans32_params, &tans32_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                } else if (scan.use_tans32 and tans_parse_fn != 0 and tans_initlut_fn != 0) {
                    // TansParsedWeights = TansData(1288B) + u32(4B) = 1292B per stream
                    const weights_bytes = @as(usize, tc) * 1296; // round up to 16B alignment
                    if (!ensureDeviceBuf(&self.d_parsed_weights, &self.d_parsed_weights_size, weights_bytes))
                        return error.BadMode;
                    var bp_weights = self.d_parsed_weights;

                    // Kernel A: parse tables (lane 0 only)
                    var parse_params = [_]?*anyopaque{
                        @ptrCast(&bp_comp), @ptrCast(&bp_descs),
                        @ptrCast(&bp_weights), @ptrCast(&bp_meta), @ptrCast(&bp_num),
                    };
                    var parse_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans_parse_fn, tans_grid, 1, 1, 32, 2, 1, 0, stream, &parse_params, &parse_extra) != CUDA_SUCCESS)
                        return error.BadMode;

                    // Kernel B: build LUTs (all 32 lanes, fully parallel)
                    var lut_params = [_]?*anyopaque{
                        @ptrCast(&bp_weights), @ptrCast(&bp_meta),
                        @ptrCast(&bp_lut), @ptrCast(&bp_num),
                    };
                    var lut_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans_initlut_fn, tans_grid, 1, 1, 32, 2, 1, 0, stream, &lut_params, &lut_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                } else {
                    // Legacy combined build kernel
                    var bp_timing: CUdeviceptr = self.d_build_timing;
                    var build_params = [_]?*anyopaque{
                        @ptrCast(&bp_comp), @ptrCast(&bp_descs),
                        @ptrCast(&bp_lut), @ptrCast(&bp_meta), @ptrCast(&bp_num),
                        @ptrCast(&bp_timing),
                    };
                    var build_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans_build_fn, tans_grid, 1, 1, 32, 2, 1, 0, stream, &build_params, &build_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                }

                if (scan.use_tans32 and tans32_kernel_fn != 0) {
                    // 32-lane tANS decode with per-stream LUT from build kernel
                    var tp_comp = self.d_comp_persist;
                    var tp_scratch = self.d_tans_scratch;
                    var tp_descs = bp_descs;
                    var tp_status = self.d_tans_status_persist + @as(u64, ts) * @sizeOf(u32);
                    var tp_num = tc;
                    var tp_lut = bp_lut;
                    var tp_meta = bp_meta;
                    var tp_shared: u32 = 0; // per-stream LUT mode
                    var tans32_params = [_]?*anyopaque{
                        @ptrCast(&tp_comp), @ptrCast(&tp_scratch),
                        @ptrCast(&tp_descs), @ptrCast(&tp_status), @ptrCast(&tp_num),
                        @ptrCast(&tp_lut), @ptrCast(&tp_meta),
                        @ptrCast(&tp_shared),
                    };
                    var tans32_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans32_kernel_fn, tans_grid, 1, 1, 32, 2, 1, 0, stream, &tans32_params, &tans32_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                } else {
                    // 5-state tANS decode (build kernel already launched above)
                    var tp_comp = self.d_comp_persist;
                    var tp_scratch = self.d_tans_scratch;
                    var tp_descs = bp_descs;
                    var tp_status = self.d_tans_status_persist + @as(u64, ts) * @sizeOf(u32);
                    var tp_num = tc;
                    var tp_lut = bp_lut;
                    var tp_meta = bp_meta;
                    var tans_params = [_]?*anyopaque{
                        @ptrCast(&tp_comp), @ptrCast(&tp_scratch),
                        @ptrCast(&tp_descs), @ptrCast(&tp_status), @ptrCast(&tp_num),
                        @ptrCast(&tp_lut), @ptrCast(&tp_meta),
                    };
                    var tans_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans_kernel_fn, tans_grid, 1, 1, 32, 2, 1, 0, stream, &tans_params, &tans_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                }

                // ── Debug: one-shot tans32 snapshot (SLZ_DUMP_TANS32=1) ──
                // Dumps the exact (src_buf, descs, decoded output) the
                // tans32 decode consumed/produced, for the standalone
                // testbed in tools/huff_test/. Inert without the env var.
                if (scan.use_tans32 and !self.tans32_dumped and std.c.getenv("SLZ_DUMP_TANS32") != null) {
                    self.tans32_dumped = true;
                    _ = sync_fn();
                    dumpTans32: {
                        const cio = @cImport({ @cInclude("stdio.h"); });
                        const a = std.heap.page_allocator;
                        const dbytes: usize = @as(usize, tc) * @sizeOf(TansDecChunkDesc);
                        const dhost = a.alloc(u8, dbytes) catch break :dumpTans32;
                        defer a.free(dhost);
                        const chost = a.alloc(u8, comp_bytes) catch break :dumpTans32;
                        defer a.free(chost);
                        const shost = a.alloc(u8, self.d_tans_scratch_size) catch break :dumpTans32;
                        defer a.free(shost);
                        _ = d2h_fn(dhost.ptr, bp_descs, dbytes);
                        _ = d2h_fn(chost.ptr, self.d_comp_persist, comp_bytes);
                        _ = d2h_fn(shost.ptr, self.d_tans_scratch, self.d_tans_scratch_size);
                        const fp = cio.fopen("c:/tmp/tans32_snapshot.bin", "wb") orelse break :dumpTans32;
                        var hdr = [_]u32{ 0x544e3332, tc, @intCast(comp_bytes), @intCast(self.d_tans_scratch_size) };
                        _ = cio.fwrite(&hdr, 4, 4, fp);
                        _ = cio.fwrite(dhost.ptr, 1, dbytes, fp);
                        _ = cio.fwrite(chost.ptr, 1, comp_bytes, fp);
                        _ = cio.fwrite(shost.ptr, 1, self.d_tans_scratch_size, fp);
                        _ = cio.fclose(fp);
                    }
                }
            }

            // Split timer: sync after tANS so we can attribute time to Pass 1.
            if (split_timer) {
                _ = stream_sync_fn(stream);
                if (t_tans_start) |ts_start| {
                    if (io) |io_val| {
                        split_tans_ns += @intCast(ts_start.untilNow(io_val, .awake).toNanoseconds());
                    }
                }
            }

            const t_lz_start = if (split_timer)
                if (io) |io_val| std.Io.Clock.awake.now(io_val) else null
            else
                null;

            // LZ decode for this group's chunks
            // chunks pointer = full array (NOT offset), chunk_base = chunk_start
            // total = chunk_end so the kernel's bounds check (chunk_idx >= total) works
            const lz_groups_in_pipe = (group_chunks + chunks_per_group - 1) / chunks_per_group;
            const lz_grid_x = (lz_groups_in_pipe + 1) / 2;

            // Fast path: no entropy in this scan → use lean L1/L2 raw kernel.
            // Huffman literals also require the general kernel (it reads tans_scratch).
            const use_raw_kernel = merged_count == 0 and n_huff == 0 and kernel_raw_fn != 0;

            if (use_raw_kernel) {
                var p_comp = self.d_comp_persist;
                var p_descs_dev = self.d_descs_persist + @as(u64, chunk_start) * @sizeOf(ChunkDesc);
                var p_dst = self.d_output;
                var p_cpg = chunks_per_group;
                var p_total = chunk_end - chunk_start;
                var p_sc_cap = sub_chunk_cap;
                var raw_params = [_]?*anyopaque{
                    @ptrCast(&p_comp),
                    @ptrCast(&p_descs_dev),
                    @ptrCast(&p_dst),
                    @ptrCast(&p_cpg),
                    @ptrCast(&p_total),
                    @ptrCast(&p_sc_cap),
                };
                var raw_extra = [_]?*anyopaque{null};
                if (launch_fn(kernel_raw_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &raw_params, &raw_extra) != CUDA_SUCCESS)
                    return error.BadMode;
            } else {
                // Drop chunk_base / tans_tok_scratch / tans_off16_scratch
                // from the kernel signature — kernel derives tok/off16 from
                // tans_scratch + slot_stride, and we pre-shift the desc
                // pointer here. Net 11 → 9 params at the kernel.
                var p_comp = self.d_comp_persist;
                var p_descs_dev = self.d_descs_persist + @as(u64, chunk_start) * @sizeOf(ChunkDesc);
                var p_dst = self.d_output;
                var p_cpg = chunks_per_group;
                var p_total = chunk_end - chunk_start;
                var p_sc_cap = sub_chunk_cap;
                var p_tans_scratch = self.d_tans_scratch;
                var p_tans_slot_stride: u64 = @as(u64, total_subchunks) * 131072;
                var p_first_sub_idx: CUdeviceptr = self.d_first_subchunk_idx +
                    if (self.d_first_subchunk_idx != 0) @as(u64, chunk_start) * @sizeOf(u32) else 0;

                var lz_params = [_]?*anyopaque{
                    @ptrCast(&p_comp),
                    @ptrCast(&p_descs_dev),
                    @ptrCast(&p_dst),
                    @ptrCast(&p_cpg),
                    @ptrCast(&p_total),
                    @ptrCast(&p_sc_cap),
                    @ptrCast(&p_tans_scratch),
                    @ptrCast(&p_tans_slot_stride),
                    @ptrCast(&p_first_sub_idx),
                };
                var lz_extra = [_]?*anyopaque{null};

                if (launch_fn(kernel_fn, lz_grid_x, 1, 1, 32, 2, 1, 0, stream, &lz_params, &lz_extra) != CUDA_SUCCESS)
                    return error.BadMode;
            }

            if (split_timer) {
                _ = stream_sync_fn(stream);
                if (t_lz_start) |ts_start| {
                    if (io) |io_val| {
                        split_lz_ns += @intCast(ts_start.untilNow(io_val, .awake).toNanoseconds());
                    }
                }
            }
        }

        // Sync all pipeline streams
        for (0..NUM_PIPELINE_STREAMS) |g| {
            const sync_rc = stream_sync_fn(self.pipeline_streams[g]);
            if (sync_rc != CUDA_SUCCESS) {
                std.debug.print("GPU pipe[{d}]: stream sync FAILED rc={d}\n", .{ g, sync_rc });
                return error.BadMode;
            }
        }

        // Read back build timing if requested
        if (want_timing and self.d_build_timing != 0 and merged_count > 0) {
            const n_read = @min(@as(usize, merged_count), 8192);
            var timing_static: [8192 * 4]u32 = undefined;
            const th = timing_static[0 .. n_read * 4];
            {
                _ = d2h_fn(@ptrCast(th.ptr), self.d_build_timing, n_read * 16);
                _ = sync_fn();

                var sum_parse: u64 = 0;
                var sum_initlut: u64 = 0;
                var sum_pack: u64 = 0;
                var sum_total: u64 = 0;
                var count: u32 = 0;
                for (0..n_read) |i| {
                    const parse = th[i * 4 + 0];
                    const initlut = th[i * 4 + 1];
                    const pack = th[i * 4 + 2];
                    const total = th[i * 4 + 3];
                    if (total > 0) {
                        sum_parse += parse;
                        sum_initlut += initlut;
                        sum_pack += pack;
                        sum_total += total;
                        count += 1;
                    }
                }
                if (count > 0) {
                    const cf: f64 = @floatFromInt(count);
                    const freq: f64 = 2230.0; // SM freq MHz from profile
                    std.debug.print("BUILD TIMING ({d} streams):\n", .{count});
                    std.debug.print("  parseTable:  avg {d:.0} cycles ({d:.2} us)\n", .{ @as(f64, @floatFromInt(sum_parse)) / cf, @as(f64, @floatFromInt(sum_parse)) / cf / freq });
                    std.debug.print("  initLut:     avg {d:.0} cycles ({d:.2} us)\n", .{ @as(f64, @floatFromInt(sum_initlut)) / cf, @as(f64, @floatFromInt(sum_initlut)) / cf / freq });
                    std.debug.print("  pack:        avg {d:.0} cycles ({d:.2} us)\n", .{ @as(f64, @floatFromInt(sum_pack)) / cf, @as(f64, @floatFromInt(sum_pack)) / cf / freq });
                    std.debug.print("  total:       avg {d:.0} cycles ({d:.2} us)\n", .{ @as(f64, @floatFromInt(sum_total)) / cf, @as(f64, @floatFromInt(sum_total)) / cf / freq });
                }
            }
        }

        if (t_before_kern) |t_start| {
            if (io) |io_val| {
                last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
                if (split_timer) {
                    last_tans_kernel_ns = split_tans_ns;
                    last_lz_kernel_ns = split_lz_ns;
                    last_huff_kernel_ns = split_huff_ns;
                } else {
                    last_tans_kernel_ns = last_kernel_ns; // pipelined, report combined time
                    last_lz_kernel_ns = 0;
                    last_huff_kernel_ns = 0;
                }
            }
        }
    } else {
        // ── Non-pipelined fallback (original sequential path) ─────
        if (tans_kernel_fn != 0) {
            // Merge all tANS descriptors into one array, adjust offsets for unified scratch
            var merged_buf: [4096 * 4]TansDecChunkDesc = undefined;
            merged_count = 0;

            for (0..scan.num_lit) |i| {
                merged_buf[merged_count] = self.tans_host_buf[i];
                merged_count += 1;
            }
            for (0..scan.num_tok) |i| {
                var d = self.tans_tok_host_buf[i];
                d.dst_offset += @intCast(tok_offset);
                merged_buf[merged_count] = d;
                merged_count += 1;
            }
            for (0..scan.num_off16hi) |i| {
                var d = self.tans_off16hi_host_buf[i];
                d.dst_offset += @intCast(off16_offset);
                merged_buf[merged_count] = d;
                merged_count += 1;
            }
            for (0..scan.num_off16lo) |i| {
                var d = self.tans_off16lo_host_buf[i];
                d.dst_offset += @intCast(off16_offset);
                merged_buf[merged_count] = d;
                merged_count += 1;
            }

            // Upload tANS descriptors to GPU (before timer)
            if (merged_count > 0) {
                const mdesc_bytes = merged_count * @sizeOf(TansDecChunkDesc);
                const mstatus_bytes = merged_count * @sizeOf(u32);
                if (!ensureDeviceBuf(&self.d_tans_descs_persist, &self.d_tans_descs_persist_size, mdesc_bytes))
                    return error.BadMode;
                if (!ensureDeviceBuf(&self.d_tans_status_persist, &self.d_tans_status_persist_size, mstatus_bytes))
                    return error.BadMode;
                _ = h2d_fn(self.d_tans_descs_persist, @ptrCast(@as([*]TansDecChunkDesc, &merged_buf)), mdesc_bytes);
                if (cuMemsetD8_fn) |memset_fn| _ = memset_fn(self.d_tans_status_persist, 0, mstatus_bytes);
            }

            _ = sync_fn();
        }

        // ── KERNEL TIMER: only pure GPU kernel time from here ──
        const t_before_kern = if (io) |io_val| std.Io.Clock.awake.now(io_val) else null;

        // Launch tANS table-build + decode kernels
        if (tans_kernel_fn != 0 and self.d_tans_descs_persist_size > 0) {
            const tans_count = @as(u32, @intCast(self.d_tans_descs_persist_size / @sizeOf(TansDecChunkDesc)));
            if (tans_count > 0) {
                // Allocate meta buffer
                const meta_bytes = @as(usize, tans_count) * 16;
                if (!ensureDeviceBuf(&self.d_tans_meta, &self.d_tans_meta_size, meta_bytes)) return error.BadMode;

                const tans_grid = (tans_count + 1) / 2;

                // Kernel A: build tables + LUTs
                if (tans_build_fn != 0) {
                    var bp_comp = self.d_comp_persist;
                    var bp_descs = self.d_tans_descs_persist;
                    var bp_lut = self.d_tans_lut;
                    var bp_meta = self.d_tans_meta;
                    var bp_num = tans_count;
                    var build_params = [_]?*anyopaque{
                        @ptrCast(&bp_comp), @ptrCast(&bp_descs),
                        @ptrCast(&bp_lut), @ptrCast(&bp_meta), @ptrCast(&bp_num),
                    };
                    var build_extra = [_]?*anyopaque{null};
                    if (launch_fn(tans_build_fn, tans_grid, 1, 1, 32, 2, 1, 0, 0, &build_params, &build_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                }

                // Kernel B: decode
                {
                    var tp_comp = self.d_comp_persist;
                    var tp_scratch = self.d_tans_scratch;
                    var tp_descs = self.d_tans_descs_persist;
                    var tp_status = self.d_tans_status_persist;
                    var tp_num = tans_count;
                    var tp_lut = self.d_tans_lut;
                    var tp_meta: CUdeviceptr = if (tans_build_fn != 0) self.d_tans_meta else 0;
                    var tans_params = [_]?*anyopaque{
                        @ptrCast(&tp_comp), @ptrCast(&tp_scratch),
                        @ptrCast(&tp_descs), @ptrCast(&tp_status), @ptrCast(&tp_num),
                        @ptrCast(&tp_lut), @ptrCast(&tp_meta),
                    };
                    var tans_extra = [_]?*anyopaque{null};
                    const decode_fn_to_use: usize = if (scan.use_tans32 and tans32_kernel_fn != 0) tans32_kernel_fn else tans_kernel_fn;
                    if (launch_fn(decode_fn_to_use, tans_grid, 1, 1, 32, 2, 1, 0, 0, &tans_params, &tans_extra) != CUDA_SUCCESS)
                        return error.BadMode;
                }
            }
        }

        if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

        if (io) |io_val| {
            if (t_before_kern) |t_start| {
                last_tans_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
            }
        }

        // ── Pass 2: LZ decode ──────────────────────────────────────
        // Fast path: when there's no entropy work (L1/L2 inputs), use the
        // dedicated 6-param kernel — same shape as the May-9 22 GB/s era.
        const no_entropy = merged_count == 0 and n_huff == 0;
        const use_raw_kernel = no_entropy and kernel_raw_fn != 0;

        const grid_x = (num_groups + 1) / 2;

        if (use_raw_kernel) {
            var p_comp = self.d_comp_persist;
            var p_descs_dev = self.d_descs_persist;
            var p_dst = self.d_output;
            var p_cpg = chunks_per_group;
            var p_total: u32 = total_chunks;
            var p_sc_cap = sub_chunk_cap;
            var raw_params = [_]?*anyopaque{
                @ptrCast(&p_comp),
                @ptrCast(&p_descs_dev),
                @ptrCast(&p_dst),
                @ptrCast(&p_cpg),
                @ptrCast(&p_total),
                @ptrCast(&p_sc_cap),
            };
            var raw_extra = [_]?*anyopaque{null};
            if (launch_fn(kernel_raw_fn, grid_x, 1, 1, 32, 2, 1, 0, 0, &raw_params, &raw_extra) != CUDA_SUCCESS)
                return error.BadMode;
        } else {
            var p_comp = self.d_comp_persist;
            var p_descs_dev = self.d_descs_persist;
            var p_dst = self.d_output;
            var p_cpg = chunks_per_group;
            var p_total: u32 = total_chunks;
            var p_sc_cap = sub_chunk_cap;
            var p_tans_scratch = self.d_tans_scratch;
            var p_tans_slot_stride: u64 = @as(u64, total_subchunks) * 131072;
            var p_first_sub_idx: CUdeviceptr = self.d_first_subchunk_idx;

            var params = [_]?*anyopaque{
                @ptrCast(&p_comp),
                @ptrCast(&p_descs_dev),
                @ptrCast(&p_dst),
                @ptrCast(&p_cpg),
                @ptrCast(&p_total),
                @ptrCast(&p_sc_cap),
                @ptrCast(&p_tans_scratch),
                @ptrCast(&p_tans_slot_stride),
                @ptrCast(&p_first_sub_idx),
            };
            var extra = [_]?*anyopaque{null};
            if (launch_fn(kernel_fn, grid_x, 1, 1, 32, 2, 1, 0, 0, &params, &extra) != CUDA_SUCCESS)
                return error.BadMode;
        }

        if (sync_fn() != CUDA_SUCCESS) return error.BadMode;

        if (t_before_kern) |t_start| {
            if (io) |io_val| {
                last_kernel_ns = @intCast(t_start.untilNow(io_val, .awake).toNanoseconds());
            }
        }
    }

    if (t_e2e0) |t0| if (io) |iv| {
        e2e_cum_predh_ns = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
    };
    _ = d2h_fn(@ptrCast(dst_full + dst_start_off), self.d_output + dst_start_off, decompressed_size);
    if (t_e2e0) |t0| if (io) |iv| {
        const cum_end_ns: i64 = @intCast(t0.untilNow(iv, .awake).toNanoseconds());
        const ms = struct {
            fn f(ns: i64) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.f;
        const preblk_ns = e2e_cum_prescan_ns - e2e_cum_h2d_ns; // shared-LUT block region
        const scanfn_ns = e2e_cum_postscan_ns - e2e_cum_prescan_ns; // scanForTansChunks call
        const rawcopy_ns = e2e_cum_scan_ns - e2e_cum_postscan_ns; // raw-off16 device-to-device scatter
        const prep_ns = (e2e_cum_predh_ns - e2e_cum_scan_ns) - last_kernel_ns; // descriptor prep
        std.debug.print("  [e2e] setup+H2D {d:.3}  preBlk {d:.3}  scanFn {d:.3}  rawD2D {d:.3}  prep {d:.3}  kernels {d:.3}  D2H {d:.3}  total {d:.3} ms\n", .{
            ms(e2e_cum_h2d_ns),
            ms(preblk_ns),
            ms(scanfn_ns),
            ms(rawcopy_ns),
            ms(prep_ns),
            ms(last_kernel_ns),
            ms(cum_end_ns - e2e_cum_predh_ns),
            ms(cum_end_ns),
        });
    };
}

// ── Helper: launch tANS kernel for a set of descriptors ───────
// When skip_post_sync is true, the kernel is launched without a trailing
// cuCtxSynchronize — the caller must sync before reading results.
fn launchTansKernel(
    self: *DecodeContext,
    h2d_fn: FnMemcpyHtoD,
    launch_fn: FnLaunchKernel,
    _: FnCtxSync,
    num_descs: u32,
    host_descs: [*]TansDecChunkDesc,
    dst_scratch: CUdeviceptr,
    d_descs: *CUdeviceptr,
    d_descs_size: *usize,
    d_status: *CUdeviceptr,
    d_status_size: *usize,
) !void {
    const desc_bytes = num_descs * @sizeOf(TansDecChunkDesc);
    const status_bytes = num_descs * @sizeOf(u32);

    if (!ensureDeviceBuf(d_descs, d_descs_size, desc_bytes))
        return error.BadMode;
    if (!ensureDeviceBuf(d_status, d_status_size, status_bytes))
        return error.BadMode;

    _ = h2d_fn(d_descs.*, @ptrCast(host_descs), desc_bytes);

    if (cuMemsetD8_fn) |memset_fn| {
        _ = memset_fn(d_status.*, 0, status_bytes);
    }

    var tp_comp = self.d_comp_persist;
    var tp_scratch = dst_scratch;
    var tp_descs = d_descs.*;
    var tp_status = d_status.*;
    var tp_num: u32 = num_descs;

    var tans_params = [_]?*anyopaque{
        @ptrCast(&tp_comp),
        @ptrCast(&tp_scratch),
        @ptrCast(&tp_descs),
        @ptrCast(&tp_status),
        @ptrCast(&tp_num),
    };
    var tans_extra = [_]?*anyopaque{null};

    const tans_grid = (num_descs + 1) / 2;
    if (launch_fn(tans_kernel_fn, tans_grid, 1, 1, 32, 2, 1, 0, 0, &tans_params, &tans_extra) != CUDA_SUCCESS)
        return error.BadMode;
}