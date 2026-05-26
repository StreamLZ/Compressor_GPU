//! Plain-old-data descriptor types shared by the GPU decode pipeline.
//!
//! Kept free of CUDA bindings on purpose — sub-modules import this for the
//! struct shapes and the few constants without dragging in `cuda_api.zig`.
//! `extern struct` layouts here MUST stay in sync with the C/CUDA mirrors
//! in `src/gpu/decode/slz_wire_format.cuh` and the `*_kernel.cuh` headers
//! that consume them (each has its own `static_assert` on sizeof).

const std = @import("std");

// ── LZ chunk descriptor (C ABI mirror: SlzChunkDesc) ────────────────
pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

// ── Huffman literal descriptors — matches decode/huffman_kernel.cu HuffDecChunkDesc.
// in_offset/in_size cover the FULL payload (128 B weights + 9 B sub-header +
// 4 stream payloads). Build kernel reads first 128 B; decode kernel skips 128.
pub const HuffDecChunkDesc = extern struct {
    in_offset: u32,
    in_size: u32,
    out_offset: u32,
    out_size: u32,
    lut_offset: u32,
};

pub const HUFF_LUT_ENTRIES: usize = 1024; // matches MAX_CODE_LEN=10 (10-bit escape LUT) in kernel

// ── Per-context buffer caps ─────────────────────────────────────
// Host-side max counts; sized for the largest frame the GPU codec
// can produce (walk_max_chunks chunks * sub-chunks-per-chunk).
//
// MAX_SUB_CHUNKS_PER_CHUNK names the host-scan path's worst-case
// fan-out (each chunk produces at most this many entropy descs per
// stream). Expressing the cap as WALK_MAX_CHUNKS / MAX_SUB_CHUNKS_PER_CHUNK
// makes the relationship visible; the scan_gpu.zig D2D path uses a
// different bound (WALK_MAX_CHUNKS * MAX_SUB_CHUNKS_PER_CHUNK) for the
// device-side compact buffers — see the comment there.
pub const MAX_SUB_CHUNKS_PER_CHUNK: u32 = 4;
pub const MAX_HUFF_DESCS_PER_STREAM: u32 = WALK_MAX_CHUNKS / MAX_SUB_CHUNKS_PER_CHUNK;
pub const MAX_RAW_OFF16_DESCS: u32 = 8192;

// ── Per-sub-chunk entropy scratch geometry ──────────────────────
// Mirror the C-side constants in decode/slz_wire_format.cuh:
//   ENTROPY_SCRATCH_SLOT_BYTES = 131072 — one per global sub-chunk
//   OFF16_HILO_SPLIT_OFFSET    = 65536  — hi bytes start, lo bytes at +offset
// (Sub-chunks decompress up to 128KB; the slot is sized for that.)
pub const ENTROPY_SCRATCH_SLOT_BYTES: u64 = 131072;
pub const OFF16_HILO_SPLIT_OFFSET: u32 = 65536;

// ── Chunk-header type field ─────────────────────────────────────
// Mirror common/gpu_wire_format.cuh:
//   ct = (chunk_header_byte >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK
pub const CHUNK_TYPE_SHIFT: u3 = 4;
pub const CHUNK_TYPE_MASK: u8 = 0x7;

// ── Per-kernel timing infrastructure ─────────────────────────────
// When `enable_profiling` is set on a DecodeContext, each kernel launch
// is wrapped in a cuEvent pair. After the final sync, finalizeProfiling
// computes elapsed time per kernel and stores results in `last_timings`,
// which slzGetLastTimings exposes via the C ABI.
pub const KernelTiming = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

pub const PendingTiming = struct {
    name: [*:0]const u8,
    start_event: usize,
    end_event: usize,
};

// Staged decode-scan output — mirror SlzScanHuffDesc / SlzScanRawDesc in
// scan_parse_kernel.cuh (filled) and compact_descs_kernels.cuh (compacted).
// slzScanParseKernel fills one entry per stream type per global sub-chunk
// index; `valid` marks an entropy-coded / raw stream present. gpuScanChunks
// compacts the valid slots into the merged HuffDecChunkDesc / RawOff16Desc
// arrays.
pub const ScanHuffDesc = extern struct {
    in_offset: u32 = 0,
    in_size: u32 = 0,
    out_offset: u32 = 0,
    out_size: u32 = 0,
    valid: u32 = 0,
};
pub const ScanRawDesc = extern struct {
    src_offset: u32 = 0,
    size: u32 = 0,
    gpu_offset: u32 = 0,
    valid: u32 = 0,
};

/// Describes a raw (type 0 memcpy) off16 sub-stream that needs H2D copy.
pub const RawOff16Desc = struct {
    src_offset: u32, // offset of raw bytes in compressed buffer
    size: u32, // number of bytes
    gpu_offset: u32, // offset in d_entropy_off16_scratch
};

pub const ScanResult = struct {
    num_raw_off16: u32 = 0,
    num_huff_lit: u32 = 0,
    num_huff_tok: u32 = 0,
    num_huff_off16hi: u32 = 0,
    num_huff_off16lo: u32 = 0,
    /// Set when gpuScanChunks ran the device-side compact kernels and
    /// `d_compact_*` buffers hold the per-stream compacted descs + counts
    /// in `d_compact_counts`. fullGpuLaunchImpl uses this to dispatch the
    /// GPU merge kernel (step 6c) instead of the CPU append loop.
    device_compact_populated: bool = false,
};

/// 4d Phase 3 step 1: GPU walk-kernel result, device-only. d_chunk_descs
/// holds up to `walk_max_chunks` SlzChunkDesc entries; d_meta is six
/// u32s: (n_chunks, decomp_size, sub_chunk_cap, block_start, block_size,
/// status). Nothing is D2H'd by the walk — downstream kernels read the
/// device pointers directly.
pub const WalkFrameResultDev = struct {
    d_chunk_descs: u64,
    d_meta: u64,
};
pub const WALK_MAX_CHUNKS: u32 = 16384;
/// Back-compat alias for the prior name; remove after call sites migrate.
pub const walk_max_chunks: u32 = WALK_MAX_CHUNKS;
pub const walk_meta_offsets = struct {
    pub const n_chunks: u32 = 0;
    pub const decomp_size: u32 = 4;
    pub const sub_chunk_cap: u32 = 8;
    pub const block_start: u32 = 12;
    pub const block_size: u32 = 16;
    pub const status: u32 = 20;
    pub const bytes: usize = 24;
};

/// Host-side mirror of the walk kernel's meta. Used only by code paths
/// that still need the values on the host (the legacy / fallback
/// decompress entries). The pure-D2D path never calls this.
pub const WalkMeta = struct {
    n_chunks: u32,
    decomp_size: u32,
    sub_chunk_cap: u32,
    block_start: u32,
    block_size: u32,
    status: u32,
};

/// 4d Phase 3 step 2: device-side prefix sum of per-chunk sub-chunk
/// counts. Reads (d_chunk_descs, d_n_chunks, d_sub_chunk_cap) — all
/// device-resident — and writes (d_first_sub_idx, d_total_subchunks).
/// No D2H, no CPU work.
pub const PrefixSumResultDev = struct {
    d_first_sub_idx: u64,
    d_total_subchunks: u64,
};

/// Error set returned by the GPU decode path. Names are intentionally
/// backend-neutral so the same set works for the CUDA driver and the
/// Vulkan compute backend (`vulkan_driver.zig`). `BadMode` stays as a
/// member because (a) it's part of `fast.DecodeError` (see
/// `src/decode/fast/fast_lz_decoder.zig`) so callers that return
/// `DecompressError` already unify, and (b) the C ABI in
/// `src/streamlz_gpu.zig` switches on `error.BadMode` as the
/// "fall back to CPU" signal. Treat the new members as more-informative
/// variants that ALSO trigger fallback — the higher-level catch sites
/// catch the full `GpuError` set, not just `BadMode`.
pub const GpuError = error{
    BackendNotAvailable, // dlopen/getProc, cuInit, vkCreateInstance failed
    OutOfDeviceMemory, // cuMemAlloc / vkAllocateMemory failed
    KernelLaunchFailed, // cuLaunchKernel / vkCmdDispatch+QueueSubmit failed
    SyncFailed, // cuCtxSync / cuStreamSync / vkWaitForFences failed
    CopyFailed, // cuMemcpy[HtoD|DtoH|DtoDAsync] / vkCmdCopyBuffer failed
    KernelMissing, // a required kernel handle is 0 (PTX loaded but symbol absent)
    BadMode, // ABI-compat + catch-all fallback signal
};

/// Tag categorizing what the wrapped CUDA call was doing, so `cudaCall`
/// can return the right `GpuError` member on failure. Threaded as a
/// `comptime` parameter so the switch is constant-folded.
pub const ErrorKind = enum {
    launch, // cuLaunchKernel
    sync, // cuCtxSynchronize / cuStreamSynchronize
    copy, // cuMemcpy* (any direction, sync or async) + cuMemsetD8*
    alloc, // cuMemAlloc (and the bool-returning ensureDeviceBuf wrappers)
    init, // cuInit / cuDeviceGet / cuCtxCreate / cuModuleLoadData
};

/// Funnel any CUDA Driver API return code into the GpuError surface so
/// callers can `try cudaCall(cuMemcpyHtoD_fn(...), .copy)` instead of
/// dropping the result with `_ = ...`. The `kind` tag selects which
/// `GpuError` member to return — see ErrorKind above for the mapping.
pub fn cudaCall(rc: c_int, comptime kind: ErrorKind) GpuError!void {
    if (rc == 0) return; // CUDA_SUCCESS == 0
    return switch (kind) {
        .launch => error.KernelLaunchFailed,
        .sync => error.SyncFailed,
        .copy => error.CopyFailed,
        .alloc => error.OutOfDeviceMemory,
        .init => error.BackendNotAvailable,
    };
}
