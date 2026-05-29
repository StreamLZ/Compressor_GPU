//! Plain-old-data descriptor types shared by the GPU decode pipeline.
//!
//! Kept free of CUDA bindings on purpose - sub-modules import this for the
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

// ── Huffman literal descriptors - matches decode/huffman_kernel.cu HuffDecChunkDesc.
// in_offset/in_size cover the FULL payload (128 B weights + 93 B sub-header +
// 32 stream payloads, per HUFF_NUM_STREAMS / HUFF_SUBHEADER_BYTES /
// HUFF_BODY_HEADER_BYTES in src/gpu/common/gpu_huffman.cuh). Build kernel
// reads first 128 B; decode kernel skips 128.
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
// can produce (WALK_MAX_CHUNKS chunks * sub-chunks-per-chunk).
//
// MAX_SUB_CHUNKS_PER_CHUNK names the host-scan path's worst-case
// fan-out (each chunk produces at most this many entropy descs per
// stream). Expressing the cap as WALK_MAX_CHUNKS / MAX_SUB_CHUNKS_PER_CHUNK
// makes the relationship visible; the scan_gpu.zig D2D path uses a
// different bound (WALK_MAX_CHUNKS * MAX_SUB_CHUNKS_PER_CHUNK) for the
// device-side compact buffers - see the comment there.
pub const MAX_SUB_CHUNKS_PER_CHUNK: u32 = 4;
pub const MAX_HUFF_DESCS_PER_STREAM: u32 = WALK_MAX_CHUNKS / MAX_SUB_CHUNKS_PER_CHUNK;
pub const MAX_RAW_OFF16_DESCS: u32 = 8192;

// ── Per-sub-chunk entropy scratch geometry ──────────────────────
// Mirror the C-side constants in decode/slz_wire_format.cuh:
//   ENTROPY_SCRATCH_SLOT_BYTES = 131072 - one per global sub-chunk
//   OFF16_HILO_SPLIT_OFFSET    = 65536  - hi bytes start, lo bytes at +offset
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

// Staged decode-scan output - mirror SlzScanHuffDesc / SlzScanRawDesc in
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
};

/// GPU walk-kernel result, device-only. `d_chunk_descs` holds up to
/// `WALK_MAX_CHUNKS` SlzChunkDesc entries; `d_meta` is six u32s:
/// `(n_chunks, decomp_size, sub_chunk_cap, block_start, block_size,
/// status)`. Nothing is D2H'd by the walk - downstream kernels read the
/// device pointers directly.
pub const WalkFrameResultDev = struct {
    d_chunk_descs: u64,
    d_meta: u64,
};
pub const WALK_MAX_CHUNKS: u32 = 16384;
pub const walk_meta_offsets = struct {
    pub const n_chunks: u32 = 0;
    pub const decomp_size: u32 = 4;
    pub const sub_chunk_cap: u32 = 8;
    pub const block_start: u32 = 12;
    pub const block_size: u32 = 16;
    pub const status: u32 = 20;
    pub const bytes: usize = 24;
};

/// Device-side prefix sum of per-chunk sub-chunk counts. Reads
/// (d_chunk_descs, d_n_chunks, d_sub_chunk_cap) - all device-resident -
/// and writes (d_first_sub_idx, d_total_subchunks). No D2H, no CPU work.
pub const PrefixSumResultDev = struct {
    d_first_sub_idx: u64,
    d_total_subchunks: u64,
};

/// Errors the GPU decode path surfaces. The C ABI in
/// `src/streamlz_gpu.zig` switches on these to route between
/// `SLZ_ERROR_CUDA` (every CUDA-side failure) and `SLZ_ERROR_UNSUPPORTED`
/// (the `BadMode` shape-rejected fallback signal).
pub const GpuError = error{
    /// CUDA driver could not be brought up. Causes: `nvcuda.dll` /
    /// `libcuda.so` missing, `cuInit` failed, no compatible device
    /// present, or a required FFI symbol was absent from `getProc`.
    BackendNotAvailable,
    /// `cuMemAlloc` returned out-of-memory for a device buffer the
    /// dispatch needed to grow.
    OutOfDeviceMemory,
    /// `cuLaunchKernel` returned non-success for one of the decode
    /// pipeline kernels (walk / prefix-sum / scan / compact / merge /
    /// huff-build / huff-decode / LZ-decode / gather-raw).
    KernelLaunchFailed,
    /// `cuCtxSynchronize` or `cuStreamSynchronize` returned non-success.
    /// Usually the symptom of an earlier kernel that ran but faulted.
    SyncFailed,
    /// Any `cuMemcpy*` call (HtoD / DtoH / DtoDAsync) or `cuMemsetD8*`
    /// returned non-success.
    CopyFailed,
    /// A required kernel symbol's handle is `0`. The PTX loaded but the
    /// extern entry point was not present - usually a build/version
    /// skew between the Zig side and the PTX.
    KernelMissing,
    /// The frame shape is not supported by the device-resident decoder
    /// (multi-block, dictionary, content-size out of bounds, etc.).
    /// The C ABI translates this to `SLZ_ERROR_UNSUPPORTED` and the
    /// caller falls back to the host-bounce decode path.
    BadMode,
};

/// Tag categorizing what the wrapped CUDA call was doing, so `cudaCall`
/// can return the right `GpuError` member on failure. Passed as a
/// `comptime` parameter so the switch is constant-folded.
pub const ErrorKind = enum {
    /// `cuLaunchKernel`.
    launch,
    /// `cuCtxSynchronize` / `cuStreamSynchronize`.
    sync,
    /// Any `cuMemcpy*` (sync or async, any direction) or `cuMemsetD8*`.
    copy,
    /// `cuMemAlloc` (and the bool-returning `ensureDeviceBuf` wrappers).
    alloc,
    /// `cuInit` / `cuDeviceGet` / `cuCtxCreate` / `cuModuleLoadData`.
    init,
};

/// Funnel any CUDA Driver API return code into the GpuError surface so
/// callers can `try cudaCall(cuMemcpyHtoD_fn(...), .copy)` instead of
/// dropping the result with `_ = ...`. The `kind` tag selects which
/// `GpuError` member to return - see ErrorKind above for the mapping.
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
