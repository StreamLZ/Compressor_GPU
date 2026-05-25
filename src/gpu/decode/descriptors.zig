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
    num_raw_off16: u32,
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

/// Parse-helper return type for type 0 (memcpy) stream headers.
pub const Type0Info = struct { data_offset: u32, size: u32 };

/// 4d Phase 3 step 1: GPU walk-kernel result, device-only. d_chunk_descs
/// holds up to `walk_max_chunks` SlzChunkDesc entries; d_meta is six
/// u32s: (n_chunks, decomp_size, sub_chunk_cap, block_start, block_size,
/// status). Nothing is D2H'd by the walk — downstream kernels read the
/// device pointers directly.
pub const WalkFrameResultDev = struct {
    d_chunk_descs: u64,
    d_meta: u64,
};
pub const walk_max_chunks: u32 = 16384;
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

/// Error set returned by `fullGpuLaunch` / `fullGpuLaunchImpl`. The GPU
/// decode path only ever fails with `BadMode` (driver/kernel unavailable,
/// device allocation failure, or a CUDA call returning non-success). Kept
/// local so this file imports nothing outside `src/gpu/`. `BadMode` is a
/// member of the decoder's `DecodeError`, so callers that return
/// `DecompressError` (which includes `fast.DecodeError`) still unify.
pub const GpuError = error{
    BadMode,
};

/// Funnel any CUDA Driver API return code into the GpuError surface so
/// callers can `try cudaCall(cuMemcpyHtoD_fn(...))` instead of dropping
/// the result with `_ = ...`. The decode path's only failure mode is
/// `BadMode`; this keeps every silent-corruption surface one `try` away.
pub fn cudaCall(rc: c_int) GpuError!void {
    if (rc != 0) return error.BadMode; // CUDA_SUCCESS == 0
}
