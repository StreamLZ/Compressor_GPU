//! Plain-old-data descriptor types shared by the GPU decode pipeline.
//!
//! Kept free of Vulkan bindings on purpose — sub-modules import this for
//! the struct shapes and the few constants without dragging in
//! `vulkan_api.zig`. `extern struct` layouts here MUST stay in sync with
//! the C/GLSL mirrors in `src_vk/decode/slz_wire_format.glsl` and the
//! `*_kernel.comp` shaders that consume them (each has its own size
//! assert on the GLSL side).
//!
//! VK PORT NOTE: ports src/decode/descriptors.zig. The descriptor
//! structs themselves stay byte-identical with the CUDA build because
//! they cross the wire format — the only port-side adaptations are:
//!   * `cudaCall` becomes `vkCall` (the error-funnel helper),
//!   * `ErrorKind` keeps its members verbatim,
//!   * `GpuError` gains `NotImplementedL2` for the L2-gate stubs.

const std = @import("std");

const error_mod = @import("../error.zig");

/// LZ chunk descriptor. Wire-compatible with the device-side
/// `SlzChunkDesc` struct (`src_vk/decode/slz_wire_format.glsl`); the
/// GLSL side asserts size/offset parity.
pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    /// Trailing pad keeping `memset_fill` from leaving the struct at a
    /// non-u32-aligned size, so `SlzChunkDesc[]` arrays stay 4-byte
    /// aligned. The device side never reads these bytes but the layout
    /// asserts on `sizeof(SlzChunkDesc) == 24`.
    reserved: [3]u8 = @splat(0),
};

// ── Huffman literal descriptors — matches huff_decode_4stream.comp's
// HuffDecChunkDesc. in_offset/in_size cover the FULL payload (128 B
// weights + 93 B sub-header + 32 stream payloads, per HUFF_NUM_STREAMS /
// HUFF_SUBHEADER_BYTES / HUFF_BODY_HEADER_BYTES). Build kernel reads
// first 128 B; decode kernel skips 128.
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
// device-side compact buffers — see the comment there.
pub const MAX_SUB_CHUNKS_PER_CHUNK: u32 = 4;
pub const MAX_HUFF_DESCS_PER_STREAM: u32 = WALK_MAX_CHUNKS / MAX_SUB_CHUNKS_PER_CHUNK;
pub const MAX_RAW_OFF16_DESCS: u32 = 8192;

// ── Per-sub-chunk entropy scratch geometry ──────────────────────
// Mirror the C-side constants in slz_wire_format.glsl:
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
// records a Vulkan timestamp pair through a VkQueryPool. After the
// final sync, finalizeProfiling computes elapsed time per kernel and
// stores results in `last_timings`, which slzGetLastTimings exposes via
// the C ABI.
//
// VK PORT NOTE: CUDA uses a pair of cuEvents; Vulkan uses VkQueryPool
// timestamps. The PendingTiming shape exposes two `usize` slots so the
// device-side query indices fit without changing the host-visible
// struct shape — the codec call sites continue to read pairs.

/// Resolved per-kernel timing reported across the C ABI. Wire-compatible
/// with `slzKernelTiming_t` in `include/streamlz_gpu.h`; `name` is a
/// static string with process lifetime, `ms` is wall-clock elapsed time
/// measured via Vulkan timestamp queries after the matching kernel
/// finished.
pub const KernelTiming = extern struct {
    name: [*:0]const u8,
    ms: f32,
};

/// In-flight per-kernel timing entry held internally on a DecodeContext
/// while the matching kernel is queued or running. NOT crossing the
/// C ABI — `finalizeProfiling` converts these to `KernelTiming` after
/// it drains the timestamp pair.
pub const PendingTiming = struct {
    name: [*:0]const u8,
    start_event: usize,
    end_event: usize,
};

// Staged decode-scan output — mirror SlzScanHuffDesc / SlzScanRawDesc in
// scan_parse.comp (filled) and compact_*_descs.comp (compacted).
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
/// status)`. Nothing is D2H'd by the walk — downstream kernels read the
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
/// (d_chunk_descs, d_n_chunks, d_sub_chunk_cap) — all device-resident —
/// and writes (d_first_sub_idx, d_total_subchunks). No D2H, no CPU work.
pub const PrefixSumResultDev = struct {
    d_first_sub_idx: u64,
    d_total_subchunks: u64,
};

/// Errors the GPU decode path surfaces. The C ABI in
/// `src_vk/streamlz_gpu.zig` switches on these to route between
/// `SLZ_ERROR_VULKAN` (every Vulkan-side failure), `SLZ_ERROR_UNSUPPORTED`
/// (the `BadMode` shape-rejected fallback signal), and the
/// `NotImplementedL2` stub — present so every L2 file under decode/ can
/// share the same `GpuError` set without growing a parallel error.
pub const GpuError = error{
    /// Vulkan loader could not be brought up. Causes: `vulkan-1.dll` /
    /// `libvulkan.so` missing, `vkCreateInstance` failed, no compatible
    /// device present, or a required FFI symbol was absent from
    /// `getProc`.
    BackendNotAvailable,
    /// `vmaCreateBuffer` returned out-of-memory for a device buffer the
    /// dispatch needed to grow.
    OutOfDeviceMemory,
    /// `vkCmdDispatch` (or any kernel submission) returned non-success
    /// for one of the decode pipeline kernels (walk / prefix-sum /
    /// scan / compact / merge / huff-build / huff-decode / LZ-decode /
    /// gather-raw).
    KernelLaunchFailed,
    /// `vkQueueWaitIdle` or `vkWaitForFences` returned non-success.
    /// Usually the symptom of an earlier kernel that ran but faulted.
    SyncFailed,
    /// Any `vkCmdCopyBuffer` (HtoD / DtoH / DtoD) or `vkCmdFillBuffer`
    /// returned non-success.
    CopyFailed,
    /// A required pipeline handle is `0`. The SPV blob loaded but the
    /// entry point was not present — usually a build/version skew
    /// between the Zig side and the compiled shaders.
    KernelMissing,
    /// The frame shape is not supported by the device-resident decoder
    /// (multi-block, dictionary, content-size out of bounds, etc.).
    /// The C ABI translates this to `SLZ_ERROR_UNSUPPORTED` and the
    /// caller falls back to the host-bounce decode path.
    BadMode,
    /// The dispatch tried to enter an L2+ code path that has not yet
    /// been ported from CUDA. Mirrors `error_mod.NotImplementedL2`;
    /// re-declared here so the `GpuError` set is the single error
    /// surface decode call sites depend on.
    NotImplementedL2,
};

comptime {
    // Smoke check: keeping `NotImplementedL2` reachable from both the
    // cross-cutting `error_mod` module and the decode-side `GpuError`
    // makes the L2 stub idiom uniform. If anyone removes one but not
    // the other, this comptime fires.
    _ = error_mod.NotImplementedL2;
}

/// Tag categorizing what the wrapped Vulkan call was doing, so `vkCall`
/// can return the right `GpuError` member on failure. Passed as a
/// `comptime` parameter so the switch is constant-folded.
///
/// VK PORT NOTE: ports `ErrorKind` from CUDA verbatim — same five
/// variants, same intent. The implementation funnel (`vkCall`) takes a
/// `VkResult` instead of a `CUresult`; the kind→error mapping is
/// otherwise identical.
pub const ErrorKind = enum {
    /// `vkCmdDispatch` / `vkQueueSubmit` for a compute pipeline.
    launch,
    /// `vkQueueWaitIdle` / `vkWaitForFences`.
    sync,
    /// Any `vkCmdCopyBuffer` (sync or async, any direction) or
    /// `vkCmdFillBuffer`.
    copy,
    /// `vmaCreateBuffer` (and the bool-returning `ensureDeviceBuf`
    /// wrappers).
    alloc,
    /// `vkCreateInstance` / `vkEnumeratePhysicalDevices` /
    /// `vkCreateDevice` / pipeline create.
    init,
};

/// Funnel any Vulkan return code into the GpuError surface so callers
/// can `try vkCall(procs.h2d(...), .copy)` instead of dropping the result
/// with `_ = ...`. The `kind` tag selects which `GpuError` member to
/// return — see ErrorKind above for the mapping.
///
/// VK PORT NOTE: ports `cudaCall` 1:1, just renamed `cudaCall → vkCall`
/// (token "cuda" is on the adaptation allow-list). VK_SUCCESS_RC is 0,
/// same convention as CUDA_SUCCESS, so the rc==0 fast path stays
/// branch-free.
pub fn vkCall(rc: c_int, comptime kind: ErrorKind) GpuError!void {
    if (rc == 0) return; // VK_SUCCESS == 0
    return switch (kind) {
        .launch => error.KernelLaunchFailed,
        .sync => error.SyncFailed,
        .copy => error.CopyFailed,
        .alloc => error.OutOfDeviceMemory,
        .init => error.BackendNotAvailable,
    };
}

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "ChunkDesc size matches CUDA wire-format" {
    // Wire-format contract: SlzChunkDesc must be 24 bytes (5 u32s + u8
    // + 3 pad). GLSL side asserts the same.
    try testing.expectEqual(@as(usize, 24), @sizeOf(ChunkDesc));
}

test "vkCall maps non-zero rc into the kind-tagged GpuError" {
    try testing.expectError(error.CopyFailed, vkCall(-1, .copy));
    try testing.expectError(error.OutOfDeviceMemory, vkCall(-1, .alloc));
    try testing.expectError(error.BackendNotAvailable, vkCall(-1, .init));
    // VK_SUCCESS_RC == 0 is the success path; should NOT error.
    try vkCall(0, .copy);
}

test "GpuError exposes NotImplementedL2 for stub bodies" {
    // L2 stub files return `error.NotImplementedL2;` directly. The
    // GpuError set must carry it so decode-path stubs typecheck against
    // the same error surface as the L1 success path.
    const sample: GpuError = error.NotImplementedL2;
    try testing.expectError(error.NotImplementedL2, @as(GpuError!void, sample));
}
