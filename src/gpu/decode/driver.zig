//! GPU decode driver - thin facade.
//!
//! The decode driver was split into focused sub-modules during the GPU
//! cleanup pass (roadmap item 5). External callers continue to import
//! `gpu/decode/driver.zig` and reach every public symbol unchanged; this
//! file owns the singleton `g_default` and the three `last_*_kernel_ns`
//! telemetry `pub var`s (storage must live in the facade - Zig cannot
//! re-export a `pub var` from another module through `pub const`).
//!
//! Sub-module layout:
//!   cuda_api.zig         - nvcuda.dll handle, CU* typedefs, cu*_fn slots,
//!                          getProc, qpcNow / qpcMs, NUM_PIPELINE_STREAMS
//!   module_loader.zig    - PTX load, kernel-handle pub vars, init(),
//!                          isAvailable(), ensurePipelineStreams
//!   descriptors.zig      - ChunkDesc / HuffDecChunkDesc / RawOff16Desc /
//!                          ScanResult / WalkFrameResultDev / WalkMeta /
//!                          KernelTiming / PendingTiming, HUFF_LUT_ENTRIES,
//!                          WALK_MAX_CHUNKS, GpuError
//!   decode_context.zig   - DecodeContext, ensureDeviceBuf / Output,
//!                          alloc/free/copy helpers, profiling fns
//!   scan_host.zig        - scanForEntropyChunks + local header parsers
//!   scan_gpu.zig         - gpuWalkFrameImpl, gpuPrefixSumChunksImpl,
//!                          walkMetaToHost, gpuScanChunks
//!   decode_dispatch.zig  - fullGpuLaunch / fullGpuLaunchImpl + the
//!                          per-decode helpers (dumpScanIfRequested,
//!                          emitE2eTrace, gatherRawOff16, mergeHuffDescs)

const cuda = @import("cuda_api.zig");
const ml = @import("module_loader.zig");
const d = @import("descriptors.zig");
const dc = @import("decode_context.zig");
const sg = @import("scan_gpu.zig");
const dd = @import("decode_dispatch.zig");

// ── Clock helpers ─────────────────────────────────────────────
pub const qpcNow = cuda.qpcNow;
pub const qpcMs = cuda.qpcMs;

// ── Module lifecycle ──────────────────────────────────────────
pub const init = ml.init;
pub const isAvailable = ml.isAvailable;

// ── Descriptor / error types ──────────────────────────────────
pub const ChunkDesc = d.ChunkDesc;
pub const HuffDecChunkDesc = d.HuffDecChunkDesc;
pub const RawOff16Desc = d.RawOff16Desc;
pub const ScanHuffDesc = d.ScanHuffDesc;
pub const ScanRawDesc = d.ScanRawDesc;
pub const ScanResult = d.ScanResult;
pub const WalkFrameResultDev = d.WalkFrameResultDev;
pub const WalkMeta = d.WalkMeta;
pub const KernelTiming = d.KernelTiming;
pub const PendingTiming = d.PendingTiming;
pub const GpuError = d.GpuError;
pub const HUFF_LUT_ENTRIES = d.HUFF_LUT_ENTRIES;
pub const WALK_MAX_CHUNKS = d.WALK_MAX_CHUNKS;

// ── Per-decode context + host I/O helpers ─────────────────────
pub const DecodeContext = dc.DecodeContext;
pub const allocHost = dc.allocHost;
pub const freeHost = dc.freeHost;
pub const copyDeviceToHost = dc.copyDeviceToHost;
pub const copyHostToDevice = dc.copyHostToDevice;
pub const bindContextToCallingThread = dc.bindContextToCallingThread;
pub const beginKernelTiming = dc.beginKernelTiming;
pub const endKernelTiming = dc.endKernelTiming;
pub const finalizeProfiling = dc.finalizeProfiling;

// ── Decode pipeline entry points ──────────────────────────────
pub const gpuWalkFrameImpl = sg.gpuWalkFrameImpl;
pub const gpuPrefixSumChunksImpl = sg.gpuPrefixSumChunksImpl;
pub const walkMetaToHost = sg.walkMetaToHost;
pub const fullGpuLaunch = dd.fullGpuLaunch;
pub const fullGpuLaunchImpl = dd.fullGpuLaunchImpl;

// ── Singletons ────────────────────────────────────────────────
// `g_default` and the three `last_*_kernel_ns` telemetry vars live on
// the facade so external callers reading `gpu_decode.X` keep working.
// Sub-modules write back via `@import("driver.zig").X = ...`.

/// Default decode context backing the module-level public API.
pub var g_default: DecodeContext = .{};

/// Per-decode wall-clock totals (nanoseconds). `last_kernel_ns` is the
/// full GPU launch window; the split vars are populated when
/// `SLZ_SPLIT_TIMER=1` separates Huff pre-decode from LZ decode.
pub var last_kernel_ns: i64 = 0;
pub var last_lz_kernel_ns: i64 = 0;
pub var last_huff_kernel_ns: i64 = 0;
