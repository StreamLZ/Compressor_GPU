//! GPU decode driver - thin facade.
//!
//! External callers import `decode/driver.zig` and reach every public
//! symbol unchanged; this file owns the singleton `g_default` and the
//! three `last_*_kernel_ns` telemetry `pub var`s (storage must live in
//! the facade - Zig cannot re-export a `pub var` from another module
//! through `pub const`).
//!
//! Sub-module layout:
//!   cuda_api.zig         - nvcuda.dll handle, CU* typedefs, cu*_fn slots,
//!                          getProc, qpcNow / qpcMs
//!   module_loader.zig    - PTX load, kernel-handle pub vars, init(),
//!                          isAvailable(), ensurePipelineStream
//!   descriptors.zig      - ChunkDesc, KernelTiming, PendingTiming,
//!                          WALK_MAX_CHUNKS, GpuError, walk_meta_offsets
//!                          (plus internal scan/huff descriptor types)
//!   decode_context.zig   - DecodeContext, ensureDeviceBuf / Output,
//!                          alloc/free/copy helpers, profiling fns
//!   scan_gpu.zig         - gpuWalkFrameImpl, gpuPrefixSumChunksImpl,
//!                          gpuScanChunks
//!   decode_dispatch.zig  - fullGpuLaunchImpl + the per-decode helpers
//!                          (emitE2eTrace, gatherRawOff16, mergeHuffDescs)

const cuda = @import("cuda_api.zig");
const module_loader = @import("module_loader.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const scan_gpu = @import("scan_gpu.zig");
const decode_dispatch = @import("decode_dispatch.zig");

// ── Clock helpers ─────────────────────────────────────────────
pub const qpcNow = cuda.qpcNow;
pub const qpcMs = cuda.qpcMs;

// ── Module lifecycle ──────────────────────────────────────────
pub const init = module_loader.init;
/// v4 #19 v2: whether the device-side per-chunk hash kernel resolved
/// (the decoder falls back to the parallel host hash when absent).
pub fn hasChunkHashKernel() bool {
    return module_loader.seg_hash_fn != 0 and module_loader.chunk_combine_fn != 0;
}
pub const isAvailable = module_loader.isAvailable;

/// Name of the active CUDA device (VK-parity: print next to every
/// benchmark). Empty slice until init() has run successfully.
pub fn deviceName() []const u8 {
    return cuda.device_name_buf[0..cuda.device_name_len];
}

// ── Descriptor / error types ──────────────────────────────────
pub const ChunkDesc = descriptors.ChunkDesc;
pub const KernelTiming = descriptors.KernelTiming;
pub const PendingTiming = descriptors.PendingTiming;
pub const GpuError = descriptors.GpuError;
pub const WALK_MAX_CHUNKS = descriptors.WALK_MAX_CHUNKS;
pub const ENTROPY_SCRATCH_SLOT_BYTES = descriptors.ENTROPY_SCRATCH_SLOT_BYTES;
pub const walk_meta_offsets = descriptors.walk_meta_offsets;

// ── Per-decode context + host I/O helpers ─────────────────────
pub const DecodeContext = decode_context.DecodeContext;
pub const allocHost = decode_context.allocHost;
pub const freeHost = decode_context.freeHost;
pub const copyDeviceToHost = decode_context.copyDeviceToHost;
pub const copyHostToDevice = decode_context.copyHostToDevice;
pub const ensureDictOnDevice = decode_context.ensureDictOnDevice;
pub const bindContextToCallingThread = decode_context.bindContextToCallingThread;
pub const beginKernelTiming = decode_context.beginKernelTiming;
pub const endKernelTiming = decode_context.endKernelTiming;
pub const finalizeProfiling = decode_context.finalizeProfiling;

// ── Decode pipeline entry points ──────────────────────────────
pub const gpuWalkFrameImpl = scan_gpu.gpuWalkFrameImpl;
pub const fullGpuLaunchImpl = decode_dispatch.fullGpuLaunchImpl;

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
