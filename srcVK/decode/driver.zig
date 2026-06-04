//! 1:1 port of src/decode/driver.zig.
//!
//! GPU decode driver facade. External callers import driver.zig and
//! reach every public symbol unchanged; this file owns the singleton
//! g_default and the three last_*_kernel_ns telemetry pub vars (storage
//! must live here — Zig cannot re-export a pub var from another module
//! through pub const).
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const vk = @import("vulkan_api.zig");
const module_loader = @import("module_loader.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const scan_gpu = @import("scan_gpu.zig");
const decode_dispatch = @import("decode_dispatch.zig");

// ── Clock helpers ──────────────────────────────────────────────────────
pub const qpcNow = vk.qpcNow;
pub const qpcMs = vk.qpcMs;

// ── Module lifecycle ───────────────────────────────────────────────────
pub const init = module_loader.init;
pub const isAvailable = module_loader.isAvailable;

// ── Descriptor / error types ───────────────────────────────────────────
pub const ChunkDesc = descriptors.ChunkDesc;
pub const KernelTiming = descriptors.KernelTiming;
pub const PendingTiming = descriptors.PendingTiming;
pub const GpuError = descriptors.GpuError;
pub const WALK_MAX_CHUNKS = descriptors.WALK_MAX_CHUNKS;
pub const ENTROPY_SCRATCH_SLOT_BYTES = descriptors.ENTROPY_SCRATCH_SLOT_BYTES;
pub const walk_meta_offsets = descriptors.walk_meta_offsets;

// ── Per-decode context + host I/O helpers ──────────────────────────────
pub const DecodeContext = decode_context.DecodeContext;
pub const allocHost = decode_context.allocHost;
pub const freeHost = decode_context.freeHost;
pub const copyDeviceToHost = decode_context.copyDeviceToHost;
pub const copyHostToDevice = decode_context.copyHostToDevice;
pub const bindContextToCallingThread = decode_context.bindContextToCallingThread;
pub const beginKernelTiming = decode_context.beginKernelTiming;
pub const endKernelTiming = decode_context.endKernelTiming;
pub const finalizeProfiling = decode_context.finalizeProfiling;

// ── Decode pipeline entry points ───────────────────────────────────────
pub const gpuWalkFrameImpl = scan_gpu.gpuWalkFrameImpl;
pub const fullGpuLaunchImpl = decode_dispatch.fullGpuLaunchImpl;

// ── Singletons ─────────────────────────────────────────────────────────
pub var g_default: DecodeContext = .{};

pub var last_kernel_ns: i64 = 0;
pub var last_lz_kernel_ns: i64 = 0;
pub var last_huff_kernel_ns: i64 = 0;
