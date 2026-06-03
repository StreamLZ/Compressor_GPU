//! One-time Vulkan driver bring-up plus the module-level pipeline handles
//! every decode path launches against.
//!
//! VK PORT NOTE: ports src/decode/module_loader.zig. CUDA loads two PTX
//! modules (`lz_kernel.ptx` and `huffman_kernel.ptx`) via
//! `cuModuleLoadData` then resolves named kernel entries via
//! `cuModuleGetFunction`. The Vulkan equivalent loads one
//! `<kernel>.spv` blob per entry point through `vkCreateShaderModule`
//! and produces a `VkPipeline` per kernel via
//! `vkCreateComputePipelines`. The `pub var` slots keep their CUDA names
//! (kernel_fn, kernel_raw_fn, walk_frame_fn, ...) so the decode
//! dispatch reads identical across the two ports — the slot type is
//! `usize` either way (pointer-sized opaque handle).
//!
//! Foundation wave: init() routes through the foundation-wave VMA setup
//! when called, and returns `false` until the subsequent wave wires the
//! actual SPV pipeline creation. L1 callers that import this module
//! (e.g. decode_context.allocHost) treat `init()` returning `false` as
//! "backend unavailable" and degrade gracefully.

const std = @import("std");

const vulkan = @import("vulkan_api.zig");
const decode_context = @import("decode_context.zig");
const descriptors = @import("descriptors.zig");

// ── Pipeline handles ─────────────────────────────────────────────
// One `pub var` per kernel entry point — matches CUDA's `cuModuleGet
// Function` slots. Foundation wave leaves them at 0; the dispatch
// wave fills them via `vkCreateComputePipelines`. Codec call sites
// check the slot against 0 (mirrors CUDA's `if (huff_build_fn != 0)`)
// before dispatching.
//
// L2 gate (see decode_dispatch.zig): `huff_build_fn`, `huff_decode_fn`,
// `compact_*_fn`, `merge_huff_descs_fn`, and `scan_parse_fn` only
// activate on level>=2 frames. L1 callers MUST NOT dispatch against
// these slots — the gate skips both allocation and dispatch.
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
    switch (vulkan.init_state) {
        .ready => return true,
        .failed, .in_progress => return false, // .in_progress catches re-entry
        .uninit => {},
    }
    vulkan.init_state = .in_progress;
    // Common bail-out: any path that exits this function while still
    // `.in_progress` transitions to `.failed` so the next init() call
    // doesn't re-try the bring-up. The success path sets `.ready`
    // explicitly at the bottom; defer then sees `.ready` and is a no-op.
    defer if (vulkan.init_state == .in_progress) {
        vulkan.init_state = .failed;
    };

    if (std.c.getenv("SLZ_NO_VULKAN") != null) return false;

    // VK PORT NOTE: foundation wave wires only the dlopen handle so the
    // codec call sites (which read `vulkan.lib != null` to detect
    // backend presence) typecheck against a real module. Actual
    // bring-up — vkCreateInstance, vkCreateDevice, VMA allocator,
    // pipeline creation — lands in a subsequent wave. Until then,
    // init() leaves `init_state == .failed`, which the rest of the
    // pipeline reads as "backend not available" and gracefully
    // declines to dispatch.
    vulkan.lib = vulkan.win32.LoadLibraryA("vulkan-1.dll");
    if (vulkan.lib == null) return false;

    vulkan.vkGetInstanceProcAddr_fn = vulkan.getProc(
        vulkan.FnGetInstanceProcAddr,
        "vkGetInstanceProcAddr",
    );
    if (vulkan.vkGetInstanceProcAddr_fn == null) return false;

    // Foundation wave stops here — leave `init_state == .in_progress`,
    // which the deferred transition above flips to `.failed`. The
    // codec sees the same "backend unavailable" surface as CUDA when
    // nvcuda.dll loads but cuCtxCreate fails. L1 decode tests that
    // require a live backend gate themselves on `init() == true`; this
    // is fine because the foundation wave's job is to compile, not to
    // run.
    return false;
}

/// Lazily allocate the persistent pipeline stream on the active device.
/// Called from init() for g_default and from fullGpuLaunchImpl for any
/// per-handle context that hasn't created its stream yet.
///
/// VK PORT NOTE: CUDA's CUstream is a primitive; the Vulkan port backs
/// it with a per-handle VkQueue (or a VkCommandBuffer ring keyed off
/// a shared queue, depending on the dispatch wave). The procs.stream_*
/// shims hide that choice from the codec.
pub fn ensurePipelineStream(d_ctx: *decode_context.DecodeContext) descriptors.GpuError!void {
    if (d_ctx.pipeline_stream_created) return;
    const create_fn = vulkan.procs.stream_create_fn orelse return error.BackendNotAvailable;
    if (create_fn(&d_ctx.pipeline_stream, 1) != vulkan.VK_SUCCESS_RC) return error.BackendNotAvailable;
    d_ctx.pipeline_stream_created = true;
}

pub fn isAvailable() bool {
    return init();
}
