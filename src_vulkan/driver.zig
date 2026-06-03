//! Vulkan driver facade — mirrors src/decode/driver.zig's role for the
//! CUDA pipeline. Owns the process-singleton `g_default` Context that
//! every later milestone's encode/decode dispatcher will read.
//!
//! `ensureInit` is idempotent: it loads vulkan-1.dll, creates a VkInstance
//! (with optional validation), picks the first compute-capable physical
//! device, creates a logical device + queue, and stashes everything on
//! `g_default`. `deinit` tears the bundle down in reverse order.

const std = @import("std");

const vk = @import("vk_api.zig");
const instance_mod = @import("instance.zig");
const device_mod = @import("device.zig");
const dispatch_mod = @import("dispatch.zig");
const probe_mod = @import("probe.zig");
const decode_workspace_mod = @import("decode_workspace.zig");
const descriptors_mod = @import("descriptors.zig");

pub const Context = struct {
    inst: vk.VkInstance = null,
    pd: vk.VkPhysicalDevice = null,
    dev: vk.VkDevice = null,
    queue: vk.VkQueue = null,
    qfi: u32 = 0,
    initialized: bool = false,

    // ── M8a: lazy-allocated dispatch + timing chassis ────────────────
    // Created on first `dispatch.submitOne` call so contexts that never
    // dispatch (probe-only callers, error paths between createInstance
    // and createDevice) don't pay the pool/fence/query-pool cost.
    // `dispatch.zig` is the sole owner — these fields are mutated only
    // from there. `deinit` calls `dispatch.releaseContextChassis` to
    // tear them down in the correct order (free cmdbuf → destroy pool +
    // fence + query pool) before destroyDevice.
    cmd_pool: vk.VkCommandPool = null,
    cmd_buf: vk.VkCommandBuffer = null,
    fence: vk.VkFence = null,
    query_pool: vk.VkQueryPool = null,
    timestamp_period_ns: f32 = 0.0,

    // ── M8c: probed sync2 capability (read-only after ensureInit) ────
    // sync.zig consults this to pick `vkCmdPipelineBarrier2` vs the
    // sync1 fallback. Populated from `probe.probe()` during ensureInit.
    // Default false keeps the sync1 fallback path safe on Contexts that
    // were never brought up via ensureInit (test code, error paths).
    has_synchronization2: bool = false,

    // ── Piece 2: VK_KHR_8bit_storage + shaderInt8 enabled at device
    // creation. Read by l1_codec to choose between the byte-typed Dst
    // fast-batch lz_decode shader and the u32-packed fallback. Mirrors
    // the probe.has_8bit_storage check that gated feature enablement.
    has_8bit_storage: bool = false,

    // ── Decode workspace pool — grow-only buffer pool reused across
    // every `decodeSlz1ToBytesEx` + `runDecodePipelineEx` invocation on
    // this context. Mirrors CUDA's `decode_context.DecodeContext` struct
    // (src/decode/decode_context.zig:188-375) which carries every
    // device buffer as a `(ptr, size)` field pair and grows lazily via
    // `ensureDeviceBuf` (src/decode/decode_context.zig:35). Before this
    // pool existed, the Vulkan decode path freed + reallocated 17
    // buffers per call (~34 ms / call alloc+free overhead in the Nsight
    // trace on NVIDIA enwik8).
    //
    // Allocated on first use (`getOrCreateDecodeWorkspace`) so contexts
    // that never decode (probe-only callers, encode-only callers) don't
    // pay the workspace heap cost. Freed once in `deinit` before
    // `destroyDevice`. The workspace is a separate heap allocation
    // because embedding it inline would inflate every Context (~17
    // Slots × ~64 bytes = ~1 KiB) for callers that never decode.
    decode_workspace: ?*decode_workspace_mod.DecodeWorkspace = null,

    // ── S001: process-lifetime pipeline cache for the decode path.
    // CUDA loads each kernel ONCE at process init (see
    // src/decode/module_loader.zig:140-141, 169-173 — cuModuleLoadData
    // + cuModuleGetFunction are called exactly once and the function
    // pointer is stored in a module-global slot for the lifetime of
    // the process). The Vulkan port previously declared a local
    // `var cache: descriptors.Cache = .{}` inside both decode entry
    // points and tore the whole thing down at the end of each call
    // via `defer descriptors.invalidateAll(...)`. That meant every
    // decode rebuilt 8 pipelines (vkCreateShaderModule +
    // vkCreateDescriptorSetLayout + vkCreatePipelineLayout +
    // vkCreateComputePipelines + vkCreateDescriptorPool, ×8 — a few
    // ms of wasted host work per call).
    //
    // The cache is created on first use by the decode path
    // (`getOrCreateDecodePipelineCache`) and torn down in `deinit`
    // before `destroyDevice`. It's stack-stored on Context (the
    // Cache struct is ~16 entries × ~40 bytes = ~640 bytes, well
    // under the heap-allocation threshold the workspace uses).
    decode_pipeline_cache: descriptors_mod.Cache = .{},
    decode_pipeline_cache_initialized: bool = false,
};

pub var g_default: Context = .{};

/// CLI-provided device selector, consulted by `ensureInit`. Default leaves
/// the historical behavior unchanged (first compute-capable device).
/// Callers set this BEFORE invoking `ensureInit` (the CLI does so after
/// parsing `--device <N|substring>`); test harnesses that want to pin a
/// device without touching CLI plumbing can also set
/// `SLZ_VK_DEVICE_INDEX=<N>` in the environment, which `ensureInit` reads
/// as a fallback when the in-process selector is still `.default`.
///
/// Lifetime note: `.by_name`'s slice MUST outlive `ensureInit`. The CLI
/// holds argv for the whole process, so passing a slice into argv there is
/// safe; tests building a synthetic selector should keep their own backing
/// buffer alive for the call.
pub var g_selector: device_mod.DeviceSelector = .default;

/// Convenience setter — keeps the global private when callers don't want
/// to import `device_mod` for the union tag names.
pub fn setSelector(s: device_mod.DeviceSelector) void {
    g_selector = s;
}

pub const DriverError = error{
    LoaderInitFailed,
} || instance_mod.InstanceError || device_mod.DeviceError;

/// Resolve the device selector that `ensureInit` should pass to
/// `pickPhysicalDeviceWith`. Priority:
///   1. Explicit `g_selector` (anything other than `.default`).
///   2. `SLZ_VK_DEVICE_INDEX=<N>` env var when set + parseable.
///   3. `.default` — historical "first compute-capable device" behavior.
fn resolveSelector() device_mod.DeviceSelector {
    if (g_selector != .default) return g_selector;
    const raw = std.c.getenv("SLZ_VK_DEVICE_INDEX") orelse return .default;
    const s = std.mem.span(raw);
    if (s.len == 0) return .default;
    const n = std.fmt.parseInt(u32, s, 10) catch return .default;
    return .{ .by_index = n };
}

/// Bring up the full Vulkan bootstrap into `g_default`. Safe to call any
/// number of times — subsequent calls after the first success are no-ops.
pub fn ensureInit() DriverError!void {
    if (g_default.initialized) return;

    if (!vk.init()) return error.LoaderInitFailed;

    // Validation default: off. Caller (CLI flag, env var elsewhere) flips
    // the want_validation argument; instance.zig still requires SLZ_VK_
    // VALIDATION=1 in the env as a belt-and-suspenders.
    const inst = try instance_mod.createInstance(true);
    errdefer instance_mod.destroyInstance(inst);

    const selector = resolveSelector();
    const pd = try device_mod.pickPhysicalDeviceWith(inst, selector);
    // Probe BEFORE createDevice so we can opt into VK_KHR_8bit_storage
    // when the device reports it. The decoder's fast-batch path needs
    // byte-typed Dst SSBO access, which only compiles when
    // storageBuffer8BitAccess + shaderInt8 are enabled on the device.
    const pr = probe_mod.probe(inst, pd);
    const want_8bit = pr.has_8bit_storage and pr.has_shader_int8;
    const bundle = try device_mod.createDevice(pd, .{ .enable_8bit_storage = want_8bit });
    errdefer device_mod.destroyDevice(bundle.dev);

    g_default = .{
        .inst = inst,
        .pd = pd,
        .dev = bundle.dev,
        .queue = bundle.queue,
        .qfi = bundle.queue_family_index,
        .initialized = true,
        .has_synchronization2 = pr.has_synchronization2,
        .has_8bit_storage = want_8bit,
    };
}

/// Lazily allocate the decode workspace pool on first use. Subsequent
/// calls return the existing pointer. Mirrors CUDA's pattern where the
/// `DecodeContext` is created once (per handle in the C ABI, or as the
/// `g_default` singleton in the CLI) and reused across every
/// `slzDecompress` call — see `src/decode/driver.zig` `g_default`.
///
/// Returns `error.OutOfMemory` if the workspace heap allocation fails;
/// callers route this through their existing error paths (the
/// `DecodeError`/`PipelineError`/`Slz1Error` sets all carry
/// `OutOfMemory`).
pub fn getOrCreateDecodeWorkspace(
    ctx: *Context,
) error{OutOfMemory}!*decode_workspace_mod.DecodeWorkspace {
    if (ctx.decode_workspace) |ws| return ws;
    const ws = std.heap.page_allocator.create(decode_workspace_mod.DecodeWorkspace) catch {
        return error.OutOfMemory;
    };
    ws.* = .{};
    ctx.decode_workspace = ws;
    return ws;
}

/// Lazily prepare the per-context decode pipeline cache. The cache
/// struct itself is value-stored on Context (no heap alloc here); this
/// function exists to (a) mark the cache as initialized so deinit knows
/// to drop it, and (b) give the call sites a single point that can
/// later grow extra setup (e.g. seeding from disk via pipeline_cache
/// after S002 lands).
///
/// Mirrors CUDA module_loader.init() (src/decode/module_loader.zig:135-148)
/// which loads every kernel once at process init.
pub fn getOrCreateDecodePipelineCache(ctx: *Context) *descriptors_mod.Cache {
    if (!ctx.decode_pipeline_cache_initialized) {
        ctx.decode_pipeline_cache = .{};
        ctx.decode_pipeline_cache_initialized = true;
    }
    return &ctx.decode_pipeline_cache;
}

pub fn deinit() void {
    if (!g_default.initialized) return;
    // Decode workspace teardown BEFORE destroyDevice — every VkBuffer +
    // VkDeviceMemory the pool owns is device-scoped. Frees nothing if
    // no decode ever ran (workspace pointer stays null).
    if (g_default.decode_workspace) |ws| {
        decode_workspace_mod.deinit(&g_default, ws);
        std.heap.page_allocator.destroy(ws);
        g_default.decode_workspace = null;
    }
    // S001: process-lifetime descriptor cache teardown BEFORE
    // destroyDevice — every VkPipeline / VkPipelineLayout /
    // VkDescriptorSetLayout / VkShaderModule / VkDescriptorPool it
    // holds is device-owned and the spec requires they be destroyed
    // before their parent VkDevice. invalidateAll is a no-op when the
    // cache was never used (no decode ran in this process).
    if (g_default.decode_pipeline_cache_initialized) {
        descriptors_mod.invalidateAll(&g_default, &g_default.decode_pipeline_cache);
        g_default.decode_pipeline_cache_initialized = false;
    }
    // M8a chassis teardown must happen BEFORE destroyDevice — command
    // pools, fences, and query pools are device-owned and the spec
    // requires they be destroyed before their parent VkDevice.
    dispatch_mod.releaseContextChassis(&g_default);
    device_mod.destroyDevice(g_default.dev);
    instance_mod.destroyInstance(g_default.inst);
    g_default = .{};
}
