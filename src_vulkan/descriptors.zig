//! M8b: pipeline + descriptor-set factory.
//!
//! Owns the per-pipeline state every compute dispatch needs:
//!   • VkShaderModule           — created from a SPIR-V byte slice.
//!   • VkDescriptorSetLayout    — N storage-buffer bindings (binding=0..N-1).
//!   • VkPipelineLayout         — the set layout + optional push-constant range.
//!   • VkComputePipeline        — bound to the above layout, single "main" entry.
//!   • VkDescriptorPool         — sized to mint one set per `getOrCreate`
//!                                lifetime (cache cap = MAX_ENTRIES).
//!
//! The cache is a flat fixed-size array (cap = MAX_ENTRIES) keyed by
//! (kernel name, tier). LRU eviction when full — every hit bumps the
//! per-entry `last_used` counter to the cache's monotonically-increasing
//! `lru_counter`, and inserts under pressure pick the lowest. The 16-entry
//! cap is from the M8b spec — production milestones M11+ will likely raise
//! it once the kernel × spec-constant matrix explodes (lz_decode is the
//! same kernel at several spec-constant shapes; counting cache slots that
//! way pushes the cap above 16).
//!
//! Lifetime contract:
//!   • `getOrCreate` lazy-creates every Vulkan object on first call for a
//!     key. Repeat calls return the cached struct by value.
//!   • Eviction (cache full) frees the LRU entry's pipeline + layout +
//!     set layout + shader module via the M8b destroy entry points BEFORE
//!     reusing the slot.
//!   • `invalidateAll` is the explicit "drop everything" entry point —
//!     called from M8b cache-hit/miss instrumentation when the underlying
//!     VkPipelineCache changed (rebuilt/rotated) so the in-memory pipeline
//!     handles are no longer the canonical ones. Idempotent; cheap to call.
//!
//! `allocSet` allocates a fresh VkDescriptorSet from the cached entry's
//! VkDescriptorPool and immediately writes the caller-supplied buffer
//! infos into bindings 0..N-1. Returned set is valid until the next
//! `invalidateAll` or until the owning entry is evicted.
//!
//! Allocations: ZERO host-side dynamic allocations in this module — every
//! buffer (binding array, pool size array, write array) is stack-bounded
//! at MAX_SSBOS. Vulkan-side allocations are obviously NOT bounded; the
//! caller pays the VkAllocateMemory/Pool cost on the underlying objects.

const std = @import("std");

const vk = @import("vk_api.zig");
const driver_mod = @import("driver.zig");
const probe = @import("probe.zig");

// ── Constants ────────────────────────────────────────────────────

/// Cache cap per M8b spec. Lifted in M11+ once the kernel × spec-constant
/// fanout exceeds 16 distinct pipelines per device.
pub const MAX_ENTRIES: usize = 16;

/// Hard cap on storage-buffer bindings per kernel. The 17 production
/// kernels top out at 12 SSBOs (mergeHuffDescs); 16 is the conservative
/// stack-allocated upper bound. Bumping this changes only the stack
/// arrays in `getOrCreate` / `allocSet` — no API churn.
pub const MAX_SSBOS: u32 = 16;

/// Descriptor sets we'll allocate from each per-pipeline pool over the
/// lifetime of the cached entry. Each `submitOne` call allocates a fresh
/// set (no reuse across dispatches at M8b), so the pool is sized to
/// support `MAX_SETS_PER_POOL` concurrent in-flight sets. M8b uses
/// FREE_DESCRIPTOR_SET_BIT so we COULD free per-dispatch, but in practice
/// the caller drives one dispatch at a time and the pool resets at
/// eviction. 16 is comfortably above any single-frame dispatch count
/// the conformance harness drives.
const MAX_SETS_PER_POOL: u32 = 16;

// ── Errors ───────────────────────────────────────────────────────

pub const DescriptorError = error{
    LoaderNotReady,
    TooManyBindings,
    ShaderModuleCreateFailed,
    SetLayoutCreateFailed,
    PipelineLayoutCreateFailed,
    PipelineCreateFailed,
    DescriptorPoolCreateFailed,
    DescriptorSetAllocateFailed,
};

// ── PipelineKey + CachedPipeline ─────────────────────────────────

/// Cache key — `kernel` is a *static-lifetime* string (typically a
/// compile-time literal like "walk_frame") and is compared byte-wise.
/// Storing the slice directly is safe because every call site passes a
/// static slice; if that invariant ever changes, the cache will need to
/// duplicate the bytes into per-entry storage.
pub const PipelineKey = struct {
    kernel: []const u8,
    tier: probe.Tier,

    pub fn eql(a: PipelineKey, b: PipelineKey) bool {
        return a.tier == b.tier and std.mem.eql(u8, a.kernel, b.kernel);
    }
};

/// Bundle returned to the caller. Lives inside the cache; callers MUST
/// NOT free any of these handles — eviction / `invalidateAll` owns the
/// lifecycle.
pub const CachedPipeline = struct {
    pipeline: vk.VkPipeline,
    pipeline_layout: vk.VkPipelineLayout,
    set_layout: vk.VkDescriptorSetLayout,
    shader_module: vk.VkShaderModule,
    /// Per-pipeline descriptor pool. `allocSet` mints VkDescriptorSet
    /// handles from here. Sized at MAX_SETS_PER_POOL × n_storage_buffers
    /// when the entry is created; once exhausted, the next allocSet
    /// returns DescriptorSetAllocateFailed (caller should invalidateAll
    /// + retry — M11+ may grow the pool in place).
    pool: vk.VkDescriptorPool,
    /// Number of bindings in the set layout. Cached so `allocSet` can
    /// validate caller-supplied buffer-info slice length without
    /// re-reading the layout.
    n_storage_buffers: u32,
};

const Entry = struct {
    key: PipelineKey,
    value: CachedPipeline,
    last_used: u32,
};

/// Fixed-size cache. `lru_counter` increments on every hit (and on every
/// insert) so the slot with the lowest `last_used` is the LRU victim.
/// Wraparound is theoretically possible at 2^32 dispatches per process
/// lifetime but practically impossible (one dispatch every 1ns for ~4s
/// of continuous use); the bookkeeping is well-defined under wrap (smallest
/// `last_used` is still LRU modulo the comparison) but production
/// milestones may switch to a doubly-linked-list LRU for exactness.
pub const Cache = struct {
    entries: [MAX_ENTRIES]?Entry = @splat(null),
    lru_counter: u32 = 0,
};

// ── Function-pointer resolution ──────────────────────────────────

/// Resolve a device-level entry point; prefer vkGetDeviceProcAddr (one
/// fewer dispatch hop) but fall back to the instance-level thunk via
/// vkGetInstanceProcAddr for completeness. Mirror of dispatch.zig's
/// `resolveDeviceFn` — same rationale, same fallback chain.
fn resolveDeviceFn(comptime T: type, dev: vk.VkDevice, name: [*:0]const u8) ?T {
    if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
        if (gdpa(dev, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    if (vk.vkGetInstanceProcAddr_fn) |gipa| {
        const inst = driver_mod.g_default.inst;
        if (gipa(inst, name)) |raw| return @ptrCast(@alignCast(raw));
    }
    return null;
}

fn ensureFnSlots(dev: vk.VkDevice) DescriptorError!void {
    if (dev == null) return error.LoaderNotReady;

    if (vk.vkCreateDescriptorSetLayout_fn == null)
        vk.vkCreateDescriptorSetLayout_fn = resolveDeviceFn(vk.FnCreateDescriptorSetLayout, dev, "vkCreateDescriptorSetLayout");
    if (vk.vkDestroyDescriptorSetLayout_fn == null)
        vk.vkDestroyDescriptorSetLayout_fn = resolveDeviceFn(vk.FnDestroyDescriptorSetLayout, dev, "vkDestroyDescriptorSetLayout");
    if (vk.vkCreateDescriptorPool_fn == null)
        vk.vkCreateDescriptorPool_fn = resolveDeviceFn(vk.FnCreateDescriptorPool, dev, "vkCreateDescriptorPool");
    if (vk.vkDestroyDescriptorPool_fn == null)
        vk.vkDestroyDescriptorPool_fn = resolveDeviceFn(vk.FnDestroyDescriptorPool, dev, "vkDestroyDescriptorPool");
    if (vk.vkAllocateDescriptorSets_fn == null)
        vk.vkAllocateDescriptorSets_fn = resolveDeviceFn(vk.FnAllocateDescriptorSets, dev, "vkAllocateDescriptorSets");
    if (vk.vkResetDescriptorPool_fn == null)
        vk.vkResetDescriptorPool_fn = resolveDeviceFn(vk.FnResetDescriptorPool, dev, "vkResetDescriptorPool");
    if (vk.vkUpdateDescriptorSets_fn == null)
        vk.vkUpdateDescriptorSets_fn = resolveDeviceFn(vk.FnUpdateDescriptorSets, dev, "vkUpdateDescriptorSets");
    if (vk.vkCreatePipelineLayout_fn == null)
        vk.vkCreatePipelineLayout_fn = resolveDeviceFn(vk.FnCreatePipelineLayout, dev, "vkCreatePipelineLayout");
    if (vk.vkDestroyPipelineLayout_fn == null)
        vk.vkDestroyPipelineLayout_fn = resolveDeviceFn(vk.FnDestroyPipelineLayout, dev, "vkDestroyPipelineLayout");
    if (vk.vkCreateComputePipelines_fn == null)
        vk.vkCreateComputePipelines_fn = resolveDeviceFn(vk.FnCreateComputePipelines, dev, "vkCreateComputePipelines");
    if (vk.vkDestroyPipeline_fn == null)
        vk.vkDestroyPipeline_fn = resolveDeviceFn(vk.FnDestroyPipeline, dev, "vkDestroyPipeline");
    if (vk.vkCreateShaderModule_fn == null)
        vk.vkCreateShaderModule_fn = resolveDeviceFn(vk.FnCreateShaderModule, dev, "vkCreateShaderModule");
    if (vk.vkDestroyShaderModule_fn == null)
        vk.vkDestroyShaderModule_fn = resolveDeviceFn(vk.FnDestroyShaderModule, dev, "vkDestroyShaderModule");
}

// ── Internal: build one Vulkan-side pipeline bundle ──────────────

/// Owns the construction of a fresh CachedPipeline. errdefer chain
/// unwinds each successfully-created child object so partial failure
/// doesn't leak. NOT a public function — `getOrCreate` is the only
/// caller and it handles cache insertion atomically.
fn buildPipeline(
    ctx: *driver_mod.Context,
    spv_bytes: []const u8,
    n_storage_buffers: u32,
    push_const_size: u32,
) DescriptorError!CachedPipeline {
    if (n_storage_buffers > MAX_SSBOS) return error.TooManyBindings;

    // 1. VkShaderModule from the SPV byte slice. The spec requires
    //    codeSize to be a multiple of 4 and pCode 4-byte aligned; the
    //    committed .spv blobs satisfy both (glslc emits 4-byte words).
    //    We re-cast as `[*]const u32` for the create-info pointer.
    const create_sm = vk.vkCreateShaderModule_fn orelse return error.LoaderNotReady;
    if (spv_bytes.len % 4 != 0) return error.ShaderModuleCreateFailed;
    const code_words: [*]const u32 = @ptrCast(@alignCast(spv_bytes.ptr));
    const sm_ci: vk.VkShaderModuleCreateInfo = .{
        .codeSize = spv_bytes.len,
        .pCode = code_words,
    };
    var shader_module: vk.VkShaderModule = null;
    if (create_sm(ctx.dev, &sm_ci, null, &shader_module) != vk.VK_SUCCESS) {
        return error.ShaderModuleCreateFailed;
    }
    errdefer if (vk.vkDestroyShaderModule_fn) |destroy| destroy(ctx.dev, shader_module, null);

    // 2. VkDescriptorSetLayout — N storage-buffer bindings at bindings
    //    0..N-1, all visible to the compute stage. Stack-allocated array
    //    sized at MAX_SSBOS; bindingCount slices it to n.
    const create_dsl = vk.vkCreateDescriptorSetLayout_fn orelse return error.LoaderNotReady;
    var bindings: [MAX_SSBOS]vk.VkDescriptorSetLayoutBinding = @splat(.{});
    var b: u32 = 0;
    while (b < n_storage_buffers) : (b += 1) {
        bindings[b] = .{
            .binding = b,
            .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    const dsl_ci: vk.VkDescriptorSetLayoutCreateInfo = .{
        .bindingCount = n_storage_buffers,
        .pBindings = if (n_storage_buffers > 0) @ptrCast(&bindings) else null,
    };
    var set_layout: vk.VkDescriptorSetLayout = null;
    if (create_dsl(ctx.dev, &dsl_ci, null, &set_layout) != vk.VK_SUCCESS) {
        return error.SetLayoutCreateFailed;
    }
    errdefer if (vk.vkDestroyDescriptorSetLayout_fn) |destroy| destroy(ctx.dev, set_layout, null);

    // 3. VkPipelineLayout — wraps the set layout + optional push-constant
    //    range. push_const_size==0 means "no push range" — we pass count=0
    //    so the spec doesn't require a (otherwise valid) zero-size range.
    const create_pl = vk.vkCreatePipelineLayout_fn orelse return error.LoaderNotReady;
    const push_range: vk.VkPushConstantRange = .{
        .stageFlags = vk.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_const_size,
    };
    const set_layouts: [1]vk.VkDescriptorSetLayout = .{set_layout};
    const pl_ci: vk.VkPipelineLayoutCreateInfo = .{
        .setLayoutCount = if (n_storage_buffers > 0) 1 else 0,
        .pSetLayouts = if (n_storage_buffers > 0) @ptrCast(&set_layouts) else null,
        .pushConstantRangeCount = if (push_const_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_const_size > 0) @ptrCast(&push_range) else null,
    };
    var pipeline_layout: vk.VkPipelineLayout = null;
    if (create_pl(ctx.dev, &pl_ci, null, &pipeline_layout) != vk.VK_SUCCESS) {
        return error.PipelineLayoutCreateFailed;
    }
    errdefer if (vk.vkDestroyPipelineLayout_fn) |destroy| destroy(ctx.dev, pipeline_layout, null);

    // 4. VkComputePipeline. Entry point name is hard-coded "main" — every
    //    committed .spv emits "main" per glslc's default. Specialization
    //    info is null at M8b (M11+ adds spec-constant overrides). Pipeline
    //    cache argument is the per-Context pipeline cache when available;
    //    otherwise we pass NULL_HANDLE and the driver builds without
    //    persistence. Either way the dispatch works.
    //
    // Chain VkPipelineShaderStageRequiredSubgroupSizeCreateInfo with
    // requiredSubgroupSize=32 onto the stage's pNext. Critical because:
    //   - Every L1 compute shader is built around WARP_SIZE=32 (the
    //     CUDA-style 32-stride match extension, 32-wide subgroupBallot,
    //     32-token decoder fast batch, …).
    //   - Intel UHD's default subgroupSize is 16 (= SIMD8 lane pairs),
    //     splitting the 32-wide workgroup into TWO subgroups whose
    //     ballots are independent. Lanes 16..31 silently form a second
    //     subgroup whose ballot result never reaches the first, so the
    //     warp-parallel match extension misses bytes 20..31 and walks
    //     past the real match boundary.
    //   - VkPhysicalDeviceSubgroupSizeControlFeatures.subgroupSizeControl
    //     is enabled in device.zig::createDevice. The
    //     VK_EXT_subgroup_size_control extension is requested there too.
    // Pin requiredSubgroupSize=32 via VkPipelineShaderStageRequired
    // SubgroupSizeCreateInfo (promoted in Vulkan 1.3 from
    // VK_EXT_subgroup_size_control). Belt-and-suspenders:
    //   - The host enables `subgroupSizeControl` in BOTH the v13 omnibus
    //     and the per-extension feature struct (device.zig).
    //   - We also set REQUIRE_FULL_SUBGROUPS_BIT on the stage flags so
    //     drivers that gate the pin on full-subgroups (Intel Anv at
    //     some driver versions) still honor it.
    // HISTORICAL NOTE (kept for context — see git history): the dev-box
    // Intel UHD iGPU was previously reporting gl_SubgroupSize == 16
    // inside compute shaders despite the requiredSubgroupSize=32 pin.
    // Root cause: the REQUIRE_FULL_SUBGROUPS flag below was set to 0x4
    // instead of the spec-defined 0x2, so the flag bit Vulkan saw was
    // a reserved zero-effect bit — Intel was therefore free to split
    // the 32-wide workgroup into 2 × SIMD16 subgroups. Fixing the
    // constant to 0x2 makes Intel honor the pin and report
    // gl_SubgroupSize=32 / gl_NumSubgroups=1, which is what every
    // subgroupBallot / subgroupShuffle in lz_encode.comp assumes.
    //
    // Spec reference (vulkan_core.h, post-promotion of
    // VK_EXT_subgroup_size_control to 1.3):
    //   VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 0x2
    const sgsc_info: vk.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = .{
        .sType = vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = null,
        .requiredSubgroupSize = 32,
    };
    const STAGE_FLAG_REQUIRE_FULL_SUBGROUPS: u32 = 0x2;
    const create_cp = vk.vkCreateComputePipelines_fn orelse return error.LoaderNotReady;
    var cp_cis: [1]vk.VkComputePipelineCreateInfo = .{.{
        .stage = .{
            .pNext = @ptrCast(&sgsc_info),
            .flags = STAGE_FLAG_REQUIRE_FULL_SUBGROUPS,
            .stage = vk.VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader_module,
            .pName = "main",
            .pSpecializationInfo = null,
        },
        .layout = pipeline_layout,
    }};
    var pipeline: vk.VkPipeline = null;
    if (create_cp(
        ctx.dev,
        null, // pipeline cache — M7 wires this into the Context in a follow-up.
        1,
        @ptrCast(&cp_cis),
        null,
        @ptrCast(&pipeline),
    ) != vk.VK_SUCCESS) {
        return error.PipelineCreateFailed;
    }
    errdefer if (vk.vkDestroyPipeline_fn) |destroy| destroy(ctx.dev, pipeline, null);

    // 5. VkDescriptorPool — sized to mint MAX_SETS_PER_POOL sets, each
    //    consuming `n_storage_buffers` storage-buffer descriptors. If
    //    n==0 we still create a pool with zero descriptors so allocSet's
    //    early-return path can stay uniform (no special-casing the empty
    //    layout). Pools with maxSets > 0 and no pool sizes are spec-legal
    //    (the set itself counts even if it carries zero descriptors).
    const create_pool = vk.vkCreateDescriptorPool_fn orelse return error.LoaderNotReady;
    const pool_sizes: [1]vk.VkDescriptorPoolSize = .{.{
        .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = MAX_SETS_PER_POOL * n_storage_buffers,
    }};
    const pool_ci: vk.VkDescriptorPoolCreateInfo = .{
        .flags = vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = MAX_SETS_PER_POOL,
        .poolSizeCount = if (n_storage_buffers > 0) 1 else 0,
        .pPoolSizes = if (n_storage_buffers > 0) @ptrCast(&pool_sizes) else null,
    };
    var pool: vk.VkDescriptorPool = null;
    if (create_pool(ctx.dev, &pool_ci, null, &pool) != vk.VK_SUCCESS) {
        return error.DescriptorPoolCreateFailed;
    }

    return .{
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .set_layout = set_layout,
        .shader_module = shader_module,
        .pool = pool,
        .n_storage_buffers = n_storage_buffers,
    };
}

/// Tear down every Vulkan-side child of a CachedPipeline in spec-required
/// reverse-creation order. Safe to call on a partially-zeroed entry (every
/// destroy is null-guarded). Used by both `invalidateAll` and the eviction
/// path inside `getOrCreate`.
fn destroyEntry(ctx: *driver_mod.Context, value: CachedPipeline) void {
    if (ctx.dev == null) return;
    if (value.pool != null) {
        if (vk.vkDestroyDescriptorPool_fn) |destroy| destroy(ctx.dev, value.pool, null);
    }
    if (value.pipeline != null) {
        if (vk.vkDestroyPipeline_fn) |destroy| destroy(ctx.dev, value.pipeline, null);
    }
    if (value.pipeline_layout != null) {
        if (vk.vkDestroyPipelineLayout_fn) |destroy| destroy(ctx.dev, value.pipeline_layout, null);
    }
    if (value.set_layout != null) {
        if (vk.vkDestroyDescriptorSetLayout_fn) |destroy| destroy(ctx.dev, value.set_layout, null);
    }
    if (value.shader_module != null) {
        if (vk.vkDestroyShaderModule_fn) |destroy| destroy(ctx.dev, value.shader_module, null);
    }
}

// ── Public API ───────────────────────────────────────────────────

/// Return a cached pipeline bundle for the (kernel, tier) key, building
/// + caching it if not already present. Eviction (LRU) when the cache
/// is full.
///
/// `spv_bytes` must be a 4-byte-multiple SPIR-V byte slice (validated).
/// `n_storage_buffers` is the binding count for the descriptor set layout;
/// kernels with no SSBOs (the M5 match_any benchmark) pass 0.
/// `push_const_size` is the push-constant range size in bytes; pass 0 if
/// the kernel uses no push constants. Caps at 128 bytes per arch §6.3.
pub fn getOrCreate(
    ctx: *driver_mod.Context,
    cache: *Cache,
    kernel: []const u8,
    tier: probe.Tier,
    spv_bytes: []const u8,
    n_storage_buffers: u32,
    push_const_size: u32,
) DescriptorError!CachedPipeline {
    if (!ctx.initialized or ctx.dev == null) return error.LoaderNotReady;
    try ensureFnSlots(ctx.dev);

    const key: PipelineKey = .{ .kernel = kernel, .tier = tier };

    // Hit path — linear scan over MAX_ENTRIES is fine at 16 entries.
    var i: usize = 0;
    while (i < MAX_ENTRIES) : (i += 1) {
        if (cache.entries[i]) |*e| {
            if (e.key.eql(key)) {
                cache.lru_counter +%= 1;
                e.last_used = cache.lru_counter;
                return e.value;
            }
        }
    }

    // Miss path. Build first (so on construction failure the cache stays
    // unchanged — callers can retry without observing a torn entry).
    const built = try buildPipeline(ctx, spv_bytes, n_storage_buffers, push_const_size);

    // Find target slot: prefer the first empty slot; otherwise evict the
    // entry with the smallest `last_used`.
    var slot: usize = MAX_ENTRIES;
    var oldest_used: u32 = std.math.maxInt(u32);
    i = 0;
    while (i < MAX_ENTRIES) : (i += 1) {
        if (cache.entries[i] == null) {
            slot = i;
            break;
        }
        // Track the LRU candidate while we scan in case no empty slot exists.
        const e = cache.entries[i].?;
        if (e.last_used < oldest_used) {
            oldest_used = e.last_used;
            slot = i;
        }
    }
    std.debug.assert(slot < MAX_ENTRIES);

    // If the chosen slot is occupied, evict — free the underlying Vulkan
    // objects before reusing.
    if (cache.entries[slot]) |evicted| {
        destroyEntry(ctx, evicted.value);
        cache.entries[slot] = null;
    }

    cache.lru_counter +%= 1;
    cache.entries[slot] = .{
        .key = key,
        .value = built,
        .last_used = cache.lru_counter,
    };
    return built;
}

/// Reset every cached entry's descriptor pool (vkResetDescriptorPool) —
/// frees every VkDescriptorSet previously minted by `allocSet` but keeps
/// the underlying VkPipeline / VkPipelineLayout / VkDescriptorSetLayout /
/// VkShaderModule / VkDescriptorPool alive for reuse.
///
/// Mirrors CUDA's per-call kernel-arg pattern: in CUDA each
/// `cuLaunchKernel` re-supplies the argument list (= "fresh descriptor
/// set" in VK terms) with zero allocator cost because args go through
/// the kernel function pointer's call ABI rather than a descriptor pool.
/// Vulkan has no equivalent, so the closest port-faithful pattern is to
/// reset the pool (constant-time, no allocator traffic) at every decode
/// entry and re-allocate the fresh sets we need for this call.
///
/// S001 needed this: with the cache promoted to process-lifetime, the
/// per-pipeline descriptor pool (sized at MAX_SETS_PER_POOL = 16) would
/// otherwise fill after ~16 / sets-per-call decode invocations and
/// vkAllocateDescriptorSets would start returning OUT_OF_POOL_MEMORY.
///
/// Safe on an empty / never-populated cache (every empty slot is
/// skipped).
pub fn resetAllPools(ctx: *driver_mod.Context, cache: *Cache) void {
    if (ctx.dev == null) return;
    const reset_pool = vk.vkResetDescriptorPool_fn orelse return;
    var i: usize = 0;
    while (i < MAX_ENTRIES) : (i += 1) {
        if (cache.entries[i]) |e| {
            if (e.value.pool != null) {
                _ = reset_pool(ctx.dev, e.value.pool, 0);
            }
        }
    }
}

/// Drop every cached entry. Destroys the Vulkan-side children of each
/// entry in spec-required reverse-creation order, then zeros the slot.
/// Safe on an empty cache.
///
/// Called from M8c's pipeline-cache rotation hook (when the on-disk
/// VkPipelineCache changes, every in-memory VkPipeline is stale) and from
/// the eventual driver.deinit teardown path.
pub fn invalidateAll(ctx: *driver_mod.Context, cache: *Cache) void {
    var i: usize = 0;
    while (i < MAX_ENTRIES) : (i += 1) {
        if (cache.entries[i]) |e| {
            destroyEntry(ctx, e.value);
            cache.entries[i] = null;
        }
    }
    cache.lru_counter = 0;
}

/// Allocate a fresh VkDescriptorSet from the cached entry's pool and
/// write the caller-supplied buffer infos into bindings 0..n-1.
///
/// `buffers.len` MUST equal `cached.n_storage_buffers` — caller bug
/// otherwise. We assert that in debug and clamp in release (writes are
/// per-binding so a short slice means the high bindings stay unbound,
/// which the driver will catch on dispatch).
///
/// Returned set lifetime: valid until the next `invalidateAll` or until
/// the entry's pool is exhausted (MAX_SETS_PER_POOL allocations).
pub fn allocSet(
    ctx: *driver_mod.Context,
    cached: CachedPipeline,
    buffers: []const vk.VkDescriptorBufferInfo,
) DescriptorError!vk.VkDescriptorSet {
    if (ctx.dev == null) return error.LoaderNotReady;
    const alloc_sets = vk.vkAllocateDescriptorSets_fn orelse return error.LoaderNotReady;
    const update_sets = vk.vkUpdateDescriptorSets_fn orelse return error.LoaderNotReady;
    std.debug.assert(buffers.len == cached.n_storage_buffers);

    const set_layouts: [1]vk.VkDescriptorSetLayout = .{cached.set_layout};
    const alloc_ci: vk.VkDescriptorSetAllocateInfo = .{
        .descriptorPool = cached.pool,
        .descriptorSetCount = 1,
        .pSetLayouts = @ptrCast(&set_layouts),
    };
    var sets: [1]vk.VkDescriptorSet = .{null};
    if (alloc_sets(ctx.dev, &alloc_ci, @ptrCast(&sets)) != vk.VK_SUCCESS) {
        return error.DescriptorSetAllocateFailed;
    }

    // Write each binding. Stack-allocate the writes array sized at
    // MAX_SSBOS; we slice it to `n` for the actual call. Skipping the
    // call entirely when n==0 keeps validation chatter quiet on the
    // empty-set path (the M5 match_any pipeline shape).
    const n: u32 = cached.n_storage_buffers;
    if (n > 0) {
        var writes: [MAX_SSBOS]vk.VkWriteDescriptorSet = @splat(.{});
        var b: u32 = 0;
        while (b < n) : (b += 1) {
            // pBufferInfo points at the caller's slice — the driver
            // copies the descriptor contents during vkUpdateDescriptorSets
            // so the slice does not need to outlive this call.
            writes[b] = .{
                .dstSet = sets[0],
                .dstBinding = b,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = @ptrCast(&buffers[b]),
            };
        }
        update_sets(ctx.dev, n, @ptrCast(&writes), 0, null);
    }
    return sets[0];
}

// ── Tests ────────────────────────────────────────────────────────
// Pure-cache-bookkeeping tests only — getOrCreate / allocSet need a real
// VkDevice and live in the M8b integration suite (alongside the smoke
// test that confirms the entire descriptor → dispatch path round-trips).

test "Cache starts empty" {
    const cache: Cache = .{};
    for (cache.entries) |e| try std.testing.expect(e == null);
    try std.testing.expectEqual(@as(u32, 0), cache.lru_counter);
}

test "PipelineKey eql by name + tier" {
    const a: PipelineKey = .{ .kernel = "walk_frame", .tier = .tier1 };
    const b: PipelineKey = .{ .kernel = "walk_frame", .tier = .tier1 };
    const c: PipelineKey = .{ .kernel = "walk_frame", .tier = .tier2 };
    const d: PipelineKey = .{ .kernel = "lz_decode", .tier = .tier1 };
    try std.testing.expect(a.eql(b));
    try std.testing.expect(!a.eql(c));
    try std.testing.expect(!a.eql(d));
}
