//! M2: physical-device vendor + feature probe and tier classifier.
//!
//! Reads VkPhysicalDeviceProperties2 (with subgroup + subgroup-size-control
//! properties chained via pNext) and VkPhysicalDeviceFeatures2 (with the
//! Vulkan 1.2 + 1.3 omnibus feature structs + shaderSubgroupExtendedTypes
//! chained via pNext). Enumerates device extensions to detect
//! VK_NV_shader_subgroup_partitioned (the Tier-1+NV bonus).
//!
//! The Tier enum below maps 1:1 with the three committed SPIR-V variant
//! directories (tier1_nv/, tier1/, tier2/) plus an `unsupported` sentinel.
//! Classification rules are pulled from docs/vulkan_port_architecture.md §9.
//!
//! No allocations: the extension-name buffer caps at MAX_DEVICE_EXTS=512
//! which is well above any current driver's count (typical ~80–180).

const std = @import("std");

const vk = @import("vk_api.zig");

pub const Tier = enum { tier1_nv, tier1, tier2, unsupported };

pub const ProbeResult = struct {
    tier: Tier,
    vendor_id: u32,
    device_id: u32,
    /// Slice into the caller-provided `VkPhysicalDeviceProperties.deviceName`
    /// buffer (sliced at the first NUL). Stable as long as the props struct
    /// the result references stays alive — for the probe_main use the result
    /// is consumed immediately, so the inline storage here is fine.
    device_name: []const u8,
    api_version: u32,
    subgroup_size: u32,
    subgroup_size_min: u32,
    subgroup_size_max: u32,
    /// Bitmask of VkShaderStageFlagBits indicating which shader stages
    /// support setting a required subgroup size via
    /// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. We need
    /// VK_SHADER_STAGE_COMPUTE_BIT (0x20) set to pin compute pipelines.
    required_subgroup_size_stages: u32 = 0,
    has_subgroup_size_control: bool,
    has_buffer_device_address: bool,
    has_timeline_semaphore: bool,
    has_synchronization2: bool,
    has_shader_int64: bool,
    has_shader_int8: bool,
    has_8bit_storage: bool,
    has_nv_subgroup_partitioned: bool,

    /// The deviceName slice above points into this private buffer so the
    /// result is self-contained. `probe()` copies the driver-written name
    /// here and slices at the first NUL.
    name_storage: [vk.VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = @splat(0),
};

const NVIDIA_VENDOR_ID: u32 = 0x10DE;
const MAX_DEVICE_EXTS: u32 = 512;
const NV_SUBGROUP_PARTITIONED_EXT: []const u8 = "VK_NV_shader_subgroup_partitioned";

pub fn probe(inst: vk.VkInstance, pd: vk.VkPhysicalDevice) ProbeResult {
    _ = inst; // instance handle currently unused; kept for symmetry + future use.

    // Default result — every field set to a tier-2-or-worse sentinel so the
    // tail-return path is well-defined even when entry points are missing.
    var result: ProbeResult = .{
        .tier = .unsupported,
        .vendor_id = 0,
        .device_id = 0,
        .device_name = &.{},
        .api_version = 0,
        .subgroup_size = 0,
        .subgroup_size_min = 0,
        .subgroup_size_max = 0,
        .required_subgroup_size_stages = 0,
        .has_subgroup_size_control = false,
        .has_buffer_device_address = false,
        .has_timeline_semaphore = false,
        .has_synchronization2 = false,
        .has_shader_int64 = false,
        .has_shader_int8 = false,
        .has_8bit_storage = false,
        .has_nv_subgroup_partitioned = false,
    };

    // ── M1 scalar properties (vendor, api, name) ────────────────────
    // Always available — instance-bring-up failed if these were missing.
    const get_props = vk.vkGetPhysicalDeviceProperties_fn orelse return result;
    var m1_props: vk.VkPhysicalDeviceProperties = .{};
    get_props(pd, &m1_props);
    result.vendor_id = m1_props.vendorID;
    result.device_id = m1_props.deviceID;
    result.api_version = m1_props.apiVersion;
    // Copy deviceName into the result's own storage and slice at NUL.
    @memcpy(result.name_storage[0..], m1_props.deviceName[0..]);
    var n: usize = 0;
    while (n < result.name_storage.len and result.name_storage[n] != 0) : (n += 1) {}
    result.device_name = result.name_storage[0..n];

    // ── Vulkan 1.1 Properties2 chain (subgroup props) ───────────────
    // If the entry point or required structs are missing, leave the
    // subgroup_size fields at 0 and the tier falls to tier2/unsupported.
    if (vk.vkGetPhysicalDeviceProperties2_fn) |get_props2| {
        var sgsc_props: vk.VkPhysicalDeviceSubgroupSizeControlProperties = .{};
        var sg_props: vk.VkPhysicalDeviceSubgroupProperties = .{};
        // Chain order: Properties2.pNext → Subgroup → SubgroupSizeControl.
        sg_props.pNext = @ptrCast(&sgsc_props);
        var props2: vk.VkPhysicalDeviceProperties2 = .{};
        props2.pNext = @ptrCast(&sg_props);
        get_props2(pd, @ptrCast(&props2));
        result.subgroup_size = sg_props.subgroupSize;
        // SubgroupSizeControl props are zeroed if the device doesn't write
        // them (no extension/feature). We treat min==max==0 as "not present"
        // for the tier1 width check; the feature-side flag below is the
        // authoritative gate. Most real drivers DO write these regardless
        // (they're core in 1.3) but we never read min/max without the flag.
        result.subgroup_size_min = sgsc_props.minSubgroupSize;
        result.subgroup_size_max = sgsc_props.maxSubgroupSize;
        result.required_subgroup_size_stages = sgsc_props.requiredSubgroupSizeStages;
    }

    // ── Vulkan 1.2 + 1.3 Features2 chain ────────────────────────────
    // We chain BOTH the omnibus (v12/v13) AND the per-extension feature
    // structs. Reality: not every driver populates the omnibus structs
    // for promoted features (the v12 spec text obliges them to, but
    // empirically some drivers under-fill timelineSemaphore /
    // bufferDeviceAddress even when the per-extension struct reports
    // VK_TRUE). Accepting `omnibus OR per-extension` covers the gap.
    if (vk.vkGetPhysicalDeviceFeatures2_fn) |get_feats2| {
        var v13: vk.VkPhysicalDeviceVulkan13Features = .{};
        var v12: vk.VkPhysicalDeviceVulkan12Features = .{};
        var sgsc_feats: vk.VkPhysicalDeviceSubgroupSizeControlFeatures = .{};
        var ext_types: vk.VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures = .{};
        var bda_feats: vk.VkPhysicalDeviceBufferDeviceAddressFeatures = .{};
        var ts_feats: vk.VkPhysicalDeviceTimelineSemaphoreFeatures = .{};
        var sync2_feats: vk.VkPhysicalDeviceSynchronization2Features = .{};
        // Chain order (head→tail): Features2 → v12 → v13 → SubgroupSizeControl
        // → ExtendedTypes → BDA → TimelineSemaphore → Synchronization2.
        sync2_feats.pNext = null;
        ts_feats.pNext = @ptrCast(&sync2_feats);
        bda_feats.pNext = @ptrCast(&ts_feats);
        ext_types.pNext = @ptrCast(&bda_feats);
        sgsc_feats.pNext = @ptrCast(&ext_types);
        v13.pNext = @ptrCast(&sgsc_feats);
        v12.pNext = @ptrCast(&v13);
        var feats2: vk.VkPhysicalDeviceFeatures2 = .{};
        feats2.pNext = @ptrCast(&v12);
        get_feats2(pd, @ptrCast(&feats2));

        // shaderInt64 lives in the *core* features struct (Vulkan 1.0).
        result.has_shader_int64 = (feats2.features.shaderInt64 == vk.VK_TRUE);

        // Vulkan 1.2 / 1.3 promoted features — read both omnibus and
        // per-extension and OR them.
        result.has_buffer_device_address =
            (v12.bufferDeviceAddress == vk.VK_TRUE) or
            (bda_feats.bufferDeviceAddress == vk.VK_TRUE);
        result.has_timeline_semaphore =
            (v12.timelineSemaphore == vk.VK_TRUE) or
            (ts_feats.timelineSemaphore == vk.VK_TRUE);
        result.has_shader_int8 = (v12.shaderInt8 == vk.VK_TRUE);
        result.has_8bit_storage = (v12.storageBuffer8BitAccess == vk.VK_TRUE);
        result.has_subgroup_size_control =
            (v13.subgroupSizeControl == vk.VK_TRUE) or
            (sgsc_feats.subgroupSizeControl == vk.VK_TRUE);
        result.has_synchronization2 =
            (v13.synchronization2 == vk.VK_TRUE) or
            (sync2_feats.synchronization2 == vk.VK_TRUE);
    }

    // ── NV_subgroup_partitioned extension (Tier-1+NV bonus) ─────────
    if (vk.vkEnumerateDeviceExtensionProperties_fn) |enum_exts| {
        var count: u32 = 0;
        if (enum_exts(pd, null, &count, null) == vk.VK_SUCCESS and count > 0) {
            var capped = count;
            if (capped > MAX_DEVICE_EXTS) capped = MAX_DEVICE_EXTS;
            var exts: [MAX_DEVICE_EXTS]vk.VkExtensionProperties = @splat(.{});
            const r = enum_exts(pd, null, &capped, @ptrCast(&exts));
            if (r == vk.VK_SUCCESS or r == vk.VK_INCOMPLETE) {
                var i: u32 = 0;
                while (i < capped) : (i += 1) {
                    var name_len: usize = 0;
                    while (name_len < exts[i].extensionName.len and exts[i].extensionName[name_len] != 0) : (name_len += 1) {}
                    if (std.mem.eql(u8, exts[i].extensionName[0..name_len], NV_SUBGROUP_PARTITIONED_EXT)) {
                        result.has_nv_subgroup_partitioned = true;
                        break;
                    }
                }
            }
        }
    }

    // ── Tier classification (architecture §9) ───────────────────────
    // tier1_nv: NVIDIA + 1.3 + subgroupSize==32 + size_control (NV-partitioned
    //           presence is a bonus, NOT a gate — the tier1_nv SPIR-V uses
    //           the extension when present and falls back otherwise).
    // tier1:    portable Tier-1 — size_control with [min,max] including 32,
    //           BDA, shaderInt64, timeline semaphore.
    // tier2:    apiVersion >= 1.2 and a compute queue (compute queue
    //           presence already validated by device.zig::pickPhysicalDevice
    //           upstream; here we trust that gate).
    const api_major = vk.VK_API_VERSION_MAJOR(result.api_version);
    const api_minor = vk.VK_API_VERSION_MINOR(result.api_version);
    const is_at_least_1_3 = (api_major > 1) or (api_major == 1 and api_minor >= 3);
    const is_at_least_1_2 = (api_major > 1) or (api_major == 1 and api_minor >= 2);

    const includes_32 = result.has_subgroup_size_control and
        (result.subgroup_size_min <= 32) and (result.subgroup_size_max >= 32);

    if (result.vendor_id == NVIDIA_VENDOR_ID and
        is_at_least_1_3 and
        result.subgroup_size == 32 and
        result.has_subgroup_size_control)
    {
        result.tier = .tier1_nv;
    } else if (result.has_subgroup_size_control and
        includes_32 and
        result.has_buffer_device_address and
        result.has_shader_int64 and
        result.has_timeline_semaphore)
    {
        result.tier = .tier1;
    } else if (is_at_least_1_2) {
        result.tier = .tier2;
    } else {
        result.tier = .unsupported;
    }

    return result;
}

/// Human-readable tier name for log/print sites.
pub fn tierName(t: Tier) []const u8 {
    return switch (t) {
        .tier1_nv => "tier1_nv",
        .tier1 => "tier1",
        .tier2 => "tier2",
        .unsupported => "unsupported",
    };
}

// ── M8a: timestamp-period accessor ───────────────────────────────
// VkPhysicalDeviceLimits.timestampPeriod is the number of nanoseconds per
// timestamp tick (e.g. 1.0 on NVIDIA, ~83.333 on AMD GCN). dispatch.zig
// multiplies vkGetQueryPoolResults' raw u64 delta by this to get ns.
//
// We pull it from the existing M1 scalar-properties query rather than
// chaining via Properties2 — the field exists in core Vulkan 1.0 and
// every VkPhysicalDevice fills it, so the simpler one-shot call suffices.
//
// Implementation note: the `VkPhysicalDeviceProperties` struct in
// vk_api.zig represents VkPhysicalDeviceLimits + sparse + tail-pad as a
// single `limits_sparse_pad_opaque: [528]u8 align(8)` blob, so we have
// to read the f32 at the spec's known byte offset within that blob.
//
// Offset derivation: within VkPhysicalDeviceLimits, timestampPeriod is
// the 93rd field per the Vulkan 1.0 spec. Walking the field types in
// order (u32 × 11, u64 × 2, u32 × 16, u32 × 31 [vertex through draw],
// f32 × 2, u32 × 4, u64 × 4 [memory/buffer alignments], i32/u32 mix × 4,
// f32 × 2, u32 × 8, u32 × 6 [sample counts], u32, f32 [timestampPeriod])
// places timestampPeriod at byte 424 inside the limits struct, which
// equals byte 424 inside `limits_sparse_pad_opaque` since the latter
// starts at the limits struct's first byte.
//
// timestampValidBits handling (per-queue, not per-device) belongs to
// M8c — at M8a we trust the device sets enough bits to not wrap inside
// one dispatch (NVIDIA: 64 bits; AMD/Intel: 36-64 bits — a 36-bit window
// at 1ns is ~68 seconds, well above one kernel dispatch).
const TIMESTAMP_PERIOD_OFFSET: usize = 424;

/// Read VkPhysicalDeviceLimits.timestampPeriod for `pd`. Returns 1.0 as a
/// best-effort fallback when the properties slot isn't resolved — that
/// keeps ns math non-zero rather than NaN/Inf, and the timing numbers
/// surface as "raw ticks" which is still useful for relative comparisons.
pub fn readTimestampPeriod(pd: vk.VkPhysicalDevice) f32 {
    const get_props = vk.vkGetPhysicalDeviceProperties_fn orelse return 1.0;
    var p: vk.VkPhysicalDeviceProperties = .{};
    get_props(pd, &p);
    // `align(8)` on the blob + offset 424 (div-by-4) means the f32 here
    // has at least 4-byte alignment, which is f32's natural alignment.
    const raw: *align(4) const f32 = @ptrCast(@alignCast(&p.limits_sparse_pad_opaque[TIMESTAMP_PERIOD_OFFSET]));
    return raw.*;
}
