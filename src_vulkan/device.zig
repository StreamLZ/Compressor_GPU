//! Physical-device selection + logical-device creation. M1 picks the first
//! VkPhysicalDevice that exposes a queue family with VK_QUEUE_COMPUTE_BIT
//! and creates a logical device with exactly one queue from that family.
//!
//! Later milestones replace `pickPhysicalDevice` with a vendor-aware
//! scorer (Tier-1 NVIDIA/AMD/Intel preference; subgroupSize-32 gate) and
//! `createDevice` with feature-chain enablement (Vulkan 1.2/1.3 features,
//! BDA, sync2, subgroupSizeControl, ...). For M1 we only need the queue.

const std = @import("std");

const vk = @import("vk_api.zig");

pub const DeviceError = error{
    NoVulkanDevice,
    NoComputeQueueFamily,
    EnumerateFailed,
    CreateDeviceFailed,
    LoaderNotReady,
    DeviceIndexOutOfRange,
    DeviceNameNotFound,
    DeviceNameAmbiguous,
};

const MAX_PHYSICAL_DEVICES: u32 = 16;
const MAX_QUEUE_FAMILIES: u32 = 16;

/// Selector passed to pickPhysicalDevice. The CLI/driver layer translates
/// the user's `--device <N>` or `--device <substring>` (or
/// SLZ_VK_DEVICE_INDEX env var) into one of these variants. Default keeps
/// the historical behavior of returning the first compute-capable device,
/// so callers that don't care never need to touch it.
pub const DeviceSelector = union(enum) {
    default,
    by_index: u32,
    by_name: []const u8,
};

fn hasComputeQueue(pd: vk.VkPhysicalDevice) bool {
    const get_qf = vk.vkGetPhysicalDeviceQueueFamilyProperties_fn orelse return false;
    var qf_count: u32 = 0;
    get_qf(pd, &qf_count, null);
    if (qf_count == 0) return false;
    if (qf_count > MAX_QUEUE_FAMILIES) qf_count = MAX_QUEUE_FAMILIES;
    var qfs: [MAX_QUEUE_FAMILIES]vk.VkQueueFamilyProperties = @splat(.{});
    get_qf(pd, &qf_count, @ptrCast(&qfs));
    var qi: u32 = 0;
    while (qi < qf_count) : (qi += 1) {
        if ((qfs[qi].queueFlags & vk.VK_QUEUE_COMPUTE_BIT) != 0 and qfs[qi].queueCount > 0) {
            return true;
        }
    }
    return false;
}

fn deviceNameCopy(pd: vk.VkPhysicalDevice, buf: []u8) []const u8 {
    const get_props = vk.vkGetPhysicalDeviceProperties_fn orelse return &.{};
    var props: vk.VkPhysicalDeviceProperties = .{};
    get_props(pd, &props);
    var n: usize = 0;
    while (n < props.deviceName.len and n < buf.len and props.deviceName[n] != 0) : (n += 1) {
        buf[n] = props.deviceName[n];
    }
    return buf[0..n];
}

fn asciiEqIgnoreCase(a: u8, b: u8) bool {
    const al = if (a >= 'A' and a <= 'Z') a + 32 else a;
    const bl = if (b >= 'A' and b <= 'Z') b + 32 else b;
    return al == bl;
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (!asciiEqIgnoreCase(haystack[i + j], needle[j])) break;
        }
        if (j == needle.len) return true;
    }
    return false;
}

pub fn pickPhysicalDevice(inst: vk.VkInstance) DeviceError!vk.VkPhysicalDevice {
    return pickPhysicalDeviceWith(inst, .default);
}

pub fn pickPhysicalDeviceWith(inst: vk.VkInstance, selector: DeviceSelector) DeviceError!vk.VkPhysicalDevice {
    const enumerate = vk.vkEnumeratePhysicalDevices_fn orelse return error.LoaderNotReady;

    var count: u32 = 0;
    if (enumerate(inst, &count, null) != vk.VK_SUCCESS) return error.EnumerateFailed;
    if (count == 0) return error.NoVulkanDevice;
    if (count > MAX_PHYSICAL_DEVICES) count = MAX_PHYSICAL_DEVICES;

    var devices: [MAX_PHYSICAL_DEVICES]vk.VkPhysicalDevice = @splat(null);
    const r = enumerate(inst, &count, @ptrCast(&devices));
    // VK_INCOMPLETE is acceptable — we capped the count and got the first
    // `count` devices, which is exactly what we want.
    if (r != vk.VK_SUCCESS and r != vk.VK_INCOMPLETE) return error.EnumerateFailed;

    switch (selector) {
        .default => {
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const pd = devices[i];
                if (pd == null) continue;
                if (hasComputeQueue(pd)) return pd;
            }
            return error.NoComputeQueueFamily;
        },
        .by_index => |idx| {
            if (idx >= count) return error.DeviceIndexOutOfRange;
            const pd = devices[idx];
            if (pd == null) return error.DeviceIndexOutOfRange;
            if (!hasComputeQueue(pd)) return error.NoComputeQueueFamily;
            return pd;
        },
        .by_name => |needle| {
            var match: vk.VkPhysicalDevice = null;
            var match_count: u32 = 0;
            var name_buf: [vk.VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = @splat(0);
            var i: u32 = 0;
            while (i < count) : (i += 1) {
                const pd = devices[i];
                if (pd == null) continue;
                const name = deviceNameCopy(pd, name_buf[0..]);
                if (containsIgnoreCase(name, needle)) {
                    match = pd;
                    match_count += 1;
                }
            }
            if (match_count == 0) return error.DeviceNameNotFound;
            if (match_count > 1) return error.DeviceNameAmbiguous;
            if (!hasComputeQueue(match)) return error.NoComputeQueueFamily;
            return match;
        },
    }
}

pub const DeviceBundle = struct {
    dev: vk.VkDevice,
    queue: vk.VkQueue,
    queue_family_index: u32,
};

pub const DeviceCreateOptions = struct {
    /// Enable VK_KHR_8bit_storage + Vulkan12Features.{storageBuffer8BitAccess,
    /// shaderInt8}. Required by the optimized lz_decode.comp fast-batch path
    /// (which reads/writes Dst as `uint8_t bytes[]` to avoid the per-byte
    /// RMW pattern that broke the original 32-token batch).
    enable_8bit_storage: bool = false,
};

pub fn createDevice(pd: vk.VkPhysicalDevice, opts: DeviceCreateOptions) DeviceError!DeviceBundle {
    const get_qf = vk.vkGetPhysicalDeviceQueueFamilyProperties_fn orelse return error.LoaderNotReady;
    const create = vk.vkCreateDevice_fn orelse return error.LoaderNotReady;
    const get_queue = vk.vkGetDeviceQueue_fn orelse return error.LoaderNotReady;

    var qf_count: u32 = 0;
    get_qf(pd, &qf_count, null);
    if (qf_count == 0) return error.NoComputeQueueFamily;
    if (qf_count > MAX_QUEUE_FAMILIES) qf_count = MAX_QUEUE_FAMILIES;
    var qfs: [MAX_QUEUE_FAMILIES]vk.VkQueueFamilyProperties = @splat(.{});
    get_qf(pd, &qf_count, @ptrCast(&qfs));

    var qfi: u32 = std.math.maxInt(u32);
    var i: u32 = 0;
    while (i < qf_count) : (i += 1) {
        if ((qfs[i].queueFlags & vk.VK_QUEUE_COMPUTE_BIT) != 0 and qfs[i].queueCount > 0) {
            qfi = i;
            break;
        }
    }
    if (qfi == std.math.maxInt(u32)) return error.NoComputeQueueFamily;

    // Single queue, priority 1.0. pQueuePriorities must point to an array
    // of `queueCount` floats — one element since we ask for one queue.
    const priority: [1]f32 = .{1.0};
    const queue_ci: VkDeviceQueueCreateInfoLocal = .{
        .queueFamilyIndex = qfi,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    // Use the public alias so the field name matches what's actually
    // declared in vk_api.zig (extern struct layout-stability matters here).
    const queue_ci_pub: vk.VkDeviceQueueCreateInfo = .{
        .queueFamilyIndex = queue_ci.queueFamilyIndex,
        .queueCount = queue_ci.queueCount,
        .pQueuePriorities = queue_ci.pQueuePriorities,
    };
    // Enable VK_KHR_8bit_storage (promoted in 1.2 but the extension string
    // is still accepted) when the codec asks for it. The shaderInt8 +
    // storageBuffer8BitAccess feature bits chained via pNext are the
    // actual gate; the extension name keeps drivers happy on the off
    // chance one only honors the extension form.
    //
    // ALSO: subgroupSizeControl is enabled unconditionally so the
    // pipeline-creation path can pin requiredSubgroupSize=32 via
    // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Without this
    // feature enabled the driver is free to pick any supported subgroup
    // size; Intel UHD picks 16 (SIMD8 pairs) which silently breaks every
    // shader that assumes WARP_SIZE=32 (the warp-cooperative match
    // extension in lz_encode, the 32-token fast batch in lz_decode,
    // etc.). The feature is core in Vulkan 1.3 and widely supported on
    // 1.2 via VK_EXT_subgroup_size_control — failing to enable here is
    // safe because pinning fails only when the device doesn't advertise
    // the feature (in which case we run on whatever the driver picked,
    // matching pre-fix behavior).
    var v12_feats: vk.VkPhysicalDeviceVulkan12Features = .{};
    v12_feats.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    // Phase 2 (L1 finish — TODO A2): enable VK_KHR_buffer_device_address.
    // Caller-side D2D buffers register via `slzRegisterBuffer_vk` and
    // hand the BDA u64 back to the codec as the value passed to
    // `slzCompress_vk(d_input)` / `slzDecompress_vk(d_output)`. The
    // codec uses the address as a registry key to find the caller's
    // VkBuffer and binds it directly into the descriptor set, skipping
    // the internal HOST_VISIBLE staging copy. Buffers must also be
    // created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT for the
    // BDA query to succeed.
    v12_feats.bufferDeviceAddress = vk.VK_TRUE;
    // Chain BOTH the v13 omnibus AND the per-extension SubgroupSizeControl
    // features struct. Per spec, drivers should look at either form, but
    // some drivers only look at one — sending both is the belt-and-
    // suspenders pattern we use elsewhere (see probe.zig).
    var sgsc_feats: vk.VkPhysicalDeviceSubgroupSizeControlFeatures = .{};
    sgsc_feats.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    sgsc_feats.pNext = null;
    sgsc_feats.subgroupSizeControl = vk.VK_TRUE;
    sgsc_feats.computeFullSubgroups = vk.VK_TRUE;
    var v13_feats: vk.VkPhysicalDeviceVulkan13Features = .{};
    v13_feats.sType = vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    v13_feats.pNext = @ptrCast(&sgsc_feats);
    v13_feats.subgroupSizeControl = vk.VK_TRUE;
    v13_feats.computeFullSubgroups = vk.VK_TRUE;
    // The v12 omnibus is always chained now (was only chained when 8bit
    // storage was requested) so `bufferDeviceAddress` is on for every
    // device — the BDA path is the new default for D2D callers.
    v12_feats.pNext = @ptrCast(&v13_feats);
    const p_next: ?*const anyopaque = @ptrCast(&v12_feats);
    // VK_EXT_subgroup_size_control was promoted in 1.3 — for a 1.3
    // device the v13 feature struct is the authoritative gate; passing
    // the EXT name to enabledExtensionNames is unnecessary and on some
    // 1.3 loaders triggers a "extension already promoted" rejection.
    var ext_names_storage: [2][*:0]const u8 = .{
        "VK_KHR_8bit_storage",
        "VK_KHR_storage_buffer_storage_class",
    };
    var ext_count: u32 = 0;
    if (opts.enable_8bit_storage) {
        v12_feats.storageBuffer8BitAccess = vk.VK_TRUE;
        v12_feats.uniformAndStorageBuffer8BitAccess = vk.VK_TRUE;
        v12_feats.shaderInt8 = vk.VK_TRUE;
        ext_count = 2;
    }
    const dev_ci: vk.VkDeviceCreateInfo = .{
        .pNext = p_next,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = @ptrCast(&queue_ci_pub),
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = ext_count,
        .ppEnabledExtensionNames = if (ext_count > 0) @ptrCast(&ext_names_storage) else null,
        .pEnabledFeatures = null,
    };

    var dev: vk.VkDevice = null;
    if (create(pd, &dev_ci, null, &dev) != vk.VK_SUCCESS) return error.CreateDeviceFailed;

    // Refine the device-level dispatch slot via vkGetDeviceProcAddr.
    // Cheaper than the instance-level thunk and what the spec recommends.
    if (vk.vkGetDeviceProcAddr_fn) |gdpa| {
        if (gdpa(dev, "vkGetDeviceQueue")) |raw| {
            vk.vkGetDeviceQueue_fn = @ptrCast(@alignCast(raw));
        }
        if (gdpa(dev, "vkDestroyDevice")) |raw| {
            vk.vkDestroyDevice_fn = @ptrCast(@alignCast(raw));
        }
    }

    var queue: vk.VkQueue = null;
    const get_queue_now = vk.vkGetDeviceQueue_fn orelse get_queue;
    get_queue_now(dev, qfi, 0, &queue);

    return .{ .dev = dev, .queue = queue, .queue_family_index = qfi };
}

pub fn destroyDevice(dev: vk.VkDevice) void {
    if (dev == null) return;
    const f = vk.vkDestroyDevice_fn orelse return;
    f(dev, null);
}

// Internal local mirror used only to keep the priority-array lifetime
// obviously bound to the surrounding scope. Not exported.
const VkDeviceQueueCreateInfoLocal = struct {
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: ?[*]const f32,
};
