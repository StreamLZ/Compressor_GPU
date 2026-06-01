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
};

const MAX_PHYSICAL_DEVICES: u32 = 16;
const MAX_QUEUE_FAMILIES: u32 = 16;

pub fn pickPhysicalDevice(inst: vk.VkInstance) DeviceError!vk.VkPhysicalDevice {
    const enumerate = vk.vkEnumeratePhysicalDevices_fn orelse return error.LoaderNotReady;
    const get_qf = vk.vkGetPhysicalDeviceQueueFamilyProperties_fn orelse return error.LoaderNotReady;

    var count: u32 = 0;
    if (enumerate(inst, &count, null) != vk.VK_SUCCESS) return error.EnumerateFailed;
    if (count == 0) return error.NoVulkanDevice;
    if (count > MAX_PHYSICAL_DEVICES) count = MAX_PHYSICAL_DEVICES;

    var devices: [MAX_PHYSICAL_DEVICES]vk.VkPhysicalDevice = @splat(null);
    const r = enumerate(inst, &count, @ptrCast(&devices));
    // VK_INCOMPLETE is acceptable — we capped the count and got the first
    // `count` devices, which is exactly what we want.
    if (r != vk.VK_SUCCESS and r != vk.VK_INCOMPLETE) return error.EnumerateFailed;

    var i: u32 = 0;
    while (i < count) : (i += 1) {
        const pd = devices[i];
        if (pd == null) continue;
        var qf_count: u32 = 0;
        get_qf(pd, &qf_count, null);
        if (qf_count == 0) continue;
        if (qf_count > MAX_QUEUE_FAMILIES) qf_count = MAX_QUEUE_FAMILIES;
        var qfs: [MAX_QUEUE_FAMILIES]vk.VkQueueFamilyProperties = @splat(.{});
        get_qf(pd, &qf_count, @ptrCast(&qfs));
        var qi: u32 = 0;
        while (qi < qf_count) : (qi += 1) {
            if ((qfs[qi].queueFlags & vk.VK_QUEUE_COMPUTE_BIT) != 0 and qfs[qi].queueCount > 0) {
                return pd;
            }
        }
    }
    return error.NoComputeQueueFamily;
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
    var v12_feats: vk.VkPhysicalDeviceVulkan12Features = .{};
    var p_next: ?*const anyopaque = null;
    var ext_names_storage: [2][*:0]const u8 = .{ "VK_KHR_8bit_storage", "VK_KHR_storage_buffer_storage_class" };
    var ext_count: u32 = 0;
    if (opts.enable_8bit_storage) {
        v12_feats.storageBuffer8BitAccess = vk.VK_TRUE;
        v12_feats.uniformAndStorageBuffer8BitAccess = vk.VK_TRUE;
        v12_feats.shaderInt8 = vk.VK_TRUE;
        p_next = @ptrCast(&v12_feats);
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
