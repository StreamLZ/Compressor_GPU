//! 1:1 port of src/decode/module_loader.zig.
//!
//! Loads the SPV blobs that compile out of every .comp under
//! srcVK/decode/ and resolves them into pipeline handles the dispatch
//! layer launches. Mirrors the CUDA cuModuleLoadData /
//! cuModuleGetFunction surface — the codec call sites stay shaped like
//! the CUDA backend.
//!
//! CUDA reference: src/decode/module_loader.zig (entire file).
//!
//! VK adaptation: where the CUDA loader resolves cu*_fn driver entry
//! points into the cuda_api slots, this loader instead populates the
//! `procs` table in vulkan_api.zig with VMA-backed Zig closures plus a
//! per-context staging command buffer + fence. The pub var `*_fn` slots
//! hold VkPipeline handles (built by vkCreateComputePipelines off the
//! corresponding VkShaderModule + a per-kernel VkPipelineLayout).
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const vulkan_api = @import("vulkan_api.zig");
const descriptors = @import("descriptors.zig");
const decode_context = @import("decode_context.zig");
const vma = @import("../vma.zig");
const spv_blobs = @import("spv_blobs");

const VkResult = vulkan_api.VkResult;
const VkDeviceBuffer = vulkan_api.VkDeviceBuffer;
const VkStream = vulkan_api.VkStream;
const VK_SUCCESS_RC = vulkan_api.VK_SUCCESS_RC;

// ── Device selection (VK port carve-out) ─────────────────────────────
// VK adaptation: CUDA always picks device 0; the VK port lets the CLI
// override the picked device via setDeviceSelector before init() runs.
// Selectors map to vkEnumeratePhysicalDevices ordering (by_index) or a
// case-insensitive substring of vkPhysicalDeviceProperties.deviceName
// (by_name); .default falls back to SLZ_VK_DEVICE_INDEX env then the
// "discrete > integrated > virtual > cpu > other" priority scorer.
pub const DeviceSelector = union(enum) {
    default,
    by_index: u32,
    by_name: []const u8,
};
var g_requested_selector: DeviceSelector = .default;

pub fn setDeviceSelector(sel: DeviceSelector) void {
    g_requested_selector = sel;
}

// Cached name of the physically-bound device, populated by init() after
// vkCreateDevice succeeds. Sliced at the first NUL for CLI printing.
var g_bound_device_name_buf: [256]u8 = @splat(0);
var g_bound_device_name_len: usize = 0;

pub fn readBoundDeviceName(buf: []u8) []const u8 {
    const n = if (g_bound_device_name_len > buf.len) buf.len else g_bound_device_name_len;
    @memcpy(buf[0..n], g_bound_device_name_buf[0..n]);
    return buf[0..n];
}

// VK adaptation: lowercase one ASCII byte for case-insensitive deviceName
// substring matching used by the `by_name` selector branch below.
fn asciiToLowerLocal(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}

fn containsIgnoreCaseLocal(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (asciiToLowerLocal(haystack[i + j]) != asciiToLowerLocal(needle[j])) { match = false; break; }
        }
        if (match) return true;
    }
    return false;
}

// VK adaptation: deviceType priority scorer (DISCRETE > INTEGRATED >
// VIRTUAL > CPU > OTHER) used by the `default` selector branch when
// SLZ_VK_DEVICE_INDEX is unset. Mirrors src_vulkan/device.zig:deviceTypeScore.
fn deviceTypeScoreLocal(device_type: c_int) u8 {
    // VkPhysicalDeviceType: OTHER=0, INTEGRATED=1, DISCRETE=2, VIRTUAL=3, CPU=4
    return switch (device_type) {
        2 => 4,
        1 => 3,
        3 => 2,
        4 => 1,
        else => 0,
    };
}

fn hasComputeQueueLocal(pd: VkPhysicalDevice) bool {
    const get_qf = vkGetPhysicalDeviceQueueFamilyProperties_fn orelse return false;
    var qf_count: u32 = 0;
    get_qf(pd, &qf_count, null);
    if (qf_count == 0) return false;
    var qf_buf: [32]VkQueueFamilyProperties = undefined;
    var qfc = if (qf_count > 32) @as(u32, 32) else qf_count;
    get_qf(pd, &qfc, @ptrCast(&qf_buf));
    for (qf_buf[0..qfc]) |qf| {
        if ((qf.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 and qf.queueCount > 0) return true;
    }
    return false;
}

fn readPdNameInto(pd: VkPhysicalDevice, buf: []u8) []const u8 {
    const gpdp = vkGetPhysicalDeviceProperties_fn orelse return &.{};
    var props: VkPhysicalDeviceProperties = undefined;
    gpdp(pd, &props);
    var n: usize = 0;
    while (n < props.deviceName.len and n < buf.len and props.deviceName[n] != 0) : (n += 1) {
        buf[n] = props.deviceName[n];
    }
    return buf[0..n];
}

// VK adaptation: resolve g_requested_selector + SLZ_VK_DEVICE_INDEX into
// one of the enumerated VkPhysicalDevice handles. Returns null on lookup
// failure (caller surfaces as init() returning false).
fn pickPhysicalDeviceFromList(devices: []const VkPhysicalDevice) ?VkPhysicalDevice {
    var effective: DeviceSelector = g_requested_selector;
    if (effective == .default) {
        if (std.c.getenv("SLZ_VK_DEVICE_INDEX")) |raw| {
            const s = std.mem.span(raw);
            if (s.len > 0) {
                if (std.fmt.parseInt(u32, s, 10)) |idx| {
                    effective = .{ .by_index = idx };
                } else |_| {}
            }
        }
    }
    switch (effective) {
        .default => {
            var best: ?VkPhysicalDevice = null;
            var best_score: i16 = -1;
            for (devices) |pd| {
                if (pd == null) continue;
                if (!hasComputeQueueLocal(pd)) continue;
                const gpdp = vkGetPhysicalDeviceProperties_fn orelse continue;
                var props: VkPhysicalDeviceProperties = undefined;
                gpdp(pd, &props);
                const s: i16 = @intCast(deviceTypeScoreLocal(props.deviceType));
                if (s > best_score) {
                    best_score = s;
                    best = pd;
                }
            }
            return best;
        },
        .by_index => |idx| {
            if (idx >= devices.len) return null;
            const pd = devices[idx];
            if (pd == null) return null;
            if (!hasComputeQueueLocal(pd)) return null;
            return pd;
        },
        .by_name => |needle| {
            var match: ?VkPhysicalDevice = null;
            var match_count: u32 = 0;
            var name_buf: [256]u8 = undefined;
            for (devices) |pd| {
                if (pd == null) continue;
                const name = readPdNameInto(pd, name_buf[0..]);
                if (containsIgnoreCaseLocal(name, needle)) {
                    match = pd;
                    match_count += 1;
                }
            }
            if (match_count != 1) return null;
            if (!hasComputeQueueLocal(match.?)) return null;
            return match;
        },
    }
}

// VK adaptation: enumeration helper for the CLI's --probe mode. Stands
// up only the loader + instance + vkEnumeratePhysicalDevices long enough
// to print one line per device, then tears down. Does NOT call
// vkCreateDevice, so it's safe to invoke before / instead of init().
pub const ProbedDevice = struct {
    name_buf: [256]u8,
    name_len: usize,
    device_type: c_int,
    vendor_id: u32,
    api_version: u32,
};

pub fn enumerateDevicesForProbe(out: []ProbedDevice) ?u32 {
    if (vulkan_api.lib == null) {
        vulkan_api.lib = vulkan_api.win32.LoadLibraryA("vulkan-1.dll");
        if (vulkan_api.lib == null) return null;
    }
    if (vkGetInstanceProcAddr_fn == null) {
        vkGetInstanceProcAddr_fn = vulkan_api.getProc(FnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return null;
    }
    const gipa = vkGetInstanceProcAddr_fn.?;
    if (vkCreateInstance_fn == null) {
        vkCreateInstance_fn = @ptrCast(gipa(null, "vkCreateInstance"));
        if (vkCreateInstance_fn == null) return null;
    }
    var inst: VkInstance = null;
    const app = VkApplicationInfo{
        .pApplicationName = "streamlz_vk",
        .applicationVersion = (1 << 22),
        .pEngineName = "streamlz",
        .engineVersion = (1 << 22),
        .apiVersion = (1 << 22) | (3 << 12),
    };
    const ici = VkInstanceCreateInfo{ .pApplicationInfo = &app };
    if (vkCreateInstance_fn.?(&ici, null, &inst) != VK_SUCCESS_RC) return null;
    if (vkEnumeratePhysicalDevices_fn == null) {
        vkEnumeratePhysicalDevices_fn = @ptrCast(gipa(inst, "vkEnumeratePhysicalDevices"));
    }
    if (vkGetPhysicalDeviceProperties_fn == null) {
        vkGetPhysicalDeviceProperties_fn = @ptrCast(gipa(inst, "vkGetPhysicalDeviceProperties"));
    }
    const enumerate = vkEnumeratePhysicalDevices_fn orelse return null;
    var count: u32 = 0;
    if (enumerate(inst, &count, null) != VK_SUCCESS_RC) return null;
    if (count == 0) return 0;
    var phys_buf: [16]VkPhysicalDevice = @splat(null);
    var pc = if (count > 16) @as(u32, 16) else count;
    if (enumerate(inst, &pc, phys_buf[0..pc].ptr) != VK_SUCCESS_RC) return null;
    const cap: u32 = @intCast(out.len);
    const n: u32 = if (pc > cap) cap else pc;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const pd = phys_buf[i];
        var name: [256]u8 = @splat(0);
        var name_len: usize = 0;
        var dtype: c_int = 0;
        var vendor: u32 = 0;
        var api: u32 = 0;
        if (vkGetPhysicalDeviceProperties_fn) |gpdp| {
            var props: VkPhysicalDeviceProperties = undefined;
            gpdp(pd, &props);
            dtype = props.deviceType;
            vendor = props.vendorID;
            api = props.apiVersion;
            while (name_len < props.deviceName.len and props.deviceName[name_len] != 0) : (name_len += 1) {
                name[name_len] = props.deviceName[name_len];
            }
        }
        out[i] = .{
            .name_buf = name,
            .name_len = name_len,
            .device_type = dtype,
            .vendor_id = vendor,
            .api_version = api,
        };
    }
    return n;
}

// VK adaptation: text label for VkPhysicalDeviceType in --probe output.
pub fn deviceTypeName(device_type: c_int) []const u8 {
    return switch (device_type) {
        0 => "OTHER",
        1 => "INTEGRATED_GPU",
        2 => "DISCRETE_GPU",
        3 => "VIRTUAL_GPU",
        4 => "CPU",
        else => "UNKNOWN",
    };
}

// ── Pipeline handle slots (CUDA: CUfunction handles → VkPipeline).
// CUDA reference: src/decode/module_loader.zig:35-47. Slot names must
// stay verbatim per audit Section C.5.1 row 180. Each slot stores the
// VkPipeline (u64 non-dispatchable) cast to usize. procs.launch_kernel
// dispatches the pipeline through the per-kernel VkPipelineLayout.
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

// ── Minimal Vulkan API surface used directly by this loader ──────────
// VK adaptation: declared inline so the file stays self-contained; the
// rest of the codec talks to Vulkan exclusively through the procs.*
// table populated below. Types and sType values lifted verbatim from
// the Vulkan 1.3 spec.

const VkInstance = ?*opaque {};
const VkPhysicalDevice = ?*opaque {};
const VkDevice = ?*opaque {};
const VkQueue = ?*opaque {};
const VkCommandPool = u64;
const VkCommandBuffer = ?*opaque {};
const VkFence = u64;
const VkShaderModule = u64;
const VkBuffer = u64;
const VkDeviceSize = u64;
const VkDeviceMemory = u64;
const VkFlags = u32;
const VkDescriptorSetLayout = u64;
const VkPipelineLayout = u64;
const VkPipeline = u64;
const VkDescriptorPool = u64;
const VkDescriptorSet = u64;
const VkPipelineCache = u64;
const PFN_vkVoidFunction = ?*const fn () callconv(.c) void;

const VK_NULL_HANDLE: u64 = 0;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;

const VK_STRUCTURE_TYPE_APPLICATION_INFO: c_int = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: c_int = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: c_int = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: c_int = 3;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: c_int = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: c_int = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: c_int = 42;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: c_int = 4;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: c_int = 8;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: c_int = 16;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: c_int = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: c_int = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: c_int = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: c_int = 35;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: c_int = 30;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: c_int = 29;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: c_int = 18;

const VK_COMMAND_BUFFER_LEVEL_PRIMARY: c_int = 0;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x00000002;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;
const VK_FENCE_CREATE_SIGNALED_BIT: u32 = 0x00000001;
const VK_WHOLE_SIZE: u64 = ~@as(u64, 0);

const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: c_int = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;
const VK_PIPELINE_BIND_POINT_COMPUTE: c_int = 1;

// Timestamp query pool surface used by procs.event_*.
const VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO: c_int = 11;
const VK_QUERY_TYPE_TIMESTAMP: c_int = 2;
const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: u32 = 0x00002000;
const VK_QUERY_RESULT_64_BIT: u32 = 0x00000002;
const VK_QUERY_RESULT_WAIT_BIT: u32 = 0x00000001;
const VkQueryPool = u64;

// VK adaptation: feature-chain sTypes for the VkDeviceCreateInfo pNext.
// Mirrors src_vulkan/vk_api.zig:113-114 + the sub-struct sTypes — we need
// these to chain bufferDeviceAddress + shaderInt8 + storageBuffer8BitAccess
// + subgroupSizeControl at vkCreateDevice time. The SPV blobs declare
// OpCapability Int8 + StorageBuffer8BitAccess (see spirv-dis output of
// lz_decode_raw_kernel.spv); without these features enabled the driver
// will accept the device create but later crash inside vkCreate*Pipeline
// or vkGetBufferDeviceAddress (NVIDIA hits a hard segfault, no validation
// message). VMA was also being created with
// VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT (see vma.zig:235); when
// the matching feature isn't enabled VMA tries to call
// vkGetBufferDeviceAddress through a nullptr fn-ptr and segfaults.
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: c_int = 51;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: c_int = 53;
const VK_TRUE: u32 = 1;

// Minimal subset of VkPhysicalDeviceVulkan12Features — only the fields
// the codec actually needs to set are named; the rest land in the
// trailing _filler so the driver's writes never overflow this struct
// (spec layout is 47 VkBool32 fields after sType/pNext). Mirrors
// src_vulkan/vk_api.zig:506-556 layout. ORDER MATTERS — the spec fixes
// the field order; storageBuffer8BitAccess is the 3rd VkBool32,
// shaderInt8 is the 9th, bufferDeviceAddress is the 39th.
const VkPhysicalDeviceVulkan12Features = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    pNext: ?*anyopaque = null,
    samplerMirrorClampToEdge: u32 = 0,
    drawIndirectCount: u32 = 0,
    storageBuffer8BitAccess: u32 = 0,
    uniformAndStorageBuffer8BitAccess: u32 = 0,
    storagePushConstant8: u32 = 0,
    shaderBufferInt64Atomics: u32 = 0,
    shaderSharedInt64Atomics: u32 = 0,
    shaderFloat16: u32 = 0,
    shaderInt8: u32 = 0,
    descriptorIndexing: u32 = 0,
    shaderInputAttachmentArrayDynamicIndexing: u32 = 0,
    shaderUniformTexelBufferArrayDynamicIndexing: u32 = 0,
    shaderStorageTexelBufferArrayDynamicIndexing: u32 = 0,
    shaderUniformBufferArrayNonUniformIndexing: u32 = 0,
    shaderSampledImageArrayNonUniformIndexing: u32 = 0,
    shaderStorageBufferArrayNonUniformIndexing: u32 = 0,
    shaderStorageImageArrayNonUniformIndexing: u32 = 0,
    shaderInputAttachmentArrayNonUniformIndexing: u32 = 0,
    shaderUniformTexelBufferArrayNonUniformIndexing: u32 = 0,
    shaderStorageTexelBufferArrayNonUniformIndexing: u32 = 0,
    descriptorBindingUniformBufferUpdateAfterBind: u32 = 0,
    descriptorBindingSampledImageUpdateAfterBind: u32 = 0,
    descriptorBindingStorageImageUpdateAfterBind: u32 = 0,
    descriptorBindingStorageBufferUpdateAfterBind: u32 = 0,
    descriptorBindingUniformTexelBufferUpdateAfterBind: u32 = 0,
    descriptorBindingStorageTexelBufferUpdateAfterBind: u32 = 0,
    descriptorBindingUpdateUnusedWhilePending: u32 = 0,
    descriptorBindingPartiallyBound: u32 = 0,
    descriptorBindingVariableDescriptorCount: u32 = 0,
    runtimeDescriptorArray: u32 = 0,
    samplerFilterMinmax: u32 = 0,
    scalarBlockLayout: u32 = 0,
    imagelessFramebuffer: u32 = 0,
    uniformBufferStandardLayout: u32 = 0,
    shaderSubgroupExtendedTypes: u32 = 0,
    separateDepthStencilLayouts: u32 = 0,
    hostQueryReset: u32 = 0,
    timelineSemaphore: u32 = 0,
    bufferDeviceAddress: u32 = 0,
    bufferDeviceAddressCaptureReplay: u32 = 0,
    bufferDeviceAddressMultiDevice: u32 = 0,
    vulkanMemoryModel: u32 = 0,
    vulkanMemoryModelDeviceScope: u32 = 0,
    vulkanMemoryModelAvailabilityVisibilityChains: u32 = 0,
    shaderOutputViewportIndex: u32 = 0,
    shaderOutputLayer: u32 = 0,
    subgroupBroadcastDynamicId: u32 = 0,
};

// VkPhysicalDeviceVulkan13Features — full spec layout so the driver's
// writes don't clobber adjacent memory. We only set subgroupSizeControl
// + computeFullSubgroups; mirrors src_vulkan/vk_api.zig:561-579.
const VkPhysicalDeviceVulkan13Features = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    pNext: ?*anyopaque = null,
    robustImageAccess: u32 = 0,
    inlineUniformBlock: u32 = 0,
    descriptorBindingInlineUniformBlockUpdateAfterBind: u32 = 0,
    pipelineCreationCacheControl: u32 = 0,
    privateData: u32 = 0,
    shaderDemoteToHelperInvocation: u32 = 0,
    shaderTerminateInvocation: u32 = 0,
    subgroupSizeControl: u32 = 0,
    computeFullSubgroups: u32 = 0,
    synchronization2: u32 = 0,
    textureCompressionASTC_HDR: u32 = 0,
    shaderZeroInitializeWorkgroupMemory: u32 = 0,
    dynamicRendering: u32 = 0,
    shaderIntegerDotProduct: u32 = 0,
    maintenance4: u32 = 0,
};

const VkApplicationInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};

const VkInstanceCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

const VkQueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: extern struct { width: u32, height: u32, depth: u32 },
};

const VkDeviceQueueCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: [*]const f32,
};

const VkDeviceCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: [*]const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null,
};

const VkCommandPoolCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32,
};

const VkCommandBufferAllocateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool,
    level: c_int = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32,
};

const VkCommandBufferBeginInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pInheritanceInfo: ?*const anyopaque = null,
};

const VkSubmitInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*const anyopaque = null,
    pWaitDstStageMask: ?*const u32 = null,
    commandBufferCount: u32,
    pCommandBuffers: [*]const VkCommandBuffer,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*const anyopaque = null,
};

const VkFenceCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
};

const VkShaderModuleCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    codeSize: usize,
    pCode: [*]const u32,
};

const VkBufferCopy = extern struct {
    srcOffset: u64,
    dstOffset: u64,
    size: u64,
};

const VkQueryPoolCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    queryType: c_int,
    queryCount: u32,
    pipelineStatistics: u32 = 0,
};

// VkPhysicalDeviceLimits as laid out by the Vulkan 1.0 spec. Only the
// fields up through `timestampPeriod` matter for the timing FFI; the
// rest are kept as opaque storage so the struct size matches what the
// driver writes into.
const VkPhysicalDeviceLimits = extern struct {
    maxImageDimension1D: u32,
    maxImageDimension2D: u32,
    maxImageDimension3D: u32,
    maxImageDimensionCube: u32,
    maxImageArrayLayers: u32,
    maxTexelBufferElements: u32,
    maxUniformBufferRange: u32,
    maxStorageBufferRange: u32,
    maxPushConstantsSize: u32,
    maxMemoryAllocationCount: u32,
    maxSamplerAllocationCount: u32,
    bufferImageGranularity: u64,
    sparseAddressSpaceSize: u64,
    maxBoundDescriptorSets: u32,
    maxPerStageDescriptorSamplers: u32,
    maxPerStageDescriptorUniformBuffers: u32,
    maxPerStageDescriptorStorageBuffers: u32,
    maxPerStageDescriptorSampledImages: u32,
    maxPerStageDescriptorStorageImages: u32,
    maxPerStageDescriptorInputAttachments: u32,
    maxPerStageResources: u32,
    maxDescriptorSetSamplers: u32,
    maxDescriptorSetUniformBuffers: u32,
    maxDescriptorSetUniformBuffersDynamic: u32,
    maxDescriptorSetStorageBuffers: u32,
    maxDescriptorSetStorageBuffersDynamic: u32,
    maxDescriptorSetSampledImages: u32,
    maxDescriptorSetStorageImages: u32,
    maxDescriptorSetInputAttachments: u32,
    maxVertexInputAttributes: u32,
    maxVertexInputBindings: u32,
    maxVertexInputAttributeOffset: u32,
    maxVertexInputBindingStride: u32,
    maxVertexOutputComponents: u32,
    maxTessellationGenerationLevel: u32,
    maxTessellationPatchSize: u32,
    maxTessellationControlPerVertexInputComponents: u32,
    maxTessellationControlPerVertexOutputComponents: u32,
    maxTessellationControlPerPatchOutputComponents: u32,
    maxTessellationControlTotalOutputComponents: u32,
    maxTessellationEvaluationInputComponents: u32,
    maxTessellationEvaluationOutputComponents: u32,
    maxGeometryShaderInvocations: u32,
    maxGeometryInputComponents: u32,
    maxGeometryOutputComponents: u32,
    maxGeometryOutputVertices: u32,
    maxGeometryTotalOutputComponents: u32,
    maxFragmentInputComponents: u32,
    maxFragmentOutputAttachments: u32,
    maxFragmentDualSrcAttachments: u32,
    maxFragmentCombinedOutputResources: u32,
    maxComputeSharedMemorySize: u32,
    maxComputeWorkGroupCount: [3]u32,
    maxComputeWorkGroupInvocations: u32,
    maxComputeWorkGroupSize: [3]u32,
    subPixelPrecisionBits: u32,
    subTexelPrecisionBits: u32,
    mipmapPrecisionBits: u32,
    maxDrawIndexedIndexValue: u32,
    maxDrawIndirectCount: u32,
    maxSamplerLodBias: f32,
    maxSamplerAnisotropy: f32,
    maxViewports: u32,
    maxViewportDimensions: [2]u32,
    viewportBoundsRange: [2]f32,
    viewportSubPixelBits: u32,
    minMemoryMapAlignment: usize,
    minTexelBufferOffsetAlignment: u64,
    minUniformBufferOffsetAlignment: u64,
    minStorageBufferOffsetAlignment: u64,
    minTexelOffset: i32,
    maxTexelOffset: u32,
    minTexelGatherOffset: i32,
    maxTexelGatherOffset: u32,
    minInterpolationOffset: f32,
    maxInterpolationOffset: f32,
    subPixelInterpolationOffsetBits: u32,
    maxFramebufferWidth: u32,
    maxFramebufferHeight: u32,
    maxFramebufferLayers: u32,
    framebufferColorSampleCounts: u32,
    framebufferDepthSampleCounts: u32,
    framebufferStencilSampleCounts: u32,
    framebufferNoAttachmentsSampleCounts: u32,
    maxColorAttachments: u32,
    sampledImageColorSampleCounts: u32,
    sampledImageIntegerSampleCounts: u32,
    sampledImageDepthSampleCounts: u32,
    sampledImageStencilSampleCounts: u32,
    storageImageSampleCounts: u32,
    maxSampleMaskWords: u32,
    timestampComputeAndGraphics: u32,
    timestampPeriod: f32,
    maxClipDistances: u32,
    maxCullDistances: u32,
    maxCombinedClipAndCullDistances: u32,
    discreteQueuePriorities: u32,
    pointSizeRange: [2]f32,
    lineWidthRange: [2]f32,
    pointSizeGranularity: f32,
    lineWidthGranularity: f32,
    strictLines: u32,
    standardSampleLocations: u32,
    optimalBufferCopyOffsetAlignment: u64,
    optimalBufferCopyRowPitchAlignment: u64,
    nonCoherentAtomSize: u64,
};

const VkPhysicalDeviceSparseProperties = extern struct {
    residencyStandard2DBlockShape: u32,
    residencyStandard2DMultisampleBlockShape: u32,
    residencyStandard3DBlockShape: u32,
    residencyAlignedMipSize: u32,
    residencyNonResidentStrict: u32,
};

const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: c_int,
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    limits: VkPhysicalDeviceLimits,
    sparseProperties: VkPhysicalDeviceSparseProperties,
};

const VkMemoryBarrier = extern struct {
    sType: c_int = 46, // VK_STRUCTURE_TYPE_MEMORY_BARRIER
    pNext: ?*const anyopaque = null,
    srcAccessMask: u32,
    dstAccessMask: u32,
};

// Pipeline-stage bits used by the host→device staging copy barriers.
const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x00001000;
const VK_PIPELINE_STAGE_HOST_BIT: u32 = 0x00004000;
const VK_ACCESS_HOST_WRITE_BIT: u32 = 0x00004000;
const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x00000800;
const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x00001000;
const VK_ACCESS_HOST_READ_BIT: u32 = 0x00002000;

// Descriptor / pipeline structs

const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32,
    descriptorType: c_int,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: ?*const anyopaque = null,
};

const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding,
};

const VkPushConstantRange = extern struct {
    stageFlags: u32,
    offset: u32,
    size: u32,
};

const VkPipelineLayoutCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32,
    pSetLayouts: ?[*]const VkDescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: ?[*]const VkPushConstantRange,
};

const VkSpecializationInfo = extern struct {
    mapEntryCount: u32 = 0,
    pMapEntries: ?*const anyopaque = null,
    dataSize: usize = 0,
    pData: ?*const anyopaque = null,
};

const VkPipelineShaderStageCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: u32,
    module: VkShaderModule,
    pName: [*:0]const u8,
    pSpecializationInfo: ?*const VkSpecializationInfo = null,
};

const VkComputePipelineCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline = 0,
    basePipelineIndex: i32 = -1,
};

const VkDescriptorPoolSize = extern struct {
    type: c_int,
    descriptorCount: u32,
};

const VkDescriptorPoolCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: [*]const VkDescriptorPoolSize,
};

const VkDescriptorSetAllocateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: [*]const VkDescriptorSetLayout,
};

const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer,
    offset: VkDeviceSize,
    range: VkDeviceSize,
};

const VkWriteDescriptorSet = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32,
    descriptorCount: u32,
    descriptorType: c_int,
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?[*]const VkDescriptorBufferInfo,
    pTexelBufferView: ?*const anyopaque = null,
};

// Function-pointer typedefs for the entry points this loader resolves.
const FnGetInstanceProcAddr = *const fn (VkInstance, [*:0]const u8) callconv(.c) PFN_vkVoidFunction;
const FnGetDeviceProcAddr = *const fn (VkDevice, [*:0]const u8) callconv(.c) PFN_vkVoidFunction;
const FnCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *VkInstance) callconv(.c) VkResult;
const FnEnumeratePhysicalDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
const FnGetPhysicalDeviceQueueFamilyProperties = *const fn (VkPhysicalDevice, *u32, ?[*]VkQueueFamilyProperties) callconv(.c) void;
const FnCreateDevice = *const fn (VkPhysicalDevice, *const VkDeviceCreateInfo, ?*const anyopaque, *VkDevice) callconv(.c) VkResult;
const FnGetDeviceQueue = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;
const FnCreateCommandPool = *const fn (VkDevice, *const VkCommandPoolCreateInfo, ?*const anyopaque, *VkCommandPool) callconv(.c) VkResult;
const FnAllocateCommandBuffers = *const fn (VkDevice, *const VkCommandBufferAllocateInfo, [*]VkCommandBuffer) callconv(.c) VkResult;
const FnFreeCommandBuffers = *const fn (VkDevice, VkCommandPool, u32, [*]const VkCommandBuffer) callconv(.c) void;
const FnBeginCommandBuffer = *const fn (VkCommandBuffer, *const VkCommandBufferBeginInfo) callconv(.c) VkResult;
const FnEndCommandBuffer = *const fn (VkCommandBuffer) callconv(.c) VkResult;
const FnQueueSubmit = *const fn (VkQueue, u32, [*]const VkSubmitInfo, VkFence) callconv(.c) VkResult;
const FnQueueWaitIdle = *const fn (VkQueue) callconv(.c) VkResult;
const FnDeviceWaitIdle = *const fn (VkDevice) callconv(.c) VkResult;
const FnCreateFence = *const fn (VkDevice, *const VkFenceCreateInfo, ?*const anyopaque, *VkFence) callconv(.c) VkResult;
const FnWaitForFences = *const fn (VkDevice, u32, [*]const VkFence, u32, u64) callconv(.c) VkResult;
const FnResetFences = *const fn (VkDevice, u32, [*]const VkFence) callconv(.c) VkResult;
const FnDestroyFence = *const fn (VkDevice, VkFence, ?*const anyopaque) callconv(.c) void;
const FnDestroyCommandPool = *const fn (VkDevice, VkCommandPool, ?*const anyopaque) callconv(.c) void;
const FnResetCommandBuffer = *const fn (VkCommandBuffer, u32) callconv(.c) VkResult;
const FnCreateShaderModule = *const fn (VkDevice, *const VkShaderModuleCreateInfo, ?*const anyopaque, *VkShaderModule) callconv(.c) VkResult;
const FnDestroyShaderModule = *const fn (VkDevice, VkShaderModule, ?*const anyopaque) callconv(.c) void;
const FnCmdCopyBuffer = *const fn (VkCommandBuffer, VkBuffer, VkBuffer, u32, [*]const VkBufferCopy) callconv(.c) void;
const FnCmdFillBuffer = *const fn (VkCommandBuffer, VkBuffer, u64, u64, u32) callconv(.c) void;
const FnCmdPipelineBarrier = *const fn (VkCommandBuffer, u32, u32, u32, u32, ?[*]const VkMemoryBarrier, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(.c) void;
const FnCreateDescriptorSetLayout = *const fn (VkDevice, *const VkDescriptorSetLayoutCreateInfo, ?*const anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
const FnDestroyDescriptorSetLayout = *const fn (VkDevice, VkDescriptorSetLayout, ?*const anyopaque) callconv(.c) void;
const FnCreatePipelineLayout = *const fn (VkDevice, *const VkPipelineLayoutCreateInfo, ?*const anyopaque, *VkPipelineLayout) callconv(.c) VkResult;
const FnDestroyPipelineLayout = *const fn (VkDevice, VkPipelineLayout, ?*const anyopaque) callconv(.c) void;
const FnCreateComputePipelines = *const fn (VkDevice, VkPipelineCache, u32, [*]const VkComputePipelineCreateInfo, ?*const anyopaque, [*]VkPipeline) callconv(.c) VkResult;
const FnDestroyPipeline = *const fn (VkDevice, VkPipeline, ?*const anyopaque) callconv(.c) void;
const FnCreateDescriptorPool = *const fn (VkDevice, *const VkDescriptorPoolCreateInfo, ?*const anyopaque, *VkDescriptorPool) callconv(.c) VkResult;
const FnDestroyDescriptorPool = *const fn (VkDevice, VkDescriptorPool, ?*const anyopaque) callconv(.c) void;
const FnResetDescriptorPool = *const fn (VkDevice, VkDescriptorPool, u32) callconv(.c) VkResult;
const FnAllocateDescriptorSets = *const fn (VkDevice, *const VkDescriptorSetAllocateInfo, [*]VkDescriptorSet) callconv(.c) VkResult;
const FnUpdateDescriptorSets = *const fn (VkDevice, u32, [*]const VkWriteDescriptorSet, u32, ?*const anyopaque) callconv(.c) void;
const FnCmdBindPipeline = *const fn (VkCommandBuffer, c_int, VkPipeline) callconv(.c) void;
const FnCmdBindDescriptorSets = *const fn (VkCommandBuffer, c_int, VkPipelineLayout, u32, u32, [*]const VkDescriptorSet, u32, ?*const u32) callconv(.c) void;
const FnCmdPushConstants = *const fn (VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const anyopaque) callconv(.c) void;
const FnCmdDispatch = *const fn (VkCommandBuffer, u32, u32, u32) callconv(.c) void;
const FnCreateQueryPool = *const fn (VkDevice, *const VkQueryPoolCreateInfo, ?*const anyopaque, *VkQueryPool) callconv(.c) VkResult;
const FnDestroyQueryPool = *const fn (VkDevice, VkQueryPool, ?*const anyopaque) callconv(.c) void;
const FnCmdResetQueryPool = *const fn (VkCommandBuffer, VkQueryPool, u32, u32) callconv(.c) void;
const FnCmdWriteTimestamp = *const fn (VkCommandBuffer, u32, VkQueryPool, u32) callconv(.c) void;
const FnGetQueryPoolResults = *const fn (VkDevice, VkQueryPool, u32, u32, usize, *anyopaque, u64, u32) callconv(.c) VkResult;
const FnResetQueryPool = *const fn (VkDevice, VkQueryPool, u32, u32) callconv(.c) void;
const FnGetPhysicalDeviceProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties) callconv(.c) void;

// Resolved Vulkan entry points. Populated by init() after bring-up.
var vkGetInstanceProcAddr_fn: ?FnGetInstanceProcAddr = null;
var vkGetDeviceProcAddr_fn: ?FnGetDeviceProcAddr = null;
var vkCreateInstance_fn: ?FnCreateInstance = null;
var vkEnumeratePhysicalDevices_fn: ?FnEnumeratePhysicalDevices = null;
var vkGetPhysicalDeviceQueueFamilyProperties_fn: ?FnGetPhysicalDeviceQueueFamilyProperties = null;
var vkCreateDevice_fn: ?FnCreateDevice = null;
var vkGetDeviceQueue_fn: ?FnGetDeviceQueue = null;
var vkCreateCommandPool_fn: ?FnCreateCommandPool = null;
var vkAllocateCommandBuffers_fn: ?FnAllocateCommandBuffers = null;
var vkFreeCommandBuffers_fn: ?FnFreeCommandBuffers = null;
var vkBeginCommandBuffer_fn: ?FnBeginCommandBuffer = null;
var vkEndCommandBuffer_fn: ?FnEndCommandBuffer = null;
var vkQueueSubmit_fn: ?FnQueueSubmit = null;
var vkQueueWaitIdle_fn: ?FnQueueWaitIdle = null;
var vkDeviceWaitIdle_fn: ?FnDeviceWaitIdle = null;
var vkCreateFence_fn: ?FnCreateFence = null;
var vkWaitForFences_fn: ?FnWaitForFences = null;
var vkResetFences_fn: ?FnResetFences = null;
var vkDestroyFence_fn: ?FnDestroyFence = null;
var vkDestroyCommandPool_fn: ?FnDestroyCommandPool = null;
var vkResetCommandBuffer_fn: ?FnResetCommandBuffer = null;
var vkCreateShaderModule_fn: ?FnCreateShaderModule = null;
var vkDestroyShaderModule_fn: ?FnDestroyShaderModule = null;
var vkCmdCopyBuffer_fn: ?FnCmdCopyBuffer = null;
var vkCmdFillBuffer_fn: ?FnCmdFillBuffer = null;
var vkCmdPipelineBarrier_fn: ?FnCmdPipelineBarrier = null;
var vkCreateDescriptorSetLayout_fn: ?FnCreateDescriptorSetLayout = null;
var vkDestroyDescriptorSetLayout_fn: ?FnDestroyDescriptorSetLayout = null;
var vkCreatePipelineLayout_fn: ?FnCreatePipelineLayout = null;
var vkDestroyPipelineLayout_fn: ?FnDestroyPipelineLayout = null;
var vkCreateComputePipelines_fn: ?FnCreateComputePipelines = null;
var vkDestroyPipeline_fn: ?FnDestroyPipeline = null;
var vkCreateDescriptorPool_fn: ?FnCreateDescriptorPool = null;
var vkDestroyDescriptorPool_fn: ?FnDestroyDescriptorPool = null;
var vkResetDescriptorPool_fn: ?FnResetDescriptorPool = null;
var vkAllocateDescriptorSets_fn: ?FnAllocateDescriptorSets = null;
var vkUpdateDescriptorSets_fn: ?FnUpdateDescriptorSets = null;
var vkCmdBindPipeline_fn: ?FnCmdBindPipeline = null;
var vkCmdBindDescriptorSets_fn: ?FnCmdBindDescriptorSets = null;
var vkCmdPushConstants_fn: ?FnCmdPushConstants = null;
var vkCmdDispatch_fn: ?FnCmdDispatch = null;
var vkCreateQueryPool_fn: ?FnCreateQueryPool = null;
var vkDestroyQueryPool_fn: ?FnDestroyQueryPool = null;
var vkCmdResetQueryPool_fn: ?FnCmdResetQueryPool = null;
var vkCmdWriteTimestamp_fn: ?FnCmdWriteTimestamp = null;
var vkGetQueryPoolResults_fn: ?FnGetQueryPoolResults = null;
var vkGetPhysicalDeviceProperties_fn: ?FnGetPhysicalDeviceProperties = null;

// ── Module-private bring-up state ────────────────────────────────────
// VK adaptation: the shared command pool + per-context staging buffer
// the procs.h2d / procs.d2h closures funnel work through. Sized at
// init(); grown lazily if a larger copy needs to stage.
var g_command_pool: VkCommandPool = VK_NULL_HANDLE;
var g_command_buffer: VkCommandBuffer = null;
var g_fence: VkFence = VK_NULL_HANDLE;
var g_staging_buffer: vma.VkBuffer = 0;
var g_staging_alloc: vma.VmaAllocation = null;
var g_staging_size: usize = 0;
var g_staging_mapped: ?*anyopaque = null;
var g_descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;

// Timestamp query pool used by procs.event_*. Each VkEvent handle from
// procs.event_create is an index into this pool's query slots; the free
// list reclaims slots after procs.event_destroy. Capacity is sized for
// many events in flight at once (decode_dispatch records ~10 per frame).
var g_timestamp_query_pool: VkQueryPool = VK_NULL_HANDLE;
const g_query_capacity: u32 = 4096;
var g_query_index_free: std.ArrayListUnmanaged(u32) = .empty;
var g_timestamp_period_ns: f32 = 1.0;

// Registry of live device buffer allocations keyed by VkDeviceBuffer (u64).
// CUDA returns a flat CUdeviceptr; VK needs to track the
// (VkBuffer, VmaAllocation) pair so free_device can call vmaDestroyBuffer.
// The handle returned to the codec is the index + 1 (so 0 stays reserved
// as the null sentinel matching CUDA's CUdeviceptr=0 contract).
const AllocEntry = struct {
    buffer: vma.VkBuffer,
    allocation: vma.VmaAllocation,
    size: usize,
};
var g_allocs: std.ArrayListUnmanaged(?AllocEntry) = .empty;

// Registry of pinned host allocations so procs.free_host knows the
// original slice length (Zig's page_allocator.free requires it; CUDA's
// cuMemFreeHost gets the size from its own driver tracking).
const HostAllocEntry = struct {
    ptr: [*]u8,
    len: usize,
};
var g_host_allocs: std.ArrayListUnmanaged(HostAllocEntry) = .empty;

// Registry of stream entries. Stream handle = index + 1 (0 reserved as
// "default stream" → the global g_command_buffer/g_fence pair used for
// sync ops). Each entry owns its own VkCommandBuffer (allocated off the
// shared g_command_pool) + VkFence.
//
// Iter 4 host-perf fix (single-submit batching): each StreamEntry also
// owns a per-stream "staging arena" — a bump-allocator on top of a
// host-visible/coherent VkBuffer that procH2DAsync / procD2HAsync write
// into instead of sharing the global g_staging_buffer. This lets the
// decode dispatcher queue multiple H2D copies + N kernel dispatches +
// the D2H readback into ONE cmdbuf + ONE vkQueueSubmit + ONE
// vkWaitForFences per decoded block, matching src_vulkan's
// submitTwoWithCopy pattern. Pending D2H memcpys (staging → host)
// land in `pending_d2h` and flush in streamEndAndWait after the GPU
// completes.
const PendingD2H = struct {
    host_dst: *anyopaque,
    staging_off: usize,
    size: usize,
};
const StreamEntry = struct {
    cmdbuf: VkCommandBuffer,
    fence: VkFence,
    recording: bool, // true if Begin has been called but End/Submit not yet
    staging_buffer: vma.VkBuffer = 0,
    staging_alloc: vma.VmaAllocation = null,
    staging_mapped: ?*anyopaque = null,
    staging_size: usize = 0,
    staging_used: usize = 0,
    pending_d2h: std.ArrayListUnmanaged(PendingD2H) = .empty,
};
var g_streams: std.ArrayListUnmanaged(?StreamEntry) = .empty;

// Iter 4: parallel-test guard. The decoder's pipeline_stream is shared
// between threads (ptest_vk runs tests on up to 16 threads against the
// singleton g_default DecodeContext). Pre-iter4 each procs.launch_kernel
// inline-submitted + waited, so the cmdbuf-race window was tiny and
// hidden. Post-iter4 the whole decode block records into one cmdbuf
// before submission, so concurrent threads can interleave their
// streamBeginIfNeeded → vkCmdDispatch → streamEndAndWait calls and
// corrupt each other's recordings. Serializing through a Win32
// SRWLOCK (used directly because std.Io.Mutex requires an Io context
// the codec doesn't thread through here) preserves correctness;
// production callers using per-thread DecodeContext (via async
// streams) skip the lock by routing through their own stream entry
// (which still serializes per-stream, not globally). A future revision
// should move the mutex onto StreamEntry itself to allow lock-free
// parallelism across distinct streams.
const SRWLOCK = extern struct { ptr: ?*anyopaque = null };
extern "kernel32" fn AcquireSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
extern "kernel32" fn ReleaseSRWLockExclusive(lock: *SRWLOCK) callconv(.c) void;
var g_dispatcher_lock: SRWLOCK = .{};

pub fn lockDispatcherMutex() void {
    AcquireSRWLockExclusive(&g_dispatcher_lock);
}

pub fn unlockDispatcherMutex() void {
    ReleaseSRWLockExclusive(&g_dispatcher_lock);
}

// Iter 4: nonCoherentAtomSize for staging-arena bump alignment. VK spec
// requires offset/size of host-visible non-coherent mappings to align to
// this when flushing/invalidating; we use HOST_COHERENT so flush is
// implicit, but keep 64 B alignment for cache-line friendliness anyway.
const STAGING_BUMP_ALIGN: usize = 64;

// Iter 4: ensure the per-stream staging arena is at least `needed` bytes
// (grow-only). Returns false on alloc failure; true when the arena is
// ready to bump-allocate.
fn ensureStreamStaging(entry: *StreamEntry, needed: usize) bool {
    if (entry.staging_size >= needed) return true;
    if (entry.staging_alloc != null) {
        vma.destroyBuffer(allocator(), entry.staging_buffer, entry.staging_alloc);
        entry.staging_buffer = 0;
        entry.staging_alloc = null;
        entry.staging_mapped = null;
        entry.staging_size = 0;
    }
    var size: usize = if (entry.staging_size == 0) STAGING_INITIAL_SIZE else entry.staging_size;
    while (size < needed) size *= 2;
    var buf: vma.VkBuffer = 0;
    var alc: vma.VmaAllocation = null;
    const bci = vma.VkBufferCreateInfo{
        .size = size,
        .usage = vma.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vma.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    const aci = vma.VmaAllocationCreateInfo{
        .flags = vma.VMA_ALLOCATION_CREATE_MAPPED_BIT | vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = vma.VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = vma.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vma.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    var info: vma.VmaAllocationInfo = .{};
    if (vma.vmaCreateBuffer(allocator(), &bci, &aci, &buf, &alc, &info) != vma.VK_SUCCESS) return false;
    entry.staging_buffer = buf;
    entry.staging_alloc = alc;
    entry.staging_size = size;
    entry.staging_mapped = info.pMappedData;
    return true;
}

// Iter 4: bump-allocate `size` bytes from the stream's staging arena.
// Returns the byte offset within entry.staging_buffer on success; null
// on alloc failure. Caller is responsible for ensureStreamStaging if a
// known large size is needed (the bump-allocator only grows lazily).
fn streamStagingBump(entry: *StreamEntry, size: usize) ?usize {
    const aligned_used = (entry.staging_used + STAGING_BUMP_ALIGN - 1) & ~@as(usize, STAGING_BUMP_ALIGN - 1);
    const end = aligned_used + size;
    if (end > entry.staging_size) {
        // Grow to next pow-2 that fits the cumulative footprint.
        if (!ensureStreamStaging(entry, end)) return null;
    }
    entry.staging_used = end;
    return aligned_used;
}

// Per-kernel metadata. Indexed by KernelKind; each entry carries the
// VkDescriptorSetLayout + VkPipelineLayout + VkPipeline triple plus the
// param-layout info procs.launch_kernel uses to map the codec's params
// array onto SSBO bindings + push constants.
const KernelMeta = struct {
    layout: VkDescriptorSetLayout = 0,
    pl_layout: VkPipelineLayout = 0,
    pipeline: VkPipeline = 0,
    n_bindings: u32 = 0,
    push_constant_size: u32 = 0,
};

const KernelKind = enum(usize) {
    kernel_fn,
    kernel_raw_fn,
    gather_off16_fn,
    scan_parse_fn,
    walk_frame_fn,
    prefix_sum_chunks_fn,
    compact_huff_descs_fn,
    compact_raw_descs_fn,
    merge_huff_descs_fn,
    huff_build_fn,
    huff_decode_fn,
};

// Binding counts + push constant sizes per .comp. Values mirror the
// layout declarations in the corresponding srcVK/decode/*_kernel.comp.
const KernelDecl = struct {
    kind: KernelKind,
    n_bindings: u32,
    push_constant_size: u32,
};

const KERNEL_DECLS = [_]KernelDecl{
    .{ .kind = .kernel_fn, .n_bindings = 6, .push_constant_size = 16 },
    // srcVK/decode/lz_decode_raw_kernel.comp:25-50 declares 4 SSBO
    // bindings (CompressedBuf, ChunksBuf, DstBuf, TotalChunksBuf) plus
    // a 2× u32 push-constant block (chunks_per_group, sub_chunk_cap).
    .{ .kind = .kernel_raw_fn, .n_bindings = 4, .push_constant_size = 8 },
    .{ .kind = .gather_off16_fn, .n_bindings = 4, .push_constant_size = 4 },
    .{ .kind = .scan_parse_fn, .n_bindings = 10, .push_constant_size = 8 },
    .{ .kind = .walk_frame_fn, .n_bindings = 8, .push_constant_size = 8 },
    // srcVK/decode/prefix_sum_chunks_kernel.comp:16-34 declares 3 SSBO
    // bindings (ChunksBuf, FirstSubIdxBuf, TotalSubchunksBuf) plus a
    // 2× u32 push-constant block (n_chunks, sub_chunk_cap).
    .{ .kind = .prefix_sum_chunks_fn, .n_bindings = 3, .push_constant_size = 8 },
    .{ .kind = .compact_huff_descs_fn, .n_bindings = 4, .push_constant_size = 0 },
    .{ .kind = .compact_raw_descs_fn, .n_bindings = 5, .push_constant_size = 0 },
    .{ .kind = .merge_huff_descs_fn, .n_bindings = 10, .push_constant_size = 8 },
    .{ .kind = .huff_build_fn, .n_bindings = 4, .push_constant_size = 0 },
    .{ .kind = .huff_decode_fn, .n_bindings = 5, .push_constant_size = 0 },
};

var g_kernel_metas: [KERNEL_DECLS.len]KernelMeta = @splat(.{});

// Held-onto VkShaderModule handles so deinit() can destroy them after
// the matching VkPipeline. Indexed alongside KERNEL_DECLS.
var g_shader_modules: [KERNEL_DECLS.len]VkShaderModule = @splat(0);

// Extra registry for pipelines registered by encode/module_loader.zig
// (which builds its own VkShaderModule + VkPipelineLayout + VkPipeline
// off the encode-side SPV blobs through registerExternalPipeline). Keyed
// by VkPipeline handle (cast to usize); the meta carries the same
// (layout, pl_layout, n_bindings, push_constant_size) the decode-side
// KERNEL_DECLS provides for its kernels.
var g_extra_metas: std.ArrayListUnmanaged(KernelMeta) = .empty;

/// Register a pipeline + its layout metadata so procs.launch_kernel can
/// dispatch it. The encode-side module_loader uses this to publish its
/// kernels to the shared procs.launch_kernel slot. The caller owns the
/// VkShaderModule / VkPipelineLayout / VkDescriptorSetLayout / VkPipeline
/// lifetime; this side only retains read-only metadata.
pub fn registerExternalPipeline(
    pipeline: u64,
    pl_layout: u64,
    layout: u64,
    n_bindings: u32,
    push_constant_size: u32,
) bool {
    const gpa = std.heap.page_allocator;
    g_extra_metas.append(gpa, .{
        .layout = layout,
        .pl_layout = pl_layout,
        .pipeline = pipeline,
        .n_bindings = n_bindings,
        .push_constant_size = push_constant_size,
    }) catch return false;
    return true;
}

// Reverse map: VkPipeline handle (cast to usize) → KernelMeta. Used by
// procs.launch_kernel to look up the per-kernel metadata from the
// `pipeline` argument the codec passes in.
fn metaForPipeline(handle: usize) ?*KernelMeta {
    if (handle == 0) return null;
    inline for (KERNEL_DECLS, 0..) |_, i| {
        if (g_kernel_metas[i].pipeline != 0 and @as(usize, @intCast(g_kernel_metas[i].pipeline)) == handle) {
            return &g_kernel_metas[i];
        }
    }
    for (g_extra_metas.items) |*entry| {
        if (entry.pipeline != 0 and @as(usize, @intCast(entry.pipeline)) == handle) {
            return entry;
        }
    }
    return null;
}

fn metaFor(kind: KernelKind) *KernelMeta {
    inline for (KERNEL_DECLS, 0..) |decl, i| {
        if (decl.kind == kind) return &g_kernel_metas[i];
    }
    unreachable;
}

/// Test-only accessor exposing the per-kernel binding count + push-constant
/// size declared in KERNEL_DECLS. Used by srcVK/tests/dispatch_unit.zig to
/// pin the kernel ABI against the params[] layout in decode_dispatch.zig.
/// Names mirror the `pub var *_fn` slot names above.
pub const KernelLayoutInfo = struct {
    n_bindings: u32,
    push_constant_size: u32,
};

pub fn kernelLayoutByName(name: []const u8) ?KernelLayoutInfo {
    inline for (KERNEL_DECLS) |decl| {
        const decl_name = @tagName(decl.kind);
        if (std.mem.eql(u8, decl_name, name)) {
            return .{
                .n_bindings = decl.n_bindings,
                .push_constant_size = decl.push_constant_size,
            };
        }
    }
    return null;
}

const STAGING_INITIAL_SIZE: usize = 1 << 20; // 1 MiB

// CUDA reference: src/decode/module_loader.zig:49-194. One-shot module
// loader. Brings up vulkan-1.dll, picks a physical device + queue,
// creates the VkDevice, fills procs, loads SPV blobs.
pub fn init() bool {
    switch (vulkan_api.init_state) {
        .ready => return true,
        .failed, .in_progress => return false,
        .uninit => {},
    }
    vulkan_api.init_state = .in_progress;
    defer if (vulkan_api.init_state == .in_progress) {
        vulkan_api.init_state = .failed;
    };

    if (std.c.getenv("SLZ_NO_VK") != null) return false;

    vulkan_api.lib = vulkan_api.win32.LoadLibraryA("vulkan-1.dll");
    if (vulkan_api.lib == null) return false;

    vkGetInstanceProcAddr_fn = vulkan_api.getProc(FnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return false;
    const gipa = vkGetInstanceProcAddr_fn.?;

    // Bootstrap: resolve global (null-instance) entry points.
    vkCreateInstance_fn = @ptrCast(gipa(null, "vkCreateInstance"));
    if (vkCreateInstance_fn == null) return false;

    const app = VkApplicationInfo{
        .pApplicationName = "streamlz_vk",
        .applicationVersion = (1 << 22) | (0 << 12) | 0,
        .pEngineName = "streamlz",
        .engineVersion = (1 << 22) | (0 << 12) | 0,
        .apiVersion = (1 << 22) | (3 << 12), // Vulkan 1.3
    };
    const ici = VkInstanceCreateInfo{
        .pApplicationInfo = &app,
    };
    var inst: VkInstance = null;
    if (vkCreateInstance_fn.?(&ici, null, &inst) != VK_SUCCESS_RC) return false;
    vulkan_api.instance = @intFromPtr(inst);

    // Resolve the instance-level entry points the rest of the loader needs.
    vkEnumeratePhysicalDevices_fn = @ptrCast(gipa(inst, "vkEnumeratePhysicalDevices"));
    vkGetPhysicalDeviceQueueFamilyProperties_fn = @ptrCast(gipa(inst, "vkGetPhysicalDeviceQueueFamilyProperties"));
    vkCreateDevice_fn = @ptrCast(gipa(inst, "vkCreateDevice"));
    vkGetDeviceProcAddr_fn = @ptrCast(gipa(inst, "vkGetDeviceProcAddr"));
    // VK adaptation: resolve vkGetPhysicalDeviceProperties NOW so the
    // pickPhysicalDeviceFromList helper below can score deviceType and
    // match deviceName for the --device / SLZ_VK_DEVICE_INDEX selectors.
    vkGetPhysicalDeviceProperties_fn = @ptrCast(gipa(inst, "vkGetPhysicalDeviceProperties"));
    if (vkEnumeratePhysicalDevices_fn == null or
        vkGetPhysicalDeviceQueueFamilyProperties_fn == null or
        vkCreateDevice_fn == null or
        vkGetDeviceProcAddr_fn == null) return false;

    // VK adaptation: pick the physical device per g_requested_selector
    // (set by CLI before init). Default falls back to SLZ_VK_DEVICE_INDEX
    // env then a deviceType priority scorer. Mirrors CUDA's "device 0"
    // shape when only one compute device exists.
    var phys_count: u32 = 0;
    if (vkEnumeratePhysicalDevices_fn.?(inst, &phys_count, null) != VK_SUCCESS_RC) return false;
    if (phys_count == 0) return false;
    var phys_buf: [16]VkPhysicalDevice = @splat(null);
    var pc = if (phys_count > 16) @as(u32, 16) else phys_count;
    if (vkEnumeratePhysicalDevices_fn.?(inst, &pc, phys_buf[0..pc].ptr) != VK_SUCCESS_RC) return false;
    const phys = pickPhysicalDeviceFromList(phys_buf[0..pc]) orelse return false;
    vulkan_api.physical_device = @intFromPtr(phys);

    // Cache deviceName for later readBoundDeviceName().
    if (vkGetPhysicalDeviceProperties_fn) |gpdp| {
        var props_for_name: VkPhysicalDeviceProperties = undefined;
        gpdp(phys, &props_for_name);
        var n: usize = 0;
        while (n < props_for_name.deviceName.len and props_for_name.deviceName[n] != 0) : (n += 1) {
            g_bound_device_name_buf[n] = props_for_name.deviceName[n];
        }
        g_bound_device_name_len = n;
    }

    // Find a queue family that supports compute. Mirrors the CUDA driver's
    // implicit "first compute capable queue" selection.
    var qf_count: u32 = 0;
    vkGetPhysicalDeviceQueueFamilyProperties_fn.?(phys, &qf_count, null);
    if (qf_count == 0) return false;
    var qf_buf: [32]VkQueueFamilyProperties = undefined;
    var qfc = if (qf_count > 32) @as(u32, 32) else qf_count;
    vkGetPhysicalDeviceQueueFamilyProperties_fn.?(phys, &qfc, qf_buf[0..qfc].ptr);
    var queue_family: u32 = std.math.maxInt(u32);
    for (qf_buf[0..qfc], 0..) |qf, idx| {
        if ((qf.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0 and qf.queueCount > 0) {
            queue_family = @intCast(idx);
            break;
        }
    }
    if (queue_family == std.math.maxInt(u32)) return false;
    vulkan_api.compute_queue_family = queue_family;

    const priorities = [_]f32{1.0};
    const qci = VkDeviceQueueCreateInfo{
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priorities,
    };
    // VK adaptation: chain VkPhysicalDeviceVulkan12Features +
    // VkPhysicalDeviceVulkan13Features off VkDeviceCreateInfo.pNext so
    // the device is created with:
    //   * bufferDeviceAddress (required by VMA's
    //     VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT in vma.zig:235 +
    //     by VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT used in
    //     procMallocDevice).
    //   * shaderInt8 + storageBuffer8BitAccess +
    //     uniformAndStorageBuffer8BitAccess (the .spv blobs declare
    //     OpCapability Int8 + OpCapability StorageBuffer8BitAccess; see
    //     spirv-dis output of lz_decode_raw_kernel.spv).
    //   * subgroupSizeControl + computeFullSubgroups (every SPV blob
    //     declares GroupNonUniform* capabilities and the dispatch path
    //     assumes WARP_SIZE=32; Intel iGPU otherwise picks 16-wide
    //     subgroups and breaks every warp-cooperative kernel).
    //
    // Mirrors src_vulkan/device.zig:338-368.
    var v13_feats = VkPhysicalDeviceVulkan13Features{
        .subgroupSizeControl = VK_TRUE,
        .computeFullSubgroups = VK_TRUE,
    };
    var v12_feats = VkPhysicalDeviceVulkan12Features{
        .storageBuffer8BitAccess = VK_TRUE,
        .uniformAndStorageBuffer8BitAccess = VK_TRUE,
        .shaderInt8 = VK_TRUE,
        .bufferDeviceAddress = VK_TRUE,
        .pNext = @ptrCast(&v13_feats),
    };
    const dci = VkDeviceCreateInfo{
        .pNext = @ptrCast(&v12_feats),
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = @ptrCast(&qci),
    };
    var dev: VkDevice = null;
    if (vkCreateDevice_fn.?(phys, &dci, null, &dev) != VK_SUCCESS_RC) return false;
    vulkan_api.ctx = @intFromPtr(dev);

    // Resolve device-level entry points.
    const gdpa = vkGetDeviceProcAddr_fn.?;
    vkGetDeviceQueue_fn = @ptrCast(gdpa(dev, "vkGetDeviceQueue"));
    vkCreateCommandPool_fn = @ptrCast(gdpa(dev, "vkCreateCommandPool"));
    vkAllocateCommandBuffers_fn = @ptrCast(gdpa(dev, "vkAllocateCommandBuffers"));
    vkFreeCommandBuffers_fn = @ptrCast(gdpa(dev, "vkFreeCommandBuffers"));
    vkBeginCommandBuffer_fn = @ptrCast(gdpa(dev, "vkBeginCommandBuffer"));
    vkEndCommandBuffer_fn = @ptrCast(gdpa(dev, "vkEndCommandBuffer"));
    vkQueueSubmit_fn = @ptrCast(gdpa(dev, "vkQueueSubmit"));
    vkQueueWaitIdle_fn = @ptrCast(gdpa(dev, "vkQueueWaitIdle"));
    vkDeviceWaitIdle_fn = @ptrCast(gdpa(dev, "vkDeviceWaitIdle"));
    vkCreateFence_fn = @ptrCast(gdpa(dev, "vkCreateFence"));
    vkWaitForFences_fn = @ptrCast(gdpa(dev, "vkWaitForFences"));
    vkResetFences_fn = @ptrCast(gdpa(dev, "vkResetFences"));
    vkDestroyFence_fn = @ptrCast(gdpa(dev, "vkDestroyFence"));
    vkDestroyCommandPool_fn = @ptrCast(gdpa(dev, "vkDestroyCommandPool"));
    vkResetCommandBuffer_fn = @ptrCast(gdpa(dev, "vkResetCommandBuffer"));
    vkCreateShaderModule_fn = @ptrCast(gdpa(dev, "vkCreateShaderModule"));
    vkDestroyShaderModule_fn = @ptrCast(gdpa(dev, "vkDestroyShaderModule"));
    vkCmdCopyBuffer_fn = @ptrCast(gdpa(dev, "vkCmdCopyBuffer"));
    vkCmdFillBuffer_fn = @ptrCast(gdpa(dev, "vkCmdFillBuffer"));
    vkCmdPipelineBarrier_fn = @ptrCast(gdpa(dev, "vkCmdPipelineBarrier"));
    vkCreateDescriptorSetLayout_fn = @ptrCast(gdpa(dev, "vkCreateDescriptorSetLayout"));
    vkDestroyDescriptorSetLayout_fn = @ptrCast(gdpa(dev, "vkDestroyDescriptorSetLayout"));
    vkCreatePipelineLayout_fn = @ptrCast(gdpa(dev, "vkCreatePipelineLayout"));
    vkDestroyPipelineLayout_fn = @ptrCast(gdpa(dev, "vkDestroyPipelineLayout"));
    vkCreateComputePipelines_fn = @ptrCast(gdpa(dev, "vkCreateComputePipelines"));
    vkDestroyPipeline_fn = @ptrCast(gdpa(dev, "vkDestroyPipeline"));
    vkCreateDescriptorPool_fn = @ptrCast(gdpa(dev, "vkCreateDescriptorPool"));
    vkDestroyDescriptorPool_fn = @ptrCast(gdpa(dev, "vkDestroyDescriptorPool"));
    vkResetDescriptorPool_fn = @ptrCast(gdpa(dev, "vkResetDescriptorPool"));
    vkAllocateDescriptorSets_fn = @ptrCast(gdpa(dev, "vkAllocateDescriptorSets"));
    vkUpdateDescriptorSets_fn = @ptrCast(gdpa(dev, "vkUpdateDescriptorSets"));
    vkCmdBindPipeline_fn = @ptrCast(gdpa(dev, "vkCmdBindPipeline"));
    vkCmdBindDescriptorSets_fn = @ptrCast(gdpa(dev, "vkCmdBindDescriptorSets"));
    vkCmdPushConstants_fn = @ptrCast(gdpa(dev, "vkCmdPushConstants"));
    vkCmdDispatch_fn = @ptrCast(gdpa(dev, "vkCmdDispatch"));
    vkCreateQueryPool_fn = @ptrCast(gdpa(dev, "vkCreateQueryPool"));
    vkDestroyQueryPool_fn = @ptrCast(gdpa(dev, "vkDestroyQueryPool"));
    vkCmdResetQueryPool_fn = @ptrCast(gdpa(dev, "vkCmdResetQueryPool"));
    vkCmdWriteTimestamp_fn = @ptrCast(gdpa(dev, "vkCmdWriteTimestamp"));
    vkGetQueryPoolResults_fn = @ptrCast(gdpa(dev, "vkGetQueryPoolResults"));
    // vkGetPhysicalDeviceProperties_fn already resolved above (before device
    // pick) so it's available to the deviceType priority scorer.
    if (vkGetDeviceQueue_fn == null or vkCreateCommandPool_fn == null or
        vkAllocateCommandBuffers_fn == null or vkCreateShaderModule_fn == null or
        vkBeginCommandBuffer_fn == null or vkEndCommandBuffer_fn == null or
        vkQueueSubmit_fn == null or vkQueueWaitIdle_fn == null or
        vkDeviceWaitIdle_fn == null or vkCreateFence_fn == null or
        vkWaitForFences_fn == null or vkCmdCopyBuffer_fn == null or
        vkCmdFillBuffer_fn == null or vkCreateDescriptorSetLayout_fn == null or
        vkCreatePipelineLayout_fn == null or vkCreateComputePipelines_fn == null or
        vkCreateDescriptorPool_fn == null or vkAllocateDescriptorSets_fn == null or
        vkUpdateDescriptorSets_fn == null or vkCmdBindPipeline_fn == null or
        vkCmdBindDescriptorSets_fn == null or vkCmdDispatch_fn == null) return false;

    var queue: VkQueue = null;
    vkGetDeviceQueue_fn.?(dev, queue_family, 0, &queue);
    if (queue == null) return false;
    vulkan_api.compute_queue = @intFromPtr(queue);

    // Per-context staging command pool + buffer + fence. Reused by every
    // procs.h2d / d2h / d2d / memset call. The pool has the
    // RESET_COMMAND_BUFFER bit so each call can vkResetCommandBuffer.
    const pool_ci = VkCommandPoolCreateInfo{
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue_family,
    };
    if (vkCreateCommandPool_fn.?(dev, &pool_ci, null, &g_command_pool) != VK_SUCCESS_RC) return false;
    const cb_alloc = VkCommandBufferAllocateInfo{
        .commandPool = g_command_pool,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers_fn.?(dev, &cb_alloc, @ptrCast(&g_command_buffer)) != VK_SUCCESS_RC) return false;
    const fence_ci = VkFenceCreateInfo{};
    if (vkCreateFence_fn.?(dev, &fence_ci, null, &g_fence) != VK_SUCCESS_RC) return false;

    // VMA allocator on the (instance, phys, dev) triple. The VMA Zig
    // wrapper hides the vmaCreateAllocator boilerplate.
    // VK adaptation: VMA is built with VMA_DYNAMIC_VULKAN_FUNCTIONS=1 and
    // VMA_STATIC_VULKAN_FUNCTIONS=0 (see srcVK/vma/vk_mem_alloc_impl.cpp).
    // It REQUIRES the caller to supply vkGetInstanceProcAddr +
    // vkGetDeviceProcAddr through VmaVulkanFunctions; otherwise VMA
    // segfaults dereferencing a null fn-ptr inside
    // ImportVulkanFunctions_Dynamic (see vk_mem_alloc.h:13035-13040).
    const vma_alloc = vma.createAllocator(
        @ptrFromInt(vulkan_api.instance),
        @ptrFromInt(vulkan_api.physical_device),
        @ptrFromInt(vulkan_api.ctx),
        @ptrCast(vkGetInstanceProcAddr_fn),
        @ptrCast(vkGetDeviceProcAddr_fn),
    ) catch return false;
    vulkan_api.vma_allocator = @intFromPtr(vma_alloc);

    // Stage one initial staging buffer (HOST_VISIBLE + HOST_COHERENT,
    // persistently mapped). Grown lazily when a larger copy needs it.
    if (!ensureStaging(STAGING_INITIAL_SIZE)) return false;

    // Per-context descriptor pool. Sized generously enough that each
    // kernel can allocate one descriptor set per launch without
    // outrunning the pool — the codec resets between dispatches in
    // procs.launch_kernel via vkResetDescriptorPool.
    const max_bindings_per_pool: u32 = 4096;
    const max_sets: u32 = 1024;
    const pool_size = VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = max_bindings_per_pool,
    };
    const dp_ci = VkDescriptorPoolCreateInfo{
        .maxSets = max_sets,
        .poolSizeCount = 1,
        .pPoolSizes = @ptrCast(&pool_size),
    };
    if (vkCreateDescriptorPool_fn.?(dev, &dp_ci, null, &g_descriptor_pool) != VK_SUCCESS_RC) return false;

    // Timestamp query pool for procs.event_*. Sized to g_query_capacity;
    // each slot is one VkEvent handle. The free list seeds with every
    // index 0..g_query_capacity so procEventCreate can pop a slot. The
    // ticks→ns conversion factor (timestampPeriod) lives on
    // VkPhysicalDeviceLimits.
    if (vkGetPhysicalDeviceProperties_fn) |gpdp| {
        var props: VkPhysicalDeviceProperties = undefined;
        gpdp(phys, &props);
        g_timestamp_period_ns = props.limits.timestampPeriod;
        if (g_timestamp_period_ns <= 0.0) g_timestamp_period_ns = 1.0;
    }
    if (vkCreateQueryPool_fn) |create_qp| {
        const qp_ci = VkQueryPoolCreateInfo{
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = g_query_capacity,
        };
        if (create_qp(dev, &qp_ci, null, &g_timestamp_query_pool) != VK_SUCCESS_RC) {
            g_timestamp_query_pool = VK_NULL_HANDLE;
        } else {
            const gpa = std.heap.page_allocator;
            g_query_index_free.ensureTotalCapacity(gpa, g_query_capacity) catch {};
            var i: u32 = 0;
            while (i < g_query_capacity) : (i += 1) {
                g_query_index_free.append(gpa, i) catch break;
            }
        }
    }

    // Wire up the procs.* surface. All slots use the module-private
    // helpers defined below.
    vulkan_api.procs.malloc_device = procMallocDevice;
    vulkan_api.procs.free_device = procFreeDevice;
    vulkan_api.procs.h2d = procH2D;
    vulkan_api.procs.d2h = procD2H;
    vulkan_api.procs.d2d = procD2D;
    vulkan_api.procs.memset_d8 = procMemsetD8;
    vulkan_api.procs.memset_d8_async = procMemsetD8Async;
    vulkan_api.procs.malloc_host = procMallocHost;
    vulkan_api.procs.free_host = procFreeHost;
    vulkan_api.procs.stream_sync = procStreamSync;
    vulkan_api.procs.ctx_sync = procCtxSync;
    vulkan_api.procs.stream_create = procStreamCreate;
    vulkan_api.procs.stream_destroy = procStreamDestroy;
    vulkan_api.procs.launch_kernel = procLaunchKernel;
    vulkan_api.procs.h2d_async = procH2DAsync;
    vulkan_api.procs.d2h_async = procD2HAsync;
    vulkan_api.procs.event_create = procEventCreate;
    vulkan_api.procs.event_record = procEventRecord;
    vulkan_api.procs.event_synchronize = procEventSynchronize;
    vulkan_api.procs.event_elapsed_time = procEventElapsedTime;
    vulkan_api.procs.event_destroy = procEventDestroy;

    // Compile every required SPV blob into a VkShaderModule + matching
    // VkDescriptorSetLayout + VkPipelineLayout + VkPipeline. The
    // pipeline handle is what gets stored in the public *_fn slots so
    // the codec call sites can pass it straight to procs.launch_kernel.
    if (!buildKernel(.kernel_fn, spv_blobs.lz_decode)) return false;
    module = @intCast(metaFor(.kernel_fn).pipeline);
    kernel_fn = module; // CUDA aliases kernel_fn := module after cuModuleGetFunction.

    if (!buildKernel(.kernel_raw_fn, spv_blobs.lz_decode_raw)) return false;
    kernel_raw_fn = @intCast(metaFor(.kernel_raw_fn).pipeline);

    if (!buildKernel(.gather_off16_fn, spv_blobs.gather_raw_off16)) return false;
    gather_off16_fn = @intCast(metaFor(.gather_off16_fn).pipeline);

    if (!buildKernel(.scan_parse_fn, spv_blobs.scan_parse)) return false;
    scan_parse_fn = @intCast(metaFor(.scan_parse_fn).pipeline);

    if (!buildKernel(.walk_frame_fn, spv_blobs.walk_frame)) return false;
    walk_frame_fn = @intCast(metaFor(.walk_frame_fn).pipeline);

    if (!buildKernel(.prefix_sum_chunks_fn, spv_blobs.prefix_sum_chunks)) return false;
    prefix_sum_chunks_fn = @intCast(metaFor(.prefix_sum_chunks_fn).pipeline);

    if (!buildKernel(.compact_huff_descs_fn, spv_blobs.compact_huff_descs)) return false;
    compact_huff_descs_fn = @intCast(metaFor(.compact_huff_descs_fn).pipeline);

    if (!buildKernel(.compact_raw_descs_fn, spv_blobs.compact_raw_descs)) return false;
    compact_raw_descs_fn = @intCast(metaFor(.compact_raw_descs_fn).pipeline);

    if (!buildKernel(.merge_huff_descs_fn, spv_blobs.merge_huff_descs)) return false;
    merge_huff_descs_fn = @intCast(metaFor(.merge_huff_descs_fn).pipeline);

    // L1 required kernels: lz_decode_raw + prefix_sum_chunks must load.
    if (kernel_raw_fn == 0 or prefix_sum_chunks_fn == 0) return false;

    // Huff decode kernels. Optional (CUDA mirrors this — Huffman path is
    // a chunk-type=4 feature that L1 frames skip). If huff_build's SPV
    // doesn't compile or descriptor layout creation fails we leave
    // huff_*_fn as 0 and the codec falls back to the CPU Huffman path.
    if (buildKernel(.huff_build_fn, spv_blobs.huff_build_lut)) {
        huff_build_fn = @intCast(metaFor(.huff_build_fn).pipeline);
        huff_module = huff_build_fn; // CUDA reuses the module handle as the table slot.
        if (buildKernel(.huff_decode_fn, spv_blobs.huff_decode_4stream)) {
            huff_decode_fn = @intCast(metaFor(.huff_decode_fn).pipeline);
        }
    }

    ensurePipelineStream(&@import("driver.zig").g_default) catch {
        vulkan_api.init_state = .failed;
        return false;
    };

    vulkan_api.init_state = .ready;
    return true;
}

/// CUDA reference: src/decode/module_loader.zig:199-204. Lazily allocate
/// the persistent pipeline stream on `ctx`.
pub fn ensurePipelineStream(d_ctx: *decode_context.DecodeContext) descriptors.GpuError!void {
    if (d_ctx.pipeline_stream_created) return;
    const create = vulkan_api.procs.stream_create orelse return error.BackendNotAvailable;
    var stream: VkStream = 0;
    if (create(&stream, 0) != VK_SUCCESS_RC) return error.BackendNotAvailable;
    d_ctx.pipeline_stream = stream;
    d_ctx.pipeline_stream_created = true;
}

/// CUDA reference: src/decode/module_loader.zig:206. True iff init()
/// completed successfully.
pub fn isAvailable() bool {
    return init();
}

// ── Internal helpers ──────────────────────────────────────────────────

fn allocator() vma.VmaAllocator {
    return @ptrFromInt(vulkan_api.vma_allocator);
}

fn loadShaderModule(spv: []const u8) VkShaderModule {
    if (spv.len == 0 or spv.len % 4 != 0) return 0;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const ci = VkShaderModuleCreateInfo{
        .codeSize = spv.len,
        .pCode = @ptrCast(@alignCast(spv.ptr)),
    };
    var sm: VkShaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule_fn.?(dev, &ci, null, &sm) != VK_SUCCESS_RC) return 0;
    return sm;
}

// Build a VkDescriptorSetLayout for N storage-buffer bindings (0..N-1).
fn buildDescriptorSetLayout(n_bindings: u32) VkDescriptorSetLayout {
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    if (n_bindings == 0) {
        const ci = VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 0,
            .pBindings = null,
        };
        var layout: VkDescriptorSetLayout = 0;
        if (vkCreateDescriptorSetLayout_fn.?(dev, &ci, null, &layout) != VK_SUCCESS_RC) return 0;
        return layout;
    }
    // Heap-allocate the bindings array; freed after the create call returns.
    const gpa = std.heap.page_allocator;
    const bindings = gpa.alloc(VkDescriptorSetLayoutBinding, n_bindings) catch return 0;
    defer gpa.free(bindings);
    for (bindings, 0..) |*b, i| {
        b.* = .{
            .binding = @intCast(i),
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    const ci = VkDescriptorSetLayoutCreateInfo{
        .bindingCount = n_bindings,
        .pBindings = bindings.ptr,
    };
    var layout: VkDescriptorSetLayout = 0;
    if (vkCreateDescriptorSetLayout_fn.?(dev, &ci, null, &layout) != VK_SUCCESS_RC) return 0;
    return layout;
}

// Build a VkPipelineLayout from a descriptor set layout + (optional) push
// constant range.
fn buildPipelineLayout(layout: VkDescriptorSetLayout, push_size: u32) VkPipelineLayout {
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    var pc_range: VkPushConstantRange = .{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_size,
    };
    const layouts_arr = [_]VkDescriptorSetLayout{layout};
    const has_layout: u32 = if (layout != 0) 1 else 0;
    const ci = VkPipelineLayoutCreateInfo{
        .setLayoutCount = has_layout,
        .pSetLayouts = if (has_layout > 0) layouts_arr[0..1].ptr else null,
        .pushConstantRangeCount = if (push_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_size > 0) @ptrCast(&pc_range) else null,
    };
    var pl: VkPipelineLayout = 0;
    if (vkCreatePipelineLayout_fn.?(dev, &ci, null, &pl) != VK_SUCCESS_RC) return 0;
    return pl;
}

fn buildKernel(kind: KernelKind, spv: []const u8) bool {
    const meta = metaFor(kind);
    // Locate the matching declaration for n_bindings / push_constant_size.
    var decl: KernelDecl = undefined;
    inline for (KERNEL_DECLS) |d| if (d.kind == kind) {
        decl = d;
    };
    meta.n_bindings = decl.n_bindings;
    meta.push_constant_size = decl.push_constant_size;

    const sm = loadShaderModule(spv);
    if (sm == 0) return false;
    // Stash so deinit can destroy after the pipeline.
    inline for (KERNEL_DECLS, 0..) |d, i| if (d.kind == kind) {
        g_shader_modules[i] = sm;
    };

    meta.layout = buildDescriptorSetLayout(decl.n_bindings);
    if (decl.n_bindings > 0 and meta.layout == 0) return false;

    meta.pl_layout = buildPipelineLayout(meta.layout, decl.push_constant_size);
    if (meta.pl_layout == 0) return false;

    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const stage = VkPipelineShaderStageCreateInfo{
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = sm,
        .pName = "main",
    };
    const ci = VkComputePipelineCreateInfo{
        .stage = stage,
        .layout = meta.pl_layout,
    };
    var pipeline: VkPipeline = 0;
    if (vkCreateComputePipelines_fn.?(dev, 0, 1, @ptrCast(&ci), null, @ptrCast(&pipeline)) != VK_SUCCESS_RC) return false;
    meta.pipeline = pipeline;
    return true;
}

fn ensureStaging(needed: usize) bool {
    if (g_staging_size >= needed) return true;
    if (g_staging_alloc != null) {
        vma.destroyBuffer(allocator(), g_staging_buffer, g_staging_alloc);
        g_staging_buffer = 0;
        g_staging_alloc = null;
        g_staging_mapped = null;
        g_staging_size = 0;
    }
    var size = if (g_staging_size == 0) STAGING_INITIAL_SIZE else g_staging_size;
    while (size < needed) size *= 2;
    var buf: vma.VkBuffer = 0;
    var alc: vma.VmaAllocation = null;
    const bci = vma.VkBufferCreateInfo{
        .size = size,
        .usage = vma.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vma.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    const aci = vma.VmaAllocationCreateInfo{
        .flags = vma.VMA_ALLOCATION_CREATE_MAPPED_BIT | vma.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        .usage = vma.VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = vma.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vma.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    var info: vma.VmaAllocationInfo = .{};
    if (vma.vmaCreateBuffer(allocator(), &bci, &aci, &buf, &alc, &info) != vma.VK_SUCCESS) return false;
    g_staging_buffer = buf;
    g_staging_alloc = alc;
    g_staging_size = size;
    g_staging_mapped = info.pMappedData;
    return true;
}

fn registerAlloc(buffer: vma.VkBuffer, alc: vma.VmaAllocation, size: usize) VkDeviceBuffer {
    const gpa = std.heap.page_allocator;
    // Reuse a freed slot if available.
    for (g_allocs.items, 0..) |slot, i| {
        if (slot == null) {
            g_allocs.items[i] = AllocEntry{ .buffer = buffer, .allocation = alc, .size = size };
            return @intCast(i + 1);
        }
    }
    g_allocs.append(gpa, AllocEntry{ .buffer = buffer, .allocation = alc, .size = size }) catch return 0;
    return @intCast(g_allocs.items.len);
}

fn lookupAlloc(handle: VkDeviceBuffer) ?AllocEntry {
    if (handle == 0) return null;
    const idx: usize = @intCast(handle - 1);
    if (idx >= g_allocs.items.len) return null;
    return g_allocs.items[idx];
}

fn releaseAlloc(handle: VkDeviceBuffer) void {
    if (handle == 0) return;
    const idx: usize = @intCast(handle - 1);
    if (idx >= g_allocs.items.len) return;
    g_allocs.items[idx] = null;
}

fn submitAndWait() VkResult {
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const queue: VkQueue = @ptrFromInt(vulkan_api.compute_queue);
    const cb = g_command_buffer;
    var rc: VkResult = VK_SUCCESS_RC;
    rc = vkEndCommandBuffer_fn.?(cb);
    if (rc != VK_SUCCESS_RC) return rc;
    const submit = VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cb),
    };
    _ = vkResetFences_fn.?(dev, 1, @ptrCast(&g_fence));
    rc = vkQueueSubmit_fn.?(queue, 1, @ptrCast(&submit), g_fence);
    if (rc != VK_SUCCESS_RC) return rc;
    rc = vkWaitForFences_fn.?(dev, 1, @ptrCast(&g_fence), 1, ~@as(u64, 0));
    return rc;
}

fn beginOneShotCB() VkResult {
    _ = vkResetCommandBuffer_fn.?(g_command_buffer, 0);
    const begin = VkCommandBufferBeginInfo{
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    return vkBeginCommandBuffer_fn.?(g_command_buffer, &begin);
}

// Acquire a stream entry by handle. handle=0 selects the shared default
// stream (g_command_buffer + g_fence); handle>=1 picks from g_streams.
fn streamEntryFor(handle: VkStream) ?*StreamEntry {
    if (handle == 0) return null;
    const idx: usize = @intCast(handle - 1);
    if (idx >= g_streams.items.len) return null;
    return if (g_streams.items[idx] != null) &g_streams.items[idx].? else null;
}

// Begin recording into a stream's cmdbuf if not already recording.
fn streamBeginIfNeeded(entry: *StreamEntry) VkResult {
    if (entry.recording) return VK_SUCCESS_RC;
    _ = vkResetCommandBuffer_fn.?(entry.cmdbuf, 0);
    const begin = VkCommandBufferBeginInfo{
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    const rc = vkBeginCommandBuffer_fn.?(entry.cmdbuf, &begin);
    if (rc == VK_SUCCESS_RC) entry.recording = true;
    return rc;
}

// End + submit the stream's cmdbuf (if recording) and wait on its fence.
//
// Iter 4: also flushes deferred D2H staging→host memcpys queued by
// procD2HAsync, then resets the staging arena bump-allocator so the
// next decode reuses the buffer. The flush MUST happen after the
// vkWaitForFences (the GPU writes to staging are not host-visible
// until then, even with HOST_COHERENT memory — coherence covers
// CPU↔GPU cache visibility, not pipeline ordering).
fn streamEndAndWait(entry: *StreamEntry) VkResult {
    if (!entry.recording) {
        // Even when no cmdbuf was recorded, reset arena/pending state
        // for symmetry (defensive: caller may have skipped the body).
        entry.staging_used = 0;
        entry.pending_d2h.clearRetainingCapacity();
        return VK_SUCCESS_RC;
    }
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const queue: VkQueue = @ptrFromInt(vulkan_api.compute_queue);
    var rc = vkEndCommandBuffer_fn.?(entry.cmdbuf);
    if (rc != VK_SUCCESS_RC) {
        entry.recording = false;
        entry.staging_used = 0;
        entry.pending_d2h.clearRetainingCapacity();
        return rc;
    }
    const cb = entry.cmdbuf;
    const submit = VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cb),
    };
    _ = vkResetFences_fn.?(dev, 1, @ptrCast(&entry.fence));
    rc = vkQueueSubmit_fn.?(queue, 1, @ptrCast(&submit), entry.fence);
    if (rc != VK_SUCCESS_RC) {
        entry.recording = false;
        entry.staging_used = 0;
        entry.pending_d2h.clearRetainingCapacity();
        return rc;
    }
    rc = vkWaitForFences_fn.?(dev, 1, @ptrCast(&entry.fence), 1, ~@as(u64, 0));
    if (rc == VK_SUCCESS_RC and entry.pending_d2h.items.len > 0) {
        if (entry.staging_mapped) |mapped| {
            const base: [*]const u8 = @ptrCast(mapped);
            for (entry.pending_d2h.items) |p| {
                @memcpy(@as([*]u8, @ptrCast(p.host_dst))[0..p.size], base[p.staging_off .. p.staging_off + p.size]);
            }
        }
    }
    entry.recording = false;
    entry.staging_used = 0;
    entry.pending_d2h.clearRetainingCapacity();
    return rc;
}

// ── procs.* implementations ───────────────────────────────────────────

fn procMallocDevice(out: *VkDeviceBuffer, size: usize) callconv(.c) VkResult {
    var buf: vma.VkBuffer = 0;
    var alc: vma.VmaAllocation = null;
    const bci = vma.VkBufferCreateInfo{
        .size = size,
        .usage = vma.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vma.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            vma.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            vma.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    const aci = vma.VmaAllocationCreateInfo{
        .usage = vma.VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = vma.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    const rc = vma.vmaCreateBuffer(allocator(), &bci, &aci, &buf, &alc, null);
    if (rc != vma.VK_SUCCESS) return rc;
    const handle = registerAlloc(buf, alc, size);
    if (handle == 0) {
        vma.destroyBuffer(allocator(), buf, alc);
        return -1;
    }
    out.* = handle;
    return VK_SUCCESS_RC;
}

fn procFreeDevice(handle: VkDeviceBuffer) callconv(.c) VkResult {
    const entry = lookupAlloc(handle) orelse return VK_SUCCESS_RC;
    vma.destroyBuffer(allocator(), entry.buffer, entry.allocation);
    releaseAlloc(handle);
    return VK_SUCCESS_RC;
}

fn procH2D(dst: VkDeviceBuffer, src: *const anyopaque, size: usize) callconv(.c) VkResult {
    const entry = lookupAlloc(dst) orelse return -1;
    if (!ensureStaging(size)) return -1;
    const mapped = g_staging_mapped orelse return -1;
    @memcpy(@as([*]u8, @ptrCast(mapped))[0..size], @as([*]const u8, @ptrCast(src))[0..size]);
    var rc = beginOneShotCB();
    if (rc != VK_SUCCESS_RC) return rc;
    const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer_fn.?(g_command_buffer, g_staging_buffer, entry.buffer, 1, @ptrCast(&region));
    rc = submitAndWait();
    return rc;
}

fn procD2H(dst: *anyopaque, src: VkDeviceBuffer, size: usize) callconv(.c) VkResult {
    const entry = lookupAlloc(src) orelse return -1;
    if (!ensureStaging(size)) return -1;
    var rc = beginOneShotCB();
    if (rc != VK_SUCCESS_RC) return rc;
    const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer_fn.?(g_command_buffer, entry.buffer, g_staging_buffer, 1, @ptrCast(&region));
    rc = submitAndWait();
    if (rc != VK_SUCCESS_RC) return rc;
    const mapped = g_staging_mapped orelse return -1;
    @memcpy(@as([*]u8, @ptrCast(dst))[0..size], @as([*]const u8, @ptrCast(mapped))[0..size]);
    return VK_SUCCESS_RC;
}

fn procD2D(dst: VkDeviceBuffer, src: VkDeviceBuffer, size: usize, stream: VkStream) callconv(.c) VkResult {
    _ = stream;
    const dst_e = lookupAlloc(dst) orelse return -1;
    const src_e = lookupAlloc(src) orelse return -1;
    var rc = beginOneShotCB();
    if (rc != VK_SUCCESS_RC) return rc;
    const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer_fn.?(g_command_buffer, src_e.buffer, dst_e.buffer, 1, @ptrCast(&region));
    rc = submitAndWait();
    return rc;
}

fn procMemsetD8(dst: VkDeviceBuffer, value: u8, size: usize) callconv(.c) VkResult {
    const entry = lookupAlloc(dst) orelse return -1;
    var rc = beginOneShotCB();
    if (rc != VK_SUCCESS_RC) return rc;
    // VK adaptation: vkCmdFillBuffer broadcasts a u32; lift the byte
    // into all four lanes to match cuMemsetD8 semantics.
    const v: u32 = @as(u32, value) * 0x01010101;
    vkCmdFillBuffer_fn.?(g_command_buffer, entry.buffer, 0, size, v);
    rc = submitAndWait();
    return rc;
}

fn procMemsetD8Async(dst: VkDeviceBuffer, value: u8, size: usize, stream: VkStream) callconv(.c) VkResult {
    const entry = lookupAlloc(dst) orelse return -1;
    const v: u32 = @as(u32, value) * 0x01010101;
    if (streamEntryFor(stream)) |se| {
        const rc = streamBeginIfNeeded(se);
        if (rc != VK_SUCCESS_RC) return rc;
        vkCmdFillBuffer_fn.?(se.cmdbuf, entry.buffer, 0, size, v);
        // VK adaptation: defer submission until procs.stream_sync.
        return VK_SUCCESS_RC;
    }
    return procMemsetD8(dst, value, size);
}

fn procH2DAsync(dst: VkDeviceBuffer, src: *const anyopaque, size: usize, stream: VkStream) callconv(.c) VkResult {
    const entry = lookupAlloc(dst) orelse return -1;
    if (streamEntryFor(stream)) |se| {
        // Iter 4: bump-allocate from the per-stream staging arena so
        // back-to-back h2d_async calls on the same stream don't clobber
        // each other (the old code shared g_staging_buffer, which made
        // it unsafe to queue >1 H2D before stream_sync). The arena
        // resets in streamEndAndWait after the GPU consumes the data.
        const rc_begin = streamBeginIfNeeded(se);
        if (rc_begin != VK_SUCCESS_RC) return rc_begin;
        const off = streamStagingBump(se, size) orelse return -1;
        const mapped = se.staging_mapped orelse return -1;
        @memcpy(@as([*]u8, @ptrCast(mapped))[off .. off + size], @as([*]const u8, @ptrCast(src))[0..size]);
        const region = VkBufferCopy{ .srcOffset = off, .dstOffset = 0, .size = size };
        vkCmdCopyBuffer_fn.?(se.cmdbuf, se.staging_buffer, entry.buffer, 1, @ptrCast(&region));
        return VK_SUCCESS_RC;
    }
    return procH2D(dst, src, size);
}

fn procD2HAsync(dst: *anyopaque, src: VkDeviceBuffer, size: usize, stream: VkStream) callconv(.c) VkResult {
    const entry = lookupAlloc(src) orelse return -1;
    if (streamEntryFor(stream)) |se| {
        // Iter 4: defer both the cmdbuf record AND the staging → host
        // memcpy so a single end-of-decode streamEndAndWait flushes all
        // queued H2Ds + kernels + D2Hs in ONE submit. The old code
        // submitted-and-waited inline per d2h_async, which defeated
        // batching for the decoder's d2h-after-LZ-kernel path.
        const rc_begin = streamBeginIfNeeded(se);
        if (rc_begin != VK_SUCCESS_RC) return rc_begin;
        const off = streamStagingBump(se, size) orelse return -1;
        const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = off, .size = size };
        vkCmdCopyBuffer_fn.?(se.cmdbuf, entry.buffer, se.staging_buffer, 1, @ptrCast(&region));
        const gpa = std.heap.page_allocator;
        se.pending_d2h.append(gpa, .{ .host_dst = dst, .staging_off = off, .size = size }) catch return -1;
        return VK_SUCCESS_RC;
    }
    return procD2H(dst, src, size);
}

fn procMallocHost(out: *?*anyopaque, size: usize) callconv(.c) VkResult {
    const gpa = std.heap.page_allocator;
    const buf = gpa.alignedAlloc(u8, .@"64", size) catch return -1;
    g_host_allocs.append(gpa, .{ .ptr = buf.ptr, .len = buf.len }) catch {
        gpa.free(buf);
        return -1;
    };
    out.* = @ptrCast(buf.ptr);
    return VK_SUCCESS_RC;
}

fn procFreeHost(buf: *anyopaque) callconv(.c) VkResult {
    const gpa = std.heap.page_allocator;
    const target: [*]u8 = @ptrCast(buf);
    for (g_host_allocs.items, 0..) |entry, i| {
        if (entry.ptr == target) {
            const slice: []align(64) u8 = @alignCast(entry.ptr[0..entry.len]);
            gpa.free(slice);
            _ = g_host_allocs.swapRemove(i);
            return VK_SUCCESS_RC;
        }
    }
    return -1;
}

fn procStreamSync(stream: VkStream) callconv(.c) VkResult {
    if (streamEntryFor(stream)) |se| {
        return streamEndAndWait(se);
    }
    // VK adaptation: stream==0 is the shared default stream whose ops
    // submit-and-wait inline (sync semantics already satisfied).
    return VK_SUCCESS_RC;
}

fn procCtxSync() callconv(.c) VkResult {
    // Drain any in-flight per-stream cmdbufs first so vkDeviceWaitIdle
    // returns with all work flushed.
    var i: usize = 0;
    while (i < g_streams.items.len) : (i += 1) {
        if (g_streams.items[i]) |*se| {
            if (se.recording) _ = streamEndAndWait(se);
        }
    }
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    return vkDeviceWaitIdle_fn.?(dev);
}

fn procStreamCreate(out: *VkStream, flags: c_uint) callconv(.c) VkResult {
    _ = flags;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    // Allocate a per-stream VkCommandBuffer off the shared pool + its
    // own VkFence. The pool was created with the
    // RESET_COMMAND_BUFFER bit so each stream owns reset rights.
    var cb: VkCommandBuffer = null;
    const cb_alloc = VkCommandBufferAllocateInfo{
        .commandPool = g_command_pool,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers_fn.?(dev, &cb_alloc, @ptrCast(&cb)) != VK_SUCCESS_RC) return -1;
    var fence: VkFence = VK_NULL_HANDLE;
    const fence_ci = VkFenceCreateInfo{};
    if (vkCreateFence_fn.?(dev, &fence_ci, null, &fence) != VK_SUCCESS_RC) {
        vkFreeCommandBuffers_fn.?(dev, g_command_pool, 1, @ptrCast(&cb));
        return -1;
    }

    const gpa = std.heap.page_allocator;
    const entry = StreamEntry{ .cmdbuf = cb, .fence = fence, .recording = false };
    // Reuse a freed slot if available.
    for (g_streams.items, 0..) |slot, i| {
        if (slot == null) {
            g_streams.items[i] = entry;
            out.* = i + 1;
            return VK_SUCCESS_RC;
        }
    }
    g_streams.append(gpa, entry) catch {
        vkDestroyFence_fn.?(dev, fence, null);
        vkFreeCommandBuffers_fn.?(dev, g_command_pool, 1, @ptrCast(&cb));
        return -1;
    };
    out.* = g_streams.items.len;
    return VK_SUCCESS_RC;
}

fn procStreamDestroy(stream: VkStream) callconv(.c) VkResult {
    if (stream == 0) return VK_SUCCESS_RC;
    const idx: usize = @intCast(stream - 1);
    if (idx >= g_streams.items.len) return VK_SUCCESS_RC;
    var slot = g_streams.items[idx] orelse return VK_SUCCESS_RC;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    // Iter 4: free per-stream staging arena before tearing down cmdbuf
    // + fence. Order doesn't matter for VK (host-visible buffers carry
    // no GPU references after the fence-wait above), but we destroy in
    // reverse-creation order for symmetry with the encode side.
    if (slot.staging_alloc != null) {
        vma.destroyBuffer(allocator(), slot.staging_buffer, slot.staging_alloc);
    }
    slot.pending_d2h.deinit(std.heap.page_allocator);
    var cb = slot.cmdbuf;
    vkFreeCommandBuffers_fn.?(dev, g_command_pool, 1, @ptrCast(&cb));
    vkDestroyFence_fn.?(dev, slot.fence, null);
    g_streams.items[idx] = null;
    return VK_SUCCESS_RC;
}

// VK adaptation: VkEvent handle = query-slot index + 1 (0 reserved as the
// null sentinel matching CUDA's CUevent=0 contract).

fn procEventCreate(event_out: *vulkan_api.VkEvent, flags: c_uint) callconv(.c) VkResult {
    _ = flags;
    if (g_timestamp_query_pool == VK_NULL_HANDLE) return -1;
    if (g_query_index_free.items.len == 0) return -1;
    const idx = g_query_index_free.pop() orelse return -1;
    event_out.* = @as(vulkan_api.VkEvent, idx) + 1;
    return VK_SUCCESS_RC;
}

fn procEventRecord(event: vulkan_api.VkEvent, stream: VkStream) callconv(.c) VkResult {
    if (event == 0 or g_timestamp_query_pool == VK_NULL_HANDLE) return -1;
    const query_index: u32 = @intCast(event - 1);
    const reset_qp = vkCmdResetQueryPool_fn orelse return -1;
    const write_ts = vkCmdWriteTimestamp_fn orelse return -1;

    var cb: VkCommandBuffer = undefined;
    if (streamEntryFor(stream)) |se| {
        const rc = streamBeginIfNeeded(se);
        if (rc != VK_SUCCESS_RC) return rc;
        cb = se.cmdbuf;
    } else {
        const rc = beginOneShotCB();
        if (rc != VK_SUCCESS_RC) return rc;
        cb = g_command_buffer;
    }
    reset_qp(cb, g_timestamp_query_pool, query_index, 1);
    write_ts(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_timestamp_query_pool, query_index);
    if (streamEntryFor(stream) == null) {
        const rc = submitAndWait();
        if (rc != VK_SUCCESS_RC) return rc;
    }
    return VK_SUCCESS_RC;
}

fn procEventSynchronize(event: vulkan_api.VkEvent) callconv(.c) VkResult {
    if (event == 0) return -1;
    // VK adaptation: VkQueryPool has no per-event wait primitive; drain
    // the device so any cmdbuf that wrote this timestamp has completed.
    var i: usize = 0;
    while (i < g_streams.items.len) : (i += 1) {
        if (g_streams.items[i]) |*se| {
            if (se.recording) _ = streamEndAndWait(se);
        }
    }
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const wait_fn = vkDeviceWaitIdle_fn orelse return -1;
    return wait_fn(dev);
}

fn procEventElapsedTime(out_ms: *f32, start_event: vulkan_api.VkEvent, end_event: vulkan_api.VkEvent) callconv(.c) VkResult {
    out_ms.* = 0.0;
    if (start_event == 0 or end_event == 0) return -1;
    if (g_timestamp_query_pool == VK_NULL_HANDLE) return -1;
    const get_results = vkGetQueryPoolResults_fn orelse return -1;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const flags: u32 = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;

    var start_ts: u64 = 0;
    var end_ts: u64 = 0;
    const s_idx: u32 = @intCast(start_event - 1);
    const e_idx: u32 = @intCast(end_event - 1);
    var rc = get_results(dev, g_timestamp_query_pool, s_idx, 1, @sizeOf(u64), @ptrCast(&start_ts), @sizeOf(u64), flags);
    if (rc != VK_SUCCESS_RC) return rc;
    rc = get_results(dev, g_timestamp_query_pool, e_idx, 1, @sizeOf(u64), @ptrCast(&end_ts), @sizeOf(u64), flags);
    if (rc != VK_SUCCESS_RC) return rc;

    if (end_ts < start_ts) {
        out_ms.* = 0.0;
        return VK_SUCCESS_RC;
    }
    const delta_ticks: u64 = end_ts - start_ts;
    const delta_ns: f64 = @as(f64, @floatFromInt(delta_ticks)) * @as(f64, g_timestamp_period_ns);
    out_ms.* = @floatCast(delta_ns / 1_000_000.0);
    return VK_SUCCESS_RC;
}

fn procEventDestroy(event: vulkan_api.VkEvent) callconv(.c) VkResult {
    if (event == 0) return VK_SUCCESS_RC;
    const idx: u32 = @intCast(event - 1);
    if (idx >= g_query_capacity) return -1;
    const gpa = std.heap.page_allocator;
    g_query_index_free.append(gpa, idx) catch return -1;
    return VK_SUCCESS_RC;
}

fn procLaunchKernel(
    pipeline: usize,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared_bytes: c_uint,
    stream: VkStream,
    params: [*]?*anyopaque,
    extra: [*]?*anyopaque,
) callconv(.c) VkResult {
    // block_* are baked into the SPV via layout(local_size_x = ...).
    // shared_bytes / extra are CUDA-shaped surfaces with no VK analogue
    // — the SPV declares its own workgroup-shared storage if needed.
    _ = block_x;
    _ = block_y;
    _ = block_z;
    _ = shared_bytes;
    _ = extra;

    const meta = metaForPipeline(pipeline) orelse return -1;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const queue: VkQueue = @ptrFromInt(vulkan_api.compute_queue);

    // Pick the recording target. stream==0 uses the shared default
    // (g_command_buffer/g_fence + inline submit+wait); per-stream
    // handles record into the stream's cmdbuf and defer to stream_sync.
    var cb: VkCommandBuffer = undefined;
    var fence_for_wait: VkFence = 0;
    var inline_submit_wait = true;
    if (streamEntryFor(stream)) |se| {
        const rc = streamBeginIfNeeded(se);
        if (rc != VK_SUCCESS_RC) return rc;
        cb = se.cmdbuf;
        inline_submit_wait = false;
    } else {
        const rc = beginOneShotCB();
        if (rc != VK_SUCCESS_RC) return rc;
        cb = g_command_buffer;
        fence_for_wait = g_fence;
    }

    // Allocate a descriptor set out of the per-context pool if the kernel
    // declares any bindings. The pool grows under sustained launches —
    // we reset it lazily when it runs out (best-effort: VK_ERROR_OUT_OF_
    // POOL_MEMORY triggers a reset + retry).
    var dset: VkDescriptorSet = 0;
    if (meta.n_bindings > 0) {
        const layouts_arr = [_]VkDescriptorSetLayout{meta.layout};
        const ai = VkDescriptorSetAllocateInfo{
            .descriptorPool = g_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = layouts_arr[0..1].ptr,
        };
        var rc = vkAllocateDescriptorSets_fn.?(dev, &ai, @ptrCast(&dset));
        if (rc != VK_SUCCESS_RC) {
            // Pool exhausted: reset and retry once. vkResetDescriptorPool
            // invalidates every set previously allocated from the pool;
            // safe here because the codec's launches are submit+wait
            // serialised (default stream) or serialised through the
            // stream's fence (deferred path).
            _ = vkResetDescriptorPool_fn.?(dev, g_descriptor_pool, 0);
            rc = vkAllocateDescriptorSets_fn.?(dev, &ai, @ptrCast(&dset));
            if (rc != VK_SUCCESS_RC) return rc;
        }

        // params layout (matches the CUDA call sites verbatim):
        //   [0..n_bindings)         → pointers to VkDeviceBuffer handles
        //   [n_bindings..)          → pointers to push-constant scalar args
        // The codec passes each arg by-pointer (CUDA's cuLaunchKernel
        // convention); we dereference the handle and look up the VMA
        // (VkBuffer, VkDeviceSize) pair for binding.
        const gpa = std.heap.page_allocator;
        const writes = gpa.alloc(VkWriteDescriptorSet, meta.n_bindings) catch return -1;
        defer gpa.free(writes);
        const infos = gpa.alloc(VkDescriptorBufferInfo, meta.n_bindings) catch return -1;
        defer gpa.free(infos);
        var b: u32 = 0;
        while (b < meta.n_bindings) : (b += 1) {
            const p = params[b] orelse return -1;
            const hp: *const VkDeviceBuffer = @ptrCast(@alignCast(p));
            const handle = hp.*;
            const entry = lookupAlloc(handle) orelse return -1;
            infos[b] = .{
                .buffer = entry.buffer,
                .offset = 0,
                .range = VK_WHOLE_SIZE,
            };
            writes[b] = .{
                .dstSet = dset,
                .dstBinding = b,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = infos[b .. b + 1].ptr,
            };
        }
        vkUpdateDescriptorSets_fn.?(dev, meta.n_bindings, writes.ptr, 0, null);
    }

    vkCmdBindPipeline_fn.?(cb, VK_PIPELINE_BIND_POINT_COMPUTE, meta.pipeline);
    if (meta.n_bindings > 0) {
        const sets = [_]VkDescriptorSet{dset};
        vkCmdBindDescriptorSets_fn.?(cb, VK_PIPELINE_BIND_POINT_COMPUTE, meta.pl_layout, 0, 1, sets[0..1].ptr, 0, null);
    }
    if (meta.push_constant_size > 0) {
        // Concatenate the trailing push-constant scalars (each pointed
        // to by a params[n_bindings + i] entry) into a contiguous byte
        // buffer the size declared in KERNEL_DECLS. The codec encodes
        // them in the order the .comp's `layout(push_constant)` struct
        // declares.
        var pc_buf: [256]u8 = undefined;
        std.debug.assert(meta.push_constant_size <= pc_buf.len);
        var off: u32 = 0;
        var i: u32 = meta.n_bindings;
        while (off < meta.push_constant_size) : (i += 1) {
            const p = params[i] orelse return -1;
            // Treat each remaining slot as a u32 scalar; uvec2 splits
            // across two slots from the CUDA side. This matches every
            // current .comp where push constants are either single u32
            // fields or a uvec2 (which the CUDA side encodes as a u64
            // pair the host packs into one pointer; we copy 8 bytes if
            // there's room).
            const remaining = meta.push_constant_size - off;
            const copy_bytes: u32 = if (remaining >= 4) 4 else remaining;
            const src_bytes: [*]const u8 = @ptrCast(p);
            @memcpy(pc_buf[off..][0..copy_bytes], src_bytes[0..copy_bytes]);
            off += copy_bytes;
        }
        vkCmdPushConstants_fn.?(cb, meta.pl_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, meta.push_constant_size, @ptrCast(&pc_buf));
    }
    vkCmdDispatch_fn.?(cb, grid_x, grid_y, grid_z);

    if (inline_submit_wait) {
        // End + submit + fence wait on the default-stream path.
        var rc = vkEndCommandBuffer_fn.?(cb);
        if (rc != VK_SUCCESS_RC) return rc;
        const submit = VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = @ptrCast(&cb),
        };
        _ = vkResetFences_fn.?(dev, 1, @ptrCast(&fence_for_wait));
        rc = vkQueueSubmit_fn.?(queue, 1, @ptrCast(&submit), fence_for_wait);
        if (rc != VK_SUCCESS_RC) return rc;
        rc = vkWaitForFences_fn.?(dev, 1, @ptrCast(&fence_for_wait), 1, ~@as(u64, 0));
        return rc;
    }
    // Per-stream path: leave the cmdbuf recording; the codec joins via
    // procs.stream_sync. The descriptor set stays bound; the next
    // procs.launch_kernel on this stream will allocate a fresh one out
    // of the pool.
    return VK_SUCCESS_RC;
}
