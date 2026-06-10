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

// Iter 7: probe whether the physical device advertises
// VK_EXT_external_memory_host. Mirrors src_vulkan/device.zig:250
// (hasExternalMemoryHostExt). Returns false if the enumerate call
// fails, the device exports zero extensions, or the name isn't in
// the list. Caller passes the same VkPhysicalDevice that will go to
// vkCreateDevice so the answer matches what's actually available on
// the bound device.
fn hasExternalMemoryHostExt(pd: VkPhysicalDevice) bool {
    const enum_ext = vkEnumerateDeviceExtensionProperties_fn orelse return false;
    var count: u32 = 0;
    if (enum_ext(pd, null, &count, null) != VK_SUCCESS_RC) return false;
    if (count == 0) return false;
    // Bounded scan — modern drivers expose <200 extensions; 256 is safe.
    var buf: [256]VkExtensionProperties = undefined;
    var n: u32 = count;
    if (n > buf.len) n = @intCast(buf.len);
    const rc = enum_ext(pd, null, &n, @ptrCast(&buf));
    // VK_INCOMPLETE (5) is non-fatal: we just have a subset, which is
    // enough to check for the EXT name. Anything else is a hard failure.
    if (rc != VK_SUCCESS_RC and rc != 5) return false;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const name_slice = std.mem.sliceTo(buf[i].extensionName[0..], 0);
        if (std.mem.eql(u8, name_slice, "VK_EXT_external_memory_host")) return true;
    }
    return false;
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
const VK_QUEUE_GRAPHICS_BIT: u32 = 0x00000001;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;
const VK_QUEUE_TRANSFER_BIT: u32 = 0x00000004;

// Iter 11: VK adaptation — pipeline stage bit used to wait on the H2D
// transfer-queue submit's binary semaphore at the compute-queue submit.
// COMPUTE_SHADER_BIT (not TRANSFER_BIT) because the compute kernels are
// the producer-consumers of the H2D bytes; the wait drops the compute
// dispatch behind the transfer copy without holding back the host.
const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x00000800;

// Iter 11: VkSemaphore + VkSemaphoreCreateInfo sType for the binary
// semaphore that gates compute on H2D. VkSemaphore is a u64 dispatchable
// handle like VkFence.
const VkSemaphore = u64;
const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO: c_int = 9;
const VkSemaphoreCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
};

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
const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x00000001;
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

// VK adaptation: subgroup size pinned to 32 via REQUIRE_FULL_SUBGROUPS_BIT
// + VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Kernels in
// gpu_warp.glsl hardcode WARP_SIZE=32 (the warpShuffle/Ballot/Any
// cooperative ops are correct only at exactly 32 lanes). Without this
// pin, Intel iGPU (supports [8,32]) would silently miscompile while
// NVIDIA (only supports [32,32]) works by accident.
// Per vulkan_core.h (1.4.341):
//   VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT = 0x00000002
//   VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO = 1000225001
//   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES = 1000225000
// Prior values for the flag bit (0x8) and pipeline-stage sType (1000225002)
// were both wrong: 0x8 is a non-existent VkPipelineShaderStageCreateFlagBits
// value (silently ignored by the driver), and 1000225002 is
// VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES (driver
// therefore interpreted our pNext as the features struct, ignoring the
// requiredSubgroupSize field). Net effect: on Intel iGPU the pinned
// pipelines ran with the driver's default subgroupSize=16 (two 16-wide
// subgroups per 32-thread workgroup) and the warpShuffle/Ballot/Any-based
// kernels (especially encode) produced corrupted output. NVIDIA hardware
// only supports subgroupSize=32 so the broken pin was inert there.
// Validation layer surfaced both errors as
// VUID-VkPipelineShaderStageCreateInfo-pNext-pNext (wrong sType) +
// VUID-VkPipelineShaderStageCreateInfo-flags-parameter (flag 0x8 invalid).
const VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT: u32 = 0x00000002;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: c_int = 1000225001;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES: c_int = 1000225000;

const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    pNext: ?*anyopaque = null,
    requiredSubgroupSize: u32,
};

const VkPhysicalDeviceSubgroupSizeControlProperties = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    pNext: ?*anyopaque = null,
    minSubgroupSize: u32 = 0,
    maxSubgroupSize: u32 = 0,
    maxComputeWorkgroupSubgroups: u32 = 0,
    requiredSubgroupSizeStages: u32 = 0,
};

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
// Iter 15: VK adaptation — single-submit consolidation needs an explicit
// COMPUTE_SHADER_WRITE → TRANSFER_READ memory barrier between the LZ
// kernel and the D2H vkCmdCopyBuffer that now share one compute cmdbuf.
const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x00000040;
// Iter 4 (this iteration): COMPUTE_SHADER_WRITE → COMPUTE_SHADER_READ
// memory barrier between the Huffman decode kernel (writes entropy_scratch)
// and the LZ general decoder (reads it).
const VK_ACCESS_SHADER_READ_BIT: u32 = 0x00000020;

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

// Iter 11: vkCreateSemaphore / vkDestroySemaphore FFI for the binary
// semaphore that chains the H2D transfer-queue submit -> compute-queue
// submit. Both are core Vulkan 1.0.
const FnCreateSemaphore = *const fn (VkDevice, *const VkSemaphoreCreateInfo, ?*const anyopaque, *VkSemaphore) callconv(.c) VkResult;
const FnDestroySemaphore = *const fn (VkDevice, VkSemaphore, ?*const anyopaque) callconv(.c) void;
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

// VK adaptation (PHASE 2 — persistent VkPipelineCache): FFI typedefs for
// vkCreatePipelineCache / vkGetPipelineCacheData / vkDestroyPipelineCache.
// All three are core Vulkan 1.0. The cache is created once at init() time
// (loaded from disk if present), passed to every vkCreateComputePipelines
// call, and written back to disk on graceful shutdown. NCU traces showed
// the LZ kernel itself runs at ~parity with CUDA (77 ms vs 71 ms) but the
// process is ~190-300 ms slower than CUDA; the prime suspect is the
// SPIR-V → ISA recompile of 17 compute pipelines on every process start
// (currently called with VkPipelineCache=0 → no cross-process reuse). The
// driver validates the cache header (vendorID + deviceID + driverUUID +
// pipelineCacheUUID + dataSize) inside vkCreatePipelineCache itself and
// silently discards mismatched data, so the on-disk file is safe across
// driver upgrades and machine moves — worst case the next launch
// recompiles and rewrites a fresh blob.
const VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO: c_int = 17;
const VkPipelineCacheCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    initialDataSize: usize = 0,
    pInitialData: ?*const anyopaque = null,
};
const FnCreatePipelineCache = *const fn (VkDevice, *const VkPipelineCacheCreateInfo, ?*const anyopaque, *VkPipelineCache) callconv(.c) VkResult;
const FnGetPipelineCacheData = *const fn (VkDevice, VkPipelineCache, *usize, ?*anyopaque) callconv(.c) VkResult;
const FnDestroyPipelineCache = *const fn (VkDevice, VkPipelineCache, ?*const anyopaque) callconv(.c) void;
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

// ── Iter 7: VK_EXT_external_memory_host FFI ──────────────────────────
// Mirrors src_vulkan/vk_api.zig:1708-1776. The extension lets the
// decoder turn the caller's pageable host pointer into a VkDeviceMemory
// the GPU can DMA into directly — eliminating the dst_b→staging copy
// plus the post-fence host @memcpy that iter-4's two-buffer path pays.
// CUDA's equivalent is cuMemcpyDtoH_v2 against a pinned-host pointer
// (src/decode/decode_dispatch.zig:485).
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: c_int = 1000059001;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: c_int = 5;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: c_int = 12;
const VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT: c_int = 1000178000;
const VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT: c_int = 1000178001;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT: c_int = 1000178002;
// VK adaptation (valfix A): VkExternalMemoryBufferCreateInfo sType — value
// verified verbatim against C:/VulkanSDK/1.4.341.1/Include/vulkan/vulkan_core.h:292.
const VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO: c_int = 1000072000;
const VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT: u32 = 0x80;
const VK_SHARING_MODE_EXCLUSIVE: c_int = 0;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
// Iter 9: H2D import fast path uses TRANSFER_SRC on the imported VkBuffer
// (caller's host pointer is the SOURCE of a vkCmdCopyBuffer into the
// device-local persistent input buffer). procH2DAsync gates on this bit
// to mirror the iter-7/8 D2H pattern. Spec value per VK 1.0 core.
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
const VK_MAX_MEMORY_TYPES: usize = 32;
const VK_MAX_MEMORY_HEAPS: usize = 16;

const VkExtensionProperties = extern struct {
    extensionName: [256]u8,
    specVersion: u32,
};

const VkPhysicalDeviceProperties2 = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    pNext: ?*anyopaque = null,
    properties: VkPhysicalDeviceProperties = undefined,
};

const VkPhysicalDeviceExternalMemoryHostPropertiesEXT = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT,
    pNext: ?*anyopaque = null,
    minImportedHostPointerAlignment: VkDeviceSize = 0,
};

const VkImportMemoryHostPointerInfoEXT = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
    pNext: ?*const anyopaque = null,
    handleType: u32 = 0,
    pHostPointer: ?*anyopaque = null,
};

// VK adaptation (valfix A): VkExternalMemoryBufferCreateInfo for chaining
// into VkBufferCreateInfo.pNext on imported-memory-bound buffers. Required
// by VUID-vkBindBufferMemory-memory-02985 — if memory was imported with
// handleType H, the buffer must have been created with H listed in
// handleTypes too. Verified layout against
// C:/VulkanSDK/1.4.341.1/Include/vulkan/vulkan_core.h:5853 — sType, pNext,
// VkExternalMemoryHandleTypeFlags handleTypes.
const VkExternalMemoryBufferCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    handleTypes: u32 = 0,
};

const VkMemoryHostPointerPropertiesEXT = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
    pNext: ?*anyopaque = null,
    memoryTypeBits: u32 = 0,
};

const VkMemoryAllocateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize,
    memoryTypeIndex: u32,
};

const VkBufferCreateInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    size: VkDeviceSize,
    usage: u32,
    sharingMode: c_int = VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

const VkMemoryRequirements = extern struct {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memoryTypeBits: u32,
};

const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

const VkMemoryHeap = extern struct {
    size: VkDeviceSize,
    flags: u32,
};

const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [VK_MAX_MEMORY_TYPES]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [VK_MAX_MEMORY_HEAPS]VkMemoryHeap,
};

const FnEnumerateDeviceExtensionProperties = *const fn (VkPhysicalDevice, ?[*:0]const u8, *u32, ?[*]VkExtensionProperties) callconv(.c) VkResult;
const FnGetPhysicalDeviceProperties2 = *const fn (VkPhysicalDevice, *VkPhysicalDeviceProperties2) callconv(.c) void;
const FnGetPhysicalDeviceMemoryProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.c) void;
const FnCreateBuffer = *const fn (VkDevice, *const VkBufferCreateInfo, ?*const anyopaque, *VkBuffer) callconv(.c) VkResult;
const FnDestroyBuffer = *const fn (VkDevice, VkBuffer, ?*const anyopaque) callconv(.c) void;
const FnAllocateMemory = *const fn (VkDevice, *const VkMemoryAllocateInfo, ?*const anyopaque, *VkDeviceMemory) callconv(.c) VkResult;
const FnFreeMemory = *const fn (VkDevice, VkDeviceMemory, ?*const anyopaque) callconv(.c) void;
const FnBindBufferMemory = *const fn (VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) callconv(.c) VkResult;
const FnGetBufferMemoryRequirements = *const fn (VkDevice, VkBuffer, *VkMemoryRequirements) callconv(.c) void;
const FnGetMemoryHostPointerPropertiesEXT = *const fn (VkDevice, u32, *const anyopaque, *VkMemoryHostPointerPropertiesEXT) callconv(.c) VkResult;

// A-008 (BDA): vkGetBufferDeviceAddress returns a u64 VkDeviceAddress for a
// VkBuffer that was created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT.
// The address is stable for the lifetime of the VkBuffer; we cache it on
// the AllocEntry when first queried so the encode hot path does not pay
// the FFI cost on every dispatch. The struct layout is the one-field
// VkBufferDeviceAddressInfo (sType=PHYS_DEV_VULKAN_1_2 const below + buffer).
const VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO: c_int = 1000244001;
const VkBufferDeviceAddressInfo = extern struct {
    sType: c_int = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    pNext: ?*const anyopaque = null,
    buffer: VkBuffer,
};
const FnGetBufferDeviceAddress = *const fn (VkDevice, *const VkBufferDeviceAddressInfo) callconv(.c) u64;

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

// Iter 11: VK adaptation — resolved at init() alongside other device-level
// entry points. Used by procStreamCreate / procStreamDestroy and by
// streamEndAndWait's 2-submit dance.
var vkCreateSemaphore_fn: ?FnCreateSemaphore = null;
var vkDestroySemaphore_fn: ?FnDestroySemaphore = null;
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
// VK adaptation (PHASE 2 — persistent VkPipelineCache): fn-ptr slots
// resolved alongside other device-level entries in init(). All three are
// core 1.0 so resolution is always expected to succeed; the create-or-
// load helper still gates everything behind a non-null check so a
// pathological gdpa miss falls back to "no cache" (slow init, correct
// output) rather than NPE.
var vkCreatePipelineCache_fn: ?FnCreatePipelineCache = null;
var vkGetPipelineCacheData_fn: ?FnGetPipelineCacheData = null;
var vkDestroyPipelineCache_fn: ?FnDestroyPipelineCache = null;
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

// Iter 7: VK_EXT_external_memory_host fn-ptr slots. Resolved at init()
// when the device advertises the extension (NVIDIA RTX 4060 Ti does;
// most Intel iGPUs do; AMD does). When unavailable the procD2HAsync
// import fast path is skipped and the iter-4 staging path stays in use.
var vkEnumerateDeviceExtensionProperties_fn: ?FnEnumerateDeviceExtensionProperties = null;
var vkGetPhysicalDeviceProperties2_fn: ?FnGetPhysicalDeviceProperties2 = null;
var vkGetPhysicalDeviceMemoryProperties_fn: ?FnGetPhysicalDeviceMemoryProperties = null;
var vkCreateBuffer_fn: ?FnCreateBuffer = null;
var vkDestroyBuffer_fn: ?FnDestroyBuffer = null;
var vkAllocateMemory_fn: ?FnAllocateMemory = null;
var vkFreeMemory_fn: ?FnFreeMemory = null;
var vkBindBufferMemory_fn: ?FnBindBufferMemory = null;
var vkGetBufferMemoryRequirements_fn: ?FnGetBufferMemoryRequirements = null;
var vkGetMemoryHostPointerPropertiesEXT_fn: ?FnGetMemoryHostPointerPropertiesEXT = null;

// A-008 (BDA): vkGetBufferDeviceAddress slot. Resolved at init() once the
// VkDevice exists; bufferDeviceAddress is already enabled at vkCreateDevice
// (Vulkan 1.2 features struct at line 2187). Used by the encode LZ kernel
// to bypass the per-binding 4 GiB SSBO range cap on the hash table.
var vkGetBufferDeviceAddress_fn: ?FnGetBufferDeviceAddress = null;

// VK_EXT_external_memory_host gate. Set at init() to true iff the
// device advertises the extension AND the loader resolved every
// fn-ptr the import path needs. False keeps procD2HAsync on the
// iter-4 staging path.
var g_ext_memory_host_supported: bool = false;
// minImportedHostPointerAlignment cached from
// VkPhysicalDeviceExternalMemoryHostPropertiesEXT. Typically 4096 on
// NVIDIA. Caller's pointer + import size must both be multiples of
// this. Stays 0 when the extension is unavailable.
var g_imported_host_alignment: u64 = 0;
// Cached physical-device memory properties used by the import path
// to pick a HOST_VISIBLE memory type compatible with the imported
// pointer. Populated at init().
var g_phys_mem_props: VkPhysicalDeviceMemoryProperties = .{
    .memoryTypeCount = 0,
    .memoryTypes = @splat(.{ .propertyFlags = 0, .heapIndex = 0 }),
    .memoryHeapCount = 0,
    .memoryHeaps = @splat(.{ .size = 0, .flags = 0 }),
};

// Iter 8 subfix 1: LRU cache of imported (host_ptr, size) → (VkBuffer,
// VkDeviceMemory). The -db bench replays the same pinned dst pointer
// across all runs, so a tiny cache lets runs 2..N skip the
// vkCreateBuffer + vkAllocateMemory + vkBindBufferMemory triplet
// (~1 ms on NVIDIA per iter 7 measurements). Entries are evicted when
// the cache fills (LRU); each entry's underlying handles are destroyed
// on eviction or at process exit (no explicit deinit — the dispatcher
// has none in iter 7 either). The `in_flight` flag is critical: a
// cached entry whose GPU writes haven't drained yet must NOT be reused
// by another D2H request (would race with the in-progress copy).
const ImportedEntry = struct {
    host_ptr: usize, // @intFromPtr key (avoids *anyopaque equality issues)
    size: usize, // requested d2h size (cache match requires exact)
    vk_buf: VkBuffer,
    vk_mem: VkDeviceMemory,
    last_used_ns: i64,
    in_flight: bool, // true while a GPU op is using vk_buf
    // Iter 9: which TRANSFER usage bit the VkBuffer was created with.
    // H2D needs TRANSFER_SRC; D2H needs TRANSFER_DST. The cache key is
    // (host_ptr, size, usage_src) so the same caller pointer can hold a
    // SRC import (compressed input upload) and a DST import (decoded
    // output) simultaneously without colliding.
    usage_src: bool,
};
const G_IMPORT_CACHE_CAP: usize = 16;
var g_import_cache: [G_IMPORT_CACHE_CAP]?ImportedEntry = @splat(@as(?ImportedEntry, null));

// Iter 8 subfix 2: log the chosen memoryTypeIndex + propertyFlags once
// (on first successful import). Read under SLZ_VK_PROFILE_PHASES=1 to
// verify HOST_CACHED was selected on NVIDIA. Pure diagnostic.
var g_import_mem_type_logged: bool = false;

// VK adaptation (OOM noise fix, 2026-06-07): sticky per-direction
// "import disabled" flags for VK_EXT_external_memory_host. NVIDIA's
// driver returns VK_ERROR_OUT_OF_DEVICE_MEMORY for every
// VkImportMemoryHostPointerInfoEXT alloc on the H2D direction
// (usage_src=true) of an RTX 4060 Ti — likely because the dedicated
// transfer queue family can't access importable HOST_VISIBLE memory
// types on this driver, even though they pass the
// vkGetMemoryHostPointerPropertiesEXT compatibility filter. The first
// failure flips the corresponding bool below; tryImportHostBuffer then
// short-circuits to staging fallback up front, eliminating the
// BestPractices-Error-Result warning the validation layer prints on
// every retry. D2H (usage_src=false) tracked separately because it
// reliably succeeds on HOST_CACHED here.
var g_import_disabled_h2d: bool = false;
var g_import_disabled_d2h: bool = false;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT_CONST: u32 = 0x00000004;
const VK_MEMORY_PROPERTY_HOST_CACHED_BIT_CONST: u32 = 0x00000008;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT_CONST: u32 = 0x00000001;

// Iter 8 subfix 3: profiling counter for prepareImportHostBuffer.
// Accumulates the time spent in vkCreateBuffer + vkAllocateMemory +
// vkBindBufferMemory on cache MISSES (cache hits are ~0). The work is
// overlapped with runBackHalf in fullGpuLaunchImpl, so this is the
// "would-have-been" cost without subfix 3. Printed under
// SLZ_VK_PROFILE_PHASES=1 via decode_dispatch's phase printer.
pub var g_import_prep_cache_hits: u64 = 0;
pub var g_import_prep_cache_misses: u64 = 0;

// Iter 12: H2D path-taken counters. Increment in procH2DAsync at the
// site of each branch so SLZ_VK_PROFILE_PHASES output can confirm the
// big comp_input H2D actually went through the import path (vs falling
// through to staging on alignment failure / extension absence). Printed
// next to the prep-cache stats by decode_dispatch's phase printer.
pub var g_h2d_path_prepared_import: u64 = 0; // hit on se.prepared_h2d_import
pub var g_h2d_path_inline_import: u64 = 0;   // inline tryImportHostBuffer in procH2DAsync
pub var g_h2d_path_staging: u64 = 0;         // iter-4 staging fallback

// Iter 14 (B): LRU import-cache telemetry. Atomic counters incremented
// inside lookupImportedBuffer / insertImportedBuffer to surface whether
// the iter-8 cache is hitting reliably in steady state. Printed and
// reset by decode_dispatch.printAndResetPhaseProfile under
// SLZ_VK_PROFILE_PHASES=1. Pure instrumentation — no behaviour change.
//   hit:      lookupImportedBuffer found a matching, not-in-flight entry
//   miss:     lookupImportedBuffer returned null (caller will allocate)
//   eviction: insertImportedBuffer displaced an LRU slot (vkDestroy/Free)
pub var g_import_cache_hits: std.atomic.Value(u64) = .{ .raw = 0 };
pub var g_import_cache_misses: std.atomic.Value(u64) = .{ .raw = 0 };
pub var g_import_cache_evictions: std.atomic.Value(u64) = .{ .raw = 0 };

// ── Module-private bring-up state ────────────────────────────────────
// VK adaptation: the shared command pool + per-context staging buffer
// the procs.h2d / procs.d2h closures funnel work through. Sized at
// init(); grown lazily if a larger copy needs to stage.
var g_command_pool: VkCommandPool = VK_NULL_HANDLE;
var g_command_buffer: VkCommandBuffer = null;
var g_fence: VkFence = VK_NULL_HANDLE;

// VK adaptation (PHASE 2 — persistent VkPipelineCache): single process-wide
// VkPipelineCache that backs every vkCreateComputePipelines call (decode
// kernels here + encode kernels in srcVK/encode/module_loader.zig, which
// reads this through pipelineCacheHandle()). Created lazily by
// loadOrCreatePipelineCache the first time init() reaches the pipeline-
// build section; populated from disk if a previously-saved blob exists at
// pipelineCachePath(). Persisted back to disk by savePipelineCache (wired
// to atexit so CLI one-shots benefit on the next invocation, not just on
// graceful API teardown). The driver validates the cache header (vendorID
// + deviceID + driverUUID + pipelineCacheUUID + dataSize) inside
// vkCreatePipelineCache itself per Vulkan 1.0 spec § 10.5.4 and discards
// mismatched payloads with VK_SUCCESS + an empty cache — no application
// hash/checksum needed. NCU + indirect evidence proved the LZ kernel is
// at parity with CUDA (77 ms vs 71 ms); the ~190-300 ms process-launch
// gap lives in the per-launch SPIR-V → ISA recompile of 17 compute
// pipelines, which this cache amortizes to the first run.
var g_pipeline_cache: VkPipelineCache = VK_NULL_HANDLE;
var g_pipeline_cache_path_buf: [512]u8 = @splat(0);
var g_pipeline_cache_path_len: usize = 0;
var g_pipeline_cache_atexit_registered: bool = false;

// Iter 11: VK adaptation — dedicated VK_QUEUE_TRANSFER_BIT queue + its
// per-queue command pool. On NVIDIA discrete GPUs the dedicated transfer
// family fronts the on-chip copy engine, mirroring CUDA's
// cuMemcpyHtoDAsync auto-routing. On platforms with no dedicated transfer
// family (Intel iGPU, AMD APUs), g_transfer_queue_family == compute and
// g_has_dedicated_transfer == false; the pool + queue handles are still
// distinct objects but reference the same family, which behaves like a
// single-queue setup with extra cmdbuf overhead. Fence-wait cost is
// unchanged in that case.
var g_transfer_queue: usize = 0;
var g_transfer_queue_family: u32 = 0;
var g_transfer_cmd_pool: VkCommandPool = VK_NULL_HANDLE;
var g_has_dedicated_transfer: bool = false;
// VK adaptation (encode D2H gather): one-shot cmdbuf + fence dedicated
// to procD2HOffsetGather. Sits on g_transfer_cmd_pool so submits land
// on the dedicated DMA queue (g_transfer_queue, qf=1 on NVIDIA). Reset
// on every gather call. Allocated lazily on the first gather call so
// init() stays branch-free. Per-gather lifecycle:
//   1. vkResetCommandBuffer + vkBeginCommandBuffer
//   2. vkCmdCopyBuffer (regionCount = N)
//   3. vkEndCommandBuffer + vkResetFences + vkQueueSubmit + vkWaitForFences
//   4. Clear in-flight marker on the cached import slot (if any)
// Single owner (the encode hot path's gather call is wrapped in
// lockEncodeDispatcherMutex by fast_framed.compressFramedOne), so no
// external serialization needed.
var g_xfer_oneshot_cb: VkCommandBuffer = null;
var g_xfer_oneshot_fence: VkFence = VK_NULL_HANDLE;
var g_xfer_oneshot_ready: bool = false;
var g_staging_buffer: vma.VkBuffer = 0;
var g_staging_alloc: vma.VmaAllocation = null;
var g_staging_size: usize = 0;
var g_staging_mapped: ?*anyopaque = null;
var g_descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;

// Iter 4c: dedicated pool for per-launch transient descriptor sets used
// by the sub-region-bind path (procLaunchKernel called with non-null
// binding_offsets). Separate from g_descriptor_pool so resetting it does
// not invalidate iter 14's persistent (stream × pipeline) sets. The
// pool is reset opportunistically when g_transient_set_count crosses
// g_transient_set_reset_threshold — the reset is gated on
// vkDeviceWaitIdle (the only point Vulkan guarantees no in-flight
// cmdbuf still references the sets). Resetting at a coarse threshold
// rather than per-decode keeps the wait-idle cost off the hot path
// while still bounding pool memory. (The lock SRWLOCK is declared
// further down at ~1601; we forward-declare its uses via raw Win32
// AcquireSRWLockExclusive once SRWLOCK is in scope.)
var g_transient_descriptor_pool: VkDescriptorPool = VK_NULL_HANDLE;
var g_transient_set_count: u32 = 0;
const g_transient_set_reset_threshold: u32 = 3500; // pool cap = 4096; leave headroom

// Timestamp query pool used by procs.event_*. Each VkEvent handle from
// procs.event_create is an index into this pool's query slots; the free
// list reclaims slots after procs.event_destroy. Capacity is sized for
// many events in flight at once (decode_dispatch records ~10 per frame).
var g_timestamp_query_pool: VkQueryPool = VK_NULL_HANDLE;
const g_query_capacity: u32 = 4096;
var g_query_index_free: std.ArrayListUnmanaged(u32) = .empty;
var g_timestamp_period_ns: f32 = 1.0;

// Iter 4c: device limit captured at init so procLaunchKernel can validate
// per-binding SSBO offsets. CUDA's CUdeviceptr `base + offset` arithmetic
// has no alignment constraint; Vulkan's VkDescriptorBufferInfo.offset
// must be a multiple of VkPhysicalDeviceLimits.minStorageBufferOffsetAlignment
// (16 on NVIDIA, 64 on Intel iGPU is typical). 0 = init has not run yet
// (or the property query failed); procLaunchKernel treats 0 as "no check".
var g_min_storage_buffer_offset_alignment: u64 = 0;

// 2026-06-10: maxStorageBufferRange (4 GiB - 1 on desktop GPUs) captured
// so procLaunchKernel can clamp per-binding ranges. A single allocation
// may legitimately exceed the per-binding cap — the 1 GB L3+ entropy
// scratch is ~6 GB, bound per-region (lit/tok/off16) at offsets that
// keep each kernel's addressable window ≤ 2.14 GB (WALK_MAX_CHUNKS ×
// ENTROPY_SCRATCH_SLOT_BYTES). 0 = property query did not run; the
// clamp falls back to 4 GiB - 1.
var g_max_storage_buffer_range: u64 = 0;

// ── Per-kernel GPU profiling (SLZ_VK_PROFILE_DECODE=1) ───────────────
// Reserves a fixed range of query-pool slots [g_profile_slot_base ..
// g_profile_slot_base + g_profile_slot_count) carved out of
// g_timestamp_query_pool. procLaunchKernel records a start+end
// timestamp around every vkCmdDispatch when g_profile_enabled is true.
// readbackAndPrintProfile (called from cli/bench_decompress.zig) drains
// streams, reads all timestamps, sums per KernelKind, prints "kper:"
// lines to stderr, and resets the recording cursor. Pure measurement
// instrumentation — no algorithm change. See M5/M6/M7 in
// srcVK/gameplan.md for the env-var contract.
var g_profile_enabled: bool = false;
const g_profile_max_records: u32 = 1024;
const g_profile_slot_count: u32 = g_profile_max_records * 2;
const g_profile_slot_base: u32 = g_query_capacity - g_profile_slot_count;
const ProfileRecord = struct { kind_index: i32, start_slot: u32, end_slot: u32 };
var g_profile_records: [g_profile_max_records]ProfileRecord = undefined;
var g_profile_record_count: u32 = 0;
var g_profile_overflow_count: u32 = 0;

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

// Iter 7: per-D2H import record (VK_EXT_external_memory_host fast path).
// procD2HAsync allocates a transient VkBuffer + VkDeviceMemory pair that
// wraps the caller's host pointer; vkCmdCopyBuffer targets it as the
// destination of a DEVICE_LOCAL → HOST_VISIBLE-imported copy. The
// resources MUST stay live until the fence the GPU writes against has
// signaled, so we queue them here and (iter 7) free in streamEndAndWait
// after vkWaitForFences. No host @memcpy is needed — the bytes land in
// the caller's buffer directly. Mirrors CUDA's cuMemcpyDtoH_v2 against
// a pinned destination (src/decode/decode_dispatch.zig:485).
//
// Iter 8 subfix 1: the (VkBuffer, VkDeviceMemory) pair is now owned by
// the LRU import cache (g_import_cache below) — streamEndAndWait only
// clears the in-flight marker; eviction or process teardown frees the
// underlying handles. `cache_idx` points back into g_import_cache.
const PendingImportedD2H = struct {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    cache_idx: i32 = -1, // -1 = legacy uncached path (iter 7 fallback)
    // Iter 9: byte offset within the imported VkBuffer where the actual
    // requested host pointer sits. Non-zero when the caller's pointer
    // wasn't page-aligned — we import the surrounding page-aligned
    // region and let vkCmdCopyBuffer skip the leading pad via srcOffset
    // (H2D side) / dstOffset (D2H side). Lets us import arbitrary
    // sub-page slices of a page-aligned parent allocation (e.g. the
    // compressed-input block_payload that starts at offset header_size
    // within a pinned src buffer).
    region_offset: u64 = 0,
};

const StreamEntry = struct {
    cmdbuf: VkCommandBuffer,
    fence: VkFence,
    recording: bool, // true if Begin has been called but End/Submit not yet
    // Iter 11: VK adaptation — second cmdbuf allocated from the dedicated
    // transfer-queue command pool, plus a binary semaphore the transfer
    // submit signals and the compute submit waits on. procH2DAsync records
    // vkCmdCopyBuffer into transfer_cmdbuf so the H2D DMA runs on the
    // copy engine in parallel with the compute queue's setup work
    // (descriptor-set allocate + bind + dispatch). Mirrors CUDA's
    // cuMemcpyHtoDAsync auto-routing to the dedicated DMA engine.
    //
    // When g_has_dedicated_transfer == false (Intel iGPU / AMD APU) the
    // transfer pool sits on the compute family; the cmdbuf split still
    // works (two cmdbufs into the same queue family is legal) but the
    // semaphore wait degenerates to a no-op serializer and the win is
    // ~0. Bench captures this as "dedicated=false" in the init log.
    transfer_cmdbuf: VkCommandBuffer = null,
    transfer_recording: bool = false,
    // Iter 13: VK adaptation — flipped true by procStreamFlushTransfer
    // when the transfer cmdbuf has been End+Submitted EARLY (at the end
    // of uploadInputAndPrefixSum). streamEndAndWait then skips the
    // transfer end+submit but PRESERVES the compute submit's semaphore
    // wait so the h2d_sem the early-flush submit signaled is properly
    // consumed. Reset to false after the wait completes.
    //
    // This early-submit lets the dedicated DMA engine start the H2D
    // copies in parallel with the host-side runBackHalf prep (kernel
    // record + descriptor binding + prepareImportedOutputBuffer),
    // mirroring CUDA cuMemcpyHtoD's natural ordering where the
    // synchronous host call has already drained the copy engine by the
    // time the back-half stream_sync fires.
    transfer_already_submitted: bool = false,
    h2d_sem: VkSemaphore = VK_NULL_HANDLE,
    staging_buffer: vma.VkBuffer = 0,
    staging_alloc: vma.VmaAllocation = null,
    staging_mapped: ?*anyopaque = null,
    staging_size: usize = 0,
    staging_used: usize = 0,
    pending_d2h: std.ArrayListUnmanaged(PendingD2H) = .empty,
    // Iter 7: D2H records whose destination was imported via
    // VK_EXT_external_memory_host. The (VkBuffer, VkDeviceMemory) pair
    // must outlive the fence wait; iter 7 freed the handles after
    // vkWaitForFences in streamEndAndWait, but iter 8 subfix 1 hands
    // ownership to g_import_cache — streamEndAndWait now only flips
    // each entry's in_flight bit via clearImportInFlight(cache_idx).
    // Entries with cache_idx == -1 (cache full at insert time) still
    // get the iter-7 destroy treatment as a defensive fallback.
    pending_imported_d2h: std.ArrayListUnmanaged(PendingImportedD2H) = .empty,
    // Iter 8 subfix 3: pre-prepared import record. fullGpuLaunchImpl
    // calls module_loader.prepareImportHostBuffer BEFORE runBackHalf to
    // overlap the vkCreateBuffer+vkAllocateMemory cost with the LZ
    // kernel. procD2HAsync consumes this in preference to running
    // tryImportHostBuffer inline (which would serialize after the
    // back-half fence wait). The dispatcher owns the lifecycle: stashes
    // here at prep time, consumed (set back to null) by the next
    // procD2HAsync call on this stream.
    prepared_d2h_import: ?PendingImportedD2H = null,
    // Iter 9: H2D analog of prepared_d2h_import. Dispatcher pre-prepares
    // an imported TRANSFER_SRC buffer for the compressed-input upload so
    // procH2DAsync's first big copy skips the inline import cost. The
    // import_src_host_addr field keys the prep to a specific source
    // pointer so two queued H2Ds on the same stream don't accidentally
    // consume each other's prep. Cleared after consumption (or
    // streamEndAndWait when unused).
    prepared_h2d_import: ?PendingImportedD2H = null,
    prepared_h2d_host_addr: usize = 0,
    prepared_h2d_size: usize = 0,
    // Iter 14 (A): persistent VkDescriptorSet per (StreamEntry × decode
    // pipeline). Allocated lazily on first procLaunchKernel for each
    // pipeline kind from g_descriptor_pool. Each stream owns its own
    // slot so concurrent decodes on distinct VkStreams race-freely; the
    // descriptor-set write/bind work shrinks from
    // vkAllocateDescriptorSets + vkUpdateDescriptorSets + vkCmdBind to
    // just vkUpdateDescriptorSets + vkCmdBind per launch. Slot index
    // matches the KERNEL_DECLS index for decode kernels; the stream==0
    // path (encode + sync-mode default) uses KernelMeta.persistent_desc_set_default
    // instead because no StreamEntry exists there.
    persistent_desc_sets: [KERNEL_DECLS.len]VkDescriptorSet = @splat(0),

    // VK adaptation (valfix B — VUID-vkCmdCopyBuffer-commandBuffer-recording):
    // staging buffers that were grown mid-recording (ensureStreamStaging
    // outgrew the current size while the cmdbuf still held vkCmdCopyBuffer
    // references to the OLD buffer). Deferring their VMA destroy until
    // streamEndAndWait fires (after fence wait) prevents the cmdbuf from
    // transitioning to INVALID state per Vulkan spec. Single-element in
    // virtually every decode (1 MiB → 2 MiB grow on web.txt). Cleared after
    // each fence wait, growth-amortized — no leak.
    deferred_staging_destroy: std.ArrayListUnmanaged(StagingHandle) = .empty,
};

// Companion record for StreamEntry.deferred_staging_destroy.
const StagingHandle = struct {
    buffer: vma.VkBuffer,
    allocation: vma.VmaAllocation,
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

// VK adaptation: init() reads + writes the global
// `vulkan_api.init_state` non-atomically (uninit → in_progress → ready).
// Pre-iter5 the only test of this path was the CLI (single-threaded),
// so the race was invisible; ptest_vk's 16-worker runner exposes it:
// thread A flips `.uninit` → `.in_progress` and starts a ~10 ms
// LoadLibraryA + vkCreateInstance + vkCreateDevice + pipeline-build
// sequence, while thread B enters init(), reads `.in_progress`, and
// short-circuits to `return false` — which the dispatcher surfaces as
// error.BackendNotAvailable. Serialize the whole init() body through a
// dedicated SRWLOCK so any concurrent re-entry waits for `.ready`
// before evaluating its branch. Hot-path callers (post-init) still
// short-circuit on the `.ready` check before taking the lock.
var g_init_lock: SRWLOCK = .{};
// Separate encode-init lock so the encode-side init can serialize its
// own one-shot setup without recursively acquiring the decode init
// lock (Win32 SRWLOCK is non-recursive; encode init() chains into
// decode_driver.init() which itself takes g_init_lock).
var g_encode_init_lock: SRWLOCK = .{};

pub fn lockInitMutex() void {
    AcquireSRWLockExclusive(&g_init_lock);
}

pub fn unlockInitMutex() void {
    ReleaseSRWLockExclusive(&g_init_lock);
}

pub fn lockEncodeInitMutex() void {
    AcquireSRWLockExclusive(&g_encode_init_lock);
}

pub fn unlockEncodeInitMutex() void {
    ReleaseSRWLockExclusive(&g_encode_init_lock);
}

// VK adaptation: encode-side serialization for the sync path. The encode
// hot path (gpuCompressImpl + gpuAssembleFrameImpl + gpuFrameAssembleImpl)
// reads/writes per-context persistent device buffers (EncodeContext.
// d_input_persist / d_output_persist / d_descs_persist / d_sizes_persist
// / d_hash_persist / d_asm_out / d_frame_*) via encode_context.ensureBuf
// AND submits its sync H2D/D2H copies through the singleton
// g_command_buffer + g_staging_buffer in procH2D/procD2H, AND submits
// kernels to vkQueueSubmit on the shared compute queue. Vulkan requires
// vkQueueSubmit on the same VkQueue to be externally synchronized
// across threads — without that, two test workers' submits can corrupt
// each other's command buffers and (silently) produce all-zero output
// or surface as DestinationTooSmall in the framer's bounds checks.
// We reuse `g_dispatcher_lock` itself (not a parallel encode lock) so
// any concurrent decode call ALSO serializes against the encode submit
// for the same queue. Async-mode callers (per-worker EncodeContext +
// non-zero work_stream) skip the lock and serialize per-stream.
pub fn lockEncodeDispatcherMutex() void {
    AcquireSRWLockExclusive(&g_dispatcher_lock);
}

pub fn unlockEncodeDispatcherMutex() void {
    ReleaseSRWLockExclusive(&g_dispatcher_lock);
}

// VK adaptation: the alloc registry (g_allocs / g_host_allocs) is touched
// by every procs.malloc_device / free_device / malloc_host / free_host
// call. Per-test EncodeContext / DecodeContext use this surface to
// allocate their own persistent device buffers; without a registry lock,
// two ptest_vk workers concurrently appending to g_allocs corrupt the
// ArrayListUnmanaged backing store (segfault, exit code 5). The lock is
// narrowly scoped to registry mutation — VMA itself is internally
// thread-safe per its docs (VMA's default VmaAllocator uses an internal
// CAS on the heap freelist).
var g_alloc_registry_lock: SRWLOCK = .{};

pub fn lockAllocRegistry() void {
    AcquireSRWLockExclusive(&g_alloc_registry_lock);
}

pub fn unlockAllocRegistry() void {
    ReleaseSRWLockExclusive(&g_alloc_registry_lock);
}

// Iter 4c: lock protecting g_transient_descriptor_pool resets +
// g_transient_set_count mutation. Held across the check + reset + alloc
// triple so a concurrent ptest_vk worker can't sneak a fresh alloc into
// a pool we're about to reset.
var g_transient_lock: SRWLOCK = .{};
fn lockTransient() void {
    AcquireSRWLockExclusive(&g_transient_lock);
}
fn unlockTransient() void {
    ReleaseSRWLockExclusive(&g_transient_lock);
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
        // VK adaptation (valfix B — VUID-vkCmdCopyBuffer-commandBuffer-recording):
        // the OLD staging buffer may be referenced by vkCmdCopyBuffer calls
        // already recorded into the transfer/compute cmdbuf (the same
        // per-stream staging is shared between procH2DAsync staging-fallback
        // copies, and we grow lazily inside streamStagingBump). Destroying
        // it mid-recording transitions the cmdbuf to INVALID state per
        // Vulkan spec (VkBuffer destroyed while bound to a recording
        // cmdbuf), and the validation layer flags every subsequent
        // vkCmdCopyBuffer + vkEndCommandBuffer call. Defer the actual VMA
        // destroy until streamEndAndWait fires (after fence wait — at which
        // point neither the cmdbuf nor the GPU can possibly reference the
        // old buffer). Empirically this fires once per first decode of
        // web.txt (1 MiB initial → 2 MiB needed) and rarely thereafter
        // because the growth is amortized.
        const gpa = std.heap.page_allocator;
        entry.deferred_staging_destroy.append(gpa, .{
            .buffer = entry.staging_buffer,
            .allocation = entry.staging_alloc,
        }) catch {
            // Catastrophic — best effort: destroy immediately and accept
            // the VUID rather than leak. (page_allocator.append basically
            // never fails on a freshly-reset stream.)
            vma.destroyBuffer(allocator(), entry.staging_buffer, entry.staging_alloc);
        };
        entry.staging_buffer = 0;
        entry.staging_alloc = null;
        entry.staging_mapped = null;
        entry.staging_size = 0;
    }
    var size: usize = if (entry.staging_size == 0) STAGING_INITIAL_SIZE else entry.staging_size;
    while (size < needed) size *= 2;
    var buf: vma.VkBuffer = 0;
    var alc: vma.VmaAllocation = null;
    // Iter 11: VK adaptation — per-stream staging buffer is read by the
    // transfer queue (procH2DAsync) AND written by the compute queue's
    // D2H readback path (procD2HAsync staging fallback). CONCURRENT mode
    // when both queue families are distinct so we skip ownership
    // transfer barriers.
    var stg_qfs = [_]u32{ vulkan_api.compute_queue_family, g_transfer_queue_family };
    const stg_sharing: c_int = if (g_has_dedicated_transfer) vma.VK_SHARING_MODE_CONCURRENT else vma.VK_SHARING_MODE_EXCLUSIVE;
    const stg_qf_count: u32 = if (g_has_dedicated_transfer) 2 else 0;
    const stg_qf_ptr: ?[*]const u32 = if (g_has_dedicated_transfer) @ptrCast(&stg_qfs) else null;
    const bci = vma.VkBufferCreateInfo{
        .size = size,
        .usage = vma.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vma.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = stg_sharing,
        .queueFamilyIndexCount = stg_qf_count,
        .pQueueFamilyIndices = stg_qf_ptr,
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
    // Iter 14 (A): persistent descriptor set used on the stream==0 (default
    // / one-shot cmdbuf) path. Allocated lazily on first procLaunchKernel.
    // The default-stream path is already serialized by g_dispatcher_lock
    // (sync-mode entry in decode_dispatch.fullGpuLaunchImpl) so a single
    // shared set per pipeline is race-free; the per-stream path uses
    // StreamEntry.persistent_desc_sets instead so concurrent decodes on
    // distinct VkStreams cannot race on the descriptor-set write/bind.
    persistent_desc_set_default: VkDescriptorSet = 0,
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
//
// VK adaptation: pin_subgroup_32 flags whether vkCreateComputePipelines
// chains REQUIRE_FULL_SUBGROUPS_BIT + requiredSubgroupSize=32 for this
// kernel. Only kernels whose local_size_x is a multiple of 32 AND that
// actually use warpShuffle/Ballot/Any (i.e. include gpu_warp.glsl
// cooperatively) need the pin. The single-thread guard kernels run with
// local_size_x=1; pinning them would fail pipeline creation (spec
// requires workgroup-X be a multiple of subgroupSize when
// REQUIRE_FULL_SUBGROUPS_BIT is set).
const KernelDecl = struct {
    kind: KernelKind,
    n_bindings: u32,
    push_constant_size: u32,
    pin_subgroup_32: bool = false,
};

const KERNEL_DECLS = [_]KernelDecl{
    // 2026-06-10: bindings 6/7 are the tok / off16 region views of the
    // SAME entropy-scratch buffer (bound at tok_offset / off16_offset via
    // binding_offsets) so every in-shader byte offset stays region-relative
    // (≤ 2.14 GB) and the 1 GB L3+ ~6 GB scratch never needs a >4 GiB
    // binding. CUDA keeps one u64 pointer + stride; this is the VK
    // equivalent under maxStorageBufferRange (see PortAdaptations A-024).
    .{ .kind = .kernel_fn, .n_bindings = 8, .push_constant_size = 16, .pin_subgroup_32 = true },
    // srcVK/decode/lz_decode_raw_kernel.comp:25-50 declares 4 SSBO
    // bindings (CompressedBuf, ChunksBuf, DstBuf, TotalChunksBuf) plus
    // a 2× u32 push-constant block (chunks_per_group, sub_chunk_cap).
    .{ .kind = .kernel_raw_fn, .n_bindings = 4, .push_constant_size = 8, .pin_subgroup_32 = true },
    .{ .kind = .gather_off16_fn, .n_bindings = 4, .push_constant_size = 4, .pin_subgroup_32 = true },
    .{ .kind = .scan_parse_fn, .n_bindings = 10, .push_constant_size = 8, .pin_subgroup_32 = true },
    // walk_frame / prefix_sum_chunks / compact_huff / compact_raw /
    // merge_huff are single-thread guards (local_size_x=1) and do not
    // call any subgroup cooperative op. Do NOT pin — pipeline creation
    // would fail on workgroup-X-not-multiple-of-subgroupSize.
    .{ .kind = .walk_frame_fn, .n_bindings = 8, .push_constant_size = 8 },
    // srcVK/decode/prefix_sum_chunks_kernel.comp:16-34 declares 3 SSBO
    // bindings (ChunksBuf, FirstSubIdxBuf, TotalSubchunksBuf) plus a
    // 2× u32 push-constant block (n_chunks, sub_chunk_cap).
    .{ .kind = .prefix_sum_chunks_fn, .n_bindings = 3, .push_constant_size = 8 },
    // A-017: fused 4-way compact (was 4 separate dispatches × 4 bindings each).
    // 10 bindings = 4× staged + total + 4× dst + counts. Dispatched with grid_x=4.
    .{ .kind = .compact_huff_descs_fn, .n_bindings = 10, .push_constant_size = 0 },
    .{ .kind = .compact_raw_descs_fn, .n_bindings = 5, .push_constant_size = 0 },
    .{ .kind = .merge_huff_descs_fn, .n_bindings = 10, .push_constant_size = 8 },
    .{ .kind = .huff_build_fn, .n_bindings = 4, .push_constant_size = 0, .pin_subgroup_32 = true },
    // huff_decode binding 5 is a u32 alias of binding 3 (OutputBuf) — see kernel comment.
    // A-024 (2026-06-10 revision): binding 6 (d_compact_counts) lets the
    // kernel pick which region (lit | tok | off16) each block_id falls
    // in; the kernel is dispatched THREE times, once per region, with
    // the output bindings (3 + 5) bound AT the region's byte offset and
    // a single u32 push constant selecting which region this dispatch
    // serves. In-shader write offsets stay region-relative (≤ 2.14 GB),
    // which keeps the 1 GB L3+ ~6 GB scratch under maxStorageBufferRange
    // per binding AND under u32 arithmetic. CUDA keeps one dispatch with
    // u64 region params; see PortAdaptations A-024.
    .{ .kind = .huff_decode_fn, .n_bindings = 7, .push_constant_size = 4, .pin_subgroup_32 = true },
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

/// A-008 (BDA): query the device address of a registered VkDeviceBuffer
/// handle. Returns 0 on lookup miss or when the BDA entry point was not
/// resolved (which only happens if bufferDeviceAddress somehow failed to
/// enable at vkCreateDevice). Hot-callable; the underlying
/// vkGetBufferDeviceAddress is cheap (driver typically returns a cached
/// u64) but the encode path queries this once per encode at most so we
/// do not bother memoising on the AllocEntry itself.
///
/// Used by srcVK/encode/encode_lz.zig to pass the global hash table as a
/// raw `uint64_t` push constant to lz_encode_kernel.comp, bypassing the
/// per-binding `maxStorageBufferRange = 4 GiB - 1` cap that previously
/// forced the L3 hash_bits clamp on inputs >= 128 MiB (see A-008).
pub fn getBufferDeviceAddress(handle: VkDeviceBuffer) u64 {
    if (handle == 0) return 0;
    const getter = vkGetBufferDeviceAddress_fn orelse return 0;
    const entry = lookupAlloc(handle) orelse return 0;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const info = VkBufferDeviceAddressInfo{ .buffer = entry.buffer };
    return getter(dev, &info);
}

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

// Per-kernel profiler: given a VkPipeline handle (cast to usize), return
// the matching KERNEL_DECLS index, or -1 for unknown / externally-
// registered (encode) pipelines. Used by procLaunchKernel to tag the
// timestamp pair with a stable kernel identifier.
fn kindIndexForPipeline(handle: usize) i32 {
    if (handle == 0) return -1;
    inline for (KERNEL_DECLS, 0..) |_, i| {
        if (g_kernel_metas[i].pipeline != 0 and @as(usize, @intCast(g_kernel_metas[i].pipeline)) == handle) {
            return @intCast(i);
        }
    }
    return -1;
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
    // Fast path: post-init, every dispatch enters here. Read .ready
    // first so the steady-state hot path never touches the SRWLOCK.
    if (vulkan_api.init_state == .ready) return true;
    // VK adaptation: serialize the one-shot init across the ptest_vk
    // 16-worker test runner. Without this lock, two threads both see
    // .uninit (or one sees .in_progress and short-circuits to false),
    // surfacing as error.BackendNotAvailable in the dispatcher.
    AcquireSRWLockExclusive(&g_init_lock);
    defer ReleaseSRWLockExclusive(&g_init_lock);
    switch (vulkan_api.init_state) {
        .ready => return true,
        .failed => return false,
        // .in_progress can only be observed under the lock if a prior
        // init aborted mid-body without resetting the state; treat it
        // as a hard failure (matches pre-iter5 semantics).
        .in_progress => return false,
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
    // Iter 7: instance-level probes for VK_EXT_external_memory_host.
    // vkEnumerateDeviceExtensionProperties is core; vkGetPhysicalDevice
    // Properties2 is core in 1.1 (the instance was created with 1.3 above).
    // vkGetPhysicalDeviceMemoryProperties is core. Resolve them all NOW
    // so we can decide whether to enable the extension at vkCreateDevice.
    vkEnumerateDeviceExtensionProperties_fn = @ptrCast(gipa(inst, "vkEnumerateDeviceExtensionProperties"));
    vkGetPhysicalDeviceProperties2_fn = @ptrCast(gipa(inst, "vkGetPhysicalDeviceProperties2"));
    vkGetPhysicalDeviceMemoryProperties_fn = @ptrCast(gipa(inst, "vkGetPhysicalDeviceMemoryProperties"));
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

    // VK adaptation: subgroup size pinned to 32 via REQUIRE_FULL_SUBGROUPS_BIT
    // + VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Kernels in
    // gpu_warp.glsl hardcode WARP_SIZE=32 (the warpShuffle/Ballot/Any
    // cooperative ops are correct only at exactly 32 lanes). Without this
    // pin, Intel iGPU (supports [8,32]) would silently miscompile while
    // NVIDIA (only supports [32,32]) works by accident. The guard below
    // makes the gpu_warp.glsl contract real — devices that cannot satisfy
    // subgroupSize==32 are rejected at init so callers fall back to CPU.
    if (vkGetPhysicalDeviceProperties2_fn) |gpdp2| {
        var sg_props: VkPhysicalDeviceSubgroupSizeControlProperties = .{};
        var props2: VkPhysicalDeviceProperties2 = .{};
        props2.pNext = @ptrCast(&sg_props);
        gpdp2(phys, &props2);
        if (sg_props.maxSubgroupSize < 32 or sg_props.minSubgroupSize > 32) {
            // Print one clear error message naming the device + its subgroup
            // constraints + the contract requirement, then fail init so the
            // dispatcher surfaces error.BackendNotAvailable to the caller.
            var name_z: [257]u8 = @splat(0);
            const nlen = g_bound_device_name_len;
            if (nlen > 0 and nlen <= 256) {
                @memcpy(name_z[0..nlen], g_bound_device_name_buf[0..nlen]);
            }
            std.debug.print(
                "[VK_INIT] rejected device '{s}': subgroupSize range [{d}, {d}] cannot satisfy WARP_SIZE=32 contract (gpu_warp.glsl). Falling back to CPU.\n",
                .{ name_z[0..nlen], sg_props.minSubgroupSize, sg_props.maxSubgroupSize },
            );
            return false;
        }
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

    // Iter 11: VK adaptation — find a dedicated VK_QUEUE_TRANSFER_BIT
    // queue family (has TRANSFER but NOT GRAPHICS or COMPUTE). On NVIDIA
    // discrete GPUs this is the on-chip DMA / copy engine — recording
    // H2D vkCmdCopyBuffer into a cmdbuf submitted on this queue avoids
    // the subchannel-switch wait-for-idle NVIDIA forces when a single
    // queue mixes copy + compute. Mirrors CUDA's cuMemcpyHtoDAsync
    // auto-routing.
    //
    // Fallback when no dedicated family exists: reuse the compute family
    // (Intel iGPU, AMD APUs only expose one queue family). The cmdbuf
    // split still compiles + runs; the binary-semaphore wait degenerates
    // to a no-op serializer.
    var transfer_qf: u32 = queue_family;
    var dedicated_transfer = false;
    for (qf_buf[0..qfc], 0..) |qf, idx| {
        const has_xfer = (qf.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        const has_gfx = (qf.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const has_compute = (qf.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
        if (has_xfer and !has_gfx and !has_compute and qf.queueCount > 0) {
            transfer_qf = @intCast(idx);
            dedicated_transfer = true;
            break;
        }
    }
    g_transfer_queue_family = transfer_qf;
    g_has_dedicated_transfer = dedicated_transfer;

    const priorities = [_]f32{1.0};
    const xfer_priorities = [_]f32{1.0};
    var qcis: [2]VkDeviceQueueCreateInfo = undefined;
    qcis[0] = .{
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priorities,
    };
    var qci_count: u32 = 1;
    if (dedicated_transfer) {
        qcis[1] = .{
            .queueFamilyIndex = transfer_qf,
            .queueCount = 1,
            .pQueuePriorities = &xfer_priorities,
        };
        qci_count = 2;
    }
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
    // Iter 7: enable VK_EXT_external_memory_host when the physical device
    // advertises it. The decoder's procD2HAsync uses the extension to
    // import the caller's pageable host pointer directly as the
    // vkCmdCopyBuffer destination — eliminating the iter-4 dst→staging
    // copy + post-fence host @memcpy. Mirrors src_vulkan/device.zig:380
    // and src_vulkan/l1_codec.zig:670 (importHostPointerBuffer).
    const ext_mem_host_available = hasExternalMemoryHostExt(phys);
    var ext_names_storage: [1][*:0]const u8 = .{"VK_EXT_external_memory_host"};
    const ext_count: u32 = if (ext_mem_host_available) 1 else 0;
    const dci = VkDeviceCreateInfo{
        .pNext = @ptrCast(&v12_feats),
        .queueCreateInfoCount = qci_count,
        .pQueueCreateInfos = @ptrCast(&qcis),
        .enabledExtensionCount = ext_count,
        .ppEnabledExtensionNames = if (ext_count > 0) @ptrCast(&ext_names_storage) else null,
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
    // Iter 11: VK adaptation — semaphore FFI for the 2-submit chain.
    vkCreateSemaphore_fn = @ptrCast(gdpa(dev, "vkCreateSemaphore"));
    vkDestroySemaphore_fn = @ptrCast(gdpa(dev, "vkDestroySemaphore"));
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
    // VK adaptation (PHASE 2 — persistent VkPipelineCache): resolve the
    // create/get/destroy entry points so loadOrCreatePipelineCache can
    // bring the cache up before any vkCreateComputePipelines call below.
    vkCreatePipelineCache_fn = @ptrCast(gdpa(dev, "vkCreatePipelineCache"));
    vkGetPipelineCacheData_fn = @ptrCast(gdpa(dev, "vkGetPipelineCacheData"));
    vkDestroyPipelineCache_fn = @ptrCast(gdpa(dev, "vkDestroyPipelineCache"));
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
    // Iter 7: device-level entry points for the VK_EXT_external_memory_host
    // import path used by procD2HAsync. vkCreateBuffer / vkAllocateMemory /
    // vkBindBufferMemory / vkGetBufferMemoryRequirements / vkFreeMemory /
    // vkDestroyBuffer are all core (Vulkan 1.0). vkGetMemoryHostPointer
    // PropertiesEXT is part of the extension itself and is only resolved
    // when the extension was enabled at vkCreateDevice above.
    vkCreateBuffer_fn = @ptrCast(gdpa(dev, "vkCreateBuffer"));
    vkDestroyBuffer_fn = @ptrCast(gdpa(dev, "vkDestroyBuffer"));
    vkAllocateMemory_fn = @ptrCast(gdpa(dev, "vkAllocateMemory"));
    vkFreeMemory_fn = @ptrCast(gdpa(dev, "vkFreeMemory"));
    vkBindBufferMemory_fn = @ptrCast(gdpa(dev, "vkBindBufferMemory"));
    vkGetBufferMemoryRequirements_fn = @ptrCast(gdpa(dev, "vkGetBufferMemoryRequirements"));
    // A-008 (BDA): resolve vkGetBufferDeviceAddress. bufferDeviceAddress is
    // already enabled at vkCreateDevice (Vulkan 1.2 features struct above),
    // so the entry point is guaranteed present. The encode LZ kernel uses
    // BDA to address the global hash table (binding-3 cap escape; see
    // A-008 in srcVK/PortAdaptations.md).
    vkGetBufferDeviceAddress_fn = @ptrCast(gdpa(dev, "vkGetBufferDeviceAddress"));
    if (vkGetBufferDeviceAddress_fn == null) {
        // Fall back to KHR alias used by 1.1 + extension setups. We don't
        // enable VK_KHR_buffer_device_address explicitly because we use the
        // core 1.2 feature, but some loaders only export the KHR symbol.
        vkGetBufferDeviceAddress_fn = @ptrCast(gdpa(dev, "vkGetBufferDeviceAddressKHR"));
    }
    if (ext_mem_host_available) {
        vkGetMemoryHostPointerPropertiesEXT_fn = @ptrCast(gdpa(dev, "vkGetMemoryHostPointerPropertiesEXT"));
    }
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

    // Iter 11: VK adaptation — resolve the dedicated transfer queue. When
    // no dedicated family exists we fall back to queue 0 of the compute
    // family (same VkQueue handle as `queue` above); the split-cmdbuf
    // record path still works but the semaphore wait is a no-op.
    var xfer_queue: VkQueue = null;
    vkGetDeviceQueue_fn.?(dev, transfer_qf, 0, &xfer_queue);
    if (xfer_queue == null) return false;
    g_transfer_queue = @intFromPtr(xfer_queue);

    // Iter 11: log queue-family flags so we can verify (under
    // SLZ_VK_PROFILE_PHASES=1) we actually grabbed the dedicated DMA
    // engine on NVIDIA. Pure diagnostic — no behavioral impact.
    if (std.c.getenv("SLZ_VK_PROFILE_PHASES") != null) {
        var qf_flags_log: u32 = 0;
        if (transfer_qf < qfc) qf_flags_log = qf_buf[transfer_qf].queueFlags;
        std.debug.print(
            "[VK_INIT] transfer queue: qf={d} flags=0x{x} dedicated={s}\n",
            .{ transfer_qf, qf_flags_log, if (dedicated_transfer) "true" else "false" },
        );
    }

    // Iter 7: cache physical-device memory properties + the imported-host
    // alignment for the procD2HAsync import fast path. The extension is
    // only "supported" when every fn-ptr we need also resolved (defensive
    // — a buggy driver might advertise the extension but not actually
    // export vkGetMemoryHostPointerPropertiesEXT via gdpa).
    if (vkGetPhysicalDeviceMemoryProperties_fn) |gpmp| {
        gpmp(phys, &g_phys_mem_props);
    }
    g_ext_memory_host_supported = ext_mem_host_available and
        vkGetMemoryHostPointerPropertiesEXT_fn != null and
        vkGetPhysicalDeviceProperties2_fn != null and
        vkCreateBuffer_fn != null and
        vkDestroyBuffer_fn != null and
        vkAllocateMemory_fn != null and
        vkFreeMemory_fn != null and
        vkBindBufferMemory_fn != null and
        vkGetBufferMemoryRequirements_fn != null;
    if (g_ext_memory_host_supported) {
        var ext_props: VkPhysicalDeviceExternalMemoryHostPropertiesEXT = .{};
        var props2: VkPhysicalDeviceProperties2 = .{};
        props2.pNext = @ptrCast(&ext_props);
        vkGetPhysicalDeviceProperties2_fn.?(phys, &props2);
        g_imported_host_alignment = ext_props.minImportedHostPointerAlignment;
        if (g_imported_host_alignment == 0) {
            // Driver returned 0 — defensive disable.
            g_ext_memory_host_supported = false;
        }
    }

    // VK adaptation (OOM noise fix, 2026-06-07): NO preemptive disable
    // of g_import_disabled_h2d / _d2h here — the runtime sticky disable
    // in tryImportHostBuffer suffices. An earlier draft preemptively
    // flipped g_import_disabled_h2d=true for NVIDIA and regressed -db
    // bench by ~22 % (decompress best 15.6 ms → 19.1 ms). The driver's
    // H2D-SRC import is fragile but not uniformly broken: it succeeds
    // when the imported region comes from procMallocHost-pinned host
    // memory (which bench mode uses for compressed input + decoded
    // output) and fails with VK_ERROR_OUT_OF_DEVICE_MEMORY when the
    // region comes from a plain page-allocator slice (which the
    // one-shot CLI uses for file-IO buffers). Leaving the disable to
    // tryImportHostBuffer's per-OOM trip yields 0 warnings in bench
    // mode and 1 warning per direction per process for -c/-d one-shot
    // mode (vs. one warning per cache miss before).

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

    // Iter 11: VK adaptation — per-context command pool bound to the
    // transfer queue family. Each StreamEntry will allocate its own
    // transfer_cmdbuf out of this pool. When dedicated_transfer is false
    // this pool sits on the compute family (same QF index); still
    // creates fine, the cmdbufs are still safely submittable.
    const xfer_pool_ci = VkCommandPoolCreateInfo{
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = transfer_qf,
    };
    if (vkCreateCommandPool_fn.?(dev, &xfer_pool_ci, null, &g_transfer_cmd_pool) != VK_SUCCESS_RC) return false;

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

    // Per-context descriptor pool.
    //
    // Iter 14 (A): the pool now backs persistent (stream × pipeline)
    // descriptor sets (see procLaunchKernel) instead of per-launch
    // allocations, so the steady-state slot count is bounded by
    // KERNEL_DECLS.len * g_streams.cap (=11 * 256 = 2816) plus a margin
    // for encode-registered extras. Bumped maxSets to 8192 to cover
    // ptest_vk's per-test-context churn without falling back to the
    // legacy reset+retry path (which would invalidate persistent sets).
    const max_bindings_per_pool: u32 = 32768;
    const max_sets: u32 = 8192;
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

    // Iter 4c: separate pool for per-launch transient sets used by the
    // sub-region binding path (binding_offsets != null in procLaunchKernel).
    // Sized for the worst case decode: ~6 transient sets per
    // gpuScanChunks call (scan_parse + 4 compact_huff + compact_raw) +
    // 3 (merge + gather + huff_build + huff_decode = 4) = ~10 per
    // decoded block. ptest_vk runs at most ~64 concurrent decodes per
    // device with up to ~128 blocks per file; cap at 4096 sets and reset
    // between fullGpuLaunchImpl calls so the pool never fills mid-decode.
    const tr_pool_size = VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 4096 * 12, // worst-case bindings * max sets
    };
    const tr_dp_ci = VkDescriptorPoolCreateInfo{
        .maxSets = 4096,
        .poolSizeCount = 1,
        .pPoolSizes = @ptrCast(&tr_pool_size),
    };
    if (vkCreateDescriptorPool_fn.?(dev, &tr_dp_ci, null, &g_transient_descriptor_pool) != VK_SUCCESS_RC) return false;

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
        // Iter 4c: capture minStorageBufferOffsetAlignment for the
        // per-binding-offset launch-kernel path. Required by Vulkan spec
        // VUID-VkDescriptorBufferInfo-offset-00925.
        g_min_storage_buffer_offset_alignment = props.limits.minStorageBufferOffsetAlignment;
        g_max_storage_buffer_range = props.limits.maxStorageBufferRange;
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
            // Only seed slots [0 .. g_profile_slot_base). The top
            // g_profile_slot_count slots are reserved for the per-kernel
            // profiler (SLZ_VK_PROFILE_DECODE) so procEvent never collides
            // with timestamps procLaunchKernel records.
            var i: u32 = 0;
            while (i < g_profile_slot_base) : (i += 1) {
                g_query_index_free.append(gpa, i) catch break;
            }
        }
    }

    // Enable per-kernel GPU profiling if SLZ_VK_PROFILE_DECODE is set.
    if (std.c.getenv("SLZ_VK_PROFILE_DECODE") != null) {
        g_profile_enabled = (g_timestamp_query_pool != VK_NULL_HANDLE);
        g_profile_record_count = 0;
        g_profile_overflow_count = 0;
    }

    // VK adaptation: pre-reserve the alloc + stream + host-alloc registries
    // to a generous fixed cap so they never reallocate at runtime. The
    // streamEntryFor / lookupAlloc readers grab an `&items[idx]` pointer
    // that would be invalidated by an ArrayList realloc; with the
    // backing storage stable, concurrent reads from per-test
    // EncodeContext / DecodeContext (ptest_vk's 16-worker runner) cannot
    // observe a torn pointer even if a procStreamCreate / procMallocDevice
    // append happens on a sibling thread. The registry SRWLOCK above
    // protects against torn slot reads + concurrent append; this reserve
    // protects against pointer invalidation across the boundary where
    // the caller already snapshotted the pointer.
    {
        const gpa = std.heap.page_allocator;
        g_allocs.ensureTotalCapacity(gpa, 4096) catch {};
        g_streams.ensureTotalCapacity(gpa, 256) catch {};
        g_host_allocs.ensureTotalCapacity(gpa, 256) catch {};
    }

    // Wire up the procs.* surface. All slots use the module-private
    // helpers defined below.
    vulkan_api.procs.malloc_device = procMallocDevice;
    vulkan_api.procs.free_device = procFreeDevice;
    vulkan_api.procs.h2d = procH2D;
    vulkan_api.procs.d2h = procD2H;
    vulkan_api.procs.d2h_offset = procD2HOffset;
    // VK adaptation (encode D2H gather): batched gather slot routed to
    // the dedicated transfer queue. See procD2HOffsetGather impl below.
    vulkan_api.procs.d2h_offset_gather = procD2HOffsetGather;
    vulkan_api.procs.d2d = procD2D;
    vulkan_api.procs.memset_d8 = procMemsetD8;
    vulkan_api.procs.memset_d8_async = procMemsetD8Async;
    vulkan_api.procs.malloc_host = procMallocHost;
    vulkan_api.procs.free_host = procFreeHost;
    vulkan_api.procs.stream_sync = procStreamSync;
    vulkan_api.procs.stream_flush_transfer = procStreamFlushTransfer;
    vulkan_api.procs.ctx_sync = procCtxSync;
    vulkan_api.procs.stream_create = procStreamCreate;
    vulkan_api.procs.stream_destroy = procStreamDestroy;
    vulkan_api.procs.launch_kernel = procLaunchKernel;
    vulkan_api.procs.compute_to_compute_barrier = procComputeToComputeBarrier;
    vulkan_api.procs.h2d_async = procH2DAsync;
    vulkan_api.procs.d2h_async = procD2HAsync;
    vulkan_api.procs.event_create = procEventCreate;
    vulkan_api.procs.event_record = procEventRecord;
    vulkan_api.procs.event_synchronize = procEventSynchronize;
    vulkan_api.procs.event_elapsed_time = procEventElapsedTime;
    vulkan_api.procs.event_destroy = procEventDestroy;

    // VK adaptation (PHASE 2 — persistent VkPipelineCache): bring the
    // cache up before any vkCreateComputePipelines call below. Best-effort
    // file load + driver-validated VkPipelineCacheCreateInfo; on cache miss
    // the create still succeeds and the first buildKernel call pays the
    // full SPIR-V → ISA compile (which then populates the in-memory cache
    // for the rest of init()). The atexit-registered savePipelineCache
    // writes the populated bytes back to disk so the *next* process
    // launch hits the warm path. This is the targeted fix for the 100-300 ms
    // process-launch overhead identified by NCU: the LZ kernel itself is at
    // CUDA parity (77 ms vs 71 ms) but every CLI invocation currently
    // recompiles all 17 decode + encode pipelines from scratch.
    loadOrCreatePipelineCache(dev);

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

// VK adaptation (PHASE 2 — persistent VkPipelineCache): public accessor so
// the encode-side module_loader can pass the same VkPipelineCache to its
// vkCreateComputePipelines call. Returns VK_NULL_HANDLE if the decode-side
// init hasn't yet reached the cache-create step (encode chains into decode
// init() before its own build, so by the time the encode buildPipeline
// fires the cache is always populated). Safe to pass NULL to
// vkCreateComputePipelines — the spec permits it, the create just runs
// uncached.
pub fn pipelineCacheHandle() VkPipelineCache {
    return g_pipeline_cache;
}

// VK adaptation (PHASE 2 — persistent VkPipelineCache): resolve the on-disk
// cache file path. Prefers %LOCALAPPDATA%/streamlz_vk/pipeline_cache.bin on
// Windows; falls back to ./.streamlz_vk_pipeline_cache.bin when
// LOCALAPPDATA is unset (CI, service accounts). The returned slice points
// into g_pipeline_cache_path_buf (no allocation needed at init or shutdown
// — atexit runs after the page allocator has potentially been torn down on
// some platforms, so keep the path in a static buffer).
fn pipelineCachePath() []const u8 {
    if (g_pipeline_cache_path_len > 0) return g_pipeline_cache_path_buf[0..g_pipeline_cache_path_len];
    const buf = g_pipeline_cache_path_buf[0..];
    var written: usize = 0;
    if (std.c.getenv("LOCALAPPDATA")) |raw| {
        const base = std.mem.span(raw);
        const suffix = "\\streamlz_vk\\pipeline_cache.bin";
        if (base.len + suffix.len < buf.len) {
            @memcpy(buf[0..base.len], base);
            @memcpy(buf[base.len .. base.len + suffix.len], suffix);
            written = base.len + suffix.len;
        }
    }
    if (written == 0) {
        const fallback = ".streamlz_vk_pipeline_cache.bin";
        @memcpy(buf[0..fallback.len], fallback);
        written = fallback.len;
    }
    g_pipeline_cache_path_len = written;
    return buf[0..written];
}

// VK adaptation (PHASE 2 — persistent VkPipelineCache): direct Win32 file
// IO surface. Zig 0.16 moved std.fs.cwd() under the std.Io.Dir context-
// passing API which doesn't compose with an atexit (no-arg) callback; the
// rest of this loader already uses raw kernel32 extern fns for
// LoadLibraryA / GetProcAddress / SRWLOCK so this stays consistent.
// Namespaced under `pcwin` to avoid colliding with the CloseHandle /
// CreateFileMappingW extern decls in srcVK/mmap.zig (each extern "kernel32"
// fn would otherwise produce a duplicate-symbol at link time).
const pcwin = struct {
    const HANDLE = ?*anyopaque;
    const INVALID_HANDLE_VALUE: HANDLE = @ptrFromInt(@as(usize, @bitCast(@as(isize, -1))));
    const GENERIC_READ: u32 = 0x80000000;
    const GENERIC_WRITE: u32 = 0x40000000;
    const FILE_SHARE_READ: u32 = 0x00000001;
    const OPEN_EXISTING: u32 = 3;
    const CREATE_ALWAYS: u32 = 2;
    const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
    const BOOL = c_int;
    extern "kernel32" fn CreateFileA(
        lpFileName: [*:0]const u8,
        dwDesiredAccess: u32,
        dwShareMode: u32,
        lpSecurityAttributes: ?*const anyopaque,
        dwCreationDisposition: u32,
        dwFlagsAndAttributes: u32,
        hTemplateFile: HANDLE,
    ) callconv(.winapi) HANDLE;
    extern "kernel32" fn CloseHandle(h: HANDLE) callconv(.winapi) BOOL;
    extern "kernel32" fn ReadFile(
        h: HANDLE,
        lpBuffer: *anyopaque,
        nNumberOfBytesToRead: u32,
        lpNumberOfBytesRead: *u32,
        lpOverlapped: ?*anyopaque,
    ) callconv(.winapi) BOOL;
    extern "kernel32" fn WriteFile(
        h: HANDLE,
        lpBuffer: *const anyopaque,
        nNumberOfBytesToWrite: u32,
        lpNumberOfBytesWritten: *u32,
        lpOverlapped: ?*anyopaque,
    ) callconv(.winapi) BOOL;
    extern "kernel32" fn GetFileSizeEx(h: HANDLE, lpFileSize: *i64) callconv(.winapi) BOOL;
    extern "kernel32" fn CreateDirectoryA(
        lpPathName: [*:0]const u8,
        lpSecurityAttributes: ?*const anyopaque,
    ) callconv(.winapi) BOOL;
};

// VK adaptation (PHASE 2 — persistent VkPipelineCache): write `path` into
// a [*:0]const u8 NUL-terminated buffer suitable for CreateFileA. Returns
// null when the path doesn't fit in `out`. Path is already short
// (<= 512 B per g_pipeline_cache_path_buf) so this never trips in
// practice.
fn cstrFromPath(path: []const u8, out: []u8) ?[*:0]const u8 {
    if (path.len + 1 > out.len) return null;
    @memcpy(out[0..path.len], path);
    out[path.len] = 0;
    return @ptrCast(out.ptr);
}

// VK adaptation (PHASE 2 — persistent VkPipelineCache): best-effort
// CreateDirectoryA of the parent directory of `path`. Ignores
// ERROR_ALREADY_EXISTS implicitly (the subsequent CreateFileA succeeds).
fn ensureParentDir(path: []const u8) void {
    var cut: usize = path.len;
    while (cut > 0) : (cut -= 1) {
        const c = path[cut - 1];
        if (c == '\\' or c == '/') break;
    }
    if (cut <= 1) return;
    const parent = path[0 .. cut - 1];
    var nul_buf: [520]u8 = undefined;
    const cstr = cstrFromPath(parent, nul_buf[0..]) orelse return;
    _ = pcwin.CreateDirectoryA(cstr, null);
}

// VK adaptation (PHASE 2 — persistent VkPipelineCache): load any existing
// cache blob from disk and feed it to vkCreatePipelineCache. The driver
// validates the header (vendorID + deviceID + driverUUID +
// pipelineCacheUUID + dataSize) and either keeps the bytes or starts
// empty — vkCreatePipelineCache returns VK_SUCCESS either way. We never
// see "header valid but data corrupt" because the spec requires the
// driver to compute a private checksum over the payload and discard
// mismatches. Per the NCU finding + 100-300 ms compile estimate, this
// load step is the difference between "every CLI invocation pays full
// SPIR-V → ISA compile" and "first invocation pays, rest reuse cached
// ISA". Idempotent: safe to call once per process from init().
fn loadOrCreatePipelineCache(dev: VkDevice) void {
    if (g_pipeline_cache != VK_NULL_HANDLE) return;
    const create_fn = vkCreatePipelineCache_fn orelse return;
    const path = pipelineCachePath();
    var data_slice: ?[]u8 = null;
    defer if (data_slice) |s| std.heap.page_allocator.free(s);
    // Best-effort read; missing file / IO error → empty cache (driver will
    // simply build everything from scratch on the first vkCreateComputePipelines).
    var nul_buf: [520]u8 = undefined;
    if (cstrFromPath(path, nul_buf[0..])) |cstr| {
        const h = pcwin.CreateFileA(
            cstr,
            pcwin.GENERIC_READ,
            pcwin.FILE_SHARE_READ,
            null,
            pcwin.OPEN_EXISTING,
            pcwin.FILE_ATTRIBUTE_NORMAL,
            null,
        );
        if (h != pcwin.INVALID_HANDLE_VALUE) {
            defer _ = pcwin.CloseHandle(h);
            var fsize: i64 = 0;
            if (pcwin.GetFileSizeEx(h, &fsize) != 0 and fsize > 0 and fsize < (32 * 1024 * 1024)) {
                const sz: usize = @intCast(fsize);
                if (std.heap.page_allocator.alloc(u8, sz)) |b| {
                    var got: u32 = 0;
                    const ok = pcwin.ReadFile(h, @ptrCast(b.ptr), @intCast(sz), &got, null);
                    if (ok != 0 and got == sz) {
                        data_slice = b;
                    } else {
                        std.heap.page_allocator.free(b);
                    }
                } else |_| {}
            }
        }
    }
    const ci = if (data_slice) |s| VkPipelineCacheCreateInfo{
        .initialDataSize = s.len,
        .pInitialData = s.ptr,
    } else VkPipelineCacheCreateInfo{};
    _ = create_fn(dev, &ci, null, &g_pipeline_cache);
    // Register one-shot save on graceful process exit so CLI invocations
    // (no explicit deinit) still persist freshly-compiled ISA for the next
    // run. atexit may not fire on hard aborts; that's fine — next launch
    // just rebuilds and rewrites the file. Zig 0.16 removed std.c.atexit,
    // so we declare the CRT entry directly (the project already links
    // against -lc via build.zig — both msvcrt and ucrtbase export it).
    if (!g_pipeline_cache_atexit_registered and g_pipeline_cache != VK_NULL_HANDLE) {
        _ = crt_atexit(pipelineCacheAtexit);
        g_pipeline_cache_atexit_registered = true;
    }
}

extern "c" fn atexit(cb: *const fn () callconv(.c) void) callconv(.c) c_int;
// Local alias so the call site documents the CRT bridge clearly.
const crt_atexit = atexit;

// VK adaptation (PHASE 2 — persistent VkPipelineCache): atexit shim.
// Bridges std.c.atexit's (no-arg, void) signature to savePipelineCache.
// Pulls the VkDevice out of vulkan_api.ctx (the only place it lives) so
// shutdown doesn't need a captured closure.
fn pipelineCacheAtexit() callconv(.c) void {
    if (vulkan_api.ctx == 0) return;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    savePipelineCache(dev);
}

// VK adaptation (PHASE 2 — persistent VkPipelineCache): query the driver
// for the current cache contents and write them to disk. Two-call pattern
// per Vulkan spec (size probe → buffer alloc → fetch). Best-effort: any
// failure (size==0, alloc failure, write failure) silently aborts — the
// in-memory cache still served this run, only the next process loses the
// benefit. We also destroy the VkPipelineCache here so a subsequent
// re-init doesn't leak.
fn savePipelineCache(dev: VkDevice) void {
    const cache = g_pipeline_cache;
    if (cache == VK_NULL_HANDLE) return;
    const get_fn = vkGetPipelineCacheData_fn;
    const destroy_fn = vkDestroyPipelineCache_fn;
    if (get_fn) |gf| {
        var size: usize = 0;
        if (gf(dev, cache, &size, null) == VK_SUCCESS_RC and size > 0) {
            if (std.heap.page_allocator.alloc(u8, size)) |b| {
                defer std.heap.page_allocator.free(b);
                var actual = size;
                if (gf(dev, cache, &actual, @ptrCast(b.ptr)) == VK_SUCCESS_RC and actual > 0) {
                    const path = pipelineCachePath();
                    ensureParentDir(path);
                    var nul_buf: [520]u8 = undefined;
                    if (cstrFromPath(path, nul_buf[0..])) |cstr| {
                        const h = pcwin.CreateFileA(
                            cstr,
                            pcwin.GENERIC_WRITE,
                            0,
                            null,
                            pcwin.CREATE_ALWAYS,
                            pcwin.FILE_ATTRIBUTE_NORMAL,
                            null,
                        );
                        if (h != pcwin.INVALID_HANDLE_VALUE) {
                            defer _ = pcwin.CloseHandle(h);
                            var written: u32 = 0;
                            // Write in a single call. >4 GiB writes aren't a
                            // concern (the largest VkPipelineCache observed on
                            // NVIDIA + Intel is under a few MiB even with all
                            // 17 streamlz pipelines populated).
                            _ = pcwin.WriteFile(h, @ptrCast(b.ptr), @intCast(actual), &written, null);
                        }
                    }
                }
            } else |_| {}
        }
    }
    if (destroy_fn) |df| df(dev, cache, null);
    g_pipeline_cache = VK_NULL_HANDLE;
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
    // VK adaptation: subgroup size pinned to 32 via REQUIRE_FULL_SUBGROUPS_BIT
    // + VkPipelineShaderStageRequiredSubgroupSizeCreateInfo. Kernels in
    // gpu_warp.glsl hardcode WARP_SIZE=32 (the warpShuffle/Ballot/Any
    // cooperative ops are correct only at exactly 32 lanes). Without this
    // pin, Intel iGPU (supports [8,32]) would silently miscompile while
    // NVIDIA (only supports [32,32]) works by accident. Only kernels with
    // local_size_x % 32 == 0 that use subgroup ops get the pin (flagged
    // via KernelDecl.pin_subgroup_32); single-thread guard kernels run
    // with local_size_x=1 and would fail pipeline creation if pinned.
    const required_size_info = VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{
        .requiredSubgroupSize = 32,
    };
    const stage = if (decl.pin_subgroup_32) VkPipelineShaderStageCreateInfo{
        .flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT,
        .pNext = @ptrCast(&required_size_info),
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = sm,
        .pName = "main",
    } else VkPipelineShaderStageCreateInfo{
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .module = sm,
        .pName = "main",
    };
    const ci = VkComputePipelineCreateInfo{
        .stage = stage,
        .layout = meta.pl_layout,
    };
    var pipeline: VkPipeline = 0;
    // VK adaptation (PHASE 2 — persistent VkPipelineCache): pass the
    // process-wide g_pipeline_cache so the driver can short-circuit
    // SPIR-V → ISA compile when the cached ISA blob for this SPV
    // module + entry point + pipeline layout already exists. On the
    // first launch the cache is empty and this is identical to the
    // prior VK_NULL_HANDLE call; on subsequent launches the cached
    // ISA halves to fully eliminates the ~10-30 ms per-pipeline
    // compile cost. Spec permits VK_NULL_HANDLE here so a failed
    // loadOrCreatePipelineCache leaves correctness intact.
    if (vkCreateComputePipelines_fn.?(dev, g_pipeline_cache, 1, @ptrCast(&ci), null, @ptrCast(&pipeline)) != VK_SUCCESS_RC) return false;
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

// VK adaptation: parallel test workers (per-test DecodeContext /
// EncodeContext) hit g_allocs through procs.malloc_device + free_device
// concurrently. The ArrayListUnmanaged backing slice can be reallocated
// inside append; without the lock a concurrent read sees a torn pointer
// and segfaults (ptest_vk surfaced this as exit code 5 with no test-fail
// trace). Hold g_alloc_registry_lock across every g_allocs.* mutation;
// also held for lookupAlloc to snapshot the value before VMA touches it.
fn registerAlloc(buffer: vma.VkBuffer, alc: vma.VmaAllocation, size: usize) VkDeviceBuffer {
    lockAllocRegistry();
    defer unlockAllocRegistry();
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
    lockAllocRegistry();
    defer unlockAllocRegistry();
    const idx: usize = @intCast(handle - 1);
    if (idx >= g_allocs.items.len) return null;
    return g_allocs.items[idx];
}

fn releaseAlloc(handle: VkDeviceBuffer) void {
    if (handle == 0) return;
    lockAllocRegistry();
    defer unlockAllocRegistry();
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

// Iter 11: VK adaptation — begin recording the per-stream transfer
// cmdbuf if not already recording. Mirrors streamBeginIfNeeded but
// targets the dedicated-transfer command pool's buffer. Called from
// procH2DAsync's per-stream path so H2D vkCmdCopyBuffer lands on the
// transfer queue instead of the compute queue.
fn streamBeginTransferIfNeeded(entry: *StreamEntry) VkResult {
    if (entry.transfer_recording) return VK_SUCCESS_RC;
    _ = vkResetCommandBuffer_fn.?(entry.transfer_cmdbuf, 0);
    const begin = VkCommandBufferBeginInfo{
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    const rc = vkBeginCommandBuffer_fn.?(entry.transfer_cmdbuf, &begin);
    if (rc == VK_SUCCESS_RC) entry.transfer_recording = true;
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
    if (!entry.recording and !entry.transfer_recording and !entry.transfer_already_submitted) {
        // Even when no cmdbuf was recorded, reset arena/pending state
        // for symmetry (defensive: caller may have skipped the body).
        entry.staging_used = 0;
        entry.pending_d2h.clearRetainingCapacity();
        freePendingImportedD2H(entry);
        return VK_SUCCESS_RC;
    }
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const compute_q: VkQueue = @ptrFromInt(vulkan_api.compute_queue);
    const xfer_q: VkQueue = @ptrFromInt(g_transfer_queue);

    // VK adaptation: H2D recorded into dedicated VK_QUEUE_TRANSFER_BIT queue
    // (NVIDIA dedicated DMA engine); compute submit waits on a binary
    // semaphore signalled by the transfer submit. Mirrors CUDA's
    // cuMemcpyHtoDAsync auto-routing to the copy engine.
    //
    // Two-submit dance:
    //   1) Transfer queue: signal entry.h2d_sem (no wait, no fence).
    //   2) Compute queue:  wait on entry.h2d_sem at COMPUTE_SHADER_BIT,
    //                      signal entry.fence.
    //   3) Host: vkWaitForFences(entry.fence) — the transfer submit
    //      finishes implicitly when compute fires, no need to wait on
    //      the transfer queue separately.
    //
    // When entry.transfer_recording is false (no H2Ds on this decode)
    // we skip the transfer submit + the wait, falling back to the
    // single-submit shape so we don't pay for an empty cmdbuf.

    // ── 1) End + submit the transfer cmdbuf if it has work ────────────
    // Iter 13: If procStreamFlushTransfer already End+Submitted the
    // transfer cmdbuf at the end of uploadInputAndPrefixSum, the
    // h2d_sem is already in-flight and we skip the end+submit here.
    // The compute submit below still waits on h2d_sem (has_xfer_work
    // remains true) so semantics are preserved.
    const has_xfer_work = entry.transfer_recording or entry.transfer_already_submitted;
    if (entry.transfer_recording) {
        const rc_end = vkEndCommandBuffer_fn.?(entry.transfer_cmdbuf);
        if (rc_end != VK_SUCCESS_RC) {
            entry.recording = false;
            entry.transfer_recording = false;
            entry.transfer_already_submitted = false;
            entry.staging_used = 0;
            entry.pending_d2h.clearRetainingCapacity();
            freePendingImportedD2H(entry);
            return rc_end;
        }
        const xfer_cb = entry.transfer_cmdbuf;
        const sem_handle = entry.h2d_sem;
        const xfer_submit = VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = @ptrCast(&xfer_cb),
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = @ptrCast(&sem_handle),
        };
        const rc_sub_xfer = vkQueueSubmit_fn.?(xfer_q, 1, @ptrCast(&xfer_submit), VK_NULL_HANDLE);
        if (rc_sub_xfer != VK_SUCCESS_RC) {
            entry.recording = false;
            entry.transfer_recording = false;
            entry.transfer_already_submitted = false;
            entry.staging_used = 0;
            entry.pending_d2h.clearRetainingCapacity();
            freePendingImportedD2H(entry);
            return rc_sub_xfer;
        }
    }

    // ── 2) End + submit the compute cmdbuf (wait on h2d_sem if any) ───
    // Defensive: if compute cmdbuf wasn't begun but transfer was, we
    // still need to drain the semaphore. Begin an empty compute cmdbuf
    // so the wait semantics line up. (In practice procH2DAsync is
    // always followed by procLaunchKernel so this branch is unreached.)
    if (!entry.recording) {
        const rc_begin = streamBeginIfNeeded(entry);
        if (rc_begin != VK_SUCCESS_RC) {
            // Cannot drain the signalled semaphore — we'd deadlock on
            // the next decode. Best effort: vkQueueWaitIdle on the
            // transfer queue so the semaphore consumer is implicit.
            if (vkQueueWaitIdle_fn) |wi| _ = wi(xfer_q);
            entry.transfer_recording = false;
            entry.transfer_already_submitted = false;
            entry.staging_used = 0;
            entry.pending_d2h.clearRetainingCapacity();
            freePendingImportedD2H(entry);
            return rc_begin;
        }
    }
    var rc = vkEndCommandBuffer_fn.?(entry.cmdbuf);
    if (rc != VK_SUCCESS_RC) {
        entry.recording = false;
        entry.transfer_recording = false;
        entry.transfer_already_submitted = false;
        entry.staging_used = 0;
        entry.pending_d2h.clearRetainingCapacity();
        freePendingImportedD2H(entry);
        return rc;
    }
    const cb = entry.cmdbuf;
    const sem_for_wait = entry.h2d_sem;
    const wait_stage: u32 = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    const compute_submit = VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cb),
        .waitSemaphoreCount = if (has_xfer_work) @as(u32, 1) else @as(u32, 0),
        .pWaitSemaphores = if (has_xfer_work) @as(?*const anyopaque, @ptrCast(&sem_for_wait)) else null,
        .pWaitDstStageMask = if (has_xfer_work) @as(?*const u32, &wait_stage) else null,
    };
    _ = vkResetFences_fn.?(dev, 1, @ptrCast(&entry.fence));
    rc = vkQueueSubmit_fn.?(compute_q, 1, @ptrCast(&compute_submit), entry.fence);
    if (rc != VK_SUCCESS_RC) {
        entry.recording = false;
        entry.transfer_recording = false;
        entry.transfer_already_submitted = false;
        entry.staging_used = 0;
        entry.pending_d2h.clearRetainingCapacity();
        freePendingImportedD2H(entry);
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
    // Iter 7: imported (VK_EXT_external_memory_host) D2H destinations
    // are safe to free now that the fence has signaled — the GPU is no
    // longer reading/writing through the VkBuffer or VkDeviceMemory.
    // No host @memcpy is needed; the GPU wrote the bytes straight into
    // the caller's pageable buffer.
    freePendingImportedD2H(entry);
    // VK adaptation (valfix B — VUID-vkCmdCopyBuffer-commandBuffer-recording /
    // VUID-vkEndCommandBuffer-commandBuffer-00059): reset both cmdbufs to
    // the INITIAL state now, before relinquishing control. After fence wait
    // the cmdbufs are in EXECUTABLE state and still hold strong references
    // to every buffer they recorded against. If the caller subsequently
    // destroys one of those buffers (e.g. CLI epilog calling
    // releaseImportsByHostRange before mmap.unmap, or VMA recycling a VMA
    // block via free_device), the executable cmdbuf transitions to INVALID
    // state per Vulkan spec, and the validation layer flags it. Resetting
    // here returns the cmdbufs to INITIAL state so they hold no references
    // — destroying a previously-recorded buffer is then safe. The next
    // streamBeginIfNeeded reset becomes a no-op idempotency check.
    if (entry.cmdbuf != null) _ = vkResetCommandBuffer_fn.?(entry.cmdbuf, 0);
    if (entry.transfer_cmdbuf != null) _ = vkResetCommandBuffer_fn.?(entry.transfer_cmdbuf, 0);

    // VK adaptation (valfix B): drain any staging buffers that were
    // deferred-destroyed by mid-recording ensureStreamStaging growths.
    // Safe now — the fence has signaled, the cmdbufs have been reset to
    // initial state, and the GPU is no longer referencing them.
    if (entry.deferred_staging_destroy.items.len > 0) {
        for (entry.deferred_staging_destroy.items) |h| {
            vma.destroyBuffer(allocator(), h.buffer, h.allocation);
        }
        entry.deferred_staging_destroy.clearRetainingCapacity();
    }
    entry.recording = false;
    entry.transfer_recording = false;
    entry.transfer_already_submitted = false;
    entry.staging_used = 0;
    entry.pending_d2h.clearRetainingCapacity();
    return rc;
}

// Iter 7: free every (VkBuffer, VkDeviceMemory) pair queued by
// procD2HAsync's import fast path on this stream. Safe to call when
// the list is empty.
//
// Iter 8 subfix 1: cached entries (cache_idx >= 0) are NOT destroyed
// — they're owned by g_import_cache and stay live for the next
// decode. We only flip their in_flight bit so subsequent lookups can
// reuse them. cache_idx == -1 entries (cache overflow path) still
// get the iter-7 destroy treatment.
fn freePendingImportedD2H(entry: *StreamEntry) void {
    if (entry.pending_imported_d2h.items.len == 0) return;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    for (entry.pending_imported_d2h.items) |p| {
        if (p.cache_idx >= 0) {
            clearImportInFlight(p.cache_idx);
        } else {
            vkDestroyBuffer_fn.?(dev, p.buffer, null);
            vkFreeMemory_fn.?(dev, p.memory, null);
        }
    }
    entry.pending_imported_d2h.clearRetainingCapacity();
}

// ── procs.* implementations ───────────────────────────────────────────

fn procMallocDevice(out: *VkDeviceBuffer, size: usize) callconv(.c) VkResult {
    var buf: vma.VkBuffer = 0;
    var alc: vma.VmaAllocation = null;
    // Iter 11: VK adaptation — when we have a dedicated transfer queue
    // family, set VK_SHARING_MODE_CONCURRENT on storage buffers so the
    // transfer queue can vkCmdCopyBuffer into them and the compute queue
    // can read/write them without an explicit queue-family ownership
    // transfer barrier. The spec lets us list any subset of queue
    // families the resource is touched by; missing entries silently make
    // the access undefined. On NVIDIA discrete (transfer != compute) the
    // perf cost of CONCURRENT vs EXCLUSIVE is negligible (driver still
    // single-owner under the hood for storage buffers); the savings from
    // skipping the ownership barrier in every per-decode submit
    // outweigh any micro-cost.
    var concurrent_qfs = [_]u32{ vulkan_api.compute_queue_family, g_transfer_queue_family };
    const sharing_mode: c_int = if (g_has_dedicated_transfer) vma.VK_SHARING_MODE_CONCURRENT else vma.VK_SHARING_MODE_EXCLUSIVE;
    const qf_count: u32 = if (g_has_dedicated_transfer) 2 else 0;
    const qf_ptr: ?[*]const u32 = if (g_has_dedicated_transfer) @ptrCast(&concurrent_qfs) else null;
    const bci = vma.VkBufferCreateInfo{
        .size = size,
        .usage = vma.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vma.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            vma.VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            vma.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        .sharingMode = sharing_mode,
        .queueFamilyIndexCount = qf_count,
        .pQueueFamilyIndices = qf_ptr,
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

// VK adaptation: device → host blocking copy starting at `src_offset`
// bytes into the device buffer. CUDA encodes the offset into the
// CUdeviceptr arithmetic (cuMemcpyDtoH(host, d_output + dst_off, size));
// on Vulkan we pass it through to VkBufferCopy.srcOffset because the
// device handle is a registry index, not a real device VA. Backs the H1
// per-chunk D2H loop in srcVK/encode/encode_lz.zig that mirrors the
// CUDA per-chunk download at src/encode/encode_lz.zig:144-150.
fn procD2HOffset(dst: *anyopaque, src: VkDeviceBuffer, size: usize, src_offset: usize) callconv(.c) VkResult {
    const entry = lookupAlloc(src) orelse return -1;
    if (!ensureStaging(size)) return -1;
    var rc = beginOneShotCB();
    if (rc != VK_SUCCESS_RC) return rc;
    const region = VkBufferCopy{ .srcOffset = src_offset, .dstOffset = 0, .size = size };
    vkCmdCopyBuffer_fn.?(g_command_buffer, entry.buffer, g_staging_buffer, 1, @ptrCast(&region));
    rc = submitAndWait();
    if (rc != VK_SUCCESS_RC) return rc;
    const mapped = g_staging_mapped orelse return -1;
    @memcpy(@as([*]u8, @ptrCast(dst))[0..size], @as([*]const u8, @ptrCast(mapped))[0..size]);
    return VK_SUCCESS_RC;
}

// VK adaptation (encode D2H gather, iter-7+11+15 transferable):
// Lazily allocate the one-shot transfer cmdbuf + fence on the first
// gather call. Cheaper than baking the alloc into init() — it costs
// nothing when the slot is never exercised (e.g. ptest_vk subtests
// that only hit decode).
fn ensureXferOneshotReady() bool {
    if (g_xfer_oneshot_ready) return true;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const cb_alloc = VkCommandBufferAllocateInfo{
        .commandPool = g_transfer_cmd_pool,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers_fn.?(dev, &cb_alloc, @ptrCast(&g_xfer_oneshot_cb)) != VK_SUCCESS_RC) return false;
    const fence_ci = VkFenceCreateInfo{};
    if (vkCreateFence_fn.?(dev, &fence_ci, null, &g_xfer_oneshot_fence) != VK_SUCCESS_RC) return false;
    g_xfer_oneshot_ready = true;
    return true;
}

// VK adaptation (encode D2H gather, mirrors decode iter-7+11+15):
// Synchronous batched D2H. Replaces the encode hot-path's per-chunk
// procD2HOffset loop (1526 submit+wait pairs on enwik8 ≈ ~210 ms at
// the Vulkan-on-WDDM ~137 us submit floor) with ONE vkCmdCopyBuffer
// (regionCount = n_regions) on a one-shot cmdbuf from
// g_transfer_cmd_pool, submitted to g_transfer_queue (NVIDIA dedicated
// DMA engine, qf=1) and drained by ONE vkWaitForFences.
//
// Pattern derivation:
//   iter-7 analog: VK_EXT_external_memory_host import wraps dst_host as
//     a VkBuffer with TRANSFER_DST usage. vkCmdCopyBuffer writes DIRECTLY
//     into the caller's host pages — zero staging memcpy. The encode hot
//     path's dst_host is EncodeContext.gpu_out_buf (page-aligned via
//     procMallocHost), so tryImportHostBuffer hits the iter-8 LRU cache
//     on every call after the first.
//   iter-11 analog: submit + wait happen on g_transfer_queue (the
//     dedicated VK_QUEUE_TRANSFER_BIT queue). NVIDIA RTX 4060 Ti has
//     this on the dedicated DMA engine (qf=1, flags=0x0c per the init
//     log); routing the gather here keeps the compute queue free and
//     lets the DMA engine run unconcurrently with whatever the compute
//     queue is doing.
//   iter-15 analog: ONE vkCmdCopyBuffer with regionCount = N folds N
//     copies into a single GPU command. Per-call cost on subsequent
//     encodes is one cache-hit lookup + N copyRegion fills + one
//     submit-wait pair (~137 us) — structurally independent of N.
//
// Sync contract: this call DOES NOT take a stream argument. The caller
// can read dst_host immediately on return. The encode hot path uses it
// AFTER the LZ kernel sync_fn has already drained the compute queue, so
// there's no inter-queue ordering concern (the GPU has finished writing
// d_output before this gather reads it).
//
// Fallback: when the import path fails (extension absent, OOM-disabled,
// span < threshold, or page-alignment mismatch), drop back to a
// per-region procD2HOffset loop so correctness is preserved at the
// price of the per-region submit cost (the pre-fix shape).
fn procD2HOffsetGather(
    dst_host: *anyopaque,
    src: VkDeviceBuffer,
    regions: [*]const vulkan_api.VkBufferCopyRegion,
    n_regions: usize,
) callconv(.c) VkResult {
    if (n_regions == 0) return VK_SUCCESS_RC;
    const src_entry = lookupAlloc(src) orelse return -1;

    // Compute the host-side span this gather will touch. dstOffset is
    // caller-buffer-local (encode passes chunk_descs[i].dst_offset for
    // both src and dst — mirrors CUDA's d_output + dst_off pointer
    // arithmetic). The import-fast-path needs the worst-case host
    // address that vkCmdCopyBuffer will write to so it can size the
    // imported VkBuffer correctly.
    var span_hi: u64 = 0;
    {
        var i: usize = 0;
        while (i < n_regions) : (i += 1) {
            const end = regions[i].dst_offset + regions[i].size;
            if (end > span_hi) span_hi = end;
        }
    }
    if (span_hi == 0) return VK_SUCCESS_RC;
    const span: usize = @intCast(span_hi);

    // Per-region procD2HOffset fallback. Used when the import path is
    // unavailable or fails. Preserves the pre-fix per-call semantics
    // exactly (same VkBufferCopy.srcOffset shape), so correctness is
    // identical — only perf differs.
    const fallbackLoop = struct {
        fn run(
            dst_h: *anyopaque,
            src_b: VkDeviceBuffer,
            regs: [*]const vulkan_api.VkBufferCopyRegion,
            n: usize,
        ) VkResult {
            const dst_bytes = @as([*]u8, @ptrCast(dst_h));
            var i: usize = 0;
            while (i < n) : (i += 1) {
                const r = regs[i];
                if (r.size == 0) continue;
                const rc = procD2HOffset(
                    @as(*anyopaque, @ptrCast(dst_bytes + r.dst_offset)),
                    src_b,
                    @intCast(r.size),
                    @intCast(r.src_offset),
                );
                if (rc != VK_SUCCESS_RC) return rc;
            }
            return VK_SUCCESS_RC;
        }
    }.run;

    // Threshold matches the H2D import gate. Tiny gathers stay on the
    // staging path (pre-fix loop) where amortization across the
    // ~200 us alloc cost is poor.
    const D2H_GATHER_IMPORT_THRESHOLD: usize = 4 * 1024 * 1024;
    if (span < D2H_GATHER_IMPORT_THRESHOLD) return fallbackLoop(dst_host, src, regions, n_regions);

    // Iter-7 fast path: import dst_host as a TRANSFER_DST VkBuffer.
    // tryImportHostBuffer returns null on extension absence / OOM-
    // disabled / misalignment — fall back to the per-region loop.
    const imported = tryImportHostBuffer(dst_host, span, false) orelse
        return fallbackLoop(dst_host, src, regions, n_regions);

    // Iter-11 fast path: end-to-end on g_transfer_queue. Lazily set up
    // the cmdbuf + fence the first time we get here.
    if (!ensureXferOneshotReady()) return fallbackLoop(dst_host, src, regions, n_regions);

    markImportInFlight(imported.cache_idx);

    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);

    // Reset + Begin the one-shot transfer cmdbuf.
    _ = vkResetCommandBuffer_fn.?(g_xfer_oneshot_cb, 0);
    const begin = VkCommandBufferBeginInfo{
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (vkBeginCommandBuffer_fn.?(g_xfer_oneshot_cb, &begin) != VK_SUCCESS_RC) {
        clearImportInFlight(imported.cache_idx);
        return fallbackLoop(dst_host, src, regions, n_regions);
    }

    // Iter-15 fast path: ONE vkCmdCopyBuffer with regionCount = N.
    // vulkan_api.VkBufferCopyRegion has the SAME field layout as
    // VkBufferCopy (u64 srcOffset, u64 dstOffset, u64 size in that
    // order), so the array pointer can be reinterpreted directly.
    if (imported.region_offset == 0) {
        const region_ptr = @as([*]const VkBufferCopy, @ptrCast(regions));
        vkCmdCopyBuffer_fn.?(g_xfer_oneshot_cb, src_entry.buffer, imported.buffer, @intCast(n_regions), region_ptr);
    } else {
        // Sub-page-aligned dst_host case: tryImportHostBuffer rounded
        // down to the surrounding page boundary so the import is legal,
        // and recorded the byte bias in imported.region_offset. We must
        // add that bias to every dstOffset so writes land at the
        // caller's intended bytes. Stack-bounded scratch suffices — the
        // encode hot path uses procMallocHost-backed dst_host so this
        // path triggers only in test contexts where dst_host comes from
        // a plain page_allocator slice that happens to be non-aligned.
        const gpa = std.heap.page_allocator;
        const scratch = gpa.alloc(VkBufferCopy, n_regions) catch {
            clearImportInFlight(imported.cache_idx);
            // Reset the cmdbuf so a future call doesn't see a half-recorded state.
            _ = vkResetCommandBuffer_fn.?(g_xfer_oneshot_cb, 0);
            return fallbackLoop(dst_host, src, regions, n_regions);
        };
        defer gpa.free(scratch);
        var k: usize = 0;
        while (k < n_regions) : (k += 1) {
            scratch[k] = .{
                .srcOffset = regions[k].src_offset,
                .dstOffset = regions[k].dst_offset + imported.region_offset,
                .size = regions[k].size,
            };
        }
        vkCmdCopyBuffer_fn.?(g_xfer_oneshot_cb, src_entry.buffer, imported.buffer, @intCast(n_regions), scratch.ptr);
    }

    var rc = vkEndCommandBuffer_fn.?(g_xfer_oneshot_cb);
    if (rc != VK_SUCCESS_RC) {
        clearImportInFlight(imported.cache_idx);
        // The cmdbuf is now in an undefined state — reset it.
        _ = vkResetCommandBuffer_fn.?(g_xfer_oneshot_cb, 0);
        return fallbackLoop(dst_host, src, regions, n_regions);
    }

    _ = vkResetFences_fn.?(dev, 1, @ptrCast(&g_xfer_oneshot_fence));
    const cb_local = g_xfer_oneshot_cb;
    const submit = VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&cb_local),
    };
    const xfer_q: VkQueue = @ptrFromInt(g_transfer_queue);
    rc = vkQueueSubmit_fn.?(xfer_q, 1, @ptrCast(&submit), g_xfer_oneshot_fence);
    if (rc != VK_SUCCESS_RC) {
        clearImportInFlight(imported.cache_idx);
        return rc;
    }
    rc = vkWaitForFences_fn.?(dev, 1, @ptrCast(&g_xfer_oneshot_fence), 1, ~@as(u64, 0));

    // Single in-flight slot drained: clear the marker so a subsequent
    // gather against the same dst_host can hit the cache. The cached
    // (vk_buf, vk_mem) pair is still live (LRU-owned), only the
    // in-flight bit changes.
    clearImportInFlight(imported.cache_idx);

    // If imported.cache_idx < 0 the entry didn't fit in the LRU cache —
    // free the standalone (vk_buf, vk_mem) pair to avoid a leak (iter-7
    // legacy semantics). The cached case (cache_idx >= 0) leaves
    // ownership with g_import_cache.
    if (imported.cache_idx < 0) {
        vkFreeMemory_fn.?(dev, imported.memory, null);
        vkDestroyBuffer_fn.?(dev, imported.buffer, null);
    }

    return rc;
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

// Iter 9: minimum size that triggers the H2D import fast path. The
// extension's vkCreateBuffer + vkAllocateMemory cost (~200 us per
// alloc on NVIDIA, ~0 on cache hit) is amortized over the H2D copy
// itself; for tiny copies (chunk_descs ~1 MB, n_chunks 4 B) the
// iter-4 staging memcpy + DMA is faster overall on a cache miss.
// Set just under the typical compressed-input block size so the big
// upload imports while the metadata uploads keep their fast path.
// Iter 12: re-enable at 4 MiB. The iter-9 regression (import path
// ~10 ms slower than staging for 58 MB enwik8 comp_input) was rooted
// in QUEUE ROUTING, not in imported-memory itself: at iter-9 time the
// H2D was submitted to the universal COMPUTE queue, whose vkCmdCopyBuffer
// path bypassed NVIDIA's dedicated blit/DMA engine. Iter 11 added a
// dedicated VK_QUEUE_TRANSFER_BIT-only queue (qf=1, flags=0x0c) and
// routes every H2D vkCmdCopyBuffer through it; the imported-memory
// DMA now hits the same fast copy engine the staging path used. The
// expected win is removing the host @memcpy(58 MB) staging cost
// (~3.5 ms) and letting the GPU DMA overlap with host prep + the
// eventual compute submit. Threshold = 4 MiB so comp_input (~58 MB)
// imports while chunk_descs (~1 MB) and n_chunks (4 B) stay on the
// already-cheap staging path.
const H2D_IMPORT_THRESHOLD: usize = 4 * 1024 * 1024;

fn procH2DAsync(dst: VkDeviceBuffer, src: *const anyopaque, size: usize, stream: VkStream) callconv(.c) VkResult {
    const entry = lookupAlloc(dst) orelse return -1;
    if (streamEntryFor(stream)) |se| {
        // VK adaptation: H2D recorded into dedicated VK_QUEUE_TRANSFER_BIT
        // queue (NVIDIA dedicated DMA engine); compute submit waits on a
        // binary semaphore signalled by the transfer submit. Mirrors
        // CUDA's cuMemcpyHtoDAsync auto-routing to the copy engine.
        //
        // The transfer cmdbuf records H2D vkCmdCopyBuffer ONLY. The
        // compute cmdbuf still owns descriptor-set updates, dispatches,
        // and D2H readbacks. streamEndAndWait does the 2-submit dance.
        const rc_begin = streamBeginTransferIfNeeded(se);
        if (rc_begin != VK_SUCCESS_RC) return rc_begin;
        // Also begin the compute cmdbuf eagerly so streamEndAndWait
        // doesn't have to deal with a "transfer-only / no-compute-work"
        // edge case (it would otherwise need to begin+end an empty
        // compute cmdbuf just to consume the semaphore). The codec
        // ALWAYS calls launch_kernel after h2d_async on the same stream
        // in the L1 decode path; this just front-loads the begin.
        const rc_begin_c = streamBeginIfNeeded(se);
        if (rc_begin_c != VK_SUCCESS_RC) return rc_begin_c;

        // Iter 9: prefer a pre-prepared upload import (set by
        // fullGpuLaunchImpl BEFORE uploadInputAndPrefixSum so the
        // ~200 us alloc cost overlaps with whatever ran earlier).
        // Only matches if (host_addr, size) line up exactly — guards
        // against the prep being consumed by the wrong H2D when
        // multiple sub-MB copies precede the big one.
        if (se.prepared_h2d_import) |prep| {
            const src_addr = @intFromPtr(src);
            if (se.prepared_h2d_host_addr == src_addr and se.prepared_h2d_size == size) {
                se.prepared_h2d_import = null;
                se.prepared_h2d_host_addr = 0;
                se.prepared_h2d_size = 0;
                g_h2d_path_prepared_import += 1;
                markImportInFlight(prep.cache_idx);
                // Iter 9: srcOffset = prep.region_offset skips the
                // leading page-pad inside the imported parent region
                // when the caller's src wasn't itself page-aligned.
                const region = VkBufferCopy{ .srcOffset = prep.region_offset, .dstOffset = 0, .size = size };
                vkCmdCopyBuffer_fn.?(se.transfer_cmdbuf, prep.buffer, entry.buffer, 1, @ptrCast(&region));
                const gpa = std.heap.page_allocator;
                se.pending_imported_d2h.append(gpa, prep) catch {
                    clearImportInFlight(prep.cache_idx);
                    if (prep.cache_idx < 0) {
                        const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
                        vkFreeMemory_fn.?(dev, prep.memory, null);
                        vkDestroyBuffer_fn.?(dev, prep.buffer, null);
                    }
                    return -1;
                };
                return VK_SUCCESS_RC;
            }
            // Prep didn't match — leave it stashed for a future call.
        }

        // Iter 9: VK_EXT_external_memory_host fast path for big H2Ds.
        // Mirrors the iter-7/8 D2H fast path: vkCmdCopyBuffer reads
        // directly from the caller's host buffer via an imported
        // TRANSFER_SRC VkBuffer — no staging memcpy on the host side,
        // no double DMA. Gated on H2D_IMPORT_THRESHOLD so the tiny
        // metadata uploads (chunk_descs ~1 MB, n_chunks 4 B) stay on
        // the iter-4 staging path where their amortization is better.
        // tryImportHostBuffer returns null on misalignment / extension
        // absence — falls through to the staging path.
        if (size >= H2D_IMPORT_THRESHOLD) {
            const mutable_src: *anyopaque = @constCast(src);
            if (tryImportHostBuffer(mutable_src, size, true)) |imported| {
                g_h2d_path_inline_import += 1;
                markImportInFlight(imported.cache_idx);
                // Iter 9: srcOffset accounts for sub-page offset of the
                // caller's src within the imported page-aligned region.
                const region = VkBufferCopy{ .srcOffset = imported.region_offset, .dstOffset = 0, .size = size };
                vkCmdCopyBuffer_fn.?(se.transfer_cmdbuf, imported.buffer, entry.buffer, 1, @ptrCast(&region));
                const gpa = std.heap.page_allocator;
                se.pending_imported_d2h.append(gpa, imported) catch {
                    clearImportInFlight(imported.cache_idx);
                    if (imported.cache_idx < 0) {
                        const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
                        vkFreeMemory_fn.?(dev, imported.memory, null);
                        vkDestroyBuffer_fn.?(dev, imported.buffer, null);
                    }
                    return -1;
                };
                return VK_SUCCESS_RC;
            }
        }

        // Iter 4 staging path (fallback): bump-allocate from the
        // per-stream staging arena so back-to-back h2d_async calls on
        // the same stream don't clobber each other. The arena resets
        // in streamEndAndWait after the GPU consumes the data.
        const off = streamStagingBump(se, size) orelse return -1;
        const mapped = se.staging_mapped orelse return -1;
        @memcpy(@as([*]u8, @ptrCast(mapped))[off .. off + size], @as([*]const u8, @ptrCast(src))[0..size]);
        const region = VkBufferCopy{ .srcOffset = off, .dstOffset = 0, .size = size };
        vkCmdCopyBuffer_fn.?(se.transfer_cmdbuf, se.staging_buffer, entry.buffer, 1, @ptrCast(&region));
        g_h2d_path_staging += 1;
        return VK_SUCCESS_RC;
    }
    return procH2D(dst, src, size);
}

// Iter 8 subfix 1: cache lookup. Returns the slot index on (host_ptr,
// size) match. Updates last_used_ns. Caller checks `in_flight` before
// using the entry: an in-flight cached entry means the previous D2H
// hasn't drained yet (e.g. concurrent stream) — fall through to a fresh
// allocation rather than racing the GPU.
fn lookupImportedBuffer(host_addr: usize, size: usize, usage_src: bool) ?usize {
    for (g_import_cache, 0..) |maybe, i| {
        if (maybe) |e| {
            if (e.host_ptr == host_addr and e.size == size and e.usage_src == usage_src and !e.in_flight) {
                g_import_cache[i].?.last_used_ns = vulkan_api.qpcNow();
                _ = g_import_cache_hits.fetchAdd(1, .seq_cst);
                return i;
            }
        }
    }
    _ = g_import_cache_misses.fetchAdd(1, .seq_cst);
    return null;
}

// Iter 8 subfix 1: insert (or evict-then-insert). Returns the slot
// index that now holds the entry, or null on catastrophic failure
// (shouldn't happen — cache is fixed-size). Evicted entry's GPU
// resources are vkDestroyBuffer + vkFreeMemory'd.
fn insertImportedBuffer(host_addr: usize, size: usize, vk_buf: VkBuffer, vk_mem: VkDeviceMemory, usage_src: bool) ?usize {
    // First try an empty slot.
    for (g_import_cache, 0..) |maybe, i| {
        if (maybe == null) {
            g_import_cache[i] = .{
                .host_ptr = host_addr,
                .size = size,
                .vk_buf = vk_buf,
                .vk_mem = vk_mem,
                .last_used_ns = vulkan_api.qpcNow(),
                .in_flight = false,
                .usage_src = usage_src,
            };
            return i;
        }
    }
    // Cache full — pick LRU among NOT-in-flight entries. (An in-flight
    // entry holds a GPU reference we must not destroy.)
    var lru_idx: ?usize = null;
    var lru_ts: i64 = std.math.maxInt(i64);
    for (g_import_cache, 0..) |maybe, i| {
        if (maybe) |e| {
            if (e.in_flight) continue;
            if (e.last_used_ns < lru_ts) {
                lru_ts = e.last_used_ns;
                lru_idx = i;
            }
        }
    }
    const evict = lru_idx orelse return null; // every slot is in-flight; drop
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const old = g_import_cache[evict].?;
    vkDestroyBuffer_fn.?(dev, old.vk_buf, null);
    vkFreeMemory_fn.?(dev, old.vk_mem, null);
    _ = g_import_cache_evictions.fetchAdd(1, .seq_cst);
    g_import_cache[evict] = .{
        .host_ptr = host_addr,
        .size = size,
        .vk_buf = vk_buf,
        .vk_mem = vk_mem,
        .last_used_ns = vulkan_api.qpcNow(),
        .in_flight = false,
        .usage_src = usage_src,
    };
    return evict;
}

// Iter 7: try to set up a transient (VkBuffer, VkDeviceMemory) pair that
// wraps the caller's host pointer via VK_EXT_external_memory_host. On
// success the GPU can DMA straight into the caller's buffer through this
// VkBuffer — no staging, no post-fence host @memcpy. Returns null when:
//   * the extension isn't supported, or
//   * the caller's pointer / size aren't aligned to the imported-host
//     alignment (typically 4096), or
//   * vkGetMemoryHostPointerPropertiesEXT reports no compatible mem type, or
//   * any of the create/allocate/bind calls fail.
//
// Iter 8 subfix 1: consults g_import_cache first. On hit, returns the
// cached (VkBuffer, VkDeviceMemory) pair with cache_idx set and skips
// the entire vkCreateBuffer+vkAllocateMemory+vkBindBufferMemory triplet
// (~1 ms on NVIDIA). On miss, allocates fresh and inserts into cache.
// The caller MUST mark cache_idx in-flight via markImportInFlight(idx)
// before the GPU touches the buffer, and clear it via
// markImportNotInFlight(idx) after the fence wait.
//
// Iter 8 subfix 2: prefers a HOST_CACHED memory type when present
// (NVIDIA exposes one — HOST_VISIBLE | HOST_COHERENT | HOST_CACHED on a
// system-memory heap), falling back to HOST_VISIBLE-only otherwise. The
// HOST_CACHED type uses cached BAR1 pages which the CPU and GPU can both
// touch at full bus speed; the HOST_VISIBLE-only fallback often lands
// on uncached BAR1 (~9 GB/s on RTX 4060 Ti vs ~14 GB/s for HOST_CACHED).
fn tryImportHostBuffer(host_ptr: *anyopaque, size: usize, usage_src: bool) ?PendingImportedD2H {
    if (!g_ext_memory_host_supported) return null;
    if (g_imported_host_alignment == 0) return null;
    // VK adaptation (OOM noise fix, 2026-06-07): per-direction sticky
    // skip after the first vkAllocateMemory(import) returns
    // VK_ERROR_OUT_OF_DEVICE_MEMORY. The fail-set lives at the very
    // bottom of this function and trips only when an import alloc
    // actually fails — so bench-mode runs whose pinned input/output
    // come from procMallocHost (which the driver accepts) keep the
    // zero-copy fast path, while one-shot -c/-d CLI runs whose IO
    // buffers come from plain page-allocator slices (which NVIDIA
    // RTX 4060 Ti rejects on the H2D-SRC direction) take the
    // staging fallback once and silently skip the alloc on every
    // subsequent cache miss. Without this short-circuit each cache
    // miss re-triggered the BestPractices-Error-Result validation
    // warning (~1-5 per encode, ~2 per decode); with it, exactly
    // one warning per direction fires per process lifetime — the
    // one that originally set the flag.
    if (usage_src and g_import_disabled_h2d) return null;
    if (!usage_src and g_import_disabled_d2h) return null;
    const align_mask = g_imported_host_alignment - 1;
    const addr = @intFromPtr(host_ptr);
    if (size == 0) return null;

    // Iter 9: allow sub-page-aligned host pointers by importing the
    // surrounding page-aligned region and recording the byte offset so
    // vkCmdCopyBuffer can skip the leading pad. Critical for H2D
    // because block_payload starts at offset header_size (14-26 B)
    // within the pinned src buffer — never page-aligned by itself. The
    // D2H side keeps working because its pointer was already page-
    // aligned (procMallocHost returns page-aligned host allocations
    // and the dispatcher offsets are page multiples).
    //
    // For the import to be safe the caller MUST own the entire
    // page-aligned region we're about to import: the leading pad bytes
    // (addr - aligned_base) and the trailing pad (aligned_size_for_buf -
    // (offset + size)). procMallocHost rounds its allocation up to a
    // page (iter 7) so its trailing pad is owned; for an externally-
    // owned pinned buffer (bench's pinned input), the read-file slice
    // was copied into a page-rounded allocation by allocHost which has
    // the same property.
    //
    // Iter 9 caveat: leading-pad ownership for the H2D src case
    // requires the caller's outer allocation to start at or before the
    // page boundary. The bench's allocHost returns the exact base of a
    // page-rounded allocation, so its region starts EXACTLY on a page
    // — leading pad is zero for offset==0 callers and at most one page
    // for offset==header_size callers, but the page itself belongs to
    // the allocation.
    const aligned_base_addr: usize = addr & ~@as(usize, align_mask);
    const region_offset: u64 = @intCast(addr - aligned_base_addr);
    const aligned_size_for_buf: u64 = ((@as(u64, region_offset) + @as(u64, size)) + align_mask) & ~@as(u64, align_mask);

    // Iter 8 subfix 1: cache lookup first. On hit, skip every Vk call
    // below — the (vk_buf, vk_mem) pair is still valid (vkAllocateMemory
    // with imported host pointer is stable as long as the underlying
    // host allocation lives, which it does for the lifetime of the CLI's
    // pinned output buffer).
    //
    // Iter 9: cache key uses (aligned_base_addr, aligned_size_for_buf,
    // usage_src) so two sub-page-offset imports against the same
    // parent page-aligned buffer share one VkBuffer. The returned
    // record carries the per-call region_offset.
    if (lookupImportedBuffer(aligned_base_addr, @intCast(aligned_size_for_buf), usage_src)) |hit| {
        g_import_prep_cache_hits += 1;
        const e = g_import_cache[hit].?;
        return .{ .buffer = e.vk_buf, .memory = e.vk_mem, .cache_idx = @intCast(hit), .region_offset = region_offset };
    }

    const aligned_size: u64 = aligned_size_for_buf;

    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const get_props = vkGetMemoryHostPointerPropertiesEXT_fn orelse return null;

    // Pass the aligned-base address (not the caller's offset pointer)
    // to vkGetMemoryHostPointerPropertiesEXT — the spec requires the
    // queried pointer be host-pointer-aligned.
    const aligned_base_ptr: *anyopaque = @ptrFromInt(aligned_base_addr);
    var host_ptr_props: VkMemoryHostPointerPropertiesEXT = .{};
    if (get_props(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, aligned_base_ptr, &host_ptr_props) != VK_SUCCESS_RC) return null;
    if (host_ptr_props.memoryTypeBits == 0) return null;

    // Create a VkBuffer sized to the aligned region. usage_src=true →
    // TRANSFER_SRC (H2D: GPU reads from this buffer into the device-local
    // input). usage_src=false → TRANSFER_DST (D2H: GPU writes into this
    // buffer from the device-local output). vkCmdCopyBuffer requires
    // matching SRC/DST bits on the source and destination respectively.
    //
    // VK adaptation (valfix A — VUID-vkBindBufferMemory-memory-02985):
    // chain VkExternalMemoryBufferCreateInfo on pNext so the buffer
    // advertises the same external handle type its memory will be imported
    // with (VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT). The
    // VkImportMemoryHostPointerInfoEXT struct chained on the
    // VkMemoryAllocateInfo below is the symmetric half — both are required.
    const ext_mem_buf_info = VkExternalMemoryBufferCreateInfo{
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
    };
    const bci = VkBufferCreateInfo{
        .pNext = @ptrCast(&ext_mem_buf_info),
        .size = aligned_size,
        .usage = if (usage_src) VK_BUFFER_USAGE_TRANSFER_SRC_BIT else VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    var buf: VkBuffer = 0;
    if (vkCreateBuffer_fn.?(dev, &bci, null, &buf) != VK_SUCCESS_RC) return null;

    var buf_req: VkMemoryRequirements = .{
        .size = 0,
        .alignment = 0,
        .memoryTypeBits = 0,
    };
    vkGetBufferMemoryRequirements_fn.?(dev, buf, &buf_req);
    // The memory type must be supported by BOTH the buffer's own
    // memory-type mask AND the pointer's importable mask.
    const compatible_mask: u32 = buf_req.memoryTypeBits & host_ptr_props.memoryTypeBits;
    if (compatible_mask == 0) {
        vkDestroyBuffer_fn.?(dev, buf, null);
        return null;
    }

    // Iter 8 subfix 2: prefer HOST_CACHED among compatible HOST_VISIBLE
    // memory types. NVIDIA's importable types include both
    // (HOST_VISIBLE | HOST_COHERENT) at index N and
    // (HOST_VISIBLE | HOST_COHERENT | HOST_CACHED) at index M; iter 7
    // picked the first match which lands on the uncached BAR1 type on
    // this driver. HOST_CACHED gives the CPU a writeback cache for
    // reads of GPU output — bench mode then reads bytes back at full
    // memory speed instead of streaming over uncached PCIe.
    //
    // Iter 9: direction-aware. HOST_CACHED is great for D2H (CPU reads
    // GPU output → cache hits help) but HARMFUL for H2D (CPU wrote the
    // bytes earlier; HOST_CACHED forces the GPU DMA to snoop the CPU
    // cache or wait on a writeback, which on NVIDIA RTX 4060 Ti slows
    // 58 MB H2D from ~4 ms (HOST_COHERENT direct DMA from RAM) to
    // ~12 ms (HOST_CACHED with snoops). Prefer NON-cached for
    // usage_src=true; cached for usage_src=false.
    var mt_idx: i32 = -1;
    var best_flags: u32 = 0;
    var i: u32 = 0;
    while (i < g_phys_mem_props.memoryTypeCount) : (i += 1) {
        if ((compatible_mask & (@as(u32, 1) << @intCast(i))) == 0) continue;
        const flags = g_phys_mem_props.memoryTypes[i].propertyFlags;
        if ((flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) continue;
        // Skip DEVICE_LOCAL+HOST_VISIBLE (resizable-BAR heap): writes
        // through it stage via BAR1 at uncached speed for D2H. The
        // system-RAM heaps almost always win for the import path.
        const is_dev_local = (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT_CONST) != 0;
        const is_cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT_CONST) != 0;
        if (mt_idx < 0) {
            // First match — always take it as a fallback.
            mt_idx = @intCast(i);
            best_flags = flags;
        } else {
            // Upgrade rules (in priority order):
            //   1. If current best is DEVICE_LOCAL and this isn't, swap.
            //   2. For D2H (usage_src=false): prefer cached.
            //      For H2D (usage_src=true): prefer NON-cached.
            const cur_dev_local = (best_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT_CONST) != 0;
            const cur_cached = (best_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT_CONST) != 0;
            if (cur_dev_local and !is_dev_local) {
                mt_idx = @intCast(i);
                best_flags = flags;
            } else if (cur_dev_local == is_dev_local) {
                const cache_better = if (usage_src) (cur_cached and !is_cached) else (!cur_cached and is_cached);
                if (cache_better) {
                    mt_idx = @intCast(i);
                    best_flags = flags;
                }
            }
        }
    }
    if (mt_idx < 0) {
        vkDestroyBuffer_fn.?(dev, buf, null);
        return null;
    }

    // Iter 8 subfix 2 / Iter 9: one-time stderr log per direction so we
    // can verify the right memory type was chosen. Gated on
    // SLZ_VK_PROFILE_PHASES=1 to avoid noise on production runs.
    {
        const LogState = struct {
            var src_logged: bool = false;
            var dst_logged: bool = false;
        };
        const need_log = if (usage_src) !LogState.src_logged else !LogState.dst_logged;
        if (need_log) {
            if (usage_src) LogState.src_logged = true else LogState.dst_logged = true;
            if (std.c.getenv("SLZ_VK_PROFILE_PHASES") != null) {
                const has_visible = (best_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
                const has_coherent = (best_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT_CONST) != 0;
                const has_cached = (best_flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT_CONST) != 0;
                const has_dev_local = (best_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT_CONST) != 0;
                const dir: []const u8 = if (usage_src) "H2D-SRC" else "D2H-DST";
                std.debug.print("import-mem-type[{s}] idx={d} flags=0x{x:0>8} HOST_VISIBLE={} HOST_COHERENT={} HOST_CACHED={} DEVICE_LOCAL={}\n", .{
                    dir, mt_idx, best_flags, has_visible, has_coherent, has_cached, has_dev_local,
                });
            }
        }
        // Keep g_import_mem_type_logged true so any pre-existing code
        // path that checks it doesn't re-log.
        g_import_mem_type_logged = true;
    }

    // Chain VkImportMemoryHostPointerInfoEXT off the allocate-info pNext.
    // Per spec allocationSize must EXACTLY equal the imported region —
    // already rounded above. Iter 9: pHostPointer is the page-aligned
    // BASE (aligned_base_ptr), not the caller's potentially-offset
    // pointer.
    //
    // VK adaptation (OOM noise fix, 2026-06-07): a single
    // vkAllocateMemory attempt — on OOM we flip the per-direction
    // sticky disable flag so all subsequent imports for this direction
    // bypass the alloc entirely and short-circuit to staging fallback.
    // This matches the pre-fix behavior of "1 alloc per call" and
    // additionally guarantees only ONE BestPractices-Error-Result
    // warning per direction per process (vs. one per cache miss before),
    // which on NVIDIA RTX 4060 Ti drops H2D from 5+ warnings per encode
    // to exactly 1 warning total (only emitted on the very first
    // import attempt that triggers the driver rejection).
    var import_info = VkImportMemoryHostPointerInfoEXT{
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        .pHostPointer = aligned_base_ptr,
    };
    const mai = VkMemoryAllocateInfo{
        .pNext = @ptrCast(&import_info),
        .allocationSize = aligned_size,
        .memoryTypeIndex = @intCast(mt_idx),
    };
    var mem: VkDeviceMemory = 0;
    const alloc_rc = vkAllocateMemory_fn.?(dev, &mai, null, &mem);
    if (alloc_rc != VK_SUCCESS_RC) {
        // Sticky direction-level disable: every subsequent call for
        // this usage_src short-circuits at the top of tryImportHost
        // Buffer. Pre-fix this OOM fired once per cache miss; now it
        // fires exactly once per direction per process.
        if (usage_src) g_import_disabled_h2d = true else g_import_disabled_d2h = true;
        vkDestroyBuffer_fn.?(dev, buf, null);
        return null;
    }
    if (vkBindBufferMemory_fn.?(dev, buf, mem, 0) != VK_SUCCESS_RC) {
        vkFreeMemory_fn.?(dev, mem, null);
        vkDestroyBuffer_fn.?(dev, buf, null);
        return null;
    }
    g_import_prep_cache_misses += 1;
    // Iter 8 subfix 1: park the fresh entry in the LRU cache. Iter 9:
    // cache key is (aligned_base_addr, aligned_size, usage_src) — see
    // lookupImportedBuffer above.
    const slot = insertImportedBuffer(aligned_base_addr, @intCast(aligned_size), buf, mem, usage_src);
    if (slot) |idx| {
        return .{ .buffer = buf, .memory = mem, .cache_idx = @intCast(idx), .region_offset = region_offset };
    }
    // Cache insert failed (every slot in-flight). Fall back to iter-7
    // legacy semantics: streamEndAndWait will free this pair after the
    // fence wait. cache_idx = -1 signals "not cached".
    return .{ .buffer = buf, .memory = mem, .cache_idx = -1, .region_offset = region_offset };
}

// Iter 8 subfix 1: mark a cache slot as in-flight. Called when the GPU
// will read/write the underlying VkBuffer. Subsequent
// lookupImportedBuffer() requests with the same (host_ptr, size) will
// skip this slot (return null) so they don't race the in-flight op.
fn markImportInFlight(cache_idx: i32) void {
    if (cache_idx < 0) return;
    const i: usize = @intCast(cache_idx);
    if (i >= G_IMPORT_CACHE_CAP) return;
    if (g_import_cache[i]) |*e| e.in_flight = true;
}

// Iter 8 subfix 1: clear the in-flight marker on every slot whose
// VkBuffer participated in a now-completed D2H. Called from
// streamEndAndWait after vkWaitForFences signals — the GPU is no
// longer touching any of these buffers, so they're safe to reuse on
// the next decode.
fn clearImportInFlight(cache_idx: i32) void {
    if (cache_idx < 0) return;
    const i: usize = @intCast(cache_idx);
    if (i >= G_IMPORT_CACHE_CAP) return;
    if (g_import_cache[i]) |*e| e.in_flight = false;
}

// Iter 8 subfix 3: PUBLIC wrapper exposed to decode_dispatch.zig so the
// dispatcher can call vkCreateBuffer+vkAllocateMemory+vkBindBufferMemory
// (or — on cache hit — nothing) BEFORE runBackHalf, overlapping the
// ~1 ms alloc cost with the LZ kernel's execution on the GPU. The
// dispatcher stashes the returned record and procD2HAsync consumes it
// directly. Returns null when the host pointer isn't import-eligible
// (unaligned, extension unavailable, etc.) — caller falls through to
// the iter-4 staging path.
pub fn prepareImportHostBuffer(host_ptr: *anyopaque, size: usize) ?PendingImportedD2H {
    return tryImportHostBuffer(host_ptr, size, false);
}

// Iter 9: H2D-side analog of prepareImportHostBuffer. Pre-creates (or
// fetches from cache) a TRANSFER_SRC imported VkBuffer wrapping the
// caller's source pointer. Lets the dispatcher overlap the
// vkCreateBuffer+vkAllocateMemory+vkBindBufferMemory cost with whatever
// else is running before the H2D actually executes. Returns null when
// the import path isn't eligible (unaligned src, extension absent, etc.)
// — caller falls through to the iter-4 staging path.
//
// Const-ness: VkImportMemoryHostPointerInfoEXT.pHostPointer is declared
// non-const in the spec but the extension treats the imported region as
// read-only when the VkBuffer is TRANSFER_SRC only. Caller's pointer
// stays logically const for H2D.
pub fn prepareImportHostBufferForUpload(host_ptr: *const anyopaque, size: usize) ?PendingImportedD2H {
    const mutable_ptr: *anyopaque = @constCast(host_ptr);
    return tryImportHostBuffer(mutable_ptr, size, true);
}

// Iter 8 subfix 3: type alias so decode_dispatch.zig can store the
// returned record without re-declaring the layout. Kept as a thin
// re-export so this remains the single source of truth.
pub const PreparedImport = PendingImportedD2H;

// Iter 8 subfix 3: stash a pre-prepared import on the given stream's
// slot. procD2HAsync consumes it on the next call to that stream.
// Returns true on success, false when the stream handle is invalid or
// the slot is already populated (shouldn't happen — one prep per
// decode). On failure the caller still owns `prepared` and must NOT
// re-cache it via tryImportHostBuffer (the cache slot is already in
// the LRU table).
pub fn stashPreparedImportForStream(stream: VkStream, prepared: PreparedImport) bool {
    const se = streamEntryFor(stream) orelse return false;
    if (se.prepared_d2h_import != null) {
        // Caller bug — silently drop the new prep. We don't free the
        // GPU resources: cache_idx >= 0 means the cache still owns
        // them; cache_idx == -1 would leak, but that's a no-op on the
        // fast path because subfix 1 always succeeds with idx >= 0
        // when cache slots are free.
        return false;
    }
    se.prepared_d2h_import = prepared;
    return true;
}

// Iter 9: H2D analog of stashPreparedImportForStream. The dispatcher
// pre-allocates (or fetches from cache) the TRANSFER_SRC imported
// VkBuffer for the compressed-input upload before procH2DAsync runs.
// procH2DAsync matches against (host_addr, size) on consumption so a
// concurrent prep against a different src pointer can't accidentally
// short-circuit the wrong copy.
pub fn stashPreparedUploadImportForStream(stream: VkStream, host_addr: usize, size: usize, prepared: PreparedImport) bool {
    const se = streamEntryFor(stream) orelse return false;
    if (se.prepared_h2d_import != null) return false;
    se.prepared_h2d_import = prepared;
    se.prepared_h2d_host_addr = host_addr;
    se.prepared_h2d_size = size;
    return true;
}

// Iter 8 subfix 3: discard a stashed prep without consuming it. Used
// when the dispatcher decides to skip the D2H entirely (e.g. direct
// d_output_target write). Does NOT free the cache resources — the
// prep entry is still valid in g_import_cache for the next decode.
pub fn discardStashedImportForStream(stream: VkStream) void {
    const se = streamEntryFor(stream) orelse return;
    se.prepared_d2h_import = null;
}

// Iter 8 subfix 1: callable from CLI / driver epilog to drop any
// cached imports that reference a now-going-away host pointer range.
// Required for one-shot decodes against an mmap'd output: the mmap
// gets unmap()'d before the file is truncated, but a cached
// VkDeviceMemory still imports against those pages — the OS may
// refuse the truncate while the pages are referenced. Walks the LRU
// cache and destroys any entry whose host_ptr falls in [base, base+len).
// Skips in-flight entries (caller bug: must drain the stream first).
pub fn releaseImportsByHostRange(base: *const anyopaque, len: usize) void {
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    if (vkDestroyBuffer_fn == null or vkFreeMemory_fn == null) return;
    const lo = @intFromPtr(base);
    const hi = lo + len;
    for (g_import_cache, 0..) |maybe, i| {
        if (maybe) |e| {
            if (e.in_flight) continue;
            if (e.host_ptr >= lo and e.host_ptr < hi) {
                vkDestroyBuffer_fn.?(dev, e.vk_buf, null);
                vkFreeMemory_fn.?(dev, e.vk_mem, null);
                g_import_cache[i] = null;
            }
        }
    }
}

// Iter 15: VK adaptation — single-submit consolidation. Pre-iter-15 the
// D2H phase recorded into a fresh compute cmdbuf AFTER runBackHalf had
// already submit+waited the cmdbuf carrying the LZ kernel. The submit
// boundary itself enforced the COMPUTE_SHADER_WRITE → TRANSFER_READ
// dependency on the LZ kernel's output buffer.
//
// In iter-15 the runBackHalf-end stream_sync is removed; the D2H is
// recorded into the SAME cmdbuf as the LZ kernel and the cmdbuf is
// submitted ONCE after finalize. Without an explicit barrier the
// vkCmdCopyBuffer (TRANSFER stage read) could be reordered against the
// LZ kernel (COMPUTE_SHADER stage write) inside the device's command
// processor — undefined behavior per the Vulkan synchronization spec.
//
// Emit a single global VkMemoryBarrier on the cmdbuf at the boundary.
// Cheap (microseconds, no buffer-list traversal); fires once per decode.
inline fn recordComputeToTransferBarrier(cb: VkCommandBuffer) void {
    const mb = VkMemoryBarrier{
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
    };
    vkCmdPipelineBarrier_fn.?(
        cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        @ptrCast(&mb),
        0,
        null,
        0,
        null,
    );
}

// Iter 4: COMPUTE_SHADER_WRITE → COMPUTE_SHADER_READ pipeline barrier.
// Vulkan does NOT insert any synchronization between consecutive
// vkCmdDispatch calls on the same cmdbuf — even on the same buffer the
// second dispatch can run concurrently with the first per the spec. The
// general LZ decoder reads entropy_scratch that huff_decode_4stream
// writes; without this barrier the LZ kernel can observe stale data.
// CUDA hides the same RAW behind cuLaunchKernel's per-stream ordering;
// the VK port must do it by hand.
inline fn recordComputeToComputeBarrier(cb: VkCommandBuffer) void {
    const mb = VkMemoryBarrier{
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier_fn.?(
        cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        @ptrCast(&mb),
        0,
        null,
        0,
        null,
    );
}

fn procComputeToComputeBarrier(stream: VkStream) callconv(.c) VkResult {
    // On per-stream paths the cmdbuf is already recording; on the
    // default-stream path each launch_kernel submits inline so there is
    // no open cmdbuf to record into — the inline submit+wait already
    // provides the strict ordering, making the barrier redundant.
    if (streamEntryFor(stream)) |se| {
        const rc = streamBeginIfNeeded(se);
        if (rc != VK_SUCCESS_RC) return rc;
        recordComputeToComputeBarrier(se.cmdbuf);
    }
    return VK_SUCCESS_RC;
}

fn procD2HAsync(dst: *anyopaque, src: VkDeviceBuffer, size: usize, stream: VkStream) callconv(.c) VkResult {
    const entry = lookupAlloc(src) orelse return -1;
    if (streamEntryFor(stream)) |se| {
        const rc_begin = streamBeginIfNeeded(se);
        if (rc_begin != VK_SUCCESS_RC) return rc_begin;

        // Iter 15: emit the COMPUTE_SHADER_WRITE → TRANSFER_READ barrier
        // before any vkCmdCopyBuffer below. See recordComputeToTransferBarrier
        // doc-comment for the single-submit rationale. Safe to emit
        // unconditionally — in the dispatcher's flow procD2HAsync is
        // always preceded by the LZ kernel's vkCmdDispatch into the same
        // cmdbuf, and even when it isn't the barrier is a no-op cost-wise.
        recordComputeToTransferBarrier(se.cmdbuf);

        // Iter 8 subfix 3: prefer a pre-prepared import (set by
        // fullGpuLaunchImpl before runBackHalf so the alloc cost
        // overlaps with the LZ kernel). Only valid when the caller's
        // dst/size match exactly — defensive check below uses the
        // host_ptr key embedded in the cache entry. On cache hit
        // (most bench runs) prep is ~0 ms and just hands back the
        // existing (vk_buf, vk_mem) pair.
        if (se.prepared_d2h_import) |prep| {
            // Reset the slot regardless of which branch we take below
            // — the prep is single-shot per decode.
            se.prepared_d2h_import = null;
            const ok_for_this_d2h = blk: {
                // Verify the prep matches THIS d2h call. Iter 9: cache
                // key is now (aligned_base, aligned_size, usage_src) —
                // a matching prep against caller's dst exists when the
                // page-aligned base + aligned region of (dst,size)
                // matches the cache entry. cache_idx == -1 (cache
                // overflow) makes us trust the caller blindly.
                if (prep.cache_idx < 0) break :blk true;
                const ci: usize = @intCast(prep.cache_idx);
                if (ci >= G_IMPORT_CACHE_CAP) break :blk false;
                if (g_import_cache[ci]) |e| {
                    const dst_addr = @intFromPtr(dst);
                    const align_mask = g_imported_host_alignment - 1;
                    const dst_base = dst_addr & ~@as(usize, align_mask);
                    const dst_off: u64 = @intCast(dst_addr - dst_base);
                    const dst_region: u64 = ((@as(u64, dst_off) + @as(u64, size)) + align_mask) & ~@as(u64, align_mask);
                    break :blk (e.host_ptr == dst_base and e.size == @as(usize, @intCast(dst_region)) and !e.usage_src);
                }
                break :blk false;
            };
            if (ok_for_this_d2h) {
                markImportInFlight(prep.cache_idx);
                // Iter 9: dstOffset = prep.region_offset to skip the
                // leading page-pad inside the imported parent region.
                const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = prep.region_offset, .size = size };
                vkCmdCopyBuffer_fn.?(se.cmdbuf, entry.buffer, prep.buffer, 1, @ptrCast(&region));
                const gpa = std.heap.page_allocator;
                se.pending_imported_d2h.append(gpa, prep) catch {
                    // Append failed — for cached entries we just clear
                    // the in-flight bit (resources stay in cache). For
                    // uncached (-1) entries we'd leak — but that's the
                    // same behavior as iter 7 in this branch and the
                    // ArrayList append basically never fails on a
                    // freshly-reset stream.
                    clearImportInFlight(prep.cache_idx);
                    if (prep.cache_idx < 0) {
                        const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
                        vkFreeMemory_fn.?(dev, prep.memory, null);
                        vkDestroyBuffer_fn.?(dev, prep.buffer, null);
                    }
                    return -1;
                };
                return VK_SUCCESS_RC;
            }
            // Prep didn't match — fall through. The prep entry stays
            // in the cache (still valid for a future matching d2h).
        }

        // Iter 7: try the VK_EXT_external_memory_host fast path first.
        // If the caller's buffer is page-aligned (procMallocHost rounds
        // to 4 KiB) and the extension is supported, we vkCmdCopyBuffer
        // straight into the imported VkBuffer wrapping the caller's
        // host memory — no staging, no host @memcpy. The transient
        // (VkBuffer, VkDeviceMemory) pair is freed after the fence
        // wait in streamEndAndWait. Mirrors src_vulkan/l1_codec.zig
        // (importHostPointerBuffer) and CUDA's cuMemcpyDtoH_v2 against
        // a pinned destination (src/decode/decode_dispatch.zig:485).
        if (tryImportHostBuffer(dst, size, false)) |imported| {
            markImportInFlight(imported.cache_idx);
            // Iter 9: dstOffset = imported.region_offset accounts for
            // sub-page offset of caller's dst within imported region.
            const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = imported.region_offset, .size = size };
            vkCmdCopyBuffer_fn.?(se.cmdbuf, entry.buffer, imported.buffer, 1, @ptrCast(&region));
            const gpa = std.heap.page_allocator;
            se.pending_imported_d2h.append(gpa, imported) catch {
                // Append failed — for cached entries just clear the
                // in-flight bit (cache still owns them). For uncached
                // entries we must free now (streamEndAndWait won't see them).
                clearImportInFlight(imported.cache_idx);
                if (imported.cache_idx < 0) {
                    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
                    vkFreeMemory_fn.?(dev, imported.memory, null);
                    vkDestroyBuffer_fn.?(dev, imported.buffer, null);
                }
                return -1;
            };
            return VK_SUCCESS_RC;
        }

        // Iter 4 staging path: defer both the cmdbuf record AND the
        // staging → host memcpy so a single end-of-decode
        // streamEndAndWait flushes all queued H2Ds + kernels + D2Hs in
        // ONE submit. Used as the fallback when the caller's buffer
        // isn't page-aligned or the extension is unavailable.
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
    // Iter 7: bump alignment from 64 B → 4 KiB (page) so the CLI's
    // pinned output buffer is eligible for the VK_EXT_external_memory_host
    // import fast path in procD2HAsync. minImportedHostPointerAlignment
    // is 4096 on NVIDIA / AMD / Intel desktop; 64 B was not enough.
    // CUDA's cuMemAllocHost returns page-aligned memory implicitly —
    // this matches that shape, eliminating the iter-4 dst→staging copy.
    // Round size up to the page so the allocation tail also satisfies
    // VkImportMemoryHostPointerInfoEXT.allocationSize % alignment == 0.
    const page: usize = 4096;
    const rounded = (size + page - 1) & ~(page - 1);
    // 4096 = 2^12 → Alignment enum value 12. std.mem.Alignment only
    // exposes named values up to 64; larger powers-of-two go through
    // @enumFromInt on the log2 exponent.
    const buf = gpa.alignedAlloc(u8, @as(std.mem.Alignment, @enumFromInt(12)), rounded) catch return -1;
    // VK adaptation: g_host_allocs is shared with procFreeHost; concurrent
    // mutate without the registry lock corrupts the ArrayList backing.
    lockAllocRegistry();
    defer unlockAllocRegistry();
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
    // VK adaptation: hold the registry lock for the lookup+swapRemove so
    // a concurrent procMallocHost append cannot mutate the underlying
    // slice mid-walk. The gpa.free is moved out of the locked region —
    // page_allocator.free is thread-safe and we no longer need the lock
    // after swapRemove returns.
    var freed_slice: ?[]align(4096) u8 = null;
    {
        lockAllocRegistry();
        defer unlockAllocRegistry();
        for (g_host_allocs.items, 0..) |entry, i| {
            if (entry.ptr == target) {
                freed_slice = @alignCast(entry.ptr[0..entry.len]);
                _ = g_host_allocs.swapRemove(i);
                break;
            }
        }
    }
    if (freed_slice) |slice| {
        gpa.free(slice);
        return VK_SUCCESS_RC;
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

// VK adaptation: end+submit the transfer cmdbuf at end of upload so
// the dedicated DMA engine starts H2D in parallel with the host
// runBackHalf prep (kernel record + descriptor binding +
// prepareImportedOutputBuffer). Mirrors CUDA cuMemcpyHtoD's natural
// ordering where the synchronous host call has already drained the
// copy engine by the time the back-half stream_sync fires.
//
// Pre-iter-13 the transfer queue submit happened inside
// streamEndAndWait alongside the compute submit, ~39us apart per
// nsys capture. The compute submit's COMPUTE_SHADER_BIT semaphore
// wait then serialized the LZ kernel behind the full H2D, so the
// back-half fence wait = transfer_time + kernel_time = ~5ms + ~3ms.
//
// With this early flush, the transfer queue starts the H2D
// immediately while host code records the LZ kernel + bindings.
// By the time streamEndAndWait submits compute, transfer is nearly
// done; the semaphore wait collapses to ~0 and the back-half fence
// shrinks to ~kernel_time = ~3ms.
//
// Subsequent procH2DAsync calls on this stream are NOT supported
// after a flush — the codec is required to have completed all H2Ds
// before calling this. Iter-13 moves the 4-byte n_chunks H2D (the
// only post-upload H2D in the L1 path) into uploadInputAndPrefixSum
// to satisfy this invariant. If a future codec change introduces a
// new post-flush H2D, this function should reject it (or fall back
// to a second-semaphore design).
fn procStreamFlushTransfer(stream: VkStream) callconv(.c) VkResult {
    const se = streamEntryFor(stream) orelse return VK_SUCCESS_RC; // stream==0 has no transfer cmdbuf
    if (!se.transfer_recording) return VK_SUCCESS_RC; // no H2D work — nothing to flush
    if (se.transfer_already_submitted) return VK_SUCCESS_RC; // idempotent

    const rc_end = vkEndCommandBuffer_fn.?(se.transfer_cmdbuf);
    if (rc_end != VK_SUCCESS_RC) return rc_end;

    const xfer_cb = se.transfer_cmdbuf;
    const sem_handle = se.h2d_sem;
    const xfer_submit = VkSubmitInfo{
        .commandBufferCount = 1,
        .pCommandBuffers = @ptrCast(&xfer_cb),
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = @ptrCast(&sem_handle),
    };
    const xfer_q: VkQueue = @ptrFromInt(g_transfer_queue);
    const rc_sub = vkQueueSubmit_fn.?(xfer_q, 1, @ptrCast(&xfer_submit), VK_NULL_HANDLE);
    if (rc_sub != VK_SUCCESS_RC) return rc_sub;

    se.transfer_recording = false;
    se.transfer_already_submitted = true;
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
    // VK adaptation: VkCommandPool external-synchronization rule —
    // vkAllocateCommandBuffers / vkFreeCommandBuffers on the same
    // VkCommandPool must be serialized across threads (Vulkan spec
    // "Host Synchronization"). Acquire the registry lock for the
    // ENTIRE procStreamCreate body so the cmdbuf allocations from
    // g_command_pool + g_transfer_cmd_pool can't race with sibling
    // procStreamCreate / procStreamDestroy / procLaunchKernel calls
    // on per-test DecodeContexts. Same lock guards g_streams append.
    lockAllocRegistry();
    defer unlockAllocRegistry();

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

    // Iter 11: VK adaptation — per-stream transfer cmdbuf off the
    // dedicated-transfer command pool, plus a binary semaphore the
    // transfer submit signals + compute submit waits on. Binary
    // semaphore is created unsignalled (default flags = 0) and auto-
    // resets on each consumed wait, so a single instance survives the
    // full lifetime of the stream.
    var xfer_cb: VkCommandBuffer = null;
    const xfer_cb_alloc = VkCommandBufferAllocateInfo{
        .commandPool = g_transfer_cmd_pool,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers_fn.?(dev, &xfer_cb_alloc, @ptrCast(&xfer_cb)) != VK_SUCCESS_RC) {
        vkDestroyFence_fn.?(dev, fence, null);
        vkFreeCommandBuffers_fn.?(dev, g_command_pool, 1, @ptrCast(&cb));
        return -1;
    }
    var h2d_sem: VkSemaphore = VK_NULL_HANDLE;
    const sem_ci = VkSemaphoreCreateInfo{};
    if (vkCreateSemaphore_fn.?(dev, &sem_ci, null, &h2d_sem) != VK_SUCCESS_RC) {
        vkFreeCommandBuffers_fn.?(dev, g_transfer_cmd_pool, 1, @ptrCast(&xfer_cb));
        vkDestroyFence_fn.?(dev, fence, null);
        vkFreeCommandBuffers_fn.?(dev, g_command_pool, 1, @ptrCast(&cb));
        return -1;
    }

    const gpa = std.heap.page_allocator;
    const entry = StreamEntry{
        .cmdbuf = cb,
        .fence = fence,
        .recording = false,
        .transfer_cmdbuf = xfer_cb,
        .transfer_recording = false,
        .h2d_sem = h2d_sem,
    };
    // VK adaptation: g_streams append is already protected by the
    // outer lockAllocRegistry (covering the full procStreamCreate
    // body so vkAllocateCommandBuffers honours the VkCommandPool
    // external-synchronization rule); the SRWLOCK is non-recursive so
    // we cannot re-acquire here.
    // Reuse a freed slot if available.
    for (g_streams.items, 0..) |slot, i| {
        if (slot == null) {
            g_streams.items[i] = entry;
            out.* = i + 1;
            return VK_SUCCESS_RC;
        }
    }
    g_streams.append(gpa, entry) catch {
        vkDestroySemaphore_fn.?(dev, h2d_sem, null);
        vkFreeCommandBuffers_fn.?(dev, g_transfer_cmd_pool, 1, @ptrCast(&xfer_cb));
        vkDestroyFence_fn.?(dev, fence, null);
        vkFreeCommandBuffers_fn.?(dev, g_command_pool, 1, @ptrCast(&cb));
        return -1;
    };
    out.* = g_streams.items.len;
    return VK_SUCCESS_RC;
}

fn procStreamDestroy(stream: VkStream) callconv(.c) VkResult {
    if (stream == 0) return VK_SUCCESS_RC;
    // VK adaptation: vkFreeCommandBuffers on g_command_pool /
    // g_transfer_cmd_pool requires external synchronization (Vulkan
    // spec) — concurrent per-test DecodeContext deinit() calls would
    // race here. Same lock guards g_streams slot clear.
    lockAllocRegistry();
    defer unlockAllocRegistry();
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
    // VK adaptation (valfix B): drain + free any lingering deferred-destroy
    // staging buffers. Defensive — streamEndAndWait normally drains, but
    // an early teardown (e.g. test cancel between begin and sync) could
    // leave entries here.
    if (slot.deferred_staging_destroy.items.len > 0) {
        for (slot.deferred_staging_destroy.items) |h| {
            vma.destroyBuffer(allocator(), h.buffer, h.allocation);
        }
        slot.deferred_staging_destroy.deinit(std.heap.page_allocator);
    } else {
        slot.deferred_staging_destroy.deinit(std.heap.page_allocator);
    }
    slot.pending_d2h.deinit(std.heap.page_allocator);
    // Iter 7: drain any pending imported-D2H records before tearing the
    // stream down. freePendingImportedD2H is safe even if the GPU never
    // completed (procStreamDestroy should only be called after a sync),
    // but be defensive in case the caller skipped the sync.
    freePendingImportedD2H(&slot);
    slot.pending_imported_d2h.deinit(std.heap.page_allocator);
    // Iter 11: tear down the per-stream transfer cmdbuf + binary semaphore.
    if (slot.h2d_sem != VK_NULL_HANDLE) {
        vkDestroySemaphore_fn.?(dev, slot.h2d_sem, null);
    }
    if (slot.transfer_cmdbuf != null) {
        var xfer_cb = slot.transfer_cmdbuf;
        vkFreeCommandBuffers_fn.?(dev, g_transfer_cmd_pool, 1, @ptrCast(&xfer_cb));
    }
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

/// Per-kernel GPU profile readback. Walks every (start, end) pair
/// recorded by procLaunchKernel under SLZ_VK_PROFILE_DECODE=1, sums
/// ns by KERNEL_DECLS kind, prints one "kper: <kind> count=N total=Xms
/// avg=Yus" line per kernel to the supplied writer, then resets the
/// record cursor so the next decode iteration starts clean.
/// Safe to call when profiling is disabled (no-op).
pub fn printAndResetProfile(w: anytype) void {
    if (!g_profile_enabled) return;
    if (g_profile_record_count == 0) return;
    // Drain every stream so the timestamps we wrote have committed.
    // (procCtxSync covers both per-stream cmdbufs and a deviceWaitIdle.)
    _ = procCtxSync();

    const get_results = vkGetQueryPoolResults_fn orelse return;
    const dev: VkDevice = @ptrFromInt(vulkan_api.ctx);
    const flags: u32 = VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT;

    // Per-kind running totals (count + total ticks).
    const N = KERNEL_DECLS.len;
    var counts: [N]u64 = @splat(0);
    var totals_ticks: [N]u64 = @splat(0);

    var idx: u32 = 0;
    while (idx < g_profile_record_count) : (idx += 1) {
        const rec = g_profile_records[idx];
        if (rec.kind_index < 0) continue;
        var start_ts: u64 = 0;
        var end_ts: u64 = 0;
        if (get_results(dev, g_timestamp_query_pool, rec.start_slot, 1, @sizeOf(u64), @ptrCast(&start_ts), @sizeOf(u64), flags) != VK_SUCCESS_RC) continue;
        if (get_results(dev, g_timestamp_query_pool, rec.end_slot, 1, @sizeOf(u64), @ptrCast(&end_ts), @sizeOf(u64), flags) != VK_SUCCESS_RC) continue;
        if (end_ts < start_ts) continue;
        const ki: usize = @intCast(rec.kind_index);
        counts[ki] += 1;
        totals_ticks[ki] += (end_ts - start_ts);
    }

    w.print("kper: --- per-kernel GPU profile ({d} dispatches recorded, {d} overflow) ---\n", .{ g_profile_record_count, g_profile_overflow_count }) catch {};
    inline for (KERNEL_DECLS, 0..) |decl, i| {
        if (counts[i] > 0) {
            const ticks_f: f64 = @floatFromInt(totals_ticks[i]);
            const total_ns: f64 = ticks_f * @as(f64, g_timestamp_period_ns);
            const total_ms: f64 = total_ns / 1_000_000.0;
            const avg_us: f64 = (total_ns / @as(f64, @floatFromInt(counts[i]))) / 1000.0;
            w.print("kper: {s} count={d} total={d:.4}ms avg={d:.3}us\n", .{
                @tagName(decl.kind), counts[i], total_ms, avg_us,
            }) catch {};
        }
    }

    g_profile_record_count = 0;
    g_profile_overflow_count = 0;
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
    // Iter 4c: per-binding byte offset into the bound storage buffer.
    // CUDA reference: src/decode/scan_gpu.zig:148-155 — CUDA encodes
    // sub-region binding as `base + offset` pointer arithmetic on a real
    // CUdeviceptr before passing it as a kernel arg. Vulkan can't fold
    // offset into the opaque VkDeviceBuffer handle, so the codec passes
    // the base handle plus a per-binding offset; the offset is wired into
    // VkDescriptorBufferInfo.offset below. `null` selects legacy behaviour
    // (all bindings at offset 0). Callers MUST align offsets to
    // VkPhysicalDeviceLimits.minStorageBufferOffsetAlignment — this fn
    // asserts in safe builds and rejects with rc=-1 in release builds so
    // the caller surfaces error.KernelLaunchFailed rather than a silent
    // VUID hit when the validation layers are absent.
    binding_offsets: ?[*]const u64,
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
    var se_opt: ?*StreamEntry = null;
    if (streamEntryFor(stream)) |se| {
        const rc = streamBeginIfNeeded(se);
        if (rc != VK_SUCCESS_RC) return rc;
        cb = se.cmdbuf;
        inline_submit_wait = false;
        se_opt = se;
    } else {
        const rc = beginOneShotCB();
        if (rc != VK_SUCCESS_RC) return rc;
        cb = g_command_buffer;
        fence_for_wait = g_fence;
    }

    // Iter 14 (A): persistent descriptor sets eliminate the per-decode
    // vkAllocateDescriptorSets cost. The set is allocated lazily on
    // first use per (stream × pipeline) and reused for the lifetime of
    // the StreamEntry (or the loader, for the stream==0 path); per call
    // we only rewrite the buffer handles via vkUpdateDescriptorSets and
    // rebind via vkCmdBindDescriptorSets. Race-free because each
    // StreamEntry owns its own slot and concurrent decodes never share
    // a stream; the default-stream slot (KernelMeta.persistent_desc_set_default)
    // is serialised by g_dispatcher_lock + the inline submit+wait below.
    //
    // VK adaptation: descriptor sets persist per (stream × pipeline)
    // for the lifetime of the StreamEntry; per-decode work is just
    // vkUpdateDescriptorSets + vkCmdBindDescriptorSets instead of
    // vkAllocateDescriptorSets. Mirrors CUDA where cuLaunchKernel takes
    // a parameter array directly without per-call descriptor allocation.
    var dset: VkDescriptorSet = 0;
    if (meta.n_bindings > 0) {
        // Iter 14 (A): persistent descriptor sets eliminate the per-decode
        // vkAllocateDescriptorSets cost. Iter 4c: re-using a persistent
        // set within the SAME cmdbuf across launches with DIFFERENT
        // bindings violates VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358
        // (the set must not be updated between record and submit unless
        // it was created with UPDATE_AFTER_BIND). So when binding_offsets
        // is non-null (i.e., the caller is binding sub-regions, which is
        // also exactly the path that repeats launches into one cmdbuf
        // with different bindings — compact_huff x4, etc.), allocate a
        // fresh transient set per launch from the dedicated
        // g_transient_descriptor_pool. The pool is reset at every
        // streamEndAndWait / submitAndWait boundary (just before the
        // next decode begins recording). The iter 14 fast path is
        // preserved for the (binding_offsets == null) common case.
        const use_transient = binding_offsets != null;
        var slot_ptr: *VkDescriptorSet = &meta.persistent_desc_set_default;
        var transient_storage: VkDescriptorSet = 0;
        if (use_transient) {
            slot_ptr = &transient_storage;
        } else if (se_opt) |se| {
            const k_idx = kindIndexForPipeline(pipeline);
            if (k_idx >= 0 and @as(usize, @intCast(k_idx)) < se.persistent_desc_sets.len) {
                slot_ptr = &se.persistent_desc_sets[@intCast(k_idx)];
            }
        }

        if (slot_ptr.* == 0) {
            if (use_transient) {
                // Iter 4c: opportunistic transient-pool reset. Tighten
                // the lock around the count check + the reset itself so
                // a concurrent ptest_vk worker can't sneak a fresh
                // alloc into a pool we're about to reset.
                lockTransient();
                defer unlockTransient();
                if (g_transient_set_count >= g_transient_set_reset_threshold) {
                    // Drain every in-flight cmdbuf so the reset doesn't
                    // invalidate sets referenced by a not-yet-submitted
                    // (or in-flight) cmdbuf. vkDeviceWaitIdle is heavy
                    // but only fires once per ~3500 transient allocs.
                    if (vkDeviceWaitIdle_fn) |wi| _ = wi(dev);
                    if (vkResetDescriptorPool_fn) |rp| _ = rp(dev, g_transient_descriptor_pool, 0);
                    g_transient_set_count = 0;
                }
                const layouts_arr = [_]VkDescriptorSetLayout{meta.layout};
                const ai = VkDescriptorSetAllocateInfo{
                    .descriptorPool = g_transient_descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = layouts_arr[0..1].ptr,
                };
                const rc = vkAllocateDescriptorSets_fn.?(dev, &ai, @ptrCast(slot_ptr));
                if (rc != VK_SUCCESS_RC) return rc;
                g_transient_set_count += 1;
            } else {
                // First touch on this (stream × pipeline) — allocate once
                // into the persistent pool (iter 14).
                const layouts_arr = [_]VkDescriptorSetLayout{meta.layout};
                const ai = VkDescriptorSetAllocateInfo{
                    .descriptorPool = g_descriptor_pool,
                    .descriptorSetCount = 1,
                    .pSetLayouts = layouts_arr[0..1].ptr,
                };
                const rc = vkAllocateDescriptorSets_fn.?(dev, &ai, @ptrCast(slot_ptr));
                if (rc != VK_SUCCESS_RC) return rc;
            }
        }
        dset = slot_ptr.*;

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
        // Iter 4c: per-binding byte offset (CUDA's `base + offset` pattern
        // expressed in Vulkan via VkDescriptorBufferInfo.offset). Offsets
        // must meet the device's minStorageBufferOffsetAlignment limit;
        // we assert it in safe builds and reject misaligned offsets in
        // release with rc=-1 so the caller sees error.KernelLaunchFailed
        // even when validation layers are off.
        const ssbo_align: u64 = g_min_storage_buffer_offset_alignment;
        var b: u32 = 0;
        while (b < meta.n_bindings) : (b += 1) {
            const p = params[b] orelse return -1;
            const hp: *const VkDeviceBuffer = @ptrCast(@alignCast(p));
            const handle = hp.*;
            const entry = lookupAlloc(handle) orelse return -1;
            const off: u64 = if (binding_offsets) |bo| bo[b] else 0;
            if (off != 0) {
                if (ssbo_align != 0 and (off % ssbo_align) != 0) {
                    std.debug.assert(false); // misaligned SSBO offset — violates VUID-VkDescriptorBufferInfo-offset-00925
                    return -1;
                }
                if (off >= entry.size) return -1;
            }
            // 2026-06-10: explicit range, clamped to maxStorageBufferRange.
            // A single allocation may exceed the per-binding cap (the 1 GB
            // L3+ entropy scratch is ~6 GB, bound per-region); WHOLE_SIZE /
            // size-off past the cap violates the storage-buffer range limit.
            // Every shader's in-binding window is ≤ 2.14 GB (a region is at
            // most WALK_MAX_CHUNKS × ENTROPY_SCRATCH_SLOT_BYTES), so the
            // clamp never hides bytes a kernel can legally address.
            const range_cap: u64 = if (g_max_storage_buffer_range != 0) g_max_storage_buffer_range else 0xFFFF_FFFF;
            const avail: u64 = entry.size - off;
            infos[b] = .{
                .buffer = entry.buffer,
                .offset = off,
                .range = if (avail > range_cap) range_cap else avail,
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

    // ── Per-kernel profile timestamps (SLZ_VK_PROFILE_DECODE=1) ────────
    // Reserve a (start, end) timestamp pair out of the profile slot
    // range and bracket the dispatch with vkCmdWriteTimestamp. Reset
    // the slots first so re-use across runs is safe. Pure measurement
    // — no behavioural impact when profiling is disabled.
    var profile_start_slot: u32 = 0;
    var profile_end_slot: u32 = 0;
    var profile_active: bool = false;
    if (g_profile_enabled and vkCmdResetQueryPool_fn != null and vkCmdWriteTimestamp_fn != null) {
        if (g_profile_record_count < g_profile_max_records) {
            const rec = g_profile_record_count;
            profile_start_slot = g_profile_slot_base + rec * 2;
            profile_end_slot = profile_start_slot + 1;
            vkCmdResetQueryPool_fn.?(cb, g_timestamp_query_pool, profile_start_slot, 2);
            vkCmdWriteTimestamp_fn.?(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_timestamp_query_pool, profile_start_slot);
            g_profile_records[rec] = .{
                .kind_index = kindIndexForPipeline(pipeline),
                .start_slot = profile_start_slot,
                .end_slot = profile_end_slot,
            };
            g_profile_record_count = rec + 1;
            profile_active = true;
        } else {
            g_profile_overflow_count += 1;
        }
    }

    vkCmdDispatch_fn.?(cb, grid_x, grid_y, grid_z);

    if (profile_active) {
        vkCmdWriteTimestamp_fn.?(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_timestamp_query_pool, profile_end_slot);
    }

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
