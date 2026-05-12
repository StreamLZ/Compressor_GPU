//! Vulkan Compute bridge for GPU LZ decompression.
//! Loads vulkan-1.dll at runtime — no compile-time Vulkan SDK dependency.
//! Pre-compiled SPIR-V is embedded via @embedFile.
//! Fallback path when CUDA is unavailable (AMD, Intel, etc).

const std = @import("std");
const fast_dec = @import("fast_lz_decoder.zig");

const win32 = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *i64) callconv(.c) i32;
    extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.c) i32;
};

fn qpcNow() i64 {
    var v: i64 = 0;
    _ = win32.QueryPerformanceCounter(&v);
    return v;
}

fn qpcFreq() i64 {
    var v: i64 = 0;
    _ = win32.QueryPerformanceFrequency(&v);
    return v;
}

// ── Vulkan handle/constant types ────────────────────────────────
const VkResult = i32;
const VK_SUCCESS: VkResult = 0;
const Handle = ?*anyopaque;
const VkDeviceSize = u64;

const VK_BUFFER_USAGE_STORAGE: u32 = 0x20;
const VK_BUFFER_USAGE_TRANSFER_DST: u32 = 0x2;
const VK_BUFFER_USAGE_TRANSFER_SRC: u32 = 0x1;
const VK_MEM_HOST_VISIBLE: u32 = 0x2;
const VK_MEM_HOST_COHERENT: u32 = 0x4;
const VK_DESC_STORAGE_BUFFER: u32 = 7;
const VK_SHADER_COMPUTE: u32 = 0x20;
const VK_BIND_COMPUTE: u32 = 1;
const VK_QUEUE_COMPUTE: u32 = 0x2;

// sType values
const STYPE_INSTANCE_CI: u32 = 1;
const STYPE_DEVICE_QUEUE_CI: u32 = 2;
const STYPE_DEVICE_CI: u32 = 3;
const STYPE_SUBMIT_INFO: u32 = 4;
const STYPE_MEM_ALLOC: u32 = 5;
const STYPE_FENCE_CI: u32 = 8;
const STYPE_BUFFER_CI: u32 = 12;
const STYPE_SHADER_MODULE_CI: u32 = 16;
const STYPE_PIPELINE_STAGE_CI: u32 = 18;
const STYPE_COMPUTE_PIPE_CI: u32 = 29;
const STYPE_PIPE_LAYOUT_CI: u32 = 30;
const STYPE_DSL_CI: u32 = 32;
const STYPE_DESC_POOL_CI: u32 = 33;
const STYPE_DESC_SET_AI: u32 = 34;
const STYPE_WRITE_DESC_SET: u32 = 35;
const STYPE_CMD_POOL_CI: u32 = 39;
const STYPE_CMD_BUF_AI: u32 = 40;
const STYPE_CMD_BUF_BEGIN: u32 = 42;

// ── Vulkan C-ABI structs (64-bit Windows) ───────────────────────

const VkInstanceCreateInfo = extern struct {
    sType: u32 = STYPE_INSTANCE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*anyopaque = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*anyopaque = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*anyopaque = null,
};

const VkDeviceQueueCreateInfo = extern struct {
    sType: u32 = STYPE_DEVICE_QUEUE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 1,
    pQueuePriorities: ?*const f32 = null,
};

const VkDeviceCreateInfo = extern struct {
    sType: u32 = STYPE_DEVICE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?*const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?*anyopaque = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?*anyopaque = null,
    pEnabledFeatures: ?*anyopaque = null,
};

const VkBufferCreateInfo = extern struct {
    sType: u32 = STYPE_BUFFER_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    size: VkDeviceSize = 0,
    usage: u32 = 0,
    sharingMode: u32 = 0,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?*anyopaque = null,
};

const VkMemoryRequirements = extern struct {
    size: VkDeviceSize,
    alignment: VkDeviceSize,
    memoryTypeBits: u32,
};

const VkMemoryAllocateInfo = extern struct {
    sType: u32 = STYPE_MEM_ALLOC,
    pNext: ?*anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [32]extern struct { propertyFlags: u32, heapIndex: u32 },
    memoryHeapCount: u32,
    memoryHeaps: [16]extern struct { size: VkDeviceSize, flags: u32, _pad: u32 = 0 },
};

const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = 0,
    descriptorType: u32 = 0,
    descriptorCount: u32 = 1,
    stageFlags: u32 = 0,
    pImmutableSamplers: ?*anyopaque = null,
};

const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: u32 = STYPE_DSL_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};

const VkPushConstantRange = extern struct {
    stageFlags: u32 = 0,
    offset: u32 = 0,
    size: u32 = 0,
};

const VkPipelineLayoutCreateInfo = extern struct {
    sType: u32 = STYPE_PIPE_LAYOUT_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?*const Handle = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?*const VkPushConstantRange = null,
};

const VkShaderModuleCreateInfo = extern struct {
    sType: u32 = STYPE_SHADER_MODULE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    codeSize: usize = 0,
    pCode: ?[*]const u32 = null,
};

const VkPipelineShaderStageCreateInfo = extern struct {
    sType: u32 = STYPE_PIPELINE_STAGE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    stage: u32 = 0,
    module: Handle = null,
    pName: ?[*:0]const u8 = null,
    pSpecializationInfo: ?*anyopaque = null,
};

const VkComputePipelineCreateInfo = extern struct {
    sType: u32 = STYPE_COMPUTE_PIPE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    stage: VkPipelineShaderStageCreateInfo = .{},
    layout: Handle = null,
    basePipelineHandle: Handle = null,
    basePipelineIndex: i32 = -1,
};

const VkDescriptorPoolSize = extern struct {
    type_: u32 = 0,
    descriptorCount: u32 = 0,
};

const VkDescriptorPoolCreateInfo = extern struct {
    sType: u32 = STYPE_DESC_POOL_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?*const VkDescriptorPoolSize = null,
};

const VkDescriptorSetAllocateInfo = extern struct {
    sType: u32 = STYPE_DESC_SET_AI,
    pNext: ?*anyopaque = null,
    descriptorPool: Handle = null,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?*const Handle = null,
};

const VkDescriptorBufferInfo = extern struct {
    buffer: Handle = null,
    offset: VkDeviceSize = 0,
    range: VkDeviceSize = 0,
};

const VkWriteDescriptorSet = extern struct {
    sType: u32 = STYPE_WRITE_DESC_SET,
    pNext: ?*anyopaque = null,
    dstSet: Handle = null,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 1,
    descriptorType: u32 = 0,
    pImageInfo: ?*anyopaque = null,
    pBufferInfo: ?*const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*anyopaque = null,
};

const VkCommandPoolCreateInfo = extern struct {
    sType: u32 = STYPE_CMD_POOL_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queueFamilyIndex: u32 = 0,
};

const VkCommandBufferAllocateInfo = extern struct {
    sType: u32 = STYPE_CMD_BUF_AI,
    pNext: ?*anyopaque = null,
    commandPool: Handle = null,
    level: u32 = 0,
    commandBufferCount: u32 = 1,
};

const VkCommandBufferBeginInfo = extern struct {
    sType: u32 = STYPE_CMD_BUF_BEGIN,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    pInheritanceInfo: ?*anyopaque = null,
};

const VkSubmitInfo = extern struct {
    sType: u32 = STYPE_SUBMIT_INFO,
    pNext: ?*anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?*anyopaque = null,
    pWaitDstStageMask: ?*anyopaque = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?*const Handle = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?*anyopaque = null,
};

const VkFenceCreateInfo = extern struct {
    sType: u32 = STYPE_FENCE_CI,
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
};

const VkBufferCopy = extern struct {
    srcOffset: VkDeviceSize = 0,
    dstOffset: VkDeviceSize = 0,
    size: VkDeviceSize = 0,
};

const VkQueueFamilyProperties = extern struct {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: extern struct { width: u32, height: u32, depth: u32 },
};

// ── Function pointer types ──────────────────────────────────────
const FnGetProcAddr = *const fn (Handle, [*:0]const u8) callconv(.c) ?*anyopaque;
const FnVoidResult = *const fn (Handle, *const anyopaque, ?*const anyopaque, *Handle) callconv(.c) VkResult;

// ── Module state ────────────────────────────────────────────────
var lib: ?*anyopaque = null;
var initialized = false;
var vk_ok = false;
var gpa: ?FnGetProcAddr = null;

var vk_instance: Handle = null;
var vk_phys: Handle = null;
var vk_dev: Handle = null;
var vk_queue: Handle = null;
var comp_family: u32 = 0;
var vk_cmd_pool: Handle = null;
var vk_cmd_buf: Handle = null;
var vk_fence: Handle = null;
var vk_pipeline: Handle = null;
var vk_pipe_layout: Handle = null;
var vk_dsl: Handle = null;
var vk_desc_pool: Handle = null;
var vk_desc_set: Handle = null;
var vk_query_pool: Handle = null;
var timestamp_period: f32 = 1.0; // nanoseconds per tick
var mem_props: VkPhysicalDeviceMemoryProperties = undefined;

// Persistent buffers (grow as needed)
var b_comp: Handle = null;
var m_comp: Handle = null;
var b_comp_sz: usize = 0;
var b_out: Handle = null;
var m_out: Handle = null;
var b_out_sz: usize = 0;
var b_desc: Handle = null;
var m_desc: Handle = null;
var b_desc_sz: usize = 0;

pub var last_kernel_ns: i64 = 0;

fn vkProc(comptime T: type, name: [*:0]const u8) ?T {
    const f = gpa orelse return null;
    const raw = f(vk_instance, name) orelse return null;
    return @ptrCast(raw);
}

// ── Initialization ──────────────────────────────────────────────

pub fn init() bool {
    if (initialized) return vk_ok;
    initialized = true;

    {
        const c = @cImport({ @cInclude("stdlib.h"); });
        if (c.getenv("SLZ_NO_VK") != null) return false;
    }

    lib = win32.LoadLibraryA("vulkan-1.dll");
    if (lib == null) return false;
    const raw = win32.GetProcAddress(lib.?, "vkGetInstanceProcAddr") orelse return false;
    gpa = @ptrCast(raw);

    if (!initInstance()) return false;
    if (!pickDevice()) return false;
    if (!initDevice()) return false;
    if (!initPipeline()) return false;
    if (!initDescPool()) return false;
    if (!initCmdResources()) return false;
    _ = initTimestampQuery(); // non-fatal: timestamps are optional

    vk_ok = true;
    return true;
}

pub fn isAvailable() bool {
    return init();
}

fn initInstance() bool {
    const f = vkProc(*const fn (*const VkInstanceCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateInstance") orelse return false;
    var ci = VkInstanceCreateInfo{};
    return f(&ci, null, &vk_instance) == VK_SUCCESS;
}

const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32,
    driverVersion: u32,
    vendorID: u32,
    deviceID: u32,
    deviceType: u32, // 1=integrated, 2=discrete
    deviceName: [256]u8,
    pipelineCacheUUID: [16]u8,
    limits: [504]u8, // VkPhysicalDeviceLimits — large, opaque here
    sparseProperties: [20]u8,
};

fn pickDevice() bool {
    const enumDev = vkProc(*const fn (Handle, *u32, ?[*]Handle) callconv(.c) VkResult, "vkEnumeratePhysicalDevices") orelse return false;
    const getQF = vkProc(*const fn (Handle, *u32, ?[*]VkQueueFamilyProperties) callconv(.c) void, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return false;
    const getMemProps = vkProc(*const fn (Handle, *VkPhysicalDeviceMemoryProperties) callconv(.c) void, "vkGetPhysicalDeviceMemoryProperties") orelse return false;
    const getProps = vkProc(*const fn (Handle, *VkPhysicalDeviceProperties) callconv(.c) void, "vkGetPhysicalDeviceProperties") orelse return false;

    var count: u32 = 0;
    if (enumDev(vk_instance, &count, null) != VK_SUCCESS or count == 0) return false;
    var devs: [16]Handle = .{null} ** 16;
    var n: u32 = @min(count, 16);
    if (enumDev(vk_instance, &n, &devs) != VK_SUCCESS) return false;

    // Two passes: prefer discrete GPUs (type=2), then accept any
    for (0..2) |pass| {
        for (devs[0..n]) |pd| {
            if (pass == 0) {
                var props: VkPhysicalDeviceProperties = undefined;
                getProps(pd, &props);
                if (props.deviceType != 2) continue;
            }
            var qf_count: u32 = 0;
            getQF(pd, &qf_count, null);
            if (qf_count == 0) continue;
            var qf: [16]VkQueueFamilyProperties = undefined;
            var qf_n: u32 = @min(qf_count, 16);
            getQF(pd, &qf_n, &qf);
            for (0..qf_n) |qi| {
                if ((qf[qi].queueFlags & VK_QUEUE_COMPUTE) != 0) {
                    vk_phys = pd;
                    comp_family = @intCast(qi);
                    getMemProps(pd, &mem_props);
                    return true;
                }
            }
        }
    }
    return false;
}

fn initDevice() bool {
    const createDev = vkProc(*const fn (Handle, *const VkDeviceCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateDevice") orelse return false;
    const getQueue = vkProc(*const fn (Handle, u32, u32, *Handle) callconv(.c) void, "vkGetDeviceQueue") orelse return false;

    var prio: f32 = 1.0;
    var qci = VkDeviceQueueCreateInfo{ .queueFamilyIndex = comp_family, .pQueuePriorities = &prio };
    var dci = VkDeviceCreateInfo{ .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };

    if (createDev(vk_phys, &dci, null, &vk_dev) != VK_SUCCESS) return false;
    getQueue(vk_dev, comp_family, 0, &vk_queue);
    return vk_queue != null;
}

fn initPipeline() bool {
    const createSM = vkProc(*const fn (Handle, *const VkShaderModuleCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateShaderModule") orelse return false;
    const createDSL = vkProc(*const fn (Handle, *const VkDescriptorSetLayoutCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateDescriptorSetLayout") orelse return false;
    const createPL = vkProc(*const fn (Handle, *const VkPipelineLayoutCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreatePipelineLayout") orelse return false;
    const createCP = vkProc(*const fn (Handle, Handle, u32, [*]const VkComputePipelineCreateInfo, ?*anyopaque, [*]Handle) callconv(.c) VkResult, "vkCreateComputePipelines") orelse return false;

    const spv align(@alignOf(u32)) = @embedFile("gpu_decode_kernel.spv");
    var sm_ci = VkShaderModuleCreateInfo{ .codeSize = spv.len, .pCode = @ptrCast(@alignCast(spv.ptr)) };
    var shader_mod: Handle = null;
    if (createSM(vk_dev, &sm_ci, null, &shader_mod) != VK_SUCCESS) return false;

    var bindings = [_]VkDescriptorSetLayoutBinding{
        .{ .binding = 0, .descriptorType = VK_DESC_STORAGE_BUFFER, .stageFlags = VK_SHADER_COMPUTE },
        .{ .binding = 1, .descriptorType = VK_DESC_STORAGE_BUFFER, .stageFlags = VK_SHADER_COMPUTE },
        .{ .binding = 2, .descriptorType = VK_DESC_STORAGE_BUFFER, .stageFlags = VK_SHADER_COMPUTE },
    };
    var dsl_ci = VkDescriptorSetLayoutCreateInfo{ .bindingCount = 3, .pBindings = &bindings };
    if (createDSL(vk_dev, &dsl_ci, null, &vk_dsl) != VK_SUCCESS) return false;

    var pc_range = VkPushConstantRange{ .stageFlags = VK_SHADER_COMPUTE, .size = 12 };
    var pl_ci = VkPipelineLayoutCreateInfo{ .setLayoutCount = 1, .pSetLayouts = &vk_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc_range };
    if (createPL(vk_dev, &pl_ci, null, &vk_pipe_layout) != VK_SUCCESS) return false;

    var cp_ci = [1]VkComputePipelineCreateInfo{.{
        .stage = .{ .stage = VK_SHADER_COMPUTE, .module = shader_mod, .pName = "main" },
        .layout = vk_pipe_layout,
    }};
    var pipes: [1]Handle = .{null};
    if (createCP(vk_dev, null, 1, &cp_ci, null, &pipes) != VK_SUCCESS) return false;
    vk_pipeline = pipes[0];
    return true;
}

fn initDescPool() bool {
    const createDP = vkProc(*const fn (Handle, *const VkDescriptorPoolCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateDescriptorPool") orelse return false;
    const allocDS = vkProc(*const fn (Handle, *const VkDescriptorSetAllocateInfo, [*]Handle) callconv(.c) VkResult, "vkAllocateDescriptorSets") orelse return false;

    var ps = VkDescriptorPoolSize{ .type_ = VK_DESC_STORAGE_BUFFER, .descriptorCount = 3 };
    var dp_ci = VkDescriptorPoolCreateInfo{ .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps };
    if (createDP(vk_dev, &dp_ci, null, &vk_desc_pool) != VK_SUCCESS) return false;

    var ds_ai = VkDescriptorSetAllocateInfo{ .descriptorPool = vk_desc_pool, .descriptorSetCount = 1, .pSetLayouts = &vk_dsl };
    var sets: [1]Handle = .{null};
    if (allocDS(vk_dev, &ds_ai, &sets) != VK_SUCCESS) return false;
    vk_desc_set = sets[0];
    return true;
}

fn initCmdResources() bool {
    const createPool = vkProc(*const fn (Handle, *const VkCommandPoolCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateCommandPool") orelse return false;
    const allocCB = vkProc(*const fn (Handle, *const VkCommandBufferAllocateInfo, [*]Handle) callconv(.c) VkResult, "vkAllocateCommandBuffers") orelse return false;
    const createFence = vkProc(*const fn (Handle, *const VkFenceCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateFence") orelse return false;

    var pool_ci = VkCommandPoolCreateInfo{ .flags = 0x2, .queueFamilyIndex = comp_family };
    if (createPool(vk_dev, &pool_ci, null, &vk_cmd_pool) != VK_SUCCESS) return false;

    var cb_ai = VkCommandBufferAllocateInfo{ .commandPool = vk_cmd_pool };
    var bufs: [1]Handle = .{null};
    if (allocCB(vk_dev, &cb_ai, &bufs) != VK_SUCCESS) return false;
    vk_cmd_buf = bufs[0];

    var fence_ci = VkFenceCreateInfo{};
    return createFence(vk_dev, &fence_ci, null, &vk_fence) == VK_SUCCESS;
}

const VkQueryPoolCreateInfo = extern struct {
    sType: u32 = 11, // VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO
    pNext: ?*anyopaque = null,
    flags: u32 = 0,
    queryType: u32 = 2, // VK_QUERY_TYPE_TIMESTAMP
    queryCount: u32 = 2,
    pipelineStatistics: u32 = 0,
};

fn initTimestampQuery() bool {
    const createQP = vkProc(*const fn (Handle, *const VkQueryPoolCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateQueryPool") orelse return false;

    var qp_ci = VkQueryPoolCreateInfo{};
    if (createQP(vk_dev, &qp_ci, null, &vk_query_pool) != VK_SUCCESS) return false;

    timestamp_period = 1.0;

    return true;
}

// ── Buffer helpers ──────────────────────────────────────────────
// Discrete GPUs need device-local memory for compute performance.
// We allocate a staging buffer (host-visible) alongside each device
// buffer and use vkCmdCopyBuffer to transfer between them.

const VK_MEM_DEVICE_LOCAL: u32 = 0x1;

const VK_MEM_HOST_CACHED: u32 = 0x8;

var has_device_local: bool = false;

// Upload staging: host-visible+coherent (write-combining, fast CPU writes)
var b_staging: Handle = null;
var m_staging: Handle = null;
var b_staging_sz: usize = 0;

// Download staging: host-visible+cached (fast CPU reads)
var b_download: Handle = null;
var m_download: Handle = null;
var b_download_sz: usize = 0;

fn findMemType(type_bits: u32, props: u32) ?u32 {
    for (0..mem_props.memoryTypeCount) |i| {
        if ((type_bits & (@as(u32, 1) << @intCast(i))) != 0 and
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return @intCast(i);
    }
    return null;
}

fn createBufWithMem(buf: *Handle, mem: *Handle, needed: usize, usage: u32, mem_flags: u32) bool {
    const destroyBuf = vkProc(*const fn (Handle, Handle, ?*anyopaque) callconv(.c) void, "vkDestroyBuffer") orelse return false;
    const freeMem = vkProc(*const fn (Handle, Handle, ?*anyopaque) callconv(.c) void, "vkFreeMemory") orelse return false;
    const createBuf = vkProc(*const fn (Handle, *const VkBufferCreateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkCreateBuffer") orelse return false;
    const getReqs = vkProc(*const fn (Handle, Handle, *VkMemoryRequirements) callconv(.c) void, "vkGetBufferMemoryRequirements") orelse return false;
    const allocMem = vkProc(*const fn (Handle, *const VkMemoryAllocateInfo, ?*anyopaque, *Handle) callconv(.c) VkResult, "vkAllocateMemory") orelse return false;
    const bindMem = vkProc(*const fn (Handle, Handle, Handle, VkDeviceSize) callconv(.c) VkResult, "vkBindBufferMemory") orelse return false;

    if (buf.* != null) destroyBuf(vk_dev, buf.*, null);
    if (mem.* != null) freeMem(vk_dev, mem.*, null);

    var bci = VkBufferCreateInfo{ .size = needed, .usage = usage };
    if (createBuf(vk_dev, &bci, null, buf) != VK_SUCCESS) return false;

    var reqs: VkMemoryRequirements = undefined;
    getReqs(vk_dev, buf.*, &reqs);

    const mt = findMemType(reqs.memoryTypeBits, mem_flags) orelse return false;
    var mai = VkMemoryAllocateInfo{ .allocationSize = reqs.size, .memoryTypeIndex = mt };
    if (allocMem(vk_dev, &mai, null, mem) != VK_SUCCESS) return false;
    return bindMem(vk_dev, buf.*, mem.*, 0) == VK_SUCCESS;
}

fn ensureBuf(buf: *Handle, mem: *Handle, cur_sz: *usize, needed: usize) bool {
    if (cur_sz.* >= needed) return true;
    cur_sz.* = 0;
    const usage = VK_BUFFER_USAGE_STORAGE | VK_BUFFER_USAGE_TRANSFER_DST | VK_BUFFER_USAGE_TRANSFER_SRC;
    // Try device-local first (fast VRAM), fall back to host-visible
    if (findMemType(0xFFFFFFFF, VK_MEM_DEVICE_LOCAL) != null) {
        if (createBufWithMem(buf, mem, needed, usage, VK_MEM_DEVICE_LOCAL)) {
            has_device_local = true;
            cur_sz.* = needed;
            return true;
        }
    }
    if (createBufWithMem(buf, mem, needed, usage, VK_MEM_HOST_VISIBLE | VK_MEM_HOST_COHERENT)) {
        has_device_local = false;
        cur_sz.* = needed;
        return true;
    }
    return false;
}

fn ensureStaging(needed: usize) bool {
    if (b_staging_sz >= needed) return true;
    b_staging_sz = 0;
    const usage = VK_BUFFER_USAGE_TRANSFER_SRC | VK_BUFFER_USAGE_TRANSFER_DST;
    if (!createBufWithMem(&b_staging, &m_staging, needed, usage, VK_MEM_HOST_VISIBLE | VK_MEM_HOST_COHERENT))
        return false;
    b_staging_sz = needed;
    return true;
}

fn ensureDownloadBuf(needed: usize) bool {
    if (b_download_sz >= needed) return true;
    b_download_sz = 0;
    const usage = VK_BUFFER_USAGE_TRANSFER_DST;
    // Prefer cached memory for fast CPU reads; fall back to coherent
    if (createBufWithMem(&b_download, &m_download, needed, usage, VK_MEM_HOST_VISIBLE | VK_MEM_HOST_CACHED)) {
        b_download_sz = needed;
        return true;
    }
    if (createBufWithMem(&b_download, &m_download, needed, usage, VK_MEM_HOST_VISIBLE | VK_MEM_HOST_COHERENT)) {
        b_download_sz = needed;
        return true;
    }
    return false;
}

fn mapMem(mem_h: Handle, offset: usize, size: usize) ?[*]u8 {
    const mapFn = vkProc(*const fn (Handle, Handle, VkDeviceSize, VkDeviceSize, u32, *?*anyopaque) callconv(.c) VkResult, "vkMapMemory") orelse return null;
    var ptr: ?*anyopaque = null;
    if (mapFn(vk_dev, mem_h, offset, size, 0, &ptr) != VK_SUCCESS) return null;
    return @ptrCast(ptr.?);
}

fn unmapMem(mem_h: Handle) void {
    const unmapFn = vkProc(*const fn (Handle, Handle) callconv(.c) void, "vkUnmapMemory") orelse return;
    unmapFn(vk_dev, mem_h);
}

fn uploadToDevice(dev_buf: Handle, dev_mem: Handle, data: [*]const u8, size: usize) bool {
    if (!has_device_local) {
        const dst = mapMem(dev_mem, 0, size) orelse return false;
        @memcpy(dst[0..size], data[0..size]);
        unmapMem(dev_mem);
        return true;
    }
    if (!ensureStaging(size)) return false;
    const dst = mapMem(m_staging, 0, size) orelse return false;
    @memcpy(dst[0..size], data[0..size]);
    unmapMem(m_staging);
    return copyBuf(b_staging, dev_buf, size);
}

fn downloadFromDevice(dev_buf: Handle, dev_mem: Handle, data: [*]u8, offset: usize, size: usize) bool {
    if (!has_device_local) {
        const src = mapMem(dev_mem, offset, size) orelse return false;
        @memcpy(data[0..size], src[0..size]);
        unmapMem(dev_mem);
        return true;
    }
    if (!ensureStaging(offset + size)) return false;
    if (!copyBuf(dev_buf, b_staging, offset + size)) return false;
    const src = mapMem(m_staging, offset, size) orelse return false;
    @memcpy(data[0..size], src[0..size]);
    unmapMem(m_staging);
    return true;
}

fn copyBuf(src_buf: Handle, dst_buf: Handle, size: usize) bool {
    const resetCB = vkProc(*const fn (Handle, u32) callconv(.c) VkResult, "vkResetCommandBuffer") orelse return false;
    const beginCB = vkProc(*const fn (Handle, *const VkCommandBufferBeginInfo) callconv(.c) VkResult, "vkBeginCommandBuffer") orelse return false;
    const endCB = vkProc(*const fn (Handle) callconv(.c) VkResult, "vkEndCommandBuffer") orelse return false;
    const cmdCopy = vkProc(*const fn (Handle, Handle, Handle, u32, [*]const VkBufferCopy) callconv(.c) void, "vkCmdCopyBuffer") orelse return false;
    const queueSubmit = vkProc(*const fn (Handle, u32, [*]const VkSubmitInfo, Handle) callconv(.c) VkResult, "vkQueueSubmit") orelse return false;
    const waitFence = vkProc(*const fn (Handle, u32, [*]const Handle, u32, u64) callconv(.c) VkResult, "vkWaitForFences") orelse return false;
    const resetFence = vkProc(*const fn (Handle, u32, [*]const Handle) callconv(.c) VkResult, "vkResetFences") orelse return false;

    _ = resetCB(vk_cmd_buf, 0);
    var begin = VkCommandBufferBeginInfo{};
    if (beginCB(vk_cmd_buf, &begin) != VK_SUCCESS) return false;

    const region = VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
    cmdCopy(vk_cmd_buf, src_buf, dst_buf, 1, &[1]VkBufferCopy{region});

    if (endCB(vk_cmd_buf) != VK_SUCCESS) return false;

    const submit = VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &vk_cmd_buf };
    if (queueSubmit(vk_queue, 1, &[1]VkSubmitInfo{submit}, vk_fence) != VK_SUCCESS) return false;
    _ = waitFence(vk_dev, 1, &[1]Handle{vk_fence}, 1, ~@as(u64, 0));
    _ = resetFence(vk_dev, 1, &[1]Handle{vk_fence});
    return true;
}

// ── Public dispatch ─────────────────────────────────────────────

pub const ChunkDesc = extern struct {
    src_offset: u32,
    comp_size: u32,
    decomp_size: u32,
    dst_offset: u32,
    flags: u32,
    memset_fill: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
};

pub fn fullVkLaunch(
    chunk_descs: []const ChunkDesc,
    compressed_block: []const u8,
    dst_full: [*]u8,
    dst_start_off: usize,
    decompressed_size: usize,
    num_groups: u32,
    chunks_per_group: u32,
    sub_chunk_cap: u32,
    io: ?std.Io,
) fast_dec.DecodeError!void {
    if (!init()) return error.BadMode;
    _ = io;

    const updateDS = vkProc(*const fn (Handle, u32, [*]const VkWriteDescriptorSet, u32, ?*anyopaque) callconv(.c) void, "vkUpdateDescriptorSets") orelse return error.BadMode;
    const beginCB = vkProc(*const fn (Handle, *const VkCommandBufferBeginInfo) callconv(.c) VkResult, "vkBeginCommandBuffer") orelse return error.BadMode;
    const endCB = vkProc(*const fn (Handle) callconv(.c) VkResult, "vkEndCommandBuffer") orelse return error.BadMode;
    const resetCB = vkProc(*const fn (Handle, u32) callconv(.c) VkResult, "vkResetCommandBuffer") orelse return error.BadMode;
    const bindPipe = vkProc(*const fn (Handle, u32, Handle) callconv(.c) void, "vkCmdBindPipeline") orelse return error.BadMode;
    const bindDS_fn = vkProc(*const fn (Handle, u32, Handle, u32, u32, [*]const Handle, u32, ?*const u32) callconv(.c) void, "vkCmdBindDescriptorSets") orelse return error.BadMode;
    const pushConst = vkProc(*const fn (Handle, Handle, u32, u32, u32, *const anyopaque) callconv(.c) void, "vkCmdPushConstants") orelse return error.BadMode;
    const dispatchFn = vkProc(*const fn (Handle, u32, u32, u32) callconv(.c) void, "vkCmdDispatch") orelse return error.BadMode;
    const queueSubmit = vkProc(*const fn (Handle, u32, [*]const VkSubmitInfo, Handle) callconv(.c) VkResult, "vkQueueSubmit") orelse return error.BadMode;
    const waitFence = vkProc(*const fn (Handle, u32, [*]const Handle, u32, u64) callconv(.c) VkResult, "vkWaitForFences") orelse return error.BadMode;
    const resetFence = vkProc(*const fn (Handle, u32, [*]const Handle) callconv(.c) VkResult, "vkResetFences") orelse return error.BadMode;
    const cmdCopy = vkProc(*const fn (Handle, Handle, Handle, u32, [*]const VkBufferCopy) callconv(.c) void, "vkCmdCopyBuffer") orelse return error.BadMode;
    const resetQP = vkProc(*const fn (Handle, Handle, u32, u32) callconv(.c) void, "vkCmdResetQueryPool") orelse return error.BadMode;
    const writeTS = vkProc(*const fn (Handle, u32, Handle, u32) callconv(.c) void, "vkCmdWriteTimestamp") orelse return error.BadMode;
    const pipeBarrier = vkProc(*const fn (Handle, u32, u32, u32, ?*const anyopaque, u32, ?*const anyopaque, u32, ?*const anyopaque) callconv(.c) void, "vkCmdPipelineBarrier") orelse return error.BadMode;

    const total_output = dst_start_off + decompressed_size;
    const comp_bytes = if (compressed_block.len > 0) compressed_block.len else 4;
    const desc_bytes = chunk_descs.len * @sizeOf(ChunkDesc);

    if (!ensureBuf(&b_comp, &m_comp, &b_comp_sz, comp_bytes)) return error.BadMode;
    if (!ensureBuf(&b_out, &m_out, &b_out_sz, total_output + 64)) return error.BadMode;
    if (!ensureBuf(&b_desc, &m_desc, &b_desc_sz, desc_bytes)) return error.BadMode;

    // ── Write upload data to host memory ──
    // Staging layout for device-local path: [compressed | descriptors | dict_prefix]
    const stage_off_comp: usize = 0;
    const stage_off_desc: usize = comp_bytes;
    const stage_off_dict: usize = comp_bytes + desc_bytes;
    const staging_upload_size = comp_bytes + desc_bytes + dst_start_off;

    if (has_device_local) {
        if (!ensureStaging(staging_upload_size)) return error.BadMode;
        if (!ensureDownloadBuf(total_output + 64)) return error.BadMode;

        // Single map, write all upload data at sequential offsets
        const p = mapMem(m_staging, 0, staging_upload_size) orelse return error.BadMode;
        if (compressed_block.len > 0)
            @memcpy(p[stage_off_comp..][0..compressed_block.len], compressed_block);
        @memcpy(p[stage_off_desc..][0..desc_bytes], @as([*]const u8, @ptrCast(chunk_descs.ptr))[0..desc_bytes]);
        if (dst_start_off > 0)
            @memcpy(p[stage_off_dict..][0..dst_start_off], dst_full[0..dst_start_off]);
        unmapMem(m_staging);
    } else {
        // Host-visible (iGPU): write directly to device buffers
        if (compressed_block.len > 0) {
            const p = mapMem(m_comp, 0, compressed_block.len) orelse return error.BadMode;
            @memcpy(p[0..compressed_block.len], compressed_block);
            unmapMem(m_comp);
        }
        {
            const p = mapMem(m_desc, 0, desc_bytes) orelse return error.BadMode;
            @memcpy(p[0..desc_bytes], @as([*]const u8, @ptrCast(chunk_descs.ptr))[0..desc_bytes]);
            unmapMem(m_desc);
        }
        if (dst_start_off > 0) {
            const p = mapMem(m_out, 0, dst_start_off) orelse return error.BadMode;
            @memcpy(p[0..dst_start_off], dst_full[0..dst_start_off]);
            unmapMem(m_out);
        }
    }

    // ── Update descriptor set ──
    var buf_infos = [3]VkDescriptorBufferInfo{
        .{ .buffer = b_comp, .range = comp_bytes },
        .{ .buffer = b_out, .range = total_output + 64 },
        .{ .buffer = b_desc, .range = desc_bytes },
    };
    var desc_writes = [3]VkWriteDescriptorSet{
        .{ .dstSet = vk_desc_set, .dstBinding = 0, .descriptorType = VK_DESC_STORAGE_BUFFER, .pBufferInfo = &buf_infos[0] },
        .{ .dstSet = vk_desc_set, .dstBinding = 1, .descriptorType = VK_DESC_STORAGE_BUFFER, .pBufferInfo = &buf_infos[1] },
        .{ .dstSet = vk_desc_set, .dstBinding = 2, .descriptorType = VK_DESC_STORAGE_BUFFER, .pBufferInfo = &buf_infos[2] },
    };
    updateDS(vk_dev, 3, &desc_writes, 0, null);

    // ── Record single command buffer: [copies] → barrier → dispatch → barrier → [copy back] ──
    _ = resetCB(vk_cmd_buf, 0);
    var begin_info = VkCommandBufferBeginInfo{};
    if (beginCB(vk_cmd_buf, &begin_info) != VK_SUCCESS) return error.BadMode;

    if (has_device_local) {
        // Staging → device copies (offsets match staging layout)
        if (compressed_block.len > 0)
            cmdCopy(vk_cmd_buf, b_staging, b_comp, 1, &[1]VkBufferCopy{.{ .srcOffset = stage_off_comp, .size = compressed_block.len }});
        cmdCopy(vk_cmd_buf, b_staging, b_desc, 1, &[1]VkBufferCopy{.{ .srcOffset = stage_off_desc, .size = desc_bytes }});
        if (dst_start_off > 0)
            cmdCopy(vk_cmd_buf, b_staging, b_out, 1, &[1]VkBufferCopy{.{ .srcOffset = stage_off_dict, .size = dst_start_off }});
        // Barrier: transfer → compute
        pipeBarrier(vk_cmd_buf, 0x1000, 0x800, 0, null, 0, null, 0, null);
    }

    bindPipe(vk_cmd_buf, VK_BIND_COMPUTE, vk_pipeline);
    var ds_arr = [1]Handle{vk_desc_set};
    bindDS_fn(vk_cmd_buf, VK_BIND_COMPUTE, vk_pipe_layout, 0, 1, &ds_arr, 0, null);
    var pc_data = [3]u32{ chunks_per_group, @intCast(chunk_descs.len), sub_chunk_cap };
    pushConst(vk_cmd_buf, vk_pipe_layout, VK_SHADER_COMPUTE, 0, 12, @ptrCast(&pc_data));

    const grid_x = (num_groups + 1) / 2;
    resetQP(vk_cmd_buf, vk_query_pool, 0, 2);
    writeTS(vk_cmd_buf, 0x1, vk_query_pool, 0);
    dispatchFn(vk_cmd_buf, grid_x, 1, 1);
    writeTS(vk_cmd_buf, 0x800, vk_query_pool, 1); // VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT

    if (has_device_local) {
        // Barrier: compute → transfer
        pipeBarrier(vk_cmd_buf, 0x800, 0x1000, 0, null, 0, null, 0, null);
        // Device → cached download buffer for output
        cmdCopy(vk_cmd_buf, b_out, b_download, 1, &[1]VkBufferCopy{.{ .srcOffset = dst_start_off, .size = decompressed_size }});
    }

    if (endCB(vk_cmd_buf) != VK_SUCCESS) return error.BadMode;

    // ── Single submit, single fence ──
    const t_submit = qpcNow();
    const submit = VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &vk_cmd_buf };
    if (queueSubmit(vk_queue, 1, &[1]VkSubmitInfo{submit}, vk_fence) != VK_SUCCESS) return error.BadMode;
    _ = waitFence(vk_dev, 1, &[1]Handle{vk_fence}, 1, ~@as(u64, 0));
    _ = resetFence(vk_dev, 1, &[1]Handle{vk_fence});
    const t_after_fence = qpcNow();

    // ── Read GPU timestamps (kernel-only, excludes transfers) ──
    if (vk_query_pool != null) {
        const getResults = vkProc(*const fn (Handle, Handle, u32, u32, usize, *anyopaque, VkDeviceSize, u32) callconv(.c) VkResult, "vkGetQueryPoolResults") orelse return error.BadMode;
        var timestamps: [2]u64 = .{ 0, 0 };
        if (getResults(vk_dev, vk_query_pool, 0, 2, @sizeOf([2]u64), @ptrCast(&timestamps), 8, 3) == VK_SUCCESS and timestamps[1] > timestamps[0]) {
            const ticks = timestamps[1] -% timestamps[0];
            last_kernel_ns = @intFromFloat(@as(f64, @floatFromInt(ticks)) * @as(f64, timestamp_period));
        }
    }
    // Fallback: use fence time if GPU timestamps unavailable
    if (last_kernel_ns <= 0) {
        const freq = qpcFreq();
        last_kernel_ns = @intCast(@divTrunc((t_after_fence - t_submit) * 1_000_000_000, freq));
    }

    // ── Read back output (from cached download buffer — fast CPU reads) ──
    if (has_device_local) {
        const p = mapMem(m_download, 0, decompressed_size) orelse return error.BadMode;
        @memcpy((dst_full + dst_start_off)[0..decompressed_size], p[0..decompressed_size]);
        unmapMem(m_staging);
    } else {
        const p = mapMem(m_out, dst_start_off, decompressed_size) orelse return error.BadMode;
        @memcpy((dst_full + dst_start_off)[0..decompressed_size], p[0..decompressed_size]);
        unmapMem(m_out);
    }
}
