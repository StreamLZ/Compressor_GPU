//! Vulkan Compute bridge for GPU LZ decompression.
//! Loads vulkan-1.dll at runtime — no compile-time Vulkan SDK dependency.
//! Pre-compiled SPIR-V is embedded via @embedFile.
//! Fallback path when CUDA is unavailable (AMD, Intel, etc).

const std = @import("std");
const win32 = struct {
    extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
};

// ── Vulkan type aliases ─────────────────────────────────────────
const VkResult = i32;
const VK_SUCCESS: VkResult = 0;
const VkBool32 = u32;
const VK_TRUE: VkBool32 = 1;
const VK_FALSE: VkBool32 = 0;

const VkInstance = ?*anyopaque;
const VkPhysicalDevice = ?*anyopaque;
const VkDevice = ?*anyopaque;
const VkQueue = ?*anyopaque;
const VkCommandPool = ?*anyopaque;
const VkCommandBuffer = ?*anyopaque;
const VkFence = ?*anyopaque;
const VkBuffer = ?*anyopaque;
const VkDeviceMemory = ?*anyopaque;
const VkShaderModule = ?*anyopaque;
const VkPipeline = ?*anyopaque;
const VkPipelineLayout = ?*anyopaque;
const VkDescriptorSetLayout = ?*anyopaque;
const VkDescriptorPool = ?*anyopaque;
const VkDescriptorSet = ?*anyopaque;
const VkDeviceSize = u64;

// ── Vulkan struct IDs (sType) ───────────────────────────────────
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;

const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x20;
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x1;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x2;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x2;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x4;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x20;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x2;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x2;
const VK_API_VERSION_1_3: u32 = (1 << 22) | (3 << 12);

// ── Module state ────────────────────────────────────────────────
var lib: ?*anyopaque = null;
var initialized = false;
var vk_available = false;

var instance: VkInstance = null;
var phys_device: VkPhysicalDevice = null;
var device: VkDevice = null;
var queue: VkQueue = null;
var compute_family: u32 = 0;
var cmd_pool: VkCommandPool = null;
var cmd_buf: VkCommandBuffer = null;
var fence: VkFence = null;
var pipeline: VkPipeline = null;
var pipe_layout: VkPipelineLayout = null;
var desc_set_layout: VkDescriptorSetLayout = null;
var desc_pool: VkDescriptorPool = null;

// Persistent buffers
var buf_compressed: VkBuffer = null;
var mem_compressed: VkDeviceMemory = null;
var buf_compressed_size: usize = 0;
var buf_output: VkBuffer = null;
var mem_output: VkDeviceMemory = null;
var buf_output_size: usize = 0;
var buf_descs: VkBuffer = null;
var mem_descs: VkDeviceMemory = null;
var buf_descs_size: usize = 0;

pub var last_kernel_ns: i64 = 0;

// ── Function pointer types ──────────────────────────────────────
// We load a minimal set: instance/device creation, memory, commands, compute.
const FnGetInstanceProcAddr = *const fn (VkInstance, [*:0]const u8) callconv(.c) ?*anyopaque;
var vkGetInstanceProcAddr_fn: ?FnGetInstanceProcAddr = null;

fn getInstanceProc(comptime T: type, name: [*:0]const u8) ?T {
    const f = vkGetInstanceProcAddr_fn orelse return null;
    const raw = f(instance, name) orelse return null;
    return @ptrCast(raw);
}

fn getDeviceProc(comptime T: type, name: [*:0]const u8) ?T {
    const f = vkGetInstanceProcAddr_fn orelse return null;
    const raw = f(instance, name) orelse return null;
    return @ptrCast(raw);
}

// ── Initialization ──────────────────────────────────────────────

pub fn init() bool {
    if (initialized) return vk_available;
    initialized = true;

    lib = win32.LoadLibraryA("vulkan-1.dll");
    if (lib == null) return false;

    const raw = win32.GetProcAddress(lib.?, "vkGetInstanceProcAddr") orelse return false;
    vkGetInstanceProcAddr_fn = @ptrCast(raw);

    // Create instance
    if (!createInstance()) return false;
    if (!pickPhysicalDevice()) return false;
    if (!createDevice()) return false;
    if (!createPipeline()) return false;
    if (!createCommandResources()) return false;

    vk_available = true;
    return true;
}

pub fn isAvailable() bool {
    return init();
}

fn createInstance() bool {
    const FnCreateInstance = *const fn (*const anyopaque, ?*const anyopaque, *VkInstance) callconv(.c) VkResult;
    const create = getInstanceProc(FnCreateInstance, "vkCreateInstance") orelse return false;

    // Minimal instance — no layers, no extensions needed for compute-only
    const app_info = [_]u8{0} ** 64; // zeroed VkApplicationInfo
    _ = app_info;

    var ci = std.mem.zeroes([128]u8);
    const ci_ptr: *align(1) u32 = @ptrCast(&ci[0]);
    ci_ptr.* = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    // apiVersion at offset 56 in VkApplicationInfo — skip, default is fine

    var temp_instance: VkInstance = null;
    if (create(@ptrCast(&ci), null, &temp_instance) != VK_SUCCESS) return false;
    instance = temp_instance;
    return true;
}

fn pickPhysicalDevice() bool {
    const FnEnumPhysDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) VkResult;
    const enumerate = getInstanceProc(FnEnumPhysDevices, "vkEnumeratePhysicalDevices") orelse return false;

    var count: u32 = 0;
    if (enumerate(instance, &count, null) != VK_SUCCESS or count == 0) return false;

    var devices: [16]VkPhysicalDevice = .{null} ** 16;
    var fetch_count = @min(count, 16);
    if (enumerate(instance, &fetch_count, &devices) != VK_SUCCESS) return false;

    // Find first device with a compute queue
    const FnGetQueueFamilyProps = *const fn (VkPhysicalDevice, *u32, ?[*]u8) callconv(.c) void;
    const getProps = getInstanceProc(FnGetQueueFamilyProps, "vkGetPhysicalDeviceQueueFamilyProperties") orelse return false;

    for (devices[0..fetch_count]) |pd| {
        var qf_count: u32 = 0;
        getProps(pd, &qf_count, null);
        if (qf_count == 0) continue;

        // Each VkQueueFamilyProperties is 24 bytes, queueFlags at offset 0
        var qf_buf: [16 * 24]u8 = undefined;
        var qf_fetch = @min(qf_count, 16);
        getProps(pd, &qf_fetch, &qf_buf);

        for (0..qf_fetch) |qi| {
            const flags: u32 = @as(*align(1) const u32, @ptrCast(&qf_buf[qi * 24])).*;
            if ((flags & VK_QUEUE_COMPUTE_BIT) != 0) {
                phys_device = pd;
                compute_family = @intCast(qi);
                return true;
            }
        }
    }
    return false;
}

fn createDevice() bool {
    const FnCreateDevice = *const fn (VkPhysicalDevice, *const anyopaque, ?*const anyopaque, *VkDevice) callconv(.c) VkResult;
    const create = getInstanceProc(FnCreateDevice, "vkCreateDevice") orelse return false;
    const FnGetQueue = *const fn (VkDevice, u32, u32, *VkQueue) callconv(.c) void;

    var priority: f32 = 1.0;
    var queue_ci = std.mem.zeroes([40]u8);
    var q_stype: *align(1) u32 = @ptrCast(&queue_ci[0]);
    q_stype.* = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    var q_family: *align(1) u32 = @ptrCast(&queue_ci[16]);
    q_family.* = compute_family;
    var q_count: *align(1) u32 = @ptrCast(&queue_ci[20]);
    q_count.* = 1;
    var q_prio: *align(1) usize = @ptrCast(&queue_ci[24]);
    q_prio.* = @intFromPtr(&priority);

    var dev_ci = std.mem.zeroes([72]u8);
    var d_stype: *align(1) u32 = @ptrCast(&dev_ci[0]);
    d_stype.* = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    var d_qci_count: *align(1) u32 = @ptrCast(&dev_ci[16]);
    d_qci_count.* = 1;
    var d_qci_ptr: *align(1) usize = @ptrCast(&dev_ci[24]);
    d_qci_ptr.* = @intFromPtr(&queue_ci);

    var temp_device: VkDevice = null;
    if (create(phys_device, @ptrCast(&dev_ci), null, &temp_device) != VK_SUCCESS) return false;
    device = temp_device;

    const getQueue = getDeviceProc(FnGetQueue, "vkGetDeviceQueue") orelse return false;
    getQueue(device, compute_family, 0, &queue);
    return queue != null;
}

fn createPipeline() bool {
    // Create shader module from embedded SPIR-V
    const spv align(@alignOf(u32)) = @embedFile("gpu_decode_kernel.spv");
    const FnCreateShaderModule = *const fn (VkDevice, *const anyopaque, ?*const anyopaque, *VkShaderModule) callconv(.c) VkResult;
    const createSM = getDeviceProc(FnCreateShaderModule, "vkCreateShaderModule") orelse return false;

    var sm_ci = std.mem.zeroes([40]u8);
    var sm_stype: *align(1) u32 = @ptrCast(&sm_ci[0]);
    sm_stype.* = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    var sm_size: *align(1) usize = @ptrCast(&sm_ci[16]);
    sm_size.* = spv.len;
    var sm_code: *align(1) usize = @ptrCast(&sm_ci[24]);
    sm_code.* = @intFromPtr(spv.ptr);

    var shader_mod: VkShaderModule = null;
    if (createSM(device, @ptrCast(&sm_ci), null, &shader_mod) != VK_SUCCESS) return false;

    // Descriptor set layout: 3 storage buffers
    const FnCreateDSL = *const fn (VkDevice, *const anyopaque, ?*const anyopaque, *VkDescriptorSetLayout) callconv(.c) VkResult;
    const createDSL = getDeviceProc(FnCreateDSL, "vkCreateDescriptorSetLayout") orelse return false;

    // VkDescriptorSetLayoutBinding is 24 bytes: binding(4), descriptorType(4), descriptorCount(4), stageFlags(4), pImmutableSamplers(8)
    var bindings: [3 * 24]u8 = std.mem.zeroes([3 * 24]u8);
    for (0..3) |bi| {
        const off = bi * 24;
        var b_binding: *align(1) u32 = @ptrCast(&bindings[off]);
        b_binding.* = @intCast(bi);
        var b_type: *align(1) u32 = @ptrCast(&bindings[off + 4]);
        b_type.* = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        var b_count: *align(1) u32 = @ptrCast(&bindings[off + 8]);
        b_count.* = 1;
        var b_stage: *align(1) u32 = @ptrCast(&bindings[off + 12]);
        b_stage.* = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    var dsl_ci = std.mem.zeroes([32]u8);
    var dsl_stype: *align(1) u32 = @ptrCast(&dsl_ci[0]);
    dsl_stype.* = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    var dsl_count: *align(1) u32 = @ptrCast(&dsl_ci[16]);
    dsl_count.* = 3;
    var dsl_bindings: *align(1) usize = @ptrCast(&dsl_ci[24]);
    dsl_bindings.* = @intFromPtr(&bindings);

    if (createDSL(device, @ptrCast(&dsl_ci), null, &desc_set_layout) != VK_SUCCESS) return false;

    // Pipeline layout with push constants (3 × u32 = 12 bytes)
    const FnCreatePL = *const fn (VkDevice, *const anyopaque, ?*const anyopaque, *VkPipelineLayout) callconv(.c) VkResult;
    const createPL = getDeviceProc(FnCreatePL, "vkCreatePipelineLayout") orelse return false;

    // VkPushConstantRange: stageFlags(4), offset(4), size(4)
    var pc_range: [12]u8 = std.mem.zeroes([12]u8);
    var pc_stage: *align(1) u32 = @ptrCast(&pc_range[0]);
    pc_stage.* = VK_SHADER_STAGE_COMPUTE_BIT;
    var pc_size: *align(1) u32 = @ptrCast(&pc_range[8]);
    pc_size.* = 12; // 3 × u32

    var pl_ci = std.mem.zeroes([48]u8);
    var pl_stype: *align(1) u32 = @ptrCast(&pl_ci[0]);
    pl_stype.* = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    var pl_dsl_count: *align(1) u32 = @ptrCast(&pl_ci[16]);
    pl_dsl_count.* = 1;
    var pl_dsl_ptr: *align(1) usize = @ptrCast(&pl_ci[24]);
    pl_dsl_ptr.* = @intFromPtr(&desc_set_layout);
    var pl_pc_count: *align(1) u32 = @ptrCast(&pl_ci[32]);
    pl_pc_count.* = 1;
    var pl_pc_ptr: *align(1) usize = @ptrCast(&pl_ci[40]);
    pl_pc_ptr.* = @intFromPtr(&pc_range);

    if (createPL(device, @ptrCast(&pl_ci), null, &pipe_layout) != VK_SUCCESS) return false;

    // Compute pipeline
    const FnCreateCP = *const fn (VkDevice, ?*anyopaque, u32, *const anyopaque, ?*const anyopaque, *VkPipeline) callconv(.c) VkResult;
    const createCP = getDeviceProc(FnCreateCP, "vkCreateComputePipelines") orelse return false;

    // VkComputePipelineCreateInfo: sType(4), pNext(8), flags(4), stage(VkPipelineShaderStageCreateInfo=48), layout(8), basePipelineHandle(8), basePipelineIndex(4)
    // Total ~96 bytes. stage starts at offset 16.
    // VkPipelineShaderStageCreateInfo: sType(4), pNext(8), flags(4), stage(4), module(8), pName(8), pSpecializationInfo(8)
    var cp_ci = std.mem.zeroes([96]u8);
    var cp_stype: *align(1) u32 = @ptrCast(&cp_ci[0]);
    cp_stype.* = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

    // Embedded stage create info at offset 16
    var stage_stype: *align(1) u32 = @ptrCast(&cp_ci[16]);
    stage_stype.* = 18; // VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
    var stage_stage: *align(1) u32 = @ptrCast(&cp_ci[32]);
    stage_stage.* = VK_SHADER_STAGE_COMPUTE_BIT;
    var stage_module: *align(1) usize = @ptrCast(&cp_ci[40]);
    stage_module.* = @intFromPtr(shader_mod);
    const entry_name: [*:0]const u8 = "main";
    var stage_name: *align(1) usize = @ptrCast(&cp_ci[48]);
    stage_name.* = @intFromPtr(entry_name);

    // layout at offset 64
    var cp_layout: *align(1) usize = @ptrCast(&cp_ci[64]);
    cp_layout.* = @intFromPtr(pipe_layout);

    if (createCP(device, null, 1, @ptrCast(&cp_ci), null, &pipeline) != VK_SUCCESS) return false;

    return true;
}

fn createCommandResources() bool {
    // Command pool
    const FnCreateCmdPool = *const fn (VkDevice, *const anyopaque, ?*const anyopaque, *VkCommandPool) callconv(.c) VkResult;
    const createPool = getDeviceProc(FnCreateCmdPool, "vkCreateCommandPool") orelse return false;

    var pool_ci = std.mem.zeroes([24]u8);
    var pool_stype: *align(1) u32 = @ptrCast(&pool_ci[0]);
    pool_stype.* = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    var pool_flags: *align(1) u32 = @ptrCast(&pool_ci[16]);
    pool_flags.* = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    var pool_family: *align(1) u32 = @ptrCast(&pool_ci[20]);
    pool_family.* = compute_family;

    if (createPool(device, @ptrCast(&pool_ci), null, &cmd_pool) != VK_SUCCESS) return false;

    // Allocate command buffer
    const FnAllocCmdBuf = *const fn (VkDevice, *const anyopaque, *VkCommandBuffer) callconv(.c) VkResult;
    const allocCB = getDeviceProc(FnAllocCmdBuf, "vkAllocateCommandBuffers") orelse return false;

    var cb_ai = std.mem.zeroes([32]u8);
    var cb_stype: *align(1) u32 = @ptrCast(&cb_ai[0]);
    cb_stype.* = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    var cb_pool: *align(1) usize = @ptrCast(&cb_ai[16]);
    cb_pool.* = @intFromPtr(cmd_pool);
    var cb_level: *align(1) u32 = @ptrCast(&cb_ai[24]);
    cb_level.* = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    var cb_count: *align(1) u32 = @ptrCast(&cb_ai[28]);
    cb_count.* = 1;

    if (allocCB(device, @ptrCast(&cb_ai), &cmd_buf) != VK_SUCCESS) return false;

    // Fence
    const FnCreateFence = *const fn (VkDevice, *const anyopaque, ?*const anyopaque, *VkFence) callconv(.c) VkResult;
    const createFence = getDeviceProc(FnCreateFence, "vkCreateFence") orelse return false;

    var fence_ci = std.mem.zeroes([16]u8);
    var fence_stype: *align(1) u32 = @ptrCast(&fence_ci[0]);
    fence_stype.* = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (createFence(device, @ptrCast(&fence_ci), null, &fence) != VK_SUCCESS) return false;

    return true;
}

// ── Public interface (placeholder) ──────────────────────────────
// TODO: implement fullVkLaunch mirroring gpu_driver.zig fullGpuLaunch

const fast_dec = @import("fast_lz_decoder.zig");

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
    _ = chunk_descs;
    _ = compressed_block;
    _ = dst_full;
    _ = dst_start_off;
    _ = decompressed_size;
    _ = num_groups;
    _ = chunks_per_group;
    _ = sub_chunk_cap;
    _ = io;
    // TODO: buffer creation, descriptor updates, command recording, dispatch
    return error.BadMode;
}
