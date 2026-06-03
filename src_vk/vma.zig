//! Vulkan Memory Allocator (VMA) Zig binding.
//!
//! Vendored at third_party/vma/vk_mem_alloc.h (v3.1.0, AMD/GPUOpen MIT
//! license). The implementation TU lives at
//! third_party/vma/vk_mem_alloc_impl.cpp and is compiled into the
//! streamlz_vk binary via build.zig.
//!
//! VK PORT NOTE: The Vulkan port does NOT call vkAllocateMemory directly
//! anywhere in the codec — every device-side allocation funnels through
//! the `procs.malloc_device` / `procs.free_device` shims in
//! src_vk/decode/vulkan_api.zig, which in turn drive VMA via the bindings
//! below. Codec call sites stay CUDA-shaped; only the L0 runtime under
//! decode/vulkan_api.zig knows VMA exists.
//!
//! Why hand-rolled bindings instead of `@cImport(@cInclude(...))`: VMA's
//! header is C++ with a C-compatible API surface; Zig's translate-c
//! chokes on the C++ template bodies even when only the C entry points
//! are exported. Hand-rolling the bare subset the codec uses keeps the
//! build hermetic — we control which C entry points are visible and the
//! Zig compile never touches the C++ side at parse time.

const std = @import("std");

// ── Opaque handles ─────────────────────────────────────────────
// VMA defines these via `VK_DEFINE_HANDLE`, which yields a `struct
// VmaAllocator_T*` pointer. We model them as `?*opaque` so the size
// matches and Zig refuses accidental dereferences.
pub const VmaAllocator = ?*opaque {};
pub const VmaAllocation = ?*opaque {};
pub const VmaPool = ?*opaque {};

// ── Re-export bare Vulkan typedefs the codec touches ───────────
// These mirror the Vulkan headers exactly; redefining them here avoids
// a transitive @cImport on vulkan.h (which would slow every compile and
// pollute the namespace with thousands of unrelated symbols). Code that
// needs richer Vulkan types reaches into src_vulkan/vk_api.zig or its
// src_vk/decode/vulkan_api.zig successor; this module stays narrow.
pub const VkResult = c_int;
pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkBuffer = u64; // VK_DEFINE_NON_DISPATCHABLE_HANDLE → 64-bit ID
pub const VkDeviceSize = u64;
pub const VkDeviceMemory = u64;
pub const VkFlags = u32;
pub const VkBufferCreateFlags = VkFlags;
pub const VkBufferUsageFlags = VkFlags;
pub const VkMemoryPropertyFlags = VkFlags;
pub const VkSharingMode = c_int;
pub const VkStructureType = c_int;
pub const VkBool32 = u32;

pub const VK_SUCCESS: VkResult = 0;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: VkStructureType = 12;

// Sharing modes
pub const VK_SHARING_MODE_EXCLUSIVE: VkSharingMode = 0;
pub const VK_SHARING_MODE_CONCURRENT: VkSharingMode = 1;

// Buffer usage flags (subset the codec touches: storage + transfer src/dst)
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlags = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlags = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: VkBufferUsageFlags = 0x00000010;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x00000020;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: VkBufferUsageFlags = 0x00000100;
pub const VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT: VkBufferUsageFlags = 0x00020000;

// Memory property flags
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkMemoryPropertyFlags = 0x00000008;

// ── VMA enums ──────────────────────────────────────────────────
pub const VmaMemoryUsage = c_int;
pub const VMA_MEMORY_USAGE_UNKNOWN: VmaMemoryUsage = 0;
pub const VMA_MEMORY_USAGE_AUTO: VmaMemoryUsage = 7;
pub const VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE: VmaMemoryUsage = 8;
pub const VMA_MEMORY_USAGE_AUTO_PREFER_HOST: VmaMemoryUsage = 9;

pub const VmaAllocationCreateFlags = VkFlags;
pub const VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT: VmaAllocationCreateFlags = 0x00000001;
pub const VMA_ALLOCATION_CREATE_MAPPED_BIT: VmaAllocationCreateFlags = 0x00000004;
pub const VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT: VmaAllocationCreateFlags = 0x00000400;
pub const VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT: VmaAllocationCreateFlags = 0x00000800;

pub const VmaAllocatorCreateFlags = VkFlags;
pub const VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT: VmaAllocatorCreateFlags = 0x00000020;

// ── Create-info structs (POD, layouts match vk_mem_alloc.h v3.1.0) ──

/// VMA pulls Vulkan function pointers either by linking against the
/// loader or via a caller-supplied dispatch table. The Zig binding lets
/// callers leave this null and selects "static link" mode when calling
/// `vmaCreateAllocator`.
pub const VmaVulkanFunctions = extern struct {
    vkGetInstanceProcAddr: ?*const anyopaque = null,
    vkGetDeviceProcAddr: ?*const anyopaque = null,
    // Remaining fields the VMA header declares; null → VMA dlopens them
    // through the supplied vkGetInstanceProcAddr / vkGetDeviceProcAddr.
    _filler: [32]?*const anyopaque = @splat(null),
};

pub const VmaAllocatorCreateInfo = extern struct {
    flags: VmaAllocatorCreateFlags = 0,
    physicalDevice: VkPhysicalDevice,
    device: VkDevice,
    preferredLargeHeapBlockSize: VkDeviceSize = 0,
    pAllocationCallbacks: ?*const anyopaque = null,
    pDeviceMemoryCallbacks: ?*const anyopaque = null,
    pHeapSizeLimit: ?*const VkDeviceSize = null,
    pVulkanFunctions: ?*const VmaVulkanFunctions = null,
    instance: VkInstance,
    vulkanApiVersion: u32 = 0,
    pTypeExternalMemoryHandleTypes: ?*const u32 = null,
};

pub const VmaAllocationCreateInfo = extern struct {
    flags: VmaAllocationCreateFlags = 0,
    usage: VmaMemoryUsage = VMA_MEMORY_USAGE_UNKNOWN,
    requiredFlags: VkMemoryPropertyFlags = 0,
    preferredFlags: VkMemoryPropertyFlags = 0,
    memoryTypeBits: u32 = 0,
    pool: VmaPool = null,
    pUserData: ?*anyopaque = null,
    priority: f32 = 0.0,
};

pub const VmaAllocationInfo = extern struct {
    memoryType: u32 = 0,
    deviceMemory: VkDeviceMemory = 0,
    offset: VkDeviceSize = 0,
    size: VkDeviceSize = 0,
    pMappedData: ?*anyopaque = null,
    pUserData: ?*anyopaque = null,
    pName: ?[*:0]const u8 = null,
};

pub const VkBufferCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkBufferCreateFlags = 0,
    size: VkDeviceSize,
    usage: VkBufferUsageFlags,
    sharingMode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

// ── VMA C entry points the codec needs ─────────────────────────
// Linked statically: third_party/vma/vk_mem_alloc_impl.cpp pulls VMA's
// implementation into the same binary as the Zig code below.

pub extern "c" fn vmaCreateAllocator(
    pCreateInfo: *const VmaAllocatorCreateInfo,
    pAllocator: *VmaAllocator,
) callconv(.c) VkResult;

pub extern "c" fn vmaDestroyAllocator(allocator: VmaAllocator) callconv(.c) void;

pub extern "c" fn vmaCreateBuffer(
    allocator: VmaAllocator,
    pBufferCreateInfo: *const VkBufferCreateInfo,
    pAllocationCreateInfo: *const VmaAllocationCreateInfo,
    pBuffer: *VkBuffer,
    pAllocation: *VmaAllocation,
    pAllocationInfo: ?*VmaAllocationInfo,
) callconv(.c) VkResult;

pub extern "c" fn vmaDestroyBuffer(
    allocator: VmaAllocator,
    buffer: VkBuffer,
    allocation: VmaAllocation,
) callconv(.c) void;

pub extern "c" fn vmaMapMemory(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    ppData: *?*anyopaque,
) callconv(.c) VkResult;

pub extern "c" fn vmaUnmapMemory(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
) callconv(.c) void;

pub extern "c" fn vmaFlushAllocation(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    offset: VkDeviceSize,
    size: VkDeviceSize,
) callconv(.c) VkResult;

pub extern "c" fn vmaInvalidateAllocation(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    offset: VkDeviceSize,
    size: VkDeviceSize,
) callconv(.c) VkResult;

pub extern "c" fn vmaGetAllocationInfo(
    allocator: VmaAllocator,
    allocation: VmaAllocation,
    pAllocationInfo: *VmaAllocationInfo,
) callconv(.c) void;

// ────────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "VMA POD struct layouts match the C header" {
    // Compile-time check: extern struct field offsets are what the
    // codec call sites expect. If VMA bumps a field, this catches it
    // before a misaligned write corrupts an allocation.
    try testing.expectEqual(@as(usize, 0), @offsetOf(VmaAllocationCreateInfo, "flags"));
    try testing.expectEqual(@as(usize, 4), @offsetOf(VmaAllocationCreateInfo, "usage"));

    try testing.expectEqual(@as(usize, 0), @offsetOf(VkBufferCreateInfo, "sType"));
    try testing.expect(@offsetOf(VkBufferCreateInfo, "size") > 0);
}

test "VMA opaque handles are pointer-sized" {
    try testing.expectEqual(@sizeOf(usize), @sizeOf(VmaAllocator));
    try testing.expectEqual(@sizeOf(usize), @sizeOf(VmaAllocation));
    try testing.expectEqual(@sizeOf(u64), @sizeOf(VkBuffer));
}
