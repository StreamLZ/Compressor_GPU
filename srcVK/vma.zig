//! Vulkan Memory Allocator (VMA) Zig binding (NEW; no CUDA counterpart).
//!
//! Wraps the VMA header vendored at srcVK/vma/vk_mem_alloc.h (copy of
//! third_party/vma/vk_mem_alloc.h, v3.1.0, AMD/GPUOpen MIT). The
//! implementation TU lives at srcVK/vma/vk_mem_alloc_impl.cpp and is
//! compiled into the streamlz_vk binary via build.zig.
//!
//! The codec never calls vkAllocateMemory directly: every device-side
//! allocation funnels through procs.malloc_device / procs.free_device in
//! srcVK/decode/vulkan_api.zig, which dispatch into VMA via the bindings
//! exposed here.
//!
//! Why hand-rolled bindings instead of @cImport(@cInclude(...)) of the
//! whole header: VMA's header is C++ with a C-compatible API surface;
//! Zig's translate-c chokes on the C++ template bodies even when only
//! the C entry points are exported. Hand-rolling the bare subset the
//! codec uses keeps the build hermetic - we control which C entry
//! points are visible and the Zig compile never touches the C++ side
//! at parse time. The fleshout still pulls VMA's header via @cImport
//! against srcVK/vma/vk_mem_alloc.h so the include path stays internal
//! to srcVK/, but the exposed surface is the hand-rolled extern-fn list
//! below.

const std = @import("std");

// VK adaptation: @cImport pulls Vulkan + VMA prototypes into one
// translation unit so the Zig side sees the same struct layouts the
// statically-linked C++ TU uses. Required because VMA's create-info
// structs reference VkBufferCreateInfo etc. transitively.
pub const c = @cImport({
    @cInclude("vma/vk_mem_alloc.h");
});

// ── Opaque handles ─────────────────────────────────────────────────────
// VMA defines these via VK_DEFINE_HANDLE, yielding a `struct
// VmaAllocator_T*` pointer. Modelled as ?*opaque so the size matches
// the C ABI and Zig refuses accidental dereferences.
pub const VmaAllocator = ?*opaque {};
pub const VmaAllocation = ?*opaque {};
pub const VmaPool = ?*opaque {};

// ── Bare Vulkan typedefs the codec touches (mirror vulkan.h) ──────────
// Re-declared here (instead of pulling vulkan.h in transitively) so the
// codec call sites read narrow types without dragging the full Vulkan
// header chain into every compile.
pub const VkResult = c_int;
pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkBuffer = u64; // VK_DEFINE_NON_DISPATCHABLE_HANDLE → 64-bit
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

// Buffer usage flags (subset the codec touches: storage + transfer src/dst).
// Values from vulkan_core.h - verbatim.
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlags = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlags = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: VkBufferUsageFlags = 0x00000010;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x00000020;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: VkBufferUsageFlags = 0x00000100;
pub const VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT: VkBufferUsageFlags = 0x00020000;

// Memory property flags - from vulkan_core.h.
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x00000004;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkMemoryPropertyFlags = 0x00000008;

// ── VMA enums ──────────────────────────────────────────────────
// Values verbatim from vk_mem_alloc.h v3.1.0.
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
/// vmaCreateAllocator. The 32-slot tail filler models the remaining
/// fields the VMA header declares; null entries cause VMA to dlopen
/// them through the supplied vkGetInstanceProcAddr / vkGetDeviceProcAddr.
pub const VmaVulkanFunctions = extern struct {
    vkGetInstanceProcAddr: ?*const anyopaque = null,
    vkGetDeviceProcAddr: ?*const anyopaque = null,
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
// Linked statically: srcVK/vma/vk_mem_alloc_impl.cpp pulls VMA's full
// implementation into the same binary as the Zig code below. The
// `extern "c"` linkage matches the C ABI prototypes in vk_mem_alloc.h.

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

// ── Thin Zig wrappers used by procs.malloc_device / procs.free_device ──
// These are the Zig-side helpers the procs.* implementations bound in
// module_loader.zig (Phase 3) will call. They keep the call sites in
// vulkan_api.zig narrow - one Zig-typed surface instead of N raw
// extern-fn invocations. Returns the underlying VkResult unchanged so
// the procs.* shim can fast-path on rc == 0 (matches CUDA's VK_SUCCESS_RC).

/// Initialise a VMA allocator on the supplied (instance, physical,
/// logical) triplet. Wraps vmaCreateAllocator with the buffer-device-
/// address flag enabled so callers can later query VAs for D2D paths.
///
/// VK adaptation: srcVK/vma/vk_mem_alloc_impl.cpp builds VMA with
/// VMA_STATIC_VULKAN_FUNCTIONS=0 + VMA_DYNAMIC_VULKAN_FUNCTIONS=1. As of
/// VMA v3.x (see vk_mem_alloc.h:13035-13040), that combination REQUIRES
/// the caller to pass a VmaVulkanFunctions struct populated with at
/// least vkGetInstanceProcAddr + vkGetDeviceProcAddr — VMA resolves
/// everything else off those two roots. Passing pVulkanFunctions = null
/// (the previous default here) made VMA dereference a null function
/// pointer inside ImportVulkanFunctions_Dynamic and segfault hard. The
/// caller supplies the two pointers it already resolved during the
/// instance/device bring-up; we cast them into the slot.
pub fn createAllocator(
    instance: VkInstance,
    phys: VkPhysicalDevice,
    device: VkDevice,
    vk_get_instance_proc_addr: ?*const anyopaque,
    vk_get_device_proc_addr: ?*const anyopaque,
) error{AllocatorCreateFailed}!VmaAllocator {
    var out: VmaAllocator = null;
    const vk_fns = VmaVulkanFunctions{
        .vkGetInstanceProcAddr = vk_get_instance_proc_addr,
        .vkGetDeviceProcAddr = vk_get_device_proc_addr,
    };
    const ci = VmaAllocatorCreateInfo{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = phys,
        .device = device,
        .pVulkanFunctions = &vk_fns,
        .instance = instance,
        // Vulkan API version baked at allocator creation time. 1.3 matches
        // the build.zig --target-env so VMA enables every feature path the
        // SPV blobs were compiled against.
        .vulkanApiVersion = (1 << 22) | (3 << 12),
    };
    const rc = vmaCreateAllocator(&ci, &out);
    if (rc != VK_SUCCESS) return error.AllocatorCreateFailed;
    return out;
}

/// Destroy a VMA allocator previously returned by createAllocator. A
/// null allocator is a no-op (matches VMA's documented contract) so
/// deinit paths can call this unconditionally.
pub fn destroyAllocator(allocator: VmaAllocator) void {
    if (allocator == null) return;
    vmaDestroyAllocator(allocator);
}

/// Allocate a VkBuffer-backed device-local block of `size` bytes with
/// usage flags `usage`. The returned (buffer, allocation) pair is freed
/// by destroyBuffer. The buffer is sized exactly; VMA may overallocate
/// internally to fit its block-size policy but the codec only ever sees
/// `size` bytes through procs.h2d / procs.d2h.
pub fn createBuffer(
    allocator: VmaAllocator,
    size: VkDeviceSize,
    usage: VkBufferUsageFlags,
) error{BufferCreateFailed}!struct { buffer: VkBuffer, allocation: VmaAllocation } {
    var buf: VkBuffer = 0;
    var alloc: VmaAllocation = null;
    const bci = VkBufferCreateInfo{
        .size = size,
        .usage = usage,
    };
    const aci = VmaAllocationCreateInfo{
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    const rc = vmaCreateBuffer(allocator, &bci, &aci, &buf, &alloc, null);
    if (rc != VK_SUCCESS) return error.BufferCreateFailed;
    return .{ .buffer = buf, .allocation = alloc };
}

/// Destroy a (VkBuffer, VmaAllocation) pair previously returned by
/// createBuffer. Null allocation is a no-op.
pub fn destroyBuffer(
    allocator: VmaAllocator,
    buffer: VkBuffer,
    allocation: VmaAllocation,
) void {
    if (allocation == null) return;
    vmaDestroyBuffer(allocator, buffer, allocation);
}

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
