//! Vulkan loader API + Win32 surface used by the GPU-encode/decode Vulkan port.
//!
//! Mirrors src/decode/cuda_api.zig: a single FFI module that bundles the
//! opaque handle typedefs, function-pointer slots, dlopen state, and the
//! minimal enum/constant subset everything else under src_vulkan/ leans on.
//!
//! Function-pointer slots are `pub var` (not `pub const`) because `init()`
//! fills the bootstrap slot after `LoadLibraryA("vulkan-1.dll")`, and the
//! instance/device-level slots are populated later by `instance.zig` and
//! `device.zig` via `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`.

const std = @import("std");

pub const win32 = struct {
    pub extern "kernel32" fn LoadLibraryA(name: [*:0]const u8) callconv(.c) ?*anyopaque;
    pub extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.c) ?*anyopaque;
};

// ── Opaque handle typedefs ───────────────────────────────────────
// Vulkan dispatchable handles are pointers to opaque driver-owned structs;
// non-dispatchable handles are 64-bit IDs on every supported platform. We
// only model dispatchable handles here (everything M1 touches is one of
// these — Instance/PhysicalDevice/Device/Queue). Non-dispatchable handles
// (VkBuffer, VkImage, VkDeviceMemory, ...) arrive in later milestones.
pub const VkInstance = ?*opaque {};
pub const VkPhysicalDevice = ?*opaque {};
pub const VkDevice = ?*opaque {};
pub const VkQueue = ?*opaque {};

// VkPipelineCache is non-dispatchable per the spec — a 64-bit driver-owned
// ID, not a pointer to an opaque struct. We model it as an opaque-pointer
// alias anyway because (a) Vulkan headers do the same trick under the
// `VK_DEFINE_NON_DISPATCHABLE_HANDLE` macro, and (b) treating it as a
// typed pointer-shaped value keeps the function-pointer signatures clean
// (the Vulkan ABI passes it by value either way; ?*opaque has the same
// pointer-sized representation as a 64-bit handle on x86_64).
pub const VkPipelineCache = ?*opaque {};

// ── M8a non-dispatchable handles ─────────────────────────────────
// Same `?*opaque` modeling rationale as VkPipelineCache above — these are
// 64-bit IDs in the C ABI but pointer-shaped on every host we target, and
// keeping them typed lets the function-pointer signatures stay precise.
// Command-buffer is the one exception: it IS dispatchable per the spec
// (the first slot of the driver-owned struct is a loader-installed jump
// table), but we never invoke loader thunks via it — vkCmd* take it by
// value and the driver handles the dispatch internally — so the opaque-
// pointer typedef is fine for us either way.
pub const VkPipeline = ?*opaque {};
pub const VkDescriptorSet = ?*opaque {};
pub const VkPipelineLayout = ?*opaque {};
pub const VkCommandPool = ?*opaque {};
pub const VkCommandBuffer = ?*opaque {};
pub const VkFence = ?*opaque {};
pub const VkQueryPool = ?*opaque {};
pub const VkSemaphore = ?*opaque {};

// Function-pointer cookie returned by vkGetInstanceProcAddr / vkGetDeviceProcAddr.
// Cast to the concrete `Fn*` signature at the call site.
pub const PFN_vkVoidFunction = ?*const fn () callconv(.c) void;

// ── VkResult subset ──────────────────────────────────────────────
// Full enum is large; M1 only needs success + the handful of failure codes
// the bootstrap path can observe. Extended as later milestones surface more.
pub const VkResult = c_int;
pub const VK_SUCCESS: VkResult = 0;
pub const VK_NOT_READY: VkResult = 1;
pub const VK_TIMEOUT: VkResult = 2;
pub const VK_EVENT_SET: VkResult = 3;
pub const VK_EVENT_RESET: VkResult = 4;
pub const VK_INCOMPLETE: VkResult = 5;
pub const VK_ERROR_OUT_OF_HOST_MEMORY: VkResult = -1;
pub const VK_ERROR_OUT_OF_DEVICE_MEMORY: VkResult = -2;
pub const VK_ERROR_INITIALIZATION_FAILED: VkResult = -3;
pub const VK_ERROR_DEVICE_LOST: VkResult = -4;
pub const VK_ERROR_LAYER_NOT_PRESENT: VkResult = -6;
pub const VK_ERROR_EXTENSION_NOT_PRESENT: VkResult = -7;
pub const VK_ERROR_FEATURE_NOT_PRESENT: VkResult = -8;
pub const VK_ERROR_INCOMPATIBLE_DRIVER: VkResult = -9;

// ── VkStructureType subset ───────────────────────────────────────
// Spec enum is open-ended; only the values M1's create-info structs need.
pub const VkStructureType = c_int;
pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: VkStructureType = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: VkStructureType = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: VkStructureType = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: VkStructureType = 3;
// Features/Properties pNext chain — values from the Vulkan 1.3 spec.
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: VkStructureType = 1000059000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: VkStructureType = 1000059001;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES: VkStructureType = 1000094000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES: VkStructureType = 1000225000;
// Correct sType values per vulkan_core.h (1.3 core promotion):
//   FEATURES        = 1000225002
//   PROPERTIES      = 1000225000
//   REQUIRED_CREATE = 1000225001
// Earlier revisions had FEATURES and the create-info struct swapped, which
// Intel's driver tolerated silently but NVIDIA's validator (correctly)
// rejects in the VkDeviceCreateInfo pNext chain.
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES: VkStructureType = 1000225002;
// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo — chained into a
// VkPipelineShaderStageCreateInfo to force a specific subgroup size at
// pipeline creation. Needed because Intel UHD silently defaults to
// subgroupSize=16 (= SIMD8 pairs) instead of the device-reported max of
// 32, which breaks every shader that assumes WARP_SIZE=32. Promoted in
// Vulkan 1.3 from VK_EXT_subgroup_size_control.
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO: VkStructureType = 1000225001;
// Correct sType values per vulkan_core.h:
//   VULKAN_1_1_FEATURES = 49 (intentionally not defined here — we don't
//                             use the v1.1 omnibus feature struct)
//   VULKAN_1_2_FEATURES = 51 (was incorrectly 49, which is _1_1_FEATURES;
//                             Intel tolerated, NVIDIA validator rejects)
//   VULKAN_1_3_FEATURES = 53
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES: VkStructureType = 51;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES: VkStructureType = 53;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES: VkStructureType = 1000175000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES: VkStructureType = 1000177000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES: VkStructureType = 1000257000;
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES: VkStructureType = 1000207000;
// Correct sType per vulkan_core.h: 1000314007 (NOT 1000314000, which is
// the unrelated VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 head — wrong sType
// caused NVIDIA's validator to reject the pNext chain).
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES: VkStructureType = 1000314007;
// VkPipelineCacheCreateInfo — core in Vulkan 1.0, sType value 17.
pub const VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO: VkStructureType = 17;

// ── M8a sType values (command/sync/query bring-up) ───────────────
// All Vulkan 1.0 core; constants pulled from vulkan_core.h.
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: VkStructureType = 4;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: VkStructureType = 8;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: VkStructureType = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: VkStructureType = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: VkStructureType = 42;
pub const VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO: VkStructureType = 11;

// ── Queue flag bits ──────────────────────────────────────────────
pub const VkQueueFlags = u32;
pub const VK_QUEUE_GRAPHICS_BIT: VkQueueFlags = 0x1;
pub const VK_QUEUE_COMPUTE_BIT: VkQueueFlags = 0x2;
pub const VK_QUEUE_TRANSFER_BIT: VkQueueFlags = 0x4;
pub const VK_QUEUE_SPARSE_BINDING_BIT: VkQueueFlags = 0x8;

// ── API version macros ───────────────────────────────────────────
// VK_MAKE_API_VERSION(variant, major, minor, patch).
pub fn VK_MAKE_API_VERSION(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}
pub const VK_API_VERSION_1_0: u32 = VK_MAKE_API_VERSION(0, 1, 0, 0);
pub const VK_API_VERSION_1_2: u32 = VK_MAKE_API_VERSION(0, 1, 2, 0);
pub const VK_API_VERSION_1_3: u32 = VK_MAKE_API_VERSION(0, 1, 3, 0);

pub fn VK_API_VERSION_MAJOR(v: u32) u32 {
    return (v >> 22) & 0x7f;
}
pub fn VK_API_VERSION_MINOR(v: u32) u32 {
    return (v >> 12) & 0x3ff;
}
pub fn VK_API_VERSION_PATCH(v: u32) u32 {
    return v & 0xfff;
}

// ── Public structs the M1 surface touches ────────────────────────
// All fields kept in spec order; pNext defaults to null because M1 never
// chains feature structs (later milestones will add Vulkan-1.2/1.3 feature
// chains via @ptrCast to `*const anyopaque`).

pub const VkAllocationCallbacks = opaque {};

pub const VkApplicationInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = 0,
};

pub const VkInstanceCreateFlags = u32;
pub const VkInstanceCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkInstanceCreateFlags = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

// VkPhysicalDeviceProperties has nested sub-structs; M1 only reads the
// scalar prefix (apiVersion, deviceName) but the binary layout MUST match
// the C ABI exactly — the driver writes the entire 824-byte payload, so a
// short struct overruns the host buffer (caught by the stack protector
// as __chk_fail at function exit).
//
// Spec layout on 64-bit:
//   u32 apiVersion / driverVersion / vendorID / deviceID    20
//   VkPhysicalDeviceType (enum, 4 bytes)                     4 → 24
//   char deviceName[256]                                   256 → 280
//   uint8_t pipelineCacheUUID[16]                           16 → 296
//   (4 bytes implicit pad — limits has 8-byte members)       4 → 300 → 304 aligned
//   VkPhysicalDeviceLimits                                 504 → 808
//   VkPhysicalDeviceSparseProperties (5 × u32)              20 → 828 → 824?
//   trailing align-to-8 pad                                  …
//
// We force the post-UUID region to live on an 8-byte boundary via
// `align(8)` and size the opaque tail at 528 bytes (504 + 20 + 4 trailing
// pad) so total sizeof matches the 824 the driver writes. Drift-detection
// comptime assert below catches the day the spec extends limits/sparse.
pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
pub const VK_UUID_SIZE: usize = 16;
pub const VkPhysicalDeviceType = c_int;

const PROPS_TAIL_BYTES: usize = 528; // VkPhysicalDeviceLimits + sparse + pad

pub const VkPhysicalDeviceProperties = extern struct {
    apiVersion: u32 = 0,
    driverVersion: u32 = 0,
    vendorID: u32 = 0,
    deviceID: u32 = 0,
    deviceType: VkPhysicalDeviceType = 0,
    deviceName: [VK_MAX_PHYSICAL_DEVICE_NAME_SIZE]u8 = @splat(0),
    pipelineCacheUUID: [VK_UUID_SIZE]u8 = @splat(0),
    // `align(8)` forces a 4-byte pad between pipelineCacheUUID (ends at
    // offset 296) and this field (8-byte aligned at 304), matching the
    // alignment VkPhysicalDeviceLimits would impose via its VkDeviceSize
    // (u64) members in the real C struct.
    limits_sparse_pad_opaque: [PROPS_TAIL_BYTES]u8 align(8) = @splat(0),
};

comptime {
    // If this fires, the spec has grown VkPhysicalDeviceLimits / sparse;
    // bump PROPS_TAIL_BYTES so the host-side buffer matches the driver
    // write size and the stack protector stops tripping.
    std.debug.assert(@sizeOf(VkPhysicalDeviceProperties) == 824);
}

// Mirror the spec layout: u32 fields followed by the opaque min-image-
// transfer-granularity (VkExtent3D, 12 bytes). Total 24 bytes.
pub const VkQueueFamilyProperties = extern struct {
    queueFlags: VkQueueFlags = 0,
    queueCount: u32 = 0,
    timestampValidBits: u32 = 0,
    minImageTransferGranularity: [3]u32 = @splat(0),
};

pub const VkDeviceQueueCreateFlags = u32;
pub const VkDeviceQueueCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDeviceQueueCreateFlags = 0,
    queueFamilyIndex: u32 = 0,
    queueCount: u32 = 0,
    pQueuePriorities: ?[*]const f32 = null,
};

pub const VkDeviceCreateFlags = u32;
pub const VkDeviceCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDeviceCreateFlags = 0,
    queueCreateInfoCount: u32 = 0,
    pQueueCreateInfos: ?[*]const VkDeviceQueueCreateInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
    pEnabledFeatures: ?*const anyopaque = null, // VkPhysicalDeviceFeatures*, not modeled at M1
};

// ── M7: VkPipelineCacheCreateInfo ────────────────────────────────
// Used by vkCreatePipelineCache to seed an in-memory cache from a previously
// serialized blob (initialDataSize > 0 + initialData != null) or to create
// an empty cache (initialDataSize == 0). The driver validates the blob's
// 32-byte header — vendorID/deviceID/pipelineCacheUUID mismatch → silently
// returns VK_SUCCESS but discards the data (callers must NOT rely on
// rejection signaling here; persistence is best-effort).
pub const VkPipelineCacheCreateFlags = u32;
pub const VkPipelineCacheCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineCacheCreateFlags = 0,
    initialDataSize: usize = 0,
    pInitialData: ?*const anyopaque = null,
};

// ── M2: feature/property chain structs (pNext-chained query) ─────
// The Vulkan 1.1+ "Features2"/"Properties2" entry points let us chain
// per-extension/per-version structs through pNext. For M2 we collect:
//   • VkPhysicalDeviceProperties2  ← scalar props + Subgroup + SubgroupSizeControl props
//   • VkPhysicalDeviceFeatures2    ← Vulkan12Features + Vulkan13Features +
//                                    ShaderSubgroupExtendedTypes
//
// All chain heads carry sType + pNext. We model only the fields the tier
// classifier reads; trailing fields are kept as zero-init opaque tails so
// the binary layout matches what the driver writes/reads (the same
// stack-protector trap that bit VkPhysicalDeviceProperties applies here).

// VkPhysicalDeviceFeatures: 55 VkBool32 fields = 220 bytes. Opaque blob.
// We never read individual feature bits from this struct (we read Vulkan12-
// Features and ShaderSubgroupExtendedTypes via pNext chain instead).
pub const VkBool32 = u32;
pub const VK_FALSE: VkBool32 = 0;
pub const VK_TRUE: VkBool32 = 1;

pub const VkPhysicalDeviceFeatures = extern struct {
    // 55 × u32 = 220 bytes. We poke 'shaderInt64' (index 43) below by name.
    robustBufferAccess: VkBool32 = 0,
    fullDrawIndexUint32: VkBool32 = 0,
    imageCubeArray: VkBool32 = 0,
    independentBlend: VkBool32 = 0,
    geometryShader: VkBool32 = 0,
    tessellationShader: VkBool32 = 0,
    sampleRateShading: VkBool32 = 0,
    dualSrcBlend: VkBool32 = 0,
    logicOp: VkBool32 = 0,
    multiDrawIndirect: VkBool32 = 0,
    drawIndirectFirstInstance: VkBool32 = 0,
    depthClamp: VkBool32 = 0,
    depthBiasClamp: VkBool32 = 0,
    fillModeNonSolid: VkBool32 = 0,
    depthBounds: VkBool32 = 0,
    wideLines: VkBool32 = 0,
    largePoints: VkBool32 = 0,
    alphaToOne: VkBool32 = 0,
    multiViewport: VkBool32 = 0,
    samplerAnisotropy: VkBool32 = 0,
    textureCompressionETC2: VkBool32 = 0,
    textureCompressionASTC_LDR: VkBool32 = 0,
    textureCompressionBC: VkBool32 = 0,
    occlusionQueryPrecise: VkBool32 = 0,
    pipelineStatisticsQuery: VkBool32 = 0,
    vertexPipelineStoresAndAtomics: VkBool32 = 0,
    fragmentStoresAndAtomics: VkBool32 = 0,
    shaderTessellationAndGeometryPointSize: VkBool32 = 0,
    shaderImageGatherExtended: VkBool32 = 0,
    shaderStorageImageExtendedFormats: VkBool32 = 0,
    shaderStorageImageMultisample: VkBool32 = 0,
    shaderStorageImageReadWithoutFormat: VkBool32 = 0,
    shaderStorageImageWriteWithoutFormat: VkBool32 = 0,
    shaderUniformBufferArrayDynamicIndexing: VkBool32 = 0,
    shaderSampledImageArrayDynamicIndexing: VkBool32 = 0,
    shaderStorageBufferArrayDynamicIndexing: VkBool32 = 0,
    shaderStorageImageArrayDynamicIndexing: VkBool32 = 0,
    shaderClipDistance: VkBool32 = 0,
    shaderCullDistance: VkBool32 = 0,
    shaderFloat64: VkBool32 = 0,
    shaderInt64: VkBool32 = 0,
    shaderInt16: VkBool32 = 0,
    shaderResourceResidency: VkBool32 = 0,
    shaderResourceMinLod: VkBool32 = 0,
    sparseBinding: VkBool32 = 0,
    sparseResidencyBuffer: VkBool32 = 0,
    sparseResidencyImage2D: VkBool32 = 0,
    sparseResidencyImage3D: VkBool32 = 0,
    sparseResidency2Samples: VkBool32 = 0,
    sparseResidency4Samples: VkBool32 = 0,
    sparseResidency8Samples: VkBool32 = 0,
    sparseResidency16Samples: VkBool32 = 0,
    sparseResidencyAliased: VkBool32 = 0,
    variableMultisampleRate: VkBool32 = 0,
    inheritedQueries: VkBool32 = 0,
};

comptime {
    std.debug.assert(@sizeOf(VkPhysicalDeviceFeatures) == 55 * 4);
}

// Spec-size sanity checks for the M2 pNext-chained structs. If any of
// these fire, our layout drifted from the driver's and the read-back
// values will be garbage (silently — Vulkan does not validate struct
// sizes, just walks pNext blindly). The compile-error message names the
// struct so you don't have to count fields when one fires.
// Spec-size sanity checks for the M2 pNext-chained structs. Sizes account
// for the trailing pad each struct picks up to round to the alignof of
// its largest field (the pNext pointer = 8 bytes on x86_64). Per the spec
// the C structs use natural alignment too, so the values below match what
// vulkan_core.h yields under MSVC/clang on 64-bit Windows.
comptime {
    // header (16) + 55 u32 (220) → rounded up to 8 = 240.
    std.debug.assert(@sizeOf(VkPhysicalDeviceFeatures2) == 240);
    // header (16) + 824 (props) → 840 (already 8-aligned).
    std.debug.assert(@sizeOf(VkPhysicalDeviceProperties2) == 840);
    // header (16) + u32(sgsize) + u32(flags) + u32(flags) + u32(bool) = 32.
    std.debug.assert(@sizeOf(VkPhysicalDeviceSubgroupProperties) == 32);
    // header (16) + 4 × u32 = 32.
    std.debug.assert(@sizeOf(VkPhysicalDeviceSubgroupSizeControlProperties) == 32);
    // header (16) + 2 × VkBool32 = 24.
    std.debug.assert(@sizeOf(VkPhysicalDeviceSubgroupSizeControlFeatures) == 24);
    // header (16) + 1 × VkBool32 + 4 trailing pad = 24.
    std.debug.assert(@sizeOf(VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures) == 24);
    // header (16) + 47 × u32 (188) → rounded up to 8 = 208.
    std.debug.assert(@sizeOf(VkPhysicalDeviceVulkan12Features) == 208);
    // header (16) + 15 × u32 (60) → rounded up to 8 = 80.
    std.debug.assert(@sizeOf(VkPhysicalDeviceVulkan13Features) == 80);
}

// VkPhysicalDeviceFeatures2 — wrapper that carries pNext.
pub const VkPhysicalDeviceFeatures2 = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    pNext: ?*anyopaque = null,
    features: VkPhysicalDeviceFeatures = .{},
};

// VkPhysicalDeviceProperties2 — wrapper that carries pNext.
pub const VkPhysicalDeviceProperties2 = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
    pNext: ?*anyopaque = null,
    properties: VkPhysicalDeviceProperties = .{},
};

// VkShaderStageFlags / VkSubgroupFeatureFlags — bitmasks.
pub const VkShaderStageFlags = u32;
pub const VK_SHADER_STAGE_COMPUTE_BIT: VkShaderStageFlags = 0x20;

pub const VkSubgroupFeatureFlags = u32;
pub const VK_SUBGROUP_FEATURE_BASIC_BIT: VkSubgroupFeatureFlags = 0x1;
pub const VK_SUBGROUP_FEATURE_VOTE_BIT: VkSubgroupFeatureFlags = 0x2;
pub const VK_SUBGROUP_FEATURE_ARITHMETIC_BIT: VkSubgroupFeatureFlags = 0x4;
pub const VK_SUBGROUP_FEATURE_BALLOT_BIT: VkSubgroupFeatureFlags = 0x8;
pub const VK_SUBGROUP_FEATURE_SHUFFLE_BIT: VkSubgroupFeatureFlags = 0x10;
pub const VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT: VkSubgroupFeatureFlags = 0x20;
pub const VK_SUBGROUP_FEATURE_CLUSTERED_BIT: VkSubgroupFeatureFlags = 0x40;
pub const VK_SUBGROUP_FEATURE_QUAD_BIT: VkSubgroupFeatureFlags = 0x80;
pub const VK_SUBGROUP_FEATURE_PARTITIONED_BIT_NV: VkSubgroupFeatureFlags = 0x100;

// VkPhysicalDeviceSubgroupProperties — chained via VkPhysicalDeviceProperties2.
pub const VkPhysicalDeviceSubgroupProperties = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    pNext: ?*anyopaque = null,
    subgroupSize: u32 = 0,
    supportedStages: VkShaderStageFlags = 0,
    supportedOperations: VkSubgroupFeatureFlags = 0,
    quadOperationsInAllStages: VkBool32 = 0,
};

// VkPhysicalDeviceSubgroupSizeControlProperties — chained via Properties2.
// (Promoted from EXT in Vulkan 1.3; same memory layout.)
pub const VkPhysicalDeviceSubgroupSizeControlProperties = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
    pNext: ?*anyopaque = null,
    minSubgroupSize: u32 = 0,
    maxSubgroupSize: u32 = 0,
    maxComputeWorkgroupSubgroups: u32 = 0,
    requiredSubgroupSizeStages: VkShaderStageFlags = 0,
};

// VkPhysicalDeviceSubgroupSizeControlFeatures — chained via Features2.
pub const VkPhysicalDeviceSubgroupSizeControlFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
    pNext: ?*anyopaque = null,
    subgroupSizeControl: VkBool32 = 0,
    computeFullSubgroups: VkBool32 = 0,
};

// VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures (Vulkan 1.2 core).
pub const VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
    pNext: ?*anyopaque = null,
    shaderSubgroupExtendedTypes: VkBool32 = 0,
};

// VkPhysicalDeviceBufferDeviceAddressFeatures (Vulkan 1.2 core, also from
// VK_KHR_buffer_device_address). Chained as a sanity-check fallback when
// the v12 omnibus reports false (some drivers under-fill the v12 struct
// and only populate the per-extension version).
pub const VkPhysicalDeviceBufferDeviceAddressFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    pNext: ?*anyopaque = null,
    bufferDeviceAddress: VkBool32 = 0,
    bufferDeviceAddressCaptureReplay: VkBool32 = 0,
    bufferDeviceAddressMultiDevice: VkBool32 = 0,
};

// VkPhysicalDeviceTimelineSemaphoreFeatures (Vulkan 1.2 core, also from
// VK_KHR_timeline_semaphore). Same fallback rationale as BDA above.
pub const VkPhysicalDeviceTimelineSemaphoreFeatures = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    pNext: ?*anyopaque = null,
    timelineSemaphore: VkBool32 = 0,
};

// VkPhysicalDeviceSynchronization2Features (Vulkan 1.3 core, also from
// VK_KHR_synchronization2). Same fallback rationale as BDA above — the
// v13 omnibus and this struct should report identically when present,
// but some drivers under-fill one of them.
pub const VkPhysicalDeviceSynchronization2Features = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
    pNext: ?*anyopaque = null,
    synchronization2: VkBool32 = 0,
};

// VkPhysicalDeviceVulkan12Features — the omnibus 1.2 feature struct.
// 47 VkBool32 fields after sType/pNext = 188 bytes payload, +16 header = 204.
pub const VkPhysicalDeviceVulkan12Features = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    pNext: ?*anyopaque = null,
    samplerMirrorClampToEdge: VkBool32 = 0,
    drawIndirectCount: VkBool32 = 0,
    storageBuffer8BitAccess: VkBool32 = 0,
    uniformAndStorageBuffer8BitAccess: VkBool32 = 0,
    storagePushConstant8: VkBool32 = 0,
    shaderBufferInt64Atomics: VkBool32 = 0,
    shaderSharedInt64Atomics: VkBool32 = 0,
    shaderFloat16: VkBool32 = 0,
    shaderInt8: VkBool32 = 0,
    descriptorIndexing: VkBool32 = 0,
    shaderInputAttachmentArrayDynamicIndexing: VkBool32 = 0,
    shaderUniformTexelBufferArrayDynamicIndexing: VkBool32 = 0,
    shaderStorageTexelBufferArrayDynamicIndexing: VkBool32 = 0,
    shaderUniformBufferArrayNonUniformIndexing: VkBool32 = 0,
    shaderSampledImageArrayNonUniformIndexing: VkBool32 = 0,
    shaderStorageBufferArrayNonUniformIndexing: VkBool32 = 0,
    shaderStorageImageArrayNonUniformIndexing: VkBool32 = 0,
    shaderInputAttachmentArrayNonUniformIndexing: VkBool32 = 0,
    shaderUniformTexelBufferArrayNonUniformIndexing: VkBool32 = 0,
    shaderStorageTexelBufferArrayNonUniformIndexing: VkBool32 = 0,
    descriptorBindingUniformBufferUpdateAfterBind: VkBool32 = 0,
    descriptorBindingSampledImageUpdateAfterBind: VkBool32 = 0,
    descriptorBindingStorageImageUpdateAfterBind: VkBool32 = 0,
    descriptorBindingStorageBufferUpdateAfterBind: VkBool32 = 0,
    descriptorBindingUniformTexelBufferUpdateAfterBind: VkBool32 = 0,
    descriptorBindingStorageTexelBufferUpdateAfterBind: VkBool32 = 0,
    descriptorBindingUpdateUnusedWhilePending: VkBool32 = 0,
    descriptorBindingPartiallyBound: VkBool32 = 0,
    descriptorBindingVariableDescriptorCount: VkBool32 = 0,
    runtimeDescriptorArray: VkBool32 = 0,
    samplerFilterMinmax: VkBool32 = 0,
    scalarBlockLayout: VkBool32 = 0,
    imagelessFramebuffer: VkBool32 = 0,
    uniformBufferStandardLayout: VkBool32 = 0,
    shaderSubgroupExtendedTypes: VkBool32 = 0,
    separateDepthStencilLayouts: VkBool32 = 0,
    hostQueryReset: VkBool32 = 0,
    timelineSemaphore: VkBool32 = 0,
    bufferDeviceAddress: VkBool32 = 0,
    bufferDeviceAddressCaptureReplay: VkBool32 = 0,
    bufferDeviceAddressMultiDevice: VkBool32 = 0,
    vulkanMemoryModel: VkBool32 = 0,
    vulkanMemoryModelDeviceScope: VkBool32 = 0,
    vulkanMemoryModelAvailabilityVisibilityChains: VkBool32 = 0,
    shaderOutputViewportIndex: VkBool32 = 0,
    shaderOutputLayer: VkBool32 = 0,
    subgroupBroadcastDynamicId: VkBool32 = 0,
};

// VkPhysicalDeviceVulkan13Features — only the fields we read; the rest are
// captured as a fixed opaque tail so the driver's writes don't overflow.
// Spec layout: 16 VkBool32 fields after sType/pNext = 64 bytes payload.
pub const VkPhysicalDeviceVulkan13Features = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    pNext: ?*anyopaque = null,
    robustImageAccess: VkBool32 = 0,
    inlineUniformBlock: VkBool32 = 0,
    descriptorBindingInlineUniformBlockUpdateAfterBind: VkBool32 = 0,
    pipelineCreationCacheControl: VkBool32 = 0,
    privateData: VkBool32 = 0,
    shaderDemoteToHelperInvocation: VkBool32 = 0,
    shaderTerminateInvocation: VkBool32 = 0,
    subgroupSizeControl: VkBool32 = 0,
    computeFullSubgroups: VkBool32 = 0,
    synchronization2: VkBool32 = 0,
    textureCompressionASTC_HDR: VkBool32 = 0,
    shaderZeroInitializeWorkgroupMemory: VkBool32 = 0,
    dynamicRendering: VkBool32 = 0,
    shaderIntegerDotProduct: VkBool32 = 0,
    maintenance4: VkBool32 = 0,
};

// VkExtensionProperties — for vkEnumerateDeviceExtensionProperties.
pub const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;
pub const VkExtensionProperties = extern struct {
    extensionName: [VK_MAX_EXTENSION_NAME_SIZE]u8 = @splat(0),
    specVersion: u32 = 0,
};

// ── M8a: command/sync/query/dispatch surface ─────────────────────
// Bare-minimum chassis for a single one-shot dispatch with a fence and a
// pair of timestamp queries flanking it. Production-grade extensions
// (sync2 timestamp writes, 68-slot query pool, shape-keyed cmd cache)
// arrive in M8b/M8c — this is the dispatch+timing substrate M5 needs.

// VkCommandPoolCreateFlags bits. RESET_COMMAND_BUFFER_BIT lets us call
// vkResetCommandBuffer on an individual buffer instead of needing a full
// pool reset; TRANSIENT_BIT is a driver-side hint that buffers in this
// pool are short-lived (M8a's "allocate one, submit, free" lifetime).
pub const VkCommandPoolCreateFlags = u32;
pub const VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: VkCommandPoolCreateFlags = 0x1;
pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: VkCommandPoolCreateFlags = 0x2;

pub const VkCommandBufferLevel = c_int;
pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: VkCommandBufferLevel = 0;
pub const VK_COMMAND_BUFFER_LEVEL_SECONDARY: VkCommandBufferLevel = 1;

// VkCommandBufferUsageFlags — ONE_TIME_SUBMIT_BIT promises we won't
// resubmit this buffer; drivers can pick a cheaper internal layout.
// M8a uses it because we record + submit + free in the same call.
pub const VkCommandBufferUsageFlags = u32;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: VkCommandBufferUsageFlags = 0x1;

pub const VkCommandBufferResetFlags = u32;

pub const VkFenceCreateFlags = u32;
pub const VK_FENCE_CREATE_SIGNALED_BIT: VkFenceCreateFlags = 0x1;

pub const VkQueryType = c_int;
pub const VK_QUERY_TYPE_TIMESTAMP: VkQueryType = 2;

// VkQueryResultFlags — WAIT_BIT blocks until queries are ready (we
// already vkWaitForFences, so the result is guaranteed ready, but we
// still pass _64_BIT so the driver writes u64 not u32 per slot).
pub const VkQueryResultFlags = u32;
pub const VK_QUERY_RESULT_64_BIT: VkQueryResultFlags = 0x1;
pub const VK_QUERY_RESULT_WAIT_BIT: VkQueryResultFlags = 0x2;

// VkPipelineStageFlags — for vkCmdWriteTimestamp's pipeline-stage arg.
// TOP_OF_PIPE for the "before dispatch" sample, BOTTOM_OF_PIPE for the
// "after dispatch" sample (locked decision #12 per milestones §M8c).
// COMPUTE_SHADER_BIT included for completeness; not used at M8a.
pub const VkPipelineStageFlags = u32;
pub const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: VkPipelineStageFlags = 0x1;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: VkPipelineStageFlags = 0x800;
pub const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: VkPipelineStageFlags = 0x2000;

pub const VkPipelineBindPoint = c_int;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: VkPipelineBindPoint = 1;

// Wait timeout for vkWaitForFences. The original 2 s budget was sized
// for M8a microbenches (<1 ms per dispatch) and tripped on production
// 200 MB silesia encodes (~3-30 s on an Intel iGPU). Raised to 60 s,
// which still catches genuinely hung GPUs in CI within a reasonable
// wall-clock while comfortably covering full-corpus L1 encode/decode
// at scale. If you ever push past this on real hardware, the right fix
// is per-dispatch sizing — not bumping this further.
pub const VK_M8A_FENCE_WAIT_NS: u64 = 60 * 1_000_000_000;

pub const VkCommandPoolCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkCommandPoolCreateFlags = 0,
    queueFamilyIndex: u32 = 0,
};

pub const VkCommandBufferAllocateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    commandPool: VkCommandPool = null,
    level: VkCommandBufferLevel = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    commandBufferCount: u32 = 0,
};

pub const VkCommandBufferInheritanceInfo = opaque {};

pub const VkCommandBufferBeginInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkCommandBufferUsageFlags = 0,
    // Primary buffers (the only kind M8a allocates) leave this null.
    pInheritanceInfo: ?*const VkCommandBufferInheritanceInfo = null,
};

pub const VkSubmitInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
    pNext: ?*const anyopaque = null,
    waitSemaphoreCount: u32 = 0,
    pWaitSemaphores: ?[*]const VkSemaphore = null,
    pWaitDstStageMask: ?[*]const VkPipelineStageFlags = null,
    commandBufferCount: u32 = 0,
    pCommandBuffers: ?[*]const VkCommandBuffer = null,
    signalSemaphoreCount: u32 = 0,
    pSignalSemaphores: ?[*]const VkSemaphore = null,
};

pub const VkFenceCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkFenceCreateFlags = 0,
};

pub const VkQueryPoolCreateFlags = u32;
pub const VkQueryPipelineStatisticFlags = u32;
pub const VkQueryPoolCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkQueryPoolCreateFlags = 0,
    queryType: VkQueryType = VK_QUERY_TYPE_TIMESTAMP,
    queryCount: u32 = 0,
    // Only meaningful for queryType == PIPELINE_STATISTICS; zero for
    // TIMESTAMP. Kept here so the layout matches the C ABI exactly.
    pipelineStatistics: VkQueryPipelineStatisticFlags = 0,
};

// ── M8a function-pointer signatures ──────────────────────────────
pub const FnCreateCommandPool = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkCommandPoolCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pCommandPool: *VkCommandPool,
) callconv(.c) VkResult;

pub const FnDestroyCommandPool = *const fn (
    device: VkDevice,
    commandPool: VkCommandPool,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnAllocateCommandBuffers = *const fn (
    device: VkDevice,
    pAllocateInfo: *const VkCommandBufferAllocateInfo,
    pCommandBuffers: [*]VkCommandBuffer,
) callconv(.c) VkResult;

pub const FnFreeCommandBuffers = *const fn (
    device: VkDevice,
    commandPool: VkCommandPool,
    commandBufferCount: u32,
    pCommandBuffers: [*]const VkCommandBuffer,
) callconv(.c) void;

pub const FnResetCommandBuffer = *const fn (
    commandBuffer: VkCommandBuffer,
    flags: VkCommandBufferResetFlags,
) callconv(.c) VkResult;

pub const FnBeginCommandBuffer = *const fn (
    commandBuffer: VkCommandBuffer,
    pBeginInfo: *const VkCommandBufferBeginInfo,
) callconv(.c) VkResult;

pub const FnEndCommandBuffer = *const fn (
    commandBuffer: VkCommandBuffer,
) callconv(.c) VkResult;

pub const FnCmdBindPipeline = *const fn (
    commandBuffer: VkCommandBuffer,
    pipelineBindPoint: VkPipelineBindPoint,
    pipeline: VkPipeline,
) callconv(.c) void;

pub const FnCmdBindDescriptorSets = *const fn (
    commandBuffer: VkCommandBuffer,
    pipelineBindPoint: VkPipelineBindPoint,
    layout: VkPipelineLayout,
    firstSet: u32,
    descriptorSetCount: u32,
    pDescriptorSets: [*]const VkDescriptorSet,
    dynamicOffsetCount: u32,
    pDynamicOffsets: ?[*]const u32,
) callconv(.c) void;

pub const FnCmdPushConstants = *const fn (
    commandBuffer: VkCommandBuffer,
    layout: VkPipelineLayout,
    stageFlags: VkShaderStageFlags,
    offset: u32,
    size: u32,
    pValues: *const anyopaque,
) callconv(.c) void;

pub const FnCmdDispatch = *const fn (
    commandBuffer: VkCommandBuffer,
    groupCountX: u32,
    groupCountY: u32,
    groupCountZ: u32,
) callconv(.c) void;

pub const FnQueueSubmit = *const fn (
    queue: VkQueue,
    submitCount: u32,
    pSubmits: [*]const VkSubmitInfo,
    fence: VkFence,
) callconv(.c) VkResult;

pub const FnQueueWaitIdle = *const fn (
    queue: VkQueue,
) callconv(.c) VkResult;

pub const FnCreateFence = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkFenceCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pFence: *VkFence,
) callconv(.c) VkResult;

pub const FnDestroyFence = *const fn (
    device: VkDevice,
    fence: VkFence,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnResetFences = *const fn (
    device: VkDevice,
    fenceCount: u32,
    pFences: [*]const VkFence,
) callconv(.c) VkResult;

pub const FnWaitForFences = *const fn (
    device: VkDevice,
    fenceCount: u32,
    pFences: [*]const VkFence,
    waitAll: VkBool32,
    timeout: u64,
) callconv(.c) VkResult;

pub const FnCreateQueryPool = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkQueryPoolCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pQueryPool: *VkQueryPool,
) callconv(.c) VkResult;

pub const FnDestroyQueryPool = *const fn (
    device: VkDevice,
    queryPool: VkQueryPool,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnCmdResetQueryPool = *const fn (
    commandBuffer: VkCommandBuffer,
    queryPool: VkQueryPool,
    firstQuery: u32,
    queryCount: u32,
) callconv(.c) void;

pub const FnCmdWriteTimestamp = *const fn (
    commandBuffer: VkCommandBuffer,
    pipelineStage: VkPipelineStageFlags,
    queryPool: VkQueryPool,
    query: u32,
) callconv(.c) void;

pub const FnGetQueryPoolResults = *const fn (
    device: VkDevice,
    queryPool: VkQueryPool,
    firstQuery: u32,
    queryCount: u32,
    dataSize: usize,
    pData: *anyopaque,
    stride: u64,
    flags: VkQueryResultFlags,
) callconv(.c) VkResult;

// ── M8a function-pointer slots ───────────────────────────────────
// Populated by device.zig after vkCreateDevice via vkGetDeviceProcAddr,
// or lazily on first use by dispatch.zig if device.zig hasn't been
// taught about them yet. Both paths fall back to the instance-level
// thunk via vkGetInstanceProcAddr when getDeviceProc is unavailable.
pub var vkCreateCommandPool_fn: ?FnCreateCommandPool = null;
pub var vkDestroyCommandPool_fn: ?FnDestroyCommandPool = null;
pub var vkAllocateCommandBuffers_fn: ?FnAllocateCommandBuffers = null;
pub var vkFreeCommandBuffers_fn: ?FnFreeCommandBuffers = null;
pub var vkResetCommandBuffer_fn: ?FnResetCommandBuffer = null;
pub var vkBeginCommandBuffer_fn: ?FnBeginCommandBuffer = null;
pub var vkEndCommandBuffer_fn: ?FnEndCommandBuffer = null;
pub var vkCmdBindPipeline_fn: ?FnCmdBindPipeline = null;
pub var vkCmdBindDescriptorSets_fn: ?FnCmdBindDescriptorSets = null;
pub var vkCmdPushConstants_fn: ?FnCmdPushConstants = null;
pub var vkCmdDispatch_fn: ?FnCmdDispatch = null;
pub var vkQueueSubmit_fn: ?FnQueueSubmit = null;
pub var vkQueueWaitIdle_fn: ?FnQueueWaitIdle = null;
pub var vkCreateFence_fn: ?FnCreateFence = null;
pub var vkDestroyFence_fn: ?FnDestroyFence = null;
pub var vkResetFences_fn: ?FnResetFences = null;
pub var vkWaitForFences_fn: ?FnWaitForFences = null;
pub var vkCreateQueryPool_fn: ?FnCreateQueryPool = null;
pub var vkDestroyQueryPool_fn: ?FnDestroyQueryPool = null;
pub var vkCmdResetQueryPool_fn: ?FnCmdResetQueryPool = null;
pub var vkCmdWriteTimestamp_fn: ?FnCmdWriteTimestamp = null;
pub var vkGetQueryPoolResults_fn: ?FnGetQueryPoolResults = null;

// ── Module state (dlopen handle + bootstrap fn ptr) ──────────────
pub var lib: ?*anyopaque = null;

/// Loader bring-up state. `uninit` → `in_progress` → `ready`|`failed`.
/// `init` re-entrance during `in_progress` returns false to break loops.
pub const InitState = enum { uninit, in_progress, ready, failed };
pub var init_state: InitState = .uninit;

// ── Function-pointer signatures ──────────────────────────────────
// Bootstrap: resolved via GetProcAddress from vulkan-1.dll.
pub const FnGetInstanceProcAddr = *const fn (instance: VkInstance, pName: [*:0]const u8) callconv(.c) PFN_vkVoidFunction;

// Global-level (instance == VK_NULL_HANDLE → vkGetInstanceProcAddr returns
// these). Resolved by `init()` after the bootstrap pointer is in place.
pub const FnCreateInstance = *const fn (
    pCreateInfo: *const VkInstanceCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pInstance: *VkInstance,
) callconv(.c) VkResult;

// Instance-level: resolved by instance.zig after vkCreateInstance succeeds.
pub const FnDestroyInstance = *const fn (
    instance: VkInstance,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnEnumeratePhysicalDevices = *const fn (
    instance: VkInstance,
    pPhysicalDeviceCount: *u32,
    pPhysicalDevices: ?[*]VkPhysicalDevice,
) callconv(.c) VkResult;

pub const FnGetPhysicalDeviceProperties = *const fn (
    physicalDevice: VkPhysicalDevice,
    pProperties: *VkPhysicalDeviceProperties,
) callconv(.c) void;

pub const FnGetPhysicalDeviceQueueFamilyProperties = *const fn (
    physicalDevice: VkPhysicalDevice,
    pQueueFamilyPropertyCount: *u32,
    pQueueFamilyProperties: ?[*]VkQueueFamilyProperties,
) callconv(.c) void;

pub const FnCreateDevice = *const fn (
    physicalDevice: VkPhysicalDevice,
    pCreateInfo: *const VkDeviceCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pDevice: *VkDevice,
) callconv(.c) VkResult;

pub const FnDestroyDevice = *const fn (
    device: VkDevice,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnGetDeviceQueue = *const fn (
    device: VkDevice,
    queueFamilyIndex: u32,
    queueIndex: u32,
    pQueue: *VkQueue,
) callconv(.c) void;

pub const FnGetDeviceProcAddr = *const fn (
    device: VkDevice,
    pName: [*:0]const u8,
) callconv(.c) PFN_vkVoidFunction;

// M2 additions — Vulkan 1.1 core entry points for Features2/Properties2.
// These are guaranteed resolvable on instances created with apiVersion >= 1.1.
// `pProperties` / `pFeatures` are typed `*anyopaque` because the *first* field
// of either chain head is the sType we already pre-populated; callers pass
// `&Features2{...}` / `&Properties2{...}` with sType + pNext chain wired up.
pub const FnGetPhysicalDeviceProperties2 = *const fn (
    physicalDevice: VkPhysicalDevice,
    pProperties: *anyopaque,
) callconv(.c) void;

pub const FnGetPhysicalDeviceFeatures2 = *const fn (
    physicalDevice: VkPhysicalDevice,
    pFeatures: *anyopaque,
) callconv(.c) void;

// vkEnumerateDeviceExtensionProperties — needed to detect
// VK_NV_shader_subgroup_partitioned (the Tier-1+NV bonus extension).
pub const FnEnumerateDeviceExtensionProperties = *const fn (
    physicalDevice: VkPhysicalDevice,
    pLayerName: ?[*:0]const u8,
    pPropertyCount: *u32,
    pProperties: ?[*]VkExtensionProperties,
) callconv(.c) VkResult;

// M7 — device-level pipeline cache management. Resolved at slzCreate_vk
// time by the device.zig sibling that owns the VkDevice, then consumed by
// `pipeline_cache.zig`'s loadOrCreate / save helpers.
pub const FnCreatePipelineCache = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkPipelineCacheCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pPipelineCache: *VkPipelineCache,
) callconv(.c) VkResult;

pub const FnDestroyPipelineCache = *const fn (
    device: VkDevice,
    pipelineCache: VkPipelineCache,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

// Two-call idiom: first invocation with pData=null returns the required
// byte count in *pDataSize; second with a buffer of that size populates
// it. VK_INCOMPLETE indicates the supplied buffer was too small (driver
// wrote what it could fit and *pDataSize was updated). Callers treat
// INCOMPLETE as "blob skipped, persistence is best-effort."
pub const FnGetPipelineCacheData = *const fn (
    device: VkDevice,
    pipelineCache: VkPipelineCache,
    pDataSize: *usize,
    pData: ?*anyopaque,
) callconv(.c) VkResult;

// ── Function-pointer slots ───────────────────────────────────────
// Bootstrap, populated in `init()`:
pub var vkGetInstanceProcAddr_fn: ?FnGetInstanceProcAddr = null;
// Global-level, also populated in `init()` once the bootstrap is up:
pub var vkCreateInstance_fn: ?FnCreateInstance = null;
// Instance-level, populated by instance.zig after vkCreateInstance:
pub var vkDestroyInstance_fn: ?FnDestroyInstance = null;
pub var vkEnumeratePhysicalDevices_fn: ?FnEnumeratePhysicalDevices = null;
pub var vkGetPhysicalDeviceProperties_fn: ?FnGetPhysicalDeviceProperties = null;
pub var vkGetPhysicalDeviceQueueFamilyProperties_fn: ?FnGetPhysicalDeviceQueueFamilyProperties = null;
pub var vkCreateDevice_fn: ?FnCreateDevice = null;
pub var vkGetDeviceProcAddr_fn: ?FnGetDeviceProcAddr = null;
// Device-level, populated by device.zig after vkCreateDevice. Device
// functions can be resolved per-device via vkGetDeviceProcAddr for one
// less dispatch hop, but at M1 the instance-level slot is fine — we
// only ever drive one VkDevice at a time.
pub var vkDestroyDevice_fn: ?FnDestroyDevice = null;
pub var vkGetDeviceQueue_fn: ?FnGetDeviceQueue = null;

// M2 — instance-level, resolved alongside the M1 slots by instance.zig.
// These are core in Vulkan 1.1+; the instance was created with at least
// apiVersion 1.2 (see instance.zig's 1.3→1.2 fallback) so both are
// expected to be present. Probe-time guards still null-check them.
pub var vkGetPhysicalDeviceProperties2_fn: ?FnGetPhysicalDeviceProperties2 = null;
pub var vkGetPhysicalDeviceFeatures2_fn: ?FnGetPhysicalDeviceFeatures2 = null;
pub var vkEnumerateDeviceExtensionProperties_fn: ?FnEnumerateDeviceExtensionProperties = null;

// M7 — device-level pipeline cache slots. Populated by device.zig (or the
// M8b descriptor-factory module) after vkCreateDevice via vkGetDeviceProcAddr.
pub var vkCreatePipelineCache_fn: ?FnCreatePipelineCache = null;
pub var vkDestroyPipelineCache_fn: ?FnDestroyPipelineCache = null;
pub var vkGetPipelineCacheData_fn: ?FnGetPipelineCacheData = null;

// ── M8b: descriptor set layouts, pools, sets, pipeline layouts, compute pipelines, shader modules ──
//
// The descriptor-factory module (descriptors.zig) owns the lifecycle of all
// per-pipeline state: VkShaderModule, VkDescriptorSetLayout (one binding per
// SSBO + optional push constants), VkPipelineLayout, VkComputePipeline, plus
// a VkDescriptorPool that mints VkDescriptorSet handles for each dispatch.
//
// Modeling note: VkDescriptorSetLayout, VkDescriptorPool, VkShaderModule are
// non-dispatchable Vulkan handles (64-bit IDs in the C ABI). Same pointer-
// shaped `?*opaque` modeling as the M8a non-dispatchable handles above —
// see the comment block at the top of this file for the rationale.
pub const VkDescriptorSetLayout = ?*opaque {};
pub const VkDescriptorPool = ?*opaque {};
pub const VkShaderModule = ?*opaque {};

// ── M8b sType values (descriptor + pipeline create-info chain) ───
// Constants pulled from vulkan_core.h; all Vulkan 1.0 core.
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: VkStructureType = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: VkStructureType = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: VkStructureType = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: VkStructureType = 35;
pub const VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET: VkStructureType = 36;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: VkStructureType = 30;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: VkStructureType = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: VkStructureType = 18;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: VkStructureType = 16;

// ── M8b: descriptor type enum (subset we actually bind) ──────────
// The cache only ever binds storage buffers — every kernel input/output is
// an SSBO (BDA-backed for the device-pointer ABI sites, descriptor-bound
// for everything else). Uniform buffers and texel buffers are intentionally
// omitted because none of the 17 kernels need them.
pub const VkDescriptorType = c_int;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: VkDescriptorType = 7;

// ── M8b: descriptor set layout binding flags / create flags ──────
// We don't use update-after-bind or push-descriptor features at M8b —
// descriptor sets are allocated, written once, and used by one dispatch.
pub const VkDescriptorSetLayoutCreateFlags = u32;
pub const VkDescriptorPoolCreateFlags = u32;
// FREE_DESCRIPTOR_SET_BIT lets us vkFreeDescriptorSets in addition to the
// pool-reset path. We don't actually free individual sets at M8b (cache
// drops the whole pool on invalidate), but the flag is cheap and keeps
// the option open for M11+ where per-dispatch set churn may need it.
pub const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: VkDescriptorPoolCreateFlags = 0x1;

pub const VkPipelineLayoutCreateFlags = u32;
pub const VkPipelineCreateFlags = u32;
pub const VkPipelineShaderStageCreateFlags = u32;
pub const VkShaderModuleCreateFlags = u32;

// ── M8b: descriptor set layout structs ───────────────────────────
pub const VkDescriptorSetLayoutBinding = extern struct {
    binding: u32 = 0,
    descriptorType: VkDescriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    descriptorCount: u32 = 1,
    stageFlags: VkShaderStageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    // pImmutableSamplers — null for storage buffers (only meaningful for
    // sampler / combined-image-sampler descriptor types).
    pImmutableSamplers: ?*const anyopaque = null,
};

pub const VkDescriptorSetLayoutCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDescriptorSetLayoutCreateFlags = 0,
    bindingCount: u32 = 0,
    pBindings: ?[*]const VkDescriptorSetLayoutBinding = null,
};

// ── M8b: descriptor pool structs ─────────────────────────────────
pub const VkDescriptorPoolSize = extern struct {
    type: VkDescriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    descriptorCount: u32 = 0,
};

pub const VkDescriptorPoolCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkDescriptorPoolCreateFlags = 0,
    maxSets: u32 = 0,
    poolSizeCount: u32 = 0,
    pPoolSizes: ?[*]const VkDescriptorPoolSize = null,
};

pub const VkDescriptorSetAllocateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    descriptorPool: VkDescriptorPool = null,
    descriptorSetCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
};

// ── M8b: VkBuffer handle + descriptor buffer info ────────────────
// VkBuffer is non-dispatchable; same `?*opaque` modeling rationale as the
// other handles above. Real VkBuffer creation lives in M7's memory module
// — we model the handle here so the descriptor-write surface can name it.
pub const VkBuffer = ?*opaque {};
pub const VkDeviceSize = u64;
pub const VK_WHOLE_SIZE: VkDeviceSize = std.math.maxInt(u64);

pub const VkDescriptorBufferInfo = extern struct {
    buffer: VkBuffer = null,
    offset: VkDeviceSize = 0,
    range: VkDeviceSize = VK_WHOLE_SIZE,
};

pub const VkWriteDescriptorSet = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    pNext: ?*const anyopaque = null,
    dstSet: VkDescriptorSet = null,
    dstBinding: u32 = 0,
    dstArrayElement: u32 = 0,
    descriptorCount: u32 = 0,
    descriptorType: VkDescriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    // pImageInfo / pTexelBufferView left null for storage-buffer writes.
    pImageInfo: ?*const anyopaque = null,
    pBufferInfo: ?[*]const VkDescriptorBufferInfo = null,
    pTexelBufferView: ?*const anyopaque = null,
};

pub const VkCopyDescriptorSet = opaque {};

// ── M8b: push-constant range ─────────────────────────────────────
pub const VkPushConstantRange = extern struct {
    stageFlags: VkShaderStageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    offset: u32 = 0,
    size: u32 = 0,
};

// ── M8b: pipeline layout + compute pipeline + shader module ──────
pub const VkPipelineLayoutCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineLayoutCreateFlags = 0,
    setLayoutCount: u32 = 0,
    pSetLayouts: ?[*]const VkDescriptorSetLayout = null,
    pushConstantRangeCount: u32 = 0,
    pPushConstantRanges: ?[*]const VkPushConstantRange = null,
};

pub const VkShaderModuleCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkShaderModuleCreateFlags = 0,
    // SPIR-V byte count (must be a multiple of 4).
    codeSize: usize = 0,
    // Pointer to the SPIR-V words; spec requires 4-byte alignment.
    pCode: ?[*]const u32 = null,
};

// VkSpecializationInfo / MapEntry — wired so future kernels can override
// constants (workgroup size, sub_chunk_cap, etc.) without recompiling SPV.
// M8b's getOrCreate does NOT pass a specialization info (defaults work for
// the M8b cache-shape test); callers wanting specialization construct one
// and pass it through to vkCreateComputePipelines directly in later
// milestones (M11+).
pub const VkSpecializationMapEntry = extern struct {
    constantID: u32 = 0,
    offset: u32 = 0,
    size: usize = 0,
};

pub const VkSpecializationInfo = extern struct {
    mapEntryCount: u32 = 0,
    pMapEntries: ?[*]const VkSpecializationMapEntry = null,
    dataSize: usize = 0,
    pData: ?*const anyopaque = null,
};

pub const VkPipelineShaderStageCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineShaderStageCreateFlags = 0,
    stage: VkShaderStageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    module: VkShaderModule = null,
    // Entry-point name in the SPIR-V module. glslc emits "main" by default;
    // we hard-code it at the call site rather than carrying it through the
    // PipelineKey because every committed .spv uses "main".
    pName: ?[*:0]const u8 = null,
    pSpecializationInfo: ?*const VkSpecializationInfo = null,
};

// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo — chain into the
// pNext slot of VkPipelineShaderStageCreateInfo to force a specific
// subgroup size at pipeline creation. Required because Intel UHD picks
// subgroupSize=16 by default (SIMD8 pairs) instead of the device's
// max-subgroupSize, which silently turns the shader's 32-wide warp
// assumptions into broken 16-wide behavior.
pub const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    requiredSubgroupSize: u32 = 0,
};

pub const VkComputePipelineCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkPipelineCreateFlags = 0,
    stage: VkPipelineShaderStageCreateInfo = .{},
    layout: VkPipelineLayout = null,
    // Pipeline derivatives are a hint we never use; M8b leaves both null.
    basePipelineHandle: VkPipeline = null,
    basePipelineIndex: i32 = -1,
};

// ── M8b function-pointer signatures ──────────────────────────────
pub const FnCreateDescriptorSetLayout = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkDescriptorSetLayoutCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pSetLayout: *VkDescriptorSetLayout,
) callconv(.c) VkResult;

pub const FnDestroyDescriptorSetLayout = *const fn (
    device: VkDevice,
    descriptorSetLayout: VkDescriptorSetLayout,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnCreateDescriptorPool = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkDescriptorPoolCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pDescriptorPool: *VkDescriptorPool,
) callconv(.c) VkResult;

pub const FnDestroyDescriptorPool = *const fn (
    device: VkDevice,
    descriptorPool: VkDescriptorPool,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnAllocateDescriptorSets = *const fn (
    device: VkDevice,
    pAllocateInfo: *const VkDescriptorSetAllocateInfo,
    pDescriptorSets: [*]VkDescriptorSet,
) callconv(.c) VkResult;

pub const FnUpdateDescriptorSets = *const fn (
    device: VkDevice,
    descriptorWriteCount: u32,
    pDescriptorWrites: ?[*]const VkWriteDescriptorSet,
    descriptorCopyCount: u32,
    pDescriptorCopies: ?*const VkCopyDescriptorSet,
) callconv(.c) void;

pub const FnCreatePipelineLayout = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkPipelineLayoutCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pPipelineLayout: *VkPipelineLayout,
) callconv(.c) VkResult;

pub const FnDestroyPipelineLayout = *const fn (
    device: VkDevice,
    pipelineLayout: VkPipelineLayout,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnCreateComputePipelines = *const fn (
    device: VkDevice,
    pipelineCache: VkPipelineCache,
    createInfoCount: u32,
    pCreateInfos: [*]const VkComputePipelineCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pPipelines: [*]VkPipeline,
) callconv(.c) VkResult;

pub const FnDestroyPipeline = *const fn (
    device: VkDevice,
    pipeline: VkPipeline,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnCreateShaderModule = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkShaderModuleCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pShaderModule: *VkShaderModule,
) callconv(.c) VkResult;

pub const FnDestroyShaderModule = *const fn (
    device: VkDevice,
    shaderModule: VkShaderModule,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

// ── M8b function-pointer slots ───────────────────────────────────
// Populated lazily on first descriptors.getOrCreate via vkGetDeviceProcAddr
// (same pattern as M8a's dispatch.zig). M-late milestones may hoist this
// into device.zig once the device-creation site stabilizes.
pub var vkCreateDescriptorSetLayout_fn: ?FnCreateDescriptorSetLayout = null;
pub var vkDestroyDescriptorSetLayout_fn: ?FnDestroyDescriptorSetLayout = null;
pub var vkCreateDescriptorPool_fn: ?FnCreateDescriptorPool = null;
pub var vkDestroyDescriptorPool_fn: ?FnDestroyDescriptorPool = null;
pub var vkAllocateDescriptorSets_fn: ?FnAllocateDescriptorSets = null;
pub var vkUpdateDescriptorSets_fn: ?FnUpdateDescriptorSets = null;
pub var vkCreatePipelineLayout_fn: ?FnCreatePipelineLayout = null;
pub var vkDestroyPipelineLayout_fn: ?FnDestroyPipelineLayout = null;
pub var vkCreateComputePipelines_fn: ?FnCreateComputePipelines = null;
pub var vkDestroyPipeline_fn: ?FnDestroyPipeline = null;
pub var vkCreateShaderModule_fn: ?FnCreateShaderModule = null;
pub var vkDestroyShaderModule_fn: ?FnDestroyShaderModule = null;

// ── M8c: sync1 + sync2 barrier wrapper + buffer/memory plumbing ──
//
// M8c needs:
//   1. A sync wrapper that picks vkCmdPipelineBarrier2 when synchronization2
//      is supported (Tier-1 path) and falls back to vkCmdPipelineBarrier
//      with the sync1-shaped VkBufferMemoryBarrier struct otherwise.
//   2. A 68-slot timestamp query pool (per arch §15).
//   3. VkBuffer + VkDeviceMemory create/alloc/bind/map so dispatch_test
//      can stand up a real SSBO and read back its contents.
//
// sync2 sType constants — promoted from VK_KHR_synchronization2 to core in
// Vulkan 1.3. Drivers that report synchronization2==VK_TRUE (whether via
// the v13 omnibus or the KHR per-extension struct) accept these structs.
pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER: VkStructureType = 44;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: VkStructureType = 5;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: VkStructureType = 12;
pub const VK_STRUCTURE_TYPE_DEPENDENCY_INFO: VkStructureType = 1000314003;
pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2: VkStructureType = 1000314001;

// ── sync1: VkAccessFlags + the existing VkPipelineStageFlags ─────
// Pulled from vulkan_core.h; the subset M8c needs at the call site.
pub const VkAccessFlags = u32;
pub const VK_ACCESS_SHADER_READ_BIT: VkAccessFlags = 0x20;
pub const VK_ACCESS_SHADER_WRITE_BIT: VkAccessFlags = 0x40;
pub const VK_ACCESS_TRANSFER_READ_BIT: VkAccessFlags = 0x800;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: VkAccessFlags = 0x1000;
pub const VK_ACCESS_HOST_READ_BIT: VkAccessFlags = 0x2000;
pub const VK_ACCESS_HOST_WRITE_BIT: VkAccessFlags = 0x4000;

// QUEUE_FAMILY_IGNORED — used in VkBufferMemoryBarrier when not actually
// transferring queue ownership (M8c's single-queue model never does).
pub const VK_QUEUE_FAMILY_IGNORED: u32 = std.math.maxInt(u32);

// ── sync2: 64-bit stage + access masks ───────────────────────────
// sync1 used 32-bit masks; sync2 promoted these to 64-bit to make room
// for new stages (e.g. RT, MESH). The named bit values are stable across
// sync1↔sync2 for the bits both versions define (TOP_OF_PIPE, COMPUTE_
// SHADER, etc. — see vk_synchronization2 spec).
pub const VkPipelineStageFlags2 = u64;
pub const VK_PIPELINE_STAGE_2_NONE: VkPipelineStageFlags2 = 0;
pub const VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT: VkPipelineStageFlags2 = 0x1;
pub const VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT: VkPipelineStageFlags2 = 0x800;
pub const VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT: VkPipelineStageFlags2 = 0x2000;
pub const VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT: VkPipelineStageFlags2 = 0x1000;
pub const VK_PIPELINE_STAGE_2_HOST_BIT: VkPipelineStageFlags2 = 0x4000;
pub const VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT: VkPipelineStageFlags2 = 0x10000;

pub const VkAccessFlags2 = u64;
pub const VK_ACCESS_2_NONE: VkAccessFlags2 = 0;
pub const VK_ACCESS_2_SHADER_READ_BIT: VkAccessFlags2 = 0x20;
pub const VK_ACCESS_2_SHADER_WRITE_BIT: VkAccessFlags2 = 0x40;
pub const VK_ACCESS_2_TRANSFER_READ_BIT: VkAccessFlags2 = 0x800;
pub const VK_ACCESS_2_TRANSFER_WRITE_BIT: VkAccessFlags2 = 0x1000;
pub const VK_ACCESS_2_HOST_READ_BIT: VkAccessFlags2 = 0x2000;
pub const VK_ACCESS_2_HOST_WRITE_BIT: VkAccessFlags2 = 0x4000;
pub const VK_ACCESS_2_MEMORY_READ_BIT: VkAccessFlags2 = 0x8000;
pub const VK_ACCESS_2_MEMORY_WRITE_BIT: VkAccessFlags2 = 0x10000;

// VkDependencyFlags — empty by default; spec bits unused at M8c.
pub const VkDependencyFlags = u32;

// VkMemoryBarrier (sync1) is left unmodeled — the M8c wrapper does NOT
// emit global memory barriers; everything is per-buffer. Add when needed.

// ── sync1: VkBufferMemoryBarrier (24-byte header + buffer/offset/size) ─
pub const VkBufferMemoryBarrier = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    pNext: ?*const anyopaque = null,
    srcAccessMask: VkAccessFlags = 0,
    dstAccessMask: VkAccessFlags = 0,
    srcQueueFamilyIndex: u32 = VK_QUEUE_FAMILY_IGNORED,
    dstQueueFamilyIndex: u32 = VK_QUEUE_FAMILY_IGNORED,
    buffer: VkBuffer = null,
    offset: VkDeviceSize = 0,
    size: VkDeviceSize = VK_WHOLE_SIZE,
};

// ── sync2: VkBufferMemoryBarrier2 (carries the stage masks per-barrier) ─
pub const VkBufferMemoryBarrier2 = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
    pNext: ?*const anyopaque = null,
    srcStageMask: VkPipelineStageFlags2 = 0,
    srcAccessMask: VkAccessFlags2 = 0,
    dstStageMask: VkPipelineStageFlags2 = 0,
    dstAccessMask: VkAccessFlags2 = 0,
    srcQueueFamilyIndex: u32 = VK_QUEUE_FAMILY_IGNORED,
    dstQueueFamilyIndex: u32 = VK_QUEUE_FAMILY_IGNORED,
    buffer: VkBuffer = null,
    offset: VkDeviceSize = 0,
    size: VkDeviceSize = VK_WHOLE_SIZE,
};

// ── sync2: VkDependencyInfo (wraps barrier arrays for pipeline-barrier2) ─
pub const VkDependencyInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
    pNext: ?*const anyopaque = null,
    dependencyFlags: VkDependencyFlags = 0,
    memoryBarrierCount: u32 = 0,
    pMemoryBarriers: ?*const anyopaque = null, // VkMemoryBarrier2* — not modeled
    bufferMemoryBarrierCount: u32 = 0,
    pBufferMemoryBarriers: ?[*]const VkBufferMemoryBarrier2 = null,
    imageMemoryBarrierCount: u32 = 0,
    pImageMemoryBarriers: ?*const anyopaque = null, // VkImageMemoryBarrier2* — not modeled
};

// ── VkBuffer create + memory plumbing ────────────────────────────
pub const VkBufferCreateFlags = u32;
pub const VkBufferUsageFlags = u32;
pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: VkBufferUsageFlags = 0x1;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: VkBufferUsageFlags = 0x2;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: VkBufferUsageFlags = 0x10;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: VkBufferUsageFlags = 0x20;

pub const VkSharingMode = c_int;
pub const VK_SHARING_MODE_EXCLUSIVE: VkSharingMode = 0;
pub const VK_SHARING_MODE_CONCURRENT: VkSharingMode = 1;

pub const VkBufferCreateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    pNext: ?*const anyopaque = null,
    flags: VkBufferCreateFlags = 0,
    size: VkDeviceSize = 0,
    usage: VkBufferUsageFlags = 0,
    sharingMode: VkSharingMode = VK_SHARING_MODE_EXCLUSIVE,
    queueFamilyIndexCount: u32 = 0,
    pQueueFamilyIndices: ?[*]const u32 = null,
};

// VkMemoryRequirements: 3 × VkDeviceSize (u64) = 24 bytes.
pub const VkMemoryRequirements = extern struct {
    size: VkDeviceSize = 0,
    alignment: VkDeviceSize = 0,
    memoryTypeBits: u32 = 0,
};

pub const VkDeviceMemory = ?*opaque {};
pub const VK_NULL_HANDLE_OBJECT: VkDeviceMemory = null;

pub const VkMemoryAllocateInfo = extern struct {
    sType: VkStructureType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    pNext: ?*const anyopaque = null,
    allocationSize: VkDeviceSize = 0,
    memoryTypeIndex: u32 = 0,
};

// ── VkPhysicalDeviceMemoryProperties ─────────────────────────────
// VK_MAX_MEMORY_TYPES = 32, VK_MAX_MEMORY_HEAPS = 16. Each VkMemoryType
// is 8 bytes (u32 propertyFlags + u32 heapIndex); each VkMemoryHeap is
// 16 bytes (u64 size + u32 flags + 4 pad). Total layout:
//   u32 memoryTypeCount                                     4
//   pad to 8 for the heap-aligned (none — VkMemoryType has 4-byte alignment)
//   VkMemoryType[32]  (32 × 8 = 256)                      260
//   u32 memoryHeapCount                                   264
//   pad to 8 (for VkMemoryHeap's u64)                     272
//   VkMemoryHeap[16] (16 × 16 = 256)                      528
// Spec-confirmed sizeof = 520 on most compilers (no trailing pad after
// the last heap). We model it explicitly so memory selection works.
pub const VK_MAX_MEMORY_TYPES: usize = 32;
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

pub const VkMemoryPropertyFlags = u32;
pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: VkMemoryPropertyFlags = 0x1;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: VkMemoryPropertyFlags = 0x2;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: VkMemoryPropertyFlags = 0x4;
pub const VK_MEMORY_PROPERTY_HOST_CACHED_BIT: VkMemoryPropertyFlags = 0x8;
pub const VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT: VkMemoryPropertyFlags = 0x10;

pub const VkMemoryHeapFlags = u32;

pub const VkMemoryType = extern struct {
    propertyFlags: VkMemoryPropertyFlags = 0,
    heapIndex: u32 = 0,
};

pub const VkMemoryHeap = extern struct {
    size: VkDeviceSize = 0,
    flags: VkMemoryHeapFlags = 0,
};

pub const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32 = 0,
    memoryTypes: [VK_MAX_MEMORY_TYPES]VkMemoryType = @splat(.{}),
    memoryHeapCount: u32 = 0,
    memoryHeaps: [VK_MAX_MEMORY_HEAPS]VkMemoryHeap = @splat(.{}),
};

// MapMemory's "map the whole rest of the buffer" sentinel.
pub const VkMemoryMapFlags = u32;

// ── M8c function-pointer signatures ──────────────────────────────
pub const FnCmdPipelineBarrier = *const fn (
    commandBuffer: VkCommandBuffer,
    srcStageMask: VkPipelineStageFlags,
    dstStageMask: VkPipelineStageFlags,
    dependencyFlags: VkDependencyFlags,
    memoryBarrierCount: u32,
    pMemoryBarriers: ?*const anyopaque,
    bufferMemoryBarrierCount: u32,
    pBufferMemoryBarriers: ?[*]const VkBufferMemoryBarrier,
    imageMemoryBarrierCount: u32,
    pImageMemoryBarriers: ?*const anyopaque,
) callconv(.c) void;

pub const FnCmdPipelineBarrier2 = *const fn (
    commandBuffer: VkCommandBuffer,
    pDependencyInfo: *const VkDependencyInfo,
) callconv(.c) void;

pub const FnCreateBuffer = *const fn (
    device: VkDevice,
    pCreateInfo: *const VkBufferCreateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pBuffer: *VkBuffer,
) callconv(.c) VkResult;

pub const FnDestroyBuffer = *const fn (
    device: VkDevice,
    buffer: VkBuffer,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnAllocateMemory = *const fn (
    device: VkDevice,
    pAllocateInfo: *const VkMemoryAllocateInfo,
    pAllocator: ?*const VkAllocationCallbacks,
    pMemory: *VkDeviceMemory,
) callconv(.c) VkResult;

pub const FnFreeMemory = *const fn (
    device: VkDevice,
    memory: VkDeviceMemory,
    pAllocator: ?*const VkAllocationCallbacks,
) callconv(.c) void;

pub const FnBindBufferMemory = *const fn (
    device: VkDevice,
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    memoryOffset: VkDeviceSize,
) callconv(.c) VkResult;

pub const FnMapMemory = *const fn (
    device: VkDevice,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    size: VkDeviceSize,
    flags: VkMemoryMapFlags,
    ppData: *?*anyopaque,
) callconv(.c) VkResult;

pub const FnUnmapMemory = *const fn (
    device: VkDevice,
    memory: VkDeviceMemory,
) callconv(.c) void;

pub const FnGetBufferMemoryRequirements = *const fn (
    device: VkDevice,
    buffer: VkBuffer,
    pMemoryRequirements: *VkMemoryRequirements,
) callconv(.c) void;

pub const FnGetPhysicalDeviceMemoryProperties = *const fn (
    physicalDevice: VkPhysicalDevice,
    pMemoryProperties: *VkPhysicalDeviceMemoryProperties,
) callconv(.c) void;

// ── L1 codec (Vulkan port phase 1): buffer-copy + fill device fns ─
// The L1 codec wires src bytes → device-local SSBO via vkCmdCopyBuffer
// from a HOST_VISIBLE staging buffer (faster than map+memcpy into a
// non-HOST_VISIBLE device-local buffer) and zero-clears its output
// SSBOs each encode via vkCmdFillBuffer (cheaper than a GPU-side memset
// kernel). Modeled here so l1_codec.zig stays vk_api-only on the wire.
pub const VkBufferCopy = extern struct {
    srcOffset: VkDeviceSize = 0,
    dstOffset: VkDeviceSize = 0,
    size: VkDeviceSize = 0,
};

pub const FnCmdCopyBuffer = *const fn (
    commandBuffer: VkCommandBuffer,
    srcBuffer: VkBuffer,
    dstBuffer: VkBuffer,
    regionCount: u32,
    pRegions: [*]const VkBufferCopy,
) callconv(.c) void;

pub const FnCmdFillBuffer = *const fn (
    commandBuffer: VkCommandBuffer,
    dstBuffer: VkBuffer,
    dstOffset: VkDeviceSize,
    size: VkDeviceSize,
    data: u32,
) callconv(.c) void;

// ── M8c function-pointer slots ───────────────────────────────────
// Same lazy-on-first-use pattern as M8a/M8b. sync.zig + timing.zig
// populate these inside their ensure helpers.
pub var vkCmdPipelineBarrier_fn: ?FnCmdPipelineBarrier = null;
pub var vkCmdPipelineBarrier2_fn: ?FnCmdPipelineBarrier2 = null;
pub var vkCreateBuffer_fn: ?FnCreateBuffer = null;
pub var vkDestroyBuffer_fn: ?FnDestroyBuffer = null;
pub var vkAllocateMemory_fn: ?FnAllocateMemory = null;
pub var vkFreeMemory_fn: ?FnFreeMemory = null;
pub var vkBindBufferMemory_fn: ?FnBindBufferMemory = null;
pub var vkMapMemory_fn: ?FnMapMemory = null;
pub var vkUnmapMemory_fn: ?FnUnmapMemory = null;
pub var vkGetBufferMemoryRequirements_fn: ?FnGetBufferMemoryRequirements = null;
pub var vkGetPhysicalDeviceMemoryProperties_fn: ?FnGetPhysicalDeviceMemoryProperties = null;
pub var vkCmdCopyBuffer_fn: ?FnCmdCopyBuffer = null;
pub var vkCmdFillBuffer_fn: ?FnCmdFillBuffer = null;

// ── Loader helpers ───────────────────────────────────────────────

/// Resolve a symbol from the loaded vulkan-1.dll via Win32 GetProcAddress.
/// Only used for the bootstrap (vkGetInstanceProcAddr); everything else
/// goes through `getInstanceProc` / `getDeviceProc` below.
pub fn getProc(comptime T: type, name: [*:0]const u8) ?T {
    const h = lib orelse return null;
    const raw = win32.GetProcAddress(h, name) orelse return null;
    return @ptrCast(@alignCast(raw));
}

/// Resolve an instance-level (or global-level when instance is null) entry
/// point via vkGetInstanceProcAddr. Caller casts the void-fn cookie to the
/// concrete signature.
pub fn getInstanceProc(comptime T: type, instance: VkInstance, name: [*:0]const u8) ?T {
    const f = vkGetInstanceProcAddr_fn orelse return null;
    const raw = f(instance, name) orelse return null;
    return @ptrCast(@alignCast(raw));
}

/// Resolve a device-level entry point via vkGetDeviceProcAddr.
pub fn getDeviceProc(comptime T: type, device: VkDevice, name: [*:0]const u8) ?T {
    const f = vkGetDeviceProcAddr_fn orelse return null;
    const raw = f(device, name) orelse return null;
    return @ptrCast(@alignCast(raw));
}

/// Load vulkan-1.dll and resolve the bootstrap (vkGetInstanceProcAddr) +
/// the one global-level entry point we need (vkCreateInstance). Idempotent;
/// safe to call multiple times. Returns true iff `init_state == .ready`.
pub fn init() bool {
    switch (init_state) {
        .ready => return true,
        .failed, .in_progress => return false,
        .uninit => {},
    }
    init_state = .in_progress;
    defer if (init_state == .in_progress) {
        init_state = .failed;
    };

    lib = win32.LoadLibraryA("vulkan-1.dll");
    if (lib == null) return false;

    vkGetInstanceProcAddr_fn = getProc(FnGetInstanceProcAddr, "vkGetInstanceProcAddr");
    if (vkGetInstanceProcAddr_fn == null) return false;

    // Global-level: instance == null. vkCreateInstance MUST be reachable
    // this way per the spec — vkEnumerateInstanceLayerProperties /
    // ...ExtensionProperties / vkCreateInstance are the four global-level
    // entry points and are guaranteed resolvable with a null instance.
    vkCreateInstance_fn = getInstanceProc(FnCreateInstance, null, "vkCreateInstance");
    if (vkCreateInstance_fn == null) return false;

    init_state = .ready;
    return true;
}
