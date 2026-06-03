// VMA single-translation-unit implementation.
//
// This file exists solely so build.zig has one C++ source to compile that
// pulls the entire VMA implementation into the streamlz_vk binary. Adding
// any code here defeats the "vendored as-is" guarantee — keep it minimal.

#define VMA_IMPLEMENTATION

// The Vulkan port resolves every Vulkan function pointer through
// `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr` at runtime (the codec
// never link-time-imports vulkan-1.lib). Force VMA to follow the same
// model: zero static linkage, full dynamic resolution through the
// pVulkanFunctions table the caller supplies to vmaCreateAllocator.
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

// VMA's static-analysis warnings fire on the bundled C++ source; silence
// them so the build stays green without touching the vendored header.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wunused-variable"
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#  pragma GCC diagnostic ignored "-Wnullability-completeness"
#endif

#include "vk_mem_alloc.h"
