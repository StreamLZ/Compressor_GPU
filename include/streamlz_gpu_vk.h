/* streamlz_gpu_vk.h — StreamLZ Vulkan-backend C ABI sibling.
 *
 * Mirror of include/streamlz_gpu.h with `_vk`-suffixed symbols. Reuses
 * the shared types (slzStatus_t, slzCompressOpts_t, slzDecompressOpts_t,
 * slzKernelTiming_t) from streamlz_gpu.h so callers can compose the two
 * backends in a single translation unit without type collisions.
 *
 * M4 status: 4 of the 16 symbols are wired (slzCreate_vk, slzDestroy_vk,
 * slzGetVersionString_vk, slzRegisterBuffer_vk, slzUnregisterBuffer_vk —
 * 5 actually, the register/unregister pair are both no-ops at Tier-1).
 * The other 11 return SLZ_ERROR_UNSUPPORTED (== 5) until the per-kernel
 * milestones (M11..M27) land them.
 */

#ifndef STREAMLZ_GPU_VK_H
#define STREAMLZ_GPU_VK_H

#include "streamlz_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-caller Vulkan-backend handle. Distinct from slzHandle_t to
 * prevent accidentally feeding a CUDA handle into a Vulkan entry point.
 */
typedef struct slzVkHandle_s* slzVkHandle_t;

/* ---- Handle lifecycle + diagnostics --------------------------------- */
slzStatus_t slzCreate_vk(slzVkHandle_t* out_handle);
void        slzDestroy_vk(slzVkHandle_t handle);
const char* slzGetVersionString_vk(void);

/* ---- Buffer registration (Tier-1: no-op; Tier-2: records mapping) --- */
slzStatus_t slzRegisterBuffer_vk(slzVkHandle_t handle,
                                 void* vk_buffer_handle,
                                 const void* d_base_address,
                                 size_t buffer_size);
slzStatus_t slzUnregisterBuffer_vk(slzVkHandle_t handle,
                                   const void* d_base_address);

/* ---- Sync compress / decompress (D2D + host-host) ------------------- */
/* All STUBBED to SLZ_ERROR_UNSUPPORTED in M4. */
int slzCompress_vk(slzVkHandle_t handle,
                   const void* d_input, size_t input_size,
                   void* d_output, size_t output_capacity,
                   slzCompressOpts_t opts);
int slzCompressHost_vk(slzVkHandle_t handle,
                       const void* input, size_t input_size,
                       void* output, size_t output_capacity,
                       slzCompressOpts_t opts);
int slzDecompress_vk(slzVkHandle_t handle,
                     const void* d_input, size_t input_size,
                     void* d_output, size_t output_capacity,
                     slzDecompressOpts_t opts);
int slzDecompressHost_vk(slzVkHandle_t handle,
                         const void* input, size_t input_size,
                         void* output, size_t output_capacity,
                         slzDecompressOpts_t opts);

size_t       slzCompressBound_vk(size_t input_size);
slzStatus_t  slzMakeDeviceOnlyHandle_vk(const void** out_handle, size_t bytes);

/* ---- Async / polling (STUBBED) -------------------------------------- */
slzStatus_t slzCompressAsync_vk(slzVkHandle_t handle,
                                const void* d_input, size_t input_size,
                                void* d_output, size_t output_capacity,
                                size_t* compressed_size,
                                slzCompressOpts_t opts);
slzStatus_t slzCompressAsyncPoll_vk(slzVkHandle_t handle, int blocking);
slzStatus_t slzDecompressAsync_vk(slzVkHandle_t handle,
                                  const void* d_input, size_t input_size,
                                  void* d_output, size_t output_capacity,
                                  slzDecompressOpts_t opts);
slzStatus_t slzDecompressAsyncPoll_vk(slzVkHandle_t handle, int blocking);

/* ---- Per-kernel timings drain (STUBBED to 0 in M4) ------------------ */
size_t slzGetLastTimings_vk(slzVkHandle_t handle,
                            slzKernelTiming_t* out, size_t capacity);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* STREAMLZ_GPU_VK_H */

/* 16 symbols total:
 *   slzCreate_vk, slzDestroy_vk, slzGetVersionString_vk,
 *   slzRegisterBuffer_vk, slzUnregisterBuffer_vk,
 *   slzCompress_vk, slzCompressHost_vk, slzDecompress_vk, slzDecompressHost_vk,
 *   slzCompressBound_vk, slzMakeDeviceOnlyHandle_vk,
 *   slzCompressAsync_vk, slzCompressAsyncPoll_vk,
 *   slzDecompressAsync_vk, slzDecompressAsyncPoll_vk,
 *   slzGetLastTimings_vk.
 */
