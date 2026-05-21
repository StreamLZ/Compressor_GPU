/* streamlz_gpu.h — StreamLZ GPU compression library, C ABI.
 *
 * StreamLZ is an LZ + entropy-coding compressor whose compress and
 * decompress paths run on the GPU. This header exposes it as a callable
 * library, in the style of nvCOMP's GPU codecs.
 *
 * --- Model -----------------------------------------------------------
 * Single buffer in, single self-describing frame out: the frame's
 * internal chunking is handled by the library, so the caller passes one
 * contiguous input and receives one contiguous frame (unlike nvCOMP's
 * batched arrays of independent chunks).
 *
 * Two buffer models, both supported:
 *   - Device->device (slzCompress / slzDecompress): the `d_`-prefixed
 *     pointers are device memory. The caller's data is already
 *     GPU-resident and the caller owns the scratch buffer — best for
 *     interop with the caller's other CUDA / nvCOMP work, no host
 *     round-trip.
 *   - Host->host (slzCompressHost / slzDecompressHost): plain host
 *     pointers. The library does the H2D/D2H copies and owns all
 *     device-side memory itself — simplest to call.
 *
 * Calls are SYNCHRONOUS: each compress/decompress runs the GPU work and
 * blocks until the result is ready. (A future opt-in stream-async
 * variant may be added; it would not change these entry points.)
 *
 * --- CUDA context / threading ----------------------------------------
 * The library runs in the CALLER's CUDA context. slzCreate() must be
 * called with a CUDA context already current; it does not create one. A
 * single handle is not safe for concurrent calls on itself — give each
 * thread its own handle. Distinct handles are fully independent.
 *
 * --- Buffer ownership ------------------------------------------------
 * The caller owns every buffer. Scratch ("temp") sizing is queried up
 * front via slz*GetTempSize and the caller allocates and passes it in, so
 * the library holds no hidden per-operation device memory.
 */

#ifndef STREAMLZ_GPU_H
#define STREAMLZ_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes --------------------------------------------------- */
typedef enum slzStatus_t {
    SLZ_SUCCESS                = 0,
    SLZ_ERROR_INVALID_HANDLE   = 1,  /* handle is NULL or not from slzCreate */
    SLZ_ERROR_INVALID_ARG      = 2,  /* a NULL/zero argument that must not be */
    SLZ_ERROR_BUFFER_TOO_SMALL = 3,  /* output or temp capacity insufficient */
    SLZ_ERROR_CORRUPT_FRAME    = 4,  /* decompress input is not a valid frame */
    SLZ_ERROR_UNSUPPORTED      = 5,  /* level / option not supported */
    SLZ_ERROR_CUDA             = 6,  /* an underlying CUDA call failed */
    SLZ_ERROR_OUT_OF_MEMORY    = 7,  /* a library-internal allocation failed */
} slzStatus_t;

/* Human-readable text for a status code. Never NULL; the returned string
 * is valid for the lifetime of the process. */
const char* slzStatusString(slzStatus_t status);

/* Null-terminated library version string, e.g. "2.0.0". */
const char* slzVersionString(void);

/* ---- Library handle ------------------------------------------------- */
/* Opaque per-caller context: CUDA module handles, kernel pointers, and
 * persistent device scratch. Create one per thread that compresses or
 * decompresses concurrently. */
typedef struct slzContext* slzHandle_t;

/* Create a handle in the caller's current CUDA context. On success
 * *handle is non-NULL. */
slzStatus_t slzCreate(slzHandle_t* handle);

/* Destroy a handle and free everything it owns. Passing NULL is a no-op
 * and returns SLZ_SUCCESS. */
slzStatus_t slzDestroy(slzHandle_t handle);

/* ---- Compression options -------------------------------------------- */
typedef struct slzCompressOpts_t {
    int level;          /* 1..5 — higher = smaller output, slower encode */
    int reserved[7];    /* must be zero — reserved for future options */
} slzCompressOpts_t;

/* Default options (level 5). */
slzCompressOpts_t slzCompressDefaultOpts(void);

/* ---- Size queries --------------------------------------------------- */
/* Worst-case compressed-frame size for `input_size` input bytes. Use it
 * to size the output buffer for slzCompress / slzCompressHost. */
slzStatus_t slzCompressBound(slzHandle_t handle, size_t input_size,
                             slzCompressOpts_t opts, size_t* max_output_size);

/* Device-scratch size the device->device slzCompress needs for
 * `input_size` bytes. (slzCompressHost manages its own scratch.) */
slzStatus_t slzCompressGetTempSize(slzHandle_t handle, size_t input_size,
                                   slzCompressOpts_t opts, size_t* temp_size);

/* Device-scratch size the device->device slzDecompress needs for a
 * `frame_size`-byte frame — a conservative bound. (slzDecompressHost
 * manages its own scratch.) */
slzStatus_t slzDecompressGetTempSize(slzHandle_t handle, size_t frame_size,
                                     size_t* temp_size);

/* Decompressed byte count recorded in a frame header. `frame_header` is a
 * HOST pointer to at least the first 16 bytes of the frame; the size is
 * returned in *decompressed_size. (A device->device caller copies the
 * leading frame bytes to host first — the header is tiny.) */
slzStatus_t slzGetDecompressedSize(slzHandle_t handle,
                                   const void* frame_header, size_t header_len,
                                   size_t* decompressed_size);

/* ---- Compress / decompress: device -> device ------------------------ */
/* Compress d_input (input_size device bytes) into a StreamLZ frame at
 * d_output. Blocks until the frame is ready; the frame's byte length is
 * written to the host-side *output_size.
 *
 *   d_temp        device scratch, >= slzCompressGetTempSize bytes
 *   d_output      device buffer, capacity >= slzCompressBound bytes
 *   output_size   host size_t; receives the frame length
 */
slzStatus_t slzCompress(slzHandle_t handle,
                        const void* d_input, size_t input_size,
                        void* d_temp, size_t temp_size,
                        void* d_output, size_t output_capacity,
                        size_t* output_size,
                        slzCompressOpts_t opts);

/* Decompress a StreamLZ frame at d_frame into d_output. Blocks until the
 * output is ready; the decompressed byte count is written to the
 * host-side *output_size.
 *
 *   d_temp        device scratch, >= slzDecompressGetTempSize bytes
 *   d_output      device buffer, capacity >= slzGetDecompressedSize bytes
 *   output_size   host size_t; receives the decompressed length
 */
slzStatus_t slzDecompress(slzHandle_t handle,
                          const void* d_frame, size_t frame_size,
                          void* d_temp, size_t temp_size,
                          void* d_output, size_t output_capacity,
                          size_t* output_size);

/* ---- Compress / decompress: host -> host ---------------------------- */
/* Same as slzCompress / slzDecompress but with plain host buffers: the
 * library performs the H2D/D2H copies and owns every device-side buffer
 * (input copy, output, and scratch) for the duration of the call. No
 * d_temp is required. One extra host<->device round-trip vs the
 * device->device entry points. */
slzStatus_t slzCompressHost(slzHandle_t handle,
                            const void* input, size_t input_size,
                            void* output, size_t output_capacity,
                            size_t* output_size,
                            slzCompressOpts_t opts);

slzStatus_t slzDecompressHost(slzHandle_t handle,
                              const void* frame, size_t frame_size,
                              void* output, size_t output_capacity,
                              size_t* output_size);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* STREAMLZ_GPU_H */
