/* streamlz_gpu.h — StreamLZ GPU compression library, C ABI.
 *
 * StreamLZ is an LZ + entropy-coding compressor whose compress and
 * decompress paths run on the GPU. This header exposes it as a callable
 * library in the same shape as nvCOMP's batched APIs:
 *
 *   - The library does the compress/decompress work; the caller does the
 *     CUDA memory and transfer plumbing (cudaMallocAsync /
 *     cudaMemcpyAsync / cudaFreeAsync), so caller code composes cleanly
 *     with the rest of their CUDA pipeline (own streams, own memory
 *     pools, own error handling).
 *   - All work is queued on the caller's CUstream / cudaStream_t. The
 *     caller's cudaStreamSynchronize is the only sync point.
 *   - For decompress, the caller learns the output size synchronously
 *     before kicking off the GPU work, via a host-side parse of the
 *     frame header (slzGetDecompressedSize).
 *   - For compress, the caller learns a worst-case bound up front
 *     (slzCompressBound); the actual compressed size is written by the
 *     library to a caller-supplied host pointer after the stream
 *     completes.
 *
 * --- Typical caller patterns ----------------------------------------
 *
 * DECOMPRESS:
 *   size_t output_size;
 *   slzGetDecompressedSize(h, host_compressed_bytes, &output_size);
 *
 *   void *d_frame, *d_output;
 *   cudaMallocAsync(&d_frame,  frame_size,  stream);
 *   cudaMallocAsync(&d_output, output_size, stream);
 *   cudaMemcpyAsync(d_frame, host_compressed_bytes, frame_size,
 *                   cudaMemcpyHostToDevice, stream);
 *   slzDecompressAsync(h, d_frame, frame_size, d_output, output_size,
 *                      dec_opts, stream);
 *   cudaMemcpyAsync(host_output, d_output, output_size,
 *                   cudaMemcpyDeviceToHost, stream);
 *   cudaStreamSynchronize(stream);
 *   cudaFreeAsync(d_output, stream);
 *   cudaFreeAsync(d_frame,  stream);
 *
 * COMPRESS:
 *   size_t max_compressed_size;
 *   slzCompressBound(h, input_size, comp_opts, &max_compressed_size);
 *
 *   void *d_input, *d_output;
 *   cudaMallocAsync(&d_input,  input_size,          stream);
 *   cudaMallocAsync(&d_output, max_compressed_size, stream);
 *   cudaMemcpyAsync(d_input, host_input, input_size,
 *                   cudaMemcpyHostToDevice, stream);
 *
 *   size_t actual_compressed_size;
 *   slzCompressAsync(h, d_input, input_size,
 *                    d_output, max_compressed_size,
 *                    &actual_compressed_size,
 *                    comp_opts, stream);
 *   cudaMemcpyAsync(host_output, d_output, max_compressed_size,
 *                   cudaMemcpyDeviceToHost, stream);
 *   cudaStreamSynchronize(stream);
 *   // host_output's first actual_compressed_size bytes are valid
 *   cudaFreeAsync(d_output, stream);
 *   cudaFreeAsync(d_input,  stream);
 *
 * --- CUDA context / threading ----------------------------------------
 * The library runs in the CALLER's CUDA context. slzCreate() must be
 * called with a CUDA context already current; it does not create one. A
 * single handle is not safe for concurrent calls on itself — give each
 * thread its own handle. Distinct handles are fully independent.
 */

#ifndef STREAMLZ_GPU_H
#define STREAMLZ_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Status codes --------------------------------------------------- */
typedef enum slzStatus_t {
    SLZ_SUCCESS                   = 0,
    SLZ_ERROR_INVALID_HANDLE      = 1,  /* handle is NULL or not from slzCreate */
    SLZ_ERROR_INVALID_ARG         = 2,  /* a NULL/zero argument that must not be */
    SLZ_ERROR_BUFFER_TOO_SMALL    = 3,  /* output or temp capacity insufficient */
    SLZ_ERROR_CORRUPT_FRAME       = 4,  /* decompress input is not a valid frame */
    SLZ_ERROR_UNSUPPORTED         = 5,  /* level / option not supported */
    SLZ_ERROR_CUDA                = 6,  /* an underlying CUDA call failed */
    SLZ_ERROR_OUT_OF_MEMORY       = 7,  /* a library-internal allocation failed */
    /* Reserved for the Vulkan backend (shared ABI so both backends speak
     * the same status vocabulary). The CUDA backend never returns either
     * of these today; the slot reservation prevents future Vulkan
     * additions from re-numbering. */
    SLZ_ERROR_DEVICE_LOST         = 8,  /* VK_ERROR_DEVICE_LOST — terminal,
                                         * VkDevice must be rebuilt */
    SLZ_ERROR_VK_FEATURE_MISSING  = 9,  /* required Vulkan feature/extension
                                         * not present (subgroup_size_control,
                                         * BDA, shaderInt64, ...) — init-time */
} slzStatus_t;

/* Human-readable text for a status code. Never NULL; the returned string
 * is valid for the lifetime of the process. */
const char* slzStatusString(slzStatus_t status);

/* Null-terminated library version string, e.g. "3.0.0". */
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
    int level;                /* 1..5 — higher = smaller output, slower encode */
    int enable_profiling;     /* 1 = capture per-kernel timings (see
                               * slzGetLastTimings); 0 = off. */
    int effective_level_out;  /* OUT: post-clamp level the backend actually
                               * used. CUDA writes opts.level (no-op — the
                               * CUDA backend never clamps). The Vulkan
                               * Tier-2 backend writes the post-clamp level
                               * when mobile-VRAM pressure forces L5->L3.
                               * Callers MUST read this back after the
                               * call and compare goldens against the
                               * clamped level, not the requested level. */
    int dictionary_id;        /* Preset dictionary ID (0 = none). Built-in
                               * IDs: 1=json 2=html 3=text 4=xml 5=css 6=js
                               * 7=general. The decoder resolves the same ID
                               * from the frame header — both sides must
                               * know the dictionary. Unknown IDs fail with
                               * SLZ_ERROR_UNSUPPORTED. Occupies a formerly
                               * reserved (must-be-zero) slot, so pre-dict
                               * callers are automatically dictionary-less. */
    int reserved[4];          /* must be zero — reserved for future options */
} slzCompressOpts_t;

/* Default options (level 5, profiling off). */
slzCompressOpts_t slzCompressDefaultOpts(void);

/* ---- Decompression options ------------------------------------------ */
typedef struct slzDecompressOpts_t {
    int enable_profiling;  /* 1 = capture per-kernel timings; 0 = off. */
    int reserved[7];       /* must be zero — reserved for future options */
} slzDecompressOpts_t;

/* Default options (profiling off). */
slzDecompressOpts_t slzDecompressDefaultOpts(void);

/* ---- Per-kernel profiling ------------------------------------------- */
/* Returned by slzGetLastTimings when enable_profiling was set on the
 * most recent compress/decompress call. `name` is a static null-terminated
 * string valid for the library's lifetime; `ms` is wall-clock kernel time
 * measured with cudaEvent_t.
 *
 * --- Pseudo-kernel timing slots -------------------------------------
 * Two reserved `name` values are NOT real kernel timings — they carry
 * an out-of-band byte count in the `ms` field as a `uint32_t`
 * reinterpret-cast of the float bits:
 *
 *   "__compressed_size__"   — actual encoded byte count (matches what
 *                              the caller would have read from
 *                              `compressed_size` in pre-3.0). Emitted
 *                              by slzCompressAsync after the assemble-
 *                              measure kernel drains.
 *   "__decompressed_size__" — actual decoded byte count. Emitted by
 *                              slzDecompressAsync for parity.
 *
 * Decoded as:
 *   uint32_t size_bytes;
 *   memcpy(&size_bytes, &kt.ms, sizeof(size_bytes));
 *
 * Both slots are present iff `enable_profiling` was set on the call;
 * they are skipped on profiling-off runs because the size is delivered
 * elsewhere (caller-known output_size for decompress; the caller still
 * has the worst-case `max_compressed_size` for compress and may bounce
 * the actual frame length out of `compressed_size_out` when that
 * parameter survives v1.0 deprecation).
 *
 * Struct size is locked at 2 * sizeof(void*) = 16 B on LP64/LLP64; the
 * pseudo-kernel scheme deliberately reuses the existing `ms` slot
 * (float-bits aliased to uint32) so the array element layout does not
 * change. Adding a union for the size would have widened the struct
 * and broken every C caller that walks the array on stride. */
typedef struct slzKernelTiming_t {
    const char* name;
    float ms;
} slzKernelTiming_t;

/* Reserved pseudo-kernel name constants. Compare via string equality;
 * the library guarantees the same static pointer is used for every
 * timing entry, so `kt.name == SLZ_PSEUDO_KERNEL_COMPRESSED_SIZE`
 * pointer-compare is valid as well. */
#define SLZ_PSEUDO_KERNEL_COMPRESSED_SIZE   "__compressed_size__"
#define SLZ_PSEUDO_KERNEL_DECOMPRESSED_SIZE "__decompressed_size__"

/* Retrieve the per-kernel timings captured during the most recent
 * compress/decompress call on this handle. `timings` may be NULL to query
 * the count only. On return, *count is the number of timings the library
 * actually has; only min(capacity, *count) entries are written into
 * `timings`. Returns SLZ_SUCCESS even when capacity < *count.
 *
 * You must cudaStreamSynchronize the stream you passed to slz*Async
 * BEFORE calling this — otherwise the cuEvent pairs have not completed
 * and their timings will be reported as 0. Use slzWaitAndGetLastTimings
 * to do the sync + drain in one call. */
slzStatus_t slzGetLastTimings(slzHandle_t handle,
                              slzKernelTiming_t* timings, size_t capacity,
                              size_t* count);

/* Async convenience: cudaStreamSynchronize(stream) + slzGetLastTimings
 * in one call. Pass `stream` = NULL to skip the sync (equivalent to
 * slzGetLastTimings directly; use when you've already synchronized).
 * Otherwise `stream` is a CUstream / cudaStream_t cast to void*, the
 * same stream you passed to slz*Async. */
slzStatus_t slzWaitAndGetLastTimings(slzHandle_t handle, void* stream,
                                     slzKernelTiming_t* timings,
                                     size_t capacity, size_t* count);

/* ---- Decompress-size discovery -------------------------------------- */
/* Synchronous, pure-host call: parses the frame header at `host_bytes`
 * and writes the decompressed byte count to *output_size. No GPU
 * involvement; ~1 us.
 *
 * Contract: `host_bytes` must point to at least 64 bytes of valid
 * StreamLZ frame data (or the entire frame if it is shorter). The
 * library only reads what it needs from the leading header — up to
 * ~26 bytes for any frame we produce — but a frame shorter than the
 * required header field is rejected with SLZ_ERROR_CORRUPT_FRAME.
 * In practice every caller has the whole compressed frame in host RAM
 * by this point, so the 64-byte minimum is trivially satisfied.
 *
 * Use this when the caller doesn't know the decompressed size from
 * their own outer metadata. Skip it (and use your own size) when you
 * do. */
slzStatus_t slzGetDecompressedSize(slzHandle_t handle,
                                   const void* host_bytes,
                                   size_t* output_size);

/* Worst-case compressed-frame size for `input_size` input bytes. The
 * bound is level-independent — callers may pass slzCompressDefaultOpts()
 * or any per-call configuration without affecting the result. Sync,
 * pure-host call. Use it to size d_output for slzCompressAsync. */
slzStatus_t slzCompressBound(slzHandle_t handle, size_t input_size,
                             slzCompressOpts_t opts, size_t* max_output_size);

/* ---- Async (stream-taking) entry points ----------------------------- */
/* slzCompressAsync and slzDecompressAsync queue all GPU work on the
 * caller's `stream` and return — the caller's cudaStreamSynchronize is
 * the only sync point. `stream` is a CUDA driver-API CUstream
 * (== cudaStream_t) cast to void*; NULL is the default stream (stream 0).
 *
 * The library does NOT do any cudaMalloc / cudaMemcpy / cudaFree of the
 * caller-facing buffers (d_input, d_output, d_frame). Caller is
 * responsible for all of those, using their own stream and memory
 * management. The library only does the work between H2D-of-input and
 * the moment results are device-resident in d_output.
 *
 * COMPRESS INPUT-SIZE LIMITS: slzCompressAsync accepts input_size in
 * (128 B, 1 GiB] and returns SLZ_ERROR_UNSUPPORTED outside that range.
 * At or below 128 B the encoder would emit the header-dominated
 * uncompressed-body form (use slzCompressHost for tiny payloads);
 * above 1 GiB (16384 sub-chunks x 64 KB) the device-resident assembly
 * path's chunk cap is the ceiling. Segment larger payloads at the
 * application layer — 1 GiB segments compress and decode independently
 * with no measurable ratio loss at that scale. */

/* Decompress a StreamLZ frame whose bytes are device-resident at
 * d_frame into d_output. `output_size` MUST equal the value returned
 * by slzGetDecompressedSize for this frame (or be the caller's own
 * accurate size from their outer metadata). The library will write
 * exactly `output_size` bytes into d_output.
 *
 *   d_frame       device pointer to the compressed frame
 *   frame_size    bytes at d_frame
 *   d_output      device output buffer, must be >= output_size bytes
 *   output_size   exact decompressed byte count (caller-supplied)
 *   stream        caller's CUstream (void*); NULL = default stream
 *
 * Queue all decompress work on `stream`, then return. */
slzStatus_t slzDecompressAsync(slzHandle_t handle,
                               const void* d_frame, size_t frame_size,
                               void* d_output, size_t output_size,
                               slzDecompressOpts_t opts,
                               void* stream);

/* Compress `input_size` device bytes at d_input into a StreamLZ frame at
 * d_output.
 *
 *   d_input         device pointer to raw input
 *   input_size      bytes at d_input
 *   d_output        device output buffer, must be >= max_compressed_size
 *                   (use slzCompressBound to size this)
 *   max_compressed  capacity of d_output
 *   compressed_size HOST pointer. RETAINED for ABI compatibility but
 *                   the contract has changed in 3.0.0: this parameter
 *                   is NOT written by slzCompressAsync. The actual
 *                   encoded byte count is produced by the on-GPU
 *                   slzAssembleMeasure kernel after the async
 *                   submission completes; reading `*compressed_size`
 *                   on return yields whatever value the caller pre-
 *                   stored (typically zero). To retrieve the real
 *                   size, set `opts.enable_profiling = 1` and, after
 *                   syncing the user stream, call slzGetLastTimings
 *                   (or slzWaitAndGetLastTimings); look up the
 *                   pseudo-kernel entry whose `name` equals
 *                   SLZ_PSEUDO_KERNEL_COMPRESSED_SIZE and reinterpret
 *                   its `ms` field as `uint32_t` for the byte count.
 *                   Migration: existing 2.x callers must move to the
 *                   drain path. The synchronous slzCompressHost
 *                   wrapper continues to write `output_size`
 *                   normally — it does the stream sync internally.
 *   stream          caller's CUstream (void*); NULL = default stream
 *
 * Queue all GPU compress work on `stream`, then return.
 *
 * NOTE on stack usage: the async entry points run inline on the
 * caller's thread (no internal worker spawn), and the compress
 * orchestration has multi-MB host stack frames. Callers on small-
 * stack threads (default thread pools, libuv workers, etc.) should
 * either size the thread stack to at least 32 MiB or use the
 * synchronous slzCompressHost/slzDecompressHost entry points, which
 * spawn an internal worker with a 32 MiB stack. */
slzStatus_t slzCompressAsync(slzHandle_t handle,
                             const void* d_input, size_t input_size,
                             void* d_output, size_t max_compressed_size,
                             size_t* compressed_size,
                             slzCompressOpts_t opts,
                             void* stream);

/* ---- Compress / decompress: host -> host convenience ---------------- */
/* Synchronous, all-CUDA-internal wrappers. Caller hands the library
 * host buffers; the library does the H2D, the GPU work, the D2H, and
 * the cleanup. One extra host<->device round-trip vs the async D2D
 * entry points above; useful for CLI tools, ad-hoc decompression, and
 * code that doesn't need overlap. */
slzStatus_t slzCompressHost(slzHandle_t handle,
                            const void* input, size_t input_size,
                            void* output, size_t output_capacity,
                            size_t* output_size,
                            slzCompressOpts_t opts);

slzStatus_t slzDecompressHost(slzHandle_t handle,
                              const void* frame, size_t frame_size,
                              void* output, size_t output_capacity,
                              size_t* output_size,
                              slzDecompressOpts_t opts);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* STREAMLZ_GPU_H */
