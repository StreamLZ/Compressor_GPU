/*
 * StreamLZ C API
 *
 * Single-call compress/decompress for StreamLZ (SLZ1 frame format).
 * Link against libstreamlz.a (static) or streamlz.dll/.so (dynamic).
 * Requires libc (the library uses c_allocator internally).
 *
 * Thread Safety: All functions are completely thread-safe and re-entrant.
 * No global state is used. You may safely call slz_compress and
 * slz_decompress concurrently from multiple host threads.
 *
 * Build:  zig build lib -Doptimize=ReleaseFast
 */

#ifndef STREAMLZ_H
#define STREAMLZ_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Error codes. Returned as negative values from slz_compress/slz_decompress.
 * Success returns a non-negative byte count.
 */
#define SLZ_ERROR_DST_TOO_SMALL  (-1)  /* Output buffer too small */
#define SLZ_ERROR_CORRUPT        (-2)  /* Input is corrupted or invalid */
#define SLZ_ERROR_OOM            (-3)  /* Out of memory */
#define SLZ_ERROR_BAD_LEVEL      (-4)  /* Invalid compression level */
#define SLZ_ERROR_UNKNOWN        (-5)  /* Unclassified internal error */

#define SLZ_IS_ERROR(ret) ((ret) < 0)

/*
 * Compress `src_len` bytes from `src` into `dst`.
 *
 * `level` selects the compression level (1-11, clamped if out of range):
 *
 *   Level  Codec  Ratio*  Compress*    Decompress*   Notes
 *   ─────  ─────  ──────  ──────────   ───────────   ─────────────────────
 *   1      Fast   54.9%   4,200 MB/s   34,000 MB/s   Fastest. Greedy parser.
 *   2      Fast   54.1%      83 MB/s   20,000 MB/s   Greedy, u32 hash.
 *   3      Fast   53.4%      72 MB/s   11,000 MB/s   Greedy + entropy.
 *   4      Fast   51.0%      62 MB/s    6,500 MB/s   Greedy + entropy.
 *   5      Fast   43.4%      40 MB/s   11,000 MB/s   Chain parser + entropy.
 *   6      High   29.1%      45 MB/s   12,000 MB/s   Optimal parser, SC parallel.
 *   7      High   29.0%      35 MB/s   12,000 MB/s   Optimal parser, SC parallel.
 *   8      High   28.6%      19 MB/s   11,000 MB/s   Optimal + BT4 finder.
 *   9      High   27.4%     7.9 MB/s    2,200 MB/s   Optimal, 128 MB window.
 *   10     High   27.3%     7.5 MB/s    2,300 MB/s   Optimal, 128 MB window.
 *   11     High   25.6%     1.3 MB/s    1,400 MB/s   BT4, 128 MB window. Best ratio.
 *
 *   * Approximate, enwik8 100 MB, 24 threads, Arrow Lake-S.
 *
 * Returns the number of compressed bytes written to `dst` (>= 0),
 * or a negative SLZ_ERROR_* code on failure.
 *
 * `dst` must be at least `slz_compress_bound(src_len)` bytes.
 */
int slz_compress(const void *src, size_t src_len,
                 void *dst, size_t dst_len,
                 int level);

/*
 * Decompress an SLZ1 frame from `src` into `dst`.
 *
 * Returns the number of decompressed bytes written to `dst` (>= 0),
 * or a negative SLZ_ERROR_* code on failure.
 *
 * `dst` must be at least `slz_content_size(src, src_len)` bytes
 * plus 64 bytes of safe-space padding.
 */
int slz_decompress(const void *src, size_t src_len,
                   void *dst, size_t dst_len);

/*
 * Returns the maximum compressed size for a given input length.
 * Use this to allocate the `dst` buffer for `slz_compress`.
 */
size_t slz_compress_bound(size_t src_len);

/*
 * Read the content size from an SLZ1 frame header.
 * Returns 0 if the header is invalid or content size is not present.
 *
 * IMPORTANT: The value is read directly from the frame header and may
 * be attacker-controlled. Always compare against slz_max_content_size()
 * before allocating, or rely on slz_decompress() which enforces the cap
 * internally (returns 0 if the header claims more than 4 GB).
 */
uint64_t slz_content_size(const void *src, size_t src_len);

/*
 * Returns the maximum content size the decoder will accept (4 GB).
 * Frames claiming a larger uncompressed size are rejected by
 * slz_decompress() with a 0 return.
 */
uint64_t slz_max_content_size(void);

/*
 * Returns the library version string (e.g. "2.0.0").
 */
const char *slz_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* STREAMLZ_H */
