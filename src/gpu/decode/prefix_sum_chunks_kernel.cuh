// ── StreamLZ prefix-sum-chunks kernel ──────────────────────────
// Computes the per-chunk first-sub-chunk index plus the total
// sub-chunk count on device. Single-threaded sequential sum - n is
// bounded by WALK_MAX_CHUNKS (16384), so trivial wall-time and not
// worth a parallel scan. Included into the single lz_kernel.cu
// translation unit.
#pragma once

#include "slz_wire_format.cuh"

// ── Prefix-sum-chunks kernel (roadmap 4d Phase 3 step 2) ───────────
// Computes the per-chunk first-sub-chunk index plus the total
// sub-chunk count, on device. Single-threaded sequential sum - n is
// bounded by WALK_MAX_CHUNKS (16384), so trivial wall-time and not
// worth a parallel scan. Replaces the CPU first_subchunk_idx loop in
// fullGpuLaunchImpl on the pure-D2D path.
extern "C" __global__ void slzPrefixSumChunksKernel(
    const SlzChunkDesc* __restrict__ d_chunks,
    uint32_t                         n_chunks,
    uint32_t                         sub_chunk_cap,
    uint32_t* __restrict__            d_first_sub_idx,
    uint32_t* __restrict__            d_total_subchunks)
{
    SLZ_GUARD_SINGLE_THREAD();
    const uint32_t n = n_chunks;
    uint32_t cap = sub_chunk_cap;
    if (cap == 0) cap = DEFAULT_SUB_CHUNK_CAP;
    uint32_t total = 0;
    for (uint32_t i = 0; i < n; i++) {
        d_first_sub_idx[i] = total;
        const SlzChunkDesc ch = d_chunks[i];
        const uint32_t n_subs = (ch.flags != 0 || ch.decomp_size == 0)
            ? 1u
            : (ch.decomp_size + cap - 1u) / cap;
        total += n_subs;
    }
    *d_total_subchunks = total;
}
