// ── StreamLZ merge-huff-descs kernel ───────────────────────────
// Combines four per-stream compacted SlzHuffDecChunkDesc arrays into a
// single merged array with per-region out_offset adjustments and
// sequential lut_offsets across the union. Mirrors the CPU `append`
// loop in fullGpuLaunchImpl. Depends on SlzHuffDecChunkDesc and
// HUFF_LUT_ENTRIES from compact_descs_kernels.cuh. Included into the
// single lz_kernel.cu translation unit.
#pragma once

#include "compact_descs_kernels.cuh"

// ── Merge-huff-descs kernel ─────────────────────────────────────────
// Combines four per-stream compacted SlzHuffDecChunkDesc arrays into a
// single merged array, adding the per-stream region offset (lit=0,
// tok=tok_region, off16hi/lo=off16_region) to out_offset and assigning
// sequential lut_offsets across the union (lit slots 0..n_lit-1, tok
// slots n_lit..n_lit+n_tok-1, etc.). Mirrors the CPU `append` loop in
// fullGpuLaunchImpl.
//
// Stream regions (4 entries): each {src, n, region_off}. Each block
// handles one region in its lane-0; per-block prefix is computed via a
// pre-pass on lane 0.
// Append one region's compacted descs into d_merged, adding region_off
// to each desc's out_offset and assigning sequential lut_offsets.
static __device__ inline void appendRegion(
    SlzHuffDecChunkDesc* __restrict__       d_merged,
    uint32_t&                               lut_slot,
    const SlzHuffDecChunkDesc* __restrict__ src,
    uint32_t                                n,
    uint32_t                                region_off
) {
    for (uint32_t i = 0; i < n; i++) {
        SlzHuffDecChunkDesc d = src[i];
        d.out_offset += region_off;
        d.lut_offset = lut_slot * HUFF_LUT_ENTRIES;
        d_merged[lut_slot] = d;
        lut_slot++;
    }
}

extern "C" __global__ void slzMergeHuffDescsKernel(
    const SlzHuffDecChunkDesc* __restrict__ d_lit,
    const SlzHuffDecChunkDesc* __restrict__ d_tok,
    const SlzHuffDecChunkDesc* __restrict__ d_hi,
    const SlzHuffDecChunkDesc* __restrict__ d_lo,
    const uint32_t* __restrict__            d_n_lit,
    const uint32_t* __restrict__            d_n_tok,
    const uint32_t* __restrict__            d_n_hi,
    const uint32_t* __restrict__            d_n_lo,
    uint32_t                                tok_region_off,
    uint32_t                                off16_region_off,
    SlzHuffDecChunkDesc* __restrict__       d_merged,
    uint32_t* __restrict__                  d_n_merged)
{
    SLZ_GUARD_SINGLE_THREAD();
    uint32_t lut_slot = 0;
    appendRegion(d_merged, lut_slot, d_lit, *d_n_lit, 0);
    appendRegion(d_merged, lut_slot, d_tok, *d_n_tok, tok_region_off);
    appendRegion(d_merged, lut_slot, d_hi,  *d_n_hi,  off16_region_off);
    appendRegion(d_merged, lut_slot, d_lo,  *d_n_lo,  off16_region_off);
    *d_n_merged = lut_slot;
}
