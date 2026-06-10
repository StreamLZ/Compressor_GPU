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
// Append one region's compacted descs into d_merged, assigning sequential
// lut_offsets. A-024 fix: out_offset is NOT touched here. The per-region
// offset (0 for lit, tok_region_off for tok, off16_region_off for hi+lo)
// is applied as uint64_t inside slzHuffDecode4StreamKernel; the old u32
// `d.out_offset += region_off` truncated silently at ~6553 sub-chunks and
// corrupted lit slots near offset 0 in entropy_scratch.
static __device__ inline void appendRegion(
    SlzHuffDecChunkDesc* __restrict__       d_merged,
    uint32_t&                               lut_slot,
    const SlzHuffDecChunkDesc* __restrict__ src,
    uint32_t                                n
) {
    for (uint32_t i = 0; i < n; i++) {
        SlzHuffDecChunkDesc d = src[i];
        d.lut_offset = lut_slot * HUFF_LUT_ENTRIES;
        d_merged[lut_slot] = d;
        lut_slot++;
    }
}

// A-024 fix: tok_region_off / off16_region_off params kept for ABI
// compatibility (host still passes them) but IGNORED here. The region
// offsets are now applied as uint64_t inside slzHuffDecode4StreamKernel.
extern "C" __global__ void slzMergeHuffDescsKernel(
    const SlzHuffDecChunkDesc* __restrict__ d_lit,
    const SlzHuffDecChunkDesc* __restrict__ d_tok,
    const SlzHuffDecChunkDesc* __restrict__ d_hi,
    const SlzHuffDecChunkDesc* __restrict__ d_lo,
    const uint32_t* __restrict__            d_n_lit,
    const uint32_t* __restrict__            d_n_tok,
    const uint32_t* __restrict__            d_n_hi,
    const uint32_t* __restrict__            d_n_lo,
    uint32_t                                /*tok_region_off (ignored)*/,
    uint32_t                                /*off16_region_off (ignored)*/,
    SlzHuffDecChunkDesc* __restrict__       d_merged,
    uint32_t* __restrict__                  d_n_merged)
{
    SLZ_GUARD_SINGLE_THREAD();
    uint32_t lut_slot = 0;
    appendRegion(d_merged, lut_slot, d_lit, *d_n_lit);
    appendRegion(d_merged, lut_slot, d_tok, *d_n_tok);
    appendRegion(d_merged, lut_slot, d_hi,  *d_n_hi);
    appendRegion(d_merged, lut_slot, d_lo,  *d_n_lo);
    *d_n_merged = lut_slot;
}

// ── B2 (2026-06-10): 4-block parallel merge ─────────────────────────
// The serial kernel above walks lit→tok→hi→lo with one thread
// (~0.20 ms at enwik8-L5 scale). The four regions are independent
// once each region's base slot is known, and the bases are just
// prefix sums of the four counts — available to every block for the
// cost of reading 4 u32s. Same shape as the A-017 compact fusion:
// gridDim.x = 4 single-thread blocks, wall cost ~= the largest
// region. Block 0 additionally writes n_merged (the total).
extern "C" __global__ void slzMergeHuffDescsParKernel(
    const SlzHuffDecChunkDesc* __restrict__ d_lit,
    const SlzHuffDecChunkDesc* __restrict__ d_tok,
    const SlzHuffDecChunkDesc* __restrict__ d_hi,
    const SlzHuffDecChunkDesc* __restrict__ d_lo,
    const uint32_t* __restrict__            d_n_lit,
    const uint32_t* __restrict__            d_n_tok,
    const uint32_t* __restrict__            d_n_hi,
    const uint32_t* __restrict__            d_n_lo,
    uint32_t                                /*tok_region_off (ignored)*/,
    uint32_t                                /*off16_region_off (ignored)*/,
    SlzHuffDecChunkDesc* __restrict__       d_merged,
    uint32_t* __restrict__                  d_n_merged)
{
    if (threadIdx.x != 0) return;
    const uint32_t n0 = *d_n_lit, n1 = *d_n_tok, n2 = *d_n_hi, n3 = *d_n_lo;
    const uint32_t which = blockIdx.x;
    if (which > 3) return;

    const SlzHuffDecChunkDesc* src =
        (which == 0) ? d_lit :
        (which == 1) ? d_tok :
        (which == 2) ? d_hi  : d_lo;
    const uint32_t n =
        (which == 0) ? n0 :
        (which == 1) ? n1 :
        (which == 2) ? n2 : n3;
    uint32_t base = 0;
    if (which > 0) base += n0;
    if (which > 1) base += n1;
    if (which > 2) base += n2;

    for (uint32_t i = 0; i < n; i++) {
        SlzHuffDecChunkDesc d = src[i];
        const uint32_t slot = base + i;
        d.lut_offset = slot * HUFF_LUT_ENTRIES;
        d_merged[slot] = d;
    }
    if (which == 0) *d_n_merged = n0 + n1 + n2 + n3;
}
