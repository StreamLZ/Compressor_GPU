// ── StreamLZ raw off16 gather kernel ───────────────────────────
// Single-launch scatter of raw (type-0) off16 sub-streams from the
// compressed blob into the off16 scratch, replacing ~1500 host-issued
// device-to-device copies (per-call driver overhead was ~8 ms).
// Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"

// ── Raw off16 gather ────────────────────────────────────────────────
// Scatters the raw (type-0) off16 sub-streams from the compressed blob
// into the off16 scratch in a single launch — replacing ~1500
// host-issued device-to-device copies (the per-call driver overhead was
// ~8 ms). Descriptor layout matches the Zig RawOff16Desc
// {src_offset, size, gpu_offset}.
//
// Launch contract: over-launch with a safe upper bound; each block
// reads `*d_count` and self-gates on `blockIdx.x >= *d_count`. The
// driver passes a device-resident counter (e.g. d_compact_counts+16
// for the n_raw slot) so no per-call D2H of the count is needed.
// blockDim.x lanes stride the copy of one stream. No __launch_bounds__:
// the kernel is bandwidth-bound, so occupancy tuning is not needed.
extern "C" __global__ void slzGatherRawOff16Kernel(
    const uint8_t* __restrict__ comp_base,
    uint32_t comp_len,
    uint8_t* __restrict__ scratch_base,
    const SlzRawOff16Desc* __restrict__ descs,
    const uint32_t* __restrict__ d_count
) {
    const uint32_t i = blockIdx.x;
    if (i >= *d_count) return;
    const SlzRawOff16Desc d = descs[i];
    if (d.size == 0 || d.src_offset + d.size > comp_len) return;
    const uint8_t* s = comp_base + d.src_offset;
    uint8_t* t = scratch_base + d.gpu_offset;
    for (uint32_t j = threadIdx.x; j < d.size; j += blockDim.x)
        t[j] = s[j];
}
