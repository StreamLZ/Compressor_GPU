// ── StreamLZ scan-output compaction kernels ────────────────────
// Owns the staged-descriptor structs shared with the scan kernel
// (SlzScanHuffDesc, SlzScanRawDesc) and the final Huffman descriptor
// (SlzHuffDecChunkDesc), plus the two single-threaded compact kernels
// that drop invalid staged entries and assign sequential lut_offsets.
// Must be included before scan_parse_kernel.cuh and
// merge_huff_descs_kernel.cuh because they reference these structs.
// Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"
// Pulled in for HUFF_LUT_ENTRIES — the shared 4-stream Huffman wire format.
#include "../common/gpu_huffman.cuh"

// ── Compact-huff-descs / Compact-raw-descs (4d Phase 3 step 4) ─────
// Single-threaded device-side compaction of the scan kernel's staged
// arrays. Each (slzScanParseKernel-produced) staged entry has a `valid`
// flag; the compaction drops invalid entries and assigns a sequential
// lut_offset = slot * HUFF_LUT_ENTRIES (defined in common/gpu_huffman.cuh,
// mirrored as HUFF_LUT_ENTRIES in decode/driver.zig).
// Used by the pure-D2D pipeline to eliminate the CPU compaction loop
// that gpuScanChunks used to do after a D2H of the staged data.

// HuffDecChunkDesc output struct — must match the Zig HuffDecChunkDesc
// (and huffman_kernel.cu's local definition). 5 u32 = 20 bytes.
struct SlzHuffDecChunkDesc {
    uint32_t in_offset, in_size, out_offset, out_size, lut_offset;
};
static_assert(sizeof(SlzHuffDecChunkDesc) == 20, "ABI: keep in sync with decode/driver.zig");

// Staged scan output (used by the compact kernels and slzScanParseKernel
// below). chunk_type=4 Huffman stream descriptor — staged form:
// lut_offset is assigned by the compact step, not the scan kernel.
struct SlzScanHuffDesc { uint32_t in_offset, in_size, out_offset, out_size, valid; };
struct SlzScanRawDesc  { uint32_t src_offset, size, gpu_offset, valid; };
static_assert(sizeof(SlzScanHuffDesc) == 20, "ABI: keep in sync with decode/driver.zig");
static_assert(sizeof(SlzScanRawDesc) == 16, "ABI: keep in sync with decode/driver.zig");

extern "C" __global__ void slzCompactHuffDescsKernel(
    const SlzScanHuffDesc* __restrict__ d_staged,
    const uint32_t* __restrict__         d_total_subs,
    SlzHuffDecChunkDesc* __restrict__    d_out,
    uint32_t* __restrict__               d_n_out)
{
    SLZ_GUARD_SINGLE_THREAD();
    const uint32_t tot = *d_total_subs;
    uint32_t k = 0;
    for (uint32_t i = 0; i < tot; i++) {
        const SlzScanHuffDesc s = d_staged[i];
        if (s.valid != 0) {
            d_out[k].in_offset  = s.in_offset;
            d_out[k].in_size    = s.in_size;
            d_out[k].out_offset = s.out_offset;
            d_out[k].out_size   = s.out_size;
            d_out[k].lut_offset = k * HUFF_LUT_ENTRIES;
            k++;
        }
    }
    *d_n_out = k;
}

extern "C" __global__ void slzCompactRawDescsKernel(
    const SlzScanRawDesc* __restrict__ d_staged_hi,
    const SlzScanRawDesc* __restrict__ d_staged_lo,
    const uint32_t* __restrict__       d_total_subs,
    SlzRawOff16Desc* __restrict__      d_out,
    uint32_t* __restrict__             d_n_out)
{
    SLZ_GUARD_SINGLE_THREAD();
    const uint32_t tot = *d_total_subs;
    uint32_t k = 0;
    for (uint32_t i = 0; i < tot; i++) {
        const SlzScanRawDesc hi = d_staged_hi[i];
        if (hi.valid != 0) {
            d_out[k].src_offset = hi.src_offset;
            d_out[k].size       = hi.size;
            d_out[k].gpu_offset = hi.gpu_offset;
            k++;
        }
        const SlzScanRawDesc lo = d_staged_lo[i];
        if (lo.valid != 0) {
            d_out[k].src_offset = lo.src_offset;
            d_out[k].size       = lo.size;
            d_out[k].gpu_offset = lo.gpu_offset;
            k++;
        }
    }
    *d_n_out = k;
}
