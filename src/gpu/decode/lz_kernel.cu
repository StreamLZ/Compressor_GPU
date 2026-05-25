// ── StreamLZ GPU Decode Kernel ──────────────────────────────────
// Each warp (32 threads) decodes one StreamLZ chunk independently.
// Two warps (WARPS_PER_BLOCK) are packed per CUDA thread-block for SM
// occupancy (blockDim 32,2,1 — the two warps do NOT cooperate).
// Lane 0 of each warp parses tokens serially; all 32 lanes participate
// in literal and match byte copies.
//
// This is a thin aggregator translation unit: it #includes the split
// implementation headers below so the whole kernel still compiles as a
// SINGLE compilation unit. tools/build_gpu.bat compiles this one
// lz_kernel.cu to lz_kernel.ptx; decode/driver.zig @embedFile's that
// single PTX. The split is purely organizational — keeping each file
// under the size cap — and does not change the build or driver.
//
// Decode chain (see lz_decode_core.cuh / lz_header_parse.cuh):
//   decodeSubChunkRawMode   — single-LZ-block path (no off32, no delta)
//   decodeSubChunkGeneral   — full path (off32, delta literals, 2 blocks)
//   parseSubChunkHeaders    — parses per-stream headers into ParsedStreams
//   parseAndDecodeSubChunk  — general entropy-capable parse + dispatch
//   parseAndDecodeSubChunkRaw — raw L1/L2 inline parse + dispatch
//
// extern "C" __global__ kernels (bound by name from driver.zig):
//   slzLzDecodeKernel       — general entropy-capable kernel
//   slzLzDecodeRawKernel    — raw L1/L2 fast-path kernel
//   slzGatherRawOff16Kernel — raw off16 scatter (1 block per stream)

#include "slz_wire_format.cuh"           // constants, descriptor structs, header parsers
#include "lz_decode_core.cuh"            // decodeSubChunkRawMode / decodeSubChunkGeneral
#include "lz_header_parse.cuh"           // parseSubChunkHeaders
#include "lz_dispatch.cuh"               // parseAndDecodeSubChunk{,Raw}
#include "lz_decode_kernels.cuh"         // slzLzDecodeKernel, slzLzDecodeRawKernel
#include "gather_raw_off16_kernel.cuh"   // slzGatherRawOff16Kernel
#include "walk_frame_kernel.cuh"         // slzWalkFrameKernel + walkRead* + SLZ_FRAME_*
#include "compact_descs_kernels.cuh"     // SlzHuffDecChunkDesc / SlzScan{Huff,Raw}Desc + compact kernels
#include "merge_huff_descs_kernel.cuh"   // slzMergeHuffDescsKernel
#include "prefix_sum_chunks_kernel.cuh"  // slzPrefixSumChunksKernel
#include "scan_parse_kernel.cuh"         // slzScanParseKernel + scan* helpers
