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
//   slzFullDecompressL1Kernel    — general entropy-capable kernel
//   slzFullDecompressL1KernelRaw — raw L1/L2 fast-path kernel
//   slzGatherRawOff16Kernel      — raw off16 scatter (1 block per stream)

#include "slz_wire_format.cuh"   // constants, descriptor structs, header parsers
#include "lz_decode_core.cuh"    // decodeSubChunkRawMode / decodeSubChunkGeneral
#include "lz_header_parse.cuh"   // parseSubChunkHeaders
#include "lz_kernels.cuh"        // parse-and-dispatch helpers + the __global__ kernels
