// 1:1 port of src/decode/lz_decode_general.cuh.
//
// General entropy-capable LZ decoder (L2+). Subgroup-cooperative variant
// of the CUDA warp-cooperative path: handles off32, delta literals, and
// multi-block sub-chunks (up to MAX_BLOCKS_PER_SUBCHUNK 64 KB LZ blocks
// each). Mode bits = (sub-chunk-header >> 19) & 0xF.

#ifndef SRCVK_DECODE_LZ_DECODE_GENERAL_GLSL
#define SRCVK_DECODE_LZ_DECODE_GENERAL_GLSL

#include "lz_decode_core.glsl"

// CUDA reference: src/decode/lz_decode_general.cuh:33-39 (TokenType enum).
// Token-type tags assigned by the lane-0 parser and consumed after the
// subgroup broadcast.
const uint TOKEN_TYPE_SHORT        = 0u;
const uint TOKEN_TYPE_LONG_LITERAL = 1u;
const uint TOKEN_TYPE_LONG_NEAR    = 2u;
const uint TOKEN_TYPE_SHORT_FAR    = 3u;
const uint TOKEN_TYPE_LONG_FAR     = 4u;

// CUDA reference: src/decode/lz_decode_general.cuh:21-25 (DecodeOutput).
// Per-call destination region for decodeSubChunkGeneral. The decoder
// only writes inside `dst[dst_offset .. dst_offset + dst_size)`.
//
// VK adaptation: the CUDA pointer field `dst` becomes an SSBO + base
// offset pair at the call site (mirrors the ParsedStreams adaptation).
// The struct here keeps the dst_size + dst_offset scalars; dst_ssbo
// is passed as a separate macro argument to decodeSubChunkGeneral_*.
struct DecodeOutput {
    uint dst_size;
    uint dst_offset;
};

// CUDA reference: src/decode/lz_decode_general.cuh:46
// (deltaLiteralCopyBounded). Declaration only at this point.
void deltaLiteralCopyBounded();

// CUDA reference: src/decode/lz_decode_general.cuh:77
// (decodeSubChunkGeneral<OFF16_SPLIT>). Two macro instantiations
// mirroring the CUDA template — same dispatch shape as
// decodeSubChunkRawMode_true / _false in lz_decode_raw.glsl.
//
// Args (both macros, all u32 unless noted):
//   comp_ssbo              - SSBO holding the literal / cmd / off16 / off32 /
//                            length streams (one SSBO; per-stream byte offsets
//                            are in ps_*_base args)
//   ps_*                   - fields of ParsedStreams (see slz_wire_format.glsl).
//                            cmd_ssbo, lit_ssbo, off16_ssbo, etc. are passed
//                            as a separate (ssbo, base) pair per stream.
//   off16_hi_ssbo / _lo_ssbo - hi/lo half SSBOs (used by _true; placeholders for _false)
//   dst_ssbo               - destination SSBO (writes go here)
//   out_dst_size, out_dst_offset - DecodeOutput fields
//   mode                   - sub-chunk mode bits (mode == 0 → delta literals)
//   lane                   - this invocation's lane id
#define decodeSubChunkGeneral_false(comp_ssbo, ps_lit_ptr, ps_lit_size,                \
                                    ps_cmd_ptr, ps_cmd_size,                           \
                                    ps_off16_raw, ps_off16_count,                      \
                                    off16_hi_ssbo, ps_off16_hi,                        \
                                    off16_lo_ssbo, ps_off16_lo,                        \
                                    ps_off32_raw1, ps_off32_count1,                    \
                                    ps_off32_raw2, ps_off32_count2,                    \
                                    ps_len_stream, ps_len_avail,                       \
                                    ps_off16_split,                                    \
                                    ps_cmd_stream2_offset, ps_initial_copy,            \
                                    dst_ssbo, out_dst_size, out_dst_offset,            \
                                    mode, lane)                                        \
    do { } while (false)

#define decodeSubChunkGeneral_true(comp_ssbo, ps_lit_ptr, ps_lit_size,                 \
                                   ps_cmd_ptr, ps_cmd_size,                            \
                                   ps_off16_raw, ps_off16_count,                       \
                                   off16_hi_ssbo, ps_off16_hi,                         \
                                   off16_lo_ssbo, ps_off16_lo,                         \
                                   ps_off32_raw1, ps_off32_count1,                     \
                                   ps_off32_raw2, ps_off32_count2,                     \
                                   ps_len_stream, ps_len_avail,                        \
                                   ps_off16_split,                                     \
                                   ps_cmd_stream2_offset, ps_initial_copy,             \
                                   dst_ssbo, out_dst_size, out_dst_offset,             \
                                   mode, lane)                                         \
    do { } while (false)

#endif
