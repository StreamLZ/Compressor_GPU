// 1:1 port of src/encode/lz_chain_parser.cuh.
// Serial chain-hash lazy LZ parser (use_chain==1 path). L5 surface.
// See srcVK/PortInstructions.md.

#ifndef SRCVK_ENCODE_LZ_CHAIN_PARSER_GLSL
#define SRCVK_ENCODE_LZ_CHAIN_PARSER_GLSL

#include "lz_format.glsl"
#include "lz_token_emit.glsl"

// CUDA reference: src/encode/lz_chain_parser.cuh:16-32. Lazy-match cost
// weights and chain-parser tuning constants.
const int  LAZY_LEN_WEIGHT       = 5;
const int  LAZY_MATCH_PENALTY    = 5;
const int  LAZY_STEP_WEIGHT      = 4;
const int  NEAR_OFFSET_COST_BITS = 16;
const int  FAR_OFFSET_COST_BITS  = 32;
const int  OFFSET_CLASS_LEN_MARGIN = 5;
const int  RECENT_MATCH_MIN       = 2;
const int  RECENT_LEN_MARGIN      = 1;
const int  RECENT_LEN_MARGIN_FAR  = 4;
const uint LONG_LIT_RUN_THRESHOLD = 64u;
const uint CHAIN_MIN_LOOKAHEAD    = 5u;

// CUDA reference: src/encode/lz_chain_parser.cuh:312-440. scanBlockChain:
// serial chain-hash lazy parser for one block; lane 0 does all work.
// VK adaptation: macro form so the SSBO buffers (input + 3 hash tables +
// output substreams) remain visible at the call site; GLSL cannot pass
// SSBOs as function arguments.
#define scanBlockChain(src_buf, src_base, src_size_in,                                              \
                       ht_buf, first_hash_base, long_hash_base, next_hash_base,                     \
                       hash_bits_in, hash_mask_in,                                                  \
                       lit_buf, lit_base, lit_count,                                                \
                       token_buf, token_base, token_count,                                          \
                       off16_buf, off16_base, off16_count,                                          \
                       off32_buf, off32_base, off32_pos, off32_count,                               \
                       len_buf, len_base, len_count,                                                \
                       anchor_inout, recent_offset_inout,                                           \
                       start_pos_in, end_pos_in, block2_start_in,                                   \
                       lane_in)                                                                     \
    do { } while (false)

#endif
