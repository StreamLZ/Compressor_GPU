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

// CUDA reference: src/encode/lz_chain_parser.cuh:38-45. isLazyMatchBetter:
// returns true if `cand` (found at pos + step) is better than `current`.
// Port of CPU isLazyMatchBetter.
bool isLazyMatchBetter(ChainMatch cand, ChainMatch current, int step) {
    return false;
}

// CUDA reference: src/encode/lz_chain_parser.cuh:49-56. isMatchBetter:
// accounts for near/far offset cost when comparing two candidate matches.
bool isMatchBetter(int match_length, int match_offset,
                   int best_length, int best_offset) {
    return false;
}

// CUDA reference: src/encode/lz_chain_parser.cuh:60-65. isBetterThanRecentMatch:
// returns true if the hash match is preferable to the recent-offset match.
bool isBetterThanRecentMatch(int recent_match_length, int match_length, int match_offset) {
    return false;
}

// CUDA reference: src/encode/lz_chain_parser.cuh:83-267. findMatchChain:
// chain-hash match finder for the lazy parser. Port of CPU
// findMatchWithChainHasher. Called by lane 0 only.
//
// VK adaptation: macro form so the SSBO buffers (input + 3 hash tables)
// remain visible at the call site; GLSL cannot pass SSBOs as function
// arguments. Writes its (length, offset) result into the in-scope
// `out_match_length` + `out_match_offset` int identifiers.
#define findMatchChain(src_buf, src_base, src_size_in,                                              \
                       pos_in, recent_offset_in,                                                    \
                       ht_buf, first_hash_base, long_hash_base, next_hash_base,                     \
                       hash_bits_in, hash_mask_in,                                                  \
                       end_pos_in, lit_run_length_in,                                               \
                       out_match_length, out_match_offset)                                          \
    do { } while (false)

// CUDA reference: src/encode/lz_chain_parser.cuh:274-302. insertChainRange:
// insert positions [from..to) into the chain hash tables. Used after
// emitting a match to populate hash entries for the matched byte range.
//
// VK adaptation: macro form so the input + hash-table SSBOs stay visible.
#define insertChainRange(src_buf, src_base, src_size_in,                                            \
                         from_in, to_in,                                                            \
                         ht_buf, first_hash_base, long_hash_base, next_hash_base,                   \
                         hash_bits_in, hash_mask_in)                                                \
    do { } while (false)

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
