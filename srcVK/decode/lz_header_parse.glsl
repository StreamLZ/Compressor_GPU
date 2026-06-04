// 1:1 port of src/decode/lz_header_parse.cuh.
//
// parseSubChunkHeaders + per-stream header parsers for the general
// (L2+) decode path. Reads every per-stream header of one sub-chunk
// (literal, token, off16, off32, length) and returns the resolved
// pointers and sizes in a ParsedStreams struct.

#ifndef SRCVK_DECODE_LZ_HEADER_PARSE_GLSL
#define SRCVK_DECODE_LZ_HEADER_PARSE_GLSL

#include "slz_wire_format.glsl"

// CUDA reference: src/decode/lz_header_parse.cuh:14 (broadcastSrc).
// Rebuilds a per-subgroup `src` cursor after lane-0 parses a header by
// broadcasting the resulting byte offset via subgroupBroadcastFirst.
uint broadcastSrc(uint sc_src_off, uint src_off);

// CUDA reference: src/decode/lz_header_parse.cuh:36
// (parseLiteralStreamHeader). chunk_type in {0, 4}: type 0 is a raw
// stream copied verbatim; type 4 is Huffman-coded.
void parseLiteralStreamHeader();

// CUDA reference: src/decode/lz_header_parse.cuh:77
// (parseCommandStreamHeader). Same dispatch shape as the literal stream.
void parseCommandStreamHeader();

// CUDA reference: src/decode/lz_header_parse.cuh:130 (parseOff16StreamHeader).
void parseOff16StreamHeader();

// CUDA reference: src/decode/lz_header_parse.cuh:188 (parseOff32StreamHeaders).
void parseOff32StreamHeaders();

// CUDA reference: src/decode/lz_header_parse.cuh:244 (parseSubChunkHeaders).
// Outer fn invoked by decodeSubChunkGeneral; gathers per-stream pointers
// and sizes into a ParsedStreams record.
//
// VK adaptation: macro form mirroring parseAndDecodeSubChunkRaw's
// adaptation in lz_dispatch.glsl. The args list mirrors the CUDA
// signature; `ps_out` is the ParsedStreams sink the caller declares
// on the stack and passes by name.
#define parseSubChunkHeaders(comp_ssbo, sc_src_off, sc_comp_size,                  \
                              sc_decomp_size, dst_ssbo, dst_offset, base_offset,   \
                              entropy_lit_ssbo, entropy_lit_base,                  \
                              entropy_tok_ssbo, entropy_tok_base,                  \
                              entropy_off16_ssbo, entropy_off16_base,              \
                              ps_out, lane)                                        \
    do { } while (false)

#endif
