// 1:1 port of src/decode/lz_header_parse.cuh.
//
// parseSubChunkHeaders + per-stream header parsers for the general
// (L2+) decode path. Reads every per-stream header of one sub-chunk
// (literal, token, off16, off32, length) and returns the resolved
// byte offsets and sizes in a ParsedStreams struct.
//
// VK adaptation: the CUDA helpers each take `const uint8_t* sc_src` +
// inout `const uint8_t*& src`; in GLSL the pointers become byte offsets
// into the compressed SSBO (passed by name into every macro). Lane 0
// runs the wire-format walk; broadcastSrc rebuilds every lane's cursor
// from lane 0's via subgroupShuffle(_, 0).

#ifndef SRCVK_DECODE_LZ_HEADER_PARSE_GLSL
#define SRCVK_DECODE_LZ_HEADER_PARSE_GLSL

#include "slz_wire_format.glsl"

// CUDA reference: src/decode/lz_header_parse.cuh:14-19 (broadcastSrc).
// Rebuilds every lane's `src` cursor after lane 0 walked the header
// alone. CUDA broadcasts the byte-delta from sc_src; the GLSL form uses
// subgroupShuffle(_, 0) on the absolute byte offset (cheaper than the
// CUDA shape because GLSL has no pointer-arithmetic constraint that
// motivates the relative-delta encoding).
//
// VK adaptation: spelled as a macro so the call site can read/write the
// caller's local `src_cursor` u32 directly (GLSL has no out-param u32
// in non-uniform call sites without extra helper friction).
#define broadcastSrc(src_cursor) \
    do { (src_cursor) = subgroupShuffle((src_cursor), 0u); } while (false)

// CUDA reference: src/decode/lz_header_parse.cuh:36-75
// (parseLiteralStreamHeader). chunk_type in {0, 4}: type 0 is a raw
// stream copied verbatim; type 4 is Huffman-coded (pre-decoded into
// entropy_lit_scratch by slzHuffDecode4StreamKernel). Any other type is
// treated as an empty stream — `lit_ptr` is left at the current cursor
// and `lit_size` is zero.
//
// VK adaptation: macro form mirroring parseAndDecodeSubChunkRaw in
// lz_dispatch.glsl. (comp_ssbo, sc_src_off, src_cursor) replace the
// CUDA (sc_src, src&) pointer pair; (entropy_lit_ssbo, entropy_lit_base)
// replace `entropy_lit_scratch`. The caller's ParsedStreams instance
// fields are written by name (ps_lit_ptr, ps_lit_size).
#define parseLiteralStreamHeader(comp_ssbo, sc_src_off, src_cursor,                  \
                                  entropy_lit_ssbo, entropy_lit_base,                \
                                  has_entropy_lit,                                   \
                                  ps_lit_ptr, ps_lit_size,                            \
                                  ps_lit_in_scratch, lane)                            \
    do {                                                                              \
        uint _plsh_lit_ptr = (src_cursor);                                            \
        uint _plsh_lit_size = 0u;                                                     \
        uint _plsh_lit_pre = 0u;                                                      \
        if (uint(lane) == 0u) {                                                       \
            uint _plsh_b0 = uint((comp_ssbo)[(src_cursor) + 0u]);                     \
            uint _plsh_chunk_type = (_plsh_b0 >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK; \
            if (_plsh_chunk_type == 0u) {                                             \
                uint _plsh_sz;                                                        \
                parseRawStreamSize(comp_ssbo, src_cursor, _plsh_sz);                  \
                _plsh_lit_size = _plsh_sz;                                            \
                _plsh_lit_ptr = (src_cursor);                                         \
                (src_cursor) = (src_cursor) + _plsh_sz;                               \
            } else if (_plsh_chunk_type == HUFF_CHUNK_TYPE && uint(has_entropy_lit) != 0u) { \
                uint _plsh_comp_size;                                                 \
                uint _plsh_dst_size;                                                  \
                parseEntropyHeader(comp_ssbo, src_cursor, _plsh_comp_size, _plsh_dst_size); \
                _plsh_lit_size = _plsh_dst_size;                                      \
                _plsh_lit_pre = 1u;                                                   \
            } else {                                                                  \
                _plsh_lit_size = 0u;                                                  \
                _plsh_lit_ptr = (src_cursor);                                         \
            }                                                                          \
        }                                                                              \
        _plsh_lit_size = subgroupShuffle(_plsh_lit_size, 0u);                         \
        _plsh_lit_pre = subgroupShuffle(_plsh_lit_pre, 0u);                           \
        broadcastSrc(src_cursor);                                                     \
        if (_plsh_lit_pre != 0u) {                                                    \
            (ps_lit_ptr) = uint(entropy_lit_base);                                    \
        } else {                                                                       \
            (ps_lit_ptr) = (src_cursor) - _plsh_lit_size;                             \
        }                                                                              \
        (ps_lit_size) = _plsh_lit_size;                                               \
        (ps_lit_in_scratch) = _plsh_lit_pre;                                          \
    } while (false)

// CUDA reference: src/decode/lz_header_parse.cuh:77-128
// (parseCommandStreamHeader). Same dispatch shape as the literal stream
// (chunk_type 0 raw vs 4 Huffman). Also reads the optional
// block2_cmd_offset (only present when sub_chunk decomp > 64 KB).
//
// VK adaptation: macro form. (comp_ssbo, src_cursor) replace the CUDA
// (sc_src, src&); (entropy_tok_ssbo, entropy_tok_base) replace
// entropy_tok_scratch. The caller's ParsedStreams.cmd_ptr / cmd_size /
// cmd_stream2_offset fields are written by name.
#define parseCommandStreamHeader(comp_ssbo, sc_src_off, src_cursor,                  \
                                  sc_decomp_size,                                     \
                                  entropy_tok_ssbo, entropy_tok_base,                \
                                  has_entropy_tok,                                   \
                                  ps_cmd_ptr, ps_cmd_size,                            \
                                  ps_cmd_stream2_offset,                              \
                                  ps_cmd_in_scratch, lane)                            \
    do {                                                                              \
        uint _pcsh_cmd_ptr = (src_cursor);                                            \
        uint _pcsh_cmd_size = 0u;                                                     \
        uint _pcsh_cmd_pre = 0u;                                                      \
        if (uint(lane) == 0u) {                                                       \
            uint _pcsh_b0 = uint((comp_ssbo)[(src_cursor) + 0u]);                     \
            uint _pcsh_ct = (_pcsh_b0 >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;         \
            if (_pcsh_ct == 0u) {                                                     \
                uint _pcsh_sz;                                                        \
                parseRawStreamSize(comp_ssbo, src_cursor, _pcsh_sz);                  \
                _pcsh_cmd_size = _pcsh_sz;                                            \
                _pcsh_cmd_ptr = (src_cursor);                                         \
                (src_cursor) = (src_cursor) + _pcsh_sz;                               \
            } else if (_pcsh_ct == HUFF_CHUNK_TYPE && uint(has_entropy_tok) != 0u) {  \
                uint _pcsh_comp_size;                                                 \
                uint _pcsh_dst_size;                                                  \
                parseEntropyHeader(comp_ssbo, src_cursor, _pcsh_comp_size, _pcsh_dst_size); \
                _pcsh_cmd_size = _pcsh_dst_size;                                      \
                _pcsh_cmd_pre = 1u;                                                   \
            } else {                                                                  \
                /* CUDA reference: src/decode/lz_header_parse.cuh:102-109. */         \
                /* Unsupported chunk type — advance past the entropy stream via */    \
                /* the generic skip so subsequent parsers see correct offsets, */     \
                /* but signal an empty cmd to the caller. */                          \
                uint _pcsh_skip_dst;                                                  \
                skipEntropyStream(comp_ssbo, src_cursor, _pcsh_skip_dst);             \
                _pcsh_cmd_size = 0u;                                                  \
            }                                                                          \
        }                                                                              \
        _pcsh_cmd_size = subgroupShuffle(_pcsh_cmd_size, 0u);                         \
        _pcsh_cmd_pre = subgroupShuffle(_pcsh_cmd_pre, 0u);                           \
        broadcastSrc(src_cursor);                                                     \
        if (_pcsh_cmd_pre != 0u) {                                                    \
            (ps_cmd_ptr) = uint(entropy_tok_base);                                    \
        } else {                                                                       \
            (ps_cmd_ptr) = (src_cursor) - _pcsh_cmd_size;                             \
        }                                                                              \
        (ps_cmd_size) = _pcsh_cmd_size;                                               \
        (ps_cmd_in_scratch) = _pcsh_cmd_pre;                                          \
                                                                                       \
        /* CUDA reference: src/decode/lz_header_parse.cuh:119-127. */                 \
        /* block2_cmd_offset present only when sub-chunk spans 2 LZ blocks. */        \
        uint _pcsh_b2 = _pcsh_cmd_size;                                               \
        if (uint(lane) == 0u && uint(sc_decomp_size) > LZ_BLOCK_SIZE) {               \
            uint _pcsh_b2l = uint((comp_ssbo)[(src_cursor) + 0u]);                     \
            uint _pcsh_b2h = uint((comp_ssbo)[(src_cursor) + 1u]);                     \
            _pcsh_b2 = readU16LE(_pcsh_b2l, _pcsh_b2h);                                \
            (src_cursor) = (src_cursor) + 2u;                                          \
        }                                                                              \
        _pcsh_b2 = subgroupShuffle(_pcsh_b2, 0u);                                     \
        broadcastSrc(src_cursor);                                                     \
        (ps_cmd_stream2_offset) = _pcsh_b2;                                           \
    } while (false)

// CUDA reference: src/decode/lz_header_parse.cuh:130-182 (parseOff16StreamHeader).
// Reads the off16 count header. When the count == OFF16_ENTROPY_MARKER
// the off16 stream is Huffman-coded as two split streams (hi/lo bytes);
// the entropy off16 scratch holds the pre-decoded hi bytes at offset 0
// and lo bytes at +OFF16_HILO_SPLIT_OFFSET.
//
// VK adaptation: macro form. Writes ps_off16_raw / ps_off16_hi /
// ps_off16_lo / ps_off16_count / ps_off16_split.
#define parseOff16StreamHeader(comp_ssbo, sc_src_off, src_cursor,                    \
                                entropy_off16_ssbo, entropy_off16_base,              \
                                has_entropy_off16,                                   \
                                ps_off16_raw, ps_off16_hi, ps_off16_lo,              \
                                ps_off16_count, ps_off16_split, lane)                 \
    do {                                                                              \
        uint _po16_count = 0u;                                                        \
        uint _po16_is_entropy = 0u;                                                   \
        uint _po16_is_split = 0u;                                                     \
        if (uint(lane) == 0u) {                                                       \
            uint _po16_b0 = uint((comp_ssbo)[(src_cursor) + 0u]);                     \
            uint _po16_b1 = uint((comp_ssbo)[(src_cursor) + 1u]);                     \
            uint _po16_cnt = readU16LE(_po16_b0, _po16_b1);                            \
            if (_po16_cnt == OFF16_ENTROPY_MARKER && uint(has_entropy_off16) != 0u) { \
                /* CUDA reference: src/decode/lz_header_parse.cuh:145-159. */         \
                /* Entropy-coded off16: skip the 2 encoded sub-streams; the */        \
                /* pre-decoded hi/lo bytes live in entropy_off16_scratch at */        \
                /* offsets 0 and +OFF16_HILO_SPLIT_OFFSET respectively. */            \
                (src_cursor) = (src_cursor) + 2u;                                     \
                uint _po16_hi_size;                                                   \
                uint _po16_lo_size;                                                   \
                skipEntropyStream(comp_ssbo, src_cursor, _po16_hi_size);              \
                skipEntropyStream(comp_ssbo, src_cursor, _po16_lo_size);              \
                _po16_count = _po16_hi_size;                                          \
                if (_po16_lo_size != _po16_hi_size) {                                 \
                    _po16_count = (_po16_lo_size < _po16_hi_size) ?                   \
                                   _po16_lo_size : _po16_hi_size;                     \
                }                                                                      \
                _po16_is_entropy = 1u;                                                \
                _po16_is_split = 1u;                                                  \
            } else {                                                                  \
                _po16_count = _po16_cnt;                                              \
                (src_cursor) = (src_cursor) + 2u + _po16_count * OFF16_ENTRY_BYTES;   \
            }                                                                          \
        }                                                                              \
        _po16_count = subgroupShuffle(_po16_count, 0u);                               \
        _po16_is_entropy = subgroupShuffle(_po16_is_entropy, 0u);                     \
        _po16_is_split = subgroupShuffle(_po16_is_split, 0u);                         \
        broadcastSrc(src_cursor);                                                     \
        if (_po16_is_entropy != 0u) {                                                 \
            (ps_off16_hi) = uint(entropy_off16_base);                                 \
            (ps_off16_lo) = uint(entropy_off16_base) + OFF16_HILO_SPLIT_OFFSET;       \
            (ps_off16_raw) = 0u;                                                      \
        } else {                                                                       \
            (ps_off16_raw) = (src_cursor) - _po16_count * OFF16_ENTRY_BYTES;          \
            (ps_off16_hi) = 0u;                                                       \
            (ps_off16_lo) = 0u;                                                       \
        }                                                                              \
        (ps_off16_count) = _po16_count;                                               \
        (ps_off16_split) = _po16_is_split;                                            \
    } while (false)

// CUDA reference: src/decode/lz_header_parse.cuh:188-230
// (parseOff32StreamHeaders). Reads packed off32 count1+count2 header,
// computes pointers, and yields the length-stream offset + remaining
// bytes. CUDA splits this from off16 parsing so the shared lane-0 cursor
// walk runs uninterrupted; the GLSL form follows the same shape.
//
// Writes ps_off32_raw1 / ps_off32_count1 / ps_off32_raw2 / ps_off32_count2
// / ps_len_stream / ps_len_avail.
#define parseOff32StreamHeaders(comp_ssbo, sc_src_off, src_cursor, src_end_off,      \
                                 ps_off32_raw1, ps_off32_count1,                     \
                                 ps_off32_raw2, ps_off32_count2,                     \
                                 ps_len_stream, ps_len_avail, lane)                   \
    do {                                                                              \
        uint _po32_count1 = 0u;                                                       \
        uint _po32_count2 = 0u;                                                       \
        uint _po32_len_avail = 0u;                                                    \
        if (uint(lane) == 0u) {                                                       \
            uint _po32_b0 = uint((comp_ssbo)[(src_cursor) + 0u]);                     \
            uint _po32_b1 = uint((comp_ssbo)[(src_cursor) + 1u]);                     \
            uint _po32_b2 = uint((comp_ssbo)[(src_cursor) + 2u]);                     \
            uint _po32_tmp = readLE24(_po32_b0, _po32_b1, _po32_b2);                   \
            (src_cursor) = (src_cursor) + 3u;                                         \
            if (_po32_tmp != 0u) {                                                    \
                _po32_count1 = _po32_tmp >> OFF32_COUNT1_SHIFT;                       \
                _po32_count2 = _po32_tmp & OFF32_COUNT2_MASK;                         \
                if (_po32_count1 == OFF32_COUNT_PACK_MAX) {                           \
                    uint _po32_e0 = uint((comp_ssbo)[(src_cursor) + 0u]);             \
                    uint _po32_e1 = uint((comp_ssbo)[(src_cursor) + 1u]);             \
                    _po32_count1 = readU16LE(_po32_e0, _po32_e1);                      \
                    (src_cursor) = (src_cursor) + 2u;                                 \
                }                                                                      \
                if (_po32_count2 == OFF32_COUNT_PACK_MAX) {                           \
                    uint _po32_f0 = uint((comp_ssbo)[(src_cursor) + 0u]);             \
                    uint _po32_f1 = uint((comp_ssbo)[(src_cursor) + 1u]);             \
                    _po32_count2 = readU16LE(_po32_f0, _po32_f1);                      \
                    (src_cursor) = (src_cursor) + 2u;                                 \
                }                                                                      \
                /* CUDA reference: src/decode/lz_header_parse.cuh:208-215. */         \
                /* off32_raw1 = src, then advance past block-0's triples; */          \
                /* off32_raw2 = src, then advance past block-1's triples. */          \
                (src_cursor) = (src_cursor)                                            \
                              + _po32_count1 * OFF32_ENTRY_BYTES                       \
                              + _po32_count2 * OFF32_ENTRY_BYTES;                      \
            }                                                                          \
            _po32_len_avail = uint(src_end_off) - (src_cursor);                       \
        }                                                                              \
        _po32_count1 = subgroupShuffle(_po32_count1, 0u);                             \
        _po32_count2 = subgroupShuffle(_po32_count2, 0u);                             \
        _po32_len_avail = subgroupShuffle(_po32_len_avail, 0u);                       \
        broadcastSrc(src_cursor);                                                     \
        (ps_off32_raw2) = (src_cursor) - _po32_count2 * OFF32_ENTRY_BYTES;            \
        (ps_off32_raw1) = (ps_off32_raw2) - _po32_count1 * OFF32_ENTRY_BYTES;         \
        (ps_off32_count1) = _po32_count1;                                             \
        (ps_off32_count2) = _po32_count2;                                             \
        (ps_len_stream) = (src_cursor);                                               \
        (ps_len_avail) = _po32_len_avail;                                             \
    } while (false)

// CUDA reference: src/decode/lz_header_parse.cuh:244-272
// (parseSubChunkHeaders). Outer fn invoked by decodeSubChunkGeneral;
// gathers per-stream pointers and sizes into a ParsedStreams record.
// The CUDA __noinline__ qualifier protects the decode hot loop from
// header-parse register pressure — GLSL has no equivalent, but the
// downstream decoder body is large enough that glslc spills/reloads
// the same way regardless.
//
// VK adaptation: macro form mirroring parseAndDecodeSubChunkRaw's
// adaptation in lz_dispatch.glsl. Each entropy_*_ssbo / _base pair
// flows into the per-stream parser; the (sc_src, src&) pointer pair
// becomes (comp_ssbo, sc_src_off) plus a local cursor.
//
// `entropy_lit_ssbo` / `entropy_tok_ssbo` / `entropy_off16_ssbo` are
// ALL the same SSBO (the kernel's EntropyScratchBuf) but kept as
// distinct macro args to mirror the CUDA signature 1:1; the
// has_entropy_* flags gate the chunk-type-4 path when the kernel was
// launched without entropy scratch (n_huff == 0).
#define parseSubChunkHeaders(comp_ssbo, sc_src_off, sc_comp_size,                    \
                              sc_decomp_size, dst_ssbo, dst_offset, base_offset,     \
                              entropy_lit_ssbo, entropy_lit_base,                    \
                              entropy_tok_ssbo, entropy_tok_base,                    \
                              entropy_off16_ssbo, entropy_off16_base,                \
                              has_entropy,                                            \
                              ps_out, lane)                                           \
    do {                                                                              \
        uint _psch_src = uint(sc_src_off);                                            \
        uint _psch_src_end = uint(sc_src_off) + uint(sc_comp_size);                   \
                                                                                       \
        (ps_out).initial_copy = 0u;                                                   \
        if (uint(base_offset) == 0u) {                                                \
            /* CUDA reference: src/decode/lz_header_parse.cuh:261-266. */             \
            /* First 8 bytes are raw literals copied straight to output. */           \
            if (uint(lane) < INITIAL_LITERAL_COPY_BYTES) {                            \
                (dst_ssbo)[uint(dst_offset) + uint(lane)] =                            \
                    (comp_ssbo)[_psch_src + uint(lane)];                              \
            }                                                                          \
            subgroupBarrier();                                                        \
            subgroupMemoryBarrierBuffer();                                            \
            _psch_src += INITIAL_LITERAL_COPY_BYTES;                                  \
            (ps_out).initial_copy = INITIAL_LITERAL_COPY_BYTES;                       \
        }                                                                              \
                                                                                       \
        parseLiteralStreamHeader(comp_ssbo, sc_src_off, _psch_src,                    \
                                  entropy_lit_ssbo, entropy_lit_base,                 \
                                  has_entropy,                                        \
                                  (ps_out).lit_ptr, (ps_out).lit_size,                \
                                  (ps_out).lit_in_scratch, lane);                     \
        parseCommandStreamHeader(comp_ssbo, sc_src_off, _psch_src,                    \
                                  sc_decomp_size,                                     \
                                  entropy_tok_ssbo, entropy_tok_base,                 \
                                  has_entropy,                                        \
                                  (ps_out).cmd_ptr, (ps_out).cmd_size,                \
                                  (ps_out).cmd_stream2_offset,                        \
                                  (ps_out).cmd_in_scratch, lane);                     \
        parseOff16StreamHeader(comp_ssbo, sc_src_off, _psch_src,                      \
                                entropy_off16_ssbo, entropy_off16_base,               \
                                has_entropy,                                          \
                                (ps_out).off16_raw, (ps_out).off16_hi,                \
                                (ps_out).off16_lo, (ps_out).off16_count,              \
                                (ps_out).off16_split, lane);                          \
        parseOff32StreamHeaders(comp_ssbo, sc_src_off, _psch_src, _psch_src_end,      \
                                 (ps_out).off32_raw1, (ps_out).off32_count1,          \
                                 (ps_out).off32_raw2, (ps_out).off32_count2,          \
                                 (ps_out).len_stream, (ps_out).len_avail, lane);      \
    } while (false)

#endif
