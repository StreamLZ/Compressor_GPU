// 1:1 port of src/decode/lz_dispatch.cuh.
// parseAndDecodeSubChunkRaw + parseAndDecodeSubChunk.

#ifndef SRCVK_DECODE_LZ_DISPATCH_GLSL
#define SRCVK_DECODE_LZ_DISPATCH_GLSL

#include "slz_wire_format.glsl"
#include "lz_decode_core.glsl"
#include "lz_decode_raw.glsl"
#include "lz_decode_general.glsl"
#include "lz_header_parse.glsl"

// CUDA reference: src/decode/lz_dispatch.cuh:23-165.
// `parseAndDecodeSubChunkRaw` in CUDA: parses each per-stream header on
// lane 0 (with broadcasts), then dispatches to decodeSubChunkRawMode_false
// (off32 empty) or decodeSubChunkGeneral_false (off32 non-empty).
//
// VK adaptation: GLSL cannot pass SSBOs as function parameters. The CUDA
// function takes `const uint8_t* sc_src` + `uint8_t* dst` pointer args
// resolved from the kernel's SSBOs. The port spells this as a function-
// like macro that resolves to (comp_ssbo, comp_base) and `dst_ssbo`
// names provided by the kernel's binding declarations.
//
// Args:
//   comp_ssbo  - the compressed-input SSBO (lit/cmd/off16/length read from here)
//   sc_src_off - byte offset within comp_ssbo where this sub-chunk starts
//   sc_comp_size - compressed payload bytes for this sub-chunk
//   sc_decomp_size - decompressed size for this sub-chunk
//   dst_ssbo   - destination SSBO
//   dst_offset - absolute write position
//   base_offset - frame-relative dst offset (==0 only for first sub-chunk in frame)
//   lane       - this invocation's lane id
#define parseAndDecodeSubChunkRaw(comp_ssbo, sc_src_off, sc_comp_size, sc_decomp_size, \
                                  dst_ssbo, dst_offset, base_offset, lane)             \
    do {                                                                               \
        uint _pd_src = uint(sc_src_off);                                               \
        uint _pd_src_end = uint(sc_src_off) + uint(sc_comp_size);                      \
                                                                                       \
        uint _pd_initial_copy = 0u;                                                    \
        if (uint(base_offset) == 0u) {                                                 \
            /* CUDA reference: src/decode/lz_dispatch.cuh:35-42. First 8 bytes */     \
            /* are raw literals copied directly to the output. */                      \
            if (uint(lane) < INITIAL_LITERAL_COPY_BYTES) {                             \
                (dst_ssbo)[uint(dst_offset) + uint(lane)] =                            \
                    (comp_ssbo)[_pd_src + uint(lane)];                                 \
            }                                                                          \
            subgroupBarrier();                                                         \
            subgroupMemoryBarrierBuffer();                                             \
            _pd_src += INITIAL_LITERAL_COPY_BYTES;                                     \
            _pd_initial_copy = INITIAL_LITERAL_COPY_BYTES;                             \
        }                                                                              \
                                                                                       \
        /* CUDA reference: src/decode/lz_dispatch.cuh:44-56. Literal stream */         \
        /* header (Type 0). Parsed on lane 0, then broadcast. */                       \
        uint _pd_lit_ptr = _pd_src;                                                    \
        uint _pd_lit_size = 0u;                                                        \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_b0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_b1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            uint _pd_b2 = uint((comp_ssbo)[_pd_src + 2u]);                              \
            Type0HdrFields _pd_h = parseType0HdrFields(_pd_b0, _pd_b1, _pd_b2);        \
            _pd_lit_size = _pd_h.size;                                                 \
            _pd_src += _pd_h.header_bytes;                                              \
            _pd_lit_ptr = _pd_src;                                                     \
            _pd_src += _pd_lit_size;                                                   \
        }                                                                              \
        _pd_lit_size = subgroupShuffle(_pd_lit_size, 0u);                              \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_lit_ptr = _pd_src - _pd_lit_size;                                          \
                                                                                       \
        /* CUDA reference: src/decode/lz_dispatch.cuh:58-70. Command stream. */        \
        uint _pd_cmd_ptr;                                                              \
        uint _pd_cmd_size = 0u;                                                        \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_c0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_c1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            uint _pd_c2 = uint((comp_ssbo)[_pd_src + 2u]);                              \
            Type0HdrFields _pd_hc = parseType0HdrFields(_pd_c0, _pd_c1, _pd_c2);       \
            _pd_cmd_size = _pd_hc.size;                                                \
            _pd_src += _pd_hc.header_bytes;                                             \
            _pd_cmd_ptr = _pd_src;                                                     \
            _pd_src += _pd_cmd_size;                                                   \
        }                                                                              \
        _pd_cmd_size = subgroupShuffle(_pd_cmd_size, 0u);                              \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_cmd_ptr = _pd_src - _pd_cmd_size;                                          \
                                                                                       \
        /* CUDA reference: src/decode/lz_dispatch.cuh:72-81. block2_cmd_offset */      \
        /* present only for sub-chunks > 64KB. Captured on lane 0 and broadcast */     \
        /* so the off32-nonzero else-branch can pass it to decodeSubChunkGeneral. */    \
        uint _pd_block2_cmd_offset = _pd_cmd_size;                                     \
        if (uint(lane) == 0u && uint(sc_decomp_size) > LZ_BLOCK_SIZE) {                \
            uint _pd_b2_lo = uint((comp_ssbo)[_pd_src + 0u]);                           \
            uint _pd_b2_hi = uint((comp_ssbo)[_pd_src + 1u]);                           \
            _pd_block2_cmd_offset = readU16LE(_pd_b2_lo, _pd_b2_hi);                    \
            _pd_src += 2u;                                                              \
        }                                                                              \
        _pd_block2_cmd_offset = subgroupShuffle(_pd_block2_cmd_offset, 0u);            \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
                                                                                       \
        /* CUDA reference: src/decode/lz_dispatch.cuh:83-95. Off16 stream */           \
        /* (raw — no entropy 0xFFFF marker handling on this path). */                  \
        uint _pd_off16_raw;                                                            \
        uint _pd_off16_count = 0u;                                                     \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_o0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_o1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            _pd_off16_count = readU16LE(_pd_o0, _pd_o1);                                \
            _pd_off16_raw = _pd_src + 2u;                                              \
            _pd_src += 2u + _pd_off16_count * OFF16_ENTRY_BYTES;                       \
        }                                                                              \
        _pd_off16_count = subgroupShuffle(_pd_off16_count, 0u);                        \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_off16_raw = _pd_src - _pd_off16_count * OFF16_ENTRY_BYTES;                 \
                                                                                       \
        /* CUDA reference: src/decode/lz_dispatch.cuh:97-131. Off32 stream sizes */    \
        /* (may be empty at sc<=0.25 with 64KB sub-chunks). */                         \
        uint _pd_off32_count1 = 0u, _pd_off32_count2 = 0u;                              \
        uint _pd_off32_raw1 = _pd_src, _pd_off32_raw2 = _pd_src;                        \
        uint _pd_len_stream;                                                            \
        uint _pd_len_avail = 0u;                                                       \
                                                                                       \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_t0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_t1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            uint _pd_t2 = uint((comp_ssbo)[_pd_src + 2u]);                              \
            uint _pd_tmp = readLE24(_pd_t0, _pd_t1, _pd_t2);                            \
            _pd_src += 3u;                                                              \
            if (_pd_tmp != 0u) {                                                       \
                _pd_off32_count1 = _pd_tmp >> OFF32_COUNT1_SHIFT;                       \
                _pd_off32_count2 = _pd_tmp & OFF32_COUNT2_MASK;                         \
                if (_pd_off32_count1 == OFF32_COUNT_PACK_MAX) {                        \
                    uint _pd_e0 = uint((comp_ssbo)[_pd_src + 0u]);                      \
                    uint _pd_e1 = uint((comp_ssbo)[_pd_src + 1u]);                      \
                    _pd_off32_count1 = readU16LE(_pd_e0, _pd_e1);                       \
                    _pd_src += 2u;                                                      \
                }                                                                       \
                if (_pd_off32_count2 == OFF32_COUNT_PACK_MAX) {                        \
                    uint _pd_f0 = uint((comp_ssbo)[_pd_src + 0u]);                      \
                    uint _pd_f1 = uint((comp_ssbo)[_pd_src + 1u]);                      \
                    _pd_off32_count2 = readU16LE(_pd_f0, _pd_f1);                       \
                    _pd_src += 2u;                                                      \
                }                                                                       \
                _pd_off32_raw1 = _pd_src;                                               \
                _pd_src += _pd_off32_count1 * OFF32_ENTRY_BYTES;                        \
                _pd_off32_raw2 = _pd_src;                                               \
                _pd_src += _pd_off32_count2 * OFF32_ENTRY_BYTES;                        \
            } else {                                                                    \
                _pd_off32_raw1 = _pd_src;                                               \
                _pd_off32_raw2 = _pd_src;                                               \
            }                                                                           \
            _pd_len_stream = _pd_src;                                                  \
            _pd_len_avail = _pd_src_end - _pd_src;                                     \
        }                                                                              \
        _pd_off32_count1 = subgroupShuffle(_pd_off32_count1, 0u);                      \
        _pd_off32_count2 = subgroupShuffle(_pd_off32_count2, 0u);                      \
        _pd_len_avail = subgroupShuffle(_pd_len_avail, 0u);                            \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_len_stream = _pd_src;                                                      \
        _pd_off32_raw2 = _pd_src - _pd_off32_count2 * OFF32_ENTRY_BYTES;                \
        _pd_off32_raw1 = _pd_off32_raw2 - _pd_off32_count1 * OFF32_ENTRY_BYTES;         \
                                                                                       \
        /* CUDA reference: src/decode/lz_dispatch.cuh:133-143. */                      \
        /* off32_count1 == 0 && off32_count2 == 0 → raw-mode L1 fast path. */          \
        /* OFF16_SPLIT=false: raw-off16 is the only off16 shape on this path. */       \
        if (_pd_off32_count1 == 0u && _pd_off32_count2 == 0u) {                        \
            decodeSubChunkRawMode_false(                                               \
                comp_ssbo, _pd_cmd_ptr, _pd_cmd_size,                                  \
                comp_ssbo, _pd_lit_ptr, _pd_lit_size,                                  \
                comp_ssbo, _pd_off16_raw, _pd_off16_count,                             \
                comp_ssbo, _pd_len_stream, _pd_len_avail,                              \
                dst_ssbo, sc_decomp_size, _pd_initial_copy, dst_offset, lane);         \
        } else {                                                                       \
            /* CUDA reference: src/decode/lz_dispatch.cuh:144-164. */                  \
            /* off32 non-empty → general decoder via positional brace-init of */       \
            /* ParsedStreams (OFF16_SPLIT=false; raw L1/L2 has no entropy off16). */   \
            /* VK adaptation: lit/cmd are ALWAYS in compressed on the raw path */      \
            /* (no Huffman scratch present); pass comp_ssbo as the scratch slot */     \
            /* placeholder + 0u as the in-scratch flag. */                              \
            decodeSubChunkGeneral_false(                                               \
                comp_ssbo,                                                             \
                _pd_lit_ptr, _pd_lit_size,                                             \
                _pd_cmd_ptr, _pd_cmd_size,                                             \
                _pd_off16_raw, _pd_off16_count,                                        \
                comp_ssbo, 0u,                                                         \
                comp_ssbo, 0u,                                                         \
                _pd_off32_raw1, _pd_off32_count1,                                      \
                _pd_off32_raw2, _pd_off32_count2,                                      \
                _pd_len_stream, _pd_len_avail,                                         \
                /*off16_split=*/0u,                                                    \
                _pd_block2_cmd_offset, _pd_initial_copy,                               \
                /*lit_scratch_ssbo=*/comp_ssbo, /*lit_in_scratch=*/0u,                  \
                /*cmd_scratch_ssbo=*/comp_ssbo, /*cmd_in_scratch=*/0u,                  \
                dst_ssbo, sc_decomp_size, dst_offset,                                  \
                /*mode=*/1u, lane);                                                    \
        }                                                                              \
    } while (false)

// CUDA reference: src/decode/lz_dispatch.cuh:23-165 with HAS_DICT=true
// (v4 #16). Textual twin of parseAndDecodeSubChunkRaw whose decode arms
// call the _dict body instantiations — the GLSL mirror of CUDA's second
// template instantiation (kernels select between the two on the
// frame-uniform `dict != nullptr` / `pc.dict_len != 0u` branch).
#define parseAndDecodeSubChunkRaw_dict(comp_ssbo, sc_src_off, sc_comp_size, sc_decomp_size, \
                                       dst_ssbo, dst_offset, base_offset, lane,             \
                                       dict_ssbo, dict_len)                                 \
    do {                                                                               \
        uint _pd_src = uint(sc_src_off);                                               \
        uint _pd_src_end = uint(sc_src_off) + uint(sc_comp_size);                      \
                                                                                       \
        uint _pd_initial_copy = 0u;                                                    \
        if (uint(base_offset) == 0u) {                                                 \
            if (uint(lane) < INITIAL_LITERAL_COPY_BYTES) {                             \
                (dst_ssbo)[uint(dst_offset) + uint(lane)] =                            \
                    (comp_ssbo)[_pd_src + uint(lane)];                                 \
            }                                                                          \
            subgroupBarrier();                                                         \
            subgroupMemoryBarrierBuffer();                                             \
            _pd_src += INITIAL_LITERAL_COPY_BYTES;                                     \
            _pd_initial_copy = INITIAL_LITERAL_COPY_BYTES;                             \
        }                                                                              \
                                                                                       \
        uint _pd_lit_ptr = _pd_src;                                                    \
        uint _pd_lit_size = 0u;                                                        \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_b0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_b1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            uint _pd_b2 = uint((comp_ssbo)[_pd_src + 2u]);                              \
            Type0HdrFields _pd_h = parseType0HdrFields(_pd_b0, _pd_b1, _pd_b2);        \
            _pd_lit_size = _pd_h.size;                                                 \
            _pd_src += _pd_h.header_bytes;                                              \
            _pd_lit_ptr = _pd_src;                                                     \
            _pd_src += _pd_lit_size;                                                   \
        }                                                                              \
        _pd_lit_size = subgroupShuffle(_pd_lit_size, 0u);                              \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_lit_ptr = _pd_src - _pd_lit_size;                                          \
                                                                                       \
        uint _pd_cmd_ptr;                                                              \
        uint _pd_cmd_size = 0u;                                                        \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_c0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_c1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            uint _pd_c2 = uint((comp_ssbo)[_pd_src + 2u]);                              \
            Type0HdrFields _pd_hc = parseType0HdrFields(_pd_c0, _pd_c1, _pd_c2);       \
            _pd_cmd_size = _pd_hc.size;                                                \
            _pd_src += _pd_hc.header_bytes;                                             \
            _pd_cmd_ptr = _pd_src;                                                     \
            _pd_src += _pd_cmd_size;                                                   \
        }                                                                              \
        _pd_cmd_size = subgroupShuffle(_pd_cmd_size, 0u);                              \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_cmd_ptr = _pd_src - _pd_cmd_size;                                          \
                                                                                       \
        uint _pd_block2_cmd_offset = _pd_cmd_size;                                     \
        if (uint(lane) == 0u && uint(sc_decomp_size) > LZ_BLOCK_SIZE) {                \
            uint _pd_b2_lo = uint((comp_ssbo)[_pd_src + 0u]);                           \
            uint _pd_b2_hi = uint((comp_ssbo)[_pd_src + 1u]);                           \
            _pd_block2_cmd_offset = readU16LE(_pd_b2_lo, _pd_b2_hi);                    \
            _pd_src += 2u;                                                              \
        }                                                                              \
        _pd_block2_cmd_offset = subgroupShuffle(_pd_block2_cmd_offset, 0u);            \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
                                                                                       \
        uint _pd_off16_raw;                                                            \
        uint _pd_off16_count = 0u;                                                     \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_o0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_o1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            _pd_off16_count = readU16LE(_pd_o0, _pd_o1);                                \
            _pd_off16_raw = _pd_src + 2u;                                              \
            _pd_src += 2u + _pd_off16_count * OFF16_ENTRY_BYTES;                       \
        }                                                                              \
        _pd_off16_count = subgroupShuffle(_pd_off16_count, 0u);                        \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_off16_raw = _pd_src - _pd_off16_count * OFF16_ENTRY_BYTES;                 \
                                                                                       \
        uint _pd_off32_count1 = 0u, _pd_off32_count2 = 0u;                              \
        uint _pd_off32_raw1 = _pd_src, _pd_off32_raw2 = _pd_src;                        \
        uint _pd_len_stream;                                                            \
        uint _pd_len_avail = 0u;                                                       \
                                                                                       \
        if (uint(lane) == 0u) {                                                        \
            uint _pd_t0 = uint((comp_ssbo)[_pd_src + 0u]);                              \
            uint _pd_t1 = uint((comp_ssbo)[_pd_src + 1u]);                              \
            uint _pd_t2 = uint((comp_ssbo)[_pd_src + 2u]);                              \
            uint _pd_tmp = readLE24(_pd_t0, _pd_t1, _pd_t2);                            \
            _pd_src += 3u;                                                              \
            if (_pd_tmp != 0u) {                                                       \
                _pd_off32_count1 = _pd_tmp >> OFF32_COUNT1_SHIFT;                       \
                _pd_off32_count2 = _pd_tmp & OFF32_COUNT2_MASK;                         \
                if (_pd_off32_count1 == OFF32_COUNT_PACK_MAX) {                        \
                    uint _pd_e0 = uint((comp_ssbo)[_pd_src + 0u]);                      \
                    uint _pd_e1 = uint((comp_ssbo)[_pd_src + 1u]);                      \
                    _pd_off32_count1 = readU16LE(_pd_e0, _pd_e1);                       \
                    _pd_src += 2u;                                                      \
                }                                                                       \
                if (_pd_off32_count2 == OFF32_COUNT_PACK_MAX) {                        \
                    uint _pd_f0 = uint((comp_ssbo)[_pd_src + 0u]);                      \
                    uint _pd_f1 = uint((comp_ssbo)[_pd_src + 1u]);                      \
                    _pd_off32_count2 = readU16LE(_pd_f0, _pd_f1);                       \
                    _pd_src += 2u;                                                      \
                }                                                                       \
                _pd_off32_raw1 = _pd_src;                                               \
                _pd_src += _pd_off32_count1 * OFF32_ENTRY_BYTES;                        \
                _pd_off32_raw2 = _pd_src;                                               \
                _pd_src += _pd_off32_count2 * OFF32_ENTRY_BYTES;                        \
            } else {                                                                    \
                _pd_off32_raw1 = _pd_src;                                               \
                _pd_off32_raw2 = _pd_src;                                               \
            }                                                                           \
            _pd_len_stream = _pd_src;                                                  \
            _pd_len_avail = _pd_src_end - _pd_src;                                     \
        }                                                                              \
        _pd_off32_count1 = subgroupShuffle(_pd_off32_count1, 0u);                      \
        _pd_off32_count2 = subgroupShuffle(_pd_off32_count2, 0u);                      \
        _pd_len_avail = subgroupShuffle(_pd_len_avail, 0u);                            \
        _pd_src = subgroupShuffle(_pd_src, 0u);                                        \
        _pd_len_stream = _pd_src;                                                      \
        _pd_off32_raw2 = _pd_src - _pd_off32_count2 * OFF32_ENTRY_BYTES;                \
        _pd_off32_raw1 = _pd_off32_raw2 - _pd_off32_count1 * OFF32_ENTRY_BYTES;         \
                                                                                       \
        if (_pd_off32_count1 == 0u && _pd_off32_count2 == 0u) {                        \
            decodeSubChunkRawMode_false_dict(                                          \
                comp_ssbo, _pd_cmd_ptr, _pd_cmd_size,                                  \
                comp_ssbo, _pd_lit_ptr, _pd_lit_size,                                  \
                comp_ssbo, _pd_off16_raw, _pd_off16_count,                             \
                comp_ssbo, _pd_len_stream, _pd_len_avail,                              \
                dst_ssbo, sc_decomp_size, _pd_initial_copy, dst_offset, lane,          \
                dict_ssbo, dict_len);                                                  \
        } else {                                                                       \
            decodeSubChunkGeneral_false_dict(                                          \
                comp_ssbo,                                                             \
                _pd_lit_ptr, _pd_lit_size,                                             \
                _pd_cmd_ptr, _pd_cmd_size,                                             \
                _pd_off16_raw, _pd_off16_count,                                        \
                comp_ssbo, 0u,                                                         \
                comp_ssbo, 0u,                                                         \
                _pd_off32_raw1, _pd_off32_count1,                                      \
                _pd_off32_raw2, _pd_off32_count2,                                      \
                _pd_len_stream, _pd_len_avail,                                         \
                /*off16_split=*/0u,                                                    \
                _pd_block2_cmd_offset, _pd_initial_copy,                               \
                /*lit_scratch_ssbo=*/comp_ssbo, /*lit_in_scratch=*/0u,                  \
                /*cmd_scratch_ssbo=*/comp_ssbo, /*cmd_in_scratch=*/0u,                  \
                dst_ssbo, sc_decomp_size, dst_offset,                                  \
                /*mode=*/1u, lane, dict_ssbo, dict_len);                               \
        }                                                                              \
    } while (false)

// CUDA reference: src/decode/lz_dispatch.cuh:173-226.
// General entropy-capable parseAndDecodeSubChunk: invokes
// parseSubChunkHeaders to fill a ParsedStreams record, then dispatches
// to decodeSubChunkRawMode_{true,false} (mode == 1 ∧ off32 empty) or
// decodeSubChunkGeneral_{true,false} (everything else). The off16_split
// bit (set by parseSubChunkHeaders) selects between the _true and
// _false instantiations.
//
// VK adaptation: macro form mirroring parseAndDecodeSubChunkRaw. The
// entropy_*_scratch arguments name the SSBOs that back the per-sub-chunk
// scratch slots when off16 / lit / tok are Huffman-coded; their (ssbo,
// base) pairs flow into ParsedStreams via parseSubChunkHeaders.
//
// Args (additional to parseAndDecodeSubChunkRaw):
//   mode                                    - sub-chunk mode bits
//   entropy_lit_ssbo, entropy_lit_base      - lit scratch SSBO + base
//   entropy_tok_ssbo, entropy_tok_base      - tok scratch SSBO + base
//   entropy_off16_ssbo, entropy_off16_base  - off16 scratch SSBO + base
#define parseAndDecodeSubChunk(comp_ssbo, sc_src_off, sc_comp_size, sc_decomp_size,    \
                                dst_ssbo, dst_offset, base_offset, mode,               \
                                entropy_lit_ssbo, entropy_lit_base,                    \
                                entropy_tok_ssbo, entropy_tok_base,                    \
                                entropy_off16_ssbo, entropy_off16_base,                \
                                has_entropy, lane)                                     \
    do {                                                                                \
        ParsedStreams _pdg_ps;                                                          \
        parseSubChunkHeaders(comp_ssbo, sc_src_off, sc_comp_size, sc_decomp_size,       \
                              dst_ssbo, dst_offset, base_offset,                        \
                              entropy_lit_ssbo, entropy_lit_base,                       \
                              entropy_tok_ssbo, entropy_tok_base,                       \
                              entropy_off16_ssbo, entropy_off16_base,                   \
                              has_entropy,                                              \
                              _pdg_ps, lane);                                           \
                                                                                        \
        /* VK adaptation: when literals or tokens are Huffman-coded the */              \
        /* corresponding ParsedStreams.lit/cmd_ptr is a byte offset into */             \
        /* `entropy_lit_ssbo` / `entropy_tok_ssbo` (NOT compressed). The */             \
        /* dispatcher hands both SSBOs + the per-stream flag to the decoder; */         \
        /* the decoder selects per-byte. CUDA collapses this transparently */           \
        /* via pointer typing. */                                                        \
        if (uint(mode) == 1u && _pdg_ps.off32_count1 == 0u && _pdg_ps.off32_count2 == 0u) { \
            /* CUDA reference: src/decode/lz_dispatch.cuh:190-214. mode==1 +  */        \
            /* off32 empty → raw-mode hot loop. CUDA dispatches polymorphically: */     \
            /* lit_ptr / cmd_ptr point either into the compressed blob or into */       \
            /* entropy scratch, transparently selected by pointer typing. */            \
            /* */                                                                        \
            /* VK adaptation (A-001 fix): GLSL macros bake SSBO names into the */       \
            /* expansion, so we enumerate the (lit_src, cmd_src) cross-product as */    \
            /* separate dispatch arms. Each arm calls the existing */                   \
            /* decodeSubChunkRawMode_{true,false} specializations with the correct */   \
            /* SSBO name in the cmd/lit slots; the byte offsets in */                   \
            /* _pdg_ps.cmd_ptr/lit_ptr are already valid in the selected SSBO */        \
            /* (parseLiteralStreamHeader / parseCommandStreamHeader set this up). */    \
            /* This is the GLSL-textual equivalent of CUDA's polymorphic pointer */     \
            /* semantics — no general-mode fallback needed. */                          \
            if (_pdg_ps.off16_split != 0u) {                                            \
                if (_pdg_ps.lit_in_scratch != 0u && _pdg_ps.cmd_in_scratch != 0u) {     \
                    decodeSubChunkRawMode_true(                                         \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                } else if (_pdg_ps.lit_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_true(                                         \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                } else if (_pdg_ps.cmd_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_true(                                         \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                } else {                                                                \
                    decodeSubChunkRawMode_true(                                         \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                }                                                                       \
            } else {                                                                    \
                if (_pdg_ps.lit_in_scratch != 0u && _pdg_ps.cmd_in_scratch != 0u) {     \
                    decodeSubChunkRawMode_false(                                        \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                } else if (_pdg_ps.lit_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_false(                                        \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                } else if (_pdg_ps.cmd_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_false(                                        \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                } else {                                                                \
                    decodeSubChunkRawMode_false(                                        \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane); \
                }                                                                       \
            }                                                                           \
        } else {                                                                        \
            /* CUDA reference: src/decode/lz_dispatch.cuh:215-225. General entropy- */  \
            /* capable path. Dispatch on off16_split.                              */   \
            if (_pdg_ps.off16_split != 0u) {                                            \
                decodeSubChunkGeneral_true(                                             \
                    comp_ssbo,                                                          \
                    _pdg_ps.lit_ptr, _pdg_ps.lit_size,                                  \
                    _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                                  \
                    _pdg_ps.off16_raw, _pdg_ps.off16_count,                             \
                    entropy_off16_ssbo, _pdg_ps.off16_hi,                               \
                    entropy_off16_ssbo, _pdg_ps.off16_lo,                               \
                    _pdg_ps.off32_raw1, _pdg_ps.off32_count1,                           \
                    _pdg_ps.off32_raw2, _pdg_ps.off32_count2,                           \
                    _pdg_ps.len_stream, _pdg_ps.len_avail,                              \
                    _pdg_ps.off16_split,                                                \
                    _pdg_ps.cmd_stream2_offset, _pdg_ps.initial_copy,                   \
                    entropy_lit_ssbo, _pdg_ps.lit_in_scratch,                           \
                    entropy_tok_ssbo, _pdg_ps.cmd_in_scratch,                           \
                    dst_ssbo, sc_decomp_size, dst_offset,                               \
                    mode, lane);                                                        \
            } else {                                                                    \
                decodeSubChunkGeneral_false(                                            \
                    comp_ssbo,                                                          \
                    _pdg_ps.lit_ptr, _pdg_ps.lit_size,                                  \
                    _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                                  \
                    _pdg_ps.off16_raw, _pdg_ps.off16_count,                             \
                    entropy_off16_ssbo, _pdg_ps.off16_hi,                               \
                    entropy_off16_ssbo, _pdg_ps.off16_lo,                               \
                    _pdg_ps.off32_raw1, _pdg_ps.off32_count1,                           \
                    _pdg_ps.off32_raw2, _pdg_ps.off32_count2,                           \
                    _pdg_ps.len_stream, _pdg_ps.len_avail,                              \
                    _pdg_ps.off16_split,                                                \
                    _pdg_ps.cmd_stream2_offset, _pdg_ps.initial_copy,                   \
                    entropy_lit_ssbo, _pdg_ps.lit_in_scratch,                           \
                    entropy_tok_ssbo, _pdg_ps.cmd_in_scratch,                           \
                    dst_ssbo, sc_decomp_size, dst_offset,                               \
                    mode, lane);                                                        \
            }                                                                           \
        }                                                                               \
    } while (false)

// CUDA reference: src/decode/lz_dispatch.cuh:173-226 with HAS_DICT=true
// (v4 #16). Textual twin of parseAndDecodeSubChunk whose decode arms
// call the _dict body instantiations — the GLSL mirror of CUDA's second
// template instantiation. Same A-001 (lit_src, cmd_src) cross-product
// enumeration; see the dict-less twin above for the arm rationale.
#define parseAndDecodeSubChunk_dict(comp_ssbo, sc_src_off, sc_comp_size, sc_decomp_size, \
                                    dst_ssbo, dst_offset, base_offset, mode,            \
                                    entropy_lit_ssbo, entropy_lit_base,                 \
                                    entropy_tok_ssbo, entropy_tok_base,                 \
                                    entropy_off16_ssbo, entropy_off16_base,             \
                                    has_entropy, lane, dict_ssbo, dict_len)             \
    do {                                                                                \
        ParsedStreams _pdg_ps;                                                          \
        parseSubChunkHeaders(comp_ssbo, sc_src_off, sc_comp_size, sc_decomp_size,       \
                              dst_ssbo, dst_offset, base_offset,                        \
                              entropy_lit_ssbo, entropy_lit_base,                       \
                              entropy_tok_ssbo, entropy_tok_base,                       \
                              entropy_off16_ssbo, entropy_off16_base,                   \
                              has_entropy,                                              \
                              _pdg_ps, lane);                                           \
                                                                                        \
        if (uint(mode) == 1u && _pdg_ps.off32_count1 == 0u && _pdg_ps.off32_count2 == 0u) { \
            if (_pdg_ps.off16_split != 0u) {                                            \
                if (_pdg_ps.lit_in_scratch != 0u && _pdg_ps.cmd_in_scratch != 0u) {     \
                    decodeSubChunkRawMode_true_dict(                                    \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                } else if (_pdg_ps.lit_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_true_dict(                                    \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                } else if (_pdg_ps.cmd_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_true_dict(                                    \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                } else {                                                                \
                    decodeSubChunkRawMode_true_dict(                                    \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        entropy_off16_ssbo, _pdg_ps.off16_hi,                           \
                        entropy_off16_ssbo, _pdg_ps.off16_lo, _pdg_ps.off16_count,      \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                }                                                                       \
            } else {                                                                    \
                if (_pdg_ps.lit_in_scratch != 0u && _pdg_ps.cmd_in_scratch != 0u) {     \
                    decodeSubChunkRawMode_false_dict(                                   \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                } else if (_pdg_ps.lit_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_false_dict(                                   \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        entropy_lit_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,            \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                } else if (_pdg_ps.cmd_in_scratch != 0u) {                              \
                    decodeSubChunkRawMode_false_dict(                                   \
                        entropy_tok_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,            \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                } else {                                                                \
                    decodeSubChunkRawMode_false_dict(                                   \
                        comp_ssbo, _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                   \
                        comp_ssbo, _pdg_ps.lit_ptr, _pdg_ps.lit_size,                   \
                        comp_ssbo, _pdg_ps.off16_raw, _pdg_ps.off16_count,              \
                        comp_ssbo, _pdg_ps.len_stream, _pdg_ps.len_avail,               \
                        dst_ssbo, sc_decomp_size, _pdg_ps.initial_copy, dst_offset, lane, \
                        dict_ssbo, dict_len);                                           \
                }                                                                       \
            }                                                                           \
        } else {                                                                        \
            if (_pdg_ps.off16_split != 0u) {                                            \
                decodeSubChunkGeneral_true_dict(                                        \
                    comp_ssbo,                                                          \
                    _pdg_ps.lit_ptr, _pdg_ps.lit_size,                                  \
                    _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                                  \
                    _pdg_ps.off16_raw, _pdg_ps.off16_count,                             \
                    entropy_off16_ssbo, _pdg_ps.off16_hi,                               \
                    entropy_off16_ssbo, _pdg_ps.off16_lo,                               \
                    _pdg_ps.off32_raw1, _pdg_ps.off32_count1,                           \
                    _pdg_ps.off32_raw2, _pdg_ps.off32_count2,                           \
                    _pdg_ps.len_stream, _pdg_ps.len_avail,                              \
                    _pdg_ps.off16_split,                                                \
                    _pdg_ps.cmd_stream2_offset, _pdg_ps.initial_copy,                   \
                    entropy_lit_ssbo, _pdg_ps.lit_in_scratch,                           \
                    entropy_tok_ssbo, _pdg_ps.cmd_in_scratch,                           \
                    dst_ssbo, sc_decomp_size, dst_offset,                               \
                    mode, lane, dict_ssbo, dict_len);                                   \
            } else {                                                                    \
                decodeSubChunkGeneral_false_dict(                                       \
                    comp_ssbo,                                                          \
                    _pdg_ps.lit_ptr, _pdg_ps.lit_size,                                  \
                    _pdg_ps.cmd_ptr, _pdg_ps.cmd_size,                                  \
                    _pdg_ps.off16_raw, _pdg_ps.off16_count,                             \
                    entropy_off16_ssbo, _pdg_ps.off16_hi,                               \
                    entropy_off16_ssbo, _pdg_ps.off16_lo,                               \
                    _pdg_ps.off32_raw1, _pdg_ps.off32_count1,                           \
                    _pdg_ps.off32_raw2, _pdg_ps.off32_count2,                           \
                    _pdg_ps.len_stream, _pdg_ps.len_avail,                              \
                    _pdg_ps.off16_split,                                                \
                    _pdg_ps.cmd_stream2_offset, _pdg_ps.initial_copy,                   \
                    entropy_lit_ssbo, _pdg_ps.lit_in_scratch,                           \
                    entropy_tok_ssbo, _pdg_ps.cmd_in_scratch,                           \
                    dst_ssbo, sc_decomp_size, dst_offset,                               \
                    mode, lane, dict_ssbo, dict_len);                                   \
            }                                                                           \
        }                                                                               \
    } while (false)

#endif
