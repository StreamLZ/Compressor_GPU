// 1:1 port of src/encode/assemble_kernel.cu device-internal helpers
// (parseRaw / writeHuffChunkHdr / emitEntropyStream / assembleSubChunk).
// Shared by assemble_measure_kernel.comp and assemble_write_kernel.comp.
//
// VK adaptation: every CUDA helper that took raw `uint8_t*` pointers is
// expressed as a function-like macro that takes the SSBO names + base
// byte offsets as plain identifiers (the same pattern lz_decode_core.glsl
// uses for warpCopy / warpLiteralCopy). The CUDA `RawStreams` struct
// dissolves into a fixed set of in-scope locals named `_pr_*`.
//

#ifndef SRCVK_ENCODE_ASSEMBLE_CORE_GLSL
#define SRCVK_ENCODE_ASSEMBLE_CORE_GLSL

#include "lz_format.glsl"

// CUDA reference: src/encode/assemble_kernel.cu:37-39. Assembler-private
// wire-format constants.
const uint RAW_CHUNK_HDR_BYTES  = 3u;
const uint HUFF_CHUNK_HDR_BYTES = 5u;
const uint OFF16_ENTROPY_MIN    = 32u;

// CUDA reference: src/encode/assemble_kernel.cu:156-164. type-4
// non-compact 5-byte Huffman chunk header.
void writeHuffChunkHdr(uint p_base, uint comp_size, uint decomp_size, out uint b0_out,
                       out uint b1_out, out uint b2_out, out uint b3_out, out uint b4_out) {
    uint dm1 = decomp_size - 1u;
    b0_out = ((HUFF_CHUNK_TYPE << CHUNK_TYPE_SHIFT)
             | ((dm1 >> ENTROPY_HDR_DM1_HIGH4_SHIFT) & ENTROPY_HDR_DM1_HIGH_MASK)) & 0xFFu;
    uint bits = comp_size | ((dm1 & ENTROPY_HDR_DM1_LOW_MASK) << ENTROPY_HDR_COMP_BITS);
    uint hi24 = bits >> 8;
    uint t0; uint t1; uint t2;
    writeBE24(t0, t1, t2, hi24);
    b1_out = t0;
    b2_out = t1;
    b3_out = t2;
    b4_out = bits & 0xFFu;
    p_base; // unused — kept for signature parity with the CUDA helper.
}

// VK adaptation: per-byte writer used by emitEntropyStream and the
// off16 split path. Caller has already loaded the 5 header bytes via
// writeHuffChunkHdr.
#define _slzWriteHuffHdr(out_ssbo, out_base, comp_size, decomp_size)                                \
    do {                                                                                            \
        uint _whh_b0; uint _whh_b1; uint _whh_b2; uint _whh_b3; uint _whh_b4;                       \
        writeHuffChunkHdr(0u, uint(comp_size), uint(decomp_size),                                   \
                          _whh_b0, _whh_b1, _whh_b2, _whh_b3, _whh_b4);                             \
        (out_ssbo)[uint(out_base) + 0u] = uint8_t(_whh_b0);                                         \
        (out_ssbo)[uint(out_base) + 1u] = uint8_t(_whh_b1);                                         \
        (out_ssbo)[uint(out_base) + 2u] = uint8_t(_whh_b2);                                         \
        (out_ssbo)[uint(out_base) + 3u] = uint8_t(_whh_b3);                                         \
        (out_ssbo)[uint(out_base) + 4u] = uint8_t(_whh_b4);                                         \
    } while (false)

// CUDA reference: src/encode/assemble_kernel.cu:173-197. emitEntropyStream:
// emit one entropy stream as type-4 (huff) or type-0 (raw). Returns the
// emitted byte count via the out param `bytes_out`. `do_write` toggles
// whether the bytes are actually written (false for the measure pass).
//
// VK adaptation: macro form. The two output choices share the same SSBO
// `out_ssbo` (the write kernel's d_frame slot). When the byte source is
// raw bytes, the macro reads from `raw_ssbo[raw_base + i]`; when it is
// the huff body, the macro reads from `huff_ssbo[huff_base + i]`.
#define emitEntropyStream(do_write, out_ssbo, out_base,                                             \
                          raw_ssbo, raw_base, raw_count,                                            \
                          huff_ssbo, huff_base, huff_size,                                          \
                          lane_in, bytes_out)                                                       \
    do {                                                                                            \
        bool _ees_use_huff = (uint(huff_size) > 0u)                                                 \
            && (uint(huff_size) + HUFF_CHUNK_HDR_BYTES < uint(raw_count) + RAW_CHUNK_HDR_BYTES);    \
        if (_ees_use_huff) {                                                                        \
            if (do_write) {                                                                         \
                if (uint(lane_in) == 0u) {                                                          \
                    _slzWriteHuffHdr(out_ssbo, uint(out_base),                                      \
                                     uint(huff_size), uint(raw_count));                             \
                }                                                                                   \
                subgroupBarrier();                                                                  \
                subgroupMemoryBarrierBuffer();                                                      \
                for (uint _ees_i = uint(lane_in); _ees_i < uint(huff_size); _ees_i += WARP_SIZE) {  \
                    (out_ssbo)[uint(out_base) + HUFF_CHUNK_HDR_BYTES + _ees_i] =                    \
                        (huff_ssbo)[uint(huff_base) + _ees_i];                                      \
                }                                                                                   \
            }                                                                                       \
            (bytes_out) = HUFF_CHUNK_HDR_BYTES + uint(huff_size);                                   \
        } else {                                                                                    \
            if (do_write) {                                                                         \
                if (uint(lane_in) == 0u) {                                                          \
                    uint _ees_b0; uint _ees_b1; uint _ees_b2;                                       \
                    writeBE24(_ees_b0, _ees_b1, _ees_b2, uint(raw_count));                          \
                    (out_ssbo)[uint(out_base) + 0u] = uint8_t(_ees_b0);                             \
                    (out_ssbo)[uint(out_base) + 1u] = uint8_t(_ees_b1);                             \
                    (out_ssbo)[uint(out_base) + 2u] = uint8_t(_ees_b2);                             \
                }                                                                                   \
                subgroupBarrier();                                                                  \
                subgroupMemoryBarrierBuffer();                                                      \
                for (uint _ees_i = uint(lane_in); _ees_i < uint(raw_count); _ees_i += WARP_SIZE) {  \
                    (out_ssbo)[uint(out_base) + RAW_CHUNK_HDR_BYTES + _ees_i] =                     \
                        (raw_ssbo)[uint(raw_base) + _ees_i];                                        \
                }                                                                                   \
            }                                                                                       \
            (bytes_out) = RAW_CHUNK_HDR_BYTES + uint(raw_count);                                    \
        }                                                                                           \
    } while (false)

// CUDA reference: src/encode/assemble_kernel.cu:74-148. parseRaw: parse
// the raw sub-chunk payload into its component stream offsets+counts.
// On success, the `_pr_*` out-params describe each stream. On failure,
// `_pr_ok = false`.
//
// VK adaptation: macro form so we can return many out-params without
// GLSL-struct overhead. All offsets are absolute byte offsets into
// `raw_ssbo`.
#define parseRaw(raw_ssbo, raw_base, raw_size, sub_decomp_size,                                     \
                 _pr_lit_off, _pr_lit_count,                                                        \
                 _pr_tok_off, _pr_tok_count,                                                        \
                 _pr_cmd2_off, _pr_cmd2_present,                                                    \
                 _pr_off16_off, _pr_off16_count,                                                    \
                 _pr_off32_off, _pr_off32_bytes,                                                    \
                 _pr_length_off, _pr_length_size, _pr_ok)                                           \
    do {                                                                                            \
        (_pr_ok) = false;                                                                           \
        uint _pr_rp = 0u;                                                                           \
        uint _pr_raw_sz = uint(raw_size);                                                           \
        uint _pr_raw_base = uint(raw_base);                                                         \
        bool _pr_keep_going = true;                                                                 \
                                                                                                    \
        if (_pr_rp + 3u > _pr_raw_sz) _pr_keep_going = false;                                       \
        uint _pr_b0; uint _pr_b1; uint _pr_b2;                                                      \
        if (_pr_keep_going) {                                                                       \
            _pr_b0 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 0u]);                                  \
            _pr_b1 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 1u]);                                  \
            _pr_b2 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 2u]);                                  \
            (_pr_lit_count) = readBE24(_pr_b0, _pr_b1, _pr_b2);                                     \
            _pr_rp += 3u;                                                                           \
            if (_pr_rp + (_pr_lit_count) > _pr_raw_sz) _pr_keep_going = false;                      \
        }                                                                                           \
        if (_pr_keep_going) {                                                                       \
            (_pr_lit_off) = _pr_raw_base + _pr_rp;                                                  \
            _pr_rp += (_pr_lit_count);                                                              \
                                                                                                    \
            if (_pr_rp + 3u > _pr_raw_sz) _pr_keep_going = false;                                   \
        }                                                                                           \
        if (_pr_keep_going) {                                                                       \
            _pr_b0 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 0u]);                                  \
            _pr_b1 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 1u]);                                  \
            _pr_b2 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 2u]);                                  \
            (_pr_tok_count) = readBE24(_pr_b0, _pr_b1, _pr_b2);                                     \
            _pr_rp += 3u;                                                                           \
            if (_pr_rp + (_pr_tok_count) > _pr_raw_sz) _pr_keep_going = false;                      \
        }                                                                                           \
        if (_pr_keep_going) {                                                                       \
            (_pr_tok_off) = _pr_raw_base + _pr_rp;                                                  \
            _pr_rp += (_pr_tok_count);                                                              \
                                                                                                    \
            (_pr_cmd2_present) = (uint(sub_decomp_size) > LZ_BLOCK_SIZE);                           \
            if ((_pr_cmd2_present)) {                                                               \
                if (_pr_rp + 2u > _pr_raw_sz) _pr_keep_going = false;                               \
                if (_pr_keep_going) {                                                               \
                    (_pr_cmd2_off) = _pr_raw_base + _pr_rp;                                         \
                    _pr_rp += 2u;                                                                   \
                }                                                                                   \
            } else {                                                                                \
                (_pr_cmd2_off) = 0u;                                                                \
            }                                                                                       \
        }                                                                                           \
        if (_pr_keep_going) {                                                                       \
            if (_pr_rp + 2u > _pr_raw_sz) _pr_keep_going = false;                                   \
        }                                                                                           \
        if (_pr_keep_going) {                                                                       \
            uint _pr_o0 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 0u]);                             \
            uint _pr_o1 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 1u]);                             \
            (_pr_off16_count) = readU16LE(_pr_o0, _pr_o1);                                          \
            _pr_rp += 2u;                                                                           \
            uint _pr_off16_bytes = (_pr_off16_count) * 2u;                                          \
            if (_pr_rp + _pr_off16_bytes > _pr_raw_sz) _pr_keep_going = false;                      \
            if (_pr_keep_going) {                                                                   \
                (_pr_off16_off) = _pr_raw_base + _pr_rp;                                            \
                _pr_rp += _pr_off16_bytes;                                                          \
            }                                                                                       \
        }                                                                                           \
        /* off32: packed 3-byte header (+ 0/2/4 extra), then variable data. */                      \
        uint _pr_off32_start = 0u;                                                                  \
        if (_pr_keep_going) {                                                                       \
            if (_pr_rp + 3u > _pr_raw_sz) _pr_keep_going = false;                                   \
        }                                                                                           \
        uint _pr_hdr_plus_extra = 3u;                                                               \
        uint _pr_c1 = 0u;                                                                           \
        uint _pr_c2 = 0u;                                                                           \
        if (_pr_keep_going) {                                                                       \
            _pr_off32_start = _pr_raw_base + _pr_rp;                                                \
            uint _pr_p0 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 0u]);                             \
            uint _pr_p1 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 1u]);                             \
            uint _pr_p2 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 2u]);                             \
            uint _pr_packed = readLE24(_pr_p0, _pr_p1, _pr_p2);                                     \
            _pr_rp += 3u;                                                                           \
            _pr_c1 = (_pr_packed >> OFF32_COUNT_FIELD_BITS) & OFF32_COUNT_PACK_MAX;                 \
            _pr_c2 = _pr_packed & OFF32_COUNT_PACK_MAX;                                             \
            if (_pr_c1 >= OFF32_COUNT_PACK_MAX) {                                                   \
                if (_pr_rp + 2u > _pr_raw_sz) _pr_keep_going = false;                               \
                if (_pr_keep_going) {                                                               \
                    uint _pr_e0 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 0u]);                     \
                    uint _pr_e1 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 1u]);                     \
                    _pr_c1 = readU16LE(_pr_e0, _pr_e1);                                             \
                    _pr_rp += 2u;                                                                   \
                    _pr_hdr_plus_extra += 2u;                                                       \
                }                                                                                   \
            }                                                                                       \
            if (_pr_keep_going && _pr_c2 >= OFF32_COUNT_PACK_MAX) {                                 \
                if (_pr_rp + 2u > _pr_raw_sz) _pr_keep_going = false;                               \
                if (_pr_keep_going) {                                                               \
                    uint _pr_e0 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 0u]);                     \
                    uint _pr_e1 = uint((raw_ssbo)[_pr_raw_base + _pr_rp + 1u]);                     \
                    _pr_c2 = readU16LE(_pr_e0, _pr_e1);                                             \
                    _pr_rp += 2u;                                                                   \
                    _pr_hdr_plus_extra += 2u;                                                       \
                }                                                                                   \
            }                                                                                       \
        }                                                                                           \
        uint _pr_data_bytes = 0u;                                                                   \
        if (_pr_keep_going) {                                                                       \
            uint _pr_total_entries = _pr_c1 + _pr_c2;                                               \
            uint _pr_scan = _pr_rp;                                                                 \
            for (uint _pr_i = 0u; _pr_i < _pr_total_entries; _pr_i += 1u) {                         \
                if (_pr_scan + 3u > _pr_raw_sz) { _pr_keep_going = false; break; }                  \
                uint _pr_t2 = uint((raw_ssbo)[_pr_raw_base + 0u + _pr_scan + 2u]);                  \
                if ((_pr_t2 & OFF32_LONG_ENTRY_TAG) == OFF32_LONG_ENTRY_TAG) {                      \
                    if (_pr_scan + 4u > _pr_raw_sz) { _pr_keep_going = false; break; }              \
                    _pr_data_bytes += 4u; _pr_scan += 4u;                                           \
                } else {                                                                            \
                    _pr_data_bytes += 3u; _pr_scan += 3u;                                           \
                }                                                                                   \
            }                                                                                       \
            if (_pr_keep_going) {                                                                   \
                _pr_rp += _pr_data_bytes;                                                           \
                if (_pr_rp > _pr_raw_sz) _pr_keep_going = false;                                    \
            }                                                                                       \
        }                                                                                           \
        if (_pr_keep_going) {                                                                       \
            (_pr_off32_off) = _pr_off32_start;                                                      \
            (_pr_off32_bytes) = _pr_hdr_plus_extra + _pr_data_bytes;                                \
            (_pr_length_off) = _pr_raw_base + _pr_rp;                                               \
            (_pr_length_size) = _pr_raw_sz - _pr_rp;                                                \
            (_pr_ok) = true;                                                                        \
        }                                                                                           \
    } while (false)

// CUDA reference: src/encode/assemble_kernel.cu:201-312. assembleSubChunk:
// for one sub-chunk, either measure (out_write=false) or write
// (out_write=true) the assembled `[verbatim init][entropy lit]
// [entropy tok][cmd2?][off16-or-split][off32 verbatim][length verbatim]`
// payload. Returns the byte count via `enc_n_out`.
//
// VK adaptation: the CUDA function takes a struct `AssembleDesc&`. Here
// the desc-base index `desc_idx` is passed and the 13 u32 fields are
// loaded inline from descs_raw (matching the AssembleDesc u32 stride
// constants in lz_format.glsl).
#define assembleSubChunk(out_write,                                                                 \
                         d_raw_ssbo, d_huff_lit_ssbo, d_huff_tok_ssbo, d_huff_off16_ssbo,           \
                         descs_raw_ssbo, desc_idx,                                                  \
                         out_ssbo, out_payload_base, lane_in, enc_n_out)                            \
    do {                                                                                            \
        uint _as_db = uint(desc_idx) * ASSEMBLE_DESC_U32_STRIDE;                                    \
        uint _as_raw_offset       = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_RAW_OFFSET];        \
        uint _as_raw_size         = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_RAW_SIZE];          \
        uint _as_huff_lit_off     = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_LIT_OFFSET];   \
        uint _as_huff_lit_size    = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_LIT_SIZE];     \
        uint _as_huff_tok_off     = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_TOK_OFFSET];   \
        uint _as_huff_tok_size    = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_TOK_SIZE];     \
        uint _as_huff_off16hi_off = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_OFF16HI_OFF];  \
        uint _as_huff_off16hi_sz  = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_OFF16HI_SIZE]; \
        uint _as_huff_off16lo_off = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_OFF16LO_OFF];  \
        uint _as_huff_off16lo_sz  = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_HUFF_OFF16LO_SIZE]; \
        uint _as_sub_decomp_size  = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_SUB_DECOMP_SIZE];   \
        uint _as_init_bytes       = (descs_raw_ssbo)[_as_db + ASSEMBLE_DESC_OFF_INIT_BYTES];        \
                                                                                                    \
        (enc_n_out) = 0u;                                                                           \
        bool _as_proceed = (_as_init_bytes <= _as_raw_size);                                        \
                                                                                                    \
        uint _pr_lit_off = 0u;       uint _pr_lit_count = 0u;                                       \
        uint _pr_tok_off = 0u;       uint _pr_tok_count = 0u;                                       \
        uint _pr_cmd2_off = 0u;      bool _pr_cmd2_present = false;                                 \
        uint _pr_off16_off = 0u;     uint _pr_off16_count = 0u;                                     \
        uint _pr_off32_off = 0u;     uint _pr_off32_bytes = 0u;                                     \
        uint _pr_length_off = 0u;    uint _pr_length_size = 0u;                                     \
        bool _pr_ok = false;                                                                        \
                                                                                                    \
        if (_as_proceed) {                                                                          \
            parseRaw(d_raw_ssbo,                                                                    \
                     _as_raw_offset + _as_init_bytes,                                               \
                     _as_raw_size - _as_init_bytes,                                                 \
                     _as_sub_decomp_size,                                                           \
                     _pr_lit_off, _pr_lit_count,                                                    \
                     _pr_tok_off, _pr_tok_count,                                                    \
                     _pr_cmd2_off, _pr_cmd2_present,                                                \
                     _pr_off16_off, _pr_off16_count,                                                \
                     _pr_off32_off, _pr_off32_bytes,                                                \
                     _pr_length_off, _pr_length_size, _pr_ok);                                      \
        }                                                                                           \
                                                                                                    \
        if (_pr_ok) {                                                                               \
            uint _as_wp = 0u;                                                                       \
                                                                                                    \
            /* 0. initial raw bytes (frame's first sub-chunk) - verbatim. */                        \
            if (_as_init_bytes > 0u) {                                                              \
                if (out_write) {                                                                    \
                    for (uint _as_i = uint(lane_in); _as_i < _as_init_bytes; _as_i += WARP_SIZE) {  \
                        (out_ssbo)[uint(out_payload_base) + _as_wp + _as_i] =                       \
                            (d_raw_ssbo)[_as_raw_offset + _as_i];                                   \
                    }                                                                               \
                    subgroupBarrier();                                                              \
                    subgroupMemoryBarrierBuffer();                                                  \
                }                                                                                   \
                _as_wp += _as_init_bytes;                                                           \
            }                                                                                       \
                                                                                                    \
            /* 1. literals — huff vs raw. */                                                        \
            {                                                                                       \
                uint _as_bytes = 0u;                                                                \
                emitEntropyStream(out_write, out_ssbo,                                              \
                                  uint(out_payload_base) + _as_wp,                                  \
                                  d_raw_ssbo, _pr_lit_off, _pr_lit_count,                           \
                                  d_huff_lit_ssbo, _as_huff_lit_off, _as_huff_lit_size,             \
                                  lane_in, _as_bytes);                                              \
                _as_wp += _as_bytes;                                                                \
            }                                                                                       \
            /* 2. tokens — huff vs raw. */                                                          \
            {                                                                                       \
                uint _as_bytes = 0u;                                                                \
                emitEntropyStream(out_write, out_ssbo,                                              \
                                  uint(out_payload_base) + _as_wp,                                  \
                                  d_raw_ssbo, _pr_tok_off, _pr_tok_count,                           \
                                  d_huff_tok_ssbo, _as_huff_tok_off, _as_huff_tok_size,             \
                                  lane_in, _as_bytes);                                              \
                _as_wp += _as_bytes;                                                                \
            }                                                                                       \
                                                                                                    \
            /* 3. cmd2 — 2 bytes verbatim, when present. */                                         \
            if (_pr_cmd2_present) {                                                                 \
                if (out_write && uint(lane_in) == 0u) {                                             \
                    (out_ssbo)[uint(out_payload_base) + _as_wp + 0u] =                              \
                        (d_raw_ssbo)[_pr_cmd2_off + 0u];                                            \
                    (out_ssbo)[uint(out_payload_base) + _as_wp + 1u] =                              \
                        (d_raw_ssbo)[_pr_cmd2_off + 1u];                                            \
                }                                                                                   \
                _as_wp += 2u;                                                                       \
            }                                                                                       \
                                                                                                    \
            /* 4. off16. */                                                                         \
            {                                                                                       \
                uint _as_off16_bytes = _pr_off16_count * 2u;                                        \
                if (_pr_off16_count >= OFF16_ENTROPY_MIN) {                                         \
                    uint _as_hi_sz = _as_huff_off16hi_sz;                                           \
                    uint _as_lo_sz = _as_huff_off16lo_sz;                                           \
                    const uint _AS_HUFF_VS_RAW_HDR_OVERHEAD = HUFF_CHUNK_HDR_BYTES - RAW_CHUNK_HDR_BYTES; \
                    bool _as_hi_huff = (_as_hi_sz > 0u)                                             \
                        && (_as_hi_sz + _AS_HUFF_VS_RAW_HDR_OVERHEAD < _pr_off16_count);             \
                    bool _as_lo_huff = (_as_lo_sz > 0u)                                             \
                        && (_as_lo_sz + _AS_HUFF_VS_RAW_HDR_OVERHEAD < _pr_off16_count);             \
                    uint _as_hi_chunk = _as_hi_huff ? (HUFF_CHUNK_HDR_BYTES + _as_hi_sz)            \
                                                   : (_pr_off16_count + RAW_CHUNK_HDR_BYTES);       \
                    uint _as_lo_chunk = _as_lo_huff ? (HUFF_CHUNK_HDR_BYTES + _as_lo_sz)            \
                                                   : (_pr_off16_count + RAW_CHUNK_HDR_BYTES);       \
                    uint _as_split_total = _as_hi_chunk + _as_lo_chunk;                             \
                    if (_as_split_total < _as_off16_bytes) {                                        \
                        if (out_write && uint(lane_in) == 0u) {                                     \
                            uint _as_mb0; uint _as_mb1;                                             \
                            storeU16LE(_as_mb0, _as_mb1, OFF16_ENTROPY_MARKER);                     \
                            (out_ssbo)[uint(out_payload_base) + _as_wp + 0u] = uint8_t(_as_mb0);    \
                            (out_ssbo)[uint(out_payload_base) + _as_wp + 1u] = uint8_t(_as_mb1);    \
                        }                                                                           \
                        _as_wp += 2u;                                                                \
                                                                                                    \
                        /* hi */                                                                    \
                        if (_as_hi_huff) {                                                          \
                            if (out_write) {                                                        \
                                if (uint(lane_in) == 0u) {                                          \
                                    _slzWriteHuffHdr(out_ssbo,                                      \
                                        uint(out_payload_base) + _as_wp,                            \
                                        _as_hi_sz, _pr_off16_count);                                \
                                }                                                                   \
                                subgroupBarrier();                                                  \
                                subgroupMemoryBarrierBuffer();                                      \
                                for (uint _as_i = uint(lane_in); _as_i < _as_hi_sz;                 \
                                     _as_i += WARP_SIZE) {                                          \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp                      \
                                              + HUFF_CHUNK_HDR_BYTES + _as_i] =                     \
                                        (d_huff_off16_ssbo)[_as_huff_off16hi_off + _as_i];          \
                                }                                                                   \
                            }                                                                       \
                            _as_wp += HUFF_CHUNK_HDR_BYTES + _as_hi_sz;                             \
                        } else {                                                                    \
                            if (out_write) {                                                        \
                                if (uint(lane_in) == 0u) {                                          \
                                    uint _as_b0; uint _as_b1; uint _as_b2;                          \
                                    writeBE24(_as_b0, _as_b1, _as_b2, _pr_off16_count);             \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp + 0u] = uint8_t(_as_b0); \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp + 1u] = uint8_t(_as_b1); \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp + 2u] = uint8_t(_as_b2); \
                                }                                                                   \
                                subgroupBarrier();                                                  \
                                subgroupMemoryBarrierBuffer();                                      \
                                for (uint _as_i = uint(lane_in); _as_i < _pr_off16_count;           \
                                     _as_i += WARP_SIZE) {                                          \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp                      \
                                              + RAW_CHUNK_HDR_BYTES + _as_i] =                      \
                                        (d_raw_ssbo)[_pr_off16_off + _as_i * 2u + 1u];              \
                                }                                                                   \
                            }                                                                       \
                            _as_wp += RAW_CHUNK_HDR_BYTES + _pr_off16_count;                        \
                        }                                                                           \
                        /* lo */                                                                    \
                        if (_as_lo_huff) {                                                          \
                            if (out_write) {                                                        \
                                if (uint(lane_in) == 0u) {                                          \
                                    _slzWriteHuffHdr(out_ssbo,                                      \
                                        uint(out_payload_base) + _as_wp,                            \
                                        _as_lo_sz, _pr_off16_count);                                \
                                }                                                                   \
                                subgroupBarrier();                                                  \
                                subgroupMemoryBarrierBuffer();                                      \
                                for (uint _as_i = uint(lane_in); _as_i < _as_lo_sz;                 \
                                     _as_i += WARP_SIZE) {                                          \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp                      \
                                              + HUFF_CHUNK_HDR_BYTES + _as_i] =                     \
                                        (d_huff_off16_ssbo)[_as_huff_off16lo_off + _as_i];          \
                                }                                                                   \
                            }                                                                       \
                            _as_wp += HUFF_CHUNK_HDR_BYTES + _as_lo_sz;                             \
                        } else {                                                                    \
                            if (out_write) {                                                        \
                                if (uint(lane_in) == 0u) {                                          \
                                    uint _as_b0; uint _as_b1; uint _as_b2;                          \
                                    writeBE24(_as_b0, _as_b1, _as_b2, _pr_off16_count);             \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp + 0u] = uint8_t(_as_b0); \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp + 1u] = uint8_t(_as_b1); \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp + 2u] = uint8_t(_as_b2); \
                                }                                                                   \
                                subgroupBarrier();                                                  \
                                subgroupMemoryBarrierBuffer();                                      \
                                for (uint _as_i = uint(lane_in); _as_i < _pr_off16_count;           \
                                     _as_i += WARP_SIZE) {                                          \
                                    (out_ssbo)[uint(out_payload_base) + _as_wp                      \
                                              + RAW_CHUNK_HDR_BYTES + _as_i] =                      \
                                        (d_raw_ssbo)[_pr_off16_off + _as_i * 2u + 0u];              \
                                }                                                                   \
                            }                                                                       \
                            _as_wp += RAW_CHUNK_HDR_BYTES + _pr_off16_count;                        \
                        }                                                                           \
                    } else {                                                                        \
                        /* raw: [u16 off16_count][off16 data] */                                    \
                        if (out_write && uint(lane_in) == 0u) {                                     \
                            uint _as_m0; uint _as_m1;                                               \
                            storeU16LE(_as_m0, _as_m1, _pr_off16_count);                            \
                            (out_ssbo)[uint(out_payload_base) + _as_wp + 0u] = uint8_t(_as_m0);     \
                            (out_ssbo)[uint(out_payload_base) + _as_wp + 1u] = uint8_t(_as_m1);     \
                        }                                                                           \
                        _as_wp += 2u;                                                                \
                        if (out_write) {                                                            \
                            subgroupBarrier();                                                      \
                            subgroupMemoryBarrierBuffer();                                          \
                            for (uint _as_i = uint(lane_in); _as_i < _as_off16_bytes;               \
                                 _as_i += WARP_SIZE) {                                              \
                                (out_ssbo)[uint(out_payload_base) + _as_wp + _as_i] =               \
                                    (d_raw_ssbo)[_pr_off16_off + _as_i];                            \
                            }                                                                       \
                        }                                                                           \
                        _as_wp += _as_off16_bytes;                                                  \
                    }                                                                               \
                } else {                                                                            \
                    /* small off16: [u16 off16_count][off16 data] */                                \
                    if (out_write && uint(lane_in) == 0u) {                                         \
                        uint _as_m0; uint _as_m1;                                                   \
                        storeU16LE(_as_m0, _as_m1, _pr_off16_count);                                \
                        (out_ssbo)[uint(out_payload_base) + _as_wp + 0u] = uint8_t(_as_m0);         \
                        (out_ssbo)[uint(out_payload_base) + _as_wp + 1u] = uint8_t(_as_m1);         \
                    }                                                                               \
                    _as_wp += 2u;                                                                    \
                    if (out_write) {                                                                \
                        subgroupBarrier();                                                          \
                        subgroupMemoryBarrierBuffer();                                              \
                        for (uint _as_i = uint(lane_in); _as_i < _as_off16_bytes;                   \
                             _as_i += WARP_SIZE) {                                                  \
                            (out_ssbo)[uint(out_payload_base) + _as_wp + _as_i] =                   \
                                (d_raw_ssbo)[_pr_off16_off + _as_i];                                \
                        }                                                                           \
                    }                                                                               \
                    _as_wp += _as_off16_bytes;                                                      \
                }                                                                                   \
            }                                                                                       \
                                                                                                    \
            /* 5. off32 — verbatim copy. */                                                         \
            if (out_write) {                                                                        \
                subgroupBarrier();                                                                  \
                subgroupMemoryBarrierBuffer();                                                      \
                for (uint _as_i = uint(lane_in); _as_i < _pr_off32_bytes; _as_i += WARP_SIZE) {     \
                    (out_ssbo)[uint(out_payload_base) + _as_wp + _as_i] =                           \
                        (d_raw_ssbo)[_pr_off32_off + _as_i];                                        \
                }                                                                                   \
            }                                                                                       \
            _as_wp += _pr_off32_bytes;                                                              \
                                                                                                    \
            /* 6. length stream — verbatim copy. */                                                 \
            if (out_write) {                                                                        \
                subgroupBarrier();                                                                  \
                subgroupMemoryBarrierBuffer();                                                      \
                for (uint _as_i = uint(lane_in); _as_i < _pr_length_size; _as_i += WARP_SIZE) {     \
                    (out_ssbo)[uint(out_payload_base) + _as_wp + _as_i] =                           \
                        (d_raw_ssbo)[_pr_length_off + _as_i];                                       \
                }                                                                                   \
            }                                                                                       \
            _as_wp += _pr_length_size;                                                              \
                                                                                                    \
            (enc_n_out) = _as_wp;                                                                   \
        }                                                                                           \
    } while (false)

#endif
