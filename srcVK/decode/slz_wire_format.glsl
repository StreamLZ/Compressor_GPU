// 1:1 port of src/decode/slz_wire_format.cuh.
// Device-side wire-format constants + descriptor structs + header
// parsers. Must match srcVK/decode/descriptors.zig byte-for-byte.
// See srcVK/PortInstructions.md.

#ifndef SRCVK_DECODE_SLZ_WIRE_FORMAT_GLSL
#define SRCVK_DECODE_SLZ_WIRE_FORMAT_GLSL

#include "gpu_warp.glsl"
#include "gpu_byteio.glsl"
#include "gpu_wire_format.glsl"

// CUDA reference: src/decode/slz_wire_format.cuh:22-29. Token format
// constants. Tokens encode lit-len + match-len + offset type in one byte.
const uint TOKEN_SHORT_MIN      = 24u;
const uint TOKEN_LONG_LITERAL   = 0u;
const uint TOKEN_LONG_NEAR      = 1u;
const uint TOKEN_LONG_FAR       = 2u;
const uint LONG_LITERAL_BASE    = 64u;
const uint LONG_NEAR_BASE       = 91u;
const uint LONG_FAR_BASE        = 29u;
const uint SHORT_FAR_BASE       = 5u;

// CUDA reference: src/decode/slz_wire_format.cuh:32-36. Standard-token
// bit-field extraction (token >= TOKEN_SHORT_MIN).
const uint TOKEN_LIT_MASK         = 7u;
const uint TOKEN_MATCH_SHIFT      = 3u;
const uint TOKEN_MATCH_MASK       = 0xFu;
const uint TOKEN_USE_RECENT_SHIFT = 7u;
const uint TOKEN_USE_RECENT_MASK  = 1u;

// CUDA reference: src/decode/slz_wire_format.cuh:46. Parallel match copy
// minimum length.
const uint MIN_PARALLEL_MATCH_LEN = 2u;

// CUDA reference: src/decode/slz_wire_format.cuh:51-52. Extended length
// stream encoding.
const uint EXT_LENGTH_EXTRA_BYTES = 2u;
const uint EXT_LENGTH_SCALE       = 4u;

// CUDA reference: src/decode/slz_wire_format.cuh:55-56. Off16/off32 entry
// sizes.
const uint OFF16_ENTRY_BYTES = 2u;
const uint OFF32_ENTRY_BYTES = 3u;

// CUDA reference: src/decode/slz_wire_format.cuh:61. Stream-header bit
// layout: byte 0 high bit selects the short form.
const uint HEADER_LONG_FORM_BIT  = 0x80u;

// CUDA reference: src/decode/slz_wire_format.cuh:64. Type-0 short header.
const uint TYPE0_SHORT_SIZE_MASK = 0xFFFu;

// CUDA reference: src/decode/slz_wire_format.cuh:68-75. Entropy chunk
// header field layout.
const uint ENTROPY_HEADER_SHORT_BYTES = 3u;
const uint ENTROPY_HEADER_LONG_BYTES  = 5u;
const uint ENTROPY_SHORT_COMP_MASK    = 0x3FFu;
const uint ENTROPY_SHORT_DELTA_SHIFT  = 10u;
const uint ENTROPY_SHORT_DELTA_MASK   = 0x3FFu;
const uint ENTROPY_LONG_SIZE_MASK     = 0x3FFFFu;
const uint ENTROPY_LONG_DELTA_SHIFT   = 18u;
const uint ENTROPY_LONG_HI_SHIFT      = 14u;

// CUDA reference: src/decode/slz_wire_format.cuh:82. Off16 hi/lo split
// offset (= LZ_BLOCK_SIZE).
const uint OFF16_HILO_SPLIT_OFFSET = 65536u;

// CUDA reference: src/decode/slz_wire_format.cuh:89-90. Off32 stream
// header (3-byte LE field): count1 in high 12 bits, count2 in low 12 bits.
const uint OFF32_COUNT1_SHIFT = OFF32_COUNT_FIELD_BITS; // 12
const uint OFF32_COUNT2_MASK  = OFF32_COUNT_PACK_MAX;   // 0xFFF

// CUDA reference: src/decode/slz_wire_format.cuh:97. Per-sub-chunk
// entropy-scratch slot size (128KB).
const uint ENTROPY_SCRATCH_SLOT_BYTES = 131072u;

// CUDA reference: src/decode/slz_wire_format.cuh:100-102. Kernel launch
// geometry: 2 warps (64 threads) per block.
const uint WARPS_PER_BLOCK             = 2u;
const uint LZ_KERNEL_BLOCK_THREADS     = 64u;
const uint LZ_KERNEL_MIN_BLOCKS_PER_SM = 24u;

// CUDA reference: src/decode/slz_wire_format.cuh:107-115. SlzChunkDesc.
// Must match descriptors.zig ChunkDesc (24 bytes total).
struct SlzChunkDesc {
    uint src_offset;     // byte offset into compressed block
    uint comp_size;      // compressed payload size
    uint decomp_size;    // decompressed size (usually 256KB)
    uint dst_offset;     // absolute output position
    uint flags;          // bit 0: uncompressed, bit 1: memset
    uint memset_and_reserved; // memset_fill in low byte + reserved[3] in upper 3 bytes
};

// CUDA reference: src/decode/slz_wire_format.cuh:121-125. SlzRawOff16Desc
// (12 bytes total).
struct SlzRawOff16Desc {
    uint src_offset;
    uint size;
    uint gpu_offset;
};

// VK adaptation: GLSL std430 layout for SlzChunkDesc packs the memset_fill
// byte plus reserved[3] into a single u32 (memset_and_reserved). The low
// 8 bits carry memset_fill (matching the CUDA struct's byte ordering);
// upper 24 bits are the reserved padding bytes.
uint chunkMemsetFill(SlzChunkDesc ch) {
    return ch.memset_and_reserved & 0xFFu;
}

// CUDA reference: src/decode/slz_wire_format.cuh:161-177. Read extended
// length from the length stream. VK adaptation: length_stream is read
// from an SSBO via loadCompByte(length_stream_off + length_offset) at the
// call site; this helper takes the resolved bytes via a function-like
// macro pattern below.
//
// The macro mirrors the CUDA signature shape — `length_offset` is an
// inout u32 and `length_remaining` is the bound. The macro expands to
// a sequence that reads from the SSBO `len_ssbo` at `len_base + length_offset`,
// updates `length_offset`, and yields the decoded length into `result_var`.
#define readLength(len_ssbo, len_base, length_offset, length_remaining, result_var) \
    do {                                                                           \
        if ((length_offset) >= (length_remaining)) {                               \
            (result_var) = 0u;                                                     \
        } else {                                                                   \
            uint _rl_v = uint((len_ssbo)[(len_base) + (length_offset)]);           \
            if (_rl_v > EXT_LENGTH_THRESHOLD) {                                    \
                if ((length_offset) + EXT_LENGTH_EXTRA_BYTES >= (length_remaining)) { \
                    (length_offset) = (length_offset) + 1u;                        \
                    (result_var) = _rl_v;                                          \
                } else {                                                           \
                    uint _rl_b0 = uint((len_ssbo)[(len_base) + (length_offset) + 1u]); \
                    uint _rl_b1 = uint((len_ssbo)[(len_base) + (length_offset) + 2u]); \
                    uint _rl_extra = readU16LE(_rl_b0, _rl_b1);                    \
                    _rl_v = _rl_v + _rl_extra * EXT_LENGTH_SCALE;                  \
                    (length_offset) = (length_offset) + EXT_LENGTH_EXTRA_BYTES + 1u; \
                    (result_var) = _rl_v;                                          \
                }                                                                  \
            } else {                                                               \
                (length_offset) = (length_offset) + 1u;                            \
                (result_var) = _rl_v;                                              \
            }                                                                      \
        }                                                                          \
    } while (false)

// CUDA reference: src/decode/slz_wire_format.cuh:189-204. Type-0 header
// fields (cursor-free parse).
struct Type0HdrFields {
    uint size;
    uint header_bytes; // 2 or 3
};

// VK adaptation: the CUDA helper takes `const uint8_t* p`. The GLSL form
// receives the three bytes already loaded from the SSBO by the caller
// (loadCompByte at p+0, p+1, p+2). All other arithmetic is identical.
Type0HdrFields parseType0HdrFields(uint p0, uint p1, uint p2) {
    Type0HdrFields h;
    if (p0 >= HEADER_LONG_FORM_BIT) {
        h.size = ((p0 << 8) | p1) & TYPE0_SHORT_SIZE_MASK;
        h.header_bytes = 2u;
    } else {
        h.size = readBE24(p0, p1, p2);
        h.header_bytes = 3u;
    }
    return h;
}

// CUDA reference: src/decode/slz_wire_format.cuh:206-228. Entropy header
// fields.
struct EntropyHdrFields {
    uint comp_size;
    uint dst_size;
    uint header_bytes; // ENTROPY_HEADER_SHORT_BYTES or ENTROPY_HEADER_LONG_BYTES
};

// VK adaptation: caller loads the 5 candidate header bytes from the SSBO
// before invocation. Short form needs the first 3, long form needs all 5.
EntropyHdrFields parseEntropyHdrFields(uint p0, uint p1, uint p2, uint p3, uint p4) {
    EntropyHdrFields h;
    if (p0 >= HEADER_LONG_FORM_BIT) {
        uint bits = readBE24(p0, p1, p2);
        h.comp_size    = bits & ENTROPY_SHORT_COMP_MASK;
        h.dst_size     = h.comp_size
                       + ((bits >> ENTROPY_SHORT_DELTA_SHIFT) & ENTROPY_SHORT_DELTA_MASK) + 1u;
        h.header_bytes = ENTROPY_HEADER_SHORT_BYTES;
    } else {
        uint bits = readU32BE(p1, p2, p3, p4);
        h.comp_size    = bits & ENTROPY_LONG_SIZE_MASK;
        h.dst_size     = (((bits >> ENTROPY_LONG_DELTA_SHIFT)
                        | (p0 << ENTROPY_LONG_HI_SHIFT)) & ENTROPY_LONG_SIZE_MASK) + 1u;
        h.header_bytes = ENTROPY_HEADER_LONG_BYTES;
    }
    return h;
}

// CUDA reference: src/decode/slz_wire_format.cuh:136-154. ParsedStreams.
// Filled by parseSubChunkHeaders, consumed by the decoders. The seven
// pointer fields in CUDA (lit_ptr / cmd_ptr / off16_raw / off16_hi /
// off16_lo / off32_raw1 / off32_raw2 / len_stream) become (ssbo, base)
// pairs at the call site; in this struct they collapse to byte offsets
// because GLSL cannot store SSBO references. The SSBO they index into
// is implied by the kernel binding the consuming macro takes as a
// parameter.
struct ParsedStreams {
    uint lit_ptr;         // byte offset of literal-byte stream
    uint cmd_ptr;         // byte offset of token (command) stream
    uint off16_raw;       // byte offset of interleaved u16 off16 stream
    uint off16_hi;        // byte offset of hi-byte half of split off16 stream
    uint off16_lo;        // byte offset of lo-byte half of split off16 stream
    uint off32_raw1;      // byte offset of off32 triples for LZ block 0
    uint off32_raw2;      // byte offset of off32 triples for LZ block 1
    uint len_stream;      // byte offset of extended-length side stream
    uint lit_size;        // literal stream length in bytes
    uint cmd_size;        // token stream length in bytes
    uint off16_count;     // number of off16 entries
    uint off16_split;     // 1 = hi/lo split format, 0 = interleaved u16
    uint off32_count1;    // off32 entry count for LZ block 0
    uint off32_count2;    // off32 entry count for LZ block 1
    uint len_avail;       // bytes available in len_stream
    uint cmd_stream2_offset; // byte offset in cmd stream where block 1 begins
    uint initial_copy;    // 8 if this sub-chunk did the initial copy, else 0
    // VK adaptation: GLSL cannot store SSBO references inside a struct, so
    // we need explicit per-stream flags to pick the right SSBO at consume
    // time. CUDA collapses these into pointer typing (lit_ptr / cmd_ptr
    // either point into the compressed blob or into entropy scratch). The
    // flags are written by parseLiteralStreamHeader / parseCommandStreamHeader
    // and consumed by the dispatch macro in lz_dispatch.glsl.
    uint lit_in_scratch;  // 1 if lit_ptr indexes entropy_lit_ssbo, else compressed
    uint cmd_in_scratch;  // 1 if cmd_ptr indexes entropy_tok_ssbo, else compressed
};

// CUDA reference: src/decode/slz_wire_format.cuh:242-246. Cursor-
// advancing wrapper that parses a Type-0 raw-stream size header,
// advances `src_cursor` past the header, and yields the raw size into
// `result_var`.
//
// VK adaptation: GLSL cannot pass an SSBO via a function parameter, so
// the CUDA `parseRawStreamSize(const uint8_t*& src)` helper expands into
// a function-like macro that takes the SSBO + base cursor and the
// inout cursor by name. The caller loads three candidate header bytes
// from `(src_ssbo, src_cursor)` via readBE24-equivalent unpacking; the
// arithmetic mirrors the CUDA implementation.
#define parseRawStreamSize(src_ssbo, src_cursor, result_var)                          \
    do {                                                                              \
        uint _prss_p0 = uint((src_ssbo)[(src_cursor) + 0u]);                          \
        uint _prss_p1 = uint((src_ssbo)[(src_cursor) + 1u]);                          \
        uint _prss_p2 = uint((src_ssbo)[(src_cursor) + 2u]);                          \
        Type0HdrFields _prss_h = parseType0HdrFields(_prss_p0, _prss_p1, _prss_p2);   \
        (src_cursor) = (src_cursor) + _prss_h.header_bytes;                           \
        (result_var) = _prss_h.size;                                                  \
    } while (false)

// CUDA reference: src/decode/slz_wire_format.cuh:263-269. Cursor-
// advancing wrapper that parses an entropy chunk header, advances
// `src_cursor` past the header + payload, writes the compressed size
// into `out_comp_size_var`, and yields the decompressed size into
// `result_var`.
//
// VK adaptation: macro form mirroring parseRawStreamSize. The five
// candidate header bytes are loaded from `(src_ssbo, src_cursor)` by
// the macro itself.
#define parseEntropyHeader(src_ssbo, src_cursor, out_comp_size_var, result_var)         \
    do {                                                                                \
        uint _peh_p0 = uint((src_ssbo)[(src_cursor) + 0u]);                             \
        uint _peh_p1 = uint((src_ssbo)[(src_cursor) + 1u]);                             \
        uint _peh_p2 = uint((src_ssbo)[(src_cursor) + 2u]);                             \
        uint _peh_p3 = uint((src_ssbo)[(src_cursor) + 3u]);                             \
        uint _peh_p4 = uint((src_ssbo)[(src_cursor) + 4u]);                             \
        EntropyHdrFields _peh_h = parseEntropyHdrFields(_peh_p0, _peh_p1, _peh_p2,      \
                                                        _peh_p3, _peh_p4);              \
        (src_cursor) = (src_cursor) + _peh_h.header_bytes + _peh_h.comp_size;           \
        (out_comp_size_var) = _peh_h.comp_size;                                         \
        (result_var) = _peh_h.dst_size;                                                 \
    } while (false)

// CUDA reference: src/decode/slz_wire_format.cuh:279-288. Reads the
// header at `(src_ssbo, src_cursor)`, advances the cursor past the
// header + payload, and yields the decompressed size into `result_var`.
//
// VK adaptation: macro form, same dispatch as CUDA — chunk-type 0
// dispatches to parseRawStreamSize and advances past the raw payload;
// any other type dispatches to parseEntropyHeader. The CUDA helper
// returns the decompressed size; the macro writes it into result_var.
#define skipEntropyStream(src_ssbo, src_cursor, result_var)                             \
    do {                                                                                \
        uint _ses_b0 = uint((src_ssbo)[(src_cursor) + 0u]);                             \
        uint _ses_ct = (_ses_b0 >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;                 \
        if (_ses_ct == 0u) {                                                            \
            uint _ses_sz;                                                               \
            parseRawStreamSize(src_ssbo, src_cursor, _ses_sz);                          \
            (src_cursor) = (src_cursor) + _ses_sz;                                      \
            (result_var) = _ses_sz;                                                     \
        } else {                                                                        \
            uint _ses_comp;                                                             \
            uint _ses_dst;                                                              \
            parseEntropyHeader(src_ssbo, src_cursor, _ses_comp, _ses_dst);              \
            (result_var) = _ses_dst;                                                    \
        }                                                                               \
    } while (false)

#endif
