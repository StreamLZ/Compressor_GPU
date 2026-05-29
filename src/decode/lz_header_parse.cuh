// ── StreamLZ sub-chunk header parser ───────────────────────────
// parseSubChunkHeaders reads every per-stream header of one sub-chunk
// (literal, token, off16, off32, length) and returns the resolved
// pointers and sizes in a ParsedStreams struct. Included into the
// single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"

// ── Helper: rebuild a per-warp `src` cursor after lane-0 parse ─
// Lane 0 alone advances `src` while parsing a header; this broadcasts
// the resulting byte offset (relative to sc_src) so every lane rebuilds
// the same cursor. Removes a class of "forgot to broadcast" bugs.
__device__ inline const uint8_t* broadcastSrc(const uint8_t* sc_src,
                                              const uint8_t* src) {
    uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
    so = __shfl_sync(FULL_WARP_MASK, so, 0);
    return sc_src + so;
}

// ── Per-stream parsers ─────────────────────────────────────────
// Each parser runs the lane-0 wire-format walk for one stream, then
// broadcasts size/flag scalars and rebuilds `src` on every lane via
// broadcastSrc. Split out of parseSubChunkHeaders so the outer fn stays
// readable; each is left without an inline hint so nvcc decides - the
// outer __noinline__ qualifier is what protects the decode hot loop
// from header-parse register pressure.

// Literal and command streams share an identical wire-format dispatch:
// `chunk_type` ∈ {0, 4}. Type 0 is a raw stream copied verbatim; type 4
// is Huffman-coded and read from the matching entropy scratch
// (pre-decoded by `slzHuffDecode4StreamKernel`). Any other type is
// unsupported — the literal path treats it as an empty stream and leaves
// `src` untouched; the command path advances past the unknown entropy
// header so subsequent parsers see correct offsets.
static __device__ void parseLiteralStreamHeader(
    const uint8_t* sc_src,
    const uint8_t*& src,
    uint8_t* entropy_lit_scratch,
    ParsedStreams& ps
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint8_t* lit_ptr = src;
    uint32_t lit_size = 0;
    uint32_t lit_pre_decoded = 0;
    if (lane == 0) {
        uint32_t chunk_type = (src[0] >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;
        if (chunk_type == 0) {
            lit_size = parseRawStreamSize(src);
            lit_ptr = src;
            src += lit_size;
        } else if (chunk_type == HUFF_CHUNK_TYPE && entropy_lit_scratch != nullptr) {
            // Huffman: 5-byte header + [128 B weights][93 B sub-header]
            // [32 streams] payload (see `HUFF_NUM_STREAMS` /
            // `HUFF_BODY_HEADER_BYTES` in `common/gpu_huffman.cuh`).
            // Pre-decoded into entropy_lit_scratch by
            // `slzHuffDecode4StreamKernel` (name retained for ABI;
            // decodes 32 streams now, not 4).
            uint32_t comp_size;
            lit_size = parseEntropyHeader(src, comp_size);
            lit_ptr = entropy_lit_scratch;
            lit_pre_decoded = 1;
        } else {
            lit_size = 0;
            lit_ptr = src;
        }
    }
    lit_size = __shfl_sync(FULL_WARP_MASK, lit_size, 0);
    lit_pre_decoded = __shfl_sync(FULL_WARP_MASK, lit_pre_decoded, 0);
    src = broadcastSrc(sc_src, src);
    if (lit_pre_decoded) lit_ptr = entropy_lit_scratch;
    else lit_ptr = src - lit_size;
    ps.lit_ptr = lit_ptr;
    ps.lit_size = lit_size;
}

static __device__ void parseCommandStreamHeader(
    const uint8_t* sc_src,
    const uint8_t*& src,
    uint32_t sc_decomp_size,
    uint8_t* entropy_tok_scratch,
    ParsedStreams& ps
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint8_t* cmd_ptr;
    uint32_t cmd_size = 0;
    uint32_t cmd_pre_decoded = 0;
    if (lane == 0) {
        uint32_t ct = (src[0] >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;
        if (ct == 0) {
            cmd_size = parseRawStreamSize(src);
            cmd_ptr = src;
            src += cmd_size;
        } else if (ct == HUFF_CHUNK_TYPE && entropy_tok_scratch != nullptr) {
            // Huffman token stream (GPU-emitted): pre-decoded by
            // `slzHuffDecode4StreamKernel` into `entropy_tok_scratch`.
            uint32_t comp_size;
            cmd_size = parseEntropyHeader(src, comp_size);
            cmd_ptr = entropy_tok_scratch;
            cmd_pre_decoded = 1;
        } else {
            // Unsupported chunk type (only 0 and 4 are produced by the
            // GPU encoder). Advance `src` past the stream via the generic
            // entropy-header skip so subsequent parsers see correct
            // offsets, but signal an empty cmd to the caller — the
            // decoder cannot consume the payload.
            (void)skipEntropyStream(src);
            cmd_size = 0;
        }
    }
    cmd_size = __shfl_sync(FULL_WARP_MASK, cmd_size, 0);
    cmd_pre_decoded = __shfl_sync(FULL_WARP_MASK, cmd_pre_decoded, 0);
    src = broadcastSrc(sc_src, src);
    if (cmd_pre_decoded) cmd_ptr = entropy_tok_scratch;
    else cmd_ptr = src - cmd_size;
    ps.cmd_ptr = cmd_ptr;
    ps.cmd_size = cmd_size;

    // block2_cmd_offset: only present when the sub-chunk spans two LZ blocks.
    uint32_t block2_cmd_offset = cmd_size;
    if (lane == 0 && sc_decomp_size > LZ_BLOCK_SIZE) {
        block2_cmd_offset = readU16LE(src);
        src += 2;
    }
    block2_cmd_offset = __shfl_sync(FULL_WARP_MASK, block2_cmd_offset, 0);
    src = broadcastSrc(sc_src, src);
    ps.cmd_stream2_offset = block2_cmd_offset;
}

static __device__ void parseOff16StreamHeader(
    const uint8_t* sc_src,
    const uint8_t*& src,
    uint8_t* entropy_off16_scratch,
    ParsedStreams& ps
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint8_t* off16_raw;
    const uint8_t* off16_hi_ptr = nullptr;
    const uint8_t* off16_lo_ptr = nullptr;
    uint32_t off16_count = 0;
    uint32_t off16_is_entropy = 0;
    uint32_t off16_is_split = 0;
    if (lane == 0) {
        uint16_t cnt = readU16LE(src);
        if (cnt == OFF16_ENTROPY_MARKER && entropy_off16_scratch != nullptr) {
            // Entropy-coded off16: skip the two encoded sub-streams,
            // read pre-decoded hi/lo bytes from entropy off16 scratch
            src += 2;
            uint32_t hi_size = skipEntropyStream(src);
            uint32_t lo_size = skipEntropyStream(src);
            // hi_size and lo_size should be equal; use hi_size as count
            off16_count = hi_size;
            if (lo_size != hi_size) off16_count = (lo_size < hi_size) ? lo_size : hi_size;
            // off16 scratch layout: hi bytes at offset 0, lo bytes at +65536
            off16_hi_ptr = entropy_off16_scratch;
            off16_lo_ptr = entropy_off16_scratch + OFF16_HILO_SPLIT_OFFSET;
            off16_raw = nullptr;
            off16_is_entropy = 1;
            off16_is_split = 1;
        } else {
            off16_count = cnt;
            off16_raw = src + 2;
            src += 2 + off16_count * OFF16_ENTRY_BYTES;
        }
    }
    off16_count = __shfl_sync(FULL_WARP_MASK, off16_count, 0);
    off16_is_entropy = __shfl_sync(FULL_WARP_MASK, off16_is_entropy, 0);
    off16_is_split = __shfl_sync(FULL_WARP_MASK, off16_is_split, 0);
    src = broadcastSrc(sc_src, src);
    if (off16_is_entropy) {
        off16_hi_ptr = entropy_off16_scratch;
        off16_lo_ptr = entropy_off16_scratch + OFF16_HILO_SPLIT_OFFSET;
        off16_raw = nullptr;
    } else {
        off16_raw = src - off16_count * OFF16_ENTRY_BYTES;
    }
    ps.off16_raw = off16_raw;
    ps.off16_hi = off16_hi_ptr;
    ps.off16_lo = off16_lo_ptr;
    ps.off16_count = off16_count;
    ps.off16_split = off16_is_split;
}

// Off32 (raw) and length-tail are parsed together because they share
// the same lane-0 cursor walk: off32 counts/pointers are computed, then
// the remaining bytes form the length stream - splitting would force a
// second broadcast that the original code intentionally avoids.
static __device__ void parseOff32StreamHeaders(
    const uint8_t* sc_src,
    const uint8_t*& src,
    const uint8_t* src_end,
    ParsedStreams& ps
) {
    const int lane = threadIdx.x & LANE_MASK;
    uint32_t off32_count1 = 0, off32_count2 = 0;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
    uint32_t len_avail = 0;

    if (lane == 0) {
        uint32_t tmp = readLE24(src);
        src += 3;
        if (tmp != 0) {
            off32_count1 = tmp >> OFF32_COUNT1_SHIFT;
            off32_count2 = tmp & OFF32_COUNT2_MASK;
            if (off32_count1 == OFF32_COUNT_PACK_MAX) { off32_count1 = readU16LE(src); src += 2; }
            if (off32_count2 == OFF32_COUNT_PACK_MAX) { off32_count2 = readU16LE(src); src += 2; }
            off32_raw1 = src;
            src += off32_count1 * OFF32_ENTRY_BYTES;
            off32_raw2 = src;
            src += off32_count2 * OFF32_ENTRY_BYTES;
        } else {
            off32_raw1 = src;
            off32_raw2 = src;
        }
        len_avail = (uint32_t)((uintptr_t)src_end - (uintptr_t)src);
    }
    off32_count1 = __shfl_sync(FULL_WARP_MASK, off32_count1, 0);
    off32_count2 = __shfl_sync(FULL_WARP_MASK, off32_count2, 0);
    len_avail = __shfl_sync(FULL_WARP_MASK, len_avail, 0);
    src = broadcastSrc(sc_src, src);
    off32_raw2 = src - off32_count2 * OFF32_ENTRY_BYTES;
    off32_raw1 = off32_raw2 - off32_count1 * OFF32_ENTRY_BYTES;
    ps.off32_raw1 = off32_raw1;
    ps.off32_raw2 = off32_raw2;
    ps.off32_count1 = off32_count1;
    ps.off32_count2 = off32_count2;
    ps.len_stream = src;
    ps.len_avail = len_avail;
}

// ── Sub-chunk header parser - __noinline__ to free registers ────
// Parses all stream headers, returns pointers/sizes in ParsedStreams.
// Registers used here are freed before the decode hot loop runs - the
// __noinline__ qualifier is load-bearing for occupancy; do not remove.
//
// Lane contract: warp-cooperative. Each header is parsed on lane 0,
// then sizes/flags are broadcast and `src` is rebuilt on every lane via
// broadcastSrc before the next header is parsed.
//
// `entropy_lit/tok/off16` scratch buffers hold the pre-decoded output
// of `slzHuffDecode4StreamKernel`; when a stream's chunk type is 4
// (Huffman) its `ParsedStreams` pointer is redirected into scratch.
__device__ __noinline__ void parseSubChunkHeaders(
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint8_t* entropy_lit_scratch,
    uint8_t* entropy_tok_scratch,
    uint8_t* entropy_off16_scratch,
    ParsedStreams& ps
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    ps.initial_copy = 0;
    if (base_offset == 0) {
        if ((uint32_t)lane < INITIAL_LITERAL_COPY_BYTES) dst[dst_offset + lane] = src[lane];
        __syncwarp();
        src += INITIAL_LITERAL_COPY_BYTES;
        ps.initial_copy = INITIAL_LITERAL_COPY_BYTES;
    }

    parseLiteralStreamHeader(sc_src, src, entropy_lit_scratch, ps);
    parseCommandStreamHeader(sc_src, src, sc_decomp_size, entropy_tok_scratch, ps);
    parseOff16StreamHeader(sc_src, src, entropy_off16_scratch, ps);
    parseOff32StreamHeaders(sc_src, src, src_end, ps);
}
