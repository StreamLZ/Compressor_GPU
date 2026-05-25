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

// ── Sub-chunk header parser — __noinline__ to free registers ────
// Parses all stream headers, returns pointers/sizes in ParsedStreams.
// Registers used here are freed before the decode hot loop runs — the
// __noinline__ qualifier is load-bearing for occupancy; do not remove.
//
// Lane contract: warp-cooperative. Each header is parsed on lane 0,
// then sizes/flags are broadcast and `src` is rebuilt on every lane via
// broadcastSrc before the next header is parsed.
//
// entropy_lit/tok/off16 scratch buffers hold the pre-decoded output of
// the entropy kernels (tANS or Huffman); when a stream's chunk type is
// entropy-coded its ParsedStreams pointer is redirected into scratch.
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
        if (lane < (int)INITIAL_LITERAL_COPY_BYTES) dst[dst_offset + lane] = src[lane];
        __syncwarp();
        src += INITIAL_LITERAL_COPY_BYTES;
        ps.initial_copy = INITIAL_LITERAL_COPY_BYTES;
    }

    // Literal stream
    const uint8_t* lit_ptr = src;
    uint32_t lit_size = 0;
    uint32_t lit_pre_decoded = 0;
    if (lane == 0) {
        uint32_t chunk_type = (src[0] >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;
        if (chunk_type == 0) {
            lit_size = parseRawStreamSize(src);
            lit_ptr = src;
            src += lit_size;
        } else if ((chunk_type == 1 || chunk_type == 6) && entropy_lit_scratch != nullptr) {
            uint32_t comp_size;
            lit_size = parseEntropyHeader(src, comp_size);
            lit_ptr = entropy_lit_scratch;
            lit_pre_decoded = 1;
        } else if (chunk_type == 7 && entropy_lit_scratch != nullptr) {
            // Paired-primary: [0x70][countA:u24 BE][inner type-6 tANS stream].
            // This unit's literals are countA symbols at the start of the
            // decoded combined buffer (the tANS kernel split-wrote them here).
            lit_size = skipPairedPrimary(src);
            lit_ptr = entropy_lit_scratch;
            lit_pre_decoded = 1;
        } else if (chunk_type == 5 && entropy_lit_scratch != nullptr) {
            // Paired-secondary: [0x50][countA:u24 BE][countB:u24 BE], no payload.
            // This unit's literals are countB symbols, split-written by the
            // tANS kernel into this chunk's region (dst_offset_b).
            uint32_t count_b = readBE24(src + 4);
            src += PAIRED_SECONDARY_HEADER_BYTES;
            lit_ptr = entropy_lit_scratch;
            lit_size = count_b;
            lit_pre_decoded = 1;
        } else if (chunk_type == 4 && entropy_lit_scratch != nullptr) {
            // Huffman: same 3/5-byte header convention as type 1/6 tANS;
            // payload is [128 B weights][9 B sub-header][4 streams].
            // Pre-decoded into entropy_lit_scratch by slzHuffDecode4StreamKernel.
            uint32_t comp_size;
            lit_size = parseEntropyHeader(src, comp_size);
            lit_ptr = entropy_lit_scratch;
            lit_pre_decoded = 1;  // reuses the scratch-redirection path
        } else {
            lit_size = 0;
            lit_ptr = src;
        }
    }
    lit_size = __shfl_sync(FULL_WARP_MASK, lit_size, 0);
    lit_pre_decoded = __shfl_sync(FULL_WARP_MASK, lit_pre_decoded, 0);
    {
        src = broadcastSrc(sc_src, src);
        if (lit_pre_decoded) lit_ptr = entropy_lit_scratch;
        else lit_ptr = src - lit_size;
    }
    ps.lit_ptr = lit_ptr;
    ps.lit_size = lit_size;

    // Command stream (tokens)
    const uint8_t* cmd_ptr;
    uint32_t cmd_size = 0;
    uint32_t cmd_pre_decoded = 0;
    if (lane == 0) {
        uint32_t ct = (src[0] >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;
        if (ct == 0) {
            cmd_size = parseRawStreamSize(src);
            cmd_ptr = src;
            src += cmd_size;
        } else if ((ct == 1 || ct == 6) && entropy_tok_scratch != nullptr) {
            // tANS-encoded token stream: skip compressed data, use pre-decoded buffer
            uint32_t comp_size;
            cmd_size = parseEntropyHeader(src, comp_size);
            cmd_ptr = entropy_tok_scratch;
            cmd_pre_decoded = 1;
        } else if (ct == 7 && entropy_tok_scratch != nullptr) {
            // Paired-primary token stream: [0x70][countA:u24][inner type-6 stream]
            cmd_size = skipPairedPrimary(src);
            cmd_ptr = entropy_tok_scratch;
            cmd_pre_decoded = 1;
        } else if (ct == 5 && entropy_tok_scratch != nullptr) {
            // Paired-secondary token stream: [0x50][countA:u24][countB:u24]
            uint32_t count_b = readBE24(src + 4);
            src += PAIRED_SECONDARY_HEADER_BYTES;
            cmd_ptr = entropy_tok_scratch;
            cmd_size = count_b;
            cmd_pre_decoded = 1;
        } else if (ct == 4 && entropy_tok_scratch != nullptr) {
            // Huffman token stream — pre-decoded by slzHuffDecode4StreamKernel
            // into entropy_tok_scratch. Same wire format as type-1 tANS.
            uint32_t comp_size;
            cmd_size = parseEntropyHeader(src, comp_size);
            cmd_ptr = entropy_tok_scratch;
            cmd_pre_decoded = 1;
        } else {
            // Unsupported entropy type (chunk_type ∉ {0, 4} on GPU). Advance
            // src past the stream so subsequent parsers see correct offsets,
            // but signal an empty cmd to the caller — the decoder cannot
            // consume this payload.
            (void)skipEntropyStream(src);
            cmd_size = 0;
        }
    }
    cmd_size = __shfl_sync(FULL_WARP_MASK, cmd_size, 0);
    cmd_pre_decoded = __shfl_sync(FULL_WARP_MASK, cmd_pre_decoded, 0);
    {
        src = broadcastSrc(sc_src, src);
        if (cmd_pre_decoded) cmd_ptr = entropy_tok_scratch;
        else cmd_ptr = src - cmd_size;
    }
    ps.cmd_ptr = cmd_ptr;
    ps.cmd_size = cmd_size;

    // block2_cmd_offset
    uint32_t block2_cmd_offset = cmd_size;
    if (lane == 0 && sc_decomp_size > LZ_BLOCK_SIZE) {
        uint16_t v; memcpy(&v, src, 2);
        block2_cmd_offset = v;
        src += 2;
    }
    block2_cmd_offset = __shfl_sync(FULL_WARP_MASK, block2_cmd_offset, 0);
    src = broadcastSrc(sc_src, src);
    ps.cmd_stream2_offset = block2_cmd_offset;

    // Off16 stream
    const uint8_t* off16_raw;
    const uint8_t* off16_hi_ptr = nullptr;
    const uint8_t* off16_lo_ptr = nullptr;
    uint32_t off16_count = 0;
    uint32_t off16_is_entropy = 0;
    uint32_t off16_is_split = 0;
    if (lane == 0) {
        uint16_t cnt; memcpy(&cnt, src, 2);
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
    {
        src = broadcastSrc(sc_src, src);
        if (off16_is_entropy) {
            off16_hi_ptr = entropy_off16_scratch;
            off16_lo_ptr = entropy_off16_scratch + OFF16_HILO_SPLIT_OFFSET;
            off16_raw = nullptr;
        } else {
            off16_raw = src - off16_count * OFF16_ENTRY_BYTES;
        }
    }
    ps.off16_raw = off16_raw;
    ps.off16_hi = off16_hi_ptr;
    ps.off16_lo = off16_lo_ptr;
    ps.off16_count = off16_count;
    ps.off16_split = off16_is_split;

    // Off32 stream sizes
    uint32_t off32_count1 = 0, off32_count2 = 0;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
    uint32_t len_avail = 0;

    if (lane == 0) {
        uint32_t tmp = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
        src += 3;
        if (tmp != 0) {
            off32_count1 = tmp >> OFF32_COUNT1_SHIFT;
            off32_count2 = tmp & OFF32_COUNT2_MASK;
            if (off32_count1 == OFF32_COUNT_PACK_MAX) { uint16_t v; memcpy(&v, src, 2); off32_count1 = v; src += 2; }
            if (off32_count2 == OFF32_COUNT_PACK_MAX) { uint16_t v; memcpy(&v, src, 2); off32_count2 = v; src += 2; }
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
    {
        src = broadcastSrc(sc_src, src);
        off32_raw2 = src - off32_count2 * OFF32_ENTRY_BYTES;
        off32_raw1 = off32_raw2 - off32_count1 * OFF32_ENTRY_BYTES;
    }
    ps.off32_raw1 = off32_raw1;
    ps.off32_raw2 = off32_raw2;
    ps.off32_count1 = off32_count1;
    ps.off32_count2 = off32_count2;
    ps.len_stream = src;
    ps.len_avail = len_avail;
}
