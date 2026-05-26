// ── StreamLZ sub-chunk parse-and-dispatch ──────────────────────
// The two device-side helpers that sit between the kernels and the
// LZ hot loops: parseAndDecodeSubChunkRaw (raw L1/L2 inline parse,
// every value in registers) and parseAndDecodeSubChunk (general
// entropy-capable, parseSubChunkHeaders → ParsedStreams → dispatch).
// Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"
#include "lz_decode_core.cuh"
#include "lz_header_parse.cuh"

// ── Fast path: raw L1/L2 sub-chunk (chunk_type 0 only) ────────
// Parses literal / cmd / cmd2 / off16 / off32 headers inline (no
// ParsedStreams struct, no chunk_type dispatch, no off16 entropy
// branch): every parsed value flows as a local → register and is never
// spilled to the stack. Called by slzLzDecodeRawKernel when a
// sub-chunk's mode == 1 (raw literals). The general parseAndDecodeSubChunk
// handles modes 0, 2-15 (entropy).
//
// Lane contract: warp-cooperative; each header is parsed on lane 0 and
// `src` is rebuilt on every lane via broadcastSrc.
__device__ void parseAndDecodeSubChunkRaw(
    const uint8_t* __restrict__ sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* __restrict__ dst,
    uint32_t dst_offset,
    uint32_t base_offset
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    uint32_t initial_copy = 0;
    if (base_offset == 0) {
        // First 8 bytes are raw literals — copy directly to output.
        if ((uint32_t)lane < INITIAL_LITERAL_COPY_BYTES) dst[dst_offset + lane] = src[lane];
        __syncwarp();
        src += INITIAL_LITERAL_COPY_BYTES;
        initial_copy = INITIAL_LITERAL_COPY_BYTES;
    }

    // Literal stream (Type 0, assumed).
    const uint8_t* lit_ptr = src;
    uint32_t lit_size = 0;
    if (lane == 0) {
        lit_size = parseRawStreamSize(src);
        lit_ptr = src;
        src += lit_size;
    }
    lit_size = __shfl_sync(FULL_WARP_MASK, lit_size, 0);
    {
        src = broadcastSrc(sc_src, src);
        lit_ptr = src - lit_size;
    }

    // Command stream (Type 0, assumed).
    const uint8_t* cmd_ptr;
    uint32_t cmd_size = 0;
    if (lane == 0) {
        cmd_size = parseRawStreamSize(src);
        cmd_ptr = src;
        src += cmd_size;
    }
    cmd_size = __shfl_sync(FULL_WARP_MASK, cmd_size, 0);
    {
        src = broadcastSrc(sc_src, src);
        cmd_ptr = src - cmd_size;
    }

    // block2_cmd_offset present only for sub-chunks > 64KB.
    uint32_t block2_cmd_offset = cmd_size;
    if (lane == 0 && sc_decomp_size > LZ_BLOCK_SIZE) {
        block2_cmd_offset = readU16LE(src);
        src += 2;
    }
    // Lane 0 may have updated block2_cmd_offset above (conditional on
    // sc_decomp_size > 64KB); broadcast so every lane sees the same value.
    block2_cmd_offset = __shfl_sync(FULL_WARP_MASK, block2_cmd_offset, 0);
    src = broadcastSrc(sc_src, src);

    // Off16 stream (raw — no entropy 0xFFFF marker handling on this path).
    const uint8_t* off16_raw;
    uint32_t off16_count = 0;
    if (lane == 0) {
        off16_count = readU16LE(src);
        off16_raw = src + 2;
        src += 2 + off16_count * OFF16_ENTRY_BYTES;
    }
    off16_count = __shfl_sync(FULL_WARP_MASK, off16_count, 0);
    {
        src = broadcastSrc(sc_src, src);
        off16_raw = src - off16_count * OFF16_ENTRY_BYTES;
    }

    // Off32 stream sizes (may be empty at sc<=0.25 with 64KB sub-chunks).
    uint32_t off32_count1 = 0, off32_count2 = 0;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
    const uint8_t* len_stream;
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
        len_stream = src;
        len_avail = (uint32_t)((uintptr_t)src_end - (uintptr_t)src);
    }
    off32_count1 = __shfl_sync(FULL_WARP_MASK, off32_count1, 0);
    off32_count2 = __shfl_sync(FULL_WARP_MASK, off32_count2, 0);
    len_avail = __shfl_sync(FULL_WARP_MASK, len_avail, 0);
    {
        src = broadcastSrc(sc_src, src);
        len_stream = src;
        off32_raw2 = src - off32_count2 * OFF32_ENTRY_BYTES;
        off32_raw1 = off32_raw2 - off32_count1 * OFF32_ENTRY_BYTES;
    }

    if (off32_count1 == 0 && off32_count2 == 0) {
        // OFF16_SPLIT=false: raw-off16 path is the only one in raw L1/L2.
        decodeSubChunkRawMode<false>(
            cmd_ptr, cmd_size,
            lit_ptr, lit_size,
            off16_raw, off16_count,
            nullptr, nullptr,  // off16_hi/lo unused at OFF16_SPLIT=false
            len_stream, len_avail,
            dst, sc_decomp_size, initial_copy,
            dst_offset
        );
    } else {
        // Positional brace-init matches ParsedStreams field order in
        // slz_wire_format.cuh (lit_ptr, cmd_ptr, off16_raw, off16_hi,
        // off16_lo, off32_raw1, off32_raw2, len_stream, lit_size,
        // cmd_size, off16_count, off16_split, off32_count1, off32_count2,
        // len_avail, cmd_stream2_offset, initial_copy).
        const ParsedStreams ps_raw = {
            lit_ptr, cmd_ptr, off16_raw,
            nullptr, nullptr, // off16_hi/lo unused (OFF16_SPLIT=false)
            off32_raw1, off32_raw2, len_stream,
            lit_size, cmd_size, off16_count,
            /*off16_split=*/0u,
            off32_count1, off32_count2, len_avail,
            block2_cmd_offset, initial_copy,
        };
        const DecodeOutput out_raw = { dst, sc_decomp_size, dst_offset };
        decodeSubChunkGeneral(ps_raw, out_raw, /*mode=*/1);
    }
}

// ── Sub-chunk decoder dispatch ────────────────────────────────
// General entropy-capable path (used by slzLzDecodeKernel only;
// the slzLzDecodeRawKernel fast path uses parseAndDecodeSubChunkRaw).
// Calls parseSubChunkHeaders (__noinline__) then dispatches to
// decodeSubChunkRawMode or decodeSubChunkGeneral. The header parser's
// registers are freed before the decode hot loop runs.
__device__ void parseAndDecodeSubChunk(
    const uint8_t* __restrict__ sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* __restrict__ dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint32_t mode,
    uint8_t* __restrict__ entropy_lit_scratch,
    uint8_t* __restrict__ entropy_tok_scratch,
    uint8_t* __restrict__ entropy_off16_scratch
) {
    ParsedStreams ps;
    parseSubChunkHeaders(sc_src, sc_comp_size, sc_decomp_size, dst,
                         dst_offset, base_offset, entropy_lit_scratch,
                         entropy_tok_scratch, entropy_off16_scratch, ps);

    if (mode == 1 && ps.off32_count1 == 0 && ps.off32_count2 == 0) {
        // Compile-time dispatch on off16_split: lets the compiler
        // eliminate the unused branch + the unused pointer params for
        // the raw-off16 case (the common L1 path).
        if (ps.off16_split) {
            decodeSubChunkRawMode<true>(
                ps.cmd_ptr, ps.cmd_size,
                ps.lit_ptr, ps.lit_size,
                ps.off16_raw, ps.off16_count,
                ps.off16_hi, ps.off16_lo,
                ps.len_stream, ps.len_avail,
                dst, sc_decomp_size, ps.initial_copy,
                dst_offset
            );
        } else {
            decodeSubChunkRawMode<false>(
                ps.cmd_ptr, ps.cmd_size,
                ps.lit_ptr, ps.lit_size,
                ps.off16_raw, ps.off16_count,
                ps.off16_hi, ps.off16_lo,
                ps.len_stream, ps.len_avail,
                dst, sc_decomp_size, ps.initial_copy,
                dst_offset
            );
        }
    } else {
        const DecodeOutput out_general = { dst, sc_decomp_size, dst_offset };
        decodeSubChunkGeneral(ps, out_general, mode);
    }
}
