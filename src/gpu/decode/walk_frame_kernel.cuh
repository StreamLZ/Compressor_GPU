// ── StreamLZ frame-walk kernel ─────────────────────────────────
// Single-threaded GPU port of decompressOneFrame + gpuBatchDecode's
// per-chunk walk: reads the entire compressed frame from device memory
// and produces a device-resident SlzChunkDesc[] array — the CPU never
// touches the compressed bytes. Included into the single lz_kernel.cu
// translation unit.
#pragma once

#include "slz_wire_format.cuh"

// ── Frame-walk kernel (roadmap 4d Phase 3) ──────────────────────────
// Single-threaded GPU port of decompressOneFrame + gpuBatchDecode's
// per-chunk walk. Reads the entire compressed frame from device memory
// (d_frame) and produces a device-resident SlzChunkDesc[] array — the
// CPU never touches the compressed bytes. Used by the D2D slzDecompress
// entry; legacy host-bounce paths keep their CPU walk.
//
// Scope: handles the slzCompress output shape — Fast codec, no
// dictionary, no parallel-decode-metadata, no checksums. Sets d_status
// non-zero on any unsupported feature so the wrapper can fall back.
//
// Status codes (must match decode/driver.zig FrameWalkStatus):
//   0 = success
//   1 = bad magic                  6 = block header truncated
//   2 = unsupported version        7 = bad internal block magic
//   3 = unsupported codec          8 = bad decoder type
//   4 = dictionary present         9 = chunk header truncated
//   5 = pdm flag / checksums set  10 = bad chunk type
//  11 = chunk_descs buffer overflow
//  12 = frame truncated mid-block
//
// Output ChunkDesc layout matches SlzChunkDesc above.

static constexpr uint32_t SLZ_FRAME_MAGIC          = 0x534C5A31u;
static constexpr uint8_t  SLZ_FRAME_VERSION        = 2;
static constexpr uint8_t  SLZ_CODEC_FAST_LZ        = 1;  // Codec.fast_lz
static constexpr uint32_t SLZ_FRAME_MIN_HDR_SIZE   = 14;
static constexpr uint32_t SLZ_FRAME_END_MARK       = 0;
static constexpr uint32_t SLZ_BLOCK_UNCOMP_FLAG    = 0x80000000u;
static constexpr uint32_t SLZ_BLOCK_PDM_FLAG       = 0x40000000u;
static constexpr uint32_t SLZ_CHUNK_SIZE_MASK      = 0x3FFFFu;   // 256KB - 1
static constexpr uint32_t SLZ_CHUNK_TYPE_SHIFT     = 18;
static constexpr uint32_t SLZ_CHUNK_TYPE_MASK      = 3u << SLZ_CHUNK_TYPE_SHIFT;
static constexpr uint8_t  SLZ_INT_BLOCK_MAGIC      = 0x05;
static constexpr uint8_t  SLZ_DECODER_FAST         = 1;
static constexpr uint8_t  SLZ_DECODER_TURBO        = 2;

__device__ __forceinline__ float walkReadF32LE(const uint8_t* p) {
    union { uint32_t u; float f; } u;
    u.u = readU32LE(p);
    return u.f;
}

// d_block_start / d_block_size: byte range of the block payload within
// d_frame. Wrapper D2D-copies this range into d_comp_persist; chunk_descs
// src_offset is block-payload-relative (matches the CPU walk in
// gpuBatchDecode). Single-block frames only (status 13 otherwise) —
// slzCompress output.
extern "C" __global__ void slzWalkFrameKernel(
    const uint8_t* __restrict__ d_frame,
    uint32_t                    frame_size,
    SlzChunkDesc* __restrict__  d_chunks,
    uint32_t                    max_chunks,
    uint32_t* __restrict__      d_n_chunks,
    uint32_t* __restrict__      d_decompressed_size,
    uint32_t* __restrict__      d_sub_chunk_cap,
    uint32_t* __restrict__      d_block_start,
    uint32_t* __restrict__      d_block_size,
    uint32_t* __restrict__      d_status)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    *d_n_chunks = 0;
    *d_decompressed_size = 0;
    *d_sub_chunk_cap = 0;
    *d_block_start = 0;
    *d_block_size = 0;
    *d_status = 0;

    if (frame_size < SLZ_FRAME_MIN_HDR_SIZE) { *d_status = 12; return; }

    if (readU32LE(d_frame) != SLZ_FRAME_MAGIC) { *d_status = 1; return; }
    const uint8_t version = d_frame[4];
    if (version != SLZ_FRAME_VERSION) { *d_status = 2; return; }
    const uint8_t flags = d_frame[5];
    const uint8_t codec = d_frame[6];
    if (codec != SLZ_CODEC_FAST_LZ) { *d_status = 3; return; }
    // d_frame[7] = level — unused here.
    // d_frame[8] = block_size_log2 (encoded) — unused (we walk by block hdr).
    const float sc_group_size = walkReadF32LE(d_frame + 9);

    uint32_t pos = SLZ_FRAME_MIN_HDR_SIZE;
    const bool content_size_present = (flags & 0x01) != 0;
    const bool dict_id_present      = (flags & 0x02) != 0;
    if (dict_id_present) { *d_status = 4; return; }
    if (content_size_present) pos += 8;
    // No PDM / checksums on the slzCompress path; the loop below
    // rejects them if encountered (status 5).

    // eff_chunk_size mirrors decompressOneFrame/gpuBatchDecode: it's the
    // CHUNK boundary at which the 2-byte internal block header repeats.
    uint32_t eff_chunk_size = (uint32_t)(sc_group_size * 262144.0f);
    if (eff_chunk_size > 262144u) eff_chunk_size = 262144u;
    if (eff_chunk_size == 0) eff_chunk_size = 65536u;
    // sub_chunk_cap is the slot stride used by the scan kernel; mirrors
    // the CPU's `eff_sc_cap = constants.sub_chunk_size` (= 131072).
    const uint32_t sub_chunk_cap = 131072u;

    uint32_t dst_off = 0;
    uint32_t n_chunks = 0;
    uint32_t blocks_seen = 0;

    while (pos + 4 <= frame_size) {
        const uint32_t w0 = readU32LE(d_frame + pos);
        if (w0 == SLZ_FRAME_END_MARK) { break; }
        if (pos + 8 > frame_size) { *d_status = 6; return; }
        const uint32_t decomp_size = readU32LE(d_frame + pos + 4);
        const bool uncompressed_block = (w0 & SLZ_BLOCK_UNCOMP_FLAG) != 0;
        const bool pdm_block          = (w0 & SLZ_BLOCK_PDM_FLAG) != 0;
        const uint32_t comp_size = w0 & ~(SLZ_BLOCK_UNCOMP_FLAG | SLZ_BLOCK_PDM_FLAG);
        pos += 8;
        blocks_seen++;
        if (blocks_seen > 1) { *d_status = 13; return; }
        if (pdm_block) { *d_status = 5; return; }

        const uint32_t block_payload_start = pos;
        *d_block_start = block_payload_start;
        *d_block_size = comp_size;

        if (uncompressed_block) {
            if (pos + decomp_size > frame_size) { *d_status = 12; return; }
            if (n_chunks >= max_chunks) { *d_status = 11; return; }
            // src_offset is BLOCK-PAYLOAD-RELATIVE (matches the CPU walk
            // in gpuBatchDecode). decompressFramedFromDevice passes
            // `d_compressed_src = d_frame + block_start` so the decode
            // kernels' `compressed + src_offset` reads land on the right
            // bytes.
            d_chunks[n_chunks] = { 0u, decomp_size, decomp_size, dst_off, 1u, 0, {0, 0, 0} };
            n_chunks++;
            dst_off += decomp_size;
            pos += comp_size;
            continue;
        }

        if (pos + 2 > frame_size) { *d_status = 12; return; }
        const uint8_t bh0 = d_frame[pos];
        const uint8_t bh1 = d_frame[pos + 1];
        if ((bh0 & 0x0F) != SLZ_INT_BLOCK_MAGIC) { *d_status = 7; return; }
        const uint8_t decoder = bh1 & 0x7F;
        if (decoder != SLZ_DECODER_FAST && decoder != SLZ_DECODER_TURBO) { *d_status = 8; return; }
        const bool use_checksums = (bh1 >> 7) != 0;
        if (use_checksums) { *d_status = 5; return; }
        const uint32_t block_end = pos + comp_size;

        uint32_t dst_remaining = decomp_size;
        bool need_internal_hdr = false;
        pos += 2;

        while (dst_remaining > 0 && pos < block_end) {
            if (need_internal_hdr) {
                if (pos + 2 > block_end) { *d_status = 7; return; }
                const uint8_t b0 = d_frame[pos];
                const uint8_t b1 = d_frame[pos + 1];
                if ((b0 & 0x0F) != SLZ_INT_BLOCK_MAGIC) { *d_status = 7; return; }
                const uint8_t dec2 = b1 & 0x7F;
                if (dec2 != SLZ_DECODER_FAST && dec2 != SLZ_DECODER_TURBO) { *d_status = 8; return; }
                if ((b1 >> 7) != 0) { *d_status = 5; return; }
                pos += 2;
            }

            const uint32_t dst_this = (dst_remaining < eff_chunk_size) ? dst_remaining : eff_chunk_size;

            if (pos + 4 > block_end) { *d_status = 9; return; }
            const uint32_t v = readU32LE(d_frame + pos);
            const uint32_t size_field = v & SLZ_CHUNK_SIZE_MASK;
            const uint32_t chunk_type = (v >> SLZ_CHUNK_TYPE_SHIFT) & 3u;

            if (chunk_type == 0) {
                const uint32_t comp = size_field + 1;
                pos += 4;
                if (pos + comp > block_end) { *d_status = 12; return; }
                if (n_chunks >= max_chunks) { *d_status = 11; return; }
                const uint32_t flg = (comp == dst_this) ? 1u : 0u;
                // src_offset is BLOCK-PAYLOAD-RELATIVE (matches CPU walk).
                d_chunks[n_chunks] = { pos - block_payload_start, comp, dst_this, dst_off, flg, 0, {0, 0, 0} };
                n_chunks++;
                pos += comp;
            } else if (chunk_type == 1) {
                if (pos + 5 > block_end) { *d_status = 9; return; }
                const uint8_t fill = d_frame[pos + 4];
                pos += 5;
                if (n_chunks >= max_chunks) { *d_status = 11; return; }
                d_chunks[n_chunks] = { 0, 0, dst_this, dst_off, 2u, fill, {0, 0, 0} };
                n_chunks++;
            } else {
                *d_status = 10; return;
            }

            dst_off += dst_this;
            dst_remaining -= dst_this;
            need_internal_hdr = true;
        }

        pos = block_end;
    }

    *d_n_chunks = n_chunks;
    *d_decompressed_size = dst_off;
    *d_sub_chunk_cap = sub_chunk_cap;
}
