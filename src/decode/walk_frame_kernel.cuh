// ── StreamLZ frame-walk kernel ─────────────────────────────────
// Single-threaded GPU port of decompressOneFrame + gpuBatchDecode's
// per-chunk walk: reads the entire compressed frame from device memory
// and produces a device-resident SlzChunkDesc[] array - the CPU never
// touches the compressed bytes. Included into the single lz_kernel.cu
// translation unit.
#pragma once

#include "slz_wire_format.cuh"

// ── Frame-walk kernel ───────────────────────────────────────────────
// Single-threaded GPU port of decompressOneFrame + gpuBatchDecode's
// per-chunk walk. Reads the entire compressed frame from device memory
// (d_frame) and produces a device-resident SlzChunkDesc[] array - the
// CPU never touches the compressed bytes. Used by the D2D slzDecompress
// entry; legacy host-bounce paths keep their CPU walk.
//
// Scope: handles the slzCompress output shape - Fast codec, no
// parallel-decode-metadata, no checksums. Dictionary frames walk
// normally (v4 #16) - the host resolves the dictionary ID. Sets
// d_status non-zero on any unsupported feature so the wrapper can
// fall back.
//
// Status codes (must match decode/driver.zig FrameWalkStatus):
//   0 = success
//   1 = bad magic                  6 = block header truncated
//   2 = unsupported version        7 = bad internal block magic
//   3 = unsupported codec          8 = bad decoder type
//   4 = retired (was: dictionary   9 = chunk header truncated
//       present - pre-v4 #16)
//   5 = pdm flag / checksums set  10 = bad chunk type
//  11 = chunk_descs buffer overflow
//  12 = frame truncated mid-block
//
// Output ChunkDesc layout matches SlzChunkDesc above.
// SLZ_FRAME_* constants live in common/gpu_wire_format.cuh (frame ABI).

__device__ __forceinline__ float walkReadF32LE(const uint8_t* p) {
    // memcpy (rather than a union type-pun) is the defined-behavior
    // bit-cast pattern in C++17+; nvcc compiles it to the same single
    // ld.global as the union form.
    float v;
    uint32_t raw = readU32LE(p);
    memcpy(&v, &raw, sizeof(v));
    return v;
}

// d_block_start / d_block_size: byte range of the block payload within
// d_frame. Wrapper D2D-copies this range into d_comp_persist; chunk_descs
// src_offset is block-payload-relative (matches the CPU walk in
// gpuBatchDecode). Single-block frames only (status 13 otherwise) -
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
    SLZ_GUARD_SINGLE_THREAD();
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
    // d_frame[7] = level - unused here.
    // d_frame[8] = block_size_log2 (encoded) - unused (we walk by block hdr).
    const float sc_group_size = walkReadF32LE(d_frame + 9);

    uint32_t pos = SLZ_FRAME_MIN_HDR_SIZE;
    const bool content_size_present = (flags & SLZ_FRAME_FLAG_CONTENT_SIZE_PRESENT) != 0;
    const bool dict_id_present      = (flags & SLZ_FRAME_FLAG_DICT_ID_PRESENT) != 0;
    if (content_size_present) pos += 8;
    // v4 #16: dictionary frames walk normally - the 4-byte ID follows
    // the optional content size; the HOST resolves it (it already
    // reads the header back for the codec level) and supplies the
    // dictionary to the LZ kernels as launch params.
    if (dict_id_present) pos += 4;
    // No PDM / checksums on the slzCompress path; the loop below
    // rejects them if encountered (status 5).

    // eff_chunk_size mirrors decompressOneFrame/gpuBatchDecode: it's the
    // CHUNK boundary at which the 2-byte internal block header repeats.
    uint32_t eff_chunk_size = (uint32_t)(sc_group_size * (float)SLZ_CHUNK_SIZE_BYTES);
    if (eff_chunk_size > SLZ_CHUNK_SIZE_BYTES) eff_chunk_size = SLZ_CHUNK_SIZE_BYTES;
    if (eff_chunk_size == 0) eff_chunk_size = DEFAULT_SUB_CHUNK_CAP;
    // sub_chunk_cap is the slot stride used by the scan kernel; mirrors
    // the CPU's `eff_sc_cap = constants.sub_chunk_size` (= 128KB).
    const uint32_t sub_chunk_cap = (uint32_t)ENTROPY_SCRATCH_SLOT_BYTES;

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
            d_chunks[n_chunks] = { 0u, decomp_size, decomp_size, dst_off, CHUNK_FLAG_UNCOMPRESSED, 0, {0, 0, 0} };
            n_chunks++;
            dst_off += decomp_size;
            pos += comp_size;
            continue;
        }

        if (pos + 2 > frame_size) { *d_status = 12; return; }
        const uint8_t bh0 = d_frame[pos];
        const uint8_t bh1 = d_frame[pos + 1];
        if ((bh0 & SLZ_BLOCK_HDR_MAGIC_MASK) != SLZ_INT_BLOCK_MAGIC) { *d_status = 7; return; }
        const uint8_t decoder = bh1 & SLZ_BLOCK_HDR_DECODER_MASK;
        if (decoder != SLZ_DECODER_FAST && decoder != SLZ_DECODER_TURBO) { *d_status = 8; return; }
        const bool use_checksums = (bh1 & SLZ_BLOCK_HDR_CHECKSUM_FLAG) != 0;
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
                if ((b0 & SLZ_BLOCK_HDR_MAGIC_MASK) != SLZ_INT_BLOCK_MAGIC) { *d_status = 7; return; }
                const uint8_t dec2 = b1 & SLZ_BLOCK_HDR_DECODER_MASK;
                if (dec2 != SLZ_DECODER_FAST && dec2 != SLZ_DECODER_TURBO) { *d_status = 8; return; }
                if ((b1 & SLZ_BLOCK_HDR_CHECKSUM_FLAG) != 0) { *d_status = 5; return; }
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
                const uint32_t flg = (comp == dst_this) ? CHUNK_FLAG_UNCOMPRESSED : 0u;
                // src_offset is BLOCK-PAYLOAD-RELATIVE (matches CPU walk).
                d_chunks[n_chunks] = { pos - block_payload_start, comp, dst_this, dst_off, flg, 0, {0, 0, 0} };
                n_chunks++;
                pos += comp;
            } else if (chunk_type == 1) {
                if (pos + 5 > block_end) { *d_status = 9; return; }
                const uint8_t fill = d_frame[pos + 4];
                pos += 5;
                if (n_chunks >= max_chunks) { *d_status = 11; return; }
                d_chunks[n_chunks] = { 0, 0, dst_this, dst_off, CHUNK_FLAG_MEMSET, fill, {0, 0, 0} };
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

// ── v4 #20: table-mode frame walk ───────────────────────────────────
// Parallel replacement for slzWalkFrameKernel when the frame carries
// the chunk-size table footer (flag bit 6): one coalesced read of the
// table, a block-wide prefix scan of the 3-byte entries, then one
// thread per chunk parses its chunk's headers independently and
// writes its descriptor. Replaces ~n_chunks serialized DRAM
// round-trips (~0.7 ms at 100 MB) with a few microseconds.
//
// Launch shape: ONE block of SLZ_TWALK_THREADS threads (the scan is
// block-local; n_chunks up to WALK_MAX_CHUNKS is handled by giving
// each thread a contiguous segment of ceil(n / threads) entries).
//
// Host-supplied geometry (all derivable from the ≤64-byte header
// readback the D2D path already performs): the table's absolute
// offset and entry count.
//
// Validation (same status vocabulary as the serial walk, plus):
//   14 = table inconsistent (entry vs parsed chunk header, or the
//        entries + SC tail do not sum to the block's compressed size)
// A hostile table cannot redirect reads outside the frame: every
// per-chunk position is bounds-checked against the block range
// before its headers are read.
#define SLZ_TWALK_THREADS 1024

extern "C" __global__ void slzWalkFrameTableKernel(
    const uint8_t* __restrict__ d_frame,
    uint32_t                    frame_size,
    uint32_t                    table_off,
    uint32_t                    n_chunks,
    SlzChunkDesc* __restrict__  d_chunks,
    uint32_t                    max_chunks,
    uint32_t* __restrict__      d_n_chunks,
    uint32_t* __restrict__      d_decompressed_size,
    uint32_t* __restrict__      d_sub_chunk_cap,
    uint32_t* __restrict__      d_block_start,
    uint32_t* __restrict__      d_block_size,
    uint32_t* __restrict__      d_status)
{
    const uint32_t tid = threadIdx.x;

    __shared__ uint32_t s_partial[SLZ_TWALK_THREADS];
    __shared__ uint32_t s_meta[6]; // block_start, block_size, decomp, eff_chunk, sc_tail, status

    // ── Thread 0: header + block-header parse (mirrors the serial walk) ──
    if (tid == 0) {
        *d_n_chunks = 0;
        *d_decompressed_size = 0;
        *d_sub_chunk_cap = (uint32_t)ENTROPY_SCRATCH_SLOT_BYTES;
        *d_block_start = 0;
        *d_block_size = 0;
        *d_status = 0;
        s_meta[5] = 0;

        do {
            if (frame_size < SLZ_FRAME_MIN_HDR_SIZE) { s_meta[5] = 12; break; }
            if (readU32LE(d_frame) != SLZ_FRAME_MAGIC) { s_meta[5] = 1; break; }
            if (d_frame[4] != SLZ_FRAME_VERSION) { s_meta[5] = 2; break; }
            const uint8_t flags = d_frame[5];
            if (d_frame[6] != SLZ_CODEC_FAST_LZ) { s_meta[5] = 3; break; }
            const float sc_group_size = walkReadF32LE(d_frame + 9);

            uint32_t pos = SLZ_FRAME_MIN_HDR_SIZE;
            if ((flags & SLZ_FRAME_FLAG_CONTENT_SIZE_PRESENT) != 0) pos += 8;
            if ((flags & SLZ_FRAME_FLAG_DICT_ID_PRESENT) != 0) pos += 4;

            uint32_t eff_chunk = (uint32_t)(sc_group_size * (float)SLZ_CHUNK_SIZE_BYTES);
            if (eff_chunk > SLZ_CHUNK_SIZE_BYTES) eff_chunk = SLZ_CHUNK_SIZE_BYTES;
            if (eff_chunk == 0) eff_chunk = DEFAULT_SUB_CHUNK_CAP;

            if (pos + 8 > frame_size) { s_meta[5] = 6; break; }
            const uint32_t w0 = readU32LE(d_frame + pos);
            const uint32_t decomp_size = readU32LE(d_frame + pos + 4);
            if ((w0 & SLZ_BLOCK_PDM_FLAG) != 0) { s_meta[5] = 5; break; }
            if ((w0 & SLZ_BLOCK_UNCOMP_FLAG) != 0) { s_meta[5] = 13; break; } // table never on uncomp bodies
            const uint32_t comp_size = w0 & ~(SLZ_BLOCK_UNCOMP_FLAG | SLZ_BLOCK_PDM_FLAG);
            pos += 8;

            if (n_chunks == 0 || n_chunks > max_chunks) { s_meta[5] = 11; break; }
            if (table_off + 3u * n_chunks > frame_size) { s_meta[5] = 12; break; }

            // First internal header: magic / decoder / checksum gates,
            // plus the self-contained bit that announces the SC tail.
            if (pos + 2 > frame_size) { s_meta[5] = 12; break; }
            const uint8_t bh0 = d_frame[pos];
            const uint8_t bh1 = d_frame[pos + 1];
            if ((bh0 & SLZ_BLOCK_HDR_MAGIC_MASK) != SLZ_INT_BLOCK_MAGIC) { s_meta[5] = 7; break; }
            const uint8_t decoder = bh1 & SLZ_BLOCK_HDR_DECODER_MASK;
            if (decoder != SLZ_DECODER_FAST && decoder != SLZ_DECODER_TURBO) { s_meta[5] = 8; break; }
            if ((bh1 & SLZ_BLOCK_HDR_CHECKSUM_FLAG) != 0) { s_meta[5] = 5; break; }
            const bool self_contained = (bh0 & SLZ_BLOCK_HDR_SC_FLAG) != 0;
            const uint32_t sc_tail = (self_contained && n_chunks > 1)
                ? (n_chunks - 1) * 8u : 0u;

            s_meta[0] = pos;        // block payload start (at the internal hdr)
            s_meta[1] = comp_size;
            s_meta[2] = decomp_size;
            s_meta[3] = eff_chunk;
            s_meta[4] = sc_tail;
        } while (false);
    }
    __syncthreads();
    if (s_meta[5] != 0) {
        if (tid == 0) *d_status = s_meta[5];
        return;
    }
    const uint32_t block_start = s_meta[0];
    const uint32_t block_size  = s_meta[1];
    const uint32_t decomp_size = s_meta[2];
    const uint32_t eff_chunk   = s_meta[3];
    const uint32_t sc_tail     = s_meta[4];

    // ── Pass 1: segment sums of the 3-byte entries ──────────────────
    const uint32_t seg = (n_chunks + SLZ_TWALK_THREADS - 1) / SLZ_TWALK_THREADS;
    const uint32_t seg_begin = tid * seg;
    const uint32_t seg_end = (seg_begin + seg < n_chunks) ? seg_begin + seg : n_chunks;
    uint32_t sum = 0;
    for (uint32_t i = seg_begin; i < seg_end; i++)
        sum += readLE24(d_frame + table_off + 3u * i);
    s_partial[tid] = sum;
    __syncthreads();

    // Hillis-Steele inclusive scan over the per-thread partials.
    for (uint32_t d = 1; d < SLZ_TWALK_THREADS; d <<= 1) {
        uint32_t v = (tid >= d) ? s_partial[tid - d] : 0u;
        __syncthreads();
        s_partial[tid] += v;
        __syncthreads();
    }
    const uint32_t total_wire = s_partial[SLZ_TWALK_THREADS - 1];
    const uint32_t seg_base = (tid > 0) ? s_partial[tid - 1] : 0u; // exclusive

    // The free integrity equation: chunk wire bytes + SC tail must be
    // exactly the block's compressed size.
    if (tid == 0 && total_wire + sc_tail != block_size) {
        *d_status = 14;
        s_meta[5] = 14;
    }
    __syncthreads();
    if (s_meta[5] != 0) return;

    // ── Pass 2: per-chunk header parse + descriptor write ──────────
    uint32_t rel = seg_base;
    for (uint32_t i = seg_begin; i < seg_end; i++) {
        const uint32_t entry = readLE24(d_frame + table_off + 3u * i);
        const uint32_t dst_off = i * eff_chunk;
        const uint32_t dst_this = (decomp_size - dst_off < eff_chunk)
            ? decomp_size - dst_off : eff_chunk;
        const uint32_t p = block_start + rel;

        // Bounds: the chunk's whole wire range must sit inside the block.
        if (rel + entry > block_size - sc_tail || entry < 2u) {
            atomicExch(d_status, 14);
            return;
        }

        const uint8_t b0 = d_frame[p];
        const uint8_t b1 = d_frame[p + 1];
        if ((b0 & SLZ_BLOCK_HDR_MAGIC_MASK) != SLZ_INT_BLOCK_MAGIC) { atomicExch(d_status, 7); return; }
        const uint8_t dec2 = b1 & SLZ_BLOCK_HDR_DECODER_MASK;
        if (dec2 != SLZ_DECODER_FAST && dec2 != SLZ_DECODER_TURBO) { atomicExch(d_status, 8); return; }

        if ((b0 & SLZ_INT_BLOCK_UNCOMP_FLAG) != 0) {
            // Uncompressed chunk: [2B internal][raw payload], no chunk hdr.
            if (entry != 2u + dst_this) { atomicExch(d_status, 14); return; }
            d_chunks[i] = { rel + 2u, dst_this, dst_this, dst_off, CHUNK_FLAG_UNCOMPRESSED, 0, {0, 0, 0} };
        } else {
            if (entry < 6u) { atomicExch(d_status, 14); return; }
            const uint32_t v = readU32LE(d_frame + p + 2);
            const uint32_t size_field = v & SLZ_CHUNK_SIZE_MASK;
            const uint32_t chunk_type = (v >> SLZ_CHUNK_TYPE_SHIFT) & 3u;
            if (chunk_type == 0) {
                const uint32_t comp = size_field + 1;
                if (entry != 6u + comp) { atomicExch(d_status, 14); return; }
                const uint32_t flg = (comp == dst_this) ? CHUNK_FLAG_UNCOMPRESSED : 0u;
                d_chunks[i] = { rel + 6u, comp, dst_this, dst_off, flg, 0, {0, 0, 0} };
            } else if (chunk_type == 1) {
                if (entry != 7u) { atomicExch(d_status, 14); return; }
                d_chunks[i] = { 0, 0, dst_this, dst_off, CHUNK_FLAG_MEMSET, d_frame[p + 6], {0, 0, 0} };
            } else {
                atomicExch(d_status, 10);
                return;
            }
        }
        rel += entry;
    }

    if (tid == 0) {
        *d_n_chunks = n_chunks;
        *d_decompressed_size = decomp_size;
        *d_block_start = block_start;
        *d_block_size = block_size;
    }
}
