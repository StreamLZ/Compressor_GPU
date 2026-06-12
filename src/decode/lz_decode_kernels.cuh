// ── StreamLZ LZ decode kernels (entry points) ──────────────────
// The two extern "C" __global__ LZ decode kernels bound by name from
// the Zig driver: slzLzDecodeKernel (general, entropy-capable) and
// slzLzDecodeRawKernel (raw L1/L2 fast path). Both iterate sub-chunks
// inside an SC group and hand off to the parse-and-dispatch helpers in
// lz_dispatch.cuh. Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"
#include "lz_dispatch.cuh"

// ── Production kernel: slzLzDecodeKernel ───────────────────────
// Full GPU LZ kernel: 1 block per SC group, parses raw compressed
// chunks on-GPU. WARPS_PER_BLOCK warps per block for SM occupancy.
// Entropy streams (GPU emits Huffman; decoder also accepts legacy
// entropy types) are decoded by a separate kernel (Pass 1) into the
// *_scratch buffers before this kernel runs (Pass 2).
//
// Parameters:
//   compressed         compressed input blob
//   chunks             per-chunk descriptor array (total_chunks entries)
//   dst                decompressed output blob
//   chunks_per_group   number of chunks decoded by one warp
//   total_chunks       number of valid entries in `chunks`
//   sub_chunk_cap      max decompressed size of a sub-chunk
//   entropy_scratch       entropy-scratch base (lit/tok/off16 sub-slots)
//   entropy_slot_stride   byte stride between the lit, tok, and off16
//                      sub-slots of entropy_scratch; 0 when no scratch
//   first_subchunk_idx global sub-chunk index of each chunk's sub-chunk 0;
//                      nullptr → fall back to chunk_idx (legacy)
extern "C" __global__ void
__launch_bounds__(LZ_KERNEL_BLOCK_THREADS, LZ_KERNEL_MIN_BLOCKS_PER_SM)
slzLzDecodeKernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    const uint32_t* __restrict__ d_total_chunks,
    uint32_t sub_chunk_cap,
    uint8_t* __restrict__ entropy_scratch,
    uint64_t entropy_slot_stride,
    const uint32_t* __restrict__ first_subchunk_idx,
    // v4 #16: preset dictionary (nullptr/0 on dictionary-less frames).
    const uint8_t* __restrict__ dict,
    uint32_t dict_len
) {
    // WARPS_PER_BLOCK warps per block: warp 0 = threadIdx.y==0, etc.
    const uint32_t total_chunks = *d_total_chunks;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t group_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int lane = threadIdx.x & LANE_MASK;
    if (group_id >= (total_chunks + chunks_per_group - 1) / chunks_per_group) return;
    const uint32_t base_chunk = group_id * chunks_per_group;

    for (uint32_t c = 0; c < chunks_per_group; c++) {
        uint32_t chunk_idx = base_chunk + c;
        if (chunk_idx >= total_chunks) return;

        const SlzChunkDesc& ch = chunks[chunk_idx];
        if (ch.decomp_size == 0) continue;

        // Uncompressed chunk: warp-cooperative copy
        if (ch.flags & CHUNK_FLAG_UNCOMPRESSED) {
            const uint8_t* src = compressed + ch.src_offset;
            for (uint32_t i = lane; i < ch.decomp_size; i += WARP_SIZE)
                dst[ch.dst_offset + i] = src[i];
            __syncwarp();
            continue;
        }

        // Memset chunk
        if (ch.flags & CHUNK_FLAG_MEMSET) {
            for (uint32_t i = lane; i < ch.decomp_size; i += WARP_SIZE)
                dst[ch.dst_offset + i] = ch.memset_fill;
            __syncwarp();
            continue;
        }

        // LZ-compressed chunk: iterate sub-chunks. Each sub-chunk gets its
        // own slot in the entropy scratch buffers (indexed by global
        // sub-chunk index). When first_subchunk_idx is nullptr (legacy /
        // no entropy), fall back to chunk_idx for backward compatibility.
        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;
        // Per-sub-chunk slot size = ENTROPY_SCRATCH_SLOT_BYTES, large
        // enough for the biggest sub-chunk's lit/tok streams. off16-hi at
        // offset 0, off16-lo at +65536 within each slot. tok and off16
        // sub-buffers live at +entropy_slot_stride and +2*entropy_slot_stride
        // from the lit slot.
        uint32_t global_subchunk_index =
            first_subchunk_idx ? first_subchunk_idx[chunk_idx] : chunk_idx;
        uint8_t* subchunk_scratch_base = entropy_scratch
            ? (entropy_scratch + (uint64_t)global_subchunk_index * ENTROPY_SCRATCH_SLOT_BYTES)
            : nullptr;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

            // Parse 3-byte sub-chunk header (big-endian)
            uint32_t sub_chunk_header = 0;
            if (lane == 0)
                sub_chunk_header = readBE24(chunk_src);
            sub_chunk_header = __shfl_sync(FULL_WARP_MASK, sub_chunk_header, 0);

            if (!subchunkIsLz(sub_chunk_header)) {
                // Non-LZ sub-chunk (entropy-only) - skip for now
                break;
            }

            uint32_t sc_comp_size = subchunkCompSize(sub_chunk_header);
            uint32_t sc_mode = subchunkMode(sub_chunk_header);
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                // Derive tok/off16 sub-slots on demand from base + stride.
                uint8_t* tok_slot = subchunk_scratch_base
                    ? (subchunk_scratch_base + entropy_slot_stride) : nullptr;
                uint8_t* off16_slot = subchunk_scratch_base
                    ? (subchunk_scratch_base + 2 * entropy_slot_stride) : nullptr;
                // Pass the absolute sc_dst_off as base_offset, so only the
                // sub-chunk at output offset 0 (the very first sub-chunk in
                // the frame) triggers the 8-byte initial copy. Per-chunk
                // init copies are restored by the host-side SC prefix table
                // post-pass instead.
                // v4 #16: dict presence is frame-uniform - one
                // predictable branch per sub-chunk selects the
                // dict-aware instantiation; the false path is the
                // pre-dict code exactly.
                if (dict != nullptr) {
                    parseAndDecodeSubChunk<true>(
                        sc_payload, sc_comp_size, sc_size,
                        dst, sc_dst_off, sc_dst_off, sc_mode,
                        subchunk_scratch_base, tok_slot, off16_slot,
                        dict, dict_len
                    );
                } else {
                    parseAndDecodeSubChunk<false>(
                        sc_payload, sc_comp_size, sc_size,
                        dst, sc_dst_off, sc_dst_off, sc_mode,
                        subchunk_scratch_base, tok_slot, off16_slot
                    );
                }
            } else {
                // Uncompressed sub-chunk: copy
                for (uint32_t i = lane; i < sc_size; i += WARP_SIZE)
                    dst[sc_dst_off + i] = sc_payload[i];
            }
            __syncwarp();

            chunk_src += 3 + sc_comp_size;
            sc_dst_off += sc_size;
            sc_remaining -= sc_size;
            // Advance to next sub-chunk's scratch slot. tok/off16 slots are
            // derived from this each iteration so no separate tracking needed.
            if (subchunk_scratch_base) subchunk_scratch_base += ENTROPY_SCRATCH_SLOT_BYTES;
        }
    }
}

// ── L1/L2 raw fast-path kernel ────────────────────────────────
// Dedicated extern "C" __global__ for raw L1/L2 input (no entropy
// coding, all chunk_type 0). Lean parameter list, single parsing path,
// all parsed values flow in registers (no ParsedStreams struct, no
// chunk_type dispatch). The driver picks this kernel when
// no entropy is present; the general slzLzDecodeKernel
// handles L3+ entropy.
//
// Parameters: compressed input blob, per-chunk descriptors, output
// blob, chunks-per-warp count, valid descriptor count, and the max
// decompressed sub-chunk size - see slzLzDecodeKernel for the
// shared parameter semantics.
extern "C" __global__ void
__launch_bounds__(LZ_KERNEL_BLOCK_THREADS, LZ_KERNEL_MIN_BLOCKS_PER_SM)
slzLzDecodeRawKernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    const uint32_t* __restrict__ d_total_chunks,
    uint32_t sub_chunk_cap,
    // v4 #16: preset dictionary (nullptr/0 on dictionary-less frames).
    const uint8_t* __restrict__ dict,
    uint32_t dict_len
) {
    const uint32_t total_chunks = *d_total_chunks;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t group_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int lane = threadIdx.x & LANE_MASK;
    if (group_id >= (total_chunks + chunks_per_group - 1) / chunks_per_group) return;
    const uint32_t base_chunk = group_id * chunks_per_group;

    for (uint32_t c = 0; c < chunks_per_group; c++) {
        uint32_t chunk_idx = base_chunk + c;
        if (chunk_idx >= total_chunks) return;

        const SlzChunkDesc& ch = chunks[chunk_idx];
        if (ch.decomp_size == 0) continue;

        if (ch.flags & CHUNK_FLAG_UNCOMPRESSED) {
            const uint8_t* src = compressed + ch.src_offset;
            for (uint32_t i = lane; i < ch.decomp_size; i += WARP_SIZE)
                dst[ch.dst_offset + i] = src[i];
            __syncwarp();
            continue;
        }

        if (ch.flags & CHUNK_FLAG_MEMSET) {
            for (uint32_t i = lane; i < ch.decomp_size; i += WARP_SIZE)
                dst[ch.dst_offset + i] = ch.memset_fill;
            __syncwarp();
            continue;
        }

        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

            uint32_t sub_chunk_header = 0;
            if (lane == 0)
                sub_chunk_header = readBE24(chunk_src);
            sub_chunk_header = __shfl_sync(FULL_WARP_MASK, sub_chunk_header, 0);

            if (!subchunkIsLz(sub_chunk_header)) break;

            uint32_t sc_comp_size = subchunkCompSize(sub_chunk_header);
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                if (dict != nullptr) {
                    parseAndDecodeSubChunkRaw<true>(
                        sc_payload, sc_comp_size, sc_size,
                        dst, sc_dst_off, sc_dst_off, dict, dict_len
                    );
                } else {
                    parseAndDecodeSubChunkRaw<false>(
                        sc_payload, sc_comp_size, sc_size,
                        dst, sc_dst_off, sc_dst_off
                    );
                }
            } else {
                for (uint32_t i = lane; i < sc_size; i += WARP_SIZE)
                    dst[sc_dst_off + i] = sc_payload[i];
            }
            __syncwarp();

            chunk_src += 3 + sc_comp_size;
            sc_dst_off += sc_size;
            sc_remaining -= sc_size;
        }
    }
}

// ── v4 #15: Pipelined raw kernel ──────────────────────────────────
// Same interface as slzLzDecodeRawKernel but both warps in a block
// cooperate on the SAME chunk group via the 2-warp pipeline
// (lz_decode_raw_pipeline.cuh). Grid must be doubled by the host.
#include "lz_decode_raw_pipeline.cuh"

extern "C" __global__ void
__launch_bounds__(SLZ_PIPE_BLOCK_THREADS, SLZ_PIPE_MIN_BLOCKS_PER_SM)
slzLzDecodeRawPipelinedKernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    const uint32_t* __restrict__ d_total_chunks,
    uint32_t sub_chunk_cap,
    // v4 #16: preset dictionary (nullptr/0 on dictionary-less frames).
    const uint8_t* __restrict__ dict,
    uint32_t dict_len
) {
    const uint32_t total_chunks = *d_total_chunks;
    const uint32_t group_id = blockIdx.x;
    const int lane = threadIdx.x & LANE_MASK;
    if (group_id >= (total_chunks + chunks_per_group - 1) / chunks_per_group) return;
    const uint32_t base_chunk = group_id * chunks_per_group;

    __shared__ PipeBatch s_pipe_batch[SLZ_PIPE_STAGES];

    for (uint32_t c = 0; c < chunks_per_group; c++) {
        uint32_t chunk_idx = base_chunk + c;
        if (chunk_idx >= total_chunks) return;

        const SlzChunkDesc& ch = chunks[chunk_idx];
        if (ch.decomp_size == 0) continue;

        if (ch.flags & CHUNK_FLAG_UNCOMPRESSED) {
            const uint8_t* src = compressed + ch.src_offset;
            uint32_t gl = threadIdx.y * WARP_SIZE + lane;
            for (uint32_t i = gl; i < ch.decomp_size; i += SLZ_PIPE_BLOCK_THREADS)
                dst[ch.dst_offset + i] = src[i];
            __syncthreads();
            continue;
        }
        if (ch.flags & CHUNK_FLAG_MEMSET) {
            uint32_t gl = threadIdx.y * WARP_SIZE + lane;
            for (uint32_t i = gl; i < ch.decomp_size; i += SLZ_PIPE_BLOCK_THREADS)
                dst[ch.dst_offset + i] = ch.memset_fill;
            __syncthreads();
            continue;
        }

        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

            __shared__ uint32_t s_sch;
            if (lane == 0 && threadIdx.y == 0) s_sch = readBE24(chunk_src);
            __syncthreads();
            uint32_t sub_chunk_header = s_sch;

            if (!subchunkIsLz(sub_chunk_header)) break;

            uint32_t sc_comp_size = subchunkCompSize(sub_chunk_header);
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                if (dict != nullptr) {
                    parseAndDecodeSubChunkRawPipelined<true>(
                        sc_payload, sc_comp_size, sc_size,
                        dst, sc_dst_off, sc_dst_off,
                        s_pipe_batch, dict, dict_len
                    );
                } else {
                    parseAndDecodeSubChunkRawPipelined<false>(
                        sc_payload, sc_comp_size, sc_size,
                        dst, sc_dst_off, sc_dst_off,
                        s_pipe_batch
                    );
                }
            } else {
                uint32_t gl = threadIdx.y * WARP_SIZE + lane;
                for (uint32_t i = gl; i < sc_size; i += SLZ_PIPE_BLOCK_THREADS)
                    dst[sc_dst_off + i] = sc_payload[i];
            }
            __syncthreads();

            chunk_src += 3 + sc_comp_size;
            sc_dst_off += sc_size;
            sc_remaining -= sc_size;
        }
    }
}

// ── v4 #15 L3+ port: Pipelined general (entropy-capable) kernel ──
// Same interface as slzLzDecodeKernel but one chunk group per block
// with the K=4 pipeline (parser warp + 3-warp copier team). Grid is
// doubled by the host. Per sub-chunk: the parser warp runs the
// existing warp-cooperative parseSubChunkHeaders, lane 0 publishes
// the ParsedStreams to shared, and the block routes:
//   - mode 1 + no off32  -> decodeSubChunkRawModePipelined, templated
//     on ps.off16_split (entropy frames read off16 from the gathered
//     hi/lo scratch planes; raw frames read the interleaved stream)
//   - anything else      -> the proven warp-level
//     decodeSubChunkGeneral on warp 0, team idles to the barrier
//     (off32 sub-chunks > 64 KB and exotic modes are rare at the
//     sc=0.25 default).
extern "C" __global__ void
__launch_bounds__(SLZ_PIPE_BLOCK_THREADS, SLZ_PIPE_MIN_BLOCKS_PER_SM)
slzLzDecodeGeneralPipelinedKernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    const uint32_t* __restrict__ d_total_chunks,
    uint32_t sub_chunk_cap,
    uint8_t* __restrict__ entropy_scratch,
    uint64_t entropy_slot_stride,
    const uint32_t* __restrict__ first_subchunk_idx,
    // v4 #16: preset dictionary (nullptr/0 on dictionary-less frames).
    const uint8_t* __restrict__ dict,
    uint32_t dict_len
) {
    const uint32_t total_chunks = *d_total_chunks;
    const uint32_t group_id = blockIdx.x;
    const int lane = threadIdx.x & LANE_MASK;
    if (group_id >= (total_chunks + chunks_per_group - 1) / chunks_per_group) return;
    const uint32_t base_chunk = group_id * chunks_per_group;

    __shared__ PipeBatch s_pipe_batch[SLZ_PIPE_STAGES];
    __shared__ ParsedStreams s_ps;
    __shared__ uint32_t s_sch_hdr;

    for (uint32_t c = 0; c < chunks_per_group; c++) {
        uint32_t chunk_idx = base_chunk + c;
        if (chunk_idx >= total_chunks) return;

        const SlzChunkDesc& ch = chunks[chunk_idx];
        if (ch.decomp_size == 0) continue;

        if (ch.flags & CHUNK_FLAG_UNCOMPRESSED) {
            const uint8_t* src = compressed + ch.src_offset;
            uint32_t gl = threadIdx.y * WARP_SIZE + lane;
            for (uint32_t i = gl; i < ch.decomp_size; i += SLZ_PIPE_BLOCK_THREADS)
                dst[ch.dst_offset + i] = src[i];
            __syncthreads();
            continue;
        }
        if (ch.flags & CHUNK_FLAG_MEMSET) {
            uint32_t gl = threadIdx.y * WARP_SIZE + lane;
            for (uint32_t i = gl; i < ch.decomp_size; i += SLZ_PIPE_BLOCK_THREADS)
                dst[ch.dst_offset + i] = ch.memset_fill;
            __syncthreads();
            continue;
        }

        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;
        uint32_t global_subchunk_index =
            first_subchunk_idx ? first_subchunk_idx[chunk_idx] : chunk_idx;
        uint8_t* subchunk_scratch_base = entropy_scratch
            ? (entropy_scratch + (uint64_t)global_subchunk_index * ENTROPY_SCRATCH_SLOT_BYTES)
            : nullptr;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

            if (lane == 0 && threadIdx.y == 0) s_sch_hdr = readBE24(chunk_src);
            __syncthreads();
            uint32_t sub_chunk_header = s_sch_hdr;

            if (!subchunkIsLz(sub_chunk_header)) break;

            uint32_t sc_comp_size = subchunkCompSize(sub_chunk_header);
            uint32_t sc_mode = subchunkMode(sub_chunk_header);
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                uint8_t* tok_slot = subchunk_scratch_base
                    ? (subchunk_scratch_base + entropy_slot_stride) : nullptr;
                uint8_t* off16_slot = subchunk_scratch_base
                    ? (subchunk_scratch_base + 2 * entropy_slot_stride) : nullptr;

                // Header parse on the parser warp (warp-cooperative,
                // includes the base_offset==0 initial 8-byte copy);
                // lane 0 publishes the result for the team.
                if (threadIdx.y == 0) {
                    ParsedStreams ps;
                    parseSubChunkHeaders(sc_payload, sc_comp_size, sc_size, dst,
                                         sc_dst_off, sc_dst_off,
                                         subchunk_scratch_base, tok_slot,
                                         off16_slot, ps);
                    if (lane == 0) s_ps = ps;
                }
                __syncthreads();

                if (sc_mode == 1 && s_ps.off32_count1 == 0 && s_ps.off32_count2 == 0) {
                    // v4 #16: the dict branch is frame-uniform; each arm
                    // repeats the off16_split dispatch so every reached
                    // combination is its own dead-code-eliminated
                    // instantiation.
                    if (dict != nullptr) {
                        if (s_ps.off16_split) {
                            decodeSubChunkRawModePipelined<true, true>(
                                s_ps.cmd_ptr, s_ps.cmd_size,
                                s_ps.lit_ptr, s_ps.lit_size,
                                s_ps.off16_raw, s_ps.off16_count,
                                s_ps.off16_hi, s_ps.off16_lo,
                                s_ps.len_stream, s_ps.len_avail,
                                dst, sc_size, s_ps.initial_copy, sc_dst_off,
                                s_pipe_batch, dict, dict_len);
                        } else {
                            decodeSubChunkRawModePipelined<false, true>(
                                s_ps.cmd_ptr, s_ps.cmd_size,
                                s_ps.lit_ptr, s_ps.lit_size,
                                s_ps.off16_raw, s_ps.off16_count,
                                s_ps.off16_hi, s_ps.off16_lo,
                                s_ps.len_stream, s_ps.len_avail,
                                dst, sc_size, s_ps.initial_copy, sc_dst_off,
                                s_pipe_batch, dict, dict_len);
                        }
                    } else if (s_ps.off16_split) {
                        decodeSubChunkRawModePipelined<true>(
                            s_ps.cmd_ptr, s_ps.cmd_size,
                            s_ps.lit_ptr, s_ps.lit_size,
                            s_ps.off16_raw, s_ps.off16_count,
                            s_ps.off16_hi, s_ps.off16_lo,
                            s_ps.len_stream, s_ps.len_avail,
                            dst, sc_size, s_ps.initial_copy, sc_dst_off,
                            s_pipe_batch);
                    } else {
                        decodeSubChunkRawModePipelined<false>(
                            s_ps.cmd_ptr, s_ps.cmd_size,
                            s_ps.lit_ptr, s_ps.lit_size,
                            s_ps.off16_raw, s_ps.off16_count,
                            s_ps.off16_hi, s_ps.off16_lo,
                            s_ps.len_stream, s_ps.len_avail,
                            dst, sc_size, s_ps.initial_copy, sc_dst_off,
                            s_pipe_batch);
                    }
                } else {
                    // off32 present or exotic mode: proven warp-level
                    // general decoder on the parser warp (ps already
                    // parsed - no re-parse).
                    if (threadIdx.y == 0) {
                        const DecodeOutput out_general = { dst, sc_size, sc_dst_off };
                        if (dict != nullptr) {
                            if (s_ps.off16_split) {
                                decodeSubChunkGeneral<true, true>(s_ps, out_general, sc_mode, dict, dict_len);
                            } else {
                                decodeSubChunkGeneral<false, true>(s_ps, out_general, sc_mode, dict, dict_len);
                            }
                        } else if (s_ps.off16_split) {
                            decodeSubChunkGeneral<true>(s_ps, out_general, sc_mode);
                        } else {
                            decodeSubChunkGeneral<false>(s_ps, out_general, sc_mode);
                        }
                    }
                }
            } else {
                uint32_t gl = threadIdx.y * WARP_SIZE + lane;
                for (uint32_t i = gl; i < sc_size; i += SLZ_PIPE_BLOCK_THREADS)
                    dst[sc_dst_off + i] = sc_payload[i];
            }
            __syncthreads();

            chunk_src += 3 + sc_comp_size;
            sc_dst_off += sc_size;
            sc_remaining -= sc_size;
            if (subchunk_scratch_base) subchunk_scratch_base += ENTROPY_SCRATCH_SLOT_BYTES;
        }
    }
}

// ── v4 #19: per-chunk content hash (Merkle-root checksum) ────────
// Decode-side verification: hash each chunk of the decoded output in
// place on device, roll up on the host, compare against the frame's
// stored root. One thread per chunk; ~free next to the LZ kernel.
#include "../common/xxh32_device.cuh"

extern "C" __global__ void slzSegHashKernel(
    const uint8_t* __restrict__ data,
    uint32_t n_chunks,
    uint32_t eff_chunk,
    uint64_t total_size,
    uint32_t* __restrict__ seg_hashes,
    const uint8_t* __restrict__ prefix_table
) SLZ_SEG_HASH_KERNEL_BODY

extern "C" __global__ void slzChunkCombineKernel(
    const uint32_t* __restrict__ seg_hashes,
    uint32_t n_chunks,
    uint32_t eff_chunk,
    uint64_t total_size,
    uint32_t* __restrict__ out_hashes
) SLZ_CHUNK_COMBINE_KERNEL_BODY

// ── v4 #19 device-only verify: SC-prefix apply + verdict ─────────
// slzScPrefixApplyKernel writes the true first-8 bytes of chunks 1+
// into the decoded output ON DEVICE (source = the SC tail table that
// is already resident in the uploaded compressed block). This fixes
// the long-standing quirk where the LZ kernel leaves garbage there
// and the HOST patches it after D2H - the output is now final in
// VRAM, which is what makes device-side verification (and future
// D2D coverage) possible. One thread per chunk.
extern "C" __global__ void slzScPrefixApplyKernel(
    uint8_t* __restrict__ dst,
    const uint8_t* __restrict__ comp,
    uint64_t prefix_off,     // offset of the SC tail table in `comp`
    uint32_t n_chunks,
    uint32_t eff_chunk,
    uint64_t total_size
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x + 1; // chunks 1..n-1
    if (i >= n_chunks) return;
    const uint64_t start = (uint64_t)i * eff_chunk;
    uint32_t len = 8;
    if (start + len > total_size) len = (uint32_t)(total_size - start);
    const uint8_t* src = comp + prefix_off + (uint64_t)(i - 1) * 8;
    for (uint32_t k = 0; k < len; k++) dst[start + k] = src[k];
}

// slzMerkleVerdictKernel: roll the per-chunk hash array into the
// Merkle root (one thread - the array is ~6 KB) and compare against
// the frame's expected root. verdict: 1 = match, 2 = mismatch.
// (0 = kernel never ran; the host falls back to its parallel hash.)
extern "C" __global__ void slzMerkleVerdictKernel(
    const uint32_t* __restrict__ hashes,
    uint32_t n_chunks,
    uint32_t expected_root,
    uint32_t* __restrict__ verdict_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t root = xxh32Device((const uint8_t*)hashes, n_chunks * 4u);
    verdict_out[0] = (root == expected_root) ? 1u : 2u;
}
