// ── StreamLZ LZ decode kernels + dispatch ──────────────────────
// The extern "C" __global__ ABI surface (bound by name from the Zig
// driver in src/gpu/decode/driver.zig) plus the two device-side
// parse-and-dispatch helpers. Included into the single lz_kernel.cu
// translation unit.
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
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset
) {
    const int lane = threadIdx.x & LANE_MASK;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    uint32_t initial_copy = 0;
    if (base_offset == 0) {
        // First 8 bytes are raw literals — copy directly to output.
        if (lane < (int)INITIAL_LITERAL_COPY_BYTES) dst[dst_offset + lane] = src[lane];
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
        uint16_t v; memcpy(&v, src, 2);
        block2_cmd_offset = v;
        src += 2;
    }
    block2_cmd_offset = __shfl_sync(FULL_WARP_MASK, block2_cmd_offset, 0);
    src = broadcastSrc(sc_src, src);

    // Off16 stream (raw — no entropy 0xFFFF marker handling on this path).
    const uint8_t* off16_raw;
    uint32_t off16_count = 0;
    if (lane == 0) {
        uint16_t cnt; memcpy(&cnt, src, 2);
        off16_count = cnt;
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
        uint32_t tmp = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
        src += 3;
        if (tmp != 0) {
            off32_count1 = tmp >> OFF32_COUNT1_SHIFT;
            off32_count2 = tmp & OFF32_COUNT2_MASK;
            if (off32_count1 == OFF32_COUNT_ESCAPE) { uint16_t v; memcpy(&v, src, 2); off32_count1 = v; src += 2; }
            if (off32_count2 == OFF32_COUNT_ESCAPE) { uint16_t v; memcpy(&v, src, 2); off32_count2 = v; src += 2; }
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
        decodeSubChunkGeneral(
            cmd_ptr, cmd_size,
            lit_ptr, lit_size,
            off16_raw, off16_count,
            nullptr, nullptr, 0,  // off16_hi/lo/split unused for raw
            off32_raw1, off32_count1,
            off32_raw2, off32_count2,
            len_stream, len_avail,
            dst, sc_decomp_size, initial_copy, block2_cmd_offset,
            dst_offset, /*mode=*/1
        );
    }
}

// ── Sub-chunk decoder dispatch ────────────────────────────────
// General entropy-capable path (used by slzLzDecodeKernel only;
// the slzLzDecodeRawKernel fast path uses parseAndDecodeSubChunkRaw).
// Calls parseSubChunkHeaders (__noinline__) then dispatches to
// decodeSubChunkRawMode or decodeSubChunkGeneral. The header parser's
// registers are freed before the decode hot loop runs.
__device__ void parseAndDecodeSubChunk(
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint32_t mode,
    uint8_t* entropy_lit_scratch,
    uint8_t* entropy_tok_scratch,
    uint8_t* entropy_off16_scratch
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
        decodeSubChunkGeneral(
            ps.cmd_ptr, ps.cmd_size,
            ps.lit_ptr, ps.lit_size,
            ps.off16_raw, ps.off16_count,
            ps.off16_hi, ps.off16_lo, ps.off16_split,
            ps.off32_raw1, ps.off32_count1,
            ps.off32_raw2, ps.off32_count2,
            ps.len_stream, ps.len_avail,
            dst, sc_decomp_size, ps.initial_copy, ps.cmd_stream2_offset,
            dst_offset, mode
        );
    }
}

// ── Production kernel: slzLzDecodeKernel ───────────────────────
// Full GPU LZ kernel: 1 block per SC group, parses raw compressed
// chunks on-GPU. WARPS_PER_BLOCK warps per block for SM occupancy.
// Entropy streams (tANS / Huffman) are decoded by a separate kernel
// (Pass 1) into the *_scratch buffers before this kernel runs (Pass 2).
//
// Parameters:
//   compressed         compressed input blob
//   chunks             per-chunk descriptor array (total_chunks entries)
//   dst                decompressed output blob
//   chunks_per_group   number of chunks decoded by one warp
//   total_chunks       number of valid entries in `chunks`
//   sub_chunk_cap      max decompressed size of a sub-chunk
//   tans_scratch       entropy-scratch base (lit/tok/off16 sub-slots)
//   tans_slot_stride   byte stride between the lit, tok, and off16
//                      sub-slots of tans_scratch; 0 when no scratch
//   first_subchunk_idx global sub-chunk index of each chunk's sub-chunk 0;
//                      nullptr → fall back to chunk_idx (legacy)
extern "C" __global__ void
__launch_bounds__(LZ_KERNEL_BLOCK_THREADS, LZ_KERNEL_MIN_BLOCKS_PER_SM)
slzLzDecodeKernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    uint32_t total_chunks,
    uint32_t sub_chunk_cap,
    uint8_t* __restrict__ tans_scratch,
    uint64_t tans_slot_stride,
    const uint32_t* __restrict__ first_subchunk_idx
) {
    // WARPS_PER_BLOCK warps per block: warp 0 = threadIdx.y==0, etc.
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
        if (ch.flags & 1) {
            const uint8_t* src = compressed + ch.src_offset;
            for (uint32_t i = lane; i < ch.decomp_size; i += WARP_SIZE)
                dst[ch.dst_offset + i] = src[i];
            __syncwarp();
            continue;
        }

        // Memset chunk
        if (ch.flags & 2) {
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
        // sub-buffers live at +tans_slot_stride and +2*tans_slot_stride
        // from the lit slot.
        uint32_t global_subchunk_index =
            first_subchunk_idx ? first_subchunk_idx[chunk_idx] : chunk_idx;
        uint8_t* subchunk_scratch_base = tans_scratch
            ? (tans_scratch + (uint64_t)global_subchunk_index * ENTROPY_SCRATCH_SLOT_BYTES)
            : nullptr;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

            // Parse 3-byte sub-chunk header (big-endian)
            uint32_t sub_chunk_header = 0;
            if (lane == 0)
                sub_chunk_header = readBE24(chunk_src);
            sub_chunk_header = __shfl_sync(FULL_WARP_MASK, sub_chunk_header, 0);

            if (!(sub_chunk_header & SUBCHUNK_LZ_FLAG_BIT)) {
                // Non-LZ sub-chunk (entropy-only) — skip for now
                break;
            }

            uint32_t sc_comp_size = sub_chunk_header & SUBCHUNK_COMP_SIZE_MASK;
            uint32_t sc_mode = (sub_chunk_header >> SUBCHUNK_MODE_SHIFT) & SUBCHUNK_MODE_MASK;
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                // Derive tok/off16 sub-slots on demand from base + stride.
                uint8_t* tok_slot = subchunk_scratch_base
                    ? (subchunk_scratch_base + tans_slot_stride) : nullptr;
                uint8_t* off16_slot = subchunk_scratch_base
                    ? (subchunk_scratch_base + 2 * tans_slot_stride) : nullptr;
                // Pass the absolute sc_dst_off as base_offset, so only the
                // sub-chunk at output offset 0 (the very first sub-chunk in
                // the frame) triggers the 8-byte initial copy. Per-chunk
                // init copies are restored by the host-side SC prefix table
                // post-pass instead.
                parseAndDecodeSubChunk(
                    sc_payload, sc_comp_size, sc_size,
                    dst, sc_dst_off, sc_dst_off, sc_mode,
                    subchunk_scratch_base, tok_slot, off16_slot
                );
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
// scan.use_tans32 == false; the general slzLzDecodeKernel
// handles L3+ entropy.
//
// Parameters: compressed input blob, per-chunk descriptors, output
// blob, chunks-per-warp count, valid descriptor count, and the max
// decompressed sub-chunk size — see slzLzDecodeKernel for the
// shared parameter semantics.
extern "C" __global__ void
__launch_bounds__(LZ_KERNEL_BLOCK_THREADS, LZ_KERNEL_MIN_BLOCKS_PER_SM)
slzLzDecodeRawKernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    uint32_t total_chunks,
    uint32_t sub_chunk_cap
) {
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

        if (ch.flags & 1) {
            const uint8_t* src = compressed + ch.src_offset;
            for (uint32_t i = lane; i < ch.decomp_size; i += WARP_SIZE)
                dst[ch.dst_offset + i] = src[i];
            __syncwarp();
            continue;
        }

        if (ch.flags & 2) {
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

            if (!(sub_chunk_header & SUBCHUNK_LZ_FLAG_BIT)) break;

            uint32_t sc_comp_size = sub_chunk_header & SUBCHUNK_COMP_SIZE_MASK;
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                parseAndDecodeSubChunkRaw(
                    sc_payload, sc_comp_size, sc_size,
                    dst, sc_dst_off, sc_dst_off
                );
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

// ── Raw off16 gather ────────────────────────────────────────────────
// Scatters the raw (type-0) off16 sub-streams from the compressed blob
// into the off16 scratch in a single launch — replacing ~1500
// host-issued device-to-device copies (the per-call driver overhead was
// ~8 ms). Descriptor layout matches the Zig RawOff16Desc
// {src_offset, size, gpu_offset}.
//
// Launch contract: a 1-D grid of `count` blocks, one block per
// descriptor; blockDim.x lanes stride the copy of one stream. This
// shape is a hard contract with the driver. No __launch_bounds__: the
// kernel is bandwidth-bound, so occupancy tuning is not needed.
extern "C" __global__ void slzGatherRawOff16Kernel(
    const uint8_t* __restrict__ comp_base,
    uint32_t comp_len,
    uint8_t* __restrict__ scratch_base,
    const SlzRawOff16Desc* __restrict__ descs,
    uint32_t count
) {
    const uint32_t i = blockIdx.x;
    if (i >= count) return;
    const SlzRawOff16Desc d = descs[i];
    if (d.size == 0 || d.src_offset + d.size > comp_len) return;
    const uint8_t* s = comp_base + d.src_offset;
    uint8_t* t = scratch_base + d.gpu_offset;
    for (uint32_t j = threadIdx.x; j < d.size; j += blockDim.x)
        t[j] = s[j];
}

// ── Decode-scan kernel (roadmap 4d Phase 2) ─────────────────────────
// GPU port of scanForTansChunks (decode/driver.zig): one thread per
// chunk walks every sub-chunk's literal / token / off16 stream headers
// and stages a descriptor per stream type per global sub-chunk index.
// The driver D2H's the staged arrays and compacts them (drops raw /
// absent slots, assigns lut_offset). Verified byte-identical to the CPU
// scan in tools/huff_test/scan_test.cu across enwik8 + silesia L3-L5.

// chunk_type=4 Huffman stream descriptor — staged form: lut_offset is
// assigned by the host compaction, not the kernel.
struct SlzScanHuffDesc { uint32_t in_offset, in_size, out_offset, out_size, valid; };
struct SlzScanRawDesc  { uint32_t src_offset, size, gpu_offset, valid; };
static_assert(sizeof(SlzScanHuffDesc) == 20, "ABI: keep in sync with decode/driver.zig");
static_assert(sizeof(SlzScanRawDesc) == 16, "ABI: keep in sync with decode/driver.zig");

static constexpr uint32_t SCAN_SUBCHUNK_SLOT = 131072; // sub_dst_off stride
static constexpr uint32_t SCAN_OFF16_LO_SLOT = 65536;  // lo half within slot
static constexpr uint32_t SCAN_SUBCHUNK_64K  = 0x10000;
static constexpr uint32_t SCAN_OFF16_MARKER  = 0xFFFF;

__device__ __forceinline__ uint32_t scanReadU24BE(const uint8_t* p) {
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | (uint32_t)p[2];
}

// Parse a chunk_type=4 Huffman header at `pos` within chunk_src.
__device__ static bool scanParseHuffHeader(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos,
    uint32_t src_offset_base, uint32_t dst_off, SlzScanHuffDesc& out) {
    if (pos >= chunk_len) return false;
    const uint8_t first = chunk_src[pos];
    uint32_t comp_size, dst_size, payload_off;
    if (first >= 0x80) {
        if (pos + 3 > chunk_len) return false;
        const uint32_t bits = scanReadU24BE(chunk_src + pos);
        comp_size = bits & 0x3FF;
        dst_size  = comp_size + ((bits >> 10) & 0x3FF) + 1;
        payload_off = src_offset_base + pos + 3;
    } else {
        if (pos + 5 > chunk_len) return false;
        const uint32_t bits = ((uint32_t)chunk_src[pos + 1] << 24)
                            | ((uint32_t)chunk_src[pos + 2] << 16)
                            | ((uint32_t)chunk_src[pos + 3] << 8)
                            | (uint32_t)chunk_src[pos + 4];
        comp_size = bits & 0x3FFFF;
        dst_size  = (((bits >> 18) | ((uint32_t)chunk_src[pos] << 14)) & 0x3FFFF) + 1;
        payload_off = src_offset_base + pos + 5;
    }
    out.in_offset = payload_off; out.in_size = comp_size;
    out.out_offset = dst_off;    out.out_size = dst_size;
    out.valid = 1;
    return true;
}

// Parse a type-0 (memcpy) stream header → data offset + size.
__device__ static bool scanParseType0(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos,
    uint32_t& data_off, uint32_t& size) {
    if (pos >= chunk_len) return false;
    const uint8_t first = chunk_src[pos];
    if (first >= 0x80) {
        if (pos + 2 > chunk_len) return false;
        size = (((uint32_t)chunk_src[pos] << 8) | (uint32_t)chunk_src[pos + 1]) & 0xFFF;
        data_off = pos + 2;
    } else {
        if (pos + 3 > chunk_len) return false;
        size = scanReadU24BE(chunk_src + pos);
        data_off = pos + 3;
    }
    return data_off != 0;
}

// Skip an entropy/raw stream header + payload. 0xFFFFFFFF = truncated.
__device__ static uint32_t scanSkipStreamHeader(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos) {
    if (pos >= chunk_len) return 0xFFFFFFFFu;
    const uint8_t first = chunk_src[pos];
    const uint32_t ct = (first >> 4) & 0x7;
    if (ct == 0) {
        if (first >= 0x80) {
            if (pos + 2 > chunk_len) return 0xFFFFFFFFu;
            const uint32_t sz = (((uint32_t)chunk_src[pos] << 8) | (uint32_t)chunk_src[pos + 1]) & 0xFFF;
            return pos + 2 + sz;
        }
        if (pos + 3 > chunk_len) return 0xFFFFFFFFu;
        return pos + 3 + scanReadU24BE(chunk_src + pos);
    } else if (ct == 1 || ct == 2 || ct == 4 || ct == 6) {
        if (first >= 0x80) {
            if (pos + 3 > chunk_len) return 0xFFFFFFFFu;
            return pos + 3 + (scanReadU24BE(chunk_src + pos) & 0x3FF);
        }
        if (pos + 5 > chunk_len) return 0xFFFFFFFFu;
        const uint32_t bits = ((uint32_t)chunk_src[pos + 1] << 24)
                            | ((uint32_t)chunk_src[pos + 2] << 16)
                            | ((uint32_t)chunk_src[pos + 3] << 8)
                            | (uint32_t)chunk_src[pos + 4];
        return pos + 5 + (bits & 0x3FFFF);
    } else if (ct == 5) {
        if (pos + 7 > chunk_len) return 0xFFFFFFFFu;
        return pos + 7;
    } else if (ct == 7) {
        if (pos + 4 > chunk_len) return 0xFFFFFFFFu;
        return scanSkipStreamHeader(chunk_src, chunk_len, pos + 4);
    }
    return 0xFFFFFFFFu;
}

// One thread per chunk; walks the chunk's sub-chunks and stages a
// descriptor per stream type per global sub-chunk index. first_sub_idx
// is the driver's prefix sum of sub-chunks-per-chunk.
extern "C" __global__ void slzScanParseKernel(
    const uint8_t* __restrict__ block,
    uint32_t                    block_len,
    const SlzChunkDesc* __restrict__ chunks,
    const uint32_t* __restrict__ first_sub_idx,
    uint32_t                    n_chunks,
    uint32_t                    sub_chunk_cap,
    SlzScanHuffDesc* __restrict__ st_lit,
    SlzScanHuffDesc* __restrict__ st_tok,
    SlzScanHuffDesc* __restrict__ st_hi,
    SlzScanHuffDesc* __restrict__ st_lo,
    SlzScanRawDesc*  __restrict__ st_raw_hi,
    SlzScanRawDesc*  __restrict__ st_raw_lo)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_chunks) return;

    const SlzChunkDesc ch = chunks[i];
    const uint32_t cap_safe = sub_chunk_cap ? sub_chunk_cap : 65536u;
    const uint32_t chunk_first_sub = first_sub_idx[i];

    if (ch.flags != 0 || ch.decomp_size == 0) {
        st_lit[chunk_first_sub].valid = 0;
        st_tok[chunk_first_sub].valid = 0;
        st_hi[chunk_first_sub].valid  = 0;
        st_lo[chunk_first_sub].valid  = 0;
        st_raw_hi[chunk_first_sub].valid = 0;
        st_raw_lo[chunk_first_sub].valid = 0;
        return;
    }
    if (ch.src_offset >= block_len) return;

    uint32_t chunk_end = ch.src_offset + ch.comp_size;
    if (chunk_end > block_len) chunk_end = block_len;
    const uint8_t* chunk_src = block + ch.src_offset;
    const uint32_t chunk_len = chunk_end - ch.src_offset;

    uint32_t sub_pos = 0;
    uint32_t remaining = ch.decomp_size;
    uint32_t sub_local = 0;
    const uint32_t n_subs_expected = (ch.decomp_size + cap_safe - 1) / cap_safe;

    while (remaining > 0 && sub_pos + 3 <= chunk_len) {
        const uint32_t sub_idx = chunk_first_sub + sub_local;
        st_lit[sub_idx].valid = 0;
        st_tok[sub_idx].valid = 0;
        st_hi[sub_idx].valid  = 0;
        st_lo[sub_idx].valid  = 0;
        st_raw_hi[sub_idx].valid = 0;
        st_raw_lo[sub_idx].valid = 0;

        if (chunk_len < 3) break;
        const uint32_t sub_hdr = scanReadU24BE(chunk_src + sub_pos);
        if ((sub_hdr & 0x800000u) == 0) break;
        const uint32_t sc_comp = sub_hdr & 0x7FFFFu;
        const uint32_t sub_end = sub_pos + 3 + sc_comp;
        if (sub_end > chunk_len) break;
        const uint32_t sub_decomp = remaining < cap_safe ? remaining : cap_safe;

        const uint32_t sub_dst_off = sub_idx * SCAN_SUBCHUNK_SLOT;
        const uint32_t init_b = (sub_idx == 0) ? 8u : 0u;
        uint32_t pos = sub_pos + 3 + init_b;
        const uint32_t end = sub_end;

        do {
            if (pos >= end) break;
            if (((chunk_src[pos] >> 4) & 0x7) == 4)
                scanParseHuffHeader(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off, st_lit[sub_idx]);
            uint32_t nxt = scanSkipStreamHeader(chunk_src, chunk_len, pos);
            if (nxt == 0xFFFFFFFFu) break;
            pos = nxt;
            if (pos >= end) break;

            if (((chunk_src[pos] >> 4) & 0x7) == 4)
                scanParseHuffHeader(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off, st_tok[sub_idx]);
            nxt = scanSkipStreamHeader(chunk_src, chunk_len, pos);
            if (nxt == 0xFFFFFFFFu) break;
            pos = nxt;
            if (pos >= end) break;

            if (sub_decomp > SCAN_SUBCHUNK_64K) {
                if (pos + 2 > end) break;
                pos += 2;
            }
            if (pos + 2 > end) break;
            const uint32_t marker = (uint32_t)chunk_src[pos] | ((uint32_t)chunk_src[pos + 1] << 8);
            if (marker != SCAN_OFF16_MARKER) break;
            pos += 2;
            if (pos >= end) break;

            const uint32_t hi_type = (chunk_src[pos] >> 4) & 0x7;
            if (hi_type == 0) {
                uint32_t doff, sz;
                if (scanParseType0(chunk_src, chunk_len, pos, doff, sz)) {
                    st_raw_hi[sub_idx].src_offset = ch.src_offset + doff;
                    st_raw_hi[sub_idx].size       = sz;
                    st_raw_hi[sub_idx].gpu_offset = sub_dst_off;
                    st_raw_hi[sub_idx].valid      = 1;
                }
            } else if (hi_type == 4) {
                scanParseHuffHeader(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off, st_hi[sub_idx]);
            }
            nxt = scanSkipStreamHeader(chunk_src, chunk_len, pos);
            if (nxt == 0xFFFFFFFFu) break;
            pos = nxt;
            if (pos >= end) break;

            const uint32_t lo_type = (chunk_src[pos] >> 4) & 0x7;
            if (lo_type == 0) {
                uint32_t doff, sz;
                if (scanParseType0(chunk_src, chunk_len, pos, doff, sz)) {
                    st_raw_lo[sub_idx].src_offset = ch.src_offset + doff;
                    st_raw_lo[sub_idx].size       = sz;
                    st_raw_lo[sub_idx].gpu_offset = sub_dst_off + SCAN_OFF16_LO_SLOT;
                    st_raw_lo[sub_idx].valid      = 1;
                }
            } else if (lo_type == 4) {
                scanParseHuffHeader(chunk_src, chunk_len, pos, ch.src_offset, sub_dst_off + SCAN_OFF16_LO_SLOT, st_lo[sub_idx]);
            }
        } while (0);

        sub_pos = sub_end;
        remaining -= sub_decomp;
        sub_local++;
        if (sub_local >= n_subs_expected) break;
    }
}
