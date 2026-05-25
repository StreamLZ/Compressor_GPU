// ── StreamLZ decode-scan kernel ────────────────────────────────
// GPU port of scanForEntropyChunks (decode/driver.zig): one thread per
// chunk walks every sub-chunk's literal / token / off16 stream headers
// and stages a descriptor per stream type per global sub-chunk index.
// The staged descriptor types (SlzScanHuffDesc / SlzScanRawDesc) live
// in compact_descs_kernels.cuh so the compact kernels can reference them.
// Included into the single lz_kernel.cu translation unit.
#pragma once

#include "slz_wire_format.cuh"
#include "compact_descs_kernels.cuh"

// ── Decode-scan kernel (roadmap 4d Phase 2) ─────────────────────────
// GPU port of scanForEntropyChunks (decode/driver.zig): one thread per
// chunk walks every sub-chunk's literal / token / off16 stream headers
// and stages a descriptor per stream type per global sub-chunk index.
// The driver D2H's the staged arrays and compacts them (drops raw /
// absent slots, assigns lut_offset). Verified byte-identical to the CPU
// scan in tools/huff_test/scan_test.cu across enwik8 + silesia L3-L5.

// SlzScanHuffDesc + SlzScanRawDesc defined earlier in the compact
// section so the compact kernels can reference them.

// Scan-local aliases for the shared wire-format slot offsets. Cast to
// uint32 here: a single sub-chunk's slot index never overflows u32, but
// the canonical ENTROPY_SCRATCH_SLOT_BYTES is u64 because it multiplies
// the global sub-chunk index in the LZ decode path.
static constexpr uint32_t SCAN_SUBCHUNK_SLOT = (uint32_t)ENTROPY_SCRATCH_SLOT_BYTES;
static constexpr uint32_t SCAN_OFF16_LO_SLOT = OFF16_HILO_SPLIT_OFFSET;

// These three parsers (scanParseHuffHeader / scanParseType0 /
// scanSkipStreamHeader) are the single-threaded scan kernel's
// position-based mirrors of slz_wire_format.cuh's
// parseEntropyHeader / parseRawStreamSize / skipEntropyStream. They
// share the actual bit-parsing core (parseEntropyHdrFields,
// parseType0HdrFields) but keep the (pos: uint32_t, chunk_len) shape
// because the scan must reject truncated frames by returning
// 0xFFFFFFFFu rather than dereferencing past the chunk end the way the
// per-warp decoder's cursor-passing helpers do.

// Parse a chunk_type=4 Huffman header at `pos` within chunk_src. The
// caller pre-clears out.valid; on any bounds-failure path here we
// simply leave it cleared (no callers consume the bool return), so the
// function returns void.
__device__ static void scanParseHuffHeader(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos,
    uint32_t src_offset_base, uint32_t dst_off, SlzScanHuffDesc& out) {
    if (pos >= chunk_len) return;
    // HEADER_LONG_FORM_BIT is named for the high bit that, when SET,
    // selects the SHORT (3-byte) header form. The 5-byte LONG form has
    // the bit clear. parseEntropyHdrFields encodes that contract via
    // h.header_bytes; this guard must match it byte-for-byte.
    const bool short_form = chunk_src[pos] >= HEADER_LONG_FORM_BIT;
    const uint32_t need = short_form ? ENTROPY_HEADER_SHORT_BYTES : ENTROPY_HEADER_LONG_BYTES;
    if (pos + need > chunk_len) return;
    const EntropyHdrFields h = parseEntropyHdrFields(chunk_src + pos);
    out.in_offset  = src_offset_base + pos + h.header_bytes;
    out.in_size    = h.comp_size;
    out.out_offset = dst_off;
    out.out_size   = h.dst_size;
    out.valid      = 1;
}

// Parse a type-0 (memcpy) stream header → data offset + size.
__device__ static bool scanParseType0(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos,
    uint32_t& data_off, uint32_t& size) {
    if (pos >= chunk_len) return false;
    // Short form = 2 bytes when HEADER_LONG_FORM_BIT is set; long form
    // = 3 bytes when clear. Matches parseType0HdrFields' header_bytes.
    const bool short_form = chunk_src[pos] >= HEADER_LONG_FORM_BIT;
    const uint32_t need = short_form ? 2u : 3u;
    if (pos + need > chunk_len) return false;
    const Type0HdrFields h = parseType0HdrFields(chunk_src + pos);
    size = h.size;
    data_off = pos + h.header_bytes;
    return data_off != 0;
}

// Skip an entropy/raw stream header + payload. 0xFFFFFFFF = truncated.
__device__ static uint32_t scanSkipStreamHeader(
    const uint8_t* chunk_src, uint32_t chunk_len, uint32_t pos) {
    if (pos >= chunk_len) return 0xFFFFFFFFu;
    const uint8_t first = chunk_src[pos];
    const uint32_t ct = (first >> CHUNK_TYPE_SHIFT) & CHUNK_TYPE_MASK;
    if (ct == 0) {
        const uint32_t need = (first >= HEADER_LONG_FORM_BIT) ? 2u : 3u;
        if (pos + need > chunk_len) return 0xFFFFFFFFu;
        const Type0HdrFields h = parseType0HdrFields(chunk_src + pos);
        return pos + h.header_bytes + h.size;
    } else if (ct == 1 || ct == 2 || ct == 4 || ct == 6) {
        const uint32_t need = (first >= HEADER_LONG_FORM_BIT)
            ? ENTROPY_HEADER_SHORT_BYTES : ENTROPY_HEADER_LONG_BYTES;
        if (pos + need > chunk_len) return 0xFFFFFFFFu;
        const EntropyHdrFields h = parseEntropyHdrFields(chunk_src + pos);
        return pos + h.header_bytes + h.comp_size;
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
    const uint32_t* __restrict__ d_n_chunks,
    uint32_t                    sub_chunk_cap,
    SlzScanHuffDesc* __restrict__ st_lit,
    SlzScanHuffDesc* __restrict__ st_tok,
    SlzScanHuffDesc* __restrict__ st_hi,
    SlzScanHuffDesc* __restrict__ st_lo,
    SlzScanRawDesc*  __restrict__ st_raw_hi,
    SlzScanRawDesc*  __restrict__ st_raw_lo)
{
    const uint32_t n_chunks = *d_n_chunks;
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

        const uint32_t sub_hdr = readBE24(chunk_src + sub_pos);
        if (!subchunkIsLz(sub_hdr)) break;
        const uint32_t sc_comp = subchunkCompSize(sub_hdr);
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

            if (sub_decomp > LZ_BLOCK_SIZE) {
                if (pos + 2 > end) break;
                pos += 2;
            }
            if (pos + 2 > end) break;
            const uint32_t marker = (uint32_t)chunk_src[pos] | ((uint32_t)chunk_src[pos + 1] << 8);
            if (marker != OFF16_ENTROPY_MARKER) break;
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
