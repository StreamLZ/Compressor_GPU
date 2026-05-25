// ── StreamLZ GPU frame-assembly kernel ──────────────────────────────
// Device-resident compress tail (roadmap 4d). Replaces the CPU
// `reencodeGpuWithEntropy` per-sub-chunk loop in fast_framed.zig: given
// the LZ kernel's raw sub-chunk streams and the GPU-Huffman bodies — all
// already device-resident — assemble each sub-chunk's payload on the GPU.
//
// Two passes, one warp per sub-chunk:
//   slzAssembleMeasureKernel — compute each sub-chunk's assembled size.
//   slzAssembleWriteKernel   — write [3-byte sub-chunk header][payload].
// Between them the driver prefix-sums the sizes (→ desc.out_offset) and
// writes the chunk / block / frame scaffolding.
//
// Built to assemble_kernel.ptx by tools/build_gpu_enc.bat (nvcc).
//
// Per-sub-chunk payload layout (matches reencodeGpuWithEntropy):
//   [literal chunk][token chunk][cmd2 2B if sub>64KB][off16][off32][length]
// Each entropy chunk is type-4 `[5B hdr][huff body]` when the Huffman
// body beats raw, else type-0 `[3B BE size][raw bytes]`. off32 and the
// length stream are copied verbatim from the raw payload.

#include <cstdint>
#include "../common/gpu_warp.cuh"          // WARP_SIZE, laneId()
#include "../common/gpu_byteio.cuh"        // readBE24
#include "../common/gpu_wire_format.cuh"   // LZ_BLOCK_SIZE, HUFF_CHUNK_TYPE, OFF16_ENTROPY_MARKER,
                                           // OFF32_COUNT_PACK_MAX, OFF32_LONG_ENTRY_TAG,
                                           // SUBCHUNK_LZ_FLAG_BIT, SUBCHUNK_MODE_SHIFT, SUBCHUNK_HDR_BYTES

// ── Wire-format constants (assembler-private) ───────────────────────
// LZ_BLOCK_SIZE, HUFF_CHUNK_TYPE, OFF16_ENTROPY_MARKER, OFF32_COUNT_PACK_MAX,
// OFF32_LONG_ENTRY_TAG, and the SUBCHUNK_* family come from
// ../common/gpu_wire_format.cuh — the encode/decode-shared contract.
static constexpr int      RAW_CHUNK_HDR_BYTES   = 3;   // type-0 [u24 BE size]
static constexpr int      HUFF_CHUNK_HDR_BYTES  = 5;   // type-4 non-compact header
static constexpr int      OFF16_ENTROPY_MIN     = 32;  // entropy-code off16 at/above this count

// ── Per-sub-chunk descriptor — mirrors Zig AssembleDesc in driver.zig ─
// Offsets are byte offsets into the corresponding device base buffer.
// A Huffman body with size 0 means "no body" → that stream is raw.
struct AssembleDesc {
    uint32_t raw_offset;          // sub-chunk raw payload, in d_raw
    uint32_t raw_size;            // raw payload byte count
    uint32_t huff_lit_offset;     // in d_huff_lit
    uint32_t huff_lit_size;
    uint32_t huff_tok_offset;     // in d_huff_tok
    uint32_t huff_tok_size;
    uint32_t huff_off16hi_offset; // in d_huff_off16
    uint32_t huff_off16hi_size;
    uint32_t huff_off16lo_offset; // in d_huff_off16
    uint32_t huff_off16lo_size;
    uint32_t sub_decomp_size;     // decompressed size of this sub-chunk
    uint32_t init_bytes;          // 8 for the frame's first sub-chunk (verbatim prefix), else 0
    uint32_t out_offset;          // assembled payload destination in d_frame (pass 2)
};

// ── Raw sub-chunk payload, parsed ───────────────────────────────────
struct RawStreams {
    const uint8_t* lit;     uint32_t lit_count;
    const uint8_t* tok;     uint32_t tok_count;
    const uint8_t* cmd2;    bool     cmd2_present;
    const uint8_t* off16;   uint32_t off16_count;   // off16 byte count = count*2
    const uint8_t* off32;   uint32_t off32_bytes;   // header + extra + data
    const uint8_t* length;  uint32_t length_size;
    bool ok;
};

// Parse a raw sub-chunk payload into its component streams. Mirrors the
// parsing half of reencodeGpuWithEntropy.
__device__ static RawStreams parseRaw(const uint8_t* raw, uint32_t raw_size,
                                      uint32_t sub_decomp_size) {
    RawStreams s;
    s.ok = false;
    uint32_t rp = 0;

    if (rp + 3 > raw_size) return s;
    s.lit_count = readBE24(raw + rp); rp += 3;
    if (rp + s.lit_count > raw_size) return s;
    s.lit = raw + rp; rp += s.lit_count;

    if (rp + 3 > raw_size) return s;
    s.tok_count = readBE24(raw + rp); rp += 3;
    if (rp + s.tok_count > raw_size) return s;
    s.tok = raw + rp; rp += s.tok_count;

    s.cmd2_present = (sub_decomp_size > LZ_BLOCK_SIZE);
    if (s.cmd2_present) {
        if (rp + 2 > raw_size) return s;
        s.cmd2 = raw + rp; rp += 2;
    } else {
        s.cmd2 = nullptr;
    }

    if (rp + 2 > raw_size) return s;
    s.off16_count = readU16LE(raw + rp); rp += 2;
    const uint32_t off16_bytes = s.off16_count * 2;
    if (rp + off16_bytes > raw_size) return s;
    s.off16 = raw + rp; rp += off16_bytes;

    // off32: 3-byte LE-packed header (+ 0/2/4 extra), then variable data.
    if (rp + 3 > raw_size) return s;
    const uint8_t* off32_start = raw + rp;
    const uint32_t packed = readLE24(raw + rp);
    rp += 3;
    uint32_t c1 = (packed >> 12) & 0xFFF;
    uint32_t c2 = packed & 0xFFF;
    uint32_t hdr_plus_extra = 3;
    if (c1 >= OFF32_COUNT_PACK_MAX) {
        if (rp + 2 > raw_size) return s;
        c1 = readU16LE(raw + rp);
        rp += 2; hdr_plus_extra += 2;
    }
    if (c2 >= OFF32_COUNT_PACK_MAX) {
        if (rp + 2 > raw_size) return s;
        c2 = readU16LE(raw + rp);
        rp += 2; hdr_plus_extra += 2;
    }
    // Scan c1+c2 entries; each is 3 bytes, or 4 when byte[2] >= 0xC0.
    const uint32_t total_entries = c1 + c2;
    uint32_t data_bytes = 0;
    uint32_t scan = rp;
    for (uint32_t i = 0; i < total_entries; i++) {
        if (scan + 3 > raw_size) return s;
        if (raw[scan + 2] >= OFF32_LONG_ENTRY_TAG) {
            if (scan + 4 > raw_size) return s;
            data_bytes += 4; scan += 4;
        } else {
            data_bytes += 3; scan += 3;
        }
    }
    rp += data_bytes;
    if (rp > raw_size) return s;
    s.off32 = off32_start;
    s.off32_bytes = hdr_plus_extra + data_bytes;

    // Remaining bytes are the length stream.
    s.length = raw + rp;
    s.length_size = raw_size - rp;
    s.ok = true;
    return s;
}

// ── Header writers ──────────────────────────────────────────────────
// type-0 raw chunk header is just `writeBE24` (3-byte big-endian size);
// every caller invokes the helper directly.

// type-4 non-compact 5-byte header (entropy_encoder.zig writeNonCompactChunkHeader).
__device__ __forceinline__ void writeHuffChunkHdr(uint8_t* d, uint32_t comp_size,
                                                  uint32_t decomp_size) {
    const uint32_t dm1 = decomp_size - 1;
    d[0] = (uint8_t)(((uint32_t)HUFF_CHUNK_TYPE << 4) | ((dm1 >> 14) & 0xF));
    const uint32_t bits = comp_size | ((dm1 & 0x3FFF) << 18);
    d[1] = (uint8_t)((bits >> 24) & 0xFF);
    d[2] = (uint8_t)((bits >> 16) & 0xFF);
    d[3] = (uint8_t)((bits >> 8) & 0xFF);
    d[4] = (uint8_t)(bits & 0xFF);
}

// Cooperative byte copy across the warp.
__device__ __forceinline__ void warpCopy(uint8_t* dst, const uint8_t* src,
                                          uint32_t n, int lane) {
    for (uint32_t i = lane; i < n; i += WARP_SIZE) dst[i] = src[i];
}

// ── One entropy stream: emit type-4 (huff) or type-0 (raw) ──────────
// Returns the emitted byte count. Writes only when `out` is non-null.
// type-4 wins when huff_body + 5 < raw_count + 3.
__device__ static uint32_t emitEntropyStream(
    uint8_t* out, const uint8_t* raw_bytes, uint32_t raw_count,
    const uint8_t* huff_body, uint32_t huff_size, int lane) {
    const bool use_huff = (huff_size > 0)
                       && (huff_size + HUFF_CHUNK_HDR_BYTES < raw_count + RAW_CHUNK_HDR_BYTES);
    if (use_huff) {
        if (out) {
            if (lane == 0) writeHuffChunkHdr(out, huff_size, raw_count);
            __syncwarp();
            warpCopy(out + HUFF_CHUNK_HDR_BYTES, huff_body, huff_size, lane);
        }
        return HUFF_CHUNK_HDR_BYTES + huff_size;
    }
    if (out) {
        if (lane == 0) writeBE24(out, raw_count);
        __syncwarp();
        warpCopy(out + RAW_CHUNK_HDR_BYTES, raw_bytes, raw_count, lane);
    }
    return RAW_CHUNK_HDR_BYTES + raw_count;
}

// ── Assemble one sub-chunk (measure when out==null, else write) ──────
// Returns the assembled payload byte count (enc_n), or 0 on a parse error.
__device__ static uint32_t assembleSubChunk(
    const uint8_t* d_raw, const uint8_t* d_huff_lit, const uint8_t* d_huff_tok,
    const uint8_t* d_huff_off16, const AssembleDesc& desc,
    uint8_t* out, int lane) {

    // The frame's first sub-chunk carries 8 verbatim raw bytes before the
    // literal header (matches reencodeGpuWithEntropy's `initial_bytes`).
    const uint8_t* raw_base = d_raw + desc.raw_offset;
    const uint32_t init_n = desc.init_bytes;
    if (init_n > desc.raw_size) return 0;
    const RawStreams s = parseRaw(raw_base + init_n, desc.raw_size - init_n,
                                  desc.sub_decomp_size);
    if (!s.ok) return 0;
    uint32_t wp = 0;

    // 0. initial raw bytes (frame's first sub-chunk only) — verbatim.
    if (init_n > 0) {
        if (out) { warpCopy(out, raw_base, init_n, lane); __syncwarp(); }
        wp += init_n;
    }

    // 1. literals — huff vs raw.
    wp += emitEntropyStream(out ? out + wp : nullptr, s.lit, s.lit_count,
                            d_huff_lit + desc.huff_lit_offset, desc.huff_lit_size, lane);

    // 2. tokens — huff vs raw.
    wp += emitEntropyStream(out ? out + wp : nullptr, s.tok, s.tok_count,
                            d_huff_tok + desc.huff_tok_offset, desc.huff_tok_size, lane);

    // 3. cmd2 — 2 bytes verbatim, when present.
    if (s.cmd2_present) {
        if (out && lane == 0) { out[wp] = s.cmd2[0]; out[wp + 1] = s.cmd2[1]; }
        wp += 2;
    }

    // 4. off16.
    const uint32_t off16_bytes = s.off16_count * 2;
    if (s.off16_count >= OFF16_ENTROPY_MIN) {
        const uint8_t* hi = d_huff_off16 + desc.huff_off16hi_offset;
        const uint8_t* lo = d_huff_off16 + desc.huff_off16lo_offset;
        const uint32_t hi_sz = desc.huff_off16hi_size;
        const uint32_t lo_sz = desc.huff_off16lo_size;
        const bool hi_huff = (hi_sz + 2 < s.off16_count);
        const bool lo_huff = (lo_sz + 2 < s.off16_count);
        const uint32_t hi_chunk = hi_huff ? (HUFF_CHUNK_HDR_BYTES + hi_sz)
                                          : (s.off16_count + RAW_CHUNK_HDR_BYTES);
        const uint32_t lo_chunk = lo_huff ? (HUFF_CHUNK_HDR_BYTES + lo_sz)
                                          : (s.off16_count + RAW_CHUNK_HDR_BYTES);
        const uint32_t split_total = hi_chunk + lo_chunk;
        if (split_total < off16_bytes) {
            // [u16 marker][hi chunk][lo chunk]
            if (out && lane == 0) storeU16LE(out + wp, OFF16_ENTROPY_MARKER);
            wp += 2;
            // hi
            if (hi_huff) {
                if (out && lane == 0) writeHuffChunkHdr(out + wp, hi_sz, s.off16_count);
                if (out) { __syncwarp(); warpCopy(out + wp + HUFF_CHUNK_HDR_BYTES, hi, hi_sz, lane); }
                wp += HUFF_CHUNK_HDR_BYTES + hi_sz;
            } else {
                if (out && lane == 0) writeBE24(out + wp, s.off16_count);
                if (out) {
                    __syncwarp();
                    for (uint32_t i = lane; i < s.off16_count; i += WARP_SIZE)
                        out[wp + RAW_CHUNK_HDR_BYTES + i] = s.off16[i * 2 + 1];
                }
                wp += RAW_CHUNK_HDR_BYTES + s.off16_count;
            }
            // lo
            if (lo_huff) {
                if (out && lane == 0) writeHuffChunkHdr(out + wp, lo_sz, s.off16_count);
                if (out) { __syncwarp(); warpCopy(out + wp + HUFF_CHUNK_HDR_BYTES, lo, lo_sz, lane); }
                wp += HUFF_CHUNK_HDR_BYTES + lo_sz;
            } else {
                if (out && lane == 0) writeBE24(out + wp, s.off16_count);
                if (out) {
                    __syncwarp();
                    for (uint32_t i = lane; i < s.off16_count; i += WARP_SIZE)
                        out[wp + RAW_CHUNK_HDR_BYTES + i] = s.off16[i * 2];
                }
                wp += RAW_CHUNK_HDR_BYTES + s.off16_count;
            }
        } else {
            // raw: [u16 off16_count][off16 data]
            if (out && lane == 0) storeU16LE(out + wp, (uint16_t)s.off16_count);
            wp += 2;
            if (out) { __syncwarp(); warpCopy(out + wp, s.off16, off16_bytes, lane); }
            wp += off16_bytes;
        }
    } else {
        // small off16: [u16 off16_count][off16 data]
        if (out && lane == 0) storeU16LE(out + wp, (uint16_t)s.off16_count);
        wp += 2;
        if (out) { __syncwarp(); warpCopy(out + wp, s.off16, off16_bytes, lane); }
        wp += off16_bytes;
    }

    // 5. off32 — verbatim copy (header + extra + data).
    if (out) { __syncwarp(); warpCopy(out + wp, s.off32, s.off32_bytes, lane); }
    wp += s.off32_bytes;

    // 6. length stream — verbatim copy.
    if (out) { __syncwarp(); warpCopy(out + wp, s.length, s.length_size, lane); }
    wp += s.length_size;

    return wp;
}

// ── Pass 1: measure each sub-chunk's assembled size ─────────────────
// Grid (n_subchunks, 1, 1), block (32, 1, 1). enc_sizes[i] = assembled
// payload byte count for sub-chunk i (0 on a parse error).
extern "C" __global__ void slzAssembleMeasureKernel(
    const uint8_t* __restrict__ d_raw,
    const uint8_t* __restrict__ d_huff_lit,
    const uint8_t* __restrict__ d_huff_tok,
    const uint8_t* __restrict__ d_huff_off16,
    const AssembleDesc* __restrict__ descs,
    uint32_t* __restrict__ enc_sizes,
    uint32_t n_subchunks)
{
    const uint32_t i = blockIdx.x;
    if (i >= n_subchunks) return;
    const int lane = laneId();
    const uint32_t n = assembleSubChunk(d_raw, d_huff_lit, d_huff_tok,
                                        d_huff_off16, descs[i], nullptr, lane);
    if (lane == 0) enc_sizes[i] = n;
}

// ── Pass 2: write [3-byte sub-chunk header][assembled payload] ──────
// Grid (n_subchunks, 1, 1), block (32, 1, 1). descs[i].out_offset is the
// destination of the 3-byte header in d_frame (driver fills it from the
// prefix-summed pass-1 sizes).
extern "C" __global__ void slzAssembleWriteKernel(
    const uint8_t* __restrict__ d_raw,
    const uint8_t* __restrict__ d_huff_lit,
    const uint8_t* __restrict__ d_huff_tok,
    const uint8_t* __restrict__ d_huff_off16,
    const AssembleDesc* __restrict__ descs,
    uint8_t* __restrict__ d_frame,
    uint32_t n_subchunks)
{
    const uint32_t i = blockIdx.x;
    if (i >= n_subchunks) return;
    const int lane = laneId();
    const AssembleDesc desc = descs[i];

    uint8_t* hdr = d_frame + desc.out_offset;
    uint8_t* payload = hdr + SUBCHUNK_HDR_BYTES;
    const uint32_t enc_n = assembleSubChunk(d_raw, d_huff_lit, d_huff_tok,
                                            d_huff_off16, desc, payload, lane);
    // 3-byte BE sub-chunk header: comp_size | (mode << SUBCHUNK_MODE_SHIFT)
    //                                       | SUBCHUNK_LZ_FLAG_BIT.
    // mode=1 here selects the LZ-with-entropy decoder path.
    if (lane == 0) {
        const uint32_t sc_hdr = enc_n | (1u << SUBCHUNK_MODE_SHIFT) | SUBCHUNK_LZ_FLAG_BIT;
        hdr[0] = (uint8_t)((sc_hdr >> 16) & 0xFF);
        hdr[1] = (uint8_t)((sc_hdr >> 8) & 0xFF);
        hdr[2] = (uint8_t)(sc_hdr & 0xFF);
    }
}

// ── Frame-assemble kernel (4d step 8) ───────────────────────────────
// Writes the complete StreamLZ frame to d_output on device:
//   [pre-formed frame_hdr+block_hdr (host-staged via d_prefix_bytes)]
//   per chunk i:
//     [2-byte internal_hdr][4-byte chunk_hdr LE u32 = (asm_total-1)]
//     [asm_total bytes copied from d_asm_out[asm_offsets[i]]]
//   [(n_chunks-1) * 8 B SC tail prefix from d_input]
//   [4-byte end mark = 0]
//
// Replaces the host loop in fast_framed.gpu_compress when SLZ_GPU_ASSEMBLE
// is set and the slzCompress D2D path is in use. Grid layout:
//   block_id < n_chunks  → write chunk's bytes (internal_hdr + chunk_hdr + asm).
//   block_id == n_chunks → write prefix bytes, SC tail, end mark.
//
// All offsets are in d_output's coordinate space.
extern "C" __global__ void slzFrameAssembleKernel(
    const uint8_t* __restrict__ d_input,            // for SC tail prefix bytes
    const uint8_t* __restrict__ d_asm_out,          // assembled sub-chunk blocks
    const uint32_t* __restrict__ d_asm_offsets,     // per-chunk first asm offset
    const uint32_t* __restrict__ d_asm_chunk_sizes, // per-chunk total asm size (sum of sub-chunk asm sizes)
    const uint32_t* __restrict__ d_chunk_dst,       // per-chunk dst offset (start of 2B internal_hdr)
    const uint8_t* __restrict__ d_prefix_bytes,     // pre-formed frame_hdr + block_hdr bytes
    uint32_t prefix_size,                            // length of d_prefix_bytes
    uint8_t  internal_hdr0,
    uint8_t  internal_hdr1,
    uint32_t n_chunks,
    uint32_t eff_chunk_size,                         // source chunk size in bytes (for SC tail src offset)
    uint32_t src_len,                                // total source length (for SC tail last-entry clamp)
    uint32_t sc_tail_off,                            // dst offset of SC tail prefix table
    uint32_t end_mark_off,                           // dst offset of end mark
    uint8_t* __restrict__ d_output)
{
    const uint32_t block_id = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    const uint32_t bdim = blockDim.x;

    if (block_id < n_chunks) {
        // Per-chunk write: 2B internal_hdr + 4B chunk_hdr + asm bytes.
        const uint32_t dst_off = d_chunk_dst[block_id];
        const uint32_t asm_start = d_asm_offsets[block_id];
        const uint32_t asm_total = d_asm_chunk_sizes[block_id];

        if (lane == 0) {
            d_output[dst_off + 0] = internal_hdr0;
            d_output[dst_off + 1] = internal_hdr1;
            const uint32_t hdr_u32 = asm_total - 1;  // chunk_hdr LE u32
            d_output[dst_off + 2] = (uint8_t)(hdr_u32 & 0xFF);
            d_output[dst_off + 3] = (uint8_t)((hdr_u32 >> 8) & 0xFF);
            d_output[dst_off + 4] = (uint8_t)((hdr_u32 >> 16) & 0xFF);
            d_output[dst_off + 5] = (uint8_t)((hdr_u32 >> 24) & 0xFF);
        }
        // Cooperative copy of the assembled sub-chunk block(s).
        const uint32_t payload_dst = dst_off + 6;
        for (uint32_t i = lane; i < asm_total; i += bdim) {
            d_output[payload_dst + i] = d_asm_out[asm_start + i];
        }
        return;
    }

    if (block_id == n_chunks) {
        // Block N: prefix bytes (frame_hdr + block_hdr) at d_output[0..prefix_size],
        // SC tail prefix table, end mark.
        for (uint32_t i = lane; i < prefix_size; i += bdim) {
            d_output[i] = d_prefix_bytes[i];
        }

        // SC tail prefix: (n_chunks - 1) entries of 8 bytes each. Each entry
        // copies the first 8 bytes of chunk (entry_idx+1) from d_input.
        // The last entry may have fewer than 8 source bytes — pad with zeros.
        if (n_chunks > 1) {
            const uint32_t total_tail_bytes = (n_chunks - 1) * 8u;
            for (uint32_t i = lane; i < total_tail_bytes; i += bdim) {
                const uint32_t entry_idx = i / 8;
                const uint32_t byte_in_entry = i - entry_idx * 8;
                const uint32_t chunk_idx = entry_idx + 1;
                const uint32_t src_off = chunk_idx * eff_chunk_size + byte_in_entry;
                const uint8_t v = (src_off < src_len) ? d_input[src_off] : (uint8_t)0;
                d_output[sc_tail_off + i] = v;
            }
        }

        // End mark: 4 zero bytes.
        if (lane < 4) d_output[end_mark_off + lane] = 0;
        return;
    }
}
