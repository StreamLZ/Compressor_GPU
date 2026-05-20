// GPU Huffman 4-stream decoder.
//
// Format mirrors the CPU reference (`huff_ref.c`):
//   [9-byte header: 3 × u24 LE stream sizes; stream4 derived]
//   [stream 0 bits]
//   [stream 1 bits]
//   [stream 2 bits]
//   [stream 3 bits]
//
// Codes are MSB-first within bytes; decoder uses a 2048-entry fast LUT
// (11-bit prefix → (length:8, symbol:8) packed in u16).
//
// One block per Huffman block (= one sub-chunk in production). 4 active
// lanes per warp each decode one of the 4 streams independently. Other
// lanes idle. We can revisit lane utilization (e.g., 8 lanes/stream with
// shared bit buffer) once correctness is proven.

#include <cstdint>
#include <cstdio>
#include <cuda_pipeline.h>   // cp.async (LDGSTS) intrinsics

static constexpr int MAX_CODE_LEN = 10;
static constexpr int LUT_SIZE     = 1 << MAX_CODE_LEN;  // 1024
static constexpr int LUT_BYTES    = LUT_SIZE * sizeof(uint32_t);  // 4096
// u16 single-symbol LUT: half the shared-memory footprint (4 KB), so a
// warp's occupancy limit roughly doubles. See decode_stream_one_lane_lut16.
static constexpr int LUT16_BYTES  = LUT_SIZE * sizeof(uint16_t);  // 4096

// Word-interleaved 4-stream format header: 4 × u24 LE per-stream byte
// sizes (the 4th can't be derived — the interleaved payload is padded to
// the longest stream, so total bytes != sum of stream sizes).
static constexpr int HUFF_4SI_HDR_BYTES = 12;

// 4-byte-aligned 4-stream format header: 4 × u24 LE per-stream byte sizes
// (12 bytes — itself 4-aligned). Each stream is zero-padded to a 4-byte
// boundary, so every stream starts 4-aligned and the refill is one
// aligned 32-bit load. See huff_encode_4s_aligned.
static constexpr int HUFF_4SA_HDR_BYTES = 12;

struct HuffBlockDesc {
    uint32_t in_offset;     // byte offset in compressed buffer (points at 9-byte header)
    uint32_t in_size;       // total compressed bytes including header
    uint32_t out_offset;    // byte offset in output buffer
    uint32_t out_size;      // expected decompressed bytes
    uint32_t lut_offset;    // entry offset in luts array (LUT_SIZE entries per block)
};

// ── Single-stream decode core ─────────────────────────────────────────
// Decodes one Huffman bit-stream of `in_size` bytes into `out_size` bytes.
// Pure single-thread; called by per-lane dispatch in the warp kernel.
// Returns number of output bytes written (== out_size on success, 0 on
// internal error).
__device__ __forceinline__ uint32_t decode_stream_one_lane(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    // Top-aligned bit buffer: bits [63 .. 64-bit_count] are valid, lower
    // are zero. Allows 8-byte big-endian refill when bit_count == 0.
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;

    // ── Hot loop: unrolled 2× — one refill guards two LUT decodes ──
    // Each LUT lookup is indexed by MAX_CODE_LEN (11) bits and consumes
    // total_len ≤ 11 bits while emitting 1 OR 2 symbols. Two decodes
    // consume ≤ 22 bits.
    //
    // The 32-bit refill `bit_buf |= v << (32 - bit_count)` is only valid
    // when bit_count ≤ 32. We refill whenever bit_count < 2*MAX_CODE_LEN
    // (< 22), so the shift amount is in [10, 32] — always valid — and the
    // post-refill bit_count is in [32, 53], i.e. ≥ 22, enough for the two
    // decodes below with no mid-group refill check. This halves the
    // refill-branch / loop-branch / bounds-check overhead per symbol.
    //
    // Each iteration emits up to 4 symbols (2 decodes × ≤2 syms), so the
    // output-room guard is `out_pos + 4 <= out_size`.
    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        out[out_pos]     = (uint8_t)(e0 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e0 >> 8) & 0xFF);
        out_pos += ns0;
        bit_buf <<= len0;
        bit_count -= len0;
        // decode B (≤ 22 bits total consumed this iter; refill above
        // guaranteed ≥ 22 valid bits, so no refill check needed here)
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        out[out_pos]     = (uint8_t)(e1 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e1 >> 8) & 0xFF);
        out_pos += ns1;
        bit_buf <<= len1;
        bit_count -= len1;
    }

    // ── Tail loop: handles last 1-2 bytes; clamps dual entries to single ──
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        // Even if we clamped to single output, we still consumed total_len
        // bits — see CPU reference rationale (only happens at end-of-stream).
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Single-stream decode core, u16 single-symbol LUT ─────────────────
// Same bitstream/format as decode_stream_one_lane, but the LUT is u16
// (2048 × 2 B = 4 KB shared vs 8 KB). Each lookup yields exactly ONE
// symbol — no dual-symbol fast path — trading ~33-50% more lookups for
// half the shared-memory footprint, which roughly doubles occupancy.
// u16 entry: bits 0-7 = symbol, bits 8-11 = code length (1..11).
__device__ __forceinline__ uint32_t decode_stream_one_lane_lut16(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint16_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;

    // Hot loop unrolled 2× — one refill guards two decodes. Two single-
    // symbol decodes consume ≤ 22 bits; refilling when bit_count < 22
    // leaves ≥ 22 valid bits, so no mid-group refill check is needed.
    while (out_pos + 2 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 8) & 0xFF;
        if (len0 == 0) return 0;
        out[out_pos++] = (uint8_t)(e0 & 0xFF);
        bit_buf <<= len0;
        bit_count -= len0;
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 8) & 0xFF;
        if (len1 == 0) return 0;
        out[out_pos++] = (uint8_t)(e1 & 0xFF);
        bit_buf <<= len1;
        bit_count -= len1;
    }

    // ── Tail loop: last 0-1 bytes ──
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 8) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Single-stream decode core, software-prefetched ───────────────────
// Identical decode semantics to decode_stream_one_lane, but issues the
// next 32-bit refill load one refill ahead so its ~200-cycle global-load
// latency overlaps the LUT-decode work of the current iteration. The
// kernel is latency-bound (NCU: 43.5% Long Scoreboard on the refill
// load), so hiding load latency — not improving bandwidth — is the lever.
//
// Depth-1 prefetch: `nextw` always holds the next refill word, already
// loaded. When a refill is needed we consume `nextw`, then immediately
// issue the load for the word after it. The two LUT decodes that follow
// are independent of `nextw`, so the load runs in their shadow.
__device__ __forceinline__ uint32_t decode_stream_prefetch(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;

    // `nextw` holds the 32-bit word at [fetch_pos-4 .. fetch_pos-1] when
    // have_next is set — loaded one refill ahead of consumption.
    uint32_t fetch_pos = 0;
    uint32_t nextw = 0;
    bool have_next = false;
    if (fetch_pos + 4 <= in_size) {
        nextw = ((uint32_t)in[fetch_pos    ] << 24)
              | ((uint32_t)in[fetch_pos + 1] << 16)
              | ((uint32_t)in[fetch_pos + 2] <<  8)
              | ((uint32_t)in[fetch_pos + 3]);
        fetch_pos += 4;
        have_next = true;
    }

    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (!have_next) break;
            // Consume the already-loaded word, then issue the NEXT load
            // immediately so its latency overlaps the two decodes below.
            uint32_t v = nextw;
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            if (fetch_pos + 4 <= in_size) {
                nextw = ((uint32_t)in[fetch_pos    ] << 24)
                      | ((uint32_t)in[fetch_pos + 1] << 16)
                      | ((uint32_t)in[fetch_pos + 2] <<  8)
                      | ((uint32_t)in[fetch_pos + 3]);
                fetch_pos += 4;
            } else {
                have_next = false;
            }
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        out[out_pos]     = (uint8_t)(e0 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e0 >> 8) & 0xFF);
        out_pos += ns0;
        bit_buf <<= len0;
        bit_count -= len0;
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        out[out_pos]     = (uint8_t)(e1 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e1 >> 8) & 0xFF);
        out_pos += ns1;
        bit_buf <<= len1;
        bit_count -= len1;
    }

    // Drain: hand any unconsumed prefetched word back to in_pos so the
    // tail loop re-reads from the correct byte position.
    in_pos = have_next ? (fetch_pos - 4) : fetch_pos;

    // ── Tail loop: identical to decode_stream_one_lane ──
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Single-stream decode core, wide (32-bit) output stores ───────────
// Same decode semantics as decode_stream_one_lane (u32 dual-symbol LUT),
// but decoded bytes are accumulated and flushed to global memory as
// 32-bit stores instead of single-byte stores. The 4 active lanes write
// to output regions 16 KB apart, so a warp's store touches 4 separate
// sectors regardless — but a 4-byte store carries 4 useful bytes per
// sector vs 1, cutting store transactions (and L2 store traffic) ~4×.
// Assumes `out` is 4-byte aligned (true for 64 KB blocks: each lane's
// region starts at a multiple of out_size/4 = 16384).
__device__ __forceinline__ uint32_t decode_stream_one_lane_wstore(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;    // total bytes decoded
    uint32_t written = 0;    // bytes flushed to global (always a multiple of 4)
    uint64_t acc = 0;        // pending decoded bytes, byte 0 = oldest
    int pending = 0;         // ≤ 3 at loop top, ≤ 5 transiently before a flush

    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A — append 1-2 bytes; entry's sym2 byte is 0 when ns==1,
        // and is overwritten by the next append, so masking 0xFFFF is safe.
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        acc |= (uint64_t)(e0 & 0xFFFF) << (pending * 8);
        pending += ns0;
        bit_buf <<= len0;
        bit_count -= len0;
        out_pos += ns0;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        acc |= (uint64_t)(e1 & 0xFFFF) << (pending * 8);
        pending += ns1;
        bit_buf <<= len1;
        bit_count -= len1;
        out_pos += ns1;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
    }

    // Drain whole pending bytes (written catches up to out_pos), then
    // finish byte-at-a-time.
    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending--;
    }
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Single-stream decode core, aligned 32-bit refill + wide stores ───
// Identical to decode_stream_one_lane_wstore, except the hot-loop refill
// is a single aligned 32-bit load + byte-swap instead of four byte loads
// — cutting ~8 instructions per refill (3 IMAD.WIDE shifts + 3 ORs + 3
// extra LDG). Requires `in` 4-byte aligned (huff_encode_4s_aligned pads
// every stream to a 4-byte boundary); in_pos is always a multiple of 4
// in the hot loop. The tail keeps byte loads.
__device__ __forceinline__ uint32_t decode_stream_one_lane_wstore_aligned(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;
    uint32_t written = 0;
    uint64_t acc = 0;
    int pending = 0;

    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            // One aligned 32-bit load; byte-swap to big-endian order so
            // `v` matches the byte-by-byte assembly (in[pos] = MSB).
            uint32_t word = *reinterpret_cast<const uint32_t*>(in + in_pos);
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        acc |= (uint64_t)(e0 & 0xFFFF) << (pending * 8);
        pending += ns0;
        bit_buf <<= len0;
        bit_count -= len0;
        out_pos += ns0;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        acc |= (uint64_t)(e1 & 0xFFFF) << (pending * 8);
        pending += ns1;
        bit_buf <<= len1;
        bit_count -= len1;
        out_pos += ns1;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
    }

    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending--;
    }
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Decode core: 10-bit LUT + 11-bit escape, aligned refill, wide store ─
// Like decode_stream_one_lane_wstore_aligned, but the LUT is 1024-entry
// (10-bit index, 4 KB) while codes may be up to 11 bits — no ratio loss.
// MAX_CODE_LEN (10) is the fast index width; codes can be one bit longer.
// An escape entry (num_syms == 3) means "length-11 code": the decoder
// reads the 11th bit to pick byte0 (bit==0) or byte1 (bit==1).
__device__ __forceinline__ uint32_t decode_stream_one_lane_wstore_aligned_esc(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;
    uint32_t written = 0;
    uint64_t acc = 0;
    int pending = 0;

    // Two decodes consume ≤ 2*(MAX_CODE_LEN+1) = 22 bits; refill below 22.
    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * (MAX_CODE_LEN + 1)) {
            if (in_pos + 4 > in_size) break;
            uint32_t word = *reinterpret_cast<const uint32_t*>(in + in_pos);
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        uint32_t bytes0; int app0;
        if (ns0 == 3) {  // escape: length-11 code
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            bytes0 = b11 ? ((e0 >> 8) & 0xFF) : (e0 & 0xFF);
            app0 = 1;
        } else {
            bytes0 = e0 & 0xFFFF;
            app0 = ns0;
        }
        acc |= (uint64_t)bytes0 << (pending * 8);
        pending += app0;
        bit_buf <<= len0;
        bit_count -= len0;
        out_pos += app0;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        uint32_t bytes1; int app1;
        if (ns1 == 3) {
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            bytes1 = b11 ? ((e1 >> 8) & 0xFF) : (e1 & 0xFF);
            app1 = 1;
        } else {
            bytes1 = e1 & 0xFFFF;
            app1 = ns1;
        }
        acc |= (uint64_t)bytes1 << (pending * 8);
        pending += app1;
        bit_buf <<= len1;
        bit_count -= len1;
        out_pos += app1;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
    }

    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending--;
    }
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN + 1 && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN + 1) bit_count = MAX_CODE_LEN + 1;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        if (num_syms == 3) {  // escape
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            out[out_pos++] = (uint8_t)(b11 ? ((e >> 8) & 0xFF) : (e & 0xFF));
        } else {
            out[out_pos++] = (uint8_t)(e & 0xFF);
            if (num_syms == 2 && out_pos < out_size) {
                out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
            }
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Decode core: cp.async ring-buffer refill (escape LUT + wide store) ─
// Same decode as decode_stream_one_lane_wstore_aligned_esc, but the
// 32-bit bit-buffer refill is fed by an asynchronous global→shared copy
// (cp.async / LDGSTS) instead of a synchronous global load. The refill
// word is staged into a small per-lane shared ring CPA_DEPTH words ahead;
// cp.async removes the ~200-cycle load from the warp's register/scoreboard
// dependency chain entirely, so it overlaps the serial decode arithmetic.
// This is the technique nvCOMP's zstd decoder uses (SASS-confirmed).
// `ring` points at this lane's CPA_DEPTH-slot shared ring (CPA_DEPTH is a
// power of two). Requires `in` 4-byte aligned (huff_encode_4s_aligned).
static constexpr int CPA_DEPTH = 4;   // refill words prefetched ahead

__device__ __forceinline__ uint32_t decode_stream_one_lane_cpasync_esc(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size,
    uint32_t* ring)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t out_pos = 0;
    uint32_t written = 0;
    uint64_t acc = 0;
    int pending = 0;

    const uint32_t n_words = in_size >> 2;   // full 32-bit words in the stream
    uint32_t fetch = 0;     // next word index to cp.async
    uint32_t consume = 0;   // next word index to read from the ring

    // Prologue: prime CPA_DEPTH pipeline stages. Every iteration commits a
    // group (a real cp.async if words remain, else an empty group), so the
    // committed-group count stays exactly CPA_DEPTH ahead of `consume` —
    // letting every wait use the constant CPA_DEPTH-1.
    #pragma unroll
    for (int d = 0; d < CPA_DEPTH; d++) {
        if (fetch < n_words) {
            __pipeline_memcpy_async(&ring[fetch & (CPA_DEPTH - 1)], in + fetch * 4, 4);
            fetch++;
        }
        __pipeline_commit();
    }

    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * (MAX_CODE_LEN + 1)) {
            if (consume >= n_words) break;
            __pipeline_wait_prior(CPA_DEPTH - 1);   // word `consume` now staged
            uint32_t word = ring[consume & (CPA_DEPTH - 1)];
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            consume++;
            // Refill the pipeline: one new group per consumed word.
            if (fetch < n_words) {
                __pipeline_memcpy_async(&ring[fetch & (CPA_DEPTH - 1)], in + fetch * 4, 4);
                fetch++;
            }
            __pipeline_commit();
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        uint32_t bytes0; int app0;
        if (ns0 == 3) {
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            bytes0 = b11 ? ((e0 >> 8) & 0xFF) : (e0 & 0xFF);
            app0 = 1;
        } else {
            bytes0 = e0 & 0xFFFF;
            app0 = ns0;
        }
        acc |= (uint64_t)bytes0 << (pending * 8);
        pending += app0;
        bit_buf <<= len0;
        bit_count -= len0;
        out_pos += app0;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        uint32_t bytes1; int app1;
        if (ns1 == 3) {
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            bytes1 = b11 ? ((e1 >> 8) & 0xFF) : (e1 & 0xFF);
            app1 = 1;
        } else {
            bytes1 = e1 & 0xFFFF;
            app1 = ns1;
        }
        acc |= (uint64_t)bytes1 << (pending * 8);
        pending += app1;
        bit_buf <<= len1;
        bit_count -= len1;
        out_pos += app1;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
    }

    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending--;
    }
    // Tail: byte loads straight from global, starting past the consumed words.
    uint32_t in_pos = consume * 4;
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN + 1 && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN + 1) bit_count = MAX_CODE_LEN + 1;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        if (num_syms == 3) {
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            out[out_pos++] = (uint8_t)(b11 ? ((e >> 8) & 0xFF) : (e & 0xFF));
        } else {
            out[out_pos++] = (uint8_t)(e & 0xFF);
            if (num_syms == 2 && out_pos < out_size) {
                out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
            }
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// Cold-path helper for length-11 escape codes. __noinline__ forces a
// real out-of-line call so the common decode path branches around it,
// instead of the compiler flattening the escape ops inline (which burned
// ~4 ops/symbol on every non-escape symbol).
__device__ __noinline__
uint32_t huff_escape_pick(uint32_t e, uint64_t bit_buf)
{
    uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
    return b11 ? ((e >> 8) & 0xFFu) : (e & 0xFFu);
}

// ── Decode core: cp.async refill + escape on a COLD branch ───────────
// Identical to decode_stream_one_lane_cpasync_esc, but the length-11
// escape is a real predicted-cold branch (huff_escape_pick) rather than
// branchless SELs — keeping ~4 escape-only ops per symbol off the common
// path. The kernel is compute-bound (NCU: 78.7% SOL), so trimming
// hot-loop instruction count converts fairly directly to speed.
__device__ __forceinline__ uint32_t decode_stream_one_lane_cpasync_esc_cb(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size,
    uint32_t* ring)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t out_pos = 0;
    uint32_t written = 0;
    uint64_t acc = 0;
    int pending = 0;

    const uint32_t n_words = in_size >> 2;
    uint32_t fetch = 0;
    uint32_t consume = 0;

    #pragma unroll
    for (int d = 0; d < CPA_DEPTH; d++) {
        if (fetch < n_words) {
            __pipeline_memcpy_async(&ring[fetch & (CPA_DEPTH - 1)], in + fetch * 4, 4);
            fetch++;
        }
        __pipeline_commit();
    }

    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * (MAX_CODE_LEN + 1)) {
            if (consume >= n_words) break;
            __pipeline_wait_prior(CPA_DEPTH - 1);
            uint32_t word = ring[consume & (CPA_DEPTH - 1)];
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            consume++;
            if (fetch < n_words) {
                __pipeline_memcpy_async(&ring[fetch & (CPA_DEPTH - 1)], in + fetch * 4, 4);
                fetch++;
            }
            __pipeline_commit();
        }
        // decode A — common path is the ≤10-bit normal entry.
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        if (len0 == 0) return 0;
        uint32_t bytes0; int app0;
        if (__builtin_expect(len0 <= MAX_CODE_LEN, 1)) {
            bytes0 = e0 & 0xFFFF;
            app0 = (int)(e0 >> 24);
        } else {
            bytes0 = huff_escape_pick(e0, bit_buf);
            app0 = 1;
        }
        acc |= (uint64_t)bytes0 << (pending * 8);
        pending += app0;
        bit_buf <<= len0;
        bit_count -= len0;
        out_pos += app0;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        if (len1 == 0) return 0;
        uint32_t bytes1; int app1;
        if (__builtin_expect(len1 <= MAX_CODE_LEN, 1)) {
            bytes1 = e1 & 0xFFFF;
            app1 = (int)(e1 >> 24);
        } else {
            bytes1 = huff_escape_pick(e1, bit_buf);
            app1 = 1;
        }
        acc |= (uint64_t)bytes1 << (pending * 8);
        pending += app1;
        bit_buf <<= len1;
        bit_count -= len1;
        out_pos += app1;
        if (pending >= 4) {
            *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
            acc >>= 32;
            pending -= 4;
            written += 4;
        }
    }

    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending--;
    }
    uint32_t in_pos = consume * 4;
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN + 1 && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN + 1) bit_count = MAX_CODE_LEN + 1;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        if (num_syms == 3) {
            uint32_t b11 = (uint32_t)((bit_buf >> (63 - MAX_CODE_LEN)) & 1u);
            out[out_pos++] = (uint8_t)(b11 ? ((e >> 8) & 0xFF) : (e & 0xFF));
        } else {
            out[out_pos++] = (uint8_t)(e & 0xFF);
            if (num_syms == 2 && out_pos < out_size) {
                out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
            }
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Single-stream decode core, wide (64-bit) output stores ───────────
// Like decode_stream_one_lane_wstore but flushes 8 bytes per global
// store, halving store transactions again vs the 32-bit variant. Each
// decode appends 1-2 bytes; a u64 accumulator can hold at most 8, and an
// append could otherwise overshoot, so bytes are appended one at a time
// and an 8-byte store fires the moment 8 are pending (wflush8).
// Assumes `out` is 8-byte aligned (true for 64 KB blocks: each lane's
// region starts at a multiple of 16384).
__device__ __forceinline__ void wflush8(uint64_t& acc, int& pending,
                                        uint8_t* out, uint32_t& written, uint8_t b)
{
    acc |= (uint64_t)b << (pending * 8);
    if (++pending == 8) {
        *reinterpret_cast<uint64_t*>(out + written) = acc;
        written += 8;
        acc = 0;
        pending = 0;
    }
}

__device__ __forceinline__ uint32_t decode_stream_one_lane_wstore64(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t in_pos = 0;
    uint32_t out_pos = 0;    // total bytes decoded
    uint32_t written = 0;    // bytes flushed to global (always a multiple of 8)
    uint64_t acc = 0;        // pending decoded bytes, byte 0 = oldest
    int pending = 0;         // 0..7 between appends

    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (in_pos + 4 > in_size) break;
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        // decode A
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        wflush8(acc, pending, out, written, (uint8_t)(e0 & 0xFF));
        if (ns0 == 2) wflush8(acc, pending, out, written, (uint8_t)((e0 >> 8) & 0xFF));
        bit_buf <<= len0;
        bit_count -= len0;
        out_pos += ns0;
        // decode B
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        wflush8(acc, pending, out, written, (uint8_t)(e1 & 0xFF));
        if (ns1 == 2) wflush8(acc, pending, out, written, (uint8_t)((e1 >> 8) & 0xFF));
        bit_buf <<= len1;
        bit_count -= len1;
        out_pos += ns1;
    }

    // Drain whole pending bytes (written catches up to out_pos), then
    // finish byte-at-a-time.
    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending--;
    }
    while (out_pos < out_size) {
        if (bit_count <= 32 && in_pos + 4 <= in_size) {
            uint32_t v = ((uint32_t)in[in_pos    ] << 24)
                       | ((uint32_t)in[in_pos + 1] << 16)
                       | ((uint32_t)in[in_pos + 2] <<  8)
                       | ((uint32_t)in[in_pos + 3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            in_pos += 4;
        }
        while (bit_count < MAX_CODE_LEN && in_pos < in_size) {
            bit_buf |= ((uint64_t)in[in_pos++]) << (56 - bit_count);
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Single-stream encode core ─────────────────────────────────────────
// Sequential encode of one stream by one lane. Writes packed bits MSB-first
// into `out`. Returns total bytes written (including trailing partial byte).
__device__ __forceinline__ uint32_t encode_stream_one_lane(
    const uint8_t* __restrict__ in, uint32_t in_size,
    const uint8_t* __restrict__ code_lengths,    // 256 entries
    const uint32_t* __restrict__ codes,          // 256 entries
    uint8_t* __restrict__ out)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t out_pos = 0;

    for (uint32_t i = 0; i < in_size; i++) {
        uint8_t  sym = in[i];
        uint32_t code = codes[sym];
        int      len  = code_lengths[sym];
        bit_buf = (bit_buf << len) | (uint64_t)(code & ((1u << len) - 1u));
        bit_count += len;
        // Drain whole bytes from the top of bit_buf.
        while (bit_count >= 8) {
            bit_count -= 8;
            out[out_pos++] = (uint8_t)((bit_buf >> bit_count) & 0xFFu);
        }
    }
    // Flush trailing partial byte (pad LSB with zeros).
    if (bit_count > 0) {
        out[out_pos++] = (uint8_t)((bit_buf << (8 - bit_count)) & 0xFFu);
    }
    return out_pos;
}

// ── Encoder kernel: 1 block per Huffman block, 4 active lanes ─────────
// Each block encodes ONE Huffman block (sub-chunk) given the input bytes
// and pre-built code tables. Lanes 0..3 each encode one of the 4 quarters
// in parallel into a SCRATCH buffer; then lane 0 writes the 9-byte header
// + concatenates the streams into the final output buffer.
//
// Scratch layout (per block): 4 contiguous regions, each with worst-case
// size = (in_quarter_size * 2 + 64) bytes. Host pre-allocates global mem.
extern "C" __global__ void huffEncode4StreamKernel(
    const uint8_t* __restrict__ input,
    const HuffBlockDesc* __restrict__ descs_in,       // (in_offset, in_size, scratch_off, scratch_per_stream)
    const uint8_t* __restrict__ code_lengths,         // 256 × n_blocks
    const uint32_t* __restrict__ codes,               // 256 × n_blocks
    uint8_t* __restrict__ scratch,                    // per-stream scratch buffer
    uint8_t* __restrict__ output,                     // packed output
    uint32_t* __restrict__ out_sizes,                 // per-block output size
    const uint32_t* __restrict__ out_offsets,         // per-block output offset
    uint32_t scratch_per_stream,                      // per-stream scratch capacity
    uint32_t tables_stride,                           // 256 (per-block table stride)
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc d = descs_in[block_id];

    // All 32 lanes compute `bytes` (lanes 4..31 stay 0). This keeps the
    // following __shfl_sync convergent (all lanes participate).
    uint32_t bytes = 0;
    if (lane < 4) {
        uint32_t q = d.in_size / 4;
        uint32_t my_in_start = (lane < 3) ? (lane * q) : (3 * q);
        uint32_t my_in_size  = (lane < 3) ? q : (d.in_size - 3 * q);
        const uint8_t* my_in = input + d.in_offset + my_in_start;

        // Scratch layout per block: 4 streams × scratch_per_stream bytes,
        // starting at scratch + block_id * 4 * scratch_per_stream.
        uint8_t* my_scratch = scratch + ((uint64_t)block_id * 4 + lane) * scratch_per_stream;

        // Per-block code table base.
        const uint8_t*  my_cl = code_lengths + (uint64_t)block_id * tables_stride;
        const uint32_t* my_cd = codes        + (uint64_t)block_id * tables_stride;

        bytes = encode_stream_one_lane(my_in, my_in_size, my_cl, my_cd, my_scratch);
    }
    __syncwarp();

    // Convergent shuffles — every lane must participate.
    uint32_t bytes_l0 = __shfl_sync(0xFFFFFFFF, bytes, 0);
    uint32_t bytes_l1 = __shfl_sync(0xFFFFFFFF, bytes, 1);
    uint32_t bytes_l2 = __shfl_sync(0xFFFFFFFF, bytes, 2);
    uint32_t bytes_l3 = __shfl_sync(0xFFFFFFFF, bytes, 3);

    if (lane == 0) {
        // Write header + concatenate scratch streams into output[out_offsets[block_id]].
        uint8_t* out = output + out_offsets[block_id];
        uint32_t s0 = bytes_l0, s1 = bytes_l1, s2 = bytes_l2, s3 = bytes_l3;
        out[0] = (uint8_t)(s0 & 0xFF); out[1] = (uint8_t)((s0 >> 8) & 0xFF); out[2] = (uint8_t)((s0 >> 16) & 0xFF);
        out[3] = (uint8_t)(s1 & 0xFF); out[4] = (uint8_t)((s1 >> 8) & 0xFF); out[5] = (uint8_t)((s1 >> 16) & 0xFF);
        out[6] = (uint8_t)(s2 & 0xFF); out[7] = (uint8_t)((s2 >> 8) & 0xFF); out[8] = (uint8_t)((s2 >> 16) & 0xFF);

        uint8_t* dst = out + 9;
        uint8_t* src0 = scratch + ((uint64_t)block_id * 4 + 0) * scratch_per_stream;
        uint8_t* src1 = scratch + ((uint64_t)block_id * 4 + 1) * scratch_per_stream;
        uint8_t* src2 = scratch + ((uint64_t)block_id * 4 + 2) * scratch_per_stream;
        uint8_t* src3 = scratch + ((uint64_t)block_id * 4 + 3) * scratch_per_stream;
        for (uint32_t i = 0; i < s0; i++) dst[i] = src0[i]; dst += s0;
        for (uint32_t i = 0; i < s1; i++) dst[i] = src1[i]; dst += s1;
        for (uint32_t i = 0; i < s2; i++) dst[i] = src2[i]; dst += s2;
        for (uint32_t i = 0; i < s3; i++) dst[i] = src3[i];

        out_sizes[block_id] = 9 + s0 + s1 + s2 + s3;
    }
}

// ── Encoder kernel: full chunk_type=4 body — STEP 2 of GPU Huffman enc ──
// Like huffEncode4StreamKernel, but emits the PRODUCTION body layout
// (huffman_encoder.zig encodeBlock, minus the 5-byte chunk header which the
// production frame assembler prepends):
//   [128 B weights — 4 bits/symbol, byte i = cl[2i] | cl[2i+1]<<4]
//   [9 B sub-header — 3 × u24 LE stream sizes; stream 3 derived]
//   [stream 0 | stream 1 | stream 2 | stream 3]
// Code tables come from slzHuffBuildTablesKernel.
extern "C" __global__ void slzHuffEncode4StreamKernel(
    const uint8_t* __restrict__ input,
    const HuffBlockDesc* __restrict__ descs_in,
    const uint8_t* __restrict__ code_lengths,         // 256 × n_blocks
    const uint32_t* __restrict__ codes,               // 256 × n_blocks
    uint8_t* __restrict__ scratch,                    // per-stream scratch
    uint8_t* __restrict__ output,                     // packed bodies
    uint32_t* __restrict__ out_sizes,                 // per-block body size
    const uint32_t* __restrict__ out_offsets,         // per-block body offset
    uint32_t scratch_per_stream,
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc d = descs_in[block_id];

    // Lanes 0..3 each encode one of the 4 quarters into scratch.
    uint32_t bytes = 0;
    if (lane < 4) {
        uint32_t q = d.in_size / 4;
        uint32_t my_in_start = (lane < 3) ? (lane * q) : (3 * q);
        uint32_t my_in_size  = (lane < 3) ? q : (d.in_size - 3 * q);
        const uint8_t* my_in = input + d.in_offset + my_in_start;
        uint8_t* my_scratch = scratch + ((uint64_t)block_id * 4 + lane) * scratch_per_stream;
        const uint8_t*  my_cl = code_lengths + (uint64_t)block_id * tables_stride;
        const uint32_t* my_cd = codes        + (uint64_t)block_id * tables_stride;
        bytes = encode_stream_one_lane(my_in, my_in_size, my_cl, my_cd, my_scratch);
    }
    __syncwarp();

    // Convergent shuffles — every lane participates.
    uint32_t s0 = __shfl_sync(0xFFFFFFFF, bytes, 0);
    uint32_t s1 = __shfl_sync(0xFFFFFFFF, bytes, 1);
    uint32_t s2 = __shfl_sync(0xFFFFFFFF, bytes, 2);
    uint32_t s3 = __shfl_sync(0xFFFFFFFF, bytes, 3);

    uint8_t* out = output + out_offsets[block_id];
    const uint8_t* cl = code_lengths + (uint64_t)block_id * tables_stride;

    // 128-byte weights — 32 lanes pack 4 entries each.
    for (int i = lane; i < 128; i += 32)
        out[i] = (uint8_t)((cl[2 * i] & 0x0F) | ((cl[2 * i + 1] & 0x0F) << 4));

    if (lane == 0) {
        // 9-byte sub-header at offset 128 (3 × u24 LE; stream 3 derived).
        uint8_t* h = out + 128;
        h[0] = (uint8_t)(s0 & 0xFF); h[1] = (uint8_t)((s0 >> 8) & 0xFF); h[2] = (uint8_t)((s0 >> 16) & 0xFF);
        h[3] = (uint8_t)(s1 & 0xFF); h[4] = (uint8_t)((s1 >> 8) & 0xFF); h[5] = (uint8_t)((s1 >> 16) & 0xFF);
        h[6] = (uint8_t)(s2 & 0xFF); h[7] = (uint8_t)((s2 >> 8) & 0xFF); h[8] = (uint8_t)((s2 >> 16) & 0xFF);

        uint8_t* dst = out + 137;
        uint8_t* src0 = scratch + ((uint64_t)block_id * 4 + 0) * scratch_per_stream;
        uint8_t* src1 = scratch + ((uint64_t)block_id * 4 + 1) * scratch_per_stream;
        uint8_t* src2 = scratch + ((uint64_t)block_id * 4 + 2) * scratch_per_stream;
        uint8_t* src3 = scratch + ((uint64_t)block_id * 4 + 3) * scratch_per_stream;
        for (uint32_t i = 0; i < s0; i++) dst[i] = src0[i]; dst += s0;
        for (uint32_t i = 0; i < s1; i++) dst[i] = src1[i]; dst += s1;
        for (uint32_t i = 0; i < s2; i++) dst[i] = src2[i]; dst += s2;
        for (uint32_t i = 0; i < s3; i++) dst[i] = src3[i];

        out_sizes[block_id] = 128 + 9 + s0 + s1 + s2 + s3;
    }
}

// ══════════════════════════════════════════════════════════════════════
// 32-STREAM VARIANT — all 32 lanes active, one stream per lane
// ══════════════════════════════════════════════════════════════════════
//
// Format:
//   [93-byte header: 31 × u24 LE stream sizes; stream 31 size derived]
//   [stream 0 bits | stream 1 | ... | stream 31]
//
// Same MSB-first per-byte bit order as the 4-stream version. Each lane
// (0..31) decodes/encodes its own quarter (= 1/32 of output). For 16 KB
// literal streams, each lane handles 512 bytes.

static constexpr int N_STREAMS_32 = 32;
static constexpr int HUFF_32S_HDR_BYTES = (N_STREAMS_32 - 1) * 3;  // 93

// ── Encoder kernel: 1 block per Huffman block, 32 active lanes ───────
extern "C" __global__ void huffEncode32StreamKernel(
    const uint8_t* __restrict__ input,
    const HuffBlockDesc* __restrict__ descs_in,
    const uint8_t* __restrict__ code_lengths,
    const uint32_t* __restrict__ codes,
    uint8_t* __restrict__ scratch,
    uint8_t* __restrict__ output,
    uint32_t* __restrict__ out_sizes,
    const uint32_t* __restrict__ out_offsets,
    uint32_t scratch_per_stream,
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc d = descs_in[block_id];

    // Per-lane quarter (32-way split).
    uint32_t q = d.in_size / 32;
    uint32_t my_in_start = (lane < 31) ? (lane * q) : (31 * q);
    uint32_t my_in_size  = (lane < 31) ? q : (d.in_size - 31 * q);
    const uint8_t* my_in = input + d.in_offset + my_in_start;

    uint8_t* my_scratch = scratch + ((uint64_t)block_id * 32 + lane) * scratch_per_stream;
    const uint8_t*  my_cl = code_lengths + (uint64_t)block_id * tables_stride;
    const uint32_t* my_cd = codes        + (uint64_t)block_id * tables_stride;

    uint32_t bytes = encode_stream_one_lane(my_in, my_in_size, my_cl, my_cd, my_scratch);

    __syncwarp();

    // Gather all 32 stream sizes into every lane (32 shuffles, convergent).
    uint32_t all_sizes[N_STREAMS_32];
    #pragma unroll
    for (int s = 0; s < N_STREAMS_32; s++) {
        all_sizes[s] = __shfl_sync(0xFFFFFFFF, bytes, s);
    }

    // Lane 0 writes header + concatenates payload.
    if (lane == 0) {
        uint8_t* out = output + out_offsets[block_id];
        uint8_t* hdr = out;
        uint32_t total = HUFF_32S_HDR_BYTES;
        for (int s = 0; s < N_STREAMS_32 - 1; s++) {
            uint32_t sz = all_sizes[s];
            hdr[s*3 + 0] = (uint8_t)(sz & 0xFF);
            hdr[s*3 + 1] = (uint8_t)((sz >> 8) & 0xFF);
            hdr[s*3 + 2] = (uint8_t)((sz >> 16) & 0xFF);
            total += sz;
        }
        total += all_sizes[N_STREAMS_32 - 1];

        uint8_t* dst = out + HUFF_32S_HDR_BYTES;
        for (int s = 0; s < N_STREAMS_32; s++) {
            uint32_t sz = all_sizes[s];
            uint8_t* src = scratch + ((uint64_t)block_id * 32 + s) * scratch_per_stream;
            for (uint32_t i = 0; i < sz; i++) dst[i] = src[i];
            dst += sz;
        }
        out_sizes[block_id] = total;
    }
}

// ── Decoder kernel: 1 block per Huffman block, 32 active lanes ───────
extern "C" __global__ void huffDecode32StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    // Cooperative LUT load.
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    // Parse header: 31 × u24 LE sizes + derive stream 31 size.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t my_size;
    if (lane < N_STREAMS_32 - 1) {
        my_size = (uint32_t)hdr[lane*3 + 0]
                | ((uint32_t)hdr[lane*3 + 1] << 8)
                | ((uint32_t)hdr[lane*3 + 2] << 16);
    } else {
        my_size = 0;  // computed via warp reduce below
    }
    // Warp reduce sum of sizes[0..30] to derive size[31].
    uint32_t sum = my_size;
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, off);
    if (lane == N_STREAMS_32 - 1) {
        my_size = (desc.in_size - HUFF_32S_HDR_BYTES) - sum;
    }
    __syncwarp();

    // Compute per-lane input offset via prefix-sum (exclusive scan).
    uint32_t in_off = my_size;
    for (int off = 1; off < 32; off <<= 1) {
        uint32_t v = __shfl_up_sync(0xFFFFFFFF, in_off, off);
        if (lane >= off) in_off += v;
    }
    in_off -= my_size;  // exclusive

    const uint8_t* my_in = hdr + HUFF_32S_HDR_BYTES + in_off;

    // Per-lane output region.
    uint32_t q = desc.out_size / 32;
    uint32_t my_out_start = (lane < 31) ? (lane * q) : (31 * q);
    uint32_t my_out_size  = (lane < 31) ? q : (desc.out_size - 31 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane(my_in, my_size, shared_lut,
                                              my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: 1 block per Huffman block, 4 active lanes ────────
//
// Grid: (n_blocks, 1, 1). Block: (32, 1, 1).
//
// Each block decodes one Huffman block (one sub-chunk). The block's warp
// cooperatively loads the LUT into shared memory (4KB), then 4 lanes
// (0..3) each decode one of the 4 streams. Other lanes idle.
extern "C" __global__ void huffDecode4StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    // Load LUT cooperatively into shared memory (2048 × 2 = 4096 bytes).
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) {
        shared_lut[i] = src_lut[i];
    }
    __syncwarp();

    if (lane >= 4) return;

    // Parse header: 3 × u24 LE sizes.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    // Compute this lane's input pointer (offset within compressed payload).
    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    // Compute this lane's output region (4 contiguous quarters).
    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane(my_in, my_in_size, shared_lut,
                                              my_out, my_out_size);
    (void)written;  // future: write per-block error flag if written != my_out_size
}

// ── Decoder kernel: 2 Huffman blocks per warp, 8 active lanes ─────────
// Each warp handles a PAIR of blocks: lanes 0-3 decode block (2*pid)'s
// 4 streams, lanes 4-7 decode block (2*pid+1)'s. 8 active lanes amortize
// the warp's per-instruction issue slot 2x vs the 4-lane kernel, while
// staying well below the 32-lane LDS-bank-conflict regime (8 random LUT
// indices into 32 banks rarely collide). Shared memory holds 2 LUTs.
// Launch: <<<(n_blocks + 1) / 2, 32, 2 * LUT_BYTES>>>
extern "C" __global__ void huffDecode4Stream2xKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t pid  = blockIdx.x;
    const uint32_t bidA = pid * 2;
    const uint32_t bidB = bidA + 1;
    if (bidA >= n_blocks) return;
    const bool haveB = (bidB < n_blocks);
    const int lane = threadIdx.x & 31;

    // Shared memory holds two LUTs: [0, LUT_SIZE) for block A,
    // [LUT_SIZE, 2*LUT_SIZE) for block B. All 32 lanes load cooperatively.
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* lutA = luts + descs[bidA].lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = lutA[i];
    if (haveB) {
        const uint32_t* lutB = luts + descs[bidB].lut_offset;
        for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[LUT_SIZE + i] = lutB[i];
    }
    __syncwarp();

    // Active lanes 0-7: lane>>2 picks the block, lane&3 picks the stream.
    if (lane >= 8) return;
    if (lane >= 4 && !haveB) return;
    const uint32_t sub   = (uint32_t)lane >> 2;   // 0 = block A, 1 = block B
    const int      slane = lane & 3;              // stream index 0-3
    const HuffBlockDesc desc = descs[sub ? bidB : bidA];
    const uint32_t* my_lut = shared_lut + sub * LUT_SIZE;

    // Parse header: 3 × u24 LE sizes; stream 3 derived.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    stream_sizes[3] = (desc.in_size - 9)
                    - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t s_in_off = 0;
    for (int s = 0; s < slane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size  = stream_sizes[slane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (slane < 3) ? (slane * q) : (3 * q);
    uint32_t my_out_size  = (slane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane(my_in, my_in_size, my_lut,
                                              my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: 1 block per warp, only 2 active lanes ─────────────
// Each of 2 lanes decodes TWO of the block's 4 streams sequentially.
// Lane 0 -> streams 0,1 ; lane 1 -> streams 2,3. This is the "narrow"
// counterpart to the 4/8/32-lane variants: it minimizes concurrent LUT
// lookups (2 lanes -> 2 indices) at the cost of 2x serial work per lane.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4Stream2LaneKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 2) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    stream_sizes[3] = (desc.in_size - 9)
                    - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t prefix[4];
    prefix[0] = 0;
    prefix[1] = stream_sizes[0];
    prefix[2] = stream_sizes[0] + stream_sizes[1];
    prefix[3] = stream_sizes[0] + stream_sizes[1] + stream_sizes[2];

    const uint32_t q = desc.out_size / 4;

    // This lane owns streams (2*lane) and (2*lane + 1).
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        const int s = 2 * lane + k;
        const uint8_t* my_in = hdr + 9 + prefix[s];
        uint32_t my_in_size  = stream_sizes[s];
        uint32_t my_out_start = (s < 3) ? (s * q) : (3 * q);
        uint32_t my_out_size  = (s < 3) ? q : (desc.out_size - 3 * q);
        uint8_t* my_out = output + desc.out_offset + my_out_start;
        uint32_t written = decode_stream_one_lane(my_in, my_in_size, shared_lut,
                                                  my_out, my_out_size);
        (void)written;
    }
}

// ── Interleaved-stream decode core ────────────────────────────────────
// Decodes one stream of a word-interleaved 4-stream block. The stream's
// 32-bit words are not contiguous: word w lives at stream_base + w*16
// (the 4 streams' word-w slots sit adjacent so the 4 decode lanes' refill
// loads coalesce into one 32-byte sector instead of 4 scattered ones).
// `in_size` is this stream's byte length (its last word may hold trailing
// zero padding, harmless — output is bounded by out_size).
__device__ __forceinline__ uint32_t decode_stream_interleaved(
    const uint8_t* __restrict__ stream_base, uint32_t in_size,
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    int bit_count = 0;
    uint32_t word_idx = 0;
    const uint32_t n_words = (in_size + 3) / 4;
    uint32_t out_pos = 0;

    // Hot loop: unrolled 2× — one strided word refill guards two decodes.
    while (out_pos + 4 <= out_size) {
        if (bit_count < 2 * MAX_CODE_LEN) {
            if (word_idx >= n_words) break;
            const uint8_t* p = stream_base + (size_t)word_idx * 16;
            uint32_t v = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
                       | ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            word_idx++;
        }
        uint32_t e0 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len0 = (e0 >> 16) & 0xFF;
        int ns0  = (e0 >> 24) & 0xFF;
        if (len0 == 0) return 0;
        out[out_pos]     = (uint8_t)(e0 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e0 >> 8) & 0xFF);
        out_pos += ns0;
        bit_buf <<= len0;
        bit_count -= len0;
        uint32_t e1 = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int len1 = (e1 >> 16) & 0xFF;
        int ns1  = (e1 >> 24) & 0xFF;
        if (len1 == 0) return 0;
        out[out_pos]     = (uint8_t)(e1 & 0xFF);
        out[out_pos + 1] = (uint8_t)((e1 >> 8) & 0xFF);
        out_pos += ns1;
        bit_buf <<= len1;
        bit_count -= len1;
    }

    // Tail: 1 decode per iter, word-at-a-time refill, end-of-stream pad.
    while (out_pos < out_size) {
        if (bit_count <= 32 && word_idx < n_words) {
            const uint8_t* p = stream_base + (size_t)word_idx * 16;
            uint32_t v = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
                       | ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
            bit_buf |= ((uint64_t)v) << (32 - bit_count);
            bit_count += 32;
            word_idx++;
        }
        if (bit_count < MAX_CODE_LEN) bit_count = MAX_CODE_LEN;
        uint32_t e = lut[(uint32_t)(bit_buf >> (64 - MAX_CODE_LEN))];
        int total_len = (e >> 16) & 0xFF;
        int num_syms  = (e >> 24) & 0xFF;
        if (total_len == 0) return 0;
        out[out_pos++] = (uint8_t)(e & 0xFF);
        if (num_syms == 2 && out_pos < out_size) {
            out[out_pos++] = (uint8_t)((e >> 8) & 0xFF);
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── Decoder kernel: word-interleaved 4-stream, 4 active lanes ─────────
// Same 4-lane layout as huffDecode4StreamKernel, but the compressed
// block is word-interleaved (see huff_encode_4s_interleaved). The 4
// lanes' refill loads now hit adjacent addresses and coalesce, fixing
// the uncoalesced-global-load stall the profiler flagged.
// Header: 4 × u24 LE per-stream byte sizes.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamInterleavedKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    // Header: 4 × u24 LE per-stream byte sizes.
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t my_in_size = (uint32_t)hdr[lane*3 + 0]
                        | ((uint32_t)hdr[lane*3 + 1] << 8)
                        | ((uint32_t)hdr[lane*3 + 2] << 16);

    // This lane decodes stream `lane`. Its word 0 is at
    // hdr + HUFF_4SI_HDR_BYTES + lane*4; successive words stride by 16.
    const uint8_t* stream_base = hdr + HUFF_4SI_HDR_BYTES + lane * 4;

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_interleaved(stream_base, my_in_size,
                                                 shared_lut, my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: software-prefetched, 4 active lanes ──────────────
// Identical layout/format to huffDecode4StreamKernel (contiguous 9-byte
// header, 3 × u24 stream sizes). Only the decode core differs:
// decode_stream_prefetch issues each 32-bit refill load one refill ahead
// to hide global-load latency — the profiled bottleneck (Long Scoreboard).
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamPrefetchKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_prefetch(my_in, my_in_size, shared_lut,
                                              my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: u16 single-symbol LUT, 4 active lanes ────────────
// Identical layout/format to huffDecode4StreamKernel (contiguous 9-byte
// header, 3 × u24). The LUT is u16 (4 KB shared vs 8 KB) — this halves
// the per-warp shared-memory footprint, so the SM's shared-mem-limited
// block count roughly doubles (NCU: occupancy was capped at 22.9% by
// the 8 KB LUT). decode_stream_one_lane_lut16 does single-symbol decode.
// Launch: <<<n_blocks, 32, LUT16_BYTES>>>
extern "C" __global__ void huffDecode4StreamLut16Kernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint16_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint16_t shared_lut16[];
    const uint16_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut16[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_lut16(my_in, my_in_size, shared_lut16,
                                                    my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: wide (32-bit) output stores, 4 active lanes ──────
// Identical layout/format/LUT to huffDecode4StreamKernel (contiguous
// 9-byte header, u32 dual-symbol LUT). Only the decode core differs:
// decode_stream_one_lane_wstore batches decoded bytes into 32-bit global
// stores, cutting store transactions / L2 store traffic ~4× — targeting
// the L2 SOL ceiling (81.6%) the profiler flagged.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamWStoreKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_wstore(my_in, my_in_size, shared_lut,
                                                     my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: wide (64-bit) output stores, 4 active lanes ──────
// Identical layout/format/LUT to huffDecode4StreamKernel. Decodes with
// decode_stream_one_lane_wstore64 — 8-byte global stores, halving store
// transactions again vs the 32-bit wstore kernel.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamWStore64Kernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 3; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }
    uint32_t total_streams = desc.in_size - 9;
    stream_sizes[3] = total_streams - stream_sizes[0] - stream_sizes[1] - stream_sizes[2];

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += stream_sizes[s];
    const uint8_t* my_in = hdr + 9 + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_wstore64(my_in, my_in_size, shared_lut,
                                                       my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: aligned 32-bit refill + 32-bit stores ────────────
// Uses the 4-byte-aligned format (huff_encode_4s_aligned): 12-byte header
// of 4 × u24 stream sizes, each stream zero-padded to a 4-byte boundary.
// Every stream start is 4-aligned, so the decode core's refill is one
// aligned 32-bit load. Combines that with the 32-bit wide output stores.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamWStoreAlignedKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    // Header: 4 × u24 LE stream sizes (12 bytes).
    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }

    // Each stream is padded to a 4-byte boundary, so offsets advance by
    // the rounded-up size.
    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += (stream_sizes[s] + 3u) & ~3u;
    const uint8_t* my_in = hdr + HUFF_4SA_HDR_BYTES + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_wstore_aligned(my_in, my_in_size, shared_lut,
                                                             my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: 10-bit-LUT + 11-bit-escape, aligned refill ───────
// Same aligned format / 4 KB LUT as huffDecode4StreamWStoreAlignedKernel,
// but codes may be 11 bits — the LUT carries escape entries for them, so
// occupancy stays doubled (4 KB LUT) at zero compression cost.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamWStoreAlignedEscKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += (stream_sizes[s] + 3u) & ~3u;
    const uint8_t* my_in = hdr + HUFF_4SA_HDR_BYTES + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_wstore_aligned_esc(my_in, my_in_size, shared_lut,
                                                                 my_out, my_out_size);
    (void)written;
}

// ── Decoder kernel: cp.async ring-buffer refill ──────────────────────
// Same aligned format / escape LUT as huffDecode4StreamWStoreAlignedEscKernel.
// Each decode lane refills its bit buffer from a small per-lane shared
// ring fed by cp.async — targeting the input-load-latency stall (NCU:
// 33.5% Long Scoreboard on the refill load).
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamCpAsyncEscKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    __shared__ uint32_t cpa_ring[4][CPA_DEPTH];   // per decode-lane refill ring
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += (stream_sizes[s] + 3u) & ~3u;
    const uint8_t* my_in = hdr + HUFF_4SA_HDR_BYTES + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_cpasync_esc(my_in, my_in_size, shared_lut,
                                                          my_out, my_out_size,
                                                          &cpa_ring[lane][0]);
    (void)written;
}

// ── Decoder kernel: cp.async refill + cold-branch escape ─────────────
// Same aligned format / escape LUT as huffDecode4StreamCpAsyncEscKernel,
// but the length-11 escape is a predicted-cold branch — trimming the
// common-path instruction count.
// Launch: <<<n_blocks, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamCpAsyncEscCbKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;

    const int lane = threadIdx.x & 31;
    const HuffBlockDesc desc = descs[block_id];

    extern __shared__ uint32_t shared_lut[];
    __shared__ uint32_t cpa_ring[4][CPA_DEPTH];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
    __syncwarp();

    if (lane >= 4) return;

    const uint8_t* hdr = comp + desc.in_offset;
    uint32_t stream_sizes[4];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                        | ((uint32_t)hdr[s*3 + 1] << 8)
                        | ((uint32_t)hdr[s*3 + 2] << 16);
    }

    uint32_t s_in_off = 0;
    for (int s = 0; s < lane; s++) s_in_off += (stream_sizes[s] + 3u) & ~3u;
    const uint8_t* my_in = hdr + HUFF_4SA_HDR_BYTES + s_in_off;
    uint32_t my_in_size = stream_sizes[lane];

    uint32_t q = desc.out_size / 4;
    uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
    uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    uint32_t written = decode_stream_one_lane_cpasync_esc_cb(my_in, my_in_size, shared_lut,
                                                             my_out, my_out_size,
                                                             &cpa_ring[lane][0]);
    (void)written;
}

// ── Decoder kernel: persistent + atomic work queue (tail-effect fix) ──
// Same per-block decode as huffDecode4StreamWStoreAlignedEscKernel, but
// launched as a PERSISTENT kernel: grid = exactly one wave of CUDA blocks
// (SMs × blocks/SM). Each block is a worker that atomically grabs the
// next Huffman block from a global counter until the queue drains. This
// targets the partial-wave tail NCU flagged, and load-balances the
// non-uniform per-block decode durations. `g_counter` must be zeroed
// before each launch.
// Launch: <<<one_wave, 32, LUT_BYTES>>>
extern "C" __global__ void huffDecode4StreamWStoreAlignedEscQueueKernel(
    const uint8_t* __restrict__ comp,
    const HuffBlockDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    uint32_t n_blocks,
    uint32_t* __restrict__ g_counter)
{
    const int lane = threadIdx.x & 31;
    extern __shared__ uint32_t shared_lut[];

    for (;;) {
        // Lane 0 grabs the next block index; broadcast to the warp.
        uint32_t block_id = 0;
        if (lane == 0) block_id = atomicAdd(g_counter, 1u);
        block_id = __shfl_sync(0xFFFFFFFFu, block_id, 0);
        if (block_id >= n_blocks) break;

        const HuffBlockDesc desc = descs[block_id];
        const uint32_t* src_lut = luts + desc.lut_offset;
        for (int i = lane; i < LUT_SIZE; i += 32) shared_lut[i] = src_lut[i];
        __syncwarp();

        if (lane < 4) {
            const uint8_t* hdr = comp + desc.in_offset;
            uint32_t stream_sizes[4];
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                stream_sizes[s] = (uint32_t)hdr[s*3 + 0]
                                | ((uint32_t)hdr[s*3 + 1] << 8)
                                | ((uint32_t)hdr[s*3 + 2] << 16);
            }
            uint32_t s_in_off = 0;
            for (int s = 0; s < lane; s++) s_in_off += (stream_sizes[s] + 3u) & ~3u;
            const uint8_t* my_in = hdr + HUFF_4SA_HDR_BYTES + s_in_off;
            uint32_t my_in_size = stream_sizes[lane];

            uint32_t q = desc.out_size / 4;
            uint32_t my_out_start = (lane < 3) ? (lane * q) : (3 * q);
            uint32_t my_out_size  = (lane < 3) ? q : (desc.out_size - 3 * q);
            uint8_t* my_out = output + desc.out_offset + my_out_start;

            decode_stream_one_lane_wstore_aligned_esc(my_in, my_in_size, shared_lut,
                                                      my_out, my_out_size);
        }
        __syncwarp();  // decode (reads shared_lut) done before next iter's load
    }
}

// ══════════════════════════════════════════════════════════════════
//  LUT-build testbed: serial vs parallel 10-bit-escape LUT construction.
//  Input  : code_lengths_all — 256 bytes per block (lengths 0..11).
//  Output : luts — LUT_SIZE (1024) u32 entries per block.
//  Both compute canonical codes internally, then build the escape LUT.
// ══════════════════════════════════════════════════════════════════

// Shared canonical-code setup (histogram + next_code + codes). All 32
// lanes participate; leaves cl[], codes[] populated, lut[] zeroed.
__device__ __forceinline__ void huffBuildSetup(
    const uint8_t* CL, uint32_t* lut, int lane,
    uint8_t* cl, uint32_t* codes, uint32_t* bl_count, uint32_t* next_code)
{
    for (int i = lane; i < 256; i += 32) cl[i] = CL[i];
    if (lane < 16) bl_count[lane] = 0;
    __syncwarp();
    for (int s = lane; s < 256; s += 32) {
        uint8_t L = cl[s];
        if (L > 0 && L <= MAX_CODE_LEN + 1) atomicAdd(&bl_count[L], 1);
    }
    __syncwarp();
    if (lane == 0) {
        uint32_t code = 0;
        next_code[0] = 0;
        for (int L = 1; L <= MAX_CODE_LEN + 1; L++) {
            code = (code + bl_count[L - 1]) << 1;
            next_code[L] = code;
        }
        for (int s = 0; s < 256; s++) {
            uint8_t L = cl[s];
            codes[s] = (L == 0) ? 0u : next_code[L]++;
        }
    }
    __syncwarp();
    for (int i = lane; i < LUT_SIZE; i += 32) lut[i] = 0;
    __syncwarp();
}

__device__ __forceinline__ uint32_t huffPack(uint32_t ns, uint32_t len,
                                              uint32_t s2, uint32_t s1) {
    return (ns << 24) | (len << 16) | (s2 << 8) | s1;
}

// Escape pass — length-(MAX_CODE_LEN+1) codes. Lane 0 only (rare; sibling
// pairs share a prefix so parallel writes would race).
__device__ __forceinline__ void huffBuildEscapePass(
    const uint8_t* cl, const uint32_t* codes, uint32_t* lut, int lane)
{
    if (lane != 0) return;
    for (int s = 0; s < 256; s++) {
        if (cl[s] != MAX_CODE_LEN + 1) continue;
        uint32_t c = codes[s];
        uint32_t p = c >> 1;
        uint32_t e = lut[p];
        uint32_t b0 = e & 0xFF, b1 = (e >> 8) & 0xFF;
        if (c & 1u) b1 = (uint32_t)s; else b0 = (uint32_t)s;
        lut[p] = huffPack(3, MAX_CODE_LEN + 1, b1, b0);
    }
}

// ── Serial build: lane 0 does the whole pass 1 + pass 2 fan-out ──────
extern "C" __global__ void huffBuildLutEscSerialKernel(
    const uint8_t* __restrict__ code_lengths_all,
    uint32_t* __restrict__ luts,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    __shared__ uint8_t  cl[256];
    __shared__ uint32_t codes[256];
    __shared__ uint32_t bl_count[16];
    __shared__ uint32_t next_code[16];

    const uint8_t* CL = code_lengths_all + (size_t)block_id * 256;
    uint32_t* lut = luts + (size_t)block_id * LUT_SIZE;
    huffBuildSetup(CL, lut, lane, cl, codes, bl_count, next_code);

    if (lane == 0) {
        for (int s = 0; s < 256; s++) {                 // pass 1
            int L = cl[s];
            if (L == 0 || L > MAX_CODE_LEN) continue;
            uint32_t aligned = codes[s] << (MAX_CODE_LEN - L);
            uint32_t span = 1u << (MAX_CODE_LEN - L);
            uint32_t entry = huffPack(1, (uint32_t)L, 0, (uint32_t)s);
            for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
        }
        for (int s1 = 0; s1 < 256; s1++) {              // pass 2
            int L1 = cl[s1];
            if (L1 == 0 || L1 >= MAX_CODE_LEN) continue;
            uint32_t C1 = codes[s1];
            for (int s2 = 0; s2 < 256; s2++) {
                int L2 = cl[s2];
                if (L2 == 0) continue;
                int total = L1 + L2;
                if (total > MAX_CODE_LEN) continue;
                uint32_t C2 = codes[s2];
                uint32_t aligned = (C1 << (MAX_CODE_LEN - L1))
                                 | (C2 << (MAX_CODE_LEN - L1 - L2));
                uint32_t span = 1u << (MAX_CODE_LEN - total);
                uint32_t entry = huffPack(2, (uint32_t)total, (uint32_t)s2, (uint32_t)s1);
                for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
            }
        }
    }
    __syncwarp();
    huffBuildEscapePass(cl, codes, lut, lane);
}

// ── Parallel build: pass 1 over symbols, pass 2 over s1, 32 lanes ────
// Canonical codes are prefix-free, so distinct symbols (pass 1) and
// distinct s1 prefixes (pass 2) fill mutually-disjoint LUT spans — the
// 32-lane fan-out is race-free with no atomics.
extern "C" __global__ void huffBuildLutEscParallelKernel(
    const uint8_t* __restrict__ code_lengths_all,
    uint32_t* __restrict__ luts,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;

    __shared__ uint8_t  cl[256];
    __shared__ uint32_t codes[256];
    __shared__ uint32_t bl_count[16];
    __shared__ uint32_t next_code[16];

    const uint8_t* CL = code_lengths_all + (size_t)block_id * 256;
    uint32_t* lut = luts + (size_t)block_id * LUT_SIZE;
    huffBuildSetup(CL, lut, lane, cl, codes, bl_count, next_code);

    for (int s = lane; s < 256; s += 32) {              // pass 1 — over symbols
        int L = cl[s];
        if (L == 0 || L > MAX_CODE_LEN) continue;
        uint32_t aligned = codes[s] << (MAX_CODE_LEN - L);
        uint32_t span = 1u << (MAX_CODE_LEN - L);
        uint32_t entry = huffPack(1, (uint32_t)L, 0, (uint32_t)s);
        for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
    }
    __syncwarp();
    for (int s1 = lane; s1 < 256; s1 += 32) {           // pass 2 — over s1
        int L1 = cl[s1];
        if (L1 == 0 || L1 >= MAX_CODE_LEN) continue;
        uint32_t C1 = codes[s1];
        for (int s2 = 0; s2 < 256; s2++) {
            int L2 = cl[s2];
            if (L2 == 0) continue;
            int total = L1 + L2;
            if (total > MAX_CODE_LEN) continue;
            uint32_t C2 = codes[s2];
            uint32_t aligned = (C1 << (MAX_CODE_LEN - L1))
                             | (C2 << (MAX_CODE_LEN - L1 - L2));
            uint32_t span = 1u << (MAX_CODE_LEN - total);
            uint32_t entry = huffPack(2, (uint32_t)total, (uint32_t)s2, (uint32_t)s1);
            for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
        }
    }
    __syncwarp();
    huffBuildEscapePass(cl, codes, lut, lane);
}

// ══════════════════════════════════════════════════════════════════════
// GPU HUFFMAN TABLE BUILDER  —  STEP 1 of the GPU Huffman encoder
// ══════════════════════════════════════════════════════════════════════
//
// 1 block per Huffman-block, 32 lanes. Histograms the input into shared
// memory (atomicAdd), then lane 0 serially builds canonical code lengths
// and codes. Mirrors the PRODUCTION CPU encoder huffman_encoder.zig
// (buildCodeLengths -> heightLimit(11) -> assignCanonicalCodes) so the
// resulting tables pair with the production decoder.
//
// NOTE: the height limit here is 11 (ENC_MAX_CODE_LEN), independent of the
// decode-side MAX_CODE_LEN=10 used by the LUT kernels above.
//
// Writes code_lengths[256] (u8) + codes[256] (u32) per block, strided by
// `tables_stride` (= 256).
static constexpr int ENC_MAX_CODE_LEN = 11;

extern "C" __global__ void slzHuffBuildTablesKernel(
    const uint8_t* __restrict__ input,
    const HuffBlockDesc* __restrict__ descs,      // in_offset, in_size used
    uint8_t* __restrict__ code_lengths_out,       // tables_stride × n_blocks
    uint32_t* __restrict__ codes_out,             // tables_stride × n_blocks
    uint32_t tables_stride,
    uint32_t n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= n_blocks) return;
    const int lane = threadIdx.x & 31;
    const HuffBlockDesc d = descs[block_id];

    __shared__ uint32_t hist[256];
    __shared__ uint32_t weights[512];   // leaves 0..255, internals 256..510
    __shared__ int      parents[512];
    __shared__ int      active[512];
    __shared__ int      idx[256];
    __shared__ uint8_t  cl[256];
    __shared__ uint32_t codes[256];

    // ── Histogram (32 lanes → shared atomics) ──
    for (int i = lane; i < 256; i += 32) hist[i] = 0;
    __syncwarp();
    const uint8_t* in = input + d.in_offset;
    for (uint32_t i = lane; i < d.in_size; i += 32)
        atomicAdd(&hist[in[i]], 1u);
    __syncwarp();

    // ── Lane 0: serial tree-build + height-limit + canonical codes ──
    if (lane == 0) {
        for (int i = 0; i < 512; i++) { weights[i] = 0; parents[i] = -1; }
        for (int s = 0; s < 256; s++) { cl[s] = 0; codes[s] = 0; }

        int symbols_used = 0;
        for (int s = 0; s < 256; s++) {
            if (hist[s] > 0) { weights[s] = hist[s]; idx[symbols_used++] = s; }
        }

        if (symbols_used == 1) {
            cl[idx[0]] = 1;
        } else if (symbols_used >= 2) {
            int n_active = symbols_used;
            int next_node = 256;
            for (int i = 0; i < n_active; i++) active[i] = idx[i];
            while (n_active > 1) {
                int a_pos = 0, b_pos = 1;
                if (weights[active[a_pos]] > weights[active[b_pos]]) {
                    int t = a_pos; a_pos = b_pos; b_pos = t;
                }
                for (int i = 2; i < n_active; i++) {
                    uint32_t w = weights[active[i]];
                    if (w < weights[active[a_pos]]) { b_pos = a_pos; a_pos = i; }
                    else if (w < weights[active[b_pos]]) { b_pos = i; }
                }
                int a = active[a_pos], b = active[b_pos];
                weights[next_node] = weights[a] + weights[b];
                parents[a] = next_node;
                parents[b] = next_node;
                int new_pos = (a_pos < b_pos) ? a_pos : b_pos;
                int old_pos = (a_pos < b_pos) ? b_pos : a_pos;
                active[new_pos] = next_node;
                active[old_pos] = active[--n_active];
                next_node++;
            }
            for (int i = 0; i < symbols_used; i++) {
                int s = idx[i];
                int depth = 0, n = s;
                while (parents[n] != -1) { depth++; n = parents[n]; }
                cl[s] = (uint8_t)depth;
            }
        }

        // height-limit to ENC_MAX_CODE_LEN (Kraft-preserving, 30-bit fixed point)
        {
            const uint32_t TARGET = 1u << 30;
            const int limit = ENC_MAX_CODE_LEN;
            uint32_t sum = 0;
            for (int s = 0; s < 256; s++)
                if (cl[s] > 0) sum += (1u << (30 - cl[s]));
            for (int s = 0; s < 256; s++) {
                if (cl[s] > limit) {
                    sum -= (1u << (30 - cl[s]));
                    cl[s] = (uint8_t)limit;
                    sum += (1u << (30 - limit));
                }
            }
            while (sum > TARGET) {
                int best = -1;
                uint32_t best_w = 0xFFFFFFFFu;
                for (int s = 0; s < 256; s++) {
                    if (cl[s] > 0 && cl[s] < limit && hist[s] < best_w) {
                        best_w = hist[s]; best = s;
                    }
                }
                if (best < 0) break;
                sum -= (1u << (30 - cl[best]));
                cl[best]++;
                sum += (1u << (30 - cl[best]));
            }
        }

        // canonical code assignment
        {
            uint32_t length_count[ENC_MAX_CODE_LEN + 2] = {0};
            for (int s = 0; s < 256; s++) length_count[cl[s]]++;
            length_count[0] = 0;
            uint32_t next_code[ENC_MAX_CODE_LEN + 2] = {0};
            uint32_t code = 0;
            for (int L = 1; L <= ENC_MAX_CODE_LEN + 1; L++) {
                code = (code + length_count[L - 1]) << 1;
                next_code[L] = code;
            }
            for (int s = 0; s < 256; s++) {
                int L = cl[s];
                codes[s] = (L != 0) ? next_code[L]++ : 0u;
            }
        }
    }
    __syncwarp();

    // ── Write tables ──
    uint8_t*  CLo = code_lengths_out + (size_t)block_id * tables_stride;
    uint32_t* CDo = codes_out        + (size_t)block_id * tables_stride;
    for (int i = lane; i < 256; i += 32) { CLo[i] = cl[i]; CDo[i] = codes[i]; }
}
