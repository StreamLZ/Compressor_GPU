// ── StreamLZ GPU Huffman Decode Kernel ──────────────────────────
// Canonical Huffman, code height limited to 11 (`MAX_CODE_LEN + 1`);
// fast-LUT index width `MAX_CODE_LEN = 10`. Double-symbol (X2) LUT
// (4 KB shared), 32-stream parallel decode (all warp lanes active -
// 8× the throughput of a 4-stream design at ~0.1pp ratio cost on
// realistic sub-chunk sizes).
//
// Wire layout for a chunk_type=4 literal stream:
//   [preceding chunk header, consumed by the frame decoder - 3 or 5 bytes]
//   [128 B weights, 4 bits per symbol, packed low-nibble-first]
//   [sub-header: (HUFF_NUM_STREAMS - 1) × u24 LE stream sizes;
//                last stream's size = total payload - sum of stored sizes]
//   [stream 0 bits | stream 1 | ... | stream N-1]
// Bits within each byte are MSB-first.
//
// Two kernels:
//   slzHuffBuildLutKernel  - 1 warp per block, builds a 1024-entry
//                            (`LUT_SIZE`) X2 LUT in global luts[] from
//                            128 B weights.
//   slzHuffDecode4StreamKernel - 1 warp per block, all 32 lanes decode
//                                in parallel (the legacy "4Stream" name
//                                is retained for Zig dispatch ABI).
//
// Built to huffman_kernel.ptx by tools/build_gpu.bat (nvcc);
// decode/driver.zig @embedFile's the PTX.

#include <cstdint>
#include <cstring>                    // memcpy (misaligned u32 refill)
#include "../common/gpu_warp.cuh"     // WARP_SIZE
#include "../common/gpu_byteio.cuh"   // readLE24
#include "../common/gpu_huffman.cuh"  // HUFF_* constants, LUT pack/unpack, buildCanonicalCodes

// MAX_CODE_LEN is the fast-LUT *index width* (10) - an alias of the
// shared HUFF_LUT_INDEX_BITS. Codes are height-limited to
// MAX_CODE_LEN+1 = HUFF_MAX_CODE_LEN = 11; length-11 codes resolve via
// escape LUT entries (num_syms == LUT_NUM_SYMS_ESCAPE) - a 1024-entry
// LUT (4 KB shared) instead of 2048 (8 KB), roughly doubling
// decode-kernel occupancy.
//
// `LUT_SIZE` is `(int)HUFF_LUT_ENTRIES` - the shared header defines the
// value, this file just narrows the type for the local int contexts
// (loop bounds, shared-memory size constants).
static constexpr int MAX_CODE_LEN = HUFF_LUT_INDEX_BITS;     // 10
static constexpr int LUT_SIZE     = (int)HUFF_LUT_ENTRIES;   // 1024

// HUFF_* wire constants, the canonical-code builder, the LUT-entry
// pack/unpack helpers and the num_syms tags come from
// common/gpu_huffman.cuh.

// Bit-buffer arithmetic.
static constexpr int      BITBUF_BITS  = 64;                   // bit_buf width in bits
static constexpr int      REFILL_BITS  = 32;                   // bits per 4-byte refill chunk

// ── Descriptor - matches Zig HuffDecChunkDesc in decode/driver.zig ──
// Unified: in_offset/in_size cover the FULL payload (HUFF_WEIGHTS_BYTES
// weights + HUFF_SUBHEADER_BYTES sub-header + HUFF_NUM_STREAMS stream
// payloads, i.e. HUFF_BODY_HEADER_BYTES fixed + sum(stream_sizes)).
// Build kernel reads the leading HUFF_WEIGHTS_BYTES; decode kernel skips
// them and works on the remainder.
struct HuffDecChunkDesc {
    uint32_t in_offset;     // byte offset in compressed buffer (points at weights)
    uint32_t in_size;       // FULL payload bytes (HUFF_BODY_HEADER_BYTES + sum(streams))
    uint32_t out_offset;    // byte offset in literal scratch
    uint32_t out_size;      // expected decompressed bytes
    uint32_t lut_offset;    // entry index into luts[] (LUT_SIZE entries per block)
};
static_assert(sizeof(HuffDecChunkDesc) == 20, "ABI: keep in sync with decode/driver.zig");

// LUT entry packing/unpacking (packLutEntry, lutSym1/lutSym2/lutSymPair/
// lutTotalLen/lutNumSyms) and the LUT_NUM_SYMS_* / LUT_MAX_SYMS_PER_STEP
// constants come from common/gpu_huffman.cuh.

// ── LUT build kernel ────────────────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (WARP_SIZE, 1, 1).
// One warp per block. Weights[] is HUFF_WEIGHTS_BYTES bytes; each byte
// has two nibbles - low nibble = symbol 2*i, high nibble = symbol 2*i+1.
// Pass 1: single-symbol fan-out (32-lane parallel over symbols).
// Pass 2: dual-symbol overwrite (32-lane parallel over s1; inner loop
// bounded by L1+L2<=10). Prefix-free codes give disjoint spans, so both
// passes are race-free with no atomics. Pass 3 escape stays lane-0.
//
// LUT is written to global memory (luts + desc.lut_offset).
extern "C" __global__ void slzHuffBuildLutKernel(
    const uint8_t* __restrict__ comp,
    const HuffDecChunkDesc* __restrict__ descs,
    uint32_t* __restrict__ luts,
    const uint32_t* __restrict__ d_n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= *d_n_blocks) return;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const HuffDecChunkDesc desc = descs[block_id];
    const uint8_t* weights = comp + desc.in_offset;
    uint32_t* lut = luts + desc.lut_offset;

    // Unpack 128 B → 256 lengths into shared memory (one cache line of accesses).
    __shared__ uint8_t code_lengths[HUFF_ALPHABET];
    for (int i = lane; i < HUFF_WEIGHTS_BYTES; i += WARP_SIZE) {
        unpackWeightByte(weights[i], code_lengths[i * 2 + 0], code_lengths[i * 2 + 1]);
    }
    __syncwarp();

    // Compute canonical codes (RFC 1951 style - assigned in length-ascending
    // order, sym-ascending within each length). buildCanonicalCodes
    // (common/gpu_huffman.cuh) does the histogram + next_code + code
    // assignment serially in lane 0; all lanes then read the same codes[]
    // table for their slice of the fan-out work.
    __shared__ uint32_t codes[HUFF_ALPHABET];
    if (lane == 0) buildCanonicalCodes(code_lengths, codes);
    __syncwarp();

    // ── Build LUT in global memory ──
    // Zero the LUT - cooperative across warp.
    for (int i = lane; i < LUT_SIZE; i += WARP_SIZE) lut[i] = 0;
    __syncwarp();

    // Pass 1: single-symbol entries - parallel over symbols (32 lanes).
    // Canonical codes are prefix-free, so distinct symbols fill mutually
    // disjoint LUT spans - the fan-out is race-free with no atomics.
    for (int s = lane; s < HUFF_ALPHABET; s += WARP_SIZE) {
        int L = code_lengths[s];
        if (L == 0 || L > MAX_CODE_LEN) continue;  // length-11 -> escape pass
        uint32_t aligned = codes[s] << (MAX_CODE_LEN - L);
        uint32_t span = 1u << (MAX_CODE_LEN - L);
        uint32_t entry = packLutEntry((uint8_t)s, 0, (uint8_t)L, LUT_NUM_SYMS_SINGLE);
        for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
    }
    __syncwarp();

    // Pass 2: dual-symbol entries - parallel over s1 (32 lanes). Distinct
    // s1 prefixes fill disjoint spans, so the fan-out stays race-free.
    for (int s1 = lane; s1 < HUFF_ALPHABET; s1 += WARP_SIZE) {
        int L1 = code_lengths[s1];
        if (L1 == 0 || L1 >= MAX_CODE_LEN) continue;
        uint32_t C1 = codes[s1];
        for (int s2 = 0; s2 < HUFF_ALPHABET; s2++) {
            int L2 = code_lengths[s2];
            if (L2 == 0) continue;
            int total = L1 + L2;
            if (total > MAX_CODE_LEN) continue;
            uint32_t C2 = codes[s2];
            uint32_t aligned = (C1 << (MAX_CODE_LEN - L1))
                             | (C2 << (MAX_CODE_LEN - L1 - L2));
            uint32_t span = 1u << (MAX_CODE_LEN - total);
            uint32_t entry = packLutEntry((uint8_t)s1, (uint8_t)s2,
                                           (uint8_t)total, LUT_NUM_SYMS_DUAL);
            for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
        }
    }
    __syncwarp();

    // Pass 3: escape entries for length-(MAX_CODE_LEN+1) codes. A 10-bit
    // prefix can't resolve an 11-bit code; the entry stores both sibling
    // symbols (num_syms == LUT_NUM_SYMS_ESCAPE) and the decoder reads the
    // 11th bit. These 10-bit prefixes are never touched by passes 1-2
    // (prefix-free).
    if (lane == 0) {
        for (int s = 0; s < HUFF_ALPHABET; s++) {
            if (code_lengths[s] != MAX_CODE_LEN + 1) continue;
            uint32_t code = codes[s];
            uint32_t prefix = code >> 1;               // 10-bit prefix = LUT index
            uint32_t entry = lut[prefix];
            uint8_t lo_sym = lutSym1(entry);
            uint8_t hi_sym = lutSym2(entry);
            if (code & 1u) hi_sym = (uint8_t)s; else lo_sym = (uint8_t)s;
            lut[prefix] = packLutEntry(lo_sym, hi_sym,
                                       (uint8_t)(MAX_CODE_LEN + 1), LUT_NUM_SYMS_ESCAPE);
        }
    }
}

// ── Single-stream decode core (one lane) — BIL refill ───────────────
// Top-aligned bit buffer; 4-byte big-endian refill; double-symbol (X2)
// LUT lookup. Refill source depends on the current word index:
//   word_idx < K  → interleaved area (32-lane coalesced sector load)
//   word_idx ≥ K  → per-lane tail (scattered, like a concat layout)
//
// Two-phase design:
//   Phase 1 (preamble): byte-drain until `out + written` is 4-aligned.
//                       Runs at most ~3 byte stores per lane.
//   Phase 2 (hot loop): 2 LUT lookups per refill check, unconditional
//                       u32 stores (alignment invariant from phase 1).
//
// Phase 2's refill threshold is 2*(MAX_CODE_LEN+1) = 22 bits. After
// refill, bit_count is in [32, 53] — strictly ≥ 22 — so two decodes
// always succeed without a mid-batch refill check, even when both are
// length-11 escape codes. 3-lookup would need ≤ 33 bits which exceeds
// the 32 bits one refill can add, forcing fallback paths.
//
// All refills go through one 4-byte load. In the BIL hot zone
// (word_idx < K) all 32 lanes' loads are at `il_base + w*128 + lane*4`
// — adjacent 4-byte slots in the same 128-byte row, coalesced into one
// L2 sector. Past K the lanes' tails sit at independent offsets and
// the loads scatter (~11% of refills on natural text per measurement).
__device__ __forceinline__ uint32_t decodeStreamBoundedIL(
    const uint8_t* __restrict__ il_base,     // start of interleaved area
    const uint8_t* __restrict__ tail_start,  // this lane's tail start
    int            lane,                     // 0..HUFF_NUM_STREAMS-1
    uint32_t       K,                        // interleaved word count
    uint32_t       my_n_words,               // this lane's total word count
    const uint32_t* __restrict__ lut,
    uint8_t* __restrict__ out, uint32_t out_size)
{
    uint64_t bit_buf = 0;
    uint32_t bit_count = 0;
    uint32_t word_idx = 0;       // next BIL word index to refill from
    uint32_t out_pos = 0;        // total bytes decoded
    uint32_t written = 0;        // bytes flushed to global
    uint64_t acc = 0;            // pending decoded bytes (byte 0 = oldest)
    uint32_t pending = 0;

    // ── Phase 1: preamble — byte-drain until `out + written` is 4-aligned ──
    //
    // Phase 2's hot loop does an UNCONDITIONAL u32 store that faults on
    // a misaligned address. Each lane's output starts at
    // `out_offset + lane * (out_size / N)`; neither term is guaranteed
    // 4-aligned, so the destination's starting alignment is arbitrary.
    // We byte-drain until aligned ONCE here; phase 2 then stays aligned
    // because every u32 store advances `written` by exactly 4.
    //
    // Cap: at most ~3 iterations — each decode emits 1-2 bytes and we
    // exit as soon as alignment is reached.
    while (out_pos < out_size && ((uintptr_t)(out + written) & (alignof(uint32_t) - 1)) != 0u) {
        if (bit_count < MAX_CODE_LEN + 1) {
            if (word_idx >= my_n_words) break;
            const uint8_t* w_ptr = (word_idx < K)
                ? (il_base + word_idx * (uint32_t)HUFF_BIL_ROW_BYTES + (uint32_t)lane * 4u)
                : (tail_start + (word_idx - K) * 4u);
            // Misalignment-safe refill: each entropy body sits at file
            // offset = chunk_offset + 5 (after the type-4 chunk header),
            // so `w_ptr` is 4-aligned only ~25% of the time. A direct
            // u32 deref traps with CUDA_ERROR_MISALIGNED_ADDRESS the
            // first time we hit a body whose chunk_offset isn't 3 mod 4.
            // memcpy is the canonical alignment-agnostic load; nvcc lowers
            // it to byte/short loads + shifts when alignment isn't provable.
            // All three other refill sites in this function use the same
            // memcpy pattern and reference this comment.
            uint32_t word;
            memcpy(&word, w_ptr, sizeof(uint32_t));
            uint32_t v = __byte_perm(word, 0, 0x0123);   // LE u32 → BE bit-stream order
            bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);
            bit_count += REFILL_BITS;
            word_idx++;
        }
        uint32_t entry = lut[(uint32_t)(bit_buf >> (BITBUF_BITS - MAX_CODE_LEN))];
        int total_len = lutTotalLen(entry);
        int num_syms  = lutNumSyms(entry);
        if (total_len == 0) return 0;
        if (num_syms == LUT_NUM_SYMS_ESCAPE) {
            uint32_t b11 = (uint32_t)((bit_buf >> (BITBUF_BITS - 1 - MAX_CODE_LEN)) & 1u);
            uint32_t sym = b11 ? lutSym2(entry) : lutSym1(entry);
            acc |= (uint64_t)sym << (pending * 8);
            pending += 1;
            out_pos += 1;
        } else {
            acc |= (uint64_t)lutSymPair(entry) << (pending * 8);
            pending += num_syms;
            out_pos += num_syms;
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
        // Drain pending byte-by-byte UNTIL we're aligned - then exit the
        // outer preamble loop and let phase 2 take over with u32 stores.
        while (pending > 0 && ((uintptr_t)(out + written) & (alignof(uint32_t) - 1)) != 0u) {
            out[written++] = (uint8_t)(acc & 0xFF);
            acc >>= 8;
            pending -= 1;
        }
    }

    // ── Phase 2a: hot loop, INTERLEAVED area (word_idx < K) ──
    //
    // 2 LUT lookups/refill, unconditional u32 store. Loads are at
    // `il_base + w*128 + lane*4` — all 32 warp lanes' loads in the same
    // iteration coalesce into one 128-byte L2 sector access. The
    // refill-threshold analysis matches the prior 32-stream design:
    // 2 × (MAX_CODE_LEN + 1) = 22 bits worst case per batch; post-refill
    // bit_count ∈ [32, 53] is strictly ≥ 22. 3-lookup would need ≤ 33
    // bits but a single 32-bit refill can only add 32, so it would force
    // fallback paths and isn't done.
    static_assert(LUT_MAX_SYMS_PER_STEP == 2,
                  "phase-2 hot loop is hard-wired to a 2-lookup batch; "
                  "see refill-threshold analysis above");
    while (out_pos + 2 * LUT_MAX_SYMS_PER_STEP <= out_size && word_idx < K) {
        if (bit_count < 2 * (MAX_CODE_LEN + 1)) {
            const uint8_t* w_ptr = il_base
                + word_idx * (uint32_t)HUFF_BIL_ROW_BYTES
                + (uint32_t)lane * 4u;
            uint32_t word;
            memcpy(&word, w_ptr, sizeof(uint32_t));  // misalignment-safe refill — see phase-1 comment
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);
            bit_count += REFILL_BITS;
            word_idx++;
        }
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            uint32_t entry = lut[(uint32_t)(bit_buf >> (BITBUF_BITS - MAX_CODE_LEN))];
            int total_len = lutTotalLen(entry);
            int num_syms  = lutNumSyms(entry);
            if (total_len == 0) return 0;
            if (num_syms == LUT_NUM_SYMS_ESCAPE) {
                uint32_t b11 = (uint32_t)((bit_buf >> (BITBUF_BITS - 1 - MAX_CODE_LEN)) & 1u);
                uint32_t sym = b11 ? lutSym2(entry) : lutSym1(entry);
                acc |= (uint64_t)sym << (pending * 8);
                pending += 1;
                out_pos += 1;
            } else {
                acc |= (uint64_t)lutSymPair(entry) << (pending * 8);
                pending += num_syms;
                out_pos += num_syms;
            }
            bit_buf <<= total_len;
            bit_count -= total_len;
            if (pending >= sizeof(uint32_t)) {
                *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
                acc >>= 32;
                written += (uint32_t)sizeof(uint32_t);
                pending -= (uint32_t)sizeof(uint32_t);
            }
        }
    }

    // ── Phase 2b: hot loop, TAIL area (word_idx ≥ K) ──
    //
    // Same 2-lookup body as phase 2a; only the refill source changes.
    // Tail loads scatter across lanes (different streams, different
    // tail offsets) and don't coalesce — but tails are short (typically
    // ~10% of total refills on natural text), so the scatter cost is
    // bounded.
    while (out_pos + 2 * LUT_MAX_SYMS_PER_STEP <= out_size) {
        if (bit_count < 2 * (MAX_CODE_LEN + 1)) {
            if (word_idx >= my_n_words) break;
            const uint8_t* w_ptr = tail_start + (word_idx - K) * 4u;
            uint32_t word;
            memcpy(&word, w_ptr, sizeof(uint32_t));  // misalignment-safe refill — see phase-1 comment
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);
            bit_count += REFILL_BITS;
            word_idx++;
        }
        #pragma unroll
        for (int k = 0; k < 2; k++) {
            uint32_t entry = lut[(uint32_t)(bit_buf >> (BITBUF_BITS - MAX_CODE_LEN))];
            int total_len = lutTotalLen(entry);
            int num_syms  = lutNumSyms(entry);
            if (total_len == 0) return 0;
            if (num_syms == LUT_NUM_SYMS_ESCAPE) {
                uint32_t b11 = (uint32_t)((bit_buf >> (BITBUF_BITS - 1 - MAX_CODE_LEN)) & 1u);
                uint32_t sym = b11 ? lutSym2(entry) : lutSym1(entry);
                acc |= (uint64_t)sym << (pending * 8);
                pending += 1;
                out_pos += 1;
            } else {
                acc |= (uint64_t)lutSymPair(entry) << (pending * 8);
                pending += num_syms;
                out_pos += num_syms;
            }
            bit_buf <<= total_len;
            bit_count -= total_len;
            if (pending >= sizeof(uint32_t)) {
                *reinterpret_cast<uint32_t*>(out + written) = (uint32_t)acc;
                acc >>= 32;
                written += (uint32_t)sizeof(uint32_t);
                pending -= (uint32_t)sizeof(uint32_t);
            }
        }
    }

    // Drain pending bytes; `written` catches up to `out_pos`.
    while (pending > 0) {
        out[written++] = (uint8_t)(acc & 0xFF);
        acc >>= 8;
        pending -= 1;
    }

    // ── Phase 3: tail — single-symbol decode, dual-source refill ──
    // Handles the trailing 1-2 bytes the X2 hot loop leaves; clamps
    // dual entries to single. No byte-at-a-time refill fallback: BIL
    // always reads full 4-byte words, and the encoder zero-pads each
    // stream's trailing partial word, so refill is always a clean 4-byte
    // load (no need to scrape a final 1-3 bytes from input).
    while (out_pos < out_size) {
        if (bit_count <= REFILL_BITS && word_idx < my_n_words) {
            const uint8_t* w_ptr = (word_idx < K)
                ? (il_base + word_idx * (uint32_t)HUFF_BIL_ROW_BYTES + (uint32_t)lane * 4u)
                : (tail_start + (word_idx - K) * 4u);
            uint32_t word;
            memcpy(&word, w_ptr, sizeof(uint32_t));  // misalignment-safe refill — see phase-1 comment
            uint32_t v = __byte_perm(word, 0, 0x0123);
            bit_buf |= ((uint64_t)v) << (REFILL_BITS - bit_count);
            bit_count += REFILL_BITS;
            word_idx++;
        }
        // Out-of-input clamp: if we drained the bit-stream mid-symbol,
        // force bit_count up so the LUT shift below stays well-defined.
        // Stream-padding zeros may produce a "fictitious" decode here,
        // but the surrounding `out_pos < out_size` gate clips writes
        // to valid output slots — and the `total_len == 0` check below
        // still catches an actually-corrupt LUT entry.
        if (bit_count < MAX_CODE_LEN + 1) bit_count = MAX_CODE_LEN + 1;
        uint32_t entry = lut[(uint32_t)(bit_buf >> (BITBUF_BITS - MAX_CODE_LEN))];
        int total_len = lutTotalLen(entry);
        int num_syms  = lutNumSyms(entry);
        if (total_len == 0) return 0;
        if (num_syms == LUT_NUM_SYMS_ESCAPE) {
            uint32_t b11 = (uint32_t)((bit_buf >> (BITBUF_BITS - 1 - MAX_CODE_LEN)) & 1u);
            out[out_pos++] = b11 ? lutSym2(entry) : lutSym1(entry);
        } else {
            out[out_pos++] = lutSym1(entry);
            if (num_syms == LUT_NUM_SYMS_DUAL && out_pos < out_size) {
                out[out_pos++] = lutSym2(entry);
            }
        }
        bit_buf <<= total_len;
        bit_count -= total_len;
    }
    return out_pos;
}

// ── 32-stream BIL decode kernel ─────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (WARP_SIZE, 1, 1) — one warp per block.
// All 32 lanes decode in parallel (one stream each). Coalesced refills
// in the interleaved area; scatter refills in the tail (~11% of words).
// Layout (after the 128 B weights, skipped via in_offset+HUFF_WEIGHTS_BYTES):
//   [HUFF_SUBHEADER_BYTES — N × u24 LE per-stream byte sizes]
//   [HUFF_BIL_K_BYTES — u32 LE interleaved word count K]
//   [interleaved area — K × HUFF_BIL_ROW_BYTES bytes]
//   [tail area — per-stream tails at exclusive-prefix-sum offsets]
//
// `__launch_bounds__(32, 8)` tells nvcc the exact block size (32 threads)
// and to leave register-budget headroom for at least 8 blocks per SM.
// Codegen-specialization win (~+0.5-1% measured); we're shared-mem-
// limited on occupancy at typically 18-24 blocks/SM, well above the 8
// floor, so the directive doesn't constrain runtime occupancy.
//
// Name retained (slzHuffDecode4StreamKernel) for ABI compatibility with
// the existing Zig dispatch — it now decodes HUFF_NUM_STREAMS streams
// in the BIL format, not literally 4 streams concatenated.
extern "C" __launch_bounds__(32, 8) __global__
void slzHuffDecode4StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffDecChunkDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ d_n_blocks)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= *d_n_blocks) return;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    const HuffDecChunkDesc desc = descs[block_id];

    // Cooperative LUT load into shared memory (LUT_SIZE × 4 = 1024 × 4 = 4 KB).
    extern __shared__ uint32_t shared_lut[];
    const uint32_t* src_lut = luts + desc.lut_offset;
    for (int i = lane; i < LUT_SIZE; i += WARP_SIZE) shared_lut[i] = src_lut[i];
    __syncwarp();

    // BIL header parse. Every lane reads its OWN size (all N sizes are
    // stored explicitly now, no derived-from-total). Then read K, which
    // is a block-uniform u32 — every lane reads it (cheap; cached).
    const uint8_t* hdr = comp + desc.in_offset + HUFF_WEIGHTS_BYTES;
    uint32_t my_size = readLE24(hdr + lane * HUFF_STREAM_SIZE_BYTES);

    // K immediately after the sub-header (u32 LE).
    const uint8_t* k_ptr = hdr + HUFF_SUBHEADER_BYTES;
    uint32_t K = ((uint32_t)k_ptr[0])
               | ((uint32_t)k_ptr[1] << 8)
               | ((uint32_t)k_ptr[2] << 16)
               | ((uint32_t)k_ptr[3] << 24);

    // Body bases. il_base is the start of the interleaved area; tails
    // follow immediately after.
    const uint8_t* il_base = hdr + HUFF_SUBHEADER_BYTES + HUFF_BIL_K_BYTES;
    const uint8_t* tail_area = il_base + (size_t)K * (uint32_t)HUFF_BIL_ROW_BYTES;

    // Per-lane word count and tail-bytes count.
    uint32_t my_n_words = (my_size + 3u) / 4u;
    uint32_t my_tail_bytes = (my_n_words > K) ? (my_n_words - K) * 4u : 0u;

    // Exclusive prefix sum of per-lane tail bytes → this lane's tail offset.
    uint32_t tail_off = my_tail_bytes;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        uint32_t v = __shfl_up_sync(0xFFFFFFFFu, tail_off, off);
        if (lane >= off) tail_off += v;
    }
    tail_off -= my_tail_bytes;  // exclusive
    const uint8_t* my_tail_start = tail_area + tail_off;

    // Per-lane output region: 32 contiguous slices. Lane (N-1) absorbs
    // the remainder when out_size isn't divisible by HUFF_NUM_STREAMS.
    uint32_t q = desc.out_size / HUFF_NUM_STREAMS;
    uint32_t my_out_start = (lane < HUFF_NUM_STREAMS - 1)
                          ? (lane * q)
                          : ((HUFF_NUM_STREAMS - 1) * q);
    uint32_t my_out_size  = (lane < HUFF_NUM_STREAMS - 1)
                          ? q
                          : (desc.out_size - (HUFF_NUM_STREAMS - 1) * q);
    uint8_t* my_out = output + desc.out_offset + my_out_start;

    (void)decodeStreamBoundedIL(
        il_base, my_tail_start, lane, K, my_n_words,
        shared_lut, my_out, my_out_size);
}
