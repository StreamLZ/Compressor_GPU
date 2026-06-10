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

// ── LUT-build inner-loop helpers (file-local) ───────────────────────
// Wide-store fill of `span` consecutive u32 entries in shared mem. `span`
// is always a power of two ≥ 1; `base` is span-aligned by construction
// (Pass 1: aligned = code << shift; Pass 2: aligned = (C1<<…) | (C2<<…)
// — both lay down `span` zero LSBs).
__device__ __forceinline__ void fillLutSpan(uint32_t* base, uint32_t entry, uint32_t span) {
    if (span == 1) {
        base[0] = entry;
    } else if (span == 2) {
        const uint64_t e64 = ((uint64_t)entry << 32) | (uint64_t)entry;
        *reinterpret_cast<uint64_t*>(base) = e64;
    } else {
        const uint4 e4 = make_uint4(entry, entry, entry, entry);
        for (uint32_t i = 0; i < span; i += 4) {
            *reinterpret_cast<uint4*>(base + i) = e4;
        }
    }
}

// Pass-2 inner body: given the outer-loop hoisted (s1, aligned_C1, L1, shift1)
// and a packed (L, code, symbol) for s2, write the dual-symbol LUT span.
__device__ __forceinline__ void emitDualPair(
    uint8_t s1, uint32_t aligned_C1, int L1, int shift1, uint32_t pkd_s2,
    uint32_t* lut)
{
    const int L2          = (int)(pkd_s2 >> 19);
    const int total       = L1 + L2;
    const uint32_t C2     = (pkd_s2 >> 8) & 0x7FFu;
    const uint8_t  s2     = (uint8_t)(pkd_s2 & 0xFFu);
    const uint32_t aligned = aligned_C1 | (C2 << (shift1 - L2));
    const uint32_t span    = 1u << (MAX_CODE_LEN - total);
    const uint32_t entry   = packLutEntry(s1, s2, (uint8_t)total, LUT_NUM_SYMS_DUAL);
    fillLutSpan(lut + aligned, entry, span);
}

// ── LUT build kernel ────────────────────────────────────────────────
// Grid: (n_blocks, 1, 1). Block: (WARP_SIZE, 1, 1). One warp per block.
//
// Wire input: HUFF_WEIGHTS_BYTES (128 B) — 256 symbol lengths, two per
// byte. Output: LUT_SIZE (1024) packed entries written to
// luts[desc.lut_offset .. +LUT_SIZE).
//
// Algorithm (built bit-identical to the reference encoder in
// tools/huff_test/huff_ref.c::build_decode_lut_esc, verified across all
// 1525 enwik8 sub-chunks; see tools/huff_test/huff_lut_build_experiments.cu):
//
//   1. Cooperative weights unpack.
//   2. Parallel histogram of code lengths into shared length_count[].
//   3. Lane-0 serial scan (≤11 iters) — len_end[L] (used-list bucket
//      boundaries) and length_count[L] in-place rewritten to next_code[L]
//      (the canonical-code bucket start for length L, RFC 1951).
//   4. Parallel zero of the shared-memory LUT via uint4 stores.
//   5. Fused code-assignment + Pass 1 (single-symbol entries): 8 batches
//      of 32 symbols. Within each batch every lane uses __match_any_sync
//      on its code length to identify peers of the same L, then
//      __popc(lt_mask & match_mask) gives intra-batch offset. The "leader"
//      (lowest lane in a same-L group, found via __ffs) atomicAdds the
//      group size to per_L_base[L] so the next batch sees the new offset
//      base. Every lane writes its packed (L, code, sym) into the
//      length-sorted used_pkd[] AND its single-symbol Pass 1 LUT span.
//   6. Pass 2: dual-symbol entries. Parallel over s1 (lane k handles
//      used_pkd[k], k+32, …). Inner s2 loop runs over the length-bounded
//      prefix used_pkd[0 .. len_end[MAX_CODE_LEN - L1]) — no continue
//      checks, no oob iterations. Inner LDS batched as uint4 (LDS.128),
//      cutting MIO issue rate 4×.
//   7. Pass 3: escape entries for length-(MAX_CODE_LEN+1) codes. Lane 0
//      scans code_lengths[] (256 iters; escape symbols are rare so
//      branch prediction handles the filter cheaply) and recomputes
//      canonical codes by counting predecessors via length_count.
//   8. Coalesced uint4 bulk-dump of the shared LUT to luts[desc.lut_offset]
//      using __stcs (cache-streaming — the LUT consumer kernel runs later
//      and may not land on this SM, so L1 caching here is wasted).
//
// Wins vs the prior dense-iteration build (measured on enwik8 with sub-
// chunk 0.25, ~24K descriptors): 2.61 ms → 0.67 ms = ~3.9× speedup.
// MIO Throttle 3.78 cyc/inst → 0.06; Long Scoreboard 3.27 → 0.85.
//
// Shared budget per block: 256 (code_lengths) + 1024 (used_pkd) +
// 52 (len_end) + 52 (length_count) + 52 (per_L_base) + 4096 (lut)
// ≈ 5.5 KB. On sm_89 (100 KB/SM), 18 blocks/SM resident — below the
// 24 block-count cap but the workload-imbalance of canonical Huffman
// at this scale already caps achieved-occupancy at ~30% regardless.
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
    uint32_t* lut_g = luts + desc.lut_offset;

    __shared__ uint8_t  code_lengths[HUFF_ALPHABET];
    // Length-sorted used-symbol list. Each entry packs (length, code, symbol)
    // into one u32:  bits 0-7 = symbol,  8-18 = canonical code (≤ 11 bits),
    // 19-22 = length (≤ MAX_CODE_LEN). Aligned to 16 B so the Pass 2 inner
    // loop's uint4 LDS doesn't trip an unaligned-access slow path.
    __shared__ __align__(16) uint32_t used_pkd[HUFF_ALPHABET];
    __shared__ int      len_end[HUFF_LEN_HIST_SIZE];
    // length_count[]: holds histogram counts first, then is rewritten in-place
    // to next_code[L] (canonical-code bucket start per length). Reusing the
    // same shared slot saves another HUFF_LEN_HIST_SIZE × 4 B array.
    __shared__ uint32_t length_count[HUFF_LEN_HIST_SIZE];
    // per_L_base[L]: running count of length-L symbols emitted in earlier
    // batches; incremented atomically by the "leader" lane of each same-L
    // group per batch. Drives the canonical-code offset across the 8 batches.
    __shared__ uint32_t per_L_base[HUFF_LEN_HIST_SIZE];
    // Build the LUT in shared memory then bulk-copy to global at the end.
    // Keeps Pass 1 + Pass 2's hot writes on the ~10× faster shared-mem path
    // and prevents global-write throttling (LG Throttle stall) from
    // dominating the kernel.
    __shared__ __align__(16) uint32_t lut[LUT_SIZE];

    // ── 1. Unpack 128 B weights → 256 code lengths. ──
    for (int i = lane; i < HUFF_WEIGHTS_BYTES; i += WARP_SIZE) {
        unpackWeightByte(weights[i], code_lengths[i * 2 + 0], code_lengths[i * 2 + 1]);
    }
    __syncwarp();

    // ── 2. Parallel histogram + reset per_L_base. ──
    for (int i = lane; i < HUFF_LEN_HIST_SIZE; i += WARP_SIZE) {
        length_count[i] = 0;
        per_L_base[i] = 0;
    }
    __syncwarp();
    for (int s = lane; s < HUFF_ALPHABET; s += WARP_SIZE) {
        uint8_t L = code_lengths[s];
        if (L > 0) atomicAdd(&length_count[L], 1u);
    }
    __syncwarp();

    // ── 3. Lane-0 serial scan: len_end + canonical-code bucket starts. ──
    if (lane == 0) {
        // len_end[L] = end index in used_pkd for length-L bucket (used in
        // Pass 2 to bound the inner s2 prefix without a runtime length check).
        int sum = 0;
        len_end[0] = 0;
        for (int L = 1; L <= MAX_CODE_LEN; L++) {
            sum += (int)length_count[L];
            len_end[L] = sum;
        }
        // RFC 1951 canonical-code bucket starts: next_code[L] = first
        // canonical code at length L. Computed via running-shift of cumulative
        // length counts; result overwrites length_count in place.
        uint32_t code = 0;
        uint32_t prev_count = 0;
        uint32_t next_code_arr[HUFF_LEN_HIST_SIZE];
        next_code_arr[0] = 0;
        for (int L = 1; L <= HUFF_MAX_CODE_LEN; L++) {
            code = (code + prev_count) << 1;
            prev_count = length_count[L];
            next_code_arr[L] = code;
        }
        for (int L = 0; L < HUFF_LEN_HIST_SIZE; L++) length_count[L] = next_code_arr[L];
    }
    __syncwarp();

    // ── 4. Zero the shared-mem LUT (parallel uint4 writes). ──
    {
        const uint4 zero4 = make_uint4(0, 0, 0, 0);
        uint4* lut4 = reinterpret_cast<uint4*>(lut);
        const int n_quads = (int)((LUT_SIZE * sizeof(uint32_t)) / sizeof(uint4));
        for (int i = lane; i < n_quads; i += WARP_SIZE) lut4[i] = zero4;
    }

    // ── 5. Fused code assignment + Pass 1 single-symbol fan-out. ──
    // 8 batches × 32 lanes covers all 256 symbols. Each lane handles one
    // symbol per batch: derives its canonical code via __match_any_sync /
    // __popc on the shared length value, writes used_pkd at the correct
    // length-bucket slot, AND writes its Pass 1 single-symbol LUT span.
    // Length-(MAX_CODE_LEN+1) escape symbols defer to Pass 3.
    for (int batch = 0; batch < HUFF_ALPHABET / WARP_SIZE; batch++) {
        const int s = batch * WARP_SIZE + lane;
        const uint8_t L = code_lengths[s];
        const bool used_short  = (L > 0 && L <= MAX_CODE_LEN);
        const bool used_escape = (L == MAX_CODE_LEN + 1);
        const bool used_any    = used_short || used_escape;

        // Within-batch intra-group offset: count of lower-numbered lanes
        // sharing the same L. __match_any_sync returns a bitmask of ALL
        // lanes (incl. self) with matching `L`; popcount of the lt_mask
        // intersection gives the predecessors-with-same-L count.
        const uint32_t match_mask = __match_any_sync(0xFFFFFFFFu, (uint32_t)L);
        const uint32_t lt_mask = (1u << lane) - 1u;
        const uint32_t intra_offset = __popc(match_mask & lt_mask);

        // Cross-batch offset comes from per_L_base[L]; grab it before any
        // lane updates it for this batch's contribution.
        const uint32_t base_for_L = used_any ? per_L_base[L] : 0u;
        const uint32_t my_offset = base_for_L + intra_offset;
        // Group leader = lowest lane with this L. After all lanes have
        // read per_L_base, leaders publish the group size for the next batch.
        const bool is_leader = used_any && (__ffs(match_mask) - 1 == lane);
        __syncwarp();
        if (is_leader) {
            atomicAdd(&per_L_base[L], (uint32_t)__popc(match_mask));
        }

        if (used_short) {
            const uint32_t c = length_count[L] + my_offset;     // canonical code
            const uint32_t pkd = ((uint32_t)L << 19) | (c << 8) | (uint32_t)s;
            const int dst = len_end[L - 1] + (int)my_offset;
            used_pkd[dst] = pkd;
            // Fused Pass 1: single-symbol fan-out span.
            const uint32_t aligned = c << (MAX_CODE_LEN - L);
            const uint32_t span    = 1u << (MAX_CODE_LEN - L);
            const uint32_t entry   = packLutEntry((uint8_t)s, 0, (uint8_t)L, LUT_NUM_SYMS_SINGLE);
            fillLutSpan(lut + aligned, entry, span);
        }
        // Length-(MAX_CODE_LEN+1) symbols are deferred to Pass 3 (rare).
        __syncwarp();
    }
    const int n_used = len_end[MAX_CODE_LEN];

    // ── 6. Pass 2: dual-symbol entries. ──
    // Parallel over s1 in used_pkd (lane k handles k, k+32, …). Inner s2
    // loop runs over the length-bounded prefix [0, s2_end). Inner LDS
    // batched as uint4 (LDS.128) — 4× fewer LDS instructions / 4× lower
    // MIO pressure vs scalar per-iteration loads.
    for (int i1 = lane; i1 < n_used; i1 += WARP_SIZE) {
        const uint32_t p1 = used_pkd[i1];
        const int L1 = (int)(p1 >> 19);
        if (L1 >= MAX_CODE_LEN) continue;   // length-MAX_CODE_LEN never pairs (no L2 ≥ 1 fits)
        const uint32_t C1 = (p1 >> 8) & 0x7FFu;
        const uint8_t  s1 = (uint8_t)(p1 & 0xFFu);
        const int s2_end  = len_end[MAX_CODE_LEN - L1];
        const int shift1  = MAX_CODE_LEN - L1;
        const uint32_t aligned_C1 = C1 << shift1;

        // Bulk 4-at-a-time via uint4 (LDS.128). &used_pkd[i2] is 16-aligned
        // for i2 a multiple of 4 (the array itself is __align__(16)).
        int i2 = 0;
        const int s2_end_4 = s2_end & ~3;
        for (; i2 < s2_end_4; i2 += 4) {
            const uint4 batch = *reinterpret_cast<const uint4*>(&used_pkd[i2]);
            emitDualPair(s1, aligned_C1, L1, shift1, batch.x, lut);
            emitDualPair(s1, aligned_C1, L1, shift1, batch.y, lut);
            emitDualPair(s1, aligned_C1, L1, shift1, batch.z, lut);
            emitDualPair(s1, aligned_C1, L1, shift1, batch.w, lut);
        }
        for (; i2 < s2_end; i2++) {
            emitDualPair(s1, aligned_C1, L1, shift1, used_pkd[i2], lut);
        }
    }
    __syncwarp();

    // ── 7. Pass 3: escape entries for length-(MAX_CODE_LEN+1) codes. ──
    // Length-MAX_CODE_LEN+1 codes can't fit in a MAX_CODE_LEN-bit prefix;
    // they form sibling pairs sharing a MAX_CODE_LEN-bit prefix, and the
    // decoder reads one extra bit. Pass 3 overlays each escape symbol on
    // the existing entry at its prefix slot (Pass 1's single-symbol entry,
    // since Pass 2 never wrote there — total > MAX_CODE_LEN).
    //
    // Escape symbols are rare on natural text (typically 0-4 per block)
    // so the 256-iter lane-0 scan is dominated by branch-predicted skips.
    // Canonical code is recomputed on the fly from length_count[L] +
    // predecessor count (lane 0 sees symbols in ascending order, so the
    // running counter advances by 1 per match).
    if (lane == 0) {
        uint32_t code = length_count[MAX_CODE_LEN + 1];
        for (int s = 0; s < HUFF_ALPHABET; s++) {
            if (code_lengths[s] != MAX_CODE_LEN + 1) continue;
            const uint32_t prefix = code >> 1;
            const uint32_t entry = lut[prefix];
            uint8_t lo_sym = lutSym1(entry);
            uint8_t hi_sym = lutSym2(entry);
            if (code & 1u) hi_sym = (uint8_t)s; else lo_sym = (uint8_t)s;
            lut[prefix] = packLutEntry(lo_sym, hi_sym,
                                       (uint8_t)(MAX_CODE_LEN + 1), LUT_NUM_SYMS_ESCAPE);
            code++;
        }
    }
    __syncwarp();

    // ── 8. Coalesced uint4 bulk-dump shared LUT → global. ──
    // __stcs (STG.CS — cache-streaming / non-temporal): the decoder kernel
    // that consumes this LUT runs separately and may not be scheduled on
    // this SM. L1 caching the bytes here would only pollute the cache.
    {
        const uint4* lut_sh4 = reinterpret_cast<const uint4*>(lut);
        uint4* lut_g4 = reinterpret_cast<uint4*>(lut_g);
        const int n_quads = (int)((LUT_SIZE * sizeof(uint32_t)) / sizeof(uint4));
        for (int i = lane; i < n_quads; i += WARP_SIZE) __stcs(&lut_g4[i], lut_sh4[i]);
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
// A-024 fix: tok_region_off and off16_region_off are passed in as uint64_t
// here instead of being added to desc.out_offset by slzMergeHuffDescsKernel.
// The merge kernel used uint32_t for those region offsets, which overflowed
// at ~6553 sub-chunks (off16 region offset ≈ 4 GB) and silently truncated
// the output address, corrupting earlier slots in entropy_scratch. The
// region offsets are now applied here with explicit uint64_t arithmetic.
// d_compact_counts is the 6-u32 array [n_lit, n_tok, n_hi, n_lo, n_raw,
// n_merged]; each block reads n_lit, n_tok, n_hi to decide which region
// its block_id falls in (the merged descriptor array is laid out
// lit | tok | hi | lo contiguously by slzMergeHuffDescsKernel).
extern "C" __launch_bounds__(32, 8) __global__
void slzHuffDecode4StreamKernel(
    const uint8_t* __restrict__ comp,
    const HuffDecChunkDesc* __restrict__ descs,
    const uint32_t* __restrict__ luts,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ d_n_blocks,
    const uint32_t* __restrict__ d_compact_counts,
    uint64_t tok_region_off,
    uint64_t off16_region_off)
{
    const uint32_t block_id = blockIdx.x;
    if (block_id >= *d_n_blocks) return;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Pick this block's region offset from its merged-array position.
    const uint32_t n_lit = d_compact_counts[0];
    const uint32_t n_tok = d_compact_counts[1];
    const uint32_t n_hi  = d_compact_counts[2];
    uint64_t region_off;
    if (block_id < n_lit) {
        region_off = 0;
    } else if (block_id < n_lit + n_tok) {
        region_off = tok_region_off;
    } else if (block_id < n_lit + n_tok + n_hi) {
        region_off = off16_region_off;
    } else {
        region_off = off16_region_off;
    }

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
    uint8_t* my_out = output + region_off + (uint64_t)desc.out_offset + (uint64_t)my_out_start;

    (void)decodeStreamBoundedIL(
        il_base, my_tail_start, lane, K, my_n_words,
        shared_lut, my_out, my_out_size);
}
