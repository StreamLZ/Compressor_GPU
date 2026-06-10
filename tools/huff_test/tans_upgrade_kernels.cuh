// ==================================================================
//  2026-06-10 upgrade variants: transplant the two production-Huffman
//  BIL learnings that tANS never received (see v4_ideas.md #11):
//    (a) u32 output stores via a 4-byte output interleave
//        dst[(i/4)*128 + lane*4 + (i%4)]  -- one STG per 4 symbols,
//        still fully coalesced across the warp (128B per round).
//    (b) BIL-style word-interleaved compressed layout: lane word w at
//        il_base + w*128 + lane*4 while w < K (coalesced + aligned
//        refill loads), per-lane tails after K. The host transcodes
//        the snapshot; a real v4 wire format would emit this directly.
//  Included at the end of tans_decode_kernel.cu.
// ==================================================================

#define TDEC_REFILL_FWD() \
    if (bitpos < 16) { \
        if (ptr + 4 <= ptr_end) { \
            bits |= readLE32_aligned(ptr) << bitpos; \
            ptr += (31 - bitpos) >> 3; \
            bitpos |= 24; \
        } else { \
            while (bitpos < 24 && ptr < ptr_end) { \
                bits |= (uint32_t)(*ptr) << bitpos; \
                ptr++; \
                bitpos += 8; \
            } \
        } \
    }

#define TDEC_SYM(symvar) { \
    uint32_t packed_ = my_lut[state]; \
    uint32_t bx_ = packed_ >> 24; \
    symvar = (uint8_t)((packed_ >> 16) & 0xFF); \
    uint32_t w_ = packed_ & 0xFFFF; \
    uint32_t x_ = (1u << bx_) - 1u; \
    state = ((bits & x_) + w_) & lut_mask; \
    bits >>= bx_; \
    bitpos -= (int32_t)bx_; \
}

// -- Variant (a) alone: original comp layout, u32-store output --
extern "C" __global__ void __launch_bounds__(64, 4) slzTans32DecodeU32StoreKernel(
    const uint8_t* __restrict__ src_buf,
    uint8_t*       __restrict__ dst2_buf,
    const TansDecChunkDesc* __restrict__ descs,
    uint32_t*      __restrict__ out_status,
    uint32_t       num_chunks,
    TansLutEnt*    __restrict__ lut_buf,
    const TansTableMeta* __restrict__ meta_buf,
    const uint32_t* __restrict__ new_off
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* header = src_buf + descs[chunk_id].src_offset - 128;
    uint32_t dst_size = descs[chunk_id].dst_size;
    uint16_t my_sub_size = (uint16_t)header[lane * 2] | ((uint16_t)header[lane * 2 + 1] << 8);
    uint16_t my_init_state = (uint16_t)header[64 + lane * 2] | ((uint16_t)header[64 + lane * 2 + 1] << 8);

    uint32_t lut_id = chunk_id;
    const uint32_t* packed_lut = (const uint32_t*)((uint8_t*)lut_buf + (uint64_t)lut_id * 2048 * sizeof(TansLutEnt));
    uint32_t log_table_bits = meta_buf[lut_id].log_table_bits;
    uint32_t lut_mask = (1u << log_table_bits) - 1;

    __shared__ uint32_t s_lut_u[2][2048];
    for (int i = lane; i < 2048; i += 32) s_lut_u[warp_id][i] = packed_lut[i];
    __syncwarp();
    const uint32_t* my_lut = s_lut_u[warp_id];

    // Per-stream mode: sub-streams begin AFTER the FSE prob table
    // (src_offset points at the table; the build kernel records the
    // post-table offset in meta). Same source the original variants use.
    const uint8_t* sub_data_start = src_buf + meta_buf[lut_id].src_after_table_off;
    uint32_t my_size32 = (uint32_t)my_sub_size;
    uint32_t prefix_sum = my_size32;
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, prefix_sum, d);
        if (lane >= d) prefix_sum += n;
    }
    const uint8_t* my_src = sub_data_start + (prefix_sum - my_size32);

    uint32_t my_sym_count = dst_size / 32;
    if ((uint32_t)lane < (dst_size % 32)) my_sym_count++;

    uint8_t*  dst8  = dst2_buf + new_off[chunk_id];
    uint32_t* dst32 = (uint32_t*)dst8;

    if (my_sym_count == 0 || my_sub_size < 4) {
        __syncwarp();
        if (lane == 0) out_status[chunk_id] = TANS_OK;
        return;
    }

    uint32_t state = (uint32_t)my_init_state & lut_mask;
    uint32_t bits = readLE32_aligned(my_src);
    int32_t bitpos = 32;
    const uint8_t* ptr = my_src + 4;
    const uint8_t* ptr_end = my_src + my_sub_size;

    uint32_t rounds = my_sym_count >> 2;
    for (uint32_t r = 0; r < rounds; r++) {
        uint32_t accw = 0;
        uint8_t sym;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            TDEC_REFILL_FWD();
            TDEC_SYM(sym);
            accw |= (uint32_t)sym << (k * 8);
        }
        dst32[r * 32 + (uint32_t)lane] = accw;
    }
    for (uint32_t i = rounds * 4; i < my_sym_count; i++) {
        uint8_t sym;
        TDEC_REFILL_FWD();
        TDEC_SYM(sym);
        dst8[(i >> 2) * 128 + (uint32_t)lane * 4 + (i & 3)] = sym;
    }

    __syncwarp();
    if (lane == 0) out_status[chunk_id] = TANS_OK;
}

// -- Variants (a)+(b): BIL word-interleaved comp + u32-store output --
extern "C" __global__ void __launch_bounds__(64, 4) slzTans32DecodeBilKernel(
    const uint8_t* __restrict__ src2_buf,   // transcoded: [128B hdr][il K*128B][tails]
    uint8_t*       __restrict__ dst2_buf,
    const TansDecChunkDesc* __restrict__ descs,
    uint32_t*      __restrict__ out_status,
    uint32_t       num_chunks,
    TansLutEnt*    __restrict__ lut_buf,
    const TansTableMeta* __restrict__ meta_buf,
    const uint32_t* __restrict__ src2_off,  // per-chunk byte offset of the 128B header (128-aligned)
    const uint32_t* __restrict__ d_K,       // per-chunk interleaved word count
    const uint32_t* __restrict__ new_off
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* header = src2_buf + src2_off[chunk_id];
    uint32_t dst_size = descs[chunk_id].dst_size;
    uint16_t my_sub_size = (uint16_t)header[lane * 2] | ((uint16_t)header[lane * 2 + 1] << 8);
    uint16_t my_init_state = (uint16_t)header[64 + lane * 2] | ((uint16_t)header[64 + lane * 2 + 1] << 8);

    uint32_t lut_id = chunk_id;
    const uint32_t* packed_lut = (const uint32_t*)((uint8_t*)lut_buf + (uint64_t)lut_id * 2048 * sizeof(TansLutEnt));
    uint32_t log_table_bits = meta_buf[lut_id].log_table_bits;
    uint32_t lut_mask = (1u << log_table_bits) - 1;

    __shared__ uint32_t s_lut_b[2][2048];
    for (int i = lane; i < 2048; i += 32) s_lut_b[warp_id][i] = packed_lut[i];
    __syncwarp();
    const uint32_t* my_lut = s_lut_b[warp_id];

    uint32_t K = d_K[chunk_id];
    const uint8_t* il_base = header + 128;
    const uint8_t* tails_base = il_base + (uint64_t)K * 128u;

    // Per-lane tail offset: prefix over (size - K*4).
    uint32_t my_tail_len = (uint32_t)my_sub_size - K * 4u;
    uint32_t tp = my_tail_len;
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, tp, d);
        if (lane >= d) tp += n;
    }
    const uint8_t* my_tail = tails_base + (tp - my_tail_len);

    uint32_t my_sym_count = dst_size / 32;
    if ((uint32_t)lane < (dst_size % 32)) my_sym_count++;

    uint8_t*  dst8  = dst2_buf + new_off[chunk_id];
    uint32_t* dst32 = (uint32_t*)dst8;

    if (my_sym_count == 0 || my_sub_size < 4) {
        __syncwarp();
        if (lane == 0) out_status[chunk_id] = TANS_OK;
        return;
    }

    uint32_t my_words = (uint32_t)my_sub_size >> 2;       // floor: full LE words
    uint32_t leftover = (uint32_t)my_sub_size & 3u;       // trailing 1-3 bytes
    const uint8_t* leftover_ptr = my_tail + (my_words - K) * 4u;

    uint64_t bits = 0;
    uint32_t bitcnt = 0;
    uint32_t widx = 0;
    uint32_t lb = 0;

    #define TDEC_REFILL_BIL() \
        if (bitcnt < 16u) { \
            if (widx < K) { \
                uint32_t w_ = *(const uint32_t*)(il_base + (uint64_t)widx * 128u + (uint32_t)lane * 4u); \
                bits |= (uint64_t)w_ << bitcnt; bitcnt += 32u; widx++; \
            } else if (widx < my_words) { \
                uint32_t w_; memcpy(&w_, my_tail + (widx - K) * 4u, 4); \
                bits |= (uint64_t)w_ << bitcnt; bitcnt += 32u; widx++; \
            } else { \
                while (lb < leftover && bitcnt <= 56u) { \
                    bits |= (uint64_t)leftover_ptr[lb] << bitcnt; lb++; bitcnt += 8u; \
                } \
            } \
        }

    #define TDEC_SYM64(symvar) { \
        uint32_t packed_ = my_lut[state]; \
        uint32_t bx_ = packed_ >> 24; \
        symvar = (uint8_t)((packed_ >> 16) & 0xFF); \
        uint32_t w2_ = packed_ & 0xFFFF; \
        uint32_t x_ = (1u << bx_) - 1u; \
        state = (((uint32_t)bits & x_) + w2_) & lut_mask; \
        bits >>= bx_; \
        bitcnt -= bx_; \
    }

    uint32_t state = (uint32_t)my_init_state & lut_mask;

    uint32_t rounds = my_sym_count >> 2;
    for (uint32_t r = 0; r < rounds; r++) {
        uint32_t accw = 0;
        uint8_t sym;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            TDEC_REFILL_BIL();
            TDEC_SYM64(sym);
            accw |= (uint32_t)sym << (k * 8);
        }
        dst32[r * 32 + (uint32_t)lane] = accw;
    }
    for (uint32_t i = rounds * 4; i < my_sym_count; i++) {
        uint8_t sym;
        TDEC_REFILL_BIL();
        TDEC_SYM64(sym);
        dst8[(i >> 2) * 128 + (uint32_t)lane * 4 + (i & 3)] = sym;
    }

    __syncwarp();
    if (lane == 0) out_status[chunk_id] = TANS_OK;
}

// ==================================================================
//  Parallel 4-way LUT spread (v4_ideas #11 table-build item).
//  The production buildPackedLut4Way walks a ~2300-iteration serial
//  recurrence redundantly on all 32 lanes. But the slot recurrence is
//  a pure function of cumulative weights: symbol s (weight w,
//  cum-weight ws) puts y_j = (w + ((ws - j - 1) & 3)) >> 2 slots into
//  quarter j at cursor q_base[j] + prefix-of-prior-symbols(y_j).
//  Five shfl-based exclusive scans (w, y0..y3) plus one for the
//  weight-1 tail make every symbol independent; each lane then fills
//  its own 8 symbols' slots with no shared memory and no syncs.
// ==================================================================

__device__ __forceinline__ uint32_t warpExclScan(uint32_t v, int lane, uint32_t* total) {
    uint32_t incl = v;
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, incl, d);
        if (lane >= d) incl += n;
    }
    *total = __shfl_sync(0xFFFFFFFF, incl, 31);
    return incl - v;
}

__device__ void buildPackedLut4WayParallel(
    const uint16_t* weights, uint32_t log_table_bits, uint32_t* lut, int lane
) {
    const uint32_t L = 1u << log_table_bits;
    const uint32_t L_mask = L - 1;

    // Phase A: per-symbol exclusive prefixes, 8 batches of 32 symbols.
    uint32_t ws_l[8], c0_l[8], c1_l[8], c2_l[8], c3_l[8], on_l[8];
    uint32_t cw = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0, cones = 0;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        uint32_t sym = (uint32_t)b * 32u + (uint32_t)lane;
        uint32_t w = weights[sym];
        uint32_t tot;
        // weights_sum in the serial spread accumulates ONLY w>1 symbols
        // (weight-1 symbols go to the tail and contribute nothing).
        uint32_t w_eff = (w > 1u) ? w : 0u;
        uint32_t ws_x = warpExclScan(w_eff, lane, &tot) + cw;
        cw += tot;
        ws_l[b] = ws_x;
        int32_t wsi = (int32_t)ws_x;
        uint32_t y0 = 0, y1 = 0, y2 = 0, y3 = 0;
        if (w > 1) {
            y0 = (uint32_t)(((int32_t)w + ((wsi - 1) & 3)) >> 2);
            y1 = (uint32_t)(((int32_t)w + ((wsi - 2) & 3)) >> 2);
            y2 = (uint32_t)(((int32_t)w + ((wsi - 3) & 3)) >> 2);
            y3 = (uint32_t)(((int32_t)w + ((wsi - 4) & 3)) >> 2);
        }
        c0_l[b] = warpExclScan(y0, lane, &tot) + c0; c0 += tot;
        c1_l[b] = warpExclScan(y1, lane, &tot) + c1; c1 += tot;
        c2_l[b] = warpExclScan(y2, lane, &tot) + c2; c2 += tot;
        c3_l[b] = warpExclScan(y3, lane, &tot) + c3; c3 += tot;
        uint32_t one = (w == 1u) ? 1u : 0u;
        on_l[b] = warpExclScan(one, lane, &tot) + cones; cones += tot;
    }

    uint32_t slots_left = L - cones;
    uint32_t sa = slots_left >> 2;
    uint32_t qb0 = 0;
    uint32_t qb1 = qb0 + sa + ((slots_left & 3) > 0 ? 1u : 0u);
    uint32_t qb2 = qb1 + sa + ((slots_left & 3) > 1 ? 1u : 0u);
    uint32_t qb3 = qb2 + sa + ((slots_left & 3) > 2 ? 1u : 0u);

    // Phase B: every symbol fills its own slots independently.
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        uint32_t sym = (uint32_t)b * 32u + (uint32_t)lane;
        uint32_t w = weights[sym];
        if (w > 1) {
            uint32_t ws_x = ws_l[b];
            int32_t wsi = (int32_t)ws_x;
            uint32_t wb_lo = ilog2(w);
            uint32_t bps_lo = log_table_bits - wb_lo;
            uint32_t bps_hi = bps_lo - 1;
            uint32_t crossover = (1u << (wb_lo + 1)) - w;
            uint32_t sym_shifted = sym << 16;
            uint32_t cursor = 0;
            uint32_t base[4];
            base[0] = qb0 + c0_l[b];
            base[1] = qb1 + c1_l[b];
            base[2] = qb2 + c2_l[b];
            base[3] = qb3 + c3_l[b];
            #pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t y = (uint32_t)(((int32_t)w + ((wsi - (int32_t)j - 1) & 3)) >> 2);
                for (uint32_t k = 0; k < y; k++) {
                    uint32_t offset = cursor;
                    uint32_t running_w = w + offset;
                    uint32_t bps = (offset < crossover) ? bps_lo : bps_hi;
                    lut[base[j] + k] = (bps << 24) | sym_shifted |
                                       (uint16_t)(L_mask & (running_w << bps));
                    cursor++;
                }
            }
        } else if (w == 1u) {
            lut[slots_left + on_l[b]] = (log_table_bits << 24) | (sym << 16);
        }
    }
}

// FSE build kernel with the parallel spread; signature matches
// slzTansFseBuildKernel so the bench can A/B them directly.
extern "C" __global__ void __launch_bounds__(64) slzTansFseBuildParKernel(
    const uint8_t* __restrict__ src_buf,
    const TansDecChunkDesc* __restrict__ descs,
    TansLutEnt*    __restrict__ lut_buf,
    TansTableMeta* __restrict__ meta_buf,
    uint32_t       num_chunks,
    uint32_t*      __restrict__ work_counter
) {
    const int lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.y;
    const uint32_t warp_global = blockIdx.x * 2 + warp_id;
    const uint32_t total_warps = gridDim.x * 2;

    __shared__ uint16_t s_weights_p[2][256];

    for (uint32_t chunk_id = warp_global; chunk_id < num_chunks; chunk_id += total_warps) {
        uint32_t log_table_bits = 0;
        uint32_t err = TANS_OK;

        if (lane == 0) {
            const uint8_t* src = src_buf + descs[chunk_id].src_offset;
            uint32_t src_size = descs[chunk_id].src_size;
            const uint8_t* after_table = nullptr;
            err = decodeFseWeights(src, src_size, s_weights_p[warp_id],
                                   log_table_bits, after_table);
            if (err == TANS_OK) {
                meta_buf[chunk_id].src_after_table_off =
                    (uint32_t)((uintptr_t)after_table - (uintptr_t)src_buf);
                meta_buf[chunk_id].src_end_off =
                    (uint32_t)((uintptr_t)(src + src_size) - (uintptr_t)src_buf);
                meta_buf[chunk_id].log_table_bits = log_table_bits;
            }
            meta_buf[chunk_id].error = err;
        }
        err = __shfl_sync(0xFFFFFFFF, err, 0);
        if (err != TANS_OK) continue;
        log_table_bits = __shfl_sync(0xFFFFFFFF, log_table_bits, 0);
        __syncwarp();

        uint32_t* glut = (uint32_t*)((uint8_t*)lut_buf +
                         (uint64_t)chunk_id * 2048 * sizeof(TansLutEnt));
        buildPackedLut4WayParallel(s_weights_p[warp_id], log_table_bits, glut, lane);
    }
}
