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
