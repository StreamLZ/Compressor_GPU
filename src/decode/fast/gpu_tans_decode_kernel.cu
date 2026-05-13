// ── StreamLZ GPU tANS Decode Kernel ─────────────────────────────
// 5-state interleaved tANS (tabled Asymmetric Numeral System) decoder.
// Each CUDA block decodes one tANS-compressed chunk.
// Lane 0 does all serial decode work; other lanes are idle (tANS is
// inherently serial — the next state depends on the previous symbol).
//
// Three phases per chunk:
//   1. decodeTable  — parse Golomb-Rice or sparse/explicit frequency table
//   2. initLut      — build the L-entry decode LUT in shared memory
//   3. decode5State — 5-state forward/backward interleaved hot loop
//
// Build: nvcc -cubin -arch=sm_89 -O3 gpu_tans_decode_kernel.cu
// Embed: @embedFile("gpu_tans_decode_kernel.cubin") in gpu_driver.zig

#include <cstdint>

// ── Chunk descriptor ──────────────────────────────────────────────
struct TansDecChunkDesc {
    uint32_t src_offset;   // offset into compressed buffer
    uint32_t src_size;     // compressed size in bytes
    uint32_t dst_offset;   // offset into output buffer
    uint32_t dst_size;     // expected decompressed size in bytes
};

// ── LUT entry — 8 bytes (used for table build, matches CPU TansLutEnt)
struct TansLutEnt {
    uint32_t x;       // mask for extracting next-state bits
    uint8_t  bits_x;  // number of bits consumed per symbol
    uint8_t  symbol;  // decoded symbol byte
    uint16_t w;       // weight offset for next-state computation
};

// ── Packed LUT entry — 4 bytes for decode hot loop ────────────────
// x = (1 << bits_x) - 1, derived at decode time. Halves LUT bandwidth.
__device__ __forceinline__ uint32_t packLutEntry(const TansLutEnt& e) {
    return ((uint32_t)e.bits_x << 24) | ((uint32_t)e.symbol << 16) | (uint32_t)e.w;
}

__device__ __forceinline__ uint8_t decodeOneSymbolPacked(
    const uint32_t* lut, uint32_t& state, uint64_t& bits,
    int32_t& bitpos, uint32_t lut_mask
) {
    uint32_t packed = __ldg(&lut[state]);
    uint32_t bits_x = packed >> 24;
    uint8_t sym = (uint8_t)((packed >> 16) & 0xFF);
    uint32_t w = packed & 0xFFFF;
    uint32_t x = (1u << bits_x) - 1;
    bitpos -= (int32_t)bits_x;
    state = ((uint32_t)(bits & (uint64_t)x) + w) & lut_mask;
    bits >>= bits_x;
    return sym;
}

// ── Error codes written to out_status ─────────────────────────────
static constexpr uint32_t TANS_OK                  = 0;
static constexpr uint32_t TANS_ERR_SRC_TRUNCATED   = 1;
static constexpr uint32_t TANS_ERR_BAD_TABLE       = 2;
static constexpr uint32_t TANS_ERR_BAD_LOG_BITS    = 3;
static constexpr uint32_t TANS_ERR_BAD_WEIGHTS     = 4;
static constexpr uint32_t TANS_ERR_LUT_FAILED      = 5;
static constexpr uint32_t TANS_ERR_STREAM_MISMATCH = 6;
static constexpr uint32_t TANS_ERR_STATE_RANGE     = 7;
static constexpr uint32_t TANS_ERR_DST_OVERFLOW    = 8;

// Maximum LUT size: 2^12 = 4096 entries (log_table_bits max 12).
// At 8 bytes each that is 32KB — fits in shared memory on all modern GPUs.
// Typical: 2^8..2^11 = 256..2048 entries = 2..16KB.
static constexpr uint32_t MAX_LUT_ENTRIES = 4096;

// ── Golomb-Rice decode tables (from CPU bit_reader_lite) ──────────
// Pre-computed value and length tables for Rice parameter k=2.
__device__ static const uint32_t k_rice_value[256] = {
    0x80000000, 0x00000007, 0x10000006, 0x00000006, 0x20000005, 0x00000105, 0x10000005, 0x00000005,
    0x30000004, 0x00000204, 0x10000104, 0x00000104, 0x20000004, 0x00010004, 0x10000004, 0x00000004,
    0x40000003, 0x00000303, 0x10000203, 0x00000203, 0x20000103, 0x00010103, 0x10000103, 0x00000103,
    0x30000003, 0x00020003, 0x10010003, 0x00010003, 0x20000003, 0x01000003, 0x10000003, 0x00000003,
    0x50000002, 0x00000402, 0x10000302, 0x00000302, 0x20000202, 0x00010202, 0x10000202, 0x00000202,
    0x30000102, 0x00020102, 0x10010102, 0x00010102, 0x20000102, 0x01000102, 0x10000102, 0x00000102,
    0x40000002, 0x00030002, 0x10020002, 0x00020002, 0x20010002, 0x01010002, 0x10010002, 0x00010002,
    0x30000002, 0x02000002, 0x11000002, 0x01000002, 0x20000002, 0x00000012, 0x10000002, 0x00000002,
    0x60000001, 0x00000501, 0x10000401, 0x00000401, 0x20000301, 0x00010301, 0x10000301, 0x00000301,
    0x30000201, 0x00020201, 0x10010201, 0x00010201, 0x20000201, 0x01000201, 0x10000201, 0x00000201,
    0x40000101, 0x00030101, 0x10020101, 0x00020101, 0x20010101, 0x01010101, 0x10010101, 0x00010101,
    0x30000101, 0x02000101, 0x11000101, 0x01000101, 0x20000101, 0x00000111, 0x10000101, 0x00000101,
    0x50000001, 0x00040001, 0x10030001, 0x00030001, 0x20020001, 0x01020001, 0x10020001, 0x00020001,
    0x30010001, 0x02010001, 0x11010001, 0x01010001, 0x20010001, 0x00010011, 0x10010001, 0x00010001,
    0x40000001, 0x03000001, 0x12000001, 0x02000001, 0x21000001, 0x01000011, 0x11000001, 0x01000001,
    0x30000001, 0x00000021, 0x10000011, 0x00000011, 0x20000001, 0x00001001, 0x10000001, 0x00000001,
    0x70000000, 0x00000600, 0x10000500, 0x00000500, 0x20000400, 0x00010400, 0x10000400, 0x00000400,
    0x30000300, 0x00020300, 0x10010300, 0x00010300, 0x20000300, 0x01000300, 0x10000300, 0x00000300,
    0x40000200, 0x00030200, 0x10020200, 0x00020200, 0x20010200, 0x01010200, 0x10010200, 0x00010200,
    0x30000200, 0x02000200, 0x11000200, 0x01000200, 0x20000200, 0x00000210, 0x10000200, 0x00000200,
    0x50000100, 0x00040100, 0x10030100, 0x00030100, 0x20020100, 0x01020100, 0x10020100, 0x00020100,
    0x30010100, 0x02010100, 0x11010100, 0x01010100, 0x20010100, 0x00010110, 0x10010100, 0x00010100,
    0x40000100, 0x03000100, 0x12000100, 0x02000100, 0x21000100, 0x01000110, 0x11000100, 0x01000100,
    0x30000100, 0x00000120, 0x10000110, 0x00000110, 0x20000100, 0x00001100, 0x10000100, 0x00000100,
    0x60000000, 0x00050000, 0x10040000, 0x00040000, 0x20030000, 0x01030000, 0x10030000, 0x00030000,
    0x30020000, 0x02020000, 0x11020000, 0x01020000, 0x20020000, 0x00020010, 0x10020000, 0x00020000,
    0x40010000, 0x03010000, 0x12010000, 0x02010000, 0x21010000, 0x01010010, 0x11010000, 0x01010000,
    0x30010000, 0x00010020, 0x10010010, 0x00010010, 0x20010000, 0x00011000, 0x10010000, 0x00010000,
    0x50000000, 0x04000000, 0x13000000, 0x03000000, 0x22000000, 0x02000010, 0x12000000, 0x02000000,
    0x31000000, 0x01000020, 0x11000010, 0x01000010, 0x21000000, 0x01001000, 0x11000000, 0x01000000,
    0x40000000, 0x00000030, 0x10000020, 0x00000020, 0x20000010, 0x00001010, 0x10000010, 0x00000010,
    0x30000000, 0x00002000, 0x10001000, 0x00001000, 0x20000000, 0x00100000, 0x10000000, 0x00000000,
};

__device__ static const uint8_t k_rice_len[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

// ══════════════════════════════════════════════════════════════════
//  MSB-first bit reader for table headers (mirrors CPU BitReaderState)
// ══════════════════════════════════════════════════════════════════

struct BitReader {
    const uint8_t* p;
    const uint8_t* p_end;
    uint32_t bits;
    int32_t  bit_pos;
};

__device__ void brRefill(BitReader& br) {
    while (br.bit_pos > 0) {
        uint32_t byte = (br.p < br.p_end) ? *br.p : 0;
        br.bits |= byte << br.bit_pos;
        br.bit_pos -= 8;
        br.p++;
    }
}

__device__ uint32_t brReadBitNoRefill(BitReader& br) {
    uint32_t r = br.bits >> 31;
    br.bits <<= 1;
    br.bit_pos++;
    return r;
}

__device__ uint32_t brReadBitsNoRefill(BitReader& br, uint32_t n) {
    uint32_t r = br.bits >> (32 - n);
    br.bits <<= n;
    br.bit_pos += n;
    return r;
}

// Reads n bits where n may be zero.
__device__ uint32_t brReadBitsNoRefillZero(BitReader& br, uint32_t n) {
    if (n == 0) return 0;
    uint32_t r = (br.bits >> 1) >> (31 - n);
    br.bits <<= n;
    br.bit_pos += n;
    return r;
}

__device__ uint32_t brReadFluff(BitReader& br, uint32_t num_symbols) {
    if (num_symbols == 256) return 0;
    uint32_t x = 257 - num_symbols;
    if (x > num_symbols) x = num_symbols;
    x *= 2;

    // log2_int for x-1 (x >= 2 guaranteed)
    uint32_t y = 0;
    { uint32_t tmp = x - 1; while (tmp > 1) { tmp >>= 1; y++; } y++; }

    uint32_t v = br.bits >> (32 - y);
    uint32_t z = (1u << y) - x;

    if ((v >> 1) >= z) {
        br.bits <<= y;
        br.bit_pos += y;
        return v - z;
    } else {
        br.bits <<= (y - 1);
        br.bit_pos += (y - 1);
        return v >> 1;
    }
}

// GPU integer log2 (floor) for values >= 1.
__device__ uint32_t ilog2(uint32_t v) {
    return 31 - __clz(v);
}

// ══════════════════════════════════════════════════════════════════
//  Golomb-Rice length decoding (mirrors CPU decodeGolombRiceLengths)
// ══════════════════════════════════════════════════════════════════

struct GolombRiceBR {
    const uint8_t* p;
    const uint8_t* p_end;
    uint32_t bit_pos;
};

// Returns false on error (source truncated).
__device__ bool decodeGolombRiceLengths(
    uint8_t* dst_buf, uint32_t size, GolombRiceBR& br
) {
    const uint8_t* p = br.p;
    const uint8_t* p_end = br.p_end;
    uint8_t* dst = dst_buf;
    const uint8_t* dst_end = dst_buf + size;

    if (p >= p_end) return false;

    int32_t count = -(int32_t)br.bit_pos;
    uint32_t initial_mask = 0xFFu >> br.bit_pos;
    uint32_t v = *p & initial_mask;
    p++;

    for (;;) {
        if (v == 0) {
            count += 8;
        } else {
            uint32_t x = k_rice_value[v];
            // Store count + low-nibble symbols. Byte-wise wrapping add.
            uint32_t lo = (uint32_t)count + (x & 0x0F0F0F0Fu);
            uint32_t hi = (x >> 4) & 0x0F0F0F0Fu;
            memcpy(dst, &lo, 4);
            memcpy(dst + 4, &hi, 4);
            dst += k_rice_len[v];
            if (dst >= dst_end) break;
            count = (int32_t)(x >> 28);
        }
        if (p >= p_end) return false;
        v = *p;
        p++;
    }

    // Overshoot correction
    if (dst > dst_end) {
        uint32_t n = (uint32_t)(dst - dst_end);
        for (uint32_t i = 0; i < n; i++)
            v &= v - 1;
    }

    uint32_t bp = 0;
    if ((v & 1) == 0) {
        p--;
        uint32_t q = __ffs(v) - 1;  // __ffs returns 1-based; ctz equivalent
        if (v == 0) q = 8;           // all zero bits means 8
        bp = 8 - q;
    }
    br.p = p;
    br.bit_pos = bp;
    return true;
}

// ══════════════════════════════════════════════════════════════════
//  Range conversion (mirrors CPU huffConvertToRanges)
// ══════════════════════════════════════════════════════════════════

struct HuffRange {
    uint16_t symbol;
    uint16_t num;
};

// Returns number of ranges, or 0 on error.
__device__ uint32_t huffConvertToRanges(
    HuffRange* ranges, uint32_t num_symbols, uint32_t p_fluff,
    const uint8_t* symlen_in, BitReader& br
) {
    uint32_t num_ranges = p_fluff >> 1;
    int32_t sym_idx = 0;
    const uint8_t* symlen = symlen_in;

    if (p_fluff & 1) {
        brRefill(br);
        uint32_t v = *symlen++;
        if (v >= 8) return 0;
        uint32_t extra = v + 1;
        sym_idx = (int32_t)(brReadBitsNoRefill(br, extra) + (1u << extra) - 1);
    }

    uint32_t syms_used = 0;
    for (uint32_t i = 0; i < num_ranges; i++) {
        brRefill(br);
        uint32_t v0 = symlen[0];
        if (v0 >= 9) return 0;
        uint32_t num = brReadBitsNoRefillZero(br, v0) + (1u << v0);
        uint32_t v1 = symlen[1];
        if (v1 >= 8) return 0;
        uint32_t v1p1 = v1 + 1;
        uint32_t space = brReadBitsNoRefill(br, v1p1) + (1u << v1p1) - 1;
        ranges[i].symbol = (uint16_t)sym_idx;
        ranges[i].num = (uint16_t)num;
        syms_used += num;
        sym_idx += (int32_t)(num + space);
        symlen += 2;
    }

    if (sym_idx >= 256 || syms_used >= num_symbols ||
        (uint32_t)sym_idx + num_symbols - syms_used > 256)
        return 0;

    ranges[num_ranges].symbol = (uint16_t)sym_idx;
    ranges[num_ranges].num = (uint16_t)(num_symbols - syms_used);
    return num_ranges + 1;
}

// ══════════════════════════════════════════════════════════════════
//  TansData intermediate (weight-1 in A[], weight>=2 in B[])
// ══════════════════════════════════════════════════════════════════

struct TansData {
    uint32_t a_used;
    uint32_t b_used;
    uint8_t  a[256];
    uint32_t b[256];
};

// ── Insertion sorts for small arrays ──────────────────────────────

__device__ void sortBytes(uint8_t* arr, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        uint8_t v = arr[i];
        uint32_t j = i;
        while (j > 0 && arr[j - 1] > v) { arr[j] = arr[j - 1]; j--; }
        arr[j] = v;
    }
}

__device__ void sortU32(uint32_t* arr, uint32_t n) {
    for (uint32_t i = 1; i < n; i++) {
        uint32_t v = arr[i];
        uint32_t j = i;
        while (j > 0 && arr[j - 1] > v) { arr[j] = arr[j - 1]; j--; }
        arr[j] = v;
    }
}

// ══════════════════════════════════════════════════════════════════
//  Phase 1: decodeTable — parse frequency table from bitstream
// ══════════════════════════════════════════════════════════════════

// Returns TANS_OK or an error code.
__device__ uint32_t decodeTable(
    BitReader& br, uint32_t log_table_bits, TansData& td
) {
    if (log_table_bits < 8 || log_table_bits > 12)
        return TANS_ERR_BAD_LOG_BITS;

    uint32_t L = 1u << log_table_bits;

    brRefill(br);

    if (brReadBitNoRefill(br) != 0) {
        // ── Golomb-Rice coded path ──
        uint32_t q = brReadBitsNoRefill(br, 3);
        uint32_t num_symbols = brReadBitsNoRefill(br, 8) + 1;
        if (num_symbols < 2) return TANS_ERR_BAD_TABLE;

        uint32_t fluff_init = brReadFluff(br, num_symbols);
        uint32_t total_rice = fluff_init + num_symbols;
        if (total_rice > 512) return TANS_ERR_BAD_TABLE;

        uint8_t rice[512 + 16];

        // Set up Golomb-Rice bit reader from current MSB reader position
        GolombRiceBR br2;
        br2.p_end = br.p_end;
        br2.bit_pos = (uint32_t)((br.bit_pos - 24) & 7);
        uint32_t step_back = (uint32_t)((24 - br.bit_pos + 7) >> 3);
        br2.p = br.p - step_back;

        if (!decodeGolombRiceLengths(rice, total_rice, br2))
            return TANS_ERR_BAD_TABLE;
        memset(rice + total_rice, 0, 16);

        // Reset MSB reader to br2's position
        br.bit_pos = 24;
        br.p = br2.p;
        br.bits = 0;
        brRefill(br);
        uint32_t br2_bp = br2.bit_pos;
        br.bits <<= br2_bp;
        br.bit_pos += br2_bp;

        if ((fluff_init >> 1) >= 133) return TANS_ERR_BAD_TABLE;

        HuffRange range_buf[133];
        uint32_t num_ranges = huffConvertToRanges(
            range_buf, num_symbols, fluff_init,
            rice + num_symbols, br
        );
        if (num_ranges == 0) return TANS_ERR_BAD_TABLE;

        brRefill(br);

        const uint8_t* cur_rice = rice;
        const uint8_t* cur_rice_end = rice + total_rice;
        int32_t average = 6;
        int32_t weight_sum = 0;
        uint32_t a_count = 0, b_count = 0;

        for (uint32_t ri = 0; ri < num_ranges; ri++) {
            uint32_t symbol = range_buf[ri].symbol;
            uint32_t num = range_buf[ri].num;
            if (num == 0 || num > 256) return TANS_ERR_BAD_TABLE;

            for (;;) {
                brRefill(br);
                if (cur_rice >= cur_rice_end) return TANS_ERR_BAD_TABLE;
                uint32_t rice_byte = *cur_rice++;
                uint32_t nextra = q + rice_byte;
                if (nextra > 15) return TANS_ERR_BAD_TABLE;

                uint32_t raw = brReadBitsNoRefillZero(br, nextra);
                int32_t v = (int32_t)(raw + (1u << nextra) - (1u << q));

                int32_t avg_div4 = average >> 2;
                int32_t limit = 2 * avg_div4;
                if (v <= limit) {
                    int32_t signed_half = -(v & 1) ^ (int32_t)((uint32_t)v >> 1);
                    v = avg_div4 + signed_half;
                }
                if (limit > v) limit = v;
                v += 1;
                average += limit - avg_div4;

                if (v == 1) {
                    if (a_count >= 256) return TANS_ERR_BAD_TABLE;
                    td.a[a_count++] = (uint8_t)symbol;
                } else {
                    if (b_count >= 256) return TANS_ERR_BAD_TABLE;
                    td.b[b_count++] = (symbol << 16) | (uint32_t)v;
                }
                weight_sum += v;
                if (weight_sum > (int32_t)L) return TANS_ERR_BAD_TABLE;
                symbol++;
                if (--num == 0) break;
            }
        }
        if (weight_sum != (int32_t)L) return TANS_ERR_BAD_TABLE;

        td.a_used = a_count;
        td.b_used = b_count;
        return TANS_OK;
    }

    // ── Sparse/explicit path ──
    bool seen[256];
    memset(seen, 0, sizeof(seen));

    uint32_t count = brReadBitsNoRefill(br, 3) + 1;
    uint32_t bits_per_sym = ilog2(log_table_bits) + 1;
    uint32_t max_delta_bits = brReadBitsNoRefill(br, bits_per_sym);
    if (max_delta_bits == 0 || max_delta_bits > log_table_bits)
        return TANS_ERR_BAD_TABLE;

    uint32_t a_used = 0, b_used = 0;
    uint32_t weight = 0, total_weights = 0;

    while (count != 0) {
        count--;
        brRefill(br);
        uint32_t sym = brReadBitsNoRefill(br, 8);
        if (seen[sym]) return TANS_ERR_BAD_TABLE;
        uint32_t delta = brReadBitsNoRefill(br, max_delta_bits);
        weight += delta;
        if (weight == 0) return TANS_ERR_BAD_TABLE;
        seen[sym] = true;

        if (weight == 1) {
            td.a[a_used++] = (uint8_t)sym;
        } else {
            td.b[b_used++] = (sym << 16) | weight;
        }
        total_weights += weight;
    }

    brRefill(br);
    uint32_t last_sym = brReadBitsNoRefill(br, 8);
    if (seen[last_sym]) return TANS_ERR_BAD_TABLE;

    int32_t diff = (int32_t)L - (int32_t)total_weights;
    if (diff < (int32_t)weight || diff <= 1) return TANS_ERR_BAD_TABLE;

    td.b[b_used++] = (last_sym << 16) | (uint32_t)(L - total_weights);

    td.a_used = a_used;
    td.b_used = b_used;

    // Sort A[] and B[] ascending
    sortBytes(td.a, a_used);
    sortU32(td.b, b_used);

    return TANS_OK;
}

// ══════════════════════════════════════════════════════════════════
//  Phase 2: initLut — FSE spread-based LUT construction
// ══════════════════════════════════════════════════════════════════

// FSE spread-based LUT construction (mirrors nvCOMP init_fse_tables).
// Phase 1 (lane 0): spread symbols across slots using step=(L>>1)+(L>>3)+3
// Phase 2 (all 32 lanes): compute bits_x/w/x per entry in parallel
// Simple arithmetic, no 4-way pointer arrays, minimal register pressure.
__device__ uint32_t initLut(
    const TansData& td, uint32_t log_table_bits, TansLutEnt* lut
) {
    const int lane = threadIdx.x & 31;
    const uint32_t L = 1u << log_table_bits;
    const uint32_t L_mask = L - 1;
    const int32_t a_used = (int32_t)td.a_used;
    if ((uint32_t)a_used > L) return TANS_ERR_LUT_FAILED;

    // FSE spread: lane 0 writes complete LUT entries directly to global memory.
    // No intermediate shared arrays — compute bits_x/w/x inline during spread.
    // All 32 lanes then do the pack step (8B->4B) in parallel.
    if (lane == 0) {
        uint32_t highThresh = L - (uint32_t)a_used;
        uint32_t step = (L >> 1) + (L >> 3) + 3;
        uint32_t pos = 0;

        // Weight >= 2 symbols: spread with step, write LUT entries directly
        for (uint32_t bi = 0; bi < td.b_used; bi++) {
            uint32_t val = td.b[bi];
            uint32_t weight = val & 0xFFFF;
            uint8_t symbol = (uint8_t)(val >> 16);

            for (uint32_t n = 0; n < weight; n++) {
                while (pos >= highThresh) pos = (pos + step) & L_mask;

                uint32_t running_w = weight + n;
                uint32_t wb = ilog2(running_w);
                uint32_t bps = log_table_bits - wb;
                lut[pos].symbol = symbol;
                lut[pos].bits_x = (uint8_t)bps;
                lut[pos].x = (1u << bps) - 1;
                lut[pos].w = (uint16_t)(L_mask & (running_w << bps));

                pos = (pos + step) & L_mask;
            }
        }

        // Weight-1 symbols: fill highThresh..L-1
        for (int32_t i = 0; i < a_used; i++) {
            uint32_t hi = highThresh + (uint32_t)i;
            lut[hi].symbol = td.a[i];
            lut[hi].bits_x = (uint8_t)log_table_bits;
            lut[hi].x = L_mask;
            lut[hi].w = 0;
        }
    }
    __syncwarp();

    return TANS_OK;
}

// ══════════════════════════════════════════════════════════════════
//  Phase 3: decode — 5-state interleaved forward/backward hot loop
// ══════════════════════════════════════════════════════════════════

// Byte-swap a 32-bit value (big-endian to little-endian).
__device__ uint32_t bswap32(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000u);
}

// Read 4 bytes as little-endian uint32 at arbitrary alignment.
__device__ uint32_t readLE32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

// Decode one symbol from forward or backward stream.
// Consumes bits from `bits`, updates `state`, returns the symbol byte.
__device__ __forceinline__ uint8_t decodeOneSymbol(
    const TansLutEnt* lut, uint32_t& state, uint32_t& bits,
    int32_t& bitpos, uint32_t lut_mask
) {
    const TansLutEnt& e = lut[state];
    uint8_t sym = e.symbol;
    bitpos -= (int32_t)e.bits_x;
    state = ((bits & e.x) + e.w) & lut_mask;
    bits >>= e.bits_x;
    return sym;
}

// Forward refill: read 4 bytes LE from ptr_f, OR shifted into bits_f.
__device__ __forceinline__ void refillForward(
    const uint8_t*& ptr_f, uint32_t& bits_f, int32_t& bitpos_f
) {
    uint32_t fw = readLE32(ptr_f);
    bits_f |= fw << bitpos_f;
    ptr_f += (31 - bitpos_f) >> 3;
    bitpos_f |= 24;
}

// Backward refill: read 4 bytes LE from (ptr_b - 4), byte-swap, OR shifted.
__device__ __forceinline__ void refillBackward(
    const uint8_t*& ptr_b, uint32_t& bits_b, int32_t& bitpos_b
) {
    uint32_t bw = readLE32(ptr_b - 4);
    bits_b |= bswap32(bw) << bitpos_b;
    ptr_b -= (31 - bitpos_b) >> 3;
    bitpos_b |= 24;
}

// ── 64-bit bit buffer variants for reduced refill frequency ──
// With ltb=11, each symbol uses ~11 bits. 64-bit buffer holds ~56 usable
// bits after refill = 5 symbols per refill (vs 2 with 32-bit buffer).

__device__ __forceinline__ uint64_t readLE64(const uint8_t* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

__device__ __forceinline__ uint64_t bswap64(uint64_t v) {
    return ((v & 0xFF00000000000000ull) >> 56) | ((v & 0x00FF000000000000ull) >> 40) |
           ((v & 0x0000FF0000000000ull) >> 24) | ((v & 0x000000FF00000000ull) >>  8) |
           ((v & 0x00000000FF000000ull) <<  8) | ((v & 0x0000000000FF0000ull) << 24) |
           ((v & 0x000000000000FF00ull) << 40) | ((v & 0x00000000000000FFull) << 56);
}

__device__ __forceinline__ uint8_t decodeOneSymbol64(
    const TansLutEnt* lut, uint32_t& state, uint64_t& bits,
    int32_t& bitpos, uint32_t lut_mask
) {
    const TansLutEnt& e = lut[state];
    uint8_t sym = e.symbol;
    bitpos -= (int32_t)e.bits_x;
    state = ((uint32_t)(bits & (uint64_t)e.x) + e.w) & lut_mask;
    bits >>= e.bits_x;
    return sym;
}

__device__ __forceinline__ void refillForward64(
    const uint8_t*& ptr_f, uint64_t& bits_f, int32_t& bitpos_f
) {
    bits_f |= readLE64(ptr_f) << bitpos_f;
    ptr_f += (63 - bitpos_f) >> 3;
    bitpos_f |= 56;
}

__device__ __forceinline__ void refillBackward64(
    const uint8_t*& ptr_b, uint64_t& bits_b, int32_t& bitpos_b
) {
    bits_b |= bswap64(readLE64(ptr_b - 8)) << bitpos_b;
    ptr_b -= (63 - bitpos_b) >> 3;
    bitpos_b |= 56;
}

// Returns TANS_OK or error code.  On success, writes final 5 states
// at dst_end[0..4] (matches CPU behavior).
__device__ uint32_t decode5State(
    const TansLutEnt* lut, uint32_t lut_mask,
    uint8_t* dst, uint8_t* dst_end,
    const uint8_t* ptr_f, const uint8_t* ptr_b,
    uint32_t bits_f, uint32_t bits_b,
    int32_t bitpos_f, int32_t bitpos_b,
    uint32_t state0, uint32_t state1,
    uint32_t state2, uint32_t state3, uint32_t state4,
    const uint8_t* src_start, const uint8_t* src_end_orig
) {
    if (ptr_f > ptr_b) return TANS_ERR_STREAM_MISMATCH;
    if (dst >= dst_end) goto done;

    {
        // Fast path: decode in batches of 10 symbols (5 forward + 5 backward)
        // with no per-symbol exit checks and batched 4-byte output writes.
        uint32_t total_syms = (uint32_t)(dst_end - dst);
        uint32_t full_iters = total_syms / 10;

        for (uint32_t iter = 0; iter < full_iters; iter++) {
            uint8_t s0, s1, s2, s3, s4;

            // ── Forward: 5 symbols ──
            refillForward(ptr_f, bits_f, bitpos_f);
            s0 = decodeOneSymbol(lut, state0, bits_f, bitpos_f, lut_mask);
            s1 = decodeOneSymbol(lut, state1, bits_f, bitpos_f, lut_mask);
            refillForward(ptr_f, bits_f, bitpos_f);
            s2 = decodeOneSymbol(lut, state2, bits_f, bitpos_f, lut_mask);
            s3 = decodeOneSymbol(lut, state3, bits_f, bitpos_f, lut_mask);
            refillForward(ptr_f, bits_f, bitpos_f);
            s4 = decodeOneSymbol(lut, state4, bits_f, bitpos_f, lut_mask);

            // Batch write 5 bytes (4-byte aligned write + 1 byte)
            { uint32_t w4 = (uint32_t)s0 | ((uint32_t)s1 << 8) | ((uint32_t)s2 << 16) | ((uint32_t)s3 << 24); memcpy(dst, &w4, 4); }
            dst[4] = s4;
            dst += 5;

            // ── Backward: 5 symbols ──
            refillBackward(ptr_b, bits_b, bitpos_b);
            s0 = decodeOneSymbol(lut, state0, bits_b, bitpos_b, lut_mask);
            s1 = decodeOneSymbol(lut, state1, bits_b, bitpos_b, lut_mask);
            refillBackward(ptr_b, bits_b, bitpos_b);
            s2 = decodeOneSymbol(lut, state2, bits_b, bitpos_b, lut_mask);
            s3 = decodeOneSymbol(lut, state3, bits_b, bitpos_b, lut_mask);
            refillBackward(ptr_b, bits_b, bitpos_b);
            s4 = decodeOneSymbol(lut, state4, bits_b, bitpos_b, lut_mask);

            { uint32_t w4 = (uint32_t)s0 | ((uint32_t)s1 << 8) | ((uint32_t)s2 << 16) | ((uint32_t)s3 << 24); memcpy(dst, &w4, 4); }
            dst[4] = s4;
            dst += 5;
        }

        // Tail: remaining < 10 symbols, use per-symbol checks
        while (dst < dst_end) {
            refillForward(ptr_f, bits_f, bitpos_f);
            *dst++ = decodeOneSymbol(lut, state0, bits_f, bitpos_f, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbol(lut, state1, bits_f, bitpos_f, lut_mask);
            if (dst >= dst_end) break;
            refillForward(ptr_f, bits_f, bitpos_f);
            *dst++ = decodeOneSymbol(lut, state2, bits_f, bitpos_f, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbol(lut, state3, bits_f, bitpos_f, lut_mask);
            if (dst >= dst_end) break;
            refillForward(ptr_f, bits_f, bitpos_f);
            *dst++ = decodeOneSymbol(lut, state4, bits_f, bitpos_f, lut_mask);
            if (dst >= dst_end) break;
            refillBackward(ptr_b, bits_b, bitpos_b);
            *dst++ = decodeOneSymbol(lut, state0, bits_b, bitpos_b, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbol(lut, state1, bits_b, bitpos_b, lut_mask);
            if (dst >= dst_end) break;
            refillBackward(ptr_b, bits_b, bitpos_b);
            *dst++ = decodeOneSymbol(lut, state2, bits_b, bitpos_b, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbol(lut, state3, bits_b, bitpos_b, lut_mask);
            if (dst >= dst_end) break;
            refillBackward(ptr_b, bits_b, bitpos_b);
            *dst++ = decodeOneSymbol(lut, state4, bits_b, bitpos_b, lut_mask);
        }
    }

done:
    // Pointer convergence check
    intptr_t ptr_diff = (intptr_t)ptr_b - (intptr_t)ptr_f;
    intptr_t adjust = (intptr_t)(bitpos_f >> 3) + (intptr_t)(bitpos_b >> 3);
    if (ptr_diff + adjust != 0) return TANS_ERR_STREAM_MISMATCH;

    uint32_t states_or = state0 | state1 | state2 | state3 | state4;
    if ((states_or & ~0xFFu) != 0) return TANS_ERR_STATE_RANGE;

    // Final states (matches CPU: written at dst_end[0..4])
    dst_end[0] = (uint8_t)state0;
    dst_end[1] = (uint8_t)state1;
    dst_end[2] = (uint8_t)state2;
    dst_end[3] = (uint8_t)state3;
    dst_end[4] = (uint8_t)state4;

    return TANS_OK;
}

// ══════════════════════════════════════════════════════════════════
//  Top-level per-chunk entry — called by lane 0
// ══════════════════════════════════════════════════════════════════

// Decodes one tANS chunk from src_buf[desc.src_offset..+desc.src_size]
// into dst_buf[desc.dst_offset..+desc.dst_size].
// Returns TANS_OK on success, error code otherwise.
__device__ uint32_t decodeTansChunk(
    const uint8_t* __restrict__ src_buf,
    uint8_t*       __restrict__ dst_buf,
    const TansDecChunkDesc& desc,
    TansLutEnt* lut  // pointer to shared-memory LUT
) {
    const uint8_t* src = src_buf + desc.src_offset;
    uint32_t src_size = desc.src_size;
    uint32_t dst_size = desc.dst_size;

    if (src_size < 8 || dst_size < 5)
        return TANS_ERR_SRC_TRUNCATED;

    const uint8_t* src_end_orig = src + src_size;
    const uint8_t* src_end = src_end_orig;

    // MSB-first bit reader for table header
    BitReader br;
    br.p = src;
    br.p_end = src_end;
    br.bits = 0;
    br.bit_pos = 24;
    brRefill(br);

    // Reserved bit must be 0
    if (brReadBitNoRefill(br) != 0)
        return TANS_ERR_BAD_TABLE;

    uint32_t log_table_bits = brReadBitsNoRefill(br, 2) + 8;

    // Phase 1: decode frequency table
    TansData td;
    td.a_used = 0;
    td.b_used = 0;
    uint32_t err = decodeTable(br, log_table_bits, td);
    if (err != TANS_OK) return err;

    // Finalize src position after table read
    // src = br.p - (24 - br.bit_pos) / 8
    int32_t byte_rewind = (24 - br.bit_pos) / 8;
    src = br.p - byte_rewind;
    const uint8_t* src_start_post_table = src;

    if (src >= src_end || (src_end - src) < 8)
        return TANS_ERR_SRC_TRUNCATED;

    int32_t L = 1 << log_table_bits;

    // Validate table weights
    int32_t a_used_i = (int32_t)td.a_used;
    int32_t b_used_i = (int32_t)td.b_used;
    if (a_used_i < 0 || a_used_i > L || b_used_i < 0 || b_used_i > 256)
        return TANS_ERR_BAD_WEIGHTS;
    int32_t w_sum = a_used_i;
    for (uint32_t i = 0; i < td.b_used; i++) {
        int32_t w = (int32_t)(td.b[i] & 0xFFFF);
        if (w < 2 || w > L) return TANS_ERR_BAD_WEIGHTS;
        w_sum += w;
    }
    if (w_sum != L) return TANS_ERR_BAD_WEIGHTS;

    // Phase 2: build LUT in shared memory
    err = initLut(td, log_table_bits, lut);
    if (err != TANS_OK) return err;

    // Phase 3: initialize 5 states from forward + backward bitstreams
    uint32_t bits_f = readLE32(src);
    src += 4;
    uint32_t bits_b = bswap32(readLE32(src_end - 4));
    src_end -= 4;
    int32_t bitpos_f = 32;
    int32_t bitpos_b = 32;

    uint32_t lut_mask = (1u << log_table_bits) - 1;
    uint32_t ltb = log_table_bits;

    uint32_t state0 = bits_f & lut_mask;
    uint32_t state1 = bits_b & lut_mask;
    bits_f >>= ltb;  bitpos_f -= ltb;
    bits_b >>= ltb;  bitpos_b -= ltb;

    uint32_t state2 = bits_f & lut_mask;
    uint32_t state3 = bits_b & lut_mask;
    bits_f >>= ltb;  bitpos_f -= ltb;
    bits_b >>= ltb;  bitpos_b -= ltb;

    // Refill forward for the 5th state
    bits_f |= readLE32(src) << bitpos_f;
    src += (31 - bitpos_f) >> 3;
    bitpos_f |= 24;

    uint32_t state4 = bits_f & lut_mask;
    bits_f >>= ltb;  bitpos_f -= ltb;

    // Compute pointer positions for decode loop
    // The CPU does: position_f = src - (bitpos_f >> 3)
    //               position_b = src_end + (bitpos_b >> 3)
    //               then bitpos_f &= 7, bitpos_b &= 7
    const uint8_t* pos_f = src - (bitpos_f >> 3);
    const uint8_t* pos_b = src_end + (bitpos_b >> 3);
    int32_t bp_f = bitpos_f & 7;
    int32_t bp_b = bitpos_b & 7;

    // Output: dst_size - 5 symbols decoded, then 5 final-state bytes at end
    uint8_t* dst = dst_buf + desc.dst_offset;
    uint8_t* dst_end_ptr = dst + dst_size - 5;

    return decode5State(
        lut, lut_mask,
        dst, dst_end_ptr,
        pos_f, pos_b,
        bits_f, bits_b,
        bp_f, bp_b,
        state0, state1, state2, state3, state4,
        src_start_post_table, src_end_orig
    );
}

// ══════════════════════════════════════════════════════════════════
//  Table parse + LUT build — __noinline__ to free registers for
//  the decode hot loop.
// ══════════════════════════════════════════════════════════════════

// Per-stream metadata written by table-build kernel, read by decode kernel
struct TansTableMeta {
    uint32_t src_after_table_off;  // offset from src_buf to post-table position
    uint32_t src_end_off;          // offset from src_buf to end of stream
    uint32_t log_table_bits;
    uint32_t error;
};

struct DecodeState {
    const uint8_t* ptr_f;
    const uint8_t* ptr_b;
    const uint8_t* src_start;
    const uint8_t* src_end_orig;
    uint32_t bits_f, bits_b;
    int32_t  bp_f, bp_b;
    uint32_t st0, st1, st2, st3, st4;
    uint32_t lut_mask;
    uint32_t total_syms;  // dst_size - 5
    uint32_t error;
};

// Intermediate state between table parse (lane 0) and LUT build (all lanes)
struct TableParseResult {
    TansData td;
    uint32_t log_table_bits;
    const uint8_t* src_after_table;
    const uint8_t* src_end;
    const uint8_t* src_end_orig;
    uint32_t dst_size;
    uint32_t error;
};

// Phase 1a: parse the tANS frequency table (lane 0 only, serial)
__device__ void parseTable(
    const uint8_t* __restrict__ src_buf,
    const TansDecChunkDesc& desc,
    TableParseResult& tp
) {
    tp.error = TANS_OK;
    tp.dst_size = desc.dst_size;

    const uint8_t* src = src_buf + desc.src_offset;
    uint32_t src_size = desc.src_size;
    if (src_size < 8 || desc.dst_size < 5) { tp.error = TANS_ERR_SRC_TRUNCATED; return; }

    const uint8_t* src_end_orig = src + src_size;
    const uint8_t* src_end = src_end_orig;
    tp.src_end_orig = src_end_orig;
    tp.src_end = src_end;

    BitReader br;
    br.p = src; br.p_end = src_end; br.bits = 0; br.bit_pos = 24;
    brRefill(br);
    if (brReadBitNoRefill(br) != 0) { tp.error = TANS_ERR_BAD_TABLE; return; }
    tp.log_table_bits = brReadBitsNoRefill(br, 2) + 8;

    tp.td.a_used = 0; tp.td.b_used = 0;
    uint32_t err = decodeTable(br, tp.log_table_bits, tp.td);
    if (err != TANS_OK) { tp.error = err; return; }

    int32_t byte_rewind = (24 - br.bit_pos) / 8;
    src = br.p - byte_rewind;
    tp.src_after_table = src;

    if (src >= src_end || (src_end - src) < 8) { tp.error = TANS_ERR_SRC_TRUNCATED; return; }

    int32_t L = 1 << tp.log_table_bits;
    int32_t a_used_i = (int32_t)tp.td.a_used;
    int32_t b_used_i = (int32_t)tp.td.b_used;
    if (a_used_i < 0 || a_used_i > L || b_used_i < 0 || b_used_i > 256) { tp.error = TANS_ERR_BAD_WEIGHTS; return; }
    int32_t w_sum = a_used_i;
    for (uint32_t i = 0; i < tp.td.b_used; i++) {
        int32_t w = (int32_t)(tp.td.b[i] & 0xFFFF);
        if (w < 2 || w > L) { tp.error = TANS_ERR_BAD_WEIGHTS; return; }
        w_sum += w;
    }
    if (w_sum != L) { tp.error = TANS_ERR_BAD_WEIGHTS; return; }
}

// Phase 1c: init 5 decode states from bitstream (lane 0 only, serial)
__device__ void initDecodeStates(
    const TableParseResult& tp,
    TansLutEnt* lut,
    DecodeState& ds
) {
    const uint8_t* src = tp.src_after_table;
    const uint8_t* src_end = tp.src_end;

    uint32_t bits_f = readLE32(src); src += 4;
    uint32_t bits_b = bswap32(readLE32(src_end - 4)); src_end -= 4;
    int32_t bitpos_f = 32, bitpos_b = 32;
    uint32_t lut_mask = (1u << tp.log_table_bits) - 1;
    uint32_t ltb = tp.log_table_bits;

    ds.st0 = bits_f & lut_mask; bits_f >>= ltb; bitpos_f -= ltb;
    ds.st1 = bits_b & lut_mask; bits_b >>= ltb; bitpos_b -= ltb;
    ds.st2 = bits_f & lut_mask; bits_f >>= ltb; bitpos_f -= ltb;
    ds.st3 = bits_b & lut_mask; bits_b >>= ltb; bitpos_b -= ltb;
    bits_f |= readLE32(src) << bitpos_f;
    src += (31 - bitpos_f) >> 3; bitpos_f |= 24;
    ds.st4 = bits_f & lut_mask; bits_f >>= ltb; bitpos_f -= ltb;

    ds.ptr_f = src - (bitpos_f >> 3);
    ds.ptr_b = src_end + (bitpos_b >> 3);
    ds.bits_f = bits_f; ds.bits_b = bits_b;
    ds.bp_f = bitpos_f & 7; ds.bp_b = bitpos_b & 7;
    ds.lut_mask = lut_mask;
    ds.total_syms = tp.dst_size - 5;
    ds.src_start = tp.src_after_table;
    ds.src_end_orig = tp.src_end_orig;
    ds.error = TANS_OK;
}

// ══════════════════════════════════════════════════════════════════
//  CUDA kernel entry point
// ══════════════════════════════════════════════════════════════════
//
// Launch config: 2 warps per block to decode 2 streams simultaneously.
// Each warp has its own LUT (16KB × 2 = 32KB shared, fits in 48KB).
// Grid = ceil(num_chunks / 2).

// ══════════════════════════════════════════════════════════════════
//  Table-build kernel: parse tables + build LUTs (all 32 lanes)
// ══════════════════════════════════════════════════════════════════
// Separate from decode so table-build registers are freed.
// All 32 lanes cooperate on initLut writes.
// Writes TansTableMeta for the decode kernel to read.

// Optional timing output: 4 x uint32 per chunk = [parse_cycles, initlut_cycles, pack_cycles, total_cycles]
// ══════════════════════════════════════════════════════════════════
//  Split build: Kernel A parses tables, Kernel B builds LUTs
// ══════════════════════════════════════════════════════════════════

// Per-stream parsed table data stored in global memory between kernels
struct TansParsedWeights {
    TansData td;
    uint32_t log_table_bits;
};

// Kernel A: parse Golomb-Rice table headers (lane 0 only, serial)
// Lightweight: no LUT construction, low register pressure
extern "C" __global__ void slzTansParseTablesKernel(
    const uint8_t* __restrict__ src_buf,
    const TansDecChunkDesc* __restrict__ descs,
    TansParsedWeights* __restrict__ weights_buf,  // [num_chunks]
    TansTableMeta* __restrict__ meta_buf,
    uint32_t       num_chunks
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;
    if (lane != 0) return;

    TableParseResult tp;
    parseTable(src_buf, descs[chunk_id], tp);

    if (tp.error != TANS_OK) {
        meta_buf[chunk_id].error = tp.error;
        return;
    }

    // Validate weights
    int32_t L = 1 << tp.log_table_bits;
    int32_t a_used_i = (int32_t)tp.td.a_used;
    int32_t b_used_i = (int32_t)tp.td.b_used;
    if (a_used_i < 0 || a_used_i > L || b_used_i < 0 || b_used_i > 256) {
        meta_buf[chunk_id].error = TANS_ERR_BAD_WEIGHTS; return;
    }
    int32_t w_sum = a_used_i;
    for (uint32_t i = 0; i < tp.td.b_used; i++) {
        int32_t w = (int32_t)(tp.td.b[i] & 0xFFFF);
        if (w < 2 || w > L) { meta_buf[chunk_id].error = TANS_ERR_BAD_WEIGHTS; return; }
        w_sum += w;
    }
    if (w_sum != L) { meta_buf[chunk_id].error = TANS_ERR_BAD_WEIGHTS; return; }

    // Write parsed weights + metadata
    weights_buf[chunk_id].td = tp.td;
    weights_buf[chunk_id].log_table_bits = tp.log_table_bits;

    meta_buf[chunk_id].error = TANS_OK;
    meta_buf[chunk_id].src_after_table_off = (uint32_t)((uintptr_t)tp.src_after_table - (uintptr_t)src_buf);
    meta_buf[chunk_id].src_end_off = (uint32_t)((uintptr_t)tp.src_end_orig - (uintptr_t)src_buf);
    meta_buf[chunk_id].log_table_bits = tp.log_table_bits;
}

// Kernel B: build + pack LUTs from parsed weights (all 32 lanes, fully parallel)
// No serial phase, no syncwarp stalls, low register pressure
extern "C" __global__ void slzTansInitLutKernel(
    const TansParsedWeights* __restrict__ weights_buf,
    const TansTableMeta* __restrict__ meta_buf,
    TansLutEnt*    __restrict__ lut_buf,
    uint32_t       num_chunks
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;
    const int lane = threadIdx.x & 31;

    if (meta_buf[chunk_id].error != TANS_OK) return;

    TansLutEnt* my_lut = lut_buf + (uint64_t)chunk_id * 2048;
    uint32_t log_table_bits = weights_buf[chunk_id].log_table_bits;

    // Build 8-byte LUT entries
    uint32_t err = initLut(weights_buf[chunk_id].td, log_table_bits, my_lut);
    __syncwarp();

    // Pack to 4-byte entries in-place
    if (err == TANS_OK) {
        uint32_t L = 1u << log_table_bits;
        uint32_t* pk = (uint32_t*)my_lut;
        for (uint32_t i = lane; i < L; i += 32)
            pk[i] = packLutEntry(my_lut[i]);
    }

    if (lane == 0 && err != TANS_OK) {
        // Overwrite meta error if initLut failed (shouldn't happen after weight validation)
        ((TansTableMeta*)meta_buf)[chunk_id].error = err;
    }
}

// Legacy combined kernel (kept for backward compat with type-1 5-state path)
extern "C" __global__ void __launch_bounds__(64) slzTansBuildTablesKernel(
    const uint8_t* __restrict__ src_buf,
    const TansDecChunkDesc* __restrict__ descs,
    TansLutEnt*    __restrict__ lut_buf,
    TansTableMeta* __restrict__ meta_buf,
    uint32_t       num_chunks,
    uint32_t*      __restrict__ timing_buf
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;

    TansLutEnt* my_lut = lut_buf + (uint64_t)chunk_id * 2048;
    const int lane = threadIdx.x & 31;

    __shared__ TableParseResult s_tp[2];
    if (lane == 0) parseTable(src_buf, descs[chunk_id], s_tp[warp_id]);
    __syncwarp();

    if (s_tp[warp_id].error != TANS_OK) {
        if (lane == 0) meta_buf[chunk_id].error = s_tp[warp_id].error;
        return;
    }

    uint32_t err = initLut(s_tp[warp_id].td, s_tp[warp_id].log_table_bits, my_lut);
    __syncwarp();

    if (err == TANS_OK) {
        uint32_t L = 1u << s_tp[warp_id].log_table_bits;
        uint32_t* packed = (uint32_t*)my_lut;
        for (uint32_t i = lane; i < L; i += 32)
            packed[i] = packLutEntry(my_lut[i]);
        __syncwarp();
    }

    if (lane == 0) {
        meta_buf[chunk_id].error = err;
        if (err == TANS_OK) {
            meta_buf[chunk_id].src_after_table_off = (uint32_t)((uintptr_t)s_tp[warp_id].src_after_table - (uintptr_t)src_buf);
            meta_buf[chunk_id].src_end_off = (uint32_t)((uintptr_t)s_tp[warp_id].src_end_orig - (uintptr_t)src_buf);
            meta_buf[chunk_id].log_table_bits = s_tp[warp_id].log_table_bits;
        }
    }
}

// ══════════════════════════════════════════════════════════════════
//  Decode-only kernel: reads pre-built LUTs + metadata
// ══════════════════════════════════════════════════════════════════

// LUT in global memory: each stream gets 2048 entries at lut_buf[chunk_id * 2048].
// Eliminates 32KB shared memory constraint, allowing higher occupancy.
// L1 cache (92% hit rate) handles the random LUT accesses.
extern "C" __global__ void __launch_bounds__(64, 4) slzTansDecodeKernel(
    const uint8_t* __restrict__ src_buf,
    uint8_t*       __restrict__ dst_buf,
    const TansDecChunkDesc* __restrict__ descs,
    uint32_t*      __restrict__ out_status,
    uint32_t       num_chunks,
    TansLutEnt*    __restrict__ lut_buf,
    const TansTableMeta* __restrict__ meta_buf,  // pre-built table metadata (or nullptr)
    uint32_t*      __restrict__ chunk_ready       // per-LZ-chunk ready counter (or nullptr)
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;

    // When meta_buf is set, LUT is packed 4-byte entries at the same offset
    const uint32_t* packed_lut = (const uint32_t*)((uint8_t*)lut_buf + (uint64_t)chunk_id * 2048 * sizeof(TansLutEnt));
    TansLutEnt* my_lut = lut_buf + (uint64_t)chunk_id * 2048;

    const int lane = threadIdx.x & 31;

    DecodeState ds;

    if (meta_buf != nullptr) {
        // Fast path: LUT already built by slzTansBuildTablesKernel
        uint32_t err = meta_buf[chunk_id].error;
        if (err != TANS_OK) {
            if (lane == 0) out_status[chunk_id] = err;
            return;
        }
        // Init decode states from metadata (lane 0 only)
        if (lane == 0) {
            TableParseResult tp;
            tp.src_after_table = src_buf + meta_buf[chunk_id].src_after_table_off;
            tp.src_end = src_buf + meta_buf[chunk_id].src_end_off;
            tp.src_end_orig = tp.src_end;
            tp.log_table_bits = meta_buf[chunk_id].log_table_bits;
            tp.dst_size = descs[chunk_id].dst_size;
            initDecodeStates(tp, my_lut, ds);
        }
    } else {
        // Fallback: build table + LUT inline (original path)
        __shared__ TableParseResult s_tp[2];
        if (lane == 0) parseTable(src_buf, descs[chunk_id], s_tp[warp_id]);
        __syncwarp();

        uint32_t err = s_tp[warp_id].error;
        if (err != TANS_OK) {
            if (lane == 0) out_status[chunk_id] = err;
            return;
        }

        err = initLut(s_tp[warp_id].td, s_tp[warp_id].log_table_bits, my_lut);
        __syncwarp();
        if (err != TANS_OK) {
            if (lane == 0) out_status[chunk_id] = err;
            return;
        }

        if (lane == 0) initDecodeStates(s_tp[warp_id], my_lut, ds);

        // Pack to 4-byte entries for decode loop
        {
            uint32_t L = 1u << s_tp[warp_id].log_table_bits;
            uint32_t* pk = (uint32_t*)my_lut;
            for (uint32_t i = lane; i < L; i += 32)
                pk[i] = packLutEntry(my_lut[i]);
            __syncwarp();
        }
    }

    uint32_t decode_err = __shfl_sync(0xFFFFFFFF, ds.error, 0);
    if (decode_err != TANS_OK) {
        if (lane == 0) out_status[chunk_id] = decode_err;
        return;
    }

    // Phase 2: decode + coalesced output
    // my_lut already set from global lut_buf above
    uint32_t total_syms = __shfl_sync(0xFFFFFFFF, ds.total_syms, 0);
    uint32_t dst_offset = __shfl_sync(0xFFFFFFFF, descs[chunk_id].dst_offset, 0);
    uint8_t* dst_base = dst_buf + dst_offset;
    uint32_t out_pos = 0;

    // Lane 0 owns the decode state — widen to 64-bit bit buffers
    const uint8_t* ptr_f, *ptr_b;
    uint64_t bits_f64, bits_b64;
    uint32_t lut_mask;
    int32_t bp_f, bp_b;
    uint32_t st0, st1, st2, st3, st4;
    if (lane == 0) {
        ptr_f = ds.ptr_f; ptr_b = ds.ptr_b;
        // Widen 32-bit residual bits to 64-bit and do initial 64-bit refill
        bits_f64 = (uint64_t)ds.bits_f;
        bits_b64 = (uint64_t)ds.bits_b;
        bp_f = ds.bp_f; bp_b = ds.bp_b;
        refillForward64(ptr_f, bits_f64, bp_f);
        refillBackward64(ptr_b, bits_b64, bp_b);
        st0 = ds.st0; st1 = ds.st1; st2 = ds.st2; st3 = ds.st3; st4 = ds.st4;
        lut_mask = ds.lut_mask;
    }

    if (lane == 0) {
        uint8_t* dst = dst_base;
        uint32_t full20 = total_syms / 20;
        for (uint32_t batch = 0; batch < full20; batch++) {
            uint8_t s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19;

            // Round 1: fwd 5 (1 refill for all 5)
            refillForward64(ptr_f, bits_f64, bp_f);
            s0 = decodeOneSymbolPacked(packed_lut, st0, bits_f64, bp_f, lut_mask);
            s1 = decodeOneSymbolPacked(packed_lut, st1, bits_f64, bp_f, lut_mask);
            s2 = decodeOneSymbolPacked(packed_lut, st2, bits_f64, bp_f, lut_mask);
            s3 = decodeOneSymbolPacked(packed_lut, st3, bits_f64, bp_f, lut_mask);
            s4 = decodeOneSymbolPacked(packed_lut, st4, bits_f64, bp_f, lut_mask);
            // Round 1: bwd 5 (1 refill for all 5)
            refillBackward64(ptr_b, bits_b64, bp_b);
            s5 = decodeOneSymbolPacked(packed_lut, st0, bits_b64, bp_b, lut_mask);
            s6 = decodeOneSymbolPacked(packed_lut, st1, bits_b64, bp_b, lut_mask);
            s7 = decodeOneSymbolPacked(packed_lut, st2, bits_b64, bp_b, lut_mask);
            s8 = decodeOneSymbolPacked(packed_lut, st3, bits_b64, bp_b, lut_mask);
            s9 = decodeOneSymbolPacked(packed_lut, st4, bits_b64, bp_b, lut_mask);
            // Round 2: fwd 5
            refillForward64(ptr_f, bits_f64, bp_f);
            s10 = decodeOneSymbolPacked(packed_lut, st0, bits_f64, bp_f, lut_mask);
            s11 = decodeOneSymbolPacked(packed_lut, st1, bits_f64, bp_f, lut_mask);
            s12 = decodeOneSymbolPacked(packed_lut, st2, bits_f64, bp_f, lut_mask);
            s13 = decodeOneSymbolPacked(packed_lut, st3, bits_f64, bp_f, lut_mask);
            s14 = decodeOneSymbolPacked(packed_lut, st4, bits_f64, bp_f, lut_mask);
            // Round 2: bwd 5
            refillBackward64(ptr_b, bits_b64, bp_b);
            s15 = decodeOneSymbolPacked(packed_lut, st0, bits_b64, bp_b, lut_mask);
            s16 = decodeOneSymbolPacked(packed_lut, st1, bits_b64, bp_b, lut_mask);
            s17 = decodeOneSymbolPacked(packed_lut, st2, bits_b64, bp_b, lut_mask);
            s18 = decodeOneSymbolPacked(packed_lut, st3, bits_b64, bp_b, lut_mask);
            s19 = decodeOneSymbolPacked(packed_lut, st4, bits_b64, bp_b, lut_mask);

            // Write 20 bytes as 8+8+4 (3 stores)
            uint64_t w0 = (uint64_t)s0 | ((uint64_t)s1<<8) | ((uint64_t)s2<<16) | ((uint64_t)s3<<24) |
                          ((uint64_t)s4<<32) | ((uint64_t)s5<<40) | ((uint64_t)s6<<48) | ((uint64_t)s7<<56);
            uint64_t w1 = (uint64_t)s8 | ((uint64_t)s9<<8) | ((uint64_t)s10<<16) | ((uint64_t)s11<<24) |
                          ((uint64_t)s12<<32) | ((uint64_t)s13<<40) | ((uint64_t)s14<<48) | ((uint64_t)s15<<56);
            uint32_t w2 = (uint32_t)s16 | ((uint32_t)s17<<8) | ((uint32_t)s18<<16) | ((uint32_t)s19<<24);
            memcpy(dst, &w0, 8);
            memcpy(dst + 8, &w1, 8);
            memcpy(dst + 16, &w2, 4);
            dst += 20;
        }
        // Handle remaining full groups of 10
        uint32_t rem = total_syms - full20 * 20;
        if (rem >= 10) {
            uint8_t s0,s1,s2,s3,s4,s5,s6,s7,s8,s9;
            refillForward64(ptr_f, bits_f64, bp_f);
            s0 = decodeOneSymbolPacked(packed_lut, st0, bits_f64, bp_f, lut_mask);
            s1 = decodeOneSymbolPacked(packed_lut, st1, bits_f64, bp_f, lut_mask);
            s2 = decodeOneSymbolPacked(packed_lut, st2, bits_f64, bp_f, lut_mask);
            s3 = decodeOneSymbolPacked(packed_lut, st3, bits_f64, bp_f, lut_mask);
            s4 = decodeOneSymbolPacked(packed_lut, st4, bits_f64, bp_f, lut_mask);
            refillBackward64(ptr_b, bits_b64, bp_b);
            s5 = decodeOneSymbolPacked(packed_lut, st0, bits_b64, bp_b, lut_mask);
            s6 = decodeOneSymbolPacked(packed_lut, st1, bits_b64, bp_b, lut_mask);
            s7 = decodeOneSymbolPacked(packed_lut, st2, bits_b64, bp_b, lut_mask);
            s8 = decodeOneSymbolPacked(packed_lut, st3, bits_b64, bp_b, lut_mask);
            s9 = decodeOneSymbolPacked(packed_lut, st4, bits_b64, bp_b, lut_mask);
            uint64_t w8 = (uint64_t)s0 | ((uint64_t)s1<<8) | ((uint64_t)s2<<16) | ((uint64_t)s3<<24) |
                          ((uint64_t)s4<<32) | ((uint64_t)s5<<40) | ((uint64_t)s6<<48) | ((uint64_t)s7<<56);
            memcpy(dst, &w8, 8);
            uint16_t wt = (uint16_t)s8 | ((uint16_t)s9 << 8);
            memcpy(dst + 8, &wt, 2);
            dst += 10;
        }
        out_pos = (uint32_t)(dst - dst_base);
    }

    // Tail: remaining < 10 symbols, lane 0 writes directly (64-bit decode)
    if (lane == 0) {
        uint8_t* dst = dst_base + out_pos;
        uint8_t* dst_end = dst_base + total_syms;

        while (dst < dst_end) {
            refillForward64(ptr_f, bits_f64, bp_f);
            *dst++ = decodeOneSymbolPacked(packed_lut, st0, bits_f64, bp_f, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st1, bits_f64, bp_f, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st2, bits_f64, bp_f, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st3, bits_f64, bp_f, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st4, bits_f64, bp_f, lut_mask);
            if (dst >= dst_end) break;
            refillBackward64(ptr_b, bits_b64, bp_b);
            *dst++ = decodeOneSymbolPacked(packed_lut, st0, bits_b64, bp_b, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st1, bits_b64, bp_b, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st2, bits_b64, bp_b, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st3, bits_b64, bp_b, lut_mask);
            if (dst >= dst_end) break;
            *dst++ = decodeOneSymbolPacked(packed_lut, st4, bits_b64, bp_b, lut_mask);
        }

        // Convergence check
        intptr_t ptr_diff = (intptr_t)ptr_b - (intptr_t)ptr_f;
        intptr_t adjust = (intptr_t)(bp_f >> 3) + (intptr_t)(bp_b >> 3);
        if (ptr_diff + adjust != 0) { out_status[chunk_id] = TANS_ERR_STREAM_MISMATCH; return; }

        uint32_t states_or = st0 | st1 | st2 | st3 | st4;
        if ((states_or & ~0xFFu) != 0) { out_status[chunk_id] = TANS_ERR_STATE_RANGE; return; }

        dst_base[total_syms + 0] = (uint8_t)st0;
        dst_base[total_syms + 1] = (uint8_t)st1;
        dst_base[total_syms + 2] = (uint8_t)st2;
        dst_base[total_syms + 3] = (uint8_t)st3;
        dst_base[total_syms + 4] = (uint8_t)st4;
        out_status[chunk_id] = TANS_OK;
    }
}

// ══════════════════════════════════════════════════════════════════
//  32-lane parallel tANS decode kernel (chunk_type = 3)
// ══════════════════════════════════════════════════════════════════
//
// Wire format after table:
//   [32 × u16 LE sub-stream sizes]   (64 bytes)
//   [32 × u8 final states]           (32 bytes)
//   [sub-stream 0] [sub-stream 1] ... [sub-stream 31]
//
// All 32 lanes decode in parallel. Each lane maintains its own state
// and reads from its own sub-stream. Output is interleaved:
//   dst[i*32 + lane] = symbol_i_from_lane
//
// The LUT is shared (same table, built once by slzTansBuildTablesKernel).

extern "C" __global__ void __launch_bounds__(64, 4) slzTans32DecodeKernel(
    const uint8_t* __restrict__ src_buf,
    uint8_t*       __restrict__ dst_buf,
    const TansDecChunkDesc* __restrict__ descs,
    uint32_t*      __restrict__ out_status,
    uint32_t       num_chunks,
    TansLutEnt*    __restrict__ lut_buf,           // per-stream packed LUTs from build kernel
    const TansTableMeta* __restrict__ meta_buf     // per-stream metadata from build kernel
) {
    const uint32_t warp_id = threadIdx.y;
    const uint32_t chunk_id = blockIdx.x * 2 + warp_id;
    if (chunk_id >= num_chunks) return;

    const int lane = threadIdx.x & 31;

    // src_offset was adjusted +128 by the scan (for the build kernel to see the table).
    // The sizes+states header is at src_offset - 128.
    const uint8_t* header = src_buf + descs[chunk_id].src_offset - 128;
    uint32_t dst_size = descs[chunk_id].dst_size;

    // Read sizes and states from fixed 128-byte header
    uint16_t my_sub_size = (uint16_t)header[lane * 2] | ((uint16_t)header[lane * 2 + 1] << 8);
    uint16_t my_init_state = (uint16_t)header[64 + lane * 2] | ((uint16_t)header[64 + lane * 2 + 1] << 8);

    // Per-stream packed LUT from build kernel
    const uint32_t* packed_lut = (const uint32_t*)((uint8_t*)lut_buf + (uint64_t)chunk_id * 2048 * sizeof(TansLutEnt));
    uint32_t log_table_bits = meta_buf[chunk_id].log_table_bits;
    uint32_t lut_mask = (1u << log_table_bits) - 1;

    // Sub-streams follow immediately after sizes+states (128 bytes)
    // The Golomb-Rice table is in the compressed data but was already parsed by build kernel
    const uint8_t* sub_data_start = src_buf + meta_buf[chunk_id].src_after_table_off;

    // Compute sub-stream start offset via warp prefix sum
    uint32_t my_size32 = (uint32_t)my_sub_size;
    uint32_t prefix_sum = my_size32;
    for (int d = 1; d < 32; d <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, prefix_sum, d);
        if (lane >= d) prefix_sum += n;
    }
    uint32_t my_offset = prefix_sum - my_size32;

    const uint8_t* my_src = sub_data_start + my_offset;

    uint32_t total_syms = dst_size;
    uint32_t my_sym_count = total_syms / 32;
    if ((uint32_t)lane < (total_syms % 32)) my_sym_count++;

    uint8_t* dst_base = dst_buf + descs[chunk_id].dst_offset;

    if (my_sym_count == 0 || my_sub_size < 4) {
        __syncwarp();
        if (lane == 0) out_status[chunk_id] = TANS_OK;
        return;
    }

    uint32_t state = (uint32_t)my_init_state & lut_mask;
    uint32_t bits = readLE32(my_src);
    int32_t bitpos = 32;
    const uint8_t* ptr = my_src + 4;
    const uint8_t* ptr_end = my_src + my_sub_size;

    // Decode loop: each lane decodes its symbols, writing interleaved
    for (uint32_t i = 0; i < my_sym_count; i++) {
        // Refill when low on bits
        if (bitpos < 16) {
            if (ptr + 4 <= ptr_end) {
                bits |= readLE32(ptr) << bitpos;
                ptr += (31 - bitpos) >> 3;
                bitpos |= 24;
            } else {
                while (bitpos < 24 && ptr < ptr_end) {
                    bits |= (uint32_t)(*ptr) << bitpos;
                    ptr++;
                    bitpos += 8;
                }
            }
        }

        uint32_t packed = __ldg(&packed_lut[state]);
        uint32_t bits_x = packed >> 24;
        uint8_t sym = (uint8_t)((packed >> 16) & 0xFF);
        uint32_t w = packed & 0xFFFF;
        uint32_t x = (1u << bits_x) - 1;
        state = ((uint32_t)(bits & (uint64_t)x) + w) & lut_mask;
        bits >>= bits_x;
        bitpos -= (int32_t)bits_x;

        // Write interleaved: position i*32 + lane
        dst_base[i * 32 + lane] = sym;
    }

    __syncwarp();
    if (lane == 0) out_status[chunk_id] = TANS_OK;
}
