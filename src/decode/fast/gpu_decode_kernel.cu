// ── StreamLZ GPU Decode Kernel ──────────────────────────────────
// Each warp (32 threads) decodes one StreamLZ chunk independently.
// Two warps are packed per CUDA thread-block for SM occupancy
// (blockDim 32,2,1 — the two warps do NOT cooperate).
// Lane 0 of each warp parses tokens serially; all 32 lanes
// participate in literal and match byte copies.
//
// Decode paths:
//   decodeSubChunkL1      — 64KB raw-mode (no off32, no delta, no block split)
//   decodeSubChunk        — general (off32, delta literals, 2-block sub-chunks)
//   parseAndDecodeSubChunk — parses stream headers, dispatches to above
//
// Build: nvcc -cubin -arch=sm_89 -O3 gpu_decode_kernel.cu
// Embed: @embedFile("gpu_decode_kernel.cubin") in gpu_driver.zig

#include <cstdint>
#include <cstdio>

// ── StreamLZ token format constants ────────────────────────────
// Tokens encode literal-length + match-length + offset type in one byte.
//   token >= 24:  standard — lit[2:0], match[6:3], use_recent[7], off16
//   token == 0:   long literal — length from length stream + 64
//   token == 1:   long near match — length from length stream + 91, off16
//   token == 2:   long far match — length from length stream + 29, off32
//   token 3-23:   short far match — match = token + 5, off32
static constexpr uint32_t TOKEN_SHORT_MIN      = 24;
static constexpr uint32_t TOKEN_LONG_LITERAL   = 0;
static constexpr uint32_t TOKEN_LONG_NEAR      = 1;
static constexpr uint32_t TOKEN_LONG_FAR       = 2;
static constexpr uint32_t LONG_LITERAL_BASE    = 64;
static constexpr uint32_t LONG_NEAR_BASE       = 91;
static constexpr uint32_t LONG_FAR_BASE        = 29;
static constexpr uint32_t SHORT_FAR_BASE       = 5;
static constexpr uint32_t BLOCK_SIZE           = 0x10000;  // 64KB

// ── Chunk descriptor (extern-matched with gpu_driver.zig ChunkDesc) ────
struct SlzChunkDesc {
    uint32_t src_offset;    // byte offset into compressed block
    uint32_t comp_size;     // compressed payload size
    uint32_t decomp_size;   // decompressed size (usually 256KB)
    uint32_t dst_offset;    // absolute output position
    uint32_t flags;         // bit 0: uncompressed, bit 1: memset
    uint8_t  memset_fill;   // fill byte for memset chunks
    uint8_t  _pad[3];
};

// ── Extended length decode ─────────────────────────────────────
// Reads a variable-length count from the length stream. Uses a uint32
// offset from len_data base to avoid 64-bit pointer register pressure.
__device__ uint32_t readLength(const uint8_t* len_data, uint32_t &len_off, uint32_t len_avail) {
    if (len_off >= len_avail) return 0;
    uint32_t v = len_data[len_off];
    if (v > 251) {
        if (len_off + 2 >= len_avail) { len_off++; return v; }
        uint16_t extra;
        memcpy(&extra, len_data + len_off + 1, 2);
        v += (uint32_t)extra * 4;
        len_off += 2;
    }
    len_off++;
    return v;
}

// ── L1 raw-mode sub-chunk decoder ──────────────────────────────
// Streamlined single-block raw decoder for 64KB chunks.
// No off32, no delta, no block split — fastest path for L1 data.
// Register-optimized: uses uint32 offsets instead of 64-bit pointers,
// removes redundant bounds checks and shuffles.
__device__ void decodeSubChunkL1(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t dst_offset
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = -8;
    uint32_t len_off = 0;

    while (cmd_pos < cmd_size) {
        uint32_t lit_len = 0, match_len = 0;
        int32_t match_offset = recent_offset;
        uint32_t use_recent = 0;

        // ── Token parse (lane 0 only) ──
        if (lane == 0) {
            uint32_t token = cmd[cmd_pos];
            if (token >= TOKEN_SHORT_MIN) {
                lit_len = token & 7;
                match_len = (token >> 3) & 0xF;
                use_recent = (token >> 7) & 1;
                if (!use_recent && off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    match_offset = -(int32_t)v;
                    off16_pos++;
                }
            } else if (token == TOKEN_LONG_LITERAL) {
                use_recent = 1;
                lit_len = readLength(len_data, len_off, len_avail) + LONG_LITERAL_BASE;
            } else if (token == TOKEN_LONG_NEAR) {
                match_len = readLength(len_data, len_off, len_avail) + LONG_NEAR_BASE;
                if (off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    match_offset = -(int32_t)v; off16_pos++;
                }
            }
        }
        cmd_pos++;

        // ── Broadcast parsed values from lane 0 to all lanes ──
        lit_len      = __shfl_sync(0xFFFFFFFF, lit_len, 0);
        match_len    = __shfl_sync(0xFFFFFFFF, match_len, 0);
        match_offset = __shfl_sync(0xFFFFFFFF, match_offset, 0);
        use_recent   = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        off16_pos    = __shfl_sync(0xFFFFFFFF, off16_pos, 0);

        // ── Warp-cooperative literal copy ──
        if (lit_len > 0) {
            for (uint32_t i = lane; i < lit_len; i += 32)
                dst[dst_pos + i] = lit[lit_pos + i];
            __syncwarp();
        }
        dst_pos += lit_len;
        lit_pos += lit_len;

        // ── Warp-cooperative match copy ──
        if (match_len > 0) {
            uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
            int32_t match_dist = -match_offset;
            if (match_dist >= (int32_t)match_len && match_len > 1) {
                for (uint32_t i = lane; i < match_len; i += 32)
                    dst[dst_pos + i] = dst[match_src + i];
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < match_len; i++)
                        dst[dst_pos + i] = dst[match_src + i];
            }
            __syncwarp();
        }
        dst_pos += match_len;

        if (!use_recent) recent_offset = match_offset;
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
        len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
    }

    // ── Trailing literals (bytes after the last token) ──
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    for (uint32_t i = lane; i < trailing; i += 32)
        dst[dst_pos + i] = lit[lit_pos + i];

}

// ── General sub-chunk decoder ──────────────────────────────────
// Full-featured path: handles off32, delta literals, and multi-block
// sub-chunks (up to 2 blocks of 64KB each).
// mode: 0 = delta literals, 1 = raw literals
__device__ void decodeSubChunk(
    const uint8_t* __restrict__ cmd, uint32_t cmd_size,
    const uint8_t* __restrict__ lit, uint32_t lit_size,
    const uint8_t* __restrict__ off16_raw, uint32_t off16_count,
    const uint8_t* __restrict__ off32_raw1, uint32_t off32_count1,
    const uint8_t* __restrict__ off32_raw2, uint32_t off32_count2,
    const uint8_t* __restrict__ len_data, uint32_t len_avail,
    uint8_t* __restrict__ dst, uint32_t dst_size,
    uint32_t initial_copy,
    uint32_t cmd_stream2_offset,
    uint32_t dst_offset,
    uint32_t mode
) {
    const int lane = threadIdx.x & 31;
    uint32_t cmd_pos = 0, lit_pos = 0, off16_pos = 0, off32_pos = 0;
    uint32_t dst_pos = dst_offset + initial_copy;
    int32_t recent_offset = -8;
    uint32_t dst_end_abs = dst_offset + dst_size;
    uint32_t len_off = 0;
    const uint8_t* off32_cur = off32_raw1;
    uint32_t off32_count_cur = off32_count1;
    uint32_t block_dst_start = dst_offset;

    uint32_t block_cmd_end = (cmd_stream2_offset > 0 && cmd_stream2_offset < cmd_size)
        ? cmd_stream2_offset : cmd_size;
    int block_iter = 0;

    for (;;) {
    while (cmd_pos < block_cmd_end) {
        uint32_t token = 0, lit_len = 0, match_len = 0;
        int32_t match_offset = recent_offset;
        uint32_t use_recent = 0, token_type = 0;

        // ── Token parse (lane 0 only) ──
        if (lane == 0) {
            token = cmd[cmd_pos++];
            if (token >= TOKEN_SHORT_MIN) {
                token_type = 0;
                lit_len = token & 7;
                match_len = (token >> 3) & 0xF;
                use_recent = (token >> 7) & 1;
                if (!use_recent && off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    match_offset = -(int32_t)v;
                    off16_pos++;
                }
            } else if (token == TOKEN_LONG_LITERAL) {
                token_type = 1;
                lit_len = readLength(len_data, len_off, len_avail) + LONG_LITERAL_BASE;
            } else if (token == TOKEN_LONG_NEAR) {
                token_type = 2;
                match_len = readLength(len_data, len_off, len_avail) + LONG_NEAR_BASE;
                if (off16_pos < off16_count) {
                    uint16_t v; memcpy(&v, off16_raw + off16_pos * 2, 2);
                    match_offset = -(int32_t)v; off16_pos++;
                }
                use_recent = 0;
            } else if (token == TOKEN_LONG_FAR) {
                token_type = 4;
                match_len = readLength(len_data, len_off, len_avail) + LONG_FAR_BASE;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    match_offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            } else {
                token_type = 3;
                match_len = token + SHORT_FAR_BASE;
                if (off32_pos < off32_count_cur) {
                    uint32_t v; memcpy(&v, off32_cur + off32_pos * 4, 4);
                    match_offset = (int32_t)block_dst_start - (int32_t)v - (int32_t)dst_pos;
                    off32_pos++;
                }
                use_recent = 0;
            }
        }

        // ── Broadcast parsed values from lane 0 to all lanes ──
        token_type   = __shfl_sync(0xFFFFFFFF, token_type, 0);
        lit_len      = __shfl_sync(0xFFFFFFFF, lit_len, 0);
        match_len    = __shfl_sync(0xFFFFFFFF, match_len, 0);
        match_offset = __shfl_sync(0xFFFFFFFF, match_offset, 0);
        use_recent   = __shfl_sync(0xFFFFFFFF, use_recent, 0);
        cmd_pos      = __shfl_sync(0xFFFFFFFF, cmd_pos, 0);
        lit_pos      = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        off16_pos    = __shfl_sync(0xFFFFFFFF, off16_pos, 0);
        off32_pos    = __shfl_sync(0xFFFFFFFF, off32_pos, 0);

        // ── Warp-cooperative literal copy ──
        if (lit_len > 0) {
            if (mode == 0) {
                for (uint32_t i = lane; i < lit_len; i += 32)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                        uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                        dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
                    }
            } else {
                for (uint32_t i = lane; i < lit_len; i += 32)
                    if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                        dst[dst_pos + i] = lit[lit_pos + i];
            }
            __syncwarp();
            dst_pos += lit_len;
            lit_pos += lit_len;
        }

        // ── Warp-cooperative match copy ──
        if (match_len > 0) {
            uint32_t match_src = (uint32_t)((int32_t)dst_pos + match_offset);
            int32_t match_dist = -match_offset;
            if (match_dist >= (int32_t)match_len && match_len > 1) {
                for (uint32_t i = lane; i < match_len; i += 32)
                    if (dst_pos + i < dst_end_abs)
                        dst[dst_pos + i] = dst[match_src + i];
            } else {
                if (lane == 0)
                    for (uint32_t i = 0; i < match_len; i++)
                        if (dst_pos + i < dst_end_abs)
                            dst[dst_pos + i] = dst[match_src + i];
            }
            __syncwarp();
            dst_pos += match_len;
        }

        dst_pos   = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
        lit_pos   = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
        if (!use_recent && (token_type != 1)) recent_offset = match_offset;
        recent_offset = __shfl_sync(0xFFFFFFFF, recent_offset, 0);
        len_off = __shfl_sync(0xFFFFFFFF, len_off, 0);
    }

    // ── Per-block trailing literals (at 64KB boundary) ──
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    {
        uint32_t block_end = block_dst_start + BLOCK_SIZE;
        if (block_end > dst_end_abs) block_end = dst_end_abs;
        uint32_t block_trailing = (block_end > dst_pos) ? (block_end - dst_pos) : 0;
        if (mode == 0) {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                    uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                    dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
                }
        } else {
            for (uint32_t i = lane; i < block_trailing; i += 32)
                if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size)
                    dst[dst_pos + i] = lit[lit_pos + i];
        }
        __syncwarp();
        dst_pos += block_trailing;
        lit_pos += block_trailing;
    }

    // ── Advance to block 2 ──
    block_iter++;
    if (block_iter >= 2) break;
    block_cmd_end = cmd_size;
    off32_cur = off32_raw2;
    off32_count_cur = off32_count2;
    off32_pos = 0;
    block_dst_start = dst_pos;
    }

    // ── Final trailing literals ──
    __syncwarp();
    dst_pos = __shfl_sync(0xFFFFFFFF, dst_pos, 0);
    lit_pos = __shfl_sync(0xFFFFFFFF, lit_pos, 0);
    uint32_t trailing = (lit_size > lit_pos) ? (lit_size - lit_pos) : 0;
    if (mode == 0) {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs && lit_pos + i < lit_size) {
                uint32_t match_src = (uint32_t)((int32_t)(dst_pos + i) + recent_offset);
                dst[dst_pos + i] = lit[lit_pos + i] + dst[match_src];
            }
    } else {
        for (uint32_t i = lane; i < trailing; i += 32)
            if (dst_pos + i < dst_end_abs)
                dst[dst_pos + i] = lit[lit_pos + i];
    }
}

// ── Type 0 entropy header parser ───────────────────────────────
// Reads a 2- or 3-byte big-endian size prefix from the compressed stream.
__device__ uint32_t parseType0Header(const uint8_t* &src) {
    if (src[0] >= 0x80) {
        uint32_t sz = (((uint32_t)src[0] << 8) | src[1]) & 0xFFF;
        src += 2;
        return sz;
    } else {
        uint32_t sz = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
        src += 3;
        return sz;
    }
}

// ══════════════════════════════════════════════════════════════════
//  tANS decode support — inlined from gpu_tans_decode_kernel.cu
// ══════════════════════════════════════════════════════════════════

// ── tANS chunk descriptor ────────────────────────────────────────
struct TansDecChunkDesc {
    uint32_t src_offset;
    uint32_t src_size;
    uint32_t dst_offset;
    uint32_t dst_size;
};

// ── LUT entry — 8 bytes, matches CPU TansLutEnt ─────────────────
struct TansLutEnt {
    uint32_t x;
    uint8_t  bits_x;
    uint8_t  symbol;
    uint16_t w;
};

static constexpr uint32_t TANS_OK                  = 0;
static constexpr uint32_t TANS_ERR_SRC_TRUNCATED   = 1;
static constexpr uint32_t TANS_ERR_BAD_TABLE       = 2;
static constexpr uint32_t TANS_ERR_BAD_LOG_BITS    = 3;
static constexpr uint32_t TANS_ERR_BAD_WEIGHTS     = 4;
static constexpr uint32_t TANS_ERR_LUT_FAILED      = 5;
static constexpr uint32_t TANS_ERR_STREAM_MISMATCH = 6;
static constexpr uint32_t TANS_ERR_STATE_RANGE     = 7;
static constexpr uint32_t TANS_ERR_DST_OVERFLOW    = 8;

static constexpr uint32_t MAX_LUT_ENTRIES = 4096;

// ── Golomb-Rice decode tables ────────────────────────────────────
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

// ── MSB-first bit reader for table headers ──────────────────────

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

__device__ uint32_t ilog2(uint32_t v) {
    return 31 - __clz(v);
}

// ── Golomb-Rice length decoding ─────────────────────────────────

struct GolombRiceBR {
    const uint8_t* p;
    const uint8_t* p_end;
    uint32_t bit_pos;
};

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

    if (dst > dst_end) {
        uint32_t n = (uint32_t)(dst - dst_end);
        for (uint32_t i = 0; i < n; i++)
            v &= v - 1;
    }

    uint32_t bp = 0;
    if ((v & 1) == 0) {
        p--;
        uint32_t q = __ffs(v) - 1;
        if (v == 0) q = 8;
        bp = 8 - q;
    }
    br.p = p;
    br.bit_pos = bp;
    return true;
}

// ── Range conversion ────────────────────────────────────────────

struct HuffRange {
    uint16_t symbol;
    uint16_t num;
};

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

// ── TansData intermediate ───────────────────────────────────────

struct TansData {
    uint32_t a_used;
    uint32_t b_used;
    uint8_t  a[256];
    uint32_t b[256];
};

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

// ── Phase 1: decodeTable ────────────────────────────────────────

__device__ uint32_t decodeTable(
    BitReader& br, uint32_t log_table_bits, TansData& td
) {
    if (log_table_bits < 8 || log_table_bits > 12)
        return TANS_ERR_BAD_LOG_BITS;

    uint32_t L = 1u << log_table_bits;

    brRefill(br);

    if (brReadBitNoRefill(br) != 0) {
        uint32_t q = brReadBitsNoRefill(br, 3);
        uint32_t num_symbols = brReadBitsNoRefill(br, 8) + 1;
        if (num_symbols < 2) return TANS_ERR_BAD_TABLE;

        uint32_t fluff_init = brReadFluff(br, num_symbols);
        uint32_t total_rice = fluff_init + num_symbols;
        if (total_rice > 512) return TANS_ERR_BAD_TABLE;

        uint8_t rice[512 + 16];

        GolombRiceBR br2;
        br2.p_end = br.p_end;
        br2.bit_pos = (uint32_t)((br.bit_pos - 24) & 7);
        uint32_t step_back = (uint32_t)((24 - br.bit_pos + 7) >> 3);
        br2.p = br.p - step_back;

        if (!decodeGolombRiceLengths(rice, total_rice, br2))
            return TANS_ERR_BAD_TABLE;
        memset(rice + total_rice, 0, 16);

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

    // Sparse/explicit path
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

    sortBytes(td.a, a_used);
    sortU32(td.b, b_used);

    return TANS_OK;
}

// ── Phase 2: initLut ────────────────────────────────────────────

__device__ uint32_t initLut(
    const TansData& td, uint32_t log_table_bits, TansLutEnt* lut
) {
    int32_t L = 1 << log_table_bits;
    uint32_t L_u = (uint32_t)L;
    int32_t a_used = (int32_t)td.a_used;
    if (a_used > L) return TANS_ERR_LUT_FAILED;

    uint32_t slots_left = (uint32_t)(L - a_used);
    uint32_t sa = slots_left >> 2;

    uint32_t pointers[4];
    pointers[0] = 0;
    uint32_t sb = sa + ((slots_left & 3) > 0 ? 1 : 0);
    pointers[1] = sb;
    sb += sa + ((slots_left & 3) > 1 ? 1 : 0);
    pointers[2] = sb;
    sb += sa + ((slots_left & 3) > 2 ? 1 : 0);
    pointers[3] = sb;

    {
        uint32_t singles_start = slots_left;
        uint8_t bits_x = (uint8_t)log_table_bits;
        uint32_t x = (1u << log_table_bits) - 1;
        TansLutEnt le;
        le.x = x; le.bits_x = bits_x; le.symbol = 0; le.w = 0;

        for (int32_t i = 0; i < a_used; i++) {
            uint32_t idx = singles_start + (uint32_t)i;
            if (idx >= L_u) return TANS_ERR_LUT_FAILED;
            le.symbol = td.a[i];
            lut[idx] = le;
        }
    }

    int32_t weights_sum = 0;
    for (uint32_t bi = 0; bi < td.b_used; bi++) {
        uint32_t val = td.b[bi];
        int32_t weight = (int32_t)(val & 0xFFFF);
        if (weight < 1) return TANS_ERR_LUT_FAILED;
        int32_t symbol = (int32_t)(val >> 16);

        if (weight > 4) {
            uint32_t sym_bits = ilog2((uint32_t)weight);
            int32_t bits_per_symbol = (int32_t)log_table_bits - (int32_t)sym_bits;
            if (bits_per_symbol < 0) return TANS_ERR_LUT_FAILED;

            TansLutEnt le;
            le.symbol = (uint8_t)symbol;
            le.bits_x = (uint8_t)bits_per_symbol;
            le.x = (1u << bits_per_symbol) - 1;
            le.w = (uint16_t)((L - 1) & (weight << bits_per_symbol));
            int32_t what_to_add = 1 << bits_per_symbol;
            int32_t upper_slot_count = (1 << (sym_bits + 1)) - weight;
            if (upper_slot_count < 0) return TANS_ERR_LUT_FAILED;

            for (uint32_t j = 0; j < 4; j++) {
                uint32_t dst_idx = pointers[j];
                int32_t quarter_weight = (weight + ((weights_sum - (int32_t)j - 1) & 3)) >> 2;

                if (upper_slot_count >= quarter_weight) {
                    int32_t n = quarter_weight;
                    while (n != 0) {
                        if (dst_idx >= L_u) return TANS_ERR_LUT_FAILED;
                        lut[dst_idx] = le;
                        dst_idx++;
                        le.w = (uint16_t)((uint32_t)le.w + (uint32_t)what_to_add);
                        n--;
                    }
                    upper_slot_count -= quarter_weight;
                } else {
                    int32_t n = upper_slot_count;
                    while (n != 0) {
                        if (dst_idx >= L_u) return TANS_ERR_LUT_FAILED;
                        lut[dst_idx] = le;
                        dst_idx++;
                        le.w = (uint16_t)((uint32_t)le.w + (uint32_t)what_to_add);
                        n--;
                    }
                    bits_per_symbol--;
                    what_to_add >>= 1;
                    le.bits_x = (uint8_t)bits_per_symbol;
                    le.w = 0;
                    le.x >>= 1;
                    n = quarter_weight - upper_slot_count;
                    while (n != 0) {
                        if (dst_idx >= L_u) return TANS_ERR_LUT_FAILED;
                        lut[dst_idx] = le;
                        dst_idx++;
                        le.w = (uint16_t)((uint32_t)le.w + (uint32_t)what_to_add);
                        n--;
                    }
                    upper_slot_count = weight;
                }
                pointers[j] = dst_idx;
            }
        } else {
            uint32_t bits_val = ((1u << weight) - 1) << ((uint32_t)weights_sum & 3);
            bits_val |= bits_val >> 4;
            int32_t n = weight;
            int32_t ww = weight;
            while (n != 0) {
                uint32_t idx = __ffs(bits_val) - 1;
                if (idx > 3) return TANS_ERR_LUT_FAILED;
                bits_val &= bits_val - 1;
                uint32_t dst_idx = pointers[idx];
                if (dst_idx >= L_u) return TANS_ERR_LUT_FAILED;

                uint32_t weight_bits = ilog2((uint32_t)ww);
                uint32_t bps = log_table_bits - weight_bits;
                TansLutEnt le;
                le.symbol = (uint8_t)symbol;
                le.bits_x = (uint8_t)bps;
                le.x = (1u << bps) - 1;
                le.w = (uint16_t)((L - 1) & (ww << bps));
                lut[dst_idx] = le;
                ww++;
                pointers[idx] = dst_idx + 1;
                n--;
            }
        }
        weights_sum += weight;
    }

    return TANS_OK;
}

// ── Phase 3: decode — 5-state interleaved hot loop ──────────────

__device__ uint32_t bswap32(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000u);
}

__device__ uint32_t readLE32(const uint8_t* p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

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

__device__ __forceinline__ void refillForward(
    const uint8_t*& ptr_f, uint32_t& bits_f, int32_t& bitpos_f
) {
    uint32_t fw = readLE32(ptr_f);
    bits_f |= fw << bitpos_f;
    ptr_f += (31 - bitpos_f) >> 3;
    bitpos_f |= 24;
}

__device__ __forceinline__ void refillBackward(
    const uint8_t*& ptr_b, uint32_t& bits_b, int32_t& bitpos_b
) {
    uint32_t bw = readLE32(ptr_b - 4);
    bits_b |= bswap32(bw) << bitpos_b;
    ptr_b -= (31 - bitpos_b) >> 3;
    bitpos_b |= 24;
}

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

    for (;;) {
        if (ptr_f > src_end_orig) return TANS_ERR_SRC_TRUNCATED;
        refillForward(ptr_f, bits_f, bitpos_f);

        *dst++ = decodeOneSymbol(lut, state0, bits_f, bitpos_f, lut_mask);
        if (dst >= dst_end) break;

        *dst++ = decodeOneSymbol(lut, state1, bits_f, bitpos_f, lut_mask);
        if (dst >= dst_end) break;

        if (ptr_f > src_end_orig) return TANS_ERR_SRC_TRUNCATED;
        refillForward(ptr_f, bits_f, bitpos_f);

        *dst++ = decodeOneSymbol(lut, state2, bits_f, bitpos_f, lut_mask);
        if (dst >= dst_end) break;

        *dst++ = decodeOneSymbol(lut, state3, bits_f, bitpos_f, lut_mask);
        if (dst >= dst_end) break;

        if (ptr_f > src_end_orig) return TANS_ERR_SRC_TRUNCATED;
        refillForward(ptr_f, bits_f, bitpos_f);

        *dst++ = decodeOneSymbol(lut, state4, bits_f, bitpos_f, lut_mask);
        if (dst >= dst_end) break;

        if (ptr_b < src_start) return TANS_ERR_SRC_TRUNCATED;
        refillBackward(ptr_b, bits_b, bitpos_b);

        *dst++ = decodeOneSymbol(lut, state0, bits_b, bitpos_b, lut_mask);
        if (dst >= dst_end) break;

        *dst++ = decodeOneSymbol(lut, state1, bits_b, bitpos_b, lut_mask);
        if (dst >= dst_end) break;

        if (ptr_b < src_start) return TANS_ERR_SRC_TRUNCATED;
        refillBackward(ptr_b, bits_b, bitpos_b);

        *dst++ = decodeOneSymbol(lut, state2, bits_b, bitpos_b, lut_mask);
        if (dst >= dst_end) break;

        *dst++ = decodeOneSymbol(lut, state3, bits_b, bitpos_b, lut_mask);
        if (dst >= dst_end) break;

        if (ptr_b < src_start) return TANS_ERR_SRC_TRUNCATED;
        refillBackward(ptr_b, bits_b, bitpos_b);

        *dst++ = decodeOneSymbol(lut, state4, bits_b, bitpos_b, lut_mask);
        if (dst >= dst_end) break;
    }

done:
    intptr_t ptr_diff = (intptr_t)ptr_b - (intptr_t)ptr_f;
    intptr_t adjust = (intptr_t)(bitpos_f >> 3) + (intptr_t)(bitpos_b >> 3);
    if (ptr_diff + adjust != 0) return TANS_ERR_STREAM_MISMATCH;

    uint32_t states_or = state0 | state1 | state2 | state3 | state4;
    if ((states_or & ~0xFFu) != 0) return TANS_ERR_STATE_RANGE;

    dst_end[0] = (uint8_t)state0;
    dst_end[1] = (uint8_t)state1;
    dst_end[2] = (uint8_t)state2;
    dst_end[3] = (uint8_t)state3;
    dst_end[4] = (uint8_t)state4;

    return TANS_OK;
}

// ── Top-level tANS chunk decode ─────────────────────────────────

__device__ uint32_t decodeTansChunk(
    const uint8_t* __restrict__ src_buf,
    uint8_t*       __restrict__ dst_buf,
    const TansDecChunkDesc& desc,
    TansLutEnt* lut
) {
    const uint8_t* src = src_buf + desc.src_offset;
    uint32_t src_size = desc.src_size;
    uint32_t dst_size = desc.dst_size;

    if (src_size < 8 || dst_size < 5)
        return TANS_ERR_SRC_TRUNCATED;

    const uint8_t* src_end_orig = src + src_size;
    const uint8_t* src_end = src_end_orig;

    BitReader br;
    br.p = src;
    br.p_end = src_end;
    br.bits = 0;
    br.bit_pos = 24;
    brRefill(br);

    if (brReadBitNoRefill(br) != 0)
        return TANS_ERR_BAD_TABLE;

    uint32_t log_table_bits = brReadBitsNoRefill(br, 2) + 8;

    TansData td;
    td.a_used = 0;
    td.b_used = 0;
    uint32_t err = decodeTable(br, log_table_bits, td);
    if (err != TANS_OK) return err;

    int32_t byte_rewind = (24 - br.bit_pos) / 8;
    src = br.p - byte_rewind;
    const uint8_t* src_start_post_table = src;

    if (src >= src_end || (src_end - src) < 8)
        return TANS_ERR_SRC_TRUNCATED;

    int32_t L = 1 << log_table_bits;

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

    err = initLut(td, log_table_bits, lut);
    if (err != TANS_OK) return err;

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

    bits_f |= readLE32(src) << bitpos_f;
    src += (31 - bitpos_f) >> 3;
    bitpos_f |= 24;

    uint32_t state4 = bits_f & lut_mask;
    bits_f >>= ltb;  bitpos_f -= ltb;

    const uint8_t* pos_f = src - (bitpos_f >> 3);
    const uint8_t* pos_b = src_end + (bitpos_b >> 3);
    int32_t bp_f = bitpos_f & 7;
    int32_t bp_b = bitpos_b & 7;

    uint8_t* tans_dst = dst_buf + desc.dst_offset;
    uint8_t* dst_end_ptr = tans_dst + dst_size - 5;

    return decode5State(
        lut, lut_mask,
        tans_dst, dst_end_ptr,
        pos_f, pos_b,
        bits_f, bits_b,
        bp_f, bp_b,
        state0, state1, state2, state3, state4,
        src_start_post_table, src_end_orig
    );
}

// ══════════════════════════════════════════════════════════════════
//  End tANS decode support
// ══════════════════════════════════════════════════════════════════

// ── Sub-chunk header parser + decoder dispatch ─────────────────
// GPU-side readLzTable + decode for one sub-chunk.
// Parses Type 0 or Type 1 (tANS) entropy headers to locate streams,
// then dispatches to decodeSubChunkL1 (fast path) or decodeSubChunk
// (general path).
__device__ void parseAndDecodeSubChunk(
    const uint8_t* sc_src,
    uint32_t sc_comp_size,
    uint32_t sc_decomp_size,
    uint8_t* dst,
    uint32_t dst_offset,
    uint32_t base_offset,
    uint32_t mode,
    TansLutEnt* tans_lut,
    uint8_t* tans_scratch_chunk
) {
    const int lane = threadIdx.x & 31;
    const uint8_t* src = sc_src;
    const uint8_t* src_end = sc_src + sc_comp_size;

    uint32_t initial_copy = 0;
    if (base_offset == 0) {
        // First 8 bytes are raw literals — copy directly to output
        if (lane < 8) dst[dst_offset + lane] = src[lane];
        __syncwarp();
        src += 8;
        initial_copy = 8;
    }

    // Parse each stream header on lane 0, then broadcast the src offset
    // to all lanes so every lane can compute derived pointers.

    // Literal stream — detect Type 0 (memcpy) vs Type 1 (tANS)
    const uint8_t* lit_ptr = src;
    uint32_t lit_size = 0;
    uint32_t lit_is_tans = 0;
    if (lane == 0) {
        uint32_t chunk_type = (src[0] >> 4) & 0x7;
        if (chunk_type == 0) {
            // Type 0: memcpy literals
            lit_size = parseType0Header(src);
            lit_ptr = src;
            src += lit_size;
        } else if (chunk_type == 1 && tans_scratch_chunk != nullptr) {
            // Type 1: tANS-encoded literals
            uint32_t tans_comp_size, tans_dst_size;
            if (src[0] >= 0x80) {
                // Compact 3-byte header
                uint32_t bits = ((uint32_t)src[0] << 16) | ((uint32_t)src[1] << 8) | src[2];
                tans_comp_size = bits & 0x3FF;
                tans_dst_size = tans_comp_size + ((bits >> 10) & 0x3FF) + 1;
                src += 3;
            } else {
                // Non-compact 5-byte header
                uint32_t bits = ((uint32_t)src[1] << 24) | ((uint32_t)src[2] << 16) | ((uint32_t)src[3] << 8) | src[4];
                tans_comp_size = bits & 0x3FFFF;
                tans_dst_size = ((((bits >> 18) | ((uint32_t)src[0] << 14)) & 0x3FFFF)) + 1;
                src += 5;
            }
            // Build a TansDecChunkDesc pointing to current src position
            TansDecChunkDesc td;
            td.src_offset = 0;
            td.src_size = tans_comp_size;
            td.dst_offset = 0;
            td.dst_size = tans_dst_size;
            decodeTansChunk(src, tans_scratch_chunk, td, tans_lut);
            lit_ptr = tans_scratch_chunk;
            lit_size = tans_dst_size;
            src += tans_comp_size;
            lit_is_tans = 1;
        } else {
            // Unknown type — fall back to Type 0 parsing
            lit_size = parseType0Header(src);
            lit_ptr = src;
            src += lit_size;
        }
    }
    lit_size = __shfl_sync(0xFFFFFFFF, lit_size, 0);
    lit_is_tans = __shfl_sync(0xFFFFFFFF, lit_is_tans, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        if (lit_is_tans) {
            // tANS: lit_ptr already points to tans_scratch_chunk (same for all lanes)
            lit_ptr = tans_scratch_chunk;
        } else {
            lit_ptr = src - lit_size;
        }
    }

    // Command stream (Type 0)
    const uint8_t* cmd_ptr;
    uint32_t cmd_size = 0;
    if (lane == 0) {
        cmd_size = parseType0Header(src);
        cmd_ptr = src;
        src += cmd_size;
    }
    cmd_size = __shfl_sync(0xFFFFFFFF, cmd_size, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        cmd_ptr = src - cmd_size;
    }

    // cmd_stream2_offset
    uint32_t cmd_stream2_offset = cmd_size;
    if (lane == 0 && sc_decomp_size > 0x10000) {
        uint16_t v; memcpy(&v, src, 2);
        cmd_stream2_offset = v;
        src += 2;
    }
    cmd_stream2_offset = __shfl_sync(0xFFFFFFFF, cmd_stream2_offset, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
    }

    // Off16 stream
    const uint8_t* off16_raw;
    uint32_t off16_count = 0;
    if (lane == 0) {
        uint16_t cnt; memcpy(&cnt, src, 2);
        off16_count = cnt;
        off16_raw = src + 2;
        src += 2 + off16_count * 2;
    }
    off16_count = __shfl_sync(0xFFFFFFFF, off16_count, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        off16_raw = src - off16_count * 2;
    }

    // Off32 stream sizes
    uint32_t off32_count1 = 0, off32_count2 = 0;
    const uint8_t* off32_raw1;
    const uint8_t* off32_raw2;
    const uint8_t* len_stream;
    uint32_t len_avail = 0;

    if (lane == 0) {
        uint32_t tmp = (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16);
        src += 3;
        if (tmp != 0) {
            off32_count1 = tmp >> 12;
            off32_count2 = tmp & 0xFFF;
            if (off32_count1 == 4095) { uint16_t v; memcpy(&v, src, 2); off32_count1 = v; src += 2; }
            if (off32_count2 == 4095) { uint16_t v; memcpy(&v, src, 2); off32_count2 = v; src += 2; }
            off32_raw1 = src;
            src += off32_count1 * 3;
            off32_raw2 = src;
            src += off32_count2 * 3;
        } else {
            off32_raw1 = src;
            off32_raw2 = src;
        }
        len_stream = src;
        len_avail = (uint32_t)((uintptr_t)src_end - (uintptr_t)src);
    }
    off32_count1 = __shfl_sync(0xFFFFFFFF, off32_count1, 0);
    off32_count2 = __shfl_sync(0xFFFFFFFF, off32_count2, 0);
    len_avail = __shfl_sync(0xFFFFFFFF, len_avail, 0);
    {
        uint32_t so = (uint32_t)((uintptr_t)src - (uintptr_t)sc_src);
        so = __shfl_sync(0xFFFFFFFF, so, 0);
        src = sc_src + so;
        len_stream = src;
        off32_raw2 = src - off32_count2 * 3;
        off32_raw1 = off32_raw2 - off32_count1 * 3;
    }

    if (mode == 1 && off32_count1 == 0 && off32_count2 == 0) {
        decodeSubChunkL1(
            cmd_ptr, cmd_size,
            lit_ptr, lit_size,
            off16_raw, off16_count,
            len_stream, len_avail,
            dst, sc_decomp_size, initial_copy,
            dst_offset
        );
    } else {
        decodeSubChunk(
            cmd_ptr, cmd_size,
            lit_ptr, lit_size,
            off16_raw, off16_count,
            off32_raw1, off32_count1,
            off32_raw2, off32_count2,
            len_stream, len_avail,
            dst, sc_decomp_size, initial_copy, cmd_stream2_offset,
            dst_offset, mode
        );
    }
}

// ── Production kernel ──────────────────────────────────────────
// Full GPU L1 kernel: 1 block per SC group, parses raw compressed
// chunks on-GPU. 2 warps per block for SM occupancy.
extern "C" __global__ void __launch_bounds__(64, 1) slzFullDecompressL1Kernel(
    const uint8_t* __restrict__ compressed,
    const SlzChunkDesc* __restrict__ chunks,
    uint8_t* __restrict__ dst,
    uint32_t chunks_per_group,
    uint32_t total_chunks,
    uint32_t sub_chunk_cap,
    uint8_t* __restrict__ tans_scratch
) {
    // Shared memory for tANS LUT: one per warp, 2048 entries * 8 bytes = 16KB each = 32KB total
    __shared__ TansLutEnt s_tans_lut[2][2048];

    // 2 warps per block: warp 0 = threadIdx.y==0, warp 1 = threadIdx.y==1
    const uint32_t warp_id = threadIdx.y;
    const uint32_t group_id = blockIdx.x * 2 + warp_id;
    const int lane = threadIdx.x & 31;
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
            for (uint32_t i = lane; i < ch.decomp_size; i += 32)
                dst[ch.dst_offset + i] = src[i];
            __syncwarp();
            continue;
        }

        // Memset chunk
        if (ch.flags & 2) {
            for (uint32_t i = lane; i < ch.decomp_size; i += 32)
                dst[ch.dst_offset + i] = ch.memset_fill;
            __syncwarp();
            continue;
        }

        // LZ-compressed chunk: iterate sub-chunks
        const uint8_t* chunk_src = compressed + ch.src_offset;
        uint32_t sc_dst_off = ch.dst_offset;
        uint32_t sc_remaining = ch.decomp_size;
        uint8_t* chunk_tans_scratch = tans_scratch ? (tans_scratch + chunk_idx * 65536) : nullptr;

        while (sc_remaining > 0) {
            uint32_t sc_size = sc_remaining;
            if (sc_size > sub_chunk_cap) sc_size = sub_chunk_cap;

            // Parse 3-byte sub-chunk header (big-endian)
            uint32_t chunkhdr = 0;
            if (lane == 0)
                chunkhdr = ((uint32_t)chunk_src[0] << 16) | ((uint32_t)chunk_src[1] << 8) | chunk_src[2];
            chunkhdr = __shfl_sync(0xFFFFFFFF, chunkhdr, 0);

            if (!(chunkhdr & 0x800000)) {
                // Non-LZ sub-chunk (entropy-only) — skip for now
                break;
            }

            uint32_t sc_comp_size = chunkhdr & 0x7FFFF;
            uint32_t sc_mode = (chunkhdr >> 19) & 0xF;
            const uint8_t* sc_payload = chunk_src + 3;

            if (sc_comp_size < sc_size) {
                // base_offset is the relative offset within the chunk
                // (0 for the first sub-chunk, triggers initial 8-byte copy)
                uint32_t base_offset_val = sc_dst_off - ch.dst_offset;

                parseAndDecodeSubChunk(
                    sc_payload, sc_comp_size, sc_size,
                    dst, sc_dst_off, base_offset_val, sc_mode,
                    s_tans_lut[warp_id], chunk_tans_scratch
                );
            } else {
                // Uncompressed sub-chunk: copy
                for (uint32_t i = lane; i < sc_size; i += 32)
                    dst[sc_dst_off + i] = sc_payload[i];
            }
            __syncwarp();

            chunk_src += 3 + sc_comp_size;
            sc_dst_off += sc_size;
            sc_remaining -= sc_size;
        }
    }
}
