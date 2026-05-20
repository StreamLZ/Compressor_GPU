// CPU reference Huffman codec — oracle for GPU port validation.
//
// Algorithm:
//   * 256-bin histogram → standard binary heap tree-build → codeword lengths.
//   * Height-limit codes to MAX_CODE_LEN bits via simple iterative
//     promote-shortest pass (loses tiny bit of ratio at extreme distributions
//     but keeps the decode LUT bounded to 2^MAX_CODE_LEN entries).
//   * Canonical codes assigned in (length, symbol) order.
//   * Encoder packs codewords MSB-first into output bytes (bit 7 of byte 0 = first
//     bit emitted).
//   * Decoder builds a 2^MAX_CODE_LEN-entry fast LUT (symbol + length) and
//     consumes bits with a u64 bit-buffer.
//
// Single-stream first. 4-stream wrapper added later.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// 11 bits matches zstd's default Huffman max length. LUT = 2048 × 2-byte
// entries = 4 KB, fits comfortably in shared memory.
#ifndef MAX_CODE_LEN
#define MAX_CODE_LEN 11
#define LUT_SIZE     (1u << MAX_CODE_LEN)  // 2048
#endif

// ─── Bit writer ─────────────────────────────────────────────────────────
typedef struct {
    uint8_t *buf;
    size_t   capacity;
    size_t   byte_pos;
    uint64_t bit_buf;
    int      bit_count;  // bits currently buffered (0..63)
} BitWriter;

static void bw_init(BitWriter *bw, uint8_t *buf, size_t capacity) {
    bw->buf = buf; bw->capacity = capacity;
    bw->byte_pos = 0; bw->bit_buf = 0; bw->bit_count = 0;
}

static int bw_write(BitWriter *bw, uint32_t bits, int n) {
    // bits is right-aligned in `bits`. Pack MSB-first into output stream.
    if (n == 0) return 0;
    if (bw->bit_count + n > 64) return -1;
    bw->bit_buf = (bw->bit_buf << n) | (bits & ((1u << n) - 1u));
    bw->bit_count += n;
    while (bw->bit_count >= 8) {
        if (bw->byte_pos >= bw->capacity) return -1;
        bw->bit_count -= 8;
        bw->buf[bw->byte_pos++] = (uint8_t)((bw->bit_buf >> bw->bit_count) & 0xFF);
    }
    return 0;
}

static int bw_flush(BitWriter *bw) {
    if (bw->bit_count > 0) {
        if (bw->byte_pos >= bw->capacity) return -1;
        bw->buf[bw->byte_pos++] = (uint8_t)((bw->bit_buf << (8 - bw->bit_count)) & 0xFF);
        bw->bit_count = 0;
    }
    return 0;
}

// ─── Code-length builder ────────────────────────────────────────────────
// Returns max code length encountered (before height-limit).
typedef struct { uint32_t weight; int parent; } HeapNode;

static int build_code_lengths(const uint32_t hist[256], uint8_t code_lengths[256]) {
    // Build tree: nodes 0..255 are leaves, 256+ are internals.
    // We need 256 leaves + up to 255 internals = up to 511 nodes total.
    HeapNode nodes[512];
    int n_active = 0;
    int idx[256];  // node index per symbol (when active)

    for (int i = 0; i < 512; i++) { nodes[i].weight = 0; nodes[i].parent = -1; }
    memset(code_lengths, 0, 256);

    // Initialize leaves with non-zero histogram counts.
    int symbols_used = 0;
    for (int s = 0; s < 256; s++) {
        if (hist[s] > 0) {
            nodes[s].weight = hist[s];
            idx[symbols_used++] = s;
        }
    }

    if (symbols_used == 0) return 0;
    if (symbols_used == 1) {
        // Edge case: single symbol, assign length 1.
        code_lengths[idx[0]] = 1;
        return 1;
    }

    n_active = symbols_used;
    int next_node = 256;

    // Use a tiny priority queue: linear scan for smallest two (n_active never > 511).
    int active[512];
    for (int i = 0; i < n_active; i++) active[i] = idx[i];

    while (n_active > 1) {
        // Find two smallest weights.
        int a_pos = 0, b_pos = 1;
        if (nodes[active[a_pos]].weight > nodes[active[b_pos]].weight) {
            int t = a_pos; a_pos = b_pos; b_pos = t;
        }
        for (int i = 2; i < n_active; i++) {
            uint32_t w = nodes[active[i]].weight;
            if (w < nodes[active[a_pos]].weight) {
                b_pos = a_pos; a_pos = i;
            } else if (w < nodes[active[b_pos]].weight) {
                b_pos = i;
            }
        }
        int a = active[a_pos], b = active[b_pos];

        // Combine into new node.
        nodes[next_node].weight = nodes[a].weight + nodes[b].weight;
        nodes[a].parent = next_node;
        nodes[b].parent = next_node;

        // Remove a, b from active list; add next_node.
        int new_pos = (a_pos < b_pos) ? a_pos : b_pos;
        int old_pos = (a_pos < b_pos) ? b_pos : a_pos;
        active[new_pos] = next_node;
        active[old_pos] = active[--n_active];
        next_node++;
    }

    // Walk parents from each leaf to compute its depth = code length.
    int max_len = 0;
    for (int i = 0; i < symbols_used; i++) {
        int s = idx[i];
        int depth = 0;
        int n = s;
        while (nodes[n].parent != -1) {
            depth++;
            n = nodes[n].parent;
        }
        code_lengths[s] = (uint8_t)depth;
        if (depth > max_len) max_len = depth;
    }

    return max_len;
}

// Kraft-preserving height-limit. Reduces any code-lengths > limit down to
// limit, then rebalances by lengthening the lowest-weight short codes until
// the Kraft inequality is exactly satisfied. Output is a valid prefix code
// with all lengths ≤ limit. Not optimal (package-merge would be) but correct.
//
// Uses a 30-bit fixed-point Kraft sum: each symbol of length L contributes
// (1 << (30 - L)). A valid prefix code has sum exactly == (1 << 30).
static void height_limit(const uint32_t hist[256], uint8_t code_lengths[256], int limit) {
    const uint32_t TARGET = 1u << 30;
    uint32_t sum = 0;
    for (int s = 0; s < 256; s++) {
        if (code_lengths[s] > 0) sum += (1u << (30 - code_lengths[s]));
    }
    // Tree-built lengths form a valid Huffman code → sum should equal TARGET.

    // Demote any code > limit down to limit. Kraft sum INCREASES.
    for (int s = 0; s < 256; s++) {
        if (code_lengths[s] > limit) {
            sum -= (1u << (30 - code_lengths[s]));
            code_lengths[s] = (uint8_t)limit;
            sum += (1u << (30 - limit));
        }
    }

    // Rebalance: while sum > TARGET, lengthen the lowest-weight code that
    // can still be lengthened (len < limit). Each +1 in length halves that
    // symbol's contribution to sum.
    while (sum > TARGET) {
        int best = -1;
        uint32_t best_w = 0xFFFFFFFFu;
        for (int s = 0; s < 256; s++) {
            if (code_lengths[s] > 0 && code_lengths[s] < limit && hist[s] < best_w) {
                best_w = hist[s]; best = s;
            }
        }
        if (best < 0) break;  // can't rebalance further (shouldn't happen if limit ≥ ceil(log2(n)))
        sum -= (1u << (30 - code_lengths[best]));
        code_lengths[best]++;
        sum += (1u << (30 - code_lengths[best]));
    }
}

// ─── Canonical code assignment ──────────────────────────────────────────
// Given code lengths per symbol, assign canonical codes. Sort by
// (length asc, symbol asc); first symbol = all-zero code of its length;
// each subsequent = prev+1, extended with zeros on length increase.
static void assign_canonical_codes(const uint8_t code_lengths[256],
                                   uint32_t codes_out[256]) {
    // Count codes of each length.
    uint32_t length_count[MAX_CODE_LEN + 1] = {0};
    for (int s = 0; s < 256; s++) length_count[code_lengths[s]]++;

    // Smallest code per length.
    uint32_t next_code[MAX_CODE_LEN + 2] = {0};
    uint32_t code = 0;
    length_count[0] = 0;
    for (int len = 1; len <= MAX_CODE_LEN + 1; len++) {
        code = (code + length_count[len - 1]) << 1;
        next_code[len] = code;
    }

    for (int s = 0; s < 256; s++) {
        int len = code_lengths[s];
        if (len != 0) {
            codes_out[s] = next_code[len]++;
        } else {
            codes_out[s] = 0;
        }
    }
}

// ─── Build decode LUT (X2 dual-symbol when both fit in MAX_CODE_LEN) ───
// LUT entry (uint32_t):
//   bits  0-7  : sym1
//   bits  8-15 : sym2 (valid iff num_syms == 2)
//   bits 16-23 : total_len (= L1 if num_syms==1, else L1+L2)
//   bits 24-31 : num_syms (1 or 2)
//
// Build:
//   1. Fill all entries with single-symbol info (sym1, L1, 1).
//   2. For each pair (S1, S2) where L1 + L2 ≤ MAX_CODE_LEN, overwrite
//      the entries whose prefix is C1<<...|C2<<... with (sym1, sym2,
//      L1+L2, 2). Pass 2 takes precedence over pass 1 for those entries.
static inline uint32_t pack_lut_entry(uint8_t sym1, uint8_t sym2,
                                      uint8_t total_len, uint8_t num_syms) {
    return ((uint32_t)num_syms << 24)
         | ((uint32_t)total_len << 16)
         | ((uint32_t)sym2 << 8)
         | ((uint32_t)sym1);
}

static void build_decode_lut(const uint8_t code_lengths[256],
                             const uint32_t codes[256],
                             uint32_t lut[LUT_SIZE]) {
    memset(lut, 0, LUT_SIZE * sizeof(uint32_t));

    // Pass 1: single-symbol entries.
    for (int s = 0; s < 256; s++) {
        int len = code_lengths[s];
        if (len == 0) continue;
        uint32_t aligned = codes[s] << (MAX_CODE_LEN - len);
        uint32_t span = 1u << (MAX_CODE_LEN - len);
        uint32_t entry = pack_lut_entry((uint8_t)s, 0, (uint8_t)len, 1);
        for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
    }

    // Pass 2: dual-symbol entries — overwrite when (L1 + L2) ≤ MAX_CODE_LEN.
    for (int s1 = 0; s1 < 256; s1++) {
        int L1 = code_lengths[s1];
        if (L1 == 0 || L1 >= MAX_CODE_LEN) continue;
        uint32_t C1 = codes[s1];
        for (int s2 = 0; s2 < 256; s2++) {
            int L2 = code_lengths[s2];
            if (L2 == 0) continue;
            int total = L1 + L2;
            if (total > MAX_CODE_LEN) continue;
            uint32_t C2 = codes[s2];
            // Compose the prefix: C1 in top L1 bits, C2 in next L2 bits.
            uint32_t aligned = (C1 << (MAX_CODE_LEN - L1))
                             | (C2 << (MAX_CODE_LEN - L1 - L2));
            uint32_t span = 1u << (MAX_CODE_LEN - total);
            uint32_t entry = pack_lut_entry((uint8_t)s1, (uint8_t)s2, (uint8_t)total, 2);
            for (uint32_t i = 0; i < span; i++) lut[aligned + i] = entry;
        }
    }
}

// ─── Encode ─────────────────────────────────────────────────────────────
// Returns bytes written (including trailing partial byte) or -1 on overflow.
static int huff_encode(const uint8_t *in, size_t n,
                       const uint8_t code_lengths[256],
                       const uint32_t codes[256],
                       uint8_t *out, size_t out_cap) {
    BitWriter bw;
    bw_init(&bw, out, out_cap);
    for (size_t i = 0; i < n; i++) {
        if (bw_write(&bw, codes[in[i]], code_lengths[in[i]]) != 0) return -1;
    }
    if (bw_flush(&bw) != 0) return -1;
    return (int)bw.byte_pos;
}

// ─── Decode ─────────────────────────────────────────────────────────────
static int huff_decode(const uint8_t *in, size_t in_bytes,
                       const uint32_t lut[LUT_SIZE],
                       uint8_t *out, size_t expected_out_bytes) {
    uint64_t bit_buf = 0;
    int bit_count = 0;
    size_t in_pos = 0;
    size_t out_pos = 0;

    while (out_pos < expected_out_bytes) {
        while (bit_count < MAX_CODE_LEN && in_pos < in_bytes) {
            bit_buf = (bit_buf << 8) | (uint64_t)in[in_pos++];
            bit_count += 8;
        }
        if (bit_count < MAX_CODE_LEN) {
            bit_buf <<= (MAX_CODE_LEN - bit_count);
            bit_count = MAX_CODE_LEN;
        }
        uint32_t idx = (uint32_t)((bit_buf >> (bit_count - MAX_CODE_LEN)) & (LUT_SIZE - 1));
        uint32_t entry = lut[idx];
        int total_len = (entry >> 16) & 0xFF;
        int num_syms  = (entry >> 24) & 0xFF;
        if (total_len == 0) return -1;
        out[out_pos++] = (uint8_t)(entry & 0xFF);
        // Second symbol only if room and entry is dual.
        if (num_syms == 2 && out_pos < expected_out_bytes) {
            out[out_pos++] = (uint8_t)((entry >> 8) & 0xFF);
            bit_count -= total_len;
        } else if (num_syms == 2) {
            // Dual entry but only 1 byte of output left — fall back to L1 only.
            // L1 = total_len - L2. We don't store L1 separately; instead the
            // single-symbol entry for the same prefix would have consumed L1.
            // For simplicity: re-lookup using a smaller window. But cheaper:
            // since this only happens at most once (last position), accept the
            // cost of consuming total_len bits and assert no underflow.
            bit_count -= total_len;
        } else {
            bit_count -= total_len;
        }
    }
    return (int)out_pos;
}

// ─── 4-stream split (zstd / nvCOMP pattern) ────────────────────────────
// Splits the input into 4 CONTIGUOUS quarters (not interleaved), encodes
// each with the same Huffman table, and prepends a 6-byte header
// (3 × u16 little-endian stream sizes; stream4 size = total - sum).
// At decode time, 4 GPU lanes/warps decode the quarters in parallel.

// Header: 3 × u24 little-endian stream sizes = 9 bytes. Stream 4 size
// derived. Supports streams up to 16 MB each (overkill for 64KB sub-chunks
// but lets the CPU reference test arbitrarily-large inputs).
#define HUFF_4S_HDR_BYTES 9

static int huff_encode_4s(const uint8_t *in, size_t n,
                          const uint8_t code_lengths[256],
                          const uint32_t codes[256],
                          uint8_t *out, size_t out_cap) {
    if (out_cap < HUFF_4S_HDR_BYTES) return -1;
    size_t q = n / 4;
    size_t boundaries[5] = { 0, q, 2*q, 3*q, n };
    size_t written = HUFF_4S_HDR_BYTES;
    uint32_t sizes[4] = {0};

    for (int s = 0; s < 4; s++) {
        size_t stream_n = boundaries[s+1] - boundaries[s];
        int sz = huff_encode(in + boundaries[s], stream_n,
                             code_lengths, codes,
                             out + written, out_cap - written);
        if (sz < 0) return -1;
        if (sz >= (1 << 24)) return -1;  // u24 overflow
        sizes[s] = (uint32_t)sz;
        written += (size_t)sz;
    }

    // Header: 3 × u24 sizes.
    for (int s = 0; s < 3; s++) {
        out[s*3 + 0] = (uint8_t)(sizes[s] & 0xFF);
        out[s*3 + 1] = (uint8_t)((sizes[s] >> 8) & 0xFF);
        out[s*3 + 2] = (uint8_t)((sizes[s] >> 16) & 0xFF);
    }
    return (int)written;
}

// ─── Word-interleaved 4-stream split ───────────────────────────────────
// Same 4-quarter split + per-stream Huffman encode as huff_encode_4s, but
// the 4 encoded streams are stored INTERLEAVED at 32-bit-word granularity:
//   [hdr][s0.w0][s1.w0][s2.w0][s3.w0][s0.w1][s1.w1]...
// so the 4 GPU decode lanes' word-w refill loads hit adjacent addresses
// and coalesce into one memory sector (vs 4 scattered sectors for the
// contiguous layout). The payload is padded to the longest stream, so a
// 12-byte header carries all 4 per-stream byte sizes explicitly (the 4th
// is no longer derivable from the total).
#ifndef HUFF_4SI_HDR_BYTES
#define HUFF_4SI_HDR_BYTES 12
#endif

static int huff_encode_4s_interleaved(const uint8_t *in, size_t n,
                                      const uint8_t code_lengths[256],
                                      const uint32_t codes[256],
                                      uint8_t *out, size_t out_cap) {
    if (out_cap < HUFF_4SI_HDR_BYTES) return -1;
    size_t q = n / 4;
    size_t boundaries[5] = { 0, q, 2*q, 3*q, n };

    uint8_t  *tmp[4]   = {0};
    uint32_t  sizes[4] = {0};
    uint32_t  words[4] = {0};
    uint32_t  max_words = 0;
    int rc = -1;

    for (int s = 0; s < 4; s++) {
        size_t stream_n = boundaries[s+1] - boundaries[s];
        size_t cap = stream_n * 2 + 64;
        tmp[s] = (uint8_t*)malloc(cap + 4);   // +4 so word padding never overruns
        if (!tmp[s]) goto done;
        int sz = huff_encode(in + boundaries[s], stream_n,
                             code_lengths, codes, tmp[s], cap);
        if (sz < 0 || sz >= (1 << 24)) goto done;
        sizes[s] = (uint32_t)sz;
        words[s] = ((uint32_t)sz + 3) / 4;
        // Zero the padding bytes of the final partial word.
        for (uint32_t p = (uint32_t)sz; p < words[s] * 4; p++) tmp[s][p] = 0;
        if (words[s] > max_words) max_words = words[s];
    }

    {
        size_t total = HUFF_4SI_HDR_BYTES + (size_t)max_words * 16;
        if (out_cap < total) goto done;

        // Header: 4 × u24 LE per-stream byte sizes.
        for (int s = 0; s < 4; s++) {
            out[s*3 + 0] = (uint8_t)(sizes[s] & 0xFF);
            out[s*3 + 1] = (uint8_t)((sizes[s] >> 8) & 0xFF);
            out[s*3 + 2] = (uint8_t)((sizes[s] >> 16) & 0xFF);
        }
        // Interleave: word w of all 4 streams, then word w+1, ...
        uint8_t *dst = out + HUFF_4SI_HDR_BYTES;
        for (uint32_t w = 0; w < max_words; w++) {
            for (int s = 0; s < 4; s++) {
                if (w < words[s]) {
                    memcpy(dst, tmp[s] + (size_t)w * 4, 4);
                } else {
                    dst[0] = dst[1] = dst[2] = dst[3] = 0;  // exhausted stream
                }
                dst += 4;
            }
        }
        rc = (int)total;
    }

done:
    for (int s = 0; s < 4; s++) free(tmp[s]);
    return rc;
}

static int huff_decode_4s_interleaved(const uint8_t *in, size_t in_bytes,
                                      const uint32_t lut[LUT_SIZE],
                                      uint8_t *out, size_t expected_out_bytes) {
    if (in_bytes < HUFF_4SI_HDR_BYTES) return -1;
    uint32_t sizes[4];
    for (int s = 0; s < 4; s++) {
        sizes[s] = (uint32_t)in[s*3 + 0]
                 | ((uint32_t)in[s*3 + 1] << 8)
                 | ((uint32_t)in[s*3 + 2] << 16);
    }
    size_t q = expected_out_bytes / 4;
    size_t boundaries[5] = { 0, q, 2*q, 3*q, expected_out_bytes };

    // De-interleave each stream into a contiguous temp, then decode.
    const uint8_t *payload = in + HUFF_4SI_HDR_BYTES;
    size_t payload_bytes = in_bytes - HUFF_4SI_HDR_BYTES;
    uint32_t max_words = (uint32_t)(payload_bytes / 16);
    for (int s = 0; s < 4; s++) {
        uint32_t nw = (sizes[s] + 3) / 4;
        if (nw > max_words) return -1;
        uint8_t *tmp = (uint8_t*)malloc((size_t)nw * 4 + 4);
        if (!tmp) return -1;
        for (uint32_t w = 0; w < nw; w++) {
            memcpy(tmp + (size_t)w * 4, payload + ((size_t)w * 4 + s) * 4, 4);
        }
        size_t stream_out = boundaries[s+1] - boundaries[s];
        int dec = huff_decode(tmp, sizes[s], lut, out + boundaries[s], stream_out);
        free(tmp);
        if (dec != (int)stream_out) return -1;
    }
    return (int)expected_out_bytes;
}

static int huff_decode_4s(const uint8_t *in, size_t in_bytes,
                          const uint32_t lut[LUT_SIZE],
                          uint8_t *out, size_t expected_out_bytes) {
    if (in_bytes < HUFF_4S_HDR_BYTES) return -1;
    uint32_t sizes[4];
    for (int s = 0; s < 3; s++) {
        sizes[s] = (uint32_t)in[s*3 + 0]
                 | ((uint32_t)in[s*3 + 1] << 8)
                 | ((uint32_t)in[s*3 + 2] << 16);
    }
    uint32_t total_streams = (uint32_t)(in_bytes - HUFF_4S_HDR_BYTES);
    if (sizes[0] + sizes[1] + sizes[2] > total_streams) return -1;
    sizes[3] = total_streams - sizes[0] - sizes[1] - sizes[2];

    size_t q = expected_out_bytes / 4;
    size_t boundaries[5] = { 0, q, 2*q, 3*q, expected_out_bytes };

    const uint8_t *stream_ptr = in + HUFF_4S_HDR_BYTES;
    for (int s = 0; s < 4; s++) {
        size_t stream_out = boundaries[s+1] - boundaries[s];
        int dec = huff_decode(stream_ptr, sizes[s], lut,
                              out + boundaries[s], stream_out);
        if (dec != (int)stream_out) return -1;
        stream_ptr += sizes[s];
    }
    return (int)expected_out_bytes;
}

// ─── N-stream split (parameterized) ─────────────────────────────────────
// Generalizes the 4-stream layout to N streams. Each stream gets a
// contiguous quarter (= n/N bytes; stream N-1 absorbs the remainder).
// Header: (N-1) × u24 LE sizes (stream N-1 size derived).

static int huff_encode_ns(int N,
                          const uint8_t *in, size_t n,
                          const uint8_t code_lengths[256],
                          const uint32_t codes[256],
                          uint8_t *out, size_t out_cap) {
    size_t hdr_bytes = (size_t)(N - 1) * 3;
    if (out_cap < hdr_bytes) return -1;
    size_t q = n / (size_t)N;
    size_t written = hdr_bytes;
    uint32_t sizes[64];  // supports up to 64 streams
    if (N > 64) return -1;

    for (int s = 0; s < N; s++) {
        size_t start = (size_t)s * q;
        size_t end   = (s == N - 1) ? n : (size_t)(s + 1) * q;
        int sz = huff_encode(in + start, end - start,
                             code_lengths, codes,
                             out + written, out_cap - written);
        if (sz < 0) return -1;
        if (sz >= (1 << 24)) return -1;
        sizes[s] = (uint32_t)sz;
        written += (size_t)sz;
    }

    for (int s = 0; s < N - 1; s++) {
        out[s*3 + 0] = (uint8_t)(sizes[s] & 0xFF);
        out[s*3 + 1] = (uint8_t)((sizes[s] >> 8) & 0xFF);
        out[s*3 + 2] = (uint8_t)((sizes[s] >> 16) & 0xFF);
    }
    return (int)written;
}

static int huff_decode_ns(int N,
                          const uint8_t *in, size_t in_bytes,
                          const uint32_t lut[LUT_SIZE],
                          uint8_t *out, size_t expected_out_bytes) {
    size_t hdr_bytes = (size_t)(N - 1) * 3;
    if (in_bytes < hdr_bytes) return -1;
    uint32_t sizes[64];
    if (N > 64) return -1;

    uint32_t sum = 0;
    for (int s = 0; s < N - 1; s++) {
        sizes[s] = (uint32_t)in[s*3 + 0]
                 | ((uint32_t)in[s*3 + 1] << 8)
                 | ((uint32_t)in[s*3 + 2] << 16);
        sum += sizes[s];
    }
    uint32_t total_streams = (uint32_t)(in_bytes - hdr_bytes);
    if (sum > total_streams) return -1;
    sizes[N - 1] = total_streams - sum;

    size_t q = expected_out_bytes / (size_t)N;
    const uint8_t *stream_ptr = in + hdr_bytes;
    for (int s = 0; s < N; s++) {
        size_t start = (size_t)s * q;
        size_t end   = (s == N - 1) ? expected_out_bytes : (size_t)(s + 1) * q;
        int dec = huff_decode(stream_ptr, sizes[s], lut, out + start, end - start);
        if (dec != (int)(end - start)) return -1;
        stream_ptr += sizes[s];
    }
    return (int)expected_out_bytes;
}

// ─── Roundtrip test ─────────────────────────────────────────────────────
static int roundtrip_test(const char *name, const uint8_t *src, size_t n) {
    uint32_t hist[256] = {0};
    for (size_t i = 0; i < n; i++) hist[src[i]]++;

    uint8_t  code_lengths[256] = {0};
    uint32_t codes[256] = {0};
    uint32_t *lut = (uint32_t*)malloc(LUT_SIZE * sizeof(uint32_t));
    if (!lut) { printf("[%s] alloc failed\n", name); return 1; }

    int max_len = build_code_lengths(hist, code_lengths);
    if (max_len > MAX_CODE_LEN) {
        height_limit(hist, code_lengths, MAX_CODE_LEN);
        max_len = 0;
        for (int s = 0; s < 256; s++) if (code_lengths[s] > max_len) max_len = code_lengths[s];
    }
    assign_canonical_codes(code_lengths, codes);
    build_decode_lut(code_lengths, codes, lut);

    uint8_t *enc = (uint8_t*)malloc(n * 2 + 16);
    uint8_t *dec = (uint8_t*)malloc(n + 16);
    int enc_bytes = huff_encode(src, n, code_lengths, codes, enc, n * 2 + 16);
    if (enc_bytes < 0) {
        printf("[%s] FAIL encode overflow\n", name);
        free(enc); free(dec); free(lut);
        return 1;
    }
    int dec_bytes = huff_decode(enc, (size_t)enc_bytes, lut, dec, n);
    if (dec_bytes != (int)n) {
        printf("[%s] FAIL decode size: got %d expected %zu\n", name, dec_bytes, n);
        free(enc); free(dec); free(lut);
        return 1;
    }
    if (memcmp(src, dec, n) != 0) {
        printf("[%s] FAIL decode mismatch\n", name);
        for (size_t i = 0; i < n; i++) {
            if (src[i] != dec[i]) {
                printf("  first diff at %zu: src=0x%02x dec=0x%02x\n", i, src[i], dec[i]);
                break;
            }
        }
        free(enc); free(dec); free(lut);
        return 1;
    }

    double ratio = 100.0 * enc_bytes / (double)n;
    printf("[%s] OK src=%zu enc=%d (%.1f%%) max_code_len=%d\n",
           name, n, enc_bytes, ratio, max_len);

    // ── Also test 4-stream variant on the same input + tables ──
    int enc4 = huff_encode_4s(src, n, code_lengths, codes, enc, n * 2 + 16);
    if (enc4 < 0) {
        printf("[%s/4s] FAIL encode_4s\n", name);
        free(enc); free(dec); free(lut);
        return 1;
    }
    int dec4 = huff_decode_4s(enc, (size_t)enc4, lut, dec, n);
    if (dec4 != (int)n || memcmp(src, dec, n) != 0) {
        printf("[%s/4s] FAIL decode_4s\n", name);
        free(enc); free(dec); free(lut);
        return 1;
    }
    double ratio4 = 100.0 * enc4 / (double)n;
    printf("[%s/4s] OK enc=%d (%.1f%%, +%d hdr+padding bytes)\n",
           name, enc4, ratio4, enc4 - enc_bytes);
    free(enc); free(dec); free(lut);
    return 0;
}

#ifndef HUFF_REF_NO_MAIN
int main(int argc, char **argv) {
    int fails = 0;

    // Test 1: synthetic uniform random
    {
        size_t n = 65536;
        uint8_t *buf = (uint8_t*)malloc(n);
        srand(12345);
        for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)rand();
        fails += roundtrip_test("uniform-random-64KB", buf, n);
        free(buf);
    }

    // Test 2: highly skewed (one symbol dominates)
    {
        size_t n = 4096;
        uint8_t *buf = (uint8_t*)malloc(n);
        for (size_t i = 0; i < n; i++) buf[i] = (i % 100 == 0) ? (uint8_t)(i & 0xFF) : 'A';
        fails += roundtrip_test("skewed-A-dominant", buf, n);
        free(buf);
    }

    // Test 3: small input
    {
        const char *s = "Hello, Huffman world! abcabcabc";
        fails += roundtrip_test("hello-world", (const uint8_t*)s, strlen(s));
    }

    // Test 4: real file if provided
    if (argc > 1) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) { perror(argv[1]); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        size_t cap = (sz > (1 << 20)) ? (1 << 20) : (size_t)sz; // first 1 MB
        uint8_t *buf = (uint8_t*)malloc(cap);
        size_t got = fread(buf, 1, cap, f);
        fclose(f);
        fails += roundtrip_test(argv[1], buf, got);
        free(buf);
    }

    printf("\n%d failures\n", fails);
    return fails ? 1 : 0;
}
#endif  // HUFF_REF_NO_MAIN
