// ── StreamLZ GPU tANS Encode Kernel ─────────────────────────────
// 5-state interleaved tANS entropy encoder, one warp per sub-chunk.
// Produces output bit-identical to the CPU encoder in
// encode/entropy/tans_encoder.zig.
//
// Kernel design:
//   - 32 threads per block (1 warp), 1 block per sub-chunk
//   - Parallel histogram build (all 32 lanes via atomicAdd)
//   - Lane 0: normalize weights, build encoding table, bit-count
//     dry run, encode, write table header + bitstreams
//   - If tANS doesn't compress, writes raw bytes (memcpy fallback)
//
// Output format (matches CPU decoder in decode/entropy/tans_decoder.zig):
//   [0]     reserved bit (0) + 2-bit (log_table_bits - 8) = 3 bits
//   [3..]   Golomb-Rice or sparse table header
//   [...]   backward bitstream bytes (grows from low address)
//   [...]   forward bitstream bytes (grows from high address)
//
// Build: nvcc -ptx -arch=sm_89 -O3 gpu_tans_kernel.cu

#include <cstdint>

// ── Descriptor for one sub-chunk to encode ────────────────────────
struct TansChunkDesc {
    uint32_t src_offset;   // offset into input buffer
    uint32_t src_size;     // bytes to encode
    uint32_t dst_offset;   // offset into output buffer
    uint32_t dst_capacity; // max output bytes
};

// ── Per-symbol encoding table entry ───────────────────────────────
struct TansEncEntry {
    int32_t  base_offset;
    uint16_t thres;
    uint8_t  num_bits;
};

// ── Log factor approximation tables (matches CPU) ─────────────────
__device__ static const float log_factor_up_table[32] = {
    0.000000f, 0.693147f, 0.405465f, 0.287682f, 0.223144f, 0.182322f, 0.154151f, 0.133531f,
    0.117783f, 0.105361f, 0.095310f, 0.087011f, 0.080043f, 0.074108f, 0.068993f, 0.064539f,
    0.060625f, 0.057158f, 0.054067f, 0.051293f, 0.048790f, 0.046520f, 0.044452f, 0.042560f,
    0.040822f, 0.039221f, 0.037740f, 0.036368f, 0.035091f, 0.033902f, 0.032790f, 0.031749f,
};

__device__ static const float log_factor_down_table[32] = {
    0.000000f,  0.000000f,  -0.693147f, -0.405465f, -0.287682f, -0.223144f, -0.182322f, -0.154151f,
    -0.133531f, -0.117783f, -0.105361f, -0.095310f, -0.087011f, -0.080043f, -0.074108f, -0.068993f,
    -0.064539f, -0.060625f, -0.057158f, -0.054067f, -0.051293f, -0.048790f, -0.046520f, -0.044452f,
    -0.042560f, -0.040822f, -0.039221f, -0.037740f, -0.036368f, -0.035091f, -0.033902f, -0.032790f,
};

__device__ float tansGetLogFactorUp(uint32_t value) {
    if (value >= 32) {
        float inv = 1.0f / (float)value;
        return inv - inv * inv * 0.5f;
    }
    return log_factor_up_table[value];
}

__device__ float tansGetLogFactorDown(uint32_t value) {
    if (value >= 32) {
        float inv = 1.0f / (float)value;
        return -inv - inv * inv * 0.5f;
    }
    return log_factor_down_table[value];
}

// ── Rounding helpers (match CPU exactly) ──────────────────────────

__device__ uint32_t doubleToUintRoundPow2(double v) {
    uint32_t u = (uint32_t)v;
    double uf = (double)u;
    return (v * v > uf * (uf + 1.0)) ? u + 1 : u;
}

__device__ uint32_t ilog2Round(uint32_t v) {
    if (v == 0) return 0;
    uint32_t bsr = 31 - __clz(v);
    uint32_t lower = 1u << bsr;
    uint32_t upper = lower << 1;
    return (v - lower >= upper - v) ? bsr + 1 : bsr;
}

// ── Max-heap for normalization ────────────────────────────────────

struct HeapEntry {
    float   score;
    int32_t index;
};

__device__ void heapMake(HeapEntry* heap, int n) {
    if (n < 2) return;
    int half = n / 2;
    while (half > 0) {
        half--;
        int t = half;
        while (true) {
            int u = 2 * t + 1;
            if (u >= n) break;
            if (u + 1 < n && heap[u].score < heap[u + 1].score) u++;
            if (heap[u].score < heap[t].score) break;
            HeapEntry tmp = heap[t]; heap[t] = heap[u]; heap[u] = tmp;
            t = u;
        }
    }
}

__device__ void heapPush(HeapEntry* heap, int n) {
    int t = n - 1;
    while (t > 0) {
        int u = (t - 1) >> 1;
        if (heap[u].score >= heap[t].score) break;
        HeapEntry tmp = heap[t]; heap[t] = heap[u]; heap[u] = tmp;
        t = u;
    }
}

__device__ void heapPop(HeapEntry* heap, int n) {
    int t = 0;
    while (true) {
        int u = 2 * t + 1;
        if (u >= n) break;
        int child = u;
        if (u + 1 < n && heap[u].score < heap[u + 1].score) child = u + 1;
        heap[t] = heap[child];
        t = child;
    }
    if (t < n - 1) {
        heap[t] = heap[n - 1];
        while (t > 0) {
            int u = (t - 1) >> 1;
            if (heap[u].score >= heap[t].score) break;
            HeapEntry tmp = heap[t]; heap[t] = heap[u]; heap[u] = tmp;
            t = u;
        }
    }
}

// ── Normalize histogram counts to sum L ───────────────────────────

__device__ uint32_t tansNormalizeCounts(
    uint32_t* weights,
    uint32_t  table_size,
    const uint32_t* histo,
    uint32_t  histo_sum,
    int       num_symbols
) {
    uint32_t syms_used = 0;
    double multiplier = (double)table_size / (double)histo_sum;
    uint32_t weight_sum = 0;

    for (int i = 0; i < num_symbols; i++) {
        uint32_t sym_count = histo[i];
        uint32_t w = 0;
        if (sym_count != 0) {
            w = doubleToUintRoundPow2((double)sym_count * multiplier);
            weight_sum += w;
            syms_used++;
        }
        weights[i] = w;
    }
    for (int i = num_symbols; i < 256; i++) weights[i] = 0;
    if (weight_sum == table_size) return syms_used;

    // Heap-based adjustment
    HeapEntry heap_buf[256];
    int heap_len = 0;
    int64_t diff = (int64_t)table_size - (int64_t)weight_sum;

    if (diff < 0) {
        for (int i = 0; i < num_symbols; i++) {
            if (weights[i] > 1) {
                heap_buf[heap_len].index = i;
                heap_buf[heap_len].score = (float)histo[i] * tansGetLogFactorDown(weights[i]);
                heap_len++;
            }
        }
    } else {
        for (int i = 0; i < num_symbols; i++) {
            if (histo[i] != 0) {
                heap_buf[heap_len].index = i;
                heap_buf[heap_len].score = (float)histo[i] * tansGetLogFactorUp(weights[i]);
                heap_len++;
            }
        }
    }
    heapMake(heap_buf, heap_len);

    if (diff < 0) {
        while (diff != 0) {
            int index = heap_buf[0].index;
            heapPop(heap_buf, heap_len);
            heap_len--;
            weights[index]--;
            if (weights[index] > 1) {
                heap_buf[heap_len].index = index;
                heap_buf[heap_len].score = (float)histo[index] * tansGetLogFactorDown(weights[index]);
                heap_len++;
                heapPush(heap_buf, heap_len);
            }
            diff++;
        }
    } else {
        while (diff != 0) {
            int index = heap_buf[0].index;
            heapPop(heap_buf, heap_len);
            heap_len--;
            weights[index]++;
            heap_buf[heap_len].index = index;
            heap_buf[heap_len].score = (float)histo[index] * tansGetLogFactorUp(weights[index]);
            heap_len++;
            heapPush(heap_buf, heap_len);
            diff--;
        }
    }
    return syms_used;
}

// ── Build encoding table (4-way interleaved, matches CPU) ─────────

__device__ void tansInitTable(
    TansEncEntry* te,
    uint16_t*     te_data,
    const uint32_t* weights,
    int           num_symbols,
    uint32_t      log_table_bits
) {
    uint32_t L = 1u << log_table_bits;

    uint32_t ones = 0;
    for (int i = 0; i < num_symbols; i++) {
        if (weights[i] == 1) ones++;
    }

    uint32_t slots_left = L - ones;
    uint32_t sa = slots_left >> 2;
    uint32_t pointers[4];
    pointers[0] = 0;
    uint32_t sb = sa + ((slots_left & 3) > 0 ? 1 : 0);
    pointers[1] = sb;
    sb += sa + ((slots_left & 3) > 1 ? 1 : 0);
    pointers[2] = sb;
    sb += sa + ((slots_left & 3) > 2 ? 1 : 0);
    pointers[3] = sb;

    uint32_t ones_ptr = slots_left;
    int32_t weights_sum = 0;

    for (int i = 0; i < num_symbols; i++) {
        uint32_t w = weights[i];
        if (w == 0) {
            te[i].base_offset = 0;
            te[i].thres = 0;
            te[i].num_bits = 0;
            continue;
        }
        if (w == 1) {
            te[i].num_bits = (uint8_t)log_table_bits;
            te[i].thres = (uint16_t)(2 * L);
            te_data[ones_ptr] = (uint16_t)(L + ones_ptr);
            te[i].base_offset = (int32_t)ones_ptr - 1;
            ones_ptr++;
        } else {
            uint32_t w_minus_1 = w - 1;
            uint32_t num_bits_high = 31 - __clz(w_minus_1) + 1;
            te[i].num_bits = (uint8_t)(log_table_bits - num_bits_high);
            te[i].thres = (uint16_t)((2 * w) << (log_table_bits - num_bits_high));
            te[i].base_offset = weights_sum - (int32_t)w;

            int32_t ptr_cursor = weights_sum;
            for (int j = 0; j < 4; j++) {
                int32_t y_signed = (int32_t)w + ((weights_sum - (int32_t)j - 1) & 3);
                int32_t y = y_signed >> 2;
                int32_t p = (int32_t)pointers[j];
                for (int32_t yl = 0; yl < y; yl++) {
                    te_data[ptr_cursor] = (uint16_t)(p + (int32_t)L);
                    ptr_cursor++;
                    p++;
                }
                pointers[j] = (uint32_t)p;
            }
            weights_sum += (int32_t)w;
        }
    }
    // Zero out remaining entries
    for (int i = num_symbols; i < 256; i++) {
        te[i].base_offset = 0;
        te[i].thres = 0;
        te[i].num_bits = 0;
    }
}

// ── Next-state lookup (matches CPU nextState) ─────────────────────

__device__ void nextState(
    const uint16_t* te_data,
    const TansEncEntry* entry,
    uint32_t state,
    uint32_t &new_state,
    uint32_t &nb
) {
    uint32_t num_bits_base = entry->num_bits;
    uint32_t above_thres = (state >= entry->thres) ? 1 : 0;
    nb = num_bits_base + above_thres;
    uint32_t state_high = state >> nb;
    int64_t idx = (int64_t)entry->base_offset + (int64_t)state_high;
    new_state = te_data[(uint32_t)idx];
}

// ── Bit-count dry run (matches CPU tansGetEncodedBitCount) ────────

__device__ void tansGetEncodedBitCount(
    const TansEncEntry* te,
    const uint16_t*     te_data,
    const uint8_t*      src,
    uint32_t            src_len,
    uint32_t            log_table_bits,
    uint32_t            &out_forward_bits,
    uint32_t            &out_backward_bits
) {
    uint32_t L = 1u << log_table_bits;
    uint32_t src_end_idx = src_len - 5;
    uint32_t states[5];
    states[0] = (uint32_t)src[src_end_idx + 0] | L;
    states[1] = (uint32_t)src[src_end_idx + 1] | L;
    states[2] = (uint32_t)src[src_end_idx + 2] | L;
    states[3] = (uint32_t)src[src_end_idx + 3] | L;
    states[4] = (uint32_t)src[src_end_idx + 4] | L;

    uint32_t forward_bits = 0;
    uint32_t backward_bits = 0;

    uint32_t body_len = src_end_idx;
    uint32_t rounds = body_len / 10;
    uint32_t remainder = body_len % 10;

    int read_idx;
    if (src_end_idx == 0) {
        out_forward_bits = 2 * log_table_bits;
        out_backward_bits = 3 * log_table_bits;
        return;
    }
    read_idx = (int)src_end_idx - 1;

    // Remainder
    uint32_t ri = 10 - remainder;
    while (ri < 10) {
        uint8_t sym = src[read_idx];
        if (read_idx > 0) read_idx--;
        const TansEncEntry* entry = &te[sym];
        uint32_t si = ri % 5;
        uint32_t sv = states[4 - si];
        uint32_t ns, nb;
        nextState(te_data, entry, sv, ns, nb);
        if (ri < 5) forward_bits += nb; else backward_bits += nb;
        states[4 - si] = ns;
        ri++;
    }

    // Main loop: 10 symbols per iteration
    for (uint32_t r = 0; r < rounds; r++) {
        // Forward group (5 symbols)
        for (int idx = 4; idx >= 0; idx--) {
            uint8_t sym = src[read_idx];
            if (read_idx > 0) read_idx--;
            const TansEncEntry* entry = &te[sym];
            uint32_t ns, nb;
            nextState(te_data, entry, states[idx], ns, nb);
            forward_bits += nb;
            states[idx] = ns;
        }
        // Backward group (5 symbols)
        for (int idx = 4; idx >= 0; idx--) {
            uint8_t sym = src[read_idx];
            if (read_idx > 0) read_idx--;
            const TansEncEntry* entry = &te[sym];
            uint32_t ns, nb;
            nextState(te_data, entry, states[idx], ns, nb);
            backward_bits += nb;
            states[idx] = ns;
        }
    }

    out_forward_bits  = forward_bits  + 2 * log_table_bits;
    out_backward_bits = backward_bits + 3 * log_table_bits;
}

// ── 64-bit bit writer (forward, big-endian flush) ─────────────────
// Matches CPU BitWriter64Forward exactly.

struct BitWriter64Forward {
    uint8_t* position;
    uint64_t bits;
    uint32_t pos;  // starts at 63

    __device__ void init(uint8_t* dst) {
        position = dst;
        bits = 0;
        pos = 63;
    }

    __device__ void flush() {
        if (pos == 63) return;
        uint32_t t = (63 - pos) >> 3;
        uint32_t shift = pos + 1;
        uint64_t v = bits << shift;
        pos += 8 * t;
        // Big-endian byte-swap of u64, matching CPU @byteSwap
        uint64_t swapped = ((v >> 56) & 0xFFULL) |
                           ((v >> 40) & 0xFF00ULL) |
                           ((v >> 24) & 0xFF0000ULL) |
                           ((v >>  8) & 0xFF000000ULL) |
                           ((v <<  8) & 0xFF00000000ULL) |
                           ((v << 24) & 0xFF0000000000ULL) |
                           ((v << 40) & 0xFF000000000000ULL) |
                           ((v << 56) & 0xFF00000000000000ULL);
        memcpy(position, &swapped, 8);
        position += t;
    }

    __device__ void write(uint32_t b, uint32_t n) {
        pos -= n;
        bits = (bits << n) | (uint64_t)b;
        flush();
    }

    __device__ void writeNoFlush(uint32_t b, uint32_t n) {
        pos -= n;
        bits = (bits << n) | (uint64_t)b;
    }

    __device__ uint8_t* getFinalPtr() {
        uint32_t adj = (pos != 63) ? 1 : 0;
        return position + adj;
    }
};

// ── 64-bit bit writer (backward, native-endian flush) ─────────────
// Matches CPU BitWriter64Backward exactly.

struct BitWriter64Backward {
    uint8_t* position;
    uint64_t bits;
    uint32_t pos;  // starts at 63

    __device__ void init(uint8_t* dst) {
        position = dst;
        bits = 0;
        pos = 63;
    }

    __device__ void flush() {
        if (pos == 63) return;
        uint32_t t = (63 - pos) >> 3;
        uint32_t shift = pos + 1;
        uint64_t v = bits << shift;
        pos += 8 * t;
        // Native-endian (little-endian) write at position-8, retreat
        uint8_t* dst = position - 8;
        memcpy(dst, &v, 8);
        position -= t;
    }

    __device__ void write(uint32_t b, uint32_t n) {
        pos -= n;
        bits = (bits << n) | (uint64_t)b;
        flush();
    }

    __device__ void writeNoFlush(uint32_t b, uint32_t n) {
        pos -= n;
        bits = (bits << n) | (uint64_t)b;
    }
};

// ── Encode symbol group (5 symbols into one writer) ───────────────

__device__ void encodeSymbolGroup(
    BitWriter64Forward* writer,
    const TansEncEntry* te,
    const uint16_t*     te_data,
    const uint8_t*      src,
    int                 &read_idx,
    uint32_t*           states
) {
    for (int idx = 4; idx >= 0; idx--) {
        uint8_t sym = src[read_idx];
        if (read_idx > 0) read_idx--;
        uint32_t sv = states[idx];
        const TansEncEntry* entry = &te[sym];
        uint32_t ns, nb;
        nextState(te_data, entry, sv, ns, nb);
        uint32_t mask = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1);
        writer->writeNoFlush(sv & mask, nb);
        states[idx] = ns;
    }
}

__device__ void encodeSymbolGroupBackward(
    BitWriter64Backward* writer,
    const TansEncEntry*  te,
    const uint16_t*      te_data,
    const uint8_t*       src,
    int                  &read_idx,
    uint32_t*            states
) {
    for (int idx = 4; idx >= 0; idx--) {
        uint8_t sym = src[read_idx];
        if (read_idx > 0) read_idx--;
        uint32_t sv = states[idx];
        const TansEncEntry* entry = &te[sym];
        uint32_t ns, nb;
        nextState(te_data, entry, sv, ns, nb);
        uint32_t mask = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1);
        writer->writeNoFlush(sv & mask, nb);
        states[idx] = ns;
    }
}

// ── Encode the body (forward + backward bitstreams) ───────────────
// Returns forward_bytes, backward_bytes, and pointers via output params.

__device__ void tansEncodeBytes(
    uint8_t*            fwd_start,
    uint8_t*            bwd_end,
    const TansEncEntry* te,
    const uint16_t*     te_data,
    const uint8_t*      src,
    uint32_t            src_len,
    uint32_t            log_table_bits,
    uint32_t            forward_bits_count,
    uint32_t            backward_bits_count,
    uint32_t            &out_forward_bytes,
    uint32_t            &out_backward_bytes,
    uint8_t*            &out_forward_start,
    uint8_t*            &out_backward_start
) {
    BitWriter64Forward  forward;
    BitWriter64Backward backward;
    forward.init(fwd_start);
    backward.init(bwd_end);

    // Pad to byte boundary
    if ((forward_bits_count & 7) != 0) {
        uint32_t pad = 8 - (forward_bits_count & 7);
        forward.writeNoFlush(0, pad);
    }
    if ((backward_bits_count & 7) != 0) {
        uint32_t pad = 8 - (backward_bits_count & 7);
        backward.writeNoFlush(0, pad);
    }

    uint32_t L = 1u << log_table_bits;
    uint32_t src_end_idx = src_len - 5;
    uint32_t states[5];
    states[0] = (uint32_t)src[src_end_idx + 0] | L;
    states[1] = (uint32_t)src[src_end_idx + 1] | L;
    states[2] = (uint32_t)src[src_end_idx + 2] | L;
    states[3] = (uint32_t)src[src_end_idx + 3] | L;
    states[4] = (uint32_t)src[src_end_idx + 4] | L;

    uint32_t body_len = src_end_idx;
    uint32_t rounds = body_len / 10;
    uint32_t remainder = body_len % 10;

    int read_idx = (int)src_end_idx;
    if (read_idx > 0) read_idx--;

    // Process remainder symbols first
    uint32_t ri = 10 - remainder;
    while (ri < 10) {
        uint8_t sym = src[read_idx];
        if (read_idx > 0) read_idx--;
        const TansEncEntry* entry = &te[sym];
        uint32_t si = ri % 5;
        uint32_t sv = states[4 - si];
        uint32_t ns, nb;
        nextState(te_data, entry, sv, ns, nb);
        uint32_t mask = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1);
        uint32_t bits = sv & mask;
        if (ri < 5)
            forward.writeNoFlush(bits, nb);
        else
            backward.writeNoFlush(bits, nb);
        states[4 - si] = ns;
        ri++;
    }
    if (remainder > 0) {
        backward.flush();
        forward.flush();
    }

    // Main loop: 10 symbols per iteration
    for (uint32_t r = 0; r < rounds; r++) {
        encodeSymbolGroup(&forward, te, te_data, src, read_idx, states);
        encodeSymbolGroupBackward(&backward, te, te_data, src, read_idx, states);
        backward.flush();
        forward.flush();
    }

    // Write final states
    uint32_t mask_L = L - 1;
    backward.writeNoFlush(states[4] & mask_L, log_table_bits);
    backward.writeNoFlush(states[2] & mask_L, log_table_bits);
    backward.writeNoFlush(states[0] & mask_L, log_table_bits);
    forward.writeNoFlush(states[3] & mask_L, log_table_bits);
    forward.writeNoFlush(states[1] & mask_L, log_table_bits);
    backward.flush();
    forward.flush();

    out_forward_bytes  = (uint32_t)(forward.position - fwd_start);
    out_backward_bytes = (uint32_t)(bwd_end - backward.position);
    out_forward_start  = fwd_start;
    out_backward_start = backward.position;
}

// ── Table header encoding ─────────────────────────────────────────

// Golomb-Rice helpers

__device__ uint32_t getBitsForArraysOfRice(const uint32_t* arr, int arr_len, uint32_t k) {
    uint32_t result = 0;
    for (int i = 0; i < arr_len; i++) {
        if (arr[i] != 0) {
            uint32_t shifted = ((uint32_t)i >> k) + 1;
            uint32_t lg = 31 - __clz(shifted);
            result += arr[i] * (k + 1 + 2 * lg);
        }
    }
    return result;
}

__device__ void writeManyRiceCodes(BitWriter64Forward* bw, const uint8_t* data, int count) {
    for (int i = 0; i < count; i++) {
        uint32_t v = data[i];
        while (v >= 24) { bw->write(0, 24); v -= 24; }
        bw->write(1, v + 1);
    }
}

__device__ void writeSymRangeLowBits(BitWriter64Forward* bw, const uint8_t* data, const uint8_t* bitcount, int count) {
    for (int i = 0; i < count; i++) {
        if (bitcount[i] == 0) continue;
        bw->write(data[i], bitcount[i]);
    }
}

__device__ void writeNumSymRange(BitWriter64Forward* bw, int num_symrange, int used_syms) {
    if (used_syms == 256) return;
    int x_val = (used_syms < 257 - used_syms) ? used_syms : (257 - used_syms);
    uint32_t nb = 32 - __clz((uint32_t)(2 * x_val - 1));
    int32_t base = (int32_t)(1u << nb) - (int32_t)(2 * x_val);
    if (num_symrange >= base) {
        bw->write((uint32_t)(num_symrange + base), nb);
    } else {
        bw->write((uint32_t)num_symrange, nb - 1);
    }
}

__device__ int encodeSymRange(
    uint8_t*  rice,
    uint8_t*  sr_bits,
    uint8_t*  bitcount,
    uint32_t  used_syms,
    const int32_t* range,
    int       numrange
) {
    if (used_syms >= 256) return 0;
    int32_t which = (range[0] == 0) ? 1 : 0;
    int range_offset = (range[0] == 0) ? 1 : 0;
    int num = (int)((range[0] != 0 ? 1 : 0) + 2 * ((numrange - 3) / 2));

    for (int i = 0; i < num; i++) {
        int32_t v = range[range_offset + i];
        int32_t ebit = (~which) & 1;
        which++;
        v += (1 << ebit) - 1;
        int32_t vshift = v >> ebit;
        uint32_t nb_u32 = (uint32_t)(31 - __clz((uint32_t)vshift));
        rice[i] = (uint8_t)nb_u32;
        uint32_t nb_total = nb_u32 + (uint32_t)ebit;
        uint32_t mask = (1u << nb_total) - 1;
        sr_bits[i] = (uint8_t)((uint32_t)v & mask);
        bitcount[i] = (uint8_t)nb_total;
    }
    return num;
}

// Insertion sort for sparse table encoding
__device__ void sortU32(uint32_t* arr, int n) {
    for (int i = 1; i < n; i++) {
        uint32_t v = arr[i];
        int j = i;
        while (j > 0 && arr[j - 1] > v) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = v;
    }
}

// Full table header encoder (matches CPU tansEncodeTable)
__device__ void tansEncodeTable(
    BitWriter64Forward* bw,
    uint32_t            log_table_bits,
    const uint32_t*     lookup,
    int                 histo_size,
    uint32_t            used_symbols
) {
    if (used_symbols <= 7) {
        // Sparse path
        bw->writeNoFlush(0, 1);
        bw->writeNoFlush(used_symbols - 2, 3);

        uint32_t sympos[8];
        int sympos_len = 0;
        for (int i = 0; i < histo_size; i++) {
            if (lookup[i] != 0) {
                sympos[sympos_len++] = (uint32_t)i | (lookup[i] << 16);
            }
        }
        sortU32(sympos, sympos_len);

        // Compute delta-bits width
        uint32_t delta_bits = 1;
        {
            uint32_t p = 0;
            for (uint32_t i = 0; i < used_symbols - 1; i++) {
                uint32_t sv = sympos[i] >> 16;
                uint32_t diff = sv - p;
                uint32_t nb = (diff != 0) ? (31 - __clz(diff) + 1) : 0;
                if (nb > delta_bits) delta_bits = nb;
                p = sv;
            }
        }
        uint32_t bps = ilog2Round(log_table_bits) + 1;
        bw->writeNoFlush(delta_bits, bps);

        // Write (count-1) delta/symbol pairs
        {
            uint32_t p = 0;
            for (uint32_t i = 0; i < used_symbols - 1; i++) {
                uint32_t sv = sympos[i] >> 16;
                uint32_t diff = sv - p;
                uint32_t sym_byte = sympos[i] & 0xFF;
                bw->write(diff | (sym_byte << delta_bits), delta_bits + 8);
                p = sv;
            }
        }
        // Last symbol byte only
        bw->write(sympos[used_symbols - 1] & 0xFF, 8);
        return;
    }

    // Golomb-Rice path (> 7 symbols)
    bw->writeNoFlush(1, 1);

    uint32_t arr_z[128];
    for (int i = 0; i < 128; i++) arr_z[i] = 0;
    int32_t ranges[257];
    for (int i = 0; i < 257; i++) ranges[i] = 0;
    int ranges_len = 0;
    int32_t arr_x[256];
    int arr_x_count = 0;
    int32_t arr_y[32];
    int arr_y_count = 0;
    uint8_t arr_w[256];
    uint8_t sr_rice[256];
    uint8_t sr_bits_buf[256];
    uint8_t sr_bitcount[256];

    int pos = 0;
    while (pos < histo_size && lookup[pos] == 0) pos++;
    ranges[ranges_len++] = pos;

    int32_t average = 6;
    uint32_t used_syms = 0;
    while (pos < histo_size) {
        int pos_start = pos;
        while (pos < histo_size) {
            int32_t vraw = (int32_t)lookup[pos];
            if (vraw == 0) break;
            int32_t v = vraw - 1;
            int32_t avg_div_4 = average >> 2;
            int32_t limit = 2 * avg_div_4;
            int32_t u;
            if (v > limit)
                u = v;
            else
                u = (2 * (v - avg_div_4)) ^ ((v - avg_div_4) >> 31);
            arr_x[arr_x_count++] = u;
            if (u >= 0x80) {
                arr_y[arr_y_count++] = u;
            } else {
                arr_z[u]++;
            }
            int32_t nlimit = (v < limit) ? v : limit;
            pos++;
            used_syms++;
            average += nlimit - avg_div_4;
        }
        ranges[ranges_len++] = pos - pos_start;
        int zero_start = pos;
        while (pos < histo_size && lookup[pos] == 0) pos++;
        ranges[ranges_len++] = pos - zero_start;
    }
    ranges[ranges_len - 1] += (int32_t)(256 - pos);

    // Find best Q
    uint32_t best_score = 0xFFFFFFFFu;
    uint32_t Q = 0;
    for (uint32_t tq = 0; tq < 8; tq++) {
        uint32_t score = getBitsForArraysOfRice(arr_z, 128, tq);
        for (int i = 0; i < arr_y_count; i++) {
            uint32_t shifted = (uint32_t)((arr_y[i] >> tq) + 1);
            uint32_t lg = 31 - __clz(shifted);
            score += tq + 2 * lg + 1;
        }
        if (score < best_score) {
            best_score = score;
            Q = tq;
        }
    }

    int num_symrange = encodeSymRange(sr_rice, sr_bits_buf, sr_bitcount, used_syms, ranges, ranges_len);
    bw->writeNoFlush((used_syms - 1) + (Q << 8), 11);
    writeNumSymRange(bw, num_symrange, (int)used_syms);

    // Compute per-entry low bits after Q split
    for (int i = 0; i < arr_x_count; i++) {
        uint32_t x = (uint32_t)(arr_x[i] + (1 << Q));
        uint32_t nb = 31 - __clz(x >> Q);
        arr_w[i] = (uint8_t)nb;
        uint32_t mask_bits = 1u << (Q + nb);
        arr_x[i] = (int32_t)(x & (mask_bits - 1));
    }

    writeManyRiceCodes(bw, arr_w, arr_x_count);
    writeManyRiceCodes(bw, sr_rice, num_symrange);
    writeSymRangeLowBits(bw, sr_bits_buf, sr_bitcount, num_symrange);

    for (int i = 0; i < arr_x_count; i++) {
        uint32_t total = Q + arr_w[i];
        if (total != 0) {
            bw->write((uint32_t)arr_x[i], total);
        }
    }
}

// ── Main kernel entry point ───────────────────────────────────────
extern "C" __global__ void slzTansEncodeKernel(
    const uint8_t*         __restrict__ src_buf,
    uint8_t*               __restrict__ dst_buf,
    const TansChunkDesc*   __restrict__ descs,
    uint32_t*              __restrict__ out_sizes,
    uint32_t               num_chunks
) {
    __shared__ uint32_t s_histo[256];

    uint32_t chunk_id = blockIdx.x;
    if (chunk_id >= num_chunks) return;

    int lane = threadIdx.x & 31;
    const TansChunkDesc desc = descs[chunk_id];
    const uint8_t* src = src_buf + desc.src_offset;
    uint8_t*       dst = dst_buf + desc.dst_offset;
    uint32_t src_len      = desc.src_size;
    uint32_t dst_capacity = desc.dst_capacity;

    // ── Too small for tANS — copy raw ───────────────────────
    if (src_len < 32) {
        if (lane == 0) {
            // Raw copy: output = raw bytes, size = src_len with MSB set
            // to signal "not compressed"
            for (uint32_t i = 0; i < src_len && i < dst_capacity; i++)
                dst[i] = src[i];
            out_sizes[chunk_id] = src_len | 0x80000000u;
        }
        return;
    }

    // ── Phase 1: Parallel histogram (all 32 lanes) ──────────
    for (int i = lane; i < 256; i += 32) s_histo[i] = 0;
    __syncwarp();

    for (uint32_t i = lane; i < src_len; i += 32)
        atomicAdd(&s_histo[src[i]], 1);
    __syncwarp();

    // ── Everything below is lane 0 only ─────────────────────
    if (lane != 0) return;

    // Subtract last 5 bytes from histogram (they are initial states)
    uint32_t src_end_idx = src_len - 5;
    s_histo[src[src_end_idx + 0]]--;
    s_histo[src[src_end_idx + 1]]--;
    s_histo[src[src_end_idx + 2]]--;
    s_histo[src[src_end_idx + 3]]--;
    s_histo[src[src_end_idx + 4]]--;

    // Choose log_table_bits from source size
    int32_t raw_log = (int32_t)ilog2Round(src_len - 5) - 2;
    if (raw_log < 8) raw_log = 8;
    if (raw_log > 11) raw_log = 11;
    uint32_t log_table_bits = (uint32_t)raw_log;

    // Find used range
    int num_symbols = 256;
    while (num_symbols > 0 && s_histo[num_symbols - 1] == 0) num_symbols--;
    if (num_symbols == 0) {
        out_sizes[chunk_id] = src_len | 0x80000000u;
        for (uint32_t i = 0; i < src_len && i < dst_capacity; i++)
            dst[i] = src[i];
        return;
    }

    // Normalize weights
    uint32_t weights[256];
    uint32_t used_symbols = tansNormalizeCounts(
        weights,
        1u << log_table_bits,
        s_histo,
        src_len - 5,
        num_symbols
    );

    if (used_symbols <= 1) {
        out_sizes[chunk_id] = src_len | 0x80000000u;
        for (uint32_t i = 0; i < src_len && i < dst_capacity; i++)
            dst[i] = src[i];
        // Restore histogram (not strictly needed on GPU but keeps logic clean)
        return;
    }

    // Build encoding table
    TansEncEntry te[256];
    uint16_t te_data[2048];
    tansInitTable(te, te_data, weights, num_symbols, log_table_bits);

    // Encode table header into scratch
    uint8_t table_buf[512];
    for (int i = 0; i < 512; i++) table_buf[i] = 0;
    BitWriter64Forward bw;
    bw.init(table_buf);
    // Format: 1 reserved bit (0) + 2-bit log_table_bits offset
    // Written as 3 bits: bit 0 = reserved (0), bits 1-2 = (log_table_bits - 8)
    // But the decoder reads: 1 bit reserved, then 2 bits = log_table_bits - 8
    // So we write: reserved=0 (1 bit), then log_table_bits-8 (2 bits) = 3 bits total
    // However the CPU encoder writes them as: writeNoFlush(log_table_bits - 8, 3)
    // which puts the 3-bit field containing the log_table_bits offset (the reserved
    // bit is the MSB of this 3-bit field and is always 0 for values 0-3).
    bw.writeNoFlush(log_table_bits - 8, 3);
    tansEncodeTable(&bw, log_table_bits, weights, num_symbols, used_symbols);
    bw.flush();
    uint8_t* table_final_ptr = bw.getFinalPtr();
    uint32_t table_bytes = (uint32_t)(table_final_ptr - table_buf);

    // Compute exact bit counts
    uint32_t forward_bits, backward_bits;
    tansGetEncodedBitCount(te, te_data, src, src_len, log_table_bits, forward_bits, backward_bits);
    uint32_t payload_bytes = ((forward_bits + 7) >> 3) + ((backward_bits + 7) >> 3);

    if (payload_bytes < 8) {
        // tANS not beneficial
        out_sizes[chunk_id] = src_len | 0x80000000u;
        for (uint32_t i = 0; i < src_len && i < dst_capacity; i++)
            dst[i] = src[i];
        return;
    }

    uint32_t total_size = table_bytes + payload_bytes;
    if (total_size >= src_len || total_size + 8 > dst_capacity) {
        // Doesn't compress — raw copy
        out_sizes[chunk_id] = src_len | 0x80000000u;
        for (uint32_t i = 0; i < src_len && i < dst_capacity; i++)
            dst[i] = src[i];
        return;
    }

    // Encode body into a scratch buffer on the stack. We need payload_bytes + 64
    // bytes of slack. Cap at a reasonable stack size.
    // For GPU, use a fixed-size scratch buffer. Sub-chunks are at most ~256KB
    // but encoded payload is always smaller. We use dst_capacity as upper bound.
    // The encoded payload must be smaller than src_len (we checked above), so
    // we use the destination buffer itself as scratch space for the body encoding,
    // offset past where the table will go.
    //
    // Strategy: use dst[table_bytes+16 ..] as forward writer space and
    // dst[dst_capacity-16 ..] as backward writer end, then assemble in place.
    //
    // But to avoid clobbering, we encode into a separate region of dst first,
    // then memcpy into final position.

    // We'll encode forward into dst[table_bytes + 16 ..] and backward from
    // dst[dst_capacity - 16]. After encoding, we assemble:
    // dst[0..table_bytes] = table header
    // dst[table_bytes .. table_bytes + bwd_bytes] = backward stream
    // dst[table_bytes + bwd_bytes .. table_bytes + bwd_bytes + fwd_bytes] = forward stream

    uint8_t* fwd_start = dst + table_bytes + 16;
    uint8_t* bwd_end   = dst + dst_capacity - 16;

    // Clear the encoding region
    for (uint32_t i = table_bytes; i < dst_capacity; i++) dst[i] = 0;

    uint32_t fwd_bytes, bwd_bytes;
    uint8_t* fwd_ptr;
    uint8_t* bwd_ptr;

    tansEncodeBytes(
        fwd_start, bwd_end,
        te, te_data, src, src_len,
        log_table_bits,
        forward_bits, backward_bits,
        fwd_bytes, bwd_bytes,
        fwd_ptr, bwd_ptr
    );

    // Verify sizes match
    if (fwd_bytes + bwd_bytes != payload_bytes) {
        out_sizes[chunk_id] = src_len | 0x80000000u;
        for (uint32_t i = 0; i < src_len && i < dst_capacity; i++)
            dst[i] = src[i];
        return;
    }

    // Assemble final output: table | backward_stream | forward_stream
    // Copy table header to dst[0..]
    for (uint32_t i = 0; i < table_bytes; i++) dst[i] = table_buf[i];

    // We need to move backward and forward streams to their final positions.
    // backward goes to dst[table_bytes .. table_bytes + bwd_bytes]
    // forward goes to dst[table_bytes + bwd_bytes .. total_size]
    //
    // Since the source and destination regions may overlap, we need to be
    // careful. Copy backward stream first (it's at bwd_ptr), then forward.
    // Use a small temp buffer approach — copy via backwards iteration if needed.

    // Assembly into dst: [table | backward_stream | forward_stream]
    // Source regions in dst may overlap with final destinations.
    // Safe strategy: copy backward stream to final position first (source
    // is in the HIGH region of dst, dest is LOW — no overlap with forward source
    // in the LOW-MID region). Then copy forward from end-to-start.
    //
    // BUT: forward dest [table_bytes+bwd_bytes, total_size) may overlap backward
    // source [bwd_ptr, bwd_ptr+bwd_bytes) when the forward stream is large.
    // Solution: copy backward to a safe non-overlapping area first.
    //
    // Use dst[total_size..dst_capacity] as temp for the backward stream
    // (guaranteed not to overlap with the final output [0..total_size]).
    uint8_t* bwd_temp = dst + total_size;
    uint32_t temp_avail = dst_capacity - total_size;
    if (temp_avail >= bwd_bytes) {
        // Stage backward in temp
        for (uint32_t i = 0; i < bwd_bytes; i++) bwd_temp[i] = bwd_ptr[i];

        // Copy forward to final position (backward-direction if overlapping)
        uint8_t* fwd_dst = dst + table_bytes + bwd_bytes;
        if (fwd_dst > fwd_ptr) {
            for (int i = (int)fwd_bytes - 1; i >= 0; i--)
                fwd_dst[i] = fwd_ptr[i];
        } else if (fwd_dst != fwd_ptr) {
            for (uint32_t i = 0; i < fwd_bytes; i++)
                fwd_dst[i] = fwd_ptr[i];
        }

        // Copy backward from temp to final position
        uint8_t* bwd_dst = dst + table_bytes;
        for (uint32_t i = 0; i < bwd_bytes; i++) bwd_dst[i] = bwd_temp[i];
    } else {
        // Not enough temp space — fall back to raw
        out_sizes[chunk_id] = src_len | 0x80000000u;
        for (uint32_t i = 0; i < src_len && i < dst_capacity; i++) dst[i] = src[i];
        return;
    }

    out_sizes[chunk_id] = total_size;
}

// ══════════════════════════════════════════════════════════════════
//  32-lane tANS encoder (chunk_type=6) — mirrors CPU encodeArrayU8Tans32
// ══════════════════════════════════════════════════════════════════
//
// One warp per descriptor. Each lane k encodes the symbols at logical
// indices k, k+32, k+64, ... in REVERSE, then bit-packs LSB-first
// in REVERSE so the decoder reads forward.
//
// Logical symbols are extracted from the source buffer using:
//     src[d.src_offset + (k + j*32) * stride]
// where stride=1 for lit/tok streams and stride=2 (with src_offset
// shifted by 0 or 1) for off16-lo/off16-hi byte-plane extraction.
//
// Output layout per descriptor (matches CPU + GPU decode kernel):
//   [0..64)     32 × u16 LE sub_sizes
//   [64..128)   32 × u16 LE sub_states (initial decode state)
//   [128..)     bit-packed FSE-style probability table
//   [...)       32 concatenated sub-streams
//
// sizes_out[i] is the encoded byte count. If tANS did not beat raw
// (or any error path), the MSB is set as a fallback signal:
//   sizes_out[i] = src_size | 0x80000000  → driver writes type-0
// In that case the output slot is left untouched.

struct Tans32EncDesc {
    uint32_t src_offset;   // offset into d_input
    uint32_t src_size;     // logical symbol count (post-stride)
    uint32_t src_stride;   // 1 for lit/tok, 2 for off16 byte planes
    uint32_t dst_offset;   // offset into d_output
    uint32_t dst_capacity; // max bytes at dst_offset
};

// FSE-style probability table emitter (truncated binary, matches CPU
// loop at tans_encoder.zig:1218-1267). Returns table byte count.
__device__ uint32_t emitTans32ProbabilityTable(
    uint8_t* table_buf,
    const uint32_t* weights,
    uint32_t num_symbols,
    uint32_t log_table_bits
) {
    uint32_t wp = 0;
    table_buf[wp++] = (uint8_t)log_table_bits;

    uint64_t tbl_bits = 0;
    uint32_t tbl_bit_count = 0;
    int32_t  remaining = (int32_t)(1u << log_table_bits);

    for (uint32_t tbl_sym = 0; remaining > 0 && tbl_sym < num_symbols; tbl_sym++) {
        int32_t w = (int32_t)weights[tbl_sym];
        uint32_t value = (uint32_t)(w + 1);

        uint32_t rem_plus_1 = (uint32_t)(remaining + 1);
        uint32_t log2_rem = (rem_plus_1 > 1) ? (31u - __clz(rem_plus_1)) : 0u;
        uint32_t nb = log2_rem + 1;
        uint32_t half = 1u << (nb - 1);
        uint32_t max_val = (half << 1) - (uint32_t)remaining - 2;

        if (value < max_val) {
            tbl_bits |= (uint64_t)value << tbl_bit_count;
            tbl_bit_count += nb - 1;
        } else if (value < half) {
            tbl_bits |= (uint64_t)value << tbl_bit_count;
            tbl_bit_count += nb;
        } else {
            tbl_bits |= (uint64_t)(value + max_val) << tbl_bit_count;
            tbl_bit_count += nb;
        }

        while (tbl_bit_count >= 8) {
            table_buf[wp++] = (uint8_t)(tbl_bits & 0xFF);
            tbl_bits >>= 8;
            tbl_bit_count -= 8;
        }

        if (w > 0) remaining -= w;
    }

    if (tbl_bit_count > 0) {
        table_buf[wp++] = (uint8_t)(tbl_bits & 0xFF);
    }
    return wp;
}

// Maximum per-lane logical symbol count we'll support in one descriptor.
// 64KB sub-chunk / 32 lanes = 2048; 128KB / 32 = 4096. Pad a bit.
#define TANS32_MAX_LANE_COUNT 4128

extern "C" __global__ void __launch_bounds__(32, 4) slzTans32EncodeKernel(
    const uint8_t* __restrict__ src_buf,
    uint8_t*       __restrict__ dst_buf,
    const Tans32EncDesc* __restrict__ descs,
    uint32_t*      __restrict__ sizes_out,
    uint32_t       num_descs
) {
    const uint32_t did = blockIdx.x;
    if (did >= num_descs) return;
    const int lane = (int)(threadIdx.x & 31);

    // ── Load descriptor + early-out for tiny streams ─────────────────
    Tans32EncDesc d = descs[did];
    if (d.src_size < 64) {
        // CPU returns error.TansNotBeneficial; signal raw-fallback.
        if (lane == 0) sizes_out[did] = d.src_size | 0x80000000u;
        return;
    }

    const uint8_t* src = src_buf + d.src_offset;
    uint8_t* dst = dst_buf + d.dst_offset;

    // ── log_table_bits = clamp(ilog2Round(src_size) - 2, 8, 11) ──────
    __shared__ uint32_t s_log_table_bits;
    __shared__ uint32_t s_num_symbols;
    __shared__ uint32_t s_used_symbols;
    __shared__ uint32_t s_table_bytes;
    __shared__ uint32_t s_abort;
    __shared__ uint32_t histo[256];
    __shared__ uint32_t weights[256];
    __shared__ TansEncEntry te[256];
    __shared__ uint16_t te_data[2048];
    __shared__ uint8_t  table_buf[512];
    __shared__ uint16_t sub_sizes[32];
    __shared__ uint16_t sub_states[32];

    if (lane == 0) {
        int raw_log = (int)ilog2Round(d.src_size) - 2;
        if (raw_log > 11) raw_log = 11;
        if (raw_log < 8) raw_log = 8;
        s_log_table_bits = (uint32_t)raw_log;
        s_abort = 0;
    }

    // ── Zero histogram (256 entries = 8 per lane) ────────────────────
    for (int i = 0; i < 8; i++) histo[lane * 8 + i] = 0;
    __syncwarp();

    // ── Histogram pass: each lane counts its stride ──────────────────
    const uint32_t src_size = d.src_size;
    const uint32_t stride = d.src_stride;
    if (stride == 1) {
        for (uint32_t j = (uint32_t)lane; j < src_size; j += 32) {
            atomicAdd_block(&histo[src[j]], 1);
        }
    } else {
        for (uint32_t j = (uint32_t)lane; j < src_size; j += 32) {
            atomicAdd_block(&histo[src[j * stride]], 1);
        }
    }
    __syncwarp();

    // ── Lane 0: find num_symbols, normalize, build encoding table ────
    if (lane == 0) {
        uint32_t num_symbols = 256;
        while (num_symbols > 0 && histo[num_symbols - 1] == 0) num_symbols--;
        s_num_symbols = num_symbols;
        if (num_symbols == 0) {
            sizes_out[did] = d.src_size | 0x80000000u;
            s_abort = 1;
        } else {
            uint32_t L = 1u << s_log_table_bits;
            uint32_t used = tansNormalizeCounts(weights, L, histo, d.src_size, (int)num_symbols);
            s_used_symbols = used;
            if (used <= 1) {
                sizes_out[did] = d.src_size | 0x80000000u;
                s_abort = 1;
            } else {
                tansInitTable(te, te_data, weights, (int)num_symbols, s_log_table_bits);
                s_table_bytes = emitTans32ProbabilityTable(table_buf, weights, num_symbols, s_log_table_bits);
            }
        }
    }
    __syncwarp();
    if (s_abort != 0) return;

    const uint32_t log_table_bits = s_log_table_bits;
    const uint32_t L = 1u << log_table_bits;
    const uint32_t table_bytes = s_table_bytes;

    // ── Phase 1: lane-local encode in REVERSE, collect (nb, bits) ────
    // lane_count = ceil((src_size - lane) / 32)
    int32_t signed_count = (int32_t)src_size - lane;
    uint32_t lane_count = (signed_count > 0) ? (uint32_t)((signed_count + 31) / 32) : 0u;

    // Per-lane local arrays (spill to local memory; ~12 KB per lane worst case)
    uint8_t  enc_nb[TANS32_MAX_LANE_COUNT];
    uint32_t enc_bits[TANS32_MAX_LANE_COUNT];

    uint32_t lane_state = L;
    uint32_t enc_count = 0;

    if (lane_count > 0) {
        // ri walks from (lane_count-1) down to 0; symbol index = ri*32 + lane.
        // We bail early if lane_count > TANS32_MAX_LANE_COUNT (caller's bug).
        uint32_t cap = (lane_count <= TANS32_MAX_LANE_COUNT) ? lane_count : TANS32_MAX_LANE_COUNT;
        for (uint32_t ri = cap; ri > 0; ) {
            ri--;
            uint32_t sym_idx = ri * 32u + (uint32_t)lane;
            // sym_idx is always < src_size because lane_count was computed to ensure that.
            uint8_t sym = src[sym_idx * stride];
            const TansEncEntry* entry = &te[sym];

            uint32_t new_state;
            uint32_t nb;
            nextState(te_data, entry, lane_state, new_state, nb);

            uint32_t mask = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1u);
            enc_nb[enc_count]   = (uint8_t)nb;
            enc_bits[enc_count] = lane_state & mask;
            enc_count++;
            lane_state = new_state;
        }
    }

    // ── Phase 2: pack bits LSB-first in REVERSE (write to lane scratch) ──
    // We pack into a small per-lane byte buffer in local memory, then memcpy
    // out to dst once sub-stream offsets are known via prefix-sum.
    // Worst case: src_size/32 + 8 bytes per lane (~520B for 4KB stream).
    // Use 4144 = TANS32_MAX_LANE_COUNT + slack so 4-byte bursts can overshoot.
    uint8_t lane_buf[TANS32_MAX_LANE_COUNT + 64];

    uint64_t bit_buf = 0;
    uint32_t bit_count = 0;
    uint32_t wp_local = 0;

    for (uint32_t ei = enc_count; ei > 0; ) {
        ei--;
        bit_buf |= (uint64_t)enc_bits[ei] << bit_count;
        bit_count += enc_nb[ei];
        while (bit_count >= 8) {
            lane_buf[wp_local++] = (uint8_t)(bit_buf & 0xFF);
            bit_buf >>= 8;
            bit_count -= 8;
        }
    }
    if (bit_count > 0) {
        lane_buf[wp_local++] = (uint8_t)(bit_buf & 0xFF);
    }

    if (wp_local > 0xFFFF) {
        // CPU returns error.TansNotBeneficial when a single lane's sub-stream
        // exceeds u16 range. Signal raw-fallback (only lane 0 writes).
        if (lane == 0) sizes_out[did] = d.src_size | 0x80000000u;
        return;
    }

    sub_sizes[lane]  = (uint16_t)wp_local;
    sub_states[lane] = (uint16_t)(lane_state & (L - 1));
    __syncwarp();

    // ── Prefix sum of sub_sizes (intra-warp scan) ────────────────────
    uint32_t my_size = (uint32_t)sub_sizes[lane];
    uint32_t scan = my_size;
    for (int o = 1; o < 32; o <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, scan, o);
        if (lane >= o) scan += n;
    }
    uint32_t my_offset = scan - my_size;
    uint32_t total_compressed = __shfl_sync(0xFFFFFFFF, scan, 31);

    const uint32_t header_size = 128u + table_bytes;
    const uint32_t total_size = header_size + total_compressed;

    // ── Reject if tANS didn't beat raw OR overflows dst capacity ─────
    if (total_size + 8 > d.dst_capacity || total_size >= d.src_size) {
        if (lane == 0) sizes_out[did] = d.src_size | 0x80000000u;
        return;
    }

    // ── Lane 0: write sizes+states header + probability table ────────
    if (lane == 0) {
        // 32 × u16 LE sizes
        for (int i = 0; i < 32; i++) {
            dst[i * 2]     = (uint8_t)(sub_sizes[i] & 0xFF);
            dst[i * 2 + 1] = (uint8_t)((sub_sizes[i] >> 8) & 0xFF);
        }
        // 32 × u16 LE states
        for (int i = 0; i < 32; i++) {
            dst[64 + i * 2]     = (uint8_t)(sub_states[i] & 0xFF);
            dst[64 + i * 2 + 1] = (uint8_t)((sub_states[i] >> 8) & 0xFF);
        }
        // probability table
        for (uint32_t i = 0; i < table_bytes; i++) {
            dst[128 + i] = table_buf[i];
        }
        sizes_out[did] = total_size;
    }
    __syncwarp();

    // ── Each lane writes its sub-stream to dst ──────────────────────
    uint8_t* my_dst = dst + header_size + my_offset;
    for (uint32_t i = 0; i < wp_local; i++) {
        my_dst[i] = lane_buf[i];
    }
}
