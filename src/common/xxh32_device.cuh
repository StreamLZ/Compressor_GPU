// Device-side XXH32 (seed 0) — exact mirror of src/format/xxhash32.zig
// (the verified host implementation; spec vectors "" = 0x02CC5D05,
// "a" = 0x550D7456, "abc" = 0x32D153FF).
//
// v4 #19: used by slzChunkHashKernel to compute per-chunk content
// hashes in parallel (one thread per chunk; each thread runs the
// serial XXH32 over its 64-256 KB chunk; 1500+ concurrent chunks make
// the pass ~free). The frame stores ONE u32: XXH32 over the
// concatenated per-chunk u32 array (LE, chunk-index order) — the
// Merkle-root construction. The per-chunk values never hit the wire.
#pragma once

#include <stdint.h>

#define XXH32_PRIME1 0x9E3779B1u
#define XXH32_PRIME2 0x85EBCA77u
#define XXH32_PRIME3 0xC2B2AE3Du
#define XXH32_PRIME4 0x27D4EB2Fu
#define XXH32_PRIME5 0x165667B1u

__device__ __forceinline__ uint32_t xxh32Rotl(uint32_t x, uint32_t r) {
    return (x << r) | (x >> (32u - r));
}

__device__ __forceinline__ uint32_t xxh32Round(uint32_t acc, uint32_t input) {
    return xxh32Rotl(acc + input * XXH32_PRIME2, 13) * XXH32_PRIME1;
}

// Unaligned-safe little-endian u32 read (chunk starts are eff_chunk
// aligned but the data pointer itself may sit at any offset).
__device__ __forceinline__ uint32_t xxh32ReadU32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ inline uint32_t xxh32Device(const uint8_t* __restrict__ data, uint32_t len) {
    uint32_t h;
    uint32_t pos = 0;

    if (len >= 16) {
        uint32_t v1 = XXH32_PRIME1 + XXH32_PRIME2;
        uint32_t v2 = XXH32_PRIME2;
        uint32_t v3 = 0;
        uint32_t v4 = 0u - XXH32_PRIME1;
        while (pos + 16 <= len) {
            v1 = xxh32Round(v1, xxh32ReadU32(data + pos));
            v2 = xxh32Round(v2, xxh32ReadU32(data + pos + 4));
            v3 = xxh32Round(v3, xxh32ReadU32(data + pos + 8));
            v4 = xxh32Round(v4, xxh32ReadU32(data + pos + 12));
            pos += 16;
        }
        h = xxh32Rotl(v1, 1) + xxh32Rotl(v2, 7) + xxh32Rotl(v3, 12) + xxh32Rotl(v4, 18);
    } else {
        h = XXH32_PRIME5;
    }
    h += len;

    while (pos + 4 <= len) {
        h += xxh32ReadU32(data + pos) * XXH32_PRIME3;
        h = xxh32Rotl(h, 17) * XXH32_PRIME4;
        pos += 4;
    }
    while (pos < len) {
        h += (uint32_t)data[pos] * XXH32_PRIME5;
        h = xxh32Rotl(h, 11) * XXH32_PRIME1;
        pos += 1;
    }

    h ^= h >> 15;
    h *= XXH32_PRIME2;
    h ^= h >> 13;
    h *= XXH32_PRIME3;
    h ^= h >> 16;
    return h;
}

// Variant: the first `head_len` (<= 8) bytes come from `head`, the
// rest from data[head_len..len). Used by the DECODE-side verify: the
// LZ kernel leaves garbage in the first 8 bytes of chunks 1+ (the
// SC-prefix post-pass restores them host-side after D2H), but the
// true prefix bytes are available host-side from the frame - so the
// hash kernel splices them in instead of requiring a device-side
// post-pass. Builds the first 16-byte stripe in registers, then
// continues on `data` for the remainder.
__device__ inline uint32_t xxh32DeviceWithHead(
    const uint8_t* __restrict__ head, uint32_t head_len,
    const uint8_t* __restrict__ data, uint32_t len
) {
    if (head_len == 0) return xxh32Device(data, len);
    // Materialize a 32-byte local prologue (head + following data
    // bytes); beyond it the stream is identical to `data`.
    uint8_t pro[32];
    uint32_t pro_len = len < 32 ? len : 32;
    for (uint32_t i = 0; i < pro_len; i++)
        pro[i] = (i < head_len) ? head[i] : data[i];

    uint32_t h;
    uint32_t pos = 0;
    if (len >= 16) {
        uint32_t v1 = XXH32_PRIME1 + XXH32_PRIME2;
        uint32_t v2 = XXH32_PRIME2;
        uint32_t v3 = 0;
        uint32_t v4 = 0u - XXH32_PRIME1;
        // Stripes inside the prologue window:
        while (pos + 16 <= pro_len) {
            v1 = xxh32Round(v1, xxh32ReadU32(pro + pos));
            v2 = xxh32Round(v2, xxh32ReadU32(pro + pos + 4));
            v3 = xxh32Round(v3, xxh32ReadU32(pro + pos + 8));
            v4 = xxh32Round(v4, xxh32ReadU32(pro + pos + 12));
            pos += 16;
        }
        // Remaining full stripes straight from data:
        while (pos + 16 <= len) {
            v1 = xxh32Round(v1, xxh32ReadU32(data + pos));
            v2 = xxh32Round(v2, xxh32ReadU32(data + pos + 4));
            v3 = xxh32Round(v3, xxh32ReadU32(data + pos + 8));
            v4 = xxh32Round(v4, xxh32ReadU32(data + pos + 12));
            pos += 16;
        }
        h = xxh32Rotl(v1, 1) + xxh32Rotl(v2, 7) + xxh32Rotl(v3, 12) + xxh32Rotl(v4, 18);
    } else {
        h = XXH32_PRIME5;
    }
    h += len;
    while (pos + 4 <= len) {
        const uint8_t* src4 = (pos + 4 <= pro_len) ? (pro + pos) : (data + pos);
        h += xxh32ReadU32(src4) * XXH32_PRIME3;
        h = xxh32Rotl(h, 17) * XXH32_PRIME4;
        pos += 4;
    }
    while (pos < len) {
        uint8_t b = (pos < pro_len) ? pro[pos] : data[pos];
        h += (uint32_t)b * XXH32_PRIME5;
        h = xxh32Rotl(h, 11) * XXH32_PRIME1;
        pos += 1;
    }
    h ^= h >> 15;
    h *= XXH32_PRIME2;
    h ^= h >> 13;
    h *= XXH32_PRIME3;
    h ^= h >> 16;
    return h;
}


// ── v4 #19 hierarchical chunk hash ───────────────────────────────
// chunk_hash := XXH32( concat of XXH32(seg_j) ), seg_j = the chunk's
// j-th 4096-byte segment (last one partial). The two-level shape
// exists for GPU parallelism: one THREAD per 4 KiB segment (24k
// threads at 100 MB) instead of one per 64 KiB chunk (1.5k threads,
// measured 3.4 ms - the serial XXH32 chain cannot be split, so the
// parallelism has to come from the definition). The host fallback
// (format/xxhash32.zig chunkMerkleRoot) implements the identical
// definition.
#define SLZ_MERKLE_SEG_BYTES 1024u
#define SLZ_MERKLE_SEGS_PER_CHUNK(eff) (((eff) + SLZ_MERKLE_SEG_BYTES - 1u) / SLZ_MERKLE_SEG_BYTES)

// One thread per (chunk, segment) slot. seg_hashes is laid out as
// [chunk][segs_per_chunk_max]; unused tail slots are left untouched
// (the combine kernel only reads the live ones).
#define SLZ_SEG_HASH_KERNEL_BODY                                            \
    {                                                                       \
        const uint32_t spc = SLZ_MERKLE_SEGS_PER_CHUNK(eff_chunk);          \
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;         \
        const uint32_t ci = gid / spc;                                      \
        const uint32_t si = gid % spc;                                      \
        if (ci >= n_chunks) return;                                         \
        const uint64_t chunk_start = (uint64_t)ci * eff_chunk;              \
        const uint64_t chunk_rem = total_size - chunk_start;                \
        const uint32_t chunk_len = chunk_rem < eff_chunk ? (uint32_t)chunk_rem : eff_chunk; \
        const uint32_t seg_start = si * SLZ_MERKLE_SEG_BYTES;               \
        if (seg_start >= chunk_len) return;                                 \
        uint32_t seg_len = chunk_len - seg_start;                           \
        if (seg_len > SLZ_MERKLE_SEG_BYTES) seg_len = SLZ_MERKLE_SEG_BYTES; \
        const uint8_t* seg = data + chunk_start + seg_start;                \
        uint32_t hv;                                                        \
        if (prefix_table != 0 && ci > 0 && si == 0) {                       \
            const uint8_t* head = prefix_table + (uint64_t)(ci - 1) * 8;    \
            const uint32_t hl = seg_len < 8 ? seg_len : 8;                  \
            hv = xxh32DeviceWithHead(head, hl, seg, seg_len);               \
        } else {                                                            \
            hv = xxh32Device(seg, seg_len);                                 \
        }                                                                   \
        seg_hashes[(uint64_t)ci * spc + si] = hv;                           \
    }

// One thread per chunk: combine its live segment hashes.
#define SLZ_CHUNK_COMBINE_KERNEL_BODY                                       \
    {                                                                       \
        const uint32_t spc = SLZ_MERKLE_SEGS_PER_CHUNK(eff_chunk);          \
        const uint32_t ci = blockIdx.x * blockDim.x + threadIdx.x;          \
        if (ci >= n_chunks) return;                                         \
        const uint64_t chunk_start = (uint64_t)ci * eff_chunk;              \
        const uint64_t chunk_rem = total_size - chunk_start;                \
        const uint32_t chunk_len = chunk_rem < eff_chunk ? (uint32_t)chunk_rem : eff_chunk; \
        const uint32_t live = (chunk_len + SLZ_MERKLE_SEG_BYTES - 1u) / SLZ_MERKLE_SEG_BYTES; \
        out_hashes[ci] = xxh32Device(                                       \
            (const uint8_t*)(seg_hashes + (uint64_t)ci * spc), live * 4u);  \
    }
