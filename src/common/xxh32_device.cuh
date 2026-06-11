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

// ── Per-chunk hash kernel body ───────────────────────────────────
// Chunk i covers decompressed bytes [i*eff_chunk, min((i+1)*eff_chunk,
// total)) of `data`. One thread per chunk. Included by both the
// encode and decode translation units (each module exports its own
// copy of the kernel under the same symbol name).
#define SLZ_CHUNK_HASH_KERNEL_BODY                                          \
    {                                                                       \
        const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;           \
        if (i >= n_chunks) return;                                          \
        const uint64_t start = (uint64_t)i * eff_chunk;                     \
        const uint64_t remain = total_size - start;                         \
        const uint32_t len = remain < eff_chunk ? (uint32_t)remain          \
                                                : eff_chunk;                \
        out_hashes[i] = xxh32Device(data + start, len);                     \
    }
