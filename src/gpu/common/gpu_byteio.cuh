// ── StreamLZ GPU — little/big-endian byte-IO primitives ─────────
// Shared by every CUDA kernel in src/gpu/. Pure header: #pragma once,
// inline device helpers only, no kernels and no translation-unit
// state. #include'd via "../common/gpu_byteio.cuh" into the existing
// single .cu translation units.
//
// Endianness rule (the two conventions used by the wire format):
//   - LZ sub-stream count headers : BIG-endian u24  -> readBE24
//   - Huffman 4-stream sub-header  : LITTLE-endian u24 -> readLE24 /
//     writeLE24
// Both are correct; keeping each call site on a named helper makes the
// convention self-documenting.
#pragma once

#include <cstdint>

// ── 24-bit big-endian read ──────────────────────────────────────
// LZ stream headers store sub-stream counts big-endian.
__device__ __forceinline__ uint32_t readBE24(const uint8_t* p) {
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
}

// ── 24-bit little-endian codec ──────────────────────────────────
// The Huffman 4-stream sub-header stores per-stream sizes little-endian.
__device__ __forceinline__ uint32_t readLE24(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
}

__device__ __forceinline__ void writeLE24(uint8_t* p, uint32_t value) {
    p[0] = (uint8_t)(value & 0xFF);
    p[1] = (uint8_t)((value >> 8) & 0xFF);
    p[2] = (uint8_t)((value >> 16) & 0xFF);
}

// ── 16-bit little-endian load / store (unaligned-safe) ──────────
__device__ __forceinline__ uint16_t readU16LE(const uint8_t* p) {
    uint16_t v; memcpy(&v, p, 2); return v;
}

__device__ __forceinline__ void storeU16LE(uint8_t* dst, uint16_t value) {
    memcpy(dst, &value, 2);
}

// ── Read up to 8 bytes, zero-padding past src_size ──────────────
// Generic zero-padded tail read used by the LZ encode hash path.
__device__ __forceinline__ uint64_t read8safe(const uint8_t* p, uint32_t pos,
                                              uint32_t src_size) {
    uint64_t v = 0;
    uint32_t avail = (pos + 8 <= src_size) ? 8 : (src_size > pos ? src_size - pos : 0);
    memcpy(&v, p, avail);
    return v;
}
