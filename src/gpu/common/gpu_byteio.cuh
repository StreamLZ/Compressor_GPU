// ── StreamLZ GPU — little/big-endian byte-IO primitives ─────────
// Shared by every CUDA kernel in src/gpu/. Pure header: #pragma once,
// inline device helpers only, no kernels and no translation-unit
// state. #include'd via "../common/gpu_byteio.cuh" into the existing
// single .cu translation units.
//
// The wire format mixes big-endian and little-endian fields. Each call
// site picks a named helper from this header — readBE24 / readU32BE for
// big-endian fields, readU16LE / readU32LE / readU64LE / readLE24 for
// little-endian fields — so the convention is self-documenting at the
// read/write site rather than buried in an open-coded byte shift.
#pragma once

#include <cstdint>

// ── 24-bit big-endian read / write ──────────────────────────────
// LZ stream headers store sub-stream counts big-endian.
__device__ __forceinline__ uint32_t readBE24(const uint8_t* p) {
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
}

__device__ __forceinline__ void writeBE24(uint8_t* p, uint32_t value) {
    p[0] = (uint8_t)((value >> 16) & 0xFF);
    p[1] = (uint8_t)((value >> 8) & 0xFF);
    p[2] = (uint8_t)(value & 0xFF);
}

// ── 32-bit big-endian read ──────────────────────────────────────
// Entropy long-form headers load 4 bytes as a single BE word.
__device__ __forceinline__ uint32_t readU32BE(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16)
         | ((uint32_t)p[2] << 8)  |  (uint32_t)p[3];
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

// ── 32-bit little-endian load / store (unaligned-safe) ──────────
// readU32LE — 4-byte LE load. Used pervasively: LZ match-finder hot loop,
// frame-walk kernel (block / chunk header fields), assemble kernel
// (chunk-internal header). writeU32LE — 4-byte LE store, used by the
// assemble kernel (chunk-internal header + end mark).
__device__ __forceinline__ uint32_t readU32LE(const uint8_t* p) {
    uint32_t v; memcpy(&v, p, 4); return v;
}

__device__ __forceinline__ void writeU32LE(uint8_t* p, uint32_t value) {
    memcpy(p, &value, 4);
}

// ── 64-bit little-endian load (unaligned-safe) ──────────────────
// Caller must have proven `p + 8` is in-bounds; for the zero-padded
// tail-safe variant use `read8safe` below.
__device__ __forceinline__ uint64_t readU64LE(const uint8_t* p) {
    uint64_t v; memcpy(&v, p, 8); return v;
}

// ── Read up to 8 bytes, zero-padding past src_size ──────────────
// Generic zero-padded tail read used by the LZ encode hash path. Reads
// 8 bytes from `base + pos` when in-range, or `src_size - pos` bytes
// followed by zero-padding when the tail is short.
//
// SAFETY: caller must guarantee `pos <= src_size` (otherwise the
// `src_size - pos` subtraction underflows). The encode parsers satisfy
// this because they only call read8safe inside `while (pos < src_size)`
// loops or after a `if (pos < src_size)` guard.
__device__ __forceinline__ uint64_t read8safe(const uint8_t* base, uint32_t pos,
                                              uint32_t src_size) {
    uint64_t v = 0;
    uint32_t avail = (pos + 8 <= src_size) ? 8 : (src_size > pos ? src_size - pos : 0);
    memcpy(&v, base + pos, avail);
    return v;
}
