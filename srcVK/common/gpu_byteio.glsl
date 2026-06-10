// 1:1 port of src/common/gpu_byteio.cuh.
// Endian byte-IO primitives: readBE24, readU32LE, writeBE24, etc.
//
// GLSL adaptation: CUDA's helpers take `const uint8_t* p`. GLSL has no
// pointer types in the host C/C++ sense; the canonical srcVK pattern is
// for the kernel to load each byte from its SSBO via a helper like
// `loadByte(off)` and then pass the loaded byte values into the
// pack/unpack helpers below. Each function below carries the CUDA name
// (`readBE24`, `readU32LE`, etc.); the only signature drift is that
// `const uint8_t* p` becomes an explicit list of byte arguments. The
// arithmetic is identical to the CUDA implementation.
//

#ifndef SRCVK_COMMON_GPU_BYTEIO_GLSL
#define SRCVK_COMMON_GPU_BYTEIO_GLSL

// CUDA reference: src/common/gpu_byteio.cuh:18-20. 24-bit big-endian
// read — LZ stream headers store sub-stream counts big-endian.
uint readBE24(uint p0, uint p1, uint p2) {
    return (p0 << 16) | (p1 << 8) | p2;
}

// CUDA reference: src/common/gpu_byteio.cuh:22-26. 24-bit big-endian
// pack — emits three bytes via the inout array `p` indexed 0..2.
void writeBE24(out uint p0, out uint p1, out uint p2, uint value) {
    p0 = (value >> 16) & 0xFFu;
    p1 = (value >> 8) & 0xFFu;
    p2 = value & 0xFFu;
}

// CUDA reference: src/common/gpu_byteio.cuh:30-33. 32-bit big-endian
// read — entropy long-form headers load 4 bytes as a single BE word.
uint readU32BE(uint p0, uint p1, uint p2, uint p3) {
    return (p0 << 24) | (p1 << 16) | (p2 << 8) | p3;
}

// CUDA reference: src/common/gpu_byteio.cuh:38-40. 24-bit little-endian
// read — the Huffman sub-header stores per-stream sizes little-endian
// (HUFF_NUM_STREAMS u24 LE).
uint readLE24(uint p0, uint p1, uint p2) {
    return p0 | (p1 << 8) | (p2 << 16);
}

// CUDA reference: src/common/gpu_byteio.cuh:42-46. 24-bit little-endian
// pack — emits three bytes.
void writeLE24(out uint p0, out uint p1, out uint p2, uint value) {
    p0 = value & 0xFFu;
    p1 = (value >> 8) & 0xFFu;
    p2 = (value >> 16) & 0xFFu;
}

// CUDA reference: src/common/gpu_byteio.cuh:49-51. 16-bit little-endian
// read.
uint readU16LE(uint p0, uint p1) {
    return p0 | (p1 << 8);
}

// CUDA reference: src/common/gpu_byteio.cuh:53-55. 16-bit little-endian
// pack.
void storeU16LE(out uint p0, out uint p1, uint value) {
    p0 = value & 0xFFu;
    p1 = (value >> 8) & 0xFFu;
}

// CUDA reference: src/common/gpu_byteio.cuh:62-64. 32-bit little-endian
// read — used pervasively by the LZ match-finder hot loop, frame-walk
// kernel, and assemble kernel.
uint readU32LE(uint p0, uint p1, uint p2, uint p3) {
    return p0 | (p1 << 8) | (p2 << 16) | (p3 << 24);
}

// CUDA reference: src/common/gpu_byteio.cuh:66-68. 32-bit little-endian
// pack.
void writeU32LE(out uint p0, out uint p1, out uint p2, out uint p3, uint value) {
    p0 = value & 0xFFu;
    p1 = (value >> 8) & 0xFFu;
    p2 = (value >> 16) & 0xFFu;
    p3 = (value >> 24) & 0xFFu;
}

// CUDA reference: src/common/gpu_byteio.cuh:73-75. 64-bit little-endian
// read — caller must have proven `p + 8` is in-bounds.
uvec2 readU64LE(uint p0, uint p1, uint p2, uint p3, uint p4, uint p5, uint p6, uint p7) {
    // Returns (lo, hi) as a uvec2 matching CUDA's uint64_t result. The
    // host can recombine via packUint2x32(uvec2(lo, hi)).
    uint lo = p0 | (p1 << 8) | (p2 << 16) | (p3 << 24);
    uint hi = p4 | (p5 << 8) | (p6 << 16) | (p7 << 24);
    return uvec2(lo, hi);
}

// CUDA reference: src/common/gpu_byteio.cuh:86-92. Read up to 8 bytes,
// zero-padding past `src_size`. Generic zero-padded tail read used by
// the LZ encode hash path; CUDA's signature is
//   uint64_t read8safe(const uint8_t* base, uint32_t pos, uint32_t src_size)
// and it returns 8 bytes from `base + pos` when in-range, or
// `src_size - pos` bytes followed by zero-padding when the tail is short.
// SAFETY: caller must guarantee `pos <= src_size` (CUDA's identical
// precondition — the `src_size - pos` subtraction would otherwise
// underflow). Signature adaptation: GLSL has no pointer/`memcpy`, so the
// kernel loads the eight candidate bytes from its SSBO (zero-filling past
// EOF is the kernel's responsibility OR is performed here by masking out
// bytes whose absolute index `pos + k >= src_size`). The returned uvec2
// is (lo, hi) matching CUDA's uint64_t (use packUint2x32 to recombine).
uvec2 read8safe(uint p0, uint p1, uint p2, uint p3, uint p4, uint p5, uint p6, uint p7,
                uint pos, uint src_size) {
    uint avail = (pos + 8u <= src_size) ? 8u : (src_size > pos ? (src_size - pos) : 0u);
    uint b0 = (0u < avail) ? p0 : 0u;
    uint b1 = (1u < avail) ? p1 : 0u;
    uint b2 = (2u < avail) ? p2 : 0u;
    uint b3 = (3u < avail) ? p3 : 0u;
    uint b4 = (4u < avail) ? p4 : 0u;
    uint b5 = (5u < avail) ? p5 : 0u;
    uint b6 = (6u < avail) ? p6 : 0u;
    uint b7 = (7u < avail) ? p7 : 0u;
    uint lo = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    uint hi = b4 | (b5 << 8) | (b6 << 16) | (b7 << 24);
    return uvec2(lo, hi);
}

#endif
