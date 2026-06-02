//! Shared wire-format + chunk-geometry constants for the Vulkan codec.
//!
//! This file is THE source of truth for every magic number that crosses
//! the Zig/GLSL boundary in the L1 codec + 5-kernel GPU decode pipeline.
//! Cluster H of `recon2.md` consolidates the previously-duplicated copies
//! across:
//!
//!   * `src_vulkan/l1_codec.zig`                — host-side L1 codec
//!   * `src_vulkan/wire_format.zig`             — CPU SLZ1 wrap/unwrap
//!   * `src_vulkan/wire_format_gpu.zig`         — GPU SLZ1 wrap
//!   * `src_vulkan/decode_pipeline_gpu.zig`     — 5-kernel orchestrator
//!   * `src_vulkan/shaders/decode_pipeline_shared.glsl` — GLSL header
//!   * `src_vulkan/shaders/lz_encode.comp`      — encoder kernel
//!   * `src_vulkan/shaders/lz_decode.comp`      — decoder kernel
//!
//! The GLSL header (decode_pipeline_shared.glsl) MUST mirror these values.
//! Each constant carries a brief description of the matching GLSL token
//! name; any change here must be reflected in the GLSL header (and vice
//! versa). The comptime block at the bottom of this file cross-checks the
//! invariants that don't depend on the GLSL side.

const std = @import("std");

// ── Chunk / block geometry ───────────────────────────────────────────
//
// CHUNK_SIZE        — per-chunk source bytes (L1 codec).
// LZ_BLOCK_SIZE     — boundary between LZ blocks 0 and 1 inside a
//                     sub-chunk (== CHUNK_SIZE today).
// sc_group_size     — sub-chunk-group fraction stamped into the SLZ1
//                     frame header. Multiplies SLZ_CHUNK_SIZE_BYTES
//                     (256 KiB) to yield the per-sub-chunk decompressed
//                     size (64 KiB at 0.25).
// SLZ_CHUNK_SIZE_BYTES — wire-format outer block boundary (256 KiB),
//                         mirrored from `src/common/gpu_wire_format.cuh`.
//
// GLSL header mirror: decode_pipeline_shared.glsl
//   const uint LZ_BLOCK_SIZE        = 0x10000u;
//   const uint SLZ_CHUNK_SIZE_BYTES = 262144u;

pub const CHUNK_SIZE: u32 = 0x10000; // 64 KiB
pub const LZ_BLOCK_SIZE: u32 = 0x10000; // 64 KiB
pub const SC_GROUP_SIZE: f32 = 0.25;
pub const SLZ_CHUNK_SIZE_BYTES: u32 = 262144;

// ── Initial-prefix + recent-offset wire constants ────────────────────
//
// INITIAL_LITERAL_COPY_BYTES — first chunk carries 8 verbatim source
//                              bytes ahead of the token stream.
// INITIAL_RECENT_OFFSET      — encoder/decoder start with recent_offset = -8.
//
// GLSL header mirror:
//   const uint INITIAL_LITERAL_COPY_BYTES = 8u;
//   const int  INITIAL_RECENT_OFFSET      = -8;

pub const INITIAL_LITERAL_COPY_BYTES: u32 = 8;
pub const INITIAL_RECENT_OFFSET: i32 = -8;

// ── Sub-chunk header (3-byte BE u24) ────────────────────────────────
//
// GLSL header mirror:
//   const uint SUBCHUNK_HDR_BYTES   = 3u;
//   const uint SUBCHUNK_LZ_FLAG_BIT = 0x800000u;
//   const uint SUBCHUNK_MODE_SHIFT  = 19u;
//   const uint SUBCHUNK_MODE_MASK   = 0xFu;
//   const uint SUBCHUNK_COMP_SIZE_MASK = 0x7FFFFu;

pub const SUBCHUNK_HDR_BYTES: u32 = 3;
pub const SUBCHUNK_LZ_FLAG_BIT: u32 = 0x800000;
pub const SUBCHUNK_MODE_SHIFT: u5 = 19;
pub const SUBCHUNK_MODE_MASK: u32 = 0xF;
pub const SUBCHUNK_COMP_SIZE_MASK: u32 = 0x7FFFF;

// ── Off16 / Off32 wire-format constants ─────────────────────────────
//
// OFF16_ENTRY_BYTES         — each off16 entry on the wire.
// OFF16_ENTROPY_MARKER      — sentinel meaning "entropy-coded" (== 0xFFFF).
// OFF32_ENTRY_BYTES         — each off32 entry on the wire (3 bytes today).
// OFF32_COUNT_FIELD_BITS    — width of one block's packed-count field.
// OFF32_COUNT_PACK_MAX      — max count fitting in OFF32_COUNT_FIELD_BITS.
// OFF32_LONG_ENTRY_TAG      — extended-form tag byte.
//
// GLSL header mirror:
//   const uint OFF16_ENTRY_BYTES         = 2u;          (lz_decode.comp)
//   const uint OFF32_ENTRY_BYTES         = 3u;          (lz_encode.comp, lz_decode.comp)
//   const uint OFF16_ENTROPY_MARKER      = 0xFFFFu;
//   const uint OFF32_COUNT_FIELD_BITS    = 12u;
//   const uint OFF32_COUNT_PACK_MAX      = 0xFFFu;
//   const uint OFF32_LONG_ENTRY_TAG      = 0xC0u;

pub const OFF16_ENTRY_BYTES: u32 = 2;
pub const OFF32_ENTRY_BYTES: u32 = 3;
pub const OFF16_ENTROPY_MARKER: u32 = 0xFFFF;
pub const OFF32_COUNT_FIELD_BITS: u5 = 12;
pub const OFF32_COUNT_PACK_MAX: u32 = (1 << OFF32_COUNT_FIELD_BITS) - 1; // 4095
pub const OFF32_LONG_ENTRY_TAG: u32 = 0xC0;

// ── Encoder match / token constants (mirror lz_encode.comp) ─────────
//
// FAR_OFFSET_MIN_MATCH    — minimum match length required to emit an
//                            off32 (far-offset) entry.
// LARGE_OFFSET_THRESHOLD  — encoder switches from 3-byte to 4-byte
//                            off32 form when adjusted_offset crosses
//                            this threshold.
//
// GLSL header mirror (lz_encode.comp lines ~104, ~138):
//   const uint FAR_OFFSET_MIN_MATCH    = 14u;
//   const uint LARGE_OFFSET_THRESHOLD  = 0xC00000u;

pub const FAR_OFFSET_MIN_MATCH: u32 = 14;
pub const LARGE_OFFSET_THRESHOLD: u32 = 0xC00000;

// ── Hash multiplier constants (mirror lz_encode.comp) ───────────────
//
// FIB_MUL_LO / FIB_MUL_HI — low/high 32-bit halves of the 64-bit
// Fibonacci hash multiplier used by the encoder's hash kernel.
//
// FIB_HASH_MUL_K6 = 0x79B97F4A7C150000ULL (= 0x9E3779B97F4A7C15ULL << 16).
//
// GLSL header mirror (lz_encode.comp lines ~144-145):
//   const uint FIB_MUL_LO = 0x7C150000u;
//   const uint FIB_MUL_HI = 0x79B97F4Au;

pub const FIB_MUL_LO: u32 = 0x7C150000;
pub const FIB_MUL_HI: u32 = 0x79B97F4A;

// ── SlzChunkDesc slot layout (mirror decode_pipeline_shared.glsl) ────
//
// 6 u32 per ChunkDesc emitted by walk_frame.comp; the slot indices
// are byte-identical to `decode_pipeline_shared.glsl::CHUNK_*_SLOT`.
//
//   slot 0  src_offset           (block-payload-relative byte offset)
//   slot 1  comp_size            (bytes in the compressed stream)
//   slot 2  decomp_size          (bytes the chunk will produce)
//   slot 3  dst_offset           (byte offset into the output buffer)
//   slot 4  flags                (CHUNK_FLAG_UNCOMPRESSED | _MEMSET)
//   slot 5  memset_fill (low 8b) + pad (upper 24b must be zero)
//
// Consumers: walk_frame.comp (writer), lz_decode.comp (binding 7
// reader — Cluster A wiring).

pub const CHUNK_DESC_U32_COUNT: u32 = 6;
pub const CHUNK_SRC_OFFSET_SLOT: u32 = 0;
pub const CHUNK_COMP_SIZE_SLOT: u32 = 1;
pub const CHUNK_DECOMP_SIZE_SLOT: u32 = 2;
pub const CHUNK_DST_OFFSET_SLOT: u32 = 3;
pub const CHUNK_FLAGS_SLOT: u32 = 4;
pub const CHUNK_MEMSET_FILL_SLOT: u32 = 5;

// SlzChunkDesc::flags bits (mirror decode_pipeline_shared.glsl).
pub const CHUNK_FLAG_UNCOMPRESSED: u32 = 0x1;
pub const CHUNK_FLAG_MEMSET: u32 = 0x2;

// ── Per-context capacity bounds (mirror decode_pipeline_shared.glsl) ─
//
// WALK_MAX_CHUNKS           — max chunks the walk-frame kernel emits
//                              into its ChunkDescs buffer.
// MAX_SUB_CHUNKS_PER_CHUNK  — worst-case sub-chunks any one chunk
//                              contributes (CHUNK_SIZE / LZ_BLOCK_SIZE = 1
//                              at the L1 default, but the wire format
//                              allows up to 4 in general).
// ENTROPY_SCRATCH_SLOT_BYTES — one slot of the per-sub-chunk entropy
//                              scratch (128 KiB). Holds lit / tok / off16-
//                              hi / off16-lo bytes for one global sub-chunk.
// OFF16_HILO_SPLIT_OFFSET    — split point inside an entropy slot
//                              between the off16-hi and off16-lo halves
//                              (== LZ_BLOCK_SIZE = 65536).
//
// GLSL header mirror:
//   const uint WALK_MAX_CHUNKS            = 16384u;
//   const uint MAX_SUB_CHUNKS_PER_CHUNK   = 4u;
//   const uint ENTROPY_SCRATCH_SLOT_BYTES = 131072u;
//   const uint OFF16_HILO_SPLIT_OFFSET    = 65536u;

pub const WALK_MAX_CHUNKS: u32 = 16384;
pub const MAX_SUB_CHUNKS_PER_CHUNK: u32 = 4;
pub const ENTROPY_SCRATCH_SLOT_BYTES: u32 = 2 * LZ_BLOCK_SIZE; // = 131072
pub const OFF16_HILO_SPLIT_OFFSET: u32 = LZ_BLOCK_SIZE; // = 65536

// ── Default sub-chunk stride ────────────────────────────────────────
//
// DEFAULT_SUB_CHUNK_CAP — slot stride the scan / prefix kernels use
//                          when push.sub_chunk_cap == 0. Equal to
//                          LZ_BLOCK_SIZE today.
//
// Note (F065): walk_frame.comp writes `sub_chunk_cap = ENTROPY_SCRATCH_
// SLOT_BYTES` (= 131072) into the meta buffer, while the scan/prefix
// kernels default `sub_chunk_cap = DEFAULT_SUB_CHUNK_CAP` (= 65536).
// These are deliberately different: the scan kernel reads from the meta
// stride (entropy slot stride, 128 KiB) and the prefix kernel uses the
// per-decompressed-byte stride (64 KiB). Both names are retained to keep
// the dichotomy explicit.

pub const DEFAULT_SUB_CHUNK_CAP: u32 = LZ_BLOCK_SIZE; // 65536

// ── Hash table sizing (L1) ──────────────────────────────────────────
//
// HASH_BITS         — log2 of per-chunk hash entries.
// HASH_SIZE_BYTES   — per-chunk hash table size = (1 << HASH_BITS) * 4.
// HASH_VRAM_MAX_BYTES — guard rail: don't allocate more than this many
//                        bytes for the per-chunk hash table across all
//                        chunks. F058: at 4096 chunks × 512 KiB = 2 GiB
//                        which can exceed available VRAM. Encoders that
//                        bump MAX_CHUNKS without revisiting hash sizing
//                        get a clear runtime error rather than a silent
//                        VRAM OOM.

pub const HASH_BITS: u32 = 17;
pub const HASH_SIZE_BYTES: u64 = (@as(u64, 1) << HASH_BITS) * @sizeOf(u32); // 512 KiB
pub const HASH_VRAM_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB

// ── Per-chunk encode stream capacity ────────────────────────────────
//
// CHUNK_STREAM_CAPACITY  — over-provisioning for each of lit/cmd/
//                           off16/length per chunk. Worst-case
//                           bound is 2 * CHUNK_SIZE; +16 keeps the
//                           lane-0 tail RMW from writing past the slice.
// CHUNK_OFF32_CAPACITY   — same bound for the off32 stream.

pub const CHUNK_STREAM_CAPACITY: u32 = (CHUNK_SIZE * 2) + 16;
pub const CHUNK_OFF32_CAPACITY: u32 = (CHUNK_SIZE * 2) + 16;

// ── Encoder bounds (host side only) ─────────────────────────────────
//
// MAX_CHUNKS — hard cap on chunks per encode. Bounds the per-chunk
//               sidecar slices in L1Streams. 4096 × 64 KiB = 256 MiB
//               max input — covers the silesia scale test (200 MB)
//               with headroom. (Previously this also bounded fixed-
//               size stack arrays inside L1Streams; F052 swapped those
//               for allocator-owned slices so this cap is a runtime
//               guard rather than a load-bearing static-size constant.)

pub const MAX_CHUNKS: u32 = 4096;

// ── SC tail / chunk-internal header constants (wire format) ─────────
//
// CHUNK_INTERNAL_HDR_BYTES — 2-byte internal block header + 4-byte
//                             chunk header (CPU wire format).
// SC_TAIL_PER_CHUNK_BYTES  — 8-byte SC tail prefix on each non-first
//                             chunk.
// UNCOMPRESSED_CHUNK_HDR_BYTES — 2-byte internal header for
//                                 uncompressed-fallback chunks.
// UNCOMPRESSED_CHUNK_MARKER     — host sentinel value (GPU wrap).

pub const CHUNK_INTERNAL_HDR_BYTES: u32 = 6;
pub const SC_TAIL_PER_CHUNK_BYTES: u32 = 8;
pub const UNCOMPRESSED_CHUNK_HDR_BYTES: u32 = 2;
pub const UNCOMPRESSED_CHUNK_MARKER: u32 = 0xFFFFFFFF;

// ── Cross-check invariants ──────────────────────────────────────────
//
// These run at compile time; any divergence between fields or between
// host and shader assumptions becomes a compile error.

comptime {
    // CHUNK_SIZE / LZ_BLOCK_SIZE coherence — the encoder's per-chunk
    // hash slice + the decoder's sub-chunk loop are sized for both
    // being equal at L1. If the two ever diverge the shaders need a
    // multi-block-per-sub-chunk pass; flag the change loudly here.
    if (CHUNK_SIZE != LZ_BLOCK_SIZE) @compileError(
        "CHUNK_SIZE != LZ_BLOCK_SIZE: shaders/lz_encode.comp + " ++
            "shaders/lz_decode.comp assume one LZ block per chunk at L1. " ++
            "If you intentionally bumped CHUNK_SIZE, the encoder's anchor " ++
            "and decoder's block-boundary code both need revisiting.",
    );

    // sc_group_size relationship: sub-chunk size = sc_group_size *
    // SLZ_CHUNK_SIZE_BYTES, and that has to equal LZ_BLOCK_SIZE so the
    // CPU wrap path and the GPU walk_frame kernel agree on chunk
    // boundaries. With sc_group_size=0.25 and SLZ_CHUNK_SIZE_BYTES=256K
    // that's 64K — same as LZ_BLOCK_SIZE.
    const computed_sub_chunk: u32 = @intFromFloat(SC_GROUP_SIZE * @as(f32, @floatFromInt(SLZ_CHUNK_SIZE_BYTES)));
    if (computed_sub_chunk != LZ_BLOCK_SIZE) @compileError(
        "SC_GROUP_SIZE * SLZ_CHUNK_SIZE_BYTES != LZ_BLOCK_SIZE — " ++
            "frame-header sc_group_size doesn't match the encoder's " ++
            "LZ block boundary. Update both before bumping either.",
    );

    // OFF16_HILO_SPLIT_OFFSET == LZ_BLOCK_SIZE — the scan kernel relies
    // on this equality to split off16 hi/lo bytes within one entropy
    // scratch slot.
    if (OFF16_HILO_SPLIT_OFFSET != LZ_BLOCK_SIZE) @compileError(
        "OFF16_HILO_SPLIT_OFFSET must equal LZ_BLOCK_SIZE (scan kernel " ++
            "assumes the off16-hi/lo split coincides with the block " ++
            "boundary inside the entropy scratch slot).",
    );

    // ENTROPY_SCRATCH_SLOT_BYTES == 2 * LZ_BLOCK_SIZE — one slot holds
    // both off16-hi (block 0) and off16-lo (block 1) plus lit/tok.
    if (ENTROPY_SCRATCH_SLOT_BYTES != 2 * LZ_BLOCK_SIZE) @compileError(
        "ENTROPY_SCRATCH_SLOT_BYTES must be 2 * LZ_BLOCK_SIZE so one " ++
            "slot fits both off16 halves of a sub-chunk's max-decomp " ++
            "size (= 2 * LZ_BLOCK_SIZE in the worst case).",
    );

    // off32 stride must equal the other stream stride — the encoder's
    // `storeOff32Byte` derives its global offset from g_dst_word_base,
    // which is set from CHUNK_STREAM_CAPACITY. If the two diverge the
    // encoder writes off32 bytes into the wrong chunk slice.
    if (CHUNK_OFF32_CAPACITY != CHUNK_STREAM_CAPACITY) @compileError(
        "CHUNK_OFF32_CAPACITY must equal CHUNK_STREAM_CAPACITY — see " ++
            "storeOff32Byte in shaders/lz_encode.comp (g_dst_word_base " ++
            "is derived from chunk_capacity and shared between off32 + " ++
            "the other streams).",
    );

    // Stream capacities must be word-aligned (the host invariant the
    // shader's lane-0 RMW relies on for byte-base alignment).
    if (CHUNK_STREAM_CAPACITY % 4 != 0) @compileError(
        "CHUNK_STREAM_CAPACITY must be a multiple of 4 (per-chunk " ++
            "byte base must land on a u32 word boundary).",
    );

    // OFF32_COUNT_PACK_MAX width must match its declared bit count
    // (4095 == (1<<12)-1).
    if (OFF32_COUNT_PACK_MAX != ((@as(u32, 1) << OFF32_COUNT_FIELD_BITS) - 1)) @compileError(
        "OFF32_COUNT_PACK_MAX must equal (1<<OFF32_COUNT_FIELD_BITS)-1.",
    );

    // F014 note: the 4-byte extended off32 form is now wired into
    // shaders/lz_encode.comp::writeOffset32. At L1 default geometry
    // (CHUNK_SIZE + LZ_BLOCK_SIZE = 0x20000) every emit still hits the
    // 3-byte short form (LARGE_OFFSET_THRESHOLD = 0xC00000). If the
    // geometry is bumped past the threshold the long form takes over
    // automatically; the decoder's loadOff32EntryFull reader handles
    // both forms (Cluster C / F014 in src_vulkan/shaders/lz_decode.comp).
    //
    // The assertion below remains useful as documentation that the L1
    // tested path stays in the short form — if it ever fires, the L1
    // decode path's loadOff32Entry24 reader (which assumes uniform
    // 3-byte entries indexed by entry_idx) needs to switch to a
    // byte-cursor walk over loadOff32EntryFull. Today nothing on the
    // L1 hot path consumes the extended form.

    // FAR_OFFSET_MIN_MATCH > 0 (any positive value works — we just want
    // a sanity bound that catches a typo zeroing out the constant).
    if (FAR_OFFSET_MIN_MATCH == 0) @compileError("FAR_OFFSET_MIN_MATCH must be > 0.");
}
