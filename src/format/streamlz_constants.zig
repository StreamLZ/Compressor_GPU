//! Shared StreamLZ wire-format constants used by both the encode and
//! decode sides on the host. The matching device-side constants live in
//! `src/common/gpu_wire_format.cuh` — the two files must agree byte for
//! byte (the `static_assert`s on the CUDA side catch drift).
//!
//! Terminology: "sc" / "SC" = "self-contained" throughout this module.

// ── Chunk sizing and header bit layout ─────────────────────────────────
pub const chunk_size: usize = 0x40000; // 256 KB
pub const chunk_size_bits: u6 = 18;
pub const chunk_size_mask: u32 = chunk_size - 1; // 0x3FFFF

/// 4-byte chunk-header type field. Bits 18-19. Type 0 = LZ-compressed,
/// type 1 = memset (single fill byte follows the 4-byte header).
pub const chunk_type_shift: u6 = chunk_size_bits;
pub const chunk_type_memset: u32 = 1 << chunk_type_shift;

/// Sub-chunk size: 128 KB. Each outer 256 KB chunk contains up to two
/// sub-chunks at sc_group_size = 1.0; the GPU encoder picks 0.25 or 0.5
/// so it never emits more than one sub-chunk per chunk.
pub const sub_chunk_size: usize = 0x20000;

// ── Decoder slack ──────────────────────────────────────────────────────

/// Extra bytes the decoder is allowed to write past `dst_len`. Several
/// copy helpers prefetch and store ahead by up to this many bytes; the
/// caller's destination buffer must include this trailing slack.
pub const safe_space: usize = 64;

// ── Frame-header default ───────────────────────────────────────────────

/// Default value the frame writer stamps into the `sc_group_size` slot
/// when the encoder did not pick a non-default value. Typed as `f32` to
/// match the v2 wire format; the current GPU encoder always overrides
/// this via `resolveScGroupSize`, so the constant only flows through the
/// canonical-header smoke tests.
pub const default_sc_group_size: f32 = 4.0;

// ── Match-window limits ────────────────────────────────────────────────

/// Maximum back-reference distance any LZ encoder is allowed to consult
/// (1 GB). The GPU encoder does not consult this directly — it sizes
/// its hash tables via `hashBitsForLevel` — but the constant still
/// flows through the public `compressBound` upper bound.
pub const max_dictionary_size: usize = 0x40000000;
