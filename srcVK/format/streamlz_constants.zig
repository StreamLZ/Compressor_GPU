//! 1:1 port of src/format/streamlz_constants.zig.
//!
//! Shared StreamLZ wire-format constants used by both encode and decode
//! on the host. The device-side constants live in
//! srcVK/common/gpu_wire_format.glsl — the two files must agree byte for
//! byte.
//!

// ── Chunk sizing and header bit layout ─────────────────────────────────
pub const chunk_size: usize = 0x40000; // 256 KB
pub const chunk_size_bits: u6 = 18;
pub const chunk_size_mask: u32 = chunk_size - 1; // 0x3FFFF

pub const chunk_type_shift: u6 = chunk_size_bits;
pub const chunk_type_memset: u32 = 1 << chunk_type_shift;

pub const sub_chunk_size: usize = 0x20000;

pub const safe_space: usize = 64;

pub const default_sc_group_size: f32 = 4.0;

pub const max_dictionary_size: usize = 0x40000000;
