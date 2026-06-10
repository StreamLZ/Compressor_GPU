//! SPIR-V blob registry for the srcVK/ build tree.
//!
//! `zig build srcvk-shaders` compiles every .comp under srcVK/ into a
//! sibling .spv blob and bundles all of them into a single generated
//! directory (see addSrcVkShaderSteps in build.zig). The build then
//! creates an anonymous Zig module rooted in that directory and adds it
//! as `@import("spv_blobs")` to the srcVK executable module, after which
//! the bare-filename `@embedFile()` calls below resolve at compile time.
//!
//! Exposed surface is one slice per .spv blob. module_loader.zig calls
//! `vkCreateShaderModule` on each slice in turn. CUDA equivalent: each
//! kernel name is fetched from the loaded .ptx via cuModuleGetFunction.

const std = @import("std");

// ── decode (13) ─────────────────────────────────────────────────────
pub const lz_decode_raw: []const u8 = @embedFile("lz_decode_raw_kernel.spv");
pub const lz_decode: []const u8 = @embedFile("lz_decode_kernel.spv");
pub const prefix_sum_chunks: []const u8 = @embedFile("prefix_sum_chunks_kernel.spv");
pub const gather_raw_off16: []const u8 = @embedFile("gather_raw_off16_kernel.spv");
pub const walk_frame: []const u8 = @embedFile("walk_frame_kernel.spv");
pub const compact_huff_descs: []const u8 = @embedFile("compact_huff_descs_kernel.spv");
pub const compact_raw_descs: []const u8 = @embedFile("compact_raw_descs_kernel.spv");
pub const compact_all_descs: []const u8 = @embedFile("compact_all_descs_kernel.spv");
pub const merge_huff_descs: []const u8 = @embedFile("merge_huff_descs_kernel.spv");
pub const merge_huff_descs_par: []const u8 = @embedFile("merge_huff_descs_par_kernel.spv");
pub const scan_parse: []const u8 = @embedFile("scan_parse_kernel.spv");
pub const huff_build_lut: []const u8 = @embedFile("huff_build_lut_kernel.spv");
pub const huff_decode_4stream: []const u8 = @embedFile("huff_decode_4stream_kernel.spv");

// ── encode (6) ──────────────────────────────────────────────────────
pub const lz_encode: []const u8 = @embedFile("lz_encode_kernel.spv");
pub const huff_build_tables: []const u8 = @embedFile("huff_build_tables_kernel.spv");
pub const huff_encode_4stream: []const u8 = @embedFile("huff_encode_4stream_kernel.spv");
pub const assemble_measure: []const u8 = @embedFile("assemble_measure_kernel.spv");
pub const assemble_write: []const u8 = @embedFile("assemble_write_kernel.spv");
pub const frame_assemble: []const u8 = @embedFile("frame_assemble_kernel.spv");
