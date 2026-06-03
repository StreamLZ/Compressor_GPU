//! Cross-cutting error set for the Vulkan port.
//!
//! Holds shared error variants that need to be reachable from every L2
//! stub file. The codec-side `descriptors.GpuError` (under
//! src_vk/decode/descriptors.zig) extends this set with the
//! decode-pipeline-specific failure modes; L2 stub files that live
//! outside `decode/` (huff_build_lut, huff_decode_4stream,
//! merge_huff_descs, lz_decode_general, ...) re-use `NotImplementedL2`
//! directly from here.
//!
//! VK PORT NOTE: This error lives in a top-level module rather than
//! inside `decode/vulkan_api.zig` so the encoder-side L2 stubs (which
//! never @import the decode tree) can still reach it.

/// Returned by every L2+ kernel/dispatcher whose body has not yet been
/// ported from CUDA. L1 control flow MUST never hit one of these
/// returns; the L2 gate in `decode/decode_dispatch.zig`
/// (`if (level >= 2) { ... }`) is the canonical place that funnels L2
/// work past these stubs on L1 frames.
pub const NotImplementedL2 = error{NotImplementedL2};

/// Convenience: the bare error value, ready to `return error.NotImplementedL2;`
/// from any stub body without re-importing this module.
pub const not_implemented_l2: NotImplementedL2 = error.NotImplementedL2;
