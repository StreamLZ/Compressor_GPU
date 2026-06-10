//! Cross-cutting error set for the Vulkan port (NEW; no CUDA counterpart).
//!
//! Holds shared error variants that every L1 and L2 stub file can reach
//! without pulling in the decode/encode trees. The codec-side
//! `descriptors.GpuError` (srcVK/decode/descriptors.zig) extends this
//! set with decode-pipeline-specific failure modes; stubs outside the
//! decode/encode trees re-use `NotImplementedL2` and `NotYetPorted`
//! directly from this module.
//!

/// Returned by L2+ entry points whose bodies have not yet been ported
/// from CUDA. The L2 gate in srcVK/decode/decode_dispatch.zig and the
/// `opts.level >= 3` gate in srcVK/encode/fast_framed.zig funnel L1
/// control flow past these stubs.
pub const NotImplementedL2 = error{NotImplementedL2};

/// Returned by L1-scope entry points whose bodies are still placeholder
/// stubs in this skeleton pass. Every function returning this error is
/// a Step-3+ fleshout target.
pub const NotYetPorted = error{NotYetPorted};
