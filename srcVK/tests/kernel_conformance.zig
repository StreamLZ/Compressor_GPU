//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Kernel conformance: feed known sub-chunk inputs into the L1 hot
//! kernels (slzLzDecodeRawKernel, slzPrefixSumChunksKernel,
//! slzLzEncodeKernel, slzAssembleMeasureKernel, slzAssembleWriteKernel,
//! slzFrameAssembleKernel) via VK; assert byte-identical outputs vs
//! CUDA goldens stored in tests/goldens/. Test bodies added by the
//! fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
