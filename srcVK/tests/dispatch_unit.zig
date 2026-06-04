//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Unit tests for srcVK/decode/decode_dispatch.zig: L2 gate behaviour
//! (level=1 skips Huff/scan/compact/merge/gather), runLzPipeline raw-
//! kernel selection, buildChunkDescriptors output. Test bodies added by
//! the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const decode_dispatch = @import("../decode/decode_dispatch.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
