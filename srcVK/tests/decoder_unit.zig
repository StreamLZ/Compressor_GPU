//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Unit tests for srcVK/decode/streamlz_decoder.zig: parseHeader /
//! parseBlockHeader / parseChunkHeader edge cases; pure host-side, no
//! GPU. Test bodies added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const decoder = @import("../decode/streamlz_decoder.zig");
const frame = @import("../format/frame_format.zig");
const block_header = @import("../format/block_header.zig");
