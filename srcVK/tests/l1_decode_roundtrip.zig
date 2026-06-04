//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Golden L1 frames (encoded via CUDA reference or a fresh VK encode)
//! decoded via VK port; byte-compare against original. Test bodies
//! added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const decoder = @import("../decode/streamlz_decoder.zig");
