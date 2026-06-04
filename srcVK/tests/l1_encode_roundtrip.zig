//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Random + structured payloads encoded via VK port, decoded via VK
//! port, byte-compare against original. Levels 1-2 exercised here.
//! Test bodies added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const encoder = @import("../encode/streamlz_encoder.zig");
const decoder = @import("../decode/streamlz_decoder.zig");
