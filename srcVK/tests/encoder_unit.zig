//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Unit tests for srcVK/encode/streamlz_encoder.zig: compressBound math,
//! Options validation, writeUncompressedFrame path for tiny inputs. No
//! GPU. Test bodies added by the fleshout agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
const encoder = @import("../encode/streamlz_encoder.zig");
