//! NEW per Exception 3 (no CUDA counterpart).
//!
//! Smoke test the VK binary: `streamlz_vk -c file -o out.slz` then
//! `streamlz_vk -d out.slz -o roundtrip` and compare to original.
//! Exercises every CLI mode at L1. Test bodies added by the fleshout
//! agent.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

const std = @import("std");
