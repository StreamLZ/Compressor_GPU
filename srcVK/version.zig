//! 1:1 port of src/version.zig.
//!
//! Library version exposed to both the C ABI (slzVersionString) and the
//! CLI (--version). One source of truth so the two surfaces cannot
//! drift.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

pub const string: [:0]const u8 = "3.0.0";
