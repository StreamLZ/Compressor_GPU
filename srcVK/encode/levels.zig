//! 1:1 port of src/encode/levels.zig.
//!
//! Level-to-parser-shape mappings. Drives hash-table sizing in the
//! greedy parser and the L5 chain-parser opt-in. Pure host code.
//!
//! See srcVK/PortInstructions.md for the fleshout checklist for this file.

/// CUDA reference: src/encode/levels.zig:21-31. Hash-bit width per level.
pub fn hashBitsForLevel(level: u8) u32 {
    _ = level;
    return 0;
}

/// CUDA reference: src/encode/levels.zig:32-end. L5 only — true when
/// the chain parser should be used.
pub fn useChainParser(level: u8) bool {
    _ = level;
    return false;
}
