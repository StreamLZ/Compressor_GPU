//! 1:1 port of src/encode/levels.zig.
//!
//! Level-to-parser-shape mappings. Drives hash-table sizing in the
//! greedy parser and the L5 chain-parser opt-in. Pure host code.

/// CUDA reference: src/encode/levels.zig:21-30. Hash-bit width per level.
/// L1-L3 grow the hash table; L4/L5 cap at 17 to keep the chain parser's
/// per-chunk hash in VRAM.
pub fn hashBitsForLevel(level: u8) u32 {
    return switch (level) {
        1 => 17,
        2 => 18,
        3 => 19,
        4 => 17,
        5 => 17,
        else => unreachable,
    };
}

/// CUDA reference: src/encode/levels.zig:32-34. True at L5; the chain
/// parser is only used at L5.
pub fn useChainParser(level: u8) bool {
    return level >= 5;
}
