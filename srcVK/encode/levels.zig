//! 1:1 port of src/encode/levels.zig.
//!
//! Level-to-parser-shape mappings. Drives hash-table sizing in the
//! greedy parser and the L5 chain-parser opt-in. Pure host code.

/// CUDA reference: src/encode/levels.zig (updated 2026-06-09): hb=17
/// at every level (L2 was 18, L3 was 19). At sub-chunk sizes ≤ 128 KB
/// hash tables beyond 2^17 entries add nothing the 64 KB match window
/// can exploit, and the larger tables blow VRAM at 1 GB × sc=0.25
/// chunk counts. Ported in step to keep encode output byte-identical.
pub fn hashBitsForLevel(level: u8) u32 {
    return switch (level) {
        1 => 17,
        2 => 17,
        3 => 17,
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
