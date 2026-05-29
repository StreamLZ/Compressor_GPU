//! Per-user-level GPU encoder knobs (hash bits, parser flavor).
//!
//! Kept in their own file so the encode kernels can pull the policy in
//! without dragging the rest of the driver. Values mirror CPU
//! `fast_framed.zig` engine-level caps so GPU L1..L5 produce hash
//! distributions equivalent to the CPU encoder modulo warp-parallel scan.

/// Hash bits per level. Mirrors the `engine_level_cap` switch inside
/// `fast_framed.compressFramedOne` so GPU L1/L2/L3/L4/L5 produce hash
/// distributions equivalent to CPU L1/L2/L3/L4/L5 (modulo warp-parallel
/// scan).
///
/// User L1 -> engine -2 -> cap 17
/// User L2 -> engine -1 -> cap 18
/// User L3 -> engine  1 -> cap 19
/// User L4 -> engine  2 -> cap 20 (was; now 17 — see below)
/// User L5 -> engine  4 -> 20 (was; now 17 — see below)
///
/// L4/L5 cap: 17 (was 20). At hb=20 the chain parser's per-chunk hash is
/// ~8 MB; at sc=0.25 for a 200 MB input this allocates ~26 GB and spills
/// to system RAM via PCIe on consumer GPUs (16 GB VRAM), giving a ~10×
/// per-byte slowdown vs the in-VRAM case. hb=17 shrinks per-chunk hash
/// to ~1 MB which fits comfortably regardless of input size. Validated
/// 2026-05-27 in tools/encode_l5_silesia/: silesia GPU encode 4419 ms
/// → 415 ms at +0.07% pre-Huffman ratio (≈0 after Huffman). enwik8 also
/// gets a smaller win from better L1/L2 cache hit rate.
pub fn hashBitsForLevel(level: u8) u32 {
    return switch (level) {
        1 => 17,
        2 => 18,
        3 => 19,
        4 => 17,
        5 => 17,
        else => 11,
    };
}

pub fn useChainParser(level: u8) bool {
    return level >= 5;
}
