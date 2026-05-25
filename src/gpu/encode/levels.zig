//! Per-user-level GPU encoder knobs (hash bits, parser flavor).
//!
//! Kept in their own file so the encode kernels can pull the policy in
//! without dragging the rest of the driver. Values mirror CPU
//! `fast_framed.zig` engine-level caps so GPU L1..L5 produce hash
//! distributions equivalent to the CPU encoder modulo warp-parallel scan.

/// Hash bits per level. Mirrors CPU `fast_framed.zig:955-963` engine-level
/// cap so GPU L1/L2/L3/L4/L5 produce hash distributions equivalent to CPU
/// L1/L2/L3/L4/L5 (modulo warp-parallel scan).
///
/// User L1 -> engine -2 -> cap 17
/// User L2 -> engine -1 -> cap 18
/// User L3 -> engine  1 -> cap 19
/// User L4 -> engine  2 -> cap 20
/// User L5 -> engine  4 -> 20 (no cap from engine; matches CPU practical)
pub fn hashBitsForLevel(level: u8) u32 {
    return switch (level) {
        1 => 17,
        2 => 18,
        3 => 19,
        4 => 20,
        5 => 20,
        else => 11,
    };
}

pub fn useGlobalHash(level: u8) bool {
    // All levels use global hash. L1's larger hash table (16-bit) needs
    // more than CUDA shared-mem allows; using global also dodges the
    // shared-mem-hash corruption bug seen at L2 sc>=0.5.
    _ = level;
    return true;
}

pub fn useChainParser(level: u8) bool {
    return level >= 5;
}
