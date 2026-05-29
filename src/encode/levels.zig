//! Per-user-level GPU encoder knobs (hash bits, parser flavor).
//!
//! Kept in their own file so the encode kernels can pull the policy in
//! without dragging the rest of the driver.

/// Hash bits per user level. L1-L3 grow the hash table; L4/L5 cap at
/// 17 to keep the chain parser's per-chunk hash in VRAM (see the cap
/// rationale below). Levels outside `1..5` are rejected by the
/// encoder front door (`streamlz_encoder.compressFramedWithIo` and
/// `streamlz_gpu.compressCore`), so the `else => unreachable` here
/// is genuinely unreachable.
///
/// L4/L5 cap: 17 (was 20). At hb=20 the chain parser's per-chunk hash
/// is ~8 MB; at sc=0.25 for a 200 MB input this allocates ~26 GB and
/// spills to system RAM via PCIe on consumer GPUs (16 GB VRAM), giving
/// a ~10× per-byte slowdown vs the in-VRAM case. hb=17 shrinks
/// per-chunk hash to ~1 MB which fits comfortably regardless of input
/// size. Validated in `tools/encode_l5_silesia/`: silesia GPU encode
/// 4419 ms → 415 ms at +0.07% pre-Huffman ratio (≈0 after Huffman);
/// enwik8 also gets a smaller win from better L1/L2 cache hit rate.
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

pub fn useChainParser(level: u8) bool {
    return level >= 5;
}
