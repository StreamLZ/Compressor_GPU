//! Per-user-level GPU encoder knobs (hash bits, parser flavor).
//!
//! Kept in their own file so the encode kernels can pull the policy in
//! without dragging the rest of the driver.

/// Hash bits per user level — 17 everywhere since 2026-06-09 (see
/// below). The level ladder is differentiated elsewhere: L2 and L4+
/// enable the greedy parser's match-range rehash (encode_lz.zig
/// `p_l4`; v4 #6, 2026-06-10 — re-differentiated L2 from L1 after
/// hb=17-everywhere collapsed them), L3+ adds the Huffman entropy
/// stage (fast_framed.zig), L5 swaps in the chain parser
/// (`useChainParser`). Levels outside `1..5` are rejected by the
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
    // 2026-06-09: L2 18→17 and L3 19→17, completing the hb=17 cap at
    // every level. With the sc=0.25 default (15,259 chunks at 1 GB),
    // L2's hb=18 hash was 16 GB — past physical VRAM on consumer
    // hardware, collapsing encode to 159 MB/s via WDDM paging. L3's
    // hb=19 was worse (silently broken + 10× slow at 1 GB before the
    // A-024 fix). Measured ratio cost of 17 vs the old values:
    // ≈ 0.0-0.1 pp on enwik8/enwik9 — sub-chunks are ≤ 128 KB, so hash
    // tables beyond 2^17 entries add collisions-free slots the 64 KB
    // match window can never exploit.
    return switch (level) {
        1 => 17,
        2 => 17,
        3 => 17,
        4 => 17,
        5 => 17,
        else => unreachable,
    };
}

pub fn useChainParser(level: u8) bool {
    return level >= 5;
}
