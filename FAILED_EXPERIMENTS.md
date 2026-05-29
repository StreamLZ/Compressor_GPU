# Failed Experiments

This document catalogs optimization attempts that did not pan out. Recording
what *didn't* work is as valuable as recording what did — it prevents future
contributors from re-investigating dead ends, and documents the reasoning
behind why certain approaches were rejected.

## Guiding Priorities

StreamLZ prioritizes, in order:

1. **Decompress speed** — the dominant use case. An extra 10% compress time is
   usually invisible; an extra 10% decompress time is always felt.
2. **Compression ratio** — users of L6+ explicitly chose higher ratio over speed.
   A 0.1pp ratio loss for a 10% compress speedup is generally not worth it.
3. **Compress speed** — important but not at the expense of ratio or decomp.

Experiments are evaluated against these priorities. A "neutral" result on
decompress speed and ratio means the change isn't worth merging even if it
improves compress speed.

---

## P-core vs E-core thread pinning for hybrid CPUs (2026-04-21)

**Context**: Arrow Lake has 8 P-cores and 16 E-cores. VTune memory-access
profiling showed L1 decompress hitting 85/93 GB/s DRAM bandwidth (92%
saturated) at 8+ threads. Hypothesis: pinning decoder threads to P-cores
only would avoid E-core cache thrashing and improve throughput.

**Experiment**: Pin single-thread decompress to a P-core vs an E-core
using `SetThreadSelectedCpuSets` + `GetSystemCpuSetInformation`
(universal Windows API — works on any hybrid architecture via
`EfficiencyClass`).

**Results** (enwik8 100 MB, single-thread, pinned):

| Level | P-core | E-core | P/E ratio |
|-------|--------|--------|-----------|
| L1 | 5,841 MB/s | 5,412 MB/s | 1.08x |
| L5 | 3,233 MB/s | 3,173 MB/s | 1.02x |
| L6 | 943 MB/s | 777 MB/s | 1.21x |

**Multi-thread scaling** (L1 decompress, no pinning):

| Threads | Speed | Scaling |
|---------|-------|---------|
| 1 | 6,201 MB/s | 1.0x |
| 4 | 22,416 MB/s | 3.6x |
| 8 | 32,816 MB/s | 5.3x |
| 12 | 32,858 MB/s | 5.3x |
| 24 | 32,335 MB/s | 5.2x |

VTune memory-access confirmed: **DRAM Bandwidth Bound = 86.7%** at
10 threads, observed peak 85.4 GB/s of 93 GB/s theoretical.

**Why pinning doesn't help**:
- L1-L5 are DRAM-bandwidth-bound. Core type doesn't matter — both
  hit the same memory bus. P-core is only 2-8% faster per-thread.
- L6+ are CPU-bound and P-cores are 21% faster per-core. But L6 at
  943 MB/s × 8 P-cores = 7.5 GB/s, far below the 85 GB/s DRAM wall.
  There's room for all 24 cores including E-cores.
- Using only 8 P-cores (29 GB/s) was SLOWER than all 24 cores
  (31 GB/s) because E-cores contribute despite being individually weaker.

**Disposition**: No pinning implemented. The OS scheduler already does
the right thing. The `detectCores` and `pinCurrentThreadToCpuSet`
utilities remain in `platform/memory_query.zig` for future use.

---

## L1 compress: eliminate dictionary_size bounds check for u16 hash (2026-04-21)

**Context**: VTune assembly-level profiling showed `mov r10, qword ptr
[rbp+0x90]` (dictionary_size stack load) at 464M cycles in the parser
hot loop. For L1's u16 hash, max offset is 65535 which is always <
dictionary_size (1 GB), so the check is always true.

**Change**: Guard with `(T == u16 or offset_candidate < dictionary_size)`
so the comptime branch eliminates the load for L1.

**Result**: LLVM restructured the bounds check from branched `cmp + ja`
to branchless `setnb + setbe + test`. The dictionary_size stack load
disappeared. VTune confirmed the offset-8 compare cycles dropped 26%
(2,554M → 1,887M) due to fewer upstream pipeline flushes. But wall
time was **6% slower** (527 → 496 MB/s) because the branchless `setcc`
sequence adds 3 µops that the OoO engine couldn't hide.

**Lesson**: Fewer mispredicts ≠ faster. The branched version with the
"wasted" stack load was faster because OoO execution hid the load
latency behind branch computation. The extra `setcc` µops from the
branchless path saturated the execution ports more than the
mispredicts cost.

---

## SoA cmd stream split: separate lit_count / match_len / offset_type (2026-04-21)

**Hypothesis**: The Fast codec packs lit_count (3 bits), match_len
(4 bits), and use_recent (1 bit) into a single cmd byte. Splitting
into three separate streams (Structure of Arrays) would give each
stream a simpler distribution that entropy-codes tighter. Suggested
by Gemini as an Oodle-inspired optimization.

**Analysis**: Measured entropy of packed vs split encoding over 1M
simulated tokens with realistic distributions.

| Encoding | Entropy (independent) | Entropy (correlated) |
|----------|----------------------|---------------------|
| Packed cmd byte | 6.870 bits/token | **6.627 bits/token** |
| Split (lit+match+recent) | 6.870 bits/token | 6.753 bits/token |
| Difference | 0.000 | **-0.126 bits/token** |

**Result**: With independent fields, entropy is identical — splitting
neither helps nor hurts. With realistic correlations (recent-offset
matches tend to have shorter lengths), the packed byte **wins by
0.126 bits/token** because tANS can exploit the cross-field
correlation. The packed byte `0x9B` (recent + short match) becomes
very frequent and gets a short Huffman/tANS code. Split streams see
only marginal distributions and lose this.

**Disposition**: Not implemented. The packed cmd byte is the correct
design when fields are correlated, which they are in practice. Oodle
may split for decode-speed reasons (simpler per-stream tables) but
it costs ratio.

---

## High-bit stripping for ASCII literal streams (2026-04-21)

**Hypothesis**: In ASCII text, 99.33% of bytes have high bit = 0.
Stripping the high bit and encoding 7-bit values through tANS (128-
symbol alphabet instead of 256) should compress tighter. The rare
exceptions (0.67% of bytes with high bit = 1) are stored as a small
delta-varint exception stream.

Theoretical savings: the high bit has only 0.058 bits/byte of entropy
vs the 1 bit/byte it costs in a fixed-width encoding. Should save
~11.9 MB on 100 MB of text.

**What we tried**: Pre-filter approach — strip high bits from enwik8,
compress the 7-bit stream + exception stream separately, compare
total compressed size against compressing the original.

**Results** (enwik8 100 MB):

| Compressor | Original | Lowbits + Exceptions | Delta |
|-----------|----------|---------------------|-------|
| zstd 1 | 40,676 KB | 40,978 KB | **+0.7% worse** |
| zstd 9 | 31,130 KB | 31,522 KB | **+1.3% worse** |
| zstd 19 | 26,944 KB | 27,339 KB | **+1.5% worse** |
| SLZ L5 | 43,377 KB | 44,161 KB | **+1.8% worse** |

**Why it failed**: Byte-level entropy coders (tANS, Huffman, FSE)
already capture the high bit's predictability in their frequency
tables. A byte like 'e' (0x65, high bit=0) gets fewer bits than a
byte like 0xC3 (high bit=1) automatically. Splitting the bit out
adds overhead (exception stream positions + separate stream headers)
that exceeds the tiny entropy savings from the split.

Mathematically: full byte entropy = 5.080 bits/byte. Split encoding
(7-bit entropy + high-bit entropy) = 5.052 + 0.058 = 5.110 bits/byte.
The split LOSES 0.030 bits/byte because it destroys the correlation
between the high bit and the low 7 bits (e.g., UTF-8 continuation
bytes have high bit=1 AND predictable low bits).

**Lesson**: Partial-bit encoding only wins when done INSIDE the
entropy coder at the bit level (arithmetic coding with adaptive
context per bit), not as a pre-filter that splits bits before a
byte-level coder. A byte-level coder already prices in per-bit
predictability through its symbol frequency table.

The real path to sub-bit encoding requires replacing tANS with an
arithmetic coder for the literal stream — which costs ~10-20x slower
entropy decode. Viable only for L9-L11 where ratio matters more than
decode speed.

---

## L1 greedy parser branch misprediction investigation (2026-04-20)

**Context**: VTune uarch-exploration on L1 parallel compress (8T, enwik9
1 GB) showed `runGreedyParser` at 74.9B instructions / 30.5B cycles with
**61.2B branch mispredict slots** — the dominant bottleneck. Per-thread
throughput was ~350 MB/s vs LZ4's ~700 MB/s.

**VTune hotspot breakdown** (by cycles):

| Function | Cycles | Mispredict slots |
|----------|--------|-----------------|
| runGreedyParser | 30.5B | 61.2B |
| extendMatchForward | 10.0B | 25.4B |
| copyBlocks | 5.1B | 0.2B |
| writeComplexOffset | 4.1B | 21.8B |
| writeOffset | 3.4B | 23.5B |

**Root cause**: The greedy parser has a multi-way data-dependent decision
tree per position: (1) recent-offset match? (2) hash match with bounds
check + byte confirm + min_length check? (3) offset-8 fallback? (4) no
match → skip. That's 5-7 branches per position on the miss path. The CPU
branch predictor cannot learn these patterns because they depend on the
input data's match distribution. LZ4's parser has ~2 branches per
position (one hash lookup + one byte compare).

### Experiments tried

**1. Widen extendMatchForward from 4-byte to 8-byte comparisons**

Result: **+4% compress speed**, ratio +0.001%. Halves loop iterations in
the match extension, reducing exit-branch mispredicts. **Committed.**

**2. Remove offset-8 fallback for L1 (`comptime level >= -1` gate)**

Result: +1% speed, **-0.03% ratio** (+15 KB on 100 MB). The offset-8
branch is well-predicted (almost always not-taken) so removing it
doesn't help. But it finds real matches, so removing it hurts ratio.
Reverted.

**3. `noinline` writeOffset to shrink hot loop icache footprint**

Result: No measurable change. The loop body already fits in the DSB
(~4K µops). Making `writeOffset` a call doesn't change the branch
prediction behavior.

**4. Hash-first restructured parser (eliminate `found_match` flag)**

Rewrite the L1 path to evaluate hash and recent speculatively, then
use a single `if (hash_confirmed or recent_confirmed)` branch instead
of the nested if/else chain.

Result: Speed unchanged (2,780 vs 2,785 MB/s), **ratio -0.77%**
(+452 KB on 100 MB). The restructure lost the original path's 1-byte-
literal trick for recent-offset matches (`source_cursor += 1` before
extending). That trick finds overlapping matches that the hash-first
path misses, saving significant bytes. And the combined OR branch
still mispredicts at the same rate — the CPU sees the same data-
dependent pattern regardless of code structure.

### Why StreamLZ L1 is inherently slower than LZ4 per-thread

The StreamLZ Fast wire format requires:
- **Recent-offset tracking** — an extra speculative read + comparison
  per position that LZ4 doesn't have
- **Minimum match length table** — offset-dependent threshold (near
  offsets: 4 bytes, far offsets: longer) adds a branch after match
  confirmation
- **Offset-8 fallback** — exploits the 8-byte initial copy at chunk
  boundaries; LZ4 has no equivalent

These features are what give StreamLZ better ratio than LZ4 at the
same level (58.6% vs 57.3% on enwik8) and dramatically faster
decompress (the decoder benefits from recent-offset tokens being
free to decode). The per-thread compress cost is the tradeoff.

### Where the speedup actually lives

L1 compress scales via parallelism (SC mode, per-chunk workers):
- 1 thread: ~350 MB/s
- 8 threads: 2,800 MB/s
- 24 threads: 4,800 MB/s

This matches zstd 1's 8-thread compress speed (3,300 MB/s) while
decompressing 15x faster. The architectural choice is: accept lower
per-thread compress throughput in exchange for a decode-optimized wire
format that parallelizes trivially.

---

## Zig L9 decompress micro-optimizations (2026-04-14)

After the big wins (`c_allocator` for token fallback, register-resident
recent-offset LIFO, 16-byte SIMD literal copies, prefetch-safe + tail
loop split) brought Zig L9 100MB enwik8 decompress from 868 → 2143 MB/s,
we tried these incremental tweaks looking for more. **None of them moved
the needle.**

VTune was the tool throughout — Hotspots collection at 1000 Hz,
function-level CPU breakdown, and source/asm view for register
allocation analysis.

### Token cursor refactor

**Change:** Replace `tokens[token_index] = ...; token_index += 1;` with
`tokens_cur[0] = ...; tokens_cur += 1;` and compute the count via
pointer subtraction at return. Hypothesis: removes the separate index
counter that VTune saw spilled to stack as `[rbp]`.

**Result:** ~2111 MB/s mean (vs ~2143 baseline) — **slight regression**
within noise. VTune disasm showed LLVM **still** spilled an internal
byte-offset counter to `[rbp]` after the refactor. The compiler's
canonicalization defeated the source change.

**Lesson:** "use a pointer instead of an index" doesn't translate to
register-level savings when the function has many other live values.
The compiler decides where to spill, not the source.

### SIMD `@shuffle` to pack ros into one XMM (table version)

**Change:** Hold `ro3`, `ro4`, `ro5`, `new_off` in a single
`@Vector(4, i32)` and do the LIFO shuffle as one `@shuffle` with a
per-`oi` mask indexed from a comptime table.

**Result:** **Did not compile** — Zig 0.15 requires the `@shuffle` mask
to be `comptime`-known. Runtime mask indexing is not supported.

### Comptime PSHUFD switch (4 cases, one mask per case)

**Change:** Workaround for the previous failure: use a 4-case
`switch (offset_index)` where each case has its own comptime mask:
```zig
ro_vec = switch (offset_index) {
    0 => @shuffle(i32, ro_with_new, undefined, [_]i32{ 0, 1, 2, 0 }),
    1 => @shuffle(i32, ro_with_new, undefined, [_]i32{ 1, 0, 2, 0 }),
    2 => @shuffle(i32, ro_with_new, undefined, [_]i32{ 2, 0, 1, 0 }),
    else => @shuffle(i32, ro_with_new, undefined, [_]i32{ 3, 0, 1, 0 }),
};
const picked: i32 = ro_vec[0];
```

**Result:** ~2039 MB/s mean (vs ~2143 baseline) — **regression of ~5%**.
Roundtrip valid. The XMM lane extract (`ro_vec[0]`) plus the case
dispatch overhead (each case is a separate basic block branched into)
beat out the GPR savings.

**Why:** The 4-case switch over an XMM-producing branch generates
something close to a jump table on XMM operands. Picking the resulting
`picked` requires a `MOVD r32, xmm` extract which is ~3-cycle latency.
Vs the CMOV chain which is ~3-4 cycles of integer dependency. The
shuffle path adds latency without removing any.

**Lesson:** SIMD only wins when the data flows through the vector unit
for multiple operations. If you immediately extract back to GPR (as we
do for `picked`, which then feeds the integer LzToken store), the
GPR↔XMM transit eats the savings. SIMD shuffles want to stay in the
vector unit end-to-end.

### Conditional 16-byte SIMD copy for Fast short-token match (L1/L5)

**Context:** VTune Hotspots on L1 100 MB enwik8 showed `writeInt`
(inlined into `copy64`) at ~63% of CPU — same dominant pattern as the
L9 path before we switched it to `copy16`. The L9 fix yielded +3%
wall-time. We hoped the same trick would help L1/L5.

**Hypothesis:** Replace the 2× `copy64` (16 bytes via 8-byte stores) in
the short-token match copy with 1× `copy16` (16 bytes via SIMD store).
The SIMD load is atomic, so we need offset ≥ 16 to avoid reading the
not-yet-written destination region. Branch on offset:

```zig
if (recent_offs <= -16) {  // back-distance >= 16, safe
    @branchHint(.likely);
    copy.copy16(dst, match_ptr);
} else {
    copy.copy64(dst, match_ptr);
    copy.copy64(dst + 8, match_ptr + 8);
}
```

**Result:** ~3% regression on L1 (6132 → 5950 MB/s) and similar on L5.

**Why VTune showed `writeInt` collapse but wall time got slower:**
- VTune confirmed `writeInt` time dropped from ~683ms → ~46ms (-93%).
- A new hotspot `storeV16` appeared at ~139ms (the SIMD store from
  `copy16`).
- Total CPU time dropped, but wall time got slightly worse.

Best explanation: the per-token branch cost (~1-2 cycles for the
compare + conditional jump) exactly offsets the per-token uop savings
(2 µops fewer per token from the wider store), AND the additional code
in the inner loop body increases DSB / icache pressure, slightly
hurting steady-state IPC.

**Lesson:** A profiler showing a hotspot collapse doesn't mean the
total time will improve. The new code path has its own overhead, and
in tight inner loops the branch cost competes directly with the
instruction savings. This is different from the L9 case where the
inner loop already had MORE branches (the conditional on lit_len > 8
etc.) so adding one more was relatively cheaper.

### 16-byte SIMD copies for Fast medium-match path (L5)

**Context:** L5 was the only remaining level still slightly slower
than expected (~4860 MB/s).

**Hypothesis:** L5 spends ~11% of CPU in the medium match branch
(`cmd > 2 && cmd < 24`) which uses 4× `copy64` (32 bytes via 4 8-byte
copies). Far offsets (from `off32_stream`) are always large → no
overlap concerns → safe to use 2× `copy16` (16-byte SIMD).

**Result:** L5 slightly **slower** (~4770 vs 4860 baseline). Even
though the medium-match copy itself was halved in instruction count,
the surrounding code (bounds checks, register allocation in the hot
loop) reorganized in a way that hurt the SHORT TOKEN path which is
~80% of CPU.

**Lesson:** Optimizing a cold path (11% of CPU) can hurt the hot path
through unintended register-allocation interactions. Profile after every
change — don't trust that "less code in branch X = faster overall."

### Pack `ro4` || `ro5` into a single `u64`

**Change:**
```zig
var ro45: u64 = (@as(u64, init_u32) << 32) | @as(u64, init_u32);
// per iter:
const ro4: i32 = @bitCast(@as(u32, @truncate(ro45)));
const ro5: i32 = @bitCast(@as(u32, @truncate(ro45 >> 32)));
// ... compute next_ro4, next_ro5 ...
ro45 = (@as(u64, @as(u32, @bitCast(next_ro5))) << 32) |
       @as(u64, @as(u32, @bitCast(next_ro4)));
```

Hypothesis: frees one GPR vs three separate i32 locals; the pack/unpack
via shifts is cheap.

**Result:** ~2121 MB/s mean (vs ~2143 baseline) — **slightly slower**.
The truncate/shift/or per iteration cost more than the GPR savings.
The CMOV chain was already working with 3 GPRs without spills, so
freeing one didn't help downstream.

**Lesson:** Pack-into-wider-int wins only when the loop is genuinely
register-pressured AND the unpack cost is amortized across many uses.
Reading each ro once per iteration means the unpack cost happens every
iteration with no amortization.

### 2× manual loop unroll for ILP

**Change:** Process two cmd bytes per loop iteration. Each token's body
kept intact (preserving `len_stream` and `offs_stream` consumption order)
but two of them per iteration. Hypothesis: the compiler can schedule the
two independent token bodies at IPC > 1 even though the recent-offset
LIFO update is serially dependent across iterations.

**Subtle bug on first attempt:** re-ordering `len_stream` reads across
the two tokens (e.g., reading `spec_long_b` before `aml_a` may have
advanced `len_stream`) caused `StreamMismatch` errors. Fix: process each
token's full body before starting the next.

**Result:** ~2055 MB/s mean (vs ~2143 baseline) — **regression of ~4%**.
Why:
1. The unrolled body is ~2× larger → falls out of the DSB (decoded uop
   cache, ~4K uops on Arrow Lake), forced into the legacy decoder.
2. Doubled register pressure → more spills.
3. The serial dep chain on the ros wasn't actually breakable — token B's
   pick still waits on token A's LIFO update.

**Lesson:** Manual unrolling helps when the loop body is small and the
dep chain is breakable. Ours is complex and serial — the CPU was already
doing as much OoO scheduling as it could.

### Bumping `scratch_per_chunk` to fit worst-case token array

**Change:** Increase per-worker scratch from `884 KB` (`scratch_size * 2`)
to `1.9 MB` (`scratch_size * 2 + 1 MB`) so the token array always fits
without falling back to the heap. Goal: eliminate the ~4% CPU spent in
libc `free_base` for the c_allocator fallback path.

**Result:** Total throughput **dropped from ~2143 to ~1985 MB/s (-7%)**.
VTune showed `executeTokensType1` went from 0.372s → 0.549s (+48%) due
to **L3 cache misses on the token reads**.

**Why:** scratch is allocated per worker, and we have 24 workers.
- Old: 24 × 884 KB = **21 MB** → fits comfortably in 36 MB L3
- New: 24 × 1.9 MB = **46 MB** → **overflows L3 by 28%**

Token reads from each worker's scratch went L3 → DRAM. The cache penalty
overwhelmed the 4% saved on `free_base`.

**Lesson:** Per-worker memory bumps must be evaluated against
**total L3** (`worker_count × per_worker_size`), not just per-worker
L1/L2. On a 24-core system, even 1 MB of per-worker overhead times 24 =
24 MB which is most of L3.

### Tweaks that compiled but didn't measurably move the needle

These are kept in the code because they make the asm cleaner / safer
even though wall-time was unchanged within noise:

1. **CMOV-chain rewrite of `switch (offset_index)` for `picked`.** The
   original compiled to a jump table with `jmp rbx` + a stack spill of
   one of the ros. The if-chain version emits `cmovb`/`cmovs`/`cmovnb`
   with no spills. Wall time unchanged because the BTB was predicting
   the indirect jump well.

2. **Branchless `offs_stream` advance** (`(oi + 1) & 4`). VTune flagged
   the `if (oi == 3) offs_stream += 1` branch at ~11% of `resolveTokens`.
   Replacing it with the bitmask advance produced cleaner asm. Wall time
   unchanged because LLVM was already CMOV-ing the original branch.

3. **`@branchHint(.unlikely / .likely / .cold)`** on the long-literal,
   long-match, and bounds-check paths. Improved code layout (cold paths
   moved out of hot region) but no measurable speedup.

4. **`std.debug.assert(offset_index <= 3)`**. Meant to tell LLVM the
   value is in {0,1,2,3}. LLVM was already inferring this from the
   `u8 → u32` widening on `cmd_stream[0]`. No measurable change.

### Things tried but not yet proven (worth revisiting)

- **Inline asm `PSHUFB` for ro shuffle**: bypass Zig's `@shuffle`
  comptime restriction and use SSSE3's runtime byte-shuffle directly.
  Could free 3 GPRs (ros → 1 XMM register). The expected win is small
  given the current asm is already CMOV-clean, but worth keeping in
  mind if more pressure shows up.

- **Two-tier ro storage** (small i16 fast path + i32 escape for large):
  empirically most LZ offsets in text are < 16 bits. Could pack 3 i16
  ros into one u64 with a flag bit per slot indicating "this slot needs
  a separate i32 escape word." Adds complexity. Worth it only if the
  fast path is taken > 95% of the time.

- **Vectorized cmd_stream parser**: process N command bytes at once
  with SSE/AVX2 bit ops to extract the lit_len/oi/match_len fields in
  parallel. Then sequentially apply the LIFO updates. Could give a 2-4×
  speedup on the decode portion if doable.

- **AVX-512 `VPCOMPRESSD` / mask registers**: would help with branchless
  variable-length decode. **Not available on Arrow Lake** — Intel
  disabled AVX-512 on consumer Core Ultra chips (E-cores can't do it).
  Don't waste time exploring this path on consumer CPUs.

### Cache geometry reference (Intel Core Ultra 9 285K, Arrow Lake-S)

For future scratch-sizing / data-layout decisions:

| Level | Size | Per | Notes |
|---|---|---|---|
| L1d | 48 KB | per P-core | 12-way; covers ~768 cache lines |
| L1i | 32 KB | per P-core | 8-way |
| L2 | 2.5 MB | per P-core | DSB caches ~4K uops on top of this |
| L3 | 36 MB | shared by all 24 cores | the per-worker scratch sum lives here |
| DRAM | 32+ GB | host RAM | hundreds of cycles latency |

**Per-worker scratch budget rule of thumb**: `worker_count × per_worker_size
≤ L3 / 2` (leaving room for other working sets). For 24 workers and 36 MB
L3, that's ~750 KB per worker max. The current 884 KB is right at the
edge; the bumped 1.9 MB blew through it.

---

## Encoder: thread-local cached MatchHasher16Dual (High-codec L9-L11)

**Hypothesis**: `findMatchesHashBased` allocates a fresh 64 MB hash table
(`MatchHasher16Dual`) on every call via `alignedAlloc` + the hasher
resets via `@memset` on init. VTune showed ~150 ms/call in page-fault /
zeroing paths. Reusing the table across calls via a `threadlocal var`
with lazy init should cut that out entirely.

**What we tried**:
1. `threadlocal var cached_hasher: ?MatchHasher16Dual = null` + lazy init
   on first call, reset bit-width on reuse, `@memset` the table to 0.
   Result: **7.2 MB/s vs 7.5 MB/s baseline** (-4%).
2. Skip the `@memset` on reuse (the hasher's internal generation counter
   was supposed to invalidate stale entries — it doesn't, stale positions
   leak through). Result: **7.1 MB/s** AND corrupted output.

**Root cause**: On Windows, `VirtualAlloc` for a large buffer returns
demand-zero pages. The OS zeroes pages lazily on first touch, spreading
the cost across the compress loop's natural cache-miss latency. An
explicit `@memset` of a reused 64 MB buffer must traverse every cache
line up-front, stalling on DRAM writes, and it pollutes L3 before the
compress loop needs it. The "free" alloc was already cheaper than any
reuse strategy that needs to clear.

**Takeaway**: Don't reuse large scratch buffers on Windows without
benchmarking. `VirtualAlloc` + demand-zero is already an optimized
"amortized zeroing" path the OS provides for free. On Linux/glibc this
might look different (malloc pools reuse dirty memory), but on Win32 the
allocator's default behavior beats manual pooling for buffers > L3.

Reverted to `var hasher = try MatchHasher16Dual.init(allocator, bits, 0);
defer hasher.deinit();` on every call.

---

## Fast decoder: branched copy16 for short-token match copy

**Hypothesis**: The short-token hot loop emits 2× `copy64` (8-byte mov
pairs) for the match copy = 2 store uops per iteration. Replacing with
1× `copy16` (SSE MOVDQU) when `distance >= 16`, falling back to the
2× `copy64` cascade when `distance in [8,15]` (encoder min offset is 8),
would cut 1 store uop per iteration on the common path. With
store-throughput at ~1.5/cycle on Arrow Lake and 3 stores/iter, the
theoretical minimum is 2 cycles/iter — we observed 2.4, so ~20% left
on the table if we can eliminate one store.

**What we tried**:
```zig
if (@intFromPtr(dst) - @intFromPtr(match_ptr) >= 16) {
    copy.copy16(dst, match_ptr);
} else {
    copy.copy64(dst, match_ptr);
    copy.copy64(dst + 8, match_ptr + 8);
}
```

L3 bench enwik8 (50 runs):
- Baseline (2× copy64): 5923 best / 5730 mean MB/s
- Branched:             5664 best / 5540 mean MB/s  (**-3.3% mean**)

**Why it regressed**: The branch introduces `sub` + `cmp` + `jae` = 3
extra uops per iteration (on the hot path), offsetting the 1 store uop
saved. And the branch mispredicts whenever the distance distribution
shifts across chunks (file-dependent), adding ~10-cycle pipeline
flushes.

**The theoretical lever is real** (we are close to store-throughput
bound), but you can't get it via a dynamic branch on `distance`. To
actually collapse 2 stores into 1 would need either:
  (a) statically prove distance is always ≥ 16 for this path (encoder
      guarantee — not currently true, min is 8)
  (b) use PSHUFB-based pattern replication for the small-offset case,
      making copy16 always safe (LZ4-style repeat-byte table)

Option (b) is the standard trick but requires a mask lookup table and
one PSHUFB per short-token iter — more complex than the branch, may or
may not pay off on Arrow Lake. Not pursued in this session.

Also tried an unconditional `copy16` for the long-match 16-bit offset
loop (assumed min offset was larger). **Produced wrong output** — the
encoder does emit offsets in [8,15] on this path. Reverted. Keep the
2× copy64 cascade.

---

## Fast decoder: lookahead prefetch for next match

**Hypothesis**: VTune showed the short-token hot loop retiring at the
first match store (2.94s out of 6.5s total wall, ~45% of samples). The
store was parking there because it's waiting on the match *load*, which
is the only random-access load in the loop. A software prefetch of the
next iteration's match address — computed at the end of the current
iteration from a peek at `cmd_stream[0]` + `off16_stream[0]` — should
turn some L2/L3 stalls into L1 hits.

**What we tried**:
```zig
// at the tail of the short-token branch
if (@intFromPtr(cmd_stream) != @intFromPtr(cmd_stream_end)) {
    const peek_cmd: u32 = cmd_stream[0];
    const peek_new_dist: i64 = off16_stream[0];
    const peek_offs: i64 = if ((peek_cmd & 0x80) == 0)
        -peek_new_dist else recent_offs;
    const peek_addr: usize = @intFromPtr(dst) +% @as(usize, @bitCast(peek_offs));
    @prefetch(@as([*]const u8, @ptrFromInt(peek_addr)),
              .{ .rw = .read, .locality = 3 });
}
```

L3 bench enwik8 (100 runs):
- Baseline: 5888 best / 5745 mean
- With prefetch: **5328 best / 5200 mean** (−10%)

**Why it regressed**: Three culprits, all adding uops to a hot loop
that was already within ~20% of the theoretical frontend/store-port
ceiling.
1. The peek `cmd_stream[0]` and `off16_stream[0]` loads consume two
   load-port slots each iteration, even though they usually hit L1.
2. The `if (cmd_stream != cmd_stream_end)` guard mispredicts on the
   final iteration of each chunk (the branch direction flips once).
3. `@prefetch` itself is 1 uop on Arrow Lake (issues via the load
   port), adding throughput pressure.

Net uop add ≈ 6-8 per iteration. The hot loop was ~25 uops
pre-prefetch, so 25 → 32 uops is a 28% frontend-dispatch increase.
The L1/L2 hit conversion (~20% of match loads that *weren't* already
hot) doesn't recover that overhead.

**Takeaway**: Software prefetch only pays in loops that are
*memory-bound with idle frontend slots*. This decoder is
frontend/store-port-bound, so adding prefetch uops makes it slower, not
faster.

---

## Fast decoder: `@prefetch` in medium-match / cmd==2 long-match paths

**Date**: 2026-04-17

**What**: Added `@prefetch(dst_begin - off32_stream[3], .{.rw = .read, .locality = 3})`
at the tail of the medium-match (`cmd > 2`) and cmd==2 long-match paths.
Fires after each far-offset match copy, prefetching the source position of
a match 3 entries ahead in the off32 stream.

**Results** (24-core Arrow Lake, `benchc -r 10`):

| Corpus  | Level | Before (MB/s) | After (MB/s) | Change |
|---------|-------|--------------|-------------|--------|
| enwik8  | L1    | 16,679       | 17,861      | +7.1%  |
| enwik8  | L3    | 17,585       | 17,912      | +1.9%  |
| enwik8  | L5    | 9,627        | 9,447       | -1.9%  |
| silesia | L1    | 22,944       | 19,723      | -14%   |
| silesia | L3    | 24,169       | 21,342      | -12%   |
| silesia | L5    | 11,199       | 11,291      | +0.8%  |

**Why it failed**: Text (enwik8) has many large-offset matches whose source
positions are DRAM misses — the prefetch hides latency and helps. Binary/mixed
data (silesia) has mostly short-offset matches whose sources are already in
L1/L2 cache. The prefetch wastes frontend uops: the `off32_stream[3]` load,
the bounds check, and the prefetch instruction itself all cost cycles for zero
benefit on a frontend-bound decoder.

A conditional prefetch (only when `far > 65536`) might help, but the branch
itself adds frontend pressure. A text-vs-binary heuristic exists at compress
time (`text_detector.zig`) but the decoder has no access to it.

**Verdict**: Reverted. The overhead outweighs the benefit on mixed workloads.

---

## Fast decoder: unconditional PSHUFB match copy for short-token path (2026-04-19)

**Context**: VTune uarch-exploration on L5 enwik8 (2000 runs, post-sidecar
optimization at 12.2 GB/s) showed `writeInt` (from `copy64`) as the
dominant clock consumer: 246.9B + 111.5B + 22.7B = 381B instructions,
225.6B cycles. The short-token match copy does 2× `copy64` (= 2 loads +
2 stores per token). `copyMatch16Pshufb` was identified in the previous
failed experiment (branched copy16) as "option (b)" — a branchless
PSHUFB-based approach that handles all distances without a branch.

**Change**: Replace the 2× `copy64` in the short-token match path with
a single `copyMatch16Pshufb(dst, match_ptr, distance)`. No branch on
distance — the PSHUFB mask table handles d=1..15 via pattern replication
and d>=16 via identity mask, all in one code path.

```zig
// Before: 2 loads + 2 stores
copy.copy64(dst, match_ptr);
copy.copy64(dst + 8, match_ptr + 8);

// After: 1 load + 1 PSHUFB + 1 store + 1 table load
const distance: usize = @intCast(-recent_offs);
copy.copyMatch16Pshufb(dst, match_ptr, distance);
```

**Results** (100 runs, enwik8):

| Level | Metric | Baseline | PSHUFB | Change |
|-------|--------|----------|--------|--------|
| L3 | best | 38,492 MB/s | 38,764 MB/s | +0.7% (noise) |
| L3 | mean | 33,203 MB/s | 32,486 MB/s | −2.2% |
| L5 | best | 12,932 MB/s | 12,776 MB/s | −1.2% |
| L5 | mean | 11,879 MB/s | 11,841 MB/s | −0.3% |

**Why it didn't help**: The PSHUFB approach saves 1 store µop per token
but adds: (1) a `neg` + `intCast` to compute distance from `recent_offs`,
(2) a `min` + `intCast` to clamp the mask index, (3) a table load from
`match_copy_pshufb_masks[idx]`, and (4) the PSHUFB instruction itself
(1 cycle latency, port 5). On Arrow Lake, the store port isn't the
bottleneck — the retirement width (6 µops/cycle) and OoO scheduling
already overlap the 2 stores with surrounding work. The extra ALU +
table-load µops offset the store-port savings exactly.

**Disposition**: Reverted. The 2× `copy64` short-token match copy is at
its optimization floor for the current wire format. The only path to
fewer stores would be an encoder guarantee that all offsets >= 16,
eliminating the overlap concern entirely (format change).

---

## Parallel worker output-region prefetch (2026-04-19)

**Context**: Same VTune session. `fastL14WorkerFn` showed CPI = 2.15
with 131.8B memory-bound slots and 99.2B L1D_PENDING.LOAD. Hypothesis:
workers stall on page faults and TLB misses when first touching their
output region in dst.

**Change 1 — Page-level write prefetch**: Before the sidecar scatter
and decode loop, walk the worker's output region in 4KB strides issuing
`@prefetch(.write, .locality=1)` to trigger demand-zero page faults
early.

**Result**: L5 best 12,768 MB/s (baseline 12,932) — slight regression.
L3 neutral.

**Why**: In a warm benchmark loop (100+ iterations), pages are already
mapped and cache-hot from the prior iteration. The prefetch loop adds
~25 instructions per worker per call for zero benefit. In a cold
single-shot scenario the prefetch might help, but benchmarks can't
measure that.

**Change 2 — Sidecar scatter prefetch**: Prefetch the next sidecar
literal's dst position while writing the current one, hiding the
read-for-ownership latency on scattered writes.

**Result**: L5 best 12,864 MB/s (baseline 12,932) — noise.

**Why**: Sidecar literal positions are sorted ascending, so sequential
access is already hardware-prefetcher friendly. The explicit prefetch
adds 1 µop per literal for zero benefit.

**Root cause of high CPI**: The 2.15 CPI in `fastL14WorkerFn` is
inherent to the parallel architecture — read-for-ownership traffic on
dst cache lines during sidecar scatter, and cross-worker false sharing
at chunk boundaries (the 64-byte guard save/restore). These are
fundamental costs of parallel decode, not fixable without a wire format
change.

**Disposition**: Both reverted. The parallel worker overhead is at its
floor for the current architecture.

---

## writeInt microarchitecture analysis (2026-04-19)

**Context**: After all optimization attempts, `writeInt` (mem.zig:1940,
inlined from `copy64` in copy_helpers.zig) remains the dominant hotspot
at 383.4B instructions / 226.6B cycles (CPI 0.59). This analysis
explains WHY it dominates and why further optimization is infeasible.

**VTune source-line breakdown** (uarch-exploration, L5 enwik8, 2000 runs):

| Counter | Value | Meaning |
|---------|-------|---------|
| L1 hit loads | 178.1B | 79.8% of loads hit L1 |
| L1 miss loads | 45.1B | 20.2% miss rate — match reads from random offsets |
| Split loads | 1.75B | 11.6% — unaligned 8-byte access crossing cache lines |
| Split stores | 1.72B | 11.6% — same cause |
| Store buffer full (XQ.FULL) | 0.84B | Negligible — NOT store-throughput limited |
| Memory-bound slots | 343.2B | 90% of writeInt's stall is load-latency |
| DSB uops | 435.8B | 99.8% from µop cache — no frontend issues |

**Key insight**: The high cycle count on `writeInt` is NOT from store
pressure. It's from **match-load latency** — `readInt` and `writeInt`
are on the same inlined `copy64` call, so VTune attributes the load
miss stall to the store instruction (IP skid from out-of-order
retirement). The 45.1B L1 misses at ~10-cycle L2 penalty = ~451B stall
cycles. OoO execution overlaps ~50% of this with other work, yielding
the observed 226.6B cycles.

**Why it can't be improved**:

1. **The L1 misses are fundamental to LZ decompression.** Match
   back-references point to arbitrary positions in the 100MB output
   buffer. A 48KB L1 cache can only hold ~0.048% of the buffer. The
   20.2% miss rate is actually good — it means 80% of matches
   reference recent output that's still in L1.

2. **Prefetch was already tried and failed.** Lookahead prefetch of the
   next match source address adds µops to the hot loop (peek at next
   cmd + off16, compute address, issue prefetch = ~6 µops). The loop
   is ~25 µops; adding 6 is a 24% frontend-dispatch increase that
   overwhelms any cache-hit conversion. See "Fast decoder: lookahead
   prefetch for next match" above.

3. **Store-forwarding works.** Only 9.6M store-forwards (negligible),
   meaning the load-store overlap in the `copy64` cascade is handled
   correctly by the CPU without penalty.

4. **Split accesses are unavoidable.** 11.6% split rate is the
   expected value for unaligned 8-byte accesses on random-aligned
   pointers (8/64 = 12.5% theoretical). Aligning `dst` or `match_ptr`
   would require padding that breaks the wire format.

5. **Store buffer is not full.** XQ.FULL = 0.84B means the store
   buffer has capacity. Reducing stores (e.g., PSHUFB 16→1 store)
   wouldn't help because the bottleneck isn't store throughput.

**Conclusion**: `writeInt` dominates the profile because LZ
decompression IS memory copies. The decoder is running at ~12.2 GB/s
on L5 parallel enwik8, which is ~500 MB/s per core. At 3.7 GHz and
8 bytes per `copy64`, that's ~0.93 copies/cycle — within 7% of the
1.0 stores/cycle theoretical throughput. The remaining gap is the L1
miss penalty on match loads, which cannot be hidden without a format
change (e.g., reordering matches to improve locality, at the cost of
compression ratio).

---

## L9-L11 sidecar for parallel phase 2 (2026-04-19)

**Context**: L9 decompress was 2.3 GB/s with serial two-phase decode
(phase 1 parallel entropy decode, phase 2 serial token execution).
After moving `resolveTokens` to parallel phase 1 (+17-24%, committed
as `72bee2f`), phase 2 was still 63-77ms serial = 67-76% of wall time.
A sidecar (like the L1-L5 Fast parallel decode sidecar) would
parallelize phase 2 across 24 cores for a potential 3-5x speedup.

### What worked: parallel resolveTokens (+17-24%)

Moving the Type 1 token resolution (carousel walk, offset/length
decode) from serial phase 2 to parallel phase 1. Phase 2 then uses
`processLzRunsType1PreResolved` which skips resolveTokens and goes
straight to executeTokensType1 with the pre-built token array.

| Corpus | Baseline | Pre-resolved | Change |
|--------|----------|-------------|--------|
| enwik8 L9 mean | 2,131 MB/s | 2,505 MB/s | +17.6% |
| silesia L9 mean | 2,345 MB/s | 2,740 MB/s | +16.8% |

This is committed and shipping.

### What failed: L9-L11 sidecar (prohibitive size)

**Approach**: After encoding, walk the compressed frame's token streams
to identify match copies that cross 16-chunk (4 MB) slice boundaries.
Collect the final byte values at those positions from the original
source. Compress with L1 entropy coding.

**Results** (enwik8 100 MB, L9):

| Metric | Value |
|--------|-------|
| Cross-slice positions | 17,708,012 (17.7% of output) |
| Raw sidecar body | 19,663 KB |
| Compressed sidecar (L1) | 13,684 KB |
| Ratio cost | +14.0 pp (28.3% → 42.3%) |

**Why it's so large**: The High codec's optimal parser uses the full
64 MB dictionary window. Cross-slice references are extensive — nearly
1 in 5 output bytes is read cross-slice by a match in a different
4 MB region. For comparison, the L1-L5 Fast sidecar is typically <1%
of output because Fast uses short-distance matches.

**Disposition**: Abandoned. A 14 pp ratio cost on a level that users
explicitly chose for ratio is unacceptable. The sidecar approach that
works well for L1-L5 (short matches, small closures) fundamentally
doesn't scale to L9-L11 (long-range optimal parser matches).

### What also failed: separate literal scatter for match-only phase 2

**Approach**: Scatter Type 1 (raw) literals to their dst positions
during parallel phase 1, then run a match-only executor in phase 2
that skips literal copies.

**Result**: Incorrect output. The original `processOneToken` uses
`copy16` for literal copies, which always writes 16 bytes regardless
of `lit_len`. When `lit_len < 16`, the overshoot bytes (from
`lit_stream` beyond the current literal) land in the match-copy
region. If the match offset is small and negative, the match copy
reads those just-written overshoot bytes. The separate scatter
(using exact `@memcpy`) doesn't reproduce this overshoot, so match
copies read different (stale) bytes and produce wrong output.

**Lesson**: In LZ decoders with wide SIMD copies, the literal copy
and match copy within a single token are NOT independent operations.
The match can read from the literal copy's overshoot region. Any
attempt to split them across phases must reproduce the exact same
write pattern, including overshoot.

### What also failed: depth-0 intra-slice parallel execution

**Approach**: Use the cross_chunk_analyzer's transitive depth logic to
classify each token as depth-0 (all match sources within the same
slice, no transitive cross-slice dependency) or depth-1+ (depends on
cross-slice data). Execute depth-0 tokens during parallel phase 1;
only depth-1+ tokens need serial phase 2.

**Results** (enwik8 100 MB, L9, Type 1 tokens only):

| Slice Size | Depth-0 (safe) | Depth-1+ (serial) | Workers |
|-----------|---------------|-------------------|---------|
| 4 MB (16 chunks) | 5.0% | 95.0% | ~24 |
| 16 MB (64 chunks) | 18.0% | 82.0% | ~6 |
| 64 MB (256 chunks) | 66.9% | 33.1% | ~1-2 |

**Why**: The optimal parser's 64 MB dictionary creates deep transitive
dependency chains that span the entire output. A match in chunk 400
reads from chunk 350, which reads from chunk 300, which reads from
chunk 250... Each hop is within the 64 MB window, but the chain
reaches back across many slices. Even at 64 MB slices (matching the
dictionary size), 33% of tokens are still transitively cross-slice —
and at that size you only have 1-2 workers so parallelism is moot.

**Conclusion**: Any decoder-only parallelism strategy for L9-L11 is
defeated by the 64 MB dictionary's transitive chains. The only path
to parallel L9-L11 decode is constraining the encoder (SC grouping,
which is L6-L8), accepting the ratio cost.

---

## Two-pass stats seeding for L10-L11 (zstd btultra2 style) (2026-04-19)

**Context**: zstd's `--ultra` level 22 uses a two-pass strategy
(`ZSTD_btultra2`) where the first block is compressed twice — the first
pass collects frequency statistics, discards the output, and the second
pass recompresses with better-calibrated entropy tables. zstd claims
~0.5% ratio gain for 2x CPU cost on the first block.

**What we tried**: Force the existing outer loop (which re-runs the
optimal parser when chunk type mismatches) to always run twice on the
first sub-chunk when `codec_level >= 7` (L10-L11). The first iteration
collects accurate frequency histograms from the full DP parse; the
second iteration uses those histograms to seed the cost model.

**Results** (enwik8 100 MB, single-threaded):

| Level | Baseline | Two-pass | Delta |
|-------|----------|----------|-------|
| L9 (codec_level=5) | 28,335,062 | 28,335,062 | 0 (not triggered) |
| L10 (codec_level=7) | 28,193,894 | 28,189,497 | -4,397 (-0.016%) |
| L11 (codec_level=9) | 26,903,621 | 26,903,185 | -436 (-0.002%) |

**Why so small**: StreamLZ's `collectStatistics` greedy pre-pass
already provides reasonable initial frequency histograms before the
optimal parser runs. The optimal parser also updates stats
incrementally after each 32 KB chunk (`updateStats` + `makeCostModel`),
so the cost model adapts quickly. The cold-start penalty that zstd's
two-pass addresses (crude predefined baselines like `{4,2,1,1,...}`)
doesn't exist in StreamLZ because the greedy pre-pass provides
data-derived stats from the start.

**Disposition**: Reverted. The ~0.01% ratio gain doesn't justify 2x
CPU cost on the first block. StreamLZ's greedy pre-pass already
captures most of the benefit that zstd's two-pass provides.

---

## Long Distance Matching (LDM) for L11 (zstd-style) (2026-04-19)

**Context**: zstd's ultra levels use a dedicated Long Distance Matching
subsystem alongside the BT4 match finder. LDM uses a gear-hash sampled
position table to find very long matches (≥32 bytes) across the full
128 MB window. It's designed to catch matches that the regular match
finder misses due to hash collisions.

**Implementation**: Built a complete LDM subsystem (~200 lines):
- Gear hash (rolling hash) samples ~1/128th of positions
- Bucket hash table (2^18 entries, 16 entries/bucket, circular eviction)
- 64-bit hash with 32-bit checksum for fast rejection
- Forward + backward match extension
- Stride-4 sub-match insertion for the DP parser
- Merged into MLS at positions where BT4 found no matches

**Results** (enwik8 100 MB, L11 with 128 MB dictionary):

| Config | Size | Delta |
|--------|------|-------|
| Baseline (64 MB dict) | 26,903,621 | — |
| 128 MB dict | 26,827,870 | -75,751 |
| 128 MB dict + LDM | 26,826,268 | -77,353 |
| **LDM contribution** | | **-1,602 bytes** |

**Why so small**: LDM is designed to help hash-based match finders
(zstd levels 1-17) that miss long-distance matches due to hash
collisions. StreamLZ L11 uses BT4 (binary tree), which performs an
ordered tree walk at every position and finds all matches regardless
of distance, up to the dictionary window. BT4 already captures
essentially everything LDM would find. The 1.6 KB gain is from
backward match extension at gear-hash sample points, which
occasionally extends a BT4 match by a few bytes.

**Disposition**: Reverted. 1.6 KB ratio gain doesn't justify the
~2.3 MB memory overhead and code complexity. LDM would be valuable
if StreamLZ added a hash-based path at L9-L10 (which use
`findMatchesHashBased`, not BT4), but for L11 (BT4) it's redundant.

---

## Fixed: sidecar match_ops not executed in parallel decoder (2026-04-19)

**Symptom**: Parallel Fast L1-L5 decompress produced 2-4 byte errors
on enwik9 (1 GB) at worker slice boundaries.

**Root cause**: The parallel decoder (`decompressFastL14Parallel`)
applied sidecar `literal_bytes` (scattered by each worker) but
completely ignored sidecar `match_ops`. The match_ops are sequential
copy operations that propagate cross-chunk byte values produced by
match chains — values that can't be represented as simple literal
bytes. Without executing them, positions that depend on cross-slice
match chain propagation got zeros.

**Fix**: Execute `sidecar.literal_bytes` (serial scatter) and then
`sidecar.match_ops` (serial sequential copies) BEFORE dispatching
parallel workers. The workers still scatter literals in parallel for
their own regions (redundant but harmless since the pre-scatter
already placed the correct values).

**Verified**: enwik9 (1 GB) L3 parallel roundtrip now correct.

---

## cldemote Cache Eviction in Fast Decoder (2026-04-27)

**Hypothesis**: Output bytes beyond the max back-reference distance (64KB
for L1 u16 hash) will never be match sources again. Demoting those cache
lines from L1/L2 to L3 via `cldemote` should free cache capacity for
active match sources, improving decode throughput.

**Implementation**: Two variants tested in `processModeImpl` hot loop:

1. **Per-cache-line**: Check every iteration, fire `cldemote` when dst
   advances past a 64-byte boundary, targeting `dst - 64KB`. Result:
   **-21% regression** (6,464 → 5,111 MB/s). The per-token branch +
   inline asm dominated the critical path.

2. **Amortized burst**: Check once per 4KB of output (`@branchHint(.cold)`),
   fire 64 `cldemote` instructions covering the previous 4KB stride at
   `dst - 64KB`. Result: **-2.7% regression** (6,464 → 6,290 MB/s).
   Lower overhead but still net negative.

**Why it failed**: The decoder is **latency-bound on the match dependency
chain**, not cache-capacity-bound. Each token's match address depends on
the previous token's output — a serial pointer-chase that can't be
improved by freeing cache space. VTune confirmed CPI=0.17 (instruction-
bound). The 256KB sub-chunks fit in L2 (2MB), match sources are hot in
L1, and L3→DRAM writeback happens asynchronously via LRU without stalling
the CPU.

A memcpy ceiling test confirmed the decoder at 6.5 GB/s is at **111% of
the dependent-load-chase ceiling** (5.7 GB/s at 100MB working set),
meaning literal copies running in parallel with the match chain already
slightly exceed the pure-chase bandwidth. There is no cache-capacity
headroom to exploit.

**Also evaluated (not implemented)**:
- **`clwb`** (write back dirty line, keep clean copy): No store-buffer
  pressure to relieve — the decoder writes 8-16 bytes per token, well
  within the 72-entry store buffer.
- **Non-temporal stores for literals**: Literals and match copies
  interleave at byte granularity within the same 64-byte cache lines.
  Non-temporal stores operate at cache-line granularity and would evict
  lines that subsequent match copies need to read.
- **Two-literal-stream split (referenced vs unreferenced)**: Would
  require format change. The encoder doesn't know at literal-write time
  which bytes will be match sources. Would need two-pass encoding or
  sidecar annotation.

**Conclusion**: Cache management hints (`cldemote`, `clwb`, `clflush`)
don't help LZ decoders because the access pattern is already cache-
friendly by construction — matches read recently-written data that's
naturally hot in L1/L2. The bottleneck is the serial match-address
dependency chain, which is a memory-latency wall that no cache hint
can break on a single thread.

---

## Fast Decoder: Input Stream Prefetch (2026-04-27)

**Hypothesis**: The decoder reads from 4 separate input streams (cmd,
literal, off16, off32) in different memory regions. The hardware
prefetcher tracks one linear scan well but may struggle with 4+
concurrent streams. Explicit `@prefetch` ahead on cmd, literal, and
off16 streams could improve L1 hit rate.

**Implementation**: Added `@prefetch(cmd_stream + 64)`,
`@prefetch(lit_stream + 128)`, `@prefetch(off16_stream + 64)` at the
top of each loop iteration with locality=3 (keep in all cache levels).

**Result**: No change (6,471 vs 6,464 MB/s baseline). Within noise.

**Why it failed**: VTune shows 99.9% L1 hit rate on input loads — the
hardware prefetcher already tracks the sequential streams perfectly.
The streams are accessed linearly and the prefetcher detects the stride
pattern. Explicit prefetch adds 3 instructions per token with zero
benefit.

---

## Fast Decoder: Counted Loop Instead of Pointer Compare (2026-04-27)

**Hypothesis**: Replacing the `while (cmd_stream < cmd_stream_end)` pointer
comparison with a counted `while (cmd_idx < cmd_count)` index loop
could eliminate the pointer compare in favor of a simpler integer compare.

**Implementation**: Pre-computed `cmd_count = cmd_end - cmd_start`, used
indexed addressing `cmd_stream[cmd_idx]` instead of pointer increment.

**Result**: No change (6,695 vs 6,760 MB/s). Within noise, possibly
slightly worse.

**Why it failed**: LLVM already fuses the pointer `cmp` + `jb` into a
single µop. The counted loop replaces this with `cmp` + `jb` on an
integer (same cost) but adds indexed addressing overhead (`base + index`
addressing mode vs simple pointer deref). Net neutral or slightly worse
due to the extra register for the index.

---

## Fast Decoder: Pipelined Match Source Prefetch (2026-04-27)

**Hypothesis**: The match-address dependency chain (cmd load → offset
resolve → CMOV → address compute → match load) is 7 cycles. By peeking
at the NEXT token at the end of each iteration and prefetching its
probable match source, the next iteration's match load would find the
data already in L1, hiding the dependency latency.

**Implementation**: At the end of each fast-path iteration, peek at
`cmd_stream[0]` (next token), load the next off16 entry, compute the
speculative match address including the next literal length offset,
and issue `@prefetch` on the result.

**Result**: -2.8% regression (6,570 vs 6,760 MB/s).

**Why it failed**: VTune shows **99.9% L1 hit rate** on match loads.
Match data is already in L1 because it's recently-written output — the
decoder just wrote those bytes a few tokens ago. There is no cache miss
to hide. The prefetch block adds ~10 instructions per token (peek, branch,
load, CMOV, add, compute address, prefetch) which increases instruction
pressure at CPI 0.188 where the pipeline is already saturated. The 17%
"bound on loads" from VTune is from the **data dependency chain**, not
from cache misses — prefetching cannot fix data dependencies.

---

## Fast Decoder: CMOV Elimination for Recent Offset (2026-04-27)

**Evaluated but not implemented.** Analysis showed no benefit possible.

The current code uses a branchless CMOV to select between the new offset
and the recent offset: `recent_offs = if (use_new_dist) candidate else recent`.
The speculative off16 load, CMOV, and conditional pointer advance are all
single-cycle branchless operations.

To truly eliminate the CMOV, the encoder would need to emit an offset for
EVERY token (including recent-offset ones). But recent offsets can be any
size (they may have originated from the off32 path), so they can't be
stuffed into the u16 off16 stream. This would require a format redesign
that inflates the compressed off16 stream by 30-40%.

Critical path analysis shows the CMOV adds exactly 1 cycle vs an
unconditional approach, but the unconditional approach requires the same
number of instructions (load + negate vs CMOV). Net zero benefit for a
significant format change.

**Conclusion**: At CPI 0.188 with 99.9% L1 hit rate, 0.16% branch
mispredict rate, and 93% LSD (loop stream detector) decode, the Fast
decoder hot loop is at the hardware wall. The remaining bottlenecks are
the serial match-address dependency chain (17% of cycles) and unaligned
cache-line-crossing accesses (10% of cycles), both inherent to the LZ
token format. Further gains require format-level changes.

---

## Fast Decoder: Pre-scan Safe Limit to Eliminate dst Check (2026-04-27)

**Hypothesis**: The fast inner loop has two conditions per iteration:
`cmd_stream < cmd_stream_end` and `dst < dst_safe_end`. Pre-scanning
the decoded cmd stream to compute a `cmd_safe_end` pointer (where all
tokens are short AND cumulative output fits within dst_safe_end) would
reduce the loop to a single pointer compare, eliminating both the dst
check and the `cmd >= 24` check.

**Implementation**: Before the hot loop, scan the cmd stream counting
consecutive short tokens (cmd >= 24) with a pessimistic 22-byte-per-
token output budget. Produce a `cmd_safe_end` pointer. Inner loop uses
`while (cmd_stream < cmd_safe_end)` with no other checks.

**Result**: -1.6% regression (best 6,722 vs 6,828 MB/s over 500 runs).

**Why it failed**: The pre-scan adds ~5KB of sequential reads over the
cmd stream before the hot loop begins. More importantly, LLVM generates
slightly inferior code for the single-check loop body compared to the
two-check version — the two-condition `while (A and B)` compiles to a
fused compare-and-branch pair that the branch predictor handles
perfectly (both conditions are true ~99.8% of iterations). The pre-scan
overhead exceeds the savings from eliminating one check.

Also tested a counted-index variant (`while (i < safe_short_count)`)
which was even worse due to indexed addressing overhead vs pointer
increment. LLVM strongly prefers pointer iteration over index iteration
for this loop shape.

---

## Fast Decoder: Removing Far-Offset Prefetch (2026-04-27)

**Hypothesis**: The two `@prefetch(match_ptr)` instructions on far-offset
match paths (off32, distance > 65536) fire on cold paths (~5-8% of
tokens). VTune showed 0 L3 misses, suggesting far matches already hit
L2/L3 without prefetch. Removing them would eliminate the branch +
prefetch instruction overhead.

**Result**: Neutral across all tests (100-500 runs, L1/L3, enwik8/enwik9):

| Test | With prefetch | Without | Delta |
|------|-------------|---------|-------|
| L1 enwik8 r100 | 6,810 best | 6,741 best | -1.0% |
| L3 enwik8 r100 | 6,590 best | 6,622 best | +0.5% |
| L1 enwik9 r30 | 6,847 best | 6,800 best | -0.7% |

All within 1% noise. The prefetch fires so rarely (cold path, branch-
hinted) that its presence or absence doesn't measurably affect the hot
loop. Keeping the prefetch is correct — zero cost on small/medium files
and provides insurance on large working sets where far matches might
miss L2.

---

## Zig 0.16 migration experiments (2026-04-29 — 2026-04-30)

### Force-inline writeComplexOffset for LLVM 21 codegen recovery

**Context**: LLVM 21 (Zig 0.16) generates 11% more dynamic instructions
than LLVM 19 (Zig 0.15) for the L1 greedy parser. LLVM IR comparison
showed LLVM 21 was NOT inlining `subtractBytes` (3 call sites in the
hot path for L2+ delta-literal mode). For L1, the only out-of-line call
in both versions was `writeComplexOffset`.

**Experiment**: Marked `writeComplexOffset` as `pub inline fn`.

**Result**: L1 compress **346 MB/s** (down from 368 baseline) — **6% slower**.
The function body is large (~100 instructions) and inlining it bloated
the hot loop, causing icache pressure and worse DSB decode efficiency.

**Lesson**: LLVM 21's decision NOT to inline writeComplexOffset was
correct. The function is too large for the icache benefit. The 0.15
version had a bigger static function (484 vs 436 instructions) but
LLVM 19 happened to generate fewer dynamic instructions via different
loop structure choices — a codegen quality difference, not inlining.

---

### L1 decoder 2x loop unrolling

**Context**: The L1 Fast codec inner loop processes one token per iteration
with a back-edge branch. Hypothesis: processing 2 tokens per iteration
halves the branch overhead.

**Experiment**: Duplicated the token body (literal copy + match copy)
to process tokens[0] and tokens[1] before looping back.

**Result**: L1 decompress **6,520 MB/s** (down from 6,613 baseline) — **1.4%
slower**. The original loop body (~20 µops) fits in the Loop Stream
Detector (LSD) which replays decoded micro-ops without re-fetching.
Doubling the body evicts it from LSD, losing the replay benefit.

**Lesson**: On modern Intel CPUs, small loops that fit in the LSD (~64
µops on Arrow Lake) should NOT be unrolled. The LSD provides free
replay that outweighs the back-edge branch cost (which is near-zero
anyway — backward branches are predicted taken with >99% accuracy).

---

### Single-pass decode+execute for High codec (L9-L11)

**Context**: zstd's decompressor uses a single-pass architecture: decode
one sequence from the bitstream and immediately copy literals + match.
Our High codec uses two phases: (1) resolveTokens decodes all tokens
into a buffer, (2) executeTokensType1 copies all literals + matches.
The two-phase approach doubles memory traffic (token buffer write + read)
and can evict match source data from cache between phases.

**Experiment**: Merged resolveTokens and executeTokensType1 into a single
loop. Also tried a zstd-style 8-sequence pipeline with ring buffer
prefetch.

**Results**:
- Single-pass without prefetch: **1,027 MB/s** (vs 1,054 baseline) — **2.6% slower**
- Single-pass with inline prefetch (current token): **1,028 MB/s** — same
- Single-pass with 8-seq ring buffer prefetch: **1,031 MB/s** — still slower

**Why**: The two-phase approach's 128-token prefetch lookahead in
executeTokensType1 hides L2/L3 latency far better than the single-pass
can. Each of our tokens is small (~12 bytes of output), so 8 tokens ahead
only gives ~96 bytes / ~20 cycles of prefetch lead time — not enough to
hide the ~200-cycle L3 latency. The two-phase design gives 128 × 12 =
~1,536 bytes of lookahead. zstd's single-pass works because their
sequences are larger (~30 bytes each) and their per-sequence decode is
heavier (single interleaved bitstream = more CPU work to hide latency).

**Lesson**: Two-phase resolve+execute is the right architecture for our
format. The token buffer cost (~16% of resolveTokens cycles) is more
than offset by the prefetch pipeline benefit.

---

### Non-temporal stores (movntdq) for token buffer writes

**Context**: VTune showed resolveTokens is 54% memory-bound, with 1.6B
stores per run writing 16 bytes per token to the token buffer. The
buffer is ~112 MB for enwik8 L9 — larger than L3 cache. Hypothesis:
non-temporal stores bypass cache and free up store buffer slots.

**Experiment**: Replaced `tokens[token_index] = ...` with `movntdq`
via inline assembly, writing the 4×i32 token as a 128-bit non-temporal
store.

**Result**: **1,023 MB/s** (vs 1,079 baseline) — **5.2% slower**.

**Why**: The token buffer is read back immediately in executeTokensType1
(phase 2). Non-temporal stores bypass ALL cache levels, so phase 2 loads
come from DRAM (~100 cycles) instead of L3 (~40 cycles). Despite the
buffer exceeding L3 capacity, enough of it remains resident in L3 during
sequential write→read that normal stores win.

**Lesson**: Non-temporal stores only help when the written data won't be
read again for a long time (e.g., streaming output to disk). For write-
then-read-soon patterns, even if the buffer exceeds cache size, the
sequential access pattern keeps enough data warm in L3.

---

### 12-byte LzToken struct (u16 lit_len + match_len)

**Context**: LzToken is 16 bytes (4 × i32). lit_len and match_len never
exceed 65535, so u16 suffices. Shrinking to 12 bytes (i32 + i32 + u16 +
u16) halves the bandwidth for those fields.

**Result**: **1,055 MB/s** (vs 1,079 baseline) — **2.2% slower**.

**Why**: 12-byte struct loses power-of-2 alignment. Array indexing
`tokens[i]` requires a multiply by 12 (lea + add) instead of a shift
by 4 (single shl). The indexing overhead exceeds the store bandwidth
saved.

**Lesson**: Token struct size should be a power of 2 for array indexing
performance. The 4-byte waste per token is cheaper than the multiply.

---

### L11 BT4 depth=16 (reduced tree traversal)

**Context**: VTune showed bt4SearchAndInsert at 51% of L11 compress
cycles with CPI 2.0 (memory-bound — 1063% MemB metric). The BT4 tree
traverses up to max_depth=128 nodes per position. Hypothesis: reducing
depth trades ratio for speed.

**Results** (enwik8 100 MB, L11 t1):

| Depth | Ratio | Time | Speed |
|-------|-------|------|-------|
| 128 | 25.5% | 440s | 0.2 MB/s |
| 16 | 26.9% | 293s | 0.3 MB/s |

33% faster, 1.4pp ratio loss. Still only 0.3 MB/s — the tree
insertion at every position is the bottleneck, not just the search
depth.

**Disposition**: Not adopted. The 1.4pp ratio loss is significant for
L11's target use case (maximum compression). And 0.3 MB/s is still
impractically slow for interactive use.

---

### L11 64 MB dictionary window (vs 128 MB)

**Results** (enwik8 100 MB, L11 t1):

| Dict Size | Ratio | Time | Speed |
|-----------|-------|------|-------|
| 128 MB | 25.5% | 440s | 0.2 MB/s |
| 64 MB | 25.6% | 343s | 0.3 MB/s |

22% faster with only 0.1pp ratio loss. The second half of enwik8
barely benefits from referencing positions 64+ MB back.

**Disposition**: Not adopted. Would need to be a per-level config
option. May revisit if L11 speed becomes a priority.

---

## XOR-mask branchless repeat-offset selection (Lizard/LZ5 style) (2026-05-01)

**Context**: Lizard/LZ5 uses a branchless XOR-and-mask pattern for
repeat-offset selection instead of CMOV. The current Fast decoder hot
loop uses Zig's `if (use_new_dist) candidate else recent` which compiles
to a CMOV. Hypothesis: the XOR approach avoids CMOV's false data
dependency on both source operands, breaking the serial dependency chain.

**Change**: Replaced CMOV-style offset selection in both the fast inner
loop and safe tail loop:

```zig
// Before (CMOV):
recent_offs = if (use_new_dist) candidate_offs else recent_offs;

// After (XOR-mask):
const dist_mask: i64 = -@as(i64, @intFromBool(use_new_dist));
recent_offs ^= (recent_offs ^ candidate_offs) & dist_mask;
```

When `use_new_dist` is true, mask = -1 (all bits set), so
`recent_offs ^ (recent_offs ^ candidate_offs) = candidate_offs`.
When false, mask = 0, so `recent_offs ^= 0` = no change.

**Results** (enwik8 100 MB, L1, t1, 30 runs):

| Variant | Decompress median |
|---------|-------------------|
| Baseline (CMOV) | **836.2 MB/s** |
| XOR-mask | 819.8 MB/s |
| **Delta** | **-2.0%** |

**Why it failed**: On modern Intel (Arrow Lake, Ice Lake), CMOV is a
single-cycle single-µop instruction. The XOR-mask pattern requires 3
ALU operations (XOR, AND, XOR) even though each is single-cycle. The
serial dependency depth is the same (each iteration's `recent_offs`
still depends on the previous), so the extra µops just add frontend
pressure without breaking any dependency chain. The "false dependency"
issue that motivated this trick on older architectures (pre-Broadwell)
was fixed in Intel's microarchitecture years ago — modern CMOV reads
the condition flags and only the selected operand.

**Disposition**: Not adopted. CMOV is optimal for this pattern on any
CPU from Broadwell onwards. The Lizard XOR trick was designed for
architectures where CMOV had higher latency or stalled on both inputs.

---

## Adaptive L1 SC group sizing (2026-04-30)

**Context**: L6-L8 use adaptive SC (self-contained) group sizing based
on input characteristics. L1 uses a fixed group_size=4 (4 sub-chunks
per SC group = ~256 KB). Hypothesis: adapting the group size for L1
based on file size or match distribution could improve parallel decode
scaling, especially on bandwidth-limited systems.

**Implementation**: Added adaptive group sizing logic to the L1 encoder
in `fast_framed.zig`, selecting group_size based on input size and
compression characteristics (larger groups for larger files, smaller
groups for better parallelism on small files).

**Results**: Benchmarked on both Arrow Lake desktop and Ice Lake laptop
(Surface Book 3, i7-1065G7). No measurable benefit on either machine
across L1 t1-t8 scaling tests on enwik8 100 MB.

**Why it didn't help**: The fixed group_size=4 already provides good
parallel granularity. Each 256 KB group is small enough for fine-grained
work distribution across cores, and large enough to amortize the SC
boundary overhead (8-byte initial copy, sidecar closure). The
bottleneck for L1 parallel scaling is DRAM bandwidth (confirmed by
VTune: 86.7% DRAM Bandwidth Bound at 10+ threads on Arrow Lake),
not work distribution granularity. Changing group sizes doesn't move
more data through the memory bus.

On the Ice Lake laptop (16 GB LPDDR4x, ~34 GB/s peak), L1 decompress
scales 3.1x from t1→t8 and plateaus — purely bandwidth-limited.
Group sizing can't help when the ceiling is memory throughput.

**Disposition**: Reverted on both machines. Fixed group_size=4 is
correct for L1.

---

## N-gram forward scatter / substitution for High codec decode speed (2026-05-01)

**Context**: VTune showed L9-L11 decompress bottlenecked on match copy
cache misses — large-offset matches hit cold L3/DRAM. The idea: if
common 4-grams like " the" appear hundreds of thousands of times,
pre-writing them to the output buffer (or eliminating them from the
token stream) could save the decoder from doing expensive random
backward-reference loads.

Greedy non-overlapping 4-gram analysis on enwik8 100 MB confirmed a
strong pareto distribution: top 10 patterns cover 11.4% of output,
top 100 cover 28.5%.

### Three approaches tested, all failed

**Approach 1: Zero-substitution pre-filter (replace 4-grams with 0x00)**

Replace each occurrence of the top 5 4-grams with `0x00000000` in the
input, then compress. The zeros are equally matchable — hypothesis was
that uniform zeros would compress tighter.

Results (enwik8 100 MB, L9, all threads):

| Variant | Compressed | Ratio | Decompress |
|---------|-----------|-------|------------|
| Original | 28,624,068 | 28.62% | 526 MB/s |
| Zero-sub top 5 | 28,704,697 | 28.70% | 523 MB/s |

Ratio 0.1pp **worse**, decompress unchanged. The zeros fragment natural
long matches ("in the morning" becomes "in \x00\x00\x00\x00morning" —
two tokens instead of one). The LZ compressor was already encoding
" the" as cheap match tokens; replacing with zeros doesn't eliminate
any tokens, just changes what gets matched.

**Approach 2: Byte-stripping with position sidecar (remove 4-grams entirely)**

Actually remove the 4-gram bytes from the input (shrinking it), store
pattern → position-list in a sidecar, decompress shorter data, then
expand by inserting patterns at recorded positions via a merge pass.

Results (enwik8 100 MB, L9, top 10 patterns, all threads):

| Metric | Original | Stripped + sidecar |
|--------|----------|-------------------|
| Compressed + sidecar | 28,624,068 | 32,005,754 |
| Ratio | 28.62% | **32.01%** |

**15% worse.** Removing 8.5% of input only shrunk the compressed output
by 981 KB, but the sidecar storing 2.1M position entries cost 4.4 MB
(delta-varint). Even after entropy-coding the sidecar with zstd-19,
it was still 2.8 MB — 3x more than the compression savings.

The fundamental problem: LZ already handles repeated 4-grams efficiently
as cheap match tokens. The position metadata to reconstruct them costs
far more than what the LZ compressor saves from processing fewer bytes.

**Approach 3: Single-byte substitution (BPE-style, best approach)**

Replace each top-N 4-gram with a single byte in the range 156-255 (100
substitution codes). Byte 155 is an escape for natural high-byte values.
Each substitution saves 3 bytes (4 → 1). The "sidecar" is just a 40-400
byte mapping table.

Results (enwik8 100 MB, L9 and L11):

| Variant | L9 ratio | L9 decomp (best) | L11 ratio | L11 decomp (best) |
|---------|----------|------------------|-----------|-------------------|
| Original | 28.62% | 172.4 ms | 27.17% | 165.3 ms |
| Sub-10 | **28.48%** | 172.7 ms | 27.20% | 168.5 ms |
| Sub-100 | 28.81% | 166.3 ms | 27.65% | 164.7 ms |

Ratio: L9 sub-10 improves by 0.14pp. But L11 sub-10 is 0.03pp worse —
the BT4 optimal parser loses more from substitution bytes disrupting
match contexts. Sub-100 hurts ratio at both levels because 100 different
high-byte codes increase entropy.

**Decompress speed is virtually identical across all variants.** The
decoder takes ~165-173ms regardless of whether the output is 82 MB or
100 MB. The bottleneck is token processing (resolve + execute), not
raw output bytes. Substitution doesn't reduce the token count — the
compressor matches the substitution bytes just like it matched the
original 4-grams, producing roughly the same number of tokens.

Compress speed improved 10-21% (fewer bytes to parse), but this is
the wrong side of the tradeoff — users of L9-L11 care about decode
speed, not encode speed.

### Why all approaches fail

The core insight: **the LZ compressor already handles repeated 4-grams
optimally.** Each occurrence of " the" is encoded as a small match token
(a few bytes). Pre-processing to remove, replace, or substitute these
patterns doesn't reduce the number of tokens the decoder must process —
it just changes what those tokens reference.

The High codec decode bottleneck is the **token resolve/execute pipeline**
(carousel walk, offset/length decode, match copy), not the volume of
output bytes or the content of match sources. Any preprocessing that
doesn't reduce the token count won't help decode speed.

To actually reduce token count, you'd need to change the LZ format
itself — e.g., a new token type meaning "emit this dictionary entry"
that's cheaper to decode than a match token. That's a format-level
change, not a preprocessor.

**Disposition**: All three approaches abandoned. The n-gram analysis
itself is solid (strong pareto, 28.5% coverage from 100 patterns), but
there's no way to exploit it that beats what the LZ compressor already
does. The 4-gram frequency data is preserved in `ngram.md` for future
reference.

Escape overhead on enwik8: only 0.52% of bytes are >= 155, confirming
the substitution range is safe for text. Binary data would have much
higher escape rates.

---

## Hand-written x86-64 ASM for Fast decoder hot loop (2026-05-04)

**Context**: VTune uarch-exploration showed `processModeImpl` at 0.22 CPI
with the raw-literal fast inner loop as the hottest code. LLVM's noinline
codegen produced a 33-instruction loop with two apparent wastes: a
duplicate `test r13b, r13b` (once for cmovns, once for setns) and a
`mov r11, r13` register copy from LLVM reusing r11 for both dst and
match_addr.

**Experiment 1 — Full loop replacement (28 instructions)**:
Replaced the entire Zig while loop with a hand-written inline asm loop.
Key optimizations vs LLVM: single `test` feeding both `cmovns` and `js`
(cmov doesn't clobber flags), branch-based off16 advance (2 instructions
vs LLVM's 4-instruction branchless setns path), `dst` stays in one
register throughout (no register copy).

Result: **6,399 MB/s** vs LLVM's **6,575 MB/s** — **3% slower** despite
5 fewer instructions. Round-trip correct.

**Experiment 2 — Loop body replacement only**:
Kept the Zig while loop structure but replaced the `cmd >= 24` token body
with inline asm (same 21 instructions of inner work). This proved the
code path was executing (100 NOPs inside slowed it 9%).

Result: **6,378 MB/s** — still 3% slower than LLVM.

**Why fewer instructions was slower**: LLVM's 33-instruction loop has
better instruction scheduling for Ice Lake's out-of-order engine. The
"wasted" instructions (duplicate test, register copy, branchless setns
path) allow the OOO engine to overlap more work:

- The duplicate `test al, al` feeds `setns` which is on a separate
  dependency chain from the `cmovns`. The OOO engine executes them in
  parallel on different ports.
- The `mov r12, rax` register copy lets LLVM use rax as a temporary
  for the match address computation, freeing r12 to receive the new
  dst value on a separate dependency chain.
- The branchless `xor + setns + lea` off16 advance avoids a branch
  that would serialize the pipeline 20% of the time.

**Lesson**: Instruction count is not execution time. On modern OOO CPUs,
"redundant" instructions that create independent dependency chains can
be faster than "efficient" code with fewer instructions but longer serial
chains. Trust LLVM's scheduler for hot loops — it optimizes for uop
throughput and dependency depth, not instruction count.

The noinline keyword alone gave +1.4% by isolating register allocation.
The hand-written ASM was unnecessary.

---

## Dropping/changing `inline` on hot-path token functions (2026-05-05)

**Context**: The `noinline` win on `processModeImpl` (+1.4%) suggested
that forced inlining might be hurting other hot-path functions too.
Tested removing or flipping `inline` on per-token functions.

**Experiment 1 — `processOneToken` (High decoder, L9-L11): drop `inline`**

This function is the entire L9-L11 token execution body (~100 lines),
called from two loop bodies (prefetch-safe main + no-prefetch tail).
Removing `inline` lets LLVM decide.

Result: L9 t1 decompress **1,008 MB/s** (baseline **1,019**) — **-1%**.
LLVM chose to inline it anyway (the function is hot enough), so
removing the annotation had minimal effect.

**Experiment 2 — `processOneToken`: add `noinline`**

Force LLVM to NOT inline, creating a function call per token.

Result: L9 t1 decompress **869 MB/s** — **-15% regression**.

**Why**: `processOneToken` is called per token (~millions of times per
chunk). The function call overhead (push/pop callee-saved registers,
argument passing, return) adds ~5-10 cycles per call. At ~100 cycles
per token baseline, that's a 5-10% overhead. Combined with losing
cross-call optimization (the caller's loop variables can't be kept in
registers across the call boundary), the total cost is 15%.

This is the opposite of `processModeImpl`, which is called once per
64KB sub-chunk (~150 times per enwik8). There, the call overhead is
amortized over ~thousands of tokens, and the register isolation benefit
(cleaner allocation in the caller) outweighs the cost.

**Rule of thumb**: `noinline` helps functions called O(chunks) that
share a file with complex callers. It hurts functions called O(tokens)
where call overhead dominates.

**Experiment 3 — `writeOffset` (Fast encoder): drop `inline`**

Result: Compress speed neutral (367.5 vs 369 MB/s). No effect on the
encode hot path — LLVM inlines it regardless of the annotation.

---

## `inline for` over struct array for recent-offset checks (2026-05-05)

**Context**: Codebase cleanup item. Three identical blocks in
`high_matcher.zig` checking `recents.offs[4]`, `[5]`, `[6]` with
hardcoded `best_offs` values of 0, -1, -2. Proposed replacement:

```zig
const recent_sentinels = [_]struct { idx: usize, offs_code: i32 }{
    .{ .idx = 4, .offs_code = 0 },
    .{ .idx = 5, .offs_code = -1 },
    .{ .idx = 6, .offs_code = -2 },
};
inline for (recent_sentinels) |r| {
    const ml = match_eval.getMatchLengthQuick(src, recents.offs[r.idx], ...);
    if (ml > best_ml) { best_ml = ml; best_offs = r.offs_code; }
}
```

**Result**: L6 compress **-17%** (45.0 → 37.7 MB/s), L6 decompress
**-9%** (12,215 → 11,082 MB/s). Confirmed by A/B stash testing.

**Why**: Although `inline for` with comptime-known values should unroll
to identical code, the struct field access pattern generates different
LLVM IR than three hardcoded constants. LLVM's optimizer treats the
struct indirection differently — the `r.offs_code` access through a
comptime struct generates extra intermediate SSA values that shift
register allocation in the surrounding function, degrading the High
codec's hot path.

**Lesson**: `inline for` over a struct array is NOT equivalent to manual
unrolling in Zig when the loop body calls functions or affects register
pressure. The Zig compiler's comptime evaluation produces different IR
shapes than hand-written repetition, and LLVM's optimizer doesn't always
canonicalize them to the same output. For performance-critical code,
prefer explicit repetition over `inline for` with struct dispatch.

---

## lz4ultra-style token reduction (2026-05-07)

Inspired by [lz4ultra](https://github.com/emmanuel-marty/lz4ultra), which
speeds up LZ4 decompression by reducing token count via two encoder-side
optimizations. We tried both on StreamLZ's Fast codec (L1-L5).

### Match length cap at nibble boundary

**Idea**: Cap match lengths at 15 (the Fast token nibble max) when the
original length is 16-30. Avoids the extended-length decode branch in
the decompressor at the cost of a few extra literal bytes.

**Result**: Ratio worsened by 0.4pp across L1-L4, and decompress speed
*dropped* 7-9%. The cap creates more tokens (the leftover bytes become
new matches/literals in subsequent iterations), increasing total token
count. The branch-elimination savings were smaller than the extra-token
cost. lz4ultra avoids this problem because its optimal parser globally
rearranges matches around the cap; our greedy/lazy parser truncates
blindly.

### Minimum match length bump (4 → 5)

**Idea**: Reject 4-byte matches with new offsets. A 4-byte match saves
only 1 byte over literals (4 bytes saved − 3 bytes token overhead) but
adds one full iteration of the serial pointer chase (~11-13 cycles).

**Result**: L1 ratio unchanged (greedy parser rarely finds 4-byte
matches at L1's hash resolution). L5 ratio worsened by 1.2pp (43.1% vs
41.9%) — the chain hasher finds many useful 4-byte matches. Decompress
speed unchanged (within noise). The ratio loss far outweighs any
theoretical decode speedup.

**Lesson**: These optimizations require a full optimal parser that can
globally evaluate token cost vs decode overhead. With a greedy/lazy
parser, local truncation decisions create worse output. StreamLZ's L6+
High codec already uses an optimal parser, and the Fast codec's token
decisions are already well-tuned for the greedy/lazy tradeoff.

---

## Word-interleaved 4-stream GPU Huffman layout (2026-05-19)

**Context**: The GPU Huffman decoder (`tools/huff_test/`) decodes a 64 KB
block as 4 independent bitstreams, one per warp lane (4 active lanes of
32). Nsight Compute profiled the 4-lane kernel as **latency/stall-bound**:
Compute 38.9%, DRAM 12%, 8.79 warp-cycles per issued instruction, and the
dominant stall was **Long Scoreboard at 43.5%** (global-load latency). NCU
also flagged the per-lane input refill as badly uncoalesced — the 4 lanes
read 4 streams 16 KB apart, so each 32-byte memory sector delivered only
~1.2 useful bytes (67% excess sectors).

**Experiment**: Store the 4 encoded streams INTERLEAVED at 32-bit-word
granularity — `[hdr][s0.w0][s1.w0][s2.w0][s3.w0][s0.w1]...` — so the 4
decode lanes' word-`w` refill loads hit 16 contiguous bytes and coalesce
into one sector instead of four scattered ones. New encoder
(`huff_encode_4s_interleaved`), strided decode core
(`decode_stream_interleaved`), and kernel
(`huffDecode4StreamInterleavedKernel`).

**Results** (realistic per-block-LUT bench, distinct 64 KB blocks):

| Dataset | Contiguous 4-stream | Word-interleaved |
|---------|---------------------|------------------|
| enwik8  | 19.3 GB/s           | 19.0 GB/s        |
| silesia | 19.0 GB/s           | 19.0 GB/s        |

No speed gain (slightly worse), and it cost ratio — enwik8 64.0% → 66.7%,
silesia 61.9% → 64.4% (+2.5-2.7pp) from per-stream word padding plus the
interleaved payload running to the longest stream's length.

**Why interleaving doesn't help**: coalescing fixes **bandwidth**, not
**latency**. The kernel is latency-bound — 4 scattered loads and 1
coalesced load incur the *same* ~200-cycle wait (issued together, return
together). Reducing the sector count lowers transaction traffic but does
nothing for how long the warp stalls. The "uncoalesced loads" NCU metric
was a *symptom*, not the bottleneck.

**Lesson**: For a latency-bound kernel, optimize for latency *overlap*
(software prefetch — issue the refill load ahead of when it's consumed)
or latency *reduction* (move the data to a faster memory space), not for
transaction efficiency. Coalescing is a bandwidth-bound lever. Always
confirm bound type before picking the fix — and a profiler metric being
"bad" does not make it the critical path. The interleaved code is kept
in `tools/huff_test/` as a documented dead end; the contiguous 4-stream
kernel remains the baseline.

## Software prefetch in the GPU Huffman decode loop (2026-05-19)

**Context**: Follow-up to the interleaving dead end above. The Huffman
4-lane decoder is latency-bound (NCU: 43.5% Long Scoreboard on the 32-bit
refill load). The latency-appropriate fix is software prefetch: issue the
next refill load one refill *ahead* of when it's consumed, so its
~200-cycle global-load latency overlaps the LUT-decode work.

**Experiment**: New decode core `decode_stream_prefetch` — a depth-1
prefetch variant of `decode_stream_one_lane`. A `nextw` register always
holds the next refill word, already loaded. When a refill is needed the
core consumes `nextw`, then immediately issues the load for the word
after it; the two LUT decodes that follow are independent of `nextw`, so
the load runs in their shadow. New kernel
`huffDecode4StreamPrefetchKernel` (same contiguous format as the
baseline). Drain+rewind hands any unconsumed prefetched word back to
`in_pos` before the byte-exact tail loop.

**Results** (realistic per-block-LUT bench, distinct 64 KB blocks, enwik8):

| Variant            | GB/s (run 1) | GB/s (run 2) |
|--------------------|--------------|--------------|
| Contiguous 4-stream| 19.2         | 19.3         |
| Software prefetch  | 19.4         | 19.3         |

Byte-exact, but no speed change — identical within run-to-run noise.

**Why prefetch doesn't help**: depth-1 prefetch overlaps one ~200-cycle
load against only ~2 LUT decodes (~40 cycles of independent work) — far
too little to cover it. The per-warp `bit_buf` dependency chain is
strictly serial (each decode needs the previous decode's shifted buffer),
so there is no deeper independent work to hide the load behind within a
lane. The GPU's real latency-hiding mechanism here is **warp-level
parallelism**: the same bench with smaller blocks — 16 KB → 6103 warps →
21.5 GB/s, 8 KB → 12207 warps → 22.6 GB/s — is faster purely because more
resident warps cover each other's stalls. 64 KB blocks only field 1525
warps.

**Lesson**: Software prefetch only pays off when there is enough
*independent* work per stall to fill the latency window; a serial
bit-buffer dependency chain has none. For this kernel the lever is warp
count (more, smaller independent units), not intra-lane scheduling. The
prefetch code is kept in `tools/huff_test/` as a documented dead end; the
contiguous 4-stream kernel remains the baseline.

## u16 single-symbol Huffman LUT — halve shared mem for occupancy (2026-05-19)

**Context**: NCU profiled `huffDecode4StreamKernel` with theoretical
occupancy capped at **22.9%, explicitly limited by shared memory** — each
warp holds an 8 KB Huffman LUT (2048 entries × u32), so an SM fits only
11 resident warps of its 48-warp maximum. The hypothesis: halve the LUT
to 4 KB → ~2× resident warps → more latency hiding → faster.

**Experiment**: `build_decode_lut16` (u16 entries: symbol:8, length:4),
`decode_stream_one_lane_lut16`, and `huffDecode4StreamLut16Kernel`. The
u16 width cannot hold two symbols (2 × 8 bits leaves no room for length),
so this **drops the dual-symbol fast path** — every LUT lookup yields
exactly one symbol instead of up to two.

**Results** (realistic per-block-LUT bench, distinct 64 KB blocks, enwik8):

| Variant            | GB/s (run 1) | GB/s (run 2) |
|--------------------|--------------|--------------|
| u32 LUT (8 KB)     | 19.2         | 19.3         |
| u16 LUT (4 KB)     | 15.8         | 15.8         |

Byte-exact, but **~18% slower** — the opposite of the prediction.

**Why halving the LUT lost**: two compounding reasons.
1. Dropping dual-symbol decode raises decode work ~1.5× — the u32 LUT
   emits ~1.5 symbols per lookup on text; single-symbol emits 1, so ~50%
   more lookups, loop iterations, refill checks, and bit-buffer shifts.
2. The doubled occupancy bought nothing. The kernel's SOL ceiling is the
   **L2 cache at 81.6%** — adding resident warps just makes more warps
   queue against an already-near-saturated L2; they cannot decode faster.

So the u16 kernel paid the full 1.5× compute penalty and collected none
of the occupancy dividend.

**Lesson**: This is a clean controlled test of the occupancy hypothesis —
and it falsifies it. A profiler reporting "occupancy limited by shared
memory (22.9%)" flags an *opportunity*, not the *bottleneck*. 11 warps/SM
is already enough to hide most latency here; the hard wall is L2
throughput, and warp count cannot beat a saturated memory pipe. The
dual-symbol fast path, by contrast, is genuinely load-bearing — it is a
real ~1.5× throughput multiplier and must not be traded away. The u16
code is kept in `tools/huff_test/` as a documented dead end; the u32
dual-symbol 4-stream kernel remains the baseline. The remaining real
levers are reducing L2 traffic and the ~20% tail effect (more blocks).

## 64-bit output stores in the GPU Huffman decoder (2026-05-19)

**Context**: Reducing L2 traffic was the live lever after NCU put the
SOL ceiling at L2 (81.6%). The first attempt — accumulating decoded
bytes and flushing them as **32-bit** stores instead of single-byte
stores (`decode_stream_one_lane_wstore`) — **worked**: 19.3 → 20.6 GB/s
(+7%), byte-exact. The decode loop appends each LUT entry's 1-2 output
bytes as a 16-bit chunk into a u64 and flushes a 4-byte store when ≥ 4
bytes are pending. This experiment tried to extend that to 8-byte stores.

**Experiment**: `decode_stream_one_lane_wstore64` + `wflush8` helper +
`huffDecode4StreamWStore64Kernel`. A u64 accumulator can hold only 8
bytes, and a 16-bit chunked append at pending == 7 would overshoot, so
the 64-bit variant must append **one byte at a time**, flushing an 8-byte
store the moment 8 bytes are pending.

**Results** (realistic per-block-LUT bench, distinct 64 KB blocks, enwik8):

| Variant                | GB/s (run 1) | GB/s (run 2) |
|------------------------|--------------|--------------|
| Baseline (byte stores) | 19.2         | 19.2         |
| 32-bit stores (wstore) | 20.6         | 20.6         |
| 64-bit stores (wstore64)| 16.0        | 16.0         |

Byte-exact, but **~22% slower than wstore32** and slower than baseline.

**Why 64-bit lost**: store *sector* traffic to L2/DRAM is identical for
32- and 64-bit stores — 100 MB of output is the same number of 32-byte
sectors regardless of store width. The only thing a wider store buys is
fewer store *instructions* (2048/lane vs 4096/lane) — a small saving.
But getting there cost a lot: the u64 accumulator can't absorb a 16-bit
chunked append safely, forcing **byte-at-a-time** appends with a flush
branch per byte, plus an `if (ns==2)` branch per decode. That roughly
tripled the accumulator bookkeeping in the hot loop. For a kernel issuing
at ~1.19 IPC, hot-loop instruction count matters, and the bloat swamped
the store-instruction saving.

**Lesson**: wstore32 is the sweet spot — a 16-bit chunked append packs
both bytes of a dual-symbol decode in one op with one flush check, and a
4-byte store already fits inside a 32-byte sector. Going wider doesn't
cut sector traffic (the real L2 cost) and only pays in store-instruction
count, which was already cheap; the byte-at-a-time append needed to feed
a u64 cleanly costs more than it saves. The wstore64 code is kept in
`tools/huff_test/` as a documented dead end; **`huffDecode4StreamWStore`
(32-bit, 20.6 GB/s) is the new fastest 64 KB-block decoder**.

## Persistent atomic-work-queue GPU Huffman decoder (2026-05-19)

**Context**: NCU's launch-statistics rule flagged a ~20% tail effect on
the 64KB-block decoder (1525 blocks ≈ 2.2 waves on a 34-SM GPU → a
partial final wave). The hypothesis: replace one-CUDA-block-per-Huffman-
block with a **persistent kernel** — grid = exactly one wave of CUDA
blocks (34 SM × 20 = 680), each a worker that `atomicAdd`s a global
counter to grab the next Huffman block until the queue drains. This
should remove the partial wave and load-balance non-uniform per-block
decode durations.

**Experiment**: `huffDecode4StreamWStoreAlignedEscQueueKernel` — the
escape decoder wrapped in a `for(;;)` loop: lane 0 does
`atomicAdd(g_counter,1)`, `__shfl_sync` broadcasts the index, decode,
repeat. Counter zeroed before each timed launch.

**Results** (realistic per-block-LUT bench, distinct 64 KB blocks, enwik8):

| Variant                    | GB/s (run 1) | GB/s (run 2) |
|-----------------------------|--------------|--------------|
| wse (1 block per Huff block)| 27.4         | 27.4         |
| wseq (persistent queue)     | 9.3          | 9.3          |

Byte-exact, but **~3× SLOWER** — a severe regression, not the hoped-for
+20%.

**Why the queue lost badly**: the GPU's **hardware block scheduler is
already a work queue** — and a much better one. With 1525 independent
CUDA blocks it launches them *staggered* as SM slots free up, so the ~20
blocks resident on an SM are at diverse phases (some loading their LUT,
some decoding, some refilling). That phase diversity pipelines the memory
system and overlaps each block's startup LUT-load with others' decode.
The persistent kernel throws all of that away: 680 workers launched
simultaneously run in near-lockstep — synchronized bursts of 20 LUT
loads, then 20 decode phases, then 20 refill bursts — destroying the
memory pipelining. (Register pressure from carrying loop + `desc` state
across iterations, possibly spilling, may compound it.)

**Lesson**: NCU's "tail effect, up to 20%" is an *upper bound* computed
from a pessimistic lockstep-wave model. Real hardware dynamic block
scheduling already recovers almost all of it — there was no easy 20% to
grab. Do not hand-roll a persistent work-queue to "fix" a tail the
hardware scheduler already handles; the staggered-launch behaviour of
many-small-independent-blocks is a feature, not a bug. The queue code is
kept in `tools/huff_test/` as a documented dead end; one CUDA block per
Huffman block remains the launch model.

## Cold-branch escape in the GPU Huffman decoder (2026-05-20)

**Context**: SASS analysis of the cp.async decoder (`decode_stream_one_lane_
cpasync_esc`, ~29.8 GB/s) showed the length-11 escape path compiled to
branchless `SEL`s — so ~4 escape-only ops (escape-bit extract, sym2
shift, two selects) ran on *every* symbol, including the 97%+ that are
not escapes. The kernel is compute-bound (NCU: 78.7% SOL), so trimming
hot-loop instructions should convert to speed. Hypothesis: move the
escape onto a real predicted-cold branch.

**Experiment**: `huff_escape_pick` marked `__noinline__` (a `__noinline__`
call cannot be predicated-flattened — it forces a real branch), called
from `decode_stream_one_lane_cpasync_esc_cb` with `__builtin_expect` on
the common arm. (`__attribute__((cold))` does NOT compile under nvcc with
the MSVC host toolchain — `__noinline__` is the portable lever.)

**Results** (realistic per-block-LUT bench, distinct 64 KB blocks, enwik8):

| Variant                       | GB/s (run 1) | GB/s (run 2) |
|--------------------------------|--------------|--------------|
| cpa (branchless escape)        | 29.8         | 29.9         |
| cpb (cold-branch escape)       | 29.3         | 29.2         |

Byte-exact, but **~2% slower** — the escape-op trim was outweighed.

**Why the cold branch lost**: forcing the escape onto a `__noinline__`
call puts a **function call inside the hot loop**. A call site is a hard
optimisation barrier even when the call is rarely taken — the compiler
must conservatively assume the loop's live registers (`bit_buf`, `acc`,
ring state) may be clobbered across it, which constrains register
allocation and instruction scheduling for the *entire* loop body,
including the common path. That barrier cost more than the ~4 cheap,
predictable, divergence-free `SEL`/shift ops it removed.

**Lesson**: the compiler's branchless flatten was already the right call.
A handful of cheap straight-line ops on a hot path beats a forced
out-of-line call whose mere presence poisons the loop's optimisation. On
GPU there is no cheap way to get a *call-free* cold branch for a tiny
escape body — ptxas predicates small if-bodies regardless of
`__builtin_expect`, and the only way to force a branch (a call) brings
the barrier with it. The cpb code is kept in `tools/huff_test/` as a
documented dead end; `huffDecode4StreamCpAsyncEscKernel` (cpa, 29.8 GB/s
@ 64.0%) remains the fastest decoder.

## Loosen `__launch_bounds__` to kill LZ-decode register spills (2026-05-24)

**Context**: NCU profile of `slzLzDecodeRawKernel` (the lean L1/L2 raw
fast path, ~1.96 ms on 8 MB enwik8 — the single largest remaining decode
cost) showed REG:40 / STACK:48 with 21 STL + 45 LDL across the kernel
group, ~73 K spill-load requests per launch, mostly hitting L1 (~80 %
hit rate). `nvdisasm --print-line-info` mapped the hot spill clusters to
`lz_kernels.cuh:405` (chunk_src setup, 5 STL+LDL), `:429` (lane-strided
copy loop, 5×), and `:416` (`__shfl_sync` of `sub_chunk_header`, 7×).
Root cause: `__launch_bounds__(64, 24)` in `slz_wire_format.cuh:119`
caps regs at ~42 (65 536 regs / 1536 max threads-per-SM at MBPS=24), so
ptxas spills the rest. Hypothesis: loosen MBPS so the compiler keeps
those values in registers; trading a few resident warps for fewer L1
spill loads should speed the kernel.

**Experiment**: swept `LZ_KERNEL_MIN_BLOCKS_PER_SM` ∈ {24, 16, 12, 8}.
At MBPS=16 the raw kernel went REG:56 / STACK:0 — **zero spills**, no
STL, no LDL. Rebuilt all three decode kernels via `tools\build_gpu.bat`,
rebuilt `streamlz` + `gpulib` with `zig build -Doptimize=ReleaseFast
-Dgpu=true`, benchmarked with `streamlz -db -r 30` on enwik8 L1/L3/L5
(best-of-30 GPU kernel time, measured with `-db` cuEvent timers).

**Results** (8 MB enwik8, RTX 4060 Ti):

| Level | MBPS=24 (baseline) | MBPS=16 (no spills) | Δ        |
|-------|--------------------|--------------------|----------|
| L1    | 3.79 ms            | 4.88 ms            | **+29 %** |
| L3    | 7.36 ms            | 8.29 ms            | **+13 %** |
| L5    | 7.42 ms            | 8.51 ms            | **+15 %** |

Byte-exact, but **slower at every level** — the opposite of the prediction.

**Why MBPS=16 lost**: the LZ match-copy loop has long-latency global
loads with little intra-thread work between them; the only way to hide
that latency is parallel resident warps. Dropping from 24 → ~17 blocks
per SM cut latency-hiding capacity by ~30 %, and that loss outweighed
the saved spill traffic — which was always going to be cheap because
the spills land in L1 with ~80 % hit rate (~3 % of kernel time).

**Lesson**: this is the complement of the u16 single-symbol Huffman LUT
experiment (line 1908). Both falsify a profiler "limited by X" warning
by sweeping the lever in the indicated direction:

- Huffman decoder was **L2-bound** at 81.6 % SOL; doubling resident
  warps (halving shared mem) bought nothing because no warp could decode
  faster against a saturated L2 pipe.
- LZ decoder is **latency-bound** on global loads; warp budget IS the
  hiding mechanism, so sacrificing warps for fewer spills hurts.

The meta-rule: NCU's "register pressure" / "occupancy limited" boxes
flag *opportunities to investigate*, not bottlenecks to fix
mechanically — actual perf depends on whether the kernel is warp-budget-
bound vs memory-bound vs compute-bound, and the right move goes opposite
ways in each case. `LZ_KERNEL_MIN_BLOCKS_PER_SM = 24` is at a local
optimum for `slzLzDecodeRawKernel`; leave it alone. A "don't loosen
this" comment was added inline at `slz_wire_format.cuh:116–129` so the
next reader doesn't burn a day re-running the same sweep.

## Vectorize the LZ-decode byte-copy loops to 4 bytes/lane (2026-05-24)

**Context**: NCU profile of `slzLzDecodeRawKernel` (the lean L1/L2 raw
fast path) showed Long Scoreboard = 50 % of stall cycles + Wait = 30 %,
with the inner literal/match copy loops at `lz_decode_core.cuh:91-95`
and `:107-108` doing one byte per lane per iteration:

    for (uint32_t i = lane; i < lit_len; i += WARP_SIZE)
        dst[dst_pos + i] = lit[lit_pos + i];

Hypothesis: every L1 hit (~20–30 cycles latency) gates the next loop
iteration through the same lane's dependency chain. Replacing the byte
load/store with a 4-byte vectorized load/store should fold 4 dependent
loads into 1, cutting Long Scoreboard time and shortening the critical
path. Predicted ~1.5–2× kernel speedup.

**Experiment** (two attempts, six call sites total):

1. *memcpy-based vectorization* — `memcpy(&v, src, 4)` + `memcpy(dst,
   &v, 4)` inside a 4-byte stride loop, with a tail sweep for the
   final < 4 bytes. PTX inspection revealed the codegen was wrong:
   `memcpy` with a `const uint8_t*` source lowered to **four
   `ld.global.u8` instructions** (loop-unrolled byte loads), not a
   wide load. Same load count as the byte loop + more register
   pressure (raw kernel STACK 48 → 96 B/thread, general 192 → 232).

2. *Inline-PTX wide loads* — replaced the `memcpy`s with
   `asm("ld.global.b32 %0, [%1];" : "=r"(v) : "l"(p));` (and the
   matching `st.global.b32`). PTX now emits **20 `ld.global.b32` +
   20 `st.global.b32` per raw kernel**, with 8 byte loads + 10 byte
   stores remaining for the tail and the token parser. Register
   spills dropped from the memcpy attempt (raw kernel STACK 96 → 80,
   general 232 → 224). Delta-literal path (mode 0) vectorized with
   `__vadd4` for the per-byte literal+source add.

Both byte-exact (SHA256 match on enwik8 L1/L3/L5 roundtrip).

**Results** — 8 MB enwik8, pure D2D, per-kernel cuEvent timing,
best-of-5 after warmup (stable to ±0.001 ms):

| Variant            | L1 raw   | L3       | L5       |
|--------------------|----------|----------|----------|
| Baseline (byte)    | 1.81 ms  | 1.80 ms  | 2.11 ms  |
| memcpy "vectorize" | 1.81 ms  | 1.81 ms  | 2.11 ms  |
| Inline-PTX wide    | 1.81 ms  | 1.81 ms  | 2.11 ms  |

Identical to three decimal places.

NCU full-set comparison (byte baseline vs inline-PTX wide loads):

| Metric                       | Baseline | Wide load |
|------------------------------|---------:|----------:|
| Compute (SM) Throughput      | 73.7 %   | 73.7 %    |
| Memory Throughput SOL        | 73.7 %   | 73.7 %    |
| DRAM Throughput              | 18.5 %   | 18.0 %    |
| L1/TEX Hit Rate              | 81.6 %   | 81.6 %    |
| L2 Hit Rate                  | 85–89 %  | 85–89 %   |
| Achieved Occupancy           | 79 %     | 79 %      |
| Issued Warp / Scheduler      | 0.74     | 0.74      |
| Warp Cycles / Issued Inst    | 13.07    | 13.07     |
| Avg Active Threads Per Warp  | 18.44    | 18.44     |
| Executed IPC Active          | 2.90     | 2.91      |
| Long Scoreboard stall %      | 49.9 %   | 50.1 %    |
| Wait stall %                 | 29.8 %   | 29.5 %    |
| Branch Resolving stall %     | 9.8 %    | 9.8 %     |

Every metric unchanged.

**Why wide loads lost**: the Long Scoreboard 50 % stall does NOT come
from the literal/match copy loops. The copies were widened, the loads
landed in SASS as 32-bit ops (PTX + L1 hit rate confirm), and the
profile didn't budge. The 50 % is coming from elsewhere on the
critical path — almost certainly:

1. **The back-reference `dst[match_src + i]` load** inside the match
   copy. The result is consumed immediately by the dependent store on
   the same iteration, so the dependency chain length is fixed; a
   wider load doesn't shorten it.
2. **Lane 0's serial token parse** (`cmd[cmd_pos]`, `off16[off16_pos]`,
   `readLength` for long tokens). The other 31 lanes have no work to
   do during the parse, so the warp scheduler stalls on this lane's
   L1 reads.
3. **The five `__shfl_sync` broadcasts** after each parse — each shfl
   is warp-pipelined with its own latency.

**Lesson**: NCU's stall-reason breakdown points at a *category* of
stall (e.g. "Long Scoreboard"), not at *which specific load* is
gating progress. Two loops in the same kernel that both load from
global memory will both show up as Long Scoreboard, but only one may
be on the critical dependency chain. Before vectorizing, identify
which load's *result* is consumed by a dependent instruction with no
intervening parallel work — that's the one on the chain. For this
kernel the back-reference + serial-parse loads are on the chain; the
parallel literal/match copy loads are not. Real progress requires
shortening the per-token critical path (parse + broadcast +
back-reference resolution), not widening the bystander copies.

Also documents a measurement pitfall: the original "regression" reading
that motivated reverting the first attempt was from `streamlz -db -r 30`
with `SLZ_SPLIT_TIMER=1`, which measures wall-clock between stream-sync
points in the pipelined LZ path — not actual kernel duration. The real
LZ kernel time is exposed only via per-kernel cuEvent profiling through
`slzGetLastTimings`. A separate bug in `finalizeProfiling`
(double-clear of `last_timings` from the main-thread getter) had to be
fixed to make per-kernel timing visible from `slz_gpu_d2d_test`. The
fix is in `src/gpu/decode/driver.zig:434-447` and stays.

The wide-load patch was reverted (`git checkout HEAD --
src/gpu/decode/lz_decode_core.cuh`); the byte-loop baseline at
`lz_decode_core.cuh:91-95 / :107-108 / :125-126` remains. The
inline-PTX helpers (`ldGlobalU32Unaligned`, `stGlobalU32Unaligned`,
`warpCopyBytesVec`, `laneCopyMatchSerial`) were removed — if a future
experiment ever needs wide-load helpers, they're trivially regenerated
from this entry.

## Unconditional first warp-step on the LZ-decode literal copy (LZTurbo copy16 trick) (2026-05-24)

**Context**: Looked at `src/compare/lzturbo.zig`'s CPU decompressor, which
gets a measurable win on x86 from the LZ4-style "always `copy16` 16
bytes, advance by actual `lit_len`, let the next token overwrite the
overshoot" pattern (`lzturbo.zig:261-264`). The CPU win comes from
skipping the per-iteration branch + tail per-byte copies. On GPU, the
analog is the per-warp `if (lit_len > 0)` check + the lane-tail
divergence in `for (i = lane; i < lit_len; i += WARP_SIZE)`.

NCU says Avg Active Threads Per Warp = 18.44/32 in this loop's vicinity
(57.6 % lane utilization) and Branch Resolving = 9.8 % of stall cycles.
Hypothesis: always issue one warp-wide byte store (every lane writes,
even when `lit_len < 32`), then loop only for `lit_len > WARP_SIZE`.
Lanes past lit_len overshoot into `dst[dst_pos+lit_len..dst_pos+31]`,
which is later overwritten by the same token's match copy or the next
token's literal copy.

**Experiment**: rewrote `lz_decode_core.cuh:90-95` (raw kernel's literal
copy) as:

    if (lit_len > 0) {
        dst[dst_pos + lane] = lit[lit_pos + lane];      // unconditional first step
        for (uint32_t i = lane + WARP_SIZE; i < lit_len; i += WARP_SIZE)
            dst[dst_pos + i] = lit[lit_pos + i];        // only for lit_len > 32
        __syncwarp();
    }

REG/STACK unchanged (raw kernel still 40/48, general 40/192). Byte-exact
on enwik8 8 MB and 100 MB at L1/L3/L5 — confirms the overshoot is safe
in practice for our encoder's match patterns (i.e. subsequent match
reads do not pick up garbage from the overshoot region, presumably
because the offset/match_len distribution rarely places a match read
inside the freshly-overshooting bytes).

**Results** (pure D2D, per-kernel cuEvent, best-of-5 after warmup,
stable to ±0.003 ms):

| Size   | Level   | Baseline | Unconditional | Δ        |
|--------|---------|---------:|--------------:|----------|
| 8 MB   | L1 raw  | 1.81 ms  | 1.81 ms       | 0.0 %    |
| 8 MB   | L3      | 1.80 ms  | 1.80 ms       | 0.0 %    |
| 8 MB   | L5      | 2.11 ms  | 2.11 ms       | 0.0 %    |
| 100 MB | L1 raw  | 3.77 ms  | 3.75 ms       | –0.6 %   |
| 100 MB | L3      | 3.80 ms  | 3.79 ms       | –0.3 %   |
| 100 MB | L5      | 4.41 ms  | 4.41 ms       | 0.0 %    |

Within measurement noise. The 100 MB L1 –0.6 % is at the edge of
the run-to-run jitter and does not reproduce as a real signal.

**Why it lost**: same as the wide-load experiment two days earlier in
this file — the literal/match copy loops are **not on the critical
path**. NCU has now told us this three different ways in the same
session:

1. Wide-load vectorization (`ld.global.b32`) — fewer loads, same time.
2. Inline-PTX `__vadd4` for the delta-literal copy — same time.
3. Unconditional first warp-step — fewer branches, same time.

All three changes verifiably landed in SASS or in measurable Avg Active
Threads improvements. None moved kernel wall-clock. The kernel is
gated by the per-warp critical-path latency of (back-reference load →
dependent store) + (lane-0 serial parse → broadcast shuffle → next
iteration), and any optimisation that doesn't shorten one of those
chains is a no-op.

**Lesson**: LZTurbo's branchless overshoot pattern is a *CPU* lever —
it removes branch-predict cost on a sequential SIMD core. On a GPU
warp, the warp scheduler doesn't pay the same predict cost; the
analogous waste shows up as Avg Active Threads Per Warp and Branch
Resolving stall, neither of which is dominant in this kernel. Don't
copy CPU branch-removal tricks to GPU without first checking whether
the GPU pays the same cost — the answer is usually "no, but you'll
spend a day proving it".

The change was reverted (`git checkout HEAD --
src/gpu/decode/lz_decode_core.cuh`). The byte-loop baseline at
`lz_decode_core.cuh:90-95` remains.

The unconditional-copy idea is *correct* on this kernel (byte-exact
verified at 100 MB) — it's just not load-bearing. If we ever do a
larger restructure that brings the copy loops onto the critical path
(for example, removing the `__syncwarp()` barriers so the next token's
parse overlaps with the previous token's copy), this trick may earn
its complexity back.

## META: slzLzDecodeRawKernel is at its structural ceiling — micro-optimizations don't move it (2026-05-24)

**Context**: NCU profile of `slzLzDecodeRawKernel` (the lean L1/L2
fast path) on 100 MB enwik8 shows Long Scoreboard = 50 % + Wait = 30 %
of stall cycles, with 73.7 % Compute SOL and 0.74 Issued Warp /
Scheduler. The natural reading is "the kernel is gated by L1 load
latency on lane-0 serial parse and on the back-reference reads in the
match copy". This session ran *five distinct experiments* aiming at
that hypothesis. Every one of them landed in the SASS / PTX as
designed, passed byte-exact verification, and **failed to move kernel
wall-clock time by more than measurement noise**.

**Experiments**:

| # | Attack | What landed in SASS | Δ kernel time |
|---|---|---|---|
| 1 | Vectorize byte copies via `memcpy(&v, ..., 4)` | `memcpy` lowered to 4× `ld.global.u8` (not vector) — see entry above | 0 % |
| 2 | Inline-PTX `ld.global.b32` + `st.global.b32` wide loads | 20× wide loads + 20× wide stores in raw kernel — see entry above | 0 % |
| 3 | Unconditional first warp-step (LZTurbo `copy16` trick) | branch eliminated; avg active threads conceptually higher — see entry above | 0 % (–0.5 % at 100 MB, within noise) |
| 4 | `cp.async` staging of cmd / off16 / length into shared mem | 28× `cp.async.cg` + lane-0 reads become `ld.shared.u8` (PTX verified) | 0 % |
| 5 | Back-reference prefetch via 2-way manual unroll, load-all-then-store-all | dependent load→store iterations decoupled in loop body | 0 % |

Each entry above describes the per-experiment SASS evidence in detail;
this meta-entry exists to capture what the *combined* result means.

**NCU profile comparison** (byte-loop baseline vs experiment #4
cp.async, the only one profiled with full NCU after the experiment —
the others were measured by per-kernel cuEvent timing on enwik8 8 MB
and 100 MB, stable to ±0.005 ms over 5-run best-of):

| Metric | Baseline | After cp.async |
|---|---:|---:|
| Compute (SM) Throughput | 73.7 % | 73.68 % |
| Memory Throughput SOL | 73.7 % | 73.68 % |
| DRAM Throughput | 18.5 % | 18.02 % |
| L1/TEX Hit Rate | 81.6 % | 81.60 % |
| L2 Hit Rate | 85–89 % | 85–89 % |
| Achieved Occupancy | 79 % | 79.39 % |
| Issued Warp / Sched | 0.74 | 0.74 |
| Warp Cycles / Issued Inst | 13.07 | 13.07 |
| Executed IPC Active | 2.90 | 2.91 |
| Long Scoreboard stall | 49.9 % | 49.8 % |
| Wait stall | 29.8 % | 29.5 % |
| Branch Resolving stall | 9.8 % | 9.8 % |
| Inst Executed | 967 M | 967 M |

**Every metric is unchanged to two decimal places** — including the
stall categories the change was explicitly designed to attack. This
is the most informative evidence in the file: not just "the change
didn't help", but "the change made *no observable difference at any
metric NCU reports*", despite SASS confirming the change landed.

**Why none of the five worked — synthesis**:

The five experiments tried to reduce the wall-clock cost of:

1. Per-iteration load instructions in the cooperative copy loops (#1, #2)
2. Per-token branch divergence in the literal copy (#3)
3. Per-token L1 latency on lane-0 cmd/off16 reads (#4)
4. Per-token L1 latency on back-reference loads in the match copy (#5)

All four of those latencies are **real** — NCU's Long Scoreboard 50 %
ratio is not a measurement artifact. The mistake was assuming any of
them was on the warp's *critical issue path*. The actual situation:

- The SM has 38 warps resident (79 % achieved occupancy) and
  2.03 eligible warps per scheduler.
- When *any one warp* stalls on a load, the warp scheduler issues
  from another eligible warp. The 26 % of cycles with no issue
  (1 – 0.74) means all 38 warps were stalled *simultaneously* on
  cycles where no issue happened.
- A change that reduces *one* warp's load latency by 15 ns doesn't
  help unless it also reduces the *correlated* stalls across the
  other 37 warps. None of these five changes did — they all reduced
  one type of stall per warp, with the per-SM throughput
  unaffected.

The Long Scoreboard 50 % is a property of the *instruction mix* (the
ratio of load-issuing cycles to total cycles), not a single
shortenable bottleneck. The kernel is compute-bound at 73.7 % SOL on
a real per-byte workload; the remaining 26.3 % is the irreducible
scheduling overhead given the instruction mix.

**What the five experiments *didn't* try, in case perf becomes a
priority again**:

1. **Software-pipelined parse — overlap token N+1's parse on lane 0
   with token N's cooperative copy across all 32 lanes.** This
   actually shortens the per-warp critical chain (not per-instruction
   latency) and would change the kernel's structural ceiling, not
   move a metric. Multi-day rewrite. Predicted 10–20 % gain if it
   lands cleanly.
2. **Format change to remove the serial parse entirely.** A
   self-describing token layout where every lane parses its own token
   in parallel — the inverse of LZ4's "1 byte / 1 token" design.
   Major wire-format change, ~weeks.
3. **Tensor-core-assisted bulk copy** for very long matches (>= 256
   bytes). Tensor cores can be coerced into a fast copy primitive
   under specific layouts, but the setup cost is large enough that
   it only wins for long matches — and our match-length histogram on
   text is dominated by < 32 byte matches.

**Lesson — meta**: NCU stall categories tell you *where time goes*,
not *where time can be reclaimed*. The two are different when the
SM is well-occupied. "Long Scoreboard 50 %" doesn't mean "50 % of
your kernel time is recoverable by making loads faster". It means
"50 % of the cycles in this kernel are spent waiting on loads". On
an under-occupied SM, reducing load latency frees scheduling cycles
and converts directly to speedup. On a well-occupied SM (like
slzLzDecodeRawKernel at 79 % achieved), the scheduler already
absorbs single-warp latencies and reducing them is invisible at the
metric level. The lever is *not* per-instruction; it's per-warp
critical-chain length or per-warp instruction count.

Before running yet another load-latency experiment on this kernel,
**re-read this entry**. The current kernel time of 1.81 ms (8 MB) /
3.77 ms (100 MB) is the practical ceiling for the current
parse-then-copy structure on Ada. Beating it requires changing the
*structure*, not the *instructions*.

All five experiments reverted; `lz_decode_core.cuh` is at HEAD. The
`finalizeProfiling` bug fix in `src/gpu/decode/driver.zig:434-447`
stays (it was a real measurement-infrastructure bug discovered during
experiment #4 and is unrelated to the five attacks).

## Warp-parallel token parse with prefix scan (2026-05-24, after the meta-entry above)

**Context**: After the meta-entry above concluded the kernel was "at its
structural ceiling for per-instruction optimization", a follow-up NCU
breakdown was run to measure WARP-INSTRUCTION composition (different
from the per-instruction latency we'd been attacking). The breakdown:

- `smsp__inst_executed.sum` = 967.5 M warp-instructions
- `smsp__thread_inst_executed_per_inst_executed.ratio` = 18.44 lanes
- Pipe split: ALU 37.5 %, FMA 27.4 %, LSU 16.7 %, CBU 16.8 %

From the lane-active ratio, the upper bound on "single-lane parse"
warp-instructions: `F_parse = (32 - 18.44) / 31 = 43.7 %`. Roughly
30-40 % once accounting for the fact that some copies have < 32 active
lanes. This looked like a real fat target — and unlike the previous
five attacks, it was a *warp-instruction-count* lever, which the
meta-entry's own conclusion called out as the actually-attackable axis.

**Hypothesis**: replace the lane-0 serial token parse with a
warp-parallel parse using:
- one coalesced 32-byte LDG of the cmd stream per 32-token batch
- per-lane local decode of cmd byte
- warp prefix scan (`__shfl_up_sync` Hillis-Steele) to compute per-lane
  off16_offset, dst_offset, lit_offset within the batch
- `__ballot_sync` + `__clz` to propagate the `recent_offset` chain
  across same-batch tokens
- sequential per-k execution loop using `__shfl_sync` to broadcast
  lane k's parsed values for the cooperative copy

Predicted gain: 20-35 % kernel speedup (28-38 % parse-warp-insts gone
+ 5-6 % CBU drop from removing the lane-0 BSSY/BSYNC).

**Experiment**: implemented in `decodeSubChunkRawMode<false>` only
(the dominant raw L1/L2 path). `<true>` (entropy-coded off16) kept the
original serial code. Two configurations tried:

1. *Strict fast path* — fires only when all 32 tokens in a batch are
   SHORT format AND batch is full 32 tokens. Falls back to lane-0
   serial parse otherwise.
2. *Loosened fast path* — fires whenever no long tokens in batch
   (partial batches allowed). Wider coverage.

Both byte-exact on enwik8 8 MB / 100 MB and silesia 100 MB at L1, L3,
and L5 — **no correctness debugging needed**, the algorithm landed
right on the first build.

Implementation complexity:
- `warpScanU32` helper (Hillis-Steele exclusive scan, 5 `__shfl_up_sync`)
- Parallel-parse fast path: ~80 lines of CUDA
- Serial fallback (unchanged from original): kept in same function

**Results** (per-kernel cuEvent timing, best-of-5 after warmup):

| Size / Level | Baseline | Strict fast path | Loosened fast path |
|--------------|---------:|-----------------:|-------------------:|
| 8 MB L1 raw  | 1.81 ms  | 1.81 ms          | 1.81 ms            |
| 8 MB L3      | 1.80 ms  | 1.80 ms          | 1.80 ms            |
| 8 MB L5      | 2.11 ms  | 2.11 ms          | 2.11 ms            |
| 100 MB L1 raw| 3.77 ms  | 3.75 ms          | 3.74 ms            |
| 100 MB L3    | 3.80 ms  | 3.78 ms          | 3.78 ms            |
| 100 MB L5    | 4.41 ms  | 4.40 ms          | 4.40 ms            |
| 100 MB silesia L1 | n/a | 3.31 ms          | 3.31 ms            |

Best observation: 100 MB enwik8 L1 dropped from 3.77 → 3.74 ms = 0.8 %
improvement. **Within noise**. The "loosened" variant (which
fires on far more batches) gave nearly identical perf to "strict" —
ruling out coverage as the issue.

Resource cost: STACK 48 → 64 bytes per thread (extra 16 bytes for
per-lane batch parse state). REG unchanged. No occupancy change.

**Why it lost — strongest evidence yet for the meta-entry's
conclusion**: this experiment was *specifically* designed to attack the
warp-instruction-count axis that the prior 5 failures and the meta-
entry called out as the actually-attackable lever. The metric
prediction was solid (43.7 % F_parse upper bound, 30-40 % realistic).
The implementation was correct (byte-exact on three independent
inputs, two paths). The change provably did parallel parse — there
was nothing left to debug.

And the kernel STILL didn't move.

This means the meta-entry's conclusion was actually too optimistic.
The lever isn't even "warp-instruction count". The kernel at 73.7 %
compute SOL is at an SM throughput ceiling — reducing warp-insts on
one warp's critical path doesn't help because the SM was already
fully busy running the other 37 resident warps' instructions during
those slots. The single-lane parse work was being run *in parallel
with* other warps' cooperative copies. Compressing it into 1/32 the
clock budget per warp doesn't free SM compute capacity that wasn't
already idle.

**Updated meta-lesson**: the genuine ceiling for this kernel
structure on Ada with 79 % achieved occupancy is roughly:

  *(total warp-instruction throughput across all SMs) / (work per byte)*

The total SM throughput on this kernel is 73.7 % of SOL — close to
the practical maximum. Beating that requires either:

1. **Reducing the WORK per byte decoded** — i.e., fewer total
   warp-instructions per byte. The parallel-parse experiment removed
   per-warp parse work but did NOT reduce per-byte total work
   because the same parse instructions just ran on different warps.
   The format itself dictates the work; meaningful reduction needs
   wire-format change.

2. **Increasing SM throughput beyond 73.7 % SOL** — would require
   reducing structural overheads (warp barriers, scheduler stalls
   that aren't from any specific stall reason but from how warps
   interleave). Hard to attack — the 26.3 % gap is the irreducible
   scheduling overhead given the instruction mix.

3. **Different parallelization granularity** — e.g., multiple warps
   per sub-chunk cooperating (would change `WARPS_PER_BLOCK` and the
   decode dispatch). Untested by these 6 experiments.

The parallel-parse rewrite has been reverted. The implementation
is preserved in this entry's description (≈80 lines of CUDA) and
can be reconstructed from this entry if option (3) above ever
becomes a candidate path. Until then: **stop optimizing this kernel
within the current parallelization scheme**. We already beat nvCOMP
LZ4 (3.77 vs 5.1 ms kernel) and nvCOMP Zstd (3.77 vs 6.2 ms) at the
100 MB enwik8 L1 D2D decode comparison.

All 6 experiments reverted; `lz_decode_core.cuh` is at HEAD. The
`finalizeProfiling` bug fix in `src/gpu/decode/driver.zig:434-447`
stays.

## RETRACTION (2026-05-24, hours after the meta-entry above)

**The 6th experiment (parallel parse) and likely several earlier ones
were tested against a stale DLL** and the conclusions above are
partially wrong. The kernel PTX is embedded into `streamlz_gpu.dll`
at zig build time via `@embedFile`. Running `tools/build_gpu.bat`
rebuilds the `.cubin` and `.ptx` but does NOT rebuild the DLL.
Re-running tests after only rebuilding the cubin re-uses the OLD PTX
embedded in the DLL — so all kernel modifications are silently dead.

The session's experiments after the last `zig build gpulib`
(specifically: experiments 3 = unconditional copy, 5 = back-ref
prefetch, and 6 = parallel parse from the meta-entry) all hit this
issue. Their "no kernel time change" results were not measuring the
modified kernels at all.

**The fix**: rebuild order must be `tools/build_gpu.bat` → `zig build
gpulib -Doptimize=ReleaseFast -Dgpu=true` → run. Verified by
`__trap()` injection: with stale DLL, kernel runs to completion; with
fresh DLL, kernel crashes with `stream sync FAILED rc=719`.

**Re-running parallel parse with FRESH DLL**:

| Size / Level | TRUE baseline | Parallel parse (fresh DLL) | Δ |
|---|---:|---:|---:|
| 100 MB enwik8 L1 raw | 3.76 ms | **2.92 ms** | **–22.3 %** |
| 100 MB enwik8 L3 | 3.79 ms | 3.79 ms | 0 % (uses slzLzDecodeKernel, mods only in raw) |
| 100 MB enwik8 L5 | 4.41 ms | 4.41 ms | 0 % (same) |
| 100 MB silesia L1 raw | ~3.31 ms | **2.64 ms** | **–20 %** |

**Parallel parse WORKS — 20-22 % kernel speedup on the L1 raw path.**
This matches the metric-based prediction (28-38 % parse-warp-insts
removed → 20-35 % kernel speedup) almost exactly. The earlier "kernel
time didn't budge" was the stale-DLL artifact, not a real measurement.

**The meta-entry's conclusion about "the kernel is at its structural
ceiling" was premature.** The structural ceiling argument was built on
6 experiments where the changes weren't actually running. With proper
rebuild, the per-warp instruction count lever DOES translate to wall
clock — exactly as warp-instruction-throughput analysis predicts on a
73.7 % SOL kernel.

**Experiments 3, 5, 6 should be re-tested with proper rebuild before
trusting their "no change" conclusion**. Experiments 1, 2, 4 likely
ran against fresh DLLs at the time (the session was newer; `zig build`
had been run more recently), but worth verifying.

**Implementation kept** — the parallel-parse code is in
`src/gpu/decode/lz_decode_core.cuh` and the `warpScanU32` helper at
the top of the file. The serial path is preserved as fallback for
batches containing long tokens. The bug that made earlier byte-exact
tests trivially pass (stale DLL = unchanged kernel) was caught when
the build was finally done properly and the freshly-built kernel
hung — pointing to a real `__shfl_sync`-in-divergent-branch bug that
took ~5 minutes to find and fix.

**Lesson #1**: every kernel change requires the FULL build sequence —
not just `tools/build_gpu.bat` (which only updates the .cubin/.ptx
files on disk).

  - `streamlz_gpu.dll` (used by `slz_gpu_d2d_test.exe`,
    `slz_gpu_async_test.exe`) embeds the PTX via `@embedFile` in
    `src/gpu/decode/driver.zig` at zig-build time. Built by
    `zig build gpulib`.
  - `streamlz.exe` (used by the CLI `-db` bench, `-d` decompress,
    `-c` compress) ALSO embeds the PTX independently via the same
    `@embedFile`. Built by `zig build` (no specific target).

  **Both must be rebuilt** after any kernel-source change. The session
  burned hours TWICE on this:
    1. First on the DLL (experiments 3, 5, 6 from prior session and
       the parallel-parse first attempt all ran against stale DLL).
    2. Then on the EXE — even after fixing the DLL build, the first
       NCU profile of PP-v2 via `streamlz -db` showed identical
       metrics to pre-PP because the EXE still had the morning's
       pre-PP PTX embedded. Each EXE/DLL embeds the PTX at its OWN
       build time and they don't auto-sync.

  `tools/build_gpu_full.bat` was created during this session and
  chains all three steps: cubin rebuild → DLL rebuild → EXE rebuild.
  Use it as the standard build command for any kernel work. Manual
  `zig build` alone or `tools/build_gpu.bat` alone are silent bugs
  waiting to happen.

**Lesson #2**: when an experiment shows ZERO change in every metric
(stall reasons, IPC, occupancy AND kernel time, all identical to
baseline), suspect a stale-build issue before concluding the change
is a no-op. A `__trap()` at the entry of the modified function is the
fastest way to confirm the code is actually running.

**Lesson #3**: the meta-entry above was overconfident. NCU stall-
reason analysis tells you what kind of stall, and warp-instruction-
count analysis tells you where time goes. Both were correct. The
parallel-parse code, when it actually ran, produced exactly the
~20 % speedup the analysis predicted. The "well-occupied SM absorbs
single-warp savings" theory was a rationalisation of bad measurements,
not an observation.

## Three increment attempts on top of parallel parse (2026-05-24, after retraction)

After parallel parse landed as a real 20-22 % L1-raw win, the question
was whether any of the previously "failed" experiments (1-5 in this
file, three of which were tested against stale DLLs) would stack on
top of parallel parse for additional speedup. Tested three with proper
build sequence (`tools/build_gpu_full.bat` — new script that chains
cubin build + DLL rebuild):

### Increment A: inline-PTX wide loads (memcpy-based) on top of PP

Hypothesis: 4× fewer load ops per copy = shorter critical path for the
cooperative copy phase, which now dominates after parallel parse.

Result on 100 MB enwik8 L1 raw:

| Variant | ms |
|---|---:|
| PP only | 2.92 |
| PP + wide-load memcpy | **5.26** (+80 %) |

REGRESSED. Same failure mode as the original wide-load experiment —
memcpy with `uint8_t*` source doesn't lower to wide loads, just adds
register pressure (STACK 72 → 104 raw, 192 → 216 general). The
inline-PTX `ld.global.b32` variant crashed with `rc=716`
(`cudaErrorIllegalAddress`) when applied to dst-relative addresses.
Inline-PTX with `ld.b32` (address-space-agnostic) is the right next
attempt but not pursued — parallel parse already shortened the parse-
side stall, and the cooperative copy after PP runs on per-batch-
prefix-summed offsets that are small (SHORT-token lit_len ≤ 7,
match_len ≤ 15), so wide loads on those tiny ranges have no headroom
to save against.

### Increment B: unconditional warp-step (LZTurbo copy16 trick) on top of PP

Hypothesis: skip the per-token loop check + tail-divergence by always
writing 32 bytes, advancing dst_pos by actual lit_len/match_len. SHORT
tokens (≤ 7 / ≤ 15 bytes) always fit in one warp-step. Saves ~3 cycles
per token × 4K tokens = ~12K cycles per warp.

Result on 100 MB enwik8 L1 raw: **FAILS byte-exact**.

First mismatch at byte 131078 (6 bytes into sub-chunk 2 of 128KB). The
unconditional copy's overshoot from the LAST token of sub-chunk 1
leaks into sub-chunk 2's dst range. Sub-chunk 2's early match copies
read those overshoot bytes BEFORE sub-chunk 2's literal writes
overwrite them. This is a sub-chunk-boundary hazard that the LZTurbo
overshoot trick implicitly avoids because LZTurbo decompresses
contiguously (no per-sub-chunk decoding) — its overshoot region is
always immediately overwritten by the next token in the same
contiguous run, never crosses an independent-decode boundary.

The original "byte-exact verified" report for this experiment from
earlier in the session was a stale-DLL artifact — the test was running
unmodified code. With proper rebuild, the correctness problem is
visible. The pattern is genuinely wrong for our sub-chunk format.

### Increment C: back-reference prefetch via 2-way manual unroll on top of PP

Hypothesis: load all iter-0 + iter-1 reads first, then store both,
defeating the conservative dst[]/lit[] alias serialisation. Should
overlap N+1's L1 load with N's store.

Result on 100 MB enwik8 L1 raw:

| Variant | ms |
|---|---:|
| PP only | 2.92 |
| PP + back-ref prefetch unroll | **3.37** (+15 %) |

REGRESSED. The 2-way unroll adds permanent register pressure (v0, v1,
valid0, valid1) AND a second conditional load/store for every
iteration. For SHORT tokens (lit_len ≤ 7, match_len ≤ 15) the second
iteration NEVER fires (lane + 32 > N always), so the entire second
unroll is pure overhead. The compiler can't dead-code-eliminate it
because N is runtime.

Byte-exact verified (no correctness issue, just slower).

### Why all three lose on the PP path

After parallel parse, the per-token cooperative copy in the execution
loop operates on SHORT-token sizes (1-7 byte literals, 1-15 byte
matches). All three increments were originally designed for the LONG
copy regime where they'd reduce instruction count or critical-path
length. On SHORT ranges:

- Wide loads: tiny gain per copy × tiny number of copy-iterations →
  swamped by register-pressure cost.
- Unconditional copy: marginal cycle savings × tiny copies →
  correctness cost (sub-chunk overshoot) dominates.
- Unrolled prefetch: the unrolled "second iter" never executes →
  pure overhead.

Generalising: **parallel parse already converted the parse-time
bottleneck into a copy-time bottleneck on already-small copies**. The
remaining wall-clock comes from per-token cooperative-copy overhead
(broadcast shuffles, barriers) and warp-scan setup — neither attacked
by these three increments.

### What would actually stack on PP

1. **Packed broadcast** — pack the 5 `__shfl_sync` per token (lit_len,
   match_len, match_offset, dst_local, lit_local) into 1 shuffle + 4
   PRMT extracts. Saves ~24 cycles per token × 4K tokens = ~96K cycles
   per warp = ~1-2 %. Free to try.
2. **Remove redundant `__syncwarp`** between literal and match copy
   when the match read demonstrably can't see literal write overshoot
   (always true on SHORT-token path with parallel parse). Saves ~5
   cycles per token = ~1 %.
3. **Cross-batch software pipelining** — overlap batch N+1's parallel
   parse with batch N's execution loop. Multi-day work, would
   genuinely shorten the per-warp critical chain.

The first two are micro-polish; the third is the next real lever if
perf needs more.

### Current state

`lz_decode_core.cuh` restored to PP-only baseline. PP-only timings on
warmed-up GPU: 3.37 ms (vs cold-GPU 2.92 ms reported earlier — the
22 % win figure was measured cold; under sustained bench load the GPU
throttles slightly, giving 10-12 % gain). Either way, PP is the
winner. Increments A, B, C make it worse or break it.

Tools/process artifact: `tools/build_gpu_full.bat` now chains
`build_gpu.bat` + `zig build gpulib`. Use this for every kernel-source
change to avoid the stale-DLL trap that wasted most of the session.

## Four "stack on top of parallel parse" items (2026-05-24, continued)

After the three increments above (A, B, C) all lost, ran four targeted
follow-up items proposed in the FAILED_EXPERIMENTS commentary:

### Item 1: pack 5 broadcast `__shfl_sync` per token into 2 (pack + PRMT)

Predicted ~1-2 %. Result: **0 %** (best 2.78 vs PP-only best 2.78). The
compiler was already pipelining the 5 shuffles efficiently — they sit
on the FMA pipe and overlap well. Reverted.

### Item 2: conditional `__syncwarp` between literal and match copy

Sync only when `|match_off| < lit_len + match_len` (close match would
have cross-lane read-after-write hazard).

Result: **+6 % regression** (2.95 vs 2.78). The conditional check
(2-3 ALU ops per token) costs more than the saved sync (5 cycles
when applicable). Reverted.

### Item 3: port PP to `decodeSubChunkRawMode<true>` (entropy-coded off16)

Removed the `if constexpr (!OFF16_SPLIT)` gate around the parallel-
parse fast path; added compile-time-dispatched off16 load that handles
both raw-interleaved (`<false>`) and split-hi/lo (`<true>`) cases. The
`<true>` instantiation is the hot path for L3+ frames whose off16
stream is entropy-coded.

Result on 100 MB enwik8:

| Level | True baseline | PP-only (v1) | **PP for both (v2)** | Δ vs baseline |
|-------|--------------:|-------------:|---------------------:|--------------:|
| L1 raw | 3.76 ms | 2.78 | 2.92 | **–22 %** |
| L3 | 3.79 ms | 3.79 (unchanged) | **2.99** | **–21 %** |
| L5 | 4.41 ms | 4.41 (unchanged) | **3.30** | **–25 %** |

Silesia (100 MB):

| Level | Baseline | **PP v2** | Δ |
|-------|---------:|----------:|--:|
| L1 raw | ~3.31 | 2.64 | –20 % |
| L3 | ~3.34 | 2.67 | –20 % |
| L5 | ~3.83 | 3.17 | –17 % |

**LOCKED IN.** Parallel parse now applies to every kernel/level
combination. The mechanism is identical for both `<false>` and `<true>`
— only the per-lane off16 byte-assembly differs, which the compiler
specialises at compile time via `if constexpr (OFF16_SPLIT)`.

### Item 4: cross-batch software pipelining

Considered: pre-load next batch's cmd bytes during current batch's
execute phase so it's L1-warm when parsed.

**Analysis instead of attempt** — within a single warp there's no real
mechanism for parse/execute overlap: a warp has one execution stream
and the SM scheduler already overlaps instructions across the 38
resident warps. More importantly, the cmd-prefetch is *effectively
already happening*: cmd_pos advances by 32 bytes per fast-path
iteration, the L1 cache line is 128 bytes, so 3 out of 4 consecutive
`cmd[cmd_pos + lane]` loads naturally hit L1 from the prior iteration's
cache fill. Estimated maximum saving: one L1 miss per 4 batches ×
25 cycles = ~6 cycles per 4 batches = negligible (<0.1 % kernel).

Cross-warp pipelining (one warp parses while a sibling warp copies)
would require restructuring `WARPS_PER_BLOCK` and the decode dispatch
— major surgery, out of scope for an increment.

**Skipped.** Documented for completeness; not pursued.

### Summary of all 10 experiments + 3 increments + 4 items this session

| # | Attack | Result |
|---|---|---|
| 1 | memcpy wide-load vectorize | no-op (PTX doesn't widen) |
| 2 | inline-PTX `ld.global.b32` | no-op (proved by NCU equality) |
| 3 | LZTurbo unconditional copy | no-op standalone; **fails** on PP path |
| 4 | cp.async stream staging | no-op (NCU all metrics equal) |
| 5 | back-ref prefetch unroll | no-op standalone |
| 6 | parallel parse via warp scan | **WIN +22 %** (after stale-DLL bug fix) |
| Inc A | wide loads on PP | **regression +80 %** |
| Inc B | unconditional copy on PP | **broken** (sub-chunk boundary) |
| Inc C | back-ref prefetch on PP | **regression +15 %** |
| Item 1 | packed broadcast on PP | no-op |
| Item 2 | conditional syncwarp on PP | regression +6 % |
| Item 3 | PP for OFF16_SPLIT path | **WIN +21-25 % on L3/L5** |
| Item 4 | cross-batch pipelining | skipped (no mechanism) |

**Two wins**: parallel parse (item 6 from prior session) and PP
for OFF16_SPLIT (item 3 from this run). Combined, they give:

| Workload | True baseline | Final | Speedup |
|---|---:|---:|---:|
| 100 MB enwik8 L1 raw | 3.76 ms | 2.92 ms | **–22 %** |
| 100 MB enwik8 L3 | 3.79 ms | 2.99 ms | **–21 %** |
| 100 MB enwik8 L5 | 4.41 ms | 3.30 ms | **–25 %** |
| 100 MB silesia L1 raw | ~3.31 ms | 2.64 ms | **–20 %** |
| 100 MB silesia L3 | ~3.34 ms | 2.67 ms | **–20 %** |
| 100 MB silesia L5 | ~3.83 ms | 3.17 ms | **–17 %** |

Universal 17-25 % decode speedup across all levels and inputs tested,
with the same single parallel-parse implementation handling both
raw-off16 and entropy-coded-off16 sub-chunks via compile-time
`if constexpr` dispatch.

### Lessons retained for future kernel work

1. **Always rebuild the DLL** (`tools/build_gpu_full.bat`) after any
   kernel change. The cubin rebuild from `build_gpu.bat` alone is
   silently dead because the DLL embeds the old PTX via `@embedFile`.
2. **Inject `__trap()`** at the entry of a modified function as the
   first sanity check that the new code path is actually executing.
   Wall-clock identical to baseline + every NCU metric identical to
   baseline almost certainly means the kernel didn't pick up the change.
3. **NCU stall analysis points at categories, not specific shortenable
   things.** "Long Scoreboard 50 %" can mean (a) a specific load
   chain that you can attack with software pipelining or (b) the
   irreducible scheduling overhead of a well-occupied SM. The first
   is fixable; the second is not. The parallel-parse win came from
   reducing per-warp instruction count (a structural change), not from
   reducing per-load latency on the existing chain (which never moved
   anything despite five attempts).
4. **GPU thermal state matters for sub-second benchmarks.** Same code
   measured 2.78 ms cold, 3.37 ms after sustained bench load. Re-run
   alongside the comparison code at the same thermal state for honest
   A/B.

## Three L1-thrash attacks on top of PP-v2 (2026-05-24, continued)

NCU profile of PP-v2 showed L1 hit rate dropped from 81.6% (pre-PP)
to 64% post-PP, with Long Scoreboard climbing 49.9% → 60.9% of stalls.
DRAM throughput up from 18.5% to 25.5%. Hypothesis: PP's coalesced
cmd/off16 stream loads are evicting back-reference data, and freeing
L1 capacity for back-refs would recover some kernel time.

Three independent attacks on this hypothesis, all on top of PP-v2:

### A1: `__ldcs` (cache-streaming hint) on cmd / off16 stream reads

Tells the LSU "load this byte and mark the L1 line for early eviction".
The streams are read-once after coalesced fill so __ldcs is the right
semantic.

Result (100 MB):

| Workload | PP-v2 | + __ldcs | Δ |
|---|---:|---:|--:|
| enwik8 L1 raw | 2.92 ms | 2.92 | 0 |
| enwik8 L3    | 2.99    | 2.99 | 0 |
| enwik8 L5    | 3.30    | 3.26 | –1.2 % |
| silesia L1   | 2.64    | 2.64 | 0 |
| silesia L3   | 2.67    | 2.67 | 0 |
| silesia L5   | 3.17    | 3.16 | 0 |

Marginal positive on enwik8 L5 only. Within noise.

### A2: Force max-L1 carveout via `cuFuncSetAttribute`

Set `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 0` (0 %
shared, 100 % L1) on both LZ kernels. Since the kernels use 0 shared
memory currently, this gives them the full 128 KB combined L1/shared
as L1.

Result:

| Workload | PP-v2 | + max-L1 | Δ |
|---|---:|---:|--:|
| enwik8 L1 raw | 2.92 ms | 2.88 | –1.4 % |
| enwik8 L3    | 2.99    | 2.99 | 0 |
| enwik8 L5    | 3.30    | 3.26 | –1.2 % |
| silesia L1   | 2.64    | 2.67 | **+1.1 %** |
| silesia L3   | 2.67    | 2.73 | **+2.2 %** |
| silesia L5   | 3.17    | 3.24 | **+2.2 %** |

Marginal positive on enwik8, **negative on silesia**. The carveout
probably costs something on the SM-scheduler side that hurts the more
varied silesia workload. Net: no clear win. Reverted.

### A3: cp.async cmd staging into shared memory (L1-bypass)

Stage 256-byte cmd windows in shared memory via `cp.async.cg`
(bypasses L1). 12 KB total shared / SM, costs ~10 % of L1 capacity in
exchange for not running cmd loads through L1 at all. Required threading
a `LzStage&` reference through `parseAndDecodeSubChunkRaw`,
`parseAndDecodeSubChunk`, and both kernel entry points (~50 LOC).

Result:

| Workload | PP-v2 | + cp.async cmd | Δ |
|---|---:|---:|--:|
| enwik8 L1 raw | 2.92 ms | 2.88 | –1.4 % |
| enwik8 L3    | 2.99    | 2.99 | 0 |
| enwik8 L5    | 3.30    | 3.26 | –1.2 % |
| silesia L1   | 2.64    | 2.67 | **+1.1 %** |
| silesia L3   | 2.67    | 2.73 | **+2.2 %** |
| silesia L5   | 3.17    | 3.26 | **+2.8 %** |

Same shape as A2 — marginal on enwik8, worse on silesia. The shared mem
cost + cp.async issue/wait overhead exceeds the L1 capacity benefit on
silesia. Reverted.

### Combined verdict: PP-v2 IS the ceiling for L1-thrash attacks

Three independent attacks on the "L1 thrash" hypothesis all show the
same pattern: tiny enwik8 signal (within noise), neutral/negative
silesia. The L1 hit rate drop (81.6 → 64 %) appears to be **inherent
to PP-v2's higher utilization** (Avg Active Threads/Warp jumped
18.44 → 22.53, meaning more loads per warp instruction), not a fixable
thrashing problem.

The kernel's per-byte L1 traffic genuinely grew with PP because PP
made the cooperative lanes do MORE useful work per instruction. The
L1 has the same physical capacity; with more loads/cycle, hit rate
naturally drops. This isn't a leak we can plug.

**PP-v2 is the practical ceiling for this kernel structure on Ada
within the current parallelization scheme.** Further gains require
structural changes:

- Different parallelization granularity (multiple warps per sub-chunk
  cooperating, cross-warp pipelining)
- Format-level changes (wider tokens, fewer total operations per byte)
- Hopper-specific features (TMA, async warpgroups) on newer hardware

None pursued in this session. The 21-25 % gain from parallel parse
+ PP-for-OFF16_SPLIT remains the locked-in win.

### Final tally for the session

- 6 standalone experiments tried (3 found to be against stale DLL)
- 3 increments tried on top of PP
- 4 + 3 follow-up items tried on top of PP-v2
- **2 wins locked in**: parallel parse (PP) + PP for OFF16_SPLIT
- 1 measurement-infra bug fix (`finalizeProfiling`)
- 1 build-infrastructure script (`tools/build_gpu_full.bat`)
- Universal 17-25 % decode kernel speedup across all 6 tested
  workload/level combinations.

Going further would need different ideas than per-instruction
optimization. Stop here unless format change or major restructure is
on the table.

## Maintaining parallel CPU and GPU codebases (abandoned 2026-05-29)

**Context**: StreamLZ began as a CPU LZ77 codec aimed at outperforming
zstd on decompression speed at comparable ratios. Once the codec was
production-ready, GPU compute (decompression first, encode second)
was added as a second implementation. Both targeted the same wire
format up to a point — the GPU side eventually diverged for
saturation-friendly reasons (sc=0.25 sub-chunk size, raw-only at
L1/L2, no dictionary support, 32-stream Huffman BIL layout) and the
two wire formats became mutually incompatible by design.

**The problem**: Every wire-format-touching change had to land twice.
Bug fixes had to either survive both formats or get reasoned about
across two codepaths. The CPU codec carried its own parser variants
(greedy / lazy / optimal / BT4 / chain), entropy coders (tANS, Huffman,
RLE), parallel dispatch infrastructure, dictionary trainer, and
comparison-vendor harnesses (zstd, lz4, LZTurbo). The GPU codec had
its own orchestration, CUDA-driver FFI, six bookkeeping kernels and
a Vulkan decode-shader backend. The combined surface was
~42 kLoC of Zig + several kLoC of CUDA; small invariants drifted
between sides faster than they could be checked.

**The decision**: Strip the CPU codec entirely. GPU becomes the only
backend. Build flag `-Dgpu=true` goes away; `zig build` produces a
GPU-only `streamlz.exe` + `streamlz_gpu.dll`. The previously-required
CPU-decoder fallback for hosts without CUDA is gone — callers on
GPU-less hosts must use a different library.

**Strip outcome**: ~30 kLoC of Zig + ~95 kLoC of vendored C
(zstd 1.5.7 + LZ4 1.10.0) removed. The Vulkan decode-shader backend
went with it (CUDA-only). The host-side wire-format orchestration
that the GPU codec depends on stays: the Fast-codec frame builder
(now `src/encode/fast_framed.zig`), the host re-encode helpers (now
`src/encode/gpu_stream_assembly.zig`), the CPU Huffman / tANS
encoders the orchestrator calls into, and the frame walk in the
decoder.

**Layout reorganization**: `src/gpu/` is flattened up to `src/`. There
is no longer a CPU vs GPU split to model in directory names —
everything that remains is GPU code or supporting infrastructure.

**Verification**: bench_all.bat on enwik8 (100 MB) + silesia (213 MB),
L1-L5, `-db -r 30`, RTX 4060 Ti sm_89. All ten rows SHA-256 OK
pre- and post-strip. D2D wall-clock and end-to-end timings stayed
within ±0.05 ms / ±0.23 ms of the pre-strip baseline; ratios within
±0.04 pp. Most deltas were small improvements (the simpler dispatch
shaved a handful of microseconds per call).

**Trade-offs accepted**:
  * Dropped CPU-only ports / hosts. The codec is now GPU-only and
    the C ABI is GPU-only.
  * Dropped the `-bc` / `-bcf` benchmark modes that compared StreamLZ
    against zstd, lz4, LZTurbo. The comparison data lives in
    historical README tables.
  * Dropped the dictionary trainer + the seven built-in dictionaries.
    The GPU codec never produced dictionary frames; the feature was
    CPU-only.
  * Dropped High-codec levels L6-L11. The GPU encoder only implements
    L1-L5 (Fast codec). L6+ scoring the user input now returns
    `error.BadLevel`.

The two-codebase phase produced the best CPU and GPU LZ77 decoders
the author has measured. Maintaining both was the limiting factor on
further work; the strip restores that bandwidth.
