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
