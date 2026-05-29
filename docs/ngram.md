# N-gram Forward Scatter — Design Document

> **Historical.** Design notes for an optimization targeting the
> CPU High codec (L9-L11) that was retired with the GPU-only strip
> on 2026-05-29. The idea was never built. Kept because the
> cache-warming-via-pre-scatter analysis may be useful for the GPU
> LZ kernel if it ever becomes memory-bandwidth-bound at larger
> inputs.

## The Problem

StreamLZ's High codec (L9-L11) decompresses at ~1,050 MB/s single-threaded
on Arrow Lake, vs zstd 19's ~1,500 MB/s on the same data. VTune profiling
identified the bottleneck: **match copy cache misses**. The `copyForwards`
function (match byte copying) accounts for 10% of cycles at 187% memory-bound —
large-offset match copies hit cold L3/DRAM cache lines.

The two-phase architecture (resolveTokens → executeTokensType1) makes this
worse: phase 1 resolves all tokens into a buffer, phase 2 executes them.
By the time phase 2 copies match data, the source bytes are cold. A 128-token
prefetch pipeline helps but can't fully compensate.

## The Idea

**Pre-populate the output buffer with frequently-repeated content before
normal LZ decode.** If "the " appears 50,000 times in the output, read it
once and scatter-write it to all 50,000 destination positions. When the
normal backward-LZ decoder later encounters a match that produces "the ",
the destination cache line is already warm (or the match can be skipped
entirely if the bytes are already correct).

This is conceptually similar to:
- **Brotli's static dictionary** — common substrings pre-loaded for decode
- **Grammar-based compression (Re-Pair)** — iterative phrase extraction
- **Prefetching** — but writing actual bytes instead of just hints

The novel aspect: using a dictionary phase specifically as a **cache-warming
strategy** for the LZ decode that follows, not for improving ratio.

## What We Found

### Attempt 1: Group by exact match content (failed)

Instrumented `resolveTokens` to hash each resolved token's match content
and group by (content, length). Results on enwik8 100 MB at L9:

- 1,048,576 unique patterns, 498,152 repeated (≥2 occurrences)
- Repeated patterns cover 42.8 MB = 44.3% of match bytes
- **But**: the top 20 patterns cover only 4.2% of match bytes
- The distribution is **flat** — no pareto concentration

**Why it failed**: The High codec's optimal parser picks the cheapest match
at each position, producing diverse long matches (avg ~10 bytes). "the cat
sat on" appears as one 14-byte match, hiding the common "the " substring
inside it. Grouping by exact content misses the shared prefixes.

### Attempt 2: Fixed 4-gram frequency on raw output (promising but overcounted)

Counted all 4-byte windows across the 100 MB decompressed output:

| Top N | Raw bytes | % of output |
|-------|-----------|-------------|
| 10    | 16.5 MB   | 16.5%       |
| 100   | 48.2 MB   | 48.2%       |
| 1000  | 132.9 MB  | 133% (!)    |

Coverage exceeds 100% because 4-grams overlap. " the" and "the " share
3 bytes at every position where " the " appears. The raw count is
meaningless — we need non-overlapping coverage.

### Attempt 3: Greedy non-overlapping 4-gram extraction (strong pareto)

Algorithm: pick the most frequent 4-gram, claim all its positions (marking
bytes as unavailable), recount remaining 4-grams, repeat.

**Results on enwik8 100 MB:**

| Top N | Unique bytes claimed | % of output |
|-------|---------------------|-------------|
| 1     | 3.2 MB (" the")     | 3.2%        |
| 5     | 7.8 MB              | 7.8%        |
| 10    | 11.4 MB             | 11.4%       |
| 25    | 17.1 MB             | 17.1%       |
| 50    | 22.2 MB             | 22.2%       |
| 100   | 28.5 MB             | 28.5%       |
| 250   | 39.1 MB             | 39.1%       |
| 500   | 47.9 MB             | 47.9%       |
| 1000  | 56.3 MB             | 56.3%       |

**This IS a pareto distribution.** The top 100 4-grams cover 28.5% of
output with zero overlap. The top entries:

1. ` the` — 804,072 occurrences (3.2 MB)
2. `    ` — 190,639 occurrences (762 KB)
3. ` of ` — 348,074 occurrences (1.4 MB)
4. ` and` — 322,193 occurrences (1.3 MB)
5. `tion` — 285,504 occurrences (1.1 MB)

## Architecture Options

### Option A: Forward scatter as cache-warming (no format change)

Post-compress: scan resolved tokens, extract top N 4-grams, emit a small
sidecar with pattern → position-list entries. On decompress:

1. Read sidecar, scatter-write all patterns to their output positions
2. Run normal two-phase decode (resolveTokens + executeTokensType1)
3. Match copies that land on pre-written positions hit warm cache

**Pros**: No skip logic needed, backward compatibility (old decoders ignore
the sidecar), simple implementation.

**Cons**: Redundant writes (pattern written twice — once by scatter, once
by backward match). Scatter-writes themselves may thrash cache if positions
are random.

### Option B: Forward scatter with match skipping

Same as A, but mark forward-handled tokens so phase 2 skips the match copy.
Options for marking:
- Bitmask: 1 bit per token, ~800 KB for 7M tokens
- Sentinel offset: set token.offset = 0 for handled matches
- Modified cmd stream: new cmd type meaning "match already written"

**Pros**: Eliminates redundant copies. Phase 2 is faster because it skips
work.

**Cons**: Requires format change (new sidecar type). Skip check adds a
branch per token in phase 2.

### Option C: N-gram dictionary as a decompressor feature

Don't embed the dictionary in the compressed stream. Instead, the
decompressor builds it on-the-fly during the first pass of a two-pass
decode:

1. Pass 1: normal resolveTokens + executeTokensType1 (as today)
2. Between passes: scan output for top N 4-grams
3. Pass 2: for subsequent frames/blocks, pre-populate from the dictionary

**Pros**: Zero compressed size overhead. Works with existing format.

**Cons**: Requires two full passes. Only helps on multi-block files where
later blocks benefit from earlier blocks' dictionary. Single-block L9
files (the common case) get no benefit.

## Key Design Questions

1. **What n-gram length?** 4 is the sweet spot for English text (matches
   common words). Binary data may prefer 8+. Should this be adaptive?

2. **How many patterns?** 100 covers 28.5% of enwik8 output. 1000 covers
   56.3%. Each pattern needs a position list in the sidecar. Cost/benefit
   crossover likely at 100-500 patterns.

3. **Position list encoding?** Delta-varint is compact. Sorted positions
   with delta encoding: [first_pos, delta1, delta2, ...]. For 804K
   occurrences of " the", the position list is ~1.6 MB with 2-byte
   average deltas.

4. **Sidecar size budget?** For 100 patterns with position lists: ~500 KB.
   For 1000 patterns: ~5 MB. This increases compressed size by 0.5-5%.
   Acceptable for a decode-speed-optimized mode, but should be optional.

5. **Does this help non-text data?** The 4-gram analysis was on enwik8
   (English Wikipedia). JSON/HTML/CSS would likely show similar patterns
   (common tags, keywords). Binary data (images, executables) would show
   much less concentration. Need to test on silesia corpus.

6. **Interaction with existing parallel decode?** The forward scatter
   phase must complete before any backward-LZ decode begins. This adds
   a serial phase. On multi-thread, this could be parallelized (each
   thread scatters its subset of patterns), but it's still extra work
   before the main decode starts.

## Sidecar Format (Proposed)

```
[4 bytes] magic: "FWD1"
[2 bytes] num_patterns (u16, max 65535)
[2 bytes] n_gram_length (u16, typically 4)

For each pattern:
  [n bytes]  pattern bytes
  [4 bytes]  num_positions (u32)
  [variable] positions: delta-varint encoded, sorted ascending
    First position: varint
    Subsequent: delta from previous, varint

[4 bytes] total sidecar length (for seeking from end)
```

## Estimated Impact

**Best case (text-heavy data like enwik8):**
- Forward scatter covers 28% of output (top 100 patterns)
- Match copies for those 28% hit L1 instead of L3/DRAM
- Cache miss cost: ~40 cycles/miss × ~2M misses saved ≈ 80M cycles saved
- At 3.5 GHz, that's ~23 ms on a 100 ms decode = **~20% speedup**
- Sidecar cost: ~500 KB = 1.8% compressed size increase

**Worst case (binary/random data):**
- No repeated 4-grams, sidecar is empty or tiny
- Zero benefit, zero cost (sidecar not emitted)

**Unknown:**
- Scatter-write cache behavior: are the 800K writes to random positions
  themselves cache-hostile enough to negate the benefit?
- Interaction with the existing 128-token prefetch pipeline: does the
  prefetch already catch most of the patterns the scatter would help?
- Impact on compress time: the n-gram analysis adds a post-processing
  step after compression.

## Likely Next Steps

1. **Test on silesia corpus** — verify the 4-gram distribution holds for
   non-text data (tar archive with mixed file types).

2. **Prototype Option A** (cache-warming, no format change) — the simplest
   to implement and test. Measure decode speed with and without the
   scatter phase on both Arrow Lake and Ice Lake.

3. **Measure scatter-write cache behavior** — does writing 800K × 4 bytes
   to random positions thrash the cache enough to offset the benefit?
   Profile with VTune memory-access analysis.

4. **Consider variable-length patterns** — instead of fixed 4-grams, use
   Re-Pair style iterative extraction to find the optimal set of patterns
   at mixed lengths. More complex but potentially higher coverage per
   pattern.

5. **Consider this as a new compression level** — e.g., L12 or a flag
   like `--forward-scatter`. Not enabled by default; opt-in for users
   who prioritize decode speed over compressed size.

## Related Work

- **Brotli static dictionary**: ~120K common English substrings baked into
  the format. Encoder emits "copy from dictionary" tokens. Similar concept
  but static (same dict for all files) and used for ratio, not speed.

- **Re-Pair (Recursive Pairing)**: Grammar-based compression. Iteratively
  replaces the most frequent bigram with a new symbol. Optimal for
  finding the best set of repeated phrases. Too slow for a post-compress
  step but the greedy n-gram extraction approximates it.

- **zstd trained dictionaries**: File-specific dictionaries trained on
  sample data. Stored separately and provided at both compress and
  decompress time. Similar to our approach but used for ratio improvement
  on small files, not decode speed on large files.

- **LZ4 prefix dictionaries**: Pre-seeds the LZ window with known content.
  The decoder must have the dictionary available. Similar concept but
  operates at the LZ window level, not as a scatter-write phase.
