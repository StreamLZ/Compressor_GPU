# StreamLZ GPU Architecture

This document explains how the GPU codec actually achieves its speed.
The README enumerates the files; this one explains the decisions
inside them.

Read this if you want to understand the codec as a designer would,
not just operate it as a user.

## The problem GPUs are good at, and the part they aren't

LZ77 decompression is famously serial. To decode the next byte, you
have to know the result of the current token. A token tells you
either "copy these literal bytes" or "copy bytes from earlier in the
output". The "earlier in the output" can be just one byte back, in
which case the next 32 lanes of a GPU warp cannot read those bytes
in parallel; lane 17 needs the byte that lane 16 has not yet written.

This is the fundamental tension. A GPU has thousands of arithmetic
units sitting idle waiting for work. An LZ77 decoder, naively
implemented, uses one of them.

The escape hatch is the wire format. If you partition the input into
self-contained chunks where each chunk has its own dictionary and
its own complete set of streams, then each chunk can be decoded by
a separate worker without any cross-chunk coordination. The codec
chooses to make each chunk small enough that a single warp can
handle it, and chooses to make the chunks small enough that a
realistic input file produces hundreds or thousands of them. On a
100 MB file the codec produces about 1500 sub-chunks, which is more
than enough to fill every streaming multiprocessor on a modern GPU
with useful work.

Inside one warp, the decoder is still serial in spirit, but the warp
itself has 32 lanes. Most of the work done while decoding a single
chunk is not token parsing; it is copying bytes. The literal copy
("paste these 50 bytes into the output") and the match copy ("paste
those 50 bytes from earlier into the output") are bulk operations
that a 32-lane warp can complete in `(50 + 31) / 32 = 2` iterations
(ceiling division: 50 bytes split across 32 lanes per iteration)
rather than 50. Token parsing remains roughly serial, but it now
runs in a small fraction of the warp time because the byte-copy
phase is so much faster.

That is the entire idea. Everything else in this document is the
mechanical work of making it actually fast.

## How the work is mapped to GPU hardware

The codec settles on a few mapping decisions and applies them
consistently.

One chunk is one CUDA block (since v4 #15, 2026-06-11). A chunk
holds at most 256 KB of decompressed output, partitioned into one or
more sub-chunks of at most 128 KB each (default 64 KB). The block's
four warps cooperate on that one chunk with fixed roles: warp 0 (the
parser) runs the token-parse scans and stages each 32-token batch's
prefix sums into a shared-memory double buffer, while warps 1-3 (the
copier team, 96 lanes) execute the staged batch's copies one batch
behind. The parser can run ahead because its state advancement comes
entirely from the warp scans, never from the copies - so its global
reads (cmd bytes, off16 entries) hide under the team's global
writes. See "The K=4 pipelined decode" section below for the
synchronization rules and measured results.

The pre-#15 model - one chunk per warp, two non-cooperating warps
packed per block (`WARPS_PER_BLOCK = 2`) purely to amortize launch
overhead - survives in `slzLzDecodeKernel` / `slzLzDecodeRawKernel`,
the fallback when `SLZ_NO_PIPELINE=1` is set or the pipelined
symbols are missing. Everything in "Inside the decode loop" below
describes per-batch work that is identical in both models; only who
executes which phase changed.

For Huffman decoding, the encoder has already split the input into
32 sub-streams so the decoder can run all 32 lanes in parallel.
`slzHuffDecode4StreamKernel` (name retained for ABI compatibility:
it decodes 32 streams, not 4) uses one warp per Huffman block with
every lane active. Each lane runs its own bit-buffer plus LUT loop
on its own input slice; there is no cross-lane traffic in the hot
loop. See the 32-stream Huffman decoder section below for the
input-offset prefix-sum + per-lane decode-loop details.

The orchestration kernels that manage the per-frame bookkeeping
(walk the frame, prefix-sum the chunk counts, scan the sub-chunk
headers, compact the descriptors, merge the per-stream-type arrays)
all run with tiny grids of one or a handful of blocks. They are not
bottlenecks; their job is to get the bookkeeping right with as
little PCIe traffic as possible. Spending a kernel launch on a
single-block coordination kernel is much cheaper than copying its
inputs back to the CPU, running the loop there, and copying the
results back to the device.

## The decode pipeline at a glance

A decode call goes through six stages on the GPU. Every stage runs
on the device. The host's only contribution after the initial launch
is a 56-byte readback of launch-plumbing counters used to size the
next kernel's grid.

```
   slzWalkFrameKernel
            |
            v
   slzPrefixSumChunksKernel
            |
            v
   slzScanParseKernel  ->  slzCompactAllDescsKernel  (ONE launch, 5 blocks:
                            4 huff streams + raw pair, 2026-06-10 fusion)
                       ->  slzMergeHuffDescsParKernel (4-block parallel merge)
            |
            v
   slzGatherRawOff16Kernel
            |
            v
   slzHuffBuildLutKernel  ->  slzHuffDecode4StreamKernel
            |
            v
   slzLzDecodeGeneralPipelinedKernel
   (or slzLzDecodeRawPipelinedKernel for L1/L2;
    slzLzDecode[Raw]Kernel under SLZ_NO_PIPELINE=1)
```

The walk kernel parses the frame header on the device and emits a
descriptor for every chunk. The prefix-sum gives every chunk its
starting sub-chunk index so later kernels can compute scratch
offsets without a second pass. The scan kernel inspects every
sub-chunk's stream headers and emits a tentative descriptor per
entropy-coded or raw stream. The fused compact kernel (one
launch, five single-thread blocks running concurrently on five SMs;
formerly five sequential launches summing ~0.4 ms) and the 4-block
parallel merge filter those tentative descriptors into the final
per-stream-type arrays the entropy kernel will consume. The unfused
legacy kernels remain in the PTX as reference/fallback. The gather kernel copies raw
sixteen-bit-offset bytes from the compressed input into the entropy
scratch region (these are not entropy-coded but they live in the
same scratch buffer to keep the LZ kernel's read pattern uniform).

Then the entropy stage runs. The LUT-build kernel takes the
canonical-Huffman code lengths and builds a 1024-entry decode LUT
per stream. The decode kernel uses those LUTs to decompress the
Huffman streams into the entropy scratch buffer.

Finally the LZ decode kernel runs. It reads from the entropy scratch
as if those bytes had been raw literals, and it reads the offset and
length sub-streams directly from the compressed input. Its output
goes straight into the caller's destination buffer.

The Huffman pre-decode and the LZ decode share a CUDA stream, so the
LZ launch is queued immediately after the Huffman launch and the
two run back to back without a host-side wait. The CUDA runtime
treats both launches as one timing window. In practice (2026-06-10,
enwik8 L5) the LZ decode dominates at ~3.2 ms vs ~0.9 ms for the
Huffman LUT-build + pre-decode pair; run with SLZ_PROFILE_DECODE=1
for the live per-kernel table.

## The parallel-parse rewrite of the decode hot loop

This is the change that most recently moved the decode numbers. It is
worth describing in detail because the technique generalizes.

The decode loop originally followed a textbook layout. Lane zero read
one token byte from the command stream, decoded it into five fields
(literal length, match length, match offset, recent-offset flag,
off16 stream position), and then broadcast each of those five fields
to the other thirty-one lanes via separate `__shfl_sync` calls.
After the broadcasts, all thirty-two lanes participated in the
literal copy and the match copy. Then the loop advanced and the next
token was parsed.

The serial parse itself is unavoidable; tokens have variable lengths
and you cannot tell where token N+1 begins until you have decoded
token N. But the five broadcasts after every single token were
expensive. With one warp per chunk and millions of tokens per
realistic input, the broadcasts dominated the kernel's instruction
mix.

The rewrite processes 32 tokens at a time in three phases.

Phase A loads a 32-byte chunk of the command stream into registers
with one coalesced load. All 32 lanes participate; lane N gets
command byte N.

Phase B classifies each lane's byte as a short token or a long
token. Short tokens (token value greater than or equal to 24, which
is the common case in real data) carry their entire payload inside
the token byte itself. Lane N can decode its own byte without
consulting any other lane: literal length is the bottom three bits,
match length is the next four bits, and the recent-offset flag is
the top bit. Long tokens require side-stream lookups and fall back
to the original serial path for the entire batch.

Phase C uses a Hillis-Steele warp-level prefix scan to compute each
lane's running offset into the destination, literal, and off16
streams. The scan itself takes five shuffles total regardless of
how many tokens are in the batch. After the scan, every lane knows
where in each stream its token's data lives, and the cooperative
literal and match copies can run.

The amortized cost dropped from five shuffles per token to five
shuffles per thirty-two tokens, a thirty-fold reduction on the
common path. On real workloads where short tokens are the
overwhelming majority, the parallel-parse path covers roughly 80%
of tokens.

The measured impact was a meaningful reduction in D2D wall-clock
across every compression level and both test corpora; see the
performance table in the README for the current numbers.

## Warp-cooperative byte copies

After the token is parsed, the warp cooperates on the actual byte
movement. Two helper functions in `lz_decode_core.cuh` do almost all
of the byte-copy work in the codec.

The literal copy is straightforward. All 32 lanes participate, each
copying one byte at a stride of WARP_SIZE until the run is
exhausted. (See the worked 50-byte example near the top of this
document for the iteration count.)

The match copy is more delicate. A back-reference at distance d for
length l is parallel-safe only when d is at least l. If d is smaller
than l, the back-reference is self-referential: lane N tries to read
the byte at destination position (current + N - d), but that byte
will be written by lane N - d, which has not run yet. The output
becomes garbage.

When d is at least l, the warp fans out the same way as the literal
copy and the match completes in a few iterations. When d is smaller
than l, lane zero runs a serial loop and the other lanes idle. For
most LZ77 workloads the parallel path covers the great majority of
bytes; the serial fallback exists only for correctness on tight
repeats.

The helpers are marked `__forceinline__` so the compiler reproduces
the same SASS that hand-inlined loops would produce. After the
extraction, both the LZ decode kernel and the LZ raw kernel had
exactly the same register and stack usage they had before, confirming
that the extraction was purely a readability change.

## The K=4 pipelined decode (v4 #15)

The single-warp decode loop alternates between parsing a 32-token
batch (global reads of cmd/off16 bytes feeding warp scans) and
copying that batch's output (global reads/writes of literal and
match bytes). After the flat batched copies (v4 #1/#2) collapsed the
copy phase's instruction count, NCU showed the kernel at 52.7% SM
throughput with `long_scoreboard` (memory latency) as the dominant
stall: the warp simply waits on its own loads. The fix is
structural - split the two phases across warps so one phase's
latency hides under the other's.

`slzLzDecodeRawPipelinedKernel` / `slzLzDecodeGeneralPipelinedKernel`
give each chunk a 128-thread block (`blockDim (32,4)`, 12 blocks/SM):

* Warp 0 (parser) runs `fillBatch` - the same PP scans as the
  single-warp loop - and stages each batch's prefix sums, match
  data, and dependent-match ballot into a `PipeBatch` shared-memory
  double buffer. Its state advancement (cmd/lit/off16/dst positions,
  recent offset) comes entirely from the scans, so it never waits on
  output bytes.
* Warps 1-3 (the copier team, 96 lanes) execute the staged batch one
  behind the parser: the flat-literal and flat-independent-match
  passes as two team-wide loops (no barrier between them - the
  independent-match reads touch only pre-batch-final bytes, and the
  write ranges are disjoint), then warp 1 alone runs the dependent
  matches in token order.
* The slot handoff is ONE `__syncthreads()` per batch, with per-slot
  staged flags. The team's internal phase order uses hardware named
  barrier 1 via `__barrier_sync_count(1, 96)`, so the parser never
  participates in a mid-batch sync.
* Serial long tokens drain the pipeline, execute cooperatively on
  all warps, and re-prime. In the general kernel, sub-chunks with
  off32 entries or exotic modes fall back to the warp-level
  `decodeSubChunkGeneral` on warp 0 inside the kernel.

Synchronization rules this design is built on (each one measured,
see FAILED_EXPERIMENTS.md "mbarrier pipeline ring"):

* `__syncthreads` beats mbarrier at this cadence: a `cuda::pipeline`
  producer/consumer ring measured 2.18-2.64 ms vs 1.90 ms on enwik8
  L1. Four mbarrier ops per ~150 ns batch cost more than one
  hardware `BAR.SYNC`; the barrier stall is mostly the inherent wait
  for the slower side, which only width rebalancing removes.
* Named barriers must use the `__barrier_sync_count` intrinsic, not
  inline-asm `bar.sync` (the `asm volatile` is an optimizer wall in
  the hot loop; +0.8 ms measured).
* Warp counts per block must be a power of two: K=3 (96-thread
  blocks) packs unevenly on the SM's four schedulers and measured
  19.0 ms vs 16.2 at 1 GB.
* Raw spin-waits between warps deadlock - NVIDIA's warp scheduler
  has no fairness guarantee.

Results (RTX 4060 Ti): enwik8 L1 2.28 → 1.77 ms, enwik9 L1
20.9 → 16.2 ms (61.7 GB/s, 2.03× nvCOMP LZ4); enwik8 L5
3.31 → 2.93 ms, enwik9 L5 29.5 → 26.3 ms (1.93× nvCOMP Zstd). NCU
after: 71.9% SM and 71.9% memory throughput moving in lockstep -
balanced saturation - with the residual barrier stall (8.2) being
the lockstep tax. One trade: parser count per SM halves vs the old
packing, so parser-bound corpora pay slightly at L1 (silesia
3.87 → 4.05 ms); at L3+ even silesia wins.

## Why the header parser is marked `__noinline__`

`parseSubChunkHeaders` is the function that walks the sub-chunk's
stream headers (literal, command, off16, off32, length) and emits
the parsed pointers and sizes for the decoder to use. It is called
once per sub-chunk, before the decode hot loop starts.

The function is marked `__noinline__` deliberately. The header
parser keeps several dozen local variables alive while it works:
cursors into the compressed buffer, chunk-type values, stream sizes,
flags for pre-decoded entropy streams. If the compiler inlined the
parser into the decoder's enclosing function, those locals would
compete for registers with the decode loop's working set. The
register pressure would force lower occupancy, and the decoder would
slow down.

The `__noinline__` attribute means the compiler allocates registers
for the parser at its own entry point, lets the parser finish and
return, and then enters the decoder with a fresh smaller register set.

This was confirmed empirically. Removing `__noinline__` and rebuilding
inflated the LZ decode kernel's per-thread register footprint enough
to cross an occupancy threshold and visibly drop measured throughput.
The attribute stays.

The opposite experiment also failed. The J4 cleanup tried adding
`__noinline__` to `decodeSubChunkGeneral` (the entropy-mode decoder's
hot loop body, much larger than the header parser). PTX STACK grew on
both `slzLzDecodeKernel` (192 → 208) and `slzLzDecodeRawKernel` (72 →
80) with REG unchanged, indicating the attribute forced extra spill
slots. The header parser is small enough that out-of-lining it is a
win; the general decoder is large enough that nvcc already places it
out-of-line implicitly. Forcing the attribute on the general decoder
only adds the spill slots. The lz_decode_general.cuh top-of-file
comment captures this conclusion.

## The 32-stream Huffman decoder

A canonical-Huffman block in this codec consists of code-length
information, followed by 32 independent compressed sub-streams. The
32 streams are interleaved at encode time so the decoder can process
them in parallel: lane k of the warp decodes sub-stream k for k in
0..31. All 32 lanes are active. The earlier design used 4 streams
(modeled on zstd's 4-way interleave), which left 28 of 32 lanes
idle; widening to 32 is the single biggest source of the kernel's
current throughput.

### BIL: bounded-interleaved wire format

The 32 streams are stored in a bounded-interleaved (BIL) layout
rather than 32 concatenated tails. Each per-lane stream is logically
sliced into 4-byte BIL words; let `K = min(words[s])` be the largest
prefix every stream has. The body is:

```
  [128 B weights - 4 bits/symbol]
  [96 B sub-header - 32 × u24 LE per-stream byte sizes]
  [4 B K - u32 LE]
  [K rows × 128 B interleaved area - row w holds lane k's word w at
                                      offset (w · 128 + k · 4)]
  [tail area - per-stream bytes from word K onward, at exclusive
               prefix-sum-of-tail-sizes offsets]
```

The hot-loop refill is a single coalesced 128-byte sector load (one
4-byte word per lane from the same row); the prior concat layout
needed 32 scattered per-stream loads because each lane's read pointer
lived in a different byte range. Only the trailing tail rows
(typically <10% of refills on natural text) keep the old scatter
pattern. The encoder zero-pads each stream's last word so refills
never read partial bytes.

Stream sizes are stored explicitly for all 32 streams (vs the prior
"31 × u24 + derived" layout): every lane needs to know its own
`words[s]` locally so the warp can min-reduce to K without an extra
total-payload round-trip. Cost is +3 B sub-header + 4 B K per
Huffman body; rounding-to-word costs ≤3 B per stream (≤96 B per
body). On 64 KB sub-chunks this is ~0.3 pp ratio across enwik8 and
silesia in exchange for the ~4–6% Huffman-kernel-time win at L3–L5.

Body offsets in the file are `chunk_offset + 5` (after the 5-byte
type-4 chunk header), so the refill pointer is 4-aligned only ~25%
of the time. The decoder uses `memcpy(&word, w_ptr, 4)` rather than
a u32 reinterpret_cast; nvcc lowers it to byte loads + shifts when
alignment isn't provable and to a single `ld.u32` when it is.

Each lane runs the same tight inner loop. It refills a 64-bit
bit buffer from the compressed input whenever the buffer drops below
the maximum code length, indexes a 1024-entry shared LUT with the
top ten bits of the buffer, and consumes the symbols encoded in the
LUT entry.

A LUT entry packs four fields: two output symbols, the combined
length of their codes in bits, and a flag indicating whether the
entry represents one symbol, two symbols, or an escape to a longer
code. The two-symbol case is the source of the codec's Huffman
throughput. For short codes (length 5 or less), two consecutive
codes both fit inside the ten-bit LUT index, so a single LUT lookup
can decode both at once. On text-like inputs where most codes are
short, this roughly doubles throughput compared to the one-symbol-
per-lookup variant.

The decoder runs in two phases to amortize a per-store alignment
check out of the hot loop. Each lane's output starts at an arbitrary
byte offset, so the first phase byte-drains the decoded stream until
the destination pointer is four-byte aligned. The cost is at most
three byte stores per lane. From there the second phase takes over
and writes four bytes per store unconditionally; alignment is
preserved because every store advances the write cursor by exactly
four.

The second phase also batches two LUT lookups per refill check. A
single decoded code is at most eleven bits, so two decodes consume
at most twenty-two bits. The refill loads four bytes (thirty-two
bits) into the buffer whenever the bit count drops below twenty-two,
which leaves the post-refill bit count at thirty-two or more, always
enough for both lookups. Three lookups would need up to thirty-three
bits, which a single thirty-two-bit refill cannot guarantee, so the
batch size is fixed at two. The combination of the dropped alignment
branch in the inner store and the halved refill check is the bulk of
the kernel's throughput on text-like inputs.

There is no warp synchronization inside the Huffman inner loop. The
prefix-sum that distributes per-lane input offsets is the last
cooperative step; after that each lane runs its own bit-buffer +
LUT loop with no cross-lane interaction. Lanes finish their streams
at different times (stream sizes are not uniform), so a barrier
inside the loop would stall fast lanes on slow ones for no
correctness benefit.

### Building the LUT

The LUT-build kernel turns the 128-byte code-length section at the
front of each Huffman body into a 1024-entry packed LUT. One warp
per Huffman body, all thirty-two lanes active. Bit-identical to the
reference encoder, verified across every sub-chunk of enwik8 and
silesia in `tools/huff_test/huff_lut_build_experiments.cu` against a
CPU oracle.

The earlier build sat on lane zero for two long serial passes - a
256-iter histogram + canonical-code assignment, then a 256-by-256
nested loop filling dual-symbol LUT entries - while the other
thirty-one lanes idled. NCU on that version showed it compute-bound
at 53% SM busy with "Fixed Latency Dependency" the top stall: the
signature of a long chain of dependent shared-memory writes on a
single lane. The redesign attacks that pattern in three places.

First, canonical-code assignment goes warp-parallel. The symbols are
processed as eight batches of thirty-two. Within each batch every
lane reads its symbol's code length, then `__match_any_sync` returns
a bitmask of the lanes in the batch that share the same length.
Popcount of that mask intersected with the lower-lane mask gives
each lane's intra-batch offset within its length bucket. A
shared-memory `per_L_base[L]` counter accumulates the per-batch
contributions; the lowest-numbered lane in each same-length group
(found via `__ffs(match_mask)`) atomic-adds the group size to that
counter so the next batch sees the new base. After eight batches
every lane in every batch has computed its symbol's canonical code
without a single serial lane-0 iteration.

Second, the inner loop of the dual-symbol pass iterates a
length-sorted used-symbol list instead of scanning all 256 symbols.
The histogram from the parallel canonical-code phase already
classifies symbols by length, so the build phase emits a
`used_pkd[]` array sorted by length ascending, each entry packing
length, canonical code, and symbol into one u32. A `len_end[L]`
array holds the bucket-end indices. The dual pass picks an outer s1
from this list, computes the maximum allowed L2 from s1's length
(`MAX_CODE_LEN − L1`), and runs the inner s2 loop bounded by
`len_end[MAX_CODE_LEN − L1]`. No length check inside the inner
loop, no skipped iterations on unused symbols, and the inner reads
are batched four at a time as uint4 loads (LDS.128). The earlier
nested-256 layout had every iteration paying a branch on `L2 == 0`
and another on `L1 + L2 > MAX_CODE_LEN`; both vanish here.

Third, the LUT is built in shared memory and bulk-dumped to global
at the end. The earlier kernel wrote each LUT entry directly to
global, which made the global-write pipeline the limiter (NCU
showed Long Scoreboard 3.27 cyc/inst, LG Throttle 1.16 on the
intermediate version that used wider shared-memory reads but still
wrote to global). Moving the LUT into shared (`__shared__ uint32_t
lut[1024]`, 4 KB) keeps Pass 1 and Pass 2's many small spans on the
~10× faster shared-memory path. At the end of the kernel the warp
copies the LUT to global with a coalesced uint4 loop using
`__stcs` - the decoder kernel that consumes the LUT runs in a
separate launch and may not be scheduled on this SM, so L1 caching
the bytes here would only pollute the cache.

Pass 1 (the single-symbol fan-out) is also folded into the
canonical-code-assignment phase. The moment a lane knows its
symbol's canonical code it writes both the `used_pkd[]` entry and
the Pass 1 LUT span; a single warp-pass produces both data
structures instead of two serial sweeps.

Pass 3 (the escape entries for length-eleven codes) stays serial on
lane zero. Length-eleven codes are typically zero to four per body
on natural text, so a 256-iter scan with a branch-predicted
`code_lengths[s] != MAX_CODE_LEN + 1` filter is cheap compared to
the cost of going parallel and synchronizing.

The kernel uses about 5.5 KB of static shared memory per block
(256 B code lengths, 1 KB used-symbol list, 4 KB LUT, plus three
small per-length arrays). At sm_89 that gives eighteen blocks per
SM resident - below the twenty-four block-count cap that the prior
build hit but, in practice, the workload-imbalance of canonical
Huffman at this scale already caps achieved occupancy at around 30%
either way. The win is in the dependent-instruction chains the new
algorithm shortens, not in occupancy.

End-to-end the new build kernel runs roughly four times faster than
the prior dense-iteration version on the canonical enwik8 workload
(measured per-kernel: 0.77 ms → 0.18 ms at L5; harness microbench
shows the same ratio at higher block counts). MIO Throttle drops
from 3.78 cyc/inst to 0.06; Long Scoreboard from 3.27 to 0.43. The
build kernel is now far enough off the critical path that the LZ
decode kernel is the next thing to shorten if that critical path
matters.

## The encode pipeline

```
   slzLzEncodeKernel        (one warp per chunk)
            |
            v
   slzHuffBuildTablesKernel ->  slzHuffEncode4StreamKernel
            |
            v
   slzAssembleMeasureKernel ->  slzAssembleWriteKernel
            |
            v
   slzFrameAssembleKernel
```

The LZ encode kernel does the heavy lifting. For levels one through
four it uses a warp-parallel greedy parser that probes hash tables
in parallel across all 32 lanes. For level five it switches to a
serial chain-hash lazy parser that runs entirely on lane zero;
the lazy parser explores deeper but cannot parallelize because each
hash chain step depends on the previous one.

The Huffman build kernel computes canonical-Huffman code lengths
from the per-sub-chunk histograms of literals, tokens, and off16
byte planes. The Huffman encode kernel then writes the chunk_type=4
wire body for each entropy-coded stream.

The assembly kernels turn the per-sub-chunk raw output into the
final wire frame. The measure kernel computes each sub-chunk's
assembled size so the host can prefix-sum them into output offsets.
The write kernel emits the per-sub-chunk header followed by the
payload. The frame-assemble kernel acts as a single manager block
that writes the frame's prefix bytes, the sub-chunk-boundary tail
prefix table, and the end mark.

Once all three assembly kernels have run, the encoded frame is
sitting in the output buffer on the device. A caller using the D2D
API receives a device pointer to that buffer with no PCIe traffic
on either side.

## The warp-parallel greedy parser

This is in `lz_greedy_parser.cuh`. The classic LZ77 task at each
position is to check whether there is a hash match, and if so, to
find the longest extension.

The serial answer is a loop. Hash the current bytes, look up the
hash bucket, walk the chain of previous positions that hashed to
the same bucket, check each one for a real match, extend the
longest one.

The parallel answer is to make 32 lanes do that work simultaneously,
each looking at a different starting position. The implementation
proceeds in five steps.

First, all 32 lanes use `__shfl_down_sync` rounds to build a
rolling 8-byte window. Lane N ends up with the 8 bytes starting at
input position (current + N). This construction uses only warp
shuffles, no extra global memory loads.

Second, each lane hashes its 8-byte value through two independent
hash functions, producing two candidate bucket indices. Two
independent hash families reduce the rate at which different
positions collide into the same bucket.

Third, the kernel uses `__match_any_sync`, a hardware warp-wide
associative match instruction. After the instruction, every lane
knows which other lanes in the warp hashed to the same bucket. This
finds intra-warp matches in one instruction; there is no memory
access at all.

Fourth, the kernel uses `__ballot_sync` to find which lanes actually
have a verified match (the hash bucket agrees and the actual bytes
at the matched position agree). The result is a 32-bit mask of
lanes with matches.

Fifth, the kernel uses `__clz` to find the highest-set lane in that
mask, then broadcasts that lane's match information to all lanes
with `__shfl_sync`. The warp then collectively extends the match
forward by checking 32 candidate extension bytes per iteration.

The net effect is that in one warp cycle the encoder probes 32
candidate positions against 32 hash entries, verifies the best
matches, and starts extending. A single-threaded CPU implementation
would need roughly 32 cycles to do the same work.

## The serial chain parser for level five

For level five, ratio matters more than encode speed. The kernel
switches to a serial chain-hash lazy parser that runs on lane zero
alone; lanes 1 through 31 idle.

The lazy parser explores up to eight candidate positions per scan
step (controlled by `CHAIN_MAX_STEPS`) and keeps looking even after
finding a match, in case the next position offers a better one.

This is intentionally serial. Chain walking is data-dependent: each
step reads `next_hash[i]` to learn where to look next, and the
walk cannot be parallelized because each step's address depends on
the previous step's result.

The level-five chain parser trades encode throughput for ratio. On
enwik8, level four takes 112 milliseconds and produces a 41.9%
ratio; level five takes 296 milliseconds and produces a 38.9% ratio.
The 2.6x slowdown on encode buys three percentage points of ratio.
Decode time is identical at both levels; the wire format is the same.

## Per-chunk hash tables in global memory

Each chunk gets its own hash tables, all in global memory. The
encoding warp for a given chunk has exclusive write access to its
slice of the tables. There are no atomics, no contention, and no
cross-chunk coordination.

The tables are sized:

```
   hash_table_first[chunk]   primary chain head, hash_size u32 entries
   hash_table_long[chunk]    secondary direct-mapped, hash_size u32 entries
   next_hash[chunk]          chain links, NEXT_HASH_SIZE u16 entries
```

The chunks live at strided offsets, one stride per chunk index. This
is wasteful in absolute memory (a full set of hash tables per
chunk), but it has two virtues. The first is that all of the writes
and reads for a single chunk's encoding fit comfortably in the L1
cache, so the encoder never sees memory latency on hash probes.
The second is that the chunks are independent: any warp can grab
any unfinished chunk, and there is no scheduling complexity.

## Wire-format choices that exist to enable GPU parallelism

A few wire-format decisions exist specifically to make GPU decode
fast. None of them would matter on a CPU implementation.

Sub-chunks are completely independent decode units. Each sub-chunk
has its own full set of streams: literals, tokens, off16 offsets,
off32 offsets, lengths. A warp can start decoding a sub-chunk
without looking at any neighboring sub-chunk. This is what allows
the codec to map each sub-chunk onto its own warp.

The offset streams are split into off16 (offsets that fit in 16
bits, the common case) and off32 (longer offsets). Within a
sub-chunk, all the off16 values live in one contiguous block and
all the off32 values live in another. The decoder loads offset
bytes with predictable strides instead of parsing variable-length
entries inline with tokens.

The Huffman wire format puts code lengths up front, followed by
31 sub-stream-size headers and 32 sub-streams. The code-length
section is exactly 128 bytes of packed nibbles. The sub-stream-size
section is 31 × 3 = 93 bytes of 24-bit little-endian integers
(the 32nd size is derived from `in_size - 128 - 93 - sum`). The
LUT-build kernel reads the code lengths directly from the
compressed buffer; the decode kernel reads the sub-stream sizes
into a warp-shuffle prefix sum to compute each lane's input offset.

The off16 stream uses a sentinel value to disambiguate raw from
entropy-coded form. A leading count field of 0xFFFF means the
stream is entropy-coded as two split byte planes (hi byte plane
followed by lo byte plane). Anything else is the literal entry
count and the stream is raw. One sentinel value avoids needing an
extra format flag, and the decoder branches on it with zero
overhead.

Every sub-chunk starts with a three-byte big-endian header. The top
bit is set if the sub-chunk is LZ-compressed; the next four bits
are the decode mode; the bottom 19 bits are the compressed payload
size. Big-endian because the parser reads it with one
`readBE24` and tests the LZ flag with a single AND. Nineteen bits
of compressed size gives up to 512 KB per sub-chunk, with plenty
of headroom over the actual 128 KB cap.

## Wire-format constants in one place

Every value that the encoder writes and the decoder reads lives in
`common/gpu_wire_format.cuh`. The encode side includes that header.
The decode side includes that header. There is no second copy. A
short selection (see the file itself for the complete current list):

```cpp
static constexpr uint32_t LZ_BLOCK_SIZE              = 0x10000u;
static constexpr int      MAX_BLOCKS_PER_SUBCHUNK    = 2;
static constexpr uint32_t INITIAL_LITERAL_COPY_BYTES = 8;
static constexpr int32_t  INITIAL_RECENT_OFFSET      = -8;
static constexpr uint8_t  HUFF_CHUNK_TYPE            = 4;
static constexpr uint32_t SUBCHUNK_HDR_BYTES         = 3;
static constexpr uint32_t SUBCHUNK_LZ_FLAG_BIT       = 0x800000u;
static constexpr uint32_t OFF16_ENTROPY_MARKER       = 0xFFFFu;
static constexpr uint32_t OFF32_COUNT_FIELD_BITS     = 12;
static constexpr uint8_t  OFF32_LONG_ENTRY_TAG       = 0xC0;
```

The encoder uses `SUBCHUNK_LZ_FLAG_BIT` to set the flag in its
output. The decoder uses the same constant to test the flag in its
input. If you change the value here, both sides see the change at
the next compile, and format drift between encode and decode is
impossible.

## The pure-D2D pipeline

Earlier versions of the GPU codec bounced data through the host at
three points: the input frame for encode, the output bytes for
decode, and several descriptor arrays during the decode walk. Each
bounce is one PCIe round trip. At PCIe 4.0 x8 (about 14 GB/s
practical), a 100 MB bounce costs about 7 milliseconds each
direction.

The cleanup pass replaced the entire descriptor orchestration with
on-device kernels. The walk kernel parses the frame header on the
device. The scan kernel runs on the device. The prefix-sum kernel
runs on the device. The compact kernels run on the device. The
merge kernel runs on the device. The gather kernel copies raw
off16 bytes inside the device.

After all that, the host reads back about 56 bytes of
launch-plumbing counters per decompress (the sub-chunk count, the
total Huffman descriptor count, the raw off16 descriptor count).
That readback is unavoidable because the next kernel's grid size
depends on those counts and CUDA does not allow a kernel to launch
itself with a runtime-determined grid.

For callers using the D2D entry points (`slzCompress` and
`slzDecompress` taking device pointers), this means zero PCIe
traffic on the data path. The benchmark numbers in the README's
"D2D wall-clock" column are the full GPU wall-clock for such
callers. There is no upload, no download, no host bounce. The
decode kernel reads from device memory, writes to device memory,
and the caller receives a device pointer to the result.

## Resource utilization

Register, stack, and shared-memory usage per kernel, compiled with
`nvcc -arch=sm_89 -O3`. (2026-06-10 snapshot: predates the v4 #15
pipelined kernels and the v4 #19 hash kernels; regenerate with
`tools\build_gpu.bat` for current numbers. The pipelined LZ kernels
additionally hold a ~2.6 KB PipeBatch double buffer plus handoff
flags in static shared memory.)

| Kernel | REG | STACK | SHARED |
|---|---:|---:|---:|
| `slzLzEncodeKernel`            | 72 | 144 | 0 |
| `slzLzDecodeKernel`            | 40 | 192 | 0 |
| `slzLzDecodeRawKernel`         | 40 | 72 | 0 |
| `slzWalkFrameKernel`           | 34 | 0 | 0 |
| `slzScanParseKernel`           | 40 | 0 | 0 |
| `slzPrefixSumChunksKernel`     | 19 | 0 | 0 |
| `slzMergeHuffDescsKernel`      | 40 | 0 | 0 |
| `slzCompactRawDescsKernel`     | 40 | 0 | 0 |
| `slzCompactHuffDescsKernel`    | 40 | 0 | 0 |
| `slzGatherRawOff16Kernel`      | 12 | 0 | 0 |
| `slzHuffBuildLutKernel`        | 40 | 0 | 5536 |
| `slzHuffDecode4StreamKernel` †  | 40 | 0 | 0 (+4096 dyn) |
| `slzHuffBuildTablesKernel`     | 50 | 0 | 9472 |
| `slzHuffEncode4StreamKernel` †  | 56 | 128 | 0 |
| `slzAssembleMeasureKernel`     | 26 | 0 | 0 |
| `slzAssembleWriteKernel`       | 46 | 0 | 0 |
| `slzFrameAssembleKernel`       | 26 | 0 | 0 |

Two things stand out. Most kernels use little or no static shared
memory. The two Huffman build kernels are the exceptions: the
decode-side LUT-build holds a 4 KB working LUT plus a length-sorted
used-symbol list and the per-length bucket arrays (5.5 KB total) so
the bulk of its writes stay on the fast shared path before a final
coalesced uint4 dump to global; the encode-side
`slzHuffBuildTablesKernel` stages 9.5 KB for the histogram, tree
nodes, and code tables. The Huffman decode kernel takes 4 KB of
dynamic shared per launch for the runtime LUT (allocated at kernel
launch, not counted in the SHARED column above). The single-warp LZ
decode kernels keep their working set in registers; the pipelined
K=4 kernels add the shared PipeBatch double buffer that the parser
and copier team hand batches through. The entropy scratch lives in
global memory and the hardware caches absorb the traffic.

† The `4Stream` suffix on the two Huffman kernel names is retained
for the Zig dispatch ABI introduced by the prior 4-stream design.
Both kernels now operate on `HUFF_NUM_STREAMS = 32` streams (one
per warp lane), not 4. See `common/gpu_huffman.cuh` for the
constants and the kernel banners in the corresponding `.cu` files
for the wire-format details.

The orchestration kernels are tiny. Walk, scan, compact, merge,
prefix-sum, and gather each use between 12 and 40 registers and no
stack at all. This was a deliberate choice. These kernels run
back-to-back on the same stream as the much heavier LZ decode
kernel, and a register-heavy orchestration kernel would steal SM
occupancy from the kernels that actually need it.

## What this codec deliberately does not do

A few things you might expect to find are not here, and each
absence is a choice.

There is no use of the hardware decompression block. Some chips
expose a dedicated LZ4 or Zstandard decompression engine accessible
through `CUmemDecompress` or a similar API. This codec does not
use it. The kernels documented here are pure compute. The cost of
that choice is that the codec does not get the dedicated hardware's
sometimes-faster throughput on its supported formats; the benefit
is portability across any sm_89-capable device, and the start of a
Vulkan compute-shader port for non-NVIDIA hardware.

There is no shared-memory output buffer. The decode kernel writes
output bytes directly to global memory. A shared-memory staging
buffer would lower the latency of self-referential match copies,
but it would also consume shared memory that the kernels cannot
spare without dropping occupancy. The hardware L1 and L2 caches
absorb the back-reference traffic fine.

There is no speculative parallel token parsing. The parallel-parse
fast path parses 32 tokens in parallel, but only when every token
in the batch is short. It does not speculatively decode long
tokens in parallel and roll back on mispredictions. Speculation
would add branch-misprediction overhead that exceeds the gain on
the rare long-token cases.

(One absence that used to be on this list is gone: as of v4 #19 the
codec DOES verify its own output - per-chunk hashes computed on
device, rolled into a 4-byte Merkle root in the frame; see FORMAT.md
and the README's Integrity section.)

There is no persistent-thread scheduling. Each kernel is a one-shot
launch that exits when its work is done. There is no thread-pool
or work-stealing layer. The grid-of-blocks model is sufficient.
Every chunk maps to exactly one block, and the GPU's hardware
scheduler handles the assignment.

## Kernel inventory

(Migrated from the retired docs/GPU_README.md, 2026-06-10.) Every
kernel is resolved by the Zig drivers via `cuModuleGetFunction`; all
symbols carry the `slz` prefix.

| Direction | Kernel | Source TU |
|-----------|--------|-----------|
| encode | `slzLzEncodeKernel` | `encode/lz_kernel.cu` |
| encode | `slzHuffBuildTablesKernel` | `encode/huffman_kernel.cu` |
| encode | `slzHuffEncode4StreamKernel` | `encode/huffman_kernel.cu` |
| encode | `slzAssembleMeasureKernel` | `encode/assemble_kernel.cu` |
| encode | `slzAssembleWriteKernel` | `encode/assemble_kernel.cu` |
| encode | `slzFrameAssembleKernel` | `encode/assemble_kernel.cu` |
| decode | `slzLzDecodeGeneralPipelinedKernel` (hot path: K=4 pipeline, L3+) | `decode/lz_kernel.cu` |
| decode | `slzLzDecodeRawPipelinedKernel` (hot path: K=4 pipeline, L1/L2) | `decode/lz_kernel.cu` |
| decode | `slzLzDecodeKernel` (fallback: SLZ_NO_PIPELINE) | `decode/lz_kernel.cu` |
| decode | `slzLzDecodeRawKernel` (fallback: SLZ_NO_PIPELINE) | `decode/lz_kernel.cu` |
| decode | `slzWalkFrameKernel` | `decode/lz_kernel.cu` |
| decode | `slzPrefixSumChunksKernel` | `decode/lz_kernel.cu` |
| decode | `slzScanParseKernel` | `decode/lz_kernel.cu` |
| decode | `slzCompactAllDescsKernel` (hot path: fused ×5) | `decode/lz_kernel.cu` |
| decode | `slzCompactHuffDescsKernel` (legacy/fallback) | `decode/lz_kernel.cu` |
| decode | `slzCompactRawDescsKernel` (legacy/fallback) | `decode/lz_kernel.cu` |
| decode | `slzMergeHuffDescsParKernel` (hot path: 4-block) | `decode/lz_kernel.cu` |
| decode | `slzMergeHuffDescsKernel` (legacy/fallback) | `decode/lz_kernel.cu` |
| decode | `slzGatherRawOff16Kernel` | `decode/lz_kernel.cu` |
| decode | `slzHuffBuildLutKernel` | `decode/huffman_kernel.cu` |
| decode | `slzHuffDecode4StreamKernel` | `decode/huffman_kernel.cu` |
| both | `slzSegHashKernel` (v4 #19: per-1 KiB-segment XXH32) | both `lz_kernel.cu` TUs |
| both | `slzChunkCombineKernel` (v4 #19: segment hashes -> chunk hash) | both `lz_kernel.cu` TUs |
| decode | `slzScPrefixApplyKernel` (v4 #19: SC prefixes -> output, on device) | `decode/lz_kernel.cu` |
| decode | `slzMerkleVerdictKernel` (v4 #19: root compare, 4-byte verdict) | `decode/lz_kernel.cu` |
| encode | `slzMerkleRootWriteKernel` (v4 #19: root trailer into the device frame) | `encode/lz_kernel.cu` |

The `4Stream` suffix on the Huffman kernels is retained for the Zig
dispatch ABI; both sides actually operate on `HUFF_NUM_STREAMS = 32`
streams (one per warp lane). The `.cuh` files are a size-only split,
`#include`d into the single per-direction aggregator `.cu` - only the
`.cu` emits a `.ptx`. Per-kernel REG/STACK/SHARED numbers come from
`tools\build_gpu.bat`'s cuobjdump res-usage printout; live per-kernel
timings from `SLZ_PROFILE_DECODE=1` / `SLZ_PROFILE_PHASES=1`.
