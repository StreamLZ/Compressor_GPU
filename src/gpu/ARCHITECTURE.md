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
rather than 50. Token parsing remains roughly serial, but it now
runs in a small fraction of the warp time because the byte-copy
phase is so much faster.

That is the entire idea. Everything else in this document is the
mechanical work of making it actually fast.

## How the work is mapped to GPU hardware

The codec settles on a few mapping decisions and applies them
consistently.

One chunk is one warp. A chunk holds at most 256 KB of decompressed
output, partitioned into one or more sub-chunks of at most 128 KB
each (default 64 KB). The warp owns the chunk end to end. There is no
cross-warp coordination during the decode loop.

Two warps are packed into one CUDA block (`WARPS_PER_BLOCK = 2`). The
two warps in a block do not cooperate; they decode independent
chunks. Packing two of them together amortizes the per-block launch
and scheduling overhead, but each is doing its own thing.

For Huffman decoding, the encoder has already split the input into
four sub-streams so the decoder can run four lanes in parallel.
`slzHuffDecode4StreamKernel` uses one warp per Huffman block, but
only lanes 0 through 3 are active during the inner decode loop. The
remaining 28 lanes return immediately and consume zero register
state. This trades total occupancy for clean register usage; the
active lanes get a larger register budget for their hot loop.

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
is a fifty-six-byte readback of launch-plumbing counters used to
size the next kernel's grid.

```
   slzWalkFrameKernel
            |
            v
   slzPrefixSumChunksKernel
            |
            v
   slzScanParseKernel  ->  slzCompactHuffDescsKernel
                       ->  slzCompactRawDescsKernel
                       ->  slzMergeHuffDescsKernel
            |
            v
   slzGatherRawOff16Kernel
            |
            v
   slzHuffBuildLutKernel  ->  slzHuffDecode4StreamKernel
            |
            v
   slzLzDecodeKernel  (or slzLzDecodeRawKernel for L1/L2)
```

The walk kernel parses the frame header on the device and emits a
descriptor for every chunk. The prefix-sum gives every chunk its
starting sub-chunk index so later kernels can compute scratch
offsets without a second pass. The scan kernel inspects every
sub-chunk's stream headers and emits a tentative descriptor per
entropy-coded or raw stream. The compact and merge kernels filter
those tentative descriptors into the final per-stream-type arrays
the entropy kernel will consume. The gather kernel copies raw
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
treats both launches as one timing window. In practice the Huffman
pre-decode and the LZ decode each take roughly half the GPU wall
clock at L3 and above.

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

The measured impact was a 17 to 25 percent reduction in D2D
wall-clock across every compression level and both test corpora.
The enwik8 L1 decode dropped from 3.8 milliseconds to 2.92
milliseconds, a 23 percent gain on its own.

## Warp-cooperative byte copies

After the token is parsed, the warp cooperates on the actual byte
movement. Two helper functions in `lz_decode_core.cuh` do almost all
of the byte-copy work in the codec.

The literal copy is straightforward. All 32 lanes participate, each
copying one byte at a stride of WARP_SIZE until the run is
exhausted. A literal of 50 bytes takes two warp iterations: lanes 0
through 31 copy bytes 0 through 31 in the first iteration, then
lanes 0 through 17 copy bytes 32 through 49 in the second. The
remaining 14 lanes idle for that second iteration.

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

The keep-out attribute means the compiler allocates registers for the
parser at its own entry point, lets the parser finish and return,
and then enters the decoder with a fresh smaller register set.

This was confirmed empirically. Removing `__noinline__` and rebuilding
moved the LZ decode kernel from 40 registers per thread to 56,
which crossed an occupancy threshold and dropped measured throughput
by roughly 8 percent. The attribute stays.

## The four-stream Huffman decoder

A canonical-Huffman block in this codec consists of code-length
information, followed by four independent compressed sub-streams.
The four streams are interleaved at encode time so the decoder can
process them in parallel: lane 0 of the warp decodes sub-stream 0,
lane 1 decodes sub-stream 1, lane 2 decodes sub-stream 2, lane 3
decodes sub-stream 3. Lanes 4 through 31 return immediately at the
start of the kernel and consume no further resources.

Each active lane runs the same tight inner loop. It refills a 64-bit
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

There is no warp synchronization inside the Huffman inner loop. The
four active lanes work entirely independently; the decoder
explicitly forbids adding a `__syncwarp` below the early-return,
because lanes 4 through 31 are no longer alive to participate in a
warp barrier and the barrier would hang.

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
three sub-stream-size headers and four sub-streams. The code-length
section is exactly 128 bytes of packed nibbles. The sub-stream-size
section is exactly nine bytes (three 24-bit little-endian integers,
since the fourth size is derived from the total). Both are loadable
in one coalesced 32-byte load, and the LUT-build kernel reads them
directly from the compressed buffer without an intermediate copy.

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
The decode side includes that header. There is no second copy.

```cpp
static constexpr uint32_t LZ_BLOCK_SIZE              = 0x10000u;
static constexpr int      MAX_BLOCKS_PER_SUBCHUNK    = 2;
static constexpr uint32_t INITIAL_LITERAL_COPY_BYTES = 8;
static constexpr int32_t  INITIAL_RECENT_OFFSET      = -8;
static constexpr uint8_t  HUFF_CHUNK_TYPE            = 4;
static constexpr uint32_t SUBCHUNK_HDR_BYTES         = 3;
static constexpr uint32_t SUBCHUNK_LZ_FLAG_BIT       = 0x800000u;
static constexpr uint32_t SUBCHUNK_MODE_SHIFT        = 19;
static constexpr uint32_t SUBCHUNK_COMP_SIZE_MASK    = 0x7FFFFu;
static constexpr uint32_t OFF16_ENTROPY_MARKER       = 0xFFFFu;
static constexpr uint32_t OFF32_COUNT_FIELD_BITS     = 12;
static constexpr uint32_t OFF32_COUNT_PACK_MAX       = (1u << OFF32_COUNT_FIELD_BITS) - 1;
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
device. The prefix-sum kernel runs on the device. The scan kernel
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

Final register, stack, and shared-memory usage for every kernel,
compiled with `nvcc -arch=sm_89 -O3`:

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
| `slzHuffBuildLutKernel`        | 40 | 104 | 1280 |
| `slzHuffDecode4StreamKernel`   | 40 | 0 | 0 |
| `slzHuffBuildTablesKernel`     | 50 | 0 | 9472 |
| `slzHuffEncode4StreamKernel`   | 38 | 0 | 0 |
| `slzAssembleMeasureKernel`     | 26 | 0 | 0 |
| `slzAssembleWriteKernel`       | 46 | 0 | 0 |
| `slzFrameAssembleKernel`       | 26 | 0 | 0 |

Two things stand out. Almost no kernel uses shared memory. Only the
Huffman LUT-build kernels use it, and only because the LUT itself
fits there comfortably. The LZ decode kernel keeps its entire
working set in registers; the entropy scratch lives in global memory
and the hardware caches absorb the traffic.

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

There is no persistent-thread scheduling. Each kernel is a one-shot
launch that exits when its work is done. There is no thread-pool
or work-stealing layer. The grid-of-blocks model is sufficient.
Every chunk maps to exactly one block, and the GPU's hardware
scheduler handles the assignment.
