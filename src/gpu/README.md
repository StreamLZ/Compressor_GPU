# StreamLZ GPU module

GPU-accelerated compression and decompression for the StreamLZ Fast codec.
CUDA kernels (NVIDIA Driver API, target `sm_89`) plus thin Zig drivers that
load them. Entropy coding is canonical Huffman (`chunk_type=4`); GPU frames
contain only raw and Huffman sub-chunks.

This folder is **standalone**. Nothing under `src/gpu/` imports anything
outside `src/gpu/`.

## Layout

```
common/                         shared device headers (#include'd by every kernel)
  gpu_warp.cuh                  warp / lane geometry, bit-width constants
  gpu_byteio.cuh                little / big-endian load + store primitives
  gpu_huffman.cuh               canonical-Huffman wire format: constants,
                                canonical-code build, weights + LUT pack/unpack
  gpu_wire_format.cuh           encode/decode-shared LZ wire constants
                                (LZ_BLOCK_SIZE, SUBCHUNK_LZ_FLAG_BIT, …)

decode/                         GPU decode
  driver.zig                    thin facade; re-exports the public API
  cuda_api.zig                  nvcuda.dll handle, CU* typedefs, cu*_fn slots,
                                qpcNow / qpcMs, NUM_PIPELINE_STREAMS
  module_loader.zig             PTX load, kernel-handle pub vars, init /
                                isAvailable / ensurePipelineStreams
  descriptors.zig               POD descriptor types + GpuError + cudaCall +
                                HUFF_LUT_ENTRIES, WALK_MAX_CHUNKS
  decode_context.zig            DecodeContext + ensure/alloc/copy helpers
                                + kernel-event profiling plumbing
  scan_host.zig                 CPU scanForEntropyChunks (H2D-bounce entry path
                                + fallback when the GPU scan kernels are absent)
  scan_gpu.zig                  GPU walk / prefix-sum / scan / compact
                                (pure-D2D entry; CPU has no readable copy)
  decode_dispatch.zig           fullGpuLaunchImpl + per-decode
                                helpers (dumpScanIfRequested, emitE2eTrace,
                                gatherRawOff16, mergeHuffDescs)
  lz_kernel.cu                  LZ decode aggregator; #includes:
    slz_wire_format.cuh         decoder-private token formats + parsers
    lz_decode_core.cuh          warp-cooperative literal / match copy helpers
    lz_decode_raw.cuh           slzLzDecodeRawKernel inner loop (no-entropy fast path)
    lz_decode_general.cuh       slzLzDecodeKernel inner loop (entropy + delta-literals)
    lz_dispatch.cuh             parseAndDecodeSubChunk + raw variant
    lz_header_parse.cuh         per-stream sub-chunk header parsers
    lz_decode_kernels.cuh       slzLzDecodeKernel + slzLzDecodeRawKernel
    walk_frame_kernel.cuh       slzWalkFrameKernel + SLZ_FRAME_* constants
    prefix_sum_chunks_kernel.cuh slzPrefixSumChunksKernel
    scan_parse_kernel.cuh       slzScanParseKernel + scan helpers
    compact_descs_kernels.cuh   slzCompact{Huff,Raw}DescsKernel + staged descs
    merge_huff_descs_kernel.cuh slzMergeHuffDescsKernel
    gather_raw_off16_kernel.cuh slzGatherRawOff16Kernel
  huffman_kernel.cu             slzHuffBuildLutKernel + slzHuffDecode4StreamKernel

encode/                         GPU encode
  driver.zig                    thin facade
  cuda_ffi.zig                  CUDA Driver API FFI
  module_loader.zig             PTX load + kernel handles + init
  encode_context.zig            EncodeContext + ABI structs + ensureBuf
  levels.zig                    hashBitsForLevel / useGlobalHash / useChainParser
  encode_lz.zig                 gpuCompressImpl (LZ launcher)
  encode_huff.zig               gpuEncodeHuffImpl + per-stream {Literals,Tokens,Off16}
  encode_assemble.zig           gpuAssembleFrameImpl + gpuFrameAssembleImpl
  lz_kernel.cu                  LZ encode aggregator; #includes:
    lz_format.cuh               encode-side hash constants + helpers
    lz_token_emit.cuh           token / offset / length stream serializer
    lz_greedy_parser.cuh        warp-parallel greedy parser (L1-L4)
    lz_chain_parser.cuh         serial chain-hash lazy parser (L5+)
  huffman_kernel.cu             slzHuffBuildTablesKernel + slzHuffEncode4StreamKernel
  assemble_kernel.cu            slzAssembleMeasureKernel + slzAssembleWriteKernel
                                + slzFrameAssembleKernel (Phase-3 pure-D2D encode)
```

## Pipeline

**Encode.** `lz_kernel.cu` finds matches and emits raw sub-streams
(literals / tokens / off16 / off32 / lengths); `huffman_kernel.cu` then
entropy-codes the literal, token and off16 streams into `chunk_type=4`
bodies; `assemble_kernel.cu` glues the per-sub-chunk wire format and
writes the final frame for the pure-D2D path.

**Decode.** `huffman_kernel.cu` pre-decodes the Huffman streams into a
scratch buffer; `lz_kernel.cu` then runs the LZ decode, reading the
pre-decoded streams and resolving matches.

### Kernel entry points

Every CUDA kernel is resolved by the Zig drivers via `cuModuleGetFunction`.

| Direction | Kernel | Source |
|-----------|--------|--------|
| encode | `slzLzEncodeKernel` | `encode/lz_kernel.cu` |
| encode | `slzHuffBuildTablesKernel` | `encode/huffman_kernel.cu` |
| encode | `slzHuffEncode4StreamKernel` | `encode/huffman_kernel.cu` |
| encode | `slzAssembleMeasureKernel` | `encode/assemble_kernel.cu` |
| encode | `slzAssembleWriteKernel` | `encode/assemble_kernel.cu` |
| encode | `slzFrameAssembleKernel` | `encode/assemble_kernel.cu` |
| decode | `slzLzDecodeKernel` | `decode/lz_kernel.cu` |
| decode | `slzLzDecodeRawKernel` | `decode/lz_kernel.cu` |
| decode | `slzWalkFrameKernel` | `decode/lz_kernel.cu` |
| decode | `slzPrefixSumChunksKernel` | `decode/lz_kernel.cu` |
| decode | `slzScanParseKernel` | `decode/lz_kernel.cu` |
| decode | `slzCompactHuffDescsKernel` | `decode/lz_kernel.cu` |
| decode | `slzCompactRawDescsKernel` | `decode/lz_kernel.cu` |
| decode | `slzMergeHuffDescsKernel` | `decode/lz_kernel.cu` |
| decode | `slzGatherRawOff16Kernel` | `decode/lz_kernel.cu` |
| decode | `slzHuffBuildLutKernel` | `decode/huffman_kernel.cu` |
| decode | `slzHuffDecode4StreamKernel` | `decode/huffman_kernel.cu` |

The `4Stream` suffix on the two Huffman kernels is retained for the Zig
dispatch ABI; both encode and decode now operate on `HUFF_NUM_STREAMS = 32`
streams (one per warp lane), not 4. See `ARCHITECTURE.md` and the kernel
banners in the corresponding `.cu` files for the wire-format details.


Decode LZ-aggregator kernels live in their per-kernel `.cuh` headers
(see the layout block above) which the single `decode/lz_kernel.cu`
aggregator includes; only the `.cu` emits a `.ptx`, hence the source
column.

## Build

CUDA kernels compile to PTX/CUBIN out-of-band (they need `nvcc` + the MSVC
toolchain, which `zig build` does not invoke):

```
tools/build_gpu.bat       compiles all five .cu files under src/gpu/
                          (decode/{lz,huffman}_kernel.cu and
                          encode/{lz,huffman,assemble}_kernel.cu) to
                          .ptx + .cubin via `nvcc -arch=sm_89 -O3`,
                          then runs `zig build -Doptimize=ReleaseFast
                          -Dgpu=true` so the freshly-built PTX is
                          embedded into streamlz.exe / streamlz_gpu.dll.
```

The generated `.ptx` is committed; the Zig drivers `@embedFile` it, so
a plain `zig build` does not require CUDA to be installed.

```
zig build -Doptimize=ReleaseFast -Dgpu=true   streamlz CLI with GPU paths
zig build gpulib                              streamlz_gpu.dll (nvCOMP-style C API)
```

`nvcuda.dll` is loaded at runtime; if CUDA is unavailable the codec falls
back to the CPU path.

## Performance

Measured 2026-05-27 on an RTX 4060 Ti (`sm_89`), 32-stream Huffman in
the BIL (bounded-interleaved) wire format with the warp-cooperative
LUT-build kernel (commit `ba46e9a`) and the L4/L5 hash-bits=17 fix
that keeps the chain hash table resident in VRAM (commit pending).
`-gpu` mode (64 KB sub-chunks default; sc=0.5 above the
`sm_count × 48 × 128 KB` GPU-saturation threshold). Corpora:
enwik8 (100 MB text), silesia (212.8 MB mixed). Decode figures are
best-of-30 (`streamlz -db -r 30`, sequential per the
no-parallel-benchmarks rule); all times in milliseconds.

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.7% | 38.0% |
| L4 | 42.7% | 37.5% |
| L5 | 39.6% | 33.9% |

L4 ratio rounds up by 0.1 pp on both files after the hash_bits=17 change
(L4 enwik8 42.66% → rounds to 42.7%; L4 silesia 37.47% → rounds to 37.5%);
L5 raw bytes shift by < 0.1 % which is invisible at this rounding.

L1–L2 are LZ-only (no entropy stage); L3–L5 add GPU Huffman. The BIL
Huffman wire format costs ~0.3 pp ratio over the prior concat layout
at 64 KB sub-chunks (sub-header +3 B for the 32nd stream size, +4 B
for K, plus ≤3 B/stream rounding to a 4-byte BIL word) in exchange
for the coalesced 128-byte sector refill that makes the 32-stream
Huffman decode memory-bandwidth-bound rather than scatter-latency-bound.

### Decode (ms): D2D wall-clock and end-to-end

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | **2.91** / 15.57 | **5.07** / 30.07 |
| L2 | **2.93** / 15.59 | **5.09** / 30.11 |
| L3 | **4.09** / 15.66 | **7.15** / 30.84 |
| L4 | **3.93** / 15.42 | **6.94** / 30.65 |
| L5 | **4.12** / 15.39 | **7.72** / 30.76 |

(Decode unchanged within noise vs the pre-hash_bits=17 baseline — the
encode-side hash-table size change has no effect on decode kernels.)

**D2D wall-clock** = the time a caller of `slzDecompress` with device-
resident input and output sees on the wire. Measured via `cudaEventRecord`
pairs around the GPU kernel launches (LZ + Huffman for L3+ are pipelined
into one timing window). No other GPU work happens around the kernels in
the D2D entry points, so this number is the full GPU wall-clock for users
of the D2D API (game-asset / LLM-context / ML pipelines). **e2e** adds
the H2D upload of the compressed frame + D2H download of the decompressed
output for users on the host-bounce path.

Notable: at L3-L5 the D2D wall-clock dropped 11-21% from the pre-`ba46e9a`
BIL baseline (enwik8 L3 4.58 → 4.09 ms, silesia L3 8.26 → 7.12 ms, etc.).
The lift comes from the warp-cooperative `slzHuffBuildLutKernel` rewrite:
the prior code was symbol-parallel with a serial inner-span fill (one
lane could write up to 1024 LUT entries serially while 31 sat idle); the
new code uses `__match_any_sync` for parallel code assignment + uint4
wide-store fill + length-sorted `used_pkd[]` to eliminate Pass-2
out-of-bounds iterations. Silesia gains more than enwik8 (~13-14% vs
~11%) because its varied code-length distributions hit the old serial
fill harder.

The 32-stream Huffman pre-decode (L3+) now lands well under half the
LZ kernel time. LZ kernel time is unchanged across this rewrite. The
D2H copy of the decompressed output remains the largest e2e cost for
host-output callers; for D2D callers using `slzDecompressAsync` it
drops to zero (the LZ kernel writes straight into the caller's device
buffer — see commit `0440532`).

### Encode (ms): GPU LZ-encode kernel

`last_kernel_ns` covers the LZ-encode kernel only (the Huffman-encode
kernels run as a separate pass and are not included).

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 |  70 |  132 |
| L2 |  77 |  151 |
| L3 |  84 |  155 |
| L4 |  84 |  151 |
| L5 | 236 |  412 |

L4 and L5 now use `hash_bits=17` (was 20). At hb=20 the chain parser's
per-chunk hash was ~8 MB; for silesia at sc=0.25 (3248 chunks) this
allocated ~26 GB and spilled to system RAM via PCIe on consumer GPUs
(16 GB VRAM), producing the prior 20.8 s L5 silesia outlier. hb=17
keeps per-chunk hash at ~1 MB which fits in VRAM for any reasonable
input size. **silesia L5 encode: 20811 → 412 ms (~50× faster);
enwik8 L5: 283 → 236 ms (-17%); L4 silesia 203 → 151 ms (-26%);
L4 enwik8 108 → 84 ms (-22%).** Ratio cost is +0.1 pp at L4 (rounding)
and < 0.07 % raw at L5 (essentially invisible after Huffman).

### vs nvCOMP (enwik8, 100 MB)

nvCOMP 5.2.0, same RTX 4060 Ti. (nvCOMP's hardware decompression engine
is Hopper or Blackwell only, not available on Ada, so only the CUDA
decode path applies here.)

#### Methodology

Three measurement windows are reported because they answer different
questions, and confusing them is easy. Earlier revisions of this section
compared StreamLZ's narrow "inner decode kernels" timer against
nvCOMP's whole-call event bracket, which inflated the reported win.
The numbers below are apples-to-apples within each row.

| Window | What it measures | StreamLZ source | nvCOMP source |
|--------|------------------|-----------------|---------------|
| **Pipeline kernel-sum** | Σ `cudaEventElapsedTime` across every kernel in the decode pipeline. Excludes memcpys, scheduler gaps, event overhead. | `bench_d2d` "kernel active best" | `nsys stats --report=cuda_gpu_kern_sum` over the 20 timing iterations |
| **Async call wall** | cuEvent pair on the caller's stream around the *whole* decompress API call. Includes any internal memcpys + per-kernel event-record overhead. What an async caller actually waits for. | `bench_d2d` "gpu kernel best" | `nvcomp_bench3` "decode kernel best" |
| **End-to-end host wall** | `chrono` wall around `H2D(compressed) + decompress + D2H(output)` with pageable host input/output. Real cost when payload starts and ends on host. | `bench_all` (`streamlz.exe -db`) "e2e best" | `nvcomp_bench3` "decode e2e best" |

Ground truth for per-kernel breakdowns comes from **Nsight Systems 2025.5**:

```
nsys profile --trace=cuda --sample=none --output=<rep> <bench.exe> [args]
nsys stats --report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum <rep>.nsys-rep
```

For nvCOMP LZ4 this surfaces one kernel (`lz4DecompressBatchKernel`)
per decompress call — its time *is* the kernel sum. For nvCOMP Zstd
it surfaces five (`zstd::decompression_kernel`, `init_fse_tables`,
`init_huff_tables`, `classify_frames`, `init_buffer_vals`); their sum
matches nvCOMP's reported "decode kernel" measurement to within noise,
which confirms nvCOMP's event bracket is effectively a kernel-sum (no
hidden internal memcpys).

For StreamLZ nsys surfaces ~10 kernels per call (`slzWalkFrame*`,
`slzPrefixSumChunks*`, `slzScanParse*`, four `slzCompactHuffDescs*` +
`slzCompactRawDescs*`, `slzGatherRawOff16*`, `slzMergeHuffDescs*`,
`slzHuffBuildLut*` + `slzHuffDecode4Stream*` for L3+,
`slzLzDecode*`). The async-call wall exceeds the kernel-sum by ~0.6 ms
on L1 — that's the front-half D2D of the compressed block into
`d_comp_persist` plus per-kernel cuEvent record overhead, both
on-stream but not in the kernel-sum.

StreamLZ numbers are best-of-30 from a single bench session; nvCOMP
numbers are best-of-20 from `nvcomp_bench3`.

#### L1 / LZ4

| Window | StreamLZ L1 | nvCOMP LZ4 | StreamLZ win |
|--------|------------:|-----------:|-------------:|
| Pipeline kernel-sum | **4.03 ms** | 4.77 ms | 1.18× |
| Async call wall     | **4.61 ms** | 4.77 ms | 1.03× |
| End-to-end host wall | **15.51 ms** | 18.29 ms | 1.18× |

Compression ratio: StreamLZ L1 58.6%, nvCOMP LZ4 60.0%.

#### L5 / Zstd

| Window | StreamLZ L5 | nvCOMP Zstd | StreamLZ win |
|--------|------------:|------------:|-------------:|
| Pipeline kernel-sum | **5.50 ms** | 6.25 ms | 1.14× |
| Async call wall     | **5.94 ms** | 6.25 ms | 1.05× |
| End-to-end host wall | **15.27 ms** | 18.16 ms | 1.19× |

Compression ratio: StreamLZ L5 39.6%, nvCOMP Zstd 40.2%.

#### Where the wins come from

**Kernel-sum gap (~0.75-0.85 ms).** Faster compute per byte: the
32-stream Huffman decode (all 32 warp lanes active vs ~4 in prior
designs) combined with the BIL bounded-interleaved refill (one
coalesced 128-byte sector load per warp vs 32 scattered per-stream
loads), the warp-cooperative `slzHuffBuildLutKernel` rewrite, and the
two-phase preamble + 2-lookup hot loop described in `ARCHITECTURE.md`.

**Async-call wall is essentially tied (1.03×, 1.05×).** Both pipelines
finish one decompress in roughly the same wall budget; StreamLZ's
kernel-sum advantage is partially absorbed by its longer pipeline (10
kernels vs 1-5) plus per-kernel cuEvent overhead and the front-half
compressed-block D2D. For a streaming caller submitting many same-shape
decompresses on the same stream, the kernel-sum gap is what compounds
over the long run — the per-call event/setup overhead is fixed.

**End-to-end host gap (~2.8 ms = 1.18-1.19×).** Mostly the D2H of the
decompressed output. At 100 MB nvCOMP's `cudaMemcpyAsync` from device
to pageable host runs slower than StreamLZ's D2H into the
page-locked `h_pinned_output` buffer; the DMA path is the same but the
pinned-buffer path doesn't stage through a driver-internal bounce
buffer.

## Invariants: do not break these

1. **One `.cu` → one `.ptx`.** The `.cuh` files are a size-only split:
   they are `#include`d into a single `.cu` translation unit, never
   compiled separately. `nvcc lz_kernel.cu` still builds the whole thing.
2. **Kernels are bound by string.** The drivers resolve each kernel via
   `cuModuleGetFunction(module, "<name>")`. Renaming a kernel means a
   coordinated change: the `extern "C" __global__` symbol, the driver
   string, and a regenerated `.ptx`.
3. **All kernel symbols use the `slz` prefix.** Every `extern "C" __global__`
   in this module starts with `slz`, both to namespace against any other
   CUDA library the caller has loaded and to make every binding obvious
   in a grep.
4. **Descriptor structs mirror Zig `extern struct`s.** The CUDA-side
   structs are hand-kept copies of the layouts in the `*.zig` files;
   each carries a `static_assert(sizeof(...))` to catch ABI drift.
5. **The committed `.ptx` are intentional.** They are tracked so a
   `git clone` builds without `nvcc`. `.cubin` is git-ignored
   (architecture-specific, regenerated by the build scripts).
6. **The module is standalone.** Code under `src/gpu/` must not `@import`
   or `#include` anything outside `src/gpu/`.
