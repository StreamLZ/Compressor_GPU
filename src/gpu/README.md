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
  decode_dispatch.zig           fullGpuLaunch / fullGpuLaunchImpl + per-decode
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

Measured 2026-05-26 on an RTX 4060 Ti (`sm_89`), 32-stream Huffman in
the BIL (bounded-interleaved) wire format, `-gpu` mode (64 KB
sub-chunks). Corpora: enwik8 (100 MB text), silesia (212.8 MB mixed).
Decode figures are best-of-30 (`streamlz -db -r 30`, sequential per the
no-parallel-benchmarks rule); all times in milliseconds.

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.7% | 38.0% |
| L4 | 42.6% | 37.4% |
| L5 | 39.6% | 33.9% |

L1–L2 are LZ-only (no entropy stage); L3–L5 add GPU Huffman. The
BIL Huffman wire format costs ~0.3 pp ratio over the prior concat
layout at 64 KB sub-chunks (sub-header +3 B for the 32nd stream size,
+4 B for K, plus ≤3 B/stream rounding to a 4-byte BIL word) in
exchange for a ~4–6% Huffman-kernel-time win at L3–L5 on both
corpora — the refill path becomes one coalesced 128-byte sector
load per warp instead of 32 scattered per-stream loads.

### Decode (ms): D2D wall-clock and end-to-end

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | **2.93** / 15.64 | **5.08** / 32.80 |
| L2 | **2.95** / 15.60 | **5.11** / 31.03 |
| L3 | **4.58** / 16.15 | **8.26** / 32.90 |
| L4 | **4.44** / 15.97 | **8.01** / 32.62 |
| L5 | **4.60** / 15.92 | **8.78** / 32.72 |

**D2D wall-clock** = the time a caller of `slzDecompress` with device-
resident input and output sees on the wire. Measured via `cudaEventRecord`
pairs around the GPU kernel launches (LZ + Huffman for L3+ are pipelined
into one timing window). No other GPU work happens around the kernels in
the D2D entry points, so this number is the full GPU wall-clock for users
of the D2D API (game-asset / LLM-context / ML pipelines). **e2e** adds
the H2D upload of the compressed frame + D2H download of the decompressed
output for users on the host-bounce path.

LZ-only sub-D2D times (L3+ pipelined together with Huffman pre-decode).
Per-kernel split below is from the prior measurement run; the LZ kernel
is unchanged in this round, the Huff figures should drop ~4–6% with BIL:

| Level | enwik8 LZ / Huff | silesia LZ / Huff |
|-------|------------------|-------------------|
| L3 | 3.12 / ~1.82 | 5.37 / ~3.33 |
| L4 | 2.93 / ~1.70 | 5.22 / ~3.31 |
| L5 | 3.41 / ~1.50 | 6.20 / ~2.85 |

Note: LZ + Huff sums above are taken from SLZ_SPLIT_TIMER runs that
fence between the two kernel groups; the unfenced D2D wall-clock above
additionally includes scan / walk / compact-merge bookkeeping that
SPLIT_TIMER doesn't attribute to either kernel. So sum may be slightly
greater than (most cases) or less than (silesia L5: 9.21 < 9.35 D2D) the
unfenced total depending on how much the unattributed portion costs.

The 32-stream Huffman pre-decode (L3+) now lands at roughly one-third of
the LZ kernel time (down from roughly equal under the 4-stream design):
all 32 warp lanes decode in parallel where only 4 used to. LZ kernel
time is unchanged. The D2H copy of the decompressed output remains the
largest e2e cost; for D2D callers, it drops to zero (the L1 D2D
wall-clock of 2.93 ms is the entire decompress time).

### Encode (ms): GPU LZ-encode kernel

`last_kernel_ns` covers the LZ-encode kernel only (the Huffman-encode
kernels run as a separate pass and are not included).

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 73 | 138 |
| L2 | 77 | 143 |
| L3 | 84 | 158 |
| L4 | 111 | 206 |
| L5 | 283 | 20811 † |

† silesia L5 in the serial chain parser is the known pathological slow
path documented in [`project_l5_silesia_pathology`](project_l5_silesia_pathology.md):
the binary heterogeneity of silesia_all.tar (mostly text + xml + dna
with short chain depth + a few large binaries with extremely deep chains)
defeats the chain parser's CHAIN_MAX_STEPS guard. The encode-time number
is dominated by a few outlier sub-chunks and not representative of the
codec on uniform inputs (enwik8 L5 ratio 39.6% in 283 ms remains a fair
data point). The compression *ratio* on silesia L5 is real and
representative; only the encode-time number is misleading.

### vs nvCOMP (enwik8, 100 MB)

nvCOMP 5.2.0, same RTX 4060 Ti. (nvCOMP's hardware decompression engine
is Hopper or Blackwell only, not available on Ada, so only the CUDA
decode path applies here.)

| Codec | ratio | D2D wall-clock | decode e2e |
|-------|------:|---------------:|-----------:|
| StreamLZ L1 | 58.6% | **2.93 ms** | 15.64 ms |
| StreamLZ L5 | 39.6% | **4.60 ms** | **15.92 ms** |
| nvCOMP LZ4  | 60.0% | 5.1 ms | 18.5 ms |
| nvCOMP Zstd | 40.2% | 6.2 ms | 18.2 ms |

nvCOMP figures are best-of-20 from the `nvcomp_bench3` harness (nvCOMP
5.2.0), measured the same way as StreamLZ's `-db`: D2D wall-clock timed
by CUDA event around the decode kernel(s), e2e as
H2D(compressed) + decode + D2H(output) wall-clock, both with a
correctness verify.

**StreamLZ L1 win vs nvCOMP LZ4.** Smaller frame (58.6% vs 60.0%) and
faster decode on both axes: D2D 2.93 vs 5.1 ms (1.74x faster),
e2e 15.64 vs 18.5 ms (1.18x faster).

**StreamLZ L5 win vs nvCOMP Zstd.** Smaller frame (39.6% vs 40.2%) and
faster decode on both axes: D2D 4.60 vs 6.2 ms (1.35x faster),
e2e 15.92 vs 18.2 ms (1.14x faster). The D2D edge comes from the
32-stream Huffman decode (all 32 warp lanes active, vs 4 in the prior
design) combined with the BIL bounded-interleaved refill (one
coalesced 128-byte sector load per warp, vs 32 scattered per-stream
loads) and the two-phase preamble + 2-lookup hot loop described in
`ARCHITECTURE.md`.

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
