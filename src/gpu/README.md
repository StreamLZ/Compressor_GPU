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
                                HUFF_LUT_ENTRIES, walk_max_chunks
  decode_context.zig            DecodeContext + ensure/alloc/copy helpers
                                + kernel-event profiling plumbing
  scan_host.zig                 CPU scanForEntropyChunks (H2D entry path)
  scan_gpu.zig                  GPU walk / prefix-sum / scan / compact orchestration
  decode_dispatch.zig           fullGpuLaunch / fullGpuLaunchImpl + per-decode
                                helpers (dumpScanIfRequested, emitE2eTrace,
                                gatherRawOff16, mergeHuffDescs)
  lz_kernel.cu                  LZ decode aggregator; #includes:
    slz_wire_format.cuh         decoder-private token formats + parsers
    lz_decode_core.cuh          the two LZ decode hot loops + warp copy helpers
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

Every CUDA kernel resolved by the Zig drivers via `cuModuleGetFunction`.

| Direction | Kernel | Source |
|-----------|--------|--------|
| encode | `slzLzEncodeKernel` | `encode/lz_kernel.cu` |
| encode | `slzHuffBuildTablesKernel` | `encode/huffman_kernel.cu` |
| encode | `slzHuffEncode4StreamKernel` | `encode/huffman_kernel.cu` |
| encode | `slzAssembleMeasureKernel` | `encode/assemble_kernel.cu` |
| encode | `slzAssembleWriteKernel` | `encode/assemble_kernel.cu` |
| encode | `slzFrameAssembleKernel` | `encode/assemble_kernel.cu` |
| decode | `slzLzDecodeKernel` | `decode/lz_decode_kernels.cuh` |
| decode | `slzLzDecodeRawKernel` | `decode/lz_decode_kernels.cuh` |
| decode | `slzWalkFrameKernel` | `decode/walk_frame_kernel.cuh` |
| decode | `slzPrefixSumChunksKernel` | `decode/prefix_sum_chunks_kernel.cuh` |
| decode | `slzScanParseKernel` | `decode/scan_parse_kernel.cuh` |
| decode | `slzCompactHuffDescsKernel` | `decode/compact_descs_kernels.cuh` |
| decode | `slzCompactRawDescsKernel` | `decode/compact_descs_kernels.cuh` |
| decode | `slzMergeHuffDescsKernel` | `decode/merge_huff_descs_kernel.cuh` |
| decode | `slzGatherRawOff16Kernel` | `decode/gather_raw_off16_kernel.cuh` |
| decode | `slzHuffBuildLutKernel` | `decode/huffman_kernel.cu` |
| decode | `slzHuffDecode4StreamKernel` | `decode/huffman_kernel.cu` |

## Build

CUDA kernels compile to PTX/CUBIN out-of-band (they need `nvcc` + the MSVC
toolchain, which `zig build` does not invoke):

```
tools/build_gpu.bat       decode/lz_kernel.cu, decode/huffman_kernel.cu,
                          encode/huffman_kernel.cu, plus the Vulkan SPIR-V
tools/build_gpu_enc.bat   encode/lz_kernel.cu, encode/assemble_kernel.cu
```

Both run `nvcc -arch=sm_89 -O3`. The generated `.ptx` is committed; the
Zig drivers `@embedFile` it, so a plain `zig build` does not require CUDA
to be installed.

```
zig build -Doptimize=ReleaseFast -Dgpu=true   streamlz CLI with GPU paths
zig build gpulib                              streamlz_gpu.dll (nvCOMP-style C API)
```

`nvcuda.dll` is loaded at runtime; if CUDA is unavailable the codec falls
back to the CPU path.

## Performance

Measured 2026-05-25 on an RTX 4060 Ti (`sm_89`), HEAD `893e043`, `-gpu`
mode (64 KB sub-chunks). Corpora: enwik8 (100 MB text), silesia
(212.8 MB mixed). Decode figures are best-of-30 (`streamlz -db -r 30`,
sequential per the no-parallel-benchmarks rule); all times in milliseconds.

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.0% | 37.3% |
| L4 | 41.9% | 36.7% |
| L5 | 38.9% | 33.1% |

L1–L2 are LZ-only (no entropy stage); L3–L5 add GPU Huffman.

### Decode (ms): D2D wall-clock and end-to-end

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | **2.92** / 15.63 | **5.07** / 30.20 |
| L2 | **2.94** / 15.66 | **5.09** / 30.30 |
| L3 | **6.57** / 18.11 | **12.27** / 36.25 |
| L4 | **6.38** / 17.84 | **12.08** / 35.91 |
| L5 | **6.26** / 17.55 | **12.41** / 35.45 |

**D2D wall-clock** = the time a caller of `slzDecompress` with device-
resident input and output sees on the wire. Measured via `cudaEventRecord`
pairs around the GPU kernel launches (LZ + Huffman for L3+ are pipelined
into one timing window). No other GPU work happens around the kernels in
the D2D entry points, so this number is the full GPU wall-clock for users
of the D2D API (game-asset / LLM-context / ML pipelines). **e2e** adds
the H2D upload of the compressed frame + D2H download of the decompressed
output for users on the host-bounce path.

LZ-only sub-D2D times (L3+ pipelined together with Huffman pre-decode):

| Level | enwik8 LZ / Huff | silesia LZ / Huff |
|-------|------------------|-------------------|
| L3 | 2.99 / 3.58 | 5.17 / 7.08 |
| L4 | 2.89 / 3.48 | 5.04 / 7.00 |
| L5 | 3.28 / 3.03 | 6.09 / 6.29 |

The Huffman pre-decode (L3+) is roughly equal to the LZ kernel. Both
contribute about half the GPU work for entropy-coded levels. The D2H
copy of the decompressed output remains the largest e2e cost; for D2D
callers, it drops to zero (the L1 D2D wall-clock of 2.92 ms is the
entire decompress time).

### Encode (ms): GPU LZ-encode kernel

`last_kernel_ns` covers the LZ-encode kernel only (the Huffman-encode
kernels run as a separate pass and are not included).

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 74 | 135 |
| L2 | 78 | 142 |
| L3 | 81 | 158 |
| L4 | 112 | 203 |
| L5 | 296 | 4502 † |

† silesia L5 is a known pathological slow path in the serial chain
parser; not representative.

### vs nvCOMP (enwik8, 100 MB)

nvCOMP 5.2.0, same RTX 4060 Ti. (nvCOMP's hardware decompression engine
is Hopper or Blackwell only, not available on Ada, so only the CUDA
decode path applies here.)

| Codec | ratio | D2D wall-clock | decode e2e |
|-------|------:|---------------:|-----------:|
| StreamLZ L1 | 58.6% | **2.92 ms** | 15.63 ms |
| StreamLZ L5 | 38.9% | **6.26 ms** | 17.55 ms |
| nvCOMP LZ4  | 60.0% | 5.1 ms | 18.5 ms |
| nvCOMP Zstd | 40.2% | 6.2 ms | 18.2 ms |

nvCOMP figures are best-of-20 from the `nvcomp_bench3` harness (nvCOMP
5.2.0), measured the same way as StreamLZ's `-db`: D2D wall-clock timed
by CUDA event around the decode kernel(s), e2e as
H2D(compressed) + decode + D2H(output) wall-clock, both with a
correctness verify.

**StreamLZ L1 win vs nvCOMP LZ4.** Smaller frame (58.6% vs 60.0%) and
faster decode on both axes: D2D 2.92 vs 5.1 ms (1.75x faster),
e2e 15.63 vs 18.5 ms (1.18x faster).

**StreamLZ L5 vs nvCOMP Zstd.** Wins ratio (38.9% vs 40.2%) and e2e
(17.55 vs 18.2 ms, 0.65 ms faster). D2D is essentially tied: Zstd 6.2 ms
vs StreamLZ 6.26 ms (Zstd narrowly ahead on the pure-GPU timer).

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
