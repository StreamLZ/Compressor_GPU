# StreamLZ GPU module

GPU-accelerated compression and decompression for the StreamLZ Fast codec.
CUDA kernels (NVIDIA Driver API, target `sm_89`) plus thin Zig drivers that
load them. Entropy coding is canonical Huffman (`chunk_type=4`); GPU frames
contain only raw and Huffman sub-chunks.

This folder is **standalone** — nothing under `src/gpu/` imports anything
outside `src/gpu/`.

## Layout

```
common/   shared device headers (#include'd by the kernels below)
  gpu_warp.cuh      warp/lane geometry, bit-width constants
  gpu_byteio.cuh    big/little-endian u24 codecs, zero-padded reads
  gpu_huffman.cuh   Huffman wire-format contract: constants, canonical-code
                    build, weights + LUT pack/unpack

decode/   GPU decode
  driver.zig          loads the PTX, marshals buffers, launches the kernels
  lz_kernel.cu        LZ decode — thin aggregator that #includes:
    slz_wire_format.cuh   wire constants, descriptor structs, header parsers
    lz_decode_core.cuh    the two LZ decode hot loops
    lz_header_parse.cuh   sub-chunk header parser
    lz_kernels.cuh        the extern "C" __global__ entry points
  huffman_kernel.cu   Huffman pre-decode (chunk_type=4 literal/token/off16)
  vulkan_driver.zig + lz_kernel.comp/.spv   Vulkan path (separate, see below)

encode/   GPU encode
  driver.zig          loads the PTX, marshals buffers, launches the kernels
  lz_kernel.cu        LZ encode — thin aggregator that #includes:
    lz_format.cuh         format/hash constants, descriptor struct, helpers
    lz_token_emit.cuh     token/offset/length stream serializer
    lz_greedy_parser.cuh  warp-parallel greedy parser (L1-L4)
    lz_chain_parser.cuh   serial chain-hash lazy parser (L5+)
  huffman_kernel.cu   Huffman entropy encoder
```

## Pipeline

**Encode** — `lz_kernel.cu` finds matches and emits raw sub-streams
(literals / tokens / off16 / off32 / lengths); `huffman_kernel.cu` then
entropy-codes the literal, token and off16 streams into `chunk_type=4`
bodies. The Zig core assembles the frame.

**Decode** — `huffman_kernel.cu` pre-decodes the Huffman streams into a
scratch buffer; `lz_kernel.cu` then runs the LZ decode, reading the
pre-decoded streams and resolving matches.

### Kernel entry points

| Direction | Kernel | File |
|-----------|--------|------|
| encode | `slzLzEncodeKernel` | `encode/lz_kernel.cu` |
| encode | `slzHuffBuildTablesKernel`, `slzHuffEncode4StreamKernel` | `encode/huffman_kernel.cu` |
| decode | `slzLzDecodeKernel`, `slzLzDecodeRawKernel`, `slzGatherRawOff16Kernel` | `decode/lz_kernels.cuh` |
| decode | `slzHuffBuildLutKernel`, `slzHuffDecode4StreamKernel` | `decode/huffman_kernel.cu` |

## Build

CUDA kernels are compiled to PTX/CUBIN out-of-band by the build scripts
(they need `nvcc` + the MSVC toolchain, which `zig build` does not invoke):

```
tools/build_gpu.bat       decode/lz_kernel.cu, decode/huffman_kernel.cu,
                          encode/huffman_kernel.cu, and the Vulkan SPIR-V
tools/build_gpu_enc.bat   encode/lz_kernel.cu
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

Measured on an RTX 4060 Ti (`sm_89`), HEAD `e6f5e0a`, `-gpu` mode (64 KB
sub-chunks). Corpora: enwik8 (100 MB text), silesia (212.8 MB mixed).
Decode figures are best-of-30 (`streamlz -db -r 30`); all times in
**milliseconds**.

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.0% | 37.3% |
| L4 | 41.9% | 36.7% |
| L5 | 38.9% | 33.1% |

L1–L2 are LZ-only (no entropy stage); L3–L5 add GPU Huffman.

### Decode (ms) — D2D wall-clock · end-to-end

| | enwik8 D2D / e2e | silesia D2D / e2e |
|--|--|--|
| L1 | **2.92** / 15.80 | **5.07** / 30.20 |
| L2 | **2.94** / 15.66 | **5.09** / 30.30 |
| L3 | **6.59** / 18.18 | **12.29** / 36.22 |
| L4 | **6.38** / 17.84 | **12.08** / 35.91 |
| L5 | **6.32** / 17.60 | **12.41** / 35.50 |

**D2D wall-clock** = the time a caller of `slzDecompress` with device-
resident input and output sees on the wire. It is measured via
`cudaEventRecord` pairs around the GPU kernel launches (LZ + Huffman
for L3+ are pipelined into one timing window). No other GPU work
happens around the kernels in the D2D entry points, so this number is
the full GPU wall-clock for users of the D2D API
(game-asset/LLM-context/ML pipelines). **e2e** adds the H2D upload of
the compressed frame + D2H download of the decompressed output for
users on the host-bounce path.

LZ-only sub-D2D times (L3+ pipelined together with Huffman pre-decode):

| | enwik8 LZ / Huff | silesia LZ / Huff |
|--|--|--|
| L3 | 2.99 / 3.58 | 5.17 / 7.08 |
| L4 | 2.89 / 3.48 | 5.04 / 7.00 |
| L5 | 3.28 / 3.03 | 6.09 / 6.29 |

The Huffman pre-decode (L3+) is now roughly equal to the LZ kernel —
both contribute ~half the GPU work for entropy-coded levels. The D2H
copy of the decompressed output remains the largest e2e cost; for D2D
callers, it drops to zero (the L1 D2D wall-clock 2.92 ms is the entire
decompress time).

**Most recent decode work** (see `FAILED_EXPERIMENTS.md`):

- **Parallel-parse rewrite** of `decodeSubChunkRawMode` (both off16
  modes): coalesced 32-byte cmd batch load → per-lane SHORT-token
  decode → warp-prefix-scan side-stream offsets → sequential
  cooperative copy of 32 parsed tokens. Replaces the lane-0 serial
  parse + 5×`__shfl_sync` broadcast per token. Long-token batches fall
  back to the original serial path.
- **Universal 17-25 % D2D speedup** across all levels/inputs. L1
  goes 3.8 → 2.92 ms on enwik8 (-23 %); silesia L1 6.3 → 5.07 ms
  (-19 %); both L5s drop ~15 %. The LZ portion for L3+ is now about
  the same time as the L1 D2D wall-clock.

### Encode (ms) — GPU LZ-encode kernel

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
is Hopper/Blackwell-only — not available on Ada — so only the CUDA
decode path applies here.)

| Codec | ratio | D2D wall-clock | decode e2e |
|-------|------:|---------------:|-----------:|
| StreamLZ L1 | 58.6% | **2.92 ms** | 15.80 ms |
| StreamLZ L5 | 38.9% | **6.32 ms** | 17.60 ms |
| nvCOMP LZ4 | 60.0% | 5.1 ms | 18.5 ms |
| nvCOMP Zstd | 40.2% | 6.2 ms | 18.2 ms |

nvCOMP figures are best-of-20 from the `nvcomp_bench3` harness (nvCOMP
5.2.0), measured the same way as StreamLZ's `-db`: D2D wall-clock
timed by CUDA event around the decode kernel(s), e2e as
H2D(compressed) + decode + D2H(output) wall-clock, both with a
correctness verify.

**StreamLZ L1 beats nvCOMP LZ4 outright** — smaller frame (58.6 % vs
60.0 %) AND faster decode on both axes (D2D **2.92 vs 5.1 ms** =
1.75× faster; e2e 15.80 vs 18.5 ms = 1.17× faster).

**StreamLZ L5 beats nvCOMP Zstd on ratio and end-to-end decode** —
ratio 38.9 % vs 40.2 % and e2e **17.60 vs 18.2 ms** (0.6 ms /
3.3 % faster). D2D 6.32 vs 6.2 ms is essentially tied (Zstd narrowly
ahead on the pure-GPU timer; the parallel-parse rewrite closed what
was a 19 % gap down to 2 %).

## Invariants — do not break these

1. **One `.cu` → one `.ptx`.** The `.cuh` files are a size-only split:
   they are `#include`d into a single `.cu` translation unit, never
   compiled separately. `nvcc lz_kernel.cu` still builds the whole thing.
2. **Kernels are bound by string.** The drivers resolve each kernel via
   `cuModuleGetFunction(module, "<name>")`. Renaming a kernel means a
   coordinated change: the `extern "C" __global__` symbol, the driver
   string, and a regenerated `.ptx`.
3. **Descriptor structs mirror Zig `extern struct`s.** The CUDA-side
   structs are hand-kept copies of the layouts in the `driver.zig` files;
   each carries a `static_assert(sizeof(...))` to catch ABI drift.
4. **The committed `.ptx`/`.spv` are intentional.** They are tracked so a
   `git clone` builds without `nvcc`. `.cubin` is git-ignored
   (architecture-specific, regenerated by the build scripts).
5. **The module is standalone.** Code under `src/gpu/` must not `@import`
   or `#include` anything outside `src/gpu/`.

## Vulkan

`decode/vulkan_driver.zig` + `lz_kernel.comp`/`.spv` are a separate,
in-progress compute-shader decode path for non-NVIDIA hardware. It does
not affect the CUDA build and is not covered by this document.
