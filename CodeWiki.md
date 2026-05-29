# StreamLZ — Code Wiki

Internal reference for contributors. For user-facing docs see
[README.md](README.md). For the GPU-only-strip rationale and the prior
two-codebase history see [FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md).

For the GPU codec's algorithmic notes (warp mapping, BIL wire format,
LUT-build kernel, etc.) see [docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md).

---

## What this codebase is

StreamLZ is a GPU-accelerated LZ77-style compression library. CUDA
kernels (NVIDIA Driver API, target `sm_89`) do the per-chunk LZ
encode + decode and the Huffman pre-decode pass; Zig drivers manage
the kernel launches and the host-side wire-format assembly.

There is only one backend — CUDA. There used to be a CPU codec and a
Vulkan compute-shader fallback; both were stripped on 2026-05-29 (see
FAILED_EXPERIMENTS.md "Maintaining parallel CPU and GPU codebases").

---

## Source layout

```
build.zig                 Zig 0.16 build script (always builds GPU)
build.zig.zon
include/
  streamlz_gpu.h          C ABI public header
src/
  main.zig                CLI dispatcher
  cli.zig                 CLI argument parser + per-mode handlers
  streamlz_gpu.zig        C ABI implementation (slzCompress, slzDecompress)
  test_runner_parallel.zig
  common/                 SHARED CUDA headers (#include'd by every kernel)
    gpu_warp.cuh          warp / lane geometry constants
    gpu_byteio.cuh        little / big-endian load + store primitives
    gpu_huffman.cuh       canonical-Huffman wire format constants
    gpu_wire_format.cuh   encode/decode shared LZ wire constants
  format/                 Wire-format parsers and writers (host-side)
    frame_format.zig
    block_header.zig
    streamlz_constants.zig
  io/                     Bit / byte helpers used by host-side entropy code
    bit_reader.zig, bit_writer.zig, copy_helpers.zig, ptr_math.zig
  platform/               OS-level helpers (memory query, mmap, cache size)
  encode/
    streamlz_encoder.zig  public encode API (compressFramed, compressBound)
    fast_framed.zig       L1-L5 frame builder; orchestrates the GPU kernels
    gpu_stream_assembly.zig  host-side reencodeGpuWithEntropy + helpers
    gpu_roundtrip_tests.zig  end-to-end roundtrip tests
    driver.zig            thin facade re-exporting the sub-modules
    cuda_ffi.zig          nvcuda.dll handle + CU* typedefs + cu*_fn slots
    module_loader.zig     PTX load, kernel-handle pub vars, init / isAvailable
    encode_context.zig    EncodeContext + per-handle ABI descriptors
    encode_lz.zig         gpuCompressImpl (LZ launcher)
    encode_huff.zig       gpuEncodeHuffImpl + per-stream Huffman launchers
    encode_assemble.zig   gpuAssembleFrameImpl + gpuFrameAssembleImpl
    levels.zig            hashBitsForLevel, useChainParser
    lz_kernel.cu          LZ encode aggregator (#includes its .cuh siblings)
    huffman_kernel.cu     Huffman encode kernels
    assemble_kernel.cu    per-sub-chunk + frame-level wire assembly kernels
    lz_format.cuh, lz_chain_parser.cuh, lz_greedy_parser.cuh,
      lz_token_emit.cuh
    entropy/
      entropy_encoder.zig encodeArrayU8 + writeNonCompactChunkHeader
      tans_encoder.zig    CPU tANS (used by paired-primary host fallback)
      huffman_encoder.zig CPU canonical Huffman (host fallback)
      byte_histogram.zig
    fast/
      fast_constants.zig  mapLevel + buildMinimumMatchLengthTable
      fast_match_hasher.zig  FastMatchHasher (typed by hash-element width)
  decode/
    streamlz_decoder.zig  public decode API + frame walk + per-block dispatch
    driver.zig            thin facade re-exporting the sub-modules
    cuda_api.zig          nvcuda.dll handle, CU* typedefs, cu*_fn slots
    module_loader.zig     PTX load, kernel-handle pub vars, init / isAvailable
    descriptors.zig       ChunkDesc + GpuError + walk_meta_offsets
    decode_context.zig    DecodeContext + ensure/alloc/copy helpers
    decode_dispatch.zig   fullGpuLaunchImpl + per-decode plumbing
    scan_host.zig         CPU walk fallback (legacy H2D-bounce path)
    scan_gpu.zig          GPU walk (pure-D2D entry, no host bounce)
    lz_kernel.cu          LZ decode aggregator
    huffman_kernel.cu     LUT build + 32-stream Huffman decode kernels
    lz_decode_core.cuh, lz_decode_raw.cuh, lz_decode_general.cuh,
      lz_decode_kernels.cuh, lz_dispatch.cuh, lz_header_parse.cuh,
      slz_wire_format.cuh, walk_frame_kernel.cuh,
      prefix_sum_chunks_kernel.cuh, scan_parse_kernel.cuh,
      compact_descs_kernels.cuh, merge_huff_descs_kernel.cuh,
      gather_raw_off16_kernel.cuh
  *.ptx                   Committed compiled kernel images (@embedFile-d
                          into the Zig drivers)
docs/
  GPU_ARCHITECTURE.md     why the GPU code is the way it is
  GPU_README.md           kernel inventory + ABI invariants
  GPU_IDEAS.md            unfinished ideas log
tools/
  build_gpu.bat           compile every .cu to .ptx + .cubin via nvcc
  bench_all.bat           encode + decode + SHA verify across L1-L5
  bench_d2d.bat           same but exercises the C-ABI device-resident path
  build_d2d_bench.bat     compile the D2D bench C harness
  slz_gpu_*.c             example C ABI consumers
```

---

## Public APIs

Two public entry points:

* **Zig library**: `src/encode/streamlz_encoder.zig`
  (`compressFramed`, `compressBound`), `src/decode/streamlz_decoder.zig`
  (`decompressFramed`, `DecompressContext`).
* **C ABI**: `src/streamlz_gpu.zig` (`slzCompress`, `slzDecompress`,
  `slzCompressAsync`, `slzDecompressAsync`, plus the host-bounce
  variants `slzCompressHost` / `slzDecompressHost`). Header is
  `include/streamlz_gpu.h`.

The CLI (`zig build run`) and the bench scripts under `tools/` are
the in-tree consumers of both APIs.

---

## Encode pipeline

`compressFramed` → `fast_framed.compressFramedOne` does the work:

1. Frame header: chosen `sc_group_size` (0.25 below the GPU
   saturation threshold, 0.5 at or above — see
   `resolveScGroupSize`), codec=Fast, content_size, no dictionary.
2. LZ kernel (`slzLzEncodeKernel`): one warp per chunk, raw-stream
   output per sub-chunk.
3. Huffman kernels (L3-L5 only): three passes over literals, tokens,
   off16. Each produces chunk_type=4 bodies.
4. Assembly:
   * **Pure D2D** (`slzFrameAssembleKernel`) when the caller handed
     us a device output buffer and every chunk compressed below raw.
     The kernel writes the whole frame straight to the caller's
     device buffer.
   * **Host walk** otherwise. For each sub-chunk:
     splice the device-assembled wire block if the assembly kernel
     produced one, else call
     `gpu_stream_assembly.reencodeGpuWithEntropy` to build the wire
     block on the host from the raw GPU streams + GPU-Huffman bodies.
5. SC tail prefix table (8 bytes per chunk past the first).
6. End mark.

The kernel inventory and per-kernel REG/STACK/SHARED numbers live in
[docs/GPU_README.md](docs/GPU_README.md) and
[docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md).

---

## Decode pipeline

`decompressFramed` → `streamlz_decoder.decompressFrameInner` walks the
frame; each compressed block is handed to
`dispatchCompressedBlock` which:

1. Strips the SC tail prefix.
2. Builds one `ChunkDesc` per chunk (flags=0 LZ, flags=1 raw,
   flags=2 memset).
3. Calls `gpu_driver.fullGpuLaunchImpl`, which fans the descriptors
   through the six bookkeeping kernels (walk, prefix-sum, scan,
   compact, merge, gather), the Huffman LUT-build + 32-stream
   decode, and finally `slzLzDecodeKernel` or `slzLzDecodeRawKernel`.

The pure-D2D entry point is `decompressFramedFromDevice` —
compressed bytes + output stay on device throughout, the frame walk
runs on the GPU via `gpuWalkFrameImpl`.

---

## Wire format

Wire-format constants are split between two locations and must stay
in sync:

* `src/format/streamlz_constants.zig` — host-side constants
* `src/common/gpu_wire_format.cuh` — device-side constants
* `src/common/gpu_huffman.cuh` — Huffman-specific constants

Every encoder-emitted byte and every decoder-consumed byte comes from
constants in those files. See
[docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md) for the
sub-chunk header layout, the off16 hi/lo split, and the BIL
bounded-interleaved Huffman wire format.

---

## Build

```
zig build                                 ReleaseFast streamlz.exe
zig build -Doptimize=ReleaseFast          same
zig build gpulib                          streamlz_gpu.dll only
zig build test                            unit tests + GPU roundtrip
zig build run -- -l 3 in.bin -o out.slz   run the CLI
```

The PTX is committed under `src/encode/` and `src/decode/`, so plain
`zig build` does not need CUDA installed — the Zig drivers
`@embedFile` the kernel images at compile time.

To rebuild PTX (after editing any `.cu` / `.cuh`):

```
tools\build_gpu.bat
```

That script invokes `nvcc -arch=sm_89 -O3` on the five `.cu`
translation units, then re-runs `zig build` so the freshly-built PTX
is embedded into `streamlz.exe` and `streamlz_gpu.dll`.

`build.zig` has a freshness gate: any `.cu` / `.cuh` newer than any
`.ptx` fails the build with a "run `tools\build_gpu.bat`" message.

---

## CUDA loading

`nvcuda.dll` is loaded at runtime via `LoadLibraryA`. If it is
missing or `cuInit` fails, the codec returns
`error.BackendNotAvailable` / `SLZ_ERROR_CUDA` rather than crashing
— there is no CPU fallback to land on.

Two separate CUDA-handle slots exist, one for encode
(`src/encode/cuda_ffi.zig` `_lib`) and one for decode
(`src/decode/cuda_api.zig` `lib`). The first side to initialize
creates the CUDA context; the second side reuses it via
`cuCtxGetCurrent`. The bench scripts and the CLI's `-b` mode
exercise both initialization orders.

---

## Levels

L1-L5 (Fast codec) is the only supported range. The mapping to the
internal engine level lives in
`src/encode/fast/fast_constants.zig` `mapLevel`:

| User | Engine | Parser | Entropy |
|------|--------|--------|---------|
| L1   | -2     | greedy, ushort hash | raw |
| L2   | -1     | greedy, uint hash | raw |
| L3   | 1      | greedy + Huffman | yes |
| L4   | 2      | greedy-rehash + Huffman | yes |
| L5   | 4      | lazy chain + Huffman | yes |

L4 skips engine 3 because the Fast 4 parser variant historically
lost to Fast 5 on every workload measured.

L6-L11 returned an error in the GPU codec even before the strip and
are now removed from the public surface — `error.BadLevel`.

---

## Performance

Recent baselines: enwik8 100 MB + silesia 213 MB, RTX 4060 Ti
sm_89, `-db -r 30` best-of-30, in milliseconds. See
[docs/GPU_README.md](docs/GPU_README.md) for the full table.

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | 2.91 / 15.49 | 5.06 / 29.93 |
| L2 | 2.93 / 15.48 | 5.07 / 29.88 |
| L3 | 4.10 / 15.54 | 7.13 / 30.53 |
| L4 | 3.94 / 15.30 | 6.97 / 30.30 |
| L5 | 4.11 / 15.24 | 7.71 / 30.38 |

The strip preserved these numbers within ±0.06 ms across every row,
verified pre/post via `tools\bench_all.bat`.

---

## Invariants — do not break these

1. **One `.cu` → one `.ptx`.** The `.cuh` files are a size-only
   split — they are `#include`d into a single `.cu` translation
   unit, never compiled separately. `nvcc lz_kernel.cu` still
   builds the whole thing.
2. **Kernels are bound by string.** The Zig drivers resolve each
   kernel via `cuModuleGetFunction(module, "<name>")`. Renaming a
   kernel needs a coordinated change to the `extern "C" __global__`
   symbol, the driver string, and a regenerated `.ptx`.
3. **All kernel symbols use the `slz` prefix.** Every
   `extern "C" __global__` starts with `slz`, to namespace against
   any other CUDA library the caller has loaded and to make every
   binding obvious to grep.
4. **Descriptor structs mirror Zig `extern struct`s.** The CUDA-side
   structs are hand-kept copies of the layouts in the corresponding
   `.zig` files; each carries a `static_assert(sizeof(...))` to catch
   ABI drift.
5. **Committed `.ptx` are intentional.** They are tracked so
   `git clone` builds without `nvcc`. `.cubin` is git-ignored —
   architecture-specific, regenerated by the build scripts for the
   res-usage printout.
6. **Wire-format constants live in one place per side.** Host:
   `src/format/streamlz_constants.zig`. Device:
   `src/common/gpu_wire_format.cuh`. The two must agree on every
   byte boundary; the existing `static_assert`s catch drift.
