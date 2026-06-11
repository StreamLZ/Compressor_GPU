# StreamLZ - Code Wiki

Internal reference for contributors (CUDA backend; refreshed
2026-06-11). For user-facing docs see [README.md](README.md). For the
GPU-only-strip rationale and prior history see
[FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md).

For the GPU codec's algorithmic notes (warp mapping, BIL wire format,
LUT-build kernel, etc.) see [docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md).
For CUDA tooling recipes (nvcc, NCU, nsys paths + invocations) see
[docs/how_to_debug_cuda.md](docs/how_to_debug_cuda.md).

**The Vulkan backend (`srcVK/`) has its own canon** - byte-identical
wire format, full L1-L5 parity: [srcVK/README.md](srcVK/README.md)
(layout), [srcVK/Handbook.md](srcVK/Handbook.md) (invariants +
recipes), [srcVK/PortAdaptations.md](srcVK/PortAdaptations.md) (the
A-NNN CUDA↔VK divergence catalog - extend it for every new
divergence). This wiki covers the CUDA tree.

---

## What this codebase is

StreamLZ is a GPU-accelerated LZ77-style compression library with two
sibling backends producing byte-identical frames: CUDA kernels
(NVIDIA Driver API, target `sm_89`) under `src/`, and a 1:1 Vulkan
port under `srcVK/`. Zig drivers manage kernel launches and host-side
wire-format assembly.

---

## Source layout (CUDA tree)

### Top level

| Path | Purpose |
|---|---|
| `build.zig` | Zig 0.16 build script. Steps: `install` / `run` / `test`+`ptest` / `gpulib` / `ptx` / `srcvk-shaders` / `streamlz_vk` / `ptest_vk` / `fuzz` / `stress18` / `chaincount` / `tans_gate2` |
| `include/streamlz_gpu.h` | C ABI public header |
| `srcVK/` | Vulkan backend (its own canon: see `srcVK/README.md`) |
| `docs/` | GPU_ARCHITECTURE (incl. kernel inventory), nvcomp_lz4_architecture, how_to_debug_cuda, cudaOptimize (incl. vs-nvCOMP methodology), ngram |
| `v4_ideas.md` | THE forward-looking work list |
| `FAILED_EXPERIMENTS.md` | Negative results (do not re-run these) |

### `src/` core

| File | Purpose |
|---|---|
| `main.zig` | Entry point + test aggregation block |
| `cli.zig`, `cli/` | CLI mode dispatcher + per-mode handlers: `util.zig` (args), compress, decompress, info, bench_compress (`-b`), bench_decompress (`-db`), bench_all (`-ba`) |
| `streamlz_gpu.zig` | C ABI implementation (root of `streamlz_gpu.dll`) |
| `c_abi_tests.zig` | C ABI tests via extern-fn binding shape |
| `cli_smoke_tests.zig` | Subprocess CLI tests (spawn `streamlz.exe`) |
| `test_runner_parallel.zig` | Parallel test runner; prints SKIP names |
| `stress18_main.zig` | v4 #18 in-process roundtrip stress harness |
| `chain_count_main.zig` | v4 #14 chain-parser instrumentation readback |
| `version.zig`, `mmap.zig` | Misc |

### `src/common/` (shared CUDA headers, `#include`'d by kernels)

| File | Purpose |
|---|---|
| `gpu_warp.cuh` | Warp/lane geometry + `SLZ_GUARD_SINGLE_THREAD` |
| `gpu_byteio.cuh` | Endian load/store primitives |
| `gpu_huffman.cuh` | Canonical-Huffman wire constants |
| `gpu_wire_format.cuh` | Encode/decode shared LZ wire constants |
| `xxh32_device.cuh` | Device XXH32 + the chunk-Merkle hash kernel bodies (v4 #19; included by BOTH `lz_kernel.cu` TUs) |

### `src/format/` (host-side wire format)

| File | Purpose |
|---|---|
| `frame_format.zig` | Frame header parse/write, flags, trailers |
| `block_header.zig` | Block + internal-block headers |
| `streamlz_constants.zig` | Wire constants (host side of the sync pair) |
| `xxhash32.zig` | XXH32 + the chunk-Merkle root (host mirror of the device definition; threaded fallback) |

### `src/encode/`

| File | Purpose |
|---|---|
| `streamlz_encoder.zig` | Public encode API (`compressFramed`, `compressBound`) |
| `fast_framed.zig` | L1-L5 frame builder; orchestrates the kernels |
| `enc_phase.zig` | `SLZ_PROFILE_PHASES=1` per-phase QPC profiler |
| `driver.zig` | Facade |
| `cuda_ffi.zig` | `nvcuda.dll` handle + `cu*_fn` slots (encode side) |
| `module_loader.zig` | PTX load + kernel handles (SRWLOCK-guarded init) |
| `encode_context.zig` | `EncodeContext` + persistent device buffers |
| `encode_lz.zig` | `gpuCompressImpl` (A-023 VRAM-budgeted batching; v4 #17 pinned gather; v4 #19 hash launches) |
| `encode_huff.zig` | Per-stream Huffman launchers (L3+) |
| `encode_assemble.zig` | Measure/write + frame-assemble launchers |
| `levels.zig` | `hashBitsForLevel` (17 at EVERY level), `useChainParser` |
| `lz_kernel.cu` | LZ encode aggregator TU |
| `huffman_kernel.cu` | Huffman encode kernels TU |
| `assemble_kernel.cu` | Wire assembly kernels TU |
| `lz_format.cuh`, `lz_chain_parser.cuh`, `lz_greedy_parser.cuh`, `lz_token_emit.cuh` | Kernel implementation headers |

Encode-side tests: `gpu_roundtrip_tests.zig` (roundtrips + the shared
GPU-test lock: `lockGpuTests` serializes + binds the CUDA context to
the worker thread; also the v4 #18 mismatch-artifact capture),
`gpu_regression_tests.zig` (A-023 forced-batch + real-corpus cases),
`huff_conformance_tests.zig` (4-kernel Huffman chain, byte-identity),
`l5_hardening_tests.zig` (chain-parser adversarial patterns).

### `src/decode/`

| File | Purpose |
|---|---|
| `streamlz_decoder.zig` | Public decode API + CPU frame walk (host path) + checksum verification |
| `driver.zig` | Facade (deviceName, `bindContextToCallingThread`) |
| `cuda_api.zig` | `nvcuda.dll` handle + `cu*_fn` slots + QPC helpers |
| `module_loader.zig` | PTX load + kernel handles + device-name cache |
| `descriptors.zig` | `ChunkDesc` + `GpuError` + `walk_meta_offsets` |
| `decode_context.zig` | `DecodeContext` + `ensureDeviceBuf` + profiling |
| `decode_dispatch.zig` | `fullGpuLaunchImpl` + huff/LZ/verify kernel launches |
| `scan_host.zig`, `scan_gpu.zig` | CPU walk fallback; GPU walk + fused compact dispatch |
| `lz_kernel.cu` | LZ decode aggregator TU |
| `huffman_kernel.cu` | LUT build + 32-stream Huffman decode TU |
| `lz_decode_*.cuh`, `lz_decode_raw_pipeline.cuh`, `lz_dispatch.cuh`, `lz_header_parse.cuh`, `slz_wire_format.cuh` | Decode kernel implementation headers (incl. the v4 #15 K=4 pipeline) |
| `walk_frame_kernel.cuh`, `prefix_sum_chunks_kernel.cuh`, `scan_parse_kernel.cuh`, `compact_descs_kernels.cuh`, `merge_huff_descs_kernel.cuh`, `gather_raw_off16_kernel.cuh` | Orchestration kernels (incl. fused `slzCompactAllDescsKernel` and `slzMergeHuffDescsParKernel`) |

Committed `*.ptx` kernel images sit beside the `.cu` files and are
`@embedFile`'d, so plain `zig build` needs no CUDA toolchain.

### `tools/`

| Path | Purpose |
|---|---|
| `build_gpu.bat` | Full nvcc rebuild + cuobjdump res-usage |
| `sanitize.bat` | compute-sanitizer gate (memcheck/racecheck) |
| `bench_all.bat`, `bench_d2d.bat`, `build_d2d_bench.bat`, `slz_gpu_*.c` | Bench harnesses (host + device-resident C ABI) |
| `ncu_profile_lz.bat` | NCU profile of the LZ decode kernels (admin shell) |
| `huff_test/` | Standalone Huffman/tANS kernel harness |
| `tans_gate2/` | v4 #11 gate-2 measurement (resurrected host tANS encoder) |
| `fuzz_frames.zig` | v4 #13 differential frame-mutation fuzzer |


## Public APIs

* **Zig library**: `src/encode/streamlz_encoder.zig`
  (`compressFramed`, `compressFramedWithIo`, `compressBound`),
  `src/decode/streamlz_decoder.zig` (`decompressFramed`,
  `DecompressContext`).
* **C ABI** (`src/streamlz_gpu.zig`, header `include/streamlz_gpu.h`):
  `slzCreate`/`slzDestroy`, `slzCompressHost`/`slzDecompressHost`
  (library-owned worker thread per call), `slzCompressAsync`/
  `slzDecompressAsync` (true D2D, nvCOMP-shaped, ride the caller's
  stream), `slzCompressBound`, `slzGetDecompressedSize`,
  `slzGetLastTimings`/`slzWaitAndGetLastTimings`, version/status
  strings, default-opts helpers. Covered by `src/c_abi_tests.zig`
  through extern-fn declarations (the real C-caller binding shape).

The CLI and `tools/` scripts are the in-tree consumers of both.

---

## Encode pipeline

`compressFramed` → `fast_framed.compressFramedOne`:

1. Frame header: `sc_group_size = 0.25` unconditionally (64 KB
   sub-chunks - the structural lever that beat nvCOMP; override via
   `-sc`), codec=Fast, content_size.
2. LZ kernel (`slzLzEncodeKernel`): one warp per chunk, raw streams
   per sub-chunk. A-023: dispatch is batched under a cuMemGetInfo
   VRAM budget so the per-chunk hash tables never page over PCIe.
3. Huffman kernels (L3+): three passes (literals, tokens, off16) →
   chunk_type=4 BIL bodies.
4. Assembly: device-side measure/write (`gpuAssembleFrameImpl`), the
   host walk picks compressed-vs-raw per chunk (expansion falls back
   to `UNCOMPRESSED_CHUNK_MARKER`), then `slzFrameAssembleKernel`
   splices the frame device-side; D2H only when the caller wants
   host output.
5. SC tail prefix table + end mark.
6. Integrity (v4 #19, default on): per-chunk hashes of `d_input` are
   computed on device per block (`slzSegHashKernel` +
   `slzChunkCombineKernel`), and `slzMerkleRootWriteKernel` writes
   the 4-byte root trailer directly into the device-resident frame
   before the final D2H. Host fallback in `format/xxhash32.zig` when
   the kernels are unavailable.

v4 #17: at L1/L2 the host-side LZ payload gather is SKIPPED entirely
(nothing on the host consumes it; the device assemble reads
`d_output` in place). At L3+ the gather rides persistent pinned
staging with async copies.

`SLZ_PROFILE_PHASES=1` prints per-phase wall times after `-b`
(enc_phase.zig). The LZ chain parser is ~86% of L5 encode wall and
is MEASURED-FUNDAMENTAL (v4 #14, parked): 2.1 chain candidates and
4.7 extend-bytes per findMatchChain call leave nothing for a warp to
parallelize; faster L5 encode means a different parse algorithm
(see the #14 entry and the #8 wire-format discussion).

---

## Decode pipeline

`decompressFramed` walks the frame on the host;
`dispatchCompressedBlock` builds `ChunkDesc`s and calls
`fullGpuLaunchImpl`, which runs (L3+; L1/L2 skip the entropy stages
entirely): prefix-sum → scan-parse → **fused 5-way compact**
(`slzCompactAllDescsKernel`: 4 huff streams + raw in one launch) →
gather-raw-off16 → **parallel merge** (`slzMergeHuffDescsParKernel`,
4 blocks) → Huffman LUT build + 32-stream decode → the K=4 pipelined
LZ kernel (`slzLzDecodeGeneralPipelinedKernel`, or
`slzLzDecodeRawPipelinedKernel` for L1/L2; one chunk per 128-thread
block, parser warp + 3-warp copier team - see GPU_ARCHITECTURE).
`SLZ_NO_PIPELINE=1` falls back to the single-warp kernels. A-024:
huff-decode region offsets are applied in-kernel as u64.

Integrity verification (v4 #19) runs after the LZ kernel on the aux
stream, overlapped with the output D2H: `slzScPrefixApplyKernel`
(makes the output final on device), `slzSegHashKernel` +
`slzChunkCombineKernel` (per-chunk hashes), `slzMerkleVerdictKernel`
(root compare). The host reads a 4-byte verdict; mismatch returns
`error.ChecksumMismatch`.

The pure-D2D entry is `decompressFramedFromDevice` (frame walk on
GPU via `gpuWalkFrameImpl`; used by `slzDecompressAsync`).

`SLZ_PROFILE_DECODE=1` on `-db` prints a per-kernel best-of-runs
table from the cuEvent pairs.

---

## Wire format

Constants are split between two locations and must stay in sync:

* `src/format/streamlz_constants.zig` - host-side
* `src/common/gpu_wire_format.cuh` + `gpu_huffman.cuh` - device-side

See [docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md) for the
sub-chunk header layout, the off16 hi/lo split, and the BIL format.
The wire format is byte-identical across CUDA and VK; the
cross-backend SHA gate in [CLAUDE.md](CLAUDE.md) enforces it after
encoder changes.

---

## Build

```
zig build                                 ReleaseFast streamlz.exe
zig build ptx                             recompile STALE .cu -> .ptx via
                                          nvcc+vcvarsall (no-op when fresh;
                                          replaces the touch-the-PTXs dance)
zig build gpulib                          streamlz_gpu.dll only
zig build test | ptest                    parallel unit tests (installs the
                                          exe first - cli_smoke spawns it)
zig build streamlz_vk | ptest_vk          Vulkan backend + its test suite
zig build fuzz | stress18 | chaincount    measurement/fuzz tools
zig build tans_gate2                      v4 #11 replay tool
zig build run -- -l 3 in.bin -o out.slz
```

PTX is committed, so plain `zig build` needs no CUDA toolchain. The
freshness gate fails the build when any `.cu`/`.cuh` is newer than
any `.ptx` and points at `zig build ptx`. `tools/build_gpu.bat`
remains for full rebuilds with cuobjdump res-usage printouts.

At milestones, run `tools\sanitize.bat` (memcheck) and
`tools\sanitize.bat racecheck` - both clean as of 2026-06-10.

---

## CUDA loading

`nvcuda.dll` loads at runtime via `LoadLibraryA`; missing driver →
`error.BackendNotAvailable` / `SLZ_ERROR_CUDA` (no CPU fallback).
Encode's `init()` defers to decode's so exactly ONE CUDA context
exists per process; both inits are SRWLOCK-guarded (a second thread
arriving mid-bring-up blocks instead of mis-reporting "unavailable").

**The context is per-thread current.** Any code running raw driver
calls from a thread that never bound it gets
`CUDA_ERROR_INVALID_CONTEXT (201)`. The C ABI binds inside every
job; tests get it from `lockGpuTests()`; new thread-crossing code
must call `gpu_driver.bindContextToCallingThread()`.

---

## Levels

L1-L5, all GPU. Policy lives in `src/encode/levels.zig`:

| Level | Parser | hash_bits | Huffman | LZ rehash |
|-------|--------|-----------|---------|-----------|
| L1 | greedy | 17 | no | no |
| L2 | greedy | 17 | no | yes (v4 #6) |
| L3 | greedy | 17 | yes | no |
| L4 | greedy | 17 | yes | yes |
| L5 | lazy chain | 17 | yes | yes |

hash_bits is 17 at EVERY level since 2026-06-09: at sc=0.25 a 1 GB
input is ~15k chunks, and any larger per-chunk hash overflows
consumer VRAM and collapses encode via WDDM paging; measured ratio
cost of the cap is ≈0.0-0.1 pp. Levels outside 1..5 →
`error.BadLevel`.

---

## Performance (RTX 4060 Ti, sm_89 - always print the Device line)

Headline (2026-06-11, enwik9 1 GB, decode-kernel best): **L1 52.6%
@ 16.3 ms (61 GB/s) vs nvCOMP LZ4 53.6% @ 33 ms (2.0x); L5 35.50% @
26.3 ms vs nvCOMP Zstd 35.75% @ 50.8 ms (1.9x) - better ratio AND
faster decode on both axes.** enwik8 100 MB: L5 kernel 2.94 ms, e2e
~15.5 ms including the default-on checksum verification; encode L1
88 ms (1.1 GB/s), L5 ~295 ms (chain parser ~86% of it, measured
fundamental - v4 #14).

Numbers move; regenerate with `-db -r 10` (+`SLZ_PROFILE_DECODE=1`
for the per-kernel table) rather than trusting tables in docs. GPU
benches run SERIALLY, never in parallel (project rule).

---

## Invariants - do not break these

1. **One `.cu` → one `.ptx`.** The `.cuh` files are a size-only
   split, `#include`d into a single TU.
2. **Kernels are bound by string** via `cuModuleGetFunction`.
   Renaming needs the symbol + driver string + regenerated `.ptx`.
3. **All kernel symbols use the `slz` prefix.**
4. **Descriptor structs mirror Zig `extern struct`s** with
   `static_assert(sizeof(...))` guards on the CUDA side.
5. **Committed `.ptx` are intentional** (clone builds without nvcc);
   `.cubin` is git-ignored.
6. **Wire-format constants live in one place per side** (see Wire
   format above); `static_assert`s catch drift.
7. **`SLZ_GUARD_SINGLE_THREAD()` also rejects `blockIdx.x != 0`** -
   multi-block single-thread kernels (the fused compact, the
   parallel merge) must use a thread-only guard.
8. **Every VK↔CUDA divergence gets an A-NNN entry** in
   `srcVK/PortAdaptations.md`, and encoder changes pass the
   cross-backend SHA gate (CLAUDE.md).
