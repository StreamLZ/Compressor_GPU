# StreamLZ Contributor Guide

StreamLZ is a GPU LZ77 codec with two sibling backends that produce
byte-identical compressed frames: a CUDA backend in `src/` and a
Vulkan port in `srcVK/`. Zig code drives the kernels and handles the
wire format on the host; the heavy work runs on the GPU in both
directions.

This page routes you: what to read, where your change goes, and
which rules will get a change rejected. If you remember one thing,
remember this: **the two backends are each other's test oracle.**
Most rules below exist to protect that property.

---

## The document map

| You want | Read |
|---|---|
| To use the codec: CLI, performance, integrity | [README.md](README.md) |
| The wire format, byte for byte | [FORMAT.md](FORMAT.md) |
| Why the kernels are shaped the way they are | [docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md) |
| CUDA debugging and profiling recipes | [docs/how_to_debug_cuda.md](docs/how_to_debug_cuda.md) |
| The Vulkan backend | [srcVK/README.md](srcVK/README.md) and [srcVK/Handbook.md](srcVK/Handbook.md) |
| Every known difference between the two backends | [srcVK/PortAdaptations.md](srcVK/PortAdaptations.md) |
| The work ledger: every idea, measurement, and verdict | [v4_ideas.md](v4_ideas.md) |
| Things that were tried and failed (do not retry them) | [FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md) |
| Safety guarantees for untrusted input | [SECURITY.md](SECURITY.md) |

---

## Where your change goes

**Changing the wire format.** Both backends and FORMAT.md must move
together, and existing frames must still decode. The format
constants live in exactly one place per side: host constants in
`src/format/streamlz_constants.zig`, device constants in
`src/common/gpu_wire_format.cuh` and `gpu_huffman.cuh`; compile-time
asserts catch drift between them. Mirror the change into the
`srcVK/` equivalents, then run the cross-backend gate (below).

**Changing the encoder.** Start at
`src/encode/streamlz_encoder.zig` (the public API), which hands off
to `fast_framed.zig` - the frame builder that orchestrates every
kernel launch for all five levels. The per-stage launchers are
`encode_lz.zig`, `encode_huff.zig`, and `encode_assemble.zig`;
level policy (which parser, which entropy stage) is `levels.zig`.
The kernels themselves are in `lz_kernel.cu` (with the parsers in
`lz_*.cuh` headers), `huffman_kernel.cu`, and `assemble_kernel.cu`.

**Changing the decoder.** Start at
`src/decode/streamlz_decoder.zig` (public API, frame walk, checksum
verification), which calls `decode_dispatch.zig` - the function
`fullGpuLaunchImpl` there launches the entire kernel sequence. The
hot decode loop lives in `lz_decode_raw_pipeline.cuh`; setting the
environment variable `SLZ_NO_PIPELINE=1` selects the older
single-warp kernels instead, which is useful for isolating pipeline
bugs. Device-resident callers enter through
`decompressFramedFromDevice`, which walks the frame on the GPU.

**Changing a kernel.** One `.cu` file compiles to one `.ptx` file;
the `.cuh` headers are just a size split, included into that single
translation unit. Kernels are looked up by name string at runtime,
so renaming one means changing the symbol, the Zig driver string,
and regenerating the `.ptx`. All kernel names start with `slz`. The
descriptor structs passed to kernels are hand-mirrored between Zig
and CUDA, with size asserts on the CUDA side. Compiled `.ptx` files
are committed to the repo so a fresh clone builds without a CUDA
toolchain; `zig build ptx` regenerates stale ones, and the build
fails loudly if a `.cu` is newer than its `.ptx`.

**Changing the C ABI.** `src/streamlz_gpu.zig` implements
`include/streamlz_gpu.h`. Two call styles: `slzCompressHost` /
`slzDecompressHost` run on a library-owned worker thread, and
`slzCompressAsync` / `slzDecompressAsync` operate on device-resident
buffers and ride the caller's CUDA stream. The tests in
`src/c_abi_tests.zig` bind through extern declarations - the same
shape a real C caller uses.

**Changing anything in `srcVK/`.** Port, don't reinvent: mirror the
CUDA values, dispatch shapes, and workgroup sizes. When a difference
is genuinely unavoidable (driver behavior, missing extension), it
gets a numbered entry in `srcVK/PortAdaptations.md` with evidence.
No silent forks.

---

## Build and test

```text
zig build                          builds streamlz.exe (no CUDA toolchain needed)
zig build ptx                      recompiles stale .cu -> .ptx (needs nvcc)
zig build test                     unit + integration tests (ptest = parallel)
zig build streamlz_vk              Vulkan backend (ptest_vk = its test suite)
zig build gpulib                   streamlz_gpu.dll only
zig build fuzz                     differential frame-mutation fuzzer
zig build run -- -l 3 in.bin -o out.slz
```

GPU tests skip cleanly on machines without a CUDA device. At
milestones, run `tools\sanitize.bat` (memory checker) and
`tools\sanitize.bat racecheck` (race checker).

---

## The gates

These checks decide whether a change lands.

(The test corpora live in `assets/`: `enwik8.txt` is a 100 MB
Wikipedia text dump, `silesia_all.tar` is the 213 MB Silesia
mixed-content corpus.)

1. **Cross-backend identity.** After any encoder change, encode
   enwik8 at levels 1, 3, and 5 on both backends and compare the
   output files by SHA-256. They must be byte-identical - even when
   both backends roundtrip their own output correctly, a frame
   divergence is a bug. Touch a level-specific path, test all five
   levels.
2. **Both test suites green**: `zig build ptest` and
   `zig build ptest_vk`.
3. **Real-corpus roundtrips**: `tools\bench_all.bat` encodes and
   decodes enwik8 and silesia at all levels and fails on the first
   SHA mismatch; `tools\bench_d2d.bat` does the same through the
   device-resident C ABI path.
4. **Honest measurements.** Performance numbers are only quotable
   from a run whose output shows which GPU ran it - on this
   project's multi-GPU dev box the Vulkan CLI silently picks the
   Intel iGPU unless `SLZ_VK_DEVICE_INDEX=1` is set. Run GPU
   benchmarks one at a time, never concurrently. And prefer
   re-measuring (`streamlz -db file.slz -r 10`; add the
   `SLZ_PROFILE_DECODE=1` environment variable for a per-kernel
   table) over trusting numbers written down in docs.

---

## How it works, in one minute

**Encode.** The frame builder writes a header, then runs the LZ
match-finding kernel (one warp per chunk - 64 KB of input each at
the default settings - dispatched in batches sized to fit VRAM), then - at level 3 and up - Huffman
kernels over the literal, token, and offset streams. A device-side
assembly pass splices everything into the final frame layout in GPU
memory, integrity hash kernels stamp a 4-byte checksum trailer into
that device-resident frame, and one transfer brings the finished
frame to the host. Set `SLZ_PROFILE_PHASES=1` to see the per-phase
time breakdown.

**Decode.** The host (or, for device-resident input, a GPU kernel)
walks the frame structure, then a chain of small bookkeeping kernels
sizes and sorts the per-chunk work, Huffman tables are built and the
entropy streams decoded 32 lanes wide, and the LZ kernel reconstructs
the output - each chunk handled by a 128-thread block in which one
warp parses tokens while three warps execute the copies a step
behind. Checksum kernels then re-hash the output on a side stream,
overlapped with the transfer back to the host, so corruption is
caught at ~zero cost. Set `SLZ_PROFILE_DECODE=1` for per-kernel
times.

**Levels.** L1 = greedy parse, LZ only. L2 = greedy plus a second
hashing pass over match ranges (~1 point better ratio, slightly
faster to DECODE than L1 because it emits fewer tokens). L3 = L1
plus Huffman. L4 = L2 plus Huffman. L5 = lazy chain parser plus
Huffman, the best ratio and the slowest encode - its parser is
inherently serial, which is a measured property, not an oversight
(v4_ideas.md entry 14 has the data).

**CUDA bring-up.** The NVIDIA driver is loaded at runtime
(`nvcuda.dll`); without it every entry point reports the backend
unavailable - there is no CPU fallback. The process owns exactly one
CUDA context, and the context binds per thread: code that calls
driver functions from a new thread must call
`gpu_driver.bindContextToCallingThread()` first or every call fails
with `CUDA_ERROR_INVALID_CONTEXT`. Tests handle this through
`lockGpuTests()`, which also serializes GPU tests against each
other.

---

## Performance anchor

As of 2026-06-11 on an RTX 4060 Ti: decoding enwik9 (a 1 GB text
corpus) takes
16.3 ms of GPU kernel time at level 1 (61 GB/s - twice as fast as
nvCOMP LZ4, at better compression) and 26.3 ms at level 5 (1.9x
nvCOMP Zstd, at better compression). Host-to-host on enwik8,
decode is ~15.5 ms including the always-on integrity check.
Encoding 100 MB: 88 ms at level 1, ~295 ms at level 5. Current
tables live in README.md; the Vulkan sweep is in
srcVK/PerfSweep.md.
