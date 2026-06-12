# StreamLZ Contributor Guide

StreamLZ is a GPU LZ77 codec with two sibling backends that produce
byte-identical compressed frames: a CUDA backend in `src/` and a
Vulkan port in `srcVK/`. Zig code drives the kernels and handles the
wire format on the host; the heavy work runs on the GPU in both
directions.

This page is about the code: what every file is, where your change
goes, and which rules will get a change rejected. (For the index of
all project documentation, see the README's Documentation section.)
If you remember one thing, remember this: **the two backends are
each other's test oracle.** Most rules below exist to protect that
property.

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

## Source layout

Repository root:

- `src/` - the CUDA backend (detailed below)
- `srcVK/` - the Vulkan backend; same structure, own docs
- `include/streamlz_gpu.h` - the C ABI header both backends implement
- `tools/` - benchmark, sanitizer, profiling, and fuzzing harnesses
- `docs/` - design and tooling notes (see the document map above)
- `assets/` - test corpora
- `build.zig` - builds all of the above

### `src/` - top level

- `main.zig` - CLI entry point; also the root that pulls every test file into `zig build test`
- `cli.zig` and `cli/` - command-line parsing and one handler file per mode: `compress.zig`, `decompress.zig`, `info.zig` (frame header dump), `bench_compress.zig` (`-b`), `bench_decompress.zig` (`-db`), `bench_all.zig` (`-ba`), `util.zig` (shared arg helpers)
- `streamlz_gpu.zig` - implements the C ABI; the root file of `streamlz_gpu.dll`
- `c_abi_tests.zig` - calls the DLL surface through extern declarations, exactly like a C program would
- `cli_smoke_tests.zig` - spawns the built `streamlz.exe` and checks real command lines end to end
- `test_runner_parallel.zig` - the custom test runner behind `zig build ptest`; runs tests in parallel and names every skipped one
- `stress18_main.zig` - standalone encode/decode loop built to chase an intermittent roundtrip mismatch (`zig build stress18`)
- `chain_count_main.zig` - runs one level-5 encode and prints how much work the match-finder did (`zig build chaincount`)
- `dict_gate0_main.zig` - trains preset dictionaries at several sizes and measures their compression-ratio lift on a small-records corpus through the production encoder (`zig build dict_gate0`)
- `dict_bench_main.zig` - per-record dictionary benchmark: encodes every record of a corpus plain and with a dictionary, decodes and byte-verifies, reports ratio + throughput (`zig build dict_bench`)
- `version.zig`, `mmap.zig` - version string; memory-mapped file IO

### `src/dict/` - preset dictionaries

- `dictionary.zig` - the dictionary registry: built-in dictionaries compiled in via `@embedFile`, identified by the well-known IDs carried in the frame header (flag bit 3). Both encoder and decoder resolve IDs through it; an ID permanently names exact bytes (retraining means a new ID). Mirrored at `srcVK/dict/`.
- `builtin/*.dict` - the built-in dictionary assets. IDs 1-7 are shared byte-for-byte with the CPU sibling project so frames are dictionary-compatible across the two; ID 8 (github-users) is StreamLZ-trained at the measured 2 KB ratio knee.
- `trainer.zig` - FASTCOVER dictionary trainer (the zstd algorithm, ported from the CPU sibling project); selects the most frequently-matched segments from sample records and packs them into a raw dictionary, best content at the tail. Host-only, no GPU interaction.

How a dictionary flows through the codec: the encoder resolves the
ID, builds a position hash table on the host (`encode_lz.zig
ensureDictOnDevice`, cached per context), and the greedy parser
probes it as its lowest-priority match source - a dictionary match
is an ordinary off16 token whose distance reaches below the
sub-chunk window. The decoder resolves the same ID from the frame
header, uploads the bytes once (`decode_context.zig
ensureDictOnDevice`), and the LZ kernels' dict-templated copy paths
(`readBackRefByte` in `lz_decode_core.cuh`) map below-window reads
onto the dictionary tail. The L5 chain parser does not search the
dictionary yet (accepts dict frames; no ratio benefit).

### `src/common/` - headers shared by encode and decode kernels

- `gpu_warp.cuh` - warp and lane index helpers, plus the guard macro that makes single-thread kernels reject every thread but one (including extra blocks)
- `gpu_byteio.cuh` - unaligned little-endian loads and stores
- `gpu_wire_format.cuh` - the LZ token and header constants, device side
- `gpu_huffman.cuh` - the canonical-Huffman constants, device side
- `xxh32_device.cuh` - the XXH32 hash function for GPU threads, plus the kernel bodies for the per-chunk integrity checksum (compiled into both the encode and decode kernel modules)

### `src/format/` - the wire format on the host side

- `streamlz_constants.zig` - the host-side twin of `gpu_wire_format.cuh`; every constant exists in both and asserts keep them equal
- `frame_format.zig` - frame header read/write: magic, version, flags, optional content size, trailers
- `block_header.zig` - block headers and the 2-byte internal block header
- `xxhash32.zig` - XXH32 and the chunk-checksum root computation on the host (the fallback when the GPU kernels are unavailable, value-identical by definition)

### `src/encode/`

- `streamlz_encoder.zig` - the public API: `compressFramed`, `compressBound`, the options struct (level, checksums, chunk sizing)
- `fast_framed.zig` - the frame builder; calls every stage below in order and owns the frame-level bookkeeping
- `levels.zig` - what each level means: parser choice, entropy on or off, hash-table size
- `encode_context.zig` - the context struct holding every persistent device buffer between calls
- `encode_lz.zig` - launches the LZ match-finding kernel, batched so the per-chunk hash tables fit in VRAM
- `encode_huff.zig` - launches the Huffman table-build and encode kernels (levels 3-5)
- `encode_assemble.zig` - launches the measure/write/assemble kernels that splice the final frame together in GPU memory
- `driver.zig`, `cuda_ffi.zig`, `module_loader.zig` - public facade, the `nvcuda.dll` function-pointer table, and PTX loading with thread-safe one-time init
- `enc_phase.zig` - per-phase wall-clock profiler (`SLZ_PROFILE_PHASES=1`)
- `lz_kernel.cu` - the LZ encode kernel module; parsers split into `lz_greedy_parser.cuh` (levels 1-4), `lz_chain_parser.cuh` (level 5), shared pieces in `lz_format.cuh` and `lz_token_emit.cuh`
- `huffman_kernel.cu`, `assemble_kernel.cu` - the Huffman encode and frame-assembly kernel modules
- tests: `gpu_roundtrip_tests.zig` (many shapes and levels through encode+decode, plus the lock that serializes GPU tests), `gpu_regression_tests.zig` (cases that once broke), `huff_conformance_tests.zig` (Huffman output byte-identity), `l5_hardening_tests.zig` (adversarial inputs for the level-5 parser)

### `src/decode/`

- `streamlz_decoder.zig` - the public API: `decompressFramed`, frame walking, trailer verification, error surface
- `dict_vector_tests.zig` - hand-crafted dictionary-frame decode vectors generated from the FORMAT spec alongside a sequential reference model; proves the kernels' dictionary reach (straddle, recent-offset, off32, hostile clamp) independent of the encoder
- `decode_dispatch.zig` - the heart of decode: one function that launches the whole kernel sequence for a block, including the integrity-check kernels
- `decode_context.zig` - persistent device buffers, stream/event handles, per-kernel timing capture
- `descriptors.zig` - the chunk descriptor structs shared with the kernels, and the error set
- `scan_gpu.zig`, `scan_host.zig` - sub-chunk header scanning on device, with the host fallback
- `driver.zig`, `cuda_api.zig`, `module_loader.zig` - facade, driver function table, PTX loading (also owns the single CUDA context)
- `lz_kernel.cu` - the LZ decode kernel module; the pipelined hot path is `lz_decode_raw_pipeline.cuh`, the single-warp originals are `lz_decode_raw.cuh` and friends, header parsing in `lz_header_parse.cuh`, format constants in `slz_wire_format.cuh`
- `huffman_kernel.cu` - decode-side LUT build and the 32-stream Huffman decode
- bookkeeping kernels, one file each: `walk_frame_kernel.cuh` (parse the frame on device), `prefix_sum_chunks_kernel.cuh`, `scan_parse_kernel.cuh`, `compact_descs_kernels.cuh`, `merge_huff_descs_kernel.cuh`, `gather_raw_off16_kernel.cuh`

Compiled `.ptx` files are committed next to their `.cu` sources.

### `tools/`

- `bench_all.bat` - encode+decode both corpora at all levels, SHA-verified; the canonical "is everything still byte-exact" check
- `bench_d2d.bat`, `build_d2d_bench.bat`, `slz_gpu_*.c` - the same through the C ABI with device-resident buffers
- `sanitize.bat` - compute-sanitizer memory and race checking
- `build_gpu.bat` - full nvcc rebuild with per-kernel register/memory usage printouts
- `ncu_profile_lz.bat` - Nsight Compute profile of the decode kernels (needs an admin shell)
- `fuzz_frames.zig` - mutates valid frames and decodes them on both backends, comparing outcomes (`zig build fuzz`)
- `huff_test/` - standalone harness for the Huffman kernels
- `tans_gate2/` - measurement harness that compares Huffman against a tANS entropy coder on real stream data

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
mixed-content corpus, and `github_users.jsonl` is a 9k-record
small-JSON corpus - one ~820-byte record per line - for the
dictionary work.)

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
