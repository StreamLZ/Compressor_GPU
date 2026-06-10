# StreamLZ

GPU-accelerated LZ77 compressor + decompressor. CUDA kernels (NVIDIA
Driver API, target `sm_89`) do the per-chunk LZ work and 32-stream
Huffman decode; thin Zig drivers manage the kernel launches and the
host-side wire-format assembly.

There are two sibling GPU backends producing byte-identical frames:
CUDA (`src/`, `streamlz.exe`) and Vulkan (`srcVK/`,
`streamlz_vk.exe`, built via `zig build streamlz_vk`). There is no
CPU codec — the historical CPU implementation and an earlier partial
Vulkan port were retired (see
[FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md) "Maintaining parallel
CPU and GPU codebases" for the rationale; the current `srcVK/` tree
is a complete 1:1 port at full L1-L5 parity).

---

## Quick start

```
zig build -Doptimize=ReleaseFast
```

That produces `zig-out/bin/streamlz.exe` (the CLI) and
`zig-out/bin/streamlz_gpu.dll` (the C ABI library that game engines /
ML pipelines link against).

```
streamlz file.txt                # compress (default L1)
streamlz -l 3 file.txt           # compress at level 3
streamlz -d file.slz             # decompress
streamlz -b -l 5 file.txt        # compress + decompress + verify
streamlz -ba file.txt            # sweep L1-L5, ratio + throughput
streamlz -db file.slz            # decompress-only benchmark
streamlz -i  file.slz            # frame / block header dump
```

Levels: L1-L5. Higher = better ratio, slower encode. Decode speed is
roughly the same across all five (within ±0.5 ms on a 100 MB input).

The PTX kernel images are committed under `src/encode/` and
`src/decode/`, so plain `zig build` does not require CUDA installed
— the Zig drivers `@embedFile` the PTX. The codec needs `nvcuda.dll`
at runtime; if it is missing or `cuInit` fails, the codec returns
`error.BackendNotAvailable`.

To rebuild PTX after editing any `.cu` / `.cuh`:

```
zig build ptx          # recompiles exactly the stale translation units
```

(`tools\build_gpu.bat` does the full rebuild with cuobjdump
res-usage printouts.) `build.zig` enforces a freshness gate that
fails the build if any source is newer than any PTX.

---

## Supported hardware

**CUDA backend (`streamlz.exe`, `streamlz_gpu.dll`)** — NVIDIA only.
The committed PTX targets `sm_89`, so the driver JIT requires compute
capability ≥ 8.9: **RTX 40-series (Ada) or newer**. Older NVIDIA GPUs
need a PTX rebuild with a lower `-arch` (untested). Requires
`nvcuda.dll` at runtime.

**Vulkan backend (`streamlz_vk.exe`)** — any Vulkan 1.2+ device that
can pin `subgroupSize = 32` (the kernels are written warp-for-warp
against the CUDA originals) and exposes `bufferDeviceAddress` +
8-bit storage features:

| Vendor | Supported | Not supported |
|--------|-----------|---------------|
| NVIDIA | Turing and newer | |
| AMD | RDNA1 and newer (RX 5000+, wave32-capable) | GCN (wave64-only): Polaris RX 400/500, Vega |
| Intel | UHD/Iris iGPUs, Arc | |

Incompatible devices are rejected at init with a message naming the
device and its subgroup range; `streamlz_vk --probe` lists every
device with its range so you can pick a compatible one via
`--device <N|name>` (or `SLZ_VK_DEVICE_INDEX=N`).

Verified configurations: NVIDIA RTX 4060 Ti (both backends, primary
perf target), Intel(R) Graphics iGPU (Vulkan, correctness), AMD
RX 590/RX 550 (correctly rejected — wave64-only).

---

## C ABI library

The handle-based C library (`include/streamlz_gpu.h`) mirrors the
nvCOMP shape:

```c
slzContext_t ctx;
slzCreate(&ctx);
size_t bound;
slzCompressBound(ctx, src_size, slzCompressDefaultOpts(), &bound);
// allocate dst...
size_t comp_size;
slzCompressHost(ctx, src, src_size, dst, bound, &comp_size, slzCompressDefaultOpts());
// later:
slzDecompressHost(ctx, comp, comp_size, out, out_size, &written, slzDecompressDefaultOpts());
slzDestroy(ctx);
```

Two buffer models:

* `slzCompressHost` / `slzDecompressHost` — caller's data on host;
  the pipeline does its own H2D + D2H.
* `slzCompressAsync` / `slzDecompressAsync` — caller's data is
  GPU-resident; the codec submits all work on the caller's CUstream
  and never bounces through host memory.

`tools/slz_gpu_d2d_bench.c` is a worked example of the async
device-resident path.

---

## Performance

Best-of-30 decode on an RTX 4060 Ti (sm_89), `streamlz -db -r 30`,
re-measured 2026-06-10 (post compact-fusion + parallel-merge
backports). Re-run by `tools\bench_all.bat`.

### Decode (ms): D2D wall-clock and end-to-end

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | **2.85** / 15.51 | **4.94** / 30.03 |
| L2 | **2.84** / 15.47 | **4.94** / 29.92 |
| L3 | **4.01** / 15.64 | **6.99** / 30.64 |
| L4 | **3.87** / 15.43 | **6.86** / 30.37 |
| L5 | **4.06** / 15.38 | **7.44** / 30.38 |

D2D wall-clock = the time a device-resident caller of
`slzDecompressAsync` sees on the wire. End-to-end adds the host-to-
device upload of the compressed frame + device-to-host download of
the decompressed output for the host-bounce path.

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.7% | 38.1% |
| L4 | 42.7% | 37.5% |
| L5 | 39.6% | 33.9% |

L1-L2 are LZ-only (no entropy stage). L3-L5 add 32-stream GPU Huffman.

### vs nvCOMP (enwik8 100 MB, RTX 4060 Ti)

| Window | StreamLZ L1 | nvCOMP LZ4 | StreamLZ win |
|--------|------------:|-----------:|-------------:|
| Pipeline kernel-sum | **2.95 ms** | 4.77 ms | 1.62× |
| Async call wall     | **4.04 ms** | 4.77 ms | 1.18× |
| End-to-end host wall | **15.51 ms** | 18.29 ms | 1.18× |

| Window | StreamLZ L5 | nvCOMP Zstd | StreamLZ win |
|--------|------------:|------------:|-------------:|
| Pipeline kernel-sum | **4.39 ms** | 6.25 ms | 1.42× |
| Async call wall     | **5.47 ms** | 6.25 ms | 1.14× |
| End-to-end host wall | **15.38 ms** | 18.16 ms | 1.18× |

StreamLZ columns re-measured 2026-06-10; nvCOMP columns are the
2026-05-27 `nvcomp_bench3` runs (our changes don't affect them).

See [docs/cudaOptimize.md](docs/cudaOptimize.md) "vs nvCOMP -
measurement methodology" for what each window measures - the
pipeline / async / end-to-end columns answer different questions and
confusing them is easy.

### vs nvCOMP (enwik9 1 GB, RTX 4060 Ti, re-measured 2026-06-10)

At 1 GB scale StreamLZ wins ratio AND decode speed simultaneously at
both ends of the level range (decode kernel = cuEvent best-of-30 for
StreamLZ, best-of-20 for nvCOMP; e2e = host wall incl. PCIe both ways):

| | StreamLZ L1 | nvCOMP LZ4 | margin |
|--------|------------:|-----------:|-------:|
| Ratio | **52.6%** | 53.6% | 10 MB smaller |
| Decode kernel | **24.3 ms** (39.3 GB/s) | 33.0 ms (30.3 GB/s) | 1.36× |
| Decode e2e | **146.0 ms** | 162.0 ms | 1.11× |

| | StreamLZ L5 | nvCOMP Zstd | margin |
|--------|------------:|------------:|-------:|
| Ratio | **35.50%** | 35.75% | 2.5 MB smaller |
| Decode kernel | **34.0 ms** (28.0 GB/s) | 50.8 ms (19.7 GB/s) | 1.49× |
| Decode e2e | **145.0 ms** | 164.4 ms | 1.13× |

These numbers use the 2026-06-09 defaults: `sc_group_size = 0.25` at
every input size (64 KB sub-chunks — more decode warps, shorter
per-warp serial chains) and `hash_bits = 17` at every level. Pass
`--sc 0.5` for ~2 pp better ratio at ~1.8× slower 1 GB-scale decode.

---

## Project layout

```
build.zig              Zig 0.16 build script (always builds GPU)
include/streamlz_gpu.h C ABI public header
src/                   CUDA backend. See CodeWiki.md for the full
                       per-file map.
  common/              CUDA headers #include'd by every kernel
  format/              Host-side wire-format parsers/writers
  encode/              GPU encode driver + kernels
  decode/              GPU decode driver + kernels
srcVK/                 Vulkan backend (full 1:1 port). See
                       srcVK/README.md + srcVK/Handbook.md.
docs/                  GPU_ARCHITECTURE.md, cudaOptimize.md, how_to_debug_cuda.md
tools/                 Build scripts + bench harnesses
CodeWiki.md            Source tree map + invariants
v4_ideas.md            Forward-looking work list
FORMAT.md              SLZ1 wire format specification
CHANGELOG.md           Release history
FAILED_EXPERIMENTS.md  Rejected experiments + war stories
SECURITY.md            Security policy + threat model
```

For the source tree map, kernel pipeline, level table, and the six
"do not break" invariants, see [CodeWiki.md](CodeWiki.md).

For the algorithmic notes (warp mapping, BIL Huffman wire format,
LUT-build kernel, parallel-parse hot loop), see
[docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md).

---

## License

MIT.
