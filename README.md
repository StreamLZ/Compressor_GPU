# StreamLZ

GPU-accelerated LZ77 compressor + decompressor. CUDA kernels (NVIDIA
Driver API, target `sm_89`) do the per-chunk LZ work and 32-stream
Huffman decode; thin Zig drivers manage the kernel launches and the
host-side wire-format assembly.

There are two sibling GPU backends producing byte-identical frames:
CUDA (`src/`, `streamlz.exe`) and Vulkan (`srcVK/`,
`streamlz_vk.exe`, built via `zig build streamlz_vk`). There is no
CPU codec - the historical CPU implementation and an earlier partial
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
streamlz -D json records.json    # compress with a preset dictionary
streamlz -b -l 5 file.txt        # compress + decompress + verify
streamlz -ba file.txt            # sweep L1-L5, ratio + throughput
streamlz -db file.slz            # decompress-only benchmark
streamlz -i  file.slz            # frame / block header dump
```

Levels: L1-L5. Higher = better ratio, slower encode. Decode speed is
roughly the same across all five (within ±0.5 ms on a 100 MB input).

The PTX kernel images are committed under `src/encode/` and
`src/decode/`, so plain `zig build` does not require CUDA installed
- the Zig drivers `@embedFile` the PTX. The codec needs `nvcuda.dll`
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

**CUDA backend (`streamlz.exe`, `streamlz_gpu.dll`)** - NVIDIA only.
The committed PTX targets `sm_89`, so the driver JIT requires compute
capability ≥ 8.9: **RTX 40-series (Ada) or newer**. Older NVIDIA GPUs
need a PTX rebuild with a lower `-arch` (untested). Requires
`nvcuda.dll` at runtime.

**Vulkan backend (`streamlz_vk.exe`)** - any Vulkan 1.2+ device that
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
RX 590/RX 550 (correctly rejected - wave64-only).

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

* `slzCompressHost` / `slzDecompressHost` - caller's data on host;
  the pipeline does its own H2D + D2H.
* `slzCompressAsync` / `slzDecompressAsync` - caller's data is
  GPU-resident; the codec submits all work on the caller's CUstream
  and never bounces through host memory.

`tools/slz_gpu_d2d_bench.c` is a worked example of the async
device-resident path.

---

## Performance

Best-of-8+ decode on an RTX 4060 Ti (sm_89), `streamlz -db`,
re-measured 2026-06-11 with the pipelined decode kernels at every
level. Frames carry the default-on checksum, so the e2e column
INCLUDES integrity verification. Test corpora: enwik8 (a 100 MB
Wikipedia text dump) and silesia (a 213 MB mixed-content archive),
both in `assets/`. Re-run by `tools\bench_all.bat`.

### Decode (ms): D2D wall-clock and end-to-end

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | **1.75** / 15.50 | **4.04** / 31.01 |
| L2 | **1.83** / 15.48 | **4.31** / 31.26 |
| L3 | **2.87** / 15.68 | **6.02** / 31.92 |
| L4 | **2.92** / 15.66 | **6.26** / 32.00 |
| L5 | **2.94** / 15.45 | **5.94** / 31.05 |

D2D wall-clock = the time a device-resident caller of
`slzDecompressAsync` sees on the wire. End-to-end adds the host-to-
device upload of the compressed frame + device-to-host download of
the decompressed output for the host-bounce path, plus the
default-on checksum verification (computed on the GPU, overlapped
with the output download; see Integrity below).

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 57.3% | 47.2% |
| L3 | 43.7% | 38.1% |
| L4 | 42.7% | 37.5% |
| L5 | 39.6% | 33.9% |

L1-L2 are LZ-only (no entropy stage); L2 adds the greedy parser's
match-range rehash (~1 pp better ratio, +29% encode cost, decodes
slightly FASTER than L1 - fewer tokens). L3-L5 add 32-stream GPU
Huffman; L4 adds the rehash on top of L3; L5 swaps in the chain
parser.

### Integrity

Every frame carries a 4-byte chunk-Merkle checksum by default (wire
flag bit 5): per-chunk hashes are computed on the GPU at encode,
rolled into one root, and the decoder recomputes them from its own
output (also on the GPU, overlapped with the transfer) - corrupted
data returns `error.ChecksumMismatch` instead of wrong bytes. Cost
is ~1 ms per 100 MB on encode and ~zero on decode. Pass
`chunk_checksum = false` to strip it; `--checksum` additionally
enables the LZ4-style whole-file XXH32 (flag bit 1). See FORMAT.md.

### Preset dictionaries

Small records barely compress cold - every frame starts with no
history. A preset dictionary gives the match finder shared context
both sides already know: `streamlz -D <name>` on encode, automatic
on decode (the frame header names its dictionary; the wire never
carries dictionary bytes). Built-ins: json, html, text, xml, css,
js, general (shared with the CPU sibling project), and
github-users (trained on the bundled corpus). Train your own with
`zig build dict_gate0`; C callers set `dictionary_id` in
`slzCompressOpts_t`.

Per-record benchmark (`zig build dict_bench`, 4,557 GitHub-API JSON
records averaging 825 B, held-out from the dictionary's training
half, RTX 4060 Ti, encode+decode byte-verified):

| Level | Plain | With 2 KB trained dict | Improvement |
|-------|------:|-----------------------:|------------:|
| L1 | 57.5% | **28.5%** | 2.02x |
| L3 | 57.5% | **28.5%** | 2.02x |
| L5 | 52.4% | 52.9% (no dict search yet) | - |

Dictionary quality is per-corpus: the same records with the generic
json dictionary stay at ~57% - train on a sample of YOUR records.
The L5 chain parser does not search dictionaries yet (frames stay
valid; no ratio benefit). Dictionary-less frames are byte-identical
to before the feature existed.

### vs nvCOMP (enwik8 100 MB, RTX 4060 Ti)

| Window | StreamLZ L1 | nvCOMP LZ4 | StreamLZ win |
|--------|------------:|-----------:|-------------:|
| Pipeline kernel-sum | **1.88 ms** | 4.77 ms | 2.54× |
| Async call wall     | **3.13 ms** | 4.77 ms | 1.52× |
| End-to-end host wall | **15.50 ms** | 18.29 ms | 1.18× |

| Window | StreamLZ L5 | nvCOMP Zstd | StreamLZ win |
|--------|------------:|------------:|-------------:|
| Pipeline kernel-sum | **3.48 ms** | 6.25 ms | 1.80× |
| Async call wall     | **4.49 ms** | 6.25 ms | 1.39× |
| End-to-end host wall | **15.45 ms** | 18.16 ms | 1.18× |

StreamLZ columns re-measured 2026-06-11; nvCOMP columns were
measured 2026-05-27 with the harness in `tools/` (StreamLZ changes
do not move them).

See [docs/cudaOptimize.md](docs/cudaOptimize.md) "vs nvCOMP -
measurement methodology" for what each window measures - the
pipeline / async / end-to-end columns answer different questions and
confusing them is easy.

### vs nvCOMP (enwik9 1 GB, RTX 4060 Ti, re-measured 2026-06-11)

At 1 GB scale StreamLZ wins ratio AND decode speed simultaneously at
both ends of the level range (decode kernel = cuEvent best-of-30 for
StreamLZ, best-of-20 for nvCOMP; e2e = host wall incl. PCIe both ways):

| | StreamLZ L1 | nvCOMP LZ4 | margin |
|--------|------------:|-----------:|-------:|
| Ratio | **52.6%** | 53.6% | 10 MB smaller |
| Decode kernel | **16.3 ms** (61.4 GB/s) | 33.0 ms (30.3 GB/s) | 2.03× |
| Decode e2e | **148.9 ms** (incl. verification) | 162.0 ms | 1.09× |

| | StreamLZ L5 | nvCOMP Zstd | margin |
|--------|------------:|------------:|-------:|
| Ratio | **35.50%** | 35.75% | 2.5 MB smaller |
| Decode kernel | **26.3 ms** (38.0 GB/s) | 50.8 ms (19.7 GB/s) | 1.93× |
| Decode e2e | **148.9 ms** (incl. verification) | 164.4 ms | 1.10× |

These numbers use the current defaults: 64 KB sub-chunks (more
decode warps, shorter per-warp serial work) and a 17-bit match hash
at every level. Pass `--sc 0.5` for ~2 percentage points better
ratio at ~1.8× slower 1 GB-scale decode.

---

## Project layout

- `src/` - the CUDA backend (per-file map in [CodeWiki.md](CodeWiki.md))
- `srcVK/` - the Vulkan backend, a full 1:1 port
- `include/streamlz_gpu.h` - the C ABI header both backends implement
- `tools/` - build scripts + bench/sanitize/fuzz harnesses
- `docs/` - design and tooling notes
- `assets/` - test corpora
- `build.zig` - builds everything

## Documentation

| You want | Read |
|---|---|
| To work on the code: layout, where changes go, the gates | [CodeWiki.md](CodeWiki.md) |
| The wire format, byte for byte | [FORMAT.md](FORMAT.md) |
| Why the kernels are shaped the way they are | [docs/GPU_ARCHITECTURE.md](docs/GPU_ARCHITECTURE.md) |
| CUDA debugging and profiling recipes | [docs/how_to_debug_cuda.md](docs/how_to_debug_cuda.md) |
| The Vulkan backend | [srcVK/README.md](srcVK/README.md) and [srcVK/Handbook.md](srcVK/Handbook.md) |
| Every known difference between the two backends | [srcVK/PortAdaptations.md](srcVK/PortAdaptations.md) |
| The work ledger: every idea, measurement, and verdict | [v4_ideas.md](v4_ideas.md) |
| Things that were tried and failed (do not retry them) | [FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md) |
| Safety guarantees for untrusted input | [SECURITY.md](SECURITY.md) |
| Release history | [CHANGELOG.md](CHANGELOG.md) |

---

## License

MIT.
