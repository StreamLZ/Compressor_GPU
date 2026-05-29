# StreamLZ

GPU-accelerated LZ77 compressor + decompressor. CUDA kernels (NVIDIA
Driver API, target `sm_89`) do the per-chunk LZ work and 32-stream
Huffman decode; thin Zig drivers manage the kernel launches and the
host-side wire-format assembly.

There is one backend: CUDA. The codec was previously available as a
CPU implementation, byte-exact with a C# reference, plus a Vulkan
compute-shader decoder fallback. Both were retired on 2026-05-29 so
the project could maintain one codec rather than three; see
[FAILED_EXPERIMENTS.md](FAILED_EXPERIMENTS.md) "Maintaining parallel
CPU and GPU codebases" for the rationale.

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
tools\build_gpu.bat
```

The script invokes `nvcc -arch=sm_89 -O3` on the five CUDA
translation units, then re-runs `zig build` so the freshly-built PTX
is embedded. `build.zig` enforces a freshness gate that fails the
build if any source is newer than any PTX.

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

Best-of-30 decode on an RTX 4060 Ti (sm_89), `streamlz -db -r 30`.
Re-run by `tools\bench_all.bat`.

### Decode (ms): D2D wall-clock and end-to-end

| Level | enwik8 D2D / e2e | silesia D2D / e2e |
|-------|------------------|-------------------|
| L1 | **2.92** / 15.49 | **5.08** / 29.89 |
| L2 | **2.93** / 15.48 | **5.08** / 29.88 |
| L3 | **4.10** / 15.54 | **7.15** / 30.53 |
| L4 | **3.94** / 15.30 | **6.97** / 30.30 |
| L5 | **4.12** / 15.24 | **7.71** / 30.38 |

D2D wall-clock = the time a device-resident caller of
`slzDecompressAsync` sees on the wire. End-to-end adds the host-to-
device upload of the compressed frame + device-to-host download of
the decompressed output for the host-bounce path.

### Compression ratio

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.7% | 38.0% |
| L4 | 42.7% | 37.5% |
| L5 | 39.6% | 33.9% |

L1-L2 are LZ-only (no entropy stage). L3-L5 add 32-stream GPU Huffman.

### vs nvCOMP (enwik8 100 MB, RTX 4060 Ti)

| Window | StreamLZ L1 | nvCOMP LZ4 | StreamLZ win |
|--------|------------:|-----------:|-------------:|
| Pipeline kernel-sum | 4.03 ms | 4.77 ms | 1.18× |
| Async call wall     | 4.61 ms | 4.77 ms | 1.03× |
| End-to-end host wall | 15.51 ms | 18.29 ms | 1.18× |

| Window | StreamLZ L5 | nvCOMP Zstd | StreamLZ win |
|--------|------------:|------------:|-------------:|
| Pipeline kernel-sum | 5.50 ms | 6.25 ms | 1.14× |
| Async call wall     | 5.94 ms | 6.25 ms | 1.05× |
| End-to-end host wall | 15.27 ms | 18.16 ms | 1.19× |

See [docs/GPU_README.md](docs/GPU_README.md) "vs nvCOMP" for the
methodology behind each measurement window — the pipeline / async /
end-to-end columns answer different questions and confusing them is
easy.

---

## Project layout

```
build.zig              Zig 0.16 build script (always builds GPU)
include/streamlz_gpu.h C ABI public header
src/                   All Zig + CUDA source. See CodeWiki.md for the
                       full per-file map.
  common/              CUDA headers #include'd by every kernel
  format/, io/, platform/  Host-side helpers shared by encode + decode
  encode/              GPU encode driver + kernels + entropy helpers
  decode/              GPU decode driver + kernels
docs/                  GPU_ARCHITECTURE.md, GPU_README.md, GPU_IDEAS.md
tools/                 Build scripts + bench harnesses
CodeWiki.md            Source tree map + invariants
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
