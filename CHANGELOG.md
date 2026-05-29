# Changelog

## [3.0.0] — 2026-05-29

GPU-only release. The CPU codec (all 11 levels, parallel decoder,
streaming compressor, dictionary subsystem, parallel-decode sidecar)
and the Vulkan compute-shader decode fallback have been removed; the
CUDA codec is the only backend.

### Breaking changes

- **CPU backend removed.** Hosts without CUDA can no longer use this
  library. `nvcuda.dll` is loaded at runtime; missing CUDA returns
  `error.BackendNotAvailable` / `SLZ_ERROR_CUDA`.
- **Vulkan decoder removed.** `vulkan_driver.zig` + the `.comp` shader
  + `.spv` artifacts are deleted. CUDA only.
- **Levels L6-L11 removed.** The High codec (optimal parser, BT4
  match finder, full entropy stage) was CPU-only. Encoders return
  `error.BadLevel` for any level outside 1..5.
- **Dictionary subsystem removed.** The seven built-in dictionaries,
  the FASTCOVER trainer, the `-D` / `--no-dict` / `--train` CLI
  flags, and the `dictionary` / `dictionary_id` encoder options are
  gone. Frames carrying a `dictionary_id` are rejected with
  `error.UnknownDictionary`.
- **CPU-comparison benchmark modes removed.** `-bc` / `-bcf` and the
  `--zstd` / `--lz4` engine selectors are gone, along with the
  vendored `zstd 1.5.7` + `LZ4 1.10.0` source trees.
- **CPU C ABI (`capi.zig` / `include/streamlz.h`) removed.** The GPU
  C ABI in `src/streamlz_gpu.zig` + `include/streamlz_gpu.h` is the
  only public C surface.
- **Build flags removed.** `-Dgpu`, `-Dbench` are gone; `zig build`
  is unconditionally GPU. The `safe`, `fuzz`, `lib` build steps and
  the ReleaseSafe + fuzz harness are removed.

### Reorganization

- **`src/gpu/` flattened up to `src/`.** `src/gpu/common/` →
  `src/common/`, `src/gpu/encode/` → `src/encode/`, `src/gpu/decode/`
  → `src/decode/`. The `.cu` / `.cuh` `#include "../common/..."`
  paths stay byte-identical because the relative move preserves
  them.
- **Orchestration files rewritten clean.** `src/encode/fast_framed.zig`
  (2125 LOC → ~660 LOC), `src/encode/streamlz_encoder.zig` (1093
  → 195), `src/decode/streamlz_decoder.zig` (1611 → ~425), `src/cli.zig`
  (1878 → ~660). Host-side wire-format helpers extracted into a
  focused `src/encode/gpu_stream_assembly.zig` module.
- **GPU docs moved.** `src/gpu/ARCHITECTURE.md` → `docs/GPU_ARCHITECTURE.md`
  (kept verbatim; the algorithmic notes still apply).
  `src/gpu/README.md` → `docs/GPU_README.md`. `src/gpu/IDEAS.md` →
  `docs/GPU_IDEAS.md`.

### Verification

`tools/bench_all.bat` (enwik8 100 MB + silesia 213 MB, L1-L5,
`-db -r 30`, RTX 4060 Ti sm_89) pre- and post-strip. All ten rows
SHA-256 OK. D2D wall-clock stayed within ±0.05 ms of the
pre-strip baseline; e2e within ±0.23 ms; ratios within ±0.04 pp.
Most deltas were small improvements (the simpler dispatch shaved a
handful of microseconds per call).

### Performance highlights — RTX 4060 Ti, enwik8 100 MB

| Level | D2D wall ms | e2e ms | ratio |
|-------|------------:|-------:|------:|
| L1 | 2.92 | 15.49 | 58.6% |
| L3 | 4.10 | 15.54 | 43.7% |
| L5 | 4.12 | 15.24 | 39.6% |

L1 D2D throughput: 33 GB/s. L5 D2D throughput: 24 GB/s. End-to-end
host-bounce throughput is dominated by the D2H copy of the
decompressed output; D2D callers see only the wall numbers.

---

## [2.0.0] — 2026-04-29

CPU-only Zig port of the C# StreamLZ codec. Superseded by 3.0.

### Highlights (historical)

- All 11 levels (Fast L1-L5, High L6-L11) implemented in Zig.
- Parallel decompress at every level.
- Seven built-in dictionaries + FASTCOVER trainer.
- 297 unit tests + 140 fixture roundtrips against the C# reference.
- ReleaseSafe + fuzz-harness builds for production hardening.

See git history at tag `v2.0.0` for the 2.x source tree.
