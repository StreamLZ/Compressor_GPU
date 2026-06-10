# Changelog

## Unreleased — 2026-06-09 (1 GB-scale correctness + decode-speed wave)

Frames produced by older builds remain decodable (sc_group_size and
all level parameters are stamped in the frame header); frames produced
by this build differ byte-wise from 3.0.0 output at the same level.

### Fixed

- **A-024: L3-L5 silent decode corruption above ~820 MB (both
  backends).** `slzMergeHuffDescsKernel` folded the per-region entropy
  scratch offset into the u32 `HuffDecChunkDesc.out_offset`; at ≥ 6,554
  sub-chunks the off16 region offset exceeded 2^32 and wrapped, making
  the Huffman predecoder overwrite chunk 0's literal slot (round-trip
  FAIL at byte 8 on enwik9). Region offsets are now applied at full
  width inside `slzHuffDecode4StreamKernel` (u64 kernel params on
  CUDA; u32 push constants + `d_compact_counts` region pick on
  Vulkan). enwik9 L3/L4/L5 1 GB now round-trip SHA-256 clean on CUDA.
- **L1/L2 decode launched the full entropy pipeline as a no-op.** The
  scan / compact / gather / merge / huff kernels (and the multi-GB
  BOUND-sized entropy scratch allocation) now skip entirely for
  level < 3 frames, and L1/L2 correctly select the lean
  `slzLzDecodeRawKernel` (the worst-case scan counts previously forced
  the general kernel). L1 1 GB decode kernel: 48.9 → 44.6 ms before
  the sc change below.
- **PP long-token window poisoning.** One long token serialized up to
  32 short tokens ahead of it in the parallel-parse fast path; the
  batch now truncates to the all-short prefix (`__ffs` on the ballot)
  and processes it in parallel. ~2% on enwik8 L1.

### Changed

- **`sc_group_size` default is now 0.25 at every input size** (was 0.5
  above a ~208 MB saturation threshold). 64 KB sub-chunks double the
  decode warp count and halve every warp's serial chain at 1 GB scale:
  L1 enwik9 decode kernel 44.2 → 24.3 ms (39.3 GB/s), L3-L5 ~2×.
  Ratio cost ~2.3 pp at L1 / ~1.4 pp at L3-L5 on enwik9. `--sc 0.5`
  remains available for maximum ratio.
- **`hash_bits = 17` at every level** (L2 was 18, L3 was 19). At
  ≤ 128 KB sub-chunks the larger tables added ≤ 0.1 pp ratio while
  blowing VRAM at sc=0.25 chunk counts (L2's hb=18 hash = 16 GB at
  1 GB input → encode collapsed to 159 MB/s; now 992 MB/s). L1 and L2
  now produce byte-identical output.
- **A-023 batched LZ dispatch backported from the Vulkan port.** The
  encoder now caps the per-chunk hash allocation to a `cuMemGetInfo`
  budget (3/4 of free VRAM) and dispatches the LZ kernel in batches
  over the smaller buffer — byte-identical output. L5 enwik9 encode:
  52 → 387 MB/s (the WDDM allocator silently paged the former ~18 GB
  chain-hash request over PCIe). `SLZ_FORCE_BATCH` env var +
  `g_force_batch_count_for_test` mirror the VK test hooks.

### Result vs nvCOMP (enwik9 1 GB, RTX 4060 Ti)

StreamLZ L1 beats nvCOMP LZ4 on ratio (52.6% vs 53.6%) and decode
kernel speed (24.3 vs 33.0 ms); StreamLZ L5 beats nvCOMP Zstd on ratio
(35.50% vs 35.75%) and decode kernel speed (34.2 vs 50.8 ms). See
README "vs nvCOMP (enwik9)".

## [3.0.0] — 2026-05-29 / amended 2026-05-31

GPU-only release. The CPU codec (all 11 levels, parallel decoder,
streaming compressor, dictionary subsystem, parallel-decode sidecar)
and the Vulkan compute-shader decode fallback have been removed; the
CUDA codec is the only backend.

### Vulkan-port ABI prerequisites (M0, 2026-05-31 amendment)

The 3.0.0 line is amended in-place (no version-string bump — the
library still reports `3.0.0`) to land the shared CUDA/Vulkan ABI
prerequisites the Vulkan port consumes from M2 onward. All changes
are in `include/streamlz_gpu.h`; the on-disk frame format and the
compressed bytes are unchanged.

**Breaking ABI changes (callers must migrate):**

- **`slzCompressOpts_t.reserved[0]` repurposed as
  `effective_level_out`.** The first padding slot is now an OUT field
  the backend writes with the post-clamp level it actually used. The
  struct size is unchanged (32 B, 8 × `int`); the remaining padding
  is `reserved[5]`. The CUDA backend writes `opts.level` verbatim
  (it never clamps). The Vulkan Tier-2 backend writes the clamped
  level (e.g. L3) when mobile-VRAM pressure forces L5→L3. Callers
  that previously wrote `reserved[0] = 0` and ignored it are
  byte-compatible; callers that read `reserved[0]` after the call
  will now see a non-zero value (the clamped level).
- **`slzCompressAsync` no longer writes `compressed_size` before
  returning.** The pre-3.0 contract documented the size as "valid as
  soon as slzCompressAsync returns"; that promise is physically
  impossible on the Vulkan path (the size comes from the GPU
  `slzAssembleMeasure` kernel after the async submission completes)
  and the CUDA path is aligned to match so the two backends share
  one contract. The `compressed_size` parameter is retained for ABI
  compatibility but is NOT written. Callers retrieve the real size
  by setting `opts.enable_profiling = 1`, syncing the user stream,
  and looking up the `SLZ_PSEUDO_KERNEL_COMPRESSED_SIZE` entry
  returned by `slzGetLastTimings` /
  `slzWaitAndGetLastTimings` — `ms` reinterpreted as `uint32_t`
  carries the byte count. `slzCompressHost` is unaffected; it does
  the sync internally and continues to write `output_size`.
- **`slzKernelTiming_t` gains two pseudo-kernel name slots.**
  `SLZ_PSEUDO_KERNEL_COMPRESSED_SIZE` ("`__compressed_size__`") and
  `SLZ_PSEUDO_KERNEL_DECOMPRESSED_SIZE` ("`__decompressed_size__`")
  appear in the timings array when `enable_profiling = 1`. Their
  `ms` field is a `uint32_t` bit-reinterpret carrying the byte
  count, not a wall-clock measurement. The struct layout is
  unchanged (still 2 × `sizeof(void*)`); existing callers that
  iterate the array by name and skip unknown entries see no
  regression.
- **`slzStatus_t` enum extended with two new codes** so the shared
  ABI carries the values the Vulkan backend needs:
  - `SLZ_ERROR_DEVICE_LOST = 8` — `VK_ERROR_DEVICE_LOST`; terminal,
    library must rebuild `VkDevice`. The CUDA backend never returns
    this code.
  - `SLZ_ERROR_VK_FEATURE_MISSING = 9` — required Vulkan feature
    or extension (`VK_KHR_subgroup_size_control`, BDA, `shaderInt64`,
    ...) not present at create time. The CUDA backend never returns
    this code; the slot reservation prevents future Vulkan additions
    from re-numbering.

**Migration checklist for 2.x → 3.0.0 callers:**

1. `slzCompressAsync` callers that read `*compressed_size` after the
   call must switch to the timings drain. Set
   `opts.enable_profiling = 1`, sync the user stream (or use
   `slzWaitAndGetLastTimings`), and look up
   `SLZ_PSEUDO_KERNEL_COMPRESSED_SIZE` in the returned array.
2. Callers that set `opts.reserved[0]` to anything nonzero will see
   that value written by the backend on return (CUDA echoes
   `opts.level` into `effective_level_out`, overwriting it). Treat
   the field as OUT-only.
3. Callers that switch on `slzStatus_t` exhaustively must add arms
   for `SLZ_ERROR_DEVICE_LOST` and `SLZ_ERROR_VK_FEATURE_MISSING`
   (the CUDA backend never returns them today, but the compiler
   should warn anyway).

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
