# StreamLZ — Vulkan backend

GPU-accelerated LZ77 compressor + decompressor. Vulkan 1.3 compute
shaders do the per-chunk LZ work and 32-stream Huffman decode; thin
Zig drivers manage the kernel launches and host-side wire-format
assembly.

Same SLZ1 wire format as the CUDA backend (`include/streamlz_gpu.h`);
binary-compatible streams in both directions. Runs on any desktop GPU
exposing Vulkan 1.3 with compute subgroups pinnable to 32 lanes
(NVIDIA discrete, Intel UHD/Iris/Arc iGPUs, AMD wave32-path GPUs).

---

## Quick start

```
zig build streamlz_vk -Doptimize=ReleaseFast
```

Produces `zig-out/bin/streamlz_vk.exe` (CLI). The SPIR-V kernel images
are compiled at build time by `glslc` (from `C:/VulkanSDK/...`) and
`@embedFile`'d into the binary.

```
streamlz_vk file.txt              # compress (default L1)
streamlz_vk -l 5 file.txt         # compress at level 5
streamlz_vk -d file.slz           # decompress
streamlz_vk -b -l 3 file.txt      # compress + decompress + verify
streamlz_vk -db file.slz          # decompress-only benchmark
streamlz_vk -i  file.slz          # frame / block header dump
```

Levels L1-L5: higher = better ratio, slower encode. Decode speed is
roughly the same across all five (within ±0.5 ms on a 100 MB input).

Device selection on multi-GPU systems:

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"   # NVIDIA discrete (recommended)
$env:SLZ_VK_DEVICE_INDEX = "0"   # Intel iGPU (works, slower)
```

Without the env var the CLI picks `physicalDevices[0]`, which on most
Windows boxes is the Intel iGPU. Verify the `Device: <name>` line the
CLI prints at startup before trusting any perf number.

---

## C ABI library

Two surfaces are exported:

**CUDA-shaped** (`include/streamlz_gpu.h`) — drop-in CUDA replacement.
Same `slzCreate` / `slzCompressHost` / `slzDecompressHost` /
`slzCompressAsync` / `slzDecompressAsync` / `slzGetLastTimings` names
as the CUDA build. Existing CUDA call sites don't need to change.

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

**VK-native** (`include/streamlz_gpu_vk.h`) — `_vk`-suffixed surface
with opaque `slzVkHandle_t`, true async + polling (`slzCompressAsync_vk`
+ `slzCompressAsyncPoll_vk` pair), VkBuffer registration via
`slzRegisterBuffer_vk`, synthetic-pointer helpers for D2D testing via
`slzMakeDeviceOnlyHandle_vk`. 16 symbols total. Use this when you
want VK-native semantics rather than CUDA-shaped emulation.

Both surfaces share the same internal codec — picking one is purely a
caller-style decision.

---

## Performance

Best-of-5 decode + single-shot encode on an RTX 4060 Ti (sm_89,
Vulkan API 1.4.325), `streamlz_vk -db -r 5` and `streamlz_vk -b -r 5`.
Full sweep tables in [PerfSweep.md](PerfSweep.md); 60 raw bench
captures under `c:/tmp/perfsweep/`.

### Decode (ms): end-to-end host wall-clock

| Level | enwik8 (95 MB) | silesia (203 MB) |
|-------|---------------:|-----------------:|
| L1 | **15.0** | **29.0** |
| L2 | **14.9** | **29.1** |
| L3 | **17.1** | **32.7** |
| L4 | **17.1** | **32.7** |
| L5 | **17.1** | **32.7** |

End-to-end = H2D upload of compressed frame + GPU decode + D2H download
of decompressed output, as a host-bounce caller sees it.

### Compression ratio (output size as % of source)

| Level | enwik8 | silesia |
|-------|-------:|--------:|
| L1 | 58.6% | 47.8% |
| L2 | 58.6% | 47.8% |
| L3 | 43.7% | 38.0% (CUDA 38.0%; 1.00019× wider, A-008) |
| L4 | 42.7% | 37.5% |
| L5 | 39.6% | 33.9% |

Byte-identical to the CUDA backend on every (level, corpus) cell
**except** silesia L3, where the Vulkan 4 GiB SSBO range cap forces a
`hash_bits=18` clamp (CUDA uses 19); cost is 0.019% larger output.
Documented as A-008.

### StreamLZ-Vulkan vs StreamLZ-CUDA (RTX 4060 Ti, enwik8 + silesia)

| Window | VK vs CUDA | Notes |
|--------|-----------:|-------|
| Decode e2e (all levels, large workloads) | **0.96-1.03×** | Inside 10% parity bar everywhere |
| Decode `gpu kernel` (L1/L2 large workloads) | **0.96-1.02×** | At or under CUDA |
| Decode `gpu kernel` (L3-L5 large workloads) | 1.20-1.37× | Over bar; absorbed by host overhead at e2e (A-021) |
| Encode (L1/L2 enwik8) | **0.81-0.86×** | VK measurably faster than CUDA |
| Encode (L3-L5 large workloads) | 0.95-1.01× | At parity (within single-shot noise band) |
| Output bytes | 14/15 cells identical | silesia L3 1.00019× (A-008) |

### StreamLZ-Vulkan vs nvCOMP (transitive)

The CUDA backend beats nvCOMP LZ4 by 1.18× (kernel-sum) / 1.03×
(async wall) / 1.18× (end-to-end host wall) on enwik8 L1, and beats
nvCOMP Zstd by 1.14× / 1.05× / 1.19× on enwik8 L5 — see the root
[README.md](../README.md) "vs nvCOMP" section for methodology + raw
numbers.

The Vulkan backend is within 5% of the CUDA backend at end-to-end on
every large-workload cell, so it inherits the nvCOMP advantage with a
small margin. Concrete enwik8 e2e numbers (RTX 4060 Ti):

| Codec | enwik8 L1 e2e | enwik8 L5 e2e |
|-------|--------------:|--------------:|
| StreamLZ-VK | **15.0 ms** | **17.1 ms** |
| StreamLZ-CUDA | 15.5 ms | 15.3 ms |
| nvCOMP LZ4 | 18.3 ms | — |
| nvCOMP Zstd | — | 18.2 ms |

VK is roughly tied with CUDA at L1, slightly behind at L5 (the
A-021 kernel-time residual on the Huffman+LZ decode chain), and still
ahead of nvCOMP at both levels.

### Web.txt small-file regime

| Window | VK | CUDA | Ratio | Notes |
|--------|---:|-----:|------:|-------|
| Decode e2e (4.5 MB) | 5.3 ms | 2.0 ms | 2.6× | Vulkan-on-WDDM submit floor (~50-150 µs per dispatch × N dispatches) |
| Encode (4.5 MB) | 19 ms | 11 ms | 1.7× | Same submit-floor cost |

Sub-10 MB inputs hit a structural Vulkan-on-Windows overhead that CUDA's
kernel-mode driver model amortizes more effectively. Documented as
accepted residual; unfixable from the app side per four negative-result
iterations. The VK backend remains the right choice for the 10+ MB
workloads that dominate real use cases.

---

## Project layout

```
build.zig::addSrcVkShaderSteps   compiles every .comp via glslc -MD
                                  with full #include dep tracking

streamlz_gpu.zig                  CUDA-shaped + _vk-shaped C ABI
                                  exports (29 functions total)
vulkan_api.zig                    procs.* surface definition (26 slots)
mmap.zig                          file mmap helpers (input read +
                                  output write paths)

cli/                              compress / decompress / bench handlers
decode/                           decode pipeline
  module_loader.zig               VkDevice init, KERNEL_DECLS, LRU
                                  import cache, persistent VkPipelineCache
  decode_dispatch.zig             fullGpuLaunchImpl orchestrator + L2
                                  gate + per-phase QPC profiler
  *.comp                          compute shaders (LZ decode, Huffman
                                  decode, scan/compact/merge chain)
  *.glsl                          shared headers (bit-buffer adapters,
                                  byte I/O, wire format)
encode/                           encode pipeline (mirror of decode/)
  module_loader.zig               encode-specific VkDevice init +
                                  Int8 SPIR-V patcher (glslc workaround)
  fast_framed.zig                 compressFramedOne orchestrator +
                                  persistent gpu_out_buf / d2h_final_buf
  *.comp                          encode kernels
common/                           shared GLSL headers (warp, byteio,
                                  Huffman tables)

tests/                            integration + cross-backend + ptest
                                  harness (144 tests, runs on both
                                  NVIDIA + Intel iGPU)

PortAdaptations.md                catalog of every CUDA-VK divergence
                                  (21 entries; 16 RESOLVED, 5 ACTIVE
                                  residuals — all documented with both
                                  static + runtime verification status)
PerfSweep.md                      Phase 5 perf parity tables (L1-L5 ×
                                  3 corpora × VK + CUDA × decode + encode)
Handbook.md                       operations handbook: invariants,
                                  env-var contract, lessons learned,
                                  measurement recipes (replaces the
                                  retired ToDo.md/gameplan.md)
```

---

## Architecture

The kernel pipeline mirrors the CUDA reference exactly:

**Decode pipeline (L3-L5):**
```
host → H2D compressed frame
     → scan_parse_kernel        (1 dispatch, walks frame on GPU)
     → prefix_sum_chunks        (1 dispatch, running offsets)
     → compact_huff_descs ×4    (fused into 1 dispatch via A-017)
     → compact_raw_descs        (1 dispatch)
     → gather_raw_off16         (1 dispatch)
     → merge_huff_descs         (1 dispatch)
     → huff_build_lut           (1 dispatch, 1024-entry shared-mem LUT)
     → huff_decode_4stream      (32 streams parallel, BIL format)
     → lz_decode (raw or huff)  (LZ workhorse, 8-arm dispatch)
     → D2H decompressed output
```

L1/L2 (no Huffman) skip the LUT-build and Huffman-decode kernels and
use the leaner `lz_decode_raw_kernel` directly.

**Encode pipeline:**
```
host → H2D source
     → lz_encode_kernel              (warp-parallel greedy or chain
                                      parser depending on level)
     → assemble_measure_kernel       (per-chunk sub-chunk sizing)
     → assemble_write_kernel         (per-chunk payload write)
     → frame_assemble_kernel         (frame splice + headers)
     → [L3+: huff_build_tables + huff_encode_4stream]
     → D2H compressed frame
```

**Subgroup pinning.** All compute pipelines require
`requiredSubgroupSize=32` + `REQUIRE_FULL_SUBGROUPS_BIT`. Device-pick
guard rejects devices that can't satisfy
`[minSubgroupSize<=32, maxSubgroupSize>=32]`. NVIDIA always offers 32;
Intel supports the pin via UHD/Iris/Arc; AMD supports via the wave32
path. Devices without 32-lane subgroups return
`error.BackendNotAvailable` so the codec falls back to the CUDA build
(if available) or surfaces an error.

**Required Vulkan features:** `bufferDeviceAddress`, `shaderInt8`,
`storageBuffer8BitAccess`, `uniformAndStorageBuffer8BitAccess`,
`subgroupSizeControl`, `computeFullSubgroups` (Vulkan 1.2 + 1.3).

**Persistent disk-backed VkPipelineCache** at
`%LOCALAPPDATA%/streamlz_vk/pipeline_cache.bin`. Saves ~38 ms cold-start
on RTX 4060 Ti (loads at init, saves at process exit). Driver validates
header per Vulkan 1.0 spec § 10.5.4 and silently discards mismatches.

**Memory pinning + zero-copy D2H/H2D.** Encode uses persistent
page-aligned host buffers (via VMA `procMallocHost`) so
`VK_EXT_external_memory_host` imports hit an LRU cache on every call
after the first. The iter-3/4/15 pattern (transfer-queue gather +
imported destination + single submit) eliminated 200+ ms of D2H staging
cost on enwik8.

For per-divergence detail see [PortAdaptations.md](PortAdaptations.md).
For invariants + bench measurement procedures see
[Handbook.md](Handbook.md).

---

## How to measure

```powershell
# Decode (-r 5 + 1 warmup) — apples-to-apples with CUDA
$env:SLZ_VK_DEVICE_INDEX = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz

# Encode roundtrip (note: encode itself is single-shot; -r controls decode reps)
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt

# Per-kernel timing breakdown (decode)
$env:SLZ_VK_PROFILE_DECODE = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz
# Prints kper: <kernel> count=N total=Xms avg=Yus

# Per-phase host overhead
$env:SLZ_VK_PROFILE_PHASES = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz

# ptest_vk on both backends
$env:SLZ_VK_DEVICE_INDEX = "1"; zig build ptest_vk -Doptimize=ReleaseFast
$env:SLZ_VK_DEVICE_INDEX = "0"; zig build ptest_vk -Doptimize=ReleaseFast
```

GPU benchmarks must run **serially** — never parallelize bench
invocations (even across backends); they contend for GPU/WDDM and
produce biased numbers. Always verify the `Device: <name>` line in the
CLI output before trusting any perf number.

---

## License

MIT (inherited from the parent project).
