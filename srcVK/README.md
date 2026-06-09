# StreamLZ — Vulkan backend (`srcVK/`)

A Vulkan compute-shader port of the CUDA StreamLZ codec. Same SLZ1 wire
format, byte-identical to CUDA on every level (L1-L5), runs on any
desktop GPU exposing Vulkan 1.3 + compute subgroups pinned to 32 lanes
(NVIDIA discrete, Intel iGPU UHD/Iris/Arc, AMD wave32-path GPUs).

This is a **port**, not a re-implementation. Every kernel mirrors its
CUDA reference 1:1 in algorithm + dispatch shape. Where Vulkan can't
express a CUDA construct (no pointer polymorphism, no `__byte_perm`, no
shaderInt64 without an extension), the adaptation is catalogued in
[PortAdaptations.md](PortAdaptations.md) with both static and runtime
verification.

The CUDA backend in `/src/` is the canonical reference. This port
should not change behavior — only the underlying GPU API.

---

## Quick start

```
zig build streamlz_vk -Doptimize=ReleaseFast
```

Produces `zig-out/bin/streamlz_vk.exe`. CLI shape mirrors the CUDA
`streamlz`:

```
streamlz_vk file.txt              # compress (default L1)
streamlz_vk -l 5 file.txt         # compress at level 5
streamlz_vk -d file.slz           # decompress
streamlz_vk -b -l 3 file.txt      # compress + decompress + verify
streamlz_vk -db file.slz          # decompress-only benchmark
```

Device selection:

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"   # 1 = NVIDIA discrete (preferred)
$env:SLZ_VK_DEVICE_INDEX = "0"   # 0 = Intel iGPU (works, slower)
```

Without the env var, the CLI picks `physicalDevices[0]` which on most
dev boxes is the Intel iGPU — verify the `Device: <name>` line the CLI
prints at startup before trusting any perf number.

The SPIR-V kernel images are compiled at build time by `glslc`
(installed under `C:/VulkanSDK/`) and embedded into the binary via
`@embedFile`. `build.zig::addSrcVkShaderSteps` wires the glslc
dependency depfile (`-MD`) into Zig's cache hash so editing any
`.glsl` header correctly invalidates the dependent `.spv` (A-012
closure).

To force a clean rebuild after major shader changes:

```
cmd.exe /c "tools\build_vk.bat"
```

Retained as a force-clean utility but no longer required after `.glsl`
edits as of 2026-06-08.

---

## Status (2026-06-08)

| Level | Decode | Encode | Notes |
|-------|--------|--------|-------|
| **L1** | ✅ byte-identical | ✅ byte-identical | Raw LZ. Decode kernel at parity; encode FASTER than CUDA on large workloads. |
| **L2** | ✅ byte-identical | ✅ byte-identical | Raw LZ + bigger hash. Same kernels as L1 plus 9 ptest cases. |
| **L3** | ✅ byte-identical | ✅ 0.019% larger on silesia | Adds 32-stream Huffman. Silesia size delta = A-008 (Vulkan 4 GiB SSBO cap, accepted residual). |
| **L4** | ✅ byte-identical | ✅ byte-identical | L3 kernels + `p_l4=1` match-range rehash. |
| **L5** | ✅ byte-identical | ✅ byte-identical | Adds chain parser (`lz_chain_parser.glsl`); 575 LOC GLSL port of CUDA's 440-line `lz_chain_parser.cuh`. |

**Tests:** ptest_vk reports **140 passed / 9 skipped / 0 failed (149
total)** on both NVIDIA RTX 4060 Ti and Intel iGPU.

**Decode perf vs CUDA** (NVIDIA RTX 4060 Ti, large workloads, `e2e best`):

| Corpus | L1 | L2 | L3 | L4 | L5 |
|--------|---:|---:|---:|---:|---:|
| enwik8 (95 MB) | 0.97× | 0.97× | 1.03× | 1.03× | 1.03× |
| silesia (203 MB) | 0.97× | 0.97× | 1.03× | 1.03× | 1.03× |

**All decode e2e cells inside the 10% bar.** Decode `gpu kernel`
time on L3-L5 is 1.20-1.37× CUDA (A-021, kernel-time-only — fully
absorbed by host overhead at e2e). web.txt (4.5 MB) hits the
Vulkan-on-WDDM submit floor (~2.6× decode / ~1.7× encode); accepted
structural residual.

Full sweep tables: [PerfSweep.md](PerfSweep.md).

---

## C ABI library

Two surfaces are exported from `srcVK/streamlz_gpu.zig`:

**CUDA-shaped** (`include/streamlz_gpu.h`) — drop-in CUDA replacement.
Same `slzCreate` / `slzCompressHost` / `slzDecompressHost` /
`slzCompressAsync` / `slzDecompressAsync` / `slzGetLastTimings` names
as the CUDA build. Useful when integrating into existing CUDA-shaped
call sites without changing call signatures.

**VK-native** (`include/streamlz_gpu_vk.h`) — `_vk`-suffixed surface
with opaque `slzVkHandle_t`, true async + `*Poll_vk` polling, buffer
registration via `slzRegisterBuffer_vk`, synthetic-pointer helpers
via `slzMakeDeviceOnlyHandle_vk`. 16 symbols total, all implemented
as of Phase 4 (`c9c28bc`).

Both surfaces share the same internal codec (`Context` /
`VkContext = Context + AsyncSlot + registry`). Tests in
`srcVK/tests/async_d2d_api.zig`.

---

## Project layout

```
build.zig::addSrcVkShaderSteps   compiles every .comp via glslc -MD,
                                  Zig parses the depfile for accurate
                                  rebuilds (A-012 closure)
PortAdaptations.md                canonical catalog of every CUDA-VK
                                  divergence (A-001..A-021), each with
                                  static + runtime verification status
PerfSweep.md                      Phase 5 perf parity tables (L1-L5 × 3
                                  corpora × VK + CUDA × decode + encode)
ToDo.md                           current status + remaining work + the
                                  CRITICAL HYGIENE RULES section for
                                  new agents/contributors

cli/                              CLI handlers (compress, decompress,
                                  bench)
decode/                           decode pipeline:
  module_loader.zig                VkDevice init, procs.* surface (26
                                  slots), KERNEL_DECLS, LRU import
                                  cache, persistent VkPipelineCache
  decode_dispatch.zig             fullGpuLaunchImpl orchestrator + L2
                                  gate + per-phase QPC profiler
  *.comp                          compute shaders (LZ decode, Huffman
                                  decode, scan/compact/merge dispatch
                                  chain)
  *.glsl                          shared headers (bit-buffer adapters,
                                  byte I/O, wire format)
encode/                           encode pipeline (mirror of decode/)
  module_loader.zig                encode-specific VkDevice init +
                                  Int8 SPIR-V patcher (glslc bug
                                  workaround, A-cataloged)
  fast_framed.zig                 compressFramedOne orchestrator +
                                  persistent gpu_out_buf / d2h_final_buf
                                  + encode phase profiler
  *.comp                          encode kernels (LZ encode, Huffman
                                  encode, frame assembly)

common/                           shared GLSL/SPIR-V headers (warp,
                                  byteio, huffman tables)

tests/                            integration + cross-backend + ptest
                                  harness. Per-test VK device pick via
                                  SLZ_VK_DEVICE_INDEX. test_runner_parallel.zig
                                  drives serial_first → parallel → serial.

streamlz_gpu.zig                  CUDA-shaped + _vk-shaped C ABI
                                  exports (29 functions total)
vulkan_api.zig                    procs.* surface definition
mmap.zig                          file mmap helpers (input read +
                                  output write paths)
```

---

## Architecture invariants — DO NOT VIOLATE

Reproduced from `ToDo.md` for visibility:

1. **`procs.*` surface** — codec calls `procs.h2d(dst, src, size)` etc.
   on both CUDA and VK. Adding new slots is OK when they express a
   Vulkan-native concept CUDA doesn't have (A-007 per-binding offset,
   `d2h_offset_gather`, `compute_to_compute_barrier`, etc.).
2. **Function decomposition** must mirror CUDA verbatim — do not
   rename or restructure `fullGpuLaunchImpl`, `compressFramedOne`,
   etc.
3. **L1 host-input path keeps CPU `buildChunkDescriptors`** — the
   prior `/src_vulkan/` workflow moved this to GPU; that was the
   canonical port violation (see `feedback_port_means_port`).
4. **Subgroup size PINNED to 32** via `requiredSubgroupSize=32` +
   `REQUIRE_FULL_SUBGROUPS_BIT`. Device-pick guard rejects devices
   that can't satisfy this. NVIDIA + Intel UHD/Iris/Arc + AMD wave32
   path all qualify.
5. **VkPhysicalDeviceVulkan12Features + Vulkan13Features required**:
   `bufferDeviceAddress`, `shaderInt8`, `storageBuffer8BitAccess`,
   `uniformAndStorageBuffer8BitAccess`, `subgroupSizeControl`,
   `computeFullSubgroups`. Init returns `error.BackendNotAvailable`
   if any are missing.

The catalog of every deliberate CUDA-VK divergence is in
[PortAdaptations.md](PortAdaptations.md). Every adaptation declares
both static AND runtime verification status — an unverified
adaptation is a known risk surface, per the A-001 byte-65544 lesson.

---

## How to measure

```powershell
# Decode (-r 5 + 1 warmup) — apples-to-apples with CUDA
$env:SLZ_VK_DEVICE_INDEX = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz

# Encode roundtrip (note: encode itself is single-shot; -r controls decode reps only)
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt

# Per-kernel timing breakdown (decode)
$env:SLZ_VK_PROFILE_DECODE = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz
# Prints kper: <kernel> count=N total=Xms avg=Yus per kernel

# Per-phase host overhead
$env:SLZ_VK_PROFILE_PHASES = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz

# Cross-backend SHA byte-identity (run for every encode change)
foreach ($a in @("web.txt", "enwik8.txt", "silesia_all.tar")) {
  ./zig-out/bin/streamlz_vk.exe -c -l 5 -o "c:/tmp/vk_$a.slz" "assets/$a"
  ./zig-out/bin/streamlz.exe    -c -l 5 -o "c:/tmp/cu_$a.slz" "assets/$a"
  $vk = (Get-FileHash -Algorithm SHA256 "c:/tmp/vk_$a.slz").Hash
  $cu = (Get-FileHash -Algorithm SHA256 "c:/tmp/cu_$a.slz").Hash
  Write-Host "$a MATCH=$($vk -eq $cu)"
}

# ptest_vk on both backends
$env:SLZ_VK_DEVICE_INDEX = "1"; zig build ptest_vk -Doptimize=ReleaseFast
$env:SLZ_VK_DEVICE_INDEX = "0"; zig build ptest_vk -Doptimize=ReleaseFast
```

**GPU benchmarks must run SERIALLY.** Never parallelize bench tool
calls (even across backends) — they contend for GPU/WDDM and produce
biased numbers. See `feedback_gpu_bench_serial.md` (auto-memory) for
the reproducible-flip example.

**Always verify `Device: <name>` in the CLI output before trusting
a perf number.** The default device pick is `physicalDevices[0]`
which on Windows is typically the Intel iGPU. Set
`SLZ_VK_DEVICE_INDEX=1` for the NVIDIA discrete on this dev box.

---

## Known accepted residuals

1. **web.txt small-file regime** — decode 2.6×, encode 1.7× CUDA.
   Structural Vulkan-on-WDDM submit floor (~50-150 µs per dispatch).
   Confirmed unfixable from the app side via four negative-result
   iterations.
2. **L3 silesia 0.019% larger** — A-008. Vulkan 4 GiB SSBO range
   cap forces `hash_bits=18` clamp on inputs > 128 MiB. Future
   BDA workaround would close it.
3. **L3/L4/L5 decode `gpu kernel` 1.20-1.37× CUDA** — A-021.
   Composition of explicit pipeline barriers (A-006), per-binding
   offset ABI (A-007), and unfused merge/compact/gather dispatches.
   Absorbed by host overhead at e2e (all cells inside 10% bar).
   Future fusion work (A-017 pattern) would close it.
4. **L2 pre-existing single-thread mode `TestExpectedEqual`** — 2
   tests skip under `SLZ_VK_TEST_THREADS=1`. Parallel mode (the
   spec'd config) is rock-solid at 140/9/0.

---

## License

MIT (inherited from the parent project).
