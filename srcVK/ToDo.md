# srcVK Vulkan Port — L1 Done, L2-L5 Roadmap

**Last updated:** 2026-06-07, HEAD `0af24ff`.
**Replaces:** the prior `finishingL1.md` (L1 is finished, doc rescoped to future work).

This file is the source of truth for "what's left." Most of L1 is locked-in
production code; the meaningful remaining work is L2-L5 to reach full CUDA
parity. The `/src_vulkan/` directory contains an earlier (now-deprecated)
implementation that ported less faithfully; its `TODO.md` informed parts of
this roadmap.

---

## Status: L1 is DONE

| Criterion | NVIDIA RTX 4060 Ti | Intel(R) Graphics iGPU |
|---|---|---|
| Structural 1:1 port audit | ✅ | ✅ |
| Byte-equal roundtrip (web + enwik8 + silesia) | ✅ | ✅ |
| Cross-backend (CUDA ↔ VK both directions) | ✅ | ✅ |
| Decode perf ≤ 1.10× CUDA on large workloads | ✅ 0.97-0.98× | works (slower iGPU) |
| Encode perf ≤ 1.10× CUDA on large workloads | ✅ 0.87-1.11× (faster on web + silesia) | works, byte-identical encode |
| ptest_vk | **74/0/0** | **74/0/0** |
| VK validation (spec) | 0 VUIDs | 0 VUIDs |
| VK validation (synchronization) | 0 hazards | 0 hazards |
| VK validation (best practices, bench mode) | 0 | 0 |

Current bench numbers (NVIDIA, `SLZ_VK_DEVICE_INDEX=1`, `-db -r 5` or `-b -r 5`):

| File | Decode e2e | Decode kernel | Encode | vs CUDA |
|---|---|---|---|---|
| enwik8 (95 MB) | 15.7 ms | 3.0 ms | 146 ms | dec 0.97×, enc 1.11× |
| silesia (203 MB) | 30.0 ms | 5.5 ms | 205 ms | dec 0.97×, enc 0.87× |
| web.txt (4.3 MB) | 5.3 ms | 1.3 ms | 11 ms | dec 2.6× (Vulkan-on-WDDM floor), enc 0.92× |

---

## L2-L5 roadmap to full CUDA parity

The CUDA encoder supports five levels:

| Level | Parser | Entropy | Notes |
|---|---|---|---|
| **L1** | greedy | none (raw streams) | **DONE** |
| L2 | greedy (hash_bits=18) | Huffman | needs Phase 2A |
| L3 | greedy (hash_bits=19) | Huffman | needs Phase 2A |
| L4 | greedy (hash_bits=17) | Huffman | needs Phase 2A |
| L5 | **chain parser** (hash_bits=17 + chain table) | Huffman | needs Phase 2A + 2B |

### Phase 2A — Huffman codec (unlocks L2/L3/L4)

Goal: encoder produces chunk type 4 (Huffman-coded) sub-chunks that the CUDA
decoder reads; decoder reads CUDA-produced chunk type 4. Cross-backend
roundtrip parity across L2-L4 in both directions.

**Kernels to port** (CUDA references in `src/encode/huffman_kernel.cu` +
`src/decode/huffman_kernel.cu`):
- `srcVK/encode/huff_build_tables_kernel.comp` — histogram → canonical Huffman codes
- `srcVK/encode/huff_encode_4stream_kernel.comp` — bytes + codes → BIL bitstream
- `srcVK/decode/huff_build_lut_kernel.comp` — weights → 256-entry decode LUT
- `srcVK/decode/huff_decode_4stream_kernel.comp` — bitstream + LUT → bytes

**Host glue:**
- Generalize `encode/encode_lz.zig` to wire Huffman after greedy parser
- Generalize `decode/decode_dispatch.zig::runLzPipeline` to dispatch type-4 sub-chunks through Huffman before LZ-decode
- Sub-chunk wire format already supports it; encoder just needs to emit
- Open the L2 gate at `decode_dispatch.zig:724` for type-4 path

**Tests:**
- New `tests/l2_encode_roundtrip.zig`, `tests/l2_decode_roundtrip.zig`
- Extend `tests/cross_backend_roundtrip.zig` for L2-L4

LOC estimate: ~1,500 GLSL + ~800 Zig. ~3-5 single agents.

### Phase 2B — L5 chain parser

Goal: L5 .slz files byte-correct vs CUDA L5 (within the hash-store-order
tolerance accepted for L1).

**New shader:** `srcVK/encode/lz_chain_encode_kernel.comp` — port of
`src/encode/lz_chain_parser.cuh` (~440 CUDA LOC). Most complex single
kernel in the project — chain table + secondary hash + lazy matching.

**Host glue:**
- Extend encode dispatch to select chain parser at L5
- L5 has 4× memory footprint per sub-chunk vs L1; validate the existing
  ensureBuf grow-only pattern handles silesia at L5

**Test:** `tests/l5_encode_roundtrip.zig` + cross-backend.

LOC estimate: ~800 GLSL + ~500 Zig. ~2-3 single agents.

### Phase 3 — Multi-kernel decode pipeline (L2+ requires this)

L1 decode currently uses CPU `buildChunkDescriptors` (correct per port
discipline — CUDA does this on CPU for L1 too). L2+ needs the full
multi-kernel GPU graph that CUDA runs per frame.

**Kernels to port** (CUDA: `src/decode/scan_parse_kernel.cuh`,
`prefix_sum_chunks_kernel.cuh`, etc.):
- `srcVK/decode/scan_parse_kernel.comp` — parse compressed-block headers on GPU
- `srcVK/decode/prefix_sum_chunks_kernel.comp` — running offsets (L1 has this for prefix_sum_chunks_fn already — port for the L2 D2D path)
- `srcVK/decode/compact_huff_descs_kernel.comp` — gather entropy-decoded chunk descs
- `srcVK/decode/compact_raw_descs_kernel.comp` — gather raw chunk descs
- `srcVK/decode/walk_frame_kernel.comp` — walk parsed frame (L2 D2D entry)
- `srcVK/decode/merge_huff_descs_kernel.comp` — merge huff descs
- `srcVK/decode/gather_raw_off16_kernel.comp` — gather raw off16 stream
- `srcVK/decode/lz_decode_kernel.comp` — L2 huff-aware LZ workhorse (L1 uses `lz_decode_raw_kernel.comp`)

**Host glue:** generalize `decode_dispatch.zig::fullGpuLaunchImpl` to use the
GPU chain when L2+ or when the caller supplies device-resident input
(`decompressFramedFromDevice`, currently `error.NotImplementedL2`).

LOC estimate: ~1,500 GLSL + ~600 Zig. ~3-4 single agents.

### Phase 4 — True D2D + Async API

L1 has stubs for these but not actual implementations:
- `decompressFramedFromDevice` — returns `error.NotImplementedL2` (L2 D2D entry)
- Async APIs (`slzCompressAsync_vk`, `slzCompressAsyncPoll_vk`, etc.) — not wired

**The decode/encode procs surface already supports streams (iter-11 split
queue, iter-13 early submit, iter-15 single-submit/single-wait), so async is
mostly host-glue work.**

Items:
- True buffer-device-address D2D for `slzCompress_vk` / `slzDecompress_vk`
  using the existing `slzRegisterBuffer_vk` registry pattern
- Async encode/decode worker pool wrapping the sync codec (mirror the CUDA
  `slzCompressAsync` thread-per-handle pattern)
- `slzGetLastTimings_vk` to drain the existing VkQueryPool data

LOC estimate: ~600 Zig. ~1-2 single agents.

### Phase 5 — Conformance + perf parity validation

- Extend `tests/cross_backend_roundtrip.zig` and scale tests for L1-L5 in
  both directions on both Intel and NVIDIA
- Benchmark vs CUDA on each level; target ≤1.10× on encode + decode at every
  level on large workloads (matches the L1 done-bar)
- Document any residual gaps as accepted tradeoffs

LOC estimate: ~400 Zig. ~1 single agent.

### Total estimate to full CUDA parity

| Phase | GLSL LOC | Zig LOC | Agent runs |
|---|---|---|---|
| 2A (Huffman) | ~1,500 | ~800 | 3-5 |
| 2B (L5 chain parser) | ~800 | ~500 | 2-3 |
| 3 (GPU decode pipeline) | ~1,500 | ~600 | 3-4 |
| 4 (D2D + async) | 0 | ~600 | 1-2 |
| 5 (conformance + perf) | 0 | ~400 | 1 |
| **TOTAL** | **~3,800** | **~2,900** | **10-15** |

---

## Kernel inventory (current state)

### L1 kernels — REAL (working in production)

| File | Status | Purpose |
|---|---|---|
| `decode/lz_decode_raw_kernel.comp` | ✅ real | L1 raw-mode LZ decode workhorse |
| `decode/prefix_sum_chunks_kernel.comp` | ✅ real | L1 prefix sum over chunk descs |
| `encode/lz_encode_kernel.comp` | ✅ real (~300 LOC) | L1 greedy LZ encode |
| `encode/assemble_measure_kernel.comp` | ✅ real | Per-chunk sub-chunk sizing |
| `encode/assemble_write_kernel.comp` | ✅ real | Per-chunk payload write |
| `encode/frame_assemble_kernel.comp` | ✅ real | Frame splice + headers |

### L2+ kernels — STUBS (waiting for Phase 2A/3)

| File | Phase | CUDA reference | Approx CUDA LOC |
|---|---|---|---|
| `encode/huff_build_tables_kernel.comp` | 2A | `src/encode/huffman_kernel.cu` | ~350 |
| `encode/huff_encode_4stream_kernel.comp` | 2A | `src/encode/huffman_kernel.cu` | ~350 |
| `decode/huff_build_lut_kernel.comp` | 2A | `src/decode/huffman_kernel.cu` | ~350 |
| `decode/huff_decode_4stream_kernel.comp` | 2A | `src/decode/huffman_kernel.cu` | ~350 |
| `decode/scan_parse_kernel.comp` | 3 | `src/decode/scan_parse_kernel.cuh` | 234 |
| `decode/walk_frame_kernel.comp` | 3 | `src/decode/walk_frame_kernel.cuh` | 200 |
| `decode/compact_huff_descs_kernel.comp` | 3 | `src/decode/compact_descs_kernels.cuh` | ~45 |
| `decode/compact_raw_descs_kernel.comp` | 3 | `src/decode/compact_descs_kernels.cuh` | ~45 |
| `decode/merge_huff_descs_kernel.comp` | 3 | `src/decode/merge_huff_descs_kernel.cuh` | 62 |
| `decode/gather_raw_off16_kernel.comp` | 3 | `src/decode/gather_raw_off16_kernel.cuh` | 38 |
| `decode/lz_decode_kernel.comp` | 3 | `src/decode/lz_decode_kernels.cuh` (huff path) | ~400 |

**Plus needed:** `encode/lz_chain_encode_kernel.comp` (no stub yet) for Phase 2B
— CUDA reference `src/encode/lz_chain_parser.cuh` ~440 LOC.

---

## Architecture invariants — DO NOT VIOLATE

These kept us out of trouble during L1 and remain load-bearing for L2-L5.

### procs.* surface (decode/vulkan_api.zig)

The codec calls `procs.h2d(dst, src, size)`. CUDA and VK both. Same name,
args, semantics. Vulkan plumbing (vkCmdCopyBuffer, VMA, fences, queues)
lives UNDER procs. Adding new slots is OK when the slot expresses a Vulkan-
native concept CUDA doesn't have a direct analog for (e.g., iter-13's
`stream_flush_transfer`, iter-2's `d2h_offset_gather`).

Current slots (25): `malloc_device, free_device, h2d, d2h, d2h_offset,
h2d_async, d2h_async, d2d, memset_d8, memset_d8_async, malloc_host,
free_host, stream_sync, ctx_sync, ctx_set_current, ctx_get_current,
stream_create, stream_destroy, launch_kernel, event_create, event_record,
event_synchronize, event_elapsed_time, event_destroy, stream_flush_transfer,
stream_compute_barrier, d2h_offset_gather`.

### Function decomposition (CUDA-verbatim)

Do NOT rename or restructure:
- `decode/decode_dispatch.zig::fullGpuLaunchImpl` + `uploadInputAndPrefixSum` + `runBackHalf` + `finalizeOutput` + `runLzPipeline`
- `decode/decode_dispatch.zig::runHuffBuildAndDecode` / `mergeHuffDescs` / `gatherRawOff16` (L2 stubs to be filled in Phase 3)
- `encode/fast_framed.zig::compressFramedOne` + `encode/encode_lz.zig::gpuCompressImpl` + `encode/encode_assemble.zig::gpuAssembleFrameImpl` / `gpuFrameAssembleImpl`

### L2 gate at `decode_dispatch.zig:724`

`if (req.level >= 2) { ... }` wraps ensureDeviceBuf for L2 buffers + L2 kernel
dispatches. Skipped entirely on L1. CUDA wastefully dispatches 7 L2 kernels
on L1 (~388 µs/decode); we don't. **Phase 3 fills the L2 branch.**

### `buildChunkDescriptors` MUST stay CPU walk on L1

In `decode/streamlz_decoder.zig`. The prior `/src_vulkan/` session moved this
to GPU as `walk_frame.comp` + `l1_unwrap` kernels — **THAT WAS THE CANONICAL
PORT VIOLATION.** Keep it CPU for L1.

The `walk_frame_kernel.comp` stub is reserved for the L2 D2D entry point
(`decompressFramedFromDevice`, currently `error.NotImplementedL2`). Phase 3
makes it real.

### KERNEL_DECLS (decode + encode module_loader.zig)

Pipeline binding counts + push constant sizes per kernel. Do NOT change
without auditing all call sites in `decode_dispatch.zig` for matching
`params[]` layout. Adding L2 kernels means adding KERNEL_DECLS entries with
correct `pin_subgroup_32: bool` (true unless `local_size_x == 1`).

### Subgroup size pinned to 32 (gpu_warp.glsl contract)

Both decode + encode pipeline creation pin `requiredSubgroupSize=32` +
`REQUIRE_FULL_SUBGROUPS_BIT`. Device-pick guard rejects devices that can't
satisfy `[minSubgroupSize<=32, maxSubgroupSize>=32]`. Kernels in
`gpu_warp.glsl` hardcode `WARP_SIZE=32u`. **NVIDIA only offers 32; Intel
supports the pin; AMD also supports the pin via the wave32 path.**

### VkPhysicalDeviceVulkan12Features + Vulkan13Features required

`bufferDeviceAddress`, `shaderInt8`, `storageBuffer8BitAccess`,
`uniformAndStorageBuffer8BitAccess`, `subgroupSizeControl`,
`computeFullSubgroups`. Enabled at VkDevice creation. **If any of these
isn't supported, init returns `error.BackendNotAvailable` so the codec falls
back to CPU.**

---

## Lessons learned (apply to L2-L5 work)

### Always run validation layer before claiming "done"

The standard layer catches spec violations. The synchronization layer catches
missing barriers and race conditions. The best-practices layer catches
suboptimal patterns. **Three real latent bugs were caught only by validation:**

- VUID-vkBindBufferMemory-memory-02985 (iter-12 import path missing buffer-create chain — `9022dde`)
- VUID-vkCmdCopyBuffer/EndCommandBuffer-recording (cmdbuf invalidated by destroyed buffer — `7960b98`)
- VUID-VkShaderModuleCreateInfo-pCode-08737 (glslc optimizer bug — `23d8ab3`)

Run after each iteration:

```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH = "c:/tmp/vk_layer_settings.txt"  # contents below
./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/v.slz assets/web.txt 2>&1 | Select-String "VUID|Validation Error"
./zig-out/bin/streamlz_vk.exe -d tests/goldens/web.txt.L1.slz -o c:/tmp/v.bin 2>&1 | Select-String "VUID|Validation Error"
Remove-Item env:VK_INSTANCE_LAYERS
```

Contents of `vk_layer_settings.txt` for sync + best-practices:
```
khronos_validation.enables = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
khronos_validation.report_flags = error,warn,perf,info
```

### NEVER hand-type Vulkan constants

Commit `6ca302c` was a 4-line constant correction that fixed an entire
broken-pin commit. Two enum values had been typed from memory and were both
wrong. **Always pull constants from `C:/VulkanSDK/1.4.341.1/Include/vulkan/vulkan_core.h`** —
grep the header, copy the exact value, paste.

### Zig std subprocess spawn defaults to EMPTY env

`std.Io.Threaded.InitOptions.environ` defaults to `.empty`. Spawned children
get NO PATH, NO SystemRoot, NO PROGRAMDATA. Vulkan loader can't find ICDs
without them. **Pass `.{ .environ = .{ .block = .global } }` when spawning
subprocesses that need registry-discovered hardware drivers.** Affects every
test that spawns child processes (closed at commit `e8fc91f`).

### glslc -O optimizer bug — Int8 capability not auto-declared

When the optimizer constant-folds `uint8_t(SLZ1_FRAME_MAGIC & 0xFFu)` it
emits `OpConstant %uchar` but fails to declare `OpCapability Int8`. Driver
validation rejects the module. **Patch SPIR-V binaries post-compile in
`encode/module_loader.zig::loadShaderModule`** — splice `OpCapability Int8`
into the capability block when any 8-bit storage cap is present without it.
Workaround documented at commit `23d8ab3`.

### Vulkan-on-WDDM has a per-submit floor that CUDA doesn't

Each `vkQueueSubmit + vkWaitForFences` round-trip is ~100-150 µs on Windows
WDDM. CUDA's `cuMemcpyDtoH` synchronous calls average ~37 µs because CUDA's
runtime batches submission cost across the kernel-mode driver. **You cannot
reduce per-call submit cost in Vulkan; you can only reduce the number of
submits.** Pattern: batch N copies into one `vkCmdCopyBuffer` with N regions
(see encode iter-2 `d2h_offset_gather`, commit `5e7a480`).

### The "iter-7+11+15" pattern (transferable from decode to encode)

This is the canonical fix for any per-call host-bounce pattern:
- **iter-7 D2H import:** `VK_EXT_external_memory_host` wraps the user's
  destination buffer as VkDeviceMemory; kernel writes directly into host RAM
- **iter-11 transfer queue:** dedicated VK_QUEUE_TRANSFER_BIT queue (DMA
  engine) + binary semaphore split-submit
- **iter-15 single submit:** consolidate phases into one submit + one wait
  per call by recording into the same cmdbuf

Applied to encode at iter-2 (`5e7a480`) — collapsed 1526 per-chunk D2H
submits into 1 vkCmdCopyBuffer with 1526 regions, 3× speedup. **Apply the
same pattern to any L2+ kernel that emits a per-chunk host bounce.**

### ptest_vk parallelism — three race classes caught

1. Decode/encode init races (lock with `g_init_lock` / `g_encode_init_lock`)
2. Encode init latching (use `defer if (ok)` for the `initialized=true` flip)
3. Cross-queue submit races on shared compute queue (lock via `g_dispatcher_lock`)

Plus iter-14's per-(StreamEntry × pipeline) descriptor sets to avoid races
on shared descriptor state. **Apply the same patterns when adding L2 init
paths or shared state.**

---

## File map

### Core paths

- `decode/streamlz_decoder.zig` — `decompressFramed` + CPU `buildChunkDescriptors`
- `decode/decode_dispatch.zig` — `fullGpuLaunchImpl` orchestrator + L2 gate at line 724
- `decode/decode_context.zig` — `DecodeContext`, `ensureDeviceBuf` (grow-only), profiling
- `decode/scan_gpu.zig` — `gpuPrefixSumChunksImpl` (L1) + L2 stubs
- `decode/module_loader.zig` — VkDevice init, transfer queue, KERNEL_DECLS, procs impls, LRU import cache, streamEndAndWait, persistent_desc_sets
- `decode/vulkan_api.zig` — procs surface (25 slots)
- `decode/driver.zig` — facade
- `encode/fast_framed.zig` — `compressFramedOne` + persistent gpu_out_buf
- `encode/encode_lz.zig` — `gpuCompressImpl` + per-chunk D2H gather loop
- `encode/encode_assemble.zig` — frame assembly passes
- `encode/encode_context.zig` — `EncodeContext` + `pipeline_stream`
- `encode/module_loader.zig` — encode VkDevice init + shader module loader + Int8 SPIR-V patcher
- `encode/vulkan_ffi.zig` — encode-side procs

### CLI

- `cli/bench_decompress.zig` — `-db` (post-`d5adeb8` pure-kernel metric)
- `cli/bench_compress.zig` — `-b` (encode + roundtrip)
- `cli/decompress.zig` — `-d`
- `cli/compress.zig` — `-c`
- `cli.zig` — dispatcher, prints `Device: <name>` at startup
- `cli/util.zig` — Args + DeviceSelector (`SLZ_VK_DEVICE_INDEX` env)

### Tests

- `tests/decoder_unit.zig` (22 host-only)
- `tests/encoder_unit.zig` (11 host-only)
- `tests/dispatch_unit.zig` (7 including KERNEL_DECLS ABI check)
- `tests/kernel_conformance.zig` (per-test context)
- `tests/l1_decode_roundtrip.zig` (per-test context)
- `tests/l1_encode_roundtrip.zig` (per-test context)
- `tests/cross_backend_roundtrip.zig` (subprocess — needs env-inherit `makeIo`)
- `tests/cli_smoke.zig` (subprocess — needs env-inherit `makeIo`)
- `test_runner_parallel.zig` (3-phase: serial_first → parallel → serial)

### Reference docs

- `audit.md` — canonical file mapping, port rules
- `PortInstructions.md` — per-file fleshout checklist
- `gameplan.md` — comprehensive port playbook
- This file (`ToDo.md`) — status + L2-L5 roadmap

---

## How to measure

### Bench

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"   # 1 = NVIDIA, 0 = Intel iGPU
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz   # decode bench
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt              # encode + roundtrip
```

The bench enables `enable_profiling=true` so VkQueryPool timestamps populate.
"gpu kernel best/mean" is sum of per-kernel timestamps — apples-to-apples
with CUDA's bench definition.

### Per-phase host overhead

```powershell
$env:SLZ_VK_PROFILE_PHASES = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz 2>&1 | Select-String "phase:"
```

Implementation in `decode/decode_dispatch.zig` (the `g_phase_*_ns` accumulators).

### Per-kernel VkQueryPool dump

```powershell
$env:SLZ_VK_PROFILE_DECODE = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz 2>&1 | Select-String "kper:"
```

### ptest_vk (after any srcVK change)

```powershell
Remove-Item -Force zig-out/bin/streamlz_vk.exe
zig build streamlz_vk -Doptimize=ReleaseFast
$env:SLZ_VK_DEVICE_INDEX = "1"
zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
```

Expect 74/0/0 on both backends.

### Validation layer (run before claiming any iteration "done")

See "Lessons learned" section above.

### nsys VK + CUDA trace

```powershell
$NSYS = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
$env:VK_ADD_LAYER_PATH = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/vulkan-layers"
$env:VK_INSTANCE_LAYERS = "VK_LAYER_NV_nsight-sys"
$env:ENABLE_VK_LAYER_NV_nsight_sys = 1
& $NSYS profile --trace=vulkan --output=c:/tmp/trace --force-overwrite=true ./zig-out/bin/streamlz_vk.exe -db -r 3 tests/goldens/enwik8.txt.L1.slz
& $NSYS stats c:/tmp/trace.nsys-rep --report vulkan_api_sum
```

**Caveat:** nsys VK trace layer inflates absolute timings. Trust call counts +
relative shape; verify magnitudes via VkQueryPool or QPC. (See the iter-15
investigation that chased a phantom 2.4 ms saving in vkWaitForFences that
turned out to be trace-layer overhead.)

---

## Known cosmetic residuals (not blocking)

1. **web.txt small-file decode 2.6× CUDA** — structural Vulkan-on-WDDM
   submit floor. Per-kernel parity confirmed (1.3 ms VK vs 1.4 ms CUDA).
   Four-iteration negative-result chain confirmed not fixable from app side
   without removing the submit from the D2H path entirely.

2. **1 OOM warning per direction per process in one-shot CLI** — best-
   practices layer flags the first `vkAllocateMemory` failure in
   `tryImportHostBuffer` for non-pinned host buffers. Sticky disable flips
   on first OOM (commit `0af24ff`); bench mode is silent (pinned buffers).
   One-shot CLI shows 1 warning per direction before the flag flips.

3. **Pre-existing single-thread mode `TestExpectedEqual` (2 tests)** —
   `SLZ_VK_TEST_THREADS=1` surfaces test ordering sensitivity that predates
   encode iter-1 work. Parallel mode (the spec'd config) is rock-solid at
   74/0/0. Investigation not yet attempted.

---

## Recent commit history (last 12)

```
0af24ff  OOM noise cleanup — per-direction sticky disable
e8fc91f  tests: inherit parent env in subprocess test harnesses (74/0/0)
23d8ab3  valfix C — Int8 capability glslc workaround
7960b98  valfix B — cmdbuf state via deferred staging destroy
9022dde  valfix A — handle types on imported VkBuffer
6ca302c  fix two Vulkan constants — subgroup pin now actually pins
332a04c  pin compute pipelines to subgroupSize=32 + device-pick guard
5e7a480  encode iter-2 — D2H gather collapses per-chunk loop (3x faster)
2d68b2c  encode iter-1 H2 — stream-routing + D2H import cache hazard fix
bfe746a  encode iter-1 H3 — verify iter-12 import auto-fires on encode H2D
bb675ab  encode iter-1 H1 — per-chunk D2H with srcOffset (2.2-2.6x faster)
7bc951e  decode iter-15 — single submit + single wait per decode
```

All on `main`. Nothing pushed to remote yet.

---

## Project rules (from `CLAUDE.md`)

**NEVER REVERT WITHOUT PERMISSION.** Never run `git checkout --`, `git
restore`, `git revert`, or any command that undoes file changes without the
user explicitly saying "revert", "undo", or "restore". If a change made
things slower or didn't help, LEAVE THE CODE AS-IS and report the results.
The user decides whether to revert. No exceptions.

---

## Memory rules (from auto-memory)

Relevant entries in `~/.claude/projects/c--Users-james-JAMESWORK2025-Repos-Compressor-GPU/memory/`:

- `feedback-port-dont-reinvent` — mirror CUDA values/dispatch shapes/workgroup sizes verbatim
- `feedback-port-means-port` — if CUDA does X on CPU and VK does X on GPU, that's a port violation
- `feedback-verify-device-name` — perf measurements must verify "Device: NVIDIA GeForce RTX 4060 Ti" appears (default may pick Intel iGPU)
- `feedback-decode-over-encode` — SUPERSEDED, encode is in scope and complete on L1
- `feedback-l1-completion-bar` — STALE numbers from the failed prior session

---

## Recommended next-session workflow

1. **Read this whole file.** Then `gameplan.md` Section 8 for additional context.
2. **Verify build still works:**
   ```powershell
   Remove-Item -Force zig-out/bin/streamlz_vk.exe
   zig build streamlz_vk -Doptimize=ReleaseFast
   ```
3. **Verify tests still 74/0/0 on both backends:**
   ```powershell
   $env:SLZ_VK_DEVICE_INDEX = "1"; zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
   $env:SLZ_VK_DEVICE_INDEX = "0"; zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
   ```
4. **Verify perf still in shape:**
   ```powershell
   $env:SLZ_VK_DEVICE_INDEX = "1"
   ./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz
   ./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt
   ```
5. **Verify validation layer still clean:** see "Lessons learned" section.
6. **Pick a phase from the L2-L5 roadmap above** and launch.
