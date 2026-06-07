# L1 Vulkan Port — Finishing Document

**Purpose:** Post-compact handoff. Everything you need to know to continue the work without re-discovering context. Read this END-TO-END before doing anything substantive.

**Date frozen:** 2026-06-07 (last commit `7ae5f3f` — iter-14).
**Project root:** `c:/Users/james.JAMESWORK2025/Repos/Compressor_GPU`

---

## TL;DR — Status at HEAD

```
Done bar (gameplan Section 1):
  Structural (1:1 pairing audit)         ✅ DONE
  Functional (byte-identical roundtrip)  ✅ DONE on web.txt + enwik8 + silesia
  Functional (cross-backend both ways)   ✅ DONE
  Perf ≤ 1.10× CUDA on enwik8 + silesia  ✅ DONE (0.96× both — VK faster)

Test suite:
  ptest_vk: 67 pass + 7 skip + 0 fail, RELIABLY across 10+ consecutive runs.
  The 7 skips are subprocess tests where the child can't see the discrete
  NVIDIA GPU when spawned from `zig build` (env quirk; not a port bug).

Open follow-ups (NOT blocking done-bar):
  1. web.txt small-file regime: 5.24 ms vs CUDA 2.02 ms = 2.60× SLOWER.
     Per-kernel parity (VK 1.32 ms vs CUDA 1.38 ms = 0.96×); gap is host
     overhead that doesn't shrink with input size. Iter-14 candidate
     identified — persistent descriptor sets + LRU cache audit + maybe
     single-wait consolidation.
  2. iter-14 candidate D2H overlap with LZ tail (not yet investigated).
  3. Discover why zig-build-spawned child processes can't see the discrete
     NVIDIA GPU (env quirk; affects 7 subprocess tests).
```

## Bench numbers at HEAD (`7ae5f3f`)

Fresh-binary verified (M6 protocol), NVIDIA RTX 4060 Ti via `SLZ_VK_DEVICE_INDEX=1`, `-db -r 5`:

| File | VK e2e best | CUDA e2e best | Ratio | VK pure kernel best | CUDA pure kernel |
|---|---|---|---|---|---|
| enwik8 (95 MB) | 15.809 ms | 16.18 ms | **0.98×** | ~3.0 ms | 3.13 ms |
| silesia (203 MB) | 30.254 ms | 31.06 ms | **0.97×** | ~5.5 ms | — |
| web.txt (4.3 MB) | 5.324 ms | 2.015 ms | **2.64×** | 1.319 ms | 1.377 ms |

"Pure kernel best" is sum of per-kernel VkQueryPool timestamps via `last_timings`, post-bench-metric-alignment at commit `d5adeb8`. **Apples-to-apples with CUDA's bench definition.**

---

## Commit history (most recent first — what each one did)

```
iter 15: single-submit/single-wait consolidation per decode. Removed runBackHalf-end stream_sync (was submit 1 + wait 1); finalizeOutput now records its D2H into the SAME compute cmdbuf as the LZ kernel; the post-finalize stream_sync is the lone submit+wait per decode. COMPUTE_SHADER_WRITE → TRANSFER_READ pipeline barrier inserted at top of procD2HAsync to enforce the dependency that the submit boundary used to. Per-decode counts: 2 submits (1 transfer + 1 compute), 1 wait — down from 3 submits + 2 waits. Phase profile confirms backhalf_fence_wait=0.0000ms (was where wait 1 lived), finalize_sync=5.2389ms now absorbs both. Perf: enwik8 15.790ms (baseline 15.666 ms, +0.124 within noise), web 5.207ms (baseline 5.249 ms, -0.042 within noise), silesia 30.001ms (baseline 30.469 ms, -0.468 ms small gain). 67/7/0 ptest_vk; byte-equal roundtrip on web+enwik8+silesia.
7ae5f3f  iter 14: persistent descriptor sets per (StreamEntry × pipeline) + LRU import-cache telemetry. (B) RULED OUT cache-miss hypothesis: web 5h/0m/0e, enwik8 10h/0m/0e in steady state. (A) within-noise perf delta but structurally cleaner.
7e7d74b  iter 13: early-submit transfer cmdbuf (+0.36 ms small gain, structurally cleaner)
d5adeb8  bench: align "gpu kernel" metric with CUDA (pure VkQueryPool sum, not wall-clock)
41c4e2c  ptest_vk 74/74 reliably (3-phase runner, per-test contexts, registry lock, env-skip)
4f00e11  ptest_vk parallelism: 3 real bugs fixed (decode init race, encode init latching, shared-queue submit race)
647bb95  iter 12: H2D zero-copy import on transfer queue — DONE BAR HIT (0.953× enwik8)
b0f810b  iter 9 + iter 11: H2D import gated + split-submit transfer queue infrastructure
8897b33  iter 8: finalize phase 3 subfixes (LRU pool + HOST_CACHED + overlap) 1.46× → 1.21×
7c39558  iter 6-7: instrumentation + D2H zero-copy import
485b829  iter 1: Vulkan init (VMA pVulkanFunctions + V12/V13 feature chain) — decode works end-to-end
22b657d  iter 2: encode-side roundtrip (3 VK-vs-CUDA param-shape bugs)
... (12 more commits down to cff79ec for the original Phases 1-6 structural port)
```

All commits are on `main`. Nothing pushed to remote yet.

---

## Architecture: key invariants that MUST be preserved

The structural-port discipline is what kept us out of trouble. Do not violate:

### `procs.*` surface field names (24 slots, in `srcVK/decode/vulkan_api.zig::Procs`)

The codec calls `procs.h2d(dst, src, size)`. CUDA and VK both. Same name, args, semantics. Vulkan plumbing (vkCmdCopyBuffer, VMA, fences, queues) lives UNDER procs.

Current slots: `malloc_device, free_device, h2d, d2h, h2d_async, d2h_async, d2d, memset_d8, memset_d8_async, malloc_host, free_host, stream_sync, ctx_sync, ctx_set_current, ctx_get_current, stream_create, stream_destroy, launch_kernel, event_create, event_record, event_synchronize, event_elapsed_time, event_destroy, stream_flush_transfer`.

The 24th (`stream_flush_transfer`) was added in iter-13 per the VMA-precedent rule: CUDA conceptually has `cuMemcpyHtoDAsync` issuing the copy on call; the new slot expresses that ordering in the procs surface. CUDA-mirrored naming preserved because `cuMemcpyHtoDAsync` IS a transfer-engine async submit.

### Function decomposition (CUDA-verbatim)

Do NOT rename or restructure:
- `decode/decode_dispatch.zig::fullGpuLaunchImpl` (the orchestrator)
- `decode/decode_dispatch.zig::uploadInputAndPrefixSum`
- `decode/decode_dispatch.zig::runBackHalf`
- `decode/decode_dispatch.zig::finalizeOutput`
- `decode/decode_dispatch.zig::runLzPipeline`
- `decode/decode_dispatch.zig::runHuffBuildAndDecode` (L2 stub — `return error.NotImplementedL2`)
- `decode/decode_dispatch.zig::mergeHuffDescs` / `gatherRawOff16` (L2 stubs)

Same on encode side: `encode/fast_framed.zig::compressFramedOne`, `encode/encode_lz.zig::gpuCompressImpl`, etc.

### L2 gate at `decode_dispatch.zig:724`

`if (req.level >= 2) { ... }` wraps:
- `ensureDeviceBuf` for `d_compact_counts`, `d_scan_staged`, `d_first_sub_idx_persist`, `d_huff_descs`, `d_huff_lut`
- `gpuScanChunks` / `gatherRawOff16` / `mergeHuffDescs` / `runHuffBuildAndDecode` calls

On L1: gate is skipped entirely. CUDA wastefully dispatches 7 L2 kernels (~388 µs/decode) on L1; VK does not. This is **audit V-008 paying off** — we save kernel work CUDA wastes.

### `buildChunkDescriptors` MUST stay CPU walk

In `decode/streamlz_decoder.zig::buildChunkDescriptors` (lines ~267-360). Walks input bytes on CPU, fills a host slice of `ChunkDesc`. The prior failed session at `/src_vulkan/` moved this to GPU as `walk_frame.comp` + `l1_unwrap` kernels — **THAT WAS THE CANONICAL PORT VIOLATION.** Keep it CPU.

`walk_frame_kernel.comp` exists as an L2 stub in `srcVK/decode/` — it's reserved for the L2 D2D entry point (`decompressFramedFromDevice`, currently `return error.NotImplementedL2`).

### KERNEL_DECLS (in `decode/module_loader.zig`)

Pipeline binding counts + push constant sizes per kernel. Do NOT change without auditing all call sites in `decode_dispatch.zig` for the matching `params[]` layout. The two L1 kernel decls:

```
kernel_raw_fn:        n_bindings=4, push_constant_size=8   (CompressedBuf, ChunksBuf, DstBuf, TotalChunksBuf + chunks_per_group + sub_chunk_cap)
prefix_sum_chunks_fn: n_bindings=3, push_constant_size=8   (ChunksBuf, FirstSubIdxBuf, TotalSubchunksBuf + n_chunks + sub_chunk_cap)
```

---

## The current decode flow at HEAD (after iter 11/12/13)

### Initialization (one-time, in `module_loader.zig::init`)

1. Load vulkan-1.dll
2. Enumerate physical devices, pick per `DeviceSelector` (default: discrete > integrated > virtual > cpu; env `SLZ_VK_DEVICE_INDEX=N` override)
3. Create VkDevice with:
   - **Critical feature chain in pNext** (per iter-1):
     - `VkPhysicalDeviceVulkan12Features`: `bufferDeviceAddress`, `shaderInt8`, `storageBuffer8BitAccess`, `uniformAndStorageBuffer8BitAccess`
     - `VkPhysicalDeviceVulkan13Features`: `subgroupSizeControl`, `computeFullSubgroups`
   - Two queue family submissions when a dedicated transfer family exists (per iter-11)
4. Resolve dedicated transfer queue (`g_transfer_queue`) at qf=1 on NVIDIA RTX 4060 Ti (flags=0x0c, TRANSFER+SPARSE_BINDING, no GRAPHICS/COMPUTE)
5. Create VMA allocator with `pVulkanFunctions` populated from `vkGetInstanceProcAddr` + `vkGetDeviceProcAddr` (per iter-1 — VMA was compiled with VMA_DYNAMIC_VULKAN_FUNCTIONS=1)
6. Create VkQueryPool with 4096 timestamp slots (per Phase 6 timing fix-up)
7. Create descriptor pool + load SPV blobs + create pipelines per KERNEL_DECLS
8. Query and cache `g_imported_host_alignment` (typically 4 KiB) + `g_phys_mem_props`
9. Pre-reserve `g_allocs` (cap=4096), `g_streams` (cap=256), `g_host_allocs` (cap=256) — stable backing for concurrent readers (per ptest_vk fix at 41c4e2c)

### Per-decode flow (`fullGpuLaunchImpl`)

1. **Resolve `VkProcs` from `vulkan_api.procs`** at entry
2. **L2 gate** (skipped on L1)
3. **`uploadInputAndPrefixSum`**:
   - Pre-prepare imported D2H buffer for the dst (iter-8 subfix 3 + iter-12)
   - `procs.h2d_async(d_descs_persist, ...)` records chunk_descs H2D into transfer cmdbuf
   - `procs.h2d_async(d_comp_persist, ...)` records the big comp_input H2D — **takes the VK_EXT_external_memory_host import path** (iter-12) when caller buffer is page-aligned + size ≥ 4 MiB (`H2D_IMPORT_THRESHOLD`); else iter-4 staging path
   - `procs.h2d_async(d_n_groups_scratch, &total_chunks, 4)` records the 4-byte n_chunks H2D (moved into here from `runBackHalf` at iter-13)
   - Record prefix_sum_chunks kernel dispatch into compute cmdbuf
   - **`procs.stream_flush_transfer(self.work_stream)` (iter-13)** — end + submit the transfer cmdbuf NOW, signaling `entry.h2d_sem`. The GPU's dedicated DMA engine starts H2D in parallel with host runBackHalf prep.
4. **`runBackHalf`**:
   - Record LZ raw kernel dispatch into compute cmdbuf
   - **`procs.stream_sync(heavy_stream)`** — this is where `streamEndAndWait` fires
5. **`streamEndAndWait`** (in module_loader.zig:2492):
   - If `transfer_already_submitted`: skip transfer submit (iter-13)
   - End compute cmdbuf
   - `vkQueueSubmit` compute cmdbuf, **waiting on `h2d_sem` at COMPUTE_SHADER_BIT**, signaling `entry.fence`
   - `vkWaitForFences(entry.fence)`
   - Flush `pending_d2h` queue (CPU memcpy from staging back to caller's host buffer for non-imported D2H paths)
   - Free pending imported D2H (iter-8 LRU-aware — just flips `in_flight=false`, cache retains)
6. **`finalizeOutput`**:
   - `procs.d2h_async(...)` for the output — takes import path when caller buffer aligned (iter-7 + iter-8 LRU)
   - Final `procs.stream_sync` to flush
7. **`finalizeProfiling`** drains `pending_timings` → `last_timings`

### The Win Pattern

CUDA's `cuMemcpyHtoD_v2` is **synchronous on the host** — by the time CUDA reaches `t_before_kern`, H2D is already done; back-half sync only waits on LZ kernel. VK now achieves the SAME effect via:

1. **Iter-11**: dedicated transfer queue (NVIDIA blit engine fast path) + binary semaphore for H2D→compute dependency
2. **Iter-12**: VK_EXT_external_memory_host import path eliminates the 3.5 ms host @memcpy(58MB) for the big H2D
3. **Iter-13**: early-submit transfer cmdbuf so DMA engine actually starts during host prep

---

## Open follow-up #1: web.txt small-file regime (2.60× slower)

### What we know

| Metric | VK | CUDA | Ratio |
|---|---|---|---|
| web.txt e2e best | 5.239 ms | 2.015 ms | 2.60× |
| web.txt pure kernel best | 1.319 ms | 1.377 ms | 0.96× (parity) |

**Kernel work is fine. Gap is ~3.2 ms host coordination overhead that doesn't shrink with input size.**

### nsys API-level findings (workflow `wkjtgdzlf`, 2026-06-07)

**Caveat:** The Nsight VK trace layer adds per-call overhead that inflates absolute timings. The `vkWaitForFences total per decode = 6.31 ms` figure is larger than e2e best (5.24 ms), which is impossible without instrumentation overhead — **do not trust absolute magnitude claims from this trace; trust the relative/structural shape.**

Per-decode API counts (likely reliable):

| Category | VK | CUDA | Note |
|---|---|---|---|
| Submit | 3× vkQueueSubmit (1 transfer + 2 compute?) | 0 explicit | iter-13 should give 2 submits/decode but trace showed 3 — needs verification |
| Wait | 2× vkWaitForFences | 2× cuStreamSynchronize | Similar structural shape |
| Descriptor allocation | 2× vkAllocateDescriptorSets @ 127 µs each ≈ 250 µs | 0 | **fixable with persistent sets** |
| Memory allocation | 1.25× vkAllocateMemory @ 246 µs avg ≈ 308 µs amortized | 4.25× cuMemAlloc @ 27 µs | LRU import may not be hitting reliably |
| Kernel launch | ~2 vkCmdDispatch | 12 cuLaunchKernel @ 7.5 µs (L2 wasteful) | CUDA cheap per launch, dispatches more |
| Command buffer | 3× vkBegin/EndCommandBuffer | 0 | ~115 µs/decode |

### Iter-14 candidates (recommended in priority order)

#### (A) Persistent descriptor sets — **DONE at iter-14 (`7ae5f3f`)**
- **Result:** Implemented per (StreamEntry × pipeline) for async streams + KernelMeta.persistent_desc_set_default for the sync default-stream path. vkUpdateDescriptorSets + vkCmdBindDescriptorSets only per call.
- **Perf delta:** within noise (web +0.025 ms, enwik8 -0.060 ms, silesia -0.234 ms — silesia improved ~0.8%, others flat). Either the nsys overcounted vkAllocateDescriptorSets, or the cost shifted to vkUpdateDescriptorSets which we still do per call.
- **Status:** Shipped for structural cleanliness; no perf win on small files.
- **Race handling:** per-stream persistent_desc_sets are written + bound on the same host thread that owns stream's cmdbuf; default-stream slot protected by existing g_dispatcher_lock.

#### (B) LRU cache audit — **DONE at iter-14 (`7ae5f3f`)**
- **Telemetry added:** atomic counters g_import_cache_hits / misses / evictions, surfaced via SLZ_VK_PROFILE_PHASES=1 as `phase: import_cache hits=N, misses=M, evictions=K`.
- **Finding (web.txt, 5 measured decodes):** **5 hits / 0 misses / 0 evictions.**
- **Finding (enwik8, 5 measured decodes):** **10 hits / 0 misses / 0 evictions.** (2 imports/decode: src + dst)
- **Conclusion:** The iter-8 16-slot LRU import cache is hitting **RELIABLY** in steady state. The nsys "vkAllocateMemory 308 µs amortized" finding was WARM-UP ONLY.
- **THIS RULES OUT the import path as the residual web.txt gap source.**
- **Note:** web.txt only does 1 import/decode (the dst) because src is 2.1 MB, below the 4 MiB H2D_IMPORT_THRESHOLD; src takes the iter-4 staging path.

#### (C) Pre-recorded compute cmdbuf reuse — DEFERRED
- **Expected gain:** ~5 µs/decode steady state. Likely not worth chasing alone.

#### (D) Eliminate the second wait point (consolidate to single submit/wait per decode) — **ONLY REMAINING ATTACK VECTOR**
- **Claimed expected gain:** 2.4 ms (BUT magnitude is suspect — agent's vkWaitForFences number was larger than e2e)
- **Risk:** HIGH (substantive architectural change)
- **Mechanism options:**
  1. GPU-driven indirect dispatch: `vkCmdDispatchIndirect` driven by prefix-sum kernel output, so the LZ dispatch sizing happens GPU-side and the whole decode can record into one cmdbuf, submit once, wait once.
  2. Statically over-allocate threads to max n_chunks bound (cheaper-to-implement variant of 1).
  3. Pre-allocate the second compute cmdbuf + persistent descriptors + persistent imports at init, so the host work between the two waits is minimal (~15 µs).
- **Decision:** After iter-14 confirmed cache + descriptor allocation are NOT the gap, (D) is the only remaining attack vector. Whether to chase it is a judgment call between expected gain (1-3 ms on web.txt, modest on large files) and substantive architectural risk. Recommend ONLY if user wants to push small-file perf below 2× of CUDA.

---

## Open follow-up #2: iter-14 D2H overlap with LZ tail

Not yet investigated. The idea: today the imported D2H copy runs sequentially AFTER the LZ kernel completes. If we record the D2H into the compute cmdbuf with a pipeline barrier, the copy engine can start writing output bytes while the LZ kernel is still finishing the trailing chunks. Requires understanding LZ kernel's output write pattern (does it write monotonically, or can later writes overwrite earlier output?).

Skip unless web.txt small-file work is done and you want more enwik8/silesia headroom.

---

## Open follow-up #3: discrete VK invisible to zig-build-spawned children

`tests/cli_smoke.zig` and `tests/cross_backend_roundtrip.zig` spawn child `streamlz_vk.exe` processes via `std.process.run`. When spawned **from `zig build ptest_vk`**, the child only sees the Intel iGPU — the discrete NVIDIA GPU isn't enumerated by the Vulkan loader. Interactive-shell invocation of the same binary sees both devices fine.

iter-fix at `41c4e2c` added `discreteVkVisibleToChild()` to skip these tests cleanly when discrete is invisible. **That's why ptest_vk reports 7 skipped.** The tests aren't broken; the spawn environment is.

To investigate:
- What's in `zig build`'s child env that suppresses the discrete ICD? `VK_ICD_FILENAMES`? `VK_LOADER_LAYERS_DISABLE`?
- Try `Get-Process` on the spawned child and inspect its env
- Compare loader-debug output (`VK_LOADER_DEBUG=all`) between interactive vs spawned

Quality-of-life only. Tests are correctly skipping; no test is silently false-passing.

---

## Known traps — DO NOT REPEAT

### Stale binary (M6 in gameplan)

We were burned multiple times by `zig-out-rf/bin/streamlz_vk.exe` (a stale prior build) being on PATH. Always:

```powershell
Remove-Item -Force zig-out/bin/streamlz_vk.exe
zig build streamlz_vk -Doptimize=ReleaseFast
# Verify mtime > newest source:
(Get-Item zig-out/bin/streamlz_vk.exe).LastWriteTime
(Get-ChildItem -Recurse srcVK/ -Include *.zig,*.glsl,*.comp | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime
# Capture SHA256 for the record:
(Get-FileHash -Algorithm SHA256 zig-out/bin/streamlz_vk.exe).Hash
```

### Wrong device (M5 in gameplan)

Default device picker selects DISCRETE first, but on systems with both Intel iGPU and NVIDIA, the bench may pick Intel without `SLZ_VK_DEVICE_INDEX=1`. Always:

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"
# Verify "Device: NVIDIA GeForce RTX 4060 Ti" appears in bench output
```

### Kernel-parity claim trap (caught at iter-5/6)

Iter 5's `nsys vulkan_api_sum` reported `vkWaitForFences avg 12.23 ms` and the bench reported `gpu_kernel_best 8.098 ms`, leading iter-5 to claim "VK kernels are 1.79× slower than CUDA." **That was wrong.** Iter 6 measured VK kernels directly via VkQueryPool and found parity (3% faster on LZ; 21% slower on tiny prefix_sum but absolute 22 µs).

**Lesson:** The bench's old `last_kernel_ns` wall-clock metric is NOT pure kernel time; it includes back-half H2D copy engine work. Always cross-check kernel-perf claims with the per-kernel VkQueryPool sum (now in `last_timings` after the d5adeb8 bench-alignment commit).

### nsys VK trace layer overhead

The Nsight VK trace layer adds per-call overhead. Absolute VK API timings from `vulkan_api_sum` are inflated relative to non-instrumented runs. **Trust the structural shape (call counts, relative magnitudes) — don't trust absolute µs.**

When you need absolute API timings, prefer:
- VkQueryPool timestamps (GPU side, no host instrumentation)
- QPC checkpoints in code (`g_phase_*_ns` under `SLZ_VK_PROFILE_PHASES=1`)

### Rationalization phrases caught session-wide

These are FORBIDDEN in code / docstrings:

- "Phase N fleshout" / "deferred to Phase N" / "downstream phase" / "wired later"
- "kept as cuda" / "per port mandate" / "VK PORT NOTE: file kept"
- "open-coded by" / "inline the loop" / "callers should inline"
- "L1 raw kernel never" / "never calls this" / "once X is fully ported"
- "for the L2 fleshout" / "in this fleshout" / "until X lands"
- "padded to" (when justifying workgroup deviation)
- "can't reliably load" / "refuse the call"

ALLOWED: `// VK adaptation: <one-line technical mechanism>` describing HOW Vulkan achieves the same logical result. Test: does the comment explain how the implementation works, or why work isn't being done? Former OK, latter forbidden.

### "Port" means port — V-001/V-002 traps

The audit caught divergences where srcVK was about to do something different from CUDA structurally (buildChunkDescriptors moved to GPU was the canonical violation). **Mirror CUDA verbatim except for legitimate Vulkan-required adaptations** (queue families, command buffers, semaphores, descriptor sets, fence-based sync). Anything that has a direct CUDA counterpart MUST be ported, not reinvented.

The VMA-precedent rule (user authorized late-session): adopting a Vulkan-native pattern (VMA, async transfer queue, semaphore split-submit, persistent descriptors) IS allowed when CUDA doesn't have a faithful counterpart AND the procs.* surface + function decomposition stay verbatim. The pattern goes UNDER the surface; call sites stay CUDA-shaped.

### Test runner parallelism — iter `4f00e11` + `41c4e2c`

Three real races were uncovered, not just test flakiness:

1. **Decode init race:** Thread A flips `init_state` to `.in_progress`; Thread B sees that and short-circuits to `error.BackendNotAvailable`. Fix: `g_init_lock` SRWLOCK in `decode/module_loader.zig::init`.
2. **Encode init latching bug:** `initialized = true` was set at function entry BEFORE init body ran. Mid-init failure left `initialized=true` with `kernel_fn=0`, permanently denying encode dispatch. Fix: moved to `defer if (ok)`. **Would have bitten any consumer hit by transient init failures.**
3. **Encode/decode shared compute-queue submit race:** Vulkan requires per-queue external synchronization. Both encode (sync path, `g_command_buffer`) and decode (per-stream cmdbuf) submitted to compute queue. Fix: `lockEncodeDispatcherMutex` routing through the existing decode `g_dispatcher_lock`.

Plus at `41c4e2c`:
- **Registry lock + VkCommandPool external sync:** Exit-code-5 segfaults surfaced only after per-test contexts landed — concurrent registry mutation tore sibling pointers; `vkAllocateCommandBuffers` raced on `g_command_pool`. Fix: `g_alloc_registry_lock` SRWLOCK + pre-reserved capacities.

All locks gated on `work_stream == 0` (sync mode only). Async bench path uses `work_stream != 0` and isn't locked — confirmed not regressed.

---

## How to measure (apples-to-apples with CUDA)

### Bench

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz
```

The bench enables `enable_profiling=true` on `gpu_dec_driver.g_default` (per d5adeb8) so VkQueryPool timestamps populate. The "gpu kernel best/mean" line is the SUM of `last_timings` entries — pure kernel time, matches CUDA's bench definition.

### Per-phase host-overhead profile

```powershell
$env:SLZ_VK_PROFILE_PHASES = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz 2>&1 | Select-String "phase:"
```

QPC checkpoints around each logical phase (upload_h2d, backhalf_h2d, backhalf_lz_launch, backhalf_fence_wait, finalize_d2h, finalize_sync, import_prep, h2d_paths counter, etc.). Implementation in `srcVK/decode/decode_dispatch.zig` (the `g_phase_*_ns` accumulators) + `srcVK/cli/bench_decompress.zig::phaseProfileInit`.

### Per-kernel VkQueryPool dump

```powershell
$env:SLZ_VK_PROFILE_DECODE = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz 2>&1 | Select-String "kper:"
```

Emits `kper: <kernel_name> count=N total=Xms avg=Yus` per kernel.

### nsys VK trace (per gameplan M7)

```powershell
$NSYS = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/nsys.exe"
$env:VK_ADD_LAYER_PATH = "C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.2/target-windows-x64/vulkan-layers"
$env:VK_INSTANCE_LAYERS = "VK_LAYER_NV_nsight-sys"
$env:ENABLE_VK_LAYER_NV_nsight_sys = 1
$env:SLZ_VK_DEVICE_INDEX = "1"
& $NSYS profile --trace=vulkan --output=c:/tmp/<name> --force-overwrite=true ./zig-out/bin/streamlz_vk.exe -db -r 3 tests/goldens/<file>.L1.slz
& $NSYS stats c:/tmp/<name>.nsys-rep --report vulkan_api_sum
```

**Caveat:** Absolute API timings are inflated by the trace layer. Use for structural / call-count analysis; verify magnitude with VkQueryPool or QPC if it matters.

### nsys CUDA trace

```powershell
& $NSYS profile --trace=cuda --output=c:/tmp/<name>_cuda --force-overwrite=true ./zig-out/bin/streamlz.exe -db -r 3 tests/goldens/<file>.L1.slz
& $NSYS stats c:/tmp/<name>_cuda.nsys-rep --report cuda_gpu_kern_sum
```

CUDA tracing is built-in; no env-var dance needed.

### CUDA bench

```powershell
./zig-out/bin/streamlz.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz
```

CUDA binary at `zig-out/bin/streamlz.exe` (mtime 2026-06-02; pre-session, unchanged). Outputs same metric shape as VK bench so they can be diffed directly.

---

## ptest_vk discipline

After ANY change to `srcVK/decode/` or `srcVK/encode/`:

```powershell
Remove-Item -Force zig-out/bin/streamlz_vk.exe
zig build streamlz_vk -Doptimize=ReleaseFast
zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
```

Expect: `67 passed, 7 skipped, 0 failed`. Any deviation = regression — investigate before committing.

If new test failures appear:
1. Reproduce by CLI: spawn `streamlz_vk.exe -d <golden>` or `streamlz_vk.exe -c <input>` and verify the underlying logic works
2. If CLI repro works but ptest_vk fails: likely a parallelism issue (shared state across workers)
3. If CLI repro fails: real port correctness bug — fix at the source

### Test runner architecture (iter `41c4e2c`)

`srcVK/test_runner_parallel.zig` does three-phase partitioning by name suffix:
- Tests ending `[serial_first]` run BEFORE the parallel pool (main thread, sequential)
- Unmarked tests run in the 16-worker parallel pool (or `SLZ_VK_TEST_THREADS=N` override)
- Tests ending `[serial]` run AFTER the pool joins (main thread, sequential)

`[serial_first]` is used for null-default assertions that must observe a fresh, uninitialized loader. `[serial]` is used for subprocess tests that spawn separate `streamlz_vk.exe` processes.

Per-test contexts: `tests/l1_encode_roundtrip.zig`, `tests/l1_decode_roundtrip.zig`, `tests/kernel_conformance.zig` allocate fresh `EncodeContext` + `DecodeContext` on the test stack instead of sharing `g_default`. Cross-test interference impossible.

---

## File map — where to look

### Core decode path

- `srcVK/decode/streamlz_decoder.zig` — top-level `decompressFramed` + CPU `buildChunkDescriptors`
- `srcVK/decode/decode_dispatch.zig` — `fullGpuLaunchImpl` orchestrator + L2 gate + phase QPC instrumentation
- `srcVK/decode/decode_context.zig` — `DecodeContext` workspace pool, `ensureDeviceBuf` (grow-only), KernelTiming/finalizeProfiling
- `srcVK/decode/scan_gpu.zig` — `gpuPrefixSumChunksImpl` (L1) + `gpuScanChunks` (L2 stub) + `gpuWalkFrameImpl` (L2 stub)
- `srcVK/decode/module_loader.zig` — **the heavy one.** VkDevice init, transfer queue, VkQueryPool, procs implementations, LRU import cache, streamCreate/Destroy, streamEndAndWait two-submit dance, procStreamFlushTransfer (iter-13)
- `srcVK/decode/vulkan_api.zig` — procs surface (24 slots)
- `srcVK/decode/descriptors.zig` — POD types (ChunkDesc, KernelTiming, GpuError)
- `srcVK/decode/driver.zig` — facade with pub vars (g_default, last_kernel_ns, etc.)

### L1 decode kernels

- `srcVK/decode/lz_decode_raw_kernel.comp` — slzLzDecodeRawKernel (the L1 workhorse)
- `srcVK/decode/prefix_sum_chunks_kernel.comp` — slzPrefixSumChunksKernel
- `srcVK/decode/lz_decode_core.glsl` — warpScanU32 / warpLiteralCopy / warpMatchCopy
- `srcVK/decode/lz_decode_raw.glsl` — decodeSubChunkRawMode_{true,false}
- `srcVK/decode/lz_dispatch.glsl` — parseAndDecodeSubChunkRaw / parseAndDecodeSubChunk
- `srcVK/decode/slz_wire_format.glsl` — wire format constants + struct layouts

### CLI

- `srcVK/cli/bench_decompress.zig` — `-db` mode; bench-metric-aligned with CUDA (d5adeb8)
- `srcVK/cli/decompress.zig` — `-d` mode; calls `releaseImportsByHostRange` before mmap.unmap (iter-8 bonus fix)
- `srcVK/cli.zig` — dispatcher; prints `Device: <deviceName>` at startup before mode dispatch
- `srcVK/cli/util.zig` — Args + DeviceSelector

### Tests

- `srcVK/tests/decoder_unit.zig` (22 tests, pure host)
- `srcVK/tests/encoder_unit.zig` (11 tests, pure host)
- `srcVK/tests/dispatch_unit.zig` (7 tests, includes KERNEL_DECLS ABI check)
- `srcVK/tests/kernel_conformance.zig` (per-test context)
- `srcVK/tests/l1_decode_roundtrip.zig` (per-test context)
- `srcVK/tests/l1_encode_roundtrip.zig` (per-test context)
- `srcVK/tests/cross_backend_roundtrip.zig` (`[serial]` — subprocess)
- `srcVK/tests/cli_smoke.zig` (`[serial]` — subprocess; skips when discrete invisible)
- `srcVK/test_runner_parallel.zig` (three-phase partitioning)
- `srcVK/decode/vulkan_api.zig::test "procs slots default to null [serial_first]"`
- `srcVK/encode/vulkan_ffi.zig::test "encode FFI slots default to null [serial_first]"`

### Reference docs (DO NOT MODIFY)

- `srcVK/audit.md` — canonical file mapping, extension/token rules
- `srcVK/PortInstructions.md` — per-file fleshout checklist
- `srcVK/gameplan.md` — comprehensive port playbook (Sections 1-10)
- This file (`srcVK/finishingL1.md`) — post-compact handoff

---

## Memory rules (from auto-memory)

Relevant rules in `~/.claude/projects/c--Users-james-JAMESWORK2025-Repos-Compressor-GPU/memory/`:

- **`feedback-port-dont-reinvent`** — for the Vulkan port, mirror CUDA values/dispatch shapes/workgroup sizes verbatim; do not design new ones from Nsight or hardware specs.
- **`feedback-port-means-port`** — STRICT corollary. If CUDA does X on CPU and VK does X on GPU (or any architectural divergence for same logical work), THAT IS NOT A PORT. Never describe such divergence as "deliberate port decision."
- **`feedback-verify-device-name`** — perf measurements must verify `Device: NVIDIA GeForce RTX 4060 Ti` is printed; default may pick Intel iGPU.
- **`feedback-decode-over-encode`** — SUPERSEDED for this port (encoder IS in scope, fully working post iter-2).
- **`feedback-l1-completion-bar`** — STALE numbers (the 1.14× was for the failed prior session); current bar is met at 0.95×.

Project rule from `CLAUDE.md`:
- **NEVER REVERT WITHOUT PERMISSION.** Use `git checkout --` / `git restore` / `git revert` only when user says "revert", "undo", or "restore". If a change made things slower or didn't help, LEAVE IT and report. The user decides.

---

## Recommended next-session workflow

If you (post-compact me, or any future agent) are picking this up:

1. **READ THIS WHOLE FILE.** Then read `srcVK/gameplan.md` Section 8 (per-phase roadmap) for additional context.
2. **Verify build still works:**
   ```powershell
   Remove-Item -Force zig-out/bin/streamlz_vk.exe
   zig build streamlz_vk -Doptimize=ReleaseFast
   ```
3. **Verify tests still 67/7/0:**
   ```powershell
   zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
   ```
4. **Verify perf still in shape:**
   ```powershell
   $env:SLZ_VK_DEVICE_INDEX = "1"
   ./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz
   # expect e2e best ~15-16 ms, gpu kernel best ~3 ms
   ```
5. **If user wants the web.txt fix:** launch iter-14 (A) + (B) combined workflow per the recipe in "Open follow-up #1" above.
6. **Always commit after each user-confirmed iteration.** The user values incremental git history; the commit message should describe what changed, the perf delta if any, and the test pass count.

---

## Things that emphatically DO NOT need to change

Avoid scope creep. The following are DONE and battle-tested:

- The L1 decode + L1 encode core kernels (work byte-equal vs CUDA across all three goldens)
- The L2 gate position and gating logic
- The VK_EXT_external_memory_host import path on the dedicated transfer queue
- The HOST_CACHED memory type selection in `tryImportHostBuffer`
- The LRU import cache (16 slots; per-host_ptr keying with usage_src/dst discriminator)
- The 3-phase test runner partitioning
- The `g_init_lock` + `g_encode_init_lock` + `g_alloc_registry_lock` SRWLOCKs (3 separate, none recursive)
- The bench's pure-VkQueryPool-sum "gpu kernel" metric
- buildChunkDescriptors CPU walk
- VkPhysicalDeviceVulkan12Features + VkPhysicalDeviceVulkan13Features pNext chain

Touch these only if you have a SPECIFIC reason and a measurement showing the change helps.

---

## Final word

The port is in good shape. The done-bar is met. The remaining work (web.txt small-file regime, optional D2H overlap, environmental zig-build child-spawn quirk) is **quality-of-life, not load-bearing**. If the user is happy stopping here, that's a complete L1 Vulkan port — structurally faithful to CUDA, byte-equal in roundtrip, cross-backend compatible, perf-competitive on large workloads, and reliably testable.

If the user wants to push for web.txt perf, iter-14 (A)+(B) is the safe path. (C) and (D) are riskier with less-certain gains.

**Do not delete this file.** Add to it as iterations land.
