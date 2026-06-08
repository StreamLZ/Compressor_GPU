# srcVK Vulkan Port — L1 Done, L2-L5 Roadmap

**Last updated:** 2026-06-07, HEAD `7bbcf77`.
**Status:** L1 + L2 SHIPPED on both decode AND encode. VK is byte-identical to CUDA and FASTER than CUDA on enwik8 + silesia for both directions on large workloads at L1 + L2. ptest_vk 83/0/0 both backends. Web.txt (small-file regime) is structurally limited by Vulkan-on-WDDM per-submit floor — same on both directions, documented as accepted.

This file is the source of truth for "what's left." All of L1 is locked-in
production code; the meaningful remaining work is L2-L5 to reach full CUDA
parity. The `/src_vulkan/` directory contains an earlier (now-deprecated)
implementation that ported less faithfully; its `TODO.md` informed parts of
this roadmap.

---

## Status: L1 is DONE on both decode AND encode

| Criterion | NVIDIA RTX 4060 Ti | Intel(R) Graphics iGPU |
|---|---|---|
| Structural 1:1 port audit | ✅ | ✅ |
| Byte-equal roundtrip (web + enwik8 + silesia) | ✅ | ✅ |
| Cross-backend (CUDA ↔ VK both directions) | ✅ | ✅ |
| **Decode** perf ≤ 1.10× CUDA on large workloads | ✅ **0.97-0.98×** | works |
| **Encode** perf ≤ 1.10× CUDA on large workloads | ✅ **0.82-0.87× (FASTER)** | byte-identical, works |
| **Encode SHA byte-identical to CUDA on all 3 goldens** | ✅ | ✅ |
| ptest_vk | **74/0/0** | **74/0/0** |
| VK validation (spec) | 0 VUIDs | 0 VUIDs |
| VK validation (synchronization) | 0 hazards | 0 hazards |
| VK validation (best practices, bench mode) | 0 | 0 |

Current bench numbers (NVIDIA, `SLZ_VK_DEVICE_INDEX=1`, `-db -r 5` or `-b -r 5`, post-warmup):

| File | Decode e2e | Decode kernel | Encode | vs CUDA decode | vs CUDA encode |
|---|---|---|---|---|---|
| enwik8 (95 MB) | 15.7 ms | 3.0 ms | **103 ms** | 0.97× | **0.82×** |
| silesia (203 MB) | 30.0 ms | 5.5 ms | **209 ms** | 0.97× | **0.87×** |
| web.txt (4.3 MB) | 5.3 ms | 1.3 ms | 19 ms | 2.6× (WDDM floor) | 1.73× (WDDM floor) |

Decoder reference: CUDA enwik8 16.2 ms / silesia 31.1 ms / web 2.0 ms.
Encoder reference: CUDA enwik8 125 ms / silesia 241 ms / web 11 ms.

---

## L2-L5 roadmap to full CUDA parity

The CUDA encoder supports five levels:

| Level | Parser | Hash bits | Huffman | LZ rehash | Status |
|---|---|---|---|---|---|
| **L1** | greedy | 17 | NO | NO | ✅ **DONE** — byte-identical to CUDA + FASTER on large workloads |
| **L2** | greedy | 18 | NO | NO | ✅ **DONE** (`7bbcf77`) — byte-identical to CUDA + FASTER; 9 ptest cases added |
| L3 | greedy | 19 | YES | NO | needs Phase 2A Huffman |
| L4 | greedy | 17 | YES | YES (`engine_level≥2`) | needs Phase 2A + small `p_l4=1` flag |
| L5 | **chain parser** | 17 | YES | YES | needs Phase 2A + 2B chain parser (+ fix broken branch) |

**Important correction from L1 work**: The original `/src_vulkan/TODO.md` claimed L2 needed Huffman. **It didn't.** Verified via `src/encode/fast_framed.zig:247-251`: Huffman gates at `opts.level >= 3`. L2 is greedy parser + bigger hash table only. The VK encoder already passes `hash_bits = levels.hashBitsForLevel(level)` through to the kernel (`srcVK/encode/encode_lz.zig:66`), so L2 worked with zero code changes once L1 was byte-identical. The encode iter-3 + iter-4 fixes that brought L1 to byte-identity automatically applied to L2.

### Phase 2-prep — VERIFY L2 WORKS — **DONE** (`7bbcf77`)

L2 SHIPPED with zero new code. Quick verification + ptest cases added:

| File | VK L2 SHA | CUDA L2 SHA | VK encode ms | CUDA encode ms | Ratio |
|---|---|---|---|---|---|
| web.txt | match | match | 24 | 12 | 2.0× (WDDM floor) |
| enwik8 | match | match | **112** | 130 | **0.86× — FASTER** |
| silesia | match | match | **223** | 249 | **0.90× — FASTER** |

9 new ptest cases (5 in-process + 4 cross-backend SHA gate). ptest_vk 74/0/0 → **83/0/0** both backends.

### Phase 2A — Huffman codec (unlocks L3 + L4)

Goal: encoder produces chunk type 4 (Huffman-coded) sub-chunks that the CUDA
decoder reads; decoder reads CUDA-produced chunk type 4. Cross-backend
roundtrip parity across L3+L4 in both directions.

**Kernels to port** (CUDA references in `src/encode/huffman_kernel.cu` +
`src/decode/huffman_kernel.cu`):
- `srcVK/encode/huff_build_tables_kernel.comp` — histogram → canonical Huffman codes
- `srcVK/encode/huff_encode_4stream_kernel.comp` — bytes + codes → BIL bitstream
- `srcVK/decode/huff_build_lut_kernel.comp` — weights → 256-entry decode LUT
- `srcVK/decode/huff_decode_4stream_kernel.comp` — bitstream + LUT → bytes

**Host glue:**
- Generalize `encode/encode_lz.zig` to wire Huffman after greedy parser
- Generalize `decode/decode_dispatch.zig::runLzPipeline` to dispatch type-4 sub-chunks through Huffman before LZ-decode
- Open the L2 gate at `decode_dispatch.zig:724` for type-4 path
- Bonus L4: also flip `p_l4=1` (one-line change in encode_lz.zig)

**Tests:**
- New `tests/l3_encode_roundtrip.zig`, `tests/l3_decode_roundtrip.zig`
- New `tests/l4_encode_roundtrip.zig`
- Extend `tests/cross_backend_roundtrip.zig` for L3-L4

LOC estimate: ~1,500 GLSL + ~800 Zig. ~3-5 single agents.

### Phase 2B — L5 chain parser

Goal: L5 .slz files byte-correct vs CUDA L5 (within the hash-store-order
tolerance accepted for L1).

**New shader:** `srcVK/encode/lz_chain_encode_kernel.comp` — port of
`src/encode/lz_chain_parser.cuh` (~440 CUDA LOC). Most complex single
kernel in the project — chain table + secondary hash + lazy matching.

**Also**: investigate why VK L5 currently produces 1,966 bytes garbage output
(the chain parser kernel branch isn't ported, so `use_chain=1` falls into a
broken code path). Fix that too.

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

**The decode/encode procs surface already supports streams** (iter-11 split
queue, iter-13 early submit, iter-15 single-submit/single-wait, encode iter-3
transfer-queue gather, encode iter-4 imported D2H), so async is mostly
host-glue work.

Items:
- True buffer-device-address D2D for `slzCompress_vk` / `slzDecompress_vk`
- Async encode/decode worker pool wrapping the sync codec
- `slzGetLastTimings_vk` to drain VkQueryPool data

LOC estimate: ~600 Zig. ~1-2 single agents.

### Phase 5 — Conformance + perf parity validation

- Extend `tests/cross_backend_roundtrip.zig` for L1-L5 in both directions on both backends
- Benchmark vs CUDA on each level; target ≤1.10× on large workloads
- Document any residual gaps as accepted tradeoffs

LOC estimate: ~400 Zig. ~1 single agent.

### Total estimate to full CUDA parity

| Phase | GLSL LOC | Zig LOC | Agent runs |
|---|---|---|---|
| ~~2-prep (verify L2)~~ | ✅ DONE | ✅ DONE | ✅ |
| 2A (Huffman, L3 + L4) | ~1,500 | ~800 | 3-5 |
| 2B (L5 chain parser + fix L5 broken branch) | ~800 | ~500 | 2-3 |
| 3 (GPU decode pipeline) | ~1,500 | ~600 | 3-4 |
| 4 (D2D + async) | 0 | ~600 | 1-2 |
| 5 (conformance + perf) | 0 | ~400 | 1 |
| **REMAINING TOTAL** | **~3,800** | **~2,900** | **10-15** |

---

## Kernel inventory (current state)

### L1 kernels — REAL (working in production, byte-identical to CUDA)

| File | Status | Purpose |
|---|---|---|
| `decode/lz_decode_raw_kernel.comp` | ✅ real | L1 raw-mode LZ decode workhorse |
| `decode/prefix_sum_chunks_kernel.comp` | ✅ real | L1 prefix sum over chunk descs |
| `encode/lz_encode_kernel.comp` | ✅ real (~300 LOC) | L1 greedy LZ encode — NCU-verified parity with CUDA PTX |
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
native concept CUDA doesn't have a direct analog for.

Current slots (26): `malloc_device, free_device, h2d, d2h, d2h_offset,
h2d_async, d2h_async, d2d, memset_d8, memset_d8_async, malloc_host,
free_host, stream_sync, ctx_sync, ctx_set_current, ctx_get_current,
stream_create, stream_destroy, launch_kernel, event_create, event_record,
event_synchronize, event_elapsed_time, event_destroy, stream_flush_transfer,
stream_compute_barrier, d2h_offset_gather`.

The `d2h_offset_gather` slot (iter-3 + iter-4) is the highest-leverage VK
adaptation — eliminates per-call submit floor for both per-chunk D2Hs AND
the final ~50 MB output D2H.

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
to GPU — **THAT WAS THE CANONICAL PORT VIOLATION.** Keep it CPU for L1.

### KERNEL_DECLS (decode + encode module_loader.zig)

Pipeline binding counts + push constant sizes per kernel. Do NOT change
without auditing all call sites for matching `params[]` layout. Adding L2
kernels means adding KERNEL_DECLS entries with correct
`pin_subgroup_32: bool` (true unless `local_size_x == 1`).

### Subgroup size pinned to 32 (gpu_warp.glsl contract)

Both decode + encode pipeline creation pin `requiredSubgroupSize=32` +
`REQUIRE_FULL_SUBGROUPS_BIT`. Device-pick guard rejects devices that can't
satisfy `[minSubgroupSize<=32, maxSubgroupSize>=32]`. Kernels in
`gpu_warp.glsl` hardcode `WARP_SIZE=32u`. **NVIDIA only offers 32; Intel
supports the pin via UHD/Iris/Arc; AMD also supports via wave32 path.**

### Persistent buffers on EncodeContext

Encoder relies on three persistent host buffers, all page-aligned via
`gpu_dec_driver.allocHost` → `procMallocHost` (Vulkan VMA host alloc), so
the iter-8 LRU import cache hits on every call after the first:

- `gpu_out_buf` (iter-3): destination of the per-chunk D2H gather. ~50 MB
  on enwik8.
- `d2h_final_buf` (iter-4): destination of the final-frame D2H. ~50 MB on
  enwik8.

These eliminate the staging memcpy at the host. Without them the iter-7/8/12
import path can't fire (refuses non-page-aligned host buffers).

### VkPhysicalDeviceVulkan12Features + Vulkan13Features required

`bufferDeviceAddress`, `shaderInt8`, `storageBuffer8BitAccess`,
`uniformAndStorageBuffer8BitAccess`, `subgroupSizeControl`,
`computeFullSubgroups`. Enabled at VkDevice creation. **If any of these
isn't supported, init returns `error.BackendNotAvailable` so the codec falls
back to CPU.**

### Persistent disk-backed VkPipelineCache (commit 02560d6)

All 17 compute pipelines (11 decode + 6 encode) share a process-wide
VkPipelineCache that loads from `%LOCALAPPDATA%/streamlz_vk/pipeline_cache.bin`
at init and saves at process exit (atexit-registered). Driver validates
header per Vulkan 1.0 spec § 10.5.4 — silently discards mismatches.

Cold-start savings on NVIDIA RTX 4060 Ti: ~38 ms (5.4%). Warm bench unchanged
(bench pre-warms within a single process). Adds correctness for one-shot
CLI tools that repeatedly invoke encode/decode.

---

## Lessons learned (apply to L2-L5 work)

### Always run validation layer before claiming "done"

Standard layer catches spec violations. Synchronization layer catches missing
barriers and race conditions. Best-practices layer catches anti-patterns.
**Three real latent bugs were caught only by validation:**

- VUID-vkBindBufferMemory-memory-02985 (iter-12 import path missing buffer-create chain — `9022dde`)
- VUID-vkCmdCopyBuffer/EndCommandBuffer-recording (cmdbuf invalidated by destroyed buffer — `7960b98`)
- VUID-VkShaderModuleCreateInfo-pCode-08737 (glslc optimizer bug — `23d8ab3`)

```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH = "c:/tmp/vk_layer_settings.txt"  # contents below
./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/v.slz assets/web.txt 2>&1 | Select-String "VUID|Validation Error"
```

`vk_layer_settings.txt`:
```
khronos_validation.enables = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
khronos_validation.report_flags = error,warn,perf,info
```

### SHA byte-identical to CUDA is MANDATORY — roundtrip-passes is INSUFFICIENT

The H2 disaster taught us this the hard way. Encode iter-1 H2 broke
compression on inputs ≥ 8 MiB — produced essentially uncompressed output
that VALID-roundtripped through VK→VK. It passed all roundtrip + cross-
backend tests because all-raw output is structurally valid SLZ. **Was caught
only by comparing output SIZE to CUDA.**

**Every encode iteration MUST verify SHA256(VK_output) == SHA256(CUDA_output)**
on web/enwik8/silesia. We had to bisect + revert 2 commits (H2 + iter-2)
when this was missed. The cost of one extra SHA check >>> the cost of
reverting a perf iteration.

The same applies to any L2-L5 work touching encode.

### NEVER hand-type Vulkan constants

Commit `6ca302c` was a 4-line constant correction that fixed an entire
broken-pin commit. Two enum values had been typed from memory and were both
wrong. **Always pull constants from `C:/VulkanSDK/1.4.341.1/Include/vulkan/vulkan_core.h`** —
grep the header, copy the exact value, paste.

### Per-phase QPC profiler is the localization tool

When perf is off and you don't know why, **add QPC checkpoints around every
phase** before guessing. The encode iter-4 win (3.2× speedup) was directly
enabled by the encode phase profiler (`d83ea21`) which localized 71% of the
encode wall to ONE call (`enc.d2h_final` at 238 ms).

Pattern: `g_phase_<name>_ns: i64 = 0` accumulators + QPC checkpoints
inline + `printAndResetPhaseProfile` gated on `SLZ_VK_PROFILE_PHASES=1`.
Decode side mirrors it (`g_phase_*_ns` in `decode/decode_dispatch.zig`).

### Zig std subprocess spawn defaults to EMPTY env

`std.Io.Threaded.InitOptions.environ` defaults to `.empty`. Spawned children
get NO PATH, NO SystemRoot, NO PROGRAMDATA. Vulkan loader can't find ICDs
without them. **Pass `.{ .environ = .{ .block = .global } }` when spawning
subprocesses that need registry-discovered hardware drivers.** Closed at
commit `e8fc91f`.

### glslc -O optimizer bug — Int8 capability not auto-declared

When the optimizer constant-folds `uint8_t(SLZ1_FRAME_MAGIC & 0xFFu)` it
emits `OpConstant %uchar` but fails to declare `OpCapability Int8`. Driver
validation rejects the module. **Patch SPIR-V binaries post-compile in
`encode/module_loader.zig::loadShaderModule`** — splice `OpCapability Int8`
into the capability block when any 8-bit storage cap is present without it.
Workaround documented at commit `23d8ab3`.

### Vulkan-on-WDDM has a per-submit floor that CUDA doesn't

Each `vkQueueSubmit + vkWaitForFences` round-trip is ~50-150 µs on Windows
WDDM. CUDA's synchronous DtoH calls batch submission cost across the
kernel-mode driver, hitting ~37 µs. **You cannot reduce per-call submit cost
in Vulkan; you can only reduce the number of submits.** Pattern: batch N
copies into one `vkCmdCopyBuffer` with N regions, or apply iter-7+11+15
(transfer queue + imported dst + single submit).

This is also why web.txt small-file workloads stay ~1.7-2.6× CUDA on both
decode + encode — the per-decode/per-encode overhead floor doesn't shrink
with input size. **Documented as accepted structural limit.** NCU + nsys +
deep-research-agent investigations all confirmed this is not fixable from
the app side.

### The iter-7+11+15 transferable pattern is GENUINELY transferable

This is the canonical fix for any per-call host-bounce pattern:
- **iter-7 D2H import:** `VK_EXT_external_memory_host` wraps the user's
  destination buffer as VkDeviceMemory; kernel writes directly into host RAM
- **iter-11 transfer queue:** dedicated VK_QUEUE_TRANSFER_BIT queue (DMA
  engine) + binary semaphore split-submit
- **iter-15 single submit:** consolidate phases into one submit + one wait
  per call by recording into the same cmdbuf

Applied to encode three times:
- **encode iter-3** (`7d37231`) — per-chunk D2H gather: 1526 submits → 1, **2× speedup**
- **encode iter-4** (`d9d8bba`) — final-frame D2H: 238 ms → 8 ms, **3.2× speedup**

Both REQUIRED the destination buffer to be page-aligned (procMallocHost). When
the host buffer comes from the caller (CLI), wrap it in a persistent
EncodeContext-owned pinned buffer + memcpy at the end (~10 ms cost vs ~200 ms
savings).

**The decoder uses the same pattern via iter-7/12 import for output/input
buffers.** Any L2+ kernel that emits a per-chunk host bounce should use this
pattern out of the gate.

### NCU is CUDA-only — use Nsight Graphics for Vulkan kernel profiling

Nsight Compute (NCU) 2025.4 has no Vulkan compute shader support. For
per-kernel SPIR-V metrics (occupancy, warp stalls, memory throughput),
use **Nsight Graphics → GPU Trace Profiler**. NCU is still useful for
profiling the CUDA reference side.

For pure GPU kernel timing on Vulkan, `VkQueryPool` timestamps inside our
own bench give per-kernel durations apples-to-apples with CUDA's bench.

### ptest_vk parallelism — three race classes caught

1. Decode/encode init races (lock with `g_init_lock` / `g_encode_init_lock`)
2. Encode init latching (use `defer if (ok)` for the `initialized=true` flip)
3. Cross-queue submit races on shared compute queue (lock via `g_dispatcher_lock`)

Plus iter-14's per-(StreamEntry × pipeline) descriptor sets to avoid races
on shared descriptor state.

### The H2 bisect lesson

Encode iter-1 H2 broke compression silently. We caught it only by careful
manual comparison of VK output size vs CUDA output size after a user
pushback ("are you sure L2 uses Huffman?"). The bisect via `git worktree`
isolated the regression cleanly. **Three takeaways:**

1. Bisect with worktrees, not destructive history rewrites
2. Don't trust a "passes all tests" claim if the tests don't check the
   property you care about (compression ratio vs file roundtrip)
3. The CLAUDE.md `NEVER REVERT WITHOUT PERMISSION` rule is right — we
   preserved the H2 + iter-2 commits in history, did clean reverts (not
   `git reset`), and could re-attempt the work on cleaner foundations

---

## File map

### Core paths

- `decode/streamlz_decoder.zig` — `decompressFramed` + CPU `buildChunkDescriptors`
- `decode/decode_dispatch.zig` — `fullGpuLaunchImpl` orchestrator + L2 gate at line 724 + per-phase QPC
- `decode/decode_context.zig` — `DecodeContext`, `ensureDeviceBuf` (grow-only), profiling
- `decode/scan_gpu.zig` — `gpuPrefixSumChunksImpl` (L1) + L2 stubs
- `decode/module_loader.zig` — VkDevice init, transfer queue, KERNEL_DECLS, procs impls, LRU import cache, streamEndAndWait, persistent_desc_sets, **persistent VkPipelineCache** (iter-cache), **procD2HOffsetGather** (iter-3)
- `decode/vulkan_api.zig` — procs surface (26 slots)
- `decode/driver.zig` — facade
- `encode/fast_framed.zig` — `compressFramedOne` + persistent gpu_out_buf + persistent d2h_final_buf + **per-phase encode QPC accumulators** (profiler)
- `encode/encode_lz.zig` — `gpuCompressImpl` + per-chunk D2H gather call (iter-3)
- `encode/encode_assemble.zig` — frame assembly passes
- `encode/encode_context.zig` — `EncodeContext` + `gpu_out_buf` (iter-3) + `d2h_final_buf` (iter-4)
- `encode/module_loader.zig` — encode VkDevice init + shader module loader + Int8 SPIR-V patcher + uses shared VkPipelineCache via `pipelineCacheHandle()`
- `encode/vulkan_ffi.zig` — encode-side procs

### CLI

- `cli/bench_decompress.zig` — `-db` (post-`d5adeb8` pure-kernel metric)
- `cli/bench_compress.zig` — `-b` (encode + roundtrip) + encode phase profiler wiring
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

### Bench (warm)

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"   # 1 = NVIDIA, 0 = Intel iGPU
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz   # decode bench
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt              # encode + roundtrip
```

Both bench paths pre-warm via an untimed first call, then measure. Reported
times EXCLUDE pipeline compile + Vulkan setup. "gpu kernel best/mean" is
sum of per-kernel VkQueryPool timestamps — apples-to-apples with CUDA.

### Per-phase host overhead (decode + encode both supported now)

```powershell
$env:SLZ_VK_PROFILE_PHASES = "1"
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz 2>&1 | Select-String "phase:"
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt 2>&1 | Select-String "phase:|Compress"
```

Decode phases in `decode/decode_dispatch.zig` (`g_phase_*_ns` accumulators).
Encode phases in `encode/fast_framed.zig` + `encode_lz.zig` + `encode_assemble.zig`
(`g_enc_phase_*_ns`).

### Cold-start measurement

```powershell
# Delete cache first
Remove-Item -Force $env:LOCALAPPDATA/streamlz_vk/pipeline_cache.bin -ErrorAction SilentlyContinue
Measure-Command { ./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/cold.slz assets/enwik8.txt 2>$null | Out-Null }
# Cache is now populated; second run is faster
Measure-Command { ./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/cold2.slz assets/enwik8.txt 2>$null | Out-Null }
```

### ptest_vk (after any srcVK change)

```powershell
Remove-Item -Force zig-out/bin/streamlz_vk.exe
zig build streamlz_vk -Doptimize=ReleaseFast
$env:SLZ_VK_DEVICE_INDEX = "1"
zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
$env:SLZ_VK_DEVICE_INDEX = "0"
zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
```

Expect 74/0/0 on both backends.

### SHA byte-identical to CUDA (MANDATORY for any encode change)

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"
foreach ($a in @("web.txt", "enwik8.txt", "silesia_all.tar")) {
  ./zig-out/bin/streamlz_vk.exe -c -l 1 -o "c:/tmp/vk_$a.slz" "assets/$a" 2>$null | Out-Null
  ./zig-out/bin/streamlz.exe -c -l 1 -o "c:/tmp/cu_$a.slz" "assets/$a" 2>$null | Out-Null
  $vk = (Get-FileHash -Algorithm SHA256 "c:/tmp/vk_$a.slz").Hash
  $cu = (Get-FileHash -Algorithm SHA256 "c:/tmp/cu_$a.slz").Hash
  Write-Host "$a MATCH=$($vk -eq $cu)"
}
```

Expected SHAs at HEAD `d9d8bba`:
- web: `AAA7F03B...`
- enwik8: `EE9471B7...`
- silesia: `D2970BF5...`

If CUDA binary is missing: rebuild via `tools\build_gpu.bat` (requires nvcc).

### Validation layer

See "Lessons learned" section above. Expect 0 VUIDs + 0 SYNC hazards.

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
relative shape; verify magnitudes via VkQueryPool / QPC / phase profiler.

For per-kernel SPIR-V metrics (occupancy, warp stalls, etc.): use **Nsight
Graphics → GPU Trace Profiler** (separate tool from NCU). NCU 2025.4 is
CUDA-only.

---

## Known accepted residuals (NOT blocking)

1. **web.txt small-file regime** — decode 2.6×, encode 1.73× CUDA. Structural
   Vulkan-on-WDDM submit floor. Four-iteration negative-result chain
   (iter-13/14/15 on decode + investigations on encode) confirmed not
   fixable from app side.

2. **Pre-existing single-thread mode `TestExpectedEqual` (2 tests)** —
   `SLZ_VK_TEST_THREADS=1` surfaces test ordering sensitivity that predates
   recent encode work. Parallel mode (the spec'd config) is rock-solid at
   74/0/0. Documented but unfixed.

3. **One-shot CLI shows 1 OOM warning per direction per process** — best-
   practices layer flags first `vkAllocateMemory` failure for non-pinned
   host buffers. Sticky disable flips on first OOM (commit `0af24ff`); bench
   mode is silent (pinned buffers).

---

## Recent commit history (last 20)

```
d9d8bba  encode iter-4 — final-frame D2H import path (3.2x encode speedup)
d83ea21  encode per-phase QPC profile — locates 238 ms d2h_final as the 190 ms gap
02560d6  persistent VkPipelineCache — disk-backed across process launches
7d37231  encode iter-3 — D2H gather (sync transfer-queue): 2.0x speedup
81bde77  Revert "encode iter-1 H2 — stream-routing + D2H cache hazard fix"
7d368fa  Revert "encode iter-2 — D2H gather collapses per-chunk loop"
d4f1f41  replace finishingL1.md with ToDo.md — L1 done, L2-L5 roadmap
0af24ff  OOM noise cleanup — per-direction sticky disable
e8fc91f  tests: inherit parent env in subprocess test harnesses (74/0/0)
23d8ab3  valfix C — Int8 capability glslc workaround
7960b98  valfix B — cmdbuf state via deferred staging destroy
9022dde  valfix A — handle types on imported VkBuffer
6ca302c  fix two Vulkan constants — subgroup pin now actually pins
332a04c  pin compute pipelines to subgroupSize=32 + device-pick guard
5e7a480  (reverted) encode iter-2 — D2H gather collapses per-chunk loop
2d68b2c  (reverted) encode iter-1 H2 — stream-routing + D2H import cache hazard fix
bfe746a  encode iter-1 H3 — verify iter-12 import auto-fires on encode H2D
bb675ab  encode iter-1 H1 — per-chunk D2H with srcOffset (2.2-2.6x faster)
7bc951e  decode iter-15 — single submit + single wait per decode
647bb95  decode iter 12 — H2D zero-copy import on transfer queue (DECODE DONE BAR HIT)
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
- `feedback-l1-completion-bar` — SUPERSEDED, current done-bar is exceeded on both decode + encode

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
   ./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz   # decode
   ./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt              # encode
   ```
5. **Verify SHA byte-identical to CUDA on all 3 goldens** — see "How to measure" above.
6. **Verify validation layer still clean** — see "Lessons learned" above.
7. **If working on encode changes: ALWAYS re-run the SHA check after.**
8. **Pick a phase from the L2-L5 roadmap above** and launch.
