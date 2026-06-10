# srcVK Operations Handbook

The operational knowledge for working on the srcVK Vulkan backend:
invariants, env-var contracts, hard-won lessons, and measurement
recipes. Distilled 2026-06-10 from the retired port-era docs
(`ToDo.md`, `gameplan.md`, `PortInstructions.md` — see git history)
when the port completed. Companions:

- `PortAdaptations.md` — THE canonical CUDA↔VK divergence catalog
  (A-NNN entries). Read it before introducing any new divergence;
  add an entry when you do.
- `audit.md` — canonical naming law: extension translation
  (`.cu`→`.comp`, `.cuh`→`.glsl`), the no-CUDA-token rule, and the
  per-file CUDA↔VK mapping. Any NEW srcVK file gets its name from
  these tables.
- `PerfSweep.md` — the L1-L5 × corpus × backend perf matrix.
- `/v4_ideas.md` — forward-looking work (includes A-021 fusion).
- `/CLAUDE.md` — project rules, including the cross-backend SHA gate.

Port status: COMPLETE. enwik9 1 GB L1-L5 SHA-256 clean (decode of
CUDA frames AND self-encode byte-identical); enwik8 byte-identical
roundtrips on NVIDIA RTX 4060 Ti and Intel(R) Graphics; ptest_vk
149 passed / 9 env-guarded skips (subprocess discrete-GPU probe) /
0 failed on both devices.

---

## Env vars + CLI contract

- `SLZ_VK_DEVICE_INDEX=N` — device override. Default selection
  prefers DISCRETE_GPU > INTEGRATED_GPU > VIRTUAL_GPU > CPU > OTHER;
  on this machine 1 = NVIDIA, 0 = Intel iGPU. The CLI also accepts
  `--device N|<name-substring>`.
- The CLI MUST print `Device: <vkPhysicalDeviceProperties.deviceName>`
  on every invocation (project memory: perf numbers without a device
  name are invalid — the Intel-iGPU default-selection saga).
- `SLZ_VK_PROFILE_DECODE=1` — per-kernel `kper:` lines (VkQueryPool
  timestamps) + per-phase QPC timing + import-path telemetry.
- `SLZ_VK_D2D=1` — reroutes `-db` through the TRUE-D2D entry
  (`decompressFramedFromDevice`, the `slzDecompressAsync` path): frame
  staged to a device buffer once, timed loop is pure device-resident
  decode, output byte-verified against the host-path result. The VK
  analog of CUDA's tools/bench_d2d.bat.
- `SLZ_VK_PROFILE_PHASES=1` — per-phase QPC accumulators on encode
  (`g_enc_phase_*_ns`) and decode (`g_phase_*_ns`); prints `phase:`
  lines. This profiler located the 238 ms d2h_final bottleneck behind
  the 3.2× encode win.
- Bench output format matches CUDA's `streamlz` exactly for
  apples-to-apples comparison. Do not add flags CUDA's CLI doesn't
  have (`--device`/`--probe` are the sanctioned VK-only exceptions).

---

## Architecture invariants — DO NOT VIOLATE

### procs.* surface (decode/vulkan_api.zig)

The codec calls `procs.h2d(dst, src, size)` etc. — same names, args,
semantics as the CUDA side. Vulkan plumbing (vkCmdCopyBuffer, VMA,
fences, queues) lives UNDER procs. Adding new slots is OK when the
slot expresses a Vulkan-native concept CUDA has no direct analog for.
27 slots at last count; `d2h_offset_gather` is the highest-leverage
adaptation (kills the per-call submit floor for per-chunk D2Hs and
the final output D2H).

### Function decomposition (CUDA-verbatim)

Do NOT rename or restructure:
- `decode/decode_dispatch.zig::fullGpuLaunchImpl` +
  `uploadInputAndPrefixSum` + `runBackHalf` + `finalizeOutput` +
  `runLzPipeline` + `runHuffBuildAndDecode` / `mergeHuffDescs` /
  `gatherRawOff16`
- `encode/fast_framed.zig::compressFramedOne` +
  `encode/encode_lz.zig::gpuCompressImpl` +
  `encode/encode_assemble.zig::gpuAssembleFrameImpl` /
  `gpuFrameAssembleImpl`

### `buildChunkDescriptors` MUST stay a CPU walk on the host-input path

In `decode/streamlz_decoder.zig`. The deleted `/src_vulkan/` attempt
moved this to GPU for the host-input path — THE canonical port
violation. CUDA has two entry points: host-input uses the CPU walk;
the D2D entry (`decompressFramedFromDevice`) uses the GPU
walk_frame_kernel because the input is device-resident. Keep the
split; never merge them.

### KERNEL_DECLS (decode + encode module_loader.zig)

Pipeline binding counts + push-constant sizes per kernel. Do NOT
change without auditing every call site's `params[]` layout. New
kernels need a KERNEL_DECLS entry with correct
`pin_subgroup_32: bool` (true unless `local_size_x == 1`).

### Subgroup size pinned to 32 (gpu_warp.glsl contract)

Decode + encode pipeline creation pin `requiredSubgroupSize=32` +
`REQUIRE_FULL_SUBGROUPS_BIT`; the device-pick guard rejects devices
that can't satisfy it (with a message naming the supported hardware;
`--probe` prints each device's subgroup range). Kernels hardcode
`WARP_SIZE=32u`. NVIDIA only offers 32; Intel UHD/Iris/Arc satisfies
the pin; AMD only on RDNA1+ (RX 5000+, wave32-capable) — wave64-only
GCN (Polaris/Vega, e.g. RX 400/500 series) reports [64,64] and can
never run these kernels (verified on RX 590/RX 550, 2026-06-10).

### Persistent buffers on EncodeContext

`gpu_out_buf` (per-chunk D2H gather dst) and `d2h_final_buf`
(final-frame D2H dst) are page-aligned via
`gpu_dec_driver.allocHost` → procMallocHost so the LRU import cache
hits on every call after the first. Without page alignment the
import path refuses the buffer and the 2×/3.2× speedups vanish.

### Required device features

`bufferDeviceAddress`, `shaderInt8`, `storageBuffer8BitAccess`,
`uniformAndStorageBuffer8BitAccess`, `subgroupSizeControl`,
`computeFullSubgroups` — enabled at VkDevice creation; absence →
`error.BackendNotAvailable` (codec falls back cleanly).

### Persistent disk-backed VkPipelineCache

All compute pipelines share a process-wide cache persisted at
`%LOCALAPPDATA%/streamlz_vk/pipeline_cache.bin` (load at init, save
at exit). ~38 ms cold-start saving; driver validates the header and
silently discards mismatches.

---

## Lessons learned

### Always run the validation layer before claiming "done"

Three real latent bugs were caught ONLY by validation (import-path
buffer-create chain, cmdbuf invalidated by destroyed buffer, glslc
Int8 capability bug):

```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:VK_LAYER_SETTINGS_PATH = "c:/tmp/vk_layer_settings.txt"
./zig-out/bin/streamlz_vk.exe -c -l 1 -o c:/tmp/v.slz assets/web.txt 2>&1 | Select-String "VUID|Validation Error"
```

`vk_layer_settings.txt`:
```
khronos_validation.enables = VK_VALIDATION_FEATURE_ENABLE_BEST_PRACTICES_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
khronos_validation.report_flags = error,warn,perf,info
```

Expect 0 VUIDs + 0 sync hazards. (CUDA analog: `tools/sanitize.bat`.)

### SHA byte-identity to CUDA is MANDATORY — roundtrip-passes is INSUFFICIENT

The H2 disaster: an encode iteration broke compression on ≥8 MiB
inputs, produced essentially-raw output that VALID-roundtripped, and
passed every test. Caught only by comparing output SIZE to CUDA.
Every encode change must verify SHA256(VK frame) == SHA256(CUDA
frame) on the goldens — now a standing rule in /CLAUDE.md. Three
takeaways from the bisect: use worktrees (not history rewrites);
don't trust "passes all tests" when the tests don't check the
property at stake; clean reverts preserve the history for
re-attempts.

### NEVER hand-type Vulkan constants

A 4-line constant correction once fixed an entire broken-pin commit —
two enum values typed from memory, both wrong. Pull constants from
`C:/VulkanSDK/<ver>/Include/vulkan/vulkan_core.h`: grep, copy, paste.

### Per-phase QPC profiling is the localization tool

When perf is off, add QPC checkpoints around every phase BEFORE
guessing (`SLZ_VK_PROFILE_PHASES=1`). The 3.2× encode win came
directly from the profiler localizing 71% of the wall to one call.

### Zig std subprocess spawn defaults to an EMPTY environment

`std.Io.Threaded.InitOptions.environ` defaults to `.empty` — children
get no PATH/SystemRoot, and the Vulkan loader can't find ICDs. Pass
`.{ .environ = .{ .block = .global } }` for any subprocess needing
registry-discovered drivers.

### glslc -O optimizer bug — Int8 capability not auto-declared

Constant-folding `uint8_t(...)` emits `OpConstant %uchar` without
declaring `OpCapability Int8`; drivers reject the module.
`encode/module_loader.zig::loadShaderModule` patches the SPIR-V
post-compile (splices the capability when 8-bit storage caps are
present without it).

### Vulkan-on-WDDM has a per-submit floor CUDA doesn't

Each vkQueueSubmit+wait round-trip costs ~50-150 µs on WDDM (CUDA
sync copies: ~37 µs). You cannot reduce per-submit cost — only the
NUMBER of submits. The canonical fix (applied three times, 2-3.2×
each): import the destination as VkDeviceMemory
(`VK_EXT_external_memory_host`), use the dedicated transfer queue,
and consolidate to one submit per call. This is also why small-file
(web.txt) workloads sit at 1.7-2.6× CUDA — accepted structural
limit, confirmed unfixable from the app side.

### NCU is CUDA-only

For per-kernel SPIR-V metrics use Nsight Graphics → GPU Trace
Profiler. For plain kernel timing, the in-tree VkQueryPool bench
(`SLZ_VK_PROFILE_DECODE=1`) is apples-to-apples with CUDA's bench.
nsys's VK trace layer inflates absolute timings — trust call counts
and shape, verify magnitudes via QueryPool/QPC.

### ptest_vk parallelism — three race classes

(1) init races → `g_init_lock`/`g_encode_init_lock`; (2) init
latching → `defer if (ok)` for the `initialized=true` flip; (3)
shared-queue submit races → `g_dispatcher_lock`; plus
per-(stream × pipeline) descriptor sets. The CUDA suite later hit a
fourth class VK can't have: per-thread context currency (cuCtxSetCurrent)
— Vulkan handles are thread-agnostic, so only mutual exclusion matters
here.

---

## How to measure

```powershell
$env:SLZ_VK_DEVICE_INDEX = "1"   # 1 = NVIDIA, 0 = Intel iGPU
./zig-out/bin/streamlz_vk.exe -db -r 5 tests/goldens/enwik8.txt.L1.slz   # decode bench
./zig-out/bin/streamlz_vk.exe -b -r 5 -l 1 assets/enwik8.txt              # encode + roundtrip
```

Benches pre-warm untimed, then measure; "gpu kernel best/mean" sums
VkQueryPool timestamps. Run GPU benches SERIALLY — never in parallel
with each other or with CUDA benches (project memory).

ptest after any srcVK change, BOTH devices:

```powershell
zig build streamlz_vk -Doptimize=ReleaseFast
$env:SLZ_VK_DEVICE_INDEX = "1"; zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
$env:SLZ_VK_DEVICE_INDEX = "0"; zig build ptest_vk -Doptimize=ReleaseFast 2>&1 | Select-String "passed|failed|skipped"
```

Expect 149 passed / 9 skipped / 0 failed on both (the skips are the
subprocess tests' discrete-GPU-visibility guard, not silent
failures). The build graph tracks `.glsl` includes via
glslc -MD depfiles (A-012 resolved) — plain `zig build` invalidates
SPIR-V correctly; `tools/build_vk.bat` is only a force-clean utility.

Cross-backend SHA gate (per /CLAUDE.md, after any encoder change):
encode the goldens on both backends, `Get-FileHash` each, compare.
Regenerate expected hashes from HEAD — historical SHAs in old docs
predate the sc=0.25 default change.

---

## Accepted residuals (NOT bugs)

1. **web.txt small-file regime** — decode ~2.6×, encode ~1.7× CUDA.
   WDDM submit floor (see lesson above); confirmed structural.
2. **L3+ decode kernel-time 1.2-1.37× CUDA on large workloads** —
   A-021 (unfused gather/merge dispatches vs CUDA). e2e stays inside
   the 10% bar because host overhead amortizes it. Close path is
   A-017-style fusion; tracked in /v4_ideas.md #10. (CUDA-side
   equivalents were fused 2026-06-10: slzCompactAllDescsKernel +
   slzMergeHuffDescsParKernel.)
3. **Single-thread test mode (`SLZ_VK_TEST_THREADS=1`) has 2
   order-sensitive failures** — predates the encode work; the spec'd
   parallel mode is solid.
4. **One OOM warning per direction per one-shot CLI process** —
   best-practices layer flags the first non-pinned vkAllocateMemory;
   sticky-disabled after first hit, silent in bench mode.

(The historical "silesia L3 0.02% larger" residual was CLOSED by the
A-008 BDA hash-table workaround at `8c8964d`.)
