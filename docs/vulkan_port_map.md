# Vulkan Port — Codebase Map

This document is the consolidated, evidence-backed map of the CUDA codebase as it stands in `src/`, written specifically as the primary input for the upcoming design and execution workflows of the parallel Vulkan port in `src_vulkan/`. Every claim cites `file:line` or quotes source. Where the surveys disagreed, the deep-dive findings have been used to lock the answer.

---

## 0. Locked constraints

1. **Shader language.** GLSL authored by hand, compiled to SPIR-V by `glslc` (Vulkan SDK).
2. **Vendor coverage.** NVIDIA, AMD (RDNA1/2/3, GCN/CDNA), Intel discrete (Arc/Xe), mobile (Adreno, Mali). Either **wave-size agnostic** OR force `subgroupSize=32` via `VK_EXT_subgroup_size_control` where supported; a documented fallback is required otherwise.
3. **Sibling build.** `streamlz_vk.exe` + `streamlz_vk.dll`, all C symbols `_vk`-suffixed. The existing CUDA `src/` stays as-is; `share/` and `format/` are shared verbatim.
4. **API parity day-1.** `slzCompress*`, `slzDecompress*`, `slzCompressAsync`, `slzDecompressAsync`, device-only-source-pointer path, per-kernel timings.
5. **Wire-format byte-identity.** The Vulkan encoder must emit byte-identical `.slz` to the CUDA encoder. The Vulkan decoder must accept any CUDA-produced frame.

Throughout this document, every porting concern is evaluated against these five constraints.

---

## 1. CUDA codebase at a glance

### 1.1 Tree shape (relevant subset)

```
src/
├── common/                  # shared device-side primitives (header-only)
│   ├── gpu_byteio.cuh       # endian-aware byte IO (memcpy-based unaligned)
│   ├── gpu_huffman.cuh      # canonical-Huffman wire constants + LUT pack/unpack
│   ├── gpu_warp.cuh         # WARP_SIZE=32, FULL_WARP_MASK, lastBitSet, warpCopy
│   └── gpu_wire_format.cuh  # LZ wire-format constants + sub-chunk accessors
├── encode/
│   ├── lz_kernel.cu              # slzLzEncodeKernel
│   ├── huffman_kernel.cu         # slzHuffBuildTablesKernel + slzHuffEncode4StreamKernel
│   ├── assemble_kernel.cu        # slzAssembleMeasureKernel/WriteKernel/FrameAssembleKernel
│   ├── lz_greedy_parser.cuh      # warp-parallel greedy parser (L1-L4)
│   ├── lz_chain_parser.cuh       # lane-0 chain parser (L5)
│   ├── lz_format.cuh             # hash helpers, hashKey6/hashTableA/hashTableB
│   ├── *.zig                     # host orchestration (driver, contexts, launchers)
│   └── *.ptx                     # committed PTX blobs (embedded via @embedFile)
├── decode/
│   ├── lz_kernel.cu              # slzLzDecodeKernel + slzLzDecodeRawKernel + 5 more
│   ├── huffman_kernel.cu         # slzHuffBuildLutKernel + slzHuffDecode4StreamKernel
│   ├── *.cuh                     # walk_frame, scan_parse, compact_descs, merge_huff_descs,
│   │                             #   prefix_sum_chunks, gather_raw_off16, lz_decode_*
│   ├── *.zig                     # driver, decode_context, decode_dispatch, scan_gpu
│   └── *.ptx                     # committed PTX blobs
├── format/                  # host-side wire-format helpers (shared with Vulkan port)
│   ├── frame_format.zig
│   ├── block_header.zig
│   └── streamlz_constants.zig
└── streamlz_gpu.zig         # C-ABI implementation (export fn shims)

include/
└── streamlz_gpu.h           # public C ABI (14 exports, slzStatus_t enum)

tools/
├── build_gpu.bat            # nvcc -arch=sm_89 -O3 → .cubin (res-usage), .ptx
└── build_*.bat              # bench harnesses, C ABI consumers
```

### 1.2 Key counts

| Category | Count | Notes |
|----------|------:|-------|
| `__global__` kernels | **17** | 5 encode-side + 12 decode-side; verified by deep-dive |
| Public C symbols | **14** | enumerated in §6 |
| Status codes | **8** | `SLZ_SUCCESS` + 7 error codes (`include/streamlz_gpu.h:80-89`) |
| Subgroup-intrinsic families used | **11** | enumerated in §3 |
| `_v2` driver-API symbols loaded | **30** (decode) + **10** (encode, subset) | §4 |
| PTX modules | **3** (encode) + **2** (decode) | `lz_kernel.ptx`, `huffman_kernel.ptx`, `assemble_kernel.ptx` (encode-only) |
| Test blocks | **22** | 5 GPU-roundtrip + 10 frame-format + 7 block-header |
| Build artifacts | **2** | `streamlz.exe` (CLI), `streamlz_gpu.dll` (C ABI) |

---

## 2. Kernel inventory

### 2.1 Encode kernels

| # | Symbol | Defined at | Launched at | Grid | Block | Shared | Subgroup ops | Atomics | Vulkan port risk |
|---|--------|-----------|-------------|------|-------|--------|--------------|---------|------------------|
| E1 | `slzLzEncodeKernel` | `src/encode/lz_kernel.cu:64` | `src/encode/encode_lz.zig:127` | `(num_chunks,1,1)` | `(32,1,1)` `__launch_bounds__(32,1)` | 0 | `__shfl_down_sync` ×7, `__shfl_sync` ×3, `__ballot_sync` ×3, `__match_any_sync` ×2, `__ffs` ×4, `__clz` ×1, `__syncwarp` ×3 (greedy path); none (chain path, `lane!=0` early-return) | 0 | **HIGH** — `__match_any_sync` hot loop |
| E2 | `slzHuffBuildTablesKernel` | `src/encode/huffman_kernel.cu:111` | `src/encode/encode_huff.zig:153` | `(n_descs,1,1)` | `(32,1,1)` | 9,472 B (hist 1KB + weights 2KB + parents 2KB + active 2KB + used_symbols 1KB + code_lengths 256B + codes 1KB) | none in build proper; histogram uses `atomicAdd` on shared `hist[]` | 1× shared-mem `atomicAdd(hist[symbol], 1)` (line 137) | **LOW** — atomic order does NOT affect output (deep-dive: histogram counts +1 are commutative; downstream pipeline is deterministic on hist[]) |
| E3 | `slzHuffEncode4StreamKernel` | `src/encode/huffman_kernel.cu:246` | `src/encode/encode_huff.zig:171` | `(n_descs,1,1)` | `(32,1,1)` | 0 | `__shfl_xor_sync` ×2 (min reduction K; sum reduction total_tail), `__shfl_up_sync` ×1 (exclusive prefix-sum offsets), `__syncwarp` ×3 | 0 | **HIGH** — BIL 4-byte aligned scalar stores, 32-lane lockstep |
| E4 | `slzAssembleMeasureKernel` | `src/encode/assemble_kernel.cu:317` | `src/encode/encode_assemble.zig:118` | `(n_subchunks,1,1)` | `(32,1,1)` | 0 | `__syncwarp` ×11 (lane-0 scalar writes + warpCopy join) | 0 | **LOW** |
| E5 | `slzAssembleWriteKernel` | `src/encode/assemble_kernel.cu:338` | `src/encode/encode_assemble.zig:152` | `(n_subchunks,1,1)` | `(32,1,1)` | 0 | `__syncwarp` (same pattern as E4) | 0 | **LOW** |
| E6 | `slzFrameAssembleKernel` | `src/encode/assemble_kernel.cu:391` | `src/encode/encode_assemble.zig:300` | `(n_chunks+1,1,1)` | `(128,1,1)` | 0 | None — pure strided byte-copy `for (i = lane; i < N; i += bdim)` | 0 | **LOW** — port this first |

### 2.2 Decode kernels

| # | Symbol | Defined at | Launched at | Grid | Block | Shared | Subgroup ops | Atomics | Vulkan port risk |
|---|--------|-----------|-------------|------|-------|--------|--------------|---------|------------------|
| D1 | `slzWalkFrameKernel` | `src/decode/walk_frame_kernel.cuh:50` | `src/decode/scan_gpu.zig:75` | `(1,1,1)` | `(1,1,1)` | 0 | None — `SLZ_GUARD_SINGLE_THREAD` early-return | 0 | **LOW** (but 31 lanes wasted) |
| D2 | `slzPrefixSumChunksKernel` | `src/decode/prefix_sum_chunks_kernel.cuh:17` | `src/decode/scan_gpu.zig:122` | `(1,1,1)` | `(1,1,1)` | 0 | None | 0 | **LOW** |
| D3 | `slzScanParseKernel` | `src/decode/scan_parse_kernel.cuh:108` | `src/decode/scan_gpu.zig:185` | `(ceil(n/256),1,1)` | `(256,1,1)` | 0 | None — one thread per chunk, sequential per-thread sub-chunk walk; **no `__syncthreads`, no warp intrinsics** | 0 | **LOW** |
| D4 | `slzCompactHuffDescsKernel` | `src/decode/compact_descs_kernels.cuh:39` | `src/decode/scan_gpu.zig:237` | `(1,1,1)` | `(1,1,1)` | 0 | None — `SLZ_GUARD_SINGLE_THREAD` | 0 | **LOW**; launched 4× in host loop (one per lit/tok/hi/lo plane) |
| D5 | `slzCompactRawDescsKernel` | `src/decode/compact_descs_kernels.cuh:62` | `src/decode/scan_gpu.zig:253` | `(1,1,1)` | `(1,1,1)` | 0 | None | 0 | **LOW** |
| D6 | `slzGatherRawOff16Kernel` | `src/decode/gather_raw_off16_kernel.cuh:23` | `src/decode/decode_dispatch.zig:177` | `(scan.num_raw_off16,1,1)` (self-gates on `*d_count`) | `(256,1,1)` | 0 | None — strided memcpy across `threadIdx.x` lanes | 0 | **LOW** |
| D7 | `slzMergeHuffDescsKernel` | `src/decode/merge_huff_descs_kernel.cuh:41` | `src/decode/decode_dispatch.zig:134` | `(1,1,1)` | `(1,1,1)` | 0 | None — `SLZ_GUARD_SINGLE_THREAD` | 0 | **LOW** |
| D8 | `slzHuffBuildLutKernel` | `src/decode/huffman_kernel.cu:155` | `src/decode/decode_dispatch.zig:329` | `(n_huff,1,1)` (self-gates on `*p_n`) | `(32,1,1)` | 5,532 B static (code_lengths 256B + used_pkd 1KB + len_end 52B + length_count 52B + per_L_base 52B + lut 4KB) | `__match_any_sync` ×1 (L collision, domain `[1,11]`), `__popc` ×2, `__ffs` ×1, `__syncwarp` ×9 | 2× shared-mem `atomicAdd` (line 204 `length_count[L]++` heavily contended; line 272 `per_L_base[L]+=__popc(mask)` leader-only) | **MEDIUM** — `__match_any_sync` emulation, but value domain tiny (12 values) so 11-ballot loop is trivial; produces LUT that is runtime-only (not wire) |
| D9 | `slzHuffDecode4StreamKernel` | `src/decode/huffman_kernel.cu:631` | `src/decode/decode_dispatch.zig:354` | `(n_huff,1,1)` (self-gates) | `(32,1,1)` `__launch_bounds__(32,8)` | 4,096 B dynamic (`HUFF_LUT_ENTRIES * 4`) | `__shfl_up_sync` ×1 (tail-bytes prefix-sum), `__syncwarp` ×1 | 0 | **HIGH** — 32-stream BIL is wire-locked (`HUFF_NUM_STREAMS=32` `static_assert`); uses `uint64_t` bit buffer; `__byte_perm` ×4 (LE→BE refill) |
| D10 | `slzLzDecodeKernel` | `src/decode/lz_decode_kernels.cuh:31` | `src/decode/decode_dispatch.zig:451` | `(lz_grid_x=(lz_groups+1)/2,1,1)` | `(32,2,1)` `__launch_bounds__(64,24)` | 0 | `__shfl_sync` ×many (lane-0 broadcasts of running parser state — `cmd_pos/lit_pos/off16_pos/off32_pos/recent_offset/length_offset/...`), `__syncwarp` ×many at warp-uniform scope | 0 | **HIGH** — 2 warps per block: deep-dive confirms they share NO state, can be split into two `(32,1,1)` workgroups with doubled grid; subgroup-size 32 must be guaranteed |
| D11 | `slzLzDecodeRawKernel` | `src/decode/lz_decode_kernels.cuh:158` | `src/decode/decode_dispatch.zig:430` | `(lz_grid_x,1,1)` | `(32,2,1)` `__launch_bounds__(64,24)` | 0 | `__ballot_sync` ×2 (any-long, fresh-offset), `__shfl_sync` ×many, `__shfl_up_sync` ×1 (warpScanU32), `__syncwarp` | 0 | **HIGH** — same as D10 plus parallel-parse fast path |

### 2.3 Common headers — every device helper

| Header | Helpers | Subgroup-coupled? |
|--------|---------|-------------------|
| `src/common/gpu_byteio.cuh` | `readBE24/writeBE24`, `readU32BE`, `readLE24/writeLE24`, `readU16LE/storeU16LE`, `readU32LE/writeU32LE`, `readU64LE`, `read8safe` | **No** — pure scalar memcpy / byte-shifts |
| `src/common/gpu_huffman.cuh` | Constants `HUFF_NUM_STREAMS=32`, `HUFF_MAX_CODE_LEN=11`, `HUFF_LUT_ENTRIES=1024`, `HUFF_BIL_ROW_BYTES=128`, `HUFF_BODY_HEADER_BYTES=228`; `packWeightByte/unpackWeightByte`; `buildCanonicalCodes` (lane-0 serial RFC-1951); LUT pack/unpack | **No** code; constants drive 32-stream BIL wire format |
| `src/common/gpu_warp.cuh` | `WARP_SIZE=32`, `LANE_MASK=31`, `FULL_WARP_MASK=0xFFFFFFFF`, `BITS_PER_BYTE=8`, `U32_BITS=32`, `U64_BITS=64`; `SLZ_GUARD_SINGLE_THREAD` macro; `lastBitSet` (`__clz`); `warpCopy` (strided byte copy) | **Yes** — `__clz` in `lastBitSet`; comment at lines 22-24 explicitly states every kernel uses `FULL_WARP_MASK` |
| `src/common/gpu_wire_format.cuh` | LZ wire constants, sub-chunk header accessors (`subchunkIsLz/Mode/CompSize`), frame magic/version, block/chunk flags | **No** — pure constants and scalar bit ops |

---

## 3. Warp / subgroup intrinsics inventory

| Intrinsic | Call sites | GLSL equivalent | Required Vulkan extension | Portability notes |
|-----------|-----------:|-----------------|---------------------------|-------------------|
| `__shfl_sync(mask, v, src_lane)` | **56** | `subgroupBroadcast(v, id)` when `id` is dynamically uniform; `subgroupShuffle(v, id)` otherwise | `GL_KHR_shader_subgroup_basic` + `_shuffle` | **All** sites pass `FULL_WARP_MASK`. Only one site has runtime-derived `src_lane` (`__ffs(match_ballot)-1` at `lz_greedy_parser.cuh:197/205`) — but it's dynamically uniform, satisfying `subgroupBroadcast` |
| `__shfl_up_sync` | **3** | `subgroupShuffleUp(v, delta)` or `subgroupExclusiveAdd` | `GL_KHR_shader_subgroup_shuffle_relative` (or `_arithmetic`) | All 3 implement Hillis-Steele prefix-sum (`warpScanU32`, encoder Huff prefix-sum, decoder Huff prefix-sum) |
| `__shfl_xor_sync` | **2** | `subgroupShuffleXor` or `subgroupMin`/`subgroupAdd` | `GL_KHR_shader_subgroup_shuffle` / `_arithmetic` | Butterfly min (K) and sum (total_tail) in encode Huff |
| `__shfl_down_sync` | **7** | `subgroupShuffleDown(v, delta)` | `GL_KHR_shader_subgroup_shuffle_relative` | All 7 in `src/encode/lz_greedy_parser.cuh:43-49` — 7-lane look-ahead window |
| `__ballot_sync` | **5** | `subgroupBallot(pred)` → `uvec4` (use `.x` for size-32) | `GL_KHR_shader_subgroup_ballot` | Mask high bits if `subgroupSize > 32` |
| `__match_any_sync` | **3** | No direct equivalent | None | **The single biggest port concern.** Deep-dive resolved: see §3.1 |
| `__syncwarp()` | **45** | `subgroupBarrier() + subgroupMemoryBarrierShared()` | `GL_KHR_shader_subgroup_basic` | Every site is at warp-uniform control flow per deep-dive; safe to lower to `barrier()` if the workgroup is exactly one subgroup |
| `__popc(u32)` | **3** | `bitCount(uint)` | core GLSL 4.00 / SPIR-V `OpBitCount` | Scalar |
| `__clz(u32)` | **2** | `findMSB(uint)` with adjustment: `__clz(x) = 31 - findMSB(x)` for nonzero; `findMSB(0) = -1`, `__clz(0) = 32` — guard `x != 0` |  core GLSL | Encoder relies on `if (lower_same_bucket != 0)` guard already (`lz_greedy_parser.cuh:94`) |
| `__ffs(u32)` | **6** | `findLSB(uint) + 1` (mind 1-based) | core GLSL | Scalar |
| `__byte_perm(w, 0, 0x0123)` | **4** | Explicit bitswap: `((w<<24)) | ((w<<8)&0xFF0000) | ((w>>8)&0xFF00) | (w>>24)` | None | All 4 sites at `src/decode/huffman_kernel.cu:431,482,528,581` — LE→BE bit-stream refill |

**Not used anywhere:** `__activemask`, `__any_sync`, `__all_sync`, `cooperative_groups`, `cp.async`, `ldmatrix`, `mma.*`, `__funnelshift_*`, `__ldg`, `__stwt`, textures, surfaces, inline PTX (the only inline-asm in `*.ptx` is nvcc's lowering of `__stcs`). No 64-bit atomics. The single device-memory atomic is **absent** — all atomicAdds are on `__shared__`.

### 3.1 `__match_any_sync` — resolved emulation strategy

Three call sites, classified by deep-dive (`match-any-sync-callsites-and-emulation`):

| Site | File:line | Value | Domain | Use of result | Path | Emulation |
|------|-----------|-------|--------|---------------|------|-----------|
| #1 | `src/encode/lz_greedy_parser.cuh:82` | `h` (hash bucket) | `2^hash_bits` (131K–524K) | Full bitmask, AND'd with active_mask & `((1<<lane)-1)` then `__clz` → `lower_same_bucket` → `highest_lower` | **HOT** (every ~32 input bytes in L1–L4) | 32× `subgroupBroadcast(h, k)` compare-and-OR; OR specialized "highest-lower-lane-with-same-bucket" primitive |
| #2 | `src/encode/lz_greedy_parser.cuh:83` | `key4` (low 4 bytes of 8-byte key) | 2^32 (but ~32 distinct/warp) | **Only ONE bit at `highest_lower` is tested** (line 95) | **HOT** | **Eliminate**: replace `(key_same & (1<<highest_lower)) != 0` with `subgroupShuffle(key4, highest_lower) == key4` — strict simplification, byte-identical |
| #3 | `src/decode/huffman_kernel.cu:259` | `L` (code length) | **[1,11], 12 values** | Bitmask used by `__popc(mask & lt_mask)` and `__ffs(mask)` | **WARM** (8 batches × 1525 LUTs ≈ 0.67 ms total) | 11-iteration `subgroupBallot(L == k)` loop |

**Byte-identity rule:** sites #1 and #2 are byte-output-DETERMINING (the matched lane wins the LZ match, which encodes the offset). Site #3 builds a runtime-only LUT (decoder, never serialized). Emulations above produce bit-identical bitmasks (or bit-identical resolved answers, for site #2's simplification).

---

## 4. CUDA Driver API ↔ Vulkan mapping

The library loads `nvcuda.dll` via `LoadLibraryA` + `GetProcAddress` (twice: once in `src/decode/cuda_api.zig` resolving 30 entry points; once in `src/encode/cuda_ffi.zig` resolving a 10-symbol subset). Encode explicitly piggybacks on decode's CUDA context.

| CUDA symbol | Decode? | Encode? | Used at | Vulkan equivalent | Notes |
|-------------|:------:|:------:|---------|-------------------|-------|
| `cuInit` | ✓ | – | `src/decode/module_loader.zig:102` | `vkCreateInstance` | One-shot |
| `cuDeviceGet` | ✓ | – | `module_loader.zig:105` | `vkEnumeratePhysicalDevices` | |
| `cuDeviceGetAttribute(MULTIPROCESSOR_COUNT)` | ✓ | – | `module_loader.zig:111-116` | `VkPhysicalDeviceProperties.limits` (no SM analogue; use `maxComputeWorkGroupCount`/queue count for adaptive launch) | |
| `cuCtxCreate_v2` | ✓ | – | `module_loader.zig:131` | `vkCreateDevice` | Piggybacks on caller's context if present |
| `cuCtxGetCurrent` | ✓ | – | `module_loader.zig:122-125` | n/a — Vulkan has no current-context per thread | |
| `cuCtxSetCurrent` | ✓ | – | `decode_context.zig:92` (`bindContextToCallingThread`) | **no-op** | Per-thread `VkCommandPool` is the equivalent thread-locality model |
| `cuCtxSynchronize` | ✓ | ✓ | encode L142/L173/L153/L304; decode finalize | `vkQueueWaitIdle` (heavy); per-stream → `vkWaitForFences` or `vkWaitSemaphores` (timeline) | Heavy fence: replace with per-submission fence/semaphore |
| `cuModuleLoadData` | ✓ | ✓ | loads `@embedFile(".ptx") ++ "\x00"` | `vkCreateShaderModule(SPIR-V uint32[])` | SPIR-V is length-prefixed, not null-terminated |
| `cuModuleGetFunction` | ✓ | ✓ | resolve kernel by name | n/a — entry-point string is in `VkPipelineShaderStageCreateInfo.pName` | One `VkComputePipeline` per kernel (see §8) |
| `cuMemAlloc_v2` | ✓ | ✓ | `ensureBuf` / `ensureDeviceBuf` | `vkAllocateMemory` + `vkCreateBuffer` (or VMA) | **CRITICAL**: ~25 owned slots; max `maxMemoryAllocationCount` ≈ 4096 desktop / 256 mobile — use a suballocator |
| `cuMemFree_v2` | ✓ | ✓ | grow path + `deinit` | `vkDestroyBuffer` + `vkFreeMemory` | |
| `cuMemAllocHost_v2` | ✓ | – | `decode_context.zig:58` (dormant in lib; CLI bench uses it) | `vkAllocateMemory` HOST_VISIBLE\|HOST_COHERENT (\|HOST_CACHED if available) + `vkMapMemory` | |
| `cuMemFreeHost` | ✓ | – | `decode_context.zig:66` | `vkUnmapMemory` + `vkFreeMemory` | |
| `cuMemcpyHtoD_v2` | ✓ | ✓ | many sites (synchronous, pageable host) | `vkCmdCopyBuffer` from HOST_VISIBLE staging | Synchronous semantics → submit + fence wait |
| `cuMemcpyDtoH_v2` | ✓ | ✓ | many sites (synchronous, pageable host) | `vkCmdCopyBuffer` to HOST_VISIBLE staging | Same |
| `cuMemcpyDtoDAsync_v2` | ✓ | ✓ | D2D paths (3 decode sites; 1 encode site) | `vkCmdCopyBuffer` (intra-cmd-buffer) | Followed by a buffer memory barrier |
| `cuMemsetD8_v2` | ✓ | ✓ | scan zero, compact zero, sizes zero (all `value=0`) | `vkCmdFillBuffer(buf, 0)` | All call sites use `value=0`; `vkCmdFillBuffer` needs `data=u32` (works with 0) |
| `cuLaunchKernel` | ✓ | ✓ | every dispatch | `vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdPushConstants` + `vkCmdDispatch[Indirect]` | Args become push-constants (≤128 B per §6.1) + descriptor-bound SSBOs (or BDA `uint64_t`) |
| `cuStreamCreate_v2(NON_BLOCKING)` | ✓ | – | `module_loader.zig:201` `pipeline_stream` | n/a — `VkQueue` + `VkCommandBuffer` per logical stream | |
| `cuStreamDestroy_v2` | ✓ | – | `decode_context.zig:365-367` | `vkDestroyCommandPool` etc. | |
| `cuStreamSynchronize_v2` | ✓ | – | `streamlz_gpu.zig:725` (`slzWaitAndGetLastTimings`); cross-stream barrier at `decode_dispatch.zig:603-605` | `vkWaitForFences` or `vkWaitSemaphores(timeline)` | The single decode cross-stream sync becomes a timeline-semaphore signal/wait |
| `cuMemcpyHtoDAsync` / `cuMemcpyDtoHAsync` / `cuMemsetD8Async` | ✓ | – | **DEAD CODE** — loaded but never called | n/a | Drop on Vulkan port |
| `cuEventCreate` / `Record` / `Synchronize` / `ElapsedTime` / `Destroy_v2` | ✓ | – (uses decode's) | `beginKernelTiming`/`endKernelTiming`/`finalizeProfiling` (`decode_context.zig:100-186`) | `VkQueryPool(VK_QUERY_TYPE_TIMESTAMP)` + `vkCmdResetQueryPool` + `vkCmdWriteTimestamp(TOP/BOTTOM_OF_PIPE)` + `vkGetQueryPoolResults(WAIT_BIT)` → ticks × `VkPhysicalDeviceLimits::timestampPeriod / 1e6` ms | Mind `timestampValidBits != 0` per queue family |

### 4.1 Device-pointer model (BDA decision — LOCKED)

Deep-dive `device-ptr-aba-and-bda-feasibility` resolved this conclusively:

**Decision: require `VK_KHR_buffer_device_address` (Vulkan 1.2 core).** Every C-ABI device pointer (`void* d_input`, `d_output`, `d_frame`) is treated as `VkDeviceAddress` — a raw `uint64_t` GPU virtual address obtainable via `vkGetBufferDeviceAddress`. This preserves the C ABI literally (`void*` semantics) and matches every host-side pointer-arithmetic site in the codebase (9 enumerated; all are simple `base + N` byte additions).

Caller-origin pointers are offset host-side in exactly two places:
- `d_frame + block_start` (`src/decode/streamlz_decoder.zig:177`) — decode D2D path.
- `dev_target + req.dst_start_off` (`src/decode/decode_dispatch.zig:482`) — finalize copy (currently always 0 on async path).

Both are simple byte additions BDA handles identically. Library-internal pointers (`d_compact_counts + 20`, `d_walk_meta + offset`, `d_entropy_scratch + off16_offset`) are library-allocated, so the port owns the `VkBuffer`+`VkDeviceAddress` pair. In GLSL: `layout(buffer_reference, scalar) buffer …`. No descriptor-set-binding hot-path overhead.

---

## 5. Host-side orchestration

### 5.1 Encode

**Lifecycle.** `EncodeContext` (`src/encode/encode_context.zig:92-277`) owns 25 persistent `CUdeviceptr` slots paired with `_size: usize`. `ensureBuf` (`encode_context.zig:284-293`) does free-then-alloc on every grow. Encode init defers to decode (`src/encode/module_loader.zig:54-57`) for the CUDA context.

**Per-call kernel sequence (L3+ full path).** Confirmed by deep-dive `encode-host-d2h-walkstream-fusion`:

1. `gpuCompressImpl` — H2D input + descs + memset sizes → `cuCtxSynchronize` → `slzLzEncodeKernel` on stream 0 → `cuCtxSynchronize` → **D2H comp_sizes (4B/block) + D2H per-block raw LZ bytes (loop, ~2048 sync copies at 256 MB)**.
2. Host walks the LZ output via `walkStream` (`encode_huff.zig:262-305`) to build `HuffEncDesc[]` for literals/tokens/off16-hi/off16-lo.
3. 3× `gpuEncodeHuffImpl` (lit / tok / off16 split into hi+lo): H2D descs → `slzHuffBuildTablesKernel` → `slzHuffEncode4StreamKernel` → `cuCtxSynchronize` → D2H sizes.
4. `gpuAssembleFrameImpl` — H2D `AssembleDesc[]` → `slzAssembleMeasureKernel` → `cuCtxSynchronize` → **D2H enc_sizes → host prefix-sum → H2D descs again** → `slzAssembleWriteKernel` → `cuCtxSynchronize`.
5. `gpuFrameAssembleImpl` — H2D 4 small tables + `prefix_bytes` (≤128 B) → `slzFrameAssembleKernel` on `work_stream` (the only kernel that rides the caller's stream).
6. Optional final D2H of full frame to caller `dst` (skipped when `d_output_override` is set).

**Streams.** Only stream 0 (default) + caller's `work_stream`. No `cuStreamCreate` on encode. No pinned host memory.

**Sync hot-list.** 8 unconditional `cuCtxSynchronize` per encode call plus 1 conditional. The two host-bounce points (LZ→Huff via `walkStream`, AssembleMeasure→AssembleWrite via host prefix-sum) are the main barriers preventing single-submission encode.

**Override fields.** `d_input_override`/`d_output_override` (raw u64) are set by `slzCompressAsync` (`streamlz_gpu.zig:286-294`) and reset via `defer`. Whole-buffer; never offset host-side. Encoder writes from offset 0.

**Worker thread.** `slzCompressHost` spawns a worker via `runOnWorker` with a **32 MiB stack**. `slzCompressAsync` runs **inline** on the caller's thread.

### 5.2 Decode

**Lifecycle.** `DecodeContext` (`src/decode/decode_context.zig:194-289`) owns 21 `CUdeviceptr` slots + 2 explicit non-owning view aliases (`d_entropy_off16_scratch` = `d_entropy_scratch + off16_offset`; `d_first_subchunk_idx` = `d_first_sub_idx_persist`). One process-wide singleton at `driver.g_default` + per-handle `Context.dec`. `pipeline_stream` is library-owned (`CU_STREAM_NON_BLOCKING`).

**Per-call kernel chain (full GPU launch).** Confirmed by §2.2 and `two-stream-cross-sync-barrier-topology` deep-dive:

```
walkFrame  →  prefixSumChunks  →  scanParse  →  4× compactHuff  →  compactRaw  →  gatherRawOff16  →  mergeHuff
            └─────────────── work_stream (or caller's stream) ────────────────┘                                  ↓
                                                                                            cuStreamSynchronize(work_stream)
                                                                                                                   ↓
                                                                            huffBuildLut → huffDecode4Stream → lzDecode{,Raw}
                                                                            └────────── heavy_stream (work_stream or pipeline_stream) ──────────┘
                                                                                                                   ↓
                                                                                                    finalize D2D/D2H
```

**Single cross-stream barrier.** `cuStreamSynchronize(work_stream)` at `decode_dispatch.zig:603-605` — the only cross-stream sync, drains front-half before back-half launches.

**Buffers crossing the boundary** (each needs a Vulkan `VkBufferMemoryBarrier` + semaphore wait): `d_comp_persist`, `d_descs_persist`, `d_first_subchunk_idx` / `d_first_sub_idx_persist`, `d_compact_{lit,tok,hi,lo,raw,counts}`, `d_huff_descs`, `d_entropy_off16_scratch`, plus `d_n_groups_scratch` (or `d_walk_meta + n_chunks_offset` on the D2D fast path).

**Self-gating pattern.** Most back-half kernels are over-launched at host-known worst-case grids and self-gate on a device-resident count (`*d_n_blocks`, `*d_count`, `*p_n`). Vulkan port can preserve this pattern OR use `vkCmdDispatchIndirect` for exact grids.

**Pinned host.** `cuMemAllocHost` resolved at startup but the only library hot-path slot (`h_pinned_output`) is **never assigned** outside `deinit` — only the CLI bench uses pinned memory.

**Timing model.** Two layers: (1) per-kernel `cuEvent` pairs queued by `beginKernelTiming`/`endKernelTiming` and drained by `finalizeProfiling` (`decode_context.zig:100-186`); (2) wall-clock `last_kernel_ns`/`last_lz_kernel_ns`/`last_huff_kernel_ns` via `std.Io.Clock.awake` reads gated by `SLZ_E2E_TIMER`/`SLZ_SPLIT_TIMER` env vars.

---

## 6. Public C/Zig API surface

`include/streamlz_gpu.h` defines an opaque-handle nvCOMP-shaped API. **No `slzMakeDeviceOnlyHandle` symbol exists** (the prompt's expectation was wrong). The device-vs-host distinction is signalled entirely by entry-point choice (`*Host` = host pointers, `*Async` = device pointers).

| # | Symbol | Signature (abbreviated) | Defined at | Vulkan sibling | Notes |
|---|--------|------------------------|-----------|----------------|-------|
| 1 | `slzStatusString` | `const char* (slzStatus_t)` | `streamlz_gpu.zig:403` | `slzStatusString_vk` | static string |
| 2 | `slzVersionString` | `const char* (void)` | `streamlz_gpu.zig:419` | `slzVersionString_vk` | "3.0.0" |
| 3 | `slzCreate` | `slzStatus_t (slzHandle_t* out)` | `streamlz_gpu.zig:424` | `slzCreate_vk` | requires available GPU |
| 4 | `slzDestroy` | `slzStatus_t (slzHandle_t)` | `streamlz_gpu.zig:434` | `slzDestroy_vk` | NULL → SUCCESS |
| 5 | `slzCompressDefaultOpts` | `slzCompressOpts_t (void)` | `streamlz_gpu.zig:447` | `slzCompressDefaultOpts_vk` | level=5, profiling=0 |
| 6 | `slzDecompressDefaultOpts` | `slzDecompressOpts_t (void)` | `streamlz_gpu.zig:451` | `slzDecompressDefaultOpts_vk` | profiling=0 |
| 7 | `slzCompressBound` | `slzStatus_t (handle, size_t in, opts, size_t* out)` | `streamlz_gpu.zig:464` | `slzCompressBound_vk` | level-independent; `opts` is ignored |
| 8 | `slzGetDecompressedSize` | `slzStatus_t (handle, const void* bytes, size_t n, size_t* out)` | `streamlz_gpu.zig:483` | `slzGetDecompressedSize_vk` | pure host frame-header parse |
| 9 | `slzCompressHost` | `slzStatus_t (handle, const void* in, size_t, void* out, size_t cap, size_t* sz, opts)` | `streamlz_gpu.zig:505` | `slzCompressHost_vk` | spawns 32 MiB-stack worker |
| 10 | `slzDecompressHost` | similar host-pointer signature | `streamlz_gpu.zig:533` | `slzDecompressHost_vk` | spawns 32 MiB-stack worker |
| 11 | `slzCompressAsync` | `… , opts, void* stream` | `streamlz_gpu.zig:565` | `slzCompressAsync_vk(…, VkSemaphore wait, VkSemaphore signal, VkFence completion)` (recommended; see §6.2) | inline; writes `*compressed_size` BEFORE returning |
| 12 | `slzDecompressAsync` | `…, opts, void* stream` | `streamlz_gpu.zig:626` | `slzDecompressAsync_vk(…, semaphores/fence)` | tries true-D2D path, falls back with `SLZ_ERROR_UNSUPPORTED` on shape mismatch |
| 13 | `slzGetLastTimings` | `(handle, slzKernelTiming_t*, size_t cap, size_t* count)` | `streamlz_gpu.zig:674` | `slzGetLastTimings_vk` | drains enc+dec; idempotent |
| 14 | `slzWaitAndGetLastTimings` | `(handle, void* stream, slzKernelTiming_t*, …)` | `streamlz_gpu.zig:717` | `slzWaitAndGetLastTimings_vk(…, VkSemaphore/VkFence)` | only stream sync if non-NULL |

### 6.1 ABI struct invariants (kept verbatim in the `_vk` sibling)

- `slzCompressOpts_t` = `{ int level; int enable_profiling; int reserved[6]; }` (32 B); comptime-asserted in `streamlz_gpu.zig:107-111`.
- `slzDecompressOpts_t` = `{ int enable_profiling; int reserved[7]; }` (32 B); asserted at `:108,112-113`.
- `slzKernelTiming_t` = `{ const char* name; float ms; }` (16 B on 64-bit; asserted at `:114-119`). `name` is **static lifetime** in caller-readable memory.
- `slzStatus_t` enum: 8 codes `0..7` (`include/streamlz_gpu.h:80-89`).
- `slzHandle_t` = `struct slzContext*` (opaque).

### 6.2 Async stream-handle ABI (DESIGN DECISION REQUIRED)

CUDA's `void* stream` casts to `CUstream`. Vulkan has no direct analog. Recommended ABI shape (locks the design):

```c
slzStatus_t slzCompressAsync_vk(
    slzHandle_vk_t        handle,
    const void*           d_input,            // VkDeviceAddress (uint64_t) — BDA-required
    size_t                input_size,
    void*                 d_output,           // VkDeviceAddress
    size_t                max_compressed_size,
    size_t*               compressed_size,    // written before return
    slzCompressOpts_t     opts,
    VkSemaphore           wait_semaphore,     // optional; library waits on submit
    VkSemaphore           signal_semaphore,   // optional; library signals on completion
    VkFence               completion_fence    // optional; library signals on completion
);
```

The current CUDA `void*` slot maps to **`VkSemaphore signal_semaphore`** (the most useful per-call handle), with `wait_semaphore`/`completion_fence` added as additional parameters. `slzWaitAndGetLastTimings_vk` accepts a `{ VkSemaphore, uint64_t value }` for timeline semaphores. Sync mode = pass `VK_NULL_HANDLE` for all three; library uses an internal queue and blocks until idle.

### 6.3 Push-constant budget per kernel (deep-dive resolved)

All 18 kernels fit in Vulkan's 128 B push-constant guarantee. Worst case is `slzFrameAssembleKernel` at ~92 B. Detailed table (deep-dive `push-constant-budget-per-kernel`):

| Kernel | Param count | Push-constant bytes | Fits 128 B |
|--------|-----------:|--------------------:|:----------:|
| `slzLzEncodeKernel` | 9 | 56 | ✓ |
| `slzHuffBuildTablesKernel` | 6 | 40 | ✓ |
| `slzHuffEncode4StreamKernel` | 10 | 68 | ✓ |
| `slzAssembleMeasureKernel` | 7 | 52 | ✓ |
| `slzAssembleWriteKernel` | 7 | 52 | ✓ |
| `slzFrameAssembleKernel` | 15 | 92 (u8→u32 widened) | ✓ |
| `slzMergeHuffDescsKernel` | 12 | 88 | ✓ |
| `slzGatherRawOff16Kernel` | 5 | 36 | ✓ |
| `slzHuffBuildLutKernel` | 4 | 32 | ✓ |
| `slzHuffDecode4StreamKernel` | 5 | 40 | ✓ |
| `slzLzDecodeRawKernel` | 6 | 40 | ✓ |
| `slzLzDecodeKernel` | 9 | 64 | ✓ |
| `slzWalkFrameKernel` | 10 | 72 (collapsible to 48) | ✓ |
| `slzPrefixSumChunksKernel` | 5 | 32 | ✓ |
| `slzScanParseKernel` | 12 | 88 | ✓ |
| `slzCompactHuffDescsKernel` | 4 | 32 | ✓ |
| `slzCompactRawDescsKernel` | 5 | 40 | ✓ |

Optimisation opportunities (not blocking): collapse sibling-offset pointer clusters (walk-meta 6 ptrs → 1 BDA + 6 u32 offsets; scan-staged 6 → 1+6; compact-counts 4 → 1+4) and hoist per-call invariants (`hash_bits`, `chunks_per_group`, `sub_chunk_cap`, etc.) into a persistent UBO bound once per encode/decode call.

---

## 7. Wire format invariants

The Vulkan encoder MUST reproduce every one of these exactly. See `FORMAT.md` for the prose reference; this section enumerates the bytes the Vulkan side must emit identically.

### 7.1 Frame header (14–26 bytes)

| Offset | Size | Field | Notes |
|-------:|----:|-------|-------|
| 0 | 4 | Magic `0x534C5A31` LE | (`src/format/frame_format.zig:37`) |
| 4 | 1 | Version `2` | rejected otherwise |
| 5 | 1 | Flags `FrameFlags` packed(u8) | bit0 content_size_present (GPU encoder always sets), bit3 dictionary_id_present (rejected by decoder) |
| 6 | 1 | Codec `1` (Fast) | only value GPU emits |
| 7 | 1 | Level | internal [1,9]; GPU maps user L1..L5 → 1/2/3/5/6 |
| 8 | 1 | `log2(block_size) - 16` | GPU always writes `2` (=256 KiB) |
| 9 | 4 | `sc_group_size` f32 LE | GPU picks {0.25, 0.5}; must be > 0 |
| 13 | 1 | Reserved `0` | |
| 14 | 8 | `content_size` i64 LE | iff flag set |
| 22 | 4 | `dictionary_id` u32 LE | iff flag set (rejected) |

### 7.2 Block header (8 bytes)

| Offset | Size | Field |
|-------:|----:|-------|
| 0 | 4 | `compressed_size_word` LE — low 30 bits size, bit 30 `parallel_decode_metadata`, bit 31 `uncompressed` |
| 4 | 4 | `decompressed_size` u32 LE |

End-mark: 4 bytes of `0` (no `decompressed_size` follows). Optional XXH32 content-checksum trailer iff flag — GPU encoder NEVER sets.

### 7.3 Internal block header (2 bytes, read byte-by-byte)

- Byte 0: low nibble = magic `0x5`, bit 4 SelfContained, bit 5 TwoPhase, bit 6 RestartDecoder, bit 7 Uncompressed.
- Byte 1: bits 0–6 `decoder_type` (1=Fast, 2=Turbo); bit 7 UseChecksums.
- **Critical:** read byte-by-byte, NOT as u16. Misinterpretation flips magic-nibble.

GPU encoder always SelfContained → every chunk past the first has its first 8 bytes overwritten from an SC tail prefix table at frame end. Table size = `(num_chunks - 1) * 8` bytes (`src/encode/assemble_kernel.cu:457-470`).

### 7.4 Chunk header (4 bytes LE)

| Bits | Field |
|------|-------|
| 0–17 | `compressed_size - 1` (18-bit; max 256 KiB) |
| 18–19 | type (0=normal LZ, 1=memset, 2+=reserved) |
| 20–31 | reserved (must be 0 on write) |

### 7.5 Sub-chunk header (3 bytes **BIG-endian**)

| Bits | Field |
|------|-------|
| 23 | LZ flag |
| 22–19 | decode mode (4 bits; GPU emits 0=raw and 1=LZ) |
| 18–0 | `compressed_size` (19-bit; GPU caps ≤ 128 KiB) |

### 7.6 LZ sub-chunk payload stream order

```
[8 init bytes (first sub-chunk of frame only)]
[literals chunk (entropy-header type 0 or 4)]
[tokens chunk    (entropy-header type 0 or 4)]
[cmd_stream2_offset u16 LE (iff sub_decomp_size > 64 KiB)]
[off16 header + data]
[off32 header + data]
[length stream raw]
```

Sub-stream count headers (literals/tokens) are 3-byte **big-endian** `LZ_SUBSTREAM_COUNT_HDR_BYTES=3`. Off16 count is 2-byte **little-endian** (`OFF16_HEADER_BYTES=2`). `cmd_stream2_offset` is 2-byte LE.

### 7.7 Entropy chunk headers

- Type 0 (raw): byte0 high nibble = 0; 3-byte BE count.
- Type 4 (Huffman): non-compact 5-byte header — byte0 = `[type:4 | dst_size_minus_1[17:14]:4]`; bytes 1..4 BE u32 = `[dst_size_minus_1[13:0]:14 | comp_size:18]`. Written by `writeHuffChunkHdr` (`src/encode/assemble_kernel.cu:156-164`).

### 7.8 Off16 split into hi/lo planes

When entropy is beneficial, off16 count word becomes the marker `0xFFFF` (`OFF16_ENTROPY_MARKER`) and the stream splits into hi + lo planes, each emitted as its own entropy chunk. The split-decision threshold `OFF16_ENTROPY_MIN = 32` is **encoder-private** (decoder doesn't care) but the byte-level form is wire.

### 7.9 Off32 entries

Header: 3-byte LE packed — bits 23–12 = block-0 count, bits 11–0 = block-1 count. Count `4095` (`OFF32_COUNT_PACK_MAX`) is an escape; a u16 LE follows for the real count. Entry: 3 bytes when `(byte[2] & 0xC0) != 0xC0`, else 4 bytes (extended 22-bit offset).

### 7.10 Type-4 Huffman BIL body (HARD-LOCKED at 32 streams)

`HUFF_BODY_HEADER_BYTES = 228`:
- 128 B **weights** (256 4-bit lengths, packed low-nibble-first).
- 96 B **sub-header** (32 × u24 LE per-stream byte sizes).
- 4 B **K** (u32 LE = `min over s of words[s]` where `words[s] = (size[s] + 3) / 4`).

Interleaved area: K rows of `HUFF_BIL_ROW_BYTES = 128` (= 32 streams × 4 B). Stream `s`'s word `w` lives at `(w*128 + s*4)`. After K rows, tail area at exclusive-prefix-sum-of-tail-sizes offsets. **Per-stream scratch zero-padded to multiple of 4 bytes** so refill `memcpy(&word, w_ptr, 4)` always lands on clean zero-padded boundary (`src/encode/huffman_kernel.cu:320-326`).

**Code lengths height-limited to 11 bits** (`HUFF_MAX_CODE_LEN`). Decode fast-LUT has `HUFF_LUT_ENTRIES = 1024` (10-bit index); an 11-bit code triggers an escape entry.

**The 32-stream count is wire-format-locked** by `static_assert(HUFF_NUM_STREAMS == 32, …)` (`src/common/gpu_huffman.cuh:70-72`). Vulkan port CANNOT use any other stream count regardless of physical subgroup size.

### 7.11 SC tail prefix table

After the last chunk's payload: `(num_chunks - 1) × 8` bytes. Decoder overwrites each non-first chunk's first 8 bytes from this table.

### 7.12 `compressBound` and `safe_space`

- `compressBound` formula at `src/encode/streamlz_encoder.zig:70-84`. Public ABI: callers size dst by it.
- `safe_space = 64` bytes — decoder over-read/over-write tolerance (`src/format/streamlz_constants.zig:23-28`). Callers must satisfy `dst.len >= cs + safe_space`.

### 7.13 Float → uint determinism (deep-dive resolved)

`sc_group_size` ∈ {0.25, 0.5}, multiplied by `SLZ_CHUNK_SIZE_BYTES = 262144` to derive `eff_chunk_size`. Both products (65536, 131072) are exactly representable as float32 — IEEE-754 1-ulp-correct multiply gives the exact integer; truncation (SPIR-V `OpConvertFToU` RTZ, identical to nvcc `__float2uint_rz` and Zig `@intFromFloat`) yields the integer itself. **Byte-identity holds across all vendors** without any FPRoundingMode decoration. `block_size` does NOT participate in the float multiply (the wire constant 262144 does), so byte-identity is also invariant to the encoded `block_size` field.

### 7.14 Histogram order-independence (deep-dive resolved)

The encode Huffman histogram uses `atomicAdd(hist[s], 1u)` (`src/encode/huffman_kernel.cu:137`). Integer addition of constant `+1` is commutative/associative — final `hist[]` is bit-exact regardless of scheduling. Every downstream step (tree build, height-limit fixup, canonical-code assignment, weights pack, BIL K and tail prefix-sum) is a pure deterministic function of `hist[]` and input bytes, with ascending-symbol-index tie-breaks. **Atomic ordering across vendors cannot change `.slz` bytes.** The decoder LUT builder's `__match_any_sync` (and atomicAdd on shared `per_L_base`) affects only the runtime LUT, which is never serialised.

---

## 8. Build system

### 8.1 Existing PTX pipeline

`tools/build_gpu.bat`:
1. `vcvarsall x64` (nvcc needs cl.exe).
2. For each of the 5 `.cu` translation units: `nvcc -cubin … -arch=sm_89 -O3` (for `cuobjdump -res-usage` printout) AND `nvcc -ptx … -arch=sm_89 -O3` (the embeddable artifact).
3. `zig build -Doptimize=ReleaseFast`.

`build.zig` itself does NOT invoke nvcc. It declares a custom **PTX-freshness step** that mtime-compares every `.cu`/`.cuh` under `src/encode|src/decode|src/common` against every `.ptx` and fails the build with a "run `tools\build_gpu.bat`" message if any source is newer. The 5 committed `.ptx` blobs are loaded via `@embedFile(name) ++ "\x00"` and passed to `cuModuleLoadData` at runtime.

Build artifacts: `streamlz.exe` (CLI from `src/main.zig`) and `streamlz_gpu.dll` (C ABI from `src/streamlz_gpu.zig`). Both depend on the PTX-freshness step. Only `-Dstrip` option exists.

**Stale CodeWiki/bat references to `-Dgpu=true` are dead text** — `build.zig` has no such option (`tools/bench_all.bat:42`, `tools/bench_d2d.bat:42`).

### 8.2 Proposed SPIR-V pipeline

**Add sibling steps `vk`, `vklib`, `test_vk` to `build.zig` that build `streamlz_vk.exe` + `streamlz_vk.dll`.** Default `zig build` (no args) keeps building CUDA-only for backward compatibility with all `.bat` scripts.

**`tools/build_glsl.bat`** (NEW):
1. Probe `%VULKAN_SDK%\Bin\glslc.exe` (fall back to PATH); error loudly if neither resolves.
2. For each `.comp` source: `glslc --target-env=vulkan1.3 -O -o <name>.spv <name>.comp`.
3. Optionally `spirv-cross --reflect` for binding/local_size inspection.
4. Compile a second variant with `-DFORCE_SUBGROUP_32=1` to a `*.sg32.spv` (paired output for the subgroup-size-control path).
5. `zig build vk -Doptimize=ReleaseFast` to rebuild Vulkan binaries.

**`build.zig` additions:**
- `newSpvFreshnessStep` mirroring `newPtxFreshnessStep` (walks `src_vulkan/encode|src_vulkan/decode|src_vulkan/common` for `.comp`/`.glsl`/`.h.glsl` vs `.spv`).
- Two new modules `vk_cli_module`, `vk_lib_module` rooted at `src_vulkan/main.zig` / `src_vulkan/streamlz_vk.zig`.
- `b.addLibrary(.{ .linkage = .dynamic, .name = "streamlz_vk", … })` so the import lib is `streamlz_vk.lib` and the DLL is `streamlz_vk.dll`.
- `b.addInstallHeaderFile("include/streamlz_vk.h", "streamlz_vk.h")`.

**SPIR-V layout** (one .comp per kernel, matching the CUDA one-.cu-per-kernel-family convention):

```
src_vulkan/shaders/
├── common/
│   ├── byteio.glsl          # readBE24/writeBE24/readU32LE/etc.
│   ├── huffman.glsl         # HUFF_* constants, packWeightByte
│   ├── warp.glsl            # WARP_SIZE=32, lastBitSet, warpCopy
│   ├── wire_format.glsl     # sub-chunk accessors, frame constants
│   └── match_any.glsl       # 11-ballot loop for L, 32-broadcast loop for h
├── encode/
│   ├── slz_lz_encode.comp
│   ├── slz_huff_build_tables.comp
│   ├── slz_huff_encode_4stream.comp
│   ├── slz_assemble_measure.comp
│   ├── slz_assemble_write.comp
│   └── slz_frame_assemble.comp
└── decode/
    ├── slz_walk_frame.comp
    ├── slz_prefix_sum_chunks.comp
    ├── slz_scan_parse.comp
    ├── slz_compact_huff_descs.comp
    ├── slz_compact_raw_descs.comp
    ├── slz_gather_raw_off16.comp
    ├── slz_merge_huff_descs.comp
    ├── slz_huff_build_lut.comp
    ├── slz_huff_decode_4stream.comp
    ├── slz_lz_decode.comp
    └── slz_lz_decode_raw.comp
```

**Commit the `.spv` files** alongside the `.ptx` files (mirrors the existing `git clone && zig build` no-SDK promise).

**Pipeline cache.** Persist `VkPipelineCache` to `%LOCALAPPDATA%/streamlz_vk/pipelines.bin` so cold-start latency is small after the first run.

---

## 9. Test infrastructure

### 9.1 Existing tests

22 `test "…"` blocks total in `src/`:

- **5 GPU roundtrip tests** in `src/encode/gpu_roundtrip_tests.zig` — payloads generated in-test (largest 128 KiB+1); skipped with `error.SkipZigTest` when `gpu_encoder.isAvailable()`, `gpu_driver.isAvailable()`, or `bindContextToCallingThread()` fails.
- **10 frame-header tests** in `src/format/frame_format.zig` — pure CPU; hand-built byte fixtures.
- **7 block/chunk header tests** in `src/format/block_header.zig` — pure CPU.

`src/assets/{enwik8.txt, silesia_all.tar, web.txt}` are read **only** by `tools/bench_*.bat` and the C harnesses — never by any `src/` test.

**Test runner.** `src/test_runner_parallel.zig` fans out across `min(cpuCount, 16)` worker threads via an atomic test-index counter (`fetchAdd(.monotonic)`). Cap rationale: "16 is the cap on parallel workers — past that, GPU contention (limited streams + serialized cuLaunchKernel) erases parallelism benefits while inflating peak memory." Wired into `build.zig:52-58` as `zig build test` and the `ptest` alias.

### 9.2 Plan for Vulkan parity

1. **Sibling Vulkan test target.** Add `b.addTest` with `root_module = vk_lib_module` and the same `test_runner_parallel.zig` runner. Expose as `zig build test_vk` (and `ptest_vk` alias). `zig build test` continues to run CUDA-only.

2. **`gpu_probe.zig` helper.** Factor a single `requireGpu()` (and `requireVkGpu()`) so the 3-line skip boilerplate condenses. Vulkan probe must also check: `VK_EXT_subgroup_size_control`, `subgroupSize == 32` capability, `shaderInt64`, `shaderInt8`/`8bit_storage` (where used), `bufferDeviceAddress`.

3. **Cross-backend conformance suite.** Add `tests/cross_backend_tests.zig`:
   - Encode same input under CUDA and Vulkan; `assertEqual(cuda_frame, vk_frame)` byte-by-byte.
   - Decode same `.slz` frame under both; assert byte-equal output.
   - Goldens at `tests/golden/*.slz` produced by CUDA — Vulkan must reproduce byte-for-byte.
   - Run via `zig build test_xbackend`, requires both backends present (skip otherwise).

4. **CI matrix.** One job per vendor where possible (NVIDIA, AMD, Intel discrete, lavapipe for headless). Currently no CI is configured; recommend Linux + lavapipe + Windows + NVIDIA for day-1.

5. **Re-derived test_worker cap.** `MAX_WORKERS = min(cpu_count, compute_queue_count × 2, 16)` (probe at runtime). Expose via `-Dtest-workers=N`.

6. **Split the consolidated GPU roundtrip test.** With per-worker `VkCommandPool` and the locked subgroup-size strategy, concurrent dispatches are safe — split the 14-case mega-test into per-case tests for parallelism.

---

## 10. Memory + timing models

### 10.1 Encode memory footprint

Per `device-memory-layout` survey, worst-case 256 MiB src at L5 sc=0.5:

| Slot | Formula | Size | Notes |
|------|---------|-----:|-------|
| `d_hash_persist` | `n_sub × (2·2^hb + 32768) × 4` (chain) | **2.5 GiB** | per `encode-hash-buffer-size-and-2gb-vkbuffer-cap` |
| `d_output_persist` | `n_gpu_blocks × per_block_cap` (`3×gpu_block`) | ~768 MiB | per-sub-chunk staging |
| `d_huff_scratch_persist` | `n_descs × 32 × scratch_per_stream` | ~520 MiB | |
| `d_input_persist` (or `d_host_wrap_input`) | `src.len` | 256 MiB | |
| `d_host_wrap_output` | `compressBound(src.len)` | ~262 MiB | |
| `d_asm_huff_{lit,tok,off16}` | sum of `bilDstCap(count)` | ~512 MiB | off16 doubled |
| Others (descs, sizes, frame tables) | small | < 32 MiB | |

**Aggregate ~5 GiB.** Beyond Vulkan's 2 GiB `maxStorageBufferRange` minimum guarantee per single VkBuffer.

### 10.2 Decode memory footprint

| Slot | Formula | Size at 256 MiB | Notes |
|------|---------|-----:|-------|
| `d_entropy_scratch` | `total_subchunks × 131072 × 3` | 1.5 GiB | dominant |
| `d_comp_persist` | `compressed_block.len + 16` | ~262 MiB | |
| `d_output` | `decompressed_size + 64` | 256 MiB | safe_space=64 |
| `d_huff_lut` | `n_huff × 4096` | ≤ 32 MiB | |
| Others | small | < 6 MiB | |

**Aggregate ~2 GiB.**

### 10.3 Allocation strategy for Vulkan (LOCKED)

- **Use VMA (Vulkan Memory Allocator)** — gives per-context `VmaAllocator` and arena-style suballocation under one or a few big `VkDeviceMemory` blocks. Avoids per-`ensureBuf`-grow `vkAllocateMemory` (which counts against `maxMemoryAllocationCount` ≈ 4096 desktop / 256 mobile).
- **Encode hash split.** Per `encode-hash-buffer-size-and-2gb-vkbuffer-cap` deep-dive: each sub-chunk's hash region is private (`global_hash + chunk_id × table_stride`), and hash indices are masked by `hash_size - 1` (per-sub-chunk modulus, NOT global). Bin-pack sub-chunks into ≤2 GiB `VkBuffer`s; per-launch base pointers via BDA. **Byte-identity preserved.**
- **Staging buffer pool.** Tiered: a small ~MB ring (HOST_VISIBLE+HOST_COHERENT) for ~13 small descriptor/size transfers per encode; a single large region (HOST_CACHED|HOST_VISIBLE, sized to peak `gpu_out = 3 × src_len`) reused across calls.
- **`cuMemAllocHost`-equivalent.** HOST_VISIBLE+HOST_COHERENT (+HOST_CACHED if exposed). Query types and prefer CACHED for D2H reads.

### 10.4 Timing model for Vulkan

- **Per-kernel timing.** `VkQueryPool(VK_QUERY_TYPE_TIMESTAMP)` sized to `MAX_KERNELS_PER_DISPATCH × 2` (≈64 slots). `vkCmdResetQueryPool` at start of each encode/decode. `vkCmdWriteTimestamp(TOP_OF_PIPE_BIT, pool, slot_start)` before `vkCmdDispatch`, `BOTTOM_OF_PIPE_BIT` after. `finalizeProfiling` → `vkGetQueryPoolResults(VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)` → `ticks × VkPhysicalDeviceLimits.timestampPeriod / 1e6` to ms.
- **Mind `timestampValidBits`.** Per queue family; 0 on some transfer queues. Restrict timing to compute queue families.
- **Cross-vendor `timestampPeriod` range:** 1.0 ns (NV) to ~83 ns (Adreno). Compute internally as f64; downcast to `slzKernelTiming_t.ms` (f32) at the boundary.
- **Wall-clock.** Drop `qpcNow` (Windows-only); standardise on `std.Io.Clock.awake` for `SLZ_E2E_TIMER`/`SLZ_SPLIT_TIMER`.

---

## 11. Error model

### 11.1 Existing error sets and C codes

Five Zig error sets funnel into 8 C status codes:

| C code | Value | Maps from |
|--------|------:|-----------|
| `SLZ_SUCCESS` | 0 | |
| `SLZ_ERROR_INVALID_HANDLE` | 1 | direct |
| `SLZ_ERROR_INVALID_ARG` | 2 | direct |
| `SLZ_ERROR_BUFFER_TOO_SMALL` | 3 | `CompressError.DestinationTooSmall`, `DecompressError.OutputTooSmall` |
| `SLZ_ERROR_CORRUPT_FRAME` | 4 | `DecompressError.{BadFrame,Truncated,SizeMismatch,Invalid…,…,ContentSizeTooLarge}`, `ParseError.*` (caught at C boundary) |
| `SLZ_ERROR_UNSUPPORTED` | 5 | `CompressError.{BadLevel,BadScGroupSize,BadBlockSize}`, `GpuError.BadMode`, async-fallback returns |
| `SLZ_ERROR_CUDA` | 6 | every `GpuError.{BackendNotAvailable, KernelLaunchFailed, SyncFailed, CopyFailed, KernelMissing}` + unknown `else` |
| `SLZ_ERROR_OUT_OF_MEMORY` | 7 | `Allocator.OutOfMemory`, `GpuError.OutOfDeviceMemory`, `std.Thread.spawn` failure |

`GpuError` has exactly 7 variants (`BackendNotAvailable`, `OutOfDeviceMemory`, `KernelLaunchFailed`, `SyncFailed`, `CopyFailed`, `KernelMissing`, `BadMode`); count is locked at compile-time by `isGpuFallbackError` (`streamlz_gpu.zig:212-220`).

Mapping switches (`mapCompressError`, `mapDecompressError`) take `anyerror` parameters → unmapped variants silently fall through to `SLZ_ERROR_CUDA`.

### 11.2 Vulkan error model

- **Keep the C enum byte-identical** to preserve ABI parity. Reuse `SLZ_ERROR_CUDA` (numeric 6) as "underlying GPU API failed" — rename its `slzStatusString_vk` text to "underlying GPU call failed".
- **Add two codes (additive ABI change):**
  - `SLZ_ERROR_DEVICE_LOST = 8` — `VK_ERROR_DEVICE_LOST` (no CUDA analogue; terminal; library may need to rebuild `VkDevice`).
  - `SLZ_ERROR_VK_FEATURE_MISSING = 9` — `VK_ERROR_FEATURE_NOT_PRESENT`/`EXTENSION_NOT_PRESENT` for subgroup-size-control, BDA, shaderInt64, etc. (init-time only).
- **`vkCall` helper.** Mirror `cudaCall` (`src/decode/descriptors.zig:212`) with tag enum `{submit, fence_wait, semaphore_wait, alloc, init, queue, copy}`. Maps `VkResult` → `VkError`.
- **Encoder DestinationTooSmall demotion.** The CUDA encoder demotes every kernel error to `DestinationTooSmall` in `fast_framed.zig`. Recommended on the Vulkan side: **stop catching GPU errors into DestinationTooSmall**; compose `VkError` into `CompressError_vk` properly. Real `BUFFER_TOO_SMALL` only when `dst.len < compressBound`.
- **ChecksumMismatch/ChunkSizeMismatch.** Declared but never raised in CUDA; preserve the same behaviour in Vulkan (or wire both backends to a `strict-mode` flag together) so byte-identity tests don't diverge.

---

## 12. Open architectural decisions

The design workflow must lock these before execution. Each is a real choice with trade-offs, not a guess.

1. **Subgroup-size strategy.** Locked-in primary: **require `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32`**. Open: ship a workgroup-of-32 + shared-mem-emulation fallback for older Adreno/Mali, or refuse to run? Recommendation: ship the fallback (per deep-dive `subgroup-size-control-target-matrix`, every intrinsic emulates trivially or feasibly).
2. **`__match_any_sync` strategy.** Locked: 11-ballot loop for site #3 (decode LUT); 32-broadcast OR-reduction for site #1 (encode bucket-hash); **eliminate site #2 entirely** (replace with single `subgroupShuffle(key4, highest_lower) == key4`). Open: ship an NVIDIA-only `NV_shader_subgroup_partitioned` fast path?
3. **LZ-decode workgroup mapping.** Per deep-dive `lz-decode-two-warp-per-block-mapping`: cleanest map is one warp per workgroup (`local_size_x=32`, grid.x doubled). Open: keep the 2-warp packing for SM-occupancy parity on NVIDIA? Recommendation: split (cost = host one-line change; benefit = portability).
4. **Encode pipeline fusion.** Per deep-dive `encode-host-d2h-walkstream-fusion`: the LZ→Huff `walkStream` and the AssembleMeasure→AssembleWrite prefix-sum are the only two CPU-in-the-loop barriers. Open: (a) fuse to a single command buffer by porting `walkStream` and the prefix-sum to GPU compute shaders, eliminating ~9 ctx-syncs and 1 bulk D2H of LZ output; OR (b) preserve the CUDA shape with two split submissions. Recommendation: fuse.
5. **Single-thread orchestration kernels (D1, D2, D4, D5, D7).** Per deep-dive `scan-and-compact-kernel-shapes`: 5 of 7 orchestration kernels are pure single-thread. Open: (a) fuse into ONE Vulkan mega-kernel (collapses 7 dispatches → 1); (b) move walk/prefix/compact/merge to CPU (host already has the compressed bytes pre-upload); (c) leave as 5 small `vkCmdDispatch(1,1,1)` calls.
6. **Device-pointer ABI.** Locked: **require BDA**; raw `void*` C ABI unchanged; one `VkSemaphore` + optional `VkFence` added to async signatures.
7. **Stream/queue mapping.** Locked: collapse the decode two-stream split (front/back) to ONE `VkQueue` with a timeline semaphore between front-half and back-half submissions. Caller-supplied `VkSemaphore` signals on full completion. Open: expose a second internal queue for overlap with caller's compute, or keep single-queue?
8. **Encode hash split into ≤2 GiB VkBuffers.** Locked: bin-pack sub-chunks per `encode-hash-buffer-size-and-2gb-vkbuffer-cap`. Open: what hash-bits cap to apply on mobile to keep VRAM under (say) 1 GiB?
9. **8-bit storage.** Open: require `VK_KHR_8bit_storage`+`shaderInt8` (drops some mobile devices) OR baseline on `uint data[]` with manual shift/mask (per deep-dive `byteio-glsl-port-with-alignment-and-storage-class`, recommended). Recommendation: baseline `uint data[]`; add 8-bit fast path later.
10. **Int64 requirement.** Used by encoder `read8safe` hash hot loop and decoder Huff bit-buffer. Open: gate Vulkan port to `shaderInt64` devices, or emulate via `uvec2`? Recommendation: require `shaderInt64` for day-1; `uvec2` emulation as future work.
11. **Pipeline-cache disk persistence.** Open: store `VkPipelineCache` blob to `%LOCALAPPDATA%/streamlz_vk/pipelines.bin`? Recommendation: yes — day-1 user-facing UX win.
12. **Validation layer toggle.** Per the error-model survey, current `slzOpts` struct has no `enable_validation`. Open: add an `slzCreateOpts_vk` struct + `slzCreateEx_vk`, OR honour `SLZ_VK_VALIDATION=1` env var? Recommendation: env var (no ABI churn).
13. **Test runner posture.** Open: split the 14-case roundtrip test for parallel submission; cross-backend conformance step; CI matrix.
14. **`zig build` default.** Locked: default = CUDA-only (no `.bat` script churn); Vulkan opt-in via `vk`/`vklib`/`test_vk` steps.
15. **`.spv` commit policy.** Locked: commit `.spv` alongside `.ptx` (preserves `git clone && zig build` no-SDK promise).
16. **Vendor-paired SPIR-V variants.** Open: compile each `.comp` twice (vanilla + `-DFORCE_SUBGROUP_32=1`), with `vk_loader.zig` picking at pipeline-create? Or single SPIR-V + runtime spec-constants?
17. **Worker-thread stack size.** Per public-API survey: re-measure `worker_stack_size = 32 << 20` after porting; document if it differs.
18. **Pipeline-stage choice for timestamp writes.** Locked: `TOP_OF_PIPE_BIT` for start, `BOTTOM_OF_PIPE_BIT` for end (matches CUDA "wraps the whole launch" semantics).
19. **Indirect dispatch for self-gating kernels.** Open: use `vkCmdDispatchIndirect` for `slzHuffBuildLut/Decode/SlzLzDecode` (true on-device count), or over-launch and early-return like CUDA? Both work; indirect is cleaner.
20. **Wave-64 (AMD GCN/CDNA / older Mali) fallback path.** Open: ship a separate SPIR-V variant that packs 2 logical warps into one physical wave-64 subgroup using upper-half masking? Defer to v2.

---

## 13. Risks

Each risk listed with severity, first-detection signal, and the deep-dive evidence that informed it.

| # | Risk | Severity | First detection signal |
|---|------|---------|------------------------|
| R1 | **`__match_any_sync` emulation regresses encode L1-L4 throughput unacceptably.** The greedy parser invokes 2 match-any per warp per ~32 input bytes; replacing with 32-broadcast OR-reduction on AMD/Intel/mobile is ~32× more cycles per call. Decode site (32× 8-batch × 1525 LUTs) is fine. | **HIGH** | Per-kernel timing of `slzLzEncodeKernel_vk` vs CUDA at L1 on enwik8 ≥ +30% slowdown. |
| R2 | **Subgroup-size = 32 not supportable on a meaningful mobile device class** (`VK_EXT_subgroup_size_control` absent or doesn't expose 32 in [`minSubgroupSize`,`maxSubgroupSize`]). Forces workgroup-of-32 shared-mem emulation everywhere, ~2-3× slower. | **HIGH** | Vendor probe table at design-workflow start; concretely Adreno < 650 and Mali Bifrost. |
| R3 | **Wire-format byte-identity regression caught only late.** No cross-backend conformance test exists today; `.slz` divergence may not surface until the user runs `streamlz_vk` against a `streamlz`-produced file. | **BLOCKER** | Add `cross_backend_tests.zig` (encode-cuda vs encode-vk, plus golden-decode) to first PR. |
| R4 | **D2D `d_input_override`/`d_output_override` raw-u64 C ABI semantics don't translate to Vulkan without BDA**; if a target device lacks `VK_KHR_buffer_device_address` (Vulkan 1.2 core but optional on some pre-2020 mobile drivers), the day-1 device-pointer path breaks. | **HIGH** | `vkGetPhysicalDeviceFeatures2` probe at `slzCreate_vk` — fail with `SLZ_ERROR_VK_FEATURE_MISSING`. |
| R5 | **Encode hash table > 2 GiB on a single VkBuffer.** Mitigation (bin-pack into ≤2 GiB chunks) is conditionally compatible; failure to ship the split silently OOMs `vkCreateBuffer` for src ≥ 128 MiB at L5. | **HIGH** | `vkCreateBuffer` returns `VK_ERROR_OUT_OF_DEVICE_MEMORY` or `INVALID_OPAQUE_CAPTURE_ADDRESS_KHR` on first 256 MB+ encode at L5. |
| R6 | **Pinned-host-equivalent (HOST_CACHED+HOST_VISIBLE) absent on some integrated/mobile vendors**; D2H bandwidth halves; encode/decode bench numbers regress. | **MEDIUM** | Memory-type probe + bench regression vs CUDA `cuMemAllocHost`. |
| R7 | **VkPipelineCache misses on first run cause ~100–500 ms warm-start latency** (significantly worse than CUDA's PTX JIT amortisation). | **MEDIUM** | Cold-start time of `SLZ_VK_E2E_TIMER` includes per-pipeline create cost; persist cache to disk. |
| R8 | **Cross-stream barrier elision regression**: the 13 implicit "stream ordering publishes X" sites (enumerated in §5.2 and `two-stream-cross-sync-barrier-topology`) must each become explicit `VkBufferMemoryBarrier`. Missing one = race condition; intermittent corruption only on specific drivers. | **HIGH** | Output corruption under Vulkan validation layers with synchronisation validation enabled. |
| R9 | **2-warp-per-block decoder hidden cross-warp coupling.** Deep-dive confirms no shared state, but `__syncwarp` semantics at warp-uniform-but-divergent control flow could behave differently under `subgroupBarrier()` on a 32-thread workgroup. | **MEDIUM** | GPU roundtrip test fails on a 32 KB LZ-decode-stressing input. |
| R10 | **Float→uint determinism non-issue on standard inputs, but external frames with non-{0.25,0.5} `sc_group_size` could expose RTZ-vs-RTE differences.** | **LOW** | Add test with `sc_group_size = 0.1` to exercise non-exact products. |
| R11 | **Histogram atomic ordering**: deep-dive proves byte-identity is preserved, but Vulkan shared-memory atomicAdd implementations on tile-based mobile may serialise differently → encoder throughput regression on degenerate (all-zeros) inputs. | **LOW** | Profile encoder Huffman build at L3 on a degenerate buffer. |
| R12 | **Per-kernel `cuMemFree`+`cuMemAlloc` regrow pattern under VMA**: transient peak memory during grow event briefly holds old+new; on a tight VRAM budget Vulkan OOMs uglier than CUDA. | **MEDIUM** | Run encode at growth boundary (e.g. 200 MB → 256 MB) under VRAM pressure (3.5 GB free). |
| R13 | **`__byte_perm(w, 0, 0x0123)` 4-site emulation correctness**: explicit bitswap is straightforward but the LE→BE byte-order swap is on the Huffman decode hot path; any off-by-one in the bit-stream pointer breaks every type-4 chunk. | **MEDIUM** | Decode any L3+ frame; mismatched bit-stream → `DecompressError.BadChunkHeader`. |
| R14 | **shaderInt64 unavailable on mobile** disables `read8safe` u64 hot path; encoder hash chain falls back to `uvec2` emulation. | **MEDIUM** | `slzCreate_vk` returns `SLZ_ERROR_VK_FEATURE_MISSING` on Mali Bifrost without int64. |
| R15 | **glslc compiles `__match_any_sync` emulation to inefficient code on AMD/Intel**: 32-broadcast loops may not unroll, falling back to runtime loops. | **LOW** | Inspect SPIR-V output of `slz_lz_encode.comp` with `spirv-cross`. |
| R16 | **Encoder `walkStream` on host stays the largest synchronisation barrier** if not ported to GPU compute. Affects single-submit pipeline target. | **MEDIUM** | Per-call `SLZ_E2E_TIMER` shows host phase ≥ 5% of total. |
| R17 | **Dead-code paths in error mapping (`ChecksumMismatch`, `ChunkSizeMismatch`)**: porter sees them, assumes verification happens, ships strict-mode that diverges across backends. | **LOW** | Cross-backend test on a frame whose checksum byte is bit-flipped. |
| R18 | **CodeWiki/`bench_all.bat` stale `-Dgpu=true` references** confuse new contributors; same risk for any new Vulkan-side docs that drift from `build.zig`. | **LOW** | New-contributor onboarding doc. |

---

*End of document.*
