# /srcVK/ Port Instructions — Step 3+ Fleshout Source of Truth

This file is the per-file fleshout checklist for the multi-step
1:1 CUDA→Vulkan port living under `/srcVK/`. Step 2 (this skeleton
pass) laid out every file from audit.md Section D as a stub with the
correct public surface. Step 3+ agents implement the bodies.

Authoritative inputs:
- `srcVK/audit.md` — file inventory, extensions, names, L1/L2 status.
- `src/` — CUDA source the bodies port from.
- `include/streamlz_gpu.h`, `include/streamlz_gpu_vk.h` — public C ABI;
  UNCHANGED by this port; both stay at `/include/`.

Universal rules (audit Section B + project EXCEPTION 2):
- No `cu` / `Cu` / `CU` / `cuda` / `Cuda` / `CUDA` / `nvidia` / `Nvidia`
  / `NVIDIA` token in any srcVK filename or in any srcVK-emitted
  identifier (type / function / module).
- L2 gate lives in `srcVK/decode/decode_dispatch.zig::fullGpuLaunchImpl`
  for decode and in `srcVK/encode/fast_framed.zig::compressFramedOne`
  for encode (the existing `opts.level >= 3` Huff gate). Per-call
  `level` is plumbed via `DecodeRequest.level` and `Options.level`.
- L1 scope = codec level 1 and 2 (LZ-only, no Huffman, no chain parser).
- L2 stubs return `error.NotImplementedL2`. L1-scope skeleton bodies
  in this pass return `error.NotYetPorted`; fleshout replaces with the
  real body.

VK-specific runtime adaptations:
- `procs.malloc_device` / `procs.free_device` wrap VMA (see
  `srcVK/vma.zig`).
- `procs.h2d` / `procs.d2h` wrap `vkCmdCopyBuffer` against staging
  buffers.
- `procs.launch_kernel` wraps `vkCmdBindPipeline` + `vkCmdBindDescriptorSets`
  + `vkCmdDispatch`.
- SPV blobs compile from every `.comp` under srcVK/ via glslangValidator
  (or glslc) at build time and embed into the binary via `@embedFile`.

---

## Phase 1 — Foundation

### srcVK/error.zig
**CUDA source:** (none — NEW)
**L1 scope?** yes (foundation)
**Description:** Cross-cutting error variants (`NotImplementedL2`,
`NotYetPorted`) reachable from every stub file without dragging in the
decode or encode trees.
**Public surface:** `NotImplementedL2`, `NotYetPorted` error sets.
**Dependencies:** none.

### srcVK/version.zig
**CUDA source:** `src/version.zig`
**L1 scope?** yes
**Description:** `pub const string` — `"3.0.0"`. Verbatim port.
**Public surface:** `string`.
**Per-function checklist:** none — single const.

### srcVK/mmap.zig
**CUDA source:** `src/mmap.zig`
**L1 scope?** yes
**Description:** Verbatim port; pure host, no GPU. Windows uses
CreateFileMappingW + MapViewOfFile; POSIX uses std.posix.mmap.
**Public surface:** `MappedFile`, `mapFileRead`, `mapFileReadWrite`.

### srcVK/format/streamlz_constants.zig
**CUDA source:** `src/format/streamlz_constants.zig`
**L1 scope?** yes
**Description:** Verbatim port; wire-format constants.
**Public surface:** `chunk_size`, `chunk_size_bits`, `chunk_size_mask`,
`chunk_type_shift`, `chunk_type_memset`, `sub_chunk_size`, `safe_space`,
`default_sc_group_size`, `max_dictionary_size`.

### srcVK/format/block_header.zig
**CUDA source:** `src/format/block_header.zig`
**L1 scope?** yes
**Description:** Verbatim port; 2-byte block + 4-byte chunk header parsers.
**Public surface:** `CodecType`, `BlockHeader`, `ChunkHeader`,
`ParseError`, `parseBlockHeader`, `parseChunkHeader`.

### srcVK/format/frame_format.zig
**CUDA source:** `src/format/frame_format.zig`
**L1 scope?** yes
**Description:** Verbatim port; SLZ1 frame header parser/writer + block
header parser/writer + scGroupSize helpers.
**Public surface:** `magic`, `version`, `end_mark`,
`block_uncompressed_flag`, `block_parallel_decode_metadata_flag`,
size constants, `FrameFlags`, `Codec`, `FrameHeader`, `ParseError`,
`parseHeader`, `scGroupSizeToBytes`, `scGroupSubChunkSize`,
`WriteHeaderOptions`, `WriteError`, `writeHeader`, `BlockHeader`,
`parseBlockHeader`, `writeBlockHeader`, `writeEndMark`.

### srcVK/common/gpu_wire_format.glsl
**CUDA source:** `src/common/gpu_wire_format.cuh`
**L1 scope?** yes
**Description:** Device-side wire-format constants — must agree
byte-for-byte with `streamlz_constants.zig`.
**Public surface:** `INITIAL_LITERAL_COPY_BYTES`, `OFF32_COUNT_PACK_MAX`,
`LZ_BLOCK_SIZE`, and the rest from the CUDA `.cuh`.

### srcVK/common/gpu_byteio.glsl
**CUDA source:** `src/common/gpu_byteio.cuh`
**L1 scope?** yes
**Description:** Endian byte-IO primitives. Translate to GLSL using
`packUint2x32` / `unpackUint2x32` etc.
**Public surface:** `readBE24`, `readU32LE`, `writeBE24`, etc.

### srcVK/common/gpu_warp.glsl
**CUDA source:** `src/common/gpu_warp.cuh`
**L1 scope?** yes
**Description:** Warp/subgroup constants. CUDA `WARP_SIZE=32` becomes
GLSL `gl_SubgroupSize`. The fleshout agent must apply the audit Section H
question 1 decision (require subgroupSize=32 or parameterize).
**Public surface:** `WARP_SIZE`, `LANE_MASK`, `FULL_WARP_MASK`, subgroup
shuffle/ballot wrappers.

### srcVK/common/gpu_huffman.glsl
**CUDA source:** `src/common/gpu_huffman.cuh`
**L1 scope?** no (L2 stub bodies)
**L2 stub?** yes
**Description:** Shared Huffman wire-format constants. Constants live so
`encode_huff.zig` references resolve at compile time; bodies are L2 stubs.
**Public surface:** `HUFF_NUM_STREAMS`, `HUFF_BODY_HEADER_BYTES`, `HUFF_*`.

---

## Phase 2 — VK driver layer + VMA wiring

### srcVK/vma.zig
**CUDA source:** (none — NEW)
**L1 scope?** yes (foundation)
**Description:** Hand-rolled VMA Zig binding. `@cInclude` the copy at
`srcVK/vma/vk_mem_alloc.h`.
**Public surface:** `VmaAllocator`, `VmaAllocation`, `VmaPool`,
re-exports of the Vulkan types the codec touches, `createAllocator`,
`destroyAllocator`, `createBuffer`, `destroyBuffer`.

### srcVK/decode/vulkan_api.zig
**CUDA source:** `src/decode/cuda_api.zig` (RENAMED per Section B)
**L1 scope?** yes
**Description:** vulkan-1.dll loader, `procs` struct (`malloc_device`,
`free_device`, `h2d`, `d2h`, `d2d`, `memset_d8`, `memset_d8_async`,
`malloc_host`, `free_host`, `stream_sync`, `ctx_sync`, `stream_create`,
`stream_destroy`, `launch_kernel`), `VkResult`, `VkDeviceBuffer`,
`VkStream`, `VK_SUCCESS_RC`, `qpcNow` / `qpcMs`, `sm_count` analogue.
**Public surface:** all of the above plus `getProc`.
**VK adaptation notes:** `procs.malloc_device` calls VMA; `procs.h2d` /
`procs.d2h` use staging buffers; `procs.launch_kernel` wraps
`vkCmdBindPipeline` / `vkCmdBindDescriptorSets` / `vkCmdDispatch`.
The `sm_count` field's fill choice is in audit Section H Q2.

### srcVK/decode/descriptors.zig
**CUDA source:** `src/decode/descriptors.zig`
**L1 scope?** yes
**Description:** POD descriptor structs + `GpuError` (gains
`NotImplementedL2`) + `vkCall` (renamed from `cudaCall`) + per-kernel
timing types.
**Public surface:** `ChunkDesc`, `HuffDecChunkDesc`, `HUFF_LUT_ENTRIES`,
`MAX_SUB_CHUNKS_PER_CHUNK`, `MAX_HUFF_DESCS_PER_STREAM`,
`MAX_RAW_OFF16_DESCS`, `ENTROPY_SCRATCH_SLOT_BYTES`,
`OFF16_HILO_SPLIT_OFFSET`, `CHUNK_TYPE_SHIFT`, `CHUNK_TYPE_MASK`,
`KernelTiming`, `PendingTiming`, `ScanHuffDesc`, `ScanRawDesc`,
`RawOff16Desc`, `ScanResult`, `WalkFrameResultDev`, `WALK_MAX_CHUNKS`,
`walk_meta_offsets`, `PrefixSumResultDev`, `GpuError`, `ErrorKind`,
`vkCall`.

### srcVK/encode/vulkan_ffi.zig
**CUDA source:** `src/encode/cuda_ffi.zig` (RENAMED per Section B)
**L1 scope?** yes
**Description:** Encode-side FFI shim. Mirrors `decode/vulkan_api.zig`
but bound to the encode subdir so the encode tree compiles without
importing decode/.
**Public surface:** `win32`, `VkResult`, `VkDevice`, `VkDeviceBuffer`,
`VK_SUCCESS_RC`, `VkPipelineLayout`, `VkPipeline`, `Fn*` typedefs,
`vk*_fn` slots, `getProc`.

---

## Phase 3 — Module loader + SPV blob loading

### srcVK/decode/module_loader.zig
**CUDA source:** `src/decode/module_loader.zig`
**L1 scope?** yes
**Description:** Brings up Vulkan: load `vulkan-1.dll`, enumerate
physical devices, pick one, create logical device + compute queue,
fill `procs`, create the VkPipeline objects for every SPV blob
referenced as a `pub var` here.
**Public surface:** `init`, `isAvailable`, `module`, `kernel_fn`,
`kernel_raw_fn`, `gather_off16_fn`, `scan_parse_fn`, `walk_frame_fn`,
`prefix_sum_chunks_fn`, `compact_huff_descs_fn`, `compact_raw_descs_fn`,
`merge_huff_descs_fn`, `huff_module`, `huff_build_fn`, `huff_decode_fn`,
`ensurePipelineStream`.

### srcVK/encode/module_loader.zig
**CUDA source:** `src/encode/module_loader.zig`
**L1 scope?** yes
**Description:** Sits on top of the decode-side `init` (single shared
VkDevice across encode + decode). Resolves the encode SPV blobs.
**Public surface:** `init`, `isAvailable`, `module`, `kernel_fn`,
`huff_module`, `huff_tables_kernel_fn`, `huff_encode_kernel_fn`,
`assemble_module`, `assemble_measure_fn`, `assemble_write_fn`,
`frame_assemble_fn`, `initialized`.

---

## Phase 4 — L1 decode kernels (.glsl + .comp)

### srcVK/decode/slz_wire_format.glsl
**CUDA source:** `src/decode/slz_wire_format.cuh`
**L1 scope?** yes
**Description:** Device-side wire-format constants + descriptor structs
+ header parsers — must match `descriptors.zig` byte-for-byte.

### srcVK/decode/lz_decode_core.glsl
**CUDA source:** `src/decode/lz_decode_core.cuh`
**L1 scope?** yes
**Description:** Shared warp primitives (`warpScanU32`,
`warpLiteralCopy`, `warpMatchCopy`).

### srcVK/decode/lz_decode_raw.glsl
**CUDA source:** `src/decode/lz_decode_raw.cuh`
**L1 scope?** yes
**Description:** Raw-mode sub-chunk decoder helpers used by
`lz_decode_raw_kernel.comp` (the L1 hot decode entry).

### srcVK/decode/lz_dispatch.glsl
**CUDA source:** `src/decode/lz_dispatch.cuh`
**L1 scope?** yes (Raw path)
**L2 stub?** partial (general path body stubbed)
**Description:** `parseAndDecodeSubChunkRaw` (L1) and
`parseAndDecodeSubChunk` (L2 stub).

### srcVK/decode/lz_decode_raw_kernel.comp
**CUDA source:** `src/decode/lz_decode_kernels.cuh` (entry
`slzLzDecodeRawKernel`)
**L1 scope?** yes (HOT)
**Description:** Raw LZ decode kernel for level-1 frames. One workgroup
per chunk; each subgroup decodes one sub-chunk.
**VK adaptation notes:** Use `gl_SubgroupShuffleNV` /
`subgroupBroadcast` for the warp shuffles; `subgroupBallot` for the
warp masks. Subgroup-size policy per audit Section H Q1.

### srcVK/decode/prefix_sum_chunks_kernel.comp
**CUDA source:** `src/decode/prefix_sum_chunks_kernel.cuh` (entry
`slzPrefixSumChunksKernel`)
**L1 scope?** yes (HOT — every decode launches this)
**Description:** Per-chunk first-sub-chunk-index prefix sum.

---

## Phase 5 — L2 stub decode kernels

### srcVK/decode/lz_decode_general.glsl
**CUDA source:** `src/decode/lz_decode_general.cuh`
**L1 scope?** no
**L2 stub?** yes
**Description:** L2+ general entropy-capable decoder.

### srcVK/decode/lz_header_parse.glsl
**CUDA source:** `src/decode/lz_header_parse.cuh`
**L1 scope?** no
**L2 stub?** yes
**Description:** `parseSubChunkHeaders` for the general path.

### srcVK/decode/lz_decode_kernel.comp
**CUDA source:** `src/decode/lz_decode_kernels.cuh` (entry
`slzLzDecodeKernel`)
**L2 stub?** yes — empty `main()` body until L2 lands.

### srcVK/decode/gather_raw_off16_kernel.comp
**CUDA source:** `src/decode/gather_raw_off16_kernel.cuh`
**L2 stub?** yes

### srcVK/decode/walk_frame_kernel.comp
**CUDA source:** `src/decode/walk_frame_kernel.cuh`
**L2 stub?** yes

### srcVK/decode/compact_huff_descs_kernel.comp
**CUDA source:** `src/decode/compact_descs_kernels.cuh`
**L2 stub?** yes

### srcVK/decode/compact_raw_descs_kernel.comp
**CUDA source:** `src/decode/compact_descs_kernels.cuh`
**L2 stub?** yes

### srcVK/decode/merge_huff_descs_kernel.comp
**CUDA source:** `src/decode/merge_huff_descs_kernel.cuh`
**L2 stub?** yes

### srcVK/decode/scan_parse_kernel.comp
**CUDA source:** `src/decode/scan_parse_kernel.cuh`
**L2 stub?** yes

### srcVK/decode/huff_build_lut_kernel.comp
**CUDA source:** `src/decode/huffman_kernel.cu` (entry
`slzHuffBuildLutKernel`)
**L2 stub?** yes

### srcVK/decode/huff_decode_4stream_kernel.comp
**CUDA source:** `src/decode/huffman_kernel.cu` (entry
`slzHuffDecode4StreamKernel`)
**L2 stub?** yes

---

## Phase 6 — Decode context + dispatch infra

### srcVK/decode/decode_context.zig
**CUDA source:** `src/decode/decode_context.zig`
**L1 scope?** yes
**Description:** Per-context state with every `d_*` device-buffer slot
typed as `VkDeviceBuffer`. The Huff-related slots (`d_huff_descs`,
`d_huff_lut`, `d_compact_*`, `d_scan_*`) exist but go unused on L1.
**Public surface:** `DecodeContext`, `ensureDeviceBuf`,
`ensureDeviceOutput`, `allocHost`, `freeHost`, `copyHostToDevice`,
`copyDeviceToHost`, `bindContextToCallingThread`, `beginKernelTiming`,
`endKernelTiming`, `finalizeProfiling`, `DecodeContext.deinit`.
**VK adaptation notes:** allocations route through `procs.malloc_device`
(VMA-backed). `bindContextToCallingThread` may be a no-op on VK if the
VkDevice is process-wide.

### srcVK/decode/scan_gpu.zig
**CUDA source:** `src/decode/scan_gpu.zig`
**L1 scope?** yes (prefix-sum)
**L2 stub?** partial
**Description:** `gpuPrefixSumChunksImpl` (L1 hot — always launched);
`gpuWalkFrameImpl` (L2 stub — async-D2D path); `gpuScanChunks`
(L2 stub — short-circuit-returns null on L1).
**Public surface:** `gpuWalkFrameImpl`, `gpuPrefixSumChunksImpl`,
`gpuScanChunks`.

---

## Phase 7 — Decode dispatch orchestration (EXCEPTION 2 L2 gate)

### srcVK/decode/decode_dispatch.zig
**CUDA source:** `src/decode/decode_dispatch.zig`
**L1 scope?** yes (L2 gate added)
**Description:** Top-level dispatcher. Owns the L2 gate per project
EXCEPTION 2: for `req.level == 1` skip Huff/scan/compact/merge/gather
and run the raw LZ-decode kernel off the host-built chunk descs +
prefix sum.
**Public surface:** `VkProcs` (renamed from CudaProcs), `DecodeRequest`
(gains a `level: u8` field), `fullGpuLaunchImpl`, `runHuffBuildAndDecode`
(L2 stub), `runLzPipeline`, `finalizeOutput`, `uploadInputAndPrefixSum`,
`runBackHalf`, `mergeHuffDescs` (L2 stub), `gatherRawOff16` (L2 stub),
`emitE2eTrace`.
**L2 gate position:** in `fullGpuLaunchImpl`, wrap the
`runHuffBuildAndDecode` call + the scan / compact / merge / gather
chain in `if (req.level >= 2) { ... }`.

### srcVK/decode/driver.zig
**CUDA source:** `src/decode/driver.zig`
**L1 scope?** yes
**Description:** Thin facade: owns `g_default` + `last_*_kernel_ns`
telemetry vars; re-exports the sub-modules' public surface.
**Public surface:** `qpcNow`, `qpcMs`, `init`, `isAvailable`,
`ChunkDesc`, `KernelTiming`, `PendingTiming`, `GpuError`,
`WALK_MAX_CHUNKS`, `ENTROPY_SCRATCH_SLOT_BYTES`, `walk_meta_offsets`,
`DecodeContext`, `allocHost`, `freeHost`, `copyDeviceToHost`,
`copyHostToDevice`, `bindContextToCallingThread`, `beginKernelTiming`,
`endKernelTiming`, `finalizeProfiling`, `gpuWalkFrameImpl`,
`fullGpuLaunchImpl`, `g_default`, `last_kernel_ns`, `last_lz_kernel_ns`,
`last_huff_kernel_ns`.

---

## Phase 8 — L1 encode kernels (.glsl + .comp)

### srcVK/encode/lz_format.glsl
**CUDA source:** `src/encode/lz_format.cuh`
**L1 scope?** yes
**Description:** Encoder-private hash constants.

### srcVK/encode/lz_token_emit.glsl
**CUDA source:** `src/encode/lz_token_emit.cuh`
**L1 scope?** yes
**Description:** Token-emission helpers shared by both parsers.

### srcVK/encode/lz_greedy_parser.glsl
**CUDA source:** `src/encode/lz_greedy_parser.cuh`
**L1 scope?** yes (HOT — drives L1-L4)
**Description:** Warp-parallel greedy LZ parser.

### srcVK/encode/lz_encode_kernel.comp
**CUDA source:** `src/encode/lz_kernel.cu` (entry `slzLzEncodeKernel`)
**L1 scope?** yes (HOT)
**Description:** L1+L2 LZ encode kernel.

### srcVK/encode/assemble_measure_kernel.comp
**CUDA source:** `src/encode/assemble_kernel.cu` (entry
`slzAssembleMeasureKernel`)
**L1 scope?** yes (HOT)

### srcVK/encode/assemble_write_kernel.comp
**CUDA source:** `src/encode/assemble_kernel.cu` (entry
`slzAssembleWriteKernel`)
**L1 scope?** yes (HOT)

### srcVK/encode/frame_assemble_kernel.comp
**CUDA source:** `src/encode/assemble_kernel.cu` (entry
`slzFrameAssembleKernel`)
**L1 scope?** yes (HOT)
**Description:** Device-resident frame writer.

---

## Phase 9 — L2 stub encode kernels

### srcVK/encode/lz_chain_parser.glsl
**CUDA source:** `src/encode/lz_chain_parser.cuh`
**L1 scope?** no
**L2 stub?** yes (L5 only)

### srcVK/encode/huff_build_tables_kernel.comp
**CUDA source:** `src/encode/huffman_kernel.cu` (entry
`slzHuffBuildTablesKernel`)
**L2 stub?** yes

### srcVK/encode/huff_encode_4stream_kernel.comp
**CUDA source:** `src/encode/huffman_kernel.cu` (entry
`slzHuffEncode4StreamKernel`)
**L2 stub?** yes

---

## Phase 10 — Encode context + LZ + assemble host orchestration

### srcVK/encode/encode_context.zig
**CUDA source:** `src/encode/encode_context.zig`
**L1 scope?** yes (Huff-related fields exist but unused on L1)
**Description:** Per-encode-operation state. Every `d_*` slot becomes
`VkDeviceBuffer`.
**Public surface:** `INITIAL_LITERAL_COPY_BYTES`,
`SC_TAIL_PER_CHUNK_BYTES`, `CHUNK_INTERNAL_HDR_BYTES`,
`UNCOMPRESSED_CHUNK_HDR_BYTES`, `NEXT_HASH_ENTRIES`,
`UNCOMPRESSED_CHUNK_MARKER`, `CompressChunkDesc`, `AssembleDesc`,
`HuffEncDesc`, `EncodeContext`, `ensureBuf`, `copyDeviceToHost`,
`copyHostToDevice`.

### srcVK/encode/levels.zig
**CUDA source:** `src/encode/levels.zig`
**L1 scope?** yes
**Description:** Verbatim port.
**Public surface:** `hashBitsForLevel`, `useChainParser`.

### srcVK/encode/encode_lz.zig
**CUDA source:** `src/encode/encode_lz.zig`
**L1 scope?** yes (HOT)
**Description:** LZ-encode kernel launcher.
**Public surface:** `gpuCompressImpl`.

### srcVK/encode/encode_assemble.zig
**CUDA source:** `src/encode/encode_assemble.zig`
**L1 scope?** yes (HOT)
**Description:** Drives the measure → write → frame-assemble chain.
**Public surface:** `gpuAssembleFrameImpl`, `FramePreamble`,
`ChunkLayout`, `gpuFrameAssembleImpl`.

### srcVK/encode/encode_huff.zig
**CUDA source:** `src/encode/encode_huff.zig`
**L1 scope?** no (L2 stub bodies)
**L2 stub?** yes
**Description:** Symbols MUST exist; bodies return false so the
`opts.level >= 3` gate in fast_framed.zig skips them.
**Public surface:** `gpuEncodeHuffImpl`, `gpuEncodeOff16HuffImpl`,
`gpuEncodeLiteralsHuffImpl`, `gpuEncodeTokensHuffImpl`.

---

## Phase 11 — Top-level encoder + decoder + C ABI

### srcVK/encode/fast_framed.zig
**CUDA source:** `src/encode/fast_framed.zig`
**L1 scope?** yes
**Description:** L1 encode orchestrator. Owns the `opts.level >= 3`
Huffman gate (audit EXCEPTION 2 — preserve verbatim from CUDA).
**Public surface:** `compressFramedOne`.
**L2 gate position:** the existing `opts.level >= 3` block.

### srcVK/encode/streamlz_encoder.zig
**CUDA source:** `src/encode/streamlz_encoder.zig`
**L1 scope?** yes
**Description:** Public encoder facade.
**Public surface:** `CompressError`, `Options`, `compressBound`,
`compressFramed`, `compressFramedWithIo`.

### srcVK/encode/driver.zig
**CUDA source:** `src/encode/driver.zig`
**L1 scope?** yes
**Description:** Thin facade: owns `g_default` + `last_kernel_ns`;
re-exports every encode sub-module's public surface.
**Public surface:** all re-exports per the file head.

### srcVK/decode/streamlz_decoder.zig
**CUDA source:** `src/decode/streamlz_decoder.zig`
**L1 scope?** yes (host-walk path is the L1 hot one)
**L2 stub?** partial — `decompressFramedFromDevice` is L2 stubbed.
**Public surface:** `DecompressError`, `DecompressResult`,
`decompressFramed`, `decompressFramedThreaded`,
`decompressFramedFromDevice`, `DecompressContext`,
`decompressFrameInner`, `dispatchCompressedBlock`,
`buildChunkDescriptors`, `safe_space`, `max_content_size`.

### srcVK/streamlz_gpu.zig
**CUDA source:** `src/streamlz_gpu.zig`
**L1 scope?** yes
**Description:** C ABI implementation. Owns the `Context` struct, the
worker-thread runOnWorker helper, and the `mapCompressError` /
`mapDecompressError` mappers.
**Public surface:** C ABI exports — `slzStatusString`,
`slzVersionString`, `slzCreate`, `slzDestroy`, `slzCompressDefaultOpts`,
`slzDecompressDefaultOpts`, `slzCompressBound`,
`slzGetDecompressedSize`, `slzCompressHost`, `slzDecompressHost`,
`slzCompressAsync`, `slzDecompressAsync`, `slzGetLastTimings`,
`slzWaitAndGetLastTimings`.

---

## Phase 12 — CLI surface

### srcVK/cli/util.zig
**CUDA source:** `src/cli/util.zig`
**L1 scope?** yes
**Description:** Verbatim port (pure host).
**Public surface:** `Mode`, `Args`, `parseArgs`, `die`, `printVersion`,
`printUsage`, `deriveCompressOutput`, `deriveDecompressOutput`,
`readFile`, `requireInput`, `checkLevel`, `medianOrMean`, `fmtBytes`,
`fmtMbps`, `getMemInfo`.

### srcVK/cli/info.zig
**CUDA source:** `src/cli/info.zig`
**L1 scope?** yes
**Description:** Verbatim port; pure format-side frame dumper, no GPU
calls.
**Public surface:** `run`.

### srcVK/cli/decompress.zig
**CUDA source:** `src/cli/decompress.zig`
**L1 scope?** yes
**Public surface:** `run`.

### srcVK/cli/compress.zig
**CUDA source:** `src/cli/compress.zig`
**L1 scope?** yes
**Public surface:** `run`.

### srcVK/cli/bench_decompress.zig
**CUDA source:** `src/cli/bench_decompress.zig`
**L1 scope?** yes
**Public surface:** `run`.

### srcVK/cli/bench_compress.zig
**CUDA source:** `src/cli/bench_compress.zig`
**L1 scope?** yes
**Public surface:** `run`.

### srcVK/cli/bench_all.zig
**CUDA source:** `src/cli/bench_all.zig`
**L1 scope?** yes (L3-L5 sweeps fall back to level-2 codec on VK until
the Huff/chain L2 stubs land)
**Public surface:** `run`.

### srcVK/cli.zig
**CUDA source:** `src/cli.zig`
**L1 scope?** yes
**Public surface:** `run`.

### srcVK/main.zig
**CUDA source:** `src/main.zig`
**L1 scope?** yes
**Public surface:** `main`.

---

## Phase 13 — Tests (NEW per Exception 3)

### srcVK/test_runner_parallel.zig
**CUDA source:** `src/test_runner_parallel.zig`
**L1 scope?** yes (test infra)
**Description:** Verbatim port.

### srcVK/encode/gpu_roundtrip_tests.zig
**CUDA source:** `src/encode/gpu_roundtrip_tests.zig`
**L1 scope?** yes
**Description:** L1 round-trip test {} blocks.

### srcVK/tests/decoder_unit.zig
**CUDA source:** (none — Exception 3)
**Description:** Unit tests for streamlz_decoder.zig header parsers.

### srcVK/tests/encoder_unit.zig
**CUDA source:** (none — Exception 3)
**Description:** Unit tests for streamlz_encoder.zig compressBound +
Options validation.

### srcVK/tests/dispatch_unit.zig
**CUDA source:** (none — Exception 3)
**Description:** Unit tests for decode_dispatch.zig L2 gate +
runLzPipeline raw selection.

### srcVK/tests/kernel_conformance.zig
**CUDA source:** (none — Exception 3)
**Description:** Kernel-level byte-identity vs CUDA goldens.

### srcVK/tests/l1_decode_roundtrip.zig
**CUDA source:** (none — Exception 3)
**Description:** L1 decode round-trip on golden frames.

### srcVK/tests/l1_encode_roundtrip.zig
**CUDA source:** (none — Exception 3)
**Description:** L1 encode→decode round-trip on random + structured
payloads.

### srcVK/tests/cross_backend_roundtrip.zig
**CUDA source:** (none — Exception 3)
**Description:** CUDA↔VK cross-backend matrix.

### srcVK/tests/cli_smoke.zig
**CUDA source:** (none — Exception 3)
**Description:** End-to-end CLI smoke (spawn streamlz_vk.exe and round-trip).

---

## Open questions deferred to fleshout (audit Section H)

1. Subgroup-size policy on Intel iGPUs (require 32 vs parameterize).
2. `sm_count` analogue — fallback constant vs NV ext vs queue-family
   compute-unit count.
3. Async D2D entry points — implement device-side frame walk vs stub
   to `error.NotImplementedL2`.
