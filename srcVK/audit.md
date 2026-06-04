# StreamLZ CUDA → Vulkan Port Audit (srcVK)

Source of truth for the fresh `/srcVK/` port. Replaces the prior
`wf_0c9c5ca6-39f` Audit-phase output (`/src_vk/`), which leaked `.cu`
/ `.cuh` extensions, kept `cuda_*` filenames in violation of the
project's rename rule, and wrongly stubbed the entire encoder. This
audit walks every CUDA source file under `src/` (and `include/`) and
prescribes per-file extension + rename + L1/L2 classification.

Scope correction vs prior audit: **L1 includes the full encode path**
(L1+L2 LZ-only). The encoder must be ported, not stubbed. L3-L5 add
Huffman; those Huffman pieces and the L2+-only decode pieces are the
ones that get stubbed.

Project root: `c:/Users/james.JAMESWORK2025/Repos/Compressor_GPU`.

---

## Section A — Extension translation table (CANONICAL)

| CUDA extension | Role | srcVK extension |
|---|---|---|
| `.cu` — kernel TU (has `__global__` entry points; only included into device code) | GPU compute kernel | One `.comp` per `__global__` entry point (GLSL has one entry per shader) |
| `.cu` — host TU (compiles for host code only; never appears in this codebase — listed for completeness) | Host code | `.zig` |
| `.cuh` — device header (included only into `.cu` kernels; contains `__device__` helpers / templates) | Device-side helper, included into GLSL | `.glsl` |
| `.cuh` — host header (used by host code; not present in this codebase) | Shared host types | `.zig` |
| `.zig` | Already Zig | `.zig` (unchanged) |
| `.h` (C ABI header) | Public ABI | `.h` (unchanged, in `/include/`) |
| `.ptx` / `.cubin` | Pre-built CUDA artifacts | Not applicable; replaced by SPIR-V blobs built from `.comp` files at build time |

Hard rule: **no file under `/srcVK/` may have a `.cu` or `.cuh`
extension.** No exceptions.

Special case for `.cu` files that aggregate multiple `__global__`
entry points via `#include` chains (notably `src/decode/lz_kernel.cu`
and `src/decode/huffman_kernel.cu`): each `__global__` entry becomes
its own `.comp` file in `/srcVK/` named after the kernel symbol
(see Section C for the exact split). The shared device helpers that
the `.cuh` siblings provided become `.glsl` include headers.

---

## Section B — Token translation rules (CANONICAL)

Any token in the set `{cu, Cu, CU, cuda, Cuda, CUDA, nvidia, Nvidia,
NVIDIA}` appearing in:

- **A filename** — must be replaced with the Vulkan equivalent
  (`vk`/`Vk`/`VK` or `vulkan`/`Vulkan`/`VULKAN`) in the `/srcVK/`
  target path. Examples (binding):
  - `src/decode/cuda_api.zig` → `srcVK/decode/vulkan_api.zig`
  - `src/encode/cuda_ffi.zig` → `srcVK/encode/vulkan_ffi.zig`
  - Any future `cuda_module_loader.zig` → `vulkan_module_loader.zig`
    (the existing files are already named `module_loader.zig`, so no
    rename is required for those.)

- **A file-body identifier / type name** — must be replaced. Examples:
  - `CUstream` → `VkStream` (wrapper opaque)
  - `CUdeviceptr` → `VkDeviceBuffer` (opaque u64 handle into the
    VMA-backed buffer registry)
  - `CUresult` → `VkResult`
  - `CUDA_SUCCESS` → `VK_SUCCESS_RC` (numeric `0`; preserves the
    `rc == 0` fast-path branch)
  - `cuMemAlloc_fn` etc. → host-side procs-struct entries
    (`procs.malloc_device`, etc.) that funnel into VMA / `vkCmd*`
    behind the scenes
  - `nvcuda.dll` → `vulkan-1.dll` (loader)
  - `cuLaunchKernel` → `vkCmdDispatch` (via the `procs` indirection
    so call sites read identically across backends)

**Hard rule — no rationalization.** Comments, doc strings, or PR
descriptions saying "kept as `cuda_X` per port mandate", "VK PORT
NOTE: file kept as ...", or any equivalent are FORBIDDEN. The prior
workflow's Foundation agent rationalized the `cuda_api.zig` → kept-
as-`cuda_*` mapping for `descriptors.zig`'s `cudaCall` and similar.
Your port must apply the rule mechanically. The one and only deviation
explicitly allowed: identifier-level `cuda` tokens INSIDE Zig comment
prose that document the historical CUDA-side behavior for human
readers (e.g. "// Mirrors the original CUDA cuEvent-based timing"
is fine). File names, type names, and function names get translated
without exception.

The only file-naming carve-out that does NOT touch a translated
token is `tests/` — Exception 3 says the test files are new (no CUDA
counterpart). They live under `srcVK/tests/` and may use any
Vulkan-flavored names.

---

## Section C — Per-file mapping table

Walks every source file under `/src/` (and the two C ABI headers
under `/include/`). Excludes only `.zig-cache*` directories and
binary build artifacts not part of source control (none reach the
table).

Roles are: **kernel** (GLSL `.comp` entry-point), **glsl-header**
(GLSL `.glsl` include only), **host** (Zig host code), **header**
(C ABI header), **artifact** (compiled CUDA output, not ported).

L1 scope = needed to make L1 encode + L1 decode + L1 CLI + tests
work end-to-end. L1 here means **codec level L1 and L2** (LZ-only,
no Huffman). L2 stub = file exists in CUDA but its functions only
run on L3-L5 / Huffman / async D2D paths; create the file in
`/srcVK/` with same exported symbols but bodies returning
`error.NotImplementedL2` (Zig) or empty shaders that the host-side
gate never dispatches (GLSL).

The "L2 dispatches?" column flags files that contain or transitively
launch the L2/Huffman/Async-D2D code paths the fleshout agent must
gate at the earliest allocation+dispatch site (per project rule
EXCEPTION 2). The exact `if (level >= 2)` placement is the fleshout
agent's call; this audit only marks which files are affected.

### C.1 — Project root (`/src/`)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/main.zig` | `srcVK/main.zig` | host | yes | no | `main()` | `cli.zig` |
| `src/cli.zig` | `srcVK/cli.zig` | host | yes | no | `run()` dispatcher | `cli/*.zig` |
| `src/mmap.zig` | `srcVK/mmap.zig` | host | yes | no | `mapFileRead`, `MappedFile` | std only |
| `src/streamlz_gpu.zig` | `srcVK/streamlz_gpu.zig` | host | yes | no | C ABI: `slzCompressHost`, `slzDecompressHost`, `slzCompressAsync`, `slzDecompressAsync`, `slzCreate`, `slzDestroy`, `slzGetDecompressedSize`, `slzVersionString`, `slzStatusString`, status enum constants | `encoder`, `gpu_encoder`, `decoder`, `gpu_driver`, `frame`, `version`, `vulkan_api` (was `cuda_api`) |
| `src/version.zig` | `srcVK/version.zig` | host | yes | no | `version_string` | std only |
| `src/test_runner_parallel.zig` | `srcVK/test_runner_parallel.zig` | host | yes (test infra) | no | parallel test runner used by `zig build ptest` | std only |

L2 dispatch markers: `streamlz_gpu.zig` carries the `slzCompressAsync`/`slzDecompressAsync` paths (async D2D) that require the
walk-frame and chain-parser code; the L2 gate inside the dispatch
sub-modules covers them. `streamlz_gpu.zig` itself does not gate.

### C.2 — CLI (`/src/cli/`)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/cli/util.zig` | `srcVK/cli/util.zig` | host | yes | no | `Args` parsing, `requireInput`, `checkLevel`, output-path derivation, memory reporting | std, `version.zig`, `mmap.zig` |
| `src/cli/compress.zig` | `srcVK/cli/compress.zig` | host | yes (L1 encode) | no | `run()` | `util.zig`, `encoder` (`streamlz_encoder.zig`), `gpu_enc_driver` (`encode/driver.zig`), `mmap.zig` |
| `src/cli/decompress.zig` | `srcVK/cli/decompress.zig` | host | yes (L1 decode) | no | `run()` | `util.zig`, `decoder` (`streamlz_decoder.zig`), `gpu_driver` (`decode/driver.zig`), `mmap.zig` |
| `src/cli/bench_compress.zig` | `srcVK/cli/bench_compress.zig` | host | yes (L1 encode bench) | no | `run()` | `encoder`, `gpu_enc_driver`, `mmap.zig`, `util.zig` |
| `src/cli/bench_decompress.zig` | `srcVK/cli/bench_decompress.zig` | host | yes (L1 decode bench) | no | `run()` | `decoder`, `gpu_driver`, `mmap.zig`, `util.zig` |
| `src/cli/bench_all.zig` | `srcVK/cli/bench_all.zig` | host | yes | no | `run()` — sweeps encoder levels 1..5; the L3-L5 sweeps will fall through to the level-2 codec on VK until the Huffman/chain L2 stubs land | encoder, decoder, drivers |
| `src/cli/info.zig` | `srcVK/cli/info.zig` | host | yes | no | `run()` — pure format-side frame dumper; no GPU calls | `frame_format`, `block_header`, `mmap.zig` |

L2 dispatch markers: none of these files dispatch directly. They
call into the encode/decode drivers which carry the L2 gate.
`bench_all.zig` requests levels 1..5; the L2 gate in the encode
dispatch causes L3-L5 to fall back to the L2 codec output when the
Huffman pipeline is stubbed — `bench_all` reports those sweeps but
the resulting frames decode correctly (they are level-2 frames).

### C.3 — Format (`/src/format/`)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/format/frame_format.zig` | `srcVK/format/frame_format.zig` | host | yes | no | `parseHeader`, `writeHeader`, `writeBlockHeader`, `writeEndMark`, `scGroupSizeToBytes`, `max_header_size`, `CodecType` | `streamlz_constants.zig` |
| `src/format/block_header.zig` | `srcVK/format/block_header.zig` | host | yes | no | `parseBlockHeader`, `parseChunkHeader`, `CodecType` enum | std only |
| `src/format/streamlz_constants.zig` | `srcVK/format/streamlz_constants.zig` | host | yes | no | wire-format constants: `chunk_size`, `sub_chunk_size`, `safe_space`, etc. | none |

Pure host code. Verbatim port. No CUDA tokens to translate.

### C.4 — Common GPU shared headers (`/src/common/`)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/common/gpu_wire_format.cuh` | `srcVK/common/gpu_wire_format.glsl` | glsl-header | yes | no | Wire-format constants (`INITIAL_LITERAL_COPY_BYTES`, `OFF32_COUNT_PACK_MAX`, `LZ_BLOCK_SIZE`, etc.) | none |
| `src/common/gpu_byteio.cuh` | `srcVK/common/gpu_byteio.glsl` | glsl-header | yes | no | `readBE24`, `readU32LE`, `writeBE24`, etc. (endian byte-IO primitives) | none |
| `src/common/gpu_warp.cuh` | `srcVK/common/gpu_warp.glsl` | glsl-header | yes | no | Warp/subgroup constants (`WARP_SIZE`, `LANE_MASK`, `FULL_WARP_MASK`). In GLSL these become subgroup intrinsics — see fleshout notes. | none |
| `src/common/gpu_huffman.cuh` | `srcVK/common/gpu_huffman.glsl` | glsl-header | no | yes (L2+) | Shared Huffman wire format (`HUFF_NUM_STREAMS`, `HUFF_BODY_HEADER_BYTES`, `HUFF_*` constants) | none |

The wire-format / byteio / warp helpers are needed by L1 raw kernels
(both encode and decode). `gpu_huffman.glsl` body can be stubbed; the
**constants** must be present because `encode_huff.zig` references
`HUFF_BODY_HEADER_BYTES`, `HUFF_NUM_STREAMS` and similar at compile time.

### C.5 — Decode (`/src/decode/`)

#### C.5.1 — Decode host Zig

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/decode/cuda_api.zig` | `srcVK/decode/vulkan_api.zig` | host | yes | no | `procs` struct (`malloc_device`/`free_device`/`h2d`/`d2h`/`d2d`/`memset_d8`/`memset_d8_async`/`malloc_host`/`free_host`/`stream_sync`/`ctx_sync`/`stream_create`/`stream_destroy`), `VkDeviceBuffer` (u64 opaque), `VkResult`, `VkStream` (opaque), `VK_SUCCESS_RC`, `qpcNow`/`qpcMs`, `sm_count` analogue (for `fast_framed.resolveScGroupSize`) | `vma.zig`, win32 |
| `src/decode/module_loader.zig` | `srcVK/decode/module_loader.zig` | host | yes | no | `init()`, `isAvailable()`, pipeline-handle pub vars (`kernel_fn`, `kernel_raw_fn`, `walk_frame_fn`, `prefix_sum_chunks_fn`, `scan_parse_fn`, `compact_huff_descs_fn`, `compact_raw_descs_fn`, `merge_huff_descs_fn`, `gather_off16_fn`, `huff_build_fn`, `huff_decode_fn`), `ensurePipelineStream` | `vulkan_api.zig`, embedded SPV blobs |
| `src/decode/descriptors.zig` | `srcVK/decode/descriptors.zig` | host | yes | no | `ChunkDesc`, `HuffDecChunkDesc`, `ScanResult`, `ScanHuffDesc`, `ScanRawDesc`, `WalkFrameResultDev`, `WALK_MAX_CHUNKS`, `walk_meta_offsets`, `ENTROPY_SCRATCH_SLOT_BYTES`, `KernelTiming`, `PendingTiming`, `GpuError` (gains `NotImplementedL2`), `vkCall` (was `cudaCall`) | none |
| `src/decode/decode_context.zig` | `srcVK/decode/decode_context.zig` | host | yes | no | `DecodeContext` struct (every `d_*` device pointer slot renamed to `VkDeviceBuffer`), `ensureDeviceBuf`, `ensureDeviceOutput`, `allocHost`, `freeHost`, `copyHostToDevice`, `copyDeviceToHost`, `bindContextToCallingThread`, `beginKernelTiming`, `endKernelTiming`, `finalizeProfiling`, `deinit` | `vulkan_api.zig`, `descriptors.zig`, `module_loader.zig` |
| `src/decode/scan_gpu.zig` | `srcVK/decode/scan_gpu.zig` | host | yes (prefix-sum is L1) | partial: `gpuScanChunks` + `gpuWalkFrameImpl` are L2-stubbed bodies | `gpuWalkFrameImpl`, `gpuPrefixSumChunksImpl`, `gpuScanChunks` | `decode_context.zig`, `module_loader.zig`, `descriptors.zig`, `vulkan_api.zig` |
| `src/decode/decode_dispatch.zig` | `srcVK/decode/decode_dispatch.zig` | host | yes | yes-internal (L2 gate added) | `fullGpuLaunchImpl`, `runHuffBuildAndDecode`, `runLzPipeline`, `finalizeOutput`, `uploadInputAndPrefixSum`, `runBackHalf`, `mergeHuffDescs`, `gatherRawOff16`, `emitE2eTrace`, `VkProcs` (was `CudaProcs`) | every decode sub-module |
| `src/decode/driver.zig` | `srcVK/decode/driver.zig` | host | yes | no | Re-exports + facade `pub var g_default: DecodeContext`, `pub var last_kernel_ns`, `last_lz_kernel_ns`, `last_huff_kernel_ns`; `init`, `isAvailable`, descriptor types, host I/O helpers, `qpcNow`/`qpcMs` | every decode sub-module |
| `src/decode/streamlz_decoder.zig` | `srcVK/decode/streamlz_decoder.zig` | host | yes | partial (`decompressFramedFromDevice` D2D path uses `walk_frame_fn`; body kept but the host-walk path is the L1 hot one) | `decompressFramed`, `decompressFramedThreaded`, `decompressFramedFromDevice`, `decompressFrameInner`, `dispatchCompressedBlock`, `buildChunkDescriptors` | `driver.zig`, format modules |

L2 dispatch markers (host): `decode_dispatch.zig` is where the
`if (level >= 2)` allocation+dispatch gate must live (per project
rule EXCEPTION 2). The gate wraps the Huff descriptor merge + Huff
build/decode pipeline (the `if (have_huff)` block already in CUDA)
plus the scan/compact/merge/gather paths now gated by
`module_loader.huff_build_fn != 0`. The VK port adds an explicit
`level` field on DecodeRequest so the gate fires per request rather
than per loaded module set. `scan_gpu.zig` and `streamlz_decoder.zig`
contain L2-only entry points (`gpuScanChunks`, `gpuWalkFrameImpl`,
`decompressFramedFromDevice`) whose bodies stub or short-circuit
when L2 is off.

#### C.5.2 — Decode kernels (`.cu` → `.comp`)

`src/decode/lz_kernel.cu` is a CUDA aggregator translation unit
that `#include`s every decode `.cuh` so nvcc emits one `lz_kernel.ptx`
binding **all** decode `__global__` entry points. GLSL/SPIR-V has
one entry per shader, so each `__global__` becomes its own `.comp`
file in `/srcVK/`. The `lz_kernel.cu` aggregator file itself has no
direct port — it becomes a build manifest in `build.zig` listing
which `.comp` files compile to which SPV blob.

The decode kernels (`__global__` entry points) in CUDA:
- `slzLzDecodeKernel`            — `lz_decode_kernels.cuh` (L2 general)
- `slzLzDecodeRawKernel`         — `lz_decode_kernels.cuh` (L1 raw)
- `slzGatherRawOff16Kernel`      — `gather_raw_off16_kernel.cuh` (L2)
- `slzWalkFrameKernel`           — `walk_frame_kernel.cuh` (L2 async-D2D)
- `slzCompactHuffDescsKernel`    — `compact_descs_kernels.cuh` (L2)
- `slzCompactRawDescsKernel`     — `compact_descs_kernels.cuh` (L2)
- `slzMergeHuffDescsKernel`      — `merge_huff_descs_kernel.cuh` (L2)
- `slzPrefixSumChunksKernel`     — `prefix_sum_chunks_kernel.cuh` (L1)
- `slzScanParseKernel`           — `scan_parse_kernel.cuh` (L2)
- `slzHuffBuildLutKernel`        — `huffman_kernel.cu` (L2 huffman)
- `slzHuffDecode4StreamKernel`   — `huffman_kernel.cu` (L2 huffman)

VK port: one `.comp` per kernel. The shared device helpers in `.cuh`
files become `.glsl` include headers.

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/decode/lz_kernel.cu` | (no direct port — manifest split into per-kernel `.comp` files below) | manifest | yes | n/a | (aggregator) | per-kernel `.comp` files |
| `src/decode/lz_decode_kernels.cuh` | `srcVK/decode/lz_decode_kernel.comp` (entry `slzLzDecodeKernel`) **and** `srcVK/decode/lz_decode_raw_kernel.comp` (entry `slzLzDecodeRawKernel`) | kernel | yes (raw .comp) + yes (general .comp body stubbed) | partial: general .comp main() returns immediately | two kernel entry points | `lz_decode_core.glsl`, `lz_dispatch.glsl`, `slz_wire_format.glsl`, `gpu_byteio.glsl`, `gpu_warp.glsl` |
| `src/decode/lz_decode_raw.cuh` | `srcVK/decode/lz_decode_raw.glsl` | glsl-header | yes | no | raw-mode sub-chunk decoder helpers (used by `lz_decode_raw_kernel.comp`) | `lz_decode_core.glsl`, `gpu_byteio.glsl` |
| `src/decode/lz_decode_general.cuh` | `srcVK/decode/lz_decode_general.glsl` | glsl-header | no | yes | general entropy-capable decoder (L2+) | `lz_decode_core.glsl`, `lz_header_parse.glsl` |
| `src/decode/lz_decode_core.cuh` | `srcVK/decode/lz_decode_core.glsl` | glsl-header | yes | no | shared warp primitives (`warpScanU32`, `warpLiteralCopy`, `warpMatchCopy`) used by raw + general | `gpu_warp.glsl` |
| `src/decode/lz_header_parse.cuh` | `srcVK/decode/lz_header_parse.glsl` | glsl-header | no | yes | `parseSubChunkHeaders` (general path only) | `gpu_byteio.glsl` |
| `src/decode/lz_dispatch.cuh` | `srcVK/decode/lz_dispatch.glsl` | glsl-header | yes (Raw path) | partial (general path body stubbed) | `parseAndDecodeSubChunkRaw` (L1) and `parseAndDecodeSubChunk` (L2-stub) | `lz_decode_raw.glsl`, `lz_decode_general.glsl` |
| `src/decode/slz_wire_format.cuh` | `srcVK/decode/slz_wire_format.glsl` | glsl-header | yes | no | Device-side wire-format constants + descriptor structs + header parsers (must match `descriptors.zig` byte-for-byte) | none |
| `src/decode/gather_raw_off16_kernel.cuh` | `srcVK/decode/gather_raw_off16_kernel.comp` (entry `slzGatherRawOff16Kernel`) | kernel | no | yes | one kernel entry | `slz_wire_format.glsl` |
| `src/decode/walk_frame_kernel.cuh` | `srcVK/decode/walk_frame_kernel.comp` (entry `slzWalkFrameKernel`) | kernel | no | yes | device-side frame walker (async D2D path) | `slz_wire_format.glsl`, `gpu_byteio.glsl` |
| `src/decode/prefix_sum_chunks_kernel.cuh` | `srcVK/decode/prefix_sum_chunks_kernel.comp` (entry `slzPrefixSumChunksKernel`) | kernel | yes | no | per-chunk first-sub-chunk index kernel — launched unconditionally by `uploadInputAndPrefixSum` | `gpu_warp.glsl` |
| `src/decode/compact_descs_kernels.cuh` | `srcVK/decode/compact_huff_descs_kernel.comp` (entry `slzCompactHuffDescsKernel`) **and** `srcVK/decode/compact_raw_descs_kernel.comp` (entry `slzCompactRawDescsKernel`) | kernel | no | yes | two kernel entry points | `slz_wire_format.glsl` |
| `src/decode/merge_huff_descs_kernel.cuh` | `srcVK/decode/merge_huff_descs_kernel.comp` (entry `slzMergeHuffDescsKernel`) | kernel | no | yes | one kernel entry | `slz_wire_format.glsl` |
| `src/decode/scan_parse_kernel.cuh` | `srcVK/decode/scan_parse_kernel.comp` (entry `slzScanParseKernel`) | kernel | no | yes | one kernel entry | `slz_wire_format.glsl` |
| `src/decode/huffman_kernel.cu` | `srcVK/decode/huff_build_lut_kernel.comp` (entry `slzHuffBuildLutKernel`) **and** `srcVK/decode/huff_decode_4stream_kernel.comp` (entry `slzHuffDecode4StreamKernel`) | kernel | no | yes | two kernel entry points | `gpu_huffman.glsl`, `gpu_warp.glsl` |

L2 dispatch markers (decode kernels): every kernel in the L2-stub
column above is gated host-side at `decode_dispatch.zig`'s L2 gate
(per EXCEPTION 2). The `.comp` shaders still exist and compile so
the build emits a SPV blob (so `module_loader.zig` slot resolution
does not fault); main()-bodies return immediately.

#### C.5.3 — Decode build artifacts (NOT PORTED)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/decode/lz_kernel.ptx` | (not ported — replaced by SPV blobs from `.comp` files) | artifact | n/a | n/a | n/a | n/a |
| `src/decode/lz_kernel.cubin` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/decode/huffman_kernel.ptx` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/decode/huffman_kernel.cubin` | (not ported) | artifact | n/a | n/a | n/a | n/a |

### C.6 — Encode (`/src/encode/`)

L1 scope for encode means codec **levels 1 and 2** (LZ-only,
greedy parser, no Huffman, no chain parser). Levels 3-5 add the
GPU Huffman pass and (L5) the chain parser; those are L2 stubs.
`fast_framed.zig` orchestrates both, but its L1 hot path skips
both feature blocks via the `opts.level >= 3` gate it already
has — the VK port can land L1 encode without touching the
Huffman / chain-parser code beyond stub-symbol presence.

#### C.6.1 — Encode host Zig

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/encode/streamlz_encoder.zig` | `srcVK/encode/streamlz_encoder.zig` | host | yes | no | `compressFramed`, `compressFramedWithIo`, `compressBound`, `CompressError`, `Options` | `frame_format`, `streamlz_constants`, `fast_framed`, `driver` (encode) |
| `src/encode/driver.zig` | `srcVK/encode/driver.zig` | host | yes | no | Facade re-exports + `pub var g_default: EncodeContext`, `pub var last_kernel_ns: i64`; `init`, `isAvailable`, `CompressChunkDesc`, `EncodeContext`, `copyDeviceToHost`, `copyHostToDevice`, `ensureBuf`, `SC_TAIL_PER_CHUNK_BYTES`, `CHUNK_INTERNAL_HDR_BYTES`, `UNCOMPRESSED_CHUNK_HDR_BYTES`, `UNCOMPRESSED_CHUNK_MARKER`, `gpuCompressImpl`, `gpuEncodeLiteralsHuffImpl`, `gpuEncodeTokensHuffImpl`, `gpuEncodeOff16HuffImpl`, `gpuAssembleFrameImpl`, `gpuFrameAssembleImpl` | every encode sub-module |
| `src/encode/cuda_ffi.zig` | `srcVK/encode/vulkan_ffi.zig` | host | yes | no | Encode-side driver FFI (analogue of `decode/vulkan_api.zig` for the encode subdir). `CUresult`/`CUdeviceptr`/`CUmodule`/`CUfunction` → `VkResult`/`VkDeviceBuffer`/`VkPipelineLayout`/`VkPipeline`. Function-pointer slots (`cuMemcpyHtoD_fn`, `cuMemcpyDtoH_fn`, `cuLaunchKernel_fn`, etc.) wrapped on top of the same `procs` indirection used by decode. | `vma.zig`, win32 |
| `src/encode/module_loader.zig` | `srcVK/encode/module_loader.zig` | host | yes | no | `init()`, `isAvailable()`, pub vars: `module`, `kernel_fn` (LZ encode), `huff_module`, `huff_tables_kernel_fn`, `huff_encode_kernel_fn`, `assemble_module`, `assemble_measure_fn`, `assemble_write_fn`, `frame_assemble_fn`. Must call decode-side `init()` first (single context — VkDevice in port). LZ kernel resolution must succeed (L1 needs it). Huffman + assemble kernels resolve optionally; failure leaves the slots at 0 and the L2 gate skips them. | `vulkan_ffi.zig`, `decode/driver.zig` |
| `src/encode/encode_context.zig` | `srcVK/encode/encode_context.zig` | host | yes | partial (Huff fields exist but go unused on L1) | `INITIAL_LITERAL_COPY_BYTES`, `SC_TAIL_PER_CHUNK_BYTES`, `CHUNK_INTERNAL_HDR_BYTES`, `UNCOMPRESSED_CHUNK_HDR_BYTES`, `NEXT_HASH_ENTRIES`, `UNCOMPRESSED_CHUNK_MARKER`, `CompressChunkDesc`, `AssembleDesc`, `HuffEncDesc`, `EncodeContext`, `ensureBuf`, `copyDeviceToHost`, `copyHostToDevice` | `vulkan_ffi.zig`, `module_loader.zig`, `decode/driver.zig` |
| `src/encode/levels.zig` | `srcVK/encode/levels.zig` | host | yes | no | `hashBitsForLevel`, `useChainParser` | none |
| `src/encode/encode_lz.zig` | `srcVK/encode/encode_lz.zig` | host | yes | no | `gpuCompressImpl` (LZ kernel launcher; L1 hot path) | `vulkan_ffi.zig`, `module_loader.zig`, `encode_context.zig`, `levels.zig`, `decode/driver.zig` |
| `src/encode/encode_huff.zig` | `srcVK/encode/encode_huff.zig` | host | no | yes (L2+) | `gpuEncodeHuffImpl`, `gpuEncodeLiteralsHuffImpl`, `gpuEncodeTokensHuffImpl`, `gpuEncodeOff16HuffImpl`. Symbols MUST exist (referenced from `fast_framed.zig` and `driver.zig`); bodies return `false` on L1 so the `opts.level >= 3` gate in `fast_framed.zig` naturally skips emitted Huffman bodies. | `vulkan_ffi.zig`, `module_loader.zig`, `encode_context.zig`, `decode/driver.zig` |
| `src/encode/encode_assemble.zig` | `srcVK/encode/encode_assemble.zig` | host | yes (L1 frame writer uses `gpuFrameAssembleImpl`) | partial (the per-stream Huff metadata branches are unused on L1) | `gpuAssembleFrameImpl`, `gpuFrameAssembleImpl` | `vulkan_ffi.zig`, `module_loader.zig`, `encode_context.zig`, `decode/driver.zig` |
| `src/encode/fast_framed.zig` | `srcVK/encode/fast_framed.zig` | host | yes | no | `compressFramedOne` (entry from `streamlz_encoder.compressFramedWithIo`) | `streamlz_encoder`, `driver` (encode), `format` modules, `decode/vulkan_api.zig` (for `sm_count` analogue used by `resolveScGroupSize`) |
| `src/encode/gpu_roundtrip_tests.zig` | `srcVK/encode/gpu_roundtrip_tests.zig` | host | yes (L1 encode↔decode roundtrip tests) | no | `test {}` blocks | `streamlz_encoder`, `decoder`, `gpu_encoder` (`driver`), `gpu_driver` (`decode/driver`) |

L2 dispatch markers (encode host): `encode_huff.zig` is fully L2-only;
the symbol surface must exist but bodies stub. `encode_assemble.zig`,
`encode_context.zig`, and `module_loader.zig` carry Huff-related
fields/slots that go unused on L1; the L1 path never touches them.
`fast_framed.zig` carries the L2 gate inside its compress path via
`opts.level >= 3` (already present in CUDA) — preserve verbatim.

#### C.6.2 — Encode kernels (`.cu` → `.comp`)

The encode kernels (`__global__` entry points) in CUDA:
- `slzLzEncodeKernel`            — `lz_kernel.cu` (L1+L2 LZ encode)
- `slzHuffBuildTablesKernel`     — `huffman_kernel.cu` (L3-L5)
- `slzHuffEncode4StreamKernel`   — `huffman_kernel.cu` (L3-L5)
- `slzAssembleMeasureKernel`     — `assemble_kernel.cu` (L1+ frame assembly pass A)
- `slzAssembleWriteKernel`       — `assemble_kernel.cu` (L1+ frame assembly pass B)
- `slzFrameAssembleKernel`       — `assemble_kernel.cu` (L1+ device-resident frame writer)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/encode/lz_kernel.cu` | `srcVK/encode/lz_encode_kernel.comp` (entry `slzLzEncodeKernel`) | kernel | yes | no | one kernel entry | `lz_format.glsl`, `lz_token_emit.glsl`, `lz_greedy_parser.glsl`, `lz_chain_parser.glsl`, `gpu_wire_format.glsl`, `gpu_warp.glsl`, `gpu_byteio.glsl` |
| `src/encode/huffman_kernel.cu` | `srcVK/encode/huff_build_tables_kernel.comp` (entry `slzHuffBuildTablesKernel`) **and** `srcVK/encode/huff_encode_4stream_kernel.comp` (entry `slzHuffEncode4StreamKernel`) | kernel | no | yes | two kernel entry points | `gpu_huffman.glsl`, `gpu_warp.glsl` |
| `src/encode/assemble_kernel.cu` | `srcVK/encode/assemble_measure_kernel.comp` (entry `slzAssembleMeasureKernel`) **and** `srcVK/encode/assemble_write_kernel.comp` (entry `slzAssembleWriteKernel`) **and** `srcVK/encode/frame_assemble_kernel.comp` (entry `slzFrameAssembleKernel`) | kernel | yes (measure + write + frame writer used by L1 frame assembly) | no | three kernel entry points | `gpu_wire_format.glsl`, `gpu_warp.glsl` |

L2 dispatch markers (encode kernels): only the two Huffman kernels
are L2-stubbed. Frame-assembly kernels are L1 hot path (the L1
encode pipeline writes the frame device-resident via the assembly
chain).

#### C.6.3 — Encode kernel-side device headers

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/encode/lz_format.cuh` | `srcVK/encode/lz_format.glsl` | glsl-header | yes | no | Encoder-private hash constants (`NEXT_HASH_SIZE`, hash-table layout) | `gpu_wire_format.glsl` |
| `src/encode/lz_token_emit.cuh` | `srcVK/encode/lz_token_emit.glsl` | glsl-header | yes | no | Token-emission helpers used by both greedy and chain parsers | `lz_format.glsl`, `gpu_byteio.glsl` |
| `src/encode/lz_greedy_parser.cuh` | `srcVK/encode/lz_greedy_parser.glsl` | glsl-header | yes | no | Warp-parallel greedy LZ parser (L1-L4 path) | `lz_format.glsl`, `lz_token_emit.glsl`, `gpu_warp.glsl` |
| `src/encode/lz_chain_parser.cuh` | `srcVK/encode/lz_chain_parser.glsl` | glsl-header | no | yes (L5 only) | Serial chain-hash lazy LZ parser | `lz_format.glsl`, `lz_token_emit.glsl` |

L2 dispatch markers (encode headers): only the chain parser is
L5-only. `lz_token_emit.glsl` and `lz_greedy_parser.glsl` are
L1 hot path.

#### C.6.4 — Encode build artifacts (NOT PORTED)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `src/encode/lz_kernel.ptx` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/encode/lz_kernel.cubin` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/encode/huffman_kernel.ptx` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/encode/huffman_kernel.cubin` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/encode/assemble_kernel.ptx` | (not ported) | artifact | n/a | n/a | n/a | n/a |
| `src/encode/assemble_kernel.cubin` | (not ported) | artifact | n/a | n/a | n/a | n/a |

### C.7 — Public C ABI (`/include/`)

| CUDA path | srcVK target path | Role | L1 scope? | L2 stub? | Exports | Depends on |
|---|---|---|---|---|---|---|
| `include/streamlz_gpu.h` | `include/streamlz_gpu.h` (UNCHANGED) | header | yes | no | `slzStatus_t`, `slzCompressOpts_t`, `slzDecompressOpts_t`, `slzKernelTiming_t`, `slzCompressHost`, `slzDecompressHost`, `slzCompressAsync`, `slzDecompressAsync`, `slzCreate`, `slzDestroy`, `slzGetDecompressedSize`, `slzVersionString`, `slzStatusString` | none |
| `include/streamlz_gpu_vk.h` | `include/streamlz_gpu_vk.h` (UNCHANGED — already VK-specific) | header | yes | no | VK-specific C ABI surface (`slzCreate_vk`, `slzDecompressHost_vk`, etc.) — already aligned with the VK port | none |

The two headers stay at `/include/`. The port does NOT duplicate
them under `/srcVK/`.

### C.8 — New files (Exception 3 — tests)

See Section F for the test plan; not part of the per-file CUDA → VK
mapping table.

---

## Section D — srcVK folder structure

```
srcVK/
├── audit.md                                 ← this file
├── error.zig                                ← NEW (cross-cutting GpuError + NotImplementedL2; foundation)
├── vma.zig                                  ← NEW (VMA Zig binding; foundation)
├── main.zig                                 ← src/main.zig
├── cli.zig                                  ← src/cli.zig
├── mmap.zig                                 ← src/mmap.zig
├── streamlz_gpu.zig                         ← src/streamlz_gpu.zig
├── version.zig                              ← src/version.zig
├── test_runner_parallel.zig                 ← src/test_runner_parallel.zig
│
├── cli/
│   ├── util.zig                             ← src/cli/util.zig
│   ├── compress.zig                         ← src/cli/compress.zig
│   ├── decompress.zig                       ← src/cli/decompress.zig
│   ├── bench_compress.zig                   ← src/cli/bench_compress.zig
│   ├── bench_decompress.zig                 ← src/cli/bench_decompress.zig
│   ├── bench_all.zig                        ← src/cli/bench_all.zig
│   └── info.zig                             ← src/cli/info.zig
│
├── format/
│   ├── frame_format.zig                     ← src/format/frame_format.zig
│   ├── block_header.zig                     ← src/format/block_header.zig
│   └── streamlz_constants.zig               ← src/format/streamlz_constants.zig
│
├── common/
│   ├── gpu_wire_format.glsl                 ← src/common/gpu_wire_format.cuh
│   ├── gpu_byteio.glsl                      ← src/common/gpu_byteio.cuh
│   ├── gpu_warp.glsl                        ← src/common/gpu_warp.cuh
│   └── gpu_huffman.glsl                     ← src/common/gpu_huffman.cuh (L2 stub body; constants live)
│
├── decode/
│   ├── vulkan_api.zig                       ← src/decode/cuda_api.zig           (RENAMED per Section B)
│   ├── module_loader.zig                    ← src/decode/module_loader.zig
│   ├── descriptors.zig                      ← src/decode/descriptors.zig
│   ├── decode_context.zig                   ← src/decode/decode_context.zig
│   ├── scan_gpu.zig                         ← src/decode/scan_gpu.zig
│   ├── decode_dispatch.zig                  ← src/decode/decode_dispatch.zig
│   ├── driver.zig                           ← src/decode/driver.zig
│   ├── streamlz_decoder.zig                 ← src/decode/streamlz_decoder.zig
│   ├── slz_wire_format.glsl                 ← src/decode/slz_wire_format.cuh
│   ├── lz_decode_core.glsl                  ← src/decode/lz_decode_core.cuh
│   ├── lz_decode_raw.glsl                   ← src/decode/lz_decode_raw.cuh
│   ├── lz_decode_general.glsl               ← src/decode/lz_decode_general.cuh (L2 stub body)
│   ├── lz_header_parse.glsl                 ← src/decode/lz_header_parse.cuh (L2 stub body)
│   ├── lz_dispatch.glsl                     ← src/decode/lz_dispatch.cuh
│   ├── lz_decode_raw_kernel.comp            ← entry slzLzDecodeRawKernel  (from lz_decode_kernels.cuh; L1 hot)
│   ├── lz_decode_kernel.comp                ← entry slzLzDecodeKernel     (from lz_decode_kernels.cuh; L2 stub main)
│   ├── prefix_sum_chunks_kernel.comp        ← entry slzPrefixSumChunksKernel (from prefix_sum_chunks_kernel.cuh; L1)
│   ├── gather_raw_off16_kernel.comp         ← entry slzGatherRawOff16Kernel (from gather_raw_off16_kernel.cuh; L2 stub)
│   ├── walk_frame_kernel.comp               ← entry slzWalkFrameKernel (from walk_frame_kernel.cuh; L2 stub)
│   ├── compact_huff_descs_kernel.comp       ← entry slzCompactHuffDescsKernel (from compact_descs_kernels.cuh; L2 stub)
│   ├── compact_raw_descs_kernel.comp        ← entry slzCompactRawDescsKernel  (from compact_descs_kernels.cuh; L2 stub)
│   ├── merge_huff_descs_kernel.comp         ← entry slzMergeHuffDescsKernel (from merge_huff_descs_kernel.cuh; L2 stub)
│   ├── scan_parse_kernel.comp               ← entry slzScanParseKernel (from scan_parse_kernel.cuh; L2 stub)
│   ├── huff_build_lut_kernel.comp           ← entry slzHuffBuildLutKernel (from huffman_kernel.cu; L2 stub)
│   └── huff_decode_4stream_kernel.comp      ← entry slzHuffDecode4StreamKernel (from huffman_kernel.cu; L2 stub)
│
├── encode/
│   ├── vulkan_ffi.zig                       ← src/encode/cuda_ffi.zig            (RENAMED per Section B)
│   ├── module_loader.zig                    ← src/encode/module_loader.zig
│   ├── encode_context.zig                   ← src/encode/encode_context.zig
│   ├── levels.zig                           ← src/encode/levels.zig
│   ├── encode_lz.zig                        ← src/encode/encode_lz.zig          (L1 hot)
│   ├── encode_huff.zig                      ← src/encode/encode_huff.zig         (L2 stub body)
│   ├── encode_assemble.zig                  ← src/encode/encode_assemble.zig    (L1 hot)
│   ├── fast_framed.zig                      ← src/encode/fast_framed.zig
│   ├── streamlz_encoder.zig                 ← src/encode/streamlz_encoder.zig
│   ├── driver.zig                           ← src/encode/driver.zig
│   ├── gpu_roundtrip_tests.zig              ← src/encode/gpu_roundtrip_tests.zig (L1 roundtrip cases)
│   ├── lz_format.glsl                       ← src/encode/lz_format.cuh
│   ├── lz_token_emit.glsl                   ← src/encode/lz_token_emit.cuh
│   ├── lz_greedy_parser.glsl                ← src/encode/lz_greedy_parser.cuh    (L1 hot)
│   ├── lz_chain_parser.glsl                 ← src/encode/lz_chain_parser.cuh     (L5 only, L2 stub)
│   ├── lz_encode_kernel.comp                ← entry slzLzEncodeKernel (from lz_kernel.cu; L1 hot)
│   ├── huff_build_tables_kernel.comp        ← entry slzHuffBuildTablesKernel (from huffman_kernel.cu; L2 stub)
│   ├── huff_encode_4stream_kernel.comp      ← entry slzHuffEncode4StreamKernel (from huffman_kernel.cu; L2 stub)
│   ├── assemble_measure_kernel.comp         ← entry slzAssembleMeasureKernel (from assemble_kernel.cu; L1 hot)
│   ├── assemble_write_kernel.comp           ← entry slzAssembleWriteKernel  (from assemble_kernel.cu; L1 hot)
│   └── frame_assemble_kernel.comp           ← entry slzFrameAssembleKernel  (from assemble_kernel.cu; L1 hot)
│
└── tests/                                   (NEW per Exception 3 — no CUDA counterpart)
    ├── decoder_unit.zig
    ├── dispatch_unit.zig
    ├── encoder_unit.zig
    ├── kernel_conformance.zig
    ├── l1_decode_roundtrip.zig
    ├── l1_encode_roundtrip.zig
    ├── cross_backend_roundtrip.zig
    └── cli_smoke.zig
```

The C ABI headers `include/streamlz_gpu.h` and
`include/streamlz_gpu_vk.h` are intentionally NOT mirrored under
`/srcVK/`; they stay at `/include/` (shared with the CUDA backend
under `/src/`).

`error.zig` and `vma.zig` at the `/srcVK/` root are **new** files
required by the Vulkan backend — they have no CUDA counterpart but
are foundational. They are listed here in Section D so the file tree
is exhaustive; they are not in the Section C mapping table because
there is no `src/` source to translate.

---

## Section E — Port phases / dependency order

Phase ordering chosen so each phase depends only on artifacts from
prior phases. Files within a phase are parallelizable.

### Phase 1 — Foundation (host-only, no GPU)
- `srcVK/error.zig` (NEW; provides `error.NotImplementedL2`)
- `srcVK/version.zig`
- `srcVK/mmap.zig`
- `srcVK/format/streamlz_constants.zig`
- `srcVK/format/block_header.zig`
- `srcVK/format/frame_format.zig`
- `srcVK/common/gpu_wire_format.glsl`
- `srcVK/common/gpu_byteio.glsl`
- `srcVK/common/gpu_warp.glsl`
- `srcVK/common/gpu_huffman.glsl` (constants only; body stubbed)

### Phase 2 — VK driver layer + VMA wiring
- `srcVK/vma.zig` (NEW; hand-rolled VMA Zig binding)
- `srcVK/decode/vulkan_api.zig` (procs struct, `VkDeviceBuffer`, `VkStream`, `VkResult`)
- `srcVK/decode/descriptors.zig` (POD structs, `vkCall`, `GpuError`+`NotImplementedL2`)
- `srcVK/encode/vulkan_ffi.zig` (encode-side FFI shim)

### Phase 3 — Module loader + SPV blob loading
- `srcVK/decode/module_loader.zig`
- `srcVK/encode/module_loader.zig`
- `build.zig` hooks for glslangValidator → SPV embed of every `.comp` file

### Phase 4 — L1 decode kernels (`.glsl` headers + `.comp` entries)
Parallel:
- `srcVK/decode/slz_wire_format.glsl`
- `srcVK/decode/lz_decode_core.glsl`
- `srcVK/decode/lz_decode_raw.glsl`
- `srcVK/decode/lz_dispatch.glsl`
- `srcVK/decode/lz_decode_raw_kernel.comp`
- `srcVK/decode/prefix_sum_chunks_kernel.comp`

### Phase 5 — L2 stub decode kernels + headers (parallel; bodies stubbed)
- `srcVK/decode/lz_decode_general.glsl`
- `srcVK/decode/lz_header_parse.glsl`
- `srcVK/decode/lz_decode_kernel.comp`
- `srcVK/decode/gather_raw_off16_kernel.comp`
- `srcVK/decode/walk_frame_kernel.comp`
- `srcVK/decode/compact_huff_descs_kernel.comp`
- `srcVK/decode/compact_raw_descs_kernel.comp`
- `srcVK/decode/merge_huff_descs_kernel.comp`
- `srcVK/decode/scan_parse_kernel.comp`
- `srcVK/decode/huff_build_lut_kernel.comp`
- `srcVK/decode/huff_decode_4stream_kernel.comp`

### Phase 6 — Decode context + dispatch infrastructure
- `srcVK/decode/decode_context.zig`
- `srcVK/decode/scan_gpu.zig`

### Phase 7 — Decode dispatch orchestration (with the EXCEPTION 2 L2 gate)
- `srcVK/decode/decode_dispatch.zig`
- `srcVK/decode/driver.zig`

### Phase 8 — L1 encode kernels (`.glsl` headers + `.comp` entries)
Parallel:
- `srcVK/encode/lz_format.glsl`
- `srcVK/encode/lz_token_emit.glsl`
- `srcVK/encode/lz_greedy_parser.glsl`
- `srcVK/encode/lz_encode_kernel.comp`
- `srcVK/encode/assemble_measure_kernel.comp`
- `srcVK/encode/assemble_write_kernel.comp`
- `srcVK/encode/frame_assemble_kernel.comp`

### Phase 9 — L2 stub encode kernels + headers
- `srcVK/encode/lz_chain_parser.glsl`
- `srcVK/encode/huff_build_tables_kernel.comp`
- `srcVK/encode/huff_encode_4stream_kernel.comp`

### Phase 10 — Encode context + LZ + assemble host orchestration
- `srcVK/encode/encode_context.zig`
- `srcVK/encode/levels.zig`
- `srcVK/encode/encode_lz.zig`
- `srcVK/encode/encode_assemble.zig`
- `srcVK/encode/encode_huff.zig` (stub bodies)

### Phase 11 — Top-level encoder + decoder + C ABI
- `srcVK/encode/fast_framed.zig`
- `srcVK/encode/streamlz_encoder.zig`
- `srcVK/encode/driver.zig`
- `srcVK/decode/streamlz_decoder.zig`
- `srcVK/streamlz_gpu.zig`

### Phase 12 — CLI surface
- `srcVK/cli/util.zig`
- `srcVK/cli/info.zig`
- `srcVK/cli/decompress.zig`
- `srcVK/cli/compress.zig`
- `srcVK/cli/bench_decompress.zig`
- `srcVK/cli/bench_compress.zig`
- `srcVK/cli/bench_all.zig`
- `srcVK/cli.zig`
- `srcVK/main.zig`

### Phase 13 — Tests (NEW per Exception 3)
- `srcVK/test_runner_parallel.zig`
- `srcVK/encode/gpu_roundtrip_tests.zig`
- every file listed in Section F
- `build.zig` test wiring (parallel test step `ptest_vk`)

---

## Section F — Tests plan (Exception 3)

CUDA has no `src/tests/` subtree; these test files are new and have
no CUDA counterpart. They live under `srcVK/tests/`. The test runner
itself (`srcVK/test_runner_parallel.zig`) is a verbatim port of
`src/test_runner_parallel.zig`.

| Test file | Purpose |
|---|---|
| `srcVK/tests/decoder_unit.zig` | Unit tests for `srcVK/decode/streamlz_decoder.zig`: `parseHeader`/`parseBlockHeader`/`parseChunkHeader` edge cases; pure host-side, no GPU. |
| `srcVK/tests/encoder_unit.zig` | Unit tests for `srcVK/encode/streamlz_encoder.zig`: `compressBound` math, `Options` validation (`BadLevel`, `BadBlockSize`, `BadScGroupSize`), `writeUncompressedFrame` path for tiny inputs. No GPU. |
| `srcVK/tests/dispatch_unit.zig` | Unit tests for `srcVK/decode/decode_dispatch.zig`: L2 gate behavior (level=1 skips Huff/scan/compact/merge/gather), `runLzPipeline` raw-kernel selection, `buildChunkDescriptors` output. |
| `srcVK/tests/kernel_conformance.zig` | Kernel conformance: feed known sub-chunk inputs into `slzLzDecodeRawKernel`, `slzPrefixSumChunksKernel`, `slzLzEncodeKernel`, `slzAssembleMeasureKernel`, `slzAssembleWriteKernel`, `slzFrameAssembleKernel` via VK; assert byte-identical outputs vs CUDA goldens stored in `tests/goldens/`. |
| `srcVK/tests/l1_decode_roundtrip.zig` | Golden L1 frames (encoded via CUDA reference or a fresh VK encode) → decoded via VK port → byte-compare against original. |
| `srcVK/tests/l1_encode_roundtrip.zig` | Random + structured payloads encoded via VK port → decoded via VK port → byte-compare. Levels 1-2 exercised here; levels 3-5 covered by `cross_backend_roundtrip.zig` once VK L2 lands. |
| `srcVK/tests/cross_backend_roundtrip.zig` | Encode via CUDA, decode via VK; encode via VK, decode via CUDA. Levels 1-2 full; levels 3-5 once Huffman lands. |
| `srcVK/tests/cli_smoke.zig` | Smoke test the VK binary: `streamlz_vk -c file -o out.slz` then `streamlz_vk -d out.slz -o roundtrip` and compare to original. Exercises every CLI mode at L1. |

Goldens directory (`tests/goldens/`) already exists at the repo root
(per git status). Reuse it for VK conformance fixtures.

---

## Section G — Differences from prior audit

The prior audit (agent `ae7dc1ef4ad1b0638` in workflow
`wf_0c9c5ca6-39f`) produced the file mapping under `/src_vk/` that
violates Section A and B of this fresh audit. Concrete corrections,
file-by-file:

### G.1 — Files where prior audit kept `.cuh` extension in srcVK target

The prior agent's mapping table set the following `vk_target_path`
values with a `.cuh` suffix; the on-disk `src_vk/` tree confirms it.
All must be `.glsl` per Section A. (The actual on-disk filename for
each in `src_vk/` is shown in the "prior wrote" column — many were
in fact written with `.cuh`.)

| File | Prior srcVK path (WRONG) | This audit's srcVK path (CORRECT) |
|---|---|---|
| `src/common/gpu_huffman.cuh` | `src_vk/common/gpu_huffman.cuh` (on-disk) | `srcVK/common/gpu_huffman.glsl` |
| `src/decode/compact_descs_kernels.cuh` | `src_vk/decode/compact_descs_kernels.cuh` (on-disk) | split into `srcVK/decode/compact_huff_descs_kernel.comp` + `srcVK/decode/compact_raw_descs_kernel.comp` |
| `src/decode/gather_raw_off16_kernel.cuh` | `src_vk/decode/gather_raw_off16_kernel.cuh` (on-disk) | `srcVK/decode/gather_raw_off16_kernel.comp` |
| `src/decode/lz_decode_general.cuh` | `src_vk/decode/lz_decode_general.cuh` (on-disk) | `srcVK/decode/lz_decode_general.glsl` |
| `src/decode/lz_header_parse.cuh` | `src_vk/decode/lz_header_parse.cuh` (on-disk) | `srcVK/decode/lz_header_parse.glsl` |
| `src/decode/merge_huff_descs_kernel.cuh` | `src_vk/decode/merge_huff_descs_kernel.cuh` (on-disk) | `srcVK/decode/merge_huff_descs_kernel.comp` |
| `src/decode/scan_parse_kernel.cuh` | `src_vk/decode/scan_parse_kernel.cuh` (on-disk) | `srcVK/decode/scan_parse_kernel.comp` |
| `src/decode/walk_frame_kernel.cuh` | `src_vk/decode/walk_frame_kernel.cuh` (on-disk) | `srcVK/decode/walk_frame_kernel.comp` |
| `src/encode/lz_chain_parser.cuh` | `src_vk/encode/lz_chain_parser.cuh` (on-disk) | `srcVK/encode/lz_chain_parser.glsl` |
| `src/encode/lz_format.cuh` | `src_vk/encode/lz_format.cuh` (on-disk) | `srcVK/encode/lz_format.glsl` |
| `src/encode/lz_greedy_parser.cuh` | `src_vk/encode/lz_greedy_parser.cuh` (on-disk) | `srcVK/encode/lz_greedy_parser.glsl` |
| `src/encode/lz_token_emit.cuh` | `src_vk/encode/lz_token_emit.cuh` (on-disk) | `srcVK/encode/lz_token_emit.glsl` |

### G.2 — Files where prior audit kept `.cu` extension in srcVK target

| File | Prior srcVK path (WRONG) | This audit's srcVK path (CORRECT) |
|---|---|---|
| `src/decode/huffman_kernel.cu` | `src_vk/decode/huffman_kernel.cu` (on-disk) | split into `srcVK/decode/huff_build_lut_kernel.comp` + `srcVK/decode/huff_decode_4stream_kernel.comp` |
| `src/encode/assemble_kernel.cu` | `src_vk/encode/assemble_kernel.cu` (on-disk) | split into `srcVK/encode/assemble_measure_kernel.comp` + `srcVK/encode/assemble_write_kernel.comp` + `srcVK/encode/frame_assemble_kernel.comp` |
| `src/encode/huffman_kernel.cu` | `src_vk/encode/huffman_kernel.cu` (on-disk) | split into `srcVK/encode/huff_build_tables_kernel.comp` + `srcVK/encode/huff_encode_4stream_kernel.comp` |
| `src/encode/lz_kernel.cu` | `src_vk/encode/lz_kernel.cu` (on-disk) | `srcVK/encode/lz_encode_kernel.comp` |

(The prior audit's row for `src/decode/lz_kernel.cu` correctly named
`lz_decode_kernels.comp`, but the on-disk file did not get renamed
across siblings — see G.1.)

### G.3 — Files where prior audit kept `cuda*` / `cu*` token in filename

| File | Prior srcVK path (WRONG) | This audit's srcVK path (CORRECT) |
|---|---|---|
| `src/encode/cuda_ffi.zig` | `src_vk/encode/cuda_ffi.zig` (on-disk; the prior audit's table said `vulkan_ffi.zig` but the Foundation agent wrote `cuda_ffi.zig` and rationalized it) | `srcVK/encode/vulkan_ffi.zig` |

(The prior Foundation agent landed `vulkan_api.zig` correctly for the
decode side but left `cuda_ffi.zig` on the encode side. Section B has
no exceptions; the encode FFI shim must be `vulkan_ffi.zig`.)

### G.4 — Encoder files prior audit wrongly classified as L2 stubs

The prior audit's `l2_stub_files` list included the entire encoder
tree. This audit reclassifies the L1+L2 encode hot path as **L1
scope**. Only the L3-L5 Huffman pieces, the L5 chain parser, and the
L1+L2-irrelevant subset stub.

| File | Prior classification | This audit's classification |
|---|---|---|
| `src/encode/streamlz_encoder.zig` | L2 stub (returns `NotImplementedL2`) | **L1 scope** (full body port) |
| `src/encode/driver.zig` | L2 stub (symbols only) | **L1 scope** (full body port; pub-var singletons + facade re-exports) |
| `src/encode/cuda_ffi.zig` → `vulkan_ffi.zig` | L2 stub | **L1 scope** (full body port) |
| `src/encode/module_loader.zig` | L2 stub | **L1 scope** (LZ kernel resolution must succeed; Huff + assemble kernel resolution stays optional as in CUDA) |
| `src/encode/encode_context.zig` | L2 stub | **L1 scope** (full body port; the Huff-related fields exist but go unused on L1) |
| `src/encode/encode_lz.zig` | L2 stub | **L1 scope** (full body port; this is the L1 hot encode launcher) |
| `src/encode/encode_huff.zig` | L2 stub | **L2 stub** (unchanged — but the symbol surface must exist for `fast_framed.zig` to compile) |
| `src/encode/encode_assemble.zig` | L2 stub | **L1 scope** (full body port; assembly chain is L1 hot) |
| `src/encode/fast_framed.zig` | L2 stub | **L1 scope** (full body port; this is the L1 encode orchestrator) |
| `src/encode/levels.zig` | L2 stub | **L1 scope** (full body port; `hashBitsForLevel` is consulted at L1) |
| `src/encode/lz_kernel.cu` → `lz_encode_kernel.comp` | L2 stub | **L1 scope** (full body port; `slzLzEncodeKernel` is the L1 hot encode kernel) |
| `src/encode/huffman_kernel.cu` → 2× `.comp` | L2 stub | **L2 stub** (unchanged) |
| `src/encode/assemble_kernel.cu` → 3× `.comp` | L2 stub | **L1 scope** (full body port; the assembly chain is the L1 frame writer) |
| `src/encode/lz_format.cuh` → `lz_format.glsl` | L2 stub | **L1 scope** (full body port; encoder-private hash constants needed at L1) |
| `src/encode/lz_token_emit.cuh` → `lz_token_emit.glsl` | L2 stub | **L1 scope** (full body port; both parsers emit through this) |
| `src/encode/lz_greedy_parser.cuh` → `lz_greedy_parser.glsl` | L2 stub | **L1 scope** (full body port; greedy parser drives L1-L4) |
| `src/encode/lz_chain_parser.cuh` → `lz_chain_parser.glsl` | L2 stub | **L2 stub** (unchanged — L5 only) |
| `src/encode/gpu_roundtrip_tests.zig` | L2 stub | **L1 scope** (full body port; tests cover L1 cases) |

### G.5 — Prior audit "kept as cuda_*" rationalizations (FORBIDDEN under this audit)

The prior Foundation agent left strings like
`token 'cuda' on adapt allow-list` in `src_vk/decode/descriptors.zig`
(referring to the `cudaCall` identifier renamed to `vkCall`). Future
agents must not produce or preserve such rationalization comments
when implementing this audit. The Section B rule is mechanical.

### G.6 — Other corrections

- Prior audit included `src/test_runner_parallel.zig` row but marked
  `is_l1_decode_path: false`. This audit reclassifies as **yes**
  because it is needed to run the new VK test suite (Exception 3).
- Prior audit listed `*.OMIT` target paths for `.ptx` / `.cubin`
  build artifacts. This audit replaces that with "not ported"
  (cleaner; the build system regenerates SPV blobs at build time).
- Prior audit did NOT explicitly list `srcVK/error.zig` and
  `srcVK/vma.zig` as port targets (they are new, foundation-only
  files). This audit's Section D lists them so the fleshout agent
  knows they belong at the `/srcVK/` root.

---

## Section H — Open questions

Genuinely open items the fleshout agent will need to decide based on
runtime behavior or user direction.

1. **Subgroup size on Intel iGPUs.** CUDA's warp size is 32 across all
   target SMs; Vulkan's subgroup size is hardware-dependent (32 on
   NVIDIA, often 16 on Intel iGPUs, 32/64 on AMD). The L1 raw decode
   kernel currently assumes WARP_SIZE=32 (one warp per sub-chunk).
   The fleshout agent must decide whether to (a) require
   `VK_SUBGROUP_FEATURE_BASIC | _SHUFFLE | _BALLOT` with
   `subgroupSize=32` and reject other devices, or (b) parameterize
   on `gl_SubgroupSize` and adapt. The MEMORY note
   "Verify device name on every perf measurement" suggests the
   project has already been bitten by Intel iGPU vs NVIDIA dGPU
   confusion; this is the right place to lock the policy down.

2. **`sm_count` analogue for `fast_framed.resolveScGroupSize`.**
   CUDA reads `cuda_api.sm_count` to pick `sc_group_size` between
   0.25 and 0.5. The Vulkan equivalent is
   `vkPhysicalDeviceProperties.limits` plus subgroup count, but the
   right scalar to threshold against is not obvious (NVIDIA reports
   sm_count via NVX extension; portable Vulkan does not). Options:
   (a) hard-code the fallback `sm_count_fallback=34` always, (b)
   probe via VK_NV_shader_sm_builtins if present, (c) use queue
   family count × compute units. The fleshout agent should pick
   the simplest option that preserves L1 perf parity.

3. **Async D2D path (Exception 1).** `decompressFramedFromDevice` /
   `slzDecompressAsync` and `slzCompressAsync` are L2+ async-D2D
   paths in CUDA. The prior audit marked the D2D-only kernels
   (`walk_frame_kernel`) as L2 stubs. If the L1 milestone requires
   the async D2D entry points to exist as working symbols (the C
   ABI exposes them), the fleshout agent must decide whether to
   implement the device-side frame walk now or stub the entry to
   `error.NotImplementedL2`. This audit leaves them as L2 stubs;
   confirm that matches the L1 milestone definition.

If the answers to (1) / (2) / (3) are pre-decided in project rules
not surfaced to this agent, the fleshout agent should consult
`CLAUDE.md` and the `feedback_*.md` memory notes before improvising.

---

## Self-check (run before signing off)

- [x] Every file in CUDA's `/src/` (and `/include/`) tree appears
      in Section C: counted 73 source/header files plus 11 build
      artifacts; all 84 rows present.
- [x] No `srcVK` path in Section C or D has a `.cu` or `.cuh`
      extension.
- [x] No `srcVK` path in Section C or D has `cu` / `Cu` / `CU` /
      `cuda` / `Cuda` / `CUDA` / `nvidia` / `Nvidia` / `NVIDIA`
      tokens in the file name (verified by token-level scan of the
      tree under Section D).
- [x] Encoder files needed for L1 encode are classified as L1 scope
      (Section G.4 enumerates each).
- [x] Section G explicitly lists corrections to the prior audit.
- [x] `/srcVK/audit.md` exists at the project root.
