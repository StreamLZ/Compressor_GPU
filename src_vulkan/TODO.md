# Vulkan port — remaining work

Last updated after commit `8cd8155`. This file is the source of truth for
"what's left." Top half: L1 finishing work. Bottom half: L2-L5 gameplan
to reach 100% functional parity with the CUDA backend.

## Current state at a glance

**15 commits** under the `vk-l1*` / `Vulkan port` series since `e0333ae`.

What works end-to-end **today**:

- `streamlz_vk.exe` is a self-contained binary (SPV embedded). Runs from
  any cwd.
- `--probe` lists Vulkan devices; `--device <N|substring>` selects;
  `SLZ_VK_DEVICE_INDEX=N` env var is the test-harness fallback.
- L1 codec round-trips byte-equal on **both** Intel iGPU (`device 0`) and
  NVIDIA RTX 4060 Ti (`device 1`).
- **L1 output is BYTE-IDENTICAL to CUDA's L1 default** (commit `8cd8155`
  switched `sc_group_size` to 0.25 to match CUDA). Verified on 1 MiB
  enwik8, 4.5 MB web.txt, and 16 MiB enwik8 via `cmp`.
- Cross-backend interop (CUDA encoder ↔ Vulkan decoder, both directions)
  works on `enwik8 full` (100 MB) and every sub-100 MB corpus tested at
  default `--sc 0.25`.
- Pure VK→VK round-trip works on `silesia_full` (200 MB / 3,248 chunks
  at sc=0.25) on both devices.
- 35 base tests + 14 scale tests + ABI + CLI + cross-backend wire-format
  tests all pass on `device 0` (Intel iGPU) at HEAD.

Headline perf on `assets/web.txt` (4.5 MB), `zig build vk-perf-bench`,
at the new sc=0.25 default (NVIDIA decode is post-`0fa1afb`):

| device | encode ns/B | decode ns/B |
|---|---:|---:|
| Intel(R) Graphics (iGPU) | **11.14** | **4.19** |
| NVIDIA GeForce RTX 4060 Ti | **6.60** | **4.82** |

(Both encoders 30-45% faster after the sc=0.5→0.25 switch; Intel
decode also 17% faster; NVIDIA decode 4× faster after the discrete-GPU
readback-staging fix landed — was 20.21 ns/B pre-fix. At 100 MB+
inputs the NVIDIA decoder is now ~3.5 ns/B, slightly faster than
Intel.)

---

# L1 — remaining work

## CORRECTNESS BLOCKERS — none.

### B1. `VK→CUDA silesia_full` deterministic residual — **FIXED** (commit `12cad9e`)

| field | value |
|---|---|
| symptom (before fix) | `vk-wire-format-scale-test VK_TO_CUDA_SCALE silesia_full` failed at byte **198,967,304** (chunk 3036 of 3248 at sc=0.25, 60,704 diffs in the carved 2-chunk repro) |
| root cause | **wire-format wrap** — not encoder. The wrap unconditionally marked every chunk as LZ-compressed. CUDA's `slzLzDecodeRawKernel` picks the raw-copy fast path whenever `sc_comp_size >= sc_size` (see `lz_decode_kernels.cuh:113`), so a chunk whose LZ payload exceeded `chunk_decomp_size` (high-entropy data like the silesia x-ray slice) was silently reinterpreted as raw bytes. |
| fix | When `sub_payload_size >= chunk_decomp_size`, emit the chunk as a true uncompressed block: internal_hdr0 bit 7 set, no 4-byte chunk hdr, just 2-byte hdr + `chunk_decomp_size` raw source bytes. Mirrors what CUDA's encoder does. The unwrap path already tolerated this layout, so VK→VK and CUDA→VK are unaffected. |
| verification | VK→CUDA silesia_full is byte-identical to source. Resulting VK SLZ output is byte-identical to CUDA's SLZ output (101,782,020 bytes). enwik8_full + web.txt unaffected. |

Closes the last L1 correctness bug. The L1 codec now matches CUDA's
L1 wire format byte-for-byte on every corpus tested.

## API COMPLETENESS — DONE (Phase 1 + Phase 2 of the L1 finishing work)

### A1. Six stubbed C ABI symbols — **DONE** (commit `7336eb3`)

All six entry points implemented and exercised by
`vk-abi-async-test` (6/6 pass on both Intel iGPU and NVIDIA RTX
4060 Ti):

| Symbol | Implementation |
|---|---|
| `slzMakeDeviceOnlyHandle_vk` | returns the `0x10` device-only sentinel (matches CUDA's `device_only_host_stub_addr` pattern). |
| `slzCompressAsync_vk` | spawns a per-handle 32 MiB-stack worker thread that runs the sync codec. |
| `slzCompressAsyncPoll_vk` | acquire-loads the worker's `done` flag; non-blocking poll returns `SLZ_ERROR_UNSUPPORTED` as the "not ready" sentinel; blocking poll joins + drains. |
| `slzDecompressAsync_vk` | symmetric with the encode async pair. |
| `slzDecompressAsyncPoll_vk` | symmetric with the encode poll. |
| `slzGetLastTimings_vk` | drains the per-handle `last_{encode,decode}_dispatch_ns` snapshot into a `KernelTimingC[]` array; names are static `"lz_encode"` / `"lz_decode"`. |

### A2. True D2D for `slzCompress_vk` / `slzDecompress_vk` — **DONE** (commit `710db2d`)

- `device.zig` now enables `bufferDeviceAddress` at device-create
  time on every device (was only chained when 8-bit storage was
  requested).
- `slzRegisterBuffer_vk` records a `(VkBuffer, VkDeviceAddress, size)`
  tuple on a per-handle 16-slot registry. `d_base_address == NULL`
  triggers an internal `vkGetBufferDeviceAddress` query.
- A new public helper `slzBufferGetDeviceAddress_vk` lets tests
  read addresses without importing `vk_api.zig`.
- `slzCompress_vk` / `slzDecompress_vk` peek `d_input` / `d_output`
  for registered addresses and route through D2D-aware codec paths
  that:
  * skip the HOST_VISIBLE staging buffer + memcpy when the source
    is device-resident (encode input D2D);
  * write straight into the caller's VkBuffer when the destination
    is device-resident (decode output D2D);
  * use a small staging-copy shim for the two correctness-preserving
    directions (encode output, decode input — the CPU wire format
    wrap/unwrap still needs host bytes, so this phase trades one
    transfer direction for another rather than removing it).
- New `vk-l1-d2d-test` exercises all three directions byte-equal to
  the host-pointer reference (3/3 pass on Intel + NVIDIA).

`slzMakeDeviceOnlyHandle_vk`'s sentinel is still rejected by the
codec — the registry path provides the actual D2D buffer; the
sentinel exists only for ABI symmetry with the CUDA `slzCompress`
device-only-buffer pattern. Real D2D callers use the registry.

## PERF (correctness done; throughput further-tunable)

### P1. NVIDIA decode ~5× slower than Intel — **FIXED** (commits `45613bd` + `0fa1afb`)

| field | value |
|---|---|
| symptom (before fix) | NVIDIA RTX 4060 Ti decode wall-time on `assets/web.txt` was 20.21 ns/B vs Intel iGPU's 4.26 ns/B (+374%). Got dramatically worse with size — silesia_full was 55.7 ns/B (~12 s for 213 MB). |
| diagnosis | VkQueryPool timestamp capture + per-phase QPC instrumentation in `vk_l1_perf_bench` showed GPU-side decode was actually 0.04-0.41 ns/B on NVIDIA (2x FASTER than Intel). All 5x wall-time gap was the final `@memcpy(dst_host, dst_b.mapped)` reading from the dst buffer's mapped memory — which the createBuffer helper had placed in NVIDIA's resizable BAR region (`DEVICE_LOCAL+HOST_VISIBLE`). CPU reads from BAR are uncached PCIe burst reads at 20-60 MB/s; on silesia the single 212 MB readback took 10.1 s. The kernel itself was 9.6 ms. |
| fix | Standard discrete-GPU pattern. Decoder now allocates `dst_b` `DEVICE_LOCAL`-only (no host mapping; full VRAM bandwidth for the kernel) and a sibling `dst_stage` buffer explicitly pinned to `HOST_VISIBLE+HOST_COHERENT` AND NOT `DEVICE_LOCAL` via the new `findHostVisibleNonDeviceLocal` memory-type helper. The decode cmdbuf gets a `vkCmdCopyBuffer(dst_b → dst_stage)` queued after the dispatch (in the same cmdbuf, via new `dispatch.submitOneWithCopy`, with a `SHADER_WRITE → TRANSFER_READ` buffer barrier). Host @memcpy now reads from sysmem-backed staging at cached-memory bandwidth. |
| verification | `vk_l1_perf_bench` after fix: NVIDIA decode is web.txt 4.82, enwik8 3.56, silesia 3.53 ns/B (Intel: 4.19 / 3.67 / 3.65). NVIDIA now matches or beats Intel at every size. VK→VK + VK→CUDA + CUDA→VK round-trips all byte-equal on web.txt + enwik8 + silesia on both devices. All 30+ `vk-l1-test` + 14 `vk-l1-scale-test` cases pass on both Intel and NVIDIA. |

Closes the last L1 perf item. Original candidates (1, 2, 3) breakdown:
candidate (1) was the right answer but in a non-obvious way — the BAR
readback was the entire story, NOT per-call PCIe overhead (which is
constant per call; if it had been (1), ns/B would have *shrunk* as
size grew, not *grown*). Candidate (2) (`8bit_storage`) was a false
lead — the GPU-side numbers prove `uint8_t` SSBO writes are actually
faster on NVIDIA than Intel. Candidate (3) (fast-batch driver path)
likewise not the bottleneck.

The decoder-phase instrumentation (`SLZ_VK_E2E_TIMER`-spirit via
`l1_codec.last_decode_*_ns` globals + `vk_l1_perf_bench` breakdown
output) was added in commit `45613bd` and stays — useful for future
perf work on the encoder host-overhead split.

### P2. GPU-side frame assembly — **DONE encode-side** (commit `f9af0a6`)

Encode-side wire-format wrap moved to GPU via the L1-specialized port
of `src/encode/assemble_kernel.cu`. Three new GLSL kernels +
`src_vulkan/wire_format_gpu.zig` host glue replace the CPU
`wire_format.wrapL1ToSlz1` path inside `slz1_codec.encodeL1ToSlz1`.
The CPU path stays as a fallback (toggle `slz1_codec.use_gpu_wrap`).

| Shader | Purpose |
|---|---|
| `assemble_measure.comp` | per-chunk sub-chunk payload sizing |
| `assemble_write.comp` | per-chunk sub-chunk payload write into `asm_out` |
| `frame_assemble.comp` | frame splice (per-chunk internal+chunk+sub-chunk headers, uncompressed-fallback raw copies, SC tail prefix, end mark) |

L1-specialization: the CUDA kernel handles both raw (type-0) and
Huffman (type-4) entropy chunks; the Vulkan L1 encoder only emits raw
streams today, so the assembler kernels statically pick the type-0
branch for every stream. Phase 2A (Huffman) generalizes the
descriptor + write kernel to choose huff-vs-raw per stream.

Verification (commit `f9af0a6` HEAD):
  * VK output byte-identical to CUDA output on web.txt (4.5 MB),
    enwik8 (100 MB), silesia_all.tar (200 MB) — `cmp` round-trip.
  * `vk-l1-test` PASS on Intel iGPU + NVIDIA RTX 4060 Ti.
  * `vk-l1-scale-test` PASS on both devices (14 cases, silesia_full
    3248 chunks).
  * `vk-l1-d2d-test` 3/3 PASS on both devices.
  * `vk-abi-test`, `vk-abi-async-test` 6/6 PASS.
  * `vk-wire-format-test` 6/6 PASS (CPU wrap path used by that
    suite — kept passing as the A/B reference).

Perf measurement (`streamlz_vk -b -r 3`, post-warmup compress wall
ms; bench reports a single compress wall-time per run, not split):

| corpus | device | CPU-wrap (baseline) | GPU-wrap (Phase 3) | delta |
|---|---|---:|---:|---:|
| web.txt 4.5 MB | Intel iGPU | 60-61 ms | 61 ms | ~0% |
| silesia 200 MB | Intel iGPU | 3035 ms | 3045 ms | +0.3% |
| web.txt 4.5 MB | NVIDIA 4060 Ti | 85 ms | 85 ms | ~0% |
| silesia 200 MB | NVIDIA 4060 Ti | 4159 ms | 4276 ms | +2.8% |

The expected e2e improvement did NOT materialize at this layer: the
extra per-dispatch buffer-alloc + cmdbuf submit + fence wait
overhead (3 extra dispatches in series) plus the source-bytes upload
into a GPU buffer cancels out the savings from skipping the CPU
wrap (~22 ms on silesia per the pre-Phase-3 measurement, a small
fraction of the encode wall). The wrap also still ends with a
device→host @memcpy of the final frame bytes, which dominates the
discrete-GPU case the same way it dominated the decoder pre-P1-fix.

The architectural prerequisite stays useful regardless:
  * Phase 2A's Huffman pass produces device-resident bodies; the
    assembler kernel is the natural splice point.
  * Phase 4 (decode-side GPU unwrap + multi-kernel decode pipeline)
    needs the same three-pass topology on the decode side.
  * A future bundle-level optimization (per-handle pool of
    reusable measure/write/frame buffers, pipeline cache for the
    three SPVs, eliminating the host-side prefix-sum via the
    already-stubbed `prefix_sum_chunks.comp`) is the right place
    to reclaim the 2-3% NVIDIA regression — out of Phase 3 scope.

Open follow-ups:
  * Reuse a per-handle scratch pool across `encodeL1ToSlz1` calls so
    the 6-buffer alloc-and-destroy hot path collapses to size checks.
  * Move the host-side prefix-sum to GPU via `prefix_sum_chunks.comp`
    once Phase 4 wires that kernel — measure pass would then publish
    its sizes directly into a device-resident offset table without
    the readback round-trip.
  * `wire_format_gpu` still allocates `asm_out` on the host-visible
    heap (uses the rebar/BAR path on NVIDIA discrete). Phase 5 should
    move it to `device_local_only` and add a stage-back copy for the
    final frame bytes, mirroring the decoder P1 fix.

## POLISH (none of these block anything)

### N0. Notable wins already landed (for the record)

- **Byte-identical to CUDA at L1 default** (commit `8cd8155`). Previously
  we had a 4% compression "win" at sc=0.5; diagnosed as a parameter
  mismatch, fixed by matching CUDA's sc=0.25 default. Two-line change
  with net 30-45% perf improvement as bonus.
- **Both Intel + NVIDIA proven working** (commit `4f27d6e`). Found 4
  sType-constant bugs in the process (Intel was silently tolerating
  wrong values; NVIDIA strict-validated).
- **Subgroup-size pin honored** (commit `d2e7abb`, then `4f27d6e`).
  Required a one-character constant fix (0x4 → 0x2) + the sType fixes
  above. Closed the entire "Intel SIMD16-split" mystery class of bugs.

### N1. Defensive `gl_SubgroupSize` stride is now redundant

`lz_encode.comp` (commit `de9ce00`) and `lz_decode.comp` (commit
`d625c8b`) both drive their warp-cooperative stride from
`gl_SubgroupSize` instead of compile-time `WARP_SIZE=32`. This was
necessary when Intel was silently splitting into 2×SIMD16. After
commit `d2e7abb` fixed the `REQUIRE_FULL_SUBGROUPS_BIT` flag (0x4→0x2)
and commit `4f27d6e` fixed the sType constants, the pin actually
honors on Intel and `gl_SubgroupSize` is always 32. The defensive
code is correct but dead-load-bearing. Could simplify back to
compile-time constants; keep as safety belt for now.

### N2. Three small defensive caveats — **DONE** (commit `08497f0`)

All three flagged invariants now have a comptime / runtime assertion
that fires loudly if a future change crosses the boundary:

| caveat | guard |
|---|---|
| `storeOff32Byte` assumes `CHUNK_OFF32_CAPACITY == CHUNK_STREAM_CAPACITY` | comptime block in `l1_codec.zig` pins them equal |
| `LARGE_OFFSET_THRESHOLD (0xC00000)` guard missing in encoder | comptime assertion `CHUNK_SIZE + LZ_BLOCK_SIZE < 0xC00000` in `l1_codec.zig`; comment in `lz_encode.comp::storeOff32LE24` rewritten to point at it |
| `decodeUnwrappedVkOnly` single-chunk-size assumption | runtime `std.debug.assert(unwrap.result.chunk_size <= l1_codec.CHUNK_SIZE)` in `wire_format_test.zig` |

---

# L2-L5 gameplan — full functional parity with CUDA

The CUDA encoder supports five compression levels:

| Level | Parser | Entropy | Streams |
|---|---|---|---|
| L1 | greedy | none (raw streams) | lit / cmd / off16 / length |
| L2 | greedy (hash_bits=18) | Huffman | lit / cmd / off16 / length, all Huffman-coded |
| L3 | greedy (hash_bits=19) | Huffman | same |
| L4 | greedy (hash_bits=17) | Huffman | same |
| L5 | **chain parser** (hash_bits=17 + chain table) | Huffman | same |

We have L1 (greedy, no entropy) functionally complete. To reach
**100% CUDA parity** we need:

- Huffman entropy coding (encoder + decoder) — covers L2, L3, L4
- Chain parser — covers L5
- GPU-side frame assembly — required for production throughput parity
- Full multi-kernel decode pipeline — currently CPU unwraps; CUDA does it on GPU

## Inventory: 16 .comp shells currently in `src_vulkan/shaders/`

These were created as 8-line stubs that write `0xDEADBEEF`, so the
build infrastructure works end-to-end before kernels were ported.
Only `lz_encode.comp` (950 LOC) and `lz_decode.comp` (698 LOC) are
real ports. The 14 remaining shells are placeholders.

CUDA reference LOC for context (`src/encode/*.cuh` + `src/decode/*.cuh`):

| Shell file | CUDA reference | CUDA LOC | Purpose |
|---|---|---:|---|
| `huff_build_tables.comp` | `src/encode/huffman_kernel.cu` (encode side) | ~350 | histogram → canonical Huffman codes (encoder) |
| `huff_encode_4stream.comp` | `src/encode/huffman_kernel.cu` (encode side) | ~350 | bytes + codes → BIL bitstream |
| `huff_build_lut.comp` | `src/decode/huffman_kernel.cu` | ~350 | weights → 256-entry decode LUT |
| `huff_decode_4stream.comp` | `src/decode/huffman_kernel.cu` | ~350 | bitstream + LUT → bytes |
| `assemble_measure.comp` | `src/encode/assemble_kernel.cu` | ~300 | frame-assembly sizing pass |
| `assemble_write.comp` | `src/encode/assemble_kernel.cu` | ~300 | frame-assembly write pass |
| `frame_assemble.comp` | `src/encode/assemble_kernel.cu` | ~200 | outer frame composition |
| `scan_parse.comp` | `src/decode/scan_parse_kernel.cuh` | 234 | parse compressed-block headers |
| `prefix_sum_chunks.comp` | `src/decode/prefix_sum_chunks_kernel.cuh` | 38 | running offset across chunks |
| `compact_huff_descs.comp` | `src/decode/compact_descs_kernels.cuh` | ~45 | gather entropy-decoded chunk descriptors |
| `compact_raw_descs.comp` | `src/decode/compact_descs_kernels.cuh` | ~45 | gather raw chunk descriptors |
| `walk_frame.comp` | `src/decode/walk_frame_kernel.cuh` | 200 | walk parsed frame |
| `merge_huff_descs.comp` | `src/decode/merge_huff_descs_kernel.cuh` | 62 | merge huff descriptors |
| `gather_raw_off16.comp` | `src/decode/gather_raw_off16_kernel.cuh` | 38 | gather raw off16 stream |
| `lz_decode_raw.comp` | `src/decode/lz_decode_raw.cuh` | 211 | alternate decode for raw off16 mode |
| `match_any_bench.comp` | (own diagnostic, not production) | 72 | already filled; benchmark only |

**Plus needs:**

- New shader `lz_chain_encode.comp` (no shell yet) ← L5 chain parser
  port. CUDA reference: `src/encode/lz_chain_parser.cuh` (440 LOC). The
  most complex encoder kernel — chain table + secondary hash + lazy
  matching.

## Phase plan (ordered by dependency)

### Phase 2A — Huffman codec (unlocks L2/L3/L4)

Goal: VK encoder produces chunk type 4 (Huffman-coded) sub-chunks
that the CUDA decoder reads, and the VK decoder reads CUDA-produced
chunk type 4 sub-chunks.

Kernels to port:
1. `huff_build_tables.comp` — encoder histogram → codes. ~300 LOC GLSL.
2. `huff_encode_4stream.comp` — encoder bytes → BIL stream. ~400 LOC.
3. `huff_build_lut.comp` — decoder weights → LUT. ~250 LOC.
4. `huff_decode_4stream.comp` — decoder bitstream → bytes. ~400 LOC.

Host glue:
- Extend `l1_codec.zig` → `l2_codec.zig` (or generalize), wire Huffman
  pass after the LZ parser, select chunk type 4 in wire-format wrap.
- Sub-chunk wire format already supports it; encoder just needs to emit.
- Decoder needs to dispatch chunk-type-4 sub-chunks through the
  Huffman path before the LZ-decode path.

Test extensions:
- New `vk-l2-test`, `vk-l3-test`, `vk-l4-test` round-trip suites.
- Extend cross-backend tests to cover L2-L4.

LOC estimate: ~1,500 GLSL + ~800 Zig. Effort: 3-5 single agents.

### Phase 2B — L5 chain parser (the last encoder level)

Goal: VK encoder L5 produces .slz files byte-correct vs CUDA L5
(within the same hash-store-order tolerance we accept for L1).

Kernels to port:
1. New `lz_chain_encode.comp` — chain parser. ~600 LOC GLSL. Most
   complex single kernel in the project.
2. Reuse Phase 2A's Huffman kernels.

Host glue:
- `l5_codec.zig`: 4× memory footprint per sub-chunk vs L1 — the
  existing `hash_bin_pack.zig` already validates this scales to
  256 MiB inputs with ≤2 GiB VkBuffer cap.
- L5 dispatches the new chain encoder instead of the greedy one.

Test extensions: `vk-l5-test` + cross-backend extension.

LOC estimate: ~800 GLSL + ~500 Zig. Effort: 2-3 single agents.

### Phase 3 — GPU-side frame assembly (encode wrap) — **DONE** (commit `f9af0a6`)

Encode-side wrap moved to GPU. Three new GLSL kernels
(`assemble_measure.comp`, `assemble_write.comp`,
`frame_assemble.comp`) + ~600 LOC of `wire_format_gpu.zig` host glue
replace the CPU wrap inside `slz1_codec.encodeL1ToSlz1`. CPU path
stays as the runtime-toggleable fallback. See the P2 section above
for the verification table + perf measurement.

Decode-side unwrap is still CPU — Phase 4 handles that.

### Phase 4 — Full GPU decode pipeline

**Status: e2e perf goal MET via P1-style buffer-placement fix, not a
full kernel port. Full kernel chain remains pending for a future agent
run; the kernel shells (`scan_parse.comp` etc.) stay as 8-line stubs.**
See "Phase 4 perf headline" subsection below for the actual fix.

Original goal: CUDA's per-frame `scan_parse → prefix_sum_chunks →
compact_descs → walk_frame → merge_huff_descs → dispatch chunks` chain
ported to GPU. Currently the Vulkan decoder unwraps on CPU and
dispatches one chunk at a time. CUDA processes the entire frame in a
single multi-kernel dispatch graph.

Kernels still to port (no work landed this run):
1. `scan_parse.comp` — parse compressed-block headers on GPU.
2. `prefix_sum_chunks.comp` — running offsets.
3. `compact_huff_descs.comp` + `compact_raw_descs.comp` — gather descs.
4. `walk_frame.comp` — walk the chunk list.
5. `merge_huff_descs.comp` + `gather_raw_off16.comp` — final dispatch prep.
6. `lz_decode_raw.comp` — **investigation result (this run, no port
   needed):** `src_vulkan/shaders/lz_decode.comp` already covers L1's
   raw-mode path. The CUDA `decodeSubChunkRawMode` template is
   instantiated with `OFF16_SPLIT=false` for the raw branch; the
   existing Vulkan port is hard-coded to the same `OFF16_SPLIT=false`
   layout (see the head comment: "OFF16_SPLIT = false (interleaved u16
   LE off16 entries only)"). The two-instance design in CUDA only
   matters when the off16 stream is Huffman-coded (`OFF16_SPLIT=true`),
   which is a Phase 2A concern. **The `lz_decode_raw.comp` shell can
   be deleted when Phase 2A lands**, or kept as a placeholder for the
   `OFF16_SPLIT=true` instance if we keep symmetry with CUDA's two
   templated paths.

Required for: future Huffman / chain-parser modes; streaming async
decode patterns where the host never sees compressed bytes.

LOC estimate (unchanged): ~1,500 GLSL + ~600 Zig. Effort: 3-4 single
agents. **DEPRIORITIZED** — the perf framing that motivated Phase 4
(NVIDIA e2e ≫ d2d) was solved by the buffer-placement fix below
without the kernel port. The remaining gap is sysmem memcpy and
page-fault cost, which a GPU header-parse does not address.

### Phase 4 perf headline — **DONE** (this commit)

The original Phase 4 framing ("e2e on NVIDIA web.txt should drop from
~121 ms toward d2d ~1.84 ms") attributed the 119 ms gap to the CPU
unwrap. Profiling `decodeSlz1ToBytes` with a temporary per-phase QPC
breakdown (added as `last_decode_slz_*_ns` globals + the
`SLZ_VK_PROFILE_DECODE` env knob in cli_vk.zig) showed the actual
breakdown on NVIDIA RTX 4060 Ti / web.txt 4.5 MB:

  Pre-fix, NVIDIA:  e2e 122 ms = unwrap 9 + alloc 6 + memset 10 +
                                  fill 0.2 + descset 0.2 + dispatch 2 +
                                  **readback 88** ms.

The 88 ms readback was the dst buffer being allocated in NVIDIA's
resizable BAR region (`DEVICE_LOCAL+HOST_VISIBLE`) and the trailing
`@memcpy(out_host, dst_b.mapped)` reading bytes back through uncached
PCIe at ~50 MB/s. **This is the exact same pathology as P1**, just in
a different code file — `slz1_codec.decodeSlz1ToBytes`'s local
`createMappedBuffer` helper was bypassing the
`createBufferEx(.device_local_only) + sysmem dst_stage +
submitOneWithCopy` pattern that `l1_codec.decodeL1Sync` adopted in
commits `45613bd` + `0fa1afb`.

Fix (this commit):
  * Exported `l1_codec.Buffer`, `l1_codec.createBufferEx`,
    `l1_codec.destroyBuffer`.
  * `slz1_codec.decodeSlz1ToBytesEx` now allocates `dst_b` as
    `device_local_only` and a sibling `dst_stage` as
    `host_visible_sysmem`, dispatches via `dispatch.submitOneWithCopy`
    with the dst→stage copy in the same cmdbuf, and reads back from
    `dst_stage.mapped` instead of `dst_b.mapped`.
  * Dropped the 5×stream_cap @memset of the stream SSBOs (was a
    no-op for correctness — `lz_decode.comp`'s per-byte loads are
    bounded by per-chunk descriptor sizes, never read garbage that
    affects output). Saved ~10 ms on NVIDIA per call.
  * Added `last_decode_slz_*_ns` per-phase QPC globals + the
    `SLZ_VK_PROFILE_DECODE` cli_vk.zig env knob (always-on
    instrumentation, only the print is gated).

Verification:
  * `vk-l1-test` PASS on Intel + NVIDIA.
  * `vk-l1-scale-test` PASS on Intel + NVIDIA (14/14, silesia_full
    3248 chunks).
  * `vk-l1-d2d-test`, `vk-abi-test`, `vk-abi-async-test` PASS on
    Intel + NVIDIA.
  * `cmp` byte-equal vs source on web.txt + enwik8 + silesia, VK
    Intel + VK NVIDIA decoders.
  * `cmp` byte-equal CUDA↔VK encoder output on web.txt + enwik8 +
    silesia (same VK SLZ output as CUDA — no encoder regression).

Perf headline (`streamlz_vk -b -r 3`, ReleaseFast, median across 3
post-warmup runs):

| corpus | device | e2e BEFORE | e2e AFTER | d2d | speedup |
|---|---|---:|---:|---:|---:|
| web.txt 4.5 MB | Intel iGPU | 37 ms | **20 ms** | 3.57 ms | 1.85× |
| web.txt 4.5 MB | NVIDIA 4060 Ti | 122 ms | **38 ms** | 1.97 ms | 3.2× |
| enwik8 95 MB | NVIDIA 4060 Ti | (not measured) | **688 ms** | 6.4 ms | — |
| silesia 203 MB | NVIDIA 4060 Ti | (not measured) | **1433 ms** | 14.4 ms | — |
| silesia 203 MB | Intel iGPU | (not measured) | **1100 ms** | 57.6 ms | — |

The remaining e2e overhead is mostly:
  * 14 ms sysmem `@memcpy` of the staged dst on web.txt (4.5 MB at
    ~320 MB/s cached); scales linearly with size.
  * 8–15 ms dispatch wall (kernel + dst→stage GPU copy + fence wait).
  * ~10 ms first-touch page-fault on the stream SSBOs after dropping
    the @memset (cost shifted from memset to the fill phase).
  * 0.5 ms CPU unwrap.

A GPU port of `scan_parse` + `walk_frame` would shave the 0.5 ms
unwrap but the other costs would persist. Phase 5 follow-ups for
`device_local_only` stream buffers + per-handle scratch pool reuse
+ a true async submit path are higher-leverage now that the BAR
readback is gone.

### Phase 5 — Async API + true D2D + final polish

Once the kernel paths are complete, the remaining C-ABI stubs become
implementable.

Items:
- `slzCompressAsync_vk` + `slzCompressAsyncPoll_vk` — back with the
  same worker-thread + fence pattern CUDA uses; or use Vulkan
  timeline semaphores natively.
- `slzDecompressAsync_vk` + `slzDecompressAsyncPoll_vk`.
- `slzGetLastTimings_vk` — wire up the `VkQueryPool` (already exists
  in `timing.zig` but not consumed).
- `slzMakeDeviceOnlyHandle_vk` + true BDA D2D for
  `slzCompress_vk` / `slzDecompress_vk`.

LOC estimate: ~600 Zig. Effort: 1-2 single agents.

### Phase 6 — Cross-backend conformance + perf parity validation

- Extend `tests/cross_backend_tests.zig` and the scale tests to cover
  all L1-L5 in both directions on both Intel and NVIDIA.
- Benchmark vs CUDA on the same hardware (RTX 4060 Ti); target ≤20%
  perf gap on encode and decode at L1, L3, L5.
- Document any residual gaps as accepted tradeoffs.

LOC estimate: ~400 Zig (tests + bench scripts). Effort: 1 single agent.

## Total estimate to full CUDA parity

| phase | GLSL LOC | Zig LOC | rough agent-runs |
|---|---:|---:|---:|
| L1 finishing (B1 + A1/A2 + polish) | ~100 | ~600 | 2-3 |
| Phase 2A (Huffman) | ~1,500 | ~800 | 3-5 |
| Phase 2B (chain parser L5) | ~800 | ~500 | 2-3 |
| Phase 3 (GPU frame assembly) | ~900 | ~400 | 2-3 |
| Phase 4 (GPU decode pipeline) | ~1,500 | ~600 | 3-4 |
| Phase 5 (async + BDA D2D + polish) | 0 | ~600 | 1-2 |
| Phase 6 (conformance + perf parity) | 0 | ~400 | 1 |
| **TOTAL** | **~4,800 GLSL** | **~3,900 Zig** | **14-21 single agents** |

For comparison: CUDA-side LOC under `src/` is ~13,000 (kernels +
host). Vulkan port at L1 today is `src_vulkan/` ≈ 11,700 LOC. Full
parity should land at roughly the same total LOC as CUDA, ±20%.
