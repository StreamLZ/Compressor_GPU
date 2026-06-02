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

## API COMPLETENESS (sync API works; async stubbed)

### A1. Six stubbed C ABI symbols in `streamlz_gpu_vk.zig`

All return `SLZ_ERROR_UNSUPPORTED` (= 5):

| Symbol | Purpose | Blocking? |
|---|---|---|
| `slzMakeDeviceOnlyHandle_vk` | sentinel for "input is already on device" | only D2D callers |
| `slzCompressAsync_vk` | non-blocking encode submission | async-pattern callers |
| `slzCompressAsyncPoll_vk` | poll async encode for completion | async-pattern callers |
| `slzDecompressAsync_vk` | non-blocking decode submission | async-pattern callers |
| `slzDecompressAsyncPoll_vk` | poll async decode | async-pattern callers |
| `slzGetLastTimings_vk` | retrieve per-kernel timings | profiling callers |

CLI + sync (`slzCompressHost_vk` / `slzDecompressHost_vk`) cover the
common case. These six matter for embedded-in-app integrations that
want non-blocking submission or per-kernel timings.

### A2. True D2D for `slzCompress_vk` / `slzDecompress_vk`

Currently the device-pointer-typed entry points (`slzCompress_vk` and
`slzDecompress_vk`) treat their `void*` args as **host pointers**
(documented in the function-header comments — no implicit copy). A
caller who has GPU-resident input/output buffers cannot use them
without first copying back to host.

Real D2D requires `VK_KHR_buffer_device_address` integration:
- Use `vkGetBufferDeviceAddress` to convert a `VkBuffer` to a
  device-address u64.
- Treat that u64 as the value passed to `slzCompress_vk(d_input)`.
- In the shader, dereference via the existing BDA path
  (`GL_EXT_buffer_reference` is already enabled).

`slzRegisterBuffer_vk` is the spec'd entry point for callers to hand
over a `VkBuffer` + its address; today it's a no-op. Wiring it would
unblock D2D for the codec.

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

### P2. GPU-side frame assembly (not yet ported)

Wire-format wrap/unwrap runs on CPU today. CUDA does it on GPU via
`src/encode/assemble_kernel.cu`. Current CPU wrap timings on
`silesia_full`:

| stage | wall-clock |
|---|---:|
| vk_encode_ns | 1,353,485,800 (~1.4 s on Intel) |
| bundle_ns (host glue) | (subsumed above) |
| wrap_ns | 21,977,000 (~22 ms — not dominant yet) |
| cuda_dec_ns | 252,736,000 |

At current corpora sizes the CPU wrap is ≤2% of total. For multi-GB
inputs it could matter. Required for true CUDA-throughput parity at
very large inputs.

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

### N2. Three small defensive caveats flagged in earlier commits

| caveat | location | severity |
|---|---|---|
| `storeOff32Byte` assumes `CHUNK_OFF32_CAPACITY == CHUNK_STREAM_CAPACITY` | `lz_encode.comp` | latent — true today but fragile |
| `LARGE_OFFSET_THRESHOLD (0xC00000)` guard missing in encoder | `lz_encode.comp` off32 emission | defensive — true for 128 KiB chunks |
| `decodeUnwrappedVkOnly` single-chunk-size assumption | `wire_format_test.zig` | test-only, not production code |

None block any current test. Bundle into a small cleanup commit when
convenient.

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

### Phase 3 — GPU-side frame assembly (CPU wrap → GPU wrap)

Goal: drop CPU wire-format wrap/unwrap; move to GPU compute. Required
for matching CUDA throughput at very large inputs (>500 MB).

Kernels to port:
1. `assemble_measure.comp` — pass-1 sizing.
2. `assemble_write.comp` — pass-2 write.
3. `frame_assemble.comp` — outer frame composition.

Host glue:
- Replace `wire_format.zig`'s `wrapL1ToSlz1` CPU path with a GPU
  dispatch sequence. Keep the CPU path as a fallback or for ABI
  compatibility.

LOC estimate: ~900 GLSL + ~400 Zig. Effort: 2-3 single agents.

### Phase 4 — Full GPU decode pipeline

Goal: CUDA's per-frame `scan_parse → prefix_sum_chunks → compact_descs
→ walk_frame → merge_huff_descs → dispatch chunks` chain ported to
GPU. Currently the Vulkan decoder unwraps on CPU and dispatches one
chunk at a time. CUDA processes the entire frame in a single
multi-kernel dispatch graph.

Kernels to port:
1. `scan_parse.comp` — parse compressed-block headers on GPU.
2. `prefix_sum_chunks.comp` — running offsets.
3. `compact_huff_descs.comp` + `compact_raw_descs.comp` — gather descs.
4. `walk_frame.comp` — walk the chunk list.
5. `merge_huff_descs.comp` + `gather_raw_off16.comp` — final dispatch prep.
6. `lz_decode_raw.comp` — alternate path for raw off16 mode (might
   already be covered by `lz_decode.comp`; needs investigation).

Required for: streaming async decode, low-latency callers, true
throughput parity at scale.

LOC estimate: ~1,500 GLSL + ~600 Zig. Effort: 3-4 single agents.

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
