# GPU cleanup follow-up — post-compact handoff

**Read this whole top section before starting any work.**

## Context (everything you need post-compact)

This file was written 2026-05-25 after an 8-phase cleanup arc on `src/gpu/`,
followed by a 5-agent re-review that surfaced these remaining items. The
cleanup arc is at commits `7480762..893e043` (P1 dead-tANS through P8
nitpick sweep, all pushed to `origin/main`). The 5-agent re-review ran
against `893e043`.

**You are picking this up after a context compaction.** Treat this file as
the source of truth for what's left. Re-reading prior conversation is not
required; everything you need to act is here.

### What the 8-phase cleanup did (summary)

1. Deleted ~1480 LOC of dead GPU-tANS scaffolding (kernels were retired
   2026-05-21 but Zig orchestration remained).
2. Small isolated fixes (HUFF_LUT_ENTRIES disambiguation, tautological
   asserts, parseRaw bounds break).
3. Hoisted wire-format constants to `common/gpu_wire_format.cuh` as the
   single source of truth (encode + decode include it).
4. Added byte-IO helpers (`readU16LE`, `readU32LE`, etc.) to
   `common/gpu_byteio.cuh`; deduped many open-coded byte-shift sites.
5. Extracted `warpLiteralCopy`/`warpMatchCopy` for the raw decoder; split
   `parseSubChunkHeaders` into per-stream helpers. 5b applied `readU32LE`
   to `lz_chain_parser.cuh` (REG win 80→72 on `slzLzEncodeKernel`).
6. Split `decode/driver.zig` (2143 → 7 sub-files + 89-LOC facade),
   `encode/driver.zig` (1213 → 7 sub-files + 111-LOC facade), and
   `decode/lz_kernels.cuh` (1075 → 8 .cuh headers).
7. Added `cudaCall(rc)` helper to `descriptors.zig`; wrapped 27
   H2D/D2H/sync sites in `decode_dispatch.zig`. Renamed stale `tans_` →
   `entropy_` throughout (~47 sites in 10 files).
8. README rewrite + new `ARCHITECTURE.md` long-form design doc; em-dash
   purge; `_t_xxx` → `t_xxx` Zig rename.

### Current file layout

```
src/gpu/
  common/
    gpu_warp.cuh           warp / lane geometry, bit-width constants
    gpu_byteio.cuh         little/big-endian load/store primitives
    gpu_huffman.cuh        canonical-Huffman wire format
    gpu_wire_format.cuh    encode/decode-shared LZ wire constants  (NEW in P3)
  decode/
    driver.zig             89-LOC facade (re-exports + g_default + telemetry vars)
    cuda_api.zig           Win32 FFI, CU* typedefs, cu*_fn slots, qpcNow/qpcMs
    module_loader.zig      PTX load + kernel handles + init / isAvailable
    descriptors.zig        POD types + GpuError + cudaCall + constants
    decode_context.zig     DecodeContext + ensureDeviceBuf/Output + helpers
    scan_host.zig          CPU fallback for scanForEntropyChunks (likely deletable)
    scan_gpu.zig           wraps walk/prefix-sum/compact/merge kernel launches
    decode_dispatch.zig    fullGpuLaunch + fullGpuLaunchImpl + Phase 5 helpers
    lz_kernel.cu           LZ decode aggregator
      lz_decode_core.cuh   hot loops + warp copy helpers  (471 LOC — needs split, see E1)
      lz_decode_kernels.cuh  slzLzDecodeKernel + slzLzDecodeRawKernel
      lz_dispatch.cuh      parseAndDecodeSubChunk + raw variant
      lz_header_parse.cuh  per-stream header parsers + outer __noinline__
      slz_wire_format.cuh  decoder-private token formats + parsers
      walk_frame_kernel.cuh / prefix_sum_chunks_kernel.cuh /
      scan_parse_kernel.cuh / compact_descs_kernels.cuh /
      merge_huff_descs_kernel.cuh / gather_raw_off16_kernel.cuh
    huffman_kernel.cu      slzHuffBuildLutKernel + slzHuffDecode4StreamKernel
    vulkan_driver.zig + lz_kernel.comp/.spv  (Vulkan path — DO NOT TOUCH)
  encode/
    driver.zig             111-LOC facade
    cuda_ffi.zig           CUDA Driver API FFI
    module_loader.zig      PTX load + handles + init  (piggybacks on decode — see A6)
    encode_context.zig     EncodeContext + ABI structs + ensureBuf
    levels.zig             hashBitsForLevel / useGlobalHash / useChainParser
    encode_lz.zig          gpuCompressImpl (LZ launcher)
    encode_huff.zig        gpuEncodeHuffImpl + per-stream {Literals,Tokens,Off16}
    encode_assemble.zig    gpuAssembleFrameImpl + gpuFrameAssembleImpl
    lz_kernel.cu           LZ encode aggregator
      lz_format.cuh + lz_token_emit.cuh + lz_greedy_parser.cuh + lz_chain_parser.cuh
    huffman_kernel.cu      slzHuffBuildTablesKernel + slzHuffEncode4StreamKernel
    assemble_kernel.cu     slzAssembleMeasureKernel + slzAssembleWriteKernel + slzFrameAssembleKernel
  README.md
  ARCHITECTURE.md          long-form design prose (em-dash-free)
  CLEANUP_TODO.md          this file
```

### Build / verify commands

```powershell
# Rebuild decode PTX (after touching any decode/*.cu / *.cuh)
cmd.exe /c "tools\build_gpu.bat"

# Rebuild encode PTX (after touching any encode/*.cu / *.cuh)
cmd.exe /c "tools\build_gpu_enc.bat"

# Zig build (always after any .zig change OR after PTX regen)
zig build -Doptimize=ReleaseFast -Dgpu=true

# Roundtrip SHA verify (sequential — do NOT run in parallel)
$enwik8_sha = (Get-FileHash -Algorithm SHA256 assets/enwik8.txt).Hash
foreach ($lvl in 1,3,5) {
  zig-out/bin/streamlz.exe -l $lvl -gpu assets/enwik8.txt -o c:/tmp/t_l$lvl.slz | Out-Null
  zig-out/bin/streamlz.exe -d -t 1 c:/tmp/t_l$lvl.slz -o c:/tmp/t_l$lvl.bin | Out-Null
  $h = (Get-FileHash -Algorithm SHA256 c:/tmp/t_l$lvl.bin).Hash
  if ($h -eq $enwik8_sha) { "enwik8 L$($lvl): PASS" } else { "enwik8 L$($lvl): FAIL" }
}
# enwik8 baseline: 2B49720EC4D78C3C9FABAEE6E4179A5E997302B3A70029F30F2D582218C024A8

# Same for silesia (use assets/silesia_all.tar; baseline:
# 86817A72FC4F404A0330247F69C1278DE8AD058AC1401A9CB1AE174FF5356C74)

# Bench (sequential ONE AT A TIME, NEVER in parallel tool calls)
zig-out/bin/streamlz.exe -db -t 1 -r 30 c:/tmp/t_l1.slz
```

### Project rules (CRITICAL — these are user-enforced)

1. **NEVER `git checkout`, `git restore`, `git revert`** without explicit
   user approval. If something broke, fix forward.
2. **NEVER run benchmarks in parallel.** Sequential tool calls only. A
   single message with 5 `-db` calls violates this rule.
3. **NEVER filter build/test/benchmark output** through `grep | tail | head |
   Select-Object -Last`. Show the full output.
4. **No em-dashes (—)** in `README.md` or `ARCHITECTURE.md`. Period.
5. **Always report decode + encode speed alongside ratio.** Never quote a
   ratio without speed numbers.
6. **GPU perf in milliseconds** with cuEvent timer (`-db` mode), never
   wall-clock for kernel timing.
7. **No commits unless asked.** No pushes unless asked.
8. **REG/STACK verification:** When touching kernel code, always rebuild
   PTX and cite REG/STACK numbers for every kernel before/after, even if
   unchanged.

### Verification protocol per batch

After every batch of fixes (don't mass-commit the whole follow-up):
1. Rebuild relevant PTX (`tools\build_gpu.bat` and/or `tools\build_gpu_enc.bat`).
2. Verify REG/STACK on every kernel (cite numbers in the commit message).
3. `zig build -Doptimize=ReleaseFast -Dgpu=true` clean.
4. Roundtrip SHA-verify on enwik8 L1/L3/L5 + silesia L3/L5.
5. After larger batches: `-db -t 1 -r 30` sequential bench, compare to
   README baseline within noise.
6. Commit the batch with a descriptive message. Push only when user asks.

### Deferred items NOT in this list

These are tracked in
`C:\Users\james.JAMESWORK2025\.claude\projects\c--Users-james-JAMESWORK2025-Repos-StreamLZ-native\memory\project_gpu_deferred_cleanup.md`
and are intentionally out of scope here:
- `scanBlock` structural split (326 LOC greedy parser, highest perf risk)
- `findMatchChain` structural split (~190 LOC chain parser, dense state)
- `int lane` vs `uint32_t lane` standardization (~15 sites, perf-neutral)
- `DevBuf` struct + `kernelParams` comptime helper (ergonomic-only)
- `gpuEncodeXxxHuffImpl` three-near-clones dedup

---

# The work

Sections:
- A. Real bugs (correctness, telemetry, doc-accuracy)
- B. Finish the byte-IO consolidation (Phase 4 follow-up)
- C. Finish the wire-format consolidation (Phase 3 follow-up)
- D. Finish the CUDA error wrapping (Phase 7 follow-up)
- E. Finish the file / function splits (Phase 5/6 follow-up)
- F. Dead code to delete
- G. Stale references to update
- H. Polish / minor correctness
- I. Docs improvements

Each item carries:
- `[risk]` low / med / high
- `[effort]` xs (one-line) / s (under 30 min) / m (under 2 hr) / l (multi-hour)
- `[verify]` the smallest check that catches a regression

---

## A. Real bugs

- [ ] **A1. Rename `"slzLzCompressKernel"` profile string and doc to `"slzLzEncodeKernel"`** [risk: low] [effort: xs] [verify: build]
  - `src/gpu/encode/encode_lz.zig:113` profile name; `:5` doc comment.
  - dlsym at `module_loader.zig:54-58` is correct (`slzLzEncodeKernel`); telemetry shows a kernel name that doesn't exist.

- [ ] **A2. Fix silent 16384-chunk cap in `encode_assemble.zig:199-200`** [risk: low] [effort: s] [verify: roundtrip on small + large inputs]
  - `var per_chunk_dst_buf: [16384]u32 = undefined; if (n_chunks > per_chunk_dst_buf.len) return null;`
  - Heap-allocate via the allocator (already on the slow assembly path) or surface an explicit error code so callers can distinguish "too many chunks" from "no kernel".

- [ ] **A3. Update README perf-section commit hash** [risk: none] [effort: xs]
  - `src/gpu/README.md` says HEAD `9768f19`; the cleanup pushed commits `9768f19`, `893e043`, and whatever this batch lands. Either re-measure on current HEAD and update both the hash + the date stamp, or move perf numbers to a date-stamped `PERFORMANCE.md`.

- [ ] **A4. Fix one em-dash slipped into `ARCHITECTURE.md`** [risk: none] [effort: xs] [verify: `grep -c — src/gpu/ARCHITECTURE.md` returns 0]
  - Re-grep to find current line — may have shifted from the reviewer's L449 cite.

- [ ] **A5. Replace Unicode `≥` (U+2265) at `gpu_wire_format.cuh:61` with ASCII `>=`** [risk: none] [effort: xs] [verify: PTX rebuilds clean on both decode and encode]

- [ ] **A6. Document or fix encode→decode init coupling** [risk: med — could be a real "encode unusable standalone" bug] [effort: m] [verify: smoke test of encode init when decode isn't imported]
  - `src/gpu/encode/module_loader.zig:35-38` calls `@import("../decode/driver.zig").init()` to piggyback on decode's CUDA context. Encode's own `cuInit`/`cuDeviceGet`/`cuCtxCreate` slots are loaded but never resolved (see F2).
  - Either lift context creation into encode's own init, or document that encode is a sub-package of decode (with a comment block + maybe renaming `cuda_ffi.zig` → `cuda_ffi_extra.zig` so the asymmetry is visible).

---

## B. Finish the byte-IO consolidation (Phase 4 follow-up)

The helpers exist in `common/gpu_byteio.cuh`. Phase 5b applied them to
`lz_chain_parser.cuh` and got REG 80→72 on `slzLzEncodeKernel`. Apply
everywhere else.

- [ ] **B1. Apply `readU32LE` to `lz_greedy_parser.cuh:107-110, 135-138, 152-155`** [risk: low] [effort: xs] [verify: PTX rebuild — likely REG drop on `slzLzEncodeKernel`]
  - Same idiom that gave the chain parser its REG drop. Three sites.

- [ ] **B2. Add `readU32BE` and `writeBE24` to `common/gpu_byteio.cuh`** [risk: low] [effort: xs]
  - Match the existing `readBE24` / `readU32LE` / `storeU16LE` style.

- [ ] **B3. Apply `readU32BE` to 4 inline 4-byte BE byte-shift sites** [risk: low] [effort: s]
  - `slz_wire_format.cuh:210-211` (`parseEntropyHeader` LONG branch)
  - `slz_wire_format.cuh:232-233` (`skipPairedPrimary`)
  - `scan_parse_kernel.cuh:46-49` (`scanParseHuffHeader`)
  - `scan_parse_kernel.cuh:98-101` (`scanSkipStreamHeader`)

- [ ] **B4. Apply `writeBE24` and delete `assemble_kernel.cu::writeRawChunkHdr`** [risk: low] [effort: xs]
  - `encode/lz_kernel.cu:206-208, 211-213` (2 inline 3-byte BE writes)
  - `assemble_kernel.cu:144-148` local `writeRawChunkHdr` is bit-identical to the new helper.

- [ ] **B5. Apply `readU16LE`/`readLE24` to `assemble_kernel.cu::parseRaw`** [risk: low] [effort: xs]
  - `:93` (off16_count, 2-byte LE), `:101-102` (packed off32 header, 3-byte LE), `:109` and `:114` (off32 count extensions, 2-byte LE).

- [ ] **B6. Apply `storeU16LE` to 3 missed sites in `assemble_kernel.cu`** [risk: low] [effort: xs]
  - `:246-247`, `:281-283`, `:291-293`. Currently open-code `byte | (byte << 8)`.

- [ ] **B7. Replace serial-fallback `memcpy(&v, ..., 2)` in `lz_decode_core.cuh:191-194, 208-211` with `readU16LE`** [risk: low] [effort: xs]
  - The parallel path at L120 already uses `readU16LE`; the serial fallback is the only inconsistency.

- [ ] **B8. Replace `walk_frame_kernel.cuh::walkReadU32LE` with `readU32LE`** [risk: low] [effort: xs]
  - Local open-coded byte-shift duplicates the common helper. Update `walkReadF32LE` to also use it.

- [ ] **B9. Apply `storeU16LE` + new `writeLE24` to `encode/lz_kernel.cu:225-226, 237-239`** [risk: low] [effort: xs]
  - Open-coded LE u16 / u24 stores; `storeU16LE` is already used at L221, so just add `writeLE24` to gpu_byteio.cuh (Phase 3 already added the reader) and apply.

---

## C. Finish the wire-format consolidation (Phase 3 follow-up)

`common/gpu_wire_format.cuh` exists as the single source of truth; the
scan kernel ignored it entirely.

- [ ] **C1. Migrate `scan_parse_kernel.cuh` bare magic numbers to shared constants** [risk: low — verify byte-exact roundtrip] [effort: s]
  - Sites: `:38, 41, 42, 50, 51, 66, 68, 85, 87, 93, 95, 102, 171`.
  - Mapping: `0x3FF` → `ENTROPY_SHORT_COMP_MASK`; `0x3FFFF` → `ENTROPY_LONG_SIZE_MASK`; `>>10` → `>>ENTROPY_SHORT_DELTA_SHIFT`; `>>14` → `>>ENTROPY_LONG_HI_SHIFT`; `>>18` → `>>ENTROPY_LONG_DELTA_SHIFT`; `0xFFF` → `TYPE0_SHORT_SIZE_MASK`; `0x80` → `HEADER_LONG_FORM_BIT`; `0x7FFFFu` → `SUBCHUNK_COMP_SIZE_MASK`.

- [ ] **C2. Replace scan kernel's private `parseEntropyHeader` / `parseRawStreamSize` / `skipEntropyStream` with `slz_wire_format.cuh` helpers** [risk: med — needs parameterization for pointer vs offset] [effort: m]
  - `scan_parse_kernel.cuh:32-58, 61-76, 79-111` reimplement what's in `slz_wire_format.cuh:178-281`.
  - Difference: scan version takes `pos` and returns advanced position; decode version mutates `src` pointer.
  - Either parameterize the common helpers, or document why the duplication is intentional and consolidate the constants only (C1).

- [ ] **C3. Move `SLZ_FRAME_*` constants from `walk_frame_kernel.cuh:34-46` to `common/gpu_wire_format.cuh`** [risk: low] [effort: xs]
  - Frame magic, PDM flag, etc. are ABI surface (matched against `decode/driver.zig::FrameWalkStatus`).

- [ ] **C4. Move `STREAM_HEADER_BYTES` + `OFF16_HEADER_BYTES` from `encode/lz_kernel.cu:36-37` to `common/gpu_wire_format.cuh`** [risk: low] [effort: xs]

- [ ] **C5. Add named constants for `assemble_kernel.cu::writeHuffChunkHdr` bit-pack** [risk: low] [effort: s]
  - `:154-159` uses bare `>>14`, `0xF`, `0x3FFF`, `<<18` for the 5-byte non-compact entropy chunk header.
  - Add `ENTROPY_HDR_DM1_HIGH4_SHIFT = 14`, `ENTROPY_HDR_DM1_HIGH_MASK = 0xF`, `ENTROPY_HDR_DM1_LOW_MASK = 0x3FFF`, `ENTROPY_HDR_COMP_BITS = 18` to `common/gpu_wire_format.cuh`.

- [ ] **C6. Add `static_assert(OFF32_LONG_ENTRY_TAG == (OFF32_LARGE_TAG >> 16))` to `encode/lz_format.cuh:30`** [risk: none] [effort: xs]
  - Enforces the cross-file constant relationship currently asserted only in a comment.

---

## D. Finish the CUDA error wrapping (Phase 7 follow-up)

Phase 7 wrapped 27 H2D/D2H/sync sites. The remaining ~20 sites in launches
and scan_gpu were missed.

- [ ] **D1. Wrap 10 `cuLaunchKernel != CUDA_SUCCESS` sites in `decode_dispatch.zig`** [risk: low — same pattern as Phase 7] [effort: s]
  - Sites: `:60, 84, 245, 261, 536, 561, 633, 662, 740, 767`.
  - Convert `if (launch_fn(...) != CUDA_SUCCESS) return error.BadMode;` to `try cudaCall(launch_fn(...));`.

- [ ] **D2. Wrap 6 silent CUDA-error sites in `scan_gpu.zig`** [risk: med — `walkMetaToHost` can currently return garbage on D2H failure] [effort: s]
  - Sites: `:60, 96, 186, 244, 260` (launch return discards), `:110, 168` (`_ = d2h(...)`), `:268, 282-290, 297` (`_ = h2d(...)`).
  - `walkMetaToHost`: returning `null` vs returning a garbage struct — fix the latter case.

- [ ] **D3. Audit `ensureDeviceBuf` and friends in `decode_context.zig`** [risk: low] [effort: xs]
  - Verify any remaining `_ = (cuda.cuXxx_fn orelse return false)(...)` callers propagate errors usefully.

---

## E. Finish the file / function splits (Phase 5/6 follow-up)

- [ ] **E1. Split `lz_decode_core.cuh` (471 LOC, over soft cap)** [risk: low — pure file split] [effort: m] [verify: PTX REG/STACK unchanged on every kernel]
  - Extract `lz_decode_raw.cuh` (decodeSubChunkRawMode, lines ~71-252) and `lz_decode_general.cuh` (decodeSubChunkGeneral, ~262-471).
  - `lz_decode_core.cuh` keeps `warpScanU32` + the warp copy helpers as the shared module.
  - Update `lz_kernel.cu` includes accordingly.

- [ ] **E2. Add bounded warpLiteralCopyBounded/warpMatchCopyBounded helpers and use in `decodeSubChunkGeneral`** [risk: med — hot decode path] [effort: m] [verify: PTX REG/STACK + roundtrip + bench]
  - Phase 5b explicitly deferred this; agent flagged it as "the helpers cover only the raw decoder; the general decoder open-codes 7 more copy loops with `dst_pos + i < dst_end_abs` bounds".
  - New helpers take an extra `dst_end_abs` parameter and per-store guard.
  - Replace ~7 inline copy loops in `decodeSubChunkGeneral`.

- [ ] **E3. Collapse pipelined-vs-sequential branch duplication in `decode_dispatch.zig`** [risk: med — touches the hot launch path] [effort: m]
  - ~80 LOC duplicated between the `if (use_pipeline)` branch (around L589+) and the `else` branch (around L703+).
  - With `NUM_PIPELINE_STREAMS = 1` hardcoded the "pipelined" loop runs exactly once doing identical work.
  - Decision: commit to single-stream and delete the pipelined branch, OR unify the two via a shared `launchLzKernel(...)` helper.
  - This is closely related to F5 (cap `pipeline_streams` array) and the user's "why so much Zig" question.

- [ ] **E4. Extract `parseInitialCopy` as 5th per-stream helper in `lz_header_parse.cuh`** [risk: low] [effort: xs]
  - Outer `parseSubChunkHeaders` is 29 LOC body (close to the claimed 26 but the initial-copy block is left inline).
  - Pulling it out gives an outer that is a pure sequence of per-stream parsers.

---

## F. Dead code to delete

- [ ] **F1. Delete `cuFuncSetAttribute_fn` slot + typedef in `cuda_api.zig:74-79`** [risk: none] [effort: xs]
  - Loaded by `module_loader.zig`; never called anywhere in `src/gpu/`.

- [ ] **F2. Delete dead encode FFI typedefs+slots in `encode/cuda_ffi.zig:24-26, 38-40`** [risk: none] [effort: xs]
  - `FnInit / FnDeviceGet / FnCtxCreate` and `cuInit_fn / cuDeviceGet_fn / cuCtxCreate_fn`. Never resolved or called.
  - This is tied to A6: encode currently piggybacks on decode's init. Either resolve these slots (A6 option 1) or delete them (A6 option 2).

- [ ] **F3. Delete `encode/module_loader.zig:19::pub var ctx: usize = 0;`** [risk: none] [effort: xs]
  - Comment admits it's "kept for symmetry"; zero readers.

- [ ] **F4. Delete unreachable `shared_ht` declaration in `encode/lz_kernel.cu:83-84`** [risk: none] [effort: xs]
  - Comment admits it's unreachable in production.

- [ ] **F5. Cap `pipeline_streams` array to actual size** [risk: low] [effort: xs]
  - `decode_context.zig:270` `pipeline_streams: [16]usize` but `NUM_PIPELINE_STREAMS = 1`.
  - Either use `[cuda.NUM_PIPELINE_STREAMS]` or document the over-allocation reason. Related to E3.

- [ ] **F6. DECISION NEEDED: Delete `scan_host.zig` (311 LOC) if pure-D2D is the only target** [risk: med — verify no callers need it] [effort: s]
  - CPU fallback path; lives behind `want_gpu_scan = false` which fires only if `SLZ_GPU_SCAN` env unset AND `d_compressed_src == null`.
  - Pure-D2D path forces `want_gpu_scan = true`. The fallback is dead in production but might be useful for testing.
  - User asked about this in the "why so much Zig" question — needs an explicit decision.
  - **If deleting:** also delete the `else gpu_scan orelse scan_host.scanForEntropyChunks(...)` fallback in `decode_dispatch.zig:443+` and the parseHuffHeader / parseType0StreamInfo / skipStreamHeader host helpers.

- [ ] **F7. Delete stale `tANS / tANS32` mentions in comments** [risk: none] [effort: xs]
  - `slz_wire_format.cuh:66, 196-197, 243-246, 264`
  - `lz_header_parse.cuh:56, 58, 65, 72, 111, 130, 275`
  - `decode_dispatch.zig:486-487, 705` "tANS is retired" archaeology comments.
  - GPU emits only types 0 (raw) and 4 (Huffman); the decoder still must SKIP type 1/5/6/7 for legacy frames but should document them as "decoder-skip-only".

---

## G. Stale references to update

- [ ] **G1. Fix `descriptors.zig:6, :52` references to deleted `lz_kernels.cuh`** [risk: none] [effort: xs]
  - Should now point at `compact_descs_kernels.cuh` / `scan_parse_kernel.cuh`.

- [ ] **G2. Fix `decode_context.zig:21` comment**: says `gpu.last_kernel_ns` but every importer uses `gpu_decode`. [risk: none] [effort: xs]

- [ ] **G3. Fix `levels.zig:8-10` line-number reference into `fast_framed.zig`** [risk: none] [effort: xs]
  - Cites L955-963 which appear to be wrong (or the lines rotted). Drop the line numbers, name the function instead.

---

## H. Polish / minor correctness

- [ ] **H1. Fix `ARCHITECTURE.md:39` integer-math bug** [risk: none] [effort: xs]
  - `ceil(50 / 32)` = `ceil(1)` = 1, but author meant `ceil(50.0 / 32)` = 2.
  - Reword to: `(50 + 31) / 32 = 2 iterations`.

- [ ] **H2. Fix `lz_decode_core.cuh:128` UB shift** [risk: low — works on nvcc, but undefined per C++ spec] [effort: xs]
  - `(1u << (lane + 1)) - 1u` is UB when `lane == 31` (shift count >= width).
  - Use `((2u << lane) - 1u)` or `__funnelshift_r`.

- [ ] **H3. Lift magic numbers in `decode_context.zig` + `decode_dispatch.zig` to named constants in `descriptors.zig`** [risk: none] [effort: s]
  - `[4096]HuffDecChunkDesc` → `MAX_HUFF_DESCS_PER_STREAM`
  - `[8192]RawOff16Desc` → `MAX_RAW_OFF16_DESCS`
  - `[16384]u32 first_subchunk_idx_buf` → `[walk_max_chunks]u32`
  - `131072` (per-subchunk scratch) → `PER_SUBCHUNK_SCRATCH`
  - `65536` (lo half offset) → `PER_SUBCHUNK_SCRATCH / 2` or `OFF16_LO_INTRA_SLOT_OFFSET`
  - `4096 * 4` (merged_huff cap) → `MAX_HUFF_DESCS_PER_STREAM * 4`

- [ ] **H4. Decide on `laneId()` helper** [risk: none] [effort: xs]
  - Defined in `gpu_warp.cuh:29`. Only 2 callers (`assemble_kernel.cu:324, 345`); the other ~23 sites open-code `threadIdx.x & LANE_MASK`.
  - Either convert all callers or delete the helper.

- [ ] **H5. Fix `read8safe` API redundancy in `gpu_byteio.cuh:53-59`** [risk: low] [effort: xs]
  - Takes `(p, pos, src_size)` where every caller passes `p = src + pos`.
  - Either rename param to make clear `p` is already offset, or change to `(base, pos, src_size)` and add internally.

- [ ] **H6. Cache `QueryPerformanceFrequency` in `cuda_api.zig::qpcMs`** [risk: none] [effort: xs]
  - Currently re-queries on every call. Frequency is fixed for the process lifetime.

---

## I. Docs improvements

- [ ] **I1. Update README kernel-entry table sources to point at `.cu` not `.cuh`** [risk: none] [effort: xs]
  - Decoder rows point at the per-kernel `.cuh` aggregator-includes, contradicting Invariant 1 "one .cu → one .ptx".
  - Should say `decode/lz_kernel.cu` for the LZ-aggregator kernels and `decode/huffman_kernel.cu` for Huffman.

- [ ] **I2. Remove concrete millisecond claims from `ARCHITECTURE.md`** [risk: none] [effort: s]
  - `:188-189` "dropped from 3.8 ms to 2.92 ms, 23 percent gain"
  - `:245-247` "from 40 registers to 56, dropped throughput 8 percent"
  - Design doc should be timeless; quantified deltas date the document.

- [ ] **I3. Change silesia L5 encode "4502 †" to "n/a †" in README:192** [risk: none] [effort: xs]
  - Listing the value invites quoting it out of context.

- [ ] **I4. Add field-access helpers (`subchunkCompSize`, `subchunkMode`, `subchunkIsLz`) to `common/gpu_wire_format.cuh`** [risk: low] [effort: xs]
  - Every consumer open-codes `(hdr & SUBCHUNK_LZ_FLAG_BIT)` etc.; `gpu_huffman.cuh` already exposes equivalent accessors (`lutSym1`, `lutTotalLen`) for parity.

- [ ] **I5. Extract `merge_huff_descs_kernel.cuh:23-71::appendRegion(...)` helper** [risk: low] [effort: xs]
  - 4 nearly identical loops differ only in `region_off`. Single-thread kernel, pure readability win.

---

## Suggested execution order

1. **A1-A5** (real bugs, batched into one commit "real bugs sweep"). ~20 min.
2. **A6 + F2 + F3** (encode-init coupling: decide whether to lift encode init or document the piggyback, then delete the dead FFI slots accordingly). ~1 hr.
3. **F1, F4, F5, F7, G1, G2, G3** (mechanical deletions and stale refs, one commit). ~30 min.
4. **B1** (apply readU32LE to greedy parser — single biggest perf-win candidate). ~10 min.
5. **B2-B9** (byte-IO consolidation finishing pass, one commit). ~1 hr.
6. **C1, C3, C4, C5, C6** (wire-format consolidation, skipping C2 which is the risky one). ~1 hr.
7. **D1, D2, D3** (CUDA error wrapping finishing pass). ~30 min.
8. **F6 DECISION + delete if user approves**: scan_host.zig (311 LOC). User input required.
9. **E1** (split lz_decode_core.cuh — pure file move, no behavior change). ~1 hr.
10. **E2** (bounded warp copy + apply to decodeSubChunkGeneral). ~1-2 hr. Requires careful bench.
11. **E3** (collapse pipelined-vs-sequential — depends on F5/F6 decisions). ~1-2 hr.
12. **C2** (scan kernel private parsers dedup). Highest risk; consider whether worth it. ~2 hr.
13. **H1-H6** (polish items, one commit). ~30 min.
14. **I1-I5** (docs improvements, one commit). ~30 min.

Each numbered step ends with: rebuild PTX, REG/STACK check, `zig build`,
5-SHA roundtrip. After the byte-IO and wire-format passes (5-6),
run a full sequential bench and update README perf numbers + hash (A3).

Total estimated effort if everything is done: 1-2 working days.
