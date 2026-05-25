# GPU cleanup — DONE log (2026-05-25)

The 14-batch follow-up to the original 8-phase cleanup arc landed in
this session as commits `5d475d7..94490e0` on `main`. This file is a
post-cleanup record so a future picker-upper can find the work without
git archaeology.

Original 8-phase cleanup arc: commits `7480762..893e043`.

## What landed (in execution order)

| Commit  | Items   | What                                                                                |
|---------|---------|-------------------------------------------------------------------------------------|
| 5d475d7 | A1–A5   | Real-bugs sweep: kernel-name telemetry typo, silent 16384-chunk cap, README hash, ASCII >= |
| f7816a4 | A6/F2/F3| Encode → decode init coupling documented; dead FFI slots deleted                    |
| f08f8f0 | F1/F4/F5/F7/G1-G3 | Mechanical deletions + stale-ref sweep                                    |
| 89e6559 | B1      | `readU32LE` applied to greedy parser (3 sites)                                      |
| c8c61a9 | B2–B9   | Byte-IO consolidation finishing pass; `readU32BE`/`writeBE24` added                 |
| 7b0657b | C1/C3-C6| Wire-format consolidation (scan-kernel constants, SLZ_FRAME_*, OFF32 static_assert) |
| a82a0f6 | D1–D3   | CUDA error wrapping: more `try cudaCall(...)` sites, silent d2h/h2d/memset fixed    |
| 64aedce | F6      | scan_host.zig KEPT (decision: it owns the H2D-entry scanner)                        |
| 7a5fc01 | E1      | Split lz_decode_core.cuh (471 LOC) into core / raw / general                        |
| 83e2a58 | E2      | Bounded warp-copy helpers added, applied to decodeSubChunkGeneral                   |
| 1b82944 | E3      | Deleted non-pipelined fallback in fullGpuLaunchImpl (was producing zero literals for L3+) |
| 68b1127 | C2 v1   | First C2 attempt — docs-only punt (later superseded)                                |
| 31726f0 | C2 v2   | Actual cursor-free `parseEntropyHdrFields` / `parseType0HdrFields` extraction       |
| c1cc1f6 | H1–H6   | ARCH math fix, UB shift fix, named constants, `laneId()` deleted, read8safe API, QPC freq cached |
| 3db0ffb | I1–I5   | Docs polish + helper consolidation (subchunkIs/Mode/Comp, appendRegion)             |
| 94490e0 | CRITICAL| Fix C2-introduced `long_form` inversion in scan_parse_kernel.cuh                    |

## Re-review (5 agents, post commit 3db0ffb)

The 5-agent reviewers ran again after the 14 batches landed. Outcome:
- 1 CRITICAL bug found and fixed in 94490e0 (the C2 inversion).
- ~80 follow-up findings across HIGH / MEDIUM / LOW priorities.
- No regressions in production paths (CLI roundtrips byte-identical
  throughout; SLZ_GPU_SCAN=1 roundtrip verifies the fixed scan path).

The next round of work, if undertaken, should pick from this list:

### HIGH-priority follow-ups

- Encode-side never adopted `cudaCall(rc)`. ~15 silent CUDA error
  sites in `encode_lz.zig`, `encode_huff.zig`, `encode_assemble.zig`
  mirror the D2 fix on the decode side.
- `huff_off16hi_data` / `huff_off16lo_data` share one allocation
  (`encode_huff.zig:265,268`). Comment-only contract preventing
  double-free; make exactly one a typed non-owning alias.
- `gatherRawOff16` launch silently falls through to D2D
  (`decode_dispatch.zig:171`). Comment claims self-gating makes
  over-launch safe; if so, the fallback is dead code — `try cudaCall`
  unconditionally.
- `_ = sf(huff_stream)` at `decode_dispatch.zig:540` missed by D1-D3.
- `scanParseHuffHeader` return value discarded at every call site
  (`scan_parse_kernel.cuh:170,177,203,220`). Either make `void` or
  propagate failure.
- `decodeSubChunkGeneral` lacks `__noinline__` while sibling
  `decodeSubChunkRawMode` has it (register-pressure consideration per
  the raw decoder's banner).

### MEDIUM-priority follow-ups

- `scan_host.zig:85,174,181` still hardcodes `131072` / `65536` —
  H3 missed this file. The .cuh has `static_assert` on the value;
  Zig scanner can drift independently.
- `scan_parse_kernel.cuh` magic numbers C1 missed: `>> 4 & 0x7` (4
  sites should use `CHUNK_TYPE_SHIFT`/`CHUNK_TYPE_MASK`), `+ 3` for
  sub-chunk header (should be `SUBCHUNK_HDR_BYTES`), `8u` for
  initial-copy (should be `INITIAL_LITERAL_COPY_BYTES`), `+ 7` / `+ 4`
  for paired headers, open-coded u16 LE read at line 188.
- `walk_frame_kernel.cuh` has the densest magic-number cluster
  remaining: block-header bit-fields (`0x0F`/`0x7F`/`0x80`), frame
  flag bits (`0x01`/`0x02`), the `262144`/`65536`/`131072` trio for
  sub_chunk_cap derivation.
- `gpuFrameAssembleImpl` has a 16-arg signature; factor into 2-3
  named structs (`FramePreamble`, `ChunkLayout`).
- `slzFrameAssembleKernel` has 2 open-coded BE3/LE4 writes B4 didn't
  reach (`assemble_kernel.cu:343-345, 394-398`). Wants a new
  `storeU32LE` helper next to `storeU16LE`.
- `gpu_byteio.cuh:7-12` endianness rule comment didn't get updated
  for `readU32BE` (still says "two conventions").
- Encode `driver.zig:55-107` has 5 dead facade wrappers (`gpuCompress`,
  `gpuEncodeHuff`, etc.) with zero callers; every real caller uses
  `*Impl`.
- F7 not fully applied — `lz_header_parse.cuh:50,113` still mentions
  chunk_type 1 and 6 by old names.
- `gpuFrameAssembleImpl` H2D-on-default-stream then launch-on-work_stream
  pattern is correct-but-subtle; async H2D on `heavy_stream` would be
  more obviously right.
- Encode-side `EncodeContext` has 21 `(ptr, size)` DevBuf pairs and
  no `deinit()`; project memory tags this deferred but worth confirming.

### LOW-priority nits

- `lz_decode_raw.cuh:147,164` uses non-`constexpr` `if (OFF16_SPLIT)`
  while the parallel path uses `if constexpr`. Modernize.
- `match_len > MIN_PARALLEL_MATCH_LEN - 1` should be `>=` in
  lz_decode_core.cuh.
- `descriptors.zig:117` `walk_max_chunks` (snake_case) vs
  `MAX_HUFF_DESCS_PER_STREAM` (SCREAMING_SNAKE) — pick one.
- README post-E1 layout block doesn't list `lz_decode_raw.cuh` /
  `lz_decode_general.cuh`.
- `decode/cuda_api.zig` `cached_qpc_freq` is racy (idempotent though).
- `decode/decode_dispatch.zig:91` allocates ~320 KB on the stack
  (`merged_huff` array). Below default 8 MB but worth a comment.
- `gpuScanChunks` has a vestigial `first_subchunk_idx` parameter
  documented "Vestigial — only `.len` is consulted." One caller; drop it.

## Project rules (still in force)

1. NEVER `git checkout` / `restore` / `revert` without explicit user
   approval.
2. NEVER run benchmarks in parallel. Sequential tool calls only.
3. NEVER filter build / test / benchmark output.
4. No em-dashes (—) in `README.md` or `ARCHITECTURE.md`.
5. Always report decode + encode speed alongside ratio.
6. GPU perf in milliseconds with cuEvent timer (`-db` mode).
7. No commits unless asked. No pushes unless asked.
8. REG/STACK verification: when touching kernel code, always rebuild
   PTX and cite REG/STACK numbers in the commit message.

## Build / verify (Windows PowerShell)

```powershell
cmd.exe /c "tools\build_gpu.bat"      # decode PTX
cmd.exe /c "tools\build_gpu_enc.bat"  # encode PTX
zig build -Doptimize=ReleaseFast -Dgpu=true

# Roundtrip baseline SHAs:
#   enwik8:  2B49720EC4D78C3C9FABAEE6E4179A5E997302B3A70029F30F2D582218C024A8
#   silesia: 86817A72FC4F404A0330247F69C1278DE8AD058AC1401A9CB1AE174FF5356C74
# GPU-scan path: SLZ_GPU_SCAN=1 ./streamlz -d ... (default CLI uses host scanner).
```

Deferred items (tracked in user memory `project_gpu_deferred_cleanup.md`):
- `scanBlock` structural split (326 LOC greedy parser, highest perf risk)
- `findMatchChain` structural split (~190 LOC chain parser, dense state)
- `int lane` vs `uint32_t lane` standardization
- `DevBuf` struct + `kernelParams` comptime helper (ergonomic only)
- `gpuEncodeXxxHuffImpl` three-near-clones dedup
