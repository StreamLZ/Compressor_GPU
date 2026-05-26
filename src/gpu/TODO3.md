# GPU cleanup follow-up #3 — post bc082ef re-review

Aggregated findings from three parallel review agents:
- **Agent 1**: `common/` headers + docs (gpu_warp.cuh, gpu_byteio.cuh, gpu_huffman.cuh, gpu_wire_format.cuh, README.md, ARCHITECTURE.md)
- **Agent 2**: decode module delta (K5.1, K5.2, K5.5, K5.6, label-strip, two-phase Huffman)
- **Agent 3**: encode module + 32-stream end-to-end (encoder kernel, encode_huff.zig, wire-format consistency)

**Headline:** no CRITICAL findings. Wire-format is byte-exact end-to-end (Agent 3 verified). The 32-stream rewrite at `27b57a4` left ~15 stale "4-stream / 9 B / lanes 0..3" comments scattered across encode + decode + docs that the rewrite missed.

## CRITICAL

(none)

## HIGH

### H1. Encoder body-concat is single-lane byte-by-byte — 8× worse with 32 streams (Agent 3 H1)
`src/gpu/encode/huffman_kernel.cu:299-308`. Lane 0 serially copies all 32 per-stream scratch buffers into the output while 31 lanes idle. Likely the largest encode-kernel perf left on the table after the wire-format change. Fix: cooperative `warpCopy` across the warp (already defined in `assemble_kernel.cu:167-170`).

### H2. Encoder `stream_bytes[HUFF_NUM_STREAMS]` allocates 128 B on every lane = entire STACK:128 growth (Agent 3 H2)
`src/gpu/encode/huffman_kernel.cu:281`. Per-lane `uint32_t stream_bytes[32]` = the `__local_depot1[128]` in PTX. Only lane 0 reads it. Move to `__shared__`: one copy per block, 128 B total, kills STACK:128 and probably drops REG:56 toward REG:40.

### H3. Encoder kernel docs entirely stale (4-stream era throughout) (Agent 3 H3/H4/H5/H6/H7)
`src/gpu/encode/huffman_kernel.cu` lines 4, 217-227, 258, 275-276, 293, 299. Every comment block describes the old 4-stream / 9 B sub-header / lanes 0..3 / stream-3-derived design. The code at these sites already uses `HUFF_NUM_STREAMS` correctly, but the surrounding prose lies. Decode side has the matching banner at `decode/huffman_kernel.cu:358-372` — copy the disclaimer block verbatim or rewrite each comment.

### H4. ARCHITECTURE.md has 3 em-dashes — project rule violation (Agent 1 M2)
Project rule (both trackers, user-enforced): "No em-dashes (—) in `README.md` or `ARCHITECTURE.md`". Three slipped through this arc:
- `ARCHITECTURE.md:65` — "slzHuffDecode4StreamKernel (name retained for ABI compatibility —"
- `ARCHITECTURE.md:250` — "hot loop body — much larger than the header parser)"
- `ARCHITECTURE.md:273` — "prefix sum — 5 shuffles for 32-way fan-out"

README is clean.

### H5. gpu_huffman.cuh sub-header overhead claim contradicts README (Agent 1 H2)
`gpu_huffman.cuh:36-40` says "~0.1pp ratio at 64 KB sub-chunks"; `README.md:156-157` says "~0.4–0.5 pp". The .cuh accounts only for the sub-header byte cost (84 / 65536 ≈ 0.13%); the stream-boundary entropy loss (32 vs 4 boundaries) is the larger share. Pick the empirically-measured 0.4–0.5pp in both places.

### H6. Decode-side scan path has stale "9 B / 4 streams" docs in 5+ sites (Agent 1 M6 + Agent 3 H8/H9)
- `decode/huffman_kernel.cu:49-58` — HuffDecChunkDesc doc says "128 + 9 + sum(streams)" — most authoritative wrong place
- `decode/huffman_kernel.cu:51` — "(128 B weights + 9 B sub-header + 4 stream payloads)"
- `decode/descriptors.zig:23-24` — Zig mirror of the same
- `decode/scan_host.zig:110` — "[128 B weights][9 B sub-header][4 streams]"
- `decode/scan_host.zig:222` — same pattern
- `decode/lz_header_parse.cuh:75-77` — same pattern

Replace literals with `HUFF_BODY_HEADER_BYTES` / `HUFF_NUM_STREAMS` or the literal "93 B / 32".

### H7. `decode_dispatch.zig:653` ctx-wide sync banner misleads async callers (Agent 2 H1)
Three ctx-wide `cuCtxSynchronize()` calls (lines 653, 700, 799) claim "acceptable because front end of pipeline". But the async-decode contract (`slzDecompressAsync` sets `work_stream`) means callers may have queued work on sibling streams before reaching this function. A ctx-wide sync stalls every stream in the process. Either downgrade to `stream_sync_fn(default_stream)` or document the async-stall trade-off explicitly. K5.4 was marked DONE-option-B (document), but the docs say "front end" which is misleading when callers can be async.

### H8. `decode_dispatch.zig:658` flattens 6 scan_gpu failure modes to error.BadMode (Agent 2 H2)
`gpuPrefixSumChunksImpl` returns null on BackendNotAvailable / KernelMissing / OutOfDeviceMemory / KernelLaunchFailed / SyncFailed. At the call site all 6 → `error.BadMode`. The post-K5.1 error fan-out is undone here. The scan_gpu doc convention "null = fall back to host scan" doesn't apply to `gpuPrefixSumChunksImpl` (no host equivalent — GPU prefix sum is required). Fix: make this function return `GpuError!T`.

### H9. `streamlz_decoder.zig:111,115` flatten null returns from gpuWalkFrameImpl / walkMetaToHost (Agent 2 H3)
Same shape as H8 — K5.1's "thread the correct tag through every call site" stops at the walk_frame boundary. Either make those functions return `GpuError!T` or distinguish at the call site.

### H10. `streamlz_gpu.zig` catch sites use `err: anyerror` — new GpuError members fall through to CORRUPT_FRAME (Agent 2 H6)
`streamlz_gpu.zig:138-146` and `:299-309`. Switch is over `anyerror`. A new GpuError member would not be listed → falls through to `else => mapDecompressError(err)` → reported as `SLZ_ERROR_CORRUPT_FRAME` instead of "GPU couldn't do it, retry on host". Either change to `err: GpuError` switch or add an `else => -1` for the GPU-fallback path.

### H11. `encode_huff.zig` "137 fixed" capacity comments now stale (Agent 3 H10)
`encode_huff.zig:195, 343, 443` — "Huffman body worst case ≈ 137 fixed + count×11/8". The new fixed body header is `HUFF_BODY_HEADER_BYTES = 221`, not 137. The bound `count*2 + 256` is still safe (256 > 221), but the safety margin shrunk from 119 → 35 bytes silently. Use `HUFF_BODY_HEADER_BYTES + 64` so the constant auto-tracks.

## MEDIUM

### M1. `slzHuffEncode4StreamKernel` name with no ABI disclaimer in 5+ doc sites (Agent 1 M1)
`README.md:53, 69, 94, 108` + `ARCHITECTURE.md:65-66, 104, 327, 569, 571`. Decode kernel `slzHuffDecode4StreamKernel` gets the "name retained for ABI compatibility — it decodes 32 streams, not 4" disclaimer at one site in ARCHITECTURE; encode kernel appears 4+ times across both docs with no such note. First-time reader sees two "4Stream" kernels and assumes the wire format is still 4-stream.

### M2. `gpu_byteio.cuh:36` stale "Huffman 4-stream sub-header" comment (Agent 1 M3)
Now 32-stream. Change to "Huffman sub-header" or "Huffman 32-stream sub-header".

### M3. `gpu_huffman.cuh:31-34` layout block uses unbound `N` (Agent 1 M4)
`[HUFF_SUBHEADER_BYTES sub-header — (N-1) × u24 LE stream sizes; stream (N-1) size derived from total]` — `N` is never defined. Bind explicitly: `N = HUFF_NUM_STREAMS`.

### M4. ARCHITECTURE Resource utilization SHARED column misleads (Agent 1 M5)
`ARCHITECTURE.md:580-587` table says `slzHuffDecode4StreamKernel | SHARED: 0`. Static shared is 0, but the kernel uses 4 KB dynamic shared (`extern __shared__ uint32_t shared_lut[]`, sized at launch — `decode/huffman_kernel.cu:387`). The prose at `:577-578` ("Only the Huffman LUT-build kernels use shared memory") is then false. Fix: cell → `0 (+4096 dyn)` and footnote the prose.

### M5. No `static_assert(HUFF_NUM_STREAMS == WARP_SIZE)` (Agent 3 M1)
`gpu_huffman.cuh:41-46`. Both kernels silently assume one lane per stream. Bumping HUFF_NUM_STREAMS to 64 compiles but fails. Add the assert near the constant declaration.

### M6. `encode_huff.zig` KEEP IN SYNC comment has no enforcement (Agent 3 M2)
`encode_huff.zig:55-60`. The comment is the only safety net. A real check would be: (a) Zig header generated from .cuh at build time, or (b) debug-only kernel that returns HUFF_NUM_STREAMS for Zig to assert at module-load. Minimum: tighten the doc to "silent wrong-output if this and the .cuh disagree — scratch slab is wrong size AND host splits the input differently than kernel expects."

### M7. `AssembleDesc` has no `static_assert(sizeof(...))` ABI guard (Agent 3 M3)
`src/gpu/encode/assemble_kernel.cu:45-59`. Sibling structs `HuffEncDesc` and `CompressChunkDesc` have the guard; `AssembleDesc` does not. Encode-PTX is what the host reads back; silent drift if a field is added on one side only.

### M8. Helpers in `decode_dispatch.zig` re-resolve FFI fn pointers the orchestrator already resolved (Agent 2 M1)
`runHuffPredecode`, `runLzPipeline`, `finalizeOutput` each re-resolve `h2d_fn`, `launch_fn`, etc. Orchestrator also resolves them at entry. Either hoist into a `Fns` struct and thread it, or drop the orchestrator's resolution and resolve only where used. Mixed style currently.

### M9. Shadow consistency in `fullGpuLaunchImpl` (Agent 2 M2/M3)
Orchestrator shadows 10 request fields as locals; `finalizeOutput` accesses `req.dst_start_off` etc. directly. Pick one style consistently.

### M10. CPU-merge fallback at `decode_dispatch.zig:104-127` silently truncates if descriptors exceed `merged_huff_buf.len` (Agent 2 M4)
If `n_huff > MAX_HUFF_DESCS_PER_STREAM * 4`, the append closure silently drops entries but `n_huff` still reflects the un-truncated sum. Kernel launches `n_huff` blocks but reads only `m` valid entries; trailing entries read uninitialized device memory. Fix: assert or `n_huff = m` after merge.

### M11. `decode_dispatch.zig:686` D2H to `first_subchunk_idx_buf` has no length check on H2D path (Agent 2 M5)
Length-check exists for the gpu_scan path at line 718 but the H2D at line 685 is unconditional. If `chunk_descs.len > WALK_MAX_CHUNKS`, the host buffer overruns. Add explicit guard.

### M12. Two-phase Huffman decode `LUT_MAX_SYMS_PER_STEP` not static-asserted (Agent 2 M17)
`huffman_kernel.cu:272`. Phase 2 gate assumes `LUT_MAX_SYMS_PER_STEP == 2`. If bumped, gate + inner loop bound + refill threshold all silently desync. Add `static_assert(LUT_MAX_SYMS_PER_STEP == 2)`.

### M13. Encode kernel always writes full 93 B sub-header even for empty/tiny streams (Agent 3 M6)
For `src_size < HUFF_NUM_STREAMS`, 31 streams have zero bytes but 93 B sub-header still written. The encoder gates entropy coding for `lit_count == 0` / `off16_count < 32` but no "use 1-stream when src_size is too small to amortize 93 B subheader" gate. Ratio loss on pathological inputs only.

### M14. `encode_huff.zig:192` magic `< 32` collides semantically with HUFF_NUM_STREAMS (Agent 3 M7)
The "matches CPU >= 32 gate" is a CPU oracle policy threshold, not HUFF_NUM_STREAMS, but the coincidence will mislead. Name the constant: `const OFF16_HUFFMAN_MIN_COUNT: u32 = 32;`.

### M15. Encode kernel 32 inline `__shfl_sync` broadcasts (Agent 3 M8)
`encode/huffman_kernel.cu:282-284`. 32 warp-sync intrinsics inline-expanded is heavy on icache. Shared-mem alternative is strictly better (covered by M4 / H2).

### M16. `huff_dbg` env-var check is "cached" per-call not per-process (Agent 2 M14)
K6.63 cached the SLZ_HUFF_DBG getenv "once per fullGpuLaunchImpl invocation" but every decompress call still re-reads the env var. Rename to `huff_dbg_this_call` to make scope explicit, or lift to module-level via `std.once.Once`.

### M17. `huffman_kernel.cu:331` tail clamp `bit_count = MAX_CODE_LEN + 1` forces under-filled LUT lookup (Agent 2 M15)
If input ran out mid-symbol, clamp pretends sufficient bits. The LUT lookup may use garbage low bits. Output write is bounded by `out_pos < out_size` so the bytes go to valid slots but are wrong. Pre-existing pattern; worth documenting the safety argument.

### M18. `decode_dispatch.zig:786` `!self.pipeline_streams_created` check is redundant after `ensurePipelineStreams` at line 610 (Agent 2 M20)
Defensive but unreachable. Either delete or comment "defensive — ensurePipelineStreams above already guarantees this".

### M19. Sync-failure path inside pipeline-streams loop early-returns (Agent 2 H4)
`decode_dispatch.zig:860-862`. With `NUM_PIPELINE_STREAMS == 1` this is harmless. Future bump would leave earlier streams' work running on device. Document the trade-off or drain remaining streams first.

### M20. `gatherRawOff16` discards launch-fn error path silently (Agent 2 H5)
The fallback comment at `decode_dispatch.zig:183-195` (K5.9 decision) is sound but doesn't log the rc. Real misconfigurations are invisible. At minimum: log the launch rc via `std.debug.print` even when falling back.

## LOW / nitpick

### L1. `gpu_huffman.cuh:115` `LUT_NUM_SYMS_ESCAPE = 3` deserves a one-line note (Agent 1 L4)
Why 3 specifically? "Any non-{1,2} value triggers the escape branch; no encode-side meaning". Five words.

### L2. `gpu_huffman.cuh:131-133` "All four return uint8_t" antecedent ambiguous (Agent 1 L1)
Five functions declared; "All four" technically correct but "all four single-field accessors" would be clearer.

### L3. `gpu_wire_format.cuh:125` `STREAM_HEADER_BYTES` alias has no `[[deprecated]]` marker (Agent 1 L5)
K6.24's deprecation prose explains the alias but autocomplete still surfaces it. Mark `// DEPRECATED — use LZ_SUBSTREAM_COUNT_HDR_BYTES`.

### L4. `gpu_warp.cuh:44-46` `lastBitSet(x=0)` UB documented but not debug-asserted (Agent 1 L6)
CUDA `assert()` (NDEBUG-stripped) would catch misuse at the four call sites.

### L5. `gpu_wire_format.cuh:30-33` "retired tANS types" sentence is history-archaeology in a contract header (Agent 1 L8)
Move to a "History" line at file bottom or delete.

### L6. `gpu_wire_format.cuh:96-98` `SLZ_BLOCK_HDR_*` namespace collision (Agent 1 L9)
Lines 90-94 describe the outer-block 32-bit word; lines 95-98 describe the internal-block per-byte mask set. Two unrelated structures share the `SLZ_BLOCK_HDR_*` prefix. Either rename one set to `SLZ_INT_BLOCK_HDR_*` or add a separator banner.

### L7. `gpu_wire_format.cuh:113` `DEFAULT_SUB_CHUNK_CAP = LZ_BLOCK_SIZE` coincidence not static-asserted (Agent 1 per-file)
One-line `static_assert(DEFAULT_SUB_CHUNK_CAP == LZ_BLOCK_SIZE, "naming coincidence; see comment");` locks the relationship.

### L8. `README.md:29` Layout block uses `walk_max_chunks` lowercase alias (Agent 1 L2)
Canonical name is `WALK_MAX_CHUNKS` (K6.32). Use canonical.

### L9. `README.md:181-192` L5 silesia LZ + Huff sum < D2D (Agent 1 L3)
6.20 + 3.01 = 9.21 < 9.35 D2D. Plausible (D2D includes scan/walk/compact overhead the sub-times don't) but prose implies sum bounds D2D from above. Either re-explain or annotate.

### L10. `README.md:158` "4-stream / 4-lane" phrasing reads as one term (Agent 1 L10)
Use "the prior design's 4 streams / 4 active lanes".

### L11. `ARCHITECTURE.md:316-319` warp-sync prose could prevent future "fix-me" regression (Agent 1 L11)
Add follow-up: "(the warp does sync after the prefix-sum that precedes the loop)" to prevent a reader from adding a syncwarp the code intentionally lacks.

### L12. `decode_dispatch.zig:594-597` em-dashes in shadow-comment block (Agent 2 L1)
Three em-dashes. Replace with hyphens (consistency with the README/ARCHITECTURE rule, though .zig is technically exempt).

### L13. `lz_decode_general.cuh:69-75` perf-justification comment duplicated 3 times (Agent 2 L5)
Extract to "PERF NOTE: see top-of-function" reference and keep terse reminders at the three call sites.

### L14. `lz_decode_general.cuh:33-39` `TokenType` enum overspecified (Agent 2 L4)
Only `TOKEN_TYPE_LONG_LITERAL` is compared by name; the other four values are unused symbolic tags.

### L15. `huffman_kernel.cu:38-39` `MAX_CODE_LEN` / `LUT_SIZE` shadow `HUFF_LUT_INDEX_BITS` / `HUFF_LUT_ENTRIES` (Agent 2 L7)
Two names for the same value. Alias or use the shared constant directly.

### L16. `huffman_kernel.cu` phase-2 banner conflates `MAX_CODE_LEN` (LUT index width = 10) with `HUFF_MAX_CODE_LEN` (height = 11) (Agent 2 L8)
"Refill threshold is 2*(MAX_CODE_LEN+1) = 22 bits" — math correct but easy to misread. Spell out the relationship.

### L17. `decode_dispatch.zig:271-296` `dumpScanIfRequested` reads SLZ_DUMP_SCAN twice (Agent 2 L9)
Cache once at function entry. Saves a libc call on non-default path.

### L18. `module_loader.zig:120, 145` PTX filenames hardcoded twice (Agent 2 L10)
Extract `const LZ_PTX_NAME = "lz_kernel.ptx";` etc.

### L19. `decode_context.zig:309` and `descriptors.zig:46` mix `walk_max_chunks` / `WALK_MAX_CHUNKS` (Agent 2 L11)
Migrate all call sites to the SCREAMING_SNAKE name; drop the alias.

### L20. `decode_dispatch.zig:1-11` module banner refers to "the four `last_*_kernel_ns` telemetry vars" (Agent 2 L13)
`driver.zig` declares three (`last_kernel_ns`, `last_lz_kernel_ns`, `last_huff_kernel_ns`). Off-by-one.

### L21. `decode_dispatch.zig` repeated `@intCast(t.untilNow(iv, .awake).toNanoseconds())` (Agent 2 L14)
~11 sites. Worth a tiny helper `inline fn nsSince(t0, iv) i64`.

### L22. `decode_context.zig:271-280` eight `d_n_*_scratch` 4-byte allocations (Agent 2 L15)
Could be slots within one contiguous counters buffer (pattern used at `d_compact_counts`). Perf-neutral but precedent exists.

### L23. `vulkan_driver.zig:705-722` per-call vkProc re-resolution (Agent 2 L17)
CUDA side caches; Vulkan side re-resolves on every decompress. Cache to match, or document why per-call resolution is acceptable (Vulkan dispatch table is per-instance).

### L24. `descriptors.zig:130` legacy alias `walk_max_chunks` still in use (Agent 2 L20)
Tracker for alias removal.

### L25. `decode_dispatch.zig:476, 857` `for (0..NUM_PIPELINE_STREAMS)` should use comptime if (Agent 2 M12/L21)
With `NUM_PIPELINE_STREAMS == 1`, single-iter loop. `comptime if` makes the intent explicit.

### L26. `lz_decode_general.cuh:33` `enum TokenType : uint32_t` is the only enum class in decode tree (Agent 2 L22)
Inconsistent with neighbouring `static constexpr uint32_t TOKEN_FOO = ...` pattern in `slz_wire_format.cuh:22-29`.

### L27. `encode/huffman_kernel.cu:299` "// Concatenate the 4 per-stream scratch buffers" (Agent 3 L2)
Should be "HUFF_NUM_STREAMS per-stream scratch buffers".

### L28. `encode/huffman_kernel.cu:288` "// 128-byte weights — 32 lanes pack 4 entries each" magic numbers (Agent 3 L1)
Use `HUFF_WEIGHTS_BYTES` / `WARP_SIZE`.

### L29. `encode/huffman_kernel.cu:310` `HUFF_WEIGHTS_BYTES + HUFF_SUBHEADER_BYTES` could be `HUFF_BODY_HEADER_BYTES` (Agent 3 M5)

### L30. `encode_huff.zig:65` `+ 64` per-stream scratch headroom rationale unclear (Agent 3 L14)
Was sized at N=4; now N=32 makes the +64 32× more relative slack. Document the rationale ("+64 covers trailing byte flush + partial-byte rounding from height-limit clamp") or tighten.

### L31. `encode_huff.zig:81` redundant memset before encode kernel that writes every entry (Agent 3 L15)
`out_sizes[block_id]` is written for every block (including 0 for empty). Memset is defensive but redundant.

### L32. `huffman_kernel.ptx` (decode side) line 4717 `__local_depot1[128]` (Agent 3 L18)
Confirms STACK:128 is entirely `stream_bytes[]` on encode side. H2 / M4 will recover this.

### L33. `encode/huffman_kernel.cu:35-46` `HuffEncDesc.dst_capacity` field is unused (Agent 3 L19)
Kernel docstring explicitly says it doesn't bounds-check. Dead 4 bytes per descriptor. Remove with ABI bump or document as advisory-only.

### L34. `encode_assemble.zig:289` `grid_x: u32 = n_chunks + 1;` theoretical u32 overflow (Agent 3 L9)
Almost certainly unreachable in practice (walk_max_chunks caps lower). Defensive check is cheap.

### L35. `decode/huffman_kernel.cu:405` "lanes 0..30 / lane 31" hardcoded (Agent 3 L6)
Should be `0..HUFF_NUM_STREAMS-2` / `lane HUFF_NUM_STREAMS-1`.

### L36. `module_loader.zig` `slzHuffEncode4StreamKernel` load needs ABI-compat-name comment (Agent 3 L18 + Agent 1 M1)
Matches the pattern on the decode side; encode side missing.

## Per-file impact summary

### CRITICAL → HIGH cluster: the 4-stream-era doc remnants
The `27b57a4` rewrite caught the README, ARCHITECTURE, `gpu_huffman.cuh` constants block, and the decode-side kernel banner. It missed:
1. **Entire encode kernel banner + 6 in-function comments** (H3, Agent 3)
2. **HuffDecChunkDesc descriptor doc** (H6 #1)
3. **5 decode-side comments in lz_header_parse.cuh / scan_host.zig / descriptors.zig** (H6 rest)
4. **gpu_byteio.cuh:36** (M2)
5. **encode_huff.zig "137 fixed" capacity comments** (H11)

All cosmetic but high-confusion-value. The most impactful single fix is the encode kernel comment block (H3) — it's the obvious thing a future maintainer reads first when investigating the encoder.

### Error-handling gaps (decode side)
K5.1 fanned BadMode into 7 GpuError members across `decode_dispatch.zig` directly, but the scan_gpu / walk_frame / streamlz_gpu boundaries silently flatten back. Three findings (H7, H8, H9, H10) all variants of the same pattern: the K5.1 spirit stops at module boundaries that K5.1 didn't directly touch.

### Encoder perf regressions from 32-stream
Two real perf issues (H1: lane-0 serial concat; H2: per-lane STACK:128) account for the encode-side REG:35→56 STACK:0→128 growth. Both have concrete fixes (warp-cooperative copy; move to shared mem). Worth measuring with the encode benchmark before/after.

### Missing static_asserts
Three (M5: HUFF_NUM_STREAMS == WARP_SIZE; M7: AssembleDesc sizeof; M12: LUT_MAX_SYMS_PER_STEP == 2) — each protects an invariant that a future change could silently break.

## Next session

Suggested order:
1. **H3 / H6 / M2 — stale comment sweep** (cheap, high-confusion-value). 10-15 minute task.
2. **H4 — ARCHITECTURE em-dashes** (project rule violation). 1 minute.
3. **H1 / H2 — encode kernel perf** (warp-cooperative concat + shared `stream_bytes`). Requires PTX rebuild + REG/STACK verify + encode bench.
4. **H5 — gpu_huffman.cuh / README contradiction** on sub-header overhead (pick one number).
5. **H7-H10 — error-handling deepening** (scan_gpu / walk_frame return GpuError; streamlz_gpu switch over GpuError directly).
6. **H11 — encode_huff.zig "137 fixed" → HUFF_BODY_HEADER_BYTES**.
7. **M5 / M7 / M12 — three static_asserts**.
8. The 36 LOW/nit items as bulk cleanup batches when needed.

No items block the K5/K6 arc closing. Wire-format is correct end-to-end; everything is polish or doc-hygiene or non-correctness-affecting error handling.
