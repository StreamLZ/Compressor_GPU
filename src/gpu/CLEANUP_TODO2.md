# GPU cleanup follow-up #2 — post-compact handoff

**Status as of commit `1212a2e` (K6d):** the bulk of the punch list is
landed. See the **Status table** immediately below for per-item disposition.

---

## Status table

Commits landed this arc: `712d593` (K1) → `651392a` (K2) → `01e9e38` (K3)
→ `7dfe3e6` (K4) → `b552f02` (K5 easy) → `b2641d1` (K5b docs)
→ `9e30a6d` (K6a docs) → `c025e21` (K6bc naming + CUDA polish)
→ `1212a2e` (K6d Zig polish).

| Section | Item | Status | Commit |
|---|---|---|---|
| K1 | C1 off32 packed header | DONE | `712d593` |
| K1 | C2 Kraft sum UB shift | DONE | `712d593` |
| K1 | C3 buildCanonicalCodes guard + loop bound | DONE | `712d593` |
| K1 | C4 dead else-if (chain parser) | DONE | `712d593` |
| K1 | C5 dead ternary | DONE | `712d593` |
| K1 | C6 dead `if (chunk_len < 3) break;` | DONE | `712d593` |
| K1 | C7 d_input_override silent fallback | DONE | `712d593` |
| K1 | C8 dead errdefer × 3 | DONE | `712d593` |
| K1 | C9 redundant __shfl_sync in lz_decode_general | DONE | `712d593` |
| K1 | C10 overflow-safe gather bounds check | DONE | `712d593` |
| K2 | K2.1 writeU32LE | DONE | `651392a` |
| K2 | K2.2 readU64LE | DONE | `651392a` |
| K2 | K2.3 writeLE24 in lz_token_emit | DONE | `651392a` |
| K2 | K2.4 writeBE24 / writeU32LE in assemble | DONE | `651392a` |
| K2 | K2.5 scan_host magic 131072 / 65536 | DONE | `651392a` |
| K2 | K2.6 scan_host readBE24 helper × 5 | DONE | `651392a` |
| K2 | K2.7 scan_host CHUNK_TYPE_SHIFT/MASK × 4 | DONE | `651392a` |
| K3 | K3.1-K3.6 scan_parse_kernel magic numbers | DONE | `01e9e38` |
| K3 | K3.7 SLZ_FRAME_FLAG_* constants | DONE | `01e9e38` |
| K3 | K3.8 SLZ_BLOCK_HDR_* constants | DONE | `01e9e38` |
| K3 | K3.9 SLZ_CHUNK_SIZE_BYTES + sub_chunk_cap | DONE | `01e9e38` |
| K3 | K3.10 CHUNK_FLAG_UNCOMPRESSED / MEMSET | DONE | `01e9e38` |
| K3 | K3.11 NEXT_HASH_ENTRIES | DONE | `01e9e38` |
| K3 | K3.12 SC_TAIL_PER_CHUNK_BYTES / CHUNK_INTERNAL_HDR_BYTES | DONE | `01e9e38` |
| K3 | K3.13 RAW_CHUNK_HDR_BYTES clarifying comment | DONE | `01e9e38` |
| K3 | K3.14 HUFF_VS_RAW_HDR_OVERHEAD | DONE | `01e9e38` |
| K3 | K3.15 LARGE_OFFSET_THRESHOLD = OFF32_LARGE_TAG | DONE | `01e9e38` |
| K3 | K3.16 EXT_LENGTH_THRESHOLD moved to common | DONE | `01e9e38` |
| K3 | K3.17 static_assert(OFF16_HILO_SPLIT_OFFSET == LZ_BLOCK_SIZE) | DONE | `01e9e38` |
| K3 | K3.18 INITIAL_RECENT_OFFSET in encode/lz_kernel | DONE | `01e9e38` |
| K4 | K4.1 delete 5 dead encode facade wrappers | DONE (kept `g_default`) | `7dfe3e6` |
| K4 | K4.2 EncodeContext.deinit | DONE | `7dfe3e6` |
| K4 | K4.3 DecodeContext.deinit + cuStreamDestroy FFI | DONE | `7dfe3e6` |
| K4 | K4.4 gpuFrameAssembleImpl 16-arg → struct | DONE | `7dfe3e6` |
| K4 | K4.5 huff_off16 no-clobber assert | DONE | `7dfe3e6` |
| K4 | K4.6 beginKernelTiming leak on launch failure | DONE | `7dfe3e6` |
| K5 | K5.1 GpuError subtyping | DEFER — heavy refactor, separate session |  |
| K5 | K5.2 fullGpuLaunchImpl extraction | DEFER — 423 LOC, hot path |  |
| K5 | K5.3 Move stack buffers to DecodeContext | DONE | `b552f02` |
| K5 | K5.4 ctx-wide sync_fn document/migrate | DONE (option B — document) | `b2641d1` |
| K5 | K5.5 Template decodeSubChunkGeneral on OFF16_SPLIT | DEFER — needs perf evaluation |  |
| K5 | K5.6 decodeSubChunkGeneral 17-arg → struct | DEFER — hot path, medium risk |  |
| K5 | K5.7 parseRawStreamSize / parseEntropyHeader SAFETY | DONE (option 2 — comments) | `b2641d1` |
| K5 | K5.8 cached_qpc_freq → std.atomic.Value | DONE | `b552f02` |
| K5 | K5.9 gatherRawOff16 fallback contract | DONE (option 2 — keep fallback) | `b552f02` |
| K5 | K5.10 module_loader.init tri-state enum | DONE | `b552f02` |
| K5 | K5.11 ensurePipelineStreams GpuError!void | DONE | `b552f02` |
| K6 | K6.2-K6.4 byteio docs | DONE | `9e30a6d` |
| K6 | K6.5 gpu_warp.cuh layout heading split | DONE | `9e30a6d` |
| K6 | K6.6 encode/lz_kernel.cu "below"→"above" | DONE | `9e30a6d` |
| K6 | K6.7 lz_format.cuh include list | DONE | `9e30a6d` |
| K6 | K6.8-K6.14 README polish | DONE | `9e30a6d` |
| K6 | K6.15-K6.21 ARCHITECTURE polish | DONE | `9e30a6d` |
| K6 | K6.22 RAW_CHUNK_TYPE comment | DONE | `c025e21` |
| K6 | K6.23 SLZ_CHUNK_TYPE_MASK rename | NOT DONE — would break encode/decode ABI |  |
| K6 | K6.24 STREAM_HEADER_BYTES → LZ_SUBSTREAM_COUNT_HDR_BYTES | DONE (alias) | `c025e21` |
| K6 | K6.25 ENTROPY_HDR_* layout comment | DONE | `c025e21` |
| K6 | K6.26 SLZ_FRAME_MIN_HDR_SIZE width | NOT DONE — type-churn for cosmetic gain |  |
| K6 | K6.27 lutTotalLen / lutNumSyms return uint8_t | DONE | `c025e21` |
| K6 | K6.28 LUT_NUM_SYMS_TAG_* vs MAX_SYMS_PER_STEP | DONE | `c025e21` |
| K6 | K6.29 lutSymPair endianness comment | DONE | `c025e21` |
| K6 | K6.30 static constexpr → inline constexpr | NOT DONE — single-TU project, no ODR risk today |  |
| K6 | K6.31 Fibonacci naming symmetry | DONE (aliases) | `c025e21` |
| K6 | K6.32 walk_max_chunks → WALK_MAX_CHUNKS | DONE (alias) | `c025e21` |
| K6 | K6.33 MIN_PARALLEL_MATCH_LEN comparison polarity | NOT DONE — perf-neutral, defer |  |
| K6 | K6.34 lz_decode_raw.cuh `if constexpr` | NOT DONE — touches hot decoder, defer |  |
| K6 | K6.35 static_assert(WARP_SIZE == 32) | NOT DONE — defer |  |
| K6 | K6.36 LAST_BIT_SET helper | NOT DONE — defer |  |
| K6 | K6.37 lz_decode_general overflow check | NOT DONE — defer with K5.6 |  |
| K6 | K6.38 prefetched_token decl placement | NOT DONE — defer |  |
| K6 | K6.39 redundant break in per-block loop | DONE | `c025e21` |
| K6 | K6.40 TokenType enum | NOT DONE — defer with K5.6 |  |
| K6 | K6.41 delta-literal trailing helper extract | NOT DONE — defer |  |
| K6 | K6.42-K6.44 __restrict__ + shfl comments | NOT DONE — defer (low-priority polish) |  |
| K6 | K6.45 walkReadF32LE memcpy bit-cast | DONE | `c025e21` |
| K6 | K6.46-K6.50 huffman_kernel.cu micro-polish | NOT DONE — defer (low-priority) |  |
| K6 | K6.51 SLZ_GUARD_SINGLE_THREAD macro | DONE | `c025e21` |
| K6 | K6.52 (uint32_t)INITIAL_LITERAL_COPY_BYTES cast | NOT DONE — defer |  |
| K6 | K6.53 chunk_type 1/6 naming | NOT DONE — defer |  |
| K6 | K6.54-K6.60 encode CUDA misc polish | NOT DONE — defer |  |
| K6 | K6.61 __syncwarp() comment in assemble | DONE | `c025e21` |
| K6 | K6.62 decode_dispatch `var dd = s;` shadow | DONE | `1212a2e` |
| K6 | K6.63 cache SLZ_HUFF_DBG getenv | DONE | `1212a2e` |
| K6 | K6.64-K6.85 decode Zig misc polish | NOT DONE — defer |  |
| K6 | K6.86 cuda_api.zig ctx → ?usize | NOT DONE — too invasive (used pervasively as usize) |  |
| K6 | K6.87 NUM_PIPELINE_STREAMS = 1 history comment | NOT DONE — defer |  |
| K6 | K6.88 levels.zig drop unused level param | DONE | `1212a2e` |
| K6 | K6.89-K6.102 encode Zig misc polish | NOT DONE — defer |  |
| K6 | K6.103 update old CLEANUP_TODO.md DONE table | NOT DONE — see "Next steps" below |  |

**Summary:** 6 of 6 K1 items, 7 of 7 K2 items, 18 of 18 K3 items, 6 of 6
K4 items, 7 of 11 K5 items (4 deferred), and ~30 of 103 K6 items —
focused on the highest-value documentation, naming, and small-radius
polish work. The remaining K6 items are all low-priority follow-up
polish; future passes can chip away at them in batches similar to K6a-d.

## Deferred items — next-session pickup

The four K5 architectural items are real refactors that need their own
session each:

- **K5.1 GpuError subtyping** — fan out `error{BadMode}` into a richer
  error set (CudaNotAvailable / OutOfDeviceMemory / CudaLaunchFailed /
  CudaSyncFailed / CudaCopyFailed / KernelMissing / BadMode). Thread the
  comptime kind tag through `cudaCall` and every call site. Heavy
  mechanical change.
- **K5.2 fullGpuLaunchImpl extraction** — split the 423-LOC, 11-param
  function into `runHuffPredecode` / `runLzPipeline` / `finalizeOutput`
  plus a `DecodeRequest` struct for the 6 invariant params.
- **K5.5 template decodeSubChunkGeneral on OFF16_SPLIT** — specialize
  the general decoder identically to the raw-mode decoder. Needs careful
  REG/STACK + decode-bench evaluation; J4's experiment with __noinline__
  on the same function showed PTX numbers can move adversely.
- **K5.6 decodeSubChunkGeneral 17-arg → struct** — group into
  `DecodeStreams` / `DecodeOutput` + 2 scalars. Touches the hottest
  decode path so any REG/STACK regression must be measured.

The ~70 deferred K6 nits are all small-radius polish (mostly
`__restrict__` adds, helper extractions, comment additions, naming
consistency). They can be batched by file in future sessions; nothing
blocks progress on K5.

## Verification baselines retained

- enwik8 (100 MB) SHA: `2B49720EC4D78C3C9FABAEE6E4179A5E997302B3A70029F30F2D582218C024A8`
- silesia_all.tar (212 MB) SHA: `86817A72FC4F404A0330247F69C1278DE8AD058AC1401A9CB1AE174FF5356C74`
- `-gpu` enwik8 L1 compressed size: 58,637,026 bytes (matches `cfca2aa` baseline).
- Every batch above re-verified both SHAs across `-gpu` L1/L3/L5 enwik8
  + L3/L5 silesia, plus CPU-encode + CPU-decode L1 via the CPU-only
  binary. REG/STACK cited in each commit message.

---

## Original handoff (preserved below for reference)

This is the SECOND-pass punch list, gathered from a 5-agent textbook-grade
re-review run against HEAD `a7a9f5f`. Treat this file as the source of
truth for what's left. Re-reading prior conversation is not required;
everything you need to act is here.

## Context

The cleanup arc is at commits `5d475d7..a7a9f5f` (16 commits in 4 phases):

- Original 14-batch arc + post-review J1-J4 follow-ups landed in:
  `5d475d7, f7816a4, f08f8f0, 89e6559, c8c61a9, 7b0657b, a82a0f6, 64aedce,
   7a5fc01, 83e2a58, 1b82944, 68b1127, 31726f0, c1cc1f6, 3db0ffb, 94490e0,
   49a5391, 166a2db, e89c1cd, a7a9f5f`. See `CLEANUP_TODO.md` for the DONE
  table of those commits.

After J1-J4 landed, the 5 agents re-ran with stronger textbook framing
(every variable, every indent, every comment). They returned **2 CRITICAL**
+ **26 HIGH** + **45 MEDIUM** + **~50 LOW** findings — collected and
prioritised into K1-K6 below.

## Project rules (CRITICAL — user-enforced)

1. **NEVER `git checkout` / `restore` / `revert`** without explicit
   user approval. If something broke, fix forward.
2. **NEVER run benchmarks in parallel.** Sequential tool calls only.
3. **NEVER filter build / test / benchmark output** through `grep`/`tail`/etc.
4. **No em-dashes (—)** in `README.md` or `ARCHITECTURE.md`.
5. **Always report decode + encode speed alongside ratio.**
6. **GPU perf in milliseconds** with cuEvent timer (`-db` mode), never
   wall-clock for kernel timing.
7. **No commits unless asked. No pushes unless asked.**
   *Exception this arc:* the user pre-authorized commit-per-batch via
   AskUserQuestion earlier. Honor that — commit each batch, do not push.
8. **REG/STACK verification:** when touching kernel code, always rebuild
   PTX and cite REG/STACK numbers in the commit message (for every kernel,
   even if unchanged).
9. **State wins exactly.** Never soften a Zstd-narrowly-ahead measurement.
   See `feedback_state_wins.md`.

## Build / verify commands (Windows PowerShell)

```powershell
# Rebuild decode PTX (after touching any decode/*.cu / *.cuh)
& "$env:ComSpec" /c "tools\build_gpu.bat"

# Rebuild encode PTX (after touching any encode/*.cu / *.cuh)
& "$env:ComSpec" /c "tools\build_gpu_enc.bat"

# Zig build (always after any .zig change OR after PTX regen)
zig build -Doptimize=ReleaseFast -Dgpu=true

# Roundtrip SHA verify (sequential — do NOT run in parallel)
# Baseline SHAs:
#   enwik8:  2B49720EC4D78C3C9FABAEE6E4179A5E997302B3A70029F30F2D582218C024A8
#   silesia: 86817A72FC4F404A0330247F69C1278DE8AD058AC1401A9CB1AE174FF5356C74

# Default CLI uses the host scanner. To exercise the GPU scan path
# (where the C2 critical bug lived), set SLZ_GPU_SCAN=1 on decode:
SLZ_GPU_SCAN=1 zig-out/bin/streamlz.exe -d -t 1 c:/tmp/t_l5.slz -o c:/tmp/t_l5.bin

# Bench (sequential ONE AT A TIME, NEVER in parallel tool calls)
zig-out/bin/streamlz.exe -db -t 1 -r 30 c:/tmp/t_l1.slz
# Baseline numbers (RTX 4060 Ti, sm_89, post-3db0ffb):
#   L1: GPU kernel best 2.92 ms
#   L3: GPU kernel best 6.57 ms
#   L5: GPU kernel best 6.26 ms
```

## Verification protocol per batch

After every batch:
1. Rebuild relevant PTX. Cite REG/STACK in the commit message.
2. `zig build -Doptimize=ReleaseFast -Dgpu=true` clean.
3. Roundtrip enwik8 L1/L3/L5 + silesia L3/L5 (sequential).
4. If a fix could affect the GPU scan path: also `SLZ_GPU_SCAN=1` roundtrip.
5. Decode bench (`-db -t 1 -r 30`) on L1 + L3 + L5, compare to baseline within noise.
6. Commit with descriptive message + verification numbers. Do not push.

## Already-deferred items (NOT in this list)

Tracked in user memory `project_gpu_deferred_cleanup.md`:
- `scanBlock` structural split (326 LOC greedy parser, highest perf risk)
- `findMatchChain` structural split (~190 LOC chain parser, dense state)
- `int lane` vs `uint32_t lane` standardization (~15 sites, perf-neutral)
- `DevBuf` struct + `kernelParams` comptime helper (ergonomic only)
- `gpuEncodeLiteralsHuffImpl` / `gpuEncodeTokensHuffImpl` /
  `gpuEncodeOff16HuffImpl` three-near-clones dedup

---

# The work — K1 through K6

Each item carries:
- `[risk]` low / med / high (regression risk)
- `[effort]` xs (one-line) / s (under 30 min) / m (under 2 hr) / l (multi-hour)
- `[verify]` smallest check that catches a regression

Suggested execution order is K1 → K2 → K3 → K4 → K5 → K6.

---

## K1 — Real correctness / dead code (CRITICAL + dead-code HIGH)

### C1. `assemble_kernel.cu:103-104` — bare `12` / `0xFFF` for off32 packed header decode

[risk: low] [effort: xs] [verify: PTX rebuild, enwik8 L3 roundtrip]

The encoder writes the packed header at `lz_kernel.cu:219` using
`OFF32_COUNT_FIELD_BITS` (= 12); the decoder side parses it with bare
literals. Drift hazard. Fix:

```cpp
// in assemble_kernel.cu::parseRaw
uint32_t c1 = (packed >> OFF32_COUNT_FIELD_BITS) & OFF32_COUNT_PACK_MAX;
uint32_t c2 = packed & OFF32_COUNT_PACK_MAX;
```

(`OFF32_COUNT_PACK_MAX == 0xFFF` is already defined in
`common/gpu_wire_format.cuh`.)

### C2. `huffman_kernel.cu:170-178` — UB shift in Kraft sum before height-limit clamp

[risk: med — touches the Huff encode build kernel] [effort: s]
[verify: PTX REG/STACK on slzHuffBuildTablesKernel; enwik8 L3 roundtrip]

```cpp
// Currently:
for (int s = 0; s < HUFF_ALPHABET; s++)
    if (code_lengths[s] > 0)
        sum += (1u << (KRAFT_PRECISION_BITS - code_lengths[s]));
// height-limit clamp at :173-178 runs AFTER this loop.
```

`code_lengths[s]` is the raw tree depth before clamping; on pathological
alphabets it can exceed `KRAFT_PRECISION_BITS` (30), producing a negative
shift count which wraps to a huge positive value — UB shift.

Fix: either clamp `code_lengths[s]` to `max_len` before the initial sum
loop, or compute the sum with `code_lengths[s] > max_len ? (1u << 0) : ...`.

### C3. `gpu_huffman.cuh:75-78, 83` — silent clamp + loop bound off-by-one

[risk: low] [effort: s] [verify: PTX rebuild; roundtrip]

Two issues in `buildCanonicalCodes`:

1. `:75-78`: the `if (L > 0 && L <= HUFF_MAX_CODE_LEN)` guard silently
   drops symbols with `L > 11` from the histogram, but the second loop
   at `:88-91` still does `codes[s] = next_code[L]++` and reads
   `next_code[12]`. Half-defense. Either trust the precondition and
   remove the guard, or add a real check that returns/zeroes.
2. `:83`: loop bound `L <= HUFF_MAX_CODE_LEN + 1` writes
   `next_code[HUFF_MAX_CODE_LEN + 1]` which is never read. Comment
   justifies "covering every possible length" but the actual reads at
   `:88-91` only go up to `HUFF_MAX_CODE_LEN`. Drop the extra iteration
   OR document the real reason (RFC-1951 algorithm presentation).

### C4. `lz_chain_parser.cuh:184-198` — dead `else if` branch (impossible condition)

[risk: low] [effort: xs] [verify: PTX REG/STACK on slzLzEncodeKernel]

```cpp
if (candidate_offset <= NEAR_OFFSET_MAX) { ... }
else if (candidate_offset <= pos && candidate_offset < LZ_BLOCK_SIZE) { ... }
```

`NEAR_OFFSET_MAX = 0xFFFF`, `LZ_BLOCK_SIZE = 0x10000`. The only integer
satisfying `cand > 0xFFFF AND cand < 0x10000` is none — the branch is
unreachable. Either delete it or fix the bound (consult CPU oracle —
likely `< LZ_BLOCK_SIZE * 2` was intended).

### C5. `lz_chain_parser.cuh:413-414` — dead ternary inside outer guard

[risk: none] [effort: xs] [verify: PTX REG/STACK on slzLzEncodeKernel]

```cpp
if (match_end > pos + 1 && match_end + 8 <= src_size) {
    uint32_t insert_end = (match_end + 8 <= src_size) ? match_end : src_size - 8;
```

The outer `if` already asserts `match_end + 8 <= src_size`, so the
ternary is always `match_end`. Drop the ternary.

### C6. `scan_parse_kernel.cuh:162` — dead `if (chunk_len < 3) break;`

[risk: none] [effort: xs] [verify: PTX rebuild]

The enclosing `while` at `:153` already requires `sub_pos + 3 <= chunk_len`,
so on iter 0 `sub_pos = 0` implies `chunk_len >= 3`. `chunk_len` is
unchanged across iters. Misleading — delete.

### C7. `encode_lz.zig:71-79` — `d_input_override` silent fallback uses possibly-stub host ptr

[risk: low — only triggers if D2D async unavailable] [effort: xs]
[verify: zig build; existing roundtrip on default CLI path]

When `self.d_input_override != 0` but `cuMemcpyDtoDAsync_fn` is null, the
code does `h2d_fn(d_input, @ptrCast(input.ptr), input.len)`. But the
contract of `d_input_override` (per `EncodeContext` field doc lines
76-81) is "caller's data lives on the GPU; the host `input` slice may
not even be valid" — callers pass a sentinel like `0x10`. The H2D will
crash or silently encode garbage. Fix: `return false` instead of the
H2D fallback when D2D async is unavailable.

```zig
if (self.d_input_override != 0) {
    const d2d = ffi.cuMemcpyDtoDAsync_fn orelse return false;
    if (d2d(d_input, self.d_input_override, input.len, 0) != ffi.CUDA_SUCCESS) return false;
} else {
    if (h2d_fn(d_input, @ptrCast(input.ptr), input.len) != ffi.CUDA_SUCCESS) return false;
}
```

### C8. `encode_huff.zig:140-142, 291, 383` — dead `errdefer`

[risk: none] [effort: xs] [verify: zig build clean]

```zig
var lo_offsets = allocator.alloc(...) catch return false;
errdefer allocator.free(lo_offsets);
```

The function uses `bool` returns and `catch { ... return false; }`
blocks for every error path — no `try` after these lines and no `!T`
return type. The errdefer never fires. The 7 manual
`catch { allocator.free(hi_offsets); allocator.free(lo_offsets); return false; }`
blocks downstream do the actual cleanup. Either lift the function to
`!bool` and convert to `try`, or delete the errdefer lines. Pick delete
(less churn).

Same pattern at `encode_huff.zig:291` (lit) and `:383` (tok).

### C9. `lz_decode_general.cuh:170-180` — redundant `__shfl_sync` calls in hot loop

[risk: low — measure REG/STACK + bench] [effort: s]
[verify: PTX REG/STACK on slzLzDecodeKernel; L3+ decode bench]

The per-block barrier does:
```cpp
dst_pos = __shfl_sync(FULL_WARP_MASK, dst_pos, 0);  // :170
lit_pos = __shfl_sync(FULL_WARP_MASK, lit_pos, 0);  // :171
```

Both `dst_pos` and `lit_pos` are mutated identically on every lane by
the lit_len/match_len additions (which themselves came from a shfl of
lane 0). So lanes are already coherent — the shuffle is a no-op
barrier. Same pair at `:179, :180`.

Audit each: `cmd_pos` at `:135` IS needed (mutated on lane 0 only at
`:68`). `lit_pos` at `:136` is redundant. Document the policy or
remove the redundant ones.

### C10. `gather_raw_off16_kernel.cuh:33` — `src_offset + size` can overflow on corrupt desc

[risk: low — defensive] [effort: xs] [verify: roundtrip]

```cpp
if (d.size == 0 || d.src_offset + d.size > comp_len) return;
```

Use `d.size > comp_len - d.src_offset` after first checking
`d.src_offset > comp_len`.

---

## K2 — Byte-IO sweep completion

The B-series in the prior arc added `readU16LE`, `readU32LE`, `readU32BE`,
`readBE24`, `writeBE24`, `readLE24`, `writeLE24`, `storeU16LE`. The
agents found these survivors.

### K2.1. Add `writeU32LE` to `common/gpu_byteio.cuh`

[risk: low] [effort: xs] [verify: PTX rebuild]

Pattern: `__device__ __forceinline__ void writeU32LE(uint8_t* dst, uint32_t v) { memcpy(dst, &v, 4); }`

Then use at `assemble_kernel.cu:397-401`:
```cpp
// Currently:
d_output[dst_off + 2] = (uint8_t)(hdr_u32 & 0xFF);
d_output[dst_off + 3] = (uint8_t)((hdr_u32 >> 8) & 0xFF);
d_output[dst_off + 4] = (uint8_t)((hdr_u32 >> 16) & 0xFF);
d_output[dst_off + 5] = (uint8_t)((hdr_u32 >> 24) & 0xFF);
// becomes:
writeU32LE(d_output + dst_off + 2, hdr_u32);
```

Also `assemble_kernel.cu:433-434` writes 4 zero bytes per-lane — replace
with single lane-0 u32 store using `writeU32LE`.

### K2.2. Add `readU64LE` to `common/gpu_byteio.cuh`

[risk: low] [effort: xs] [verify: PTX REG/STACK on slzLzEncodeKernel]

Pattern: `memcpy(&v, p, 8)` style.

Use at `lz_greedy_parser.cuh:310-317` to replace the open-coded 8-byte
load. The surrounding guard already proves `rp + 8 <= src_size`. (May
also be usable as the in-bounds path of `read8safe`.)

### K2.3. `lz_token_emit.cuh:77-79, 82-84` — open-coded LE24 store

[risk: low] [effort: xs] [verify: PTX REG/STACK on slzLzEncodeKernel]

Replace with `writeLE24(off32_buf + off32_pos, truncated); off32_pos += 3;`
(both the short and long branches).

### K2.4. `assemble_kernel.cu:343-348` — open-coded BE24 sub-chunk header write

[risk: low] [effort: xs] [verify: PTX REG/STACK on slzAssembleWriteKernel]

```cpp
// Currently:
hdr[0] = (uint8_t)((sc_hdr >> 16) & 0xFF);
hdr[1] = (uint8_t)((sc_hdr >>  8) & 0xFF);
hdr[2] = (uint8_t)( sc_hdr        & 0xFF);
// becomes:
writeBE24(hdr, sc_hdr);
```

### K2.5. `scan_host.zig:85, 174, 181` — magic 131072 / 65536

[risk: low — verify roundtrip on host scan path] [effort: xs]

```zig
// :85
const sub_dst_off: usize = @as(usize, sub_idx) * @as(usize, @intCast(d.ENTROPY_SCRATCH_SLOT_BYTES));
// :174, :181
... + @as(usize, d.OFF16_HILO_SPLIT_OFFSET);
```

The .cuh has `static_assert(ENTROPY_SCRATCH_SLOT_BYTES == 131072)`; the
Zig scanner having a separate literal defeats it.

### K2.6. `scan_host.zig:71, 231, 269, 298, 303` — open-coded BE24 read

[risk: low] [effort: s]

Add Zig helper in `decode/scan_host.zig` (private) or in
`decode/descriptors.zig` (shared):

```zig
fn readBE24(p: [*]const u8) u32 {
    return (@as(u32, p[0]) << 16) | (@as(u32, p[1]) << 8) | @as(u32, p[2]);
}
```

Apply at all 5 sites.

### K2.7. `scan_host.zig:101, 115, 144, 167` — magic `>> 4) & 0x7` chunk-type extraction

[risk: low] [effort: xs]

Add Zig mirrors in `decode/descriptors.zig`:
```zig
pub const CHUNK_TYPE_SHIFT: u8 = 4;
pub const CHUNK_TYPE_MASK: u8 = 0x7;
```

Apply at all 4 sites.

---

## K3 — Wire-format consolidation finishing

The C-series in the prior arc consolidated most magic numbers, but
`scan_parse_kernel.cuh` and `walk_frame_kernel.cuh` were missed in
several places. Constants live in `common/gpu_wire_format.cuh` (encode/
decode-shared) and `decode/slz_wire_format.cuh` (decoder-private).

### K3.1. `scan_parse_kernel.cuh:153, 166, 172` — magic `3` for sub-chunk header

[risk: low] [effort: xs] [verify: PTX REG/STACK on slzScanParseKernel]

Use `SUBCHUNK_HDR_BYTES` (already defined in `common/gpu_wire_format.cuh:40`).

### K3.2. `scan_parse_kernel.cuh:171` — magic `8u` for initial-copy

[risk: low] [effort: xs]

Use `INITIAL_LITERAL_COPY_BYTES` (already in `common/gpu_wire_format.cuh:26`).

### K3.3. `scan_parse_kernel.cuh:177, 184, 201, 218` — magic `>> 4) & 0x7`

[risk: low] [effort: xs]

Use `CHUNK_TYPE_SHIFT` / `CHUNK_TYPE_MASK` (already in
`common/gpu_wire_format.cuh:32-33`). Note that `scanSkipStreamHeader` in
the same file at `:85` already uses these — internal-to-file
inconsistency.

### K3.4. `scan_parse_kernel.cuh:196` — open-coded u16 LE read

[risk: low] [effort: xs]

Use `readU16LE(chunk_src + pos)`.

### K3.5. `scan_parse_kernel.cuh:98, 101` — magic `7` / `4` for paired headers

[risk: low] [effort: xs]

Use `PAIRED_SECONDARY_HEADER_BYTES` (= 7) and
`PAIRED_PRIMARY_HEADER_BYTES` (= 4) from `slz_wire_format.cuh:81-82`.

### K3.6. `scan_parse_kernel.cuh:129` — `cap_safe = sub_chunk_cap ? sub_chunk_cap : 65536u`

[risk: low] [effort: xs]

Add `DEFAULT_SUB_CHUNK_CAP = 65536u` to `common/gpu_wire_format.cuh` (or
note that this is `OFF16_HILO_SPLIT_OFFSET` which happens to equal it).
Same magic also at `walk_frame_kernel.cuh:88-93` and
`prefix_sum_chunks_kernel.cuh:27`.

### K3.7. `walk_frame_kernel.cuh:79-80` — magic `0x01` / `0x02` frame flag bits

[risk: low] [effort: s]

Add to `common/gpu_wire_format.cuh`:
```cpp
static constexpr uint8_t SLZ_FRAME_FLAG_CONTENT_SIZE_PRESENT = 0x01;
static constexpr uint8_t SLZ_FRAME_FLAG_DICT_ID_PRESENT      = 0x02;
```

### K3.8. `walk_frame_kernel.cuh:107, 122, 154, 187` — magic `0x0F` / `0x7F` / `0x80` block-header bits

[risk: low] [effort: s]

Add named constants alongside `SLZ_INT_BLOCK_MAGIC` in
`common/gpu_wire_format.cuh`:
```cpp
static constexpr uint8_t SLZ_BLOCK_HDR_MAGIC_MASK    = 0x0F;
static constexpr uint8_t SLZ_BLOCK_HDR_DECODER_MASK  = 0x7F;
static constexpr uint8_t SLZ_BLOCK_HDR_CHECKSUM_FLAG = 0x80;
```

### K3.9. `walk_frame_kernel.cuh:88-93` — magic 262144 / 65536 / 131072

[risk: low] [effort: s]

`262144` = 256KB chunk size. `131072` = 128KB sub-chunk slot
(= `ENTROPY_SCRATCH_SLOT_BYTES`, already in slz_wire_format.cuh).
Add `SLZ_CHUNK_SIZE_BYTES = 262144u` to `common/gpu_wire_format.cuh`
and reference where applicable.

### K3.10. `lz_decode_kernels.cuh:60, 69, 182, 190` — `& 1` / `& 2` chunk-flag magic

[risk: low] [effort: xs]

Add to `common/gpu_wire_format.cuh`:
```cpp
static constexpr uint32_t CHUNK_FLAG_UNCOMPRESSED = 0x1u;
static constexpr uint32_t CHUNK_FLAG_MEMSET       = 0x2u;
```

Apply at the 4 sites + `scan_parse_kernel.cuh:132` (`ch.flags != 0`).

### K3.11. `encode_lz.zig:53` — magic `65536` for NEXT_HASH

[risk: low] [effort: xs]

`65536` is `NEXT_HASH_SIZE` from `lz_format.cuh:21`. Either define
`NEXT_HASH_ENTRIES = 65536` in a shared place or add a Zig-side
comment cross-referencing the .cuh constant.

### K3.12. `encode_assemble.zig:208, 213` — magic `8` and `6`

[risk: low] [effort: xs]

`8` = SC-tail-per-chunk bytes. Same value appears at `encode_huff.zig:147,
296, 388` and `fast_framed.zig:1365`. Define
`SC_TAIL_PER_CHUNK_BYTES = 8` (somewhere accessible to all five sites).

`6` (at `:208`) = chunk-internal-header size. Comment its semantic if not
defining a constant.

### K3.13. `assemble_kernel.cu:32` — `RAW_CHUNK_HDR_BYTES` vs `SUBCHUNK_HDR_BYTES` naming collision

[risk: low] [effort: xs]

Both equal 3 but semantically distinct (raw entropy chunk header vs
sub-chunk header). Either rename to `RAW_ENTROPY_CHUNK_HDR_BYTES` or
add a two-line distinguishing comment.

### K3.14. `assemble_kernel.cu:233-235` — magic `+ 2`

[risk: low] [effort: xs]

Express as `HUFF_CHUNK_HDR_BYTES - RAW_CHUNK_HDR_BYTES` (both already
defined in the file).

### K3.15. `lz_format.cuh:29-30` — `LARGE_OFFSET_THRESHOLD` and `OFF32_LARGE_TAG` same value

[risk: low] [effort: xs]

Both = `0xC00000`. Define one in terms of the other:
```cpp
static constexpr uint32_t OFF32_LARGE_TAG  = 0xC00000u;
static constexpr uint32_t LARGE_OFFSET_THRESHOLD = OFF32_LARGE_TAG;  // by construction
```
Add a comment explaining the encoder's truncate-then-OR.

### K3.16. `slz_wire_format.cuh` — move `EXT_LENGTH_THRESHOLD` to common

[risk: low] [effort: xs]

`EXT_LENGTH_THRESHOLD = 251` at `:50` is the same constant the encoder
emits against (in `lz_token_emit.cuh`). Move to
`common/gpu_wire_format.cuh` so both sides reference the same name.

### K3.17. `slz_wire_format.cuh` — add `static_assert(OFF16_HILO_SPLIT_OFFSET == LZ_BLOCK_SIZE)`

[risk: none] [effort: xs]

Both equal `0x10000`. The actual invariant is "one block's worth of hi
bytes followed by lo bytes." Make the relationship compile-time enforced.

### K3.18. `lz_kernel.cu:108` (encode side) — magic `recent_offset = -8`

[risk: low] [effort: xs]

Use `INITIAL_RECENT_OFFSET` (already in `common/gpu_wire_format.cuh:27`).

---

## K4 — Dead facade + lifecycle

### K4.1. Delete 5 dead encode facade wrappers + `g_default`

[risk: low — confirmed zero callers] [effort: s] [verify: zig build]

In `src/gpu/encode/driver.zig:55-107` — delete:
- `gpuCompress` (5 LOC)
- `gpuEncodeHuff` (8 LOC)
- `gpuEncodeLiteralsHuff` (8 LOC)
- `gpuEncodeTokensHuff` (8 LOC)
- `gpuEncodeOff16Huff` (8 LOC)
- `pub var g_default: EncodeContext = .{};` (line 47)

Also update the file-level docstring lines 5-8 + 46-47 that advertise
these as part of the API surface.

Verify via `Grep` first to confirm zero callers outside the file.

### K4.2. Add `EncodeContext.deinit()`

[risk: low — new method, no callers required to update] [effort: m]

Walk every `d_*_persist: CUdeviceptr = 0` field in `encode_context.zig`
and the `*_size: usize = 0` companion (~21 pairs); add a `deinit(self:
*EncodeContext)` that frees each via `ffi.cuMemFree_fn` when non-zero,
plus the `assembled_data` / `assembled_offsets` / `assembled_sizes`
host slices (need allocator parameter), plus the `huff_*` host slices
(also need allocator).

Signature: `pub fn deinit(self: *EncodeContext, allocator: std.mem.Allocator) void`.

Does not need to be called from anywhere yet — the next user-library
extraction will use it. Add a smoke test if you want to validate.

### K4.3. Add `DecodeContext.deinit()`

[risk: low] [effort: m]

Same pattern. ~21 device-buffer pairs in `decode_context.zig` plus the
`h_pinned_output` (which needs `cuMemFreeHost`) plus the
`pipeline_streams`.

### K4.4. `gpuFrameAssembleImpl` 16-arg signature

[risk: med — touches the hot path] [effort: m]

Factor into named structs:
```zig
const FramePreamble = struct {
    prefix_bytes: []const u8,
    internal_hdr0: u8,
    internal_hdr1: u8,
};
const ChunkLayout = struct {
    n_chunks: u32,
    eff_chunk_size: u32,
    src_len: u32,
    per_chunk_asm_off: []const u32,
    per_chunk_asm_size: []const u32,
};
pub fn gpuFrameAssembleImpl(
    self: *EncodeContext,
    allocator: std.mem.Allocator,
    layout: ChunkLayout,
    preamble: FramePreamble,
    d_input_dev: u64,
    d_output: u64,
) ?u32 { ... }
```

Update the single caller in `src/encode/fast_framed.zig:1385`.

### K4.5. Add "no-clobber" assert at huff_off16 assignment site

[risk: none] [effort: xs]

`encode_huff.zig:265-269` — before the assignment, assert
`self.huff_off16hi_data == null and self.huff_off16lo_data == null`.
Catches the double-encode-without-free bug at write time rather than
free time.

### K4.6. `beginKernelTiming` leak on launch failure

[risk: low] [effort: s]

`encode_lz.zig:113-117`, `encode_huff.zig:89-92, 106-109`,
`encode_assemble.zig:105-108, 131-134, 256-263` — `beginKernelTiming`
returns a handle that must be paired with `endKernelTiming`, but the
launch-failure path returns false without calling end. Fix:

```zig
const t_lz = gpu_decode.beginKernelTiming(...);
defer gpu_decode.endKernelTiming(t_lz, 0);
if (launch_fn(...) != ffi.CUDA_SUCCESS) return false;
```

OR document that profiling leaks the begin event on launch failure (with
caveat in `decode_context.zig` where `finalizeProfiling` is defined).

---

## K5 — Architectural (harder; do last)

### K5.1. `error.BadMode` subtyping

[risk: med — touches GpuError + every catch] [effort: l]

In `decode/descriptors.zig:155-160`, expand `GpuError` to:
```zig
pub const GpuError = error{
    CudaNotAvailable,     // dlopen/getProc/cuInit failed
    OutOfDeviceMemory,    // ensureDeviceBuf / cuMemAlloc rc != 0
    CudaLaunchFailed,     // cuLaunchKernel rc != 0
    CudaSyncFailed,       // ctx/stream sync rc != 0
    CudaCopyFailed,       // memcpy[HtoD|DtoH|DtoDAsync] rc != 0
    KernelMissing,        // required kernel slot == 0
    BadMode,              // ABI-compat alias; kept for slzDecompress C wrapper
};
```

Update `cudaCall` to accept a comptime tag:
```zig
pub fn cudaCall(rc: cuda.CUresult, comptime kind: ErrorKind) GpuError!void { ... }
```

Then thread the correct tag through every call site. Heavy mechanical
change.

### K5.2. `fullGpuLaunchImpl` extraction

[risk: med — touches the hottest decode dispatch] [effort: l]

`decode_dispatch.zig:308-734` — 423 LOC, 11 params. Extract:
- `runHuffPredecode(self, ml, ...)` — the huff_build + huff_decode block
  (lines ~508-560)
- `runLzPipeline(self, ml, ...)` — the for-loop over pipeline groups
  (lines ~584-668)
- `finalizeOutput(self, ...)` — the d_output_target D2D-copy logic at
  the tail

Add `DecodeRequest` struct for the 6 invariant params (chunk_descs,
compressed_block, dst_full, dst_start_off, decompressed_size,
sub_chunk_cap).

### K5.3. Move stack buffers to `DecodeContext`

[risk: low — pure storage relocation] [effort: s]

`decode_dispatch.zig:91` `merged_huff: [d.MAX_HUFF_DESCS_PER_STREAM * 4]HuffDecChunkDesc`
(~320 KiB) and `:378` `first_subchunk_idx_buf: [d.walk_max_chunks]u32`
(64 KiB) — combined ~384 KiB stack per call. Move both to
`DecodeContext` fields with `_buf` suffix.

### K5.4. Replace ctx-wide `sync_fn()` with `stream_sync_fn(heavy_stream)`

[risk: med — verify async-decode callers still see correct ordering]
[effort: s]

`decode_dispatch.zig:364, 407, 502` — three ctx-wide syncs that stall
every CUDA stream in the process. Async-decode callers
(`work_stream != 0`) are trying to avoid exactly this. Document the
ordering contract at each site or migrate to per-stream sync.

### K5.5. Template `decodeSubChunkGeneral` on `OFF16_SPLIT`

[risk: low — see J4 experiment first; PTX REG/STACK might shift]
[effort: m]

`lz_decode_general.cuh:18` — currently takes `off16_split` as a runtime
`uint32_t`; sibling `decodeSubChunkRawMode` templates on it. Adding
`template <bool OFF16_SPLIT>` would specialize the general decoder
identically. Verify PTX numbers and decide.

### K5.6. `decodeSubChunkGeneral` 17-arg signature → struct

[risk: med — register-pressure consideration; J4 showed __noinline__ is
net-negative here] [effort: m]

Group the args into:
- `DecodeStreams { cmd, cmd_size, lit, lit_size, off16_*, off32_*, length_* }`
- `DecodeOutput { dst, dst_size, dst_offset, initial_copy }`
- Plus `block2_cmd_offset` and `mode` as scalar params

### K5.7. `parseRawStreamSize` / `parseEntropyHeader` bounds checks

[risk: med — adds parameters across many call sites] [effort: l]

`slz_wire_format.cuh:237-241, 249-255` have zero bounds checks. The doc
says caller is responsible, but every call site (`lz_dispatch.cuh:48,
62`, `lz_header_parse.cuh:47, 79, 116, 136`) is `if (lane == 0)` inside
`parseAndDecodeSubChunk*` with no `src + N <= src_end` guard. Options:
1. Propagate `src_end` cursor; bounds-check each call (scan kernel
   demonstrably does this).
2. Promote the doc to a `// SAFETY:` comment at every call site listing
   the upstream guarantee.

Pick (2) for time-efficient fix; (1) for textbook correctness.

### K5.8. `cached_qpc_freq` → `std.atomic.Value(i64)`

[risk: none — idempotent value] [effort: xs]

`cuda_api.zig:32`. Replace `var cached_qpc_freq: i64 = 0;` with
`var cached_qpc_freq: std.atomic.Value(i64) = .init(0);` and use
`.monotonic` load/store.

### K5.9. `gatherRawOff16` fallback contract — commit one direction

[risk: low] [effort: s]

`decode_dispatch.zig:170-184`. Pick:
1. `try cudaCall(launch_fn(...))` and delete the D2D/H2D fallback
   below (lines 186-199). Trust the self-gate.
2. Keep the fallback but document why a launch failure here would NOT
   also break the D2D copy below — i.e., justify the failure-mode
   disjointness.

J2 commented but didn't pick. Pick now.

### K5.10. `module_loader.init` — tri-state enum

[risk: low — re-entry doesn't happen today] [effort: s]

Replace `pub var initialized: bool = false;` with:
```zig
const InitState = enum { uninit, in_progress, ready, failed };
pub var init_state: InitState = .uninit;
```

Use `.in_progress` while running to make re-entry detectable.

### K5.11. `ensurePipelineStreams` — return `GpuError!void`

[risk: low] [effort: s]

`module_loader.zig:147-153` currently returns void; partial stream
creation leaks. Return `GpuError!void` and free what was created on
failure.

---

## K6 — Nits, docs, naming, comments (everything else)

### Documentation

- **K6.1.** `CLEANUP_TODO.md` — extend DONE table through `a7a9f5f`,
  prune the "HIGH-priority follow-ups" list (5 of 6 items closed by
  J2/J3; only `decodeSubChunkGeneral __noinline__` was investigated
  and documented). Fix em-dash usage in the table. Fix
  `gpuEncodeXxxHuffImpl` → actual function names. [effort: s]

- **K6.2.** `gpu_byteio.cuh:7-12` — rewrite endianness rule comment
  (now 4 conventions: BE24, LE24, U32BE, U32LE/U16LE), or drop the
  enumeration entirely. [effort: xs]

- **K6.3.** `gpu_byteio.cuh:60` — fix `readU32LE` doc (callers now
  include walk_frame, not just LZ match-finder). [effort: xs]

- **K6.4.** `gpu_byteio.cuh:64-74` — `read8safe` doc: "caller
  guarantees pos ≤ src_size". Document or guard. [effort: xs]

- **K6.5.** `gpu_warp.cuh:13-20` — split "geometry" from "participation
  mask" in the layout heading. [effort: xs]

- **K6.6.** `lz_kernel.cu:19-21` (encode) — fix "below" → "above" in
  the file docstring (J1 vintage gaffe). [effort: xs]

- **K6.7.** `lz_format.cuh:11` — fix the under-describing include
  comment ("read8safe" alone — now also `readU32LE`, `readBE24`,
  `writeBE24`, etc.). [effort: xs]

- **K6.8.** `README.md:32` — clarify `scan_host` description
  ("host-bounce path; pure-D2D uses scan_gpu"). [effort: xs]

- **K6.9.** `README.md:39-48` — update layout block for
  `lz_decode_raw.cuh` / `lz_decode_general.cuh` (currently lists only
  `lz_decode_core.cuh`). [effort: xs]

- **K6.10.** `README.md:84` — fix missing verb ("Every CUDA kernel is
  resolved by the Zig drivers via `cuModuleGetFunction`."). [effort: xs]

- **K6.11.** `README.md:106-109` — drop the partial
  `(lz_decode_kernels.cuh, …)` parenthetical (lists only 2 of 9
  .cuh headers). [effort: xs]

- **K6.12.** `README.md:192` — clarify L5 silesia ratio vs encode-time
  inconsistency (ratio is real, encode time is "not representative"
  — explain). [effort: xs]

- **K6.13.** `README.md:225-227` — change "essentially tied" to state
  Zstd narrowly ahead exactly (`6.2 vs 6.26 ms, ~1% gap`). Per
  `feedback_state_wins` memory rule. [effort: xs]

- **K6.14.** README footnote `†` — link to memory entry / issue rather
  than just stating "known pathological slow path". [effort: xs]

- **K6.15.** `ARCHITECTURE.md:39` — explain ceiling division
  `(50 + 31) / 32 = 2`. Half a sentence. [effort: xs]

- **K6.16.** `ARCHITECTURE.md:84, 134` — pick spelled-out-vs-digit
  convention ("fifty-six-byte" vs "L3"). [effort: xs]

- **K6.17.** `ARCHITECTURE.md:198-201` — tighten redundant 50-byte
  literal restatement (math is already at :37-43). [effort: xs]

- **K6.18.** `ARCHITECTURE.md:224-246` — reference the J4 finding
  alongside the `parseSubChunkHeaders` explanation (why raw decoder is
  `__noinline__` but general is not). [effort: xs]

- **K6.19.** `ARCHITECTURE.md:241` — "keep-out attribute" → standard
  "`__noinline__` attribute" (novel terminology). [effort: xs]

- **K6.20.** `ARCHITECTURE.md:459-471` — trim or update constants
  block (lags the actual `common/gpu_wire_format.cuh` contents).
  [effort: s]

- **K6.21.** `ARCHITECTURE.md:494` — add scan kernel to the "runs on
  device" enumeration. [effort: xs]

### Naming consistency

- **K6.22.** `gpu_wire_format.cuh:32-34` — after `RAW_CHUNK_TYPE`
  deletion, `HUFF_CHUNK_TYPE = 4` stands alone with no `ct == 0`
  reference. Either reintroduce `RAW_CHUNK_TYPE = 0` and use at
  `scan_parse_kernel.cuh:86, 202, 219`, OR add a comment "raw is 0,
  implied". [effort: xs]

- **K6.23.** `gpu_wire_format.cuh:84-86` — rename `SLZ_CHUNK_TYPE_MASK`
  to `SLZ_CHUNK_TYPE_MASK_BITS` (it's post-shift, unlike its neighbor
  `SLZ_CHUNK_SIZE_MASK` which is pre-shift). [effort: xs]

- **K6.24.** `gpu_wire_format.cuh:92` — rename `STREAM_HEADER_BYTES`
  → `LZ_SUBSTREAM_COUNT_HDR_BYTES`. [effort: xs]

- **K6.25.** `gpu_wire_format.cuh:95-102` — rewrite ENTROPY_HDR_*
  layout comment with actual bit layout (not "see writeHuffChunkHdr").
  [effort: xs]

- **K6.26.** `gpu_wire_format.cuh:78-89` — types should reflect on-wire
  width (`SLZ_FRAME_MIN_HDR_SIZE` could be `uint8_t` not `uint32_t`).
  [effort: xs]

- **K6.27.** `gpu_huffman.cuh:122-124` — `lutSym1/2` return `uint8_t`
  but `lutTotalLen`/`lutNumSyms` return `int`. Make all four return
  the same width. [effort: xs]

- **K6.28.** `gpu_huffman.cuh:107-109` — distinguish
  `LUT_NUM_SYMS_TAG_SINGLE`/`_TAG_DUAL` from `LUT_MAX_SYMS_PER_STEP`
  (currently the tag values happen to equal the per-step counts).
  [effort: xs]

- **K6.29.** `gpu_huffman.cuh:122` — document `lutSymPair` endianness
  (sym1 in low byte, sym2 in high byte). [effort: xs]

- **K6.30.** All `common/*.cuh` — `static constexpr` → `inline
  constexpr` (C++17 ODR-merge across TUs). [effort: s]

- **K6.31.** `lz_format.cuh:55-57` — Fibonacci-multiplier naming
  symmetry (`HASH_MUL_FIB_K6`, `HASH_MUL_FIB_64`, `HASH_MUL_A` for
  parallel structure). [effort: xs]

- **K6.32.** `descriptors.zig:117` — `walk_max_chunks` (snake_case) →
  `WALK_MAX_CHUNKS` (matches the SCREAMING_SNAKE in surrounding
  module-level constants). [effort: xs]

### Decode CUDA polish

- **K6.33.** `lz_decode_core.cuh:49, 80` — `match_len >=
  MIN_PARALLEL_MATCH_LEN` not `match_len > MIN_PARALLEL_MATCH_LEN - 1`.
  [effort: xs]

- **K6.34.** `lz_decode_raw.cuh:147, 164` — `if constexpr (OFF16_SPLIT)`
  not runtime `if`. [effort: xs]

- **K6.35.** `lz_decode_raw.cuh:84` — add `static_assert(WARP_SIZE ==
  32)` next to `(2u << lane) - 1u`. [effort: xs]

- **K6.36.** `lz_decode_raw.cuh:127, 85` — `31 - __clz(...)` magic;
  introduce `LAST_BIT_SET(x)` helper or `(int)U32_BITS - 1 -
  __clz(...)`. [effort: xs]

- **K6.37.** `lz_decode_general.cuh:46` — overflow check on
  `dst_end_abs` (defensive against corrupt descriptors). [effort: xs]

- **K6.38.** `lz_decode_general.cuh:56` — move `prefetched_token` decl
  into the for-loop (or document hoist reason). [effort: xs]

- **K6.39.** `lz_decode_general.cuh:201` — delete redundant `break`
  (loop bound already exits on last iter). [effort: xs]

- **K6.40.** `lz_decode_general.cuh:163` — introduce `TokenType` enum
  for `token_type != 1` magic. Same for the other token_type literals
  in the function. [effort: s]

- **K6.41.** `lz_decode_general.cuh:177-181, 207-211` — extract
  delta-literal trailing helper (5-line near-identical bodies).
  [effort: s]

- **K6.42.** `lz_dispatch.cuh:24-30` — add `__restrict__` to
  `parseAndDecodeSubChunkRaw` pointer args (sibling
  `parseAndDecodeSubChunk` has them; `decodeSubChunkRawMode` does too).
  [effort: xs]

- **K6.43.** `lz_dispatch.cuh:163` — add `__restrict__` to
  `parseAndDecodeSubChunk`'s `entropy_*_scratch` args. [effort: xs]

- **K6.44.** `lz_dispatch.cuh:78` — comment on `__shfl_sync` of
  `block2_cmd_offset` ("lane 0 may have updated; broadcast"). [effort: xs]

- **K6.45.** `walk_frame_kernel.cuh:35-39` — replace `walkReadF32LE`
  union type-pun with `memcpy`. [effort: xs]

- **K6.46.** `huffman_kernel.cu:170-174` — `int` → `uint32_t` for
  `bit_count` / `pending`. [effort: xs]

- **K6.47.** `huffman_kernel.cu:68` — drop redundant mask
  `(1u << len) - 1u` (codes are guaranteed to fit in `code_lengths[s]`
  bits). [effort: xs]

- **K6.48.** `huffman_kernel.cu:67` — add `assert(len > 0)` (CUDA
  device-side assert works) for the "no zero-length code" contract.
  [effort: xs]

- **K6.49.** `huffman_kernel.cu:211-220` — lift alignment check out
  of inner loop (once established, 4-byte writes maintain it). [effort: s]

- **K6.50.** `huffman_kernel.cu` `code_lengths` UB — see C2 (in K1).

- **K6.51.** `compact_descs_kernels.cuh / walk_frame_kernel.cuh /
  prefix_sum_chunks_kernel.cuh / merge_huff_descs_kernel.cuh` —
  extract `SLZ_GUARD_SINGLE_THREAD()` macro (5 sites use
  `if (blockIdx.x != 0 || threadIdx.x != 0) return;`). [effort: xs]

- **K6.52.** `lz_header_parse.cuh:300, lz_dispatch.cuh:38` — `(int)
  INITIAL_LITERAL_COPY_BYTES` cast; prefer `(uint32_t)lane <
  INITIAL_LITERAL_COPY_BYTES`. [effort: xs]

- **K6.53.** `lz_header_parse.cuh:50, 113` — F7 not fully applied:
  still mentions chunk_type 1 and 6 by old "tANS" names. Rewrite to
  match the F7-applied wording elsewhere. [effort: xs]

- **K6.54.** `lz_kernel.cu:108` (encode) — uses literal `-8` for
  `recent_offset`. Use `INITIAL_RECENT_OFFSET`. (Already in K3.18.)

### Encode CUDA polish

- **K6.55.** `lz_token_emit.cuh:60-66` — comment the negative-tag
  underflow idiom (`low2 - LENGTH_EXT_TAG_BIAS` wraps to 0xFFFFFFFC..).
  [effort: xs]

- **K6.56.** `lz_token_emit.cuh:184` — `> 0` → `>= 0` (or document
  equivalence). [effort: xs]

- **K6.57.** `lz_kernel.cu:88, 92-97` (encode) — sub-stream sizing
  divisors `/4`, `/2` should be named (`MAX_TOKEN_BYTES_PER_SRC_DIV`
  etc.) or have an extended comment explaining each region's worst
  case. [effort: s]

- **K6.58.** `lz_chain_parser.cuh:117` — comment the 26-bit
  truncation in `pos << LONG_HASH_TAG_BITS` (top 6 bits zero by
  chunk-size construction). [effort: xs]

- **K6.59.** `lz_chain_parser.cuh:127-131, 386-389` — cite the CPU
  oracle function names in the copy-paste comments. [effort: xs]

- **K6.60.** `lz_chain_parser.cuh:122` — comment `__ffs(xor_val) - 1`
  / 8 == trailing-zero-bits / 8 == matching low bytes. [effort: xs]

- **K6.61.** `assemble_kernel.cu:174-187` — comment the `__syncwarp()`
  between lane-0 header write and `warpCopy` ("publish lane-0 header
  write before cooperative body copy"). [effort: xs]

### Decode Zig polish

- **K6.62.** `decode_dispatch.zig:99` — rename `var dd = s;` (shadows
  module alias `dd = decode_dispatch`). [effort: xs]

- **K6.63.** `decode_dispatch.zig:474` — cache
  `std.c.getenv("SLZ_HUFF_DBG")` result (called on every decode).
  [effort: xs]

- **K6.64.** `decode_dispatch.zig:606-607` — comment the `/2` magic
  in `lz_grid_x` (each block handles 2 groups). [effort: xs]

- **K6.65.** `decode_dispatch.zig:686` — log all sync errors OR none
  (currently inconsistent). [effort: xs]

- **K6.66.** `decode_context.zig:271` — `.{0}` not `.{0} ** 1`
  (awkward when array length is 1). [effort: xs]

- **K6.67.** `decode_dispatch.zig:367, 388` — collapse the
  `d_first_subchunk_idx = persistent; ... = 0;` two-step into a single
  conditional assignment. [effort: xs]

- **K6.68.** `decode_dispatch.zig:618-619, 637-638` — extract
  `setupLzCommonParams(self, chunk_start, chunks_per_group,
  sub_chunk_cap)` helper (the 6 vars `p_comp` / `p_descs_dev` / `p_dst`
  / `p_cpg` / `p_total` / `p_sc_cap` are identical between the raw and
  general branches). [effort: s]

- **K6.69.** `scan_gpu.zig:60, 96` — rename `_t` → `t_walk` / `t_prefix`
  (matches the sibling `t_xxx` pattern). [effort: xs]

- **K6.70.** `scan_gpu.zig:117-135` — drop vestigial
  `first_subchunk_idx` parameter (one caller, only `.len` consulted).
  [effort: xs]

- **K6.71.** `scan_gpu.zig:155` — fix comment grammar: "Zero so
  sub-chunk slots that no thread reaches keep valid=0." [effort: xs]

- **K6.72.** `scan_gpu.zig:153, 220` — convention drift: scan_gpu uses
  raw `if (rc != CUDA_SUCCESS) return null;` while decode_dispatch uses
  `try cudaCall(...)`. Pick one for decode/. [effort: s]

- **K6.73.** `descriptors.zig:93` — `ScanResult.num_raw_off16` missing
  `= 0` default (siblings have it). [effort: xs]

- **K6.74.** `descriptors.zig:106` — move `Type0Info` to `scan_host.zig`
  (sole consumer). [effort: xs]

- **K6.75.** `descriptors.zig:155-160` — `GpuError` doc references
  "decoder's DecodeError" without a path. Add a `// see
  src/decode/...` pointer. [effort: xs]

- **K6.76.** `decode_context.zig:25` — silent `cuMemFree` on grow:
  log via `std.debug.print` on non-success, or document why it's
  intentionally swallowed. [effort: xs]

- **K6.77.** `decode_context.zig:117` — `endKernelTiming` silent
  `cuEventRecord` failure can leave finalizeProfiling blocked forever.
  Document the relationship, or detect and skip the corresponding
  pending entry. [effort: s]

- **K6.78.** `decode_context.zig:147-152` — `finalizeProfiling`
  silently drops 4 CUDA results per pending event. Add a one-line
  "errors swallowed by design — profiling is opt-in" comment. [effort: xs]

- **K6.79.** `decode_context.zig:98, 102, 108` — rename `dd` (the
  destroy_fn local) → `destroy` or `destroy_fn` (avoids shadow with
  the dd module alias). [effort: xs]

- **K6.80.** `decode_context.zig:181` — move `h_pinned_output` next to
  `d_output` for grouping. [effort: xs]

- **K6.81.** `decode_context.zig:255-256` — promote per-field doc
  comments to `///` so each appears in field doctips. [effort: xs]

- **K6.82.** `descriptors.zig:38-39` — express
  `MAX_HUFF_DESCS_PER_STREAM = walk_max_chunks / MAX_SUB_CHUNKS_PER_CHUNK`
  to make the relationship visible. [effort: xs]

- **K6.83.** `decode_dispatch.zig:251-285` — `dumpScanIfRequested` uses
  inline `@cImport({@cInclude("stdio.h");})`. Hoist to module scope or
  switch to `std.fs.cwd().createFile`. Path `c:/tmp/scan_dump.bin` is
  Windows-only. [effort: s]

- **K6.84.** `module_loader.zig:97, 122` — extract
  `nullTerminatedPtx(name)` helper (repeated `@embedFile + "\x00"`).
  [effort: xs]

- **K6.85.** `module_loader.zig:128-132` — improve SLZ_E2E_TIMER print
  to distinguish "create context" vs "piggyback" branches. [effort: xs]

- **K6.86.** `cuda_api.zig:50` — `pub var ctx: usize = 0;` → `?usize`
  (self-documents the "no context yet" sentinel). [effort: xs]

- **K6.87.** `cuda_api.zig:111` — comment "why NUM_PIPELINE_STREAMS = 1"
  (was higher in earlier history; for-loops still parameterized).
  [effort: xs]

### Encode Zig polish

- **K6.88.** `encode/levels.zig:29-35` — drop the unused `level`
  parameter from `useGlobalHash` (or document why preserved for future
  per-level differentiation). [effort: xs]

- **K6.89.** `encode/cuda_ffi.zig:30` — rename `lib` → `_lib` or stash
  in an internal struct (currently `pub var` exposes the LoadLibrary
  handle). [effort: xs]

- **K6.90.** `encode/cuda_ffi.zig:32-41` — alias `CUmodule`/
  `CUfunction` as typed `usize` so module_loader can use the typed
  names instead of bare `usize`. [effort: xs]

- **K6.91.** `encode_context.zig:166` — reword "set both to null" to
  list all 6 slots that must be nulled (sizes/data/offsets × hi/lo).
  [effort: xs]

- **K6.92.** `encode_huff.zig:78` — add `// Huffman source = LZ output
  (raw streams written by gpuCompressImpl)` comment. [effort: xs]

- **K6.93.** `encode_huff.zig:269` — move the "SAME pointer" inline
  comment to the field doc in `encode_context.zig` (already has the
  OWNERSHIP RULE); drop the inline shouty comment. [effort: xs]

- **K6.94.** `encode_huff.zig:48` vs `:134, 286, 378` — pick one return
  for empty input (`gpuEncodeHuffImpl` returns true; the wrappers
  return false). [effort: xs]

- **K6.95.** `encode_assemble.zig:117` — document why
  `enc_sizes[i] == 0` is unambiguously a kernel parse error (vs an
  "empty sub-chunk" interpretation that a future change might add).
  [effort: xs]

- **K6.96.** `encode_lz.zig:33-35` vs `:81` — pick a convention for
  FFI fn resolution (upfront vs inline). Document. [effort: xs]

- **K6.97.** `encode_context.zig:190` — `_ = free_fn(ptr.*)` ignoring
  CUDA result; add comment ("free failure on known-valid ptr means
  driver is dying; nothing useful we can do"). [effort: xs]

- **K6.98.** `encode_huff.zig:84-87, 100-105` — params arrays
  formatted with manual alignment that won't survive `zig fmt`. Let
  zig fmt reflow or extract a comptime tuple helper. [effort: xs]

- **K6.99.** `driver.zig:43` (encode) — trim "the @import cycle is
  fine" comment ("sub-modules can `@import("driver.zig")` to read the
  singletons; not a cycle"). [effort: xs]

- **K6.100.** `encode/driver.zig:49-52` — clarify `last_kernel_ns`
  field doc: "(decode driver has its own; not interchangeable)".
  [effort: xs]

- **K6.101.** `encode_huff.zig:231, 346, 442` — extract `bytes`
  alloc-or-fallback into a named local before the assignment.
  [effort: xs]

- **K6.102.** `encode_assemble.zig:208-211` — name the `6`
  chunk-internal-header size constant (related to K3.12). [effort: xs]

### Cleanup self-references

- **K6.103.** `CLEANUP_TODO.md` (the OLD one, not this file) — the
  DONE table needs J1-J4 entries (`49a5391`, `166a2db`, `e89c1cd`,
  `a7a9f5f`). The "HIGH-priority follow-ups" list shows 5 of 6 items
  that were closed by J2/J3 — prune to just the open ones. Em-dash
  usage in the table headers. Concrete file/function names instead of
  `gpuEncodeXxxHuffImpl` placeholder. [effort: s — covers K6.1]

---

# Suggested execution order

1. **K1** (real correctness / dead code) — ~45 min, batch into 1-2 commits.
2. **K2** (byte-IO sweep completion) — ~30 min, 1 commit.
3. **K3** (wire-format consolidation finishing) — ~45 min, 1 commit.
4. **K4** (dead facade + lifecycle) — ~90 min, batch into 2-3 commits
   (K4.1 separate; K4.2+K4.3 together; K4.4 separate; K4.5+K4.6 together).
5. **K5** (architectural) — multi-hour, separate commit per item. Each
   item may want its own roundtrip + bench cycle.
6. **K6** (nits) — ~90 min, batch by file/area into ~5-6 commits.

Each batch ends with: rebuild PTX, REG/STACK check, `zig build`,
5-SHA roundtrip (+ `SLZ_GPU_SCAN=1` for items that touch the scan
path), decode bench on L1+L3+L5 (compare to baseline within noise),
commit.

# When this file is itself done

Replace `CLEANUP_TODO.md` (the older DONE log) with an updated DONE
log that includes all K1-K6 commits and any newly-identified
follow-ups. Delete `CLEANUP_TODO2.md` or fold its remaining items
back into the DONE log.
