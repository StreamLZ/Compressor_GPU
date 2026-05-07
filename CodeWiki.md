# StreamLZ Zig — Code Wiki

Internal reference for contributors. For user-facing docs see [README.md](README.md).

---

## Source layout

```
build.zig                         Zig 0.16 build script
build.zig.zon
src/
  main.zig                        CLI dispatcher
  cli.zig                         CLI argument parser + command handlers
  streamlz.zig                    public library API (re-exports)
  capi.zig                        C-ABI exports (slz_compress, slz_decompress, error codes)
  test_runner_parallel.zig        Parallel test runner for zig build test/ptest
  dict/
    dictionary.zig                dictionary registry + auto-detect
    trainer.zig                   FASTCOVER dictionary trainer
    builtin/*.dict                7 compiled-in dictionaries
  format/
    streamlz_constants.zig        centralized constants
    frame_format.zig              SLZ1 frame + block headers
    block_header.zig              internal 2-byte + 4-byte chunk headers
    parallel_decode_metadata.zig  sidecar wire format (legacy)
  io/
    bit_reader.zig                MSB-first bit reader
    bit_writer.zig                4 bit-writer variants (with debug bounds)
    copy_helpers.zig              SIMD copy helpers + alignment
    ptr_math.zig                  signed-offset pointer arithmetic
  platform/
    memory_query.zig              OS memory query (Windows/Linux/macOS)
    mmap.zig                      cross-platform mmap helpers (read/read-write)
  decode/
    streamlz_decoder.zig          framed decompress + DecompressContext
    decompress_parallel.zig       parallel dispatch (SC + two-phase)
    cross_chunk_analyzer.zig      cross-chunk dependency analyzer (legacy sidecar builder)
    fixture_tests.zig             test-only: fixture roundtrips
    fast/
      fast_lz_decoder.zig         L1-L5 codec hot loop
    high/
      high_lz_decoder.zig         L6-L11 codec entry
      high_lz_token_executor.zig  L6-L11 token resolve + execute
    entropy/
      entropy_decoder.zig         tANS / Huffman / RLE dispatch
      huffman_decoder.zig         canonical Huffman 11-bit LUT
      tans_decoder.zig            tANS 5-state decode
      bit_reader_lite.zig         shared Golomb-Rice infrastructure
  encode/
    streamlz_encoder.zig          public compress API
    fast_framed.zig               Fast codec frame builder
    high_framed.zig               High codec frame builder
    compress_parallel.zig         parallel compress dispatch
    forward_lz.zig                Forward-LZ experimental module
    cost_coefficients.zig         empirical timing coefficients
    match_eval.zig                shared match evaluation helpers
    match_hasher.zig              MatchHasher family (bucket-based)
    offset_encoder.zig            LZ offset encoding pipeline
    text_detector.zig             text-probability heuristic
    encode_fixture_tests.zig      test-only: encode roundtrips
    fast/
      fast_lz_encoder.zig         sub-chunk encoders (raw/entropy)
      fast_lz_parser.zig          greedy + lazy chain parser
      fast_stream_writer.zig      6-stream output buffer
      fast_match_hasher.zig       single-entry Fibonacci hash
      fast_token_writer.zig       Fast cmd encoding helpers
      fast_constants.zig          level mapping + min-match table
      fast_cost_model.zig         decode-time cost estimates
    high/
      high_compressor.zig         level dispatch + setup
      high_encoder.zig            stream writer + assemble output
      high_optimal_parser.zig     forward DP + backward extraction
      high_greedy_parser.zig      greedy/lazy for L6-L8
      high_matcher.zig            match ranking
      high_types.zig              Token, State, Stats, CostModel
      high_cost_model.zig         per-symbol cost calculation
      managed_match_len_storage.zig  MLS + VarLen codec
      match_finder.zig            hash-based finder (SIMD probe)
      match_finder_bt4.zig        binary-tree finder for L11
    entropy/
      entropy_encoder.zig         encodeArrayU8 + tANS + memcpy
      tans_encoder.zig            tANS encoder
      byte_histogram.zig          byte frequency histogram
```

The decode side has zero dependencies on the encode side; either can be vendored
independently.

---

## Two codecs

- **Fast** (levels 1-5) — greedy/lazy parser, raw or entropy-coded literals
- **High** (levels 6-11) — optimal DP parser, full entropy coding

**Entry points:**
- CLI: `main.zig` → `cli.zig`
- Library: `streamlz.zig` (re-exports `compressFramed` / `decompressFramed`)

---

## Parallel architecture

### Decode

| Levels | Strategy | How it works |
|--------|----------|--------------|
| L1 | SC group-parallel | Encoder compresses each 256KB chunk independently (no cross-chunk refs). Decoded via `decompressCoreParallel` using frame header's `sc_group_size`. No sidecar. |
| L2-L5 | SC group-parallel | Encoder constrains chunks to self-contained groups with adaptive group sizing. Each group decoded independently. |
| L6-L8 | SC group-parallel | Encoder constrains chunks to self-contained groups (adaptive size, ~16 groups per file). Each group decoded independently. |
| L9-L11 | Two-phase | Phase 1: parallel entropy decode + resolveTokens. Phase 2: serial token execution. |

### Compress

| Levels | Threading | How it works |
|--------|-----------|--------------|
| L1 | Parallel | Per-chunk workers with independent hashers (SC mode). |
| L2-L5 | Parallel | Per-group workers with independent hashers (SC mode, adaptive group sizing). |
| L6-L8 | Parallel | Per-group workers with independent match finders (SC mode). |
| L9-L11 | Parallel | Per-block workers sharing pre-computed MLS. |

All decode paths share a persistent `std.Thread.Pool` to avoid per-call thread spawn cost.

---

## Pointer safety

- `[]u8` slices at all public API boundaries.
- `[*]u8` raw pointers for struct fields that advance through a buffer (hot-loop iterators).
- Every raw pointer stored in a struct field has a matching `std.debug.assert` that validates bounds in Debug / ReleaseSafe builds — zero cost in ReleaseFast.
- See `io/bit_writer.zig` `initBounded` for the canonical pattern.

---

## Key invariants

For anyone modifying the hot loops:

1. **`@Vector(16, u8)` not `@Vector(32, u8)`.** AVX2 (32-byte
   loads/stores) throttles Arrow Lake frequency under load and is measurably
   slower for this workload. Confirmed via direct A/B tests.

2. **`wildCopy16` interleaved load-store-load-store ordering** (not load-load-
   store-store). The interleave is REQUIRED for small-offset LZ match
   propagation: a single 16-byte vector load would read unwritten bytes from
   past the write cursor, breaking RLE-like patterns. Locked in by an
   isolation test in `copy_helpers.zig`.

3. **tANS `src_start` / `src_end` capture order.** Must be set on the parameter
   block BEFORE state-init shifts `src` / `srcEnd` by ±4. The decode hot loop's
   bounds check uses the "outer" range and intentionally steps into the
   state-init overlap region. Easy to miss when porting — see commit `bbfadc1`.

4. **SC per-group `dst_start`.** L6-L8 chunks group in 4s for the SC path. The
   decoder must pass a group-local `dst_start` (not whole-buffer-start) to each
   chunk so the encoder's per-group "first chunk = base_offset 0" assumption
   holds and the initial 8-byte raw `Copy64` fires at each group's start.

5. **Tail prefix restoration (SC only).** Encoder stores `(num_chunks − 1) × 8`
   bytes at the end of each frame block. After decoding all chunks, overwrite
   first 8 bytes of every chunk except chunk 0 with those tail bytes.

6. **`comptime mode: LiteralMode`** in `decode/fast/fast_lz_decoder.zig`
   (`processModeImpl`) generates two specialized functions (delta-literal vs
   raw-literal). Unambiguous win for branch-free hot loops because there's only
   one callsite per specialization.

7. **Default build target `x86_64_v3`.** Set in `build.zig` so binaries are
   portable across modern Intel and AMD x86_64 hosts. AVX-512 instructions
   from `-mcpu=native` on Arrow Lake crashed earlier builds on Ryzen 3800x.

---

## Cleanness analyzer (`streamlz analyze`)

A diagnostic-only token-dependency-DAG analyzer lives in
`decode/cross_chunk_analyzer.zig`. It walks an `.slz` file's LZ tokens, builds
the per-byte dependency DAG, and reports:
- Total match tokens
- Critical path depth (longest dependency chain)
- Round histogram (how many tokens at each depth in the DAG)
- Theoretical parallel-decode speedup with N cores

Built during a session that explored speculative parallel decode for L1-L5.
Conclusion: theoretical parallelism is large (~24× at 24 cores), but the
analyzer itself takes 60-300 ms per 100 MB input — far longer than the ~16 ms
decode it would speed up. Kept for offline diagnostic work.

---

## Per-token serial dependency and L1 load latency

Every LZ decompressor has a serial pointer chase through the compressed
stream: read token → decode lengths → copy literals → advance src
pointer → read next token. The critical path per token is:

```
load [src]  →  decode (ALU)  →  compute new src  →  load [src]
   4-5 cyc       2-3 cyc           1 cyc             4-5 cyc
```

On Skylake (4-cycle L1 load): ~11 cycles/token.
On Arrow Lake (5-cycle L1 load): ~13 cycles/token.
That 18% per-clock penalty is architectural — no code change can fix it.

Verified empirically: the LZTurbo binary's exact instruction sequence
(extracted via VTune, reassembled as a .s file) runs at 3,756 MB/s on
Arrow Lake vs the author's published 3,970 MB/s on Skylake — matching
the predicted ratio after adjusting for clock speed and load latency.

StreamLZ's Fast decoder has the same serial dependency through
`cmd_stream`. The architectural response is SC group-parallel decode:
each thread runs its own independent pointer chase, so aggregate
throughput is memory-bandwidth-bound (~40 GB/s at 24 threads), not
latency-bound. This is the primary reason L1-L5 use SC encoding.

---

## Glossary

| Term | Meaning |
|------|---------|
| Fast codec | Greedy/lazy LZ parser, levels 1-5. Fast compress + decompress. |
| High codec | Optimal DP parser, levels 6-11. Slower compress, same decompress speed. |
| tANS | Tabled Asymmetric Numeral System — entropy coder used for literal streams. |
| BT4 | Binary Tree 4-way — match finder used by the High codec's optimal parser. |
| SC | Self-Contained — block format flag; each block decodes independently. |
| Sidecar | (Legacy) Optional parallel-decode metadata block (cross-chunk dependency map). No longer emitted by current encoder. |
| Chunk | 256 KB decompression unit. Sub-chunk = 128 KB half. |
| Cross-chunk | A match whose source bytes are in a different chunk than its target. |
| Cleanness | Whether a byte's dependency chain stays within one chunk (no cross-chunk refs). |
| PPOC | (Legacy) Parallel Producer/Consumer — sidecar builder. No longer emitted by current encoder. |
| MLS | Managed Match Length Storage — variable-length match table for the High codec. |
