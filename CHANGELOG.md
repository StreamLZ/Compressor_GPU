# Changelog

## [2.0.0] — 2026-04-29

Initial native (Zig) release. Full port of the C# StreamLZ library to
Zig 0.15.2 with byte-exact wire-format compatibility.

### Highlights

- **All 11 compression levels** (Fast L1-L5, High L6-L11) with compress
  and decompress at full parity with v1.
- **Parallel decompress** at all levels: SC group-parallel for L1-L5
  (adaptive group size), SC group-parallel for L6-L8 (adaptive group
  size), two-phase parallel for L9-L11.
- **Dictionary support**: 7 built-in dictionaries (JSON, HTML, CSS, JS,
  XML, text, general) with auto-detection by file extension. Custom
  dictionary training via FASTCOVER algorithm. Zero-copy decompress
  (no memmove overhead).
- **128 MB dictionary window** for L11 (BT4 match finder).
- **Flag-driven CLI** (no subcommands): compress, decompress, benchmark,
  bench-all, decompress-bench, info, train, version.
- **297 unit tests** + 140 fixture roundtrips against C# reference output.
- **Fuzz harness** for decompressor safety testing.
- Streaming decompress to `std.Io.Writer` with XXH32 content-checksum
  verification.

### Performance (Arrow Lake-S, 24 cores, enwik8 100 MB, best of 3)

- L1 parallel decompress: 34 GB/s
- L2-L5 SC group-parallel decompress: 37-39 GB/s
- L6-L8 SC group-parallel decompress: 11-12 GB/s
- L9-L11 two-phase parallel decompress: 1.4-2.3 GB/s
- L1 parallel compress: 4.2 GB/s
- L9 compress: 7.9 MB/s (SIMD hash probe + dual-bucket prefetch)

### Key optimizations over v1

- L1 ratio 58.6% → 54.9%: u32 hash positions, 17-bit hash table,
  content-first probe, on-the-fly skip computation
- L6-L8 ratio 2.3pp improvement: adaptive SC group sizing (~16 groups
  per file, wider match window per parallel worker)
- mmap I/O for compress and decompress (disk-to-disk L1 decompress
  37ms vs lzturbo's 46ms on enwik8)
- Fast decoder split loop: bounds-check-free fast inner loop + safe
  tail (+4.6% L1 single-thread decompress)
- High decoder: redundant match_addr bounds check eliminated from
  processOneToken fast path (~1.2% L9 decompress improvement)
- CMOV LIFO swap in Fast short-token loop
- Far-offset MOVDQU widening for medium/long match paths
- Conditional far-offset prefetch for L3-L5
- SIMD hash probe + dual-bucket prefetch for L9-L11 encoder
- Parallel resolveTokens for L9-L11 (+17-24% decompress)
- SC group-parallel for L2-L5 (supersedes v2 sidecar frame format)
- L1 SC per-chunk independence (no sidecar needed)

### Build

- Optional vendor libs (`-Dbench=true`): default build excludes
  zstd/lz4, binary 2.1 MB → 1.3 MB, incremental builds ~200ms
- Static library caching for zstd/lz4 vendor code

### Format

- v2 frame header: `sc_group_size` byte replaces v1 reserved byte.
  Encoders write actual group size (1-255); decoders must use this
  value for SC group boundary calculations.
