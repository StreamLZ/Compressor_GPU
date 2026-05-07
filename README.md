# StreamLZ Native

Zig 0.16 implementation of [StreamLZ](https://github.com/StreamLZ/StreamLZ), a fast LZ77-family
compressor/decompressor. Covers all 11 compression levels (Fast L1-L5,
High L6-L11) with byte-exact wire-format compatibility with the C#
reference. Primary goal: **fast decompression** on consumer x86-64.

---

## Quick start

```
zig build -Doptimize=ReleaseFast
```

The CLI binary lands at `zig-out/bin/streamlz.exe` (Windows) or
`zig-out/bin/streamlz` (Linux/macOS).

```
streamlz file.txt                    # compress (default L3)
streamlz -l 9 file.txt              # compress at level 9
streamlz -d file.slz                # decompress
streamlz -b -l 5 file.txt           # benchmark level 5
streamlz -ba file.txt               # benchmark all L1-L11
streamlz -db file.slz               # decompress-only benchmark
streamlz -i file.slz                # frame/block info
streamlz --train -o dict.bin corpus/ # train custom dictionary
```

Dictionary flags: `-D name` selects a built-in dictionary, `--no-dict`
disables auto-detection.

---

## Build, test, fuzz

```
zig build -Doptimize=ReleaseFast                # release binary
zig build -Doptimize=ReleaseFast -Dstrip=false  # release + symbols (VTune)
zig build test --summary all                    # 329 unit tests
zig build safe                                  # ReleaseSafe build
zig build fuzz                                  # fuzz harness
```

Default target is `x86_64_v3` (Haswell+) for Intel/AMD portability.
Override with `-Dcpu=native` for host-specific tuning.

The fixture suite (`fixture_tests` + `encode_fixture_tests`) roundtrips
140 corpus files byte-exact against the C# reference. Generate fixtures
with `scripts/gen_fixtures.sh` and set `STREAMLZ_FIXTURES_DIR=./fixtures`
before running tests.

---

## Benchmarks

Intel Core Ultra 9 285K (Arrow Lake-S), Windows 11, Zig 0.16 / LLVM 21,
`-Doptimize=ReleaseFast`. Best of 100 runs.

### vs zstd and LZ4 (single-threaded, enwik8 100 MB)

Single-threaded comparison. All compressors use full-stream mode (no block splitting).

| Compressor | Ratio | Compress | Decompress |
|-----------|-------|----------|------------|
| LZ4       | 57.3% |     683 MB/s |  5,103 MB/s |
| LZ4 HC 4  | 43.0% |    96.8 MB/s |  4,793 MB/s |
| LZ4 HC 9  | 42.2% |    51.9 MB/s |  4,912 MB/s |
| LZ4 HC 12 | 41.9% |    24.9 MB/s |  4,839 MB/s |
| zstd 1    | 40.7% |     459 MB/s |  1,771 MB/s |
| zstd 3    | 35.4% |     311 MB/s |  1,435 MB/s |
| zstd 9    | 31.1% |    78.2 MB/s |  1,556 MB/s |
| zstd 19   | 26.9% |     2.7 MB/s |  1,571 MB/s |
| **SLZ L1**  | 53.7% |     308 MB/s |  **6,441 MB/s** |
| **SLZ L3**  | 51.7% |     222 MB/s |  **4,940 MB/s** |
| **SLZ L5**  | 41.9% |    71.7 MB/s |  **5,269 MB/s** |
| **SLZ L6**  | 27.4% |     3.1 MB/s |  1,127 MB/s |
| **SLZ L8**  | 25.5% |     0.9 MB/s |  1,108 MB/s |
| **SLZ L9**  | 27.4% |     3.1 MB/s |  1,092 MB/s |
| **SLZ L11** | 25.5% |     0.2 MB/s |  1,075 MB/s |

At the fast tier: SLZ L1 decompresses **3.6x faster** than zstd 1
(6.4 vs 1.8 GB/s) and **1.3x faster** than LZ4 (6.4 vs 5.1 GB/s).
SLZ L5 matches LZ4 HC 12's ratio (41.9%) while decoding **faster**
(5.3 vs 4.8 GB/s) and compressing **faster** (72 vs 25 MB/s).
At the best-ratio tier: SLZ L11 achieves 25.5% vs zstd 19's 26.9%.

### vs zstd and LZ4 (single-threaded, silesia 203 MB)

| Compressor | Ratio | Compress | Decompress |
|-----------|-------|----------|------------|
| LZ4       | 47.5% |     915 MB/s |  5,282 MB/s |
| LZ4 HC 9  | 36.6% |    55.6 MB/s |  5,440 MB/s |
| zstd 1    | 34.7% |     555 MB/s |  2,101 MB/s |
| zstd 3    | 31.4% |     399 MB/s |  1,874 MB/s |
| zstd 9    | 27.9% |     103 MB/s |  2,024 MB/s |
| zstd 19   | 24.8% |     4.4 MB/s |  1,847 MB/s |
| **SLZ L1**  | 44.8% |     488 MB/s |  **6,853 MB/s** |
| **SLZ L3**  | 43.2% |     328 MB/s |  **5,649 MB/s** |
| **SLZ L5**  | 35.8% |    95.1 MB/s |  **5,916 MB/s** |
| **SLZ L6**  | 25.0% |     4.6 MB/s |  1,316 MB/s |
| **SLZ L8**  | 24.2% |     1.2 MB/s |  1,377 MB/s |
| **SLZ L9**  | 25.0% |     4.7 MB/s |  1,288 MB/s |
| **SLZ L11** | 24.2% |     0.3 MB/s |  1,326 MB/s |

### All levels (24 cores, enwik8 100 MB)

Full-speed numbers with all 24 cores. `streamlz -ba -r 100`.

| Level | Compressed | Ratio | Compress | Decompress |
|-------|------------|-------|----------|------------|
| L1  | 53,741,965 | 53.7% | 2,706 MB/s | 41,949 MB/s |
| L2  | 53,045,282 | 53.0% | 2,722 MB/s | 41,756 MB/s |
| L3  | 51,702,960 | 51.7% | 1,741 MB/s | 37,987 MB/s |
| L4  | 48,963,945 | 49.0% | 1,074 MB/s | 40,748 MB/s |
| L5  | 41,929,522 | 41.9% |   805 MB/s | 44,934 MB/s |
| L6  | 29,137,356 | 29.1% |    42 MB/s | 13,321 MB/s |
| L7  | 29,031,250 | 29.0% |    30 MB/s | 13,420 MB/s |
| L8  | 28,639,482 | 28.6% |    17 MB/s | 13,459 MB/s |
| L9  | 27,430,880 | 27.4% |   7.6 MB/s |  3,726 MB/s |
| L10 | 27,280,109 | 27.3% |   7.3 MB/s |  3,753 MB/s |
| L11 | 25,550,450 | 25.6% |   1.5 MB/s |  3,308 MB/s |

L1-L5 compress is parallel (SC, per-chunk/per-group workers);
L6-L11 compress parallel (High codec). All decompress is parallel:
L1-L5 SC group-parallel (adaptive group size), L6-L8 SC group-parallel
(adaptive group size), L9-L11 two-phase parallel.

### All levels (24 cores, silesia 203 MB)

| Level | Compressed | Ratio | Compress | Decompress |
|-------|------------|-------|----------|------------|
| L1  |  97,488,921 | 45.8% | 3,665 MB/s | 36,253 MB/s |
| L3  |  91,834,394 | 43.2% | 2,128 MB/s | 33,191 MB/s |
| L5  |  76,285,339 | 35.8% |   897 MB/s | 35,688 MB/s |
| L6  |  53,934,154 | 25.3% |    42 MB/s | 13,508 MB/s |
| L8  |  52,746,282 | 24.8% |    13 MB/s | 14,832 MB/s |
| L9  |  53,090,767 | 24.9% |    11 MB/s |  3,253 MB/s |
| L11 |  51,541,859 | 24.2% |   3.5 MB/s |  3,954 MB/s |

---

## Dictionary support

7 built-in dictionaries (32 KB each, compiled into the binary): JSON,
HTML, CSS, JS, XML, plain text, and a general-purpose dictionary.

Dictionaries are auto-detected by file extension (`.json` → JSON,
`.html` → HTML, `.txt` → text, etc.). Unknown extensions fall back to
the general dictionary. Override with `-D name` or disable with
`--no-dict`.

Custom dictionaries can be trained from a corpus:

```
streamlz --train -o my_dict.bin path/to/corpus/
```

The trainer uses the FASTCOVER algorithm (based on zstd's dictionary
builder).

---

## What's missing (v2.1)

- **Streaming compress wrapper** (`StreamLzFrameCompressor` equivalent)
- **SlzStream** reader-writer pair
- **Level enum** (currently takes an integer)

These are API surface gaps; all compression/decompression functionality
is complete.

---

## Project layout

```
build.zig              build script
src/                   all Zig source
scripts/               fixture generation + fuzz harness
CODEWIKI.md            source tree map, invariants, glossary
BENCHMARKS.md          historical benchmark numbers
CHANGELOG.md           release history
FORMAT.md              SLZ1 wire format specification
FAILED_EXPERIMENTS.md  optimization dead-ends (valuable context)
SECURITY.md            security policy + fuzz testing
```

For the full source tree map, key invariants, and glossary, see
[CODEWIKI.md](CODEWIKI.md).

---

## License

MIT — same as the upstream StreamLZ project.
