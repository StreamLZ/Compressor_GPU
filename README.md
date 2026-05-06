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
`-Doptimize=ReleaseFast`. Best of 30 decompress runs.

### vs zstd and LZ4 (single-threaded, enwik8 100 MB)

Single-threaded comparison. All compressors use full-stream mode (no block splitting).

| Compressor | Ratio | Compress | Decompress |
|-----------|-------|----------|------------|
| LZ4       | 57.7% |     689 MB/s |  5,022 MB/s |
| LZ4 HC 9  | 42.2% |    48.0 MB/s |  4,835 MB/s |
| zstd 1    | 40.9% |     466 MB/s |  1,796 MB/s |
| zstd 3    | 35.9% |     310 MB/s |  1,464 MB/s |
| zstd 9    | 31.8% |    80.1 MB/s |  1,547 MB/s |
| zstd 19   | 28.3% |     2.8 MB/s |  1,498 MB/s |
| **SLZ L1**  | 55.3% |     370 MB/s |  **6,769 MB/s** |
| **SLZ L3**  | 53.6% |    73.1 MB/s |  **4,568 MB/s** |
| **SLZ L5**  | 43.0% |    39.2 MB/s |  **4,505 MB/s** |
| **SLZ L6**  | 28.6% |     3.3 MB/s |  1,072 MB/s |
| **SLZ L8**  | 27.2% |     1.0 MB/s |  1,066 MB/s |
| **SLZ L9**  | 28.6% |     3.4 MB/s |  1,026 MB/s |
| **SLZ L11** | 27.2% |     0.3 MB/s |  1,020 MB/s |

At the fast tier: SLZ L1 decompresses **3.8x faster** than zstd 1
(6.8 vs 1.8 GB/s) and **1.3x faster** than LZ4 (6.8 vs 5.0 GB/s).
SLZ L5 matches LZ4 HC 9's ratio (43.0% vs 42.2%) while decoding
nearly as fast (4.5 vs 4.8 GB/s). At the best-ratio tier:
SLZ L11 achieves 27.2% vs zstd 19's 28.3%.

### vs zstd and LZ4 (single-threaded, silesia 203 MB)

| Compressor | Ratio | Compress | Decompress |
|-----------|-------|----------|------------|
| LZ4       | 47.5% |     920 MB/s |  5,417 MB/s |
| LZ4 HC 9  | 36.6% |    55.4 MB/s |  5,486 MB/s |
| zstd 1    | 34.7% |     599 MB/s |  2,038 MB/s |
| zstd 3    | 31.4% |     427 MB/s |  1,787 MB/s |
| zstd 9    | 27.9% |     109 MB/s |  1,936 MB/s |
| zstd 19   | 24.8% |     4.2 MB/s |  1,785 MB/s |
| **SLZ L1**  | 45.8% |     515 MB/s |  **7,367 MB/s** |
| **SLZ L3**  | 45.6% |    82.1 MB/s |  **3,443 MB/s** |
| **SLZ L5**  | 37.6% |    49.3 MB/s |  **4,713 MB/s** |
| **SLZ L6**  | 24.9% |     4.6 MB/s |  1,367 MB/s |
| **SLZ L8**  | 24.3% |     1.2 MB/s |  1,418 MB/s |
| **SLZ L9**  | 24.9% |     4.8 MB/s |  1,329 MB/s |
| **SLZ L11** | 24.3% |     0.3 MB/s |  1,366 MB/s |

### All levels (24 cores, enwik8 100 MB)

Full-speed numbers with all 24 cores. `streamlz -ba -r 30`.

| Level | Compressed | Ratio | Compress | Decompress |
|-------|------------|-------|----------|------------|
| L1  | 55,317,964 | 55.3% | 2,798 MB/s | 36,954 MB/s |
| L2  | 53,454,868 | 53.5% | 2,047 MB/s | 38,943 MB/s |
| L3  | 52,248,526 | 52.2% | 1,533 MB/s | 37,049 MB/s |
| L4  | 49,407,767 | 49.4% | 1,036 MB/s | 37,209 MB/s |
| L5  | 42,013,817 | 42.0% |   755 MB/s | 38,233 MB/s |
| L6  | 30,024,917 | 30.0% |    44 MB/s | 12,253 MB/s |
| L7  | 29,925,233 | 29.9% |    33 MB/s | 12,267 MB/s |
| L8  | 29,602,506 | 29.6% |    19 MB/s | 12,582 MB/s |
| L9  | 28,625,632 | 28.6% |   7.9 MB/s |  3,269 MB/s |
| L10 | 28,489,882 | 28.5% |   7.6 MB/s |  3,673 MB/s |
| L11 | 27,175,320 | 27.2% |   1.5 MB/s |  3,416 MB/s |

L1-L5 compress is parallel (SC, per-chunk/per-group workers);
L6-L11 compress parallel (High codec). All decompress is parallel:
L1-L5 SC group-parallel (adaptive group size), L6-L8 SC group-parallel
(adaptive group size), L9-L11 two-phase parallel.

### All levels (24 cores, silesia 203 MB)

| Level | Compressed | Ratio | Compress | Decompress |
|-------|------------|-------|----------|------------|
| L1  |  97,488,921 | 45.8% | 3,841 MB/s | 35,856 MB/s |
| L3  |  91,834,394 | 43.2% | 2,096 MB/s | 33,147 MB/s |
| L5  |  76,285,339 | 35.8% |   920 MB/s | 35,834 MB/s |
| L6  |  53,858,094 | 25.3% |    41 MB/s | 14,587 MB/s |
| L8  |  52,831,020 | 24.8% |    13 MB/s | 13,555 MB/s |
| L9  |  53,010,373 | 24.9% |    11 MB/s |  2,925 MB/s |
| L11 |  51,638,057 | 24.3% |   3.5 MB/s |  3,258 MB/s |

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
