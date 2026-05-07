# StreamLZ Zig Benchmarks

Intel Ultra 9 285K (Arrow Lake-S), 24 cores, Windows 11.

## Post-audit6 (2026-05-07)

Zig 0.16.0 `ReleaseFast -Dbench=true`. All 6 audit rounds complete.
L1-L5 SC group-parallel with hash bit caps (L1=17, L2=18, L3=19, L4=20,
L5=24 chain). SC group size capped: L1-L4 min(adaptive,16), L5 adaptive.

### enwik8 100 MB, 24 threads, -ba

| Level | Ratio | Compress MB/s | Decompress MB/s |
|-------|------:|-----:|-------:|
| L1  | 53.7% | 2,566 | 34,685 |
| L2  | 53.0% | 2,451 | 33,092 |
| L3  | 51.7% | 1,729 | 37,235 |
| L4  | 49.0% | 1,168 | 36,678 |
| L5  | 41.9% |   746 | 40,052 |
| L6  | 29.1% |  43.5 | 11,633 |
| L7  | 29.0% |  33.4 | 11,381 |
| L8  | 28.6% |  17.9 | 12,414 |
| L9  | 27.4% |   7.0 |  2,671 |
| L10 | 27.3% |   6.8 |  3,209 |
| L11 | 25.6% |   1.3 |  3,091 |

All Fast levels (L1-L5) compress and decompress in parallel via SC
groups. Decompress speed is memory-bandwidth-bound at 33-40 GB/s
across all Fast levels.

---

## Historical: single-threaded Zig vs C# (Phase 7b, 2026-04)

Zig 0.15.2, single-threaded decompress only. Zig mean vs C# median.

| fixture | level | Zig MB/s | C# MB/s | Zig / C# |
|---|---|---:|---:|---:|
| silesia_all.tar | L1  | 6,582 | 6,307 | 1.04× |
| silesia_all.tar | L5  | 5,472 | 5,527 | 0.99× |
| silesia_all.tar | L9  |   992 | 1,196 | 0.83× |
| silesia_all.tar | L11 |   941 | 1,225 | 0.77× |
| enwik8.txt      | L1  | 6,051 | 5,886 | 1.03× |
| enwik8.txt      | L9  |   672 |   922 | 0.73× |
| enwik8.txt      | L11 |   691 |   915 | 0.76× |

L1/L5 at parity. L9/L11 gap is the High codec hot loop (prefetch,
cascading literal copy). Now superseded by parallel decompress numbers
above.
