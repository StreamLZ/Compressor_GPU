# L5 decode harness baseline

Captured 2026-05-27 on RTX 4060 Ti (sm_89, 34 SMs, 16 GB VRAM), commit
579b608+ with the L4/L5 hash_bits=17 + adaptive sc patches landed.

The harness in this directory replays the production L5 decode pipeline
end-to-end on a SCN1 scan-dump from `SLZ_DUMP_SCAN=… streamlz.exe -d -gpu`.
It launches the same 4 GPU kernels production launches, in the same order,
on the same data. Output bytes are verified SHA-equal to the original
source file before any timing is reported.

## End-to-end timing parity vs production

`-db -r 30` on production binary, `decode_l5_harness -r 30` for the
harness; both measure the cuEvent wall-clock around the full kernel
pipeline (best of 30):

| File           | Decompressed | Production kernel best | Harness kernel best | Δ     | SHA verified |
|----------------|-------------:|-----------------------:|--------------------:|------:|:------------:|
| enwik8 (L5)    |   95.37 MB   |               4.12 ms  |           4.40 ms   | +6.8% | ✓ matches `assets/enwik8.txt`        |
| silesia (L5)   |  202.94 MB   |               7.72 ms  |           7.77 ms   | +0.6% | ✓ matches `assets/silesia_all.tar`   |

silesia is within run-to-run noise (0.6%). enwik8 has ~7% gap — small
absolute (0.28 ms) and consistent with the smaller workload seeing more
relative weight from CUDA launch overhead inside the cuEvent window.
Either way: the kernels are the production kernels, the inputs are the
production inputs, and the output is byte-identical to the source.
Experiments measured against this baseline are meaningful.

## Per-kernel NCU breakdown (silesia L5)

```
$ tools/decode_l5/profile_decode_l5.bat c:/tmp/sil_l5_scan.bin assets/silesia_all.tar
```

| Kernel                       | Grid × Block            | Duration  | %Mem (SoL) | %Compute SM | DRAM%  | L1 hit | L2 hit | Occupancy | Avg active threads/warp | Warp cycles / inst | IPC active |
|------------------------------|-------------------------|----------:|-----------:|------------:|-------:|-------:|-------:|----------:|------------------------:|-------------------:|-----------:|
| `slzGatherRawOff16Kernel`    | (2133, 1, 1) × (256, 1) |   63.2 µs |     50.5%  |     25.9%   | 50.5%  | 59.4%  | 50.0%  |   91.5%   |          31.87 / 32     |              72.07 |       0.62 |
| `slzHuffBuildLutKernel`      | (10294, 1, 1) × (32, 1) |  367.2 µs |     49.4%  |     49.4%   | 27.4%  |  3.2%  | 95.8%  |   30.5%   |           6.79 / 32     |               8.42 |       1.72 |
| `slzHuffDecode4StreamKernel` | (10294, 1, 1) × (32, 1) |   1.37 ms |     96.3%  |     32.9%   | 46.7%  | 50.3%  | 87.3%  |   41.1%   |          25.37 / 32     |              21.70 |       0.91 |
| `slzLzDecodeKernel`          | (1624, 1, 1) × (32, 2)  |   7.14 ms |     71.5%  |     71.5%   | 22.4%  | 71.0%  | 89.9%  |   78.6%   |          21.82 / 32     |              14.46 |       2.61 |
| **Pipeline (sum)**           |                         | **~8.94 ms** | — | — | — | — | — | — | — | — | — |

(NCU instrumentation overhead inflates kernel duration ~15% vs unprofiled
`-db` runs; the ratio between kernels is still the right shape.)

## Per-kernel NCU breakdown (enwik8 L5)

```
$ tools/decode_l5/profile_decode_l5.bat c:/tmp/ew8_l5_scan.bin assets/enwik8.txt
```

| Kernel                       | Grid × Block            | Duration  | %Mem (SoL) | %Compute SM | DRAM%  | L1 hit | L2 hit | Occupancy | Avg active threads/warp | Warp cycles / inst | IPC active |
|------------------------------|-------------------------|----------:|-----------:|------------:|-------:|-------:|-------:|----------:|------------------------:|-------------------:|-----------:|
| `slzGatherRawOff16Kernel`    | (1525, 1, 1) × (256, 1) |   74.1 µs |     61.7%  |     26.2%   | 61.7%  | 58.3%  | 45.5%  |   92.2%   |          31.92 / 32     |              72.17 |       0.61 |
| `slzHuffBuildLutKernel`      | (4577, 1, 1) × (32, 1)  |  178.8 µs |     49.7%  |     49.7%   |  7.9%  |  3.2%  | 95.0%  |   30.1%   |           6.22 / 32     |               8.07 |       1.77 |
| `slzHuffDecode4StreamKernel` | (4577, 1, 1) × (32, 1)  |  759.4 µs |     95.4%  |     31.9%   | 41.7%  | 49.9%  | 88.0%  |   40.3%   |          24.01 / 32     |              21.71 |       0.89 |
| `slzLzDecodeKernel`          | (763, 1, 1) × (32, 2)   |   3.70 ms |     73.7%  |     73.7%   | 23.6%  | 58.1%  | 88.8%  |   84.0%   |          22.62 / 32     |              15.49 |       2.60 |
| **Pipeline (sum)**           |                         | **~4.71 ms** | — | — | — | — | — | — | — | — | — |

## Bottleneck reading — what matters for future experiments

**1. `slzLzDecodeKernel` is the dominant cost** — 7.14 / 8.94 ≈ **80% of silesia
pipeline time**, 3.70 / 4.71 ≈ **79% of enwik8**. Anything we optimize here
moves the headline number. Everything else is rounding error in comparison.

LZ-decode is the most balanced kernel we have: high occupancy (78-84%),
high IPC (~2.6, close to Ada's 4-IPC ceiling on a 4-wide scheduler),
high SM throughput (71-74%), high memory throughput (71-74%). It's
already healthy. The headroom signals:
- **Mem Pipes Busy = 71-74%** but DRAM only 22-24% — most memory traffic
  hits L1/L2 (71 / 58% L1 hit, 90 / 89% L2 hit). The bottleneck is
  L1-pipe throughput, not DRAM bandwidth.
- **Avg active threads/warp = 21-22 / 32** — ~68% lane utilization. The
  remaining ~30% comes from divergent literal/match-copy short tails and
  the per-token serial parse phase. nvCOMP LZ4's MATCH.ANY-style
  warp-cooperative parse would attack this.
- **Warp cycles per issued inst = 14-15** — long-latency dependent loads
  (the L1 hit pattern stalls a warp ~15 cycles between dependent ops).

**2. `slzHuffDecode4StreamKernel` is the second largest cost** — 1.37 ms
on silesia (15% of pipeline) / 759 µs on enwik8 (16%). Already at 96% of
memory SoL ceiling — bandwidth-bound. Hard to speed up without a smaller
working set.

**3. `slzHuffBuildLutKernel` is small** — 367 µs / 179 µs (~4% each). Per
NCU it's at 1.72-1.77 IPC active but only 6-7 active threads per warp.
Looks like a candidate for warp-cooperative LUT fill (the existing
ba46e9a rewrite already did this — what we see is post-rewrite numbers).
Diminishing returns here.

**4. `slzGatherRawOff16Kernel` is the smallest cost** — 63-74 µs each.
Bandwidth-bound (50-62% of SoL on the gather). Not worth pursuing.

### Optimization targets, ordered by ROI

1. **LZ decode kernel lane utilization** — push avg active threads/warp
   from ~22 toward 32. The literal-copy and match-copy phases are the
   obvious targets (short tails leave many lanes idle).
2. **LZ decode L1-pipe pressure** — see if `__ldg` / `__ldcs` cache hints
   on the match-source reads change anything. NCU says L1 hit 71%,
   so most reads are cache-resident, but the pipe is busy at 71%.
3. **Reduce HuffDecode bandwidth pressure** — already at SoL ceiling; a
   smaller LUT or in-shared-mem reuse pattern might help, but the
   kernel is already in a warp-cooperative state.
4. **Fuse Gather + HuffBuild** — gather is 1% of pipeline, fusing saves a
   tiny launch overhead but is risky. Probably not worth it.

## Reproducing this baseline

```
# 1. Encode silesia + enwik8 at L5 (uses the production binary)
zig-out/bin/streamlz.exe -c -l 5 -gpu assets/silesia_all.tar -o c:/tmp/sil_L5.slz
zig-out/bin/streamlz.exe -c -l 5 -gpu assets/enwik8.txt       -o c:/tmp/ew8_L5.slz

# 2. Dump the SCN1 scan via production decode (the dump captures all the
#    GPU kernel inputs the harness will replay)
SLZ_DUMP_SCAN=c:/tmp/sil_l5_scan.bin zig-out/bin/streamlz.exe -d -gpu c:/tmp/sil_L5.slz -o c:/tmp/sil_dec.bin
SLZ_DUMP_SCAN=c:/tmp/ew8_l5_scan.bin zig-out/bin/streamlz.exe -d -gpu c:/tmp/ew8_L5.slz -o c:/tmp/ew8_dec.bin

# 3. Build the harness
tools/decode_l5/build_decode_l5.bat

# 4. Run the harness for speed + correctness check
tools/decode_l5/decode_l5_harness.exe c:/tmp/sil_l5_scan.bin assets/silesia_all.tar 30
tools/decode_l5/decode_l5_harness.exe c:/tmp/ew8_l5_scan.bin assets/enwik8.txt 30

# 5. NCU profile
tools/decode_l5/profile_decode_l5.bat c:/tmp/sil_l5_scan.bin assets/silesia_all.tar
tools/decode_l5/profile_decode_l5.bat c:/tmp/ew8_l5_scan.bin assets/enwik8.txt
```
