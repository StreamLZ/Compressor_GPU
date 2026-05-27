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

## Stall-reason breakdown (silesia L5, from NCU `--set full` PC sampling)

Per-issued-instruction warp cycle accounting. The "% of cycles" column is each
stall reason's share of total warp cycles per issued instruction; the bigger
the share, the bigger the win from removing that stall reason.

### slzLzDecodeKernel — total 14.5 warp cycles / issued inst

| Stall reason | Cycles | % of cycles | What it means |
|---|---:|---:|---|
| **long_scoreboard** | **5.76** | **39.7%** | Waiting on L1TEX (global/local/surface/texture) data dependency. The NCU rule pegs this exactly: 5.8 cycles waiting for L1TEX scoreboard. Mostly the dst[match_src] reads in warpMatchCopy. |
| **wait** | 3.36 | 23.2% | Fixed-latency dependency (e.g. `__syncwarp`, FP/INT pipe latency). Hard to attack directly. |
| **not_selected** | 1.35 | 9.3% | Warp was ready but the scheduler picked another (warps competing for the same issue slot — symptom of healthy occupancy, not a problem). |
| **branch_resolving** | 1.02 | 7.0% | Waiting for branch target to resolve. With 25.4% of instructions being branches (349M / 1373M) and 9.2% of those divergent (32M), this is structural. |
| **short_scoreboard** | 0.65 | 4.5% | Waiting on MIO (shared mem, constant, special) data dep. |
| **mio_throttle** | 0.52 | 3.6% | MIO pipe at throughput cap. |
| **no_instruction** | 0.47 | 3.2% | I-cache miss — the kernel is large enough that I-cache pressure shows. |
| **math_pipe_throttle** | 0.29 | 2.0% | FP/INT pipe at cap. |
| _other_ | 1.08 | 7.5% | dispatch_stall, drain, lg_throttle, imc_miss, etc. |

### slzHuffDecode4StreamKernel — total 21.7 warp cycles / issued inst

| Stall reason | Cycles | % of cycles | Note |
|---|---:|---:|---|
| **long_scoreboard** | **16.09** | **74.2%** | Memory-latency-bound. Kernel already at 96% memory SoL — this is the BIL refill pattern paying off (one stall, big refill), not a sign of a fixable problem. |
| wait | 1.71 | 7.9% | |
| short_scoreboard | 1.59 | 7.3% | Shared-mem LUT dependencies. |
| not_selected | 0.38 | 1.7% | |
| math_pipe_throttle | 0.35 | 1.6% | |
| mio_throttle | 0.36 | 1.7% | |
| imc_miss | 0.21 | 1.0% | |
| _other_ | 1.00 | 4.6% | |

### slzHuffBuildLutKernel — total 8.4 warp cycles / issued inst

| Stall reason | Cycles | % of cycles | Note |
|---|---:|---:|---|
| **wait** | **2.55** | **30.3%** | Fixed-latency / `__syncwarp`. |
| **short_scoreboard** | **2.04** | **24.3%** | Shared-mem dependencies (the LUT-build pattern reads/writes the shared LUT extensively). |
| barrier | 1.32 | 15.6% | `__syncwarp` between passes. |
| no_instruction | 0.67 | 8.0% | |
| long_scoreboard | 0.33 | 3.9% | Very little — kernel is mostly compute + shared mem. |
| not_selected | 0.32 | 3.8% | |

### slzGatherRawOff16Kernel — total 72.1 warp cycles / issued inst

| Stall reason | Cycles | % of cycles | Note |
|---|---:|---:|---|
| **long_scoreboard** | **64.26** | **89.2%** | Pure DRAM-latency wait. The kernel is a sparse byte-gather (256 lanes × 256-byte stride writes) — bound by DRAM latency, not bandwidth. Expected for a gather pattern; not worth attacking. |
| wait | 3.03 | 4.2% | |
| drain | 0.47 | 0.7% | |
| _other_ | 4.32 | 6.0% | |

## Source-level signals from NCU rules

Beyond the stall breakdown, NCU's static analysis surfaced these specific
problems in `slzLzDecodeKernel`:

1. **Register spill to local memory: 8,367,316 requests** — the kernel is using 40
   registers/thread but has live-state pressure that spills to L1/L2/DRAM.
2. **Local stores: 1.0 of 32 bytes per sector actually used** (NCU est. speedup 70%).
   The spill writes are scattered (one byte per lane per sector). Every spill
   store wastes ~31/32 of L1 bandwidth.
3. **Local loads: 6.9 of 32 bytes per sector used** (NCU est. speedup 57%).
   Spill reads are slightly better-aligned but still ~22% efficient.
4. **Global stores: 5.4 of 32 bytes per sector used** (NCU est. speedup 60%).
   Output-write coalescing is poor — short literal/match copies have small tails
   that fragment the stores.
5. **Uncoalesced global accesses: 12.6 MB excessive sectors** (15% of total 86 MB,
   NCU est. speedup 14%). Probably the `dst[pos - off]` match-source reads at
   non-warp-aligned positions.
6. **Achieved occupancy 79% vs theoretical 100%** (NCU est. speedup 21%). One
   full wave + a partial wave of 808 blocks; the partial wave costs ~50% of the
   theoretical kernel runtime that a no-tail launch would have.
7. **Avg active threads/warp 21.8, predicated-on 20.6** (NCU est. speedup 26%).
   Divergent literal/match-copy short tails leave ~1/3 of the warp idle.

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

Concrete from the stall + source-counter analysis above:

1. **Eliminate register spill in LzDecode** (NCU est. speedup ~30-50%).
   8.37M local-memory spill requests with 1.0/32 byte sector utilization
   means every spilled register write wastes 31/32 of L1 bandwidth and
   probably explains a large fraction of the 39.7% `long_scoreboard`
   stall and the 4.5% `short_scoreboard` stall. Approach: trim live
   register state in the inner parse loop (`ParsedStreams` struct fields,
   loop-carried cursors) or split the kernel so spilled state isn't live
   across the heavy memory phases.
2. **LzDecode lane utilization** (NCU est. speedup 26%). Push active
   threads/warp from 21.8 → 32. The literal-copy and match-copy short
   tails are the target — when `lit_len < 32` or `match_len < 32`, some
   lanes idle. A predicated-write approach or per-lane work redistribution
   would help.
3. **LzDecode tail-wave occupancy** (NCU est. speedup 21%). 1624 blocks
   on 34 SMs × ~12 blocks/SM ≈ 1 full wave + a 808-block partial wave.
   Larger-per-warp work (fewer blocks, more chunks per warp) would
   eliminate the tail. Current `chunks_per_group=1` is the lever.
4. **LzDecode uncoalesced output stores** (NCU est. speedup 60%). The
   global-store sector utilization is 5.4/32. Writing literals/matches
   one byte at a time per lane is the cause. A 4-byte vectorized store
   path (alignment-permitting) would dramatically improve this.
5. **HuffDecode** — already at 96% mem SoL with 74% of stalls being
   long_scoreboard. Bandwidth-bound. No easy wins; would need a smaller
   working set (smaller LUT, different code length distribution).
6. **HuffBuild** — 30% wait + 24% short_scoreboard + 16% barrier. Already
   warp-cooperative; further wins require either fewer barrier rounds or
   a fundamentally different LUT-build algorithm.
7. **Gather** — 89% long_scoreboard. Inherent for sparse byte-gather.
   Not worth pursuing.

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
