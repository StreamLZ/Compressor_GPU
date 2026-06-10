# StreamLZ GPU Kernel Optimization — Findings & Architecture Comparison

## Current Performance (RTX 4060 Ti, 100 MB enwik8, L1, sc_group=0.25)

| Metric | StreamLZ | nvCOMP LZ4 |
|--------|----------|------------|
| Kernel throughput | **17.2 GB/s** | **18.6 GB/s** |
| End-to-end (incl PCIe) | 5.2 GB/s | — |
| Registers/thread | 48 | 48 |
| Shared memory/block | 0 | 1,792 bytes (896/warp) |
| Stack spill | 32 bytes | 0 |
| Warps active/cycle | 28.27 | 28.14 |
| Grid | 763 blocks | 763 blocks |
| Block | (32,2,1) = 2 warps | (32,2,1) = 2 warps |

## Optimization Journey

| Step | Kernel GB/s | Change |
|------|------------|--------|
| Batched token model (shared mem) | 4.7 | 40 KB shared → 1 block/SM |
| Remove shared memory entirely | 16.8 | 0 shared → 20 blocks/SM |
| `__launch_bounds__(64, 20)` | 17.1 | 64 → 48 registers |
| Simplified L1 decoder (no off32/delta/block-split) | 17.2 | Fewer live variables |
| Shared staging buffer (cmd+off16) | 15.7 | **Regression** — spill increased 32→40 bytes |
| Reverted to no-staging | 17.2 | Best achievable |

## Why We're 8% Slower — Root Cause (SASS Analysis)

### 1. Token read path: Global vs Shared memory

StreamLZ reads tokens from **global memory** (L1 cache):
```
LDG.E.U8.CONSTANT R18, [R16.64]    ; global load → goes through LSU pipeline
```

nvCOMP reads tokens from a **shared memory staging buffer**:
```
LDS.U8 R19, [R19]                   ; shared load → separate pipe, no LSU contention
```

`LDG` goes through the Load/Store Unit even on L1 hit. `LDS` uses a dedicated shared-memory pipe. This is the primary cause of our **4.4× higher mio_throttle** stalls (84K vs 19K samples).

### 2. Offset read: 2 global loads vs pre-staged

StreamLZ reads off16 values byte-by-byte from global memory:
```
LDG.E.U8 R19, [R16.64]       ; low byte
LDG.E.U8 R16, [R16.64+0x1]   ; high byte  
PRMT R16, R16, 0x7604, R19   ; combine to 16-bit
```

nvCOMP's offset is already in the shared staging window — zero additional LSU traffic.

### 3. Token parsing: 5-way branch vs single shift

StreamLZ has 5 token types requiring a branch chain:
```
ISETP.GT.U32 P0, R18, 0x17     ; >= 24? (standard match+lit)
@!P0 BRA ...
ISETP.NE    P0, R18, RZ        ; == 0? (long literal)
@!P0 BRA ...
ISETP.NE    P0, R18, 0x1       ; == 1? (long match + off16)
@!P0 BRA ...
ISETP.NE    P0, R18, 0x2       ; == 2? (long match + off32)
@!P0 BRA ...
; else: token 3-23 (short match + off32)
```

LZ4 has one token format — a single shift extracts both fields:
```
SHF.R.U32.HI R27, RZ, 0x4, R19   ; literal_length = token >> 4
; bottom nibble = match_length - 4 (extracted later)
```

### 4. The Circular Problem

Staging would fix mio_throttle, but staging variables (base pointer, loaded count, warp index) need registers. Our 5-type parser already consumes all 48 registers with 32 bytes of spill. Adding staging pushes spill to 40 bytes — the extra LSU traffic from spills exceeds the staging benefit.

nvCOMP's single-type parser uses fewer registers, leaving headroom for the 896-byte staging buffer without spilling.

**The 8% gap is the irreducible cost of a richer token format at the same register budget.**

## ncu Profile Comparison (warp stall samples)

| Stall Reason | StreamLZ | nvCOMP | Ratio |
|-------------|----------|--------|-------|
| long_scoreboard (global mem latency) | 303,228 | 156,802 | 1.93× |
| **mio_throttle (LSU overload)** | **84,384** | **19,222** | **4.39×** |
| wait (syncwarp) | 267,190 | 267,729 | 1.00× |
| branch_resolving | 79,466 | 77,186 | 1.03× |
| short_scoreboard | 42,913 | 57,754 | 0.74× |

## Projected Scaling (SM-count × clock, L1/compute-bound)

| GPU | SMs | Clock | 100 MB | 1 GB+ |
|-----|-----|-------|--------|-------|
| RTX 4060 Ti | 34 | 2535 MHz | 17 GB/s (measured) | ~17 GB/s |
| A100 | 108 | 1410 MHz | ~21 GB/s (35% idle) | ~30 GB/s |
| H100 | 132 | 1830 MHz | ~28 GB/s (71% idle) | ~48 GB/s |
| H200 | 132 | 1980 MHz | ~30 GB/s (71% idle) | ~52 GB/s |

Data center GPUs need ≥340 MB input to fully saturate all SMs at sc_group=0.25.

## Architecture Reference

- nvCOMP LZ4 SASS analysis: `c:\tmp\nvcomp_test\cuda.md`
- ncu profiles: `c:\tmp\slz_profile.ncu-rep`, `c:\tmp\nvcomp_profile.ncu-rep`
- Profiling scripts: `c:\tmp\ncu_profile.bat`, `c:\tmp\ncu_profile_nvcomp.bat`

## vs nvCOMP — measurement methodology

(Migrated from the retired docs/GPU_README.md, 2026-06-10. The number
tables that accompanied this section were 2026-05-27 / sc=0.5-era and
are retired with it — the current headline is enwik9 1 GB:
L1 52.6% @ 24.3 ms d2d vs nvCOMP LZ4 53.6% @ 33 ms; L5 35.50% @
34.2 ms vs nvCOMP Zstd 35.75% @ 50.8 ms. Regenerate per-window
numbers rather than quoting old tables.)

Three measurement windows answer different questions; confusing them
is easy (an early revision compared StreamLZ's narrow inner-kernel
timer against nvCOMP's whole-call event bracket, inflating the win):

| Window | What it measures | StreamLZ source | nvCOMP source |
|--------|------------------|-----------------|---------------|
| **Pipeline kernel-sum** | Σ `cudaEventElapsedTime` across every kernel in the decode pipeline. Excludes memcpys, scheduler gaps, event overhead. | `bench_d2d` "kernel active best" | `nsys stats --report=cuda_gpu_kern_sum` over the timing iterations |
| **Async call wall** | cuEvent pair on the caller's stream around the whole decompress API call. What an async caller actually waits for. | `bench_d2d` "gpu kernel best" | `nvcomp_bench3` "decode kernel best" |
| **End-to-end host wall** | wall around `H2D(compressed) + decompress + D2H(output)` with host input/output. | `streamlz.exe -db` "e2e best" | `nvcomp_bench3` "decode e2e best" |

Ground truth for per-kernel breakdowns: Nsight Systems —

```
nsys profile --trace=cuda --sample=none --output=<rep> <bench.exe> [args]
nsys stats --report=cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum <rep>.nsys-rep
```

nvCOMP LZ4 surfaces one kernel per decompress
(`lz4DecompressBatchKernel`) — its time IS the kernel sum. nvCOMP
Zstd surfaces five; their sum matches nvCOMP's reported "decode
kernel" bracket to within noise, confirming that bracket is
effectively a kernel-sum with no hidden memcpys. StreamLZ surfaces
~8 kernels per L3+ call (walk, prefix-sum, scan, fused compact,
gather, parallel merge, LUT-build + huff-decode, LZ decode); its
async-call wall exceeds its kernel-sum by the front-half D2D of the
compressed block plus per-kernel cuEvent overhead — on-stream but
not in the kernel-sum. (StreamLZ's CLI now prints the per-kernel
table directly via `SLZ_PROFILE_DECODE=1`, so nsys is only needed
for cross-library comparisons.)

Where the wins come from (structure unchanged since 2026-05-27):
the 32-stream Huffman decode with BIL coalesced 128-byte refills
keeps the kernel-sum ahead; the async wall is closer because
StreamLZ runs a longer pipeline; the end-to-end gap is dominated by
StreamLZ's pinned-buffer D2H vs nvCOMP's pageable-host copy.
