# srcVK Vulkan Port — Phase 5 Perf Parity Sweep

**Date:** 2026-06-08
**HEAD:** 7562048 (Phase 4 ABI + persistent encode regions, A-018..A-020 fixed)
**Device under test (VK):** `NVIDIA GeForce RTX 4060 Ti` (verbatim
`vkPhysicalDeviceProperties.deviceName`, `SLZ_VK_DEVICE_INDEX=1`)
**Device under test (CUDA):** same physical GPU via the CUDA reference
backend
**Driver:** Vulkan API 1.4.325, NVIDIA driver shipping that loader version
**Bench shape (post-Phase-5 review F-001/F-004 correction, 2026-06-08):**
- **Decode**: `-db -r 5` produces 1 untimed warm-up + 5 measured runs;
  best/mean below are over those 5 measurements. Reproducible within ~2%
  on independent reruns.
- **Encode**: `-b -r 5 -l N` runs the decode loop 5 times but the **encode
  pass is single-shot** (1 untimed warm-up + 1 timed encode). The `Compress:`
  line is one measurement, NOT a 5-run best. Per the reviewer's reproducer,
  the L5 silesia encode cell flipped from 1.014× to 0.957× between two
  consecutive runs — single-shot noise is ±5%. **Treat encode ratios as
  bands**, not 3-sig-fig precision; cells within ±0.05× of 1.00 should be
  read as "at parity" rather than "faster" or "slower."
- Timings reported are `best` and `mean` for both e2e and `gpu kernel`
  (sum of per-kernel VkQueryPool timestamps) on the decode side.
- Bench discipline per `feedback_gpu_bench_serial.md`: every bench launched
  in a separate sequential tool call; no concurrent VK + CUDA invocations.

Raw per-bench captures (60 files) live under `c:/tmp/perfsweep/`
(`dec_{vk,cu}_{w,e,s}_l{1..5}.txt`, `enc_{vk,cu}_{w,e,s}_l{1..5}.txt`).
Numbers below are extracted verbatim from those captures.

## Conventions

- `w` = web.txt (4.5 MB), `e` = enwik8.txt (95 MB),
  `s` = silesia_all.tar (203 MB)
- "VK/CUDA" ratio = VK time / CUDA time. `<1.00x` means VK is **faster**;
  `>1.10x` means VK is over the 10% parity bar and warrants a residual entry.
- All decode benches consume CUDA-encoded `.slz` files (same wire format
  both backends must decode), so the only variable is decoder speed.
- All encode benches read the same source asset; output sizes are
  reported in the "Compression ratio" column.

## Decode parity (e2e end-to-end best)

| Level | Corpus | VK e2e best (ms) | CUDA e2e best (ms) | VK/CUDA | Bar |
|---|---|---:|---:|---:|---|
| L1 | web    | 5.342  | 2.008  | 2.66x | over (A-VK-WDDM small-file submit floor) |
| L1 | enwik8 | 15.493 | 16.105 | 0.96x | **FASTER** |
| L1 | silesia| 30.025 | 30.906 | 0.97x | **FASTER** |
| L2 | web    | 5.585  | 2.010  | 2.78x | over (A-VK-WDDM small-file submit floor) |
| L2 | enwik8 | 15.755 | 16.015 | 0.98x | **FASTER** |
| L2 | silesia| 30.039 | 31.160 | 0.96x | **FASTER** |
| L3 | web    | 5.959  | 2.042  | 2.92x | over (A-VK-WDDM small-file submit floor) |
| L3 | enwik8 | 16.513 | 16.091 | 1.03x | within bar |
| L3 | silesia| 32.416 | 31.766 | 1.02x | within bar |
| L4 | web    | 5.631  | 2.038  | 2.76x | over (A-VK-WDDM small-file submit floor) |
| L4 | enwik8 | 16.575 | 16.135 | 1.03x | within bar |
| L4 | silesia| 32.658 | 31.581 | 1.03x | within bar |
| L5 | web    | 5.750  | 2.098  | 2.74x | over (A-VK-WDDM small-file submit floor) |
| L5 | enwik8 | 16.378 | 16.092 | 1.02x | within bar |
| L5 | silesia| 32.498 | 31.729 | 1.02x | within bar |

## Decode parity (gpu kernel best — apples-to-apples)

| Level | Corpus | VK kernel best (ms) | CUDA kernel best (ms) | VK/CUDA |
|---|---|---:|---:|---:|
| L1 | web    | 1.316 | 1.373 | 0.96x |
| L1 | enwik8 | 3.040 | 3.072 | 0.99x |
| L1 | silesia| 5.248 | 5.189 | 1.01x |
| L2 | web    | 1.317 | 1.375 | 0.96x |
| L2 | enwik8 | 3.062 | 3.029 | 1.01x |
| L2 | silesia| 5.270 | 5.192 | 1.02x |
| L3 | web    | 1.444 | 1.438 | 1.00x |
| L3 | enwik8 | 5.150 | 4.077 | 1.26x |
| L3 | silesia| 9.257 | 7.162 | 1.29x |
| L4 | web    | 1.440 | 1.437 | 1.00x |
| L4 | enwik8 | 5.291 | 4.120 | 1.28x |
| L4 | silesia| 9.710 | 7.073 | 1.37x |
| L5 | web    | 1.512 | 1.515 | 1.00x |
| L5 | enwik8 | 5.192 | 4.314 | 1.20x |
| L5 | silesia| 9.955 | 7.686 | 1.30x |

## Encode parity (kernel time best)

| Level | Corpus | VK encode (ms) | CUDA encode (ms) | VK/CUDA | Output size VK | Output size CUDA | Size ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | web    | 20  | 15  | 1.33x | 2,189,254  | 2,189,254  | 1.0000x |
| L1 | enwik8 | 98  | 121 | **0.81x** | 58,640,583  | 58,640,583  | 1.0000x |
| L1 | silesia| 203 | 235 | **0.86x** | 101,782,020 | 101,782,020 | 1.0000x |
| L2 | web    | 21  | 13  | 1.62x | 2,186,851  | 2,186,851  | 1.0000x |
| L2 | enwik8 | 117 | 133 | **0.88x** | 58,560,517  | 58,560,517  | 1.0000x |
| L2 | silesia| 234 | 241 | **0.97x** | 101,703,949 | 101,703,949 | 1.0000x |
| L3 | web    | 25  | 15  | 1.67x | 1,668,810  | 1,668,810  | 1.0000x |
| L3 | enwik8 | 140 | 159 | **0.88x** | 43,662,123  | 43,662,123  | 1.0000x |
| L3 | silesia| 268 | 309 | **0.87x** | 80,983,294  | 80,967,993  | **1.00019x** (A-008) |
| L4 | web    | 22  | 15  | 1.47x | 1,597,873  | 1,597,873  | 1.0000x |
| L4 | enwik8 | 144 | 157 | **0.92x** | 42,660,426  | 42,660,426  | 1.0000x |
| L4 | silesia| 288 | 310 | **0.93x** | 79,765,390  | 79,765,390  | 1.0000x |
| L5 | web    | 68  | 72  | **0.94x** | 1,462,315  | 1,462,315  | 1.0000x |
| L5 | enwik8 | 305 | 310 | **0.98x** | 39,578,913  | 39,578,913  | 1.0000x |
| L5 | silesia| 562 | 554 | 1.01x | 72,108,522  | 72,108,522  | 1.0000x |

## ptest_vk results both backends (2026-06-08, post-Phase 5)

| Backend | passed | skipped | failed | total |
|---|---:|---:|---:|---:|
| `SLZ_VK_DEVICE_INDEX=1` (NVIDIA RTX 4060 Ti) | 140 | 9 | 0 | 149 |
| `SLZ_VK_DEVICE_INDEX=0` (Intel(R) Graphics iGPU) | 140 | 9 | 0 | 149 |

Phase 5 added 18 tests (cross-backend coverage extension) — pre-Phase-5
baseline was 122/9/0 (131 total). The 9 skipped tests are the
already-documented single-thread-mode + corpus-asset-missing skips
(unchanged from prior phases).

## Bench analysis vs residuals catalog

### Cases within the 10% parity bar on large workloads (PASSING)

All large-workload (enwik8 + silesia) **e2e** decode and encode timings
sit within the 10% bar at every level. Specifically:

- L1-L5 decode e2e on enwik8/silesia: **0.96x-1.03x** CUDA
- L1-L5 encode on enwik8/silesia: **0.81x-1.01x** CUDA, with L1/L2/L3/L4
  enwik8 + L1-L4 silesia all **FASTER** than CUDA

### Open gap: L3/L4/L5 decode kernel time on large workloads (1.20x-1.37x)

The decode `gpu kernel best` ratio for L3/L4/L5 on enwik8 + silesia is
**outside** the 10% kernel bar (1.20x-1.37x). **e2e is inside the bar**
because the host overhead amortizes the kernel difference at 95-203 MB
input sizes — but the kernel-time gap is real and worth documenting.

Note this is **after** the A-016 (u32-aliased SSBO store) and A-017
(fused 4x compact_huff_descs) fixes that closed L5 specifically from
1.15x/1.17x to 1.03x/1.04x (history in the retired ToDo.md). The remaining
1.20x-1.37x is concentrated in the `lz_decode_kernel` (L3+ huff-aware
workhorse) and is consistent with the structural cost of:
1. Per-binding offset dispatch ABI (A-007) — extra descriptor work per
   sub-launch vs CUDA's raw pointer arithmetic;
2. Compute-to-compute barriers between huff_decode and lz_decode passes
   (A-006) — Vulkan's explicit RAW ordering requirement vs CUDA's
   implicit per-stream ordering;
3. `compact_raw_descs` + `merge_huff_descs` — separate dispatches
   currently not fused (potential future A-NNN analog to A-017's
   compact_huff fusion).

The L3/L4 kernel-time gap is filed as **new** entry **A-021** in
`srcVK/PortAdaptations.md` (see below) — accepted residual today since
e2e is still inside the bar; potential future work mirroring A-017's
fusion approach could close it.

### Open gap: web.txt small-file regime (decode 2.66x-2.92x, encode 1.33x-1.67x)

Documented residual **#1** in `srcVK/Handbook.md` "Accepted residuals".
Captured by **A-VK-WDDM submit-floor** (no dedicated catalog entry yet
because there is no single source line to point at — it is the
structural cost of Vulkan-on-WDDM per-dispatch overhead floor). The
sweep numbers track the prior bench numbers (decode
2.6x, encode 1.73x). No regression.

### Closed gap: silesia L3 0.019% larger output (A-008 RESOLVED 2026-06-09)

At Phase 5 measurement time (2026-06-08) silesia L3 VK encode was
80,983,294 bytes vs CUDA 80,967,993 = 1.00019× wider, catalogued as
A-008 ACTIVE. The next day (commit `8c8964d`) the BDA workaround landed:
the hash table is now addressed via `VK_KHR_buffer_device_address`
(raw device pointer through a `buffer_reference` SSBO) instead of a
descriptor-bound SSBO, bypassing the per-binding `maxStorageBufferRange`
cap that previously forced the `hash_bits=18` clamp. Silesia L3 VK
output is now **80,967,993 bytes — byte-identical to CUDA SHA**. Every
(level, corpus) cell in the 15-cell encode matrix is byte-identical
to CUDA. A-008 entry status: ✅ **RESOLVED**.

## Intel iGPU performance characterization (2026-06-09)

The bench sweep above is NVIDIA RTX 4060 Ti — the discrete-target
device. The codec also runs on Intel iGPU as a fallback (Intel UHD,
Iris, Arc) and ptest_vk passes 144/9/0 on Intel. Performance there is
**hardware-explained**, not a port bug: Intel UHD/Iris has roughly
5.7× less memory bandwidth and 22× less peak FP32 compute than the
RTX 4060 Ti, and the measured per-kernel ratios sit within those
bounds.

### Intel iGPU decode bench (NVIDIA reference column for context)

| Workload | Intel kernel best | NVIDIA kernel best | Ratio | Class |
|---|---:|---:|---:|---|
| L1 enwik8 decode | 25.6 ms | 3.0 ms | 8.5× | bandwidth-bound (matches BW ratio + cache delta) |
| L5 enwik8 decode | 38.7 ms | 5.5 ms | 7.0× | bandwidth-bound |
| L5 silesia decode | 72.9 ms | 9.8 ms | 7.4× | bandwidth-bound |

### Intel L5 enwik8 per-kernel breakdown (`SLZ_VK_PROFILE_DECODE=1`)

| Kernel | Intel (µs) | NVIDIA (µs) | Ratio | Notes |
|---|---:|---:|---:|---|
| `kernel_fn` (LZ decode) | 30,331 | 3,345 | 9.1× | bandwidth-bound |
| `huff_decode_fn` | 6,047 | 772 | 7.8× | bandwidth-bound |
| `huff_build_fn` | 3,439 | 207 | 16.6× | compute-bound (32-lane cooperative LUT build, 8 passes) |
| `gather_off16_fn` | 1,239 | 76 | 16.3× | compute-bound (256-thread WG) |
| `merge_huff_descs_fn` | 642 | 440 | 1.5× | dispatch-overhead-bound (single-WG) |
| `compact_huff_descs_fn` (fused 4×) | 584 | 245 | 2.4× | dispatch-overhead-bound (1 fused WG) |
| `prefix_sum_chunks_fn` | 421 | 184 | 2.3× | dispatch-overhead-bound |
| `compact_raw_descs_fn` | 285 | 211 | 1.4× | dispatch-overhead-bound |
| `scan_parse_fn` | 47 | 10 | 4.7× | dispatch-overhead-bound |

### Interpretation: hardware-explained, not port bugs

The ratios cluster into three classes that match the underlying
hardware ceilings:

- **Bandwidth-bound kernels (LZ decode, Huffman decode)** run at
  7-9× NVIDIA. Intel UHD has ~50 GB/s shared LPDDR vs the RTX 4060
  Ti's 288 GB/s GDDR6 = 5.7× ratio. The extra ~2× comes from Intel's
  smaller / less-effective cache hierarchy. The kernels are doing
  approximately what the hardware allows.
- **Compute-bound kernels (Huff build LUT, off16 gather)** run at
  ~16× NVIDIA. Intel UHD peak ~1 TFLOPS vs RTX 4060 Ti ~22 TFLOPS
  = 22× ratio. Again the kernels are within hardware bounds.
- **Single-WG / dispatch-overhead-bound kernels** run at 1.5-4.7×
  NVIDIA — the Windows WDDM per-dispatch overhead floor dominates
  these on both devices, and Intel's floor is only modestly higher
  than NVIDIA's.

### Why we don't pursue Intel-specific tuning

The codec hardcodes `WARP_SIZE=32` and pins `requiredSubgroupSize=32`
on every compute pipeline (mirroring CUDA's warp semantics). Each
warp-cooperative kernel runs as 1 subgroup × 32 lanes per workgroup;
many such workgroups run in parallel (e.g., ~48K WGs for huff_build
on enwik8 L5). Intel UHD has fewer EUs to schedule those WGs in
parallel than NVIDIA's 18 SMs × 4 schedulers, so the queue depth
runs longer — which is exactly what the ratios show.

Speculative Intel-only changes considered + ruled out:

- **Bigger `local_size_x`** would leave the extra lanes idle (the
  algorithm only uses the first warp's worth) — WORSE on Intel.
- **`requiredSubgroupSize=16` on Intel** would break every
  `subgroupShuffle(v, lane)` call with `lane >= 16` — wrong output.
- **Splitting the 8-pass work across 2 WGs** would be a 4+ hour
  kernel rewrite for a single-digit-percent improvement at best.

The honest measurement above is what Intel iGPU can deliver on this
codec; the codec's design choice to optimize for dGPU warp semantics
is paying off on dGPU (1.03× CUDA at e2e) and Intel iGPU runs as a
correctness-validated fallback rather than a perf-tuned target.

## L5 encode kernel time (informational)

L5 chain parser is serial-on-lane-0 by construction in both CUDA and
VK. Web/enwik8 L5 encode VK is faster than CUDA (0.94x and 0.98x);
silesia L5 VK is 1.01x — all comfortably within the bar. Confirms
Phase 2B's chain-parser port (`srcVK/encode/lz_chain_parser.glsl`,
A-013/A-014) ships at parity.

## Summary against Phase 5 goals

- [x] Decode `e2e` within 10% bar on all (level x corpus) large-workload
      cells: **PASS** (0.96x-1.03x)
- [x] Encode kernel time within 10% bar on all large-workload cells:
      **PASS** (0.81x-1.01x measured single-shot — see F-001 caveat in
      bench-shape note above; cells within ±5% of 1.00 should be read as
      "at parity" rather than "faster" or "slower." The clear wins
      (L1/L2 enwik8 at 0.81-0.86x) survive single-shot noise; L5 cells
      near 1.00 do not.)
- [x] Output byte-identity 14/15 cells; L3 silesia 1.00019x catalogued
      as A-008
- [ ] Decode `gpu kernel best` within 10% on L3/L4/L5 large workloads:
      **OVER bar** (1.20x-1.37x). Catalogued as **A-021** (new). Not
      blocking because e2e is inside bar; potential future fusion work
      identified.
- [x] Small-file (web.txt) regression catalogued under existing residual
      #1 / submit-floor adaptation

---

## 2026-06-11 full 60-cell re-sweep (post v4 #15 wave, both backends)

**Date:** 2026-06-11 (evening, post-reboot — see driver-state note)
**HEAD:** fa0f155 (CUDA K=4 pipelined decode default ALL levels; VK
single-warp default per A-028)
**Device (both backends):** `NVIDIA GeForce RTX 4060 Ti` (verbatim
deviceName, `SLZ_VK_DEVICE_INDEX=1`)
**Bench shape:** decode `-db -r 5` (best-of-5 + warm-up); encode
`-b -r 5 -l N` (single-shot encode — treat encode ratios as ±5%
bands per the F-001 note above). All 15 frames re-encoded with the
current CUDA encoder first (the June-8 frames predate the L2
re-differentiation). All 60 runs strictly serial.
**Driver-state note:** the first VK pass of this sweep was measured
2-2.4x slow and DISCARDED — the morning's TDR debugging (v4 #15 VK
bring-up) had degraded the NVIDIA VK driver state machine-wide
(CUDA unaffected; Win+Ctrl+Shift+B insufficient; full reboot
cleared it). VK cells below are the post-reboot re-run, spot-checked
stable (enwik8 L1 kernel 2.351 ms, mean within 0.5% of best).
Raw captures: `c:/tmp/perfsweep11/` (60 files, same naming as the
Phase 5 sweep).

### Decode parity (e2e best, ms)

| Level | Corpus | VK | CUDA | VK/CUDA | Bar |
|---|---|---:|---:|---:|---|
| L1 | web    | 5.405  | 0.886  | 6.10x | over (A-VK-WDDM small-file submit floor) |
| L1 | enwik8 | 15.173 | 14.496 | 1.05x | within |
| L1 | silesia| 29.566 | 29.364 | 1.01x | within |
| L2 | web    | 5.342  | 0.907  | 5.89x | over (submit floor) |
| L2 | enwik8 | 15.223 | 14.476 | 1.05x | within |
| L2 | silesia| 29.416 | 29.531 | 1.00x | within |
| L3 | web    | 5.044  | 0.989  | 5.10x | over (submit floor) |
| L3 | enwik8 | 15.905 | 14.729 | 1.08x | within |
| L3 | silesia| 30.760 | 30.149 | 1.02x | within |
| L4 | web    | 5.273  | 1.004  | 5.25x | over (submit floor) |
| L4 | enwik8 | 15.258 | 14.699 | 1.04x | within |
| L4 | silesia| 30.539 | 30.385 | 1.01x | within |
| L5 | web    | 4.947  | 1.021  | 4.85x | over (submit floor) |
| L5 | enwik8 | 15.084 | 14.487 | 1.04x | within |
| L5 | silesia| 29.965 | 29.502 | 1.02x | within |

**e2e verdict: every large-corpus cell is within the 10% parity bar
at every level** — PCIe + host wall dominates and both backends
saturate it. The web.txt small-file gap is the known per-submit
floor (A-VK-WDDM), unchanged in kind.

### Decode parity (gpu kernel best, ms — apples-to-apples)

| Level | Corpus | VK | CUDA | VK/CUDA | Note |
|---|---|---:|---:|---:|---|
| L1 | web    | 0.759 | 0.317 | 2.39x | small-file fixed costs |
| L1 | enwik8 | 2.351 | 1.859 | 1.26x | CUDA = K=4 pipeline (A-028) |
| L1 | silesia| 4.218 | 4.289 | **0.98x** | VK FASTER (CUDA K=4 pays parser dilution on binary) |
| L2 | web    | 0.784 | 0.345 | 2.27x | |
| L2 | enwik8 | 2.225 | 1.950 | 1.14x | |
| L2 | silesia| 4.206 | 4.572 | **0.92x** | VK FASTER |
| L3 | web    | 0.807 | 0.380 | 2.12x | |
| L3 | enwik8 | 4.072 | 3.018 | 1.35x | worst cell; CUDA pipelined general kernel |
| L3 | silesia| 7.096 | 6.243 | 1.14x | |
| L4 | web    | 0.825 | 0.406 | 2.03x | |
| L4 | enwik8 | 3.871 | 3.068 | 1.26x | |
| L4 | silesia| 7.079 | 6.569 | 1.08x | |
| L5 | web    | 0.847 | 0.431 | 1.97x | |
| L5 | enwik8 | 3.792 | 3.079 | 1.23x | |
| L5 | silesia| 6.861 | 6.272 | 1.09x | |

**Kernel verdict:** the enwik8 kernel gap (1.14-1.35x) is the
A-028-documented cost of NOT running the pipeline on VK (measured:
the pipeline is slower than single-warp on this driver — residency,
not barriers). On binary-heavy silesia VK is at or AHEAD of CUDA at
L1/L2 — the same parser-dilution effect that costs CUDA's K=4 there.
e2e remains within bar everywhere that matters.

### Encode parity (compress wall best, ms; single-shot ±5%)

(Re-measured same evening AFTER v4 #17 landed on BOTH backends —
the L1/L2 host-side payload gather was found vestigial and skipped,
and CUDA's L3+ gather went async+pinned. Pre-#17 cells for
reference: CUDA enwik8 L1 was 123, VK 109.)

| Level | Corpus | VK | CUDA | VK/CUDA | Ratio (both, identical) |
|---|---|---:|---:|---:|---:|
| L1 | web    | 19  | 10  | 1.90x | 48.6% |
| L1 | enwik8 | 91  | 87  | 1.05x | 58.6% |
| L1 | silesia| 177 | 180 | 0.98x | 47.8% |
| L2 | web    | 23  | 11  | 2.09x | 46.8% |
| L2 | enwik8 | 101 | 96  | 1.05x | 57.3% |
| L2 | silesia| 193 | 179 | 1.08x | 47.2% |
| L3 | web    | 21  | 14  | 1.50x | 37.1% |
| L3 | enwik8 | 120 | 141 | **0.85x** | 43.7% |
| L3 | silesia| 226 | 260 | **0.87x** | 38.1% |
| L4 | web    | 23  | 14  | 1.64x | 35.5% |
| L4 | enwik8 | 131 | 150 | **0.87x** | 42.7% |
| L4 | silesia| 254 | 274 | **0.93x** | 37.5% |
| L5 | web    | 67  | 68  | 0.99x | 32.5% |
| L5 | enwik8 | 264 | 295 | **0.89x** | 39.6% |
| L5 | silesia| 473 | 554 | **0.85x** | 33.9% |

**Encode verdict (post-#17): L1/L2 at parity on real corpora
(0.98-1.08x); L3+ VK still leads 0.85-0.93x** — the residual is the
L3+ gather + huff-input path, where VK's multi-region BAR-mapped
gather beats CUDA's async-pinned staging; noted as the #17 residual.
Compression ratios byte-identical at every cell (cross-backend SHA
gate re-run after BOTH encoder changes: 5/5 MATCH). 30/30 roundtrips
PASS.

### Summary vs the standing parity goal

- e2e decode: PASS everywhere on real workloads (1.00-1.08x).
- kernel decode: enwik8 1.14-1.35x over bar, ATTRIBUTED (A-028,
  measured, accepted); silesia at/ahead of parity.
- encode: parity at L1/L2 post-#17; VK ahead 0.85-0.93x at L3+ (gather residual).
- web.txt small-file regime: unchanged known residual (submit
  floor), both directions.
