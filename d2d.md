# Pure-D2D Pipeline — Handoff Notes

Status: post-compact recovery doc for the StreamLZ GPU library's true device-to-device
compress/decompress goal. Pick this up cold.

## The goal (non-negotiable)

`slzCompress(d_input) → d_output` and `slzDecompress(d_frame) → d_output`.

- **Zero host buffers of payload bytes.** No malloc/D2H/H2D of compressed or decoded data.
- **Zero CPU loops over frame data.** Every parse, every loop, every `if` runs on GPU.
- The CPU's only role is `cuLaunchKernel` (and tiny launch-plumbing D2Hs for grid sizes
  — see "What counts as 'work'" below).

The user has been emphatic and repeated: **"0.0000000000000000% host involvement"**,
**"the GPU doing ABSOLUTELY 100% of the work"**. Do not deliver another "almost D2D".

## What counts as "work" vs. "plumbing"

| Action | Verdict |
|---|---|
| CPU calling `cuLaunchKernel` | Plumbing — fine. CUDA API has no other way. |
| CPU D2H of 4 bytes (a count) to set the next kernel's grid size | Plumbing — fine *for now*; the user may push for self-gating kernels later (Option A below). |
| CPU loop over `chunk_descs` to compute prefix sum | **Work — NOT allowed.** |
| CPU `append` loop merging 4 huff-desc arrays | **Work — NOT allowed.** |
| CPU `memcpy` splicing M2 assembled bytes into host frame | **Work — NOT allowed.** |
| Buffer sizing from host-known params (`output_capacity / 65536`) | Fine — one-time setup, no per-data CPU work. The frame header has `content_size` (i.e. the original filesize) so any upper bound the caller doesn't pass can be sized from that. |

## Current state — what's committed and what's not

### Decompress side — committed work (commits, in order):

- **`3f46651`** — Decode-output D2D: `fullGpuLaunchImpl` D2D-copies decoded bytes from
  `self.d_output` to caller's `d_output` instead of D2H to host. `d_output_target: ?u64`
  added to the signature.
- **`8dd0381`** — TRUE D2D decompress (initial cut). New kernel `slzWalkFrameKernel`
  (lz_kernels.cuh) parses the entire compressed frame on GPU, produces chunk_descs +
  meta in device memory. `gpuWalkFrameImpl` driver fn. `d_compressed_src: ?u64` param
  on `fullGpuLaunchImpl` — when set, D2D-copies frame from device + forces GPU scan
  (CPU `scanForTansChunks` would segfault on the sentinel host slice).
- **`0d3ba57`** — Encode input bounce removed: `slzCompress` no longer allocates
  `host_in` or D2H's d_input. Sentinel host slice (ptr 0x10, real len) handed down so
  any accidental host-side read of input segfaults loud. `d_input_override` in
  `EncodeContext` carries the device pointer; M2's `gpuCompressImpl` D2D-copies from
  it instead of H2D from host. Two fast_framed fallback paths (uncompressed-chunk,
  SC tail-prefix) gate on `d_input_override` and call `gpu_enc.copyDeviceToHost` for
  the small bytes they need.
- **`02405c2`** — Pure-D2D foundations: `gpuWalkFrameImpl` no longer D2Hs anything
  (returns `WalkFrameResultDev { d_chunk_descs, d_meta }`). Walk kernel now emits
  **FRAME-ABSOLUTE** `src_offset`s (= `pos`, not `pos - block_payload_start`) so
  downstream kernels can read `d_frame` directly. Three new kernels added but not
  yet wired:
    - `slzPrefixSumChunksKernel` — device prefix-sum of per-chunk sub-chunk counts.
    - `slzCompactHuffDescsKernel` — single-threaded compaction of one staged huff
      stream (run 4× for lit/tok/hi/lo).
    - `slzCompactRawDescsKernel` — interleaves hi/lo raw-off16 in sub-chunk order.
    - `slzMergeHuffDescsKernel` — combines 4 compacted huff arrays into one merged
      array with region offsets + sequential `lut_offset`s.
- **`c78ff04`** — Prefix-sum kernel sig refined: takes `n_chunks` + `sub_chunk_cap`
  as launch args (not device pointers) — both are host-known at every call site.
- **`1867aff`** — Prefix-sum kernel **wired in `fullGpuLaunchImpl`**. CPU
  `first_subchunk_idx` loop is GONE. 4-byte D2H of `total_subchunks` after the
  kernel for host-side buffer sizing. `self.d_first_subchunk_idx` is aliased to
  `self.d_first_sub_idx_persist`. For sc>=1 multi-sub-chunk modes, `d_first_sub_idx`
  is D2H'd to host (~few KB) for the pipeline branch.

### Encode side — committed work (commits, in order):

- **`dcd4b87`** — Encode input D2D: `EncodeContext.d_input_override`. When set,
  `gpuCompressImpl` D2D-copies from it into `d_input_persist` (added
  `cuMemcpyDtoDAsync_fn` load in encode driver).
- (Plus `0d3ba57` above — the host_in alloc + D2H rip-out.)

### What still has CPU work (the remaining target):

**Decompress side:**
1. **`gpuScanChunks` (driver.zig:1232)** — runs scan kernel, then **D2Hs the staged
   arrays + CPU-compacts** into `huff_*_host_buf` host arrays. The compact kernels
   exist (`slzCompactHuffDescsKernel`, `slzCompactRawDescsKernel`) but aren't wired.
2. **`fullGpuLaunchImpl`'s `append` loop (driver.zig ~1699)** — CPU loop that merges
   4 host huff arrays into `merged_huff` (host), then H2Ds to `self.d_huff_descs`.
   `slzMergeHuffDescsKernel` exists but isn't wired.

**Encode side:**
3. **`gpuAssembleFrameImpl` (encode/driver.zig:1181)** — at the end, D2Hs the
   assembled sub-chunk blocks to `assembled_data` (host). `fast_framed`'s gpu_splice
   then `@memcpy`s those host bytes into host `dst`, and `slzCompress`'s wrapper
   H2Ds the full frame back to `d_output`. This is the entire encode-output bounce
   — 20MB for a 100MB input.
4. **`fast_framed.gpu_compress` block** — many CPU loops compute frame structure,
   write headers, splice payloads. To go pure-D2D, the entire frame must be built
   on device (new "frame-assemble-on-device" kernel).

## File map

| File | Role | Key lines |
|---|---|---|
| `src/gpu/decode/lz_kernels.cuh` | Decode kernels (LZ + walk + prefix + compact + merge + scan) | Compact/merge: ~675–800; prefix-sum: ~810–840; scan: ~939+; walk: ~521 |
| `src/gpu/decode/lz_kernel.ptx` | Built artifact (rebuild via `tools\build_gpu.bat`) | — |
| `src/gpu/decode/driver.zig` | Decode driver | walk fn: ~1110; prefix-sum fn: ~1190; gpuScanChunks: ~1232; fullGpuLaunchImpl: ~1321; CPU prefix-sum (already replaced): ~1442; CPU merge (`append`): ~1682–1702 |
| `src/gpu/decode/huffman_kernel.cu` | Huffman build + decode kernels | `slzHuffBuildLutKernel`, `slzHuffDecode4StreamKernel` |
| `src/decode/streamlz_decoder.zig` | Frame-level orchestration | `decompressFramedFromDevice`: ~102; `decompressFramedParallelToDevice`: ~143; `decompressFramedInner`: ~163 |
| `src/streamlz_gpu.zig` | C ABI library | `slzDecompress`: ~423; tiered jobs (`DecompressJobTrueD2D` → `DecompressJobD2D` → `DecompressJob`) |
| `src/gpu/encode/driver.zig` | Encode driver | `gpuCompressImpl`: ~1417; `gpuAssembleFrameImpl` (M2): ~1181; `EncodeContext`: ~403; `d_input_override`: ~408 |
| `src/encode/fast_framed.zig` | Encode pipeline | `gpu_compress:` block: ~1096; `gpu_splice`: ~1387 |
| `tools/huff_test/scan_test.cu` | Phase 2 scan-kernel verification harness | Use as template for any new harness |
| `tools/build_gpu.bat` | Rebuild all decode kernel PTX | Run via PowerShell `cmd.exe /c` |
| `tools/build_gpu_enc.bat` | Rebuild encode kernel PTX (lz + M2 assemble) | — |
| `tools/build_smoke_dev.bat` | Rebuild `slz_gpu_smoke_dev.exe` (D2D library smoke test) | Links against `streamlz_gpu.dll` |

## Key constants and ABI

- `walk_max_chunks = 16384` (driver.zig). Max chunk_descs the walk kernel will emit.
  Used everywhere as upper bound for over-allocated buffers.
- `HUFF_LUT_ENTRIES = 1024` (driver.zig:305). `lut_offset = slot * 1024`.
- `NUM_PIPELINE_STREAMS = 1` (driver.zig:109). Pipeline branch is 1-stream — for
  sc=0.25 (every -gpu frame), `first_subchunk_idx_buf[0] = 0` is the only host read
  the pipeline branch makes.
- `per_subchunk_scratch = 131072` (driver.zig:1465). Sub-chunk slot stride for the
  tans scratch buffer. `tok_region_off = total_subchunks * 131072`,
  `off16_region_off = total_subchunks * 2 * 131072`.
- Walk meta layout (`d_meta`, 6 × u32):
  - +0 `n_chunks`
  - +4 `decomp_size`
  - +8 `sub_chunk_cap`
  - +12 `block_start`
  - +16 `block_size`
  - +20 `status` (0 = success; see `slzWalkFrameKernel` for codes)
- `WalkFrameResultDev = struct { d_chunk_descs: u64, d_meta: u64 }`.
- `walkMetaToHost(d_meta) → ?WalkMeta` — explicit legacy bridge; pure-D2D doesn't call it.
- ABI-matched device structs (kernel ↔ Zig):
  - `SlzChunkDesc` (24 B): `src_offset, comp_size, decomp_size, dst_offset, flags, memset_fill, _pad[3]`.
    Note: walk kernel emits **frame-absolute** `src_offset` (= `pos`, NOT
    `pos - block_payload_start`). gpuBatchDecode (legacy CPU walk) still emits
    block-relative — fullGpuLaunchImpl is offset-agnostic.
  - `SlzHuffDecChunkDesc` (20 B): `in_offset, in_size, out_offset, out_size, lut_offset`.
  - `SlzScanHuffDesc` (20 B): same layout, last field is `valid` (0/1).
  - `SlzScanRawDesc` (16 B): `src_offset, size, gpu_offset, valid`.
  - `SlzRawOff16Desc` (12 B): `src_offset, size, gpu_offset`.

## Existing pure-D2D entry path (slzDecompress)

`slzDecompress` (streamlz_gpu.zig:423) tries 3 paths in order:
1. **`DecompressJobTrueD2D`** — calls `decompressFramedFromDevice` (streamlz_decoder.zig:102).
   That fn runs walk → `walkMetaToHost` → host-allocate chunk_descs + D2H from
   `d_chunk_descs` → `fullGpuLaunchImpl(..., d_output_target=d_output, d_compressed_src=d_frame)`.
   This is the **path you're upgrading**.
2. `DecompressJobD2D` — D2Hs frame to host, then `decompressCoreD2D` (legacy half-D2D).
3. `DecompressJob` (host-bounce) — full host fallback.

Tiered fallback on `error.BadMode`. Keep the tiered structure — pure-D2D might
hit unsupported frame shapes (dict, PDM, multi-block, checksums).

## What's left, step-by-step

### Step 6b — Scan compaction on device

**Goal:** `gpuScanChunks` produces device-resident compacted huff/raw arrays,
D2Hs only the 5 counts (20 B total).

1. Add to `DecodeContext`:
   ```zig
   d_compact_lit: CUdeviceptr = 0, d_compact_lit_size: usize = 0,
   d_compact_tok: CUdeviceptr = 0, d_compact_tok_size: usize = 0,
   d_compact_hi:  CUdeviceptr = 0, d_compact_hi_size:  usize = 0,
   d_compact_lo:  CUdeviceptr = 0, d_compact_lo_size:  usize = 0,
   d_compact_raw: CUdeviceptr = 0, d_compact_raw_size: usize = 0,
   d_compact_counts: CUdeviceptr = 0, d_compact_counts_size: usize = 0, // 5 u32
   ```
   Size each huff buf to `walk_max_chunks * @sizeOf(HuffDecChunkDesc)` (320 KB).
   Raw buf to `walk_max_chunks * 2 * @sizeOf(RawOff16Desc)` (384 KB).
   `d_compact_counts` is 20 B (5 u32: n_lit, n_tok, n_hi, n_lo, n_raw).

2. In `gpuScanChunks` (driver.zig:1232), after the scan kernel launch (and `sync()`),
   replace the entire D2H + CPU compaction block (lines 1294–1344) with kernel
   launches:
   - 4× `slzCompactHuffDescsKernel(staged_X, &d_compact_counts+0, d_compact_X, &d_compact_counts+offset)`.
     Each: `launch(compact_huff_descs_fn, 1, 1, 1, 1, 1, 1, 0, 0, params, extra)`.
     Args: `d_staged_X` (offset into `self.d_scan_staged`), `d_total_subchunks_buf`
     (the prefix-sum kernel's output), `d_compact_X`, `d_compact_counts + (stream_idx * 4)`.
   - 1× `slzCompactRawDescsKernel(d_staged_hi, d_staged_lo, d_total_subchunks_buf,
     d_compact_raw, d_compact_counts + 16)`.
   - Sync.
   - D2H `d_compact_counts` (20 B) into a `[5]u32` on host.
3. Populate `ScanResult` with the 5 counts. The legacy code still expects the
   host arrays (`huff_*_host_buf`) to be filled — for now, also D2H each compacted
   array into them (small — `n_huff * 20 B` per stream, max ~80 KB total per call).
   The follow-up (step 6c) eliminates this when the merge moves to GPU.

   *Alternative cleaner path:* skip the host-array D2H entirely; populate
   `ScanResult` with device pointers + counts. Then step 6c can consume those.
   Choose this if you can do 6b + 6c in one push — otherwise the intermediate
   commit has dead D2Hs that work but aren't progress.

### Step 6c — Merge on device

**Goal:** replace `fullGpuLaunchImpl`'s CPU `append` loop (driver.zig:1682–1702)
with `slzMergeHuffDescsKernel`. `self.d_huff_descs` is populated by the merge
kernel directly — no H2D.

1. The merge kernel signature (lz_kernels.cuh:~755):
   ```
   slzMergeHuffDescsKernel(
       d_lit, d_tok, d_hi, d_lo,           // 4 compacted device arrays
       d_n_lit, d_n_tok, d_n_hi, d_n_lo,   // 4 device count pointers (single u32 each)
       tok_region_off, off16_region_off,   // host u32 args
       d_merged,                            // output device array
       d_n_merged)                          // output device count
   ```
2. Replace the `if (have_huff) { ... append ... H2D merged_huff ... }` block.
   New flow:
   - Ensure `self.d_huff_descs` sized for `walk_max_chunks * 4 * 20`.
   - Ensure `d_compact_counts` already populated by step 6b.
   - Compute `tok_region_off = total_subchunks * 131072`, `off16_region_off =
     2 * total_subchunks * 131072` (from the `total_subchunks` D2H you already
     have post-prefix-sum at line ~1450 area).
   - Launch merge kernel with the 4 compact-count device pointers
     (`d_compact_counts + 0/4/8/12`).
   - D2H `d_n_merged` (a single u32) to set the grid for downstream huff
     build/decode kernels.
3. Remove the H2D at the end of the merge block. `self.d_huff_descs` is the
   merge kernel's output — already device-resident.
4. Downstream huff build/decode kernels read `self.d_huff_descs` and use
   `n_merged` (host) for their grid sizes. Already correct — just plumb the
   D2H'd count into the existing grid expressions.

### Backward compat: scanForTansChunks (CPU scan) path

`fullGpuLaunchImpl` uses `gpuScanChunks` (GPU) OR `scanForTansChunks` (CPU) at
~line 1525 (`want_gpu_scan` check). CPU scan fills `huff_*_host_buf` host arrays.

After 6b + 6c, the merge kernel reads device-resident compacted arrays. For the
CPU-scan path, you have two choices:
- **Keep the CPU merge for CPU-scan** (simpler): gate the merge-kernel use on
  `want_gpu_scan` — when CPU scan ran, fall back to the existing `append` loop +
  H2D.
- **Always merge on GPU**: H2D the `huff_*_host_buf` arrays to the compact buffers,
  then launch merge kernel. More uniform, slightly more H2D for the CPU-scan path.

Choose the first — less invasive, the CPU-scan path is a fallback anyway.

### Step 7 — Audit and tighten launch counts

After 6b + 6c, the residual host-side D2Hs are launch-plumbing:
- 24 B walk meta (after walk).
- 4 B total_subchunks (after prefix-sum).
- 20 B compact counts (after compact).
- 4 B n_merged (after merge).
- 4 B decomp_size (final — for `out_size.* = ...` return).
- Total: **~56 B of launch-plumbing per decompress.**

For pure-pure 0% data flow (the user may push for this — "Option A" below): each
existing decode kernel takes a `n` launch arg. Modify them to read `d_n` from
device, over-launch with max grid, kernels self-gate via `if (blockIdx.x >= *d_n) return;`.

Affected kernels:
- `slzScanParseKernel` (lz_kernels.cuh) — grid currently `(n_chunks+255)/256`.
- `slzGatherRawOff16Kernel` — grid `n_raw_off16`.
- `slzHuffBuildLutKernel`, `slzHuffDecode4StreamKernel` (huffman_kernel.cu) — grid `n_merged`.
- `slzLzDecodeKernel` (lz_kernels.cuh:~939) — grid `(num_groups, ...)`.

Pattern per kernel:
```cpp
extern "C" __global__ void slzXxxKernel(..., const uint32_t* __restrict__ d_n, ...) {
    if (blockIdx.x >= *d_n) return;
    ...
}
```
Existing callers H2D their `n` into a small device counter buf (one-time per
call) and pass the device pointer. This keeps backward compat — the kernel still
ignores out-of-range blocks.

### Step 8 — Encode side (after decode is done)

Same approach. The encode pipeline is structurally different:
- `gpuAssembleFrameImpl` (M2) currently D2Hs `assembled_data` to host. **Stop the D2H** —
  keep on `d_asm_out`.
- Write a new **frame-assemble-on-device** kernel that takes:
  - `d_asm_out` (assembled sub-chunk blocks, device).
  - `assembled_offsets[]`, `assembled_sizes[]` (device — or compute device-side from
    M2's `d_asm_sizes`).
  - Frame metadata (computable host-side: level, content_size, sc_group, etc.).
  - `d_input` (for SC tail prefix bytes — first 8 of each chunk).
  - `d_output` (caller's device buffer).
  - Writes the full frame: frame header + block header + per-chunk (internal_hdr +
    chunk_hdr + sub_chunk_block) + SC tail + end mark.
- Replace the `fast_framed.gpu_compress` block's host frame-build with a single
  launch of this kernel.
- `slzCompress` wrapper: no more `host_out` alloc, no H2D of frame.

The frame layout details to bake into the kernel:
```
[frame_hdr 22B (no dict)][frame_block_hdr 8B]
  per chunk i (1..n_chunks):
    [internal_block_hdr 2B][chunk_hdr 4B (LE, encodes comp_size-1)][assembled_block_3B_hdr_plus_payload]
[SC tail (n_chunks-1) * 8B][end_mark 4B (zeros)]
```
Per-chunk dst-offset = `22 + 8 + sum_{j<i} (2 + 4 + assembled_sizes[j])`. Host
computes the prefix sum once host-side (M2 already does this for `assembled_offsets`).

## Build + verification

### Build commands
```sh
# Decode/encode kernels → PTX (run via PowerShell when bash interleave is iffy):
cmd.exe /c "tools\build_gpu.bat"        # rebuilds lz_kernel.ptx + huffman_kernel.ptx
cmd.exe /c "tools\build_gpu_enc.bat"    # rebuilds encode kernels (lz + M2 assemble)

# Zig build:
zig build -Doptimize=ReleaseFast -Dgpu=true              # streamlz.exe etc.
zig build gpulib -Doptimize=ReleaseFast -Dgpu=true       # streamlz_gpu.dll

# D2D smoke test rebuild (if .c source changed; otherwise just relink against new DLL):
cmd.exe /c "tools\build_smoke_dev.bat"
```

### Verification matrix

1. **D2D smoke**: `zig-out/bin/slz_gpu_smoke_dev.exe`. 1 MB compressible input
   (alphabet repeating). Roundtrip must say OK.

2. **Regression matrix** (legacy host-output decode — must not regress):
   ```sh
   for f in enwik8_L1 enwik8_L3 enwik8_L5 silesia_L3 silesia_L5; do
     case $f in enwik8*) src=assets/enwik8.txt;; silesia*) src=assets/silesia_all.tar;; esac
     zig-out/bin/streamlz.exe -d -t 1 "c:/tmp/m2/${f}_gpu.slz" -o "c:/tmp/m2/${f}_check.dec" >/dev/null 2>&1
     cmp -s "$src" "c:/tmp/m2/${f}_check.dec" && echo "$f PASS" || echo "$f FAIL"
   done
   ```
   These frames are produced by `streamlz -l N -gpu <input> -o c:/tmp/m2/<name>_LN_gpu.slz`
   for N in 1,3,5.

3. **Diagnostics**: `SLZ_E2E_TIMER=1 ...` enables:
   - `[gpu-init]` (one-time CUDA init, in driver.zig:122 area).
   - `[walk] n_chunks=... decomp=... status=...` (in walkMetaToHost — currently the
     legacy bridge prints; pure-D2D path won't have this unless you add it).
   - `[dec]` (CLI phase breakdown in cli.zig).
   - `[e2e]` (fullGpuLaunchImpl phase breakdown).

## Pitfalls + gotchas (learned the hard way)

1. **Forward declarations in `lz_kernels.cuh`**: `SlzScanHuffDesc` /
   `SlzScanRawDesc` are defined at the **top** of the compact section (~line 685)
   so the compact kernels can reference them. The scan kernel section uses them
   later. Don't move the struct definitions back down.

2. **Frame-absolute offsets in walk kernel**: chunk_descs.src_offset = `pos`
   (frame-absolute). `decompressFramedFromDevice` passes `d_compressed_src = d_frame`
   (NOT `d_frame + block_start`) and `compressed_block.len = block_start + block_size`
   so the D2D-copied region covers the offset range. If you change the walk to
   block-relative, update both call sites.

3. **Sentinel host slices**: `slzCompress` hands fast_framed a `[]const u8` with
   `.ptr = @ptrFromInt(0x10)`, real `.len`. Any accidental host-side read
   segfaults loud. `slzDecompress` doesn't currently use sentinels because the
   pipeline still D2Hs frame bytes (the host_frame fallback); when you finish
   pure-D2D, you'll be able to sentinel-ize the wrapper input slice too.

4. **`fullGpuLaunchImpl`'s `if (compressed_block.len > 0)` block** at ~1417:
   when `d_compressed_src` is set, this branch does the D2D copy; else H2D.
   The outer `if` must wrap a `{}` block (Zig requires braces for multi-statement
   bodies). I had a bug here where the brace-less `if` ate the next statement.

5. **`scanForTansChunks` reads `compressed_block` host bytes**. In pure-D2D mode
   with a sentinel slice, that's a segfault. `fullGpuLaunchImpl` forces
   `want_gpu_scan = true` when `d_compressed_src != null` (line ~1525). Don't
   undo that.

6. **`first_subchunk_idx_buf` host mirror** (driver.zig ~1444): I left a
   zero-initialized `[16384]u32` because the pipeline branch (NUM_PIPELINE_STREAMS=1)
   only reads index 0 (= 0). If you ever bump `NUM_PIPELINE_STREAMS`, you need a
   small selective D2H of group boundaries from `d_first_sub_idx_persist`.

7. **`gpuPrefixSumChunksImpl` signature change** (commit `c78ff04`): takes
   `n_chunks: u32` and `sub_chunk_cap: u32` as host args, NOT device pointers.
   If you re-add device pointers for self-gating, change both the Zig fn and
   the kernel together.

8. **`HUFF_LUT_ENTRIES` is hardcoded as `1024u` in compact kernels.** Match Zig
   `HUFF_LUT_ENTRIES = 1024` constant. If you ever change the Huffman LUT size,
   update both.

9. **`walk_max_chunks = 16384`** is the over-allocation cap for the walk kernel.
   For inputs bigger than `16384 * sub_chunk_cap` (= 16384 × 64 KB = 1 GB),
   the walk kernel returns status=11 (buffer overflow). Bump if needed.

10. **`cuMemcpyDtoDAsync_fn`** is loaded by **both** decode and encode drivers
    (encode loads via `cuMemcpyDtoDAsync_v2` symbol name — added in commit `dcd4b87`).
    Don't duplicate the loader.

11. **Counter buffer sizing**: `d_total_subchunks_buf` etc. are single u32 (4 B).
    `ensureDeviceBuf` is fine — it never shrinks; min size is whatever CUDA gives
    (typically page-aligned, but the 4 B is what you'll read).

12. **Existing decode kernels' grid args** are u32 launch params. If you do the
    Option A self-gating sweep, the kernel signature changes; update every
    launch site in `fullGpuLaunchImpl` (and any pipeline-branch launches).

13. **For 100 MB enwik8 L3 frame**: `n_chunks=1526`, `block_start=30`, `block_size`
    varies (~20MB compressed). Per-decode walk kernel output is consistent. Useful
    sanity values when single-stepping.

14. **`std.time.nanoTimestamp` doesn't exist in Zig 0.16.** Use the QPC helpers
    in `src/gpu/decode/driver.zig`: `gpu_driver.qpcNow()` / `gpu_driver.qpcMs(a, b)`.
    Already loaded in cli.zig for the `[dec]` timer.

15. **`@ptrFromInt(0x10)`** for sentinel host slices needs `[*]const u8`. Slicing
    is fine; any dereference segfaults — that's intentional.

## What to do FIRST when picking this back up

1. `git log --oneline | head -15` — confirm last commit is `1867aff` (or later).
2. `zig build -Doptimize=ReleaseFast -Dgpu=true && zig-out/bin/slz_gpu_smoke_dev.exe`
   — confirm D2D smoke still passes.
3. Read `gpuScanChunks` (driver.zig:1232) and `fullGpuLaunchImpl`'s merge block
   (driver.zig ~1682) — they're what step 6b + 6c modify.
4. Decide: do 6b + 6c **together** (cleaner, no dead intermediate D2Hs) or
   separately (lower-risk per commit). Either works.
5. Run regression matrix after every change. The matrix is fast (~1 min per file).
6. When pure-D2D decompress lands, mark `gpuScanChunks`'s host-array params as
   `_` (unused) — they're vestigial then. Or remove the params and update callers.
7. Encode side comes after decode is done. Don't start encode while decompress is
   half-converted.
