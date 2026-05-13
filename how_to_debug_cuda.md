# CUDA Debugging & Profiling Guide for StreamLZ GPU

## Tool Locations

- **nvcc**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe`
- **cuobjdump**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cuobjdump.exe`
- **nvdisasm**: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvdisasm.exe`
- **NCU (Nsight Compute)**: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat`
- **NCU UI**: `C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu-ui.bat`
- **cl.exe (MSVC)**: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64`

## Building CUDA Code

nvcc requires cl.exe in PATH. Always set PATH first:

```powershell
$env:PATH = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;$env:PATH"
```

### Build PTX (for embedding in Zig via @embedFile):
```powershell
nvcc -ptx -arch=sm_89 -O3 src\decode\fast\gpu_tans_decode_kernel.cu -o src\decode\fast\gpu_tans_decode_kernel.ptx
```

### Build cubin (for SASS disassembly / resource analysis):
```powershell
nvcc -cubin -arch=sm_89 -O3 src\decode\fast\gpu_tans_decode_kernel.cu -o c:\tmp\kernel.cubin
```

### Build standalone benchmark .exe:
```powershell
nvcc -o c:\tmp\bench.exe c:\tmp\bench.cu -arch=sm_89 -O2 --std=c++17
```

### IMPORTANT: After rebuilding PTX, clear Zig cache:
```powershell
Remove-Item .zig-cache -Recurse -Force -Confirm:$false
zig build -Doptimize=ReleaseFast -Dgpu=true
```
Zig caches the `@embedFile` result. If only the PTX changed, Zig may use the old cached binary.

## Profiling with NCU

### Profile batch file template:
```batch
@echo off
set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat"
set SLZ="C:\Users\james.JAMESWORK2025\Repos\StreamLZ_native\zig-out\bin\streamlz.exe"
set INPUT=c:\tmp\test_file.slz
set REPORT=c:\tmp\profile_name

echo Profiling kernel_name
%NCU% --set full --kernel-name KERNEL_NAME --launch-count 1 -f -o %REPORT% %SLZ% -d -t 1 %INPUT% -o c:\tmp\prof_out.bin

echo Report: %REPORT%.ncu-rep
pause
```

### Kernel names:
- `slzTansBuildTablesKernel` — tANS table parse + LUT build
- `slzTans32DecodeKernel` — 32-lane parallel tANS decode
- `slzTansDecodeKernel` — 5-state serial tANS decode (legacy)
- `slzFullDecompressL1Kernel` — LZ decode
- `slzTansParseTablesKernel` — parse-only (split kernel experiment)
- `slzTansInitLutKernel` — LUT-only (split kernel experiment)

### Reading NCU reports from command line:
```powershell
# Key metrics summary
& ncu.bat --import report.ncu-rep --csv --page details 2>&1 | Select-String "Duration|Memory Throughput|Compute.*Throughput|SM Busy|Occupancy|Warp Cycles|Grid Size|REG|Issue Slots|Ipc"

# Full raw metrics
& ncu.bat --import report.ncu-rep --csv --page raw 2>&1
```

### Key metrics to look for:
| Metric | Good | Bad | Meaning |
|--------|------|-----|---------|
| Duration | < 1ms | > 2ms | Total kernel wall time |
| SM Busy | > 60% | < 30% | Compute utilization |
| Mem Busy | < 50% | > 80% | Memory bottleneck |
| L1 Hit Rate | > 80% | < 50% | Cache efficiency |
| Occupancy | > 70% | < 40% | Warp scheduling |
| STL count | 0 | > 100 | Register spills (local store) |

### Environment variables for profiling:
- `SLZ_NO_VK=1` — skip Vulkan init (avoids NCU conflicts)
- `SLZ_BUILD_TIMING=1` — enable per-stream timing in build kernel
- `SLZ_DUMP_TANS=1` — dump tANS descriptors + compressed data to c:\tmp

## SASS / ASM Analysis

### Dump our kernel's SASS:
```powershell
cuobjdump --dump-sass c:\tmp\kernel.cubin > c:\tmp\our_sass.txt
```

### Check register usage / spills:
```powershell
cuobjdump --dump-resource-usage c:\tmp\kernel.cubin
```
Key fields: `REG` (register count), `STACK` (spill bytes), `SHARED` (shared mem bytes)

### Count instruction types:
```powershell
$sass = Get-Content c:\tmp\our_sass.txt
# Find kernel boundaries
# Then count specific instructions:
($sass | Select-String "STL ").Count   # local store = SPILL (bad)
($sass | Select-String "LDL ").Count   # local load = spill reload (bad)
($sass | Select-String "STG").Count    # global store
($sass | Select-String "LDG").Count    # global load
($sass | Select-String "STS ").Count   # shared store
($sass | Select-String "LDS ").Count   # shared load
($sass | Select-String "SHFL").Count   # warp shuffle
($sass | Select-String "ATOM").Count   # atomic operations
```

### nvCOMP SASS analysis:

nvCOMP DLL location:
```
c:\tmp\nvcomp\nvcomp-windows-x86_64-5.2.0.10_cuda13-archive\bin\nvcomp64_5.dll
```

Extract cubins from nvCOMP DLL:
```powershell
# List all cubins
cuobjdump --list-elf nvcomp64_5.dll

# Extract specific sm_89 cubin (find the right index first)
cuobjdump --extract-elf nvcomp64_5.276.sm_89.cubin nvcomp64_5.dll

# Dump SASS
cuobjdump --dump-sass nvcomp64_5.276.sm_89.cubin > c:\tmp\nvcomp_zstd_sass.txt

# Check resource usage
cuobjdump --dump-resource-usage nvcomp64_5.276.sm_89.cubin
```

**nvCOMP cubin index for zstd kernels**: File **276** (sm_89) contains:
- `init_fse_tables` — FSE table initialization (0.4ms, 64 regs, 2720B shared)
- `classify_frames` — frame classification
- `init_huff_tables` — Huffman table init
- `gather_frame_blocks` — block gathering
- zstd decompression kernel

**nvCOMP init_fse_tables profile** (from `c:\tmp\profile_zstd_only.ncu-rep`):
- Duration: 395.84 μs
- Grid: (763, 1, 1), Block: (64, 1, 1)
- REG: 64, STACK: 0, SHARED: 2720
- STL: 0, LDL: 0, STG: 1, LDG: 2, SHFL: 24, ATOM: 83
- Key insight: uses warp-cooperative atomic allocation, NOT sequential LUT fill

**init_fse_tables ASM location in nvcomp_zstd_sass.txt**: lines ~7544-13899

## Existing SASS dumps:
- `c:\tmp\nvcomp_zstd_sass.txt` — nvCOMP zstd sm_89 SASS (all kernels)
- `c:\tmp\our_build_sass.txt` — our build kernel SASS (may be stale)

## Existing NCU profiles:
- `c:\tmp\profile_tans32.ncu-rep` — 32-lane tANS decode
- `c:\tmp\profile_slz_lz_t32.ncu-rep` — LZ decode with tANS32 input
- `c:\tmp\profile_tans_build.ncu-rep` — tANS build kernel
- `c:\tmp\profile_zstd_only.ncu-rep` — nvCOMP zstd (init_fse + decomp)

## Existing batch files:
- `c:\tmp\profile_tans32.bat` — profile 32-lane tANS decode
- `c:\tmp\profile_slz_lz_t32.bat` — profile LZ kernel
- `c:\tmp\profile_tans_build.bat` — profile build kernel
- `c:\tmp\build_bench3.bat` — build nvcomp benchmark

## Existing test files:
- `c:\tmp\test_buildlut5.slz` — L5 GPU with Golomb-Rice table + 32-lane tANS (39.2%, byte-exact)
- `c:\tmp\test_baseline2.slz` — L5 GPU baseline with 5-state tANS (38.6%, byte-exact)

## GPU Info
- **GPU**: NVIDIA GeForce RTX 4060 Ti (34 SMs, sm_89, 16GB)
- **Driver**: 591.86
- **CUDA**: 13.1
- **SM freq**: ~2.23 GHz
- **Max threads/SM**: 1536 (48 warps × 32)
- **Shared mem/SM**: 65536 bytes (up to 100KB configurable)
- **Registers/SM**: 65536

## GPU Error Recovery

If CUDA error 700 (ILLEGAL_INSTRUCTION) or 716 (ILLEGAL_ADDRESS) persists:
1. The GPU context is poisoned from a previous kernel crash
2. Try: **Win+Ctrl+Shift+B** (restarts display driver)
3. If that doesn't help, reboot
4. Error 700 can also mean genuine bad PTX — test with a known-good file first

## Current Architecture (as of last commit)

### Encode pipeline (L5 GPU with tANS32):
1. GPU LZ compress → raw sub-chunks
2. `reencodeGpuWithEntropy` → type-6 tANS32 encoding per stream
3. Wire format per stream: `[sizes 64B][states 64B][Golomb-Rice table][sub-streams]`
4. chunk_type 6 in the 5-byte non-compact header

### Decode pipeline:
1. `scanForTansChunks` — walks compressed data, finds type-6 streams, creates descriptors
   - For type-6: `src_offset += 128` so build kernel sees table after sizes/states header
2. `slzTansBuildTablesKernel` — parses Golomb-Rice table, builds packed 4-byte LUT in global memory
3. `slzTans32DecodeKernel` — all 32 lanes decode in parallel from per-stream LUT
   - Reads sizes/states from `src_offset - 128` (the 128-byte header before the table)
   - Reads sub-stream data from `meta_buf[chunk_id].src_after_table_off`
   - Reads packed LUT from `lut_buf[chunk_id * 2048 * 8]`
4. `slzFullDecompressL1Kernel` — LZ decode, reads pre-decoded literals/tokens from scratch

### Performance (100MB enwik8, L5 GPU):
- Build kernel: ~1.44ms (4578 streams × 0.31μs)
- tANS32 decode: ~0.77ms (all 32 lanes parallel)
- LZ decode: ~4.62ms
- Total kernel: ~7.75ms
- Ratio: 39.2%
- nvCOMP Zstd: 6.8ms kernel, 40.2% ratio

### Key constants:
- `0x37` (55), `0x4c3` (1219), `0x223` (547) — nvCOMP allocation sizes per FSE table type
- FSE spread step: `(L >> 1) + (L >> 3) + 3` for table size L
- Our table sizes: L = 2^log_table_bits where log_table_bits ∈ {8,9,10,11}

## Reverse Engineering Status

A standalone `c:\tmp\reverse_fse.cu` was started to match nvCOMP's `init_fse_tables` ASM.
Key finding: nvCOMP's init kernel does **memory allocation + metadata setup**, NOT LUT construction.
The actual FSE decode table is built inline during decompression. Their 0.4ms is just pointer setup.
Our 1.44ms does actual Golomb-Rice parsing + full LUT construction — fundamentally more work.
