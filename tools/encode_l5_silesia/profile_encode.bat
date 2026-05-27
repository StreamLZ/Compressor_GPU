@echo off
REM Profile the LZ encode kernel with NCU. Use small subset (32 MB) so NCU
REM overhead is bounded. Hash for 32 MB fits in VRAM (~4 GB) so this
REM measures the "fits-in-VRAM" regime. The larger silesia regime that
REM PCIe-spills is the bottleneck — see encode_l5_harness.exe output
REM (115 ms / 279 MB/s at 32MB vs 4120 ms / 49 MB/s at 203MB).
REM
REM Args:
REM   %1 = input bytes (default 33554432 = 32 MB)
REM   %2 = hash_bits override (default = level policy, 20 for L5)
set BYTES=%1
if "%BYTES%"=="" set BYTES=33554432
set HB=%2
if "%HB%"=="" set HB=20

set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat"
set EXE=%~dp0encode_l5_harness.exe
set REPORT=c:\tmp\encode_l5_profile.ncu-rep

echo Profiling %BYTES% bytes, hash_bits=%HB% ...
%NCU% --target-processes all --launch-skip 1 --launch-count 1 ^
    --section SpeedOfLight ^
    --section MemoryWorkloadAnalysis ^
    --section WarpStateStats ^
    --section ComputeWorkloadAnalysis ^
    --section Occupancy ^
    --force-overwrite ^
    -o %REPORT% ^
    "%EXE%" assets/silesia_all.tar 2 5 %BYTES% %HB%
echo NCU exit: %ERRORLEVEL%
echo Report: %REPORT%.ncu-rep
