@echo off
REM NCU profile of all 4 GPU kernels in the L5 decode pipeline:
REM   slzGatherRawOff16Kernel  (raw off16 → entropy scratch)
REM   slzHuffBuildLutKernel    (per-chunk Huffman LUTs)
REM   slzHuffDecode4StreamKernel (32-stream BIL Huff decode)
REM   slzLzDecodeKernel        (final LZ decode)
REM
REM Args:
REM   %1 = SCN1 dump path (default c:/tmp/sil_l5_scan.bin for silesia L5)
REM   %2 = expected decompressed file path (default assets/silesia_all.tar)
set SCAN=%1
if "%SCAN%"=="" set SCAN=c:/tmp/sil_l5_scan.bin
set EXPECTED=%2
if "%EXPECTED%"=="" set EXPECTED=assets/silesia_all.tar

set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat"
set EXE=%~dp0decode_l5_harness.exe
set REPORT=c:\tmp\decode_l5_profile.ncu-rep

echo Profiling %SCAN% ...
REM --launch-skip 4 skips warm-up + first timed run (which often has init noise)
REM --launch-count 4 captures one full pass of (gather, huff_build, huff_decode, lz_decode)
%NCU% --target-processes all --launch-skip 4 --launch-count 4 ^
    --section SpeedOfLight ^
    --section MemoryWorkloadAnalysis ^
    --section WarpStateStats ^
    --section ComputeWorkloadAnalysis ^
    --section Occupancy ^
    --force-overwrite ^
    -o %REPORT% ^
    "%EXE%" "%SCAN%" "%EXPECTED%" 2
echo NCU exit: %ERRORLEVEL%
echo Report: %REPORT%.ncu-rep
