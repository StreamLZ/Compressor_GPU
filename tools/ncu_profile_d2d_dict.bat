@echo off
rem -- NCU profile of the D2D decode front-half + dict LZ kernel ----
rem Run from an ADMIN shell (GPU performance counters need elevated
rem access: ERR_NVGPUCTRPERM otherwise).
rem
rem Profiles three kernels on the enwik8-with-text-dict D2D decode:
rem   slzWalkFrameKernel            - the serial frame walk (v4 #16
rem                                   speed question: expected ~100%%
rem                                   long-scoreboard stall at ~zero
rem                                   occupancy - the 1-thread chain)
rem   slzPrefixSumChunksKernel      - fusion candidate
rem   slzLzDecodeRawPipelinedKernel - post-fu7 flat-dict-pass profile
rem
rem Prereqs (run these first if missing):
rem   zig build gpulib -Doptimize=ReleaseFast
rem   zig-out\bin\streamlz.exe -D text assets\enwik8.txt -o c:\tmp\e8_dict.slz
rem
rem Output: c:\tmp\slz_d2d_dict_<kernel>.ncu-rep + console summaries.

setlocal
set NCU="C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat"
set BENCH=%~dp0..\zig-out\bin\slz_gpu_d2d_bench.exe
set FRAME=c:\tmp\e8_dict.slz
set SRC=%~dp0..\assets\enwik8.txt

if not exist %NCU% (
    echo error: ncu not found at %NCU%
    exit /b 1
)
if not exist "%BENCH%" (
    echo error: build the bench first: zig build d2d_bench / see tools\build_d2d_bench.bat
    exit /b 1
)
if not exist "%FRAME%" (
    echo error: %FRAME% missing - create with:
    echo   zig-out\bin\streamlz.exe -D text assets\enwik8.txt -o c:\tmp\e8_dict.slz
    exit /b 1
)

echo === slzWalkFrameKernel (full set) ===
call %NCU% --set full --kernel-name slzWalkFrameKernel --launch-count 1 ^
  -o c:\tmp\slz_d2d_dict_walk -f ^
  "%BENCH%" "%FRAME%" "%SRC%" --runs 1

echo === slzPrefixSumChunksKernel (full set) ===
call %NCU% --set full --kernel-name slzPrefixSumChunksKernel --launch-count 1 ^
  -o c:\tmp\slz_d2d_dict_prefix -f ^
  "%BENCH%" "%FRAME%" "%SRC%" --runs 1

echo === slzLzDecodeRawPipelinedKernel (full set) ===
call %NCU% --set full --kernel-name slzLzDecodeRawPipelinedKernel --launch-count 1 ^
  -o c:\tmp\slz_d2d_dict_lz -f ^
  "%BENCH%" "%FRAME%" "%SRC%" --runs 1

echo.
echo ================================================================
echo Reports: c:\tmp\slz_d2d_dict_{walk,prefix,lz}.ncu-rep
echo Quick text summaries:
call %NCU% --import c:\tmp\slz_d2d_dict_walk.ncu-rep --page details 2>nul | findstr /C:"SM Frequency" /C:"Duration" /C:"Achieved Occupancy" /C:"SM Busy" /C:"Issue Slots Busy" /C:"Long Scoreboard"
echo ================================================================
echo Done. Hand the .ncu-rep paths back for analysis.
