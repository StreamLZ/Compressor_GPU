@echo off
REM ============================================================
REM tools/bench_d2d_vk.bat
REM
REM VK companion to tools/bench_d2d.bat: sweeps the TRUE-D2D decode
REM path (decompressFramedFromDevice, the slzDecompressAsync_vk
REM analog) across L1-L5 on enwik8 via the SLZ_VK_D2D=1 env mode of
REM `streamlz_vk -db`. Each cell byte-verifies against the host-path
REM output (the CLI exits 1 on mismatch). Sequential by design (per
REM the no-parallel-benchmarks rule).
REM
REM Frames are re-encoded fresh so the sweep is self-contained on any
REM machine with a supported VK device.
REM ============================================================
setlocal enabledelayedexpansion
pushd "%~dp0\.."

set SLZ=zig-out\bin\streamlz_vk.exe
set TMP=c:\tmp
set FAIL=0

if not exist "%SLZ%" (
    echo ERROR: %SLZ% not found. Run: zig build streamlz_vk -Doptimize=ReleaseFast
    exit /b 1
)
if not exist "assets\enwik8.txt" (
    echo ERROR: assets\enwik8.txt not found.
    exit /b 1
)

for %%L in (1 2 3 4 5) do (
    echo.
    echo --- enwik8 L%%L encode ---
    "%SLZ%" -c -l %%L assets\enwik8.txt -o "%TMP%\bench_d2d_vk_e_L%%L.slz"
    if errorlevel 1 set FAIL=1
)

set SLZ_VK_D2D=1
for %%L in (1 2 3 4 5) do (
    echo.
    echo --- enwik8 L%%L TRUE-D2D decode bench ---
    "%SLZ%" -db -r 10 "%TMP%\bench_d2d_vk_e_L%%L.slz"
    if errorlevel 1 (
        echo bench enwik8 L%%L FAILED
        set FAIL=1
    )
)
set SLZ_VK_D2D=

echo.
if "%FAIL%"=="1" (
    echo bench_d2d_vk: FAIL
    exit /b 1
)
echo bench_d2d_vk: all cells verified OK
exit /b 0
