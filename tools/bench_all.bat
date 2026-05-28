@echo off
REM ============================================================
REM tools/bench_all.bat
REM
REM Post-phase validation: re-encodes the canonical L1-L5 sets on
REM enwik8 + silesia, runs -db -r 30 decode benchmarks on each, and
REM verifies SHA-256 roundtrip. Use after any change that could
REM plausibly affect the GPU encode or decode hot path.
REM
REM Sequential by design (per the no-parallel-benchmarks rule).
REM Shows all output (no grep filtering).
REM
REM Expected baseline (RTX 4060 Ti, sm_89, May 2026) — see
REM src/gpu/README.md for the authoritative table:
REM
REM   enwik8    L1 D2D 2.91 / e2e 15.57 / ratio 58.6%
REM   enwik8    L2 D2D 2.93 / e2e 15.59 / ratio 58.6%
REM   enwik8    L3 D2D 4.09 / e2e 15.66 / ratio 43.7%
REM   enwik8    L4 D2D 3.93 / e2e 15.42 / ratio 42.7%
REM   enwik8    L5 D2D 4.12 / e2e 15.39 / ratio 39.6%
REM   silesia   L1 D2D 5.07 / e2e 30.07 / ratio 47.8%
REM   silesia   L2 D2D 5.09 / e2e 30.11 / ratio 47.8%
REM   silesia   L3 D2D 7.15 / e2e 30.84 / ratio 38.0%
REM   silesia   L4 D2D 6.94 / e2e 30.65 / ratio 37.5%
REM   silesia   L5 D2D 7.72 / e2e 30.76 / ratio 33.9%
REM ============================================================
setlocal
pushd "%~dp0\.."

set SLZ=zig-out\bin\streamlz.exe
set TMP=c:\tmp

if not exist "%SLZ%" (
    echo ERROR: %SLZ% not found. Run: zig build -Doptimize=ReleaseFast -Dgpu=true
    exit /b 1
)
if not exist "assets\enwik8.txt" (
    echo ERROR: assets\enwik8.txt not found.
    exit /b 1
)
if not exist "assets\silesia_all.tar" (
    echo ERROR: assets\silesia_all.tar not found.
    exit /b 1
)

echo.
echo ============================================================
echo  Phase 1/3: encode L1-L5 on both corpora
echo ============================================================
for %%L in (1 2 3 4 5) do (
    echo --- enwik8 L%%L ---
    "%SLZ%" -l %%L -gpu assets\enwik8.txt -o "%TMP%\bench_e_L%%L.slz"
    if errorlevel 1 (echo encode enwik8 L%%L FAILED & exit /b 1)
)
for %%L in (1 2 3 4 5) do (
    echo --- silesia L%%L ---
    "%SLZ%" -l %%L -gpu assets\silesia_all.tar -o "%TMP%\bench_s_L%%L.slz"
    if errorlevel 1 (echo encode silesia L%%L FAILED & exit /b 1)
)

echo.
echo ============================================================
echo  Phase 2/3: decode benchmarks (-db -r 30)
echo ============================================================
for %%L in (1 2 3 4 5) do (
    echo.
    echo --- enwik8 L%%L decode bench ---
    "%SLZ%" -db -r 30 -gpu "%TMP%\bench_e_L%%L.slz"
    if errorlevel 1 (echo bench enwik8 L%%L FAILED & exit /b 1)
)
for %%L in (1 2 3 4 5) do (
    echo.
    echo --- silesia L%%L decode bench ---
    "%SLZ%" -db -r 30 -gpu "%TMP%\bench_s_L%%L.slz"
    if errorlevel 1 (echo bench silesia L%%L FAILED & exit /b 1)
)

echo.
echo ============================================================
echo  Phase 3/3: SHA-256 roundtrip verify
echo ============================================================
for %%L in (1 2 3 4 5) do (
    "%SLZ%" -d -gpu "%TMP%\bench_e_L%%L.slz" -o "%TMP%\bench_e_L%%L.bin" >nul
    if errorlevel 1 (echo decode enwik8 L%%L FAILED & exit /b 1)
)
for %%L in (1 2 3 4 5) do (
    "%SLZ%" -d -gpu "%TMP%\bench_s_L%%L.slz" -o "%TMP%\bench_s_L%%L.bin" >nul
    if errorlevel 1 (echo decode silesia L%%L FAILED & exit /b 1)
)
powershell -NoProfile -Command ^
    "$e = (Get-FileHash assets/enwik8.txt).Hash;" ^
    "$s = (Get-FileHash assets/silesia_all.tar).Hash;" ^
    "Write-Host ('enwik8.txt        SHA = ' + $e);" ^
    "1..5 | ForEach-Object { $h = (Get-FileHash ('%TMP%/bench_e_L' + $_ + '.bin')).Hash; $ok = if ($h -eq $e) {'OK '} else {'FAIL'}; Write-Host ('enwik8 L'  + $_ + '  ' + $ok + '  ' + $h) };" ^
    "Write-Host ('silesia_all.tar   SHA = ' + $s);" ^
    "1..5 | ForEach-Object { $h = (Get-FileHash ('%TMP%/bench_s_L' + $_ + '.bin')).Hash; $ok = if ($h -eq $s) {'OK '} else {'FAIL'}; Write-Host ('silesia L' + $_ + '  ' + $ok + '  ' + $h) }"

echo.
echo bench_all done.
popd
endlocal
