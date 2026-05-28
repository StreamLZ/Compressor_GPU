@echo off
REM ============================================================
REM tools/bench_all.bat
REM
REM Post-phase validation: re-encodes the canonical L1-L5 sets on
REM enwik8 + silesia, runs -db -r 30 decode benchmarks on each,
REM verifies SHA-256 roundtrip, and prints a summary table with
REM D2D / e2e / ratio vs the README baseline. Use after any change
REM that could plausibly affect the GPU encode or decode hot path.
REM
REM Sequential by design (per the no-parallel-benchmarks rule).
REM Shows all output (no grep filtering); summary at the end.
REM
REM Baselines are the current-best post-Phase-3c state (commit 2cb5c37,
REM May 2026, RTX 4060 Ti sm_89). They differ slightly from the
REM src/gpu/README.md table because Phase 3b/3c eliminated one D2H+H2D
REM round trip and moved walk-meta sync to work_stream — net ~0.07-0.37
REM ms e2e improvement vs the README values. Update both this table and
REM the README when shipping changes that move the bar.
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
    "%SLZ%" -db -r 30 -gpu "%TMP%\bench_e_L%%L.slz" > "%TMP%\bench_out_e_L%%L.txt"
    if errorlevel 1 (echo bench enwik8 L%%L FAILED & exit /b 1)
    type "%TMP%\bench_out_e_L%%L.txt"
)
for %%L in (1 2 3 4 5) do (
    echo.
    echo --- silesia L%%L decode bench ---
    "%SLZ%" -db -r 30 -gpu "%TMP%\bench_s_L%%L.slz" > "%TMP%\bench_out_s_L%%L.txt"
    if errorlevel 1 (echo bench silesia L%%L FAILED & exit /b 1)
    type "%TMP%\bench_out_s_L%%L.txt"
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

echo.
echo ============================================================
echo  Summary
echo ============================================================
powershell -NoProfile -Command ^
    "$srcEnwik = (Get-Item 'assets/enwik8.txt').Length;" ^
    "$srcSiles = (Get-Item 'assets/silesia_all.tar').Length;" ^
    "$shaE = (Get-FileHash 'assets/enwik8.txt').Hash;" ^
    "$shaS = (Get-FileHash 'assets/silesia_all.tar').Hash;" ^
    "$base = @{" ^
        "'e_1' = @{D2D=2.92; E2E=15.50; Ratio=58.6};" ^
        "'e_2' = @{D2D=2.93; E2E=15.50; Ratio=58.6};" ^
        "'e_3' = @{D2D=4.08; E2E=15.54; Ratio=43.7};" ^
        "'e_4' = @{D2D=3.92; E2E=15.30; Ratio=42.7};" ^
        "'e_5' = @{D2D=4.10; E2E=15.26; Ratio=39.6};" ^
        "'s_1' = @{D2D=5.06; E2E=29.97; Ratio=47.8};" ^
        "'s_2' = @{D2D=5.08; E2E=29.93; Ratio=47.8};" ^
        "'s_3' = @{D2D=7.13; E2E=30.59; Ratio=38.0};" ^
        "'s_4' = @{D2D=6.95; E2E=30.31; Ratio=37.5};" ^
        "'s_5' = @{D2D=7.70; E2E=30.49; Ratio=33.9} };" ^
    "function ParseBench($p) {" ^
        "$t = Get-Content $p;" ^
        "$e2e = ($t | Where-Object { $_ -match '^\s+best:\s+([\d.]+) ms' } | ForEach-Object { [double]$matches[1] });" ^
        "$d2d = ($t | Where-Object { $_ -match '^\s+gpu kernel best:\s+([\d.]+) ms' } | ForEach-Object { [double]$matches[1] });" ^
        "return @{E2E=$e2e; D2D=$d2d} };" ^
    "$fmt = '{0,-12} {1,7} {2,7} {3,8}   {4,7} {5,7} {6,8}   {7,6} {8,6} {9,7}   {10,4}';" ^
    "Write-Host ($fmt -f 'corpus/lvl','D2D','base','d-D2D','e2e','base','d-e2e','ratio','base','d-ratio','SHA');" ^
    "Write-Host ($fmt -f '------------','-----','-----','-----','-----','-----','-----','-----','-----','-----','---');" ^
    "function Row($corp, $lvl, $slz, $bin, $srcLen, $srcSha) {" ^
        "$out = ParseBench (\"$env:TEMP_BENCH/bench_out_${corp}_L${lvl}.txt\" -replace '/', '\\');" ^
        "$key = \"${corp}_${lvl}\";" ^
        "$b = $base[$key];" ^
        "$frame = (Get-Item $slz).Length;" ^
        "$ratio = [math]::Round(($frame / $srcLen) * 100, 2);" ^
        "$sha = (Get-FileHash $bin).Hash;" ^
        "$shaOk = if ($sha -eq $srcSha) {'OK'} else {'FAIL'};" ^
        "$dD2D = [math]::Round($out.D2D - $b.D2D, 2);" ^
        "$dE2E = [math]::Round($out.E2E - $b.E2E, 2);" ^
        "$dRat = [math]::Round($ratio - $b.Ratio, 2);" ^
        "$label = if ($corp -eq 'e') {\"enwik8  L$lvl\"} else {\"silesia L$lvl\"};" ^
        "Write-Host ($fmt -f $label, $out.D2D, $b.D2D, $dD2D, $out.E2E, $b.E2E, $dE2E, $ratio, $b.Ratio, $dRat, $shaOk) };" ^
    "$env:TEMP_BENCH = '%TMP%';" ^
    "1..5 | ForEach-Object { Row 'e' $_ \"%TMP%/bench_e_L${_}.slz\" \"%TMP%/bench_e_L${_}.bin\" $srcEnwik $shaE };" ^
    "1..5 | ForEach-Object { Row 's' $_ \"%TMP%/bench_s_L${_}.slz\" \"%TMP%/bench_s_L${_}.bin\" $srcSiles $shaS };" ^
    "Write-Host '';" ^
    "Write-Host 'Times in ms (best of 30). d-* columns are now minus baseline; negative = improvement.';" ^
    "Write-Host 'Baseline: current-best post-Phase-3c (commit 2cb5c37, RTX 4060 Ti). Refresh after any perf-relevant change.'"

echo.
echo bench_all done.
popd
endlocal
