@echo off
REM ============================================================
REM tools/bench_d2d.bat
REM
REM Companion to tools/bench_all.bat. bench_all.bat covers the CLI
REM `-gpu` decode path (host-bounce, CPU scan, CPU walk). This one
REM covers the C-ABI D2D path (slzDecompressAsync, GPU scan, GPU
REM walk) — what game / LLM devs see when they call our library
REM with device-resident input. Use after changes that touch
REM src/gpu/decode/scan_gpu.zig, src/gpu/decode/decode_dispatch.zig,
REM or src/decode/streamlz_decoder.zig's D2D entry points.
REM
REM Why a separate harness: the two paths run different front-half
REM kernels (CLI's gpuBatchDecode does the chunk walk on the CPU and
REM uses scan_host.zig; the D2D path uses gpuWalkFrameImpl +
REM gpuScanChunks). A regression in one is invisible to the other.
REM
REM Sequential by design (per the no-parallel-benchmarks rule).
REM Shows all output (no grep filtering); summary at the end.
REM
REM Input sizing — full enwik8 (100 MB) works. Full silesia
REM (213 MB) at L1-L4 compresses to a multi-block frame that the
REM single-block GPU walk kernel can't decode; per the GPU contract
REM (see feedback_cpu_gpu_separate_formats) we allow that to be a
REM loud failure rather than paying perf on the happy path. The
REM D2D bench therefore caps silesia to 128 MB — large enough to
REM exercise the pipeline at scale, small enough to fit in a
REM single block at every level.
REM
REM Baselines: current-best post-Phase-3d (commit 6b13f38), RTX
REM 4060 Ti sm_89. Refresh after any perf-relevant change.
REM ============================================================
setlocal
pushd "%~dp0\.."

set SLZ=zig-out\bin\streamlz.exe
set BENCH=zig-out\bin\slz_gpu_d2d_bench.exe
set TMP=c:\tmp
set SILESIA_128=%TMP%\silesia_128mb.bin

if not exist "%SLZ%" (
    echo ERROR: %SLZ% not found. Run: zig build -Doptimize=ReleaseFast -Dgpu=true
    exit /b 1
)
if not exist "%BENCH%" (
    echo ERROR: %BENCH% not found. Run: tools\build_d2d_bench.bat
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
if not exist "%SILESIA_128%" (
    echo Creating silesia_128mb.bin from the first 134217728 bytes of silesia_all.tar
    powershell -NoProfile -Command "$b = New-Object byte[] 134217728; $r = [System.IO.File]::OpenRead('assets/silesia_all.tar'); $null = $r.Read($b, 0, 134217728); $r.Close(); [System.IO.File]::WriteAllBytes('%SILESIA_128%', $b)"
    if errorlevel 1 echo silesia_128mb.bin creation FAILED
    if errorlevel 1 exit /b 1
)
for %%L in (1 2 3 4 5) do (
    echo --- enwik8 L%%L ---
    "%SLZ%" -l %%L -gpu assets\enwik8.txt -o "%TMP%\bench_d2d_e_L%%L.slz"
    if errorlevel 1 (echo encode enwik8 L%%L FAILED & exit /b 1)
)
for %%L in (1 2 3 4 5) do (
    echo --- silesia 128MB L%%L ---
    "%SLZ%" -l %%L -gpu "%SILESIA_128%" -o "%TMP%\bench_d2d_s_L%%L.slz"
    if errorlevel 1 (echo encode silesia L%%L FAILED & exit /b 1)
)

echo.
echo ============================================================
echo  Phase 2/3: D2D decode benchmarks (slzDecompressAsync, -r 30)
echo ============================================================
for %%L in (1 2 3 4 5) do (
    echo.
    echo --- enwik8 L%%L D2D bench ---
    "%BENCH%" "%TMP%\bench_d2d_e_L%%L.slz" assets\enwik8.txt --runs 30 > "%TMP%\bench_d2d_out_e_L%%L.txt"
    if errorlevel 1 (echo bench enwik8 L%%L FAILED & type "%TMP%\bench_d2d_out_e_L%%L.txt" & exit /b 1)
    type "%TMP%\bench_d2d_out_e_L%%L.txt"
)
for %%L in (1 2 3 4 5) do (
    echo.
    echo --- silesia 128MB L%%L D2D bench ---
    "%BENCH%" "%TMP%\bench_d2d_s_L%%L.slz" "%SILESIA_128%" --runs 30 > "%TMP%\bench_d2d_out_s_L%%L.txt"
    if errorlevel 1 (echo bench silesia L%%L FAILED & type "%TMP%\bench_d2d_out_s_L%%L.txt" & exit /b 1)
    type "%TMP%\bench_d2d_out_s_L%%L.txt"
)

echo.
echo ============================================================
echo  Phase 3/3: Summary
echo ============================================================
powershell -NoProfile -Command ^
    "$srcEnwik = (Get-Item 'assets/enwik8.txt').Length;" ^
    "$srcSiles = (Get-Item '%SILESIA_128%').Length;" ^
    "$base = @{" ^
        "'e_1' = @{D2D=5.39; Wall=5.40; Ratio=58.6};" ^
        "'e_2' = @{D2D=5.43; Wall=5.43; Ratio=58.6};" ^
        "'e_3' = @{D2D=6.67; Wall=6.68; Ratio=43.7};" ^
        "'e_4' = @{D2D=6.53; Wall=6.53; Ratio=42.7};" ^
        "'e_5' = @{D2D=6.72; Wall=6.73; Ratio=39.6};" ^
        "'s_1' = @{D2D=6.99; Wall=7.00; Ratio=47.8};" ^
        "'s_2' = @{D2D=6.99; Wall=7.00; Ratio=47.8};" ^
        "'s_3' = @{D2D=8.50; Wall=8.51; Ratio=38.0};" ^
        "'s_4' = @{D2D=8.20; Wall=8.21; Ratio=37.5};" ^
        "'s_5' = @{D2D=9.40; Wall=9.41; Ratio=33.9} };" ^
    "function ParseBench($p) {" ^
        "$t = Get-Content $p;" ^
        "$wall = ($t | Where-Object { $_ -match '^\s+best:\s+([\d.]+) ms' } | ForEach-Object { [double]$matches[1] });" ^
        "$d2d = ($t | Where-Object { $_ -match '^\s+gpu kernel best:\s+([\d.]+) ms' } | ForEach-Object { [double]$matches[1] });" ^
        "$kern = ($t | Where-Object { $_ -match '^\s+kernel active best:\s+([\d.]+) ms' } | ForEach-Object { [double]$matches[1] });" ^
        "$verify = ($t | Where-Object { $_ -match '^\s+verify:\s+(\S+)' } | ForEach-Object { $matches[1] });" ^
        "return @{Wall=$wall; D2D=$d2d; Kern=$kern; Verify=$verify} };" ^
    "$fmt = '{0,-16} {1,7} {2,7} {3,7} {4,5}   {5,6} {6,6}   {7,6}';" ^
    "Write-Host ($fmt -f 'corpus/lvl','wall','D2D','kern','gap','ratio','base','verify');" ^
    "Write-Host ($fmt -f '----------------','-----','-----','-----','-----','------','------','------');" ^
    "function Row($corp, $lvl, $slz, $srcLen, $label) {" ^
        "$out = ParseBench (\"$env:TEMP_BENCH/bench_d2d_out_${corp}_L${lvl}.txt\" -replace '/', '\\');" ^
        "$key = \"${corp}_${lvl}\";" ^
        "$b = $base[$key];" ^
        "$frame = (Get-Item $slz).Length;" ^
        "$ratio = [math]::Round(($frame / $srcLen) * 100, 2);" ^
        "$gap = [math]::Round($out.D2D - $out.Kern, 2);" ^
        "Write-Host ($fmt -f $label, $out.Wall, $out.D2D, $out.Kern, $gap, $ratio, $b.Ratio, $out.Verify) };" ^
    "$env:TEMP_BENCH = '%TMP%';" ^
    "1..5 | ForEach-Object { Row 'e' $_ \"%TMP%/bench_d2d_e_L${_}.slz\" $srcEnwik \"enwik8       L$_\" };" ^
    "1..5 | ForEach-Object { Row 's' $_ \"%TMP%/bench_d2d_s_L${_}.slz\" $srcSiles \"silesia 128M L$_\" };" ^
    "Write-Host '';" ^
    "Write-Host 'Times in ms (best of 30).';" ^
    "Write-Host '  wall = host submit-to-sync (slzDecompressAsync + cudaStreamSync) — what game/LLM devs see.';" ^
    "Write-Host '  D2D  = cudaEvent pair around entire slzDecompressAsync work on the caller stream.';" ^
    "Write-Host '  kern = sum of per-kernel cudaEventElapsedTime (slzGetLastTimings) — pure GPU active time.';" ^
    "Write-Host '  gap  = D2D - kern = stream-idle time between kernel launches; the recoverable headroom for Phase 4 (CUDA Graphs).';" ^
    "Write-Host 'Ratio baselines from src/gpu/README.md. Wall/D2D/kern have no baseline yet — first run sets the expectation.'"

echo.
echo bench_d2d done.
popd
endlocal
