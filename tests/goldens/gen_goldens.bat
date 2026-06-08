@echo off
REM Generate the test golden .slz files via the CUDA reference encoder.
REM
REM Goldens are NOT tracked in git (~2.4 GiB total; silesia .slz files
REM exceed GitHub's 100 MB per-file limit). This script regenerates
REM them locally from the user-provided assets in assets/.
REM
REM USAGE
REM   cmd.exe /c tests\goldens\gen_goldens.bat
REM
REM PREREQUISITES
REM   - assets/web.txt           (any size; typically ~4.5 MB)
REM   - assets/enwik8.txt        (100 MB — download from
REM                                https://mattmahoney.net/dc/enwik8.zip)
REM   - assets/silesia_all.tar   (~200 MB — concatenated Silesia corpus)
REM   - zig-out/bin/streamlz.exe (CUDA encoder — run `zig build` first)
REM
REM OUTPUT
REM   tests/goldens/{web.txt,enwik8.txt,silesia_all.tar}.L{1,2,3,4,5}.slz
REM   (15 files, ~2.4 GiB total)
REM
REM VERIFICATION
REM   After generation, this script verifies each file's SHA-256
REM   against tests/goldens/manifest.sha256.  A mismatch indicates
REM   either (a) a different encoder version (manifest may need
REM   regeneration) or (b) corrupt source data.

setlocal

set ROOT=%~dp0..\..
set ENC=%ROOT%\zig-out\bin\streamlz.exe
set ASSETS=%ROOT%\assets
set OUT=%~dp0

if not exist "%ENC%" (
    echo ERROR: streamlz.exe not found at %ENC%
    echo Run `zig build -Doptimize=ReleaseFast` from the repo root first.
    exit /b 1
)

for %%A in (web.txt enwik8.txt silesia_all.tar) do (
    if not exist "%ASSETS%\%%A" (
        echo ERROR: source asset not found at %ASSETS%\%%A
        echo See header of this script for asset download links.
        exit /b 1
    )
)

echo Generating goldens via %ENC%
echo Source dir : %ASSETS%
echo Output dir : %OUT%
echo.

for %%A in (web.txt enwik8.txt silesia_all.tar) do (
    for %%L in (1 2 3 4 5) do (
        echo   %%A  L%%L
        "%ENC%" -c -l %%L -o "%OUT%%%A.L%%L.slz" "%ASSETS%\%%A" >nul 2>&1
        if errorlevel 1 (
            echo ERROR: encode failed for %%A at level %%L
            exit /b 1
        )
    )
)

echo.
echo === Verifying SHA-256 against manifest.sha256 ===
pushd "%ROOT%"
where sha256sum >nul 2>&1
if errorlevel 1 (
    echo NOTE: sha256sum not on PATH — skipping verification.
    echo       Install Git for Windows ^(includes sha256sum^) to enable.
) else (
    sha256sum -c tests/goldens/manifest.sha256
    if errorlevel 1 (
        echo.
        echo WARNING: one or more goldens did not match manifest.
        echo The encoder may have changed since the manifest was generated.
        echo If the changes are intentional, regenerate the manifest with:
        echo   sha256sum tests/goldens/*.slz ^> tests/goldens/manifest.sha256
        popd
        exit /b 1
    )
)
popd

echo.
echo === Goldens generated + verified ===

endlocal
