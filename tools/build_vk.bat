@echo off
REM Force-clean rebuild of streamlz_vk.exe. Wipes cached binaries +
REM .spv blobs + the entire .zig-cache, then rebuilds from scratch.
REM
REM A-012 (the underlying stale-SPV trap that originally motivated this
REM script) was RESOLVED on 2026-06-08: build.zig now wires glslc -MD
REM depfile output through addDepFileOutputArg, so editing any .glsl
REM #include header correctly invalidates the dependent .spv blobs.
REM Plain `zig build streamlz_vk` after a .glsl edit now Does The
REM Right Thing.
REM
REM This script is kept as a force-clean utility for other cache-
REM busting scenarios — debugging suspected cache corruption, verifying
REM a fix isn't masked by stale state, etc. ~30-60 seconds of rebuild
REM cost vs the immediate confidence that nothing is stale.
REM
REM See srcVK/PortAdaptations.md A-012 for the tracking entry.

setlocal

echo Deleting stale build artifacts...

REM Delete cached binaries that may be stale. Errors are silenced —
REM the file may not exist on a fresh clone, that's fine.
del /q "%~dp0..\zig-out\bin\streamlz_vk.exe" 2>nul
del /q "%~dp0..\zig-out\srcvk_shaders\*.spv" 2>nul

REM Also nuke the in-Zig cache for the srcvk shader steps. This is the
REM ultra-defensive cleanup — addresses build-graph misses we haven't
REM caught yet. Cheap: ~2-3 seconds extra build time, vs the hours saved
REM by avoiding stale-SPV false positives.
echo Wiping .zig-cache (defensive)...
rmdir /s /q "%~dp0..\.zig-cache" 2>nul

echo.
echo === Rebuilding streamlz_vk.exe ===
pushd "%~dp0.."
zig build streamlz_vk -Doptimize=ReleaseFast
set RC=%ERRORLEVEL%
popd

if %RC% neq 0 goto :failed

echo.
echo === streamlz_vk.exe rebuilt clean ===
endlocal
exit /b 0

:failed
echo.
echo BUILD FAILED with exit code %RC%
endlocal
exit /b %RC%
