@echo off
REM Clean-build streamlz_vk.exe. Forces full recompilation of GLSL .comp
REM kernels into SPIR-V, then the Zig binary that embeds them.
REM
REM WHY THIS EXISTS:
REM build.zig at addSrcVkShaderSteps (around line 1031-1071) declares the
REM `.comp` file as glslc's addFileArg() input but does NOT declare the
REM `.glsl` files it #include's as dependencies. So editing any shared
REM GLSL header (lz_dispatch.glsl, lz_decode_general.glsl, etc.) leaves
REM the cached .spv stale and the SPV is silently re-embedded into the
REM binary with no recompilation. This caused a ~5-hour debugging
REM nightmare on iter 4f (commit 4270ea4) where the fix worked but ran
REM as stale bytecode through every "verify" run.
REM
REM Until build.zig is fixed to track .glsl #include deps via
REM `addFileInput()` or glslc -MD depfile output, USE THIS SCRIPT when
REM you've edited any .glsl shared header.
REM
REM See srcVK/PortAdaptations.md A-012 for the upstream tracking entry.

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
