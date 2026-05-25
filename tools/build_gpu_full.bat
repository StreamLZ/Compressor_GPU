@echo off
REM Rebuild GPU kernels AND every binary that embeds them.
REM tools\build_gpu.bat alone only updates .cubin/.ptx — both the
REM DLL (streamlz_gpu.dll, used by d2d_test/async_test) and the EXE
REM (streamlz.exe, used by -db bench and -d CLI) embed PTX via
REM @embedFile and so must be rebuilt separately whenever any .cu or
REM .cuh file under src/gpu/ changes.
call "%~dp0build_gpu.bat"
if errorlevel 1 exit /b 1
echo.
echo === Rebuilding streamlz_gpu.dll AND streamlz.exe (both embed PTX) ===
pushd "%~dp0.."
zig build -Doptimize=ReleaseFast -Dgpu=true
set RC=%ERRORLEVEL%
popd
if %RC% neq 0 exit /b %RC%
echo === All rebuilt ===
dir /B /TC "%~dp0..\src\gpu\decode\lz_kernel.ptx" "%~dp0..\zig-out\bin\streamlz_gpu.dll" "%~dp0..\zig-out\bin\streamlz.exe"
