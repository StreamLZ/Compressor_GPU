@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
echo Compiling D2D-path decode benchmark...
nvcc -cudart static "%~dp0slz_gpu_d2d_bench.c" -I "%~dp0..\include" -L "%~dp0..\zig-out\lib" -lstreamlz_gpu -o "%~dp0..\zig-out\bin\slz_gpu_d2d_bench.exe"
echo NVCC exit code: %ERRORLEVEL%
