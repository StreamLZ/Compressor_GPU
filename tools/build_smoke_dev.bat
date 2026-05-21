@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
echo Compiling device-path smoke test...
nvcc -cudart static "%~dp0slz_gpu_smoke_dev.c" -I "%~dp0..\include" -L "%~dp0..\zig-out\lib" -lstreamlz_gpu -o "%~dp0..\zig-out\bin\slz_gpu_smoke_dev.exe"
echo NVCC exit code: %ERRORLEVEL%
