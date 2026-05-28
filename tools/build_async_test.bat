@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
echo Compiling async test...
nvcc -cudart static "%~dp0slz_gpu_async_test.c" -I "%~dp0..\include" -L "%~dp0..\zig-out\lib" -lstreamlz_gpu -o "%~dp0..\zig-out\bin\slz_gpu_async_test.exe"
echo NVCC exit code: %ERRORLEVEL%
