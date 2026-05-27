@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set SRC=%~dp0decode_l5_harness.cu
set OUT=%~dp0decode_l5_harness.exe
echo Compiling decode_l5_harness ...
nvcc -O3 -arch=sm_89 -lineinfo -o "%OUT%" "%SRC%"
echo NVCC exit code: %ERRORLEVEL%
