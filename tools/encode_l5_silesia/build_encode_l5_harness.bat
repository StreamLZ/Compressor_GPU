@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set SRC=%~dp0encode_l5_harness.cu
set OUT=%~dp0encode_l5_harness.exe
set CHAIN=%1
if "%CHAIN%"=="" set CHAIN=8
set EXTRA=%2
echo Compiling encode_l5_harness with CHAIN_MAX_STEPS=%CHAIN% EXTRA=%EXTRA% ...
nvcc -O3 -arch=sm_89 -lineinfo -DHARNESS_CHAIN_MAX_STEPS=%CHAIN% %EXTRA% -o "%OUT%" "%SRC%"
echo NVCC exit code: %ERRORLEVEL%
