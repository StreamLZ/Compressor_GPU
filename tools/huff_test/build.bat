@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set SRC=%~dp0huff_test.cu
set OUT=%~dp0huff_test.exe
echo Compiling GPU huffman test...
nvcc -O3 -arch=sm_89 -o "%OUT%" "%SRC%"
echo NVCC exit code: %ERRORLEVEL%
