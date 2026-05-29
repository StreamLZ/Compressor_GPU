@echo off
REM Compile every CUDA kernel to PTX + CUBIN (cubin produced for the
REM res-usage printout). Five .cu translation units cover the entire
REM device code:
REM   - src/decode/lz_kernel.cu
REM   - src/decode/huffman_kernel.cu
REM   - src/encode/huffman_kernel.cu
REM   - src/encode/lz_kernel.cu
REM   - src/encode/assemble_kernel.cu
REM
REM build.zig has a PTX-freshness gate that fails the build if any
REM .cu/.cuh is newer than its .ptx, so it's impossible to accidentally
REM embed stale device code.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

echo Compiling decode LZ kernel...
set SRC=%~dp0..\src\decode\lz_kernel.cu
set OUT=%~dp0..\src\decode\lz_kernel
nvcc -cubin -o "%OUT%.cubin" "%SRC%" -arch=sm_89 -O3
echo NVCC exit code: %ERRORLEVEL%
cuobjdump -res-usage "%OUT%.cubin"
nvcc -ptx -o "%OUT%.ptx" "%SRC%" -arch=sm_89 -O3

echo.
echo Compiling decode Huffman kernel...
set HUFF_SRC=%~dp0..\src\decode\huffman_kernel.cu
set HUFF_OUT=%~dp0..\src\decode\huffman_kernel
nvcc -cubin -o "%HUFF_OUT%.cubin" "%HUFF_SRC%" -arch=sm_89 -O3
echo NVCC Huff exit code: %ERRORLEVEL%
cuobjdump -res-usage "%HUFF_OUT%.cubin"
nvcc -ptx -o "%HUFF_OUT%.ptx" "%HUFF_SRC%" -arch=sm_89 -O3
echo Huff PTX exit code: %ERRORLEVEL%

echo.
echo Compiling encode Huffman kernel...
set HENC_SRC=%~dp0..\src\encode\huffman_kernel.cu
set HENC_OUT=%~dp0..\src\encode\huffman_kernel
nvcc -cubin -o "%HENC_OUT%.cubin" "%HENC_SRC%" -arch=sm_89 -O3
echo NVCC Huff-enc exit code: %ERRORLEVEL%
cuobjdump -res-usage "%HENC_OUT%.cubin"
nvcc -ptx -o "%HENC_OUT%.ptx" "%HENC_SRC%" -arch=sm_89 -O3
echo Huff-enc PTX exit code: %ERRORLEVEL%

echo.
echo Compiling encode LZ kernel...
set ELZ_SRC=%~dp0..\src\encode\lz_kernel.cu
set ELZ_OUT=%~dp0..\src\encode\lz_kernel
nvcc -cubin -o "%ELZ_OUT%.cubin" "%ELZ_SRC%" -arch=sm_89 -O3
echo NVCC enc-lz exit code: %ERRORLEVEL%
cuobjdump -res-usage "%ELZ_OUT%.cubin"
nvcc -ptx -o "%ELZ_OUT%.ptx" "%ELZ_SRC%" -arch=sm_89 -O3
echo enc-lz PTX exit code: %ERRORLEVEL%

echo.
echo Compiling encode frame-assembly kernel...
set ASM_SRC=%~dp0..\src\encode\assemble_kernel.cu
set ASM_OUT=%~dp0..\src\encode\assemble_kernel
nvcc -cubin -o "%ASM_OUT%.cubin" "%ASM_SRC%" -arch=sm_89 -O3
echo NVCC asm exit code: %ERRORLEVEL%
cuobjdump -res-usage "%ASM_OUT%.cubin"
nvcc -ptx -o "%ASM_OUT%.ptx" "%ASM_SRC%" -arch=sm_89 -O3
echo asm PTX exit code: %ERRORLEVEL%

echo.
echo === Rebuilding streamlz.exe and streamlz_gpu.dll (both embed PTX) ===
pushd "%~dp0.."
zig build -Doptimize=ReleaseFast
set RC=%ERRORLEVEL%
popd
if %RC% neq 0 exit /b %RC%
echo === All rebuilt ===
