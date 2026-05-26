@echo off
REM Compile every CUDA kernel under src/gpu/ to PTX and cubin (cubin
REM produced for the res-usage printout). Covers all five .cu files:
REM   - decode/lz_kernel.cu
REM   - decode/huffman_kernel.cu
REM   - encode/huffman_kernel.cu
REM   - encode/lz_kernel.cu
REM   - encode/assemble_kernel.cu
REM
REM Then runs `zig build` so the resulting streamlz.exe / streamlz_gpu.dll
REM embed the freshly-built PTX. build.zig has a PTX-freshness gate that
REM fails the build if any .cu/.cuh is newer than its .ptx, so it's
REM impossible to accidentally embed stale kernels.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set SRC=%~dp0..\src\gpu\decode\lz_kernel.cu
set OUT=%~dp0..\src\gpu\decode\lz_kernel
echo Compiling GPU kernel...
nvcc -cubin -o "%OUT%.cubin" "%SRC%" -arch=sm_89 -O3
echo NVCC exit code: %ERRORLEVEL%
cuobjdump -res-usage "%OUT%.cubin"
echo.
echo Also generating PTX...
nvcc -ptx -o "%OUT%.ptx" "%SRC%" -arch=sm_89 -O3

echo.
echo Compiling GPU Huffman decode kernel...
set HUFF_SRC=%~dp0..\src\gpu\decode\huffman_kernel.cu
set HUFF_OUT=%~dp0..\src\gpu\decode\huffman_kernel
nvcc -cubin -o "%HUFF_OUT%.cubin" "%HUFF_SRC%" -arch=sm_89 -O3
echo NVCC Huff exit code: %ERRORLEVEL%
cuobjdump -res-usage "%HUFF_OUT%.cubin"
nvcc -ptx -o "%HUFF_OUT%.ptx" "%HUFF_SRC%" -arch=sm_89 -O3
echo Huff PTX exit code: %ERRORLEVEL%

echo.
echo Compiling GPU Huffman encode kernel...
set HENC_SRC=%~dp0..\src\gpu\encode\huffman_kernel.cu
set HENC_OUT=%~dp0..\src\gpu\encode\huffman_kernel
nvcc -cubin -o "%HENC_OUT%.cubin" "%HENC_SRC%" -arch=sm_89 -O3
echo NVCC Huff-enc exit code: %ERRORLEVEL%
cuobjdump -res-usage "%HENC_OUT%.cubin"
nvcc -ptx -o "%HENC_OUT%.ptx" "%HENC_SRC%" -arch=sm_89 -O3
echo Huff-enc PTX exit code: %ERRORLEVEL%

echo.
echo Compiling GPU encode-LZ kernel...
set ELZ_SRC=%~dp0..\src\gpu\encode\lz_kernel.cu
set ELZ_OUT=%~dp0..\src\gpu\encode\lz_kernel
nvcc -cubin -o "%ELZ_OUT%.cubin" "%ELZ_SRC%" -arch=sm_89 -O3
echo NVCC enc-lz exit code: %ERRORLEVEL%
cuobjdump -res-usage "%ELZ_OUT%.cubin"
nvcc -ptx -o "%ELZ_OUT%.ptx" "%ELZ_SRC%" -arch=sm_89 -O3
echo enc-lz PTX exit code: %ERRORLEVEL%

echo.
echo Compiling GPU frame-assembly kernel...
set ASM_SRC=%~dp0..\src\gpu\encode\assemble_kernel.cu
set ASM_OUT=%~dp0..\src\gpu\encode\assemble_kernel
nvcc -cubin -o "%ASM_OUT%.cubin" "%ASM_SRC%" -arch=sm_89 -O3
echo NVCC asm exit code: %ERRORLEVEL%
cuobjdump -res-usage "%ASM_OUT%.cubin"
nvcc -ptx -o "%ASM_OUT%.ptx" "%ASM_SRC%" -arch=sm_89 -O3
echo asm PTX exit code: %ERRORLEVEL%

echo.
echo === Rebuilding streamlz.exe and streamlz_gpu.dll (both embed PTX) ===
pushd "%~dp0.."
zig build -Doptimize=ReleaseFast -Dgpu=true
set RC=%ERRORLEVEL%
popd
if %RC% neq 0 exit /b %RC%
echo === All rebuilt ===
