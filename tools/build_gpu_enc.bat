@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set SRC=%~dp0..\src\gpu\encode\lz_kernel.cu
set OUT=%~dp0..\src\gpu\encode\lz_kernel
echo Compiling GPU encode kernel...
nvcc -cubin -o "%OUT%.cubin" "%SRC%" -arch=sm_89 -O3
echo NVCC exit code: %ERRORLEVEL%
cuobjdump -res-usage "%OUT%.cubin"
echo.
nvcc -ptx -o "%OUT%.ptx" "%SRC%" -arch=sm_89 -O3
echo PTX exit code: %ERRORLEVEL%

echo.
echo Compiling GPU frame-assembly kernel...
set ASM=%~dp0..\src\gpu\encode\assemble_kernel.cu
set ASMOUT=%~dp0..\src\gpu\encode\assemble_kernel
nvcc -cubin -o "%ASMOUT%.cubin" "%ASM%" -arch=sm_89 -O3
echo NVCC asm exit code: %ERRORLEVEL%
cuobjdump -res-usage "%ASMOUT%.cubin"
nvcc -ptx -o "%ASMOUT%.ptx" "%ASM%" -arch=sm_89 -O3
echo asm PTX exit code: %ERRORLEVEL%
