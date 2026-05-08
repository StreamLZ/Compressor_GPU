@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
set SRC=%~dp0..\src\decode\fast\gpu_decode_kernel.cu
set OUT=%~dp0..\src\decode\fast\gpu_decode_kernel
echo Compiling GPU kernel...
nvcc -cubin -o "%OUT%.cubin" "%SRC%" -arch=sm_89 -O3
echo NVCC exit code: %ERRORLEVEL%
cuobjdump -res-usage "%OUT%.cubin"
echo.
echo Also generating PTX...
nvcc -ptx -o "%OUT%.ptx" "%SRC%" -arch=sm_89 -O3
