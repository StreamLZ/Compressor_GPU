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

echo.
echo Compiling GPU Huffman decode kernel...
set HUFF_SRC=%~dp0..\src\decode\fast\gpu_huff_decode_kernel.cu
set HUFF_OUT=%~dp0..\src\decode\fast\gpu_huff_decode_kernel
nvcc -cubin -o "%HUFF_OUT%.cubin" "%HUFF_SRC%" -arch=sm_89 -O3
echo NVCC Huff exit code: %ERRORLEVEL%
cuobjdump -res-usage "%HUFF_OUT%.cubin"
nvcc -ptx -o "%HUFF_OUT%.ptx" "%HUFF_SRC%" -arch=sm_89 -O3
echo Huff PTX exit code: %ERRORLEVEL%

echo.
echo Compiling Vulkan SPIR-V...
set COMP=%~dp0..\src\decode\fast\gpu_decode_kernel.comp
set SPV=%~dp0..\src\decode\fast\gpu_decode_kernel.spv
"C:\VulkanSDK\1.4.341.1\Bin\glslangValidator.exe" -V -S comp --target-env vulkan1.3 -o "%SPV%" "%COMP%"
echo glslang exit code: %ERRORLEVEL%
