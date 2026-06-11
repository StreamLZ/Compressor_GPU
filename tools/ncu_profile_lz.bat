@echo off
echo Profiling slzLzDecodeRawKernel (post-#1/#2 flat copies)...
echo.
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat" ^
  --set full ^
  --kernel-name slzLzDecodeRawPipelinedKernel ^
  --launch-count 1 ^
  -o c:\tmp\slz_lz_pipe_post15 ^
  %~dp0..\zig-out\bin\streamlz.exe -d c:\tmp\bench_e_L1.slz -o c:\tmp\ncu_out.bin
echo.
echo ================================================================
echo Profile saved to c:\tmp\slz_lz_pipe_post15.ncu-rep
echo.
echo Also profiling slzLzDecodeKernel (general/huff path)...
echo.
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.0\ncu.bat" ^
  --set full ^
  --kernel-name slzLzDecodeKernel ^
  --launch-count 1 ^
  -o c:\tmp\slz_lz_general_post15 ^
  %~dp0..\zig-out\bin\streamlz.exe -d c:\tmp\bench_e_L5.slz -o c:\tmp\ncu_out.bin
echo.
echo ================================================================
echo Profile saved to c:\tmp\slz_lz_general_post15.ncu-rep
echo.
echo Both profiles done. I will read the .ncu-rep files.
pause
