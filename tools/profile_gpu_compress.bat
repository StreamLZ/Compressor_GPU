@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1

echo Profiling GPU compress kernel on enwik8 (100MB)...
echo.

ncu --set full --target-processes all -o c:\tmp\slz_compress_profile "c:\Users\james.JAMESWORK2025\Repos\StreamLZ_native\zig-out\bin\streamlz.exe" -l 2 -gpu "c:\Users\james.JAMESWORK2025\Repos\StreamLZ_native\assets\enwik8.txt" -o c:\tmp\ncu_dummy.slz

echo.
echo NCU exit code: %ERRORLEVEL%
echo Report: c:\tmp\slz_compress_profile.ncu-rep
pause
