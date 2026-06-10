@echo off
rem -- compute-sanitizer pass over the CUDA codec --------------------
rem Backport of the srcVK "always run the validation layer before
rem claiming done" practice (it caught 3 real bugs on the VK side).
rem CUDA analog: compute-sanitizer memcheck (default) or racecheck.
rem
rem Usage:
rem   tools\sanitize.bat            -> memcheck over an encode+decode
rem                                    roundtrip (web.txt, L1 + L5)
rem   tools\sanitize.bat racecheck  -> same workload under racecheck
rem                                    (slower; shared-mem hazards)
rem
rem Run at milestones / before claiming a kernel change done. Uses
rem web.txt (4.5 MB) so the sanitized run stays in the tens of seconds;
rem scale checks belong to the normal bench flow, not this gate.

setlocal
set SAN="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\compute-sanitizer\compute-sanitizer.exe"
set SLZ=%~dp0..\zig-out\bin\streamlz.exe
set WEB=%~dp0..\assets\web.txt
set TOOL=memcheck
if /i "%1"=="racecheck" set TOOL=racecheck

if not exist %SAN% (
    echo error: compute-sanitizer not found at %SAN%
    exit /b 1
)
if not exist "%SLZ%" (
    echo error: build streamlz.exe first with zig build -Doptimize=ReleaseFast
    exit /b 1
)

echo -- %TOOL%: L1 roundtrip --------------------------------------
%SAN% --tool %TOOL% --error-exitcode 1 "%SLZ%" -b -l 1 -r 1 "%WEB%"
if errorlevel 1 goto :fail

echo -- %TOOL%: L5 roundtrip (huffman + chain parser + decode) ----
%SAN% --tool %TOOL% --error-exitcode 1 "%SLZ%" -b -l 5 -r 1 "%WEB%"
if errorlevel 1 goto :fail

echo.
echo sanitize PASS (%TOOL%)
exit /b 0

:fail
echo.
echo sanitize FAIL (%TOOL%) - see report above
exit /b 1
