@echo off
REM StreamLZ benchmark script — downloads enwik8 if needed, runs -bc at t0, t1, t16.

set SLZ=%~dp0streamlz.exe
set ENWIK8=%~dp0enwik8

if not exist "%ENWIK8%" (
    echo Downloading enwik8 ^(34 MB^)...
    curl -L -o "%~dp0enwik8.zip" https://mattmahoney.net/dc/enwik8.zip
    tar -xf "%~dp0enwik8.zip" -C "%~dp0"
    del "%~dp0enwik8.zip"
)

echo.
echo === Auto threads ===
"%SLZ%" -bc -r 30 -t 0 "%ENWIK8%"

echo.
echo === Single thread ===
"%SLZ%" -bc -r 30 -t 1 "%ENWIK8%"

echo.
echo === 16 threads ===
"%SLZ%" -bc -r 30 -t 16 "%ENWIK8%"

pause
