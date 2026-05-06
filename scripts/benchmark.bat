@echo off
REM StreamLZ benchmark script — downloads enwik8 if needed, runs -bc at t0, t1, t16.

set "SLZ=%~dp0streamlz.exe"
set "ENWIK8=%~dp0enwik8"
set "ZIPFILE=%~dp0enwik8.zip"

if not exist "%ENWIK8%" (
    echo Downloading enwik8 ^(34 MB^)...
    curl -L -o "%ZIPFILE%" https://mattmahoney.net/dc/enwik8.zip
    powershell -Command "Expand-Archive -Path '%ZIPFILE%' -DestinationPath '%~dp0' -Force"
    del "%ZIPFILE%"
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
