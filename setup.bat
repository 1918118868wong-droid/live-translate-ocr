@echo off
rem ============================================================
rem  live-translate-ocr : environment setup
rem ------------------------------------------------------------
rem  This file is intentionally PURE ASCII. cmd.exe reads batch
rem  files byte by byte; non-ASCII text breaks line parsing and
rem  produces errors like "'cho' is not recognized".
rem  All Chinese messages live in tools\bootstrap.py instead.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    where py >nul 2>nul && set "PY=py -3"
)
if not defined PY goto nopython

%PY% tools\bootstrap.py %*
goto end

:nopython
echo.
echo [ERROR] Python not found on this machine.
echo.
echo   Install Python 3.11 or newer:
echo     https://www.python.org/downloads/
echo.
echo   During install, tick "Add python.exe to PATH",
echo   then close this window and run setup.bat again.
echo.

:end
pause
