@echo off
rem  live-translate-ocr : offline self-check.  Pure ASCII - see setup.bat.
rem  Checks the caption state machine, both dictionaries, and the translation
rem  fallback chain. Does not touch the network or the screen.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% tools\selfcheck.py %*
pause
