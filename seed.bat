@echo off
rem  live-translate-ocr : warm up the local translation memory.
rem  Pure ASCII on purpose - see setup.bat.
rem
rem  Fills the memory with sentence pairs that then hit with zero latency and
rem  cost nothing: the common-word dictionary, plus whatever the old
rem  live-translate project already translated (read-only).
rem
rem  Run it again any time; already-present pairs are skipped.
chcp 65001 >nul
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% tools\seed_tm.py %*
pause
