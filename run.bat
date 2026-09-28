@echo off
rem  live-translate-ocr : start.  Pure ASCII on purpose - see setup.bat.
rem  All user-facing messages are printed by Python.
rem
rem  Useful flags:
rem    run.bat --pick              pick the subtitle region (do this first)
rem    run.bat --selftest          just show the overlay, no capture, no network
rem    run.bat --text              type English, press Enter, see the translation
rem    run.bat --image shot.png    OCR a screenshot and translate it
rem    run.bat --list-monitors     show monitor geometry
chcp 65001 >nul
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

if not exist "config.toml" if exist "config.example.toml" copy "config.example.toml" "config.toml" >nul

%PY% -m livetranslate_ocr.main %*
set "RC=%errorlevel%"

if not "%RC%"=="0" (
    echo.
    pause
)
