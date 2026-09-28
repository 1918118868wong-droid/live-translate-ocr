@echo off
rem  live-translate-ocr : pick the subtitle region on screen.
rem  Pure ASCII on purpose - see setup.bat.
rem
rem  Drag a box around the English subtitles only. When you let go it grabs a
rem  frame, runs recognition once, and shows you what it read - so you know
rem  right away whether the box is right, instead of guessing while watching
rem  a stream. The result is written into config.toml.
rem
rem  Also reachable as: run.bat --pick  (and Ctrl+Alt+R while running)
chcp 65001 >nul
cd /d "%~dp0"

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

%PY% -m livetranslate_ocr.regionpick %*
pause
