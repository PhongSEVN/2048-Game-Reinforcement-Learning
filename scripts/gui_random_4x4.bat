@echo off
REM Watch a random-move baseline in the GUI (no checkpoint needed).
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.gui --size 4 --random %*
