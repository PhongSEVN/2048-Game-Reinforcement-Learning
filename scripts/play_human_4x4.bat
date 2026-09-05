@echo off
REM Play 2048 yourself in the GUI with the arrow keys.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.gui --size 4 --human %*
