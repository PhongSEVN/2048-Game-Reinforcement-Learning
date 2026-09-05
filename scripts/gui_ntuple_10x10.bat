@echo off
REM Watch the trained 10x10 n-tuple agent play in the GUI.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.gui --size 10 --checkpoint runs\ntuple_10x10\best.npz %*
