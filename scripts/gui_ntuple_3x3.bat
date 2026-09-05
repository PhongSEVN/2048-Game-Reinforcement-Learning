@echo off
REM Watch the trained 3x3 n-tuple agent play in the GUI.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.gui --size 3 --checkpoint runs\ntuple_3x3\best.npz %*
