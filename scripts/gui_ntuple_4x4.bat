@echo off
REM Watch the trained 4x4 n-tuple agent play in the GUI.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.gui --size 4 --checkpoint runs\ntuple_4x4\best.npz %*
