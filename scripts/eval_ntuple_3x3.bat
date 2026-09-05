@echo off
REM Play 200 greedy games with the trained 3x3 n-tuple agent and print stats.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.play --size 3 --checkpoint runs\ntuple_3x3\best.npz --games 200 --quiet %*
echo.
pause
