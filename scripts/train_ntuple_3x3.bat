@echo off
REM Train the strong n-tuple afterstate agent on 3x3 (~30 min CPU).
REM Output: runs\ntuple_3x3\best.npz
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.train_afterstate --size 3 --episodes 20000 --eps-decay-steps 60000 --log-every 500 %*
echo.
pause
