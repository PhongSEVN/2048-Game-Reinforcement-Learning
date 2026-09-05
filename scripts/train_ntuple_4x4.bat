@echo off
REM Train the n-tuple afterstate agent on 4x4. Learns fast per-episode but
REM episodes are long - stop it (close window) once you are happy, the best
REM checkpoint is saved continuously to runs\ntuple_4x4\best.npz
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.train_afterstate --size 4 --episodes 20000 --eps-decay-steps 80000 --log-every 200 %*
echo.
pause
