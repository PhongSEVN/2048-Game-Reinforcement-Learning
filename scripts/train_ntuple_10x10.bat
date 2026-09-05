@echo off
REM Train the n-tuple afterstate agent on 10x10 (81 square tuples, step cap).
REM Plateaus quickly - close the window when the log stops improving.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.train_afterstate --size 10 --episodes 1500 --max-steps 500 --lr 0.15 --log-every 100 %*
echo.
pause
