@echo off
REM Random-move baseline on 3x3 - the score the agent must beat.
setlocal
cd /d "%~dp0.."
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m twenty48.play --size 3 --random --games 300 --quiet %*
echo.
pause
