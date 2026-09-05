@echo off
REM Create a local virtual env and install dependencies.
REM Optional - the other .bat files also work with a system Python that
REM already has numpy / matplotlib installed.
setlocal
cd /d "%~dp0.."
python -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup done. You can now run the other scripts in this folder.
pause
