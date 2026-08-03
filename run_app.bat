@echo off
title AI Relief App Launcher

:: Change directory to the folder where this batch file is located
cd /d "%~dp0"

:: Check if the virtual environment exists
if not exist ".venv\Scripts\activate.bat" goto :NO_VENV

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting AI Relief Application in Hot Reload Mode...
gradio ai_relief/gui/app.py

pause
exit /b 0

:NO_VENV
echo Error: Virtual environment .venv not found in %~dp0
echo Please ensure the .venv folder exists.
pause
exit /b 1
