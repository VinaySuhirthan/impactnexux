@echo off
REM AdGPT Server Launcher
REM Activates virtual environment and starts the server

echo.
echo ==========================================
echo   AdGPT Server Startup
echo ==========================================
echo.

REM Check if .venv exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Please run: python -m venv .venv
    pause
    exit /b 1
)

echo [+] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [+] Starting server...
python run.py

pause