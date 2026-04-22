@echo off
REM Start the ProMe platform portal (Windows)
REM Usage: start.bat

setlocal

set "REPO_DIR=%~dp0"
set "PYTHON=%REPO_DIR%productivity-tracker\.venv\Scripts\python.exe"
set "SERVER=%REPO_DIR%memory_system\platform\server.py"

if not exist "%PYTHON%" (
    echo Error: Run install.bat first
    exit /b 1
)

echo.
echo   Starting ProMe...
echo   Portal:  http://localhost:8000
echo   API:     http://localhost:8000/docs
echo   Press Ctrl+C to stop
echo.

cd /d "%REPO_DIR%memory_system\platform"
"%PYTHON%" "%SERVER%"
