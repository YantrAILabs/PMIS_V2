@echo off
REM Start all ProMe web services (Windows)
REM Usage: start.bat
REM
REM Launches three servers, each in its own cmd window so you can see logs:
REM   http://localhost:8100  -- ProMe API + Wiki (main dashboard)
REM   http://localhost:8200  -- Ops Dashboard (health, diagnostics)
REM   http://localhost:8000  -- Platform Portal (external integrations)
REM
REM Close any window to stop that service.

setlocal

set "REPO_DIR=%~dp0"
set "PYTHON=%REPO_DIR%productivity-tracker\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Error: Run install.bat first
    exit /b 1
)

echo.
echo   Starting ProMe services (three windows will open)...
echo     http://localhost:8100/wiki/goals   -- Main dashboard
echo     http://localhost:8200/             -- Ops Monitor
echo     http://localhost:8000/             -- Platform Portal
echo.

start "ProMe API + Wiki (8100)" cmd /k "cd /d %REPO_DIR%pmis_v2 && "%PYTHON%" server.py"
start "ProMe Ops Dashboard (8200)" cmd /k "cd /d %REPO_DIR%pmis_v2 && "%PYTHON%" health_dashboard.py"
start "ProMe Platform (8000)" cmd /k "cd /d %REPO_DIR%memory_system\platform && "%PYTHON%" server.py"

echo   All three services launched.
echo   Opening http://localhost:8100/wiki/goals in your default browser...
timeout /t 3 /nobreak >nul
start "" "http://localhost:8100/wiki/goals"

endlocal
