@echo off
REM ═══════════════════════════════════════════════════════
REM  ProMe — one-command installer (Windows)
REM ═══════════════════════════════════════════════════════
REM Thin wrapper that delegates to prome_installer\ (Python).
REM All install logic lives in prome_installer\steps.py so the same code path
REM runs on macOS via install.sh.
REM
REM Usage:  install.bat
REM Idempotent - safe to run multiple times.

setlocal

cd /d "%~dp0"

REM Find a Python >= 3.11 to bootstrap the installer.
REM Prefer `py` launcher (comes with python.org installers), fall back to `python`.
set "PYTHON_EXE="

where py >nul 2>nul
if %ERRORLEVEL% == 0 (
    py -3.11 --version >nul 2>nul && set "PYTHON_EXE=py -3.11"
    if not defined PYTHON_EXE (
        py -3.12 --version >nul 2>nul && set "PYTHON_EXE=py -3.12"
    )
    if not defined PYTHON_EXE (
        py -3.13 --version >nul 2>nul && set "PYTHON_EXE=py -3.13"
    )
    if not defined PYTHON_EXE (
        py -3.14 --version >nul 2>nul && set "PYTHON_EXE=py -3.14"
    )
    if not defined PYTHON_EXE (
        py -3 --version >nul 2>nul && set "PYTHON_EXE=py -3"
    )
)

if not defined PYTHON_EXE (
    where python >nul 2>nul
    if %ERRORLEVEL% == 0 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>nul
        if %ERRORLEVEL% == 0 set "PYTHON_EXE=python"
    )
)

if not defined PYTHON_EXE (
    echo.
    echo Error: Python ^>= 3.11 is required but was not found.
    echo.
    echo Install it via winget:
    echo   winget install Python.Python.3.12
    echo.
    echo Or download from: https://www.python.org/downloads/windows/
    echo.
    exit /b 1
)

%PYTHON_EXE% -m prome_installer %*
exit /b %ERRORLEVEL%
