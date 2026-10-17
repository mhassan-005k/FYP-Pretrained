@echo off
title GeoSR Studio - Satellite Super-Resolution
color 0B

echo ===============================================================================
echo            GeoSR Studio: Satellite Image Super-Resolution System
echo          SRS v1.2 Compliant | Hamdard University ^& SUPARCO FYP Project
echo ===============================================================================
echo.

set "PY_EXE=D:\ANACONDA\python.exe"

if not exist "%PY_EXE%" (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_EXE=python"
    ) else (
        echo [ERROR] Python not found! Please ensure Anaconda or Python is installed.
        pause
        exit /b 1
    )
)

echo [OK] Using Python: %PY_EXE%
echo [INFO] Starting GeoSR Studio local server...
echo [INFO] Opening your default web browser at http://localhost:8501
echo.

"%PY_EXE%" -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
