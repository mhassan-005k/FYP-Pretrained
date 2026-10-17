@echo off
setlocal enabledelayedexpansion
title GeoSR Studio - Satellite Super-Resolution
color 0B

echo ===============================================================================
echo            GeoSR Studio: Satellite Image Super-Resolution System
echo          SRS v1.2 Compliant ^| Hamdard University ^& SUPARCO FYP Project
echo ===============================================================================
echo.

cd /d "%~dp0"

set "PY_EXE=D:\ANACONDA\python.exe"

if not exist "!PY_EXE!" (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PY_EXE=python"
    ) else (
        echo [ERROR] Python not found on your system!
        echo Please ensure Anaconda or Python 3.10+ is installed and added to PATH.
        pause
        exit /b 1
    )
)

echo [1/3] Using Python: !PY_EXE!

echo [2/3] Checking required libraries...
"!PY_EXE!" -c "import torch, streamlit, rasterio, PIL, skimage, safetensors, matplotlib" >nul 2>nul
if !errorlevel! neq 0 (
    echo [INFO] Missing packages detected! Automatically installing required libraries...
    echo [INFO] Running pip install -r requirements.txt...
    "!PY_EXE!" -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [WARNING] Some packages failed to install. Launching app anyway...
    ) else (
        echo [OK] All dependencies installed successfully!
    )
) else (
    echo [OK] All dependencies are already verified!
)

echo.
echo [3/3] Starting GeoSR Studio local server...
echo [INFO] Opening default browser at http://localhost:8501
echo.
echo -------------------------------------------------------------------------------
echo   Press Ctrl+C in this window at any time to stop the server.
echo -------------------------------------------------------------------------------
echo.

start "" http://localhost:8501

"!PY_EXE!" -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
