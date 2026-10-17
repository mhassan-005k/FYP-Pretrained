@echo off
title GeoSR Studio - Automatic Installer
color 0A

echo ===============================================================================
echo            GeoSR Studio: Automatic Offline/Online Dependency Installer
echo ===============================================================================
echo.

set "PY_EXE=D:\ANACONDA\python.exe"

if not exist "%PY_EXE%" (
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PY_EXE=python"
    ) else (
        echo [ERROR] Python not found on this system!
        pause
        exit /b 1
    )
)

echo [OK] Using Python: %PY_EXE%
echo [INFO] Installing required dependencies from requirements.txt...
echo.

"%PY_EXE%" -m pip install -r requirements.txt

echo.
echo ===============================================================================
echo [SUCCESS] All dependencies are ready!
echo You can now launch the application by double-clicking 'run_app.bat'.
echo ===============================================================================
echo.
pause
