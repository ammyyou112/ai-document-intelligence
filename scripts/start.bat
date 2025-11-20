@echo off
REM Change to project root directory
cd /d %~dp0\..
if errorlevel 1 (
    echo Error: Could not change to project root directory
    pause
    exit /b 1
)

echo Starting AI Document OCR Application...
echo.
python app.py
pause


