@echo off
REM Daily Auto-Prediction Script
REM Run this every day to predict tomorrow's numbers

echo ========================================
echo    4D Auto-Learning Prediction System
echo ========================================
echo.

cd /d "%~dp0"

REM Activate virtual environment if exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Run prediction
python auto_predictor.py predict

echo.
echo ========================================
echo    Prediction Complete!
echo ========================================
pause
