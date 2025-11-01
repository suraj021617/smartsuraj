@echo off
echo ========================================
echo  SMART 4D LEARNING SYSTEM
echo ========================================
echo.
echo Choose an option:
echo.
echo 1. Start Flask App (View Dashboard)
echo 2. Add Result and Learn
echo 3. Auto-Evaluate All Predictions
echo 4. Test System
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting Flask App...
    python app.py
) else if "%choice%"=="2" (
    echo.
    python add_result_and_learn.py
    pause
) else if "%choice%"=="3" (
    echo.
    python auto_evaluate.py
    pause
) else if "%choice%"=="4" (
    echo.
    python test_learning_system.py
    pause
) else if "%choice%"=="5" (
    exit
) else (
    echo Invalid choice!
    pause
)
