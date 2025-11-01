@echo off
REM Master Auto-Learning Lottery System
REM Complete pipeline: Predict -> Learn -> Improve

echo ========================================
echo   MASTER AUTO-LEARNING LOTTERY SYSTEM
echo ========================================
echo.

cd /d "%~dp0"

if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

:menu
echo.
echo Select an option:
echo 1. Predict Tomorrow
echo 2. Add Actual Result
echo 3. View Statistics
echo 4. Retrain Model
echo 5. Full Auto-Run (Predict + Stats)
echo 6. Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto predict
if "%choice%"=="2" goto add_result
if "%choice%"=="3" goto stats
if "%choice%"=="4" goto retrain
if "%choice%"=="5" goto auto_run
if "%choice%"=="6" goto end

:predict
echo.
echo Running prediction engine...
python prediction_engine.py
pause
goto menu

:add_result
echo.
set /p date="Enter draw date (YYYY-MM-DD): "
set /p numbers="Enter actual numbers (comma-separated): "
python learning_engine.py add_result %date% %numbers%
pause
goto menu

:stats
echo.
python learning_engine.py stats
pause
goto menu

:retrain
echo.
python learning_engine.py retrain
pause
goto menu

:auto_run
echo.
echo Running full auto-prediction...
python prediction_engine.py
echo.
echo Current statistics:
python learning_engine.py stats
pause
goto menu

:end
echo.
echo Goodbye!
pause
