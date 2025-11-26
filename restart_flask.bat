@echo off
echo Stopping Flask...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul
echo Starting Flask...
cd /d c:\Users\Acer\Desktop\smartsuraj
start "Flask Server" cmd /k python app.py
echo.
echo Flask restarted! Wait 5 seconds then open: http://127.0.0.1:5000
timeout /t 5
