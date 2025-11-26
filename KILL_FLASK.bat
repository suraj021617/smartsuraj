@echo off
echo Killing all Python processes...
taskkill /F /IM python.exe
timeout /t 2
echo.
echo Checking port 5000...
netstat -ano | findstr :5000
echo.
echo If you see any processes above, run this again!
echo Otherwise, you can now run: python app.py
pause
