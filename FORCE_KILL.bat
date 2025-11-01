@echo off
echo Killing specific Flask processes...
taskkill /F /PID 12948
taskkill /F /PID 29612
taskkill /F /PID 18936
taskkill /F /PID 17236
echo.
echo Killing all Python...
taskkill /F /IM python.exe
timeout /t 3
echo.
echo Checking if port 5000 is clear...
netstat -ano | findstr :5000
echo.
echo If nothing appears above, port is clear!
echo Now run: python app.py
pause
