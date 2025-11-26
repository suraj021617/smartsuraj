@echo off
taskkill /F /IM python.exe 2>nul
timeout /t 1 /nobreak >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py
