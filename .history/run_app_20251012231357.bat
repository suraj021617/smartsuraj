@echo off
:: This batch file sets up the environment for development and runs the Flask app.
:: It's intended to be run from the project's root directory.

echo [+] Setting environment variables for development...
set FLASK_ENV=development
set FLASK_DEBUG=1

echo [+] Starting the Flask application...
python app.py

pause