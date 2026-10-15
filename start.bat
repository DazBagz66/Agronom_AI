@echo off
cd /d "%~dp0"
python -m uvicorn main:app --app-dir backend --reload
pause
