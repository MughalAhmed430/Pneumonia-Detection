@echo off
echo =======================================================
echo Starting CheXNet AI Medical Diagnostic Platform...
echo =======================================================
cd /d "%~dp0\CheXNet"
start http://localhost:5000
python app.py
pause
