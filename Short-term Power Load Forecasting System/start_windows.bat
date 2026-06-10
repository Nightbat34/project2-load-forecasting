@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=D:\Anaconda\envs\pytorch_gpu\python.exe"
if not exist "%PYTHON_EXE%" (
  set "PYTHON_EXE=python"
)

echo [1/2] Starting backend API at http://127.0.0.1:5000 ...
start "Power Load Forecast Backend" cmd /k ""%PYTHON_EXE%" "%~dp0backend\app.py""

echo [2/2] Opening frontend dashboard ...
start "" "%~dp0frontend\index.html"

echo.
echo Backend health check: http://127.0.0.1:5000/api/health
echo Frontend entry: %~dp0frontend\index.html
echo.
pause
