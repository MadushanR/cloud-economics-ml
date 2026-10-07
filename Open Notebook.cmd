@echo off
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -m jupyterlab "%~dp0EC2_Analysis.ipynb"
pause
