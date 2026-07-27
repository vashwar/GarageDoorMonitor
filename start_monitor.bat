@echo off
cd /d "%~dp0"
start /B pythonw garage_monitor.py
echo Garage monitor started in background.
echo To stop it: taskkill /F /IM pythonw.exe
