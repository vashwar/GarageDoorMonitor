@echo off
cd /d C:\VashwarTests\GarageCamera
start /B pythonw garage_monitor.py
echo Garage monitor started in background.
echo To stop it: taskkill /F /IM pythonw.exe
