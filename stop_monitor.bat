@echo off
taskkill /F /IM pythonw.exe 2>nul
if %errorlevel%==0 (
    echo Garage monitor stopped.
) else (
    echo No garage monitor process found.
)
