@echo off
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
    python ejecutar.py
) else (
    py -3.12 ejecutar.py
)
if errorlevel 1 pause
