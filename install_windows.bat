@echo off
title Installation IA Paris Sportifs Ultime
set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    echo Python introuvable. Installe Python 3 depuis https://www.python.org/downloads/
    echo Pendant l'installation, coche "Add Python to PATH".
    pause
    exit /b 1
)
echo ============================================
echo Installation des librairies Python
echo ============================================
%PYTHON_CMD% -m pip install --upgrade pip
%PYTHON_CMD% -m pip install -r requirements.txt
echo.
echo Installation terminee.
echo Si Python ne marche pas, essaye: py -m pip install -r requirements.txt
pause
