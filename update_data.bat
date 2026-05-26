@echo off
title Update Data IA Paris Sportifs
set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    echo Python introuvable. Lance install_windows.bat apres avoir installe Python 3.
    pause
    exit /b 1
)
echo ============================================
echo Recuperation historique football + tennis
echo ============================================
%PYTHON_CMD% scripts\update_data.py
pause
