@echo off
title IA Paris Sportifs - FULL AUTO
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
echo 1/6 Update data
echo ============================================
%PYTHON_CMD% scripts\update_data.py
echo ============================================
echo 2/6 Update player scorers
echo ============================================
%PYTHON_CMD% scripts\update_player_scorers.py
echo ============================================
echo 3/6 Pipeline predictions
echo ============================================
%PYTHON_CMD% scripts\run_pipeline.py
echo ============================================
echo 4/6 Save bets to tracking
echo ============================================
%PYTHON_CMD% scripts\save_bets_to_tracking.py
echo ============================================
echo 5/6 Update WIN / LOSS / ROI
echo ============================================
%PYTHON_CMD% scripts\update_results_auto.py
echo ============================================
echo 6/6 Export Excel
echo ============================================
%PYTHON_CMD% scripts\export_to_excel.py
echo.
echo Termine. Ouvre systeme_prediction_ultime.xlsx ou lance run_dashboard.bat
pause
