@echo off
title IA Paris Sportifs - FULL AUTO
echo ============================================
echo 1/3 Update data
echo ============================================
python scripts\update_data.py
echo ============================================
echo 2/3 Pipeline predictions
echo ============================================
python scripts\run_pipeline.py
echo ============================================
echo 3/3 Export Excel
echo ============================================
python scripts\export_to_excel.py
echo.
echo Termine. Ouvre systeme_prediction_ultime.xlsx ou lance run_dashboard.bat
pause
