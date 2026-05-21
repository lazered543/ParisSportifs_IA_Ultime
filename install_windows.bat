@echo off
title Installation IA Paris Sportifs Ultime
echo ============================================
echo Installation des librairies Python
echo ============================================
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Installation terminee.
echo Si Python ne marche pas, essaye: py -m pip install -r requirements.txt
pause
