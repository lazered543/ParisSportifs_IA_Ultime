# TUTO COMPLET - IA Paris Sportifs Ultime

## 1. Ce que fait le système

Le système :
- télécharge l'historique football de la saison passée + saison actuelle via Football-Data,
- télécharge l'historique tennis ATP/WTA via les bases Jeff Sackmann,
- récupère les cotes à venir via The Odds API si tu mets une clé API,
- calcule les probabilités IA,
- compare les probabilités IA aux cotes bookmakers,
- détecte les value bets,
- calcule une mise conseillée avec quart Kelly,
- affiche tout dans un dashboard,
- exporte un CSV et un Excel.

## 2. Liens utiles

Python :
https://www.python.org/downloads/

VS Code :
https://code.visualstudio.com/

API-Football :
https://www.api-football.com/

The Odds API :
https://the-odds-api.com/

Football-Data :
https://www.football-data.co.uk/data.php

Tennis ATP data :
https://github.com/JeffSackmann/tennis_atp

Tennis WTA data :
https://github.com/JeffSackmann/tennis_wta

## 3. Installation

### Étape A
Dézippe le dossier sur ton Bureau.

### Étape B
Double-clique sur :

install_windows.bat

### Étape C
Copie le fichier :

.env.example

Renomme la copie en :

.env

Puis colle tes clés API.

## 4. Utilisation quotidienne

Tous les matins :

1. update_data.bat
2. run_full_pipeline.bat
3. run_dashboard.bat
4. ouvrir systeme_prediction_ultime.xlsx

## 5. Automatisation Windows

Pour automatiser à 9h tous les jours :
- clic droit sur Windows PowerShell
- exécuter en administrateur
- va dans le dossier scripts
- lance :

powershell -ExecutionPolicy Bypass -File setup_task_scheduler.ps1

## 6. Comprendre la décision

- VALUE BET = le modèle estime que la cote est supérieure à la vraie probabilité.
- NO BET = le modèle ne voit pas assez d'avantage.
- suggested_stake = mise conseillée, limitée pour éviter les gros risques.

## 7. Règles de sécurité

- Ne jamais jouer tous les matchs.
- Ne jamais all-in.
- Mise max recommandée : 1 à 3% de bankroll.
- Ne joue pas si tu ne comprends pas pourquoi le modèle propose le pari.
- Le but est le long terme, pas un ticket miracle.
