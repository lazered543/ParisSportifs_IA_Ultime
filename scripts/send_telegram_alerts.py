import os
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bets_path = Path("data/predictions/value_bets_today.csv")

if not TOKEN or not CHAT_ID:
    print("Token Telegram ou Chat ID manquant.")
    raise SystemExit

if not bets_path.exists():
    print("Aucun fichier value_bets_today.csv trouvé.")
    raise SystemExit

df = pd.read_csv(bets_path)

if df.empty:
    print("Aucun VALUE BET aujourd'hui.")
    raise SystemExit

sent = 0

for _, row in df.iterrows():
    badge = row.get("ia_badge", "")
    score = row.get("score_exact_1", "N/A")
    score_proba = row.get("score_exact_1_proba", "N/A")
    draw = row.get("draw_hunter", "")
    scorer = row.get("scorer_prediction", "")

    message = f"""
🔥 VALUE BET DÉTECTÉ

{badge}

🏟️ Match :
{row['home_team']} vs {row['away_team']}

🏆 Compétition :
{row.get('sport', 'N/A')}

🎯 Marché :
{row['market']}

📊 Probabilité IA :
{round(row['ai_probability'] * 100, 2)}%

💰 Cote :
{row['bookmaker_odds']}

📈 Value :
{round(row['value'] * 100, 2)}%

🧠 Confiance :
{row['confidence']}

💸 Stake conseillé :
{row['suggested_stake']}€

🎯 Score exact probable :
{score} — {score_proba}%

🤝 Draw Hunter :
{draw}

⚽ Buteur probable :
{scorer}

⚠️ Conseil : paper betting recommandé avant argent réel.
"""

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        },
        timeout=15
    )

    if response.status_code == 200:
        sent += 1
        print("Alerte envoyée :", row["home_team"], "vs", row["away_team"])
    else:
        print("Erreur Telegram :", response.text)

print(f"Alertes Telegram envoyées : {sent}")