import os
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bets_path = Path(
    "data/predictions/value_bets_today.csv"
)

if not TOKEN or not CHAT_ID:
    print(
        "Token Telegram ou Chat ID manquant."
    )
    raise SystemExit

if not bets_path.exists():
    print(
        "Aucun fichier value_bets_today.csv trouvé."
    )
    raise SystemExit

df = pd.read_csv(bets_path)

if df.empty:
    print(
        "Aucun VALUE BET aujourd'hui."
    )
    raise SystemExit

for _, row in df.iterrows():

    message = f"""
🔥 VALUE BET DETECTE

🏟️ Match :
{row['home_team']} vs {row['away_team']}

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
{row['suggested_stake']}
"""

    url = (
        f"https://api.telegram.org/bot"
        f"{TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": message
        }
    )

    if response.status_code == 200:

        print(
            "Alerte envoyée :",
            row["home_team"],
            "vs",
            row["away_team"]
        )

    else:

        print(
            "Erreur Telegram :",
            response.text
        )