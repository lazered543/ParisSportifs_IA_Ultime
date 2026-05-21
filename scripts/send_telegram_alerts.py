import os
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

bets_path = Path("data/predictions/value_bets_today.csv")
sent_path = Path("data/telegram_sent.csv")

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

if sent_path.exists():
    sent_df = pd.read_csv(sent_path)
else:
    sent_df = pd.DataFrame(columns=["match_id"])

if "match_id" not in sent_df.columns:
    sent_df["match_id"] = []

df = df.sort_values("value", ascending=False).head(10)


def send_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
        },
        timeout=20,
    )

    if response.status_code == 200:
        print("Message envoyé.")
        return True

    print("Erreur Telegram :", response.text)
    return False


new_rows = []

for _, row in df.iterrows():
    match_id = (
        f"{row.get('date', '')}_"
        f"{row['home_team']}_"
        f"{row['away_team']}_"
        f"{row['market']}"
    )

    if match_id in sent_df["match_id"].astype(str).values:
        print("Déjà envoyé :", match_id)
        continue

    new_rows.append((match_id, row))

if not new_rows:
    print("Aucune nouvelle alerte Telegram à envoyer.")
    raise SystemExit

intro = f"""
━━━━━━━━━━━━━━━━━━━━
🤖 <b>IA PARIS SPORTIFS</b>
━━━━━━━━━━━━━━━━━━━━

🔥 <b>{len(new_rows)} NOUVELLE(S) ALERTE(S)</b>

📅 Mise à jour automatique
⚠️ Paper betting conseillé au début
━━━━━━━━━━━━━━━━━━━━
"""

send_message(intro)

for match_id, row in new_rows:
    value_pct = round(float(row["value"]) * 100, 2)
    prob_pct = round(float(row["ai_probability"]) * 100, 2)

    badge = row.get("ia_badge", "⚪ NO VALUE")
    score = row.get("score_exact_1", "N/A")
    score_proba = row.get("score_exact_1_proba", "N/A")
    draw = row.get("draw_hunter", "N/A")
    scorer = row.get("scorer_prediction", "N/A")

    monster_bet = value_pct >= 25 and prob_pct >= 70

    if monster_bet:
        title = "💣 MONSTER BET"
    elif value_pct >= 25:
        title = "🚨 ULTRA VALUE BET"
    elif value_pct >= 10:
        title = "🔥 STRONG VALUE BET"
    else:
        title = "✅ VALUE BET"

    message = f"""
━━━━━━━━━━━━━━━━━━━━
{title}
━━━━━━━━━━━━━━━━━━━━

⚽ <b>{row['home_team']} vs {row['away_team']}</b>

🏆 <b>Compétition :</b>
{row.get('sport', 'N/A')}

🎯 <b>Pari conseillé :</b>
{row['market']}

📊 <b>Probabilité IA :</b>
{prob_pct}%

💰 <b>Cote bookmaker :</b>
{row['bookmaker_odds']}

💎 <b>Value :</b>
+{value_pct}%

🧠 <b>Badge IA :</b>
{badge}

📌 <b>Confiance :</b>
{row['confidence']}

💸 <b>Stake conseillé :</b>
{row['suggested_stake']}€

🎯 <b>Score exact probable :</b>
{score} — {score_proba}%

🤝 <b>Draw Hunter :</b>
{draw}

🎯 <b>ANYTIME GOALSCORER IA :</b>
{scorer}

━━━━━━━━━━━━━━━━━━━━
"""

    if send_message(message):
        sent_df.loc[len(sent_df)] = [match_id]

summary = """
━━━━━━━━━━━━━━━━━━━━
✅ <b>FIN DES NOUVELLES ALERTES IA</b>
━━━━━━━━━━━━━━━━━━━━

📲 Dashboard mis à jour
📈 Tracking mis à jour
🤖 IA prête pour la prochaine analyse

━━━━━━━━━━━━━━━━━━━━
"""

send_message(summary)

sent_path.parent.mkdir(parents=True, exist_ok=True)
sent_df.drop_duplicates(subset=["match_id"], keep="last").to_csv(sent_path, index=False)

print("Anti-spam mis à jour :", sent_path)