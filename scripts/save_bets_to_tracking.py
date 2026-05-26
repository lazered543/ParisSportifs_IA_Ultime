from pathlib import Path
import pandas as pd

pred_path = Path("data/predictions/value_bets_today.csv")
track_path = Path("tracking_results.csv")

if not pred_path.exists():
    print("Aucun fichier value_bets_today.csv trouvé.")
    raise SystemExit

bets = pd.read_csv(pred_path)

if bets.empty:
    print("Aucun value bet à tracker aujourd'hui.")
    raise SystemExit

cols = [
    "date",
    "sport",
    "home_team",
    "away_team",
    "market",
    "ai_probability",
    "bookmaker_odds",
    "value",
    "suggested_stake",
    "bet_mode",
    "stake_percent",
    "kelly_fraction",
    "bankroll",
]

bets = bets[cols].copy()
bets = bets.rename(columns={"suggested_stake": "stake"})
bets["result"] = "PENDING"
bets["profit"] = 0

if track_path.exists():
    existing = pd.read_csv(track_path)
    final = pd.concat([existing, bets], ignore_index=True)
    final = final.drop_duplicates(
        subset=["date", "home_team", "away_team", "market"],
        keep="last"
    )
else:
    final = bets

final.to_csv(track_path, index=False)

print("Paris ajoutés au tracking :", len(bets))
print("Fichier mis à jour : tracking_results.csv")