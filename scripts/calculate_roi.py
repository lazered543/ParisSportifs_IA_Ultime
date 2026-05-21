import pandas as pd
from pathlib import Path

track_path = Path("tracking_results.csv")

if not track_path.exists():
    print("tracking_results.csv introuvable.")
    raise SystemExit

df = pd.read_csv(track_path)

if df.empty:
    print("Aucun pari tracké.")
    raise SystemExit

def calc_profit(row):
    if row["result"] == "WIN":
        return row["stake"] * (row["bookmaker_odds"] - 1)
    if row["result"] == "LOSS":
        return -row["stake"]
    return 0

df["profit"] = df.apply(calc_profit, axis=1)

settled = df[df["result"].isin(["WIN", "LOSS"])]

total_staked = settled["stake"].sum()
profit = settled["profit"].sum()
roi = profit / total_staked if total_staked > 0 else 0
hit_rate = (settled["result"] == "WIN").mean() if len(settled) > 0 else 0

df.to_csv(track_path, index=False)

print("Paris terminés :", len(settled))
print("Misé total :", round(total_staked, 2))
print("Profit :", round(profit, 2))
print("ROI :", round(roi * 100, 2), "%")
print("Hit rate :", round(hit_rate * 100, 2), "%")