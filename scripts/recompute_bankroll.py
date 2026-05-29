from pathlib import Path
from datetime import datetime
import pandas as pd

START_BANKROLL = 10.0
TRACK_PATH = Path("tracking_results.csv")
STATE_PATH = Path("data/bankroll_state.csv")

def main():
    current = START_BANKROLL
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    losses = 0
    finished_count = 0

    if TRACK_PATH.exists():
        tr = pd.read_csv(TRACK_PATH, low_memory=False)

        if not tr.empty:
            if "date" in tr.columns:
                tr["_dt"] = pd.to_datetime(tr["date"], errors="coerce", utc=True)
                tr = tr.sort_values("_dt")
            else:
                tr["_dt"] = pd.NaT

            if "result" not in tr.columns:
                tr["result"] = "PENDING"

            tr["result"] = tr["result"].fillna("PENDING").astype(str).str.upper()
            tr["stake"] = pd.to_numeric(tr.get("stake", 0), errors="coerce").fillna(0)
            tr["profit"] = pd.to_numeric(tr.get("profit", 0), errors="coerce").fillna(0)

            finished = tr[tr["result"].isin(["WIN", "LOSS"])].copy()

            total_staked = float(finished["stake"].sum()) if not finished.empty else 0.0
            total_profit = float(finished["profit"].sum()) if not finished.empty else 0.0
            wins = int((finished["result"] == "WIN").sum())
            losses = int((finished["result"] == "LOSS").sum())
            finished_count = int(len(finished))
            current = max(0.0, START_BANKROLL + total_profit)

    roi = total_profit / total_staked if total_staked > 0 else 0.0
    winrate = wins / finished_count if finished_count > 0 else 0.0

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "initial_bankroll": START_BANKROLL,
        "current_bankroll": round(current, 2),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "total_finished": finished_count,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 4),
        "roi": round(roi, 4),
        "updated_at": datetime.now().isoformat(),
    }]).to_csv(STATE_PATH, index=False)

    print("Bankroll actuelle :", round(current, 2), "€")
    print("Profit total :", round(total_profit, 2), "€")
    print("Winrate :", round(winrate * 100, 2), "%")
    print("ROI :", round(roi * 100, 2), "%")

if __name__ == "__main__":
    main()
