import os
import re
import unicodedata
import requests
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")

TRACK_PATH = Path("tracking_results.csv")
BASE_URL = "https://api.the-odds-api.com/v4"


def safe_str(x):
    text = str(x).lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", "", text)

    replacements = {
        "paris saint germain": "psg",
        "manchester united": "man utd",
        "manchester city": "man city",
        "internazionale": "inter",
        "fc internazionale milano": "inter",
        "bayern munich": "bayern",
        "real madrid cf": "real madrid",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return " ".join(text.split())


def get_score_number(score):
    try:
        return float(score)
    except Exception:
        return None


def fetch_scores(sport):
    url = f"{BASE_URL}/sports/{sport}/scores"

    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": 3
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"Erreur scores {sport} :", e)
        return []


def names_match(a, b):
    a = safe_str(a)
    b = safe_str(b)

    if not a or not b:
        return False

    return (
        a == b
        or a in b
        or b in a
    )


def find_matching_score(row, scores):
    home = row.get("home_team", "")
    away = row.get("away_team", "")

    for event in scores:
        event_home = event.get("home_team", "")
        event_away = event.get("away_team", "")

        same_match = names_match(home, event_home) and names_match(away, event_away)
        reversed_match = names_match(home, event_away) and names_match(away, event_home)

        if same_match or reversed_match:
            return event

    return None


def get_winner_from_event(event):
    if not event:
        return None, None, None

    if not event.get("completed", False):
        return None, None, None

    scores = event.get("scores", [])

    if not scores or len(scores) < 2:
        return None, None, None

    name_1 = scores[0].get("name")
    name_2 = scores[1].get("name")

    score_1 = get_score_number(scores[0].get("score"))
    score_2 = get_score_number(scores[1].get("score"))

    if score_1 is None or score_2 is None:
        return None, None, None

    if score_1 > score_2:
        return name_1, score_1, score_2

    if score_2 > score_1:
        return name_2, score_1, score_2

    return "DRAW", score_1, score_2


def try_resolve_tennis_from_history(row):
    hist_paths = [
        Path("data/processed/tennis_history_all.csv"),
        Path("data/processed/tennis_history.csv"),
        Path("data/raw/tennis_history.csv"),
    ]

    history = None

    for path in hist_paths:
        if path.exists():
            try:
                history = pd.read_csv(path, low_memory=False)
                break
            except Exception:
                pass

    if history is None or history.empty:
        return None, None

    home = safe_str(row.get("home_team", ""))
    away = safe_str(row.get("away_team", ""))

    winner_cols = [
        "winner",
        "Winner",
        "winner_name",
        "player_winner",
        "match_winner",
    ]

    for _, h in history.tail(10000).iterrows():
        line = " ".join([safe_str(v) for v in h.values])

        if home in line and away in line:
            for col in winner_cols:
                if col in history.columns:
                    winner = h.get(col)

                    if pd.notna(winner):
                        return str(winner), h

    return None, None


def evaluate_bet(row, winner):
    market = safe_str(row.get("market", ""))
    home = safe_str(row.get("home_team", ""))
    away = safe_str(row.get("away_team", ""))
    selection = safe_str(row.get("selection", ""))

    if winner is None:
        return "PENDING"

    winner_clean = safe_str(winner)

    if market == "home win":
        return "WIN" if names_match(winner_clean, home) else "LOSS"

    if market == "away win":
        return "WIN" if names_match(winner_clean, away) else "LOSS"

    if market == "draw":
        return "WIN" if winner_clean == "draw" else "LOSS"

    if market == "player 1 win":
        if selection:
            return "WIN" if names_match(winner_clean, selection) else "LOSS"
        return "WIN" if names_match(winner_clean, home) else "LOSS"

    if market == "player 2 win":
        if selection:
            return "WIN" if names_match(winner_clean, selection) else "LOSS"
        return "WIN" if names_match(winner_clean, away) else "LOSS"

    return "PENDING"


def calc_profit(row):
    result = row.get("result", "PENDING")

    stake = pd.to_numeric(
        row.get("stake", row.get("suggested_stake", 0)),
        errors="coerce"
    )

    odds = pd.to_numeric(
        row.get("bookmaker_odds", 0),
        errors="coerce"
    )

    if pd.isna(stake):
        stake = 0

    if pd.isna(odds):
        odds = 0

    if result == "WIN":
        return round(float(stake) * (float(odds) - 1), 2)

    if result == "LOSS":
        return round(-float(stake), 2)

    return 0.0

def main():
    if not ODDS_API_KEY:
        print("ODDS_API_KEY manquante.")
        return

    if not TRACK_PATH.exists():
        print("tracking_results.csv introuvable.")
        return

    tracking = pd.read_csv(TRACK_PATH)

    if tracking.empty:
        print("Tracking vide.")
        return

    if "result" not in tracking.columns:
        tracking["result"] = "PENDING"

    if "stake" not in tracking.columns:
        tracking["stake"] = pd.to_numeric(
            tracking.get("suggested_stake", 0),
            errors="coerce"
        ).fillna(0.0).astype(float)

    tracking["profit"] = pd.to_numeric(
        tracking.get("profit", 0),
        errors="coerce"
    ).fillna(0.0).astype(float)

    for col in ["final_winner", "final_score_home", "final_score_away", "status_detail"]:
        if col not in tracking.columns:
            tracking[col] = ""

    pending = tracking[
        ~tracking["result"].isin(["WIN", "LOSS"])
    ].copy()

    if pending.empty:
        print("Aucun pari en attente.")
        return

    sports = pending["sport"].dropna().unique()

    scores_by_sport = {}

    for sport in sports:
        if "tennis" in str(sport).lower():
            print("Tennis détecté : résolution via historique tennis :", sport)
            scores_by_sport[sport] = []
            continue

        print("Récupération scores :", sport)
        scores_by_sport[sport] = fetch_scores(sport)

    updated = 0
    not_found = 0

    for idx, row in tracking.iterrows():
        if row.get("result") in ["WIN", "LOSS"]:
            continue

        sport = row.get("sport", "")
        scores = scores_by_sport.get(sport, [])

        event = find_matching_score(row, scores)
        winner, score_home, score_away = get_winner_from_event(event)

        if winner is None and "tennis" in str(sport).lower():
            winner, _ = try_resolve_tennis_from_history(row)

            if winner is not None:
                score_home = None
                score_away = None
                tracking.at[idx, "status_detail"] = "RESOLVED_FROM_TENNIS_HISTORY"

        if winner is None:
            tracking.at[idx, "status_detail"] = "RESULT_NOT_FOUND"
            not_found += 1
            continue

        result = evaluate_bet(row, winner)

        if result in ["WIN", "LOSS"]:
            tracking.at[idx, "result"] = result
            tracking.at[idx, "final_winner"] = winner
            tracking.at[idx, "final_score_home"] = score_home
            tracking.at[idx, "final_score_away"] = score_away
            tracking.at[idx, "profit"] = calc_profit(tracking.loc[idx])

            if not tracking.at[idx, "status_detail"]:
                tracking.at[idx, "status_detail"] = "RESOLVED_FROM_ODDS_API"

            updated += 1

            print(
                row.get("home_team"),
                "vs",
                row.get("away_team"),
                "=>",
                result,
                "| gagnant :",
                winner
            )

    tracking.to_csv(TRACK_PATH, index=False)

    print("Résultats mis à jour :", updated)
    print("Résultats introuvables :", not_found)


if __name__ == "__main__":
    main()