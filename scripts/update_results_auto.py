import os
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ODDS_API_KEY = os.getenv("ODDS_API_KEY")

TRACK_PATH = Path("tracking_results.csv")
MANUAL_RESULTS_PATH = Path("manual_results.csv")
LEARNING_PATH = Path("data/learning/ai_learning_profile.csv")
LEARNING_SUMMARY_PATH = Path("data/learning/ai_learning_summary.csv")
BASE_URL = "https://api.the-odds-api.com/v4"


def safe_str(x):
    text = str(x).lower().strip()

    if text in ["nan", "none"]:
        return ""

    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", "", text)

    replacements = {
        "paris saint germain": "psg",
        "paris sg": "psg",
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


def names_match(a, b):
    a = safe_str(a)
    b = safe_str(b)

    if not a or not b:
        return False

    return a == b or a in b or b in a


def get_score_number(score):
    try:
        return float(score)
    except Exception:
        return None


def ensure_manual_results_template():
    if MANUAL_RESULTS_PATH.exists():
        return

    template = pd.DataFrame(
        columns=[
            "date",
            "sport",
            "home_team",
            "away_team",
            "winner",
            "score_home",
            "score_away",
            "source",
            "notes",
        ]
    )

    template.to_csv(MANUAL_RESULTS_PATH, index=False)


def load_manual_results():
    ensure_manual_results_template()

    try:
        df = pd.read_csv(MANUAL_RESULTS_PATH, low_memory=False)
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    for col in ["home_team", "away_team", "winner"]:
        if col not in df.columns:
            df[col] = ""

    return df


def find_manual_result(row, manual_df):
    if manual_df.empty:
        return None

    home = row.get("home_team", "")
    away = row.get("away_team", "")

    for _, manual in manual_df.iterrows():
        m_home = manual.get("home_team", "")
        m_away = manual.get("away_team", "")

        same_match = names_match(home, m_home) and names_match(away, m_away)
        reversed_match = names_match(home, m_away) and names_match(away, m_home)

        if same_match or reversed_match:
            winner = manual.get("winner", "")

            if pd.notna(winner) and str(winner).strip():
                score_home = manual.get("score_home", None)
                score_away = manual.get("score_away", None)

                return str(winner), score_home, score_away

    return None


def fetch_scores(sport):
    url = f"{BASE_URL}/sports/{sport}/scores"

    params = {
        "apiKey": ODDS_API_KEY,
        "daysFrom": 3,
    }

    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    except Exception as e:
        print(f"Erreur scores {sport} :", e)
        return []


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
        errors="coerce",
    )

    odds = pd.to_numeric(
        row.get("bookmaker_odds", 0),
        errors="coerce",
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


def mode_group(mode):
    mode = str(mode).upper().strip()

    if "MEGA" in mode or "MONSTER" in mode:
        return "MEGA VALUE"

    if "ULTRA SAFE" in mode or mode == "SAFE":
        return "SAFE"

    if "VALUE" in mode:
        return "MEDIUM"

    if "AGGRESSIVE" in mode or "RISKY" in mode:
        return "RISKY"

    return "OTHER"


def build_learning_profile(tracking):
    LEARNING_PATH.parent.mkdir(parents=True, exist_ok=True)

    finished = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()

    if finished.empty:
        pd.DataFrame().to_csv(LEARNING_PATH, index=False)
        pd.DataFrame([{
            "status": "NO_FINISHED_BETS",
            "message": "Pas encore assez de résultats pour apprendre.",
        }]).to_csv(LEARNING_SUMMARY_PATH, index=False)
        return

    if "bet_mode" not in finished.columns:
        finished["bet_mode"] = "UNKNOWN"

    if "category" not in finished.columns:
        finished["category"] = "unknown"

    finished["mode_group"] = finished["bet_mode"].apply(mode_group)
    finished["stake"] = pd.to_numeric(finished.get("stake", 0), errors="coerce").fillna(0)
    finished["profit"] = pd.to_numeric(finished.get("profit", 0), errors="coerce").fillna(0)
    finished["bookmaker_odds"] = pd.to_numeric(finished.get("bookmaker_odds", 0), errors="coerce").fillna(0)

    groups = []

    dimensions = [
        ("mode_group", "Mode de pari"),
        ("category", "Sport"),
        ("sport", "Compétition"),
        ("market", "Marché"),
    ]

    for col, label in dimensions:
        if col not in finished.columns:
            continue

        grouped = (
            finished
            .groupby(col)
            .agg(
                bets=("result", "count"),
                wins=("result", lambda x: (x == "WIN").sum()),
                losses=("result", lambda x: (x == "LOSS").sum()),
                stake=("stake", "sum"),
                profit=("profit", "sum"),
                avg_odds=("bookmaker_odds", "mean"),
            )
            .reset_index()
            .rename(columns={col: "segment"})
        )

        grouped["dimension"] = label
        grouped["winrate"] = grouped["wins"] / grouped["bets"]
        grouped["roi"] = grouped.apply(
            lambda r: r["profit"] / r["stake"] if r["stake"] > 0 else 0,
            axis=1,
        )

        def recommendation(r):
            if r["bets"] < 3:
                return "WAIT_MORE_DATA"
            if r["roi"] > 0.15 and r["winrate"] >= 0.65:
                return "BOOST"
            if r["roi"] < -0.10 or r["winrate"] < 0.45:
                return "REDUCE"
            return "KEEP"

        grouped["ai_recommendation"] = grouped.apply(recommendation, axis=1)
        groups.append(grouped)

    profile = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    profile.to_csv(LEARNING_PATH, index=False)

    total_bets = len(finished)
    total_profit = finished["profit"].sum()
    total_stake = finished["stake"].sum()
    global_roi = total_profit / total_stake if total_stake > 0 else 0
    global_winrate = (finished["result"] == "WIN").mean()

    summary = pd.DataFrame([{
        "finished_bets": total_bets,
        "profit": round(total_profit, 2),
        "stake": round(total_stake, 2),
        "roi": round(global_roi, 4),
        "winrate": round(global_winrate, 4),
        "best_action": "BOOST les segments rentables, REDUCE les segments négatifs.",
    }])

    summary.to_csv(LEARNING_SUMMARY_PATH, index=False)


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
            errors="coerce",
        ).fillna(0.0).astype(float)

    tracking["profit"] = pd.to_numeric(
        tracking.get("profit", 0),
        errors="coerce",
    ).fillna(0.0).astype(float)

    for col in [
        "final_winner",
        "final_score_home",
        "final_score_away",
        "status_detail",
    ]:
        if col not in tracking.columns:
            tracking[col] = ""

    manual_df = load_manual_results()

    pending = tracking[
        ~tracking["result"].isin(["WIN", "LOSS", "VOID"])
    ].copy()

    if pending.empty:
        print("Aucun pari en attente.")
        build_learning_profile(tracking)
        return

    sports = pending["sport"].dropna().unique()

    scores_by_sport = {}

    for sport in sports:
        if "tennis" in str(sport).lower():
            print("Tennis détecté : manuel + historique tennis :", sport)
            scores_by_sport[sport] = []
            continue

        print("Récupération scores :", sport)
        scores_by_sport[sport] = fetch_scores(sport)

    updated = 0
    manual_updated = 0
    not_found = 0

    for idx, row in tracking.iterrows():
        if row.get("result") in ["WIN", "LOSS", "VOID"]:
            continue

        manual_result = find_manual_result(row, manual_df)

        if manual_result is not None:
            winner, score_home, score_away = manual_result
            tracking.at[idx, "status_detail"] = "RESOLVED_FROM_MANUAL_RESULTS"
            manual_updated += 1

        else:
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
            tracking.at[idx, "status_detail"] = "RESULT_NOT_FOUND_VERIFY_MANUALLY"
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
                winner,
            )

    tracking.to_csv(TRACK_PATH, index=False)
    build_learning_profile(tracking)

    print("Résultats mis à jour :", updated)
    print("Résultats manuels utilisés :", manual_updated)
    print("Résultats à vérifier manuellement :", not_found)
    print("Auto-learning mis à jour :", LEARNING_PATH)


if __name__ == "__main__":
    main()
