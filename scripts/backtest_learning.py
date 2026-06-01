from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import (  # noqa: E402
    build_elo_ratings,
    build_team_strength,
    probability_bucket,
    process_football_match,
)


FOOTBALL_HISTORY_PATH = ROOT / "data" / "processed" / "football_history_all.csv"
TRACKING_PATH = ROOT / "tracking_results.csv"
LEARNING_DIR = ROOT / "data" / "learning"
BACKTEST_PATH = LEARNING_DIR / "football_backtest.csv"
SUMMARY_PATH = LEARNING_DIR / "backtest_summary.csv"
CALIBRATION_PATH = LEARNING_DIR / "probability_calibration.csv"
SEGMENTS_PATH = LEARNING_DIR / "ai_auto_learning_segments.csv"


def parse_match_datetime(history):
    date = pd.to_datetime(history.get("Date", ""), dayfirst=True, errors="coerce")
    if "Time" in history.columns:
        time_text = history["Time"].fillna("00:00").astype(str)
        combined = history["Date"].astype(str) + " " + time_text
        combined_dt = pd.to_datetime(combined, dayfirst=True, errors="coerce")
        date = combined_dt.fillna(date)
    return date


def actual_selection(result, home, away):
    if result == "H":
        return "Home Win", home
    if result == "A":
        return "Away Win", away
    return "Draw", "Draw"


def historical_fixture(row):
    return {
        "sport": str(row.get("LeagueCode", row.get("Div", "historical_football"))),
        "commence_time": row.get("_dt"),
        "home_team": row.get("HomeTeam", ""),
        "away_team": row.get("AwayTeam", ""),
        "odds_home": row.get("B365H"),
        "odds_draw": row.get("B365D"),
        "odds_away": row.get("B365A"),
        "source": "historical-backtest",
    }


def run_football_backtest(max_matches=350):
    if not FOOTBALL_HISTORY_PATH.exists():
        return pd.DataFrame()

    history = pd.read_csv(FOOTBALL_HISTORY_PATH, low_memory=False)
    needed = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
    for col in needed:
        if col not in history.columns:
            return pd.DataFrame()

    history = history.dropna(subset=needed).copy()
    history = history[history["FTR"].isin(["H", "D", "A"])].copy()
    history["_dt"] = parse_match_datetime(history)
    history = history.sort_values("_dt").reset_index(drop=True)
    if len(history) < 80:
        return pd.DataFrame()

    start = max(60, len(history) - max_matches)
    rows = []
    context = history.iloc[:start].copy()
    if len(context) < 60:
        context = history.iloc[: max(start, 60)].copy()

    strengths = build_team_strength(context)
    ratings = build_elo_ratings(context)

    for idx in range(start, len(history)):
        match = history.iloc[idx]
        predictions = process_football_match(historical_fixture(match), strengths, ratings)

        market_won, selection_won = actual_selection(match.get("FTR"), match.get("HomeTeam"), match.get("AwayTeam"))
        for pred in predictions:
            won = pred.get("market") == market_won and pred.get("selection") == selection_won
            pred["result"] = "WIN" if won else "LOSS"
            pred["won"] = int(won)
            pred["stake"] = pred.get("suggested_stake", 0)
            pred["profit"] = (
                pred["stake"] * (pred.get("bookmaker_odds", 0) - 1)
                if won else -pred["stake"]
            )
            pred["flat_stake"] = 1.0
            pred["flat_profit"] = pred.get("bookmaker_odds", 0) - 1 if won else -1.0
            pred["backtest_match_date"] = match.get("_dt")
            pred["actual_result"] = match.get("FTR")
            rows.append(pred)

    return pd.DataFrame(rows)


def tracking_finished_rows():
    if not TRACKING_PATH.exists():
        return pd.DataFrame()

    tracking = pd.read_csv(TRACKING_PATH, low_memory=False)
    if tracking.empty or "result" not in tracking.columns:
        return pd.DataFrame()

    tracking["result"] = tracking["result"].fillna("").astype(str).str.upper()
    finished = tracking[tracking["result"].isin(["WIN", "LOSS"])].copy()
    if finished.empty:
        return pd.DataFrame()

    finished["won"] = (finished["result"] == "WIN").astype(int)
    if "category" not in finished.columns:
        finished["category"] = finished.get("sport", "").astype(str)
    return finished


def calibration_rows(source):
    if source.empty:
        return pd.DataFrame()

    data = source.copy()
    for col in ["ai_probability", "bookmaker_odds", "won"]:
        data[col] = pd.to_numeric(data.get(col, 0), errors="coerce")
    data = data.dropna(subset=["ai_probability", "won"])
    data = data[(data["ai_probability"] > 0) & (data["ai_probability"] < 1)].copy()
    if data.empty:
        return pd.DataFrame()

    data["probability_bucket"] = data["ai_probability"].apply(probability_bucket)

    groups = []
    for keys in [["category", "market", "probability_bucket"], ["category", "probability_bucket"]]:
        grouped = (
            data.groupby(keys)
            .agg(
                bets=("won", "count"),
                wins=("won", "sum"),
                avg_probability=("ai_probability", "mean"),
                avg_odds=("bookmaker_odds", "mean"),
            )
            .reset_index()
        )
        if "market" not in grouped.columns:
            grouped["market"] = "__all__"
        groups.append(grouped)

    calibration = pd.concat(groups, ignore_index=True)
    prior = 8
    calibration["smoothed_winrate"] = (
        calibration["wins"] + calibration["avg_probability"] * prior
    ) / (calibration["bets"] + prior)
    calibration["raw_winrate"] = calibration["wins"] / calibration["bets"]
    calibration["additive_adjustment"] = (
        calibration["smoothed_winrate"] - calibration["avg_probability"]
    ).clip(-0.06, 0.04)
    calibration["action"] = calibration["additive_adjustment"].apply(
        lambda x: "REDUCE" if x < -0.015 else "BOOST" if x > 0.015 else "KEEP"
    )

    ordered = [
        "category", "market", "probability_bucket", "bets", "wins",
        "raw_winrate", "smoothed_winrate", "avg_probability", "avg_odds",
        "additive_adjustment", "action",
    ]
    return calibration[ordered].sort_values(["category", "market", "probability_bucket"])


def summary_rows(backtest):
    if backtest.empty:
        return pd.DataFrame()

    data = backtest.copy()
    data["flat_stake"] = pd.to_numeric(data.get("flat_stake", 1), errors="coerce").fillna(1)
    data["flat_profit"] = pd.to_numeric(data.get("flat_profit", 0), errors="coerce").fillna(0)

    summaries = []
    for col in ["category", "market", "bet_mode", "odds_source"]:
        if col not in data.columns:
            continue
        grouped = (
            data.groupby(col)
            .agg(
                bets=("result", "count"),
                wins=("won", "sum"),
                stake=("flat_stake", "sum"),
                profit=("flat_profit", "sum"),
                avg_probability=("ai_probability", "mean"),
                avg_odds=("bookmaker_odds", "mean"),
            )
            .reset_index()
            .rename(columns={col: "segment"})
        )
        grouped["dimension"] = col
        grouped["winrate"] = grouped["wins"] / grouped["bets"]
        grouped["roi"] = grouped.apply(lambda r: r["profit"] / r["stake"] if r["stake"] > 0 else 0, axis=1)
        summaries.append(grouped)

    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def write_learning_segments(summary):
    if summary.empty:
        pd.DataFrame().to_csv(SEGMENTS_PATH, index=False)
        return

    segments = summary.copy()
    segments["recommendation"] = segments.apply(
        lambda r: "BOOST" if r["bets"] >= 20 and r["roi"] > 0.06 and r["winrate"] > 0.53
        else "REDUCE" if r["bets"] >= 20 and (r["roi"] < -0.06 or r["winrate"] < 0.47)
        else "KEEP",
        axis=1,
    )
    segments.to_csv(SEGMENTS_PATH, index=False)


def main():
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    backtest = run_football_backtest()
    tracking = tracking_finished_rows()
    combined = pd.concat([backtest, tracking], ignore_index=True, sort=False)

    backtest.to_csv(BACKTEST_PATH, index=False)
    calibration = calibration_rows(combined)
    calibration.to_csv(CALIBRATION_PATH, index=False)

    summary = summary_rows(backtest)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_learning_segments(summary)

    print("Backtest lignes :", len(backtest))
    print("Calibration segments :", len(calibration))
    if not summary.empty:
        print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
