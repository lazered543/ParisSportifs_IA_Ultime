from __future__ import annotations

import sys
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import (  # noqa: E402
    build_elo_ratings,
    build_team_strength,
    clean_name,
    clamp,
    detect_surface,
    probability_bucket,
    process_football_match,
    safe_float,
    sigmoid,
)


FOOTBALL_HISTORY_PATH = ROOT / "data" / "processed" / "football_history_all.csv"
TENNIS_HISTORY_PATH = ROOT / "data" / "processed" / "tennis_history_all.csv"
TRACKING_PATH = ROOT / "tracking_results.csv"
LEARNING_DIR = ROOT / "data" / "learning"
BACKTEST_PATH = LEARNING_DIR / "football_backtest.csv"
TENNIS_BACKTEST_PATH = LEARNING_DIR / "tennis_backtest.csv"
SUMMARY_PATH = LEARNING_DIR / "backtest_summary.csv"
CALIBRATION_PATH = LEARNING_DIR / "probability_calibration.csv"
SEGMENTS_PATH = LEARNING_DIR / "ai_auto_learning_segments.csv"
THRESHOLD_PATH = LEARNING_DIR / "threshold_optimizer.csv"


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


def empty_player():
    return {
        "matches": 0,
        "wins": 0,
        "losses": 0,
        "elo": 1500.0,
        "recent": deque(maxlen=12),
        "surface": defaultdict(lambda: {"matches": 0, "wins": 0}),
        "rank": None,
        "rank_points": None,
    }


def update_tennis_rank(player, rank, points):
    if pd.notna(rank):
        player["rank"] = safe_float(rank, player["rank"] or 999)
    if pd.notna(points):
        player["rank_points"] = safe_float(points, player["rank_points"] or 0)


def update_tennis_players(players, match):
    winner_name = match.get("winner_name", "")
    loser_name = match.get("loser_name", "")
    w_key = clean_name(winner_name)
    l_key = clean_name(loser_name)
    if not w_key or not l_key:
        return

    winner = players[w_key]
    loser = players[l_key]
    expected_w = 1 / (1 + 10 ** ((loser["elo"] - winner["elo"]) / 400))
    expected_l = 1 - expected_w
    k = 26

    winner["elo"] += k * (1 - expected_w)
    loser["elo"] += k * (0 - expected_l)
    winner["matches"] += 1
    winner["wins"] += 1
    winner["recent"].append(1)
    loser["matches"] += 1
    loser["losses"] += 1
    loser["recent"].append(0)

    surface = str(match.get("surface", "") or "")
    if surface:
        winner["surface"][surface]["matches"] += 1
        winner["surface"][surface]["wins"] += 1
        loser["surface"][surface]["matches"] += 1

    update_tennis_rank(winner, match.get("winner_rank"), match.get("winner_rank_points"))
    update_tennis_rank(loser, match.get("loser_rank"), match.get("loser_rank_points"))


def tennis_player_stats(players, name, surface):
    player = players.get(clean_name(name))
    if player is None:
        return {
            "matches": 0,
            "elo": 1500.0,
            "winrate": 0.5,
            "form": 0.5,
            "rank": 999,
            "rank_points": 0,
            "surface_form": 0.5,
        }

    matches = max(player["matches"], 1)
    recent = list(player["recent"])
    surface_stats = player["surface"].get(surface) if surface else None
    surface_form = (
        surface_stats["wins"] / surface_stats["matches"]
        if surface_stats and surface_stats["matches"] > 0
        else 0.5
    )

    return {
        "matches": player["matches"],
        "elo": player["elo"],
        "winrate": player["wins"] / matches,
        "form": sum(recent) / len(recent) if recent else player["wins"] / matches,
        "rank": player["rank"] if player["rank"] is not None else 999,
        "rank_points": player["rank_points"] if player["rank_points"] is not None else 0,
        "surface_form": surface_form,
    }


def tennis_model_probability(home_stats, away_stats):
    rank_component = 0.0
    if home_stats["rank"] and away_stats["rank"]:
        rank_component = clamp(math.log((away_stats["rank"] + 12) / (home_stats["rank"] + 12)), -1.6, 1.6)

    model_logit = (
        (home_stats["elo"] - away_stats["elo"]) / 185
        + (home_stats["form"] - away_stats["form"]) * 1.15
        + (home_stats["surface_form"] - away_stats["surface_form"]) * 0.55
        + rank_component * 0.38
    )
    return clamp(sigmoid(model_logit), 0.08, 0.92)


def run_tennis_backtest(max_matches=700):
    if not TENNIS_HISTORY_PATH.exists():
        return pd.DataFrame()

    history = pd.read_csv(TENNIS_HISTORY_PATH, low_memory=False)
    needed = ["winner_name", "loser_name", "tourney_date"]
    for col in needed:
        if col not in history.columns:
            return pd.DataFrame()

    history = history.dropna(subset=needed).copy()
    history["_date"] = pd.to_numeric(history["tourney_date"], errors="coerce")
    history = history.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)
    if len(history) < 150:
        return pd.DataFrame()

    start = max(100, len(history) - max_matches)
    players = defaultdict(empty_player)
    for _, match in history.iloc[:start].iterrows():
        update_tennis_players(players, match)

    rows = []
    for idx in range(start, len(history)):
        match = history.iloc[idx]
        winner = str(match.get("winner_name", ""))
        loser = str(match.get("loser_name", ""))
        if not winner or not loser:
            continue

        flip = idx % 2 == 0
        home = loser if flip else winner
        away = winner if flip else loser
        actual_home_win = int(home == winner)
        surface = str(match.get("surface", "") or detect_surface(match.get("tourney_name", "")))

        home_stats = tennis_player_stats(players, home, surface)
        away_stats = tennis_player_stats(players, away, surface)
        home_prob = tennis_model_probability(home_stats, away_stats)
        away_prob = 1 - home_prob

        for market, selection, probability, won in [
            ("Player 1 Win", home, home_prob, actual_home_win),
            ("Player 2 Win", away, away_prob, 1 - actual_home_win),
        ]:
            rows.append({
                "date": match.get("tourney_date"),
                "sport": str(match.get("tourney_name", "tennis_historical")),
                "category": "tennis",
                "home_team": home,
                "away_team": away,
                "market": market,
                "selection": selection,
                "ai_probability": round(probability, 4),
                "bookmaker_odds": pd.NA,
                "value": pd.NA,
                "bet_mode": "BACKTEST",
                "odds_source": "tennis-model-backtest",
                "result": "WIN" if won else "LOSS",
                "won": int(won),
                "surface": surface,
                "home_matches": home_stats["matches"],
                "away_matches": away_stats["matches"],
            })

        update_tennis_players(players, match)

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


def football_threshold_score(data, value_probability, safe_probability, mega_probability, value_value, safe_value, mega_value):
    selected = data[
        (
            (data["ai_probability"] >= mega_probability)
            & (data["value"] >= mega_value)
            & (data["bookmaker_odds"] <= 2.20)
        )
        | (
            (data["ai_probability"] >= safe_probability)
            & (data["value"] >= safe_value)
            & (data["bookmaker_odds"] <= 1.90)
        )
        | (
            (data["ai_probability"] >= value_probability)
            & (data["value"] >= value_value)
            & (data["bookmaker_odds"] <= 3.00)
        )
    ].copy()

    if len(selected) < 8:
        return None

    profit = pd.to_numeric(selected.get("flat_profit", 0), errors="coerce").fillna(0).sum()
    roi = profit / len(selected)
    winrate = pd.to_numeric(selected.get("won", 0), errors="coerce").fillna(0).mean()
    score = roi + min(len(selected), 60) / 600 + max(winrate - 0.50, -0.20) * 0.20
    return score, roi, winrate, len(selected)


def optimize_football_thresholds(backtest):
    if backtest.empty:
        return None

    data = backtest.copy()
    for col in ["ai_probability", "value", "bookmaker_odds", "flat_profit", "won"]:
        data[col] = pd.to_numeric(data.get(col, 0), errors="coerce")
    data = data.dropna(subset=["ai_probability", "value", "bookmaker_odds", "flat_profit", "won"])

    best = None
    for value_probability in [0.56, 0.58, 0.60, 0.62]:
        for safe_probability in [0.62, 0.63, 0.65, 0.67]:
            for mega_probability in [0.68, 0.70, 0.72, 0.74]:
                if not (value_probability <= safe_probability <= mega_probability):
                    continue
                for value_value in [0.02, 0.03, 0.04, 0.05]:
                    for safe_value in [0.005, 0.01, 0.02]:
                        for mega_value in [0.02, 0.03, 0.04]:
                            result = football_threshold_score(
                                data,
                                value_probability,
                                safe_probability,
                                mega_probability,
                                value_value,
                                safe_value,
                                mega_value,
                            )
                            if result is None:
                                continue
                            score, roi, winrate, bets = result
                            if best is None or score > best["score"]:
                                best = {
                                    "category": "football",
                                    "mega_probability": mega_probability,
                                    "mega_value": mega_value,
                                    "safe_probability": safe_probability,
                                    "safe_value": safe_value,
                                    "value_probability": value_probability,
                                    "value_value": value_value,
                                    "backtest_bets": bets,
                                    "backtest_roi": roi,
                                    "backtest_winrate": winrate,
                                    "score": score,
                                }
    return best


def optimize_tennis_thresholds(backtest):
    if backtest.empty:
        return None

    data = backtest.copy()
    data["ai_probability"] = pd.to_numeric(data.get("ai_probability", 0), errors="coerce")
    data["won"] = pd.to_numeric(data.get("won", 0), errors="coerce")
    data = data.dropna(subset=["ai_probability", "won"])
    if data.empty:
        return None

    best = None
    for safe_probability in [0.60, 0.62, 0.63, 0.65, 0.67, 0.70]:
        selected = data[data["ai_probability"] >= safe_probability].copy()
        if len(selected) < 30:
            continue

        winrate = selected["won"].mean()
        calibration_gap = abs(winrate - selected["ai_probability"].mean())
        score = winrate - calibration_gap * 0.55 + min(len(selected), 160) / 1600
        if best is None or score > best["score"]:
            production_safe = min(max(safe_probability, 0.63), 0.67)
            best = {
                "category": "tennis",
                "mega_probability": min(production_safe + 0.07, 0.74),
                "mega_value": 0.03,
                "safe_probability": production_safe,
                "safe_value": 0.01,
                "value_probability": max(production_safe - 0.05, 0.58),
                "value_value": 0.03,
                "backtest_bets": len(selected),
                "backtest_roi": pd.NA,
                "backtest_winrate": winrate,
                "score": score,
            }
    return best


def write_threshold_optimizer(football_backtest, tennis_backtest):
    rows = []
    football = optimize_football_thresholds(football_backtest)
    tennis = optimize_tennis_thresholds(tennis_backtest)
    if football:
        rows.append(football)
    if tennis:
        rows.append(tennis)

    defaults = {
        "mega_probability": 0.70,
        "mega_value": 0.03,
        "safe_probability": 0.63,
        "safe_value": 0.01,
        "value_probability": 0.58,
        "value_value": 0.03,
        "backtest_bets": 0,
        "backtest_roi": pd.NA,
        "backtest_winrate": pd.NA,
        "score": 0,
    }
    for category in ["football", "tennis"]:
        if not any(row["category"] == category for row in rows):
            row = defaults.copy()
            row["category"] = category
            rows.append(row)

    pd.DataFrame(rows).to_csv(THRESHOLD_PATH, index=False)


def main():
    LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    backtest = run_football_backtest()
    tennis_backtest = run_tennis_backtest()
    tracking = tracking_finished_rows()
    combined = pd.concat([backtest, tennis_backtest, tracking], ignore_index=True, sort=False)

    backtest.to_csv(BACKTEST_PATH, index=False)
    tennis_backtest.to_csv(TENNIS_BACKTEST_PATH, index=False)
    calibration = calibration_rows(combined)
    calibration.to_csv(CALIBRATION_PATH, index=False)

    summary = summary_rows(pd.concat([backtest, tennis_backtest], ignore_index=True, sort=False))
    summary.to_csv(SUMMARY_PATH, index=False)
    write_learning_segments(summary)
    write_threshold_optimizer(backtest, tennis_backtest)

    print("Backtest foot lignes :", len(backtest))
    print("Backtest tennis lignes :", len(tennis_backtest))
    print("Calibration segments :", len(calibration))
    if not summary.empty:
        print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
