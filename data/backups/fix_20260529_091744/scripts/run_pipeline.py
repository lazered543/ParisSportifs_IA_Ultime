from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.football_features import build_team_strength, estimate_xg
from src.models.elo import build_elo_ratings, get_match_elo
from src.models.poisson import football_poisson_probs
from src.utils.config import BANKROLL_START

UPCOMING_PATH = Path("data/processed/upcoming_odds.csv")
FOOTBALL_HISTORY_PATH = Path("data/processed/football_history_all.csv")
TENNIS_HISTORY_PATH = Path("data/processed/tennis_history_all.csv")
PREDICTIONS_PATH = Path("data/predictions/predictions_today.csv")
VALUE_BETS_PATH = Path("data/predictions/value_bets_today.csv")
LEARNING_PROFILE_PATH = Path("data/learning/ai_learning_profile.csv")

OUTPUT_COLUMNS = [
    "last_update",
    "bet_mode",
    "date",
    "sport",
    "category",
    "home_team",
    "away_team",
    "market",
    "selection",
    "ai_probability",
    "bookmaker_odds",
    "implied_probability",
    "value",
    "confidence",
    "ia_badge",
    "reliable_only",
    "safety_score",
    "safety_level",
    "football_trap_signal",
    "learning_adjustment",
    "home_recent_form",
    "away_recent_form",
    "home_recent_attack",
    "away_recent_attack",
    "home_recent_defense",
    "away_recent_defense",
    "football_data_quality",
    "decision",
    "bankroll",
    "stake_percent",
    "kelly_fraction",
    "suggested_stake",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_xg",
    "away_xg",
    "draw_probability",
    "draw_hunter",
    "score_exact_1",
    "score_exact_1_proba",
    "score_exact_2",
    "score_exact_2_proba",
    "score_exact_3",
    "score_exact_3_proba",
    "score_exact_alert",
    "scorer_prediction",
    "over_25",
    "under_25",
    "btts_yes",
    "btts_no",
    "top_scores",
    "tennis_engine_score",
    "tennis_form_home",
    "tennis_form_away",
    "tennis_edge",
    "priority",
]

RECOMMENDED_MODES = {"MEGA VALUE", "SAFE PICK", "VALUE BET", "RISKY VALUE"}


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def sigmoid(value):
    value = clamp(value, -9, 9)
    return 1 / (1 + math.exp(-value))


def logit(value):
    value = clamp(value, 0.001, 0.999)
    return math.log(value / (1 - value))


def clean_name(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    replacements = {
        "paris saint germain": "psg",
        "paris sg": "psg",
        "st etienne": "saint etienne",
        "as saint etienne": "saint etienne",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def is_tennis_sport(sport):
    return "tennis" in str(sport).lower()


def is_football_sport(sport):
    s = str(sport).lower()
    return "soccer" in s or "football" in s


def implied_probability(odds):
    odds = safe_float(odds)
    if odds <= 1:
        return 0.0
    return 1 / odds


def normalized_probabilities(items):
    implied = {
        key: implied_probability(odds)
        for key, odds in items.items()
        if safe_float(odds) > 1
    }
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in implied.items()}


def confidence_label(probability):
    if probability >= 0.74:
        return "Elite"
    if probability >= 0.64:
        return "Forte"
    if probability >= 0.56:
        return "Moyenne"
    if probability >= 0.49:
        return "Faible"
    return "A EVITER"


def load_upcoming():
    if not UPCOMING_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(UPCOMING_PATH)
    required = ["sport", "commence_time", "home_team", "away_team"]
    for col in required:
        if col not in df.columns:
            df[col] = ""
    return df


def load_football_context():
    if not FOOTBALL_HISTORY_PATH.exists():
        return pd.DataFrame(), pd.DataFrame(), {}

    history = pd.read_csv(FOOTBALL_HISTORY_PATH, low_memory=False)
    history = history.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    history["FTHG"] = pd.to_numeric(history["FTHG"], errors="coerce")
    history["FTAG"] = pd.to_numeric(history["FTAG"], errors="coerce")
    history = history.dropna(subset=["FTHG", "FTAG"]).copy()

    strengths = build_team_strength(history)
    ratings = build_elo_ratings(history)
    return history, strengths, ratings


def team_row(strengths, team):
    if strengths.empty:
        return None
    exact = strengths[strengths["team"].astype(str).str.lower() == str(team).lower()]
    if not exact.empty:
        return exact.iloc[0]
    cleaned = clean_name(team)
    strengths = strengths.copy()
    strengths["_clean"] = strengths["team"].apply(clean_name)
    fuzzy = strengths[strengths["_clean"] == cleaned]
    if not fuzzy.empty:
        return fuzzy.iloc[0]
    return None


def team_metric(strengths, team, metric, default):
    row = team_row(strengths, team)
    if row is None:
        return default
    return safe_float(row.get(metric, default), default)


def market_xg_from_odds(book_probs):
    home_prob = book_probs.get("home", 0.42)
    away_prob = book_probs.get("away", 0.31)
    draw_prob = book_probs.get("draw", 0.27)
    diff = clamp(home_prob - away_prob, -0.48, 0.48)
    draw_drag = clamp((draw_prob - 0.27) * 1.1, -0.20, 0.25)

    home_xg = 1.28 + diff * 1.35 - draw_drag * 0.45
    away_xg = 1.08 - diff * 1.08 - draw_drag * 0.35
    return round(clamp(home_xg, 0.35, 3.8), 3), round(clamp(away_xg, 0.25, 3.4), 3)


def blend_football_probabilities(book_probs, poisson_probs, elo_home_prob, quality):
    draw_anchor = book_probs.get("draw", poisson_probs["p_draw"])
    elo_draw = clamp((poisson_probs["p_draw"] + draw_anchor) / 2, 0.16, 0.34)
    non_draw = max(1 - elo_draw, 0.01)
    elo_probs = {
        "home": non_draw * elo_home_prob,
        "away": non_draw * (1 - elo_home_prob),
        "draw": elo_draw,
    }

    raw = {}
    for key, poisson_key in [("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")]:
        b = book_probs.get(key, 1 / 3)
        p = poisson_probs.get(poisson_key, 1 / 3)
        e = elo_probs.get(key, 1 / 3)
        raw[key] = 0.48 * b + 0.32 * p + 0.20 * e

    total = sum(raw.values())
    raw = {key: value / total for key, value in raw.items()}

    model_weight = 0.52 + 0.20 * clamp(quality, 0, 1)
    calibrated = {
        key: book_probs.get(key, raw[key]) + (raw[key] - book_probs.get(key, raw[key])) * model_weight
        for key in raw
    }
    total = sum(calibrated.values())
    return {key: clamp(value / total, 0.03, 0.86) for key, value in calibrated.items()}


def safety_score(probability, value, odds, data_quality, mode_hint=""):
    odds = safe_float(odds)
    score = 42
    score += probability * 42
    score += clamp(value, -0.10, 0.22) * 115
    score += clamp(data_quality, 0, 1) * 13

    if 1.35 <= odds <= 2.30:
        score += 8
    elif 2.30 < odds <= 3.50:
        score += 3
    elif odds > 4.00:
        score -= 10

    if "draw" in mode_hint.lower():
        score -= 4

    return round(clamp(score, 0, 100), 2)


def select_bet_mode(probability, value, odds, safety, category):
    if odds <= 1 or value <= 0:
        if probability >= 0.55 and value > -0.035:
            return "WATCHLIST"
        return "NO BET"

    if value >= 0.095 and probability >= 0.62 and safety >= 72 and odds <= 3.20:
        return "MEGA VALUE"

    if probability >= 0.64 and value >= 0.003 and 1.25 <= odds <= 1.95 and safety >= 61:
        return "SAFE PICK"

    min_value = 0.018 if category == "tennis" else 0.024
    if value >= min_value and probability >= 0.52 and safety >= 50 and odds <= 3.85:
        return "VALUE BET"

    if category == "football" and value >= 0.05 and probability >= 0.20 and safety >= 32 and odds <= 5.50:
        return "RISKY VALUE"

    if value >= 0.006 and probability >= 0.47 and odds <= 5.20 and safety >= 40:
        return "RISKY VALUE"

    if value > -0.025 and probability >= 0.50:
        return "WATCHLIST"

    return "NO BET"


def safety_level(mode, safety):
    if mode == "MEGA VALUE":
        return "1 - MEGA VALUE"
    if mode == "SAFE PICK":
        return "2 - SAFE PICK"
    if mode == "VALUE BET":
        return "3 - VALUE BET"
    if mode == "RISKY VALUE":
        return "4 - RISKY VALUE"
    if safety >= 55:
        return "5 - WATCHLIST"
    return "6 - NO BET"


def ia_badge(mode):
    badges = {
        "MEGA VALUE": "MEGA VALUE",
        "SAFE PICK": "SAFE PICK",
        "VALUE BET": "VALUE BET",
        "RISKY VALUE": "RISKY VALUE",
        "WATCHLIST": "WATCHLIST",
    }
    return badges.get(mode, "NO BET")


def bankroll_management(probability, odds, mode, bankroll=BANKROLL_START):
    if mode not in RECOMMENDED_MODES or odds <= 1:
        return 0.0, 0.0, 0.0

    b = odds - 1
    edge = probability * odds - 1
    if edge <= 0:
        return 0.0, 0.0, 0.0

    kelly = max(0.0, ((b * probability) - (1 - probability)) / b)
    kelly = min(kelly, 0.18)

    fractions = {
        "MEGA VALUE": 0.34,
        "SAFE PICK": 0.26,
        "VALUE BET": 0.18,
        "RISKY VALUE": 0.08,
    }
    caps = {
        "MEGA VALUE": 0.030,
        "SAFE PICK": 0.022,
        "VALUE BET": 0.016,
        "RISKY VALUE": 0.006,
    }

    stake_percent = min(kelly * fractions[mode], caps[mode])
    if stake_percent < 0.0025:
        return 0.0, round(stake_percent, 4), round(kelly, 4)

    stake = bankroll * stake_percent
    stake = max(stake, 0.30 if mode == "RISKY VALUE" else 0.50)
    return round(stake, 2), round(stake_percent, 4), round(kelly, 4)


def score_exact_fields(poisson_probs):
    top_scores = poisson_probs.get("top_scores", [])
    fields = {}
    for idx in range(3):
        score, probability = ("", 0.0)
        if idx < len(top_scores):
            score, probability = top_scores[idx]
        fields[f"score_exact_{idx + 1}"] = score
        fields[f"score_exact_{idx + 1}_proba"] = round(probability * 100, 2)
    return fields


def football_trap_signal(market, probability, book_prob, value, odds, draw_probability):
    if value > 0.06 and probability > book_prob + 0.035:
        return "VALUE CLAIRE"
    if str(market).lower() == "draw" and draw_probability < 0.22:
        return "DRAW RISQUE"
    if odds > 4.5:
        return "COTE HAUTE"
    if abs(probability - book_prob) < 0.015:
        return "MARCHE JUSTE"
    return "OK"


def process_football_match(row, strengths, ratings):
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    sport = row.get("sport", "")
    date = row.get("commence_time", "")

    odds = {
        "home": safe_float(row.get("odds_home")),
        "draw": safe_float(row.get("odds_draw")),
        "away": safe_float(row.get("odds_away")),
    }
    book_probs = normalized_probabilities(odds)
    if len(book_probs) < 2:
        return []

    market_home_xg, market_away_xg = market_xg_from_odds(book_probs)
    model_home_xg, model_away_xg = estimate_xg(home, away, strengths)

    home_matches = team_metric(strengths, home, "matches", 0)
    away_matches = team_metric(strengths, away, "matches", 0)
    quality = clamp((home_matches + away_matches) / 40, 0, 1)

    home_xg = model_home_xg * quality + market_home_xg * (1 - quality)
    away_xg = model_away_xg * quality + market_away_xg * (1 - quality)
    poisson_probs = football_poisson_probs(home_xg, away_xg, max_goals=8)

    elo = get_match_elo(home, away, ratings)
    if home_matches == 0 or away_matches == 0:
        elo_home_prob = book_probs.get("home", 0.45) / max(book_probs.get("home", 0.45) + book_probs.get("away", 0.35), 0.01)
    else:
        elo_home_prob = 1 / (1 + 10 ** (-safe_float(elo["elo_diff"]) / 400))

    ai_probs = blend_football_probabilities(book_probs, poisson_probs, elo_home_prob, quality)
    score_fields = score_exact_fields(poisson_probs)
    top_scores = [(score, round(prob, 4)) for score, prob in poisson_probs.get("top_scores", [])]

    rows = []
    for key, market, selection, bookmaker_odds in [
        ("home", "Home Win", home, odds.get("home")),
        ("draw", "Draw", "Draw", odds.get("draw")),
        ("away", "Away Win", away, odds.get("away")),
    ]:
        bookmaker_odds = safe_float(bookmaker_odds)
        if bookmaker_odds <= 1:
            continue

        probability = ai_probs.get(key, 0.0)
        implied = implied_probability(bookmaker_odds)
        value = probability * bookmaker_odds - 1
        safety = safety_score(probability, value, bookmaker_odds, quality, market)
        mode = select_bet_mode(probability, value, bookmaker_odds, safety, "football")
        stake, stake_percent, kelly = bankroll_management(probability, bookmaker_odds, mode)
        if stake <= 0 and mode in RECOMMENDED_MODES:
            mode = "WATCHLIST"

        confidence = confidence_label(probability)
        priority = round(safety * 1.5 + max(value, 0) * 420 + probability * 70, 2)

        rows.append({
            "date": date,
            "sport": sport,
            "category": "football",
            "home_team": home,
            "away_team": away,
            "market": market,
            "selection": selection,
            "ai_probability": round(probability, 4),
            "bookmaker_odds": round(bookmaker_odds, 3),
            "implied_probability": round(implied, 4),
            "value": round(value, 4),
            "confidence": confidence,
            "ia_badge": ia_badge(mode),
            "reliable_only": mode in {"MEGA VALUE", "SAFE PICK", "VALUE BET"},
            "safety_score": safety,
            "safety_level": safety_level(mode, safety),
            "football_trap_signal": football_trap_signal(market, probability, book_probs.get(key, implied), value, bookmaker_odds, poisson_probs["p_draw"]),
            "learning_adjustment": "BASELINE",
            "home_recent_form": round(team_metric(strengths, home, "form", 0.50), 3),
            "away_recent_form": round(team_metric(strengths, away, "form", 0.50), 3),
            "home_recent_attack": round(team_metric(strengths, home, "attack", 1.20), 3),
            "away_recent_attack": round(team_metric(strengths, away, "attack", 1.10), 3),
            "home_recent_defense": round(team_metric(strengths, home, "defense", 1.20), 3),
            "away_recent_defense": round(team_metric(strengths, away, "defense", 1.20), 3),
            "football_data_quality": round(quality, 3),
            "decision": "VALUE BET" if mode in RECOMMENDED_MODES and stake > 0 else "NO BET",
            "bankroll": round(BANKROLL_START, 2),
            "stake_percent": stake_percent,
            "kelly_fraction": kelly,
            "suggested_stake": stake,
            "bet_mode": mode,
            "home_elo": elo["home_elo"],
            "away_elo": elo["away_elo"],
            "elo_diff": elo["elo_diff"],
            "home_xg": round(home_xg, 3),
            "away_xg": round(away_xg, 3),
            "draw_probability": round(poisson_probs["p_draw"], 4),
            "draw_hunter": "DRAW POSSIBLE" if poisson_probs["p_draw"] >= 0.285 else "NO DRAW SIGNAL",
            "score_exact_alert": "TOP SCORE OK" if score_fields["score_exact_1"] else "",
            "scorer_prediction": "A recalculer via player_scorers",
            "over_25": round(poisson_probs["over_25"], 4),
            "under_25": round(poisson_probs["under_25"], 4),
            "btts_yes": round(poisson_probs["btts_yes"], 4),
            "btts_no": round(poisson_probs["btts_no"], 4),
            "top_scores": str(top_scores),
            "tennis_engine_score": 0.0,
            "tennis_form_home": 0.0,
            "tennis_form_away": 0.0,
            "tennis_edge": 0.0,
            "priority": priority,
            **score_fields,
        })

    return rows


def detect_surface(sport):
    s = str(sport).lower()
    if "french" in s or "clay" in s or "roland" in s:
        return "Clay"
    if "wimbledon" in s or "grass" in s:
        return "Grass"
    if "hard" in s or "us_open" in s or "australian" in s:
        return "Hard"
    return ""


def build_tennis_model():
    if not TENNIS_HISTORY_PATH.exists():
        return {}

    history = pd.read_csv(TENNIS_HISTORY_PATH, low_memory=False)
    if history.empty:
        return {}

    if "tourney_date" in history.columns:
        history["_date"] = pd.to_numeric(history["tourney_date"], errors="coerce")
        history = history.sort_values("_date")

    players = defaultdict(lambda: {
        "matches": 0,
        "wins": 0,
        "losses": 0,
        "elo": 1500.0,
        "recent": deque(maxlen=12),
        "surface": defaultdict(lambda: {"matches": 0, "wins": 0}),
        "rank": None,
        "rank_points": None,
    })

    def update_rank(player, rank, points):
        if pd.notna(rank):
            player["rank"] = safe_float(rank, player["rank"] or 999)
        if pd.notna(points):
            player["rank_points"] = safe_float(points, player["rank_points"] or 0)

    for _, match in history.iterrows():
        winner_name = match.get("winner_name", "")
        loser_name = match.get("loser_name", "")
        if not winner_name or not loser_name:
            continue

        w_key = clean_name(winner_name)
        l_key = clean_name(loser_name)
        if not w_key or not l_key:
            continue

        winner = players[w_key]
        loser = players[l_key]

        expected_w = 1 / (1 + 10 ** ((loser["elo"] - winner["elo"]) / 400))
        expected_l = 1 - expected_w
        k = 26
        winner["elo"] += k * (1 - expected_w)
        loser["elo"] += k * (0 - expected_l)

        surface = str(match.get("surface", "") or "")
        winner["matches"] += 1
        winner["wins"] += 1
        winner["recent"].append(1)
        loser["matches"] += 1
        loser["losses"] += 1
        loser["recent"].append(0)

        if surface:
            winner["surface"][surface]["matches"] += 1
            winner["surface"][surface]["wins"] += 1
            loser["surface"][surface]["matches"] += 1
            loser["surface"][surface]["wins"] += 0

        update_rank(winner, match.get("winner_rank"), match.get("winner_rank_points"))
        update_rank(loser, match.get("loser_rank"), match.get("loser_rank_points"))

    return players


def tennis_stats(players, name):
    key = clean_name(name)
    player = players.get(key)
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
    return {
        "matches": player["matches"],
        "elo": player["elo"],
        "winrate": player["wins"] / matches,
        "form": sum(recent) / len(recent) if recent else player["wins"] / matches,
        "rank": player["rank"] if player["rank"] is not None else 999,
        "rank_points": player["rank_points"] if player["rank_points"] is not None else 0,
        "surface_form": 0.5,
    }


def tennis_surface_form(players, name, surface):
    if not surface:
        return 0.5
    player = players.get(clean_name(name))
    if player is None:
        return 0.5
    stats = player["surface"].get(surface)
    if not stats or stats["matches"] <= 0:
        return 0.5
    return stats["wins"] / stats["matches"]


def tennis_probabilities(row, players):
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    odds = {
        "home": safe_float(row.get("odds_home")),
        "away": safe_float(row.get("odds_away")),
    }
    book_probs = normalized_probabilities(odds)
    if len(book_probs) < 2:
        return None

    surface = detect_surface(row.get("sport", ""))
    home_stats = tennis_stats(players, home)
    away_stats = tennis_stats(players, away)
    home_stats["surface_form"] = tennis_surface_form(players, home, surface)
    away_stats["surface_form"] = tennis_surface_form(players, away, surface)

    quality = clamp((min(home_stats["matches"], 24) + min(away_stats["matches"], 24)) / 48, 0, 1)

    rank_component = 0.0
    if home_stats["rank"] and away_stats["rank"]:
        rank_component = clamp(math.log((away_stats["rank"] + 12) / (home_stats["rank"] + 12)), -1.6, 1.6)

    model_logit = (
        (home_stats["elo"] - away_stats["elo"]) / 185
        + (home_stats["form"] - away_stats["form"]) * 1.15
        + (home_stats["surface_form"] - away_stats["surface_form"]) * 0.55
        + rank_component * 0.38
    )
    model_home = sigmoid(model_logit)

    book_home = book_probs.get("home", 0.5)
    model_weight = 0.38 + 0.22 * quality
    final_home = book_home + (model_home - book_home) * model_weight
    final_home = clamp(final_home, 0.08, 0.88)

    return {
        "home": final_home,
        "away": 1 - final_home,
        "book": book_probs,
        "quality": quality,
        "home_stats": home_stats,
        "away_stats": away_stats,
    }


def tennis_set_scores(probability):
    straight = clamp(0.46 + max(probability - 0.50, 0) * 0.82, 0.43, 0.80)
    three_sets = 1 - straight
    if straight >= three_sets:
        return "2-0", round(straight * 100, 2), "2-1", round(three_sets * 100, 2)
    return "2-1", round(three_sets * 100, 2), "2-0", round(straight * 100, 2)


def process_tennis_match(row, players):
    probs = tennis_probabilities(row, players)
    if probs is None:
        return []

    home = row.get("home_team", "")
    away = row.get("away_team", "")
    sport = row.get("sport", "")
    date = row.get("commence_time", "")

    rows = []
    for key, market, selection, bookmaker_odds in [
        ("home", "Player 1 Win", home, row.get("odds_home")),
        ("away", "Player 2 Win", away, row.get("odds_away")),
    ]:
        bookmaker_odds = safe_float(bookmaker_odds)
        if bookmaker_odds <= 1:
            continue

        probability = probs[key]
        implied = implied_probability(bookmaker_odds)
        value = probability * bookmaker_odds - 1
        safety = safety_score(probability, value, bookmaker_odds, probs["quality"])
        mode = select_bet_mode(probability, value, bookmaker_odds, safety, "tennis")
        stake, stake_percent, kelly = bankroll_management(probability, bookmaker_odds, mode)
        if stake <= 0 and mode in RECOMMENDED_MODES:
            mode = "WATCHLIST"

        score1, score1_proba, score2, score2_proba = tennis_set_scores(probability)
        engine_score = round(clamp(probability * 70 + probs["quality"] * 20 + max(value, 0) * 85, 0, 100), 2)
        priority = round(safety * 1.45 + max(value, 0) * 430 + probability * 70, 2)

        rows.append({
            "date": date,
            "sport": sport,
            "category": "tennis",
            "home_team": home,
            "away_team": away,
            "market": market,
            "selection": selection,
            "ai_probability": round(probability, 4),
            "bookmaker_odds": round(bookmaker_odds, 3),
            "implied_probability": round(implied, 4),
            "value": round(value, 4),
            "confidence": confidence_label(probability),
            "ia_badge": ia_badge(mode),
            "reliable_only": mode in {"MEGA VALUE", "SAFE PICK", "VALUE BET"},
            "safety_score": safety,
            "safety_level": safety_level(mode, safety),
            "football_trap_signal": "",
            "learning_adjustment": "BASELINE",
            "home_recent_form": "",
            "away_recent_form": "",
            "home_recent_attack": "",
            "away_recent_attack": "",
            "home_recent_defense": "",
            "away_recent_defense": "",
            "football_data_quality": "",
            "decision": "VALUE BET" if mode in RECOMMENDED_MODES and stake > 0 else "NO BET",
            "bankroll": round(BANKROLL_START, 2),
            "stake_percent": stake_percent,
            "kelly_fraction": kelly,
            "suggested_stake": stake,
            "bet_mode": mode,
            "home_elo": round(probs["home_stats"]["elo"], 2),
            "away_elo": round(probs["away_stats"]["elo"], 2),
            "elo_diff": round(probs["home_stats"]["elo"] - probs["away_stats"]["elo"], 2),
            "home_xg": "",
            "away_xg": "",
            "draw_probability": "",
            "draw_hunter": "NO DRAW SPORT",
            "score_exact_1": score1,
            "score_exact_1_proba": score1_proba,
            "score_exact_2": score2,
            "score_exact_2_proba": score2_proba,
            "score_exact_3": "",
            "score_exact_3_proba": "",
            "score_exact_alert": "SET SCORE ESTIMATION",
            "scorer_prediction": "Tennis : aucun buteur",
            "over_25": "",
            "under_25": "",
            "btts_yes": "",
            "btts_no": "",
            "top_scores": f"{score1} {score1_proba}% | {score2} {score2_proba}%",
            "tennis_engine_score": engine_score,
            "tennis_form_home": round(probs["home_stats"]["form"], 3),
            "tennis_form_away": round(probs["away_stats"]["form"], 3),
            "tennis_edge": round(value, 4),
            "priority": priority,
        })

    return rows


def apply_learning_adjustment(predictions):
    if not LEARNING_PROFILE_PATH.exists() or predictions.empty:
        return predictions

    try:
        profile = pd.read_csv(LEARNING_PROFILE_PATH)
    except Exception:
        return predictions

    if profile.empty or "ai_recommendation" not in profile.columns:
        return predictions

    out = predictions.copy()
    for _, segment in profile.iterrows():
        recommendation = str(segment.get("ai_recommendation", "")).upper()
        dimension = str(segment.get("dimension", ""))
        value = str(segment.get("segment", ""))
        if recommendation not in {"BOOST", "REDUCE"}:
            continue

        if dimension == "Sport":
            mask = out["category"].astype(str) == value
        elif dimension == "Competition":
            mask = out["sport"].astype(str) == value
        elif dimension == "Marche":
            mask = out["market"].astype(str) == value
        else:
            continue

        multiplier = 1.025 if recommendation == "BOOST" else 0.975
        out.loc[mask, "ai_probability"] = (
            pd.to_numeric(out.loc[mask, "ai_probability"], errors="coerce").fillna(0) * multiplier
        ).clip(0.02, 0.90)
        out.loc[mask, "learning_adjustment"] = recommendation

    out["value"] = (
        pd.to_numeric(out["ai_probability"], errors="coerce").fillna(0)
        * pd.to_numeric(out["bookmaker_odds"], errors="coerce").fillna(0)
        - 1
    ).round(4)
    return out


def finalise_predictions(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = apply_learning_adjustment(df)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["last_update"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0)
    df["suggested_stake"] = pd.to_numeric(df["suggested_stake"], errors="coerce").fillna(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["ai_probability"] = pd.to_numeric(df["ai_probability"], errors="coerce").fillna(0)

    df = df.sort_values(
        ["priority", "suggested_stake", "value", "ai_probability"],
        ascending=[False, False, False, False],
    )

    return df[OUTPUT_COLUMNS]


def main():
    upcoming = load_upcoming()
    if upcoming.empty:
        print("Aucun match/cote dans data/processed/upcoming_odds.csv")
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(PREDICTIONS_PATH, index=False)
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(VALUE_BETS_PATH, index=False)
        return

    football_history, strengths, ratings = load_football_context()
    tennis_players = build_tennis_model()

    rows = []
    for _, match in upcoming.iterrows():
        sport = match.get("sport", "")
        if is_football_sport(sport):
            rows.extend(process_football_match(match, strengths, ratings))
        elif is_tennis_sport(sport):
            rows.extend(process_tennis_match(match, tennis_players))

    predictions = finalise_predictions(rows)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    value_bets = predictions[
        predictions["bet_mode"].isin(RECOMMENDED_MODES)
        & (pd.to_numeric(predictions["suggested_stake"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(predictions["value"], errors="coerce").fillna(0) > 0)
    ].copy()
    value_bets.to_csv(VALUE_BETS_PATH, index=False)

    print("Predictions creees :", len(predictions))
    print("Paris avec mise :", len(value_bets))
    if not value_bets.empty:
        print(value_bets[["date", "sport", "home_team", "away_team", "market", "selection", "ai_probability", "bookmaker_odds", "value", "suggested_stake", "bet_mode"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
