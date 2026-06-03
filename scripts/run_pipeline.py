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
from src.utils.config import BANKROLL_START as CONFIG_BANKROLL_START

BANKROLL_START = 10.0

UPCOMING_PATH = Path("data/processed/upcoming_odds.csv")
FOOTBALL_HISTORY_PATH = Path("data/processed/football_history_all.csv")
TENNIS_HISTORY_PATH = Path("data/processed/tennis_history_all.csv")
PREDICTIONS_PATH = Path("data/predictions/predictions_today.csv")
VALUE_BETS_PATH = Path("data/predictions/value_bets_today.csv")
LEARNING_PROFILE_PATH = Path("data/learning/ai_learning_profile.csv")
LEARNING_SUMMARY_PATH = Path("data/learning/ai_learning_summary.csv")
CALIBRATION_PATH = Path("data/learning/probability_calibration.csv")
THRESHOLD_PROFILE_PATH = Path("data/learning/threshold_optimizer.csv")

OUTPUT_COLUMNS = [
    "last_update",
    "bet_mode",
    "date",
    "sport",
    "odds_source",
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
    "calibration_adjustment",
    "threshold_profile",
    "decision_status",
    "refusal_reason",
    "decision_reason",
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

RECOMMENDED_MODES = {"MEGA VALUE", "SAFE PICK", "VALUE BET"}
DEFAULT_THRESHOLDS = {
    "mega_probability": 0.70,
    "mega_value": 0.03,
    "safe_probability": 0.63,
    "safe_value": 0.01,
    "value_probability": 0.58,
    "value_value": 0.03,
}
_THRESHOLD_CACHE = None


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
    if probability >= 0.78:
        return "Elite"
    if probability >= 0.68:
        return "Forte"
    if probability >= 0.58:
        return "Moyenne"
    if probability >= 0.50:
        return "Faible"
    return "A EVITER"


def load_threshold_profiles():
    global _THRESHOLD_CACHE
    if _THRESHOLD_CACHE is not None:
        return _THRESHOLD_CACHE

    profiles = {"default": DEFAULT_THRESHOLDS.copy()}
    if THRESHOLD_PROFILE_PATH.exists():
        try:
            df = pd.read_csv(THRESHOLD_PROFILE_PATH)
            for _, row in df.iterrows():
                category = str(row.get("category", "default")).strip().lower() or "default"
                profile = DEFAULT_THRESHOLDS.copy()
                for key in DEFAULT_THRESHOLDS:
                    if key in row and pd.notna(row.get(key)):
                        profile[key] = safe_float(row.get(key), DEFAULT_THRESHOLDS[key])
                profiles[category] = profile
        except Exception:
            pass

    _THRESHOLD_CACHE = profiles
    return profiles


def threshold_profile(category):
    profiles = load_threshold_profiles()
    category = str(category or "").strip().lower()
    return profiles.get(category, profiles.get("default", DEFAULT_THRESHOLDS))


def load_upcoming():
    if not UPCOMING_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(UPCOMING_PATH)
    required = ["sport", "commence_time", "home_team", "away_team"]
    for col in required:
        if col not in df.columns:
            df[col] = ""
    parsed_dates = pd.to_datetime(df["commence_time"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    fresh_mask = parsed_dates.isna() | (parsed_dates >= now - pd.Timedelta(hours=4))
    stale_count = int((~fresh_mask).sum())
    if stale_count:
        print(f"Matchs deja passes ignores : {stale_count}")
    df = df[fresh_mask].copy()
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
    """
    Mélange stable bookmaker + Poisson + Elo.
    Sécurité : ne jamais s'éloigner énormément du marché.
    Exemple : Arsenal 75% contre PSG alors que le marché donne 48% => interdit.
    """
    draw_anchor = book_probs.get("draw", poisson_probs["p_draw"])
    elo_draw = clamp((poisson_probs["p_draw"] + draw_anchor) / 2, 0.16, 0.34)
    non_draw = max(1 - elo_draw, 0.01)
    elo_probs = {"home": non_draw * elo_home_prob, "away": non_draw * (1 - elo_home_prob), "draw": elo_draw}
    raw = {}
    for key, poisson_key in [("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")]:
        b = book_probs.get(key, 1 / 3); p = poisson_probs.get(poisson_key, 1 / 3); e = elo_probs.get(key, 1 / 3)
        raw[key] = 0.62 * b + 0.23 * p + 0.15 * e
    total = sum(raw.values())
    if total <= 0: return {"home": 0.36, "draw": 0.28, "away": 0.36}
    raw = {key: value / total for key, value in raw.items()}
    quality = clamp(quality, 0, 1)
    max_gap = 0.08 + 0.06 * quality
    calibrated = {}
    for key in ["home", "draw", "away"]:
        book = book_probs.get(key, raw[key])
        calibrated[key] = clamp(raw[key], book - max_gap, book + max_gap)
    total = sum(calibrated.values())
    calibrated = {key: value / total for key, value in calibrated.items()}
    return {key: clamp(value, 0.03, 0.86) for key, value in calibrated.items()}


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
    """
    Mode équilibré demandé :
    - VALUE BET dès 58% avec value >= 3%
    - SAFE PICK dès 63% avec value >= 1%
    - MEGA VALUE dès 70% avec value >= 3%
    - aucune cote jouée sous 1.10
    - pas de RISKY VALUE en vraie mise
    - l'IA garde quand même une watchlist pour l'analyse
    """
    odds = safe_float(odds)
    probability = safe_float(probability)
    value = safe_float(value)
    safety = safe_float(safety)
    thresholds = threshold_profile(category)

    if odds < 1.10:
        return "WATCHLIST"

    if odds <= 1 or value <= 0:
        if probability >= thresholds["value_probability"] and odds >= 1.10:
            return "WATCHLIST"
        return "NO BET"

    if probability < thresholds["value_probability"]:
        return "WATCHLIST" if value > 0 else "NO BET"

    if probability >= thresholds["mega_probability"] and value >= thresholds["mega_value"] and 1.10 <= odds <= 2.20:
        return "MEGA VALUE"

    if probability >= thresholds["safe_probability"] and value >= thresholds["safe_value"] and 1.10 <= odds <= 1.90:
        return "SAFE PICK"

    if probability >= thresholds["value_probability"] and value >= thresholds["value_value"] and 1.10 <= odds <= 3.00:
        return "VALUE BET"

    return "WATCHLIST"


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
        "NO BET": "NO BET",
    }
    return badges.get(str(mode), "NO BET")



def load_current_bankroll(default=10.0):
    path = Path("data/bankroll_state.csv")

    if not path.exists():
        return float(default)

    try:
        state = pd.read_csv(path)

        if state.empty or "current_bankroll" not in state.columns:
            return float(default)

        bankroll = pd.to_numeric(state["current_bankroll"], errors="coerce").fillna(default).iloc[0]
        bankroll = float(bankroll)

        return max(bankroll, 0.0)

    except Exception:
        return float(default)


def bankroll_management(probability, odds, mode, bankroll=None):
    """
    Bankroll évolutive :
    - départ 10€
    - si l'IA gagne, la balance augmente
    - les mises sûres augmentent avec la balance
    - si l'IA perd, les mises baissent automatiquement

    Plafonds par type de pari :
    - MEGA VALUE : jusqu'à 25% de la balance, plafonné à 15€
    - SAFE PICK : jusqu'à 15% de la balance, plafonné à 10€
    - VALUE BET : jusqu'à 8% de la balance, plafonné à 5€
    """
    if bankroll is None:
        bankroll = load_current_bankroll(BANKROLL_START)

    bankroll = max(float(bankroll), 0.0)
    odds = safe_float(odds)
    probability = safe_float(probability)

    if bankroll <= 0 or odds <= 1:
        return 0.0, 0.0, 0.0

    if mode not in {"MEGA VALUE", "SAFE PICK", "VALUE BET"}:
        return 0.0, 0.0, 0.0

    edge = probability * odds - 1
    if edge <= 0:
        return 0.0, 0.0, 0.0

    b = odds - 1
    kelly = max(0.0, ((b * probability) - (1 - probability)) / b)
    kelly = min(kelly, 0.22)

    # Plus la balance est haute, plus l'IA peut monter progressivement.
    growth_factor = clamp(bankroll / BANKROLL_START, 1.0, 6.0)

    fractions = {
        "MEGA VALUE": 0.55,
        "SAFE PICK": 0.42,
        "VALUE BET": 0.28,
    }

    caps_pct = {
        "MEGA VALUE": 0.25,
        "SAFE PICK": 0.15,
        "VALUE BET": 0.08,
    }

    caps_abs = {
        "MEGA VALUE": 15.00,
        "SAFE PICK": 10.00,
        "VALUE BET": 5.00,
    }

    # Planchers qui évoluent aussi, mais doucement.
    floors = {
        "MEGA VALUE": min(1.50 * growth_factor, 8.00),
        "SAFE PICK": min(1.00 * growth_factor, 5.00),
        "VALUE BET": min(0.50 * growth_factor, 3.00),
    }

    stake_percent = min(kelly * fractions.get(mode, 0), caps_pct.get(mode, 0))
    stake = bankroll * stake_percent

    # Si le pari est classé MEGA/SAFE/VALUE, il a une mise visible.
    stake = max(stake, floors.get(mode, 0.0))
    stake = min(stake, bankroll * caps_pct.get(mode, 0.01), caps_abs.get(mode, bankroll), bankroll)

    if stake < 0.10:
        return 0.0, round(stake_percent, 4), round(kelly, 4)

    return round(stake, 2), round(stake / bankroll, 4), round(kelly, 4)


def score_exact_fields(poisson_probs, confidence_factor=1.0):
    top_scores = poisson_probs.get("top_scores", [])
    fields = {}
    for idx in range(3):
        score, probability = ("", 0.0)
        if idx < len(top_scores):
            score, probability = top_scores[idx]
        fields[f"score_exact_{idx + 1}"] = score
        fields[f"score_exact_{idx + 1}_proba"] = round(probability * confidence_factor * 100, 2)
    return fields


def score_exact_confidence_factor(odds_source, quality, poisson_probs):
    top_scores = poisson_probs.get("top_scores", [])
    quality = clamp(safe_float(quality), 0, 1)
    if not top_scores:
        return 0.0

    first = safe_float(top_scores[0][1])
    second = safe_float(top_scores[1][1]) if len(top_scores) > 1 else 0
    separation = clamp((first - second) * 8, 0, 0.10)

    if not is_live_odds_source(odds_source):
        return clamp(0.52 + quality * 0.08 + separation, 0.50, 0.64)

    return clamp(0.68 + quality * 0.12 + separation, 0.66, 0.84)


def scale_top_scores(top_scores, confidence_factor):
    return [
        (score, round(safe_float(probability) * confidence_factor, 4))
        for score, probability in top_scores
    ]


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


def is_live_odds_source(source):
    source = str(source or "").strip().lower()
    return source not in {"offline-fallback", "fallback", "manual-fallback", ""}


def source_label(source):
    source = str(source or "").strip()
    return source if source else "unknown"


def probability_bucket(probability):
    probability = safe_float(probability)
    lower = int(clamp(math.floor(probability * 100 / 5) * 5, 0, 95))
    upper = lower + 5
    return f"{lower:02d}-{upper:02d}"


def build_decision_reason(row):
    source = source_label(row.get("odds_source", ""))
    probability = safe_float(row.get("ai_probability"))
    value = safe_float(row.get("value"))
    odds = safe_float(row.get("bookmaker_odds"))
    mode = str(row.get("bet_mode", ""))
    learning = str(row.get("learning_adjustment", "BASELINE"))
    calibration = str(row.get("calibration_adjustment", "BASELINE"))
    trap = str(row.get("football_trap_signal", ""))

    reasons = []
    if not is_live_odds_source(source):
        reasons.append("Analyse seulement: cote fallback non live")
    if odds < 1.10:
        reasons.append("Cote trop basse")
    if value <= 0:
        reasons.append("Value negative ou nulle")
    else:
        reasons.append(f"Value positive {value * 100:.1f}%")
    reasons.append(f"Proba IA {probability * 100:.1f}%")
    if mode in RECOMMENDED_MODES:
        reasons.append(f"Mode {mode}")
    if trap and trap not in {"", "OK"}:
        reasons.append(trap)
    if learning not in {"", "BASELINE", "nan"}:
        reasons.append(f"Learning {learning}")
    if calibration not in {"", "BASELINE", "nan"}:
        reasons.append(f"Calibration {calibration}")
    reasons.append(f"Source {source}")
    return " | ".join(reasons)


def refusal_reason(row):
    mode = str(row.get("bet_mode", ""))
    stake = safe_float(row.get("suggested_stake", 0))
    probability = safe_float(row.get("ai_probability", 0))
    value = safe_float(row.get("value", 0))
    odds = safe_float(row.get("bookmaker_odds", 0))
    source = row.get("odds_source", "")
    category = str(row.get("category", "")).lower()
    thresholds = threshold_profile(category)

    if mode in RECOMMENDED_MODES and stake > 0:
        return "ok"
    if not is_live_odds_source(source):
        return "source fallback"
    if odds <= 1:
        return "cote absente"
    if odds < 1.10:
        return "cote trop basse"
    if odds > 3.00:
        return "cote trop haute"
    if value <= 0:
        return "value negative"
    if probability < thresholds["value_probability"]:
        return "proba trop basse"
    if mode == "WATCHLIST":
        return "watchlist prudente"
    return "mise zero"


def decision_status(row):
    mode = str(row.get("bet_mode", ""))
    stake = safe_float(row.get("suggested_stake", 0))
    if mode in RECOMMENDED_MODES and stake > 0:
        return "JOUABLE"
    if mode == "WATCHLIST":
        return "WATCHLIST"
    return "REFUSE"


def threshold_profile_label(category):
    thresholds = threshold_profile(category)
    return (
        f"MEGA {thresholds['mega_probability']:.0%}/{thresholds['mega_value']:.0%} | "
        f"SAFE {thresholds['safe_probability']:.0%}/{thresholds['safe_value']:.0%} | "
        f"VALUE {thresholds['value_probability']:.0%}/{thresholds['value_value']:.0%}"
    )


def process_football_match(row, strengths, ratings):
    home = row.get("home_team", "")
    away = row.get("away_team", "")
    sport = row.get("sport", "")
    date = row.get("commence_time", "")
    odds_source = source_label(row.get("source", "unknown"))

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

    home_home_form = team_metric(strengths, home, "home_form", team_metric(strengths, home, "form", 0.50))
    away_away_form = team_metric(strengths, away, "away_form", team_metric(strengths, away, "form", 0.50))
    home_context_edge = clamp((home_home_form - away_away_form) * 0.045, -0.025, 0.025)

    poisson_probs = football_poisson_probs(home_xg, away_xg, max_goals=8)

    elo = get_match_elo(home, away, ratings)
    if home_matches == 0 or away_matches == 0:
        elo_home_prob = book_probs.get("home", 0.45) / max(book_probs.get("home", 0.45) + book_probs.get("away", 0.35), 0.01)
    else:
        elo_home_prob = 1 / (1 + 10 ** (-safe_float(elo["elo_diff"]) / 400))

    ai_probs = blend_football_probabilities(book_probs, poisson_probs, elo_home_prob, quality)
    ai_probs["home"] = clamp(ai_probs.get("home", 0) + home_context_edge, 0.02, 0.90)
    ai_probs["away"] = clamp(ai_probs.get("away", 0) - home_context_edge, 0.02, 0.90)
    total_three_way = ai_probs.get("home", 0) + ai_probs.get("draw", 0) + ai_probs.get("away", 0)
    if total_three_way > 0:
        ai_probs = {k: v / total_three_way for k, v in ai_probs.items()}

    score_confidence = score_exact_confidence_factor(odds_source, quality, poisson_probs)
    score_fields = score_exact_fields(poisson_probs, score_confidence)
    top_scores = scale_top_scores(poisson_probs.get("top_scores", []), score_confidence)

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
        if not is_live_odds_source(odds_source):
            mode = "WATCHLIST"
        stake, stake_percent, kelly = bankroll_management(probability, bookmaker_odds, mode)
        if stake <= 0 and mode in RECOMMENDED_MODES:
            mode = "WATCHLIST"

        confidence = confidence_label(probability)
        priority = round(safety * 1.5 + max(value, 0) * 420 + probability * 70, 2)

        rows.append({
            "date": date,
            "sport": sport,
            "odds_source": odds_source,
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
            "calibration_adjustment": "BASELINE",
            "threshold_profile": threshold_profile_label("football"),
            "decision_status": "",
            "refusal_reason": "",
            "decision_reason": "",
            "home_recent_form": round(team_metric(strengths, home, "form", 0.50), 3),
            "away_recent_form": round(team_metric(strengths, away, "form", 0.50), 3),
            "home_recent_attack": round(team_metric(strengths, home, "attack", 1.20), 3),
            "away_recent_attack": round(team_metric(strengths, away, "attack", 1.10), 3),
            "home_recent_defense": round(team_metric(strengths, home, "defense", 1.20), 3),
            "away_recent_defense": round(team_metric(strengths, away, "defense", 1.20), 3),
            "football_data_quality": round(quality, 3),
            "decision": "VALUE BET" if mode in RECOMMENDED_MODES and stake > 0 else "NO BET",
            "bankroll": round(load_current_bankroll(BANKROLL_START), 2),
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
            "score_exact_alert": (
                "SCORE INDICATIF - SOURCE FALLBACK"
                if not is_live_odds_source(odds_source)
                else "SCORE CALIBRE - PROBA PRUDENTE"
                if score_fields["score_exact_1"]
                else ""
            ),
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

        rows[-1]["decision_reason"] = build_decision_reason(rows[-1])

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



def calibrate_tennis_vs_market(model_home, book_home, quality, home_stats, away_stats):
    """
    Calibration tennis LEVEL MAX.
    Le bookmaker reste l'ancre principale. L'IA ne peut s'écarter fortement
    que si elle possède beaucoup de données propres.
    """
    quality = clamp(quality, 0, 1)

    home_matches = safe_float(home_stats.get("matches", 0))
    away_matches = safe_float(away_stats.get("matches", 0))
    data_depth = clamp((home_matches + away_matches) / 100, 0, 1)

    model_weight = 0.12 + 0.22 * data_depth + 0.08 * quality
    model_weight = clamp(model_weight, 0.12, 0.38)

    blended = book_home + (model_home - book_home) * model_weight

    max_gap = 0.035 + 0.045 * data_depth

    if book_home >= 0.70:
        max_gap = min(max_gap, 0.030)

    if book_home <= 0.40:
        max_gap = min(max_gap, 0.040)

    calibrated = clamp(blended, book_home - max_gap, book_home + max_gap)

    # Les 80% doivent rester rares et justifiés par le marché.
    if calibrated > 0.78 and book_home < 0.74:
        calibrated = 0.78

    return clamp(calibrated, 0.14, 0.82)


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
    final_home = calibrate_tennis_vs_market(
        model_home,
        book_home,
        quality,
        home_stats,
        away_stats,
    )

    return {
        "home": final_home,
        "away": 1 - final_home,
        "book": book_probs,
        "quality": quality,
        "home_stats": home_stats,
        "away_stats": away_stats,
    }


def tennis_set_scores(probability, selected_stats=None, opponent_stats=None, quality=0.5):
    """
    Estimation des sets basée sur :
    - probabilité IA
    - forme récente
    - forme sur surface
    - écart ELO
    - écart de classement
    - qualité des données

    Ce n'est pas un score exact garanti, mais une estimation automatique plus logique
    que l'ancien 2-0 répété partout.
    """
    probability = clamp(probability, 0.01, 0.99)
    selected_stats = selected_stats or {}
    opponent_stats = opponent_stats or {}

    selected_form = safe_float(selected_stats.get("form", 0.5), 0.5)
    opponent_form = safe_float(opponent_stats.get("form", 0.5), 0.5)
    selected_surface = safe_float(selected_stats.get("surface_form", 0.5), 0.5)
    opponent_surface = safe_float(opponent_stats.get("surface_form", 0.5), 0.5)
    selected_elo = safe_float(selected_stats.get("elo", 1500), 1500)
    opponent_elo = safe_float(opponent_stats.get("elo", 1500), 1500)
    selected_rank = safe_float(selected_stats.get("rank", 999), 999)
    opponent_rank = safe_float(opponent_stats.get("rank", 999), 999)

    form_gap = selected_form - opponent_form
    surface_gap = selected_surface - opponent_surface
    elo_gap = clamp((selected_elo - opponent_elo) / 400, -1, 1)
    rank_gap = clamp((opponent_rank - selected_rank) / 250, -1, 1)
    prob_gap = probability - 0.50

    domination_score = (
        prob_gap * 1.35
        + form_gap * 0.42
        + surface_gap * 0.28
        + elo_gap * 0.30
        + rank_gap * 0.18
        + clamp(quality, 0, 1) * 0.08
    )

    # Plus le match est serré, plus 2-1 devient probable.
    if domination_score >= 0.36:
        straight = 0.66
    elif domination_score >= 0.26:
        straight = 0.60
    elif domination_score >= 0.17:
        straight = 0.55
    elif domination_score >= 0.08:
        straight = 0.48
    else:
        straight = 0.40

    # Ajustement : si la forme récente est mauvaise, on baisse le 2-0.
    if selected_form < 0.50:
        straight -= 0.06
    if opponent_form > 0.58:
        straight -= 0.05
    if selected_surface < opponent_surface:
        straight -= 0.04

    straight = clamp(straight, 0.34, 0.70)
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
    odds_source = source_label(row.get("source", "unknown"))

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
        if not is_live_odds_source(odds_source):
            mode = "WATCHLIST"
        stake, stake_percent, kelly = bankroll_management(probability, bookmaker_odds, mode)
        if stake <= 0 and mode in RECOMMENDED_MODES:
            mode = "WATCHLIST"

        selected_stats = probs["home_stats"] if key == "home" else probs["away_stats"]
        opponent_stats = probs["away_stats"] if key == "home" else probs["home_stats"]
        score1, score1_proba, score2, score2_proba = tennis_set_scores(
            probability,
            selected_stats,
            opponent_stats,
            probs["quality"],
        )
        engine_score = round(clamp(probability * 70 + probs["quality"] * 20 + max(value, 0) * 85, 0, 100), 2)
        priority = round(safety * 1.45 + max(value, 0) * 430 + probability * 70, 2)

        rows.append({
            "date": date,
            "sport": sport,
            "odds_source": odds_source,
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
            "calibration_adjustment": "BASELINE",
            "threshold_profile": threshold_profile_label("tennis"),
            "decision_status": "",
            "refusal_reason": "",
            "decision_reason": "",
            "home_recent_form": "",
            "away_recent_form": "",
            "home_recent_attack": "",
            "away_recent_attack": "",
            "home_recent_defense": "",
            "away_recent_defense": "",
            "football_data_quality": "",
            "decision": "VALUE BET" if mode in RECOMMENDED_MODES and stake > 0 else "NO BET",
            "bankroll": round(load_current_bankroll(BANKROLL_START), 2),
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

        rows[-1]["decision_reason"] = build_decision_reason(rows[-1])

    return rows


def normalise_label(value):
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())


def prediction_mode_group(mode):
    text = str(mode or "").upper()
    if "MEGA" in text:
        return "ELITE"
    if "SAFE" in text:
        return "SAFE"
    if "VALUE" in text:
        return "MEDIUM"
    if "RISKY" in text:
        return "RISKY"
    return "OTHER"


def learning_segment_mask(predictions, dimension, segment):
    dim = normalise_label(dimension)
    target = str(segment or "")

    if dim == "sport":
        return predictions["category"].astype(str).str.lower() == target.lower()
    if dim == "competition":
        return predictions["sport"].astype(str).str.lower() == target.lower()
    if dim in {"marche", "market"}:
        return predictions["market"].astype(str).str.lower() == target.lower()
    if dim == "mode de pari":
        return predictions["bet_mode"].apply(prediction_mode_group).astype(str) == target

    return pd.Series(False, index=predictions.index)


def learning_multiplier(segment):
    recommendation = str(segment.get("ai_recommendation", "")).upper()
    if recommendation not in {"BOOST", "REDUCE"}:
        return 1.0, ""

    bets = safe_float(segment.get("bets", 0))
    if bets < 3:
        return 1.0, ""

    roi = safe_float(segment.get("roi", 0))
    winrate = safe_float(segment.get("winrate", 0.5))
    evidence = clamp(bets / 20, 0.25, 1.0)

    if recommendation == "BOOST":
        strength = clamp(0.015 + max(roi, 0) * 0.12 + max(winrate - 0.55, 0) * 0.10, 0.015, 0.07)
        return 1 + strength * evidence, "BOOST"

    strength = clamp(0.025 + max(-roi, 0) * 0.14 + max(0.50 - winrate, 0) * 0.12, 0.025, 0.10)
    return 1 - strength * evidence, "REDUCE"


def append_learning_tag(series, tag):
    def add_tag(value):
        current = str(value or "").strip()
        if current in {"", "nan", "None", "BASELINE"}:
            return tag
        if tag in current.split("|"):
            return current
        return f"{current}|{tag}"

    return series.apply(add_tag)


def apply_global_learning_guard(predictions):
    if not LEARNING_SUMMARY_PATH.exists() or predictions.empty:
        return predictions, False

    try:
        summary = pd.read_csv(LEARNING_SUMMARY_PATH)
    except Exception:
        return predictions, False

    if summary.empty:
        return predictions, False

    row = summary.iloc[0]
    finished = safe_float(row.get("finished_bets", 0))
    if finished < 15:
        return predictions, False

    roi = safe_float(row.get("roi", 0))
    winrate = safe_float(row.get("winrate", 0.5))
    multiplier = 1.0
    tag = ""

    if roi < -0.05 or winrate < 0.48:
        multiplier = 0.96
        tag = "GLOBAL_REDUCE"
    elif roi > 0.08 and winrate > 0.55:
        multiplier = 1.015
        tag = "GLOBAL_BOOST"

    if multiplier == 1.0:
        return predictions, False

    out = predictions.copy()
    out["_learning_multiplier"] *= multiplier
    out["learning_adjustment"] = append_learning_tag(out["learning_adjustment"], tag)
    return out, True


def calibration_key(row):
    return (
        str(row.get("category", "")).lower(),
        str(row.get("market", "")).lower(),
        probability_bucket(row.get("ai_probability", 0)),
    )


def apply_probability_calibration(predictions):
    if not CALIBRATION_PATH.exists() or predictions.empty:
        return predictions

    try:
        calibration = pd.read_csv(CALIBRATION_PATH)
    except Exception:
        return predictions

    required = {"category", "market", "probability_bucket", "bets", "additive_adjustment"}
    if calibration.empty or not required.issubset(set(calibration.columns)):
        return predictions

    usable = calibration.copy()
    usable["bets"] = pd.to_numeric(usable["bets"], errors="coerce").fillna(0)
    usable["additive_adjustment"] = pd.to_numeric(usable["additive_adjustment"], errors="coerce").fillna(0)
    usable = usable[usable["bets"] >= 12]
    if usable.empty:
        return predictions

    table = {}
    category_table = {}
    for _, row in usable.iterrows():
        cat = str(row.get("category", "")).lower()
        market = str(row.get("market", "")).lower()
        bucket = str(row.get("probability_bucket", ""))
        adjustment = clamp(safe_float(row.get("additive_adjustment")), -0.06, 0.04)
        label = f"{adjustment:+.1%} ({int(row.get('bets', 0))} obs)"
        if market == "__all__":
            category_table[(cat, bucket)] = (adjustment, label)
        else:
            table[(cat, market, bucket)] = (adjustment, label)

    out = predictions.copy()
    out["calibration_adjustment"] = out.get("calibration_adjustment", "BASELINE")

    for idx, row in out.iterrows():
        cat, market, bucket = calibration_key(row)
        if cat == "tennis":
            continue
        found = table.get((cat, market, bucket)) or category_table.get((cat, bucket))
        if not found:
            continue

        adjustment, label = found
        probability = safe_float(row.get("ai_probability"))
        out.at[idx, "ai_probability"] = round(clamp(probability + adjustment, 0.02, 0.90), 4)
        out.at[idx, "calibration_adjustment"] = label

    if (out["calibration_adjustment"].astype(str) != "BASELINE").any():
        out = refresh_predictions_after_learning(out)

    return out


def refresh_predictions_after_learning(predictions):
    out = predictions.copy()
    bankroll = load_current_bankroll(BANKROLL_START)

    for idx, row in out.iterrows():
        probability = safe_float(row.get("ai_probability"))
        odds = safe_float(row.get("bookmaker_odds"))
        value = probability * odds - 1 if odds > 0 else -1

        quality = safe_float(row.get("football_data_quality"), 0)
        if quality <= 0:
            tennis_score = safe_float(row.get("tennis_engine_score"), 70)
            quality = clamp(tennis_score / 100, 0.45, 0.95)

        market = row.get("market", "")
        category = row.get("category", "")
        safety = safety_score(probability, value, odds, quality, market)
        mode = select_bet_mode(probability, value, odds, safety, category)
        if not is_live_odds_source(row.get("odds_source", "")):
            mode = "WATCHLIST"
        stake, stake_percent, kelly = bankroll_management(probability, odds, mode, bankroll=bankroll)

        if stake <= 0 and mode in RECOMMENDED_MODES:
            mode = "WATCHLIST"
            stake, stake_percent, kelly = 0.0, 0.0, 0.0

        out.at[idx, "value"] = round(value, 4)
        out.at[idx, "confidence"] = confidence_label(probability)
        out.at[idx, "ia_badge"] = ia_badge(mode)
        out.at[idx, "reliable_only"] = mode in RECOMMENDED_MODES
        out.at[idx, "safety_score"] = safety
        out.at[idx, "safety_level"] = safety_level(mode, safety)
        out.at[idx, "decision"] = "VALUE BET" if mode in RECOMMENDED_MODES and stake > 0 else "NO BET"
        out.at[idx, "bankroll"] = round(bankroll, 2)
        out.at[idx, "stake_percent"] = stake_percent
        out.at[idx, "kelly_fraction"] = kelly
        out.at[idx, "suggested_stake"] = stake
        out.at[idx, "bet_mode"] = mode
        out.at[idx, "priority"] = round(safety * 1.48 + max(value, 0) * 430 + probability * 70, 2)
        out.at[idx, "threshold_profile"] = threshold_profile_label(category)
        out.at[idx, "decision_status"] = decision_status(out.loc[idx])
        out.at[idx, "refusal_reason"] = refusal_reason(out.loc[idx])
        out.at[idx, "decision_reason"] = build_decision_reason(out.loc[idx])

    return out


def apply_learning_adjustment(predictions):
    if predictions.empty:
        return predictions

    out = predictions.copy()
    out["_learning_multiplier"] = 1.0

    learned = False

    if LEARNING_PROFILE_PATH.exists():
        try:
            profile = pd.read_csv(LEARNING_PROFILE_PATH)
        except Exception:
            profile = pd.DataFrame()

        if not profile.empty and "ai_recommendation" in profile.columns:
            for _, segment in profile.iterrows():
                multiplier, tag = learning_multiplier(segment)
                if multiplier == 1.0:
                    continue

                mask = learning_segment_mask(out, segment.get("dimension", ""), segment.get("segment", ""))
                if not mask.any():
                    continue

                out.loc[mask, "_learning_multiplier"] *= multiplier
                out.loc[mask, "learning_adjustment"] = append_learning_tag(out.loc[mask, "learning_adjustment"], tag)
                learned = True

    out, guarded = apply_global_learning_guard(out)
    learned = learned or guarded

    if not learned:
        return out.drop(columns=["_learning_multiplier"], errors="ignore")

    out["ai_probability"] = (
        pd.to_numeric(out["ai_probability"], errors="coerce").fillna(0)
        * pd.to_numeric(out["_learning_multiplier"], errors="coerce").fillna(1)
    ).clip(0.02, 0.90)
    out = refresh_predictions_after_learning(out)
    return out.drop(columns=["_learning_multiplier"], errors="ignore")



def _clean_day(value):
    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt): return str(value)[:10]
    return dt.strftime("%Y-%m-%d")

def one_real_bet_per_match(df, max_bets=10):
    """
    LEVEL MAX :
    - 1 seul vrai pari par match
    - max 5 paris avec mise
    - priorité aux probabilités élevées et cotes raisonnables
    """
    if df.empty:
        return df

    out = df.copy()

    out["_match_key"] = (
        out["sport"].astype(str).str.lower()
        + "|"
        + out["date"].apply(_clean_day).astype(str)
        + "|"
        + out["home_team"].astype(str).str.lower()
        + "|"
        + out["away_team"].astype(str).str.lower()
    )

    for col in ["suggested_stake", "value", "priority", "safety_score", "ai_probability", "bookmaker_odds"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["_min_probability"] = out.apply(
        lambda row: threshold_profile(row.get("category", ""))["value_probability"],
        axis=1,
    )

    out["_is_real_bet"] = (
        out["bet_mode"].isin({"MEGA VALUE", "SAFE PICK", "VALUE BET"})
        & (out["suggested_stake"] > 0)
        & (out["value"] > 0)
        & (out["ai_probability"] >= out["_min_probability"])
        & (out["bookmaker_odds"] >= 1.10)
        & (out["bookmaker_odds"] <= 3.00)
        & (out.get("odds_source", "").apply(is_live_odds_source) if "odds_source" in out.columns else True)
    )

    out["_keep_score"] = (
        out["ai_probability"] * 520
        + out["safety_score"] * 5
        + out["value"].clip(lower=0, upper=0.16) * 180
        - out["bookmaker_odds"] * 12
        + out["priority"] * 2
    )

    for key, g in out.groupby("_match_key"):
        real = g[g["_is_real_bet"]]

        if real.empty:
            out.loc[g.index, "suggested_stake"] = 0
            continue

        best_idx = real["_keep_score"].idxmax()
        other = [idx for idx in g.index if idx != best_idx]

        out.loc[other, "suggested_stake"] = 0
        mask = out.index.isin(other) & out["bet_mode"].isin(RECOMMENDED_MODES)
        out.loc[mask, "bet_mode"] = "WATCHLIST"
        out.loc[mask, "decision"] = "NO BET"

    real_after = out[
        out["bet_mode"].isin({"MEGA VALUE", "SAFE PICK", "VALUE BET"})
        & (out["suggested_stake"] > 0)
        & (out["value"] > 0)
        & (out["ai_probability"] >= out["_min_probability"])
        & (out["bookmaker_odds"] >= 1.10)
        & (out["bookmaker_odds"] <= 3.00)
        & (out.get("odds_source", "").apply(is_live_odds_source) if "odds_source" in out.columns else True)
    ].sort_values("_keep_score", ascending=False)

    keep_indices = set(real_after.head(max_bets).index)

    cut_mask = (
        out["bet_mode"].isin(RECOMMENDED_MODES)
        & (out["suggested_stake"] > 0)
        & (~out.index.isin(keep_indices))
    )

    out.loc[cut_mask, "suggested_stake"] = 0
    out.loc[cut_mask, "bet_mode"] = "WATCHLIST"
    out.loc[cut_mask, "decision"] = "NO BET"

    return out.drop(columns=["_match_key", "_is_real_bet", "_keep_score", "_min_probability"], errors="ignore")


def cap_stakes_to_bankroll(df, bankroll):
    """
    Protection de balance :
    - max 30% exposé / jour
    - jamais plus de 25% sur un seul pari
    """
    if df.empty:
        return df

    out = df.copy()

    if "suggested_stake" not in out.columns:
        return out

    out["suggested_stake"] = pd.to_numeric(out["suggested_stake"], errors="coerce").fillna(0)

    if bankroll <= 0:
        out["suggested_stake"] = 0
        if "decision" in out.columns:
            out["decision"] = "NO BET"
        if "bet_mode" in out.columns:
            out["bet_mode"] = out["bet_mode"].replace({
                "MEGA VALUE": "WATCHLIST",
                "SAFE PICK": "WATCHLIST",
                "VALUE BET": "WATCHLIST",
                "RISKY VALUE": "WATCHLIST",
            })
        return out

    max_daily_exposure = bankroll * 0.30
    max_single_bet = bankroll * 0.25

    out["suggested_stake"] = out["suggested_stake"].clip(lower=0, upper=max_single_bet)

    total = out["suggested_stake"].sum()
    if total > max_daily_exposure:
        factor = max_daily_exposure / total
        out["suggested_stake"] = (out["suggested_stake"] * factor).round(2)

    small = out["suggested_stake"] < 0.10
    out.loc[small, "suggested_stake"] = 0

    if "bet_mode" in out.columns:
        mask = small & out["bet_mode"].isin(RECOMMENDED_MODES)
        out.loc[mask, "bet_mode"] = "WATCHLIST"
        if "decision" in out.columns:
            out.loc[mask, "decision"] = "NO BET"

    return out


def force_daily_best_bet(df):
    """
    Si aucune mise ne sort, on force le meilleur spot VALABLE :
    - proba >= 58%
    - cote >= 1.10
    - value positive
    Ça évite un dashboard vide, sans jouer n'importe quoi.
    """
    if df.empty:
        return df

    out = df.copy()
    for col in ["suggested_stake", "ai_probability", "bookmaker_odds", "value", "safety_score", "priority"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["_min_probability"] = out.apply(
        lambda row: threshold_profile(row.get("category", ""))["value_probability"],
        axis=1,
    )

    current_real = out[
        out["bet_mode"].isin(RECOMMENDED_MODES)
        & (out["suggested_stake"] > 0)
        & (out["value"] > 0)
    ]
    if not current_real.empty:
        return out.drop(columns=["_min_probability"], errors="ignore")

    candidates = out[
        (out["ai_probability"] >= out["_min_probability"])
        & (out["bookmaker_odds"] >= 1.10)
        & (out["value"] > 0)
        & (out.get("odds_source", "").apply(is_live_odds_source) if "odds_source" in out.columns else True)
    ].copy()

    if candidates.empty:
        return out.drop(columns=["_min_probability"], errors="ignore")

    candidates["_force_score"] = (
        candidates["ai_probability"] * 500
        + candidates["safety_score"] * 4
        + candidates["value"].clip(lower=0, upper=0.20) * 200
        - candidates["bookmaker_odds"] * 8
        + candidates["priority"] * 2
    )
    best_idx = candidates["_force_score"].idxmax()
    prob = float(out.loc[best_idx, "ai_probability"])
    val = float(out.loc[best_idx, "value"])
    odds = float(out.loc[best_idx, "bookmaker_odds"])

    mode = select_bet_mode(prob, val, odds, out.loc[best_idx, "safety_score"], out.loc[best_idx].get("category", ""))
    if mode not in RECOMMENDED_MODES:
        mode = "VALUE BET"

    stake, stake_percent, kelly = bankroll_management(prob, odds, mode)
    out.loc[best_idx, "bet_mode"] = mode
    out.loc[best_idx, "decision"] = "VALUE BET"
    out.loc[best_idx, "ia_badge"] = ia_badge(mode)
    out.loc[best_idx, "safety_level"] = safety_level(mode, out.loc[best_idx, "safety_score"])
    out.loc[best_idx, "suggested_stake"] = stake
    out.loc[best_idx, "stake_percent"] = stake_percent
    out.loc[best_idx, "kelly_fraction"] = kelly

    return out.drop(columns=["_min_probability"], errors="ignore")


def finalise_predictions(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = apply_probability_calibration(df)
    df = apply_learning_adjustment(df)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["last_update"] = datetime.now().strftime("%d/%m/%Y %H:%M")
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(0)
    df["suggested_stake"] = pd.to_numeric(df["suggested_stake"], errors="coerce").fillna(0)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df["ai_probability"] = pd.to_numeric(df["ai_probability"], errors="coerce").fillna(0)

    df = one_real_bet_per_match(df)
    df = force_daily_best_bet(df)

    df = df.sort_values(
        ["priority", "suggested_stake", "value", "ai_probability"],
        ascending=[False, False, False, False],
    )

    df = cap_stakes_to_bankroll(df, load_current_bankroll(BANKROLL_START))
    df["threshold_profile"] = df["category"].apply(threshold_profile_label)
    df["decision_status"] = df.apply(decision_status, axis=1)
    df["refusal_reason"] = df.apply(refusal_reason, axis=1)
    df["decision_reason"] = df.apply(build_decision_reason, axis=1)
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
