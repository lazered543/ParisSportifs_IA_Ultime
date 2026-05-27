from pathlib import Path
from datetime import datetime, timedelta, timezone
import math
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.football_features import build_team_strength, estimate_xg
from src.models.poisson import football_poisson_probs
from src.models.elo import build_elo_ratings, get_match_elo
from src.models.calibration import calibrate_probability
from src.models.xgboost_model import load_xgboost_model

from src.betting.value_bet import value_score, classify_confidence
from src.betting.bankroll import safe_stake

from src.utils.config import BANKROLL_START, MAX_STAKE_PCT, MIN_VALUE, MIN_CONFIDENCE


# ============================================================
# LOADERS
# ============================================================

def load_player_scorers():
    path = Path("data/processed/player_scorers.csv")

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def load_or_demo_upcoming():
    path = Path("data/processed/upcoming_odds.csv")

    if path.exists():
        df = pd.read_csv(path)

        if not df.empty:
            before = len(df)
            df = filter_upcoming_window(df)
            print(f"Matchs gardés après filtre dynamique : {len(df)} / {before}")
            return df

    return pd.DataFrame([])


def load_tennis_history():
    possible_paths = [
        Path("data/processed/tennis_history_all.csv"),
        Path("data/processed/tennis_history.csv"),
        Path("data/raw/tennis_history.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)

                if not df.empty:
                    print("Historique tennis chargé :", path)
                    return df

            except Exception as e:
                print("Erreur lecture historique tennis :", e)

    print("Aucun historique tennis trouvé.")
    return pd.DataFrame()


# ============================================================
# UTILS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def clamp(value, low, high):
    return max(low, min(high, value))


def is_tennis_sport(sport):
    return "tennis" in str(sport).lower()


def is_football_sport(sport):
    return "soccer" in str(sport).lower()


def normalize_name(name):
    return str(name).strip().lower()


def american_to_probability_from_decimal(odds):
    odds = safe_float(odds, 0)

    if odds <= 1:
        return 0

    return 1 / odds


def normalized_market_probabilities(odds_home, odds_draw, odds_away):
    implied = {
        "Home Win": american_to_probability_from_decimal(odds_home),
        "Draw": american_to_probability_from_decimal(odds_draw),
        "Away Win": american_to_probability_from_decimal(odds_away),
    }

    total = sum(v for v in implied.values() if v > 0)

    if total <= 0:
        return {
            "Home Win": 0.36,
            "Draw": 0.28,
            "Away Win": 0.36,
        }

    return {
        market: prob / total
        for market, prob in implied.items()
        if prob > 0
    }


# ============================================================
# MATCH FILTERING / PRIORITY
# ============================================================

PAST_GRACE_HOURS = 4

def dynamic_hours_window(sport):
    """
    Fenêtre dynamique :
    - Tennis : 48h pour garder les matchs du jour / lendemain
    - Foot classique : 7 jours pour garder Ligue 1, Ligue 2, barrages, week-end
    - World Cup / international : 10 jours car les calendriers sortent plus tôt
    """
    sport = str(sport).lower()

    if "tennis" in sport:
        return 48

    if "world_cup" in sport or "international" in sport:
        return 240  # 10 jours

    if "soccer" in sport or "football" in sport:
        return 168  # 7 jours

    return 72

BIG_PLAYERS = [
    "djokovic", "alcaraz", "sinner", "nadal", "federer", "medvedev", "zverev",
    "tsitsipas", "rune", "rublev", "fritz", "ruud", "de minaur", "hurkacz",
    "swiatek", "sabalenka", "gauff", "rybakina", "pegula", "paolini",
    "osaka", "raducanu", "andreeva", "jabeur", "ostapenko",
]

BIG_COMPETITIONS = [
    "french_open", "roland", "wimbledon", "us_open", "australian_open",
    "champions_league", "premier_league", "laliga", "serie_a", "bundesliga",
    "ligue_one", "ligue_two", "ligue_2", "france", "world_cup",
]


def parse_match_datetime(value):
    try:
        if pd.isna(value):
            return pd.NaT

        return pd.to_datetime(value, utc=True, errors="coerce")

    except Exception:
        return pd.NaT


def priority_score(home_team, away_team, sport):
    text = f"{home_team} {away_team} {sport}".lower()
    score = 0

    for player in BIG_PLAYERS:
        if player in text:
            score += 120

    for competition in BIG_COMPETITIONS:
        if competition in text:
            score += 40

    if "tennis" in text:
        score += 15

    if "soccer" in text or "football" in text:
        score += 10

    return score


def filter_upcoming_window(df):
    """
    Garde les matchs utiles sans tout couper :
    - tennis proche uniquement
    - football sur 7 jours
    - Coupe du monde / international sur 10 jours
    """
    if df.empty or "commence_time" not in df.columns:
        return df

    out = df.copy()
    out["_dt"] = out["commence_time"].apply(parse_match_datetime)

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=PAST_GRACE_HOURS)

    valid = out["_dt"].notna()

    out["_max_hours"] = out["sport"].apply(dynamic_hours_window)
    out["_end"] = out["_max_hours"].apply(lambda h: now + timedelta(hours=int(h)))

    near = out[
        valid
        & (out["_dt"] >= start)
        & (out["_dt"] <= out["_end"])
    ].copy()

    if near.empty:
        print("Aucun match dans la fenêtre dynamique. Vérifie data/processed/upcoming_odds.csv.")
        return near.drop(columns=["_dt", "_max_hours", "_end"], errors="ignore")

    near["_priority"] = near.apply(
        lambda r: priority_score(
            r.get("home_team", ""),
            r.get("away_team", ""),
            r.get("sport", "")
        ),
        axis=1,
    )

    near = near.sort_values(
        ["_priority", "_dt"],
        ascending=[False, True]
    )

    print(
        "Fenêtre dynamique :",
        "tennis=48h, foot=7j, international=10j",
        "| matchs gardés =",
        len(near),
    )

    return near.drop(columns=["_dt", "_max_hours", "_end"], errors="ignore")

    near["_priority"] = near.apply(
        lambda r: priority_score(
            r.get("home_team", ""),
            r.get("away_team", ""),
            r.get("sport", "")
        ),
        axis=1,
    )

    near = near.sort_values(
        ["_priority", "_dt"],
        ascending=[False, True]
    )

    return near.drop(columns=["_dt"], errors="ignore")


def odds_based_expected_goals(odds_home, odds_draw, odds_away, home_profile=None, away_profile=None):
    """
    Estime des xG réalistes quand l'historique équipe est faible ou absent.
    Logique :
    - les cotes donnent la force relative,
    - le nul donne une indication sur le total de buts,
    - la forme récente ajuste légèrement attaque/défense.
    """
    market_probs = normalized_market_probabilities(odds_home, odds_draw, odds_away)

    if not isinstance(market_probs, dict):
        market_probs = {
            "Home Win": 0.36,
            "Draw": 0.28,
            "Away Win": 0.36,
        }

    p_home = market_probs.get("Home Win", 0.36)
    p_draw = market_probs.get("Draw", 0.28)
    p_away = market_probs.get("Away Win", 0.36)

    # Total de buts attendu : si le nul est très haut, match plus ouvert.
    # Si le nul est bas, match plus fermé/équilibré.
    total_goals = 2.35 + (0.28 - p_draw) * 1.25
    total_goals = clamp(total_goals, 1.80, 3.35)

    dominance = clamp(p_home - p_away, -0.45, 0.45)

    home_xg = total_goals * (0.50 + dominance * 0.55)
    away_xg = total_goals - home_xg

    if home_profile and away_profile:
        form_edge = home_profile.get("form_points", 0.5) - away_profile.get("form_points", 0.5)
        attack_edge = home_profile.get("attack_recent", 1.2) - away_profile.get("defense_recent", 1.2)
        away_attack_edge = away_profile.get("attack_recent", 1.2) - home_profile.get("defense_recent", 1.2)
        quality = min(home_profile.get("data_quality", 0.25), away_profile.get("data_quality", 0.25))

        home_xg += (form_edge * 0.28 + attack_edge * 0.08) * quality
        away_xg += (-form_edge * 0.22 + away_attack_edge * 0.08) * quality

    home_xg = clamp(home_xg, 0.25, 3.60)
    away_xg = clamp(away_xg, 0.25, 3.60)

    return round(home_xg, 3), round(away_xg, 3)


def blend_xg(history_xg, odds_xg, data_quality):
    """
    Mélange xG historique et xG marché.
    Si peu de données : on écoute davantage le marché.
    Si beaucoup de données : on garde plus d'historique.
    """
    hxg = safe_float(history_xg, 1.2)
    oxg = safe_float(odds_xg, 1.2)
    q = clamp(safe_float(data_quality, 0.25), 0, 1)

    hist_weight = 0.25 + 0.50 * q
    odds_weight = 1 - hist_weight

    return round(clamp(hxg * hist_weight + oxg * odds_weight, 0.20, 3.80), 3)


def format_top_scores(top_scores):
    clean = []
    for score, proba in top_scores[:5]:
        clean.append((score, round(float(proba), 4)))
    return str(clean)



def team_variation_seed(home, away):
    """
    Variation stable par match, pas du hasard pur.
    Même match = même variation.
    """
    raw = f"{home}-{away}".lower()
    return sum(ord(c) for c in raw) % 100


def real_match_simulation_scores(home, away, home_xg, away_xg, home_prob, draw_prob, away_prob, elo_diff):
    """
    Simulation score exact plus réaliste :
    - xG
    - domination
    - total buts attendu
    - draw probability
    - clean sheet
    - variation stable par match pour éviter le clonage
    """

    home_xg = safe_float(home_xg, 1.35)
    away_xg = safe_float(away_xg, 1.05)
    home_prob = safe_float(home_prob, 0.40)
    draw_prob = safe_float(draw_prob, 0.28)
    away_prob = safe_float(away_prob, 0.32)
    elo_diff = safe_float(elo_diff, 0)

    seed = team_variation_seed(home, away)
    total_xg = home_xg + away_xg
    dominance = home_prob - away_prob

    # Variation stable : donne un profil différent selon le match
    variation = (seed % 7) - 3
    openness = total_xg + variation * 0.06

    home_strength = home_xg + max(elo_diff, 0) / 450
    away_strength = away_xg + max(-elo_diff, 0) / 450

    clean_sheet_home = clamp(
        0.38 + (home_strength - away_strength) * 0.16 - max(openness - 2.8, 0) * 0.10,
        0.12,
        0.62,
    )

    clean_sheet_away = clamp(
        0.30 + (away_strength - home_strength) * 0.14 - max(openness - 2.8, 0) * 0.10,
        0.08,
        0.55,
    )

    candidates = []

    def add(score, weight):
        candidates.append((score, max(weight, 0.001)))

    # HOME FAVORITE
    if dominance > 0.10:
        fav = dominance + max(elo_diff, 0) / 600

        # Favori léger : diversité 1-0 / 2-1 / 1-1
        if fav < 0.18:
            add("1-0", 0.25 + clean_sheet_home * 0.12)
            add("2-1", 0.24 + max(openness - 2.4, 0) * 0.08)
            add("1-1", 0.20 + draw_prob * 0.20)
            add("2-0", 0.16 + clean_sheet_home * 0.08)

        # Favori moyen : alterne 2-0 / 2-1 / 3-1 selon profil
        elif fav < 0.30:
            if seed % 3 == 0:
                add("2-1", 0.30)
                add("2-0", 0.23)
                add("1-0", 0.18)
                add("3-1", 0.14)
            elif seed % 3 == 1:
                add("2-0", 0.29)
                add("1-0", 0.21)
                add("2-1", 0.20)
                add("3-0", 0.13)
            else:
                add("3-1", 0.25 if openness >= 2.65 else 0.14)
                add("2-1", 0.25)
                add("2-0", 0.22)
                add("1-0", 0.13)

        # Gros favori
        else:
            add("3-0", 0.26)
            add("2-0", 0.25)
            add("3-1", 0.20)
            add("4-1", 0.08 if openness >= 3 else 0.03)

    # AWAY FAVORITE
    elif dominance < -0.10:
        fav = abs(dominance) + max(-elo_diff, 0) / 600

        if fav < 0.18:
            add("0-1", 0.25 + clean_sheet_away * 0.12)
            add("1-2", 0.24 + max(openness - 2.4, 0) * 0.08)
            add("1-1", 0.20 + draw_prob * 0.20)
            add("0-2", 0.16 + clean_sheet_away * 0.08)

        elif fav < 0.30:
            if seed % 3 == 0:
                add("1-2", 0.30)
                add("0-2", 0.23)
                add("0-1", 0.18)
                add("1-3", 0.14)
            elif seed % 3 == 1:
                add("0-2", 0.29)
                add("0-1", 0.21)
                add("1-2", 0.20)
                add("0-3", 0.13)
            else:
                add("1-3", 0.25 if openness >= 2.65 else 0.14)
                add("1-2", 0.25)
                add("0-2", 0.22)
                add("0-1", 0.13)

        else:
            add("0-3", 0.26)
            add("0-2", 0.25)
            add("1-3", 0.20)
            add("1-4", 0.08 if openness >= 3 else 0.03)

    # BALANCED MATCH
    else:
        if draw_prob >= 0.30:
            add("1-1", 0.32)
            add("0-0", 0.18 if openness < 2.45 else 0.08)
            add("2-2", 0.15 if openness >= 2.75 else 0.06)
            add("1-0", 0.14)
            add("0-1", 0.14)
        else:
            add("2-1", 0.20 + max(home_xg - away_xg, 0) * 0.10)
            add("1-2", 0.20 + max(away_xg - home_xg, 0) * 0.10)
            add("1-1", 0.20)
            add("2-2", 0.12 if openness >= 2.75 else 0.04)

    # Match très ouvert : ajoute des scores avec buts des deux côtés
    if openness >= 2.90:
        if dominance >= 0:
            add("3-2", 0.09)
            add("2-2", 0.08)
        else:
            add("2-3", 0.09)
            add("2-2", 0.08)

    # Match fermé : boost scores bas
    if openness <= 2.15:
        add("1-0", 0.12 if dominance >= 0 else 0.04)
        add("0-1", 0.12 if dominance <= 0 else 0.04)
        add("0-0", 0.08)
        add("1-1", 0.10)

    merged = {}
    for score, weight in candidates:
        merged[score] = merged.get(score, 0) + weight

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    total_weight = sum(w for _, w in ranked)
    result = []

    for score, weight in ranked[:5]:
        proba = (weight / total_weight) * 0.42
        proba = clamp(proba, 0.045, 0.17)
        result.append((score, round(proba, 4)))

    return result




# ============================================================
# IA AUTO-CORRECTIVE / CALIBRATION
# ============================================================

def load_tracking_learning():
    path = Path("tracking_results.csv")

    learning = {
        "global_bets": 0,
        "global_winrate": 0.50,
        "global_roi": 0.0,
        "sport_stats": {},
        "mode_stats": {},
        "prob_bins": {},
        "blacklist_sports": set(),
        "reduce_sports": set(),
        "reduce_modes": set(),
    }

    if not path.exists():
        return learning

    try:
        tr = pd.read_csv(path, low_memory=False)
    except Exception:
        return learning

    if tr.empty or "result" not in tr.columns:
        return learning

    tr = tr.copy()
    tr["result"] = tr["result"].fillna("").astype(str).str.upper()
    finished = tr[tr["result"].isin(["WIN", "LOSS"])].copy()

    if finished.empty:
        return learning

    finished["stake"] = pd.to_numeric(finished.get("stake", 0), errors="coerce").fillna(0)
    finished["profit"] = pd.to_numeric(finished.get("profit", 0), errors="coerce").fillna(0)
    finished["ai_probability"] = pd.to_numeric(finished.get("ai_probability", 0), errors="coerce").fillna(0)

    if "sport" not in finished.columns:
        finished["sport"] = ""
    if "bet_mode" not in finished.columns:
        finished["bet_mode"] = ""

    finished["sport"] = finished["sport"].fillna("").astype(str)
    finished["bet_mode"] = finished["bet_mode"].fillna("").astype(str)

    total_stake = finished["stake"].sum()
    total_profit = finished["profit"].sum()

    learning["global_bets"] = int(len(finished))
    learning["global_winrate"] = round(float((finished["result"] == "WIN").mean()), 4)
    learning["global_roi"] = round(float(total_profit / total_stake), 4) if total_stake > 0 else 0.0

    for sport, g in finished.groupby("sport"):
        stake = g["stake"].sum()
        profit = g["profit"].sum()
        bets = len(g)
        winrate = (g["result"] == "WIN").mean()
        roi = profit / stake if stake > 0 else 0

        learning["sport_stats"][sport] = {
            "bets": int(bets),
            "winrate": round(float(winrate), 4),
            "roi": round(float(roi), 4),
            "profit": round(float(profit), 2),
        }

        if bets >= 30 and roi <= -0.15:
            learning["blacklist_sports"].add(sport)
        elif bets >= 8 and roi <= -0.12:
            learning["reduce_sports"].add(sport)

    for mode, g in finished.groupby("bet_mode"):
        stake = g["stake"].sum()
        profit = g["profit"].sum()
        bets = len(g)
        winrate = (g["result"] == "WIN").mean()
        roi = profit / stake if stake > 0 else 0

        learning["mode_stats"][mode] = {
            "bets": int(bets),
            "winrate": round(float(winrate), 4),
            "roi": round(float(roi), 4),
            "profit": round(float(profit), 2),
        }

        if bets >= 6 and roi <= -0.12:
            learning["reduce_modes"].add(mode)

    bins = [
        (0.00, 0.54, "0-54"),
        (0.54, 0.58, "54-58"),
        (0.58, 0.62, "58-62"),
        (0.62, 0.66, "62-66"),
        (0.66, 0.72, "66-72"),
        (0.72, 1.00, "72+"),
    ]

    for low, high, label in bins:
        g = finished[
            (finished["ai_probability"] >= low)
            & (finished["ai_probability"] < high)
        ]

        if g.empty:
            continue

        predicted = g["ai_probability"].mean()
        actual = (g["result"] == "WIN").mean()

        learning["prob_bins"][label] = {
            "bets": int(len(g)),
            "predicted": round(float(predicted), 4),
            "actual": round(float(actual), 4),
            "gap": round(float(actual - predicted), 4),
        }

    return learning


def probability_bin(prob):
    prob = safe_float(prob, 0)

    if prob < 0.54:
        return "0-54"
    if prob < 0.58:
        return "54-58"
    if prob < 0.62:
        return "58-62"
    if prob < 0.66:
        return "62-66"
    if prob < 0.72:
        return "66-72"

    return "72+"


def apply_learning_calibration(prob, sport, mode, learning):
    prob = safe_float(prob, 0.5)
    sport = str(sport)
    mode = str(mode)

    if learning is None or learning.get("global_bets", 0) <= 0:
        return round(clamp(prob, 0.03, 0.92), 4), "NO_LEARNING_YET"

    correction = 0.0
    reasons = []

    bin_name = probability_bin(prob)
    bin_stats = learning.get("prob_bins", {}).get(bin_name)

    if bin_stats and bin_stats.get("bets", 0) >= 3:
        gap = safe_float(bin_stats.get("gap", 0), 0)
        weight = min(bin_stats.get("bets", 0) / 30, 0.65)
        correction += gap * weight
        reasons.append(f"CALIB_{bin_name}")

    sport_stats = learning.get("sport_stats", {}).get(sport)

    if sport_stats:
        bets = sport_stats.get("bets", 0)
        roi = safe_float(sport_stats.get("roi", 0), 0)
        winrate = safe_float(sport_stats.get("winrate", 0.5), 0.5)

        if bets >= 8 and roi < -0.10:
            correction -= min(abs(roi) * 0.18, 0.045)
            reasons.append("SPORT_REDUCED")

        if bets >= 8 and winrate < 0.45:
            correction -= 0.025
            reasons.append("SPORT_LOW_WINRATE")

    mode_stats = learning.get("mode_stats", {}).get(mode)

    if mode_stats:
        bets = mode_stats.get("bets", 0)
        roi = safe_float(mode_stats.get("roi", 0), 0)

        if bets >= 6 and roi < -0.10:
            correction -= min(abs(roi) * 0.14, 0.040)
            reasons.append("MODE_REDUCED")

    if prob < 0.54:
        correction -= 0.030
        reasons.append("COINFLIP_PENALTY")

    if learning.get("global_bets", 0) >= 8 and learning.get("global_roi", 0) < -0.10:
        correction -= 0.015
        reasons.append("GLOBAL_PRUDENCE")

    calibrated = round(clamp(prob + correction, 0.03, 0.92), 4)
    return calibrated, "|".join(reasons) if reasons else "OK"


def learning_adjusted_mode(mode, prob, value, sport, learning, reason):
    sport = str(sport)
    mode = str(mode)

    if learning and sport in learning.get("blacklist_sports", set()):
        return "NO BET"

    if prob < 0.54:
        return "WATCHLIST"

    if "SPORT_REDUCED" in str(reason) and mode in ["RISKY VALUE", "VALUE BET"]:
        return "WATCHLIST"

    if "MODE_REDUCED" in str(reason) and mode == "RISKY VALUE":
        return "WATCHLIST"

    if value < 0.02 and mode == "VALUE BET":
        return "WATCHLIST"

    return mode


def learning_summary_to_csv(learning):
    out_dir = Path("data/learning")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for sport, stats in learning.get("sport_stats", {}).items():
        rows.append({
            "dimension": "sport",
            "segment": sport,
            **stats,
            "action": (
                "BLACKLIST" if sport in learning.get("blacklist_sports", set())
                else "REDUCE" if sport in learning.get("reduce_sports", set())
                else "KEEP"
            ),
        })

    for mode, stats in learning.get("mode_stats", {}).items():
        rows.append({
            "dimension": "mode",
            "segment": mode,
            **stats,
            "action": (
                "REDUCE" if mode in learning.get("reduce_modes", set())
                else "KEEP"
            ),
        })

    for bin_name, stats in learning.get("prob_bins", {}).items():
        rows.append({
            "dimension": "probability_bin",
            "segment": bin_name,
            "bets": stats.get("bets", 0),
            "winrate": stats.get("actual", 0),
            "roi": "",
            "profit": "",
            "predicted": stats.get("predicted", 0),
            "actual": stats.get("actual", 0),
            "gap": stats.get("gap", 0),
            "action": "CALIBRATE",
        })

    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "ai_auto_learning_segments.csv", index=False)

# ============================================================
# BET MODES
# ============================================================

def bet_mode(prob, odds, value, confidence, sport):
    """
    Version plus stricte :
    - SAFE PICK = pari assez sûr
    - VALUE BET = vraie value raisonnable
    - RISKY VALUE = visible dans le dashboard mais très petite mise
    - WATCHLIST / NO BET = pas conseillé
    """
    sport = str(sport).lower()
    confidence = str(confidence)

    if odds <= 1 or odds > 4.50:
        return "NO BET"

    risky_competition = any(x in sport for x in [
        "itf",
        "challenger",
        "friendly",
    ])

    if risky_competition:
        return "NO BET"

    if confidence in ["A éviter", "A Ã©viter"]:
        return "NO BET"

    # Très bon spot : proba solide + vraie value
    if prob >= 0.64 and value >= 0.055 and 1.30 <= odds <= 2.90:
        return "MEGA VALUE"

    # Match plus sûr : proba haute, cote correcte, value pas négative
    if prob >= 0.60 and value >= 0.000 and 1.18 <= odds <= 2.20:
        return "SAFE PICK"

    # Value raisonnable : edge positif réel
    if prob >= 0.545 and value >= 0.030 and 1.30 <= odds <= 3.20:
        return "VALUE BET"

    # Risqué : affiché mais pas pari principal
    if prob >= 0.49 and value >= 0.060 and 1.80 <= odds <= 3.80:
        return "RISKY VALUE"

    if prob >= 0.55 and 1.18 <= odds <= 3.60:
        return "WATCHLIST"

    return "NO BET"


# ============================================================
# BANKROLL MANAGEMENT IA
# ============================================================

def bankroll_management(prob, odds, value, mode, bankroll):
    """
    Bankroll plus logique :
    - RISKY = très petite mise, jamais gros montant
    - SAFE / MEGA = mises plus élevées
    - VALUE = intermédiaire
    """

    if odds <= 1 or mode in ["NO BET", "WATCHLIST"]:
        return 0.0, 0.0, 0.0

    b = odds - 1
    p = prob
    q = 1 - p

    kelly = ((b * p) - q) / b

    if kelly <= 0:
        return 0.0, 0.0, 0.0

    kelly = max(0.0, min(kelly, 0.10))

    # Fractions plus prudentes
    fractions = {
        "MEGA VALUE": 0.30,
        "SAFE PICK": 0.24,
        "VALUE BET": 0.14,
        "RISKY VALUE": 0.035,
        "WATCHLIST": 0.0,
    }

    # Plafonds beaucoup plus logiques
    max_by_mode = {
        "MEGA VALUE": 0.030,   # max 3.00€ pour 100€
        "SAFE PICK": 0.022,    # max 2.20€
        "VALUE BET": 0.014,    # max 1.40€
        "RISKY VALUE": 0.005,  # max 0.50€
        "WATCHLIST": 0.0,
    }

    fraction = fractions.get(mode, 0.0)
    max_percent = max_by_mode.get(mode, 0.0)

    stake_percent = kelly * fraction
    stake_percent = max(0.0, min(stake_percent, max_percent))

    if stake_percent <= 0:
        return 0.0, 0.0, round(kelly, 4)

    stake = bankroll * stake_percent

    # Plancher uniquement sur les vrais paris, mais RISKY reste faible
    if mode == "RISKY VALUE":
        stake = min(max(stake, 0.30), 0.50)
    elif mode == "VALUE BET":
        stake = min(max(stake, 0.60), 1.40)
    elif mode == "SAFE PICK":
        stake = min(max(stake, 1.00), 2.20)
    elif mode == "MEGA VALUE":
        stake = min(max(stake, 1.50), 3.00)

    return round(stake, 2), round(stake / bankroll, 4), round(kelly, 4)


# ============================================================
# BADGES / FILTERS
# ============================================================

def mode_badge(mode, value, confidence, odds):
    mode = str(mode).upper().strip()

    if mode == "MEGA VALUE":
        return "💎 MEGA VALUE"

    if mode == "SAFE PICK":
        if value >= 0:
            return "🟢 SAFE PICK"
        return "🟢 SAFE PICK / COTE FAIBLE"

    if mode == "VALUE BET":
        return "🟡 VALUE BET"

    if mode == "RISKY VALUE":
        return "🔴 RISKY VALUE"

    if mode == "WATCHLIST":
        return "👀 WATCHLIST"

    return "⚪ NO BET"


def ia_badge(value, confidence, odds):
    if value >= 0.07 and confidence in ["Moyen", "Fort", "Elite"] and 1.35 <= odds <= 3.40:
        return "🟢 STRONG VALUE"

    if value >= 0.015 and confidence in ["Moyen", "Fort", "Elite"] and 1.30 <= odds <= 3.60:
        return "🟡 VALUE"

    if value >= 0.05:
        return "🔴 RISKY VALUE"

    return "⚪ WATCH / NO VALUE"


def reliable_filter(decision, confidence, odds, value, prob):
    return (
        decision == "VALUE BET"
        and confidence in ["Moyen", "Fort", "Elite"]
        and 1.20 <= odds <= 3.20
        and value >= 0.030
        and prob >= 0.545
    )


def safe_value(prob, odds):
    if odds <= 1:
        return 0

    value = value_score(prob, odds)

    return round(clamp(value, -0.50, 0.80), 4)


def safety_score(prob, odds, value, confidence, mode, reliable):
    odds = safe_float(odds, 0)
    prob = safe_float(prob, 0)
    value = safe_float(value, 0)

    confidence_bonus = {
        "Elite": 18,
        "Fort": 13,
        "Moyen": 7,
        "Faible": 0,
        "A Ã©viter": -12,
        "A éviter": -12,
    }.get(str(confidence), 0)

    mode_bonus = {
        "MEGA VALUE": 14,
        "SAFE PICK": 12,
        "VALUE BET": 8,
        "RISKY VALUE": -1,
        "WATCHLIST": 1,
        "NO BET": -16,
    }.get(str(mode), 0)

    if 1.25 <= odds <= 2.20:
        odds_bonus = 10
    elif 2.20 < odds <= 3.20:
        odds_bonus = 5
    elif 1.01 <= odds < 1.25:
        odds_bonus = -2
    elif odds > 3.20:
        odds_bonus = -8
    else:
        odds_bonus = -20

    score = (
        prob * 62
        + clamp(value, -0.25, 0.35) * 70
        + confidence_bonus
        + mode_bonus
        + odds_bonus
        + (6 if reliable else 0)
    )

    return round(clamp(score, 0, 100), 2)


def safety_level(score, mode, decision, prob=None, value=None):
    mode = str(mode).upper().strip()
    prob = safe_float(prob, 0)
    value = safe_float(value, 0)

    if mode == "MEGA VALUE":
        return "1 - MEGA VALUE"

    if mode == "SAFE PICK":
        return "2 - SAFE PICK"

    if mode == "VALUE BET":
        return "3 - VALUE BET"

    if mode == "RISKY VALUE":
        return "4 - RISKY VALUE"

    if mode == "WATCHLIST":
        if prob >= 0.65 and value < 0:
            return "5 - PROBABLE MAIS COTE FAIBLE"
        return "5 - WATCHLIST"

    if prob >= 0.68 and value < 0:
        return "6 - PROBABLE SANS VALUE"

    return "7 - A EVITER"



def poisson_pmf(k, lam):
    lam = max(float(lam), 0.05)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def true_poisson_exact_scores(home_xg, away_xg, home_prob=None, draw_prob=None, away_prob=None):
    """
    VRAI moteur score exact :
    - aucun score forcé
    - calcule chaque score 0-0 à 5-5 avec loi de Poisson
    - ajuste légèrement selon les probabilités 1N2 du modèle
    - retourne les scores réellement les plus probables
    """
    home_xg = clamp(safe_float(home_xg, 1.35), 0.20, 3.20)
    away_xg = clamp(safe_float(away_xg, 1.05), 0.20, 3.20)

    home_prob = safe_float(home_prob, 0.0)
    draw_prob = safe_float(draw_prob, 0.0)
    away_prob = safe_float(away_prob, 0.0)

    scores = []

    for h in range(0, 6):
        for a in range(0, 6):
            p = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)

            # Petit ajustement de cohérence 1N2, sans forcer le résultat
            if h > a and home_prob > 0:
                p *= 0.85 + home_prob * 0.35
            elif h == a and draw_prob > 0:
                p *= 0.85 + draw_prob * 0.45
            elif a > h and away_prob > 0:
                p *= 0.85 + away_prob * 0.35

            # Les 5-5 et scores ultra hauts sont rarement le top exact
            if h + a >= 7:
                p *= 0.75

            scores.append((f"{h}-{a}", p))

    total = sum(p for _, p in scores)

    if total <= 0:
        return [("1-1", 0.10), ("1-0", 0.09), ("2-1", 0.08)]

    scores = [(s, p / total) for s, p in scores]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Évite les doublons absurdes et garde les 5 meilleurs
    return [(s, round(float(p), 4)) for s, p in scores[:5]]


def market_total_goal_adjustment(odds_home, odds_draw, odds_away, base_home_xg, base_away_xg):
    """
    Ajuste le total de buts selon les cotes 1N2 :
    - nul assez fort => match plus fermé
    - favori net => domination plus nette
    - mais sans inventer 3-0 partout
    """
    probs = normalized_market_probabilities(odds_home, odds_draw, odds_away)

    if not isinstance(probs, dict):
        probs = {
            "Home Win": 0.36,
            "Draw": 0.28,
            "Away Win": 0.36,
        }

    p_home = probs.get("Home Win", 0.36)
    p_draw = probs.get("Draw", 0.28)
    p_away = probs.get("Away Win", 0.36)

    total = base_home_xg + base_away_xg

    # Le nul élevé indique souvent un match plus serré/fermé.
    if p_draw >= 0.30:
        total *= 0.93
    elif p_draw <= 0.23:
        total *= 1.06

    dominance = p_home - p_away

    if abs(dominance) < 0.08:
        # Match équilibré
        target_home_share = 0.50 + dominance * 0.35
    else:
        # Favori, mais pas explosion automatique
        target_home_share = 0.50 + dominance * 0.42

    target_home_share = clamp(target_home_share, 0.35, 0.68)

    new_home = total * target_home_share
    new_away = total * (1 - target_home_share)

    return (
        round(clamp(new_home, 0.25, 2.80), 3),
        round(clamp(new_away, 0.20, 2.50), 3),
    )

# ============================================================
# FOOTBALL ENGINE V2 HELPERS
# ============================================================

def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _match_result_points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _extract_team_matches(history, team, limit=8):
    """
    Récupère les derniers matchs d'une équipe avec un maximum de compatibilité
    selon les colonnes disponibles dans tes CSV football.
    """
    if history is None or history.empty or not team:
        return pd.DataFrame()

    home_col = _find_col(history, ["HomeTeam", "home_team", "home", "Home"])
    away_col = _find_col(history, ["AwayTeam", "away_team", "away", "Away"])
    hg_col = _find_col(history, ["FTHG", "home_goals", "HomeGoals", "HG"])
    ag_col = _find_col(history, ["FTAG", "away_goals", "AwayGoals", "AG"])
    date_col = _find_col(history, ["Date", "date", "match_date"])

    if not home_col or not away_col or not hg_col or not ag_col:
        return pd.DataFrame()

    h = history.copy()

    if date_col:
        h[date_col] = pd.to_datetime(h[date_col], errors="coerce", dayfirst=True)
        h = h.sort_values(date_col)

    team_clean = normalize_name(team)

    mask = (
        h[home_col].astype(str).str.lower().str.strip().eq(team_clean)
        | h[away_col].astype(str).str.lower().str.strip().eq(team_clean)
    )

    # Fallback plus souple si les noms exacts ne matchent pas
    if mask.sum() == 0:
        mask = (
            h[home_col].astype(str).str.lower().str.contains(team_clean, na=False)
            | h[away_col].astype(str).str.lower().str.contains(team_clean, na=False)
        )

    matches = h[mask].tail(limit).copy()

    rows = []

    for _, r in matches.iterrows():
        is_home = normalize_name(r.get(home_col)) == team_clean

        try:
            home_goals = float(r.get(hg_col))
            away_goals = float(r.get(ag_col))
        except Exception:
            continue

        gf = home_goals if is_home else away_goals
        ga = away_goals if is_home else home_goals

        rows.append({
            "gf": gf,
            "ga": ga,
            "points": _match_result_points(gf, ga),
            "is_home": is_home,
        })

    return pd.DataFrame(rows)


def football_recent_profile(team, history, limit=8):
    matches = _extract_team_matches(history, team, limit=limit)

    if matches.empty:
        return {
            "form_points": 0.50,
            "attack_recent": 1.20,
            "defense_recent": 1.20,
            "home_boost": 0.00,
            "data_quality": 0.25,
            "matches_used": 0,
        }

    n = len(matches)
    weights = list(range(1, n + 1))
    weight_sum = sum(weights)

    form_points = sum(matches["points"].iloc[i] * weights[i] for i in range(n)) / (3 * weight_sum)
    attack_recent = sum(matches["gf"].iloc[i] * weights[i] for i in range(n)) / weight_sum
    defense_recent = sum(matches["ga"].iloc[i] * weights[i] for i in range(n)) / weight_sum

    home_matches = matches[matches["is_home"] == True]
    away_matches = matches[matches["is_home"] == False]

    home_ppg = home_matches["points"].mean() / 3 if not home_matches.empty else form_points
    away_ppg = away_matches["points"].mean() / 3 if not away_matches.empty else form_points
    home_boost = clamp(home_ppg - away_ppg, -0.18, 0.18)

    return {
        "form_points": round(float(form_points), 4),
        "attack_recent": round(float(attack_recent), 4),
        "defense_recent": round(float(defense_recent), 4),
        "home_boost": round(float(home_boost), 4),
        "data_quality": round(min(n / limit, 1), 4),
        "matches_used": int(n),
    }


def adjust_football_probability(base_prob, market, home_profile, away_profile, elo_diff, market_prob=None):
    """
    Ajuste la probabilité football avec forme récente + dynamique attaque/défense.
    """
    prob = float(base_prob)

    home_form_edge = home_profile["form_points"] - away_profile["form_points"]
    attack_edge = home_profile["attack_recent"] - away_profile["defense_recent"]
    away_attack_edge = away_profile["attack_recent"] - home_profile["defense_recent"]

    data_quality = min(home_profile["data_quality"], away_profile["data_quality"])

    if market == "Home Win":
        boost = (
            home_form_edge * 0.10
            + attack_edge * 0.025
            + home_profile["home_boost"] * 0.08
            + clamp(float(elo_diff), -220, 220) / 4000
        ) * data_quality
        prob += boost

    elif market == "Away Win":
        boost = (
            -home_form_edge * 0.10
            + away_attack_edge * 0.025
            - home_profile["home_boost"] * 0.06
            + clamp(float(-elo_diff), -220, 220) / 4000
        ) * data_quality
        prob += boost

    elif market == "Draw":
        # Plus les équipes sont proches, plus le nul est plausible
        closeness = 1 - min(abs(float(elo_diff)) / 250, 1)
        prob += (closeness - 0.5) * 0.035 * data_quality

    # Ne pas trop s'écarter du marché quand il est disponible
    if market_prob is not None and market_prob > 0:
        prob = prob * 0.80 + float(market_prob) * 0.20

    return clamp(prob, 0.03, 0.92)


def football_trap_signal(prob, odds, market_prob, value, confidence):
    """
    Détecte les faux favoris / pièges bookmaker.
    """
    if odds <= 1:
        return "NO ODDS"

    if market_prob is None or market_prob <= 0:
        return "NO MARKET"

    edge = prob - market_prob

    if prob >= 0.58 and value >= 0.025 and edge >= 0.025:
        return "BOOKMAKER VALUE"

    if market_prob >= 0.62 and prob <= 0.52:
        return "FALSE FAVORITE ALERT"

    if edge <= -0.06:
        return "MARKET OVERPRICED"

    if confidence == "Faible" and value > 0.10:
        return "HIGH VARIANCE VALUE"

    return "OK"


def poisson_pmf(k, lam):
    lam = max(float(lam), 0.05)
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def true_poisson_exact_scores(home_xg, away_xg, home_prob=None, draw_prob=None, away_prob=None):
    """
    VRAI moteur score exact :
    - aucun score forcé
    - calcule chaque score 0-0 à 5-5 avec loi de Poisson
    - ajuste légèrement selon les probabilités 1N2 du modèle
    - retourne les scores réellement les plus probables
    """
    home_xg = clamp(safe_float(home_xg, 1.35), 0.20, 3.20)
    away_xg = clamp(safe_float(away_xg, 1.05), 0.20, 3.20)

    home_prob = safe_float(home_prob, 0.0)
    draw_prob = safe_float(draw_prob, 0.0)
    away_prob = safe_float(away_prob, 0.0)

    scores = []

    for h in range(0, 6):
        for a in range(0, 6):
            p = poisson_pmf(h, home_xg) * poisson_pmf(a, away_xg)

            # Petit ajustement de cohérence 1N2, sans forcer le résultat
            if h > a and home_prob > 0:
                p *= 0.85 + home_prob * 0.35
            elif h == a and draw_prob > 0:
                p *= 0.85 + draw_prob * 0.45
            elif a > h and away_prob > 0:
                p *= 0.85 + away_prob * 0.35

            # Les 5-5 et scores ultra hauts sont rarement le top exact
            if h + a >= 7:
                p *= 0.75

            scores.append((f"{h}-{a}", p))

    total = sum(p for _, p in scores)

    if total <= 0:
        return [("1-1", 0.10), ("1-0", 0.09), ("2-1", 0.08)]

    scores = [(s, p / total) for s, p in scores]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Évite les doublons absurdes et garde les 5 meilleurs
    return [(s, round(float(p), 4)) for s, p in scores[:5]]


def market_total_goal_adjustment(odds_home, odds_draw, odds_away, base_home_xg, base_away_xg):
    """
    Ajuste le total de buts selon les cotes 1N2 :
    - nul assez fort => match plus fermé
    - favori net => domination plus nette
    - mais sans inventer 3-0 partout
    """
    probs = normalized_market_probabilities(odds_home, odds_draw, odds_away)

    if not isinstance(probs, dict):
        probs = {
            "Home Win": 0.36,
            "Draw": 0.28,
            "Away Win": 0.36,
        }

    p_home = probs.get("Home Win", 0.36)
    p_draw = probs.get("Draw", 0.28)
    p_away = probs.get("Away Win", 0.36)

    total = base_home_xg + base_away_xg

    # Le nul élevé indique souvent un match plus serré/fermé.
    if p_draw >= 0.30:
        total *= 0.93
    elif p_draw <= 0.23:
        total *= 1.06

    dominance = p_home - p_away

    if abs(dominance) < 0.08:
        # Match équilibré
        target_home_share = 0.50 + dominance * 0.35
    else:
        # Favori, mais pas explosion automatique
        target_home_share = 0.50 + dominance * 0.42

    target_home_share = clamp(target_home_share, 0.35, 0.68)

    new_home = total * target_home_share
    new_away = total * (1 - target_home_share)

    return (
        round(clamp(new_home, 0.25, 2.80), 3),
        round(clamp(new_away, 0.20, 2.50), 3),
    )

# ============================================================
# FOOTBALL ENGINE
# ============================================================

def draw_hunter(probs, elo_diff):
    draw_prob = float(probs["p_draw"])
    score_names = [s[0] for s in probs["top_scores"][:3]]

    if draw_prob >= 0.30 and abs(float(elo_diff)) <= 80 and ("0-0" in score_names or "1-1" in score_names):
        return "🟢 DRAW HUNTER"

    if draw_prob >= 0.26 and abs(float(elo_diff)) <= 80:
        return "🟡 DRAW WATCH"

    return "⚪ NO DRAW"


def exact_score_alert(score_proba):
    if score_proba >= 13:
        return "🟢 SCORE FORT"

    if score_proba >= 9:
        return "🟡 SCORE INTÉRESSANT"

    return "⚪ SCORE FAIBLE"


def scorer_prediction(home_team, away_team, home_xg, away_xg, player_df):
    if player_df.empty:
        return "Aucune donnée joueur"

    team = away_team if away_xg > home_xg else home_team

    team_players = player_df[player_df["team"] == team].copy()

    if team_players.empty:
        return f"Aucun joueur trouvé ({team})"

    team_players["minutes"] = pd.to_numeric(
        team_players["minutes"],
        errors="coerce"
    ).fillna(0)

    team_players["goals"] = pd.to_numeric(
        team_players["goals"],
        errors="coerce"
    ).fillna(0)

    team_players["shots_on"] = pd.to_numeric(
        team_players["shots_on"],
        errors="coerce"
    ).fillna(0)

    team_players["scorer_score"] = pd.to_numeric(
        team_players["scorer_score"],
        errors="coerce"
    ).fillna(0)

    team_players = team_players[
        team_players["minutes"] >= 300
    ]

    if team_players.empty:
        return f"Aucun joueur avec assez de minutes ({team})"

    top_players = team_players.sort_values(
        "scorer_score",
        ascending=False
    ).head(3)

    total_score = top_players["scorer_score"].sum()

    results = []

    for _, p in top_players.iterrows():
        if total_score > 0:
            proba = round(
                max(
                    (p["scorer_score"] / total_score) * 100,
                    1
                ),
                1
            )
        else:
            proba = 1

        player_score = round(
            float(p["scorer_score"]),
            1
        )

        results.append(
            f"{p['player']} "
            f"⚽{int(p['goals'])} "
            f"🎯{int(p['shots_on'])} "
            f"⏱️{int(p['minutes'])}min "
            f"🔥{proba}% "
            f"(IA {player_score})"
        )

    return " | ".join(results)


def process_football_match(m, strengths, elo_ratings, ml_model, player_df, bankroll, last_update, history=None, learning=None):
    rows = []

    home = m.get("home_team")
    away = m.get("away_team")
    sport = str(m.get("sport", "")).lower()

    home_profile = football_recent_profile(home, history)
    away_profile = football_recent_profile(away, history)

    home_known = not strengths[
        strengths["team"] == home
    ].empty

    away_known = not strengths[
        strengths["team"] == away
    ].empty

    market_probs = normalized_market_probabilities(
        m.get("odds_home"),
        m.get("odds_draw"),
        m.get("odds_away")
    )

    if not isinstance(market_probs, dict):
        market_probs = {
            "Home Win": 0.36,
            "Draw": 0.28,
            "Away Win": 0.36,
        }

    market_led = (
        "world_cup" in sport
        or "international" in sport
        or not home_known
        or not away_known
    )

    elo = get_match_elo(
        home,
        away,
        elo_ratings
    )

    history_home_xg, history_away_xg = estimate_xg(
        home,
        away,
        strengths
    )

    odds_home_xg, odds_away_xg = odds_based_expected_goals(
        m.get("odds_home"),
        m.get("odds_draw"),
        m.get("odds_away"),
        home_profile,
        away_profile,
    )

    data_quality = min(
        home_profile["data_quality"],
        away_profile["data_quality"]
    )

    # Si l'historique est faible ou trop neutre, on évite les scores copiés/collés.
    home_xg = blend_xg(
        history_home_xg,
        odds_home_xg,
        data_quality
    )

    away_xg = blend_xg(
        history_away_xg,
        odds_away_xg,
        data_quality
    )

    # Ajustement propre du total de buts via cotes 1N2.
    # Ça évite les scores clonés type 3-0 partout.
    home_xg, away_xg = market_total_goal_adjustment(
        m.get("odds_home"),
        m.get("odds_draw"),
        m.get("odds_away"),
        home_xg,
        away_xg,
    )

    features = pd.DataFrame([{
        "home_goals": home_xg,
        "away_goals": away_xg,
        "goal_diff": home_xg - away_xg,
        "total_goals": home_xg + away_xg,
    }])

    if ml_model is not None:
        ml_home_prob = float(
            ml_model.predict_proba(features)[0][1]
        )
    else:
        ml_home_prob = 0.5

    probs = football_poisson_probs(
        home_xg,
        away_xg
    )

    poisson_scores = true_poisson_exact_scores(
        home_xg,
        away_xg,
        probs["p_home"],
        probs["p_draw"],
        probs["p_away"],
    )

    probs["top_scores"] = poisson_scores

    score_1 = poisson_scores[0][0]
    score_1_proba = round(float(poisson_scores[0][1]) * 100, 2)

    score_2 = poisson_scores[1][0]
    score_2_proba = round(float(poisson_scores[1][1]) * 100, 2)

    score_3 = poisson_scores[2][0]
    score_3_proba = round(float(poisson_scores[2][1]) * 100, 2)

    draw_probability = round(float(probs["p_draw"]) * 100, 2)

    draw_signal = draw_hunter(
        probs,
        elo["elo_diff"]
    )

    score_signal = exact_score_alert(
        score_1_proba
    )

    scorer_hint = scorer_prediction(
        home,
        away,
        home_xg,
        away_xg,
        player_df
    )

    markets = [
        ("Home Win", probs["p_home"] * 0.55 + ml_home_prob * 0.45, m.get("odds_home")),
        ("Draw", probs["p_draw"], m.get("odds_draw")),
        ("Away Win", probs["p_away"], m.get("odds_away")),
        ("Over 2.5", probs["over_25"], None),
        ("Under 2.5", probs["under_25"], None),
        ("BTTS Yes", probs["btts_yes"], None),
        ("BTTS No", probs["btts_no"], None),
    ]

    for market, prob, odds in markets:
        prob = float(prob)

        odds = 0 if odds is None or pd.isna(odds) else float(odds)

        if market == "Home Win":
            prob = calibrate_probability(
                prob,
                elo["elo_diff"]
            )

        elif market == "Away Win":
            prob = calibrate_probability(
                prob,
                -elo["elo_diff"]
            )

        else:
            prob = calibrate_probability(
                prob,
                0
            )

        market_prob = market_probs.get(market)

        if market_prob is not None:
            market_weight = 0.42 if market_led else 0.24
            prob = (
                prob * (1 - market_weight)
                + market_prob * market_weight
            )
            prob = clamp(prob, 0.03, 0.97)

        if market in ["Home Win", "Away Win", "Draw"]:
            prob = adjust_football_probability(
                prob,
                market,
                home_profile,
                away_profile,
                elo["elo_diff"],
                market_prob
            )

        value = safe_value(
            prob,
            odds
        )

        confidence = classify_confidence(
            prob
        )

        mode = bet_mode(
            prob,
            odds,
            value,
            confidence,
            m.get("sport")
        )

        learning_reason = "NO_LEARNING"

        if learning is not None:
            prob, learning_reason = apply_learning_calibration(
                prob,
                m.get("sport"),
                mode,
                learning
            )

            value = safe_value(prob, odds)
            confidence = classify_confidence(prob)
            mode = bet_mode(prob, odds, value, confidence, m.get("sport"))
            mode = learning_adjusted_mode(
                mode,
                prob,
                value,
                m.get("sport"),
                learning,
                learning_reason
            )

        recommended_modes = ["MEGA VALUE", "SAFE PICK", "VALUE BET"]

        decision = (
            "VALUE BET"
            if mode in recommended_modes
            else "NO BET"
        )

        reliable = reliable_filter(
            decision,
            confidence,
            odds,
            value,
            prob
        )

        safe_score = safety_score(
            prob,
            odds,
            value,
            confidence,
            mode,
            reliable
        )

        stake, stake_percent, kelly_fraction = bankroll_management(
            prob,
            odds,
            value,
            mode,
            bankroll
        )

        if decision == "VALUE BET" and stake <= 0:
            decision = "NO BET"

        trap_signal = football_trap_signal(
            prob,
            odds,
            market_prob,
            value,
            confidence
        )

        rows.append({
            "last_update": last_update,
            "bet_mode": mode,
            "date": m.get("commence_time"),
            "sport": m.get("sport"),
            "category": "football",
            "home_team": home,
            "away_team": away,
            "market": market,
            "ai_probability": round(prob, 4),
            "bookmaker_odds": round(odds, 2) if odds > 0 else "",
            "implied_probability": round(1 / odds, 4) if odds > 0 else "",
            "value": round(value, 4),
            "confidence": confidence,
            "ia_badge": mode_badge(mode, value, confidence, odds),
            "reliable_only": reliable,
            "safety_score": safe_score,
            "safety_level": safety_level(safe_score, mode, decision, prob, value),
            "football_trap_signal": trap_signal,
            "learning_adjustment": learning_reason,
            "home_recent_form": home_profile["form_points"],
            "away_recent_form": away_profile["form_points"],
            "home_recent_attack": home_profile["attack_recent"],
            "away_recent_attack": away_profile["attack_recent"],
            "home_recent_defense": home_profile["defense_recent"],
            "away_recent_defense": away_profile["defense_recent"],
            "football_data_quality": min(home_profile["data_quality"], away_profile["data_quality"]),
            "decision": decision,
            "bankroll": bankroll,
            "stake_percent": stake_percent,
            "kelly_fraction": kelly_fraction,
            "suggested_stake": stake if decision == "VALUE BET" else 0,

            "home_elo": elo["home_elo"],
            "away_elo": elo["away_elo"],
            "elo_diff": elo["elo_diff"],
            "home_xg": home_xg,
            "away_xg": away_xg,

            "draw_probability": draw_probability,
            "draw_hunter": draw_signal,

            "score_exact_1": score_1,
            "score_exact_1_proba": score_1_proba,
            "score_exact_2": score_2,
            "score_exact_2_proba": score_2_proba,
            "score_exact_3": score_3,
            "score_exact_3_proba": score_3_proba,
            "score_exact_alert": score_signal,

            "scorer_prediction": scorer_hint,

            "over_25": round(probs["over_25"], 4),
            "under_25": round(probs["under_25"], 4),
            "btts_yes": round(probs["btts_yes"], 4),
            "btts_no": round(probs["btts_no"], 4),

            "top_scores": format_top_scores(probs["top_scores"]),

            "tennis_engine_score": "",
            "tennis_form_home": "",
            "tennis_form_away": "",
            "tennis_edge": "",
        })

    return rows


# ============================================================
# TENNIS ENGINE
# ============================================================

def detect_tennis_columns(df):
    cols = list(df.columns)

    player1_candidates = [
        "player1",
        "player_1",
        "home_player",
        "home_team",
        "winner_name",
        "Winner",
        "winner",
    ]

    player2_candidates = [
        "player2",
        "player_2",
        "away_player",
        "away_team",
        "loser_name",
        "Loser",
        "loser",
    ]

    winner_candidates = [
        "winner",
        "Winner",
        "winner_name",
        "player_winner",
        "match_winner",
    ]

    date_candidates = [
        "date",
        "Date",
        "match_date",
        "tourney_date",
    ]

    player1_col = next(
        (c for c in player1_candidates if c in cols),
        None
    )

    player2_col = next(
        (c for c in player2_candidates if c in cols),
        None
    )

    winner_col = next(
        (c for c in winner_candidates if c in cols),
        None
    )

    date_col = next(
        (c for c in date_candidates if c in cols),
        None
    )

    return player1_col, player2_col, winner_col, date_col


def build_tennis_player_model(history):
    ratings = {}

    if history.empty:
        return ratings

    p1_col, p2_col, winner_col, date_col = detect_tennis_columns(history)

    if not p1_col or not p2_col:
        print("Colonnes tennis insuffisantes pour créer un modèle historique.")
        return ratings

    hist = history.copy()

    if date_col:
        try:
            hist[date_col] = pd.to_datetime(
                hist[date_col],
                errors="coerce"
            )

            hist = hist.sort_values(
                date_col
            )

        except Exception:
            pass

    for _, row in hist.iterrows():
        p1 = row.get(p1_col)
        p2 = row.get(p2_col)

        if pd.isna(p1) or pd.isna(p2):
            continue

        p1 = str(p1)
        p2 = str(p2)

        for p in [p1, p2]:
            key = normalize_name(p)

            if key not in ratings:
                ratings[key] = {
                    "name": p,
                    "matches": 0,
                    "wins": 0,
                    "losses": 0,
                    "elo": 1500.0,
                    "recent": [],
                }

        winner = None

        if winner_col:
            raw_winner = row.get(winner_col)

            if not pd.isna(raw_winner):
                winner = str(raw_winner)

        if winner:
            winner_norm = normalize_name(winner)

            if winner_norm == normalize_name(p1):
                win_player = p1
                lose_player = p2

            elif winner_norm == normalize_name(p2):
                win_player = p2
                lose_player = p1

            else:
                continue

        else:
            continue

        w_key = normalize_name(win_player)
        l_key = normalize_name(lose_player)

        w_elo = ratings[w_key]["elo"]
        l_elo = ratings[l_key]["elo"]

        expected_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
        expected_l = 1 - expected_w

        k = 24

        ratings[w_key]["elo"] = w_elo + k * (1 - expected_w)
        ratings[l_key]["elo"] = l_elo + k * (0 - expected_l)

        ratings[w_key]["matches"] += 1
        ratings[w_key]["wins"] += 1
        ratings[w_key]["recent"].append(1)

        ratings[l_key]["matches"] += 1
        ratings[l_key]["losses"] += 1
        ratings[l_key]["recent"].append(0)

        ratings[w_key]["recent"] = ratings[w_key]["recent"][-10:]
        ratings[l_key]["recent"] = ratings[l_key]["recent"][-10:]

    return ratings


def tennis_player_stats(player, ratings):
    key = normalize_name(player)

    if key not in ratings:
        return {
            "elo": 1500.0,
            "matches": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.5,
            "recent_form": 0.5,
        }

    r = ratings[key]

    matches = max(
        int(r.get("matches", 0)),
        0
    )

    wins = max(
        int(r.get("wins", 0)),
        0
    )

    recent = r.get("recent", [])

    win_rate = wins / matches if matches > 0 else 0.5
    recent_form = sum(recent) / len(recent) if recent else win_rate

    return {
        "elo": float(r.get("elo", 1500.0)),
        "matches": matches,
        "wins": wins,
        "losses": int(r.get("losses", 0)),
        "win_rate": win_rate,
        "recent_form": recent_form,
    }


def tennis_probability(player_a, player_b, odds_a, odds_b, ratings):
    stats_a = tennis_player_stats(
        player_a,
        ratings
    )

    stats_b = tennis_player_stats(
        player_b,
        ratings
    )

    elo_a = stats_a["elo"]
    elo_b = stats_b["elo"]

    elo_prob_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    form_edge = stats_a["recent_form"] - stats_b["recent_form"]
    winrate_edge = stats_a["win_rate"] - stats_b["win_rate"]

    implied_a = american_to_probability_from_decimal(
        odds_a
    )

    implied_b = american_to_probability_from_decimal(
        odds_b
    )

    if implied_a > 0 and implied_b > 0:
        total_imp = implied_a + implied_b

        market_prob_a = implied_a / total_imp

    else:
        market_prob_a = 0.5

    experience_a = min(stats_a["matches"] / 30, 1)
    experience_b = min(stats_b["matches"] / 30, 1)

    history_strength = max(
        experience_a,
        experience_b
    )

    model_prob_a = (
        0.45 * market_prob_a
        + 0.35 * elo_prob_a
        + 0.12 * clamp(0.5 + form_edge * 0.6, 0.05, 0.95)
        + 0.08 * clamp(0.5 + winrate_edge * 0.5, 0.05, 0.95)
    )

    if history_strength < 0.25:
        model_prob_a = (
            0.75 * market_prob_a
            + 0.25 * model_prob_a
        )

    model_prob_a = clamp(
        model_prob_a,
        0.05,
        0.95
    )

    return model_prob_a, {
        "elo_a": elo_a,
        "elo_b": elo_b,
        "elo_diff": elo_a - elo_b,
        "form_a": stats_a["recent_form"],
        "form_b": stats_b["recent_form"],
        "winrate_a": stats_a["win_rate"],
        "winrate_b": stats_b["win_rate"],
        "matches_a": stats_a["matches"],
        "matches_b": stats_b["matches"],
        "market_prob_a": market_prob_a,
        "history_strength": history_strength,
    }


def tennis_confidence(prob, odds, edge, history_strength):
    if history_strength < 0.25:
        if prob >= 0.60 and edge >= 0.025:
            return "Moyen"
        if prob >= 0.54:
            return "Faible"
        return "A éviter"

    if prob >= 0.68 and edge >= 0.050 and 1.25 <= odds <= 2.50:
        return "Elite"

    if prob >= 0.62 and edge >= 0.035 and 1.25 <= odds <= 2.80:
        return "Fort"

    if prob >= 0.56 and edge >= 0.020 and 1.20 <= odds <= 3.20:
        return "Moyen"

    if prob >= 0.52:
        return "Faible"

    return "A éviter"


def tennis_set_probabilities(match_prob, elo_diff, history_strength):
    """
    Estime 2-0 / 2-1 en conditionnel sur le joueur sélectionné.
    Plus l'écart ELO est grand, plus le 2-0 augmente.
    """
    dominance = clamp(abs(safe_float(elo_diff, 0)) / 260, 0, 1)
    confidence = clamp(history_strength, 0, 1)

    straight = 0.50 + dominance * 0.22 + (match_prob - 0.55) * 0.18 + confidence * 0.04
    straight = clamp(straight, 0.42, 0.78)

    three_sets = 1 - straight

    return round(straight * 100, 2), round(three_sets * 100, 2)


def tennis_badge(value, confidence, odds):
    if value >= 0.06 and confidence in ["Fort", "Elite"] and 1.25 <= odds <= 3.20:
        return "🟢 TENNIS STRONG VALUE"

    if value >= 0.015 and confidence in ["Moyen", "Fort", "Elite"] and 1.25 <= odds <= 3.60:
        return "🟡 TENNIS VALUE"

    if value >= 0.05:
        return "🔴 TENNIS RISKY VALUE"

    return "⚪ WATCH / NO VALUE"


def tennis_reliable_filter(decision, confidence, odds, value, prob):
    return (
        decision == "VALUE BET"
        and confidence in ["Moyen", "Fort", "Elite"]
        and 1.20 <= odds <= 3.20
        and value >= 0.030
        and prob >= 0.545
    )


def process_tennis_match(m, tennis_ratings, bankroll, last_update, learning=None):
    rows = []

    player_a = m.get("home_team")
    player_b = m.get("away_team")

    odds_a = safe_float(
        m.get("odds_home"),
        0
    )

    odds_b = safe_float(
        m.get("odds_away"),
        0
    )

    if not player_a or not player_b:
        return rows

    markets = [
        ("Player 1 Win", player_a, player_b, odds_a, odds_b),
        ("Player 2 Win", player_b, player_a, odds_b, odds_a),
    ]

    for market, player, opponent, odds, opponent_odds in markets:
        if odds <= 1:
            continue

        prob, meta = tennis_probability(
            player,
            opponent,
            odds,
            opponent_odds,
            tennis_ratings
        )

        implied = 1 / odds
        value = safe_value(
            prob,
            odds
        )

        edge = prob - implied

        confidence = tennis_confidence(
            prob,
            odds,
            edge,
            meta["history_strength"]
        )


        mode = bet_mode(
            prob,
            odds,
            value,
            confidence,
            m.get("sport")
        )

        learning_reason = "NO_LEARNING"

        if learning is not None:
            prob, learning_reason = apply_learning_calibration(
                prob,
                m.get("sport"),
                mode,
                learning
            )

            implied = 1 / odds
            value = safe_value(prob, odds)
            edge = prob - implied
            confidence = tennis_confidence(
                prob,
                odds,
                edge,
                meta["history_strength"]
            )
            mode = bet_mode(prob, odds, value, confidence, m.get("sport"))
            mode = learning_adjusted_mode(
                mode,
                prob,
                value,
                m.get("sport"),
                learning,
                learning_reason
            )

        recommended_modes = ["MEGA VALUE", "SAFE PICK", "VALUE BET"]

        decision = (
            "VALUE BET"
            if mode in recommended_modes
            else "NO BET"
        )

        stake, stake_percent, kelly_fraction = bankroll_management(
            prob,
            odds,
            value,
            mode,
            bankroll
        )

        if decision == "VALUE BET" and stake <= 0:
            decision = "NO BET"

        badge = mode_badge(mode, value, confidence, odds)

        # Ancien badge tennis conservé seulement si besoin de debug
        _tennis_raw_badge = tennis_badge(
            value,
            confidence,
            odds
        )

        reliable = tennis_reliable_filter(
            decision,
            confidence,
            odds,
            value,
            prob
        )

        safe_score = safety_score(
            prob,
            odds,
            value,
            confidence,
            mode,
            reliable
        )

        tennis_engine_score = round(
            (
                prob * 60
                + max(edge, 0) * 200
                + meta["history_strength"] * 20
            ),
            2
        )

        set_20_proba, set_21_proba = tennis_set_probabilities(
            prob,
            meta["elo_diff"],
            meta["history_strength"]
        )

        if set_20_proba >= set_21_proba:
            set_score_1 = "2-0"
            set_score_1_proba = set_20_proba
            set_score_2 = "2-1"
            set_score_2_proba = set_21_proba
        else:
            set_score_1 = "2-1"
            set_score_1_proba = set_21_proba
            set_score_2 = "2-0"
            set_score_2_proba = set_20_proba

        rows.append({
            "last_update": last_update,
            "bet_mode": mode,
            "date": m.get("commence_time"),
            "sport": m.get("sport"),
            "category": "tennis",
            "home_team": player_a,
            "away_team": player_b,
            "market": market,
            "selection": player,

            "ai_probability": round(prob, 4),
            "bookmaker_odds": round(odds, 2),
            "implied_probability": round(implied, 4),
            "value": round(value, 4),
            "confidence": confidence,
            "ia_badge": badge,
            "reliable_only": reliable,
            "safety_score": safe_score,
            "safety_level": safety_level(safe_score, mode, decision, prob, value),
            "football_trap_signal": "",
            "learning_adjustment": learning_reason,
            "home_recent_form": "",
            "away_recent_form": "",
            "home_recent_attack": "",
            "away_recent_attack": "",
            "home_recent_defense": "",
            "away_recent_defense": "",
            "football_data_quality": "",
            "decision": decision,
            "bankroll": bankroll,
            "stake_percent": stake_percent,
            "kelly_fraction": kelly_fraction,
            "suggested_stake": stake if decision == "VALUE BET" else 0,

            "home_elo": round(meta["elo_a"], 2),
            "away_elo": round(meta["elo_b"], 2),
            "elo_diff": round(meta["elo_diff"], 2),

            "home_xg": "",
            "away_xg": "",

            "draw_probability": "",
            "draw_hunter": "🎾 NO DRAW SPORT",

            "score_exact_1": set_score_1,
            "score_exact_1_proba": set_score_1_proba,
            "score_exact_2": set_score_2,
            "score_exact_2_proba": set_score_2_proba,
            "score_exact_3": "",
            "score_exact_3_proba": "",
            "score_exact_alert": "🎾 SET SCORE ESTIMATION",

            "scorer_prediction": "🎾 Tennis : aucun buteur",

            "over_25": "",
            "under_25": "",
            "btts_yes": "",
            "btts_no": "",

            "top_scores": f"{set_score_1} {set_score_1_proba}% | {set_score_2} {set_score_2_proba}%",

            "tennis_engine_score": tennis_engine_score,
            "tennis_form_home": round(meta["form_a"], 3),
            "tennis_form_away": round(meta["form_b"], 3),
            "tennis_edge": round(edge, 4),
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    hist_path = Path("data/processed/football_history_all.csv")

    if hist_path.exists():
        history = pd.read_csv(
            hist_path
        )

        strengths = build_team_strength(
            history
        )

        elo_ratings = build_elo_ratings(
            history
        )

    else:
        strengths = pd.DataFrame(
            columns=["team", "attack", "defense", "form"]
        )

        elo_ratings = {}

    tennis_history = load_tennis_history()

    tennis_ratings = build_tennis_player_model(
        tennis_history
    )

    player_df = load_player_scorers()
    upcoming = load_or_demo_upcoming()

    learning = load_tracking_learning()
    learning_summary_to_csv(learning)

    print(
        "IA learning:",
        "bets=",
        learning.get("global_bets", 0),
        "winrate=",
        learning.get("global_winrate", 0),
        "roi=",
        learning.get("global_roi", 0),
    )

    rows = []

    bankroll = BANKROLL_START

    ml_model = load_xgboost_model()

    last_update = datetime.now().strftime(
        "%d/%m/%Y %H:%M"
    )

    for _, m in upcoming.iterrows():
        sport = m.get("sport")

        if is_tennis_sport(sport):
            match_rows = process_tennis_match(
                m,
                tennis_ratings,
                bankroll,
                last_update,
                learning
            )

        elif is_football_sport(sport):
            match_rows = process_football_match(
                m,
                strengths,
                elo_ratings,
                ml_model,
                player_df,
                bankroll,
                last_update,
                history,
                learning
            )

        else:
            match_rows = []

        rows.extend(match_rows)

    out = pd.DataFrame(
        rows
    )

    if not out.empty:
        out["priority"] = out.apply(
            lambda r: priority_score(
                r.get("home_team", ""),
                r.get("away_team", ""),
                r.get("sport", "")
            ),
            axis=1,
        )

        if "date" in out.columns:
            out["_dt"] = out["date"].apply(parse_match_datetime)
            now = datetime.now(timezone.utc)
            out["_max_hours"] = out["sport"].apply(dynamic_hours_window)
            out["_end"] = out["_max_hours"].apply(lambda h: now + timedelta(hours=int(h)))

            out = out[
                out["_dt"].isna()
                | (
                    (out["_dt"] >= now - timedelta(hours=PAST_GRACE_HOURS))
                    & (out["_dt"] <= out["_end"])
                )
            ].copy()
            out = out.drop(columns=["_dt", "_max_hours", "_end"], errors="ignore")

    if out.empty:
        print("Aucune prédiction.")
        return

    out["bookmaker_odds_num"] = pd.to_numeric(
        out["bookmaker_odds"],
        errors="coerce"
    )

    out["ai_probability_num"] = pd.to_numeric(
        out["ai_probability"],
        errors="coerce"
    ).fillna(0)

    out_display = out[
        out["bookmaker_odds_num"] > 1
    ].copy()

    decision_rank = {
        "MEGA VALUE": 0,
        "SAFE PICK": 1,
        "VALUE BET": 2,
        "RISKY VALUE": 3,
        "WATCHLIST": 4,
        "NO BET": 5,
    }

    out_display["decision_rank"] = out_display["bet_mode"].map(decision_rank).fillna(9)

    if "priority" not in out_display.columns:
        out_display["priority"] = 0

    out_display = out_display.sort_values(
        ["priority", "decision_rank", "safety_score", "value", "ai_probability_num"],
        ascending=[False, True, False, False, False]
    )

    out_display = out_display.drop(
        columns=[
            "bookmaker_odds_num",
            "ai_probability_num",
            "decision_rank"
        ]
    )

    Path("data/predictions").mkdir(
        parents=True,
        exist_ok=True
    )

    out_display.to_csv(
        "data/predictions/predictions_today.csv",
        index=False
    )

    recommended_output = out_display[
        (out_display["decision"] == "VALUE BET")
        & (pd.to_numeric(out_display["suggested_stake"], errors="coerce").fillna(0) > 0)
    ].copy()

    recommended_output.to_csv(
        "data/predictions/value_bets_today.csv",
        index=False
    )

    print(
        "Prédictions générées :",
        len(out_display)
    )

    print(
        "Value bets retenus :",
        len(recommended_output)
    )

    if "category" in out_display.columns:
        print("Répartition recommandations :")
        reco_stats = (
            out_display[out_display["decision"] == "VALUE BET"]
            .groupby(["category", "bet_mode"])
            .size()
        )
        print(reco_stats.to_string() if len(reco_stats) else "Aucune recommandation avec mise positive.")

    preview = (
        out_display
        .sort_values(["safety_score", "value"], ascending=[False, False])
        .head(30)
        .to_string(index=False)
    )

    print(preview.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
