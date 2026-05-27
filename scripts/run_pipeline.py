from pathlib import Path
from datetime import datetime
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
        return {}

    return {
        market: prob / total
        for market, prob in implied.items()
        if prob > 0
    }


def odds_based_expected_goals(odds_home, odds_draw, odds_away, home_profile=None, away_profile=None):
    """
    Estime des xG réalistes quand l'historique équipe est faible ou absent.
    Logique :
    - les cotes donnent la force relative,
    - le nul donne une indication sur le total de buts,
    - la forme récente ajuste légèrement attaque/défense.
    """
    market_probs = normalized_market_probabilities(odds_home, odds_draw, odds_away)

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

# ============================================================
# BET MODES
# ============================================================

def bet_mode(prob, odds, value, confidence, sport):
    """
    Recalibrage réaliste :
    - SAFE PICK : probabilité forte, cote correcte, value légèrement positive ou neutre
    - VALUE BET : edge mathématique positif
    - RISKY VALUE : edge fort mais proba/cote plus volatile
    - WATCHLIST : intéressant mais pas assez rentable pour miser
    """
    sport = str(sport).lower()
    confidence = str(confidence)

    if odds <= 1 or odds > 5.00:
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

    # Très gros edge avec proba solide
    if prob >= 0.60 and value >= 0.055 and 1.35 <= odds <= 3.40:
        return "MEGA VALUE"

    # Favori probable avec cote acceptable
    if prob >= 0.66 and value >= -0.01 and 1.18 <= odds <= 2.05:
        return "SAFE PICK"

    # Pari value principal : l'IA estime la cote mal payée par le bookmaker
    if prob >= 0.515 and value >= 0.010 and 1.28 <= odds <= 3.80:
        return "VALUE BET"

    # Value risquée : on accepte un peu plus de variance
    if prob >= 0.46 and value >= 0.040 and 1.75 <= odds <= 4.60:
        return "RISKY VALUE"

    # Pas pari conseillé, mais à surveiller dans le dashboard
    if prob >= 0.55 and 1.18 <= odds <= 3.80:
        return "WATCHLIST"

    return "NO BET"

# ============================================================
# BANKROLL MANAGEMENT IA
# ============================================================

def bankroll_management(prob, odds, value, mode, bankroll):
    """
    Gestion intelligente de bankroll :
    - Kelly Criterion fractionné
    - plafonds de sécurité par mode
    - mise minimale seulement si le pari est accepté
    """

    if odds <= 1 or mode in ["NO BET", "WATCHLIST"]:
        return 0.0, 0.0, 0.0

    b = odds - 1
    p = prob
    q = 1 - p

    # Kelly brut
    kelly = ((b * p) - q) / b

    # Si Kelly négatif, aucun edge réel
    if kelly <= 0:
        return 0.0, 0.0, 0.0

    # Plafond Kelly sécurité
    kelly = max(0.0, min(kelly, 0.12))

    # Fractionnement selon le profil du pari
    fractions = {
        "SAFE PICK": 0.20,
        "VALUE BET": 0.14,
        "RISKY VALUE": 0.06,
        "MEGA VALUE": 0.32,
        "WATCHLIST": 0.0,
    }

    fraction = fractions.get(mode, 0.0)

    stake_percent = kelly * fraction

    # Plafonds par mode pour éviter les grosses mises dangereuses
    max_by_mode = {
        "SAFE PICK": 0.025,
        "VALUE BET": 0.018,
        "RISKY VALUE": 0.008,
        "MEGA VALUE": 0.04,
        "WATCHLIST": 0.0,
    }

    max_percent = max_by_mode.get(mode, 0.0)

    stake_percent = max(0.0, min(stake_percent, max_percent))

    # Mise trop faible => on évite seulement les ultra micro-paris
    if stake_percent < 0.001:
        return 0.0, round(stake_percent, 4), round(kelly, 4)

    stake = bankroll * stake_percent

    return round(stake, 2), round(stake_percent, 4), round(kelly, 4)

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
        and confidence in ["Faible", "Moyen", "Fort", "Elite"]
        and 1.20 <= odds <= 3.80
        and value >= 0.010
        and prob >= 0.50
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


def process_football_match(m, strengths, elo_ratings, ml_model, player_df, bankroll, last_update, history=None):
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

    score_1 = probs["top_scores"][0][0]
    score_1_proba = round(float(probs["top_scores"][0][1]) * 100, 2)

    score_2 = probs["top_scores"][1][0]
    score_2_proba = round(float(probs["top_scores"][1][1]) * 100, 2)

    score_3 = probs["top_scores"][2][0]
    score_3_proba = round(float(probs["top_scores"][2][1]) * 100, 2)

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

        recommended_modes = ["MEGA VALUE", "SAFE PICK", "VALUE BET", "RISKY VALUE"]

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
        if prob >= 0.58 and edge >= 0.010:
            return "Moyen"
        if prob >= 0.52:
            return "Faible"
        return "A éviter"

    if prob >= 0.68 and edge >= 0.040 and 1.25 <= odds <= 2.60:
        return "Elite"

    if prob >= 0.60 and edge >= 0.020 and 1.25 <= odds <= 3.00:
        return "Fort"

    if prob >= 0.54 and edge >= 0.005 and 1.20 <= odds <= 3.50:
        return "Moyen"

    if prob >= 0.50:
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
        and confidence in ["Faible", "Moyen", "Fort", "Elite"]
        and 1.20 <= odds <= 3.80
        and value >= 0.010
        and prob >= 0.50
    )


def process_tennis_match(m, tennis_ratings, bankroll, last_update):
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

        recommended_modes = ["MEGA VALUE", "SAFE PICK", "VALUE BET", "RISKY VALUE"]

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
                last_update
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
                history
            )

        else:
            match_rows = []

        rows.extend(match_rows)

    out = pd.DataFrame(
        rows
    )

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

    out_display = out_display.sort_values(
        ["decision_rank", "safety_score", "value", "ai_probability_num"],
        ascending=[True, False, False, False]
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
