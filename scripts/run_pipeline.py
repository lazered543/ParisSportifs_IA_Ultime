from pathlib import Path
from datetime import datetime
import pandas as pd

from src.features.football_features import build_team_strength, estimate_xg
from src.models.poisson import football_poisson_probs
from src.models.elo import build_elo_ratings, get_match_elo
from src.models.calibration import calibrate_probability
from src.models.xgboost_model import load_xgboost_model
from src.betting.value_bet import value_score, classify_confidence
from src.betting.bankroll import safe_stake
from src.utils.config import BANKROLL_START, MAX_STAKE_PCT, MIN_VALUE, MIN_CONFIDENCE


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


def ia_badge(value, confidence, odds):
    if value >= 0.10 and confidence in ["Moyen", "Fort", "Elite"] and 1.40 <= odds <= 2.80:
        return "🟢 STRONG VALUE"
    if value >= 0.05 and confidence in ["Moyen", "Fort", "Elite"] and 1.40 <= odds <= 3.50:
        return "🟡 MEDIUM VALUE"
    if value > 0:
        return "🔴 RISKY VALUE"
    return "⚪ NO VALUE"


def draw_hunter(probs, elo_diff):
    draw_prob = float(probs["p_draw"])
    score_names = [s[0] for s in probs["top_scores"][:3]]

    if draw_prob >= 0.30 and abs(float(elo_diff)) <= 80 and ("0-0" in score_names or "1-1" in score_names):
        return "🟢 DRAW HUNTER"
    if draw_prob >= 0.26 and abs(float(elo_diff)) <= 80:
        return "🟡 DRAW WATCH"
    return "⚪ NO DRAW"


def exact_score_alert(score_proba):
    if score_proba >= 10:
        return "🟢 SCORE FORT"
    if score_proba >= 8:
        return "🟡 SCORE INTÉRESSANT"
    return "⚪ SCORE FAIBLE"


def scorer_prediction(home_team, away_team, home_xg, away_xg, player_df):
    if player_df.empty:
        return "Aucune donnée joueur"

    team = away_team if away_xg > home_xg else home_team

    team_players = player_df[player_df["team"] == team].copy()

    if team_players.empty:
        return f"Aucun joueur trouvé ({team})"

    team_players["minutes"] = pd.to_numeric(team_players["minutes"], errors="coerce").fillna(0)
    team_players["goals"] = pd.to_numeric(team_players["goals"], errors="coerce").fillna(0)
    team_players["shots_on"] = pd.to_numeric(team_players["shots_on"], errors="coerce").fillna(0)
    team_players["scorer_score"] = pd.to_numeric(team_players["scorer_score"], errors="coerce").fillna(0)

    team_players = team_players[team_players["minutes"] >= 300]

    if team_players.empty:
        return f"Aucun joueur avec assez de minutes ({team})"

    top_players = team_players.sort_values("scorer_score", ascending=False).head(3)

    total_score = top_players["scorer_score"].sum()
    results = []

    for _, p in top_players.iterrows():
        if total_score > 0:
            proba = round(max((p["scorer_score"] / total_score) * 100, 1), 1)
        else:
            proba = 1

        player_score = round(float(p["scorer_score"]), 1)

        results.append(
            f"{p['player']} ⚽{int(p['goals'])} "
            f"🎯{int(p['shots_on'])} "
            f"⏱️{int(p['minutes'])}min "
            f"🔥{proba}% "
            f"(IA {player_score})"
        )

    return " | ".join(results)


def reliable_filter(decision, confidence, odds, value, prob):
    return (
        decision == "VALUE BET"
        and confidence in ["Moyen", "Fort", "Elite"]
        and 1.40 <= odds <= 2.80
        and value >= 0.05
        and prob >= 0.52
    )


def safe_value(prob, odds):
    if odds <= 1:
        return 0
    value = value_score(prob, odds)
    return min(value, 1.5)


def main():
    hist_path = Path("data/processed/football_history_all.csv")

    if hist_path.exists():
        history = pd.read_csv(hist_path)
        strengths = build_team_strength(history)
        elo_ratings = build_elo_ratings(history)
    else:
        strengths = pd.DataFrame(columns=["team", "attack", "defense", "form"])
        elo_ratings = {}

    player_df = load_player_scorers()
    upcoming = load_or_demo_upcoming()

    rows = []
    bankroll = BANKROLL_START
    ml_model = load_xgboost_model()
    last_update = datetime.now().strftime("%d/%m/%Y %H:%M")

    for _, m in upcoming.iterrows():
        home = m.get("home_team")
        away = m.get("away_team")

        elo = get_match_elo(home, away, elo_ratings)
        home_xg, away_xg = estimate_xg(home, away, strengths)

        features = pd.DataFrame([{
            "home_goals": home_xg,
            "away_goals": away_xg,
            "goal_diff": home_xg - away_xg,
            "total_goals": home_xg + away_xg,
        }])

        ml_home_prob = float(ml_model.predict_proba(features)[0][1]) if ml_model is not None else 0.5

        probs = football_poisson_probs(home_xg, away_xg)

        score_1 = probs["top_scores"][0][0]
        score_1_proba = round(float(probs["top_scores"][0][1]) * 100, 2)
        score_2 = probs["top_scores"][1][0]
        score_2_proba = round(float(probs["top_scores"][1][1]) * 100, 2)
        score_3 = probs["top_scores"][2][0]
        score_3_proba = round(float(probs["top_scores"][2][1]) * 100, 2)

        draw_probability = round(float(probs["p_draw"]) * 100, 2)
        draw_signal = draw_hunter(probs, elo["elo_diff"])
        score_signal = exact_score_alert(score_1_proba)
        scorer_hint = scorer_prediction(home, away, home_xg, away_xg, player_df)

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
                prob = calibrate_probability(prob, elo["elo_diff"])
            elif market == "Away Win":
                prob = calibrate_probability(prob, -elo["elo_diff"])
            else:
                prob = calibrate_probability(prob, 0)

            value = safe_value(prob, odds)
            confidence = classify_confidence(prob)

            stake = safe_stake(bankroll, prob, odds, MAX_STAKE_PCT) if odds > 0 else 0

            decision = (
                "VALUE BET"
                if (
                    value >= MIN_VALUE
                    and prob >= MIN_CONFIDENCE
                    and 1.40 <= odds <= 3.50
                    and confidence != "A éviter"
                )
                else "NO BET"
            )

            rows.append({
                "last_update": last_update,
                "date": m.get("commence_time"),
                "sport": m.get("sport"),
                "home_team": home,
                "away_team": away,
                "market": market,
                "ai_probability": round(prob, 4),
                "bookmaker_odds": round(odds, 2) if odds > 0 else "",
                "implied_probability": round(1 / odds, 4) if odds > 0 else "",
                "value": round(value, 4),
                "confidence": confidence,
                "ia_badge": ia_badge(value, confidence, odds),
                "reliable_only": reliable_filter(decision, confidence, odds, value, prob),
                "decision": decision,
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
                "top_scores": str(probs["top_scores"]),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        print("Aucune prédiction.")
        return

    out["bookmaker_odds_num"] = pd.to_numeric(out["bookmaker_odds"], errors="coerce")

    out_filtered = out[
        (
            out["bookmaker_odds_num"].isna()
            & (out["ai_probability"] >= 0.45)
        )
        |
        (
            out["bookmaker_odds_num"].notna()
            & (out["bookmaker_odds_num"] >= 1.40)
            & (out["bookmaker_odds_num"] <= 3.50)
            & (out["ai_probability"] >= 0.45)
        )
    ].copy()

    out_filtered = out_filtered.drop(columns=["bookmaker_odds_num"])

    Path("data/predictions").mkdir(parents=True, exist_ok=True)

    out_filtered.to_csv("data/predictions/predictions_today.csv", index=False)

    out_filtered[out_filtered["decision"] == "VALUE BET"].to_csv(
        "data/predictions/value_bets_today.csv",
        index=False
    )

    print("Prédictions générées :", len(out_filtered))
    print(out_filtered.sort_values("value", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()