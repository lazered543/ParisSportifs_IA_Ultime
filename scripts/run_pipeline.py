from pathlib import Path
import pandas as pd

from src.features.football_features import build_team_strength, estimate_xg
from src.models.poisson import football_poisson_probs
from src.models.elo import build_elo_ratings, get_match_elo
from src.models.calibration import calibrate_probability
from src.models.machine_learning import load_model, predict_match

from src.betting.value_bet import value_score, classify_confidence
from src.betting.bankroll import safe_stake

from src.utils.config import (
    BANKROLL_START,
    MAX_STAKE_PCT,
    MIN_VALUE,
    MIN_CONFIDENCE
)


def load_or_demo_upcoming():

    p = Path("data/processed/upcoming_odds.csv")

    if p.exists():

        df = pd.read_csv(p)

        if not df.empty:
            return df

    return pd.DataFrame([])


def main():

    hist_path = Path(
        "data/processed/football_history_all.csv"
    )

    if hist_path.exists():

        history = pd.read_csv(hist_path)

        strengths = build_team_strength(history)

        elo_ratings = build_elo_ratings(history)

    else:

        strengths = pd.DataFrame(
            columns=[
                "team",
                "attack",
                "defense",
                "form"
            ]
        )

        elo_ratings = {}

    upcoming = load_or_demo_upcoming()

    rows = []

    bankroll = BANKROLL_START

    ml_model = load_model()

    for _, m in upcoming.iterrows():

        home = m.get("home_team")

        away = m.get("away_team")

        elo = get_match_elo(
            home,
            away,
            elo_ratings
        )

        hxg, axg = estimate_xg(
            home,
            away,
            strengths
        )

        ml_home_prob = predict_match(
            ml_model,
            elo["home_elo"],
            elo["away_elo"],
            elo["elo_diff"],
            hxg,
            axg
        )

        probs = football_poisson_probs(
            hxg,
            axg
        )

        markets = [

            (
                "Home Win",
                probs["p_home"] * 0.55 + ml_home_prob * 0.45,
                m.get("odds_home")
            ),

            (
                "Draw",
                probs["p_draw"],
                m.get("odds_draw")
            ),

            (
                "Away Win",
                probs["p_away"],
                m.get("odds_away")
            ),

            (
                "Over 1.5",
                probs["over_15"],
                None
            ),

            (
                "Over 2.5",
                probs["over_25"],
                None
            ),

            (
                "Under 2.5",
                probs["under_25"],
                None
            ),

            (
                "BTTS Yes",
                probs["btts_yes"],
                None
            ),

            (
                "BTTS No",
                probs["btts_no"],
                None
            ),

            (
                "Score Exact 1",
                probs["top_scores"][0][1],
                None
            ),
        ]

        for market, prob, odds in markets:

            prob = float(prob)

            if odds is None or pd.isna(odds):

                odds = 0

            odds = float(odds) if odds else 0

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

            val = (
                value_score(prob, odds)
                if odds > 0
                else 0
            )

            confidence = classify_confidence(
                prob
            )

            stake = (
                safe_stake(
                    bankroll,
                    prob,
                    odds,
                    MAX_STAKE_PCT
                )
                if odds > 0
                else 0
            )

            decision = (

                "VALUE BET"

                if (

                    val
                    and val >= MIN_VALUE
                    and prob >= MIN_CONFIDENCE
                    and 1.40 <= odds <= 3.50
                    and prob >= 0.45
                    and confidence != "A éviter"
                )

                else "NO BET"
            )

            rows.append({

                "home_elo": elo["home_elo"],

                "away_elo": elo["away_elo"],

                "elo_diff": elo["elo_diff"],

                "date": m.get("commence_time"),

                "sport": m.get("sport"),

                "home_team": home,

                "away_team": away,

                "market": market,

                "ai_probability": round(prob, 4),

                "bookmaker_odds": (
                    round(odds, 2)
                    if odds > 0
                    else ""
                ),

                "implied_probability": (
                    round(1 / odds, 4)
                    if odds > 0
                    else ""
                ),

                "value": round(float(val), 4),

                "confidence": confidence,

                "suggested_stake": (
                    stake
                    if decision == "VALUE BET"
                    else 0
                ),

                "home_xg": hxg,

                "away_xg": axg,

                "over_15": round(
                    probs["over_15"],
                    4
                ),

                "over_25": round(
                    probs["over_25"],
                    4
                ),

                "under_25": round(
                    probs["under_25"],
                    4
                ),

                "btts_yes": round(
                    probs["btts_yes"],
                    4
                ),

                "btts_no": round(
                    probs["btts_no"],
                    4
                ),

                "score_exact_1": (
                    probs["top_scores"][0][0]
                ),

                "score_exact_1_proba": round(
                    probs["top_scores"][0][1],
                    4
                ),

                "score_exact_2": (
                    probs["top_scores"][1][0]
                ),

                "score_exact_2_proba": round(
                    probs["top_scores"][1][1],
                    4
                ),

                "score_exact_3": (
                    probs["top_scores"][2][0]
                ),

                "score_exact_3_proba": round(
                    probs["top_scores"][2][1],
                    4
                ),

                "top_scores": str(
                    probs["top_scores"]
                ),

                "decision": decision,
            })

    out = pd.DataFrame(rows)

    if out.empty:

        print(
            "Aucune prédiction générée."
        )

        return

    out["bookmaker_odds_num"] = pd.to_numeric(
        out["bookmaker_odds"],
        errors="coerce"
    )

    out_filtered = out[
        (
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
        )
    ].copy()

    out_filtered = out_filtered.drop(
        columns=["bookmaker_odds_num"]
    )

    Path(
        "data/predictions"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    out_filtered.to_csv(
        "data/predictions/predictions_today.csv",
        index=False
    )

    out_filtered[
        out_filtered["decision"] == "VALUE BET"
    ].to_csv(
        "data/predictions/value_bets_today.csv",
        index=False
    )

    print(
        "Prédictions générées :",
        len(out_filtered)
    )

    if not out_filtered.empty:

        print(
            out_filtered
            .sort_values(
                "value",
                ascending=False
            )
            .head(20)
            .to_string(index=False)
        )

    else:

        print(
            "Aucune prédiction après filtrage."
        )


if __name__ == "__main__":

    main()