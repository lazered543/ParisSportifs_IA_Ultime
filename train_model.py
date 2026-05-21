import pandas as pd

from src.models.machine_learning import (
    train_model
)

from src.models.elo import (
    build_elo_ratings,
    get_match_elo
)

from src.features.football_features import (
    build_team_strength,
    estimate_xg
)


history = pd.read_csv(
    "data/processed/football_history_all.csv"
)

elo_ratings = build_elo_ratings(history)

strengths = build_team_strength(history)

rows = []

for _, row in history.iterrows():

    home = row["HomeTeam"]
    away = row["AwayTeam"]

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

    rows.append({

        "home_elo": elo["home_elo"],
        "away_elo": elo["away_elo"],
        "elo_diff": elo["elo_diff"],
        "home_xg": hxg,
        "away_xg": axg,

        "FTHG": row["FTHG"],
        "FTAG": row["FTAG"]
    })

dataset = pd.DataFrame(rows)

train_model(dataset)