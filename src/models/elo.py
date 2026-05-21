import pandas as pd

BASE_ELO = 1500
K = 20
HOME_ADVANTAGE = 65


def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating, expected, actual):
    return rating + K * (actual - expected)


def build_elo_ratings(history_df):

    ratings = {}

    for _, row in history_df.iterrows():

        home = row.get("HomeTeam")
        away = row.get("AwayTeam")

        if pd.isna(home) or pd.isna(away):
            continue

        home_rating = ratings.get(home, BASE_ELO)
        away_rating = ratings.get(away, BASE_ELO)

        home_expected = expected_score(
            home_rating + HOME_ADVANTAGE,
            away_rating
        )

        away_expected = expected_score(
            away_rating,
            home_rating + HOME_ADVANTAGE
        )

        home_goals = row.get("FTHG", 0)
        away_goals = row.get("FTAG", 0)

        if home_goals > away_goals:
            home_actual = 1
            away_actual = 0

        elif home_goals < away_goals:
            home_actual = 0
            away_actual = 1

        else:
            home_actual = 0.5
            away_actual = 0.5

        new_home = update_elo(
            home_rating,
            home_expected,
            home_actual
        )

        new_away = update_elo(
            away_rating,
            away_expected,
            away_actual
        )

        ratings[home] = new_home
        ratings[away] = new_away

    return ratings


def get_match_elo(home_team, away_team, ratings):

    home_elo = ratings.get(home_team, BASE_ELO)
    away_elo = ratings.get(away_team, BASE_ELO)

    diff = (home_elo + HOME_ADVANTAGE) - away_elo

    return {
        "home_elo": round(home_elo, 1),
        "away_elo": round(away_elo, 1),
        "elo_diff": round(diff, 1)
    }