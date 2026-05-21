import pandas as pd


def build_team_strength(history_df):

    teams = []

    grouped = history_df.groupby("HomeTeam")

    for team, matches in grouped:

        recent = matches.tail(5)

        avg_goals_scored = recent["FTHG"].mean()
        avg_goals_conceded = recent["FTAG"].mean()

        wins = (recent["FTHG"] > recent["FTAG"]).sum()
        draws = (recent["FTHG"] == recent["FTAG"]).sum()

        form_points = (wins * 3 + draws) / 15

        attack = avg_goals_scored
        defense = max(0.5, avg_goals_conceded)

        teams.append({
            "team": team,
            "attack": round(attack, 2),
            "defense": round(defense, 2),
            "form": round(form_points, 2)
        })

    return pd.DataFrame(teams)


def estimate_xg(home_team, away_team, strengths):

    home_data = strengths[
        strengths["team"] == home_team
    ]

    away_data = strengths[
        strengths["team"] == away_team
    ]

    if home_data.empty:
        home_attack = 1.5
        home_form = 0.5
    else:
        home_attack = home_data.iloc[0]["attack"]
        home_form = home_data.iloc[0]["form"]

    if away_data.empty:
        away_attack = 1.2
        away_form = 0.5
    else:
        away_attack = away_data.iloc[0]["attack"]
        away_form = away_data.iloc[0]["form"]

    # Boost forme récente
    home_xg = (
        home_attack * 0.7
        + home_form * 1.2
        + 0.35
    )

    away_xg = (
        away_attack * 0.7
        + away_form * 0.9
    )

    # Limites réalistes
    home_xg = min(max(home_xg, 0.4), 3.5)
    away_xg = min(max(away_xg, 0.3), 3.0)

    return round(home_xg, 2), round(away_xg, 2)