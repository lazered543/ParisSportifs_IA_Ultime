import pandas as pd


def build_team_strength(history_df):
    history_df = history_df.dropna(
        subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ).copy()

    rows = []

    for _, match in history_df.iterrows():
        rows.append({
            "team": match["HomeTeam"],
            "goals_for": float(match["FTHG"]),
            "goals_against": float(match["FTAG"]),
            "points": 3 if match["FTHG"] > match["FTAG"] else 1 if match["FTHG"] == match["FTAG"] else 0,
            "is_home": 1,
        })
        rows.append({
            "team": match["AwayTeam"],
            "goals_for": float(match["FTAG"]),
            "goals_against": float(match["FTHG"]),
            "points": 3 if match["FTAG"] > match["FTHG"] else 1 if match["FTHG"] == match["FTAG"] else 0,
            "is_home": 0,
        })

    team_matches = pd.DataFrame(rows)
    teams = []

    for team, matches in team_matches.groupby("team", sort=False):
        recent = matches.tail(8)

        avg_goals_scored = recent["goals_for"].mean()
        avg_goals_conceded = recent["goals_against"].mean()
        form_points = recent["points"].sum() / max(len(recent) * 3, 1)

        teams.append({
            "team": team,
            "attack": round(max(0.2, avg_goals_scored), 2),
            "defense": round(max(0.2, avg_goals_conceded), 2),
            "form": round(form_points, 2),
            "matches": int(len(matches)),
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
        home_attack = 1.35
        home_defense = 1.15
        home_form = 0.5
    else:
        home_attack = home_data.iloc[0]["attack"]
        home_defense = home_data.iloc[0]["defense"]
        home_form = home_data.iloc[0]["form"]

    if away_data.empty:
        away_attack = 1.15
        away_defense = 1.30
        away_form = 0.5
    else:
        away_attack = away_data.iloc[0]["attack"]
        away_defense = away_data.iloc[0]["defense"]
        away_form = away_data.iloc[0]["form"]

    form_gap = home_form - away_form

    home_xg = (
        0.42 * home_attack
        + 0.28 * away_defense
        + 0.22 * home_form
        + 0.22
        + max(form_gap, -0.35) * 0.12
    )

    away_xg = (
        0.42 * away_attack
        + 0.28 * home_defense
        + 0.20 * away_form
        + 0.12
        - max(form_gap, -0.35) * 0.08
    )

    home_xg = min(max(home_xg, 0.35), 3.8)
    away_xg = min(max(away_xg, 0.25), 3.4)

    return round(home_xg, 2), round(away_xg, 2)
