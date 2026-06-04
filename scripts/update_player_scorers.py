import os
import time
import pandas as pd

from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import requests
except Exception:
    requests = None

if load_dotenv is not None:
    load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

BASE_URL = "https://v3.football.api-sports.io"

HEADERS = {
    "x-apisports-key": API_KEY
}

# Grandes ligues européennes
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

# Multi saisons
SEASONS = [
    2024,
    2025,
]


def api_get(endpoint, params):
    if requests is None:
        raise RuntimeError("Module requests indisponible")

    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json().get(
        "response",
        []
    )


def compute_scorer_score(
    goals,
    assists,
    shots,
    shots_on,
    penalties,
    minutes,
    season_weight,
):

    minutes_factor = min(
        minutes / 900,
        5
    )

    score = (

        goals * 5

        + assists * 1.2

        + shots_on * 1.5

        + shots * 0.45

        + penalties * 2.5

        + minutes_factor

    )

    return round(
        score * season_weight,
        2
    )


def get_players_for_league(
    league_id,
    league_name,
    season,
):

    rows = []

    print(
        f"\n=== {league_name} {season} ==="
    )

    try:

        data = api_get(

            "players",

            {
                "league": league_id,
                "season": season,
                "page": 1,
            }
        )

        # pagination
        paging = 1

        while True:

            print(
                f"Page {paging}"
            )

            data = api_get(

                "players",

                {
                    "league": league_id,
                    "season": season,
                    "page": paging,
                }
            )

            if not data:
                break

            for item in data:

                player = item.get(
                    "player",
                    {}
                )

                statistics = item.get(
                    "statistics",
                    []
                )

                if not statistics:
                    continue

                stats = statistics[0]

                team = stats.get(
                    "team",
                    {}
                )

                games = stats.get(
                    "games",
                    {}
                )

                goals = stats.get(
                    "goals",
                    {}
                )

                shots = stats.get(
                    "shots",
                    {}
                )

                penalty = stats.get(
                    "penalty",
                    {}
                )

                minutes = (
                    games.get(
                        "minutes"
                    ) or 0
                )

                total_goals = (
                    goals.get(
                        "total"
                    ) or 0
                )

                assists = (
                    goals.get(
                        "assists"
                    ) or 0
                )

                total_shots = (
                    shots.get(
                        "total"
                    ) or 0
                )

                shots_on = (
                    shots.get(
                        "on"
                    ) or 0
                )

                penalties = (
                    penalty.get(
                        "scored"
                    ) or 0
                )

                # pondération saison
                if season == 2025:
                    season_weight = 1.0
                else:
                    season_weight = 0.55

                scorer_score = compute_scorer_score(

                    total_goals,
                    assists,
                    total_shots,
                    shots_on,
                    penalties,
                    minutes,
                    season_weight,
                )

                rows.append({

                    "season":
                        season,

                    "league_id":
                        league_id,

                    "league_name":
                        league_name,

                    "team":
                        team.get(
                            "name"
                        ),

                    "player":
                        player.get(
                            "name"
                        ),

                    "age":
                        player.get(
                            "age"
                        ),

                    "position":
                        stats.get(
                            "games",
                            {}
                        ).get(
                            "position"
                        ),

                    "minutes":
                        minutes,

                    "goals":
                        total_goals,

                    "assists":
                        assists,

                    "shots":
                        total_shots,

                    "shots_on":
                        shots_on,

                    "penalties":
                        penalties,

                    "scorer_score":
                        scorer_score,
                })

            paging += 1

            time.sleep(8)

    except Exception as e:

        print(
            f"Erreur {league_name} {season} :",
            e
        )

    return rows


def main():
    if requests is None:
        print("Module requests indisponible : player_scorers.csv existant conserve.")
        return

    if not API_KEY:

        print(
            "FOOTBALL_API_KEY manquante"
        )

        return

    all_rows = []

    for season in SEASONS:

        for league_id, league_name in LEAGUES.items():

            rows = get_players_for_league(

                league_id,
                league_name,
                season,
            )

            all_rows.extend(
                rows
            )

            time.sleep(12)

    df = pd.DataFrame(
        all_rows
    )

    if df.empty:

        print(
            "Aucune donnée récupérée."
        )

        return

    # fusion multi-saisons
    final_df = (

        df.groupby(
            [
                "team",
                "player",
                "position",
            ],
            as_index=False,
        )

        .agg({

            "minutes": "sum",

            "goals": "sum",

            "assists": "sum",

            "shots": "sum",

            "shots_on": "sum",

            "penalties": "sum",

            "scorer_score": "sum",
        })
    )

    final_df = final_df.sort_values(

        [
            "team",
            "scorer_score",
        ],

        ascending=[
            True,
            False,
        ]
    )

    Path(
        "data/processed"
    ).mkdir(

        parents=True,
        exist_ok=True,
    )

    final_df.to_csv(

        "data/processed/player_scorers.csv",

        index=False,
    )

    print(
        "\n======================"
    )

    print(
        "PLAYER SCORER ENGINE V2 OK"
    )

    print(
        "Joueurs récupérés :",
        len(final_df)
    )

    print(
        "Fichier sauvegardé :"
    )

    print(
        "data/processed/player_scorers.csv"
    )

    print(
        "======================\n"
    )


if __name__ == "__main__":

    main()
