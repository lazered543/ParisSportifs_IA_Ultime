import requests
import pandas as pd

from pathlib import Path
from src.utils.config import ODDS_API_KEY, REGION


BASE_URL = "https://api.the-odds-api.com/v4"


FOOTBALL_SPORTS = [
    "soccer_epl",
    "soccer_france_ligue_one",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
]


def get_available_tennis_sports():
    url = f"{BASE_URL}/sports"

    params = {
        "apiKey": ODDS_API_KEY
    }

    try:
        r = requests.get(
            url,
            params=params,
            timeout=30
        )

        r.raise_for_status()

        sports = r.json()

        tennis_sports = []

        for sport in sports:
            key = sport.get("key", "")
            title = sport.get("title", "")

            if "tennis" in key.lower() or "tennis" in title.lower():
                tennis_sports.append(key)

        print("Tournois tennis disponibles :", tennis_sports)

        return tennis_sports

    except Exception as e:
        print("Erreur récupération sports tennis :", e)

        return [
            "tennis_atp_french_open",
            "tennis_atp_hamburg_open",
            "tennis_wta_french_open",
            "tennis_wta_strasbourg",
        ]


def fetch_sport_odds(sport):
    url = f"{BASE_URL}/sports/{sport}/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }

    rows = []

    try:
        r = requests.get(
            url,
            params=params,
            timeout=30
        )

        r.raise_for_status()

        events = r.json()

        for ev in events:
            bookmakers = ev.get("bookmakers", [])

            best_home = None
            best_away = None
            best_draw = None

            for bk in bookmakers:
                for market in bk.get("markets", []):

                    if market.get("key") != "h2h":
                        continue

                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = float(outcome.get("price", 0))

                        if name == ev.get("home_team"):
                            best_home = max(
                                best_home or 0,
                                price
                            )

                        elif name == ev.get("away_team"):
                            best_away = max(
                                best_away or 0,
                                price
                            )

                        elif name.lower() == "draw":
                            best_draw = max(
                                best_draw or 0,
                                price
                            )

            rows.append({
                "sport": sport,
                "commence_time": ev.get("commence_time"),
                "home_team": ev.get("home_team"),
                "away_team": ev.get("away_team"),
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away,
                "source": "the-odds-api"
            })

        print(
            f"{sport} : {len(rows)} matchs récupérés"
        )

    except Exception as e:
        print(
            f"Erreur odds {sport}: {e}"
        )

    return rows


def fetch_upcoming_odds():
    if not ODDS_API_KEY or "COLLE" in ODDS_API_KEY:
        print("ODDS_API_KEY manquante : mode demo.")
        return pd.DataFrame()

    tennis_sports = get_available_tennis_sports()

    sports = FOOTBALL_SPORTS + tennis_sports

    all_rows = []

    for sport in sports:
        rows = fetch_sport_odds(sport)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        "data/processed/upcoming_odds.csv",
        index=False
    )

    print(
        "Fichier cotes créé : data/processed/upcoming_odds.csv"
    )

    print(
        "Total matchs récupérés :",
        len(df)
    )

    return df