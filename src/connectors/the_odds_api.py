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


FALLBACK_TENNIS_SPORTS = [
    "tennis_atp_french_open",
    "tennis_atp_hamburg_open",
    "tennis_wta_french_open",
    "tennis_wta_strasbourg",
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

        if tennis_sports:
            return tennis_sports

        return FALLBACK_TENNIS_SPORTS

    except Exception as e:
        print("Erreur récupération sports tennis :", e)

        return FALLBACK_TENNIS_SPORTS


def fetch_sport_odds(sport):
    url = f"{BASE_URL}/sports/{sport}/odds"

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": "h2h",
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

            home_team = ev.get("home_team")
            away_team = ev.get("away_team")

            best_home = None
            best_away = None
            best_draw = None

            for bookmaker in bookmakers:
                for market in bookmaker.get("markets", []):

                    if market.get("key") != "h2h":
                        continue

                    outcomes = market.get("outcomes", [])

                    if "tennis" in sport and len(outcomes) >= 2:
                        if not home_team:
                            home_team = outcomes[0].get("name")

                        if not away_team:
                            away_team = outcomes[1].get("name")

                    for outcome in outcomes:
                        name = outcome.get("name", "")
                        price = outcome.get("price", None)

                        if price is None:
                            continue

                        price = float(price)

                        if name == home_team:
                            best_home = max(best_home or 0, price)

                        elif name == away_team:
                            best_away = max(best_away or 0, price)

                        elif name.lower() == "draw":
                            best_draw = max(best_draw or 0, price)

                    if "tennis" in sport and len(outcomes) >= 2:
                        if best_home is None:
                            home_team = outcomes[0].get("name")
                            best_home = float(outcomes[0].get("price", 0))

                        if best_away is None:
                            away_team = outcomes[1].get("name")
                            best_away = float(outcomes[1].get("price", 0))

            rows.append({
                "sport": sport,
                "commence_time": ev.get("commence_time"),
                "home_team": home_team,
                "away_team": away_team,
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away,
                "source": "the-odds-api"
            })

        print(f"{sport} : {len(rows)} matchs récupérés")

    except Exception as e:
        print(f"Erreur odds {sport}: {e}")

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

    print("Fichier cotes créé : data/processed/upcoming_odds.csv")
    print("Total matchs récupérés :", len(df))

    return df