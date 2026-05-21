import requests
import pandas as pd
from src.utils.config import ODDS_API_KEY, REGION

SPORTS = [
    "soccer_epl", "soccer_france_ligue_one", "soccer_spain_la_liga",
    "soccer_germany_bundesliga", "soccer_italy_serie_a",
    "tennis_atp", "tennis_wta"
]

def fetch_upcoming_odds():
    if not ODDS_API_KEY or "COLLE" in ODDS_API_KEY:
        print("ODDS_API_KEY manquante : mode demo.")
        return pd.DataFrame()
    rows = []
    for sport in SPORTS:
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": REGION,
            "markets": "h2h,totals",
            "oddsFormat": "decimal"
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            for ev in r.json():
                bookmakers = ev.get("bookmakers", [])
                best_home = best_away = best_draw = None
                for bk in bookmakers:
                    for m in bk.get("markets", []):
                        if m.get("key") == "h2h":
                            for o in m.get("outcomes", []):
                                name = o.get("name", "")
                                price = float(o.get("price", 0))
                                if name == ev.get("home_team"):
                                    best_home = max(best_home or 0, price)
                                elif name == ev.get("away_team"):
                                    best_away = max(best_away or 0, price)
                                elif name.lower() == "draw":
                                    best_draw = max(best_draw or 0, price)
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
        except Exception as e:
            print(f"Erreur odds {sport}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/upcoming_odds.csv", index=False)
    return df
