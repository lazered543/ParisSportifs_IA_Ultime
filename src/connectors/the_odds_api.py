import requests
import pandas as pd
from pathlib import Path
from src.utils.config import ODDS_API_KEY, REGION

BASE_URL = "https://api.the-odds-api.com/v4"

FOOTBALL_SPORTS = [
    "soccer_epl", "soccer_france_ligue_one", "soccer_france_ligue_two",
    "soccer_spain_la_liga", "soccer_germany_bundesliga", "soccer_italy_serie_a",
    "soccer_uefa_champs_league", "soccer_uefa_europa_league", "soccer_fifa_world_cup",
]
FALLBACK_TENNIS_SPORTS = ["tennis_atp_french_open", "tennis_wta_french_open", "tennis_atp_hamburg_open", "tennis_wta_strasbourg"]

def _parse_dt(value):
    try: return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception: return pd.NaT

def _is_in_window(sport, commence_time):
    dt = _parse_dt(commence_time)
    if pd.isna(dt): return True
    now = pd.Timestamp.now(tz="UTC")
    sport_l = str(sport).lower()
    days = 2 if "tennis" in sport_l else 10 if ("world_cup" in sport_l or "international" in sport_l) else 7 if ("soccer" in sport_l or "football" in sport_l) else 3
    return (dt >= now - pd.Timedelta(hours=6)) and (dt <= now + pd.Timedelta(days=days))

def get_available_sports():
    try:
        r = requests.get(f"{BASE_URL}/sports", params={"apiKey": ODDS_API_KEY}, timeout=30)
        r.raise_for_status()
        return [s.get("key", "") for s in r.json() if s.get("key")]
    except Exception as e:
        print("Erreur récupération sports disponibles :", e)
        return []

def get_available_tennis_sports():
    keys = get_available_sports()
    tennis = [k for k in keys if "tennis" in k.lower() and any(x in k.lower() for x in ["atp", "wta", "french", "roland", "open"])]
    merged = list(dict.fromkeys(tennis + FALLBACK_TENNIS_SPORTS))
    print("Tournois tennis utilisés :", merged)
    return merged

def get_available_football_sports():
    keys = set(get_available_sports())
    merged = [s for s in FOOTBALL_SPORTS if (not keys or s in keys)] or FOOTBALL_SPORTS
    print("Compétitions foot utilisées :", merged)
    return merged

def _best_prices(event, sport):
    bookmakers = event.get("bookmakers", [])
    home_team, away_team = event.get("home_team"), event.get("away_team")
    best_home = best_away = best_draw = None
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h": continue
            outcomes = market.get("outcomes", [])
            if "tennis" in sport and len(outcomes) >= 2:
                home_team = home_team or outcomes[0].get("name")
                away_team = away_team or outcomes[1].get("name")
            for outcome in outcomes:
                name, price = outcome.get("name", ""), outcome.get("price")
                if price is None: continue
                price = float(price)
                if name == home_team: best_home = max(best_home or 0, price)
                elif name == away_team: best_away = max(best_away or 0, price)
                elif str(name).lower() == "draw": best_draw = max(best_draw or 0, price)
            if "tennis" in sport and len(outcomes) >= 2:
                if best_home is None:
                    home_team, best_home = outcomes[0].get("name"), float(outcomes[0].get("price", 0))
                if best_away is None:
                    away_team, best_away = outcomes[1].get("name"), float(outcomes[1].get("price", 0))
    return home_team, away_team, best_home, best_draw, best_away

def fetch_sport_odds(sport):
    params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": "h2h", "oddsFormat": "decimal"}
    rows = []
    try:
        r = requests.get(f"{BASE_URL}/sports/{sport}/odds", params=params, timeout=30)
        r.raise_for_status()
        for ev in r.json():
            commence_time = ev.get("commence_time")
            if not _is_in_window(sport, commence_time): continue
            home_team, away_team, best_home, best_draw, best_away = _best_prices(ev, sport)
            if not home_team or not away_team or not best_home or not best_away: continue
            rows.append({"sport":sport,"commence_time":commence_time,"home_team":home_team,"away_team":away_team,"odds_home":best_home,"odds_draw":best_draw,"odds_away":best_away,"source":"the-odds-api"})
        print(f"{sport} : {len(rows)} matchs récupérés")
    except Exception as e:
        print(f"Erreur odds {sport}: {e}")
    return rows

def fetch_upcoming_odds():
    if not ODDS_API_KEY or "COLLE" in str(ODDS_API_KEY):
        print("ODDS_API_KEY manquante.")
        return pd.DataFrame()
    sports = list(dict.fromkeys(get_available_football_sports() + get_available_tennis_sports()))
    all_rows = []
    for sport in sports: all_rows.extend(fetch_sport_odds(sport))
    df = pd.DataFrame(all_rows)
    out_path = Path("data/processed/upcoming_odds.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("Aucun match récupéré. Ancien upcoming_odds.csv conservé si présent.")
        if out_path.exists(): return pd.read_csv(out_path)
        df.to_csv(out_path, index=False)
        return df
    df["_dt"] = df["commence_time"].apply(_parse_dt)
    df = df.sort_values(["sport", "_dt", "home_team", "away_team"]).drop(columns=["_dt"], errors="ignore")
    df["_key"] = df["sport"].astype(str).str.lower() + "|" + df["commence_time"].astype(str).str[:10] + "|" + df["home_team"].astype(str).str.lower() + "|" + df["away_team"].astype(str).str.lower()
    df = df.drop_duplicates("_key", keep="last").drop(columns=["_key"])
    df.to_csv(out_path, index=False)
    print("Fichier cotes créé : data/processed/upcoming_odds.csv")
    print("Total matchs récupérés :", len(df))
    print(df["sport"].value_counts().to_string())
    return df
