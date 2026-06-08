from pathlib import Path

import pandas as pd

try:
    import requests
except Exception:
    requests = None

from src.utils.config import ODDS_API_KEY, REGION


BASE_URL = "https://api.the-odds-api.com/v4"
UPCOMING_PATH = Path("data/processed/upcoming_odds.csv")
LOCAL_TZ = "Europe/Paris"

FOOTBALL_SPORTS = [
    "soccer_epl",
    "soccer_efl_champ",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_italy_serie_a",
    "soccer_italy_serie_b",
    "soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie",
    "soccer_belgium_first_div",
    "soccer_uefa_champs_league",
    "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_nations_league",
    "soccer_uefa_euro_qualification",
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_qualification",
    "soccer_fifa_club_world_cup",
    "soccer_usa_mls",
    "soccer_brazil_campeonato",
    "soccer_argentina_primera_division",
    "soccer_mexico_ligamx",
    "soccer_copa_america",
    "soccer_conmebol_copa_libertadores",
    "soccer_conmebol_copa_sudamericana",
]

FALLBACK_TENNIS_SPORTS = [
    "tennis_atp_french_open",
    "tennis_wta_french_open",
    "tennis_atp_hamburg_open",
    "tennis_wta_strasbourg",
]

OFFLINE_FOOTBALL_FIXTURES = [
    ("soccer_fifa_world_cup", "2026-06-11T19:00:00Z", "Mexico", "South Africa", 1.53, 4.20, 6.60),
    ("soccer_fifa_world_cup", "2026-06-16T19:00:00Z", "France", "Senegal", 1.48, 4.40, 7.00),
    ("soccer_fifa_world_cup", "2026-06-17T01:00:00Z", "Argentina", "Algeria", 1.43, 4.55, 7.80),
    ("soccer_fifa_world_cup", "2026-06-18T19:00:00Z", "Switzerland", "Bosnia & Herzegovina", 1.58, 3.95, 6.10),
    ("soccer_fifa_world_cup", "2026-06-18T22:00:00Z", "Canada", "Qatar", 1.53, 4.10, 6.30),
    ("soccer_fifa_world_cup", "2026-06-20T20:00:00Z", "Germany", "Ivory Coast", 1.57, 4.05, 6.20),
    ("soccer_fifa_world_cup", "2026-06-21T19:00:00Z", "Belgium", "Iran", 1.48, 4.25, 7.20),
    ("soccer_fifa_world_cup", "2026-06-21T22:00:00Z", "Uruguay", "Cape Verde", 1.48, 4.20, 7.10),
    ("soccer_fifa_world_cup", "2026-06-24T02:00:00Z", "Colombia", "DR Congo", 1.55, 4.00, 6.20),
    ("soccer_fifa_world_cup", "2026-06-26T19:00:00Z", "Senegal", "Iraq", 1.53, 4.10, 6.40),
]

VERIFIED_TODAY_FOOTBALL_FIXTURES = [
    # Amicaux internationaux ajoutes quand l'API ne remonte pas le foot du jour.
    ("soccer_international_friendlies", "2026-06-03T17:00:00Z", "Gibraltar", "British Virgin Islands", 1.38, 5.55, 9.10),
    ("soccer_international_friendlies", "2026-06-03T18:00:00Z", "Albania", "Israel", 2.51, 3.40, 2.90),
    ("soccer_international_friendlies", "2026-06-03T18:00:00Z", "DR Congo", "Denmark", 6.00, 3.80, 1.50),
    ("soccer_international_friendlies", "2026-06-03T18:45:00Z", "Luxembourg", "Italy", 7.50, 4.80, 1.47),
    ("soccer_international_friendlies", "2026-06-03T18:45:00Z", "Netherlands", "Algeria", 1.30, 5.60, 10.00),
    ("soccer_international_friendlies", "2026-06-03T18:45:00Z", "Poland", "Nigeria", 2.10, 3.45, 3.80),
    ("soccer_international_friendlies", "2026-06-04T11:00:00Z", "Maldives", "Pakistan", 2.35, 3.25, 3.10),
    ("soccer_international_friendlies", "2026-06-04T12:00:00Z", "Cambodia", "Bhutan", 1.82, 3.40, 4.50),
    ("soccer_international_friendlies", "2026-06-04T13:00:00Z", "Lesotho", "Kenya", 4.50, 3.40, 1.82),
    ("soccer_international_friendlies", "2026-06-04T16:00:00Z", "Northern Ireland", "Guinea", 2.35, 3.23, 3.51),
    ("soccer_international_friendlies", "2026-06-04T16:00:00Z", "Slovenia", "Cyprus", 1.37, 4.86, 9.80),
    ("soccer_international_friendlies", "2026-06-04T16:00:00Z", "Equatorial Guinea", "Burundi", 1.84, 3.43, 5.50),
    ("soccer_international_friendlies", "2026-06-04T17:00:00Z", "Andorra", "Liechtenstein", 1.58, 3.55, 8.00),
    ("soccer_international_friendlies", "2026-06-04T17:00:00Z", "Sweden", "Greece", 2.04, 3.65, 4.80),
    ("soccer_international_friendlies", "2026-06-04T19:00:00Z", "Spain", "Iraq", 1.05, 21.00, 70.00),
    ("soccer_international_friendlies", "2026-06-04T19:10:00Z", "France", "Côte d'Ivoire", 1.35, 5.50, 9.75),
]


def _parse_dt(value):
    try:
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _today_local():
    return pd.Timestamp.now(tz=LOCAL_TZ).date()


def _local_date(value):
    dt = _parse_dt(value)
    if pd.isna(dt):
        return None
    return dt.tz_convert(LOCAL_TZ).date()


def _is_in_window(sport, commence_time):
    dt = _parse_dt(commence_time)
    if pd.isna(dt):
        return True

    now = pd.Timestamp.now(tz="UTC")
    sport_l = str(sport).lower()
    if "tennis" in sport_l:
        days = 5
    elif "world_cup" in sport_l or "international" in sport_l:
        days = 35
    elif "soccer" in sport_l or "football" in sport_l:
        days = 21
    else:
        days = 3

    return (dt >= now - pd.Timedelta(hours=6)) and (dt <= now + pd.Timedelta(days=days))


def _is_match_market_key(key):
    key = str(key).lower()
    return not any(x in key for x in ["winner", "outright", "futures"])


def get_available_sports():
    if requests is None:
        print("Module requests indisponible : utilisation du fallback local.")
        return []

    try:
        r = requests.get(f"{BASE_URL}/sports", params={"apiKey": ODDS_API_KEY}, timeout=30)
        r.raise_for_status()
        return [s.get("key", "") for s in r.json() if s.get("key")]
    except Exception as e:
        print("Erreur recuperation sports disponibles :", e)
        return []


def get_available_tennis_sports(available=None):
    keys = available if available is not None else get_available_sports()
    tennis = [k for k in keys if "tennis" in k.lower() and _is_match_market_key(k)]
    merged = list(dict.fromkeys(tennis + FALLBACK_TENNIS_SPORTS))
    print("Tournois tennis utilises :", merged)
    return merged


def get_available_football_sports(available=None):
    keys = available if available is not None else get_available_sports()
    key_set = set(keys)
    priority = [s for s in FOOTBALL_SPORTS if not key_set or s in key_set]
    football = [
        k for k in keys
        if k in FOOTBALL_SPORTS and _is_match_market_key(k)
    ]
    merged = list(dict.fromkeys(priority + football)) or FOOTBALL_SPORTS
    print("Competitions foot utilisees :", merged)
    return merged


def _best_prices(event, sport):
    bookmakers = event.get("bookmakers", [])
    home_team, away_team = event.get("home_team"), event.get("away_team")
    best_home = best_away = best_draw = None

    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue

            outcomes = market.get("outcomes", [])
            if "tennis" in sport and len(outcomes) >= 2:
                home_team = home_team or outcomes[0].get("name")
                away_team = away_team or outcomes[1].get("name")

            for outcome in outcomes:
                name, price = outcome.get("name", ""), outcome.get("price")
                if price is None:
                    continue

                price = float(price)
                if name == home_team:
                    best_home = max(best_home or 0, price)
                elif name == away_team:
                    best_away = max(best_away or 0, price)
                elif str(name).lower() == "draw":
                    best_draw = max(best_draw or 0, price)

            if "tennis" in sport and len(outcomes) >= 2:
                if best_home is None:
                    home_team, best_home = outcomes[0].get("name"), float(outcomes[0].get("price", 0))
                if best_away is None:
                    away_team, best_away = outcomes[1].get("name"), float(outcomes[1].get("price", 0))

    return home_team, away_team, best_home, best_draw, best_away


def fetch_sport_odds(sport):
    if requests is None:
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": "h2h",
        "oddsFormat": "decimal",
    }
    rows = []

    try:
        r = requests.get(f"{BASE_URL}/sports/{sport}/odds", params=params, timeout=30)
        r.raise_for_status()
        for ev in r.json():
            commence_time = ev.get("commence_time")
            if not _is_in_window(sport, commence_time):
                continue

            home_team, away_team, best_home, best_draw, best_away = _best_prices(ev, sport)
            if not home_team or not away_team or not best_home or not best_away:
                continue

            rows.append({
                "sport": sport,
                "commence_time": commence_time,
                "home_team": home_team,
                "away_team": away_team,
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away,
                "source": "the-odds-api",
            })
        print(f"{sport} : {len(rows)} matchs recuperes")
    except Exception as e:
        print(f"Erreur odds {sport}: {e}")

    return rows


def offline_football_rows():
    rows = []
    for sport, commence_time, home, away, home_odds, draw_odds, away_odds in OFFLINE_FOOTBALL_FIXTURES:
        if not _is_in_window(sport, commence_time):
            continue
        rows.append({
            "sport": sport,
            "commence_time": commence_time,
            "home_team": home,
            "away_team": away,
            "odds_home": home_odds,
            "odds_draw": draw_odds,
            "odds_away": away_odds,
            "source": "offline-fallback",
        })
    return rows


def verified_today_football_rows():
    rows = []
    today = _today_local()
    for sport, commence_time, home, away, home_odds, draw_odds, away_odds in VERIFIED_TODAY_FOOTBALL_FIXTURES:
        if _local_date(commence_time) != today:
            continue
        rows.append({
            "sport": sport,
            "commence_time": commence_time,
            "home_team": home,
            "away_team": away,
            "odds_home": home_odds,
            "odds_draw": draw_odds,
            "odds_away": away_odds,
            "source": "verified-web-fallback",
        })
    return rows


def add_missing_verified_today_football(df):
    fallback = pd.DataFrame(verified_today_football_rows())
    if fallback.empty:
        return df

    if df.empty:
        print("Ajout du fallback web foot du jour.")
        return fallback

    existing = df.copy()
    for col in ["sport", "commence_time", "home_team", "away_team"]:
        if col not in existing.columns:
            existing[col] = ""

    existing["_key"] = (
        existing["sport"].astype(str).str.lower()
        + "|"
        + existing["commence_time"].astype(str).str[:10]
        + "|"
        + existing["home_team"].astype(str).str.lower()
        + "|"
        + existing["away_team"].astype(str).str.lower()
    )
    fallback["_key"] = (
        fallback["sport"].astype(str).str.lower()
        + "|"
        + fallback["commence_time"].astype(str).str[:10]
        + "|"
        + fallback["home_team"].astype(str).str.lower()
        + "|"
        + fallback["away_team"].astype(str).str.lower()
    )
    missing = fallback[~fallback["_key"].isin(set(existing["_key"]))].drop(columns=["_key"], errors="ignore")
    existing = existing.drop(columns=["_key"], errors="ignore")

    if not missing.empty:
        print(f"Ajout fallback web foot du jour : {len(missing)} match(s).")
        existing = pd.concat([existing, missing], ignore_index=True)
    return existing


def write_upcoming(df):
    UPCOMING_PATH.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        df.to_csv(UPCOMING_PATH, index=False)
        return df

    df["_dt"] = df["commence_time"].apply(_parse_dt)
    df = df.sort_values(["sport", "_dt", "home_team", "away_team"]).drop(columns=["_dt"], errors="ignore")
    df["_key"] = (
        df["sport"].astype(str).str.lower()
        + "|"
        + df["commence_time"].astype(str).str[:10]
        + "|"
        + df["home_team"].astype(str).str.lower()
        + "|"
        + df["away_team"].astype(str).str.lower()
    )
    df = df.drop_duplicates("_key", keep="last").drop(columns=["_key"])
    df.to_csv(UPCOMING_PATH, index=False)
    return df


def fetch_upcoming_odds():
    if not ODDS_API_KEY or "COLLE" in str(ODDS_API_KEY):
        print("ODDS_API_KEY manquante.")
        existing = pd.read_csv(UPCOMING_PATH) if UPCOMING_PATH.exists() else pd.DataFrame()
        existing = add_missing_verified_today_football(existing)
        if existing.empty or not existing.get("sport", pd.Series(dtype=str)).astype(str).str.contains("soccer|football", case=False, regex=True).any():
            existing = pd.concat([existing, pd.DataFrame(offline_football_rows())], ignore_index=True)
            existing = write_upcoming(existing)
        else:
            existing = write_upcoming(existing)
        return existing

    available = get_available_sports()
    sports = list(dict.fromkeys(get_available_football_sports(available) + get_available_tennis_sports(available)))

    all_rows = []
    for sport in sports:
        all_rows.extend(fetch_sport_odds(sport))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("Aucun match recupere. Ancien upcoming_odds.csv conserve si present.")
        df = pd.read_csv(UPCOMING_PATH) if UPCOMING_PATH.exists() else pd.DataFrame()

    df = add_missing_verified_today_football(df)

    has_football = (
        not df.empty
        and df.get("sport", pd.Series(dtype=str)).astype(str).str.contains("soccer|football", case=False, regex=True).any()
    )
    if not has_football:
        print("Aucun foot live recupere : ajout du fallback local football.")
        df = pd.concat([df, pd.DataFrame(offline_football_rows())], ignore_index=True)

    df = write_upcoming(df)
    print("Fichier cotes cree : data/processed/upcoming_odds.csv")
    print("Total matchs recuperes :", len(df))
    if not df.empty:
        print(df["sport"].value_counts().to_string())
    return df
