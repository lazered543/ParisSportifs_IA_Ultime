import requests
import pandas as pd
from src.utils.config import API_FOOTBALL_KEY

BASE = "https://v3.football.api-sports.io"

def fetch_fixtures_today():
    if not API_FOOTBALL_KEY or "COLLE" in API_FOOTBALL_KEY:
        print("API_FOOTBALL_KEY manquante : mode demo.")
        return pd.DataFrame()
    headers = {"x-apisports-key": API_FOOTBALL_KEY}
    url = f"{BASE}/fixtures"
    params = {"date": pd.Timestamp.today().strftime("%Y-%m-%d")}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    rows = []
    for item in r.json().get("response", []):
        rows.append({
            "fixture_id": item["fixture"]["id"],
            "date": item["fixture"]["date"],
            "league": item["league"]["name"],
            "country": item["league"]["country"],
            "home_team": item["teams"]["home"]["name"],
            "away_team": item["teams"]["away"]["name"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/api_football_fixtures_today.csv", index=False)
    return df
