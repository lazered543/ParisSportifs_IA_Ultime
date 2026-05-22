import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")

url = "https://api.the-odds-api.com/v4/sports"

params = {
    "apiKey": API_KEY
}

response = requests.get(url, params=params, timeout=20)
response.raise_for_status()

sports = response.json()

print("\n=== SPORTS TENNIS DISPONIBLES ===\n")

for sport in sports:
    key = sport.get("key", "")
    title = sport.get("title", "")

    if "tennis" in key.lower() or "tennis" in title.lower():
        print(key, "=>", title)