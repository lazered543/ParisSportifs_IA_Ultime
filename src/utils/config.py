import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BANKROLL_START = float(os.getenv("BANKROLL_START", "100"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.03"))
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.03"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.58"))
REGION = os.getenv("REGION", "eu")
