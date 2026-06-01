import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[2]

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")
else:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(
                key.strip(),
                value.strip().strip('"').strip("'"),
            )

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
BANKROLL_START = float(os.getenv("BANKROLL_START", "10"))
MAX_STAKE_PCT = float(os.getenv("MAX_STAKE_PCT", "0.03"))
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.03"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.58"))
REGION = os.getenv("REGION", "eu")
