from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START_BANKROLL = 10.0
BACKUP_DIR = ROOT / "data" / "backups" / f"full_reset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def backup(path: Path):
    if not path.exists():
        return

    target = BACKUP_DIR / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def write_csv(path: Path, rows=None, columns=None):
    backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows or [], columns=columns).to_csv(path, index=False)


def reset_bankroll():
    write_csv(
        ROOT / "data" / "bankroll_state.csv",
        rows=[{
            "initial_bankroll": START_BANKROLL,
            "current_bankroll": START_BANKROLL,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "total_finished": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "roi": 0.0,
            "updated_at": datetime.now().isoformat(),
        }],
    )


def reset_tracking():
    tracking_cols = [
        "date", "sport", "category", "home_team", "away_team", "market", "selection",
        "odds_source",
        "ai_probability", "bookmaker_odds", "value", "safety_score", "safety_level",
        "stake", "bet_mode", "stake_percent", "kelly_fraction", "bankroll",
        "confidence", "ia_badge", "learning_adjustment", "calibration_adjustment",
        "decision_reason", "result", "profit", "final_winner",
        "final_score_home", "final_score_away", "status_detail", "resolved_at",
    ]
    archive_cols = [
        "resolved_at", "date", "sport", "category", "home_team", "away_team",
        "market", "selection", "odds_source", "bet_mode", "ai_probability", "bookmaker_odds",
        "value", "stake", "result", "profit", "final_winner",
        "final_score_home", "final_score_away", "status_detail",
    ]

    write_csv(ROOT / "tracking_results.csv", columns=tracking_cols)
    write_csv(ROOT / "data" / "archive" / "finished_bets_archive.csv", columns=archive_cols)
    write_csv(ROOT / "data" / "learning" / "ai_learning_profile.csv")
    write_csv(ROOT / "data" / "learning" / "ai_learning_summary.csv")
    write_csv(ROOT / "data" / "learning" / "ai_auto_learning_segments.csv")
    write_csv(ROOT / "data" / "telegram_sent.csv")


def main():
    reset_bankroll()
    reset_tracking()
    print(f"Reset complet termine. Backup cree dans : {BACKUP_DIR}")


if __name__ == "__main__":
    main()
