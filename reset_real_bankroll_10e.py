from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
BACKUP_DIR = ROOT / "data" / "backups" / f"reset_bankroll_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
START_BANKROLL = 10.0


def log(msg):
    print(f"[BANKROLL RESET] {msg}")


def backup(path: Path):
    if path.exists():
        dest = BACKUP_DIR / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        log(f"Backup créé : {dest}")


def reset_csv(path: Path, columns=None):
    backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns or []).to_csv(path, index=False)
    log(f"Réinitialisé : {path}")


def patch_config():
    path = ROOT / "src" / "utils" / "config.py"
    if not path.exists():
        log("config.py introuvable, ignoré.")
        return

    backup(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    if "BANKROLL_START" in text:
        text = re.sub(r"BANKROLL_START\s*=\s*[0-9.]+", f"BANKROLL_START = {START_BANKROLL}", text)
    else:
        text += f"\nBANKROLL_START = {START_BANKROLL}\n"

    path.write_text(text, encoding="utf-8")
    log("BANKROLL_START mis à 10€")


def create_bankroll_state():
    path = ROOT / "data" / "bankroll_state.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    backup(path)

    pd.DataFrame([{
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
    }]).to_csv(path, index=False)

    log("Bankroll initialisée à 10€")


def create_recompute_bankroll_script():
    path = ROOT / "scripts" / "recompute_bankroll.py"
    backup(path)

    code = """
from pathlib import Path
from datetime import datetime
import pandas as pd

START_BANKROLL = 10.0
TRACK_PATH = Path("tracking_results.csv")
STATE_PATH = Path("data/bankroll_state.csv")

def main():
    current = START_BANKROLL
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    losses = 0
    finished_count = 0

    if TRACK_PATH.exists():
        tr = pd.read_csv(TRACK_PATH, low_memory=False)

        if not tr.empty:
            if "date" in tr.columns:
                tr["_dt"] = pd.to_datetime(tr["date"], errors="coerce", utc=True)
                tr = tr.sort_values("_dt")
            else:
                tr["_dt"] = pd.NaT

            if "result" not in tr.columns:
                tr["result"] = "PENDING"

            tr["result"] = tr["result"].fillna("PENDING").astype(str).str.upper()
            tr["stake"] = pd.to_numeric(tr.get("stake", 0), errors="coerce").fillna(0)
            tr["profit"] = pd.to_numeric(tr.get("profit", 0), errors="coerce").fillna(0)

            finished = tr[tr["result"].isin(["WIN", "LOSS"])].copy()

            total_staked = float(finished["stake"].sum()) if not finished.empty else 0.0
            total_profit = float(finished["profit"].sum()) if not finished.empty else 0.0
            wins = int((finished["result"] == "WIN").sum())
            losses = int((finished["result"] == "LOSS").sum())
            finished_count = int(len(finished))
            current = max(0.0, START_BANKROLL + total_profit)

    roi = total_profit / total_staked if total_staked > 0 else 0.0
    winrate = wins / finished_count if finished_count > 0 else 0.0

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "initial_bankroll": START_BANKROLL,
        "current_bankroll": round(current, 2),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "total_finished": finished_count,
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate, 4),
        "roi": round(roi, 4),
        "updated_at": datetime.now().isoformat(),
    }]).to_csv(STATE_PATH, index=False)

    print("Bankroll actuelle :", round(current, 2), "€")
    print("Profit total :", round(total_profit, 2), "€")
    print("Winrate :", round(winrate * 100, 2), "%")
    print("ROI :", round(roi * 100, 2), "%")

if __name__ == "__main__":
    main()
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code.strip() + "\n", encoding="utf-8")
    log("Script créé : scripts/recompute_bankroll.py")


def patch_run_pipeline_bankroll():
    path = ROOT / "scripts" / "run_pipeline.py"
    if not path.exists():
        log("run_pipeline.py introuvable, ignoré.")
        return

    backup(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    if "def load_current_bankroll(" not in text:
        helper = """
def load_current_bankroll(default=10.0):
    path = Path("data/bankroll_state.csv")

    if not path.exists():
        return float(default)

    try:
        state = pd.read_csv(path)

        if state.empty or "current_bankroll" not in state.columns:
            return float(default)

        bankroll = pd.to_numeric(state["current_bankroll"], errors="coerce").fillna(default).iloc[0]
        bankroll = float(bankroll)

        return max(bankroll, 0.0)

    except Exception:
        return float(default)

"""
        if "\ndef main():" in text:
            text = text.replace("\ndef main():", "\n" + helper + "\ndef main():")
        else:
            text += "\n" + helper

    text = text.replace("bankroll = BANKROLL_START", "bankroll = load_current_bankroll(BANKROLL_START)")
    text = text.replace("bankroll=BANKROLL_START", "bankroll=load_current_bankroll(BANKROLL_START)")

    if "def cap_stakes_to_bankroll(" not in text:
        helper2 = """
def cap_stakes_to_bankroll(df, bankroll):
    if df.empty:
        return df

    out = df.copy()

    if "suggested_stake" not in out.columns:
        return out

    out["suggested_stake"] = pd.to_numeric(out["suggested_stake"], errors="coerce").fillna(0)

    if bankroll <= 0:
        out["suggested_stake"] = 0
        if "decision" in out.columns:
            out["decision"] = "NO BET"
        if "bet_mode" in out.columns:
            out["bet_mode"] = out["bet_mode"].replace({
                "MEGA VALUE": "WATCHLIST",
                "SAFE PICK": "WATCHLIST",
                "VALUE BET": "WATCHLIST",
                "RISKY VALUE": "WATCHLIST",
            })
        return out

    total = out["suggested_stake"].sum()

    if total > bankroll:
        factor = bankroll / total
        out["suggested_stake"] = (out["suggested_stake"] * factor).round(2)

    return out

"""
        text = text.replace("\ndef finalise_predictions(rows):", "\n" + helper2 + "\ndef finalise_predictions(rows):")

    if "cap_stakes_to_bankroll(df, load_current_bankroll(BANKROLL_START))" not in text:
        text = text.replace(
            "return df[OUTPUT_COLUMNS]",
            "df = cap_stakes_to_bankroll(df, load_current_bankroll(BANKROLL_START))\n    return df[OUTPUT_COLUMNS]"
        )

    path.write_text(text, encoding="utf-8")
    log("run_pipeline.py patché pour bankroll réelle")


def patch_dashboard_bankroll():
    path = ROOT / "dashboard.py"
    if not path.exists():
        log("dashboard.py introuvable, ignoré.")
        return

    backup(path)
    text = path.read_text(encoding="utf-8", errors="ignore")

    if 'BANKROLL_STATE_PATH = Path("data/bankroll_state.csv")' not in text:
        if 'ARCHIVE_PATH = Path("data/archive/finished_bets_archive.csv")' in text:
            text = text.replace(
                'ARCHIVE_PATH = Path("data/archive/finished_bets_archive.csv")',
                'ARCHIVE_PATH = Path("data/archive/finished_bets_archive.csv")\nBANKROLL_STATE_PATH = Path("data/bankroll_state.csv")'
            )
        else:
            text = text.replace(
                'TRACK_PATH = Path("tracking_results.csv")',
                'TRACK_PATH = Path("tracking_results.csv")\nBANKROLL_STATE_PATH = Path("data/bankroll_state.csv")'
            )

    if "bankroll_state = load_csv(BANKROLL_STATE_PATH)" not in text:
        if "archive = load_csv(ARCHIVE_PATH)" in text:
            text = text.replace(
                "archive = load_csv(ARCHIVE_PATH)",
                "archive = load_csv(ARCHIVE_PATH)\nbankroll_state = load_csv(BANKROLL_STATE_PATH)"
            )
        else:
            text = text.replace(
                "tracking = load_csv(TRACK_PATH)",
                "tracking = load_csv(TRACK_PATH)\nbankroll_state = load_csv(BANKROLL_STATE_PATH)"
            )

    if "current_bankroll_display" not in text:
        helper = """
if not bankroll_state.empty and "current_bankroll" in bankroll_state.columns:
    current_bankroll_display = float(pd.to_numeric(bankroll_state["current_bankroll"], errors="coerce").fillna(10.0).iloc[0])
    initial_bankroll_display = float(pd.to_numeric(bankroll_state.get("initial_bankroll", pd.Series([10.0])), errors="coerce").fillna(10.0).iloc[0])
else:
    current_bankroll_display = 10.0
    initial_bankroll_display = 10.0
"""
        text = text.replace("df = prepare_data(df)", "df = prepare_data(df)\n" + helper)

    text = text.replace(
        "Dashboard propre • Tennis sets probables • Score exact football • Mises conseillées • ROI",
        "Bankroll réelle 10€ • L’IA joue uniquement avec ses gains • Tennis • Football • ROI"
    )

    if 'metric("Bankroll IA"' not in text:
        target = 'c5.metric("Mise max", f"{filtered_today[\'suggested_stake\'].max() if not filtered_today.empty else 0:.2f}€")'
        repl = target + '\n\nb1, b2 = st.columns(2)\nb1.metric("Bankroll IA", f"{current_bankroll_display:.2f}€")\nb2.metric("Capital de départ", f"{initial_bankroll_display:.2f}€")'
        text = text.replace(target, repl)

    path.write_text(text, encoding="utf-8")
    log("dashboard.py patché pour afficher bankroll réelle")


def reset_stats_files():
    tracking_cols = [
        "date", "sport", "category", "home_team", "away_team", "market", "selection",
        "ai_probability", "bookmaker_odds", "value", "safety_score", "safety_level",
        "stake", "bet_mode", "stake_percent", "kelly_fraction", "bankroll",
        "confidence", "ia_badge", "result", "profit", "final_winner",
        "final_score_home", "final_score_away", "status_detail", "resolved_at",
    ]

    archive_cols = [
        "resolved_at", "date", "sport", "category", "home_team", "away_team",
        "market", "selection", "bet_mode", "ai_probability", "bookmaker_odds",
        "value", "stake", "result", "profit", "final_winner",
        "final_score_home", "final_score_away", "status_detail",
    ]

    reset_csv(ROOT / "tracking_results.csv", tracking_cols)
    reset_csv(ROOT / "data" / "archive" / "finished_bets_archive.csv", archive_cols)
    reset_csv(ROOT / "data" / "learning" / "ai_learning_profile.csv", [])
    reset_csv(ROOT / "data" / "learning" / "ai_learning_summary.csv", [])

    if (ROOT / "data" / "telegram_sent.csv").exists():
        reset_csv(ROOT / "data" / "telegram_sent.csv", [])


def main():
    log("Réinitialisation bankroll réelle 10€")

    patch_config()
    create_bankroll_state()
    create_recompute_bankroll_script()
    patch_run_pipeline_bankroll()
    patch_dashboard_bankroll()
    reset_stats_files()

    log("Terminé.")
    log("Maintenant lance :")
    log("python scripts/update_data.py")
    log("python scripts/run_pipeline.py")
    log("python scripts/save_bets_to_tracking.py")
    log("python scripts/recompute_bankroll.py")
    log("python -m streamlit run dashboard.py")


if __name__ == "__main__":
    main()
