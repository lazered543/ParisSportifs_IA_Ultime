from __future__ import annotations

import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
BACKUP_DIR = ROOT / "data" / "backups" / f"fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def log(msg: str):
    print(f"[FIX] {msg}")


def backup(path: Path):
    if path.exists():
        dest = BACKUP_DIR / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        log(f"Backup : {path} -> {dest}")


def write_text_file(path: Path, content: str):
    backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    log(f"Fichier corrigé : {path}")


def read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def safe_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"nan", "none", "nat"}:
        return ""
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    replacements = {
        "paris saint germain": "psg", "paris sg": "psg",
        "st etienne": "saint etienne", "as saint etienne": "saint etienne",
        "manchester united": "man utd", "manchester city": "man city",
        "tottenham hotspur": "tottenham", "internazionale": "inter",
        "fc internazionale milano": "inter",
    }
    for a, b in replacements.items():
        text = text.replace(a, b)
    return " ".join(text.split())


def clean_date_day(value) -> str:
    try:
        dt = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(dt):
            return str(value)[:10]
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(value)[:10]


def infer_selection(row) -> str:
    selection = safe_str(row.get("selection", ""))
    if selection:
        return selection
    market = safe_str(row.get("market", ""))
    if market in {"home win", "player 1 win"}:
        return safe_str(row.get("home_team", ""))
    if market in {"away win", "player 2 win"}:
        return safe_str(row.get("away_team", ""))
    if market == "draw":
        return "draw"
    return ""


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["date", "sport", "home_team", "away_team", "market", "selection"]:
        if col not in df.columns:
            df[col] = ""
    df["_date_day"] = df["date"].apply(clean_date_day)
    df["_selection_clean"] = df.apply(infer_selection, axis=1)
    df["_bet_key"] = (
        df["_date_day"].astype(str) + "|" + df["sport"].map(safe_str) + "|"
        + df["home_team"].map(safe_str) + "|" + df["away_team"].map(safe_str) + "|"
        + df["market"].map(safe_str) + "|" + df["_selection_clean"].astype(str)
    )
    # Une seule ligne par match dans les résultats/archives pour ne plus fausser le ROI.
    df["_match_key"] = (
        df["_date_day"].astype(str) + "|" + df["sport"].map(safe_str) + "|"
        + df["home_team"].map(safe_str) + "|" + df["away_team"].map(safe_str)
    )
    return df


def dedupe_tracking_file(path: Path):
    if not path.exists():
        log(f"Introuvable, ignoré : {path}")
        return
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        log(f"Lecture impossible {path}: {e}")
        return
    if df.empty:
        log(f"Vide : {path}")
        return
    backup(path)
    before = len(df)
    df = add_keys(df)
    for col in ["priority", "safety_score", "value", "stake", "suggested_stake", "ai_probability"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    if "result" not in df.columns:
        df["result"] = "PENDING"
    df["_resolved_rank"] = df["result"].astype(str).str.upper().isin(["WIN", "LOSS", "VOID"]).astype(int)
    df["_score_keep"] = (
        df["_resolved_rank"] * 100000 + df["priority"] * 10 + df["safety_score"] * 6
        + df["value"] * 350 + df["stake"].fillna(df["suggested_stake"]) * 120
        + df["ai_probability"] * 40
    )
    df = df.sort_values(["_bet_key", "_score_keep"], ascending=[True, False]).drop_duplicates("_bet_key", keep="first")
    df = df.sort_values(["_match_key", "_score_keep"], ascending=[True, False]).drop_duplicates("_match_key", keep="first")
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    after = len(df)
    df.to_csv(path, index=False)
    log(f"{path.name} nettoyé : {before} -> {after} lignes ({before-after} doublons supprimés)")


def build_learning_from_tracking():
    track = ROOT / "tracking_results.csv"
    out_dir = ROOT / "data" / "learning"
    profile_path = out_dir / "ai_learning_profile.csv"
    summary_path = out_dir / "ai_learning_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not track.exists():
        pd.DataFrame().to_csv(profile_path, index=False)
        return
    df = pd.read_csv(track, low_memory=False)
    if df.empty or "result" not in df.columns:
        pd.DataFrame().to_csv(profile_path, index=False)
        return
    df["result"] = df["result"].fillna("").astype(str).str.upper()
    finished = df[df["result"].isin(["WIN", "LOSS"])].copy()
    if finished.empty:
        pd.DataFrame([{"status":"NO_FINISHED_BETS","message":"Pas assez de résultats terminés pour apprendre."}]).to_csv(summary_path,index=False)
        pd.DataFrame().to_csv(profile_path,index=False)
        return
    for col in ["stake", "profit", "ai_probability", "bookmaker_odds"]:
        if col not in finished.columns:
            finished[col] = 0
        finished[col] = pd.to_numeric(finished[col], errors="coerce").fillna(0)
    if "category" not in finished.columns:
        finished["category"] = finished.get("sport", "").astype(str).str.lower().apply(lambda s: "tennis" if "tennis" in s else "football" if ("soccer" in s or "football" in s) else "autre")
    for col in ["sport", "market", "bet_mode"]:
        if col not in finished.columns:
            finished[col] = "UNKNOWN"
    rows = []
    def seg(dim, segment, g):
        stake, profit, bets = g["stake"].sum(), g["profit"].sum(), len(g)
        winrate = (g["result"] == "WIN").mean()
        roi = profit / stake if stake > 0 else 0
        rec, factor = "KEEP", 1.0
        if bets >= 5 and roi <= -0.12:
            rec, factor = "REDUCE", 0.965
        elif bets >= 5 and roi >= 0.10 and winrate >= 0.52:
            rec, factor = "BOOST", 1.015
        rows.append({"dimension":dim,"segment":segment,"bets":bets,"wins":int((g["result"]=="WIN").sum()),"losses":int((g["result"]=="LOSS").sum()),"stake":round(stake,2),"profit":round(profit,2),"roi":round(roi,4),"winrate":round(float(winrate),4),"ai_recommendation":rec,"probability_factor":factor})
    for val, g in finished.groupby("category"): seg("Sport", val, g)
    for val, g in finished.groupby("sport"): seg("Competition", val, g)
    for val, g in finished.groupby("market"): seg("Marche", val, g)
    for val, g in finished.groupby("bet_mode"): seg("Mode", val, g)
    pd.DataFrame(rows).to_csv(profile_path,index=False)
    pd.DataFrame([{"finished_bets":len(finished),"global_winrate":round(float((finished["result"]=="WIN").mean()),4),"total_stake":round(finished["stake"].sum(),2),"total_profit":round(finished["profit"].sum(),2),"global_roi":round(finished["profit"].sum()/finished["stake"].sum(),4) if finished["stake"].sum()>0 else 0,"updated_at":datetime.now(timezone.utc).isoformat()}]).to_csv(summary_path,index=False)
    log(f"Learning créé : {profile_path}")


THE_ODDS_API_CODE = r'''
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
'''


SAVE_BETS_CODE = r'''
from pathlib import Path
import re
import unicodedata
import pandas as pd

pred_path = Path("data/predictions/value_bets_today.csv")
track_path = Path("tracking_results.csv")
ALLOWED_MODES = ["SAFE PICK", "VALUE BET", "MEGA VALUE", "RISKY VALUE"]

def safe_text(value):
    if pd.isna(value): return ""
    text = str(value).strip().lower()
    if text in ["nan", "none"]: return ""
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return " ".join(text.split())

def clean_date_day(value):
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    return str(value)[:10] if pd.isna(dt) else dt.strftime("%Y-%m-%d")

def inferred_selection(row):
    selection = safe_text(row.get("selection", ""))
    if selection: return selection
    market = safe_text(row.get("market", ""))
    if market in ["home win", "player 1 win"]: return safe_text(row.get("home_team", ""))
    if market in ["away win", "player 2 win"]: return safe_text(row.get("away_team", ""))
    if market == "draw": return "draw"
    return ""

def category_from_sport(sport):
    sport = safe_text(sport)
    if "tennis" in sport: return "tennis"
    if "soccer" in sport or "football" in sport: return "football"
    return "autre"

def add_keys(df):
    df = df.copy()
    for col in ["date", "sport", "home_team", "away_team", "market", "selection"]:
        if col not in df.columns: df[col] = ""
    blank = df["selection"].fillna("").astype(str).str.strip() == ""
    df.loc[blank, "selection"] = df[blank].apply(inferred_selection, axis=1)
    if "category" not in df.columns: df["category"] = ""
    blank_cat = df["category"].fillna("").astype(str).str.strip() == ""
    df.loc[blank_cat, "category"] = df.loc[blank_cat, "sport"].apply(category_from_sport)
    df["_date_day"] = df["date"].apply(clean_date_day)
    df["_selection_clean"] = df.apply(inferred_selection, axis=1)
    df["_bet_key"] = df["_date_day"].astype(str)+"|"+df["sport"].map(safe_text)+"|"+df["home_team"].map(safe_text)+"|"+df["away_team"].map(safe_text)+"|"+df["market"].map(safe_text)+"|"+df["_selection_clean"].astype(str)
    df["_match_key"] = df["_date_day"].astype(str)+"|"+df["sport"].map(safe_text)+"|"+df["home_team"].map(safe_text)+"|"+df["away_team"].map(safe_text)
    return df

def main():
    if not pred_path.exists():
        print("Aucun fichier value_bets_today.csv trouvé."); return
    bets = pd.read_csv(pred_path, low_memory=False)
    if bets.empty:
        print("Aucun pari trouvé."); return
    if "bet_mode" in bets.columns: bets = bets[bets["bet_mode"].isin(ALLOWED_MODES)].copy()
    bets["suggested_stake"] = pd.to_numeric(bets.get("suggested_stake", 0), errors="coerce").fillna(0)
    bets["value"] = pd.to_numeric(bets.get("value", 0), errors="coerce").fillna(0)
    bets = bets[(bets["suggested_stake"] > 0) & (bets["value"] > 0)].copy()
    if bets.empty:
        print("Aucun pari recommandé après filtrage."); return
    cols = ["date","sport","category","home_team","away_team","market","selection","ai_probability","bookmaker_odds","value","safety_score","safety_level","suggested_stake","bet_mode","stake_percent","kelly_fraction","bankroll","confidence","ia_badge","score_exact_1","score_exact_1_proba","score_exact_2","score_exact_2_proba","score_exact_3","score_exact_3_proba","tennis_engine_score","tennis_edge","priority"]
    bets = bets[[c for c in cols if c in bets.columns]].copy().rename(columns={"suggested_stake":"stake"})
    bets["result"], bets["profit"] = "PENDING", 0
    for col in ["final_winner","final_score_home","final_score_away","status_detail","resolved_at"]:
        if col not in bets.columns: bets[col] = ""
    bets = add_keys(bets)
    for col in ["priority","safety_score","value","stake","ai_probability"]:
        if col not in bets.columns: bets[col] = 0
        bets[col] = pd.to_numeric(bets[col], errors="coerce").fillna(0)
    bets["_score_keep"] = bets["priority"]*10 + bets["safety_score"]*6 + bets["value"]*350 + bets["stake"]*120 + bets["ai_probability"]*40
    bets = bets.sort_values(["_match_key","_score_keep"], ascending=[True,False]).drop_duplicates("_match_key", keep="first")
    if track_path.exists():
        existing = add_keys(pd.read_csv(track_path, low_memory=False))
        new_bets = bets[~bets["_match_key"].isin(set(existing["_match_key"].dropna().astype(str)))].copy()
        final = pd.concat([existing, new_bets], ignore_index=True)
    else:
        new_bets, final = bets.copy(), bets.copy()
    final = add_keys(final)
    for col in ["priority","safety_score","value","stake","suggested_stake","ai_probability"]:
        if col not in final.columns: final[col] = 0
        final[col] = pd.to_numeric(final[col], errors="coerce").fillna(0)
    final["_resolved_rank"] = final.get("result", "PENDING").astype(str).str.upper().isin(["WIN","LOSS","VOID"]).astype(int)
    final["_score_keep"] = final["_resolved_rank"]*100000 + final["priority"]*10 + final["safety_score"]*6 + final["value"]*350 + final["stake"].fillna(final["suggested_stake"])*120 + final["ai_probability"]*40
    final = final.sort_values(["_match_key","_score_keep"], ascending=[True,False]).drop_duplicates("_match_key", keep="first")
    final = final.drop(columns=[c for c in final.columns if c.startswith("_")], errors="ignore")
    final.to_csv(track_path, index=False)
    print("Paris ajoutés au tracking :", len(new_bets))
    print("Fichier mis à jour : tracking_results.csv")

if __name__ == "__main__": main()
'''


def patch_run_pipeline():
    path = ROOT / "scripts" / "run_pipeline.py"
    if not path.exists():
        log("scripts/run_pipeline.py introuvable, patch ignoré."); return
    text = read_text_safe(path); original = text
    pattern = r"def blend_football_probabilities\(book_probs, poisson_probs, elo_home_prob, quality\):.*?\n\ndef safety_score"
    replacement = '''def blend_football_probabilities(book_probs, poisson_probs, elo_home_prob, quality):
    """
    Mélange stable bookmaker + Poisson + Elo.
    Sécurité : ne jamais s'éloigner énormément du marché.
    Exemple : Arsenal 75% contre PSG alors que le marché donne 48% => interdit.
    """
    draw_anchor = book_probs.get("draw", poisson_probs["p_draw"])
    elo_draw = clamp((poisson_probs["p_draw"] + draw_anchor) / 2, 0.16, 0.34)
    non_draw = max(1 - elo_draw, 0.01)
    elo_probs = {"home": non_draw * elo_home_prob, "away": non_draw * (1 - elo_home_prob), "draw": elo_draw}
    raw = {}
    for key, poisson_key in [("home", "p_home"), ("draw", "p_draw"), ("away", "p_away")]:
        b = book_probs.get(key, 1 / 3); p = poisson_probs.get(poisson_key, 1 / 3); e = elo_probs.get(key, 1 / 3)
        raw[key] = 0.62 * b + 0.23 * p + 0.15 * e
    total = sum(raw.values())
    if total <= 0: return {"home": 0.36, "draw": 0.28, "away": 0.36}
    raw = {key: value / total for key, value in raw.items()}
    quality = clamp(quality, 0, 1)
    max_gap = 0.08 + 0.06 * quality
    calibrated = {}
    for key in ["home", "draw", "away"]:
        book = book_probs.get(key, raw[key])
        calibrated[key] = clamp(raw[key], book - max_gap, book + max_gap)
    total = sum(calibrated.values())
    calibrated = {key: value / total for key, value in calibrated.items()}
    return {key: clamp(value, 0.03, 0.86) for key, value in calibrated.items()}


def safety_score'''
    text = re.sub(pattern, replacement, text, flags=re.S)
    if "def one_real_bet_per_match(" not in text:
        helper = '''
def _clean_day(value):
    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt): return str(value)[:10]
    return dt.strftime("%Y-%m-%d")

def one_real_bet_per_match(df):
    if df.empty: return df
    out = df.copy()
    out["_match_key"] = out["sport"].astype(str).str.lower()+"|"+out["date"].apply(_clean_day).astype(str)+"|"+out["home_team"].astype(str).str.lower()+"|"+out["away_team"].astype(str).str.lower()
    for col in ["suggested_stake", "value", "priority", "safety_score", "ai_probability"]:
        if col not in out.columns: out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["_is_real_bet"] = out["bet_mode"].isin(RECOMMENDED_MODES) & (out["suggested_stake"] > 0) & (out["value"] > 0)
    out["_keep_score"] = out["priority"]*10 + out["safety_score"]*6 + out["value"]*350 + out["suggested_stake"]*120 + out["ai_probability"]*40
    for key, g in out.groupby("_match_key"):
        real = g[g["_is_real_bet"]]
        if real.empty:
            out.loc[g.index, "suggested_stake"] = 0; continue
        best_idx = real["_keep_score"].idxmax()
        other = [idx for idx in g.index if idx != best_idx]
        out.loc[other, "suggested_stake"] = 0
        mask = out.index.isin(other) & out["bet_mode"].isin(RECOMMENDED_MODES)
        out.loc[mask, "bet_mode"] = "WATCHLIST"; out.loc[mask, "decision"] = "NO BET"
    return out.drop(columns=["_match_key", "_is_real_bet", "_keep_score"], errors="ignore")
'''
        text = text.replace("\ndef finalise_predictions(rows):", "\n" + helper + "\ndef finalise_predictions(rows):")
    text = text.replace('''    df = df.sort_values(
        ["priority", "suggested_stake", "value", "ai_probability"],
        ascending=[False, False, False, False],
    )

    return df[OUTPUT_COLUMNS]''', '''    df = one_real_bet_per_match(df)

    df = df.sort_values(
        ["priority", "suggested_stake", "value", "ai_probability"],
        ascending=[False, False, False, False],
    )

    return df[OUTPUT_COLUMNS]''')
    if text != original:
        backup(path); path.write_text(text, encoding="utf-8"); log("run_pipeline.py patché.")
    else: log("run_pipeline.py déjà patché ou structure différente.")


def patch_dashboard():
    path = ROOT / "dashboard.py"
    if not path.exists():
        log("dashboard.py introuvable, patch ignoré."); return
    text = read_text_safe(path); original = text
    pattern = r"def upcoming_only\(data, hours=72\):.*?return out\.drop\(columns=\[\"_dt\"\], errors=\"ignore\"\)"
    repl = '''def dashboard_window_hours(sport):
    sport = str(sport).lower()
    if "tennis" in sport: return 48
    if "world_cup" in sport or "international" in sport: return 240
    if "soccer" in sport or "football" in sport: return 168
    return 72

def upcoming_only(data, hours=72):
    if data.empty or "date" not in data.columns: return data
    out = data.copy(); out["_dt"] = out["date"].apply(parse_display_datetime)
    now = pd.Timestamp.utcnow()
    out["_max_hours"] = out["sport"].apply(dashboard_window_hours)
    out["_end"] = out["_max_hours"].apply(lambda h: now + pd.Timedelta(hours=int(h)))
    out = out[out["_dt"].isna() | ((out["_dt"] >= now - pd.Timedelta(hours=4)) & (out["_dt"] <= out["_end"]))].copy()
    return out.drop(columns=["_dt", "_max_hours", "_end"], errors="ignore")'''
    text = re.sub(pattern, repl, text, flags=re.S)
    if "def best_card_rows(" not in text:
        helper = '''
def ensure_match_key(data):
    if data.empty: return data
    out = data.copy()
    if "match_key" not in out.columns:
        out["match_key"] = out["sport"].astype(str).str.lower()+"|"+out["date"].astype(str).str[:10]+"|"+out["home_team"].astype(str).str.lower()+"|"+out["away_team"].astype(str).str.lower()
    return out

def best_card_rows(data):
    if data.empty: return data
    out = ensure_match_key(data).copy()
    out["_stake"] = pd.to_numeric(out.get("suggested_stake", 0), errors="coerce").fillna(0)
    out["_value"] = pd.to_numeric(out.get("value", 0), errors="coerce").fillna(-9)
    out["_safety"] = pd.to_numeric(out.get("safety_score", 0), errors="coerce").fillna(0)
    out["_prob"] = pd.to_numeric(out.get("ai_probability", 0), errors="coerce").fillna(0)
    out["_priority"] = pd.to_numeric(out.get("priority", 0), errors="coerce").fillna(0)
    out = out[out["bet_mode"].isin(RECOMMENDED_MODES) & (out["_stake"] > 0) & (out["_value"] > 0)].copy()
    if out.empty: return out.drop(columns=["_stake", "_value", "_safety", "_prob", "_priority"], errors="ignore")
    out["_card_score"] = out["_priority"]*1000 + out["_stake"]*100 + out["_value"]*100 + out["_safety"] + out["_prob"]*10
    out = out.sort_values("_card_score", ascending=False).drop_duplicates("match_key", keep="first")
    return out.drop(columns=["_stake", "_value", "_safety", "_prob", "_priority", "_card_score"], errors="ignore")
'''
        text = text.replace("\ndef render_cards(data, limit=6):", "\n" + helper + "\ndef render_cards(data, limit=6):")
    text = text.replace('''recommended["suggested_stake"] = recommended.apply(
    rebalance_stake,
    axis=1
)''', '''recommended["suggested_stake"] = recommended.apply(
    rebalance_stake,
    axis=1
)
recommended = best_card_rows(recommended)''')
    text = text.replace('''football_reco = sort_recommendations(football_df)''', '''football_reco = best_card_rows(sort_recommendations(football_df))
    football_analysis = sort_recommendations(football_df)''')
    text = text.replace("score_df = football_score_table(football_reco)", "score_df = football_score_table(football_analysis)")
    text = text.replace("show_table(football_reco, height=520)", "show_table(football_analysis, height=520)")
    text = text.replace('''tennis_reco = sort_recommendations(tennis_df)''', '''tennis_reco = best_card_rows(sort_recommendations(tennis_df))
    tennis_analysis = sort_recommendations(tennis_df)''')
    text = text.replace("sets_df = tennis_sets_table(tennis_reco)", "sets_df = tennis_sets_table(tennis_analysis)")
    text = text.replace("show_table(tennis_reco, height=520)", "show_table(tennis_analysis, height=520)")
    if text != original:
        backup(path); path.write_text(text, encoding="utf-8"); log("dashboard.py patché.")
    else: log("dashboard.py déjà patché ou structure différente.")


def main():
    log("Début correction complète du système IA Paris Sportifs")
    write_text_file(ROOT / "src" / "connectors" / "the_odds_api.py", THE_ODDS_API_CODE)
    write_text_file(ROOT / "scripts" / "save_bets_to_tracking.py", SAVE_BETS_CODE)
    patch_run_pipeline()
    patch_dashboard()
    dedupe_tracking_file(ROOT / "tracking_results.csv")
    dedupe_tracking_file(ROOT / "data" / "archive" / "finished_bets_archive.csv")
    build_learning_from_tracking()
    log("Correction terminée.")
    log("Ensuite lance : python scripts/update_data.py ; python scripts/run_pipeline.py ; python scripts/save_bets_to_tracking.py ; python scripts/update_results_auto.py ; python -m streamlit run dashboard.py")

if __name__ == "__main__":
    main()
