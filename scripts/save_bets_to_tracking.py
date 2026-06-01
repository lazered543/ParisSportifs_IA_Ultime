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
    cols = ["date","sport","odds_source","category","home_team","away_team","market","selection","ai_probability","bookmaker_odds","value","safety_score","safety_level","suggested_stake","bet_mode","stake_percent","kelly_fraction","bankroll","confidence","ia_badge","learning_adjustment","calibration_adjustment","decision_reason","score_exact_1","score_exact_1_proba","score_exact_2","score_exact_2_proba","score_exact_3","score_exact_3_proba","tennis_engine_score","tennis_edge","priority"]
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
    final["suggested_stake"] = final["stake"].where(final["stake"] > 0, final["suggested_stake"])
    final["_resolved_rank"] = final.get("result", "PENDING").astype(str).str.upper().isin(["WIN","LOSS","VOID"]).astype(int)
    final["_score_keep"] = final["_resolved_rank"]*100000 + final["priority"]*10 + final["safety_score"]*6 + final["value"]*350 + final["stake"].fillna(final["suggested_stake"])*120 + final["ai_probability"]*40
    final = final.sort_values(["_match_key","_score_keep"], ascending=[True,False]).drop_duplicates("_match_key", keep="first")
    final = final.drop(columns=[c for c in final.columns if c.startswith("_")], errors="ignore")
    final.to_csv(track_path, index=False)
    print("Paris ajoutés au tracking :", len(new_bets))
    print("Fichier mis à jour : tracking_results.csv")

if __name__ == "__main__": main()
