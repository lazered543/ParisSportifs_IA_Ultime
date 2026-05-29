from pathlib import Path
import pandas as pd

pred_path = Path("data/predictions/value_bets_today.csv")
track_path = Path("tracking_results.csv")

TRACKING_KEY_COLS = [
    "date",
    "sport",
    "home_team",
    "away_team",
    "market",
    "selection",
]

# ✅ MODES AUTORISÉS DANS LE TRACKING
ALLOWED_MODES = [
    "SAFE PICK",
    "VALUE BET",
    "MEGA VALUE",
    "RISKY VALUE",
]

if not pred_path.exists():
    print("Aucun fichier value_bets_today.csv trouvé.")
    raise SystemExit

bets = pd.read_csv(pred_path)

if bets.empty:
    print("Aucun pari trouvé.")
    raise SystemExit

# ✅ FILTRE DES MODES
if "bet_mode" in bets.columns:
    bets = bets[bets["bet_mode"].isin(ALLOWED_MODES)].copy()

if bets.empty:
    print("Aucun pari recommandé après filtrage.")
    raise SystemExit

cols = [
    "date",
    "sport",
    "category",
    "home_team",
    "away_team",
    "market",
    "selection",
    "ai_probability",
    "bookmaker_odds",
    "value",
    "safety_score",
    "safety_level",
    "suggested_stake",
    "bet_mode",
    "stake_percent",
    "kelly_fraction",
    "bankroll",
    "confidence",
    "ia_badge",
    "score_exact_1",
    "score_exact_1_proba",
    "score_exact_2",
    "score_exact_2_proba",
    "score_exact_3",
    "score_exact_3_proba",
    "tennis_engine_score",
    "tennis_edge",
    "priority",
]

bets = bets[[c for c in cols if c in bets.columns]].copy()

bets = bets.rename(columns={"suggested_stake": "stake"})

bets["result"] = "PENDING"
bets["profit"] = 0

for col in [
    "final_winner",
    "final_score_home",
    "final_score_away",
    "status_detail",
    "resolved_at",
]:
    if col not in bets.columns:
        bets[col] = ""

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()

def inferred_selection(row):
    selection = safe_text(row.get("selection", ""))

    if selection:
        return selection

    market = safe_text(row.get("market", ""))

    if market in ["home win", "player 1 win"]:
        return safe_text(row.get("home_team", ""))

    if market in ["away win", "player 2 win"]:
        return safe_text(row.get("away_team", ""))

    if market == "draw":
        return "draw"

    return ""

def add_tracking_key(df):
    df = df.copy()

    for col in TRACKING_KEY_COLS:
        if col not in df.columns:
            df[col] = ""

    key_parts = []

    for col in TRACKING_KEY_COLS:
        if col == "selection":
            key_parts.append(df.apply(inferred_selection, axis=1))
        else:
            key_parts.append(
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )

    df["_tracking_key"] = key_parts[0]

    for part in key_parts[1:]:
        df["_tracking_key"] = (
            df["_tracking_key"] + "|" + part
        )

    return df

if track_path.exists():
    existing = pd.read_csv(track_path)

    existing = add_tracking_key(existing)
    bets = add_tracking_key(bets)

    existing_keys = set(
        existing["_tracking_key"]
        .dropna()
        .astype(str)
    )

    new_bets = bets[
        ~bets["_tracking_key"].isin(existing_keys)
    ].copy()

    final = pd.concat(
        [existing, new_bets],
        ignore_index=True
    )

    final = final.drop_duplicates(
        subset=["_tracking_key"],
        keep="first"
    )

    final = final.drop(columns=["_tracking_key"])

else:
    new_bets = bets
    final = add_tracking_key(bets).drop(
        columns=["_tracking_key"]
    )

if "selection" not in final.columns:
    final["selection"] = ""

selection_blank = (
    final["selection"]
    .fillna("")
    .astype(str)
    .str.strip() == ""
)

final.loc[
    selection_blank,
    "selection"
] = final[selection_blank].apply(
    inferred_selection,
    axis=1
)

if "category" not in final.columns:
    final["category"] = ""

category_blank = (
    final["category"]
    .fillna("")
    .astype(str)
    .str.strip() == ""
)

final.loc[
    category_blank,
    "category"
] = (
    final.loc[category_blank, "sport"]
    .fillna("")
    .astype(str)
    .str.lower()
    .apply(
        lambda sport:
        "tennis"
        if "tennis" in sport
        else "football"
        if "soccer" in sport or "football" in sport
        else "autre"
    )
)

final.to_csv(track_path, index=False)

print("Paris ajoutés au tracking :", len(new_bets))
print("Fichier mis à jour : tracking_results.csv")
